"""Federated Decision Engine (FDE) - Multi-factor routing algorithm.

The FDE is the core algorithm that routes prompts based on:
- Privacy (PII detection)
- Cost (per-token pricing)
- Complexity (prompt analysis)
- Latency requirements
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from aigentic.core.pii_detector import PIIDetector
from aigentic.core.provider import Provider


class ComplexityLevel(Enum):
    """Prompt complexity classification."""
    SIMPLE = "simple"          # <50 tokens, factual queries
    MODERATE = "moderate"      # 50-200 tokens, reasoning required
    COMPLEX = "complex"        # >200 tokens, multi-step reasoning


@dataclass
class RoutingContext:
    """Context for routing decisions.

    Attributes:
        prompt: User prompt
        has_pii: Whether prompt contains PII
        complexity: Estimated complexity level
        max_cost: Maximum acceptable cost per request
        max_latency_ms: Maximum acceptable latency
        require_local: Force local routing (data sovereignty)
    """
    prompt: str
    has_pii: bool = False
    complexity: ComplexityLevel = ComplexityLevel.SIMPLE
    max_cost: Optional[float] = None
    max_latency_ms: Optional[int] = None
    require_local: bool = False
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0


@dataclass
class RoutingScore:
    """Scoring for a provider based on routing context.

    Attributes:
        provider_name: Provider identifier
        score: Overall score (0-100, higher is better)
        privacy_score: Privacy compliance score
        cost_score: Cost efficiency score
        capability_score: Capability match score
        latency_score: Latency score
        eligible: Whether provider meets hard constraints
        reason: Explanation for the score/decision
        estimated_cost: Estimated cost for this request
    """
    provider_name: str
    score: float
    privacy_score: float
    cost_score: float
    capability_score: float
    latency_score: float
    eligible: bool = True
    reason: str = ""
    estimated_cost: float = 0.0


class FederatedDecisionEngine:
    """Core routing algorithm for DIO.

    The FDE evaluates providers based on multiple weighted factors:
    - Privacy: Hard constraint (PII must use local/compliant providers)
    - Cost: Optimization target
    - Complexity: Capability matching
    - Latency: Performance requirements
    """

    # Default weights for scoring (can be customized)
    DEFAULT_WEIGHTS = {
        "privacy": 0.40,      # 40% weight - highest priority
        "cost": 0.25,         # 25% weight
        "capability": 0.25,   # 25% weight
        "latency": 0.10,      # 10% weight
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        privacy_providers: Optional[List[str]] = None,
    ):
        """Initialize FDE.

        Args:
            weights: Custom weights for scoring factors
            privacy_providers: List of provider names approved for PII
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.privacy_providers = privacy_providers or []

    @staticmethod
    def analyze_complexity(prompt: str) -> ComplexityLevel:
        """Analyze prompt complexity.

        Args:
            prompt: User prompt

        Returns:
            Complexity level classification
        """
        token_estimate = len(prompt.split())

        # Simple heuristics (can be enhanced with ML)
        if token_estimate < 50:
            # Check for factual question indicators
            factual_keywords = ["what is", "who is", "when did", "define", "meaning"]
            if any(kw in prompt.lower() for kw in factual_keywords):
                return ComplexityLevel.SIMPLE

        if token_estimate > 200:
            return ComplexityLevel.COMPLEX

        # Check for reasoning indicators
        reasoning_keywords = ["explain", "analyze", "compare", "evaluate", "design"]
        if any(kw in prompt.lower() for kw in reasoning_keywords):
            return ComplexityLevel.COMPLEX

        return ComplexityLevel.MODERATE

    def score_provider(
        self,
        provider: Provider,
        context: RoutingContext,
    ) -> RoutingScore:
        """Score a provider for the given routing context.

        Args:
            provider: Provider to evaluate
            context: Routing context with requirements

        Returns:
            Routing score with detailed breakdown
        """
        # Privacy scoring (hard constraint)
        privacy_score = self._score_privacy(provider, context)
        if privacy_score == 0:
            return RoutingScore(
                provider_name=provider.name,
                score=0,
                privacy_score=0,
                cost_score=0,
                capability_score=0,
                latency_score=0,
                eligible=False,
                reason="Privacy constraint violated (PII detected, provider not approved)"
            )

        # Cost scoring
        cost_score = self._score_cost(provider, context)

        # Capability scoring (complexity match)
        capability_score = self._score_capability(provider, context)

        # Latency scoring
        latency_score = self._score_latency(provider, context)

        # Weighted overall score
        overall_score = (
            privacy_score * self.weights["privacy"] +
            cost_score * self.weights["cost"] +
            capability_score * self.weights["capability"] +
            latency_score * self.weights["latency"]
        )

        estimated_cost = provider.estimated_cost(
            context.estimated_input_tokens,
            context.estimated_output_tokens,
        )

        return RoutingScore(
            provider_name=provider.name,
            score=round(overall_score, 2),
            privacy_score=privacy_score,
            cost_score=cost_score,
            capability_score=capability_score,
            latency_score=latency_score,
            eligible=True,
            reason=self._build_reason(provider, context, cost_score, capability_score),
            estimated_cost=estimated_cost,
        )

    def _build_reason(
        self,
        provider: Provider,
        context: RoutingContext,
        cost_score: float,
        capability_score: float,
    ) -> str:
        """Build a human-readable routing reason for the winning provider."""
        # Privacy / local override takes top priority in the explanation
        if context.has_pii:
            return (
                f"Privacy-approved {provider.type} provider — PII detected, "
                "prompt kept off cloud"
            )
        if context.require_local:
            return "Local provider required — data sovereignty constraint"

        # Identify primary driver from weighted contributions (privacy excluded —
        # it's 100 for all eligible providers at this point).
        cost_contrib = cost_score * self.weights["cost"]
        cap_contrib = capability_score * self.weights["capability"]

        # Describe cost tier — "free" only when actual pricing is zero
        if provider.cost_per_million_input_token == 0.0 and provider.cost_per_million_output_token == 0.0:
            cost_label = f"free {provider.type}"
        elif cost_score >= 75:
            cost_label = f"low-cost {provider.type}"
        else:
            cost_label = f"{provider.type}"

        cap_pct = f"{provider.capability:.0%}"

        if context.complexity == ComplexityLevel.COMPLEX:
            cap_label = f"high capability ({cap_pct}) for complex reasoning"
        elif context.complexity == ComplexityLevel.SIMPLE:
            cap_label = f"sufficient capability ({cap_pct}) for simple queries"
        else:
            cap_label = f"capability ({cap_pct}) matched for moderate queries"

        # Lead with whichever factor contributes more to the final score
        if cap_contrib >= cost_contrib:
            return f"{cost_label.capitalize()} provider — {cap_label}"
        return f"{cost_label.capitalize()} provider — cost-efficient, {cap_label}"

    def _score_privacy(self, provider: Provider, context: RoutingContext) -> float:
        """Score privacy compliance.

        Returns 0 if provider violates privacy constraints, 100 otherwise.
        """
        # Hard constraint: PII must use approved providers
        if context.has_pii and provider.name not in self.privacy_providers:
            return 0.0

        # Force local if required
        if context.require_local and provider.type != "local":
            return 0.0

        # Prefer local for PII even if cloud is approved
        if context.has_pii and provider.type == "local":
            return 100.0

        return 100.0

    @staticmethod
    def _estimate_output_tokens(input_tokens: int, complexity: ComplexityLevel) -> int:
        """Estimate output tokens based on input tokens and complexity.

        Args:
            input_tokens: Estimated input token count
            complexity: Prompt complexity level

        Returns:
            Estimated output token count
        """
        multipliers = {
            ComplexityLevel.SIMPLE: 1,
            ComplexityLevel.MODERATE: 2,
            ComplexityLevel.COMPLEX: 3,
        }
        return input_tokens * multipliers[complexity]

    def _score_cost(self, provider: Provider, context: RoutingContext) -> float:
        """Score cost efficiency using a log scale.

        Lower cost = higher score. Each 10x increase in cost costs ~20 points.
        Reference: $0.001/request scores 100; $0.01 → 80; $0.10 → 60; $1.00 → 40.
        """
        estimated_cost = provider.estimated_cost(
            context.estimated_input_tokens,
            context.estimated_output_tokens,
        )

        # Hard constraint: cost limit
        if context.max_cost is not None and estimated_cost > context.max_cost:
            return 0.0

        # Free providers get perfect score
        if estimated_cost == 0:
            return 100.0

        # Scoring floor: costs at or below this earn 100 pts.
        # Each 10x above this costs −20 pts. Calibrated so that:
        #   free/local = 100, cheap cloud (gemini-flash) ≈ 75, frontier (gpt-4o) ≈ 46.
        reference_cost = 0.000001
        log_ratio = math.log10(estimated_cost / reference_cost)
        return max(0.0, min(100.0, 100.0 - 20.0 * log_ratio))

    def _score_capability(self, provider: Provider, context: RoutingContext) -> float:
        """Score capability match for complexity.

        Cloud models generally better for complex tasks.
        Local models sufficient for simple tasks.
        The base score is scaled by provider.capability (0.0-1.0) to
        differentiate providers of the same type.

        For SIMPLE tasks, scoring is "Goldilocks": capability peaks at ~0.5.
        Under-powered models score proportionally lower; over-powered models
        incur a small penalty (wasteful for simple queries). This prevents a
        heavyweight model from always winning simple queries purely on capability.
        """
        if context.complexity == ComplexityLevel.SIMPLE:
            # Simple queries work well on both, prefer local for cost
            base = 100.0 if provider.type == "local" else 85.0
            # Goldilocks: score peaks at capability=0.5 (sufficient for simple tasks).
            # Below 0.5: scale linearly from 0. Above 0.5: slight penalty for excess.
            if provider.capability <= 0.5:
                return base * (provider.capability / 0.5)
            overshoot = provider.capability - 0.5
            return base * max(0.6, 1.0 - overshoot * 0.8)
        elif context.complexity == ComplexityLevel.MODERATE:
            # Moderate queries work on both, slight cloud preference
            base = 90.0 if provider.type == "cloud" else 85.0
        else:  # COMPLEX
            # Complex queries benefit from larger cloud models
            base = 100.0 if provider.type == "cloud" else 70.0
        return base * provider.capability

    def _score_latency(self, provider: Provider, context: RoutingContext) -> float:
        """Score latency expectations.

        Three-level fallback for latency estimation:
          1. provider.latency_ms — explicit measured value (most accurate, hardware-specific)
          2. Capability-derived estimate for local models — model size correlates with
             inference speed: latency_ms = 200 + capability * 1800 (200ms–2000ms range)
          3. Type-based heuristic — local=500ms, cloud=1500ms
        """
        if provider.latency_ms is not None:
            estimated_latency_ms = provider.latency_ms
        elif provider.type == "local" and provider.capability is not None:
            # Larger/more-capable models are slower: 0.0→200ms, 0.5→1100ms, 1.0→2000ms
            estimated_latency_ms = 200 + int(provider.capability * 1800)
        else:
            estimated_latency_ms = 500 if provider.type == "local" else 1500

        if context.max_latency_ms is not None:
            if estimated_latency_ms > context.max_latency_ms:
                return 0.0

        # Normalize latency to 0-100 scale (assuming max acceptable of 5000ms)
        max_acceptable_latency = 5000
        latency_ratio = min(estimated_latency_ms / max_acceptable_latency, 1.0)
        return (1.0 - latency_ratio) * 100

    def _build_context(self, prompt: str, **kwargs) -> RoutingContext:
        """Build a RoutingContext from a prompt and routing kwargs."""
        has_pii = PIIDetector.has_pii(prompt)
        complexity = self.analyze_complexity(prompt)
        estimated_input_tokens = len(prompt.split())
        estimated_output_tokens = self._estimate_output_tokens(
            estimated_input_tokens, complexity
        )
        return RoutingContext(
            prompt=prompt,
            has_pii=has_pii,
            complexity=complexity,
            max_cost=kwargs.get("max_cost"),
            max_latency_ms=kwargs.get("max_latency_ms"),
            require_local=kwargs.get("require_local", False),
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
        )

    def score_all(
        self,
        providers: Dict[str, Provider],
        prompt: str,
        **kwargs
    ) -> List[RoutingScore]:
        """Score all providers and return eligible ones sorted by score (best first).

        Args:
            providers: Available providers
            prompt: User prompt
            **kwargs: Routing hints (max_cost, max_latency_ms, require_local)

        Returns:
            Eligible RoutingScores sorted descending by overall score
        """
        context = self._build_context(prompt, **kwargs)
        scores = [self.score_provider(p, context) for p in providers.values()]
        return sorted(
            (s for s in scores if s.eligible),
            key=lambda s: s.score,
            reverse=True,
        )

    def route(
        self,
        providers: Dict[str, Provider],
        prompt: str,
        **kwargs
    ) -> tuple[str, RoutingScore]:
        """Return the single best eligible provider.

        Args:
            providers: Available providers
            prompt: User prompt
            **kwargs: Additional routing parameters

        Returns:
            Tuple of (selected_provider_name, routing_score)
        """
        scores = self.score_all(providers, prompt, **kwargs)
        if not scores:
            raise ValueError("No eligible provider found for the given constraints")
        best = scores[0]
        return best.provider_name, best
