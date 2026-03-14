"""Main DIO (Distributed Intelligence Orchestration) class."""

from typing import Callable, Dict, List, Optional, Type

from aigentic.core.fde import FederatedDecisionEngine
from aigentic.core.provider import MockProvider, Provider, ProviderAdapter
from aigentic.core.response import Response
from aigentic.core.router import Policy, Router
from aigentic.registry import start as _registry_start


class DIO:
    """Main DIO orchestration class for adaptive LLM routing.

    Supports two modes:
    1. Policy mode (simple, for workshops): Rule-based routing with policies
    2. FDE mode (advanced): Multi-factor Federated Decision Engine
    """

    def __init__(
        self,
        use_fde: bool = False,
        fde_weights: Optional[Dict[str, float]] = None,
        privacy_providers: Optional[List[str]] = None,
    ):
        """Initialize DIO orchestrator.

        Args:
            use_fde: Use Federated Decision Engine (default: False for workshop compatibility)
            fde_weights: Custom weights for FDE scoring factors
            privacy_providers: Provider names approved for PII (for FDE mode)
        """
        self.use_fde = use_fde
        self.router = Router()
        self.adapters: Dict[str, ProviderAdapter] = {}
        self.fallback_config: Optional[Dict] = None
        self.providers: Dict[str, Provider] = {}

        # Start registry background sync (CDN → Redis → memory). No-op if already running.
        _registry_start()

        # Initialize FDE if enabled
        if use_fde:
            self.fde = FederatedDecisionEngine(
                weights=fde_weights,
                privacy_providers=privacy_providers or []
            )
        else:
            self.fde = None

    @property
    def fde_weights(self) -> Optional[Dict[str, float]]:
        """Get the current weights from the Federated Decision Engine."""
        if self.fde:
            return self.fde.weights
        return None

    @fde_weights.setter
    def fde_weights(self, value: Dict[str, float]):
        """Set the weights for the Federated Decision Engine."""
        if self.fde:
            self.fde.weights = value
        elif self.use_fde:
            # This case handles if FDE was intended but not yet created,
            # though current logic in __init__ should prevent this.
            self.fde = FederatedDecisionEngine(weights=value)

    def add_provider(self, provider: Provider, adapter: Optional[ProviderAdapter] = None):
        """Add a provider to the routing system.

        Args:
            provider: Provider configuration
            adapter: Optional provider adapter (uses MockProvider if not provided)
        """
        self.providers[provider.name] = provider
        self.router.add_provider(provider)

        # Use provided adapter or create a mock one
        if adapter is None:
            adapter = MockProvider(provider)

        self.adapters[provider.name] = adapter

    def add_policy(self, rule: Callable, enforcement: str = "strict", **kwargs):
        """Add a routing policy.

        Args:
            rule: Callable that takes a Request and returns a classification
            enforcement: Enforcement level ("strict" or "advisory")
            **kwargs: Additional policy metadata
        """
        policy = Policy(rule=rule, enforcement=enforcement, metadata=kwargs)
        self.router.add_policy(policy)

    def set_classification_mapping(self, classification: str, provider: Provider):
        """Map a classification to a specific provider.

        Args:
            classification: Classification string (e.g., "RESTRICTED", "PUBLIC")
            provider: Provider to use for this classification
        """
        self.router.set_classification_mapping(classification, provider.name)

    def set_fallback(
        self,
        primary: Provider,
        fallback: Provider,
        trigger: Type[Exception] = Exception
    ):
        """Configure fallback behavior for provider failures.

        Args:
            primary: Primary provider to try first
            fallback: Fallback provider to use if primary fails
            trigger: Exception type that triggers fallback
        """
        self.fallback_config = {
            "primary": primary.name,
            "fallback": fallback.name,
            "trigger": trigger,
        }

    def route(self, prompt: str, **kwargs) -> Response:
        """Route a request to the appropriate provider.

        Args:
            prompt: User's prompt/query
            **kwargs: Additional parameters to pass to the provider (and routing hints)

        Returns:
            Response object with the result
        """
        if self.use_fde:
            # FDE mode: score all eligible providers, try in order until one succeeds.
            # This provides automatic resilience — a provider that throws (auth error,
            # rate limit, timeout) is skipped and the next-best is tried instead.
            # AIgentic Premium DIO extends this with real-time health scoring so
            # degraded providers are deprioritized before they fail.
            all_scores = self.fde.score_all(self.providers, prompt, **kwargs)
            if not all_scores:
                raise ValueError("No eligible provider found for this request")

            failed: list[dict] = []
            for routing_score in all_scores:
                provider_name = routing_score.provider_name
                try:
                    content = self.adapters[provider_name].generate(prompt, **kwargs)
                    metadata = {
                        "routing_mode": "fde",
                        "score": routing_score.score,
                        "privacy_score": routing_score.privacy_score,
                        "cost_score": routing_score.cost_score,
                        "capability_score": routing_score.capability_score,
                        "latency_score": routing_score.latency_score,
                        "estimated_cost": routing_score.estimated_cost,
                        "reason": routing_score.reason,
                    }
                    if failed:
                        metadata["skipped_providers"] = [f["provider"] for f in failed]
                        metadata["fallback_reason"] = failed[0]["error"]
                    return Response(
                        content=content,
                        provider=provider_name,
                        was_fallback=bool(failed),
                        metadata=metadata,
                    )
                except Exception as e:
                    failed.append({"provider": provider_name, "error": str(e)})

            raise ValueError(
                f"All {len(failed)} eligible provider(s) failed. "
                f"Tried: {', '.join(f['provider'] for f in failed)}"
            )
        else:
            # Policy mode: route to a single provider; use set_fallback() for resilience.
            provider_name = self.router.route(prompt)
            metadata = {
                "routing_mode": "policy",
                "classification": self.router.classification,
            }

            if provider_name is None:
                raise ValueError("No suitable provider found for this request")

            try:
                content = self.adapters[provider_name].generate(prompt, **kwargs)
            except Exception as e:
                if self.fallback_config and isinstance(e, self.fallback_config["trigger"]):
                    provider_name = self.fallback_config["fallback"]
                    content = self.adapters[provider_name].generate(prompt, **kwargs)
                    metadata["fallback_reason"] = str(e)
                    return Response(
                        content=content,
                        provider=provider_name,
                        was_fallback=True,
                        metadata=metadata,
                    )
                raise

            return Response(
                content=content,
                provider=provider_name,
                was_fallback=False,
                metadata=metadata,
            )
