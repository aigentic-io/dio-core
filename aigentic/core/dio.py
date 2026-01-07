"""Main DIO (Distributed Intelligence Orchestration) class."""

from typing import Callable, Dict, List, Optional, Type

from aigentic.core.fde import FederatedDecisionEngine
from aigentic.core.provider import Provider, ProviderAdapter, MockProvider
from aigentic.core.response import Response
from aigentic.core.router import Policy, Router


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
        # Determine which provider to use
        if self.use_fde:
            # Use Federated Decision Engine
            provider_name, routing_score = self.fde.route(
                self.providers,
                prompt,
                **kwargs
            )
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
        else:
            # Use simple policy-based router
            provider_name = self.router.route(prompt)
            metadata = {"routing_mode": "policy", "classification": "routed"}

        if provider_name is None:
            raise ValueError("No suitable provider found for this request")

        # Try to get response from the selected provider
        was_fallback = False

        try:
            adapter = self.adapters[provider_name]
            content = adapter.generate(prompt, **kwargs)
        except Exception as e:
            # Check if fallback is configured and this exception triggers it
            if self.fallback_config and isinstance(e, self.fallback_config["trigger"]):
                provider_name = self.fallback_config["fallback"]
                adapter = self.adapters[provider_name]
                content = adapter.generate(prompt, **kwargs)
                was_fallback = True
                metadata["fallback_reason"] = str(e)
            else:
                raise

        return Response(
            content=content,
            provider=provider_name,
            was_fallback=was_fallback,
            metadata=metadata,
        )
