"""Routing logic for DIO provider selection."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from aigentic.core.provider import Provider


@dataclass
class Request:
    """Request object for routing decisions.

    Attributes:
        prompt: The user's prompt/query
        metadata: Additional request metadata
    """
    prompt: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Policy:
    """Policy configuration for routing decisions.

    Attributes:
        rule: Callable that takes a Request and returns a classification
        enforcement: Enforcement level ("strict" or "advisory")
        metadata: Additional policy metadata
    """
    rule: Callable[[Request], str]
    enforcement: str = "strict"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Router:
    """Routes requests to appropriate providers based on policies."""

    def __init__(self):
        """Initialize the router."""
        self.providers: Dict[str, Provider] = {}
        self.policies: List[Policy] = []
        self.classification_map: Dict[str, str] = {}

    def add_provider(self, provider: Provider):
        """Add a provider to the routing table.

        Args:
            provider: Provider configuration to add
        """
        self.providers[provider.name] = provider

    def add_policy(self, policy: Policy):
        """Add a routing policy.

        Args:
            policy: Policy configuration to add
        """
        self.policies.append(policy)

    def set_classification_mapping(self, classification: str, provider_name: str):
        """Map a classification to a provider.

        Args:
            classification: Classification string (e.g., "RESTRICTED", "PUBLIC")
            provider_name: Name of the provider to use for this classification
        """
        self.classification_map[classification] = provider_name

    def _get_provider_by_type(self, provider_type: str) -> Optional[str]:
        """Get first provider matching the given type.

        Args:
            provider_type: Provider type to match ("local", "cloud", etc.)

        Returns:
            Provider name or None if not found
        """
        for provider in self.providers.values():
            if provider.type == provider_type:
                return provider.name
        return None

    def route(self, prompt: str) -> Optional[str]:
        """Determine which provider should handle the request.

        Args:
            prompt: User's prompt text

        Returns:
            Provider name to use, or None if no suitable provider found
        """
        request = Request(prompt=prompt)

        fallback_classification = None

        # Apply policies to determine classification
        for policy in self.policies:
            classification = policy.rule(request)

            # For strict enforcement, use classification to select provider
            if policy.enforcement == "strict":
                # Check explicit mapping first
                if classification in self.classification_map:
                    return self.classification_map[classification]

                # Smart defaults for security classifications (hard constraints)
                if classification == "RESTRICTED" or classification == "PRIVATE":
                    # Restricted/private data must use local providers
                    provider = self._get_provider_by_type("local")
                    if provider:
                        return provider

                elif classification == "PUBLIC":
                    # PUBLIC is not a hard constraint — allow advisory
                    # policies to refine routing. Save as fallback.
                    fallback_classification = classification
                    continue

            # For advisory enforcement, check mappings
            if policy.enforcement == "advisory":
                if classification in self.classification_map:
                    return self.classification_map[classification]

        # Fall back to PUBLIC smart default if no advisory policy matched
        if fallback_classification == "PUBLIC":
            provider = self._get_provider_by_type("cloud")
            if provider:
                return provider

        # Default: return first available provider or None
        if self.providers:
            return next(iter(self.providers.keys()))
        return None
