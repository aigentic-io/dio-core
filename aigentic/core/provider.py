"""Provider configuration and base classes for DIO."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Provider:
    """Provider configuration for DIO routing.

    Attributes:
        name: Unique identifier for the provider
        type: Provider type (e.g., "cloud", "local")
        cost_per_input_token: Cost per input token (for routing decisions)
        cost_per_output_token: Cost per output token (for routing decisions)
        capability: Model capability level (0.0-1.0, default 1.0). Used by FDE to
            differentiate providers of the same type (e.g., GPT-4o-mini vs Gemini Flash).
        metadata: Additional provider metadata
    """
    name: str
    type: str
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    capability: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def estimated_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost for a request with the given token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost
        """
        return (input_tokens * self.cost_per_input_token +
                output_tokens * self.cost_per_output_token)

    @property
    def cost(self) -> float:
        """Backward-compatible cost estimate using default token counts (500 input + 500 output)."""
        return self.estimated_cost(500, 500)


class ProviderAdapter(ABC):
    """Abstract base class for provider implementations."""

    def __init__(self, provider: Provider):
        """Initialize the adapter with a provider configuration.

        Args:
            provider: Provider configuration object
        """
        self.provider = provider

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: Input prompt text
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated response text
        """
        pass


class MockProvider(ProviderAdapter):
    """Mock provider for testing and workshops.

    Returns a simple echo response with provider metadata.
    """

    def __init__(self, provider: Provider, response_template: Optional[str] = None):
        """Initialize mock provider.

        Args:
            provider: Provider configuration
            response_template: Optional template for responses (default: echo prompt)
        """
        super().__init__(provider)
        self.response_template = response_template

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response.

        Args:
            prompt: Input prompt text
            **kwargs: Ignored for mock provider

        Returns:
            Mock response based on template or echo
        """
        if self.response_template:
            return self.response_template.format(prompt=prompt, provider=self.provider.name)
        return f"Mock response from {self.provider.name}: {prompt}"
