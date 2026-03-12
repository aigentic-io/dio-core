"""Provider configuration and base classes for DIO."""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Provider:
    """Provider configuration for DIO routing.

    Attributes:
        name: Unique identifier for the provider
        type: Provider type (e.g., "cloud", "local")
        cost_per_input_token: Cost in USD per input token (e.g. 0.0000025 for gpt-4o at $2.50/M)
        cost_per_output_token: Cost in USD per output token (e.g. 0.00001 for gpt-4o at $10/M)
        capability: Model capability level (0.0-1.0). When omitted and model= is
            set, auto-looked up from the LMSYS Arena ELO registry; falls back to 1.0
            for unknown models. Pass explicitly to override the registry value.
        model: Specific model name (e.g., "gpt-4o-mini", "claude-haiku-4-5").
               Enables model-level routing within the same provider/API key.
        latency_ms: Measured or expected inference latency in milliseconds.
                    When omitted, auto-estimated from capability for local models
                    (200–2000ms range) or falls back to type-based heuristic
                    (local=500ms, cloud=1500ms). Set explicitly for your hardware.
        metadata: Additional provider metadata
    """
    name: str
    type: str
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    capability: Optional[float] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.capability is None:
            self._resolve_capability()

    def _resolve_capability(self) -> None:
        """Auto-populate capability from the LMSYS Arena ELO registry when model= is set."""
        if not self.model:
            self.capability = 1.0
            return

        from aigentic.model_registry import get_capability, snapshot_info
        score, found = get_capability(self.model)
        self.capability = score
        if found:
            info = snapshot_info()
            warnings.warn(
                f"Provider '{self.name}': capability={score:.2f} auto-loaded from "
                f"{info['source']} (snapshot {info['snapshot_date']}). "
                f"This is a general-purpose human-preference score — override with "
                f"capability=... in Provider() for task-specific tuning. "
                f"AIgentic Premium DIO offers dynamic scoring, specialization profiles, "
                f"multi-dimension capability metrics, and custom provider scoring for advanced use cases. "
                f"https://ai-gentic.io/#contact",
                UserWarning,
                stacklevel=4,
            )

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
