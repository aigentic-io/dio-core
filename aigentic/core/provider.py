"""Provider configuration and base classes for DIO."""

import asyncio
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple


@dataclass
class Provider:
    """Provider configuration for DIO routing.

    Attributes:
        name: Unique identifier for the provider
        type: Provider type (e.g., "cloud", "local")
        cost_per_million_input_token: Cost in USD per million input tokens (e.g. 2.5 for gpt-4o at $2.50/M)
        cost_per_million_output_token: Cost in USD per million output tokens (e.g. 10.0 for gpt-4o at $10/M)
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
    cost_per_million_input_token: float = 0.0
    cost_per_million_output_token: float = 0.0
    cache_read_cost_per_million_token: Optional[float] = None
    capability: Optional[float] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.capability is None:
            self._resolve_capability()
        self._resolve_cost()

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

    def _resolve_cost(self) -> None:
        """Auto-populate costs from the dio-registry CDN when model= is set and no cost is given.

        Registry values are in $/M tokens and are assigned directly without conversion.
        Warns when the registry has not yet synced or the model is unknown, as the provider
        will be treated as free (cost_score=100) until costs are resolved.
        """
        if self.type != "cloud" or not self.model or self.cost_per_million_input_token != 0.0:
            return

        from aigentic.registry.client import get_pricing
        plan = get_pricing(self.model, modality="text")
        if plan is None:
            warnings.warn(
                f"Provider '{self.name}': cost for model '{self.model}' could not be resolved "
                f"(registry not yet synced or model unknown). Defaulting to $3.00/M input, $12.00/M output — "
                f"set cost_per_million_input_token=... explicitly to suppress this warning.",
                UserWarning,
                stacklevel=4,
            )
            self.cost_per_million_input_token = 3.0
            self.cost_per_million_output_token = 12.0
            return

        base = plan.get("base", {})
        input_price = base.get("input")
        output_price = base.get("output")
        if input_price is None or output_price is None:
            return

        self.cost_per_million_input_token = input_price
        self.cost_per_million_output_token = output_price
        cache_read = plan.get("caching", {}).get("read")
        if cache_read is not None:
            self.cache_read_cost_per_million_token = cache_read

        warnings.warn(
            f"Provider '{self.name}': costs auto-loaded from dio-registry "
            f"(${input_price:.4f}/M input, ${output_price:.4f}/M output). "
            f"Override with cost_per_million_input_token=... in Provider() for custom pricing.",
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
        return (
            input_tokens * self.cost_per_million_input_token / 1_000_000
            + output_tokens * self.cost_per_million_output_token / 1_000_000
        )

    def cost_breakdown(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
    ) -> Dict[str, float]:
        """Return itemized actual cost breakdown for given token counts.

        Keys present depend on which pricing tiers apply:
            input_usd         — always present
            output_usd        — always present
            cache_read_usd    — only when cache_read_tokens > 0 and provider has cache pricing
            total_usd         — sum of all present line items

        Args:
            input_tokens: Actual input token count from provider response
            output_tokens: Actual output token count from provider response
            cache_read_tokens: Tokens served from prompt cache (default 0)
        """
        def _r(v: float) -> float:
            return round(v, 10)

        breakdown: Dict[str, float] = {
            "input_usd": _r(input_tokens * self.cost_per_million_input_token / 1_000_000),
            "output_usd": _r(output_tokens * self.cost_per_million_output_token / 1_000_000),
        }
        if cache_read_tokens > 0 and self.cache_read_cost_per_million_token is not None:
            breakdown["cache_read_usd"] = _r(
                cache_read_tokens * self.cache_read_cost_per_million_token / 1_000_000
            )
        breakdown["total_usd"] = _r(sum(breakdown.values()))
        return breakdown

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
    def generate(self, messages: List[dict], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate a response for the given messages.

        Args:
            messages: Conversation turns in OpenAI format
                      [{"role": "user"|"system"|"assistant", "content": "..."}]
            **kwargs: Additional provider-specific parameters (temperature, max_tokens)

        Returns:
            Tuple of (content_string, usage_dict) where usage_dict contains
            {"input_tokens": int, "output_tokens": int} with exact counts
            from the provider response.
        """
        pass

    async def generate_stream(
        self, messages: List[dict], **kwargs
    ) -> AsyncIterator[str | dict]:
        """Stream a response for the given messages.

        Yields content string chunks as they arrive.  The final yield is a
        usage sentinel dict ``{"__usage__": True, "input_tokens": N,
        "output_tokens": N}`` so the caller can compute cost without a
        separate round-trip.

        The default implementation calls the synchronous ``generate()`` in a
        thread and yields the full response as a single chunk.  Adapters that
        support native upstream streaming (OpenAI, OpenRouter) override this.
        """
        content, usage = await asyncio.to_thread(self.generate, messages, **kwargs)
        yield content
        yield {
            "__usage__": True,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }


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

    def generate(self, messages: List[dict], **_kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate a mock response.

        Args:
            messages: Conversation turns (only last user message is used)
            **_kwargs: Ignored for mock provider

        Returns:
            Tuple of (mock_content, usage_dict)
        """
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if self.response_template:
            content = self.response_template.format(prompt=last_user, provider=self.provider.name)
        else:
            content = f"Mock response from {self.provider.name}: {last_user}"
        return content, {"input_tokens": 10, "output_tokens": 20}
