"""Ollama provider adapter for local LLM inference."""

from typing import Dict, List, Tuple

from aigentic.core.provider import Provider, ProviderAdapter


class OllamaProvider(ProviderAdapter):
    """Ollama adapter for local model inference.

    Connects to a local Ollama instance for privacy-preserving inference.
    """

    def __init__(
        self,
        provider: Provider,
        base_url: str = "http://localhost:11434",
    ):
        """Initialize Ollama provider.

        Args:
            provider: Provider configuration
            base_url: Ollama API base URL
        """
        super().__init__(provider)
        self.model = self.provider.model or self.provider.metadata.get("model")
        if not self.model:
            raise ValueError(
                "Ollama model name must be specified via Provider(model=...) "
                "or provider metadata"
            )
        self.base_url = base_url

    def generate(self, messages: List[dict], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate a response using Ollama.

        Args:
            messages: Conversation turns in OpenAI format
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Tuple of (content, usage_dict)

        Note:
            Stub implementation — real calls would use the Ollama HTTP API or SDK.
        """
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return (
            f"[Ollama {self.model} response to: {last_user[:50]}...]",
            {"input_tokens": 0, "output_tokens": 0},
        )
