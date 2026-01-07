"""Anthropic Claude provider adapter for cloud inference."""

from typing import Optional

from aigentic.core.provider import Provider, ProviderAdapter


class ClaudeProvider(ProviderAdapter):
    """Anthropic Claude adapter for cloud model inference.

    Requires: pip install anthropic
    """

    def __init__(
        self,
        provider: Provider,
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """Initialize Claude provider.

        Args:
            provider: Provider configuration
            api_key: Anthropic API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(provider)
        self.api_key = api_key
        self.model = self.provider.metadata.get("model")
        if not self.model:
            raise ValueError("Claude model name must be specified in provider metadata")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic package not found. Install with: pip install anthropic"
                )
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using Claude.

        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated response text
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
