"""OpenAI provider adapter for cloud inference."""

from typing import Optional

from aigentic.core.provider import Provider, ProviderAdapter


class OpenAIProvider(ProviderAdapter):
    """OpenAI adapter for cloud model inference.

    Requires: pip install openai
    """

    def __init__(
        self,
        provider: Provider,
        api_key: str,
        temperature: float = 0.7,
    ):
        """Initialize OpenAI provider.

        Args:
            provider: Provider configuration
            api_key: OpenAI API key
            temperature: Sampling temperature
        """
        super().__init__(provider)
        self.api_key = api_key
        self.model = self.provider.metadata.get("model")
        if not self.model:
            raise ValueError("OpenAI model name must be specified in provider metadata")
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not found. Install with: pip install openai"
                )
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using OpenAI.

        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated response text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", 1000),
        )
        return response.choices[0].message.content
