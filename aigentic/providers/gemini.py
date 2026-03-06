"""Google Gemini provider adapter for cloud inference."""

from aigentic.core.provider import Provider, ProviderAdapter


class GeminiProvider(ProviderAdapter):
    """Google Gemini adapter for cloud model inference.

    Requires: pip install google-genai
    """

    def __init__(
        self,
        provider: Provider,
        api_key: str,
        temperature: float = 0.7,
    ):
        """Initialize Gemini provider.

        Args:
            provider: Provider configuration
            api_key: Google API key
            temperature: Sampling temperature
        """
        super().__init__(provider)
        self.api_key = api_key
        self.model_name = self.provider.model or self.provider.metadata.get("model")
        if not self.model_name:
            raise ValueError(
                "Gemini model name must be specified via Provider(model=...) "
                "or provider metadata"
            )
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """Lazy load Gemini client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Google GenAI package not found. "
                    "Install with: pip install google-genai"
                )
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using Gemini.

        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated response text
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", 1000),
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return response.text
