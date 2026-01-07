"""Google Gemini provider adapter for cloud inference."""

from typing import Optional

from aigentic.core.provider import Provider, ProviderAdapter


class GeminiProvider(ProviderAdapter):
    """Google Gemini adapter for cloud model inference.

    Requires: pip install google-generativeai
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
        # Get model from provider metadata, not a hardcoded default
        self.model_name = self.provider.metadata.get("model")
        if not self.model_name:
            raise ValueError("Gemini model name must be specified in provider metadata")
        self.temperature = temperature
        self._model = None

    @property
    def model(self):
        """Lazy load Gemini model using the new 'google.genai' library."""
        if self._model is None:
            try:
                # Use the new, recommended 'google.genai' library
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError(
                    "Google Generative AI package not found. "
                    "Install with: pip install google-generativeai"
                )
        return self._model

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using Gemini.

        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated response text
        """
        generation_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_output_tokens": kwargs.get("max_tokens", 1000),
        }

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text