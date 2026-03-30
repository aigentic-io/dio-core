"""Google Gemini provider adapter for cloud inference."""

from typing import Dict, List, Tuple

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

    def generate(self, messages: List[dict], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate a response using Gemini.

        Args:
            messages: Conversation turns in OpenAI format (system message extracted
                      as system_instruction; user/assistant roles mapped to Gemini format)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Tuple of (content, usage_dict) with exact token counts
        """
        from google.genai import types

        system_instruction = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
            else:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append(
                    types.Content(role=role, parts=[types.Part(text=m["content"])])
                )

        config_kwargs = dict(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", 1000),
        )
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        config = types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return (
            response.text,
            {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            },
        )
