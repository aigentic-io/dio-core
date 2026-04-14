"""Anthropic Claude provider adapter for cloud inference."""

from typing import Dict, List, Tuple

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
        self.model = self.provider.model or self.provider.metadata.get("model")
        if not self.model:
            raise ValueError(
                "Claude model name must be specified via Provider(model=...) "
                "or provider metadata"
            )
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

    def generate(self, messages: List[dict], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate a response using Claude.

        Args:
            messages: Conversation turns in OpenAI format (system message extracted
                      and passed as top-level system= parameter per Anthropic API)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Tuple of (content, usage_dict) with exact token counts
        """
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)

        create_kwargs = dict(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=chat_messages,
        )
        if system:
            create_kwargs["system"] = system

        response = self.client.messages.create(**create_kwargs)
        return (
            response.content[0].text,
            {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
