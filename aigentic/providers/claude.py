"""Anthropic Claude provider adapter for cloud inference."""

from typing import AsyncIterator, Dict, List, Tuple

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
        self._async_client = None

    @property
    def client(self):
        """Lazy load Anthropic sync client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic package not found. Install with: pip install anthropic"
                )
        return self._client

    @property
    def async_client(self):
        """Lazy load Anthropic async client."""
        if self._async_client is None:
            try:
                from anthropic import AsyncAnthropic
                self._async_client = AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic package not found. Install with: pip install anthropic"
                )
        return self._async_client

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

    async def generate_stream(self, messages: List[dict], **kwargs) -> AsyncIterator:
        """Stream a response using Claude's native streaming API."""
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

        async with self.async_client.messages.stream(**create_kwargs) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
            yield {
                "__usage__": True,
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
            }
