"""OpenAI provider adapter for cloud inference."""

from typing import AsyncIterator, Dict, List, Tuple

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
        self.model = self.provider.model or self.provider.metadata.get("model")
        if not self.model:
            raise ValueError(
                "OpenAI model name must be specified via Provider(model=...) "
                "or provider metadata"
            )
        self.temperature = temperature
        self._client = None
        self._async_client = None

    @property
    def client(self):
        """Lazy load OpenAI sync client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not found. Install with: pip install openai"
                )
        return self._client

    @property
    def async_client(self):
        """Lazy load OpenAI async client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
                self._async_client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "OpenAI package not found. Install with: pip install openai"
                )
        return self._async_client

    def generate(self, messages: List[dict], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate a response using OpenAI.

        Args:
            messages: Conversation turns in OpenAI format
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Tuple of (content, usage_dict) with exact token counts
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", 1000),
        )
        return (
            response.choices[0].message.content,
            {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        )

    async def generate_stream(
        self, messages: List[dict], **kwargs
    ) -> AsyncIterator[str | dict]:
        """Stream via OpenAI with stream=True.  Yields content chunks, then a
        usage sentinel so the caller can compute cost after the stream ends.
        """
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", 1000),
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            # Final chunk: choices=[], usage populated (stream_options above)
            if chunk.usage:
                yield {
                    "__usage__": True,
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                }
