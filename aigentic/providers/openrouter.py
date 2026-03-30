"""OpenRouter passthrough adapter — forwards to openrouter.ai with exact token counts."""

import json
import urllib.request
from typing import Dict, List, Tuple

from aigentic.core.provider import Provider, ProviderAdapter


class OpenRouterProvider(ProviderAdapter):
    """Forwards requests to OpenRouter's /v1/chat/completions API.

    Used as the 'original model' adapter in shadow mode so DIO can replay
    the exact same request to OpenRouter and compare against its own routing.

    Supports all models available on OpenRouter (pass model name via
    Provider(model=...) or the model kwarg at call time).

    Requires: OPENROUTER_API_KEY environment variable (or pass api_key directly)

    Example:
        p = Provider(name="openrouter", type="cloud", model="openai/gpt-4o-mini")
        adapter = OpenRouterProvider(p, api_key=os.getenv("OPENROUTER_API_KEY"))
        dio.add_provider(p, adapter=adapter)
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, provider: Provider, api_key: str, timeout: int = 60):
        """Initialize OpenRouterProvider.

        Args:
            provider: Provider configuration (model sets the default OR model)
            api_key: OpenRouter API key
            timeout: Request timeout in seconds
        """
        super().__init__(provider)
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, messages: List[dict], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Forward a chat completion request to OpenRouter.

        Args:
            messages: Conversation turns in OpenAI format
            **kwargs: temperature, max_tokens, model (overrides provider.model)

        Returns:
            Tuple of (content, usage_dict) with exact token counts from OR response
        """
        model = kwargs.get("model") or self.provider.model
        if not model:
            raise ValueError(
                "model must be specified via Provider(model=...) or kwargs['model']"
            )

        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }).encode()

        req = urllib.request.Request(
            self.BASE_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://ai-gentic.io",
                "X-Title": "DIO",
            },
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return (
            content,
            {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        )
