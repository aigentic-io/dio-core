"""WebhostProvider — HTTP adapter for self-hosted Ollama instances."""

import json
import urllib.request
from typing import Dict, List, Tuple

from aigentic.core.provider import Provider, ProviderAdapter


class WebhostProvider(ProviderAdapter):
    """Adapter for self-hosted Ollama instances accessible via HTTP.

    Use this when your Ollama server runs on a remote machine (homelab,
    Tailscale network, cloud VM, etc.) rather than localhost.

    Requires: no additional dependencies (uses Python stdlib urllib)

    Example:
        provider = Provider(name="my-llama", type="local", model="llama3.2:3b")
        adapter = WebhostProvider(provider, base_url="http://192.168.1.100:11434")
        dio.add_provider(provider, adapter=adapter)
    """

    def __init__(
        self,
        provider: Provider,
        base_url: str,
        timeout: int = 500,
    ):
        """Initialize WebhostProvider.

        Args:
            provider: Provider configuration (must include model)
            base_url: Base URL of the Ollama server (e.g. "http://base_url:11434")
            timeout: Request timeout in seconds (default 500; large models may need more)
        """
        super().__init__(provider)
        self.model = self.provider.model or self.provider.metadata.get("model")
        if not self.model:
            raise ValueError(
                "Model name must be specified via Provider(model=...) or provider metadata"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, messages: List[dict], **kwargs) -> Tuple[str, Dict[str, int]]:
        """Generate a response via the Ollama /api/chat endpoint.

        Args:
            messages: Conversation turns in OpenAI format
            **kwargs: temperature (float), max_tokens (int)

        Returns:
            Tuple of (content, usage_dict) with token counts from Ollama response
        """
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1000),
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
            return (
                data["message"]["content"],
                {
                    # prompt_eval_count is omitted by Ollama on cached/continued sessions;
                    # defaults to 0 in that case — cost tracking may undercount for long sessions.
                    "input_tokens": data.get("prompt_eval_count", 0),
                    "output_tokens": data.get("eval_count", 0),
                },
            )
