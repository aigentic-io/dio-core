"""Ollama provider adapter for local LLM inference."""

from typing import Optional

from aigentic.core.provider import Provider, ProviderAdapter


class OllamaProvider(ProviderAdapter):
    """Ollama adapter for local model inference.

    Connects to a local Ollama instance for privacy-preserving inference.
    """

    def __init__(
        self,
        provider: Provider,
        base_url: str = "http://localhost:11434",
    ):
        """Initialize Ollama provider.

        Args:
            provider: Provider configuration
            base_url: Ollama API base URL
        """
        super().__init__(provider)
        self.model = self.provider.metadata.get("model")
        if not self.model:
            raise ValueError("Ollama model name must be specified in provider metadata")
        self.base_url = base_url

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using Ollama.

        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Generated response text

        Note:
            This is a stub implementation for the workshop.
            Actual implementation would use the Ollama Python SDK or HTTP API.
        """
        # Stub implementation for workshop - would normally call Ollama API
        # Example:
        # import requests
        # response = requests.post(
        #     f"{self.base_url}/api/generate",
        #     json={"model": self.model, "prompt": prompt, **kwargs}
        # )
        # return response.json()["response"]

        return f"[Ollama {self.model} response to: {prompt[:50]}...]"
