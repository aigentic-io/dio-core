"""Unit tests for WebhostProvider (mocked HTTP — no live connection needed)."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from aigentic.core.provider import Provider
from aigentic.providers.webhost import WebhostProvider


def _make_response(text: str):
    """Create a mock urllib response returning the given Ollama response text."""
    body = json.dumps({"response": text}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestWebhostProvider:
    def _provider(self, model="llama3.2:3b"):
        return Provider(name="test-ollama", type="local", model=model)

    def test_calls_correct_endpoint(self):
        """Verifies the request is sent to {base_url}/api/generate."""
        provider = self._provider()
        adapter = WebhostProvider(provider, base_url="http://10.0.0.1:11434")

        with patch("urllib.request.urlopen", return_value=_make_response("hi")) as mock_open:
            adapter.generate("hello")

        req = mock_open.call_args[0][0]
        assert req.full_url == "http://10.0.0.1:11434/api/generate"

    def test_returns_response_field(self):
        """Verifies the 'response' field from Ollama JSON is returned."""
        provider = self._provider()
        adapter = WebhostProvider(provider, base_url="http://10.0.0.1:11434")

        with patch("urllib.request.urlopen", return_value=_make_response("Hello, world!")):
            result = adapter.generate("say hello")

        assert result == "Hello, world!"

    def test_missing_model_raises(self):
        """Verifies ValueError when no model is specified."""
        provider = Provider(name="no-model", type="local")  # model=None
        with pytest.raises(ValueError, match="Model name must be specified"):
            WebhostProvider(provider, base_url="http://10.0.0.1:11434")

    def test_custom_base_url(self):
        """Verifies a non-default base URL (e.g. Tailscale IP) is used in the request."""
        tailscale_url = "http://100.94.44.55:11434"
        provider = self._provider()
        adapter = WebhostProvider(provider, base_url=tailscale_url)

        with patch("urllib.request.urlopen", return_value=_make_response("ok")) as mock_open:
            adapter.generate("ping")

        req = mock_open.call_args[0][0]
        assert req.full_url == f"{tailscale_url}/api/generate"

    def test_payload_shape(self):
        """Verifies model, prompt, stream=False, and options are sent in the payload."""
        provider = self._provider(model="gpt-oss:20b")
        adapter = WebhostProvider(provider, base_url="http://10.0.0.1:11434")

        with patch("urllib.request.urlopen", return_value=_make_response("ok")) as mock_open:
            adapter.generate("test prompt", temperature=0.5, max_tokens=512)

        req = mock_open.call_args[0][0]
        payload = json.loads(req.data.decode())
        assert payload["model"] == "gpt-oss:20b"
        assert payload["prompt"] == "test prompt"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_predict"] == 512
