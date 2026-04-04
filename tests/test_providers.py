"""Unit tests for provider adapters (mocked — no live connection needed)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from aigentic.core.provider import Provider
from aigentic.providers.webhost import WebhostProvider
from aigentic.providers.openrouter import OpenRouterProvider
from aigentic.providers.openai import OpenAIProvider
from aigentic.providers.claude import ClaudeProvider


def _make_response(text: str):
    """Create a mock urllib response returning the given Ollama /api/chat response."""
    body = json.dumps({
        "message": {"content": text},
        "prompt_eval_count": 5,
        "eval_count": 10,
    }).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestWebhostProvider:
    def _provider(self, model="llama3.2:3b"):
        return Provider(name="test-ollama", type="local", model=model)

    def _msgs(self, text: str):
        return [{"role": "user", "content": text}]

    def test_calls_correct_endpoint(self):
        """Verifies the request is sent to {base_url}/api/chat."""
        provider = self._provider()
        adapter = WebhostProvider(provider, base_url="http://10.0.0.1:11434")

        with patch("urllib.request.urlopen", return_value=_make_response("hi")) as mock_open:
            adapter.generate(self._msgs("hello"))

        req = mock_open.call_args[0][0]
        assert req.full_url == "http://10.0.0.1:11434/api/chat"

    def test_returns_response_field(self):
        """Verifies the content field from Ollama /api/chat JSON is returned."""
        provider = self._provider()
        adapter = WebhostProvider(provider, base_url="http://10.0.0.1:11434")

        with patch("urllib.request.urlopen", return_value=_make_response("Hello, world!")):
            content, usage = adapter.generate(self._msgs("say hello"))

        assert content == "Hello, world!"
        assert usage == {"input_tokens": 5, "output_tokens": 10}

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
            adapter.generate(self._msgs("ping"))

        req = mock_open.call_args[0][0]
        assert req.full_url == f"{tailscale_url}/api/chat"

    def test_payload_shape(self):
        """Verifies model, messages, stream=False, and options are sent in the payload."""
        provider = self._provider(model="gpt-oss:20b")
        adapter = WebhostProvider(provider, base_url="http://10.0.0.1:11434")

        with patch("urllib.request.urlopen", return_value=_make_response("ok")) as mock_open:
            adapter.generate(self._msgs("test prompt"), temperature=0.5, max_tokens=512)

        req = mock_open.call_args[0][0]
        payload = json.loads(req.data.decode())
        assert payload["model"] == "gpt-oss:20b"
        assert payload["messages"] == [{"role": "user", "content": "test prompt"}]
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_predict"] == 512


# ── OpenRouterProvider ────────────────────────────────────────────────────────

def _make_openrouter_response(text: str, prompt_tokens: int = 8, completion_tokens: int = 15):
    """Create a mock urllib response returning an OpenRouter /v1/chat/completions response."""
    body = json.dumps({
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestOpenRouterProvider:
    def _provider(self, model="openai/gpt-4o-mini"):
        return Provider(name="openrouter", type="cloud", model=model)

    def _msgs(self, text: str):
        return [{"role": "user", "content": text}]

    def test_returns_content_and_usage(self):
        """Verifies (content, usage) tuple is returned with correct token counts."""
        adapter = OpenRouterProvider(self._provider(), api_key="test-key")

        with patch("urllib.request.urlopen", return_value=_make_openrouter_response("Hello!", 8, 15)):
            content, usage = adapter.generate(self._msgs("hi"))

        assert content == "Hello!"
        assert usage == {"input_tokens": 8, "output_tokens": 15}

    def test_payload_shape(self):
        """Verifies model, messages, temperature, and max_tokens are sent."""
        adapter = OpenRouterProvider(self._provider(model="openai/gpt-4o-mini"), api_key="test-key")

        with patch("urllib.request.urlopen", return_value=_make_openrouter_response("ok")) as mock_open:
            adapter.generate(self._msgs("test"), temperature=0.5, max_tokens=256)

        req = mock_open.call_args[0][0]
        payload = json.loads(req.data.decode())
        assert payload["model"] == "openai/gpt-4o-mini"
        assert payload["messages"] == [{"role": "user", "content": "test"}]
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 256

    def test_correct_endpoint(self):
        """Verifies request is sent to OpenRouter API URL."""
        adapter = OpenRouterProvider(self._provider(), api_key="test-key")

        with patch("urllib.request.urlopen", return_value=_make_openrouter_response("ok")) as mock_open:
            adapter.generate(self._msgs("hi"))

        req = mock_open.call_args[0][0]
        assert req.full_url == "https://openrouter.ai/api/v1/chat/completions"

    def test_missing_model_raises(self):
        """Verifies ValueError when no model is specified."""
        provider = Provider(name="openrouter", type="cloud")
        adapter = OpenRouterProvider(provider, api_key="test-key")
        with pytest.raises(ValueError, match="model must be specified"):
            adapter.generate(self._msgs("hi"))

    def test_missing_usage_defaults_to_zero(self):
        """Verifies token counts default to 0 when usage is absent from response."""
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        adapter = OpenRouterProvider(self._provider(), api_key="test-key")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            content, usage = adapter.generate(self._msgs("hi"))

        assert content == "ok"
        assert usage == {"input_tokens": 0, "output_tokens": 0}


# ── OpenAIProvider ────────────────────────────────────────────────────────────

class TestOpenAIProvider:
    def _provider(self, model="gpt-4o-mini"):
        return Provider(name="openai", type="cloud", model=model)

    def _msgs(self, text: str):
        return [{"role": "user", "content": text}]

    def _mock_response(self, content: str, prompt_tokens: int = 10, completion_tokens: int = 20):
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        message = MagicMock()
        message.content = content
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        return response

    def test_returns_content_and_usage(self):
        """Verifies (content, usage) tuple is returned with correct token counts."""
        provider = self._provider()
        adapter = OpenAIProvider(provider, api_key="test-key")
        mock_response = self._mock_response("Hello!", prompt_tokens=10, completion_tokens=20)

        with patch.object(adapter, "_client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            adapter._client = mock_client
            content, usage = adapter.generate(self._msgs("hi"))

        assert content == "Hello!"
        assert usage == {"input_tokens": 10, "output_tokens": 20}

    def test_missing_model_raises(self):
        """Verifies ValueError when no model is specified."""
        provider = Provider(name="openai", type="cloud")
        with pytest.raises(ValueError, match="OpenAI model name must be specified"):
            OpenAIProvider(provider, api_key="test-key")


# ── ClaudeProvider ────────────────────────────────────────────────────────────

class TestClaudeProvider:
    def _provider(self, model="claude-3-5-haiku-20241022"):
        return Provider(name="claude", type="cloud", model=model)

    def _msgs(self, text: str, system: str = None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": text})
        return msgs

    def _mock_response(self, content: str, input_tokens: int = 10, output_tokens: int = 20):
        usage = MagicMock()
        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens
        content_block = MagicMock()
        content_block.text = content
        response = MagicMock()
        response.content = [content_block]
        response.usage = usage
        return response

    def test_returns_content_and_usage(self):
        """Verifies (content, usage) tuple is returned with correct token counts."""
        provider = self._provider()
        adapter = ClaudeProvider(provider, api_key="test-key")
        mock_response = self._mock_response("Hello!", input_tokens=10, output_tokens=20)

        with patch.object(adapter, "_client") as mock_client:
            mock_client.messages.create.return_value = mock_response
            adapter._client = mock_client
            content, usage = adapter.generate(self._msgs("hi"))

        assert content == "Hello!"
        assert usage == {"input_tokens": 10, "output_tokens": 20}

    def test_system_message_extracted(self):
        """Verifies system message is passed as top-level system= param, not in messages."""
        provider = self._provider()
        adapter = ClaudeProvider(provider, api_key="test-key")
        mock_response = self._mock_response("ok")

        with patch.object(adapter, "_client") as mock_client:
            mock_client.messages.create.return_value = mock_response
            adapter._client = mock_client
            adapter.generate(self._msgs("hello", system="You are helpful."))
            call_kwargs = mock_client.messages.create.call_args[1]

        assert call_kwargs["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in call_kwargs["messages"])
