"""Provider adapters for different LLM backends."""

from aigentic.providers.mock import MockProvider
from aigentic.providers.ollama import OllamaProvider

# Optional cloud providers (require additional dependencies)
try:
    from aigentic.providers.openai import OpenAIProvider
except ImportError:
    OpenAIProvider = None

try:
    from aigentic.providers.gemini import GeminiProvider
except ImportError:
    GeminiProvider = None

try:
    from aigentic.providers.claude import ClaudeProvider
except ImportError:
    ClaudeProvider = None

__all__ = ["MockProvider", "OllamaProvider", "OpenAIProvider", "GeminiProvider", "ClaudeProvider"]
