"""DIO REST API server — optional module (pip install aigentic[server]).

Usage:
    uvicorn aigentic.server.app:app --reload

Environment variables:
    DIO_API_KEY         Bearer token for auth (optional; open mode if unset)
    OPENAI_API_KEY      Enables OpenAI gpt-4o provider
    ANTHROPIC_API_KEY   Enables Claude claude-3-5-haiku provider
    GOOGLE_API_KEY      Enables Gemini gemini-2.5-flash-lite provider
    OLLAMA_BASE_URL     Enables remote Ollama provider
    OPENROUTER_API_KEY  Enables OpenRouter passthrough provider (shadow mode)
    LOG_LEVEL           Logging level: DEBUG, INFO, WARNING (default: INFO)
    DIO_MAX_TOKENS      Server-side max_tokens cap per request (default: 4096)

Request headers (consumed server-side, not in the request body):
    Authorization: Bearer <jwt>      — identity; JWT sub claim → user_id for logging
    Accept-Language: zh-CN,en;q=0.8  — ordered language preference (RFC 7231)
    X-Session-ID: <uuid>             — groups requests in a conversation
    X-Request-ID: <uuid>             — per-request trace ID (echoed in response)
"""

import asyncio
import json
import logging
import os
import threading
import warnings

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from fastapi import FastAPI
except ImportError as exc:
    raise ImportError(
        "FastAPI is required for the DIO server. "
        "Install it with: pip install aigentic[server]"
    ) from exc

import aigentic.registry.client as _registry_client
from aigentic.core import DIO, Provider
from aigentic.registry import sync_registry

# ── Logging ───────────────────────────────────────────────────────────────────
# Bare format — we emit NDJSON ourselves so log aggregators can parse each line.
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(message)s",
)
logger = logging.getLogger("dio.server")


# ── DIO instance ──────────────────────────────────────────────────────────────
def _build_dio() -> DIO:
    """Build DIO from environment variables.

    Always registers two mock on-device providers (gemini-nano, phi-3-mini) to
    simulate Android on-device inference — routing stubs only, actual inference
    runs on the device itself. Capabilities auto-loaded from the model registry.
    Cloud providers are only added when their API key is present.
    """
    # sync_registry must run outside any running event loop (uvicorn already has one).
    # Wait up to 30s for real pricing data; abort startup if registry fails to load.
    t = threading.Thread(target=lambda: asyncio.run(sync_registry()), daemon=True)
    t.start()
    t.join(timeout=30)
    if t.is_alive() or _registry_client._memory_cache is None:
        raise RuntimeError(
            "Registry failed to load within 30s — cannot start server without pricing data. "
            "Check CDN connectivity or set REDIS_URL to use a cached registry."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        dio = DIO(
            use_fde=True,
            fde_weights={
                "privacy":    0.40,
                "cost":       0.20,
                "capability": 0.30,
                "latency":    0.10,
            },
            privacy_providers=["gemini-nano", "phi-3-mini"],
        )

        # On-device simulation providers — always registered, no API key needed.
        # Latency values reflect simulated on-device inference (not network latency).
        dio.add_provider(Provider(
            name="gemini-nano", type="local",
            model="gemini-nano", latency_ms=50,
        ))
        dio.add_provider(Provider(
            name="phi-3-mini", type="local",
            model="phi-3-mini", latency_ms=120,
        ))

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from aigentic.providers.openai import OpenAIProvider
                p = Provider(
                    name="gpt-4o", type="cloud",
                    model="gpt-4o",
                )
                dio.add_provider(p, adapter=OpenAIProvider(p, api_key=openai_key))
            except Exception as e:
                logger.warning(json.dumps({"event": "provider_skipped", "provider": "gpt-4o", "reason": str(e)}))

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                from aigentic.providers.claude import ClaudeProvider
                p = Provider(
                    name="claude-haiku", type="cloud",
                    model="claude-3-5-haiku-20241022",
                )
                dio.add_provider(p, adapter=ClaudeProvider(p, api_key=anthropic_key))
            except Exception as e:
                logger.warning(json.dumps({"event": "provider_skipped", "provider": "claude-haiku", "reason": str(e)}))

        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            try:
                from aigentic.providers.gemini import GeminiProvider
                p = Provider(
                    name="gemini-flash", type="cloud",
                    model="gemini-2.5-flash-lite",
                )
                dio.add_provider(p, adapter=GeminiProvider(p, api_key=google_key))
            except Exception as e:
                logger.warning(json.dumps({"event": "provider_skipped", "provider": "gemini-flash", "reason": str(e)}))

        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if ollama_url:
            try:
                from aigentic.providers.webhost import WebhostProvider
                p = Provider(
                    name="ollama-local", type="local",
                    cost_per_million_input_token=0.01, cost_per_million_output_token=0.01,
                    model="llama3.1:8b", latency_ms=1000,
                )
                dio.add_provider(p, adapter=WebhostProvider(p, base_url=ollama_url))
            except Exception as e:
                logger.warning(json.dumps({"event": "provider_skipped", "provider": "ollama-local", "reason": str(e)}))

        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            try:
                from aigentic.providers.openrouter import OpenRouterProvider
                p = Provider(
                    name="openrouter", type="cloud",
                    model="openai/gpt-4o-mini",
                )
                dio.add_provider(p, adapter=OpenRouterProvider(p, api_key=openrouter_key))
            except Exception as e:
                logger.warning(json.dumps({"event": "provider_skipped", "provider": "openrouter", "reason": str(e)}))

    return dio


# ── FastAPI app ────────────────────────────────────────────────────────────────
# Import routes after dotenv is loaded so _API_KEY reads correctly.
from aigentic.server.routes import router  # noqa: E402

app = FastAPI(
    title="DIO Routing API",
    description=(
        "Federated Decision Engine for adaptive LLM routing. "
        "Routes content to the optimal provider based on privacy, cost, "
        "capability, and latency. User identity is resolved from the Bearer token."
    ),
    version="0.1.0",
)
app.state.dio = _build_dio()
app.include_router(router)
