"""DIO API endpoint handlers."""

import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from aigentic.core.fde import FederatedDecisionEngine
from aigentic.core.pii_detector import PIIDetector
from aigentic.server.device_context import to_fde_kwargs
from aigentic.server.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    Message,
)

logger = logging.getLogger("dio.server")
_SERVER_MAX_TOKENS = int(os.getenv("DIO_MAX_TOKENS", "4096"))
router = APIRouter()


def _extract_text(messages) -> str:
    """Extract plain text from messages for routing analysis (PII, complexity, tokens)."""
    parts = []
    for msg in messages:
        if isinstance(msg.content, str):
            parts.append(msg.content)
        else:
            for part in msg.content:
                if part.type == "text":
                    parts.append(part.text)
    return " ".join(parts)


def _messages_to_dicts(messages) -> list:
    """Convert Pydantic Message models to plain dicts for provider adapters."""
    result = []
    for msg in messages:
        if isinstance(msg.content, str):
            result.append({"role": msg.role, "content": msg.content})
        else:
            text = " ".join(p.text for p in msg.content if p.type == "text")
            result.append({"role": msg.role, "content": text})
    return result


# ── Auth ───────────────────────────────────────────────────────────────────────
_API_KEY = os.getenv("DIO_API_KEY")
_bearer = HTTPBearer(auto_error=False)


def _check_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
):
    """Validate Bearer token. No-op when DIO_API_KEY is not set (dev/demo mode)."""
    if not _API_KEY:
        return  # open mode
    if credentials is None or credentials.credentials != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── JWT helpers ────────────────────────────────────────────────────────────────
def _extract_user_id(token: str) -> Optional[str]:
    """Extract user_id from JWT sub claim (payload decode only — no verification)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        payload += "=" * (4 - len(payload) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(payload))
        return decoded.get("sub")
    except Exception:
        return None


# ── Endpoints ──────────────────────────────────────────────────────────────────
@router.get("/health")
def health(request: Request):
    """Health check — lists configured providers. No auth required."""
    dio = request.app.state.dio
    return {
        "status": "ok",
        "fde": True,
        "auth": bool(_API_KEY),
        "providers": [
            {
                "name": name,
                "type": p.type,
                "model": p.model,
                "capability": p.capability,
            }
            for name, p in dio.providers.items()
        ],
    }


@router.get("/providers")
def list_providers(request: Request, _=Depends(_check_auth)):
    """List configured providers with their routing parameters."""
    dio = request.app.state.dio
    return [
        {
            "name": name,
            "type": p.type,
            "model": p.model,
            "capability": p.capability,
            "latency_ms": p.latency_ms,
            "cost_per_million_input_token": p.cost_per_million_input_token,
            "cost_per_million_output_token": p.cost_per_million_output_token,
        }
        for name, p in dio.providers.items()
    ]


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    response: Response,
    _=Depends(_check_auth),
):
    """Route a chat completion to the best available provider via FDE.

    OpenAI / OpenRouter compatible endpoint — clients swap the base URL only.

    ClientContext drives battery/connectivity/on-device routing decisions.
    User identity is resolved from the Bearer token, never from the request body.

    The x_dio extension field carries routing metadata (provider, score, reason,
    estimated cost). Standard OR clients that ignore x_dio are unaffected.
    """
    dio = request.app.state.dio
    t0 = time.monotonic()

    # ── Headers ───────────────────────────────────────────────────────────────
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    session_id = request.headers.get("X-Session-ID")
    accept_language = request.headers.get("Accept-Language")

    user_id = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        user_id = _extract_user_id(auth_header[7:])

    response.headers["X-Request-ID"] = request_id

    # ── FDE routing kwargs ────────────────────────────────────────────────────
    fde_kwargs = to_fde_kwargs(req.client_context)
    if req.require_local is not None:
        fde_kwargs["require_local"] = req.require_local
    if req.max_cost is not None:
        fde_kwargs["max_cost"] = req.max_cost
    if req.max_latency_ms is not None:
        fde_kwargs["max_latency_ms"] = req.max_latency_ms
    if req.temperature is not None:
        fde_kwargs["temperature"] = req.temperature
    if req.max_tokens is not None:
        if req.max_tokens > _SERVER_MAX_TOKENS:
            raise HTTPException(
                status_code=422,
                detail={"error": f"max_tokens exceeds server limit of {_SERVER_MAX_TOKENS}"},
            )
        fde_kwargs["max_tokens"] = req.max_tokens
    else:
        fde_kwargs["max_tokens"] = _SERVER_MAX_TOKENS

    # Reject multimodal content — not yet supported in dio-core.
    if any(isinstance(msg.content, list) for msg in req.messages):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Multimodal routing (images/audio) is not yet supported in dio-core. "
                         "Implement a custom multimodal adapter or contact us for DIO Premium. "
                         "https://ai-gentic.io/#contact"
            },
        )

    routing_text = _extract_text(req.messages)
    messages_dicts = _messages_to_dicts(req.messages)

    try:
        result = dio.route(routing_text, messages=messages_dicts, **fde_kwargs)
    except Exception as exc:
        wall_ms = int((time.monotonic() - t0) * 1000)
        logger.error(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "session_id": session_id,
            "user_id": user_id,
            "error": str(exc),
            "fde_kwargs": {k: v for k, v in fde_kwargs.items()},
            "wall_ms": wall_ms,
        }))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "No provider could fulfil the request", "reason": str(exc)},
        ) from exc
    wall_ms = int((time.monotonic() - t0) * 1000)

    # ── Build x_dio routing metadata ─────────────────────────────────────────
    has_pii = PIIDetector.has_pii(routing_text)
    complexity = FederatedDecisionEngine.analyze_complexity(routing_text)

    x_dio: dict = {
        "provider": result.provider,
        "routed_by": "fde",
        "score": result.metadata.get("score"),
        "reason": result.metadata.get("reason"),
        "routing_mode": result.metadata.get("routing_mode"),
    }
    if result.was_fallback:
        x_dio["skipped_providers"] = result.metadata.get("skipped_providers", [])
        x_dio["fallback_reason"] = result.metadata.get("fallback_reason")

    # Augment reason with context so clients see the real routing driver
    if has_pii:
        x_dio["pii_detected"] = True
        x_dio["reason"] = "PII detected — routed to privacy provider (never sent to cloud)"
    elif fde_kwargs.get("require_local"):
        if req.require_local is not None:
            x_dio["reason"] = "require_local (explicit override)"
        elif req.client_context:
            offline = req.client_context.connectivity == "offline"
            low_bat = (req.client_context.battery_level is not None
                       and req.client_context.battery_level < 20)
            if offline and low_bat:
                x_dio["reason"] = (
                    f"Offline and low battery ({req.client_context.battery_level}%)"
                    " — local provider selected"
                )
            elif offline:
                x_dio["reason"] = "Offline — no network, routed to local provider"
            elif low_bat:
                x_dio["reason"] = (
                    f"Low battery ({req.client_context.battery_level}%)"
                    " — optimal for battery conservation"
                )
            else:
                x_dio["reason"] = "require_local — local provider selected"
        else:
            x_dio["reason"] = "require_local — local provider selected"

    # ── Actual cost breakdown from provider response tokens ───────────────────
    usage = result.usage or {"input_tokens": 0, "output_tokens": 0}
    x_dio["cost"] = dio.providers[result.provider].cost_breakdown(
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        cache_read_tokens=usage.get("cache_read_tokens", 0),
    )

    # ── Telemetry: NDJSON to stdout ───────────────────────────────────────────
    logger.info(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "session_id": session_id,
        "user_id": user_id,
        "accept_language": accept_language,
        "platform": req.client_context.platform if req.client_context else None,
        "provider": result.provider,
        "model": dio.providers[result.provider].model,
        "scores": {
            k: result.metadata.get(f"{k}_score")
            for k in ("privacy", "cost", "capability", "latency")
        },
        "has_pii": has_pii,
        "complexity": complexity.value,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cost_usd": x_dio["cost"]["total_usd"],
        "wall_ms": wall_ms,
    }))

    return ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        created=int(time.time()),
        model=dio.providers[result.provider].model or result.provider,
        choices=[
            ChatCompletionChoice(
                message=Message(role="assistant", content=result.content),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=usage["input_tokens"],
            completion_tokens=usage["output_tokens"],
            total_tokens=usage["input_tokens"] + usage["output_tokens"],
        ),
        x_dio=x_dio,
    )
