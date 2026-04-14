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
    ShadowIngestRequest,
)
from aigentic.server.shadow import ShadowWriter, build_shadow_record

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


@router.post("/v1/shadow/ingest", status_code=202)
def shadow_ingest(
    req: ShadowIngestRequest,
    request: Request,
    _=Depends(_check_auth),
):
    """Passive shadow ingestion — DIO is never in the client's critical path.

    Client provides the exact ChatCompletionRequest they sent to their provider
    plus the original response. DIO runs FDE scoring using the same
    client_context, records what it would have routed to and at what cost.
    No LLM call is made. dio_response and similarity_score are filled in by
    offline batch processing.

    Returns 202 with record_id. Pass X-Request-ID to correlate with your
    own observability traces.
    """
    if not req.request.model:
        raise HTTPException(
            status_code=422,
            detail={"error": "request.model is required for shadow ingestion"},
        )

    dio = request.app.state.dio
    shadow_writer: ShadowWriter = request.app.state.shadow_writer
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    messages_dicts = _messages_to_dicts(req.request.messages)
    routing_text = _extract_text(req.request.messages)
    has_pii = PIIDetector.has_pii(routing_text)
    complexity = FederatedDecisionEngine.analyze_complexity(routing_text)

    # FDE scoring with the same client_context as the original call
    fde_kwargs = to_fde_kwargs(req.request.client_context)
    all_scores = dio.fde.score_all(dio.providers, routing_text, **fde_kwargs)
    top_score = all_scores[0] if all_scores else None

    if top_score is None:
        return {"accepted": True, "record_id": None, "detail": "no eligible provider"}

    provider = dio.providers[top_score.provider_name]
    dio_model = provider.model or top_score.provider_name
    input_t = req.original_usage.get("prompt_tokens", 0)
    output_t = req.original_usage.get("completion_tokens", 0)
    dio_cost_usd = provider.cost_breakdown(input_t, output_t)["total_usd"]

    # Original cost: client-supplied or looked up from registry
    if req.original_cost_usd is not None:
        original_cost_usd = req.original_cost_usd
    else:
        from aigentic.registry.client import get_pricing
        plan = get_pricing(req.request.model, modality="text")
        if plan:
            base = plan.get("base", {})
            original_cost_usd = round(
                input_t * base.get("input", 0) / 1_000_000
                + output_t * base.get("output", 0) / 1_000_000,
                10,
            )
        else:
            original_cost_usd = None  # model not in registry — saving fields will be null

    record = build_shadow_record(
        request_id=request_id,
        org_id=req.org_id,
        request_messages=messages_dicts,
        original_model=req.request.model,
        original_response=req.original_response,
        original_usage=req.original_usage,
        original_latency_ms=req.original_latency_ms,
        original_cost_usd=original_cost_usd,
        dio_model=dio_model,
        dio_response=None,
        dio_usage={"prompt_tokens": input_t, "completion_tokens": output_t},
        dio_latency_ms=provider.latency_ms or 0,
        dio_cost_usd=dio_cost_usd,
        fde_score=top_score.score,
        complexity=complexity.value,
        has_pii=has_pii,
    )
    shadow_writer.write(record)
    return {"accepted": True, "record_id": record["record_id"]}
