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
from aigentic.server.models import InferRequest, InferResult

logger = logging.getLogger("dio.server")
router = APIRouter()


def _extract_text(messages) -> str:
    """Extract plain text from messages for routing analysis (PII, complexity, tokens).

    In v1, routing and inference both operate on text. Multimodal providers will
    receive the full messages list directly in a future update.
    """
    parts = []
    for msg in messages:
        if isinstance(msg.content, str):
            parts.append(msg.content)
        else:
            for part in msg.content:
                if part.type == "text":
                    parts.append(part.text)
    return " ".join(parts)

# ── Auth ───────────────────────────────────────────────────────────────────────
# Read at import time (after dotenv is loaded by app.py).
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
    """Extract user_id from JWT sub claim (payload decode only — no verification).

    For logging and tracing only. Production deployments should validate the
    JWT signature using JWT_SECRET or a JWKS endpoint before trusting claims.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        payload += "=" * (4 - len(payload) % 4)  # restore base64 padding
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
            "cost_per_input_token": p.cost_per_input_token,
            "cost_per_output_token": p.cost_per_output_token,
        }
        for name, p in dio.providers.items()
    ]


@router.post("/infer", response_model=InferResult)
def infer(req: InferRequest, request: Request, response: Response, _=Depends(_check_auth)):
    """Route content to the best available provider via FDE and return the result.

    ClientContext drives battery/connectivity/on-device routing decisions.
    User identity (tier, org, policies) is resolved from the Bearer token —
    never accepted from the request body.

    Explicit override fields (require_local, max_cost, max_latency_ms) bypass
    all context and server-resolved policy when set.
    """
    dio = request.app.state.dio
    t0 = time.monotonic()

    # ── Headers ───────────────────────────────────────────────────────────────
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    session_id = request.headers.get("X-Session-ID")
    accept_language = request.headers.get("Accept-Language")

    # Extract user_id from JWT sub claim (logging only — no signature verification here)
    user_id = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        user_id = _extract_user_id(auth_header[7:])

    # Echo trace ID so clients can correlate requests across retries
    response.headers["X-Request-ID"] = request_id

    # ── FDE routing ───────────────────────────────────────────────────────────
    # Build kwargs from device context; explicit request fields take precedence.
    # User-level policy (tier caps, privacy rules) is applied here by dio-platform
    # after resolving the token — in dio-core it falls through to FDE defaults.
    fde_kwargs = to_fde_kwargs(req.client_context)
    if req.require_local is not None:
        fde_kwargs["require_local"] = req.require_local
    if req.max_cost is not None:
        fde_kwargs["max_cost"] = req.max_cost
    if req.max_latency_ms is not None:
        fde_kwargs["max_latency_ms"] = req.max_latency_ms
    # Inference params — passed through to the selected provider adapter
    if req.temperature is not None:
        fde_kwargs["temperature"] = req.temperature
    if req.max_tokens is not None:
        fde_kwargs["max_tokens"] = req.max_tokens

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

    # Extract text for routing analysis (PII, complexity, token count).
    # v1: routing operates on text; multimodal providers receive full messages in v2.
    routing_text = _extract_text(req.messages)

    try:
        result = dio.route(routing_text, **fde_kwargs)
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

    # ── Telemetry: NDJSON to stdout ───────────────────────────────────────────
    has_pii = PIIDetector.has_pii(routing_text)
    complexity = FederatedDecisionEngine.analyze_complexity(routing_text)

    # Augment FDE metadata with routing context so clients see the real reason
    metadata = dict(result.metadata)
    if has_pii:
        metadata["pii_detected"] = True
        metadata["reason"] = "PII detected — routed to privacy provider (never sent to cloud)"
    elif fde_kwargs.get("require_local"):
        if req.require_local is not None:
            metadata["reason"] = "require_local (explicit override)"
        elif req.client_context:
            offline = req.client_context.connectivity == "offline"
            low_bat = (req.client_context.battery_level is not None
                       and req.client_context.battery_level < 20)
            if offline and low_bat:
                metadata["reason"] = (
                    f"Offline and low battery ({req.client_context.battery_level}%)"
                    " — local provider selected"
                )
            elif offline:
                metadata["reason"] = "Offline — no network, routed to local provider"
            elif low_bat:
                metadata["reason"] = (
                    f"Low battery ({req.client_context.battery_level}%)"
                    " — optimal for battery conservation"
                )
            else:
                metadata["reason"] = "require_local — local provider selected"
        else:
            metadata["reason"] = "require_local — local provider selected"

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
        "estimated_cost": result.metadata.get("estimated_cost", 0),
        "has_pii": has_pii,
        "complexity": complexity.value,
        "content_tokens": len(routing_text.split()),
        "wall_ms": wall_ms,
    }))

    return InferResult(
        provider=result.provider,
        model=dio.providers[result.provider].model,
        content=result.content,
        routed_by="fde",
        metadata=metadata,
    )
