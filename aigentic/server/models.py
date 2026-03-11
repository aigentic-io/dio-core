"""Pydantic models for the DIO REST API."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


# ── Multimodal content parts (OpenAI / OpenRouter / LiteLLM compatible) ───────

class ImageUrl(BaseModel):
    url: str  # HTTPS URL or data:image/...;base64,...
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = Union[TextPart, ImagePart]


class Message(BaseModel):
    """A single turn in a conversation.

    Follows the OpenAI / OpenRouter / LiteLLM message format so clients can
    reuse the same message construction code across providers.

    content can be:
      - str: plain text (most common)
      - List[ContentPart]: multimodal (text + images in v1; audio/video in v2)
    """
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentPart]]


class ClientContext(BaseModel):
    """Runtime context of the client making the request.

    Describes the device/environment state that should influence routing —
    things that change per request (battery drains, network switches). Works
    across Android, iOS, web, and desktop clients.

    Attributes:
        platform: Client platform type.
        connectivity: Active network type. "offline" forces local-only routing.
        battery_level: Battery percentage 0-100. Below 20 forces local routing
            to avoid HTTP drain. None = plugged in / not applicable (web/desktop).
        on_device_model: Name of the on-device model available on this device,
            if any (e.g. "gemini-nano" on Pixel 9, "phi-3-mini" on mid-range).
            Used by future routing logic to prefer a specific local provider.
        memory_mb: Available RAM in MB. Hints whether on-device inference is
            feasible. Future: skip local routing if memory_mb < model threshold.
    """
    platform: Optional[Literal["android", "ios", "web", "desktop", "server"]] = None
    connectivity: Optional[Literal["wifi", "cellular", "ethernet", "offline"]] = None
    battery_level: Optional[int] = None
    on_device_model: Optional[str] = None
    memory_mb: Optional[int] = None


class InferRequest(BaseModel):
    """Request to route content through DIO and get an inference result.

    User identity is resolved server-side from the Authorization Bearer token
    (JWT sub → user_id, org_id, tier, policies). Clients must not self-report
    identity — the server derives everything from the token.

    Headers consumed by the server (not part of this model):
        Authorization: Bearer <jwt>      — identity, tier, org, routing policies
        Accept-Language: zh-CN,en;q=0.8  — ordered language preference list
        X-Session-ID: <uuid>             — groups requests in a conversation
        X-Request-ID: <uuid>             — per-request trace ID (echoed in response)

    The `content` field is text-only in v1. In v2 it will become
    Union[str, list[ContentPart]] to support multimodal inputs (images, audio).
    Keeping the field named `content` now avoids a breaking rename later.

    Follows the OpenAI / OpenRouter / LiteLLM messages format so clients can
    integrate with minimal changes. DIO-specific fields are additive.

    Attributes:
        messages: Conversation turns. content can be a string or a list of
            ContentParts (text + image_url in v1; audio/video in v2).
        client_context: Optional device/environment context. Drives battery,
            connectivity, and on-device model routing decisions.
        temperature: Sampling temperature passed through to the provider (0–2).
        max_tokens: Maximum output tokens passed through to the provider.
        max_cost: Explicit cost cap in USD. Overrides server-resolved tier cap.
        max_latency_ms: Maximum acceptable latency in milliseconds.
        require_local: Explicit local-only flag. Overrides all context fields.
    """
    messages: List[Message]
    client_context: Optional[ClientContext] = None
    # Standard inference params — passed through to the selected provider
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Explicit FDE overrides — bypass server-resolved policy when set
    max_cost: Optional[float] = None
    max_latency_ms: Optional[int] = None
    require_local: Optional[bool] = None


class InferResult(BaseModel):
    """Result of a DIO routing and inference operation.

    Attributes:
        provider: Name of the selected provider.
        model: Specific model used, if set on the provider.
        content: Generated response text (v2: may include multimodal output).
        routed_by: Routing mode used (always "fde" in this server).
        metadata: Routing scores and decision metadata from the FDE.
    """
    provider: str
    model: Optional[str] = None
    content: str
    routed_by: str
    metadata: Dict[str, Any] = {}
