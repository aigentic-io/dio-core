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


# ── OpenAI / OpenRouter compatible request & response ─────────────────────────

class ChatCompletionRequest(BaseModel):
    """OpenAI / OpenRouter compatible chat completion request.

    Standard fields (temperature, max_tokens, stream) are passed through to the
    selected provider. DIO-specific fields (max_cost, max_latency_ms,
    require_local, client_context) are routing hints — standard OR clients that
    omit them will get pure FDE routing with no extra constraints.

    The model field is a hint only in Sprint 1. FDE picks the best available
    provider regardless. In Sprint 2 shadow mode, model identifies the 'original'
    request to replay for quality comparison.

    Headers consumed server-side (not in this model):
        Authorization: Bearer <jwt>      — identity, tier, org, routing policies
        Accept-Language: zh-CN,en;q=0.8  — ordered language preference
        X-Session-ID: <uuid>             — groups requests in a conversation
        X-Request-ID: <uuid>             — per-request trace ID (echoed in response)
        X-DIO-Shadow: true               — enable shadow mode (Sprint 2)
    """
    messages: List[Message]
    model: Optional[str] = None          # OR-style hint, e.g. "openai/gpt-4o-mini"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    # DIO routing overrides — ignored by standard OR clients
    max_cost: Optional[float] = None
    max_latency_ms: Optional[int] = None
    require_local: Optional[bool] = None
    client_context: Optional[ClientContext] = None


class ChatCompletionUsage(BaseModel):
    """Exact token counts from the provider response."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI / OpenRouter compatible chat completion response.

    Fully compatible with the OpenAI response schema so clients can swap the
    base URL and receive structured responses without code changes.

    x_dio is an extension field (null for non-DIO clients) carrying routing
    metadata: which provider was selected, why, and the estimated cost.
    """
    id: str                                    # "chatcmpl-<uuid>"
    object: str = "chat.completion"
    created: int                               # unix timestamp
    model: str                                 # actual model used
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage                 # exact counts from provider
    x_dio: Optional[Dict[str, Any]] = None    # routing metadata extension


# ── Shadow mode ───────────────────────────────────────────────────────────────

class ShadowIngestRequest(BaseModel):
    """Passive shadow ingestion — client provides their original request + response.

    The `request` field is the exact ChatCompletionRequest the client sent to
    their provider. `request.model` is required (identifies the original model).
    `request.client_context` drives FDE scoring so DIO's routing decision
    reflects the same device/network conditions as the original call.

    DIO runs FDE scoring, writes a self-contained shadow record, and returns
    202 immediately. No LLM call is made. DIO-side fields (dio_model, dio_cost,
    fde_score) are computed from the FDE analysis. dio_response and
    similarity_score are left null and filled in by offline batch processing.

    org_id scopes the record for multi-tenant analytics.
    When original_cost_usd is omitted, DIO looks it up from the registry.
    """
    org_id: Optional[str] = None
    request: ChatCompletionRequest               # exact request sent to original provider
    original_response: str                       # text content their provider returned
    original_usage: Dict[str, int]               # {"prompt_tokens": N, "completion_tokens": N}
    original_latency_ms: int
    original_cost_usd: Optional[float] = None    # omit → DIO looks up from registry
