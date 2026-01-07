"""Response dataclass for DIO routing results."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Response:
    """Response from a DIO routing operation.

    Attributes:
        content: The response content from the provider
        provider: Name of the provider that handled the request
        was_fallback: Whether this response came from a fallback provider
        metadata: Additional metadata about the routing decision
    """
    content: str
    provider: str
    was_fallback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
