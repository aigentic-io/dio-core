"""Translates ClientContext into FDE routing kwargs.

User identity (tier, policies, preferred_providers, etc.) is resolved
server-side from the Bearer token and is not part of this translation layer.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from aigentic.server.models import ClientContext


def to_fde_kwargs(
    client_context: Optional["ClientContext"] = None,
) -> Dict[str, Any]:
    """Translate ClientContext into DIO.route() kwargs.

    Rules applied in priority order:
    1. ClientContext.connectivity="offline" → require_local (no network)
    2. ClientContext.battery_level < 20    → require_local (avoid HTTP drain)

    User-level constraints (tier cost caps, privacy policies, org policies)
    are resolved server-side from the auth token and applied by the caller
    before passing to DIO.route().

    Args:
        client_context: Device/environment state from the client.

    Returns:
        Dict of kwargs ready to pass to DIO.route().
    """
    kwargs: Dict[str, Any] = {}

    if client_context is not None:
        if client_context.connectivity == "offline":
            kwargs["require_local"] = True
        elif client_context.battery_level is not None and client_context.battery_level < 20:
            kwargs["require_local"] = True  # avoid HTTP drain on critical battery

    return kwargs
