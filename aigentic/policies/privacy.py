"""Privacy policy for routing sensitive data to secure providers."""

from aigentic.core.pii_detector import has_pii
from aigentic.core.router import Request


def privacy_policy(request: Request) -> str:
    """Route requests with PII to restricted/local providers.

    Args:
        request: Request object containing the prompt

    Returns:
        "RESTRICTED" if PII detected, "PUBLIC" otherwise
    """
    if has_pii(request.prompt):
        return "RESTRICTED"
    return "PUBLIC"
