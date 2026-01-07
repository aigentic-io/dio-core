"""PII (Personally Identifiable Information) detection using regex patterns.

Privacy-first implementation with no AI/cloud calls.
"""

import re
from typing import Dict, List, Pattern


class PIIDetector:
    """Detects PII in text using regex patterns."""

    # Regex patterns for common PII types
    PATTERNS: Dict[str, Pattern] = {
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    }

    @classmethod
    def detect(cls, text: str) -> Dict[str, List[str]]:
        """Detect all PII types in the given text.

        Args:
            text: Text to scan for PII

        Returns:
            Dictionary mapping PII type to list of matches
        """
        results = {}
        for pii_type, pattern in cls.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                results[pii_type] = matches
        return results

    @classmethod
    def has_pii(cls, text: str) -> bool:
        """Check if text contains any PII.

        Args:
            text: Text to scan for PII

        Returns:
            True if any PII is detected, False otherwise
        """
        for pattern in cls.PATTERNS.values():
            if pattern.search(text):
                return True
        return False


# Convenience function for workshop slides
def has_pii(text: str) -> bool:
    """Check if text contains PII (convenience function).

    Args:
        text: Text to scan for PII

    Returns:
        True if any PII is detected, False otherwise
    """
    return PIIDetector.has_pii(text)
