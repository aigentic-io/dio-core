"""Fallback and resilience policies for DIO."""

from typing import Type


class FallbackPolicy:
    """Policy for handling provider failures with fallback options."""

    def __init__(
        self,
        primary_provider: str,
        fallback_provider: str,
        trigger_exception: Type[Exception] = Exception,
    ):
        """Initialize fallback policy.

        Args:
            primary_provider: Name of the primary provider to try first
            fallback_provider: Name of the fallback provider
            trigger_exception: Exception type that triggers fallback
        """
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.trigger_exception = trigger_exception

    def should_fallback(self, exception: Exception) -> bool:
        """Determine if the exception should trigger fallback.

        Args:
            exception: Exception that occurred

        Returns:
            True if fallback should be triggered
        """
        return isinstance(exception, self.trigger_exception)
