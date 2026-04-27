"""Structured logging utility for DIO.

All server and core components use log_event() to emit NDJSON log entries
with a consistent schema. Centralising here means:
  - `ts` (UTC ISO 8601) is always present and always correct
  - Switching to structlog or wiring a pipeline processor requires
    changes in exactly one place
  - Call sites stay readable — no repeated json.dumps boilerplate

Usage:
    from aigentic.logging import log_event
    log_event(logger, "warning", "stream_provider_failed",
              provider="claude-haiku", error_type="AuthenticationError", error="...")
"""

import json
import logging
from datetime import datetime, timezone


def log_event(logger: logging.Logger, level: str, event: str, **fields) -> None:
    """Emit a structured NDJSON log entry.

    Args:
        logger: Logger instance to use.
        level:  Standard logging level name: "debug", "info", "warning", "error".
        event:  Snake_case event name — the primary key for log pipelines.
        **fields: Additional key/value pairs merged into the record.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    getattr(logger, level)(json.dumps(record))
