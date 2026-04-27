"""Unit tests for aigentic.logging.log_event."""

import json
from unittest.mock import MagicMock

from aigentic.logging import log_event


def test_log_event_always_includes_ts_and_event():
    """log_event must always emit ts (UTC ISO 8601) and event in the record."""
    mock_logger = MagicMock()
    log_event(mock_logger, "info", "request_complete", provider="claude-haiku", wall_ms=42)
    mock_logger.info.assert_called_once()
    record = json.loads(mock_logger.info.call_args[0][0])
    assert "ts" in record
    assert record["ts"].endswith("+00:00") or record["ts"].endswith("Z")
    assert record["event"] == "request_complete"
    assert record["provider"] == "claude-haiku"
    assert record["wall_ms"] == 42


def test_log_event_routes_level_correctly():
    """log_event must call only the requested level method."""
    for level in ("debug", "info", "warning", "error"):
        mock_logger = MagicMock()
        log_event(mock_logger, level, "some_event")
        getattr(mock_logger, level).assert_called_once()
        for other in ("debug", "info", "warning", "error"):
            if other != level:
                getattr(mock_logger, other).assert_not_called()


def test_log_event_output_is_valid_ndjson():
    """log_event output must be valid JSON (parseable by log pipelines)."""
    mock_logger = MagicMock()
    log_event(mock_logger, "warning", "stream_provider_failed",
              provider="gemini-flash", error_type="ClientError", error="400 Bad Request")
    raw = mock_logger.warning.call_args[0][0]
    record = json.loads(raw)  # raises if invalid JSON
    assert record["event"] == "stream_provider_failed"
    assert record["error_type"] == "ClientError"


def test_log_event_no_extra_fields():
    """log_event with no kwargs emits only ts and event."""
    mock_logger = MagicMock()
    log_event(mock_logger, "info", "health_check")
    record = json.loads(mock_logger.info.call_args[0][0])
    assert set(record.keys()) == {"ts", "event"}
