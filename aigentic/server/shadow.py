"""Shadow mode: record building and NDJSON file writer."""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_shadow_record(
    *,
    request_id: Optional[str],
    org_id: Optional[str],
    request_messages: List[dict],
    original_model: str,
    original_response: str,
    original_usage: Dict[str, int],
    original_latency_ms: int,
    original_cost_usd: Optional[float],
    dio_model: str,
    dio_response: Optional[str],
    dio_usage: Dict[str, int],
    dio_latency_ms: int,
    dio_cost_usd: float,
    fde_score: Optional[float],
    complexity: str,
    has_pii: bool,
) -> Dict[str, Any]:
    """Build a self-contained shadow record suitable for NDJSON serialisation.

    When dio_response is None (FDE-only analysis, no actual DIO call),
    similarity_score and response_match are null — cost fields still reflect
    the estimated saving based on token counts from the original request.
    """
    if original_cost_usd is not None:
        saving = round(original_cost_usd - dio_cost_usd, 10)
        saving_pct = round(saving / original_cost_usd * 100, 2) if original_cost_usd > 0 else 0.0
    else:
        saving, saving_pct = None, None

    similarity_score, response_match = None, None

    return {
        "record_id": str(uuid.uuid4()),
        "request_id": request_id,
        "org_id": org_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_messages": request_messages,
        "original_model": original_model,
        "original_response": original_response,
        "original_usage": original_usage,
        "original_latency_ms": original_latency_ms,
        "original_cost_usd": original_cost_usd,
        "dio_model": dio_model,
        "dio_response": dio_response,
        "dio_usage": dio_usage,
        "dio_latency_ms": dio_latency_ms,
        "dio_cost_usd": dio_cost_usd,
        "cost_saving_usd": saving,
        "cost_saving_pct": saving_pct,
        "similarity_score": similarity_score,
        "response_match": response_match,
        "fde_score": fde_score,
        "complexity": complexity,
        "has_pii": has_pii,
    }


class _DatedRotatingHandler(logging.Handler):
    """logging.Handler that writes NDJSON to SHADOW_LOG_DIR/YYYY-MM-DD/shadow_NNN.ndjson.

    Files are rotated when they exceed max_bytes. On day rollover a new
    date subdirectory is created automatically. On server restart the last
    file for the current day is re-opened for appending if still under quota,
    otherwise a new sequence number is started.

    The sequential NNN suffix (000, 001, …) is unambiguous and makes it
    straightforward for a future S3/GCS shipper to list completed files
    (everything except the highest-numbered file in a day dir).
    """

    def __init__(self, base_dir: str, max_bytes: int = 1_073_741_824):
        super().__init__()
        self._base_dir = Path(base_dir)
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._date: Optional[str] = None
        self._seq: int = 0
        self._fh = None
        self._open()

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _day_dir(self, date: str) -> Path:
        return self._base_dir / date

    def _path(self, date: str, seq: int) -> Path:
        return self._day_dir(date) / f"shadow_{seq:03d}.ndjson"

    def _open(self) -> None:
        today = self._today()
        day_dir = self._day_dir(today)
        day_dir.mkdir(parents=True, exist_ok=True)

        # Find the highest existing sequence number for today.
        existing = sorted(day_dir.glob("shadow_*.ndjson"))
        if existing:
            last = existing[-1]
            try:
                seq = int(last.stem.split("_")[1])
            except (IndexError, ValueError):
                seq = len(existing) - 1
            # Resume appending to the last file if it's still under quota.
            if last.stat().st_size < self._max_bytes:
                self._date = today
                self._seq = seq
                self._fh = last.open("a", encoding="utf-8")
                return
            # Last file is full — start the next one.
            seq += 1
        else:
            seq = 0

        self._date = today
        self._seq = seq
        self._fh = self._path(today, seq).open("a", encoding="utf-8")

    def _rotate(self) -> None:
        """Close the current file and open the next one (new seq or new day)."""
        if self._fh:
            self._fh.close()
        today = self._today()
        if today != self._date:
            # Day rolled over — new directory, reset sequence.
            day_dir = self._day_dir(today)
            day_dir.mkdir(parents=True, exist_ok=True)
            self._date = today
            self._seq = 0
        else:
            self._seq += 1
        self._fh = self._path(self._date, self._seq).open("a", encoding="utf-8")

    # ── logging.Handler interface ─────────────────────────────────────────────

    def emit(self, record: logging.LogRecord) -> None:
        line = self.format(record) + "\n"
        with self._lock:
            # Day rollover or size quota exceeded → open next file.
            if self._today() != self._date or self._fh.tell() >= self._max_bytes:
                self._rotate()
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None
        super().close()


class ShadowWriter:
    """Writes shadow records as NDJSON to a dated directory tree or stdout.

    When log_dir is set (via SHADOW_LOG_DIR env var), records are written to:
        <log_dir>/YYYY-MM-DD/shadow_NNN.ndjson

    Files rotate when they exceed max_bytes (SHADOW_LOG_MAX_BYTES, default 1 GB).
    The S3/GCS shipper treats all but the highest-numbered file per day as
    complete and eligible for upload — no additional coordination needed.

    When log_dir is unset, records propagate to the root logger (stdout).
    """

    def __init__(self, log_dir: Optional[str] = None, max_bytes: int = 1_073_741_824):
        self._logger = logging.getLogger("dio.shadow")
        self._logger.setLevel(logging.INFO)
        # Clear handlers from any previous instance sharing this logger (e.g. test fixtures).
        for h in self._logger.handlers[:]:
            h.close()
            self._logger.removeHandler(h)
        if log_dir:
            handler = _DatedRotatingHandler(log_dir, max_bytes=max_bytes)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.propagate = False  # don't duplicate to root/stdout

    def write(self, record: Dict[str, Any]) -> None:
        self._logger.info(json.dumps(record, default=str))
