"""Model capability registry backed by a static LMSYS Arena ELO snapshot.

Provides capability scores (0.0–1.0) for common LLM model names so that
Provider() can auto-populate the capability field without requiring the caller
to look up benchmark data manually.

Matching strategy (in order):
  1. Exact match on the lowercased model name
  2. Ollama tag stripping — "llama3:latest" → "llama3"
  3. Ollama colon-to-hyphen — "gpt-oss:20b" → "gpt-oss-20b" (size-specific entry)
  4. Prefix match (longest key first) — "gpt-4o-2024-11-20" → "gpt-4o"

Unknown models return (0.7, False) — a conservative mid-tier assumption.
Providers with no model= set at all fall back to 0.7.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

_DATA_FILE = Path(__file__).parent / "data" / "model_capabilities.json"

_registry: Optional[dict] = None


def _load() -> dict:
    global _registry
    if _registry is None:
        with open(_DATA_FILE) as f:
            _registry = json.load(f)
    return _registry


def get_capability(model_name: str) -> Tuple[float, bool]:
    """Look up the capability score for a model.

    Args:
        model_name: Model identifier string (e.g. "gpt-4o", "llama3:latest",
                    "claude-3-5-sonnet-20241022").

    Returns:
        (score, found) where score is 0.0–1.0 and found indicates a registry
        hit. When found is False the score is the fallback default (0.7).
    """
    if not model_name:
        return 0.7, False

    reg = _load()
    scores: dict = reg.get("scores", {})

    name = model_name.lower().strip()

    # 1. Exact match
    if name in scores:
        return scores[name], True

    # 2. Strip Ollama-style tag (e.g. "llama3:latest" → "llama3")
    base = name.split(":")[0]
    if base != name and base in scores:
        return scores[base], True

    # 3. Ollama colon-to-hyphen conversion (e.g. "gpt-oss:20b" → "gpt-oss-20b")
    #    Keeps size-specific entries distinct ("gpt-oss:20b"→0.74, "gpt-oss:120b"→0.79)
    if ":" in name:
        hyphen_name = name.replace(":", "-", 1)
        if hyphen_name in scores:
            return scores[hyphen_name], True

    # 4. Prefix match — check longest keys first to avoid short-circuit mismatches
    #    (e.g. "llama3.2" should win over "llama3" for "llama3.2:3b")
    for key in sorted(scores.keys(), key=len, reverse=True):
        if name.startswith(key) or base.startswith(key):
            return scores[key], True

    return 0.7, False


def snapshot_info() -> dict:
    """Return metadata about the current registry snapshot."""
    reg = _load()
    return {
        "source": reg.get("_source"),
        "snapshot_date": reg.get("_snapshot_date"),
        "model_count": len(reg.get("scores", {})),
    }
