"""Registry client: fetches model pricing from dio-registry CDN, caches in Redis + memory.

Background sync starts automatically when DIO is instantiated. No manual setup required.

Environment:
    REDIS_URL            Redis connection URL (e.g. redis://localhost:6379/0).
                         If unset, pricing data is cached in-memory only (lost on restart).
    REGISTRY_SYNC_HOUR   Local hour (0-23) for daily sync. Default: 3 (3am local time).
"""

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("dio.registry")

_CDN_URL = "https://cdn.jsdelivr.net/gh/aigentic-io/dio-registry@main/model_registry.json"
_REDIS_KEY = "dio:registry:models"
_SYNC_HOUR = int(os.getenv("REGISTRY_SYNC_HOUR", "3"))

_memory_cache: Optional[dict] = None
_write_lock = threading.Lock()
_redis_client = None
_bg_thread: Optional[threading.Thread] = None


# ── Redis ──────────────────────────────────────────────────────────────────────

def _init_redis() -> None:
    global _redis_client
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        logger.info(json.dumps({
            "event": "registry_redis_skipped",
            "reason": "REDIS_URL not set — using in-memory cache only",
        }))
        return
    try:
        import redis
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        logger.info(json.dumps({"event": "registry_redis_connected"}))
    except ImportError:
        logger.warning(json.dumps({
            "event": "registry_redis_unavailable",
            "reason": "install redis: pip install aigentic[registry]",
        }))
    except Exception as e:
        logger.warning(json.dumps({"event": "registry_redis_connect_failed", "reason": str(e)}))


def _redis_read() -> Optional[dict]:
    if _redis_client is None:
        return None
    try:
        raw = _redis_client.get(_REDIS_KEY)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning(json.dumps({"event": "registry_redis_read_failed", "reason": str(e)}))
        return None


def _redis_write(data: dict) -> None:
    if _redis_client is None:
        return
    try:
        _redis_client.set(_REDIS_KEY, json.dumps(data))
    except Exception as e:
        logger.warning(json.dumps({"event": "registry_redis_write_failed", "reason": str(e)}))


# ── CDN fetch ─────────────────────────────────────────────────────────────────

async def _fetch_cdn() -> Optional[dict]:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_CDN_URL)
            resp.raise_for_status()
            return resp.json()
    except ImportError:
        logger.warning(json.dumps({
            "event": "registry_cdn_unavailable",
            "reason": "install httpx: pip install aigentic[registry]",
        }))
        return None
    except Exception as e:
        logger.warning(json.dumps({"event": "registry_cdn_fetch_failed", "reason": str(e)}))
        return None


# ── Sync ──────────────────────────────────────────────────────────────────────

async def sync_registry() -> None:
    """Fetch CDN → update Redis and in-memory cache. Keeps existing data on fetch failure."""
    global _memory_cache
    data = await _fetch_cdn()
    if data is None:
        logger.warning(json.dumps({
            "event": "registry_sync_skipped",
            "reason": "cdn_fetch_failed — existing data retained",
        }))
        return
    with _write_lock:
        _memory_cache = data
    _redis_write(data)
    logger.info(json.dumps({
        "event": "registry_synced",
        "model_count": len(data.get("models", {})),
        "version": data.get("metadata", {}).get("version"),
    }))


async def _startup() -> None:
    _init_redis()
    data = _redis_read()
    if data:
        global _memory_cache
        with _write_lock:
            _memory_cache = data
        logger.info(json.dumps({"event": "registry_loaded_from_redis", "model_count": len(data.get("models", {}))}))
    else:
        await sync_registry()


async def _daily_loop() -> None:
    while True:
        now = datetime.now().astimezone()
        target = now.replace(hour=_SYNC_HOUR, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        sleep_secs = (target - now).total_seconds()
        logger.info(json.dumps({"event": "registry_next_sync_scheduled", "seconds": round(sleep_secs)}))
        await asyncio.sleep(sleep_secs)
        await sync_registry()


def _thread_main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_startup())
        loop.run_until_complete(_daily_loop())
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()


# ── Public API ────────────────────────────────────────────────────────────────

def start() -> None:
    """Start registry background sync thread. Called automatically by DIO.__init__."""
    global _bg_thread
    if _bg_thread is not None and _bg_thread.is_alive():
        return
    _bg_thread = threading.Thread(target=_thread_main, name="dio-registry", daemon=True)
    _bg_thread.start()


def _four_step_lookup(models: dict, name: str) -> Optional[dict]:
    # Strip gateway namespace prefix (e.g. "openai/gpt-4o" → "gpt-4o")
    if "/" in name:
        name = name.split("/", 1)[1]
    # 1. Exact match
    if name in models:
        return models[name]
    # 2. Strip Ollama tag suffix (e.g. "llama3:latest" → "llama3")
    if ":" in name:
        stripped = name.split(":")[0]
        if stripped in models:
            return models[stripped]
    # 3. Colon → hyphen
    hyphenated = name.replace(":", "-")
    if hyphenated in models:
        return models[hyphenated]
    # 4. Longest prefix match
    for key in sorted(models, key=len, reverse=True):
        if name.startswith(key):
            return models[key]
    return None


def get_pricing(model_name: str, modality: str = "text") -> Optional[dict]:
    """Return the matching pricing plan for a model and modality, or None if not in registry.

    The returned dict contains: plan_name, currency, modality[], base (tier/input/output),
    caching (creation/read/storage), and optionally tiered_overrides[].

    Falls back to the first plan if no plan explicitly covers the requested modality.
    Returns None if the registry has not yet synced or the model is unknown.
    """
    if not model_name or _memory_cache is None:
        return None
    models = _memory_cache.get("models", {})
    entry = _four_step_lookup(models, model_name.lower().strip())
    if entry is None:
        return None
    plans = entry.get("pricing", {}).get("pricing_plans", [])
    for plan in plans:
        if modality in plan.get("modality", []):
            return plan
    return plans[0] if plans else None
