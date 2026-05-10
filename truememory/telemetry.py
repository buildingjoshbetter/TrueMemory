"""
Fire-and-forget usage telemetry for TrueMemory.

Disabled via TRUEMEMORY_TELEMETRY=off or {"telemetry": false} in config.
Never blocks, never crashes, never slows down the user. All HTTP calls
have a 3-second timeout. If the endpoint is unreachable, events are
silently dropped.

What is tracked:
  - Tool call counts and latencies (which MCP tools are used)
  - Session start/end events
  - Tier, version, platform
  - Email + UUID (on first registration)

What is NEVER tracked:
  - Query content or search terms
  - Memory content or stored facts
  - File paths, API keys, or credentials
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import uuid
from functools import wraps
from pathlib import Path

_TELEMETRY_ENDPOINT = "https://telemetry.truememory.ai/v1/events"
_FLUSH_INTERVAL = 60  # seconds
_HTTP_TIMEOUT = 3  # seconds

_enabled: bool | None = None
_user_id: str = ""
_session_events: list[dict] = []
_lock = threading.Lock()
_flush_thread: threading.Thread | None = None


def init(config: dict) -> None:
    """Initialize telemetry. Call once during MCP server startup."""
    global _enabled, _user_id, _flush_thread

    if not is_enabled():
        _enabled = False
        return

    _enabled = True

    # Get or generate user_id
    _user_id = config.get("user_id", "")
    if not _user_id:
        _user_id = str(uuid.uuid4())
        config["user_id"] = _user_id
        _save_user_id(config)

    # Start background flush thread
    _flush_thread = threading.Thread(target=_flush_loop, daemon=True)
    _flush_thread.start()

    # Track session start
    track("session_start", {
        "tier": config.get("tier", "edge"),
        "version": _get_version(),
        "platform": sys.platform,
        "arch": platform.machine(),
        "python": platform.python_version(),
    })


def is_enabled() -> bool:
    """Check if telemetry is enabled."""
    global _enabled
    if _enabled is not None:
        return _enabled

    # Check env var
    env = os.environ.get("TRUEMEMORY_TELEMETRY", "").lower()
    if env in ("off", "false", "0", "no"):
        _enabled = False
        return False

    # Check config
    try:
        config_path = Path.home() / ".truememory" / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("telemetry") is False:
                _enabled = False
                return False
    except Exception:
        pass

    _enabled = True
    return True


def track(event: str, properties: dict | None = None) -> None:
    """Record a telemetry event. Non-blocking, fire-and-forget."""
    if not _enabled:
        return

    entry = {
        "event": event,
        "user_id": _user_id,
        "timestamp": time.time(),
        "properties": properties or {},
    }

    with _lock:
        _session_events.append(entry)


def identify(email: str, properties: dict | None = None) -> None:
    """Register user identity (first-run only)."""
    if not _enabled:
        return

    track("identify", {
        "email": email,
        **(properties or {}),
    })


def tracked(event_name: str):
    """Decorator that emits a telemetry event after a tool function runs."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not _enabled:
                return fn(*args, **kwargs)
            start = time.monotonic()
            success = True
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception:
                success = False
                raise
            finally:
                try:
                    track(event_name, {
                        "latency_ms": round((time.monotonic() - start) * 1000, 1),
                        "success": success,
                    })
                except Exception:
                    pass
        return wrapper
    return decorator


def _flush_loop() -> None:
    """Background thread that flushes events periodically."""
    while True:
        time.sleep(_FLUSH_INTERVAL)
        try:
            _flush()
        except Exception:
            pass


def _flush() -> None:
    """Send batched events to the telemetry endpoint."""
    with _lock:
        if not _session_events:
            return
        batch = _session_events.copy()
        _session_events.clear()

    try:
        import httpx
        httpx.post(
            _TELEMETRY_ENDPOINT,
            json={"events": batch},
            timeout=_HTTP_TIMEOUT,
        )
    except Exception:
        pass


def _save_user_id(config: dict) -> None:
    """Persist user_id back to config.json."""
    try:
        config_path = Path.home() / ".truememory" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        config_path.chmod(0o600)
    except Exception:
        pass


def _get_version() -> str:
    """Get the truememory version string."""
    try:
        from truememory import __version__
        return __version__
    except Exception:
        return "unknown"
