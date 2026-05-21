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

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from functools import wraps
from pathlib import Path

_TELEMETRY_ENDPOINT = "https://telemetry-api-production-c2a3.up.railway.app/v1/events"
_FLUSH_INTERVAL = 60  # seconds
_HTTP_TIMEOUT = 3  # seconds

_enabled: bool | None = None
_user_id: str = ""
_session_events: list[dict] = []
_lock = threading.Lock()
_flush_thread: threading.Thread | None = None


_device_id_cache: str | None = None
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,10}$")


def get_device_id() -> str:
    """Return a SHA256-hashed machine ID for privacy-safe device counting."""
    global _device_id_cache
    if _device_id_cache is not None:
        return _device_id_cache

    raw = ""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True, text=True, timeout=5,
            )
            for line in out.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    raw = line.split('"')[-2]
                    break
        elif sys.platform == "linux":
            mid = Path("/etc/machine-id")
            if mid.exists():
                raw = mid.read_text().strip()
        elif sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            )
            raw, _ = winreg.QueryValueEx(key, "MachineGuid")
            winreg.CloseKey(key)
    except Exception:
        pass

    if raw:
        _device_id_cache = hashlib.sha256(raw.encode()).hexdigest()[:16]
    else:
        return "unknown"
    return _device_id_cache


def _is_valid_email(email: str) -> bool:
    """Basic email format check. Rejects garbage without being overly strict."""
    if not email or len(email) > 254:
        return False
    return _EMAIL_RE.fullmatch(email) is not None


def init(config: dict) -> dict | None:
    """Initialize telemetry. Call once during MCP server startup.

    Returns update info dict if a newer version is available, or None.
    """
    global _enabled, _user_id, _flush_thread

    if not is_enabled():
        _enabled = False
        return None

    _enabled = True

    # Get or generate user_id
    _user_id = config.get("user_id", "")
    if not _user_id:
        _user_id = str(uuid.uuid4())
        config["user_id"] = _user_id
        _save_user_id(config)

    # Track session start and do a synchronous flush to check for updates
    session_props = {
        "tier": config.get("tier", "edge"),
        "version": _get_version(),
        "platform": sys.platform,
        "arch": platform.machine(),
        "python": platform.python_version(),
        "device_id": get_device_id(),
    }
    if config.get("email"):
        session_props["email"] = config["email"]
    track("session_start", session_props)

    def _init_flush():
        info = _flush_sync()
        if info and info.get("update_available"):
            try:
                _p = Path.home() / ".truememory" / ".update_available"
                _tmp = _p.with_suffix(".tmp")
                _tmp.write_text(json.dumps(info), encoding="utf-8")
                _tmp.rename(_p)
            except Exception:
                pass

    threading.Thread(target=_init_flush, daemon=True).start()

    _flush_thread = threading.Thread(target=_flush_loop, daemon=True)
    _flush_thread.start()

    return None


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
    if not _is_valid_email(email):
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
    _flush_sync()


def _flush_sync() -> dict | None:
    """Flush events and return the server response (for update checks)."""
    with _lock:
        if not _session_events:
            return None
        batch = _session_events.copy()
        _session_events.clear()

    try:
        import httpx
        resp = httpx.post(
            _TELEMETRY_ENDPOINT,
            json={"events": batch},
            timeout=_HTTP_TIMEOUT,
        )
        data = resp.json()
        if data.get("update_available"):
            return data
    except Exception:
        pass
    return None


def _save_user_id(config: dict) -> None:
    """Persist user_id back to config.json."""
    try:
        config_path = Path.home() / ".truememory" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.parent.chmod(0o700)
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
