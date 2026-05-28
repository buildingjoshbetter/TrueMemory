"""Internal debug log + opt-in gate for the instrumentation overlay.

``is_enabled()`` is the single opt-in switch for the whole package: every
public emit and every monkey-patch checks it, so when
``TRUEMEMORY_INSTRUMENTATION`` is unset the overlay is inert.

The gate is read **live** from the environment on every call (not cached at
import) so a host or a test can toggle ``TRUEMEMORY_INSTRUMENTATION`` after
the module has been imported and have it take effect immediately.

``dlog()`` appends a timestamped line to
``~/.truememory/logs/instrumentation.log`` — a human-readable trace of which
patches applied and which signals fired. All writes are exception-swallowed
so a logging failure can never bring down the MCP server. When the overlay
is disabled, ``dlog()`` is a no-op.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path

_LOG_PATH = Path.home() / ".truememory" / "logs" / "instrumentation.log"
_write_lock = threading.Lock()
_dir_created = False

_TRUTHY = ("1", "true", "yes", "on")


def is_enabled() -> bool:
    """Return True when ``TRUEMEMORY_INSTRUMENTATION`` is set to a truthy value.

    This is the opt-in gate for the entire instrumentation package. Read live
    so the host/tests can flip it after import. Default (unset) → disabled.
    """
    return os.environ.get("TRUEMEMORY_INSTRUMENTATION", "").strip().lower() in _TRUTHY


def dlog(msg: str) -> None:
    """Append a timestamped, pid-tagged line to the instrumentation log.

    No-op when the overlay is disabled. Never raises.
    """
    if not is_enabled():
        return
    global _dir_created
    try:
        if not _dir_created:
            try:
                _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                _dir_created = True
            except Exception:
                pass
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:23]
        line = f"{ts} pid={os.getpid()} tid={threading.get_ident()} {msg}\n"
        with _write_lock:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass
