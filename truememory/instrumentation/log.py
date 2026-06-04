"""
Instrumentation logging and opt-in check.

The ``is_enabled()`` result is cached at ``install()`` time so the hot
path never reads ``os.environ`` per call.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Cached enabled state.  None = not yet locked (read env each time).
# True/False = locked by install()/uninstall().
_ENABLED: bool | None = None


def is_enabled() -> bool:
    """Return True if instrumentation is active.

    After ``install()`` runs, the result is cached and env is not re-read.
    """
    if _ENABLED is not None:
        return _ENABLED
    return os.environ.get("TRUEMEMORY_INSTRUMENTATION", "").strip().lower() in _TRUTHY


def _lock_enabled(value: bool | None) -> None:
    """Lock (or unlock) the cached enabled state.

    Called by ``install()`` and ``uninstall()``.  Tests call
    ``_lock_enabled(None)`` to reset.
    """
    global _ENABLED
    _ENABLED = value


def dlog(msg: str, *args) -> None:
    """Debug-level log for instrumentation internals."""
    log.debug("[instrumentation] " + msg, *args)
