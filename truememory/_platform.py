"""Platform helpers shared between model_server and model_client.

Centralises transport detection, PID liveness checks, and subprocess
creation flags so the two modules stay in sync.
"""

import os
import socket
import sys

# ---------------------------------------------------------------------------
# Transport detection
# ---------------------------------------------------------------------------

_USE_UNIX: bool = hasattr(socket, "AF_UNIX") and sys.platform != "win32"
"""True when AF_UNIX sockets are available (all POSIX systems)."""

_LOOPBACK_HOST: str = "127.0.0.1"
"""TCP fallback binds/connects here on Windows."""


# ---------------------------------------------------------------------------
# Cross-platform PID liveness
# ---------------------------------------------------------------------------

def pid_is_alive(pid: int) -> bool:
    """Return *True* if *pid* refers to a running process."""
    if sys.platform == "win32":
        import psutil  # always available -- core dependency
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


# ---------------------------------------------------------------------------
# Subprocess creation flags
# ---------------------------------------------------------------------------

def spawn_kwargs() -> dict:
    """Return platform-specific :class:`subprocess.Popen` kwargs to detach
    the model-server process.

    On Windows the combination of ``CREATE_NO_WINDOW``,
    ``DETACHED_PROCESS``, and ``CREATE_NEW_PROCESS_GROUP`` prevents a
    console window from flashing and fully detaches the child.
    """
    if sys.platform == "win32":
        CREATE_NO_WINDOW = 0x08000000
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        return {
            "creationflags": (
                CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            ),
        }
    return {"start_new_session": hasattr(os, "setsid")}
