"""Platform helpers shared between model_server and model_client.

Centralises transport detection, PID liveness checks, and subprocess
creation flags so the two modules stay in sync.
"""

import errno
import json
import math
import os
import socket
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Transport detection
# ---------------------------------------------------------------------------

_USE_UNIX: bool = hasattr(socket, "AF_UNIX") and sys.platform != "win32"
"""True when AF_UNIX sockets are available (all POSIX systems)."""

_LOOPBACK_HOST: str = "127.0.0.1"
"""TCP fallback binds/connects here on Windows."""

_MODEL_SERVER_GENERATION_ENV = "TRUEMEMORY_MODEL_SERVER_GENERATION"


def try_file_lock(path: Path) -> int | None:
    """Acquire a process-owned, nonblocking lock; closing the fd releases it."""
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if sys.platform == "win32":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.close(fd)
        if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            return None
        raise
    return fd


def process_birth(pid: int) -> float | None:
    """Identify a process incarnation without trusting a reused PID alone."""
    import psutil
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process.create_time()
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied as error:
        raise OSError("Cannot verify model-server launch ownership") from error


def read_start_claim(path: Path) -> dict | None:
    """Read a small launch claim. Invalid generations are never authorized."""
    try:
        with path.open(encoding="utf-8") as stream:
            data = stream.read(4097)
    except FileNotFoundError:
        return None
    except UnicodeError:
        return None
    if len(data) > 4096:
        return None
    try:
        claim = json.loads(data)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(claim, dict):
        return None
    generation = claim.get("generation")
    if (not isinstance(generation, str) or len(generation) != 32
            or any(char not in "0123456789abcdef" for char in generation)):
        return None
    for key in ("launcher", "child"):
        identity = claim.get(key)
        if identity is None and key == "child":
            continue
        if not isinstance(identity, dict):
            return None
        pid, born = identity.get("pid"), identity.get("born")
        if (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                or isinstance(born, bool) or not isinstance(born, (int, float))
                or not math.isfinite(born) or born <= 0):
            return None
    claim.setdefault("child", None)
    return claim


# ---------------------------------------------------------------------------
# Safe environment-variable parsing (issue #639)
# ---------------------------------------------------------------------------

def _env_int(
    name: str,
    default: int,
    lo: int | None = None,
    hi: int | None = None,
) -> int:
    """Read integer env var *name*, never crashing at import.

    An unset, empty, or non-numeric value (e.g. ``""``, ``"abc"``, ``"1.5"``)
    returns *default* instead of raising ``ValueError`` — a module-level bare
    ``int(os.environ.get(...))`` would otherwise crash the whole hook/server
    at import, before any ``main()`` try/except can catch it (M-27).

    When *lo* / *hi* are given the parsed value is clamped into ``[lo, hi]``,
    guarding against negative/zero knobs that parse but misbehave — e.g. a
    negative SQLite ``LIMIT`` meaning "unlimited", or a zero "consolidate
    every N" meaning "consolidate on every add" (M-59).
    """
    raw = os.environ.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except (ValueError, TypeError):
            value = default
    if lo is not None and value < lo:
        value = lo
    if hi is not None and value > hi:
        value = hi
    return value


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
    except PermissionError:
        # Process exists but is owned by another user.
        return True
    except OSError:
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
