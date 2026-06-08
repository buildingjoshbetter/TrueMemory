"""Shared utilities for TrueMemory hooks."""

from pathlib import Path
import errno
import json
import logging
import os
import time

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

log = logging.getLogger(__name__)

EXTRACTED_DIR = Path.home() / ".truememory" / "extracted"
BACKLOG_DIR = Path.home() / ".truememory" / "backlog"

_BUDGET_FILE = Path.home() / ".truememory" / ".extraction_budget"
_MAX_EXTRACTIONS_PER_HOUR = int(os.environ.get("TRUEMEMORY_MAX_EXTRACTIONS_PER_HOUR", "20"))

_STALE_PROCESSING_THRESHOLD = 1800  # 30 minutes


def _safe_session_id(session_id: str) -> str:
    """Sanitize session_id to prevent path traversal."""
    return "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]


def should_extract_session(session_id: str, transcript_path: str) -> bool:
    """Check if a session's transcript has new content since last extraction.

    Compares the current transcript file size against the size recorded
    at last extraction. Only returns True if the file grew by >1KB
    (enough for at least a few new messages, avoids re-extracting for
    minor metadata appends).

    Returns True if:
    - No prior extraction marker exists (first time)
    - Transcript grew by >1024 bytes since last extraction
    - Marker is corrupted/unreadable (extract to be safe)
    """
    if not session_id or session_id == "unknown":
        return True
    if not transcript_path or not transcript_path.strip():
        return True

    safe_id = _safe_session_id(session_id)
    if not safe_id:
        return True

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    marker = EXTRACTED_DIR / safe_id

    if not marker.exists():
        return True

    try:
        current_size = Path(transcript_path).stat().st_size
    except OSError:
        return True

    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        last_size = data.get("size", 0)
    except (json.JSONDecodeError, OSError, ValueError):
        return True

    # If the process that wrote this marker is still alive, an extraction
    # is already running for this session — skip to avoid piling up
    # concurrent extractions of the same actively-growing transcript.
    marker_pid = data.get("pid", 0)
    if marker_pid and _pid_is_alive(marker_pid):
        return False

    if current_size < last_size:
        return True
    return (current_size - last_size) > 1024


def _pid_is_alive(pid: int) -> bool:
    """Check if a PID is still running.

    Distinguishes "no such process" (dead) from "permission denied" (alive
    but owned by another user). ``os.kill(pid, 0)`` raises ``ProcessLookupError``
    (errno ESRCH) only when the process genuinely does not exist; an
    ``EPERM`` error means the process *is* alive but we lack permission to
    signal it, so it must be treated as alive — otherwise a live-but-EPERM
    worker's claim would be wrongly reclaimed/re-queued.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        # errno.ESRCH — no such process: genuinely dead.
        return False
    except PermissionError:
        # errno.EPERM — process exists but is not signalable by us: alive.
        return True
    except OSError as e:
        # ESRCH may surface as a bare OSError on some platforms; treat only
        # that as dead, anything else (e.g. EPERM) as alive to be safe.
        return e.errno != errno.ESRCH


def check_extraction_budget() -> bool:
    """Check if the hourly extraction budget allows another extraction.

    Returns True if extraction is allowed, False if budget is exhausted.
    Uses flock for atomicity across concurrent processes (no-op on Windows).
    """
    if _MAX_EXTRACTIONS_PER_HOUR <= 0:
        return True
    fd = -1
    try:
        _BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(_BUDGET_FILE), os.O_RDWR | os.O_CREAT)
        if _HAS_FCNTL:
            fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            raw = os.read(fd, 4096).decode("utf-8", errors="replace").strip()
            data = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, ValueError):
            data = {}
        current_hour = int(time.time() // 3600)
        if data.get("hour") != current_hour:
            data = {"hour": current_hour, "count": 0}
        if data["count"] >= _MAX_EXTRACTIONS_PER_HOUR:
            return False
        data["count"] += 1
        payload = json.dumps(data).encode("utf-8")
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, payload)
        return True
    except OSError:
        log.warning("Budget file I/O error — allowing extraction", exc_info=True)
        return True
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def record_stale_processing_pid(processing_path: Path, pid: int) -> None:
    """Write the spawned PID into a .processing file for liveness checks."""
    try:
        data = json.loads(processing_path.read_text(encoding="utf-8"))
        data["claimed_pid"] = pid
        processing_path.write_text(json.dumps(data), encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        pass


def clear_backlog_processing(session_id: str, backlog_dir: Path | None = None) -> bool:
    """Remove a backlog ``.processing`` claim marker on confirmed success.

    Called by the ingest CLI once a session has been ingested successfully so
    that the claim marker for that session is deleted. If the worker instead
    crashes / exits non-zero, this is NOT called, the ``.processing`` marker is
    left in place, and ``cleanup_stale_processing`` later restores it to
    ``.json`` (once the claiming PID is dead and the 30-minute threshold has
    elapsed) so the session is re-queued rather than silently lost.

    The drainers (``session_start._drain_backlog`` and ``cli._cascade_next``)
    name claim markers ``{sanitized_session_id}.processing``, mirroring how
    ``stop._queue_to_backlog`` names ``{sanitized_session_id}.json``. We
    reconstruct that path from ``session_id`` here.

    Returns True if a marker was removed, False otherwise (including when no
    marker existed, which is the common case for non-backlog ingests spawned
    directly by the Stop hook).
    """
    if not session_id or session_id == "unknown":
        return False
    safe_id = _safe_session_id(session_id)
    if not safe_id:
        return False
    target = (backlog_dir or BACKLOG_DIR) / f"{safe_id}.processing"
    try:
        target.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def cleanup_stale_processing(backlog_dir: Path) -> None:
    """Restore .processing files whose owning process has died.

    Uses a 30-minute threshold AND PID liveness check. Only restores
    if the file is old enough AND the claiming process is no longer running.
    """
    for stale in backlog_dir.glob("*.processing"):
        try:
            age = time.time() - stale.stat().st_mtime
            if age <= _STALE_PROCESSING_THRESHOLD:
                continue
            try:
                data = json.loads(stale.read_text(encoding="utf-8"))
                pid = data.get("claimed_pid", 0)
                if pid and _pid_is_alive(pid):
                    continue
            except (json.JSONDecodeError, OSError):
                pass
            stale.rename(stale.with_suffix(".json"))
        except OSError:
            pass


def mark_session_extracted(session_id: str, transcript_path: str, spawned_pid: int = 0) -> None:
    """Record that a session's transcript was extracted at its current size.

    Written by the ingest CLI on successful completion, and also by
    triggers before spawning (optimistic) to prevent concurrent duplicates.
    The CLI write is authoritative; the trigger write is best-effort.

    Args:
        spawned_pid: The PID of the background ingest process. When called
            from a hook, pass the Popen PID so the liveness check in
            should_extract_session() can detect a running extraction.
            When called from the CLI itself, defaults to os.getpid().
    """
    if not session_id or session_id == "unknown":
        return
    if not transcript_path or not transcript_path.strip():
        return

    safe_id = _safe_session_id(session_id)
    if not safe_id:
        return

    try:
        EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
        current_size = Path(transcript_path).stat().st_size
        marker = EXTRACTED_DIR / safe_id
        marker.write_text(json.dumps({
            "size": current_size,
            "timestamp": time.time(),
            "pid": spawned_pid or os.getpid(),
        }), encoding="utf-8")
    except OSError:
        pass
