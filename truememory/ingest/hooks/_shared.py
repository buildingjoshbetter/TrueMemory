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
RECALL_MARKER_DIR = Path.home() / ".truememory" / "recall_markers"
RECALL_CACHE_PATH = Path.home() / ".truememory" / "recall_cache.json"

# How long cached recall results remain valid (seconds).  Default 5 min.
# Set TRUEMEMORY_RECALL_CACHE_TTL=0 to disable caching entirely.
try:
    RECALL_CACHE_TTL = float(os.environ.get("TRUEMEMORY_RECALL_CACHE_TTL", "300"))
except ValueError:
    RECALL_CACHE_TTL = 300.0

_BUDGET_FILE = Path.home() / ".truememory" / ".extraction_budget"
_MAX_EXTRACTIONS_PER_HOUR = int(os.environ.get("TRUEMEMORY_MAX_EXTRACTIONS_PER_HOUR", "20"))

_STALE_PROCESSING_THRESHOLD = 1800  # 30 minutes

# How long after SessionStart injects recall the first user prompt is treated
# as redundant. SessionStart writes the marker just before the first prompt
# arrives, so a short window is enough to cover that first message (issue #561).
# Defensive parse: a typo in this user-facing knob must not crash every hook
# at import time — fall back to the default instead.
try:
    _RECALL_DEBOUNCE_SECONDS = float(os.environ.get("TRUEMEMORY_RECALL_DEBOUNCE_SECONDS", "60"))
except ValueError:
    _RECALL_DEBOUNCE_SECONDS = 60.0


def get_recall_deadline() -> float | None:
    """Per-request model-server deadline for hook recall searches (#577).

    Hooks block on the shared model server; under contention (batch
    ingestion, MPS OOM recovery) a single embed could previously stall for
    the full 120s client timeout. Recall paths arm this short deadline via
    ``model_client.set_request_timeout`` so embeds fast-fail and the
    engine's FTS-only fallback actually triggers.

    Configured by ``TRUEMEMORY_HOOK_RECALL_TIMEOUT`` (seconds, default 5).
    ``0`` or negative disables the deadline (legacy 120s behavior); a
    malformed value falls back to the default instead of crashing the hook.
    """
    try:
        deadline = float(os.environ.get("TRUEMEMORY_HOOK_RECALL_TIMEOUT", "5"))
    except ValueError:
        return 5.0
    if deadline <= 0:
        return None
    return deadline


def _safe_session_id(session_id: str) -> str:
    """Sanitize session_id to prevent path traversal."""
    return "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via tmp file + os.replace.

    A reader can never observe a torn / partially-written marker: it sees
    either the old contents or the fully-written new contents. The tmp file
    is unique per-process so concurrent writers do not clobber each other's
    temp file before the rename (issue #644 / M-14).
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


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


def refund_extraction_budget() -> None:
    """Return one consumed extraction slot to the hourly budget.

    Callers consume a slot with ``check_extraction_budget()`` *before* the
    spawn gate; when the gate denies the spawn no extraction actually runs,
    so the slot must be returned or a denied spawn permanently burns budget
    (issue #644 / M-71). No-op when budget tracking is disabled. Only
    decrements within the same hour the slot was consumed.
    """
    if _MAX_EXTRACTIONS_PER_HOUR <= 0:
        return
    fd = -1
    try:
        if not _BUDGET_FILE.exists():
            return
        fd = os.open(str(_BUDGET_FILE), os.O_RDWR)
        if _HAS_FCNTL:
            fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            raw = os.read(fd, 4096).decode("utf-8", errors="replace").strip()
            data = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, ValueError):
            return
        current_hour = int(time.time() // 3600)
        # Only refund if we're still in the hour the slot was consumed —
        # a rollover already reset the counter, nothing to give back.
        if data.get("hour") != current_hour:
            return
        if data.get("count", 0) <= 0:
            return
        data["count"] -= 1
        payload = json.dumps(data).encode("utf-8")
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, payload)
    except OSError:
        pass
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
        _atomic_write_text(processing_path, json.dumps(data))
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


def _quarantine_marker(path: Path) -> None:
    """Move a corrupt backlog marker aside to ``.corrupt`` (issue #644 / M-14).

    A marker whose JSON fails to parse must never be recycled back to
    ``.json`` — the drainers would re-claim it every session, and a few such
    poison pills permanently consume every ``_DRAIN_CAP`` slot so healthy
    sessions are never extracted. Renaming to ``.corrupt`` (which the
    ``*.json`` glob ignores) takes it out of rotation while preserving it for
    forensics. Best-effort: on rename failure, unlink so it cannot poison.
    """
    target = path.with_suffix(".corrupt")
    try:
        os.replace(path, target)
        log.warning("Quarantined corrupt backlog marker: %s -> %s", path.name, target.name)
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass


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
        _atomic_write_text(marker, json.dumps({
            "size": current_size,
            "timestamp": time.time(),
            "pid": spawned_pid or os.getpid(),
        }))
    except OSError:
        pass


def mark_recall_injected(session_id: str) -> None:
    """Record that SessionStart injected recall for this session.

    Lets UserPromptSubmit skip its redundant per-message auto-recall on the
    first prompt of the session (issue #561). Best-effort: never raises.
    """
    safe_id = _safe_session_id(session_id)
    if not safe_id:
        return
    try:
        RECALL_MARKER_DIR.mkdir(parents=True, exist_ok=True)
        # Opportunistic sweep: markers from sessions that never sent a prompt
        # are never consumed; remove anything well past the debounce window so
        # the dir cannot grow unboundedly (mirrors _prune_old_buffers).
        cutoff = time.time() - max(_RECALL_DEBOUNCE_SECONDS * 10, 3600.0)
        for stale in RECALL_MARKER_DIR.iterdir():
            try:
                if stale.stat().st_mtime < cutoff:
                    stale.unlink()
            except OSError:
                continue
        # Wall clock (not monotonic) on purpose: the timestamp is compared
        # across the SessionStart and UserPromptSubmit processes.
        (RECALL_MARKER_DIR / safe_id).write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass


def get_recall_cache(db_path: str, user_id: str = "") -> str | None:
    """Return cached recall context if it exists and is within the TTL.

    Returns the cached context string, or None if the cache is missing,
    stale, or disabled. Cache key is (db_path, user_id) so multi-DB and
    multi-user setups get separate caches.
    """
    if RECALL_CACHE_TTL <= 0:
        return None
    try:
        if not RECALL_CACHE_PATH.exists():
            return None
        data = json.loads(RECALL_CACHE_PATH.read_text(encoding="utf-8"))
        key = _recall_cache_key(db_path, user_id)
        entry = data.get(key)
        if entry is None:
            return None
        ts = entry.get("timestamp", 0)
        if (time.time() - ts) >= RECALL_CACHE_TTL:
            return None
        return entry.get("context")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
        return None


def set_recall_cache(context: str, db_path: str, user_id: str = "") -> None:
    """Write recall results to the cache file with a timestamp.

    Preserves entries for other (db_path, user_id) combinations so
    multi-DB setups coexist in a single cache file.
    """
    if RECALL_CACHE_TTL <= 0:
        return
    try:
        RECALL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Read existing entries (best-effort)
        existing: dict = {}
        try:
            if RECALL_CACHE_PATH.exists():
                existing = json.loads(RECALL_CACHE_PATH.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    existing = {}
        except (OSError, json.JSONDecodeError):
            existing = {}
        key = _recall_cache_key(db_path, user_id)
        existing[key] = {
            "timestamp": time.time(),
            "context": context,
        }
        tmp = RECALL_CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing), encoding="utf-8")
        os.replace(tmp, RECALL_CACHE_PATH)
    except OSError:
        pass


def invalidate_recall_cache(db_path: str = "", user_id: str = "") -> None:
    """Delete the recall cache, called when new memories are stored.

    If db_path is provided, only the matching entry is removed; otherwise
    the entire cache file is deleted (safest default).
    """
    try:
        if not RECALL_CACHE_PATH.exists():
            return
        if not db_path:
            RECALL_CACHE_PATH.unlink(missing_ok=True)
            return
        data = json.loads(RECALL_CACHE_PATH.read_text(encoding="utf-8"))
        key = _recall_cache_key(db_path, user_id)
        if key in data:
            del data[key]
            if data:
                tmp = RECALL_CACHE_PATH.with_suffix(".tmp")
                tmp.write_text(json.dumps(data), encoding="utf-8")
                os.replace(tmp, RECALL_CACHE_PATH)
            else:
                RECALL_CACHE_PATH.unlink(missing_ok=True)
    except (OSError, json.JSONDecodeError, TypeError):
        # Best-effort: nuke the file on any error
        try:
            RECALL_CACHE_PATH.unlink(missing_ok=True)
        except OSError:
            pass


def _recall_cache_key(db_path: str, user_id: str = "") -> str:
    """Deterministic cache key from db_path and user_id.

    Normalizes db_path via PurePosixPath so the same logical path produces
    the same key on both Unix (``/a.db``) and Windows (``\\a.db``).
    """
    normalized = str(Path(db_path).as_posix()) if db_path else "default"
    return f"{normalized}:{user_id or ''}"


def consume_recall_injected(session_id: str, within_seconds: float = _RECALL_DEBOUNCE_SECONDS) -> bool:
    """Return True if SessionStart recall ran for this session recently.

    One-shot: the marker is removed on read (whether fresh or stale), so only
    the first user prompt after SessionStart is debounced and the marker dir
    self-cleans for any session that sends at least one prompt. Returns True
    only when a marker existed and is younger than ``within_seconds``.
    """
    safe_id = _safe_session_id(session_id)
    if not safe_id:
        return False
    marker = RECALL_MARKER_DIR / safe_id
    try:
        raw = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    try:
        marker.unlink()
    except OSError:
        pass
    if within_seconds <= 0:
        return False
    try:
        ts = float(raw)
    except ValueError:
        return False
    return (time.time() - ts) < within_seconds
