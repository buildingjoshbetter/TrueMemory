"""Shared utilities for TrueMemory hooks."""

from pathlib import Path
import json
import logging
import os
import time

# fcntl is POSIX-only — Windows has no equivalent file-range locking primitive.
# Without this guard the entire module fails to import on Windows, which in
# turn breaks every Claude Code hook that imports from here (session_start,
# user_prompt_submit, stop). The Windows code path falls back to non-atomic
# read/write for the extraction budget; the worst-case race window allows
# 1-2 extra extractions per hour, which is acceptable given the alternative
# is no enforcement at all.
try:
    import fcntl  # type: ignore[import-not-found]
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover — Windows path
    _HAS_FCNTL = False

log = logging.getLogger(__name__)

EXTRACTED_DIR = Path.home() / ".truememory" / "extracted"
TURN_INJECTED_DIR = Path.home() / ".truememory" / "turn_injected"

_BUDGET_FILE = Path.home() / ".truememory" / ".extraction_budget"
_MAX_EXTRACTIONS_PER_HOUR = int(os.environ.get("TRUEMEMORY_MAX_EXTRACTIONS_PER_HOUR", "20"))

_STALE_PROCESSING_THRESHOLD = 1800  # 30 minutes


def _safe_session_id(session_id: str) -> str:
    """Sanitize session_id to prevent path traversal."""
    return "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]


def is_subagent_invocation(input_data: dict | None = None, transcript_path: str = "") -> bool:
    """Detect whether this hook fire is from a Claude Code sub-agent (Task tool).

    Hooks fire per-session, and sub-agents are separate sessions from the
    orchestrator. Without a guard, every sub-agent spawn produces 1-3 hook
    fires, which floods the user with terminal flashes and pollutes the
    memory store with orchestrator-generated prompts (not real user input).

    Two detection paths, either is sufficient:
    1. ``input_data["agent_id"]`` or ``input_data["agent_type"]`` is present
       (newer Claude Code versions include these fields in hook stdin JSON).
    2. The transcript path contains ``/subagents/`` — Claude Code stores
       sub-agent transcripts under
       ``<project>/<session_id>/subagents/agent-<id>.jsonl``. Robust fallback
       for older Claude Code versions that don't emit the agent_id field.
    """
    if isinstance(input_data, dict):
        if input_data.get("agent_id") or input_data.get("agent_type"):
            return True
        tp = transcript_path or input_data.get("transcript_path", "")
    else:
        tp = transcript_path
    if not tp:
        return False
    return "/subagents/" in tp.replace("\\", "/")


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
    """Check if a PID is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def check_extraction_budget() -> bool:
    """Check if the hourly extraction budget allows another extraction.

    Returns True if extraction is allowed, False if budget is exhausted.

    On POSIX, uses ``fcntl.flock`` for atomic read-modify-write — the only
    safe way to coordinate this across the concurrent ingest processes the
    backlog drainer spawns. On Windows (no ``fcntl``), falls back to a
    non-atomic read-modify-write with a narrow race window: two processes
    could both pass the check before either writes, briefly exceeding the
    cap by 1-2 extractions per hour. This is acceptable given the
    alternative is either no enforcement or a Windows-specific named-mutex
    implementation whose complexity dwarfs the rare overshoot.
    """
    if _MAX_EXTRACTIONS_PER_HOUR <= 0:
        return True
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
            os.close(fd)
            return False
        data["count"] += 1
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(data).encode("utf-8"))
        os.close(fd)
        return True
    except OSError:
        return True


def record_stale_processing_pid(processing_path: Path, pid: int) -> None:
    """Write the spawned PID into a .processing file for liveness checks."""
    try:
        data = json.loads(processing_path.read_text(encoding="utf-8"))
        data["claimed_pid"] = pid
        processing_path.write_text(json.dumps(data), encoding="utf-8")
    except (OSError, json.JSONDecodeError):
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
        marker.write_text(json.dumps({
            "size": current_size,
            "timestamp": time.time(),
            "pid": spawned_pid or os.getpid(),
        }), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Turn-based memory injection (UserPromptSubmit) — per-session one-shot dedup
# ---------------------------------------------------------------------------


def count_user_turns(transcript_path: str) -> int:
    """Count user turns in a Claude Code transcript file.

    Mirrors the two-path parsing of ``stop._has_enough_messages`` but
    returns the count instead of a bool gate. The "user" role is anything
    with ``type``/``role`` set to ``"human"`` or ``"user"`` — string
    substring counting is avoided so transcripts that mention the literal
    words ``"human"``/``"user"`` in content don't inflate the total.
    """
    try:
        content = Path(transcript_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0

    if not content.strip():
        return 0

    # Try JSON array first (Claude Code may serialize as one array)
    try:
        if content.lstrip().startswith("["):
            data = json.loads(content)
            if isinstance(data, list):
                count = 0
                for entry in data:
                    if isinstance(entry, dict):
                        role = entry.get("type") or entry.get("role") or ""
                        if role in ("human", "user"):
                            count += 1
                return count
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL (one JSON object per line — the common Claude Code shape)
    count = 0
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            role = entry.get("type") or entry.get("role") or ""
            if role in ("human", "user"):
                count += 1
    return count


def _turn_injected_marker(session_id: str) -> Path:
    safe_id = _safe_session_id(session_id)
    return TURN_INJECTED_DIR / safe_id


def already_injected(session_id: str) -> bool:
    """Return True iff turn-based injection has already fired for this session."""
    if not session_id or session_id == "unknown":
        return False
    safe_id = _safe_session_id(session_id)
    if not safe_id:
        return False
    return _turn_injected_marker(session_id).exists()


def mark_injected(session_id: str, meta: dict) -> None:
    """Atomically record that turn-based injection has fired for this session.

    Writes via tempfile + rename so partial writes can't corrupt the marker.
    The ``meta`` dict records the trigger reason, query size, and result
    count — useful for debugging via ``cat ~/.truememory/turn_injected/<id>``.
    """
    if not session_id or session_id == "unknown":
        return
    safe_id = _safe_session_id(session_id)
    if not safe_id:
        return
    try:
        TURN_INJECTED_DIR.mkdir(parents=True, exist_ok=True)
        marker = _turn_injected_marker(session_id)
        tmp = marker.with_suffix(".tmp")
        payload = {**meta, "timestamp": time.time()}
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(marker)
    except OSError:
        pass
