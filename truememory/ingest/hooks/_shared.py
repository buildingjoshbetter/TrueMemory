"""Shared utilities for TrueMemory hooks."""

from pathlib import Path
import json
import os
import time

EXTRACTED_DIR = Path.home() / ".truememory" / "extracted"


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
    """Check if a PID is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def mark_session_extracted(session_id: str, transcript_path: str) -> None:
    """Record that a session's transcript was extracted at its current size.

    Written by the ingest CLI on successful completion, and also by
    triggers before spawning (optimistic) to prevent concurrent duplicates.
    The CLI write is authoritative; the trigger write is best-effort.
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
            "pid": os.getpid(),
        }), encoding="utf-8")
    except OSError:
        pass
