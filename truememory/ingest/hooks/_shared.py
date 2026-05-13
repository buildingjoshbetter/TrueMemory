"""Shared utilities for TrueMemory hooks."""

from pathlib import Path
import json
import os
import time

MARKER_PATH = Path.home() / ".truememory" / "last_incremental_extraction"
DEFAULT_INTERVAL = 14400  # 4 hours in seconds

EXTRACTED_DIR = Path.home() / ".truememory" / "extracted"


def should_extract(interval: int = DEFAULT_INTERVAL) -> bool:
    """Check if enough time has elapsed since the last incremental extraction."""
    if not MARKER_PATH.exists():
        return True
    try:
        return (time.time() - MARKER_PATH.stat().st_mtime) >= interval
    except OSError:
        return True


def mark_extracted():
    """Update the timestamp marker after a successful extraction trigger."""
    try:
        MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        MARKER_PATH.write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass


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

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    marker = EXTRACTED_DIR / session_id

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

    return (current_size - last_size) > 1024


def mark_session_extracted(session_id: str, transcript_path: str) -> None:
    """Record that a session's transcript was extracted at its current size.

    Written by the ingest CLI on successful completion, and also by
    triggers before spawning (optimistic) to prevent concurrent duplicates.
    The CLI write is authoritative; the trigger write is best-effort.
    """
    if not session_id or session_id == "unknown":
        return

    try:
        EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
        current_size = Path(transcript_path).stat().st_size
        marker = EXTRACTED_DIR / session_id
        marker.write_text(json.dumps({
            "size": current_size,
            "timestamp": time.time(),
            "pid": os.getpid(),
        }), encoding="utf-8")
    except OSError:
        pass
