"""Shared utilities for TrueMemory hooks."""

from pathlib import Path
import time

MARKER_PATH = Path.home() / ".truememory" / "last_incremental_extraction"
DEFAULT_INTERVAL = 14400  # 4 hours in seconds


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
