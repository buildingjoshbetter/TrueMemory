#!/usr/bin/env python3
"""
UserPromptSubmit Hook — Lightweight Message Buffer
===================================================

Fires on every user message submission. Appends a one-line JSON record
to a per-session buffer so debugging tools can see what the user said
even if the transcript is corrupted or truncated.

Design notes:
- The Stop hook reads `transcript_path` directly, not the buffer, so
  this is defensive / diagnostic rather than load-bearing.
- Uses `fcntl.flock` to make concurrent writes from overlapping sessions
  safe (previously could interleave).
- Automatically prunes buffer files older than 7 days on each invocation
  so they don't grow unbounded.

Input (stdin JSON):
    {"session_id": "...", "prompt": "...", "transcript_path": "..."}

Output: None (silent hook, no additionalContext)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Optional: fcntl isn't available on Windows, so we gracefully degrade
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


# Buffer location
BUFFER_DIR = Path(os.environ.get(
    "TRUEMEMORY_BUFFER_DIR",
    str(Path.home() / ".truememory" / "buffers"),
))

# Delete buffer files older than this many days
RETENTION_DAYS = int(os.environ.get("TRUEMEMORY_BUFFER_RETENTION_DAYS", "7"))
# Max size per buffer file (bytes) before we rotate
MAX_BUFFER_SIZE = int(os.environ.get("TRUEMEMORY_BUFFER_MAX_BYTES", str(10 * 1024 * 1024)))


def _parse_args() -> argparse.Namespace:
    """Parse command-line overrides the installer threads through.

    UserPromptSubmit doesn't actually use ``--user`` or ``--db`` — it only
    writes a per-session diagnostic buffer — but the installer passes the
    same flags to every hook for consistency, so we must accept them here
    without erroring out. ``parse_known_args`` ensures forward compat with
    future flags.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--user", default=os.environ.get("TRUEMEMORY_USER_ID", ""))
    p.add_argument("--db", default=os.environ.get("TRUEMEMORY_DB_PATH", ""))
    args, _ = p.parse_known_args()
    return args


def main():
    _parse_args()  # accept installer-passed flags even though we don't use them

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return

    prompt = input_data.get("prompt", "").strip()
    session_id = input_data.get("session_id", "unknown")

    if not prompt or len(prompt) < 3:
        return

    try:
        buffer_message(session_id, prompt)
        _prune_old_buffers()
    except Exception:
        pass  # Never crash the hook


def buffer_message(session_id: str, prompt: str):
    """Append a user message to the session buffer file (with file locking)."""
    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    try:
        BUFFER_DIR.chmod(0o700)
    except OSError:
        pass

    # Sanitize session_id to prevent path traversal (e.g., "../../etc/passwd")
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
    if not safe_id:
        safe_id = "unknown"

    buffer_file = BUFFER_DIR / f"{safe_id}.jsonl"

    # Rotate if buffer has grown too large
    try:
        if buffer_file.exists() and buffer_file.stat().st_size > MAX_BUFFER_SIZE:
            rotated = buffer_file.with_suffix(f".{int(time.time())}.jsonl")
            buffer_file.rename(rotated)
    except OSError:
        pass

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": "user",
        "content": prompt,
    }

    # Append with file locking to prevent interleaved writes from concurrent sessions
    with open(buffer_file, "a", encoding="utf-8") as f:
        if _HAS_FCNTL:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(entry) + "\n")
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                # If locking fails, write anyway — single hook invocation
                f.write(json.dumps(entry) + "\n")
        else:
            f.write(json.dumps(entry) + "\n")


def _prune_old_buffers():
    """Delete buffer files older than RETENTION_DAYS."""
    if not BUFFER_DIR.exists():
        return
    cutoff = time.time() - (RETENTION_DAYS * 86400)
    for path in BUFFER_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


if __name__ == "__main__":
    main()
