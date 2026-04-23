#!/usr/bin/env python3
"""
Stop Hook — Trigger Background Extraction
===========================================

Fires when a Claude Code session ends. Reads the conversation transcript
and launches the ingestion pipeline to extract and store memories.

This is the "sleep consolidation" trigger — the conversation is over,
now process it for long-term storage. The pipeline runs in a background
subprocess so it doesn't block Claude Code's shutdown.

Input (stdin JSON):
    {"session_id": "...", "transcript_path": "...", "stop_reason": "..."}

Output: None (processing happens in background)
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

GATE_THRESHOLD = float(os.environ.get("TRUEMEMORY_GATE_THRESHOLD", "0.30"))
MIN_MESSAGES = int(os.environ.get("TRUEMEMORY_MIN_MESSAGES", "5"))
TRACE_DIR = Path(os.environ.get(
    "TRUEMEMORY_TRACE_DIR",
    str(Path.home() / ".truememory" / "traces"),
))
LOG_DIR = Path(os.environ.get(
    "TRUEMEMORY_LOG_DIR",
    str(Path.home() / ".truememory" / "logs"),
))
# Hunter F33: when Popen fails we queue the ingestion here instead of
# running inline (which would block Claude Code shutdown).
BACKLOG_DIR = Path(os.environ.get(
    "TRUEMEMORY_BACKLOG_DIR",
    str(Path.home() / ".truememory" / "backlog"),
))
# Hunter F21: cap concurrent ingest processes. N parallel Stop hooks
# (multi-session close, session-restart loop) would otherwise load N
# embedding models at once — ~600MB RSS each on Pro, easy OOM on laptops.
# Tunable via env var; POSIX-only via pgrep (on Windows the cap is a no-op).
SPAWN_CAP = int(os.environ.get("TRUEMEMORY_INGEST_SPAWN_CAP", "2"))


def _parse_args() -> argparse.Namespace:
    """Parse command-line overrides for user_id and db_path.

    Resolution order is: command-line arg > env var > empty default. The
    installer threads these flags through so multiple Claude Code profiles
    can share a single interpreter while still writing to separate DBs.
    ``parse_known_args`` is used so the hook tolerates unknown flags from
    future installer versions.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--user", default=os.environ.get("TRUEMEMORY_USER_ID", ""))
    p.add_argument("--db", default=os.environ.get("TRUEMEMORY_DB_PATH", ""))
    args, _ = p.parse_known_args()
    return args


def main():
    args = _parse_args()

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        input_data = {}

    transcript_path = input_data.get("transcript_path", "")
    session_id = input_data.get("session_id", "unknown")

    if not transcript_path or not Path(transcript_path).exists():
        return

    # Pre-flight check: ensure our write directories exist and are writable
    # so we fail early with a clear message instead of silently losing work
    if not _writable_dirs_ok():
        return

    # Proper transcript validation: parse and count actual message objects
    # instead of substring-matching (which false-positives on content that
    # contains the literal strings "human" or "user").
    if not _has_enough_messages(transcript_path, MIN_MESSAGES):
        return

    # Run ingestion in the background so we don't block Claude Code
    _run_background_ingestion(transcript_path, session_id, args.user, args.db)


def _writable_dirs_ok() -> bool:
    """Verify the trace and log directories exist and are writable.

    Writes a short marker to ~/.truememory/.health so that a user who is
    debugging knows the hook at least fired and checked its environment.
    Returns False (and prints to stderr) if the directories can't be used.
    """
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"truememory-ingest stop hook: cannot create ~/.truememory dirs: {e}",
              file=sys.stderr)
        return False

    # Check disk free space — abort if less than 10 MB available
    try:
        stats = shutil.disk_usage(LOG_DIR)
        if stats.free < 10 * 1024 * 1024:
            print(f"truememory-ingest stop hook: disk full (free={stats.free} bytes)",
                  file=sys.stderr)
            return False
    except OSError:
        pass  # Non-fatal if disk_usage fails

    # Test write access with a touch file
    try:
        health_file = LOG_DIR / ".health"
        health_file.write_text("ok", encoding="utf-8")
    except OSError as e:
        print(f"truememory-ingest stop hook: logs dir not writable: {e}",
              file=sys.stderr)
        return False

    return True


def _has_enough_messages(transcript_path: str, min_messages: int) -> bool:
    """Check whether the transcript has at least `min_messages` user turns.

    Parses the transcript properly instead of substring-counting so that
    conversations containing the literal strings "human"/"user" in content
    don't inflate the count.
    """
    try:
        content = Path(transcript_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False

    if not content.strip():
        return False

    # Try JSON array first (Claude Code format)
    count = 0
    try:
        if content.lstrip().startswith("["):
            data = json.loads(content)
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        role = entry.get("type") or entry.get("role") or ""
                        if role in ("human", "user"):
                            count += 1
                return count >= min_messages
    except json.JSONDecodeError:
        pass

    # Try JSONL
    try:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    role = entry.get("type") or entry.get("role") or ""
                    if role in ("human", "user"):
                        count += 1
            except json.JSONDecodeError:
                continue
        if count > 0:
            return count >= min_messages
    except Exception:
        pass

    # Fall back to length heuristic for plain text
    return len(content) > min_messages * 50


def _count_active_ingest_processes() -> int:
    """Return a best-effort count of live ``truememory.ingest.cli`` children.

    Hunter F21: used to bound concurrent hook-spawned ingestions. pgrep
    is POSIX-only; on Windows this returns 0 (cap disabled) to avoid a
    platform-dependent hard stop — the typical burst-close scenario is
    rare on Windows anyway. Any pgrep failure is swallowed (return 0)
    so the cap never becomes a hard fence that prevents ingestion.
    """
    if sys.platform == "win32":
        return 0
    try:
        result = subprocess.run(
            ["pgrep", "-f", "truememory.ingest.cli"],
            capture_output=True, text=True, timeout=1,
        )
        return len([ln for ln in (result.stdout or "").splitlines() if ln.strip()])
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return 0


def _queue_to_backlog(
    transcript_path: str,
    session_id: str,
    user_id: str,
    db_path: str,
    reason: str,
) -> None:
    """Drop a queue marker so a later session can re-attempt this ingestion.

    Hunter F33: when Popen fails (disk full for log file, permission error,
    etc.) we used to fall back to synchronous inline ingestion — which
    blocked Claude Code's shutdown for 10–60s. Now we write a marker to
    BACKLOG_DIR and let the next session's startup hook drain it (drain
    path is a follow-up item).
    """
    try:
        from datetime import datetime, timezone
        BACKLOG_DIR.mkdir(parents=True, exist_ok=True)
        marker = BACKLOG_DIR / f"{session_id or 'unknown'}.json"
        marker.write_text(json.dumps({
            "transcript_path": transcript_path,
            "session_id": session_id,
            "user_id": user_id,
            "db_path": db_path,
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }))
    except Exception as e:
        # Best-effort: if we can't write the backlog marker, the session's
        # memories are lost — log and move on. Must not raise.
        log.error("stop hook: failed to queue backlog marker: %s", e)


def _run_background_ingestion(
    transcript_path: str,
    session_id: str,
    user_id: str,
    db_path: str,
):
    """Launch the ingestion pipeline as a background process.

    Captures stderr/stdout to a log file so silent failures are recoverable.
    Handles Windows (CREATE_NEW_PROCESS_GROUP) and POSIX (start_new_session)
    subprocess detachment.

    Hunter F21: bounds concurrent spawns via ``SPAWN_CAP`` so a burst of
    Stop hooks doesn't load N embedding models at once.
    Hunter F33: on Popen failure, queue the ingestion to ``BACKLOG_DIR``
    for a later session to re-attempt — NEVER fall back to synchronous
    inline ingestion (that blocks Claude Code's shutdown).
    """
    # Log the effective config for debugging. Operators commonly wire up
    # multiple profiles and need to confirm which user/db the hook actually
    # saw after the arg-parse + env-var resolution.
    log.info(
        "stop hook: launching ingestion user=%r db=%r session=%r",
        user_id, db_path, session_id,
    )

    # Build the command
    cmd = [
        sys.executable, "-m", "truememory.ingest.cli",
        "ingest", transcript_path,
    ]

    if user_id:
        cmd.extend(["--user", user_id])
    if db_path:
        cmd.extend(["--db", db_path])

    cmd.extend(["--threshold", str(GATE_THRESHOLD)])
    if session_id:
        cmd.extend(["--session", session_id])

    # Save trace for debugging
    trace_path = TRACE_DIR / f"{session_id}.json"
    log_path = LOG_DIR / f"{session_id}.log"
    cmd.extend(["--trace", str(trace_path)])

    # OS-specific subprocess detachment kwargs
    # POSIX: start_new_session to detach from parent's process group
    # Windows: CREATE_NEW_PROCESS_GROUP flag
    detach_kwargs: dict = {}
    if sys.platform == "win32":
        detach_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        detach_kwargs["start_new_session"] = True

    # Hunter F21: refuse to spawn if we're already at the concurrent-ingest
    # cap. Over-cap events queue to the backlog so the memories aren't lost.
    active = _count_active_ingest_processes()
    if active >= SPAWN_CAP:
        log.warning(
            "stop hook: at spawn cap (%d active / cap %d); queueing session "
            "%r to backlog for later",
            active, SPAWN_CAP, session_id,
        )
        _queue_to_backlog(
            transcript_path, session_id, user_id, db_path,
            reason=f"spawn_cap_reached:{active}>=SPAWN_CAP={SPAWN_CAP}",
        )
        return

    log_file = None
    try:
        # Open the log file and hand it to the subprocess. We MUST close our
        # parent-side handle after Popen — the subprocess has its own dup'd
        # copy, so closing ours here is safe and prevents FD leaks across
        # many invocations of this hook (one per session).
        log_file = open(log_path, "a", encoding="utf-8")
        subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            **detach_kwargs,
        )
    except Exception as e:
        # Hunter F33: NEVER fall back to synchronous inline ingestion —
        # that blocks Claude Code's shutdown for 10–60s. Queue instead;
        # a later session's drain path will re-attempt.
        log.warning(
            "stop hook: background launch failed (%s); queueing session "
            "%r to backlog for later",
            e, session_id,
        )
        _queue_to_backlog(
            transcript_path, session_id, user_id, db_path,
            reason=f"popen_failed:{type(e).__name__}:{e}",
        )
    finally:
        # Always close our parent-side handle to prevent FD leaks.
        # The subprocess still has its own dup'd copy of the FD and will
        # continue writing to the log file normally.
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
