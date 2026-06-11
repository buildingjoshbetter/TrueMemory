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
import time
from pathlib import Path

from truememory._platform import _env_int

log = logging.getLogger(__name__)

def _safe_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


def _safe_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


GATE_THRESHOLD = _safe_float_env("TRUEMEMORY_GATE_THRESHOLD", 0.30)
MIN_MESSAGES = _safe_int_env("TRUEMEMORY_MIN_MESSAGES", 5)
TRACE_DIR = Path(os.environ.get(
    "TRUEMEMORY_TRACE_DIR",
    str(Path.home() / ".truememory" / "traces"),
))
LOG_DIR = Path(os.environ.get(
    "TRUEMEMORY_LOG_DIR",
    str(Path.home() / ".truememory" / "logs"),
))
# when Popen fails we queue the ingestion here instead of
# running inline (which would block Claude Code shutdown).
BACKLOG_DIR = Path(os.environ.get(
    "TRUEMEMORY_BACKLOG_DIR",
    str(Path.home() / ".truememory" / "backlog"),
))
# cap concurrent ingest processes. N parallel Stop hooks
# (multi-session close, session-restart loop) would otherwise load N
# embedding models at once — ~600MB RSS each on Pro, easy OOM on laptops.
# Unified env var across stop.py and hooks/core.py.
# Crash-safe + clamped (issue #639): a garbage value in either knob must not
# crash the Stop hook at import, and a zero/negative cap must not disable
# spawning entirely. Resolve the legacy alias first, then the primary.
_SPAWN_CAP_DEFAULT = _env_int("TRUEMEMORY_INGEST_SPAWN_CAP", 2, lo=1)
SPAWN_CAP = _env_int("TRUEMEMORY_SPAWN_CAP", _SPAWN_CAP_DEFAULT, lo=1)


def _sanitize_session_id(session_id: str) -> str:
    """Sanitize session_id to prevent path traversal."""
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
    return safe or "unknown"


def _prune_old_files(directory: Path, retention_days: int = 30) -> None:
    """Delete files in directory older than retention_days."""
    if not directory.exists():
        return
    cutoff = time.time() - (retention_days * 86400)
    for path in directory.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


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
    if os.environ.get("TRUEMEMORY_EXTRACTION"):
        try:
            input_data = json.load(sys.stdin)
        except (json.JSONDecodeError, EOFError):
            input_data = {}
        transcript_path = input_data.get("transcript_path", "")
        session_id = input_data.get("session_id", "")
        if transcript_path and session_id:
            from truememory.ingest.hooks._shared import mark_session_extracted
            try:
                mark_session_extracted(session_id, transcript_path)
            except Exception as exc:
                log.debug("stop hook: failed to mark extraction session %s: %s", session_id, exc)
        return

    args = _parse_args()

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        input_data = {}

    transcript_path = input_data.get("transcript_path", "")
    session_id = input_data.get("session_id", "unknown")

    if not transcript_path or not Path(transcript_path).exists():
        return
    # M-90: transcript_path comes from attacker-influenceable hook stdin.
    # parse_transcript's plaintext fallback would read ANY readable file into
    # the memory store, so confirm containment in an allowed transcripts root
    # (the compact hook already did this; stop/drain previously did not).
    from truememory.ingest.hooks._shared import is_allowed_transcript
    if not is_allowed_transcript(transcript_path):
        log.warning("stop hook: transcript_path %r outside allowed roots; skipping", transcript_path)
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

    from truememory.ingest.hooks._shared import should_extract_session, mark_session_extracted
    if not should_extract_session(session_id, transcript_path):
        log.info("stop hook: session %s already extracted at this size, skipping", session_id)
        return

    spawned_pid = _run_background_ingestion(transcript_path, session_id, args.user, args.db)

    # Only mark the session extracted when a real ingest process was spawned.
    # _run_background_ingestion returns 0 when the spawn cap denied it or Popen
    # failed — in those cases it has already queued the session to the backlog.
    # Marking it here would make should_extract_session() treat it as processed
    # (until the transcript grows >1KB), silently dropping the session if the
    # backlog drain never completes. The CLI marks it on successful completion.
    if spawned_pid > 0:
        try:
            mark_session_extracted(session_id, transcript_path, spawned_pid=spawned_pid)
        except Exception:
            pass


def _writable_dirs_ok() -> bool:
    """Verify the trace and log directories exist and are writable.

    Writes a short marker to ~/.truememory/.health so that a user who is
    debugging knows the hook at least fired and checked its environment.
    Returns False (and prints to stderr) if the directories can't be used.
    """
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        TRACE_DIR.chmod(0o700)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.chmod(0o700)
    except OSError as e:
        print(f"truememory-ingest stop hook: cannot create ~/.truememory dirs: {e}",
              file=sys.stderr)
        return False

    _prune_old_files(TRACE_DIR)
    _prune_old_files(LOG_DIR)
    _prune_old_files(BACKLOG_DIR)

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

    used to bound concurrent hook-spawned ingestions. pgrep
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

    when Popen fails (disk full for log file, permission error,
    etc.) we used to fall back to synchronous inline ingestion — which
    blocked Claude Code's shutdown for 10–60s. Now we write a marker to
    BACKLOG_DIR and let the next session's startup hook drain it (drain
    path is a follow-up item).
    """
    try:
        from datetime import datetime, timezone
        BACKLOG_DIR.mkdir(parents=True, exist_ok=True)
        BACKLOG_DIR.chmod(0o700)
        from truememory.ingest.hooks._shared import _atomic_write_text
        safe_sid = _sanitize_session_id(session_id)
        # M-15: never clobber a live worker's in-flight claim. If a
        # .processing marker exists for this session, a worker is already
        # ingesting (or the stale watcher will recover it); re-queuing would
        # race two workers onto one transcript.
        if (BACKLOG_DIR / f"{safe_sid}.processing").exists():
            log.info("stop hook: backlog claim in-flight for %r; not re-queueing", session_id)
            return
        marker = BACKLOG_DIR / f"{safe_sid}.json"
        _atomic_write_text(marker, json.dumps({
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
) -> int:
    """Launch the ingestion pipeline as a background process.

    Captures stderr/stdout to a log file so silent failures are recoverable.
    Handles Windows (CREATE_NEW_PROCESS_GROUP) and POSIX (start_new_session)
    subprocess detachment.

    bounds concurrent spawns via ``SPAWN_CAP`` so a burst of
    Stop hooks doesn't load N embedding models at once.
    on Popen failure, queue the ingestion to ``BACKLOG_DIR``
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
    safe_session = _sanitize_session_id(session_id)
    trace_path = TRACE_DIR / f"{safe_session}.json"
    log_path = LOG_DIR / f"{safe_session}.log"
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

    from truememory.ingest.hooks._shared import check_extraction_budget
    if not check_extraction_budget():
        log.warning("stop hook: extraction budget exhausted (20/hr); queueing session %r", session_id)
        _queue_to_backlog(
            transcript_path, session_id, user_id, db_path,
            reason="extraction_budget_exhausted",
        )
        return 0

    # Use flock-based spawn gate to prevent the TOCTOU race where N hooks
    # all check pgrep simultaneously, all see 0, and all spawn.
    from truememory.hooks.core import spawn_gate, register_spawned_pid, _load_cap_state

    effective_cap = _load_cap_state().get("cap", SPAWN_CAP)

    with spawn_gate() as allowed:
        if not allowed:
            log.warning(
                "stop hook: at spawn cap (cap %d); queueing session "
                "%r to backlog for later",
                effective_cap, session_id,
            )
            _queue_to_backlog(
                transcript_path, session_id, user_id, db_path,
                reason=f"spawn_cap_reached:SPAWN_CAP={effective_cap}",
            )
            return 0

        log_file = None
        try:
            log_file = open(log_path, "a", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                close_fds=(sys.platform != "win32"),
                **detach_kwargs,
            )
            register_spawned_pid(proc.pid)
            return proc.pid
        except Exception as e:
            log.warning(
                "stop hook: background launch failed (%s); queueing session "
                "%r to backlog for later",
                e, session_id,
            )
            _queue_to_backlog(
                transcript_path, session_id, user_id, db_path,
                reason=f"popen_failed:{type(e).__name__}:{e}",
            )
            return 0
        finally:
            if log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass
    return 0


if __name__ == "__main__":
    main()
