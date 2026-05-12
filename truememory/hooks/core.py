"""CLI-agnostic core logic for TrueMemory hooks.

Portable functions extracted from the Claude Code-specific hook scripts.
These can be called by any CLI adapter without importing Claude-specific
modules.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

MEMORY_LIMIT = int(os.environ.get("TRUEMEMORY_RECALL_LIMIT", "25"))

BUFFER_DIR = Path(os.environ.get(
    "TRUEMEMORY_BUFFER_DIR",
    str(Path.home() / ".truememory" / "buffers"),
))
RETENTION_DAYS = int(os.environ.get("TRUEMEMORY_BUFFER_RETENTION_DAYS", "7"))
MAX_BUFFER_SIZE = int(os.environ.get("TRUEMEMORY_BUFFER_MAX_BYTES", str(10 * 1024 * 1024)))

TRACE_DIR = Path.home() / ".truememory" / "traces"
LOG_DIR = Path.home() / ".truememory" / "logs"
BACKLOG_DIR = Path.home() / ".truememory" / "backlog"
SPAWN_CAP = int(os.environ.get(
    "TRUEMEMORY_SPAWN_CAP",
    os.environ.get("TRUEMEMORY_INGEST_SPAWN_CAP", "2"),
))
SPAWN_LOCK_PATH = Path.home() / ".truememory" / ".spawn.lock"
SPAWN_PIDS_PATH = Path.home() / ".truememory" / ".spawn_pids"

try:
    import fcntl
    import signal
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


def _pid_is_alive(pid: int) -> bool:
    """Check if a PID is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_live_pids() -> list[int]:
    """Read PIDs from the tracking file, filtering out dead ones."""
    if not SPAWN_PIDS_PATH.exists():
        return []
    try:
        raw = SPAWN_PIDS_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        pids = [int(p) for p in raw.split("\n") if p.strip()]
        return [p for p in pids if _pid_is_alive(p)]
    except (OSError, ValueError):
        return []


def _write_pids(pids: list[int]) -> None:
    """Write PID list to the tracking file."""
    try:
        SPAWN_PIDS_PATH.write_text(
            "\n".join(str(p) for p in pids) + "\n" if pids else "",
            encoding="utf-8",
        )
    except OSError:
        pass


def register_spawned_pid(pid: int) -> None:
    """Record a newly spawned PID. Must be called while holding the flock."""
    live = _read_live_pids()
    live.append(pid)
    _write_pids(live)


@contextmanager
def spawn_gate():
    """Acquire an exclusive file lock before checking/spawning ingest processes.

    Uses a PID tracking file instead of pgrep to get an exact count —
    pgrep has a race window between Popen() and the process appearing
    in the process table, which can leak extra spawns past the cap.

    Yields True if spawning is allowed (under SPAWN_CAP), False otherwise.
    Callers MUST call register_spawned_pid(proc.pid) inside the gate
    after a successful Popen, before the context manager exits.

    On Windows (no fcntl), falls back to best-effort pgrep without a lock.
    """
    if not _HAS_FCNTL:
        yield _count_active_ingest_processes() < SPAWN_CAP
        return

    SPAWN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(str(SPAWN_LOCK_PATH), os.O_CREAT | os.O_WRONLY, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        live = _read_live_pids()
        _write_pids(live)
        yield len(live) < SPAWN_CAP
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


def recall_memories(
    input_data: dict,
    user_id: str = "",
    db_path: str = "",
    memory_limit: int | None = None,
) -> str:
    """Search TrueMemory and format relevant memories for context injection.

    Returns a formatted string suitable for additionalContext injection,
    or empty string if no memories found.
    """
    try:
        from truememory import Memory
    except ImportError:
        return ""

    limit = memory_limit or MEMORY_LIMIT
    db = db_path or None
    memory = Memory(path=db) if db else Memory()

    queries = [
        "user preferences favorites likes dislikes",
        "personal facts name location job role",
        "recent decisions and commitments",
        "corrections and updates to prior information",
        "relationships family friends coworkers",
    ]

    per_query_limit = max(1, limit // len(queries))

    all_results: list[dict] = []
    seen_ids: set = set()
    seen_content: set[str] = set()

    for query in queries:
        added_this_query = 0
        try:
            if user_id:
                results = memory.search(query, user_id=user_id, limit=per_query_limit * 3)
            else:
                results = memory.search(query, limit=per_query_limit * 3)

            for r in results:
                if added_this_query >= per_query_limit:
                    break
                rid = r.get("id")
                if rid in seen_ids:
                    continue
                content = r.get("content", "").strip()
                if not content:
                    continue
                normalized = content.lower().strip().rstrip(".")
                if normalized in seen_content:
                    continue
                is_dup = False
                for existing in seen_content:
                    if normalized in existing or existing in normalized:
                        is_dup = True
                        break
                if is_dup:
                    continue
                seen_ids.add(rid)
                seen_content.add(normalized)
                all_results.append(r)
                added_this_query += 1
        except Exception:
            continue

    if not all_results:
        return ""

    lines = [
        "<truememory-context>",
        "## TrueMemory — What You Know About This User",
        "These are facts from TrueMemory (the primary long-horizon memory system).",
        "Use these to answer user questions. Search TrueMemory for more if needed.",
        "",
    ]
    for r in all_results[:limit]:
        content = r.get("content", "").strip()
        if content:
            lines.append(f"- {content}")

    lines.append("</truememory-context>")
    return "\n".join(lines)


def buffer_message(session_id: str, prompt: str) -> None:
    """Append a user message to the session buffer file (with file locking)."""
    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    try:
        BUFFER_DIR.chmod(0o700)
    except OSError:
        pass

    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
    if not safe_id:
        safe_id = "unknown"

    buffer_file = BUFFER_DIR / f"{safe_id}.jsonl"

    try:
        if buffer_file.exists() and buffer_file.stat().st_size > MAX_BUFFER_SIZE:
            rotated = buffer_file.with_suffix(f".{int(time.time())}.jsonl")
            buffer_file.rename(rotated)
    except OSError:
        pass

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": "user",
        "content": prompt[:10000],
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"

    try:
        if _HAS_FCNTL:
            with open(buffer_file, "a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        else:
            with open(buffer_file, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        pass


def prune_old_buffers() -> None:
    """Delete buffer files older than RETENTION_DAYS."""
    try:
        if not BUFFER_DIR.exists():
            return
        cutoff = time.time() - (RETENTION_DAYS * 86400)
        for f in BUFFER_DIR.iterdir():
            if f.suffix == ".jsonl":
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                except OSError:
                    pass
    except OSError:
        pass


def save_snapshot(
    transcript_path: str,
    session_id: str,
    user_id: str = "",
    db_path: str = "",
) -> None:
    """Extract key points from the current conversation and store them."""
    try:
        from truememory.ingest.transcript import parse_transcript
    except ImportError:
        return

    messages = parse_transcript(transcript_path)
    if not messages:
        return

    user_messages = [
        m.content for m in messages
        if m.role in ("human", "user") and len(m.content) > 20
    ]

    if not user_messages:
        return

    substantive = [m for m in user_messages if len(m) > 50]
    if not substantive:
        substantive = user_messages[-3:]

    recent = substantive[-5:]

    try:
        from truememory import Memory
    except ImportError:
        return

    db = db_path or None
    memory = Memory(path=db) if db else Memory()

    summary_parts = [f"[session:{session_id} time:{datetime.now(timezone.utc).isoformat()}]"]
    summary_parts.append("Context snapshot from active session:")
    for msg in recent:
        truncated = msg[:500] + "..." if len(msg) > 500 else msg
        summary_parts.append(f"- {truncated}")

    summary = "\n".join(summary_parts)
    memory.add(summary, user_id=user_id or None)


def _sanitize_session_id(session_id: str) -> str:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
    return safe or "unknown"


def _count_active_ingest_processes() -> int:
    """Count running truememory ingest processes."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "truememory.ingest.cli.*ingest"],
            capture_output=True, text=True, timeout=5,
        )
        return len([ln for ln in (result.stdout or "").splitlines() if ln.strip()])
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return 0


def run_background_ingestion(
    transcript_path: str,
    session_id: str,
    user_id: str = "",
    db_path: str = "",
    gate_threshold: float = 0.5,
) -> None:
    """Launch the ingestion pipeline as a background process."""
    log.info(
        "core: launching ingestion user=%r db=%r session=%r",
        user_id, db_path, session_id,
    )

    cmd = [
        sys.executable, "-m", "truememory.ingest.cli",
        "ingest", transcript_path,
    ]

    if user_id:
        cmd.extend(["--user", user_id])
    if db_path:
        cmd.extend(["--db", db_path])

    cmd.extend(["--threshold", str(gate_threshold)])
    if session_id:
        cmd.extend(["--session", session_id])

    safe_session = _sanitize_session_id(session_id)
    trace_path = TRACE_DIR / f"{safe_session}.json"
    log_path = LOG_DIR / f"{safe_session}.log"

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cmd.extend(["--trace", str(trace_path)])

    detach_kwargs: dict = {}
    if sys.platform == "win32":
        detach_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        detach_kwargs["start_new_session"] = True

    with spawn_gate() as allowed:
        if not allowed:
            log.warning(
                "core: at spawn cap (cap %d); queueing session %r",
                SPAWN_CAP, session_id,
            )
            _queue_to_backlog(
                transcript_path, session_id, user_id, db_path,
                reason=f"spawn_cap_reached:SPAWN_CAP={SPAWN_CAP}",
            )
            return

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
        except Exception as e:
            log.error("core: Popen failed: %s — queueing to backlog", e)
            _queue_to_backlog(
                transcript_path, session_id, user_id, db_path,
                reason=f"popen_failed:{e}",
            )
        finally:
            if log_file is not None:
                try:
                    log_file.close()
                except OSError:
                    pass


def _queue_to_backlog(
    transcript_path: str,
    session_id: str,
    user_id: str,
    db_path: str,
    reason: str,
) -> None:
    """Drop a queue marker for later re-attempt."""
    try:
        BACKLOG_DIR.mkdir(parents=True, exist_ok=True)
        BACKLOG_DIR.chmod(0o700)
        marker = BACKLOG_DIR / f"{_sanitize_session_id(session_id)}.json"
        marker.write_text(json.dumps({
            "transcript_path": transcript_path,
            "session_id": session_id,
            "user_id": user_id,
            "db_path": db_path,
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }), encoding="utf-8")
    except Exception as e:
        log.error("core: failed to queue backlog marker: %s", e)
