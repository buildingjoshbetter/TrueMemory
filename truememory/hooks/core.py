"""CLI-agnostic core logic for TrueMemory hooks.

Portable functions extracted from the Claude Code-specific hook scripts.
These can be called by any CLI adapter without importing Claude-specific
modules.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

log = logging.getLogger(__name__)

_BASE_MEMORY_LIMIT = int(os.environ.get("TRUEMEMORY_RECALL_LIMIT", "25"))

# Issue #396: search intensity scales the recall limit.
# Standard = 25, Enhanced/Max = 35.
_INTENSITY_MEMORY_LIMITS = {
    "standard": _BASE_MEMORY_LIMIT,
    "enhanced": 35,
    "max": 35,
}


def _get_search_intensity() -> str:
    """Read search_intensity from persistent config (default: standard)."""
    try:
        import json as _json
        config_path = Path.home() / ".truememory" / "config.json"
        if config_path.exists():
            config = _json.loads(config_path.read_text(encoding="utf-8"))
            return config.get("search_intensity", "standard")
    except Exception:
        pass
    return "standard"


MEMORY_LIMIT = _BASE_MEMORY_LIMIT  # module-level default; callers use recall_memories()

BUFFER_DIR = Path(os.environ.get(
    "TRUEMEMORY_BUFFER_DIR",
    str(Path.home() / ".truememory" / "buffers"),
))
RETENTION_DAYS = int(os.environ.get("TRUEMEMORY_BUFFER_RETENTION_DAYS", "7"))
MAX_BUFFER_SIZE = int(os.environ.get("TRUEMEMORY_BUFFER_MAX_BYTES", str(10 * 1024 * 1024)))

TRACE_DIR = Path.home() / ".truememory" / "traces"
LOG_DIR = Path.home() / ".truememory" / "logs"
BACKLOG_DIR = Path.home() / ".truememory" / "backlog"
# ---------------------------------------------------------------------------
# Dynamic spawn cap — adapts to tier, hardware, and system health
# ---------------------------------------------------------------------------

# Edge: CPU-bound (Model2Vec is numpy, no GPU). Cap by core count.
_EDGE_HARD_CEILING = 5
# Base/Pro: GPU-bound (Qwen3 on MPS). Cap by unified memory.
_GPU_HARD_CEILING = 6
_GPU_OS_RESERVE_GB = 2
_GPU_PROCESS_COST_GB = {"base": 1.0, "pro": 1.2}

_HARD_FLOOR = 1
_WARN_CONSECUTIVE_THRESHOLD = 2
_RAMP_UP_COOLDOWN_SECONDS = 120

# State file for persisting cap + swap readings across cascade processes
_SPAWN_CAP_STATE_PATH = Path.home() / ".truememory" / ".spawn_cap_state"
_STATE_EXPIRY_SECONDS = 300


def _get_current_tier() -> str:
    try:
        import json as _json
        config_path = Path.home() / ".truememory" / "config.json"
        if config_path.exists():
            return _json.loads(config_path.read_text(encoding="utf-8")).get("tier", "edge")
    except Exception:
        pass
    return "edge"


def _get_physical_cores() -> int:
    """Physical CPU core count (not logical/hyperthreaded)."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"],
            capture_output=True, text=True, timeout=2,
        )
        return int(result.stdout.strip())
    except Exception:
        return os.cpu_count() or 4


def _get_total_memory_gb() -> int:
    """Total unified memory in GB (Apple Silicon shares CPU + GPU)."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
        )
        return int(result.stdout.strip()) // (1024 ** 3)
    except Exception:
        return 8


def _get_memory_free_pct() -> int:
    """macOS system-wide memory free percentage (0-100).

    Parses the `memory_pressure` command output line:
      System-wide memory free percentage: 81%
    """
    try:
        import re
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r"free percentage:\s*(\d+)%", result.stdout)
        if match:
            return int(match.group(1))
        return 100
    except Exception:
        return 100


def _classify_memory_pressure(free_pct: int) -> str:
    """Map free memory percentage to pressure level."""
    if free_pct < 15:
        return "critical"
    if free_pct < 40:
        return "warn"
    return "normal"


def _get_swap_used_gb() -> float:
    """Current swap usage in GB on macOS.

    Parses `sysctl vm.swapusage` output like:
      vm.swapusage: total = 2048.00M  used = 1024.00M  free = 1024.00M
    Handles both M (megabytes) and G (gigabytes) units.
    """
    try:
        import re
        result = subprocess.run(
            ["sysctl", "vm.swapusage"],
            capture_output=True, text=True, timeout=2,
        )
        match = re.search(r"used\s*=\s*([\d.]+)([MG])", result.stdout)
        if match:
            val = float(match.group(1))
            unit = match.group(2)
            return val / 1024.0 if unit == "M" else val
        return 0.0
    except Exception:
        return 0.0


def _load_cap_state() -> dict:
    """Load persisted spawn cap state from disk.

    Returns dict with keys: cap, warn_count, last_swap_gb, timestamp.
    State expires after _STATE_EXPIRY_SECONDS (e.g., after reboot).
    Must be called while holding the spawn flock.
    """
    try:
        if not _SPAWN_CAP_STATE_PATH.exists():
            return {}
        import json as _json
        data = _json.loads(_SPAWN_CAP_STATE_PATH.read_text(encoding="utf-8"))
        ts = data.get("timestamp", 0)
        if time.time() - ts > _STATE_EXPIRY_SECONDS:
            return {}
        return data
    except Exception:
        return {}


def _save_cap_state(
    cap: int, warn_count: int, swap_gb: float,
    last_ramp_time: float | None = None,
) -> None:
    """Persist spawn cap state to disk atomically.

    Must be called while holding the spawn flock.
    """
    try:
        import json as _json
        _SPAWN_CAP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _SPAWN_CAP_STATE_PATH.with_suffix(".tmp")
        data = {
            "cap": cap,
            "warn_count": warn_count,
            "last_swap_gb": swap_gb,
            "timestamp": time.time(),
        }
        if last_ramp_time is not None:
            data["last_ramp_time"] = last_ramp_time
        tmp.write_text(_json.dumps(data), encoding="utf-8")
        tmp.rename(_SPAWN_CAP_STATE_PATH)
    except Exception:
        pass


def _compute_ceiling(tier: str) -> int:
    """Compute the hard ceiling using the correct metric for each tier.

    Edge:     CPU core count (Model2Vec is numpy, CPU-only, RAM irrelevant)
    Base/Pro: Unified memory (Qwen3 loads ~1-1.2GB per process via MPS)
    """
    if tier == "edge":
        cores = _get_physical_cores()
        return min(max(_HARD_FLOOR, cores - 1), _EDGE_HARD_CEILING)

    total_gb = _get_total_memory_gb()
    cost_gb = _GPU_PROCESS_COST_GB.get(tier, 1.2)
    usable_gb = total_gb - _GPU_OS_RESERVE_GB
    memory_slots = max(_HARD_FLOOR, int(usable_gb / cost_gb))
    return min(memory_slots, _GPU_HARD_CEILING)


def _is_swap_growing(current_swap_gb: float, last_swap_gb: float) -> bool:
    """Detect actively growing swap (not just historical usage)."""
    delta = current_swap_gb - last_swap_gb
    return delta > 0.5


def _get_spawn_cap() -> int:
    """Return the effective spawn cap with ramp-up/ramp-down.

    Algorithm:
    1. Compute ceiling using the correct metric per tier:
       - Edge: physical CPU cores - 1 (capped at 8)
       - Base/Pro: (unified_memory - 2GB) / model_cost (capped at 6)
    2. Load persisted state (cap, warn_count, swap baseline)
    3. Check memory pressure (parsed from free percentage, not string match)
    4. Check swap growth (delta from last reading, not absolute value)
    5. Ramp down immediately on critical/growing swap
    6. Halve on sustained warn (2+ consecutive, hysteresis)
    7. Ramp up by 1 per tick toward ceiling when healthy
    8. Persist state for next cascade process
    9. Env var override bypasses everything (for power users)
    """
    # Bug 3 fix: read env var at call time, not module load
    override = os.environ.get(
        "TRUEMEMORY_SPAWN_CAP",
        os.environ.get("TRUEMEMORY_INGEST_SPAWN_CAP", ""),
    )
    if override:
        return int(override)

    tier = _get_current_tier()
    ceiling = _compute_ceiling(tier)

    # Bug 2 fix: load persisted state from disk
    state = _load_cap_state()
    current_cap = state.get("cap", _HARD_FLOOR)
    warn_count = state.get("warn_count", 0)
    last_swap_gb = state.get("last_swap_gb", 0.0)
    last_ramp_time = state.get("last_ramp_time", 0.0)

    # Bug 1 fix: parse actual free percentage from memory_pressure
    free_pct = _get_memory_free_pct()
    pressure = _classify_memory_pressure(free_pct)

    # Bug 4 fix: check swap growth, not absolute value
    swap_gb = _get_swap_used_gb()
    swap_growing = _is_swap_growing(swap_gb, last_swap_gb)

    # Emergency: critical pressure or actively growing swap
    if pressure == "critical" or swap_growing:
        current_cap = _HARD_FLOOR
        warn_count = 0
        log.warning(
            "Spawn cap: EMERGENCY (free=%d%%, pressure=%s, swap=%.1fGB, "
            "delta=%.1fGB) → cap=%d",
            free_pct, pressure, swap_gb, swap_gb - last_swap_gb, _HARD_FLOOR,
        )
        _save_cap_state(current_cap, warn_count, swap_gb, 0.0)
        return _HARD_FLOOR

    # Sustained warn: halve after 2+ consecutive readings (hysteresis)
    if pressure == "warn":
        warn_count += 1
        if warn_count >= _WARN_CONSECUTIVE_THRESHOLD:
            current_cap = max(_HARD_FLOOR, current_cap // 2)
            log.warning(
                "Spawn cap: WARN sustained (%d checks, free=%d%%) → cap=%d",
                warn_count, free_pct, current_cap,
            )
            _save_cap_state(current_cap, warn_count, swap_gb, 0.0)
            return current_cap
    else:
        warn_count = 0

    # Healthy: ramp up by 1 toward ceiling, but only every _RAMP_UP_COOLDOWN_SECONDS
    now = time.time()
    if current_cap < ceiling and (now - last_ramp_time) >= _RAMP_UP_COOLDOWN_SECONDS:
        current_cap += 1
        last_ramp_time = now
        log.info("Spawn cap: ramp-up → cap=%d (ceiling=%d, free=%d%%)",
                 current_cap, ceiling, free_pct)

    _save_cap_state(current_cap, warn_count, swap_gb, last_ramp_time)
    return current_cap


SPAWN_CAP = 2
SPAWN_LOCK_PATH = Path.home() / ".truememory" / ".spawn.lock"
SPAWN_PIDS_PATH = Path.home() / ".truememory" / ".spawn_pids"

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


def _pid_is_alive(pid: int) -> bool:
    """Check if a PID is a live (non-zombie) process."""
    if pid <= 0:
        return False
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            return proc.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
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
    cap = _get_spawn_cap()

    if not _HAS_FCNTL:
        yield _count_active_ingest_processes() < cap
        return

    SPAWN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(str(SPAWN_LOCK_PATH), os.O_CREAT | os.O_WRONLY, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        live = _read_live_pids()
        _write_pids(live)
        yield len(live) < cap
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

    Uses a file-based cache (issue #559): after running the 5 search
    queries, results are cached with a timestamp. Subsequent calls within
    the TTL (default 5 min, env TRUEMEMORY_RECALL_CACHE_TTL) return the
    cached context instead of re-querying the full search pipeline.
    """
    # --- Issue #559: check cache first ---
    try:
        from truememory.ingest.hooks._shared import get_recall_cache, set_recall_cache
        cached = get_recall_cache(db_path or "", user_id)
        if cached is not None:
            return cached
    except Exception:
        pass

    try:
        from truememory import Memory
    except ImportError:
        return ""

    # Issue #396: scale recall limit based on search intensity
    if memory_limit:
        limit = memory_limit
    else:
        intensity = _get_search_intensity()
        limit = _INTENSITY_MEMORY_LIMITS.get(intensity, _BASE_MEMORY_LIMIT)
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
    context = "\n".join(lines)

    # Cache results for subsequent calls (issue #559)
    try:
        set_recall_cache(context, db_path or "", user_id)
    except Exception:
        pass

    return context


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
    if psutil is None:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "truememory.ingest.cli.*ingest"],
                capture_output=True, text=True, timeout=5,
            )
            return len([ln for ln in (result.stdout or "").splitlines() if ln.strip()])
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return 0
    count = 0
    try:
        for proc in psutil.process_iter(["cmdline", "status"]):
            try:
                if proc.info["status"] == psutil.STATUS_ZOMBIE:
                    continue
                cmdline = proc.info.get("cmdline") or []
                cmd_str = " ".join(cmdline)
                if "truememory.ingest.cli" in cmd_str and "ingest" in cmd_str:
                    count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    return count


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

    effective_cap = _load_cap_state().get("cap", SPAWN_CAP)

    with spawn_gate() as allowed:
        if not allowed:
            log.warning(
                "core: at spawn cap (cap %d); queueing session %r",
                effective_cap, session_id,
            )
            _queue_to_backlog(
                transcript_path, session_id, user_id, db_path,
                reason=f"spawn_cap_reached:SPAWN_CAP={effective_cap}",
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
        from truememory.ingest.hooks._shared import _atomic_write_text
        marker = BACKLOG_DIR / f"{_sanitize_session_id(session_id)}.json"
        _atomic_write_text(marker, json.dumps({
            "transcript_path": transcript_path,
            "session_id": session_id,
            "user_id": user_id,
            "db_path": db_path,
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }))
    except Exception as e:
        log.error("core: failed to queue backlog marker: %s", e)
