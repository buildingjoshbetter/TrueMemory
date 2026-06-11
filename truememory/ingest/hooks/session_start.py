#!/usr/bin/env python3
"""
SessionStart Hook — Memory Injection + First-Run Onboarding
=============================================================

Fires when a new Claude Code session begins. Two modes:

1. **First run** (no ~/.truememory/.onboarded marker):
   Injects the TrueMemory banner and guided setup instructions so
   Claude walks the user through tier selection on first launch.

2. **Normal run** (marker exists):
   Searches TrueMemory for relevant memories and injects them as
   additionalContext so Claude has full context from the start.

Input (stdin JSON):
    {"session_id": "...", "cwd": "...", "transcript_path": "..."}

Output (stdout JSON):
    {"additionalContext": "<truememory-context>...</truememory-context>"}
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from truememory import _platform

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

log = logging.getLogger(__name__)

_BASE_MEMORY_LIMIT = int(os.environ.get("TRUEMEMORY_RECALL_LIMIT", "25"))

# Issue #396: search intensity scales the recall limit at session start.
# Standard = 25, Enhanced/Max = 35.
_INTENSITY_MEMORY_LIMITS = {
    "standard": _BASE_MEMORY_LIMIT,
    "enhanced": 35,
    "max": 35,
}


# Issue #636 (M-16): valid intensity levels. Normalize raw config values
# (lowercase + allowlist) so "MAX"/"Enhanced"/garbage/null resolve to a known
# level instead of silently mapping to base via .get()'s default — keeping
# this reader consistent with user_prompt_submit's fail-closed dispatch.
_VALID_INTENSITIES = frozenset({"standard", "enhanced", "max"})


def _normalize_intensity(value: object) -> str:
    """Normalize a raw config intensity value to a known level (issue #636)."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _VALID_INTENSITIES:
            return normalized
    return "standard"


def _get_search_intensity() -> str:
    """Read search_intensity from persistent config (default: standard).

    Normalized through the allowlist (issue #636) so an invalid or
    mismatched-case value resolves predictably rather than slipping through.
    """
    try:
        config_path = Path.home() / ".truememory" / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            return _normalize_intensity(config.get("search_intensity", "standard"))
    except Exception:
        pass
    return "standard"


MEMORY_LIMIT = _BASE_MEMORY_LIMIT  # module-level default; overridden in main()
# Max directives force-injected at session start. Uncapped injection let a
# large directive set consume unbounded context (issue #589, D-4).
DIRECTIVE_LIMIT = int(os.environ.get("TRUEMEMORY_DIRECTIVE_LIMIT", "50"))
# Per-memory character cap and total payload budget (issue #578).
# Memories exceeding the per-entry cap are sliced on a word boundary and
# suffixed with a pointer so the agent can fetch the full text on demand.
RECALL_MEMORY_CHARS = int(os.environ.get("TRUEMEMORY_RECALL_MEMORY_CHARS", "500"))
RECALL_BUDGET_CHARS = int(os.environ.get("TRUEMEMORY_RECALL_BUDGET_CHARS", "8192"))

# Issue #636 (M-72): "max" intensity raises MEMORY_LIMIT to 35 but the recall
# payload was still capped at RECALL_BUDGET_CHARS (8KB), so _apply_budget
# dropped the extra memories and the knob was a silent no-op. Scale the budget
# with intensity so "max" actually injects a larger payload. An explicit
# TRUEMEMORY_RECALL_BUDGET_CHARS override is authoritative (multiplier x1).
_INTENSITY_BUDGET_MULTIPLIER = {
    "standard": 1.0,
    "enhanced": 1.0,
    "max": 2.0,
}
_BUDGET_ENV_OVERRIDDEN = "TRUEMEMORY_RECALL_BUDGET_CHARS" in os.environ


def _intensity_budget(intensity: str) -> int:
    """Effective recall payload budget for *intensity* (issue #636, M-72).

    An explicit env override pins the budget (multiplier x1) so operators keep
    full control; otherwise "max" gets a larger budget so its higher memory
    limit is not silently nullified by the 8KB cap.
    """
    if _BUDGET_ENV_OVERRIDDEN:
        return RECALL_BUDGET_CHARS
    return int(RECALL_BUDGET_CHARS * _INTENSITY_BUDGET_MULTIPLIER.get(intensity, 1.0))
ONBOARDED_MARKER = Path.home() / ".truememory" / ".onboarded"
BACKLOG_DIR = Path.home() / ".truememory" / "backlog"
_DRAIN_CAP = 3
_SCAN_MARKER = Path.home() / ".truememory" / ".last_stale_scan"
_SCAN_INTERVAL = 900  # 15 minutes
_SCAN_CAP = 3  # max sessions to queue per scan
_EXTRACTED_MARKER_MAX_AGE = int(os.environ.get("TRUEMEMORY_EXTRACTED_MARKER_MAX_AGE_DAYS", "30")) * 86400

_EXTRACTION_SENTINEL = "[[TRUEMEMORY_INTERNAL_EXTRACTION]]"
_EXTRACTION_LEGACY_PREFIXES = (
    "You are a memory extraction system",
    "You are comparing a NEW fact",
)

BANNER = r"""
████████╗██████╗ ██╗   ██╗███████╗    ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗
╚══██╔══╝██╔══██╗██║   ██║██╔════╝    ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝
   ██║   ██████╔╝██║   ██║█████╗      ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝
   ██║   ██╔══██╗██║   ██║██╔══╝      ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝
   ██║   ██║  ██║╚██████╔╝███████╗    ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝
                                  a sauron company
""".strip()

SETUP_GUIDE = """
Welcome to TrueMemory — persistent memory for AI agents.

TrueMemory needs a quick one-time setup. IMPORTANT: Present this setup guide to the user NOW, before responding to anything else. Walk them through these steps:

1. **Choose a tier** — ask the user to pick one:
   - **Edge** — fastest, lightweight. Model2Vec embeddings (8M params), MiniLM reranker. Best for: local-only, low-resource machines.
   - **Base** — balanced. Qwen3 embeddings (256d), gte-reranker-modernbert. Best for: most users. Recommended.
   - **Pro** — maximum accuracy. Qwen3 + HyDE query expansion. Requires an API key (Anthropic, OpenRouter, or OpenAI).

2. **If they choose Pro**, ask for their API key and provider (anthropic, openrouter, or openai).

3. **Ask for their email** — ask: "What's your email? We'll use it to send you important updates." Always include it in the configure call if provided.

4. **Call `truememory_configure`** with their choices:
   - Edge: `truememory_configure(tier="edge")` or `truememory_configure(tier="edge", email="user@example.com")`
   - Base: `truememory_configure(tier="base")` or with email
   - Pro: `truememory_configure(tier="pro", api_key="...", api_provider="...", email="...")`

5. **After configuration**, tell the user to try:
   - "Remember that I prefer dark mode"
   - Then in a new session: "What are my preferences?"

6. **Done!** TrueMemory will now automatically remember facts, preferences, and decisions across all sessions.
""".strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--user", default=os.environ.get("TRUEMEMORY_USER_ID", ""))
    p.add_argument("--db", default=os.environ.get("TRUEMEMORY_DB_PATH", ""))
    p.add_argument("--scan-stale", action="store_true", help=argparse.SUPPRESS)
    args, _ = p.parse_known_args()
    return args


def _is_first_run() -> bool:
    return not ONBOARDED_MARKER.exists()


def _check_for_update() -> str:
    """Check if the MCP server wrote an update notice during startup."""
    try:
        update_path = Path.home() / ".truememory" / ".update_available"
        if not update_path.exists():
            return ""
        data = json.loads(update_path.read_text(encoding="utf-8"))
        if data.get("shown"):
            return ""
        if data.get("update_available"):
            data["shown"] = True
            try:
                tmp = update_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data), encoding="utf-8")
                tmp.rename(update_path)
            except Exception:
                pass
            return (
                "<truememory-update>\n"
                f"A new version of TrueMemory is available: v{data.get('latest_version', '?')}. "
                f"Tell the user: \"{data.get('message', 'Run: uv tool upgrade truememory')}\"\n"
                "</truememory-update>"
            )
    except Exception:
        pass
    return ""


def _check_email_needed() -> str:
    """Prompt for email if the user hasn't provided one yet."""
    try:
        config_path = Path.home() / ".truememory" / "config.json"
        if not config_path.exists():
            return ""
        config = json.loads(config_path.read_text(encoding="utf-8"))
        # Only prompt if onboarded but no email
        if config.get("tier") and not config.get("email"):
            return (
                "<truememory-email-request>\n"
                "TrueMemory doesn't have your email yet. Ask the user: "
                "\"What's your email? We use it to send important updates about TrueMemory.\" "
                "If they provide one, call truememory_configure with their current tier and the email: "
                f"truememory_configure(tier=\"{config.get('tier', 'edge')}\", email=\"their@email.com\"). "
                "If they decline, respect that and don't ask again this session.\n"
                "</truememory-email-request>"
            )
    except Exception:
        pass
    return ""


def _drain_backlog() -> None:
    """Process queued sessions from the backlog directory.

    Uses the flock-based spawn gate from core to prevent the avalanche
    scenario where N concurrent SessionStart hooks all drain simultaneously,
    spawning N × _DRAIN_CAP ingest processes.

    Uses atomic rename (.json → .processing) to prevent TOCTOU races where
    multiple drainers read the same marker before either acquires the flock.
    """
    if not BACKLOG_DIR.exists():
        return

    from truememory.ingest.hooks._shared import (
        cleanup_stale_processing,
        check_extraction_budget,
        refund_extraction_budget,
        record_stale_processing_pid,
        mark_session_extracted,
        _quarantine_marker,
    )
    cleanup_stale_processing(BACKLOG_DIR)

    try:
        markers = sorted(BACKLOG_DIR.glob("*.json"))[:_DRAIN_CAP]
    except Exception:
        return

    from truememory.hooks.core import spawn_gate, register_spawned_pid

    for marker_path in markers:
        claimed_path = marker_path.with_suffix(".processing")
        try:
            marker_path.rename(claimed_path)
        except (FileNotFoundError, OSError):
            continue

        # M-14: a corrupt JSON marker must NOT be recycled back to .json (it
        # would poison a _DRAIN_CAP slot forever). Read + parse first; on a
        # decode error quarantine it to .corrupt and move on.
        try:
            raw = claimed_path.read_text(encoding="utf-8")
        except OSError:
            try:
                claimed_path.rename(marker_path)
            except OSError:
                pass
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            _quarantine_marker(claimed_path)
            continue

        try:
            transcript = data.get("transcript_path", "")
            if not transcript or not Path(transcript).exists():
                claimed_path.unlink(missing_ok=True)
                continue

            if not check_extraction_budget():
                log.info("Drain: extraction budget exhausted, leaving backlog for next hour")
                try:
                    claimed_path.rename(marker_path)
                except OSError:
                    pass
                return

            with spawn_gate() as allowed:
                if not allowed:
                    # M-71: the budget slot consumed above is never used when the
                    # gate denies — refund it so a denied spawn does not burn budget.
                    refund_extraction_budget()
                    log.info("Drain: spawn cap reached, leaving remaining backlog for next session")
                    try:
                        claimed_path.rename(marker_path)
                    except OSError:
                        pass
                    return

                import subprocess
                cmd = [
                    sys.executable, "-m", "truememory.ingest.cli",
                    "ingest", transcript,
                ]
                session_id = data.get("session_id", "")
                if session_id:
                    cmd.extend(["--session", session_id])
                if data.get("user_id"):
                    cmd.extend(["--user", data["user_id"]])
                if data.get("db_path"):
                    cmd.extend(["--db", data["db_path"]])
                from truememory.ingest.hooks._shared import _safe_session_id
                _log_dir = Path.home() / ".truememory" / "logs"
                _log_dir.mkdir(parents=True, exist_ok=True)
                _safe_sid = _safe_session_id(data.get('session_id', 'unknown')) or 'unknown'
                _log_file = open(
                    _log_dir / f"{_safe_sid}.log",
                    "a", encoding="utf-8",
                )
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=_log_file,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        **_platform.spawn_kwargs(),
                    )
                finally:
                    _log_file.close()
                register_spawned_pid(proc.pid)
                record_stale_processing_pid(claimed_path, proc.pid)
                # M-34: also record an optimistic EXTRACTED_DIR marker tagged
                # with the worker PID. Without it, should_extract_session()
                # sees no marker for the worker's whole runtime and the Stop
                # hook spawns a parallel ingest of the same transcript.
                if session_id:
                    mark_session_extracted(session_id, transcript, spawned_pid=proc.pid)
            # NOTE (issue #422): do NOT unlink the .processing claim here.
            # Removing it on spawn (before the worker finishes) means a worker
            # that exits non-zero — crash, OOM, embed-model error — leaves no
            # claim for cleanup_stale_processing to recover, silently dropping
            # the session. We now leave the claim in place: the ingest CLI
            # deletes it on confirmed success (clear_backlog_processing), and a
            # dead worker leaves it so the 30-minute stale watcher restores it
            # to .json and re-queues the session.
            log.info("Drained backlog session: %s", data.get("session_id", "?"))
        except Exception as e:
            try:
                claimed_path.rename(marker_path)
            except OSError:
                pass
            log.debug("Failed to drain backlog entry %s: %s", marker_path.name, e)


def _is_extraction_transcript(transcript_path: Path) -> bool:
    """Check if a transcript is TrueMemory extraction noise, not a real conversation.

    Looks for the structured sentinel tag or legacy extraction prompt prefixes
    in the first user message. Reads only the first 30 lines of the file.
    """
    try:
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 30:
                    break
                try:
                    data = json.loads(line)
                    if data.get("type") != "user":
                        continue
                    msg = data.get("message", {})
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    if isinstance(content, list):
                        content = content[0].get("text", "") if content else ""
                    if _EXTRACTION_SENTINEL in content:
                        return True
                    for prefix in _EXTRACTION_LEGACY_PREFIXES:
                        if content.startswith(prefix):
                            return True
                    return False
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    except OSError:
        pass
    return False


def _cleanup_extracted_markers() -> None:
    """Remove extracted/ marker files older than _EXTRACTED_MARKER_MAX_AGE.

    The extraction pipeline writes one marker file per processed transcript.
    Without periodic cleanup these grow unboundedly (54K+ files observed in
    prod), causing slow directory listings and inode exhaustion (issue #579).
    """
    import time as _time

    from truememory.ingest.hooks._shared import EXTRACTED_DIR

    if not EXTRACTED_DIR.exists():
        return

    cutoff = _time.time() - _EXTRACTED_MARKER_MAX_AGE
    removed = 0
    try:
        for marker in EXTRACTED_DIR.iterdir():
            try:
                if marker.stat().st_mtime < cutoff:
                    marker.unlink()
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    if removed > 0:
        log.info("Extracted marker cleanup: removed %d markers older than %d days",
                 removed, _EXTRACTED_MARKER_MAX_AGE // 86400)


def _read_scan_watermark(scan_fd: int) -> float:
    """Read the watermark timestamp stored in the scan marker file.

    Returns the stored timestamp, or 0.0 if the file is empty / corrupt.
    """
    try:
        os.lseek(scan_fd, 0, os.SEEK_SET)
        raw = os.read(scan_fd, 64)
        if raw:
            return float(raw.strip())
    except (OSError, ValueError):
        pass
    return 0.0


def _scan_stale_sessions() -> None:
    """Find transcripts from recent sessions that were never extracted.

    Runs at most once per _SCAN_INTERVAL.  Uses a watermark-based approach:
    the marker file stores the timestamp of the last successful scan.
    Only transcripts modified *after* that watermark are checked, making the
    scanner O(new) instead of O(all).  On first run (no watermark) it falls
    back to a 24-hour lookback window.

    Uses ``os.scandir()`` instead of ``Path.iterdir()`` to piggy-back on
    the DirEntry stat cache and avoid redundant syscalls.
    """
    import time
    import re

    _SCAN_MARKER.parent.mkdir(parents=True, exist_ok=True)
    try:
        scan_fd = os.open(str(_SCAN_MARKER), os.O_RDWR | os.O_CREAT)
        if _HAS_FCNTL:
            fcntl.flock(scan_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        return

    try:
        now = time.time()
        try:
            # Read watermark *before* the interval gate — an empty file
            # (first run, or O_CREAT just created it) must not short-circuit.
            watermark = _read_scan_watermark(scan_fd)
            if watermark > 0 and (now - watermark) < _SCAN_INTERVAL:
                return
        except OSError:
            return

        # Fall back to 24-hour lookback when no previous watermark exists
        # (first scan or corrupted marker).
        cutoff = watermark if watermark > 0 else (now - 86400)

        # M-37: the new watermark is written AFTER the scan, not before. If we
        # advanced to ``now`` up front and then stopped at ``_SCAN_CAP``, every
        # candidate beyond the cap (mtime < now) would fall outside the next
        # scan's window and never be re-checked. We instead advance only as far
        # as the oldest candidate we did NOT queue, so deferred work stays in
        # range. ``None`` means "no deferred candidate" → advance fully to now.
        oldest_deferred_mtime: float | None = None

        def _commit_watermark(value: float) -> None:
            try:
                os.lseek(scan_fd, 0, os.SEEK_SET)
                os.ftruncate(scan_fd, 0)
                os.write(scan_fd, str(value).encode("utf-8"))
            except OSError:
                pass

        claude_dir = Path.home() / ".claude" / "projects"
        if not claude_dir.exists():
            # Nothing to scan, but still gate the next interval.
            _commit_watermark(now)
            return

        from truememory.ingest.hooks._shared import EXTRACTED_DIR, _safe_session_id, mark_session_extracted
        from truememory.ingest.hooks.stop import _queue_to_backlog

        uuid_re = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
        queued = 0
        skipped_noise = 0

        try:
            project_entries = os.scandir(str(claude_dir))
        except OSError:
            _commit_watermark(now)
            return

        for proj_entry in project_entries:
            if not proj_entry.is_dir(follow_symlinks=False):
                continue
            try:
                file_entries = os.scandir(proj_entry.path)
            except OSError:
                continue
            for entry in file_entries:
                if not entry.name.endswith(".jsonl"):
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                session_id = entry.name[:-6]  # strip .jsonl
                if not uuid_re.match(session_id):
                    continue
                try:
                    stat = entry.stat()
                    if stat.st_mtime < cutoff:
                        continue
                    if stat.st_size < 5000:
                        continue
                except OSError:
                    continue

                safe_id = _safe_session_id(session_id)
                if not safe_id:
                    continue

                marker = EXTRACTED_DIR / safe_id
                if marker.exists():
                    continue

                # M-15: skip if a backlog claim already exists for this session.
                # A queued .json or an in-flight .processing means a worker has
                # (or will) ingest this transcript; re-queueing would overwrite a
                # live worker's claim and cause two workers on one transcript.
                if (BACKLOG_DIR / f"{safe_id}.json").exists() or \
                        (BACKLOG_DIR / f"{safe_id}.processing").exists():
                    continue

                transcript = Path(entry.path)
                if _is_extraction_transcript(transcript):
                    try:
                        mark_session_extracted(session_id, str(transcript))
                    except Exception:
                        pass
                    skipped_noise += 1
                    continue

                if queued >= _SCAN_CAP:
                    # M-37: cap reached — this candidate (and any others in this
                    # window) are deferred to a later scan. Mark that we must not
                    # advance the watermark past the scan cutoff, or deferred
                    # candidates older than ``now`` would fall outside the next
                    # window and never be re-checked. (We break before queueing.)
                    oldest_deferred_mtime = cutoff
                    break

                _queue_to_backlog(
                    str(transcript), session_id, "", "",
                    reason="stale_session_recovery",
                )
                queued += 1
            if queued >= _SCAN_CAP:
                break

        if queued > 0:
            log.info("Stale session scanner: queued %d unextracted sessions", queued)
        if skipped_noise > 0:
            log.info("Stale session scanner: skipped %d extraction noise transcripts", skipped_noise)

        # M-37: when the cap was hit, hold the watermark at the scan cutoff so
        # the deferred candidates stay inside the next scan's [watermark, now]
        # window (re-queueing is idempotent — M-15 skips sessions that already
        # have a backlog claim). Otherwise the scan fully drained the window and
        # the watermark advances to ``now``. The _SCAN_INTERVAL gate still
        # throttles how often the scan actually runs.
        if oldest_deferred_mtime is not None:
            _commit_watermark(oldest_deferred_mtime)
        else:
            _commit_watermark(now)

        # Piggyback on the scan window to prune stale extracted/ markers
        # (issue #579). Runs at most once per _SCAN_INTERVAL alongside the
        # stale-session scan so it adds no extra scheduling overhead.
        _cleanup_extracted_markers()
    finally:
        try:
            os.close(scan_fd)
        except OSError:
            pass


def _run_maintenance_background() -> None:
    """Spawn backlog drain + stale-session scan in a detached subprocess.

    Uses ``Popen(start_new_session=True)`` so the child survives hook exit
    (daemon threads die when the hook process exits — issue #557).
    stdout/stderr are redirected to a log file to avoid blocking.
    """
    # M-38: honor the guard env var. A maintenance child is spawned with
    # TRUEMEMORY_MAINTENANCE_CHILD=1; if such a child ever re-enters this hook
    # it must NOT spawn yet another maintenance subprocess (recursion / fork
    # storm). The flag was previously set but never read.
    if os.environ.get("TRUEMEMORY_MAINTENANCE_CHILD") == "1":
        return

    import subprocess

    _log_dir = Path.home() / ".truememory" / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _log_dir / "session_maintenance.log"

    script = (
        "from truememory.ingest.hooks.session_start import _drain_backlog, _scan_stale_sessions; "
        "_drain_backlog(); "
        "_scan_stale_sessions()"
    )

    try:
        log_file = open(_log_path, "a", encoding="utf-8")
        try:
            subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env={**os.environ, "TRUEMEMORY_MAINTENANCE_CHILD": "1"},
                **_platform.spawn_kwargs(),
            )
        finally:
            log_file.close()
    except Exception as exc:
        # M-38: do NOT inline drain+scan on spawn failure. Spawn failures occur
        # precisely when the system is unhealthy (fd exhaustion, OOM, disk
        # full); running maintenance synchronously here would block SessionStart
        # and delay recall injection at the worst possible time. The next
        # session retries maintenance via its own spawn.
        log.debug("Failed to spawn background maintenance; skipping (next session retries): %s", exc)


def _spawn_stale_scan() -> None:
    """Launch _scan_stale_sessions() in a detached background process.

    Uses ``Popen(start_new_session=True)`` so the scan does not block
    SessionStart.  The subprocess re-enters this module with ``--scan-stale``
    and performs the scan independently (issue #558).
    """
    import subprocess

    cmd = [sys.executable, "-m", "truememory.ingest.hooks.session_start", "--scan-stale"]
    try:
        _log_dir = Path.home() / ".truememory" / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _log_file = open(_log_dir / "stale_scan.log", "a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=_log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                **_platform.spawn_kwargs(),
            )
        finally:
            _log_file.close()
        log.debug("Spawned stale-session scanner (pid=%d)", proc.pid)
    except Exception as e:
        log.debug("Failed to spawn stale-session scanner: %s", e)


def main():
    if os.environ.get("TRUEMEMORY_EXTRACTION"):
        return

    args = _parse_args()

    # --scan-stale: background-only entry point (issue #558).
    if args.scan_stale:
        _scan_stale_sessions()
        return

    _run_maintenance_background()

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        input_data = {}

    # Issue #396: scale recall limit based on search intensity
    global MEMORY_LIMIT
    intensity = _get_search_intensity()
    MEMORY_LIMIT = _INTENSITY_MEMORY_LIMITS.get(intensity, _BASE_MEMORY_LIMIT)
    # Issue #636 (M-72): scale the recall payload budget with intensity so the
    # higher "max" memory limit isn't silently capped at 8KB.
    recall_budget = _intensity_budget(intensity)

    try:
        if _is_first_run():
            context = _first_run_context()
            recall_injected = False
        else:
            context = recall_memories(
                input_data, user_id=args.user, db_path=args.db, budget=recall_budget,
            )
            recall_injected = bool(context)

        # Check for available updates
        update_notice = _check_for_update()
        if update_notice:
            context = (context or "") + "\n\n" + update_notice

        # Prompt for email if not set yet
        email_notice = _check_email_needed()
        if email_notice:
            context = (context or "") + "\n\n" + email_notice

        if context:
            output = {"additionalContext": context}
            print(json.dumps(output))
            # Mark recall as injected so the first UserPromptSubmit can skip
            # its redundant per-message auto-recall (issue #561). Written only
            # after the context has actually been emitted, and only when the
            # recall portion was non-empty — an empty/failed recall must not
            # suppress the first prompt's targeted recall.
            if recall_injected:
                session_id = input_data.get("session_id", "")
                if session_id:
                    from truememory.ingest.hooks._shared import mark_recall_injected
                    mark_recall_injected(session_id)
    except Exception as e:
        log.error("SessionStart hook failed: %s", e)


def _first_run_context() -> str:
    lines = [
        "<truememory-first-run>",
        BANNER,
        "",
        SETUP_GUIDE,
        "</truememory-first-run>",
    ]
    return "\n".join(lines)


def _directive_scope_sql(user_id: str) -> tuple[str, list]:
    """WHERE-clause suffix + params for directive queries.

    Directives stored without a sender (``truememory_store`` defaults
    ``user_id=''``) must stay visible under ``--user`` scoping — they used to
    be silently hidden (issue #589, D-4).
    """
    if user_id:
        return " AND (sender = ? OR sender = '')", [user_id]
    return "", []


def _count_directives(memory, user_id: str = "") -> int:
    """Total stored directives in scope (cheap COUNT on idx_messages_directive)."""
    try:
        scope_sql, params = _directive_scope_sql(user_id)
        row = memory._engine.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE directive = 1" + scope_sql, params
        ).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        log.warning("Failed to count directives", exc_info=True)
        return 0


def _load_directives(memory, user_id: str = "") -> list[dict]:
    """Load directive memories for session injection, capped at DIRECTIVE_LIMIT.

    Failures are logged (not swallowed): a directive-column error on a
    half-migrated DB must not silently disable the feature (issue #589, D-4).
    """
    try:
        memory._engine._ensure_connection()
        query = "SELECT id, content, sender, timestamp, category FROM messages WHERE directive = 1"
        scope_sql, params = _directive_scope_sql(user_id)
        query += scope_sql
        # LIMIT cap+1 so truncation is detectable without an unbounded fetch.
        query += " ORDER BY id LIMIT ?"
        params.append(DIRECTIVE_LIMIT + 1)
        rows = memory._engine.conn.execute(query, params).fetchall()
        if len(rows) > DIRECTIVE_LIMIT:
            log.warning(
                "Directive injection capped at %d (more are stored). Prune "
                "stale directives with truememory_forget, or raise "
                "TRUEMEMORY_DIRECTIVE_LIMIT.",
                DIRECTIVE_LIMIT,
            )
            rows = rows[:DIRECTIVE_LIMIT]
        return [{"id": r[0], "content": r[1], "sender": r[2]} for r in rows]
    except Exception:
        log.warning("Failed to load directives for session injection", exc_info=True)
        return []


def _truncate_memory(content: str, memory_id, max_chars: int = 0) -> str:
    """Truncate *content* to *max_chars* on a word boundary.

    If the content is already within the limit it is returned unchanged.
    Otherwise it is sliced to the last whitespace boundary before *max_chars*
    and a pointer suffix is appended so the agent can retrieve the full text.
    """
    if max_chars <= 0:
        max_chars = RECALL_MEMORY_CHARS
    if len(content) <= max_chars:
        return content
    # Slice to the last space at or before the limit
    cut = content[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    suffix = f" [truncated, id={memory_id} — use truememory_get]"
    return cut.rstrip() + suffix


def _apply_budget(
    memory_lines: list[tuple[str, float]],
    directive_block: str,
    budget: int = 0,
) -> list[str]:
    """Enforce a total character budget on the additionalContext payload.

    *memory_lines* is a list of ``(formatted_line, score)`` tuples.
    *directive_block* is the already-formatted directive XML (exempt from
    truncation but counted against the budget).

    Returns the list of formatted memory lines that fit within *budget*.
    Lowest-score entries are dropped first.
    """
    if budget <= 0:
        budget = RECALL_BUDGET_CHARS
    # Directive block is mandatory — subtract its size from the available budget.
    available = budget - len(directive_block)
    if available <= 0:
        return []
    # The memory block has a fixed header/footer that wraps the lines.  Account
    # for a generous estimate so we don't exceed the budget by the wrapper text.
    _WRAPPER_OVERHEAD = 300  # header lines + closing tag
    available = max(0, available - _WRAPPER_OVERHEAD)

    # Sort by score ascending so we can drop cheapest first.
    indexed = list(enumerate(memory_lines))
    indexed.sort(key=lambda t: t[1][1])  # sort by score

    drop = set()
    total = sum(len(line) for line, _score in memory_lines)
    for idx, (line, _score) in indexed:
        if total <= available:
            break
        total -= len(line)
        drop.add(idx)

    return [line for i, (line, _score) in enumerate(memory_lines) if i not in drop]


def recall_memories(input_data: dict, user_id: str = "", db_path: str = "", budget: int = 0) -> str:
    """Search TrueMemory and format relevant memories for injection.

    Uses a file-based cache (issue #559): after running the 5 search
    queries, results are cached with a timestamp. Subsequent calls within
    the TTL (default 5 min, env TRUEMEMORY_RECALL_CACHE_TTL) return the
    cached context instead of re-querying the full search pipeline.
    Directives are always loaded fresh (cheap SQL, not cached).
    """
    try:
        from truememory import Memory
    except ImportError:
        return ""

    # Issue #577: arm a short per-request deadline for every model-server
    # call this hook makes. Under server contention (batch ingestion / MPS
    # OOM recovery) embeds previously stalled up to 120s each (5 serial
    # searches = 10 min worst case); with the deadline they fast-fail and
    # engine.search falls back to FTS-only retrieval.
    try:
        from truememory.ingest.hooks._shared import get_recall_deadline
        from truememory.model_client import set_request_timeout
        set_request_timeout(get_recall_deadline())
    except Exception:
        pass

    db = db_path or None
    memory = Memory(path=db) if db else Memory()

    # Load directives first — these are always injected (cheap SQL query,
    # not cached since they change infrequently and the query is fast).
    directives = _load_directives(memory, user_id=user_id)
    directive_ids = {d["id"] for d in directives}

    parts = []

    if directives:
        dir_lines = [
            "<truememory-directives>",
            "## User Directives (always loaded)",
            "These directives override defaults and apply to every session:",
            "",
        ]
        for d in directives:
            content = d.get("content", "").strip()
            if content:
                dir_lines.append(f"- {content}")
        if len(directives) >= DIRECTIVE_LIMIT:
            total = _count_directives(memory, user_id=user_id)
            if total > len(directives):
                dir_lines.append(
                    f"({len(directives)} of {total} directives shown — use "
                    "truememory_directives to view all, truememory_forget to "
                    "prune stale ones)"
                )
        dir_lines.append("</truememory-directives>")
        parts.append("\n".join(dir_lines))

    # --- Issue #559: cache-TTL for the 5 recall queries ---
    from truememory.ingest.hooks._shared import get_recall_cache, set_recall_cache

    cached = get_recall_cache(db_path or "", user_id)
    if cached is not None:
        parts.append(cached)
        return "\n\n".join(parts) if parts else ""

    queries = [
        "user preferences favorites likes dislikes",
        "personal facts name location job role",
        "recent decisions and commitments",
        "corrections and updates to prior information",
        "relationships family friends coworkers",
    ]

    per_query_limit = max(1, MEMORY_LIMIT // len(queries))

    all_results = []
    seen_ids = set(directive_ids)
    seen_content = set()

    for query in queries:
        added_this_query = 0
        try:
            results = memory._engine.search(query, limit=per_query_limit * 3, _skip_reranker=True)
            if user_id:
                results = [r for r in results if r.get("sender", "") == user_id]

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

    recall_context = ""
    if all_results:
        # -- Issue #578: per-memory truncation + total payload budget ----------
        directive_block = parts[0] if parts else ""
        memory_lines: list[tuple[str, float]] = []
        for r in all_results[:MEMORY_LIMIT]:
            content = r.get("content", "").strip()
            if not content:
                continue
            truncated = _truncate_memory(content, r.get("id", "?"))
            score = r.get("score", 0.0)
            memory_lines.append((f"- {truncated}", score))

        kept = _apply_budget(memory_lines, directive_block, budget=budget)

        if kept:
            lines = [
                "<truememory-context>",
                "## TrueMemory — What You Know About This User",
                "These are facts from TrueMemory (the primary long-horizon memory system).",
                "Use these to answer user questions. Search TrueMemory for more if needed.",
                "",
            ]
            lines.extend(kept)
            lines.append("</truememory-context>")
            recall_context = "\n".join(lines)
            parts.append(recall_context)

    # Cache the recall portion (not directives) for subsequent calls
    set_recall_cache(recall_context if all_results else "", db_path or "", user_id)

    return "\n\n".join(parts) if parts else ""


if __name__ == "__main__":
    main()
