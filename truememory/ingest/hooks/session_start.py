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

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

log = logging.getLogger(__name__)

MEMORY_LIMIT = int(os.environ.get("TRUEMEMORY_RECALL_LIMIT", "25"))
# Max directives force-injected at session start. Uncapped injection let a
# large directive set consume unbounded context (issue #589, D-4).
DIRECTIVE_LIMIT = int(os.environ.get("TRUEMEMORY_DIRECTIVE_LIMIT", "50"))
# Per-memory character cap and total payload budget (issue #578).
# Memories exceeding the per-entry cap are sliced on a word boundary and
# suffixed with a pointer so the agent can fetch the full text on demand.
RECALL_MEMORY_CHARS = int(os.environ.get("TRUEMEMORY_RECALL_MEMORY_CHARS", "500"))
RECALL_BUDGET_CHARS = int(os.environ.get("TRUEMEMORY_RECALL_BUDGET_CHARS", "8192"))
ONBOARDED_MARKER = Path.home() / ".truememory" / ".onboarded"
BACKLOG_DIR = Path.home() / ".truememory" / "backlog"
_DRAIN_CAP = 3
_SCAN_MARKER = Path.home() / ".truememory" / ".last_stale_scan"
_SCAN_INTERVAL = 900  # 15 minutes
_SCAN_CAP = 3  # max sessions to queue per scan

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

    from truememory.ingest.hooks._shared import cleanup_stale_processing, check_extraction_budget, record_stale_processing_pid
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

        try:
            data = json.loads(claimed_path.read_text(encoding="utf-8"))
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
                        start_new_session=hasattr(os, 'setsid'),
                    )
                finally:
                    _log_file.close()
                register_spawned_pid(proc.pid)
                record_stale_processing_pid(claimed_path, proc.pid)
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


def _scan_stale_sessions() -> None:
    """Find transcripts from recent sessions that were never extracted.

    Runs at most once per _SCAN_INTERVAL. Scans Claude Code's project
    directories for session transcripts modified in the last 24 hours
    that have no corresponding extraction marker. Skips extraction noise
    transcripts (identified by sentinel tag or legacy prompt prefixes).
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
        try:
            if _SCAN_MARKER.exists():
                if time.time() - _SCAN_MARKER.stat().st_mtime < _SCAN_INTERVAL:
                    return
            os.lseek(scan_fd, 0, os.SEEK_SET)
            os.ftruncate(scan_fd, 0)
            os.write(scan_fd, str(time.time()).encode("utf-8"))
        except OSError:
            return

        claude_dir = Path.home() / ".claude" / "projects"
        if not claude_dir.exists():
            return

        from truememory.ingest.hooks._shared import EXTRACTED_DIR, _safe_session_id, mark_session_extracted
        from truememory.ingest.hooks.stop import _queue_to_backlog

        uuid_re = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
        cutoff = time.time() - 86400
        queued = 0
        skipped_noise = 0

        for project_dir in claude_dir.iterdir():
            if not project_dir.is_dir():
                continue
            for transcript in project_dir.iterdir():
                if transcript.suffix != ".jsonl":
                    continue
                if not transcript.is_file():
                    continue
                session_id = transcript.stem
                if not uuid_re.match(session_id):
                    continue
                try:
                    stat = transcript.stat()
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

                if _is_extraction_transcript(transcript):
                    try:
                        mark_session_extracted(session_id, str(transcript))
                    except Exception:
                        pass
                    skipped_noise += 1
                    continue

                _queue_to_backlog(
                    str(transcript), session_id, "", "",
                    reason="stale_session_recovery",
                )
                queued += 1
                if queued >= _SCAN_CAP:
                    break
            if queued >= _SCAN_CAP:
                break

        if queued > 0:
            log.info("Stale session scanner: queued %d unextracted sessions", queued)
        if skipped_noise > 0:
            log.info("Stale session scanner: skipped %d extraction noise transcripts", skipped_noise)
    finally:
        try:
            os.close(scan_fd)
        except OSError:
            pass


def main():
    if os.environ.get("TRUEMEMORY_EXTRACTION"):
        return

    args = _parse_args()

    _drain_backlog()
    _scan_stale_sessions()

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        input_data = {}

    try:
        if _is_first_run():
            context = _first_run_context()
            recall_injected = False
        else:
            context = recall_memories(input_data, user_id=args.user, db_path=args.db)
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


def recall_memories(input_data: dict, user_id: str = "", db_path: str = "") -> str:
    """Search TrueMemory and format relevant memories for injection."""
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

    # Load directives first — these are always injected
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

        kept = _apply_budget(memory_lines, directive_block)

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
            parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else ""


if __name__ == "__main__":
    main()
