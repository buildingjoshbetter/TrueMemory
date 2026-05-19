#!/usr/bin/env python3
"""
UserPromptSubmit Hook — Buffer + Recall + Turn-Based Injection
==============================================================

Fires on every user message submission. Five things happen:

1. **Buffer write** — appends a one-line JSON record to a per-session
   buffer (defensive snapshot in case the transcript file corrupts).
2. **Email capture** — opportunistic scrape of the user's email into
   ``~/.truememory/config.json`` when first observed.
3. **Background-ingestion trigger** — when transcript has ≥10 user
   messages and isn't already mid-extraction, spawns the cold-path
   extractor (mirrors SessionEnd, catches sessions that don't terminate
   cleanly).
4. **Explicit-recall injection** (``_try_auto_recall``) — regex-gated
   path that fires when the prompt matches a question pattern
   (``"what's"``, ``"do you remember"``, etc.). Searches TrueMemory
   with the raw prompt, emits a ``<truememory-recall>`` block.
5. **Turn-based injection** (``_try_turn_based_injection``) — once per
   session, when conversation has signal: ``turns >= 13`` OR
   ``len(prompt) > 333``. Uses the last K user turns as a richer query,
   runs ``Memory.search_deep`` with HyDE expansion (Claude CLI llm_fn),
   emits a ``<truememory-context>`` block. Marker file dedupes re-fires.

Both injection paths can emit on the same turn; combined output goes
out as a single ``additionalContext`` JSON line to avoid double-emit.

Input (stdin JSON):
    {"session_id": "...", "prompt": "...", "transcript_path": "..."}

Output (stdout): either nothing, or a single JSON line with
    ``additionalContext`` containing the concatenated recall blocks.
"""

import argparse
import json
import os
import re
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


# ---------------------------------------------------------------------------
# Turn-based memory injection — once-per-session targeted recall
# ---------------------------------------------------------------------------

# Fire injection when user turns reach this count (whichever trigger first wins).
INJECT_AFTER_TURNS = int(os.environ.get("TRUEMEMORY_INJECT_AFTER_TURNS", "13"))
# Fire injection when current prompt exceeds this character count (one substantive prompt is enough signal).
INJECT_AFTER_CHARS = int(os.environ.get("TRUEMEMORY_INJECT_AFTER_CHARS", "333"))
# Number of recent user turns to concatenate as the search query (richer semantic context than raw prompt alone).
INJECT_QUERY_TURNS = int(os.environ.get("TRUEMEMORY_INJECT_QUERY_TURNS", "6"))
# Per-turn truncation when joining turns into the query — prevents 100K-char pasted code from blowing up the embedding input.
INJECT_QUERY_TURN_CHARS = int(os.environ.get("TRUEMEMORY_INJECT_QUERY_TURN_CHARS", "500"))
# How many memories to fetch and inject.
INJECT_RECALL_LIMIT = int(os.environ.get("TRUEMEMORY_INJECT_RECALL_LIMIT", "15"))
# Per-bullet truncation in the emitted block (200 in the existing recall path; bumped to 300 for the once-per-session budget).
INJECT_BULLET_CHARS = int(os.environ.get("TRUEMEMORY_INJECT_BULLET_CHARS", "300"))
# Kill switch — set TRUEMEMORY_INJECT_DISABLED=1 to disable turn-based injection entirely.
INJECT_DISABLED = os.environ.get("TRUEMEMORY_INJECT_DISABLED", "").lower() in ("1", "true", "yes")


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


_EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')

_RECALL_RE = re.compile(
    r'\b(?:what(?:\'s|\s+is|\s+was|\s+are|\s+were|\s+did|\s+do)\b'
    r'|who\s+(?:is|was|did)\b'
    r'|when\s+(?:is|was|did)\b'
    r'|where\s+(?:is|was|did|does)\b'
    r'|do\s+you\s+remember\b'
    r'|did\s+(?:we|i|you)\b'
    r'|what\'s\s+my\b'
    r'|what\s+do\s+I\b'
    r'|remind\s+me\b'
    r'|have\s+(?:we|i)\s+(?:ever|already)\b'
    r'|my\s+(?:favorite|preferred|usual)\b)',
    re.IGNORECASE,
)

_CODE_RE = re.compile(
    r'\b(?:function|class|def|import|const|let|var|return|console\.log|print\(|TypeError|SyntaxError)\b'
    r'|```'
    r'|(?:what\s+does\s+(?:this|the)\s+(?:function|code|class|method)\b)',
    re.IGNORECASE,
)


def _detect_recall(prompt: str) -> bool:
    if len(prompt) < 10 or len(prompt) > 500:
        return False
    if _CODE_RE.search(prompt):
        return False
    return bool(_RECALL_RE.search(prompt))


def _try_auto_recall(prompt: str, user_id: str, db_path: str) -> str | None:
    """Search TrueMemory if prompt looks like a recall question."""
    if not _detect_recall(prompt):
        return None
    try:
        from truememory.client import Memory
        m = Memory(path=db_path or None)
        results = m.search(prompt, user_id=user_id or None, limit=5)
        if not results:
            return None
        lines = []
        for r in results[:5]:
            content = r.get("content", "")[:200]
            lines.append(f"- {content}")
        return (
            "<truememory-recall>\n"
            "Relevant memories for this question:\n"
            + "\n".join(lines)
            + "\n</truememory-recall>"
        )
    except Exception:
        return None


def _try_capture_email(prompt: str) -> None:
    """If the user typed an email and config has no email, save it."""
    try:
        config_path = Path.home() / ".truememory" / "config.json"
        if not config_path.exists():
            return
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("email"):
            return
        match = _EMAIL_RE.search(prompt)
        if not match:
            return
        email = match.group(0)
        config["email"] = email
        tmp = config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
        tmp.rename(config_path)
    except Exception:
        pass


def _build_turn_based_query(transcript_path: str, k: int) -> str:
    """Return the last ``k`` user turns concatenated as a single search query.

    Reads the transcript JSONL, collects user-role entries' content,
    truncates each turn to ``INJECT_QUERY_TURN_CHARS`` so a single pasted
    code block can't blow the embedding input to 100K chars, joins with
    ``"\\n---\\n"`` so the vector encoder sees them as distinct passages.
    Returns ``""`` if the transcript can't be parsed.
    """
    try:
        content = Path(transcript_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if not content.strip():
        return ""

    entries: list[str] = []
    # Try JSON array first
    try:
        if content.lstrip().startswith("["):
            data = json.loads(content)
            if isinstance(data, list):
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    role = entry.get("type") or entry.get("role") or ""
                    if role not in ("human", "user"):
                        continue
                    msg = entry.get("content") or entry.get("message") or ""
                    if isinstance(msg, list):  # Claude Code: content blocks
                        msg = " ".join(
                            b.get("text", "") for b in msg
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    if isinstance(msg, str) and msg.strip():
                        entries.append(msg.strip()[:INJECT_QUERY_TURN_CHARS])
    except json.JSONDecodeError:
        pass

    if not entries:
        # JSONL fallback
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            role = entry.get("type") or entry.get("role") or ""
            if role not in ("human", "user"):
                continue
            msg = entry.get("content") or entry.get("message") or ""
            if isinstance(msg, dict):
                msg = msg.get("content") or msg.get("text") or ""
            if isinstance(msg, list):
                msg = " ".join(
                    b.get("text", "") for b in msg
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if isinstance(msg, str) and msg.strip():
                entries.append(msg.strip()[:INJECT_QUERY_TURN_CHARS])

    if not entries:
        return ""

    return "\n---\n".join(entries[-k:])


def _try_turn_based_injection(
    prompt: str,
    session_id: str,
    transcript_path: str,
    user_id: str,
    db_path: str,
) -> str | None:
    """One-shot per session: when the conversation has signal, search with
    the last K user turns as a richer query and emit a
    ``<truememory-context>`` block. Marker file prevents re-fire.
    """
    if INJECT_DISABLED:
        return None
    if not session_id or session_id == "unknown":
        return None
    if not transcript_path:
        return None

    try:
        from truememory.ingest.hooks._shared import (
            already_injected, mark_injected, count_user_turns,
        )
    except ImportError:
        return None

    if already_injected(session_id):
        return None

    # Gate — length first (cheap), then turn count (transcript I/O)
    if len(prompt) > INJECT_AFTER_CHARS:
        trigger = "length"
    else:
        try:
            turns = count_user_turns(transcript_path)
        except Exception:
            return None
        if turns < INJECT_AFTER_TURNS:
            return None
        trigger = "turns"

    query = _build_turn_based_query(transcript_path, INJECT_QUERY_TURNS)
    if not query.strip():
        return None

    # search_deep with llm_fn — HyDE expansion on Pro tier (Claude CLI).
    # Gracefully degrades if mcp_server / Memory not importable in this env.
    try:
        from truememory.client import Memory
        from truememory.mcp_server import _get_llm_fn  # singleton, cached
        m = Memory(path=db_path or None)
        llm_fn = _get_llm_fn()
        results = m.search_deep(
            query, user_id=user_id or None,
            limit=INJECT_RECALL_LIMIT, llm_fn=llm_fn,
        )
    except Exception:
        return None

    if not results:
        # Still mark — don't re-pay the search cost on every subsequent prompt
        mark_injected(session_id, {
            "trigger": trigger, "query_chars": len(query), "n_results": 0,
        })
        return None

    lines = [
        f"- {(r.get('content') or '')[:INJECT_BULLET_CHARS]}"
        for r in results[:INJECT_RECALL_LIMIT]
    ]
    out = (
        "<truememory-context>\n"
        f"Relevant memories (turn-based injection, trigger={trigger}):\n"
        + "\n".join(lines)
        + "\n</truememory-context>"
    )

    mark_injected(session_id, {
        "trigger": trigger,
        "query_chars": len(query),
        "n_results": len(results),
    })
    return out


def main():
    if os.environ.get("TRUEMEMORY_EXTRACTION"):
        return

    args = _parse_args()

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

    _try_capture_email(prompt)

    transcript_path = input_data.get("transcript_path", "")
    if transcript_path and Path(transcript_path).exists():
        try:
            from truememory.ingest.hooks._shared import should_extract_session, mark_session_extracted
            if should_extract_session(session_id, transcript_path):
                from truememory.ingest.hooks.stop import (
                    _has_enough_messages, _run_background_ingestion,
                    TRACE_DIR, LOG_DIR,
                )
                TRACE_DIR.mkdir(parents=True, exist_ok=True)
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                if _has_enough_messages(transcript_path, 10):
                    spawned_pid = _run_background_ingestion(
                        transcript_path, session_id, args.user, args.db,
                    )
                    mark_session_extracted(session_id, transcript_path, spawned_pid=spawned_pid)
        except Exception:
            pass

    # Two injection paths, both can fire on the same turn. We MUST emit at most
    # one `additionalContext` JSON line — Claude Code reads exactly one stdout
    # JSON from the hook, so concat both blocks if both fire.
    recall_context = _try_auto_recall(prompt, args.user, args.db)
    turn_context = _try_turn_based_injection(
        prompt, session_id, transcript_path, args.user, args.db,
    )
    if recall_context and turn_context:
        combined = recall_context + "\n\n" + turn_context
        print(json.dumps({"additionalContext": combined}))
    elif recall_context:
        print(json.dumps({"additionalContext": recall_context}))
    elif turn_context:
        print(json.dumps({"additionalContext": turn_context}))


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
