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


_EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+(?:\.[A-Za-z]{2,10}(?![A-Za-z])){1,3}', re.ASCII)

_INJECTION_RE = re.compile(
    r'\b(?:DROP|SELECT|INSERT|DELETE|UPDATE|UNION|ALTER|EXEC)\b'
    r'|[`{};]'
    r'|--\s',
    re.IGNORECASE,
)

_INTENT_RE = re.compile(
    r'(?:^|\b)(?:'
    r'my\s+email\s+(?:is|address\s+is)\s+'
    r'|email\s*:\s*'
    r'|reach\s+me\s+at\s+'
    r'|contact\s+me\s+at\s+'
    r"|i(?:'m|\s+am)\s+at\s+"
    r')',
    re.IGNORECASE,
)

_TRIVIAL_WORDS = frozenset({
    'yeah', 'yep', 'yes', 'sure', 'ok', 'okay',
    'here', "here's", 'its', "it's",
    'please', 'thanks', 'thx', 'hi', 'hey',
})

_RECALL_RE = re.compile(
    r'\b(?:'
    r'what(?:\'s|\s+is|\s+was|\s+are|\s+were|\s+did|\s+do)\b'
    r'|who\s+(?:is|was|did)\b'
    r'|when\s+(?:is|was|did)\b'
    r'|where\s+(?:is|was|did|does)\b'
    r'|do\s+you\s+remember\b'
    r'|can\s+you\s+recall\b'
    r'|remind\s+me\b'
    r'|what\'s\s+my\b'
    r'|what\s+do\s+I\b'
    r'|did\s+(?:we|i|you)\b'
    r'|have\s+(?:we|i)\s+(?:ever|already)\b'
    r'|you\s+(?:told|said|mentioned)\b'
    r'|(?:we|i)\s+(?:discussed|decided|agreed)\b'
    r'|last\s+(?:time|session|conversation)\s+we\b'
    r'|(?:earlier|previously)\s+(?:you|we|i)\b'
    r'|yesterday\s+(?:you|we|i)\b'
    r'|previous\s+(?:session|conversation|chat)\b'
    r'|my\s+(?:favorite|preferred|usual)\b'
    r')',
    re.IGNORECASE,
)

_CODE_RE = re.compile(
    r'\b(?:function|class|def|import|const|let|var|return|console\.log|print\(|TypeError|SyntaxError)\b'
    r'|```'
    r'|(?:what\s+does\s+(?:this|the)\s+(?:function|code|class|method)\b)',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Issue #396: Memory Intensity — per-exchange evaluator + proactive recall
# ---------------------------------------------------------------------------

# Storable content heuristics for per-exchange evaluator
_STORABLE_RE = re.compile(
    r'\b(?:'
    r'(?:i|my|we)\s+(?:prefer|like|dislike|hate|love|use|always|never|want)\b'
    r'|(?:i|my)\s+(?:name|email|phone|address|birthday|age|job|role|team)\s+(?:is|are)\b'
    r"|(?:i(?:'m|\s+am)\s+)"
    r'|(?:we|i)\s+(?:decided|agreed|committed|chose|picked)\b'
    r'|(?:actually|no|wrong|correction|not\s+right|instead)\b'
    r'|(?:remember\s+(?:that|this|to))\b'
    r'|(?:from\s+now\s+on|always\s+do|never\s+do|every\s+session)\b'
    r'|(?:my\s+(?:favorite|preferred|usual|default))\b'
    r')',
    re.IGNORECASE,
)

# Prompt counter file for proactive recall scheduling
_PROMPT_COUNTER_DIR = Path(os.environ.get(
    "TRUEMEMORY_BUFFER_DIR",
    str(Path.home() / ".truememory" / "buffers"),
))


def _get_intensity_config() -> tuple[str, str]:
    """Read search_intensity and store_intensity from persistent config."""
    try:
        config_path = Path.home() / ".truememory" / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            return (
                config.get("search_intensity", "standard"),
                config.get("store_intensity", "standard"),
            )
    except Exception:
        pass
    return ("standard", "standard")


def _get_prompt_count(session_id: str) -> int:
    """Read the prompt counter for a session."""
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64] or "unknown"
    counter_file = _PROMPT_COUNTER_DIR / f"{safe_id}.count"
    try:
        if counter_file.exists():
            return int(counter_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        pass
    return 0


def _increment_prompt_count(session_id: str) -> int:
    """Increment and return the prompt counter for a session."""
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64] or "unknown"
    counter_file = _PROMPT_COUNTER_DIR / f"{safe_id}.count"
    count = _get_prompt_count(session_id) + 1
    try:
        _PROMPT_COUNTER_DIR.mkdir(parents=True, exist_ok=True)
        counter_file.write_text(str(count), encoding="utf-8")
    except OSError:
        pass
    return count


def _detect_storable_content(prompt: str) -> bool:
    """Fast heuristic: does the prompt contain content worth storing?"""
    if len(prompt) < 15 or len(prompt) > 2000:
        return False
    if _CODE_RE.search(prompt):
        return False
    return bool(_STORABLE_RE.search(prompt))


def _check_conversation_depth(session_id: str, prompt: str, window: int = 5) -> bool:
    """Check if recent exchanges show thematic depth (keyword overlap).

    Reads the last `window` buffer entries and checks if the current prompt
    shares significant keywords with them, suggesting a sustained topic
    worth capturing.
    """
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64] or "unknown"
    buffer_file = _PROMPT_COUNTER_DIR.parent / "buffers" / f"{safe_id}.jsonl"
    if not buffer_file.exists():
        buffer_file = Path(os.environ.get(
            "TRUEMEMORY_BUFFER_DIR",
            str(Path.home() / ".truememory" / "buffers"),
        )) / f"{safe_id}.jsonl"
    if not buffer_file.exists():
        return False

    try:
        recent_words: set[str] = set()
        lines = buffer_file.read_text(encoding="utf-8").strip().splitlines()
        for line in lines[-window:]:
            try:
                entry = json.loads(line)
                content = entry.get("content", "")
                words = {w.lower() for w in re.findall(r'\b\w{4,}\b', content)}
                recent_words.update(words)
            except (json.JSONDecodeError, KeyError):
                continue

        prompt_words = {w.lower() for w in re.findall(r'\b\w{4,}\b', prompt)}
        overlap = prompt_words & recent_words
        # At least 3 shared meaningful words suggests thematic depth
        return len(overlap) >= 3
    except Exception:
        return False


def _try_per_exchange_store(prompt: str, session_id: str, user_id: str, db_path: str, store_intensity: str) -> None:
    """Per-exchange evaluator: detect storable content and store via encoding gate.

    Enhanced: only stores if conversation has depth (keyword overlap).
    Max: stores on every exchange that passes heuristics + gate.
    """
    if store_intensity == "standard":
        return

    if not _detect_storable_content(prompt):
        return

    # Enhanced: require conversation depth; Max: skip depth check
    if store_intensity == "enhanced" and not _check_conversation_depth(session_id, prompt):
        return

    # Novelty check: quick search to avoid dupes
    try:
        from truememory.client import Memory
        m = Memory(path=db_path or None)
        existing = m.search(prompt, user_id=user_id or None, limit=3)
        if existing:
            for r in existing:
                score = r.get("score", 0.0)
                if score > 0.85:
                    return  # Too similar to existing memory
    except Exception:
        return  # If search fails, skip storing

    # Run through encoding gate
    try:
        from truememory.ingest.encoding_gate import EncodingGate
        gate = EncodingGate(memory=m, user_id=user_id or "")
        decision = gate.evaluate(prompt)
        if not decision.passed:
            return

        # Store the content
        m.add(content=prompt, user_id=user_id or None)
    except Exception:
        pass  # Never crash the hook


def _try_proactive_recall(prompt: str, user_id: str, db_path: str, session_id: str, search_intensity: str, prompt_count: int) -> str | None:
    """Proactive recall based on search intensity.

    Enhanced: recall every ~5th prompt (8 results).
    Max: every prompt gets a memory search (10 results).
    Standard: handled by existing _try_auto_recall (recall-intent only).
    """
    if search_intensity == "standard":
        return None

    # Dedup: skip if session_start already injected recall for this session
    try:
        from truememory.ingest.hooks._shared import consume_recall_injected
        if session_id and consume_recall_injected(session_id):
            return None
    except Exception:
        pass

    if search_intensity == "enhanced":
        # Every 5th prompt
        if prompt_count % 5 != 0:
            return None
        limit = 8
    else:  # max
        limit = 10

    # Skip code-heavy prompts
    if _CODE_RE.search(prompt):
        return None

    try:
        try:
            from truememory.ingest.hooks._shared import get_recall_deadline
            from truememory.model_client import set_request_timeout
            set_request_timeout(get_recall_deadline())
        except Exception:
            pass
        from truememory.client import Memory
        m = Memory(path=db_path or None)
        results = m.search(prompt, user_id=user_id or None, limit=limit)
        if not results:
            return None
        lines = []
        for r in results[:limit]:
            content = r.get("content", "")[:200]
            lines.append(f"- {content}")
        return (
            "<truememory-recall>\n"
            "Proactive memory recall:\n"
            + "\n".join(lines)
            + "\n</truememory-recall>"
        )
    except Exception:
        return None


def _detect_recall(prompt: str) -> bool:
    if len(prompt) < 10 or len(prompt) > 500:
        return False
    if _CODE_RE.search(prompt):
        return False
    return bool(_RECALL_RE.search(prompt))


def _try_auto_recall(prompt: str, user_id: str, db_path: str, session_id: str = "") -> str | None:
    """Search TrueMemory if prompt looks like a recall question.

    Skips the search entirely on the first prompt right after SessionStart,
    which already injected recall (issue #561). The gate runs before detection
    and the Memory load so the redundant first-message recall costs nothing.
    """
    from truememory.ingest.hooks._shared import consume_recall_injected
    if session_id and consume_recall_injected(session_id):
        return None
    if not _detect_recall(prompt):
        return None
    try:
        # Issue #577: short model-server deadline so a contended server
        # fast-fails and the search falls back to FTS-only instead of
        # stalling this hook for up to 120s.
        try:
            from truememory.ingest.hooks._shared import get_recall_deadline
            from truememory.model_client import set_request_timeout
            set_request_timeout(get_recall_deadline())
        except Exception:
            pass
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
    """If the user typed their email and config has no email, save it."""
    try:
        config_path = Path.home() / ".truememory" / "config.json"
        if not config_path.exists():
            return
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("email"):
            return

        stripped = prompt.strip()
        email = None

        m = _EMAIL_RE.fullmatch(stripped)
        if m:
            email = m.group(0)

        if email is None and len(stripped) < 80:
            em = _EMAIL_RE.search(stripped)
            if em:
                remainder = stripped[:em.start()] + stripped[em.end():]
                words = re.sub(r'[,.\s!?:]+', ' ', remainder).strip().lower().split()
                if all(w in _TRIVIAL_WORDS for w in words):
                    email = em.group(0)

        if email is None:
            if len(prompt) > 200:
                return
            if _INTENT_RE.search(prompt):
                if _INJECTION_RE.search(prompt):
                    return
                em = _EMAIL_RE.search(prompt)
                if em:
                    email = em.group(0)

        if email is None:
            return

        config["email"] = email
        tmp = config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
        tmp.rename(config_path)
    except Exception:
        pass


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
        # A too-short first prompt is still the session's first prompt:
        # consume any recall marker now so it cannot strand and debounce the
        # next, real prompt (issue #561).
        try:
            from truememory.ingest.hooks._shared import consume_recall_injected
            consume_recall_injected(session_id)
        except Exception:
            pass
        return

    try:
        buffer_message(session_id, prompt)
        _prune_old_buffers()
    except Exception:
        pass  # Never crash the hook

    _try_capture_email(prompt)

    # Issue #396: read intensity settings
    search_intensity, store_intensity = _get_intensity_config()

    # Issue #396: track prompt count for proactive recall scheduling
    prompt_count = _increment_prompt_count(session_id)

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
                    # Only mark extracted on a real spawn; pid==0 means the
                    # session was queued to the backlog and must stay eligible
                    # so it is not silently dropped (see #400).
                    if spawned_pid > 0:
                        mark_session_extracted(session_id, transcript_path, spawned_pid=spawned_pid)
        except Exception:
            pass

    # Issue #396: per-exchange store for enhanced/max store intensity
    try:
        _try_per_exchange_store(prompt, session_id, args.user, args.db, store_intensity)
    except Exception:
        pass  # Never crash the hook

    # Issue #396: proactive recall for enhanced/max search intensity
    recall_context = None
    if search_intensity != "standard":
        recall_context = _try_proactive_recall(
            prompt, args.user, args.db, session_id, search_intensity, prompt_count,
        )

    # Fall back to standard recall-intent detection
    if not recall_context:
        recall_context = _try_auto_recall(prompt, args.user, args.db, session_id)

    if recall_context:
        print(json.dumps({"additionalContext": recall_context}))


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
