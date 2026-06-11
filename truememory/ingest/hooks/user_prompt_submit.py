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


def _safe_session_id_local(session_id: str) -> str:
    """Sanitize a session_id for use in buffer/counter filenames.

    Single source of truth so the prompt-counter, conversation-depth, and
    buffer paths all derive the SAME on-disk name for a given session (issue
    #635, M-73) — they previously inlined the same expression in three places,
    risking drift. Mirrors ``buffer_message``'s scheme (alnum + ``-_``, 64-char
    cap, ``unknown`` fallback).
    """
    return "".join(c for c in session_id if c.isalnum() or c in "-_")[:64] or "unknown"


def _word_overlap(a: str, b: str) -> float:
    """Scale-free Jaccard similarity on word sets (issue #632).

    Used as the novelty fallback when retrieval scores are relative/fused
    (FTS-only / degraded embedder) rather than absolute cosine similarity,
    so the 0.85 cutoff cannot be trusted as a true cosine value.
    """
    words_a = set(re.findall(r"\w+", a.lower()))
    words_b = set(re.findall(r"\w+", b.lower()))
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


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

# Storable content heuristics for per-exchange evaluator.
#
# Issue #635 (M-40): the previous pattern used bare standalone-word
# alternations ("no"/"actually"/"instead"/"I'm") with no awareness of
# negation, questions, quotes, or code, flagging 17/19 adversarial filler
# cases. Tightened here to clause-level anchors that need a real subject +
# preference/fact predicate:
#   - bare "no" is dropped entirely (it is overwhelmingly filler/answer);
#   - "actually"/"instead"/"wrong" only fire as a *correction* of a stated
#     fact, anchored to a following first-person clause or a copula, not as
#     standalone discourse markers;
#   - "I'm" only fires when followed by an identity/state predicate
#     ("I'm a ...", "I'm using ...", "I'm based in ..."), not bare "I'm".
# Interrogatives and quoted spans are stripped before matching in
# _detect_storable_content() so a question or a quote that merely *contains*
# these tokens does not trigger the expensive store path.
_STORABLE_RE = re.compile(
    r'\b(?:'
    r'(?:i|my|we)\s+(?:prefer|like|dislike|hate|love|use|always|never|want)\b'
    r'|(?:i|my)\s+(?:name|email|phone|address|birthday|age|job|role|team)\s+(?:is|are)\b'
    r"|i(?:'m|\s+am)\s+(?:a|an|the|using|on|in|at|based|from|currently|now|working|building)\b"
    r'|(?:we|i)\s+(?:decided|agreed|committed|chose|picked)\b'
    r"|(?:actually|instead|correction)\b[^?]*?\b(?:i|my|we|it(?:'s|\s+is)|that(?:'s|\s+is))\b"
    r"|(?:that(?:'s|\s+is)|it(?:'s|\s+is)|you(?:'re|\s+are))\s+(?:wrong|not\s+right|incorrect)\b"
    r'|(?:remember\s+(?:that|this|to))\b'
    r'|(?:from\s+now\s+on|always\s+do|never\s+do|every\s+session)\b'
    r'|(?:my\s+(?:favorite|preferred|usual|default))\b'
    r')',
    re.IGNORECASE,
)

# Strips double/single-quoted spans so a quoted token inside an otherwise
# non-storable prompt cannot trigger the store path (issue #635, M-40).
_QUOTED_SPAN_RE = re.compile(r'"[^"]*"' r"|'[^']*'" r'|`[^`]*`')

# Prompts that OPEN with a question word are interrogatives (asks), even when
# they lack a trailing "?" ("do you prefer tabs", "what is my default")
# (issue #635, M-40).
_INTERROGATIVE_RE = re.compile(
    r'^(?:do|does|did|can|could|would|will|are|is|was|were|have|has|'
    r'what|who|when|where|why|how|which)\b',
    re.IGNORECASE,
)

# Prompt counter file for proactive recall scheduling
_PROMPT_COUNTER_DIR = Path(os.environ.get(
    "TRUEMEMORY_BUFFER_DIR",
    str(Path.home() / ".truememory" / "buffers"),
))


# Issue #636 (M-16): the only valid intensity values. Any other value —
# "MAX", "Enhanced", garbage, JSON null — must NOT silently enable the most
# expensive every-prompt mode. Readers normalize (lowercase + allowlist) and
# fall back to "standard" so an unrecognized config value fails CLOSED.
_VALID_INTENSITIES = frozenset({"standard", "enhanced", "max"})


def _normalize_intensity(value: object) -> str:
    """Normalize a raw config intensity value to a known level (issue #636).

    Lowercases strings and validates against the allowlist; any non-string or
    unrecognized value (None/null, "MAX", "garbage", numbers) falls back to
    "standard". This makes invalid config fail closed to the cheapest mode
    instead of the exclusion-based dispatch failing open to "max".
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _VALID_INTENSITIES:
            return normalized
    return "standard"


def _get_intensity_config() -> tuple[str, str]:
    """Read search_intensity and store_intensity from persistent config.

    Values are normalized through the allowlist (issue #636) so invalid or
    mismatched-case config cannot enable an unintended intensity mode.
    """
    try:
        config_path = Path.home() / ".truememory" / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            return (
                _normalize_intensity(config.get("search_intensity", "standard")),
                _normalize_intensity(config.get("store_intensity", "standard")),
            )
    except Exception:
        pass
    return ("standard", "standard")


def _get_prompt_count(session_id: str) -> int:
    """Read the prompt counter for a session."""
    safe_id = _safe_session_id_local(session_id)
    counter_file = _PROMPT_COUNTER_DIR / f"{safe_id}.count"
    try:
        if counter_file.exists():
            return int(counter_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        pass
    return 0


def _increment_prompt_count(session_id: str) -> int:
    """Increment and return the prompt counter for a session.

    Issue #635 (M-73): the read-modify-write is serialized with ``flock`` so
    two overlapping hook invocations for the same session cannot both read N
    and both write N+1 (losing a tick, which would skew the every-5th proactive
    recall cadence). Falls back to the unlocked path on Windows / lock failure
    rather than crashing the hook.
    """
    safe_id = _safe_session_id_local(session_id)
    counter_file = _PROMPT_COUNTER_DIR / f"{safe_id}.count"
    try:
        _PROMPT_COUNTER_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    if not _HAS_FCNTL:
        count = _get_prompt_count(session_id) + 1
        try:
            counter_file.write_text(str(count), encoding="utf-8")
        except OSError:
            pass
        return count

    fd = -1
    try:
        fd = os.open(str(counter_file), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            pass
        try:
            raw = os.read(fd, 64).decode("utf-8", errors="replace").strip()
            current = int(raw) if raw else 0
        except (ValueError, OSError):
            current = 0
        count = current + 1
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, str(count).encode("utf-8"))
        except OSError:
            pass
        return count
    except OSError:
        # Could not open the counter under lock — fall back to best-effort.
        count = _get_prompt_count(session_id) + 1
        try:
            counter_file.write_text(str(count), encoding="utf-8")
        except OSError:
            pass
        return count
    finally:
        if fd >= 0:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass


def _detect_storable_content(prompt: str) -> bool:
    """Fast heuristic: does the prompt contain content worth storing?

    Issue #635 (M-40): exclude interrogatives and quoted spans before the
    storable-pattern match. A question ("do you prefer X?") is a recall/ask,
    not a statement of the user's own preference, and a token that only
    appears inside a quote ("the docs say 'I'm deprecated'") is not the user
    asserting a fact. Both used to spuriously trip the expensive store path.
    """
    if len(prompt) < 15 or len(prompt) > 2000:
        return False
    if _CODE_RE.search(prompt):
        return False
    # Strip quoted spans so a quoted token cannot trigger storage.
    stripped = _QUOTED_SPAN_RE.sub(" ", prompt)
    # An interrogative is an ask, not a self-statement: a trailing question
    # mark, or a prompt that opens with a question word ("do you prefer...",
    # "what is my..."). We do NOT exclude every _RECALL_RE phrasing here —
    # "we decided ..." is both a recall cue and a genuine decision worth
    # storing — so we gate on interrogative *form* only (issue #635, M-40).
    if stripped.rstrip().endswith("?"):
        return False
    if _INTERROGATIVE_RE.match(stripped.lstrip()):
        return False
    return bool(_STORABLE_RE.search(stripped))


def _check_conversation_depth(session_id: str, prompt: str, window: int = 5) -> bool:
    """Check if recent exchanges show thematic depth (keyword overlap).

    Reads the last `window` buffer entries and checks if the current prompt
    shares significant keywords with them, suggesting a sustained topic
    worth capturing.

    The current prompt has already been appended to the buffer by ``main()``
    before this runs, so the final buffer entry IS the current prompt. We
    exclude it (``lines[:-1]``) — otherwise the prompt always overlaps with
    itself and "enhanced" becomes indistinguishable from "max" (issue #634,
    M-17). This means a shallow/first-prompt exchange (no prior on-topic
    history) correctly does NOT fire under "enhanced".
    """
    # Issue #635 (M-73): derive the buffer file from the buffer dir directly.
    # The previous `_PROMPT_COUNTER_DIR.parent / "buffers"` was wrong whenever
    # TRUEMEMORY_BUFFER_DIR pointed somewhere other than a ".../buffers" path
    # (e.g. /custom/dir -> /custom/buffers), so depth never saw the buffer and
    # "enhanced" silently degraded to never firing. The prompt counter and the
    # message buffer share one directory (both default to TRUEMEMORY_BUFFER_DIR),
    # so reading from _PROMPT_COUNTER_DIR points at the same place
    # buffer_message() writes.
    safe_id = _safe_session_id_local(session_id)
    buffer_file = _PROMPT_COUNTER_DIR / f"{safe_id}.jsonl"
    if not buffer_file.exists():
        return False

    try:
        recent_words: set[str] = set()
        lines = buffer_file.read_text(encoding="utf-8").strip().splitlines()
        # Exclude the final entry: it is the current prompt (issue #634, M-17).
        prior_lines = lines[:-1]
        for line in prior_lines[-window:]:
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

    # Enhanced: require conversation depth; Max: skip depth check.
    # The depth check excludes the current prompt (already buffered by main())
    # so "enhanced" does not trivially match itself (issue #634, M-17).
    if store_intensity == "enhanced" and not _check_conversation_depth(session_id, prompt):
        return

    # Issue #634 (M-18): arm the same model-server deadline the recall paths
    # use so a slow/contended model server can't hang prompt submission on the
    # two searches + add below. Without this the store path runs synchronously
    # with no deadline before any recall path gets to set one.
    try:
        from truememory.ingest.hooks._shared import get_recall_deadline
        from truememory.model_client import set_request_timeout
        set_request_timeout(get_recall_deadline())
    except Exception:
        pass

    # Issue #634 (M-04): the encoding gate returns EncodingDecision with a
    # `should_encode` attribute (not `passed`). The previous code read
    # `decision.passed`, which raised AttributeError swallowed by the blanket
    # except below, so the store path stored ZERO rows. The Memory instance
    # was also never closed (leak). Both are fixed here.
    try:
        from truememory.client import Memory
        with Memory(path=db_path or None) as m:
            # Novelty check: quick search to avoid dupes.
            # Score-space contract (issue #632): a score is only a true
            # absolute cosine similarity when its result is tagged
            # score_space="cosine". The full search() pipeline can return
            # RELATIVELY normalized scores (FTS top hit pinned to 1.0;
            # reranker fused scores min-max pinned) — comparing those to the
            # absolute 0.85 cutoff drops any prompt sharing one keyword with
            # a stored memory as a bogus "duplicate", exactly when the
            # embedder is dead.
            #
            # Apply the 0.85 cosine cutoff ONLY to cosine-space scores.
            # Missing tag defaults to cosine for back-compat (matches
            # dedup.py): raw vector hits and callers that predate the tag
            # are genuine cosine. When the tag is explicitly "relative" the
            # number is untrustworthy, so fall back to the scale-free
            # word-overlap heuristic instead. (#631 makes the vector path
            # emit true cosine, so this stays correct as it lands.)
            existing = m.search(prompt, user_id=user_id or None, limit=3)
            if existing:
                for r in existing:
                    score = r.get("score", 0.0)
                    if r.get("score_space", "cosine") != "relative":
                        if score > 0.85:
                            return  # Too similar to existing memory (cosine)
                    elif _word_overlap(prompt, r.get("content", "")) > 0.85:
                        return  # Scale-free near-duplicate (relative score)

            # Run through encoding gate
            from truememory.ingest.encoding_gate import EncodingGate
            gate = EncodingGate(memory=m, user_id=user_id or "")
            decision = gate.evaluate(prompt)
            if not decision.should_encode:
                return

            # Issue #635 (M-39): route the actual store through the SAME
            # dedup-store critical section the Stop pipeline uses, instead of
            # a bare m.add(). The cheap novelty search above only sees full
            # search() scores; check_duplicate() runs the authoritative
            # vector + (optional) LLM dedup against existing memories. Holding
            # _dedup_store_lock() across the check + add makes the per-exchange
            # store atomic w.r.t. concurrent Stop-pipeline ingests, so a fact
            # stored here and later re-extracted (rephrased) by the pipeline is
            # SKIPped rather than double-stored, and vice versa.
            from truememory.ingest.dedup import check_duplicate, DedupAction
            from truememory.ingest.pipeline import _dedup_store_lock
            with _dedup_store_lock():
                dedup = check_duplicate(prompt, m, user_id=user_id or "")
                if dedup.action == DedupAction.ADD:
                    m.add(content=dedup.fact, user_id=user_id or None)
                # UPDATE/SKIP: an equivalent memory already exists — the
                # Stop pipeline owns merge/update semantics, so the
                # per-exchange path simply declines to double-store.
    except Exception:
        pass  # Never crash the hook


def _try_proactive_recall(
    prompt: str,
    user_id: str,
    db_path: str,
    session_id: str,
    search_intensity: str,
    prompt_count: int,
    debounced: bool = False,
) -> tuple[str | None, bool]:
    """Proactive recall based on search intensity.

    Enhanced: recall every ~5th prompt (8 results).
    Max: every prompt gets a memory search (10 results).
    Standard: handled by existing _try_auto_recall (recall-intent only).

    Returns ``(recall_context, searched)``. ``searched`` is True when this
    function ran (or short-circuited) the search for this prompt — the caller
    uses it to decide whether the auto-recall fallback would be a redundant
    second search of the same prompt (issue #636, M-41/M-42).

    ``debounced`` is the result of the one-shot SessionStart recall marker,
    consumed once by ``main()`` (issue #636, M-41): when True, SessionStart
    already injected recall for this prompt, so proactive recall is suppressed
    AND the caller must skip the auto-recall fallback (it would re-run the
    exact first-prompt search #561 suppressed).
    """
    if search_intensity == "standard":
        return None, False

    # Dedup: SessionStart already injected recall for this session's first
    # prompt. Suppress here and tell the caller it was handled so the
    # auto-recall fallback does not fire the same search again (M-41).
    if debounced:
        return None, True

    if search_intensity == "enhanced":
        # Every 5th prompt; off-cadence prompts fall through to auto-recall
        # (recall-intent detection), so report searched=False.
        if prompt_count % 5 != 0:
            return None, False
        limit = 8
    else:  # max
        limit = 10

    # Skip code-heavy prompts; auto-recall also skips these, so nothing to do.
    if _CODE_RE.search(prompt):
        return None, True

    try:
        try:
            from truememory.ingest.hooks._shared import get_recall_deadline
            from truememory.model_client import set_request_timeout
            set_request_timeout(get_recall_deadline())
        except Exception:
            pass
        from truememory.client import Memory
        with Memory(path=db_path or None) as m:
            results = m.search(prompt, user_id=user_id or None, limit=limit)
        # We searched this prompt — whether or not it found anything, the
        # auto-recall fallback must not search it again (M-42).
        if not results:
            return None, True
        lines = []
        for r in results[:limit]:
            content = r.get("content", "")[:200]
            lines.append(f"- {content}")
        return (
            "<truememory-recall>\n"
            "Proactive memory recall:\n"
            + "\n".join(lines)
            + "\n</truememory-recall>"
        ), True
    except Exception:
        return None, True


def _detect_recall(prompt: str) -> bool:
    if len(prompt) < 10 or len(prompt) > 500:
        return False
    if _CODE_RE.search(prompt):
        return False
    return bool(_RECALL_RE.search(prompt))


def _try_auto_recall(prompt: str, user_id: str, db_path: str, session_id: str = "", debounced: bool = False) -> str | None:
    """Search TrueMemory if prompt looks like a recall question.

    Skips the search entirely on the first prompt right after SessionStart,
    which already injected recall (issue #561). The gate runs before detection
    and the Memory load so the redundant first-message recall costs nothing.

    ``debounced`` is the one-shot SessionStart recall marker, consumed once by
    ``main()`` (issue #636, M-41): consuming it here too would double-consume
    and let a duplicate search slip through, so the marker is read upstream and
    passed in.
    """
    if debounced:
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
        with Memory(path=db_path or None) as m:
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

    # Issue #561 / #636 (M-41): consume the one-shot SessionStart recall marker
    # exactly ONCE per prompt. Both the proactive and auto-recall paths used to
    # call consume_recall_injected() independently — the proactive path consumed
    # it, returned None, and the auto-recall path then saw no marker and ran the
    # exact first-prompt search #561 was meant to suppress. We read it here and
    # pass the result into both paths.
    debounced = False
    try:
        from truememory.ingest.hooks._shared import consume_recall_injected
        if session_id:
            debounced = consume_recall_injected(session_id)
    except Exception:
        pass

    # Issue #396: proactive recall for enhanced/max search intensity.
    # ``searched`` is True when the proactive path already searched (or was
    # debounced for) this prompt, so the auto-recall fallback below would be a
    # redundant second Memory init + search of the same prompt (issue #636,
    # M-41/M-42). Only fall back when proactive did not handle this prompt.
    recall_context = None
    searched = False
    if search_intensity != "standard":
        recall_context, searched = _try_proactive_recall(
            prompt, args.user, args.db, session_id, search_intensity, prompt_count,
            debounced=debounced,
        )

    # Fall back to standard recall-intent detection (only if proactive didn't
    # already search this prompt).
    if not recall_context and not searched:
        recall_context = _try_auto_recall(
            prompt, args.user, args.db, session_id, debounced=debounced,
        )

    if recall_context:
        print(json.dumps({"additionalContext": recall_context}))


def buffer_message(session_id: str, prompt: str):
    """Append a user message to the session buffer file (with file locking)."""
    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    try:
        BUFFER_DIR.chmod(0o700)
    except OSError:
        pass

    # Sanitize session_id to prevent path traversal (e.g., "../../etc/passwd").
    # Shared helper keeps this name in lockstep with the counter/depth paths
    # (issue #635, M-73).
    safe_id = _safe_session_id_local(session_id)

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
