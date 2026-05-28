"""Standalone novelty-ratio surprise scorer for the instrumentation overlay.

⚠ DISTINCT FROM TRUEMEMORY'S NATIVE ``surprise_scores``.
================================================================
The ``surprise`` signal this module computes is the *instrumentation layer's*
own lightweight, per-process novelty-ratio heuristic. It is NOT the same as
TrueMemory's consolidation-based ``surprise_scores`` (built by
``predictive.build_surprise_index`` and used by the L5 surprise reranker).

The reasons they differ:
  - ``predictive.compute_surprise_score`` requires a caller-managed
    ``existing_facts`` set and is designed for the batch surprise-index
    pipeline, which only runs during ``consolidate()``. Consolidation is not
    guaranteed to be on, so native surprise is often simply absent.
  - The dashboard wants a surprise value at *ingest* time, on every candidate
    fact, with no dependency on consolidation. This module provides exactly
    that: a stateful, per-process accumulator of seen fact-fingerprints that
    yields a novelty ratio the moment a fact is evaluated by the encoding gate.

The signal is intentionally named ``surprise`` because that is the column name
the dashboard reads. Treat it as "info-value at birth," not as the engine's
consolidation-derived surprise.

What was dropped vs. the native scorer:
  - ``detail_bonus`` (numbers/dates credit) — overlaps with salience's own
    number/date features; removing it keeps surprise and salience orthogonal.
  - All DB writes — this module emits via ``signals``, not directly to SQLite.
  - ``build_surprise_index`` / ``get_high_surprise_messages`` / stats queries.

Public API: ``compute_surprise(fact: str) -> float``  returns [0, 1].
"""
from __future__ import annotations

import re
import threading

# ---------------------------------------------------------------------------
# Module-level state — rebuilds from scratch on each process start.
# No cross-restart persistence: surprise is "novelty since this process began."
# ---------------------------------------------------------------------------

_existing_facts: set[str] = set()
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Regex + noise constants
# ---------------------------------------------------------------------------

_NOISE_SET: frozenset[str] = frozenset({
    "ok", "okay", "k", "kk", "sure", "yes", "no", "yeah", "yep", "yup",
    "nah", "nope", "cool", "nice", "great", "thanks", "thank you",
    "thx", "ty", "sounds good", "sounds great", "perfect", "got it",
    "gotcha", "will do", "on it", "bet", "word", "facts", "true",
    "same", "right", "exactly", "agreed", "absolutely", "definitely",
    "good morning", "good night", "gn", "gm", "hey", "hi", "hello",
    "what's up", "sup", "yo", "how are you", "how's it going",
    "see you", "see ya", "later", "bye", "ttyl", "talk later",
    "have a good one", "take care",
    "lol", "lmao", "haha", "hahaha", "lmfao", "rofl",
    "omg", "wow", "damn", "dude", "bruh",
})

_NUMBER_RE = re.compile(
    r"""
    \$[\d,.]+[KMBkmb]?
    | \d+\.?\d*\s*%
    | \d+\.?\d*\s*(?:ms|seconds?|hrs?|hours?|minutes?)
    | \d{1,3}(?:,\d{3})+
    | \d+\.?\d*\s*(?:lbs?|pounds?|kg)
    | \d+\.?\d*\s*(?:miles?|km|steps?)
    | \d+:\d{2}(?::\d{2})?
    """,
    re.VERBOSE | re.IGNORECASE,
)

_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

_DATE_RE = re.compile(
    r"""
    \b\d{4}-\d{2}-\d{2}\b
    | \b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May
    |Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?
    |Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,?\s*\d{4})?
    """,
    re.VERBOSE | re.IGNORECASE,
)

_EVENT_KEYWORDS: frozenset[str] = frozenset({
    "quit", "hired", "fired", "joined", "started", "founded",
    "launched", "raised", "closed", "signed", "moved", "switched",
    "migrated", "decided", "announced", "incorporated", "accepted",
    "rejected", "bought", "sold", "promoted", "deployed", "released",
    "published", "graduated", "married", "engaged", "broke up",
    "diagnosed", "recovered", "completed", "won", "lost",
})

_COMMON_PROPER_NOUNS: frozenset[str] = frozenset({
    "I", "The", "A", "An", "Is", "It", "He", "She", "We", "They",
    "My", "Your", "His", "Her", "Our", "This", "That", "What",
    "When", "Where", "Who", "How", "Why", "But", "And", "Or",
    "So", "If", "Not", "No", "Yes", "Can", "Will", "Just",
    "Do", "Did", "Has", "Have", "Had", "Was", "Were", "Are",
    "Been", "Be", "Would", "Could", "Should", "May", "Might",
    "Let", "Also", "Still", "Even", "Too", "Very", "Really",
    "About", "Like", "Here", "There", "Now", "Then", "Well",
    "Hey", "Yeah", "Yep", "Thanks", "Thank", "Sure", "Ok",
    "For", "With", "From", "Into", "Over", "After", "Before",
    "Between", "During", "Through", "Some", "Any", "All",
    "Each", "Every", "New", "Good", "Great", "First", "Last",
    "Going", "Looking", "Don",
})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_facts(content: str) -> set[str]:
    """Extract fact fingerprints from a string. Pure-stdlib, no engine deps."""
    facts: set[str] = set()

    for match in _NUMBER_RE.finditer(content):
        fact = match.group(0).strip().lower().replace(",", "")
        facts.add(f"num:{fact}")

    for match in _PROPER_NOUN_RE.finditer(content):
        noun = match.group(0)
        if noun not in _COMMON_PROPER_NOUNS and len(noun) > 1:
            facts.add(f"entity:{noun.lower()}")

    for match in _DATE_RE.finditer(content):
        facts.add(f"date:{match.group(0).strip().lower()}")

    lower = content.lower()
    words = lower.split()
    for i, word in enumerate(words):
        clean = word.strip(".,!?\"'()")
        if clean in _EVENT_KEYWORDS:
            start, end = max(0, i - 2), min(len(words), i + 3)
            facts.add(f"event:{' '.join(words[start:end])}")

    for subj, pred in re.findall(
        r"(\w[\w\s]{2,30})\s+(?:is|are|was|were)\s+(\w[\w\s]{2,30})", content
    ):
        s, p = subj.strip().lower(), pred.strip().lower()
        if len(s) > 3 and len(p) > 3:
            facts.add(f"def:{s}={p}")

    return facts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_surprise(fact: str) -> float:
    """Return info-value score in [0, 1] for *fact* relative to seen history.

    High = genuinely new claim. Low = restatement of known information.
    Never raises; returns 0.5 (neutral) on any internal error.

    detail_bonus is intentionally omitted (salience already captures
    number/date density — keeping the two signals orthogonal).
    """
    try:
        # ---- Quick noise check ----
        stripped = fact.strip().lower()
        cleaned = re.sub(r"[^\w\s]", "", stripped)
        if cleaned in _NOISE_SET or len(cleaned) < 4:
            return 0.05
        words = cleaned.split()
        if len(words) <= 2 and all(w in _NOISE_SET for w in words):
            return 0.05

        # ---- Extract facts ----
        message_facts = _extract_facts(fact)
        total_facts = len(message_facts)

        # ---- Thread-safe snapshot + update ----
        with _lock:
            snapshot = _existing_facts.copy()
            _existing_facts.update(message_facts)

        if total_facts == 0:
            length = len(fact)
            if length < 30:
                return 0.1
            elif length < 80:
                return 0.2
            else:
                return 0.3

        # ---- Novelty ratio (the unique component vs salience) ----
        new_facts = message_facts - snapshot
        new_count = len(new_facts)
        if new_count == 0:
            return 0.1

        novelty_ratio = new_count / total_facts
        base_score = novelty_ratio * 0.6  # max: 0.6

        # ---- Length bonus ----
        length = len(fact)
        if length > 300:
            length_bonus = 0.15
        elif length > 150:
            length_bonus = 0.10
        elif length > 80:
            length_bonus = 0.05
        else:
            length_bonus = 0.0

        # ---- Event bonus ----
        event_facts = [f for f in new_facts if f.startswith("event:")]
        event_bonus = min(0.10, len(event_facts) * 0.05)

        score = base_score + length_bonus + event_bonus
        return max(0.05, min(1.0, score))

    except Exception:
        return 0.5
