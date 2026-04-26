"""
TrueMemory Salience Guard Module (L3)
====================================

Filters noise and handles entity disambiguation in search results. Solves
two major problems that plague vector-search-based memory systems:

**Query pollution**
    ChromaDB and similar systems return "omg jordan!!" for every query
    because the name "jordan" creates false semantic similarity. The salience
    guard ensures casual noise is demoted in favor of substantive content.

**Entity disambiguation**
    "Who is Marcus?" is ambiguous when there are TWO Marcus characters in the
    dataset. The entity filter boosts results that are directly from/to/about
    the queried entities, instead of relying on embedding similarity alone.

This module is a **post-processing** step: it takes search results and
re-ranks them. It does not replace the search layer -- it refines it.
"""

import json
import logging
import re
import sqlite3
from math import exp, log
from pathlib import Path

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known low-salience patterns
# ---------------------------------------------------------------------------

_NOISE_EXACT = frozenset({
    "ok", "okay", "k", "kk",
    "yes", "yeah", "yep", "yup", "ya", "yea",
    "no", "nah", "nope",
    "lol", "lmao", "lmfao", "haha", "hahaha", "heh",
    "omg", "omfg", "wtf",
    "nice", "cool", "dope", "sick", "lit", "fire",
    "thanks", "thx", "ty", "thank you",
    "got it", "gotcha",
    "sounds good", "sounds great",
    "bet", "word",
    "sure", "for sure",
    "same", "mood",
    "idk", "idc",
    "np", "no problem",
    "gn", "goodnight", "good night",
    "gm", "good morning",
    "brb", "ttyl",
})

# Regex for messages that are predominantly emoji (legacy scorer)
_EMOJI_PATTERN = re.compile(
    r"^[\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF"
    r"\U00002702-\U000027B0"
    r"\U000024C2-\U0001F251"
    r"\U0001f900-\U0001f9FF"
    r"\U0001fa00-\U0001fa6f"
    r"\U0001fa70-\U0001faff"
    r"\s]+$",
    re.UNICODE,
)

# Per-character emoji regex for learned scorer feature extraction
_EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF"
    r"\U00002702-\U000027B0"
    r"\U000024C2-\U0001F251"
    r"\U0001f900-\U0001f9FF"
    r"\U0001fa00-\U0001fa6f"
    r"\U0001fa70-\U0001faff]",
    re.UNICODE,
)

# High-signal modalities (OCR, structured data)
_HIGH_SIGNAL_MODALITIES = frozenset({
    "ocr", "email", "calendar", "note", "health_data",
    "strava", "receipt", "document", "bank_statement",
    "vet_record", "cap_table", "home_inspection",
})

# Patterns that indicate substantive content
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")
_MONEY_PATTERN = re.compile(r"\$[\d,]+(?:\.\d{2})?")
_DATE_PATTERN = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"\s+\d{1,2}",
    re.IGNORECASE,
)
_CAPS_WORDS_RE = re.compile(r"\b[A-Z]{3,}\b")
_BULLET_RE = re.compile(r"^[-*•]\s", re.MULTILINE)

_HIGH_AROUSAL: frozenset[str] = frozenset({
    "amazing", "incredible", "devastating", "heartbreaking",
    "thrilled", "furious", "terrified", "ecstatic", "crushed",
    "panic", "emergency", "urgent", "critical", "breakthrough",
    "milestone", "promoted", "fired", "pregnant", "engaged",
    "diagnosed", "accident", "passed away", "died",
})

_LIFE_EVENTS: frozenset[str] = frozenset({
    "got married", "got engaged", "having a baby", "got promoted",
    "got fired", "broke up", "moved to", "graduated", "launched",
    "raised funding", "demo day", "ipo", "acquisition",
})


# ---------------------------------------------------------------------------
# L3 learned weights (loaded once at import time)
# ---------------------------------------------------------------------------

_L3_WEIGHTS_PATH = Path(__file__).parent / "data" / "l3_weights.json"
_L3_WEIGHTS: tuple[float, ...] | None = None
_L3_BIAS: float | None = None

try:
    with open(_L3_WEIGHTS_PATH) as _f:
        _L3_DATA = json.load(_f)
        _L3_WEIGHTS = tuple(_L3_DATA["weights"])
        _L3_BIAS = float(_L3_DATA["bias"])
        assert len(_L3_WEIGHTS) == 13, f"Expected 13 weights, got {len(_L3_WEIGHTS)}"
        del _L3_DATA
    _log.debug("L3 learned weights loaded from %s", _L3_WEIGHTS_PATH)
except FileNotFoundError:
    _log.warning(
        "L3 weight file not found at %s — falling back to legacy hand-tuned scorer.",
        _L3_WEIGHTS_PATH,
    )
except Exception as _exc:
    _log.warning(
        "Failed to load L3 weights from %s: %s — falling back to legacy scorer.",
        _L3_WEIGHTS_PATH,
        _exc,
    )


# ---------------------------------------------------------------------------
# Feature extraction for the learned scorer
# ---------------------------------------------------------------------------

def _extract_features(content: str, modality: str = "") -> tuple[float, ...]:
    """Extract the 13 continuous features for the L3 salience model.

    Feature order matches l3_weights.json and _features.py FEATURE_NAMES:
        f_noise, f_emoji, f_length, f_num, f_money, f_date,
        f_mod, f_nl, f_bul, f_excl, f_caps, f_arou, f_life
    """
    text_stripped = content.strip()
    text_lower = text_stripped.lower().strip("!?.… ")

    f_noise = 1.0 if text_lower in _NOISE_EXACT else 0.0

    if text_stripped:
        emoji_chars = sum(1 for _ in _EMOJI_RE.finditer(text_stripped))
        f_emoji = min(1.0, emoji_chars / max(1, len(text_stripped)))
    else:
        f_emoji = 0.0

    f_length = log(1 + len(text_stripped)) / 7.0
    f_num = log(1 + len(_NUMBER_PATTERN.findall(text_stripped))) / 3.0
    f_money = min(1.0, len(_MONEY_PATTERN.findall(text_stripped)) / 2.0)
    f_date = min(1.0, len(_DATE_PATTERN.findall(text_stripped)) / 2.0)
    f_mod = 1.0 if modality.lower() in _HIGH_SIGNAL_MODALITIES else 0.0
    f_nl = 1.0 if ("\n" in text_stripped and len(text_stripped) > 50) else 0.0
    f_bul = 1.0 if _BULLET_RE.search(text_stripped) else 0.0
    f_excl = min(1.0, text_stripped.count("!") / 3.0)
    f_caps = min(1.0, len(_CAPS_WORDS_RE.findall(text_stripped)) / 5.0)
    arou_hits = sum(1 for w in _HIGH_AROUSAL if w in text_lower)
    f_arou = min(1.0, arou_hits / 3.0)
    life_hits = sum(1 for e in _LIFE_EVENTS if e in text_lower)
    f_life = min(1.0, life_hits / 2.0)

    return (
        f_noise, f_emoji, f_length, f_num, f_money, f_date,
        f_mod, f_nl, f_bul, f_excl, f_caps, f_arou, f_life,
    )


# ---------------------------------------------------------------------------
# Salience scoring
# ---------------------------------------------------------------------------

def compute_message_salience(content: str, modality: str = "") -> float:
    """Score a message's salience (importance) on a 0-1 scale.

    Uses a learned logistic regression model if weights are available,
    otherwise falls back to the legacy hand-tuned additive scorer.
    """
    if not content or not content.strip():
        return 0.0

    if _L3_WEIGHTS is not None and _L3_BIAS is not None:
        features = _extract_features(content, modality)
        logit = sum(w * f for w, f in zip(_L3_WEIGHTS, features)) + _L3_BIAS
        return 1.0 / (1.0 + exp(-logit))

    return _score_legacy(content, modality)


def _score_legacy(content: str, modality: str = "") -> float:
    """Legacy hand-tuned additive scorer. Used as fallback when learned
    weights are not available."""
    if not content:
        return 0.0

    text = content.strip()
    text_lower = text.lower().strip("!?.… ")

    score = 0.3  # Base score

    # --- Noise penalty ---
    if text_lower in _NOISE_EXACT:
        score -= 0.3

    # --- Emoji-only penalty ---
    if _EMOJI_PATTERN.match(text):
        score -= 0.2

    # --- Length bonus (log-scaled, caps at ~200 chars) ---
    length = len(text)
    if length < 10:
        score -= 0.1  # Very short is likely noise
    elif length < 30:
        pass  # Neutral
    elif length < 100:
        score += 0.1
    elif length < 200:
        score += 0.2
    else:
        score += 0.25

    # --- Number / date / money bonus ---
    has_numbers = bool(_NUMBER_PATTERN.search(text))
    has_money = bool(_MONEY_PATTERN.search(text))
    has_dates = bool(_DATE_PATTERN.search(text))

    if has_money:
        score += 0.15
    elif has_numbers or has_dates:
        score += 0.1

    # --- Modality bonus ---
    if modality.lower() in _HIGH_SIGNAL_MODALITIES:
        score += 0.15

    # --- Structural content bonus (newlines, bullet points suggest notes) ---
    if "\n" in text and len(text) > 50:
        score += 0.05
    if re.search(r"^[-*•]\s", text, re.MULTILINE):
        score += 0.05

    # --- Emotional salience bonus (C2) ---
    # Exclamation density
    excl_count = text.count('!')
    if excl_count >= 3:
        score += 0.15
    elif excl_count >= 1:
        score += 0.05

    # ALL CAPS words (3+ chars)
    caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
    if caps_words:
        score += min(0.1, len(caps_words) * 0.05)

    # High-arousal vocabulary
    arousal_hits = sum(1 for w in _HIGH_AROUSAL if w in text_lower)
    score += min(0.2, arousal_hits * 0.1)

    # Life event markers
    event_hits = sum(1 for e in _LIFE_EVENTS if e in text_lower)
    score += min(0.3, event_hits * 0.15)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Entity detection and known entities
# ---------------------------------------------------------------------------

def get_known_entities(conn: sqlite3.Connection) -> list[str]:
    """
    Get all unique sender/recipient names from the messages table.

    Returns lowercase, deduplicated, sorted list excluding empty strings
    and ``"self"``.

    Args:
        conn: Open database connection.

    Returns:
        Sorted list of entity name strings.
    """
    rows = conn.execute("""
        SELECT DISTINCT LOWER(name) FROM (
            SELECT sender AS name FROM messages WHERE sender != ''
            UNION
            SELECT recipient AS name FROM messages WHERE recipient != ''
        )
        ORDER BY name
    """).fetchall()
    return [r[0] for r in rows if r[0] and r[0] != "self"]


_FALLBACK_ENTITIES = []


def detect_entities(query: str, conn: sqlite3.Connection | None = None) -> list[str]:
    """
    Extract entity references (person names) from a query.

    Matches against known entities from the database for accurate detection.
    The database entity list is augmented with a hardcoded fallback list
    to catch entities that appear in message content but are never direct
    senders/recipients (e.g. "marcus" is discussed but never sends messages).

    If no database connection is provided, uses only the fallback list.

    Args:
        query: Natural language query string.
        conn:  Optional database connection for dynamic entity lookup.

    Returns:
        List of entity names (lowercase) found in the query.
        Example: ``"What does Jordan discuss with Dev vs Sam?"``
        returns ``["jordan", "dev", "sam"]``.
    """
    if conn is not None:
        db_entities = get_known_entities(conn)
        # Merge DB entities with fallback list (dedup, preserve order)
        seen = set(db_entities)
        known = list(db_entities)
        for e in _FALLBACK_ENTITIES:
            if e not in seen:
                known.append(e)
                seen.add(e)
    else:
        known = list(_FALLBACK_ENTITIES)

    query_lower = query.lower()
    found = []

    for entity in known:
        # Use word boundary matching to avoid partial matches
        # (e.g., "sam" shouldn't match "sample")
        pattern = r"\b" + re.escape(entity) + r"\b"
        if re.search(pattern, query_lower):
            found.append(entity)

    return found


# ---------------------------------------------------------------------------
# Filtering and re-ranking
# ---------------------------------------------------------------------------

def filter_by_salience(
    results: list[dict],
    min_salience: float = 0.10,
) -> list[dict]:
    """
    Remove low-salience noise from search results.

    Each result gets a ``salience`` score added to its dict. Results below
    ``min_salience`` are removed entirely.

    Args:
        results:      List of message dicts from search.
        min_salience: Minimum salience score to keep (default 0.3).

    Returns:
        Filtered list with ``salience`` scores attached.
    """
    filtered = []
    for r in results:
        salience = compute_message_salience(
            r.get("content", ""),
            r.get("modality", ""),
        )
        r["salience"] = salience
        if salience >= min_salience:
            filtered.append(r)
    return filtered


def filter_by_entity(
    results: list[dict],
    target_entities: list[str],
) -> list[dict]:
    """
    Boost results that mention the target entities.

    Instead of removing non-matching results (which would be too aggressive),
    this function re-scores results based on entity relevance:

    - Results **from or to** a target entity: ``+0.3`` boost
    - Results **mentioning** a target entity in content: ``+0.2`` boost
    - Results with **no entity connection**: ``-0.15`` penalty

    The ``entity_boost`` value is stored on each result dict for
    transparency.

    Args:
        results:         List of message dicts from search.
        target_entities: Entity names to boost (lowercase).

    Returns:
        Re-ranked list sorted by combined score (original score + entity boost),
        with ``entity_boost`` attached to each dict.
    """
    if not target_entities:
        return results

    target_set = {e.lower() for e in target_entities}

    for r in results:
        sender = r.get("sender", "").lower()
        recipient = r.get("recipient", "").lower()
        content_lower = r.get("content", "").lower()

        boost = 0.0

        # Direct involvement (from/to)
        if sender in target_set or recipient in target_set:
            boost += 0.3

        # Mentioned in content
        for entity in target_set:
            if entity in content_lower:
                boost += 0.2
                break  # One content match is enough

        # No connection at all -> small penalty
        if boost == 0.0:
            boost = -0.15

        r["entity_boost"] = boost

    # Sort by combined score: original search score (if present) + entity boost
    def sort_key(r: dict) -> float:
        base = r.get("score", r.get("salience", 0.5))
        return base + r.get("entity_boost", 0.0)

    results.sort(key=sort_key, reverse=True)

    # Entity saturation penalty (C1): if same sender dominates top results,
    # apply diminishing returns. Only active when there are enough unique
    # senders to provide diversity (>5). In small 2-3 person conversations
    # (like LoCoMo), all results come from the same people so penalizing
    # repetition would destroy recall.
    unique_senders = {r.get("sender", "").lower() for r in results if r.get("sender")}
    unique_senders.discard("")
    if len(unique_senders) > 5:
        sender_counts: dict[str, int] = {}
        for r in results:
            sender = r.get("sender", "").lower()
            if not sender:
                continue
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
            if sender_counts[sender] > 3:
                penalty = 0.3 * (sender_counts[sender] - 3)
                base = r.get("score", r.get("salience", 0.5))
                r["score"] = max(0.01, base - penalty * 0.1)
                r["entity_saturation_penalty"] = penalty

        # Re-sort after saturation penalty
        results.sort(key=sort_key, reverse=True)
    return results


def apply_salience_guard(
    results: list[dict],
    query: str,
    conn: sqlite3.Connection | None = None,
    min_salience: float = 0.10,
) -> list[dict]:
    """
    Main entry point for the L4 Salience Guard.

    Applies all salience filtering and entity boosting in sequence:

    1. **Entity detection**: Identify person names in the query.
    2. **Salience filtering**: Remove low-value noise (``"ok"``, ``"lol"``).
    3. **Entity boosting**: Promote results relevant to queried entities.
    4. **Return**: Re-ranked results with ``salience`` and ``entity_boost``
       scores attached for downstream inspection.

    Args:
        results:      List of message dicts from any search layer (FTS5, hybrid,
                      temporal, etc.).
        query:        The original natural language query.
        conn:         Optional database connection for dynamic entity lookup.
                      If not provided, uses hardcoded entity list.
        min_salience: Minimum salience threshold for filtering (default 0.25).
                      Lower values (e.g. 0.15) allow more results through for
                      broad/diffuse queries.

    Returns:
        Re-ranked, filtered list of message dicts.
    """
    if not results:
        return results

    # Step 1: Detect entities in the query
    entities = detect_entities(query, conn=conn)

    # Step 2: Filter low-salience noise
    results = filter_by_salience(results, min_salience=min_salience)

    # Step 3: Boost entity-relevant results
    if entities:
        results = filter_by_entity(results, entities)

    return results
