"""
TrueMemory Predictive Coding Filter
===================================

Stores only information-theoretic surprises.  Most messages are noise —
``"ok"``, ``"sounds good"``, ``"lol"``, ``"see you later"``.  Only about
7% contain genuinely new information.  Predictive coding keeps only the
surprises, which paradoxically **improves** retrieval because there is
less noise to wade through.

The theory comes from neuroscience: the brain does not store raw percepts
but rather *prediction errors* — deviations from what was expected.
Similarly, this module scores each message by how much new information it
contributes relative to everything that came before.

Key design decisions:

- Messages are processed **chronologically** — a message's surprise score
  depends on all preceding messages.
- The first mention of any topic is always surprising (score = 1.0).
- We do **not delete** low-surprise messages.  We score them so the search
  engine can prioritize high-surprise results, but the raw data is never
  discarded.
- This is a **ranking signal**, not a hard filter.

All functions use only ``sqlite3`` and the Python standard library.
"""

import re
import sqlite3


# ---------------------------------------------------------------------------
# Fact extraction helpers
# ---------------------------------------------------------------------------

# Common low-information messages that should always score near zero.
_NOISE_PATTERNS = {
    # Bare acknowledgments
    "ok", "okay", "k", "kk", "sure", "yes", "no", "yeah", "yep", "yup",
    "nah", "nope", "cool", "nice", "great", "thanks", "thank you",
    "thx", "ty", "sounds good", "sounds great", "perfect", "got it",
    "gotcha", "will do", "on it", "bet", "word", "facts", "true",
    "same", "right", "exactly", "agreed", "absolutely", "definitely",
    # Pleasantries
    "good morning", "good night", "gn", "gm", "hey", "hi", "hello",
    "what's up", "sup", "yo", "how are you", "how's it going",
    "see you", "see ya", "later", "bye", "ttyl", "talk later",
    "have a good one", "take care",
    # Reactions
    "lol", "lmao", "haha", "hahaha", "lmfao", "rofl",
    "omg", "wow", "damn", "dude", "bruh",
}

# Noise patterns as a set of normalized strings for fast lookup.
_NOISE_SET = frozenset(_NOISE_PATTERNS)

# Regex patterns for extracting structured facts.
_NUMBER_RE = re.compile(
    r"""
    \$[\d,.]+[KMBkmb]?                   # Dollar amounts
    | \d+\.?\d*\s*%                       # Percentages
    | \d+\.?\d*\s*(?:ms|seconds?|hrs?|hours?|minutes?)  # Durations
    | \d{1,3}(?:,\d{3})+                  # Large integers (1,200)
    | \d+\.?\d*\s*(?:lbs?|pounds?|kg)     # Weights
    | \d+\.?\d*\s*(?:miles?|km|steps?)    # Distances/counts
    | \d+:\d{2}(?::\d{2})?               # Times (1:52:34)
    """,
    re.VERBOSE | re.IGNORECASE,
)

_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)

_DATE_RE = re.compile(
    r"""
    \b\d{4}-\d{2}-\d{2}\b               # ISO dates
    | \b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May
    |Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?
    |Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,?\s*\d{4})?  # "July 29, 2024"
    """,
    re.VERBOSE | re.IGNORECASE,
)

_EVENT_KEYWORDS = frozenset({
    "quit", "hired", "fired", "joined", "started", "founded",
    "launched", "raised", "closed", "signed", "moved", "switched",
    "migrated", "decided", "announced", "incorporated", "accepted",
    "rejected", "bought", "sold", "promoted", "deployed", "released",
    "published", "graduated", "married", "engaged", "broke up",
    "diagnosed", "recovered", "completed", "won", "lost",
})

# Common English words that should not be treated as facts.
_COMMON_PROPER_NOUNS = frozenset({
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
# Schema helper
# ---------------------------------------------------------------------------

def _ensure_surprise_table(conn: sqlite3.Connection) -> None:
    """
    Create the ``surprise_scores`` table if it does not exist.

    This is a separate table (rather than a column on ``messages``) so
    that the predictive coding layer is additive and does not require
    schema changes to the core messages table.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS surprise_scores (
            message_id INTEGER PRIMARY KEY,
            surprise    REAL NOT NULL DEFAULT 0.0,
            fact_count  INTEGER NOT NULL DEFAULT 0,
            new_fact_count INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    """)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_facts(content: str) -> set:
    """
    Extract fact fingerprints from a message.

    A "fact" is a key piece of information that could answer a future query:

    - **Numbers**: ``"$1.5M"``, ``"96.1%"``, ``"14 employees"``.
    - **Proper nouns**: ``"Meridian Steel"``, ``"ClickHouse"``,
      ``"Elevation Ventures"``.
    - **Dates**: ``"July 29"``, ``"2025-01-15"``.
    - **Event keywords**: ``"quit"``, ``"hired"``, ``"raised"``,
      ``"moved to"``.

    Each fact is returned as a normalized lowercase string so that
    duplicates can be detected across messages.

    Args:
        content: The raw message text.

    Returns:
        Set of normalized fact strings.
    """
    facts: set[str] = set()

    # Numbers and quantitative facts
    for match in _NUMBER_RE.finditer(content):
        # Normalize: strip whitespace, lowercase
        fact = match.group(0).strip().lower()
        # Remove commas for canonical form
        fact = fact.replace(",", "")
        facts.add(f"num:{fact}")

    # Proper nouns (potential entity/product/place names)
    for match in _PROPER_NOUN_RE.finditer(content):
        noun = match.group(0)
        if noun not in _COMMON_PROPER_NOUNS and len(noun) > 1:
            facts.add(f"entity:{noun.lower()}")

    # Dates
    for match in _DATE_RE.finditer(content):
        date_str = match.group(0).strip().lower()
        facts.add(f"date:{date_str}")

    # Event keywords — capture the surrounding context
    lower = content.lower()
    words = lower.split()
    for i, word in enumerate(words):
        clean_word = word.strip(".,!?\"'()")
        if clean_word in _EVENT_KEYWORDS:
            # Capture a small window around the event word
            start = max(0, i - 2)
            end = min(len(words), i + 3)
            context = " ".join(words[start:end])
            facts.add(f"event:{context}")

    # Key phrases with "is", "are", "was", "were" (definitional statements)
    definitional = re.findall(
        r"(\w[\w\s]{2,30})\s+(?:is|are|was|were)\s+(\w[\w\s]{2,30})",
        content,
    )
    for subj, pred in definitional:
        subj_clean = subj.strip().lower()
        pred_clean = pred.strip().lower()
        if len(subj_clean) > 3 and len(pred_clean) > 3:
            facts.add(f"def:{subj_clean}={pred_clean}")

    return facts


def compute_surprise_score(content: str, existing_facts: set) -> float:
    """
    Score how "surprising" a message is on a 0-1 scale.

    **High surprise** (closer to 1.0) — the message should be kept:
      - Contains **new** information not in ``existing_facts``.
      - Contains numbers, dates, specific details.
      - Contradicts known information.
      - First mention of a new entity or topic.

    **Low surprise** (closer to 0.0) — the message is noise:
      - Repeats known information.
      - Generic pleasantries or acknowledgments.
      - Very short with no substance.

    The score is computed as:

    .. code-block:: text

        base = new_facts / max(total_facts, 1)
        + length_bonus (0-0.15 based on message length)
        + detail_bonus (0-0.15 for numbers/dates)
        - noise_penalty (0.5-0.8 for pure noise messages)

    Args:
        content:        The raw message text.
        existing_facts: Set of fact fingerprints already seen in prior
                        messages.

    Returns:
        Surprise score between 0.0 and 1.0.
    """
    # ---- Quick noise check ----
    stripped = content.strip().lower()
    # Remove punctuation for noise matching
    cleaned = re.sub(r"[^\w\s]", "", stripped)

    if cleaned in _NOISE_SET or len(cleaned) < 4:
        return 0.05  # near-zero but not exactly zero

    # Check if it is very close to a noise pattern
    words = cleaned.split()
    if len(words) <= 2 and all(w in _NOISE_SET for w in words):
        return 0.05

    # ---- Extract facts ----
    message_facts = extract_facts(content)
    total_facts = len(message_facts)

    if total_facts == 0:
        # No extractable facts — score based on length and content
        if len(content) < 30:
            return 0.1
        elif len(content) < 80:
            return 0.2
        else:
            return 0.3  # long but no structured facts

    # ---- Count new vs. known facts ----
    new_facts = message_facts - existing_facts
    new_count = len(new_facts)

    if new_count == 0:
        # All facts already known — this is a repetition
        return 0.1

    # ---- Compute base surprise ----
    # Ratio of new facts to total facts
    novelty_ratio = new_count / total_facts
    base_score = novelty_ratio * 0.6  # max contribution: 0.6

    # ---- Length bonus ----
    # Longer messages tend to be more informative
    length = len(content)
    if length > 300:
        length_bonus = 0.15
    elif length > 150:
        length_bonus = 0.10
    elif length > 80:
        length_bonus = 0.05
    else:
        length_bonus = 0.0

    # ---- Detail bonus ----
    # Extra credit for messages with numbers, dates, or specific data
    num_count = len(_NUMBER_RE.findall(content))
    date_count = len(_DATE_RE.findall(content))
    detail_bonus = min(0.15, (num_count + date_count) * 0.05)

    # ---- Event bonus ----
    # Messages describing events are inherently surprising
    event_facts = [f for f in new_facts if f.startswith("event:")]
    event_bonus = min(0.10, len(event_facts) * 0.05)

    # ---- Combine ----
    score = base_score + length_bonus + detail_bonus + event_bonus

    return max(0.05, min(1.0, score))


def _compute_surprise_rows(
    rows: list[tuple],
) -> tuple[dict[int, float], list[tuple[int, float, int, int]]]:
    existing_facts: set[str] = set()
    scores: dict[int, float] = {}
    stored_rows: list[tuple[int, float, int, int]] = []
    for msg_id, content, _timestamp in rows:
        message_facts = extract_facts(content)
        new_facts = message_facts - existing_facts
        surprise = compute_surprise_score(content, existing_facts)
        stored_rows.append(
            (msg_id, round(surprise, 4), len(message_facts), len(new_facts))
        )
        scores[msg_id] = surprise
        existing_facts.update(message_facts)
    return scores, stored_rows


def _publish_surprise_rows(
    conn: sqlite3.Connection,
    source_rows: list[tuple],
    stored_rows: list[tuple[int, float, int, int]],
) -> None:
    conn.execute("SAVEPOINT truememory_surprise_publish")
    try:
        _ensure_surprise_table(conn)
        # Claim SQLite writer ownership without changing rows. A stale caller
        # WAL snapshot fails here; a fresh connection validates its source
        # below while no other writer can change it before publication.
        conn.execute("DELETE FROM surprise_scores WHERE 0")
        current = conn.execute(
            "SELECT id, content, timestamp FROM messages ORDER BY timestamp, id"
        )
        try:
            for expected in source_rows:
                if current.fetchone() != expected:
                    raise sqlite3.OperationalError(
                        "Surprise index source changed during computation; retry the rebuild"
                    )
            if current.fetchone() is not None:
                raise sqlite3.OperationalError(
                    "Surprise index source changed during computation; retry the rebuild"
                )
        finally:
            current.close()
        conn.execute("DELETE FROM surprise_scores")
        conn.executemany(
            "INSERT INTO surprise_scores "
            "(message_id, surprise, fact_count, new_fact_count) VALUES (?, ?, ?, ?)",
            stored_rows,
        )
        conn.execute("RELEASE SAVEPOINT truememory_surprise_publish")
    except BaseException:
        # Include cancellation. Never release the savepoint if rollback fails,
        # because releasing an outermost savepoint could commit partial rows.
        conn.execute("ROLLBACK TO SAVEPOINT truememory_surprise_publish")
        conn.execute("RELEASE SAVEPOINT truememory_surprise_publish")
        raise


def build_surprise_index(conn: sqlite3.Connection) -> dict[int, float]:
    """
    Score all messages by surprise value and store the results.

    Messages are processed **chronologically** — each message's surprise
    score depends on the accumulated facts from all preceding messages.
    The first message about any topic always scores high.

    Results are stored in the ``surprise_scores`` table with columns:

    - **message_id**: foreign key to ``messages.id``.
    - **surprise**: the 0-1 surprise score.
    - **fact_count**: total facts extracted from this message.
    - **new_fact_count**: facts that were new at time of processing.

    Args:
        conn: Open database connection (from :func:`truememory.storage.create_db`).

    Returns:
        ``{message_id: surprise_score}`` for every message in the database.

    Computation precedes this function's write transaction. Publication checks
    the exact source snapshot under writer ownership and atomically replaces
    the index. A changed source raises ``sqlite3.OperationalError`` for the
    caller to retry. Existing caller transactions are never committed or
    restarted; any write lock they already hold remains their responsibility.
    """
    rows = conn.execute(
        "SELECT id, content, timestamp FROM messages ORDER BY timestamp, id"
    ).fetchall()
    scores, stored_rows = _compute_surprise_rows(rows)
    _publish_surprise_rows(conn, rows, stored_rows)
    return scores


def get_high_surprise_messages(
    conn: sqlite3.Connection,
    min_surprise: float = 0.5,
) -> list[dict]:
    """
    Retrieve only the surprising / informative messages.

    Joins ``surprise_scores`` with ``messages`` to return full message data
    for messages that exceed the surprise threshold.

    Args:
        conn:         Open database connection.
        min_surprise: Minimum surprise score to include (default 0.5).

    Returns:
        List of message dicts sorted by surprise score (highest first).
        Each dict includes ``id``, ``content``, ``sender``, ``recipient``,
        ``timestamp``, ``category``, ``modality``, ``surprise``,
        ``fact_count``, ``new_fact_count``.
    """
    _ensure_surprise_table(conn)

    rows = conn.execute(
        """
        SELECT m.id, m.content, m.sender, m.recipient, m.timestamp,
               m.category, m.modality,
               s.surprise, s.fact_count, s.new_fact_count
        FROM surprise_scores s
        JOIN messages m ON m.id = s.message_id
        WHERE s.surprise >= ?
        ORDER BY s.surprise DESC
        """,
        (min_surprise,),
    ).fetchall()

    return [
        {
            "id": r[0],
            "content": r[1],
            "sender": r[2],
            "recipient": r[3],
            "timestamp": r[4],
            "category": r[5],
            "modality": r[6],
            "surprise": r[7],
            "fact_count": r[8],
            "new_fact_count": r[9],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_surprise_stats(conn: sqlite3.Connection) -> dict:
    """
    Return summary statistics about the surprise index.

    Useful for understanding the distribution and tuning the threshold.

    Args:
        conn: Open database connection with surprise_scores populated.

    Returns:
        Dict with ``total_messages``, ``high_surprise_count`` (>= 0.5),
        ``low_surprise_count`` (< 0.2), ``mean_surprise``,
        ``median_surprise``, ``high_surprise_pct``.
    """
    _ensure_surprise_table(conn)

    row = conn.execute(
        "SELECT COUNT(*), AVG(surprise), MIN(surprise), MAX(surprise) "
        "FROM surprise_scores"
    ).fetchone()

    total = row[0] or 0
    mean = row[1] or 0.0

    if total == 0:
        return {
            "total_messages": 0,
            "high_surprise_count": 0,
            "low_surprise_count": 0,
            "mean_surprise": 0.0,
            "median_surprise": 0.0,
            "high_surprise_pct": 0.0,
        }

    high_count = conn.execute(
        "SELECT COUNT(*) FROM surprise_scores WHERE surprise >= 0.5"
    ).fetchone()[0]

    low_count = conn.execute(
        "SELECT COUNT(*) FROM surprise_scores WHERE surprise < 0.2"
    ).fetchone()[0]

    # Approximate median via ORDER BY + LIMIT/OFFSET
    median_row = conn.execute(
        "SELECT surprise FROM surprise_scores ORDER BY surprise "
        "LIMIT 1 OFFSET ?",
        (total // 2,),
    ).fetchone()
    median = median_row[0] if median_row else mean

    return {
        "total_messages": total,
        "high_surprise_count": high_count,
        "low_surprise_count": low_count,
        "mean_surprise": round(mean, 4),
        "median_surprise": round(median, 4),
        "high_surprise_pct": round(high_count / total * 100, 1) if total else 0.0,
    }
