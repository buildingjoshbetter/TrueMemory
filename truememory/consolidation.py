"""
TrueMemory L5 — Consolidation
============================

Handles hierarchical summarization and contradiction resolution.  This is
the layer that competitors consistently fail on:

- ``"What database does CarbonSense use?"`` should return **ClickHouse**
  (current), not PostgreSQL (superseded).
- ``"Summarize the CarbonSense journey"`` needs information scattered across
  hundreds of messages condensed into a coherent narrative.
- ``"How did Sam's involvement evolve?"`` requires tracking an entity across
  many conversations and detecting role changes.

The module provides:

1. **Entity timelines** — chronologically sorted messages per entity.
2. **Contradiction detection** — finds facts that changed over time and
   records them with supersession links so the *latest* fact is preferred.
3. **Extractive summaries** — groups messages by month (and by entity)
   and picks the most information-dense ones as representative summaries.
4. **Contradiction-aware search** — ensures queries about changed facts
   return the current state with historical context.
5. **Consolidated search** — searches summaries for broad, journey-style
   queries that span many messages.

All functions operate on a ``sqlite3.Connection`` produced by
:func:`truememory.storage.create_db`.  No external dependencies beyond the
Python standard library.
"""

import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

from truememory.fts_search import _build_safe_fts_query, _fts_search


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_all_messages_chrono(conn: sqlite3.Connection) -> list[dict]:
    """Fetch all messages ordered by timestamp."""
    rows = conn.execute(
        "SELECT id, content, sender, recipient, timestamp, category, modality "
        "FROM messages ORDER BY timestamp"
    ).fetchall()
    return [
        {
            "id": r[0], "content": r[1], "sender": r[2],
            "recipient": r[3], "timestamp": r[4],
            "category": r[5], "modality": r[6],
        }
        for r in rows
    ]


def _extract_month(timestamp: str) -> str:
    """Extract 'YYYY-MM' from an ISO timestamp string."""
    if not timestamp:
        return "unknown"
    # Handle both "2024-07-01" and "2024-07-01T09:00:00" formats
    match = re.match(r"(\d{4}-\d{2})", timestamp)
    return match.group(1) if match else "unknown"


def _extract_numbers(content: str) -> list[str]:
    """Extract numeric values (dollar amounts, percentages, counts) from text."""
    patterns = [
        r"\$[\d,.]+[KMBkmb]?",          # Dollar amounts: $1.5M, $45K, $2,000
        r"\d+\.?\d*%",                   # Percentages: 96.1%, 84.7%
        r"\d+\.?\d*\s*(?:ms|seconds?)",  # Latencies: 47ms, 2.3 seconds
        r"\d{1,3}(?:,\d{3})+",          # Large numbers: 1,200, 50,000
        r"\d+\s*(?:employees?|people|team|hires?)",  # Headcounts
        r"\d+\s*(?:customers?|clients?|users?)",     # Customer counts
    ]
    results = []
    for pattern in patterns:
        results.extend(re.findall(pattern, content, re.IGNORECASE))
    return results


def _message_salience(msg: dict) -> float:
    """
    Score a message's information density (salience) on a 0-1 scale.

    High-salience messages contain numbers, dates, decisions, events,
    or significant changes.  Low-salience messages are pleasantries,
    acknowledgments, or very short texts.
    """
    content = msg["content"]
    score = 0.0

    # Length bonus (longer messages tend to be more informative)
    length = len(content)
    if length > 200:
        score += 0.3
    elif length > 100:
        score += 0.2
    elif length > 50:
        score += 0.1
    elif length < 20:
        score -= 0.2  # Very short, likely not informative

    # Numbers and specific data
    numbers = _extract_numbers(content)
    score += min(0.3, len(numbers) * 0.1)

    # Decision/event keywords
    event_keywords = {
        "decided", "decision", "quit", "hired", "fired", "joined",
        "launched", "raised", "closed", "signed", "moved", "switched",
        "migrated", "started", "finished", "completed", "announced",
        "incorporated", "founded", "accepted", "rejected", "bought",
        "sold", "promoted", "deployed", "released", "published",
    }
    lower = content.lower()
    event_hits = sum(1 for kw in event_keywords if kw in lower)
    score += min(0.3, event_hits * 0.1)

    # Proper nouns (specific entities, places, products)
    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)
    score += min(0.1, len(proper_nouns) * 0.02)

    return max(0.0, min(1.0, score))


def _extract_sentences(text):
    """Split text into sentences."""
    # Split on period, exclamation, question mark followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _score_sentence(sentence, key_entities=None):
    """Score an individual sentence by information density."""
    score = 0.0
    length = len(sentence)

    if length > 100:
        score += 0.2
    elif length > 50:
        score += 0.1

    # Numbers
    if re.search(r'\d', sentence):
        score += 0.15

    # Money
    if re.search(r'\$[\d,.]+', sentence):
        score += 0.2

    # Event keywords
    event_words = {"quit", "hired", "launched", "raised", "moved", "switched",
                   "migrated", "started", "finished", "signed", "decided"}
    lower = sentence.lower()
    event_hits = sum(1 for w in event_words if w in lower)
    score += min(0.3, event_hits * 0.1)

    # Proper nouns
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
    score += min(0.1, len(proper_nouns) * 0.02)

    # Entity relevance
    if key_entities:
        entity_hits = sum(1 for e in key_entities if e.lower() in lower)
        score += min(0.15, entity_hits * 0.05)

    return score


# ---------------------------------------------------------------------------
# Contradiction detection patterns
# ---------------------------------------------------------------------------

# Each pattern has: category, regex to find the old/new pattern, subject key.
# Patterns are intentionally strict to avoid false positives.
_CHANGE_PATTERNS = [
    # Explicit change language: "switched from X to Y", "migrated from X to Y"
    # Requires "from ... to ..." to avoid matching casual usage of these verbs.
    {
        "category": "technology",
        "pattern": re.compile(
            r"(?:switched|migrated?|transitioned?|converted?|moved?)\s+"
            r"from\s+([A-Z][\w\s.+-]{2,25}?)\s+to\s+([A-Z][\w\s.+-]{2,25}?)(?:[.,!?\s]|$)",
            re.IGNORECASE,
        ),
        "type": "explicit_change",
    },
    # Pricing with "per facility" / "per month" / "/facility" context.
    # Must have $/unit pattern with a business-relevant unit.
    {
        "category": "pricing",
        "pattern": re.compile(
            r"(\$[\d,.]+[KMBkmb]?)\s*(?:per|/)\s*(facility|month|seat|user|year|license)",
            re.IGNORECASE,
        ),
        "type": "pricing",
    },
    # Office/location changes with explicit "new office", "moved office to"
    {
        "category": "location",
        "pattern": re.compile(
            r"(?:new\s+office\s+(?:at|is|in)|office\s+(?:moved?|switched?|relocated?)\s+to"
            r"|moved?\s+(?:the\s+)?office\s+to|relocated?\s+to)\s+"
            r"([A-Z][\w\s',.-]{3,40}?)(?:[.,!?\s]|$)",
            re.IGNORECASE,
        ),
        "type": "location_change",
    },
    # Status changes: requires a proper noun + verb (e.g. "Dev quit")
    {
        "category": "status",
        "pattern": re.compile(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+"
            r"(?:quit|left|started|joined|hired|fired|resigned|promoted)\b",
        ),
        "type": "status_change",
    },
    # Schedule changes with gym/workout/meditation context
    {
        "category": "schedule",
        "pattern": re.compile(
            r"(?:switched|changed?|moved?)\s+(?:my\s+)?(?:gym|workout|meditation|exercise)"
            r"\s+(?:to|back\s+to)\s+(\w[\w\s]{2,20}?)(?:[.,!?\s]|$)",
            re.IGNORECASE,
        ),
        "type": "schedule_change",
    },
]

# Known subject normalization: maps keyword fragments to canonical subjects.
_SUBJECT_NORMALIZERS = {
    "postgres": "database",
    "postgresql": "database",
    "timescaledb": "database",
    "clickhouse": "database",
    "sqlite": "database",
    "mysql": "database",
    "kubernetes": "container_orchestration",
    "k8s": "container_orchestration",
    "ecs": "container_orchestration",
    "docker": "container_orchestration",
    "wework": "office_location",
    "office": "office_location",
    "headspace": "meditation_app",
    "insight timer": "meditation_app",
    "morning": "gym_schedule",
    "evening": "gym_schedule",
    "mornings": "gym_schedule",
    "evenings": "gym_schedule",
}

# Minimum length for an extracted fact value to be meaningful.
_MIN_FACT_LEN = 3
_MAX_FACT_LEN = 60


def _normalize_subject(raw: str, context: str = "") -> str:
    """Normalize a raw subject string to a canonical subject key."""
    lower = raw.lower().strip()

    # Check direct normalizer matches
    for keyword, normalized in _SUBJECT_NORMALIZERS.items():
        if keyword in lower or keyword in context.lower():
            return normalized

    # Fall back to the raw string, cleaned up
    return re.sub(r"\s+", "_", lower)[:50]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_entity_timelines(conn: sqlite3.Connection) -> dict:
    """
    Group messages by entity and sort chronologically.

    An "entity" here is either a sender or a recipient — any person or
    system that appears in the message flow.  Messages are included in an
    entity's timeline if they sent or received the message.

    Args:
        conn: Open database connection.

    Returns:
        ``{entity_name: [messages sorted by timestamp]}``.  Each message
        is a dict with ``id``, ``content``, ``sender``, ``recipient``,
        ``timestamp``, ``category``, ``modality``.
    """
    all_msgs = _get_all_messages_chrono(conn)

    timelines: dict[str, list[dict]] = defaultdict(list)
    seen_per_entity: dict[str, set] = defaultdict(set)

    for msg in all_msgs:
        for entity_field in ("sender", "recipient"):
            entity = msg[entity_field]
            if entity and msg["id"] not in seen_per_entity[entity]:
                seen_per_entity[entity].add(msg["id"])
                timelines[entity].append(msg)

    # Ensure chronological order (should already be, but be safe)
    for entity in timelines:
        timelines[entity].sort(key=lambda m: m["timestamp"])

    return dict(timelines)


def detect_contradictions(conn: sqlite3.Connection) -> list[dict]:
    """
    Find facts that changed over time (contradictions).

    Scans all messages chronologically for patterns that indicate a fact
    has been updated:

    - **Pricing changes**: dollar amounts for the same product/service at
      different times.
    - **Technology changes**: ``"migrated from X to Y"``, ``"switched to"``,
      ``"rewrite"``.
    - **Location changes**: ``"moved to"``, ``"new office"``.
    - **Status changes**: ``"quit"``, ``"hired"``, ``"started"``.
    - **Schedule changes**: ``"switched to mornings"``, ``"back to mornings"``.

    For each detected change, a record is inserted into ``fact_timeline``
    with the subject, fact value, source message, and timestamp.  When a
    newer fact supersedes an older one, the older record's
    ``superseded_by`` column is updated.

    Returns:
        List of contradiction records, each a dict with ``subject``,
        ``old_fact``, ``new_fact``, ``old_timestamp``, ``new_timestamp``,
        ``source_message_id``.
    """
    all_msgs = _get_all_messages_chrono(conn)

    # Clear existing fact_timeline for a clean rebuild
    conn.execute("DELETE FROM fact_timeline")

    # Track facts by subject: subject -> [(fact_value, timestamp, msg_id, db_id)]
    fact_history: dict[str, list[tuple]] = defaultdict(list)
    contradictions: list[dict] = []

    for msg in all_msgs:
        content = msg["content"]
        timestamp = msg["timestamp"]
        msg_id = msg["id"]

        for pattern_def in _CHANGE_PATTERNS:
            matches = pattern_def["pattern"].finditer(content)

            for match in matches:
                groups = match.groups()

                if pattern_def["type"] == "explicit_change" and len(groups) >= 2:
                    old_val = groups[0].strip()
                    new_val = groups[1].strip()
                    subject = _normalize_subject(old_val, content)

                    # Try to extract entity context
                    entity_context = ""
                    nearby_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content[:100])
                    common_words = {"The", "This", "That", "We", "They", "Our", "My", "But", "And", "Just", "Not"}
                    entities = [n for n in nearby_nouns if n not in common_words and len(n) > 2]
                    if entities:
                        entity_context = entities[0].lower()

                    # Record the old fact if not already tracked
                    if subject not in fact_history or not fact_history[subject]:
                        cursor = conn.execute(
                            "INSERT INTO fact_timeline "
                            "(subject, fact, source_message_id, timestamp, entity_scope, valid_from) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (subject, old_val, msg_id, timestamp, entity_context, timestamp),
                        )
                        old_db_id = cursor.lastrowid
                        fact_history[subject].append(
                            (old_val, timestamp, msg_id, old_db_id)
                        )

                    # Record the new fact and supersede the old
                    cursor = conn.execute(
                        "INSERT INTO fact_timeline "
                        "(subject, fact, source_message_id, timestamp, entity_scope, valid_from) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (subject, new_val, msg_id, timestamp, entity_context, timestamp),
                    )
                    new_db_id = cursor.lastrowid

                    # Update the previous fact's superseded_by and valid_to
                    if fact_history[subject]:
                        prev = fact_history[subject][-1]
                        conn.execute(
                            "UPDATE fact_timeline SET superseded_by = ? "
                            "WHERE id = ?",
                            (new_db_id, prev[3]),
                        )
                        conn.execute(
                            "UPDATE fact_timeline SET valid_to = ? WHERE id = ?",
                            (timestamp, prev[3]),
                        )

                    fact_history[subject].append(
                        (new_val, timestamp, msg_id, new_db_id)
                    )

                    contradictions.append({
                        "subject": subject,
                        "old_fact": old_val,
                        "new_fact": new_val,
                        "old_timestamp": fact_history[subject][-2][1]
                        if len(fact_history[subject]) >= 2 else "",
                        "new_timestamp": timestamp,
                        "source_message_id": msg_id,
                    })

                elif pattern_def["type"] == "pricing" and len(groups) >= 2:
                    price = groups[0].strip()
                    unit = groups[1].strip()

                    # Extract entity context (company/product name) from surrounding text
                    entity_context = ""
                    # Look for proper nouns near the price
                    nearby_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content)
                    common_words = {"The", "This", "That", "We", "They", "Our", "My", "But", "And", "Just", "Not"}
                    entities = [n for n in nearby_nouns if n not in common_words and len(n) > 2]
                    if entities:
                        entity_context = entities[0].lower()

                    subject = f"pricing_{entity_context}_{unit}" if entity_context else f"pricing_{unit}"

                    cursor = conn.execute(
                        "INSERT INTO fact_timeline "
                        "(subject, fact, source_message_id, timestamp, entity_scope, valid_from) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (subject, price, msg_id, timestamp, entity_context, timestamp),
                    )
                    new_db_id = cursor.lastrowid

                    # Check if this contradicts a previous pricing fact
                    if subject in fact_history and fact_history[subject]:
                        prev = fact_history[subject][-1]
                        if prev[0] != price:
                            conn.execute(
                                "UPDATE fact_timeline SET superseded_by = ? "
                                "WHERE id = ?",
                                (new_db_id, prev[3]),
                            )
                            conn.execute(
                                "UPDATE fact_timeline SET valid_to = ? WHERE id = ?",
                                (timestamp, prev[3]),
                            )
                            contradictions.append({
                                "subject": subject,
                                "old_fact": prev[0],
                                "new_fact": price,
                                "old_timestamp": prev[1],
                                "new_timestamp": timestamp,
                                "source_message_id": msg_id,
                            })

                    fact_history[subject].append(
                        (price, timestamp, msg_id, new_db_id)
                    )

                elif pattern_def["type"] in ("location_change",
                                              "schedule_change"):
                    if groups:
                        fact_val = groups[0].strip()
                        if (len(fact_val) < _MIN_FACT_LEN
                                or len(fact_val) > _MAX_FACT_LEN):
                            continue

                        subject = _normalize_subject(fact_val, content)

                        # Try to extract entity context
                        entity_context = ""
                        nearby_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content[:100])
                        common_words = {"The", "This", "That", "We", "They", "Our", "My", "But", "And", "Just", "Not"}
                        entities = [n for n in nearby_nouns if n not in common_words and len(n) > 2]
                        if entities:
                            entity_context = entities[0].lower()

                        cursor = conn.execute(
                            "INSERT INTO fact_timeline "
                            "(subject, fact, source_message_id, timestamp, entity_scope, valid_from) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (subject, fact_val, msg_id, timestamp, entity_context, timestamp),
                        )
                        new_db_id = cursor.lastrowid

                        # Check for contradiction with previous fact on
                        # same subject
                        if (subject in fact_history
                                and fact_history[subject]):
                            prev = fact_history[subject][-1]
                            if prev[0].lower() != fact_val.lower():
                                conn.execute(
                                    "UPDATE fact_timeline "
                                    "SET superseded_by = ? WHERE id = ?",
                                    (new_db_id, prev[3]),
                                )
                                conn.execute(
                                    "UPDATE fact_timeline SET valid_to = ? WHERE id = ?",
                                    (timestamp, prev[3]),
                                )
                                contradictions.append({
                                    "subject": subject,
                                    "old_fact": prev[0],
                                    "new_fact": fact_val,
                                    "old_timestamp": prev[1],
                                    "new_timestamp": timestamp,
                                    "source_message_id": msg_id,
                                })

                        fact_history[subject].append(
                            (fact_val, timestamp, msg_id, new_db_id)
                        )

                elif pattern_def["type"] == "status_change" and groups:
                    person = groups[0].strip()
                    if (len(person) < _MIN_FACT_LEN
                            or len(person) > _MAX_FACT_LEN):
                        continue

                    # Extract the verb
                    verb_match = re.search(
                        r"(quit|left|started|joined|hired|fired|"
                        r"resigned|promoted)",
                        content, re.IGNORECASE,
                    )
                    verb = verb_match.group(1).lower() if verb_match else "changed"

                    subject = f"status_{person.lower()}"
                    fact_val = f"{person} {verb}"
                    entity_context = person.lower()

                    cursor = conn.execute(
                        "INSERT INTO fact_timeline "
                        "(subject, fact, source_message_id, timestamp, entity_scope, valid_from) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (subject, fact_val, msg_id, timestamp, entity_context, timestamp),
                    )
                    new_db_id = cursor.lastrowid
                    fact_history[subject].append(
                        (fact_val, timestamp, msg_id, new_db_id)
                    )

    conn.commit()
    return contradictions


def build_summaries(conn: sqlite3.Connection) -> int:
    """
    Build time-based and entity-based extractive summaries.

    Groups messages by month and selects the most salient (information-dense)
    messages to form each summary.  Also builds per-entity summaries for
    entities that have more than 10 messages.

    Summaries are stored in the ``summaries`` table with:

    - **period**: ``"monthly"`` or ``"entity_monthly"``.
    - **start_date** / **end_date**: the time range covered.
    - **entity**: the entity name (empty for global monthly summaries).
    - **summary**: concatenated text of the top salient messages.
    - **key_facts**: JSON list of extracted numbers/events.
    - **message_ids**: JSON list of source message IDs.

    This is *extractive* summarization — it picks the most important raw
    messages rather than generating new text.  This keeps the system local,
    fast, and hallucination-free.

    Args:
        conn: Open database connection.

    Returns:
        Number of summaries built.
    """
    all_msgs = _get_all_messages_chrono(conn)
    if not all_msgs:
        return 0

    # Clear existing summaries for a clean rebuild
    conn.execute("DELETE FROM summaries")

    now = datetime.now(timezone.utc).isoformat()
    summary_count = 0

    # ---- Monthly summaries ----
    by_month: dict[str, list[dict]] = defaultdict(list)
    for msg in all_msgs:
        month = _extract_month(msg["timestamp"])
        by_month[month].append(msg)

    for month, messages in sorted(by_month.items()):
        if month == "unknown":
            continue

        # Score each message by salience
        scored = [(msg, _message_salience(msg)) for msg in messages]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Ring-width proportional: event-heavy months get more detail
        high_salience_count = sum(1 for _, s in scored if s > 0.5)
        fact_change_count = 0  # count fact changes in this month
        for msg in messages:
            for pattern_def in _CHANGE_PATTERNS:
                if pattern_def["pattern"].search(msg["content"]):
                    fact_change_count += 1

        ring_width = high_salience_count + fact_change_count * 2

        # Proportional summary size: event-heavy months get 3x more coverage
        if ring_width > 10:
            top_count = max(25, len(messages) // 3)  # Detailed
        elif ring_width > 5:
            top_count = max(15, len(messages) // 5)  # Normal
        else:
            top_count = max(8, len(messages) // 8)   # Compressed

        top_messages = scored[:top_count]

        # Sort selected messages back into chronological order
        top_messages.sort(key=lambda x: x[0]["timestamp"])

        # Sentence-level extraction: pick most informative sentences
        all_sentences = []
        for msg, salience in top_messages:
            sentences = _extract_sentences(msg["content"])
            sender = msg["sender"]
            for sent in sentences:
                sent_score = _score_sentence(sent) + salience * 0.3
                all_sentences.append((f"[{sender}] {sent}", sent_score, msg))

        # Sort by sentence score and pick top sentences
        all_sentences.sort(key=lambda x: x[1], reverse=True)
        max_sentences = max(20, len(all_sentences) // 3)
        top_sentences = all_sentences[:max_sentences]

        # Re-sort selected sentences chronologically
        top_sentences.sort(key=lambda x: x[2]["timestamp"])

        summary_lines = [s[0] for s in top_sentences]
        message_ids = list(set(s[2]["id"] for s in top_sentences))

        key_facts = []
        for msg, salience in top_messages:
            # Extract key facts
            numbers = _extract_numbers(msg["content"])
            if numbers:
                key_facts.extend(numbers[:3])

        summary_text = "\n".join(summary_lines)

        # Determine date range
        timestamps = [m["timestamp"] for m, _ in top_messages if m["timestamp"]]
        start_date = min(timestamps) if timestamps else f"{month}-01"
        end_date = max(timestamps) if timestamps else f"{month}-28"

        conn.execute(
            "INSERT INTO summaries "
            "(period, start_date, end_date, entity, summary, "
            " key_facts, message_ids, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "monthly",
                start_date,
                end_date,
                "",
                summary_text,
                json.dumps(key_facts),
                json.dumps(message_ids),
                now,
            ),
        )
        summary_count += 1

    # ---- Per-entity summaries ----
    by_entity: dict[str, list[dict]] = defaultdict(list)
    for msg in all_msgs:
        if msg["sender"]:
            by_entity[msg["sender"]].append(msg)

    for entity, messages in by_entity.items():
        if len(messages) < 10:
            continue

        # Further group by month
        entity_by_month: dict[str, list[dict]] = defaultdict(list)
        for msg in messages:
            month = _extract_month(msg["timestamp"])
            entity_by_month[month].append(msg)

        for month, month_msgs in sorted(entity_by_month.items()):
            if month == "unknown" or len(month_msgs) < 3:
                continue

            scored = [(msg, _message_salience(msg)) for msg in month_msgs]
            scored.sort(key=lambda x: x[1], reverse=True)

            top_count = max(5, len(month_msgs) // 4)
            top_messages = scored[:top_count]
            top_messages.sort(key=lambda x: x[0]["timestamp"])

            summary_lines = []
            key_facts = []
            message_ids = []

            for msg, salience in top_messages:
                text = msg["content"][:500]
                summary_lines.append(text)
                message_ids.append(msg["id"])
                numbers = _extract_numbers(msg["content"])
                if numbers:
                    key_facts.extend(numbers[:3])

            summary_text = "\n".join(summary_lines)
            timestamps = [m["timestamp"] for m, _ in top_messages
                          if m["timestamp"]]
            start_date = min(timestamps) if timestamps else f"{month}-01"
            end_date = max(timestamps) if timestamps else f"{month}-28"

            conn.execute(
                "INSERT INTO summaries "
                "(period, start_date, end_date, entity, summary, "
                " key_facts, message_ids, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "entity_monthly",
                    start_date,
                    end_date,
                    entity,
                    summary_text,
                    json.dumps(key_facts),
                    json.dumps(message_ids),
                    now,
                ),
            )
            summary_count += 1

    conn.commit()
    return summary_count


def search_contradictions(conn: sqlite3.Connection,
                          query: str) -> list[dict]:
    """
    Check if a query is about something that changed over time.

    Searches the ``fact_timeline`` table for subjects related to the query
    keywords.  If a matching subject is found, returns the **latest**
    (non-superseded) fact along with full historical context.

    This ensures that queries like ``"What database does CarbonSense use?"``
    return ``"ClickHouse"`` (current) and not ``"PostgreSQL"`` (old).

    Args:
        conn:  Open database connection.
        query: Natural-language query string.

    Returns:
        List of dicts, each with ``subject``, ``current_fact``,
        ``current_timestamp``, ``history`` (list of all facts in
        chronological order), and ``source_message_id``.
        Empty list if no relevant contradictions found.
    """
    # Extract query keywords for matching against fact_timeline subjects
    stop_words = {
        "what", "which", "where", "when", "how", "does", "did", "is",
        "are", "was", "were", "the", "a", "an", "of", "for", "to",
        "in", "on", "at", "by", "with", "and", "or", "but", "not",
        "its", "it", "do", "has", "have", "had", "use", "used",
        "using", "current", "currently", "now", "today",
    }
    query_words = [
        w.lower().strip("?.,!\"'")
        for w in query.split()
        if w.lower().strip("?.,!\"'") not in stop_words and len(w) > 2
    ]

    if not query_words:
        return []

    results: list[dict] = []

    # Get all unique subjects from fact_timeline
    subjects = conn.execute(
        "SELECT DISTINCT subject FROM fact_timeline"
    ).fetchall()

    for (subject,) in subjects:
        # Check if any query word matches the subject
        subject_lower = subject.lower()
        match_score = sum(
            1 for w in query_words
            if w in subject_lower or subject_lower in w
        )

        # Also check if query words appear in the fact values themselves
        facts = conn.execute(
            "SELECT id, fact, timestamp, superseded_by, source_message_id "
            "FROM fact_timeline WHERE subject = ? ORDER BY timestamp",
            (subject,),
        ).fetchall()

        fact_match = sum(
            1 for _, fact, _, _, _ in facts
            for w in query_words
            if w in fact.lower()
        )

        if match_score > 0 or fact_match > 0:
            history = [
                {
                    "id": r[0],
                    "fact": r[1],
                    "timestamp": r[2],
                    "superseded": r[3] is not None,
                    "source_message_id": r[4],
                }
                for r in facts
            ]

            # Current fact = the one NOT superseded (or the latest one)
            current = [h for h in history if not h["superseded"]]
            if current:
                latest = current[-1]
            elif history:
                latest = history[-1]
            else:
                continue

            results.append({
                "subject": subject,
                "current_fact": latest["fact"],
                "current_timestamp": latest["timestamp"],
                "source_message_id": latest["source_message_id"],
                "history": history,
                "relevance": match_score + fact_match,
            })

    # Sort by relevance
    results.sort(key=lambda r: r["relevance"], reverse=True)
    return results


def search_consolidated(conn: sqlite3.Connection, query: str,
                        limit: int = 10) -> list[dict]:
    """
    Search summaries and consolidated data for broad, journey-style queries.

    Useful for queries that span many messages:

    - ``"Summarize the CarbonSense journey from founding to Series A"``
    - ``"What were the key turning points?"``
    - ``"How did X evolve over time?"``

    Searches both the ``summaries`` table (monthly/entity summaries) and
    ``fact_timeline`` (for contradiction-aware answers).  Results are ranked
    by keyword overlap with the query.

    Args:
        conn:  Open database connection.
        query: Natural-language query string.
        limit: Maximum number of results.

    Returns:
        List of result dicts, each with ``content``, ``period``,
        ``start_date``, ``end_date``, ``entity``, ``source``
        (``"summary"`` or ``"fact_timeline"`` or ``"fts"``), and ``score``.
    """
    results: list[dict] = []
    lower_query = query.lower()

    # ---- Search summaries table ----
    # Check for entity mentions in query to filter entity summaries
    target_entity = None
    all_entities = conn.execute(
        "SELECT DISTINCT entity FROM summaries WHERE entity != ''"
    ).fetchall()
    for (entity,) in all_entities:
        if entity.lower() in lower_query:
            target_entity = entity.lower()
            break

    # Also check for time range indicators
    time_filter_start = None
    _time_filter_end = None
    year_match = re.search(r"(\d{4})", query)
    if year_match:
        year = year_match.group(1)
        # Check for month mentions
        month_names = {
            "january": "01", "february": "02", "march": "03",
            "april": "04", "may": "05", "june": "06",
            "july": "07", "august": "08", "september": "09",
            "october": "10", "november": "11", "december": "12",
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "jun": "06", "jul": "07", "aug": "08", "sep": "09",
            "oct": "10", "nov": "11", "dec": "12",
        }
        for name, num in month_names.items():
            if name in lower_query:
                time_filter_start = f"{year}-{num}-01"
                _time_filter_end = f"{year}-{num}-31"
                break

    # Fetch relevant summaries
    if target_entity:
        summaries = conn.execute(
            "SELECT id, period, start_date, end_date, entity, summary, "
            "       key_facts, message_ids "
            "FROM summaries "
            "WHERE LOWER(entity) = ? OR entity = '' "
            "ORDER BY start_date",
            (target_entity,),
        ).fetchall()
    else:
        summaries = conn.execute(
            "SELECT id, period, start_date, end_date, entity, summary, "
            "       key_facts, message_ids "
            "FROM summaries ORDER BY start_date"
        ).fetchall()

    # Score summaries by keyword overlap with the query
    query_words = set(
        w.lower().strip("?.,!\"'")
        for w in query.split()
        if len(w) > 3
    )

    for row in summaries:
        summary_text = row[5]
        summary_lower = summary_text.lower()

        # Keyword overlap score
        overlap = sum(1 for w in query_words if w in summary_lower)

        # Time range relevance
        time_bonus = 0.0
        if time_filter_start and row[2]:
            if row[2] <= time_filter_start <= (row[3] or row[2]):
                time_bonus = 2.0
            elif time_filter_start <= row[2]:
                time_bonus = 1.0

        score = overlap + time_bonus
        if score > 0:
            results.append({
                "content": summary_text,
                "period": row[1],
                "start_date": row[2],
                "end_date": row[3],
                "entity": row[4],
                "key_facts": json.loads(row[6]) if row[6] else [],
                "source": "summary",
                "score": score,
            })

    # ---- Also check fact_timeline for contradiction-aware context ----
    contradiction_results = search_contradictions(conn, query)
    for cr in contradiction_results[:5]:
        history_text_parts = []
        for h in cr["history"]:
            status = " (superseded)" if h["superseded"] else " (current)"
            history_text_parts.append(
                f"{h['timestamp']}: {h['fact']}{status}"
            )
        history_text = "\n".join(history_text_parts)

        results.append({
            "content": f"[Fact Timeline: {cr['subject']}]\n"
                       f"Current: {cr['current_fact']}\n"
                       f"History:\n{history_text}",
            "period": "fact_timeline",
            "start_date": cr["history"][0]["timestamp"] if cr["history"] else "",
            "end_date": cr["current_timestamp"],
            "entity": "",
            "key_facts": [cr["current_fact"]],
            "source": "fact_timeline",
            "score": cr["relevance"] * 2,  # boost contradiction results
        })

    # ---- If no summary results, fall back to direct FTS search ----
    if not results:
        fts_terms = [w for w in query_words if len(w) > 3]
        if fts_terms:
            fts_query = _build_safe_fts_query(list(fts_terms)[:8])
            fts_results = _fts_search(conn, fts_query, limit=limit)
            for r in fts_results:
                results.append({
                    "content": r["content"],
                    "period": "message",
                    "start_date": r["timestamp"],
                    "end_date": r["timestamp"],
                    "entity": r["sender"],
                    "key_facts": [],
                    "source": "fts",
                    "score": abs(r.get("score", 0)),
                })

    # Sort by score and return top results
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]


def build_entity_summary_sheets(conn):
    """
    Build searchable entity profile summaries stored as special records
    in the summaries table with period='entity_profile'.

    .. deprecated:: 0.6.0
        Disabled by default in ``TrueMemoryEngine.consolidate()`` as of
        2026-04-24 per the MEMORIST-L4 research finding that the function
        produces fat profile rows that saturate top-1 retrieval by keyword
        match and leak superseded facts into contradiction scoring
        (+5.3 pts composite probe metric when disabled). The function
        itself is retained for backward-compatible imports and for users
        who re-enable it via ``TRUEMEMORY_ENTITY_SHEETS=1``.

        See ``CHANGELOG.md`` v0.6.0 or
        https://github.com/buildingjoshbetter/TrueMemory/issues
        for rationale.
    """
    import warnings
    warnings.warn(
        "build_entity_summary_sheets is deprecated as of v0.6.0 per "
        "MEMORIST-L4 research: its output harms contradiction resolution "
        "and retrieval precision. Disabled by default; set "
        "TRUEMEMORY_ENTITY_SHEETS=1 to re-enable. See CHANGELOG.md "
        "v0.6.0 or https://github.com/buildingjoshbetter/TrueMemory/issues "
        "for rationale.",
        DeprecationWarning,
        stacklevel=2,
    )
    from datetime import datetime, timezone

    # Get all entities with significant message counts
    entities = conn.execute(
        "SELECT sender, COUNT(*) as cnt FROM messages "
        "WHERE sender != '' GROUP BY sender HAVING cnt >= 5 "
        "ORDER BY cnt DESC"
    ).fetchall()

    now = datetime.now(timezone.utc).isoformat()
    count = 0

    for entity_name, msg_count in entities:
        # Gather all messages for this entity
        rows = conn.execute(
            "SELECT id, content, sender, recipient, timestamp, category, modality "
            "FROM messages WHERE LOWER(sender) = LOWER(?) OR LOWER(recipient) = LOWER(?) "
            "ORDER BY timestamp",
            (entity_name, entity_name)
        ).fetchall()

        messages = [{"id": r[0], "content": r[1], "sender": r[2], "recipient": r[3],
                     "timestamp": r[4], "category": r[5], "modality": r[6]} for r in rows]

        if not messages:
            continue

        # Build profile summary
        first_seen = messages[0]["timestamp"]
        last_seen = messages[-1]["timestamp"]

        # Count interactions per counterpart
        counterparts = defaultdict(int)
        for m in messages:
            if m["sender"].lower() == entity_name.lower():
                if m["recipient"]:
                    counterparts[m["recipient"]] += 1
            else:
                counterparts[m["sender"]] += 1

        top_contacts = sorted(counterparts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Extract key topics and notable messages
        notable_msgs = []
        for m in messages:
            if m["sender"].lower() == entity_name.lower() and len(m["content"]) > 50:
                sal = _message_salience(m)
                if sal > 0.3:
                    notable_msgs.append((m, sal))

        notable_msgs.sort(key=lambda x: x[1], reverse=True)
        top_notable = notable_msgs[:10]

        # Build summary text
        profile_parts = [
            f"Entity Profile: {entity_name}",
            f"Total messages: {msg_count}",
            f"Active period: {first_seen[:10]} to {last_seen[:10]}",
            f"Top contacts: {', '.join(f'{name} ({cnt})' for name, cnt in top_contacts)}",
            "",
            "Notable messages:",
        ]
        for msg, sal in top_notable:
            profile_parts.append(f"  [{msg['timestamp'][:10]}] {msg['content'][:200]}")

        summary_text = "\n".join(profile_parts)

        # Store as special summary record
        conn.execute(
            "INSERT INTO summaries "
            "(period, start_date, end_date, entity, summary, key_facts, message_ids, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "entity_profile",
                first_seen,
                last_seen,
                entity_name,
                summary_text,
                json.dumps([f"{msg_count} messages", f"{len(counterparts)} contacts"]),
                json.dumps([m[0]["id"] for m in top_notable]),
                now,
            )
        )
        count += 1

    conn.commit()
    return count


def build_structured_facts(conn):
    """
    Extract structured facts (team roster, locations, key events) and store
    as searchable summary records. Enables aggregation queries.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    all_msgs = _get_all_messages_chrono(conn)

    # --- Team roster extraction ---
    team_members = set()
    team_roles = {}
    role_patterns = re.compile(
        r'(\w+(?:\s+\w+)?)\s+(?:is|as)\s+(?:our\s+)?(?:the\s+)?'
        r'(CTO|CEO|COO|CFO|VP|lead|manager|engineer|designer|intern|cofounder|co-founder)',
        re.IGNORECASE
    )

    for msg in all_msgs:
        matches = role_patterns.finditer(msg["content"])
        for m in matches:
            name = m.group(1).strip()
            role = m.group(2).strip()
            if len(name) > 1 and name[0].isupper():
                team_members.add(name)
                team_roles[name] = role

    # Also add all senders as potential team/contact members
    hire_pattern = re.compile(r'(?:hired|brought on|recruited|onboarded)\s+(\w+(?:\s+\w+)?)', re.IGNORECASE)
    for msg in all_msgs:
        matches = hire_pattern.finditer(msg["content"])
        for m in matches:
            name = m.group(1).strip()
            if len(name) > 1 and name[0].isupper():
                team_members.add(name)

    if team_members:
        roster_text = "Team Roster:\n" + "\n".join(
            f"  {name}: {team_roles.get(name, 'member')}" for name in sorted(team_members)
        )
        conn.execute(
            "INSERT INTO summaries "
            "(period, start_date, end_date, entity, summary, key_facts, message_ids, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("structured_fact", "", "", "", roster_text, json.dumps(list(team_members)), "[]", now)
        )
        count += 1

    # --- Location extraction ---
    locations = set()
    location_patterns = [
        re.compile(r'(?:office|headquarters|hq)\s+(?:at|in|is)\s+(.+?)(?:[.,!?]|$)', re.IGNORECASE),
        re.compile(r'(?:moved|relocated|based)\s+(?:to|in)\s+(.+?)(?:[.,!?]|$)', re.IGNORECASE),
    ]

    for msg in all_msgs:
        for pat in location_patterns:
            matches = pat.finditer(msg["content"])
            for m in matches:
                loc = m.group(1).strip()
                if 3 < len(loc) < 50:
                    locations.add(loc)

    if locations:
        location_text = "Known Locations:\n" + "\n".join(f"  {loc}" for loc in sorted(locations))
        conn.execute(
            "INSERT INTO summaries "
            "(period, start_date, end_date, entity, summary, key_facts, message_ids, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("structured_fact", "", "", "", location_text, json.dumps(list(locations)), "[]", now)
        )
        count += 1

    conn.commit()
    return count
