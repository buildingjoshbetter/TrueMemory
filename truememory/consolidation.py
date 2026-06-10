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
        "FROM messages WHERE directive = 0 ORDER BY timestamp"
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
    # ── Informal / conversational contradiction patterns (#580) ──────
    # "actually X" — speaker corrects a previous statement
    {
        "category": "correction",
        "pattern": re.compile(
            r"(?:^|[.!?]\s+)(?:actually|correction:?|update:?)\s+"
            r"(.{5,80}?)(?:[.,!?]|$)",
            re.IGNORECASE | re.MULTILINE,
        ),
        "type": "informal_correction",
    },
    # "not X anymore" / "no longer X"
    {
        "category": "correction",
        "pattern": re.compile(
            r"(?:not\s+(.{3,40}?)\s+anymore|no\s+longer\s+(.{3,40}?))(?:[.,!?\s]|$)",
            re.IGNORECASE,
        ),
        "type": "negation_change",
    },
    # "X is wrong" / "X is incorrect" / "that's wrong" / "that's incorrect"
    {
        "category": "correction",
        "pattern": re.compile(
            r"(?:(?:that(?:'s|\s+is)|this\s+is|it(?:'s|\s+is))\s+"
            r"(?:wrong|incorrect|inaccurate|outdated|not\s+(?:right|correct|true)))",
            re.IGNORECASE,
        ),
        "type": "invalidation",
    },
    # "changed my mind about X" / "I was wrong about X"
    {
        "category": "correction",
        "pattern": re.compile(
            r"(?:changed?\s+(?:my|our)\s+mind\s+about"
            r"|(?:I|we)\s+(?:was|were)\s+wrong\s+about"
            r"|turns?\s+out)\s+"
            r"(.{3,60}?)(?:[.,!?]|$)",
            re.IGNORECASE,
        ),
        "type": "retraction",
    },
    # "scratch that" / "forget what I said" / "disregard" / "never mind"
    {
        "category": "correction",
        "pattern": re.compile(
            r"(?:scratch\s+that|forget\s+(?:what\s+I\s+said|that)|disregard"
            r"|never\s*mind(?:\s+(?:about|that))?)",
            re.IGNORECASE,
        ),
        "type": "retraction",
    },
    # "replaced X with Y"
    {
        "category": "technology",
        "pattern": re.compile(
            r"replaced?\s+([A-Z][\w\s.+-]{2,25}?)\s+"
            r"(?:with|by)\s+([A-Z][\w\s.+-]{2,25}?)(?:[.,!?\s]|$)",
            re.IGNORECASE,
        ),
        "type": "explicit_change",
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


def _compute_contradictions(all_msgs: list[dict]) -> tuple[list[tuple], list[tuple], list[dict]]:
    """Compute contradiction facts entirely in memory (no DB access).

    This is the expensive phase: regex scanning every message against every
    change pattern.  Factored out of :func:`detect_contradictions` so it can
    run OUTSIDE the write transaction (#591).

    Returns:
        ``(insert_rows, supersede_updates, contradictions)``

        *insert_rows* — list of tuples
        ``(local_id, subject, fact, source_message_id, timestamp,
        entity_scope, valid_from)`` ready for bulk INSERT.

        *supersede_updates* — list of ``(superseded_local_id,
        superseding_local_id, valid_to_timestamp)`` pairs expressed with
        local (in-memory) IDs so the caller can translate to real DB IDs
        after inserting.

        *contradictions* — the user-facing list of contradiction dicts.
    """
    # Local auto-increment for in-memory IDs (translated to real DB IDs
    # after the bulk INSERT).
    next_local_id = 1
    # subject -> [(fact_value, timestamp, msg_id, local_id)]
    fact_history: dict[str, list[tuple]] = defaultdict(list)
    contradictions: list[dict] = []
    insert_rows: list[tuple] = []
    supersede_updates: list[tuple] = []  # (old_local_id, new_local_id, valid_to)

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

                    entity_context = ""
                    nearby_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content[:100])
                    common_words = {"The", "This", "That", "We", "They", "Our", "My", "But", "And", "Just", "Not"}
                    entities = [n for n in nearby_nouns if n not in common_words and len(n) > 2]
                    if entities:
                        entity_context = entities[0].lower()

                    # Record the old fact if not already tracked
                    if subject not in fact_history or not fact_history[subject]:
                        old_local_id = next_local_id
                        next_local_id += 1
                        insert_rows.append((
                            old_local_id, subject, old_val, msg_id,
                            timestamp, entity_context, timestamp,
                        ))
                        fact_history[subject].append(
                            (old_val, timestamp, msg_id, old_local_id)
                        )

                    # Record the new fact and supersede the old
                    new_local_id = next_local_id
                    next_local_id += 1
                    insert_rows.append((
                        new_local_id, subject, new_val, msg_id,
                        timestamp, entity_context, timestamp,
                    ))

                    if fact_history[subject]:
                        prev = fact_history[subject][-1]
                        supersede_updates.append(
                            (prev[3], new_local_id, timestamp)
                        )

                    fact_history[subject].append(
                        (new_val, timestamp, msg_id, new_local_id)
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

                    entity_context = ""
                    nearby_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content)
                    common_words = {"The", "This", "That", "We", "They", "Our", "My", "But", "And", "Just", "Not"}
                    entities = [n for n in nearby_nouns if n not in common_words and len(n) > 2]
                    if entities:
                        entity_context = entities[0].lower()

                    subject = f"pricing_{entity_context}_{unit}" if entity_context else f"pricing_{unit}"

                    new_local_id = next_local_id
                    next_local_id += 1
                    insert_rows.append((
                        new_local_id, subject, price, msg_id,
                        timestamp, entity_context, timestamp,
                    ))

                    if subject in fact_history and fact_history[subject]:
                        prev = fact_history[subject][-1]
                        if prev[0] != price:
                            supersede_updates.append(
                                (prev[3], new_local_id, timestamp)
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
                        (price, timestamp, msg_id, new_local_id)
                    )

                elif pattern_def["type"] in ("location_change",
                                              "schedule_change"):
                    if groups:
                        fact_val = groups[0].strip()
                        if (len(fact_val) < _MIN_FACT_LEN
                                or len(fact_val) > _MAX_FACT_LEN):
                            continue

                        subject = _normalize_subject(fact_val, content)

                        entity_context = ""
                        nearby_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content[:100])
                        common_words = {"The", "This", "That", "We", "They", "Our", "My", "But", "And", "Just", "Not"}
                        entities = [n for n in nearby_nouns if n not in common_words and len(n) > 2]
                        if entities:
                            entity_context = entities[0].lower()

                        new_local_id = next_local_id
                        next_local_id += 1
                        insert_rows.append((
                            new_local_id, subject, fact_val, msg_id,
                            timestamp, entity_context, timestamp,
                        ))

                        if (subject in fact_history
                                and fact_history[subject]):
                            prev = fact_history[subject][-1]
                            if prev[0].lower() != fact_val.lower():
                                supersede_updates.append(
                                    (prev[3], new_local_id, timestamp)
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
                            (fact_val, timestamp, msg_id, new_local_id)
                        )

                elif pattern_def["type"] == "status_change" and groups:
                    person = groups[0].strip()
                    if (len(person) < _MIN_FACT_LEN
                            or len(person) > _MAX_FACT_LEN):
                        continue

                    verb_match = re.search(
                        r"(quit|left|started|joined|hired|fired|"
                        r"resigned|promoted)",
                        content, re.IGNORECASE,
                    )
                    verb = verb_match.group(1).lower() if verb_match else "changed"

                    subject = f"status_{person.lower()}"
                    fact_val = f"{person} {verb}"
                    entity_context = person.lower()

                    new_local_id = next_local_id
                    next_local_id += 1
                    insert_rows.append((
                        new_local_id, subject, fact_val, msg_id,
                        timestamp, entity_context, timestamp,
                    ))
                    fact_history[subject].append(
                        (fact_val, timestamp, msg_id, new_local_id)
                    )

                elif pattern_def["type"] == "informal_correction" and groups:
                    # "actually X", "correction: X", "update: X"
                    fact_val = groups[0].strip()
                    if (len(fact_val) < _MIN_FACT_LEN
                            or len(fact_val) > _MAX_FACT_LEN):
                        continue
                    subject = _normalize_subject(fact_val, content)
                    entity_context = ""
                    nearby_nouns = re.findall(
                        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                        content[:100],
                    )
                    common_words = {"The", "This", "That", "We", "They",
                                    "Our", "My", "But", "And", "Just", "Not",
                                    "Actually", "Correction", "Update"}
                    entities = [n for n in nearby_nouns
                                if n not in common_words and len(n) > 2]
                    if entities:
                        entity_context = entities[0].lower()

                    new_local_id = next_local_id
                    next_local_id += 1
                    insert_rows.append((
                        new_local_id, subject, fact_val, msg_id,
                        timestamp, entity_context, timestamp,
                    ))

                    if subject in fact_history and fact_history[subject]:
                        prev = fact_history[subject][-1]
                        supersede_updates.append(
                            (prev[3], new_local_id, timestamp)
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
                        (fact_val, timestamp, msg_id, new_local_id)
                    )

                elif pattern_def["type"] == "negation_change":
                    # "not X anymore" / "no longer X" — two capture groups
                    fact_val = (groups[0] or groups[1] or "").strip()
                    if (len(fact_val) < _MIN_FACT_LEN
                            or len(fact_val) > _MAX_FACT_LEN):
                        continue
                    subject = _normalize_subject(fact_val, content)

                    new_local_id = next_local_id
                    next_local_id += 1
                    insert_rows.append((
                        new_local_id, subject, f"no longer {fact_val}",
                        msg_id, timestamp, "", timestamp,
                    ))

                    if subject in fact_history and fact_history[subject]:
                        prev = fact_history[subject][-1]
                        supersede_updates.append(
                            (prev[3], new_local_id, timestamp)
                        )
                        contradictions.append({
                            "subject": subject,
                            "old_fact": prev[0],
                            "new_fact": f"no longer {fact_val}",
                            "old_timestamp": prev[1],
                            "new_timestamp": timestamp,
                            "source_message_id": msg_id,
                        })

                    fact_history[subject].append(
                        (f"no longer {fact_val}", timestamp, msg_id,
                         new_local_id)
                    )

                elif pattern_def["type"] == "invalidation":
                    # "that's wrong" / "that's incorrect" — no capture
                    # group, just records the event as a contradiction
                    # signal on the message itself.
                    new_local_id = next_local_id
                    next_local_id += 1
                    insert_rows.append((
                        new_local_id, "_invalidation",
                        match.group(0).strip(), msg_id,
                        timestamp, "", timestamp,
                    ))
                    fact_history["_invalidation"].append(
                        (match.group(0).strip(), timestamp, msg_id,
                         new_local_id)
                    )

                elif pattern_def["type"] == "retraction" and groups:
                    # "changed my mind about X", "I was wrong about X",
                    # "turns out X"
                    fact_val = groups[0].strip()
                    if (len(fact_val) < _MIN_FACT_LEN
                            or len(fact_val) > _MAX_FACT_LEN):
                        continue
                    subject = _normalize_subject(fact_val, content)

                    new_local_id = next_local_id
                    next_local_id += 1
                    insert_rows.append((
                        new_local_id, subject, fact_val, msg_id,
                        timestamp, "", timestamp,
                    ))

                    if subject in fact_history and fact_history[subject]:
                        prev = fact_history[subject][-1]
                        supersede_updates.append(
                            (prev[3], new_local_id, timestamp)
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
                        (fact_val, timestamp, msg_id, new_local_id)
                    )

                elif pattern_def["type"] == "retraction" and not groups:
                    # "scratch that", "forget what I said", "disregard",
                    # "never mind" — no capture group
                    new_local_id = next_local_id
                    next_local_id += 1
                    insert_rows.append((
                        new_local_id, "_retraction",
                        match.group(0).strip(), msg_id,
                        timestamp, "", timestamp,
                    ))
                    fact_history["_retraction"].append(
                        (match.group(0).strip(), timestamp, msg_id,
                         new_local_id)
                    )

    return insert_rows, supersede_updates, contradictions


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
    - **Informal corrections**: ``"actually X"``, ``"correction: X"``.
    - **Negation changes**: ``"not X anymore"``, ``"no longer X"``.
    - **Invalidations**: ``"that's wrong"``, ``"that's incorrect"``.
    - **Retractions**: ``"changed my mind about X"``, ``"scratch that"``.

    For each detected change, a record is inserted into ``fact_timeline``
    with the subject, fact value, source message, and timestamp.  When a
    newer fact supersedes an older one, the older record's
    ``superseded_by`` column is updated.

    The implementation follows a three-phase pattern (#591) so that
    expensive regex computation does NOT hold the SQLite write lock:

    1. **Read** messages (short read, or no transaction).
    2. **Compute** contradictions entirely in memory
       (:func:`_compute_contradictions`).
    3. **Write** results in one short, atomic transaction.

    Returns:
        List of contradiction records, each a dict with ``subject``,
        ``old_fact``, ``new_fact``, ``old_timestamp``, ``new_timestamp``,
        ``source_message_id``.
    """
    # --- Phase 1: read ---------------------------------------------------
    all_msgs = _get_all_messages_chrono(conn)

    # --- Phase 2: compute (no transaction held) --------------------------
    insert_rows, supersede_updates, contradictions = _compute_contradictions(
        all_msgs,
    )

    # --- Phase 3: write (short atomic transaction) -----------------------
    # Same manual-transaction pattern used by build_summaries (#401).
    if conn.in_transaction:
        conn.commit()
    prev_isolation = conn.isolation_level
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM fact_timeline")

        # Bulk-insert all fact rows and build local_id -> real_db_id map.
        local_to_db: dict[int, int] = {}
        for row in insert_rows:
            local_id = row[0]
            cursor = conn.execute(
                "INSERT INTO fact_timeline "
                "(subject, fact, source_message_id, timestamp, "
                " entity_scope, valid_from) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                row[1:],
            )
            local_to_db[local_id] = cursor.lastrowid

        # Apply supersession updates using real DB IDs.
        for old_local, new_local, valid_to in supersede_updates:
            old_db_id = local_to_db[old_local]
            new_db_id = local_to_db[new_local]
            conn.execute(
                "UPDATE fact_timeline SET superseded_by = ?, status = 'superseded' WHERE id = ?",
                (new_db_id, old_db_id),
            )
            conn.execute(
                "UPDATE fact_timeline SET valid_to = ? WHERE id = ?",
                (valid_to, old_db_id),
            )

        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.isolation_level = prev_isolation

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

    now = datetime.now(timezone.utc).isoformat()
    # Collect all summary rows first WITHOUT opening a write transaction, so the
    # heavy salience/sentence scoring below does not hold the SQLite write lock.
    # The clear + bulk insert happens in one short transaction at the very end
    # (see #401: the old code ran DELETE first and held the write lock for the
    # entire 30-60s computation, causing "database is locked" for concurrent
    # writers once the 10s busy_timeout was exceeded).
    rows: list[tuple] = []

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

        rows.append((
            "monthly",
            start_date,
            end_date,
            "",
            summary_text,
            json.dumps(key_facts),
            json.dumps(message_ids),
            now,
        ))

    # ---- Per-entity summaries ----
    by_entity: dict[str, list[dict]] = defaultdict(list)
    for msg in all_msgs:
        if msg["sender"]:
            by_entity[msg["sender"].lower()].append(msg)

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

            rows.append((
                "entity_monthly",
                start_date,
                end_date,
                entity,
                summary_text,
                json.dumps(key_facts),
                json.dumps(message_ids),
                now,
            ))

    # Short, explicit, atomic write: hold the write lock only for the clear +
    # bulk insert, not the computation above. We drive the transaction manually
    # and switch the connection to autocommit (isolation_level=None) for the
    # duration so pysqlite does NOT also manage transactions implicitly — this
    # makes our BEGIN IMMEDIATE / COMMIT / ROLLBACK authoritative regardless of
    # the connection's configured isolation_level (avoids both "cannot start a
    # transaction within a transaction" and pysqlite's commit()/rollback()
    # becoming no-ops after a manual BEGIN). Any pending implicit transaction is
    # flushed first so the write lock is acquired cleanly, and isolation_level
    # is always restored. Rollback-on-error guarantees the summaries table is
    # never left emptied without its replacement rows.
    if conn.in_transaction:
        conn.commit()
    prev_isolation = conn.isolation_level
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM summaries")
        if rows:
            conn.executemany(
                "INSERT INTO summaries "
                "(period, start_date, end_date, entity, summary, "
                " key_facts, message_ids, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.isolation_level = prev_isolation
    return len(rows)


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
            "SELECT id, fact, timestamp, superseded_by, source_message_id, "
            "       COALESCE(status, 'active') "
            "FROM fact_timeline WHERE subject = ? ORDER BY timestamp",
            (subject,),
        ).fetchall()

        fact_match = sum(
            1 for _, fact, _, _, _, _ in facts
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
                    "status": r[5],
                }
                for r in facts
            ]

            # Current fact = the one NOT superseded (or the latest one)
            current = [h for h in history
                        if not h["superseded"]
                        and h.get("status", "active") != "superseded"]
            if current:
                latest = current[-1]
            elif history:
                latest = history[-1]
            else:
                continue

            # Penalise relevance when the latest fact is superseded
            # (still retrievable, but ranked lower).
            relevance = match_score + fact_match
            if latest.get("status") == "superseded" or latest.get("superseded"):
                relevance *= 0.5

            results.append({
                "id": latest["source_message_id"],
                "subject": subject,
                "current_fact": latest["fact"],
                "current_timestamp": latest["timestamp"],
                "source_message_id": latest["source_message_id"],
                "history": history,
                "relevance": relevance,
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
                "id": f"summary_{row[0]}",
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
            "id": cr["id"],
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
                    "id": r.get("id"),
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
        "WHERE sender != '' AND directive = 0 GROUP BY sender HAVING cnt >= 5 "
        "ORDER BY cnt DESC"
    ).fetchall()

    now = datetime.now(timezone.utc).isoformat()
    count = 0

    for entity_name, msg_count in entities:
        # Gather all messages for this entity
        rows = conn.execute(
            "SELECT id, content, sender, recipient, timestamp, category, modality "
            "FROM messages WHERE (LOWER(sender) = LOWER(?) OR LOWER(recipient) = LOWER(?)) "
            "AND directive = 0 ORDER BY timestamp",
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


def _compute_structured_facts(all_msgs: list[dict]) -> list[tuple]:
    """Compute structured fact rows entirely in memory (no DB access).

    Factored out of :func:`build_structured_facts` so the expensive regex
    scanning runs OUTSIDE the write transaction (#591).

    Returns:
        List of row tuples ready for INSERT into the ``summaries`` table:
        ``(period, start_date, end_date, entity, summary, key_facts_json,
        message_ids_json, created_at)``.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows: list[tuple] = []

    # --- Team roster extraction ---
    team_members: set[str] = set()
    team_roles: dict[str, str] = {}
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

    hire_pattern = re.compile(
        r'(?:hired|brought on|recruited|onboarded)\s+(\w+(?:\s+\w+)?)',
        re.IGNORECASE,
    )
    for msg in all_msgs:
        matches = hire_pattern.finditer(msg["content"])
        for m in matches:
            name = m.group(1).strip()
            if len(name) > 1 and name[0].isupper():
                team_members.add(name)

    if team_members:
        roster_text = "Team Roster:\n" + "\n".join(
            f"  {name}: {team_roles.get(name, 'member')}"
            for name in sorted(team_members)
        )
        rows.append((
            "structured_fact", "", "", "", roster_text,
            json.dumps(list(team_members)), "[]", now,
        ))

    # --- Location extraction ---
    locations: set[str] = set()
    location_patterns = [
        re.compile(
            r'(?:office|headquarters|hq)\s+(?:at|in|is)\s+(.+?)(?:[.,!?]|$)',
            re.IGNORECASE,
        ),
        re.compile(
            r'(?:moved|relocated|based)\s+(?:to|in)\s+(.+?)(?:[.,!?]|$)',
            re.IGNORECASE,
        ),
    ]

    for msg in all_msgs:
        for pat in location_patterns:
            matches = pat.finditer(msg["content"])
            for m in matches:
                loc = m.group(1).strip()
                if 3 < len(loc) < 50:
                    locations.add(loc)

    if locations:
        location_text = "Known Locations:\n" + "\n".join(
            f"  {loc}" for loc in sorted(locations)
        )
        rows.append((
            "structured_fact", "", "", "", location_text,
            json.dumps(list(locations)), "[]", now,
        ))

    return rows


def build_structured_facts(conn):
    """
    Extract structured facts (team roster, locations, key events) and store
    as searchable summary records. Enables aggregation queries.

    The implementation follows a three-phase pattern (#591) so that
    expensive regex computation does NOT hold the SQLite write lock:

    1. **Read** messages (short read, or no transaction).
    2. **Compute** structured facts entirely in memory
       (:func:`_compute_structured_facts`).
    3. **Write** results in one short, atomic transaction.
    """
    # --- Phase 1: read ---------------------------------------------------
    all_msgs = _get_all_messages_chrono(conn)

    # --- Phase 2: compute (no transaction held) --------------------------
    rows = _compute_structured_facts(all_msgs)

    # --- Phase 3: write (short atomic transaction) -----------------------
    if conn.in_transaction:
        conn.commit()
    prev_isolation = conn.isolation_level
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        # Remove old structured_fact rows (don't touch other summary types).
        conn.execute("DELETE FROM summaries WHERE period = 'structured_fact'")
        if rows:
            conn.executemany(
                "INSERT INTO summaries "
                "(period, start_date, end_date, entity, summary, "
                " key_facts, message_ids, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.isolation_level = prev_isolation

    return len(rows)
