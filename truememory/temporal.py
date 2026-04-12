"""
TrueMemory Temporal Reasoning Module (L2 Enhancement)
====================================================

Adds temporal intelligence to TrueMemory search results. This is one of
TrueMemory's biggest competitive advantages: every benchmarked competitor
(Mem0, LangMem, ChromaDB, Cognee) scored 2/10 or less on temporal queries.
FTS5 + SQL timestamp filtering gives us instant, free temporal filtering
that none of them can match.

Temporal queries this module handles:
    - "What happened in the month after Demo Day (June 15, 2025)?"
    - "How did CarbonSense's MRR grow over time?"
    - "When did Jordan quit his job and what happened in the first month?"
    - "What are Jordan's upcoming events as of January 2026?"
    - "What was Jordan's health trajectory from early 2025 to late 2025?"

Design:
    - Temporal detection uses regex pattern matching (no LLM needed).
    - Date parsing handles natural language ("early 2025", "June 15, 2025",
      "last month", "January 2026") and converts to ISO timestamps.
    - Temporal filtering is a SQL WHERE clause -- it's free and instant.
    - This module ENHANCES existing search results; it does not replace them.
"""

import re
import sqlite3
from datetime import datetime, timedelta

from truememory.storage import _row_to_dict, _SELECT_COLS


# ---------------------------------------------------------------------------
# Month name mapping
# ---------------------------------------------------------------------------

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

# Patterns for month names in regex alternation
_MONTH_PATTERN = "|".join(_MONTH_NAMES.keys())


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_date_reference(text: str) -> str | None:
    """
    Extract a date from natural language text and return an ISO date string.

    Supported formats (case-insensitive):
        - "June 15, 2025"      -> "2025-06-15"
        - "June 15 2025"       -> "2025-06-15"
        - "15 June 2025"       -> "2025-06-15"
        - "2025-06-15"         -> "2025-06-15"
        - "2025-06"            -> "2025-06-01"
        - "2025"               -> "2025-01-01"
        - "January 2026"       -> "2026-01-01"
        - "Jan 2026"           -> "2026-01-01"
        - "early 2025"         -> "2025-01-01"
        - "mid 2025"           -> "2025-05-01"
        - "late 2025"          -> "2025-09-01"
        - "early January 2025" -> "2025-01-01"

    Returns:
        ISO date string (``"YYYY-MM-DD"``) or ``None`` if no date found.
    """
    if not text:
        return None

    text = text.strip()

    # ISO format: "2025-06-15"
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # ISO partial: "2025-06"
    m = re.search(r"\b(\d{4})-(\d{2})\b", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-01"

    # "June 15, 2025" or "June 15 2025"
    m = re.search(
        rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})\b",
        text, re.IGNORECASE,
    )
    if m:
        month = _MONTH_NAMES[m.group(1).lower()]
        day = int(m.group(2))
        year = int(m.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"

    # "15 June 2025"
    m = re.search(
        rf"\b(\d{{1,2}})\s+({_MONTH_PATTERN})\s+(\d{{4}})\b",
        text, re.IGNORECASE,
    )
    if m:
        day = int(m.group(1))
        month = _MONTH_NAMES[m.group(2).lower()]
        year = int(m.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"

    # "early/mid/late [Month] YYYY" (e.g., "early January 2025")
    m = re.search(
        rf"\b(early|mid|late)\s+({_MONTH_PATTERN})\s+(\d{{4}})\b",
        text, re.IGNORECASE,
    )
    if m:
        qualifier = m.group(1).lower()
        month = _MONTH_NAMES[m.group(2).lower()]
        year = int(m.group(3))
        if qualifier == "early":
            return f"{year:04d}-{month:02d}-01"
        elif qualifier == "mid":
            return f"{year:04d}-{month:02d}-15"
        else:  # late
            # Last day varies, use 25th as a reasonable "late" anchor
            return f"{year:04d}-{month:02d}-25"

    # "early/mid/late YYYY" (no month)
    m = re.search(r"\b(early|mid|late)\s+(\d{4})\b", text, re.IGNORECASE)
    if m:
        qualifier = m.group(1).lower()
        year = int(m.group(2))
        if qualifier == "early":
            return f"{year:04d}-01-01"
        elif qualifier == "mid":
            return f"{year:04d}-05-01"
        else:  # late
            return f"{year:04d}-09-01"

    # "January 2026" or "Jan 2026" (month + year, no day)
    m = re.search(
        rf"\b({_MONTH_PATTERN})\s+(\d{{4}})\b",
        text, re.IGNORECASE,
    )
    if m:
        month = _MONTH_NAMES[m.group(1).lower()]
        year = int(m.group(2))
        return f"{year:04d}-{month:02d}-01"

    # Bare year: "2025"
    m = re.search(r"\b(20\d{2})\b", text)
    if m:
        return f"{m.group(1)}-01-01"

    return None


def _end_of_month(year: int, month: int) -> str:
    """Return the last day of a given month as an ISO date string."""
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1
    last_day = (datetime(next_year, next_month, 1) - timedelta(days=1)).day
    return f"{year:04d}-{month:02d}-{last_day:02d}"


def _add_months(iso_date: str, months: int) -> str:
    """Add (or subtract) months from an ISO date string."""
    dt = datetime.fromisoformat(iso_date)
    new_month = dt.month + months
    new_year = dt.year + (new_month - 1) // 12
    new_month = ((new_month - 1) % 12) + 1
    # Clamp day to valid range for the target month
    max_day = (
        datetime(new_year + (1 if new_month == 12 else 0),
                 (new_month % 12) + 1 if new_month < 12 else 1, 1)
        - timedelta(days=1)
    ).day
    new_day = min(dt.day, max_day)
    return f"{new_year:04d}-{new_month:02d}-{new_day:02d}"


# ---------------------------------------------------------------------------
# Temporal intent detection
# ---------------------------------------------------------------------------

def detect_temporal_intent(query: str) -> dict:
    """
    Analyze a query to detect temporal constraints.

    Scans for temporal keywords and date references to build a constraint
    dict that downstream functions use for filtering and sorting.

    Detection rules:
        - ``"after [date/event]"``               -> sets ``after``
        - ``"before [date/event]"``               -> sets ``before``
        - ``"in [month] [year]"``                 -> month boundary window
        - ``"from X to Y"`` / ``"between X and Y"`` -> sets both ``after`` and ``before``
        - ``"last month/week/year"``              -> relative dates
        - ``"over time"``, ``"trajectory"``, ``"grew"`` etc. -> trajectory mode
        - ``"upcoming"``, ``"future"``, ``"next"``   -> after=reference_date
        - ``"as of [date]"``                       -> before=date (reference point)
        - Parenthesized dates like ``"(June 15, 2025)"`` are parsed directly.

    Args:
        query: Natural language query string.

    Returns:
        Dict with keys::

            {
                'has_temporal': bool,
                'after': str or None,      # ISO date
                'before': str or None,     # ISO date
                'sort_by_time': bool,       # True if chronological order matters
                'is_trajectory': bool,      # True if asking about change over time
                'reference_date': str or None,  # Extracted date reference
            }
    """
    result = {
        "has_temporal": False,
        "after": None,
        "before": None,
        "sort_by_time": False,
        "is_trajectory": False,
        "reference_date": None,
    }

    q = query.strip()
    ql = q.lower()

    # --- Trajectory / "over time" detection ---
    trajectory_patterns = [
        r"\bover\s+time\b",
        r"\btrajectory\b",
        r"\bgrew\b",
        r"\bgrow\b",
        r"\bgrowth\b",
        r"\bchanged?\b",
        r"\bevolved?\b",
        r"\bevolution\b",
        r"\bprogress(?:ion|ed)?\b",
        r"\bdecline[ds]?\b",
        r"\bimproved?\b",
        r"\bdeteriorate[ds]?\b",
        r"\bfrom\s+early\b.*\bto\s+late\b",
        r"\bfrom\s+.+\bto\s+",
        r"\btimeline\b",
        r"\bchronolog",
    ]
    for pat in trajectory_patterns:
        if re.search(pat, ql):
            result["is_trajectory"] = True
            result["sort_by_time"] = True
            result["has_temporal"] = True
            break

    # --- "from X to Y" range ---
    m = re.search(
        r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\?|$|\.)",
        ql,
    )
    if m:
        from_date = parse_date_reference(m.group(1))
        to_date = parse_date_reference(m.group(2))
        if from_date:
            result["after"] = from_date
            result["has_temporal"] = True
        if to_date:
            result["before"] = to_date
            result["has_temporal"] = True

    # --- "between X and Y" range ---
    m = re.search(
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$|\.)",
        ql,
    )
    if m and not result["after"]:
        from_date = parse_date_reference(m.group(1))
        to_date = parse_date_reference(m.group(2))
        if from_date:
            result["after"] = from_date
            result["has_temporal"] = True
        if to_date:
            result["before"] = to_date
            result["has_temporal"] = True

    # --- "in [Month] [Year]" → month boundaries ---
    m = re.search(
        rf"\bin\s+({_MONTH_PATTERN})\s+(\d{{4}})\b",
        ql,
    )
    if m and result["after"] is None and result["before"] is None:
        month = _MONTH_NAMES[m.group(1).lower()]
        year = int(m.group(2))
        result["after"] = f"{year:04d}-{month:02d}-01"
        result["before"] = _end_of_month(year, month)
        result["has_temporal"] = True

    # --- "in [Year]" (bare year) ---
    if result["after"] is None and result["before"] is None:
        m = re.search(r"\bin\s+(20\d{2})\b", ql)
        if m:
            year = int(m.group(1))
            result["after"] = f"{year:04d}-01-01"
            result["before"] = f"{year:04d}-12-31"
            result["has_temporal"] = True

    # --- "after [date/event]" ---
    m = re.search(r"\bafter\s+(.+?)(?:\?|$|,|\band\b)", ql)
    if m and result["after"] is None:
        # Look for a date, including in parentheses in the original query
        after_text = m.group(1)
        # Also check the original query for parenthesized dates near this match
        paren_match = re.search(r"\(([^)]+)\)", q[m.start():])
        if paren_match:
            after_text = after_text + " " + paren_match.group(1)
        parsed = parse_date_reference(after_text)
        if parsed:
            result["after"] = parsed
            result["reference_date"] = parsed
            result["has_temporal"] = True
            result["sort_by_time"] = True

    # --- "before [date/event]" ---
    m = re.search(r"\bbefore\s+(.+?)(?:\?|$|,|\band\b)", ql)
    if m and result["before"] is None:
        before_text = m.group(1)
        paren_match = re.search(r"\(([^)]+)\)", q[m.start():])
        if paren_match:
            before_text = before_text + " " + paren_match.group(1)
        parsed = parse_date_reference(before_text)
        if parsed:
            result["before"] = parsed
            result["has_temporal"] = True

    # --- "as of [date]" → sets a reference point (everything before/up to) ---
    m = re.search(r"\bas\s+of\s+(.+?)(?:\?|$|,)", ql)
    if m:
        parsed = parse_date_reference(m.group(1))
        if parsed:
            result["reference_date"] = parsed
            result["has_temporal"] = True
            # "upcoming as of X" means after=X, otherwise before=X
            if re.search(r"\b(upcoming|future|next|scheduled)\b", ql):
                if result["after"] is None:
                    result["after"] = parsed
            else:
                if result["before"] is None:
                    result["before"] = parsed

    # --- "upcoming" / "future" / "next" / "scheduled" ---
    if re.search(r"\b(upcoming|future|next|scheduled)\b", ql):
        result["has_temporal"] = True
        result["sort_by_time"] = True
        # If we have a reference date from "as of", use it; otherwise leave
        # after as None (caller should set it to "now" or latest timestamp).
        if result["after"] is None and result["reference_date"]:
            result["after"] = result["reference_date"]

    # --- "in the [first/last] month/week after..." ---
    m = re.search(
        r"\b(?:in\s+the\s+)?(?:first|next)\s+(month|week|year)\b",
        ql,
    )
    if m and result["after"] is not None and result["before"] is None:
        unit = m.group(1)
        try:
            after_dt = datetime.fromisoformat(result["after"])
            if unit == "month":
                end_dt = after_dt + timedelta(days=31)
            elif unit == "week":
                end_dt = after_dt + timedelta(days=7)
            else:  # year
                end_dt = after_dt + timedelta(days=365)
            result["before"] = end_dt.strftime("%Y-%m-%d")
            result["sort_by_time"] = True
        except ValueError:
            pass

    # --- "month after [date]" (without "first"/"next" prefix) ---
    m = re.search(
        r"\bmonth\s+after\b",
        ql,
    )
    if m and result["after"] is not None and result["before"] is None:
        try:
            after_dt = datetime.fromisoformat(result["after"])
            end_dt = after_dt + timedelta(days=31)
            result["before"] = end_dt.strftime("%Y-%m-%d")
            result["sort_by_time"] = True
        except ValueError:
            pass

    # --- "last month/week/year" (relative) ---
    m = re.search(r"\blast\s+(month|week|year)\b", ql)
    if m and not result["has_temporal"]:
        # Without a known "now", we can't compute exact dates, but we flag it
        # and let the caller supply the reference via the latest timestamp.
        result["has_temporal"] = True
        result["sort_by_time"] = True

    # --- Parenthesized date extraction as fallback reference ---
    if result["reference_date"] is None:
        paren = re.search(r"\(([^)]+)\)", q)
        if paren:
            parsed = parse_date_reference(paren.group(1))
            if parsed:
                result["reference_date"] = parsed
                result["has_temporal"] = True
                # If we got a date from parens and "after" was flagged but
                # not resolved, set it now.
                if result["after"] is None and re.search(r"\bafter\b", ql):
                    result["after"] = parsed
                    result["sort_by_time"] = True
                    # Also set "month after" window if applicable
                    if re.search(r"\bmonth\s+after\b", ql):
                        try:
                            dt = datetime.fromisoformat(parsed)
                            result["before"] = (
                                dt + timedelta(days=31)
                            ).strftime("%Y-%m-%d")
                        except ValueError:
                            pass

    # --- Final: extract any standalone date as reference if none found ---
    if not result["has_temporal"]:
        ref = parse_date_reference(q)
        if ref:
            result["reference_date"] = ref
            result["has_temporal"] = True

    return result


# ---------------------------------------------------------------------------
# Temporal search and timeline
# ---------------------------------------------------------------------------

def search_temporal(
    conn: sqlite3.Connection,
    query: str,
    fts_results: list[dict] | None = None,
    hybrid_results: list[dict] | None = None,
    limit: int = 10,
) -> list[dict]:
    """
    Apply temporal filtering and re-ranking to search results.

    This function **enhances** existing search results -- pass in results
    from FTS5 or hybrid search, and this adds temporal intelligence on top.

    Processing steps:
        1. Detect temporal intent from the query.
        2. If no temporal intent, return the input results unchanged.
        3. If temporal intent is detected:
           a. Filter results to the detected time window (``after``/``before``).
           b. If ``is_trajectory``, sort results chronologically.
           c. If not enough results remain after filtering, fall back to a
              fresh SQL query on the messages table with the time constraints.
        4. Return the filtered/re-ranked results, trimmed to ``limit``.

    Args:
        conn:           Open database connection.
        query:          Natural language query string.
        fts_results:    Results from FTS5 search (list of message dicts).
        hybrid_results: Results from hybrid (FTS5+vector) search.
        limit:          Maximum number of results to return.

    Returns:
        List of message dicts, temporally filtered and ordered.
    """
    intent = detect_temporal_intent(query)

    # Merge input result sets (prefer hybrid if both are provided)
    results = []
    if hybrid_results:
        results = list(hybrid_results)
    elif fts_results:
        results = list(fts_results)

    if not intent["has_temporal"]:
        return results[:limit]

    after = intent["after"]
    before = intent["before"]

    # --- Filter by time window ---
    if after or before:
        filtered = []
        for r in results:
            ts = r.get("timestamp", "")
            if not ts:
                continue
            if after and ts < after:
                continue
            if before and ts > before + "T23:59:59":
                continue
            filtered.append(r)
        results = filtered

    # --- If trajectory, sort chronologically ---
    if intent["is_trajectory"] or intent["sort_by_time"]:
        results.sort(key=lambda r: r.get("timestamp", ""))

    # --- If not enough results, do a fresh SQL query with time constraints ---
    if len(results) < limit and (after or before):
        existing_ids = {r.get("id") for r in results}
        extra = get_timeline(conn, after=after, before=before)
        for msg in extra:
            if msg["id"] not in existing_ids:
                results.append(msg)
                existing_ids.add(msg["id"])

        # Re-sort if trajectory
        if intent["is_trajectory"] or intent["sort_by_time"]:
            results.sort(key=lambda r: r.get("timestamp", ""))

    return results[:limit]


def get_timeline(
    conn: sqlite3.Connection,
    entity: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """
    Get a chronological timeline of messages.

    Useful for trajectory queries ("how did X change over time") and for
    backfilling when temporal search produces too few results from FTS5.

    Filters can be combined: get all messages from ``entity`` within a
    time range.

    Args:
        conn:   Open database connection.
        entity: If provided, restrict to messages where the entity is
                either the sender or the recipient (case-insensitive).
        after:  Inclusive lower bound timestamp (ISO string).
        before: Inclusive upper bound timestamp (ISO string).

    Returns:
        List of message dicts in chronological order.
    """
    clauses: list[str] = []
    params: list[str] = []

    if after is not None:
        clauses.append("timestamp >= ?")
        params.append(after)
    if before is not None:
        # Include the full day for date-only bounds
        if len(before) == 10:  # "YYYY-MM-DD"
            clauses.append("timestamp <= ?")
            params.append(before + "T23:59:59")
        else:
            clauses.append("timestamp <= ?")
            params.append(before)
    if entity is not None:
        entity_lower = entity.lower()
        clauses.append("(LOWER(sender) = ? OR LOWER(recipient) = ?)")
        params.extend([entity_lower, entity_lower])

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT {_SELECT_COLS} FROM messages{where} ORDER BY timestamp"

    rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Episode boundaries (B1: 6-hour gap heuristic)
# ---------------------------------------------------------------------------

def detect_episodes(conn, gap_hours=6):
    """
    Group messages into episodes using a time-gap heuristic.
    Messages within gap_hours of each other belong to the same episode.
    Stores episodes in the episodes table and updates episode_id on messages.

    Args:
        conn:      Open database connection.
        gap_hours: Maximum gap (in hours) between consecutive messages
                   within the same episode (default 6).

    Returns:
        Number of episodes detected.
    """
    rows = conn.execute(
        "SELECT id, timestamp FROM messages WHERE timestamp != '' ORDER BY timestamp"
    ).fetchall()

    if not rows:
        return 0

    # Clear existing episodes
    conn.execute("DELETE FROM episodes")
    conn.execute("UPDATE messages SET episode_id = NULL")

    episodes = []
    current_episode_msgs = [rows[0]]
    gap_delta = timedelta(hours=gap_hours)

    for i in range(1, len(rows)):
        msg_id, ts = rows[i]
        prev_id, prev_ts = rows[i - 1]

        try:
            # Parse timestamps (handle both date-only and datetime formats)
            curr_dt = datetime.fromisoformat(ts.replace('Z', '+00:00').split('+')[0])
            prev_dt = datetime.fromisoformat(prev_ts.replace('Z', '+00:00').split('+')[0])

            if (curr_dt - prev_dt) > gap_delta:
                # New episode
                episodes.append(current_episode_msgs)
                current_episode_msgs = [(msg_id, ts)]
            else:
                current_episode_msgs.append((msg_id, ts))
        except (ValueError, TypeError):
            current_episode_msgs.append((msg_id, ts))

    # Don't forget the last episode
    if current_episode_msgs:
        episodes.append(current_episode_msgs)

    # Store episodes and update messages
    for ep_msgs in episodes:
        if not ep_msgs:
            continue

        start_time = ep_msgs[0][1]
        end_time = ep_msgs[-1][1]
        msg_count = len(ep_msgs)

        cursor = conn.execute(
            "INSERT INTO episodes (start_time, end_time, message_count) VALUES (?, ?, ?)",
            (start_time, end_time, msg_count)
        )
        episode_id = cursor.lastrowid

        msg_ids = [m[0] for m in ep_msgs]
        for mid in msg_ids:
            conn.execute("UPDATE messages SET episode_id = ? WHERE id = ?", (episode_id, mid))

    conn.commit()
    return len(episodes)


def get_episode_messages(conn, episode_id):
    """
    Return all messages in an episode, ordered by timestamp.

    Args:
        conn:       Open database connection.
        episode_id: The episode ID to retrieve messages for.

    Returns:
        List of message dicts in chronological order.
    """
    rows = conn.execute(
        "SELECT id, content, sender, recipient, timestamp, category, modality "
        "FROM messages WHERE episode_id = ? ORDER BY timestamp",
        (episode_id,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def expand_to_episodes(conn, results, max_expansion=3):
    """
    For top results, expand to include their full episode context.
    Returns the original results plus surrounding episode messages.

    Args:
        conn:          Open database connection.
        results:       List of message dicts from search.
        max_expansion: Maximum number of episodes to expand (default 3).

    Returns:
        Extended list with episode context messages appended.
    """
    if not results:
        return results

    existing_ids = {r.get("id") for r in results if r.get("id")}
    expanded = list(results)
    expansions = 0

    for r in results[:5]:  # Only expand top-5 results
        if expansions >= max_expansion:
            break

        msg_id = r.get("id")
        if not msg_id:
            continue

        # Get episode_id for this message
        row = conn.execute(
            "SELECT episode_id FROM messages WHERE id = ?", (msg_id,)
        ).fetchone()

        if not row or not row[0]:
            continue

        episode_id = row[0]
        ep_msgs = get_episode_messages(conn, episode_id)

        for em in ep_msgs:
            if em["id"] not in existing_ids:
                em["source"] = "episode_context"
                em["score"] = r.get("score", 0) * 0.6  # lower score for context
                expanded.append(em)
                existing_ids.add(em["id"])

        expansions += 1

    return expanded


# ---------------------------------------------------------------------------
# Landmark event detection (E3)
# ---------------------------------------------------------------------------

def detect_landmark_events(conn):
    """
    Detect significant life/career events from messages and store in landmark_events table.
    Enables "after Demo Day" type queries without date parsing.
    """
    import json

    conn.execute("DELETE FROM landmark_events")

    rows = conn.execute(
        "SELECT id, content, sender, recipient, timestamp FROM messages "
        "WHERE timestamp != '' ORDER BY timestamp"
    ).fetchall()

    # Landmark event patterns
    event_patterns = {
        "job_change": re.compile(
            r'\b(?:quit|left|resigned|fired|hired|started|joined|promoted)\b',
            re.IGNORECASE
        ),
        "move": re.compile(
            r'\b(?:moved to|relocated to|moving to|new apartment|new house|new office)\b',
            re.IGNORECASE
        ),
        "launch": re.compile(
            r'\b(?:launched|shipped|released|deployed|went live|demo day|pitch day)\b',
            re.IGNORECASE
        ),
        "relationship": re.compile(
            r'\b(?:broke up|got engaged|got married|dating|anniversary)\b',
            re.IGNORECASE
        ),
        "health": re.compile(
            r'\b(?:diagnosed|surgery|hospital|marathon|pr time|personal record)\b',
            re.IGNORECASE
        ),
        "financial": re.compile(
            r'\b(?:raised|funding|investment|series [a-c]|seed round|revenue milestone|ipo)\b',
            re.IGNORECASE
        ),
        "milestone": re.compile(
            r'\b(?:graduated|completed|certified|milestone|achievement|award|won)\b',
            re.IGNORECASE
        ),
    }

    count = 0
    for msg_id, content, sender, recipient, timestamp in rows:
        for event_type, pattern in event_patterns.items():
            match = pattern.search(content)
            if match:
                # Extract event name from context
                _matched_text = match.group(0).strip()

                # Get surrounding context for event name
                start = max(0, match.start() - 30)
                end = min(len(content), match.end() + 50)
                event_context = content[start:end].strip()

                # Build related entities list
                related = []
                if sender:
                    related.append(sender)
                if recipient:
                    related.append(recipient)

                # Extract proper nouns from context
                proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
                common = {"The", "This", "That", "What", "Just", "But", "And", "Not"}
                for noun in proper_nouns:
                    if noun not in common and noun.lower() not in [r.lower() for r in related]:
                        related.append(noun)

                conn.execute(
                    "INSERT INTO landmark_events "
                    "(event_name, timestamp, event_type, related_entities, source_message_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (event_context, timestamp, event_type, json.dumps(related[:10]), msg_id)
                )
                count += 1
                break  # One event per message is enough

    conn.commit()
    return count
