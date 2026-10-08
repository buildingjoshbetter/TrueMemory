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
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta

from truememory.storage import _row_to_dict, select_message_cols


# ---------------------------------------------------------------------------
# Date-boundary helpers (used by temporal.py and fts_search.py)
# ---------------------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_iso_date(value: str | None) -> str | None:
    """Return *value* unchanged if it looks like ``YYYY-MM-DD``, else ``None``.

    Guards against ``None``, empty strings, and malformed dates before they
    are concatenated into SQL parameters or comparison strings.
    """
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if _ISO_DATE_RE.match(value):
        return value
    # Also accept full ISO timestamps (YYYY-MM-DDTHH:MM:SS…)
    if len(value) >= 10 and _ISO_DATE_RE.match(value[:10]):
        return value
    return None


def _exclusive_upper_bound(date_str: str) -> str:
    """Convert a date-only ``YYYY-MM-DD`` into the *next* day for ``< ?`` comparisons.

    If the input already contains a time component, return it as-is (the
    caller is expected to use ``<`` instead of ``<=``).

    >>> _exclusive_upper_bound("2025-06-15")
    '2025-06-16'
    >>> _exclusive_upper_bound("2025-12-31")
    '2026-01-01'
    """
    if len(date_str) == 10:  # "YYYY-MM-DD"
        dt = datetime.fromisoformat(date_str)
        return (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    # Full timestamp — return as-is; the caller should use strict `<`.
    return date_str


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
    # Use a greedy-enough pattern that captures American date formats with
    # commas (e.g. "January 15, 2026") before terminating at sentence-level
    # punctuation.  The old pattern used `,` as a terminator which truncated
    # "January 15, 2026" to "January 15".
    m = re.search(r"\bafter\s+(.+?)(?:\?|$|\.\s|\band\b)", ql)
    if m and result["after"] is None:
        # Look for a date, including in parentheses in the original query
        after_text = m.group(1).rstrip(", ")
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
    m = re.search(r"\bbefore\s+(.+?)(?:\?|$|\.\s|\band\b)", ql)
    if m and result["before"] is None:
        before_text = m.group(1).rstrip(", ")
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

    # --- Relative temporal expressions ---
    # Resolve "yesterday", "last week", "last month", "last year", and
    # "last N days/weeks/months" to concrete date ranges using the
    # current date.  Previously these were detected but never resolved
    # to actual after/before values (#509).

    _now = datetime.now()

    # "yesterday"
    if re.search(r"\byesterday\b", ql) and not result["after"] and not result["before"]:
        yesterday = _now - timedelta(days=1)
        result["after"] = yesterday.strftime("%Y-%m-%d")
        result["before"] = yesterday.strftime("%Y-%m-%d")
        result["has_temporal"] = True
        result["sort_by_time"] = True

    # "last N days/weeks/months"
    m_rel_n = re.search(r"\blast\s+(\d+)\s+(day|week|month|year)s?\b", ql)
    if m_rel_n and not result["after"] and not result["before"]:
        n = int(m_rel_n.group(1))
        unit = m_rel_n.group(2)
        if unit == "day":
            delta = timedelta(days=n)
        elif unit == "week":
            delta = timedelta(weeks=n)
        elif unit == "month":
            delta = timedelta(days=n * 30)
        else:  # year
            delta = timedelta(days=n * 365)
        result["after"] = (_now - delta).strftime("%Y-%m-%d")
        result["before"] = _now.strftime("%Y-%m-%d")
        result["has_temporal"] = True
        result["sort_by_time"] = True

    # "last month/week/year" (single unit, no number)
    m = re.search(r"\blast\s+(month|week|year)\b", ql)
    if m and not result["after"] and not result["before"]:
        unit = m.group(1)
        if unit == "week":
            result["after"] = (_now - timedelta(weeks=1)).strftime("%Y-%m-%d")
        elif unit == "month":
            result["after"] = (_now - timedelta(days=30)).strftime("%Y-%m-%d")
        else:  # year
            result["after"] = (_now - timedelta(days=365)).strftime("%Y-%m-%d")
        result["before"] = _now.strftime("%Y-%m-%d")
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
    include_directives: bool = False,
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

    after = _validate_iso_date(intent["after"])
    before = _validate_iso_date(intent["before"])

    # --- Filter by time window ---
    if after or before:
        # Compute exclusive upper bound once (next day for date-only strings).
        before_excl = _exclusive_upper_bound(before) if before else None
        filtered = []
        for r in results:
            ts = r.get("timestamp", "")
            if not ts:
                continue
            if after and ts < after:
                continue
            if before_excl and ts >= before_excl:
                continue
            filtered.append(r)
        results = filtered

    # --- If trajectory, sort chronologically ---
    if intent["is_trajectory"] or intent["sort_by_time"]:
        results.sort(key=lambda r: r.get("timestamp", ""))

    # --- If not enough results, do a fresh SQL query with time constraints ---
    if len(results) < limit and (after or before):
        existing_ids = {r.get("id") for r in results}
        extra = get_timeline(conn, after=after, before=before,
                             include_directives=include_directives)
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
    include_directives: bool = False,
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
        before: Exclusive upper bound date (ISO string).  For a date-only
                value the bound is the start of the *next* day so that
                events on ``before`` are included but midnight of the next
                day is not.

    Returns:
        List of message dicts in chronological order.
    """
    clauses: list[str] = []
    params: list[str] = []

    after = _validate_iso_date(after)
    before = _validate_iso_date(before)

    if after is not None:
        clauses.append("timestamp >= ?")
        params.append(after)
    if before is not None:
        # Exclusive upper bound: use '<' with the next day so that events
        # on `before` are included but midnight of `before+1` is not.
        clauses.append("timestamp < ?")
        params.append(_exclusive_upper_bound(before))
    if entity is not None:
        entity_lower = entity.lower()
        clauses.append("(LOWER(sender) = ? OR LOWER(recipient) = ?)")
        params.extend([entity_lower, entity_lower])

    # Defense-in-depth (#637 M-91): exclude directives unless explicitly
    # requested.  The engine masks these via its final filter, but direct
    # callers of the fallback would otherwise receive directive content.
    if not include_directives:
        clauses.append("(directive = 0 OR directive IS NULL)")

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT {select_message_cols(conn)} FROM messages{where} ORDER BY timestamp"

    rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Episode boundaries (B1: 6-hour gap heuristic)
# ---------------------------------------------------------------------------

_TZ_SUFFIX_RE = re.compile(r'([+-]\d{2}:\d{2}|Z)$')


def _parse_naive(ts: str) -> datetime:
    """Parse an ISO timestamp to a naive (tz-unaware) datetime.

    Strips any timezone suffix (Z, +HH:MM, -HH:MM) to guarantee all
    results are naive, avoiding TypeError on mixed-tz comparisons.
    """
    stripped = _TZ_SUFFIX_RE.sub('', ts)
    return datetime.fromisoformat(stripped)


def _group_episode_rows(rows: list[tuple], gap_hours: float) -> list[list[tuple]]:
    """Keep the existing lexical ordering and naive timestamp gap semantics."""
    if not rows:
        return []
    episodes = []
    current_episode_msgs = [rows[0]]
    gap_delta = timedelta(hours=gap_hours)

    for i in range(1, len(rows)):
        ts = rows[i][1]
        prev_ts = rows[i - 1][1]

        try:
            curr_dt = _parse_naive(ts)
            prev_dt = _parse_naive(prev_ts)

            if (curr_dt - prev_dt) > gap_delta:
                # New episode
                episodes.append(current_episode_msgs)
                current_episode_msgs = [rows[i]]
            else:
                current_episode_msgs.append(rows[i])
        except (ValueError, TypeError):
            current_episode_msgs.append(rows[i])

    # Don't forget the last episode
    if current_episode_msgs:
        episodes.append(current_episode_msgs)
    return episodes


@contextmanager
def _episode_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    owned = not conn.in_transaction
    conn.execute("BEGIN" if owned else "SAVEPOINT truememory_episodes")
    completed = False
    try:
        yield
        conn.execute("COMMIT" if owned else "RELEASE truememory_episodes")
        completed = True
    finally:
        if not completed and conn.in_transaction:
            if owned:
                conn.rollback()
            else:
                conn.execute("ROLLBACK TO truememory_episodes")
                conn.execute("RELEASE truememory_episodes")


def _reconcile_episodes(conn: sqlite3.Connection, gap_hours: float) -> int:
    rows = conn.execute(
        "SELECT id, timestamp, episode_id FROM messages WHERE timestamp != '' ORDER BY timestamp"
    ).fetchall()
    assigned = conn.execute(
        "SELECT id, episode_id FROM messages WHERE episode_id IS NOT NULL"
    ).fetchall()
    previous = conn.execute(
        "SELECT id, start_time, end_time, message_count, summary FROM episodes"
    ).fetchall()
    groups = _group_episode_rows(rows, gap_hours)
    members: dict[int, set[int]] = {}
    for mid, episode_id in assigned:
        members.setdefault(episode_id, set()).add(mid)
    matching = {
        frozenset(members[episode[0]]): episode
        for episode in previous
        if episode[0] in members and len(members[episode[0]]) == episode[3]
    }

    kept: set[int] = set()
    eligible: set[int] = set()
    assignments = []
    for ep_msgs in groups:
        member_ids = frozenset(message[0] for message in ep_msgs)
        eligible.update(member_ids)
        start_time = ep_msgs[0][1]
        end_time = ep_msgs[-1][1]
        msg_count = len(ep_msgs)
        old = matching.get(member_ids)
        if old is None:
            cursor = conn.execute(
                "INSERT INTO episodes (start_time, end_time, message_count) VALUES (?, ?, ?)",
                (start_time, end_time, msg_count),
            )
            episode_id = cursor.lastrowid
        else:
            episode_id = old[0]
            # Membership alone cannot prove a summary's source text unchanged.
            # Preserve the old detector's invalidation without rewriting empties.
            if tuple(old[1:4]) != (start_time, end_time, msg_count) or old[4] not in ("", None):
                conn.execute(
                    "UPDATE episodes SET start_time = ?, end_time = ?, message_count = ?, summary = '' WHERE id = ?",
                    (start_time, end_time, msg_count, episode_id),
                )
        kept.add(episode_id)
        assignments.extend((episode_id, message[0], episode_id)
                           for message in ep_msgs if message[2] != episode_id)

    assignments.extend((None, mid, None) for mid, _ in assigned if mid not in eligible)
    conn.executemany(
        "UPDATE messages SET episode_id = ? WHERE id = ? AND episode_id IS NOT ?",
        assignments,
    )
    conn.executemany("DELETE FROM episodes WHERE id = ?",
                     ((episode[0],) for episode in previous if episode[0] not in kept))
    return len(groups)


def detect_episodes(conn: sqlite3.Connection, gap_hours: float = 6) -> int:
    """Reconcile six-hour groups atomically, preserving unchanged episode IDs.

    Exact member sets keep their IDs. Merges, splits and membership changes
    get new IDs; nonempty derived summaries are invalidated. Only changed rows
    are written. A caller transaction remains open and owns the final commit.
    """
    owned = not conn.in_transaction
    for attempt in range(3):
        version = conn.execute("PRAGMA data_version").fetchone()[0] if owned else None
        try:
            with _episode_transaction(conn):
                return _reconcile_episodes(conn, gap_hours)
        except sqlite3.OperationalError as error:
            # WAL readers cannot upgrade a snapshot after another writer
            # commits. Retry only our own transaction, at most twice; never
            # restart a caller's snapshot or commit its unrelated writes.
            code = getattr(error, "sqlite_errorcode", None)
            stale = code == 517  # SQLITE_BUSY_SNAPSHOT
            if code is None and str(error) == "database is locked" and owned:
                # Python 3.10 does not expose extended SQLite error codes.
                stale = conn.execute("PRAGMA data_version").fetchone()[0] != version
            if not owned or not stale or attempt == 2:
                raise
    raise AssertionError("Episode snapshot retry limit was not enforced")


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

    source_sql = (
        "SELECT id, content, sender, recipient, timestamp FROM messages "
        "WHERE timestamp != '' ORDER BY timestamp"
    )
    rows = conn.execute(source_sql).fetchall()

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

    events = []
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

                events.append(
                    (event_context, timestamp, event_type, json.dumps(related[:10]), msg_id)
                )
                break  # One event per message is enough

    conn.execute("SAVEPOINT truememory_landmarks")
    completed = False
    try:
        # Own the writer before validating the source used for computation.
        conn.execute("DELETE FROM landmark_events WHERE 0")
        if conn.execute(source_sql).fetchall() != rows:
            raise sqlite3.OperationalError("Landmark source changed during computation; retry the operation")
        conn.execute("DELETE FROM landmark_events")
        conn.executemany(
            "INSERT INTO landmark_events "
            "(event_name, timestamp, event_type, related_entities, source_message_id) "
            "VALUES (?, ?, ?, ?, ?)",
            events,
        )
        conn.execute("RELEASE truememory_landmarks")
        completed = True
    finally:
        if not completed and conn.in_transaction:
            conn.execute("ROLLBACK TO truememory_landmarks")
            conn.execute("RELEASE truememory_landmarks")
    return len(events)
