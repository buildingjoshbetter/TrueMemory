"""
TrueMemory FTS5 Search Module
============================

Full-text search powered by SQLite FTS5 with BM25 ranking and score
normalization. Provides the keyword-search layer of the TrueMemory hybrid
retrieval pipeline (L3 Semantic layer uses FTS5 + vector + RRF fusion).

Key design decisions:
    - Safe query handling: all queries are sanitized (quoted + OR-joined)
      before use as FTS5 MATCH expressions to prevent syntax injection.
    - Score normalization: BM25 raw scores (negative, more negative = better)
      are flipped and scaled to 0-1 where 1.0 = most relevant result.
    - Time and sender filtering happen in SQL (not post-hoc) where possible,
      with FTS5 providing the candidate set.
"""

import sqlite3

from truememory.storage import (
    _deserialize_metadata,
    directive_filter_sql,
    select_message_cols,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_safe_query(query: str) -> str:
    """
    Convert a natural-language query into a safe FTS5 MATCH expression.

    Splits on whitespace, wraps each token in double quotes to neutralize
    special characters, and joins with OR so any matching term counts.
    """
    tokens = [f'"{w.replace(chr(34), "")}"' for w in query.split() if w.strip()]
    return " OR ".join(tokens) if tokens else '""'


def _build_safe_fts_query(terms: list[str]) -> str:
    """Build a safe OR-joined FTS5 query from a list of terms."""
    quoted = [f'"{t.replace(chr(34), "")}"' for t in terms if t.strip()]
    return " OR ".join(quoted) if quoted else '""'


def _normalize_scores(results: list[dict]) -> None:
    """
    Normalize raw BM25 scores to 0-1 range in place.

    FTS5 ``rank`` values are negative: more negative means more relevant.
    We flip them so that the most relevant result gets score 1.0 and the
    least relevant approaches 0.0.

    When all results share the same raw score (or there is only one result),
    every result receives a score of 1.0.
    """
    if not results:
        return

    min_score = min(r["raw_score"] for r in results)
    max_score = max(r["raw_score"] for r in results)
    score_range = max_score - min_score

    if score_range == 0:
        # All results have identical relevance -- treat them equally.
        for r in results:
            r["score"] = 1.0
    else:
        for r in results:
            # Flip: most negative raw_score -> highest normalized score
            r["score"] = (max_score - r["raw_score"]) / score_range


def _rows_to_results(rows: list[tuple]) -> list[dict]:
    """Convert raw query rows into result dicts with raw_score preserved."""
    return [
        {
            "id": row[0],
            "content": row[1],
            "sender": row[2],
            "recipient": row[3],
            "timestamp": row[4],
            "category": row[5],
            "modality": row[6],
            "directive": bool(row[7]),
            "metadata": _deserialize_metadata(row[8]),
            "raw_score": row[9],
            "score": 0.0,  # placeholder, filled by _normalize_scores
        }
        for row in rows
    ]


def _fts_select(conn: sqlite3.Connection) -> str:
    return (
        f"SELECT {select_message_cols(conn, alias='m')}, "
        "messages_fts.rank AS bm25_score "
        "FROM messages_fts "
        "JOIN messages m ON m.id = messages_fts.rowid"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    include_directives: bool = False,
) -> list[dict]:
    """
    Search messages using FTS5 with BM25 ranking.

    The query is always sanitized before use: each word is quoted and
    joined with OR to prevent FTS5 syntax injection.

    Scores are normalized to the 0-1 range where 1.0 = most relevant.

    Args:
        conn:  Open database connection (must have messages_fts table).
        query: Search query -- plain English or FTS5 syntax.
        limit: Maximum number of results to return.
        include_directives: If False (default), exclude directive=1 rows.

    Returns:
        List of result dicts ordered by relevance (best first).
        Each dict contains: id, content, sender, recipient, timestamp,
        category, modality, raw_score, score.
    """
    if not query or not query.strip():
        return []

    directive_filter = directive_filter_sql(conn, alias="m", include_directives=include_directives)
    sql = f"{_fts_select(conn)} WHERE messages_fts MATCH ?{directive_filter} ORDER BY messages_fts.rank LIMIT ?"
    safe = _build_safe_query(query)
    if not safe:
        return []
    try:
        rows = conn.execute(sql, (safe, limit)).fetchall()
    except sqlite3.OperationalError:
        return []

    results = _rows_to_results(rows)
    _normalize_scores(results)
    return results


def search_fts_by_sender(
    conn: sqlite3.Connection,
    query: str,
    sender: str,
    limit: int = 10,
    include_directives: bool = False,
) -> list[dict]:
    """
    Search within a specific sender's messages only.

    Combines FTS5 MATCH with a sender filter in a single query so SQLite
    can optimize the join.

    Args:
        conn:   Open database connection.
        query:  Search query.
        sender: Sender name to restrict results to (case-sensitive).
        limit:  Maximum number of results.
        include_directives: If False (default), exclude directive=1 rows.

    Returns:
        List of result dicts filtered to *sender*, ordered by relevance.
    """
    if not query or not query.strip():
        return []

    directive_filter = directive_filter_sql(conn, alias="m", include_directives=include_directives)
    sql = (
        f"{_fts_select(conn)}"
        f" WHERE messages_fts MATCH ? AND m.sender = ?{directive_filter}"
        " ORDER BY messages_fts.rank LIMIT ?"
    )

    safe = _build_safe_query(query)
    if not safe:
        return []
    try:
        rows = conn.execute(sql, (safe, sender, limit)).fetchall()
    except sqlite3.OperationalError:
        return []

    results = _rows_to_results(rows)
    _normalize_scores(results)
    return results


def search_fts_in_range(
    conn: sqlite3.Connection,
    query: str,
    after: str | None = None,
    before: str | None = None,
    limit: int = 10,
    include_directives: bool = False,
) -> list[dict]:
    """
    Search with timestamp filtering.

    Retrieves a larger candidate set from FTS5 (up to 10x *limit*), then
    applies timestamp bounds. This approach lets FTS5 do the heavy lifting
    on relevance while SQL handles temporal filtering.

    Timestamps are compared lexicographically (ISO-8601 strings).

    Args:
        conn:   Open database connection.
        query:  Search query.
        after:  Inclusive lower bound timestamp (e.g. ``"2025-06-01"``).
        before: Inclusive upper bound timestamp (e.g. ``"2025-07-01"``).
        limit:  Maximum number of results to return after filtering.
        include_directives: If False (default), exclude directive=1 rows.

    Returns:
        List of result dicts within the time range, ordered by relevance.
    """
    if not query or not query.strip():
        return []

    # Fetch a generous candidate pool so temporal filtering still yields
    # enough results.
    candidate_limit = min(max(limit * 10, 100), 1000)

    directive_filter = directive_filter_sql(conn, alias="m", include_directives=include_directives)
    sql = (
        f"{_fts_select(conn)}"
        f" WHERE messages_fts MATCH ?{directive_filter}"
        " ORDER BY messages_fts.rank LIMIT ?"
    )

    safe = _build_safe_query(query)
    if not safe:
        return []
    try:
        rows = conn.execute(sql, (safe, candidate_limit)).fetchall()
    except sqlite3.OperationalError:
        return []

    results = _rows_to_results(rows)

    # Apply timestamp bounds: inclusive lower, exclusive upper.
    # For date-only upper bounds ("YYYY-MM-DD"), we use the start of the
    # *next* day as the exclusive ceiling so that events on `before` are
    # included but midnight of the next day is not.
    from truememory.temporal import _validate_iso_date, _exclusive_upper_bound

    after = _validate_iso_date(after)
    before = _validate_iso_date(before)

    # M-70: empty-timestamp rows have no comparable date. ``"" < before`` is
    # True and ``"" >= after`` is False, so an upper-bound-only filter would
    # silently let them through. Drop them uniformly whenever any temporal
    # bound is active.
    if after is not None or before is not None:
        results = [r for r in results if r["timestamp"]]
    if after is not None:
        results = [r for r in results if r["timestamp"] >= after]
    if before is not None:
        before_excl = _exclusive_upper_bound(before)
        results = [r for r in results if r["timestamp"] < before_excl]

    # Normalize across the full in-range candidate set BEFORE trimming so a
    # single survivor of the limit slice does not normalize to exactly 1.0
    # and dominate the RRF-scored pool it is merged into (#633 M-10).
    _normalize_scores(results)
    return results[:limit]


def _fts_search(conn: sqlite3.Connection, fts_query: str,
                limit: int = 20,
                include_directives: bool = False) -> list[dict]:
    """Run an FTS5 search and return result dicts."""
    directive_filter = directive_filter_sql(conn, alias="m", include_directives=include_directives)
    sql = (
        f"{_fts_select(conn)} "
        f"WHERE messages_fts MATCH ?{directive_filter} "
        "ORDER BY messages_fts.rank LIMIT ?"
    )
    try:
        rows = conn.execute(sql, (fts_query, limit)).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        {
            "id": r[0], "content": r[1], "sender": r[2],
            "recipient": r[3], "timestamp": r[4],
            "category": r[5], "modality": r[6], "directive": bool(r[7]),
            "metadata": _deserialize_metadata(r[8]), "score": r[9],
        }
        for r in rows
    ]
