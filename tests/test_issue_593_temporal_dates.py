"""Tests for issue #593: temporal date-boundary guards.

Verifies:
    1. None dates handled gracefully (no crash).
    2. Malformed dates handled gracefully (treated as None).
    3. Exclusive upper bound — event at midnight of the next day is excluded.
    4. Normal date range works correctly.
"""
from __future__ import annotations

import sqlite3

import pytest

from truememory.temporal import (
    _exclusive_upper_bound,
    _validate_iso_date,
    detect_temporal_intent,
    get_timeline,
    search_temporal,
)
from truememory.fts_search import search_fts_in_range


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    """Create an in-memory DB with a minimal messages + FTS5 schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE messages ("
        "  id INTEGER PRIMARY KEY,"
        "  content TEXT,"
        "  sender TEXT,"
        "  recipient TEXT,"
        "  timestamp TEXT,"
        "  category TEXT DEFAULT '',"
        "  modality TEXT DEFAULT 'text',"
        "  directive INTEGER DEFAULT 0,"
        "  episode_id INTEGER"
        ")"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5("
        "  content, sender, recipient,"
        "  content=messages, content_rowid=id"
        ")"
    )
    return conn


def _insert(conn: sqlite3.Connection, content: str, ts: str, sender: str = "alice"):
    cur = conn.execute(
        "INSERT INTO messages (content, sender, recipient, timestamp) VALUES (?, ?, '', ?)",
        (content, sender, ts),
    )
    mid = cur.lastrowid
    conn.execute(
        "INSERT INTO messages_fts (rowid, content, sender, recipient) VALUES (?, ?, ?, '')",
        (mid, content, sender),
    )
    conn.commit()
    return mid


# ---------------------------------------------------------------------------
# _validate_iso_date
# ---------------------------------------------------------------------------

class TestValidateIsoDate:
    def test_none_returns_none(self):
        assert _validate_iso_date(None) is None

    def test_empty_returns_none(self):
        assert _validate_iso_date("") is None

    def test_malformed_returns_none(self):
        assert _validate_iso_date("not-a-date") is None
        assert _validate_iso_date("2025/06/15") is None
        assert _validate_iso_date("06-15-2025") is None

    def test_valid_date_only(self):
        assert _validate_iso_date("2025-06-15") == "2025-06-15"

    def test_valid_datetime(self):
        assert _validate_iso_date("2025-06-15T10:30:00") == "2025-06-15T10:30:00"

    def test_whitespace_stripped(self):
        assert _validate_iso_date("  2025-06-15  ") == "2025-06-15"

    def test_non_string_returns_none(self):
        assert _validate_iso_date(12345) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _exclusive_upper_bound
# ---------------------------------------------------------------------------

class TestExclusiveUpperBound:
    def test_normal_date(self):
        assert _exclusive_upper_bound("2025-06-15") == "2025-06-16"

    def test_end_of_month(self):
        assert _exclusive_upper_bound("2025-06-30") == "2025-07-01"

    def test_end_of_year(self):
        assert _exclusive_upper_bound("2025-12-31") == "2026-01-01"

    def test_leap_day(self):
        assert _exclusive_upper_bound("2024-02-29") == "2024-03-01"

    def test_full_timestamp_passthrough(self):
        ts = "2025-06-15T10:30:00"
        assert _exclusive_upper_bound(ts) == ts


# ---------------------------------------------------------------------------
# get_timeline: None / malformed dates don't crash
# ---------------------------------------------------------------------------

class TestGetTimelineGuards:
    def test_none_after_and_before(self):
        conn = _make_db()
        _insert(conn, "hello", "2025-06-15T12:00:00")
        results = get_timeline(conn, after=None, before=None)
        assert len(results) == 1

    def test_malformed_after_ignored(self):
        conn = _make_db()
        _insert(conn, "hello", "2025-06-15T12:00:00")
        # Malformed date should be treated as None (no filter), not crash
        results = get_timeline(conn, after="garbage", before=None)
        assert len(results) == 1

    def test_malformed_before_ignored(self):
        conn = _make_db()
        _insert(conn, "hello", "2025-06-15T12:00:00")
        results = get_timeline(conn, after=None, before="garbage")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Exclusive upper bound: midnight event is NOT double-counted
# ---------------------------------------------------------------------------

class TestExclusiveUpperBoundFiltering:
    """An event at exactly midnight of the next day must be excluded."""

    def _setup_db(self):
        conn = _make_db()
        # Event on March 5
        _insert(conn, "meeting notes from march 5", "2025-03-05T14:00:00")
        # Event at the very end of March 5
        _insert(conn, "late night work march 5", "2025-03-05T23:59:59")
        # Event at midnight of March 6 — should NOT be included in "March 5" queries
        _insert(conn, "midnight event march 6", "2025-03-06T00:00:00")
        # Event clearly on March 6
        _insert(conn, "march 6 morning standup", "2025-03-06T09:00:00")
        return conn

    def test_get_timeline_exclusive_upper(self):
        conn = self._setup_db()
        results = get_timeline(conn, after="2025-03-05", before="2025-03-05")
        contents = [r["content"] for r in results]
        assert "meeting notes from march 5" in contents
        assert "late night work march 5" in contents
        # Midnight of 2025-03-06 should NOT appear
        assert "midnight event march 6" not in contents
        assert "march 6 morning standup" not in contents

    def test_search_temporal_exclusive_upper(self):
        conn = self._setup_db()
        # Build fake results that include all 4 messages
        all_results = get_timeline(conn)
        assert len(all_results) == 4

        # "in March 2025" triggers the month-boundary range detection,
        # which sets after="2025-03-01" and before="2025-03-31".
        # We use a tighter query: "between March 5 2025 and March 5 2025"
        # to test the single-day exclusive upper bound.
        results = search_temporal(
            conn, "what happened between March 5 2025 and March 5 2025",
            hybrid_results=all_results,
            limit=10,
        )
        # Only the two March-5 events should remain
        contents = [r["content"] for r in results]
        # The March-5 events must be present
        assert any("march 5" in c.lower() for c in contents)
        # The midnight March-6 event must NOT be present
        assert "midnight event march 6" not in contents


# ---------------------------------------------------------------------------
# Normal date range works correctly
# ---------------------------------------------------------------------------

class TestNormalDateRange:
    def test_timeline_range(self):
        conn = _make_db()
        _insert(conn, "event A", "2025-01-10T08:00:00")
        _insert(conn, "event B", "2025-01-15T12:00:00")
        _insert(conn, "event C", "2025-01-20T18:00:00")
        _insert(conn, "event D", "2025-02-01T09:00:00")

        results = get_timeline(conn, after="2025-01-10", before="2025-01-20")
        contents = [r["content"] for r in results]
        assert "event A" in contents
        assert "event B" in contents
        assert "event C" in contents
        assert "event D" not in contents

    def test_fts_in_range_none_dates(self):
        """search_fts_in_range with None dates should not crash."""
        conn = _make_db()
        _insert(conn, "test message about coding", "2025-06-15T12:00:00")
        # Should not crash — None dates are simply ignored
        results = search_fts_in_range(conn, "coding", after=None, before=None, limit=10)
        assert len(results) == 1

    def test_fts_in_range_exclusive_upper(self):
        conn = _make_db()
        _insert(conn, "coding session morning", "2025-06-15T08:00:00")
        _insert(conn, "coding session midnight", "2025-06-16T00:00:00")
        results = search_fts_in_range(
            conn, "coding", after="2025-06-15", before="2025-06-15", limit=10
        )
        contents = [r["content"] for r in results]
        assert "coding session morning" in contents
        assert "coding session midnight" not in contents


# ---------------------------------------------------------------------------
# detect_temporal_intent still works with the new guards
# ---------------------------------------------------------------------------

class TestDetectTemporalIntent:
    def test_empty_query(self):
        result = detect_temporal_intent("")
        assert result["has_temporal"] is False

    def test_date_range_query(self):
        result = detect_temporal_intent("what happened from January 2025 to March 2025")
        assert result["has_temporal"] is True
        assert result["after"] == "2025-01-01"
        assert result["before"] == "2025-03-01"

    def test_yesterday(self):
        result = detect_temporal_intent("what happened yesterday")
        assert result["has_temporal"] is True
        assert result["after"] is not None
        assert result["before"] is not None
