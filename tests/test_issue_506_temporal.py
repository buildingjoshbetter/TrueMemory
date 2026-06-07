"""Regression tests for issues #506 (M9), #507 (M10), #509 (M12).

#506 (M9):  fts_search_in_range excludes messages on the boundary day
            because bare-date 'before' bounds ("2025-07-01") compare less
            than timestamps with a time component ("2025-07-01T10:00:00").

#507 (M10): Temporal regex for "after/before" uses comma as a terminator,
            truncating American-format dates like "January 15, 2026" to
            "January 15" (year lost, parse fails).

#509 (M12): Relative temporal queries ("last month", "yesterday",
            "last week") are detected (has_temporal=True) but never
            resolved to actual after/before date ranges.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from truememory.temporal import detect_temporal_intent, parse_date_reference
from truememory.fts_search import search_fts_in_range
from truememory.storage import create_db, insert_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db_with_messages(messages: list[dict]) -> sqlite3.Connection:
    """Create an in-memory DB with the given messages inserted."""
    conn = create_db(":memory:")
    for msg in messages:
        insert_message(conn, {
            "content": msg.get("content", "test"),
            "sender": msg.get("sender", "alice"),
            "recipient": msg.get("recipient", "bob"),
            "timestamp": msg.get("timestamp", ""),
            "category": msg.get("category", ""),
            "modality": msg.get("modality", ""),
        })
    return conn


# ═══════════════════════════════════════════════════════════════════════════
# M9 (#506) — Boundary-day inclusion in fts_search_in_range
# ═══════════════════════════════════════════════════════════════════════════

class TestM9BoundaryInclusion:
    """fts_search_in_range must include messages whose timestamp falls
    on the boundary day, even when the timestamp has a time component."""

    def test_before_boundary_includes_same_day_timestamps(self):
        """A message at '2025-07-01T10:00:00' must be included when
        before='2025-07-01'."""
        conn = _make_db_with_messages([
            {"content": "meeting about budget", "timestamp": "2025-07-01T10:00:00"},
            {"content": "meeting about roadmap", "timestamp": "2025-06-15T09:00:00"},
            {"content": "meeting about hiring", "timestamp": "2025-08-01T14:00:00"},
        ])

        results = search_fts_in_range(conn, "meeting", before="2025-07-01", limit=10)
        timestamps = [r["timestamp"] for r in results]

        assert "2025-07-01T10:00:00" in timestamps, (
            f"Boundary-day message excluded. Got timestamps: {timestamps}"
        )
        assert "2025-06-15T09:00:00" in timestamps
        # The August message should be excluded
        assert "2025-08-01T14:00:00" not in timestamps

    def test_after_boundary_includes_same_day_timestamps(self):
        """A message at '2025-06-01T08:00:00' must be included when
        after='2025-06-01'."""
        conn = _make_db_with_messages([
            {"content": "meeting about budget", "timestamp": "2025-06-01T08:00:00"},
            {"content": "meeting about roadmap", "timestamp": "2025-05-15T09:00:00"},
        ])

        results = search_fts_in_range(conn, "meeting", after="2025-06-01", limit=10)
        timestamps = [r["timestamp"] for r in results]

        assert "2025-06-01T08:00:00" in timestamps
        assert "2025-05-15T09:00:00" not in timestamps

    def test_both_boundaries_inclusive(self):
        """Messages on both the after and before days must be included."""
        conn = _make_db_with_messages([
            {"content": "meeting alpha", "timestamp": "2025-06-01T00:01:00"},
            {"content": "meeting beta", "timestamp": "2025-06-15T12:00:00"},
            {"content": "meeting gamma", "timestamp": "2025-06-30T23:59:00"},
            {"content": "meeting delta", "timestamp": "2025-07-01T00:00:01"},
        ])

        results = search_fts_in_range(
            conn, "meeting",
            after="2025-06-01", before="2025-06-30",
            limit=10,
        )
        timestamps = {r["timestamp"] for r in results}

        assert "2025-06-01T00:01:00" in timestamps, "Start-boundary message excluded"
        assert "2025-06-30T23:59:00" in timestamps, "End-boundary message excluded"
        assert "2025-07-01T00:00:01" not in timestamps, "Day-after message should be excluded"

    def test_before_with_full_datetime_bound(self):
        """When 'before' already contains a time component, no T23:59:59
        suffix should be appended."""
        conn = _make_db_with_messages([
            {"content": "meeting morning", "timestamp": "2025-07-01T08:00:00"},
            {"content": "meeting afternoon", "timestamp": "2025-07-01T15:00:00"},
        ])

        results = search_fts_in_range(
            conn, "meeting", before="2025-07-01T12:00:00", limit=10,
        )
        timestamps = [r["timestamp"] for r in results]

        assert "2025-07-01T08:00:00" in timestamps
        assert "2025-07-01T15:00:00" not in timestamps


# ═══════════════════════════════════════════════════════════════════════════
# M10 (#507) — American date format with comma
# ═══════════════════════════════════════════════════════════════════════════

class TestM10AmericanDateComma:
    """The 'after/before' regex must not use comma as a terminator,
    so "after January 15, 2026" parses the full date correctly."""

    def test_after_american_date_with_comma(self):
        """'after January 15, 2026' must resolve after='2026-01-15'."""
        intent = detect_temporal_intent("what happened after January 15, 2026?")
        assert intent["has_temporal"] is True
        assert intent["after"] == "2026-01-15", (
            f"Expected after='2026-01-15', got after={intent['after']!r}. "
            "Comma in 'January 15, 2026' likely truncated the date."
        )

    def test_before_american_date_with_comma(self):
        """'before March 20, 2025' must resolve before='2025-03-20'."""
        intent = detect_temporal_intent("what was said before March 20, 2025?")
        assert intent["has_temporal"] is True
        assert intent["before"] == "2025-03-20", (
            f"Expected before='2025-03-20', got before={intent['before']!r}."
        )

    def test_after_date_without_comma_still_works(self):
        """Ensure the fix didn't break 'after June 15 2025' (no comma)."""
        intent = detect_temporal_intent("what happened after June 15 2025?")
        assert intent["has_temporal"] is True
        assert intent["after"] == "2025-06-15"

    def test_after_with_comma_and_and_conjunction(self):
        """'after January 15, 2026 and before March 1, 2026' must parse both."""
        intent = detect_temporal_intent(
            "what happened after January 15, 2026 and before March 1, 2026?"
        )
        assert intent["has_temporal"] is True
        assert intent["after"] == "2026-01-15", f"after={intent['after']!r}"
        assert intent["before"] == "2026-03-01", f"before={intent['before']!r}"

    def test_parse_date_reference_handles_comma_format(self):
        """parse_date_reference must handle 'January 15, 2026'."""
        assert parse_date_reference("January 15, 2026") == "2026-01-15"
        assert parse_date_reference("Dec 25, 2025") == "2025-12-25"

    def test_after_abbreviated_month_with_comma(self):
        """'after Dec 25, 2025' must resolve after='2025-12-25'."""
        intent = detect_temporal_intent("what happened after Dec 25, 2025?")
        assert intent["has_temporal"] is True
        assert intent["after"] == "2025-12-25"


# ═══════════════════════════════════════════════════════════════════════════
# M12 (#509) — Relative temporal query resolution
# ═══════════════════════════════════════════════════════════════════════════

class TestM12RelativeTemporalResolution:
    """Relative queries like 'last month', 'yesterday', 'last week'
    must resolve to concrete after/before date ranges."""

    def test_yesterday_resolves_to_date_range(self):
        """'yesterday' must produce after=before=yesterday's date."""
        fake_now = datetime(2026, 6, 7, 14, 0, 0)
        with patch("truememory.temporal.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            intent = detect_temporal_intent("what did I say yesterday?")

        assert intent["has_temporal"] is True
        assert intent["after"] == "2026-06-06", f"after={intent['after']!r}"
        assert intent["before"] == "2026-06-06", f"before={intent['before']!r}"

    def test_last_week_resolves_to_date_range(self):
        """'last week' must produce after ~7 days ago, before=today."""
        fake_now = datetime(2026, 6, 7, 14, 0, 0)
        with patch("truememory.temporal.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            intent = detect_temporal_intent("what happened last week?")

        assert intent["has_temporal"] is True
        assert intent["after"] == "2026-05-31", f"after={intent['after']!r}"
        assert intent["before"] == "2026-06-07", f"before={intent['before']!r}"

    def test_last_month_resolves_to_date_range(self):
        """'last month' must produce after ~30 days ago, before=today."""
        fake_now = datetime(2026, 6, 7, 14, 0, 0)
        with patch("truememory.temporal.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            intent = detect_temporal_intent("messages from last month")

        assert intent["has_temporal"] is True
        assert intent["after"] == "2026-05-08", f"after={intent['after']!r}"
        assert intent["before"] == "2026-06-07", f"before={intent['before']!r}"

    def test_last_year_resolves_to_date_range(self):
        """'last year' must produce after ~365 days ago, before=today."""
        fake_now = datetime(2026, 6, 7, 14, 0, 0)
        with patch("truememory.temporal.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            intent = detect_temporal_intent("what happened last year?")

        assert intent["has_temporal"] is True
        assert intent["after"] == "2025-06-07", f"after={intent['after']!r}"
        assert intent["before"] == "2026-06-07", f"before={intent['before']!r}"

    def test_last_n_days_resolves(self):
        """'last 3 days' must produce after=3 days ago, before=today."""
        fake_now = datetime(2026, 6, 7, 14, 0, 0)
        with patch("truememory.temporal.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            intent = detect_temporal_intent("what happened in the last 3 days?")

        assert intent["has_temporal"] is True
        assert intent["after"] == "2026-06-04", f"after={intent['after']!r}"
        assert intent["before"] == "2026-06-07", f"before={intent['before']!r}"

    def test_last_2_weeks_resolves(self):
        """'last 2 weeks' must produce after=14 days ago, before=today."""
        fake_now = datetime(2026, 6, 7, 14, 0, 0)
        with patch("truememory.temporal.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            intent = detect_temporal_intent("conversations from the last 2 weeks")

        assert intent["has_temporal"] is True
        assert intent["after"] == "2026-05-24", f"after={intent['after']!r}"
        assert intent["before"] == "2026-06-07", f"before={intent['before']!r}"

    def test_relative_query_not_clobbered_by_absolute(self):
        """A query with both absolute and relative dates should prefer
        the absolute date (absolute was set first by earlier patterns)."""
        intent = detect_temporal_intent("what happened after January 2026 last month?")
        # "after January 2026" should set after, "last month" should not overwrite
        assert intent["after"] == "2026-01-01"

    def test_last_month_produces_search_results(self):
        """End-to-end: 'last month' should actually filter messages to the
        correct date range when used with search_temporal."""
        from truememory.temporal import search_temporal

        fake_now = datetime(2026, 6, 7, 14, 0, 0)

        conn = _make_db_with_messages([
            {"content": "recent meeting alpha", "timestamp": "2026-05-20T10:00:00"},
            {"content": "recent meeting beta", "timestamp": "2026-06-01T10:00:00"},
            {"content": "old meeting gamma", "timestamp": "2025-01-01T10:00:00"},
        ])

        fts_results = [
            {"id": 1, "content": "recent meeting alpha", "timestamp": "2026-05-20T10:00:00", "score": 0.9},
            {"id": 2, "content": "recent meeting beta", "timestamp": "2026-06-01T10:00:00", "score": 0.8},
            {"id": 3, "content": "old meeting gamma", "timestamp": "2025-01-01T10:00:00", "score": 0.7},
        ]

        with patch("truememory.temporal.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            results = search_temporal(conn, "meeting last month", fts_results=fts_results, limit=10)

        result_ids = [r["id"] for r in results]
        # Message from 2026-05-20 is within "last month" (after 2026-05-08)
        assert 1 in result_ids, f"Recent message should be included. Got ids: {result_ids}"
        # Message from 2025-01-01 should be excluded
        assert 3 not in result_ids, f"Old message should be excluded. Got ids: {result_ids}"
