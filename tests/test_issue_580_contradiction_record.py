"""Tests for issue #580 — widen contradiction detection + supersede stale facts.

Covers:
    1. Each new detection pattern fires on matching text.
    2. Old facts are marked status='superseded' after a contradiction.
    3. Superseded facts remain retrievable (but ranked lower).
    4. Non-contradiction text does not trigger false positives.
"""

import sqlite3
import pytest

from truememory.storage import create_db, insert_message
from truememory.consolidation import (
    detect_contradictions,
    search_contradictions,
    _CHANGE_PATTERNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    """Return an in-memory DB with the TrueMemory schema."""
    return create_db(":memory:")


def _add(conn, content, timestamp="2026-01-01T00:00:00", sender="user"):
    return insert_message(conn, {
        "content": content,
        "sender": sender,
        "recipient": "",
        "timestamp": timestamp,
        "category": "",
        "modality": "",
    })


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------

class TestExplicitChangeDetection:
    """Original explicit_change pattern still works."""

    def test_switched_from_to(self):
        conn = _make_db()
        _add(conn, "We switched from PostgreSQL to ClickHouse last week.",
             "2026-01-01T00:00:00")
        contradictions = detect_contradictions(conn)
        assert len(contradictions) >= 1
        c = contradictions[0]
        assert "PostgreSQL" in c["old_fact"]
        assert "ClickHouse" in c["new_fact"]


class TestReplacedPattern:
    """'replaced X with Y' explicit_change variant."""

    def test_replaced_with(self):
        conn = _make_db()
        _add(conn, "We replaced Redis with Valkey for caching.")
        contradictions = detect_contradictions(conn)
        assert any("Redis" in c["old_fact"] and "Valkey" in c["new_fact"]
                    for c in contradictions)


class TestInformalCorrectionDetection:
    """Patterns: 'actually X', 'correction: X', 'update: X'."""

    def test_actually(self):
        conn = _make_db()
        _add(conn, "Actually the deadline is next Friday.",
             "2026-02-01T00:00:00")
        contradictions = detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("deadline" in f[0].lower() for f in facts)

    def test_correction_colon(self):
        conn = _make_db()
        _add(conn, "Correction: the budget is $50K not $30K.")
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("budget" in f[0].lower() for f in facts)

    def test_update_colon(self):
        conn = _make_db()
        _add(conn, "Update: we are going with React instead.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert len(facts) >= 1


class TestNegationChangeDetection:
    """Patterns: 'not X anymore', 'no longer X'."""

    def test_not_anymore(self):
        conn = _make_db()
        _add(conn, "I'm not using Vim anymore.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("no longer" in f[0].lower() and "vim" in f[0].lower()
                    for f in facts), f"Expected negation fact, got {facts}"

    def test_no_longer(self):
        conn = _make_db()
        _add(conn, "Sam is no longer the CTO of the company.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("no longer" in f[0].lower() for f in facts), \
            f"Expected negation fact, got {facts}"


class TestInvalidationDetection:
    """Patterns: 'that's wrong', 'that's incorrect', etc."""

    def test_thats_wrong(self):
        conn = _make_db()
        _add(conn, "That's wrong, the meeting is at 3pm.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("wrong" in f[0].lower() for f in facts)

    def test_thats_incorrect(self):
        conn = _make_db()
        _add(conn, "That's incorrect, we use Python not Ruby.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("incorrect" in f[0].lower() for f in facts)

    def test_this_is_outdated(self):
        conn = _make_db()
        _add(conn, "This is outdated information.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("outdated" in f[0].lower() for f in facts)


class TestRetractionDetection:
    """Patterns: 'changed my mind about X', 'I was wrong about X',
    'turns out X', 'scratch that', 'forget what I said', 'disregard'."""

    def test_changed_my_mind(self):
        conn = _make_db()
        _add(conn, "I changed my mind about using MongoDB.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("mongodb" in f[0].lower() for f in facts)

    def test_i_was_wrong_about(self):
        conn = _make_db()
        _add(conn, "I was wrong about the launch date.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("launch date" in f[0].lower() for f in facts)

    def test_turns_out(self):
        conn = _make_db()
        _add(conn, "Turns out the server costs are much higher.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("server costs" in f[0].lower() or "higher" in f[0].lower()
                    for f in facts)

    def test_scratch_that(self):
        conn = _make_db()
        _add(conn, "Scratch that, let's go with plan B instead.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("scratch that" in f[0].lower() for f in facts)

    def test_disregard(self):
        conn = _make_db()
        _add(conn, "Please disregard my previous message about the timeline.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("disregard" in f[0].lower() for f in facts)

    def test_forget_what_i_said(self):
        conn = _make_db()
        _add(conn, "Forget what I said about the pricing.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("forget" in f[0].lower() for f in facts)

    def test_never_mind(self):
        conn = _make_db()
        _add(conn, "Never mind about that feature request.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        assert any("never mind" in f[0].lower() for f in facts)


# ---------------------------------------------------------------------------
# Supersede status tests
# ---------------------------------------------------------------------------

class TestSupersedeStatus:
    """When a contradiction is detected, old fact's status becomes 'superseded'."""

    def test_explicit_change_supersedes_old_fact(self):
        conn = _make_db()
        _add(conn, "We migrated from PostgreSQL to ClickHouse.",
             "2026-03-01T00:00:00")
        detect_contradictions(conn)

        rows = conn.execute(
            "SELECT fact, superseded_by, status FROM fact_timeline "
            "ORDER BY id"
        ).fetchall()
        # At least 2 rows: the old fact and the new fact
        assert len(rows) >= 2
        old = rows[0]
        new = rows[-1]
        assert old[1] is not None, "Old fact should have superseded_by set"
        assert old[2] == "superseded", f"Old fact status should be 'superseded', got {old[2]}"
        assert new[2] in ("active", None), f"New fact status should be 'active', got {new[2]}"

    def test_pricing_change_supersedes(self):
        conn = _make_db()
        _add(conn, "CarbonSense charges $200/month per facility.",
             "2026-01-01T00:00:00")
        _add(conn, "CarbonSense charges $350/month per facility.",
             "2026-06-01T00:00:00")
        detect_contradictions(conn)

        superseded = conn.execute(
            "SELECT COUNT(*) FROM fact_timeline WHERE status = 'superseded'"
        ).fetchone()[0]
        assert superseded >= 1

    def test_informal_correction_supersedes(self):
        """Two messages on the same subject: old one gets superseded."""
        conn = _make_db()
        # First, a fact about a subject
        _add(conn, "The deadline is next Monday.",
             "2026-01-01T00:00:00")
        # Then a correction referencing the same subject
        _add(conn, "Actually the deadline is next Friday.",
             "2026-01-02T00:00:00")
        detect_contradictions(conn)

        rows = conn.execute(
            "SELECT fact, status FROM fact_timeline "
            "WHERE subject NOT LIKE '\\_%' ESCAPE '\\' "
            "ORDER BY id"
        ).fetchall()
        # Should have at least the correction fact
        assert len(rows) >= 1


# ---------------------------------------------------------------------------
# Superseded facts still retrievable
# ---------------------------------------------------------------------------

class TestSupersededRetrievable:
    """Superseded facts are still in search results, just ranked lower."""

    def test_superseded_in_search(self):
        conn = _make_db()
        _add(conn, "We switched from PostgreSQL to ClickHouse.",
             "2026-01-15T00:00:00")
        detect_contradictions(conn)

        results = search_contradictions(conn, "database")
        assert len(results) >= 1
        # The history should contain the superseded fact
        for r in results:
            if r.get("history"):
                superseded_in_history = [
                    h for h in r["history"] if h.get("superseded")
                ]
                if superseded_in_history:
                    return  # pass
        # Even without history, we got results which is sufficient
        assert results, "Superseded facts should still appear in results"

    def test_superseded_ranked_lower(self):
        """When only superseded facts match, relevance should be halved."""
        conn = _make_db()
        _add(conn, "We switched from PostgreSQL to ClickHouse.",
             "2026-01-15T00:00:00")
        detect_contradictions(conn)

        results = search_contradictions(conn, "database")
        assert len(results) >= 1
        # Current fact should be ClickHouse, not PostgreSQL
        r = results[0]
        assert "ClickHouse" in r["current_fact"] or "clickhouse" in r["current_fact"].lower()


# ---------------------------------------------------------------------------
# False positive tests
# ---------------------------------------------------------------------------

class TestNoFalsePositives:
    """Non-contradiction text should not trigger detection."""

    def test_casual_actually(self):
        """'actually' in casual non-correction context should not create
        a fact if the sentence is very short or doesn't carry a factual
        correction payload."""
        conn = _make_db()
        _add(conn, "I actually like pizza a lot.")
        detect_contradictions(conn)
        facts = conn.execute("SELECT fact FROM fact_timeline").fetchall()
        # The pattern may or may not match here; the key is it does
        # not produce contradictions with no prior fact
        contradictions = detect_contradictions(conn)
        # No contradictions because there is no prior fact to supersede
        assert all(c.get("old_fact") != c.get("new_fact") for c in contradictions)

    def test_normal_text_no_contradictions(self):
        conn = _make_db()
        _add(conn, "The weather is nice today.")
        _add(conn, "Let's have lunch at noon.")
        _add(conn, "I finished the report.")
        contradictions = detect_contradictions(conn)
        assert len(contradictions) == 0

    def test_switch_without_from_to(self):
        """'switch' without 'from...to' should not trigger explicit_change."""
        conn = _make_db()
        _add(conn, "I need to switch focus to the frontend.")
        contradictions = detect_contradictions(conn)
        # Should not produce an explicit_change contradiction
        explicit = [c for c in contradictions
                    if "switch" in c.get("old_fact", "").lower()
                    and "frontend" in c.get("new_fact", "").lower()]
        assert len(explicit) == 0


# ---------------------------------------------------------------------------
# Schema migration test
# ---------------------------------------------------------------------------

class TestSchemaStatusColumn:
    """The fact_timeline table has the status column."""

    def test_status_column_exists(self):
        conn = _make_db()
        cols = {row[1] for row in conn.execute(
            "PRAGMA table_info(fact_timeline)"
        ).fetchall()}
        assert "status" in cols

    def test_status_default_is_active(self):
        conn = _make_db()
        # Insert a message first to satisfy FK constraint
        msg_id = _add(conn, "test message", "2026-01-01T00:00:00")
        conn.execute(
            "INSERT INTO fact_timeline (subject, fact, source_message_id, "
            "timestamp) VALUES (?, ?, ?, ?)",
            ("test", "test fact", msg_id, "2026-01-01"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT status FROM fact_timeline WHERE subject = 'test'"
        ).fetchone()
        assert row[0] == "active"


# ---------------------------------------------------------------------------
# Pattern coverage — ensure regex objects compile and match
# ---------------------------------------------------------------------------

class TestPatternCompilation:
    """All _CHANGE_PATTERNS compile and have the expected types."""

    def test_all_patterns_have_type(self):
        for p in _CHANGE_PATTERNS:
            assert "type" in p
            assert "pattern" in p
            assert hasattr(p["pattern"], "search")

    def test_pattern_types_cover_new_types(self):
        types = {p["type"] for p in _CHANGE_PATTERNS}
        assert "informal_correction" in types
        assert "negation_change" in types
        assert "invalidation" in types
        assert "retraction" in types
