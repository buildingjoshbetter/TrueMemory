"""Tests for issue #591: detect_contradictions and build_structured_facts must
not hold the SQLite write lock during expensive regex computation.

Pre-fix: detect_contradictions interleaved regex scanning with INSERTs into
fact_timeline inside one long implicit transaction, blocking concurrent
writers for the entire duration.  build_structured_facts had the same
problem with its regex scanning and INSERT into summaries.

Fix: both functions now follow the three-phase pattern established in #401
for build_summaries: read -> compute (no transaction) -> short atomic write.
"""
from __future__ import annotations

import sqlite3

import pytest

from truememory.storage import create_db
from truememory import consolidation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_contradictions(conn, n_changes=5):
    """Insert messages that trigger contradiction detection patterns."""
    rows = []
    techs = [
        ("PostgreSQL", "TimescaleDB"),
        ("TimescaleDB", "ClickHouse"),
        ("React", "Vue"),
        ("AWS", "GCP"),
        ("Slack", "Discord"),
    ]
    for i, (old, new) in enumerate(techs[:n_changes]):
        rows.append((
            f"We switched from {old} to {new} last week.",
            "alice", "bob",
            f"2026-{(i + 1):02d}-15T10:00:00Z",
            "session", "conversation",
        ))
    # Add some filler messages so there is enough data
    for i in range(20):
        rows.append((
            f"General update #{i}: things are going well with the project.",
            "alice", "bob",
            f"2026-01-{(i % 27) + 1:02d}T09:00:00Z",
            "session", "conversation",
        ))
    conn.executemany(
        "INSERT INTO messages "
        "(content, sender, recipient, timestamp, category, modality) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def _seed_structured_facts(conn):
    """Insert messages that trigger structured fact extraction patterns."""
    rows = [
        ("Alice is our CTO and she runs the engineering team.",
         "bob", "carol", "2026-01-10T10:00:00Z", "session", "conversation"),
        ("We hired Dave as a senior engineer last month.",
         "alice", "bob", "2026-02-05T10:00:00Z", "session", "conversation"),
        ("Our office is in downtown Austin near the river.",
         "alice", "bob", "2026-01-20T10:00:00Z", "session", "conversation"),
        ("The company headquarters is in San Francisco.",
         "carol", "dave", "2026-03-01T10:00:00Z", "session", "conversation"),
    ]
    # Add filler for realism
    for i in range(15):
        rows.append((
            f"Status report #{i}: Sprint velocity is stable.",
            "alice", "bob",
            f"2026-01-{(i % 27) + 1:02d}T08:00:00Z",
            "session", "conversation",
        ))
    conn.executemany(
        "INSERT INTO messages "
        "(content, sender, recipient, timestamp, category, modality) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# detect_contradictions — correctness
# ---------------------------------------------------------------------------

class TestDetectContradictionsCorrectness:

    def test_detects_explicit_change(self, tmp_path):
        """Contradictions from 'switched from X to Y' are detected."""
        conn = create_db(str(tmp_path / "c.db"))
        _seed_contradictions(conn)
        result = consolidation.detect_contradictions(conn)
        assert len(result) > 0
        subjects = [r["subject"] for r in result]
        # Should detect database-related change (PostgreSQL -> TimescaleDB)
        assert any("database" in s for s in subjects), (
            f"Expected a database contradiction, got subjects: {subjects}"
        )
        conn.close()

    def test_fact_timeline_populated(self, tmp_path):
        """fact_timeline table has rows after detect_contradictions."""
        conn = create_db(str(tmp_path / "c.db"))
        _seed_contradictions(conn)
        consolidation.detect_contradictions(conn)
        count = conn.execute(
            "SELECT COUNT(*) FROM fact_timeline"
        ).fetchone()[0]
        assert count > 0
        conn.close()

    def test_supersession_links(self, tmp_path):
        """Superseded facts have superseded_by set to the newer fact's ID."""
        conn = create_db(str(tmp_path / "c.db"))
        _seed_contradictions(conn)
        consolidation.detect_contradictions(conn)
        superseded = conn.execute(
            "SELECT COUNT(*) FROM fact_timeline WHERE superseded_by IS NOT NULL"
        ).fetchone()[0]
        assert superseded > 0, "Expected at least one superseded fact"
        # Verify superseded_by points to a real row
        broken = conn.execute(
            "SELECT COUNT(*) FROM fact_timeline f "
            "WHERE f.superseded_by IS NOT NULL "
            "AND f.superseded_by NOT IN (SELECT id FROM fact_timeline)"
        ).fetchone()[0]
        assert broken == 0, "superseded_by points to non-existent row"
        conn.close()

    def test_idempotent(self, tmp_path):
        """Running detect_contradictions twice produces the same results."""
        conn = create_db(str(tmp_path / "c.db"))
        _seed_contradictions(conn)
        r1 = consolidation.detect_contradictions(conn)
        count1 = conn.execute(
            "SELECT COUNT(*) FROM fact_timeline"
        ).fetchone()[0]
        r2 = consolidation.detect_contradictions(conn)
        count2 = conn.execute(
            "SELECT COUNT(*) FROM fact_timeline"
        ).fetchone()[0]
        assert len(r1) == len(r2)
        assert count1 == count2
        conn.close()


# ---------------------------------------------------------------------------
# detect_contradictions — transaction behavior
# ---------------------------------------------------------------------------

class TestDetectContradictionsTxn:

    def test_no_write_lock_during_compute(self, tmp_path, monkeypatch):
        """A concurrent writer must succeed WHILE _compute_contradictions runs.

        We hook _compute_contradictions so that during its execution a second
        connection performs a quick write.  If the write lock were held this
        would fail with 'database is locked'.
        """
        db = str(tmp_path / "c.db")
        conn = create_db(db)
        conn.execute("PRAGMA journal_mode=WAL")
        _seed_contradictions(conn)

        state = {"attempted": False, "ok": None, "err": None}
        real_compute = consolidation._compute_contradictions

        def hooked(all_msgs):
            result = real_compute(all_msgs)
            if not state["attempted"]:
                state["attempted"] = True
                try:
                    other = sqlite3.connect(db, timeout=0.5)
                    other.execute("PRAGMA busy_timeout=500")
                    other.execute(
                        "INSERT INTO messages "
                        "(content, sender, recipient, timestamp, category, modality) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        ("concurrent probe", "x", "y",
                         "2026-06-01T00:00:00Z", "session", "conversation"),
                    )
                    other.commit()
                    other.close()
                    state["ok"] = True
                except Exception as e:
                    state["ok"] = False
                    state["err"] = repr(e)
            return result

        monkeypatch.setattr(consolidation, "_compute_contradictions", hooked)
        consolidation.detect_contradictions(conn)
        conn.close()

        assert state["attempted"], "compute hook never ran"
        assert state["ok"] is True, (
            f"#591 regression: concurrent write blocked during "
            f"detect_contradictions compute phase: {state['err']}"
        )

    def test_atomic_rollback_on_write_failure(self, tmp_path):
        """If the write phase fails, fact_timeline is not left empty."""
        db = str(tmp_path / "c.db")
        conn = create_db(db)
        _seed_contradictions(conn)
        # First run to populate fact_timeline
        consolidation.detect_contradictions(conn)
        before = conn.execute(
            "SELECT COUNT(*) FROM fact_timeline"
        ).fetchone()[0]
        assert before > 0

        # Use a proxy that fails on INSERT INTO fact_timeline
        class _FailInsert:
            """Proxy: forwards everything but fails on fact_timeline INSERT."""
            def __init__(self, real):
                object.__setattr__(self, "_real", real)
                object.__setattr__(self, "_count", 0)

            def __getattr__(self, name):
                return getattr(self._real, name)

            def __setattr__(self, name, value):
                setattr(self._real, name, value)

            def execute(self, sql, *args, **kwargs):
                if "INSERT INTO fact_timeline" in str(sql):
                    n = object.__getattribute__(self, "_count")
                    object.__setattr__(self, "_count", n + 1)
                    if n == 0:
                        raise sqlite3.OperationalError(
                            "simulated write failure"
                        )
                return self._real.execute(sql, *args, **kwargs)

        proxy = _FailInsert(conn)
        with pytest.raises(sqlite3.OperationalError):
            consolidation.detect_contradictions(proxy)

        after = conn.execute(
            "SELECT COUNT(*) FROM fact_timeline"
        ).fetchone()[0]
        assert after == before, (
            "rollback should preserve prior fact_timeline when write fails"
        )
        conn.close()


# ---------------------------------------------------------------------------
# build_structured_facts — correctness
# ---------------------------------------------------------------------------

class TestBuildStructuredFactsCorrectness:

    def test_extracts_team_and_locations(self, tmp_path):
        """Structured facts include team roster and location records."""
        conn = create_db(str(tmp_path / "c.db"))
        _seed_structured_facts(conn)
        n = consolidation.build_structured_facts(conn)
        assert n > 0
        rows = conn.execute(
            "SELECT summary FROM summaries WHERE period = 'structured_fact'"
        ).fetchall()
        texts = [r[0] for r in rows]
        has_roster = any("Team Roster" in t for t in texts)
        has_location = any("Known Locations" in t or "Location" in t for t in texts)
        assert has_roster or has_location, (
            f"Expected team roster or location facts, got: {texts}"
        )
        conn.close()

    def test_idempotent(self, tmp_path):
        """Running build_structured_facts twice produces the same count."""
        conn = create_db(str(tmp_path / "c.db"))
        _seed_structured_facts(conn)
        n1 = consolidation.build_structured_facts(conn)
        n2 = consolidation.build_structured_facts(conn)
        assert n1 == n2
        # Should not duplicate rows
        total = conn.execute(
            "SELECT COUNT(*) FROM summaries WHERE period = 'structured_fact'"
        ).fetchone()[0]
        assert total == n2
        conn.close()

    def test_does_not_clobber_other_summaries(self, tmp_path):
        """build_structured_facts only deletes structured_fact rows."""
        conn = create_db(str(tmp_path / "c.db"))
        _seed_structured_facts(conn)
        # Build regular summaries first
        consolidation.build_summaries(conn)
        monthly_before = conn.execute(
            "SELECT COUNT(*) FROM summaries WHERE period = 'monthly'"
        ).fetchone()[0]
        # Now build structured facts
        consolidation.build_structured_facts(conn)
        monthly_after = conn.execute(
            "SELECT COUNT(*) FROM summaries WHERE period = 'monthly'"
        ).fetchone()[0]
        assert monthly_after == monthly_before, (
            "build_structured_facts should not delete monthly summaries"
        )
        conn.close()


# ---------------------------------------------------------------------------
# build_structured_facts — transaction behavior
# ---------------------------------------------------------------------------

class TestBuildStructuredFactsTxn:

    def test_no_write_lock_during_compute(self, tmp_path, monkeypatch):
        """A concurrent writer must succeed WHILE _compute_structured_facts runs."""
        db = str(tmp_path / "c.db")
        conn = create_db(db)
        conn.execute("PRAGMA journal_mode=WAL")
        _seed_structured_facts(conn)

        state = {"attempted": False, "ok": None, "err": None}
        real_compute = consolidation._compute_structured_facts

        def hooked(all_msgs):
            result = real_compute(all_msgs)
            if not state["attempted"]:
                state["attempted"] = True
                try:
                    other = sqlite3.connect(db, timeout=0.5)
                    other.execute("PRAGMA busy_timeout=500")
                    other.execute(
                        "INSERT INTO messages "
                        "(content, sender, recipient, timestamp, category, modality) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        ("concurrent probe", "x", "y",
                         "2026-06-01T00:00:00Z", "session", "conversation"),
                    )
                    other.commit()
                    other.close()
                    state["ok"] = True
                except Exception as e:
                    state["ok"] = False
                    state["err"] = repr(e)
            return result

        monkeypatch.setattr(consolidation, "_compute_structured_facts", hooked)
        consolidation.build_structured_facts(conn)
        conn.close()

        assert state["attempted"], "compute hook never ran"
        assert state["ok"] is True, (
            f"#591 regression: concurrent write blocked during "
            f"build_structured_facts compute phase: {state['err']}"
        )

    def test_atomic_rollback_on_write_failure(self, tmp_path):
        """If the write phase fails, prior structured_fact rows survive."""
        db = str(tmp_path / "c.db")
        conn = create_db(db)
        _seed_structured_facts(conn)
        consolidation.build_structured_facts(conn)
        before = conn.execute(
            "SELECT COUNT(*) FROM summaries WHERE period = 'structured_fact'"
        ).fetchone()[0]
        assert before > 0

        class _FailExecMany:
            """Proxy that makes executemany raise."""
            def __init__(self, real):
                object.__setattr__(self, "_real", real)

            def __getattr__(self, name):
                return getattr(self._real, name)

            def __setattr__(self, name, value):
                setattr(self._real, name, value)

            def executemany(self, *a, **k):
                raise sqlite3.OperationalError("simulated write failure")

        proxy = _FailExecMany(conn)
        with pytest.raises(sqlite3.OperationalError):
            consolidation.build_structured_facts(proxy)

        after = conn.execute(
            "SELECT COUNT(*) FROM summaries WHERE period = 'structured_fact'"
        ).fetchone()[0]
        assert after == before, (
            "rollback should preserve prior structured_fact rows when write fails"
        )
        conn.close()
