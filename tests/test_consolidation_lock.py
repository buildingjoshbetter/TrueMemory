"""Regression locks for #401 — build_summaries must not hold the SQLite write
lock during its (30-60s) salience/sentence computation.

Pre-fix: build_summaries ran `DELETE FROM summaries` first, then did all the
per-month/per-entity scoring inside that open write transaction, committing
only at the end. The exclusive write lock was held for the whole computation,
so any concurrent writer failed with "database is locked" once the 10s
busy_timeout elapsed.

Fix: compute all rows first (reads only, no write transaction), then do
DELETE + executemany(INSERT) + commit in one short transaction (rollback on
error). The write lock is held only for the fast bulk write.
"""
from __future__ import annotations

import sqlite3
import sys

import pytest

from truememory.storage import create_db
from truememory import consolidation


def _seed(conn, per_month=8):
    rows = []
    for month in ("2026-01", "2026-02", "2026-03"):
        for i in range(per_month):
            rows.append((
                f"Project update {month} #{i}: revenue grew 12 percent and we shipped 3 features.",
                "alice", "bob", f"{month}-{(i % 27) + 1:02d}T10:00:00Z",
                "session", "conversation",
            ))
    conn.executemany(
        "INSERT INTO messages (content, sender, recipient, timestamp, category, modality) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def test_build_summaries_correctness(tmp_path):
    """Behavior preserved: summaries are built and the table row count matches."""
    conn = create_db(str(tmp_path / "c.db"))
    _seed(conn)
    n = consolidation.build_summaries(conn)
    assert n > 0
    total = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
    assert total == n
    monthly = conn.execute(
        "SELECT COUNT(*) FROM summaries WHERE period='monthly'"
    ).fetchone()[0]
    assert monthly >= 1
    conn.close()


def test_build_summaries_releases_write_lock_during_compute(tmp_path, monkeypatch):
    """A concurrent writer must succeed WHILE build_summaries is computing.

    This is the #401 lock. We hook `_message_salience` (called during the
    monthly scoring loop, before the final write) so that on its first call a
    second connection performs a quick write with a short busy_timeout. Pre-fix
    the DELETE-first transaction held the write lock and this write would block
    past the timeout; post-fix there is no open write txn during compute.
    """
    db = str(tmp_path / "c.db")
    conn = create_db(db)
    conn.execute("PRAGMA journal_mode=WAL")
    _seed(conn)

    state = {"attempted": False, "ok": None, "err": None}
    real_salience = consolidation._message_salience

    def hooked(msg):
        if not state["attempted"]:
            state["attempted"] = True
            try:
                other = sqlite3.connect(db, timeout=0.5)
                other.execute("PRAGMA busy_timeout=500")
                other.execute(
                    "INSERT INTO messages (content, sender, recipient, timestamp, category, modality) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ("concurrent write probe", "carol", "dave",
                     "2026-04-01T10:00:00Z", "session", "conversation"),
                )
                other.commit()
                other.close()
                state["ok"] = True
            except Exception as e:  # pragma: no cover - failure path is the assertion
                state["ok"] = False
                state["err"] = repr(e)
        return real_salience(msg)

    monkeypatch.setattr(consolidation, "_message_salience", hooked)
    consolidation.build_summaries(conn)
    conn.close()

    assert state["attempted"], "compute hook never ran — test setup invalid"
    assert state["ok"] is True, (
        f"#401 regression: concurrent write blocked while build_summaries was "
        f"computing (write lock held during compute): {state['err']}"
    )


def test_build_summaries_atomic_rollback_on_write_failure(tmp_path):
    """If the bulk insert fails, the prior summaries must survive (the DELETE
    is rolled back, never leaving the table emptied)."""
    db = str(tmp_path / "c.db")
    conn = create_db(db)
    _seed(conn)
    n_before = consolidation.build_summaries(conn)
    assert n_before > 0

    class _FailExecMany:
        """Forwards everything to the real connection but makes executemany raise."""
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
        consolidation.build_summaries(proxy)

    after = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
    assert after == n_before, (
        "rollback should preserve the prior summaries when the bulk insert fails"
    )
    conn.close()
