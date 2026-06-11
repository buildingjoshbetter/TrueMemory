"""Regression lock: build_summaries must use the #649 SAVEPOINT transaction
hygiene (D1-4 / issue #692) — completing the fix that landed on
detect_contradictions / build_structured_facts but missed build_summaries.

Pre-fix build_summaries did `if conn.in_transaction: conn.commit()` (committing
the CALLER's open transaction) and mutated `conn.isolation_level`. Post-fix it
wraps its writes in `_consolidation_write` (a SAVEPOINT), which nests in the
caller's txn without committing it and never touches isolation_level.

No model loads.
"""


import inspect

from truememory.consolidation import build_summaries
from truememory.storage import create_db, insert_message


def test_build_summaries_uses_savepoint_helper():
    # writes go through the SAVEPOINT helper (the no-commit / no-isolation-level
    # behavior is asserted directly in the two behavioral tests below).
    src = inspect.getsource(build_summaries)
    assert "_consolidation_write(conn" in src, "build_summaries must use the _consolidation_write SAVEPOINT"


def _seed(conn, n=6):
    for i in range(n):
        insert_message(conn, {
            "content": f"alice decided to migrate to clickhouse in month {i}",
            "sender": "alice", "recipient": "bob",
            "timestamp": f"2026-{(i % 12) + 1:02d}-01T10:00:00Z",
            "category": "s", "modality": "conversation",
        })
    conn.commit()


def test_build_summaries_does_not_commit_caller_writes(tmp_path):
    """A caller's UNCOMMITTED write must survive build_summaries and still be
    rollback-able (proving build_summaries didn't commit it)."""
    conn = create_db(tmp_path / "s.db")
    _seed(conn)

    # Begin a caller transaction with an uncommitted insert.
    conn.execute(
        "INSERT INTO messages (content, sender, recipient, timestamp, category, modality) "
        "VALUES ('UNCOMMITTED caller row', 'x', 'y', '2026-01-01T00:00:00Z', 's', 'conversation')"
    )
    assert conn.in_transaction

    build_summaries(conn)  # must NOT commit the caller's open transaction

    # The caller's row is still in the open txn and can be rolled back.
    assert conn.in_transaction, "build_summaries committed/closed the caller's transaction"
    conn.rollback()
    leaked = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE content = 'UNCOMMITTED caller row'"
    ).fetchone()[0]
    assert leaked == 0, "caller's uncommitted row was committed by build_summaries (leaked txn)"
    conn.close()


def test_build_summaries_isolation_level_unchanged(tmp_path):
    conn = create_db(tmp_path / "s2.db")
    _seed(conn)
    before = conn.isolation_level
    build_summaries(conn)
    assert conn.isolation_level == before, "build_summaries leaked an isolation_level change"
    conn.close()
