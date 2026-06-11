"""Regression locks for issue #650 — DB corruption UX hardening.

Covers:
- M-24: pre-migration backups are rotated (keep <= N); a migration-failing DB
  does NOT accumulate unbounded backups across repeated opens.
- M-55: opening a corrupt DB yields an actionable DatabaseOpenError naming a
  backup, not a raw DatabaseError swallowed silently.
- M-57: a read-only DB dir yields a message naming the directory.
- M-84: the L4 entity_profile purge runs via _ensure_connection (not only the
  deprecated open()).

All tests use tmp dirs/copies. No model loads (HF_HUB_OFFLINE handled by the
suite); these tests never embed.
"""
from __future__ import annotations

import os
import sqlite3
import sys

import pytest

from truememory import storage
from truememory.storage import (
    DatabaseOpenError,
    _MAX_PRE_MIGRATION_BACKUPS,
    create_db,
    newest_backup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_legacy_db(path) -> None:
    """Create a pre-migration ``messages`` table missing the new columns."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "content TEXT NOT NULL, sender TEXT DEFAULT '')"
    )
    conn.execute("INSERT INTO messages (content) VALUES ('hello')")
    conn.commit()
    conn.close()


def _count_backups(db_path) -> int:
    parent = db_path.parent
    prefix = f"{db_path.name}.backup-pre-migration-"
    return len(
        [
            p
            for p in parent.glob(f"{prefix}*")
            if not p.name.endswith("-wal") and not p.name.endswith("-shm")
        ]
    )


# ---------------------------------------------------------------------------
# M-24: backup rotation
# ---------------------------------------------------------------------------

def test_backup_rotation_keeps_at_most_n(tmp_path):
    """Repeated _backup_database calls keep only the newest N backups."""
    db = tmp_path / "memories.db"
    _make_legacy_db(db)

    # Force more backups than the cap. Each call writes a uniquely-named copy.
    for _ in range(_MAX_PRE_MIGRATION_BACKUPS + 4):
        storage._backup_database(db)

    assert _count_backups(db) <= _MAX_PRE_MIGRATION_BACKUPS


def test_failing_migration_does_not_accumulate_backups(tmp_path, monkeypatch):
    """A DB that fails migration must not re-back-up on every open (M-24).

    We force the ALTER to fail, which writes the 'migration-failed' marker.
    Subsequent create_db opens must skip the backup entirely (the marker is
    honored), so the backup count stays bounded across many opens.
    """
    db = tmp_path / "memories.db"
    # Build a DB that already has the full real schema, then inject one extra
    # "expected" column whose typedef is invalid SQL. The migration sees only
    # the bogus column as missing, the ALTER fails, the transaction rolls back
    # (leaving the real schema intact and usable), and the 'migration-failed'
    # marker is written.
    create_db(db).close()
    bad_columns = dict(storage._EXPECTED_COLUMNS)
    bad_columns["bogus"] = "NOT A VALID TYPE ("
    monkeypatch.setattr(storage, "_EXPECTED_COLUMNS", bad_columns)

    # First open: backup taken, ALTER fails, marker written.
    conn = create_db(db)
    conn.close()
    marker = db.parent / f"{db.name}.migration-failed"
    assert marker.exists(), "expected migration-failed marker after ALTER failure"
    count_after_first = _count_backups(db)

    # Many more opens: must NOT re-back-up (marker present).
    for _ in range(6):
        create_db(db).close()

    assert _count_backups(db) == count_after_first, (
        "failing-migration DB re-backed-up on subsequent opens (disk-fill bug)"
    )
    assert _count_backups(db) <= _MAX_PRE_MIGRATION_BACKUPS


# ---------------------------------------------------------------------------
# M-55: corrupt DB -> actionable error naming a backup
# ---------------------------------------------------------------------------

def test_corrupt_db_raises_actionable_error_naming_backup(tmp_path):
    """A corrupted DB yields DatabaseOpenError pointing at the newest backup."""
    db = tmp_path / "memories.db"
    # Build a healthy DB and a backup of it, then corrupt the live file.
    create_db(db).close()
    backup = storage._backup_database(db)
    assert backup is not None

    # Flip bytes in the middle of the page region to corrupt the image while
    # keeping the SQLite magic header intact (so it parses as a DB, then fails
    # integrity).
    data = bytearray(db.read_bytes())
    for i in range(100, min(len(data), 4000)):
        data[i] ^= 0xFF
    db.write_bytes(bytes(data))

    with pytest.raises(DatabaseOpenError) as exc_info:
        create_db(db).close()

    msg = str(exc_info.value)
    assert "corrupt" in msg.lower()
    # Must name a restore candidate.
    assert backup.name in msg
    # And it is a DatabaseError subclass so legacy callers still catch it.
    assert isinstance(exc_info.value, sqlite3.DatabaseError)


def test_newest_backup_returns_latest(tmp_path):
    db = tmp_path / "memories.db"
    create_db(db).close()
    assert newest_backup(db) is None
    b1 = storage._backup_database(db)
    assert b1 is not None
    assert newest_backup(db) == b1


# ---------------------------------------------------------------------------
# M-57: read-only directory -> message naming the directory
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform == "win32", reason="chmod semantics differ on Windows"
)
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses dir permissions",
)
def test_readonly_dir_raises_error_naming_directory(tmp_path):
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    db = ro_dir / "memories.db"
    os.chmod(ro_dir, 0o555)
    try:
        with pytest.raises(DatabaseOpenError) as exc_info:
            create_db(db).close()
        msg = str(exc_info.value)
        assert str(ro_dir) in msg
        assert "writable" in msg.lower()
    finally:
        os.chmod(ro_dir, 0o755)


# ---------------------------------------------------------------------------
# M-84: L4 purge runs via _ensure_connection
# ---------------------------------------------------------------------------

def test_l4_purge_runs_via_ensure_connection(tmp_path, monkeypatch):
    """entity_profile summary rows are purged on the production open path."""
    monkeypatch.delenv("TRUEMEMORY_ENTITY_SHEETS", raising=False)
    db = tmp_path / "memories.db"

    # Seed a DB with a legacy entity_profile summary row.
    conn = create_db(db)
    conn.execute(
        "INSERT INTO summaries (period, summary) VALUES ('entity_profile', 'x')"
    )
    conn.execute(
        "INSERT INTO summaries (period, summary) VALUES ('daily', 'keep me')"
    )
    conn.commit()
    conn.close()

    from truememory.engine import TrueMemoryEngine

    eng = TrueMemoryEngine(db)
    eng._ensure_connection()
    try:
        rows = eng.conn.execute(
            "SELECT period FROM summaries ORDER BY period"
        ).fetchall()
        periods = [r[0] for r in rows]
        assert "entity_profile" not in periods, "L4 purge did not run via _ensure_connection"
        assert "daily" in periods, "purge removed non-entity_profile rows"
        # Migration flag recorded for idempotency.
        flag = eng.conn.execute(
            "SELECT value FROM metadata WHERE key = 'l4_entity_profile_migration_done'"
        ).fetchone()
        assert flag is not None and flag[0] == "1"
    finally:
        if eng.conn is not None:
            eng.conn.close()
