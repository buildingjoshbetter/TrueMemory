"""Issue #491: PRAGMA foreign_keys is never enabled.

The schema defines FK constraints (fact_timeline, landmark_events, causal_edges)
but SQLite requires ``PRAGMA foreign_keys = ON`` per connection for enforcement.
Without it, FK constraints are decorative.

This test verifies that every connection-creation path enables foreign keys.
"""

import sqlite3

import pytest


class TestIssue491ForeignKeys:
    """Verify PRAGMA foreign_keys = ON on all connection paths."""

    def test_create_db_enables_foreign_keys(self, tmp_path):
        """create_db() must return a connection with foreign_keys = 1."""
        from truememory.storage import create_db

        conn = create_db(tmp_path / "fk_test.db")
        result = conn.execute("PRAGMA foreign_keys").fetchone()
        assert result is not None
        assert result[0] == 1, (
            f"PRAGMA foreign_keys should be 1 (ON) after create_db(), got {result[0]}"
        )
        conn.close()

    def test_create_db_memory_enables_foreign_keys(self):
        """create_db(':memory:') must also enable foreign keys."""
        from truememory.storage import create_db

        conn = create_db(":memory:")
        result = conn.execute("PRAGMA foreign_keys").fetchone()
        assert result is not None
        assert result[0] == 1
        conn.close()

    def test_foreign_keys_enforced_on_insert(self, tmp_path):
        """Inserting a fact_timeline row with a bad source_message_id must fail."""
        from truememory.storage import create_db

        conn = create_db(tmp_path / "fk_enforce.db")

        # Confirm foreign_keys is on
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

        # Insert into fact_timeline with a source_message_id that doesn't
        # exist in messages — this must raise IntegrityError.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO fact_timeline (subject, fact, source_message_id, timestamp) "
                "VALUES (?, ?, ?, ?)",
                ("test_subject", "test_fact", 99999, "2026-01-01"),
            )

        conn.close()

    def test_engine_ensure_connection_enables_foreign_keys(self, tmp_path):
        """TrueMemoryEngine._ensure_connection() must enable foreign keys."""
        from truememory.engine import TrueMemoryEngine

        engine = TrueMemoryEngine(db_path=tmp_path / "engine_fk.db")
        engine._ensure_connection()
        result = engine.conn.execute("PRAGMA foreign_keys").fetchone()
        assert result is not None
        assert result[0] == 1, (
            f"PRAGMA foreign_keys should be 1 after _ensure_connection(), got {result[0]}"
        )

    def test_wal_mode_preserved(self, tmp_path):
        """FK pragma must not break WAL mode."""
        from truememory.storage import create_db

        conn = create_db(tmp_path / "wal_fk.db")
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal.lower() == "wal", f"Expected WAL mode, got {journal}"
        conn.close()

    def test_busy_timeout_preserved(self, tmp_path):
        """FK pragma must not break busy_timeout."""
        from truememory.storage import create_db, DEFAULT_BUSY_TIMEOUT_MS

        conn = create_db(tmp_path / "timeout_fk.db")
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == DEFAULT_BUSY_TIMEOUT_MS, (
            f"Expected busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}, got {timeout}"
        )
        conn.close()
