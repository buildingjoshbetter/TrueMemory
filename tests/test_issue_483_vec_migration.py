"""Regression tests for issue #483: migrate_legacy_vec_tables destroys fresh tables.

On a fresh database, init_vec_table() creates vec_messages with 0 rows.
migrate_legacy_vec_tables() finds it empty, DROPs it without replacement,
leaving the database with no vector table and broken semantic search.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


def _has_table(conn, table_name):
    """Check if a table exists in the database."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name=? AND type='table'",
        (table_name,),
    ).fetchone()
    return row is not None


class TestIssue483VecMigration:

    def test_issue_483_fresh_db_keeps_vec_tables(self):
        """Fresh DB: vec_messages must survive migrate_legacy_vec_tables."""
        from truememory.storage import create_db
        from truememory.vector_search import init_vec_table, migrate_legacy_vec_tables

        conn = create_db(":memory:")

        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except (ImportError, OSError):
            pytest.skip("sqlite-vec not available")

        init_vec_table(conn)

        assert _has_table(conn, "vec_messages"), (
            "vec_messages should exist after init_vec_table"
        )

        migrate_legacy_vec_tables(conn)

        assert _has_table(conn, "vec_messages"), (
            "vec_messages was destroyed by migrate_legacy_vec_tables on fresh DB! "
            "The migration should not drop empty tables unless tiered replacements exist."
        )

    def test_issue_483_fresh_db_embed_works(self):
        """Fresh DB: adding a memory must succeed with embedding."""
        from truememory.client import Memory

        m = Memory(path=":memory:")
        fake_embedding = np.random.rand(256).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.array(
            [fake_embedding] * len(texts)
        )

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            m.add(content="Test memory for embedding", user_id="test_user")

        conn = m._engine.conn
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vec_%'"
        ).fetchall()]
        assert len(tables) > 0, (
            f"No vec tables found after add(). Available tables: {tables}"
        )

    def test_issue_483_legacy_migration_still_works(self):
        """Legacy DB with populated vec tables: migration must still function."""
        from truememory.storage import create_db
        from truememory.vector_search import migrate_legacy_vec_tables

        conn = create_db(":memory:")

        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except (ImportError, OSError):
            pytest.skip("sqlite-vec not available")

        conn.execute(
            "CREATE VIRTUAL TABLE vec_messages USING vec0(embedding float[256])"
        )
        test_embedding = np.random.rand(256).astype(np.float32)
        conn.execute(
            "INSERT INTO vec_messages(rowid, embedding) VALUES (1, ?)",
            (test_embedding.tobytes(),),
        )
        conn.commit()

        result = migrate_legacy_vec_tables(conn)

        assert result is True, "Migration should have occurred"
        assert not _has_table(conn, "vec_messages"), (
            "Legacy vec_messages should be dropped after migration"
        )
