"""Regression tests for issue #462: delete() orphans data in dependent tables.

engine.delete(memory_id) removes from messages and vec_messages but does
NOT clean up fact_timeline, causal_edges, surprise_scores, or
landmark_events. Orphaned data wastes storage and can produce stale
search results.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestIssue462DeleteCascade:
    """Verify delete() cascades to dependent tables."""

    def _make_memory_with_data(self):
        """Create a Memory with messages and populate dependent tables."""
        from truememory.client import Memory

        m = Memory(path=":memory:")
        fake_embedding = np.random.rand(256).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.array(
            [fake_embedding] * len(texts)
        )

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            m.add(content="Met Alice at the conference", user_id="tester")
            m.add(content="Alice introduced me to Bob", user_id="tester")
            m.add(content="Bob is starting a new project", user_id="tester")

        conn = m._engine.conn
        msg_ids = [r[0] for r in conn.execute(
            "SELECT id FROM messages ORDER BY id"
        ).fetchall()]

        conn.execute(
            "CREATE TABLE IF NOT EXISTS fact_timeline ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "subject TEXT, fact TEXT, source_message_id INTEGER, "
            "timestamp TEXT, superseded_by INTEGER, "
            "entity_scope TEXT DEFAULT '', "
            "valid_from TEXT DEFAULT '', valid_to TEXT DEFAULT '')"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS causal_edges ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "cause_msg_id INTEGER, effect_msg_id INTEGER, "
            "relationship TEXT DEFAULT '', confidence REAL DEFAULT 0.0)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS surprise_scores ("
            "message_id INTEGER PRIMARY KEY, "
            "surprise REAL DEFAULT 0.0, "
            "fact_count INTEGER DEFAULT 0, "
            "new_fact_count INTEGER DEFAULT 0)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS landmark_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "event_name TEXT, timestamp TEXT, event_type TEXT DEFAULT '', "
            "related_entities TEXT DEFAULT '[]', source_message_id INTEGER)"
        )

        conn.execute(
            "INSERT INTO fact_timeline (subject, fact, source_message_id) "
            "VALUES (?, ?, ?)", ("Alice", "works at company", msg_ids[0])
        )
        conn.execute(
            "INSERT INTO causal_edges (cause_msg_id, effect_msg_id, relationship) "
            "VALUES (?, ?, ?)", (msg_ids[0], msg_ids[1], "led_to")
        )
        conn.execute(
            "INSERT INTO surprise_scores (message_id, surprise, fact_count, new_fact_count) "
            "VALUES (?, ?, ?, ?)", (msg_ids[0], 0.8, 3, 2)
        )
        conn.execute(
            "INSERT INTO landmark_events (event_name, timestamp, source_message_id) "
            "VALUES (?, ?, ?)", ("conference", "2026-01-01", msg_ids[0])
        )
        conn.commit()

        return m, msg_ids

    def test_issue_462_delete_cascades_to_dependent_tables(self):
        """delete() must remove orphaned rows from all dependent tables."""
        m, msg_ids = self._make_memory_with_data()
        conn = m._engine.conn
        target_id = msg_ids[0]

        m.delete(target_id)

        fact_count = conn.execute(
            "SELECT COUNT(*) FROM fact_timeline WHERE source_message_id = ?",
            (target_id,)
        ).fetchone()[0]
        assert fact_count == 0, (
            f"fact_timeline has {fact_count} orphaned rows after delete"
        )

        causal_count = conn.execute(
            "SELECT COUNT(*) FROM causal_edges "
            "WHERE cause_msg_id = ? OR effect_msg_id = ?",
            (target_id, target_id)
        ).fetchone()[0]
        assert causal_count == 0, (
            f"causal_edges has {causal_count} orphaned rows after delete"
        )

        surprise_count = conn.execute(
            "SELECT COUNT(*) FROM surprise_scores WHERE message_id = ?",
            (target_id,)
        ).fetchone()[0]
        assert surprise_count == 0, (
            f"surprise_scores has {surprise_count} orphaned rows after delete"
        )

        landmark_count = conn.execute(
            "SELECT COUNT(*) FROM landmark_events WHERE source_message_id = ?",
            (target_id,)
        ).fetchone()[0]
        assert landmark_count == 0, (
            f"landmark_events has {landmark_count} orphaned rows after delete"
        )

    def test_issue_462_delete_preserves_unrelated_rows(self):
        """Cascade must only remove rows for the deleted message, not others."""
        m, msg_ids = self._make_memory_with_data()
        conn = m._engine.conn

        conn.execute(
            "INSERT INTO fact_timeline (subject, fact, source_message_id) "
            "VALUES (?, ?, ?)", ("Bob", "has project", msg_ids[2])
        )
        conn.execute(
            "INSERT INTO surprise_scores (message_id, surprise, fact_count, new_fact_count) "
            "VALUES (?, ?, ?, ?)", (msg_ids[2], 0.5, 1, 1)
        )
        conn.commit()

        m.delete(msg_ids[0])

        remaining_facts = conn.execute(
            "SELECT COUNT(*) FROM fact_timeline WHERE source_message_id = ?",
            (msg_ids[2],)
        ).fetchone()[0]
        assert remaining_facts == 1, "Cascade deleted unrelated rows"

        remaining_surprise = conn.execute(
            "SELECT COUNT(*) FROM surprise_scores WHERE message_id = ?",
            (msg_ids[2],)
        ).fetchone()[0]
        assert remaining_surprise == 1, "Cascade deleted unrelated surprise scores"

    def test_issue_462_delete_handles_missing_tables(self):
        """delete() must not crash if dependent tables don't exist yet."""
        from truememory.client import Memory

        m = Memory(path=":memory:")
        fake_embedding = np.random.rand(256).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.array(
            [fake_embedding] * len(texts)
        )

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            m.add(content="Test memory", user_id="tester")

        msg_id = m._engine.conn.execute(
            "SELECT id FROM messages LIMIT 1"
        ).fetchone()[0]

        for tbl in ("fact_timeline", "causal_edges", "surprise_scores", "landmark_events"):
            m._engine.conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        m._engine.conn.commit()

        result = m.delete(msg_id)
        assert result is True, "delete should succeed even without dependent tables"
