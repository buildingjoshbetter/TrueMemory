"""Regression tests for issues #493 and #500: cascade delete misses tables.

H7 #493: delete_all() orphans surprise_scores, message_clusters,
         cluster_centroids tables.
M3 #500: message_clusters missing from delete(), delete_all(), and
         delete_all(user_id) paths.

All three delete paths must clean up surprise_scores, message_clusters,
and cluster_centroids so no orphaned rows remain.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


def _setup_memory_with_clusters():
    """Create an in-memory Memory with messages plus cluster/surprise data."""
    from truememory.client import Memory

    m = Memory(path=":memory:")
    fake_embedding = np.random.rand(256).astype(np.float32)
    mock_model = MagicMock()
    mock_model.encode = lambda texts, **kw: np.array(
        [fake_embedding] * len(texts)
    )

    with patch("truememory.vector_search.get_model", return_value=mock_model):
        m.add(content="Alice likes hiking", user_id="alice")
        m.add(content="Alice goes running daily", user_id="alice")
        m.add(content="Bob prefers swimming", user_id="bob")

    conn = m._engine.conn
    msg_ids = [r[0] for r in conn.execute(
        "SELECT id FROM messages ORDER BY id"
    ).fetchall()]

    # Create the tables that cascade-delete must clean
    conn.execute(
        "CREATE TABLE IF NOT EXISTS surprise_scores ("
        "message_id INTEGER PRIMARY KEY, "
        "surprise REAL DEFAULT 0.0, "
        "fact_count INTEGER DEFAULT 0, "
        "new_fact_count INTEGER DEFAULT 0, "
        "FOREIGN KEY (message_id) REFERENCES messages(id))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS message_clusters ("
        "message_id INTEGER PRIMARY KEY REFERENCES messages(id), "
        "cluster_id INTEGER NOT NULL, "
        "noise INTEGER DEFAULT 0)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cluster_centroids ("
        "cluster_id INTEGER PRIMARY KEY, "
        "centroid BLOB NOT NULL, "
        "message_count INTEGER DEFAULT 0, "
        "session_range TEXT DEFAULT '', "
        "summary TEXT DEFAULT '')"
    )

    # Populate dependent data
    for mid in msg_ids:
        conn.execute(
            "INSERT INTO surprise_scores (message_id, surprise, fact_count, new_fact_count) "
            "VALUES (?, 0.5, 2, 1)", (mid,)
        )
    # All three messages in cluster 0
    for mid in msg_ids:
        conn.execute(
            "INSERT INTO message_clusters (message_id, cluster_id, noise) "
            "VALUES (?, 0, 0)", (mid,)
        )
    # One centroid for cluster 0
    conn.execute(
        "INSERT INTO cluster_centroids (cluster_id, centroid, message_count) "
        "VALUES (?, ?, ?)", (0, b"\x00" * 16, 3)
    )
    conn.commit()

    return m, msg_ids


class TestIssue493DeleteAll:
    """delete_all() must wipe surprise_scores, message_clusters, cluster_centroids."""

    def test_delete_all_clears_surprise_scores(self):
        m, _ = _setup_memory_with_clusters()
        conn = m._engine.conn

        m.delete_all()

        count = conn.execute("SELECT COUNT(*) FROM surprise_scores").fetchone()[0]
        assert count == 0, f"surprise_scores has {count} orphaned rows after delete_all()"

    def test_delete_all_clears_message_clusters(self):
        m, _ = _setup_memory_with_clusters()
        conn = m._engine.conn

        m.delete_all()

        count = conn.execute("SELECT COUNT(*) FROM message_clusters").fetchone()[0]
        assert count == 0, f"message_clusters has {count} orphaned rows after delete_all()"

    def test_delete_all_clears_cluster_centroids(self):
        m, _ = _setup_memory_with_clusters()
        conn = m._engine.conn

        m.delete_all()

        count = conn.execute("SELECT COUNT(*) FROM cluster_centroids").fetchone()[0]
        assert count == 0, f"cluster_centroids has {count} orphaned rows after delete_all()"


class TestIssue493DeleteAllUser:
    """delete_all(user_id) must clean these tables for that user's messages."""

    def test_delete_all_user_clears_surprise_scores(self):
        m, msg_ids = _setup_memory_with_clusters()
        conn = m._engine.conn

        # alice owns msg_ids[0] and msg_ids[1]
        alice_ids = set(msg_ids[:2])

        m.delete_all(user_id="alice")

        for mid in alice_ids:
            count = conn.execute(
                "SELECT COUNT(*) FROM surprise_scores WHERE message_id = ?", (mid,)
            ).fetchone()[0]
            assert count == 0, f"surprise_scores still has row for deleted msg {mid}"

        # bob's row should remain
        bob_count = conn.execute(
            "SELECT COUNT(*) FROM surprise_scores WHERE message_id = ?", (msg_ids[2],)
        ).fetchone()[0]
        assert bob_count == 1, "delete_all(user_id) deleted bob's surprise_scores"

    def test_delete_all_user_clears_message_clusters(self):
        m, msg_ids = _setup_memory_with_clusters()
        conn = m._engine.conn

        alice_ids = set(msg_ids[:2])

        m.delete_all(user_id="alice")

        for mid in alice_ids:
            count = conn.execute(
                "SELECT COUNT(*) FROM message_clusters WHERE message_id = ?", (mid,)
            ).fetchone()[0]
            assert count == 0, f"message_clusters still has row for deleted msg {mid}"

        bob_count = conn.execute(
            "SELECT COUNT(*) FROM message_clusters WHERE message_id = ?", (msg_ids[2],)
        ).fetchone()[0]
        assert bob_count == 1, "delete_all(user_id) deleted bob's message_clusters"


class TestIssue500DeleteSingle:
    """delete() must remove message_clusters row for the deleted message."""

    def test_delete_clears_message_clusters(self):
        m, msg_ids = _setup_memory_with_clusters()
        conn = m._engine.conn

        target = msg_ids[0]
        m.delete(target)

        count = conn.execute(
            "SELECT COUNT(*) FROM message_clusters WHERE message_id = ?", (target,)
        ).fetchone()[0]
        assert count == 0, f"message_clusters has {count} orphaned rows after delete()"

    def test_delete_preserves_other_clusters(self):
        m, msg_ids = _setup_memory_with_clusters()
        conn = m._engine.conn

        m.delete(msg_ids[0])

        for mid in msg_ids[1:]:
            count = conn.execute(
                "SELECT COUNT(*) FROM message_clusters WHERE message_id = ?", (mid,)
            ).fetchone()[0]
            assert count == 1, f"delete() incorrectly removed clusters for msg {mid}"

    def test_delete_handles_missing_cluster_tables(self):
        """delete() must not crash if cluster tables don't exist."""
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

        # Drop tables to simulate a DB without them
        for tbl in ("message_clusters", "cluster_centroids", "surprise_scores"):
            m._engine.conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        m._engine.conn.commit()

        result = m.delete(msg_id)
        assert result is True, "delete should succeed even without cluster tables"
