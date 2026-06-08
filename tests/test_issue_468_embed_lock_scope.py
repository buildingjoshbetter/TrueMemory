"""Regression tests for issue #468: embed_single runs inside _write_lock.

Model inference (embedding) should NOT block other writes. The embedding
should be pre-computed outside the lock, with only the DB INSERT inside.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestIssue468EmbedLockScope:
    """Verify embedding is computed outside the write lock."""

    def _make_mock_model(self, delay=0.0):
        mock_model = MagicMock()
        def slow_encode(texts, **kw):
            if delay > 0:
                time.sleep(delay)
            return np.array(
                [np.random.rand(256).astype(np.float32)] * len(texts)
            )
        mock_model.encode = slow_encode
        return mock_model

    def test_issue_468_concurrent_adds_not_serialized_on_embedding(self):
        """Two concurrent add() calls should not serialize on model inference."""
        from truememory.client import Memory

        mock_model = self._make_mock_model(delay=0.1)

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            m = Memory(path=":memory:")

            timings = []

            def timed_add(content):
                start = time.monotonic()
                m.add(content=content, user_id="alice")
                elapsed = time.monotonic() - start
                timings.append(elapsed)

            t1 = threading.Thread(target=timed_add, args=("First memory",))
            t2 = threading.Thread(target=timed_add, args=("Second memory",))

            start = time.monotonic()
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            wall_clock = time.monotonic() - start

            assert wall_clock < 0.5, (
                f"Two adds took {wall_clock:.2f}s — embedding is likely "
                f"serialized inside the write lock (expected <0.5s with "
                f"0.1s embed delay if parallelized)"
            )

    def test_issue_468_add_still_works_correctly(self):
        """add() must still produce correct results after restructuring."""
        from truememory.client import Memory

        mock_model = self._make_mock_model()

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            m = Memory(path=":memory:")
            result = m.add(content="Test memory", user_id="alice")

        assert result["id"] > 0
        assert result["content"] == "Test memory"

        conn = m._engine.conn
        msg = conn.execute("SELECT * FROM messages WHERE id = ?", (result["id"],)).fetchone()
        assert msg is not None

    def test_issue_468_vectors_still_created(self):
        """Vectors must still be created after restructuring add()."""
        from truememory.client import Memory

        mock_model = self._make_mock_model()

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            m = Memory(path=":memory:")
            result = m.add(content="Test memory", user_id="alice")

        conn = m._engine.conn
        from truememory.vector_search import _active_vec_table
        vec_tbl = _active_vec_table(conn)
        vec_row = conn.execute(
            f"SELECT rowid FROM {vec_tbl} WHERE rowid = ?", (result["id"],)
        ).fetchone()
        assert vec_row is not None, "Vector embedding not created after add()"
