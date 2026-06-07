"""Regression tests for issue #484: truememory_consolidate bypasses write lock.

The consolidation MCP tool accesses the engine's connection directly without
acquiring _write_lock, allowing races with concurrent add() calls.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestIssue484ConsolidateLock:

    def _make_engine(self):
        from truememory.client import Memory
        m = Memory(path=":memory:")
        fake_embedding = np.random.rand(256).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.array(
            [fake_embedding] * len(texts)
        )
        with patch("truememory.vector_search.get_model", return_value=mock_model):
            for i in range(5):
                m.add(content=f"Test message {i} about project alpha", user_id="alice")
        return m

    def test_issue_484_consolidate_holds_write_lock(self):
        """Consolidation must acquire _write_lock before modifying tables."""
        m = self._make_engine()
        engine = m._engine
        engine._ensure_connection()

        lock_was_held = []

        original_build_summaries = None
        try:
            from truememory.consolidation import build_summaries as _orig
            original_build_summaries = _orig
        except ImportError:
            pytest.skip("consolidation module not available")

        def patched_build_summaries(conn):
            lock_was_held.append(engine._write_lock.locked())
            return original_build_summaries(conn)

        with patch("truememory.consolidation.build_summaries", side_effect=patched_build_summaries):
            engine.consolidate()

        assert any(lock_was_held), (
            "build_summaries ran without _write_lock held. "
            f"Lock states observed: {lock_was_held}"
        )

    def test_issue_484_consolidate_concurrent_add(self):
        """Concurrent add() and consolidate() must not corrupt data."""
        m = self._make_engine()
        engine = m._engine

        fake_embedding = np.random.rand(256).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.array(
            [fake_embedding] * len(texts)
        )

        errors = []

        def do_adds():
            try:
                with patch("truememory.vector_search.get_model", return_value=mock_model):
                    for i in range(3):
                        m.add(content=f"Concurrent message {i}", user_id="bob")
            except Exception as e:
                errors.append(e)

        def do_consolidate():
            try:
                engine.consolidate()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=do_adds)
        t2 = threading.Thread(target=do_consolidate)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert len(errors) == 0, f"Concurrent errors: {errors}"

    def test_issue_484_consolidate_no_busy_errors(self):
        """Consolidation under write load must not produce SQLITE_BUSY."""
        m = self._make_engine()
        engine = m._engine

        busy_errors = []

        def do_consolidate():
            try:
                engine.consolidate()
            except Exception as e:
                if "SQLITE_BUSY" in str(e) or "database is locked" in str(e):
                    busy_errors.append(e)

        t = threading.Thread(target=do_consolidate)
        t.start()
        t.join(timeout=30)

        assert len(busy_errors) == 0, f"SQLITE_BUSY errors: {busy_errors}"
