"""Regression tests for issue #465: Qwen3 NaN fix blocks MCP synchronously.

The NaN fix should:
1. Not block _ensure_connection() for minutes
2. Skip on fresh/empty databases
3. Set the flag immediately to prevent re-runs
4. Run re-embedding in background for file-backed DBs
"""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestIssue465Qwen3NanAsync:
    """Verify NaN fix doesn't block MCP startup."""

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

    def test_issue_465_fresh_db_not_blocked(self):
        """Fresh database should not trigger NaN re-embed."""
        from truememory.client import Memory

        mock_model = self._make_mock_model(delay=0.5)

        start = time.monotonic()
        with patch("truememory.vector_search.get_model", return_value=mock_model):
            m = Memory(path=":memory:")
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, (
            f"Fresh DB init took {elapsed:.1f}s — NaN fix likely triggered "
            "on empty database"
        )

    def test_issue_465_flag_set_immediately(self):
        """The qwen3_nan_fix_applied flag must be set before re-embedding."""
        from truememory.client import Memory

        mock_model = self._make_mock_model()

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            m = Memory(path=":memory:")
            m.add(content="test", user_id="alice")

        conn = m._engine.conn
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'qwen3_nan_fix_applied'"
        ).fetchone()
        assert row is not None, (
            "qwen3_nan_fix_applied flag not set — NaN fix will re-run"
        )

    def test_issue_465_second_connection_fast(self):
        """Second connection to same DB must not re-run NaN fix."""
        import tempfile
        from pathlib import Path
        from truememory.client import Memory

        mock_model = self._make_mock_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with patch("truememory.vector_search.get_model", return_value=mock_model):
                m1 = Memory(path=str(db_path))
                m1.add(content="first", user_id="alice")

            encode_count = [0]
            original_encode = mock_model.encode

            def counting_encode(texts, **kw):
                encode_count[0] += 1
                return np.array(
                    [np.random.rand(256).astype(np.float32)] * len(texts)
                )

            mock_model.encode = counting_encode

            start = time.monotonic()
            with patch("truememory.vector_search.get_model", return_value=mock_model):
                with patch("truememory.vector_search._model", None):
                    m2 = Memory(path=str(db_path))
            elapsed = time.monotonic() - start

            assert elapsed < 2.0, (
                f"Second connection took {elapsed:.1f}s — NaN fix re-ran"
            )
