"""Regression tests for issues #485 and #499: NaN background thread safety.

Issue #485 — Background NaN Re-embed Drops Tables Mid-Search:
    The qwen3_nan_fix_applied metadata flag is set BEFORE the background thread
    starts.  If the thread fails, the flag is already committed, so NaN vectors
    persist permanently with no way to retry.

Issue #499 — Background Thread Doesn't Load sqlite-vec:
    The background thread opens a new SQLite connection but never calls
    sqlite_vec.load(), so build_vectors() / build_separation_vectors() fail
    because the virtual-table module is missing.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from tests.conftest import requires_sqlite_ext

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Qwen3 NaN background thread only runs on macOS",
)


def _make_mock_model():
    """Return a mock embedding model that produces valid 256-d vectors."""
    mock = MagicMock()
    def _encode(texts, **kw):
        return np.array(
            [np.random.rand(256).astype(np.float32)] * len(texts)
        )
    mock.encode = _encode
    return mock


def _seed_db(db_path: Path, n: int = 3) -> None:
    """Seed *n* messages into a fresh TrueMemory database at *db_path*.

    Uses the Memory client so schema, FTS, vec tables, and the metadata row
    are all created normally.
    """
    from truememory.client import Memory

    mock_model = _make_mock_model()
    with patch("truememory.vector_search.get_model", return_value=mock_model):
        m = Memory(path=str(db_path))
        for i in range(n):
            m.add(content=f"message {i}", user_id="alice")


def _clear_nan_flag(db_path: Path) -> None:
    """Remove the qwen3_nan_fix_applied flag so next init triggers migration."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM metadata WHERE key = 'qwen3_nan_fix_applied'")
    conn.commit()
    conn.close()


@requires_sqlite_ext
class TestIssue485NanFlagAfterCompletion:
    """qwen3_nan_fix_applied must only be set AFTER successful re-embed."""

    def test_issue_485_nan_flag_set_after_completion(self):
        """Flag must not exist until the background thread finishes OK."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            mock_model = _make_mock_model()
            with patch("truememory.vector_search.get_model", return_value=mock_model):
                _seed_db(db_path, n=2)

            _clear_nan_flag(db_path)

            # Use an event to control when build_vectors proceeds
            proceed_event = threading.Event()
            flag_before_build = []

            from truememory import vector_search as _vs
            _real_bv = _vs.build_vectors

            def _checking_build(conn_arg, **kw):
                """Check whether the flag exists before build_vectors runs."""
                # Read the flag from a separate connection to avoid locking
                check_conn = sqlite3.connect(str(db_path))
                row = check_conn.execute(
                    "SELECT value FROM metadata WHERE key = 'qwen3_nan_fix_applied'"
                ).fetchone()
                check_conn.close()
                flag_before_build.append(row is not None)
                proceed_event.set()
                return _real_bv(conn_arg, **kw)

            with patch("truememory.vector_search.get_model", return_value=mock_model):
                with patch.object(_vs, "build_vectors", side_effect=_checking_build):
                    from truememory.client import Memory
                    m = Memory(path=str(db_path))
                    # Trigger lazy _ensure_connection
                    m.search(query="test", user_id="alice")

                    # Wait for the background thread to reach build_vectors
                    proceed_event.wait(timeout=5.0)
                    # Give it a bit more to finish
                    time.sleep(1.0)

            assert len(flag_before_build) > 0, (
                "build_vectors was never called — test setup is wrong"
            )
            assert not flag_before_build[0], (
                "qwen3_nan_fix_applied flag was set BEFORE build_vectors — "
                "if the thread crashes, the flag is stuck and NaN vectors "
                "persist forever (issue #485)"
            )


@requires_sqlite_ext
class TestIssue485ThreadFailureAllowsRetry:
    """A failed background thread must NOT set the flag, so next init retries."""

    def test_issue_485_nan_thread_failure_allows_retry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            mock_model = _make_mock_model()
            with patch("truememory.vector_search.get_model", return_value=mock_model):
                _seed_db(db_path, n=2)

            _clear_nan_flag(db_path)

            # Make build_vectors crash inside the thread
            crash_event = threading.Event()
            from truememory import vector_search as _vs

            def _crashing_build(*a, **kw):
                crash_event.set()
                raise RuntimeError("simulated GPU OOM")

            with patch("truememory.vector_search.get_model", return_value=mock_model):
                with patch.object(_vs, "build_vectors", side_effect=_crashing_build):
                    from truememory.client import Memory
                    m = Memory(path=str(db_path))
                    # Trigger lazy _ensure_connection
                    m.search(query="test", user_id="alice")

                    # Wait for the background thread to crash
                    crash_event.wait(timeout=5.0)
                    time.sleep(1.0)

            # The flag must NOT be set after a failure
            conn = sqlite3.connect(str(db_path))
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = 'qwen3_nan_fix_applied'"
            ).fetchone()
            conn.close()

            assert row is None, (
                "qwen3_nan_fix_applied flag was set despite thread failure — "
                "future inits will never retry, leaving NaN vectors "
                "permanently (issue #485)"
            )


@requires_sqlite_ext
class TestIssue499NanThreadLoadsSqliteVec:
    """Background thread must load sqlite-vec before calling build_vectors."""

    def test_issue_499_nan_thread_loads_sqlite_vec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            mock_model = _make_mock_model()
            with patch("truememory.vector_search.get_model", return_value=mock_model):
                _seed_db(db_path, n=2)

            _clear_nan_flag(db_path)

            # Track whether sqlite_vec.load is called inside the thread
            vec_load_calls = []
            done_event = threading.Event()

            try:
                import sqlite_vec as _real_sv
                original_load = _real_sv.load
            except ImportError:
                pytest.skip("sqlite_vec not installed")

            def _tracking_load(conn_arg):
                vec_load_calls.append(threading.current_thread().name)
                return original_load(conn_arg)

            from truememory import vector_search as _vs
            _real_bv = _vs.build_vectors

            def _signaling_build(conn_arg, **kw):
                result = _real_bv(conn_arg, **kw)
                done_event.set()
                return result

            with patch("truememory.vector_search.get_model", return_value=mock_model):
                with patch("sqlite_vec.load", side_effect=_tracking_load):
                    with patch.object(_vs, "build_vectors", side_effect=_signaling_build):
                        from truememory.client import Memory
                        m = Memory(path=str(db_path))
                        # Trigger lazy _ensure_connection
                        m.search(query="test", user_id="alice")

                        # Give background thread time to complete
                        done_event.wait(timeout=5.0)
                        time.sleep(0.5)

            # sqlite_vec.load must have been called from a non-main thread
            # (i.e., the background migration thread loaded the extension)
            bg_calls = [
                name for name in vec_load_calls
                if name != threading.main_thread().name
            ]
            assert len(bg_calls) > 0, (
                "sqlite_vec.load was never called from the background thread — "
                "build_vectors() will fail because vec virtual tables "
                "are not available (issue #499)"
            )
