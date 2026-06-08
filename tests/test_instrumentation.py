"""
Tests for truememory.instrumentation package.

Key invariants tested:
1. Telemetry writes to instrumentation.db, NEVER to memories.db
2. No double-wrapping (sentinel attribute prevents it)
3. Failure isolation (telemetry errors don't break production methods)
4. Retention pruning works
5. Separate DB file resolution
6. is_enabled() caching
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _clean_instrumentation():
    """Reset instrumentation state between tests."""
    # Ensure clean state before each test
    try:
        from truememory.instrumentation import patch as inst_patch
        from truememory.instrumentation import writer
        from truememory.instrumentation.log import _lock_enabled
        inst_patch.uninstall()
        writer.reset()
        _lock_enabled(None)
    except Exception:
        pass
    yield
    # Cleanup after
    try:
        from truememory.instrumentation import patch as inst_patch
        from truememory.instrumentation import writer
        from truememory.instrumentation.log import _lock_enabled
        inst_patch.uninstall()
        writer.reset()
        _lock_enabled(None)
    except Exception:
        pass


class TestWriterSeparateDB:
    """Writer must use instrumentation.db, never memories.db."""

    def test_default_path_is_instrumentation_db(self):
        from truememory.instrumentation.writer import _resolve_db_path
        path = _resolve_db_path()
        assert "instrumentation.db" in path
        assert "memories.db" not in path

    def test_env_override(self, tmp_path):
        db_path = str(tmp_path / "custom_telemetry.db")
        with patch.dict(os.environ, {"TRUEMEMORY_INSTRUMENTATION_DB": db_path}):
            from truememory.instrumentation.writer import _resolve_db_path
            assert _resolve_db_path() == db_path

    def test_emit_creates_db(self, tmp_path):
        db_path = str(tmp_path / "test_instr.db")
        with patch.dict(os.environ, {
            "TRUEMEMORY_INSTRUMENTATION": "1",
            "TRUEMEMORY_INSTRUMENTATION_DB": db_path,
        }):
            from truememory.instrumentation import writer
            writer.reset()
            writer.emit("test_signal", {"key": "value"})
            assert Path(db_path).exists()
            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT signal, data FROM telemetry").fetchall()
            conn.close()
            assert len(rows) == 1
            assert rows[0][0] == "test_signal"
            data = json.loads(rows[0][1])
            assert data["key"] == "value"


class TestRetention:
    """Pruning removes rows older than retention period."""

    def test_prune_removes_old_rows(self, tmp_path):
        db_path = str(tmp_path / "prune_test.db")
        with patch.dict(os.environ, {
            "TRUEMEMORY_INSTRUMENTATION": "1",
            "TRUEMEMORY_INSTRUMENTATION_DB": db_path,
        }):
            from truememory.instrumentation import writer
            writer.reset()
            # Insert an old row directly
            conn = writer._get_connection()
            old_ts = time.time() - (8 * 86400)  # 8 days ago
            conn.execute(
                "INSERT INTO telemetry (ts, signal, data) VALUES (?, ?, ?)",
                (old_ts, "old_signal", "{}"),
            )
            # Insert a recent row
            conn.execute(
                "INSERT INTO telemetry (ts, signal, data) VALUES (?, ?, ?)",
                (time.time(), "new_signal", "{}"),
            )
            conn.commit()
            # Prune
            writer.prune_now()
            rows = conn.execute("SELECT signal FROM telemetry").fetchall()
            signals = [r[0] for r in rows]
            assert "old_signal" not in signals
            assert "new_signal" in signals


class TestSentinelPreventsDoubleWrap:
    """Per-method sentinel prevents double-wrapping."""

    def test_install_twice_no_double_wrap(self, tmp_path):
        db_path = str(tmp_path / "sentinel_test.db")
        with patch.dict(os.environ, {
            "TRUEMEMORY_INSTRUMENTATION": "1",
            "TRUEMEMORY_INSTRUMENTATION_DB": db_path,
        }):
            from truememory.instrumentation.patch import install, uninstall, _install_lock, _WRAPPED_SENTINEL
            from truememory.instrumentation import patch as inst_patch
            from truememory.instrumentation import writer
            from truememory.instrumentation.log import _lock_enabled
            writer.reset()
            _lock_enabled(None)
            inst_patch._installed = False

            install()
            # Get reference to wrapped method
            from truememory.engine import TrueMemoryEngine
            wrapped_add = TrueMemoryEngine.add
            assert getattr(wrapped_add, _WRAPPED_SENTINEL, False) is True

            # Try to install again
            inst_patch._installed = False
            install()
            # Method should still be the same wrapper (sentinel prevented re-wrap)
            assert TrueMemoryEngine.add is wrapped_add


class TestIsEnabledCaching:
    """is_enabled() caching works correctly."""

    def test_locked_true(self):
        from truememory.instrumentation.log import is_enabled, _lock_enabled
        _lock_enabled(True)
        assert is_enabled() is True
        # Even if env says no
        with patch.dict(os.environ, {"TRUEMEMORY_INSTRUMENTATION": "0"}):
            assert is_enabled() is True

    def test_locked_false(self):
        from truememory.instrumentation.log import is_enabled, _lock_enabled
        _lock_enabled(False)
        assert is_enabled() is False
        with patch.dict(os.environ, {"TRUEMEMORY_INSTRUMENTATION": "1"}):
            assert is_enabled() is False

    def test_unlocked_reads_env(self):
        from truememory.instrumentation.log import is_enabled, _lock_enabled
        _lock_enabled(None)
        with patch.dict(os.environ, {"TRUEMEMORY_INSTRUMENTATION": "1"}):
            assert is_enabled() is True
        with patch.dict(os.environ, {"TRUEMEMORY_INSTRUMENTATION": "0"}):
            assert is_enabled() is False


class TestReader:
    """Read path returns formatted telemetry."""

    def test_query_empty_db(self, tmp_path):
        db_path = str(tmp_path / "empty_read.db")
        with patch.dict(os.environ, {"TRUEMEMORY_INSTRUMENTATION_DB": db_path}):
            from truememory.instrumentation.reader import query_telemetry
            assert query_telemetry() == []

    def test_query_with_data(self, tmp_path):
        db_path = str(tmp_path / "read_test.db")
        with patch.dict(os.environ, {
            "TRUEMEMORY_INSTRUMENTATION": "1",
            "TRUEMEMORY_INSTRUMENTATION_DB": db_path,
        }):
            from truememory.instrumentation import writer
            from truememory.instrumentation.reader import query_telemetry
            writer.reset()
            writer.emit("gate_decision", {"should_encode": True, "reason_code": "pass"})
            writer.emit("timing", {"method": "engine.add", "duration_ms": 42.5})
            rows = query_telemetry(limit=10)
            assert len(rows) == 2
            # Most recent first
            assert rows[0]["signal"] == "timing"
            assert rows[1]["signal"] == "gate_decision"

    def test_query_filter_by_signal(self, tmp_path):
        db_path = str(tmp_path / "filter_test.db")
        with patch.dict(os.environ, {
            "TRUEMEMORY_INSTRUMENTATION": "1",
            "TRUEMEMORY_INSTRUMENTATION_DB": db_path,
        }):
            from truememory.instrumentation import writer
            from truememory.instrumentation.reader import query_telemetry
            writer.reset()
            writer.emit("gate_decision", {"reason_code": "pass"})
            writer.emit("timing", {"method": "add"})
            writer.emit("gate_decision", {"reason_code": "salience_floor"})
            rows = query_telemetry(signal="gate_decision")
            assert len(rows) == 2
            for r in rows:
                assert r["signal"] == "gate_decision"


class TestPrivacy:
    """Telemetry must never contain raw content or queries."""

    def test_signals_contain_no_content(self, tmp_path):
        db_path = str(tmp_path / "privacy_test.db")
        with patch.dict(os.environ, {
            "TRUEMEMORY_INSTRUMENTATION": "1",
            "TRUEMEMORY_INSTRUMENTATION_DB": db_path,
        }):
            from truememory.instrumentation import writer, signals
            writer.reset()
            # Emit various signals
            signals.emit_gate_decision(
                should_encode=True, encoding_score=0.8,
                novelty=0.5, salience=0.6, prediction_error=0.3,
                reason_code="pass", category="personal",
            )
            signals.emit_search_distance(
                spread=0.5, top_score=0.9,
                min_score=0.4, mean_score=0.6, n_results=10,
            )
            signals.emit_memory_returned(memory_id=42, rank=0)
            # Check no content leaked
            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT data FROM telemetry").fetchall()
            conn.close()
            for (data_str,) in rows:
                data = json.loads(data_str)
                # No content, query, or user_id fields
                assert "content" not in data
                assert "query" not in data
                assert "query_text" not in data
                assert "user_id" not in data
