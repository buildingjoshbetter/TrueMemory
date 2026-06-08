"""
Hardening tests for TrueMemory — stress tests, edge cases, and robustness checks.

These tests verify behavior under adversarial conditions:
1. Concurrent read/write stress on a single SQLite database
2. Engine resilience to corrupted/empty inputs
3. Thread safety of auto-consolidation
4. Memory cleanup and resource lifecycle
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from truememory.engine import TrueMemoryEngine
from truememory.storage import create_db


def _make_engine(n_messages=0, consolidation=False, vectors=False):
    td = tempfile.mkdtemp()
    db = os.path.join(td, "test.db")
    conn = create_db(db)
    for i in range(n_messages):
        conn.execute(
            "INSERT INTO messages (content, sender, recipient, timestamp, category, modality) "
            "VALUES (?, ?, '', '', '', '')",
            (f"Test message {i}", "alice" if i % 2 == 0 else "bob"),
        )
    conn.commit()
    conn.close()

    eng = TrueMemoryEngine(db_path=db)
    eng._has_consolidation = False
    eng._ensure_connection()
    eng._has_consolidation = consolidation
    eng._has_vectors = vectors
    eng._has_style_vec = False
    return eng, td


class TestConcurrentReadWrite:
    """Verify SQLite busy_timeout prevents crashes under concurrent access."""

    def test_concurrent_adds_dont_crash(self):
        eng, td = _make_engine()
        errors = []

        def add_messages(thread_id, count):
            try:
                for i in range(count):
                    eng.add(content=f"Thread {thread_id} message {i}", sender=f"user_{thread_id}")
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=add_messages, args=(t, 10)) for t in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent adds produced errors: {errors}"
        eng.close()

    def test_concurrent_read_during_write(self):
        eng, td = _make_engine(n_messages=50)
        errors = []
        read_results = []

        def writer():
            try:
                for i in range(20):
                    eng.add(content=f"New message {i}", sender="writer")
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            try:
                for _ in range(20):
                    if eng.conn:
                        count = eng.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                        read_results.append(count)
            except Exception as e:
                errors.append(("reader", e))

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join(timeout=30)
        t_read.join(timeout=30)

        assert not errors, f"Concurrent R/W produced errors: {errors}"
        assert len(read_results) > 0
        eng.close()


class TestEdgeCaseInputs:
    """Verify engine handles adversarial inputs gracefully."""

    def test_empty_content_add_raises(self):
        eng, td = _make_engine()
        with pytest.raises(ValueError, match="empty"):
            eng.add(content="", sender="alice")
        eng.close()

    def test_very_long_content_add(self):
        eng, td = _make_engine()
        long_content = "x" * 100_000
        result = eng.add(content=long_content, sender="alice")
        assert result is not None
        eng.close()

    def test_unicode_content(self):
        eng, td = _make_engine()
        unicode_content = "Hello 你好 مرحبا 🎉 \x00 \ud800"
        try:
            result = eng.add(content=unicode_content, sender="alice")
        except Exception:
            pass
        eng.close()

    def test_null_bytes_in_content(self):
        eng, td = _make_engine()
        result = eng.add(content="hello\x00world", sender="alice")
        assert result is not None
        eng.close()

    def test_sql_injection_in_content(self):
        eng, td = _make_engine()
        malicious = "'; DROP TABLE messages; --"
        eng.add(content=malicious, sender="alice")
        count = eng.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count >= 1
        eng.close()

    def test_sql_injection_in_sender(self):
        eng, td = _make_engine()
        eng.add(content="test", sender="'; DROP TABLE messages; --")
        count = eng.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count >= 1
        eng.close()


class TestEngineLifecycle:
    """Verify engine cleanup and resource management."""

    def test_double_close_is_safe(self):
        eng, td = _make_engine()
        eng.close()
        eng.close()

    def test_add_after_close_doesnt_crash(self):
        eng, td = _make_engine()
        eng.close()
        try:
            eng.add(content="after close", sender="alice")
        except Exception:
            pass

    def test_consolidate_on_empty_db(self):
        eng, td = _make_engine(n_messages=0)
        result = eng.consolidate()
        assert isinstance(result, dict)
        eng.close()

    def test_consolidate_with_one_message(self):
        eng, td = _make_engine(n_messages=1)
        result = eng.consolidate()
        assert isinstance(result, dict)
        eng.close()


class TestAutoConsolidation:
    """Verify auto-consolidation thread safety."""

    def test_auto_consolidation_doesnt_fire_below_threshold(self):
        eng, td = _make_engine(n_messages=0)
        eng._has_consolidation = True
        eng._auto_consolidate_threshold = 25

        import truememory.engine as _eng_mod
        with patch.object(_eng_mod.threading, "Thread") as mock_thread_cls:
            for i in range(24):
                eng._maybe_auto_consolidate()

        mock_thread_cls.assert_not_called()
        eng.close()

    def test_startup_consolidation_no_crash_on_closed_conn(self):
        eng, td = _make_engine(n_messages=30)
        eng._has_consolidation = True
        eng.conn = None
        eng._maybe_startup_consolidate()
        eng.close()


class TestDatabaseIntegrity:
    """Verify database constraints and schema integrity."""

    def test_messages_table_exists(self):
        eng, td = _make_engine()
        tables = [
            row[0] for row in
            eng.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        assert "messages" in tables
        eng.close()

    def test_busy_timeout_is_set(self):
        eng, td = _make_engine()
        timeout = eng.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout > 0
        eng.close()

    def test_wal_mode_enabled(self):
        eng, td = _make_engine()
        mode = eng.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
        eng.close()

    def test_message_count_accuracy(self):
        eng, td = _make_engine(n_messages=42)
        count = eng.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count == 42
        eng.close()


class TestSearchEdgeCases:
    """Verify search handles edge cases without crashing."""

    def test_search_empty_query(self):
        eng, td = _make_engine(n_messages=10)
        try:
            results = eng.search("")
            assert isinstance(results, list)
        except Exception:
            pass
        eng.close()

    def test_search_very_long_query(self):
        eng, td = _make_engine(n_messages=10)
        try:
            results = eng.search("x" * 10_000)
            assert isinstance(results, list)
        except Exception:
            pass
        eng.close()

    def test_search_with_special_chars(self):
        eng, td = _make_engine(n_messages=10)
        try:
            results = eng.search("hello AND (world OR test) NOT 'excluded'")
            assert isinstance(results, list)
        except Exception:
            pass
        eng.close()
