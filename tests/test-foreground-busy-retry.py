"""Foreground writes retry a writer still busy after the busy timeout (round v8)."""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from truememory.tier_switch import writer
from truememory.tier_switch.writer import writer_transaction


class ForegroundBusyRetry(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="synthetic-busy-retry-")
        self.addCleanup(directory.cleanup)
        self.path = str(Path(directory.name) / "busy.db")
        conn = self.connect()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT)")
        conn.close()

    def connect(self, timeout: float = 0.1) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=timeout, isolation_level=None, check_same_thread=False)
        self.addCleanup(conn.close)
        return conn

    def hold_writer(self, seconds: float) -> threading.Thread:
        holder = self.connect()
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT INTO messages(content) VALUES ('synthetic holder')")
        thread = threading.Timer(seconds, holder.commit)
        thread.start()
        self.addCleanup(thread.join)
        return thread

    def test_a_writer_busy_past_the_busy_timeout_is_retried_until_it_frees(self) -> None:
        conn = self.connect()
        self.hold_writer(1.0)
        started = time.monotonic()
        writer.begin_immediate(conn, retry_seconds=10)
        waited = time.monotonic() - started
        conn.execute("INSERT INTO messages(content) VALUES ('synthetic foreground')")
        conn.commit()
        self.assertGreater(waited, 0.5)
        self.assertLess(waited, 5)
        self.assertEqual(conn.execute("PRAGMA busy_timeout").fetchone()[0], 100)
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)

    def test_retries_stop_at_their_budget_and_raise_with_nothing_held(self) -> None:
        conn = self.connect()
        self.hold_writer(3.0)
        started = time.monotonic()
        with self.assertRaises(sqlite3.OperationalError) as caught:
            writer.begin_immediate(conn, retry_seconds=0.5)
        self.assertTrue(writer._busy_error(caught.exception))
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertFalse(conn.in_transaction)
        self.assertEqual(conn.execute("PRAGMA busy_timeout").fetchone()[0], 100)

    def test_no_retry_without_a_budget_or_for_other_errors(self) -> None:
        conn = self.connect()
        self.hold_writer(1.0)
        started = time.monotonic()
        with self.assertRaises(sqlite3.OperationalError):
            writer.begin_immediate(conn)
        self.assertLess(time.monotonic() - started, 0.5)
        other = self.connect()
        other.execute("BEGIN")
        started = time.monotonic()
        with self.assertRaises(sqlite3.OperationalError):  # Inside a transaction: never retried.
            writer.begin_immediate(other, retry_seconds=10)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(other.in_transaction)
        other.rollback()
        calls = []

        class Failing:
            in_transaction = False

            def execute(self, sql: str) -> None:
                calls.append(sql)
                raise sqlite3.OperationalError("disk I/O error")

        with self.assertRaises(sqlite3.OperationalError):  # Not a busy writer: never retried.
            writer.begin_immediate(Failing(), retry_seconds=10)
        self.assertEqual(calls, ["BEGIN IMMEDIATE"])

    def test_each_backoff_runs_outside_a_transaction_with_the_busy_timeout_restored(self) -> None:
        # Review v7 L-v7-1: the backoff is where the engine lets go of its locks,
        # so other threads must find the connection exactly as they left it.
        conn = self.connect()
        self.hold_writer(1.0)
        seen = []

        @contextmanager
        def backoff():
            seen.append((conn.in_transaction, conn.execute("PRAGMA busy_timeout").fetchone()[0]))
            yield

        writer.begin_immediate(conn, retry_seconds=10, backoff=backoff)
        conn.rollback()
        self.assertGreater(len(seen), 1)
        self.assertEqual(set(seen), {(False, 100)})

    def test_a_transaction_or_close_during_the_backoff_is_never_joined(self) -> None:
        conn = self.connect()
        self.hold_writer(1.0)

        @contextmanager
        def foreign_transaction():
            conn.execute("BEGIN")  # Someone else's, opened while the locks were free.
            yield

        started = time.monotonic()
        with self.assertRaises(sqlite3.OperationalError) as caught:
            writer.begin_immediate(conn, retry_seconds=10, backoff=foreign_transaction)
        self.assertTrue(writer._busy_error(caught.exception))
        self.assertLess(time.monotonic() - started, 0.9)
        self.assertTrue(conn.in_transaction)  # Left to its owner.
        conn.rollback()
        self.assertEqual(conn.execute("PRAGMA busy_timeout").fetchone()[0], 100)

        closing = self.connect()

        @contextmanager
        def closed():
            closing.close()
            yield

        with self.assertRaises(sqlite3.ProgrammingError):
            writer.begin_immediate(closing, retry_seconds=10, backoff=closed)

    def test_the_winning_attempt_checks_a_selection_committed_during_the_backoff(self) -> None:
        conn = self.connect()
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
        capture = writer.capture_writer_selection(conn)
        holder = self.connect()
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT INTO metadata(key, value) VALUES ('tier_selected_v1', 'synthetic')")
        timer = threading.Timer(1.0, holder.commit)
        timer.start()
        self.addCleanup(timer.join)
        with self.assertRaises(writer.WriterSelectionChanged):
            with writer_transaction(conn, capture, retry_busy=True):
                conn.execute("INSERT INTO messages(content) VALUES ('synthetic stale selection')")
        self.assertFalse(conn.in_transaction)
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)

    def test_writer_transaction_retries_only_an_owned_transaction(self) -> None:
        conn = self.connect()
        self.hold_writer(1.0)
        with writer_transaction(conn, None, retry_busy=True):
            conn.execute("INSERT INTO messages(content) VALUES ('synthetic owned')")
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)

        borrowed = self.connect()
        borrowed.execute("BEGIN")
        borrowed.execute("SELECT count(*) FROM messages").fetchone()
        self.hold_writer(1.0)
        started = time.monotonic()
        with patch.object(writer, "FOREGROUND_BUSY_RETRY_SECONDS", 10.0):
            with self.assertRaises(sqlite3.OperationalError):  # The caller's transaction is not ours to wait in.
                with writer_transaction(borrowed, None, retry_busy=True):
                    self.fail("A borrowed writer acquired the lock")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(borrowed.in_transaction)
        borrowed.rollback()


if __name__ == "__main__":
    unittest.main()
