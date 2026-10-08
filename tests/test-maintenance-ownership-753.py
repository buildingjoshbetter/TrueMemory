"""Real process/file ownership tests; work and databases are synthetic."""

import os
import queue
import runpy
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch


# Reuse the explicitly synthetic loader, without importing the application or
# any native/model package. This test file also serves as the child executable.
_loader = runpy.run_path(str(Path(__file__).with_name("test-maintenance-source-revision-753.py")))
STORAGE, MAINTENANCE = _loader["STORAGE"], _loader["MAINTENANCE"]


class TestMaintenanceOwnership(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-owner-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.sqlite"
        self.conn = STORAGE.create_db(self.path)
        self.addCleanup(self.conn.close)

    def child(self) -> subprocess.Popen:
        process = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--synthetic-owner", str(self.path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        def stop() -> None:
            if process.poll() is None:
                process.terminate()
            try:
                process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=3)

        self.addCleanup(stop)
        return process

    def line(self, process: subprocess.Popen) -> str:
        received = queue.Queue()
        reader = threading.Thread(target=lambda: received.put(process.stdout.readline()), daemon=True)
        reader.start()
        return received.get(timeout=5).strip()

    def blocked_work(self, coordinator) -> tuple[threading.Event, threading.Event, list]:
        entered = threading.Event()
        release = threading.Event()
        connections = []

        def work(conn: sqlite3.Connection, cancel: threading.Event) -> None:
            connections.append(conn)
            entered.set()
            release.wait(3)

        self.assertTrue(coordinator.request(work))
        self.addCleanup(coordinator.wait, 4)
        self.addCleanup(release.set)
        self.assertTrue(entered.wait(2))
        return entered, release, connections

    def test_canonical_paths_share_coordinator_and_one_worker(self) -> None:
        first = MAINTENANCE.get_coordinator(self.path)
        second = MAINTENANCE.get_coordinator(self.path.parent / "." / self.path.name)
        self.assertIs(first, second)
        _, release, _ = self.blocked_work(first)
        self.assertFalse(second.request(lambda conn, cancel: self.fail("duplicate worker")))
        self.assertEqual(self.line(self.child()), "BUSY")
        release.set()
        self.assertTrue(first.wait(2))
        self.assertEqual(first.status, ("success", None))

    def test_symlink_alias_resolves_to_same_database_owner(self) -> None:
        alias = self.path.parent / "synthetic-alias.sqlite"
        try:
            alias.symlink_to(self.path)
        except OSError:
            self.skipTest("Creating file symlinks is unavailable on this test host")
        self.assertIs(MAINTENANCE.get_coordinator(alias), MAINTENANCE.get_coordinator(self.path))

    def test_process_owner_crash_releases_actual_lock(self) -> None:
        first = self.child()
        self.assertEqual(self.line(first), "READY")
        self.assertEqual(self.line(self.child()), "BUSY")
        first.kill()
        first.wait(timeout=3)
        replacement = self.child()
        self.assertEqual(self.line(replacement), "READY")
        replacement.stdin.write("release\n")
        replacement.stdin.flush()
        self.assertEqual(self.line(replacement), "DONE")

    def test_cancellation_keeps_ownership_until_work_returns(self) -> None:
        coordinator = MAINTENANCE.get_coordinator(self.path)
        _, release, _ = self.blocked_work(coordinator)
        coordinator.cancel()
        self.assertFalse(coordinator.wait(0))
        self.assertEqual(self.line(self.child()), "BUSY")
        release.set()
        self.assertTrue(coordinator.wait(2))
        self.assertEqual(coordinator.status, ("cancelled", None))
        self.assertEqual(self.line(self.child()), "READY")

    def test_worker_owns_connection_and_foreground_close_cannot_close_it(self) -> None:
        coordinator = MAINTENANCE.get_coordinator(self.path)
        _, release, connections = self.blocked_work(coordinator)
        self.assertIsNot(connections[0], self.conn)
        self.conn.close()
        self.assertEqual(connections[0].execute("SELECT count(*) FROM messages").fetchone(), (0,))
        release.set()
        self.assertTrue(coordinator.wait(2))
        with self.assertRaises(sqlite3.ProgrammingError):
            connections[0].execute("SELECT 1")

    def test_owner_is_retained_through_connection_close(self) -> None:
        coordinator = MAINTENANCE.get_coordinator(self.path)
        entered = threading.Event()
        release = threading.Event()
        create = MAINTENANCE.create_db

        class PausedClose:
            def __init__(self, conn: sqlite3.Connection) -> None:
                self.conn = conn

            def __getattr__(self, name: str) -> object:
                return getattr(self.conn, name)

            def close(self) -> None:
                entered.set()
                release.wait(3)
                self.conn.close()

        with patch.object(MAINTENANCE, "create_db", side_effect=lambda path: PausedClose(create(path))):
            self.assertTrue(coordinator.request(lambda conn, cancel: None))
            self.addCleanup(coordinator.wait, 4)
            self.addCleanup(release.set)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.line(self.child()), "BUSY")
            self.assertFalse(coordinator.wait(0))
            release.set()
            self.assertTrue(coordinator.wait(2))

    def test_thread_start_failure_releases_owner_for_next_request(self) -> None:
        coordinator = MAINTENANCE.get_coordinator(self.path)
        with patch.object(MAINTENANCE.threading.Thread, "start", side_effect=RuntimeError("synthetic start failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic start failure"):
                coordinator.request(lambda conn, cancel: None)
        self.assertTrue(coordinator.wait(0))
        self.assertEqual(coordinator.status, ("failed", "worker_start"))
        self.assertTrue(coordinator.request(lambda conn, cancel: None))
        self.assertTrue(coordinator.wait(2))
        self.assertEqual(coordinator.status, ("success", None))

    def test_start_failure_after_launch_cannot_release_active_worker_ownership(self) -> None:
        coordinator = MAINTENANCE.get_coordinator(self.path)
        entered = threading.Event()
        release = threading.Event()
        start = threading.Thread.start

        def work(conn: sqlite3.Connection, cancel: threading.Event) -> None:
            entered.set()
            release.wait(3)

        def launch_then_fail(worker: threading.Thread) -> None:
            start(worker)
            if not entered.wait(2):
                raise AssertionError("Synthetic worker did not start")
            raise RuntimeError("synthetic failure after launch")

        self.addCleanup(coordinator.wait, 4)
        self.addCleanup(release.set)
        with patch.object(MAINTENANCE.threading.Thread, "start", launch_then_fail):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure after launch"):
                coordinator.request(work)
        self.assertFalse(coordinator.wait(0))
        self.assertEqual(self.line(self.child()), "BUSY")
        release.set()
        self.assertTrue(coordinator.wait(2))
        self.assertEqual(coordinator.status, ("cancelled", None))

    def test_binding_failure_after_start_releases_claim(self) -> None:
        coordinator = MAINTENANCE.get_coordinator(self.path)
        with patch.object(MAINTENANCE.uuid, "uuid4", side_effect=OSError("synthetic identity failure")):
            self.assertTrue(coordinator.request(lambda conn, cancel: None))
            self.assertTrue(coordinator.wait(2))
        self.assertEqual(coordinator.status, ("failed", "OSError"))
        self.assertEqual(self.line(self.child()), "READY")

    def test_open_failure_and_callback_exception_release_owner_without_error_text(self) -> None:
        coordinator = MAINTENANCE.get_coordinator(self.path)
        with patch.object(MAINTENANCE, "create_db", side_effect=sqlite3.OperationalError("synthetic private sentinel")):
            self.assertTrue(coordinator.request(lambda conn, cancel: None))
            self.assertTrue(coordinator.wait(2))
        self.assertEqual(coordinator.status, ("failed", "OperationalError"))

        def fail(conn: sqlite3.Connection, cancel: threading.Event) -> None:
            conn.execute("INSERT INTO messages(content) VALUES ('synthetic rollback')")
            raise RuntimeError("synthetic private sentinel")

        self.assertTrue(coordinator.request(fail))
        self.assertTrue(coordinator.wait(2))
        self.assertEqual(coordinator.status, ("failed", "RuntimeError"))
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone(), (0,))
        self.assertEqual(self.line(self.child()), "READY")

    def test_unfinished_worker_transaction_is_rolled_back_and_reported_failed(self) -> None:
        coordinator = MAINTENANCE.get_coordinator(self.path)
        self.assertTrue(coordinator.request(lambda conn, cancel: conn.execute("INSERT INTO messages(content) VALUES ('synthetic uncommitted')")))
        self.assertTrue(coordinator.wait(2))
        self.assertEqual(coordinator.status, ("failed", "RuntimeError"))
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone(), (0,))

    def test_in_memory_request_is_pending_without_thread_or_file(self) -> None:
        coordinator = MAINTENANCE.get_coordinator(":memory:")
        with patch.object(MAINTENANCE, "try_file_lock") as file_lock, patch.object(MAINTENANCE, "create_db") as create:
            self.assertFalse(coordinator.request(lambda conn, cancel: self.fail("in-memory worker")))
        self.assertEqual(coordinator.status, ("pending_in_memory", None))
        self.assertTrue(coordinator.wait(0))
        file_lock.assert_not_called()
        create.assert_not_called()

    def test_synchronous_ownership_nests_without_releasing_outer_claim(self) -> None:
        with MAINTENANCE.maintenance_owner(self.path) as outer:
            with MAINTENANCE.maintenance_owner(self.path) as inner:
                self.assertIs(inner, outer)
            self.assertEqual(self.line(self.child()), "BUSY")
            self.assertFalse(MAINTENANCE.get_coordinator(self.path).request(lambda conn, cancel: None))
        self.assertEqual(self.line(self.child()), "READY")

    def test_worker_can_call_synchronous_builder_without_reacquiring_ownership(self) -> None:
        coordinator = MAINTENANCE.get_coordinator(self.path)
        tokens = []

        def work(conn: sqlite3.Connection, cancel: threading.Event) -> None:
            path = MAINTENANCE.connection_database_path(conn)
            self.assertEqual(path, self.path.resolve())
            with MAINTENANCE.maintenance_owner(path) as first:
                with MAINTENANCE.maintenance_owner(path) as second:
                    tokens.extend([first, second])
                self.assertEqual(self.line(self.child()), "BUSY")

        self.assertTrue(coordinator.request(work))
        self.assertTrue(coordinator.wait(3))
        self.assertEqual(coordinator.status, ("success", None))
        self.assertEqual(len(tokens), 2)
        self.assertIs(tokens[0], tokens[1])

    def test_sync_owner_excludes_other_threads_and_releases_after_exception(self) -> None:
        errors = []

        def contender() -> None:
            try:
                with MAINTENANCE.maintenance_owner(self.path):
                    errors.append("incorrectly acquired")
            except MAINTENANCE.MaintenanceBusyError:
                errors.append("busy")

        with self.assertRaisesRegex(ValueError, "synthetic failure"):
            with MAINTENANCE.maintenance_owner(self.path):
                worker = threading.Thread(target=contender)
                worker.start()
                worker.join(2)
                self.assertEqual(errors, ["busy"])
                raise ValueError("synthetic failure")
        with MAINTENANCE.maintenance_owner(self.path):
            pass

    def test_in_memory_sync_owner_uses_original_connection_without_new_file(self) -> None:
        conn = STORAGE.create_db(":memory:")
        self.addCleanup(conn.close)
        conn.execute("INSERT INTO messages(content) VALUES ('synthetic pending')")
        with patch.object(MAINTENANCE, "try_file_lock") as file_lock:
            path = MAINTENANCE.connection_database_path(conn)
            self.assertIsNone(path)
            with MAINTENANCE.maintenance_owner(path) as token:
                self.assertIsNone(token.path)
                self.assertTrue(conn.in_transaction)
        file_lock.assert_not_called()
        conn.rollback()

    @unittest.skipUnless(hasattr(os, "fork"), "Fork inheritance applies only on POSIX")
    def test_fork_child_exiting_sync_context_does_not_close_reused_descriptor(self) -> None:
        with MAINTENANCE.maintenance_owner(self.path):
            owner_fd = next(iter(MAINTENANCE._held_fds))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="This process .* is multi-threaded", category=DeprecationWarning)
                pid = os.fork()
            if pid == 0:
                replacement = os.open(os.devnull, os.O_RDWR)
                if replacement != owner_fd:
                    os.dup2(replacement, owner_fd)
                    os.close(replacement)
            else:
                _, status = os.waitpid(pid, 0)
        if pid == 0:
            try:
                os.fstat(owner_fd)
            except OSError:
                os._exit(2)
            os._exit(0)
        self.assertEqual(status, 0)

    @unittest.skipUnless(hasattr(os, "fork"), "Fork inheritance applies only on POSIX")
    def test_fork_child_does_not_retain_parent_owner_descriptor(self) -> None:
        coordinator = MAINTENANCE.get_coordinator(self.path)
        _, release, _ = self.blocked_work(coordinator)
        read_fd, write_fd = os.pipe()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="This process .* is multi-threaded", category=DeprecationWarning)
            pid = os.fork()
        if pid == 0:
            os.close(write_fd)
            try:
                try:
                    coordinator.status
                except RuntimeError:
                    pass
                else:
                    os._exit(2)
                os.read(read_fd, 1)
            finally:
                os._exit(0)
        os.close(read_fd)
        try:
            release.set()
            self.assertTrue(coordinator.wait(2))
            self.assertEqual(self.line(self.child()), "READY")
        finally:
            os.write(write_fd, b"x")
            os.close(write_fd)
            _, status = os.waitpid(pid, 0)
        self.assertEqual(status, 0)


def child_main(path: str) -> None:
    coordinator = MAINTENANCE.get_coordinator(path)

    def work(conn: sqlite3.Connection, cancel: threading.Event) -> None:
        print("READY", flush=True)
        sys.stdin.readline()

    if not coordinator.request(work):
        print("BUSY", flush=True)
        return
    coordinator.wait()
    print("DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--synthetic-owner":
        child_main(sys.argv[2])
    else:
        unittest.main()
