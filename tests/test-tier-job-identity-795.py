"""Synthetic job admission; local tests use mocked files and in-memory SQLite."""

import builtins
import dataclasses
import os
import sqlite3
import stat
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONNECT = sqlite3.connect
SYNTHETIC_PATH = Path(Path.cwd().anchor) / "synthetic" / "tier database?#.db"


def load_modules() -> dict[str, types.ModuleType]:
    modules = {}

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"numpy", "torch", "sentence_transformers", "model2vec", "sqlite_vec"}:
            raise AssertionError("Native/model imports are excluded from job admission")
        if name.startswith("truememory."):
            short = name.removeprefix("truememory.")
            if short in modules:
                return modules[short]
            raise AssertionError("Unexpected application import: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    for name in ("storage", "_platform", "maintenance", "rebuild_source", "embedding_target", "tier_switch.job"):
        module = types.ModuleType("synthetic_tier_job_" + name.replace(".", "_"))
        sys.modules[module.__name__] = module
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
        path = ROOT / "truememory" / (name.replace(".", "/") + ".py")
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
        modules[name] = module
    return modules


class MockFileConnection(sqlite3.Connection):
    closed = False
    fake_path = str(SYNTHETIC_PATH)
    extra_database = False
    rollback_failure = None
    close_failure = None

    def execute(self, sql: str, parameters: tuple = ()):
        if sql == "PRAGMA database_list":
            rows = [(0, "main", self.fake_path)]
            if self.extra_database:
                rows.append((2, "extra", "/synthetic/attached.db"))
            return rows
        return super().execute(sql, parameters)

    def close(self) -> None:
        self.closed = True
        super().close()
        if self.close_failure is not None:
            raise self.close_failure

    def rollback(self) -> None:
        if self.rollback_failure == "retained":
            return
        if self.rollback_failure is not None:
            raise self.rollback_failure
        super().rollback()


class Cancelled(BaseException):
    pass


class TestTierJobIdentity(unittest.TestCase):
    def setUp(self) -> None:
        self.modules = load_modules()
        self.api = self.modules["tier_switch.job"]
        self.source = self.modules["rebuild_source"]
        self.target = self.modules["embedding_target"].EmbeddingTarget("base", "qwen3_256", 256, "basepro")
        self.conn = CONNECT(":memory:", factory=MockFileConnection)
        self.addCleanup(self.conn.close)
        self.conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT)")
        self.source.ensure_rebuild_tracking(self.conn)
        self.identity = (7, 19)
        self.observations = 0
        self.events = []

        @contextmanager
        def observe(path):
            self.assertEqual(path, SYNTHETIC_PATH)
            self.observations += 1
            self.events.append("file")
            yield self.identity

        self.observe = observe
        self.addCleanup(patch.stopall)
        patch.object(self.api, "_observe_file", observe).start()
        patch.object(self.api.Path, "resolve", lambda path, strict=False: path).start()

    def capture(self):
        return self.api.capture_tier_job(self.conn, self.target)

    def reopen_hooks(self, *, before_open=None, opened=None):
        @contextmanager
        def owner(path):
            self.assertEqual(path, SYNTHETIC_PATH)
            self.events.append(("owner", threading.get_ident()))
            try:
                yield object()
            finally:
                self.events.append("release")

        reopened = []

        def connect(database, **kwargs):
            self.events.append("connect")
            self.assertEqual(database, SYNTHETIC_PATH.as_uri() + "?mode=rw")
            self.assertEqual(kwargs, {"uri": True})
            if before_open is not None:
                before_open()
            conn = CONNECT(":memory:", factory=MockFileConnection)
            self.conn.backup(conn)
            reopened.append(conn)
            if opened is not None:
                opened(conn)
            return conn

        patch.object(self.api, "maintenance_owner", owner).start()
        patch.object(self.api.sqlite3, "connect", connect).start()
        return reopened

    def test_capture_is_frozen_read_only_and_retains_no_connection(self) -> None:
        changes = self.conn.total_changes
        job = self.capture()
        self.assertEqual(self.conn.total_changes, changes)
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(job.target, self.target)
        self.assertEqual(job.database_path, SYNTHETIC_PATH)
        self.assertEqual(job.file_identity, (7, 19))
        self.assertEqual(job.tracker, "rebuild-bridge-v1")
        self.assertNotEqual(job.job_id, self.capture().job_id)
        self.assertNotIn(str(SYNTHETIC_PATH), repr(job))
        self.assertEqual({field.name for field in dataclasses.fields(job)}, {
            "job_id", "target", "database_path", "file_identity", "tracker", "source_epoch", "source_schema_signature",
        })
        with self.assertRaises(dataclasses.FrozenInstanceError):
            job.job_id = "changed"

    def test_missing_tracking_never_installs_it(self) -> None:
        conn = CONNECT(":memory:", factory=MockFileConnection)
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT)")
        before = conn.total_changes
        with self.assertRaises(self.api.TierJobIdentityError):
            self.api.capture_tier_job(conn, self.target)
        self.assertEqual(conn.total_changes, before)
        self.assertFalse(conn.in_transaction)
        self.assertEqual(conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("messages",)])

    def test_borrowed_transaction_is_untouched(self) -> None:
        job = self.capture()
        self.conn.execute("BEGIN")
        for call in (lambda: self.capture(), lambda: self.api.check_tier_job(self.conn, job)):
            with self.assertRaises(self.api.TierJobIdentityError):
                call()
            self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()

    def test_memory_and_attachments_fail_before_file_observation(self) -> None:
        for path, attached in (("", False), (str(SYNTHETIC_PATH), True)):
            with self.subTest(path=path, attached=attached):
                self.conn.fake_path = path
                self.conn.extra_database = attached
                with self.assertRaises(self.api.TierJobIdentityError):
                    self.capture()
        self.assertEqual(self.observations, 0)
        with CONNECT(":memory:") as real_memory:
            with self.assertRaises(self.api.TierJobIdentityError):
                self.api.capture_tier_job(real_memory, self.target)
        real_memory.close()

    def test_temp_shadow_and_unrelated_temp_objects_fail_closed(self) -> None:
        for ddl in ("CREATE TEMP TABLE MESSAGES(id)", "CREATE TEMP VIEW unrelated AS SELECT 1"):
            with self.subTest(ddl=ddl):
                conn = CONNECT(":memory:", factory=MockFileConnection)
                try:
                    self.conn.backup(conn)
                    conn.execute(ddl)
                    with self.assertRaises(self.api.TierJobIdentityError):
                        self.api.capture_tier_job(conn, self.target)
                finally:
                    conn.close()
        self.assertEqual(self.observations, 0)

    def test_source_mutations_with_same_epoch_are_adapter_responsibility(self) -> None:
        job = self.capture()
        self.conn.execute("INSERT INTO messages VALUES (1,'synthetic source')")
        self.conn.commit()
        self.api.check_tier_job(self.conn, job)
        self.conn.execute("UPDATE messages SET content='synthetic correction'")
        self.conn.commit()
        self.api.check_tier_job(self.conn, job)
        self.assertFalse(self.conn.in_transaction)

    def test_epoch_schema_tracker_and_trigger_changes_refuse(self) -> None:
        job = self.capture()
        actions = (
            "UPDATE truememory_rebuild_source_v1 SET epoch='replacement'",
            "ALTER TABLE messages ADD COLUMN synthetic TEXT",
            "DROP TRIGGER truememory_rebuild_messages_v1_ai",
        )
        for sql in actions:
            with self.subTest(sql=sql):
                conn = CONNECT(":memory:", factory=MockFileConnection)
                try:
                    self.conn.backup(conn)
                    conn.execute(sql)
                    conn.commit()
                    with self.assertRaises(self.api.TierJobIdentityError):
                        self.api.check_tier_job(conn, job)
                    self.assertFalse(conn.in_transaction)
                finally:
                    conn.close()
        altered = dataclasses.replace(job, tracker="canonical-v1")
        with self.assertRaises(self.api.TierJobIdentityError):
            self.api.check_tier_job(self.conn, altered)

    def test_replaced_file_refuses_before_connect(self) -> None:
        job = self.capture()
        reopened = self.reopen_hooks()
        self.identity = (7, 20)
        with self.assertRaises(self.api.TierJobIdentityError):
            with self.api.open_tier_job(job):
                self.fail("Replacement must not yield")
        self.assertEqual(reopened, [])
        self.assertNotIn("connect", self.events)
        self.assertEqual(self.events[-1], "release")

    def test_change_during_open_closes_before_yield(self) -> None:
        job = self.capture()
        reopened = self.reopen_hooks(before_open=lambda: setattr(self, "identity", (7, 20)))
        with self.assertRaises(self.api.TierJobIdentityError):
            with self.api.open_tier_job(job):
                self.fail("Raced open must not yield")
        self.assertEqual(len(reopened), 1)
        self.assertTrue(reopened[0].closed)
        self.assertEqual(self.events[-1], "release")

    def test_reopened_epoch_and_pragma_path_checked_before_yield(self) -> None:
        job = self.capture()
        for change in ("epoch", "path"):
            def opened(conn):
                if change == "epoch":
                    conn.execute("UPDATE truememory_rebuild_source_v1 SET epoch='replacement'")
                    conn.commit()
                else:
                    conn.fake_path = "/synthetic/other.db"
            reopened = self.reopen_hooks(opened=opened)
            with self.assertRaises(self.api.TierJobIdentityError):
                with self.api.open_tier_job(job):
                    self.fail("Unmatched reopened identity must not yield")
            self.assertTrue(reopened[0].closed)

    def test_reopen_is_rw_only_owner_is_in_calling_worker_thread_and_cleanup_is_exact(self) -> None:
        job = self.capture()
        self.events.clear()
        reopened = self.reopen_hooks()
        sentinel = Cancelled("synthetic cancellation")
        with self.assertRaises(Cancelled) as raised:
            with self.api.open_tier_job(job) as conn:
                self.assertFalse(conn.in_transaction)
                self.api.check_tier_job(conn, job)
                conn.execute("BEGIN")
                conn.execute("INSERT INTO messages VALUES (1,'uncommitted synthetic row')")
                raise sentinel
        self.assertIs(raised.exception, sentinel)
        self.assertEqual(self.events[0], ("owner", threading.get_ident()))
        self.assertTrue(reopened[0].closed)
        self.assertEqual(self.events[-1], "release")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)

    def test_body_oserror_is_not_relabelled_as_identity_failure(self) -> None:
        job = self.capture()
        self.reopen_hooks()
        sentinel = OSError("synthetic caller operation")
        with self.assertRaises(OSError) as raised:
            with self.api.open_tier_job(job):
                raise sentinel
        self.assertIs(raised.exception, sentinel)

    def test_open_and_source_failure_close_and_release(self) -> None:
        job = self.capture()
        self.reopen_hooks()
        for error in (sqlite3.OperationalError("/private/path synthetic"), Cancelled("synthetic cancellation")):
            with patch.object(self.api.sqlite3, "connect", side_effect=error):
                expected = Cancelled if isinstance(error, Cancelled) else self.api.TierJobIdentityError
                with self.assertRaises(expected) as caught:
                    with self.api.open_tier_job(job):
                        self.fail("Failed connect must not yield")
                self.assertEqual(self.events[-1], "release")
                if expected is self.api.TierJobIdentityError:
                    self.assertNotIn("/private", str(caught.exception))
                    self.assertTrue(caught.exception.__suppress_context__)
        reopened = self.reopen_hooks()
        with patch.object(self.api, "read_rebuild_revision", side_effect=Cancelled("synthetic snapshot cancellation")):
            with self.assertRaises(Cancelled):
                with self.api.open_tier_job(job):
                    self.fail("Cancelled source read must not yield")
        self.assertTrue(reopened[0].closed)
        self.assertEqual(self.events[-1], "release")

    def test_snapshot_failure_rolls_back_and_does_not_log_source_error(self) -> None:
        sentinel = sqlite3.OperationalError("synthetic identifying source detail")
        with patch.object(self.api, "read_rebuild_revision", side_effect=sentinel):
            with self.assertRaises(self.api.TierJobIdentityError) as caught:
                self.capture()
        self.assertFalse(self.conn.in_transaction)
        self.assertNotIn("identifying", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_uncertain_snapshot_rollback_propagates_and_owned_reopen_closes(self) -> None:
        job = self.capture()
        for failure in ("retained", sqlite3.OperationalError("synthetic rollback failure")):
            self.conn.rollback_failure = failure
            with self.assertRaises(self.api.TierJobIdentityError):
                self.capture()
            self.assertTrue(self.conn.in_transaction)
            self.conn.rollback_failure = None
            self.conn.rollback()
            reopened = self.reopen_hooks(opened=lambda conn: setattr(conn, "rollback_failure", failure))
            with self.assertRaises(self.api.TierJobIdentityError):
                with self.api.open_tier_job(job):
                    self.fail("Uncertain snapshot cleanup must not yield")
            self.assertTrue(reopened[0].closed)
            self.assertEqual(self.events[-1], "release")

    def test_connection_close_failure_still_releases_owner_and_is_sanitized(self) -> None:
        job = self.capture()
        reopened = self.reopen_hooks(opened=lambda conn: setattr(
            conn, "close_failure", sqlite3.OperationalError("/private/synthetic.db"),
        ))
        with self.assertRaises(self.api.TierJobIdentityError) as caught:
            with self.api.open_tier_job(job):
                pass
        self.assertTrue(reopened[0].closed)
        self.assertEqual(self.events[-1], "release")
        self.assertNotIn("/private", str(caught.exception))

    def test_descriptor_handoff_acquires_owner_only_inside_worker(self) -> None:
        job = self.capture()
        dump = "\n".join(self.conn.iterdump())
        caller_thread = threading.get_ident()
        owners = []
        results = []
        opened = []
        @contextmanager
        def owner(path):
            owners.append(threading.get_ident())
            try:
                yield object()
            finally:
                owners.append("released")
        def connect(database, **kwargs):
            conn = CONNECT(":memory:", factory=MockFileConnection)
            conn.executescript(dump)
            opened.append(conn)
            return conn
        def worker():
            try:
                with self.api.open_tier_job(job) as conn:
                    results.append((threading.get_ident(), conn.in_transaction))
            except BaseException as error:
                results.append(type(error).__name__)
        with patch.object(self.api, "maintenance_owner", owner), patch.object(self.api.sqlite3, "connect", connect):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], (thread.ident, False))
        self.assertEqual(owners, [thread.ident, "released"])
        self.assertNotEqual(thread.ident, caller_thread)
        self.assertTrue(opened[0].closed)

    def test_busy_owner_prevents_all_file_and_database_opening(self) -> None:
        job = self.capture()
        self.events.clear()
        reopened = self.reopen_hooks()
        with patch.object(self.api, "maintenance_owner", side_effect=self.modules["maintenance"].MaintenanceBusyError("synthetic busy")):
            with self.assertRaises(self.modules["maintenance"].MaintenanceBusyError):
                with self.api.open_tier_job(job):
                    self.fail("Busy owner must not yield")
        self.assertEqual(reopened, [])
        self.assertEqual(self.events, [])

    def test_missing_path_is_sanitized_and_no_reopen_occurs(self) -> None:
        job = self.capture()
        reopened = self.reopen_hooks()
        with patch.object(self.api, "_observe_file", side_effect=FileNotFoundError("/private/synthetic.db")):
            with self.assertRaises(self.api.TierJobIdentityError) as caught:
                with self.api.open_tier_job(job):
                    self.fail("Missing database must not yield")
        self.assertEqual(reopened, [])
        self.assertNotIn("/private", str(caught.exception))

    def test_capture_and_final_check_detect_second_observation_change(self) -> None:
        original = self.api.read_rebuild_revision
        def replace_after_read(conn):
            result = original(conn)
            self.identity = (7, 20)
            return result
        job = self.capture()
        for operation in (lambda: self.capture(), lambda: self.api.check_tier_job(self.conn, job)):
            self.identity = (7, 19)
            with patch.object(self.api, "read_rebuild_revision", replace_after_read):
                with self.assertRaises(self.api.TierJobIdentityError):
                    operation()
            self.assertFalse(self.conn.in_transaction)

    def test_same_epoch_stale_connection_cannot_be_disproved_by_path_observations(self) -> None:
        # Explicit limitation: capture assumes this connection still names the
        # current file. A copied epoch plus a stale handle violates that contract.
        self.identity = (7, 99)
        job = self.capture()
        reopened = self.reopen_hooks()
        with self.api.open_tier_job(job) as conn:
            self.api.check_tier_job(conn, job)
        self.assertEqual(job.file_identity, (7, 99))
        self.assertTrue(reopened[0].closed)


class TestFileObservation(unittest.TestCase):
    def test_same_handle_api_ignores_mutable_and_cross_api_ctime(self) -> None:
        api = load_modules()["tier_switch.job"]
        value = types.SimpleNamespace(st_mode=stat.S_IFREG, st_dev=5, st_ino=19, st_ctime_ns=10)
        with patch.object(api.os, "open", return_value=81) as opened, patch.object(api.os, "fstat", return_value=value) as fstat, patch.object(api.os, "close") as closed:
            with api._observe_file(SYNTHETIC_PATH) as first:
                value.st_ctime_ns = 999
            with api._observe_file(SYNTHETIC_PATH) as second:
                self.assertEqual(first, second)
        self.assertEqual(first, (5, 19))
        self.assertEqual(fstat.call_count, 2)
        self.assertEqual(opened.call_args.args, (SYNTHETIC_PATH, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)))
        self.assertEqual(closed.call_count, 2)
        self.assertEqual(closed.call_args.args, (81,))

    def test_unsupported_identity_and_open_errors_fail_closed_without_path(self) -> None:
        api = load_modules()["tier_switch.job"]
        for mode, device, inode in ((stat.S_IFDIR, 5, 19), (stat.S_IFIFO, 5, 19), (stat.S_IFREG, 5, 0), (stat.S_IFREG, -1, 19), (stat.S_IFREG, True, 19)):
            value = types.SimpleNamespace(st_mode=mode, st_dev=device, st_ino=inode)
            with patch.object(api.os, "open", return_value=81), patch.object(api.os, "fstat", return_value=value), patch.object(api.os, "close") as closed:
                with self.assertRaises(api.TierJobIdentityError):
                    with api._observe_file(SYNTHETIC_PATH):
                        self.fail("Unsupported identity must not yield")
            self.assertEqual(closed.call_count, 1)
        with patch.object(api.os, "open", side_effect=OSError("/private/synthetic.db")):
            with self.assertRaises(api.TierJobIdentityError) as caught:
                with api._observe_file(SYNTHETIC_PATH):
                    self.fail("Failed file observation must not yield")
        self.assertNotIn("/private", str(caught.exception))

    def test_observer_preserves_caller_baseexception_and_oserror(self) -> None:
        api = load_modules()["tier_switch.job"]
        value = types.SimpleNamespace(st_mode=stat.S_IFREG, st_dev=5, st_ino=19)
        for sentinel in (Cancelled("synthetic"), OSError("synthetic"), ValueError("synthetic")):
            with patch.object(api.os, "open", return_value=81), patch.object(api.os, "fstat", return_value=value), patch.object(api.os, "close") as closed:
                with self.assertRaises(type(sentinel)) as raised:
                    with api._observe_file(SYNTHETIC_PATH):
                        raise sentinel
                self.assertIs(raised.exception, sentinel)
                self.assertEqual(closed.call_count, 1)

    def test_fstat_cancellation_and_failure_close_the_readonly_descriptor(self) -> None:
        api = load_modules()["tier_switch.job"]
        for sentinel in (Cancelled("synthetic"), OSError("synthetic")):
            with patch.object(api.os, "open", return_value=81), patch.object(api.os, "fstat", side_effect=sentinel), patch.object(api.os, "close") as closed:
                expected = Cancelled if isinstance(sentinel, Cancelled) else api.TierJobIdentityError
                with self.assertRaises(expected):
                    with api._observe_file(SYNTHETIC_PATH):
                        self.fail("Failed observation must not yield")
                self.assertEqual(closed.call_args.args, (81,))


@unittest.skipUnless(os.environ.get("TRUEMEMORY_TIER_JOB_FILE_TEST") == "1", "opt-in synthetic file DB gate only")
class TestTierJobFileDatabase(unittest.TestCase):
    def test_existing_database_reopen_replacement_missing_and_worker_ownership(self) -> None:
        import tempfile

        modules = load_modules()
        api = modules["tier_switch.job"]
        target = modules["embedding_target"].EmbeddingTarget("base", "qwen3_256", 256, "basepro")
        with tempfile.TemporaryDirectory(prefix="truememory-synthetic-job-") as directory:
            path = Path(directory) / "synthetic.db"
            conn = CONNECT(path)
            try:
                conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT)")
                modules["rebuild_source"].ensure_rebuild_tracking(conn)
                job = api.capture_tier_job(conn, target)
            finally:
                conn.close()
            outcomes = []
            def worker():
                with api.open_tier_job(job) as opened:
                    api.check_tier_job(opened, job)
                    outcomes.append(not opened.in_transaction)
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(outcomes, [True])
            archived = path.with_suffix(".old")
            path.rename(archived)
            with self.assertRaises(api.TierJobIdentityError):
                with api.open_tier_job(job):
                    self.fail("Missing file must not yield")
            self.assertFalse(path.exists())
            replacement = CONNECT(path)
            replacement.close()
            with self.assertRaises(api.TierJobIdentityError):
                with api.open_tier_job(job):
                    self.fail("Replaced file must not yield")


if __name__ == "__main__":
    unittest.main()
