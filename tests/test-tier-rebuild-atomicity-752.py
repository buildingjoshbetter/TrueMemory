"""Paired tier-rebuild durability with synthetic models and real SQLite/vec0."""
from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
VEC = "vec_messages_edge"
SEP = "vec_messages_sep_edge"


def load_worker() -> tuple[types.ModuleType, types.ModuleType]:
    modules = []
    for name in ("cache", "worker"):
        path = ROOT / "truememory/tier_switch" / (name + ".py")
        tree = ast.parse(path.read_text(), filename=str(path))
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and (node.module or "").startswith("truememory.")
        ) and not (isinstance(node, ast.Import) and any(item.name == "psutil" for item in node.names))]
        module = types.ModuleType("synthetic_rebuild_" + name)
        if name == "worker":
            module.VectorCacheRegistry = modules[0].VectorCacheRegistry
            module.DynamicThrottler = FakeThrottler
        exec(compile(tree, str(path), "exec"), module.__dict__)
        modules.append(module)
    return modules[1], modules[0]


class FakeThrottler:
    batch_size = 2

    def __init__(self) -> None:
        self.ooms = 0
        self.completed = []

    def before_batch(self) -> tuple[int, dict]:
        return self.batch_size, {}

    def after_batch(self, count: int, duration: float) -> None:
        self.completed.append(count)

    def on_oom(self) -> None:
        self.ooms += 1
        if self.ooms > 1:
            raise AssertionError("synthetic OOM retry did not recover")
        self.batch_size = 1

    def should_flush_cache(self) -> bool:
        return False

    @staticmethod
    def flush_gpu_cache() -> None:
        pass

    def get_eta_seconds(self, remaining: int) -> float:
        return float(remaining)

    def get_throughput(self) -> float:
        return 1.0


def serialize(vector: list[float]) -> bytes:
    return struct.pack("256f", *vector)


class WorkerAtomicity(unittest.TestCase):
    use_vec = False

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="synthetic-tier-752-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "synthetic.sqlite"
        self.conn = self.open_connection()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
            CREATE TABLE notes(value TEXT);
            CREATE TABLE vector_cache_registry (
                tier_group TEXT PRIMARY KEY, last_embedded_id INTEGER,
                vector_count INTEGER, last_updated REAL
            );
            INSERT INTO vector_cache_registry VALUES ('edge',0,0,0);
            CREATE TABLE rebuild_status (
                id INTEGER PRIMARY KEY, status TEXT, processed_messages INTEGER,
                progress_pct REAL, eta_seconds REAL, batch_size INTEGER,
                throughput_ips REAL, ram_pct REAL, last_heartbeat REAL,
                completed_at REAL, error TEXT
            );
            INSERT INTO rebuild_status(id,status,processed_messages) VALUES (1,'running',0);
        """)
        for table in (VEC, SEP):
            if self.use_vec:
                self.conn.execute(f"CREATE VIRTUAL TABLE {table} USING vec0(embedding float[256] distance_metric=cosine)")
            else:
                self.conn.execute(f"CREATE TABLE {table}(rowid INTEGER PRIMARY KEY,embedding BLOB)")
        self.conn.commit()
        self.module, self.cache = load_worker()
        self.throttler = FakeThrottler()
        self.worker = self.module.RebuildWorker(self.conn, "edge", "edge", self.throttler, status_id=1)
        self.messages = [{"id": index, "content": f"synthetic record {index}",
                          "sender": "synthetic sender", "recipient": "synthetic recipient",
                          "timestamp": "2026-01-01"} for index in range(1, 5)]
        self.calls = []
        self.encode_hook = None
        self.serialize_hook = None

        def encode(texts: list[str], **kwargs: object) -> list[list[float]]:
            self.calls.append({"texts": list(texts), "in_transaction": self.conn.in_transaction})
            if self.encode_hook is not None:
                result = self.encode_hook(len(self.calls), texts)
                if result is not None:
                    return result
            return [[float(index + 1)] + [0.0] * 255 for index in range(len(texts))]

        self.model = types.SimpleNamespace(encode=encode)
        vector = types.ModuleType("truememory.vector_search")
        vector.get_model = lambda: self.model
        vector.set_embedding_model = lambda tier: None
        vector.init_vec_table = lambda conn, tier_group: None
        vector._build_sep_text = lambda sender, recipient, timestamp, content: "separation " + content

        def selected_serialize(value: list[float]) -> bytes:
            if self.serialize_hook is not None:
                self.serialize_hook(value)
            return serialize(value)

        vector.serialize_f32 = selected_serialize
        self.vector = vector
        ownership = types.ModuleType("truememory.mps_utils")
        ownership.encode_with_model_ownership = lambda model, texts, **kwargs: model.encode(texts, **kwargs)
        torch = types.ModuleType("torch")
        torch.no_grad = nullcontext
        psutil = types.ModuleType("psutil")
        psutil.virtual_memory = lambda: types.SimpleNamespace(percent=10.0)
        self.dependencies = patch.dict(sys.modules, {
            "truememory.vector_search": vector, "truememory.mps_utils": ownership,
            "torch": torch, "psutil": psutil,
        })
        self.dependencies.start()
        self.addCleanup(self.dependencies.stop)

    def open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=0.5)
        self.addCleanup(conn.close)
        if self.use_vec:
            try:
                import sqlite_vec
            except ImportError:
                self.skipTest("sqlite-vec package is unavailable")
            if not hasattr(conn, "enable_load_extension"):
                self.skipTest("SQLite cannot load extensions")
            conn.enable_load_extension(True)
            try:
                sqlite_vec.load(conn)
            finally:
                conn.enable_load_extension(False)
        return conn

    def ids(self, conn: sqlite3.Connection | None = None) -> tuple[list[int], list[int]]:
        connection = conn or self.conn
        return tuple([row[0] for row in connection.execute(f"SELECT rowid FROM {table} ORDER BY rowid")]
                     for table in (VEC, SEP))

    def checkpoint(self, conn: sqlite3.Connection | None = None) -> tuple[int, int] | None:
        return (conn or self.conn).execute(
            "SELECT last_embedded_id,vector_count FROM vector_cache_registry WHERE tier_group='edge'"
        ).fetchone()

    def seed_prior(self) -> None:
        for table in (VEC, SEP):
            self.conn.execute(f"INSERT INTO {table}(rowid,embedding) VALUES (?,?)", (1, serialize([1.0] + [0.0] * 255)))
        self.conn.execute("UPDATE vector_cache_registry SET last_embedded_id=1,vector_count=1")
        self.conn.commit()

    def process(self, messages: list[dict] | None = None) -> bool:
        batch = messages or self.messages[:2]
        return self.worker._process_batch(
            batch, self.model, VEC, SEP, self.vector.serialize_f32, self.vector._build_sep_text,
            record_progress=lambda: self.cache.VectorCacheRegistry.update_progress(
                self.worker.conn, "edge", batch[-1]["id"], self.worker._count_vectors(VEC), commit=False),
        )

    def fail_sql_once(self, stage: str, error: BaseException) -> list[str]:
        connection = self.conn
        failures = []

        class FaultConnection:
            def __getattr__(self, name: str) -> object:
                return getattr(connection, name)

            def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                prefixes = {"checkpoint": "UPDATE vector_cache_registry", "commit": "COMMIT",
                            "clear_separation": f"DELETE FROM {SEP}", "status": "UPDATE rebuild_status"}
                if stage in prefixes and sql.startswith(prefixes[stage]) and not failures:
                    failures.append(stage)
                    if stage != "commit":
                        connection.execute(sql, *args)
                    raise error
                return connection.execute(sql, *args)

            def executemany(self, sql: str, values: list[tuple]) -> sqlite3.Cursor:
                table = VEC if stage == "completion" else SEP if stage == "separation" else None
                if table is not None and sql.startswith(f"INSERT INTO {table}(") and not failures:
                    failures.append(stage)
                    connection.execute(sql, values[0])
                    raise error
                return connection.executemany(sql, values)

        self.worker.conn = FaultConnection()
        return failures

    def assert_complete(self, expected: list[int]) -> None:
        self.assertEqual(self.ids(), (expected, expected))
        self.assertEqual(self.checkpoint(), (expected[-1], len(expected)))
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT status FROM rebuild_status").fetchone()[0], "complete")

    def test_oom_at_either_encode_retries_cleanly_without_writer_lock(self) -> None:
        for failing_call in (1, 2):
            with self.subTest(failing_call=failing_call):
                def fail(call: int, texts: list[str]) -> None:
                    if call == failing_call:
                        raise RuntimeError("synthetic out of memory")
                self.encode_hook = fail
                self.assertEqual(self.worker.run(self.messages[:2], True), (True, 2))
                self.assert_complete([1, 2])
                self.assertEqual(self.throttler.ooms, 1)
                self.assertTrue(all(not call["in_transaction"] for call in self.calls))
                self.calls.clear()
                self.throttler.ooms = 0
                self.throttler.batch_size = 2

    def test_oom_at_each_write_or_commit_retries_pair_and_checkpoint(self) -> None:
        for stage in ("completion", "separation", "checkpoint", "commit"):
            with self.subTest(stage=stage):
                failures = self.fail_sql_once(stage, RuntimeError("synthetic out of memory"))
                self.assertEqual(self.worker.run(self.messages[:2], False), (True, 2))
                self.assertEqual(failures, [stage])
                self.assert_complete([1, 2])
                self.assertEqual(self.throttler.ooms, 1)
                self.worker.conn = self.conn
                with self.conn:
                    self.conn.execute(f"DELETE FROM {VEC}")
                    self.conn.execute(f"DELETE FROM {SEP}")
                    self.conn.execute("UPDATE vector_cache_registry SET last_embedded_id=0,vector_count=0")
                self.throttler.ooms = 0
                self.throttler.batch_size = 2

    def test_non_oom_partial_insert_failure_status_cannot_publish_half_pair(self) -> None:
        self.seed_prior()
        self.fail_sql_once("separation", sqlite3.IntegrityError("synthetic insertion failure"))
        self.assertEqual(self.worker.run(self.messages[1:3], False), (False, 0))
        self.conn.commit()
        other = self.open_connection()
        self.assertEqual(self.ids(other), ([1], [1]))
        self.assertEqual(self.checkpoint(other), (1, 1))
        self.assertEqual(other.execute("SELECT status,processed_messages FROM rebuild_status").fetchone(), ("failed", 0))

    def test_cancellation_during_either_insert_cannot_escape_through_status_commit(self) -> None:
        self.seed_prior()
        for stage in ("completion", "separation", "checkpoint", "commit"):
            with self.subTest(stage=stage):
                self.fail_sql_once(stage, KeyboardInterrupt("synthetic cancellation"))
                with self.assertRaisesRegex(KeyboardInterrupt, "synthetic cancellation"):
                    self.process(self.messages[1:3])
                self.worker._update_status("cancelled", 0, 2)
                self.conn.commit()
                self.assertEqual(self.ids(), ([1], [1]))
                self.assertEqual(self.checkpoint(), (1, 1))
                self.assertFalse(self.conn.in_transaction)
                self.worker.conn = self.conn

    def test_cancellation_between_encodes_leaves_no_partial_work(self) -> None:
        def cancel(call: int, texts: list[str]) -> None:
            if call == 2:
                raise KeyboardInterrupt("synthetic encode cancellation")
        self.encode_hook = cancel
        with self.assertRaises(KeyboardInterrupt):
            self.process()
        self.worker._update_status("cancelled", 0, 2)
        self.assertEqual(self.ids(), ([], []))
        self.assertEqual(self.checkpoint(), (0, 0))

    def test_cancel_flag_finishes_current_pair_and_reports_committed_progress(self) -> None:
        def cancel(call: int, texts: list[str]) -> None:
            if call == 2:
                self.worker.cancel()
        self.encode_hook = cancel
        self.assertEqual(self.worker.run(self.messages, False), (False, 2))
        self.assertEqual(self.ids(), ([1, 2], [1, 2]))
        self.assertEqual(self.checkpoint(), (2, 2))
        self.assertEqual(self.conn.execute("SELECT status,processed_messages FROM rebuild_status").fetchone(), ("cancelled", 2))

    def test_full_rebuild_clear_failure_restores_pair_and_checkpoint(self) -> None:
        self.seed_prior()
        self.fail_sql_once("clear_separation", sqlite3.OperationalError("synthetic clear failure"))
        self.assertEqual(self.worker.run(self.messages, True), (False, 0))
        self.conn.commit()
        self.assertEqual(self.ids(), ([1], [1]))
        self.assertEqual(self.checkpoint(), (1, 1))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.conn.execute("SELECT status FROM rebuild_status").fetchone()[0], "failed")

    def test_status_failure_after_checkpoint_does_not_retry_the_batch(self) -> None:
        self.fail_sql_once("status", sqlite3.OperationalError("synthetic status failure"))
        self.assertEqual(self.worker.run(self.messages[:2], False), (True, 2))
        self.assert_complete([1, 2])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.throttler.ooms, 0)

    def test_real_commit_denial_cannot_be_published_by_failure_status(self) -> None:
        denied = []

        def deny_commit(action: int, operation: str | None, *unused: str | None) -> int:
            if action == sqlite3.SQLITE_TRANSACTION and operation == "COMMIT" and not denied:
                denied.append(True)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        self.conn.set_authorizer(deny_commit)
        try:
            self.assertEqual(self.worker.run(self.messages[:2], False), (False, 0))
        finally:
            # Disabling with None is supported only on Python 3.11+.
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertEqual(denied, [True])
        self.conn.commit()
        other = self.open_connection()
        self.assertEqual(self.ids(other), ([], []))
        self.assertEqual(self.checkpoint(other), (0, 0))
        self.assertEqual(other.execute("SELECT status FROM rebuild_status").fetchone()[0], "failed")

    def test_malformed_output_counts_fail_before_any_vector_write(self) -> None:
        for stage in (1, 2):
            with self.subTest(stage=stage):
                self.encode_hook = lambda call, texts: [] if call == stage else None
                self.assertEqual(self.worker.run(self.messages[:2], False), (False, 0))
                self.assertEqual(self.ids(), ([], []))
                self.assertEqual(self.checkpoint(), (0, 0))
                self.assertFalse(self.conn.in_transaction)
                self.calls.clear()

    def test_serialization_failure_precedes_writer_and_leaves_no_rows(self) -> None:
        for target in (2, 4):
            with self.subTest(target=target):
                serializations = []
                def fail(vector: list[float]) -> None:
                    serializations.append(1)
                    if len(serializations) == target:
                        raise ValueError("synthetic serialization failure")
                self.serialize_hook = fail
                self.assertEqual(self.worker.run(self.messages[:2], False), (False, 0))
                self.assertEqual(self.ids(), ([], []))
                self.assertEqual(self.checkpoint(), (0, 0))
                self.assertTrue(all(not call["in_transaction"] for call in self.calls))

    def test_pair_and_checkpoint_become_visible_in_the_same_commit(self) -> None:
        other = self.open_connection()
        observed = []
        original = self.cache.VectorCacheRegistry.update_progress
        def witness(*args: object, **kwargs: object) -> None:
            original(*args, **kwargs)
            observed.append((self.ids(other), self.checkpoint(other)))
        with patch.object(self.cache.VectorCacheRegistry, "update_progress", side_effect=witness):
            self.assertTrue(self.process())
        self.assertEqual(observed, [(([], []), (0, 0))])
        self.assertEqual(self.ids(other), ([1, 2], [1, 2]))
        self.assertEqual(self.checkpoint(other), (2, 2))

    def test_nested_batch_success_preserves_caller_transaction(self) -> None:
        self.conn.execute("INSERT INTO notes VALUES ('synthetic caller work')")
        self.assertTrue(self.process())
        self.assertTrue(self.conn.in_transaction)
        other = self.open_connection()
        self.assertEqual(self.ids(other), ([], []))
        self.assertEqual(self.checkpoint(other), (0, 0))
        self.conn.rollback()
        self.assertEqual(self.ids(), ([], []))
        self.assertEqual(self.checkpoint(), (0, 0))
        self.assertEqual(self.conn.execute("SELECT count(*) FROM notes").fetchone()[0], 0)

    def test_nested_batch_failure_preserves_unrelated_caller_work(self) -> None:
        self.conn.execute("INSERT INTO notes VALUES ('synthetic caller work')")
        self.fail_sql_once("separation", sqlite3.IntegrityError("synthetic nested failure"))
        with self.assertRaises(sqlite3.IntegrityError):
            self.process()
        self.assertTrue(self.conn.in_transaction)
        self.conn.commit()
        self.assertEqual(self.ids(), ([], []))
        self.assertEqual(self.checkpoint(), (0, 0))
        self.assertEqual(self.conn.execute("SELECT count(*) FROM notes").fetchone()[0], 1)

    def test_real_nested_release_denial_preserves_unrelated_caller_work(self) -> None:
        self.conn.execute("INSERT INTO notes VALUES ('synthetic caller work')")
        denied = []

        def deny_release(action: int, operation: str | None, name: str | None, *unused: str | None) -> int:
            if (action == sqlite3.SQLITE_SAVEPOINT and operation == "RELEASE"
                    and name == "truememory_rebuild_batch" and not denied):
                denied.append(True)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        self.conn.set_authorizer(deny_release)
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                self.process()
        finally:
            # Disabling with None is supported only on Python 3.11+.
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertEqual(denied, [True])
        self.assertTrue(self.conn.in_transaction)
        self.conn.commit()
        self.assertEqual(self.ids(), ([], []))
        self.assertEqual(self.checkpoint(), (0, 0))
        self.assertEqual(self.conn.execute("SELECT count(*) FROM notes").fetchone()[0], 1)

    def test_run_rejects_uncommitted_caller_work_before_model_encode(self) -> None:
        self.conn.execute("INSERT INTO notes VALUES ('synthetic caller work')")
        with self.assertRaisesRegex(RuntimeError, "no active transaction"):
            self.worker.run(self.messages, False)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.open_connection().execute("SELECT count(*) FROM notes").fetchone()[0], 0)

    def test_missing_registry_row_keeps_existing_full_restart_contract(self) -> None:
        self.conn.execute("DELETE FROM vector_cache_registry")
        self.conn.commit()
        self.assertEqual(self.worker.run(self.messages[:2], True), (True, 2))
        self.assertEqual(self.ids(), ([1, 2], [1, 2]))
        self.assertIsNone(self.checkpoint())

    def test_registry_default_commit_remains_compatible(self) -> None:
        self.conn.execute("INSERT INTO notes VALUES ('synthetic caller work')")
        self.cache.VectorCacheRegistry.update_progress(self.conn, "edge", 7, 7)
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.open_connection().execute("SELECT count(*) FROM notes").fetchone()[0], 1)


class SqliteVecWorkerAtomicity(WorkerAtomicity):
    """The same rollback/retry contract exercised against the actual extension."""
    use_vec = True


if __name__ == "__main__":
    unittest.main()
