"""Actual SQLite terminal writes with explicit synthetic clustering boundaries."""

import contextlib
import runpy
import sqlite3
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


TESTS = Path(__file__).resolve().parent
PRIMITIVES = runpy.run_path(str(TESTS / "test-maintenance-source-revision-753.py"))
STORAGE, MAINTENANCE = PRIMITIVES["STORAGE"], PRIMITIVES["MAINTENANCE"]
CLUSTERS = runpy.run_path(str(TESTS / "test-cluster-publication-749.py"))


class OuterClusterFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.vector, self.boundary = CLUSTERS["load_clustering"]()
        directory = tempfile.TemporaryDirectory(prefix="synthetic-outer-clusters-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "synthetic.sqlite"
        self.conn = self.open_db()
        self.conn.execute("CREATE TABLE vec_messages_edge(embedding BLOB)")
        for mid, values in enumerate(((1, 0), (0, 1), (0, 0), (2, 2), (4, 0)), 1):
            self.conn.execute(
                "INSERT INTO messages(id,content,sender,recipient,timestamp,category,modality) "
                "VALUES (?,'synthetic source','synthetic-sender','synthetic-recipient','2026-01-01','synthetic-session','text')", (mid,),
            )
            self.conn.execute("INSERT INTO vec_messages_edge(rowid,embedding) VALUES (?,?)", (mid, struct.pack("2f", *values)))
        self.conn.execute("INSERT INTO vector_cache_registry VALUES ('edge','vec_messages_edge','vec_messages_sep_edge',5,5,'synthetic-model-a',2,1.0,1.0)")
        self.conn.executemany("INSERT INTO metadata(key,value,updated_at) VALUES (?,?,'synthetic-stamp')", [
            ("embed_model", "synthetic-model-a"), ("embed_dim", "2"),
            ("vec_source_v1:vec_messages_edge", '{"generation":"synthetic-v1","cursor":5,"complete":true}'),
        ])
        self.conn.execute("INSERT INTO message_clusters(message_id,cluster_id,noise) VALUES (1,7,0)")
        self.conn.execute("INSERT INTO cluster_centroids(cluster_id,centroid,message_count,session_range) VALUES (7,?,1,'synthetic-prior')", (struct.pack("2f", 9, 9),))
        self.conn.commit()
        self.dependency = MAINTENANCE.make_layer_dependency(1, {"algorithm": "synthetic-hdbscan"})

    def open_db(self) -> sqlite3.Connection:
        conn = STORAGE.create_db(self.path)
        conn.execute("PRAGMA busy_timeout=100")
        self.addCleanup(conn.close)
        return conn

    def output(self, conn: sqlite3.Connection | None = None) -> tuple:
        conn = conn or self.conn
        return (conn.execute("SELECT * FROM message_clusters ORDER BY message_id").fetchall(),
                conn.execute("SELECT * FROM cluster_centroids ORDER BY cluster_id").fetchall())

    def success(self, conn: sqlite3.Connection | None = None) -> tuple:
        return (conn or self.conn).execute(
            "SELECT successful_epoch,successful_revision,successful_dependency,output_count "
            "FROM maintenance_layers WHERE layer='clusters'"
        ).fetchone()

    def spec(self, after_build=None, count_action=None, resolver=None, factory=None):
        def build(conn):
            self.module.cluster_messages(conn)
            if after_build is not None:
                after_build()

        def count(conn):
            if count_action is not None:
                count_action()
            return conn.execute("SELECT count(*) FROM cluster_centroids").fetchone()[0]

        return MAINTENANCE.LayerSpec(
            "clusters", "cluster_messages", resolver or (lambda: self.dependency), build, count,
            publication_guard=factory or self.module.cluster_publication_guard,
        )

    def run_layer(self, spec=None, **kwargs):
        return MAINTENANCE.run_layers(self.conn, (spec or self.spec(),), force=True, **kwargs)[0]


class TestClusterOuterOwnership(OuterClusterFixture):
    def test_actual_commit_holds_model_lock_and_publishes_checkpoint_with_output(self) -> None:
        reader = self.open_db()
        previous = self.output(reader), self.success(reader)
        observed = []

        def trace(sql):
            if sql == "COMMIT":
                observed.append((self.vector._lock.locked(), self.output(reader), self.success(reader)))

        result = self.run_layer(self.spec(count_action=lambda: self.conn.set_trace_callback(trace)))
        self.assertEqual(result.outcome, "success")
        self.assertEqual(result.output_count, 2)
        self.assertEqual(observed, [(True, *previous)])
        self.assertFalse(self.vector._lock.locked())
        self.assertNotEqual((self.output(reader), self.success(reader)), previous)
        self.assertEqual(self.success(reader)[3], 2)

    def test_guard_factory_captures_before_attempt_snapshot_and_resolvers_run_outside_lock(self) -> None:
        captures = []
        resolutions = []
        original = self.module.cluster_publication_guard

        def factory(conn):
            captures.append(conn.in_transaction)
            self.assertEqual(conn.execute("SELECT outcome FROM maintenance_layers WHERE layer='clusters'").fetchone(), ("running",))
            return original(conn)

        def resolver():
            available = self.vector._lock.acquire(blocking=False)
            resolutions.append(available)
            if available:
                self.vector._lock.release()
            return self.dependency

        self.assertEqual(self.run_layer(self.spec(factory=factory, resolver=resolver)).outcome, "success")
        self.assertEqual(captures, [False])
        self.assertGreaterEqual(len(resolutions), 5)
        self.assertTrue(all(resolutions))

    def test_model_changes_after_inner_release_roll_back_output_and_checkpoint(self) -> None:
        old = self.output(), self.success()
        rollbacks = []

        def change_model():
            self.assertTrue(self.conn.in_transaction)
            self.assertFalse(self.vector._lock.locked())
            with self.vector._lock:
                self.vector.EMBEDDING_MODEL = "synthetic-model-a-replacement"
            self.conn.set_trace_callback(lambda sql: rollbacks.append(self.vector._lock.locked()) if sql == "ROLLBACK" else None)

        result = self.run_layer(self.spec(after_build=change_model))
        self.assertEqual(result.outcome, "failed")
        self.assertEqual((self.output(), self.success()), old)
        self.assertEqual(rollbacks, [True])
        self.assertFalse(self.vector._lock.locked())

    def test_dimension_tier_table_registry_and_manifest_changes_are_fenced(self) -> None:
        changes = {
            "dimension": lambda: setattr(self.vector, "_embedding_dim", 3),
            "tier": lambda: setattr(self.vector, "_active_tier_group", lambda: "synthetic-new-tier"),
            "table": lambda: self.conn.execute("UPDATE vector_cache_registry SET vec_table='vec_messages_replaced'"),
            "registry": lambda: self.conn.execute("UPDATE vector_cache_registry SET last_updated=2.0"),
            "generation": lambda: self.conn.execute("UPDATE metadata SET value=? WHERE key='vec_source_v1:vec_messages_edge'", ('{"generation":"synthetic-v2","cursor":5,"complete":true}',)),
            "progress": lambda: self.conn.execute("UPDATE metadata SET value=? WHERE key='vec_source_v1:vec_messages_edge'", ('{"generation":"synthetic-v1","cursor":4,"complete":false}',)),
            "schema": lambda: self.conn.execute("ALTER TABLE vec_messages_edge ADD COLUMN synthetic INTEGER"),
        }
        original_group = self.vector._active_tier_group
        for kind, mutate in changes.items():
            with self.subTest(kind=kind):
                old = self.output(), self.success()
                result = self.run_layer(self.spec(after_build=mutate))
                self.assertEqual(result.outcome, "failed")
                self.assertEqual((self.output(), self.success()), old)
                self.assertFalse(self.vector._lock.locked())
                self.vector._embedding_dim = 2
                self.vector._active_tier_group = original_group

    def test_source_mutation_after_builder_is_rejected_before_checkpoint(self) -> None:
        old = self.output(), self.success()
        result = self.run_layer(self.spec(after_build=lambda: self.conn.execute(
            "UPDATE messages SET content='synthetic replacement' WHERE id=1"
        )))
        self.assertEqual(result.outcome, "failed")
        self.assertEqual((self.output(), self.success()), old)

    def test_busy_final_model_lock_does_not_wait_or_release_other_owner(self) -> None:
        old = self.output(), self.success()
        try:
            result = self.run_layer(self.spec(after_build=self.vector._lock.acquire))
            self.assertEqual(result.outcome, "failed")
            self.assertEqual((self.output(), self.success()), old)
            self.assertTrue(self.vector._lock.locked())
        finally:
            if self.vector._lock.locked():
                self.vector._lock.release()

    def test_model_changed_after_capture_before_builder_cannot_relabel_output(self) -> None:
        old = self.output(), self.success()
        original = self.module.cluster_publication_guard

        def factory(conn):
            guard = original(conn)
            self.vector.EMBEDDING_MODEL = "synthetic-model-a-replacement"
            return guard

        result = self.run_layer(self.spec(factory=factory))
        self.assertEqual(result.outcome, "failed")
        self.assertEqual((self.output(), self.success()), old)

    def test_guard_does_not_add_another_full_raw_source_scan(self) -> None:
        original = self.module._cluster_source_state
        calls = []

        def source(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        with patch.object(self.module, "_cluster_source_state", source):
            self.assertEqual(self.run_layer().outcome, "success")
        self.assertEqual(len(calls), 2)

    def test_concurrent_writer_can_commit_while_initial_capture_waits_for_model(self) -> None:
        writer = self.open_db()
        started = threading.Event()
        results = []
        self.vector._lock.acquire()

        def capture():
            started.set()
            try:
                results.append(self.module.cluster_publication_guard(self.conn))
            except Exception as error:
                results.append(error)

        worker = threading.Thread(target=capture)
        worker.start()
        try:
            self.assertTrue(started.wait(1))
            self.assertFalse(self.conn.in_transaction)
            writer.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-writer','committed')")
            writer.commit()
        finally:
            self.vector._lock.release()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(results), 1)
        self.assertNotIsInstance(results[0], Exception)
        self.assertFalse(self.vector._lock.locked())


class TestClusterOuterFailure(OuterClusterFixture):
    def test_actual_commit_failure_rolls_back_while_lock_held_then_releases(self) -> None:
        old = self.output(), self.success()
        terminal = []
        denied = []

        def authorizer(action, operation, *unused):
            if action == sqlite3.SQLITE_TRANSACTION and operation == "COMMIT" and not denied:
                denied.append(self.vector._lock.locked())
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        def observe(sql):
            if sql == "ROLLBACK":
                terminal.append(self.vector._lock.locked())

        def inject():
            self.conn.set_authorizer(authorizer)
            self.conn.set_trace_callback(observe)

        try:
            result = self.run_layer(self.spec(count_action=inject))
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertEqual(result.outcome, "failed")
        self.assertEqual(denied, [True])
        self.assertEqual(terminal, [True])
        self.assertEqual((self.output(), self.success()), old)
        self.assertFalse(self.vector._lock.locked())

    def test_cancel_after_guard_entry_rolls_back_before_releasing(self) -> None:
        old = self.output(), self.success()
        cancel = threading.Event()
        original = self.module.cluster_publication_guard
        rollbacks = []

        def factory(conn):
            guard = original(conn)

            @contextlib.contextmanager
            def held():
                with guard as validate:
                    cancel.set()
                    yield validate

            return held()

        self.conn.set_trace_callback(lambda sql: rollbacks.append(self.vector._lock.locked()) if sql == "ROLLBACK" else None)
        result = self.run_layer(self.spec(factory=factory), cancel=cancel)
        self.assertEqual(result.outcome, "abandoned")
        self.assertEqual(rollbacks, [True])
        self.assertEqual((self.output(), self.success()), old)
        self.assertFalse(self.vector._lock.locked())

    def test_guard_validation_exception_releases_model_and_preserves_old_output(self) -> None:
        old = self.output(), self.success()
        original = self.module._cluster_dependency_identity
        calls = []

        def identity(*args):
            calls.append(True)
            if len(calls) == 4:
                raise ValueError("synthetic guard failure")
            return original(*args)

        with patch.object(self.module, "_cluster_dependency_identity", identity):
            result = self.run_layer()
        self.assertEqual(result.outcome, "failed")
        self.assertEqual(len(calls), 4)
        self.assertEqual((self.output(), self.success()), old)
        self.assertFalse(self.vector._lock.locked())

    def test_interruption_during_validation_rolls_back_under_model_ownership(self) -> None:
        old = self.output(), self.success()
        original = self.module._cluster_dependency_identity
        calls = []
        rollbacks = []

        def identity(*args):
            calls.append(True)
            if len(calls) == 4:
                raise KeyboardInterrupt()
            return original(*args)

        self.conn.set_trace_callback(lambda sql: rollbacks.append(self.vector._lock.locked()) if sql == "ROLLBACK" else None)
        with patch.object(self.module, "_cluster_dependency_identity", identity):
            with self.assertRaises(KeyboardInterrupt):
                self.run_layer()
        self.assertEqual(rollbacks, [True])
        self.assertEqual((self.output(), self.success()), old)
        self.assertFalse(self.vector._lock.locked())

    def test_failed_rollback_propagates_and_does_not_leave_model_lock_held(self) -> None:
        old = self.output(), self.success()
        denied = []

        def authorizer(action, operation, *unused):
            if action == sqlite3.SQLITE_TRANSACTION and operation in {"COMMIT", "ROLLBACK"}:
                denied.append((operation, self.vector._lock.locked()))
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        try:
            with self.assertRaises(RuntimeError):
                self.run_layer(self.spec(count_action=lambda: self.conn.set_authorizer(authorizer)))
            self.assertTrue(self.conn.in_transaction)
            self.assertFalse(self.vector._lock.locked())
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
            self.conn.rollback()
        self.assertEqual(denied, [("COMMIT", True), ("ROLLBACK", True)])
        self.assertEqual((self.output(), self.success()), old)

    def test_dependency_change_in_final_resolver_rejects_before_guard_entry(self) -> None:
        old = self.output(), self.success()
        original = MAINTENANCE.record_layer_success_in_transaction

        def checkpoint(*args, **kwargs):
            original(*args, **kwargs)
            self.dependency = MAINTENANCE.make_layer_dependency(2)

        with patch.object(MAINTENANCE, "record_layer_success_in_transaction", checkpoint):
            result = self.run_layer()
        self.assertEqual(result.outcome, "failed")
        self.assertEqual((self.output(), self.success()), old)
        self.assertFalse(self.vector._lock.locked())

    def test_incomplete_index_and_missing_hdbscan_never_publish_empty_success(self) -> None:
        old = self.output(), self.success()
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('vec_build_state:vec_messages_edge','in_progress')")
        self.conn.commit()
        self.assertEqual(self.run_layer().outcome, "failed")
        self.assertEqual(self.boundary.inputs, [])
        self.conn.execute("DELETE FROM metadata WHERE key='vec_build_state:vec_messages_edge'")
        self.conn.commit()
        self.boundary.missing = True
        self.assertEqual(self.run_layer().outcome, "failed")
        self.assertEqual((self.output(), self.success()), old)
        self.dependency = MAINTENANCE.make_layer_dependency(1, available=False, error_category="ImportError")
        self.assertEqual(self.run_layer().outcome, "unavailable")
        self.assertEqual((self.output(), self.success()), old)

    def test_all_noise_publishes_valid_empty_cluster_count_and_complete_assignments(self) -> None:
        self.boundary.labels = [-1] * 5
        result = self.run_layer()
        self.assertEqual((result.outcome, result.output_count), ("success_empty", 0))
        self.assertEqual(self.output()[0], [(i, -1, 1) for i in range(1, 6)])
        self.assertEqual(self.output()[1], [])
        self.assertEqual(self.success()[3], 0)

    def test_capture_rejects_borrowed_transaction_without_committing_it(self) -> None:
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-pending','retained')")
        with self.assertRaises(RuntimeError):
            self.module.cluster_publication_guard(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertIsNone(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-pending'").fetchone())

    def test_existing_six_layer_specs_do_not_capture_or_enter_cluster_guards(self) -> None:
        self.assertTrue(all(spec.publication_guard is None for spec in MAINTENANCE.nonvector_layer_specs()))


if __name__ == "__main__":
    unittest.main()
