"""Foreground publication admission with real SQLite and synthetic models."""

import runpy
import sqlite3
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


FOREGROUND = runpy.run_path(str(Path(__file__).with_name("test-rebuild-foreground-publication-756.py")))
CLUSTERS = runpy.run_path(str(Path(__file__).with_name("test-cluster-publication-749.py")))


class AdmissionFixture(unittest.TestCase):
    setUp = FOREGROUND["TestForegroundPublication"].setUp
    seed = FOREGROUND["TestForegroundPublication"].seed
    ids = FOREGROUND["TestForegroundPublication"].ids
    writer = FOREGROUND["TestForegroundPublication"].writer
    engine = FOREGROUND["TestForegroundPublication"].engine

    def start(self, action, name="synthetic-worker"):
        results, done = [], threading.Event()

        def run():
            try:
                results.append(action())
            except BaseException as error:
                results.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=run, name=name, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        return thread, results, done

    def released(self):
        self.assertTrue(self.vector._lock.acquire(blocking=False))
        self.vector._lock.release()

    def sentinel(self):
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-caller','pending')")

    def preserved(self):
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-caller'").fetchone()[0], "pending")


class TestForegroundAdmission(AdmissionFixture):
    def test_style_computation_runs_outside_capture_and_publication_locks(self) -> None:
        engine = self.engine()
        engine._has_style_vec = True
        maintenance = self.modules["maintenance"]
        maintenance.prepare_style_maintenance(self.conn)
        self.conn.execute("BEGIN IMMEDIATE")
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('style_vec_hash_version','2')")
        maintenance.record_layer_success_in_transaction(
            self.conn, layer="style_vectors",
            dependency=maintenance.style_layer_spec(self.conn).resolve_dependency(),
            source=maintenance.read_source_revision(self.conn), output_count=0,
        )
        self.conn.commit()
        events = []
        style = [0.25, 0.75]

        def compute(content: str) -> list[float]:
            self.assertFalse(engine._write_lock.locked())
            self.assertFalse(self.vector._lock.locked())
            self.assertFalse(self.conn.in_transaction)
            events.append("compute")
            return style

        def publish(conn: sqlite3.Connection, sender: str, content: str,
                    _pre_computed_vec: list[float]) -> None:
            self.assertTrue(engine._write_lock.locked())
            self.assertTrue(self.vector._lock.locked())
            self.assertTrue(conn.in_transaction)
            self.assertIs(_pre_computed_vec, style)
            events.append("publish")

        module = types.ModuleType("synthetic_style")
        module.compute_style_vector = compute
        with patch.dict(self.modules, personality_style_vec=module), \
                patch.dict(engine.add.__globals__, _update_style_vec=publish):
            engine.add("synthetic styled content", sender="synthetic sender")
        self.assertEqual(events, ["compute", "publish"])
        self.assertEqual(self.ids(), [1])
        self.assertEqual(self.ids("vec_messages_sep"), [1])

    def test_actual_cluster_snapshot_delays_add_before_writer_then_add_commits_both_vectors(self):
        engine = self.engine()
        cluster, runtime, _native = CLUSTERS["load_clustering"]()
        encoding, snapshot, admission, release = (threading.Event() for _ in range(4))
        underlying = self.vector._lock
        fixture = self

        class ObservedLock:
            def acquire(self, blocking=True):
                if threading.current_thread().name == "synthetic-add" and snapshot.is_set():
                    fixture.assertTrue(blocking)
                    fixture.assertFalse(engine.conn.in_transaction)
                    fixture.assertTrue(engine._write_lock.locked())
                    admission.set()
                return underlying.acquire(blocking=blocking)

            def release(self):
                underlying.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *_args):
                self.release()

        self.vector._lock = runtime._lock = ObservedLock()
        runtime.EMBEDDING_MODEL = self.vector.EMBEDDING_MODEL
        cluster_conn, independent = self.writer(), self.writer()
        original = cluster._get_all_embeddings

        def read(conn):
            snapshot.set()
            self.assertTrue(release.wait(3))
            return original(conn)

        def encode(texts):
            if len(self.calls) == 1:
                encoding.set()
                self.assertTrue(snapshot.wait(3))
            return [[1., 0.] for _ in texts]

        def consolidate():
            self.assertTrue(encoding.wait(3))
            return cluster.cluster_messages(cluster_conn)

        self.on_encode = encode
        with patch.object(cluster, "_get_all_embeddings", read):
            cluster_thread, cluster_result, cluster_done = self.start(consolidate)
            add_thread, result, done = self.start(lambda: engine.add("synthetic first add"), "synthetic-add")
            try:
                self.assertTrue(admission.wait(2))
                self.assertFalse(done.is_set())
                independent.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-independent','committed')")
                independent.commit()
                self.assertFalse(self.conn.in_transaction)
            finally:
                release.set()
                add_thread.join(3)
                cluster_thread.join(3)
        self.assertTrue(done.is_set())
        self.assertTrue(cluster_done.is_set())
        self.assertIsInstance(cluster_result[0], (int, RuntimeError), cluster_result)
        self.assertIsInstance(result[0], dict, result)
        self.assertEqual(self.ids(), [1])
        self.assertEqual(self.ids("vec_messages_sep"), [1])
        self.assertEqual(self.conn.execute("SELECT content FROM messages").fetchone()[0], "synthetic first add")
        self.released()

    def test_writer_owner_defers_nonblocking_model_fence_and_foreground_finishes(self):
        owner = self.writer()
        owner.execute("BEGIN IMMEDIATE")
        admission, attempted = threading.Event(), threading.Event()
        underlying = self.vector._lock

        class AdmittedLock:
            def acquire(self, blocking=True):
                acquired = underlying.acquire(blocking=blocking)
                if acquired and threading.current_thread().name == "synthetic-publish":
                    admission.set()
                return acquired

            def release(self):
                underlying.release()

        self.vector._lock = AdmittedLock()
        identity = (0, self.vector.EMBEDDING_MODEL, 2)

        def publish():
            with self.vector._foreground_vector_publication(self.conn, identity, 1):
                self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-published','yes')")

        thread, result, done = self.start(publish, "synthetic-publish")
        try:
            self.assertTrue(admission.wait(2))
            with self.assertRaisesRegex(RuntimeError, "model is busy"):
                with self.vector._rebuild_model_fence(identity):
                    attempted.set()
            self.assertFalse(attempted.is_set())
        finally:
            owner.rollback()
            thread.join(3)
        self.assertTrue(done.is_set())
        self.assertEqual(result, [None])
        self.released()

    def test_add_waiting_for_same_engine_update_does_not_own_model(self):
        self.seed((1,))
        engine = self.engine()
        update_ready, add_waiting = threading.Event(), threading.Event()
        underlying = engine._write_lock

        class EngineLock:
            def __enter__(self):
                if threading.current_thread().name == "synthetic-add":
                    add_waiting.set()
                underlying.acquire()
                return self

            def __exit__(self, *_args):
                underlying.release()

        engine._write_lock = EngineLock()
        original = engine.update.__globals__["update_message"]

        def update(conn, memory_id, **fields):
            result = original(conn, memory_id, **fields)
            update_ready.set()
            self.assertTrue(add_waiting.wait(3))
            self.released()
            return result

        with patch.dict(engine.update.__globals__, update_message=update):
            update_thread, updated, update_done = self.start(lambda: engine.update(1, "synthetic updated"), "synthetic-update")
            self.assertTrue(update_ready.wait(2))
            add_thread, added, add_done = self.start(lambda: engine.add("synthetic added"), "synthetic-add")
            update_thread.join(3)
            add_thread.join(3)
        self.assertTrue(update_done.is_set())
        self.assertTrue(add_done.is_set())
        self.assertIsInstance(updated[0], dict, updated)
        self.assertIsInstance(added[0], dict, added)
        self.assertEqual(self.ids(), [1, 2])
        self.assertEqual(self.ids("vec_messages_sep"), [1, 2])

    def test_same_engine_writer_finishes_before_capture_classifies_transaction(self) -> None:
        self.seed((1,))
        for operation in ("add", "update"):
            with self.subTest(operation=operation):
                engine = self.engine()
                writer_ready, waiting, release = (threading.Event() for _ in range(3))
                underlying = engine._write_lock

                class EngineLock:
                    def __enter__(self) -> "EngineLock":
                        if threading.current_thread().name == "synthetic-capture":
                            waiting.set()
                        underlying.acquire()
                        return self

                    def __exit__(self, *_args: object) -> None:
                        underlying.release()

                engine._write_lock = EngineLock()
                self.vector._model = None
                loads = []

                def load() -> object:
                    self.assertFalse(self.conn.in_transaction)
                    self.assertTrue(underlying.locked())
                    loads.append(True)
                    self.vector._model = self.model
                    return self.model

                def writer() -> None:
                    with engine._write_lock:
                        self.conn.execute("BEGIN IMMEDIATE")
                        self.conn.execute("INSERT INTO metadata(key,value) VALUES (?, 'pending')",
                                          ("synthetic-owner-" + operation,))
                        writer_ready.set()
                        self.assertTrue(release.wait(3))
                        self.conn.commit()

                with patch.object(self.vector, "get_model", load):
                    writer_thread, written, writer_done = self.start(writer)
                    self.assertTrue(writer_ready.wait(2))
                    action = (lambda: engine.add("synthetic added")) if operation == "add" else (
                        lambda: engine.update(1, "synthetic updated")
                    )
                    mutation_thread, result, done = self.start(action, "synthetic-capture")
                    try:
                        self.assertTrue(waiting.wait(2))
                        self.assertFalse(done.is_set())
                        self.assertEqual(loads, [])
                    finally:
                        release.set()
                        mutation_thread.join(3)
                        writer_thread.join(3)
                self.assertTrue(writer_done.is_set())
                self.assertEqual(written, [None])
                self.assertTrue(done.is_set())
                self.assertIsInstance(result[0], dict, result)
                self.assertEqual(loads, [True])
                self.assertFalse(self.conn.in_transaction)
                self.released()

    def test_generation_change_during_model_load_rejects_add_and_update_before_source_write(self):
        self.seed((1,))
        engine = self.engine()

        def changed():
            self.vector._model_generation += 1
            return self.model

        with patch.object(self.vector, "get_model", changed):
            for action in (lambda: engine.add("synthetic rejected"), lambda: engine.update(1, "synthetic rejected")):
                with self.assertRaisesRegex(self.vector.VectorPublicationChanged, "changed while loading"):
                    action()
        self.assertEqual(self.conn.execute("SELECT id,content FROM messages").fetchall(), [(1, "synthetic-1")])
        self.assertEqual(self.ids(), [])
        self.released()

    def test_generation_change_while_waiting_rejects_before_opening_transaction(self):
        captured = (0, self.vector.EMBEDDING_MODEL, 2)
        attempted = threading.Event()
        underlying = self.vector._lock

        class WaitingLock:
            def acquire(self, blocking=True):
                attempted.set()
                return underlying.acquire(blocking=blocking)

            def release(self):
                underlying.release()

        self.vector._lock = WaitingLock()
        underlying.acquire()

        def publish():
            with self.vector._foreground_vector_publication(self.conn, captured, 1):
                self.fail("Changed generation entered publication")

        thread, results, done = self.start(publish)
        try:
            self.assertTrue(attempted.wait(2))
            self.assertFalse(done.is_set())
            self.assertFalse(self.conn.in_transaction)
            self.vector._model_generation += 1
        finally:
            underlying.release()
            thread.join(3)
        self.assertIsInstance(results[0], self.vector.VectorPublicationChanged)
        self.assertIn("model changed", str(results[0]))
        self.assertFalse(self.conn.in_transaction)
        self.released()

    def test_add_failure_after_source_insert_rolls_back_source_and_vectors(self):
        engine = self.engine()
        with patch.object(self.vector, "_validate_foreground_vector_target", side_effect=self.vector.VectorPublicationChanged("synthetic target changed")):
            with self.assertRaises(self.vector.VectorPublicationChanged):
                engine.add("synthetic rollback")
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.ids("vec_messages_sep"), [])
        self.assertFalse(self.conn.in_transaction)
        self.released()

    def test_admission_begin_and_commit_failures_release_model(self):
        real = self.conn

        class FailedConnection:
            def __init__(self, phase):
                self.phase = phase

            def __getattr__(self, name):
                return getattr(real, name)

            def execute(self, sql, *args):
                if self.phase == "begin" and sql == "BEGIN IMMEDIATE":
                    raise sqlite3.OperationalError("synthetic writer admission")
                return real.execute(sql, *args)

            def commit(self):
                if self.phase == "commit":
                    raise sqlite3.OperationalError("synthetic commit failure")
                return real.commit()

        for phase in ("begin", "commit"):
            with self.subTest(phase=phase):
                with self.assertRaises(sqlite3.OperationalError):
                    self.engine(FailedConnection(phase)).add("synthetic rollback")
                self.assertFalse(real.in_transaction)
                self.assertEqual(real.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
                self.assertEqual(self.ids(), [])
                self.released()

    def test_owned_model_load_and_encoding_finish_before_writer(self):
        def load():
            self.assertFalse(self.conn.in_transaction)
            return self.model

        def encode(texts):
            self.assertFalse(self.conn.in_transaction)
            self.released()
            return [[1., 0.] for _ in texts]

        self.on_encode = encode
        with patch.object(self.vector, "get_model", load):
            self.engine().add("synthetic outside writer")
        self.assertEqual(self.ids(), [1])


class TestBorrowedAdmission(AdmissionFixture):
    def actions(self):
        engine = self.engine()
        return (
            ("add", lambda: engine.add("synthetic rejected")),
            ("update", lambda: engine.update(1, "synthetic rejected")),
            ("embed_single", lambda: self.vector.embed_single(self.conn, 1, "synthetic-1")),
            ("completion", lambda: self.vector.build_vectors(self.conn)),
            ("separation", lambda: self.vector.build_separation_vectors(self.conn)),
        )

    def test_busy_capture_defers_every_mutation_without_waiting_or_loading(self):
        self.seed((1,))
        self.sentinel()
        calls = []

        class BusyLock:
            def acquire(self, blocking=True):
                calls.append(blocking)
                if blocking:
                    raise AssertionError("Borrowed caller attempted a blocking model acquisition")
                return False

        with patch.object(self.vector, "_lock", BusyLock()), patch.object(self.vector, "get_model", side_effect=AssertionError("Borrowed model load")):
            for name, action in self.actions():
                with self.subTest(action=name):
                    with self.assertRaisesRegex(self.vector.VectorPublicationChanged, "model is busy"):
                        action()
                    self.preserved()
        self.assertEqual(calls, [False] * 5)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.conn.execute("SELECT id,content FROM messages").fetchall(), [(1, "synthetic-1")])
        self.assertEqual(self.ids(), [])
        self.conn.rollback()
        self.released()

    def test_unloaded_capture_defers_every_mutation_and_preserves_sentinel(self):
        self.seed((1,))
        self.sentinel()
        self.vector._model = None
        with patch.object(self.vector, "get_model", side_effect=AssertionError("Borrowed model load")) as loader:
            for name, action in self.actions():
                with self.subTest(action=name):
                    with self.assertRaisesRegex(self.vector.VectorPublicationChanged, "not loaded"):
                        action()
                    self.preserved()
                    self.released()
            loader.assert_not_called()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.conn.execute("SELECT id,content FROM messages").fetchall(), [(1, "synthetic-1")])
        self.conn.rollback()

    def test_borrowed_loaded_snapshot_skips_loader_and_add_remains_rollbackable(self):
        self.sentinel()
        with patch.object(self.vector, "get_model", side_effect=AssertionError("Borrowed model load")) as loader:
            item = self.engine().add("synthetic borrowed")
            loader.assert_not_called()
        self.preserved()
        self.assertEqual(self.ids(), [item["id"]])
        self.assertEqual(self.ids("vec_messages_sep"), [item["id"]])
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
        self.assertEqual(self.ids(), [])
        self.released()

    def test_borrowed_publication_contention_preserves_caller_transaction(self):
        self.sentinel()
        held, release = threading.Event(), threading.Event()

        def hold():
            with self.vector._lock:
                held.set()
                self.assertTrue(release.wait(3))

        holder = []

        def encode(texts):
            if len(self.calls) == 2:
                holder.append(self.start(hold)[0])
                self.assertTrue(held.wait(2))
            return [[1., 0.] for _ in texts]

        self.on_encode = encode
        try:
            with self.assertRaisesRegex(self.vector.VectorPublicationChanged, "model is busy"):
                self.engine().add("synthetic rejected")
            self.preserved()
            self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
        finally:
            release.set()
            for thread in holder:
                thread.join(3)
        self.conn.rollback()
        self.released()


if __name__ == "__main__":
    unittest.main()
