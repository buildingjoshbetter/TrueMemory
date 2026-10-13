"""Synthetic SQLite and actual rebuild control flow; no model imports."""

import ast
import builtins
from collections.abc import Iterator
import datetime
import logging
import math
import os
import sqlite3
import struct
import sys
import tempfile
import threading
import types
import unittest
from contextlib import closing, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_modules():
    serving_source = ast.parse((ROOT / "truememory/tier_switch/serving.py").read_text(encoding="utf-8"))
    serving_names = {"ServingLeaseError", "ServingLeaseTimeout", "ServingLeaseCancelled"}
    serving_namespace = {}
    exec(compile(ast.Module(body=[node for node in serving_source.body
                                 if isinstance(node, ast.ClassDef) and node.name in serving_names],
                            type_ignores=[]), "actual-serving-errors", "exec"), serving_namespace)
    runtime_source = ast.parse((ROOT / "truememory/tier_switch/runtime.py").read_text(encoding="utf-8"))
    runtime_names = {"TierRuntimeError", "raise_if_serving_rejection", "_read_selection", "_read_policy",
                     "require_legacy_vector_mutation", "legacy_vector_mutation",
                     "open_serving_connection", "_open_serving_connection_at_path", "_modules", "_legacy_key"}
    runtime_namespace = {"__builtins__": dict(vars(builtins)), "sqlite3": sqlite3,
                         "contextmanager": contextmanager, "serving": types.SimpleNamespace(
        **{name: serving_namespace[name] for name in serving_names})}
    runtime_nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    runtime_nodes.extend(node for node in runtime_source.body
                         if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in runtime_names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=runtime_nodes, type_ignores=[])),
                 "actual-runtime-guards", "exec"), runtime_namespace)

    def legacy_operation(conn: sqlite3.Connection) -> types.SimpleNamespace:
        vector = modules["vector_search"]
        return types.SimpleNamespace(selection=None, policy=None,
                                     tables=(vector._active_vec_table(conn), vector._active_sep_table(conn)),
                                     key=runtime_namespace["_legacy_key"]())

    modules = {"tier_switch.runtime": types.SimpleNamespace(
        **{name: runtime_namespace[name] for name in runtime_names},
        serving_operation=lambda conn, **kwargs: nullcontext(legacy_operation(conn)),
        current_operation=legacy_operation),
        "reranker": types.SimpleNamespace(get_current_reranker_name=lambda: "synthetic/reranker")}

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("Native models are excluded from synthetic tests")
        if name == "psutil":
            return types.SimpleNamespace()
        if name == "truememory":
            return types.SimpleNamespace(vector_search=modules["vector_search"], reranker=modules["reranker"])
        if name.startswith("truememory."):
            short = name.removeprefix("truememory.")
            if short in modules:
                return modules[short]
            raise AssertionError("Unexpected application import: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    runtime_namespace["__builtins__"]["__import__"] = safe_import
    for name in ("storage", "_platform", "maintenance", "rebuild_source", "tier_switch.writer",
                 "embedding_target", "tier_config", "tier_switch.cache", "tier_switch.job",
                 "tier_switch.source", "tier_switch.activation"):
        module = types.ModuleType("synthetic_stream_" + name.replace(".", "_"))
        sys.modules[module.__name__] = module
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
        path = ROOT / "truememory" / (name.replace(".", "/") + ".py")
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
        modules[name] = module
    activation = modules["tier_switch.activation"]
    runtime_namespace.update(_guard_storage=activation._guard_storage, _state=activation._state,
                             read_activation_state=activation.read_activation_state)
    vector = types.ModuleType("synthetic_stream_vector")
    vector.__dict__.update(
        __builtins__=dict(vars(builtins), __import__=safe_import, Path=Path),
        database_operation=lambda function: function,
        contextmanager=contextmanager, closing=closing, Iterator=Iterator, replace=replace, datetime=datetime,
        WriterCapture=modules["tier_switch.writer"].WriterCapture,
        sqlite3=sqlite3, logger=logging.getLogger(__name__), _lock=threading.Lock(),
        _model_generation=0, EMBEDDING_MODEL="synthetic-model", _embedding_dim=2,
        resolve_tier=lambda: "edge",
        _BUILD_VECTORS_TXN_BATCH=100, _BUILD_STATE_KEY_PREFIX="vec_build_state:",
        _get_batch_size=lambda: 2, _flush_mps_cache=lambda: None,
        _active_vec_table=lambda conn: "vec_messages", _active_sep_table=lambda conn: "vec_messages_sep",
        serialize_f32=lambda vector: struct.pack("2f", *vector),
        _normalize_for_cosine=lambda vector: vector if all(math.isfinite(v) for v in vector) and any(vector) else None,
    )
    selected = {"_capture_rebuild_model", "_rebuild_model_fence", "_build_streamed_vectors",
                "build_vectors", "build_separation_vectors", "_build_sep_text", "_ensure_metadata_table",
                "_build_state_key", "_mark_build_in_progress", "_clear_build_in_progress",
                "_build_in_progress", "_write_embedder_metadata_no_commit", "_write_embedder_metadata",
                "_write_foreground_embedder_metadata_no_commit",
                "_foreground_vector_publication", "_foreground_model_fence", "_validate_foreground_vector_target",
                "VectorPublicationChanged", "embed_single"}
    tree = ast.parse((ROOT / "truememory/vector_search.py").read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in selected]
    exec(compile(ast.Module(body=body, type_ignores=[]), "actual-vector-builders", "exec"), vector.__dict__)
    modules["vector_search"] = vector
    runtime_namespace["sys"] = types.SimpleNamespace(
        modules={"truememory." + name: module for name, module in modules.items()})
    return modules


class TestStreamingBuilders(unittest.TestCase):
    def setUp(self):
        self.modules = load_modules()
        self.source = self.modules["rebuild_source"]
        self.vector = self.modules["vector_search"]
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-streaming-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.sqlite"
        self.conn = self.modules["storage"].create_db(self.path)
        self.addCleanup(self.conn.close)
        self.conn.executescript("CREATE TABLE vec_messages(embedding BLOB); CREATE TABLE vec_messages_sep(embedding BLOB);")
        self.model = object()
        self.vector._model = self.model
        self.vector.get_model = lambda: self.model
        self.calls = []
        self.on_encode = None

        def encode(model, texts, **kwargs):
            self.assertIs(model, self.model)
            self.calls.append(list(texts))
            if self.on_encode is not None:
                return self.on_encode(texts)
            return [[1., 0.] for _ in texts]
        self.vector._encode_with_mps_fallback = encode

    def seed(self, ids=range(1, 10)):
        for mid in ids:
            self.conn.execute("INSERT INTO messages(id,content,sender,recipient,timestamp) VALUES (?,?,?,?,?)",
                              (mid, f"synthetic-{mid}", "sender", "recipient", "2026-01-02"))
        self.conn.commit()

    def ids(self, table="vec_messages"):
        return [row[0] for row in self.conn.execute(f"SELECT rowid FROM {table} ORDER BY rowid")]

    def manifest(self, table="vec_messages"):
        return self.source.load_manifest(self.conn, self.source.manifest_key((table,)))

    def writer(self):
        conn = self.modules["storage"].create_db(self.path)
        self.addCleanup(conn.close)
        return conn

    def test_exact_ids_texts_and_separation_with_negative_zero_and_holes(self):
        self.seed((-10, 0, 2, 8, 11))
        self.assertEqual(self.vector.build_vectors(self.conn, txn_batch=2), 5)
        self.assertEqual(self.ids(), [-10, 0, 2, 8, 11])
        self.assertEqual(sum(self.calls, []), [f"synthetic-{mid}" for mid in (-10, 0, 2, 8, 11)])
        self.calls.clear()
        self.assertEqual(self.vector.build_separation_vectors(self.conn, txn_batch=2), 5)
        self.assertEqual(sum(self.calls, []), [f"sender to recipient on 2026-01-02: synthetic-{mid}" for mid in (-10, 0, 2, 8, 11)])

    def test_pages_release_read_and_writer_transactions_before_each_encode(self):
        self.seed(range(1, 22))
        writer = self.writer()
        def encode(texts):
            self.assertFalse(self.conn.in_transaction)
            writer.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('synthetic-writer','ok')")
            writer.commit()
            self.assertLessEqual(len(texts), 2)
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        self.vector.build_vectors(self.conn, txn_batch=3)
        self.assertEqual(len(self.ids()), 21)
        self.assertFalse(self.conn.in_transaction)

    def test_resume_uses_consumed_input_when_last_vector_skipped(self):
        self.seed(range(1, 7))
        def encode(texts):
            if len(self.calls) == 2:
                raise RuntimeError("synthetic interruption")
            return [[1., 0.], [0., 0.]]
        self.on_encode = encode
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            self.vector.build_vectors(self.conn, txn_batch=1)
        self.assertEqual(self.ids(), [1])
        self.assertEqual((self.manifest().cursor, self.manifest().consumed), (2, 2))
        self.on_encode = None
        self.calls.clear()
        self.assertEqual(self.vector.build_vectors(self.conn, txn_batch=1), 4)
        self.assertEqual(sum(self.calls, []), [f"synthetic-{mid}" for mid in range(3, 7)])
        self.assertEqual(self.ids(), [1, 3, 4, 5, 6])

    def test_all_invalid_outputs_finish_with_nonzero_consumed_cursor(self):
        self.seed()
        self.on_encode = lambda texts: [[0., 0.] for _ in texts]
        self.assertEqual(self.vector.build_vectors(self.conn, txn_batch=2), 0)
        manifest = self.manifest()
        self.assertEqual((manifest.consumed, manifest.cursor, manifest.outputs, manifest.complete), (9, 9, 0, True))
        self.assertFalse(self.vector._build_in_progress(self.conn, "vec_messages"))

    def test_explicit_list_keeps_order_and_text_without_database_rows(self):
        messages = [{"id": 9, "content": "synthetic-nine"}, {"id": -1, "content": "synthetic-minus"},
                    {"id": 3, "content": "synthetic-three"}]
        self.assertEqual(self.vector.build_vectors(self.conn, messages, txn_batch=1), 3)
        self.assertEqual(sum(self.calls, []), [message["content"] for message in messages])
        self.assertIsNone(self.manifest().source)

    def test_caller_transaction_is_preserved_on_success(self):
        self.seed()
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('caller','pending')")
        reader = self.writer()
        self.vector.build_vectors(self.conn, txn_batch=1)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(reader.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0], 0)
        self.conn.rollback()
        self.assertEqual(self.ids(), [])
        self.assertIsNone(self.manifest())

    def test_correction_during_compute_rejects_publication_and_fresh_run_restarts(self):
        self.seed()
        writer = self.writer()
        def encode(texts):
            writer.execute("UPDATE messages SET content='synthetic-changed' WHERE id=1")
            writer.commit()
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaises(self.source.RebuildSourceChanged):
            self.vector.build_vectors(self.conn, txn_batch=1)
        self.assertEqual(self.ids(), [])
        self.assertTrue(self.vector._build_in_progress(self.conn, "vec_messages"))
        self.on_encode = None
        self.vector.build_vectors(self.conn, txn_batch=1)
        self.assertEqual(len(self.ids()), 9)

    def test_model_change_during_compute_rejects_rows_and_metadata(self):
        self.seed()
        def encode(texts):
            self.vector._model_generation += 1
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaisesRegex(RuntimeError, "model changed"):
            self.vector.build_vectors(self.conn, txn_batch=1)
        self.assertEqual(self.ids(), [])
        self.assertIsNone(self.conn.execute("SELECT value FROM metadata WHERE key='embed_model'").fetchone())

    def test_empty_database_finishes_without_loading_a_native_model(self):
        def no_load():
            raise AssertionError("Empty source must not load a model")
        self.vector.get_model = no_load
        self.assertEqual(self.vector.build_vectors(self.conn), 0)
        self.assertTrue(self.manifest().complete)
        self.assertEqual(self.calls, [])

    def test_incomplete_model_output_is_rejected_before_writes(self):
        self.seed()
        self.on_encode = lambda texts: []
        with self.assertRaisesRegex(ValueError, "count"):
            self.vector.build_vectors(self.conn, txn_batch=1)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.manifest().consumed, 0)

    def test_epoch_change_invalidates_saved_resume(self):
        self.seed()
        def encode(texts):
            if len(self.calls) > 1:
                raise RuntimeError("synthetic interruption")
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaises(RuntimeError):
            self.vector.build_vectors(self.conn, txn_batch=1)
        old = self.manifest()
        self.conn.execute("UPDATE maintenance_source_state SET epoch='synthetic-new-epoch'")
        self.conn.commit()
        self.on_encode = None
        self.calls.clear()
        self.vector.build_vectors(self.conn, txn_batch=1)
        self.assertNotEqual(self.manifest().generation, old.generation)
        self.assertEqual(sum(self.calls, []), [f"synthetic-{mid}" for mid in range(1, 10)])

    def test_source_mutations_after_fetch_are_classified_without_partial_publication(self):
        cases = (
            ("hole", "INSERT INTO messages(id,content) VALUES (3,'synthetic-hole')", False),
            ("delete", "DELETE FROM messages WHERE id=2", False),
            ("rowid", "UPDATE messages SET rowid=3 WHERE id=2", False),
            ("sender", "UPDATE messages SET sender='synthetic-change' WHERE id=2", False),
            ("timestamp", "UPDATE messages SET timestamp='2025-01-01' WHERE id=2", False),
            ("derived", "UPDATE messages SET episode_id=13 WHERE id=2", True),
            ("append", "INSERT INTO messages(id,content) VALUES (100,'synthetic-tail')", True),
        )
        for name, sql, accepted in cases:
            with self.subTest(name=name):
                # Each mutation gets an independent generation and database.
                case = TestStreamingBuilders()
                case.setUp()
                try:
                    case.seed((2, 4, 6))
                    writer = case.writer()
                    changed = []
                    def encode(texts):
                        if not changed:
                            changed.append(True)
                            writer.execute(sql)
                            writer.commit()
                        return [[1., 0.] for _ in texts]
                    case.on_encode = encode
                    if accepted:
                        case.vector.build_vectors(case.conn, txn_batch=1)
                        self.assertEqual(case.ids(), [2, 4, 6])
                        self.assertEqual(case.manifest().consumed, 3)
                    else:
                        with self.assertRaises(case.source.RebuildSourceChanged):
                            case.vector.build_vectors(case.conn, txn_batch=1)
                        self.assertEqual(case.ids(), [])
                finally:
                    case.doCleanups()

    def test_rolled_back_source_correction_does_not_invalidate(self):
        self.seed()
        writer = self.writer()
        def encode(texts):
            writer.execute("UPDATE messages SET content='synthetic-rolled-back' WHERE id=1")
            writer.rollback()
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        self.assertEqual(self.vector.build_vectors(self.conn, txn_batch=1), 9)

    def test_short_pages_do_not_pin_a_wal_reader_during_native_work(self):
        self.seed()
        writer = self.writer()
        checks = []
        def encode(texts):
            self.assertFalse(self.conn.in_transaction)
            writer.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('checkpoint-probe','synthetic')")
            writer.commit()
            busy, log_frames, checkpointed = writer.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            checks.append((busy, log_frames, checkpointed))
            self.assertEqual(busy, 0)
            self.assertEqual(log_frames, checkpointed)
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        self.vector.build_vectors(self.conn, txn_batch=2)
        self.assertTrue(checks)

    def test_group_staging_and_commit_cadence_stay_bounded(self):
        self.seed(range(1, 24))
        real = self.conn
        sizes = []
        class Observed:
            commits = 0
            def __getattr__(self, name):
                return getattr(real, name)
            def execute(self, sql, parameters=()):
                normalized = " ".join(sql.lower().split())
                if normalized.startswith("select id, content"):
                    self_check.assertIn(" limit ?", normalized)
                    self_check.assertLessEqual(parameters[-1], 2)
                return real.execute(sql, parameters)
            def executemany(self, sql, rows):
                sizes.append((len(rows), sum(len(row[1]) for row in rows)))
                return real.executemany(sql, rows)
            def commit(self):
                self.commits += 1
                real.commit()
        self_check = self
        observed = Observed()
        self.vector.build_vectors(observed, txn_batch=3)
        self.assertEqual([item[0] for item in sizes], [6, 6, 6, 5])
        self.assertLessEqual(max(item[1] for item in sizes), 3 * 2 * 2 * 4)
        self.assertEqual(observed.commits, 1 + 4 + 1)

    def test_cancellation_before_group_commit_has_no_partial_group(self):
        self.seed()
        def encode(texts):
            if len(self.calls) == 2:
                raise KeyboardInterrupt("synthetic cancellation")
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaises(KeyboardInterrupt):
            self.vector.build_vectors(self.conn, txn_batch=3)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.manifest().consumed, 0)
        self.assertFalse(self.conn.in_transaction)

    def test_failed_commit_cannot_leave_vectors_or_cursor_for_later_commit(self):
        self.seed()
        real = self.conn
        class DeniedCommit:
            commits = 0
            def __getattr__(self, name):
                return getattr(real, name)
            def commit(self):
                self.commits += 1
                if self.commits == 2:
                    raise sqlite3.OperationalError("synthetic commit failure")
                real.commit()
        with self.assertRaisesRegex(sqlite3.OperationalError, "synthetic commit failure"):
            self.vector.build_vectors(DeniedCommit(), txn_batch=1)
        real.commit()
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.manifest().consumed, 0)

    def test_savepoint_release_denial_rolls_back_group_but_keeps_caller_work(self):
        self.seed()
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('caller','pending')")
        denied = []
        def authorize(action, first, second, *_args):
            if action == sqlite3.SQLITE_SAVEPOINT and first == "RELEASE" and second == "rebuild_source" and self.calls and not denied:
                denied.append(True)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        def encode(texts):
            # Reset the statement cache after initial savepoints were prepared.
            self.conn.set_authorizer(authorize)
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                self.vector.build_vectors(self.conn, txn_batch=1)
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.manifest().consumed, 0)
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='caller'").fetchone()[0], "pending")
        self.conn.commit()
        self.assertEqual(self.ids(), [])

    def test_legacy_marker_with_high_output_id_is_not_a_resume_certificate(self):
        self.seed((1, 2, 3))
        self.conn.execute("INSERT INTO vec_messages(rowid,embedding) VALUES (999,?)", (struct.pack("2f", 1., 0.),))
        self.vector._mark_build_in_progress(self.conn, "vec_messages")
        self.conn.commit()
        self.assertEqual(self.vector.build_vectors(self.conn, txn_batch=1), 3)
        self.assertEqual(self.ids(), [1, 2, 3])

    def test_explicit_input_resume_tracks_position_not_sorted_id(self):
        messages = [{"id": mid, "content": f"synthetic-{mid}"} for mid in (9, -4, 8, 2)]
        def encode(texts):
            if len(self.calls) == 2:
                raise RuntimeError("synthetic interruption")
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaises(RuntimeError):
            self.vector.build_vectors(self.conn, messages, txn_batch=1)
        self.on_encode = None
        self.calls.clear()
        self.assertEqual(self.vector.build_vectors(self.conn, messages, txn_batch=1), 2)
        self.assertEqual(sum(self.calls, []), ["synthetic-8", "synthetic-2"])

    def test_competing_owner_is_rejected_before_loading_or_clearing(self):
        self.seed()
        ready = threading.Event()
        release = threading.Event()
        errors = []
        def own():
            try:
                with self.modules["maintenance"].maintenance_owner(self.path):
                    ready.set()
                    if not release.wait(3):
                        raise RuntimeError("synthetic owner timeout")
            except BaseException as exc:
                errors.append(exc)
        owner = threading.Thread(target=own)
        owner.start()
        try:
            self.assertTrue(ready.wait(3))
            with self.assertRaises(self.modules["maintenance"].MaintenanceBusyError):
                self.vector.build_vectors(self.conn)
            self.assertEqual(self.calls, [])
            self.assertIsNone(self.manifest())
        finally:
            release.set()
            owner.join(3)
        self.assertFalse(owner.is_alive())
        self.assertEqual(errors, [])

    def test_unmigrated_legacy_connection_keeps_bridge_after_full_storage_migration(self):
        legacy_path = Path(self.temp.name) / "synthetic-legacy.sqlite"
        legacy = sqlite3.connect(legacy_path)
        self.addCleanup(legacy.close)
        legacy.executescript("CREATE TABLE messages(id INTEGER PRIMARY KEY,content TEXT,sender TEXT,recipient TEXT,timestamp TEXT);"
                             "CREATE TABLE vec_messages(embedding BLOB);"
                             "INSERT INTO messages VALUES(1,'synthetic-legacy','sender','recipient','2026-01-01');")
        legacy.execute("INSERT INTO vec_messages(rowid,embedding) VALUES (1,?)", (struct.pack("2f", 1., 0.),))
        legacy.commit()
        original_columns = legacy.execute("PRAGMA table_info(messages)").fetchall()
        self.assertEqual(self.vector.build_vectors(legacy, txn_batch=1), 1)
        self.assertEqual(legacy.execute("SELECT rowid FROM vec_messages").fetchall(), [(1,)])
        self.assertFalse(legacy.in_transaction)
        self.assertEqual(legacy.execute("PRAGMA table_info(messages)").fetchall(), original_columns)
        self.assertIsNone(legacy.execute("SELECT name FROM sqlite_master WHERE name='maintenance_source_state'").fetchone())
        before = self.source.capture_source(legacy)
        self.assertEqual(before.tracker, "rebuild-bridge-v1")
        migrated = self.modules["storage"].create_db(legacy_path)
        self.addCleanup(migrated.close)
        with self.assertRaises(self.source.RebuildSourceChanged):
            before.check(migrated)
        self.assertEqual(self.vector.build_vectors(migrated, txn_batch=1), 1)
        self.assertEqual(self.source.capture_source(migrated).tracker, "rebuild-bridge-v1")
        self.assertEqual(len(self.source._installed_triggers(migrated, self.source._BRIDGE_TRIGGERS)), 3)
        self.modules["maintenance"].read_source_revision(migrated)

    def test_schema_replacement_during_compute_rejects_target_identity(self):
        self.seed()
        writer = self.writer()
        changed = []
        def encode(texts):
            if not changed:
                changed.append(True)
                writer.execute("CREATE TABLE synthetic_new_schema(value TEXT)")
                writer.commit()
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaisesRegex(self.source.RebuildSourceChanged, "schema changed"):
            self.vector.build_vectors(self.conn, txn_batch=1)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.manifest().consumed, 0)

    def test_model_fence_covers_initial_group_final_and_empty_commits(self):
        self.seed((1, 2, 3))
        real = self.conn
        fixture = self
        class Observed:
            commits = 0
            def __getattr__(self, name):
                return getattr(real, name)
            def commit(self):
                acquired = fixture.vector._lock.acquire(blocking=False)
                if acquired:
                    fixture.vector._lock.release()
                fixture.assertFalse(acquired, "Model identity was released before publication commit")
                self.commits += 1
                real.commit()
        observed = Observed()
        self.vector.build_vectors(observed, txn_batch=1)
        self.assertEqual(observed.commits, 4)
        real.execute("DELETE FROM messages")
        real.commit()
        self.vector.build_vectors(observed, txn_batch=1)
        self.assertEqual(observed.commits, 5)

    def test_competing_model_switch_waits_until_actual_final_commit(self):
        self.seed((1, 2))
        underlying = threading.Lock()
        attempted = threading.Event()
        switched = threading.Event()
        threads = []
        fixture = self
        class ObservedLock:
            def acquire(self, *args, **kwargs):
                if threading.current_thread().name == "synthetic-model-switch":
                    attempted.set()
                return underlying.acquire(*args, **kwargs)
            def release(self):
                underlying.release()
            def __enter__(self):
                self.acquire()
                return self
            def __exit__(self, *_args):
                self.release()
        self.vector._lock = ObservedLock()
        real = self.conn
        def switch():
            with fixture.vector._lock:
                fixture.vector._model_generation += 1
                fixture.vector.EMBEDDING_MODEL = "synthetic-next-model"
                switched.set()
        class CommitBoundary:
            def __getattr__(self, name):
                return getattr(real, name)
            def commit(self):
                manifest = fixture.manifest()
                if manifest is not None and manifest.complete:
                    thread = threading.Thread(target=switch, name="synthetic-model-switch")
                    threads.append(thread)
                    thread.start()
                    fixture.assertTrue(attempted.wait(3))
                    fixture.assertFalse(switched.is_set(), "Model switched before final commit")
                real.commit()
        try:
            self.vector.build_vectors(CommitBoundary(), txn_batch=1)
        finally:
            for thread in threads:
                thread.join(3)
        self.assertEqual(len(threads), 1)
        self.assertFalse(threads[0].is_alive())
        self.assertTrue(switched.is_set())
        self.assertTrue(self.manifest().complete)
        self.assertEqual(self.manifest().model, "synthetic-model")
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='embed_model'").fetchone()[0], "synthetic-model")

    def test_caller_writer_acquisition_precedes_model_fence_and_release_is_fenced(self):
        self.seed((1, 2))
        real = self.conn
        real.execute("INSERT INTO metadata(key,value) VALUES ('caller','pending')")
        fixture = self
        releases = []
        class Observed:
            def __init__(self):
                self.savepoints = []
            def __getattr__(self, name):
                return getattr(real, name)
            def execute(self, sql, parameters=()):
                if sql == "SAVEPOINT rebuild_source":
                    self.savepoints.append(False)
                if sql == "UPDATE messages SET id = id WHERE 0":
                    acquired = fixture.vector._lock.acquire(blocking=False)
                    fixture.assertTrue(acquired, "Writer acquisition must precede model fence")
                    fixture.vector._lock.release()
                    self.savepoints[-1] = True
                if sql == "RELEASE SAVEPOINT rebuild_source":
                    publication = self.savepoints.pop()
                    if publication:
                        acquired = fixture.vector._lock.acquire(blocking=False)
                        if acquired:
                            fixture.vector._lock.release()
                        fixture.assertFalse(acquired, "Publication RELEASE must retain the model fence")
                        releases.append(True)
                return real.execute(sql, parameters)
        self.vector.build_vectors(Observed(), txn_batch=1)
        self.assertGreaterEqual(len(releases), 3)
        self.assertTrue(real.in_transaction)
        real.rollback()
        self.assertEqual(self.ids(), [])


@unittest.skipUnless(os.environ.get("TRUEMEMORY_TEST_NATIVE_VEC") == "1",
                     "Actual sqlite-vec verification is an explicit isolated-host gate")
class TestStreamingSQLiteVec(TestStreamingBuilders):
    def setUp(self):
        super().setUp()
        import sqlite_vec
        self.sqlite_vec = sqlite_vec
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        for table in ("vec_messages", "vec_messages_sep"):
            self.conn.execute(f"DROP TABLE {table}")
            self.conn.execute(f"CREATE VIRTUAL TABLE {table} USING vec0(embedding float[2] distance_metric=cosine)")
        self.conn.commit()

    def writer(self):
        conn = super().writer()
        conn.enable_load_extension(True)
        self.sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn


class TestLegacyRebuildBridge(unittest.TestCase):
    setUp = TestStreamingBuilders.setUp

    def legacy(self, *, minimal=False, unique_content=False, ids=(-3, 0, 2, 5), cached_statements=0):
        path = Path(self.temp.name) / f"synthetic-legacy-{len(getattr(self, 'legacy_paths', []))}.sqlite"
        self.legacy_paths = [*getattr(self, "legacy_paths", []), path]
        conn = sqlite3.connect(path, cached_statements=cached_statements)
        self.addCleanup(conn.close)
        fields = "id INTEGER PRIMARY KEY,content TEXT"
        if unique_content:
            fields += " UNIQUE"
        if not minimal:
            fields += ",sender TEXT,recipient TEXT,timestamp TEXT"
        conn.execute(f"CREATE TABLE messages({fields})")
        conn.executescript("CREATE TABLE vec_messages(embedding BLOB); CREATE TABLE vec_messages_sep(embedding BLOB);")
        for mid in ids:
            if minimal:
                conn.execute("INSERT INTO messages VALUES(?,?)", (mid, f"synthetic-{mid}"))
            else:
                conn.execute("INSERT INTO messages VALUES(?,?,?,?,?)", (mid, f"synthetic-{mid}", "sender", "recipient", "2026-01-02"))
        conn.commit()
        return conn, path

    def reopen(self, path):
        conn = sqlite3.connect(path)
        self.addCleanup(conn.close)
        return conn

    def test_original_minimal_separation_shape_reaches_encoding_without_full_migration(self):
        conn, _ = self.legacy(ids=range(1, 6))
        columns = conn.execute("PRAGMA table_info(messages)").fetchall()
        self.assertEqual(self.vector.build_separation_vectors(conn, txn_batch=1), 5)
        self.assertEqual(sum(self.calls, []), [f"sender to recipient on 2026-01-02: synthetic-{mid}" for mid in range(1, 6)])
        self.assertEqual(conn.execute("PRAGMA table_info(messages)").fetchall(), columns)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(tables, {"messages", "vec_messages", "vec_messages_sep", "metadata", self.source._BRIDGE_TABLE})
        self.assertEqual(self.source.capture_source(conn).tracker, "rebuild-bridge-v1")

    def test_id_content_only_completion_preserves_negative_zero_and_hole_ids(self):
        conn, _ = self.legacy(minimal=True)
        self.assertEqual(self.vector.build_vectors(conn, txn_batch=1), 4)
        self.assertEqual(conn.execute("SELECT rowid FROM vec_messages ORDER BY rowid").fetchall(), [(-3,), (0,), (2,), (5,)])
        first = self.source.capture_source(conn)
        self.source.ensure_rebuild_tracking(conn)
        self.assertEqual(first, self.source.capture_source(conn))
        self.assertEqual(len(conn.execute("PRAGMA table_info(messages)").fetchall()), 2)

    def test_bridge_install_and_successful_build_preserve_caller_transaction(self):
        conn, _ = self.legacy()
        conn.execute("CREATE TABLE synthetic_caller(value TEXT)")
        conn.execute("INSERT INTO synthetic_caller VALUES('pending')")
        self.assertEqual(self.vector.build_vectors(conn, txn_batch=1), 4)
        self.assertTrue(conn.in_transaction)
        self.assertEqual(conn.execute("SELECT value FROM synthetic_caller").fetchone(), ("pending",))
        conn.rollback()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0], 0)
        self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.source._BRIDGE_TABLE,)).fetchone())
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM synthetic_caller").fetchone()[0], 0)

    def test_bridge_resume_across_connection_uses_consumed_cursor(self):
        conn, path = self.legacy()
        def encode(texts):
            if len(self.calls) == 2:
                raise RuntimeError("synthetic interruption")
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            self.vector.build_vectors(conn, txn_batch=1)
        key = self.source.manifest_key(("vec_messages",))
        before = self.source.load_manifest(conn, key)
        self.assertEqual((before.cursor, before.consumed), (0, 2))
        self.on_encode = None
        self.calls.clear()
        reopened = self.reopen(path)
        self.assertEqual(self.vector.build_vectors(reopened, txn_batch=1), 2)
        after = self.source.load_manifest(reopened, key)
        self.assertEqual(before.generation, after.generation)
        self.assertEqual(sum(self.calls, []), ["synthetic-2", "synthetic-5"])

    def test_bridge_rejects_concurrent_corrections_deletes_low_inserts_and_replace(self):
        mutations = ("UPDATE messages SET content='synthetic-corrected' WHERE id=2",
                     "UPDATE messages SET sender='synthetic-new-sender' WHERE id=2",
                     "UPDATE messages SET rowid=3 WHERE id=2", "DELETE FROM messages WHERE id=2",
                     "INSERT INTO messages(id,content) VALUES(1,'synthetic-hole')",
                     "INSERT OR REPLACE INTO messages(id,content) VALUES(2,'synthetic-replacement')")
        for statement in mutations:
            with self.subTest(statement=statement):
                conn, path = self.legacy()
                other = self.reopen(path)
                other.execute("PRAGMA recursive_triggers=OFF")
                changed = []
                def encode(texts):
                    self.assertFalse(conn.in_transaction)
                    if not changed:
                        changed.append(True)
                        other.execute(statement)
                        other.commit()
                    return [[1., 0.] for _ in texts]
                self.on_encode = encode
                with self.assertRaises(self.source.RebuildSourceChanged):
                    self.vector.build_vectors(conn, txn_batch=1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0], 0)

    def test_bridge_allows_high_id_append_outside_captured_range(self):
        conn, path = self.legacy()
        other = self.reopen(path)
        changed = []
        def encode(texts):
            if not changed:
                changed.append(True)
                other.execute("INSERT INTO messages(id,content) VALUES(9,'synthetic-tail')")
                other.commit()
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        self.assertEqual(self.vector.build_vectors(conn, txn_batch=1), 4)
        self.assertEqual(conn.execute("SELECT rowid FROM vec_messages ORDER BY rowid").fetchall(), [(-3,), (0,), (2,), (5,)])
        manifest = self.source.load_manifest(conn, self.source.manifest_key(("vec_messages",)))
        self.assertEqual((manifest.source.high_id, manifest.consumed), (5, 4))
        manifest.source.check(conn)

    def test_bridge_noop_and_derived_only_writes_do_not_dirty_source(self):
        conn, _ = self.legacy()
        conn.execute("ALTER TABLE messages ADD COLUMN episode_id INTEGER")
        self.source.ensure_rebuild_tracking(conn)
        before = self.source.capture_source(conn)
        conn.execute("UPDATE messages SET content=content,episode_id=99")
        conn.commit()
        self.assertEqual(before.revision, self.source.capture_source(conn).revision)

    def test_missing_replaced_trigger_new_column_or_missing_state_rotates_epoch(self):
        for fault in ("missing_trigger", "replaced_trigger", "new_column", "missing_state"):
            with self.subTest(fault=fault):
                conn, _ = self.legacy()
                self.source.ensure_rebuild_tracking(conn)
                before = self.source.capture_source(conn)
                trigger = self.source._BRIDGE_TRIGGERS[1]
                if fault in ("missing_trigger", "replaced_trigger"):
                    conn.execute(f"DROP TRIGGER {trigger}")
                    if fault == "replaced_trigger":
                        conn.execute(f"CREATE TRIGGER {trigger} AFTER UPDATE ON messages BEGIN SELECT 1; END")
                    conn.execute("UPDATE messages SET content='synthetic-untracked' WHERE id=2")
                elif fault == "new_column":
                    conn.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")
                else:
                    conn.execute(f"DELETE FROM {self.source._BRIDGE_TABLE}")
                conn.commit()
                with self.assertRaises(self.source.RebuildSourceChanged):
                    before.check(conn)
                self.source.ensure_rebuild_tracking(conn)
                after = self.source.capture_source(conn)
                self.assertNotEqual(before.revision.epoch, after.revision.epoch)
                with self.assertRaises(self.source.RebuildSourceChanged):
                    before.check(conn)

    def test_rolled_back_bridge_source_change_does_not_invalidate(self):
        conn, _ = self.legacy()
        self.source.ensure_rebuild_tracking(conn)
        before = self.source.capture_source(conn)
        conn.execute("UPDATE messages SET content='synthetic-rolled-back' WHERE id=2")
        conn.rollback()
        self.assertEqual(before, self.source.capture_source(conn))

    def test_denied_setup_or_release_preserves_target_and_caller_data_before_model(self):
        # Python 3.10 keeps at least five cached statements even when given zero.
        for fault, cached_statements in ((fault, size) for fault in ("ddl", "release") for size in (0, 5, 100)):
            with self.subTest(fault=fault, cached_statements=cached_statements):
                conn, _ = self.legacy(cached_statements=cached_statements)
                conn.execute("INSERT INTO vec_messages(rowid,embedding) VALUES (99,?)", (struct.pack("2f", 1., 0.),))
                conn.execute("CREATE TABLE synthetic_caller(value TEXT)")
                conn.commit()
                conn.execute("INSERT INTO synthetic_caller VALUES('pending')")
                denied = []
                def authorize(action, first, second, _database, _source):
                    is_target = ((fault == "ddl" and action == sqlite3.SQLITE_CREATE_TRIGGER)
                                 or (fault == "release" and action == sqlite3.SQLITE_SAVEPOINT and first == "RELEASE"
                                     and second == "rebuild_source"))
                    if is_target and not denied:
                        denied.append(True)
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK
                bridge_table = self.source._BRIDGE_TABLE
                armed = []

                class Probe:
                    def __getattr__(self, name: str) -> object:
                        return getattr(conn, name)

                    def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                        if fault == "release" and sql == "RELEASE SAVEPOINT rebuild_source" and not armed:
                            installed = conn.execute(
                                "SELECT 1 FROM sqlite_master WHERE name=?", (bridge_table,),
                            ).fetchone() is not None
                            if installed:
                                # Authorizers run at prepare time. Re-arm outside
                                # the callback so cached RELEASE is reauthorized.
                                conn.set_authorizer(authorize)
                                armed.append(True)
                        return conn.execute(sql, *args)

                if fault == "ddl":
                    conn.set_authorizer(authorize)
                before_calls = len(self.calls)
                try:
                    with self.assertRaises(sqlite3.DatabaseError):
                        self.vector.build_vectors(Probe(), txn_batch=1)
                finally:
                    conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                self.assertEqual(denied, [True])
                self.assertEqual(armed, [True] if fault == "release" else [])
                self.assertTrue(conn.in_transaction)
                self.assertEqual(len(self.calls), before_calls)
                conn.commit()
                self.assertEqual(conn.execute("SELECT value FROM synthetic_caller").fetchone(), ("pending",))
                self.assertEqual(conn.execute("SELECT rowid FROM vec_messages").fetchall(), [(99,)])
                self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.source._BRIDGE_TABLE,)).fetchone())

    def test_bridge_commit_failure_rolls_back_install_before_model_or_clear(self):
        conn, _ = self.legacy()
        conn.execute("INSERT INTO vec_messages(rowid,embedding) VALUES (99,?)", (struct.pack("2f", 1., 0.),))
        conn.commit()
        class Probe:
            def __getattr__(self, name):
                return getattr(conn, name)
            def commit(self):
                raise sqlite3.OperationalError("synthetic setup commit failure")
        with self.assertRaisesRegex(sqlite3.OperationalError, "synthetic setup commit failure"):
            self.vector.build_vectors(Probe(), txn_batch=1)
        self.assertEqual(self.calls, [])
        conn.commit()
        self.assertEqual(conn.execute("SELECT rowid FROM vec_messages").fetchall(), [(99,)])
        self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.source._BRIDGE_TABLE,)).fetchone())

    def test_readonly_legacy_install_fails_before_model_or_clear(self):
        conn, path = self.legacy()
        readonly = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.addCleanup(readonly.close)
        with self.assertRaises(sqlite3.OperationalError):
            self.vector.build_vectors(readonly, txn_batch=1)
        self.assertEqual(self.calls, [])
        self.assertFalse(readonly.in_transaction)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 4)

    def test_unsupported_source_schema_fails_before_model(self):
        conn, _ = self.legacy()
        conn.execute("ALTER TABLE messages RENAME TO synthetic_old_messages")
        conn.execute("CREATE TABLE messages(id TEXT PRIMARY KEY,content TEXT)")
        with self.assertRaisesRegex(self.modules["maintenance"].MaintenanceUnavailableError, "INTEGER PRIMARY KEY"):
            self.vector.build_vectors(conn, txn_batch=1)
        self.assertEqual(self.calls, [])

    def test_empty_legacy_generation_never_loads_a_model(self):
        conn, _ = self.legacy(ids=())
        def no_model():
            self.fail("Empty legacy build loaded a model")
        self.vector.get_model = no_model
        self.assertEqual(self.vector.build_vectors(conn, txn_batch=1), 0)
        manifest = self.source.load_manifest(conn, self.source.manifest_key(("vec_messages",)))
        self.assertTrue(manifest.complete)
        self.assertEqual((manifest.source.tracker, manifest.total), ("rebuild-bridge-v1", 0))

    def test_bridge_schema_change_during_encode_rejects_publication_and_restarts(self):
        for fault in ("drop_trigger", "add_source_column"):
            with self.subTest(fault=fault):
                conn, path = self.legacy()
                other = self.reopen(path)
                changed = []
                def encode(texts):
                    if not changed:
                        changed.append(True)
                        if fault == "drop_trigger":
                            other.execute(f"DROP TRIGGER {self.source._BRIDGE_TRIGGERS[1]}")
                            other.execute("UPDATE messages SET content='synthetic-untracked-correction' WHERE id=2")
                        else:
                            other.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")
                        other.commit()
                    return [[1., 0.] for _ in texts]
                self.on_encode = encode
                with self.assertRaises(self.source.RebuildSourceChanged):
                    self.vector.build_vectors(conn, txn_batch=1)
                key = self.source.manifest_key(("vec_messages",))
                before = self.source.load_manifest(conn, key)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0], 0)
                self.on_encode = None
                self.calls.clear()
                self.assertEqual(self.vector.build_vectors(conn, txn_batch=1), 4)
                after = self.source.load_manifest(conn, key)
                self.assertNotEqual(before.source.revision.epoch, after.source.revision.epoch)
                self.assertNotEqual(before.generation, after.generation)
                self.assertEqual(len(sum(self.calls, [])), 4)

    def test_canonical_handoff_does_not_resume_an_incomplete_bridge_prefix(self):
        conn = self.conn
        conn.executemany("INSERT INTO messages(id,content) VALUES (?,?)",
                         [(mid, f"synthetic-{mid}") for mid in (-3, 0, 2, 5)])
        conn.execute("UPDATE maintenance_source_state SET tracking_ready=0")
        conn.commit()
        def encode(texts):
            if len(self.calls) == 2:
                raise RuntimeError("synthetic interruption")
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            self.vector.build_vectors(conn, txn_batch=1)
        key = self.source.manifest_key(("vec_messages",))
        before = self.source.load_manifest(conn, key)
        self.modules["storage"]._initialize_maintenance_tracking(conn)
        self.on_encode = None
        self.calls.clear()
        self.assertEqual(self.vector.build_vectors(conn, txn_batch=1), 4)
        after = self.source.load_manifest(conn, key)
        self.assertNotEqual(before.generation, after.generation)
        self.assertEqual((before.source.tracker, after.source.tracker), ("rebuild-bridge-v1", "canonical-v1"))
        self.assertEqual(sum(self.calls, []), ["synthetic--3", "synthetic-0", "synthetic-2", "synthetic-5"])
        self.assertEqual(self.source._installed_triggers(conn, self.source._BRIDGE_TRIGGERS), {})
        self.assertEqual(conn.execute(f"SELECT tracking_ready FROM {self.source._BRIDGE_TABLE}").fetchone()[0], 0)

    def test_actual_deprecated_open_rebuilds_existing_legacy_connection(self):
        import warnings
        from unittest.mock import patch
        conn, path = self.legacy()
        tree = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
        method = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "open")
        schedule = next(node for node in original.body
                        if isinstance(node, ast.FunctionDef) and node.name == "_maybe_auto_consolidate")
        cls = ast.ClassDef(name="LegacyEngine", bases=[], keywords=[], body=[method, schedule], decorator_list=[])
        self.vector._check_embedder_compatibility = lambda _conn: None
        self.vector.vectors_are_built = lambda _conn, _table: False
        namespace = dict(__builtins__=self.vector.__dict__["__builtins__"],
                         warnings=warnings, sqlite3=sqlite3, DEFAULT_BUSY_TIMEOUT_MS=1000,
                         logger=logging.getLogger(__name__), _HAS_VECTOR=True, _HAS_STYLE_VEC=False,
                         _HAS_HYBRID=False, _HAS_CLUSTERING=False, resolve_tier=lambda: "base",
                         _check_rebuild_allowed=lambda _conn: None, init_vec_table=lambda _conn: None,
                         build_vectors=self.vector.build_vectors,
                         TrueMemoryMigrationError=type("SyntheticMigrationError", (RuntimeError,), {}))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "actual-legacy-open", "exec"), namespace)
        engine = namespace["LegacyEngine"]()
        engine.db_path, engine.stats = path, {}
        engine._write_lock = threading.Lock()
        engine._has_consolidation = engine._has_style_vec = False
        engine._purge_legacy_entity_profile_summaries = lambda: None
        fake_vec = types.ModuleType("sqlite_vec")
        fake_vec.load = lambda _conn: None
        with patch.dict(sys.modules, {"sqlite_vec": fake_vec}), warnings.catch_warnings(record=True):
            self.assertIs(engine.open(), engine)
        self.addCleanup(engine.conn.close)
        self.assertTrue(engine._has_vectors)
        self.assertTrue(engine.ready)
        self.assertEqual(engine.stats["message_count"], 4)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0], 4)
        self.assertEqual(self.source.capture_source(engine.conn).tracker, "rebuild-bridge-v1")
        self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_source_state'").fetchone())

    def test_custom_unique_replacements_invalidate_conservative_bridge_even_with_canonical_ready(self):
        for statement in (
            "INSERT OR REPLACE INTO messages(id,content,synthetic_unique) VALUES(9,'synthetic-high-replacement','slot-2')",
            "UPDATE OR REPLACE messages SET synthetic_unique='slot-2' WHERE id=1",
        ):
            with self.subTest(statement=statement):
                path = Path(self.temp.name) / ("synthetic-canonical-insert.sqlite" if statement.startswith("INSERT")
                                              else "synthetic-canonical-update.sqlite")
                conn = self.modules["storage"].create_db(path)
                self.addCleanup(conn.close)
                conn.execute("ALTER TABLE messages ADD COLUMN synthetic_unique TEXT")
                conn.execute("CREATE UNIQUE INDEX synthetic_unique_key ON messages(synthetic_unique)")
                conn.executemany("INSERT INTO messages(id,content,synthetic_unique) VALUES(?,?,?)",
                                 [(1, "synthetic-first", "slot-1"), (2, "synthetic-second", "slot-2")])
                conn.commit()
                other = self.reopen(path)
                other.execute("PRAGMA recursive_triggers=OFF")
                canonical_before = self.modules["maintenance"].read_source_revision(conn)
                changed = []
                def encode(texts):
                    if not changed:
                        changed.append(True)
                        other.execute(statement)
                        other.commit()
                    return [[1., 0.] for _ in texts]
                self.on_encode = encode
                conn.executescript("CREATE TABLE vec_messages(embedding BLOB); CREATE TABLE vec_messages_sep(embedding BLOB);")
                with self.assertRaises(self.source.RebuildSourceChanged):
                    self.vector.build_vectors(conn, txn_batch=1)
                # Canonical tracking also fences these UNIQUE writes; the
                # rebuild retains its independently scoped bridge certificate.
                canonical_after = self.modules["maintenance"].read_source_revision(conn)
                self.assertFalse(canonical_after.is_append_only_since(canonical_before))
                manifest = self.source.load_manifest(conn, self.source.manifest_key(("vec_messages",)))
                self.assertEqual(manifest.source.tracker, "rebuild-bridge-conservative-v1")
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0], 0)

    def test_expression_partial_unique_index_is_conservative_and_index_changes_rotate_epoch(self):
        conn, _ = self.legacy()
        conn.execute("ALTER TABLE messages ADD COLUMN synthetic_unique TEXT")
        self.source.ensure_rebuild_tracking(conn)
        normal = self.source.capture_source(conn)
        conn.execute("CREATE UNIQUE INDEX synthetic_unique_expression ON messages(lower(synthetic_unique)) WHERE synthetic_unique IS NOT NULL")
        conn.commit()
        with self.assertRaises(self.source.RebuildSourceChanged):
            normal.check(conn)
        self.source.ensure_rebuild_tracking(conn)
        conservative = self.source.capture_source(conn)
        self.assertEqual(conservative.tracker, "rebuild-bridge-conservative-v1")
        self.assertNotEqual(normal.revision.epoch, conservative.revision.epoch)
        conn.execute("INSERT INTO messages(id,content) VALUES(99,'synthetic-high-append')")
        conn.commit()
        with self.assertRaises(self.source.RebuildSourceChanged):
            conservative.check(conn)

    def test_canonical_handoff_retires_only_bridge_owned_triggers(self):
        conn = self.conn
        conn.execute("UPDATE maintenance_source_state SET tracking_ready=0")
        conn.commit()
        self.source.ensure_rebuild_tracking(conn)
        conn.execute("CREATE TRIGGER synthetic_unrelated AFTER UPDATE ON messages BEGIN SELECT 1; END")
        conn.commit()
        self.modules["storage"]._initialize_maintenance_tracking(conn)
        self.source.ensure_rebuild_tracking(conn)
        self.assertEqual(self.source.capture_source(conn).tracker, "canonical-v1")
        self.assertEqual(self.source._installed_triggers(conn, self.source._BRIDGE_TRIGGERS), {})
        self.assertIsNotNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name='synthetic_unrelated'").fetchone())

    def test_unowned_reserved_trigger_is_not_replaced_or_silently_adopted(self):
        conn, _ = self.legacy()
        trigger = self.source._BRIDGE_TRIGGERS[0]
        conn.execute(f"CREATE TRIGGER {trigger} AFTER INSERT ON messages BEGIN SELECT 1; END")
        original = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (trigger,)).fetchone()[0]
        with self.assertRaisesRegex(self.modules["maintenance"].MaintenanceUnavailableError, "already in use"):
            self.vector.build_vectors(conn)
        self.assertEqual(self.calls, [])
        self.assertEqual(conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (trigger,)).fetchone()[0], original)

    def test_schema_attestation_accepts_caller_row_factory(self):
        conn, _ = self.legacy()
        conn.row_factory = sqlite3.Row
        self.assertEqual(self.vector.build_vectors(conn, txn_batch=1), 4)
        self.assertEqual(self.source.capture_source(conn).tracker, "rebuild-bridge-v1")

    def test_unique_column_constraint_detects_hidden_replacement_deletion(self):
        conn, _ = self.legacy(unique_content=True)
        conn.execute("PRAGMA recursive_triggers=OFF")
        self.source.ensure_rebuild_tracking(conn)
        before = self.source.capture_source(conn)
        self.assertEqual(before.tracker, "rebuild-bridge-conservative-v1")
        conn.execute("INSERT OR REPLACE INTO messages(id,content) VALUES(99,'synthetic-2')")
        conn.commit()
        self.assertIsNone(conn.execute("SELECT 1 FROM messages WHERE id=2").fetchone())
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 4)
        with self.assertRaises(self.source.RebuildSourceChanged):
            before.check(conn)

    def test_indexed_integer_primary_key_replacement_on_rowid_invalidates_prefix(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY DESC, content TEXT)")
        conn.execute("INSERT INTO messages(rowid,id,content) VALUES(1,1,'synthetic-first')")
        conn.commit()
        conn.execute("PRAGMA recursive_triggers=OFF")
        self.source.ensure_rebuild_tracking(conn)
        before = self.source.capture_source(conn)
        self.assertEqual(before.tracker, "rebuild-bridge-conservative-v1")
        conn.execute("INSERT OR REPLACE INTO messages(rowid,id,content) VALUES(1,9,'synthetic-replacement')")
        conn.commit()
        self.assertEqual(conn.execute("SELECT id FROM messages").fetchall(), [(9,)])
        with self.assertRaises(self.source.RebuildSourceChanged):
            before.check(conn)

    def test_removing_custom_unique_index_enables_safe_canonical_handoff(self):
        conn = self.conn
        conn.execute("CREATE UNIQUE INDEX synthetic_content_key ON messages(content)")
        self.source.ensure_rebuild_tracking(conn)
        before = self.source.capture_source(conn)
        self.assertEqual(before.tracker, "rebuild-bridge-conservative-v1")
        conn.execute("DROP INDEX synthetic_content_key")
        conn.commit()
        with self.assertRaises(self.source.RebuildSourceChanged):
            before.check(conn)
        self.source.ensure_rebuild_tracking(conn)
        self.assertEqual(self.source.capture_source(conn).tracker, "canonical-v1")
        self.assertEqual(self.source._installed_triggers(conn, self.source._BRIDGE_TRIGGERS), {})

    def test_collation_equivalent_changes_reject_actual_builder_publication(self):
        for canonical_ready in (False, True):
            for collation, changed_text in (("NOCASE", "synthetic"), ("RTRIM", "Synthetic ")):
                with self.subTest(canonical_ready=canonical_ready, collation=collation):
                    path = Path(self.temp.name) / f"synthetic-collation-{canonical_ready}-{collation}.sqlite"
                    conn = sqlite3.connect(path)
                    self.addCleanup(conn.close)
                    if canonical_ready:
                        schema = self.modules["storage"]._SCHEMA_SQL.replace(
                            "content TEXT NOT NULL,", f"content TEXT COLLATE {collation} NOT NULL,", 1,
                        )
                        conn.executescript(schema)
                        self.modules["storage"]._initialize_maintenance_tracking(conn)
                    else:
                        conn.execute(f"CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT COLLATE {collation})")
                    conn.executescript("CREATE TABLE vec_messages(embedding BLOB); CREATE TABLE vec_messages_sep(embedding BLOB);")
                    conn.execute("INSERT INTO messages(id,content) VALUES(1,'Synthetic')")
                    conn.commit()
                    other = self.reopen(path)
                    if canonical_ready:
                        canonical_before = self.modules["maintenance"].read_source_revision(conn)
                    def encode(texts):
                        self.assertEqual(texts, ["Synthetic"])
                        self.assertFalse(conn.in_transaction)
                        other.execute("UPDATE messages SET content=? WHERE id=1", (changed_text,))
                        other.commit()
                        return [[1., 0.]]
                    self.on_encode = encode
                    with self.assertRaises(self.source.RebuildSourceChanged):
                        self.vector.build_vectors(conn, txn_batch=1)
                    if canonical_ready:
                        # Explicit BINARY comparison also invalidates the
                        # canonical source counters for this collated edit.
                        canonical_after = self.modules["maintenance"].read_source_revision(conn)
                        self.assertGreater(canonical_after.revision, canonical_before.revision)
                        self.assertGreater(canonical_after.correction_count, canonical_before.correction_count)
                        self.assertFalse(canonical_after.is_append_only_since(canonical_before))
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0], 0)
                    manifest = self.source.load_manifest(conn, self.source.manifest_key(("vec_messages",)))
                    self.assertEqual(manifest.source.tracker, "rebuild-bridge-v1")
                    self.assertEqual((manifest.complete, manifest.consumed), (False, 0))
                    self.on_encode = None
                    self.calls.clear()
                    self.assertEqual(self.vector.build_vectors(conn, txn_batch=1), 1)
                    self.assertEqual(self.calls, [[changed_text]])

    def test_bridge_compares_storage_type_when_blob_cast_bytes_match(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, content)")
        conn.execute("INSERT INTO messages VALUES(1,?)", ("synthetic",))
        conn.commit()
        self.source.ensure_rebuild_tracking(conn)
        before = self.source.capture_source(conn)
        conn.execute("UPDATE messages SET content=? WHERE id=1", (b"synthetic",))
        conn.commit()
        self.assertEqual(conn.execute("SELECT CAST(content AS BLOB) FROM messages").fetchone()[0], b"synthetic")
        with self.assertRaises(self.source.RebuildSourceChanged):
            before.check(conn)

    def test_standard_messages_definition_reuses_canonical_without_bridge(self):
        self.source.ensure_rebuild_tracking(self.conn)
        self.assertEqual(self.source.capture_source(self.conn).tracker, "canonical-v1")
        self.assertIsNone(self.conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.source._BRIDGE_TABLE,)).fetchone())

    def test_compatible_custom_definition_keeps_bridge_despite_ready_canonical(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        schema = self.modules["storage"]._SCHEMA_SQL.replace("sender TEXT DEFAULT '',", "sender TEXT DEFAULT '' COLLATE BINARY,", 1)
        conn.executescript(schema)
        self.modules["storage"]._initialize_maintenance_tracking(conn)
        self.source.ensure_rebuild_tracking(conn)
        self.assertEqual(self.source.capture_source(conn).tracker, "rebuild-bridge-v1")
        self.modules["maintenance"].read_source_revision(conn)


if __name__ == "__main__":
    unittest.main()
