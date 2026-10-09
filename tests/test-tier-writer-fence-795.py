"""Real in-memory writer transactions with synthetic selected records and models.

Certificate provenance and native vec0 are tested by their own gates. These
tests exercise actual journal parsing, SQL integrity, storage, Engine method
bodies, and vector writer control flow without a model or file-backed database.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import json
import logging
import runpy
import sqlite3
import sys
import threading
import types
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(Path(__file__).with_name("test-rebuild-source-streaming-756.py")))


class Cancelled(BaseException):
    pass


class Connection(sqlite3.Connection):
    armed = False
    mode = ""
    failure: BaseException | None = None
    commits = 0
    rollbacks = 0
    after_commit = None
    after_execute = None

    def execute(self, sql: str, parameters: tuple = ()):
        if hasattr(self, "statements"):
            self.statements.append(sql)
        if self.armed:
            if self.mode == "rollback_to" and sql.startswith("ROLLBACK TO SAVEPOINT tier_writer_"):
                raise self.failure
            if self.mode == "inner_before" and sql == "RELEASE SAVEPOINT rebuild_source":
                raise self.failure
            if self.mode == "guard_before" and sql.startswith("RELEASE SAVEPOINT tier_writer_"):
                raise self.failure
        result = super().execute(sql, parameters)
        if self.after_execute is not None:
            self.after_execute(sql, parameters)
        if self.armed:
            if self.mode == "inner_after" and sql == "RELEASE SAVEPOINT rebuild_source":
                raise self.failure
            if self.mode == "guard_after" and sql.startswith("RELEASE SAVEPOINT tier_writer_"):
                raise self.failure
        return result

    def commit(self) -> None:
        self.commits += 1
        if self.armed and self.mode == "commit_before":
            raise self.failure
        if self.armed and self.mode == "commit_retained":
            return
        super().commit()
        if self.after_commit is not None:
            callback, self.after_commit = self.after_commit, None
            callback()
        if self.armed and self.mode == "commit_after":
            raise self.failure

    def rollback(self) -> None:
        self.rollbacks += 1
        if self.armed and self.mode == "rollback_before":
            raise self.failure
        if self.armed and self.mode == "rollback_retained":
            return
        super().rollback()
        if self.armed and self.mode == "rollback_after":
            raise self.failure


class TestWriterFence(unittest.TestCase):
    def setUp(self) -> None:
        self.modules = BASE["load_modules"]()
        self.vector = self.modules["vector_search"]
        underlying = self.vector.__dict__["__builtins__"]["__import__"]

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"numpy", "torch", "model2vec", "sentence_transformers", "sqlite_vec"}:
                raise AssertionError("Native imports are excluded from writer tests")
            if name == "psutil":
                return types.SimpleNamespace()
            return underlying(name, globals, locals, fromlist, level)

        for name in ("embedding_target", "tier_config", "tier_switch.cache", "tier_switch.job",
                     "tier_switch.source", "tier_switch.activation"):
            module = types.ModuleType("synthetic_writer_" + name.replace(".", "_"))
            sys.modules[module.__name__] = module
            module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
            path = ROOT / "truememory" / (name.replace(".", "/") + ".py")
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
            self.modules[name] = module
        self.api = self.modules["tier_switch.writer"]
        self.activation = self.modules["tier_switch.activation"]
        self.conn = self.connect()
        self.conn.executescript(self.modules["storage"]._SCHEMA_SQL)
        for name in ("vec_messages", "vec_messages_sep", "vec_messages_custom", "vec_messages_sep_custom"):
            self.conn.execute(f"CREATE TABLE {name}(embedding BLOB)")
        self.conn.commit()
        self.calls = []
        self.encode_hook = lambda: None
        self.vector._model = object()
        self.vector.get_model = lambda: self.vector._model

        def encode(model, texts, **kwargs):
            self.calls.append(list(texts))
            self.encode_hook()
            return [[1., 0.] for _ in texts]

        self.vector._encode_with_mps_fallback = encode
        self.target = self.modules["embedding_target"].EmbeddingTarget("custom", "synthetic/encoder", 2, "custom")
        self.generation = 0
        self.addCleanup(patch.stopall)
        # No operation is admitted by the fixture unless a test explicitly
        # installs the runtime accessor contract. Never import the real bridge.
        patch.dict(sys.modules, {"truememory.tier_switch.runtime": types.SimpleNamespace(current_operation=lambda conn: None)}).start()

    def connect(self, uri: str | None = None):
        conn = sqlite3.connect(uri or ":memory:", uri=uri is not None, factory=Connection)
        conn.statements = []
        self.addCleanup(conn.close)
        return conn

    def publish(self, conn=None, *, target=None, acknowledged=True, commit=True):
        conn = self.conn if conn is None else conn
        target = target or self.target
        self.generation += 1
        generation = f"{self.generation:032x}"
        intent = self.activation.ActivationIntent(
            generation, None, target, "synthetic/reranker", generation,
            "rebuild-bridge-v1", "synthetic-epoch", "a" * 64,
            state="config_acknowledged" if acknowledged else "db_selected",
        )
        selected = self.activation.TierSelection(
            generation, generation, target, intent.reranker_id, target.tables, generation,
            intent.tracker, intent.source_epoch, intent.source_schema_signature,
            generation, "b" * 64, "c" * 64, "d" * 64, "e" * 64, 0, None, acknowledged,
        )
        self.activation._put(conn, "tier_activation_v1", self.activation._dump(intent))
        self.activation._put(conn, "tier_selected_v1", self.activation._dump(selected))
        self.activation._put(conn, "embed_model", target.model_id)
        self.activation._put(conn, "embed_dim", str(target.dimension))
        self.modules["tier_switch.cache"].VectorCacheRegistry.set(
            conn, target.tier_group, model_name=target.model_id, embedding_dim=target.dimension, commit=False,
        )
        if commit:
            conn.commit()
        return selected

    def selected_runtime(self):
        selected = self.publish()
        self.vector.EMBEDDING_MODEL = selected.target.model_id
        self.vector._embedding_dim = selected.target.dimension
        self.vector._active_vec_table = lambda conn: selected.tables[0]
        self.vector._active_sep_table = lambda conn: selected.tables[1]
        return selected

    def seed(self, mid=1, content="synthetic-original") -> None:
        self.conn.execute("INSERT INTO messages(id,content,sender) VALUES(?,?,?)", (mid, content, "synthetic-sender"))
        self.conn.commit()

    def contents(self, conn=None):
        conn = self.conn if conn is None else conn
        return conn.execute("SELECT id,content FROM messages ORDER BY id").fetchall()

    def engine(self, *, vectors=True):
        tree = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrueMemoryEngine")
        methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in {"add", "update", "delete", "delete_all"}]
        cls = ast.ClassDef(name="SyntheticEngine", bases=[], keywords=[], body=methods, decorator_list=[])
        constants = [n for n in tree.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {"_ALLOWED_TABLES", "_ALLOWED_COLUMNS", "_ALL_VEC_TABLES", "_SQLITE_IN_CHUNK"}
            for t in n.targets
        )]
        helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_delete_in_chunks", "_resolve_vec_tables"}]
        namespace = dict(__builtins__=self.vector.__dict__["__builtins__"], logger=logging.getLogger(__name__),
                         MAX_CONTENT_LENGTH=10000, sqlite3=sqlite3, engine_operation=lambda function: function,
                         insert_message=self.modules["storage"].insert_message,
                         update_message=self.modules["storage"].update_message,
                         delete_message=self.modules["storage"].delete_message)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[*constants, *helpers, cls], type_ignores=[])),
                     "actual-writer-engine", "exec"), namespace)
        engine = namespace["SyntheticEngine"]()
        engine.conn = self.conn
        engine._write_lock = threading.Lock()
        engine._ensure_connection = lambda: None
        engine._maybe_auto_consolidate = lambda: None
        engine._has_vectors = vectors
        engine._has_personality = engine._has_style_vec = False
        engine.get = lambda mid: self.modules["storage"].get_message(self.conn, mid)
        return engine

    def test_capture_is_frozen_read_only_and_connection_bound(self) -> None:
        before = self.conn.total_changes
        captured = self.api.capture_writer_selection(self.conn)
        self.assertIsNone(captured.selection)
        self.assertIs(captured.connection, self.conn)
        self.assertNotIn("Connection", repr(captured))
        self.assertEqual(self.conn.total_changes, before)
        self.assertFalse(self.conn.in_transaction)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            captured.selection = "changed"
        other = self.connect()
        other.executescript("CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT)")
        with self.assertRaises(self.api.WriterSelectionChanged):
            with self.api.writer_transaction(other, captured):
                other.execute("INSERT INTO messages VALUES(1,'wrong connection')")
        self.assertEqual(self.contents(other), [])

    def test_runtime_legacy_none_never_recaptures_first_selection(self) -> None:
        selected = self.selected_runtime()
        runtime = types.SimpleNamespace(current_operation=lambda conn: types.SimpleNamespace(selection=None))
        with patch.dict(sys.modules, {"truememory.tier_switch.runtime": runtime}):
            captured = self.api.capture_writer_selection(self.conn)
        self.assertIsNone(captured.selection)
        with self.assertRaises(self.api.WriterSelectionChanged):
            with self.api.writer_transaction(self.conn, captured):
                self.fail("Legacy admission cannot publish into selected generation")
        self.assertEqual(self.activation.read_activation_state(self.conn).selection, selected)

    def test_runtime_wrong_connection_refusal_propagates_without_fallback(self) -> None:
        def current(conn):
            raise RuntimeError("synthetic wrong accepted connection")
        with patch.dict(sys.modules, {"truememory.tier_switch.runtime": types.SimpleNamespace(current_operation=current)}):
            before = len(self.conn.statements)
            with self.assertRaisesRegex(RuntimeError, "wrong accepted connection"):
                self.api.capture_writer_selection(self.conn)
            self.assertEqual(len(self.conn.statements), before)

    def test_two_actual_handles_reject_first_selection_and_same_width_generation_change(self) -> None:
        uri = "file:synthetic-writer-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
        left, right = self.connect(uri), self.connect(uri)
        left.executescript(self.modules["storage"]._SCHEMA_SQL)
        captured = self.api.capture_writer_selection(left)
        first = self.publish(right)
        with self.assertRaises(self.api.WriterSelectionChanged):
            with self.api.writer_transaction(left, captured):
                left.execute("INSERT INTO messages(content) VALUES('stale legacy')")
        self.assertEqual(self.contents(left), [])
        captured = self.api.capture_writer_selection(left)
        second = self.publish(right)
        self.assertEqual(first.target.dimension, second.target.dimension)
        self.assertNotEqual(first.generation, second.generation)
        with self.assertRaises(self.api.WriterSelectionChanged):
            with self.api.writer_transaction(left, captured):
                left.execute("INSERT INTO messages(content) VALUES('stale selected')")
        self.assertEqual(self.contents(left), [])

    def test_borrowed_reader_cannot_upgrade_past_another_writer(self) -> None:
        uri = "file:synthetic-lock-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
        left, right = self.connect(uri), self.connect(uri)
        left.executescript(self.modules["storage"]._SCHEMA_SQL)
        right.execute("BEGIN IMMEDIATE")
        left.execute("BEGIN")
        captured = self.api.capture_writer_selection(left)
        with self.assertRaises(sqlite3.OperationalError):
            with self.api.writer_transaction(left, captured):
                left.execute("INSERT INTO messages(content) VALUES('must not mutate')")
        self.assertTrue(left.in_transaction)
        self.assertTrue(right.in_transaction)
        self.assertEqual(self.contents(left), [])
        right.rollback()
        left.rollback()

    def test_exact_model_dimension_pair_registry_and_metadata_guard(self) -> None:
        self.selected_runtime()
        captured = self.api.capture_writer_selection(self.conn)
        for options in ({"model_id": "synthetic/wrong"}, {"dimension": 3},
                        {"dimension": True}, {"tables": tuple(reversed(self.target.tables))}):
            with self.subTest(options=options), self.assertRaises(self.api.WriterSelectionChanged):
                with self.api.writer_transaction(self.conn, captured, **options):
                    self.fail("Mismatched output must not be admitted")
        for sql in ("UPDATE vector_cache_registry SET vec_table='wrong'",
                    "UPDATE metadata SET value='wrong' WHERE key='embed_model'",
                    "UPDATE metadata SET value='02' WHERE key='embed_dim'"):
            self.conn.execute("BEGIN")
            self.conn.execute(sql)
            with self.subTest(sql=sql), self.assertRaises(self.api.WriterSelectionChanged):
                with self.api.writer_transaction(self.conn, captured):
                    self.fail("Changed metadata must not be admitted")
            self.assertTrue(self.conn.in_transaction)
            self.conn.rollback()

    def test_acknowledgement_alone_does_not_change_writer_space(self) -> None:
        selected = self.publish(acknowledged=False)
        captured = self.api.capture_writer_selection(self.conn)
        self.activation.acknowledge_config(self.conn, generation=selected.generation)
        with self.api.writer_transaction(self.conn, captured):
            self.conn.execute("INSERT INTO messages(content) VALUES('same embedding space')")
        self.assertEqual(self.contents(), [(1, "same embedding space")])

    def test_borrowed_success_and_refusal_preserve_outer_work(self) -> None:
        captured = self.api.capture_writer_selection(self.conn)
        self.conn.execute("INSERT INTO messages(id,content) VALUES(11,'caller')")
        before = self.conn.commits
        with self.api.writer_transaction(self.conn, captured):
            self.conn.execute("INSERT INTO messages(id,content) VALUES(12,'operation')")
        self.assertEqual(self.conn.commits, before)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.contents(), [(11, "caller"), (12, "operation")])
        self.conn.rollback()
        self.assertEqual(self.contents(), [])
        self.conn.execute("INSERT INTO messages(id,content) VALUES(11,'caller')")
        self.publish(commit=False)
        with self.assertRaises(self.api.WriterSelectionChanged):
            with self.api.writer_transaction(self.conn, captured):
                self.fail("Stale capture must refuse before mutation")
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.contents(), [(11, "caller")])

    def test_capture_borrows_without_ending_or_upgrading_outer_transaction(self) -> None:
        self.conn.execute("INSERT INTO messages(id,content) VALUES(11,'caller')")
        before = self.conn.commits, self.conn.rollbacks
        self.assertIsNone(self.api.capture_writer_selection(self.conn).selection)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.contents(), [])

    def test_owned_commit_faults_before_after_and_retained_do_not_fake_success(self) -> None:
        for mode in ("commit_before", "commit_after", "commit_retained"):
            with self.subTest(mode=mode):
                captured = self.api.capture_writer_selection(self.conn)
                self.conn.armed, self.conn.mode, self.conn.failure = True, mode, Cancelled("synthetic commit fault")
                rollbacks = self.conn.rollbacks
                with self.assertRaises((Cancelled, self.api.WriterSelectionChanged)):
                    with self.api.writer_transaction(self.conn, captured):
                        self.conn.execute("INSERT INTO messages(id,content) VALUES(12,'operation')")
                self.conn.armed = False
                self.assertFalse(self.conn.in_transaction)
                if mode == "commit_after":
                    self.assertEqual(self.conn.rollbacks, rollbacks)
                    self.assertEqual(self.contents(), [(12, "operation")])
                    self.conn.execute("DELETE FROM messages")
                    self.conn.commit()
                else:
                    self.assertEqual(self.conn.rollbacks, rollbacks + 1)
                    self.assertEqual(self.contents(), [])

    def test_owned_uncertain_rollback_is_attempted_once(self) -> None:
        for mode in ("rollback_before", "rollback_after", "rollback_retained"):
            with self.subTest(mode=mode):
                captured = self.api.capture_writer_selection(self.conn)
                self.conn.armed, self.conn.mode = True, mode
                self.conn.failure = sqlite3.OperationalError("synthetic rollback uncertainty")
                before = self.conn.rollbacks
                try:
                    with self.assertRaises((sqlite3.OperationalError, self.api.WriterSelectionChanged)):
                        with self.api.writer_transaction(self.conn, captured):
                            self.conn.execute("INSERT INTO messages(id,content) VALUES(12,'operation')")
                            raise Cancelled("synthetic body cancellation")
                    self.assertEqual(self.conn.rollbacks, before + 1)
                    self.assertEqual(self.conn.in_transaction, mode != "rollback_after")
                finally:
                    self.conn.armed = False
                    self.conn.rollback()

    def test_repeated_caller_savepoint_survives_inner_release_before_and_after_fault(self) -> None:
        for mode in ("inner_before", "inner_after"):
            with self.subTest(mode=mode):
                captured = self.api.capture_writer_selection(self.conn)
                self.conn.execute("BEGIN")
                self.conn.execute("SAVEPOINT rebuild_source")
                self.conn.execute("INSERT INTO messages(id,content) VALUES(11,'caller')")
                self.conn.armed, self.conn.mode, self.conn.failure = True, mode, Cancelled("synthetic inner release fault")
                with self.assertRaises(Cancelled):
                    with self.api.writer_transaction(self.conn, captured):
                        self.conn.execute("INSERT INTO messages(id,content) VALUES(12,'operation')")
                self.conn.armed = False
                self.assertTrue(self.conn.in_transaction)
                self.assertEqual(self.contents(), [(11, "caller")])
                self.conn.execute("RELEASE SAVEPOINT rebuild_source")
                self.assertTrue(self.conn.in_transaction)
                self.conn.rollback()

    def test_ambiguous_final_guard_release_never_attempts_cleanup(self) -> None:
        for mode in ("guard_before", "guard_after"):
            with self.subTest(mode=mode):
                captured = self.api.capture_writer_selection(self.conn)
                self.conn.execute("INSERT INTO messages(id,content) VALUES(11,'caller')")
                self.conn.armed, self.conn.mode, self.conn.failure = True, mode, Cancelled("synthetic guard release fault")
                before = len(self.conn.statements)
                with self.assertRaises(Cancelled):
                    with self.api.writer_transaction(self.conn, captured):
                        self.conn.execute("INSERT INTO messages(id,content) VALUES(12,'operation')")
                tail = self.conn.statements[before:]
                self.assertTrue(tail[-1].startswith("RELEASE SAVEPOINT tier_writer_"))
                self.assertFalse(any(sql.startswith("ROLLBACK") for sql in tail))
                self.assertTrue(self.conn.in_transaction)
                self.assertEqual(self.contents(), [(11, "caller"), (12, "operation")])
                self.conn.armed = False
                self.conn.rollback()

    def test_borrowed_rollback_to_failure_never_releases_or_discards_outer_work(self) -> None:
        captured = self.api.capture_writer_selection(self.conn)
        self.conn.execute("INSERT INTO messages(id,content) VALUES(11,'caller')")
        self.conn.armed, self.conn.mode = True, "rollback_to"
        self.conn.failure = sqlite3.OperationalError("synthetic rollback-to failure")
        before = len(self.conn.statements)
        with self.assertRaises(sqlite3.OperationalError):
            with self.api.writer_transaction(self.conn, captured):
                self.conn.execute("INSERT INTO messages(id,content) VALUES(12,'operation')")
                raise Cancelled("synthetic cancellation")
        tail = self.conn.statements[before:]
        self.assertEqual(sum(sql.startswith("ROLLBACK TO") for sql in tail), 1)
        self.assertFalse(any(sql.startswith("RELEASE") for sql in tail))
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.contents(), [(11, "caller"), (12, "operation")])
        self.conn.armed = False
        self.conn.rollback()

    def test_add_stale_capture_refuses_even_when_native_preparation_fails(self) -> None:
        self.selected_runtime()
        def fail():
            self.publish()
            raise RuntimeError("synthetic encoder unavailable")
        self.encode_hook = fail
        with self.assertRaises(self.api.WriterSelectionChanged):
            self.engine().add("must not store")
        self.assertEqual(self.contents(), [])

    def test_update_stale_capture_refuses_before_source_change(self) -> None:
        self.selected_runtime()
        self.seed()
        fired = []
        def change():
            if not fired:
                fired.append(True)
                self.publish()
        self.encode_hook = change
        with self.assertRaises(self.api.WriterSelectionChanged):
            self.engine().update(1, "must not store")
        self.assertEqual(self.contents(), [(1, "synthetic-original")])

    def test_source_commit_then_selection_change_keeps_retry_wording(self) -> None:
        self.selected_runtime()
        self.seed()
        self.conn.after_commit = self.publish
        with self.assertRaisesRegex(self.vector.VectorPublicationChanged, "Source update committed; retry update to publish vectors"):
            self.engine().update(1, "committed-source")
        self.assertEqual(self.contents(), [(1, "committed-source")])
        self.assertEqual(self.conn.execute("SELECT * FROM vec_messages_custom").fetchall(), [])

    def test_delete_and_delete_all_reject_stale_admitted_operation_before_any_delete(self) -> None:
        selected = self.selected_runtime()
        self.seed()
        runtime = types.SimpleNamespace(current_operation=lambda conn: types.SimpleNamespace(selection=selected))
        self.publish()
        with patch.dict(sys.modules, {"truememory.tier_switch.runtime": runtime}):
            for operation in (lambda: self.engine().delete(1), lambda: self.engine().delete_all(),
                              lambda: self.engine().delete_all("synthetic-sender")):
                with self.assertRaises(self.api.WriterSelectionChanged):
                    operation()
                self.assertEqual(self.contents(), [(1, "synthetic-original")])

    def test_selected_delete_all_uses_exact_custom_pair_and_borrowed_rollback(self) -> None:
        self.selected_runtime()
        self.seed()
        for table in self.target.tables:
            self.conn.execute(f"INSERT INTO {table}(rowid,embedding) VALUES(1,x'0102')")
        self.conn.commit()
        self.conn.execute("INSERT INTO metadata(key,value) VALUES('caller','pending')")
        self.assertTrue(self.engine(vectors=False).delete_all("synthetic-sender"))
        self.assertEqual(self.contents(), [])
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.contents(), [(1, "synthetic-original")])
        for table in self.target.tables:
            self.assertEqual(self.conn.execute(f"SELECT rowid FROM {table}").fetchall(), [(1,)])
        self.assertTrue(self.engine(vectors=False).delete_all("synthetic-sender"))
        self.assertFalse(self.conn.in_transaction)
        for table in self.target.tables:
            self.assertEqual(self.conn.execute(f"SELECT rowid FROM {table}").fetchall(), [])

    def test_direct_embed_refuses_changed_selection_without_new_metadata(self) -> None:
        self.selected_runtime()
        self.seed()
        fired = []
        def change():
            if not fired:
                fired.append(self.publish())
        self.encode_hook = change
        with self.assertRaises(self.vector.VectorPublicationChanged):
            self.vector.embed_single(self.conn, 1, "synthetic-original")
        self.assertEqual(self.conn.execute("SELECT * FROM vec_messages_custom").fetchall(), [])
        self.assertEqual(self.activation.read_activation_state(self.conn).selection, fired[0])

    def test_legacy_crud_and_borrowed_update_delete_remain_default_compatible(self) -> None:
        engine = self.engine(vectors=False)
        item = engine.add("legacy")
        self.assertEqual(engine.update(item["id"], "legacy-updated")["content"], "legacy-updated")
        self.conn.execute("INSERT INTO metadata(key,value) VALUES('caller','pending')")
        engine.update(item["id"], "pending-update")
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.contents(), [(item["id"], "legacy-updated")])
        self.conn.execute("INSERT INTO metadata(key,value) VALUES('caller','pending')")
        self.assertTrue(engine.delete(item["id"]))
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.contents(), [(item["id"], "legacy-updated")])
        self.assertTrue(engine.delete(item["id"]))
        self.assertFalse(self.conn.in_transaction)

    def test_storage_defaults_commit_and_false_never_commits(self) -> None:
        self.seed()
        before = self.conn.commits
        self.modules["storage"].update_message(self.conn, 1, content="default")
        self.assertEqual(self.conn.commits, before + 1)
        self.modules["storage"].update_message(self.conn, 1, content="pending", commit=False)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.contents(), [(1, "default")])
        self.modules["storage"].delete_message(self.conn, 1, commit=False)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.contents(), [(1, "default")])
        self.modules["storage"].delete_message(self.conn, 1)
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.contents(), [])

    def test_bulk_prefix_cannot_recapture_new_generation(self) -> None:
        self.selected_runtime()
        self.seed()
        @contextmanager
        def owner(path):
            yield None
        maintenance = self.modules["maintenance"]
        fired = []
        def change():
            if not fired:
                fired.append(self.publish())
        self.encode_hook = change
        with patch.object(maintenance, "maintenance_owner", owner), patch.object(maintenance, "connection_database_path", lambda conn: Path("/synthetic/database")):
            with self.assertRaises(self.api.WriterSelectionChanged):
                self.vector.build_vectors(self.conn)
        self.assertEqual(self.conn.execute("SELECT * FROM vec_messages_custom").fetchall(), [])
        self.assertEqual(self.activation.read_activation_state(self.conn).selection, fired[0])

    def test_bulk_initial_and_empty_clear_refuse_stale_capture(self) -> None:
        for nonempty in (False, True):
            with self.subTest(nonempty=nonempty):
                fixture = TestWriterFence()
                fixture.setUp()
                try:
                    fixture.selected_runtime()
                    if nonempty:
                        fixture.seed()
                    fixture.conn.execute("INSERT INTO vec_messages_custom(rowid,embedding) VALUES(99,x'0102')")
                    fixture.conn.commit()
                    @contextmanager
                    def owner(path):
                        fixture.publish()
                        yield None
                    maintenance = fixture.modules["maintenance"]
                    with patch.object(maintenance, "maintenance_owner", owner), patch.object(maintenance, "connection_database_path", lambda conn: Path("/synthetic/database")):
                        with self.assertRaises(fixture.api.WriterSelectionChanged):
                            fixture.vector.build_vectors(fixture.conn)
                    self.assertEqual(fixture.calls, [])
                    self.assertEqual(fixture.conn.execute("SELECT rowid,embedding FROM vec_messages_custom").fetchall(), [(99, b"\x01\x02")])
                finally:
                    fixture.doCleanups()

    def test_bulk_finish_cannot_publish_completion_or_metadata_after_selection_change(self) -> None:
        self.selected_runtime()
        self.seed()
        @contextmanager
        def owner(path):
            yield None
        changed = []
        def observe(sql, parameters):
            if sql.startswith("INSERT OR REPLACE INTO metadata(key,value,updated_at)") and parameters[0].startswith("vec_source_v1:"):
                value = json.loads(parameters[1])
                if value["consumed"] == 1 and not value["complete"]:
                    self.conn.after_commit = lambda: changed.append(self.publish())
        self.conn.after_execute = observe
        maintenance = self.modules["maintenance"]
        with patch.object(maintenance, "maintenance_owner", owner), patch.object(maintenance, "connection_database_path", lambda conn: Path("/synthetic/database")):
            with self.assertRaises(self.api.WriterSelectionChanged):
                self.vector.build_vectors(self.conn)
        self.conn.after_execute = None
        self.assertEqual(len(changed), 1)
        self.assertEqual(self.conn.execute("SELECT rowid FROM vec_messages_custom").fetchall(), [(1,)])
        manifest = self.modules["rebuild_source"].load_manifest(self.conn, "vec_source_v1:vec_messages_custom")
        self.assertEqual(manifest.consumed, 1)
        self.assertFalse(manifest.complete)
        self.assertEqual(self.activation.read_activation_state(self.conn).selection, changed[0])


if __name__ == "__main__":
    unittest.main()
