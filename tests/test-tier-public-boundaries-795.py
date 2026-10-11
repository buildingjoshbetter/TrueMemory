"""Public tier boundaries with production AST, stdlib stubs and memory SQLite."""
from __future__ import annotations

import ast
import builtins
import contextlib
import dataclasses
import datetime
import enum
import gc
import logging
import os
from pathlib import Path
import runpy
import sqlite3
import sys
import threading
import time
import types
import unittest
import uuid
import weakref
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = runpy.run_path(str(Path(__file__).with_name("test-tier-runtime-bridge-795.py")))
_REAL_PATH_RESOLVE = Path.resolve


def enter_context(test_case: unittest.TestCase, context: contextlib.AbstractContextManager[object]) -> object:
    result = type(context).__enter__(context)
    test_case.addCleanup(type(context).__exit__, context, None, None, None)
    return result


def definitions(path: str, names: set[str], namespace: dict, *, methods: bool = False) -> types.ModuleType:
    if path == "maintenance.py" and "_acquire_owner" in names:
        names = names | {"_AutomaticAdmission", "_automatic_admission_locked"}
        namespace = {"threading": threading, "weakref": weakref, **namespace}
    tree = ast.parse((ROOT / "truememory" / path).read_text(encoding="utf-8"))
    if methods:
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    else:
        nodes = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
                 or isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = types.ModuleType("synthetic_public_" + path.replace("/", "_").replace(".", "_"))
    module.__dict__.update(namespace)
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), path, "exec"), module.__dict__)
    return module


class TestFixtureCompatibility(unittest.TestCase):
    def test_context_cleanup_works_without_testcase_enter_context(self):
        case = unittest.TestCase()
        entered, exited = [], []
        @contextlib.contextmanager
        def context():
            entered.append(True)
            try:
                yield "synthetic context"
            finally:
                exited.append(True)
        result = enter_context(case, context())
        self.assertEqual(result, "synthetic context")
        self.assertEqual(entered, [True])
        self.assertEqual(exited, [])
        case.doCleanups()
        self.assertEqual(exited, [True])

    def test_nested_setup_failure_restores_global_path_stat(self):
        warm = runpy.run_path(str(Path(__file__).with_name("test-tier-warm-admission-795.py")))
        active = warm["ACTIVE"]["TestActiveInitialization"]
        original = Path.stat
        def fail_setup(case):
            enter_context(case, patch.object(Path, "stat", return_value=types.SimpleNamespace()))
            raise RuntimeError("synthetic partial setup")
        case = warm["TestWarmAdmission"]()
        with patch.object(active, "setUp", fail_setup):
            try:
                with self.assertRaisesRegex(RuntimeError, "synthetic partial setup"):
                    case.setUp()
            finally:
                case.doCleanups()
        self.assertIs(Path.stat, original)


class TestPublicBoundaries(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = BRIDGE["TestRuntimeBridge"]()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.api, self.conn = self.fixture.api, self.fixture.conn
        self.modules = self.fixture.modules
        self.modules["maintenance"] = types.SimpleNamespace(
            maintenance_owner=lambda path: contextlib.nullcontext(), connection_database_path=lambda conn: None)
        enter_context(self, patch.dict(sys.modules, {"truememory.tier_switch.runtime": self.api}))

    def imports(self, name, globals=None, locals=None, fromlist=(), level=0):
        if name == "truememory.tier_switch.runtime":
            return self.api
        return self.fixture.safe_import(name, globals, locals, fromlist, level)

    def namespace(self, **values):
        return dict(__builtins__=dict(vars(builtins), __import__=self.imports), **values)

    def test_direct_selected_legacy_vector_mutators_refuse_before_writes(self) -> None:
        self.fixture.select()
        names = {"_write_embedder_metadata", "_write_embedder_metadata_no_commit", "_check_embedder_compatibility",
                 "init_vec_table", "_rebuild_table_as_cosine", "_finish_cosine_swap",
                 "migrate_to_cosine_metric", "migrate_legacy_vec_tables"}
        module = definitions("vector_search.py", names, self.namespace())
        queries = []
        self.conn.set_trace_callback(queries.append)
        for name in sorted(names):
            args = (self.conn, "vec_messages") if name == "_rebuild_table_as_cosine" else (
                (self.conn, "vec_messages", 256) if name == "_finish_cosine_swap" else (self.conn,))
            with self.subTest(name=name), self.assertRaises(self.api.TierRuntimeError):
                getattr(module, name)(*args)
        self.assertFalse(any(sql.lstrip().split(" ", 1)[0].upper() in {"INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"}
                             for sql in queries))
        self.assertFalse(self.conn.in_transaction)

    def test_legacy_guard_preserves_borrowed_transaction_and_source(self) -> None:
        self.conn.execute("INSERT INTO messages(id,content) VALUES(77,'synthetic pending')")
        before = (self.conn.commits, self.conn.rollbacks)
        self.api.require_legacy_vector_mutation(self.conn)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=77").fetchone()[0], "synthetic pending")

    def test_destructive_ingest_refuses_selected_before_create_or_unlink(self) -> None:
        self.fixture.select()
        create = Mock(side_effect=AssertionError("Unexpected create"))
        module = definitions("engine.py", {"ingest"}, self.namespace(Path=Path, create_db=create), methods=True)
        engine = types.SimpleNamespace(conn=self.conn, db_path=Path(":memory:"))
        maintenance = types.SimpleNamespace(maintenance_owner=lambda path: contextlib.nullcontext())
        self.modules["maintenance"] = maintenance
        with patch.object(Path, "unlink", side_effect=AssertionError("Unexpected unlink")) as unlink:
            with self.assertRaises(self.api.TierRuntimeError):
                module.ingest(engine, "synthetic-input")
        create.assert_not_called()
        unlink.assert_not_called()
        self.assertFalse(self.conn.in_transaction)

    def test_destructive_ingest_rejects_borrowed_before_any_cleanup(self) -> None:
        self.conn.execute("INSERT INTO messages(id,content) VALUES(88,'synthetic pending')")
        self.modules["maintenance"] = types.SimpleNamespace(maintenance_owner=lambda path: contextlib.nullcontext())
        before = (self.conn.commits, self.conn.rollbacks)
        with self.assertRaises(self.api.TierRuntimeError):
            with self.api.destructive_ingest_operation(types.SimpleNamespace(conn=self.conn, db_path=Path(":memory:"))):
                self.fail("Unexpected destructive body")
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)

    def test_selected_maintenance_declines_without_native_reconciliation(self) -> None:
        self.fixture.select()
        with self.api.maintenance_serving_operation(self.conn) as allowed:
            self.assertFalse(allowed)
        self.assertEqual(self.fixture.loads, [])

    def test_cluster_only_runner_defers_before_dependency_or_builder(self) -> None:
        self.fixture.select()
        events = []
        @contextlib.contextmanager
        def owner(path):
            events.append("owner")
            yield types.SimpleNamespace(generation="synthetic")
            events.append("released")
        real = self.api.maintenance_serving_operation
        @contextlib.contextmanager
        def serving(conn, **kwargs):
            self.assertEqual(events, ["owner"])
            with real(conn, **kwargs) as value:
                events.append("serving-check")
                yield value
        source = definitions("maintenance.py", {"LayerResult", "_maintenance_embedding_scope", "_selected_cluster_deferred", "run_layers"},
                             self.namespace(NamedTuple=__import__("typing").NamedTuple, contextmanager=contextlib.contextmanager,
                                            maintenance_owner=owner, connection_database_path=lambda conn: None))
        spec = types.SimpleNamespace(layer="clusters", connection=self.conn,
                                     resolve_dependency=Mock(side_effect=AssertionError("Unexpected dependency")),
                                     build=Mock(side_effect=AssertionError("Unexpected compute")))
        with patch.object(self.api, "maintenance_serving_operation", side_effect=serving):
            result = source.run_layers(self.conn, (spec,), force=True)
        self.assertEqual(result[0].outcome, "deferred")
        self.assertEqual(result[0].error_category, "SelectedVectorLayerUnsupported")
        self.assertFalse(result[0].attempted)
        self.assertEqual(events, ["owner", "serving-check", "released"])
        spec.resolve_dependency.assert_not_called()
        spec.build.assert_not_called()

    def test_generic_nonvector_scope_has_no_serving_import_or_model_work(self) -> None:
        module = definitions("maintenance.py", {"_maintenance_embedding_scope"}, self.namespace(contextmanager=contextlib.contextmanager))
        with patch.object(self.api, "maintenance_serving_operation", side_effect=AssertionError("Unexpected model scope")):
            with module._maintenance_embedding_scope(self.conn, False, None) as allowed:
                self.assertTrue(allowed)

    def gate(self, names):
        tree = ast.parse((ROOT / "truememory/ingest/encoding_gate.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EncodingGate")
        nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = self.namespace(log=logging.getLogger("synthetic-public"))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])), "actual-gate-methods", "exec"), namespace)
        return namespace

    def test_gate_generation_drops_old_native_reference_and_cached_search(self) -> None:
        refresh = self.gate({"_refresh_serving_generation"})["_refresh_serving_generation"]
        class Encoder:
            pass
        model = Encoder()
        previous = weakref.ref(model)
        gate = types.SimpleNamespace(_embed_model=model, _serving_key=("old",), _last_search_results=[{"content": "synthetic stale"}], _pe_available=False)
        del model
        with self.api.serving_operation(self.conn) as operation:
            refresh(gate)
            gc.collect()
            self.assertIsNone(previous())
            self.assertEqual(gate._serving_key, operation.key)
            self.assertEqual(gate._last_search_results, [])
            self.assertTrue(gate._pe_available)
            retained = object()
            gate._embed_model = retained
            refresh(gate)
            self.assertIs(gate._embed_model, retained)

    def test_typed_rejections_escape_gate_search_and_normal_availability_degrades(self) -> None:
        search = self.gate({"_search"})["_search"]
        for error in (self.api.TierRuntimeError("synthetic"), self.fixture.gate.ServingLeaseTimeout("synthetic")):
            gate = types.SimpleNamespace(user_id="", memory=types.SimpleNamespace(search=Mock(side_effect=error)))
            with self.assertRaises(type(error)) as raised:
                search(gate, "synthetic")
            self.assertIs(raised.exception, error)
        gate = types.SimpleNamespace(user_id="", memory=types.SimpleNamespace(search=Mock(side_effect=RuntimeError("synthetic unavailable"))))
        self.assertEqual(search(gate, "synthetic"), [])

    def test_receipt_and_writer_errors_are_typed_not_generic_runtime_or_value(self) -> None:
        class ReceiptError(ConnectionError):
            pass
        class WriterError(RuntimeError):
            pass
        with patch.dict(sys.modules, {"truememory.model_client": types.SimpleNamespace(ProtocolMismatchError=ReceiptError),
                                      "truememory.tier_switch.writer": types.SimpleNamespace(WriterSelectionChanged=WriterError)}):
            for error in (ReceiptError("synthetic"), WriterError("synthetic")):
                with self.assertRaises(type(error)) as raised:
                    self.api.raise_if_serving_rejection(error)
                self.assertIs(raised.exception, error)
            self.api.raise_if_serving_rejection(RuntimeError("ordinary unavailable"))
            self.api.raise_if_serving_rejection(ValueError("ordinary unavailable"))

    def pipeline(self):
        names = {"IngestionResult", "IngestionPipeline"}
        module = definitions("ingest/pipeline.py", names, self.namespace(
            dataclass=dataclasses.dataclass, field=dataclasses.field, contextlib=contextlib,
            sqlite3=sqlite3, time=time, datetime=datetime, Path=Path, os=os, log=logging.getLogger("synthetic-public"),
            _invalidate_recall_after_batch=lambda result: contextlib.nullcontext(),
            _dedup_store_lock=contextlib.nullcontext, _safe_log=lambda value: value))
        return module

    def test_store_and_update_typed_errors_never_fall_back_to_add(self) -> None:
        module = self.pipeline()
        pipeline = module.IngestionPipeline.__new__(module.IngestionPipeline)
        pipeline.user_id = ""
        error = self.api.TierRuntimeError("synthetic stale")
        fallback = Mock(side_effect=AssertionError("Unexpected fallback"))
        pipeline.memory = types.SimpleNamespace(_engine=types.SimpleNamespace(add=Mock(side_effect=error)), add=fallback,
                                               update=Mock(side_effect=error))
        fact = types.SimpleNamespace(category="general")
        for action in (lambda: pipeline._store_fact("synthetic", fact, ""),
                       lambda: pipeline._update_fact(1, "synthetic", fact, "")):
            with self.assertRaises(self.api.TierRuntimeError) as raised:
                action()
            self.assertIs(raised.exception, error)
        fallback.assert_not_called()

    def test_dedup_typed_search_refusal_does_not_become_add(self) -> None:
        module = definitions("ingest/dedup.py", {"DedupAction", "DedupDecision", "check_duplicate"}, self.namespace(
            Enum=enum.Enum, dataclass=dataclasses.dataclass, log=logging.getLogger("synthetic-public")))
        error = self.api.TierRuntimeError("synthetic stale")
        memory = types.SimpleNamespace(search_vectors=Mock(side_effect=error))
        with self.assertRaises(self.api.TierRuntimeError) as raised:
            module.check_duplicate("synthetic", memory)
        self.assertIs(raised.exception, error)
        memory.search_vectors.side_effect = RuntimeError("synthetic availability")
        self.assertEqual(module.check_duplicate("synthetic", memory).action, module.DedupAction.ADD)

    def test_both_pipeline_routes_pin_one_fact_and_release_before_next_extraction(self) -> None:
        module = self.pipeline()
        pipeline = module.IngestionPipeline.__new__(module.IngestionPipeline)
        engine = types.SimpleNamespace(conn=self.conn, _write_lock=threading.Lock(), _open_connection_handle=lambda: None)
        scheduling = definitions("engine.py", {"_maybe_auto_consolidate"}, self.namespace(), methods=True)
        engine._has_consolidation = False
        engine._maybe_auto_consolidate = types.MethodType(scheduling._maybe_auto_consolidate, engine)
        pipeline.memory = types.SimpleNamespace(_engine=engine)
        pipeline.llm_config, pipeline.use_llm_dedup, pipeline.user_id, pipeline.gate_enabled = None, False, "", True
        fact = types.SimpleNamespace(content="synthetic fact", category="general", confidence=1.0)
        observed = []
        def check(label):
            op = self.api.current_operation(self.conn)
            self.assertIsNotNone(op)
            observed.append((label, op))
        def evaluate(*args):
            check("gate")
            return types.SimpleNamespace(should_encode=True, encoding_score=1.0, novelty=1.0, salience=1.0,
                                         prediction_error=1.0, reason="synthetic")
        pipeline.gate = types.SimpleNamespace(reset_batch=lambda: None, evaluate=evaluate)
        class Action(enum.Enum):
            ADD = "add"
            UPDATE = "update"
            SKIP = "skip"
        def dedup(*args, **kwargs):
            check("dedup")
            return types.SimpleNamespace(action=Action.ADD, fact=fact.content, reason="synthetic", existing_id=None)
        pipeline._store_fact = lambda *args: check("store")
        def extract(*args):
            self.assertIsNone(self.api.current_operation(self.conn))
            return [fact, fact]
        module.extract_facts_simple = extract
        module.DedupAction, module.check_duplicate = Action, dedup
        module.parse_transcript = lambda path: ["synthetic"]
        module.format_for_extraction = lambda messages: "synthetic " * 20
        for action in (lambda: pipeline.ingest_text("synthetic " * 20), lambda: pipeline.ingest_transcript("synthetic-input")):
            observed.clear()
            result = action()
            self.assertEqual(result.facts_stored, 2)
            self.assertEqual([label for label, op in observed], ["gate", "dedup", "store"] * 2)
            self.assertEqual(len({id(op) for label, op in observed[:3]}), 1)
            self.assertIsNot(observed[0][1], observed[3][1])
            self.assertIsNone(self.api.current_operation(self.conn))


    def load_application(self, name: str) -> types.ModuleType:
        module = types.ModuleType("synthetic_public_" + name.replace(".", "_"))
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=self.imports)
        enter_context(self, patch.dict(sys.modules, {module.__name__: module}))
        path = ROOT / "truememory" / (name.replace(".", "/") + ".py")
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
        self.modules[name] = module
        return module

    def seed_policy(self, *, uppercase: bool = False):
        activation = self.modules["tier_switch.activation"]
        target = self.modules["embedding_target"].EmbeddingTarget("pro", "qwen3_256", 256, "basepro")
        pair = ("vec_messages", "vec_messages_sep")
        intent = activation.PolicyIntent("a" * 32, None, None, target,
            "Alibaba-NLP/gte-reranker-modernbert-base", activation.ConfigGuard(True, "base", None), "b" * 32, pair)
        for key, value in (("embed_model", target.model_id), ("embed_dim", "256"),
                           ("tier_activation_v1", activation._dump(intent))):
            activation._put(self.conn, key, value)
        self.conn.commit()
        if uppercase:
            ddl = self.conn.execute("SELECT sql FROM main.sqlite_master WHERE name='metadata'").fetchone()[0]
            self.conn.execute("ALTER TABLE metadata RENAME TO temporary_metadata")
            self.conn.execute(ddl.replace("metadata", "METADATA", 1))
            self.conn.execute("INSERT INTO METADATA SELECT * FROM temporary_metadata")
            self.conn.commit()
        return activation.read_activation_state(self.conn).legacy_policy

    def test_policy_reader_does_not_invent_selection_or_allow_legacy_mutation(self) -> None:
        policy = self.seed_policy(uppercase=True)
        self.assertIsNone(self.api._read_selection(self.conn))
        self.assertEqual(self.api._read_policy(self.conn), policy)
        with self.assertRaises(self.api.TierRuntimeError):
            self.api.require_legacy_vector_mutation(self.conn)
        with self.assertRaises(self.api.TierRuntimeError):
            with self.api.destructive_ingest_operation(types.SimpleNamespace(conn=self.conn, db_path=Path(":memory:"))):
                self.fail("Policy database reached destructive body")
        self.assertFalse(self.conn.in_transaction)

    def test_policy_writer_fences_absence_generation_and_exact_connection(self) -> None:
        writer = self.load_application("tier_switch.writer")
        old = writer.capture_writer_selection(self.conn)
        policy = self.seed_policy()
        fresh = writer.capture_writer_selection(self.conn)
        self.assertEqual(fresh.policy, policy)
        self.assertIsNone(fresh.selection)
        for captured in (old, writer.WriterCapture(self.conn, None), None):
            with self.subTest(captured=captured), self.assertRaises(writer.WriterSelectionChanged):
                with writer.writer_transaction(self.conn, captured):
                    self.fail("Uncaptured policy was accepted")
        with writer.writer_transaction(self.conn, fresh, model_id=policy.target.model_id,
                                       dimension=256, tables=policy.tables):
            self.conn.execute("INSERT INTO messages(id,content) VALUES(901,'synthetic policy fact')")
        self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=901").fetchone()[0], "synthetic policy fact")
        self.conn.execute("BEGIN IMMEDIATE")
        for captured in (writer.WriterCapture(self.conn, None, dataclasses.replace(policy, generation="c" * 32)),
                         writer.WriterCapture(object(), None, policy)):
            with self.assertRaises(writer.WriterSelectionChanged):
                writer.require_writer_selection(self.conn, captured)
        writer.require_writer_selection(self.conn, writer.WriterCapture(self.conn, None,
            dataclasses.replace(policy, config_acknowledged=True)))
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()

    def install_policy_projectors(self):
        client = types.SimpleNamespace()
        class EmbeddingProxy:
            pass
        class CertifiedEmbeddingProxy(EmbeddingProxy):
            def __init__(self, target):
                self.target = target
        class RerankerProxy:
            pass
        class CertifiedRerankerProxy(RerankerProxy):
            def __init__(self, name):
                self.name = name
        client.EmbeddingProxy, client.CertifiedEmbeddingProxy = EmbeddingProxy, CertifiedEmbeddingProxy
        client.RerankerProxy, client.CertifiedRerankerProxy = RerankerProxy, CertifiedRerankerProxy
        self.modules["model_client"] = client
        vector = definitions("vector_search.py", {"apply_embedding_policy"}, self.namespace(
            _model=object(), EMBEDDING_MODEL="qwen3_256", _embedding_dim=256, _model_generation=0,
            _target_state_lock=lambda deadline: contextlib.nullcontext(), _target_remaining=lambda deadline: None,
            _load_frozen_embedding_target=Mock(side_effect=AssertionError("Unexpected native load"))))
        reranker = definitions("reranker.py", {"apply_reranker_policy"}, self.namespace(
            _model=object(), _model_name="Alibaba-NLP/gte-reranker-modernbert-base", _model_certified=True,
            _reranker_load_lock=lambda control: contextlib.nullcontext()))
        self.fixture.vector, self.fixture.reranker = vector, reranker
        self.modules["vector_search"], self.modules["reranker"] = vector, reranker
        return vector, reranker, client

    def test_same_space_policy_projection_retains_local_models_and_has_zero_loads(self) -> None:
        policy = self.seed_policy()
        vector, reranker, _ = self.install_policy_projectors()
        before = vector._model, reranker._model
        with self.fixture.gate.exclusive_activation():
            self.api.apply_runtime_policy(policy)
        self.assertIs(vector._model, before[0])
        self.assertIs(reranker._model, before[1])
        self.assertEqual(self.api.runtime_acknowledgement(), self.api.policy_key(policy))
        with self.api.serving_operation(self.conn) as operation:
            self.assertEqual(operation.policy, policy)
            self.assertIsNone(operation.selection)
            self.assertEqual((operation.tables, operation.tier), (policy.tables, "pro"))
            with self.api.serving_operation(self.conn) as nested:
                self.assertIs(nested, operation)
            writer = self.load_application("tier_switch.writer")
            self.assertEqual(writer.capture_writer_selection(self.conn).policy, policy)
        vector._load_frozen_embedding_target.assert_not_called()

    def test_policy_projection_rebinds_proxies_and_keeps_absent_slots_lazy(self) -> None:
        policy = self.seed_policy()
        vector, reranker, client = self.install_policy_projectors()
        vector._model, reranker._model = client.EmbeddingProxy(), client.RerankerProxy()
        self.api.apply_runtime_policy(policy)
        self.assertIsInstance(vector._model, client.CertifiedEmbeddingProxy)
        self.assertEqual(vector._model.target, policy.target)
        self.assertIsInstance(reranker._model, client.CertifiedRerankerProxy)
        vector._model = reranker._model = None
        self.api.apply_runtime_policy(policy)
        self.assertIsNone(vector._model)
        self.assertIsNone(reranker._model)
        self.assertIsNone(self.api.runtime_acknowledgement())
        vector._load_frozen_embedding_target.assert_not_called()
        self.conn.execute("INSERT INTO messages(id,content) VALUES(902,'synthetic borrowed')")
        with self.assertRaises(self.api.TierRuntimeError):
            with self.api.serving_operation(self.conn):
                self.fail("Borrowed policy admission attempted native preparation")
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        vector._load_frozen_embedding_target.side_effect = None
        vector._load_frozen_embedding_target.return_value = object()
        def prepare(tier, name, **kwargs):
            self.assertFalse(self.conn.in_transaction)
            reranker._model, reranker._model_certified = object(), True
        reranker.apply_frozen_reranker = prepare
        with self.api.serving_operation(self.conn) as operation:
            self.assertEqual(operation.policy, policy)
        vector._load_frozen_embedding_target.assert_called_once()

    def test_policy_projection_failure_never_acknowledges_partial_slots(self) -> None:
        policy = self.seed_policy()
        _, reranker, _ = self.install_policy_projectors()
        self.api.apply_runtime_policy(policy)
        error = MemoryError("synthetic projection failure")
        reranker.apply_reranker_policy = Mock(side_effect=error)
        with self.assertRaises(MemoryError) as raised:
            self.api.apply_runtime_policy(policy)
        self.assertIs(raised.exception, error)
        self.assertIsNone(self.api.runtime_acknowledgement())

    def test_gate_prediction_error_releases_reference_on_all_outcomes(self) -> None:
        namespace = self.gate({"_compute_prediction_error"})
        namespace["EncodingGate"] = types.SimpleNamespace(_refresh_serving_generation=lambda gate: None)
        namespace["_PE_NOISE"] = set()
        namespace["np"] = types.SimpleNamespace(linalg=types.SimpleNamespace(norm=lambda value: 1.0), dot=lambda a, b: 1.0)
        class Model:
            pass
        model = Model()
        gate = types.SimpleNamespace(_embed_model=model, _last_search_results=[{"content": "synthetic nearest"}],
                                     _pe_available=True, _pe_degradation_count=0)
        encode = Mock(return_value=[[1.0]] * 4)
        self.modules["mps_utils"] = types.SimpleNamespace(encode_with_model_ownership=encode)
        for error in (None, self.api.TierRuntimeError("synthetic refusal"), KeyboardInterrupt("synthetic stop")):
            gate._embed_model = model
            encode.side_effect = error
            if error is None:
                self.assertEqual(namespace["_compute_prediction_error"](gate, "synthetic fact"), 0.0)
            else:
                with self.assertRaises(type(error)) as raised:
                    namespace["_compute_prediction_error"](gate, "synthetic fact")
                self.assertIs(raised.exception, error)
            self.assertIsNone(gate._embed_model)

    def test_legacy_mutator_owner_refuses_before_sql_and_does_not_end_borrowed_work(self) -> None:
        error = RuntimeError("synthetic owner busy")
        self.modules["maintenance"].maintenance_owner = Mock(side_effect=error)
        self.conn.execute("INSERT INTO messages(id,content) VALUES(903,'synthetic borrowed')")
        before = self.conn.commits, self.conn.rollbacks
        with self.api.serving_operation(self.conn):
            with self.assertRaises(RuntimeError) as raised:
                with self.api.legacy_vector_mutation(self.conn):
                    self.fail("Contended owner admitted mutation")
        self.assertIs(raised.exception, error)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.assertTrue(self.conn.in_transaction)

    def test_legacy_mutator_rechecks_policy_after_owner_acquisition(self) -> None:
        @contextlib.contextmanager
        def owner(path):
            self.seed_policy()
            yield
        self.modules["maintenance"].maintenance_owner = owner
        with self.assertRaises(self.api.TierRuntimeError):
            with self.api.legacy_vector_mutation(self.conn):
                self.fail("Policy appearing before owner admission reached writes")

    def test_existing_controlled_open_never_calls_schema_factory_and_closes_on_rejection(self) -> None:
        policy = self.seed_policy()
        factory = Mock(side_effect=AssertionError("Unexpected legacy initialization"))
        self.modules["storage"] = types.SimpleNamespace(DEFAULT_BUSY_TIMEOUT_MS=10000)
        path = Path(Path.cwd().anchor) / "synthetic" / "database.db"
        statements = []
        self.conn.set_trace_callback(statements.append)
        with patch.object(Path, "exists", return_value=True), patch.object(self.api.sqlite3, "connect", return_value=self.conn) as connect:
            opened = self.api.open_serving_connection(path, factory)
        self.assertIs(opened, self.conn)
        self.assertEqual(self.api._read_policy(opened), policy)
        self.assertIn("?mode=rw", connect.call_args.args[0])
        self.assertFalse(any(sql.lstrip().split(" ", 1)[0].upper() in {"INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"}
                             for sql in statements))
        factory.assert_not_called()
        error = self.api.TierRuntimeError("synthetic invalid journal")
        close = Mock()
        with patch.object(Path, "exists", return_value=True), patch.object(self.api.sqlite3, "connect", return_value=self.conn), \
                patch.object(self.api, "read_activation_state", side_effect=error), patch.object(self.conn, "close", close):
            with self.assertRaises(self.api.TierRuntimeError) as raised:
                self.api.open_serving_connection(path, factory)
        self.assertIs(raised.exception, error)
        close.assert_called_once()


    def test_nonblocking_owner_breaks_reader_activation_cycle(self) -> None:
        path = Path(Path.cwd().anchor) / "synthetic" / "database.db"
        owner = definitions("maintenance.py", {"MaintenanceBusyError", "MaintenanceOwnership", "_acquire_owner",
            "_release_owner", "_bind_owner", "maintenance_owner"}, self.namespace(
            NamedTuple=__import__("typing").NamedTuple, contextmanager=contextlib.contextmanager,
            Path=Path, os=types.SimpleNamespace(getpid=os.getpid, close=lambda fd: None), uuid=uuid,
            canonical_database_path=lambda value: path, _registry_lock=threading.Lock(),
            _held_paths={}, _held_fds=set(), _owner_waiters={}, _coordinators={},
            _thread_owners=threading.local(), try_file_lock=lambda value: 987))
        owner.connection_database_path = lambda conn: path
        self.modules["maintenance"] = owner
        held, finished = threading.Event(), threading.Event()
        entered, errors = [], []
        def activate():
            try:
                with owner.maintenance_owner(path):
                    held.set()
                    with self.fixture.gate.exclusive_activation():
                        entered.append(True)
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()
        self.conn.execute("INSERT INTO messages(id,content) VALUES(904,'synthetic borrowed')")
        before = self.conn.commits, self.conn.rollbacks
        with self.api.serving_operation(self.conn):
            worker = threading.Thread(target=activate)
            worker.start()
            self.assertTrue(held.wait(2))
            with self.assertRaises(owner.MaintenanceBusyError):
                with self.api.legacy_vector_mutation(self.conn):
                    self.fail("Busy activation owner allowed legacy writes")
            self.assertEqual(entered, [])
            self.assertTrue(self.conn.in_transaction)
            self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.assertTrue(finished.wait(2))
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(entered, [True])
        self.assertEqual(owner._held_paths, {})

    def test_cold_selected_consolidate_defers_before_model_or_engine_initialization(self) -> None:
        self.fixture.select()
        events = []
        report = types.SimpleNamespace(results=())
        coordinator = types.SimpleNamespace(path=None, refresh_capabilities=lambda: events.append("capabilities"))
        @contextlib.contextmanager
        def observation(value, **kwargs):
            events.append("owner")
            yield []
        lock = threading.Lock()
        def run(conn, supplied, **kwargs):
            self.assertFalse(lock.locked())
            with self.api.maintenance_serving_operation(conn) as supported:
                self.assertFalse(supported)
                events.append("deferred")
            return report
        self.modules["maintenance"] = types.SimpleNamespace(
            MaintenanceBusyError=type("Busy", (RuntimeError,), {}), format_maintenance_report=lambda value: {"clusters": "deferred"},
            maintenance_busy_result=lambda: {}, maintenance_observation=observation,
            observe_style=Mock(side_effect=AssertionError("Unexpected style observation")),
            run_engine_maintenance=run, _style_unready_result=lambda value: value)
        source = definitions("engine.py", {"consolidate"}, self.namespace(), methods=True)
        engine = types.SimpleNamespace(conn=self.conn, _write_lock=lock, _open_connection_handle=lambda: events.append("open"),
            _ensure_connection=Mock(side_effect=AssertionError("Unexpected runtime initialization")),
            _get_maintenance_coordinator=lambda: coordinator, _auto_consolidate_threshold=25,
            _apply_manual_maintenance_capabilities=lambda results: None)
        self.assertEqual(source.consolidate(engine), {"clusters": "deferred"})
        self.assertEqual(events, ["open", "capabilities", "owner", "deferred"])
        self.assertEqual(self.fixture.loads, [])
        engine._ensure_connection.assert_not_called()

    def test_selected_foreground_metadata_is_fenced_noop_preserving_caller_sql(self) -> None:
        selected = self.fixture.select()
        writer = self.load_application("tier_switch.writer")
        activation = self.modules["tier_switch.activation"]
        activation._put(self.conn, "embed_model", selected.target.model_id)
        activation._put(self.conn, "embed_dim", str(selected.target.dimension))
        self.conn.commit()
        module = definitions("vector_search.py", {"_write_embedder_metadata_no_commit"}, self.namespace(
            EMBEDDING_MODEL=selected.target.model_id, _embedding_dim=selected.target.dimension))
        with self.api.serving_operation(self.conn):
            captured = writer.capture_writer_selection(self.conn)
            with writer.writer_transaction(self.conn, captured):
                changes = self.conn.total_changes
                module._write_embedder_metadata_no_commit(self.conn)
                self.assertEqual(self.conn.total_changes, changes)
                self.assertTrue(self.conn.in_transaction)
                module._embedding_dim += 1
                with self.assertRaises(writer.WriterSelectionChanged):
                    module._write_embedder_metadata_no_commit(self.conn)
        self.assertFalse(self.conn.in_transaction)

    def test_legacy_initializer_propagates_typed_guard_or_owner_refusal(self) -> None:
        class Busy(RuntimeError):
            pass
        self.modules["maintenance"].MaintenanceBusyError = Busy
        enter_context(self, patch.dict(sys.modules, {"truememory.maintenance": self.modules["maintenance"]}))
        def imports(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sqlite_vec":
                return types.SimpleNamespace(load=lambda conn: None)
            return self.imports(name, globals, locals, fromlist, level)
        self.conn.enable_load_extension = lambda value: None
        module = definitions("engine.py", {"_initialize_connection"}, dict(
            __builtins__=dict(vars(builtins), __import__=imports), _HAS_VECTOR=True,
            TrueMemoryMigrationError=type("Migration", (RuntimeError,), {})), methods=True)
        engine = types.SimpleNamespace(conn=self.conn, ready=False, _runtime_initialized=False,
                                       _init_lock=threading.Lock(), _write_lock=threading.Lock(), _has_vectors=False)
        for error in (self.api.TierRuntimeError("synthetic changed"), Busy("synthetic owner busy")):
            module.init_vec_table = Mock(side_effect=error)
            with self.api.serving_operation(self.conn):
                with self.assertRaises(type(error)) as raised:
                    module._initialize_connection(engine, _suppress_maintenance=True)
            self.assertIs(raised.exception, error)
            self.assertFalse(engine.ready)
            self.assertFalse(engine._runtime_initialized)

    def test_hybrid_optional_separation_refusal_does_not_degrade(self) -> None:
        source = definitions("hybrid.py", {"search_hybrid"}, self.namespace(database_operation=lambda function: function))
        error = self.api.TierRuntimeError("synthetic stale separation")
        self.modules["fts_search"] = types.SimpleNamespace(search_fts=Mock())
        self.modules["vector_search"] = types.SimpleNamespace(search_vector=Mock(), search_vector_separation=Mock(),
                                                            _active_sep_table=Mock(side_effect=error))
        with self.assertRaises(self.api.TierRuntimeError) as raised:
            source.search_hybrid(self.conn, "synthetic")
        self.assertIs(raised.exception, error)
        self.modules["vector_search"].search_vector.assert_not_called()
        self.modules["fts_search"].search_fts.assert_not_called()


    def test_policy_initialization_validates_exact_pair_without_ddl_or_metadata(self) -> None:
        policy = self.seed_policy()
        self.install_policy_projectors()
        self.api.apply_runtime_policy(policy)
        for table in policy.tables:
            self.conn.execute(f"CREATE TABLE {table}(embedding BLOB)")
        original = type(self.conn).execute
        def execute(conn, sql, parameters=()):
            if sql == "SELECT sql FROM main.sqlite_master WHERE name=? COLLATE NOCASE" and parameters[0] in policy.tables:
                return BRIDGE["JOURNAL"]["Rows"]([(f"CREATE VIRTUAL TABLE {parameters[0]} USING vec0(embedding float[256] distance_metric=cosine)",)])
            return original(conn, sql, parameters)
        module = definitions("vector_search.py", {"init_vec_table"}, self.namespace())
        statements = []
        self.conn.set_trace_callback(statements.append)
        with patch.object(type(self.conn), "execute", execute):
            with self.api.serving_operation(self.conn):
                module.init_vec_table(self.conn)
                with self.assertRaises(ValueError):
                    module.init_vec_table(self.conn, tier_group="basepro")
        self.assertFalse(any(sql.lstrip().split(" ", 1)[0].upper() in {"INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"}
                             for sql in statements))
        self.assertFalse(self.conn.in_transaction)

    def test_policy_loaded_runtime_cannot_be_inherited_by_unselected_database(self) -> None:
        policy = self.seed_policy()
        self.install_policy_projectors()
        self.api.apply_runtime_policy(policy)
        other = sqlite3.connect(":memory:")
        self.addCleanup(other.close)
        with self.assertRaises(self.api.TierRuntimeError):
            with self.api.serving_operation(other):
                self.fail("Unselected database silently inherited another policy")


    def maintenance_runner(self, *, owner=None, run_layers=None):
        owner = owner or (lambda path: contextlib.nullcontext())
        module = definitions("maintenance.py", {
            "LayerResult", "MaintenanceReport", "SCHEDULED_PREFERENCES_UNAVAILABLE",
            "_maintenance_embedding_scope", "_maintenance_completion", "run_engine_maintenance",
        }, self.namespace(
            NamedTuple=__import__("typing").NamedTuple, contextmanager=contextlib.contextmanager,
            maintenance_owner=owner, connection_database_path=lambda conn: None,
            engine_layer_specs=lambda *args, **kwargs: (), all_layer_specs=lambda conn: (),
            run_layers=run_layers or (lambda *args, **kwargs: ()), time=time,
            importlib=types.SimpleNamespace(import_module=lambda name: types.SimpleNamespace(extract_preferences=lambda conn: None)),
        ))
        return module

    def memory_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:", check_same_thread=False,
                               factory=BRIDGE["JOURNAL"]["JournalConnection"])
        self.conn.backup(conn)
        self.addCleanup(conn.close)
        return conn

    def real_memory_owner(self, acquire):
        return definitions("maintenance.py", {"MaintenanceOwnership", "MaintenanceBusyError",
            "canonical_database_path", "_bind_owner", "maintenance_owner", "_active_initialization_receipt"}, self.namespace(
                NamedTuple=__import__("typing").NamedTuple, contextmanager=contextlib.contextmanager,
                Path=Path, os=os, uuid=uuid, _thread_owners=threading.local(), weakref=weakref,
                _registry_lock=threading.Lock(), _coordinators=weakref.WeakValueDictionary(), _held_paths={},
                _acquire_owner=acquire, _release_owner=lambda fd: None))

    def test_maintenance_authority_waits_for_managed_writer_before_sql(self) -> None:
        conn = self.memory_connection()
        engine = self.fixture.lifecycle_engine("_open_connection_handle", conn)
        owned, waiting, release, finished = (threading.Event() for _ in range(4))
        failures, reads = [], []
        api = self.api

        class ObservedLock(api.ConnectionWriteLock):
            def acquire(self, blocking=True, timeout=-1):
                if self.owns_other_transaction(conn):
                    waiting.set()
                return super().acquire(blocking, timeout)

        engine._write_lock = lock = ObservedLock(engine)
        original = api._read_selection

        def read(connection):
            self.assertFalse(lock.owns_other_transaction(connection))
            self.assertTrue(lock.locked())
            reads.append(connection)
            return original(connection)

        def writer():
            try:
                with lock:
                    conn.execute("INSERT INTO messages(id,content) VALUES(991,'synthetic writer')")
                    owned.set()
                    if not release.wait(2):
                        raise AssertionError("Synthetic writer was not released")
                    conn.commit()
            except BaseException as error:
                failures.append(error)

        def reader():
            try:
                with api.maintenance_serving_operation(conn, connection_lock=lock,
                                                       deadline=time.monotonic() + 2) as allowed:
                    self.assertTrue(allowed)
                    self.assertFalse(lock.locked())
                finished.set()
            except BaseException as error:
                failures.append(error)

        first, second = threading.Thread(target=writer), threading.Thread(target=reader)
        with patch.object(api, "_read_selection", side_effect=read):
            first.start()
            try:
                self.assertTrue(owned.wait(2))
                second.start()
                self.assertTrue(waiting.wait(2))
                self.assertEqual(reads, [])
            finally:
                release.set()
                first.join(2)
                if second.ident is not None:
                    second.join(2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(finished.is_set())
        self.assertGreaterEqual(len(reads), 3)

    def test_maintenance_borrowed_busy_cancel_and_deadline_preserve_sql(self) -> None:
        engine = self.fixture.lifecycle_engine("_open_connection_handle")
        self.conn.execute("INSERT INTO messages(id,content) VALUES(991,'synthetic caller')")
        before = self.conn.commits, self.conn.rollbacks
        with engine._write_lock:
            with self.assertRaises(self.api.TierRuntimeError):
                with self.api.maintenance_serving_operation(self.conn, connection_lock=engine._write_lock):
                    self.fail("Borrowed busy admission entered maintenance")
        cancelled = threading.Event()
        cancelled.set()
        for controls, kind in (({"cancelled": cancelled}, self.fixture.gate.ServingLeaseCancelled),
                               ({"deadline": time.monotonic() - 1}, self.fixture.gate.ServingLeaseTimeout)):
            with self.subTest(kind=kind.__name__), self.assertRaises(kind):
                with self.api.maintenance_serving_operation(self.conn, connection_lock=engine._write_lock, **controls):
                    self.fail("Rejected admission entered maintenance")
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=991").fetchone()[0], "synthetic caller")
        self.assertFalse(engine._write_lock.locked())

    def test_nested_maintenance_reuses_lease_without_sql_or_writer_reacquisition(self) -> None:
        engine = self.fixture.lifecycle_engine("_open_connection_handle")
        with self.api.maintenance_serving_operation(self.conn, connection_lock=engine._write_lock) as allowed:
            self.assertTrue(allowed)
            operation = self.api.current_operation(self.conn)
            with engine._write_lock, \
                 patch.object(self.api, "_read_selection", side_effect=AssertionError("Nested authority SQL")), \
                 patch.object(self.api, "_read_policy", side_effect=AssertionError("Nested policy SQL")), \
                 patch.object(self.api, "_connection_read", side_effect=AssertionError("Nested writer acquisition")):
                with self.api.maintenance_serving_operation(self.conn, connection_lock=engine._write_lock) as nested:
                    self.assertTrue(nested)
                    self.assertIs(self.api.current_operation(self.conn), operation)

    def test_runtime_final_table_and_database_snapshots_hold_writer(self) -> None:
        engine = self.fixture.lifecycle_engine("_open_connection_handle")
        observed = []

        def snapshot(name, value):
            def read(conn):
                self.assertTrue(engine._write_lock.locked())
                observed.append(name)
                return value
            return read

        with patch.object(self.fixture.vector, "_active_vec_table", side_effect=snapshot("normal", "vec_messages")), \
             patch.object(self.fixture.vector, "_active_sep_table", side_effect=snapshot("separation", "vec_messages_sep")), \
             patch.object(self.api, "_database_identity", side_effect=snapshot("database", ("memory", self.conn))):
            with self.api.serving_operation(self.conn, connection_lock=engine._write_lock) as operation:
                self.assertEqual(operation.tables, ("vec_messages", "vec_messages_sep"))
                self.assertFalse(engine._write_lock.locked())
        self.assertEqual(observed, ["normal", "separation", "database"])

    def test_runtime_preparation_and_gate_admission_release_writer(self) -> None:
        self.fixture.select()
        engine = self.fixture.lifecycle_engine("_open_connection_handle")
        prepare, lease = self.api.apply_frozen_selection, self.fixture.gate.operation_lease

        def project(*args, **kwargs):
            self.assertFalse(engine._write_lock.locked())
            return prepare(*args, **kwargs)

        @contextlib.contextmanager
        def admit(*args, **kwargs):
            self.assertFalse(engine._write_lock.locked())
            with lease(*args, **kwargs) as value:
                yield value

        with patch.object(self.api, "apply_frozen_selection", side_effect=project), \
             patch.object(self.fixture.gate, "operation_lease", side_effect=admit):
            with self.api.serving_operation(self.conn, connection_lock=engine._write_lock):
                self.assertFalse(engine._write_lock.locked())
        self.assertEqual(len(self.fixture.loads), 2)

    def test_runner_completion_cleans_owned_sql_under_lock_and_keeps_borrowed_sql(self) -> None:
        engine = self.fixture.lifecycle_engine("_open_connection_handle")

        def layers(conn, *args, **kwargs):
            self.assertTrue(engine._write_lock.locked())
            conn.execute("INSERT INTO messages(id,content) VALUES(992,'synthetic unfinished')")
            return ()

        module = self.maintenance_runner(run_layers=layers)
        rollback = self.conn.rollback
        calls = []

        def clean():
            calls.append(engine._write_lock.locked())
            return rollback()

        with patch.object(self.conn, "rollback", side_effect=clean):
            with self.assertRaisesRegex(RuntimeError, "unfinished transaction"):
                module.run_engine_maintenance(self.conn, object(), prepare_extensions=False,
                                              _connection_lock=engine._write_lock)
        self.assertTrue(calls)
        self.assertTrue(all(calls))
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages WHERE id=992").fetchone()[0], 0)
        self.conn.execute("INSERT INTO messages(id,content) VALUES(991,'synthetic caller')")
        before = self.conn.commits, self.conn.rollbacks
        module.run_engine_maintenance(self.conn, object(), prepare_extensions=False,
                                      _connection_lock=engine._write_lock, allow_caller_transaction=True)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages WHERE id IN (991,992)").fetchone()[0], 2)


    def test_opener_owns_canonical_path_before_writer_and_reenters_real_owner(self) -> None:
        replacement = self.memory_connection()
        engine = self.fixture.lifecycle_engine("_open_connection_handle")
        engine.conn = None
        engine.db_path = Path("~synthetic-unexpanded/store.sqlite")
        enter_context(self, patch.object(Path, "resolve", _REAL_PATH_RESOLVE))
        expected = engine.db_path.resolve()
        acquired, created = [], []

        def acquire(path):
            self.assertTrue(engine._init_lock.locked())
            self.assertFalse(engine._write_lock.locked())
            self.assertEqual(path, expected)
            acquired.append(path)
            return 17

        owner = self.real_memory_owner(acquire)
        self.modules["maintenance"] = owner

        def create(path):
            self.assertEqual(path, expected)
            self.assertTrue(engine._init_lock.locked())
            self.assertTrue(engine._write_lock.locked())
            self.assertIn(expected, owner._thread_owners.owners)
            created.append(path)
            return replacement

        engine._open_connection_handle.__globals__["create_db"] = create
        with patch.object(Path, "mkdir") as mkdir, patch.object(Path, "chmod") as chmod, \
             patch.object(Path, "exists", return_value=False):
            engine._open_connection_handle()
        self.assertEqual(acquired, [expected])
        self.assertEqual(created, [expected])
        self.assertIs(engine.conn, replacement)
        self.assertEqual(owner._thread_owners.owners, {})
        self.assertFalse(engine.ready)
        self.assertFalse(engine._runtime_initialized)
        self.assertFalse(engine._init_lock.locked())
        self.assertFalse(engine._write_lock.locked())
        mkdir.assert_called_once_with(parents=True, exist_ok=True)
        chmod.assert_called_once_with(0o700)

    def test_opener_healthy_and_borrowed_handles_do_not_request_owner(self) -> None:
        owner = Mock(side_effect=AssertionError("Healthy handle requested maintenance owner"))
        self.modules["maintenance"].maintenance_owner = owner
        engine = self.fixture.lifecycle_engine("_open_connection_handle")
        engine._open_connection_handle()
        self.conn.execute("INSERT INTO messages(id,content) VALUES(991,'synthetic caller')")
        before = self.conn.commits, self.conn.rollbacks
        engine._open_connection_handle()
        error = sqlite3.OperationalError("disk I/O error")
        self.conn.armed, self.conn.fault_contains, self.conn.fault = True, "PRAGMA schema_version", error
        with patch.object(self.conn, "close", wraps=self.conn.close) as close:
            with self.assertRaises(sqlite3.OperationalError) as raised:
                engine._open_connection_handle()
            close.assert_not_called()
        self.assertIs(raised.exception, error)
        self.assertIs(engine.conn, self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.assertTrue(engine.ready)
        owner.assert_not_called()

    def test_opener_owner_busy_preserves_ioerr_handle_and_ready_state(self) -> None:
        owner = self.real_memory_owner(lambda path: None)
        self.modules["maintenance"] = owner
        engine = self.fixture.lifecycle_engine("_open_connection_handle")
        engine.db_path = Path("synthetic/store.sqlite")
        self.conn.armed, self.conn.fault_contains = True, "PRAGMA schema_version"
        self.conn.fault = sqlite3.OperationalError("disk I/O error")
        before = self.conn.commits, self.conn.rollbacks
        with patch.object(Path, "mkdir"), patch.object(Path, "chmod"), \
             patch.object(self.conn, "close", wraps=self.conn.close) as close:
            with self.assertRaises(owner.MaintenanceBusyError):
                engine._open_connection_handle()
            close.assert_not_called()
        self.assertIs(engine.conn, self.conn)
        self.assertTrue(engine.ready)
        self.assertTrue(engine._runtime_initialized)
        self.assertIs(engine._runtime_vector_connection, self.conn)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.assertFalse(engine._write_lock.locked())
        self.assertFalse(engine._init_lock.locked())

    def test_opener_second_probe_does_not_retire_a_new_healthy_handle(self) -> None:
        replacement = self.memory_connection()
        engine = self.fixture.lifecycle_engine("_open_connection_handle")
        engine.db_path = Path("synthetic/store.sqlite")
        self.conn.armed, self.conn.fault_contains = True, "PRAGMA schema_version"
        self.conn.fault = sqlite3.OperationalError("disk I/O error")

        def acquire(path):
            self.assertFalse(engine._write_lock.locked())
            engine.conn = replacement
            return 17

        self.modules["maintenance"] = self.real_memory_owner(acquire)
        create = Mock(side_effect=AssertionError("Rechecked healthy connection was replaced"))
        engine._open_connection_handle.__globals__["create_db"] = create
        with patch.object(Path, "mkdir"), patch.object(Path, "chmod"), \
             patch.object(self.conn, "close", wraps=self.conn.close) as close:
            engine._open_connection_handle()
            close.assert_not_called()
        self.assertIs(engine.conn, replacement)
        create.assert_not_called()

    def test_concurrent_first_open_uses_one_factory_and_no_owner_contention(self) -> None:
        replacement = self.memory_connection()
        engine = self.fixture.lifecycle_engine("_open_connection_handle")
        engine.conn, engine.db_path = None, Path("synthetic/store.sqlite")
        acquired, created, failures = [], [], []
        factory_entered, release, second_waiting = (threading.Event() for _ in range(3))
        real_init = engine._init_lock

        class ObservedInit:
            def acquire(self, blocking=True, timeout=-1):
                if real_init.locked():
                    second_waiting.set()
                return real_init.acquire(blocking, timeout)

            def release(self):
                real_init.release()

            def locked(self):
                return real_init.locked()

        engine._init_lock = ObservedInit()

        def acquire(path):
            self.assertFalse(engine._write_lock.locked())
            acquired.append(path)
            return 17

        self.modules["maintenance"] = self.real_memory_owner(acquire)

        def create(path):
            created.append(path)
            factory_entered.set()
            if not release.wait(2):
                raise AssertionError("Synthetic factory release timed out")
            return replacement

        engine._open_connection_handle.__globals__["create_db"] = create

        def opening():
            try:
                engine._open_connection_handle()
            except BaseException as error:
                failures.append(error)

        first, second = threading.Thread(target=opening), threading.Thread(target=opening)
        with patch.object(Path, "mkdir"), patch.object(Path, "chmod"), patch.object(Path, "exists", return_value=False):
            first.start()
            try:
                self.assertTrue(factory_entered.wait(2))
                second.start()
                self.assertTrue(second_waiting.wait(2))
                self.assertEqual(len(acquired), 1)
                self.assertEqual(len(created), 1)
            finally:
                release.set()
                first.join(2)
                if second.ident is not None:
                    second.join(2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(acquired), 1)
        self.assertEqual(len(created), 1)
        self.assertIs(engine.conn, replacement)


    def test_consolidate_guards_authority_and_style_table_snapshots(self) -> None:
        self.fixture.select()
        engine = self.fixture.lifecycle_engine("_open_connection_handle")
        engine._open_connection_handle = lambda: None
        engine._ensure_connection = Mock(side_effect=AssertionError("Controlled store initialized models"))
        engine._auto_consolidate_threshold = 25
        engine._get_maintenance_coordinator = lambda: types.SimpleNamespace(path=None, refresh_capabilities=lambda: None)
        engine._apply_manual_maintenance_capabilities = lambda results: None
        report = types.SimpleNamespace(results=())

        def run(conn, coordinator, **kwargs):
            self.assertFalse(engine._write_lock.locked())
            self.assertIs(kwargs["_connection_lock"], engine._write_lock)
            return report

        self.modules["maintenance"] = types.SimpleNamespace(
            MaintenanceBusyError=type("Busy", (RuntimeError,), {}), format_maintenance_report=lambda value: {},
            maintenance_busy_result=lambda: {}, maintenance_observation=lambda *args, **kwargs: contextlib.nullcontext([]),
            observe_style=Mock(side_effect=AssertionError("Unexpected file style probe")),
            run_engine_maintenance=run, _style_unready_result=lambda value: value)
        source = definitions("engine.py", {"consolidate"}, self.namespace(), methods=True)
        execute = self.conn.execute
        observed = []

        def guarded(sql, *args, **kwargs):
            self.assertTrue(engine._write_lock.locked())
            observed.append(sql)
            return execute(sql, *args, **kwargs)

        with patch.object(self.conn, "execute", side_effect=guarded):
            self.assertEqual(source.consolidate(engine), {})
        self.assertTrue(any("tier_selected_v1" in str(args) for args in observed)
                        or any("main.metadata" in sql for sql in observed))
        self.assertTrue(any("entity_style_vectors" in sql for sql in observed))
        engine._ensure_connection.assert_not_called()

    def test_consolidate_release_handoff_preserves_next_writer_with_negative_control(self) -> None:
        for restore_old_cleanup in (False, True):
            with self.subTest(restore_old_cleanup=restore_old_cleanup):
                conn = self.memory_connection()
                engine = self.fixture.lifecycle_engine("_open_connection_handle", conn)
                engine._open_connection_handle = lambda: None
                engine._ensure_connection = lambda **kwargs: None
                engine._auto_consolidate_threshold = 25
                engine._has_style_vec = False
                coordinator = types.SimpleNamespace(path=None, refresh_capabilities=lambda: None)
                engine._get_maintenance_coordinator = lambda: coordinator
                engine._apply_manual_maintenance_capabilities = lambda results: None
                armed, released, writer_started, writer_release = (threading.Event() for _ in range(4))
                failures = []
                main_thread = threading.get_ident()
                api = self.api
                owner = self

                class HandoffLock(api.ConnectionWriteLock):
                    def release(self):
                        handoff = armed.is_set() and threading.get_ident() == main_thread
                        if handoff:
                            armed.clear()
                        super().release()
                        if handoff:
                            released.set()
                            owner.assertTrue(writer_started.wait(2), "Synthetic next writer did not start")

                engine._write_lock = lock = HandoffLock(engine)

                def layers(*args, **kwargs):
                    self.assertTrue(lock.locked())
                    armed.set()
                    return ()

                runner = self.maintenance_runner(run_layers=layers)
                self.modules["maintenance"] = types.SimpleNamespace(
                    MaintenanceBusyError=type("Busy", (RuntimeError,), {}),
                    format_maintenance_report=lambda report: {"result": "synthetic complete"}, maintenance_busy_result=lambda: {},
                    maintenance_observation=lambda *args, **kwargs: contextlib.nullcontext([]),
                    observe_style=Mock(side_effect=AssertionError("Unexpected style probe")),
                    run_engine_maintenance=runner.run_engine_maintenance, _style_unready_result=lambda value: value)
                source = definitions("engine.py", {"consolidate"}, self.namespace(), methods=True)
                if restore_old_cleanup:
                    tree = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
                    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
                    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "consolidate")

                    class RestoreCleanup(ast.NodeTransformer):
                        inserted = 0

                        def visit_Assign(self, node):
                            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                                    and node.value.func.id == "run_engine_maintenance"
                                    and any(item.arg == "_connection_lock" for item in node.value.keywords)):
                                self.inserted += 1
                                old = ast.parse("if not borrowed and self.conn.in_transaction:\n"
                                                "    self.conn.rollback()\n"
                                                "    raise RuntimeError('Maintenance work left an unfinished transaction')").body[0]
                                return [node, old]
                            return node

                    restore = RestoreCleanup()
                    method = restore.visit(method)
                    self.assertEqual(restore.inserted, 1)
                    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                                 "synthetic-old-cleanup-negative-control", "exec"), source.__dict__)

                def next_writer():
                    try:
                        if not released.wait(2):
                            raise AssertionError("Maintenance mutation lock was not released")
                        with lock:
                            conn.execute("INSERT INTO messages(id,content) VALUES(993,'synthetic next writer')")
                            writer_started.set()
                            if not writer_release.wait(2):
                                raise AssertionError("Synthetic next writer was not released")
                            conn.rollback()
                    except BaseException as error:
                        failures.append(error)

                thread = threading.Thread(target=next_writer)
                thread.start()
                try:
                    if restore_old_cleanup:
                        with self.assertRaisesRegex(RuntimeError, "unfinished transaction"):
                            source.consolidate(engine)
                        self.assertFalse(conn.in_transaction)
                        self.assertEqual(conn.execute("SELECT count(*) FROM messages WHERE id=993").fetchone()[0], 0)
                    else:
                        self.assertEqual(source.consolidate(engine), {"result": "synthetic complete"})
                        self.assertTrue(conn.in_transaction)
                        self.assertEqual(conn.execute("SELECT content FROM messages WHERE id=993").fetchone()[0],
                                         "synthetic next writer")
                finally:
                    writer_release.set()
                    thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(failures, [])
                self.assertTrue(writer_started.is_set())


    def test_already_cancelled_runner_does_not_admit_or_wait_for_managed_writer(self) -> None:
        conn = self.memory_connection()
        engine = self.fixture.lifecycle_engine("_open_connection_handle", conn)
        owned, release = threading.Event(), threading.Event()
        failures = []
        cancel = threading.Event()
        cancel.set()
        module = self.maintenance_runner(owner=Mock(side_effect=AssertionError("Cancelled work requested owner")))
        module.connection_database_path = Mock(side_effect=AssertionError("Cancelled work read database identity"))
        module.run_routed_style = Mock(side_effect=AssertionError("Cancelled work called mutable style route"))

        def writer() -> None:
            try:
                with engine._write_lock:
                    conn.execute("INSERT INTO messages(id,content) VALUES(994,'synthetic managed writer')")
                    owned.set()
                    if not release.wait(2):
                        raise AssertionError("Synthetic writer was not released")
                    conn.rollback()
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            self.assertTrue(owned.wait(2))
            before = conn.commits, conn.rollbacks
            with patch.object(self.api, "_connection_read", side_effect=AssertionError("Cancelled work entered lock admission")), \
                 patch.object(conn, "execute", side_effect=AssertionError("Cancelled work executed SQL")):
                for include_style in (False, True):
                    with self.subTest(include_style=include_style):
                        report = module.run_engine_maintenance(conn, object(), cancel=cancel,
                            include_style=include_style, _connection_lock=engine._write_lock)
                        self.assertEqual(report.results, ())
                        self.assertEqual(report.preferences, "CANCELLED")
                        if include_style:
                            self.assertEqual(tuple(report.style_result), (
                                "style_vectors", "style_vectors", "deferred", None, 0.0,
                                "StyleCancelled", False, "unverified", True))
                        else:
                            self.assertIsNone(report.style_result)
            self.assertTrue(engine._write_lock.owns_other_transaction(conn))
            self.assertTrue(conn.in_transaction)
            self.assertEqual((conn.commits, conn.rollbacks), before)
            self.assertEqual(conn.execute("SELECT content FROM messages WHERE id=994").fetchone()[0],
                             "synthetic managed writer")
            module.maintenance_owner.assert_not_called()
            module.connection_database_path.assert_not_called()
            module.run_routed_style.assert_not_called()
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])

    def test_cancelled_snapshot_stays_terminal_when_event_clears(self) -> None:
        module = self.maintenance_runner(owner=Mock(side_effect=AssertionError("Cancelled snapshot requested owner")))
        module.connection_database_path = Mock(side_effect=AssertionError("Cancelled snapshot read database identity"))
        module.run_routed_style = Mock(side_effect=AssertionError("Cleared event reached style SQL"))
        for borrowed in (False, True):
            if borrowed:
                self.conn.execute("INSERT INTO messages(id,content) VALUES(994,'synthetic caller SQL')")
            before = self.conn.commits, self.conn.rollbacks
            for include_style in (False, True):
                with self.subTest(borrowed=borrowed, include_style=include_style):
                    cancel = threading.Event()
                    cancel.set()

                    def capture_and_clear() -> bool:
                        cancel.clear()
                        return True

                    with patch.object(cancel, "is_set", side_effect=capture_and_clear) as checked, \
                         patch.object(self.api, "_connection_read", side_effect=AssertionError("Cancelled snapshot waited")), \
                         patch.object(self.conn, "execute", side_effect=AssertionError("Cancelled snapshot executed SQL")):
                        report = module.run_engine_maintenance(self.conn, object(), cancel=cancel,
                            include_style=include_style, allow_caller_transaction=borrowed)
                        checked.assert_called_once_with()
                    self.assertFalse(cancel.is_set())
                    self.assertEqual(report.results, ())
                    self.assertEqual(report.preferences, "CANCELLED")
                    if include_style:
                        self.assertEqual(tuple(report.style_result), (
                            "style_vectors", "style_vectors", "deferred", None, 0.0,
                            "StyleCancelled", False, "unverified", borrowed))
                    else:
                        self.assertIsNone(report.style_result)
                    self.assertEqual(self.conn.in_transaction, borrowed)
                    self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
            if borrowed:
                self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=994").fetchone()[0],
                                 "synthetic caller SQL")
                self.conn.rollback()
        module.maintenance_owner.assert_not_called()
        module.connection_database_path.assert_not_called()
        module.run_routed_style.assert_not_called()


class TestManagedModelAdmission(unittest.TestCase):
    """Actual model guard classifies the owner of SQL on a shared handle."""

    setUp = TestPublicBoundaries.setUp
    memory_connection = TestPublicBoundaries.memory_connection

    def writer_lock(self, conn):
        class Engine:
            pass
        engine = Engine()
        engine.conn = conn
        lock = self.api.ConnectionWriteLock(engine)
        return engine, lock

    def test_model_guard_reprobes_owner_release_without_using_stale_sql_flag(self):
        # Negative control is the exact pre-fix ordering: read SQL first, then
        # ask whether the competing managed writer still owns the handle.
        def previous_guard(operation, lock):
            if operation.borrowed or operation._connection.in_transaction and not (
                    isinstance(lock, self.api.ConnectionWriteLock)
                    and lock.owns_other_transaction(operation._connection)):
                raise self.api.TierRuntimeError("Model loading requires a clean connection")
        for previous in (True, False):
            with self.subTest(previous_order=previous):
                conn = self.memory_connection()
                engine, lock = self.writer_lock(conn)
                entered, release, finished = threading.Event(), threading.Event(), threading.Event()
                errors, observations = [], []
                def writer():
                    try:
                        with lock:
                            conn.execute("BEGIN IMMEDIATE")
                            entered.set()
                            if not release.wait(3):
                                raise AssertionError("Synthetic release race timed out")
                            conn.commit()
                    except BaseException as error:
                        errors.append(error)
                        entered.set()
                    finally:
                        finished.set()
                actual_owner_check = lock.owns_other_transaction
                def owner_check(connection):
                    observations.append(connection.in_transaction)
                    release.set()
                    if not finished.wait(2):
                        raise AssertionError("Synthetic owner did not finish")
                    return actual_owner_check(connection)
                with self.api.serving_operation(conn, connection_lock=lock) as operation:
                    thread = threading.Thread(target=writer)
                    thread.start()
                    try:
                        self.assertTrue(entered.wait(1))
                        self.assertEqual(errors, [])
                        with patch.object(lock, "owns_other_transaction", side_effect=owner_check), \
                             patch.object(lock, "acquire", wraps=lock.acquire) as acquire:
                            if previous:
                                with self.assertRaisesRegex(self.api.TierRuntimeError, "clean connection"):
                                    previous_guard(operation, lock)
                                acquire.assert_not_called()
                            else:
                                self.api.require_model_load_allowed()
                                acquire.assert_called_once_with(blocking=False)
                        self.assertEqual(observations, [True])
                        self.assertFalse(conn.in_transaction)
                        self.assertFalse(lock.locked())
                        self.assertIs(engine.conn, conn)
                    finally:
                        release.set()
                        thread.join(3)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])

    def test_model_guard_current_thread_clean_lock_is_not_reacquired(self):
        conn = self.memory_connection()
        engine, lock = self.writer_lock(conn)
        with self.api.serving_operation(conn, connection_lock=lock):
            with lock:
                with patch.object(lock, "acquire", side_effect=AssertionError("Nonreentrant writer reacquired")):
                    self.api.require_model_load_allowed()
                self.assertTrue(lock.owned_by_current_thread())
                self.assertFalse(conn.in_transaction)
        self.assertFalse(lock.locked())
        self.assertIs(engine.conn, conn)

    def test_model_guard_allows_other_managed_writer_without_acquiring_its_lock(self):
        conn = self.memory_connection()
        engine, lock = self.writer_lock(conn)
        entered, release = threading.Event(), threading.Event()
        errors = []
        def writer():
            try:
                with lock:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute("INSERT INTO messages(id,content) VALUES(995,'synthetic writer')")
                    entered.set()
                    if not release.wait(3):
                        raise AssertionError("Synthetic writer release timed out")
                    conn.rollback()
            except BaseException as error:
                errors.append(error)
                entered.set()
        with self.api.serving_operation(conn, connection_lock=lock) as operation:
            self.assertFalse(operation.borrowed)
            self.assertIs(operation._connection_lock, lock)
            thread = threading.Thread(target=writer)
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                self.assertEqual(errors, [])
                self.assertTrue(lock.owns_other_transaction(conn))
                with patch.object(lock, "acquire", side_effect=AssertionError("Model guard waited on writer")):
                    self.api.require_model_load_allowed()
                self.assertTrue(conn.in_transaction)
                self.assertEqual(conn.execute("SELECT content FROM messages WHERE id=995").fetchone()[0], "synthetic writer")
                self.assertIs(engine.conn, conn)
            finally:
                release.set()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_model_guard_preserves_initial_borrowed_refusal_after_caller_commit(self):
        conn = self.memory_connection()
        engine, lock = self.writer_lock(conn)
        conn.execute("BEGIN")
        with self.api.serving_operation(conn, connection_lock=lock) as operation:
            self.assertTrue(operation.borrowed)
            with self.assertRaisesRegex(self.api.TierRuntimeError, "clean connection"):
                self.api.require_model_load_allowed()
            self.assertTrue(conn.in_transaction)
            conn.commit()
            with self.assertRaisesRegex(self.api.TierRuntimeError, "clean connection"):
                self.api.require_model_load_allowed()
        self.assertIs(engine.conn, conn)

    def test_model_guard_refuses_current_thread_owned_sql_without_ending_it(self):
        conn = self.memory_connection()
        engine, lock = self.writer_lock(conn)
        with self.api.serving_operation(conn, connection_lock=lock):
            with lock:
                conn.execute("BEGIN")
                try:
                    with self.assertRaisesRegex(self.api.TierRuntimeError, "clean connection"):
                        self.api.require_model_load_allowed()
                    self.assertTrue(conn.in_transaction)
                finally:
                    conn.rollback()
        self.assertIs(engine.conn, conn)

    def test_model_guard_refuses_unmanaged_or_duck_typed_transaction_ownership(self):
        class UnmanagedLock:
            def owns_other_transaction(self, conn):
                raise AssertionError("Foreign ownership callback was consulted")
        conn = self.memory_connection()
        for lock in (None, threading.Lock(), UnmanagedLock()):
            with self.subTest(lock=type(lock).__name__), self.api.serving_operation(conn) as operation:
                replacement = dataclasses.replace(operation, _connection_lock=lock)
                with self.api._bound(replacement):
                    conn.execute("BEGIN")
                    try:
                        with self.assertRaisesRegex(self.api.TierRuntimeError, "clean connection"):
                            self.api.require_model_load_allowed()
                        self.assertTrue(conn.in_transaction)
                    finally:
                        conn.rollback()

    def test_model_guard_does_not_use_managed_owner_for_another_handle(self):
        conn, other = self.memory_connection(), self.memory_connection()
        engine, lock = self.writer_lock(other)
        entered, release = threading.Event(), threading.Event()
        errors = []
        def writer():
            try:
                with lock:
                    other.execute("BEGIN")
                    entered.set()
                    if not release.wait(3):
                        raise AssertionError("Synthetic stale-handle writer release timed out")
                    other.rollback()
            except BaseException as error:
                errors.append(error)
                entered.set()
        with self.api.serving_operation(conn, connection_lock=lock):
            thread = threading.Thread(target=writer)
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                self.assertEqual(errors, [])
                self.assertTrue(lock.owns_other_transaction(other))
                self.assertFalse(lock.owns_other_transaction(conn))
                conn.execute("BEGIN")
                with self.assertRaisesRegex(self.api.TierRuntimeError, "clean connection"):
                    self.api.require_model_load_allowed()
                self.assertTrue(conn.in_transaction)
                conn.rollback()
            finally:
                release.set()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertIs(engine.conn, other)

    def test_model_guard_closed_handle_error_is_not_reclassified_or_suppressed(self):
        conn = self.memory_connection()
        with self.api.serving_operation(conn):
            conn.close()
            with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                self.api.require_model_load_allowed()

    def test_runtime_child_carries_lock_only_for_original_handle(self):
        conn, other = self.memory_connection(), self.memory_connection()
        engine, lock = self.writer_lock(conn)
        joined, errors = [], []
        with patch.object(self.api, "_database_identity", return_value=("file", "synthetic-database")):
            with self.api.serving_operation(conn, connection_lock=lock) as operation:
                same_child, other_child = operation.fork_child(), operation.fork_child()
                def join(child, connection, expected_lock):
                    try:
                        with child.join(connection) as current:
                            self.assertIs(current._connection_lock, expected_lock)
                            self.assertIs(current._connection, connection)
                            connection.execute("BEGIN")
                            try:
                                with self.assertRaisesRegex(self.api.TierRuntimeError, "clean connection"):
                                    self.api.require_model_load_allowed()
                            finally:
                                connection.rollback()
                            joined.append(connection)
                    except BaseException as error:
                        errors.append(error)
                # Sequential children isolate their SQL while retaining the
                # production thread-owned child admission protocol.
                for child, connection, expected in ((same_child, conn, lock), (other_child, other, None)):
                    thread = threading.Thread(target=join, args=(child, connection, expected))
                    thread.start()
                    thread.join(3)
                    self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(joined, [conn, other])
        self.assertIs(engine.conn, conn)


class TestEngineReconnectReceipt(unittest.TestCase):
    setUp = TestPublicBoundaries.setUp
    imports = TestPublicBoundaries.imports
    namespace = TestPublicBoundaries.namespace
    memory_connection = TestPublicBoundaries.memory_connection
    real_memory_owner = TestPublicBoundaries.real_memory_owner

    def engine(self, *, vectors=False):
        enter_context(self, patch.object(Path, "resolve", _REAL_PATH_RESOLVE))
        self.stat = types.SimpleNamespace(st_dev=7, st_ino=11)
        enter_context(self, patch.object(Path, "stat", return_value=self.stat))
        enter_context(self, patch.object(Path, "mkdir"))
        enter_context(self, patch.object(Path, "chmod"))
        self.modules["storage"] = definitions("storage.py", {"DatabaseOpenError", "_integrity_message", "_validate_db_path"},
            self.namespace(sqlite3=sqlite3, Path=Path, os=os, newest_backup=lambda path: None))
        namespace = self.namespace(sqlite3=sqlite3, Path=Path, logger=logging.getLogger("synthetic-reconnect"),
            _HAS_VECTOR=vectors, DEFAULT_BUSY_TIMEOUT_MS=3000,
            _ALL_VEC_TABLES=("vec_messages", "vec_messages_sep", "vec_messages_edge", "vec_messages_sep_edge",
                             "vec_messages_basepro", "vec_messages_sep_basepro"),
            create_db=Mock(side_effect=AssertionError("Unexpected schema factory")))
        tree = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrueMemoryEngine")
        names = {"_open_connection_handle", "_reconnect_registry", "_capture_reconnect_receipt",
                 "_reopen_initialized_connection", "close"}
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])),
                     "actual-engine-reconnect", "exec"), namespace)
        engine = namespace["TrueMemoryEngine"]()
        engine.conn, engine.db_path = self.conn, Path("/synthetic/reconnect.sqlite").absolute()
        engine._init_lock = threading.Lock()
        engine._write_lock = self.api.ConnectionWriteLock(engine)
        engine.ready = engine._runtime_initialized = True
        engine._has_vectors = vectors
        engine._runtime_vector_connection = self.conn
        engine._runtime_vector_generation = None
        engine._reconnect_receipt = None
        self.fixture.vector._active_tier_group = lambda: "edge"
        self.conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('l4_entity_profile_migration_done','1')")
        self.conn.execute("DELETE FROM metadata WHERE key IN ('embed_model','embed_dim')")
        if vectors:
            self.conn.execute("CREATE TABLE IF NOT EXISTS vec_messages(rowid INTEGER PRIMARY KEY)")
            self.conn.execute("CREATE TABLE IF NOT EXISTS vec_messages_sep(rowid INTEGER PRIMARY KEY)")
        self.conn.commit()
        operation = types.SimpleNamespace(key=self.api._legacy_key(), _database=("file", str(engine.db_path)))
        with engine._write_lock:
            engine._capture_reconnect_receipt(operation)
        self.assertIsNotNone(engine._reconnect_receipt)
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(len(engine._reconnect_receipt), 10)
        self.engine_namespace = namespace
        return engine

    def candidate(self, engine):
        candidate = self.memory_connection()
        # SQLite backup assigns a new schema cookie; simulate reopening the same file.
        schema = self.conn.execute("PRAGMA schema_version").fetchone()[0]
        candidate.execute(f"PRAGMA schema_version={schema}")
        connect = Mock(return_value=candidate)
        proxy = types.SimpleNamespace(**{name: getattr(sqlite3, name) for name in
            ("Connection", "ProgrammingError", "OperationalError", "DatabaseError", "Error")}, connect=connect)
        self.engine_namespace["sqlite3"] = proxy
        self.connect = connect
        return candidate

    def forbid_owner(self):
        owner = Mock(side_effect=AssertionError("Validated reconnect requested maintenance ownership"))
        self.modules["maintenance"].maintenance_owner = owner
        return owner

    def assert_closed(self, conn):
        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_extension_unavailable_fts_receipt_preserves_capability_and_source(self):
        engine = self.engine()
        self.engine_namespace["_HAS_VECTOR"] = True
        self.conn.enable_load_extension = None
        self.conn.execute("INSERT INTO messages(id,content) VALUES(883,'synthetic FTS source')")
        self.conn.commit()
        operation = types.SimpleNamespace(key=self.api._legacy_key(), _database=("file", str(engine.db_path)))
        engine._capture_reconnect_receipt(operation)
        self.assertIsNotNone(engine._reconnect_receipt)
        self.assertEqual(engine._reconnect_receipt[7:9], (False, True))
        candidate = self.candidate(engine)
        candidate.enable_load_extension = None
        owner = self.forbid_owner()
        self.conn.close()
        engine._open_connection_handle()
        self.assertIs(engine.conn, candidate)
        self.assertTrue(engine.ready)
        self.assertTrue(engine._runtime_initialized)
        self.assertFalse(engine._has_vectors)
        self.assertFalse(candidate.in_transaction)
        self.assertEqual(candidate.execute("SELECT content FROM messages WHERE id=883").fetchone()[0],
                         "synthetic FTS source")
        owner.assert_not_called()

    def test_fts_receipt_cannot_skip_new_extension_capability(self):
        engine = self.engine()
        self.engine_namespace["_HAS_VECTOR"] = True
        self.conn.enable_load_extension = None
        operation = types.SimpleNamespace(key=self.api._legacy_key(), _database=("file", str(engine.db_path)))
        engine._capture_reconnect_receipt(operation)
        receipt = engine._reconnect_receipt
        self.assertIsNotNone(receipt)
        candidate = self.candidate(engine)
        candidate.enable_load_extension = lambda enabled: None
        self.assertIsNone(engine._reopen_initialized_connection(engine.db_path))
        self.assert_closed(candidate)
        self.assertIs(engine.conn, self.conn)
        self.assertIs(engine._reconnect_receipt, receipt)
        self.assertFalse(self.conn.in_transaction)

    def test_other_vector_initialization_failure_does_not_publish_fts_receipt(self):
        engine = self.engine()
        self.engine_namespace["_HAS_VECTOR"] = True
        self.conn.enable_load_extension = lambda enabled: None
        operation = types.SimpleNamespace(key=self.api._legacy_key(), _database=("file", str(engine.db_path)))
        queries = []
        self.conn.set_trace_callback(queries.append)
        engine._capture_reconnect_receipt(operation)
        self.assertIsNone(engine._reconnect_receipt)
        self.assertEqual(queries, [])
        self.assertFalse(self.conn.in_transaction)

    def test_closed_initialized_fts_handle_reopens_without_owner_or_schema_writes(self):
        engine = self.engine()
        self.conn.execute("INSERT INTO messages(id,content) VALUES(881,'synthetic prior source')")
        self.conn.commit()
        candidate = self.candidate(engine)
        queries = []
        candidate.set_trace_callback(queries.append)
        owner = self.forbid_owner()
        self.conn.close()
        engine._open_connection_handle()
        self.assertIs(engine.conn, candidate)
        self.assertTrue(engine.ready)
        self.assertTrue(engine._runtime_initialized)
        self.assertFalse(engine._has_vectors)
        self.assertFalse(candidate.in_transaction)
        self.assertEqual(candidate.execute("SELECT content FROM messages WHERE id=881").fetchone()[0], "synthetic prior source")
        self.connect.assert_called_once_with(engine.db_path.as_uri() + "?mode=rw", uri=True, check_same_thread=False)
        self.assertIn("PRAGMA quick_check(1)", queries)
        self.assertIn("PRAGMA foreign_keys=ON", queries)
        self.assertFalse(any(q.lstrip().split(" ", 1)[0].upper() in {"INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"} for q in queries))
        owner.assert_not_called()

    def test_persistent_clean_ioerr_retires_old_handle_only_after_validation(self):
        engine = self.engine()
        candidate = self.candidate(engine)
        self.forbid_owner()
        self.conn.armed, self.conn.fault_contains = True, "PRAGMA schema_version"
        self.conn.fault = sqlite3.OperationalError("disk I/O error")
        with patch.object(self.conn, "close", wraps=self.conn.close) as close:
            engine._open_connection_handle()
            close.assert_called_once_with()
        self.assertIs(engine.conn, candidate)
        self.assertTrue(engine.ready)

    def test_receipt_does_not_bypass_borrowed_ioerr_or_unrelated_error(self):
        engine = self.engine()
        self.candidate(engine)
        self.forbid_owner()
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                if borrowed:
                    self.conn.execute("INSERT INTO messages(id,content) VALUES(882,'synthetic pending')")
                error = sqlite3.OperationalError("disk I/O error" if borrowed else "database is locked")
                self.conn.armed, self.conn.fault_contains, self.conn.fault = True, "PRAGMA schema_version", error
                with self.assertRaises(sqlite3.OperationalError) as raised:
                    engine._open_connection_handle()
                self.assertIs(raised.exception, error)
                self.assertEqual(self.conn.in_transaction, borrowed)
                self.conn.armed = False
        self.connect.assert_not_called()
        self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=882").fetchone()[0], "synthetic pending")

    def test_new_engine_without_receipt_retains_owner_required_refusal(self):
        engine = self.engine()
        engine.conn = None
        engine._reconnect_receipt = None
        owner = self.real_memory_owner(lambda path: None)
        self.modules["maintenance"] = owner
        self.candidate(engine)
        with self.assertRaises(owner.MaintenanceBusyError):
            engine._open_connection_handle()
        self.assertIsNone(engine.conn)
        self.connect.assert_not_called()

    def test_explicit_close_clears_receipt_and_next_open_requires_owner(self):
        engine = self.engine()
        engine.close()
        self.assertIsNone(engine._reconnect_receipt)
        owner = self.real_memory_owner(lambda path: None)
        self.modules["maintenance"] = owner
        with self.assertRaises(owner.MaintenanceBusyError):
            engine._open_connection_handle()
        self.assertIsNone(engine.conn)

    def test_healthy_ready_handle_has_no_receipt_sql_or_file_stat(self):
        engine = self.engine()
        self.candidate(engine)
        self.forbid_owner()
        queries = []
        self.conn.set_trace_callback(queries.append)
        with patch.object(Path, "stat", side_effect=AssertionError("Ready path stat")):
            engine._open_connection_handle()
        self.assertEqual(queries, ["PRAGMA schema_version"])
        self.connect.assert_not_called()

    def test_file_schema_registry_or_embedding_change_refuses_receipt(self):
        for change in ("inode", "schema", "registry", "model", "dimension", "partial", "oversized"):
            with self.subTest(change=change):
                engine = self.engine()
                candidate = self.candidate(engine)
                if change == "inode":
                    self.stat.st_ino += 1
                elif change == "schema":
                    candidate.execute("CREATE TABLE synthetic_new_schema(x)")
                elif change == "registry":
                    candidate.execute("INSERT INTO vector_cache_registry(tier_group,vec_table,sep_table) VALUES('edge','vec_messages_edge','vec_messages_sep_edge')")
                else:
                    candidate.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('embed_model',?)",
                                      ("different" if change == "model" else "x" * 1024 if change == "oversized" else "model2vec",))
                    if change != "partial":
                        candidate.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('embed_dim',?)", ("384" if change == "dimension" else "256",))
                candidate.commit()
                self.assertIsNone(engine._reopen_initialized_connection(engine.db_path))
                if change == "inode":
                    self.connect.assert_not_called()
                else:
                    self.assert_closed(candidate)
                self.assertIs(engine.conn, self.conn)
                self.assertTrue(engine.ready)

    def test_matching_metadata_published_after_initialization_is_accepted(self):
        engine = self.engine()
        candidate = self.candidate(engine)
        candidate.executemany("INSERT INTO metadata(key,value) VALUES(?,?)", [("embed_model", "model2vec"), ("embed_dim", "256")])
        candidate.commit()
        replacement = engine._reopen_initialized_connection(engine.db_path)
        self.assertEqual(replacement, (candidate, False))
        self.assertFalse(candidate.in_transaction)

    def test_quick_check_failure_closes_candidate_preserves_old_handle_and_ready(self):
        for failure in ("disk I/O error", "database disk image is malformed"):
            with self.subTest(failure=failure):
                engine = self.engine()
                candidate = self.candidate(engine)
                candidate.armed, candidate.fault_contains = True, "PRAGMA quick_check(1)"
                candidate.fault = sqlite3.OperationalError(failure)
                self.conn.armed, self.conn.fault_contains = True, "PRAGMA schema_version"
                self.conn.fault = sqlite3.OperationalError("disk I/O error")
                with patch.object(self.conn, "close", wraps=self.conn.close) as close:
                    with self.assertRaises(self.modules["storage"].DatabaseOpenError) as raised:
                        engine._open_connection_handle()
                    close.assert_not_called()
                self.assertIn("Close ALL" if "I/O" in failure else "appears corrupt", str(raised.exception))
                self.assert_closed(candidate)
                self.assertIs(engine.conn, self.conn)
                self.assertTrue(engine.ready)
                self.conn.armed = False

    def test_recheck_uses_healthy_replacement_and_closes_unused_candidate(self):
        engine = self.engine()
        candidate = self.candidate(engine)
        healthy = self.memory_connection()
        self.forbid_owner()
        self.conn.armed, self.conn.fault_contains = True, "PRAGMA schema_version"
        self.conn.fault = sqlite3.OperationalError("disk I/O error")
        def connect(*args, **kwargs):
            engine.conn = healthy
            return candidate
        self.connect.side_effect = connect
        with patch.object(self.conn, "close", wraps=self.conn.close) as close:
            engine._open_connection_handle()
            close.assert_not_called()
        self.assertIs(engine.conn, healthy)
        self.assert_closed(candidate)

    def test_changed_file_after_candidate_validation_is_refused(self):
        engine = self.engine()
        candidate = self.candidate(engine)
        enter_context(self, patch.object(Path, "stat", side_effect=[self.stat, types.SimpleNamespace(st_dev=7, st_ino=12)]))
        self.assertIsNone(engine._reopen_initialized_connection(engine.db_path))
        self.assert_closed(candidate)

    def test_vector_metadata_absence_requires_both_tables_empty_and_extension_registration(self):
        for table in (None, "vec_messages", "vec_messages_sep"):
            with self.subTest(table=table):
                engine = self.engine(vectors=True)
                candidate = self.candidate(engine)
                if table:
                    candidate.execute(f'INSERT INTO "{table}"(rowid) VALUES(1)')
                    candidate.commit()
                load = Mock()
                original_import = self.imports
                def imports(name, *args, **kwargs):
                    if name == "sqlite_vec":
                        return types.SimpleNamespace(load=load)
                    return original_import(name, *args, **kwargs)
                self.engine_namespace["__builtins__"]["__import__"] = imports
                with patch.object(candidate, "enable_load_extension", create=True) as enabled:
                    result = engine._reopen_initialized_connection(engine.db_path)
                load.assert_called_once_with(candidate)
                self.assertEqual([c.args for c in enabled.call_args_list], [(True,), (False,)])
                if table is None:
                    self.assertEqual(result, (candidate, False))
                else:
                    self.assertIsNone(result)
                    self.assert_closed(candidate)



    def test_policy_supersedes_legacy_ready_flags_without_legacy_metadata_reuse(self):
        engine = self.engine()
        policy = TestPublicBoundaries.seed_policy(self)
        candidate = self.candidate(engine)
        self.forbid_owner()
        self.conn.close()
        engine._open_connection_handle()
        self.assertIs(engine.conn, candidate)
        self.assertFalse(engine.ready)
        self.assertFalse(engine._runtime_initialized)
        self.assertFalse(engine._has_vectors)
        self.assertIsNone(engine._reconnect_receipt)
        self.assertIsNone(engine._runtime_vector_connection)
        self.assertEqual(self.api._read_policy(candidate), policy)
        self.assertEqual(self.fixture.loads, [])

    def test_registry_progress_mutation_invalidates_same_schema_receipt(self):
        engine = self.engine()
        self.conn.execute("INSERT INTO vector_cache_registry(tier_group,vec_table,sep_table) VALUES('edge','vec_messages_edge','vec_messages_sep_edge')")
        self.conn.commit()
        operation = types.SimpleNamespace(key=self.api._legacy_key(), _database=("file", str(engine.db_path)))
        with engine._write_lock:
            engine._capture_reconnect_receipt(operation)
        candidate = self.candidate(engine)
        candidate.execute("UPDATE vector_cache_registry SET vector_count=1 WHERE tier_group='edge'")
        candidate.commit()
        self.assertEqual(candidate.execute("PRAGMA schema_version").fetchone()[0], engine._reconnect_receipt[3])
        self.assertIsNone(engine._reopen_initialized_connection(engine.db_path))
        self.assert_closed(candidate)

    def test_registry_record_bounds_and_pending_caller_sql_prevent_capture(self):
        engine = self.engine()
        operation = types.SimpleNamespace(key=self.api._legacy_key(), _database=("file", str(engine.db_path)))
        self.conn.execute("INSERT INTO vector_cache_registry(tier_group,vec_table,sep_table,model_name) VALUES('edge','vec_messages_edge','vec_messages_sep_edge',?)", ("x" * 513,))
        self.conn.commit()
        with engine._write_lock:
            engine._capture_reconnect_receipt(operation)
        self.assertIsNone(engine._reconnect_receipt)
        self.conn.execute("INSERT INTO messages(id,content) VALUES(883,'synthetic caller')")
        before = self.conn.commits, self.conn.rollbacks
        with engine._write_lock:
            engine._capture_reconnect_receipt(operation)
        self.assertIsNone(engine._reconnect_receipt)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)

    def test_bad_integrity_result_and_late_stat_failure_close_only_candidate(self):
        for fault in ("quick_check", "stat"):
            with self.subTest(fault=fault):
                engine = self.engine()
                candidate = self.candidate(engine)
                execute = candidate.execute
                def checked(sql, *args, **kwargs):
                    if sql == "PRAGMA quick_check(1)":
                        return types.SimpleNamespace(fetchone=lambda: ("synthetic corruption",))
                    return execute(sql, *args, **kwargs)
                with contextlib.ExitStack() as stack:
                    if fault == "quick_check":
                        stack.enter_context(patch.object(candidate, "execute", side_effect=checked))
                        expected = self.modules["storage"].DatabaseOpenError
                    else:
                        stack.enter_context(patch.object(Path, "stat", side_effect=[self.stat, OSError("synthetic unavailable file")]))
                        expected = OSError
                    with self.assertRaises(expected):
                        engine._reopen_initialized_connection(engine.db_path)
                self.assert_closed(candidate)
                self.assertIs(engine.conn, self.conn)
                self.assertTrue(engine.ready)
                self.assertFalse(self.conn.in_transaction)

    def test_deprecated_open_legacy_callback_is_existing_only_without_migrations(self):
        engine = self.engine()
        candidate = self.candidate(engine)
        queries = []
        candidate.set_trace_callback(queries.append)
        module = definitions("engine.py", {"open"}, self.namespace(Path=Path,
            sqlite3=self.engine_namespace["sqlite3"], DEFAULT_BUSY_TIMEOUT_MS=3000,
            warnings=types.SimpleNamespace(warn=lambda *args, **kwargs: None)), methods=True)
        class Completed(Exception):
            pass
        def opener(path, callback):
            self.assertIs(callback(engine.db_path), candidate)
            raise Completed()
        with patch.object(self.api, "open_serving_connection", side_effect=opener), patch.object(Path, "exists", return_value=True):
            with self.assertRaises(Completed):
                module.open(engine)
        self.connect.assert_called_once_with(engine.db_path.as_uri() + "?mode=rw", uri=True, check_same_thread=False)
        self.assertEqual(queries, ["PRAGMA journal_mode=WAL", "PRAGMA busy_timeout=3000", "PRAGMA foreign_keys=ON",
                                   "PRAGMA synchronous=NORMAL", "PRAGMA cache_size=-64000", "PRAGMA mmap_size=268435456"])
        self.assertIs(engine.conn, self.conn)
        self.assertIsNone(candidate.row_factory)

    def test_deprecated_open_legacy_callback_failure_closes_only_candidate(self):
        engine = self.engine()
        candidate = self.candidate(engine)
        candidate.armed, candidate.fault_contains = True, "PRAGMA foreign_keys=ON"
        candidate.fault = sqlite3.OperationalError("synthetic pragma failure")
        module = definitions("engine.py", {"open"}, self.namespace(Path=Path,
            sqlite3=self.engine_namespace["sqlite3"], DEFAULT_BUSY_TIMEOUT_MS=3000,
            warnings=types.SimpleNamespace(warn=lambda *args, **kwargs: None)), methods=True)
        def opener(path, callback):
            return callback(engine.db_path)
        with patch.object(self.api, "open_serving_connection", side_effect=opener), patch.object(Path, "exists", return_value=True):
            with self.assertRaises(sqlite3.OperationalError):
                module.open(engine)
        self.assert_closed(candidate)
        self.assertIs(engine.conn, self.conn)
        self.assertTrue(engine.ready)



    def test_nul_tail_metadata_and_registry_are_rejected_before_materialization(self):
        engine = self.engine()
        candidate = self.candidate(engine)
        candidate.executemany("INSERT INTO metadata(key,value) VALUES(?,?)", [
            ("embed_model", "model2vec\0" + "x" * 10000), ("embed_dim", "256")])
        candidate.commit()
        self.assertIsNone(engine._reopen_initialized_connection(engine.db_path))
        self.assert_closed(candidate)
        self.conn.execute("INSERT INTO vector_cache_registry(tier_group,vec_table,sep_table,model_name) VALUES('edge','vec_messages_edge','vec_messages_sep_edge',?)",
                          ("synthetic\0" + "x" * 10000,))
        self.conn.commit()
        statements = []
        self.conn.set_trace_callback(statements.append)
        self.assertIsNone(engine._reconnect_registry(self.conn))
        self.assertFalse(any(sql.startswith("SELECT tier_group,vec_table") for sql in statements))

    def test_capture_cleanup_failure_never_publishes_receipt(self):
        engine = self.engine()
        operation = types.SimpleNamespace(key=self.api._legacy_key(), _database=("file", str(engine.db_path)))
        with engine._write_lock, patch.object(self.conn, "rollback", side_effect=sqlite3.OperationalError("synthetic rollback failure")):
            with self.assertRaises(sqlite3.OperationalError):
                engine._capture_reconnect_receipt(operation)
        self.assertIsNone(engine._reconnect_receipt)
        self.assertTrue(self.conn.in_transaction)
        self.assertIs(engine.conn, self.conn)
        self.conn.rollback()

    def test_registry_rows_are_copied_to_primitive_tuples(self):
        engine = self.engine()
        self.conn.execute("INSERT INTO vector_cache_registry(tier_group,vec_table,sep_table) VALUES('edge','vec_messages_edge','vec_messages_sep_edge')")
        self.conn.commit()
        self.conn.row_factory = sqlite3.Row
        operation = types.SimpleNamespace(key=self.api._legacy_key(), _database=("file", str(engine.db_path)))
        with engine._write_lock:
            engine._capture_reconnect_receipt(operation)
        receipt = engine._reconnect_receipt
        self.assertIsNotNone(receipt)
        self.assertIs(type(receipt[6]), tuple)
        self.assertIs(type(receipt[6][0]), tuple)
        self.assertTrue(all(type(value) in (str, int, float, type(None)) for value in receipt[6][0]))



    def test_silent_capture_rollback_cannot_publish_receipt(self):
        engine = self.engine()
        operation = types.SimpleNamespace(key=self.api._legacy_key(), _database=("file", str(engine.db_path)))
        with engine._write_lock, patch.object(self.conn, "rollback", return_value=None) as rollback:
            with self.assertRaises(self.api.TierRuntimeError):
                engine._capture_reconnect_receipt(operation)
            rollback.assert_called_once_with()
        self.assertIsNone(engine._reconnect_receipt)
        self.assertTrue(self.conn.in_transaction)
        self.assertIs(engine.conn, self.conn)
        self.conn.rollback()

    def test_silent_candidate_rollback_closes_only_candidate(self):
        engine = self.engine()
        candidate = self.candidate(engine)
        self.conn.armed, self.conn.fault_contains = True, "PRAGMA schema_version"
        self.conn.fault = sqlite3.OperationalError("disk I/O error")
        before = engine._reconnect_receipt
        with patch.object(candidate, "rollback", return_value=None) as rollback, \
                patch.object(self.conn, "close", wraps=self.conn.close) as close:
            with self.assertRaises(self.api.TierRuntimeError):
                engine._open_connection_handle()
            rollback.assert_called_once_with()
            close.assert_not_called()
        self.assert_closed(candidate)
        self.assertIs(engine.conn, self.conn)
        self.assertEqual(engine._reconnect_receipt, before)
        self.assertTrue(engine.ready)
        self.assertTrue(engine._runtime_initialized)
        self.assertFalse(self.conn.in_transaction)



class TestColdPublicInitialization(unittest.TestCase):
    imports = TestPublicBoundaries.imports
    namespace = TestPublicBoundaries.namespace

    def setUp(self):
        TestPublicBoundaries.setUp(self)
        import re
        self.events = []
        self.conn.enable_load_extension = lambda value: self.events.append(("extension", value))
        native = types.SimpleNamespace(load=lambda conn: self.events.append(("load", conn)))
        original_import = self.imports
        def imports(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sqlite_vec":
                return native
            return original_import(name, globals, locals, fromlist, level)
        self.imports = imports
        original_execute = self.conn.execute
        def execute(sql, parameters=()):
            match = re.fullmatch(r"CREATE VIRTUAL TABLE IF NOT EXISTS (\w+) USING vec0\(embedding float\[(\d+)\] distance_metric=cosine\)", sql)
            if match:
                sql = f'CREATE TABLE IF NOT EXISTS {match[1]} (rowid INTEGER PRIMARY KEY, embedding "float[{match[2]}] distance_metric=cosine")'
            return original_execute(sql, parameters)
        self.conn.execute = execute
        known = ("vec_messages", "vec_messages_sep", "vec_messages_edge", "vec_messages_sep_edge",
                 "vec_messages_basepro", "vec_messages_sep_basepro")
        for table in (*known, "vec_messages_custom", "vec_messages_sep_custom"):
            self.conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        self.conn.commit()
        self.vector = definitions("vector_search.py", {
            "_active_vec_table", "_active_sep_table", "_ensure_metadata_table", "_read_embedder_metadata",
            "_detect_existing_vec_dim", "_migration_hint", "_check_embedder_compatibility", "_table_uses_cosine",
            "_vec0_column_decl", "init_vec_table", "migrate_to_cosine_metric", "migrate_legacy_vec_tables",
        }, self.namespace(sqlite3=sqlite3, re=re, EMBEDDING_MODEL="model2vec", _embedding_dim=256,
            _KNOWN_VEC_TABLES=known, _VALID_GROUPS={"edge", "basepro"}, _MODEL_TO_GROUP={"model2vec": "edge"},
            _VEC_DISTANCE_METRIC="cosine", _active_tier_group=lambda: "edge",
            logger=logging.getLogger("synthetic-initialization"), TrueMemoryMigrationError=RuntimeError))
        self.fixture.vector._active_vec_table = self.vector._active_vec_table
        self.fixture.vector._active_sep_table = self.vector._active_sep_table
        self.modules["vector_search"] = self.vector
        self.vector.search_vector_raw = lambda conn, query, limit: conn.execute(
            f'SELECT rowid, embedding FROM "{self.vector._active_vec_table(conn)}" LIMIT ?', (limit,)).fetchall()
        self.namespace_values = self.namespace(_HAS_VECTOR=True, _HAS_HYBRID=True,
            init_vec_table=self.vector.init_vec_table, TrueMemoryMigrationError=RuntimeError,
            resolve_tier=lambda: "edge", logger=logging.getLogger("synthetic-initialization"),
            engine_operation=self.api.engine_operation)
        self.methods = definitions("engine.py", {"_ensure_connection", "_initialize_connection", "_maybe_auto_consolidate",
            "_maybe_startup_consolidate", "search_vectors_raw"}, self.namespace_values, methods=True)
        self.engine = types.SimpleNamespace(conn=self.conn, ready=False, _runtime_initialized=False,
            _has_vectors=False, _has_hybrid=False, _has_consolidation=False, _has_style_vec=False,
            _init_lock=threading.Lock(), _write_lock=threading.Lock(), _open_connection_handle=lambda: None,
            _purge_legacy_entity_profile_summaries=Mock(), _reconnect_receipt=None)
        for name in ("_ensure_connection", "_initialize_connection", "_maybe_auto_consolidate",
                     "_maybe_startup_consolidate", "search_vectors_raw"):
            setattr(self.engine, name, types.MethodType(getattr(self.methods, name), self.engine))
        self.conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('embed_model','model2vec')")
        self.conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('embed_dim','256')")
        self.conn.commit()

    def generic(self, *, populated=True):
        for table in ("vec_messages", "vec_messages_sep"):
            self.conn.execute(f'CREATE TABLE {table}(rowid INTEGER PRIMARY KEY, embedding "float[256] distance_metric=cosine")')
            if populated:
                self.conn.execute(f"INSERT INTO {table}(rowid,embedding) VALUES(7,?)", (b"synthetic vector",))
        self.conn.commit()

    def test_public_search_admits_post_migration_pair(self):
        self.generic()
        captured = []
        actual = self.vector.search_vector_raw
        def search(*args, **kwargs):
            captured.append(self.api.current_operation(self.conn))
            return actual(*args, **kwargs)
        self.vector.search_vector_raw = search
        rows = self.engine.search_vectors_raw("synthetic", limit=3)
        self.assertEqual(rows, [(7, b"synthetic vector")])
        self.assertEqual(captured[0].tables, ("vec_messages_edge", "vec_messages_sep_edge"))
        self.assertTrue(captured[0]._defer_maintenance)
        self.assertIsNone(self.conn.execute("SELECT 1 FROM sqlite_master WHERE name='vec_messages'").fetchone())
        self.assertTrue(self.engine.ready)
        self.assertIsNone(self.api.current_operation(self.conn))

    def test_fact_scope_is_after_migration_and_before_storage(self):
        self.generic()
        with self.api.fact_operation(types.SimpleNamespace(_engine=self.engine)) as operation:
            self.assertEqual(operation.tables, ("vec_messages_edge", "vec_messages_sep_edge"))
            self.assertEqual(self.engine.search_vectors_raw("synthetic"), [(7, b"synthetic vector")])
            self.assertIs(self.api.current_operation(self.conn), operation)

    def test_nested_populated_generic_refuses_before_mutation(self):
        self.generic()
        queries = []
        self.conn.set_trace_callback(queries.append)
        with self.api.serving_operation(self.conn) as operation:
            with self.assertRaisesRegex(self.api.TierRuntimeError, "before entering"):
                self.engine._ensure_connection()
            self.assertIs(self.api.current_operation(self.conn), operation)
            self.assertEqual(operation.tables, ("vec_messages", "vec_messages_sep"))
        self.assertFalse(any(sql.lstrip().split()[0].upper() in {"CREATE", "DROP", "ALTER", "INSERT", "DELETE", "UPDATE"} for sql in queries))
        self.assertFalse(self.engine.ready)
        self.assertEqual(self.conn.execute("SELECT embedding FROM vec_messages").fetchall(), [(b"synthetic vector",)])

    def test_nested_first_empty_generic_can_initialize(self):
        with self.api.serving_operation(self.conn) as operation:
            self.engine._ensure_connection(_suppress_maintenance=True)
            self.assertIs(self.api.current_operation(self.conn), operation)
            self.assertEqual(operation.tables, ("vec_messages", "vec_messages_sep"))
        self.assertTrue(self.engine.ready)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM vec_messages").fetchone(), (0,))

    def test_nested_absent_generic_preserves_unrelated_cached_pair(self):
        self.conn.execute('CREATE TABLE vec_messages_custom(embedding BLOB)')
        self.conn.execute('CREATE TABLE vec_messages_sep_custom(embedding BLOB)')
        self.conn.execute("INSERT INTO vec_messages_custom VALUES(?)", (b"inactive synthetic",))
        self.conn.commit()
        with self.api.serving_operation(self.conn) as operation:
            self.engine._ensure_connection(_suppress_maintenance=True)
            self.assertEqual(operation.tables, ("vec_messages", "vec_messages_sep"))
        self.assertEqual(self.conn.execute("SELECT embedding FROM vec_messages_custom").fetchall(), [(b"inactive synthetic",)])
        self.assertTrue(self.engine.ready)
        self.assertTrue(self.engine._has_vectors)

    def test_nested_stable_tiered_pair_only_loads_connection_extension(self):
        self.generic()
        self.engine._ensure_connection(_suppress_maintenance=True)
        self.engine.ready = self.engine._runtime_initialized = False
        queries = []
        self.conn.set_trace_callback(queries.append)
        with self.api.serving_operation(self.conn) as operation:
            self.engine._ensure_connection(_suppress_maintenance=True)
            self.assertEqual(operation.tables, ("vec_messages_edge", "vec_messages_sep_edge"))
        self.assertTrue(self.engine.ready)
        self.assertFalse(any(sql.lstrip().split()[0].upper() in {"CREATE", "DROP", "ALTER", "INSERT", "DELETE", "UPDATE"} for sql in queries))

    def test_maintenance_waits_for_outer_scope_and_nested_crud_calls(self):
        self.generic()
        scheduled = []
        self.engine._has_consolidation = True
        self.modules["maintenance"].engine_layer_specs = lambda conn, coordinator: ()
        self.modules["maintenance"].observe_style = Mock(side_effect=AssertionError("Unexpected style"))
        self.modules["maintenance"].plan_layers = lambda *args, **kwargs: scheduled.append(self.api.current_runtime_operation()) or ()
        self.engine._get_maintenance_coordinator = lambda: object()
        self.engine._auto_consolidate_threshold = 32
        with self.api.engine_serving_operation(self.engine) as operation:
            self.engine._maybe_auto_consolidate()
            self.engine._ensure_connection()
            with self.api.engine_serving_operation(self.engine) as nested:
                self.assertIs(nested, operation)
                self.engine._maybe_auto_consolidate()
            self.assertEqual(scheduled, [])
        self.assertEqual(scheduled, [None])

    def test_borrowed_cold_initialization_preserves_caller_sql(self):
        self.conn.execute("INSERT INTO messages(id,content) VALUES(91,'synthetic caller')")
        before = (self.conn.commits, self.conn.rollbacks)
        with self.assertRaisesRegex(self.api.TierRuntimeError, "clean connection"):
            self.engine._ensure_connection()
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.assertFalse(self.engine.ready)

    def test_nested_custom_pair_preserves_tables_and_metadata(self):
        self.generic()
        self.engine._ensure_connection(_suppress_maintenance=True)
        self.conn.execute("ALTER TABLE vec_messages_edge RENAME TO vec_messages_custom")
        self.conn.execute("ALTER TABLE vec_messages_sep_edge RENAME TO vec_messages_sep_custom")
        self.conn.execute("UPDATE vector_cache_registry SET vec_table='vec_messages_custom', sep_table='vec_messages_sep_custom'")
        self.conn.commit()
        self.engine.ready = self.engine._runtime_initialized = False
        before = self.conn.total_changes
        with self.api.serving_operation(self.conn) as operation:
            self.engine._ensure_connection(_suppress_maintenance=True)
            self.assertEqual(operation.tables, ("vec_messages_custom", "vec_messages_sep_custom"))
        self.assertEqual(self.conn.total_changes, before)
        self.assertTrue(self.engine.ready)

    def test_nested_cosine_migration_refuses_without_changing_source(self):
        self.conn.execute('CREATE TABLE vec_messages(embedding "float[256]")')
        self.conn.execute('CREATE TABLE vec_messages_sep(embedding "float[256]")')
        with self.api.serving_operation(self.conn) as operation:
            before = self.conn.execute("PRAGMA schema_version").fetchone()
            with self.assertRaisesRegex(self.api.TierRuntimeError, "completed vector migration"):
                self.engine._ensure_connection()
            self.assertIs(self.api.current_operation(self.conn), operation)
            self.assertEqual(self.conn.execute("PRAGMA schema_version").fetchone(), before)
        self.assertFalse(self.engine.ready)

    def test_child_inherits_deferral_and_cannot_schedule_before_parent_exit(self):
        self.generic()
        scheduled, failures = [], []
        self.engine._has_consolidation = True
        self.modules["maintenance"].engine_layer_specs = lambda conn, coordinator: ()
        self.modules["maintenance"].observe_style = Mock(side_effect=AssertionError("Unexpected style"))
        self.modules["maintenance"].plan_layers = lambda *args, **kwargs: scheduled.append(self.api.current_runtime_operation()) or ()
        self.engine._get_maintenance_coordinator = lambda: object()
        self.engine._auto_consolidate_threshold = 32
        with self.api.engine_serving_operation(self.engine) as operation:
            child = operation.fork_child()
            def run():
                try:
                    with child.join() as inherited:
                        self.assertTrue(inherited._defer_maintenance)
                        self.assertEqual(inherited.tables, operation.tables)
                        self.engine._maybe_auto_consolidate()
                except BaseException as error:
                    failures.append(error)
            worker = threading.Thread(target=run)
            worker.start()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(scheduled, [])
        self.assertEqual(scheduled, [None])

    def test_synchronous_nan_repair_precedes_legacy_name_migration(self):
        self.generic()
        self.engine.db_path = Path(":memory:")
        self.vector.EMBEDDING_MODEL = self.fixture.vector.EMBEDDING_MODEL = "qwen3_256"
        self.vector._MODEL_TO_GROUP["qwen3_256"] = "basepro"
        self.vector._active_tier_group = lambda: "basepro"
        self.fixture.vector.resolve_tier = self.methods.resolve_tier = lambda: "base"
        self.conn.execute("UPDATE metadata SET value='qwen3_256' WHERE key='embed_model'")
        self.conn.execute("INSERT INTO messages(id,content) VALUES(7,'synthetic repair source')")
        self.conn.commit()
        repaired = []
        def rebuild(conn, *, separation=False):
            table = self.vector._active_sep_table(conn) if separation else self.vector._active_vec_table(conn)
            repaired.append(table)
            conn.execute(f'DELETE FROM "{table}"')
            conn.execute(f'INSERT INTO "{table}"(rowid,embedding) VALUES(7,?)', (b"synthetic repaired",))
            conn.commit()
        self.vector.build_vectors = rebuild
        self.vector.build_separation_vectors = lambda conn: rebuild(conn, separation=True)
        with patch.object(sys, "platform", "darwin"):
            with self.api.engine_serving_operation(self.engine) as operation:
                self.assertEqual(operation.tables, ("vec_messages_basepro", "vec_messages_sep_basepro"))
                self.assertEqual(self.conn.execute("SELECT embedding FROM vec_messages_basepro").fetchall(), [(b"synthetic repaired",)])
                self.assertEqual(self.conn.execute("SELECT embedding FROM vec_messages_sep_basepro").fetchall(), [(b"synthetic repaired",)])
        self.assertEqual(repaired, ["vec_messages", "vec_messages_sep"])
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='qwen3_nan_fix_applied'").fetchone(), ("1",))
        self.assertIsNone(self.conn.execute("SELECT 1 FROM sqlite_master WHERE name='vec_messages'").fetchone())


class TestBorrowedPublicCompatibility(unittest.TestCase):
    imports = TestPublicBoundaries.imports
    namespace = TestPublicBoundaries.namespace

    def setUp(self):
        TestPublicBoundaries.setUp(self)
        storage = self.modules["storage"]
        self.conn = storage.create_db(":memory:")
        self.addCleanup(self.conn.close)
        self.storage = storage
        fts = definitions("fts_search.py", {"search_fts", "_build_safe_query", "_fts_select",
            "_rows_to_results", "_normalize_scores"}, self.namespace(sqlite3=sqlite3,
            _deserialize_metadata=storage._deserialize_metadata, directive_filter_sql=storage.directive_filter_sql,
            select_message_cols=storage.select_message_cols))
        source = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
        cap = next(node.value for node in source.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "MAX_CONTENT_LENGTH" for target in node.targets))
        self.limit = ast.literal_eval(cap)
        validators = definitions("engine.py", {"_validate_add_content", "_validate_delete_user"},
                                 self.namespace(MAX_CONTENT_LENGTH=self.limit))
        self.methods = definitions("engine.py", {"_open_connection_handle", "_initialize_connection",
            "add", "delete_all", "search_simple"}, self.namespace(
            _validate_add_content=validators._validate_add_content, _validate_delete_user=validators._validate_delete_user,
            engine_operation=self.api.engine_operation, engine_handle_operation=self.api.engine_handle_operation,
            search_fts=fts.search_fts, sqlite3=sqlite3, _HAS_VECTOR=False, _HAS_HYBRID=False), methods=True)
        self.engine = types.SimpleNamespace(conn=self.conn, ready=False, _runtime_initialized=False,
            _write_lock=threading.Lock(), _init_lock=threading.Lock(), _has_vectors=False,
            _has_consolidation=False, _has_style_vec=False,
            _ensure_connection=Mock(side_effect=AssertionError("Unexpected body initialization")),
            _maybe_auto_consolidate=Mock(side_effect=AssertionError("Unexpected maintenance")),
            _maybe_startup_consolidate=Mock(side_effect=AssertionError("Unexpected startup maintenance")))
        for name in ("_open_connection_handle", "_initialize_connection", "add", "delete_all", "search_simple"):
            setattr(self.engine, name, types.MethodType(getattr(self.methods, name), self.engine))

    def borrow(self):
        self.conn.execute("BEGIN")
        identity = self.storage.insert_message(self.conn, {"content": "syntheticpending borrowed lexical source"})
        self.assertTrue(self.conn.in_transaction)
        self.queries = []
        self.conn.set_trace_callback(self.queries.append)
        return identity

    def assert_borrowed(self, identity):
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=?", (identity,)).fetchone(),
                         ("syntheticpending borrowed lexical source",))
        self.assertFalse(any(query.lstrip().split()[0].upper() in {"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE"}
                             for query in self.queries))
        self.assertFalse(self.engine.ready)
        self.assertFalse(self.engine._runtime_initialized)
        self.engine._maybe_auto_consolidate.assert_not_called()
        self.engine._maybe_startup_consolidate.assert_not_called()

    def test_simple_search_reads_borrowed_fts_without_vector_initialization(self):
        identity = self.borrow()
        with patch.object(self.engine, "_initialize_connection", side_effect=AssertionError("Unexpected vector initialization")) as initialize:
            results = self.engine.search_simple("syntheticpending")
        self.assertEqual([(row["id"], row["source"]) for row in results], [(identity, "fts")])
        self.assertEqual(results[0]["score"], 1.0)
        initialize.assert_not_called()
        self.assert_borrowed(identity)

    def test_simple_search_clean_cold_engine_stays_fts_only(self):
        identity = self.storage.insert_message(self.conn, {"content": "synthetic lexical only"})
        self.conn.commit()
        with patch.object(self.engine, "_initialize_connection", side_effect=AssertionError("Unexpected vector initialization")) as initialize:
            results = self.engine.search_simple("synthetic")
        self.assertEqual([(row["id"], row["source"]) for row in results], [(identity, "fts")])
        initialize.assert_not_called()
        self.assertFalse(self.engine.ready)
        self.engine._maybe_auto_consolidate.assert_not_called()
        self.engine._maybe_startup_consolidate.assert_not_called()

    def test_simple_search_propagates_typed_refusal_without_ending_caller_sql(self):
        identity = self.borrow()
        error = self.api.TierRuntimeError("synthetic admission changed")
        self.methods.search_fts = Mock(side_effect=error)
        with self.assertRaises(self.api.TierRuntimeError) as raised:
            self.engine.search_simple("syntheticpending")
        self.assertIs(raised.exception, error)
        self.assert_borrowed(identity)

    def test_invalid_add_precedes_any_handle_or_sql_admission(self):
        identity = self.borrow()
        with patch.object(self.engine, "_open_connection_handle", side_effect=AssertionError("Unexpected handle admission")) as opening:
            with self.assertRaises(TypeError) as raised:
                self.engine.add(None)
        self.assertEqual(str(raised.exception), "content must be a string, got NoneType")
        self.assertEqual(self.queries, [])
        opening.assert_not_called()
        self.assert_borrowed(identity)

    def test_oversized_add_keeps_exact_limit_error_and_caller_sql(self):
        identity = self.borrow()
        with patch.object(self.engine, "_open_connection_handle", side_effect=AssertionError("Unexpected handle admission")) as opening:
            with self.assertRaises(ValueError) as raised:
                self.engine.add(content="x" * (self.limit + 1))
        self.assertEqual(str(raised.exception), f"Content too large ({self.limit + 1} chars). Maximum is {self.limit}.")
        self.assertEqual(self.queries, [])
        opening.assert_not_called()
        self.assert_borrowed(identity)

    def test_valid_add_arguments_still_refuse_cold_vector_initialization_in_caller_sql(self):
        identity = self.borrow()
        for content in ("", "synthetic", "x" * self.limit):
            with self.subTest(size=len(content)), self.assertRaisesRegex(self.api.TierRuntimeError, "clean connection"):
                self.engine.add(content)
        self.assert_borrowed(identity)

    def test_invalid_delete_scope_precedes_handle_or_sql_admission(self):
        identity = self.borrow()
        for value, error_type, message in ((3, TypeError, "user_id must be a string or None, got int"),
                ([], TypeError, "user_id must be a string or None, got list"),
                ("", ValueError, "user_id cannot be an empty string"),
                (" \t", ValueError, "user_id cannot be an empty string")):
            with self.subTest(value=value), patch.object(self.engine, "_open_connection_handle", side_effect=AssertionError("Unexpected handle admission")) as opening:
                with self.assertRaises(error_type) as raised:
                    self.engine.delete_all(user_id=value)
                self.assertEqual(str(raised.exception), message)
                opening.assert_not_called()
        self.assertEqual(self.queries, [])
        self.assert_borrowed(identity)

    def test_valid_delete_scope_still_requires_clean_cold_initialization(self):
        identity = self.borrow()
        for value in (None, "synthetic"):
            with self.subTest(value=value), self.assertRaisesRegex(self.api.TierRuntimeError, "clean connection"):
                self.engine.delete_all(user_id=value)
        self.assertFalse(any(query.lstrip().upper().startswith("DELETE") for query in self.queries))
        self.assert_borrowed(identity)


if __name__ == "__main__":
    unittest.main()
