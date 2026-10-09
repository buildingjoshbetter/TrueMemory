"""Runtime selection with in-memory journals and inert model-slot doubles."""

from __future__ import annotations

import ast
import builtins
import dataclasses
import gc
import os
import runpy
import sqlite3
import sys
import threading
import types
import unittest
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
JOURNAL = runpy.run_path(str(Path(__file__).with_name("test-tier-activation-journal-795.py")))
GATE = runpy.run_path(str(Path(__file__).with_name("test-tier-serving-gate-795.py")))
PREPARED = runpy.run_path(str(Path(__file__).with_name("test-prepared-target-795.py")))


class Cancelled(BaseException):
    pass


class TestRuntimeBridge(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = JOURNAL["TestActivationJournal"]()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.conn = self.fixture.conn
        self.modules = self.fixture.modules
        self.gate = GATE["load"]()
        self.modules["tier_switch.serving"] = self.gate
        self.loads = []
        self.vector = types.SimpleNamespace(
            EMBEDDING_MODEL="model2vec", _embedding_dim=256, _model=object(),
            _frozen_embedding_target=None, resolve_tier=lambda: "edge",
            _active_vec_table=lambda conn: "vec_messages", _active_sep_table=lambda conn: "vec_messages_sep",
        )
        self.reranker = types.SimpleNamespace(
            _active_tier="edge", _model_name="synthetic/default", _model=object(), _frozen_reranker_id=None,
            _model_certified=False,
            get_current_reranker_name=lambda: "synthetic/default",
        )

        def embedding(target, **kwargs):
            self.assertFalse(self.conn.in_transaction)
            self.loads.append(("embedding", target))
            self.vector._frozen_embedding_target = target
            self.vector.EMBEDDING_MODEL, self.vector._embedding_dim = target.model_id, target.dimension
            self.vector._model = object()

        def reranking(tier, name, **kwargs):
            self.assertFalse(self.conn.in_transaction)
            self.loads.append(("reranker", name))
            self.reranker._active_tier, self.reranker._model_name = tier, name
            self.reranker._frozen_reranker_id = name
            self.reranker._model = object()
            self.reranker._model_certified = True

        self.vector.apply_frozen_embedding_target = embedding
        self.reranker.apply_frozen_reranker = reranking
        self.reranker.get_reranker = lambda model_name, **kwargs: reranking("edge", model_name, **kwargs)
        self.modules["vector_search"], self.modules["reranker"] = self.vector, self.reranker

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"numpy", "torch", "sentence_transformers", "model2vec", "sqlite_vec"}:
                raise AssertionError("Native/model imports are forbidden in safe bridge tests")
            if name == "truememory":
                return types.SimpleNamespace(vector_search=self.vector, reranker=self.reranker)
            if name == "truememory.tier_switch":
                return types.SimpleNamespace(serving=self.gate)
            if name.startswith("truememory."):
                short = name.removeprefix("truememory.")
                if short in self.modules:
                    return self.modules[short]
                raise AssertionError("Unexpected application import: " + name)
            return builtins.__import__(name, globals, locals, fromlist, level)

        self.safe_import = safe_import
        self.api = types.ModuleType("synthetic_runtime_bridge")
        sys.modules[self.api.__name__] = self.api
        self.api.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
        path = ROOT / "truememory/tier_switch/runtime.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), self.api.__dict__)
        self.modules["tier_switch.runtime"] = self.api
        self.api._database_identity = lambda conn: ("memory", conn)

    def select(self):
        intent, plan = self.fixture.build()
        return self.fixture.commit(intent, plan)

    def test_legacy_no_record_stays_lazy_and_bound_none_is_distinct(self) -> None:
        self.assertIsNone(self.api.current_operation(self.conn))
        before = (self.conn.commits, self.conn.rollbacks)
        with self.api.serving_operation(self.conn) as operation:
            self.assertIsNone(operation.selection)
            self.assertIs(self.api.current_operation(self.conn), operation)
            self.assertEqual(operation.tables, ("vec_messages", "vec_messages_sep"))
            self.assertEqual(operation.tier, "edge")
        self.assertEqual(self.loads, [])
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.assertIsNone(self.api.current_operation(self.conn))

    def test_selected_reconciles_exact_identity_before_body_and_ack(self) -> None:
        selected = self.select()
        with self.api.serving_operation(self.conn) as operation:
            self.assertEqual(operation.selection, selected)
            self.assertEqual(operation.tables, selected.tables)
            self.assertEqual(operation.reranker_id, selected.reranker_id)
            self.assertEqual(operation.tier, "custom")
            self.assertFalse(self.conn.in_transaction)
            self.assertEqual(self.api.runtime_acknowledgement(), self.api.selection_key(selected))
        self.assertEqual(self.loads, [("embedding", selected.target), ("reranker", selected.reranker_id)])
        with self.api.serving_operation(self.conn):
            pass
        self.assertEqual(len(self.loads), 2)

    def test_projection_rechecks_controls_after_readiness_modules(self) -> None:
        selected = self.select()
        cancelled = threading.Event()
        original = self.api._modules
        def modules():
            cancelled.set()
            return original()
        with patch.object(self.api, "_modules", side_effect=modules):
            with self.assertRaises(self.gate.ServingLeaseCancelled):
                self.api.apply_frozen_selection(selected, cancelled=cancelled)
        self.assertEqual(self.loads, [])
        self.assertIsNone(self.api.runtime_acknowledgement())

    def test_legacy_explicit_override_forwards_admission_controls(self) -> None:
        cancelled = threading.Event()
        observed = []
        original = self.reranker.get_reranker
        def loader(**kwargs):
            observed.append(kwargs)
            return original(**kwargs)
        with patch.object(self.reranker, "get_reranker", side_effect=loader):
            with self.api.serving_operation(self.conn, reranker_id="synthetic/deep", cancelled=cancelled):
                pass
        self.assertEqual(observed, [{"model_name": "synthetic/deep", "deadline": None, "cancelled": cancelled}])

    def test_nested_search_has_zero_selection_sql_and_reuses_context(self) -> None:
        self.select()
        queries = []
        self.conn.set_trace_callback(queries.append)
        with self.api.serving_operation(self.conn) as outer:
            before = len(queries)
            with self.api.serving_operation(self.conn) as inner:
                self.assertIs(inner, outer)
                self.assertIs(self.api.current_operation(self.conn), outer)
            self.assertEqual(len(queries), before)

    def test_outer_override_is_frozen_without_changing_journal(self) -> None:
        selected = self.select()
        with self.api.serving_operation(self.conn, reranker_id="synthetic/deep") as operation:
            self.assertEqual(operation.reranker_id, "synthetic/deep")
            self.assertEqual(operation.selection.reranker_id, selected.reranker_id)
            self.assertEqual(self.api.runtime_acknowledgement(), self.api.selection_key(selected, "synthetic/deep"))
            with self.api.serving_operation(self.conn) as nested:
                self.assertIs(nested, operation)
            with self.assertRaises(self.api.TierRuntimeError):
                with self.api.serving_operation(self.conn, reranker_id="synthetic/other"):
                    self.fail("Nested override cannot change the admitted model")
        self.assertEqual(self.fixture.api.read_activation_state(self.conn).selection, selected)

    def test_same_thread_projection_upgrade_refuses(self) -> None:
        selected = self.select()
        with self.api.serving_operation(self.conn):
            with self.assertRaises(self.api.TierRuntimeError):
                self.api.apply_frozen_selection(selected)

    def test_projection_reenters_existing_exclusive_guard(self) -> None:
        selected = self.select()
        with self.gate.exclusive_activation():
            self.api.apply_frozen_selection(selected)
        self.assertEqual(self.api.runtime_acknowledgement(), self.api.selection_key(selected))

    def test_loader_failure_clears_ack_and_preserves_exception(self) -> None:
        selected = self.select()
        self.api.apply_frozen_selection(selected)
        changed = dataclasses.replace(selected, generation="f" * 32)
        for module, name in ((self.vector, "apply_frozen_embedding_target"), (self.reranker, "apply_frozen_reranker")):
            sentinel = Cancelled("synthetic")
            self.api._acknowledged = self.api.selection_key(selected)
            with patch.object(module, name, side_effect=sentinel):
                with self.assertRaises(Cancelled) as raised:
                    self.api.apply_frozen_selection(changed)
            self.assertIs(raised.exception, sentinel)
            self.assertIsNone(self.api.runtime_acknowledgement())

    def test_body_baseexception_clears_context_and_gate(self) -> None:
        sentinel = Cancelled("synthetic")
        with self.assertRaises(Cancelled) as raised:
            with self.api.serving_operation(self.conn):
                raise sentinel
        self.assertIs(raised.exception, sentinel)
        self.assertIsNone(self.api.current_operation(self.conn))
        with self.gate.exclusive_activation(timeout=1):
            pass

    def test_bound_different_connection_refuses_even_legacy_none(self) -> None:
        other = sqlite3.connect(":memory:")
        self.addCleanup(other.close)
        with self.api.serving_operation(self.conn):
            with self.assertRaises(self.api.TierRuntimeError):
                self.api.current_operation(other)
            with self.assertRaises(self.api.TierRuntimeError):
                with self.api.serving_operation(other):
                    self.fail("Different connection cannot inherit the operation")

    def test_borrowed_legacy_is_nonblocking_and_never_ends_sql(self) -> None:
        self.conn.execute("BEGIN")
        before = (self.conn.commits, self.conn.rollbacks)
        with self.api.serving_operation(self.conn) as operation:
            self.assertTrue(operation.borrowed)
            with self.assertRaises(self.api.TierRuntimeError):
                self.api.require_model_load_allowed()
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.assertEqual(self.loads, [])
        self.conn.rollback()

    def test_borrowed_selected_requires_existing_ready_runtime(self) -> None:
        selected = self.select()
        self.conn.execute("BEGIN")
        before = (self.conn.commits, self.conn.rollbacks)
        with self.assertRaises(self.api.TierRuntimeError):
            with self.api.serving_operation(self.conn):
                self.fail("Reconciliation cannot load in a borrowed transaction")
        self.assertEqual(self.loads, [])
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.conn.rollback()
        self.api.apply_frozen_selection(selected)
        self.conn.execute("BEGIN")
        before = (self.conn.commits, self.conn.rollbacks)
        with self.api.serving_operation(self.conn):
            pass
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
        self.conn.rollback()

    def test_selected_admission_never_lazy_loads_inside_body(self) -> None:
        self.select()
        with self.api.serving_operation(self.conn):
            with self.assertRaises(self.api.TierRuntimeError):
                self.api.require_model_load_allowed()

    def test_child_reservation_outlives_parent_and_clears_context(self) -> None:
        with self.api.serving_operation(self.conn) as parent:
            child = parent.fork_child()
        with child.join() as operation:
            self.assertIs(self.api.current_operation(self.conn), operation)
            self.assertEqual(operation.tables, parent.tables)
        self.assertIsNone(self.api.current_operation(self.conn))
        with self.assertRaises(self.gate.ServingLeaseError):
            with child.join():
                self.fail("Consumed child cannot join twice")

    def test_child_different_database_refuses_and_releases_admitted_pin(self) -> None:
        other = sqlite3.connect(":memory:")
        self.addCleanup(other.close)
        with self.api.serving_operation(self.conn) as parent:
            child = parent.fork_child()
        with self.assertRaises(self.api.TierRuntimeError):
            with child.join(other):
                self.fail("Different database cannot join")
        with self.gate.exclusive_activation(timeout=1):
            pass

    def test_child_explicit_matching_database_binds_own_clean_connection(self) -> None:
        other = sqlite3.connect(":memory:")
        self.addCleanup(other.close)
        self.api._database_identity = lambda conn: ("file", "synthetic-same-db")
        with self.api.serving_operation(self.conn) as parent:
            child = parent.fork_child()
        with child.join(other) as operation:
            self.assertIs(self.api.current_operation(other), operation)
            with self.assertRaises(self.api.TierRuntimeError):
                self.api.current_operation(self.conn)

    def test_forked_process_ack_is_not_ready(self) -> None:
        selected = self.select()
        self.api.apply_frozen_selection(selected)
        with patch.object(self.api.os, "getpid", return_value=self.api._pid + 1):
            self.assertIsNone(self.api.runtime_acknowledgement())

    def test_nonblocking_admission_attempts_mutex_once(self) -> None:
        gate = self.gate.ServingGate()
        lock = types.SimpleNamespace(acquire=lambda **kwargs: False)
        with patch.object(gate, "_condition", lock):
            with self.assertRaises(self.gate.ServingLeaseError):
                with gate.operation_lease(("synthetic",), blocking=False):
                    self.fail("Unavailable mutex cannot admit")
        with self.assertRaises(self.gate.ServingLeaseError):
            with gate.operation_lease(("synthetic",), blocking=0):
                self.fail("Only exact bool is accepted")

    def test_nonblocking_conflict_never_condition_waits(self) -> None:
        gate = self.gate.ServingGate()
        ticket = object()
        gate._waiting_writers.append(ticket)
        with patch.object(gate._condition, "wait", side_effect=AssertionError("Admission cannot wait")):
            with self.assertRaises(self.gate.ServingLeaseError):
                with gate.operation_lease(("synthetic",), blocking=False):
                    self.fail("Queued writer must block unrelated reader")
        gate._waiting_writers.clear()
        self.assertFalse(gate._readers)

    def test_nonblocking_nested_read_inherits_cancel_and_identity(self) -> None:
        gate = self.gate.ServingGate()
        cancelled = threading.Event()
        with gate.operation_lease(("synthetic",), cancelled=cancelled):
            with gate.operation_lease(("synthetic",), blocking=False):
                pass
            with self.assertRaises(self.gate.ServingLeaseError):
                with gate.operation_lease(("different",), blocking=False):
                    self.fail("Nested identity cannot change")
            cancelled.set()
            with self.assertRaises(self.gate.ServingLeaseCancelled):
                with gate.operation_lease(("synthetic",), blocking=False):
                    self.fail("Inherited cancellation precedes admission")

    def test_engine_and_hybrid_outer_boundaries_are_present(self) -> None:
        engine = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
        klass = next(node for node in engine.body if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
        for name in ("search", "search_agentic", "search_vectors_raw", "add", "update", "delete", "delete_all"):
            node = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertIn("engine_operation", [ast.unparse(value) for value in node.decorator_list])
        hybrid = ast.parse((ROOT / "truememory/hybrid.py").read_text(encoding="utf-8"))
        function = next(node for node in hybrid.body if isinstance(node, ast.FunctionDef) and node.name == "search_hybrid")
        self.assertIn("database_operation", [ast.unparse(value) for value in function.decorator_list])

    def test_cancellation_during_final_journal_read_refuses_body(self) -> None:
        self.select()
        event = threading.Event()
        original = self.api._read_selection
        calls = []
        def read(conn):
            value = original(conn)
            calls.append(value)
            if len(calls) == 2:
                event.set()
            return value
        with patch.object(self.api, "_read_selection", side_effect=read):
            with self.assertRaises(self.gate.ServingLeaseCancelled):
                with self.api.serving_operation(self.conn, cancelled=event):
                    self.fail("Cancellation after SQL must prevent body admission")
        self.assertIsNone(self.api.current_operation(self.conn))
        with self.gate.exclusive_activation(timeout=1):
            pass

    def test_deadline_consumed_by_final_journal_read_refuses_body(self) -> None:
        self.select()
        clock = [1000.0]
        original = self.api._read_selection
        calls = []
        def read(conn):
            value = original(conn)
            calls.append(value)
            if len(calls) == 2:
                clock[0] = 1002.0
            return value
        with patch.object(self.api.time, "monotonic", side_effect=lambda: clock[0]), patch.object(self.api, "_read_selection", side_effect=read):
            with self.assertRaises(self.gate.ServingLeaseTimeout):
                with self.api.serving_operation(self.conn, deadline=1001.0):
                    self.fail("Expired deadline after SQL must prevent body admission")

    def test_child_rechecks_inherited_event_after_new_connection_read(self) -> None:
        other = sqlite3.connect(":memory:")
        self.addCleanup(other.close)
        self.api._database_identity = lambda conn: ("file", "synthetic-same-db")
        event = threading.Event()
        with self.api.serving_operation(self.conn, cancelled=event) as operation:
            child = operation.fork_child()
        original = self.api._read_selection
        def read(conn):
            value = original(conn)
            event.set()
            return value
        with patch.object(self.api, "_read_selection", side_effect=read):
            with self.assertRaises(self.gate.ServingLeaseCancelled):
                with child.join(other):
                    self.fail("Child must enforce inherited cancellation after SQL")
        self.assertIsNone(self.api.current_operation(other))
        with self.gate.exclusive_activation(timeout=1):
            pass

    def test_nested_reservation_inherits_stricter_nested_control(self) -> None:
        event = threading.Event()
        with self.api.serving_operation(self.conn):
            with self.api.serving_operation(self.conn, cancelled=event) as operation:
                child = operation.fork_child()
        event.set()
        with self.assertRaises(self.gate.ServingLeaseCancelled):
            with child.join():
                self.fail("Nested control must reach the forked child")
        child.close()

    def test_actual_selected_engine_init_reconciles_before_validation_and_keeps_schedule(self) -> None:
        selected = self.select()
        tree = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
        source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
        source.body = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in {
            "_ensure_connection", "_initialize_connection", "_open_connection_handle", "close",
        }]
        calls = []
        def validate(conn):
            self.assertEqual(self.api.current_operation(conn).selection, selected)
            self.assertEqual(self.api.runtime_acknowledgement(), self.api.selection_key(selected))
            calls.append("validate")
        def imports(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sqlite_vec":
                return types.SimpleNamespace(load=lambda conn: calls.append("extension"))
            return self.safe_import(name, globals, locals, fromlist, level)
        namespace = {
            "__builtins__": dict(vars(builtins), __import__=imports),
            "_HAS_VECTOR": True, "_HAS_HYBRID": True, "init_vec_table": validate, "sqlite3": sqlite3,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[source], type_ignores=[])), "synthetic-engine-init", "exec"), namespace)
        engine = namespace["TrueMemoryEngine"]()
        engine.conn = self.conn
        self.conn.enable_load_extension = lambda enabled: None
        engine._write_lock = threading.Lock()
        engine._init_lock = threading.Lock()
        engine._runtime_initialized = False
        engine._open_connection_handle = lambda: calls.append("plain-open")
        engine._purge_legacy_entity_profile_summaries = lambda: calls.append("purge")
        engine._maybe_startup_consolidate = lambda: calls.append("startup-schedule")
        engine._maybe_auto_consolidate = lambda: calls.append("cached-schedule")
        engine._ensure_connection()
        self.assertEqual(calls, ["plain-open", "extension", "validate", "purge", "startup-schedule"])
        self.assertTrue(engine._has_vectors)
        self.assertTrue(engine._has_hybrid)
        engine._ensure_connection()
        self.assertEqual(calls[-2:], ["plain-open", "cached-schedule"])
        replacement = sqlite3.connect(":memory:", factory=JOURNAL["JournalConnection"])
        self.addCleanup(replacement.close)
        self.conn.backup(replacement)
        replacement.enable_load_extension = lambda enabled: None
        namespace["create_db"] = lambda path: replacement
        engine.db_path = Path(":memory:")
        engine._open_connection_handle = types.MethodType(namespace["TrueMemoryEngine"]._open_connection_handle, engine)
        self.conn.close()
        calls.clear()
        engine._ensure_connection()
        self.assertEqual(calls, ["extension", "validate", "purge", "startup-schedule"])
        self.assertIs(engine._runtime_vector_connection, replacement)
        self.assertTrue(engine._runtime_initialized)
        self.assertTrue(engine._has_vectors)
        self.assertTrue(engine.ready)
        engine.close()
        self.assertIsNone(engine._runtime_vector_connection)
        self.assertIsNone(engine._runtime_vector_generation)
        self.assertFalse(engine._runtime_initialized)


    def test_ready_legacy_borrowed_connection_does_not_reinitialize(self) -> None:
        tree = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
        source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
        source.body = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_initialize_connection"]
        namespace = {"__builtins__": dict(vars(builtins), __import__=self.safe_import)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[source], type_ignores=[])), "synthetic-legacy-ready", "exec"), namespace)
        engine = namespace["TrueMemoryEngine"]()
        engine.conn, engine.ready, engine._runtime_initialized = self.conn, True, False
        engine._init_lock = threading.Lock()
        engine._write_lock = threading.Lock()
        self.conn.execute("INSERT INTO messages(id,content) VALUES(901,'synthetic borrowed legacy')")
        before = self.conn.commits, self.conn.rollbacks
        with self.api.serving_operation(self.conn):
            engine._initialize_connection(_suppress_maintenance=True)
            self.assertTrue(self.conn.in_transaction)
            self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
            engine.ready = False
            with self.assertRaises(self.api.TierRuntimeError):
                engine._initialize_connection(_suppress_maintenance=True)

    def test_fresh_or_closed_handle_clears_legacy_ready_before_initialization(self) -> None:
        tree = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
        source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
        source.body = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_open_connection_handle"]
        replacement = sqlite3.connect(":memory:")
        self.addCleanup(replacement.close)
        namespace = {"sqlite3": sqlite3, "create_db": lambda path: replacement,
                     "__builtins__": dict(vars(builtins), __import__=self.safe_import)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[source], type_ignores=[])), "synthetic-legacy-reopen", "exec"), namespace)
        engine = namespace["TrueMemoryEngine"]()
        engine.db_path, engine._init_lock = Path(":memory:"), threading.Lock()
        engine._write_lock = threading.Lock()
        for prior in (None, self.conn):
            if prior is not None:
                prior.close()
            engine.conn, engine.ready, engine._runtime_initialized = prior, True, True
            engine._open_connection_handle()
            self.assertIs(engine.conn, replacement)
            self.assertFalse(engine.ready)
            self.assertFalse(engine._runtime_initialized)

    def test_managed_writer_transaction_waits_and_is_not_caller_borrowed(self) -> None:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE synthetic(value TEXT)")
        class Engine:
            pass
        engine = Engine()
        engine.conn = conn
        waiting, release, owned, entered = (threading.Event() for _ in range(4))
        failures = []
        api = self.api
        class ObservedLock(api.ConnectionWriteLock):
            def acquire(self, blocking=True, timeout=-1):
                if blocking and self.owns_other_transaction(conn):
                    waiting.set()
                return super().acquire(blocking, timeout)
        lock = ObservedLock(engine)
        def writer():
            try:
                with lock:
                    conn.execute("INSERT INTO synthetic VALUES('owned')")
                    owned.set()
                    if not release.wait(2):
                        raise AssertionError("Synthetic writer release timed out")
                    conn.commit()
            except BaseException as exc:
                failures.append(exc)
        def reader():
            try:
                with api._connection_read(conn, lock, api.time.monotonic() + 2, None):
                    self.assertFalse(conn.in_transaction)
                    self.assertEqual(conn.execute("SELECT value FROM synthetic").fetchall(), [("owned",)])
                    entered.set()
            except BaseException as exc:
                failures.append(exc)
        owner = threading.Thread(target=writer)
        contender = threading.Thread(target=reader)
        owner.start()
        try:
            self.assertTrue(owned.wait(2))
            contender.start()
            self.assertTrue(waiting.wait(2))
            self.assertFalse(entered.is_set())
        finally:
            release.set()
            owner.join(2)
            if contender.ident is not None:
                contender.join(2)
        self.assertFalse(owner.is_alive())
        self.assertFalse(contender.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(entered.is_set())

    def test_managed_lock_does_not_adopt_preexisting_caller_transaction(self) -> None:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE synthetic(value TEXT)")
        conn.execute("INSERT INTO synthetic VALUES('caller')")
        class Engine:
            pass
        engine = Engine()
        engine.conn = conn
        lock = self.api.ConnectionWriteLock(engine)
        owned, release = threading.Event(), threading.Event()
        failures = []
        def holder():
            try:
                with lock:
                    owned.set()
                    if not release.wait(2):
                        raise AssertionError("Synthetic holder release timed out")
            except BaseException as exc:
                failures.append(exc)
        thread = threading.Thread(target=holder)
        thread.start()
        try:
            self.assertTrue(owned.wait(2))
            with self.assertRaisesRegex(self.api.TierRuntimeError, "Borrowed connection admission is busy"):
                with self.api._connection_read(conn, lock, None, None):
                    self.fail("Caller SQL must not wait for an unrelated lock holder")
            self.assertTrue(conn.in_transaction)
            self.assertEqual(conn.execute("SELECT value FROM synthetic").fetchall(), [("caller",)])
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        conn.rollback()
        self.assertEqual(conn.execute("SELECT value FROM synthetic").fetchall(), [])

    def test_connection_probe_error_preserves_handle_under_both_locks(self) -> None:
        tree = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
        source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
        source.body = [node for node in source.body if isinstance(node, ast.FunctionDef)
                       and node.name == "_open_connection_handle"]
        namespace = {"sqlite3": sqlite3,
                     "__builtins__": dict(vars(builtins), __import__=self.safe_import)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[source], type_ignores=[])),
                     "synthetic-probe-failure", "exec"), namespace)
        engine = namespace["TrueMemoryEngine"]()
        engine._init_lock, engine._write_lock = threading.Lock(), threading.Lock()
        owner = self
        class Connection:
            in_transaction = False
            def execute(self, sql):
                owner.assertTrue(engine._init_lock.locked())
                owner.assertTrue(engine._write_lock.locked())
                raise sqlite3.OperationalError("synthetic transient probe failure")
            def close(self):
                owner.fail("A transient SQL error must not close a shared connection")
        engine.conn = Connection()
        original = engine.conn
        with self.assertRaisesRegex(sqlite3.OperationalError, "synthetic transient"):
            engine._open_connection_handle()
        self.assertIs(engine.conn, original)
        self.assertFalse(engine._init_lock.locked())
        self.assertFalse(engine._write_lock.locked())


    def lifecycle_engine(
        self, method: str, conn: sqlite3.Connection | None = None,
        create_db: Callable[[Path], sqlite3.Connection] | None = None,
    ) -> object:
        tree = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
        source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
        source.body = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == method]
        namespace = {"sqlite3": sqlite3, "create_db": create_db,
                     "__builtins__": dict(vars(builtins), __import__=self.safe_import)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[source], type_ignores=[])),
                     "synthetic-engine-lifecycle", "exec"), namespace)
        engine = namespace["TrueMemoryEngine"]()
        engine.conn = self.conn if conn is None else conn
        engine.db_path = Path(":memory:")
        engine._init_lock = threading.Lock()
        engine._write_lock = self.api.ConnectionWriteLock(engine)
        engine.ready = engine._has_vectors = engine._runtime_initialized = True
        engine._runtime_vector_generation = "synthetic-initialized"
        engine._runtime_vector_connection = engine.conn
        return engine

    @contextmanager
    def held_lifecycle_lock(self, lock: object) -> Iterator[None]:
        entered, release = threading.Event(), threading.Event()
        failures = []

        def holder() -> None:
            try:
                with lock:
                    entered.set()
                    if not release.wait(2):
                        raise AssertionError("Synthetic lifecycle holder release timed out")
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=holder)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            yield
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])

    def test_actual_open_borrowed_busy_preserves_caller_sql_at_each_lock(self) -> None:
        engine = self.lifecycle_engine("_open_connection_handle")
        self.conn.execute("INSERT INTO messages(id,content) VALUES(901,'synthetic caller SQL')")
        before = self.conn.commits, self.conn.rollbacks
        queries = []
        self.conn.set_trace_callback(queries.append)
        for lock in (engine._init_lock, engine._write_lock):
            with self.subTest(lock=type(lock).__name__):
                queries.clear()
                with self.held_lifecycle_lock(lock):
                    with self.assertRaisesRegex(self.api.TierRuntimeError, "Borrowed connection admission is busy"):
                        engine._open_connection_handle()
                self.assertEqual(queries, [])
                self.assertIs(engine.conn, self.conn)
                self.assertTrue(self.conn.in_transaction)
                self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
                self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=901").fetchone()[0],
                                 "synthetic caller SQL")
                self.assertFalse(engine._init_lock.locked())
                self.assertFalse(engine._write_lock.locked())
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages WHERE id=901").fetchone()[0], 0)

    def test_actual_open_clean_ioerr_reconnects_and_invalidates_runtime(self) -> None:
        coded = sqlite3.OperationalError("synthetic extended I/O error")
        coded.sqlite_errorcode = getattr(sqlite3, "SQLITE_IOERR", 10) | (7 << 8)
        for error in (sqlite3.OperationalError("disk I/O error"), coded):
            with self.subTest(error=str(error)):
                original = sqlite3.connect(":memory:", factory=JOURNAL["JournalConnection"])
                replacement = sqlite3.connect(":memory:")
                self.addCleanup(original.close)
                self.addCleanup(replacement.close)
                original.armed, original.fault_contains, original.fault = True, "PRAGMA schema_version", error
                opened = []

                def create(path: Path) -> sqlite3.Connection:
                    self.assertEqual(path, Path(":memory:"))
                    opened.append(path)
                    return replacement

                engine = self.lifecycle_engine("_open_connection_handle", original, create)
                with patch.object(original, "close", wraps=original.close) as close:
                    engine._open_connection_handle()
                    close.assert_called_once_with()
                self.assertEqual(opened, [Path(":memory:")])
                self.assertIs(engine.conn, replacement)
                self.assertFalse(engine.ready)
                self.assertFalse(engine._has_vectors)
                self.assertFalse(engine._runtime_initialized)
                self.assertIsNone(engine._runtime_vector_generation)
                self.assertIsNone(engine._runtime_vector_connection)
                self.assertFalse(engine._init_lock.locked())
                self.assertFalse(engine._write_lock.locked())

    def test_actual_open_borrowed_ioerr_preserves_handle_and_caller_sql(self) -> None:
        coded = sqlite3.OperationalError("synthetic extended I/O error")
        coded.sqlite_errorcode = getattr(sqlite3, "SQLITE_IOERR", 10) | (7 << 8)
        for error in (sqlite3.OperationalError("disk I/O error"), coded):
            with self.subTest(error=str(error)):
                engine = self.lifecycle_engine("_open_connection_handle")
                self.conn.execute("INSERT INTO messages(id,content) VALUES(901,'synthetic caller SQL')")
                before = self.conn.commits, self.conn.rollbacks
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
                self.assertTrue(engine._has_vectors)
                self.assertTrue(engine._runtime_initialized)
                self.assertEqual(engine._runtime_vector_generation, "synthetic-initialized")
                self.assertIs(engine._runtime_vector_connection, self.conn)
                self.assertFalse(engine._init_lock.locked())
                self.assertFalse(engine._write_lock.locked())
                self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=901").fetchone()[0],
                                 "synthetic caller SQL")
                self.conn.armed = False
                self.conn.rollback()
                self.assertEqual(self.conn.execute("SELECT count(*) FROM messages WHERE id=901").fetchone()[0], 0)

    def test_actual_open_extended_ioerr_without_sqlite_constant(self) -> None:
        class SQLiteWithoutIOErr:
            missing_constant_reads = 0

            def __getattr__(self, name: str) -> object:
                if name == "SQLITE_IOERR":
                    self.missing_constant_reads += 1
                    raise AttributeError(name)
                return getattr(sqlite3, name)

        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                sqlite_api = SQLiteWithoutIOErr()
                self.assertIs(sqlite_api.connect, sqlite3.connect)
                self.assertIs(sqlite_api.OperationalError, sqlite3.OperationalError)
                original = sqlite3.connect(":memory:", factory=JOURNAL["JournalConnection"])
                replacement = sqlite3.connect(":memory:")
                self.addCleanup(original.close)
                self.addCleanup(replacement.close)
                original.execute("CREATE TABLE synthetic(value TEXT)")
                if borrowed:
                    original.execute("INSERT INTO synthetic VALUES('synthetic caller SQL')")
                before = original.commits, original.rollbacks
                error = sqlite3.OperationalError("synthetic extended I/O error")
                error.sqlite_errorcode = 10 | (7 << 8)
                original.armed, original.fault_contains, original.fault = True, "PRAGMA schema_version", error
                opened = []

                def create(path: Path) -> sqlite3.Connection:
                    self.assertEqual(path, Path(":memory:"))
                    opened.append(path)
                    return replacement

                engine = self.lifecycle_engine("_open_connection_handle", original, create)
                engine._open_connection_handle.__globals__["sqlite3"] = sqlite_api
                with patch.object(original, "close", wraps=original.close) as close:
                    if borrowed:
                        with self.assertRaises(sqlite3.OperationalError) as raised:
                            engine._open_connection_handle()
                        self.assertIs(raised.exception, error)
                        close.assert_not_called()
                        self.assertEqual(opened, [])
                        self.assertIs(engine.conn, original)
                        self.assertTrue(original.in_transaction)
                        self.assertTrue(engine.ready)
                        self.assertTrue(engine._has_vectors)
                        self.assertTrue(engine._runtime_initialized)
                        self.assertEqual(engine._runtime_vector_generation, "synthetic-initialized")
                        self.assertIs(engine._runtime_vector_connection, original)
                        self.assertEqual(original.execute("SELECT value FROM synthetic").fetchall(),
                                         [("synthetic caller SQL",)])
                    else:
                        engine._open_connection_handle()
                        close.assert_called_once_with()
                        self.assertEqual(opened, [Path(":memory:")])
                        self.assertIs(engine.conn, replacement)
                        self.assertFalse(engine.ready)
                        self.assertFalse(engine._has_vectors)
                        self.assertFalse(engine._runtime_initialized)
                        self.assertIsNone(engine._runtime_vector_generation)
                        self.assertIsNone(engine._runtime_vector_connection)
                self.assertEqual(sqlite_api.missing_constant_reads, 1)
                self.assertEqual((original.commits, original.rollbacks), before)
                self.assertFalse(engine._init_lock.locked())
                self.assertFalse(engine._write_lock.locked())
                if borrowed:
                    original.armed = False
                    original.rollback()
                    self.assertEqual(original.execute("SELECT count(*) FROM synthetic").fetchone()[0], 0)

    def test_actual_selected_ready_borrowed_init_ignores_competing_writer(self) -> None:
        selected = self.select()
        self.api.apply_frozen_selection(selected)
        engine = self.lifecycle_engine("_initialize_connection")
        engine._runtime_vector_generation = selected.generation
        self.conn.execute("INSERT INTO messages(id,content) VALUES(901,'synthetic caller SQL')")
        before = self.conn.commits, self.conn.rollbacks
        loads = list(self.loads)
        queries = []
        self.conn.set_trace_callback(queries.append)
        with self.api.serving_operation(self.conn, connection_lock=engine._write_lock) as operation:
            self.assertTrue(operation.borrowed)
            with self.held_lifecycle_lock(engine._write_lock):
                queries.clear()
                with patch.object(self.api, "_connection_read", side_effect=AssertionError("Ready init must not acquire SQL locks")), \
                     patch.object(self.api, "apply_frozen_selection", side_effect=AssertionError("Ready init must not reload models")):
                    engine._initialize_connection(_suppress_maintenance=True)
                self.assertTrue(engine._write_lock.locked())
                self.assertEqual(queries, [])
                self.assertEqual(self.loads, loads)
                self.assertTrue(self.conn.in_transaction)
                self.assertEqual((self.conn.commits, self.conn.rollbacks), before)
                self.assertIs(self.api.current_operation(self.conn), operation)
        self.assertTrue(engine.ready)
        self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=901").fetchone()[0],
                         "synthetic caller SQL")
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages WHERE id=901").fetchone()[0], 0)


class TestRuntimeModels(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = PREPARED["TestPreparedTargets"]()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.vector = self.fixture.vector
        self.vector._frozen_embedding_target = None
        self.gate = GATE["load"]()
        self.operation = None
        self.runtime = types.SimpleNamespace(
            require_model_load_allowed=lambda: None, current_runtime_operation=lambda: self.operation,
            TierRuntimeError=RuntimeError,
        )
        package = types.ModuleType("truememory.tier_switch")
        package.__path__ = []
        modules = patch.dict(sys.modules, {
            "truememory.tier_switch": package,
            "truememory.tier_switch.serving": self.gate,
            "truememory.tier_switch.runtime": self.runtime,
        })
        modules.start()
        self.addCleanup(modules.stop)

    def test_actual_local_factory_releases_old_slot_before_constructor(self) -> None:
        class Model:
            pass
        self.vector._model = Model()
        old = weakref.ref(self.vector._model)
        def check():
            gc.collect()
            self.assertIsNone(old())
            self.assertIsNone(self.vector._model)
        self.fixture.build_hook = check
        self.vector.apply_frozen_embedding_target(self.fixture.target)
        self.assertEqual(self.vector.EMBEDDING_MODEL, "qwen3_256")
        self.assertEqual(self.vector._embedding_dim, 256)
        self.assertEqual(len(self.fixture.calls), 1)
        self.assertEqual(self.fixture.encoded[0][1], [self.fixture.target_module.TARGET_PROBE_TEXT])
        self.assertIsNotNone(self.vector._model)

    def test_failed_constructor_keeps_exact_target_and_empty_slot(self) -> None:
        sentinel = Cancelled("synthetic")
        def fail():
            raise sentinel
        self.fixture.build_hook = fail
        with self.assertRaises(Cancelled) as raised:
            self.vector.apply_frozen_embedding_target(self.fixture.target)
        self.assertIs(raised.exception, sentinel)
        self.assertIsNone(self.vector._model)
        self.assertEqual(self.vector._frozen_embedding_target, self.fixture.target)
        self.assertFalse(self.vector._prepared_target_slot.locked())

    def test_wrong_width_and_busy_prepared_slot_never_publish_active_model(self) -> None:
        self.fixture.width = 257
        with self.assertRaises(self.fixture.target_module.EmbeddingTargetError):
            self.vector.apply_frozen_embedding_target(self.fixture.target)
        self.assertIsNone(self.vector._model)
        self.fixture.width = None
        self.vector._prepared_target_slot.acquire()
        try:
            before = len(self.fixture.calls)
            with self.assertRaises(RuntimeError):
                self.vector.apply_frozen_embedding_target(self.fixture.target)
            self.assertEqual(len(self.fixture.calls), before)
        finally:
            self.vector._prepared_target_slot.release()

    def test_base_pro_identity_reuse_has_zero_second_native_work(self) -> None:
        self.vector.apply_frozen_embedding_target(self.fixture.target)
        original = self.vector._model
        before = (len(self.fixture.calls), len(self.fixture.encoded), self.vector._model_generation)
        pro = self.fixture.Target.capture("pro")
        self.vector.apply_frozen_embedding_target(pro)
        self.assertIs(self.vector._model, original)
        self.assertEqual(self.vector._frozen_embedding_target, pro)
        self.assertEqual((len(self.fixture.calls), len(self.fixture.encoded), self.vector._model_generation), before)

    def test_deadline_consumed_by_state_lock_prevents_constructor(self) -> None:
        clock = [1000.0]
        class Lock:
            def acquire(self, **kwargs):
                clock[0] = 1002.0
                return True
            def release(self):
                pass
        with patch.object(self.vector.time, "monotonic", side_effect=lambda: clock[0]), patch.object(self.vector, "_lock", Lock()):
            with self.assertRaises(TimeoutError):
                self.vector.apply_frozen_embedding_target(self.fixture.target, timeout=1)
        self.assertEqual(self.fixture.calls, [])

    def test_setter_and_unload_cannot_mutate_admitted_model(self) -> None:
        self.vector.apply_frozen_embedding_target(self.fixture.target)
        model = self.vector._model
        with self.gate.operation_lease(("synthetic",)):
            for action in (lambda: self.vector.set_embedding_model("edge"), self.vector.unload_model):
                with self.assertRaises(self.gate.ServingLeaseError):
                    action()
                self.assertIs(self.vector._model, model)

    def test_shared_builtin_retains_normal_fast_protocol_custom_uses_target_protocol(self) -> None:
        self.fixture.client.use_model_server = lambda: True
        self.fixture.client.np.integer = int
        self.fixture.client.PROTOCOL_VERSION = 1
        requests = []
        def request(value, timeout=None):
            requests.append(value)
            return self.fixture.server.handle_request(value)
        self.fixture.client._request_with_autostart = request
        self.vector.apply_frozen_embedding_target(self.fixture.target)
        self.vector._model.encode(["synthetic"], batch_size=2, show_progress_bar=False)
        self.assertEqual([entry["op"] for entry in requests], ["prepare_embed_target_v1", "embed_batched"])
        self.assertEqual(requests[-1]["tier"], "qwen3_256")
        custom = self.fixture.Target.capture("custom")
        requests.clear()
        self.vector.apply_frozen_embedding_target(custom)
        self.vector._model.encode(["synthetic"], batch_size=2, show_progress_bar=False)
        self.assertEqual([entry["op"] for entry in requests], ["prepare_embed_target_v1", "embed_target_v1"])
        self.assertEqual(requests[-1]["target"], custom.to_wire())

    def load_reranker(self):
        return PREPARED["load"]("reranker", definitions=True, namespace={
            "threading": threading, "contextmanager": contextmanager,
            "_active_tier": "edge", "_frozen_reranker_id": None,
            "_model_certified": False,
            "_model": None, "_model_name": "synthetic/old", "_lock": threading.Lock(),
            "_TIER_RERANKERS": {"edge": "synthetic/default"},
        })

    def test_reranker_actual_constructor_releases_old_reference_and_honors_exact_name(self) -> None:
        reranker = self.load_reranker()
        class Model:
            pass
        reranker._model = Model()
        old = weakref.ref(reranker._model)
        def build(name, **kwargs):
            gc.collect()
            self.assertIsNone(old())
            self.assertEqual(name, "synthetic/deep")
            return Model()
        with patch.dict(sys.modules, {"sentence_transformers": types.SimpleNamespace(CrossEncoder=build)}):
            result = reranker.get_reranker(model_name="synthetic/deep", device="cpu")
        self.assertIs(reranker._model, result)
        self.assertEqual(reranker._model_name, "synthetic/deep")

    def test_same_name_uncertified_proxy_is_replaced_without_daemon_preflight(self) -> None:
        reranker = self.load_reranker()
        reranker._model, reranker._model_name = object(), "synthetic/deep"
        old = reranker._model
        self.fixture.client.use_model_server = lambda: True
        with patch.object(self.fixture.client, "_request_with_autostart", side_effect=AssertionError("Unexpected preflight")):
            result = reranker.get_reranker("synthetic/deep", certified=True)
        self.assertIsNot(result, old)
        self.assertIsInstance(result, self.fixture.client.CertifiedRerankerProxy)
        self.assertTrue(reranker._model_certified)
        self.assertIs(reranker.get_reranker("synthetic/deep", certified=True), result)

    def test_reranker_setter_unload_and_nested_override_refuse_during_read(self) -> None:
        reranker = self.load_reranker()
        reranker._model = object()
        self.operation = types.SimpleNamespace(reranker_id="synthetic/default", selection=None)
        with self.gate.operation_lease(("synthetic",)):
            for action in (lambda: reranker.set_active_tier("pro"), reranker.unload_reranker,
                           lambda: reranker.get_reranker(model_name="synthetic/other")):
                with self.assertRaises(RuntimeError):
                    action()


@unittest.skipUnless(os.environ.get("TRUEMEMORY_TEST_NATIVE_VEC") == "1", "GPUBox-only native sqlite-vec composition")
class TestNativeRuntimeComposition(unittest.TestCase):
    def test_selected_engine_crud_search_and_reopen_use_one_pair(self) -> None:
        import numpy as np
        import sqlite_vec
        from truememory import engine as engine_module, model_client, reranker, storage, vector_search
        from truememory.embedding_target import EmbeddingTarget
        from truememory.tier_switch import activation, runtime

        target = EmbeddingTarget.capture("base")
        selection = activation.TierSelection(
            generation="1" * 32, intent_id="2" * 32, target=target,
            reranker_id="synthetic/reranker", tables=target.tables, job_id="3" * 32,
            tracker="canonical-v1", source_epoch="synthetic-epoch", source_schema_signature="4" * 64,
            manifest_generation="5" * 32, manifest_hash="6" * 64, receipt_hash="7" * 64,
            pair_digest="8" * 64, schema="9" * 64, vector_count=0, cursor=None,
        )
        intent = activation.ActivationIntent(
            intent_id=selection.intent_id, expected_generation=None, target=target,
            reranker_id=selection.reranker_id, job_id=selection.job_id, tracker=selection.tracker,
            source_epoch=selection.source_epoch, source_schema_signature=selection.source_schema_signature,
            state="db_selected",
        )
        handles, encodes, projections = [], [], []
        engine = engine_module.TrueMemoryEngine(":memory:")
        self.addCleanup(engine.close)

        def create(path):
            self.assertEqual(str(path), ":memory:")
            conn = storage.create_db(":memory:")
            handles.append(conn)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            vector_search.init_prepared_target_tables(conn, target)
            for table in ("vec_messages_edge", "vec_messages_sep_edge"):
                conn.execute(f"CREATE VIRTUAL TABLE {table} USING vec0(embedding float[256] distance_metric=cosine)")
            conn.execute("INSERT INTO vector_cache_registry(tier_group,vec_table,sep_table,model_name,embedding_dim) VALUES (?,?,?,?,?)",
                         (target.tier_group, *target.tables, target.model_id, target.dimension))
            # This controlled fixture seeds an already selected journal; certification
            # faults are covered by the separate activation-journal suite.
            activation._put(conn, activation._SELECTED, activation._dump(selection))
            activation._put(conn, activation._INTENT, activation._dump(intent))
            activation._put(conn, "embed_model", target.model_id)
            activation._put(conn, "embed_dim", str(target.dimension))
            conn.commit()
            return conn

        class Encoder:
            device = "cpu"
            def encode(self, texts, **kwargs):
                operation = runtime.current_operation(engine.conn)
                if operation is None or operation.selection != selection or operation.tables != target.tables:
                    raise AssertionError("Encode escaped selected serving operation")
                encodes.append(operation)
                result = np.zeros((len(texts), target.dimension), dtype=np.float32)
                result[:, 0] = 1.0
                return result

        def embedding(frozen, **kwargs):
            self.assertEqual(frozen, target)
            self.assertFalse(engine.conn.in_transaction)
            projections.append(frozen)
            return Encoder()

        class CrossEncoder:
            device = "cpu"
            def __init__(self, name, **kwargs):
                if name != selection.reranker_id:
                    raise AssertionError("Unexpected synthetic reranker identity")
            def predict(self, pairs, **kwargs):
                return np.ones(len(pairs), dtype=np.float32)

        with ExitStack() as stack:
            stack.enter_context(patch.multiple(vector_search, _model=None, _frozen_embedding_target=None,
                                               EMBEDDING_MODEL="model2vec", _embedding_dim=256, _model_generation=0))
            stack.enter_context(patch.multiple(reranker, _model=None, _model_name="synthetic/old", _active_tier="edge",
                                               _frozen_reranker_id=None, _model_certified=False))
            stack.enter_context(patch.multiple(runtime, _local=threading.local(), _acknowledged=None))
            stack.enter_context(patch.object(engine_module, "create_db", side_effect=create))
            stack.enter_context(patch.object(vector_search, "_load_frozen_embedding_target", side_effect=embedding))
            stack.enter_context(patch.object(model_client, "use_model_server", return_value=False))
            absent = object()
            previous_transformers = sys.modules.get("sentence_transformers", absent)

            def restore_transformers() -> None:
                if previous_transformers is absent:
                    sys.modules.pop("sentence_transformers", None)
                else:
                    sys.modules["sentence_transformers"] = previous_transformers

            # A whole-registry mock would evict torch first imported by device
            # detection here, leaving live native objects unsafe to re-import.
            stack.callback(restore_transformers)
            sys.modules["sentence_transformers"] = types.SimpleNamespace(CrossEncoder=CrossEncoder)
            stack.enter_context(patch.object(engine, "_maybe_auto_consolidate", return_value=None))
            stack.enter_context(patch.object(engine, "_maybe_startup_consolidate", return_value=None))
            for flag in ("_has_personality", "_has_style_vec", "_has_temporal", "_has_salience", "_has_consolidation",
                         "_has_predictive", "_has_hyde", "_has_clustering", "_has_reranker"):
                setattr(engine, flag, False)
            first = engine.add("synthetic cobalt record")
            self.assertTrue(engine.ready)
            self.assertEqual(projections, [target])
            for table in target.tables:
                self.assertEqual(engine.conn.execute(f"SELECT rowid FROM {table}").fetchall(), [(first["id"],)])
            self.assertEqual(engine.conn.execute("SELECT count(*) FROM vec_messages_edge").fetchone()[0], 0)
            self.assertEqual(engine.update(first["id"], "synthetic cobalt corrected")["content"], "synthetic cobalt corrected")
            self.assertEqual(engine.search_vectors_raw("synthetic cobalt")[0]["id"], first["id"])
            before = len(encodes)
            result = engine.search_agentic("synthetic cobalt", limit=1, max_rounds=1,
                                           use_hyde=False, use_reranker=False, use_clustering=False)
            self.assertEqual(result[0]["id"], first["id"])
            self.assertGreater(len(encodes), before)
            self.assertEqual(len({id(op) for op in encodes[before:]}), 1)
            self.assertTrue(engine.delete(first["id"]))
            for table in target.tables:
                self.assertEqual(engine.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)
            engine.add("synthetic second record")
            self.assertTrue(engine.delete_all())
            self.assertEqual(engine.conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
            original = engine.conn
            original.close()
            engine.add("synthetic reopened record")
            self.assertIsNot(engine.conn, original)
            self.assertTrue(engine.ready)
            self.assertIs(engine._runtime_vector_connection, engine.conn)
            self.assertEqual(projections, [target])
            self.assertEqual(len(handles), 2)
            self.assertIsNone(runtime.current_operation(engine.conn))
            native_modules = {name: id(module) for name, module in sys.modules.items()
                              if name == "torch" or name.startswith("torch.")}
        self.assertEqual({name: id(sys.modules.get(name)) for name in native_modules}, native_modules)
        self.assertIs(sys.modules.get("sentence_transformers", absent), previous_transformers)


if __name__ == "__main__":
    unittest.main()
