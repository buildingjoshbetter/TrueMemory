"""Engine routing with real SQLite, event-controlled owners and synthetic models."""

import ast
import builtins
import datetime
import gc
import logging
import os
import select
import signal
import runpy
import sqlite3
import sys
import threading
import time
import types
import unittest
import warnings
import weakref
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = runpy.run_path(str(Path(__file__).with_name("test-maintenance-adapters-753.py")))
STORAGE, MAINTENANCE = ADAPTERS["STORAGE"], ADAPTERS["MAINTENANCE"]


def stdlib_module(name: str, modules: dict) -> types.ModuleType:
    module = types.ModuleType("synthetic_routing_" + name)

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in modules:
            return modules[name]
        if name.startswith("truememory"):
            raise ImportError("Synthetic boundary excludes native modules")
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
    path = ROOT / "truememory" / (name + ".py")
    missing = object()
    previous = sys.modules.get(module.__name__, missing)
    sys.modules[module.__name__] = module
    try:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    finally:
        if previous is missing:
            sys.modules.pop(module.__name__, None)
        else:
            sys.modules[module.__name__] = previous
    return module


def load_engine(modules: dict) -> types.ModuleType:
    modules["truememory.tier_switch.writer"] = stdlib_module("tier_switch/writer", modules)
    # Runtime guards execute unchanged; only native identities are synthetic.
    modules["psutil"] = types.SimpleNamespace()
    for name in ("embedding_target", "tier_config", "tier_switch/cache",
                 "tier_switch/job", "tier_switch/source", "tier_switch/activation",
                 "tier_switch/serving"):
        modules["truememory." + name.replace("/", ".")] = stdlib_module(name, modules)
    serving = modules["truememory.tier_switch.serving"]
    modules["truememory.tier_switch"] = types.SimpleNamespace(serving=serving)
    vector = modules["truememory.vector_search"]
    vector._frozen_embedding_target = None
    vector._runtime_policy_tier = None
    vector.resolve_tier = lambda: "edge"
    vector_source = ast.parse((ROOT / "truememory/vector_search.py").read_text(encoding="utf-8"))
    sep = next(node for node in vector_source.body
               if isinstance(node, ast.FunctionDef) and node.name == "_active_sep_table")
    exec(compile(ast.Module(body=[sep], type_ignores=[]), "actual-separation-resolver", "exec"),
         vector.__dict__)
    modules["truememory"] = types.SimpleNamespace(
        vector_search=vector,
        reranker=types.SimpleNamespace(get_current_reranker_name=lambda: "synthetic/reranker"),
    )
    runtime = stdlib_module("tier_switch/runtime", modules)
    runtime.sys = types.SimpleNamespace(modules=modules)
    modules["truememory.tier_switch.runtime"] = runtime
    vector.__dict__.update(__builtins__=runtime.__dict__["__builtins__"],
                           datetime=datetime, _model_generation=0)
    metadata_nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    metadata_nodes.extend(node for node in vector_source.body
                          if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                          and node.name in {"_write_foreground_embedder_metadata_no_commit", "VectorPublicationChanged"})
    exec(compile(ast.fix_missing_locations(ast.Module(body=metadata_nodes, type_ignores=[])),
                 "actual-foreground-metadata", "exec"), vector.__dict__)
    module = types.ModuleType("synthetic_routing_engine")

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in modules:
            return modules[name]
        if name.startswith("truememory"):
            raise ImportError("Synthetic boundary excludes native modules")
        return builtins.__import__(name, globals, locals, fromlist, level)

    module.__dict__.update({"__builtins__": dict(vars(builtins), __import__=safe_import),
        "__file__": str(ROOT / "truememory/engine.py"), "Path": Path, "sqlite3": sqlite3,
        "threading": threading, "time": time, "os": os, "warnings": warnings,
        "engine_operation": lambda function: function,
        "engine_handle_operation": runtime.engine_handle_operation,
        "logger": logging.getLogger("synthetic-routing"), "MAX_CONTENT_LENGTH": 50_000,
        "_env_int": lambda name, default, **kwargs: default})
    source = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in source.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nodes.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "truememory.storage":
            module.__dict__.update({item.asname or item.name: getattr(STORAGE, item.name) for item in node.names})
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id.startswith("_HAS_"):
                        module.__dict__[target.id] = False
                    elif target.id in {"_ALLOWED_TABLES", "_ALLOWED_COLUMNS", "_ALL_VEC_TABLES", "_SQLITE_IN_CHUNK"}:
                        nodes.append(node)
    module.__dict__["_HAS_CONSOLIDATION"] = True
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(ROOT / "truememory/engine.py"), "exec"), module.__dict__)
    return module


class RoutingFixture(ADAPTERS["AdapterFixture"]):
    def setUp(self) -> None:
        style = stdlib_module("personality_style_vec", {})
        self.enterContext(patch.dict(sys.modules, {"truememory.personality_style_vec": style}))
        super().setUp()
        self.modules.update({"truememory.storage": STORAGE, "truememory.maintenance": MAINTENANCE,
                             "truememory.vector_search": self.vector,
                             "sqlite_vec": types.SimpleNamespace(load=lambda conn: None)})
        self.modules["truememory.rebuild_source"] = stdlib_module("rebuild_source", self.modules)
        self.engine_module = load_engine(self.modules)
        original_import = MAINTENANCE.__dict__["__builtins__"]["__import__"]

        def maintenance_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in self.modules:
                return self.modules[name]
            if name.startswith("truememory"):
                raise ImportError("Synthetic boundary excludes native modules")
            return original_import(name, globals, locals, fromlist, level)

        self.enterContext(patch.dict(MAINTENANCE.__dict__["__builtins__"],
                                     {"__import__": maintenance_import}))
        create = STORAGE.create_db

        class ExtensionBoundary:
            def __init__(boundary, conn):
                boundary.conn = conn

            def __getattr__(boundary, name):
                return getattr(boundary.conn, name)

            def enable_load_extension(boundary, enabled):
                # The fixture uses ordinary BLOB tables and a fake sqlite-vec
                # loader, including hosts compiled without extension support.
                self.assertIs(type(enabled), bool)

        self.enterContext(patch.object(MAINTENANCE, "create_db", lambda path: ExtensionBoundary(create(path))))
        self.enterContext(patch.object(self.engine_module, "create_db", lambda path: ExtensionBoundary(create(path))))
        self.engine = self.new_engine(self.conn)
        self.coordinator = self.engine._get_maintenance_coordinator()
        self.addCleanup(self.coordinator.wait, 3)
        self.addCleanup(self.coordinator.cancel)

    def new_engine(self, conn: sqlite3.Connection | None = None):
        engine = self.engine_module.TrueMemoryEngine(self.path)
        engine.conn = conn or self.open_db()
        engine.ready = True
        engine._has_consolidation = False
        self.addCleanup(engine.close)
        return engine

    def completed(self) -> None:
        self.assertTrue(self.coordinator.wait(3), "Synthetic maintenance did not terminate")
        self.assertFalse(self.coordinator.snapshot()["active"])

    def fake_report(self):
        return MAINTENANCE.MaintenanceReport((), "SKIPPED (synthetic)")


class TestCoordinatorHandoff(RoutingFixture):
    def test_local_owner_retains_one_pending_wake_after_last_engine_close(self) -> None:
        path = self.path.with_name("synthetic-detached.sqlite")
        engine = self.engine_module.TrueMemoryEngine(path)
        engine.conn, engine.ready = STORAGE.create_db(path), True
        coordinator = engine._get_maintenance_coordinator()
        reference, engine_reference = weakref.ref(coordinator), weakref.ref(engine)
        entered, release = threading.Event(), threading.Event()
        observed = []

        def work(conn, owner, **kwargs):
            observed.append(kwargs["threshold"])
            entered.set()
            self.assertTrue(release.wait(3))
            return self.fake_report()

        with patch.object(MAINTENANCE, "run_engine_maintenance", work):
            try:
                with MAINTENANCE.maintenance_owner(path):
                    coordinator.request_layers(threshold=31)
                    coordinator.request_layers(threshold=37)
                    self.assertEqual(len(MAINTENANCE._owner_waiters), 1)
                    engine.close()
                    del engine, coordinator
                    gc.collect()
                    self.assertIsNone(engine_reference())
                    self.assertIsNotNone(reference())
                self.assertTrue(entered.wait(2))
                self.assertEqual(MAINTENANCE._owner_waiters, {})
                coordinator = reference()
                self.assertIsNotNone(coordinator)
            finally:
                release.set()
                if reference() is not None:
                    self.assertTrue(reference().wait(3))
        self.assertEqual(observed, [37])

    def test_cancel_releases_local_waiter_and_admission_cannot_restore_it(self) -> None:
        original = MAINTENANCE._acquire_owner

        def cancelled_admission(path, on_busy=None):
            self.coordinator.cancel()
            return original(path, on_busy=on_busy)

        with patch.object(MAINTENANCE, "run_engine_maintenance") as work:
            with MAINTENANCE.maintenance_owner(self.path):
                self.coordinator.request_layers()
                self.assertEqual(len(MAINTENANCE._owner_waiters), 1)
                self.coordinator.cancel()
                self.assertEqual(MAINTENANCE._owner_waiters, {})
                with patch.object(MAINTENANCE, "_acquire_owner", cancelled_admission):
                    self.assertFalse(self.coordinator.request_layers())
                self.assertEqual(self.coordinator.snapshot()["pending_generations"], 0)
                self.assertEqual(MAINTENANCE._owner_waiters, {})
            work.assert_not_called()
        self.assertEqual(self.coordinator.status, ("cancelled", None))

    def test_delayed_start_exception_cannot_cancel_successor_or_its_pending_wake(self) -> None:
        start = threading.Thread.start
        entered, release = threading.Event(), threading.Event()
        attempts, observed, cancellation = [], [], []

        def work(conn, coordinator, **kwargs):
            observed.append(kwargs["threshold"])
            if kwargs["threshold"] == 31:
                entered.set()
                self.assertTrue(release.wait(3))
                cancellation.append(kwargs["cancel"].is_set())
            return self.fake_report()

        def late_failure(worker):
            attempts.append(worker)
            start(worker)
            if len(attempts) == 1:
                self.completed()
                self.assertTrue(self.coordinator.request_layers(threshold=31))
                self.assertTrue(entered.wait(2))
                self.assertFalse(self.coordinator.request_layers(threshold=37))
                raise RuntimeError("synthetic delayed start failure")

        with patch.object(MAINTENANCE.threading.Thread, "start", late_failure), \
             patch.object(MAINTENANCE, "run_engine_maintenance", work):
            try:
                with self.assertRaisesRegex(RuntimeError, "synthetic delayed start failure"):
                    self.coordinator.request_layers()
                self.assertFalse(self.coordinator._cancel.is_set())
                self.assertEqual(self.coordinator.snapshot()["pending_generations"], 1)
            finally:
                release.set()
                self.completed()
        self.assertEqual(observed, [25, 31, 37])
        self.assertEqual(cancellation, [False])

    @unittest.skipUnless(hasattr(os, "fork"), "POSIX fork required")
    def test_fork_discards_inherited_pending_worker_and_rejects_stale_coordinator(self) -> None:
        entered, release = threading.Event(), threading.Event()

        def active(conn, cancel):
            entered.set()
            self.assertTrue(release.wait(3))

        self.coordinator.request(active)
        self.assertTrue(entered.wait(2))
        self.coordinator.request_layers()
        waiting_path = self.path.with_name("synthetic-fork-waiter.sqlite")
        self.enterContext(MAINTENANCE.maintenance_owner(waiting_path))
        waiting = MAINTENANCE.get_coordinator(waiting_path)
        self.addCleanup(waiting.cancel)
        waiting.request_layers()
        self.assertEqual(len(MAINTENANCE._owner_waiters), 1)
        read_fd, write_fd = os.pipe()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            child = os.fork()
        if child == 0:
            os.close(read_fd)
            try:
                rejected = False
                try:
                    self.coordinator.request_layers()
                except RuntimeError:
                    rejected = True
                fresh = MAINTENANCE.get_coordinator(self.path)
                clean = (not fresh.snapshot()["active"] and fresh.snapshot()["pending_generations"] == 0
                         and not MAINTENANCE._owner_waiters)
                os.write(write_fd, b"PASS" if rejected and clean else b"FAIL")
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        try:
            self.assertTrue(select.select([read_fd], [], [], 2)[0], "Synthetic child did not answer")
            self.assertEqual(os.read(read_fd, 4), b"PASS")
        finally:
            os.close(read_fd)
            if os.waitpid(child, os.WNOHANG)[0] == 0:
                os.kill(child, signal.SIGKILL)
                os.waitpid(child, 0)
            self.coordinator.cancel()
            release.set()
            self.completed()

    def test_start_failure_services_only_a_real_later_notification(self) -> None:
        start = threading.Thread.start
        attempts, observed = [], []

        def fail_once(worker):
            attempts.append(True)
            if len(attempts) == 1:
                self.assertFalse(self.coordinator.request_layers(threshold=31))
                raise RuntimeError("synthetic start failure")
            return start(worker)

        with patch.object(MAINTENANCE.threading.Thread, "start", fail_once), \
             patch.object(MAINTENANCE, "run_engine_maintenance", lambda *args, **kwargs: observed.append(kwargs["threshold"]) or self.fake_report()):
            with self.assertRaisesRegex(RuntimeError, "synthetic start failure"):
                self.coordinator.request_layers()
            self.completed()
        self.assertEqual(observed, [31])
        self.assertEqual(len(attempts), 2)

    def test_start_failure_without_notification_has_no_retry(self) -> None:
        with patch.object(MAINTENANCE.threading.Thread, "start", side_effect=RuntimeError("synthetic")) as start:
            with self.assertRaises(RuntimeError):
                self.coordinator.request_layers()
            self.completed()
        start.assert_called_once()
        self.assertEqual(self.coordinator.snapshot()["pending_generations"], 0)
        self.assertEqual(self.coordinator.status, ("failed", "worker_start"))

    def test_close_detaches_without_cancelling_or_retaining_engine(self) -> None:
        entered, release = threading.Event(), threading.Event()

        def active(conn, cancel):
            entered.set()
            self.assertTrue(release.wait(2))
            self.assertFalse(cancel.is_set())
            self.assertEqual(conn.execute("SELECT 1").fetchone()[0], 1)

        with patch.object(MAINTENANCE, "run_engine_maintenance", lambda *args, **kwargs: self.fake_report()):
            self.coordinator.request(active)
            self.assertTrue(entered.wait(2))
            engine = self.engine_module.TrueMemoryEngine(self.path)
            engine.conn, engine.ready = self.open_db(), True
            reference = weakref.ref(engine)
            engine._maybe_auto_consolidate()
            self.assertEqual(self.coordinator.snapshot()["pending_generations"], 1)
            engine.close()
            del engine
            gc.collect()
            self.assertIsNone(reference())
            self.assertFalse(self.coordinator.wait(0))
            release.set()
            self.completed()
        self.assertEqual(self.coordinator.status, ("success", None))

    def test_busy_admission_and_concurrent_owner_release_cannot_lose_wake(self) -> None:
        held, release, leaving, ran = (threading.Event() for _ in range(4))
        original_acquire, original_release = MAINTENANCE._acquire_owner, MAINTENANCE._release_owner

        def owner():
            with MAINTENANCE.maintenance_owner(self.path):
                held.set()
                self.assertTrue(release.wait(2))

        def release_owner(fd):
            leaving.set()
            original_release(fd)

        def acquire(path, on_busy=None):
            def busy():
                on_busy()
                release.set()
                self.assertTrue(leaving.wait(2))
            return original_acquire(path, on_busy=busy if on_busy is not None else None)

        owner_thread = threading.Thread(target=owner)
        with patch.object(MAINTENANCE, "_release_owner", release_owner), \
             patch.object(MAINTENANCE, "run_engine_maintenance", lambda *args, **kwargs: ran.set() or self.fake_report()):
            owner_thread.start()
            try:
                self.assertTrue(held.wait(2))
                with patch.object(MAINTENANCE, "_acquire_owner", acquire):
                    self.assertFalse(self.coordinator.request_layers())
                    self.assertTrue(ran.wait(2))
                    self.completed()
            finally:
                release.set()
                owner_thread.join(3)
        self.assertFalse(owner_thread.is_alive())

    def test_many_active_notifications_coalesce_to_one_static_successor(self) -> None:
        entered, release = threading.Event(), threading.Event()
        observed = []

        def active(conn, cancel):
            entered.set()
            self.assertTrue(release.wait(2))

        def successor(conn, coordinator, **kwargs):
            observed.append((conn is self.conn, kwargs["threshold"]))
            return self.fake_report()

        with patch.object(MAINTENANCE, "run_engine_maintenance", successor):
            self.assertTrue(self.coordinator.request(active))
            self.assertTrue(entered.wait(2))
            for threshold in range(1, 101):
                self.assertFalse(self.coordinator.request_layers(threshold=threshold))
            self.assertEqual(self.coordinator.snapshot()["pending_generations"], 1)
            self.assertFalse(self.coordinator.wait(0))
            release.set()
            self.completed()
        self.assertEqual(observed, [(False, 100)])
        self.assertEqual(self.coordinator.snapshot()["pending_generations"], 0)

    def test_local_synchronous_owner_release_services_pending_wake(self) -> None:
        observed = []
        with patch.object(MAINTENANCE, "run_engine_maintenance", lambda *args, **kwargs: observed.append(True) or self.fake_report()):
            with MAINTENANCE.maintenance_owner(self.path):
                self.assertFalse(self.coordinator.request_layers())
                self.assertEqual(self.coordinator.status, ("busy", None))
                self.assertEqual(self.coordinator.snapshot()["pending_generations"], 1)
                self.assertEqual(observed, [])
            self.completed()
        self.assertEqual(observed, [True])

    def test_external_busy_requires_new_boundary_and_does_not_spin(self) -> None:
        observed = []
        with patch.object(MAINTENANCE, "try_file_lock", return_value=None), \
             patch.object(MAINTENANCE.threading, "Thread") as thread:
            self.assertFalse(self.coordinator.request_layers())
            thread.assert_not_called()
        self.assertEqual(self.coordinator.snapshot()["pending_generations"], 1)
        self.assertEqual(MAINTENANCE._owner_waiters, {})
        with patch.object(MAINTENANCE, "run_engine_maintenance", lambda *args, **kwargs: observed.append(True) or self.fake_report()):
            self.assertTrue(self.coordinator.request_layers())
            self.completed()
        self.assertEqual(observed, [True])

    def test_wake_between_descriptor_release_and_idle_transition_is_not_lost(self) -> None:
        released, finish = threading.Event(), threading.Event()
        original_release = MAINTENANCE._release_owner
        observed = []
        first = []

        def pause(fd):
            original_release(fd)
            if not first:
                first.append(True)
                released.set()
                self.assertTrue(finish.wait(2))

        with patch.object(MAINTENANCE, "_release_owner", pause), \
             patch.object(MAINTENANCE, "run_engine_maintenance", lambda *args, **kwargs: observed.append(True) or self.fake_report()):
            self.assertTrue(self.coordinator.request(lambda conn, cancel: None))
            self.assertTrue(released.wait(2))
            self.assertFalse(self.coordinator.request_layers())
            self.assertFalse(self.coordinator.wait(0))
            finish.set()
            self.completed()
        self.assertEqual(observed, [True])

    def test_model_deferral_has_no_self_generated_successor(self) -> None:
        observed = []
        report = MAINTENANCE.MaintenanceReport((MAINTENANCE.LayerResult(
            "clusters", "cluster_messages", "deferred", None, 0., "ModelBusy", False),), "SKIPPED")
        with patch.object(MAINTENANCE, "run_engine_maintenance", lambda *args, **kwargs: observed.append(True) or report):
            self.assertTrue(self.coordinator.request_layers())
            self.completed()
        self.assertEqual(observed, [True])
        self.assertEqual(self.coordinator.status, ("deferred", "ModelBusy"))
        self.assertEqual(self.coordinator.snapshot()["pending_generations"], 0)

    def test_cancellation_drops_pending_but_keeps_active_ownership(self) -> None:
        entered, release = threading.Event(), threading.Event()

        def active(conn, cancel):
            entered.set()
            self.assertTrue(release.wait(2))

        with patch.object(MAINTENANCE, "run_engine_maintenance") as successor:
            self.coordinator.request(active)
            self.assertTrue(entered.wait(2))
            self.coordinator.request_layers()
            self.coordinator.cancel()
            self.assertFalse(self.coordinator.wait(0))
            self.assertFalse(self.coordinator.request_layers())
            self.assertEqual(self.coordinator.snapshot()["pending_generations"], 0)
            release.set()
            self.completed()
            successor.assert_not_called()
        self.assertEqual(self.coordinator.status, ("cancelled", None))


class TestEngineRouting(RoutingFixture):
    def test_status_reports_categorical_layer_failure_and_caller_transaction(self) -> None:
        self.add(1)
        with patch.object(self.cluster, "cluster_messages", side_effect=RuntimeError("synthetic private sentinel")):
            self.engine.consolidate()
        state = self.engine.get_stats()["maintenance"]
        self.assertEqual(len(state["layers"]), 8)
        self.assertEqual(state["layers"]["clusters"]["outcome"], "failed")
        self.assertEqual(state["layers"]["clusters"]["error_category"], "RuntimeError")
        self.assertEqual(state["layers"]["clusters"]["freshness"], "untrusted")
        self.assertNotIn("synthetic private sentinel", str(state))
        self.assertNotIn(str(self.path), str(state))
        self.conn.execute("UPDATE messages SET content='synthetic uncommitted status'")
        self.assertTrue(self.engine.get_stats()["maintenance"]["pending_caller_commit"])
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()

    def test_manual_repair_restores_actual_cluster_search_and_automatic_eligibility(self) -> None:
        self.add(5)
        self.native.fit = lambda: setattr(self.native, "labels", [0, 0, 1, 1, 2])
        self.engine._has_clustering = False
        self.engine._has_vectors = True
        self.modules["truememory.agentic_search"] = types.SimpleNamespace(normalize_scores=lambda rows: rows)
        with patch.object(self.coordinator, "request_layers") as request, \
             patch.object(self.engine, "search", return_value=[{"id": 1, "score": .5}]), \
             patch.object(self.engine, "_entity_focused_search", return_value=[]), \
             patch.object(self.engine, "_check_sufficiency", return_value=True), \
             patch.object(self.engine, "_apply_surprise_boost", side_effect=lambda rows: rows), \
             patch.object(self.engine, "_clean_results", side_effect=lambda rows, *_args, **_kwargs: rows), \
             patch.object(self.engine_module, "search_clustered", return_value=[{"id": 2, "score": .5}], create=True) as search:
            self.engine._maybe_auto_consolidate()
            request.assert_not_called()
            self.engine.search_agentic("synthetic query", use_hyde=False, use_reranker=False)
            search.assert_not_called()
            result = self.engine.consolidate()
            self.assertIn("3 clusters", result["cluster_messages"])
            self.assertTrue(self.engine._has_clustering)
            self.assertTrue(self.engine._has_consolidation)
            rows = self.engine.search_agentic("synthetic query", use_hyde=False, use_reranker=False)
            search.assert_called_once()
            self.assertEqual([row["id"] for row in rows], [1, 2])
            request.assert_not_called()
            self.conn.execute("UPDATE messages SET content='synthetic repaired correction' WHERE id=1")
            self.conn.commit()
            self.engine._maybe_auto_consolidate()
            request.assert_called_once_with(threshold=25)

    def test_failed_manual_repair_does_not_enable_capabilities_or_remove_prior_success(self) -> None:
        self.engine._has_clustering = False
        with patch.object(self.cluster, "cluster_messages", side_effect=RuntimeError("synthetic")), \
             patch.object(self.modules["truememory.consolidation"], "build_summaries", side_effect=RuntimeError("synthetic")):
            result = self.engine.consolidate()
            self.assertIn("ERROR", result["cluster_messages"])
            self.assertIn("ERROR", result["build_summaries"])
            self.assertFalse(self.engine._has_clustering)
            self.assertFalse(self.engine._has_consolidation)
            self.engine._has_clustering = self.engine._has_consolidation = True
            self.engine.consolidate()
            self.assertTrue(self.engine._has_clustering)
            self.assertTrue(self.engine._has_consolidation)

    def test_unavailable_manual_cluster_repair_preserves_only_prior_capability(self) -> None:
        self.engine._has_clustering = False
        with patch.object(self.modules["sqlite_vec"], "load", side_effect=RuntimeError("synthetic")):
            self.assertIn("UNAVAILABLE", self.engine.consolidate()["cluster_messages"])
            self.assertFalse(self.engine._has_clustering)
            self.engine._has_clustering = True
            self.assertIn("UNAVAILABLE", self.engine.consolidate()["cluster_messages"])
            self.assertTrue(self.engine._has_clustering)

    def test_first_open_manual_does_not_start_its_own_competitor(self) -> None:
        self.add(1)
        engine = self.engine_module.TrueMemoryEngine(self.path)
        self.addCleanup(engine.close)
        with patch.object(self.coordinator, "request_layers") as request:
            result = engine.consolidate()
            request.assert_not_called()
        self.assertTrue(engine.ready)
        self.assertEqual(set(result), set(MAINTENANCE._MAINTENANCE_RESULT_KEYS))

    def test_current_empty_startup_and_rolled_back_correction_do_not_schedule(self) -> None:
        self.engine.consolidate()
        self.engine._has_consolidation = True
        with patch.object(self.coordinator, "request_layers") as request:
            self.engine._maybe_startup_consolidate()
            self.engine._ensure_connection()
            request.assert_not_called()
        self.add(1)
        self.engine.consolidate()
        with patch.object(self.coordinator, "request_layers") as request:
            self.conn.execute("UPDATE messages SET content='synthetic uncommitted'")
            self.engine._ensure_connection()
            self.assertEqual(self.engine._maintenance_pending_reason, "pending_caller_commit")
            self.conn.rollback()
            self.engine._ensure_connection()
            request.assert_not_called()
            self.conn.execute("UPDATE messages SET content='synthetic committed'")
            self.conn.commit()
            self.engine._ensure_connection()
            request.assert_called_once_with(threshold=25)

    def test_add_probe_uses_no_corpus_scan_native_import_or_blocking_model_lock(self) -> None:
        self.engine.consolidate()
        self.engine._has_consolidation = True
        operations = []
        underlying = self.vector._lock

        class ProbeLock:
            def acquire(lock, blocking=True):
                self.assertFalse(blocking)
                return underlying.acquire(blocking=False)

            def release(lock):
                underlying.release()

        def authorize(action, table, column, *_args):
            if action == sqlite3.SQLITE_READ:
                operations.append((table, column))
            return sqlite3.SQLITE_OK

        self.conn.set_authorizer(authorize)
        try:
            with patch.object(self.vector, "_lock", ProbeLock()), \
                 patch.object(MAINTENANCE, "_installed_dependency", side_effect=AssertionError("per-add package scan")), \
                 patch.object(self.coordinator, "request_layers", side_effect=AssertionError("unexpected work")):
                self.engine._maybe_auto_consolidate()
                self.assertEqual(self.engine._maintenance_pending_reason, "awaiting_eligibility")
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertFalse({table for table, _ in operations}.intersection({"messages", "vec_messages_edge", "summaries", "cluster_centroids"}))

    def test_source_commit_followed_by_a17_vector_error_still_notifies(self) -> None:
        self.add(1)
        self.engine.consolidate()
        self.engine._has_vectors = self.engine._has_consolidation = True
        changed = type("VectorPublicationChanged", (RuntimeError,), {})

        @contextmanager
        def publication(*args, **kwargs):
            raise changed("synthetic model replacement")
            yield

        def capture(conn: sqlite3.Connection) -> tuple[object, str]:
            self.assertIs(conn, self.engine.conn)
            return object(), "synthetic-identity"

        fields = {"_capture_rebuild_model": capture,
                  "_encode_with_mps_fallback": lambda *args: [[1., 2.]],
                  "_build_sep_text": lambda *args: "synthetic separated text",
                  "_foreground_vector_publication": publication,
                  "serialize_f32": lambda values: b"synthetic-unused-vector",
                  "_write_embedder_metadata_no_commit": lambda conn: None,
                  "VectorPublicationChanged": changed}
        with patch.dict(self.vector.__dict__, fields), patch.object(self.coordinator, "request_layers") as request:
            with self.assertRaisesRegex(changed, "Source update committed"):
                self.engine.update(1, content="synthetic corrected source")
            request.assert_called_once_with(threshold=25)
        self.assertFalse(self.conn.in_transaction)
        self.assertFalse(self.engine._write_lock.locked())
        self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=1").fetchone()[0], "synthetic corrected source")

    def test_manual_busy_is_truthful_and_does_not_queue_forced_work(self) -> None:
        entered, release = threading.Event(), threading.Event()

        def active(conn, cancel):
            entered.set()
            self.assertTrue(release.wait(2))

        self.coordinator.request(active)
        self.assertTrue(entered.wait(2))
        try:
            result = self.engine.consolidate()
            self.assertEqual(set(result), set(MAINTENANCE._MAINTENANCE_RESULT_KEYS))
            self.assertTrue(all(value.startswith("BUSY") for value in result.values()))
            self.assertEqual(self.coordinator.snapshot()["pending_generations"], 0)
        finally:
            release.set()
            self.completed()

    def test_file_manual_uses_dedicated_connection_all_eight_and_preferences(self) -> None:
        self.add(5)
        seen = []
        original = MAINTENANCE.run_engine_maintenance

        def run(conn, coordinator, **kwargs):
            seen.append(conn)
            self.assertIsNot(conn, self.conn)
            self.assertTrue(self.engine._write_lock.acquire(blocking=False))
            self.engine._write_lock.release()
            return original(conn, coordinator, **kwargs)

        with patch.object(MAINTENANCE, "run_engine_maintenance", run), \
             patch.object(self.coordinator, "request_layers") as automatic:
            result = self.engine.consolidate()
            automatic.assert_not_called()
        self.assertEqual(set(result), set(MAINTENANCE._MAINTENANCE_RESULT_KEYS))
        self.assertTrue(all("ERROR" not in value and "UNAVAILABLE" not in value
                            for key, value in result.items() if key != "extract_preferences"), result)
        self.assertEqual(result["extract_preferences"], "UNAVAILABLE (ScheduledPreferencesUnsupported)")
        self.assertIn("vector_generation_unverified", result["cluster_messages"])
        with self.assertRaises(sqlite3.ProgrammingError):
            seen[0].execute("SELECT 1")

    def test_24_then_25_committed_adds_across_engines_use_durable_counts(self) -> None:
        self.engine.consolidate()
        second = self.new_engine()
        self.engine._has_consolidation = second._has_consolidation = True
        with patch.object(self.coordinator, "request_layers", wraps=self.coordinator.request_layers) as request:
            for index in range(24):
                engine = self.engine if index % 2 else second
                engine.add("synthetic append", sender="synthetic-sender")
            request.assert_not_called()
            second.add("synthetic twenty fifth", sender="synthetic-sender")
            self.completed()
            self.assertGreaterEqual(request.call_count, 1)
        states = MAINTENANCE.read_layer_states(self.conn, self.specs())
        self.assertTrue(all(state.attempted_insert_count == 25 for state in states.values()), states)

    def test_borrowed_manual_preserves_caller_writes_and_rollback(self) -> None:
        self.add(1, commit=False)
        with patch.object(self.engine_module, "create_db", side_effect=AssertionError("new handle")), \
             patch.object(self.modules["truememory.personality"], "extract_preferences",
                          side_effect=AssertionError("unsupported preference work")) as preferences:
            result = self.engine.consolidate()
        preferences.assert_not_called()
        self.assertEqual(set(result), set(MAINTENANCE._MAINTENANCE_RESULT_KEYS))
        self.assertTrue(all("pending caller commit" in result[key]
                            for key in MAINTENANCE._MAINTENANCE_RESULT_KEYS if key != "extract_preferences"), result)
        self.assertEqual(result["extract_preferences"], "UNAVAILABLE (ScheduledPreferencesUnsupported)")
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
        self.assertEqual(self.output(), ([], []))

    def test_in_memory_auto_stays_pending_and_manual_uses_original_rows(self) -> None:
        conn = STORAGE.create_db(":memory:")
        self.addCleanup(conn.close)
        engine = self.engine_module.TrueMemoryEngine(":memory:")
        engine.conn, engine.ready = conn, True
        self.addCleanup(engine.close)
        conn.execute("INSERT INTO messages(content,timestamp) VALUES ('synthetic in memory','2026-01-01')")
        conn.commit()
        with patch.object(MAINTENANCE, "create_db", side_effect=AssertionError("new handle")), \
             patch.object(MAINTENANCE.threading, "Thread", side_effect=AssertionError("memory worker")):
            engine._maybe_auto_consolidate()
            self.assertEqual(engine._maintenance_coordinator.status, ("pending_in_memory", None))
            result = engine.consolidate()
        self.assertIn("1 episodes", result["detect_episodes"])
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)


class TestCapabilityEvidence(RoutingFixture):
    def test_transient_model_busy_cannot_clear_prior_extension_failure_baseline(self) -> None:
        self.add(5)
        with patch.object(self.modules["sqlite_vec"], "load", side_effect=RuntimeError("synthetic")):
            self.engine.consolidate()
            prior = self.coordinator.capability_snapshot()[2]
            self.add(1)
            worker = MAINTENANCE.create_db(self.path)
            try:
                with self.vector._lock:
                    MAINTENANCE._prepare_worker_extensions(worker, self.coordinator)
            finally:
                worker.close()
        self.assertEqual(self.coordinator.capability_snapshot()[2], prior)
        self.assertEqual(MAINTENANCE.plan_layers(self.conn, MAINTENANCE.engine_layer_specs(self.conn, self.coordinator)), ())

    def test_reopen_during_worker_failure_preserves_current_run_local_evidence(self) -> None:
        self.add(1)

        def fail_and_refresh(conn):
            self.coordinator.refresh_capabilities()
            raise RuntimeError("synthetic load failure after reopen")

        with patch.object(self.modules["sqlite_vec"], "load", fail_and_refresh):
            result = self.engine.consolidate()
        self.assertIn("VectorExtensionUnavailable", result["cluster_messages"])
        self.assertIsNone(self.coordinator.capability_snapshot()[2])

    def test_worker_extension_failure_keeps_same_key_and_25_insert_baseline(self) -> None:
        self.add(5)
        loader = self.modules["sqlite_vec"]
        with patch.object(loader, "load", side_effect=RuntimeError("synthetic private sentinel")):
            result = self.engine.consolidate()
            self.assertIn("VectorExtensionUnavailable", result["cluster_messages"])
            self.engine._has_consolidation = True
            with patch.object(self.coordinator, "request_layers") as request:
                self.engine.add("synthetic sixth")
                self.engine._ensure_connection()
                request.assert_not_called()
            state = MAINTENANCE.read_layer_states(self.conn, MAINTENANCE.engine_layer_specs(self.conn, self.coordinator))["clusters"]
            self.assertEqual(state.attempted_insert_count, 5)
            self.assertEqual(state.dependency.key, state.attempted_dependency)
        repaired = self.engine.consolidate()
        self.assertNotIn("UNAVAILABLE", repaired["cluster_messages"])
        self.assertIsNone(self.coordinator.capability_snapshot()[2])

    def test_reopen_refreshes_static_evidence_and_new_model_invalidates_failure_key(self) -> None:
        self.add(1)
        with patch.object(self.modules["sqlite_vec"], "load", side_effect=RuntimeError("synthetic")):
            self.engine.consolidate()
        epoch, versions, failure = self.coordinator.capability_snapshot()
        self.assertEqual(len(versions), 3)
        self.assertIsNotNone(failure)
        self.vector.EMBEDDING_MODEL = "synthetic-model-c"
        dependency = MAINTENANCE.engine_layer_specs(self.conn, self.coordinator)[0].resolve_dependency()
        self.assertTrue(dependency.available)
        engine = self.new_engine()
        self.assertIs(engine._get_maintenance_coordinator(), self.coordinator)
        self.assertGreater(self.coordinator.capability_snapshot()[0], epoch)
        self.assertIsNone(self.coordinator.capability_snapshot()[2])

    def test_old_worker_cannot_publish_capability_error_after_refresh(self) -> None:
        epoch, _, _ = self.coordinator.capability_snapshot()
        failure = MAINTENANCE.make_layer_dependency(1, available=False, error_category="VectorExtensionUnavailable")
        self.coordinator.refresh_capabilities()
        self.coordinator._publish_extension_evidence(epoch, failure)
        self.assertIsNone(self.coordinator.capability_snapshot()[2])

    def test_actual_connection_unavailability_is_not_overridden_by_cached_success(self) -> None:
        dependency = self.spec().resolve_dependency()
        unavailable = MAINTENANCE.make_layer_dependency(1, MAINTENANCE._dependency_parameters(dependency),
                                                       available=False, error_category="VectorExtensionUnavailable")
        with patch.object(MAINTENANCE, "_cluster_schedule_dependency", return_value=unavailable):
            observed = MAINTENANCE.engine_layer_specs(self.conn, self.coordinator)[0].resolve_dependency()
        self.assertEqual(observed, unavailable)


if __name__ == "__main__":
    unittest.main()
