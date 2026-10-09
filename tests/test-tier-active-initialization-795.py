"""Live automatic-worker initialization receipts, with stdlib and memory SQLite."""
from __future__ import annotations

import contextlib
import ast
import logging
import os
from pathlib import Path
import runpy
import re
import sqlite3
import threading
import types
import unittest
import uuid
import weakref
from typing import NamedTuple
from unittest.mock import Mock, patch

BOUNDARIES = runpy.run_path(str(Path(__file__).with_name("test-tier-public-boundaries-795.py")))
definitions = BOUNDARIES["definitions"]


class TestActiveInitialization(unittest.TestCase):
    def setUp(self):
        self.fixture = BOUNDARIES["TestEngineReconnectReceipt"]()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.first = self.fixture.engine()
        self.api = self.fixture.api
        self.closed_fds = []
        self.worker_conn = self.fixture.memory_connection()
        self.entered, self.release = threading.Event(), threading.Event()
        thread_api = types.SimpleNamespace(Thread=threading.Thread, Lock=threading.Lock,
                                          Event=threading.Event, local=threading.local)
        names = {"MaintenanceBusyError", "MaintenanceOwnership", "MaintenanceRequest", "MaintenanceCoordinator",
                 "MaintenanceReport", "LayerResult", "_StyleOpenRejected", "_style_error_category",
                 "canonical_database_path", "_bind_owner", "maintenance_owner", "_acquire_owner", "_release_owner",
                 "_after_fork", "get_coordinator", "_valid_initialization_receipt", "_active_initialization_receipt"}
        self.maintenance = definitions("maintenance.py", names, self.fixture.namespace(
            Path=Path, re=re, os=types.SimpleNamespace(getpid=os.getpid, close=self.closed_fds.append, PathLike=os.PathLike, fspath=os.fspath),
            uuid=uuid, weakref=weakref, threading=thread_api, sqlite3=sqlite3, NamedTuple=NamedTuple,
            contextmanager=contextlib.contextmanager, _registry_lock=threading.Lock(),
            _coordinators=weakref.WeakValueDictionary(), _held_fds=set(), _held_paths={}, _owner_waiters={},
            _thread_owners=threading.local(), try_file_lock=lambda path: 71,
            create_db=lambda path: self.worker_conn, logger=logging.getLogger("synthetic-active-receipt")))
        self.fixture.modules["maintenance"] = self.maintenance
        storage = ast.parse((BOUNDARIES["ROOT"] / "truememory/storage.py").read_text(encoding="utf-8"))
        self.fixture.modules["storage"].DEFAULT_BUSY_TIMEOUT_MS = next(
            ast.literal_eval(node.value) for node in storage.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "DEFAULT_BUSY_TIMEOUT_MS" for target in node.targets))
        self.coordinator = self.maintenance.get_coordinator(self.first.db_path)
        self.fixture.engine_namespace["_HAS_HYBRID"] = True
        self.maintenance.run_engine_maintenance = self.work
        self.fixture.enterContext(patch.object(Path, "exists", return_value=False))
        self.addCleanup(self.finish)

    def work(self, conn, coordinator, **kwargs):
        self.entered.set()
        if not self.release.wait(3):
            raise AssertionError("Synthetic automatic worker was not released")
        return None

    def finish(self):
        self.release.set()
        self.coordinator.wait(3)

    def start(self, receipt=True):
        values = {"_initialization_receipt": self.first._reconnect_receipt} if receipt else {}
        self.assertTrue(self.coordinator.request_layers(**values))
        self.assertTrue(self.entered.wait(2))
        self.assertEqual(self.coordinator.status[0], "running")

    def fresh(self):
        engine = self.first.__class__()
        engine.conn, engine.db_path = None, self.first.db_path
        engine._init_lock = threading.Lock()
        engine._write_lock = self.api.ConnectionWriteLock(engine)
        engine.ready = engine._runtime_initialized = engine._has_vectors = False
        engine._has_hybrid = True
        engine._runtime_vector_connection = engine._runtime_vector_generation = None
        engine._reconnect_receipt = None
        return engine

    def record(self):
        with self.maintenance._active_initialization_receipt(self.first.db_path) as value:
            return value

    def test_fresh_engine_adopts_only_validated_active_automatic_receipt(self):
        self.start()
        record = self.record()
        self.assertIs(record[0](), self.coordinator)
        self.assertEqual(record[2], self.first._reconnect_receipt)
        engine = self.fresh()
        candidate = self.fixture.candidate(self.first)
        queries = []
        candidate.set_trace_callback(queries.append)
        engine._open_connection_handle()
        self.assertIs(engine.conn, candidate)
        self.assertTrue(engine.ready)
        self.assertTrue(engine._runtime_initialized)
        self.assertFalse(engine._has_vectors)
        self.assertFalse(engine._has_hybrid)
        self.assertEqual(engine._reconnect_receipt, self.first._reconnect_receipt)
        self.assertFalse(candidate.in_transaction)
        self.assertFalse(self.release.is_set())
        self.assertEqual(self.maintenance._held_paths[self.first.db_path], 71)
        self.assertFalse(any(q.lstrip().split(" ", 1)[0].upper() in {"INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"}
                             for q in queries))
        self.assertEqual(self.fixture.fixture.loads, [])

    def test_manual_or_receiptless_automatic_owner_remains_busy(self):
        for automatic in (False, True):
            with self.subTest(automatic=automatic):
                if automatic:
                    self.start(receipt=False)
                else:
                    self.assertTrue(self.coordinator.request(lambda conn, cancel: self.work(conn, self.coordinator)))
                    self.assertTrue(self.entered.wait(2))
                self.assertIsNone(self.record())
                engine = self.fresh()
                self.fixture.candidate(self.first)
                with self.assertRaises(self.maintenance.MaintenanceBusyError):
                    engine._open_connection_handle()
                self.assertIsNone(engine.conn)
                self.assertFalse(engine.ready)
                self.fixture.connect.assert_not_called()
                self.finish()
                self.entered.clear()
                self.release.clear()
                self.worker_conn = self.fixture.memory_connection()

    def test_worker_finish_revokes_before_descriptor_release(self):
        self.start()
        original = self.maintenance._release_owner
        observations = []
        def release(fd):
            observations.append(self.record())
            self.assertIsNone(self.coordinator._active_initialization)
            return original(fd)
        self.maintenance._release_owner = release
        self.finish()
        self.assertEqual(observations, [None])
        self.assertIsNone(self.record())
        self.assertIsNone(self.coordinator._pending_initialization)
        self.assertIsNone(self.coordinator._reserved_initialization)

    def test_cancel_during_validation_does_not_install_or_close_old_engine_state(self):
        self.start()
        engine = self.fresh()
        candidate = self.fixture.candidate(self.first)
        original = engine._reopen_initialized_connection
        def validate(path, **kwargs):
            result = original(path, **kwargs)
            self.coordinator.cancel()
            return result
        engine._reopen_initialized_connection = validate
        with self.assertRaises(self.maintenance.MaintenanceBusyError):
            engine._open_connection_handle()
        self.fixture.assert_closed(candidate)
        self.assertIsNone(engine.conn)
        self.assertFalse(engine.ready)
        self.assertIsNone(engine._reconnect_receipt)
        self.assertIsNone(self.record())
        self.assertEqual(self.maintenance._held_paths[self.first.db_path], 71)

    def test_same_fd_different_launch_or_coordinator_cannot_revalidate_old_record(self):
        self.start()
        old = self.record()
        with self.coordinator._mutex:
            token, receipt = self.coordinator._active_initialization
            self.coordinator._active_initialization = ((token[0], token[1], "different-launch"), receipt)
        with self.maintenance._active_initialization_receipt(self.first.db_path, expected=old[:2]) as value:
            self.assertIsNone(value)
        replacement = self.maintenance.MaintenanceCoordinator(self.first.db_path)
        with self.maintenance._registry_lock:
            self.maintenance._coordinators[self.first.db_path] = replacement
        with self.maintenance._active_initialization_receipt(self.first.db_path, expected=old[:2]) as value:
            self.assertIsNone(value)
        with self.maintenance._registry_lock:
            self.maintenance._coordinators[self.first.db_path] = self.coordinator

    def test_receipt_validation_does_not_hold_registry_or_coordinator_mutex(self):
        self.start()
        engine = self.fresh()
        self.fixture.candidate(self.first)
        original = engine._reopen_initialized_connection
        def validate(path, **kwargs):
            self.assertTrue(self.maintenance._registry_lock.acquire(blocking=False))
            self.maintenance._registry_lock.release()
            self.assertTrue(self.coordinator._mutex.acquire(blocking=False))
            self.coordinator._mutex.release()
            self.assertFalse(engine._write_lock.locked())
            self.assertIsNone(engine._reconnect_receipt)
            self.assertEqual(kwargs["receipt"], self.first._reconnect_receipt)
            return original(path, **kwargs)
        engine._reopen_initialized_connection = validate
        engine._open_connection_handle()
        self.assertTrue(engine.ready)

    def test_exact_pending_generation_and_receiptless_request_do_not_revive_evidence(self):
        self.start()
        first_generation = self.coordinator._notification_generation
        self.assertFalse(self.coordinator.request_layers(_initialization_receipt=self.first._reconnect_receipt))
        self.assertEqual(self.coordinator._pending_initialization[0], first_generation + 1)
        self.assertFalse(self.coordinator.request_layers(threshold=37))
        self.assertIsNone(self.coordinator._pending_initialization)
        self.assertEqual(self.coordinator._pending.generation, first_generation + 2)
        self.assertEqual(self.coordinator._pending.threshold, 37)
        self.coordinator.cancel()
        self.assertIsNone(self.coordinator._active_initialization)
        self.assertIsNone(self.coordinator._pending_initialization)
        self.assertIsNone(self.coordinator._reserved_initialization)

    def test_busy_launch_drops_evidence_and_keeps_exact_pending_work(self):
        self.maintenance._held_paths[self.first.db_path] = 91
        try:
            self.assertFalse(self.coordinator.request_layers(_initialization_receipt=self.first._reconnect_receipt))
            self.assertEqual(self.coordinator._pending.generation, 1)
            self.assertIsNone(self.coordinator._active_initialization)
            self.assertIsNone(self.coordinator._pending_initialization)
            self.assertIsNone(self.coordinator._reserved_initialization)
            self.assertIsNone(self.record())
        finally:
            self.maintenance._held_paths.pop(self.first.db_path)
            self.coordinator.cancel()

    def test_invalid_receipts_are_not_retained_or_exposed(self):
        receipt = self.first._reconnect_receipt
        invalid = [object(), ("/other/database", *receipt[1:]), (*receipt[:4], ("legacy", object(), "256", "edge", "rr"), *receipt[5:]),
                   (*receipt[:4], ("legacy", "x" * 513, "256", "edge", "rr"), *receipt[5:]),
                   (*receipt[:4], ("legacy", "\ud800", "256", "edge", "rr"), *receipt[5:])]
        self.start()
        for value in invalid:
            with self.subTest(kind=type(value).__name__):
                self.assertFalse(self.coordinator.request_layers(_initialization_receipt=value))
                self.assertIsNone(self.coordinator._pending_initialization)
        self.coordinator.cancel()

    def test_start_failure_revokes_before_owner_release(self):
        releases = []
        original = self.maintenance._release_owner
        def release(fd):
            releases.append(self.record())
            return original(fd)
        self.maintenance._release_owner = release
        self.maintenance.threading.Thread = lambda **kwargs: types.SimpleNamespace(start=Mock(side_effect=RuntimeError("synthetic start failure")))
        with self.assertRaises(RuntimeError):
            self.coordinator.request_layers(_initialization_receipt=self.first._reconnect_receipt)
        self.assertEqual(releases, [None])
        self.assertIsNone(self.coordinator._active_initialization)
        self.assertIsNone(self.coordinator._pending_initialization)
        self.assertIsNone(self.coordinator._reserved_initialization)
        self.assertFalse(self.coordinator._active)

    def test_fork_reset_clears_inherited_evidence_and_registry(self):
        self.coordinator._active = True
        self.coordinator._automatic_initialization = True
        self.coordinator._active_initialization = ((os.getpid(), 71, "synthetic-live"), self.first._reconnect_receipt)
        self.coordinator._pending_initialization = (1, self.first._reconnect_receipt)
        self.coordinator._reserved_initialization = (1, self.first._reconnect_receipt)
        self.maintenance._held_paths[self.first.db_path] = 71
        self.maintenance._held_fds.add(71)
        self.maintenance._after_fork()
        self.assertIsNone(self.coordinator._active_initialization)
        self.assertIsNone(self.coordinator._pending_initialization)
        self.assertIsNone(self.coordinator._reserved_initialization)
        self.assertEqual(len(self.maintenance._coordinators), 0)
        self.assertEqual(self.maintenance._held_paths, {})
        self.assertEqual(self.closed_fds, [71])
        self.assertIsNone(self.record())
        self.coordinator._active = False

    def test_admitted_model_or_dimension_change_refuses_ready_receipt_but_deep_override_remains_valid(self):
        self.start()
        engine = self.fresh()
        self.fixture.candidate(self.first)
        engine._open_connection_handle()
        module = definitions("engine.py", {"_initialize_connection"}, self.fixture.namespace(), methods=True)
        for key in (("legacy", "qwen3_256", "256", "base", "synthetic/default"),
                    ("legacy", "model2vec", "384", "edge", "synthetic/default")):
            operation = types.SimpleNamespace(selection=None, policy=None, key=key)
            with patch.object(self.api, "current_operation", return_value=operation):
                with self.assertRaises(self.api.TierRuntimeError):
                    module._initialize_connection(engine, _suppress_maintenance=True)
        for key in (self.first._reconnect_receipt[4], ("legacy", "model2vec", "256", "edge", "synthetic/deep")):
            operation = types.SimpleNamespace(selection=None, policy=None, key=key)
            with patch.object(self.api, "current_operation", return_value=operation), \
                    patch.object(engine.conn, "execute", side_effect=AssertionError("Ready check performed SQL")), \
                    patch.object(Path, "stat", side_effect=AssertionError("Ready check stat")):
                module._initialize_connection(engine, _suppress_maintenance=True)
        self.assertEqual(self.fixture.fixture.loads, [])


    def test_receiptless_successor_cannot_reuse_previous_live_owner_record(self):
        second_conn = self.fixture.memory_connection()
        connections = iter((self.worker_conn, second_conn))
        self.maintenance.create_db = lambda path: next(connections)
        second_entered, second_release = threading.Event(), threading.Event()
        calls = []
        def work(conn, coordinator, **kwargs):
            calls.append(conn)
            if len(calls) == 1:
                self.entered.set()
                if not self.release.wait(3):
                    raise AssertionError("First synthetic worker release timed out")
            else:
                second_entered.set()
                if not second_release.wait(3):
                    raise AssertionError("Successor synthetic worker release timed out")
        self.maintenance.run_engine_maintenance = work
        self.start()
        old = self.record()
        self.assertFalse(self.coordinator.request_layers())
        self.release.set()
        try:
            self.assertTrue(second_entered.wait(2))
            self.assertEqual(len(calls), 2)
            self.assertIsNone(self.record())
            with self.maintenance._active_initialization_receipt(self.first.db_path, expected=old[:2]) as observed:
                self.assertIsNone(observed)
            self.assertIsNone(self.coordinator._pending_initialization)
            self.assertIsNone(self.coordinator._reserved_initialization)
        finally:
            second_release.set()
            self.assertTrue(self.coordinator.wait(3))

    def test_actual_serving_admission_rechecks_embedding_after_candidate_validation(self):
        self.start()
        engine = self.fresh()
        candidate = self.fixture.candidate(self.first)
        original = engine._reopen_initialized_connection
        vector = self.fixture.fixture.vector
        def validate(path, **kwargs):
            result = original(path, **kwargs)
            vector.EMBEDDING_MODEL = "qwen3_256"
            return result
        engine._reopen_initialized_connection = validate
        engine._open_connection_handle()
        self.assertIs(engine.conn, candidate)
        module = definitions("engine.py", {"_initialize_connection"}, self.fixture.namespace(), methods=True)
        with self.api.serving_operation(candidate):
            with self.assertRaises(self.api.TierRuntimeError):
                module._initialize_connection(engine, _suppress_maintenance=True)
        vector.EMBEDDING_MODEL = "model2vec"
        vector._embedding_dim = 384
        with self.api.serving_operation(candidate):
            with self.assertRaises(self.api.TierRuntimeError):
                module._initialize_connection(engine, _suppress_maintenance=True)
        vector._embedding_dim = 256
        self.fixture.fixture.reranker._model_name = "synthetic/deep"
        with self.api.serving_operation(candidate, reranker_id="synthetic/deep"):
            with patch.object(candidate, "execute", side_effect=AssertionError("Ready admission performed SQL")):
                module._initialize_connection(engine, _suppress_maintenance=True)
        self.assertEqual(self.fixture.fixture.loads, [])



    def test_failed_start_preserves_only_newer_pending_generation_receipt(self):
        for donate in (True, False):
            with self.subTest(donate=donate):
                start_entered, allow_failure = threading.Event(), threading.Event()
                attempts, errors = [], []
                receipt = tuple(list(self.first._reconnect_receipt))
                class FailedStart:
                    def start(inner):
                        start_entered.set()
                        if not allow_failure.wait(3):
                            raise AssertionError("Synthetic start interleaving timed out")
                        raise RuntimeError("synthetic first start failed")
                def thread_factory(**kwargs):
                    attempts.append(kwargs)
                    return FailedStart() if len(attempts) == 1 else threading.Thread(**kwargs)
                self.maintenance.threading.Thread = thread_factory
                def request_first():
                    try:
                        self.coordinator.request_layers(threshold=31,
                            _initialization_receipt=self.first._reconnect_receipt)
                    except BaseException as error:
                        errors.append(error)
                requester = threading.Thread(target=request_first)
                requester.start()
                try:
                    self.assertTrue(start_entered.wait(2))
                    values = {"_initialization_receipt": receipt} if donate else {}
                    self.assertFalse(self.coordinator.request_layers(threshold=32, **values))
                    generation = self.coordinator._pending.generation
                    if donate:
                        self.assertEqual(self.coordinator._pending_initialization, (generation, receipt))
                    else:
                        self.assertIsNone(self.coordinator._pending_initialization)
                    allow_failure.set()
                    requester.join(2)
                    self.assertFalse(requester.is_alive())
                    self.assertEqual(len(errors), 1)
                    self.assertIs(type(errors[0]), RuntimeError)
                    self.assertTrue(self.entered.wait(2))
                    offered = self.record()
                    if donate:
                        self.assertIsNotNone(offered)
                        self.assertIs(offered[2], receipt)
                    else:
                        self.assertIsNone(offered)
                    self.assertEqual(len(attempts), 2)
                    self.assertEqual(self.coordinator._notification_generation, generation)
                    self.assertIsNone(self.coordinator._pending_initialization)
                    self.assertIsNone(self.coordinator._reserved_initialization)
                finally:
                    allow_failure.set()
                    self.release.set()
                    requester.join(3)
                    self.assertTrue(self.coordinator.wait(3))
                self.entered.clear()
                self.release.clear()
                self.worker_conn = self.fixture.memory_connection()

    def test_busy_acquisition_preserves_only_surviving_pending_generation_receipt(self):
        for donate in (True, False):
            with self.subTest(donate=donate):
                acquire_entered, allow_busy = threading.Event(), threading.Event()
                results, errors = [], []
                receipt = tuple(list(self.first._reconnect_receipt))
                def file_lock(path):
                    acquire_entered.set()
                    if not allow_busy.wait(3):
                        raise AssertionError("Synthetic owner interleaving timed out")
                    return None
                self.maintenance.try_file_lock = file_lock
                def request_first():
                    try:
                        results.append(self.coordinator.request_layers(threshold=31,
                            _initialization_receipt=self.first._reconnect_receipt))
                    except BaseException as error:
                        errors.append(error)
                requester = threading.Thread(target=request_first)
                requester.start()
                try:
                    self.assertTrue(acquire_entered.wait(2))
                    values = {"_initialization_receipt": receipt} if donate else {}
                    self.assertFalse(self.coordinator.request_layers(threshold=32, **values))
                    generation = self.coordinator._pending.generation
                    allow_busy.set()
                    requester.join(2)
                    self.assertFalse(requester.is_alive())
                    self.assertEqual(errors, [])
                    self.assertEqual(results, [False])
                    self.assertEqual(self.coordinator._pending.generation, generation)
                    self.assertEqual(self.coordinator._pending.threshold, 32)
                    if donate:
                        self.assertEqual(self.coordinator._pending_initialization, (generation, receipt))
                        self.assertIs(self.coordinator._pending_initialization[1], receipt)
                    else:
                        self.assertIsNone(self.coordinator._pending_initialization)
                    self.assertIsNone(self.coordinator._active_initialization)
                    self.assertIsNone(self.coordinator._reserved_initialization)
                    self.assertIsNone(self.record())
                finally:
                    allow_busy.set()
                    requester.join(3)
                    self.coordinator.cancel()



if __name__ == "__main__":
    unittest.main()
