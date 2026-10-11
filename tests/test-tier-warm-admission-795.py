"""Fresh public admission against real lifecycle code and memory SQLite only."""
from __future__ import annotations

import contextlib
from pathlib import Path
import runpy
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

ACTIVE = runpy.run_path(str(Path(__file__).with_name("test-tier-active-initialization-795.py")))
enter_context = ACTIVE["enter_context"]


class TestWarmAdmission(unittest.TestCase):
    def setUp(self):
        self.active = ACTIVE["TestActiveInitialization"]()
        self.addCleanup(self.active.doCleanups)
        self.active.setUp()
        self.fixture = self.active.fixture
        self.api, self.maintenance = self.active.api, self.active.maintenance
        self.coordinator = self.active.coordinator
        self.gate = self.fixture.fixture.gate
        self.maintenance.threading.TIMEOUT_MAX = threading.TIMEOUT_MAX
        loaded = ACTIVE["definitions"]("maintenance.py", {"wait_for_automatic_owner"},
            {**self.maintenance.__dict__, "time": time})
        self.maintenance.wait_for_automatic_owner = loaded.wait_for_automatic_owner
        self.opened = self.fixture.memory_connection()
        self.fixture.engine_namespace["create_db"] = Mock(return_value=self.opened)
        self.errors = []
        initialization = ACTIVE["definitions"]("engine.py", {
            "_initialize_connection", "_purge_legacy_entity_profile_summaries",
            "_maybe_startup_consolidate", "_maybe_auto_consolidate"},
            {**self.fixture.engine_namespace, "_HAS_HYBRID": False, "os": __import__("os")}, methods=True)
        for name in ("_initialize_connection", "_purge_legacy_entity_profile_summaries",
                     "_maybe_startup_consolidate", "_maybe_auto_consolidate"):
            setattr(self.active.first.__class__, name, getattr(initialization, name))
        self.active.first.__class__._has_consolidation = False

    def drifted_candidate(self):
        candidate = self.fixture.candidate(self.active.first)
        candidate.execute("CREATE TABLE synthetic_changed_schema(value INTEGER)")
        return candidate

    def run_public(self, engine):
        try:
            self.api.open_engine_for_operation(engine)
        except BaseException as error:
            self.errors.append(error)

    def observe_wait(self, action=None):
        entered = threading.Event()
        actual = self.maintenance.wait_for_automatic_owner
        def wait(error, *, deadline):
            self.assertIsNone(self.api.current_runtime_operation())
            self.assertFalse(self.gate.current_thread_admitted())
            self.assertTrue(self.maintenance._registry_lock.acquire(blocking=False))
            self.maintenance._registry_lock.release()
            self.assertTrue(self.coordinator._mutex.acquire(blocking=False))
            self.coordinator._mutex.release()
            entered.set()
            if action is not None:
                action()
            return actual(error, deadline=deadline)
        enter_context(self, patch.object(self.maintenance, "wait_for_automatic_owner", side_effect=wait))
        return entered

    def test_schema_drift_waits_then_uses_normal_factory_without_stale_readiness(self):
        self.active.start()
        candidate = self.drifted_candidate()
        engine = self.active.fresh()
        entered = self.observe_wait()
        worker = threading.Thread(target=self.run_public, args=(engine,))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertIsNone(engine.conn)
            self.assertFalse(engine.ready)
            self.assertTrue(engine._write_lock.acquire(blocking=False))
            engine._write_lock.release()
            self.assertTrue(engine._init_lock.acquire(blocking=False))
            engine._init_lock.release()
            self.fixture.assert_closed(candidate)
            self.active.release.set()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(self.errors, [])
            self.assertIs(engine.conn, self.opened)
            self.assertFalse(engine.ready)
            self.assertIsNone(engine._reconnect_receipt)
            self.assertEqual(self.fixture.fixture.loads, [])
            self.fixture.engine_namespace["create_db"].assert_called_once()
        finally:
            self.active.release.set()
            worker.join(3)

    def test_private_opener_still_refuses_same_schema_drift_without_wait(self):
        self.active.start()
        self.drifted_candidate()
        engine = self.active.fresh()
        with patch.object(self.maintenance, "wait_for_automatic_owner") as wait:
            with self.assertRaises(self.maintenance.MaintenanceBusyError):
                engine._open_connection_handle()
        wait.assert_not_called()
        self.assertIsNone(engine.conn)

    def test_successful_first_attempt_preserves_existing_raw_reader_and_exclusive_behavior(self):
        for context in (lambda: self.gate.operation_lease(("synthetic",)), self.gate.exclusive_activation):
            with self.subTest(context=context), context():
                engine = self.active.fresh()
                with patch.object(self.maintenance, "wait_for_automatic_owner") as wait:
                    self.api.open_engine_for_operation(engine)
                self.assertIs(engine.conn, self.opened)
                wait.assert_not_called()

    def test_busy_runtime_raw_reader_and_exclusive_scopes_never_wait(self):
        self.active.start()
        scopes = (lambda: self.api.serving_operation(self.fixture.conn),
                  lambda: self.gate.operation_lease(("synthetic",)), self.gate.exclusive_activation)
        for scope in scopes:
            with self.subTest(scope=scope), scope():
                self.drifted_candidate()
                engine = self.active.fresh()
                with patch.object(self.maintenance, "wait_for_automatic_owner") as wait:
                    with self.assertRaises(self.maintenance.MaintenanceBusyError):
                        self.api.open_engine_for_operation(engine)
                wait.assert_not_called()
                self.assertIsNone(engine.conn)

    def test_current_writer_refuses_before_nonreentrant_open(self):
        engine = self.active.fresh()
        with engine._write_lock, patch.object(engine, "_open_connection_handle") as opened:
            with self.assertRaises(self.maintenance.MaintenanceBusyError):
                self.api.open_engine_for_operation(engine)
        opened.assert_not_called()

    def test_existing_connection_and_caller_sql_never_enter_wait_loop(self):
        engine = self.active.first
        engine.conn.execute("BEGIN")
        error = self.maintenance.MaintenanceBusyError("synthetic existing connection")
        with patch.object(engine, "_open_connection_handle", side_effect=error) as opened, \
                patch.object(self.maintenance, "wait_for_automatic_owner") as wait:
            with self.assertRaises(self.maintenance.MaintenanceBusyError) as raised:
                self.api.open_engine_for_operation(engine)
        self.assertIs(raised.exception, error)
        self.assertTrue(engine.conn.in_transaction)
        opened.assert_called_once()
        wait.assert_not_called()
        engine.conn.rollback()

    def test_manual_receiptless_and_foreign_owners_have_no_wait_permission(self):
        for kind in ("manual", "receiptless", "foreign"):
            with self.subTest(kind=kind):
                if kind == "manual":
                    self.assertTrue(self.coordinator.request(self.active.work))
                    self.assertTrue(self.active.entered.wait(2))
                elif kind == "receiptless":
                    self.active.start(receipt=False)
                else:
                    self.maintenance._held_paths[self.active.first.db_path] = 991
                try:
                    with self.assertRaises(self.maintenance.MaintenanceBusyError) as raised:
                        self.api.open_engine_for_operation(self.active.fresh())
                    self.assertIsNone(getattr(raised.exception, "_automatic_owner", None))
                finally:
                    if kind == "foreign":
                        self.maintenance._held_paths.clear()
                    else:
                        self.active.release.set()
                        self.assertTrue(self.coordinator.wait(2))
                        self.active.entered.clear()
                        self.active.release.clear()
                        self.active.worker_conn = self.fixture.memory_connection()

    def test_launch_gap_wait_origin_precedes_active_receipt_publication(self):
        acquired, continue_launch = threading.Event(), threading.Event()
        actual = self.coordinator._begin_observation
        def observe():
            if not acquired.is_set():
                acquired.set()
                if not continue_launch.wait(3):
                    raise AssertionError("Synthetic launch barrier timed out")
            return actual()
        enter_context(self, patch.object(self.coordinator, "_begin_observation", side_effect=observe))
        requester = threading.Thread(target=lambda: self.coordinator.request_layers(
            _initialization_receipt=self.active.first._reconnect_receipt))
        requester.start()
        engine = self.active.fresh()
        entered = self.observe_wait()
        worker = threading.Thread(target=self.run_public, args=(engine,))
        try:
            self.assertTrue(acquired.wait(2))
            self.assertIsNone(self.active.record())
            self.assertIsNotNone(self.coordinator._admission_owner)
            worker.start()
            self.assertTrue(entered.wait(2))
            continue_launch.set()
            self.active.release.set()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(self.errors, [])
            self.assertIs(engine.conn, self.opened)
        finally:
            continue_launch.set()
            self.active.release.set()
            requester.join(3)
            if worker.ident is not None:
                worker.join(3)

    def test_teardown_gap_keeps_wait_origin_until_actual_descriptor_release(self):
        self.active.start()
        at_release, continue_release = threading.Event(), threading.Event()
        actual = self.maintenance._release_owner
        def release(fd):
            at_release.set()
            if not continue_release.wait(3):
                raise AssertionError("Synthetic release barrier timed out")
            return actual(fd)
        enter_context(self, patch.object(self.maintenance, "_release_owner", side_effect=release))
        self.active.release.set()
        self.assertTrue(at_release.wait(2))
        self.assertIsNone(self.active.record())
        origin = self.coordinator._admission_owner
        self.assertIsNotNone(origin)
        entered = self.observe_wait()
        engine = self.active.fresh()
        worker = threading.Thread(target=self.run_public, args=(engine,))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertFalse(origin.released.is_set())
            continue_release.set()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(self.errors, [])
            self.assertTrue(origin.released.is_set())
            self.assertIsNone(self.coordinator._admission_owner)
        finally:
            continue_release.set()
            worker.join(3)

    def test_cancel_wakes_waiter_without_releasing_owner_or_retrying(self):
        self.active.start()
        self.drifted_candidate()
        engine = self.active.fresh()
        entered = self.observe_wait()
        worker = threading.Thread(target=self.run_public, args=(engine,))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            origin = self.coordinator._admission_owner
            self.coordinator.cancel()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(self.errors), 1)
            self.assertIs(type(self.errors[0]), self.maintenance.MaintenanceBusyError)
            self.assertTrue(origin.cancelled.is_set())
            self.assertIn(self.active.first.db_path, self.maintenance._held_paths)
            self.assertIsNone(engine.conn)
            self.fixture.engine_namespace["create_db"].assert_not_called()
        finally:
            self.active.release.set()
            worker.join(3)

    def test_timeout_leaves_owner_and_engine_untouched(self):
        self.active.start()
        candidate = self.drifted_candidate()
        engine = self.active.fresh()
        with patch.object(self.fixture.modules["storage"], "DEFAULT_BUSY_TIMEOUT_MS", 20):
            with self.assertRaises(self.maintenance.MaintenanceBusyError):
                self.api.open_engine_for_operation(engine)
        self.assertIsNone(engine.conn)
        self.assertFalse(engine.ready)
        self.assertFalse(self.active.release.is_set())
        self.assertIn(self.active.first.db_path, self.maintenance._held_paths)
        self.fixture.assert_closed(candidate)

    def test_successive_automatic_generations_share_one_deadline(self):
        self.active.start()
        self.drifted_candidate()
        deadlines = []
        actual = self.maintenance.wait_for_automatic_owner
        def wait(error, *, deadline):
            deadlines.append(deadline)
            self.active.release.set()
            actual(error, deadline=deadline)
            self.assertTrue(self.coordinator.wait(2))
            if len(deadlines) == 1:
                self.active.entered.clear()
                self.active.release.clear()
                self.active.worker_conn = self.fixture.memory_connection()
                self.active.start()
                self.drifted_candidate()
        engine = self.active.fresh()
        with patch.object(self.maintenance, "wait_for_automatic_owner", side_effect=wait):
            self.api.open_engine_for_operation(engine)
        self.assertEqual(len(deadlines), 2)
        self.assertEqual(deadlines[0], deadlines[1])
        self.assertIs(engine.conn, self.opened)

    def test_context_is_rechecked_after_wake_before_retry(self):
        self.active.start()
        self.drifted_candidate()
        engine = self.active.fresh()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        actual = self.maintenance.wait_for_automatic_owner
        def wait(error, *, deadline):
            self.active.release.set()
            actual(error, deadline=deadline)
            stack.enter_context(self.gate.operation_lease(("synthetic",)))
        with patch.object(self.maintenance, "wait_for_automatic_owner", side_effect=wait):
            with self.assertRaises(self.maintenance.MaintenanceBusyError):
                self.api.open_engine_for_operation(engine)
        self.assertIsNone(engine.conn)
        self.fixture.engine_namespace["create_db"].assert_not_called()

    def test_busy_origin_survives_release_before_wait_is_called(self):
        self.active.start()
        self.drifted_candidate()
        engine = self.active.fresh()
        actual = engine._open_connection_handle
        calls = []
        def opening():
            calls.append(None)
            try:
                return actual()
            except self.maintenance.MaintenanceBusyError as error:
                self.assertIsNotNone(error._automatic_owner)
                self.active.release.set()
                self.assertTrue(self.coordinator.wait(2))
                self.assertTrue(error._automatic_owner.released.is_set())
                self.assertIsNone(self.coordinator._admission_owner)
                raise
        with patch.object(engine, "_open_connection_handle", side_effect=opening):
            self.api.open_engine_for_operation(engine)
        self.assertEqual(len(calls), 2)
        self.assertIs(engine.conn, self.opened)

    def test_receiptless_successor_cannot_inherit_wait_permission(self):
        self.active.start()
        self.drifted_candidate()
        actual = self.maintenance.wait_for_automatic_owner
        origins, deadlines = [], []
        def wait(error, *, deadline):
            origins.append(getattr(error, "_automatic_owner", None))
            deadlines.append(deadline)
            if len(origins) == 1:
                self.active.release.set()
            actual(error, deadline=deadline)
            self.assertTrue(self.coordinator.wait(2))
            self.active.entered.clear()
            self.active.release.clear()
            self.active.worker_conn = self.fixture.memory_connection()
            self.active.start(receipt=False)
        engine = self.active.fresh()
        with patch.object(self.maintenance, "wait_for_automatic_owner", side_effect=wait):
            with self.assertRaises(self.maintenance.MaintenanceBusyError):
                self.api.open_engine_for_operation(engine)
        self.assertEqual(len(origins), 2)
        self.assertIsNotNone(origins[0])
        self.assertIsNone(origins[1])
        self.assertEqual(deadlines[0], deadlines[1])
        self.assertIsNone(engine.conn)
        self.fixture.engine_namespace["create_db"].assert_not_called()

    def test_cancelled_origin_stays_cancelled_after_later_launch_and_fd_reuse(self):
        self.active.start()
        with self.assertRaises(self.maintenance.MaintenanceBusyError) as raised:
            with self.maintenance.maintenance_owner(self.active.first.db_path):
                self.fail("Automatic ownership was stolen")
        old = raised.exception._automatic_owner
        self.coordinator.cancel()
        self.active.release.set()
        self.assertTrue(self.coordinator.wait(2))
        self.active.entered.clear()
        self.active.release.clear()
        self.active.worker_conn = self.fixture.memory_connection()
        self.active.start()
        current = self.coordinator._admission_owner
        self.assertEqual(old.descriptor, current.descriptor)
        self.assertNotEqual(old.launch, current.launch)
        self.assertTrue(old.cancelled.is_set())
        self.assertFalse(current.cancelled.is_set())
        with self.assertRaises(self.maintenance.MaintenanceBusyError) as again:
            self.maintenance.wait_for_automatic_owner(raised.exception, deadline=time.monotonic() + 10)
        self.assertIs(again.exception, raised.exception)

    def test_decorator_fact_and_ensure_connection_use_public_helper(self):
        called = []
        engine = self.active.fresh()
        actual = self.api.open_engine_for_operation
        def opening(value):
            called.append(value)
            return actual(value)
        @self.api.engine_operation
        def body(value):
            self.assertIsNotNone(self.api.current_runtime_operation())
            return value.conn
        with patch.object(self.api, "open_engine_for_operation", side_effect=opening):
            self.assertIs(body(engine), self.opened)
            with self.api.fact_operation(types.SimpleNamespace(_engine=engine)):
                self.assertIsNotNone(self.api.current_runtime_operation())
            methods = ACTIVE["definitions"]("engine.py", {"_ensure_connection"},
                self.fixture.namespace(), methods=True)
            engine._initialize_connection = Mock()
            methods._ensure_connection(engine)
        self.assertEqual(called, [engine, engine, engine])
        engine._initialize_connection.assert_called_once_with(_suppress_maintenance=False)

    def test_gate_query_refuses_contention_without_waiting(self):
        held, release = threading.Event(), threading.Event()
        def holder():
            with self.gate._DEFAULT_GATE._condition:
                held.set()
                release.wait(3)
        thread = threading.Thread(target=holder)
        thread.start()
        try:
            self.assertTrue(held.wait(2))
            with self.assertRaises(self.gate.ServingLeaseError):
                self.gate.current_thread_admitted()
        finally:
            release.set()
            thread.join(3)

    def test_wait_eligibility_refuses_registry_and_coordinator_contention(self):
        self.active.start()
        with self.assertRaises(self.maintenance.MaintenanceBusyError) as raised:
            with self.maintenance.maintenance_owner(self.active.first.db_path):
                self.fail("Automatic ownership was stolen")
        error = raised.exception
        for lock in (self.maintenance._registry_lock, self.coordinator._mutex):
            with self.subTest(lock=lock):
                entered, done = threading.Event(), threading.Event()
                errors = []
                def waiting():
                    entered.set()
                    try:
                        self.maintenance.wait_for_automatic_owner(error, deadline=time.monotonic() + 0.035)
                    except BaseException as failure:
                        errors.append(failure)
                    finally:
                        done.set()
                with lock:
                    worker = threading.Thread(target=waiting)
                    worker.start()
                    self.assertTrue(entered.wait(2))
                    finished_while_held = done.wait(0.1)
                worker.join(2)
                self.assertTrue(finished_while_held)
                self.assertEqual(errors, [error])
                self.assertFalse(self.active.release.is_set())

    def test_runtime_presence_refuses_without_validating_lease_under_contended_gate(self):
        self.active.start()
        self.drifted_candidate()
        engine = self.active.fresh()
        admitted_connection = self.fixture.memory_connection()
        admitted, proceed, caught = threading.Event(), threading.Event(), threading.Event()
        errors = []
        def nested():
            with self.api.serving_operation(admitted_connection):
                admitted.set()
                if not proceed.wait(3):
                    raise AssertionError("Synthetic runtime barrier timed out")
                try:
                    self.api.open_engine_for_operation(engine)
                except BaseException as error:
                    errors.append(error)
                finally:
                    caught.set()
        worker = threading.Thread(target=nested)
        worker.start()
        try:
            self.assertTrue(admitted.wait(2))
            with self.gate._DEFAULT_GATE._condition:
                proceed.set()
                refused_while_held = caught.wait(0.1)
            worker.join(2)
            self.assertTrue(refused_while_held)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIs(type(errors[0]), self.maintenance.MaintenanceBusyError)
            self.assertIsNone(engine.conn)
        finally:
            proceed.set()
            worker.join(3)


if __name__ == "__main__":
    unittest.main()
