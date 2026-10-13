"""Serving admission through stdlib events and controlled condition wait points."""

from __future__ import annotations

import importlib.util
import gc
import math
import sys
import threading
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "truememory/tier_switch/serving.py"
KEY = ("synthetic-database", "synthetic-generation")


def load() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("synthetic_serving_gate", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Cancelled(BaseException):
    pass


class TestServingGate(unittest.TestCase):
    def setUp(self) -> None:
        self.api = load()
        self.gate = self.api.ServingGate()
        self.errors = []
        self.threads = []
        self.releases = []
        self.waiting = {}
        self.wait_arguments = []
        original = self.gate._condition.wait

        def observed_wait(timeout=None):
            name = threading.current_thread().name
            self.wait_arguments.append((name, timeout))
            if name in self.waiting:
                self.waiting[name].set()
            return original(timeout)

        self.gate._condition.wait = observed_wait
        self.addCleanup(self.cleanup_threads)

    def cleanup_threads(self) -> None:
        for event in self.releases:
            event.set()
        for thread in self.threads:
            thread.join(timeout=3)
        self.assertFalse(any(thread.is_alive() for thread in self.threads), "Synthetic thread did not finish")

    def event(self) -> threading.Event:
        event = threading.Event()
        self.releases.append(event)
        return event

    def start(self, name, work):
        done = threading.Event()
        self.waiting[name] = threading.Event()
        def run():
            try:
                work()
            except BaseException as error:
                self.errors.append(error)
            finally:
                done.set()
        thread = threading.Thread(target=run, name=name, daemon=True)
        self.threads.append(thread)
        thread.start()
        return done

    def reached(self, event: threading.Event) -> None:
        self.assertTrue(event.wait(timeout=3), "Synthetic event was not reached")

    def writer(self, entered, release=None, **kwargs):
        def work():
            with self.gate.exclusive_activation(**kwargs):
                entered.set()
                if release is not None:
                    self.reached(release)
        return work

    def assert_idle(self) -> None:
        with self.gate.exclusive_activation(timeout=1):
            self.assertFalse(self.gate._readers)
            self.assertFalse(self.gate._children)
            self.assertFalse(self.gate._waiting_writers)
        self.assertFalse(self.gate._writers)
        self.assertEqual(self.errors, [])

    def test_concurrent_readers_and_same_thread_nested_calls(self) -> None:
        child_entered, child_release = threading.Event(), self.event()
        def reader():
            with self.gate.operation_lease(KEY) as lease:
                self.assertEqual(lease.selection_key, KEY)
                child_entered.set()
                self.reached(child_release)
        with self.gate.operation_lease(KEY) as outer:
            done = self.start("reader", reader)
            self.reached(child_entered)
            with self.gate.operation_lease(KEY) as inner:
                self.assertEqual(outer.selection_key, inner.selection_key)
                self.assertIsNot(inner, outer)
            child_release.set()
            self.reached(done)
        self.assert_idle()

    def test_waiting_writer_blocks_unrelated_reader_but_not_nested_reader(self) -> None:
        writer_entered, writer_release, reader_entered = threading.Event(), self.event(), threading.Event()
        with self.gate.operation_lease(KEY):
            writer_done = self.start("writer", self.writer(writer_entered, writer_release))
            self.reached(self.waiting["writer"])
            def reader():
                with self.gate.operation_lease(KEY):
                    reader_entered.set()
            reader_done = self.start("late-reader", reader)
            self.reached(self.waiting["late-reader"])
            with self.gate.operation_lease(KEY):
                self.assertFalse(writer_entered.is_set())
                self.assertFalse(reader_entered.is_set())
        self.reached(writer_entered)
        self.assertFalse(reader_entered.is_set())
        writer_release.set()
        self.reached(writer_done)
        self.reached(reader_done)
        self.assert_idle()

    def test_reserved_child_outlives_parent_and_joins_ahead_of_waiting_writer(self) -> None:
        writer_entered, child_entered, child_release = threading.Event(), threading.Event(), self.event()
        with self.gate.operation_lease(KEY) as lease:
            child = lease.fork_child()
        writer_done = self.start("writer", self.writer(writer_entered))
        self.reached(self.waiting["writer"])
        def task():
            with child.join() as joined:
                self.assertEqual(joined.selection_key, KEY)
                child_entered.set()
                self.reached(child_release)
        child_done = self.start("child", task)
        self.reached(child_entered)
        self.assertFalse(writer_entered.is_set())
        child_release.set()
        self.reached(child_done)
        self.reached(writer_done)
        with self.assertRaises(self.api.ServingLeaseError):
            with child.join():
                self.fail("Consumed reservation was reused")
        child.close()
        self.assert_idle()

    def test_parent_can_wait_for_parallel_children_with_a_writer_queued(self) -> None:
        writer_entered = threading.Event()
        with self.gate.operation_lease(KEY) as lease:
            children = [lease.fork_child() for _ in range(3)]
            writer_done = self.start("writer", self.writer(writer_entered))
            self.reached(self.waiting["writer"])
            child_events = []
            for index, child in enumerate(children):
                def task(reservation=child):
                    with reservation.join() as joined:
                        with self.gate.operation_lease(KEY):
                            self.assertEqual(joined.selection_key, KEY)
                child_events.append(self.start(f"child-{index}", task))
            for done in child_events:
                self.reached(done)
            self.assertFalse(writer_entered.is_set())
        self.reached(writer_done)
        self.assert_idle()

    def test_unused_reservation_close_unpins_writer_and_is_idempotent(self) -> None:
        entered = threading.Event()
        with self.gate.operation_lease(KEY) as lease:
            child = lease.fork_child()
        done = self.start("writer", self.writer(entered))
        self.reached(self.waiting["writer"])
        child.close()
        child.close()
        self.reached(done)
        self.assertTrue(entered.is_set())
        with self.assertRaises(self.api.ServingLeaseError):
            with child.join():
                self.fail("Closed reservation was joined")
        self.assert_idle()

    def test_active_join_cannot_be_closed_or_joined_twice(self) -> None:
        entered, release = threading.Event(), self.event()
        with self.gate.operation_lease(KEY) as lease:
            child = lease.fork_child()
        def task():
            with child.join():
                entered.set()
                self.reached(release)
        done = self.start("child", task)
        self.reached(entered)
        with self.assertRaises(self.api.ServingLeaseError):
            child.close()
        with self.assertRaises(self.api.ServingLeaseError):
            with child.join():
                self.fail("Concurrent duplicate join was admitted")
        self.assertEqual(len(self.gate._readers), 1)
        release.set()
        self.reached(done)
        self.assert_idle()

    def test_child_can_reserve_grandchild_without_extending_writer_privilege(self) -> None:
        with self.gate.operation_lease(KEY) as lease:
            child = lease.fork_child()
        with child.join() as joined:
            grandchild = joined.fork_child()
        with grandchild.join() as final:
            self.assertEqual(final.selection_key, KEY)
        self.assert_idle()

    def test_read_upgrade_refuses_without_waiting_or_poisoning_queue(self) -> None:
        with self.gate.operation_lease(KEY):
            with self.assertRaises(self.api.ServingLeaseError):
                with self.gate.exclusive_activation():
                    self.fail("Read was upgraded")
            self.assertFalse(self.gate._waiting_writers)
        self.assert_idle()

    def test_writer_reentry_and_sync_reader_are_safe_but_not_exportable(self) -> None:
        with self.gate.exclusive_activation():
            with self.gate.exclusive_activation():
                with self.gate.operation_lease(KEY) as lease:
                    with self.gate.exclusive_activation():
                        self.assertEqual(lease.selection_key, KEY)
                    with self.assertRaises(self.api.ServingLeaseError):
                        lease.fork_child()
        self.assert_idle()

    def test_writer_owned_read_still_pins_after_outer_writer_exits_first(self) -> None:
        writer = self.gate.exclusive_activation()
        writer.__enter__()
        reader = self.gate.operation_lease(KEY)
        lease = reader.__enter__()
        writer.__exit__(None, None, None)
        entered = threading.Event()
        done = self.start("next-writer", self.writer(entered))
        self.reached(self.waiting["next-writer"])
        with self.gate.operation_lease(KEY) as nested:
            with self.assertRaises(self.api.ServingLeaseError):
                nested.fork_child()
        self.assertFalse(entered.is_set())
        reader.__exit__(None, None, None)
        self.reached(done)
        with self.assertRaises(self.api.ServingLeaseError):
            lease.fork_child()
        self.assert_idle()

    def test_writer_fifo_and_cancelled_writer_wakes_unrelated_readers(self) -> None:
        order, first_release = [], self.event()
        def first():
            with self.gate.exclusive_activation():
                order.append("first")
                self.reached(first_release)
        def second():
            with self.gate.exclusive_activation():
                order.append("second")
        with self.gate.operation_lease(KEY):
            first_done = self.start("first-writer", first)
            self.reached(self.waiting["first-writer"])
            second_done = self.start("second-writer", second)
            self.reached(self.waiting["second-writer"])
        # The first writer must remain ahead until its release event.
        self.reached(self.waiting["second-writer"])
        first_release.set()
        self.reached(first_done)
        self.reached(second_done)
        self.assertEqual(order, ["first", "second"])
        self.assert_idle()

    def test_cancellation_while_blocker_is_held_uses_bounded_condition_wait(self) -> None:
        cancel, done, reader_entered = threading.Event(), None, threading.Event()
        outcomes = []
        def waiting_writer():
            try:
                with self.gate.exclusive_activation(cancelled=cancel):
                    self.fail("Cancelled writer admitted")
            except self.api.ServingLeaseCancelled:
                outcomes.append("cancelled")
        with self.gate.operation_lease(KEY):
            done = self.start("cancel-writer", waiting_writer)
            self.reached(self.waiting["cancel-writer"])
            def late_reader():
                with self.gate.operation_lease(KEY):
                    reader_entered.set()
            late_done = self.start("reader-after-cancel", late_reader)
            self.reached(self.waiting["reader-after-cancel"])
            cancel.set()
            self.reached(done)
            self.reached(late_done)
            self.assertTrue(reader_entered.is_set())
        self.assertEqual(outcomes, ["cancelled"])
        waits = [timeout for name, timeout in self.wait_arguments if name == "cancel-writer"]
        self.assertTrue(waits)
        self.assertTrue(all(0 < timeout <= 0.05 for timeout in waits))
        self.assert_idle()

    def test_expired_admission_and_inherited_child_cancellation_release_resources(self) -> None:
        cancel = threading.Event()
        with self.gate.operation_lease(KEY, cancelled=cancel) as lease:
            child = lease.fork_child()
        cancel.set()
        with self.assertRaises(self.api.ServingLeaseCancelled):
            with child.join():
                self.fail("Child escaped inherited cancellation")
        self.assertIn(child, self.gate._children)
        child.close()
        self.assertFalse(self.gate._children)
        with patch.object(self.api.time, "monotonic", return_value=100.0):
            for context in (self.gate.operation_lease(KEY, deadline=100.0),
                            self.gate.exclusive_activation(deadline=100.0)):
                with self.assertRaises(self.api.ServingLeaseTimeout):
                    with context:
                        self.fail("Expired admission ran")
        self.assert_idle()

    def test_deadline_consumed_during_wait_refuses_and_removes_writer_ticket(self) -> None:
        clock = [100.0]
        with self.gate.operation_lease(KEY):
            def timed_wait(timeout=None):
                clock[0] = 102.0
            # A separate thread is needed to avoid the explicit upgrade guard.
            outcomes = []
            def waiter():
                try:
                    with self.gate.exclusive_activation(deadline=101.0):
                        self.fail("Expired queued writer ran")
                except self.api.ServingLeaseTimeout:
                    outcomes.append("expired")
            with patch.object(self.api.time, "monotonic", side_effect=lambda: clock[0]), patch.object(self.gate._condition, "wait", timed_wait):
                done = self.start("deadline-writer", waiter)
                self.reached(done)
            self.assertEqual(outcomes, ["expired"])
            self.assertFalse(self.gate._waiting_writers)
        self.assert_idle()

    def test_initial_bookkeeping_lock_timeout_does_not_wait_again_for_cleanup(self) -> None:
        class RefusingCondition:
            def __init__(self):
                self.calls = []
            def acquire(self, **kwargs):
                self.calls.append(kwargs)
                clock[0] = 102.0
                return False
            def release(self):
                raise AssertionError("Unacquired lock released")
        clock = [100.0]
        condition = RefusingCondition()
        for kind in ("reader", "writer"):
            gate = self.api.ServingGate()
            gate._condition = condition
            clock[0] = 100.0
            with patch.object(self.api.time, "monotonic", side_effect=lambda: clock[0]):
                context = gate.operation_lease(KEY, deadline=101.0) if kind == "reader" else gate.exclusive_activation(deadline=101.0)
                with self.assertRaises(self.api.ServingLeaseTimeout):
                    with context:
                        self.fail("Timed-out bookkeeping admission ran")
            self.assertFalse(gate._readers)
            self.assertFalse(gate._writers)
        self.assertEqual(len(condition.calls), 2)
        self.assertTrue(all("timeout" in call for call in condition.calls))

    def test_inherited_controls_apply_before_mutex_for_nested_read_fork_and_join(self) -> None:
        for operation in ("nested", "fork", "join"):
            for mode in ("deadline", "cancelled"):
                with self.subTest(operation=operation, mode=mode):
                    gate = self.api.ServingGate()
                    clock, cancelled = [100.0], threading.Event()
                    options = {"deadline": 101.0} if mode == "deadline" else {"cancelled": cancelled}
                    with patch.object(self.api.time, "monotonic", side_effect=lambda: clock[0]):
                        parent = gate.operation_lease(KEY, **options)
                        lease = parent.__enter__()
                        child = lease.fork_child() if operation == "join" else None
                        if child is not None:
                            parent.__exit__(None, None, None)
                        real_condition = gate._condition
                        arguments = []
                        class RefusingCondition:
                            def acquire(self, **kwargs):
                                arguments.append(kwargs)
                                clock[0] = 102.0
                                cancelled.set()
                                return False
                            def release(self):
                                raise AssertionError("Unacquired inherited lock released")
                        gate._condition = RefusingCondition()
                        expected = self.api.ServingLeaseTimeout if mode == "deadline" else self.api.ServingLeaseCancelled
                        try:
                            with self.assertRaises(expected):
                                if operation == "fork":
                                    lease.fork_child()
                                else:
                                    context = child.join() if child is not None else gate.operation_lease(KEY)
                                    with context:
                                        self.fail("Inherited admission bound was ignored")
                        finally:
                            gate._condition = real_condition
                        self.assertEqual(len(arguments), 1)
                        self.assertIn("timeout", arguments[0])
                        if mode == "cancelled":
                            self.assertLessEqual(arguments[0]["timeout"], 0.05)
                        clock[0] = 100.0
                        cancelled.clear()
                        if child is not None:
                            self.assertIn(child, gate._children)
                            with child.join() as joined:
                                self.assertEqual(joined.selection_key, KEY)
                        else:
                            if operation == "fork":
                                retry = lease.fork_child()
                                retry.close()
                            else:
                                with gate.operation_lease(KEY):
                                    pass
                            parent.__exit__(None, None, None)
                        with gate.exclusive_activation():
                            self.assertFalse(gate._children)
                            self.assertFalse(gate._readers)

    def test_cancelled_unadmitted_child_can_be_closed_without_joining(self) -> None:
        cancel = threading.Event()
        with self.gate.operation_lease(KEY) as lease:
            child = lease.fork_child()
        cancel.set()
        with self.assertRaises(self.api.ServingLeaseCancelled):
            with child.join(cancelled=cancel):
                self.fail("Cancelled child entered")
        self.assertIn(child, self.gate._children)
        child.close()
        self.assert_idle()

    def test_retired_authenticity_registry_does_not_retain_closed_children(self) -> None:
        references = []
        with self.gate.operation_lease(KEY) as lease:
            for _ in range(32):
                child = lease.fork_child()
                references.append(weakref.ref(child))
                child.close()
        del child
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(len(self.gate._retired_children), 0)
        self.assert_idle()

    def test_child_join_failure_and_body_baseexception_release_exactly_once(self) -> None:
        sentinel = Cancelled("synthetic interruption")
        with self.gate.operation_lease(KEY) as lease:
            child = lease.fork_child()
        with self.assertRaises(Cancelled) as caught:
            with child.join():
                raise sentinel
        self.assertIs(caught.exception, sentinel)
        self.assertFalse(self.gate._children)
        for context in (self.gate.operation_lease(KEY), self.gate.exclusive_activation()):
            with self.assertRaises(Cancelled) as caught:
                with context:
                    raise sentinel
            self.assertIs(caught.exception, sentinel)
        self.assert_idle()

    def test_wait_baseexception_unregisters_writer_and_notifies_readers(self) -> None:
        sentinel = Cancelled("synthetic wait interruption")
        outcomes = []
        with self.gate.operation_lease(KEY):
            with patch.object(self.gate._condition, "wait", side_effect=sentinel):
                def waiter():
                    try:
                        with self.gate.exclusive_activation():
                            self.fail("Writer should not enter")
                    except Cancelled as error:
                        outcomes.append(error)
                done = self.start("interrupted-writer", waiter)
                self.reached(done)
            self.assertEqual(outcomes, [sentinel])
            self.assertFalse(self.gate._waiting_writers)
        self.assert_idle()

    def test_foreign_thread_cannot_fork_or_inspect_parent_lease(self) -> None:
        outcomes = []
        with self.gate.operation_lease(KEY) as lease:
            def wrong_thread():
                for call in (lease.fork_child, lambda: lease.selection_key):
                    try:
                        call()
                    except self.api.ServingLeaseError:
                        outcomes.append("refused")
            done = self.start("foreign-reader", wrong_thread)
            self.reached(done)
        self.assertEqual(outcomes, ["refused", "refused"])
        self.assert_idle()

    def test_forged_foreign_stale_and_process_inherited_tokens_refuse(self) -> None:
        for kind in (self.api.OperationLease, self.api.ChildReservation):
            with self.assertRaises(self.api.ServingLeaseError):
                kind()
        forged = object.__new__(self.api.ChildReservation)
        with self.assertRaises(self.api.ServingLeaseError):
            forged.close()
        object.__setattr__(forged, "_gate", self.gate)
        object.__setattr__(forged, "_closed", True)
        with self.assertRaises(self.api.ServingLeaseError):
            forged.close()
        with self.gate.operation_lease(KEY) as lease:
            child = lease.fork_child()
        self.assertNotIn(KEY[0], repr(child))
        foreign = self.api.ServingGate()
        with self.assertRaises(self.api.ServingLeaseError):
            foreign._close_child(child)
        self.gate._condition.acquire()
        try:
            with patch.object(self.api.os, "getpid", return_value=self.gate._pid + 1):
                with self.assertRaises(self.api.ServingLeaseError):
                    child.close()
                with self.assertRaises(self.api.ServingLeaseError):
                    with self.gate.operation_lease(KEY):
                        self.fail("Inherited explicit gate admitted")
        finally:
            self.gate._condition.release()
        child.close()
        with self.assertRaises(self.api.ServingLeaseError):
            lease.fork_child()
        self.assert_idle()

    def test_module_gate_reset_rejects_inherited_tokens_and_allows_fresh_operations(self) -> None:
        old = self.api._DEFAULT_GATE
        with self.api.operation_lease(KEY) as lease:
            child = lease.fork_child()
        with patch.object(self.api.os, "getpid", return_value=old._pid + 1):
            self.api._after_fork()
            self.assertIsNot(self.api._DEFAULT_GATE, old)
            with self.api.operation_lease(KEY) as fresh:
                self.assertEqual(fresh.selection_key, KEY)
            with self.assertRaises(self.api.ServingLeaseError):
                with child.join():
                    self.fail("Inherited reservation joined")
        child.close()

    def test_keys_are_inert_bounded_exact_types_and_never_disclosed(self) -> None:
        class Hostile(str):
            def __eq__(self, other):
                raise AssertionError("User comparator invoked")
            def __len__(self):
                raise AssertionError("User length invoked")
        for key in (["synthetic"], (), ("",), ("x" * 513,), ("x",) * 9, (Hostile("secret-key"),), (True,)):
            with self.assertRaises(self.api.ServingLeaseError) as caught:
                with self.gate.operation_lease(key):
                    self.fail("Invalid key admitted")
            self.assertNotIn("secret-key", str(caught.exception))
        with self.gate.operation_lease(KEY):
            with self.assertRaises(self.api.ServingLeaseError) as caught:
                with self.gate.operation_lease(("different-private-key",)):
                    self.fail("Selection mixed within operation")
            self.assertNotIn("private", str(caught.exception))
        for value in (True, math.inf, math.nan, "1", 10**1000, -1):
            with self.assertRaises(self.api.ServingLeaseError):
                with self.gate.operation_lease(KEY, timeout=value):
                    self.fail("Invalid timeout admitted")
        self.assert_idle()


if __name__ == "__main__":
    unittest.main()
