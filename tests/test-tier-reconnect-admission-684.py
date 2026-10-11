"""Poisoned-handle public admission using production AST and synthetic owners."""
from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
import gc
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import types
from typing import NamedTuple
import unittest
import uuid
import weakref
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    commit = os.environ.get("TRUEMEMORY_RECONNECT_SOURCE_COMMIT")
    if commit:
        return subprocess.run(
            ["git", "show", f"{commit}:truememory/{path}"], cwd=ROOT,
            check=True, capture_output=True, text=True,
        ).stdout
    return (ROOT / "truememory" / path).read_text(encoding="utf-8")


def definitions(path: str, names: set[str], namespace: dict[str, object]) -> types.ModuleType:
    tree = ast.parse(source(path))
    selected = [node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != names:
        raise AssertionError("Missing production definitions")
    module = types.ModuleType("synthetic_reconnect_684")
    module.__dict__.update(namespace)
    selected.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    loaded = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(loaded, path, "exec"), module.__dict__)
    return module


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now


class Signal:
    def __init__(self, action: Callable[[float], None] | None = None) -> None:
        self.set_value = False
        self.action = action
        self.waits: list[float] = []

    def is_set(self) -> bool:
        return self.set_value

    def set(self) -> None:
        self.set_value = True

    def wait(self, timeout: float) -> bool:
        self.waits.append(timeout)
        if self.action is not None:
            self.action(timeout)
        return self.set_value


class Writer:
    def __init__(self) -> None:
        self.owned = False

    def owned_by_current_thread(self) -> bool:
        return self.owned

    def locked(self) -> bool:
        return self.owned


class SyntheticPath:
    def __init__(self) -> None:
        self.mkdir_calls = 0
        self.chmod_calls = 0

    def __str__(self) -> str:
        return "/synthetic/reconnect-684.db"

    def resolve(self) -> SyntheticPath:
        return self

    @property
    def parent(self) -> SyntheticPath:
        return self

    def mkdir(self, *, parents: bool, exist_ok: bool) -> None:
        self.mkdir_calls += 1

    def chmod(self, mode: int) -> None:
        self.chmod_calls += 1


class Handle:
    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason
        self._in_transaction = False
        self.probes = 0
        self.closes = 0

    def execute(self, sql: str) -> None:
        if sql != "PRAGMA schema_version":
            raise AssertionError("Unexpected synthetic SQL")
        self.probes += 1
        if self.reason == "closed":
            raise sqlite3.ProgrammingError("Cannot operate on a closed database.")
        if self.reason == "io_error":
            raise sqlite3.OperationalError("disk I/O error")
        if self.reason == "other_error":
            raise sqlite3.OperationalError("synthetic unrelated error")

    def close(self) -> None:
        self.reason = "closed"
        self.closes += 1

    @property
    def in_transaction(self) -> bool:
        if self.reason == "closed":
            raise sqlite3.ProgrammingError("Cannot operate on a closed database.")
        return self._in_transaction

    @in_transaction.setter
    def in_transaction(self, borrowed: bool) -> None:
        self._in_transaction = borrowed


class SyntheticEngine:
    pass


class Fixture:
    def __init__(self, reason: str | None = "closed", *, actual_lock: bool = False) -> None:
        self.clock = Clock()
        self.path = SyntheticPath()
        self.scope_depth = 0
        self.wait_scopes: list[int] = []
        self.deadlines: list[float] = []
        self.local = types.SimpleNamespace(operation=None)
        self.admitted = False
        self.admission_error = False
        self.factory_calls = 0
        self.acquisitions = 0
        self.open_calls = 0
        self.previous_modules: dict[str, types.ModuleType | None] = {}
        self.registry = threading.Lock()
        self.held_paths: dict[object, int] = {}
        self.coordinators: dict[object, object] = {}
        self.handle = Handle(reason)
        self.replacement = Handle()
        self.reopen_calls: list[tuple[object, object]] = []
        self.reopened: tuple[Handle, bool] | None = None
        self.gate_error = type("ServingLeaseError", (RuntimeError,), {})
        self.runtime_error = type("TierRuntimeError", (RuntimeError,), {})
        self.engine = SyntheticEngine()
        self.engine.__dict__.update(
            conn=self.handle, db_path=self.path, _write_lock=Writer(), _init_lock=object(),
            _reconnect_receipt=None, ready=True, _runtime_initialized=True, _has_vectors=True,
            _runtime_vector_generation="synthetic-generation", _runtime_vector_connection=self.handle,
        )

        @contextmanager
        def connection_read(*args: object, **kwargs: object) -> Iterator[None]:
            self.scope_depth += 1
            try:
                yield
            finally:
                self.scope_depth -= 1

        def factory(path: object, create: object) -> Handle:
            self.factory_calls += 1
            return self.replacement

        def file_lock(path: object) -> int:
            self.acquisitions += 1
            return 901

        def release(fd: int) -> None:
            for path, claimed in tuple(self.held_paths.items()):
                if claimed == fd:
                    del self.held_paths[path]

        self.maintenance = definitions("maintenance.py", {
            "MaintenanceBusyError", "MaintenanceOwnership", "_AutomaticAdmission",
            "_automatic_admission_locked", "_acquire_owner", "_bind_owner", "maintenance_owner",
            "wait_for_automatic_owner",
        }, {
            "os": os, "weakref": weakref, "uuid": uuid, "NamedTuple": NamedTuple,
            "contextmanager": contextmanager, "time": self.clock,
            "threading": types.SimpleNamespace(Event=Signal, TIMEOUT_MAX=threading.TIMEOUT_MAX),
            "Path": lambda value: value, "_registry_lock": self.registry,
            "_coordinators": self.coordinators, "_held_paths": self.held_paths,
            "_held_fds": set(), "_thread_owners": threading.local(),
            "try_file_lock": file_lock, "_release_owner": release,
            "canonical_database_path": lambda value: value,
        })
        self.maintenance._active_initialization_receipt = lambda path: nullcontext(None)
        self.runtime = definitions("tier_switch/runtime.py", {"open_engine_for_operation"}, {
            "time": self.clock, "ConnectionWriteLock": Writer, "_local": self.local,
            "TierRuntimeError": self.runtime_error,
            "serving": types.SimpleNamespace(current_thread_admitted=self.currently_admitted,
                                               ServingLeaseError=self.gate_error),
        })
        self.runtime._connection_read = connection_read
        self.runtime._open_serving_connection_at_path = factory
        if actual_lock:
            locks = definitions("tier_switch/runtime.py", {
                "ConnectionWriteLock", "_connection_read", "_check", "_remaining",
            }, {
                "threading": threading, "sqlite3": sqlite3, "time": self.clock,
                "contextmanager": contextmanager, "TierRuntimeError": self.runtime_error,
                "serving": self.runtime.serving,
            })
            self.runtime.ConnectionWriteLock = locks.ConnectionWriteLock
            self.runtime._connection_read = locks._connection_read
            self.engine._write_lock = locks.ConnectionWriteLock(self.engine)
            self.engine._init_lock = threading.Lock()
        self.storage = types.ModuleType("truememory.storage")
        storage_tree = ast.parse(source("storage.py"))
        self.storage.DEFAULT_BUSY_TIMEOUT_MS = next(
            ast.literal_eval(node.value) for node in storage_tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "DEFAULT_BUSY_TIMEOUT_MS"
                    for target in node.targets)
        )
        engine_module = definitions("engine.py", {"_open_connection_handle"}, {
            "sqlite3": sqlite3, "create_db": lambda path: self.replacement, "_HAS_HYBRID": False,
        })
        actual_open = engine_module._open_connection_handle

        def opening() -> None:
            self.open_calls += 1
            actual_open(self.engine)

        def reopen(path: object, *, receipt: object) -> tuple[Handle, bool] | None:
            self.reopen_calls.append((path, receipt))
            return self.reopened

        self.engine._open_connection_handle = opening
        self.engine._reopen_initialized_connection = reopen
        actual_wait = self.maintenance.wait_for_automatic_owner

        def waiting(error: BaseException, *, deadline: float) -> None:
            self.wait_scopes.append(self.scope_depth)
            self.deadlines.append(deadline)
            actual_wait(error, deadline=deadline)

        self.maintenance.wait_for_automatic_owner = waiting
        for name, module in {
            "truememory": types.ModuleType("truememory"),
            "truememory.tier_switch": types.ModuleType("truememory.tier_switch"),
            "truememory.tier_switch.runtime": self.runtime,
            "truememory.maintenance": self.maintenance, "truememory.storage": self.storage,
        }.items():
            self.previous_modules[name] = sys.modules.get(name)
            sys.modules[name] = module

    def close(self) -> None:
        for name, previous in self.previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def currently_admitted(self) -> bool:
        if self.admission_error:
            raise self.gate_error("synthetic gate unavailable")
        return self.admitted

    def automatic_owner(self, *, release_after: float = 1.0) -> object:
        coordinator = types.SimpleNamespace(
            _mutex=threading.Lock(), _pid=os.getpid(), _active=True, _cancel=Signal(),
        )
        # SimpleNamespace has no weak-reference slot; the callable preserves identity.
        released = Signal()
        origin = self.maintenance._AutomaticAdmission(
            lambda: coordinator, self.path, os.getpid(), 71, "synthetic-launch", released, Signal(),
        )
        coordinator._admission_owner = origin
        self.coordinators[self.path] = coordinator
        self.held_paths[self.path] = 71

        def release(timeout: float) -> None:
            self.clock.now += min(timeout, release_after)
            if timeout >= release_after:
                self.held_paths.clear()
                coordinator._active = False
                released.set()

        released.action = release
        return origin


class TestReconnectAdmission(unittest.TestCase):
    def fixture(self, reason: str | None = "closed", *, actual_lock: bool = False) -> Fixture:
        fixture = Fixture(reason, actual_lock=actual_lock)
        self.addCleanup(fixture.close)
        return fixture

    def test_closed_handle_retries_exact_automatic_owner_outside_engine_locks(self) -> None:
        self.assertTrue(gc.isenabled())
        fixture = self.fixture()
        origin = fixture.automatic_owner()
        fixture.runtime.open_engine_for_operation(fixture.engine)
        self.assertEqual(origin.released.waits, [10.0])
        self.assertEqual(fixture.wait_scopes, [0])
        self.assertEqual(fixture.deadlines, [110.0])
        self.assertIs(fixture.engine.conn, fixture.replacement)
        self.assertEqual(fixture.factory_calls, 1)
        self.assertEqual(fixture.open_calls, 2)
        self.assertFalse(fixture.engine.ready)
        self.assertFalse(fixture.engine._runtime_initialized)

    def test_io_error_handle_retries_same_strict_automatic_owner(self) -> None:
        self.assertTrue(gc.isenabled())
        fixture = self.fixture("io_error")
        origin = fixture.automatic_owner()
        fixture.runtime.open_engine_for_operation(fixture.engine)
        self.assertEqual(origin.released.waits, [10.0])
        self.assertEqual(fixture.handle.closes, 1)
        self.assertEqual(fixture.wait_scopes, [0])
        self.assertIs(fixture.engine.conn, fixture.replacement)

    def test_healthy_handle_never_claims_maintenance_or_waits(self) -> None:
        fixture = self.fixture(None)
        origin = fixture.automatic_owner()
        fixture.runtime.open_engine_for_operation(fixture.engine)
        self.assertIs(fixture.engine.conn, fixture.handle)
        self.assertTrue(fixture.engine.ready)
        self.assertEqual(fixture.handle.probes, 1)
        self.assertEqual(fixture.open_calls, 1)
        self.assertEqual(fixture.factory_calls, 0)
        self.assertEqual(fixture.acquisitions, 0)
        self.assertEqual(origin.released.waits, [])
        self.assertEqual(fixture.deadlines, [])

    def test_private_opener_attests_exact_failed_handle_without_waiting(self) -> None:
        fixture = self.fixture()
        origin = fixture.automatic_owner()
        with self.assertRaises(fixture.maintenance.MaintenanceBusyError) as raised:
            fixture.engine._open_connection_handle()
        self.assertIs(raised.exception._automatic_owner, origin)
        self.assertIs(getattr(raised.exception, "_reconnect_engine", None), fixture.engine)
        self.assertIs(getattr(raised.exception, "_reconnect_handle", None), fixture.handle)
        self.assertIs(fixture.engine.conn, fixture.handle)
        self.assertEqual(fixture.wait_scopes, [])
        self.assertEqual(origin.released.waits, [])
        self.assertEqual(fixture.factory_calls, 0)

    def test_missing_handle_keeps_existing_exact_origin_retry(self) -> None:
        fixture = self.fixture()
        fixture.engine.conn = None
        origin = fixture.automatic_owner()
        fixture.runtime.open_engine_for_operation(fixture.engine)
        self.assertEqual(origin.released.waits, [10.0])
        self.assertEqual(fixture.wait_scopes, [0])
        self.assertIs(fixture.engine.conn, fixture.replacement)
        self.assertEqual(fixture.open_calls, 2)

    def test_manual_receiptless_foreign_and_cancelled_owners_never_wait(self) -> None:
        for kind in ("manual", "receiptless", "foreign_pid", "cancelled", "foreign_descriptor"):
            with self.subTest(kind=kind):
                fixture = self.fixture()
                origin = fixture.automatic_owner()
                coordinator = fixture.coordinators[fixture.path]
                if kind == "manual":
                    fixture.coordinators.clear()
                elif kind == "receiptless":
                    coordinator._admission_owner = None
                elif kind == "foreign_pid":
                    coordinator._pid = -1
                elif kind == "cancelled":
                    origin.cancelled.set()
                else:
                    fixture.held_paths[fixture.path] = 991
                with self.assertRaises(fixture.maintenance.MaintenanceBusyError):
                    fixture.runtime.open_engine_for_operation(fixture.engine)
                self.assertEqual(origin.released.waits, [])
                self.assertIs(fixture.engine.conn, fixture.handle)
                self.assertEqual(fixture.factory_calls, 0)
                self.assertEqual(fixture.open_calls, 1)

    def test_serving_runtime_raw_or_unqueryable_scopes_refuse_before_wait(self) -> None:
        for kind in ("runtime", "raw_or_exclusive", "gate_error", "writer"):
            with self.subTest(kind=kind):
                fixture = self.fixture()
                origin = fixture.automatic_owner()
                if kind == "runtime":
                    fixture.local.operation = object()
                elif kind == "raw_or_exclusive":
                    fixture.admitted = True
                elif kind == "gate_error":
                    fixture.admission_error = True
                else:
                    fixture.engine._write_lock.owned = True
                with self.assertRaises(fixture.maintenance.MaintenanceBusyError):
                    fixture.runtime.open_engine_for_operation(fixture.engine)
                self.assertEqual(origin.released.waits, [])
                self.assertEqual(fixture.wait_scopes, [])
                self.assertEqual(fixture.factory_calls, 0)
                self.assertIs(fixture.engine.conn, fixture.handle)

    def test_fresh_bound_writer_is_refused_before_opening(self) -> None:
        fixture = self.fixture()
        fixture.engine.conn = None
        fixture.engine._write_lock.owned = True
        with self.assertRaises(fixture.maintenance.MaintenanceBusyError):
            fixture.runtime.open_engine_for_operation(fixture.engine)
        self.assertEqual(fixture.open_calls, 0)
        self.assertEqual(fixture.wait_scopes, [])

    def test_healthy_or_borrowed_unmarked_busy_error_never_waits(self) -> None:
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                fixture = self.fixture(None)
                fixture.handle.in_transaction = borrowed
                origin = fixture.automatic_owner()
                error = fixture.maintenance.MaintenanceBusyError("synthetic unmarked failure")
                error._automatic_owner = origin

                def reject() -> None:
                    raise error

                fixture.engine._open_connection_handle = reject
                with self.assertRaises(fixture.maintenance.MaintenanceBusyError) as raised:
                    fixture.runtime.open_engine_for_operation(fixture.engine)
                self.assertIs(raised.exception, error)
                self.assertEqual(fixture.wait_scopes, [])
                self.assertEqual(origin.released.waits, [])
                self.assertIs(fixture.engine.conn, fixture.handle)
                self.assertEqual(fixture.handle.in_transaction, borrowed)

    def test_validated_receipt_reopen_preserves_existing_ownership_free_path(self) -> None:
        for controlled in (False, True):
            with self.subTest(controlled=controlled):
                fixture = self.fixture()
                receipt = object()
                fixture.engine._reconnect_receipt = receipt
                fixture.reopened = (fixture.replacement, controlled)
                origin = fixture.automatic_owner()
                fixture.runtime.open_engine_for_operation(fixture.engine)
                self.assertEqual(fixture.reopen_calls, [(fixture.path, receipt)])
                self.assertIs(fixture.engine.conn, fixture.replacement)
                self.assertEqual(origin.released.waits, [])
                self.assertEqual(fixture.wait_scopes, [])
                self.assertEqual(fixture.factory_calls, 0)
                self.assertEqual(fixture.engine.ready, not controlled)
                self.assertIs(fixture.engine._reconnect_receipt, None if controlled else receipt)

    def test_rejected_receipt_uses_owned_fallback_and_resets_readiness(self) -> None:
        fixture = self.fixture()
        receipt = object()
        fixture.engine._reconnect_receipt = receipt
        origin = fixture.automatic_owner()
        fixture.runtime.open_engine_for_operation(fixture.engine)
        self.assertEqual(fixture.reopen_calls, [(fixture.path, receipt), (fixture.path, receipt)])
        self.assertEqual(origin.released.waits, [10.0])
        self.assertIs(fixture.engine.conn, fixture.replacement)
        self.assertIsNone(fixture.engine._reconnect_receipt)
        self.assertIsNone(fixture.engine._runtime_vector_generation)
        self.assertIsNone(fixture.engine._runtime_vector_connection)
        self.assertFalse(fixture.engine._has_vectors)
        self.assertFalse(fixture.engine.ready)

    def test_successive_verified_owners_share_one_ten_second_budget(self) -> None:
        fixture = self.fixture()
        first = fixture.automatic_owner(release_after=3.0)
        release = first.released.action
        successors: list[object] = []

        def next_owner(timeout: float) -> None:
            release(timeout)
            successors.append(fixture.automatic_owner(release_after=2.0))

        first.released.action = next_owner
        fixture.runtime.open_engine_for_operation(fixture.engine)
        self.assertEqual(fixture.deadlines, [110.0, 110.0])
        self.assertEqual(first.released.waits, [10.0])
        self.assertEqual(successors[0].released.waits, [7.0])
        self.assertEqual(fixture.clock.now, 105.0)
        self.assertEqual(fixture.wait_scopes, [0, 0])
        self.assertEqual(fixture.open_calls, 3)
        self.assertEqual(fixture.factory_calls, 1)

    def test_timed_out_owner_does_not_reset_budget_or_poisoned_state(self) -> None:
        fixture = self.fixture()
        origin = fixture.automatic_owner(release_after=20.0)
        with self.assertRaises(fixture.maintenance.MaintenanceBusyError):
            fixture.runtime.open_engine_for_operation(fixture.engine)
        self.assertEqual(origin.released.waits, [10.0])
        self.assertEqual(fixture.clock.now, 110.0)
        self.assertEqual(fixture.deadlines, [110.0])
        self.assertEqual(fixture.open_calls, 1)
        self.assertEqual(fixture.factory_calls, 0)
        self.assertIs(fixture.engine.conn, fixture.handle)
        self.assertTrue(fixture.engine.ready)

    def test_initial_attempt_consumes_the_same_budget(self) -> None:
        fixture = self.fixture()
        origin = fixture.automatic_owner()
        opening = fixture.engine._open_connection_handle

        def delayed() -> None:
            fixture.clock.now += 3.0
            opening()

        fixture.engine._open_connection_handle = delayed
        fixture.runtime.open_engine_for_operation(fixture.engine)
        self.assertEqual(origin.released.waits, [7.0])
        self.assertEqual(fixture.deadlines, [110.0])

    def test_swapped_handle_before_wait_cannot_reuse_cached_failure(self) -> None:
        for replacement in (None, Handle()):
            with self.subTest(replacement=replacement):
                fixture = self.fixture()
                origin = fixture.automatic_owner()

                def swap() -> bool:
                    fixture.engine.conn = replacement
                    return False

                fixture.runtime.serving.current_thread_admitted = swap
                with self.assertRaises(fixture.maintenance.MaintenanceBusyError):
                    fixture.runtime.open_engine_for_operation(fixture.engine)
                self.assertEqual(origin.released.waits, [])
                self.assertEqual(fixture.open_calls, 1)
                self.assertEqual(fixture.factory_calls, 0)
                self.assertIs(fixture.engine.conn, replacement)

    def test_swapped_handle_during_wait_is_rejected_before_retry(self) -> None:
        for replacement in (None, Handle()):
            with self.subTest(replacement=replacement):
                fixture = self.fixture()
                origin = fixture.automatic_owner()
                release = origin.released.action

                def swap(timeout: float) -> None:
                    release(timeout)
                    fixture.engine.conn = replacement

                origin.released.action = swap
                with self.assertRaises(fixture.maintenance.MaintenanceBusyError):
                    fixture.runtime.open_engine_for_operation(fixture.engine)
                self.assertEqual(origin.released.waits, [10.0])
                self.assertEqual(fixture.open_calls, 1)
                self.assertEqual(fixture.factory_calls, 0)
                self.assertIs(fixture.engine.conn, replacement)

    def test_wrong_engine_or_wrong_handle_attestation_never_grants_wait(self) -> None:
        for kind in ("engine", "handle"):
            with self.subTest(kind=kind):
                fixture = self.fixture()
                origin = fixture.automatic_owner()
                error = fixture.maintenance.MaintenanceBusyError("synthetic stale failure")
                error._automatic_owner = origin
                error._reconnect_engine = fixture.engine if kind == "handle" else object()
                error._reconnect_handle = fixture.handle if kind == "engine" else Handle()

                def reject() -> None:
                    raise error

                fixture.engine._open_connection_handle = reject
                with self.assertRaises(fixture.maintenance.MaintenanceBusyError) as raised:
                    fixture.runtime.open_engine_for_operation(fixture.engine)
                self.assertIs(raised.exception, error)
                self.assertEqual(origin.released.waits, [])
                self.assertEqual(fixture.wait_scopes, [])

    def test_receiptless_successor_cannot_inherit_previous_origin(self) -> None:
        fixture = self.fixture()
        origin = fixture.automatic_owner()
        release = origin.released.action

        def receiptless(timeout: float) -> None:
            release(timeout)
            fixture.held_paths[fixture.path] = 991
            fixture.coordinators.clear()

        origin.released.action = receiptless
        with self.assertRaises(fixture.maintenance.MaintenanceBusyError) as raised:
            fixture.runtime.open_engine_for_operation(fixture.engine)
        self.assertIsNone(raised.exception._automatic_owner)
        self.assertEqual(origin.released.waits, [10.0])
        self.assertEqual(fixture.deadlines, [110.0, 110.0])
        self.assertEqual(fixture.open_calls, 2)
        self.assertEqual(fixture.factory_calls, 0)

    def test_unrelated_sql_failure_and_borrowed_io_error_are_not_retried(self) -> None:
        for reason, borrowed in (("other_error", False), ("io_error", True)):
            with self.subTest(reason=reason, borrowed=borrowed):
                fixture = self.fixture(reason)
                fixture.handle.in_transaction = borrowed
                origin = fixture.automatic_owner()
                with self.assertRaises(sqlite3.OperationalError):
                    fixture.runtime.open_engine_for_operation(fixture.engine)
                self.assertEqual(origin.released.waits, [])
                self.assertEqual(fixture.wait_scopes, [])
                self.assertEqual(fixture.factory_calls, 0)
                self.assertEqual(fixture.open_calls, 1)

    def test_cancelled_origin_during_wait_never_retries(self) -> None:
        fixture = self.fixture()
        origin = fixture.automatic_owner()

        def cancel(timeout: float) -> None:
            fixture.clock.now += 1.0
            origin.cancelled.set()
            origin.released.set()

        origin.released.action = cancel
        with self.assertRaises(fixture.maintenance.MaintenanceBusyError):
            fixture.runtime.open_engine_for_operation(fixture.engine)
        self.assertEqual(origin.released.waits, [10.0])
        self.assertEqual(fixture.open_calls, 1)
        self.assertEqual(fixture.factory_calls, 0)
        self.assertIs(fixture.engine.conn, fixture.handle)

    def test_registry_or_coordinator_contention_does_not_block_wait_eligibility(self) -> None:
        for kind in ("registry", "coordinator"):
            with self.subTest(kind=kind):
                fixture = self.fixture()
                origin = fixture.automatic_owner()
                lock = fixture.registry if kind == "registry" else fixture.coordinators[fixture.path]._mutex
                opening = fixture.engine._open_connection_handle

                def occupied_after_failure() -> None:
                    try:
                        opening()
                    except fixture.maintenance.MaintenanceBusyError:
                        lock.acquire()
                        raise

                fixture.engine._open_connection_handle = occupied_after_failure
                try:
                    with self.assertRaises(fixture.maintenance.MaintenanceBusyError):
                        fixture.runtime.open_engine_for_operation(fixture.engine)
                    self.assertEqual(origin.released.waits, [])
                    self.assertEqual(fixture.open_calls, 1)
                    self.assertEqual(fixture.factory_calls, 0)
                finally:
                    lock.release()

    def test_expired_initial_attempt_never_enters_wait(self) -> None:
        fixture = self.fixture()
        origin = fixture.automatic_owner()
        opening = fixture.engine._open_connection_handle

        def expired() -> None:
            fixture.clock.now += 10.0
            opening()

        fixture.engine._open_connection_handle = expired
        with self.assertRaises(fixture.maintenance.MaintenanceBusyError):
            fixture.runtime.open_engine_for_operation(fixture.engine)
        self.assertEqual(origin.released.waits, [])
        self.assertEqual(fixture.wait_scopes, [])
        self.assertEqual(fixture.factory_calls, 0)
        self.assertEqual(fixture.open_calls, 1)

    def test_poisoned_current_writer_refuses_before_actual_nonreentrant_acquire(self) -> None:
        self.assertTrue(gc.isenabled())
        for reason in ("closed", "io_error"):
            with self.subTest(reason=reason):
                fixture = self.fixture(reason, actual_lock=True)
                origin = fixture.automatic_owner()
                lock = fixture.engine._write_lock
                with lock:
                    with patch.object(lock, "acquire", side_effect=AssertionError("Nonreentrant writer reacquired")) as acquired:
                        with self.assertRaises(fixture.maintenance.MaintenanceBusyError):
                            fixture.runtime.open_engine_for_operation(fixture.engine)
                        acquired.assert_not_called()
                    self.assertTrue(lock.owned_by_current_thread())
                    self.assertIs(fixture.engine.conn, fixture.handle)
                    self.assertEqual(fixture.open_calls, 0)
                    self.assertEqual(origin.released.waits, [])
                    self.assertEqual(fixture.wait_scopes, [])
                    self.assertEqual(fixture.factory_calls, 0)

    def test_healthy_unowned_actual_writer_reuses_handle_inside_nested_admission(self) -> None:
        for admitted in (False, True):
            with self.subTest(admitted=admitted):
                fixture = self.fixture(None, actual_lock=True)
                fixture.admitted = admitted
                fixture.local.operation = object() if admitted else None
                origin = fixture.automatic_owner()
                fixture.runtime.open_engine_for_operation(fixture.engine)
                self.assertIs(fixture.engine.conn, fixture.handle)
                self.assertEqual(fixture.handle.probes, 1)
                self.assertEqual(fixture.open_calls, 1)
                self.assertEqual(fixture.factory_calls, 0)
                self.assertEqual(origin.released.waits, [])
                self.assertFalse(fixture.engine._write_lock.locked())
                self.assertTrue(fixture.engine.ready)

    def test_healthy_borrowed_actual_writer_preserves_unowned_caller_transaction(self) -> None:
        for admitted in (False, True):
            with self.subTest(admitted=admitted):
                fixture = self.fixture(None, actual_lock=True)
                fixture.handle.in_transaction = True
                fixture.admitted = admitted
                fixture.local.operation = object() if admitted else None
                origin = fixture.automatic_owner()
                fixture.runtime.open_engine_for_operation(fixture.engine)
                self.assertTrue(fixture.handle.in_transaction)
                self.assertIs(fixture.engine.conn, fixture.handle)
                self.assertEqual(fixture.handle.probes, 1)
                self.assertEqual(fixture.open_calls, 1)
                self.assertEqual(fixture.factory_calls, 0)
                self.assertEqual(origin.released.waits, [])
                self.assertFalse(fixture.engine._write_lock.locked())

    def test_current_actual_writer_refuses_before_healthy_or_borrowed_opening(self) -> None:
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                fixture = self.fixture(None, actual_lock=True)
                fixture.handle.in_transaction = borrowed
                lock = fixture.engine._write_lock
                with lock:
                    with patch.object(lock, "acquire", side_effect=AssertionError("Owned writer reacquired")) as acquired:
                        with self.assertRaises(fixture.maintenance.MaintenanceBusyError):
                            fixture.runtime.open_engine_for_operation(fixture.engine)
                        acquired.assert_not_called()
                    self.assertTrue(lock.owned_by_current_thread())
                    self.assertEqual(fixture.handle.in_transaction, borrowed)
                    self.assertEqual(fixture.handle.probes, 0)
                    self.assertEqual(fixture.open_calls, 0)
                    self.assertEqual(fixture.factory_calls, 0)
                    self.assertEqual(fixture.wait_scopes, [])


if __name__ == "__main__":
    unittest.main()
