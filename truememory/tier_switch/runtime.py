"""Process-local projection and operation snapshots of durable tier selection.

Database selection is authoritative. This module never writes configuration,
selection metadata or vector rows, and stores no per-database model cache.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import ParamSpec, TypeVar

from truememory.tier_switch import serving
from truememory.tier_switch.activation import TierSelection, _guard_storage, _state, read_activation_state

P = ParamSpec("P")
R = TypeVar("R")


class TierRuntimeError(RuntimeError):
    """The process cannot safely serve the captured database selection."""


class ConnectionWriteLock:
    """Identify Engine-owned SQL without claiming ownership of caller SQL."""

    def __init__(self, engine: object) -> None:
        import weakref
        self._engine = weakref.ref(engine)
        self._lock = threading.Lock()
        self._owner: tuple[int, object, bool] | None = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        acquired = self._lock.acquire(blocking, timeout)
        if not acquired:
            return False
        try:
            engine = self._engine()
            conn = None if engine is None else engine.conn
            try:
                borrowed = conn is not None and conn.in_transaction
            except sqlite3.ProgrammingError as exc:
                if str(exc) != "Cannot operate on a closed database.":
                    raise
                borrowed = False
            self._owner = (threading.get_ident(), conn, borrowed)
        except BaseException:
            self._lock.release()
            raise
        return True

    def owns_other_transaction(self, conn: sqlite3.Connection) -> bool:
        owner = self._owner
        return (owner is not None and owner[0] != threading.get_ident()
                and owner[1] is conn and not owner[2])

    def release(self) -> None:
        owner = self._owner
        if owner is None or owner[0] != threading.get_ident():
            raise RuntimeError("Only the Engine writer owner can release its lock")
        self._owner = None
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self) -> ConnectionWriteLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


_local = threading.local()
_acknowledged: tuple[str, ...] | None = None
_pid = os.getpid()


def _after_fork() -> None:
    global _local, _acknowledged, _pid
    _local = threading.local()
    _acknowledged = None
    _pid = os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _reranker_id(value: str) -> str:
    if (type(value) is not str or not 1 <= len(value) <= 512
            or re.fullmatch(r"[\w][\w.\-]*(/[\w][\w.\-]*)?", value) is None):
        raise TierRuntimeError("An explicit bounded reranker identity is required")
    return value


def selection_key(selection: TierSelection, reranker_id: str | None = None) -> tuple[str, ...]:
    if type(selection) is not TierSelection:
        raise TierRuntimeError("An immutable certified selection is required")
    return (selection.generation, selection.source_epoch, selection.target.model_id,
            str(selection.target.dimension), selection.target.tier,
            selection.reranker_id if reranker_id is None else _reranker_id(reranker_id),
            *selection.tables)


def _check(deadline: float | None, cancelled: threading.Event | None) -> None:
    if cancelled is not None and cancelled.is_set():
        raise serving.ServingLeaseCancelled("Runtime admission was cancelled")
    if deadline is not None and time.monotonic() >= deadline:
        raise serving.ServingLeaseTimeout("Runtime admission deadline expired")


def _remaining(deadline: float | None) -> float | None:
    _check(deadline, None)
    return None if deadline is None else deadline - time.monotonic()


def _modules() -> tuple[object, object]:
    from truememory import reranker, vector_search
    return vector_search, reranker


def _read_selection(conn: sqlite3.Connection) -> TierSelection | None:
    # Absence preserves the legacy path without requiring the new journal's
    # registry schema or inventing a certificate for existing vectors.
    if not conn.execute("SELECT 1 FROM main.sqlite_master WHERE name='metadata' COLLATE NOCASE").fetchone():
        return None
    if not conn.execute(
        "SELECT 1 FROM main.metadata WHERE key IN ('tier_selected_v1','tier_activation_v1') LIMIT 1"
    ).fetchone():
        return None
    if conn.in_transaction:
        _guard_storage(conn)
        return _state(conn).selection
    return read_activation_state(conn).selection


def _legacy_key(reranker_id: str | None = None) -> tuple[str, ...]:
    vector, reranker = _modules()
    return ("legacy", vector.EMBEDDING_MODEL, str(vector._embedding_dim),
            vector.resolve_tier(), reranker.get_current_reranker_name() if reranker_id is None else reranker_id)


def _ready(selection: TierSelection, reranker_id: str) -> bool:
    vector, reranker = _modules()
    return (_pid == os.getpid() and _acknowledged == selection_key(selection, reranker_id)
            and vector._frozen_embedding_target == selection.target
            and vector.EMBEDDING_MODEL == selection.target.model_id
            and vector._embedding_dim == selection.target.dimension
            and vector._model is not None and reranker._model is not None
            and reranker._model_certified
            and reranker._model_name == reranker_id
            and reranker._frozen_reranker_id == reranker_id
            and reranker._active_tier == selection.target.tier)


def runtime_acknowledgement() -> tuple[str, ...] | None:
    """Process-local readiness only; a config acknowledgement is unrelated."""
    return _acknowledged if _pid == os.getpid() else None


def apply_frozen_selection(
    selection: TierSelection, *, deadline: float | None = None,
    cancelled: threading.Event | None = None, reranker_id: str | None = None,
) -> None:
    """Project outside SQL; the caller may already own exclusive activation.

    Failure after database publication leaves activation pending. No previous
    process acknowledgement survives partial projection. This function has no
    connection argument; its caller must establish the clean-SQL precondition.
    """
    global _acknowledged, _pid
    effective = selection.reranker_id if reranker_id is None else _reranker_id(reranker_id)
    key = selection_key(selection, effective)
    if getattr(_local, "operation", None) is not None:
        raise TierRuntimeError("Runtime projection cannot upgrade a serving operation")
    with serving.exclusive_activation(deadline=deadline, cancelled=cancelled):
        _check(deadline, cancelled)
        if _ready(selection, effective):
            return
        _acknowledged = None
        vector, reranker = _modules()
        _check(deadline, cancelled)
        vector.apply_frozen_embedding_target(selection.target, timeout=_remaining(deadline))
        _check(deadline, cancelled)
        reranker.apply_frozen_reranker(selection.target.tier, effective, deadline=deadline, cancelled=cancelled)
        _check(deadline, cancelled)
        _pid = os.getpid()
        _acknowledged = key


@dataclass(frozen=True, repr=False)
class ServingOperation:
    selection: TierSelection | None
    tables: tuple[str, str]
    tier: str
    reranker_id: str
    key: tuple[str, ...] = field(repr=False)
    _connection: sqlite3.Connection = field(repr=False, compare=False)
    _lease: serving.OperationLease = field(repr=False, compare=False)
    _database: tuple[str, object] = field(repr=False, compare=False)
    borrowed: bool = False

    def fork_child(self) -> RuntimeChild:
        if current_operation(self._connection) is not self:
            raise TierRuntimeError("Only the admitted operation can reserve a runtime child")
        return RuntimeChild(self, _local.lease.fork_child())


@dataclass(frozen=True, repr=False)
class RuntimeChild:
    _operation: ServingOperation
    _reservation: serving.ChildReservation

    @contextmanager
    def join(self, conn: sqlite3.Connection | None = None, *, deadline: float | None = None,
             cancelled: threading.Event | None = None) -> Iterator[ServingOperation]:
        if getattr(_local, "operation", None) is not None:
            raise TierRuntimeError("Runtime child must enter on an unbound thread")
        with self._reservation.join(deadline=deadline, cancelled=cancelled) as lease:
            original = self._operation
            connection = original._connection if conn is None else conn
            if connection is not original._connection:
                if connection.in_transaction or _database_identity(connection) != original._database:
                    raise TierRuntimeError("Runtime child requires the captured clean database")
                selected = _read_selection(connection)
                if ((selection_key(selected) if selected is not None else None)
                        != (selection_key(original.selection) if original.selection is not None else None)):
                    raise TierRuntimeError("Runtime child database selection changed")
            operation = ServingOperation(original.selection, original.tables, original.tier,
                                         original.reranker_id, original.key, connection, lease, original._database,
                                         original.borrowed)
            lease.check_admission()
            with _bound(operation):
                yield operation

    def close(self) -> None:
        self._reservation.close()


def current_operation(conn: sqlite3.Connection) -> ServingOperation | None:
    operation = getattr(_local, "operation", None)
    if operation is None:
        return None
    # A second connection must start its own explicit operation. This avoids
    # silently treating a different database or snapshot as the parent's DB.
    if operation._connection is not conn:
        raise TierRuntimeError("Serving operation belongs to a different connection")
    _local.lease.selection_key  # validates PID, lifetime and owning thread
    return operation


def _database_identity(conn: sqlite3.Connection) -> tuple[str, object]:
    from truememory.maintenance import connection_database_path
    path = connection_database_path(conn)
    return ("memory", conn) if path is None else ("file", str(path))


def current_runtime_operation() -> ServingOperation | None:
    """Read this thread's admitted public tier/model policy without SQL."""
    operation = getattr(_local, "operation", None)
    return None if operation is None else current_operation(operation._connection)


@contextmanager
def _bound(operation: ServingOperation, lease: serving.OperationLease | None = None) -> Iterator[None]:
    previous = getattr(_local, "operation", None)
    previous_lease = getattr(_local, "lease", None)
    _local.operation = operation
    _local.lease = operation._lease if lease is None else lease
    try:
        yield
    finally:
        _local.operation = previous
        _local.lease = previous_lease


@contextmanager
def _connection_read(
    conn: sqlite3.Connection | None, lock: object | None, deadline: float | None,
    cancelled: threading.Event | None, *, transaction_owner: object | None = None,
    allow_closed: bool = False,
) -> Iterator[None]:
    if lock is None:
        _check(deadline, cancelled)
        yield
        return
    acquired = False
    try:
        try:
            borrowed = conn is not None and conn.in_transaction
        except sqlite3.ProgrammingError as exc:
            if not allow_closed or str(exc) != "Cannot operate on a closed database.":
                raise
            borrowed = False
        owner = lock if transaction_owner is None else transaction_owner
        if (borrowed and not (
                isinstance(owner, ConnectionWriteLock) and owner.owns_other_transaction(conn))):
            _check(deadline, cancelled)
            acquired = lock.acquire(blocking=False)
            if not acquired:
                raise TierRuntimeError("Borrowed connection admission is busy")
        else:
            while not acquired:
                _check(deadline, cancelled)
                interval = _remaining(deadline)
                if cancelled is not None:
                    interval = 0.05 if interval is None else min(interval, 0.05)
                acquired = lock.acquire() if interval is None else lock.acquire(timeout=min(interval, threading.TIMEOUT_MAX))
        _check(deadline, cancelled)
        yield
    finally:
        if acquired:
            lock.release()


@contextmanager
def serving_operation(
    conn: sqlite3.Connection, *, deadline: float | None = None,
    cancelled: threading.Event | None = None, connection_lock: object | None = None,
    reranker_id: str | None = None,
) -> Iterator[ServingOperation]:
    serving._control(None, deadline, cancelled)
    if reranker_id is not None:
        _reranker_id(reranker_id)
    current = current_operation(conn)
    if current is not None:
        if reranker_id is not None and reranker_id != current.reranker_id:
            raise TierRuntimeError("Nested operation requested a different reranker")
        with serving.operation_lease(current.key, deadline=deadline, cancelled=cancelled) as lease:
            with _bound(current, lease):
                yield current
        return
    # Engine serializes this short snapshot with its connection writer. The
    # writer is released before gate admission or any model preparation.
    with _connection_read(conn, connection_lock, deadline, cancelled):
        selection = _read_selection(conn)
        borrowed = conn.in_transaction
    if selection is not None:
        effective = selection.reranker_id if reranker_id is None else reranker_id
        if not _ready(selection, effective):
            if borrowed:
                raise TierRuntimeError("Runtime reconciliation requires a clean connection")
            apply_frozen_selection(selection, deadline=deadline, cancelled=cancelled, reranker_id=effective)
        key = selection_key(selection, effective)
    else:
        vector, _ = _modules()
        if vector._frozen_embedding_target is not None:
            raise TierRuntimeError("Legacy database requires explicit coherent runtime selection")
        if reranker_id is not None:
            _, reranker = _modules()
            if reranker._model is None or reranker._model_name != reranker_id:
                if borrowed:
                    raise TierRuntimeError("Reranker loading requires a clean connection")
                with serving.exclusive_activation(deadline=deadline, cancelled=cancelled):
                    reranker.get_reranker(model_name=reranker_id, deadline=deadline, cancelled=cancelled)
        key = _legacy_key(reranker_id)
    with serving.operation_lease(key, deadline=deadline, cancelled=cancelled, blocking=not borrowed) as lease:
        # A competing process can commit selection while this thread waits.
        # Refuse rather than loop through unbounded native reconciliation.
        with _connection_read(conn, connection_lock, deadline, cancelled):
            observed = _read_selection(conn)
            vector, reranker = _modules()
            tables = (selection.tables if selection is not None else
                      (vector._active_vec_table(conn), vector._active_sep_table(conn)))
            database = _database_identity(conn)
        if ((selection_key(observed) if observed is not None else None)
                != (selection_key(selection) if selection is not None else None)):
            raise TierRuntimeError("Database selection changed during runtime admission")
        vector, reranker = _modules()
        if selection is not None:
            if not _ready(selection, effective):
                raise TierRuntimeError("Process selection changed during runtime admission")
            tables, tier, effective = selection.tables, selection.target.tier, effective
        else:
            if key != _legacy_key(reranker_id):
                raise TierRuntimeError("Legacy runtime changed during admission")
            tier = vector.resolve_tier()
            effective = reranker.get_current_reranker_name() if reranker_id is None else reranker_id
        operation = ServingOperation(selection, tables, tier, effective, key, conn, lease,
                                     database, borrowed)
        lease.check_admission()
        with _bound(operation):
            yield operation


def require_model_load_allowed() -> None:
    operation = getattr(_local, "operation", None)
    if operation is not None and operation._connection.in_transaction:
        raise TierRuntimeError("Model loading requires a clean connection")
    if operation is not None and operation.selection is not None:
        raise TierRuntimeError("Selected model must be prepared before serving admission")


def database_operation(function: Callable[P, R]) -> Callable[P, R]:
    """Wrap a direct connection-first search without changing its arguments."""
    @wraps(function)
    def wrapped(conn: sqlite3.Connection, *args: object, **kwargs: object) -> R:
        with serving_operation(conn):
            return function(conn, *args, **kwargs)
    return wrapped


def engine_operation(function: Callable[P, R]) -> Callable[P, R]:
    """Open plainly, then pin the whole Engine call before writer admission."""
    @wraps(function)
    def wrapped(engine: object, *args: object, **kwargs: object) -> R:
        engine._open_connection_handle()
        with serving_operation(engine.conn, connection_lock=engine._write_lock):
            return function(engine, *args, **kwargs)
    return wrapped
