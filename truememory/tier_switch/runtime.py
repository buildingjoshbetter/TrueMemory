"""Process-local projection and operation snapshots of durable tier selection.

Database selection is authoritative. This module never writes configuration,
selection metadata or vector rows, and stores no per-database model cache.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from truememory.tier_switch import serving
from truememory.tier_switch.activation import TierSelection, _guard_storage, _state, read_activation_state

if TYPE_CHECKING:
    from truememory.tier_switch.activation import LegacyTierPolicy

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

    def owned_by_current_thread(self) -> bool:
        owner = self._owner
        return owner is not None and owner[0] == threading.get_ident()

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


def _read_policy(conn: sqlite3.Connection) -> LegacyTierPolicy | None:
    if not conn.execute("SELECT 1 FROM main.sqlite_master WHERE name='metadata' COLLATE NOCASE").fetchone():
        return None
    if not conn.execute("SELECT 1 FROM main.metadata WHERE key='tier_activation_v1'").fetchone():
        return None
    if conn.in_transaction:
        _guard_storage(conn)
        state = _state(conn)
    else:
        state = read_activation_state(conn)
    return getattr(state, "legacy_policy", None)


def policy_key(policy: LegacyTierPolicy, reranker_id: str | None = None) -> tuple[str, ...]:
    from truememory.tier_switch.activation import LegacyTierPolicy
    if type(policy) is not LegacyTierPolicy:
        raise TierRuntimeError("Runtime policy requires an immutable journal result")
    effective = policy.reranker_id if reranker_id is None else _reranker_id(reranker_id)
    return ("policy", policy.generation, policy.target.model_id, str(policy.target.dimension),
            policy.target.tier, effective, *policy.tables)


def _authority_key(selection: TierSelection | None, policy: LegacyTierPolicy | None) -> tuple[str, ...] | None:
    return selection_key(selection) if selection is not None else policy_key(policy) if policy is not None else None


def _policy_ready(policy: LegacyTierPolicy, effective: str) -> bool:
    vector, reranker = _modules()
    return (_acknowledged == policy_key(policy, effective) and _pid == os.getpid()
            and vector.EMBEDDING_MODEL == policy.target.model_id and vector._embedding_dim == policy.target.dimension
            and vector._frozen_embedding_target is None and vector._runtime_policy_tier == policy.target.tier
            and vector._model is not None and reranker._model is not None and reranker._model_certified
            and reranker._model_name == effective and reranker._active_tier == policy.target.tier
            and reranker._frozen_reranker_id == effective)


def apply_runtime_policy(result: TierSelection | LegacyTierPolicy, *, deadline: float | None = None,
                         cancelled: threading.Event | None = None) -> None:
    """Project a same-space policy without encode, probe, or native construction."""
    global _acknowledged, _pid
    certified = type(result) is TierSelection
    key = selection_key(result) if certified else policy_key(result)
    if current_runtime_operation() is not None:
        raise TierRuntimeError("Policy projection cannot upgrade a serving operation")
    with serving.exclusive_activation(deadline=deadline, cancelled=cancelled):
        _acknowledged = None
        vector, reranker = _modules()
        _check(deadline, cancelled)
        vector.apply_embedding_policy(result.target, certified=certified, deadline=deadline)
        _check(deadline, cancelled)
        reranker.apply_reranker_policy(result.target.tier, result.reranker_id, deadline=deadline, cancelled=cancelled)
        _check(deadline, cancelled)
        _pid = os.getpid()
        if vector._model is not None and reranker._model is not None and reranker._model_certified:
            _acknowledged = key


def _prepare_legacy_policy(policy: LegacyTierPolicy, effective: str, *, deadline: float | None,
                           cancelled: threading.Event | None) -> None:
    global _acknowledged, _pid
    with serving.exclusive_activation(deadline=deadline, cancelled=cancelled):
        if _policy_ready(policy, effective):
            return
        _acknowledged = None
        vector, reranker = _modules()
        _check(deadline, cancelled)
        vector.apply_embedding_policy(policy.target, certified=False, deadline=deadline, load=True)
        _check(deadline, cancelled)
        reranker.apply_frozen_reranker(policy.target.tier, effective, deadline=deadline, cancelled=cancelled)
        _check(deadline, cancelled)
        _pid, _acknowledged = os.getpid(), policy_key(policy, effective)


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
    policy: LegacyTierPolicy | None = None
    _connection_lock: object | None = field(default=None, repr=False, compare=False)
    _defer_maintenance: bool = field(default=False, repr=False, compare=False)

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
                policy = _read_policy(connection) if selected is None else None
                if _authority_key(selected, policy) != _authority_key(original.selection, original.policy):
                    raise TierRuntimeError("Runtime child database selection changed")
            operation = ServingOperation(original.selection, original.tables, original.tier,
                                         original.reranker_id, original.key, connection, lease, original._database,
                                         original.borrowed, original.policy,
                                         original._connection_lock if connection is original._connection else None,
                                         original._defer_maintenance)
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
    reranker_id: str | None = None, _defer_maintenance: bool = False,
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
        policy = _read_policy(conn) if selection is None else None
        borrowed = conn.in_transaction
    if selection is not None:
        effective = selection.reranker_id if reranker_id is None else reranker_id
        if not _ready(selection, effective):
            if borrowed:
                raise TierRuntimeError("Runtime reconciliation requires a clean connection")
            apply_frozen_selection(selection, deadline=deadline, cancelled=cancelled, reranker_id=effective)
        key = selection_key(selection, effective)
    elif policy is not None:
        effective = policy.reranker_id if reranker_id is None else reranker_id
        if not _policy_ready(policy, effective):
            if borrowed:
                raise TierRuntimeError("Policy reconciliation requires a clean connection")
            _prepare_legacy_policy(policy, effective, deadline=deadline, cancelled=cancelled)
        key = policy_key(policy, effective)
    else:
        vector, _ = _modules()
        if vector._frozen_embedding_target is not None or getattr(vector, "_runtime_policy_tier", None) is not None:
            raise TierRuntimeError("Legacy database requires explicit coherent runtime selection")
        if reranker_id is not None:
            _, reranker = _modules()
            if reranker._model is None or reranker._model_name != reranker_id:
                if borrowed:
                    raise TierRuntimeError("Reranker loading requires a clean connection")
                with serving.exclusive_activation(deadline=deadline, cancelled=cancelled):
                    try:
                        reranker.get_reranker(model_name=reranker_id, deadline=deadline, cancelled=cancelled)
                    except Exception as error:
                        # Legacy retrieval already reports optional reranker unavailability.
                        raise_if_serving_rejection(error)
        key = _legacy_key(reranker_id)
    with serving.operation_lease(key, deadline=deadline, cancelled=cancelled, blocking=not borrowed) as lease:
        # A competing process can commit selection while this thread waits.
        # Refuse rather than loop through unbounded native reconciliation.
        with _connection_read(conn, connection_lock, deadline, cancelled):
            observed = _read_selection(conn)
            observed_policy = _read_policy(conn) if observed is None else None
            vector, reranker = _modules()
            tables = (selection.tables if selection is not None else policy.tables if policy is not None else
                      (vector._active_vec_table(conn), vector._active_sep_table(conn)))
            database = _database_identity(conn)
        if _authority_key(observed, observed_policy) != _authority_key(selection, policy):
            raise TierRuntimeError("Database selection changed during runtime admission")
        vector, reranker = _modules()
        if selection is not None:
            if not _ready(selection, effective):
                raise TierRuntimeError("Process selection changed during runtime admission")
            tables, tier, effective = selection.tables, selection.target.tier, effective
        elif policy is not None:
            if not _policy_ready(policy, effective):
                raise TierRuntimeError("Process policy changed during runtime admission")
            tables, tier = policy.tables, policy.target.tier
        else:
            if key != _legacy_key(reranker_id):
                raise TierRuntimeError("Legacy runtime changed during admission")
            tier = vector.resolve_tier()
            effective = reranker.get_current_reranker_name() if reranker_id is None else reranker_id
        operation = ServingOperation(selection, tables, tier, effective, key, conn, lease,
                                     database, borrowed, policy, connection_lock, _defer_maintenance)
        lease.check_admission()
        with _bound(operation):
            yield operation


def require_model_load_allowed() -> None:
    operation = getattr(_local, "operation", None)
    if operation is not None:
        if operation.borrowed:
            raise TierRuntimeError("Model loading requires a clean connection")
        if operation.selection is not None or operation.policy is not None:
            raise TierRuntimeError("Controlled model must be prepared before serving admission")
        lock = operation._connection_lock
        conn = operation._connection
        if isinstance(lock, ConnectionWriteLock):
            if lock.owns_other_transaction(conn):
                return
            if not lock.owned_by_current_thread():
                acquired = lock.acquire(blocking=False)
                if acquired:
                    try:
                        if conn.in_transaction:
                            raise TierRuntimeError("Model loading requires a clean connection")
                    finally:
                        lock.release()
                    return
                # The owner may have changed between the first observation
                # and the nonblocking probe. Never use an earlier SQL flag.
                if lock.owns_other_transaction(conn):
                    return
        if conn.in_transaction:
            raise TierRuntimeError("Model loading requires a clean connection")


def database_operation(function: Callable[P, R]) -> Callable[P, R]:
    """Wrap a direct connection-first search without changing its arguments."""
    @wraps(function)
    def wrapped(conn: sqlite3.Connection, *args: object, **kwargs: object) -> R:
        with serving_operation(conn):
            return function(conn, *args, **kwargs)
    return wrapped


def open_engine_for_operation(engine: object) -> None:
    """Bound fresh public admission without waiting inside an admitted call."""
    if engine.conn is not None:
        engine._open_connection_handle()
        return
    from truememory.maintenance import MaintenanceBusyError, wait_for_automatic_owner
    from truememory.storage import DEFAULT_BUSY_TIMEOUT_MS
    deadline = time.monotonic() + DEFAULT_BUSY_TIMEOUT_MS / 1000

    def outside_admission() -> None:
        lock = engine._write_lock
        owned = (lock.owned_by_current_thread() if isinstance(lock, ConnectionWriteLock)
                 else lock.locked())
        try:
            admitted = getattr(_local, "operation", None) is not None or serving.current_thread_admitted()
        except (TierRuntimeError, serving.ServingLeaseError):
            admitted = True
        if owned or admitted:
            raise MaintenanceBusyError("Fresh public opening requires an unbound caller")

    if isinstance(engine._write_lock, ConnectionWriteLock) and engine._write_lock.owned_by_current_thread():
        raise MaintenanceBusyError("Fresh public opening requires an unbound writer")
    failure = None
    while True:
        if failure is not None:
            outside_admission()
            if time.monotonic() >= deadline:
                raise failure
        try:
            engine._open_connection_handle()
            return
        except MaintenanceBusyError as error:
            failure = error
            if engine.conn is not None:
                raise
            outside_admission()
            wait_for_automatic_owner(error, deadline=deadline)


def _legacy_initialization_is_local(conn: sqlite3.Connection, operation: ServingOperation) -> bool:
    """A nested initializer may attach to a stable pair, but cannot migrate it."""
    known = ("vec_messages", "vec_messages_sep", "vec_messages_edge",
             "vec_messages_sep_edge", "vec_messages_basepro", "vec_messages_sep_basepro",
             "vec_messages_custom", "vec_messages_sep_custom")
    if any(table not in known for table in operation.tables):
        raise TierRuntimeError("Nested initialization requires a supported captured pair")
    rows = dict(conn.execute(
        "SELECT name, CASE WHEN length(CAST(sql AS BLOB)) <= 16384 THEN sql END "
        "FROM sqlite_master WHERE type='table' AND name IN (?,?,?,?,?,?,?,?)", known).fetchall())
    if not any(table in rows for table in operation.tables) and operation.tables == known[:2]:
        return False
    for table in operation.tables:
        sql = rows.get(table)
        dimension = re.search(r"float\[(\d+)\]", sql or "")
        if (not sql or not dimension or str(int(dimension.group(1))) != operation.key[2]
                or "distance_metric=cosine" not in sql.replace(" ", "").replace("'", "").lower()
                or conn.execute("SELECT 1 FROM sqlite_master WHERE name IN (?,?)",
                                (table + "_cos_stage", table + "_cos_stage_done")).fetchone()):
            raise TierRuntimeError("Nested initialization requires completed vector migration")
    if operation.tables == known[:2] and conn.execute("SELECT 1 FROM vec_messages LIMIT 1").fetchone():
        raise TierRuntimeError("Initialize legacy vector tables before entering a serving operation")
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='metadata' AND type='table'").fetchone():
        metadata = dict(conn.execute(
            "SELECT key, CASE WHEN length(CAST(value AS BLOB)) <= 512 THEN value END "
            "FROM metadata WHERE key IN ('embed_model','embed_dim')").fetchall())
        if (("embed_model" in metadata and metadata["embed_model"] != operation.key[1])
                or ("embed_dim" in metadata and metadata["embed_dim"] != str(operation.key[2]))):
            raise TierRuntimeError("Nested initialization requires the captured embedding identity")
    return True


@contextmanager
def engine_serving_operation(
    engine: object, *, reranker_override: str | None = None,
    _suppress_maintenance: bool = False,
) -> Iterator[ServingOperation]:
    """Finish cold initialization before capturing tables for the public body."""
    open_engine_for_operation(engine)
    parent = current_operation(engine.conn)
    owns_scope = parent is None
    cold = not (getattr(engine, "_runtime_initialized", True) or getattr(engine, "ready", False))
    initialized = False
    try:
        if cold:
            with serving_operation(engine.conn, connection_lock=engine._write_lock,
                                   reranker_id=reranker_override, _defer_maintenance=owns_scope) as preparing:
                engine._initialize_connection(_suppress_maintenance=True, _preparing=owns_scope)
                initialized = True
                if parent is not None and current_operation(engine.conn) is not preparing:
                    raise TierRuntimeError("Initialization changed the captured serving operation")
        with serving_operation(engine.conn, connection_lock=engine._write_lock,
                               reranker_id=reranker_override, _defer_maintenance=owns_scope) as operation:
            yield operation
    finally:
        if owns_scope and not _suppress_maintenance and (initialized or not cold):
            if initialized:
                engine._maybe_startup_consolidate()
            else:
                engine._maybe_auto_consolidate()


def engine_handle_operation(function: Callable[P, R]) -> Callable[P, R]:
    """Pin a handle-only operation without vector initialization or scheduling."""
    @wraps(function)
    def wrapped(engine: object, *args: object, **kwargs: object) -> R:
        open_engine_for_operation(engine)
        with serving_operation(engine.conn, connection_lock=engine._write_lock):
            return function(engine, *args, **kwargs)
    return wrapped


def engine_operation(function: Callable[P, R]) -> Callable[P, R]:
    """Initialize before pinning the public call's immutable table selection."""
    @wraps(function)
    def wrapped(engine: object, *args: object, **kwargs: object) -> R:
        with engine_serving_operation(engine):
            return function(engine, *args, **kwargs)
    return wrapped


def raise_if_serving_rejection(error: BaseException) -> None:
    """Prevent identity/admission refusals from becoming availability fallbacks."""
    classes: list[type[BaseException]] = [TierRuntimeError, serving.ServingLeaseError, serving.ServingLeaseTimeout]
    names = {
        "truememory.tier_switch.writer": ("WriterSelectionChanged",),
        "truememory.maintenance": ("MaintenanceBusyError",),
        "truememory.tier_switch.activation": ("TierActivationError",),
        "truememory.rebuild_source": ("RebuildSourceChanged",),
        "truememory.vector_search": ("VectorPublicationChanged",),
        "truememory.embedding_target": ("EmbeddingTargetError",),
        "truememory.model_client": ("ProtocolMismatchError", "ModelServerBusyError", "ServingIdentityMismatchError"),
    }
    for module_name, attributes in names.items():
        module = sys.modules.get(module_name)
        if module is not None:
            for attribute in attributes:
                value = getattr(module, attribute, None)
                if isinstance(value, type) and issubclass(value, BaseException):
                    classes.append(value)
    if isinstance(error, tuple(classes)):
        raise error


def require_legacy_vector_mutation(conn: sqlite3.Connection) -> None:
    """Refuse legacy vector DDL/metadata mutation on a selected database."""
    if _read_selection(conn) is not None or _read_policy(conn) is not None:
        raise TierRuntimeError("Controlled vector storage requires certified publication")


@contextmanager
def fact_operation(memory: object) -> Iterator[ServingOperation | None]:
    """Pin gate, dedup and storage for one fact; extraction remains outside."""
    engine = getattr(memory, "_engine", None)
    if engine is None or not callable(getattr(engine, "_open_connection_handle", None)):
        # Third-party source-only Memory doubles retain their legacy contract.
        yield None
        return
    with engine_serving_operation(engine) as operation:
        yield operation


@contextmanager
def maintenance_serving_operation(
    conn: sqlite3.Connection, *, cancelled: threading.Event | None = None,
    deadline: float | None = None, connection_lock: object | None = None,
) -> Iterator[bool]:
    """Called only under maintenance ownership; controlled derived vectors defer."""
    serving._control(None, deadline, cancelled)
    _check(deadline, cancelled)
    current = current_operation(conn)
    if current is not None:
        if current.selection is not None or current.policy is not None:
            yield False
            return
    else:
        with _connection_read(conn, connection_lock, deadline, cancelled):
            controlled = _read_selection(conn) is not None or _read_policy(conn) is not None
        if controlled:
            yield False
            return
    with serving_operation(conn, cancelled=cancelled, deadline=deadline, connection_lock=connection_lock):
        yield True


@contextmanager
def destructive_ingest_operation(engine: object) -> Iterator[None]:
    """Reject controlled databases before the legacy destructive import path."""
    from pathlib import Path
    from truememory.maintenance import maintenance_owner
    path = Path(engine.db_path)
    with maintenance_owner(None if str(path) == ":memory:" else path), serving.exclusive_activation():
        conn = engine.conn
        owned = False
        try:
            if conn is None and str(path) != ":memory:" and path.exists():
                conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
                owned = True
            if conn is not None:
                if conn.in_transaction:
                    raise TierRuntimeError("Destructive ingestion requires a clean connection")
                if _read_selection(conn) is not None or conn.execute(
                    "SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='metadata' COLLATE NOCASE"
                ).fetchone() and conn.execute(
                    "SELECT 1 FROM main.metadata WHERE key='tier_activation_v1' LIMIT 1"
                ).fetchone():
                    raise TierRuntimeError("Controlled databases cannot use destructive ingestion")
        finally:
            if owned:
                conn.close()
        yield


def open_serving_connection(db_path: str | os.PathLike[str],
                            legacy_factory: Callable[[object], sqlite3.Connection]) -> sqlite3.Connection:
    """Resolve one database identity before owning its absence/schema decision."""
    from pathlib import Path
    if str(db_path) == ":memory:":
        return legacy_factory(db_path)
    return _open_serving_connection_at_path(Path(db_path).resolve(), legacy_factory)


def _open_serving_connection_at_path(
    path: os.PathLike[str], legacy_factory: Callable[[object], sqlite3.Connection],
) -> sqlite3.Connection:
    """Open a canonical path; an Engine caller already owns maintenance.

    Same-thread owner reentry does not reacquire the process/file lock. The
    caller must pass the same resolved Path used for its outer ownership.
    """
    from truememory.maintenance import maintenance_owner
    if str(path) == ":memory:":
        return legacy_factory(path)
    from truememory.storage import DEFAULT_BUSY_TIMEOUT_MS
    with maintenance_owner(path):
        if not path.exists():
            return legacy_factory(path)
        conn = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, check_same_thread=False)
        try:
            present = conn.execute(
                "SELECT 1 FROM main.sqlite_master WHERE name='metadata' COLLATE NOCASE"
            ).fetchone()
            controlled = present and conn.execute(
                "SELECT 1 FROM main.metadata WHERE key IN ('tier_selected_v1','tier_activation_v1') LIMIT 1"
            ).fetchone()
            if controlled:
                read_activation_state(conn)
                for sql in (f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}", "PRAGMA foreign_keys=ON",
                            "PRAGMA synchronous=NORMAL", "PRAGMA cache_size=-64000", "PRAGMA mmap_size=268435456"):
                    conn.execute(sql)
                return conn
        except BaseException:
            conn.close()
            raise
        conn.close()
        return legacy_factory(path)


@contextmanager
def legacy_vector_mutation(conn: sqlite3.Connection) -> Iterator[None]:
    """Hold the existing nonblocking maintenance owner across legacy mutation.

    Legacy initialization can already hold a serving read. Owner contention
    refuses immediately instead of waiting or upgrading that read lease.
    Caller SQL is neither committed nor rolled back by this admission helper.
    """
    from truememory.maintenance import connection_database_path, maintenance_owner
    with maintenance_owner(connection_database_path(conn)):
        require_legacy_vector_mutation(conn)
        yield


@contextmanager
def legacy_rebuild_operation(conn: sqlite3.Connection) -> Iterator[None]:
    """Legacy rebuilds own maintenance before serving or model work."""
    with legacy_vector_mutation(conn), serving_operation(conn):
        yield
