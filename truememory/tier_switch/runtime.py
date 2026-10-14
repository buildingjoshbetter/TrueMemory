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


class AdmissionSelectionChanged(TierRuntimeError):
    """The selection changed while this admission waited; nothing ran yet."""


class SelectedModelsUnavailable(RuntimeError):
    """This admission runs without its tier's models; callers degrade (FTS only).

    Deliberately not a serving rejection: search steps fall back exactly as a
    legacy embedding failure does.
    """


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
# The database whose selection or policy the process-global projection serves.
# Only that database's legacy admission (its journal was retired or restored)
# may reset it; any other legacy database refuses, so a process alternating
# between a controlled and a legacy store never reloads models per call.
_projected_database: tuple[str, object] | None = None
_OWNER_POLL_SECONDS = 0.05


def _after_fork() -> None:
    global _local, _acknowledged, _pid, _projected_database
    _local = threading.local()
    _acknowledged = None
    _pid = os.getpid()
    _projected_database = None


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


def reset_legacy_runtime(tier: str, *, deadline: float | None = None,
                         cancelled: threading.Event | None = None) -> None:
    """Drop a controlled projection; legacy serving re-resolves lazily.

    Used when a database's journal is gone (rolled back or retired) while this
    process still holds its frozen target or policy tier.
    """
    global _acknowledged, _projected_database
    bounded = time.monotonic() + serving._KEY_WAIT_SECONDS if deadline is None else deadline
    with serving.exclusive_activation(deadline=bounded, cancelled=cancelled):
        _acknowledged = _projected_database = None
        vector, reranker = _modules()
        vector.set_embedding_model(tier)
        reranker.set_active_tier(tier)
        _pin_process_tier(tier)


def note_projected_database(conn: sqlite3.Connection) -> None:
    """Record which database the process projection now serves (after a switch)."""
    global _projected_database
    _projected_database = _database_identity(conn)


def _pin_process_tier(tier: str) -> None:
    """Keep a pinned process tier equal to the legacy tier this process serves.

    The MCP server pins TRUEMEMORY_EMBED_MODEL at startup and legacy
    resolution reads it first. Controlled serving never consults it, so it is
    only updated when legacy serving resumes.
    """
    if os.environ.get("TRUEMEMORY_EMBED_MODEL", "").strip():
        os.environ["TRUEMEMORY_EMBED_MODEL"] = tier


def _legacy_reset_tier(conn: sqlite3.Connection) -> str:
    """The store's configured tier, verified against its legacy pair and metadata.

    config.json is the store's shared truth; this process's env may still pin
    the tier it started with. A tier whose registry pair or embedder metadata
    disagrees would serve stale vectors, so it is refused instead.
    """
    from truememory.embedding_target import EmbeddingTarget
    from truememory.tier_switch.projection import _guard, _paths, _read_config

    refused = "Legacy database requires explicit coherent runtime selection"
    try:
        tier = _guard(_read_config(_paths(None, None)[0])).tier
        target = EmbeddingTarget.capture(tier)
        row = conn.execute("SELECT vec_table, sep_table, embedding_dim FROM main.vector_cache_registry "
                           "WHERE tier_group=?", (target.tier_group,)).fetchone()
        populated = conn.execute("SELECT 1 FROM main.messages LIMIT 1").fetchone() is not None
        metadata = dict(conn.execute("SELECT key, value FROM main.metadata WHERE key IN ('embed_model','embed_dim')"))
    except Exception as error:
        raise TierRuntimeError(refused) from error
    generic = ("vec_messages", "vec_messages_sep")
    tables = generic if row is None else (row[0], row[1])
    if tables not in (target.tables, generic) or (row is not None and row[2] not in (None, target.dimension)):
        raise TierRuntimeError(refused)
    if (populated or tier != "edge") and metadata != {"embed_model": target.model_id,
                                                      "embed_dim": str(target.dimension)}:
        raise TierRuntimeError(refused)
    return tier


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
    # False only for controlled admissions that did not (or could not) project
    # the selected models: SQL-only bodies may run, model use is refused.
    models_ready: bool = field(default=True, compare=False)

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
                                         original._defer_maintenance, original.models_ready)
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
    reranker_id: str | None = None, _defer_maintenance: bool = False, models: str = "required",
) -> Iterator[ServingOperation]:
    """Admit one public operation on a captured database selection.

    ``models`` applies to controlled databases only: "required" projects the
    selected embedding and reranker before admission, "none" never does (SQL-
    only bodies such as get/delete), and "optional" projects when possible and
    otherwise admits with models_ready=False (FTS-only add).
    """
    global _projected_database
    serving._control(None, deadline, cancelled)
    if models not in _MODEL_MODES:
        raise TierRuntimeError("Unsupported model admission mode")
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
    ready = True
    if selection is not None:
        effective = selection.reranker_id if reranker_id is None else reranker_id
        if not _ready(selection, effective):
            if borrowed and models != "none":
                # A caller transaction cannot wait for model loading; never
                # silently store or search without the vectors it could have.
                raise TierRuntimeError("Runtime reconciliation requires a clean connection")
            ready = not borrowed and models != "none" and _project(
                models, lambda: apply_frozen_selection(selection, deadline=deadline, cancelled=cancelled,
                                                       reranker_id=effective))
        key = selection_key(selection, effective)
    elif policy is not None:
        effective = policy.reranker_id if reranker_id is None else reranker_id
        if not _policy_ready(policy, effective):
            if borrowed and models != "none":
                raise TierRuntimeError("Policy reconciliation requires a clean connection")
            ready = not borrowed and models != "none" and _project(
                models, lambda: _prepare_legacy_policy(policy, effective, deadline=deadline, cancelled=cancelled))
        key = policy_key(policy, effective)
    else:
        vector, _ = _modules()
        if vector._frozen_embedding_target is not None or getattr(vector, "_runtime_policy_tier", None) is not None:
            # This process projected a selection that is gone (retired by an
            # open elsewhere, or rolled back): serve the configured legacy tier.
            if borrowed:
                raise TierRuntimeError("Legacy database requires explicit coherent runtime selection")
            with _connection_read(conn, connection_lock, deadline, cancelled):
                if _projected_database not in (None, _database_identity(conn)):
                    raise TierRuntimeError("Another database's tier selection is loaded in this process; "
                                           "serve this legacy database from a separate process")
                tier = _legacy_reset_tier(conn)
            try:
                reset_legacy_runtime(tier, deadline=deadline, cancelled=cancelled)
            except serving.ServingLeaseError as error:
                raise TierRuntimeError("Legacy database requires explicit coherent runtime selection") from error
        if reranker_id is not None:
            _, reranker = _modules()
            if reranker._model is None or reranker._model_name != reranker_id:
                if borrowed:
                    raise TierRuntimeError("Reranker loading requires a clean connection")
                # Swapping the shared reranker drains every reader; never wait
                # unboundedly behind a long one (for example a maintenance run).
                bounded = time.monotonic() + serving._KEY_WAIT_SECONDS if deadline is None else deadline
                with serving.exclusive_activation(deadline=bounded, cancelled=cancelled):
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
            raise AdmissionSelectionChanged("Database selection changed during runtime admission")
        vector, reranker = _modules()
        if selection is not None:
            if ready and not _ready(selection, effective):
                raise AdmissionSelectionChanged("Process selection changed during runtime admission")
            tables, tier, effective = selection.tables, selection.target.tier, effective
        elif policy is not None:
            if ready and not _policy_ready(policy, effective):
                raise AdmissionSelectionChanged("Process policy changed during runtime admission")
            tables, tier = policy.tables, policy.target.tier
        else:
            if key != _legacy_key(reranker_id):
                raise AdmissionSelectionChanged("Legacy runtime changed during admission")
            tier = vector.resolve_tier()
            effective = reranker.get_current_reranker_name() if reranker_id is None else reranker_id
        operation = ServingOperation(selection, tables, tier, effective, key, conn, lease,
                                     database, borrowed, policy, connection_lock, _defer_maintenance, ready)
        lease.check_admission()
        if ready and (selection is not None or policy is not None):
            _projected_database = database
        with _bound(operation):
            yield operation


_MODEL_MODES = frozenset({"required", "optional", "none"})


def _project(models: str, projection: Callable[[], None]) -> bool:
    """Run model projection; only optional admission survives a model fault."""
    try:
        projection()
        return True
    except (TierRuntimeError, serving.ServingLeaseError, serving.ServingLeaseTimeout):
        raise
    except Exception as error:
        if models == "required" or _identity_error(error):
            raise
        import logging
        logging.getLogger(__name__).warning(
            "Selected tier models are unavailable; admitting without vector work", exc_info=True)
        return False


def _identity_error(error: BaseException) -> bool:
    """A loaded model that is not the certified target is never mere unavailability."""
    classes = []
    for module_name, attribute in (("truememory.embedding_target", "EmbeddingTargetError"),
                                   ("truememory.model_client", "ServingIdentityMismatchError")):
        value = getattr(sys.modules.get(module_name), attribute, None)
        if isinstance(value, type) and issubclass(value, BaseException):
            classes.append(value)
    return isinstance(error, tuple(classes))


def require_admitted_models() -> None:
    """Refuse model use inside a controlled admission whose models did not load."""
    operation = getattr(_local, "operation", None)
    if operation is not None and not operation.models_ready:
        raise SelectedModelsUnavailable("Selected tier models are unavailable for this operation")


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
    from truememory.maintenance import MaintenanceBusyError, wait_for_automatic_owner
    from truememory.storage import DEFAULT_BUSY_TIMEOUT_MS
    deadline = time.monotonic() + DEFAULT_BUSY_TIMEOUT_MS / 1000
    if isinstance(engine._write_lock, ConnectionWriteLock) and engine._write_lock.owned_by_current_thread():
        raise MaintenanceBusyError("Fresh public opening requires an unbound writer")

    def retry_allowed(error: MaintenanceBusyError) -> bool:
        cached = getattr(error, "_reconnect_engine", None)
        if cached is None:
            return engine.conn is None
        return (cached is engine and engine.conn is not None
                and getattr(error, "_reconnect_handle", None) is engine.conn)

    failure = None
    if engine.conn is not None:
        try:
            engine._open_connection_handle()
            return
        except MaintenanceBusyError as error:
            if getattr(error, "_reconnect_engine", None) is not engine or not retry_allowed(error):
                raise
            failure = error

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

    while True:
        if failure is not None:
            outside_admission()
            if time.monotonic() >= deadline:
                raise failure
            if not retry_allowed(failure):
                raise failure
            try:
                wait_for_automatic_owner(failure, deadline=deadline)
            except MaintenanceBusyError:
                # Besides automatic owners, only a certified tier activation commit
                # (in any process) grants a bounded wait; other owners still refuse.
                from truememory.tier_switch.job import tier_activation_in_progress
                if (getattr(failure, "_automatic_owner", None) is not None
                        or str(engine.db_path) == ":memory:" or not tier_activation_in_progress(engine.db_path)):
                    raise
                time.sleep(min(_OWNER_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
            outside_admission()
            if not retry_allowed(failure):
                raise failure
        try:
            engine._open_connection_handle()
            return
        except MaintenanceBusyError as error:
            failure = error
            if not retry_allowed(error):
                raise


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
    _suppress_maintenance: bool = False, models: str = "required",
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
                                   reranker_id=reranker_override, _defer_maintenance=owns_scope,
                                   models=models) as preparing:
                engine._initialize_connection(_suppress_maintenance=True, _preparing=owns_scope)
                initialized = True
                if parent is not None and current_operation(engine.conn) is not preparing:
                    raise TierRuntimeError("Initialization changed the captured serving operation")
        with serving_operation(engine.conn, connection_lock=engine._write_lock,
                               reranker_id=reranker_override, _defer_maintenance=owns_scope,
                               models=models) as operation:
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
        return _admitted_call(engine, function, args, kwargs, models="required")
    return wrapped


def engine_source_operation(function: Callable[P, R]) -> Callable[P, R]:
    """Admit a source-only operation without selected-model readiness."""
    @wraps(function)
    def wrapped(engine: object, *args: object, **kwargs: object) -> R:
        return _admitted_call(engine, function, args, kwargs, models="none")
    return wrapped


def engine_fallback_operation(function: Callable[P, R]) -> Callable[P, R]:
    """Admit with selected models when they load, otherwise without them.

    A tier activation committing between admission and the writer rolls the
    first attempt back; a top-level call is retried once under the new selection.
    """
    @wraps(function)
    def wrapped(engine: object, *args: object, **kwargs: object) -> R:
        return _admitted_call(engine, function, args, kwargs, models="optional", retry_writer=True)
    return wrapped


def _admitted_call(engine: object, function: Callable[..., R], args: tuple, kwargs: dict, *,
                   models: str, retry_writer: bool = False) -> R:
    """Retry a top-level call once when a selection change refused it before any effect.

    Admission refusals happen before the body runs; a writer refusal rolled
    back the body's transaction. Nested calls and caller transactions refuse.
    """
    retry = current_runtime_operation() is None and not _in_caller_transaction(engine)
    started = False
    try:
        with engine_serving_operation(engine, models=models):
            started = True
            return function(engine, *args, **kwargs)
    except Exception as error:
        admission = not started and isinstance(error, AdmissionSelectionChanged)
        if not retry or not (admission or (retry_writer and _selection_changed(error))):
            raise
    with engine_serving_operation(engine, models=models):
        return function(engine, *args, **kwargs)


def _in_caller_transaction(engine: object) -> bool:
    conn = getattr(engine, "conn", None)
    try:
        return conn is not None and conn.in_transaction
    except sqlite3.ProgrammingError:
        return False


def _selection_changed(error: BaseException) -> bool:
    writer = sys.modules.get("truememory.tier_switch.writer")
    changed = getattr(writer, "WriterSelectionChanged", None)
    return changed is not None and (isinstance(error, changed) or isinstance(error.__cause__, changed))


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
                if reconcile_controlled_database(conn, path):
                    return conn
                # The journal was retired: open through the ordinary legacy path.
        except BaseException:
            conn.close()
            raise
        conn.close()
        return legacy_factory(path)


def reconcile_controlled_database(conn: sqlite3.Connection, path: os.PathLike[str]) -> bool:
    """Upgrade and validate a controlled database under the caller's maintenance.

    Runs the idempotent create_db migrations (controlled opens skip create_db),
    retires a journal whose selected pair another binary changed, and mirrors an
    unacknowledged selection into the default config left by a crash. Returns
    whether the database is still controlled.
    """
    import logging
    from pathlib import Path
    from truememory.storage import _apply_schema_migrations
    from truememory.tier_switch.activation import retire_incoherent_activation
    _apply_schema_migrations(conn, path)
    if retire_incoherent_activation(conn):
        logging.getLogger(__name__).warning(
            "Tier selection no longer matches its vector pair (changed by another release); "
            "serving the configured legacy tier")
        return False
    if _is_default_store(path):
        if reconcile_config_tier(conn):
            return False
        _mirror_unacknowledged_selection(conn, Path(path))
    return True


def _is_default_store(path: os.PathLike[str]) -> bool:
    """Only the store the shared config describes (RebuildManager's default) mirrors it."""
    from pathlib import Path
    default = Path(os.environ.get("TRUEMEMORY_DB_PATH") or os.environ.get("TRUEMEMORY_DB")
                   or Path.home() / ".truememory" / "memories.db").expanduser()
    return Path(path).resolve() == default.resolve()


def reconcile_config_tier(conn: sqlite3.Connection, *, config_path: os.PathLike[str] | None = None,
                          lock_path: os.PathLike[str] | None = None) -> bool:
    """Resolve an acknowledged journal whose same-generation config names another tier.

    An older release (or a hand edit) changed config.json without touching the
    journal. In the same embedding space the config is the newer request: the
    journal is retired (True) and legacy serving follows the config. Otherwise
    no rebuild happened, so the journal stays authoritative and the config is
    written back to it. Caller owns maintenance.
    """
    import logging
    from truememory.embedding_target import EmbeddingTarget, EmbeddingTargetError
    from truememory.tier_switch.activation import retire_same_space_activation
    from truememory.tier_switch.projection import (
        CONFIG_WRITE_LOCK, ConfigFileLock, _guard, _paths, _read_config, _write_config,
    )

    state = read_activation_state(conn)
    result = state.selection or state.legacy_policy
    if result is None or not result.config_acknowledged:
        return False
    config_path, lock_path = _paths(config_path, lock_path)
    with ConfigFileLock(lock_path, strict=True), CONFIG_WRITE_LOCK:
        config = _read_config(config_path)
        guard = _guard(config)
        if guard.generation != result.generation or guard.tier == result.target.tier:
            return False
        try:
            requested = EmbeddingTarget.capture(guard.tier)
        except (EmbeddingTargetError, ValueError):
            requested = None
        if requested is not None and requested.identity == result.target.identity:
            retired = retire_same_space_activation(conn, generation=result.generation)
            if retired:
                logging.getLogger(__name__).warning(
                    "Config selects tier %s in the same embedding space; serving it instead of %s",
                    guard.tier, result.target.tier)
            return retired
        config["tier"] = result.target.tier
        _write_config(config_path, config)
        logging.getLogger(__name__).warning(
            "Config named tier %s without a rebuild; restored the selected tier %s", guard.tier, result.target.tier)
        return False


def _mirror_unacknowledged_selection(conn: sqlite3.Connection, path: os.PathLike[str]) -> None:
    import logging

    state = read_activation_state(conn)
    result = state.selection or state.legacy_policy
    expected = getattr(state.intent, "expected_config", None)
    if result is None or result.config_acknowledged or expected is None:
        return
    from truememory.tier_switch.activation import (
        LegacyTierPolicy, TierActivationError, acknowledge_config, acknowledge_policy_config,
    )
    from truememory.tier_switch.job import tier_job_alive
    from truememory.tier_switch.projection import ConfigProjectionConflict, mirror_activation
    if tier_job_alive(path):
        return  # A live job finishes its own projection.
    try:
        mirror_activation(result, expected_config=expected)
        if type(result) is LegacyTierPolicy:
            acknowledge_policy_config(conn, generation=result.generation)
        else:
            acknowledge_config(conn, generation=result.generation)
    except (ConfigProjectionConflict, TierActivationError, OSError):
        logging.getLogger(__name__).warning("Committed tier selection still awaits its config mirror", exc_info=True)


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
