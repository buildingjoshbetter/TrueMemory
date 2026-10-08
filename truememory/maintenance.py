"""Database maintenance ownership, source revisions and durable layer scheduling.

Importing this module loads no models and starts no worker or polling loop.
"""

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

from truememory._platform import try_file_lock
from truememory.storage import create_db


logger = logging.getLogger(__name__)


class MaintenanceUnavailableError(RuntimeError):
    """Source tracking is unavailable; a freshness token cannot be issued."""


class MaintenanceBusyError(RuntimeError):
    """Another operation owns maintenance for this database."""


class LayerUnavailableError(MaintenanceUnavailableError):
    """A builder could not use a dependency advertised by its cheap probe."""


class LayerDeferredError(RuntimeError):
    """Transient model ownership contention does not establish an attempt baseline."""


class MaintenanceOwnership(NamedTuple):
    path: Path | None
    process_id: int
    generation: str


class SourceRevision(NamedTuple):
    epoch: str
    revision: int
    insert_count: int
    correction_count: int
    max_seen_message_id: int
    nonappend_revision: int

    def is_append_only_since(self, previous: "SourceRevision") -> bool:
        """Certify stable IDs at/below the earlier high watermark, not time order."""
        return (
            self.epoch == previous.epoch
            and self.revision >= previous.revision
            and self.insert_count >= previous.insert_count
            and self.correction_count == previous.correction_count
            and self.max_seen_message_id >= previous.max_seen_message_id
            and self.nonappend_revision <= previous.revision
        )


def read_source_revision(conn: sqlite3.Connection) -> SourceRevision:
    """Read in the caller's snapshot without starting or committing a transaction."""
    try:
        row = conn.execute(
            "SELECT epoch, revision, insert_count, correction_count, max_seen_message_id, "
            "nonappend_revision, tracking_ready FROM maintenance_source_state WHERE singleton = 1"
        ).fetchone()
    except sqlite3.OperationalError as error:
        if str(error) == "no such table: maintenance_source_state":
            raise MaintenanceUnavailableError("Committed source revision tracking is unavailable") from error
        raise
    if row is None or not row[6]:
        raise MaintenanceUnavailableError("Committed source revision tracking is unavailable")
    return SourceRevision(*row[:6])


def canonical_database_path(db_path: str | os.PathLike[str]) -> Path | None:
    """A private in-memory database cannot be reopened by a worker connection."""
    if os.fspath(db_path) == ":memory:":
        return None
    return Path(db_path).expanduser().resolve()


def connection_database_path(conn: sqlite3.Connection) -> Path | None:
    """Identify the main database without changing connection/transaction state."""
    for _, name, path in conn.execute("PRAGMA database_list"):
        if name == "main":
            return canonical_database_path(path) if path else None
    raise MaintenanceUnavailableError("The connection has no main database")


_registry_lock = threading.Lock()
_coordinators: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_held_fds: set[int] = set()
_held_paths: dict[Path, int] = {}
_owner_waiters: dict[Path, tuple[int, int, "MaintenanceCoordinator"]] = {}
_thread_owners = threading.local()


def _acquire_owner(path: Path, on_busy: Callable[[], None] | None = None) -> int | None:
    with _registry_lock:
        if path in _held_paths:
            if on_busy is not None:
                on_busy()
            return None
        fd = try_file_lock(Path(str(path) + ".maintenance.lock"))
        if fd is not None:
            _held_fds.add(fd)
            _held_paths[path] = fd
        elif on_busy is not None:
            on_busy()
        return fd


def _release_owner(fd: int) -> None:
    notify = None
    with _registry_lock:
        os.close(fd)
        _held_fds.discard(fd)
        for path, claimed_fd in tuple(_held_paths.items()):
            if claimed_fd == fd:
                del _held_paths[path]
                waiter = _owner_waiters.get(path)
                if waiter is not None and waiter[0] == fd:
                    del _owner_waiters[path]
                    notify = waiter[2]
                else:
                    notify = _coordinators.get(path)
    if notify is not None:
        notify._owner_released()


@contextmanager
def _bind_owner(path: Path | None) -> Iterator[MaintenanceOwnership]:
    owners = getattr(_thread_owners, "owners", None)
    if owners is None:
        owners = {}
        _thread_owners.owners = owners
    token = MaintenanceOwnership(path, os.getpid(), uuid.uuid4().hex)
    owners[path] = token
    try:
        yield token
    finally:
        if token.process_id == os.getpid():
            del owners[path]


@contextmanager
def maintenance_owner(db_path: str | os.PathLike[str] | None) -> Iterator[MaintenanceOwnership]:
    """Own synchronous maintenance, nesting within the same thread's worker.

    In-memory callers still own serialization of their original connection;
    no additional connection or cross-process lock is created for that case.
    """
    path = canonical_database_path(db_path) if db_path is not None else None
    current = getattr(_thread_owners, "owners", {}).get(path)
    if current is not None and current.process_id == os.getpid():
        yield current
        return
    fd = _acquire_owner(path) if path is not None else None
    if path is not None and fd is None:
        raise MaintenanceBusyError("Maintenance is already running for this database")
    owner_pid = os.getpid()
    try:
        with _bind_owner(path) as token:
            yield token
    finally:
        # Fork cleanup already closed inherited descriptors. The same number
        # may now refer to an unrelated descriptor opened by the child.
        if fd is not None and owner_pid == os.getpid():
            _release_owner(fd)


def _after_fork() -> None:
    global _registry_lock, _coordinators, _held_fds, _held_paths, _owner_waiters, _thread_owners
    # The child must not prolong its parent's ownership by retaining an
    # inherited descriptor. Inherited coordinator objects reject child use.
    for fd in _held_fds:
        os.close(fd)
    _held_fds = set()
    _held_paths = {}
    _owner_waiters = {}
    _thread_owners = threading.local()
    _coordinators = weakref.WeakValueDictionary()
    _registry_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=lambda: _registry_lock.acquire(),
        after_in_parent=lambda: _registry_lock.release(),
        after_in_child=_after_fork,
    )


class MaintenanceCoordinator:
    """One worker per canonical file, with ownership lasting through teardown."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._pid = os.getpid()
        self._mutex = threading.Lock()
        self._active = False
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._done.set()
        self._status = "pending"
        self._error_category: str | None = None
        self._notification_generation = 0
        self._pending: tuple[int, int] | None = None
        self._last_report = None
        self._capability_epoch = 0
        self._capability_versions: tuple[tuple[str, str | None], ...] | None = None
        self._extension_failure: LayerDependency | None = None
        self._clustering_warning: tuple | None = None

    def _check_process(self) -> None:
        if self._pid != os.getpid():
            raise RuntimeError("Get a new maintenance coordinator after process fork")

    @property
    def status(self) -> tuple[str, str | None]:
        self._check_process()
        with self._mutex:
            return self._status, self._error_category

    def snapshot(self) -> dict:
        self._check_process()
        with self._mutex:
            return {"status": self._status, "error_category": self._error_category,
                    "active": self._active, "pending_generations": int(self._pending is not None),
                    "capability_epoch": self._capability_epoch,
                    "last_results": self._last_report.results if self._last_report is not None else ()}

    def refresh_capabilities(self, *, reset_extension: bool = True) -> None:
        """Refresh three package records at open/manual setup, never per add."""
        self._check_process()
        versions = tuple((module, _installed_dependency(module, distribution)) for module, distribution in (
            ("numpy", "numpy"), ("hdbscan", "hdbscan"), ("sqlite_vec", "sqlite-vec"),
        ))
        with self._mutex:
            self._capability_epoch += 1
            if reset_extension or versions != self._capability_versions:
                self._extension_failure = None
            self._capability_versions = versions

    def capability_snapshot(self) -> tuple[int, tuple, "LayerDependency | None"]:
        self._check_process()
        with self._mutex:
            return self._capability_epoch, self._capability_versions or (), self._extension_failure

    def _publish_extension_evidence(self, epoch: int, dependency: "LayerDependency | None") -> None:
        with self._mutex:
            if epoch == self._capability_epoch:
                self._extension_failure = dependency

    def observe_clustering_outcome(self, outcome: str, error_category: str | None = None) -> None:
        """Warn once per failure transition, only after an actual work attempt."""
        self._check_process()
        if outcome in {"success", "success_empty"}:
            with self._mutex:
                self._clustering_warning = None
            return
        if outcome not in {"unavailable", "failed"}:
            return
        category = _clustering_error_category(error_category) or "ClusteringUnavailable"
        missing = _missing_clustering_dependencies()
        key = (outcome, category, missing)
        with self._mutex:
            if key == self._clustering_warning:
                return
            self._clustering_warning = key
        guidance = _clustering_install_guidance(missing)
        logger.warning("Clustering %s (%s).%s", outcome, category,
                       " " + guidance if guidance else "")

    def clustering_failure(self) -> tuple[str, str] | None:
        """Last failure observed in this process, cleared by successful work."""
        self._check_process()
        with self._mutex:
            return self._clustering_warning[:2] if self._clustering_warning is not None else None

    def _reserve_locked(self) -> None:
        self._active = True
        self._done.clear()
        self._cancel.clear()
        self._status = "starting"
        self._error_category = None

    def request_layers(self, *, threshold: int = 25) -> bool:
        """Coalesce a real foreground wake; no callback captures its engine."""
        self._check_process()
        if type(threshold) is not int or threshold < 1:
            raise ValueError("Maintenance threshold must be a positive integer")
        with self._mutex:
            if self.path is None:
                self._status = "pending_in_memory"
                return False
            if self._active and self._cancel.is_set():
                return False
            self._notification_generation += 1
            self._pending = (self._notification_generation, threshold)
            pending = self._take_pending_locked()
        return self._launch_layers(pending) if pending is not None else False

    def _take_pending_locked(self) -> tuple[int, int] | None:
        if self._active or self._pending is None:
            return None
        pending, self._pending = self._pending, None
        self._reserve_locked()
        return pending

    def _launch_layers(self, pending: tuple[int, int]) -> bool:
        def work(conn: sqlite3.Connection, cancel: threading.Event) -> "MaintenanceReport":
            return run_engine_maintenance(conn, self, threshold=pending[1], cancel=cancel)
        return self._launch(work, pending)

    def _owner_released(self) -> None:
        self._check_process()
        with self._mutex:
            pending = self._take_pending_locked()
        if pending is not None:
            try:
                self._launch_layers(pending)
            except Exception:
                # The launch path records a bounded failure. Notification must
                # not replace a synchronous owner's unrelated operation error.
                pass

    def request(self, work: Callable[[sqlite3.Connection, threading.Event], None]) -> bool:
        """Start one owned worker, or report busy/in-memory pending without retry."""
        self._check_process()
        with self._mutex:
            if self.path is None:
                self._status = "pending_in_memory"
                return False
            if self._active:
                return False
            self._reserve_locked()
        return self._launch(work)

    def _launch(self, work: Callable, pending: tuple[int, int] | None = None) -> bool:
        def busy() -> None:
            # Registry -> coordinator ordering closes the release/admission
            # race. No coordinator-mutex holder acquires the registry lock.
            with self._mutex:
                self._active = False
                self._status = "cancelled" if self._cancel.is_set() else "busy"
                if pending is not None and self._pending is None and not self._cancel.is_set():
                    self._pending = pending
                if self._pending is not None and self.path in _held_paths:
                    _owner_waiters[self.path] = (_held_paths[self.path], self._pending[0], self)
                self._done.set()

        fd = None
        handoff = None
        handoff_lock = threading.Lock()
        try:
            # Serializing descriptor acquisition with fork registration makes
            # every inherited owner descriptor visible to the child cleanup.
            fd = _acquire_owner(self.path, on_busy=busy)
            if fd is None:
                return False
            handoff = {"fd": fd, "started": False}

            def run() -> None:
                with handoff_lock:
                    handoff["started"] = True
                    worker_fd = handoff.pop("fd", None)
                if worker_fd is not None:
                    self._run(worker_fd, work)

            worker = threading.Thread(target=run, daemon=True, name="truememory-maintenance")
            with self._mutex:
                self._thread = worker
                self._status = "running"
            worker.start()
            return True
        except BaseException:
            started = False
            if handoff is not None:
                with handoff_lock:
                    started = handoff["started"]
                    fd = handoff.pop("fd", None)
            if started:
                # start() can be interrupted after launching the target. That
                # worker owns cleanup until it really finishes, even on error.
                with self._mutex:
                    if self._active and self._thread is worker:
                        self._pending = None
                        self._cancel.set()
                raise
            if fd is not None:
                _release_owner(fd)
            with self._mutex:
                self._active = False
                self._thread = None
                self._status = "failed"
                self._error_category = "worker_start"
                successor = self._take_pending_locked()
                if successor is None:
                    self._done.set()
            if successor is not None:
                try:
                    self._launch_layers(successor)
                except Exception:
                    pass
            raise

    def _run(self, fd: int, work: Callable[[sqlite3.Connection, threading.Event], None]) -> None:
        conn = None
        outcome = "cancelled"
        error_category = None
        try:
            with _bind_owner(self.path):
                try:
                    if not self._cancel.is_set():
                        conn = create_db(self.path)
                        report = work(conn, self._cancel)
                        if conn.in_transaction:
                            conn.rollback()
                            raise RuntimeError("Maintenance work left an unfinished transaction")
                        outcome = "cancelled" if self._cancel.is_set() else "success"
                        if isinstance(report, MaintenanceReport):
                            outcome, error_category = maintenance_report_status(report, self._cancel)
                            with self._mutex:
                                self._last_report = report
                finally:
                    try:
                        if conn is not None:
                            conn.close()
                    finally:
                        conn = None
        except BaseException as error:
            # Thread boundary: expose only a category, never source-bearing
            # exception text or a traceback through threading.excepthook.
            outcome = "cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            error_category = type(error).__name__[:64]
        finally:
            _release_owner(fd)
            with self._mutex:
                self._active = False
                self._thread = None
                self._status = outcome
                self._error_category = error_category
                if self._cancel.is_set():
                    self._pending = None
                pending = self._take_pending_locked()
                if pending is None:
                    self._done.set()
            if pending is not None:
                try:
                    self._launch_layers(pending)
                except Exception:
                    pass  # Launch already recorded the categorical failure.

    def cancel(self) -> None:
        """Signal a phase boundary; never release ownership of active work."""
        self._check_process()
        with _registry_lock, self._mutex:
            self._pending = None
            self._cancel.set()
            waiter = _owner_waiters.get(self.path)
            if waiter is not None and waiter[2] is self:
                del _owner_waiters[self.path]

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for this worker if requested; report whether teardown completed."""
        self._check_process()
        return self._done.wait(timeout)


def get_coordinator(db_path: str | os.PathLike[str]) -> MaintenanceCoordinator:
    path = canonical_database_path(db_path)
    if path is None:
        return MaintenanceCoordinator(None)
    with _registry_lock:
        coordinator = _coordinators.get(path)
        if coordinator is None:
            coordinator = MaintenanceCoordinator(path)
            _coordinators[path] = coordinator
        return coordinator


class LayerDependency(NamedTuple):
    key: str
    builder_version: int
    available: bool
    error_category: str | None = None
    deferred: bool = False


def make_layer_dependency(
    builder_version: int, parameters: dict | None = None, *,
    available: bool = True, error_category: str | None = None,
) -> LayerDependency:
    """Version each provenance key independently; parameters must be nonpersonal."""
    if type(builder_version) is not int or builder_version < 1:
        raise ValueError("Builder version must be a positive integer")
    if error_category is not None and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", error_category) is None:
        raise ValueError("Dependency errors must be bounded categories")
    key = json.dumps({
        "builder_version": builder_version, "parameters": parameters or {},
        "available": available, "error_category": error_category,
        "python": list(sys.version_info[:3]),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return LayerDependency(key, builder_version, available, error_category)


class LayerSpec(NamedTuple):
    layer: str
    result_key: str
    resolve_dependency: Callable[[], LayerDependency]
    build: Callable[[sqlite3.Connection], object]
    count_output: Callable[[sqlite3.Connection], int]
    publication_guard: Callable[[sqlite3.Connection], AbstractContextManager[Callable[[], None]]] | None = None
    read_coverage: Callable[[sqlite3.Connection], str] | None = None
    borrowed_publication_guard: Callable[[sqlite3.Connection], AbstractContextManager[Callable[[], None]]] | None = None
    connection: sqlite3.Connection | None = None


class LayerState(NamedTuple):
    layer: str
    source: SourceRevision
    dependency: LayerDependency
    outcome: str
    successful_epoch: str | None
    successful_revision: int | None
    successful_dependency: str | None
    attempted_epoch: str | None
    attempted_revision: int | None
    attempted_dependency: str | None
    attempted_insert_count: int | None
    full_rebuild_required: bool
    output_count: int | None
    run_generation: str | None
    error_category: str | None
    successful_coverage: str = "unverified"


class LayerResult(NamedTuple):
    layer: str
    result_key: str
    outcome: str
    output_count: int | None
    elapsed_seconds: float
    error_category: str | None
    attempted: bool
    coverage: str = "unverified"
    pending_caller_commit: bool = False


_LAYER_COVERAGE = frozenset({"unverified", "complete", "legacy_contacts_unowned", "vector_generation_unverified"})


class _LayerRollbackFailed(RuntimeError):
    """Do not continue publication or diagnostics after a rejected rollback."""


def _validate_layer(layer: str) -> None:
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", layer) is None:
        raise ValueError("Invalid maintenance layer name")


def layer_freshness(source: SourceRevision, state: LayerState, dependency: LayerDependency) -> str:
    """Separate a prior valid success from the latest attempt's outcome."""
    if dependency.deferred:
        return "dependency_deferred"
    if (state.successful_epoch != source.epoch or state.successful_revision is None
            or state.full_rebuild_required or state.output_count is None):
        return "untrusted"
    if state.successful_dependency != dependency.key or not dependency.available:
        return "dependency_pending"
    if state.successful_revision == source.revision:
        return "current"
    if state.successful_revision > source.revision or source.nonappend_revision > state.successful_revision:
        return "correction_pending"
    return "append_pending"


def read_layer_states(conn: sqlite3.Connection, specs: tuple[LayerSpec, ...]) -> dict[str, LayerState]:
    """Read source and checkpoint rows in one SQLite snapshot; never commit."""
    if not specs:
        return {}
    dependencies = {}
    for spec in specs:
        _validate_layer(spec.layer)
        if spec.connection is not None and spec.connection is not conn:
            raise ValueError("Layer adapters belong to another connection")
        if spec.layer in dependencies:
            raise ValueError("Duplicate maintenance layer")
        dependencies[spec.layer] = spec.resolve_dependency()
    placeholders = ",".join("?" for _ in dependencies)
    try:
        rows = conn.execute(
            "SELECT s.epoch,s.revision,s.insert_count,s.correction_count,s.max_seen_message_id,"
            "s.nonappend_revision,s.tracking_ready,l.layer,l.outcome,l.successful_epoch,"
            "l.successful_revision,l.successful_dependency,l.attempted_epoch,l.attempted_revision,"
            "l.attempted_dependency,l.attempted_insert_count,l.full_rebuild_required,l.output_count,"
            "l.run_generation,l.error_category,l.successful_coverage FROM maintenance_source_state s LEFT JOIN maintenance_layers l "
            f"ON l.layer IN ({placeholders}) WHERE s.singleton=1", tuple(dependencies),
        ).fetchall()
    except sqlite3.OperationalError as error:
        if str(error).startswith(("no such table:", "no such column:")):
            raise MaintenanceUnavailableError("Maintenance tracking schema is unavailable") from error
        raise
    if not rows or not rows[0][6]:
        raise MaintenanceUnavailableError("Committed source revision tracking is unavailable")
    source = SourceRevision(*rows[0][:6])
    stored = {row[7]: row for row in rows if row[7] is not None}
    states = {}
    for layer, dependency in dependencies.items():
        row = stored.get(layer)
        values = row[8:] if row is not None else ("pending", None, None, None, None, None, None, None, 1, None, None, None, "unverified")
        states[layer] = LayerState(layer, source, dependency, *values)
    return states


@contextmanager
def layer_read_snapshot(
    conn: sqlite3.Connection, layer: str, dependency: LayerDependency,
) -> Iterator[LayerState]:
    """A read-only getter can read its row and provenance in the same snapshot."""
    owned = not conn.in_transaction
    if owned:
        conn.execute("BEGIN")
    try:
        spec = LayerSpec(layer, layer, lambda: dependency, lambda _conn: None, lambda _conn: 0)
        yield read_layer_states(conn, (spec,))[layer]
    finally:
        if owned and conn.in_transaction:
            conn.rollback()


def _eligible(state: LayerState, threshold: int, force: bool) -> bool:
    if state.dependency.deferred:
        return False
    if state.outcome == "running":
        return False
    if force or state.outcome == "abandoned":
        return True
    source = state.source
    if layer_freshness(source, state, state.dependency) == "current":
        return False
    if (state.attempted_epoch != source.epoch or state.attempted_revision is None
            or state.attempted_insert_count is None or state.attempted_dependency != state.dependency.key):
        return True
    terminal_failure = state.outcome in {"failed", "unavailable"}
    if terminal_failure and state.attempted_revision == source.revision:
        return False
    # A failed initial build still has a durable scheduling baseline even
    # though it has never established successful output provenance.
    if not terminal_failure and layer_freshness(source, state, state.dependency) == "untrusted":
        return True
    if source.nonappend_revision > state.attempted_revision:
        return True
    return source.insert_count - state.attempted_insert_count >= threshold


def plan_layers(
    conn: sqlite3.Connection, specs: tuple[LayerSpec, ...], *, threshold: int = 25, force: bool = False,
) -> tuple[LayerSpec, ...]:
    """Plan from durable attempt counts, including first/empty bootstrap work."""
    if type(threshold) is not int or threshold < 1:
        raise ValueError("Maintenance threshold must be a positive integer")
    states = read_layer_states(conn, specs)
    # A previous process may have died while recording "running". Let the
    # real ownership claim distinguish active work from an abandoned record.
    return tuple(spec for spec in specs if not states[spec.layer].dependency.deferred
                 and (states[spec.layer].outcome == "running" or _eligible(states[spec.layer], threshold, force)))


def record_layer_success_in_transaction(
    conn: sqlite3.Connection, *, layer: str, dependency: LayerDependency,
    source: SourceRevision, output_count: int, run_generation: str | None = None,
    coverage: str = "complete",
) -> None:
    """Publish provenance beside completed output, without owning caller commit."""
    _validate_layer(layer)
    if not conn.in_transaction:
        raise RuntimeError("Layer publication requires an existing transaction")
    if not dependency.available or type(output_count) is not int or output_count < 0:
        raise ValueError("Successful layer publication requires available output")
    if coverage not in _LAYER_COVERAGE:
        raise ValueError("Invalid layer coverage")
    if read_source_revision(conn) != source:
        raise sqlite3.OperationalError("Maintenance source changed before publication")
    conn.execute(
        "INSERT INTO maintenance_layers(layer,builder_version,outcome,successful_epoch,successful_revision,"
        "successful_dependency,attempted_epoch,attempted_revision,attempted_dependency,attempted_insert_count,"
        "full_rebuild_required,output_count,run_generation,error_category,successful_coverage) VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,NULL,?) "
        "ON CONFLICT(layer) DO UPDATE SET builder_version=excluded.builder_version,outcome=excluded.outcome,"
        "successful_epoch=excluded.successful_epoch,successful_revision=excluded.successful_revision,"
        "successful_dependency=excluded.successful_dependency,attempted_epoch=excluded.attempted_epoch,"
        "attempted_revision=excluded.attempted_revision,attempted_dependency=excluded.attempted_dependency,"
        "attempted_insert_count=excluded.attempted_insert_count,full_rebuild_required=0,"
        "output_count=excluded.output_count,run_generation=excluded.run_generation,error_category=NULL,"
        "successful_coverage=excluded.successful_coverage",
        (layer, dependency.builder_version, "success" if output_count else "success_empty", source.epoch,
         source.revision, dependency.key, source.epoch, source.revision, dependency.key,
         source.insert_count, output_count, run_generation, coverage),
    )


@contextmanager
def _owned_transaction(
    conn: sqlite3.Connection, *, on_rollback: Callable[[], None] | None = None,
) -> Iterator[None]:
    if conn.in_transaction:
        raise RuntimeError("Maintenance runner requires a clean transaction boundary")
    conn.execute("BEGIN")
    completed = False
    try:
        yield
        conn.execute("COMMIT")
        completed = True
    finally:
        if not completed:
            if conn.in_transaction:
                try:
                    conn.rollback()
                except BaseException as error:
                    if on_rollback is not None:
                        raise _LayerRollbackFailed("Output rollback failed") from error
                    raise
                if on_rollback is not None:
                    on_rollback()
            elif on_rollback is not None:
                raise _LayerRollbackFailed("Output transaction ended without a confirmed rollback")


@contextmanager
def _borrowed_transaction(
    conn: sqlite3.Connection, *, on_rollback: Callable[[], None] | None = None,
) -> Iterator[None]:
    if not conn.in_transaction:
        raise RuntimeError("Borrowed maintenance requires the caller's transaction")
    name = "truememory_layer_" + uuid.uuid4().hex
    conn.execute(f"SAVEPOINT {name}")
    completed = False
    try:
        yield
        conn.execute(f"RELEASE {name}")
        completed = True
    finally:
        if not completed and conn.in_transaction:
            try:
                conn.execute(f"ROLLBACK TO {name}")
            except BaseException as error:
                raise _LayerRollbackFailed("Caller must roll back the failed maintenance transaction") from error
            if on_rollback is not None:
                on_rollback()
            conn.execute(f"RELEASE {name}")
        elif not completed and on_rollback is not None:
            raise _LayerRollbackFailed("Caller transaction ended without a confirmed rollback")


def _record_attempt(
    conn: sqlite3.Connection, state: LayerState, outcome: str, generation: str, error_category: str | None,
    *, borrowed: bool = False,
) -> None:
    source, dependency = state.source, state.dependency
    if dependency.deferred:
        raise ValueError("Transient deferral cannot be recorded as a durable attempt")
    with (_borrowed_transaction(conn) if borrowed else _owned_transaction(conn)):
        conn.execute(
            "INSERT INTO maintenance_layers(layer,builder_version,outcome,attempted_epoch,attempted_revision,"
            "attempted_dependency,attempted_insert_count,run_generation,error_category) VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(layer) DO UPDATE SET builder_version=excluded.builder_version,outcome=excluded.outcome,"
            "attempted_epoch=excluded.attempted_epoch,attempted_revision=excluded.attempted_revision,"
            "attempted_dependency=excluded.attempted_dependency,attempted_insert_count=excluded.attempted_insert_count,"
            "run_generation=excluded.run_generation,error_category=excluded.error_category",
            (state.layer, dependency.builder_version, outcome, source.epoch, source.revision,
             dependency.key, source.insert_count, generation, error_category),
        )


class _LayerCancelled(Exception):
    """Internal cooperative cancellation before publication."""


_LAYER_ROW_COLUMNS = (
    "layer", "builder_version", "outcome", "successful_epoch", "successful_revision", "successful_dependency",
    "successful_coverage", "attempted_epoch", "attempted_revision", "attempted_dependency", "attempted_insert_count",
    "full_rebuild_required", "output_count", "run_generation", "error_category",
)


def _read_layer_row(conn: sqlite3.Connection, layer: str) -> tuple | None:
    row = conn.execute("SELECT " + ",".join(_LAYER_ROW_COLUMNS) + " FROM maintenance_layers WHERE layer=?", (layer,)).fetchone()
    return tuple(row) if row is not None else None


def _restore_deferred_attempt(
    conn: sqlite3.Connection, layer: str, previous: tuple | None, running: tuple,
    owner: MaintenanceOwnership,
) -> None:
    """Restore only this owner's unchanged diagnostic, after output rollback."""
    if conn.in_transaction:
        raise _LayerRollbackFailed("Output rollback is not complete")
    current_owner = getattr(_thread_owners, "owners", {}).get(owner.path)
    if current_owner != owner or owner.process_id != os.getpid():
        raise MaintenanceBusyError("Maintenance ownership changed before diagnostic restoration")
    with _owned_transaction(conn):
        conn.execute("UPDATE maintenance_layers SET layer=layer WHERE 0")
        current = _read_layer_row(conn, layer)
        if (current != running or current is None or current[2] != "running"
                or current[_LAYER_ROW_COLUMNS.index("run_generation")] != owner.generation):
            raise MaintenanceBusyError("Maintenance diagnostic changed before restoration")
        if previous is None:
            conn.execute("DELETE FROM maintenance_layers WHERE layer=? AND outcome='running' AND run_generation=?",
                         (layer, owner.generation))
        else:
            assignments = ",".join(column + "=?" for column in _LAYER_ROW_COLUMNS[1:])
            conn.execute("UPDATE maintenance_layers SET " + assignments
                         + " WHERE layer=? AND outcome='running' AND run_generation=?",
                         (*previous[1:], layer, owner.generation))


def _check_layer_dependency(current: LayerDependency, expected: LayerDependency) -> None:
    if current.deferred:
        raise LayerDeferredError("Embedding runtime is temporarily busy")
    if current != expected:
        raise MaintenanceUnavailableError("Layer dependency changed before publication")


def run_layers(
    conn: sqlite3.Connection, specs: tuple[LayerSpec, ...], *, force: bool = False,
    threshold: int = 25, cancel: threading.Event | None = None,
    allow_caller_transaction: bool = False,
) -> tuple[LayerResult, ...]:
    """Run eligible adapters once; borrowing requires explicit caller opt-in."""
    if type(allow_caller_transaction) is not bool:
        raise ValueError("Caller transaction opt-in must be a boolean")
    borrowed = conn.in_transaction and allow_caller_transaction
    if conn.in_transaction and not borrowed:
        raise RuntimeError("Maintenance runner requires a clean transaction boundary")
    if type(threshold) is not int or threshold < 1:
        raise ValueError("Maintenance threshold must be a positive integer")
    if any(spec.connection is not None and spec.connection is not conn for spec in specs):
        raise ValueError("Layer adapters belong to another connection")
    results = []
    with maintenance_owner(connection_database_path(conn)) as owner:
        def record_attempt(state: LayerState, outcome: str, category: str | None) -> None:
            if borrowed:
                _record_attempt(conn, state, outcome, owner.generation, category, borrowed=True)
            else:
                _record_attempt(conn, state, outcome, owner.generation, category)

        # Validate uniqueness before any builder is invoked.
        read_layer_states(conn, specs)
        for spec in specs:
            if cancel is not None and cancel.is_set():
                break
            state = read_layer_states(conn, (spec,))[spec.layer]
            if state.dependency.deferred:
                results.append(LayerResult(spec.layer, spec.result_key, "deferred", state.output_count, 0.0,
                    state.dependency.error_category, False, state.successful_coverage, borrowed))
                continue
            if state.outcome == "running" and state.run_generation != owner.generation:
                # Ownership proves abandonment. Do not acquire a caller's
                # writer merely to persist a diagnostic before computation.
                state = state._replace(outcome="abandoned")
            if not _eligible(state, threshold, force):
                results.append(LayerResult(spec.layer, spec.result_key,
                    layer_freshness(state.source, state, state.dependency), state.output_count, 0.0,
                    state.error_category, False, state.successful_coverage, borrowed))
                continue
            started = time.monotonic()
            baseline_state = state
            if not state.dependency.available:
                record_attempt(state, "unavailable", state.dependency.error_category)
                results.append(LayerResult(spec.layer, spec.result_key, "unavailable", state.output_count,
                    time.monotonic() - started, state.dependency.error_category, True, state.successful_coverage, borrowed))
                continue
            # This diagnostic commits before the read/compute snapshot starts.
            previous_row = running_row = None
            if not borrowed:
                with _owned_transaction(conn):
                    previous_row = _read_layer_row(conn, spec.layer)
                    conn.execute(
                        "INSERT INTO maintenance_layers(layer,builder_version,outcome,run_generation,error_category) "
                        "VALUES (?,?,'running',?,NULL) ON CONFLICT(layer) DO UPDATE SET "
                        "builder_version=excluded.builder_version,outcome='running',"
                        "run_generation=excluded.run_generation,error_category=NULL",
                        (spec.layer, state.dependency.builder_version, owner.generation),
                    )
                    running_row = _read_layer_row(conn, spec.layer)
            count = None
            coverage = state.successful_coverage
            built = output_started = rolled_back = False

            def confirm_rollback() -> None:
                nonlocal rolled_back
                rolled_back = True

            try:
                # Factories capture identity before the attempt's read snapshot.
                # The deferred guard enters only at publication, then outlives
                # the transaction so both COMMIT and rollback retain ownership.
                factory = spec.borrowed_publication_guard if borrowed else spec.publication_guard
                if borrowed and spec.publication_guard is not None and factory is None:
                    raise LayerUnavailableError("The adapter has no borrowed publication contract")
                guard = factory(conn) if factory is not None else None
                with ExitStack() as publication:
                    transaction = _borrowed_transaction if borrowed else _owned_transaction
                    with transaction(conn, on_rollback=confirm_rollback):
                        output_started = True
                        state = read_layer_states(conn, (spec,))[spec.layer]
                        if state.dependency.deferred:
                            raise LayerDeferredError("Embedding runtime is temporarily busy")
                        if not state.dependency.available:
                            raise MaintenanceUnavailableError("Layer dependency became unavailable")
                        built = True
                        spec.build(conn)
                        if not conn.in_transaction:
                            raise RuntimeError("Layer builder committed the runner transaction")
                        count = spec.count_output(conn)
                        coverage = spec.read_coverage(conn) if spec.read_coverage is not None else "complete"
                        if cancel is not None and cancel.is_set():
                            raise _LayerCancelled()
                        _check_layer_dependency(spec.resolve_dependency(), state.dependency)
                        record_layer_success_in_transaction(conn, layer=spec.layer, dependency=state.dependency,
                            source=state.source, output_count=count, run_generation=owner.generation, coverage=coverage)
                        _check_layer_dependency(spec.resolve_dependency(), state.dependency)
                        if guard is not None:
                            # Register release before validation can fail, so
                            # validation errors also roll back under ownership.
                            validate_publication = publication.enter_context(guard)
                            validate_publication()
                        if cancel is not None and cancel.is_set():
                            raise _LayerCancelled()
            except BaseException as error:
                if isinstance(error, _LayerRollbackFailed):
                    raise
                if isinstance(error, LayerDeferredError):
                    if output_started and not rolled_back:
                        raise _LayerRollbackFailed("Deferred output has no confirmed rollback") from error
                    if not borrowed:
                        _restore_deferred_attempt(conn, spec.layer, previous_row, running_row, owner)
                    results.append(LayerResult(spec.layer, spec.result_key, "deferred", baseline_state.output_count,
                        time.monotonic() - started, "ModelBusy", built, baseline_state.successful_coverage, borrowed))
                    if cancel is not None and cancel.is_set():
                        break
                    continue
                interrupted = not isinstance(error, Exception) or isinstance(error, _LayerCancelled)
                unavailable = not state.dependency.available or isinstance(error, LayerUnavailableError)
                outcome = "abandoned" if interrupted else ("unavailable" if unavailable else "failed")
                category = "Cancelled" if interrupted else type(error).__name__[:64]
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", category) is None:
                    category = "Error"
                record_attempt(state, outcome, category)
                results.append(LayerResult(spec.layer, spec.result_key, outcome, state.output_count,
                    time.monotonic() - started, category, True, state.successful_coverage, borrowed))
                if not isinstance(error, Exception):
                    raise
                if interrupted:
                    break
            else:
                results.append(LayerResult(spec.layer, spec.result_key, "success" if count else "success_empty",
                    count, time.monotonic() - started, None, True, coverage, borrowed))
    return tuple(results)


def nonvector_layer_specs() -> tuple[LayerSpec, ...]:
    """The original six standard-library adapters, with unchanged defaults."""
    definitions = (
        ("summaries", "build_summaries", "consolidation", "build_summaries",
         "SELECT count(*) FROM summaries WHERE period IN ('monthly','entity_monthly')", {}),
        ("structured_facts", "structured_facts", "consolidation", "build_structured_facts",
         "SELECT count(*) FROM summaries WHERE period='structured_fact'", {}),
        ("contradictions", "detect_contradictions", "consolidation", "detect_contradictions",
         "SELECT count(*) FROM fact_timeline", {}),
        ("surprise", "build_surprise_index", "predictive", "build_surprise_index",
         "SELECT count(*) FROM surprise_scores", {}),
        ("episodes", "detect_episodes", "temporal", "detect_episodes", "SELECT count(*) FROM episodes", {"gap_hours": 6}),
        ("landmarks", "detect_landmarks", "temporal", "detect_landmark_events",
         "SELECT count(*) FROM landmark_events", {}),
    )
    specs = []
    for layer, result_key, module_name, function_name, count_sql, parameters in definitions:
        def resolve(module_name: str = module_name, function_name: str = function_name,
                    parameters: dict = parameters) -> LayerDependency:
            try:
                getattr(importlib.import_module("truememory." + module_name), function_name)
            except (ImportError, AttributeError) as error:
                return make_layer_dependency(1, parameters, available=False, error_category=type(error).__name__)
            return make_layer_dependency(1, parameters)

        def build(conn: sqlite3.Connection, module_name: str = module_name,
                  function_name: str = function_name, parameters: dict = parameters) -> object:
            return getattr(importlib.import_module("truememory." + module_name), function_name)(conn, **parameters)

        def count(conn: sqlite3.Connection, sql: str = count_sql) -> int:
            return conn.execute(sql).fetchone()[0]

        specs.append(LayerSpec(layer, result_key, resolve, build, count))
    return tuple(specs)


def _installed_dependency(module: str, distribution: str) -> str | None:
    """Inspect package metadata without importing a native dependency."""
    try:
        if importlib.util.find_spec(module) is None:
            return None
        return importlib.metadata.version(distribution)
    except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
        return None


def _cluster_schedule_dependency(
    conn: sqlite3.Connection, *, versions: dict[str, str | None] | None = None,
) -> LayerDependency:
    parameters = {"min_cluster_size": 10, "min_samples": 5, "metric": "euclidean",
                  "cluster_selection_method": "eom"}
    if versions is None:
        versions = {module: _installed_dependency(module, distribution) for module, distribution in (
            ("numpy", "numpy"), ("hdbscan", "hdbscan"), ("sqlite_vec", "sqlite-vec"),
        )}
    parameters["versions"] = versions

    def unavailable(category: str) -> LayerDependency:
        return make_layer_dependency(1, parameters, available=False, error_category=category)

    if any(version is None for version in versions.values()):
        return unavailable("DependencyMissing")
    vector = sys.modules.get("truememory.vector_search")
    if vector is None:
        return unavailable("VectorRuntimeUnavailable")
    if not vector._lock.acquire(blocking=False):
        # There is no safe stable-key observation while the model is owned.
        # This ephemeral value must never become a durable version or attempt.
        return LayerDependency("", 1, False, "ModelBusy", deferred=True)
    owned = not conn.in_transaction
    try:
        if owned:
            conn.execute("BEGIN")
        group = vector._active_tier_group()
        table = vector._active_vec_table(conn)
        schema = [tuple(row) for row in conn.execute(
            "SELECT type,name,rootpage,sql FROM sqlite_master WHERE name=?", (table,),
        )]
        registry = [tuple(row) for row in conn.execute(
            "SELECT tier_group,vec_table,sep_table,model_name,embedding_dim "
            "FROM vector_cache_registry WHERE tier_group=?", (group,),
        )]
        metadata = [tuple(row) for row in conn.execute(
            "SELECT key,value FROM metadata WHERE key IN ('embed_model','embed_dim',?,?) ORDER BY key",
            (f"vec_build_state:{table}", f"vec_source_v1:{table}"),
        )]
        # Timestamp/count rewrites do not mean a new embedding space. Keep
        # them in the publication fence, not in the durable scheduling key.
        identity = [vector.EMBEDDING_MODEL, vector._embedding_dim, group, table, schema, registry, metadata]
        try:
            encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        except (TypeError, ValueError):
            return unavailable("VectorMetadataInvalid")
        parameters["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
        if not schema:
            return unavailable("VectorIndexMissing")
        if any("USING VEC0" in (row[3] or "").upper() for row in schema):
            if "vec0" not in {row[0] for row in conn.execute("PRAGMA module_list")}:
                return unavailable("VectorExtensionUnavailable")
        if (f"vec_build_state:{table}", "in_progress") in metadata:
            return unavailable("VectorRebuildInProgress")
        descriptor = dict(metadata).get(f"vec_source_v1:{table}")
        if descriptor is not None:
            try:
                manifest = json.loads(descriptor)
            except (TypeError, ValueError):
                return unavailable("VectorManifestInvalid")
            if (not isinstance(manifest, dict) or type(manifest.get("version")) is not int
                    or manifest["version"] != 1 or type(manifest.get("complete")) is not bool):
                return unavailable("VectorManifestInvalid")
            if not manifest["complete"]:
                return unavailable("VectorRebuildIncomplete")
        return make_layer_dependency(1, parameters)
    except sqlite3.OperationalError:
        return unavailable("VectorSchemaUnavailable")
    finally:
        try:
            if owned and conn.in_transaction:
                conn.rollback()
        finally:
            vector._lock.release()


def _cluster_module() -> ModuleType:
    try:
        return importlib.import_module("truememory.clustering")
    except ImportError as error:
        raise LayerUnavailableError("Clustering dependency import failed") from error


def _cluster_guard(conn: sqlite3.Connection) -> AbstractContextManager[Callable[[], None]]:
    module = _cluster_module()
    guard = module.cluster_publication_guard(conn)

    @contextmanager
    def hold() -> Iterator[Callable[[], None]]:
        try:
            with guard as validate:
                yield validate
        except module.ClusterModelBusyError as error:
            raise LayerDeferredError("Embedding runtime is temporarily busy") from error

    return hold()


def _borrowed_cluster_guard(conn: sqlite3.Connection) -> AbstractContextManager[Callable[[], None]]:
    """Fence a borrowed RELEASE, never the caller's later outer COMMIT."""
    module = _cluster_module()
    vector = sys.modules.get("truememory.vector_search")
    if vector is None:
        raise LayerUnavailableError("Embedding runtime is unavailable")
    if not vector._lock.acquire(blocking=False):
        raise LayerDeferredError("Embedding runtime is temporarily busy")
    try:
        expected = module._cluster_dependency_identity(conn, vector)
    finally:
        vector._lock.release()

    @contextmanager
    def hold() -> Iterator[Callable[[], None]]:
        conn.execute("UPDATE messages SET id=id WHERE 0")
        if not vector._lock.acquire(blocking=False):
            raise LayerDeferredError("Embedding runtime is temporarily busy")
        try:
            def validate() -> None:
                if module._cluster_dependency_identity(conn, vector) != expected:
                    raise MaintenanceUnavailableError("Clustering dependency changed before caller savepoint release")
            yield validate
        finally:
            vector._lock.release()

    return hold()


def _build_clusters(conn: sqlite3.Connection) -> int:
    module = _cluster_module()
    try:
        return module.cluster_messages(conn)
    except module.ClusterModelBusyError as error:
        raise LayerDeferredError("Embedding runtime is temporarily busy") from error
    except ImportError as error:
        raise LayerUnavailableError("Clustering dependency import failed") from error


def _build_dunbar(conn: sqlite3.Connection) -> object:
    # run_layers supplies the read snapshot, including primary selection.
    row = conn.execute(
        "SELECT sender,COUNT(*) AS cnt FROM messages WHERE sender != '' AND sender IS NOT NULL "
        "GROUP BY sender ORDER BY cnt DESC LIMIT 1"
    ).fetchone()
    primary = row[0] if row and row[0] and row[0].strip() else None
    return importlib.import_module("truememory.personality").build_dunbar_hierarchy(conn, primary_entity=primary)


def _dunbar_coverage(conn: sqlite3.Connection) -> str:
    coverage = importlib.import_module("truememory.personality").read_dunbar_coverage(conn)
    return {"managed_only": "complete", "partial": "legacy_contacts_unowned", "untracked": "unverified"}[coverage["status"]]


def all_layer_specs(conn: sqlite3.Connection) -> tuple[LayerSpec, ...]:
    """Eight persisted adapters; no imports of native packages or model loads.

    These adapters are bound to this connection. Vector users supply sqlite-vec
    and the initialized vector_search runtime. The cheap dependency resolver
    reports unavailable otherwise. Preferences remain a separate nonpersisted call.
    """
    specs = {spec.layer: spec for spec in nonvector_layer_specs()}

    def dunbar_dependency() -> LayerDependency:
        try:
            module = importlib.import_module("truememory.personality")
            getattr(module, "build_dunbar_hierarchy")
            getattr(module, "read_dunbar_coverage")
        except (ImportError, AttributeError) as error:
            return make_layer_dependency(1, available=False, error_category=type(error).__name__)
        return make_layer_dependency(1)

    specs["clusters"] = LayerSpec(
        "clusters", "cluster_messages", lambda: _cluster_schedule_dependency(conn), _build_clusters,
        lambda connection: connection.execute("SELECT count(*) FROM cluster_centroids").fetchone()[0],
        _cluster_guard, lambda _conn: "vector_generation_unverified", _borrowed_cluster_guard,
        conn,
    )
    specs["dunbar"] = LayerSpec(
        "dunbar", "dunbar_hierarchy", dunbar_dependency, _build_dunbar,
        lambda connection: importlib.import_module("truememory.personality").read_dunbar_coverage(connection)["managed_rows"],
        read_coverage=_dunbar_coverage,
    )
    return tuple(specs[name] for name in (
        "clusters", "summaries", "contradictions", "structured_facts", "surprise", "episodes", "landmarks", "dunbar",
    ))


class MaintenanceReport(NamedTuple):
    results: tuple[LayerResult, ...]
    preferences: str


def _dependency_parameters(dependency: LayerDependency) -> dict | None:
    if dependency.deferred:
        return None
    return json.loads(dependency.key)["parameters"]


def engine_layer_specs(
    conn: sqlite3.Connection, coordinator: MaintenanceCoordinator | None, *, evidence: tuple | None = None,
) -> tuple[LayerSpec, ...]:
    """Small live probes plus one bounded worker-capability evidence record."""
    if evidence is None:
        if coordinator is None:
            raise ValueError("Maintenance capability evidence is required")
        evidence = coordinator.capability_snapshot()
    _, versions, failure = evidence

    def resolve() -> LayerDependency:
        dependency = _cluster_schedule_dependency(conn, versions=dict(versions))
        if (failure is not None and not dependency.deferred
                and _dependency_parameters(dependency) == _dependency_parameters(failure)):
            return failure
        return dependency

    return tuple(spec._replace(resolve_dependency=resolve) if spec.layer == "clusters" else spec
                 for spec in all_layer_specs(conn))


def _clustering_error_category(value: str | None) -> str | None:
    if value is None:
        return None
    return value if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value) else "UnknownError"


def _missing_clustering_dependencies() -> tuple[str, ...]:
    """Only an absent top-level module warrants missing-package guidance."""
    missing = []
    for module, distribution in (("numpy", "numpy"), ("hdbscan", "hdbscan"), ("sqlite_vec", "sqlite-vec")):
        try:
            absent = importlib.util.find_spec(module) is None
        except (ImportError, ValueError):
            absent = False
        if absent:
            missing.append(distribution)
    return tuple(missing)


def _clustering_install_guidance(missing: tuple[str, ...]) -> str | None:
    if "hdbscan" in missing:
        return "Install the optional clustering extra: pip install 'truememory[clustering]'."
    return None


def clustering_health(
    conn: sqlite3.Connection | None = None, coordinator: MaintenanceCoordinator | None = None,
    *, pending_reason: str | None = None,
) -> dict:
    """Inspect existing evidence without opening a connection or loading models.

    Callers serialize their existing connection. Package probes inspect only
    top-level specs and distribution metadata; installed packages do not prove
    that their native imports or the database's published clusters are usable.
    """
    health = {
        "status": "degraded", "state": "unknown", "availability": "unknown", "outcome": None,
        "freshness": "unknown", "coverage": "unverified", "output_count": None,
        "last_error": None, "dependency_error": None, "missing_dependencies": [],
        "process_error": None,
        "guidance": None, "pending_reason": pending_reason or "not_connected",
        "pending_caller_commit": False,
    }
    owned = False
    try:
        missing = _missing_clustering_dependencies()
        versions = tuple((module, None if distribution in missing else _installed_dependency(module, distribution))
                         for module, distribution in (
            ("numpy", "numpy"), ("hdbscan", "hdbscan"), ("sqlite_vec", "sqlite-vec"),
        ))
        health["missing_dependencies"] = list(missing)
        health["guidance"] = _clustering_install_guidance(missing)
        if missing or any(version is None for _, version in versions):
            health.update(state="degraded", availability="unavailable", dependency_error="DependencyMissing")
        elif pending_reason is not None:
            health["state"] = "deferred"
        if conn is None or pending_reason is not None:
            return health
        health["pending_reason"] = None
        health["pending_caller_commit"] = conn.in_transaction
        process_failure = coordinator.clustering_failure() if coordinator is not None else None
        if process_failure is not None:
            health["process_error"] = process_failure[1]
        owned = not conn.in_transaction
        if owned:
            conn.execute("BEGIN")
        failure = coordinator.capability_snapshot()[2] if coordinator is not None else None
        spec = next(spec for spec in engine_layer_specs(conn, coordinator, evidence=(0, versions, failure))
                    if spec.layer == "clusters")
        state = read_layer_states(conn, (spec,))["clusters"]
        dependency = state.dependency
        freshness = layer_freshness(state.source, state, dependency)
        health.update(
            availability="deferred" if dependency.deferred else "available" if dependency.available else "unavailable",
            outcome=state.outcome, freshness=freshness, coverage=state.successful_coverage,
            output_count=state.output_count, last_error=_clustering_error_category(state.error_category),
            dependency_error=_clustering_error_category(dependency.error_category),
        )
        if dependency.deferred:
            health.update(state="deferred", pending_reason="model_busy")
        elif not dependency.available or state.outcome in {"failed", "unavailable"} or process_failure is not None:
            health["state"] = "degraded"
        elif conn.in_transaction and not owned:
            health.update(state="pending", pending_reason="pending_caller_commit")
        elif freshness != "current" or state.outcome not in {"success", "success_empty"}:
            health.update(state="pending", pending_reason="maintenance_pending")
        elif state.successful_coverage != "complete":
            health.update(state="degraded", pending_reason="coverage_unverified")
        else:
            health.update(state="ready", status="ok")
        return health
    except Exception as error:
        health.update(state="unknown", last_error=_clustering_error_category(type(error).__name__),
                      pending_reason="inspection_unavailable")
        return health
    finally:
        if owned and conn is not None:
            try:
                if conn.in_transaction:
                    conn.rollback()
            except sqlite3.Error as error:
                health.update(status="degraded", state="unknown",
                              last_error=_clustering_error_category(type(error).__name__),
                              pending_reason="inspection_rollback_failed")


def _prepare_worker_extensions(conn: sqlite3.Connection, coordinator: MaintenanceCoordinator) -> tuple:
    coordinator.refresh_capabilities(reset_extension=False)
    epoch, versions, previous_failure = coordinator.capability_snapshot()
    failed = False
    enabled = False
    try:
        if dict(versions).get("sqlite_vec") is None:
            failed = True
        else:
            module = importlib.import_module("sqlite_vec")
            conn.enable_load_extension(True)
            enabled = True
            module.load(conn)
    except Exception:
        failed = True
    finally:
        if enabled:
            conn.enable_load_extension(False)
    dependency = _cluster_schedule_dependency(conn, versions=dict(versions))
    failure = previous_failure if failed else None
    if failed and not dependency.deferred:
        parameters = _dependency_parameters(dependency)
        if parameters is not None and "identity_sha256" in parameters:
            failure = make_layer_dependency(dependency.builder_version, parameters,
                                            available=False, error_category="VectorExtensionUnavailable")
    elif dependency.error_category == "VectorExtensionUnavailable":
        failure = dependency
    coordinator._publish_extension_evidence(epoch, failure)
    return epoch, versions, failure


def run_engine_maintenance(
    conn: sqlite3.Connection, coordinator: MaintenanceCoordinator, *, threshold: int = 25,
    force: bool = False, cancel: threading.Event | None = None,
    prepare_extensions: bool = True, allow_caller_transaction: bool = False,
) -> MaintenanceReport:
    """Use only the supplied owned worker/original explicitly borrowed handle."""
    evidence = _prepare_worker_extensions(conn, coordinator) if prepare_extensions else None
    specs = engine_layer_specs(conn, coordinator, evidence=evidence)
    results = run_layers(conn, specs, threshold=threshold, force=force, cancel=cancel,
                         allow_caller_transaction=allow_caller_transaction)
    for result in results:
        if result.layer == "clusters" and result.attempted:
            # RELEASE only publishes into the caller's still-rollbackable
            # transaction. It cannot clear a previously observed failure.
            if not (result.pending_caller_commit and result.outcome in {"success", "success_empty"}):
                coordinator.observe_clustering_outcome(result.outcome, result.error_category)
    preferences = "SKIPPED (no maintenance attempt)"
    if cancel is not None and cancel.is_set():
        preferences = "CANCELLED"
    elif force or any(result.attempted for result in results):
        started = time.monotonic()
        try:
            importlib.import_module("truememory.personality").extract_preferences(conn)
        except (ImportError, AttributeError):
            preferences = "UNAVAILABLE (DependencyMissing)"
        except Exception as error:
            preferences = "ERROR (" + type(error).__name__[:64] + ")"
        else:
            preferences = f"{time.monotonic() - started:.3f}s"
        if conn.in_transaction and allow_caller_transaction:
            preferences += " (pending caller commit)"
    return MaintenanceReport(results, preferences)


def maintenance_report_status(report: MaintenanceReport, cancel: threading.Event | None = None) -> tuple[str, str | None]:
    if cancel is not None and cancel.is_set():
        return "cancelled", "Cancelled"
    for outcome in ("failed", "unavailable", "deferred", "abandoned"):
        for result in report.results:
            if result.outcome == outcome:
                return outcome, result.error_category
    if report.preferences.startswith(("ERROR", "UNAVAILABLE")):
        return "failed", "PreferencesUnavailable"
    if any(result.outcome not in {"success", "success_empty", "current"} for result in report.results):
        return "pending", None
    if any(result.coverage != "complete" for result in report.results):
        return "completed_with_limits", None
    return "success", None


_MAINTENANCE_RESULT_KEYS = (
    "cluster_messages", "build_summaries", "detect_contradictions", "structured_facts",
    "build_surprise_index", "detect_episodes", "detect_landmarks", "dunbar_hierarchy", "extract_preferences",
)


def maintenance_busy_result() -> dict[str, str]:
    return dict.fromkeys(_MAINTENANCE_RESULT_KEYS, "BUSY (maintenance owner active)")


def format_maintenance_report(report: MaintenanceReport) -> dict[str, str]:
    counts = {"clusters": "clusters", "structured_facts": "facts", "episodes": "episodes",
              "landmarks": "events", "dunbar": "relationships"}
    output = {}
    for result in report.results:
        if result.outcome in {"success", "success_empty"}:
            prefix = f"{result.output_count} {counts[result.layer]} in " if result.layer in counts else ""
            value = prefix + f"{result.elapsed_seconds:.3f}s"
            if result.coverage != "complete":
                value += " (" + result.coverage + ")"
        else:
            label = "ERROR" if result.outcome == "failed" else result.outcome.upper()
            value = label + (" (" + result.error_category + ")" if result.error_category else "")
        if result.pending_caller_commit:
            value += " (pending caller commit)"
        output[result.result_key] = value
    output["extract_preferences"] = report.preferences
    return output
