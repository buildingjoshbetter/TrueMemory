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
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

from truememory._platform import try_file_lock
from truememory.storage import (
    DEFAULT_BUSY_TIMEOUT_MS, DatabaseOpenError, _MAINTENANCE_SOURCE_FIELDS,
    _STYLE_OUTPUT_INVALIDATE_SQL, _STYLE_OUTPUT_TRIGGERS,
    _maintenance_trigger_definitions, _messages_have_rowid_id,
    _prepare_style_maintenance, _style_output_tracking_ready, create_db,
)


logger = logging.getLogger(__name__)


class MaintenanceUnavailableError(RuntimeError):
    """Source tracking is unavailable; a freshness token cannot be issued."""


class MaintenanceBusyError(RuntimeError):
    """Another operation owns maintenance for this database."""


class LayerUnavailableError(MaintenanceUnavailableError):
    """A builder could not use a dependency advertised by its cheap probe."""


class LayerDeferredError(RuntimeError):
    """A narrowly categorized transient condition establishes no attempt baseline."""

    def __init__(self, message: str, *, category: str = "ModelBusy") -> None:
        if category not in {"ModelBusy", "StyleCancelled", "StyleSourceChanged", "StyleWriterBusy", "StyleTriggersUnsupported"}:
            raise ValueError("Unsupported layer deferral category")
        self.category = category
        super().__init__(message)


class StyleAccumulatorUnsupported(LayerUnavailableError):
    """Tracked publication cannot replace a newer accumulator format."""


class _StyleSQLInterrupted(LayerDeferredError):
    """Only the owned progress callback can attest SQLite's automatic rollback."""

    def __init__(self, *, automatic_rollback: bool) -> None:
        super().__init__("Style work cancelled", category="StyleCancelled")
        self.automatic_rollback = automatic_rollback


def _style_sqlite_deferral(error: sqlite3.OperationalError, *, cancelled: bool = False) -> str | None:
    # SQLite's primary result codes are stable; Python 3.10 exposes neither
    # these named constants nor sqlite_errorcode on native exceptions.
    code = getattr(error, "sqlite_errorcode", None)
    if isinstance(code, int):
        primary = code & 255
        if primary in {5, 6}:  # SQLITE_BUSY / SQLITE_LOCKED, including extended codes.
            return "StyleWriterBusy"
        if primary == 9 and cancelled:  # SQLITE_INTERRUPT.
            return "StyleCancelled"
        return None
    message = str(error)
    if cancelled and message == "interrupted":
        return "StyleCancelled"
    if (message in {"database is locked", "database table is locked", "database schema is locked"}
            or message.startswith(("database table is locked: ", "database schema is locked: "))):
        return "StyleWriterBusy"
    return None


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


class _AutomaticAdmission(NamedTuple):
    coordinator: weakref.ReferenceType
    path: Path
    process_id: int
    descriptor: int
    launch: str
    released: threading.Event
    cancelled: threading.Event


def _automatic_admission_locked(path: Path) -> _AutomaticAdmission | None:
    """Read only under registry ownership, in registry/coordinator order."""
    coordinator = _coordinators.get(path)
    if coordinator is None:
        return None
    if not coordinator._mutex.acquire(blocking=False):
        return None
    try:
        origin = coordinator._admission_owner
        if (origin is not None and origin.coordinator() is coordinator
                and origin.process_id == os.getpid() == coordinator._pid
                and origin.path == path and _held_paths.get(path) == origin.descriptor
                and coordinator._active and not coordinator._cancel.is_set()
                and not origin.cancelled.is_set() and not origin.released.is_set()):
            return origin
    finally:
        coordinator._mutex.release()
    return None


def _acquire_owner(path: Path, on_busy: Callable[[], None] | None = None) -> int | None:
    with _registry_lock:
        if path in _held_paths:
            if hasattr(_thread_owners, "admission_failure"):
                _thread_owners.admission_failure = _automatic_admission_locked(path)
            if on_busy is not None:
                on_busy()
            return None
        fd = try_file_lock(Path(str(path) + ".maintenance.lock"))
        if fd is not None:
            _held_fds.add(fd)
            _held_paths[path] = fd
            launch = getattr(_thread_owners, "automatic_launch", None)
            if launch is not None:
                coordinator, identity = launch
                with coordinator._mutex:
                    if (coordinator is _coordinators.get(path) and coordinator.path == path
                            and coordinator._pid == os.getpid() and coordinator._active
                            and not coordinator._cancel.is_set()):
                        coordinator._admission_owner = _AutomaticAdmission(
                            weakref.ref(coordinator), path, os.getpid(), fd, identity,
                            threading.Event(), threading.Event())
        elif on_busy is not None:
            on_busy()
        return fd


def _release_owner(fd: int) -> None:
    notify = None
    with _registry_lock:
        origins = []
        for path, claimed_fd in tuple(_held_paths.items()):
            if claimed_fd == fd:
                coordinator = _coordinators.get(path)
                if coordinator is not None:
                    with coordinator._mutex:
                        origin = coordinator._admission_owner
                        if origin is not None and origin.descriptor == fd and origin.process_id == os.getpid():
                            coordinator._admission_owner = None
                            origins.append(origin)
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
        for origin in origins:
            origin.released.set()
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
    previous = getattr(_thread_owners, "admission_failure", None)
    _thread_owners.admission_failure = None
    try:
        fd = _acquire_owner(path) if path is not None else None
        failed_owner = _thread_owners.admission_failure
    finally:
        _thread_owners.admission_failure = previous
    if path is not None and fd is None:
        error = MaintenanceBusyError("Maintenance is already running for this database")
        error._automatic_owner = failed_owner
        raise error
    owner_pid = os.getpid()
    try:
        with _bind_owner(path) as token:
            yield token
    finally:
        # Fork cleanup already closed inherited descriptors. The same number
        # may now refer to an unrelated descriptor opened by the child.
        if fd is not None and owner_pid == os.getpid():
            _release_owner(fd)


def wait_for_automatic_owner(error: MaintenanceBusyError, *, deadline: float) -> None:
    """Wait only for the exact automatic owner observed at failed acquisition."""
    origin = getattr(error, "_automatic_owner", None)
    if (type(origin) is not _AutomaticAdmission or origin.process_id != os.getpid()
            or origin.cancelled.is_set()):
        raise error
    if not _registry_lock.acquire(blocking=False):
        raise error
    try:
        if not origin.released.is_set() and _automatic_admission_locked(origin.path) is not origin:
            raise error
    finally:
        _registry_lock.release()
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not origin.released.wait(min(remaining, threading.TIMEOUT_MAX)):
        raise error
    if origin.cancelled.is_set() or origin.process_id != os.getpid() or time.monotonic() >= deadline:
        raise error


def _valid_initialization_receipt(receipt: object, path: Path | None) -> bool:
    def bounded(value: object, limit: int) -> bool:
        if type(value) is not str or len(value) > limit:
            return False
        try:
            return len(value.encode()) <= limit
        except UnicodeError:
            return False

    if type(receipt) is not tuple or len(receipt) != 10 or path is None:
        return False
    if (not bounded(receipt[0], 4096) or receipt[0] != str(path)
            or any(type(value) is not int or value < 0 for value in receipt[1:4])
            or type(receipt[4]) is not tuple or len(receipt[4]) != 5
            or any(not bounded(value, 512) for value in receipt[4])
            or receipt[4][0] != "legacy"
            or type(receipt[5]) is not tuple or len(receipt[5]) != 2
            or type(receipt[6]) is not tuple or len(receipt[6]) > 3
            or type(receipt[7]) is not bool or type(receipt[8]) is not bool
            or type(receipt[9]) is not str or len(receipt[9]) > 32):
        return False
    tables = {"vec_messages", "vec_messages_sep", "vec_messages_edge", "vec_messages_sep_edge",
              "vec_messages_basepro", "vec_messages_sep_basepro"}
    if any(type(table) is not str or table not in tables for table in receipt[5]):
        return False
    for row in receipt[6]:
        if (type(row) is not tuple or len(row) != 9
                or any(not bounded(value, 512) for value in row[:3])
                or row[1] not in tables or row[2] not in tables
                or (row[3] is not None and not bounded(row[3], 512))
                or any(value is not None and type(value) not in (int, float) for value in row[4:])):
            return False
    return True


@contextmanager
def _active_initialization_receipt(path: Path, *, expected: tuple | None = None) -> Iterator[tuple | None]:
    """Guard a live automatic owner's evidence; the body must perform no I/O."""
    with _registry_lock:
        coordinator = _coordinators.get(path)
        if coordinator is None:
            yield None
            return
        with coordinator._mutex:
            record = coordinator._active_initialization
            current = (record is not None and coordinator._pid == os.getpid()
                       and coordinator._active and coordinator._automatic_initialization
                       and not coordinator._cancel.is_set() and coordinator._thread is not None
                       and record[0][0] == os.getpid() and _held_paths.get(path) == record[0][1])
            if expected is not None:
                current = current and expected[0]() is coordinator and expected[1] == record[0]
            yield (weakref.ref(coordinator), record[0], record[1]) if current else None


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
    for coordinator in tuple(_coordinators.values()):
        coordinator._admission_owner = None
        coordinator._active_initialization = None
        coordinator._pending_initialization = None
        coordinator._reserved_initialization = None
        coordinator._automatic_initialization = False
    _coordinators = weakref.WeakValueDictionary()
    _registry_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=lambda: _registry_lock.acquire(),
        after_in_parent=lambda: _registry_lock.release(),
        after_in_child=_after_fork,
    )


class MaintenanceRequest(NamedTuple):
    generation: int
    threshold: int
    include_layers: bool = True
    include_style: bool = False

    def merge(self, older: "MaintenanceRequest | None") -> "MaintenanceRequest":
        if older is None:
            return self
        latest = self if self.generation >= older.generation else older
        return latest._replace(include_layers=self.include_layers or older.include_layers,
                               include_style=self.include_style or older.include_style)


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
        self._pending: MaintenanceRequest | None = None
        self._last_report = None
        self._observation_generation = 0
        self._capability_epoch = 0
        self._capability_versions: tuple[tuple[str, str | None], ...] | None = None
        self._extension_failure: LayerDependency | None = None
        self._clustering_warning: tuple | None = None
        self._pending_initialization: tuple | None = None
        self._reserved_initialization: tuple | None = None
        self._active_initialization: tuple | None = None
        self._automatic_initialization = False
        self._admission_owner: _AutomaticAdmission | None = None

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

    def _begin_observation(self) -> int:
        self._check_process()
        with self._mutex:
            self._observation_generation += 1
            return self._observation_generation

    def _finish_observation(self, token: int, report: "MaintenanceReport | None", outcome: str,
                            error_category: str | None, *, borrowed: bool = False) -> None:
        self._check_process()
        with self._mutex:
            if token != self._observation_generation or borrowed:
                return
            self._status = outcome
            self._error_category = _style_error_category(error_category)
            if report is not None:
                self._last_report = report

    def style_observation(self) -> tuple["LayerResult | None", str | None]:
        self._check_process()
        with self._mutex:
            return (self._last_report.style_result if self._last_report is not None else None,
                    self._error_category)

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
        self._active_initialization = None
        self._reserved_initialization = None
        self._automatic_initialization = False

    def request_layers(self, *, threshold: int = 25, include_layers: bool = True,
                       include_style: bool = False, _initialization_receipt: tuple | None = None) -> bool:
        """Coalesce a real foreground wake; no callback captures its engine."""
        self._check_process()
        if type(threshold) is not int or threshold < 1:
            raise ValueError("Maintenance threshold must be a positive integer")
        if type(include_layers) is not bool or type(include_style) is not bool or not (include_layers or include_style):
            raise ValueError("Maintenance request requires explicit work kinds")
        valid_receipt = _valid_initialization_receipt(_initialization_receipt, self.path)
        with self._mutex:
            if self.path is None:
                self._pending_initialization = None
                self._status = "pending_in_memory"
                return False
            if self._active and self._cancel.is_set():
                self._pending_initialization = None
                return False
            self._notification_generation += 1
            self._pending_initialization = (
                (self._notification_generation, _initialization_receipt) if valid_receipt else None)
            self._pending = MaintenanceRequest(self._notification_generation, threshold,
                                               include_layers, include_style).merge(self._pending)
            pending = self._take_pending_locked()
        return self._launch_layers(pending) if pending is not None else False

    def _take_pending_locked(self) -> MaintenanceRequest | None:
        if self._active or self._pending is None:
            return None
        pending, self._pending = self._pending, None
        evidence, self._pending_initialization = self._pending_initialization, None
        self._reserve_locked()
        if evidence is not None and evidence[0] == pending.generation:
            self._reserved_initialization = evidence
        return pending

    def _launch_layers(self, pending: MaintenanceRequest) -> bool:
        with self._mutex:
            evidence, self._reserved_initialization = self._reserved_initialization, None
        receipt = evidence[1] if evidence is not None and evidence[0] == pending.generation else None
        def work(conn: sqlite3.Connection, cancel: threading.Event) -> "MaintenanceReport":
            return run_engine_maintenance(conn, self, threshold=pending.threshold, cancel=cancel,
                include_layers=pending.include_layers, include_style=pending.include_style, connection_owned=True)
        receipt_kwargs = {} if receipt is None else {"_initialization_receipt": receipt}
        if not pending.include_layers:
            return self._launch(work, pending, style_only=True, **receipt_kwargs)
        return self._launch(work, pending, **receipt_kwargs)

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

    def _launch(self, work: Callable, pending: MaintenanceRequest | None = None, *, style_only: bool = False,
                _initialization_receipt: tuple | None = None) -> bool:
        def busy() -> None:
            # Registry -> coordinator ordering closes the release/admission
            # race. No coordinator-mutex holder acquires the registry lock.
            with self._mutex:
                self._active = False
                self._active_initialization = self._reserved_initialization = None
                self._automatic_initialization = False
                self._status = "cancelled" if self._cancel.is_set() else "busy"
                if pending is not None and not self._cancel.is_set():
                    self._pending = pending.merge(self._pending)
                if (self._cancel.is_set() or self._pending is None
                        or (self._pending_initialization is not None
                            and self._pending_initialization[0] != self._pending.generation)):
                    self._pending_initialization = None
                if self._pending is not None and self.path in _held_paths:
                    _owner_waiters[self.path] = (_held_paths[self.path], self._pending[0], self)
                self._done.set()

        fd = None
        handoff = None
        handoff_lock = threading.Lock()
        with self._mutex:
            launch_token = self._observation_generation
        try:
            # Serializing descriptor acquisition with fork registration makes
            # every inherited owner descriptor visible to the child cleanup.
            previous_launch = getattr(_thread_owners, "automatic_launch", None)
            _thread_owners.automatic_launch = (
                (self, uuid.uuid4().hex) if pending is not None
                and _valid_initialization_receipt(_initialization_receipt, self.path) else None)
            try:
                fd = _acquire_owner(self.path, on_busy=busy)
            finally:
                _thread_owners.automatic_launch = previous_launch
            if fd is None:
                return False
            launch_token = self._begin_observation()
            handoff = {"fd": fd, "started": False}

            def run() -> None:
                with handoff_lock:
                    handoff["started"] = True
                    worker_fd = handoff.pop("fd", None)
                if worker_fd is not None:
                    if style_only:
                        self._run(worker_fd, work, style_only=True)
                    else:
                        self._run(worker_fd, work)

            worker = threading.Thread(target=run, daemon=True, name="truememory-maintenance")
            with self._mutex:
                self._thread = worker
                self._status = "running"
                self._automatic_initialization = pending is not None
                if (pending is not None and not self._cancel.is_set()
                        and _valid_initialization_receipt(_initialization_receipt, self.path)):
                    self._active_initialization = ((os.getpid(), fd, uuid.uuid4().hex), _initialization_receipt)
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
                        if self._admission_owner is not None:
                            self._admission_owner.cancelled.set()
                            self._admission_owner.released.set()
                        self._pending = None
                        self._active_initialization = self._pending_initialization = self._reserved_initialization = None
                        self._automatic_initialization = False
                        self._cancel.set()
                raise
            self._finish_observation(launch_token, None, "failed", "worker_start")
            with self._mutex:
                self._active_initialization = self._reserved_initialization = None
                self._automatic_initialization = False
                if (self._cancel.is_set() or self._pending is None
                        or (self._pending_initialization is not None
                            and self._pending_initialization[0] != self._pending.generation)):
                    self._pending_initialization = None
            if fd is not None:
                _release_owner(fd)
            with self._mutex:
                self._active = False
                self._thread = None
                if self._pending is not None and pending is not None:
                    self._pending = self._pending.merge(pending)
                if (self._pending_initialization is not None and (self._pending is None
                        or self._pending_initialization[0] != self._pending.generation)):
                    self._pending_initialization = None
                successor = self._take_pending_locked()
                if successor is None:
                    self._done.set()
            if successor is not None:
                try:
                    self._launch_layers(successor)
                except Exception:
                    pass
            raise

    def _run(self, fd: int, work: Callable[[sqlite3.Connection, threading.Event], None], *, style_only: bool = False) -> None:
        conn = None
        outcome = "cancelled"
        error_category = None
        report = None
        token = None
        try:
            token = self._begin_observation()
            with _bind_owner(self.path):
                try:
                    if not self._cancel.is_set():
                        if style_only:
                            conn = _open_style_database(self.path, self._cancel)
                        else:
                            from truememory.tier_switch.runtime import open_serving_connection
                            conn = open_serving_connection(self.path, create_db)
                        report = work(conn, self._cancel)
                        if conn.in_transaction:
                            conn.rollback()
                            raise RuntimeError("Maintenance work left an unfinished transaction")
                        outcome = "cancelled" if self._cancel.is_set() else "success"
                        if isinstance(report, MaintenanceReport):
                            outcome, error_category = maintenance_report_status(report, self._cancel)
                finally:
                    try:
                        if conn is not None:
                            conn.close()
                    finally:
                        conn = None
        except BaseException as error:
            # Thread boundary: expose only a category, never source-bearing
            # exception text or a traceback through threading.excepthook.
            if isinstance(error, _StyleOpenRejected):
                report = MaintenanceReport((), "SKIPPED (style-only)", error.result)
                outcome, error_category = maintenance_report_status(report, self._cancel)
            else:
                outcome = "cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                error_category = type(error).__name__[:64]
                report = None
        finally:
            if token is not None:
                self._finish_observation(token, report if isinstance(report, MaintenanceReport) else None,
                                         outcome, error_category)
            with self._mutex:
                self._active_initialization = None
                self._automatic_initialization = False
            _release_owner(fd)
            with self._mutex:
                self._active = False
                self._thread = None
                if self._cancel.is_set():
                    self._pending = None
                    self._pending_initialization = self._reserved_initialization = None
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
            if self._admission_owner is not None:
                self._admission_owner.cancelled.set()
                self._admission_owner.released.set()
            self._pending = None
            self._active_initialization = self._pending_initialization = self._reserved_initialization = None
            self._automatic_initialization = False
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


@contextmanager
def maintenance_observation(coordinator: MaintenanceCoordinator, *, borrowed: bool = False) -> Iterator[list]:
    """Publish explicit work after caller cleanup and before releasing ownership."""
    with maintenance_owner(coordinator.path):
        token = coordinator._begin_observation()
        reports = []
        try:
            yield reports
        except BaseException as error:
            coordinator._finish_observation(token, None, "failed", type(error).__name__, borrowed=borrowed)
            raise
        else:
            report = reports[-1] if reports else None
            outcome, category = maintenance_report_status(report) if report is not None else ("failed", "MissingReport")
            coordinator._finish_observation(token, report, outcome, category, borrowed=borrowed)


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
    execution_guard: Callable[[sqlite3.Connection, bool], AbstractContextManager[None]] | None = None


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


def _read_layer_run_baseline(
    conn: sqlite3.Connection, specs: tuple[LayerSpec, ...], *, _canonical_source: bool = False,
) -> dict[str, LayerState] | None:
    """Return no style observation only for a read lock with unchanged ownership.

    This boundary is used before running diagnostics and for cancelled reads,
    never around builders or publication transactions.
    Canonical style readiness and checkpoint values share one read snapshot.
    """
    borrowed = conn.in_transaction
    try:
        if not _canonical_source:
            return read_layer_states(conn, specs)
        if not borrowed:
            conn.execute("BEGIN")
        try:
            _validate_routed_style_source(conn)
            states = read_layer_states(conn, specs)
            state = states.get("style_vectors")
            if state is not None:
                # Canonical preparation already established enrollment. A lost
                # proof here is repairable, not a completed unavailable attempt.
                if state.dependency.deferred or not state.dependency.available:
                    raise _StyleAppendChanged(state.dependency.error_category or "StyleMetadataChanged")
                if state.successful_coverage == "complete":
                    try:
                        dependency, _schema, _marker = _style_append_metadata(conn)
                        if dependency != state.dependency:
                            raise _StyleAppendChanged("StyleMetadataChanged")
                    except _StyleAppendChanged:
                        if state.outcome not in {"failed", "unavailable"}:
                            raise
                        # Keep a real builder's durable retry baseline. Its old
                        # coverage no longer certifies this read's metadata.
                        states[state.layer] = state._replace(successful_coverage="unverified")
            if not conn.in_transaction:
                raise _LayerRollbackFailed("Canonical style read lost its transaction")
            return states
        finally:
            if not borrowed:
                if not conn.in_transaction:
                    raise _LayerRollbackFailed("Canonical style read ended without rollback")
                try:
                    conn.rollback()
                except sqlite3.Error as error:
                    raise _LayerRollbackFailed("Canonical style read rollback failed") from error
                if conn.in_transaction:
                    raise _LayerRollbackFailed("Canonical style read rollback left a transaction")
    except sqlite3.OperationalError as error:
        if (len(specs) == 1 and specs[0].layer == "style_vectors" and conn.in_transaction == borrowed
                and _style_sqlite_deferral(error) == "StyleWriterBusy"):
            return None
        raise


def _style_baseline_busy_result(spec: LayerSpec, borrowed: bool) -> LayerResult:
    return LayerResult(spec.layer, spec.result_key, "deferred", None, 0.0,
                       "StyleWriterBusy", False, "unverified", borrowed)


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
    conn: sqlite3.Connection, *, on_rollback: Callable[[], None] | None = None, immediate: bool = False,
) -> Iterator[None]:
    if conn.in_transaction:
        raise RuntimeError("Maintenance runner requires a clean transaction boundary")
    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    completed = False
    automatic_rollback = False
    try:
        yield
        conn.execute("COMMIT")
        completed = True
    except _StyleSQLInterrupted as error:
        automatic_rollback = error.automatic_rollback
        raise
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
                if automatic_rollback:
                    on_rollback()
                else:
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
    *, borrowed: bool = False, canonical_source: bool = False,
) -> bool | None:
    source, dependency = state.source, state.dependency
    if dependency.deferred:
        raise ValueError("Transient deferral cannot be recorded as a durable attempt")
    transaction = _borrowed_transaction if borrowed else _owned_transaction
    with (transaction(conn, on_rollback=lambda: None) if canonical_source else transaction(conn)):
        if state.layer == "style_vectors":
            conn.execute("UPDATE maintenance_layers SET layer=layer WHERE 0")
            if canonical_source:
                _validate_routed_style_source(conn)
            if _style_extra_triggers(conn):
                return False
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
    owner: MaintenanceOwnership, *, protect_rollback: bool = False, canonical_source: bool = False,
) -> None:
    """Restore only this owner's unchanged diagnostic, after output rollback."""
    if conn.in_transaction:
        raise _LayerRollbackFailed("Output rollback is not complete")
    current_owner = getattr(_thread_owners, "owners", {}).get(owner.path)
    if current_owner != owner or owner.process_id != os.getpid():
        raise MaintenanceBusyError("Maintenance ownership changed before diagnostic restoration")
    with _owned_transaction(conn, on_rollback=(lambda: None) if protect_rollback else None):
        conn.execute("UPDATE maintenance_layers SET layer=layer WHERE 0")
        if canonical_source:
            _validate_routed_style_source(conn)
        if layer == "style_vectors" and _style_extra_triggers(conn):
            raise _StyleAppendChanged("StyleTriggersUnsupported")
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
        if current.error_category == "StyleTriggersUnsupported":
            raise LayerDeferredError("Style triggers prevent publication", category="StyleTriggersUnsupported")
        raise LayerDeferredError("Embedding runtime is temporarily busy")
    if current != expected:
        raise MaintenanceUnavailableError("Layer dependency changed before publication")


@contextmanager
def _maintenance_embedding_scope(conn: sqlite3.Connection, enabled: bool,
                                 cancel: threading.Event | None, *, connection_lock: object | None = None,
                                 deadline: float | None = None) -> Iterator[bool]:
    if not enabled or cancel is not None and cancel.is_set():
        yield True
        return
    from truememory.tier_switch.runtime import maintenance_serving_operation
    with maintenance_serving_operation(conn, cancelled=cancel, deadline=deadline,
                                       connection_lock=connection_lock) as supported:
        yield supported


def _selected_cluster_deferred(conn: sqlite3.Connection) -> LayerResult:
    return LayerResult("clusters", "cluster_messages", "deferred", None, 0.0,
                       "SelectedVectorLayerUnsupported", False, "unverified", conn.in_transaction)


def run_layers(
    conn: sqlite3.Connection, specs: tuple[LayerSpec, ...], *, force: bool = False,
    threshold: int = 25, cancel: threading.Event | None = None,
    allow_caller_transaction: bool = False, _canonical_source: bool = False,
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
    with maintenance_owner(connection_database_path(conn)) as owner, _maintenance_embedding_scope(
        conn, any(spec.layer == "clusters" for spec in specs), cancel,
    ) as supported:
        if not supported:
            results.append(_selected_cluster_deferred(conn))
            specs = tuple(spec for spec in specs if spec.layer != "clusters")
            if not specs:
                return tuple(results)
        def record_attempt(state: LayerState, outcome: str, category: str | None) -> bool:
            if _canonical_source:
                return _record_attempt(conn, state, outcome, owner.generation, category,
                                       borrowed=borrowed, canonical_source=True) is not False
            if borrowed:
                return _record_attempt(conn, state, outcome, owner.generation, category, borrowed=True) is not False
            return _record_attempt(conn, state, outcome, owner.generation, category) is not False

        # Validate uniqueness before any builder is invoked.
        baseline_options = {"_canonical_source": True} if _canonical_source else {}
        if _read_layer_run_baseline(conn, specs, **baseline_options) is None:
            return (_style_baseline_busy_result(specs[0], borrowed),)
        for spec in specs:
            if cancel is not None and cancel.is_set():
                break
            observed = _read_layer_run_baseline(conn, (spec,), **baseline_options)
            if observed is None:
                results.append(_style_baseline_busy_result(spec, borrowed))
                continue
            state = observed[spec.layer]
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
                recorded = record_attempt(state, "unavailable", state.dependency.error_category)
                results.append(LayerResult(spec.layer, spec.result_key, "unavailable" if recorded else "deferred",
                    state.output_count, time.monotonic() - started,
                    state.dependency.error_category if recorded else "StyleTriggersUnsupported",
                    recorded, state.successful_coverage, borrowed))
                continue
            # This diagnostic commits before the read/compute snapshot starts.
            previous_row = running_row = None
            if not borrowed:
                diagnostic_rolled_back = False

                def confirm_diagnostic_rollback() -> None:
                    nonlocal diagnostic_rolled_back
                    diagnostic_rolled_back = True

                try:
                    with _owned_transaction(conn, on_rollback=(
                        confirm_diagnostic_rollback if spec.layer == "style_vectors" else None
                    )):
                        if spec.layer == "style_vectors":
                            conn.execute("UPDATE maintenance_layers SET layer=layer WHERE 0")
                            if _canonical_source:
                                _validate_routed_style_source(conn)
                            if _style_extra_triggers(conn):
                                raise _StyleAppendChanged("StyleTriggersUnsupported")
                        previous_row = _read_layer_row(conn, spec.layer)
                        conn.execute(
                            "INSERT INTO maintenance_layers(layer,builder_version,outcome,run_generation,error_category) "
                            "VALUES (?,?,'running',?,NULL) ON CONFLICT(layer) DO UPDATE SET "
                            "builder_version=excluded.builder_version,outcome='running',"
                            "run_generation=excluded.run_generation,error_category=NULL",
                            (spec.layer, state.dependency.builder_version, owner.generation),
                        )
                        running_row = _read_layer_row(conn, spec.layer)
                except (sqlite3.OperationalError, _StyleAppendChanged) as error:
                    category = error.category if isinstance(error, _StyleAppendChanged) else _style_sqlite_deferral(error)
                    if (spec.layer != "style_vectors" or not diagnostic_rolled_back
                            or category not in {"StyleWriterBusy", "StyleTriggersUnsupported", "StyleSourceChanged"}):
                        raise
                    results.append(LayerResult(spec.layer, spec.result_key, "deferred", state.output_count,
                        time.monotonic() - started, category, False, state.successful_coverage))
                    continue
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
                    with transaction(conn, on_rollback=confirm_rollback), (
                        spec.execution_guard(conn, not borrowed) if spec.execution_guard is not None else nullcontext()
                    ):
                        output_started = True
                        state = read_layer_states(conn, (spec,))[spec.layer]
                        if (_canonical_source and spec.layer == "style_vectors"
                                and baseline_state.successful_coverage == "unverified"):
                            state = state._replace(successful_coverage="unverified")
                        if state.dependency.deferred:
                            _check_layer_dependency(state.dependency, state.dependency)
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
                runtime = sys.modules.get("truememory.tier_switch.runtime")
                reject = getattr(runtime, "raise_if_serving_rejection", None)
                if reject is not None:
                    reject(error)
                if isinstance(error, LayerDeferredError):
                    if output_started and not rolled_back:
                        raise _LayerRollbackFailed("Deferred output has no confirmed rollback") from error
                    if not borrowed:
                        try:
                            if _canonical_source:
                                _restore_deferred_attempt(conn, spec.layer, previous_row, running_row, owner,
                                                          protect_rollback=True, canonical_source=True)
                            elif spec.layer == "style_vectors":
                                _restore_deferred_attempt(conn, spec.layer, previous_row, running_row, owner,
                                                          protect_rollback=True)
                            else:
                                _restore_deferred_attempt(conn, spec.layer, previous_row, running_row, owner)
                        except _StyleAppendChanged as restore_error:
                            if spec.layer != "style_vectors" or conn.in_transaction:
                                raise
                            error = LayerDeferredError("Style triggers prevent diagnostic restoration",
                                                       category=restore_error.category)
                        except sqlite3.OperationalError as restore_error:
                            if (spec.layer != "style_vectors" or conn.in_transaction
                                    or _style_sqlite_deferral(restore_error) != "StyleWriterBusy"):
                                raise
                            # Another writer still owns SQLite. Keep the
                            # committed running row for later owner recovery;
                            # do not claim restoration or retry under this owner.
                    results.append(LayerResult(spec.layer, spec.result_key, "deferred", baseline_state.output_count,
                        time.monotonic() - started, error.category, built, baseline_state.successful_coverage, borrowed))
                    if cancel is not None and cancel.is_set():
                        break
                    continue
                interrupted = not isinstance(error, Exception) or isinstance(error, _LayerCancelled)
                unavailable = not state.dependency.available or isinstance(error, LayerUnavailableError)
                outcome = "abandoned" if interrupted else ("unavailable" if unavailable else "failed")
                category = "Cancelled" if interrupted else type(error).__name__[:64]
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", category) is None:
                    category = "Error"
                if not record_attempt(state, outcome, category):
                    outcome, category = "deferred", "StyleTriggersUnsupported"
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


def prepare_style_maintenance(
    conn: sqlite3.Connection, *, _on_safe_failure: Callable[[sqlite3.OperationalError], None] | None = None,
) -> None:
    """Explicit lazy enrollment; preserve a caller's final commit/rollback."""
    borrowed = conn.in_transaction
    try:
        read_source_revision(conn)
        if _style_extra_triggers(conn):
            raise _StyleAppendChanged("StyleTriggersUnsupported")
    except sqlite3.OperationalError as error:
        if _on_safe_failure is not None and conn.in_transaction == borrowed:
            _on_safe_failure(error)
        raise
    class GuardedPreparation:
        def __getattr__(self, name: str) -> object:
            return getattr(conn, name)

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
            cursor = conn.execute(sql, *args)
            # Storage upgrades the writer with this zero-row statement before
            # enrollment or invalidation. Its existing rollback boundary owns
            # failures here, including a trigger installed after the first probe.
            if sql == "UPDATE maintenance_layers SET layer=layer WHERE 0" and _style_extra_triggers(conn):
                raise _StyleAppendChanged("StyleTriggersUnsupported")
            return cursor

    _prepare_style_maintenance(GuardedPreparation(), _on_safe_failure=_on_safe_failure)


class _StyleAppendChanged(RuntimeError):
    """A fixed metadata or checkpoint proof no longer permits publication."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


def _style_extra_triggers(conn: sqlite3.Connection) -> bool:
    names = tuple(_STYLE_OUTPUT_TRIGGERS)
    main = conn.execute(
        "SELECT 1 FROM main.sqlite_master WHERE type='trigger' "
        "AND lower(tbl_name) IN ('entity_style_vectors','maintenance_layers','metadata') "
        "AND NOT (lower(tbl_name)='entity_style_vectors' AND name IN (?,?,?)) LIMIT 1", names,
    ).fetchone()
    temporary = conn.execute(
        "SELECT 1 FROM temp.sqlite_master WHERE type='trigger' "
        "AND lower(tbl_name) IN ('entity_style_vectors','maintenance_layers','metadata') LIMIT 1",
    ).fetchone()
    return main is not None or temporary is not None


def _style_error_category(category: str | None) -> str | None:
    if category is None:
        return None
    return category if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", category) else "Error"


def _style_source_ready(conn: sqlite3.Connection) -> None:
    """Observe current canonical tracking, never repair or infer its history."""
    required = {
        "messages": set(_MAINTENANCE_SOURCE_FIELDS),
        "maintenance_source_state": {"singleton", "epoch", "revision", "insert_count", "correction_count",
                                     "max_seen_message_id", "nonappend_revision", "tracking_ready"},
        "maintenance_layers": set(_LAYER_ROW_COLUMNS),
        "metadata": {"key", "value"},
    }
    shadows = (*required, "entity_style_vectors")
    placeholders = ",".join("?" for _ in shadows)
    if conn.execute("SELECT 1 FROM temp.sqlite_master WHERE type IN ('table','view') "
                    f"AND lower(name) IN ({placeholders}) LIMIT 1", shadows).fetchone() is not None:
        raise MaintenanceUnavailableError("Canonical style tables are shadowed")
    for table, columns in required.items():
        actual = {row[1].lower() for row in conn.execute(f"PRAGMA main.table_xinfo({table})")}
        if not columns.issubset(actual):
            raise MaintenanceUnavailableError("Canonical source schema is unavailable")
    if not _messages_have_rowid_id(conn) or not _valid_style_source(read_source_revision(conn)):
        raise MaintenanceUnavailableError("Canonical source identity is unavailable")
    definitions = _maintenance_trigger_definitions(conn)
    placeholders = ",".join("?" for _ in definitions)
    current = {row[0]: " ".join(row[1].strip().rstrip(";").split()) for row in conn.execute(
        "SELECT name,sql FROM main.sqlite_master WHERE type='trigger' "
        f"AND name IN ({placeholders})", tuple(definitions),
    )}
    expected = {name: " ".join(sql.split()) for name, sql in definitions.items()}
    shadow = conn.execute("SELECT 1 FROM temp.sqlite_master WHERE type='trigger' "
                          f"AND lower(name) IN ({placeholders}) LIMIT 1", tuple(definitions)).fetchone()
    if current != expected or shadow is not None:
        raise MaintenanceUnavailableError("Canonical source triggers are unavailable")


class StyleObservation(NamedTuple):
    state: LayerState | None
    checkpoint: tuple | None
    eligible: bool
    health: dict


def observe_style(conn: sqlite3.Connection | None = None, *, threshold: int = 25,
                  pending_reason: str | None = None) -> StyleObservation:
    """Bounded native-free readiness; no profile/source scans or schema writes."""
    if type(threshold) is not int or threshold < 1:
        raise ValueError("Maintenance threshold must be a positive integer")
    health = {"available": False, "outcome": "pending", "freshness": "unknown", "coverage": "unverified",
              "error_category": None, "output_count": None, "pending_reason": pending_reason or "not_connected",
              "pending_caller_commit": False, "hash_ready": False, "format_ready": False}
    if conn is None:
        return StyleObservation(None, None, False, health)
    borrowed = conn.in_transaction
    health["pending_caller_commit"] = borrowed
    owned = False
    state = checkpoint = None
    eligible = False
    try:
        if not borrowed:
            conn.execute("BEGIN")
            owned = True
        if _style_extra_triggers(conn):
            health.update(outcome="deferred", error_category="StyleTriggersUnsupported", pending_reason="unsupported_triggers")
            return StyleObservation(None, None, False, health)
        _style_source_ready(conn)
        spec = style_layer_spec(conn)
        state = read_layer_states(conn, (spec,))[spec.layer]
        checkpoint = _read_layer_row(conn, spec.layer)
        marker = conn.execute("SELECT value FROM metadata WHERE key='style_vec_hash_version'").fetchone()
        hash_ready = marker == ("2",)
        health["hash_ready"] = hash_ready
        default = next((row[4] for row in conn.execute("PRAGMA main.table_info(entity_style_vectors)")
                        if row[1] == "accumulator_version"), "unsupported")
        format_ready = default is None or default.strip(" ()'\"") in {"0", "1", "NULL", "null"}
        health["format_ready"] = format_ready
        ready = state.dependency.available
        freshness = layer_freshness(state.source, state, state.dependency)
        same_failure = (state.outcome in {"failed", "unavailable"}
            and state.attempted_epoch == state.source.epoch and state.attempted_revision == state.source.revision
            and state.attempted_dependency == state.dependency.key)
        scheduled = _eligible(state, threshold, False)
        # A terminal attempt owns its retry budget even without successful
        # coverage or hash2. Readiness shortcuts cannot bypass that baseline.
        eligible = scheduled if state.outcome in {"failed", "unavailable"} else (
            not ready or not hash_ready or not format_ready or state.successful_coverage != "complete"
            or state.outcome == "running" or scheduled)
        reason = (("tracking_repair" if not ready else "hash_migration" if not hash_ready else
                   "accumulator_pending" if not format_ready else "eligible") if eligible else
                  "attempt_unchanged" if same_failure else "current" if freshness == "current" else "awaiting_threshold")
        health.update(available=ready and format_ready, outcome=state.outcome,
                      freshness=freshness if hash_ready and format_ready else "dependency_pending",
                      coverage=state.successful_coverage if ready and hash_ready and format_ready else "unverified",
                      error_category=_style_error_category(state.error_category or state.dependency.error_category
                                                          or (None if format_ready else "StyleAccumulatorUnsupported")),
                      output_count=state.output_count, pending_reason=reason)
        if borrowed:
            eligible = False
            health.update(outcome="pending", pending_reason="pending_caller_commit")
    except MaintenanceUnavailableError:
        state = checkpoint = None
        eligible = False
        health.update(outcome="unavailable", error_category="StyleSourceUnavailable", pending_reason="source_unavailable")
    except sqlite3.Error as error:
        state = checkpoint = None
        eligible = False
        category = _style_sqlite_deferral(error) if isinstance(error, sqlite3.OperationalError) else None
        health.update(outcome="deferred" if category else "failed", error_category=category or type(error).__name__,
                      pending_reason="inspection_failed")
    finally:
        if owned:
            try:
                conn.rollback()
            except sqlite3.Error as error:
                state = checkpoint = None
                eligible = False
                health.update(available=False, outcome="failed", freshness="unknown", coverage="unverified",
                              error_category=type(error).__name__, pending_reason="inspection_rollback_failed")
    return StyleObservation(state, checkpoint, eligible, health)


def style_health(conn: sqlite3.Connection | None = None, coordinator: MaintenanceCoordinator | None = None,
                 *, pending_reason: str | None = None) -> dict:
    health = observe_style(conn, pending_reason=pending_reason).health
    if coordinator is not None:
        previous, category = coordinator.style_observation()
        current = (not health["pending_caller_commit"] and health["freshness"] == "current"
                   and health["coverage"] == "complete" and health["outcome"] in {"success", "success_empty"})
        if not current:
            health["last_error"] = category or (previous.error_category if previous is not None else None)
    return health


class _StyleOpenRejected(RuntimeError):
    def __init__(self, result: LayerResult) -> None:
        self.result = result
        super().__init__(result.error_category)


def _style_unready_result(health: dict) -> LayerResult:
    return LayerResult("style_vectors", "style_vectors", health["outcome"], health["output_count"], 0.0,
                       health["error_category"], False, health["coverage"], health["pending_caller_commit"])


def _open_style_database(path: Path, cancel: threading.Event | None = None) -> sqlite3.Connection:
    """Open an existing canonical file without general initialization or migrations."""
    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    def deferred(category: str) -> _StyleOpenRejected:
        return _StyleOpenRejected(LayerResult("style_vectors", "style_vectors", "deferred", None, 0.0,
                                             category, False, "unverified"))

    if cancelled():
        raise deferred("StyleCancelled")
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("Style worker requires a canonical file path")
    conn = None
    progress = False
    try:
        conn = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True)
        conn.row_factory = None
        for sql in (f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}", "PRAGMA foreign_keys=ON",
                    "PRAGMA synchronous=NORMAL", "PRAGMA cache_size=-64000", "PRAGMA mmap_size=268435456"):
            conn.execute(sql)
        if _style_extra_triggers(conn):
            raise deferred("StyleTriggersUnsupported")
        try:
            _style_source_ready(conn)
        except MaintenanceUnavailableError as error:
            raise _StyleOpenRejected(LayerResult("style_vectors", "style_vectors", "unavailable", None, 0.0,
                                                 "StyleSourceUnavailable", False, "unverified")) from error
        observation = observe_style(conn)
        if observation.state is None:
            raise _StyleOpenRejected(_style_unready_result(observation.health))
        if cancelled():
            raise deferred("StyleCancelled")
        if cancel is not None:
            conn.set_progress_handler(lambda: int(cancelled()), 1000)
            progress = True
        try:
            checked = conn.execute("PRAGMA quick_check(1)").fetchone()
        finally:
            if progress:
                conn.set_progress_handler(None, 0)
                progress = False
        if cancelled():
            raise deferred("StyleCancelled")
        if checked != ("ok",):
            raise DatabaseOpenError("Database integrity check failed; restore a known-good backup before retrying")
        return conn
    except BaseException as error:
        if conn is not None:
            try:
                if progress:
                    conn.set_progress_handler(None, 0)
                if conn.in_transaction:
                    conn.rollback()
            finally:
                conn.close()
        if isinstance(error, sqlite3.OperationalError):
            category = _style_sqlite_deferral(error, cancelled=cancelled())
            if category:
                raise deferred(category) from error
        if isinstance(error, sqlite3.DatabaseError) and not isinstance(error, DatabaseOpenError):
            message = str(error).lower()
            if "i/o error" in message:
                raise DatabaseOpenError("Database I/O failed; close all TrueMemory processes and retry") from error
            raise DatabaseOpenError("Database could not be validated; check access or restore a known-good backup") from error
        raise


def _validate_routed_style_source(conn: sqlite3.Connection) -> None:
    try:
        _style_source_ready(conn)
    except MaintenanceUnavailableError as error:
        raise _StyleAppendChanged("StyleSourceChanged") from error


def _prepare_routed_style_maintenance(conn: sqlite3.Connection) -> None:
    """Validate canonical tracking under writer ownership before enrollment."""
    borrowed = conn.in_transaction
    entered = rolled_back = False

    def confirmed_rollback() -> None:
        nonlocal rolled_back
        rolled_back = True

    transaction = (_borrowed_transaction(conn, on_rollback=confirmed_rollback) if borrowed else
                   _owned_transaction(conn, on_rollback=confirmed_rollback, immediate=True))
    try:
        if _style_output_tracking_ready(conn):
            return
        with transaction:
            entered = True
            if borrowed:
                conn.execute("UPDATE maintenance_layers SET layer=layer WHERE 0")
            _validate_routed_style_source(conn)
            prepare_style_maintenance(conn)
    except sqlite3.OperationalError as error:
        if (conn.in_transaction == borrowed and (not entered or rolled_back)
                and _style_sqlite_deferral(error) == "StyleWriterBusy"):
            raise LayerDeferredError("Style preparation writer busy", category="StyleWriterBusy") from error
        raise


def run_routed_style(conn: sqlite3.Connection, *, force: bool = False, threshold: int = 25,
                     cancel: threading.Event | None = None, allow_caller_transaction: bool = False,
                     connection_owned: bool = False) -> LayerResult:
    """Canonical engine route; explicit low-level style APIs retain their contract."""
    if cancel is not None and cancel.is_set():
        return LayerResult("style_vectors", "style_vectors", "deferred", None, 0.0, "StyleCancelled", False,
                           "unverified", conn.in_transaction)
    with maintenance_owner(connection_database_path(conn)):
        observation = observe_style(conn, threshold=threshold)
        if observation.state is None:
            return _style_unready_result(observation.health)
        if conn.in_transaction and not allow_caller_transaction:
            return _style_unready_result(observation.health)
        # A wrong marker cannot leave a previously certified checkpoint current.
        # Compare the complete old row under writer ownership; preserve failures.
        old = observation.checkpoint
        if (old is not None and old[2] in {"success", "success_empty"}
                and (not observation.health["hash_ready"] or not observation.health["format_ready"]
                     or observation.state.successful_coverage != "complete")):
            borrowed = conn.in_transaction
            transaction = _borrowed_transaction if borrowed else _owned_transaction
            rolled_back = False

            def confirmed_rollback() -> None:
                nonlocal rolled_back
                rolled_back = True

            try:
                with transaction(conn, on_rollback=confirmed_rollback):
                    conn.execute("UPDATE maintenance_layers SET layer=layer WHERE 0")
                    if _style_extra_triggers(conn):
                        raise LayerDeferredError("Style triggers changed", category="StyleTriggersUnsupported")
                    try:
                        _style_source_ready(conn)
                    except MaintenanceUnavailableError as error:
                        raise LayerDeferredError("Canonical source readiness changed", category="StyleSourceChanged") from error
                    if _read_layer_row(conn, "style_vectors") != old:
                        raise LayerDeferredError("Style checkpoint changed", category="StyleSourceChanged")
                    conn.execute(_STYLE_OUTPUT_INVALIDATE_SQL)
            except (sqlite3.OperationalError, LayerDeferredError) as error:
                category = error.category if isinstance(error, LayerDeferredError) else _style_sqlite_deferral(error)
                if not rolled_back or conn.in_transaction != borrowed or category is None:
                    raise
                return LayerResult("style_vectors", "style_vectors", "deferred", None, 0.0,
                                   category, False, "unverified", borrowed)
        return run_style_maintenance(conn, force=force, threshold=threshold, cancel=cancel,
            allow_caller_transaction=allow_caller_transaction, connection_owned=connection_owned,
            _canonical_source=True)


def _style_append_metadata(conn: sqlite3.Connection) -> tuple:
    dependency = style_layer_spec(conn).resolve_dependency()
    if dependency.error_category == "StyleTriggersUnsupported":
        raise _StyleAppendChanged("StyleTriggersUnsupported")
    if not dependency.available:
        raise _StyleAppendChanged("StyleMetadataChanged")
    schema = tuple(tuple(row) for row in conn.execute("PRAGMA main.table_info(entity_style_vectors)"))
    default = next((row[4] for row in schema if row[1] == "accumulator_version"), "unsupported")
    if default is not None and default.strip(" ()'\"") not in {"0", "1", "NULL", "null"}:
        raise _StyleAppendChanged("StyleAccumulatorUnsupported")
    marker = conn.execute("SELECT value FROM metadata WHERE key='style_vec_hash_version'").fetchone()
    if marker is None or marker[0] != "2":
        raise _StyleAppendChanged("StyleHashPending")
    return dependency, schema, tuple(marker)


def _style_checkpoint_projection(row: tuple, **changes: object) -> tuple:
    fields = dict(zip(_LAYER_ROW_COLUMNS, row))
    fields.update(changes)
    return tuple(fields[name] for name in _LAYER_ROW_COLUMNS)


def _style_invalidated_checkpoint(row: tuple) -> tuple:
    return _style_checkpoint_projection(
        row, outcome="pending", full_rebuild_required=1, successful_coverage="unverified",
        attempted_epoch=None, attempted_revision=None, attempted_dependency=None,
        attempted_insert_count=None, run_generation=None, error_category=None,
    )


class _StyleAppendSnapshot(NamedTuple):
    connection: sqlite3.Connection
    checkpoint: tuple | None
    state: LayerState | None
    error_category: str | None
    failed: bool = False
    metadata: tuple | None = None


def _valid_style_source(source: SourceRevision) -> bool:
    return (isinstance(source.epoch, str) and bool(source.epoch)
            and all(type(value) is int and value >= 0 for value in source[1:])
            and source.revision == source.insert_count + source.correction_count
            and source.nonappend_revision <= source.revision)


def _style_append_error_category(error: Exception) -> str:
    category = type(error).__name__[:64]
    return category if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", category) is not None else "Error"


def _capture_style_append(conn: sqlite3.Connection) -> _StyleAppendSnapshot:
    """Freeze metadata before one source insert, under its existing writer."""
    if not conn.in_transaction:
        raise RuntimeError("Style append capture requires the source transaction")
    checkpoint = None
    state = None
    try:
        checkpoint = _read_layer_row(conn, "style_vectors")
        spec = style_layer_spec(conn)
        state = read_layer_states(conn, (spec,))[spec.layer]
        if state.dependency.error_category == "StyleTriggersUnsupported":
            return _StyleAppendSnapshot(conn, checkpoint, state, "StyleTriggersUnsupported")
        if (checkpoint is None or checkpoint[1] != state.dependency.builder_version
                or state.outcome not in {"success", "success_empty"}
                or state.successful_coverage != "complete"
                or layer_freshness(state.source, state, state.dependency) != "current"
                or type(state.output_count) is not int or state.output_count < 0
                or (state.outcome == "success_empty") != (state.output_count == 0)
                or not _valid_style_source(state.source)
                or (state.attempted_epoch, state.attempted_revision, state.attempted_dependency,
                    state.attempted_insert_count) != (state.source.epoch, state.source.revision,
                                                     state.dependency.key, state.source.insert_count)):
            return _StyleAppendSnapshot(conn, checkpoint, state, "StyleCoveragePending")
        metadata = _style_append_metadata(conn)
    except _StyleAppendChanged as error:
        return _StyleAppendSnapshot(conn, checkpoint, state, error.category)
    except (sqlite3.Error, MaintenanceUnavailableError, StopIteration) as error:
        if not conn.in_transaction:
            raise _LayerRollbackFailed("Style capture lost the source transaction") from error
        category = _style_sqlite_deferral(error) if isinstance(error, sqlite3.OperationalError) else None
        return _StyleAppendSnapshot(conn, checkpoint, state, category or _style_append_error_category(error), category is None)
    return _StyleAppendSnapshot(conn, checkpoint, state, None, metadata=metadata)


@contextmanager
def _style_append_savepoint(conn: sqlite3.Connection) -> Iterator[None]:
    """Keep source writes when style fails, only after complete savepoint cleanup."""
    name = "truememory_style_coverage_" + uuid.uuid4().hex
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
        conn.execute(f"RELEASE {name}")
    except BaseException:
        if not conn.in_transaction:
            raise _LayerRollbackFailed("Style append lost the source transaction")
        try:
            conn.execute(f"ROLLBACK TO {name}")
            conn.execute(f"RELEASE {name}")
        except BaseException as error:
            raise _LayerRollbackFailed("Style append rollback or release failed") from error
        raise


def _publish_style_append(
    conn: sqlite3.Connection, previous: _StyleAppendSnapshot, message_id: int, *,
    sender: str, content: str, directive: bool, precomputed: bool, publish: Callable[[], None],
) -> LayerResult:
    """Advance one proven append without committing or rebuilding for its caller."""
    if previous.connection is not conn or not conn.in_transaction:
        raise RuntimeError("Style append publication requires its original source transaction")
    state = previous.state

    def pending(category: str, *, failed: bool = False) -> LayerResult:
        # Keep a prior failed attempt's retry baseline. A newly missed append
        # supersedes successful/running provenance, never a newer invalidation.
        if previous.checkpoint is not None and previous.checkpoint[2] in {"success", "success_empty", "running"}:
            try:
                with _style_append_savepoint(conn):
                    if _style_extra_triggers(conn):
                        category = "StyleTriggersUnsupported"
                    elif _read_layer_row(conn, "style_vectors") == previous.checkpoint:
                        conn.execute(_STYLE_OUTPUT_INVALIDATE_SQL)
                        expected = _style_invalidated_checkpoint(previous.checkpoint)
                        # The invalidation itself may invoke SQL triggers. Set
                        # its category only if every field still belongs to it.
                        if _read_layer_row(conn, "style_vectors") == expected:
                            predicate = " AND ".join(name + " IS ?" for name in _LAYER_ROW_COLUMNS)
                            conn.execute("UPDATE maintenance_layers SET error_category=? WHERE " + predicate,
                                         (category, *expected))
            except _LayerRollbackFailed:
                raise
            except sqlite3.Error as error:
                if not conn.in_transaction:
                    raise _LayerRollbackFailed("Style pending report lost the source transaction") from error
                category, failed = _style_append_error_category(error), True
        return LayerResult("style_vectors", "style_vectors", "failed" if failed else "deferred",
                           state.output_count if state is not None else None, 0.0, category,
                           False, "unverified", True)

    if previous.error_category is not None:
        return pending(previous.error_category, failed=previous.failed)
    try:
        if _read_layer_row(conn, "style_vectors") != previous.checkpoint:
            return pending("StyleCoverageChanged")
        if _style_append_metadata(conn) != previous.metadata:
            return pending("StyleMetadataChanged")
        source = read_source_revision(conn)
        before = state.source
        if (not _valid_style_source(source) or source.epoch != before.epoch
                or source.revision != before.revision + 1 or source.insert_count != before.insert_count + 1
                or source.correction_count != before.correction_count
                or source.nonappend_revision != before.nonappend_revision
                or type(message_id) is not int or message_id <= before.max_seen_message_id
                or source.max_seen_message_id != message_id):
            return pending("StyleAppendUnproven")
        row = conn.execute("SELECT sender,content,directive FROM messages WHERE id=?", (message_id,)).fetchone()
        if (not isinstance(sender, str) or not isinstance(content, str) or not content.strip()
                or row is None or tuple(row) != (sender, content, int(bool(directive)))):
            return pending("StyleAppendUnproven")
        included = bool(sender) and not directive
        if included and not precomputed:
            return pending("StylePrecomputeFailed", failed=True)
        with _style_append_savepoint(conn):
            count = state.output_count
            if included:
                exists = conn.execute("SELECT 1 FROM entity_style_vectors WHERE entity=?", (sender.lower(),)).fetchone()
                publish()
                count += int(exists is None)
            expected = _style_invalidated_checkpoint(previous.checkpoint) if included else previous.checkpoint
            if _read_layer_row(conn, "style_vectors") != expected:
                raise _StyleAppendChanged("StyleCoverageChanged")
            if _style_append_metadata(conn) != previous.metadata:
                raise _StyleAppendChanged("StyleMetadataChanged")
            if read_source_revision(conn) != source:
                raise LayerDeferredError("Style source changed during append", category="StyleSourceChanged")
            record_layer_success_in_transaction(conn, layer="style_vectors", dependency=state.dependency,
                                                source=source, output_count=count)
            success = _style_checkpoint_projection(
                previous.checkpoint, builder_version=state.dependency.builder_version,
                outcome="success" if count else "success_empty", successful_epoch=source.epoch,
                successful_revision=source.revision, successful_dependency=state.dependency.key,
                successful_coverage="complete", attempted_epoch=source.epoch, attempted_revision=source.revision,
                attempted_dependency=state.dependency.key, attempted_insert_count=source.insert_count,
                full_rebuild_required=0, output_count=count, run_generation=None, error_category=None,
            )
            if _read_layer_row(conn, "style_vectors") != success:
                raise _StyleAppendChanged("StyleCoverageChanged")
            if _style_append_metadata(conn) != previous.metadata or read_source_revision(conn) != source:
                raise _StyleAppendChanged("StyleMetadataChanged")
    except _LayerRollbackFailed:
        raise
    except Exception as error:
        if not conn.in_transaction:
            raise _LayerRollbackFailed("Style append lost the source transaction") from error
        category = (error.category if isinstance(error, (LayerDeferredError, _StyleAppendChanged)) else
                    _style_sqlite_deferral(error) if isinstance(error, sqlite3.OperationalError) else None)
        return pending(category or _style_append_error_category(error), failed=category is None)
    return LayerResult("style_vectors", "style_vectors", "success" if count else "success_empty",
                       count, 0.0, None, True, "complete", True)


def style_layer_spec(
    conn: sqlite3.Connection, *, cancel: threading.Event | None = None, connection_owned: bool = False,
    _canonical_source: bool = False,
) -> LayerSpec:
    """Describe private style work without installing schema or reading profiles."""
    if type(connection_owned) is not bool:
        raise ValueError("Connection ownership opt-in must be a boolean")
    parameters = {
        "accumulator_version": 1, "hash_version": 2, "dimension": 256,
        "ngrams": [3, 4, 5], "sum_format": "<256d",
    }

    def resolve_dependency() -> LayerDependency:
        if _style_extra_triggers(conn):
            return make_layer_dependency(1, parameters, available=False,
                error_category="StyleTriggersUnsupported")._replace(deferred=True)
        ready = _style_output_tracking_ready(conn)
        return make_layer_dependency(1, parameters, available=ready,
                                     error_category=None if ready else "StyleTrackingUnprepared")

    def check_cancel() -> None:
        if cancel is not None and cancel.is_set():
            raise LayerDeferredError("Style work cancelled", category="StyleCancelled")

    @contextmanager
    def execution_guard(connection: sqlite3.Connection, owned: bool) -> Iterator[None]:
        style = importlib.import_module("truememory.personality_style_vec")
        progress = connection_owned and owned and cancel is not None
        interrupted_by_callback = False

        def progress_handler() -> int:
            nonlocal interrupted_by_callback
            if cancel.is_set():
                interrupted_by_callback = True
                return 1
            return 0

        if progress:
            connection.set_progress_handler(progress_handler, 1000)
        try:
            check_cancel()
            yield
        except _LayerCancelled as error:
            raise LayerDeferredError("Style work cancelled", category="StyleCancelled") from error
        except style._StyleSourceChanged as error:
            raise LayerDeferredError("Style source changed", category="StyleSourceChanged") from error
        except sqlite3.OperationalError as error:
            category = _style_sqlite_deferral(error, cancelled=cancel is not None and cancel.is_set())
            if category == "StyleCancelled":
                code = getattr(error, "sqlite_errorcode", None)
                native_interrupt = (isinstance(code, int) and (code & 255) == 9) or (
                    code is None and not hasattr(sqlite3, "SQLITE_INTERRUPT") and str(error) == "interrupted"
                )
                raise _StyleSQLInterrupted(automatic_rollback=bool(
                    progress and interrupted_by_callback and native_interrupt and not connection.in_transaction
                )) from error
            if category == "StyleWriterBusy":
                raise LayerDeferredError("Style writer busy", category="StyleWriterBusy") from error
            raise
        finally:
            # Exit before the runner rolls back or commits. Borrowed handlers
            # cannot be recovered through SQLite's API, so never replace them.
            if progress:
                connection.set_progress_handler(None, 0)

    def compatible(connection: sqlite3.Connection) -> None:
        # A custom future schema can be empty. Do not downgrade its declared
        # accumulator format merely because no row demonstrates it yet.
        default = next(row[4] for row in connection.execute("PRAGMA table_info(entity_style_vectors)")
                       if row[1] == "accumulator_version")
        if default is not None:
            literal = default.strip(" ()'\"")
            if literal not in {"0", "1", "NULL", "null"}:
                raise StyleAccumulatorUnsupported("Unsupported style accumulator default")
        if connection.execute(
            "SELECT 1 FROM entity_style_vectors WHERE accumulator_version > 1 LIMIT 1"
        ).fetchone():
            raise StyleAccumulatorUnsupported("Unsupported style accumulator version")

    def build(connection: sqlite3.Connection) -> object:
        style = importlib.import_module("truememory.personality_style_vec")
        source = read_source_revision(connection)
        schema, query, rows = style._capture_style_source(connection, check_cancel)
        result, stored_rows = style._compute_entity_style_vectors(rows, check_cancel=check_cancel)
        check_cancel()
        # The runner owns final commit. Upgrade before both format and source
        # validation; a stale reader cannot replace a newer generation.
        connection.execute("UPDATE messages SET sender=sender WHERE 0")
        if _canonical_source:
            try:
                _validate_routed_style_source(connection)
            except _StyleAppendChanged as error:
                raise LayerDeferredError("Canonical source readiness changed", category=error.category) from error
        if _style_extra_triggers(connection):
            raise LayerDeferredError("Style triggers prevent tracked publication", category="StyleTriggersUnsupported")
        style._publish_style_vectors(connection, schema, query, rows, stored_rows,
                                     check_cancel=check_cancel, before_replace=compatible)
        if read_source_revision(connection) != source:
            raise style._StyleSourceChanged("Style source changed before publication")
        connection.execute("INSERT INTO metadata(key,value) VALUES ('style_vec_hash_version','2') "
                           "ON CONFLICT(key) DO UPDATE SET value=excluded.value")
        return result

    return LayerSpec("style_vectors", "style_vectors", resolve_dependency, build,
        lambda connection: connection.execute("SELECT count(*) FROM entity_style_vectors").fetchone()[0],
        connection=conn, execution_guard=execution_guard)


def run_style_maintenance(
    conn: sqlite3.Connection, *, force: bool = False, threshold: int = 25,
    cancel: threading.Event | None = None, allow_caller_transaction: bool = False,
    connection_owned: bool = False, _canonical_source: bool = False,
) -> LayerResult:
    """Explicit tracked style rebuild, separate from all eight public adapters."""
    if type(allow_caller_transaction) is not bool:
        raise ValueError("Caller transaction opt-in must be a boolean")
    if type(connection_owned) is not bool:
        raise ValueError("Connection ownership opt-in must be a boolean")
    borrowed = conn.in_transaction
    if borrowed and not allow_caller_transaction:
        raise RuntimeError("Maintenance runner requires a clean transaction boundary")
    if type(threshold) is not int or threshold < 1:
        raise ValueError("Maintenance threshold must be a positive integer")
    with maintenance_owner(connection_database_path(conn)):
        started = time.monotonic()

        def preparation_failure(error: sqlite3.OperationalError) -> None:
            if _style_sqlite_deferral(error) == "StyleWriterBusy":
                raise LayerDeferredError("Style preparation writer busy", category="StyleWriterBusy") from error

        try:
            if _canonical_source:
                _prepare_routed_style_maintenance(conn)
            else:
                prepare_style_maintenance(conn, _on_safe_failure=preparation_failure)
        except _StyleAppendChanged as error:
            return LayerResult("style_vectors", "style_vectors", "deferred", None,
                time.monotonic() - started, error.category, False, "unverified", borrowed)
        except LayerDeferredError as error:
            return LayerResult("style_vectors", "style_vectors", "deferred", None,
                time.monotonic() - started, error.category, False, "unverified", borrowed)
        routed = {"_canonical_source": True} if _canonical_source else {}
        spec = style_layer_spec(conn, cancel=cancel, connection_owned=connection_owned, **routed)
        try:
            results = run_layers(conn, (spec,), force=force, threshold=threshold, cancel=cancel,
                                 allow_caller_transaction=allow_caller_transaction, **routed)
            if results:
                return results[0]
            observed = _read_layer_run_baseline(conn, (spec,), **routed)
        except _StyleAppendChanged as error:
            if not _canonical_source or error.category not in {
                "StyleSourceChanged", "StyleTrackingUnprepared", "StyleTriggersUnsupported",
                "StyleMetadataChanged", "StyleAccumulatorUnsupported", "StyleHashPending",
            }:
                raise
            if conn.in_transaction != borrowed:
                raise _LayerRollbackFailed("Canonical readiness rejection changed transaction ownership") from error
            return LayerResult("style_vectors", "style_vectors", "deferred", None,
                               time.monotonic() - started, error.category, False, "unverified", borrowed)
        if observed is None:
            return _style_baseline_busy_result(spec, borrowed)
        state = observed[spec.layer]
        return LayerResult(spec.layer, spec.result_key, "deferred", state.output_count, 0.0,
                           "StyleCancelled", False, state.successful_coverage, borrowed)


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
    from truememory.tier_switch.runtime import _read_selection, _read_policy
    if _read_selection(conn) is not None or _read_policy(conn) is not None:
        return LayerDependency("", 1, False, "SelectedVectorLayerUnsupported", deferred=True)
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
    style_result: LayerResult | None = None


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


@contextmanager
def _maintenance_completion(conn: sqlite3.Connection, allow_caller_transaction: bool) -> Iterator[None]:
    """Validate successful owned work before its connection mutation lock leaves."""
    if type(allow_caller_transaction) is not bool:
        raise ValueError("Caller transaction opt-in must be a boolean")
    borrowed = conn.in_transaction
    if borrowed and not allow_caller_transaction:
        raise RuntimeError("Maintenance runner requires a clean transaction boundary")
    yield
    if not borrowed and conn.in_transaction:
        conn.rollback()
        raise RuntimeError("Maintenance work left an unfinished transaction")


def run_engine_maintenance(
    conn: sqlite3.Connection, coordinator: MaintenanceCoordinator, *, threshold: int = 25,
    force: bool = False, cancel: threading.Event | None = None,
    prepare_extensions: bool = True, allow_caller_transaction: bool = False,
    include_layers: bool = True, include_style: bool = False, connection_owned: bool = False,
    _connection_lock: object | None = None, deadline: float | None = None,
) -> MaintenanceReport:
    """Use only the supplied owned worker/original explicitly borrowed handle."""
    if cancel is not None and cancel.is_set():
        # Keep this cancellation snapshot terminal even if another caller
        # clears the event before the optional style result is constructed.
        style = (LayerResult("style_vectors", "style_vectors", "deferred", None, 0.0,
                             "StyleCancelled", False, "unverified", conn.in_transaction)
                 if include_style else None)
        return MaintenanceReport((), "CANCELLED", style)
    from truememory.tier_switch.runtime import _connection_read
    with _connection_read(conn, _connection_lock, deadline, cancel):
        path = connection_database_path(conn)
    with maintenance_owner(path), _maintenance_embedding_scope(
        conn, include_layers, cancel, connection_lock=_connection_lock, deadline=deadline,
    ) as supported, _connection_read(conn, _connection_lock, deadline, cancel), \
            _maintenance_completion(conn, allow_caller_transaction):
        style_result = None
        if include_style:
            style_result = run_routed_style(conn, force=force, threshold=threshold, cancel=cancel,
                allow_caller_transaction=allow_caller_transaction, connection_owned=connection_owned)
        if not include_layers or (cancel is not None and cancel.is_set()):
            return MaintenanceReport((), "CANCELLED" if cancel is not None and cancel.is_set() else
                                     "SKIPPED (style-only)", style_result)
        evidence = _prepare_worker_extensions(conn, coordinator) if prepare_extensions and supported else None
        specs = (engine_layer_specs(conn, coordinator, evidence=evidence) if supported else
                 tuple(spec for spec in all_layer_specs(conn) if spec.layer != "clusters"))
        results = run_layers(conn, specs, threshold=threshold, force=force, cancel=cancel,
                             allow_caller_transaction=allow_caller_transaction)
        if not supported:
            results = (_selected_cluster_deferred(conn), *results)
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
        return MaintenanceReport(results, preferences, style_result)


def maintenance_report_status(report: MaintenanceReport, cancel: threading.Event | None = None) -> tuple[str, str | None]:
    if cancel is not None and cancel.is_set():
        return "cancelled", "Cancelled"
    results = report.results + ((report.style_result,) if report.style_result is not None else ())
    for outcome in ("failed", "unavailable", "deferred", "abandoned"):
        for result in results:
            if result.outcome == outcome:
                return outcome, result.error_category
    if report.preferences.startswith(("ERROR", "UNAVAILABLE")):
        return "failed", "PreferencesUnavailable"
    if any(result.pending_caller_commit or result.outcome not in {"success", "success_empty", "current"} for result in results):
        return "pending", None
    if any(result.coverage != "complete" for result in results):
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
