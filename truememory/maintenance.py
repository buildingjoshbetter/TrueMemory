"""Database maintenance ownership and committed source revision primitives.

The engine scheduler is not routed here yet. No models, polling loop or automatic
builder invocation is introduced by importing this module.
"""

import os
import sqlite3
import threading
import uuid
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from truememory._platform import try_file_lock
from truememory.storage import create_db


class MaintenanceUnavailableError(RuntimeError):
    """Source tracking is unavailable; a freshness token cannot be issued."""


class MaintenanceBusyError(RuntimeError):
    """Another operation owns maintenance for this database."""


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
_thread_owners = threading.local()


def _acquire_owner(path: Path) -> int | None:
    with _registry_lock:
        if path in _held_paths:
            return None
        fd = try_file_lock(Path(str(path) + ".maintenance.lock"))
        if fd is not None:
            _held_fds.add(fd)
            _held_paths[path] = fd
        return fd


def _release_owner(fd: int) -> None:
    with _registry_lock:
        os.close(fd)
        _held_fds.discard(fd)
        for path, claimed_fd in tuple(_held_paths.items()):
            if claimed_fd == fd:
                del _held_paths[path]


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
    global _registry_lock, _coordinators, _held_fds, _held_paths, _thread_owners
    # The child must not prolong its parent's ownership by retaining an
    # inherited descriptor. Inherited coordinator objects reject child use.
    for fd in _held_fds:
        os.close(fd)
    _held_fds = set()
    _held_paths = {}
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

    def _check_process(self) -> None:
        if self._pid != os.getpid():
            raise RuntimeError("Get a new maintenance coordinator after process fork")

    @property
    def status(self) -> tuple[str, str | None]:
        self._check_process()
        with self._mutex:
            return self._status, self._error_category

    def request(self, work: Callable[[sqlite3.Connection, threading.Event], None]) -> bool:
        """Start one owned worker, or report busy/in-memory pending without retry."""
        self._check_process()
        with self._mutex:
            if self.path is None:
                self._status = "pending_in_memory"
                return False
            if self._active:
                return False
            self._active = True
            self._done.clear()
            self._cancel.clear()
            self._status = "starting"
            self._error_category = None

        fd = None
        handoff = None
        handoff_lock = threading.Lock()
        try:
            # Serializing descriptor acquisition with fork registration makes
            # every inherited owner descriptor visible to the child cleanup.
            fd = _acquire_owner(self.path)
            if fd is None:
                with self._mutex:
                    self._active = False
                    self._status = "busy"
                    self._done.set()
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
                self._cancel.set()
                raise
            if fd is not None:
                _release_owner(fd)
            with self._mutex:
                self._active = False
                self._thread = None
                self._status = "failed"
                self._error_category = "worker_start"
                self._done.set()
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
                        work(conn, self._cancel)
                        if conn.in_transaction:
                            conn.rollback()
                            raise RuntimeError("Maintenance work left an unfinished transaction")
                        outcome = "cancelled" if self._cancel.is_set() else "success"
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
                self._status = outcome
                self._error_category = error_category
                self._done.set()

    def cancel(self) -> None:
        """Signal a phase boundary; never release ownership of active work."""
        self._check_process()
        self._cancel.set()

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
