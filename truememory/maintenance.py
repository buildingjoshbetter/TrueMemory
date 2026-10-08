"""Database maintenance ownership and committed source revision primitives.

The engine scheduler is not routed here yet. No models, polling loop or automatic
builder invocation is introduced by importing this module.
"""

import importlib
import json
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


class LayerDependency(NamedTuple):
    key: str
    builder_version: int
    available: bool
    error_category: str | None = None


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


class LayerResult(NamedTuple):
    layer: str
    result_key: str
    outcome: str
    output_count: int | None
    elapsed_seconds: float
    error_category: str | None
    attempted: bool


def _validate_layer(layer: str) -> None:
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", layer) is None:
        raise ValueError("Invalid maintenance layer name")


def layer_freshness(source: SourceRevision, state: LayerState, dependency: LayerDependency) -> str:
    """Separate a prior valid success from the latest attempt's outcome."""
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
            "l.run_generation,l.error_category FROM maintenance_source_state s LEFT JOIN maintenance_layers l "
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
        values = row[8:] if row is not None else ("pending", None, None, None, None, None, None, None, 1, None, None, None)
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
    return tuple(spec for spec in specs if states[spec.layer].outcome == "running"
                 or _eligible(states[spec.layer], threshold, force))


def record_layer_success_in_transaction(
    conn: sqlite3.Connection, *, layer: str, dependency: LayerDependency,
    source: SourceRevision, output_count: int, run_generation: str | None = None,
) -> None:
    """Publish provenance beside completed output, without owning caller commit."""
    _validate_layer(layer)
    if not conn.in_transaction:
        raise RuntimeError("Layer publication requires an existing transaction")
    if not dependency.available or type(output_count) is not int or output_count < 0:
        raise ValueError("Successful layer publication requires available output")
    if read_source_revision(conn) != source:
        raise sqlite3.OperationalError("Maintenance source changed before publication")
    conn.execute(
        "INSERT INTO maintenance_layers(layer,builder_version,outcome,successful_epoch,successful_revision,"
        "successful_dependency,attempted_epoch,attempted_revision,attempted_dependency,attempted_insert_count,"
        "full_rebuild_required,output_count,run_generation,error_category) VALUES (?,?,?,?,?,?,?,?,?,?,0,?,?,NULL) "
        "ON CONFLICT(layer) DO UPDATE SET builder_version=excluded.builder_version,outcome=excluded.outcome,"
        "successful_epoch=excluded.successful_epoch,successful_revision=excluded.successful_revision,"
        "successful_dependency=excluded.successful_dependency,attempted_epoch=excluded.attempted_epoch,"
        "attempted_revision=excluded.attempted_revision,attempted_dependency=excluded.attempted_dependency,"
        "attempted_insert_count=excluded.attempted_insert_count,full_rebuild_required=0,"
        "output_count=excluded.output_count,run_generation=excluded.run_generation,error_category=NULL",
        (layer, dependency.builder_version, "success" if output_count else "success_empty", source.epoch,
         source.revision, dependency.key, source.epoch, source.revision, dependency.key,
         source.insert_count, output_count, run_generation),
    )


@contextmanager
def _owned_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    if conn.in_transaction:
        raise RuntimeError("Maintenance runner requires a clean transaction boundary")
    conn.execute("BEGIN")
    completed = False
    try:
        yield
        conn.execute("COMMIT")
        completed = True
    finally:
        if not completed and conn.in_transaction:
            conn.rollback()


def _record_attempt(
    conn: sqlite3.Connection, state: LayerState, outcome: str, generation: str, error_category: str | None,
) -> None:
    source, dependency = state.source, state.dependency
    with _owned_transaction(conn):
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


def run_layers(
    conn: sqlite3.Connection, specs: tuple[LayerSpec, ...], *, force: bool = False,
    threshold: int = 25, cancel: threading.Event | None = None,
) -> tuple[LayerResult, ...]:
    """Run each eligible trusted adapter once on a dedicated connection."""
    if conn.in_transaction:
        raise RuntimeError("Maintenance runner requires a clean transaction boundary")
    if type(threshold) is not int or threshold < 1:
        raise ValueError("Maintenance threshold must be a positive integer")
    results = []
    with maintenance_owner(connection_database_path(conn)) as owner:
        if conn.execute(
            "SELECT 1 FROM maintenance_layers WHERE outcome='running' AND run_generation IS NOT ? LIMIT 1",
            (owner.generation,),
        ).fetchone():
            with _owned_transaction(conn):
                conn.execute(
                    "UPDATE maintenance_layers SET outcome='abandoned',error_category='Interrupted' "
                    "WHERE outcome='running' AND run_generation IS NOT ?", (owner.generation,),
                )
        # Validate uniqueness before any builder is invoked.
        read_layer_states(conn, specs)
        for spec in specs:
            if cancel is not None and cancel.is_set():
                break
            state = read_layer_states(conn, (spec,))[spec.layer]
            if not _eligible(state, threshold, force):
                results.append(LayerResult(spec.layer, spec.result_key,
                    layer_freshness(state.source, state, state.dependency), state.output_count, 0.0,
                    state.error_category, False))
                continue
            started = time.monotonic()
            if not state.dependency.available:
                _record_attempt(conn, state, "unavailable", owner.generation, state.dependency.error_category)
                results.append(LayerResult(spec.layer, spec.result_key, "unavailable", state.output_count,
                    time.monotonic() - started, state.dependency.error_category, True))
                continue
            # This diagnostic commits before the read/compute snapshot starts.
            with _owned_transaction(conn):
                conn.execute(
                    "INSERT INTO maintenance_layers(layer,builder_version,outcome,run_generation,error_category) "
                    "VALUES (?,?,'running',?,NULL) ON CONFLICT(layer) DO UPDATE SET "
                    "builder_version=excluded.builder_version,outcome='running',"
                    "run_generation=excluded.run_generation,error_category=NULL",
                    (spec.layer, state.dependency.builder_version, owner.generation),
                )
            count = None
            try:
                # Factories capture identity before the attempt's read snapshot.
                # The deferred guard enters only at publication, then outlives
                # the transaction so both COMMIT and rollback retain ownership.
                guard = spec.publication_guard(conn) if spec.publication_guard is not None else None
                with ExitStack() as publication:
                    with _owned_transaction(conn):
                        state = read_layer_states(conn, (spec,))[spec.layer]
                        if not state.dependency.available:
                            raise MaintenanceUnavailableError("Layer dependency became unavailable")
                        spec.build(conn)
                        if not conn.in_transaction:
                            raise RuntimeError("Layer builder committed the runner transaction")
                        count = spec.count_output(conn)
                        if cancel is not None and cancel.is_set():
                            raise _LayerCancelled()
                        if spec.resolve_dependency() != state.dependency:
                            raise MaintenanceUnavailableError("Layer dependency changed before publication")
                        record_layer_success_in_transaction(conn, layer=spec.layer, dependency=state.dependency,
                            source=state.source, output_count=count, run_generation=owner.generation)
                        if spec.resolve_dependency() != state.dependency:
                            raise MaintenanceUnavailableError("Layer dependency changed before commit")
                        if guard is not None:
                            # Register release before validation can fail, so
                            # validation errors also roll back under ownership.
                            validate_publication = publication.enter_context(guard)
                            validate_publication()
                        if cancel is not None and cancel.is_set():
                            raise _LayerCancelled()
            except BaseException as error:
                interrupted = not isinstance(error, Exception) or isinstance(error, _LayerCancelled)
                outcome = "abandoned" if interrupted else ("failed" if state.dependency.available else "unavailable")
                category = "Cancelled" if interrupted else type(error).__name__[:64]
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", category) is None:
                    category = "Error"
                _record_attempt(conn, state, outcome, owner.generation, category)
                results.append(LayerResult(spec.layer, spec.result_key, outcome, state.output_count,
                    time.monotonic() - started, category, True))
                if not isinstance(error, Exception):
                    raise
                if interrupted:
                    break
            else:
                results.append(LayerResult(spec.layer, spec.result_key, "success" if count else "success_empty",
                    count, time.monotonic() - started, None, True))
    return tuple(results)


def nonvector_layer_specs() -> tuple[LayerSpec, ...]:
    """Lazy standard-library adapters; no vector or Dunbar enrollment yet."""
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
