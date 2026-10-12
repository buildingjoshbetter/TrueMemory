"""Immutable background job identity for a cooperatively stable database path.

This is an admission guard, not activation or a lock against arbitrary pathname
replacement. Capture requires a known-current caller connection. SQLite's Python
API exposes no database file descriptor to prove that historical association.
"""

from __future__ import annotations

import os
import re
import sqlite3
import stat
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from truememory.embedding_target import EmbeddingTarget
from truememory.maintenance import MaintenanceUnavailableError, maintenance_owner
from truememory.rebuild_source import read_rebuild_revision


class TierJobIdentityError(RuntimeError):
    """The background database identity cannot be established safely."""


@dataclass(frozen=True)
class TierJob:
    job_id: str
    target: EmbeddingTarget
    database_path: Path = field(repr=False)
    file_identity: tuple[int, int]
    tracker: str
    source_epoch: str
    source_schema_signature: str

    def __post_init__(self) -> None:
        if (not isinstance(self.job_id, str) or re.fullmatch(r"[0-9a-f]{32}", self.job_id) is None
                or not isinstance(self.target, EmbeddingTarget)
                or not isinstance(self.database_path, Path) or not self.database_path.is_absolute()
                or type(self.file_identity) is not tuple or len(self.file_identity) != 2
                or any(type(value) is not int for value in self.file_identity)
                or self.file_identity[0] < 0 or self.file_identity[1] <= 0
                or self.tracker not in {"canonical-v1", "rebuild-bridge-v1", "rebuild-bridge-conservative-v1"}
                or not isinstance(self.source_epoch, str) or not self.source_epoch
                or not isinstance(self.source_schema_signature, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.source_schema_signature) is None):
            raise TierJobIdentityError("Invalid background tier job descriptor")


@contextmanager
def _observe_file(path: Path) -> Iterator[tuple[int, int]]:
    # Use fstat on every observation, including a fresh pathname reopen.
    # ctime/mtime/size can change during legitimate writes and are not identity.
    with ExitStack() as stack:
        try:
            flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
            descriptor = os.open(path, flags)
            stack.callback(os.close, descriptor)
            value = os.fstat(descriptor)
            if (not stat.S_ISREG(value.st_mode) or type(value.st_dev) is not int
                    or type(value.st_ino) is not int or value.st_dev < 0 or value.st_ino <= 0):
                raise TierJobIdentityError("Stable database file identity is unavailable")
        except (OSError, ValueError):
            raise TierJobIdentityError("Database file identity is unavailable") from None
        yield value.st_dev, value.st_ino


def _connection_path(conn: sqlite3.Connection) -> Path:
    if conn.in_transaction:
        raise TierJobIdentityError("Background identity requires a clean connection")
    databases = tuple(conn.execute("PRAGMA database_list"))
    mains = [row[2] for row in databases if row[1] == "main"]
    if len(mains) != 1 or not mains[0] or any(row[1] not in {"main", "temp"} for row in databases):
        raise TierJobIdentityError("Background identity requires one existing file database")
    # The tracker uses unqualified source tables. Refuse every TEMP object so
    # temporary tables, views and triggers cannot redirect identity reads.
    if conn.execute("SELECT 1 FROM sqlite_temp_master LIMIT 1").fetchone() is not None:
        raise TierJobIdentityError("Temporary database objects make background identity ambiguous")
    path = Path(mains[0]).expanduser().resolve(strict=True)
    return path


def _source_identity(conn: sqlite3.Connection) -> tuple[str, str, str]:
    conn.execute("BEGIN")
    try:
        tracker, revision, signature = read_rebuild_revision(conn)
        return tracker, revision.epoch, signature
    finally:
        conn.rollback()
        if conn.in_transaction:
            raise TierJobIdentityError("Background identity could not release its read transaction")


def capture_tier_job(conn: sqlite3.Connection, target: EmbeddingTarget) -> TierJob:
    """Capture a clean, known-current connection with tracking already installed.

    No connection or ownership token is retained. Private/shared-memory databases
    cannot use this background reopen protocol. Caller owns stable-path admission.
    """
    try:
        path = _connection_path(conn)
        with _observe_file(path) as identity:
            tracker, epoch, signature = _source_identity(conn)
            with _observe_file(path) as after:
                if identity != after or _connection_path(conn) != path:
                    raise TierJobIdentityError("Database identity changed during job capture")
        return TierJob(uuid.uuid4().hex, target, path, identity, tracker, epoch, signature)
    except (OSError, ValueError, sqlite3.Error, MaintenanceUnavailableError):
        raise TierJobIdentityError("Background database identity could not be captured") from None


def check_tier_job(conn: sqlite3.Connection, job: TierJob) -> None:
    """Recheck before model admission or a future activation transaction.

    This short read snapshot ends here. Source-content freshness, current job
    selection and final activation serialization remain separate caller duties.
    """
    try:
        if _connection_path(conn) != job.database_path:
            raise TierJobIdentityError("Background database path changed")
        with _observe_file(job.database_path) as before:
            if before != job.file_identity:
                raise TierJobIdentityError("Background database file changed")
            if _source_identity(conn) != (job.tracker, job.source_epoch, job.source_schema_signature):
                raise TierJobIdentityError("Background database source identity changed")
            with _observe_file(job.database_path) as after:
                if before != after or _connection_path(conn) != job.database_path:
                    raise TierJobIdentityError("Background database identity changed during validation")
    except (OSError, ValueError, sqlite3.Error, MaintenanceUnavailableError):
        raise TierJobIdentityError("Background database identity could not be validated") from None


def _close_connection(conn: sqlite3.Connection) -> None:
    try:
        conn.close()
    except (OSError, sqlite3.Error):
        raise TierJobIdentityError("Background database connection could not be closed") from None


@contextmanager
def open_tier_job(job: TierJob) -> Iterator[sqlite3.Connection]:
    """Own and reopen an existing database in this worker thread, never create it.

    The connection and cooperative maintenance owner both end on BaseException.
    The descriptor conveys no caller-thread maintenance ownership.
    """
    with ExitStack() as stack:
        try:
            stack.enter_context(maintenance_owner(job.database_path))
            identity = stack.enter_context(_observe_file(job.database_path))
            if identity != job.file_identity:
                raise TierJobIdentityError("Background database file changed before reopen")
            conn = sqlite3.connect(job.database_path.as_uri() + "?mode=rw", uri=True)
            stack.callback(_close_connection, conn)
            check_tier_job(conn, job)
        except (OSError, ValueError, sqlite3.Error, MaintenanceUnavailableError):
            raise TierJobIdentityError("Background database could not be reopened") from None
        yield conn


_SELECTION_TABLE = "truememory_tier_selected_job_v1"
_SELECTION_SQL = f"""CREATE TABLE {_SELECTION_TABLE} (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    job_id TEXT NOT NULL CHECK (typeof(job_id) = 'text' AND length(job_id) = 32),
    state TEXT NOT NULL CHECK (state IN ('selected', 'cancelled', 'failed')),
    tier TEXT NOT NULL CHECK (tier IN ('edge', 'base', 'pro', 'custom')),
    model_id TEXT NOT NULL CHECK (typeof(model_id) = 'text' AND length(model_id) BETWEEN 1 AND 512 AND length(CAST(model_id AS BLOB)) <= 2048),
    dimension INTEGER NOT NULL CHECK (typeof(dimension) = 'integer' AND dimension BETWEEN 1 AND 4096),
    tier_group TEXT NOT NULL CHECK (tier_group IN ('edge', 'basepro', 'custom')),
    tracker TEXT NOT NULL CHECK (tracker IN ('canonical-v1', 'rebuild-bridge-v1', 'rebuild-bridge-conservative-v1')),
    source_epoch TEXT NOT NULL CHECK (typeof(source_epoch) = 'text' AND length(source_epoch) BETWEEN 1 AND 128 AND length(CAST(source_epoch AS BLOB)) <= 512),
    source_schema_signature TEXT NOT NULL CHECK (typeof(source_schema_signature) = 'text' AND length(source_schema_signature) = 64)
)"""
_SELECTION_FIELDS = ("job_id", "tier", "model_id", "dimension", "tier_group", "tracker",
                     "source_epoch", "source_schema_signature")
_SELECTION_BOUNDS = (32, 6, 512, None, 7, 30, 128, 64)


class TierJobSelectionError(TierJobIdentityError):
    """The durable selected job is unavailable or does not match this request."""


@dataclass(frozen=True)
class SelectedJobMarker:
    job_id: str
    target: EmbeddingTarget
    tracker: str
    source_epoch: str
    source_schema_signature: str
    state: str


def _marker_in_snapshot(conn: sqlite3.Connection) -> SelectedJobMarker | None:
    if conn.execute("SELECT 1 FROM temp.sqlite_master LIMIT 1").fetchone() is not None:
        raise TierJobSelectionError("TEMP objects make selected marker identity ambiguous")
    shape = conn.execute(
        "SELECT typeof(sql), length(CAST(sql AS BLOB)) FROM main.sqlite_master "
        "WHERE name=? COLLATE NOCASE", (_SELECTION_TABLE,),
    ).fetchone()
    if shape is None:
        return None
    if shape[0] != "text" or not 1 <= shape[1] <= 8192:
        raise TierJobSelectionError("Selected job schema exceeds its protocol bounds")
    _selection_schema(conn)
    row = _selection_row(conn)
    return None if row is None else SelectedJobMarker(
        row[0], EmbeddingTarget(row[1], row[2], row[3], row[4]), row[5], row[6], row[7], row[8],
    )


def read_selected_job_marker(conn: sqlite3.Connection) -> SelectedJobMarker | None:
    """Bounded marker read; borrow a caller snapshot without ending it.

    This is not a reopened-file identity certificate or permission to resume a
    disappeared worker. Terminal IDs permit explicit normal job replacement.
    """
    if conn.in_transaction:
        return _marker_in_snapshot(conn)
    with _selection_transaction(conn, write=False):
        return _marker_in_snapshot(conn)


def _cancel_policy_job_in_writer(
    conn: sqlite3.Connection, *, job_id: str, target: EmbeddingTarget, tracker: str,
    source_epoch: str, source_schema_signature: str,
) -> None:
    """Cancel only the exact superseded intent's marker; caller owns writer.

    Maintenance/stable-path ownership and writer acquisition are caller duties.
    No source read, file observation, DELETE, native call, commit or rollback.
    """
    if not conn.in_transaction:
        raise TierJobSelectionError("Policy supersession requires writer ownership")
    row = _marker_in_snapshot(conn)
    if row is None or (row.job_id, row.target, row.tracker, row.source_epoch, row.source_schema_signature) != (
        job_id, target, tracker, source_epoch, source_schema_signature,
    ):
        raise TierJobSelectionError("Superseded activation job changed")
    if row.state == "selected":
        changed = conn.execute(
            f"UPDATE main.{_SELECTION_TABLE} SET state='cancelled' "
            "WHERE singleton=1 AND job_id=? AND state='selected'", (job_id,),
        ).rowcount
        if changed != 1:
            raise TierJobSelectionError("Superseded activation job changed during cancellation")


def _selection_values(job: TierJob) -> tuple[str | int, ...]:
    values = (job.job_id, job.target.tier, job.target.model_id, job.target.dimension,
              job.target.tier_group, job.tracker, job.source_epoch, job.source_schema_signature)
    for value, bound in zip(values, _SELECTION_BOUNDS):
        if bound is None:
            if type(value) is not int or not 1 <= value <= 4096:
                raise TierJobSelectionError("Invalid selected job identity")
        elif type(value) is not str or not 1 <= len(value) <= bound:
            raise TierJobSelectionError("Selected job identity exceeds its protocol bounds")
    return values


def _selection_schema(conn: sqlite3.Connection, *, create: bool = False) -> None:
    row = conn.execute(
        "SELECT type, sql FROM main.sqlite_master WHERE name = ? COLLATE NOCASE", (_SELECTION_TABLE,),
    ).fetchone()
    if row is None and create:
        conn.execute(_SELECTION_SQL)
    elif (row is None or row[0] != "table" or type(row[1]) is not str
          or " ".join(row[1].strip().rstrip(";").split()) != " ".join(_SELECTION_SQL.split())):
        raise TierJobSelectionError("Selected job storage has an unsupported schema")
    if conn.execute(
        "SELECT 1 FROM main.sqlite_master WHERE tbl_name = ? COLLATE NOCASE "
        "AND type IN ('index', 'trigger') LIMIT 1", (_SELECTION_TABLE,),
    ).fetchone() is not None:
        raise TierJobSelectionError("Selected job storage has unsupported side effects")


def _selection_row(conn: sqlite3.Connection) -> tuple[str | int, ...] | None:
    identifiers = [tuple(row) for row in conn.execute(f"SELECT singleton FROM main.{_SELECTION_TABLE} LIMIT 2")]
    if not identifiers:
        return None
    if identifiers != [(1,)]:
        raise TierJobSelectionError("Selected job storage contains an invalid singleton")
    # Read only sizes/types before copying bounded marker values into Python.
    probes = ", ".join(f"typeof({name}), length(CAST({name} AS BLOB))" for name in (*_SELECTION_FIELDS, "state"))
    metadata = conn.execute(f"SELECT {probes} FROM main.{_SELECTION_TABLE} WHERE singleton = 1").fetchone()
    if metadata is None:
        raise TierJobSelectionError("Selected job storage changed during validation")
    for index, bound in enumerate((*_SELECTION_BOUNDS, 9)):
        kind, length = metadata[index * 2:index * 2 + 2]
        if ((bound is None and kind != "integer")
                or (bound is not None and (kind != "text" or type(length) is not int or not 1 <= length <= 4 * bound))):
            raise TierJobSelectionError("Selected job storage contains invalid values")
    fields = ", ".join((*_SELECTION_FIELDS, "state"))
    row = tuple(conn.execute(f"SELECT {fields} FROM main.{_SELECTION_TABLE} WHERE singleton = 1").fetchone())
    if any(bound is not None and (type(value) is not str or not 1 <= len(value) <= bound)
           for value, bound in zip(row, (*_SELECTION_BOUNDS, 9))):
        raise TierJobSelectionError("Selected job storage contains oversized values")
    if (re.fullmatch(r"[0-9a-f]{32}", row[0]) is None
            or re.fullmatch(r"[0-9a-f]{64}", row[7]) is None
            or row[5] not in {"canonical-v1", "rebuild-bridge-v1", "rebuild-bridge-conservative-v1"}
            or row[8] not in {"selected", "cancelled", "failed"}):
        raise TierJobSelectionError("Selected job storage contains an invalid identity")
    EmbeddingTarget(row[1], row[2], row[3], row[4])
    return row


def _rollback_selection(conn: sqlite3.Connection) -> None:
    try:
        conn.rollback()
    except (OSError, sqlite3.Error):
        raise TierJobSelectionError("Selected job transaction rollback is uncertain") from None
    if conn.in_transaction:
        raise TierJobSelectionError("Selected job transaction rollback is uncertain")


@contextmanager
def _selection_transaction(conn: sqlite3.Connection, *, write: bool) -> Iterator[None]:
    if conn.in_transaction:
        raise TierJobSelectionError("Selected job admission requires a clean connection")
    conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
    try:
        yield
        if write:
            conn.commit()
            if conn.in_transaction:
                raise TierJobSelectionError("Selected job transaction did not commit")
    except BaseException:
        if conn.in_transaction:
            _rollback_selection(conn)
        raise
    else:
        if not write:
            _rollback_selection(conn)


def _selection_source(conn: sqlite3.Connection, job: TierJob) -> None:
    databases = tuple(conn.execute("PRAGMA database_list"))
    mains = [row[2] for row in databases if row[1] == "main"]
    if (len(mains) != 1 or not mains[0]
            or any(row[1] not in {"main", "temp"} for row in databases)
            or Path(mains[0]).expanduser().resolve(strict=True) != job.database_path):
        raise TierJobSelectionError("Selected job database path changed")
    if conn.execute("SELECT 1 FROM sqlite_temp_master LIMIT 1").fetchone() is not None:
        raise TierJobSelectionError("Temporary database objects make selected job identity ambiguous")
    tracker, revision, signature = read_rebuild_revision(conn)
    if (tracker, revision.epoch, signature) != (job.tracker, job.source_epoch, job.source_schema_signature):
        raise TierJobSelectionError("Selected job source identity changed")
    with _observe_file(job.database_path) as identity:
        if identity != job.file_identity:
            raise TierJobSelectionError("Selected job database file changed")


def select_tier_job(
    conn: sqlite3.Connection, target: EmbeddingTarget, *, previous_job_id: str | None = None,
) -> TierJob:
    """Commit a new selection before planning, without replacing a live job.

    A terminal marker can be replaced only by explicitly naming its previous ID.
    The new ID is written through the accepted connection, not a pathname reopen.
    """
    if previous_job_id is not None and (type(previous_job_id) is not str
            or re.fullmatch(r"[0-9a-f]{32}", previous_job_id) is None):
        raise TierJobSelectionError("Invalid previous selected job identity")
    job = capture_tier_job(conn, target)
    values = _selection_values(job)
    try:
        with _selection_transaction(conn, write=True):
            _selection_source(conn, job)
            _selection_schema(conn, create=True)
            previous = _selection_row(conn)
            if previous is None:
                if previous_job_id is not None:
                    raise TierJobSelectionError("Previous selected job is no longer present")
                fields = ", ".join(_SELECTION_FIELDS)
                conn.execute(
                    f"INSERT INTO main.{_SELECTION_TABLE}(singleton, {fields}, state) "
                    "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 'selected')", values,
                )
            else:
                if previous[0] != previous_job_id or previous[8] == "selected":
                    raise TierJobSelectionError("Another job already owns the selection")
                assignments = ", ".join(f"{name} = ?" for name in _SELECTION_FIELDS)
                changed = conn.execute(
                    f"UPDATE main.{_SELECTION_TABLE} SET {assignments}, state = 'selected' "
                    "WHERE singleton = 1 AND job_id = ? AND state IN ('cancelled', 'failed')",
                    (*values, previous_job_id),
                ).rowcount
                if changed != 1:
                    raise TierJobSelectionError("Selected job changed during replacement")
            _selection_source(conn, job)
        return job
    except (OSError, ValueError, sqlite3.Error, MaintenanceUnavailableError):
        raise TierJobSelectionError("Background job could not be selected") from None


def check_selected_tier_job(conn: sqlite3.Connection, job: TierJob) -> None:
    """Read a coherent source/marker snapshot; return with no transaction open."""
    values = _selection_values(job)
    try:
        check_tier_job(conn, job)
        with _selection_transaction(conn, write=False):
            _selection_source(conn, job)
            _selection_schema(conn)
            if _selection_row(conn) != (*values, "selected"):
                raise TierJobSelectionError("Background job is no longer selected")
            _selection_source(conn, job)
    except (OSError, ValueError, sqlite3.Error, MaintenanceUnavailableError):
        raise TierJobSelectionError("Selected background job could not be validated") from None


def check_selected_tier_job_in_writer(conn: sqlite3.Connection, job: TierJob) -> None:
    """Read only inside the caller's owned BEGIN IMMEDIATE transaction.

    Python sqlite3 exposes whether a transaction exists, not its lock kind.
    The caller must establish writer ownership; this helper neither acquires it
    nor commits/rolls back caller work. Stable-path/maintenance ownership is
    still required, just as for the public clean-connection check.
    """
    if not conn.in_transaction:
        raise TierJobSelectionError("Selected job certification requires caller writer ownership")
    values = _selection_values(job)
    shape = conn.execute(
        "SELECT typeof(sql), length(CAST(sql AS BLOB)) FROM main.sqlite_master "
        "WHERE name = ? COLLATE NOCASE", (_SELECTION_TABLE,),
    ).fetchone()
    if shape is None or shape[0] != "text" or not 1 <= shape[1] <= 8192:
        raise TierJobSelectionError("Selected job schema exceeds its protocol bounds")
    _selection_source(conn, job)
    _selection_schema(conn)
    if _selection_row(conn) != (*values, "selected"):
        raise TierJobSelectionError("Background job is no longer selected")
    _selection_source(conn, job)


def _retire_selected_tier_job_in_writer(conn: sqlite3.Connection, job: TierJob) -> None:
    """Retire this exact successful marker only within certified selection SQL.

    Reject incoming foreign keys before DELETE, including when enforcement is
    disabled on this connection. The bounded schema walk never copies table
    definitions or unbounded names into Python. No commit or rollback occurs.
    """
    check_selected_tier_job_in_writer(conn, job)
    count = 0
    for (name,) in conn.execute(
        "SELECT CASE WHEN typeof(name)='text' AND length(CAST(name AS BLOB)) BETWEEN 1 AND 512 "
        "THEN name END FROM main.sqlite_master WHERE type='table' LIMIT 4097",
    ):
        count += 1
        if name is None or count > 4096:
            raise TierJobSelectionError("Database schema exceeds safe marker retirement bounds")
        if conn.execute(
            'SELECT 1 FROM pragma_foreign_key_list(?, \'main\') WHERE "table" = ? COLLATE NOCASE LIMIT 1',
            (name, _SELECTION_TABLE),
        ).fetchone() is not None:
            raise TierJobSelectionError("Selected job marker has an incoming foreign key")
    where = " AND ".join(f"{name} = ?" for name in _SELECTION_FIELDS)
    changed = conn.execute(
        f"DELETE FROM main.{_SELECTION_TABLE} WHERE singleton=1 AND state='selected' AND {where}",
        _selection_values(job),
    ).rowcount
    if changed != 1:
        raise TierJobSelectionError("Selected job changed before retirement")


@contextmanager
def open_selected_tier_job(job: TierJob) -> Iterator[sqlite3.Connection]:
    """Reopen under worker ownership and validate the committed selection."""
    with open_tier_job(job) as conn:
        check_selected_tier_job(conn, job)
        yield conn


def end_selected_tier_job(
    conn: sqlite3.Connection, job: TierJob, *, outcome: Literal["cancelled", "failed"],
) -> bool:
    """End only this exact selected job; never modify a replacement's marker.

    This is a terminal write after abandoning any source plan, not page progress
    or successful completion/activation. False means the expected job is absent.
    """
    if outcome not in {"cancelled", "failed"}:
        raise TierJobSelectionError("Unsupported terminal selected job outcome")
    values = _selection_values(job)
    try:
        check_tier_job(conn, job)
        with _selection_transaction(conn, write=True):
            _selection_source(conn, job)
            _selection_schema(conn)
            row = _selection_row(conn)
            if row != (*values, "selected"):
                return False
            conditions = " AND ".join(f"{name} = ?" for name in _SELECTION_FIELDS)
            changed = conn.execute(
                f"UPDATE main.{_SELECTION_TABLE} SET state = ? "
                f"WHERE singleton = 1 AND {conditions} AND state = 'selected'", (outcome, *values),
            ).rowcount
            if changed != 1:
                raise TierJobSelectionError("Selected job changed during termination")
            _selection_source(conn, job)
        return True
    except (OSError, ValueError, sqlite3.Error, MaintenanceUnavailableError):
        raise TierJobSelectionError("Selected background job could not be ended") from None
