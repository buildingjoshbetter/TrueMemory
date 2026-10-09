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
