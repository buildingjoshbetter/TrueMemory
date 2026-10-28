"""Selection fences for already-admitted foreground/source writers.

No native model, serving lease, configuration, or selection is changed here.
An admitted runtime operation supplies its immutable selection; an unbound
direct caller captures a short database snapshot before preparing any output.
"""

from __future__ import annotations

import random
import sqlite3
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from truememory.tier_switch.activation import TierSelection, LegacyTierPolicy


# A foreground write that still finds the writer busy after the busy timeout
# tries again for at most this long in all. Under saturated writers SQLite's
# busy handler can starve one writer past its timeout: after its first polls it
# sleeps 100 ms between tries and loses each gap to writers that loop straight
# back. A new attempt restarts its fast polls.
FOREGROUND_BUSY_RETRY_SECONDS = 30.0
_RETRY_ATTEMPT_MS = 1_000


def _busy_error(error: sqlite3.OperationalError) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    if isinstance(code, int):
        return code & 255 in {5, 6}  # SQLITE_BUSY / SQLITE_LOCKED, including extended codes.
    return str(error) in {"database is locked", "database table is locked"}


def begin_immediate(
    conn: sqlite3.Connection, *, retry_seconds: float = 0.0,
    backoff: Callable[[], AbstractContextManager[object]] | None = None,
) -> None:
    """BEGIN IMMEDIATE outside a transaction; on a busy writer, retry for ``retry_seconds`` in all.

    Nothing has been written when BEGIN fails, so retrying cannot repeat any
    part of a write. Retries wait with jitter between attempts, each attempt
    with at most a one-second busy wait and never past the deadline.
    ``backoff`` is entered around each wait. It may let go of the caller's
    locks: the connection holds no transaction then and keeps its own busy
    timeout, since the shorter one is set for a single attempt only.
    """
    started = time.monotonic()
    try:
        conn.execute("BEGIN IMMEDIATE")
        return
    except sqlite3.OperationalError as error:
        if retry_seconds <= 0 or conn.in_transaction or not _busy_error(error):
            raise
        last = error
    deadline = started + retry_seconds
    attempt = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise last
        attempt += 1
        with backoff() if backoff is not None else nullcontext():
            time.sleep(min(remaining, random.uniform(0.01, 0.05) * min(attempt, 10)))
        if conn.in_transaction:
            raise last  # Opened by someone else during the wait: not ours to join.
        previous = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        wait_ms = int(min(previous, _RETRY_ATTEMPT_MS, max(deadline - time.monotonic(), 0) * 1000))
        conn.execute(f"PRAGMA busy_timeout = {wait_ms}")
        try:
            conn.execute("BEGIN IMMEDIATE")
            return
        except sqlite3.OperationalError as error:
            if conn.in_transaction or not _busy_error(error):
                raise
            last = error
        finally:
            conn.execute(f"PRAGMA busy_timeout = {previous}")


@contextmanager
def _released_for_backoff(
    runtime: object | None, release: Callable[[], AbstractContextManager[object]] | None,
) -> Iterator[None]:
    # The model fence is taken after the engine lock: let go of it first, take it back last.
    with release() if release is not None else nullcontext(), \
            runtime.released_connection_locks() if runtime is not None else nullcontext():
        yield


class WriterSelectionChanged(RuntimeError):
    """Retry the whole operation after serving selection reconciliation."""


@dataclass(frozen=True)
class WriterCapture:
    connection: sqlite3.Connection = field(repr=False)
    selection: TierSelection | None
    policy: LegacyTierPolicy | None = None

    def __post_init__(self) -> None:
        _validate(self.selection)
        if self.policy is not None:
            from truememory.tier_switch.activation import LegacyTierPolicy
            if type(self.policy) is not LegacyTierPolicy or self.selection is not None:
                raise WriterSelectionChanged("Writer policy capture must be an immutable journal result")


def _authority(conn: sqlite3.Connection) -> tuple[TierSelection | None, LegacyTierPolicy | None]:
    table = conn.execute(
        "SELECT type FROM main.sqlite_master WHERE name='metadata' COLLATE NOCASE",
    ).fetchone()
    if table is None:
        return None, None
    if table[0] != "table":
        raise WriterSelectionChanged("Selected metadata is not an ordinary table")
    if conn.execute("SELECT 1 FROM main.metadata WHERE key IN ('tier_selected_v1','tier_activation_v1') LIMIT 1").fetchone() is None:
        return None, None
    from truememory.tier_switch.activation import TierActivationError, _guard_storage, _state

    try:
        _guard_storage(conn)
        state = _state(conn)
        return state.selection, getattr(state, "legacy_policy", None)
    except (TierActivationError, sqlite3.Error) as exc:
        raise WriterSelectionChanged("Selected journal cannot be validated") from exc


def _selection(conn: sqlite3.Connection) -> TierSelection | None:
    return _authority(conn)[0]


def _validate(selection: TierSelection | None) -> None:
    if selection is not None:
        from truememory.tier_switch.activation import TierSelection

        if type(selection) is not TierSelection:
            raise WriterSelectionChanged("Writer capture must be an immutable selected descriptor")


def capture_writer_selection(conn: sqlite3.Connection) -> WriterCapture:
    """Capture before native/source work; never substitute a newer admitted key.

    The optional runtime module is consulted only when already loaded. Its
    current_operation accessor must raise for a mismatched active connection;
    None means unbound, while operation.selection=None is an admitted legacy
    selection. Caller transactions are read without ending or upgrading them.
    """
    runtime = sys.modules.get("truememory.tier_switch.runtime")
    if runtime is not None:
        operation = runtime.current_operation(conn)
        if operation is not None:
            _validate(operation.selection)
            return WriterCapture(conn, operation.selection, getattr(operation, "policy", None))
    owned = not conn.in_transaction
    if owned:
        conn.execute("BEGIN")
    try:
        return WriterCapture(conn, *_authority(conn))
    finally:
        if owned:
            # One cleanup attempt only, including failure after rollback.
            conn.rollback()
            if conn.in_transaction:
                raise WriterSelectionChanged("Writer capture snapshot cleanup is uncertain")


def require_writer_selection(
    conn: sqlite3.Connection, captured: WriterCapture | None, *,
    model_id: str | None = None, dimension: int | None = None,
    tables: tuple[str, str] | None = None,
) -> None:
    """Read-only comparison under the caller's actual database writer.

    sqlite3 exposes transaction existence, not lock kind. Use writer_transaction
    or another proven writer acquisition before this helper. It never loads,
    recaptures an admitted selection, mutates metadata, commits, or rolls back.
    Config acknowledgement alone does not change the captured embedding space.
    """
    if not conn.in_transaction:
        raise WriterSelectionChanged("Selected publication requires database writer ownership")
    if captured is not None and (type(captured) is not WriterCapture or captured.connection is not conn):
        raise WriterSelectionChanged("Writer capture belongs to a different accepted connection")
    selected = captured.selection if captured is not None else None
    current, current_policy = _authority(conn)
    policy = captured.policy if captured is not None else None
    if policy is None:
        if current_policy is not None:
            raise WriterSelectionChanged("A legacy policy appeared after writer admission")
    elif current_policy is None or replace(current_policy, config_acknowledged=policy.config_acknowledged) != policy:
        raise WriterSelectionChanged("Legacy policy generation changed; retry the operation")
    if selected is None and policy is None:
        if current is not None:
            raise WriterSelectionChanged("A selected tier appeared after legacy admission")
        return
    if selected is not None and (current is None or replace(current, config_acknowledged=selected.config_acknowledged) != selected):
        raise WriterSelectionChanged("Selected tier generation or descriptor changed; retry the operation")
    if policy is not None and current is not None:
        raise WriterSelectionChanged("A selected tier appeared after policy admission")
    selected = selected if selected is not None else policy
    if ((model_id is not None and model_id != selected.target.model_id)
            or (dimension is not None and (type(dimension) is not int or dimension != selected.target.dimension))
            or (tables is not None and (type(tables) is not tuple or tables != selected.tables))):
        raise WriterSelectionChanged("Prepared output does not match the captured selected pair")
    if policy is None:
        # Boolean SQL avoids copying unbounded registry/metadata values into Python.
        row = conn.execute(
            "SELECT vec_table=? AND sep_table=? AND model_name=? "
            "AND typeof(embedding_dim)='integer' AND embedding_dim=? "
            "FROM main.vector_cache_registry WHERE tier_group=?",
            (*selected.tables, selected.target.model_id, selected.target.dimension, selected.target.tier_group),
        ).fetchone()
        if row is None or row[0] != 1:
            raise WriterSelectionChanged("Selected vector registry identity changed")
    for key, value in (("embed_model", selected.target.model_id), ("embed_dim", str(selected.target.dimension))):
        row = conn.execute(
            "SELECT typeof(value)='text' AND value=? FROM main.metadata WHERE key=?", (value, key),
        ).fetchone()
        if row is None or row[0] != 1:
            raise WriterSelectionChanged("Selected embedder metadata changed")


@contextmanager
def writer_transaction(
    conn: sqlite3.Connection, captured: WriterCapture | None, *,
    model_id: str | None = None, dimension: int | None = None,
    tables: tuple[str, str] | None = None, retry_busy: bool = False,
    release: Callable[[], AbstractContextManager[object]] | None = None,
) -> Iterator[None]:
    """Acquire the writer before comparison; preserve borrowed outer work.

    Own BEGIN IMMEDIATE or borrow a SAVEPOINT and upgrade via a zero-row source
    UPDATE, never a metadata write. A stale read snapshot may fail that upgrade
    for retry. Only owned work is committed. Uncertain cleanup is not retried.
    With ``retry_busy`` an owned BEGIN that finds the writer busy past the busy
    timeout tries again (FOREGROUND_BUSY_RETRY_SECONDS in all); a
    borrowed one never does, since the caller's transaction is not ours to wait in.
    Between attempts it lets go of this thread's engine connection locks and of
    ``release`` (the caller's model fence), so other threads are not held up
    by the wait. Everything validated here is checked again on every attempt.
    """
    owned = not conn.in_transaction
    guard = "tier_writer_" + uuid.uuid4().hex
    if owned:
        backoff = None
        if retry_busy:
            # Admission judged a caller transaction under the engine writer;
            # such a borrowed operation must not wait here either.
            runtime = sys.modules.get("truememory.tier_switch.runtime")
            operation = runtime.current_operation(conn) if runtime is not None else None
            retry_busy = not getattr(operation, "borrowed", False)
            backoff = partial(_released_for_backoff, runtime, release)
        begin_immediate(conn, retry_seconds=FOREGROUND_BUSY_RETRY_SECONDS if retry_busy else 0.0, backoff=backoff)
    else:
        conn.execute(f"SAVEPOINT {guard}")
    guard_release_attempted = False
    try:
        if not owned:
            conn.execute("SAVEPOINT rebuild_source")
            conn.execute("UPDATE messages SET id = id WHERE 0")
        require_writer_selection(conn, captured, model_id=model_id, dimension=dimension, tables=tables)
        yield
        if owned:
            conn.commit()
            if conn.in_transaction:
                raise WriterSelectionChanged("Writer transaction did not commit")
        else:
            conn.execute("RELEASE SAVEPOINT rebuild_source")
            guard_release_attempted = True
            conn.execute(f"RELEASE SAVEPOINT {guard}")
    except BaseException:
        if owned:
            if conn.in_transaction:
                conn.rollback()
                if conn.in_transaction:
                    raise WriterSelectionChanged("Writer rollback is uncertain")
        elif conn.in_transaction and not guard_release_attempted:
            # Only our unique guard is eligible for cleanup. The compatibility
            # inner name may now refer to a caller savepoint after RELEASE.
            # A final guard-release error is ambiguous and permits no cleanup.
            conn.execute(f"ROLLBACK TO SAVEPOINT {guard}")
            conn.execute(f"RELEASE SAVEPOINT {guard}")
        raise
