"""Selection fences for already-admitted foreground/source writers.

No native model, serving lease, configuration, or selection is changed here.
An admitted runtime operation supplies its immutable selection; an unbound
direct caller captures a short database snapshot before preparing any output.
"""

from __future__ import annotations

import sqlite3
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from truememory.tier_switch.activation import TierSelection, LegacyTierPolicy


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
    tables: tuple[str, str] | None = None,
) -> Iterator[None]:
    """Acquire the writer before comparison; preserve borrowed outer work.

    Own BEGIN IMMEDIATE or borrow a SAVEPOINT and upgrade via a zero-row source
    UPDATE, never a metadata write. A stale read snapshot may fail that upgrade
    for retry. Only owned work is committed. Uncertain cleanup is not retried.
    """
    owned = not conn.in_transaction
    guard = "tier_writer_" + uuid.uuid4().hex
    conn.execute("BEGIN IMMEDIATE" if owned else f"SAVEPOINT {guard}")
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
