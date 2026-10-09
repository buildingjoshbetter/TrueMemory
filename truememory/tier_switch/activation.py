"""Database-only certification of an explicitly selected, inactive tier pair.

Callers own maintenance, coherent old serving state, inactive target admission,
and the drained local serving boundary. None is inferred from a database row.
This module never applies config/runtime state or certifies legacy vectors.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace

from truememory.embedding_target import EmbeddingTarget
from truememory.tier_switch.cache import VectorCacheRegistry
from truememory.tier_switch.job import (
    TierJob, _retire_selected_tier_job_in_writer, check_selected_tier_job_in_writer,
)
from truememory.tier_switch.source import TierSourcePlan, require_current_completion_in_writer

_SELECTED = "tier_selected_v1"
_INTENT = "tier_activation_v1"
_MAX_RECORD_BYTES = 16384
_MAX_SCHEMA_BYTES = 65536
_TRACKERS = {"canonical-v1", "rebuild-bridge-v1", "rebuild-bridge-conservative-v1"}
_METADATA_SQL = "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT)"
_REGISTRY_SQL = """CREATE TABLE vector_cache_registry (
    tier_group TEXT PRIMARY KEY, vec_table TEXT NOT NULL, sep_table TEXT NOT NULL,
    last_embedded_id INTEGER DEFAULT 0, vector_count INTEGER DEFAULT 0,
    model_name TEXT, embedding_dim INTEGER DEFAULT 256, last_updated REAL, created REAL
)"""


class TierActivationError(RuntimeError):
    """Selection cannot be certified; no runtime/config success is implied."""


def _hex(value: object, size: int) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{" + str(size) + "}", value) is not None


def _text(value: object, limit: int) -> bool:
    if type(value) is not str or not 1 <= len(value) <= limit or "\x00" in value:
        return False
    try:
        return len(value.encode("utf-8")) <= limit * 4
    except UnicodeError:
        return False


def _identity(target: EmbeddingTarget, reranker_id: str, job_id: str,
              tracker: str, source_epoch: str, source_schema_signature: str) -> None:
    if (type(target) is not EmbeddingTarget or not _text(target.model_id, 512)
            or any(type(value) is not str for value in (target.tier, target.tier_group))
            or not _text(reranker_id, 512)
            or re.fullmatch(r"[\w][\w.\-]*(/[\w][\w.\-]*)?", reranker_id) is None
            or not _hex(job_id, 32) or type(tracker) is not str or tracker not in _TRACKERS
            or not _text(source_epoch, 128) or not _hex(source_schema_signature, 64)):
        raise TierActivationError("Invalid frozen activation identity")
    EmbeddingTarget.from_wire(target.to_wire())


@dataclass(frozen=True)
class ActivationIntent:
    intent_id: str
    expected_generation: str | None
    target: EmbeddingTarget
    reranker_id: str
    job_id: str
    tracker: str
    source_epoch: str
    source_schema_signature: str
    state: str = "staged"

    def __post_init__(self) -> None:
        _identity(self.target, self.reranker_id, self.job_id, self.tracker,
                  self.source_epoch, self.source_schema_signature)
        if (not _hex(self.intent_id, 32)
                or (self.expected_generation is not None and not _hex(self.expected_generation, 32))
                or type(self.state) is not str or self.state not in {"staged", "db_selected", "config_acknowledged"}):
            raise TierActivationError("Invalid activation intent")


@dataclass(frozen=True)
class TierSelection:
    generation: str
    intent_id: str
    target: EmbeddingTarget
    reranker_id: str
    tables: tuple[str, str]
    job_id: str
    tracker: str
    source_epoch: str
    source_schema_signature: str
    manifest_generation: str
    manifest_hash: str
    receipt_hash: str
    pair_digest: str
    schema: str
    vector_count: int
    cursor: int | None
    config_acknowledged: bool = False

    def __post_init__(self) -> None:
        _identity(self.target, self.reranker_id, self.job_id, self.tracker,
                  self.source_epoch, self.source_schema_signature)
        if (any(not _hex(value, 32) for value in (self.generation, self.intent_id, self.manifest_generation))
                or any(not _hex(value, 64) for value in (self.manifest_hash, self.receipt_hash, self.pair_digest, self.schema))
                or type(self.tables) is not tuple or self.tables != self.target.tables
                or type(self.vector_count) is not int or not 0 <= self.vector_count <= 2**63 - 1
                or (self.cursor is not None and (type(self.cursor) is not int or not -(2**63) <= self.cursor < 2**63))
                or ((self.vector_count == 0) != (self.cursor is None))
                or type(self.config_acknowledged) is not bool):
            raise TierActivationError("Invalid certified selection")


@dataclass(frozen=True)
class ActivationState:
    selection: TierSelection | None
    intent: ActivationIntent | None


def _dump(record: ActivationIntent | TierSelection) -> str:
    value = asdict(record)
    value["target"] = record.target.to_wire()
    raw = json.dumps(dict(value, version=1), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(raw.encode("utf-8")) > _MAX_RECORD_BYTES:
        raise TierActivationError("Activation record exceeds protocol bounds")
    return raw


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise TierActivationError("Duplicate activation record field")
    return value


def _parse(raw: str, kind: type[ActivationIntent] | type[TierSelection]) -> ActivationIntent | TierSelection:
    try:
        value = json.loads(raw, object_pairs_hook=_unique)
        if (type(value) is not dict or set(value) != set(kind.__dataclass_fields__) | {"version"}
                or type(value["version"]) is not int or value.pop("version") != 1):
            raise TierActivationError("Unsupported activation record schema")
        value["target"] = EmbeddingTarget.from_wire(value["target"])
        if kind is TierSelection:
            if type(value["tables"]) is not list or len(value["tables"]) != 2:
                raise TierActivationError("Invalid selected table pair")
            value["tables"] = tuple(value["tables"])
        return kind(**value)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise TierActivationError("Malformed activation record") from exc


def _normalized(sql: str) -> str:
    return re.sub(r"\s*([(),])\s*", r"\1", " ".join(sql.strip().rstrip(";").split())).lower()


def _schema_sql(conn: sqlite3.Connection, name: str) -> str:
    shape = conn.execute(
        "SELECT type, typeof(sql), length(CAST(sql AS BLOB)) FROM main.sqlite_master "
        "WHERE name=? COLLATE NOCASE", (name,),
    ).fetchone()
    if shape is None or shape[:2] != ("table", "text") or not 1 <= shape[2] <= _MAX_SCHEMA_BYTES:
        raise TierActivationError("Initialized bounded table schema is required")
    return conn.execute("SELECT sql FROM main.sqlite_master WHERE name=? COLLATE NOCASE", (name,)).fetchone()[0]


def _guard_storage(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT 1 FROM temp.sqlite_master LIMIT 1").fetchone() is not None:
        raise TierActivationError("TEMP objects make activation storage ambiguous")
    if conn.execute("SELECT 1 FROM pragma_database_list WHERE name NOT IN ('main','temp') LIMIT 1").fetchone():
        raise TierActivationError("Activation requires a single MAIN database")
    for name, expected in (("metadata", _METADATA_SQL), ("vector_cache_registry", _REGISTRY_SQL)):
        if _normalized(_schema_sql(conn, name)) != _normalized(expected):
            raise TierActivationError("Unsupported activation storage schema")
        if conn.execute(
            "SELECT 1 FROM main.sqlite_master WHERE tbl_name=? COLLATE NOCASE "
            "AND (type='trigger' OR (type='index' AND sql IS NOT NULL)) LIMIT 1", (name,),
        ).fetchone():
            raise TierActivationError("Activation storage has unsupported side effects")


def _guard_pair(conn: sqlite3.Connection, target: EmbeddingTarget) -> None:
    for name in target.tables:
        declaration = (
            r"CREATE\s+VIRTUAL\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
            + r'(?:"' + name + r'"|`' + name + r'`|\[' + name + r'\]|' + name + r')'
            + r"\s+USING\s+vec0\s*\(\s*embedding\s+float\[" + str(target.dimension)
            + r"\]\s+distance_metric\s*=\s*cosine\s*\)\s*"
        )
        if re.fullmatch(declaration, _schema_sql(conn, name), re.IGNORECASE) is None:
            raise TierActivationError("Selected pair must have the exact prepared vec0 schema")


def _read(conn: sqlite3.Connection, key: str, kind: type[ActivationIntent] | type[TierSelection]
          ) -> ActivationIntent | TierSelection | None:
    shape = conn.execute(
        "SELECT typeof(value), length(CAST(value AS BLOB)) FROM main.metadata WHERE key=?", (key,),
    ).fetchone()
    if shape is None:
        return None
    if shape[0] != "text" or not 1 <= shape[1] <= _MAX_RECORD_BYTES:
        raise TierActivationError("Activation record exceeds its type or byte bound")
    raw = conn.execute("SELECT value FROM main.metadata WHERE key=?", (key,)).fetchone()[0]
    return _parse(raw, kind)


def _put(conn: sqlite3.Connection, key: str, value: str) -> None:
    # Preserve the primary key: REPLACE could cascade an incoming foreign key.
    conn.execute(
        "INSERT INTO main.metadata(key,value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value),
    )


@contextmanager
def _transaction(conn: sqlite3.Connection, *, write: bool) -> Iterator[None]:
    if conn.in_transaction:
        raise TierActivationError("Activation requires an owned clean connection")
    conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
    try:
        yield
        if write:
            conn.commit()
            if conn.in_transaction:
                raise TierActivationError("Activation transaction did not commit")
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
            if conn.in_transaction:
                raise TierActivationError("Activation rollback is uncertain")
        raise
    else:
        # Cleanup failure is terminal. Keeping this outside the handler above
        # prevents a second rollback after an uncertain first attempt.
        if not write:
            conn.rollback()
            if conn.in_transaction:
                raise TierActivationError("Activation snapshot did not close")


def _state(conn: sqlite3.Connection) -> ActivationState:
    selected, intent = _read(conn, _SELECTED, TierSelection), _read(conn, _INTENT, ActivationIntent)
    if selected is not None and intent is None:
        raise TierActivationError("Selected record has no activation intent")
    if intent is not None and intent.state == "staged":
        _expected(selected, intent.expected_generation)
        if selected is not None and not selected.config_acknowledged:
            raise TierActivationError("Pending config selection cannot have a new staged intent")
    if intent is not None and intent.state != "staged":
        if (selected is None or selected.intent_id != intent.intent_id
                or selected.target != intent.target or selected.reranker_id != intent.reranker_id
                or (selected.job_id, selected.tracker, selected.source_epoch, selected.source_schema_signature)
                != (intent.job_id, intent.tracker, intent.source_epoch, intent.source_schema_signature)
                or selected.config_acknowledged != (intent.state == "config_acknowledged")):
            raise TierActivationError("Selected record and activation result disagree")
    return ActivationState(selected, intent)


def read_activation_state(conn: sqlite3.Connection) -> ActivationState:
    """Read committed journal state separately after reopening a safe connection.

    This is not config/runtime reconciliation, fresh vector recertification, or
    permission to reuse a connection with uncertain rollback state.
    """
    with _transaction(conn, write=False):
        _guard_storage(conn)
        return _state(conn)


def _expected(selection: TierSelection | None, generation: str | None) -> None:
    if generation is not None and not _hex(generation, 32):
        raise TierActivationError("Invalid expected selection generation")
    if (selection.generation if selection else None) != generation:
        raise TierActivationError("Selected generation changed")


def stage_activation_intent(conn: sqlite3.Connection, job: TierJob, *,
                            expected_generation: str | None, reranker_id: str) -> ActivationIntent:
    """Commit intent before source planning; identical current intent is idempotent.

    None expects absent selection and permits only a later certified target. It
    does not bootstrap/certify the legacy serving pair. A different job can
    replace an abandoned staged intent only after selected-job admission has
    replaced its marker. Conflicting intent for a live job is refused.
    """
    intent = ActivationIntent(uuid.uuid4().hex, expected_generation, job.target, reranker_id,
                              job.job_id, job.tracker, job.source_epoch, job.source_schema_signature)
    with _transaction(conn, write=True):
        _guard_storage(conn)
        check_selected_tier_job_in_writer(conn, job)
        state = _state(conn)
        _expected(state.selection, expected_generation)
        if state.selection is not None and not state.selection.config_acknowledged:
            raise TierActivationError("Current selection awaits config acknowledgement")
        if state.intent is not None and state.intent.job_id == job.job_id:
            if state.intent == replace(intent, intent_id=state.intent.intent_id):
                return state.intent
            raise TierActivationError("A conflicting intent already owns this live job")
        _put(conn, _INTENT, _dump(intent))
    return intent


def commit_certified_selection(conn: sqlite3.Connection, intent: ActivationIntent,
                               job: TierJob, plan: TierSourcePlan) -> TierSelection:
    """Certify and select atomically, then retire this exact successful marker.

    Caller must drain serving operations, prove an inactive target, and retain
    maintenance/stable-path ownership. No native/config/runtime action occurs.
    Commit/rollback exceptions propagate; use separate safe readback to learn
    whether durable selection occurred. A returned result is still config-pending.
    """
    if type(intent) is not ActivationIntent or intent.state != "staged":
        raise TierActivationError("A staged immutable intent is required")
    if (intent.target != job.target
            or (intent.job_id, intent.tracker, intent.source_epoch, intent.source_schema_signature)
            != (job.job_id, job.tracker, job.source_epoch, job.source_schema_signature)):
        raise TierActivationError("Activation intent belongs to a different job")
    with _transaction(conn, write=True):
        _guard_storage(conn)
        check_selected_tier_job_in_writer(conn, job)
        state = _state(conn)
        _expected(state.selection, intent.expected_generation)
        if state.intent != intent:
            raise TierActivationError("Activation intent changed")
        if (plan.manifest.model != intent.target.model_id or plan.manifest.dimension != intent.target.dimension
                or plan.manifest.targets != intent.target.tables
                or plan.manifest.source is None
                or (plan.manifest.source.tracker, plan.manifest.source.revision.epoch,
                    plan.manifest.source.schema_signature)
                != (intent.tracker, intent.source_epoch, intent.source_schema_signature)):
            raise TierActivationError("Paired certificate belongs to a different target or source")
        _guard_pair(conn, intent.target)
        require_current_completion_in_writer(conn, plan)
        selection = TierSelection(
            uuid.uuid4().hex, intent.intent_id, intent.target, intent.reranker_id, intent.target.tables,
            intent.job_id, intent.tracker, intent.source_epoch, intent.source_schema_signature,
            plan.manifest.generation, hashlib.sha256(plan.saved_manifest.encode()).hexdigest(),
            hashlib.sha256(plan.saved_receipt.encode()).hexdigest(), plan.digest, plan.schema,
            plan.prefix_count + plan.manifest.outputs, plan.manifest.cursor,
        )
        # The marker helper rechecks the exact row and refuses all incoming FKs.
        # Both registry/metadata writes preserve their primary keys and cannot
        # invoke triggers under the exact schemas accepted above.
        _retire_selected_tier_job_in_writer(conn, job)
        VectorCacheRegistry.set(
            conn, intent.target.tier_group, vec_table=selection.tables[0], sep_table=selection.tables[1],
            last_embedded_id=selection.cursor if selection.cursor is not None else 0,
            vector_count=selection.vector_count, model_name=intent.target.model_id,
            embedding_dim=intent.target.dimension, commit=False,
        )
        _put(conn, "embed_model", intent.target.model_id)
        _put(conn, "embed_dim", str(intent.target.dimension))
        _put(conn, _SELECTED, _dump(selection))
        _put(conn, _INTENT, _dump(replace(intent, state="db_selected")))
    return selection


def acknowledge_config(conn: sqlite3.Connection, *, generation: str) -> TierSelection:
    """Generation-CAS acknowledgement after caller-confirmed config mirroring.

    This performs no config write and acknowledges no process's runtime state.
    """
    if not _hex(generation, 32):
        raise TierActivationError("A concrete selected generation is required")
    with _transaction(conn, write=True):
        _guard_storage(conn)
        state = _state(conn)
        _expected(state.selection, generation)
        if state.intent.state == "staged" or state.intent.intent_id != state.selection.intent_id:
            raise TierActivationError("A newer activation intent is pending")
        if state.selection.config_acknowledged:
            return state.selection
        selected = replace(state.selection, config_acknowledged=True)
        _put(conn, _SELECTED, _dump(selected))
        _put(conn, _INTENT, _dump(replace(state.intent, state="config_acknowledged")))
    return selected
