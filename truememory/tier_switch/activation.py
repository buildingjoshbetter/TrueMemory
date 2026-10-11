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


@dataclass(frozen=True)
class ConfigGuard:
    tier_present: bool
    tier: str
    generation: str | None

    def __post_init__(self) -> None:
        if (type(self.tier_present) is not bool or type(self.tier) is not str
                or self.tier not in {"edge", "base", "pro", "custom"}
                or (not self.tier_present and self.tier != "edge")
                or (self.generation is not None and not _hex(self.generation, 32))):
            raise TierActivationError("Invalid expected config identity")


@dataclass(frozen=True)
class LegacyTierPolicy:
    generation: str
    intent_id: str
    target: EmbeddingTarget
    reranker_id: str
    tables: tuple[str, str]
    config_acknowledged: bool = False

    def __post_init__(self) -> None:
        _policy_identity(self.target, self.reranker_id, self.tables)
        if (not _hex(self.generation, 32) or not _hex(self.intent_id, 32)
                or type(self.config_acknowledged) is not bool):
            raise TierActivationError("Invalid legacy serving policy")


def _policy_identity(target: EmbeddingTarget, reranker_id: str, tables: tuple[str, str]) -> None:
    if (type(target) is not EmbeddingTarget or target.tier not in {"base", "pro"}
            or target.identity != ("qwen3_256", 256, "basepro")
            or type(reranker_id) is not str or reranker_id != "Alibaba-NLP/gte-reranker-modernbert-base"
            or type(tables) is not tuple or len(tables) != 2 or any(type(name) is not str for name in tables)
            or tables not in {target.tables, ("vec_messages", "vec_messages_sep")}):
        raise TierActivationError("Policy-only publication requires the unchanged base/pro embedding space")


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
    expected_config: ConfigGuard | None = None
    status_id: int | None = None
    previous_policy: LegacyTierPolicy | None = None

    def __post_init__(self) -> None:
        _identity(self.target, self.reranker_id, self.job_id, self.tracker,
                  self.source_epoch, self.source_schema_signature)
        if (not _hex(self.intent_id, 32)
                or (self.expected_generation is not None and not _hex(self.expected_generation, 32))
                or type(self.state) is not str or self.state not in {"staged", "db_selected", "config_acknowledged"}):
            raise TierActivationError("Invalid activation intent")
        if (self.expected_config is not None and type(self.expected_config) is not ConfigGuard
                or self.status_id is not None and (type(self.status_id) is not int or not 1 <= self.status_id < 2**63)
                or self.previous_policy is not None and type(self.previous_policy) is not LegacyTierPolicy
                or self.expected_config is None and (self.status_id is not None or self.previous_policy is not None)
                or self.previous_policy is not None and self.expected_generation is not None):
            raise TierActivationError("Invalid activation projection evidence")


@dataclass(frozen=True)
class PolicyIntent:
    intent_id: str
    expected_generation: str | None
    expected_intent_id: str | None
    target: EmbeddingTarget
    reranker_id: str
    expected_config: ConfigGuard
    result_generation: str
    tables: tuple[str, str]
    state: str = "db_selected"

    def __post_init__(self) -> None:
        _policy_identity(self.target, self.reranker_id, self.tables)
        if (not _hex(self.intent_id, 32) or not _hex(self.result_generation, 32)
                or self.expected_generation is not None and not _hex(self.expected_generation, 32)
                or self.expected_intent_id is not None and not _hex(self.expected_intent_id, 32)
                or self.result_generation == self.expected_generation
                or self.intent_id == self.expected_intent_id
                or type(self.expected_config) is not ConfigGuard
                or self.expected_config.tier not in {"base", "pro"}
                or type(self.state) is not str or self.state not in {"db_selected", "config_acknowledged"}):
            raise TierActivationError("Invalid policy activation intent")


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
    intent: ActivationIntent | PolicyIntent | None
    legacy_policy: LegacyTierPolicy | None = None


def _config_wire(value: ConfigGuard) -> dict:
    return dict(asdict(value), version=1)


def _parse_config(value: object) -> ConfigGuard:
    if (type(value) is not dict or set(value) != {"version", "tier_present", "tier", "generation"}
            or type(value["version"]) is not int or value["version"] != 1):
        raise TierActivationError("Unsupported expected config descriptor")
    return ConfigGuard(value["tier_present"], value["tier"], value["generation"])


def _policy_wire(value: LegacyTierPolicy) -> dict:
    return dict(asdict(value), version=1, target=value.target.to_wire())


def _parse_policy(value: object) -> LegacyTierPolicy:
    if (type(value) is not dict or set(value) != set(LegacyTierPolicy.__dataclass_fields__) | {"version"}
            or type(value["version"]) is not int or value["version"] != 1
            or type(value["tables"]) is not list or len(value["tables"]) != 2):
        raise TierActivationError("Unsupported previous legacy policy")
    return LegacyTierPolicy(value["generation"], value["intent_id"], EmbeddingTarget.from_wire(value["target"]),
                            value["reranker_id"], tuple(value["tables"]), value["config_acknowledged"])


def _dump(record: ActivationIntent | TierSelection | PolicyIntent) -> str:
    value = asdict(record)
    value["target"] = record.target.to_wire()
    version = 1
    if type(record) is ActivationIntent:
        if record.expected_config is None:
            for field in ("expected_config", "status_id", "previous_policy"):
                value.pop(field)
        else:
            version = 2
            value["kind"] = "rebuild"
            value["expected_config"] = _config_wire(record.expected_config)
            value["previous_policy"] = None if record.previous_policy is None else _policy_wire(record.previous_policy)
    elif type(record) is PolicyIntent:
        version = 2
        value["kind"] = "policy"
        value["expected_config"] = _config_wire(record.expected_config)
    raw = json.dumps(dict(value, version=version), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(raw.encode("utf-8")) > _MAX_RECORD_BYTES:
        raise TierActivationError("Activation record exceeds protocol bounds")
    return raw


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise TierActivationError("Duplicate activation record field")
    return value


def _parse(raw: str, kind: type[ActivationIntent] | type[TierSelection]) -> ActivationIntent | TierSelection | PolicyIntent:
    try:
        value = json.loads(raw, object_pairs_hook=_unique)
        if type(value) is not dict or type(value.get("version")) is not int:
            raise TierActivationError("Unsupported activation record schema")
        version = value.pop("version")
        expected = set(kind.__dataclass_fields__)
        if kind is ActivationIntent and version == 1:
            expected -= {"expected_config", "status_id", "previous_policy"}
        elif kind is ActivationIntent and version == 2:
            discriminator = value.pop("kind", None)
            if discriminator == "policy":
                kind = PolicyIntent
            elif discriminator != "rebuild":
                raise TierActivationError("Unsupported activation intent kind")
            expected = set(kind.__dataclass_fields__)
        elif version != 1:
            raise TierActivationError("Unsupported activation record version")
        if set(value) != expected:
            raise TierActivationError("Unsupported activation record fields")
        if version == 2:
            value["expected_config"] = _parse_config(value["expected_config"])
            if kind is ActivationIntent and value["previous_policy"] is not None:
                value["previous_policy"] = _parse_policy(value["previous_policy"])
        value["target"] = EmbeddingTarget.from_wire(value["target"])
        if kind in (TierSelection, PolicyIntent):
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


def _guard_pair(conn: sqlite3.Connection, target: EmbeddingTarget, *, tables: tuple[str, str] | None = None) -> None:
    for name in target.tables if tables is None else tables:
        declaration = (
            r"CREATE\s+VIRTUAL\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
            + r'(?:"' + name + r'"|`' + name + r'`|\[' + name + r'\]|' + name + r')'
            + r"\s+USING\s+vec0\s*\(\s*embedding\s+float\[" + str(target.dimension)
            + r"\]\s+distance_metric\s*=\s*cosine\s*\)\s*"
        )
        if re.fullmatch(declaration, _schema_sql(conn, name), re.IGNORECASE) is None:
            raise TierActivationError("Selected pair must have the exact prepared vec0 schema")


def _read(conn: sqlite3.Connection, key: str, kind: type[ActivationIntent] | type[TierSelection]
          ) -> ActivationIntent | TierSelection | PolicyIntent | None:
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
    if type(intent) is PolicyIntent:
        acknowledged = intent.state == "config_acknowledged"
        if selected is None:
            if intent.expected_generation is not None:
                raise TierActivationError("Certified policy lost its selection")
            return ActivationState(None, intent, LegacyTierPolicy(
                intent.result_generation, intent.intent_id, intent.target, intent.reranker_id,
                intent.tables, acknowledged,
            ))
        if ((selected.generation, selected.intent_id, selected.target, selected.reranker_id,
                selected.tables, selected.config_acknowledged)
                != (intent.result_generation, intent.intent_id, intent.target, intent.reranker_id,
                    intent.tables, acknowledged) or intent.expected_generation is None):
            raise TierActivationError("Certified policy and selected record disagree")
        return ActivationState(selected, intent)
    if intent is not None and intent.state == "staged":
        _expected(selected, intent.expected_generation)
        if selected is not None and not selected.config_acknowledged:
            raise TierActivationError("Pending config selection cannot have a new staged intent")
        if intent.previous_policy is not None and (
            not intent.previous_policy.config_acknowledged
            or intent.expected_config.tier != intent.previous_policy.target.tier
            or intent.expected_config.generation != intent.previous_policy.generation
        ):
            raise TierActivationError("Staged intent has incoherent prior legacy policy evidence")
    if intent is not None and intent.state != "staged":
        if (selected is None or selected.intent_id != intent.intent_id
                or selected.target != intent.target or selected.reranker_id != intent.reranker_id
                or (selected.job_id, selected.tracker, selected.source_epoch, selected.source_schema_signature)
                != (intent.job_id, intent.tracker, intent.source_epoch, intent.source_schema_signature)
                or selected.config_acknowledged != (intent.state == "config_acknowledged")):
            raise TierActivationError("Selected record and activation result disagree")
    return ActivationState(selected, intent, intent.previous_policy if intent is not None and intent.state == "staged" else None)


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
                            expected_generation: str | None, reranker_id: str,
                            expected_config: ConfigGuard | None = None,
                            status_id: int | None = None) -> ActivationIntent:
    """Commit intent before source planning; identical current intent is idempotent.

    None expects absent selection and permits only a later certified target. It
    does not bootstrap/certify the legacy serving pair. A different job can
    replace an abandoned staged intent only after selected-job admission has
    replaced its marker. Conflicting intent for a live job is refused.
    """
    intent = ActivationIntent(uuid.uuid4().hex, expected_generation, job.target, reranker_id,
                              job.job_id, job.tracker, job.source_epoch, job.source_schema_signature,
                              expected_config=expected_config, status_id=status_id)
    with _transaction(conn, write=True):
        _guard_storage(conn)
        check_selected_tier_job_in_writer(conn, job)
        state = _state(conn)
        _expected(state.selection, expected_generation)
        if state.selection is not None and not state.selection.config_acknowledged:
            raise TierActivationError("Current selection awaits config acknowledgement")
        if expected_config is not None and state.selection is not None and (
            expected_config.tier != state.selection.target.tier or expected_config.generation != state.selection.generation
        ):
            raise TierActivationError("Prior config does not identify the selected generation")
        if state.legacy_policy is not None:
            if (not state.legacy_policy.config_acknowledged or expected_config is None
                    or expected_config.tier != state.legacy_policy.target.tier
                    or expected_config.generation != state.legacy_policy.generation):
                raise TierActivationError("Legacy policy requires acknowledged, guarded activation")
            intent = replace(intent, previous_policy=state.legacy_policy)
        if type(state.intent) is ActivationIntent and state.intent.job_id == job.job_id:
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


def _guard_legacy_policy_pair(conn: sqlite3.Connection, target: EmbeddingTarget, tables: tuple[str, str],
                              *, certified: bool = False) -> None:
    _guard_pair(conn, target, tables=tables)
    for key, value in (("embed_model", target.model_id), ("embed_dim", str(target.dimension))):
        row = conn.execute("SELECT typeof(value)='text' AND value=? FROM main.metadata WHERE key=?", (value, key)).fetchone()
        if row is None or row[0] != 1:
            raise TierActivationError("Legacy embedding metadata is not coherent with base/pro")
    model_match = "model_name='qwen3_256'" if certified else "model_name IN ('qwen3_256','Qwen3-Embedding-0.6B')"
    row = conn.execute(
        "SELECT vec_table=? AND sep_table=? AND typeof(embedding_dim)='integer' AND embedding_dim=? "
        f"AND {model_match} FROM main.vector_cache_registry WHERE tier_group='basepro'", (*tables, target.dimension),
    ).fetchone()
    if (row is None and (certified or tables != ("vec_messages", "vec_messages_sep"))) or (row is not None and row[0] != 1):
        raise TierActivationError("Serving pair disagrees with its registry")



def commit_config_only_transition(
    conn: sqlite3.Connection, *, expected_selection_generation: str | None,
    expected_intent_id: str | None, target: EmbeddingTarget, reranker_id: str,
    expected_config: ConfigGuard, legacy_pair: tuple[str, str] | None = None,
) -> TierSelection | LegacyTierPolicy:
    """Publish only base/pro policy, without a corpus read or vector certificate.

    Caller owns stable-path maintenance and exclusive serving. A legacy pair
    must be its coherently observed serving pair, never inferred from counts.
    Existing certificates are historical provenance, not recertified coverage.
    """
    from truememory.tier_switch.job import _cancel_policy_job_in_writer, read_selected_job_marker

    if type(expected_config) is not ConfigGuard or expected_config.tier not in {"base", "pro"}:
        raise TierActivationError("Policy transition requires actual prior config evidence")
    if expected_intent_id is not None and not _hex(expected_intent_id, 32):
        raise TierActivationError("Invalid expected prior activation intent")
    with _transaction(conn, write=True):
        _guard_storage(conn)
        state = _state(conn)
        _expected(state.selection, expected_selection_generation)
        if (state.intent.intent_id if state.intent is not None else None) != expected_intent_id:
            raise TierActivationError("Prior activation intent changed")
        previous = state.selection or state.legacy_policy
        if previous is not None and not previous.config_acknowledged:
            raise TierActivationError("Prior serving policy awaits config acknowledgement")
        tables = state.selection.tables if state.selection is not None else legacy_pair
        _policy_identity(target, reranker_id, tables)
        if target.tier == expected_config.tier:
            raise TierActivationError("Policy-only transition must change base/pro public tier")
        if previous is not None and (
            previous.target.tier != expected_config.tier or previous.generation != expected_config.generation
            or previous.target.identity != target.identity
            or previous.tables != tables or previous.reranker_id != reranker_id
        ):
            raise TierActivationError("Policy-only transition would change the serving embedding space")
        _guard_legacy_policy_pair(conn, target, tables, certified=state.selection is not None)
        if type(state.intent) is ActivationIntent and state.intent.state == "staged":
            _cancel_policy_job_in_writer(
                conn, job_id=state.intent.job_id, target=state.intent.target, tracker=state.intent.tracker,
                source_epoch=state.intent.source_epoch, source_schema_signature=state.intent.source_schema_signature,
            )
        else:
            marker = read_selected_job_marker(conn)
            if marker is not None and marker.state == "selected":
                raise TierActivationError("An unrelated selected job cannot be superseded")
        intent = PolicyIntent(uuid.uuid4().hex, expected_selection_generation, expected_intent_id,
                              target, reranker_id, expected_config, uuid.uuid4().hex, tables)
        if state.selection is not None:
            result = replace(state.selection, generation=intent.result_generation, intent_id=intent.intent_id,
                             target=target, config_acknowledged=False)
            _put(conn, _SELECTED, _dump(result))
        else:
            result = LegacyTierPolicy(intent.result_generation, intent.intent_id, target, reranker_id, tables)
        _put(conn, _INTENT, _dump(intent))
    return result


def acknowledge_policy_config(conn: sqlite3.Connection, *, generation: str) -> LegacyTierPolicy:
    """Acknowledge only this legacy policy's config mirror, never native state."""
    if not _hex(generation, 32):
        raise TierActivationError("A concrete policy generation is required")
    with _transaction(conn, write=True):
        _guard_storage(conn)
        state = _state(conn)
        if type(state.intent) is not PolicyIntent or state.legacy_policy is None or state.legacy_policy.generation != generation:
            raise TierActivationError("Legacy policy generation changed")
        if state.legacy_policy.config_acknowledged:
            return state.legacy_policy
        _put(conn, _INTENT, _dump(replace(state.intent, state="config_acknowledged")))
        result = replace(state.legacy_policy, config_acknowledged=True)
    return result
