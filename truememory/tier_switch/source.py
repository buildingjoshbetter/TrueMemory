"""Model-free, bounded source checkpoints for caller-owned inactive tier pairs.

The caller owns the maintenance lease, exclusive cooperating pair writer, target
schema/model validation, and database identity across reopen. Tracking, metadata,
and both target tables must exist before planning. This adapter neither clears
tables nor activates them. It rejects borrowed transactions and concurrent writes
until the caller replans. Receipts detect changed stored bytes, not a hostile SQL
writer capable of forging both data and receipts.
"""

import hashlib
import json
import re
import sqlite3
import struct
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace

from truememory.maintenance import SourceRevision
from truememory.rebuild_source import RebuildManifest, RebuildSource, RebuildSourceChanged, capture_source, manifest_key

_PAGE_ROWS = 64
_MAX_METADATA_BYTES = 65536
_SEED = hashlib.sha256(b"truememory-tier-pair-chain-v1").hexdigest()
_FIELDS = ("id", "content", "sender", "recipient", "timestamp")
_ID_MIN, _ID_MAX = -(2**63), 2**63 - 1
_HEX = re.compile(r"[0-9a-f]{64}\Z")
SQLiteValue = str | bytes | int | float | None


class TierSourceUntrusted(RebuildSourceChanged):
    """A fresh empty pair or explicit revalidation is required."""


@dataclass(frozen=True)
class TierSourcePlan:
    action: str
    manifest: RebuildManifest
    schema: str
    digest: str
    prefix_count: int
    prefix_cursor: int | None
    connection: sqlite3.Connection
    data_version: int
    changes: int
    saved_manifest: str | None
    saved_receipt: str | None


@dataclass(frozen=True)
class TierSourcePage:
    generation: str
    cursor: int | None
    consumed: int
    rows: tuple[tuple[SQLiteValue, ...], ...]

    def messages(self) -> list[dict[str, SQLiteValue]]:
        """A bounded copy suitable for the existing completion/separation format."""
        return [dict(zip(_FIELDS, row)) for row in self.rows]


@dataclass(frozen=True)
class TierSourceCompletion:
    plan: TierSourcePlan
    captured_range_complete: bool
    current_source_complete: bool


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _integer(value: object, *, minimum: int = 0, maximum: int = _ID_MAX) -> bool:
    return type(value) is int and minimum <= value <= maximum


def _identifier(value: object) -> bool:
    return _integer(value, minimum=_ID_MIN)


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _identity(model_id: str, dimension: int, targets: tuple[str, str]) -> None:
    if (not isinstance(model_id, str) or not model_id or len(model_id) > 4096
            or not _integer(dimension, minimum=1, maximum=4096)):
        raise ValueError("A concrete model and integer dimension in 1..4096 are required")
    if (type(targets) is not tuple or len(targets) != 2
            or not all(isinstance(name, str) for name in targets)):
        raise ValueError("An exact ordered completion/separation pair is required")
    match = re.fullmatch(r"vec_messages_([a-z][a-z0-9_]{0,63})", targets[0])
    if match is None or targets[1] != "vec_messages_sep_" + match[1]:
        raise ValueError("Target names must be a canonical paired tier group")


def _guard_schema(conn: sqlite3.Connection, targets: tuple[str, ...]) -> str:
    guarded = (*targets, "messages", "metadata", "maintenance_source_state", "truememory_rebuild_source_v1")
    marks = ",".join("?" for _ in guarded)
    if conn.execute(
        f"SELECT 1 FROM temp.sqlite_master WHERE name COLLATE NOCASE IN ({marks}) OR "
        f"(type='trigger' AND tbl_name COLLATE NOCASE IN ({marks})) LIMIT 1", (*guarded, *guarded),
    ).fetchone():
        raise TierSourceUntrusted("TEMP shadows or triggers invalidate tier source ownership")
    if conn.execute(
        "SELECT 1 FROM main.sqlite_master WHERE type='trigger' AND tbl_name COLLATE NOCASE IN (?,?,?) LIMIT 1",
        (*targets, "metadata"),
    ).fetchone():
        raise TierSourceUntrusted("Tier pair and receipt tables must have no side-effect triggers")
    definitions = []
    for name in (*targets, "metadata"):
        row = conn.execute("SELECT type, rootpage, sql FROM main.sqlite_master WHERE name=?", (name,)).fetchone()
        if row is None or row[0] != "table" or not row[2]:
            raise TierSourceUntrusted("Initialized MAIN target and metadata tables are required")
        columns = [tuple(item) for item in conn.execute(f"PRAGMA main.table_xinfo({_quote(name)})")]
        if name in targets and "embedding" not in {column[1] for column in columns}:
            raise TierSourceUntrusted("Target table has no embedding column")
        if name == "metadata" and ([column[1] for column in columns if column[5]] != ["key"]
                                   or "value" not in {column[1] for column in columns}):
            raise TierSourceUntrusted("Metadata requires a unique key and value column")
        definitions.append((name, tuple(row), columns))
    return _hash([conn.execute("PRAGMA main.schema_version").fetchone()[0], definitions])


def _metadata(conn: sqlite3.Connection, key: str) -> str | None:
    shape = conn.execute(
        "SELECT typeof(value), length(CAST(value AS BLOB)) FROM main.metadata WHERE key=?", (key,),
    ).fetchone()
    if shape is None:
        return None
    if shape[0] != "text" or shape[1] > _MAX_METADATA_BYTES:
        raise TierSourceUntrusted("Tier source metadata has an unsupported type or size")
    return conn.execute("SELECT value FROM main.metadata WHERE key=?", (key,)).fetchone()[0]


def _receipt_key(targets: tuple[str, ...]) -> str:
    return "tier_pair_receipt_v1:" + ":".join(targets)


def _manifest_json(manifest: RebuildManifest) -> str:
    return _json(dict(asdict(manifest), version=1))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate metadata field")
    return result


def _parse_manifest(raw: str, model: str, dimension: int, targets: tuple[str, str]) -> RebuildManifest:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
        expected = set(RebuildManifest.__dataclass_fields__) | {"version"}
        if (type(value) is not dict or set(value) != expected
                or type(value["version"]) is not int or value.pop("version") != 1):
            raise ValueError("manifest shape")
        source = value["source"]
        if type(source) is not dict or set(source) != set(RebuildSource.__dataclass_fields__):
            raise ValueError("source shape")
        revision = source["revision"]
        if (type(revision) is not list or len(revision) != 6 or not isinstance(revision[0], str)
                or not revision[0] or not all(_integer(revision[index]) for index in (1, 2, 3, 5))
                or not _identifier(revision[4]) or revision[5] > revision[1]
                or revision[2] + revision[3] != revision[1]):
            raise ValueError("source revision")
        source["revision"] = SourceRevision(*revision)
        if (source["tracker"] not in ("canonical-v1", "rebuild-bridge-v1", "rebuild-bridge-conservative-v1")
                or not isinstance(source["schema_signature"], str) or not _HEX.fullmatch(source["schema_signature"])
                or not _integer(source["total"])):
            raise ValueError("source descriptor")
        for field in ("after_id", "high_id"):
            if source[field] is not None and not _identifier(source[field]):
                raise ValueError("source identifier")
        if ((source["total"] == 0) != (source["high_id"] is None)
                or (source["high_id"] is not None and source["after_id"] is not None
                    and source["high_id"] <= source["after_id"])):
            raise ValueError("source range")
        if (value["model"] != model or type(value["dimension"]) is not int or value["dimension"] != dimension
                or value["targets"] != list(targets) or value["input_digest"] is not None
                or not isinstance(value["generation"], str) or not re.fullmatch(r"[0-9a-f]{32}", value["generation"])
                or not all(_integer(value[field]) for field in ("total", "consumed", "outputs", "schema_version"))
                or value["total"] != source["total"] or value["outputs"] != value["consumed"]
                or value["consumed"] > value["total"] or type(value["complete"]) is not bool
                or (value["complete"] and value["consumed"] != value["total"])):
            raise ValueError("manifest invariant")
        cursor = value["cursor"]
        if value["consumed"] == 0:
            if cursor != source["after_id"] or type(cursor) is not type(source["after_id"]):
                raise ValueError("initial cursor")
        elif (not _identifier(cursor) or cursor > source["high_id"]
                or (source["after_id"] is not None and cursor <= source["after_id"])
                or ((value["consumed"] == value["total"]) != (cursor == source["high_id"]))):
            raise ValueError("consumed cursor")
        value["source"] = RebuildSource(**source)
        value["targets"] = tuple(value["targets"])
        return RebuildManifest(**value)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise TierSourceUntrusted("Malformed or incompatible tier source manifest") from error


def _receipt(plan: TierSourcePlan) -> str:
    return _json({"version": 1, "manifest": _hash(json.loads(_manifest_json(plan.manifest))),
                  "schema": plan.schema, "digest": plan.digest, "prefix_count": plan.prefix_count,
                  "prefix_cursor": plan.prefix_cursor})


def _parse_receipt(raw: str, manifest: RebuildManifest, schema: str) -> tuple[str, int, int | None]:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
        if (type(value) is not dict or set(value) != {"version", "manifest", "schema", "digest", "prefix_count", "prefix_cursor"}
                or type(value["version"]) is not int or value["version"] != 1
                or value["manifest"] != _hash(json.loads(_manifest_json(manifest))) or value["schema"] != schema
                or not isinstance(value["digest"], str) or not _HEX.fullmatch(value["digest"])
                or not _integer(value["prefix_count"])):
            raise ValueError("receipt invariant")
        cursor = value["prefix_cursor"]
        if ((cursor is not None and not _identifier(cursor)) or cursor != manifest.source.after_id
                or (value["prefix_count"] == 0) != (cursor is None)):
            raise ValueError("receipt prefix")
        return value["digest"], value["prefix_count"], cursor
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        raise TierSourceUntrusted("Malformed or incompatible tier pair receipt") from error


def _chain(digest: str, row_id: int, completion: bytes, separation: bytes) -> str:
    return hashlib.sha256(bytes.fromhex(digest) + struct.pack(">qQQ", row_id, len(completion), len(separation))
                          + completion + separation).hexdigest()


def _bounded_rows(cursor: sqlite3.Cursor) -> Iterator[tuple]:
    try:
        while rows := cursor.fetchmany(_PAGE_ROWS):
            yield from rows
    finally:
        cursor.close()


def _audit_pair(conn: sqlite3.Connection, plan: TierSourcePlan) -> None:
    """One snapshot, bounded resident rows; called only by plan/resume."""
    captured = plan.manifest.source
    lower = " AND id > ?" if captured.after_id is not None else ""
    args = (captured.high_id, captured.after_id) if lower else (captured.high_id,)
    total, high = conn.execute(
        f"SELECT COUNT(*), MAX(id) FROM main.messages WHERE id <= ?{lower}", args,
    ).fetchone()
    if (total, high) != (captured.total, captured.high_id):
        raise TierSourceUntrusted("Captured source bounds do not match the manifest")
    width = plan.manifest.dimension * 4
    readers = [_bounded_rows(conn.execute(
        f"SELECT rowid, CASE WHEN typeof(embedding)='blob' AND length(embedding)=? THEN embedding END "
        f"FROM main.{_quote(table)} ORDER BY rowid", (width,),
    )) for table in plan.manifest.targets]
    high = plan.manifest.cursor
    source = _bounded_rows(conn.execute(
        "SELECT id FROM main.messages WHERE id <= ? ORDER BY id", (high,),
    ))
    readers.append(source)
    digest, count = _SEED, 0
    try:
        while True:
            left, right, original = (next(reader, None) for reader in readers)
            if left is None and right is None and original is None:
                break
            if (left is None or right is None or original is None or left[0] != right[0] or left[0] != original[0]
                    or not _identifier(left[0]) or type(left[1]) is not bytes or type(right[1]) is not bytes
                    or len(left[1]) != width or len(right[1]) != width):
                raise TierSourceUntrusted("Tier pair bytes or source coverage do not match the receipt")
            digest = _chain(digest, left[0], left[1], right[1])
            count += 1
    finally:
        for reader in readers:
            reader.close()
    if digest != plan.digest or count != plan.prefix_count + plan.manifest.outputs:
        raise TierSourceUntrusted("Tier pair digest or consumed count changed")


def _fence(conn: sqlite3.Connection, plan: TierSourcePlan) -> None:
    if (conn is not plan.connection or conn.total_changes != plan.changes
            or conn.execute("PRAGMA main.data_version").fetchone()[0] != plan.data_version):
        raise TierSourceUntrusted("Connection changed; explicitly replan and revalidate the tier pair")


@contextmanager
def _transaction(conn: sqlite3.Connection, *, write: bool = False) -> Iterator[None]:
    if conn.in_transaction:
        raise TierSourceUntrusted("Tier source work requires an owned, clean connection")
    conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
    try:
        yield
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _check(conn: sqlite3.Connection, plan: TierSourcePlan) -> None:
    _fence(conn, plan)
    if _guard_schema(conn, plan.manifest.targets) != plan.schema:
        raise TierSourceUntrusted("Tier target schema changed")
    if (_metadata(conn, manifest_key(plan.manifest.targets)) != plan.saved_manifest
            or _metadata(conn, _receipt_key(plan.manifest.targets)) != plan.saved_receipt):
        raise TierSourceUntrusted("Tier generation or receipt changed")
    plan.manifest.source.check(conn)


def plan_tier_source(conn: sqlite3.Connection, *, model_id: str, dimension: int,
                     targets: tuple[str, str], force: bool = False) -> TierSourcePlan:
    """Plan full, resume, delta or complete without changing persistent state.

    A full plan can only initialize an already empty inactive pair. Existing
    unreceipted rows are refused, even with force. Snapshot consistency and
    exclusive cooperating writer ownership are required throughout this call.
    """
    _identity(model_id, dimension, targets)
    version, changes = conn.execute("PRAGMA main.data_version").fetchone()[0], conn.total_changes
    with _transaction(conn):
        schema = _guard_schema(conn, targets)
        saved = _metadata(conn, manifest_key(targets))
        receipt = _metadata(conn, _receipt_key(targets))
        if force or (saved is None and receipt is None):
            if any(conn.execute(f"SELECT 1 FROM main.{_quote(table)} LIMIT 1").fetchone() for table in targets):
                raise TierSourceUntrusted("Full initialization requires a separately owned empty inactive pair")
            source = capture_source(conn)
            manifest = replace(RebuildManifest.new(model_id, dimension, targets, source=source),
                               schema_version=conn.execute("PRAGMA main.schema_version").fetchone()[0])
            plan = TierSourcePlan("full", manifest, schema, _SEED, 0, None, conn, version, changes, saved, receipt)
        else:
            if saved is None or receipt is None:
                raise TierSourceUntrusted("Both manifest and pair receipt are required for compatibility")
            manifest = _parse_manifest(saved, model_id, dimension, targets)
            if manifest.schema_version != conn.execute("PRAGMA main.schema_version").fetchone()[0]:
                raise TierSourceUntrusted("Manifest schema generation changed")
            manifest.source.check(conn)
            digest, prefix, cursor = _parse_receipt(receipt, manifest, schema)
            plan = TierSourcePlan("resume", manifest, schema, digest, prefix, cursor, conn, version, changes, saved, receipt)
            _audit_pair(conn, plan)
            if manifest.complete:
                source = capture_source(conn, after_id=manifest.cursor)
                if source.total:
                    updated = replace(RebuildManifest.new(model_id, dimension, targets, source=source),
                                      schema_version=manifest.schema_version)
                    plan = replace(plan, action="delta", manifest=updated,
                                   prefix_count=prefix + manifest.outputs, prefix_cursor=manifest.cursor)
                else:
                    plan = replace(plan, action="complete")
        _fence(conn, plan)
    _fence(conn, plan)
    return plan


def _persist(conn: sqlite3.Connection, plan: TierSourcePlan) -> TierSourcePlan:
    saved, receipt = _manifest_json(plan.manifest), _receipt(plan)
    for key, value in ((manifest_key(plan.manifest.targets), saved), (_receipt_key(plan.manifest.targets), receipt)):
        conn.execute("INSERT OR REPLACE INTO main.metadata(key,value) VALUES (?,?)", (key, value))
    return replace(plan, saved_manifest=saved, saved_receipt=receipt, changes=conn.total_changes)


def initialize_tier_source(conn: sqlite3.Connection, plan: TierSourcePlan) -> TierSourcePlan:
    """Persist a planned generation; never create or clear target tables."""
    with _transaction(conn, write=True):
        _check(conn, plan)
        if plan.action not in ("full", "delta"):
            return plan
        if plan.action == "full" and any(
            conn.execute(f"SELECT 1 FROM main.{_quote(table)} LIMIT 1").fetchone() for table in plan.manifest.targets
        ):
            raise TierSourceUntrusted("Full target pair is no longer empty")
        updated = _persist(conn, replace(plan, action="resume"))
    return updated


def _read_rows(conn: sqlite3.Connection, plan: TierSourcePlan, size: int) -> tuple[tuple[SQLiteValue, ...], ...]:
    source = plan.manifest.source
    if source.high_id is None:
        return ()
    lower = " AND id > ?" if plan.manifest.cursor is not None else ""
    args = (source.high_id, plan.manifest.cursor, size) if lower else (source.high_id, size)
    return tuple(tuple(row) for row in conn.execute(
        f"SELECT {', '.join(_FIELDS)} FROM main.messages WHERE id <= ?{lower} ORDER BY id LIMIT ?", args,
    ))


def _same_rows(left: tuple[tuple[SQLiteValue, ...], ...], right: tuple[tuple[SQLiteValue, ...], ...]) -> bool:
    if len(left) != len(right):
        return False
    for current, pending in zip(left, right):
        if len(current) != len(pending):
            return False
        for actual, captured in zip(current, pending):
            if type(actual) is not type(captured):
                return False
            if isinstance(actual, float):
                if struct.pack(">d", actual) != struct.pack(">d", captured):
                    return False
            elif actual != captured:
                return False
    return True


def read_tier_page(conn: sqlite3.Connection, plan: TierSourcePlan, *, size: int = _PAGE_ROWS) -> TierSourcePage:
    """Release the read snapshot before returning any texts to the encoder."""
    if not _integer(size, minimum=1):
        raise ValueError("Page size must be a positive integer")
    if plan.action not in ("resume", "complete"):
        raise TierSourceUntrusted("Initialize the generation before reading")
    with _transaction(conn):
        _check(conn, plan)
        rows = _read_rows(conn, plan, size)
        if len(rows) != min(size, plan.manifest.total - plan.manifest.consumed):
            raise TierSourceUntrusted("Source range no longer matches its captured count")
    _fence(conn, plan)
    return TierSourcePage(plan.manifest.generation, plan.manifest.cursor, plan.manifest.consumed, rows)


def publish_tier_prefix(conn: sqlite3.Connection, plan: TierSourcePlan, page: TierSourcePage, *,
                        completion: Sequence[bytes], separation: Sequence[bytes]) -> tuple[TierSourcePlan, TierSourcePage]:
    """Atomically publish only an encoded page prefix and retain its pending suffix.

    The caller supplies already serialized vectors from the captured model. This
    function checks byte widths, but does not infer model identity from vectors.
    On failure the original immutable plan/page remain the retry checkpoint;
    after database write failures replan because total_changes includes rollbacks.
    """
    count = len(completion)
    if (plan.action != "resume" or plan.manifest.complete or page.generation != plan.manifest.generation
            or page.cursor != plan.manifest.cursor or page.consumed != plan.manifest.consumed
            or count < 1 or count != len(separation) or count > len(page.rows)):
        raise TierSourceUntrusted("Encoded prefix does not belong to the current pending page")
    width = plan.manifest.dimension * 4
    if any(type(blob) is not bytes or len(blob) != width for vectors in (completion, separation) for blob in vectors):
        raise ValueError("Both serialized vector widths must match the captured dimension")
    digest = plan.digest
    with _transaction(conn, write=True):
        _check(conn, plan)
        rows = _read_rows(conn, plan, len(page.rows))
        if not _same_rows(rows, page.rows):
            raise TierSourceUntrusted("Pending page does not match the source under writer ownership")
        for row, left, right in zip(rows[:count], completion, separation):
            for table, vector in zip(plan.manifest.targets, (left, right)):
                conn.execute(f"INSERT INTO main.{_quote(table)}(rowid,embedding) VALUES (?,?)", (row[0], vector))
            digest = _chain(digest, row[0], left, right)
        manifest = plan.manifest.advance(rows[count - 1][0], count, count)
        updated = _persist(conn, replace(plan, manifest=manifest, digest=digest))
    pending = TierSourcePage(manifest.generation, manifest.cursor, manifest.consumed, page.rows[count:])
    return updated, pending


def finish_tier_source(conn: sqlite3.Connection, plan: TierSourcePlan) -> TierSourceCompletion:
    """Certify captured coverage only; no registry, model or serving activation."""
    if plan.action not in ("resume", "complete") or plan.manifest.consumed != plan.manifest.total:
        raise TierSourceUntrusted("The captured source range still has pending inputs")
    with _transaction(conn, write=True):
        _check(conn, plan)
        current = capture_source(conn)
        covers_current = (current.total == plan.prefix_count + plan.manifest.outputs
                          and current.high_id == plan.manifest.cursor)
        updated = _persist(conn, replace(plan, action="complete", manifest=replace(plan.manifest, complete=True)))
    return TierSourceCompletion(updated, True, covers_current)


def require_current_completion_in_writer(conn: sqlite3.Connection, plan: TierSourcePlan) -> None:
    """Read-only final certification inside an owned BEGIN IMMEDIATE.

    sqlite3 cannot expose transaction lock kind: the caller must hold the writer
    and cooperative pair ownership. Unlike finish, this never persists or ends
    a transaction. One bounded-memory full pair audit occurs at activation,
    not after every page. Strict native table/model validation remains caller
    responsibility; the adapter attests stored bytes and source coverage.
    """
    if not conn.in_transaction:
        raise TierSourceUntrusted("Current completion requires caller writer ownership")
    if (plan.action != "complete" or not plan.manifest.complete
            or plan.manifest.consumed != plan.manifest.total
            or plan.saved_manifest is None or plan.saved_receipt is None):
        raise TierSourceUntrusted("A finished paired generation is required")
    _check(conn, plan)
    parsed = _parse_manifest(plan.saved_manifest, plan.manifest.model,
                             plan.manifest.dimension, plan.manifest.targets)
    if parsed != plan.manifest or _parse_receipt(plan.saved_receipt, parsed, plan.schema) != (
        plan.digest, plan.prefix_count, plan.prefix_cursor,
    ):
        raise TierSourceUntrusted("Completion does not match its persisted paired certificate")
    current = capture_source(conn)
    if (current.total != plan.prefix_count + plan.manifest.outputs
            or current.high_id != plan.manifest.cursor):
        raise TierSourceUntrusted("Captured pair does not cover the current source")
    _audit_pair(conn, plan)
