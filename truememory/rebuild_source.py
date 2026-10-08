"""Bounded rebuild inputs and durable, source-attested progress.

Manifests contain identifiers and counters only. Page reads finish before the
caller computes embeddings; publication must revalidate under writer ownership.
"""

import datetime
import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace

from truememory.maintenance import MaintenanceUnavailableError, SourceRevision, read_source_revision
from truememory.storage import _MAINTENANCE_SOURCE_FIELDS, _MESSAGES_TABLE_SQL, _maintenance_trigger_definitions

_BRIDGE_TABLE = "truememory_rebuild_source_v1"
_BRIDGE_OWNER = "truememory.rebuild-source.v1"
_BRIDGE_TRIGGERS = tuple(f"truememory_rebuild_messages_v1_{suffix}" for suffix in ("ai", "au", "ad"))
_BRIDGE_SQL = f"""CREATE TABLE {_BRIDGE_TABLE} (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    owner TEXT NOT NULL CHECK (owner = '{_BRIDGE_OWNER}'),
    epoch TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (typeof(revision) = 'integer' AND revision >= 0),
    insert_count INTEGER NOT NULL CHECK (typeof(insert_count) = 'integer' AND insert_count >= 0),
    correction_count INTEGER NOT NULL CHECK (typeof(correction_count) = 'integer' AND correction_count >= 0),
    max_seen_message_id INTEGER NOT NULL,
    nonappend_revision INTEGER NOT NULL,
    source_signature TEXT NOT NULL,
    covered_columns TEXT NOT NULL,
    tracking_ready INTEGER NOT NULL CHECK (tracking_ready IN (0, 1))
)"""


class RebuildSourceChanged(RuntimeError):
    """The captured source or generation can no longer be published."""


def _normalized_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split())


def _source_schema(conn: sqlite3.Connection) -> tuple[tuple[str, ...], str, bool]:
    table = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages'").fetchone()
    columns = [tuple(row) for row in conn.execute("PRAGMA table_xinfo(messages)")]
    ordinary = {row[1]: row for row in columns if row[6] == 0}
    present = {row[1] for row in columns}
    identifier = ordinary.get("id")
    if (table is None or not table[0] or table[0].upper().startswith("CREATE VIRTUAL TABLE")
            or "content" not in present or identifier is None
            or identifier[2].upper() != "INTEGER" or identifier[5] != 1
            or sum(row[5] > 0 for row in columns) != 1):
        raise MaintenanceUnavailableError("Rebuild tracking requires messages with an INTEGER PRIMARY KEY id and content")
    indexes = []
    conservative = False
    for index in conn.execute("PRAGMA index_list(messages)"):
        index = tuple(index)
        name = index[1]
        definition = conn.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (name,)).fetchone()
        quoted = '"' + name.replace('"', '""') + '"'
        indexes.append([index, tuple(definition) if definition else None,
                        [tuple(row) for row in conn.execute(f"PRAGMA index_xinfo({quoted})")]])
        # An indexed INTEGER PRIMARY KEY (for example inline DESC) is not
        # a rowid alias. REPLACE on its separate rowid can delete a prefix row.
        conservative |= bool(index[2])
    covered = tuple(name for name in _MAINTENANCE_SOURCE_FIELDS if name in present)
    signature = hashlib.sha256(json.dumps([table[0], columns, indexes], separators=(",", ":")).encode()).hexdigest()
    return covered, signature, conservative


def _installed_triggers(conn: sqlite3.Connection, names: tuple[str, ...]) -> dict[str, str]:
    placeholders = ",".join("?" for _ in names)
    return {name: _normalized_sql(sql) for name, sql in conn.execute(
        f"SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name IN ({placeholders})", names,
    )}


def _canonical_revision(conn: sqlite3.Connection, covered: tuple[str, ...], conservative: bool) -> SourceRevision | None:
    if conservative or covered != _MAINTENANCE_SOURCE_FIELDS:
        return None
    table = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages'").fetchone()
    # Canonical triggers use declared column comparisons. Only our trusted
    # definition attests their byte-sensitive semantics; custom DDL stays on
    # the bridge without trying to parse arbitrary collations or affinities.
    trusted = _MESSAGES_TABLE_SQL.replace("CREATE TABLE IF NOT EXISTS ", "CREATE TABLE ", 1)
    if table is None or _normalized_sql(table[0]) != _normalized_sql(trusted):
        return None
    required = {"epoch", "revision", "insert_count", "correction_count", "max_seen_message_id",
                "nonappend_revision", "tracking_ready", "singleton"}
    if not required.issubset({row[1] for row in conn.execute("PRAGMA table_info(maintenance_source_state)")}):
        return None
    definitions = _maintenance_trigger_definitions()
    if _installed_triggers(conn, tuple(definitions)) != {name: _normalized_sql(sql) for name, sql in definitions.items()}:
        return None
    try:
        return read_source_revision(conn)
    except MaintenanceUnavailableError:
        return None


def _bridge_definitions(conn: sqlite3.Connection, covered: tuple[str, ...], conservative: bool) -> dict[str, str]:
    # SQLite's BLOB cast formats REALs as text and can lose precision. Numeric
    # comparison restores that distinction, but signed REAL zero needs a
    # conservative invalidation because SQL equality does not expose its sign.
    changed = " OR ".join(
        f"typeof(old.{name}) IS NOT typeof(new.{name}) "
        f"OR CAST(old.{name} AS BLOB) IS NOT CAST(new.{name} AS BLOB) "
        f"OR old.{name} COLLATE BINARY IS NOT new.{name} COLLATE BINARY "
        f"OR (typeof(old.{name}) = 'real' AND old.{name} = 0)"
        for name in covered
    )
    update_columns = ""
    if not conservative:
        columns = {row[1]: row[6] for row in conn.execute("PRAGMA table_xinfo(messages)")}
        if not any(columns[name] for name in covered):
            # Generated source fields can depend on any column. Ordinary
            # sources can limit zero invalidation to source writes, including
            # unshadowed rowid aliases that also update INTEGER PRIMARY KEY id.
            aliases = tuple(name for name in ("rowid", "_rowid_", "oid")
                            if name not in {column.lower() for column in columns})
            update_columns = " OF " + ", ".join((*covered, *aliases))
    update_guard = "" if conservative else f" WHEN {changed}"
    insert_fence = ("revision + 1" if conservative else
                    "CASE WHEN new.id <= max_seen_message_id THEN revision + 1 ELSE nonappend_revision END")
    insert, update, delete = _BRIDGE_TRIGGERS
    return {
        insert: f"""CREATE TRIGGER {insert} AFTER INSERT ON messages BEGIN
            UPDATE {_BRIDGE_TABLE} SET
                nonappend_revision = {insert_fence},
                max_seen_message_id = MAX(max_seen_message_id, new.id),
                revision = revision + 1, insert_count = insert_count + 1
            WHERE singleton = 1 AND tracking_ready = 1;
        END""",
        update: f"""CREATE TRIGGER {update} AFTER UPDATE{update_columns} ON messages{update_guard} BEGIN
            UPDATE {_BRIDGE_TABLE} SET nonappend_revision = revision + 1,
                max_seen_message_id = MAX(max_seen_message_id, new.id),
                revision = revision + 1, correction_count = correction_count + 1
            WHERE singleton = 1 AND tracking_ready = 1;
        END""",
        delete: f"""CREATE TRIGGER {delete} AFTER DELETE ON messages BEGIN
            UPDATE {_BRIDGE_TABLE} SET nonappend_revision = revision + 1,
                revision = revision + 1, correction_count = correction_count + 1
            WHERE singleton = 1 AND tracking_ready = 1;
        END""",
    }


def _bridge_row(conn: sqlite3.Connection) -> tuple | None:
    table = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (_BRIDGE_TABLE,)).fetchone()
    if table is None:
        return None
    if _normalized_sql(table[0]) != _normalized_sql(_BRIDGE_SQL):
        raise MaintenanceUnavailableError("Reserved rebuild tracking table has an unsupported definition")
    return conn.execute(
        f"SELECT epoch, revision, insert_count, correction_count, max_seen_message_id, nonappend_revision, "
        f"source_signature, covered_columns, tracking_ready FROM {_BRIDGE_TABLE} WHERE singleton = 1 AND owner = ?",
        (_BRIDGE_OWNER,),
    ).fetchone()


def _valid_bridge(conn: sqlite3.Connection, covered: tuple[str, ...], signature: str,
                  conservative: bool) -> SourceRevision | None:
    row = _bridge_row(conn)
    definitions = _bridge_definitions(conn, covered, conservative)
    if (row is None or not row[8] or row[6] != signature or row[7] != json.dumps(covered)
            or _installed_triggers(conn, _BRIDGE_TRIGGERS) != {name: _normalized_sql(sql) for name, sql in definitions.items()}):
        return None
    return SourceRevision(*row[:6])


def read_rebuild_revision(conn: sqlite3.Connection) -> tuple[str, SourceRevision, str]:
    """Read only; a scoped bridge never certifies unrelated maintenance layers."""
    covered, signature, conservative = _source_schema(conn)
    canonical = _canonical_revision(conn, covered, conservative)
    if canonical is not None:
        return "canonical-v1", canonical, signature
    bridge = _valid_bridge(conn, covered, signature, conservative)
    if bridge is None:
        raise MaintenanceUnavailableError("Rebuild source tracking is unavailable or its schema identity changed")
    return "rebuild-bridge-conservative-v1" if conservative else "rebuild-bridge-v1", bridge, signature


def ensure_rebuild_tracking(conn: sqlite3.Connection) -> None:
    """Install only this rebuild's tracker, under writer ownership before models."""
    with rebuild_transaction(conn):
        covered, signature, conservative = _source_schema(conn)
        canonical = _canonical_revision(conn, covered, conservative)
        bridge_row = _bridge_row(conn)
        installed = _installed_triggers(conn, _BRIDGE_TRIGGERS)
        if canonical is not None and not installed and (bridge_row is None or not bridge_row[8]):
            return
        if canonical is None and _valid_bridge(conn, covered, signature, conservative) is not None:
            return
    with rebuild_transaction(conn, write=True):
        covered, signature, conservative = _source_schema(conn)
        canonical = _canonical_revision(conn, covered, conservative)
        bridge_row = _bridge_row(conn)
        installed = _installed_triggers(conn, _BRIDGE_TRIGGERS)
        owned_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (_BRIDGE_TABLE,)).fetchone()
        if not owned_table and installed:
            raise MaintenanceUnavailableError("Reserved rebuild trigger names are already in use without owned tracking state")
        if canonical is not None:
            if owned_table:
                for name in installed:
                    conn.execute(f"DROP TRIGGER {name}")
                conn.execute(f"UPDATE {_BRIDGE_TABLE} SET tracking_ready = 0 WHERE singleton = 1")
            return
        if _valid_bridge(conn, covered, signature, conservative) is not None:
            return
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (_BRIDGE_TABLE,)).fetchone() is None:
            conn.execute(_BRIDGE_SQL)
        for name in installed:
            conn.execute(f"DROP TRIGGER {name}")
        for definition in _bridge_definitions(conn, covered, conservative).values():
            conn.execute(definition)
        # A repair cannot certify the interval before installation. A fresh
        # epoch also prevents a later fallback from reviving a retired bridge.
        conn.execute(
            f"INSERT OR REPLACE INTO {_BRIDGE_TABLE} VALUES (1, ?, ?, 0, 0, 0, "
            "(SELECT MAX(0, COALESCE(MAX(id), 0)) FROM messages), 0, ?, ?, 1)",
            (_BRIDGE_OWNER, uuid.uuid4().hex, signature, json.dumps(covered)),
        )


@contextmanager
def rebuild_transaction(conn: sqlite3.Connection, *, write: bool = False,
                        publication_fence: AbstractContextManager | None = None) -> Iterator[None]:
    owned = not conn.in_transaction
    conn.execute(("BEGIN IMMEDIATE" if write else "BEGIN") if owned else "SAVEPOINT rebuild_source")
    try:
        if write and not owned:
            conn.execute("UPDATE messages SET id = id WHERE 0")
        with publication_fence if publication_fence is not None else nullcontext():
            yield
            if owned:
                if write:
                    conn.commit()
                else:
                    conn.execute("COMMIT")
            else:
                conn.execute("RELEASE SAVEPOINT rebuild_source")
    except BaseException:
        if owned:
            conn.rollback()
        else:
            conn.execute("ROLLBACK TO SAVEPOINT rebuild_source")
            conn.execute("RELEASE SAVEPOINT rebuild_source")
        raise


@dataclass(frozen=True)
class RebuildSource:
    revision: SourceRevision
    after_id: int | None
    high_id: int | None
    total: int
    tracker: str = "canonical-v1"
    schema_signature: str | None = None

    def check(self, conn: sqlite3.Connection) -> None:
        try:
            tracker, revision, signature = read_rebuild_revision(conn)
        except MaintenanceUnavailableError as error:
            raise RebuildSourceChanged("Rebuild source tracking changed; start a fresh generation") from error
        if (tracker != self.tracker or signature != self.schema_signature
                or not revision.is_append_only_since(self.revision)):
            raise RebuildSourceChanged("Rebuild source changed; start a fresh generation")

    def page(self, conn: sqlite3.Connection, after_id: int | None, size: int, *,
             include_metadata: bool = True) -> list[dict]:
        if size < 1:
            raise ValueError("Rebuild page size must be positive")
        with rebuild_transaction(conn):
            self.check(conn)
            if self.high_id is None:
                return []
            lower = " AND id > ?" if after_id is not None else ""
            parameters = (self.high_id, after_id, size) if lower else (self.high_id, size)
            columns = (("id", "content", "sender", "recipient", "timestamp")
                       if include_metadata else ("id", "content"))
            cursor = conn.execute(
                f"SELECT {', '.join(columns)} FROM messages WHERE id <= ?{lower} ORDER BY id LIMIT ?", parameters,
            )
            try:
                return [dict(zip(columns, row)) for row in cursor]
            finally:
                cursor.close()


def capture_source(conn: sqlite3.Connection, *, after_id: int | None = None) -> RebuildSource:
    with rebuild_transaction(conn):
        tracker, revision, signature = read_rebuild_revision(conn)
        predicate = " WHERE id > ?" if after_id is not None else ""
        count, high = conn.execute(
            f"SELECT COUNT(*), MAX(id) FROM messages{predicate}",
            (after_id,) if predicate else (),
        ).fetchone()
        return RebuildSource(revision, after_id, high, count, tracker, signature)


def input_fingerprint(messages: list[dict], *, separation: bool) -> str:
    """Bind explicit caller input without copying its corpus or changing order."""
    digest = hashlib.sha256()
    fields = ("id", "content", "sender", "recipient", "timestamp") if separation else ("id", "content")
    for message in messages:
        payload = json.dumps([message.get(field) for field in fields], ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class RebuildManifest:
    generation: str
    model: str
    dimension: int
    targets: tuple[str, ...]
    source: RebuildSource | None
    input_digest: str | None
    total: int
    cursor: int | None = None
    consumed: int = 0
    outputs: int = 0
    complete: bool = False
    schema_version: int | None = None

    @classmethod
    def new(cls, model: str, dimension: int, targets: tuple[str, ...], *,
            source: RebuildSource | None = None, input_digest: str | None = None,
            total: int = 0) -> "RebuildManifest":
        return cls(uuid.uuid4().hex, model, dimension, targets, source, input_digest,
                   source.total if source is not None else total,
                   source.after_id if source is not None else None)

    def advance(self, last_id: int, consumed: int, outputs: int) -> "RebuildManifest":
        return replace(self, cursor=last_id, consumed=self.consumed + consumed,
                       outputs=self.outputs + outputs)

    def check(self, conn: sqlite3.Connection, key: str) -> None:
        if self.schema_version != conn.execute("PRAGMA schema_version").fetchone()[0]:
            raise RebuildSourceChanged("Rebuild database schema changed; start a fresh generation")
        if self.source is not None:
            self.source.check(conn)
        saved = load_manifest(conn, key)
        if saved != self:
            raise RebuildSourceChanged("Rebuild generation changed; retry with its current owner")


def manifest_key(targets: tuple[str, ...]) -> str:
    return "vec_source_v1:" + ":".join(targets)


def load_manifest(conn: sqlite3.Connection, key: str) -> RebuildManifest | None:
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    except sqlite3.OperationalError as error:
        if str(error) == "no such table: metadata":
            return None
        raise
    if row is None:
        return None
    try:
        payload = json.loads(row[0])
        if not isinstance(payload, dict) or payload.pop("version") != 1:
            return None
        source = payload["source"]
        if source is not None:
            source["revision"] = SourceRevision(*source["revision"])
            payload["source"] = RebuildSource(**source)
        payload["targets"] = tuple(payload["targets"])
        manifest = RebuildManifest(**payload)
        if not 0 <= manifest.consumed <= manifest.total or manifest.outputs < 0:
            return None
        return manifest
    except (TypeError, ValueError, KeyError):
        return None


def save_manifest(conn: sqlite3.Connection, key: str, manifest: RebuildManifest) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT)")
    payload = asdict(manifest)
    payload["version"] = 1
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key,value,updated_at) VALUES (?,?,?)",
        (key, json.dumps(payload, separators=(",", ":")),
         datetime.datetime.now(datetime.timezone.utc).isoformat()),
    )
