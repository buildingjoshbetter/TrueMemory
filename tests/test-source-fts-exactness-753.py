"""Actual storage SQL on synthetic SQLite only; no application/native imports."""

import ast
import builtins
import math
import sqlite3
import sys
import types
import unittest
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def allow_sqlite_operation(*_args: object) -> int:
    # Disabling authorizers with None requires Python 3.11 or newer.
    return sqlite3.SQLITE_OK


def load_primitives() -> tuple[types.ModuleType, types.ModuleType]:
    def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                    fromlist: tuple[str, ...] = (), level: int = 0) -> object:
        if name.split(".", 1)[0] not in sys.stdlib_module_names:
            raise AssertionError("Non-stdlib import forbidden: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    storage = types.ModuleType("synthetic_source_fts_storage")
    storage.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
    path = ROOT / "truememory/storage.py"
    exec(compile(ast.parse(path.read_text(encoding="utf-8")), str(path), "exec"), storage.__dict__)
    maintenance = types.ModuleType("synthetic_source_fts_maintenance")
    maintenance.__dict__.update(sqlite3=sqlite3, NamedTuple=NamedTuple)
    names = {"MaintenanceUnavailableError", "SourceRevision", "read_source_revision"}
    path = ROOT / "truememory/maintenance.py"
    body = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), maintenance.__dict__)
    return storage, maintenance


STORAGE, MAINTENANCE = load_primitives()
FIELDS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "content": "TEXT NOT NULL",
    "sender": "TEXT DEFAULT ''",
    "recipient": "TEXT DEFAULT ''",
    "timestamp": "TEXT DEFAULT ''",
    "category": "TEXT DEFAULT ''",
    "modality": "TEXT DEFAULT ''",
    "episode_id": "INTEGER",
    "emotional_valence": "REAL DEFAULT 0.0",
    "embedding_separation": "BLOB",
    "directive": "INTEGER DEFAULT 0",
    "metadata": "TEXT DEFAULT '{}'",
}


class SourceFTSFixture(unittest.TestCase):
    def database(self, replacements: dict[str, str] | None = None, *, extra: str = "",
                 suffix: str = "", migrate: bool = True) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        if replacements is not None or extra or suffix:
            fields = dict(FIELDS)
            fields.update(replacements or {})
            columns = ", ".join(name + " " + declaration for name, declaration in fields.items())
            conn.execute("CREATE TABLE messages (" + columns + (", " + extra if extra else "") + ")" + suffix)
        conn.executescript(STORAGE._SCHEMA_SQL)
        if migrate:
            STORAGE._migrate_messages_fts_trigger(conn)
            STORAGE._initialize_maintenance_tracking(conn)
        return conn

    def token(self, conn: sqlite3.Connection) -> MAINTENANCE.SourceRevision:
        return MAINTENANCE.read_source_revision(conn)

    def insert(self, conn: sqlite3.Connection, mid: int = 1, content: str = "synthetic original") -> None:
        conn.execute("INSERT INTO messages(id,content) VALUES (?,?)", (mid, content))

    def fts_rows(self, conn: sqlite3.Connection) -> list[tuple]:
        return conn.execute(
            "SELECT rowid,content,sender,recipient,category,modality FROM messages_fts ORDER BY rowid"
        ).fetchall()

    def assert_fts_matches(self, conn: sqlite3.Connection) -> None:
        self.assertEqual(self.fts_rows(conn), conn.execute(
            "SELECT id,content,sender,recipient,category,modality FROM messages ORDER BY id"
        ).fetchall())


class TestSourceComparisons(SourceFTSFixture):
    def test_declared_collations_do_not_hide_source_or_fts_edits(self) -> None:
        for collation, before, after in (("NOCASE", "mixed", "MIXED"), ("RTRIM", "mixed", "mixed ")):
            for field in STORAGE._MAINTENANCE_SOURCE_FIELDS[1:]:
                with self.subTest(collation=collation, field=field):
                    conn = self.database({field: "TEXT COLLATE " + collation})
                    self.insert(conn)
                    conn.execute(f"UPDATE messages SET {field}=?", (before,))
                    old = self.token(conn)
                    conn.execute(f"UPDATE messages SET {field}=?", (after,))
                    new = self.token(conn)
                    self.assertEqual(new.revision, old.revision + 1)
                    self.assertFalse(new.is_append_only_since(old))
                    self.assert_fts_matches(conn)

    def test_adjacent_reals_and_storage_types_are_distinct(self) -> None:
        for field in ("content", "metadata"):
            with self.subTest(field=field):
                conn = self.database({field: ""})
                self.insert(conn)
                previous = None
                for value in (1, 1.0, math.nextafter(1.0, 2), b"1", "1", None):
                    conn.execute(f"UPDATE messages SET {field}=?", (value,))
                    current = self.token(conn)
                    if previous is not None:
                        self.assertEqual(current.revision, previous.revision + 1)
                        self.assertFalse(current.is_append_only_since(previous))
                    self.assert_fts_matches(conn)
                    previous = current

    def test_signed_zero_is_conservative_but_derived_only_is_clean(self) -> None:
        for field in ("content", "metadata"):
            with self.subTest(field=field):
                conn = self.database({field: ""})
                self.insert(conn)
                conn.execute(f"UPDATE messages SET {field}=?", (-0.0,))
                conn.commit()
                self.assertEqual(math.copysign(1, conn.execute(f"SELECT {field} FROM messages").fetchone()[0]), -1)
                old = self.token(conn)
                changes = conn.total_changes
                conn.execute("UPDATE messages SET episode_id=4, emotional_valence=0.75")
                self.assertEqual(self.token(conn), old)
                self.assertEqual(conn.total_changes - changes, 1)
                for value in (0.0, 0.0, -0.0):
                    old = self.token(conn)
                    conn.execute(f"UPDATE messages SET {field}=?", (value,))
                    self.assertFalse(self.token(conn).is_append_only_since(old))
                    self.assert_fts_matches(conn)

    def test_generated_source_dependencies_are_observed(self) -> None:
        for field in ("content", "metadata"):
            for kind in ("VIRTUAL", "STORED"):
                with self.subTest(field=field, kind=kind):
                    conn = self.database({field: f"GENERATED ALWAYS AS (seed) {kind}"}, extra="seed")
                    if field == "content":
                        conn.execute("INSERT INTO messages(id,seed) VALUES (1,'synthetic generated')")
                    else:
                        conn.execute("INSERT INTO messages(id,content,seed) VALUES (1,'synthetic fixed','synthetic generated')")
                    old = self.token(conn)
                    conn.execute("UPDATE messages SET seed='synthetic changed'")
                    self.assertFalse(self.token(conn).is_append_only_since(old))
                    self.assert_fts_matches(conn)
                    conn.execute("UPDATE messages SET seed=?", (-0.0,))
                    old = self.token(conn)
                    conn.execute("UPDATE messages SET episode_id=9")
                    self.assertFalse(self.token(conn).is_append_only_since(old))
                    self.assert_fts_matches(conn)

    def test_all_unshadowed_rowid_aliases_are_single_corrections(self) -> None:
        conn = self.database()
        self.insert(conn)
        for mid, alias in enumerate(("id", "rowid", "_rowid_", "oid"), 20):
            old = self.token(conn)
            conn.execute(f"UPDATE messages SET {alias}=?", (mid,))
            new = self.token(conn)
            self.assertEqual(new.revision, old.revision + 1)
            self.assertEqual(new.correction_count, old.correction_count + 1)
            self.assertEqual(new.max_seen_message_id, mid)
            self.assert_fts_matches(conn)

    def test_shadowed_aliases_do_not_invalidate_real_zero_sources(self) -> None:
        conn = self.database({"metadata": ""}, extra="rowid REAL, _rowid_ REAL, oid REAL")
        self.insert(conn)
        conn.execute("UPDATE messages SET metadata=?", (-0.0,))
        conn.commit()
        old = self.token(conn)
        changes = conn.total_changes
        conn.execute("UPDATE messages SET rowid=10, _rowid_=20, oid=30")
        self.assertEqual(self.token(conn), old)
        self.assertEqual(conn.total_changes - changes, 1)
        conn.execute("UPDATE messages SET id=2")
        self.assertFalse(self.token(conn).is_append_only_since(old))
        self.assert_fts_matches(conn)

    def test_default_noops_nulls_append_and_rollback(self) -> None:
        conn = self.database()
        self.insert(conn)
        conn.commit()
        old = self.token(conn)
        changes = conn.total_changes
        conn.execute("UPDATE messages SET content=content,id=id,directive=directive,metadata=metadata")
        conn.execute("UPDATE messages SET episode_id=9,emotional_valence=0.25,embedding_separation=X'0102'")
        self.assertEqual(self.token(conn), old)
        self.assertEqual(conn.total_changes - changes, 2)
        conn.execute("UPDATE messages SET sender=NULL")
        null = self.token(conn)
        self.assertFalse(null.is_append_only_since(old))
        conn.execute("UPDATE messages SET sender=NULL")
        self.assertEqual(self.token(conn), null)
        conn.rollback()
        self.assertEqual(self.token(conn), old)
        self.insert(conn, 2)
        self.assertTrue(self.token(conn).is_append_only_since(old))
        self.assert_fts_matches(conn)


class TestUniqueReplacement(SourceFTSFixture):
    def test_unique_replacement_matrix_and_trigger_order(self) -> None:
        indexes = (
            "CREATE UNIQUE INDEX synthetic_unique ON messages(unique_key)",
            "CREATE UNIQUE INDEX synthetic_unique ON messages(lower(unique_key))",
            "CREATE UNIQUE INDEX synthetic_unique ON messages(unique_key) WHERE unique_key IS NOT NULL",
            "CREATE UNIQUE INDEX synthetic_unique ON messages(unique_key,pair_key)",
        )
        for index in indexes:
            for recursive in (0, 1):
                for reverse in (False, True):
                    for operation in ("insert", "update", "move"):
                        with self.subTest(index=index, recursive=recursive, reverse=reverse, operation=operation):
                            conn = self.database(extra="unique_key TEXT, pair_key TEXT")
                            conn.execute(index)
                            conn.execute(f"PRAGMA recursive_triggers={recursive}")
                            definitions = {**STORAGE._messages_fts_trigger_definitions(conn),
                                           **STORAGE._maintenance_trigger_definitions(conn)}
                            for name in definitions:
                                conn.execute(f"DROP TRIGGER {name}")
                            ordered = list(definitions.values())
                            for definition in (reversed(ordered) if reverse else ordered):
                                conn.execute(definition)
                            conn.executemany(
                                "INSERT INTO messages(id,content,unique_key,pair_key) VALUES (?,?,?,'shared')",
                                ((1, "synthetic victim", "A"), (2, "synthetic survivor", "B")),
                            )
                            old = self.token(conn)
                            key = "a" if "lower(" in index else "A"
                            if operation == "insert":
                                conn.execute("INSERT OR REPLACE INTO messages(id,content,unique_key,pair_key) "
                                             "VALUES (3,'synthetic replacement',?,'shared')", (key,))
                            elif operation == "update":
                                conn.execute("UPDATE OR REPLACE messages SET unique_key=? WHERE id=2", (key,))
                            else:
                                conn.execute("UPDATE OR REPLACE messages SET id=3,unique_key=? WHERE id=2", (key,))
                            self.assertFalse(self.token(conn).is_append_only_since(old))
                            self.assert_fts_matches(conn)
                            self.assertIsNone(conn.execute("SELECT rowid FROM messages_fts WHERE rowid=1").fetchone())

    def test_conservative_unique_updates_and_appends_without_victims(self) -> None:
        conn = self.database()
        self.insert(conn)
        old = self.token(conn)
        conn.execute("CREATE UNIQUE INDEX synthetic_unique ON messages(episode_id)")
        self.insert(conn, 2)
        self.assertFalse(self.token(conn).is_append_only_since(old))
        old = self.token(conn)
        conn.execute("UPDATE messages SET episode_id=7 WHERE id=2")
        new = self.token(conn)
        self.assertEqual(new.revision, old.revision + 1)
        self.assertEqual(new.correction_count, old.correction_count + 1)
        self.assertFalse(new.is_append_only_since(old))
        self.assert_fts_matches(conn)
        conn.execute("DROP INDEX synthetic_unique")
        old = self.token(conn)
        self.insert(conn, 3)
        self.assertTrue(self.token(conn).is_append_only_since(old))

    def test_multiple_hidden_victims_and_same_id_replacement(self) -> None:
        conn = self.database(extra="unique_a TEXT UNIQUE, unique_b TEXT UNIQUE")
        conn.execute("PRAGMA recursive_triggers=0")
        conn.executemany("INSERT INTO messages(id,content,unique_a,unique_b) VALUES (?,?,?,?)",
                         ((1, "synthetic first", "a", "x"), (2, "synthetic second", "b", "y")))
        old = self.token(conn)
        conn.execute("INSERT OR REPLACE INTO messages(id,content,unique_a,unique_b) VALUES (3,'synthetic third','a','y')")
        self.assertFalse(self.token(conn).is_append_only_since(old))
        self.assert_fts_matches(conn)
        self.assertEqual(len(self.fts_rows(conn)), 1)
        conn = self.database()
        self.insert(conn)
        old = self.token(conn)
        conn.execute("INSERT OR REPLACE INTO messages(id,content) VALUES (1,'synthetic replaced')")
        self.assertFalse(self.token(conn).is_append_only_since(old))
        self.assert_fts_matches(conn)


class TestMigrationAndContracts(SourceFTSFixture):
    def install_legacy_update(self, conn: sqlite3.Connection) -> None:
        conn.execute("DROP TRIGGER messages_au")
        conn.execute("""CREATE TRIGGER messages_au AFTER UPDATE ON messages
            WHEN old.content IS NOT new.content BEGIN
            DELETE FROM messages_fts WHERE rowid=old.id;
            INSERT INTO messages_fts(rowid,content,sender,recipient,category,modality)
            VALUES(new.id,new.content,new.sender,new.recipient,new.category,new.modality); END""")
        for name in ("messages_fts_cleanup_ai", "messages_fts_cleanup_au"):
            conn.execute(f"DROP TRIGGER {name}")

    def test_fts_migration_repairs_stale_missing_and_orphan_content_once(self) -> None:
        conn = self.database({"content": "TEXT COLLATE NOCASE"})
        self.insert(conn, 1, "synthetic before")
        self.insert(conn, 2, "synthetic missing")
        self.install_legacy_update(conn)
        conn.execute("UPDATE messages SET content='SYNTHETIC BEFORE' WHERE id=1")
        conn.execute("DELETE FROM messages_fts WHERE rowid=2")
        conn.execute("INSERT INTO messages_fts(rowid,content) VALUES (99,'synthetic orphan')")
        conn.commit()
        self.assertNotEqual(self.fts_rows(conn), conn.execute(
            "SELECT id,content,sender,recipient,category,modality FROM messages ORDER BY id").fetchall())
        STORAGE._migrate_messages_fts_trigger(conn)
        self.assert_fts_matches(conn)
        self.assertEqual(conn.execute("SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'missing'").fetchall(), [(2,)])
        changes = conn.total_changes
        schema = conn.execute("PRAGMA schema_version").fetchone()[0]
        STORAGE._migrate_messages_fts_trigger(conn)
        STORAGE._initialize_maintenance_tracking(conn)
        self.assertEqual(conn.total_changes, changes)
        self.assertEqual(conn.execute("PRAGMA schema_version").fetchone()[0], schema)

    def test_tracking_definition_migration_rotates_epoch_once(self) -> None:
        conn = self.database()
        self.insert(conn)
        conn.commit()
        old = self.token(conn)
        conn.execute("DROP TRIGGER messages_maintenance_unique_au")
        STORAGE._initialize_maintenance_tracking(conn)
        new = self.token(conn)
        self.assertNotEqual(old.epoch, new.epoch)
        self.assertFalse(new.is_append_only_since(old))
        changes = conn.total_changes
        STORAGE._initialize_maintenance_tracking(conn)
        self.assertEqual(self.token(conn), new)
        self.assertEqual(conn.total_changes, changes)

    def test_caller_rollback_restores_both_migrations(self) -> None:
        conn = self.database({"content": "TEXT COLLATE NOCASE"})
        self.insert(conn)
        self.install_legacy_update(conn)
        conn.execute("DROP TRIGGER messages_maintenance_unique_au")
        conn.commit()
        old = self.token(conn)
        old_sql = conn.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name").fetchall()
        old_fts = self.fts_rows(conn)
        conn.execute("BEGIN")
        conn.execute("UPDATE messages SET content='SYNTHETIC ORIGINAL'")
        STORAGE._migrate_messages_fts_trigger(conn)
        STORAGE._initialize_maintenance_tracking(conn)
        self.assertTrue(conn.in_transaction)
        self.assertNotEqual(self.token(conn).epoch, old.epoch)
        self.assert_fts_matches(conn)
        conn.rollback()
        self.assertEqual(self.token(conn), old)
        self.assertEqual(self.fts_rows(conn), old_fts)
        self.assertEqual(conn.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name").fetchall(), old_sql)

    def test_failed_fts_refill_restores_definitions_and_caller_writes(self) -> None:
        conn = self.database()
        self.insert(conn)
        self.install_legacy_update(conn)
        conn.commit()
        old_sql = conn.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name").fetchall()
        old_fts = self.fts_rows(conn)
        conn.execute("BEGIN")
        conn.execute("UPDATE messages SET episode_id=7")

        def deny_refill(action: int, name: str | None, column: str | None,
                        database: str | None, trigger: str | None) -> int:
            if action == sqlite3.SQLITE_INSERT and name == "messages_fts" and trigger is None:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(deny_refill)
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                STORAGE._migrate_messages_fts_trigger(conn)
        finally:
            conn.set_authorizer(allow_sqlite_operation)
        self.assertTrue(conn.in_transaction)
        self.assertEqual(conn.execute("SELECT episode_id FROM messages").fetchone()[0], 7)
        self.assertEqual(self.fts_rows(conn), old_fts)
        self.assertEqual(conn.execute("SELECT name,sql FROM sqlite_master WHERE type='trigger' ORDER BY name").fetchall(), old_sql)

    def test_unsupported_ids_remain_readable_but_cannot_issue_source_tokens(self) -> None:
        for declaration, suffix in (("INTEGER PRIMARY KEY DESC", ""),
                                    ("INTEGER PRIMARY KEY", " WITHOUT ROWID"),
                                    ("TEXT PRIMARY KEY", "")):
            with self.subTest(declaration=declaration, suffix=suffix):
                conn = self.database({"id": declaration}, suffix=suffix, migrate=False)
                with patch.object(STORAGE.sqlite3, "connect", return_value=conn), self.assertLogs(STORAGE.log, level="WARNING"):
                    self.assertIs(STORAGE.create_db(":memory:"), conn)
                self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
                with self.assertRaises(MAINTENANCE.MaintenanceUnavailableError):
                    self.token(conn)
                self.assertEqual(conn.execute("SELECT tracking_ready FROM maintenance_source_state").fetchone()[0], 0)

    def test_non_rowid_integer_primary_keys_accept_fts_incompatible_ids(self) -> None:
        for declaration, suffix in (("INTEGER PRIMARY KEY DESC", ""), ("INTEGER PRIMARY KEY", " WITHOUT ROWID")):
            with self.subTest(declaration=declaration, suffix=suffix):
                conn = sqlite3.connect(":memory:")
                self.addCleanup(conn.close)
                conn.execute(f"CREATE TABLE messages(id {declaration},content TEXT){suffix}")
                conn.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(content)")
                conn.execute("INSERT INTO messages VALUES (1.25,'synthetic real ID')")
                self.assertEqual(conn.execute("SELECT typeof(id) FROM messages").fetchone()[0], "real")
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute("INSERT INTO messages_fts(rowid,content) VALUES (1.25,'synthetic real ID')")
                if not suffix:
                    conn.execute("INSERT INTO messages VALUES (NULL,'synthetic null ID')")
                    conn.execute("INSERT INTO messages_fts(rowid,content) VALUES (NULL,'synthetic null ID')")
                    self.assertEqual(conn.execute("SELECT count(*) FROM messages_fts").fetchone()[0], 1)
                    self.assertEqual(conn.execute(
                        "SELECT count(*) FROM messages_fts f JOIN messages m ON m.id=f.rowid"
                    ).fetchone()[0], 0)

    def test_incomplete_source_schema_keeps_fts_usable(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE messages (" + ", ".join(
            name + " " + declaration for name, declaration in FIELDS.items() if name != "directive"
        ) + ")")
        conn.executescript(STORAGE._SCHEMA_SQL)
        STORAGE._migrate_messages_fts_trigger(conn)
        with self.assertLogs(STORAGE.log, level="WARNING"):
            STORAGE._initialize_maintenance_tracking(conn)
        with self.assertRaises(MAINTENANCE.MaintenanceUnavailableError):
            self.token(conn)
        self.insert(conn)
        self.assert_fts_matches(conn)

    def test_incomplete_fts_schema_remains_readable_without_refill(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE messages (" + ", ".join(
            name + " " + declaration for name, declaration in FIELDS.items() if name != "recipient"
        ) + ")")
        with patch.object(STORAGE.sqlite3, "connect", return_value=conn), \
                patch.object(STORAGE, "_migrate_messages_schema", return_value=None), \
                self.assertLogs(STORAGE.log, level="WARNING"):
            self.assertIs(STORAGE.create_db(":memory:"), conn)
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
        with self.assertRaises(MAINTENANCE.MaintenanceUnavailableError):
            self.token(conn)

    def test_default_definition_interface_still_matches_installed_triggers(self) -> None:
        conn = self.database()
        expected = STORAGE._maintenance_trigger_definitions()
        self.assertEqual(expected, STORAGE._maintenance_trigger_definitions(conn))
        for name, sql in expected.items():
            installed = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()[0]
            self.assertEqual(" ".join(installed.split()), " ".join(sql.split()))

    def test_repeated_initialization_does_not_read_source_or_fts_contents(self) -> None:
        conn = self.database()
        self.insert(conn)
        conn.commit()

        def deny_contents(action: int, name: str | None, column: str | None,
                          database: str | None, trigger: str | None) -> int:
            if action == sqlite3.SQLITE_READ and name in ("messages", "messages_fts", "messages_fts_content"):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(deny_contents)
        try:
            STORAGE._migrate_messages_fts_trigger(conn)
            STORAGE._initialize_maintenance_tracking(conn)
        finally:
            conn.set_authorizer(allow_sqlite_operation)
        self.assertEqual(conn.execute("SELECT content FROM messages").fetchall(), [("synthetic original",)])


class TestDefaultCost(SourceFTSFixture):
    def statement_steps(self, conn: sqlite3.Connection, sql: str) -> int:
        steps = 0

        def step() -> int:
            nonlocal steps
            steps += 1
            return 0

        conn.set_progress_handler(step, 1)
        try:
            conn.execute(sql).fetchall()
        finally:
            conn.set_progress_handler(None, 0)
        return steps

    def test_default_append_derived_and_revision_read_do_not_scan_messages(self) -> None:
        samples = []
        for count in (100, 10000):
            conn = self.database()
            conn.executemany("INSERT INTO messages(content) VALUES ('synthetic repeated source')", [()] * count)
            conn.commit()
            samples.append(tuple(self.statement_steps(conn, sql) for sql in (
                "INSERT INTO messages(content) VALUES ('synthetic appended source')",
                "UPDATE messages SET episode_id=5 WHERE id=1",
                "SELECT epoch,revision,insert_count,correction_count,max_seen_message_id,nonappend_revision,tracking_ready "
                "FROM maintenance_source_state WHERE singleton=1",
            )))
        for small, large in zip(*samples):
            self.assertLessEqual(large, small * 3, samples)


if __name__ == "__main__":
    unittest.main()
