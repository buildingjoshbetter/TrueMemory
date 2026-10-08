"""Synthetic SQLite ownership and transaction checks for Dunbar publication."""

import ast
import random
import sqlite3
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_modules() -> tuple[types.ModuleType, types.ModuleType]:
    modules = []
    for name in ("storage", "personality"):
        module = types.ModuleType("synthetic_dunbar_" + name)
        source = ROOT / "truememory" / (name + ".py")
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module in {"truememory.storage", "truememory.fts_search"}
        )]
        if modules:
            module._initialize_dunbar_ownership = modules[0]._initialize_dunbar_ownership
            module._dunbar_ownership_ready = modules[0]._dunbar_ownership_ready
        exec(compile(tree, str(source), "exec"), module.__dict__)
        modules.append(module)
    return tuple(modules)


STORAGE, PERSONALITY = load_modules()


def legacy_reference(conn: sqlite3.Connection, primary: str | None) -> dict:
    """Frozen pre-checkpoint computation, without its destructive publication."""
    if primary is None:
        return {}
    primary = primary.lower()
    rows = conn.execute("""
        SELECT LOWER(name) as name, COUNT(*) as cnt, MAX(ts) as last_ts FROM (
            SELECT recipient as name, timestamp as ts FROM messages WHERE LOWER(sender) = ? AND recipient != ''
            UNION ALL
            SELECT sender as name, timestamp as ts FROM messages WHERE LOWER(recipient) = ? AND sender != ''
        ) GROUP BY LOWER(name) ORDER BY cnt DESC
        """, (primary, primary)).fetchall()
    if not rows:
        return {}
    maximum = rows[0][1]
    result = {}
    for name, count, last in rows:
        score = count / maximum
        layer = "intimate" if score > 0.6 else "close" if score > 0.3 else "friend" if score > 0.1 else "acquaintance"
        result[name] = {"message_count": count, "last_interaction": last or "",
                        "dunbar_layer": layer, "strength": round(score, 3)}
    return result


class DunbarFixture(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="synthetic-dunbar-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "synthetic.sqlite"
        self.conn = self.open_db()

    def open_db(self) -> sqlite3.Connection:
        conn = STORAGE.create_db(self.path)
        conn.execute("PRAGMA busy_timeout=100")
        self.addCleanup(conn.close)
        return conn

    def seed(self) -> None:
        self.conn.executemany(
            "INSERT INTO messages(content,sender,recipient,timestamp) VALUES ('synthetic',?,?,?)",
            [("Synthetic-Primary", "Synthetic-Contact", "2026-01-01"),
             ("synthetic-contact", "synthetic-primary", "2026-01-02"),
             ("synthetic-primary", "synthetic-other", "2026-01-03")],
        )
        self.conn.commit()

    def build(self, primary: str | None = "Synthetic-Primary") -> dict:
        return PERSONALITY.build_dunbar_hierarchy(self.conn, primary)

    def snapshot(self, conn: sqlite3.Connection | None = None) -> tuple[list, list]:
        conn = conn or self.conn
        return (conn.execute("SELECT * FROM entity_relationships ORDER BY id").fetchall(),
                conn.execute("SELECT * FROM dunbar_generated_relationships ORDER BY relationship_id").fetchall())

    def compute_hook(self, action):
        original = PERSONALITY._compute_dunbar_hierarchy

        def compute(rows):
            action()
            return original(rows)

        return patch.object(PERSONALITY, "_compute_dunbar_hierarchy", compute)


class TestDunbarOwnership(DunbarFixture):
    def test_bounded_reference_corpus_preserves_full_dict_and_insertion_order(self) -> None:
        rng = random.Random(753)
        entities = ["synthetic-alpha", "SYNTHETIC-ALPHA", "synthetic-beta", "synthetic-gamma", "", None]
        self.conn.executemany(
            "INSERT INTO messages(content,sender,recipient,timestamp) VALUES ('synthetic',?,?,?)",
            [(rng.choice(entities), rng.choice(entities), rng.choice([None, "", "2026-01-01", "2026-01-02"]))
             for _ in range(120)],
        )
        self.conn.commit()
        for primary in entities:
            with self.subTest(primary=primary):
                expected = legacy_reference(self.conn, primary)
                self.assertEqual(list(self.build(primary).items()), list(expected.items()))

    def test_reference_ties_thresholds_rounding_case_and_timestamps(self) -> None:
        for name, count in (("alpha", 10), ("beta", 6), ("gamma", 3), ("delta", 1), ("epsilon", 6)):
            self.conn.executemany(
                "INSERT INTO messages(content,sender,recipient,timestamp) VALUES ('synthetic','PRIMARY',?,?)",
                [(name.upper() if i % 2 else name, None if i == 0 else "2026-01-01") for i in range(count)],
            )
        self.conn.commit()
        expected = legacy_reference(self.conn, "Primary")
        result = self.build("Primary")
        self.assertEqual(list(result.items()), list(expected.items()))
        self.assertEqual([result[name]["dunbar_layer"] for name in ("alpha", "beta", "gamma", "delta")],
                         ["intimate", "close", "friend", "acquaintance"])
        self.assertEqual(PERSONALITY.read_dunbar_coverage(self.conn)["managed_rows"], 5)

    def test_reference_preserves_self_contact_unicode_directive_and_empty_rules(self) -> None:
        self.seed()
        self.conn.executemany(
            "INSERT INTO messages(content,sender,recipient,timestamp,directive) VALUES ('synthetic',?,?,?,?)",
            [("synthetic-primary", "synthetic-primary", "", 1), ("Synthetic-Primary", "", "2026-01-01", 0),
             ("İ", "synthetic-unicode", None, 0), ("synthetic-primary", None, None, 0)],
        )
        self.conn.commit()
        for primary in ("Synthetic-Primary", "İ", "", None):
            with self.subTest(primary=primary):
                expected = legacy_reference(self.conn, primary)
                self.assertEqual(list(self.build(primary).items()), list(expected.items()))

    def test_legacy_contacts_and_other_producers_are_never_adopted_or_deleted(self) -> None:
        self.seed()
        for kind in ("contact", "synthetic-other", None):
            self.conn.execute("INSERT INTO entity_relationships(entity_a,entity_b,relationship_type) VALUES (?,?,?)",
                              ("synthetic-primary", "synthetic-contact", kind))
        self.conn.commit()
        legacy = self.snapshot()[0]
        self.build()
        self.build()
        self.assertEqual(self.snapshot()[0][:3], legacy)
        self.assertEqual(PERSONALITY.read_dunbar_coverage(self.conn),
                         {"status": "partial", "managed_rows": 2, "unowned_contacts": 1, "invalid_ownership": 0})
        self.assertEqual(len(self.snapshot()[0]), 5)

    def test_changed_primary_empty_and_none_retire_only_verified_owned_rows(self) -> None:
        self.seed()
        self.conn.execute("INSERT INTO entity_relationships(entity_a,entity_b,relationship_type) VALUES ('synthetic-legacy','synthetic-contact','contact')")
        self.conn.commit()
        legacy = self.snapshot()[0]
        self.build()
        self.build("Synthetic-Contact")
        generated = self.conn.execute(
            "SELECT r.entity_a FROM entity_relationships r JOIN dunbar_generated_relationships g ON r.id=g.relationship_id"
        ).fetchall()
        self.assertEqual(generated, [("synthetic-contact",)])
        self.assertEqual(self.build("synthetic-absent"), {})
        self.assertEqual(self.snapshot(), (legacy, []))
        self.build()
        self.assertEqual(self.build(None), {})
        self.assertEqual(self.snapshot(), (legacy, []))

    def test_every_generated_field_modification_relinquishes_ownership(self) -> None:
        self.seed()
        for column, value in (("entity_a", "synthetic-external"), ("entity_b", "synthetic-external"),
                              ("relationship_type", "synthetic-external"), ("strength", 0.123),
                              ("dunbar_layer", "synthetic-external"), ("last_interaction", None)):
            with self.subTest(column=column):
                self.build()
                row_id = self.snapshot()[1][0][0]
                self.conn.execute(f"UPDATE entity_relationships SET {column}=? WHERE id=?", (value, row_id))
                self.conn.commit()
                changed = self.conn.execute("SELECT * FROM entity_relationships WHERE id=?", (row_id,)).fetchone()
                self.assertIsNone(self.conn.execute("SELECT 1 FROM dunbar_generated_relationships WHERE relationship_id=?", (row_id,)).fetchone())
                self.build()
                self.assertEqual(self.conn.execute("SELECT * FROM entity_relationships WHERE id=?", (row_id,)).fetchone(), changed)
                self.assertIsNone(self.conn.execute("SELECT 1 FROM dunbar_generated_relationships WHERE relationship_id=?", (row_id,)).fetchone())

    def test_reused_ids_with_disabled_foreign_keys_are_preserved(self) -> None:
        self.seed()
        self.build()
        row_id = self.snapshot()[1][0][0]
        self.conn.execute("PRAGMA foreign_keys=OFF")
        self.conn.execute("DELETE FROM entity_relationships WHERE id=?", (row_id,))
        self.conn.execute("INSERT INTO entity_relationships(id,entity_a,entity_b,relationship_type) VALUES (?,'synthetic-reused','synthetic-contact','contact')", (row_id,))
        self.conn.commit()
        self.build()
        self.assertEqual(self.conn.execute("SELECT entity_a FROM entity_relationships WHERE id=?", (row_id,)).fetchone(), ("synthetic-reused",))
        self.assertIsNone(self.conn.execute("SELECT 1 FROM dunbar_generated_relationships WHERE relationship_id=?", (row_id,)).fetchone())

    def test_identical_id_and_fields_replacement_never_inherits_ownership(self) -> None:
        self.seed()
        self.conn.execute("PRAGMA foreign_keys=OFF")
        self.conn.execute("PRAGMA recursive_triggers=OFF")
        for delete_first in (False, True):
            with self.subTest(delete_first=delete_first):
                self.build()
                row_id = self.snapshot()[1][0][0]
                row = self.conn.execute("SELECT * FROM entity_relationships WHERE id=?", (row_id,)).fetchone()
                if delete_first:
                    self.conn.execute("DELETE FROM entity_relationships WHERE id=?", (row_id,))
                self.conn.execute("INSERT OR REPLACE INTO entity_relationships VALUES (?,?,?,?,?,?,?)", row)
                self.conn.commit()
                self.assertIsNone(self.conn.execute("SELECT 1 FROM dunbar_generated_relationships WHERE relationship_id=?", (row_id,)).fetchone())
                self.build(None)
                self.assertEqual(self.conn.execute("SELECT * FROM entity_relationships WHERE id=?", (row_id,)).fetchone(), row)

    def test_edit_then_restore_relinquishes_but_noop_update_preserves_ownership(self) -> None:
        self.seed()
        self.build()
        owned = self.snapshot()[1]
        self.conn.execute("UPDATE entity_relationships SET strength=strength,last_interaction=last_interaction")
        self.conn.commit()
        self.assertEqual(self.snapshot()[1], owned)
        row_id = owned[0][0]
        row = self.conn.execute("SELECT * FROM entity_relationships WHERE id=?", (row_id,)).fetchone()
        self.conn.execute("UPDATE entity_relationships SET id=? WHERE id=?", (row_id + 1000, row_id))
        self.conn.execute("UPDATE entity_relationships SET id=? WHERE id=?", (row_id, row_id + 1000))
        self.conn.commit()
        self.build(None)
        self.assertEqual(self.snapshot(), ([row], []))

    def test_fingerprint_mismatch_and_exact_matching_legacy_row_are_preserved(self) -> None:
        self.seed()
        self.build()
        row_id = self.snapshot()[1][0][0]
        row = self.conn.execute("SELECT * FROM entity_relationships WHERE id=?", (row_id,)).fetchone()
        self.conn.execute("UPDATE dunbar_generated_relationships SET row_fingerprint=? WHERE relationship_id=?", (bytes(32), row_id))
        legacy_id = self.conn.execute(
            "INSERT INTO entity_relationships(entity_a,entity_b,relationship_type,strength,dunbar_layer,last_interaction) VALUES (?,?,?,?,?,?)", row[1:]
        ).lastrowid
        self.conn.commit()
        self.assertEqual(PERSONALITY.read_dunbar_coverage(self.conn)["invalid_ownership"], 1)
        self.build(None)
        self.assertEqual([r[0] for r in self.snapshot()[0]], [row_id, legacy_id])
        self.assertEqual(self.snapshot()[1], [])

    def test_deleted_rows_cascade_and_orphaned_ownership_is_cleaned(self) -> None:
        self.seed()
        self.build()
        first, second = [row[0] for row in self.snapshot()[1]]
        self.conn.execute("DELETE FROM entity_relationships WHERE id=?", (first,))
        self.conn.commit()
        self.assertIsNone(self.conn.execute("SELECT 1 FROM dunbar_generated_relationships WHERE relationship_id=?", (first,)).fetchone())
        self.conn.execute("PRAGMA foreign_keys=OFF")
        self.conn.execute("DROP TRIGGER dunbar_relationships_ad")
        self.conn.execute("DELETE FROM entity_relationships WHERE id=?", (second,))
        self.conn.commit()
        self.assertEqual(PERSONALITY.read_dunbar_coverage(self.conn)["invalid_ownership"], 1)
        self.build(None)
        self.assertEqual(self.snapshot(), ([], []))


class TestDunbarTransactions(DunbarFixture):
    def test_source_and_ownership_changes_during_compute_reject_publication(self) -> None:
        self.seed()
        self.build()
        writer = self.open_db()
        changes = (
            "UPDATE messages SET sender='synthetic-changed' WHERE id=1",
            "UPDATE messages SET recipient='synthetic-changed' WHERE id=1",
            "UPDATE messages SET timestamp='2026-02-01' WHERE id=1",
            "UPDATE messages SET id=id+100 WHERE id=2",
            "INSERT INTO messages(content,sender,recipient) VALUES ('synthetic','synthetic-primary','synthetic-new')",
            "DELETE FROM messages WHERE id=1",
            "UPDATE entity_relationships SET strength=0.125 WHERE id=(SELECT MIN(id) FROM entity_relationships)",
            "UPDATE dunbar_generated_relationships SET generation='synthetic-external'",
        )
        for sql in changes:
            with self.subTest(operation=sql):
                captured = []

                def mutate():
                    writer.execute(sql)
                    writer.commit()
                    captured.append(self.snapshot(writer))

                with self.compute_hook(mutate):
                    with self.assertRaises(sqlite3.OperationalError):
                        self.build()
                self.assertEqual(self.snapshot(), captured[0])
                self.assertFalse(self.conn.in_transaction)

    def test_unrelated_writer_and_reader_can_complete_during_compute(self) -> None:
        self.seed()
        self.build()
        writer = self.open_db()
        old = self.snapshot()

        def observe():
            self.assertFalse(self.conn.in_transaction)
            self.assertEqual(self.snapshot(writer), old)
            writer.execute("INSERT INTO metadata(key,value) VALUES ('synthetic','committed')")
            writer.commit()

        with self.compute_hook(observe):
            self.assertEqual(self.build(), legacy_reference(self.conn, "Synthetic-Primary"))

    def test_caller_changes_and_outputs_remain_rollbackable(self) -> None:
        self.seed()
        self.build()
        old = self.snapshot()
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-pending','retained')")
        self.conn.execute("UPDATE messages SET recipient='synthetic-caller' WHERE id=1")
        self.build()
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.snapshot(), old)
        self.assertIsNone(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-pending'").fetchone())

    def test_stale_caller_snapshot_fails_without_committing_or_restarting_it(self) -> None:
        self.seed()
        self.build()
        old = self.snapshot()
        writer = self.open_db()
        self.conn.execute("BEGIN")
        self.conn.execute("SELECT count(*) FROM messages").fetchone()

        def mutate():
            writer.execute("UPDATE messages SET recipient='synthetic-new' WHERE id=1")
            writer.commit()

        with self.compute_hook(mutate), self.assertRaises(sqlite3.OperationalError):
            self.build()
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.snapshot(), old)

    def test_compute_failure_and_cancellation_preserve_previous_generation(self) -> None:
        self.seed()
        self.build()
        old = self.snapshot()
        for failure in (ValueError("synthetic"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                with patch.object(PERSONALITY, "_compute_dunbar_hierarchy", side_effect=failure):
                    with self.assertRaises(type(failure)):
                        self.build()
                self.assertEqual(self.snapshot(), old)

    def test_cancellation_after_owned_rows_are_deleted_rolls_back_publication(self) -> None:
        self.seed()
        self.build()
        old = self.snapshot()
        with patch.object(PERSONALITY.uuid, "uuid4", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self.build()
        self.assertEqual(self.snapshot(), old)
        self.assertFalse(self.conn.in_transaction)

    def test_midpublication_failure_keeps_ledger_and_rows_atomic_after_caller_commit(self) -> None:
        self.seed()
        self.build()
        old = self.snapshot()
        self.conn.execute("CREATE TRIGGER synthetic_fail BEFORE INSERT ON dunbar_generated_relationships BEGIN SELECT RAISE(ABORT,'synthetic publication'); END")
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-pending','retained')")
        with self.assertRaises(sqlite3.IntegrityError):
            self.build()
        self.conn.commit()
        self.assertEqual(self.snapshot(), old)
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-pending'").fetchone(), ("retained",))

    def test_commit_and_release_failure_roll_back_before_cleanup(self) -> None:
        self.seed()
        self.build()
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                old = self.snapshot()
                if borrowed:
                    self.conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('synthetic-pending','retained')")
                denied = []

                def authorizer(action, operation, *unused):
                    expected = sqlite3.SQLITE_SAVEPOINT if borrowed else sqlite3.SQLITE_TRANSACTION
                    terminal = "RELEASE" if borrowed else "COMMIT"
                    if action == expected and operation == terminal and not denied:
                        denied.append(operation)
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK

                with self.compute_hook(lambda: self.conn.set_authorizer(authorizer)):
                    try:
                        with self.assertRaises(sqlite3.DatabaseError):
                            self.build()
                    finally:
                        self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                self.assertEqual(len(denied), 1)
                self.conn.commit()
                self.assertEqual(self.snapshot(), old)

    def test_rollback_failure_does_not_release_failed_savepoint(self) -> None:
        self.seed()
        self.build()
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-pending','retained')")
        operations = []

        def authorizer(action, operation, *unused):
            if action == sqlite3.SQLITE_SAVEPOINT:
                operations.append(operation)
                if operation == "ROLLBACK":
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        original = PERSONALITY._initialize_dunbar_ownership

        def fail(conn):
            original(conn)
            conn.set_authorizer(authorizer)
            raise ValueError("synthetic publication")

        with patch.object(PERSONALITY, "_initialize_dunbar_ownership", fail):
            try:
                with self.assertRaises(sqlite3.DatabaseError):
                    self.build()
            finally:
                self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                self.conn.rollback()
        self.assertEqual(operations, ["ROLLBACK"])

    def test_reader_sees_old_output_and_ledger_until_publication_commits(self) -> None:
        self.seed()
        self.build()
        old = self.snapshot()
        reader = self.open_db()
        original = PERSONALITY._dunbar_fingerprint
        observed = []

        def observe(rows):
            if self.conn.in_transaction and self.conn.total_changes > before:
                observed.append(self.snapshot(reader))
            return original(rows)

        before = self.conn.total_changes
        with patch.object(PERSONALITY, "_dunbar_fingerprint", observe):
            self.build("synthetic-contact")
        self.assertTrue(observed)
        self.assertTrue(all(snapshot == old for snapshot in observed))
        self.assertNotEqual(self.snapshot(reader), old)


class TestDunbarMigrationAndCoverage(DunbarFixture):
    def test_additive_ledger_preserves_existing_rows_and_source_epoch(self) -> None:
        self.seed()
        for name in STORAGE._DUNBAR_OWNERSHIP_TRIGGERS:
            self.conn.execute(f"DROP TRIGGER {name}")
        self.conn.execute("DROP TABLE dunbar_generated_relationships")
        self.conn.execute("INSERT INTO entity_relationships(entity_a,entity_b,relationship_type) VALUES ('synthetic-legacy','synthetic-contact','contact')")
        self.conn.commit()
        epoch = self.conn.execute("SELECT epoch FROM maintenance_source_state").fetchone()
        self.assertEqual(PERSONALITY.read_dunbar_coverage(self.conn)["unowned_contacts"], 1)
        self.assertFalse(PERSONALITY._dunbar_has_ledger(self.conn))
        reopened = self.open_db()
        version = reopened.execute("PRAGMA schema_version").fetchone()
        self.assertEqual(reopened.execute("SELECT epoch FROM maintenance_source_state").fetchone(), epoch)
        self.open_db()
        self.assertEqual(reopened.execute("PRAGMA schema_version").fetchone(), version)
        self.assertEqual(PERSONALITY.read_dunbar_coverage(reopened)["unowned_contacts"], 1)

    def test_missing_or_replaced_trigger_invalidates_all_ownership_without_adoption(self) -> None:
        self.seed()
        epoch = self.conn.execute("SELECT epoch FROM maintenance_source_state").fetchone()
        for index, name in enumerate(STORAGE._DUNBAR_OWNERSHIP_TRIGGERS):
            with self.subTest(trigger=name):
                self.build()
                relationships, owned = self.snapshot()
                self.assertTrue(owned)
                self.conn.execute(f"DROP TRIGGER {name}")
                if index == 1:
                    self.conn.execute(f"CREATE TRIGGER {name} AFTER UPDATE ON entity_relationships BEGIN SELECT 1; END")
                self.conn.commit()
                self.assertEqual(PERSONALITY.read_dunbar_coverage(self.conn)["managed_rows"], 0)
                STORAGE._initialize_dunbar_ownership(self.conn)
                self.assertTrue(STORAGE._dunbar_ownership_ready(self.conn))
                self.assertEqual(self.snapshot(), (relationships, []))
                self.assertEqual(self.conn.execute("SELECT epoch FROM maintenance_source_state").fetchone(), epoch)
                version = self.conn.execute("PRAGMA schema_version").fetchone()
                changes = self.conn.total_changes
                STORAGE._initialize_dunbar_ownership(self.conn)
                self.assertEqual(self.conn.execute("PRAGMA schema_version").fetchone(), version)
                self.assertEqual(self.conn.total_changes, changes)

    def test_trigger_repair_failure_rolls_back_ledger_and_preserves_caller_work(self) -> None:
        self.seed()
        self.build()
        self.conn.execute("DROP TRIGGER dunbar_relationships_ai")
        self.conn.commit()
        old = self.snapshot()
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-pending','retained')")

        def deny_create(action, *unused):
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_TRIGGER else sqlite3.SQLITE_OK

        self.conn.set_authorizer(deny_create)
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                STORAGE._initialize_dunbar_ownership(self.conn)
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertTrue(self.conn.in_transaction)
        self.conn.commit()
        self.assertEqual(self.snapshot(), old)
        self.assertFalse(STORAGE._dunbar_ownership_ready(self.conn))
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-pending'").fetchone(), ("retained",))

    def test_trigger_migration_commit_or_release_failure_keeps_old_state(self) -> None:
        self.seed()
        self.build()
        self.conn.execute("DROP TRIGGER dunbar_relationships_ai")
        self.conn.commit()
        old = self.snapshot()
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                if borrowed:
                    self.conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('synthetic-pending','retained')")
                denied = []

                def authorizer(action, operation, name, *unused):
                    terminal = action == sqlite3.SQLITE_TRANSACTION and operation == "COMMIT"
                    if borrowed:
                        terminal = action == sqlite3.SQLITE_SAVEPOINT and operation == "RELEASE" and name == "truememory_dunbar_schema"
                    if terminal and not denied:
                        denied.append(operation)
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK

                self.conn.set_authorizer(authorizer)
                try:
                    with self.assertRaises(sqlite3.DatabaseError):
                        STORAGE._initialize_dunbar_ownership(self.conn)
                finally:
                    self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                self.assertEqual(len(denied), 1)
                self.assertEqual(self.conn.in_transaction, borrowed)
                self.conn.commit()
                self.assertEqual(self.snapshot(), old)
                self.assertFalse(STORAGE._dunbar_ownership_ready(self.conn))

    def test_minimal_schema_ledger_migration_and_output_rollback_with_caller(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY,sender TEXT,recipient TEXT,timestamp TEXT)")
        conn.execute("CREATE TABLE entity_relationships(id INTEGER PRIMARY KEY AUTOINCREMENT,entity_a TEXT,entity_b TEXT,relationship_type TEXT,strength REAL,dunbar_layer TEXT,last_interaction TEXT)")
        conn.execute("INSERT INTO messages VALUES (1,'synthetic-primary','synthetic-contact','2026-01-01')")
        self.assertEqual(len(PERSONALITY.build_dunbar_hierarchy(conn, "synthetic-primary")), 1)
        self.assertTrue(conn.in_transaction)
        self.assertTrue(PERSONALITY._dunbar_has_ledger(conn))
        conn.rollback()
        self.assertFalse(PERSONALITY._dunbar_has_ledger(conn))
        self.assertEqual(conn.execute("SELECT * FROM messages").fetchall(), [])
        self.assertEqual(conn.execute("SELECT * FROM entity_relationships").fetchall(), [])

    def test_coverage_is_one_snapshot_and_does_not_commit_caller(self) -> None:
        self.seed()
        self.build()
        writer = self.open_db()
        original = PERSONALITY._dunbar_owned_rows

        def interleave(conn):
            yield from original(conn)
            writer.execute("INSERT INTO entity_relationships(entity_a,entity_b,relationship_type) VALUES ('synthetic-external','synthetic-contact','contact')")
            writer.commit()

        with patch.object(PERSONALITY, "_dunbar_owned_rows", interleave):
            self.assertEqual(PERSONALITY.read_dunbar_coverage(self.conn)["unowned_contacts"], 0)
        self.assertEqual(PERSONALITY.read_dunbar_coverage(self.conn)["unowned_contacts"], 1)
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-pending','retained')")
        PERSONALITY.read_dunbar_coverage(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertIsNone(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-pending'").fetchone())

    def test_failed_coverage_read_ends_only_its_owned_snapshot(self) -> None:
        self.seed()
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                if borrowed:
                    self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-pending','retained')")
                with patch.object(PERSONALITY, "_dunbar_owned_rows", side_effect=ValueError("synthetic")):
                    with self.assertRaises(ValueError):
                        PERSONALITY.read_dunbar_coverage(self.conn)
                self.assertEqual(self.conn.in_transaction, borrowed)
                self.conn.rollback()
                self.assertIsNone(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-pending'").fetchone())


if __name__ == "__main__":
    unittest.main()
