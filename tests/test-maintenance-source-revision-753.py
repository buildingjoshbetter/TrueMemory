"""Transactional maintenance revisions on synthetic SQLite databases only."""

import builtins
import sqlite3
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_primitives() -> tuple[types.ModuleType, types.ModuleType]:
    modules = {}
    for name in ("storage", "_platform", "maintenance"):
        module = types.ModuleType(f"synthetic_maintenance_{name}")

        def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                        fromlist: tuple[str, ...] = (), level: int = 0) -> object:
            if name in ("truememory.storage", "truememory._platform"):
                return modules[name.rsplit(".", 1)[1]]
            return builtins.__import__(name, globals, locals, fromlist, level)

        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
        path = ROOT / "truememory" / f"{name}.py"
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
        modules[name] = module
    return modules["storage"], modules["maintenance"]


STORAGE, MAINTENANCE = load_primitives()


class TestSourceRevision(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-revision-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.sqlite"
        self.conn = STORAGE.create_db(self.path)
        self.addCleanup(self.conn.close)

    def token(self, conn: sqlite3.Connection | None = None):
        return MAINTENANCE.read_source_revision(conn or self.conn)

    def insert(self, mid: int, timestamp: str = "2026-01-01") -> None:
        self.conn.execute("INSERT INTO messages(id, content, timestamp) VALUES (?, 'synthetic source', ?)", (mid, timestamp))

    def test_insert_update_delete_are_transactional_and_visible_only_after_commit(self) -> None:
        reader = STORAGE.create_db(self.path)
        self.addCleanup(reader.close)
        initial = self.token()
        self.insert(1)
        self.assertEqual(self.token().revision, 1)
        self.assertEqual(self.token(reader), initial)
        self.conn.rollback()
        self.assertEqual(self.token(), initial)
        self.insert(1)
        self.conn.commit()
        inserted = self.token(reader)
        self.assertEqual((inserted.revision, inserted.insert_count, inserted.correction_count), (1, 1, 0))
        self.conn.execute("UPDATE messages SET content = 'synthetic changed' WHERE id = 1")
        self.conn.commit()
        self.assertEqual(self.token().correction_count, 1)
        self.conn.execute("DELETE FROM messages WHERE id = 1")
        self.conn.rollback()
        self.assertEqual(self.token().revision, 2)
        self.conn.execute("DELETE FROM messages WHERE id = 1")
        self.conn.commit()
        token = self.token(reader)
        self.assertEqual((token.revision, token.insert_count, token.correction_count, token.nonappend_revision), (3, 1, 2, 3))
        self.assertEqual(token.max_seen_message_id, 1)

    def test_every_source_field_and_directive_change_invalidates_append_proof(self) -> None:
        self.insert(1)
        self.conn.commit()
        fields = {
            "content": "synthetic changed", "sender": "synthetic-sender", "recipient": "synthetic-recipient",
            "timestamp": "2026-02-01", "category": "synthetic-category", "modality": "synthetic-modality",
            "directive": 1, "metadata": '{"synthetic": true}',
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                before = self.token()
                self.conn.execute(f"UPDATE messages SET {field} = ? WHERE id = 1", (value,))
                self.conn.commit()
                after = self.token()
                self.assertEqual(after.revision, before.revision + 1)
                self.assertEqual(after.nonappend_revision, after.revision)
                self.assertFalse(after.is_append_only_since(before))

    def test_noop_null_and_derived_only_updates_do_not_self_dirty(self) -> None:
        self.insert(1)
        self.conn.commit()
        before = self.token()
        self.conn.execute("UPDATE messages SET content = content, id = id, directive = directive, metadata = metadata")
        self.conn.execute("UPDATE messages SET episode_id = 42, emotional_valence = 0.25, embedding_separation = X'0102'")
        self.conn.commit()
        self.assertEqual(self.token(), before)
        self.conn.execute("UPDATE messages SET sender = NULL")
        self.conn.commit()
        changed = self.token()
        self.assertEqual(changed.revision, before.revision + 1)
        self.conn.execute("UPDATE messages SET sender = NULL")
        self.conn.commit()
        self.assertEqual(self.token(), changed)
        self.conn.execute("UPDATE messages SET sender = ''")
        self.conn.commit()
        self.assertEqual(self.token().revision, changed.revision + 1)

    def test_access_only_extension_columns_do_not_schedule_source_work(self) -> None:
        self.insert(1)
        self.conn.commit()
        before = self.token()
        self.conn.execute("ALTER TABLE messages ADD COLUMN access_count INTEGER DEFAULT 0")
        self.conn.execute("ALTER TABLE messages ADD COLUMN last_accessed TEXT")
        self.conn.execute("UPDATE messages SET access_count = 1, last_accessed = '2026-01-01'")
        self.conn.commit()
        self.assertEqual(self.token(), before)

    def test_rowid_alias_changes_are_single_corrections_and_advance_historical_max(self) -> None:
        self.insert(1)
        self.conn.commit()
        for new_id, alias in enumerate(("id", "rowid", "_rowid_", "oid"), 101):
            with self.subTest(alias=alias):
                before = self.token()
                self.conn.execute(f"UPDATE messages SET {alias} = ?", (new_id,))
                self.conn.commit()
                after = self.token()
                self.assertEqual(after.revision, before.revision + 1)
                self.assertEqual(after.max_seen_message_id, new_id)
                self.assertFalse(after.is_append_only_since(before))

    def test_append_certificate_rejects_id_holes_deletes_and_rolled_back_tokens(self) -> None:
        self.insert(10)
        self.conn.commit()
        before = self.token()
        self.insert(20)
        self.conn.commit()
        appended = self.token()
        self.assertTrue(appended.is_append_only_since(before))
        self.insert(15)
        self.assertFalse(self.token().is_append_only_since(appended))
        uncommitted = self.token()
        self.conn.rollback()
        self.assertEqual(self.token(), appended)
        self.assertFalse(self.token().is_append_only_since(uncommitted))
        self.insert(15)
        self.conn.commit()
        inserted_hole = self.token()
        self.assertEqual(inserted_hole.max_seen_message_id, 20)
        self.assertEqual(inserted_hole.nonappend_revision, inserted_hole.revision)
        self.conn.execute("DELETE FROM messages WHERE id = 20")
        self.conn.commit()
        self.assertEqual(self.token().max_seen_message_id, 20)
        self.insert(19)
        self.conn.commit()
        self.assertFalse(self.token().is_append_only_since(inserted_hole))

    def test_backdated_and_tied_timestamps_do_not_change_id_append_semantics(self) -> None:
        self.insert(1, "2026-02-01")
        self.conn.commit()
        before = self.token()
        self.insert(2, "2026-01-01")
        self.insert(3, "2026-02-01")
        self.conn.commit()
        self.assertTrue(self.token().is_append_only_since(before))
        self.conn.execute("UPDATE messages SET timestamp = '2026-03-01' WHERE id = 1")
        self.conn.commit()
        self.assertFalse(self.token().is_append_only_since(before))

    def test_insert_replace_invalidates_even_without_recursive_delete_triggers(self) -> None:
        self.insert(1)
        self.conn.commit()
        before = self.token()
        self.conn.execute("PRAGMA recursive_triggers = OFF")
        self.conn.execute("INSERT OR REPLACE INTO messages(id, content) VALUES (1, 'synthetic replacement')")
        self.conn.commit()
        self.assertGreater(self.token().revision, before.revision)
        self.assertFalse(self.token().is_append_only_since(before))

    def test_bulk_delete_and_insert_do_not_lose_mutation_counts(self) -> None:
        for mid in (1, 2, 3):
            self.insert(mid)
        self.conn.commit()
        self.conn.execute("DELETE FROM messages")
        self.insert(4)
        self.insert(5)
        self.conn.commit()
        token = self.token()
        self.assertEqual((token.revision, token.insert_count, token.correction_count), (8, 5, 3))
        self.assertEqual(token.max_seen_message_id, 5)

    def test_read_helper_preserves_caller_snapshot_and_transaction(self) -> None:
        writer = STORAGE.create_db(self.path)
        self.addCleanup(writer.close)
        self.conn.execute("BEGIN")
        old = self.token()
        writer.execute("INSERT INTO messages(content) VALUES ('synthetic new')")
        writer.commit()
        self.assertEqual(self.token(), old)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.token().revision, old.revision + 1)

    def test_reopen_is_idempotent_and_empty_outputs_are_not_marked_success(self) -> None:
        self.insert(7)
        self.conn.commit()
        before = self.token()
        schema_version = self.conn.execute("PRAGMA schema_version").fetchone()[0]
        reopened = STORAGE.create_db(self.path)
        self.addCleanup(reopened.close)
        self.assertEqual(self.token(reopened), before)
        self.assertEqual(reopened.execute("PRAGMA schema_version").fetchone()[0], schema_version)
        rows = reopened.execute("SELECT layer, outcome, successful_revision, full_rebuild_required FROM maintenance_layers").fetchall()
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row[1:] == ("pending", None, 1) for row in rows))

    def test_missing_trigger_bootstraps_new_epoch_instead_of_certifying_untracked_history(self) -> None:
        self.insert(10)
        self.conn.commit()
        before = self.token()
        self.conn.execute("DROP TRIGGER messages_maintenance_au")
        self.conn.execute("UPDATE messages SET content = 'synthetic untracked'")
        self.conn.commit()
        reopened = STORAGE.create_db(self.path)
        self.addCleanup(reopened.close)
        after = self.token(reopened)
        self.assertNotEqual(after.epoch, before.epoch)
        self.assertEqual(after.max_seen_message_id, 10)
        self.assertFalse(after.is_append_only_since(before))

    def test_missing_layer_registration_is_repaired_without_resetting_source_epoch(self) -> None:
        self.insert(1)
        self.conn.commit()
        before = self.token()
        self.conn.execute("DELETE FROM maintenance_layers WHERE layer = 'clusters'")
        self.conn.commit()
        STORAGE._initialize_maintenance_tracking(self.conn)
        self.assertEqual(self.token(), before)
        self.assertEqual(self.conn.execute("SELECT outcome, full_rebuild_required FROM maintenance_layers WHERE layer = 'clusters'").fetchone(), ("pending", 1))

    def test_unmigrated_connection_cannot_issue_token(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        with self.assertRaises(MAINTENANCE.MaintenanceUnavailableError):
            MAINTENANCE.read_source_revision(conn)

    def test_existing_rows_bootstrap_without_fabricated_historical_event_counts(self) -> None:
        path = Path(self.temp.name) / "synthetic-existing.sqlite"
        original = sqlite3.connect(path)
        original.executescript(STORAGE._SCHEMA_SQL)
        original.execute("INSERT INTO messages(id, content) VALUES (91, 'synthetic historical')")
        original.commit()
        original.close()
        reopened = STORAGE.create_db(path)
        self.addCleanup(reopened.close)
        token = self.token(reopened)
        self.assertEqual((token.revision, token.insert_count, token.correction_count, token.max_seen_message_id), (0, 0, 0, 91))

    def test_failed_trigger_installation_rolls_back_source_state_and_recovers(self) -> None:
        self.conn.execute("DROP TRIGGER messages_maintenance_au")
        self.conn.commit()
        before = self.token()

        def deny_create(action: int, name: str, *unused: str | None) -> int:
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_TRIGGER and name == "messages_maintenance_au" else sqlite3.SQLITE_OK

        self.conn.set_authorizer(deny_create)
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                STORAGE._initialize_maintenance_tracking(self.conn)
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.token(), before)
        STORAGE._initialize_maintenance_tracking(self.conn)
        self.assertNotEqual(self.token().epoch, before.epoch)

    def test_incomplete_legacy_schema_stays_open_but_cannot_issue_revision_token(self) -> None:
        path = Path(self.temp.name) / "synthetic-incomplete.sqlite"
        conn = sqlite3.connect(path)
        conn.executescript(STORAGE._SCHEMA_SQL.replace("    directive INTEGER DEFAULT 0,\n", ""))
        conn.close()
        with patch.object(STORAGE, "_migrate_messages_schema", return_value=None), self.assertLogs(STORAGE.log, level="WARNING"):
            incomplete = STORAGE.create_db(path)
        self.addCleanup(incomplete.close)
        incomplete.execute("INSERT INTO messages(content) VALUES ('synthetic legacy')")
        incomplete.commit()
        with self.assertRaises(MAINTENANCE.MaintenanceUnavailableError):
            self.token(incomplete)
        incomplete.execute("ALTER TABLE messages ADD COLUMN directive INTEGER DEFAULT 0")
        incomplete.commit()
        STORAGE._initialize_maintenance_tracking(incomplete)
        self.assertEqual(self.token(incomplete).max_seen_message_id, 1)
        incomplete.execute("UPDATE messages SET directive = 1")
        incomplete.commit()
        self.assertEqual(self.token(incomplete).correction_count, 1)


if __name__ == "__main__":
    unittest.main()
