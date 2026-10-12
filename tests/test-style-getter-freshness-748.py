"""Bounded tracked style getter controls with actual source and in-memory SQLite."""
from __future__ import annotations

import hashlib
import json
import runpy
import sqlite3
import unittest
from pathlib import Path

BASE = runpy.run_path(str(Path(__file__).with_name("test-l0-style-maintenance-747.py")))
STORAGE, STYLE, MAINTENANCE = BASE["STORAGE"], BASE["STYLE"], BASE["MAINTENANCE"]


class StyleGetterFreshnessTests(unittest.TestCase):
    def db(self, *, certified: bool = True) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:", cached_statements=0)
        self.addCleanup(conn.close)
        conn.executescript(STORAGE._SCHEMA_SQL)
        STORAGE._initialize_maintenance_tracking(conn)
        conn.execute("INSERT INTO messages(content,sender,recipient,timestamp) VALUES('synthetic original style','Synthetic','Other','2020-01-01')")
        conn.commit()
        if certified:
            result = MAINTENANCE.run_style_maintenance(conn)
            self.assertEqual(result.outcome, "success")
        else:
            STYLE.build_entity_style_vectors(conn)
        return conn

    def rows(self, conn: sqlite3.Connection) -> list[tuple]:
        return conn.execute("SELECT * FROM entity_style_vectors").fetchall()

    def test_proof_fingerprints_match_actual_storage_contract(self) -> None:
        definitions = STORAGE._maintenance_trigger_definitions()
        definitions.update(STORAGE._STYLE_OUTPUT_TRIGGERS)
        hashes = {name: hashlib.sha256(" ".join(sql.split()).encode()).hexdigest()
                  for name, sql in definitions.items()}
        self.assertEqual(STYLE._STYLE_READ_TRIGGER_SHA256, hashes)

    def test_current_generation_and_case_normalization_remain_exact(self) -> None:
        conn = self.db()
        expected = json.loads(self.rows(conn)[0][1])
        self.assertEqual(STYLE.get_entity_style_vector(conn, "SYNTHETIC"), expected)
        self.assertIsNone(STYLE.get_entity_style_vector(conn, "missing"))
        self.assertFalse(conn.in_transaction)

    def test_every_source_correction_hides_old_rows_without_erasing(self) -> None:
        edits = {"content": "synthetic changed", "sender": "Changed", "recipient": "Changed",
                 "timestamp": "2021-01-01", "category": "changed", "modality": "changed",
                 "directive": 1, "metadata": '{"synthetic":true}'}
        for column, value in edits.items():
            with self.subTest(column=column):
                conn = self.db(); before = self.rows(conn)
                conn.execute(f"UPDATE messages SET {column}=?", (value,)); conn.commit()
                self.assertEqual(MAINTENANCE.observe_style(conn).health["freshness"], "correction_pending")
                self.assertIsNone(STYLE.get_entity_style_vector(conn, "Synthetic"))
                self.assertEqual(self.rows(conn), before)
                self.assertFalse(conn.in_transaction)

    def test_append_pending_prior_valid_generation_remains_readable(self) -> None:
        conn = self.db(); expected = STYLE.get_entity_style_vector(conn, "Synthetic")
        conn.execute("INSERT INTO messages(content,sender) VALUES('synthetic appended','Synthetic')"); conn.commit()
        self.assertEqual(MAINTENANCE.observe_style(conn).health["freshness"], "append_pending")
        self.assertEqual(STYLE.get_entity_style_vector(conn, "Synthetic"), expected)

    def test_successful_repair_restores_current_exact_generation(self) -> None:
        conn = self.db()
        conn.execute("UPDATE messages SET content='synthetic corrected',sender='Changed'"); conn.commit()
        self.assertIsNone(STYLE.get_entity_style_vector(conn, "Synthetic"))
        self.assertEqual(MAINTENANCE.run_style_maintenance(conn).outcome, "success")
        self.assertIsNone(STYLE.get_entity_style_vector(conn, "Synthetic"))
        self.assertEqual(STYLE.get_entity_style_vector(conn, "CHANGED"), json.loads(self.rows(conn)[0][1]))

    def test_direct_artifact_mutation_invalidates_the_generation(self) -> None:
        conn = self.db(); conn.execute("UPDATE entity_style_vectors SET vector='[123]'"); conn.commit()
        before = self.rows(conn)
        self.assertIsNone(STYLE.get_entity_style_vector(conn, "Synthetic"))
        self.assertEqual(self.rows(conn), before)

    def test_epoch_reset_and_missing_checkpoint_do_not_revive_old_rows(self) -> None:
        for statement in ("UPDATE maintenance_source_state SET epoch='synthetic changed epoch'",
                          "DELETE FROM maintenance_layers WHERE layer='style_vectors'",
                          "DROP TABLE maintenance_layers"):
            with self.subTest(statement=statement):
                conn = self.db(); before = self.rows(conn); conn.execute(statement); conn.commit()
                self.assertIsNone(STYLE.get_entity_style_vector(conn, "Synthetic"))
                self.assertEqual(self.rows(conn), before)

    def test_missing_or_replaced_tracking_and_output_triggers_are_unavailable(self) -> None:
        for name in STYLE._STYLE_READ_TRIGGER_SHA256:
            with self.subTest(name=name):
                conn = self.db(); before = self.rows(conn)
                conn.execute(f"DROP TRIGGER {name}"); conn.commit()
                self.assertIsNone(STYLE.get_entity_style_vector(conn, "Synthetic"))
                self.assertEqual(self.rows(conn), before)

    def test_untracked_legacy_and_direct_builder_payloads_remain_exact(self) -> None:
        conn = self.db(certified=False)
        self.assertEqual(STYLE.get_entity_style_vector(conn, "Synthetic"), json.loads(self.rows(conn)[0][1]))
        legacy = sqlite3.connect(":memory:"); self.addCleanup(legacy.close)
        legacy.execute("CREATE TABLE entity_style_vectors(entity TEXT PRIMARY KEY,vector TEXT)")
        legacy.execute("INSERT INTO entity_style_vectors VALUES('synthetic','[1]')"); legacy.commit()
        self.assertEqual(STYLE.get_entity_style_vector(legacy, "SYNTHETIC"), [1])

    def test_borrowed_source_change_is_hidden_and_caller_rollback_preserved(self) -> None:
        conn = self.db(); expected = STYLE.get_entity_style_vector(conn, "Synthetic")
        conn.execute("UPDATE messages SET content='synthetic pending caller change'")
        self.assertIsNone(STYLE.get_entity_style_vector(conn, "Synthetic"))
        self.assertTrue(conn.in_transaction)
        conn.rollback()
        self.assertEqual(STYLE.get_entity_style_vector(conn, "Synthetic"), expected)

    def test_bounded_inspection_never_reads_source_or_changes_schema(self) -> None:
        conn = self.db(); expected = STYLE.get_entity_style_vector(conn, "Synthetic"); seen = []
        def authorize(action: int, table: str | None, *_args: object) -> int:
            seen.append((action, table))
            if action == sqlite3.SQLITE_READ and table == "messages":
                return sqlite3.SQLITE_DENY
            if action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                          sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_DROP_TABLE):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        conn.set_authorizer(authorize)
        try:
            self.assertEqual(STYLE.get_entity_style_vector(conn, "Synthetic"), expected)
        finally:
            conn.set_authorizer(None)
        self.assertNotIn((sqlite3.SQLITE_READ, "messages"), seen)
        self.assertFalse(conn.in_transaction)

    def test_shadowed_tables_and_unknown_output_triggers_are_unavailable(self) -> None:
        for statement in ("CREATE TEMP TABLE messages(id INTEGER)",
                          "CREATE TRIGGER synthetic_foreign AFTER UPDATE ON entity_style_vectors BEGIN SELECT 1; END"):
            with self.subTest(statement=statement):
                conn = self.db(); conn.execute(statement); conn.commit()
                self.assertIsNone(STYLE.get_entity_style_vector(conn, "Synthetic"))

    def test_incompatible_dependency_and_hash_never_claim_current(self) -> None:
        for statement in ("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'",
                          "UPDATE maintenance_layers SET successful_dependency='{}' WHERE layer='style_vectors'",
                          "UPDATE maintenance_layers SET successful_coverage='unverified' WHERE layer='style_vectors'"):
            with self.subTest(statement=statement):
                conn = self.db(); conn.execute(statement); conn.commit()
                self.assertIsNone(STYLE.get_entity_style_vector(conn, "Synthetic"))


if __name__ == "__main__":
    unittest.main()
