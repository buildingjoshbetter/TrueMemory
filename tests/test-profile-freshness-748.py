"""Actual profile token and incremental deferral controls; synthetic in-memory SQLite."""
from __future__ import annotations

import hashlib
import json
import runpy
import sqlite3
import unittest
from pathlib import Path

LOAD = runpy.run_path(str(Path(__file__).with_name("test-source-invalidation-748.py")))
P, _STYLE = LOAD["_load_modules"]()
STORAGE = runpy.run_path(str(Path(__file__).with_name("test-l0-style-accumulator-747.py")))["STORAGE"]


class ProfileFreshnessTests(unittest.TestCase):
    def db(self, *, certified: bool = True) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:", cached_statements=0); self.addCleanup(conn.close)
        conn.executescript(STORAGE._SCHEMA_SQL); STORAGE._initialize_maintenance_tracking(conn)
        conn.execute("INSERT INTO messages(content,sender,recipient,timestamp) VALUES('synthetic code original','Synthetic','Other','2020-01-01')")
        conn.commit()
        if certified:
            P.build_entity_profiles(conn)
        return conn

    def rows(self, conn: sqlite3.Connection) -> list[tuple]:
        return conn.execute("SELECT * FROM entity_profiles ORDER BY entity").fetchall()

    def append(self, conn: sqlite3.Connection, *, content: str = "synthetic code append", sender: str = "Synthetic", recipient: str = "Other", directive: int = 0) -> None:
        conn.execute("INSERT INTO messages(content,sender,recipient,timestamp,directive) VALUES(?,?,?,'2020-01-02',?)", (content,sender,recipient,directive)); conn.commit()

    def test_source_proof_matches_actual_storage_contract(self) -> None:
        self.assertEqual(P._PROFILE_SOURCE_TRIGGER_SHA256,
            {name: hashlib.sha256(" ".join(sql.split()).encode()).hexdigest()
             for name, sql in STORAGE._maintenance_trigger_definitions().items()})

    def test_bulk_certifies_exact_current_generation(self) -> None:
        conn = self.db(); profile = P.get_entity_profile(conn, "SYNTHETIC")
        self.assertEqual(profile["message_count"], 1)
        source, previous = P._profile_read_state(conn)
        self.assertEqual(source, previous)
        self.assertFalse(conn.in_transaction)

    def test_every_correction_hides_retained_rows_and_defers_increment(self) -> None:
        for column, value in {"content":"synthetic changed","sender":"Changed","recipient":"Changed",
                              "timestamp":"2021-01-01","directive":1,"metadata":'{"synthetic":true}'}.items():
            with self.subTest(column=column):
                conn = self.db(); before = self.rows(conn)
                conn.execute(f"UPDATE messages SET {column}=?", (value,)); conn.commit()
                self.assertIsNone(P.get_entity_profile(conn, "Synthetic"))
                with self.assertRaises(P.ProfileRebuildRequired):
                    P.update_entity_profile_incremental(conn, "Synthetic", "synthetic partial", "Other")
                self.assertEqual(self.rows(conn), before)
                self.assertFalse(conn.in_transaction)

    def test_exact_ordinary_append_extends_certificate_without_formula_change(self) -> None:
        conn = self.db(); self.append(conn)
        self.assertEqual(P.get_entity_profile(conn, "Synthetic")["message_count"], 1)
        P.update_entity_profile_incremental(conn, "SYNTHETIC", "synthetic code append", "OTHER")
        self.assertEqual(P.get_entity_profile(conn, "Synthetic")["message_count"], 2)
        source, previous = P._profile_read_state(conn); self.assertEqual(source, previous)

    def test_multiple_appends_defer_without_partial_generation(self) -> None:
        conn = self.db(); before = self.rows(conn); self.append(conn); self.append(conn)
        with self.assertRaises(P.ProfileRebuildRequired):
            P.update_entity_profile_incremental(conn, "Synthetic", "synthetic code append", "Other")
        self.assertEqual(self.rows(conn), before)

    def test_mismatched_content_sender_recipient_and_directive_refuse(self) -> None:
        for kwargs in ({"content":"different"},{"sender":"Different"},{"recipient":"Different"},{"directive":1}):
            with self.subTest(kwargs=kwargs):
                conn = self.db(); before = self.rows(conn); self.append(conn, **kwargs)
                with self.assertRaises(P.ProfileRebuildRequired):
                    P.update_entity_profile_incremental(conn, "Synthetic", "synthetic code append", "Other")
                self.assertEqual(self.rows(conn), before)

    def test_source_free_direct_call_cannot_extend_certified_generation(self) -> None:
        conn = self.db(); before = self.rows(conn)
        with self.assertRaises(P.ProfileRebuildRequired):
            P.update_entity_profile_incremental(conn, "Synthetic", "synthetic fabricated", "Other")
        self.assertEqual(self.rows(conn), before)

    def test_direct_output_mutation_and_missing_token_preserve_enrollment(self) -> None:
        for sql in ("UPDATE entity_profiles SET message_count=999", "DELETE FROM metadata WHERE key='entity_profile_source_v1'",
                    "UPDATE metadata SET value='{}' WHERE key='entity_profile_source_v1'"):
            with self.subTest(sql=sql):
                conn = self.db(); conn.execute(sql); conn.commit(); before = self.rows(conn)
                self.assertIsNone(P.get_entity_profile(conn, "Synthetic"))
                with self.assertRaises(P.ProfileRebuildRequired):
                    P.update_entity_profile_incremental(conn, "Synthetic", "synthetic append", "Other")
                self.assertEqual(self.rows(conn), before)

    def test_epoch_reset_and_tracking_loss_do_not_revive_old_rows(self) -> None:
        for sql in ("UPDATE maintenance_source_state SET epoch='synthetic reset'", "DROP TABLE maintenance_source_state"):
            with self.subTest(sql=sql):
                conn = self.db(); conn.execute(sql); conn.commit(); before = self.rows(conn)
                self.assertIsNone(P.get_entity_profile(conn, "Synthetic"))
                self.assertEqual(self.rows(conn), before)

    def test_borrowed_corrective_repair_and_rollback_restore_output_token_source(self) -> None:
        conn = self.db(); before = self.rows(conn); token = conn.execute("SELECT value FROM metadata WHERE key=?", (P._PROFILE_TOKEN_KEY,)).fetchone()
        conn.execute("UPDATE messages SET content='synthetic corrected code'")
        self.assertIsNone(P.get_entity_profile(conn, "Synthetic")); self.assertTrue(conn.in_transaction)
        P.build_entity_profiles(conn)
        self.assertIsNotNone(P.get_entity_profile(conn, "Synthetic")); self.assertTrue(conn.in_transaction)
        conn.rollback(); self.assertEqual(self.rows(conn), before)
        self.assertEqual(conn.execute("SELECT value FROM metadata WHERE key=?", (P._PROFILE_TOKEN_KEY,)).fetchone(), token)

    def test_borrowed_append_rollback_restores_certificate_and_rows(self) -> None:
        conn = self.db(); before = self.rows(conn)
        conn.execute("INSERT INTO messages(content,sender,recipient) VALUES('synthetic append','Synthetic','Other')")
        P.update_entity_profile_incremental(conn, "Synthetic", "synthetic append", "Other")
        self.assertEqual(P.get_entity_profile(conn, "Synthetic")["message_count"], 2); self.assertTrue(conn.in_transaction)
        conn.rollback(); self.assertEqual(self.rows(conn), before)
        source, previous = P._profile_read_state(conn); self.assertEqual(source, previous)

    def test_known_nonappend_history_without_certificate_is_unavailable(self) -> None:
        conn = self.db(certified=False)
        conn.execute("INSERT INTO entity_profiles(entity,message_count) VALUES('synthetic',99)")
        conn.execute("UPDATE messages SET content='synthetic correction'"); conn.commit()
        self.assertIsNone(P.get_entity_profile(conn, "Synthetic"))
        with self.assertRaises(P.ProfileRebuildRequired):
            P.update_entity_profile_incremental(conn, "Synthetic", "synthetic partial")
        P.build_entity_profiles(conn); self.assertEqual(P.get_entity_profile(conn, "Synthetic")["message_count"], 1)

    def test_legacy_bulk_and_source_free_helpers_remain_compatible(self) -> None:
        conn = LOAD["_db"](":memory:"); self.addCleanup(conn.close); conn.commit()
        P.update_entity_profile_incremental(conn, "Synthetic", "synthetic first", "Other")
        P.update_entity_profile_incremental(conn, "SYNTHETIC", "synthetic second", "OTHER")
        self.assertEqual(P.get_entity_profile(conn, "Synthetic")["message_count"], 2)
        self.assertEqual(P.build_entity_profiles(conn), {})
        self.assertIsNone(P.get_entity_profile(conn, "Synthetic"))
        canonical = self.db(certified=False)
        P.update_entity_profile_incremental(canonical, "Direct", "synthetic source-free")
        self.assertEqual(P.get_entity_profile(canonical, "Direct")["message_count"], 1)

    def test_read_is_bounded_and_never_reads_messages(self) -> None:
        conn = self.db(); expected = P.get_entity_profile(conn, "Synthetic")
        def authorize(action: int, table: str | None, *_args: object) -> int:
            if action == sqlite3.SQLITE_READ and table == 'messages':return sqlite3.SQLITE_DENY
            if action in (sqlite3.SQLITE_INSERT,sqlite3.SQLITE_UPDATE,sqlite3.SQLITE_DELETE):return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        conn.set_authorizer(authorize)
        try:self.assertEqual(P.get_entity_profile(conn, "Synthetic"), expected)
        finally:conn.set_authorizer(None)
        self.assertFalse(conn.in_transaction)

    def test_missing_tracking_or_output_trigger_is_unavailable(self) -> None:
        for name in (*P._PROFILE_OUTPUT_TRIGGERS, *P._PROFILE_SOURCE_TRIGGER_SHA256):
            with self.subTest(name=name):
                conn = self.db(); conn.execute(f"DROP TRIGGER {name}"); conn.commit()
                self.assertIsNone(P.get_entity_profile(conn, "Synthetic"))
                self.assertTrue(self.rows(conn))

    def test_duplicate_keys_in_source_token_cannot_certify_rows(self) -> None:
        conn = self.db(); before = self.rows(conn)
        value = conn.execute("SELECT value FROM metadata WHERE key=?", (P._PROFILE_TOKEN_KEY,)).fetchone()[0]
        duplicate = '{"schema_version":9,' + value[1:]
        conn.execute("UPDATE metadata SET value=? WHERE key=?", (duplicate, P._PROFILE_TOKEN_KEY)); conn.commit()
        self.assertIsNone(P.get_entity_profile(conn, "Synthetic"))
        with self.assertRaises(P.ProfileRebuildRequired):
            P.update_entity_profile_incremental(conn, "Synthetic", "synthetic append", "Other")
        self.assertEqual(self.rows(conn), before)

    def test_invalid_utf8_source_epoch_or_token_is_controlled_unavailable(self) -> None:
        for sql in ("UPDATE maintenance_source_state SET epoch=CAST(X'eda080' AS TEXT)",
                    "UPDATE metadata SET value=CAST(X'eda080' AS TEXT) WHERE key='entity_profile_source_v1'"):
            with self.subTest(sql=sql):
                conn = self.db(); before = self.rows(conn)
                conn.text_factory = lambda value: value.decode('utf-8', 'surrogatepass')
                conn.execute(sql); conn.commit()
                self.assertIsNone(P.get_entity_profile(conn, "Synthetic"))
                with self.assertRaises(P.ProfileRebuildRequired):
                    P.update_entity_profile_incremental(conn, "Synthetic", "synthetic append", "Other")
                self.assertEqual(self.rows(conn), before)


if __name__ == "__main__":
    unittest.main()
