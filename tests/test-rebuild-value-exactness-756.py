"""Source-value fences using real SQLite and model-free rebuild control flow."""

import math
from pathlib import Path
import runpy
import sqlite3
import struct
import unittest
import uuid


_LOAD_MODULES = runpy.run_path(
    str(Path(__file__).with_name("test-rebuild-source-streaming-756.py")),
)["load_modules"]


class TestRebuildValueExactness(unittest.TestCase):
    def setUp(self) -> None:
        modules = _LOAD_MODULES()
        self.source = modules["rebuild_source"]
        self.storage = modules["storage"]
        self.vector = modules["vector_search"]
        self.model = object()
        self.vector.get_model = lambda: self.model
        self.vector._get_batch_size = lambda: 1
        self.calls = []
        self.vector._encode_with_mps_fallback = self.encode

    def encode(self, model: object, texts: list, **kwargs: object) -> list[list[float]]:
        self.assertIs(model, self.model)
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    def database(self, fields: str = "content", *, shared: bool = False) -> tuple[sqlite3.Connection, sqlite3.Connection | None]:
        uri = "file:synthetic-value-fence-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
        conn = sqlite3.connect(uri if shared else ":memory:", uri=shared, cached_statements=0)
        self.addCleanup(conn.close)
        conn.execute(f"CREATE TABLE messages(id INTEGER PRIMARY KEY, {fields})")
        conn.execute("CREATE TABLE vec_messages(embedding BLOB)")
        conn.execute("CREATE TABLE vec_messages_sep(embedding BLOB)")
        other = sqlite3.connect(uri, uri=True, cached_statements=0) if shared else None
        if other is not None:
            self.addCleanup(other.close)
        return conn, other

    def seed(self, conn: sqlite3.Connection, value: object) -> None:
        conn.execute("INSERT INTO messages(id,content) VALUES(1,?)", (value,))
        conn.commit()
        self.source.ensure_rebuild_tracking(conn)

    def manifest(self, conn: sqlite3.Connection, table: str = "vec_messages_sep") -> object:
        return self.source.load_manifest(conn, self.source.manifest_key((table,)))

    def install_previous_update_trigger(self, conn: sqlite3.Connection) -> None:
        covered, _, conservative = self.source._source_schema(conn)
        self.assertFalse(conservative)
        changed = " OR ".join(
            f"typeof(old.{name}) IS NOT typeof(new.{name}) "
            f"OR CAST(old.{name} AS BLOB) IS NOT CAST(new.{name} AS BLOB)"
            for name in covered
        )
        conn.execute("DROP TRIGGER truememory_rebuild_messages_v1_au")
        conn.execute(f"""CREATE TRIGGER truememory_rebuild_messages_v1_au
            AFTER UPDATE ON messages WHEN {changed} BEGIN
            UPDATE truememory_rebuild_source_v1 SET nonappend_revision = revision + 1,
                max_seen_message_id = MAX(max_seen_message_id, new.id),
                revision = revision + 1, correction_count = correction_count + 1
            WHERE singleton = 1 AND tracking_ready = 1;
        END""")
        conn.commit()

    def test_storage_value_comparison_matrix(self) -> None:
        values = [None, 0, 1, -1, 9007199254740992, 9007199254740993,
                  0.0, -0.0, 1.0, 1.0000000000000002, 1.0000000000000004,
                  5e-324, float("inf"), float("-inf"), "0", "synthetic", "Synthetic",
                  "Synthetic ", "a\x00b", "a\x00c", b"synthetic", b"a\x00b", b"a\x00c"]
        conn, _ = self.database()
        self.source.ensure_rebuild_tracking(conn)
        for before_value in values:
            for after_value in values:
                with self.subTest(before=repr(before_value), after=repr(after_value)):
                    conn.execute("DELETE FROM messages")
                    conn.execute("INSERT INTO messages VALUES(1,?)", (before_value,))
                    conn.commit()
                    before = self.source.capture_source(conn)
                    stored_before = conn.execute("SELECT content FROM messages").fetchone()[0]
                    conn.execute("UPDATE messages SET content=?", (after_value,))
                    conn.commit()
                    stored_after = conn.execute("SELECT content FROM messages").fetchone()[0]
                    changed = repr(stored_before) != repr(stored_after)
                    zero = isinstance(stored_before, float) and stored_before == 0
                    if changed or zero:
                        with self.assertRaises(self.source.RebuildSourceChanged):
                            before.check(conn)
                    else:
                        before.check(conn)
                        self.assertEqual(before, self.source.capture_source(conn))

    def test_text_affinity_noop_and_collation_controls(self) -> None:
        cases = [("TEXT", 1.0000000000000002, 1.0000000000000004, False),
                 ("REAL", 1.25, 1.25, False),
                 ("TEXT COLLATE NOCASE", "Synthetic", "synthetic", True),
                 ("TEXT COLLATE RTRIM", "Synthetic", "Synthetic ", True)]
        for affinity, old, new, expected_change in cases:
            with self.subTest(affinity=affinity):
                conn, _ = self.database("content " + affinity)
                self.seed(conn, old)
                before = self.source.capture_source(conn)
                conn.execute("UPDATE messages SET content=?", (new,))
                conn.commit()
                if expected_change:
                    with self.assertRaises(self.source.RebuildSourceChanged):
                        before.check(conn)
                else:
                    before.check(conn)
                    self.assertEqual(before, self.source.capture_source(conn))

    def test_real_sender_precision_rejects_actual_separation_publication(self) -> None:
        for affinity in ("REAL", ""):
            with self.subTest(sender_affinity=affinity):
                conn, other = self.database(
                    f"content TEXT, sender {affinity}, recipient TEXT, timestamp TEXT", shared=True,
                )
                conn.execute("INSERT INTO messages VALUES(1,?,?,?,?)",
                             ("synthetic content", 1.0000000000000002, "recipient", "2026-01-02"))
                conn.commit()
                encoded = []

                def encode(model: object, texts: list, **kwargs: object) -> list[list[float]]:
                    encoded.extend(texts)
                    self.assertTrue(all(isinstance(text, str) for text in texts))
                    self.assertFalse(conn.in_transaction)
                    other.execute("UPDATE messages SET sender=?", (1.0000000000000004,))
                    other.commit()
                    return [[1.0, 0.0]]

                self.vector._encode_with_mps_fallback = encode
                with self.assertRaises(self.source.RebuildSourceChanged):
                    self.vector.build_separation_vectors(conn, txn_batch=1)
                row = conn.execute("SELECT sender,recipient,timestamp,content FROM messages").fetchone()
                self.assertNotEqual(encoded[0], self.vector._build_sep_text(*row))
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM vec_messages_sep").fetchone()[0], 0)
                self.assertEqual((self.manifest(conn).complete, self.manifest(conn).consumed), (False, 0))

    def test_signed_zero_directions_reject_actual_separation_publication(self) -> None:
        for field in ("content", "sender"):
            for old, new in ((0.0, -0.0), (-0.0, 0.0)):
                with self.subTest(field=field, old=repr(old)):
                    fields = "content, sender TEXT" if field == "content" else "content TEXT, sender"
                    conn, other = self.database(fields + ", recipient TEXT, timestamp TEXT", shared=True)
                    values = {"content": "synthetic content", "sender": "sender"}
                    values[field] = old
                    conn.execute("INSERT INTO messages VALUES(1,?,?,?,?)",
                                 (values["content"], values["sender"], "recipient", "2026-01-02"))
                    conn.commit()
                    encoded = []

                    def encode(model: object, texts: list, **kwargs: object) -> list[list[float]]:
                        encoded.extend(texts)
                        self.assertTrue(all(isinstance(text, str) for text in texts))
                        other.execute(f"UPDATE messages SET {field}=?", (new,))
                        other.commit()
                        return [[1.0, 0.0]]

                    self.vector._encode_with_mps_fallback = encode
                    with self.assertRaises(self.source.RebuildSourceChanged):
                        self.vector.build_separation_vectors(conn, txn_batch=1)
                    stored = conn.execute(f"SELECT {field} FROM messages").fetchone()[0]
                    self.assertEqual(math.copysign(1, stored), math.copysign(1, new))
                    row = conn.execute("SELECT sender,recipient,timestamp,content FROM messages").fetchone()
                    if field == "content":
                        self.assertNotEqual(encoded[0], self.vector._build_sep_text(*row))
                    else:
                        # Metadata zero becomes '?'; the source fence is more
                        # conservative than this particular text transform.
                        self.assertEqual(encoded[0], self.vector._build_sep_text(*row))
                    self.assertFalse(self.manifest(conn).complete)
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM vec_messages_sep").fetchone()[0], 0)

    def test_zero_source_write_invalidates_but_derived_only_write_does_not(self) -> None:
        conn, _ = self.database("content, episode_id INTEGER")
        self.seed(conn, -0.0)
        before = self.source.capture_source(conn)
        conn.execute("UPDATE messages SET episode_id=7")
        conn.commit()
        before.check(conn)
        self.assertEqual(before, self.source.capture_source(conn))
        conn.execute("UPDATE messages SET content=content")
        conn.commit()
        with self.assertRaises(self.source.RebuildSourceChanged):
            before.check(conn)

    def test_unshadowed_rowid_aliases_still_invalidate_source(self) -> None:
        for name in ("rowid", "ROWID", "_rowid_", "OID"):
            with self.subTest(alias=name):
                conn, _ = self.database()
                self.seed(conn, "synthetic")
                before = self.source.capture_source(conn)
                conn.execute(f"UPDATE messages SET {name}=2")
                conn.commit()
                self.assertEqual(conn.execute("SELECT id FROM messages").fetchone()[0], 2)
                with self.assertRaises(self.source.RebuildSourceChanged):
                    before.check(conn)

    def test_shadowed_rowid_field_is_derived_and_other_alias_still_tracks_id(self) -> None:
        conn, _ = self.database("content, ROWID TEXT")
        self.seed(conn, 0.0)
        before = self.source.capture_source(conn)
        conn.execute("UPDATE messages SET rowid='derived'")
        conn.commit()
        before.check(conn)
        conn.execute("UPDATE messages SET _rowid_=2")
        conn.commit()
        with self.assertRaises(self.source.RebuildSourceChanged):
            before.check(conn)

    def test_generated_source_tracks_uncovered_dependency_updates(self) -> None:
        for kind in ("VIRTUAL", "STORED"):
            with self.subTest(kind=kind):
                conn, _ = self.database(f"raw_value, content GENERATED ALWAYS AS (raw_value) {kind}")
                conn.execute("INSERT INTO messages(id,raw_value) VALUES(1,?)", (1.0000000000000002,))
                conn.commit()
                self.source.ensure_rebuild_tracking(conn)
                before = self.source.capture_source(conn)
                conn.execute("UPDATE messages SET raw_value=?", (1.0000000000000004,))
                conn.commit()
                with self.assertRaises(self.source.RebuildSourceChanged):
                    before.check(conn)

    def test_unique_bridge_keeps_all_update_invalidation(self) -> None:
        conn, _ = self.database("content, derived_key TEXT UNIQUE")
        self.seed(conn, "synthetic")
        before = self.source.capture_source(conn)
        self.assertEqual(before.tracker, "rebuild-bridge-conservative-v1")
        conn.execute("UPDATE messages SET derived_key='derived'")
        conn.commit()
        with self.assertRaises(self.source.RebuildSourceChanged):
            before.check(conn)

    def test_precision_change_rollback_restores_token_and_preserves_caller_work(self) -> None:
        conn, _ = self.database()
        self.seed(conn, 1.0000000000000002)
        conn.execute("CREATE TABLE synthetic_caller(value TEXT)")
        conn.execute("INSERT INTO synthetic_caller VALUES('pending')")
        before = self.source.capture_source(conn)
        with self.assertRaisesRegex(RuntimeError, "synthetic rollback"):
            with self.source.rebuild_transaction(conn, write=True):
                conn.execute("UPDATE messages SET content=?", (1.0000000000000004,))
                with self.assertRaises(self.source.RebuildSourceChanged):
                    before.check(conn)
                raise RuntimeError("synthetic rollback")
        self.assertTrue(conn.in_transaction)
        self.assertEqual(conn.execute("SELECT value FROM synthetic_caller").fetchone()[0], "pending")
        self.assertEqual(conn.execute("SELECT content FROM messages").fetchone()[0], 1.0000000000000002)
        before.check(conn)
        self.assertEqual(before, self.source.capture_source(conn))

    def test_failed_second_publication_keeps_committed_prefix_and_cursor(self) -> None:
        conn, other = self.database("content TEXT, sender REAL, recipient TEXT, timestamp TEXT", shared=True)
        conn.executemany("INSERT INTO messages VALUES(?,?,?,?,?)", [
            (1, "synthetic first", 1.25, "recipient", "2026-01-02"),
            (2, "synthetic second", 1.0000000000000002, "recipient", "2026-01-02"),
        ])
        conn.commit()

        def encode(model: object, texts: list, **kwargs: object) -> list[list[float]]:
            result = self.encode(model, texts, **kwargs)
            if len(self.calls) == 2:
                other.execute("UPDATE messages SET sender=? WHERE id=2", (1.0000000000000004,))
                other.commit()
            return result

        self.vector._encode_with_mps_fallback = encode
        with self.assertRaises(self.source.RebuildSourceChanged):
            self.vector.build_separation_vectors(conn, txn_batch=1)
        conn.commit()
        self.assertEqual(conn.execute("SELECT rowid,embedding FROM vec_messages_sep").fetchall(),
                         [(1, struct.pack("2f", 1.0, 0.0))])
        manifest = self.manifest(conn)
        self.assertEqual((manifest.complete, manifest.cursor, manifest.consumed), (False, 1, 1))

    def test_previous_trigger_repair_changes_epoch_and_invalidates_manifest(self) -> None:
        conn, _ = self.database("content TEXT")
        self.seed(conn, "synthetic")
        self.vector.build_vectors(conn, txn_batch=1)
        before = self.source.capture_source(conn)
        manifest = self.manifest(conn, "vec_messages")
        self.install_previous_update_trigger(conn)
        with self.assertRaises(self.source.RebuildSourceChanged):
            before.check(conn)
        self.source.ensure_rebuild_tracking(conn)
        after = self.source.capture_source(conn)
        self.assertNotEqual(before.revision.epoch, after.revision.epoch)
        with self.assertRaises(self.source.RebuildSourceChanged):
            manifest.check(conn, self.source.manifest_key(("vec_messages",)))
        self.calls.clear()
        self.assertEqual(self.vector.build_vectors(conn, txn_batch=1), 1)
        self.assertEqual(self.calls, [["synthetic"]])
        self.assertEqual(self.manifest(conn, "vec_messages").source.revision.epoch, after.revision.epoch)

    def test_denied_trigger_repair_preserves_target_and_caller_transaction(self) -> None:
        for caller_owned in (False, True):
            with self.subTest(caller_owned=caller_owned):
                conn, _ = self.database("content TEXT")
                self.seed(conn, "synthetic")
                self.install_previous_update_trigger(conn)
                conn.execute("INSERT INTO vec_messages(rowid,embedding) VALUES(99,?)", (b"synthetic-old",))
                conn.execute("CREATE TABLE synthetic_caller(value TEXT)")
                conn.commit()
                before_row = self.source._bridge_row(conn)
                before_triggers = self.source._installed_triggers(conn, self.source._BRIDGE_TRIGGERS)
                if caller_owned:
                    conn.execute("INSERT INTO synthetic_caller VALUES('pending')")
                denied = []

                def authorize(action: int, first: str, second: str, *_args: object) -> int:
                    if action == sqlite3.SQLITE_DROP_TRIGGER and first in self.source._BRIDGE_TRIGGERS:
                        denied.append(first)
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK

                conn.set_authorizer(authorize)
                self.calls.clear()
                try:
                    with self.assertRaises(sqlite3.DatabaseError):
                        self.vector.build_vectors(conn, txn_batch=1)
                finally:
                    conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                self.assertTrue(denied)
                self.assertEqual(self.calls, [])
                self.assertEqual(conn.in_transaction, caller_owned)
                self.assertEqual(self.source._bridge_row(conn), before_row)
                self.assertEqual(self.source._installed_triggers(conn, self.source._BRIDGE_TRIGGERS), before_triggers)
                self.assertEqual(conn.execute("SELECT rowid,embedding FROM vec_messages").fetchall(), [(99, b"synthetic-old")])
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM synthetic_caller").fetchone()[0], int(caller_owned))

    def test_trusted_default_schema_stays_canonical(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.executescript(self.storage._SCHEMA_SQL)
        self.storage._initialize_maintenance_tracking(conn)
        self.source.ensure_rebuild_tracking(conn)
        self.assertEqual(self.source.capture_source(conn).tracker, "canonical-v1")
        self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.source._BRIDGE_TABLE,)).fetchone())


if __name__ == "__main__":
    unittest.main()
