"""Synthetic in-memory source/pair lifecycle; no package or model initialization."""

import builtins
import json
import os
import sqlite3
import struct
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("vec_messages_synthetic", "vec_messages_sep_synthetic")


def load_modules():
    modules = {}

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"numpy", "torch", "sentence_transformers", "sqlite_vec"}:
            raise AssertionError("Native/model imports are excluded from the adapter")
        if name.startswith("truememory."):
            short = name.removeprefix("truememory.")
            if short in modules:
                return modules[short]
            raise AssertionError("Unexpected application import: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    for name in ("storage", "_platform", "maintenance", "rebuild_source", "tier_switch.source"):
        module = types.ModuleType("synthetic_tier_source_" + name.replace(".", "_"))
        sys.modules[module.__name__] = module
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
        path = ROOT / "truememory" / (name.replace(".", "/") + ".py")
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
        modules[name] = module
    return modules


class FaultConnection(sqlite3.Connection):
    fail_sql = None
    fail_commit = False

    def execute(self, sql, parameters=()):
        if self.fail_sql is not None and self.fail_sql in sql:
            self.fail_sql = None
            raise sqlite3.OperationalError("synthetic pair write failure")
        return super().execute(sql, parameters)

    def commit(self):
        if self.fail_commit:
            self.fail_commit = False
            raise sqlite3.OperationalError("synthetic commit failure")
        return super().commit()


class TestTierSource(unittest.TestCase):
    def setUp(self):
        self.modules = load_modules()
        self.api = self.modules["tier_switch.source"]
        self.source = self.modules["rebuild_source"]
        self.conn = sqlite3.connect(":memory:", factory=FaultConnection)
        self.addCleanup(self.conn.close)
        self.create_schema(self.conn)

    def create_schema(self, conn):
        conn.executescript("""
            CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT, sender TEXT, recipient TEXT, timestamp TEXT);
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT);
            CREATE TABLE vec_messages_synthetic(embedding BLOB);
            CREATE TABLE vec_messages_sep_synthetic(embedding BLOB);
        """)
        self.source.ensure_rebuild_tracking(conn)

    def seed(self, ids=(-3, 0, 2, 8)):
        self.conn.executemany("INSERT INTO messages VALUES (?,?,?,?,?)", [
            (mid, f"synthetic-{mid}", "sender", "recipient", "2026-01-02") for mid in ids
        ])
        self.conn.commit()

    def plan(self, **kwargs):
        return self.api.plan_tier_source(self.conn, model_id="synthetic-model", dimension=2, targets=TARGETS, **kwargs)

    def start(self):
        return self.api.initialize_tier_source(self.conn, self.plan())

    def vectors(self, count):
        return [struct.pack("2f", 1.0, 0.0)] * count, [struct.pack("2f", 0.0, 1.0)] * count

    def publish(self, plan, page, count=None):
        left, right = self.vectors(len(page.rows) if count is None else count)
        return self.api.publish_tier_prefix(self.conn, plan, page, completion=left, separation=right)

    def build(self, plan=None):
        plan = self.start() if plan is None else plan
        while plan.manifest.consumed < plan.manifest.total:
            page = self.api.read_tier_page(self.conn, plan, size=2)
            plan, pending = self.publish(plan, page)
            self.assertEqual(pending.rows, ())
        return self.api.finish_tier_source(self.conn, plan)

    def ids(self, table=TARGETS[0]):
        return [row[0] for row in self.conn.execute(f'SELECT rowid FROM "{table}" ORDER BY rowid')]

    def test_plan_is_read_only_and_initialization_never_clears(self):
        self.seed()
        before = self.conn.total_changes
        plan = self.plan()
        self.assertEqual(plan.action, "full")
        self.assertEqual(self.conn.total_changes, before)
        self.assertIsNone(self.source.load_manifest(self.conn, self.source.manifest_key(TARGETS)))
        plan = self.api.initialize_tier_source(self.conn, plan)
        self.assertEqual(plan.action, "resume")
        self.assertEqual(self.ids(), [])
        self.assertFalse(self.conn.in_transaction)

    def test_negative_zero_sparse_ids_and_exact_metadata_survive(self):
        self.seed()
        plan = self.start()
        page = self.api.read_tier_page(self.conn, plan, size=2)
        self.assertEqual(page.messages(), [
            {"id": mid, "content": f"synthetic-{mid}", "sender": "sender", "recipient": "recipient", "timestamp": "2026-01-02"}
            for mid in (-3, 0)
        ])
        self.assertFalse(self.conn.in_transaction)
        result = self.build(plan)
        self.assertTrue(result.captured_range_complete)
        self.assertTrue(result.current_source_complete)
        self.assertEqual(self.ids(), [-3, 0, 2, 8])
        self.assertEqual(self.ids(TARGETS[1]), [-3, 0, 2, 8])
        self.assertEqual((result.plan.manifest.cursor, result.plan.manifest.consumed, result.plan.manifest.outputs), (8, 4, 4))
        self.assertEqual(self.plan().action, "complete")

    def test_prefix_commit_retains_exact_pending_suffix_for_reduced_batches(self):
        self.seed()
        plan = self.start()
        page = self.api.read_tier_page(self.conn, plan, size=4)
        updated, pending = self.publish(plan, page, 1)
        self.assertEqual(plan.manifest.consumed, 0)
        self.assertEqual(updated.manifest.cursor, -3)
        self.assertEqual([row[0] for row in pending.rows], [0, 2, 8])
        self.assertEqual(page.rows[1:], pending.rows)
        updated, pending = self.publish(updated, pending, 2)
        self.assertEqual(updated.manifest.cursor, 2)
        self.assertEqual([row[0] for row in pending.rows], [8])
        updated, pending = self.publish(updated, pending, 1)
        self.assertEqual(pending.rows, ())
        self.assertEqual(self.ids(), [-3, 0, 2, 8])
        self.assertTrue(self.api.finish_tier_source(self.conn, updated).current_source_complete)

    def test_resume_recomputes_digest_once_and_not_on_each_page(self):
        self.seed(range(1, 140))
        plan = self.start()
        plan, _ = self.publish(plan, self.api.read_tier_page(self.conn, plan, size=70))
        calls = []
        original = self.api._audit_pair

        def audit(conn, candidate):
            calls.append(candidate.manifest.consumed)
            return original(conn, candidate)

        self.api._audit_pair = audit
        resumed = self.plan()
        self.assertEqual(calls, [70])
        self.assertEqual(resumed.manifest.generation, plan.manifest.generation)
        result = self.build(resumed)
        self.assertEqual(calls, [70])
        self.assertEqual(result.plan.manifest.consumed, 139)

    def test_restart_on_new_in_memory_connection_revalidates_receipt(self):
        uri = "file:tier_source_reopen_synthetic?mode=memory&cache=shared"
        self.conn = sqlite3.connect(uri, uri=True, factory=FaultConnection)
        self.addCleanup(self.conn.close)
        self.create_schema(self.conn)
        self.seed()
        plan = self.start()
        plan, _ = self.publish(plan, self.api.read_tier_page(self.conn, plan, size=2))
        other = sqlite3.connect(uri, uri=True, factory=FaultConnection)
        self.addCleanup(other.close)
        with self.assertRaisesRegex(self.api.TierSourceUntrusted, "Connection changed"):
            self.api.read_tier_page(other, plan)
        resumed = self.api.plan_tier_source(other, model_id="synthetic-model", dimension=2, targets=TARGETS)
        self.assertEqual(resumed.manifest, plan.manifest)
        self.assertEqual([row[0] for row in self.api.read_tier_page(other, resumed).rows], [2, 8])

    def test_empty_source_finishes_without_model_or_vector_work(self):
        result = self.build()
        self.assertTrue(result.current_source_complete)
        self.assertIsNone(result.plan.manifest.cursor)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.plan().action, "complete")

    def test_high_tail_requires_replan_and_fixed_range_is_not_current_coverage(self):
        self.seed((1, 2))
        plan = self.start()
        self.seed((3, 4))
        with self.assertRaisesRegex(self.api.TierSourceUntrusted, "explicitly replan"):
            self.api.read_tier_page(self.conn, plan)
        resumed = self.plan()
        self.assertEqual(resumed.manifest.source.high_id, 2)
        result = self.build(resumed)
        self.assertTrue(result.captured_range_complete)
        self.assertFalse(result.current_source_complete)
        delta = self.plan()
        self.assertEqual((delta.action, delta.prefix_count, delta.prefix_cursor, delta.manifest.total), ("delta", 2, 2, 2))
        delta = self.api.initialize_tier_source(self.conn, delta)
        self.assertEqual([row[0] for row in self.api.read_tier_page(self.conn, delta).rows], [3, 4])
        result = self.build(delta)
        self.assertTrue(result.current_source_complete)
        self.assertEqual(self.ids(), [1, 2, 3, 4])
        self.assertEqual(self.plan().action, "complete")

    def test_empty_complete_pair_can_begin_first_delta(self):
        self.build()
        self.seed((1, 2))
        delta = self.plan()
        self.assertEqual((delta.action, delta.prefix_count, delta.prefix_cursor), ("delta", 0, None))
        self.assertTrue(self.build(self.api.initialize_tier_source(self.conn, delta)).current_source_complete)
        self.assertEqual(self.plan().action, "complete")

    def test_correction_delete_and_low_id_insert_refuse_resume(self):
        for sql in ("UPDATE messages SET content='corrected' WHERE id=2", "DELETE FROM messages WHERE id=2",
                    "INSERT INTO messages(id,content) VALUES (1,'late-low-id')"):
            with self.subTest(sql=sql):
                conn = sqlite3.connect(":memory:")
                try:
                    self.create_schema(conn)
                    conn.execute("INSERT INTO messages(id,content) VALUES (2,'synthetic')")
                    conn.commit()
                    plan = self.api.plan_tier_source(conn, model_id="synthetic-model", dimension=2, targets=TARGETS)
                    self.api.initialize_tier_source(conn, plan)
                    conn.execute(sql)
                    conn.commit()
                    with self.assertRaises(self.source.RebuildSourceChanged):
                        self.api.plan_tier_source(conn, model_id="synthetic-model", dimension=2, targets=TARGETS)
                finally:
                    conn.close()

    def test_existing_unreceipted_rows_are_never_certified_or_cleared(self):
        self.seed((1,))
        blob = self.vectors(1)[0][0]
        for table in TARGETS:
            self.conn.execute(f'INSERT INTO "{table}"(rowid,embedding) VALUES (1,?)', (blob,))
        self.conn.commit()
        for force in (False, True):
            with self.assertRaisesRegex(self.api.TierSourceUntrusted, "empty inactive pair"):
                self.plan(force=force)
        self.assertEqual(self.ids(), [1])
        self.assertEqual(self.ids(TARGETS[1]), [1])

    def test_same_count_tampered_vectors_and_wrong_width_refuse_restart(self):
        self.seed((1,))
        self.build()
        for blob in (struct.pack("2f", 0.5, 0.5), b"bad"):
            with self.subTest(blob=blob):
                self.conn.execute(f'UPDATE "{TARGETS[0]}" SET embedding=? WHERE rowid=1', (blob,))
                self.conn.commit()
                with self.assertRaises(self.api.TierSourceUntrusted):
                    self.plan()
        self.assertEqual(self.ids(), [1])

    def test_pair_row_deletion_or_addition_refuses_restart(self):
        self.seed((1, 2))
        result = self.build()
        self.conn.execute(f'DELETE FROM "{TARGETS[1]}" WHERE rowid=1')
        self.conn.commit()
        with self.assertRaises(self.api.TierSourceUntrusted):
            self.plan()
        self.assertTrue(result.plan.manifest.complete)

    def test_pair_schema_drop_recreate_is_not_matching_content_proof(self):
        self.seed((1,))
        self.build()
        self.conn.executescript(f'DROP TABLE "{TARGETS[0]}"; CREATE TABLE "{TARGETS[0]}"(embedding BLOB);')
        self.conn.execute(f'INSERT INTO "{TARGETS[0]}"(rowid,embedding) VALUES (1,?)', (self.vectors(1)[0][0],))
        self.conn.commit()
        with self.assertRaisesRegex(self.api.TierSourceUntrusted, "schema"):
            self.plan()

    def test_temp_case_variant_shadows_and_temp_triggers_refuse(self):
        for statement in (
            'CREATE TEMP TABLE MeSsAgEs(id INTEGER)', 'CREATE TEMP TABLE MeTaDaTa(key TEXT)',
            'CREATE TEMP TABLE VEC_MESSAGES_SYNTHETIC(embedding BLOB)',
            'CREATE TEMP TABLE maintenance_source_state(singleton INTEGER)',
            'CREATE TEMP TABLE truememory_rebuild_source_v1(singleton INTEGER)',
            'CREATE TEMP TRIGGER synthetic_shadow AFTER INSERT ON main.messages BEGIN SELECT 1; END',
        ):
            with self.subTest(statement=statement):
                conn = sqlite3.connect(":memory:")
                try:
                    self.create_schema(conn)
                    conn.execute(statement)
                    with self.assertRaisesRegex(self.api.TierSourceUntrusted, "TEMP"):
                        self.api.plan_tier_source(conn, model_id="synthetic-model", dimension=2, targets=TARGETS)
                finally:
                    conn.close()

    def test_shadow_added_after_planning_refuses_publication(self):
        self.seed((1,))
        plan = self.start()
        page = self.api.read_tier_page(self.conn, plan)
        self.conn.execute('CREATE TEMP TABLE VEC_MESSAGES_SEP_SYNTHETIC(embedding BLOB)')
        with self.assertRaisesRegex(self.api.TierSourceUntrusted, "TEMP"):
            self.publish(plan, page)
        self.assertEqual(self.ids(), [])

    def test_unrelated_write_invalidates_fence_without_changing_checkpoint(self):
        self.seed((1,))
        plan = self.start()
        page = self.api.read_tier_page(self.conn, plan)
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('unrelated','value')")
        self.conn.commit()
        with self.assertRaisesRegex(self.api.TierSourceUntrusted, "Connection changed"):
            self.publish(plan, page)
        resumed = self.plan()
        self.assertEqual(resumed.manifest, plan.manifest)
        self.assertEqual(self.publish(resumed, page)[0].manifest.consumed, 1)

    def test_external_commit_data_version_requires_explicit_revalidation(self):
        uri = "file:tier_source_synthetic?mode=memory&cache=shared"
        first = sqlite3.connect(uri, uri=True)
        second = sqlite3.connect(uri, uri=True)
        try:
            self.create_schema(first)
            plan = self.api.plan_tier_source(first, model_id="synthetic-model", dimension=2, targets=TARGETS)
            plan = self.api.initialize_tier_source(first, plan)
            second.execute("INSERT INTO metadata(key,value) VALUES ('external','synthetic')")
            second.commit()
            with self.assertRaisesRegex(self.api.TierSourceUntrusted, "Connection changed"):
                self.api.read_tier_page(first, plan)
            self.assertEqual(first.total_changes, plan.changes)
        finally:
            second.close()
            first.close()

    def test_second_pair_write_failure_rolls_back_both_rows_and_checkpoint(self):
        self.seed((1, 2))
        plan = self.start()
        page = self.api.read_tier_page(self.conn, plan)
        self.conn.fail_sql = f'INSERT INTO main."{TARGETS[1]}"'
        with self.assertRaisesRegex(sqlite3.OperationalError, "pair write"):
            self.publish(plan, page)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.ids(TARGETS[1]), [])
        self.assertFalse(self.conn.in_transaction)
        resumed = self.plan()
        self.assertEqual(resumed.manifest, plan.manifest)
        updated, pending = self.publish(resumed, page, 1)
        self.assertEqual(updated.manifest.cursor, 1)
        self.assertEqual([row[0] for row in pending.rows], [2])

    def test_commit_failure_rolls_back_pair_receipt_and_manifest(self):
        self.seed((1,))
        plan = self.start()
        page = self.api.read_tier_page(self.conn, plan)
        self.conn.fail_commit = True
        with self.assertRaisesRegex(sqlite3.OperationalError, "commit failure"):
            self.publish(plan, page)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.plan().manifest, plan.manifest)
        self.assertFalse(self.conn.in_transaction)

    def test_pending_inputs_prevent_finish_and_bad_prefixes_do_not_advance(self):
        self.seed()
        plan = self.start()
        page = self.api.read_tier_page(self.conn, plan, size=2)
        with self.assertRaises(self.api.TierSourceUntrusted):
            self.api.finish_tier_source(self.conn, plan)
        for left, right in (([], []), (self.vectors(1)[0], []), (self.vectors(3)[0], self.vectors(3)[1])):
            with self.assertRaises(self.api.TierSourceUntrusted):
                self.api.publish_tier_prefix(self.conn, plan, page, completion=left, separation=right)
        with self.assertRaises(ValueError):
            self.api.publish_tier_prefix(self.conn, plan, page, completion=[b"bad"], separation=[b"bad"])
        changed = replace(page, rows=((page.rows[0][0], "wrong-text", *page.rows[0][2:]), *page.rows[1:]))
        with self.assertRaisesRegex(self.api.TierSourceUntrusted, "Pending page"):
            self.publish(plan, changed, 1)
        self.assertEqual(self.plan().manifest, plan.manifest)
        self.assertEqual(self.ids(), [])

    def test_borrowed_transaction_is_never_committed_or_rolled_back(self):
        plan = self.start()
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('borrowed','pending')")
        with self.assertRaisesRegex(self.api.TierSourceUntrusted, "owned, clean"):
            self.api.read_tier_page(self.conn, plan)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='borrowed'").fetchone()[0], "pending")
        self.conn.rollback()

    def test_page_fields_require_exact_sqlite_type_and_signed_real_zero(self):
        self.conn = sqlite3.connect(":memory:", factory=FaultConnection)
        self.addCleanup(self.conn.close)
        self.conn.executescript("""
            CREATE TABLE messages(id INTEGER PRIMARY KEY, content, sender, recipient, timestamp);
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE vec_messages_synthetic(embedding BLOB);
            CREATE TABLE vec_messages_sep_synthetic(embedding BLOB);
        """)
        self.source.ensure_rebuild_tracking(self.conn)
        self.conn.executemany("INSERT INTO messages(id,content) VALUES (?,?)", [(1, 1), (2, -0.0)])
        self.conn.commit()
        plan = self.start()
        page = self.api.read_tier_page(self.conn, plan)
        integer_as_real = replace(page, rows=((1, 1.0, None, None, None), page.rows[1]))
        signed_zero = replace(page, rows=(page.rows[0], (2, 0.0, None, None, None)))
        for changed in (integer_as_real, signed_zero):
            with self.assertRaisesRegex(self.api.TierSourceUntrusted, "Pending page"):
                self.publish(plan, changed)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.publish(plan, page)[0].manifest.consumed, 2)

    def test_malformed_manifest_types_ranges_and_complete_flags_refuse(self):
        self.seed()
        plan = self.start()
        key = self.source.manifest_key(TARGETS)
        original = json.loads(plan.saved_manifest)
        mutations = (
            {"version": True}, {"dimension": True}, {"consumed": -1}, {"consumed": True},
            {"outputs": 1}, {"consumed": 5, "outputs": 5}, {"cursor": 0}, {"complete": True},
            {"complete": 1}, {"total": 5}, {"source": None}, {"schema_version": None},
            {"targets": list(reversed(TARGETS))}, {"generation": "not-a-generation"},
        )
        for update in mutations:
            with self.subTest(update=update):
                self.conn.execute("UPDATE metadata SET value=? WHERE key=?", (json.dumps(dict(original, **update)), key))
                self.conn.commit()
                with self.assertRaises(self.api.TierSourceUntrusted):
                    self.plan()

    def test_missing_receipt_is_not_registry_or_manifest_compatibility(self):
        self.start()
        self.conn.execute("DELETE FROM metadata WHERE key=?", (self.api._receipt_key(TARGETS),))
        self.conn.commit()
        with self.assertRaisesRegex(self.api.TierSourceUntrusted, "Both manifest and pair receipt"):
            self.plan()

    def test_oversized_and_duplicate_metadata_are_rejected(self):
        plan = self.start()
        key = self.source.manifest_key(TARGETS)
        for raw in ("x" * 65537, plan.saved_manifest[:-1] + ',"version":1}'):
            self.conn.execute("UPDATE metadata SET value=? WHERE key=?", (raw, key))
            self.conn.commit()
            with self.assertRaises(self.api.TierSourceUntrusted):
                self.plan()

    def test_wrong_model_dimension_and_target_identifiers_refuse(self):
        self.start()
        for model, dimension, targets in (
            ("other", 2, TARGETS), ("synthetic-model", 3, TARGETS),
            ("synthetic-model", 4097, TARGETS),
            ("synthetic-model", True, TARGETS), ("synthetic-model", 2, ("vec_messages_bad;--", TARGETS[1])),
        ):
            with self.assertRaises((ValueError, self.api.TierSourceUntrusted)):
                self.api.plan_tier_source(self.conn, model_id=model, dimension=dimension, targets=targets)

    def test_restart_digest_reader_uses_bounded_pages(self):
        self.seed(range(1, 170))
        self.build()
        batches = []
        original = self.api._bounded_rows

        def bounded(cursor):
            class Reader:
                def fetchmany(inner, size):
                    batches.append(size)
                    return cursor.fetchmany(size)

                def close(inner):
                    cursor.close()
            yield from original(Reader())

        self.api._bounded_rows = bounded
        self.assertEqual(self.plan().action, "complete")
        self.assertGreater(len(batches), 3)
        self.assertEqual(set(batches), {64})


@unittest.skipUnless(os.environ.get("TRUEMEMORY_PREPARED_NATIVE_SQLITE_TEST") == "1", "GPUBox-only native SQLite gate")
class TestNativeTierSource(unittest.TestCase):
    def test_real_vec0_pair_restart_and_delta(self):
        import sqlite_vec

        modules = load_modules()
        api, source = modules["tier_switch.source"], modules["rebuild_source"]
        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.executescript("""
                CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT, sender TEXT, recipient TEXT, timestamp TEXT);
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT);
                CREATE VIRTUAL TABLE vec_messages_synthetic USING vec0(embedding float[2] distance_metric=cosine);
                CREATE VIRTUAL TABLE vec_messages_sep_synthetic USING vec0(embedding float[2] distance_metric=cosine);
            """)
            source.ensure_rebuild_tracking(conn)
            conn.executemany("INSERT INTO messages(id,content) VALUES (?,?)", [(-3, "synthetic-negative"), (0, "synthetic-zero")])
            conn.commit()

            def plan():
                return api.plan_tier_source(conn, model_id="synthetic-model", dimension=2, targets=TARGETS)

            current = api.initialize_tier_source(conn, plan())
            page = api.read_tier_page(conn, current)
            left, right = [struct.pack("2f", 1., 0.)] * 2, [struct.pack("2f", 0., 1.)] * 2
            current, pending = api.publish_tier_prefix(conn, current, page, completion=left[:1], separation=right[:1])
            self.assertEqual(plan().manifest, current.manifest)
            current, pending = api.publish_tier_prefix(conn, current, pending, completion=left[1:], separation=right[1:])
            self.assertTrue(api.finish_tier_source(conn, current).current_source_complete)
            conn.execute("INSERT INTO messages(id,content) VALUES (2,'synthetic-tail')")
            conn.commit()
            current = api.initialize_tier_source(conn, plan())
            page = api.read_tier_page(conn, current)
            current, _ = api.publish_tier_prefix(conn, current, page, completion=left[:1], separation=right[:1])
            self.assertTrue(api.finish_tier_source(conn, current).current_source_complete)
            self.assertEqual(plan().action, "complete")
            for table in TARGETS:
                self.assertEqual([row[0] for row in conn.execute(f'SELECT rowid FROM "{table}" ORDER BY rowid')], [-3, 0, 2])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
