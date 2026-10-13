"""Database journal faults with synthetic in-memory SQL and no model imports."""

from __future__ import annotations

import builtins
import dataclasses
import json
import os
import runpy
import sqlite3
import struct
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = runpy.run_path(str(Path(__file__).with_name("test-tier-job-identity-795.py")))
ROOT = Path(__file__).resolve().parents[1]
TABLES = ("vec_messages_custom", "vec_messages_sep_custom")


class Rows:
    def __init__(self, values: list[tuple]) -> None:
        self.values = values

    def fetchone(self):
        return self.values[0] if self.values else None


class JournalConnection(BASE["MockFileConnection"]):
    fake_vec_schema = True
    vec_declaration: str | None = None
    armed = False
    fault_contains: str | None = None
    fault: BaseException | None = None
    commit_mode = ""
    rollback_mode = ""
    commits = 0
    rollbacks = 0
    record_payload_reads = 0

    def execute(self, sql: str, parameters: tuple = ()):
        if sql == "SELECT value FROM main.metadata WHERE key=?" and parameters[0] in {
            "tier_selected_v1", "tier_activation_v1",
        }:
            self.record_payload_reads += 1
        if self.armed and self.fault_contains is not None and self.fault_contains in sql:
            raise self.fault
        if (self.fake_vec_schema and sql == "SELECT sql FROM main.sqlite_master WHERE name=? COLLATE NOCASE"
                and parameters[0] in TABLES):
            ddl = self.vec_declaration or (
                f'CREATE VIRTUAL TABLE "{parameters[0]}" USING vec0(embedding float[2] distance_metric=cosine)'
            )
            return Rows([(ddl,)])
        return super().execute(sql, parameters)

    def commit(self) -> None:
        self.commits += 1
        if self.armed and self.commit_mode == "before":
            raise self.fault
        if self.armed and self.commit_mode == "retained":
            return
        super().commit()
        if self.armed and self.commit_mode == "after":
            raise self.fault

    def rollback(self) -> None:
        self.rollbacks += 1
        if self.armed and self.rollback_mode == "before":
            raise self.fault
        if self.armed and self.rollback_mode == "retained":
            return
        super().rollback()
        if self.armed and self.rollback_mode == "after":
            raise self.fault


class TestActivationJournal(unittest.TestCase):
    native_pair = False

    def setUp(self) -> None:
        self.fixture = BASE["TestTierJobIdentity"]()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.modules = self.fixture.modules

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"torch", "numpy", "sentence_transformers", "model2vec", "sqlite_vec"}:
                raise AssertionError("Journal imports must stay model/native free")
            if name == "psutil":
                return types.SimpleNamespace()
            if name.startswith("truememory."):
                short = name.removeprefix("truememory.")
                if short in self.modules:
                    return self.modules[short]
                raise AssertionError("Unexpected application import: " + name)
            return builtins.__import__(name, globals, locals, fromlist, level)

        for name in ("tier_config", "tier_switch.cache", "tier_switch.source", "tier_switch.activation"):
            module = types.ModuleType("synthetic_activation_" + name.replace(".", "_"))
            sys.modules[module.__name__] = module
            module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
            path = ROOT / "truememory" / (name.replace(".", "/") + ".py")
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
            self.modules[name] = module
        self.api = self.modules["tier_switch.activation"]
        self.job_api = self.modules["tier_switch.job"]
        self.source = self.modules["tier_switch.source"]
        self.conn = BASE["CONNECT"](":memory:", factory=JournalConnection)
        self.addCleanup(self.conn.close)
        self.conn.executescript("""
            CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT, sender TEXT, recipient TEXT, timestamp TEXT);
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT);
            CREATE TABLE vector_cache_registry (
                tier_group TEXT PRIMARY KEY, vec_table TEXT NOT NULL, sep_table TEXT NOT NULL,
                last_embedded_id INTEGER DEFAULT 0, vector_count INTEGER DEFAULT 0,
                model_name TEXT, embedding_dim INTEGER DEFAULT 256, last_updated REAL, created REAL
            );
            CREATE TABLE old_serving_pair(embedding BLOB);
            INSERT INTO old_serving_pair(rowid,embedding) VALUES(9,x'0102');
            INSERT INTO metadata(key,value) VALUES('embed_model','old-synthetic'),('embed_dim','7');
        """)
        if self.native_pair:
            import sqlite_vec
            self.conn.fake_vec_schema = False
            self.conn.enable_load_extension(True)
            try:
                sqlite_vec.load(self.conn)
            finally:
                self.conn.enable_load_extension(False)
            for name in TABLES:
                self.conn.execute(f'CREATE VIRTUAL TABLE "{name}" USING vec0(embedding float[2] distance_metric=cosine)')
        else:
            for name in TABLES:
                self.conn.execute(f'CREATE TABLE "{name}"(embedding BLOB)')
        self.modules["rebuild_source"].ensure_rebuild_tracking(self.conn)
        self.target = self.modules["embedding_target"].EmbeddingTarget("custom", "synthetic/encoder", 2, "custom")
        self.job = self.job_api.select_tier_job(self.conn, self.target)
        self.reranker = "synthetic/reranker"

    def stage(self, *, expected=None, reranker=None):
        return self.api.stage_activation_intent(
            self.conn, self.job, expected_generation=expected, reranker_id=reranker or self.reranker,
        )

    def build(self, ids=(-3, 0, 5), *, intent=None):
        if ids:
            self.conn.executemany("INSERT INTO messages VALUES(?,?,?,?,?)", [
                (mid, f"synthetic-{mid}", "sender", "recipient", "2026-01-01") for mid in ids
            ])
            self.conn.commit()
        intent = self.stage() if intent is None else intent
        plan = self.source.plan_tier_source(
            self.conn, model_id=self.target.model_id, dimension=self.target.dimension, targets=self.target.tables,
        )
        plan = self.source.initialize_tier_source(self.conn, plan)
        while plan.manifest.consumed < plan.manifest.total:
            page = self.source.read_tier_page(self.conn, plan, size=2)
            count = len(page.rows)
            plan, pending = self.source.publish_tier_prefix(
                self.conn, plan, page, completion=[struct.pack("2f", 1, 0)] * count,
                separation=[struct.pack("2f", 0, 1)] * count,
            )
            self.assertEqual(pending.rows, ())
        completion = self.source.finish_tier_source(self.conn, plan)
        self.assertTrue(completion.current_source_complete)
        return intent, completion.plan

    def commit(self, intent, plan):
        return self.api.commit_certified_selection(self.conn, intent, self.job, plan)

    def marker(self):
        return self.conn.execute(f"SELECT * FROM {self.job_api._SELECTION_TABLE}").fetchone()

    def assert_old(self) -> None:
        self.assertEqual(self.conn.execute("SELECT value FROM main.metadata WHERE key='embed_model'").fetchone(), ("old-synthetic",))
        self.assertEqual(self.conn.execute("SELECT value FROM main.metadata WHERE key='embed_dim'").fetchone(), ("7",))
        self.assertIsNone(self.conn.execute("SELECT value FROM main.metadata WHERE key='tier_selected_v1'").fetchone())
        self.assertEqual(self.conn.execute("SELECT * FROM old_serving_pair").fetchall(), [(b"\x01\x02",)])

    def test_owned_certification_exact_records_one_commit_and_job_retirement(self) -> None:
        intent, plan = self.build()
        before = self.conn.commits
        selected = self.commit(intent, plan)
        self.assertEqual(self.conn.commits, before + 1)
        self.assertFalse(self.conn.in_transaction)
        self.assertIsNone(self.marker())
        self.assertEqual(selected.target, self.target)
        self.assertEqual(selected.reranker_id, self.reranker)
        self.assertEqual(selected.tables, TABLES)
        self.assertEqual((selected.vector_count, selected.cursor), (3, 5))
        self.assertFalse(selected.config_acknowledged)
        self.assertEqual(self.api.read_activation_state(self.conn).selection, selected)
        self.assertEqual(self.api.read_activation_state(self.conn).intent.state, "db_selected")
        self.assertEqual(self.conn.execute("SELECT vec_table,sep_table,last_embedded_id,vector_count,model_name,embedding_dim FROM vector_cache_registry").fetchone(),
                         (*TABLES, 5, 3, "synthetic/encoder", 2))
        self.assertEqual(self.conn.execute("SELECT * FROM old_serving_pair").fetchall(), [(b"\x01\x02",)])
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='embed_model'").fetchone(), ("synthetic/encoder",))
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='embed_dim'").fetchone(), ("2",))
        new_job = self.job_api.select_tier_job(self.conn, self.target)
        self.assertNotEqual(new_job.job_id, self.job.job_id)
        self.assertEqual(self.marker()[1], new_job.job_id)
        with self.assertRaises(self.job_api.TierJobSelectionError):
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.job_api._retire_selected_tier_job_in_writer(self.conn, self.job)
            finally:
                self.conn.rollback()
        self.assertEqual(self.marker()[1], new_job.job_id)

    def test_empty_source_is_certified_without_fabricated_last_id(self) -> None:
        intent, plan = self.build(ids=())
        result = self.commit(intent, plan)
        self.assertEqual((result.vector_count, result.cursor), (0, None))
        self.assertEqual(self.conn.execute("SELECT last_embedded_id,vector_count FROM vector_cache_registry").fetchone(), (0, 0))

    def test_stage_is_idempotent_and_conflicting_live_intent_refuses(self) -> None:
        intent = self.stage()
        before = self.conn.total_changes
        self.assertEqual(self.stage(), intent)
        self.assertEqual(self.conn.total_changes, before)
        with self.assertRaises(self.api.TierActivationError):
            self.stage(reranker="synthetic/other")
        self.assertEqual(self.api.read_activation_state(self.conn).intent, intent)
        self.assert_old()

    def test_replacement_job_may_replace_abandoned_intent_only_after_binding(self) -> None:
        first = self.stage()
        self.job_api.end_selected_tier_job(self.conn, self.job, outcome="failed")
        self.job = self.job_api.select_tier_job(self.conn, self.target, previous_job_id=self.job.job_id)
        second = self.stage()
        self.assertNotEqual(first.intent_id, second.intent_id)
        self.assertEqual(second.job_id, self.job.job_id)
        self.assert_old()

    def test_config_acknowledgement_is_generation_cas_and_no_runtime_claim(self) -> None:
        intent, plan = self.build()
        selected = self.commit(intent, plan)
        with self.assertRaises(self.api.TierActivationError):
            self.api.acknowledge_config(self.conn, generation="0" * 32)
        self.assertFalse(self.api.read_activation_state(self.conn).selection.config_acknowledged)
        acked = self.api.acknowledge_config(self.conn, generation=selected.generation)
        self.assertTrue(acked.config_acknowledged)
        before = self.conn.total_changes
        self.assertEqual(self.api.acknowledge_config(self.conn, generation=selected.generation), acked)
        self.assertEqual(self.conn.total_changes, before)
        self.assertNotIn("runtime", dataclasses.asdict(acked))
        self.job = self.job_api.select_tier_job(self.conn, self.target)
        self.stage(expected=selected.generation)
        with self.assertRaises(self.api.TierActivationError):
            self.api.acknowledge_config(self.conn, generation=selected.generation)

    def test_missing_or_wrong_expected_generation_never_overwrites_selection(self) -> None:
        with self.assertRaises(self.api.TierActivationError):
            self.stage(expected="0" * 32)
        intent, plan = self.build()
        selected = self.commit(intent, plan)
        self.job = self.job_api.select_tier_job(self.conn, self.target)
        for expected in (None, "0" * 32, True):
            with self.subTest(expected=expected), self.assertRaises(self.api.TierActivationError):
                self.stage(expected=expected)
        self.assertEqual(self.api.read_activation_state(self.conn).selection, selected)

    def test_final_writer_rejects_changed_selected_generation(self) -> None:
        intent, plan = self.build()
        previous = self.commit(intent, plan)
        self.api.acknowledge_config(self.conn, generation=previous.generation)
        self.job = self.job_api.select_tier_job(self.conn, self.target)
        intent = self.stage(expected=previous.generation)
        plan = self.source.plan_tier_source(self.conn, model_id=self.target.model_id,
                                            dimension=2, targets=TABLES)
        changed = dataclasses.replace(previous, generation="0" * 32)
        self.conn.execute("UPDATE metadata SET value=? WHERE key='tier_selected_v1'", (self.api._dump(changed),))
        self.conn.commit()
        marker = self.marker()
        with self.assertRaisesRegex(self.api.TierActivationError, "Selected generation changed"):
            self.commit(intent, plan)
        self.assertEqual(self.marker(), marker)
        raw = self.conn.execute("SELECT value FROM main.metadata WHERE key='tier_selected_v1'").fetchone()[0]
        self.assertEqual(self.api._parse(raw, self.api.TierSelection), changed)
        with self.assertRaises(self.api.TierActivationError):
            self.api.read_activation_state(self.conn)

    def test_staged_intent_must_match_current_selection_or_absence(self) -> None:
        intent = self.stage()
        malformed = dataclasses.replace(intent, expected_generation="0" * 32)
        self.conn.execute("UPDATE metadata SET value=? WHERE key='tier_activation_v1'", (self.api._dump(malformed),))
        self.conn.commit()
        with self.assertRaisesRegex(self.api.TierActivationError, "Selected generation changed"):
            self.api.read_activation_state(self.conn)
        self.conn.execute("UPDATE metadata SET value=? WHERE key='tier_activation_v1'", (self.api._dump(intent),))
        self.conn.commit()
        intent, plan = self.build(intent=intent)
        selected = self.commit(intent, plan)
        for expected in (None, "0" * 32, selected.generation):
            malformed = dataclasses.replace(intent, expected_generation=expected)
            self.conn.execute("UPDATE metadata SET value=? WHERE key='tier_activation_v1'", (self.api._dump(malformed),))
            self.conn.commit()
            with self.subTest(expected=expected), self.assertRaises(self.api.TierActivationError):
                self.api.read_activation_state(self.conn)
        self.assertNotEqual(selected.generation, "0" * 32)

    def test_new_intent_preserves_pending_config_acknowledgement_then_retries(self) -> None:
        intent, plan = self.build()
        selected = self.commit(intent, plan)
        previous_state = self.api.read_activation_state(self.conn)
        self.job = self.job_api.select_tier_job(self.conn, self.target)
        before = self.conn.total_changes
        with self.assertRaisesRegex(self.api.TierActivationError, "awaits config acknowledgement"):
            self.stage(expected=selected.generation)
        self.assertEqual(self.conn.total_changes, before)
        self.assertEqual(self.api.read_activation_state(self.conn), previous_state)
        acked = self.api.acknowledge_config(self.conn, generation=selected.generation)
        self.assertTrue(acked.config_acknowledged)
        next_intent = self.stage(expected=selected.generation)
        self.assertEqual(next_intent.expected_generation, selected.generation)
        self.assertNotEqual(next_intent.intent_id, intent.intent_id)

    def test_frozen_custom_identity_never_reresolves_configuration(self) -> None:
        intent, plan = self.build()
        with patch.object(self.modules["tier_config"], "get_tier_config", side_effect=AssertionError("No config reads")):
            result = self.commit(intent, plan)
        self.assertEqual(result.target, self.target)
        self.assertEqual(result.reranker_id, self.reranker)

    def test_idempotent_intent_does_not_invalidate_existing_source_plan(self) -> None:
        intent, plan = self.build()
        before = self.conn.total_changes
        self.assertEqual(self.stage(), intent)
        self.assertEqual(self.conn.total_changes, before)
        self.assertEqual(self.commit(intent, plan).manifest_generation, plan.manifest.generation)

    def test_borrowed_transactions_and_read_only_helpers_never_end_caller_work(self) -> None:
        intent, plan = self.build()
        self.conn.execute("BEGIN IMMEDIATE")
        before = self.conn.total_changes
        self.job_api.check_selected_tier_job_in_writer(self.conn, self.job)
        self.source.require_current_completion_in_writer(self.conn, plan)
        self.assertEqual(self.conn.total_changes, before)
        self.assertTrue(self.conn.in_transaction)
        for call in (lambda: self.stage(), lambda: self.commit(intent, plan),
                     lambda: self.api.read_activation_state(self.conn),
                     lambda: self.api.acknowledge_config(self.conn, generation="0" * 32)):
            with self.assertRaises(self.api.TierActivationError):
                call()
            self.assertTrue(self.conn.in_transaction)
            self.assertEqual(self.conn.total_changes, before)
        self.conn.rollback()
        for call in (lambda: self.job_api.check_selected_tier_job_in_writer(self.conn, self.job),
                     lambda: self.source.require_current_completion_in_writer(self.conn, plan)):
            with self.assertRaises((self.job_api.TierJobSelectionError, self.source.TierSourceUntrusted)):
                call()

    def test_post_stage_unrelated_status_write_invalidates_plan(self) -> None:
        intent, plan = self.build()
        self.conn.execute("INSERT INTO metadata VALUES('unrelated','status',NULL)")
        self.conn.commit()
        with self.assertRaises(self.source.TierSourceUntrusted):
            self.commit(intent, plan)
        self.assert_old()
        self.assertIsNotNone(self.marker())

    def test_writer_rechecks_source_marker_receipt_and_target(self) -> None:
        cases = (
            "INSERT INTO messages VALUES(8,'tail','sender','recipient','time')",
            "UPDATE messages SET content='corrected' WHERE id=0",
            f"UPDATE {self.job_api._SELECTION_TABLE} SET state='cancelled'",
            "UPDATE metadata SET value='{}' WHERE key LIKE 'tier_pair_receipt_v1:%'",
            "UPDATE vec_messages_custom SET embedding=x'0000000000000000' WHERE rowid=0",
        )
        for mutation in cases:
            with self.subTest(mutation=mutation):
                fixture = TestActivationJournal()
                fixture.setUp()
                try:
                    intent, plan = fixture.build()
                    fixture.conn.execute(mutation)
                    fixture.conn.commit()
                    with self.assertRaises((fixture.api.TierActivationError, fixture.source.TierSourceUntrusted,
                                            fixture.job_api.TierJobSelectionError)):
                        fixture.commit(intent, plan)
                    fixture.assert_old()
                finally:
                    fixture.doCleanups()

    def test_revalidation_rejects_same_width_corrupted_pair(self) -> None:
        intent, plan = self.build()
        self.conn.execute("UPDATE vec_messages_custom SET embedding=? WHERE rowid=0", (struct.pack("2f", 0, 0),))
        self.conn.commit()
        # Bypass only the cheap change counter to exercise the independent final
        # digest audit; no forged receipt is supplied.
        plan = dataclasses.replace(plan, changes=self.conn.total_changes)
        with self.assertRaises(self.source.TierSourceUntrusted):
            self.commit(intent, plan)
        self.assert_old()

    def test_incomplete_fabricated_and_wrong_identity_certificates_refuse(self) -> None:
        intent, plan = self.build()
        altered = (
            dataclasses.replace(plan, action="resume"),
            dataclasses.replace(plan, digest="0" * 64),
            dataclasses.replace(plan, manifest=dataclasses.replace(plan.manifest, dimension=3)),
            dataclasses.replace(plan, manifest=dataclasses.replace(plan.manifest, model="synthetic/wrong")),
            dataclasses.replace(plan, manifest=dataclasses.replace(plan.manifest, complete=False)),
        )
        for current in altered:
            with self.subTest(plan=current.manifest.dimension), self.assertRaises((self.api.TierActivationError, self.source.TierSourceUntrusted)):
                self.commit(intent, current)
            self.assert_old()
        with self.assertRaises(self.api.TierActivationError):
            self.commit(dataclasses.replace(intent, reranker_id="synthetic/changed"), plan)

    def test_schema_dimension_metric_and_ordinary_table_refuse(self) -> None:
        intent, plan = self.build()
        for ddl in (
            "CREATE TABLE vec_messages_custom(embedding BLOB)",
            "CREATE VIRTUAL TABLE vec_messages_custom USING vec0(embedding float[3] distance_metric=cosine)",
            "CREATE VIRTUAL TABLE vec_messages_custom USING vec0(embedding float[2] distance_metric=l2)",
        ):
            self.conn.vec_declaration = ddl
            with self.subTest(ddl=ddl), self.assertRaises(self.api.TierActivationError):
                self.commit(intent, plan)
            self.assert_old()

    def test_trigger_side_effects_and_temp_case_aliases_refuse_before_writes(self) -> None:
        for ddl in (
            "CREATE TRIGGER bad_metadata AFTER UPDATE ON metadata BEGIN UPDATE messages SET content='changed'; END",
            "CREATE TRIGGER bad_registry AFTER INSERT ON vector_cache_registry BEGIN DELETE FROM messages; END",
            "CREATE INDEX bad_index ON metadata(value)",
            "CREATE TEMP TABLE METADATA(key TEXT,value TEXT)",
            "CREATE TEMP TABLE VECTOR_CACHE_REGISTRY(tier_group TEXT)",
        ):
            with self.subTest(ddl=ddl):
                fixture = TestActivationJournal()
                fixture.setUp()
                try:
                    intent, plan = fixture.build()
                    fixture.conn.execute(ddl)
                    before = fixture.conn.total_changes
                    with self.assertRaises(fixture.api.TierActivationError):
                        fixture.commit(intent, plan)
                    self.assertEqual(fixture.conn.total_changes, before)
                    fixture.assert_old()
                finally:
                    fixture.doCleanups()

    def test_marker_incoming_foreign_key_refuses_even_with_enforcement_off(self) -> None:
        for enforcement in (False, True):
            with self.subTest(enforcement=enforcement):
                fixture = TestActivationJournal()
                fixture.setUp()
                try:
                    fixture.conn.execute(f"PRAGMA foreign_keys={int(enforcement)}")
                    fixture.conn.execute(f"CREATE TABLE marker_child(id INTEGER REFERENCES {fixture.job_api._SELECTION_TABLE.upper()}(singleton) ON DELETE CASCADE)")
                    fixture.conn.execute("INSERT INTO marker_child VALUES(1)")
                    fixture.conn.commit()
                    intent, plan = fixture.build()
                    with self.assertRaises(fixture.job_api.TierJobSelectionError):
                        fixture.commit(intent, plan)
                    self.assertEqual(fixture.conn.execute("SELECT * FROM marker_child").fetchall(), [(1,)])
                    self.assertIsNotNone(fixture.marker())
                    fixture.assert_old()
                finally:
                    fixture.doCleanups()

    def test_registry_and_metadata_upserts_do_not_delete_parent_rows(self) -> None:
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript("""
            INSERT INTO vector_cache_registry(tier_group,vec_table,sep_table) VALUES('custom','old','old_sep');
            CREATE TABLE metadata_child(parent TEXT REFERENCES metadata(key) ON DELETE CASCADE);
            INSERT INTO metadata_child VALUES('embed_model');
            CREATE TABLE registry_child(parent TEXT REFERENCES vector_cache_registry(tier_group) ON DELETE CASCADE);
            INSERT INTO registry_child VALUES('custom');
        """)
        intent, plan = self.build()
        self.commit(intent, plan)
        self.assertEqual(self.conn.execute("SELECT * FROM metadata_child").fetchall(), [("embed_model",)])
        self.assertEqual(self.conn.execute("SELECT * FROM registry_child").fetchall(), [("custom",)])

    def test_marker_retirement_bounds_table_names_before_materialization(self) -> None:
        self.conn.execute('CREATE TABLE "' + 'x' * 600 + '"(id INTEGER)')
        intent, plan = self.build()
        with self.assertRaisesRegex(self.job_api.TierJobSelectionError, "schema exceeds"):
            self.commit(intent, plan)
        self.assertIsNotNone(self.marker())
        self.assert_old()

    def test_each_publication_write_failure_rolls_back_marker_registry_and_journal(self) -> None:
        for fault in ("DELETE FROM main.truememory_tier_selected_job_v1", "INSERT INTO vector_cache_registry",
                      "INSERT INTO main.metadata"):
            with self.subTest(fault=fault):
                fixture = TestActivationJournal()
                fixture.setUp()
                try:
                    intent, plan = fixture.build()
                    marker = fixture.marker()
                    failure = sqlite3.OperationalError("synthetic statement failure")
                    fixture.conn.armed, fixture.conn.fault_contains, fixture.conn.fault = True, fault, failure
                    with self.assertRaises(sqlite3.OperationalError) as caught:
                        fixture.commit(intent, plan)
                    self.assertIs(caught.exception, failure)
                    fixture.conn.armed = False
                    self.assertFalse(fixture.conn.in_transaction)
                    self.assertEqual(fixture.marker(), marker)
                    fixture.assert_old()
                    self.assertEqual(fixture.api.read_activation_state(fixture.conn).intent.state, "staged")
                finally:
                    fixture.doCleanups()

    def test_commit_before_after_and_noop_never_fake_success(self) -> None:
        for mode in ("before", "after", "retained"):
            with self.subTest(mode=mode):
                fixture = TestActivationJournal()
                fixture.setUp()
                try:
                    intent, plan = fixture.build()
                    failure = BASE["Cancelled"]("synthetic commit cancellation")
                    fixture.conn.armed, fixture.conn.commit_mode, fixture.conn.fault = True, mode, failure
                    with self.assertRaises((BASE["Cancelled"], fixture.api.TierActivationError)):
                        fixture.commit(intent, plan)
                    fixture.conn.armed = False
                    self.assertFalse(fixture.conn.in_transaction)
                    state = fixture.api.read_activation_state(fixture.conn)
                    if mode == "after":
                        self.assertIsNotNone(state.selection)
                        self.assertFalse(state.selection.config_acknowledged)
                        self.assertIsNone(fixture.marker())
                    else:
                        fixture.assert_old()
                        self.assertIsNotNone(fixture.marker())
                finally:
                    fixture.doCleanups()

    def test_uncertain_rollback_propagates_without_followup_status_sql(self) -> None:
        for mode in ("before", "after", "retained"):
            with self.subTest(mode=mode):
                fixture = TestActivationJournal()
                fixture.setUp()
                try:
                    intent, plan = fixture.build()
                    fixture.conn.armed = True
                    fixture.conn.fault_contains = "INSERT INTO main.metadata"
                    fixture.conn.fault = MemoryError("synthetic rollback failure")
                    fixture.conn.rollback_mode = mode
                    before = fixture.conn.commits
                    with self.assertRaises((MemoryError, fixture.api.TierActivationError)):
                        fixture.commit(intent, plan)
                    self.assertEqual(fixture.conn.commits, before)
                    self.assertEqual(fixture.conn.in_transaction, mode != "after")
                    if fixture.conn.in_transaction:
                        with self.assertRaises(fixture.api.TierActivationError):
                            fixture.api.read_activation_state(fixture.conn)
                    fixture.conn.armed = False
                    fixture.conn.rollback()
                    fixture.assert_old()
                    self.assertIsNotNone(fixture.marker())
                finally:
                    fixture.doCleanups()

    def test_read_snapshot_cleanup_never_retries_uncertain_rollback(self) -> None:
        self.stage()
        for mode in ("before", "after", "retained"):
            with self.subTest(mode=mode):
                failure = sqlite3.OperationalError("synthetic read rollback failure")
                self.conn.armed, self.conn.rollback_mode, self.conn.fault = True, mode, failure
                before = self.conn.rollbacks
                try:
                    with self.assertRaises((sqlite3.OperationalError, self.api.TierActivationError)) as caught:
                        self.api.read_activation_state(self.conn)
                    self.assertEqual(self.conn.rollbacks, before + 1)
                    self.assertEqual(self.conn.in_transaction, mode != "after")
                    if mode != "retained":
                        self.assertIs(caught.exception, failure)
                finally:
                    self.conn.armed = False
                    self.conn.rollback()
                self.assert_old()

    def test_oversized_and_nul_tail_records_reject_before_payload_read(self) -> None:
        self.stage()
        for value in ("x" * 20000, "{}\x00" + "x" * 20000, b"{}"):
            self.conn.execute("UPDATE metadata SET value=? WHERE key='tier_activation_v1'", (value,))
            self.conn.commit()
            before = self.conn.record_payload_reads
            with self.assertRaises(self.api.TierActivationError):
                self.api.read_activation_state(self.conn)
            self.assertEqual(self.conn.record_payload_reads, before)

    def test_strict_json_fields_types_and_frozen_identity(self) -> None:
        intent = self.stage()
        valid = json.loads(self.api._dump(intent))
        malformed = ["[]", "{}", '{"version":1,"version":1}', self.api._dump(intent) + "\x00"]
        for key, value in (("version", True), ("unexpected", 1), ("expected_generation", False), ("state", "complete")):
            item = dict(valid)
            item[key] = value
            malformed.append(json.dumps(item))
        for field, value in (("dimension", True), ("model_id", "x" * 513), ("tier", "pro")):
            item = dict(valid, target=dict(valid["target"], **{field: value}))
            malformed.append(json.dumps(item))
        for raw in malformed:
            self.conn.execute("UPDATE metadata SET value=? WHERE key='tier_activation_v1'", (raw,))
            self.conn.commit()
            with self.subTest(raw=raw[:30]), self.assertRaises(self.api.TierActivationError):
                self.api.read_activation_state(self.conn)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            intent.reranker_id = "changed"

    def test_selected_record_counts_tables_hashes_and_ack_are_strict(self) -> None:
        intent, plan = self.build()
        selected = self.commit(intent, plan)
        valid = json.loads(self.api._dump(selected))
        for field, value in (("vector_count", True), ("vector_count", -1), ("vector_count", 2**63),
                             ("cursor", None), ("cursor", 2**63), ("config_acknowledged", 1),
                             ("tables", list(reversed(TABLES))), ("pair_digest", "wrong"),
                             ("job_id", "0" * 32), ("config_acknowledged", True)):
            self.conn.execute("UPDATE metadata SET value=? WHERE key='tier_selected_v1'", (json.dumps(dict(valid, **{field: value})),))
            self.conn.commit()
            with self.subTest(field=field, value=value), self.assertRaises(self.api.TierActivationError):
                self.api.read_activation_state(self.conn)

    def test_registry_default_commits_and_additive_mode_preserves_borrowed_transaction(self) -> None:
        registry = self.modules["tier_switch.cache"].VectorCacheRegistry
        before = self.conn.commits
        registry.set(self.conn, "custom", model_name="synthetic/encoder", embedding_dim=2)
        self.assertEqual(self.conn.commits, before + 1)
        self.conn.execute("BEGIN")
        registry.set(self.conn, "custom", model_name="synthetic/other", embedding_dim=3, commit=False)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT model_name,embedding_dim FROM vector_cache_registry").fetchone(), ("synthetic/encoder", 2))


@unittest.skipUnless(os.environ.get("TRUEMEMORY_TEST_NATIVE_VEC") == "1", "Native vec0 gate belongs to GPUBox")
class TestNativeActivationJournal(unittest.TestCase):
    def test_actual_vec0_certification_and_job_retirement(self) -> None:
        fixture = TestActivationJournal()
        fixture.native_pair = True
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.test_owned_certification_exact_records_one_commit_and_job_retirement()


if __name__ == "__main__":
    unittest.main()
