"""Policy/projection faults using synthetic config and in-memory SQLite only."""
from __future__ import annotations

import builtins
import dataclasses
import json
import runpy
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = runpy.run_path(str(Path(__file__).with_name("test-tier-activation-journal-795.py")))
ROOT = Path(__file__).resolve().parents[1]
RERANKER = "Alibaba-NLP/gte-reranker-modernbert-base"
LEGACY = ("vec_messages", "vec_messages_sep")


class TestActivationPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = BASE["TestActivationJournal"]()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.api = self.fixture.api
        self.conn = self.fixture.conn
        self.jobs = self.fixture.job_api
        self.source = self.fixture.source
        self.Target = self.fixture.modules["embedding_target"].EmbeddingTarget
        self.base = self.Target("base", "qwen3_256", 256, "basepro")
        self.pro = self.Target("pro", "qwen3_256", 256, "basepro")
        self.guard = self.api.ConfigGuard(True, "base", "a" * 32)
        for table in (*LEGACY, *self.base.tables):
            self.conn.execute(f'CREATE TABLE "{table}"(embedding BLOB)')
        self.conn.execute("UPDATE metadata SET value='qwen3_256' WHERE key='embed_model'")
        self.conn.execute("UPDATE metadata SET value='256' WHERE key='embed_dim'")
        self.conn.commit()
        schema = self.api._schema_sql

        def fake_native_schema(conn, table):
            actual = schema(conn, table)
            if table in (*LEGACY, *self.base.tables):
                return f'CREATE VIRTUAL TABLE "{table}" USING vec0(embedding float[256] distance_metric=cosine)'
            return actual

        self.schema_patch = patch.object(self.api, "_schema_sql", fake_native_schema)
        self.schema_patch.start()
        self.addCleanup(self.schema_patch.stop)

        projection = types.ModuleType("synthetic_policy_projection")
        sys.modules[projection.__name__] = projection
        self.addCleanup(sys.modules.pop, projection.__name__, None)

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "truememory.tier_switch.activation":
                return self.api
            if name.startswith("truememory") or name in {"torch", "numpy", "sqlite_vec", "sentence_transformers"}:
                raise AssertionError("Projection must stay model/native free")
            return builtins.__import__(name, globals, locals, fromlist, level)

        projection.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
        path = ROOT / "truememory/tier_switch/projection.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), projection.__dict__)
        self.projection = projection

    def stage(self):
        return self.api.stage_activation_intent(
            self.conn, self.fixture.job, expected_generation=None, reranker_id="synthetic/reranker",
            expected_config=self.guard, status_id=17,
        )

    def publish(self, intent=None, **kwargs):
        arguments = dict(expected_selection_generation=None, expected_intent_id=intent.intent_id if intent else None,
                         target=self.pro, reranker_id=RERANKER, expected_config=self.guard, legacy_pair=LEGACY)
        arguments.update(kwargs)
        return self.api.commit_config_only_transition(self.conn, **arguments)

    def terminal_marker(self):
        self.jobs.end_selected_tier_job(self.conn, self.fixture.job, outcome="cancelled")

    def test_legacy_policy_changes_only_journal_and_exact_marker(self):
        intent = self.stage()
        before = self.conn.execute("SELECT key,value FROM metadata WHERE key!='tier_activation_v1' ORDER BY key").fetchall()
        statements = []
        self.conn.set_trace_callback(statements.append)
        result = self.publish(intent)
        self.conn.set_trace_callback(None)
        self.assertIs(type(result), self.api.LegacyTierPolicy)
        self.assertEqual(result.tables, LEGACY)
        self.assertEqual(result.target, self.pro)
        self.assertFalse(result.config_acknowledged)
        self.assertIsNone(self.api.read_activation_state(self.conn).selection)
        self.assertEqual(self.api.read_activation_state(self.conn).legacy_policy, result)
        self.assertEqual(before, self.conn.execute("SELECT key,value FROM metadata WHERE key!='tier_activation_v1' ORDER BY key").fetchall())
        self.assertEqual(self.jobs.read_selected_job_marker(self.conn).state, "cancelled")
        self.assertFalse(any("FROM MAIN.MESSAGES" in sql.upper() or "FROM MAIN.VEC_MESSAGES" in sql.upper() for sql in statements))
        self.assertFalse(any("DELETE" in sql.upper() for sql in statements))

    def test_pending_policy_ack_cas_and_roundtrip(self):
        self.terminal_marker()
        result = self.publish()
        with self.assertRaises(self.api.TierActivationError):
            self.api.acknowledge_policy_config(self.conn, generation="f" * 32)
        with self.assertRaises(self.api.TierActivationError):
            self.publish(expected_intent_id=result.intent_id, target=self.base,
                         expected_config=self.api.ConfigGuard(True, "pro", result.generation))
        acknowledged = self.api.acknowledge_policy_config(self.conn, generation=result.generation)
        self.assertTrue(acknowledged.config_acknowledged)
        self.assertEqual(acknowledged, self.api.acknowledge_policy_config(self.conn, generation=result.generation))
        second = self.publish(expected_intent_id=result.intent_id, target=self.base,
                              expected_config=self.api.ConfigGuard(True, "pro", result.generation))
        self.assertNotEqual(result.generation, second.generation)
        self.assertEqual(second.target, self.base)
        self.assertIsNone(self.api.read_activation_state(self.conn).selection)

    def test_v2_rebuild_retains_acknowledged_legacy_policy_and_old_v1_bytes(self):
        old = self.fixture.stage()
        raw = self.api._dump(old)
        self.assertEqual(json.loads(raw)["version"], 1)
        self.assertNotIn("expected_config", json.loads(raw))
        self.assertEqual(self.api._dump(self.api._parse(raw, self.api.ActivationIntent)), raw)
        self.jobs.end_selected_tier_job(self.conn, self.fixture.job, outcome="cancelled")
        policy = self.publish(old)
        policy = self.api.acknowledge_policy_config(self.conn, generation=policy.generation)
        job = self.jobs.select_tier_job(self.conn, self.fixture.target, previous_job_id=self.fixture.job.job_id)
        guard = self.api.ConfigGuard(True, "pro", policy.generation)
        staged = self.api.stage_activation_intent(self.conn, job, expected_generation=None,
                                                  reranker_id="synthetic/reranker", expected_config=guard, status_id=23)
        self.assertEqual(staged.previous_policy, policy)
        self.assertEqual(self.api.read_activation_state(self.conn).legacy_policy, policy)
        self.assertEqual(json.loads(self.api._dump(staged))["kind"], "rebuild")
        self.assertEqual(self.api._parse(self.api._dump(staged), self.api.ActivationIntent), staged)
        self.assertEqual(self.api.stage_activation_intent(self.conn, job, expected_generation=None,
                         reranker_id="synthetic/reranker", expected_config=guard, status_id=23), staged)
        with self.assertRaises(self.api.TierActivationError):
            self.api.stage_activation_intent(self.conn, job, expected_generation=None, reranker_id="synthetic/reranker")

    def test_exact_superseded_job_and_rollback(self):
        intent = self.stage()
        original = self.jobs.read_selected_job_marker(self.conn)
        self.conn.execute(f"UPDATE {self.jobs._SELECTION_TABLE} SET job_id=?", ("b" * 32,))
        self.conn.commit()
        with self.assertRaises(self.jobs.TierJobSelectionError):
            self.publish(intent)
        self.assertEqual(self.jobs.read_selected_job_marker(self.conn).job_id, "b" * 32)
        self.conn.execute(f"UPDATE {self.jobs._SELECTION_TABLE} SET job_id=?", (original.job_id,))
        self.conn.commit()
        self.conn.fault_contains = "INSERT INTO main.metadata"
        self.conn.fault = MemoryError("synthetic journal write")
        self.conn.armed = True
        with self.assertRaises(MemoryError):
            self.publish(intent)
        self.conn.armed = False
        self.assertEqual(self.jobs.read_selected_job_marker(self.conn), original)
        self.assertEqual(self.api.read_activation_state(self.conn).intent, intent)

    def test_commit_after_success_is_pending_readback_not_false_rollback(self):
        intent = self.stage()
        self.conn.armed = True
        self.conn.commit_mode = "after"
        self.conn.fault = TimeoutError("synthetic post-commit")
        before = self.conn.rollbacks
        with self.assertRaises(TimeoutError):
            self.publish(intent)
        self.conn.armed = False
        self.assertEqual(self.conn.rollbacks, before)
        state = self.api.read_activation_state(self.conn)
        self.assertIsNotNone(state.legacy_policy)
        self.assertFalse(state.legacy_policy.config_acknowledged)
        self.assertEqual(self.jobs.read_selected_job_marker(self.conn).state, "cancelled")

    def test_certified_policy_keeps_certificate_without_recertifying_vectors(self):
        self.terminal_marker()
        self.fixture.target = self.base
        self.fixture.job = self.jobs.select_tier_job(self.conn, self.base, previous_job_id=self.fixture.job.job_id)
        self.fixture.reranker = RERANKER
        intent, plan = self.fixture.build(ids=())
        selected = self.fixture.commit(intent, plan)
        selected = self.api.acknowledge_config(self.conn, generation=selected.generation)
        statements = []
        self.conn.set_trace_callback(statements.append)
        result = self.publish(expected_selection_generation=selected.generation, expected_intent_id=selected.intent_id,
                              expected_config=self.api.ConfigGuard(True, "base", selected.generation), legacy_pair=None)
        self.conn.set_trace_callback(None)
        for field in dataclasses.fields(selected):
            if field.name not in {"generation", "intent_id", "target", "config_acknowledged"}:
                self.assertEqual(getattr(result, field.name), getattr(selected, field.name))
        self.assertNotEqual(result.generation, selected.generation)
        self.assertEqual(result.target, self.pro)
        self.assertIsNone(self.api.read_activation_state(self.conn).legacy_policy)
        self.assertFalse(any("FROM MAIN.MESSAGES" in sql.upper() for sql in statements))
        self.assertTrue(self.api.acknowledge_config(self.conn, generation=result.generation).config_acknowledged)

    def test_strict_invalid_policy_shapes_and_no_new_certificate(self):
        self.terminal_marker()
        bad = [dict(target=self.fixture.target), dict(reranker_id="synthetic/other"),
               dict(legacy_pair=("vec_messages_custom", "vec_messages_sep_custom")),
               dict(expected_config=None), dict(target=self.base), dict(expected_intent_id="b" * 32),
               dict(expected_selection_generation="b" * 32)]
        for fields in bad:
            with self.subTest(fields=list(fields)), self.assertRaises(self.api.TierActivationError):
                self.publish(**fields)
        self.assertIsNone(self.api.read_activation_state(self.conn).intent)
        self.assertIsNone(self.api.read_activation_state(self.conn).selection)

    def test_legacy_registry_metadata_and_native_schema_rejection(self):
        self.terminal_marker()
        self.conn.execute("UPDATE metadata SET value='model2vec' WHERE key='embed_model'")
        self.conn.commit()
        with self.assertRaises(self.api.TierActivationError):
            self.publish()
        self.conn.execute("UPDATE metadata SET value='qwen3_256' WHERE key='embed_model'")
        self.conn.execute("INSERT INTO vector_cache_registry(tier_group,vec_table,sep_table,model_name) VALUES('basepro','wrong','wrong','qwen3_256')")
        self.conn.commit()
        with self.assertRaises(self.api.TierActivationError):
            self.publish()
        self.schema_patch.stop()
        with self.assertRaises(self.api.TierActivationError):
            self.publish()

    def test_strict_nested_policy_and_nul_tail_byte_bound(self):
        self.terminal_marker()
        result = self.publish()
        raw = self.conn.execute("SELECT value FROM metadata WHERE key='tier_activation_v1'").fetchone()[0]
        value = json.loads(raw)
        for field, invalid in (("result_generation", True), ("state", []), ("tables", [[], []]),
                               ("expected_config", {"tier": "base"}), ("unknown", 1)):
            altered = dict(value, **{field: invalid})
            with self.subTest(field=field), self.assertRaises(self.api.TierActivationError):
                self.api._parse(json.dumps(altered), self.api.ActivationIntent)
        self.conn.execute("UPDATE metadata SET value=? WHERE key='tier_activation_v1'", (raw + "\x00" + "x" * 17000,))
        self.conn.commit()
        before = self.conn.record_payload_reads
        with self.assertRaises(self.api.TierActivationError):
            self.api.read_activation_state(self.conn)
        self.assertEqual(self.conn.record_payload_reads, before)
        self.assertEqual(result.target, self.pro)

    def test_progress_is_bounded_manifest_read_and_borrows_snapshot(self):
        intent, plan = self.fixture.build()
        statements = []
        self.conn.set_trace_callback(statements.append)
        self.conn.execute("BEGIN")
        marker = self.jobs.read_selected_job_marker(self.conn)
        progress = self.source.read_tier_progress(self.conn, model_id=self.fixture.target.model_id,
                    dimension=2, targets=self.fixture.target.tables)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.conn.set_trace_callback(None)
        self.assertEqual((progress.processed, progress.total, progress.outputs, progress.cursor, progress.complete), (3, 3, 3, 5, True))
        self.assertEqual((progress.tracker, progress.source_epoch, progress.source_schema_signature),
                         (marker.tracker, marker.source_epoch, marker.source_schema_signature))
        self.assertEqual(marker.job_id, intent.job_id)
        self.assertEqual(progress.generation, plan.manifest.generation)
        self.assertFalse(any("FROM MAIN.MESSAGES" in sql.upper() or "FROM MAIN.VEC_MESSAGES" in sql.upper() for sql in statements))

    def test_progress_missing_half_receipt_and_read_cleanup_failure(self):
        _, plan = self.fixture.build()
        self.conn.execute("DELETE FROM metadata WHERE key=?", (self.source._receipt_key(plan.manifest.targets),))
        self.conn.commit()
        with self.assertRaises(self.source.TierSourceUntrusted):
            self.source.read_tier_progress(self.conn, model_id=self.fixture.target.model_id, dimension=2, targets=plan.manifest.targets)
        self.assertFalse(self.conn.in_transaction)
        self.conn.armed = True
        self.conn.rollback_mode = "before"
        self.conn.fault = MemoryError("synthetic cleanup")
        before = self.conn.rollbacks
        with self.assertRaises(MemoryError):
            self.source.read_tier_progress(self.conn, model_id=self.fixture.target.model_id, dimension=2, targets=plan.manifest.targets)
        self.assertEqual(self.conn.rollbacks, before + 1)
        self.assertTrue(self.conn.in_transaction)
        self.conn.armed = False
        self.conn.rollback()

    def test_config_guard_seed_mirror_preserves_fresh_independent_fields(self):
        self.terminal_marker()
        with tempfile.TemporaryDirectory(prefix="synthetic-policy-") as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"tier": "base", "synthetic_setting": 1}))
            guard = self.projection.capture_config_guard(config_path=config)
            self.assertEqual(guard.tier, "base")
            self.assertEqual(len(guard.generation), 32)
            self.assertEqual(self.projection.capture_config_guard(config_path=config), guard)
            result = self.publish(expected_config=guard)
            self.projection.patch_config_fields({"synthetic_setting": 2, "other": True}, config_path=config)
            mirrored = self.projection.mirror_activation(result, expected_config=guard, config_path=config)
            self.assertEqual((mirrored.tier, mirrored.generation), ("pro", result.generation))
            self.assertEqual(json.loads(config.read_text(encoding="utf-8")), {"tier": "pro", "tier_activation_generation": result.generation,
                             "synthetic_setting": 2, "other": True})
            self.assertEqual(self.projection.mirror_activation(result, expected_config=None, config_path=config), mirrored)
            self.assertFalse(self.api.read_activation_state(self.conn).legacy_policy.config_acknowledged)

    def test_config_unknown_and_concurrent_legacy_edit_refuse_without_write(self):
        self.terminal_marker()
        result = self.publish()
        with tempfile.TemporaryDirectory(prefix="synthetic-policy-") as directory:
            config = Path(directory) / "config.json"
            config.write_text('{"tier":"base"}')
            original = config.read_bytes()
            with self.assertRaises(self.projection.ConfigProjectionConflict):
                self.projection.mirror_activation(result, expected_config=None, config_path=config)
            self.assertEqual(config.read_bytes(), original)
            guard = self.projection.capture_config_guard(config_path=config)
            config.write_text('{"tier":"edge"}')
            edited = config.read_bytes()
            with self.assertRaises(self.projection.ConfigProjectionConflict):
                self.projection.mirror_activation(result, expected_config=guard, config_path=config)
            self.assertEqual(config.read_bytes(), edited)
            for field in ("tier", "tier_activation_generation"):
                with self.assertRaises(self.projection.ConfigProjectionConflict):
                    self.projection.patch_config_fields({field: "synthetic"}, config_path=config)

    def test_config_lock_failure_strict_refuses_old_best_effort_remains(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-policy-") as directory:
            config = Path(directory) / "config.json"
            with patch.dict(self.projection.__dict__["__builtins__"], {"open": lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("synthetic lock"))}):
                with self.projection.ConfigFileLock(config.with_suffix(".lock")):
                    pass
                with self.assertRaises(self.projection.ConfigProjectionConflict):
                    self.projection.capture_config_guard(config_path=config)
            self.assertFalse(config.exists())

    def test_config_invalid_or_oversized_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-policy-") as directory:
            config = Path(directory) / "config.json"
            for raw in (b'null', b'{"tier":"base","tier":"pro"}', b'{"tier":"base"}\x00' + b'x' * (1024 * 1024)):
                config.write_bytes(raw)
                with self.subTest(length=len(raw)), self.assertRaises(self.projection.ConfigProjectionConflict):
                    self.projection.capture_config_guard(config_path=config)
                self.assertEqual(config.read_bytes(), raw)


    def test_policy_refuses_temp_alias_trigger_and_stale_config_generation(self):
        intent = self.stage()
        self.conn.execute("CREATE TEMP TABLE METADATA(key TEXT, value TEXT)")
        with self.assertRaises(self.api.TierActivationError):
            self.publish(intent)
        self.conn.execute("DROP TABLE temp.METADATA")
        self.conn.execute("CREATE TRIGGER journal_side_effect AFTER UPDATE ON metadata BEGIN INSERT INTO old_serving_pair VALUES(x'01'); END")
        with self.assertRaises(self.api.TierActivationError):
            self.publish(intent)
        self.conn.execute("DROP TRIGGER journal_side_effect")
        result = self.publish(intent)
        self.api.acknowledge_policy_config(self.conn, generation=result.generation)
        with self.assertRaises(self.api.TierActivationError):
            self.publish(expected_intent_id=result.intent_id, target=self.base,
                         expected_config=self.api.ConfigGuard(True, "pro", "b" * 32))
        self.assertEqual(self.api.read_activation_state(self.conn).legacy_policy.generation, result.generation)

    def test_corrupt_previous_policy_rejected_before_projection(self):
        self.terminal_marker()
        result = self.publish()
        self.api.acknowledge_policy_config(self.conn, generation=result.generation)
        job = self.jobs.select_tier_job(self.conn, self.fixture.target, previous_job_id=self.fixture.job.job_id)
        intent = self.api.stage_activation_intent(self.conn, job, expected_generation=None, reranker_id="synthetic/ranker",
                 expected_config=self.api.ConfigGuard(True, "pro", result.generation), status_id=1)
        for replacement in (None, "b" * 32):
            value = json.loads(self.api._dump(intent))
            value["expected_config"]["generation"] = replacement
            self.conn.execute("UPDATE metadata SET value=? WHERE key='tier_activation_v1'", (json.dumps(value),))
            self.conn.commit()
            with self.assertRaises(self.api.TierActivationError):
                self.api.read_activation_state(self.conn)

    def test_progress_oversized_nul_tail_never_materializes_payload(self):
        _, plan = self.fixture.build()
        key = self.source.manifest_key(plan.manifest.targets)
        self.conn.execute("UPDATE metadata SET value=value||? WHERE key=?", ("\x00" + "x" * 66000, key))
        self.conn.commit()
        statements = []
        self.conn.set_trace_callback(statements.append)
        with self.assertRaises(self.source.TierSourceUntrusted):
            self.source.read_tier_progress(self.conn, model_id=self.fixture.target.model_id, dimension=2, targets=plan.manifest.targets)
        self.conn.set_trace_callback(None)
        self.assertFalse(any(sql.startswith("SELECT value FROM main.metadata") for sql in statements))

    def test_config_independent_patch_and_mirror_share_one_lock(self):
        self.terminal_marker()
        with tempfile.TemporaryDirectory(prefix="synthetic-policy-") as directory:
            config = Path(directory) / "config.json"
            config.write_text('{"tier":"base"}')
            guard = self.projection.capture_config_guard(config_path=config)
            result = self.publish(expected_config=guard)
            barrier = threading.Barrier(2)
            errors = []
            def patcher():
                try:
                    barrier.wait(timeout=2)
                    self.projection.patch_config_fields({"independent": 73}, config_path=config)
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=patcher)
            thread.start()
            barrier.wait(timeout=2)
            self.projection.mirror_activation(result, expected_config=guard, config_path=config)
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(json.loads(config.read_text(encoding="utf-8")), {"tier":"pro", "tier_activation_generation":result.generation, "independent":73})

    def test_config_replace_after_success_error_preserves_pending_reconciliation(self):
        self.terminal_marker()
        with tempfile.TemporaryDirectory(prefix="synthetic-policy-") as directory:
            config = Path(directory) / "config.json"
            config.write_text('{"tier":"base"}')
            guard = self.projection.capture_config_guard(config_path=config)
            result = self.publish(expected_config=guard)
            original = self.projection._replace
            def fail_after(src, dst):
                original(src, dst)
                raise OSError("synthetic after replace")
            with patch.object(self.projection, "_replace", fail_after), self.assertRaises(OSError):
                self.projection.mirror_activation(result, expected_config=guard, config_path=config)
            self.assertFalse(self.api.read_activation_state(self.conn).legacy_policy.config_acknowledged)
            self.assertEqual(self.projection.mirror_activation(result, expected_config=guard, config_path=config).generation, result.generation)



class TestIdentityErrorWire(unittest.TestCase):
    def setUp(self):
        fixtures = runpy.run_path(str(Path(__file__).with_name("test-tier-serving-receipt-795.py")))
        self.fixture = fixtures["TestServingReceipts"]()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def test_identity_error_roundtrip_native_errors_still_generic(self):
        fixture = self.fixture
        target_module = sys.modules.get("truememory.embedding_target")
        self.assertIsNotNone(target_module)
        requests = [({"op": "embed", "expected_target": fixture.edge.to_wire()}, target_module.EmbeddingTargetError("synthetic identity")),
                    ({"op": "embed_target_v1"}, target_module.EmbeddingTargetError("synthetic custom identity")),
                    ({"op": "rerank", "expected_model_name": "synthetic/ranker"}, fixture.ms._ServingIdentityMismatch("synthetic reranker")),
                    ({"op": "embed", "expected_target": fixture.edge.to_wire()}, RuntimeError("synthetic native unavailable")),
                    ({"op": ["malformed"]}, ValueError("synthetic ordinary invalid request"))]
        for request, error in requests:
            replies = []
            server = fixture.server
            server._recv_exact = lambda *args: json.dumps(request).encode()
            server._send_response = lambda conn, response: replies.append(response)
            server.handle_request = lambda request: (_ for _ in ()).throw(error)
            server._serve_client(types.SimpleNamespace(settimeout=lambda _: None), types.SimpleNamespace(length=1, frame_expires_at=None))
            self.assertEqual(len(replies), 1)
            response = replies[0]
            self.assertFalse(response["ok"])
            if isinstance(error, (target_module.EmbeddingTargetError, fixture.ms._ServingIdentityMismatch)):
                self.assertEqual(response["error_code"], "serving_identity_mismatch")
                with self.assertRaises(fixture.client.ServingIdentityMismatchError):
                    fixture.client._check_model_response(response, request)
            else:
                self.assertNotIn("error_code", response)
                with self.assertRaises(RuntimeError):
                    fixture.client._check_model_response(response, request)


if __name__ == "__main__":
    unittest.main()
