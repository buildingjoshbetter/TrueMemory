"""Clustering health with SQLite in memory and synthetic native boundaries."""

import ast
import builtins
import json
import logging
import os
import re
import runpy
import sqlite3
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_functions(path: str, names: set[str], namespace: dict, *, method: bool = False) -> dict:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = next(node.body for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "TrueMemoryEngine") if method else tree.body
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), path, "exec"), namespace)
    return namespace


def isolated_builtins(modules: dict) -> dict:
    runtime = MAINTENANCE._fixture_modules["tier_switch.runtime"]

    def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                    fromlist: tuple = (), level: int = 0) -> object:
        if name in modules:
            return modules[name]
        if name == "truememory.tier_switch.runtime":
            return runtime
        if name.startswith("truememory") or name.split(".")[0] in {"torch", "numpy", "hdbscan", "sqlite_vec"}:
            raise AssertionError("Unexpected production or native import: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)
    return dict(vars(builtins), __import__=safe_import)


def load_primitives() -> tuple[types.ModuleType, types.ModuleType]:
    loader = runpy.run_path(str(Path(__file__).with_name("test-maintenance-source-revision-753.py")))
    return loader["STORAGE"], loader["MAINTENANCE"]


STORAGE, MAINTENANCE = load_primitives()
INSTALLED_DEPENDENCY = MAINTENANCE._installed_dependency


class HealthFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = STORAGE.create_db(":memory:")
        self.addCleanup(self.conn.close)
        self.conn.execute("CREATE TABLE synthetic_vectors(embedding BLOB)")
        self.conn.execute(
            "INSERT INTO vector_cache_registry VALUES "
            "('edge','synthetic_vectors','synthetic_separation',0,0,'synthetic-model',2,0,0)"
        )
        self.conn.commit()
        self.vector = types.SimpleNamespace(
            _lock=threading.Lock(), EMBEDDING_MODEL="synthetic-model", _embedding_dim=2,
            _active_tier_group=lambda: "edge", _active_vec_table=lambda conn: "synthetic_vectors",
            _active_sep_table=lambda conn: "synthetic_separation", resolve_tier=lambda: "edge",
            _frozen_embedding_target=None, _runtime_policy_tier=None,
        )
        self.enter_context(patch.dict(MAINTENANCE._fixture_modules, {"vector_search": self.vector}))
        self.versions = {"numpy": "synthetic", "hdbscan": "synthetic", "sqlite_vec": "synthetic"}
        self.absent: set[str] = set()
        self.probes = []

        def find_spec(name: str) -> object:
            self.probes.append(name)
            return None if name in self.absent else object()

        self.enter_context(patch.dict(sys.modules, {"truememory.vector_search": self.vector}))
        self.enter_context(patch.object(MAINTENANCE, "_installed_dependency", lambda name, dist: self.versions[name]))
        self.enter_context(patch.object(MAINTENANCE, "importlib", types.SimpleNamespace(
            util=types.SimpleNamespace(find_spec=find_spec),
            import_module=Mock(side_effect=AssertionError("Health imported a module")),
        )))
        self.coordinator = MAINTENANCE.MaintenanceCoordinator(None)
        self.coordinator.refresh_capabilities()
        self.namespace = load_functions("truememory/engine.py", {"get_clustering_health"}, {
            "__builtins__": isolated_builtins({"truememory.maintenance": MAINTENANCE}),
        }, method=True)
        self.engine = types.SimpleNamespace(
            conn=self.conn, _write_lock=threading.Lock(), _maintenance_coordinator=self.coordinator,
            _maintenance_handle=self.conn,
            _ensure_connection=Mock(side_effect=AssertionError("Health opened database")),
            _get_maintenance_coordinator=Mock(side_effect=AssertionError("Health initialized scheduler")),
        )
        self.engine.get_clustering_health = types.MethodType(self.namespace["get_clustering_health"], self.engine)

    def enter_context(self, context: object) -> object:
        result = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return result

    def health(self) -> dict:
        return self.engine.get_clustering_health()

    def publish(self, count: int = 0, coverage: str = "complete") -> None:
        dependency = MAINTENANCE._cluster_schedule_dependency(self.conn, versions=self.versions)
        source = MAINTENANCE.read_source_revision(self.conn)
        self.conn.execute("BEGIN")
        MAINTENANCE.record_layer_success_in_transaction(
            self.conn, layer="clusters", dependency=dependency, source=source,
            output_count=count, run_generation="synthetic", coverage=coverage,
        )
        self.conn.commit()


class TestClusteringHealth(HealthFixture):
    def test_package_inspection_is_bounded_and_import_free(self) -> None:
        versions = Mock(return_value="synthetic")
        with patch.object(MAINTENANCE, "_installed_dependency", INSTALLED_DEPENDENCY), \
             patch.object(MAINTENANCE.importlib, "metadata", types.SimpleNamespace(version=versions), create=True):
            self.probes.clear()
            health = MAINTENANCE.clustering_health()
        self.assertEqual(health["state"], "unknown")
        self.assertEqual(len(self.probes), 6)
        self.assertEqual(versions.call_count, 3)
        MAINTENANCE.importlib.import_module.assert_not_called()

    def test_missing_hdbscan_before_any_attempt_is_visible(self) -> None:
        self.absent.add("hdbscan")
        self.versions["hdbscan"] = None
        health = self.health()
        self.assertEqual((health["status"], health["availability"], health["outcome"]),
                         ("degraded", "unavailable", "pending"))
        self.assertEqual(health["missing_dependencies"], ["hdbscan"])
        self.assertIn("truememory[clustering]", health["guidance"])
        self.assertEqual(health["dependency_error"], "DependencyMissing")
        self.assertFalse(self.coordinator.snapshot()["active"])

    def test_metadata_failure_does_not_claim_package_is_missing(self) -> None:
        self.versions["hdbscan"] = None
        health = self.health()
        self.assertEqual(health["availability"], "unavailable")
        self.assertEqual(health["missing_dependencies"], [])
        self.assertIsNone(health["guidance"])

    def test_disconnected_engine_is_unknown_without_opening_or_scheduling(self) -> None:
        self.engine.conn = None
        health = self.health()
        self.assertEqual((health["status"], health["state"], health["availability"]),
                         ("degraded", "unknown", "unknown"))
        self.engine._ensure_connection.assert_not_called()
        self.engine._get_maintenance_coordinator.assert_not_called()
        self.assertFalse(self.coordinator.snapshot()["active"])

    def test_foreground_and_model_busy_remain_distinct_and_do_not_wait(self) -> None:
        self.engine._write_lock.acquire()
        try:
            health = self.health()
        finally:
            self.engine._write_lock.release()
        self.assertEqual((health["state"], health["pending_reason"]), ("deferred", "foreground_busy"))
        self.vector._lock.acquire()
        try:
            health = self.health()
        finally:
            self.vector._lock.release()
        self.assertEqual((health["state"], health["availability"], health["pending_reason"]),
                         ("deferred", "deferred", "model_busy"))
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT outcome FROM maintenance_layers WHERE layer='clusters'").fetchone()[0],
                         "pending")

    def test_pending_and_successful_empty_are_not_missing_dependency(self) -> None:
        pending = self.health()
        self.assertEqual((pending["state"], pending["availability"], pending["freshness"]),
                         ("pending", "available", "untrusted"))
        self.publish()
        empty = self.health()
        self.assertEqual((empty["status"], empty["state"], empty["outcome"], empty["output_count"]),
                         ("ok", "ready", "success_empty", 0))
        self.assertEqual(empty["freshness"], "current")

    def test_unverified_vector_coverage_cannot_claim_full_health(self) -> None:
        self.publish(2, "vector_generation_unverified")
        health = self.health()
        self.assertEqual((health["status"], health["outcome"], health["output_count"]), ("degraded", "success", 2))
        self.assertEqual((health["coverage"], health["pending_reason"]),
                         ("vector_generation_unverified", "coverage_unverified"))
        self.assertIsNone(health["last_error"])

    def test_append_and_correction_preserve_prior_count_but_report_pending(self) -> None:
        self.publish(2)
        self.conn.execute("INSERT INTO messages(id,content) VALUES(1,'synthetic')")
        self.conn.commit()
        health = self.health()
        self.assertEqual((health["state"], health["freshness"], health["output_count"]),
                         ("pending", "append_pending", 2))
        self.conn.execute("UPDATE messages SET content='synthetic changed' WHERE id=1")
        self.conn.commit()
        self.assertEqual(self.health()["freshness"], "correction_pending")

    def test_borrowed_transaction_is_never_committed_or_rolled_back(self) -> None:
        self.publish()
        self.conn.execute("INSERT INTO messages(id,content) VALUES(1,'synthetic pending')")
        queries = []
        self.conn.set_trace_callback(queries.append)
        health = self.health()
        self.conn.set_trace_callback(None)
        self.assertTrue(health["pending_caller_commit"])
        self.assertEqual(health["pending_reason"], "pending_caller_commit")
        self.assertTrue(self.conn.in_transaction)
        self.assertFalse(any(q.strip().upper() in {"COMMIT", "ROLLBACK", "BEGIN"} for q in queries))
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)

    def test_runtime_import_failure_remains_degraded_with_installed_packages(self) -> None:
        self.publish(3)
        self.conn.execute("UPDATE maintenance_layers SET outcome='unavailable',error_category='LayerUnavailableError' "
                          "WHERE layer='clusters'")
        self.conn.commit()
        health = self.health()
        self.assertEqual((health["status"], health["availability"], health["last_error"], health["output_count"]),
                         ("degraded", "available", "LayerUnavailableError", 3))
        self.assertIsNone(health["guidance"])
        self.publish(3)
        self.assertEqual(self.health()["status"], "ok")

    def test_worker_extension_failure_and_dependency_recovery_are_observed(self) -> None:
        dependency = MAINTENANCE._cluster_schedule_dependency(self.conn, versions=self.versions)
        failure = MAINTENANCE.make_layer_dependency(1, MAINTENANCE._dependency_parameters(dependency),
                                                  available=False, error_category="VectorExtensionUnavailable")
        epoch = self.coordinator.capability_snapshot()[0]
        self.coordinator._publish_extension_evidence(epoch, failure)
        self.assertEqual(self.health()["dependency_error"], "VectorExtensionUnavailable")
        self.coordinator.refresh_capabilities()
        self.assertEqual(self.health()["availability"], "available")

    def test_errors_are_categorical_and_inspection_is_read_only(self) -> None:
        self.publish(3)
        self.conn.execute("UPDATE maintenance_layers SET outcome='failed',error_category=? WHERE layer='clusters'",
                          ("synthetic private text\n/path",))
        self.conn.commit()
        queries = []
        self.conn.set_trace_callback(queries.append)
        changes = self.conn.total_changes
        health = self.health()
        self.conn.set_trace_callback(None)
        self.assertEqual(health["last_error"], "UnknownError")
        self.assertNotIn("synthetic private", json.dumps(health))
        self.assertEqual(self.conn.total_changes, changes)
        self.assertFalse(any(q.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "PRAGMA"))
                             for q in queries))
        self.assertFalse(self.conn.in_transaction)
        self.assertFalse(self.coordinator.snapshot()["active"])

    def test_unavailable_tracking_reports_unknown_and_releases_locks(self) -> None:
        with patch.object(MAINTENANCE, "read_layer_states", side_effect=sqlite3.OperationalError("synthetic private")):
            health = self.health()
        self.assertEqual((health["state"], health["last_error"]), ("unknown", "OperationalError"))
        self.assertNotIn("synthetic private", json.dumps(health))
        self.assertFalse(self.conn.in_transaction)
        self.assertTrue(self.engine._write_lock.acquire(blocking=False))
        self.engine._write_lock.release()

    def test_rollback_failure_is_categorical_and_releases_engine_lock(self) -> None:
        class ReadBoundary:
            def __getattr__(boundary, name: str) -> object:
                return getattr(self.conn, name)

            def rollback(boundary) -> None:
                raise sqlite3.OperationalError("synthetic private rollback")

        self.engine.conn = ReadBoundary()
        health = self.health()
        self.assertEqual((health["state"], health["last_error"], health["pending_reason"]),
                         ("unknown", "OperationalError", "inspection_rollback_failed"))
        self.assertNotIn("synthetic private", json.dumps(health))
        self.assertTrue(self.engine._write_lock.acquire(blocking=False))
        self.engine._write_lock.release()
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()

    def test_missing_package_repair_changes_live_evidence_without_starting_work(self) -> None:
        self.absent.add("hdbscan")
        self.versions["hdbscan"] = None
        self.assertEqual(self.health()["availability"], "unavailable")
        self.absent.clear()
        self.versions["hdbscan"] = "synthetic-repaired"
        self.assertEqual((self.health()["availability"], self.health()["state"]), ("available", "pending"))
        self.assertFalse(self.coordinator.snapshot()["active"])


class TestClusteringWarnings(HealthFixture):
    def test_borrowed_success_and_rollback_preserve_prior_process_failure(self) -> None:
        self.conn.execute("CREATE TABLE synthetic_cluster_output(id INTEGER)")
        self.conn.execute("INSERT INTO synthetic_cluster_output VALUES(1)")
        self.conn.commit()
        self.publish(1)
        personality = types.SimpleNamespace(extract_preferences=lambda conn: {})
        for count in (0, 2):
            with self.subTest(count=count):
                with patch.object(MAINTENANCE.logger, "warning"):
                    self.coordinator.observe_clustering_outcome("failed", "SyntheticBulkFailure")
                dependency = MAINTENANCE._cluster_schedule_dependency(self.conn, versions=self.versions)

                def build(conn: sqlite3.Connection) -> None:
                    conn.execute("DELETE FROM synthetic_cluster_output")
                    conn.executemany("INSERT INTO synthetic_cluster_output VALUES(?)", [(i,) for i in range(count)])

                spec = MAINTENANCE.LayerSpec(
                    "clusters", "cluster_messages", lambda: dependency, build,
                    lambda conn: conn.execute("SELECT count(*) FROM synthetic_cluster_output").fetchone()[0],
                )
                self.conn.execute("BEGIN")
                try:
                    with patch.object(MAINTENANCE, "engine_layer_specs", return_value=(spec,)), \
                         patch.object(MAINTENANCE.importlib, "import_module", return_value=personality, side_effect=None):
                        report = MAINTENANCE.run_engine_maintenance(
                            self.conn, self.coordinator, force=True, prepare_extensions=False, allow_caller_transaction=True)
                    self.assertEqual(report.results[0].output_count, count)
                    self.assertTrue(report.results[0].pending_caller_commit)
                    self.assertEqual(self.coordinator.clustering_failure(), ("failed", "SyntheticBulkFailure"))
                finally:
                    self.conn.rollback()
                self.assertEqual(self.health()["process_error"], "SyntheticBulkFailure")
                self.assertEqual(self.conn.execute("SELECT count(*) FROM synthetic_cluster_output").fetchone()[0], 1)
                self.assertEqual(self.health()["output_count"], 1)
        with patch.object(MAINTENANCE, "engine_layer_specs", return_value=(spec,)), \
             patch.object(MAINTENANCE.importlib, "import_module", return_value=personality, side_effect=None):
            report = MAINTENANCE.run_engine_maintenance(
                self.conn, self.coordinator, force=True, prepare_extensions=False)
        self.assertFalse(report.results[0].pending_caller_commit)
        self.assertIsNone(self.coordinator.clustering_failure())

    def test_bulk_ingest_failure_is_visible_and_recovery_clears_process_error(self) -> None:
        namespace = {
            "__builtins__": isolated_builtins({}), "Path": Path, "time": time, "os": os,
            "logger": logging.getLogger("synthetic_ingest_health"),
            "create_db": lambda path: self.conn, "load_messages_from_file": lambda conn, path: 0,
            "init_vec_table": lambda conn: None, "build_vectors": lambda conn: 0,
            "build_separation_vectors": lambda conn: 0,
            "cluster_messages": Mock(side_effect=ImportError("synthetic private sentinel")),
        }
        for flag in ("VECTOR", "HYBRID", "CLUSTERING", "PERSONALITY", "STYLE_VEC", "CONSOLIDATION",
                     "PREDICTIVE", "TEMPORAL"):
            namespace["_HAS_" + flag] = flag in {"VECTOR", "CLUSTERING"}
        load_functions("truememory/engine.py", {"ingest"}, namespace, method=True)
        self.engine.db_path = Path(":memory:")
        self.engine.stats = {}
        for flag in ("hybrid", "temporal", "salience", "personality", "style_vec", "consolidation",
                     "predictive", "reranker", "hyde"):
            setattr(self.engine, "_has_" + flag, False)
        self.engine._get_maintenance_coordinator = lambda: self.coordinator
        with self.assertLogs(MAINTENANCE.logger, level=logging.WARNING) as captured:
            for _ in range(2):
                result = namespace["ingest"](self.engine, "synthetic-unused-input")
        self.assertEqual(len(captured.records), 1)
        self.assertNotIn("synthetic private sentinel", " ".join(captured.output))
        self.assertIn("ERROR", result["build_clusters"])
        self.assertEqual(self.health()["process_error"], "ImportError")
        self.assertEqual(self.health()["status"], "degraded")
        namespace["cluster_messages"] = lambda conn: 0
        namespace["ingest"](self.engine, "synthetic-unused-input")
        self.assertIsNone(self.health()["process_error"])

    def test_warnings_are_sanitized_deduplicated_and_clear_after_success(self) -> None:
        self.absent.add("hdbscan")
        with self.assertLogs(MAINTENANCE.logger, level=logging.WARNING) as captured:
            for _ in range(3):
                self.coordinator.observe_clustering_outcome("unavailable", "synthetic private\n/path")
            self.coordinator.observe_clustering_outcome("success_empty")
            self.coordinator.observe_clustering_outcome("unavailable", "DependencyMissing")
        self.assertEqual(len(captured.records), 2)
        self.assertIn("UnknownError", captured.output[0])
        self.assertNotIn("synthetic private", " ".join(captured.output))
        self.assertTrue(all("truememory[clustering]" in line for line in captured.output))

    def test_runtime_failure_has_no_false_install_guidance_and_reads_never_warn(self) -> None:
        with self.assertLogs(MAINTENANCE.logger, level=logging.WARNING) as captured:
            self.coordinator.observe_clustering_outcome("failed", "RuntimeError")
        self.assertNotIn("pip install", captured.output[0])
        with patch.object(MAINTENANCE.logger, "warning") as warning:
            for _ in range(3):
                self.health()
                self.coordinator.observe_clustering_outcome("deferred", "ModelBusy")
            warning.assert_not_called()

    def test_runner_reports_only_actual_cluster_attempts(self) -> None:
        result = MAINTENANCE.LayerResult("clusters", "cluster_messages", "unavailable", None, 0,
                                         "LayerUnavailableError", True)
        personality = types.SimpleNamespace(extract_preferences=lambda conn: {})
        with patch.object(MAINTENANCE, "engine_layer_specs", return_value=()), \
             patch.object(MAINTENANCE, "run_layers", return_value=(result,)), \
             patch.object(MAINTENANCE.importlib, "import_module", return_value=personality, side_effect=None), \
             patch.object(self.coordinator, "observe_clustering_outcome") as observed:
            MAINTENANCE.run_engine_maintenance(self.conn, self.coordinator, prepare_extensions=False)
            observed.assert_called_once_with("unavailable", "LayerUnavailableError")
            observed.reset_mock()
            with patch.object(MAINTENANCE, "run_layers", return_value=(result._replace(attempted=False),)):
                MAINTENANCE.run_engine_maintenance(self.conn, self.coordinator, prepare_extensions=False)
            observed.assert_not_called()


class TestMCPClusteringHealth(HealthFixture):
    def server(self, memory: object | None) -> dict:
        modules = {
            "truememory.maintenance": MAINTENANCE,
            "truememory.engine": types.SimpleNamespace(get_vectors_load_error=lambda: "SyntheticVectorError"),
            "truememory.tier_switch.manager": types.SimpleNamespace(RebuildManager=types.SimpleNamespace(
                get_instance=lambda: types.SimpleNamespace(get_status=lambda status_id: {"status_id": status_id}))),
        }
        return load_functions("truememory/mcp_server.py", {"_build_health_payload", "truememory_status", "truememory_stats"}, {
            "__builtins__": isolated_builtins(modules), "json": json, "re": re, "_memory": memory,
            "_get_memory": Mock(side_effect=AssertionError("Health constructed Memory")),
            "_reranker_error_lock": threading.Lock(), "_reranker_last_error": "SyntheticRerankerError",
            "_llm_error_lock": threading.Lock(), "_llm_last_error": {"synthetic": "SyntheticLLMError"},
            "_current_llm_provider_name": "synthetic", "_encoding_gate_error_lock": threading.Lock(),
            "_encoding_gate_last_error": "SyntheticGateError", "_encoding_gate_degradation_count": 2,
            "_build_model_server_health": lambda: {"status": "ok", "running": True},
            "_load_config": lambda: {"tier": "edge"}, "__version__": "synthetic",
            "_get_rss_mb": lambda: 0, "_MAX_RSS_MB": 0,
        })

    def test_existing_health_contract_and_values_are_preserved(self) -> None:
        server = self.server(types.SimpleNamespace(_engine=self.engine))
        health = server["_build_health_payload"]()
        self.assertEqual(health["reranker"], {"status": "degraded", "last_error": "SyntheticRerankerError"})
        self.assertEqual(health["hyde_llm"], {"status": "degraded", "active_provider": "synthetic",
                                            "last_error_by_provider": {"synthetic": "SyntheticLLMError"}})
        self.assertEqual(health["vectors"], {"status": "degraded", "last_error": "SyntheticVectorError"})
        self.assertEqual(health["encoding_gate"], {"status": "degraded", "last_error": "SyntheticGateError",
                                                 "degradation_count": 2})
        self.assertEqual(health["model_server"], {"status": "ok", "running": True})
        self.assertTrue(all(item["status"] in {"ok", "degraded"} for item in health.values()))
        self.assertIn("clustering", health)
        self.assertEqual(health["clustering"]["state"], "pending")

    def test_status_before_memory_initialization_reports_missing_dependency_without_initialization(self) -> None:
        self.absent.add("hdbscan")
        self.versions["hdbscan"] = None
        server = self.server(None)
        result = json.loads(server["truememory_status"](17))
        self.assertEqual(result["rebuild"], {"status_id": 17})
        self.assertIn("clustering", result["degradation"])
        self.assertEqual(result["degradation"]["clustering"]["availability"], "unavailable")
        self.assertEqual(result["degradation"]["clustering"]["missing_dependencies"], ["hdbscan"])
        server["_get_memory"].assert_not_called()

    def test_stats_tool_includes_the_same_clustering_health(self) -> None:
        self.engine._ensure_connection = Mock()
        memory = types.SimpleNamespace(_engine=self.engine, stats=lambda: {"message_count": 0})
        server = self.server(memory)
        server["_get_memory"] = lambda: memory
        result = json.loads(server["truememory_stats"]())
        self.assertEqual(result["health"]["clustering"], self.health())
        self.assertEqual(result["message_count"], 0)
        self.engine._ensure_connection.assert_called_once_with()

    def test_clustering_inspection_failure_cannot_hide_existing_subsystems(self) -> None:
        server = self.server(types.SimpleNamespace(_engine=self.engine))
        expected = server["_build_health_payload"]()
        expected.pop("clustering")
        self.engine.get_clustering_health = Mock(side_effect=RuntimeError("synthetic private inspection"))
        result = server["_build_health_payload"]()
        clustering = result.pop("clustering")
        self.assertEqual(result, expected)
        self.assertEqual((clustering["status"], clustering["state"], clustering["last_error"]),
                         ("degraded", "unknown", "RuntimeError"))
        self.assertNotIn("synthetic private", json.dumps(clustering))
        result = json.loads(server["truememory_status"]())
        self.assertEqual(result["degradation"]["clustering"], clustering)


if __name__ == "__main__":
    unittest.main()
