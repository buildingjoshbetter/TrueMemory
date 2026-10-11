"""Truthful scheduled-preference status with stdlib and in-memory SQLite only.

Actual maintenance/serving ownership, layer transactions, status formatting and
ingest admission execute. Synthetic layer builders replace unrelated algorithms;
the personality control loads actual FTS and char-gram retrieval without models.
"""
from __future__ import annotations

import ast
import logging
import runpy
import sqlite3
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PRIMITIVES = runpy.run_path(str(Path(__file__).with_name("test-maintenance-source-revision-753.py")))
STORAGE, MAINTENANCE = PRIMITIVES["STORAGE"], PRIMITIVES["MAINTENANCE"]
UNAVAILABLE = "UNAVAILABLE (ScheduledPreferencesUnsupported)"
LAYERS = (
    ("clusters", "cluster_messages"), ("summaries", "build_summaries"),
    ("contradictions", "detect_contradictions"), ("structured_facts", "structured_facts"),
    ("surprise", "build_surprise_index"), ("episodes", "detect_episodes"),
    ("landmarks", "detect_landmarks"), ("dunbar", "dunbar_hierarchy"),
)


def source_function(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


class TestScheduledPreferences(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = STORAGE.create_db(":memory:")
        self.addCleanup(self.conn.close)
        self.conn.execute("CREATE TABLE synthetic_outputs(layer TEXT PRIMARY KEY)")
        self.conn.commit()
        self.coordinator = MAINTENANCE.MaintenanceCoordinator(None)
        self.builds: list[str] = []
        self.imports: list[str] = []

    def add(self, *, commit: bool = True) -> None:
        self.conn.execute(
            "INSERT INTO messages(content,sender,timestamp) VALUES "
            "('Synthetic coffee routine','synthetic-entity','2026-01-01')"
        )
        if commit:
            self.conn.commit()

    def specs(self) -> tuple:
        def spec(layer: str, key: str) -> object:
            def build(conn: sqlite3.Connection) -> None:
                self.assertIs(conn, self.conn)
                self.assertTrue(conn.in_transaction)
                self.builds.append(layer)
                conn.execute("INSERT OR REPLACE INTO synthetic_outputs VALUES (?)", (layer,))

            return MAINTENANCE.LayerSpec(
                layer, key, lambda: MAINTENANCE.make_layer_dependency(1, {"synthetic": layer}),
                build, lambda conn: conn.execute(
                    "SELECT count(*) FROM synthetic_outputs WHERE layer=?", (layer,),
                ).fetchone()[0],
            )
        return tuple(spec(layer, key) for layer, key in LAYERS)

    def forbidden_import(self, name: str) -> object:
        self.imports.append(name)
        raise AssertionError("The synthetic runner must not import preference/model builders")

    def run_report(self, **kwargs: object) -> object:
        with patch.object(MAINTENANCE, "engine_layer_specs", lambda *_args, **_kwargs: self.specs()), \
             patch.object(MAINTENANCE, "importlib", types.SimpleNamespace(import_module=self.forbidden_import)):
            return MAINTENANCE.run_engine_maintenance(
                self.conn, self.coordinator, prepare_extensions=False, **kwargs,
            )

    def result(self, outcome: str = "success", **kwargs: object) -> object:
        return MAINTENANCE.LayerResult(
            "summaries", "build_summaries", outcome, 1, 0.0,
            kwargs.get("error_category"), kwargs.get("attempted", True),
            kwargs.get("coverage", "complete"), kwargs.get("pending_caller_commit", False),
        )

    def test_forced_empty_and_populated_runs_keep_eight_layers_and_report_limit(self) -> None:
        self.assertEqual(MAINTENANCE.SCHEDULED_PREFERENCES_UNAVAILABLE, UNAVAILABLE)
        for populated in (False, True):
            with self.subTest(populated=populated):
                if populated:
                    self.add()
                self.builds.clear()
                report = self.run_report(force=True)
                self.assertEqual(self.builds, [layer for layer, _ in LAYERS])
                self.assertTrue(all(result.outcome == "success" for result in report.results))
                self.assertEqual(report.preferences, UNAVAILABLE)
                self.assertEqual(MAINTENANCE.maintenance_report_status(report), ("completed_with_limits", None))
                formatted = MAINTENANCE.format_maintenance_report(report)
                self.assertEqual(set(formatted), {key for _, key in LAYERS} | {"extract_preferences"})
                self.assertEqual(formatted["extract_preferences"], UNAVAILABLE)
                self.assertFalse(self.conn.in_transaction)
                self.assertEqual(self.imports, [])

    def test_eligible_automatic_attempt_and_current_no_attempt_differ(self) -> None:
        self.add()
        attempted = self.run_report(threshold=1)
        self.assertTrue(all(result.attempted for result in attempted.results))
        self.assertEqual(attempted.preferences, UNAVAILABLE)
        self.builds.clear()
        current = self.run_report(threshold=1)
        self.assertEqual(self.builds, [])
        self.assertTrue(all(not result.attempted for result in current.results))
        self.assertEqual(current.preferences, "SKIPPED (no maintenance attempt)")
        self.assertEqual(MAINTENANCE.maintenance_report_status(current), ("success", None))
        self.assertEqual(self.imports, [])

    def test_borrowed_writes_still_rollback_and_keep_pending_precedence(self) -> None:
        self.add(commit=False)
        report = self.run_report(force=True, allow_caller_transaction=True)
        self.assertTrue(self.conn.in_transaction)
        self.assertTrue(all(result.pending_caller_commit for result in report.results))
        self.assertEqual(report.preferences, UNAVAILABLE)
        self.assertEqual(MAINTENANCE.maintenance_report_status(report), ("pending", None))
        self.assertEqual(self.conn.execute("SELECT count(*) FROM synthetic_outputs").fetchone()[0], 8)
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT count(*) FROM synthetic_outputs").fetchone()[0], 0)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)

    def test_captured_cancellation_survives_cleared_event(self) -> None:
        for include_style in (False, True):
            with self.subTest(include_style=include_style):
                cancel = threading.Event()
                cancel.set()
                report = self.run_report(force=True, cancel=cancel, include_style=include_style)
                cancel.clear()
                self.assertEqual(report.preferences, "CANCELLED")
                self.assertEqual(MAINTENANCE.maintenance_report_status(report, cancel), ("cancelled", "Cancelled"))
                self.assertEqual(MAINTENANCE.maintenance_report_status(report), ("cancelled", "Cancelled"))
                self.assertEqual(self.builds, [])
                self.assertEqual(self.imports, [])

    def test_real_layer_and_style_outcomes_outrank_known_limitation(self) -> None:
        for outcome in ("failed", "unavailable", "deferred", "abandoned"):
            for style in (False, True):
                with self.subTest(outcome=outcome, style=style):
                    result = self.result(outcome, error_category="SyntheticFailure")
                    report = MAINTENANCE.MaintenanceReport(
                        () if style else (result,), UNAVAILABLE, result if style else None,
                    )
                    self.assertEqual(MAINTENANCE.maintenance_report_status(report), (outcome, "SyntheticFailure"))
                    cancel = threading.Event()
                    cancel.set()
                    self.assertEqual(MAINTENANCE.maintenance_report_status(report, cancel), ("cancelled", "Cancelled"))

    def test_pending_layer_and_style_results_outrank_known_limitation(self) -> None:
        for result in (self.result(pending_caller_commit=True), self.result("stale"), self.result("untrusted")):
            for style in (False, True):
                with self.subTest(result=result, style=style):
                    report = MAINTENANCE.MaintenanceReport(
                        () if style else (result,), UNAVAILABLE, result if style else None,
                    )
                    self.assertEqual(MAINTENANCE.maintenance_report_status(report), ("pending", None))

    def test_only_exact_unsupported_token_is_a_completion_limit(self) -> None:
        for status in ("ERROR (SyntheticFailure)", "UNAVAILABLE (DependencyMissing)",
                       "UNAVAILABLE (ScheduledPreferencesUnsupportedExtra)", UNAVAILABLE + " extra",
                       UNAVAILABLE + " (pending caller commit)"):
            with self.subTest(status=status):
                report = MAINTENANCE.MaintenanceReport((self.result(),), status)
                self.assertEqual(MAINTENANCE.maintenance_report_status(report), ("failed", "PreferencesUnavailable"))
        report = MAINTENANCE.MaintenanceReport((self.result(),), UNAVAILABLE)
        self.assertEqual(MAINTENANCE.maintenance_report_status(report), ("completed_with_limits", None))

    def test_style_only_and_busy_contracts_remain_distinct(self) -> None:
        style = self.result()._replace(layer="style_vectors", result_key="style_vectors")
        with patch.object(MAINTENANCE, "run_routed_style", return_value=style) as builder:
            report = self.run_report(force=True, include_layers=False, include_style=True)
        self.assertIs(report.style_result, style)
        builder.assert_called_once()
        self.assertIs(builder.call_args.args[0], self.conn)
        self.assertEqual(report.preferences, "SKIPPED (style-only)")
        self.assertEqual(MAINTENANCE.maintenance_report_status(report), ("success", None))
        self.assertEqual(self.builds, [])
        busy = MAINTENANCE.maintenance_busy_result()
        self.assertEqual(set(busy), {key for _, key in LAYERS} | {"extract_preferences"})
        self.assertTrue(all(value == "BUSY (maintenance owner active)" for value in busy.values()))

    def test_preference_phase_uses_no_connection_or_helper(self) -> None:
        function = source_function("truememory/maintenance.py", "run_engine_maintenance")
        body = function.body[-1].body
        start = next(index for index, node in enumerate(body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "preferences" for target in node.targets))
        wrapper = ast.parse("def phase(conn, results, force, cancel, style_result):\n    pass\n").body[0]
        wrapper.body = body[start:]
        forbidden: list[str] = []

        class NoConnectionWork:
            def __getattribute__(self, name: str) -> object:
                forbidden.append(name)
                raise AssertionError("Unsupported preference phase cannot use its connection")

        namespace = dict(MAINTENANCE.__dict__)
        namespace["importlib"] = types.SimpleNamespace(import_module=self.forbidden_import)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                     "synthetic-preference-phase", "exec"), namespace)
        for force, attempted in ((True, False), (False, True), (False, False)):
            with self.subTest(force=force, attempted=attempted):
                result = namespace["phase"](NoConnectionWork(), (self.result(attempted=attempted),), force, None, None)
                self.assertEqual(result.preferences, UNAVAILABLE if force or attempted else "SKIPPED (no maintenance attempt)")
        self.assertEqual(forbidden, [])
        self.assertEqual(self.imports, [])

    def test_actual_ingest_reports_present_and_absent_personality_truthfully(self) -> None:
        function = source_function("truememory/engine.py", "ingest")
        calls: list[str] = []

        def create(path: Path) -> sqlite3.Connection:
            self.assertEqual(str(path), ":memory:")
            conn = STORAGE.create_db(":memory:")
            self.addCleanup(conn.close)
            return conn

        def forbidden_preferences(*_args: object, **_kwargs: object) -> None:
            calls.append("preferences")
            raise AssertionError("Ingest must not call no-entity preference extraction")

        def load(conn: sqlite3.Connection, _path: Path) -> int:
            conn.execute("INSERT INTO messages(content,sender,timestamp) VALUES ('Synthetic coffee','synthetic','2026-01-01')")
            conn.commit()
            return 1

        for available in (True, False):
            with self.subTest(personality_available=available):
                namespace = {
                    "__builtins__": STORAGE.__dict__["__builtins__"], "Path": Path, "time": time,
                    "os": types.SimpleNamespace(environ={}), "logger": logging.getLogger("synthetic-preference-ingest"),
                    "create_db": create, "load_messages_from_file": load,
                    "build_entity_profiles": lambda _conn: calls.append("profiles"),
                    "build_dunbar_hierarchy": lambda _conn, **_kwargs: {},
                    "extract_preferences": forbidden_preferences,
                }
                namespace.update({node.id: False for node in ast.walk(function)
                                  if isinstance(node, ast.Name) and node.id.startswith("_HAS_")})
                namespace["_HAS_PERSONALITY"] = available
                exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                             "synthetic-actual-ingest", "exec"), namespace)
                flags = {name: False for name in ("vectors", "hybrid", "temporal", "salience", "personality",
                                                "style_vec", "consolidation", "predictive", "reranker", "hyde", "clustering")}
                engine = types.SimpleNamespace(db_path=Path(":memory:"), conn=None, stats={},
                    _get_maintenance_coordinator=lambda: self.coordinator,
                    **{"_has_" + key: value for key, value in flags.items()})
                calls.clear()
                result = namespace["ingest"](engine, Path("synthetic-unused-input.json"))
                expected = UNAVAILABLE if available else "SKIPPED (personality module not available)"
                self.assertEqual(result["extract_preferences"], expected)
                self.assertEqual(calls, ["profiles"] if available else [])
                self.assertEqual(engine.conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)
                self.assertTrue(engine.ready)

    def test_entity_helper_and_actual_query_time_retrieval_remain_available(self) -> None:
        modules = MAINTENANCE._fixture_modules.copy()
        builtins_map = STORAGE.__dict__["__builtins__"]
        base_import = builtins_map["__import__"]

        def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                        fromlist: tuple = (), level: int = 0) -> object:
            if name.startswith("truememory.") and name.removeprefix("truememory.") in modules:
                return modules[name.removeprefix("truememory.")]
            return base_import(name, globals, locals, fromlist, level)

        for name in ("fts_search", "personality_style_vec", "personality"):
            module = types.ModuleType("synthetic_preference_" + name)
            module.__dict__["__builtins__"] = dict(builtins_map, __import__=safe_import)
            path = ROOT / "truememory" / (name + ".py")
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
            modules[name] = module
        api = modules["personality"]
        self.conn.executemany("INSERT INTO messages(content,sender,timestamp,directive) VALUES (?,?,?,?)", [
            ("Synthetic Cedar likes coffee every morning", "Cedar", "2026-01-01", 0),
            ("Synthetic Birch likes pizza for dinner", "Birch", "2026-01-02", 0),
            ("Synthetic Cedar directive says sushi", "Cedar", "2026-01-03", 1),
        ])
        self.conn.commit()
        statements: list[str] = []
        changes = self.conn.total_changes
        self.conn.set_trace_callback(statements.append)
        self.assertEqual(api.extract_preferences(self.conn), {})
        self.assertEqual(statements, [])
        cedar = api.extract_preferences(self.conn, "Cedar")
        birch = api.extract_preferences(self.conn, "Birch")
        self.assertTrue(any("coffee" in item["text"] for item in cedar["food"]))
        self.assertTrue(any("pizza" in item["text"] for item in birch["food"]))
        self.assertFalse(any("sushi" in item["text"] for item in cedar["food"]))
        results = api.search_personality(self.conn, "What food does Cedar like?", limit=10)
        self.assertTrue(any("coffee" in item["content"] for item in results))
        self.assertFalse(any("sushi" in item["content"] for item in results))
        self.assertTrue(any("MATCH" in statement.upper() for statement in statements))
        self.assertEqual(self.conn.total_changes, changes)
        self.assertFalse(self.conn.in_transaction)
        self.conn.set_trace_callback(None)


if __name__ == "__main__":
    unittest.main()
