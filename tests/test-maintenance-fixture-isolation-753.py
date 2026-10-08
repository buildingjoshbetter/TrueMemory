"""Cross-fixture worker isolation using actual fixtures and SQLite in memory."""

import ast
import builtins
import sqlite3
import sys
import threading
import types
import unittest
import uuid
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def fixture_loader(process_sys: object) -> Callable[[str], dict]:
    """Keep fixture imports and module registries inside one synthetic process."""
    loaded = {}

    def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                    fromlist: tuple = (), level: int = 0) -> object:
        if name == "sys":
            return process_sys
        if name == "runpy":
            return types.SimpleNamespace(run_path=load)
        if name == "truememory.personality_style_vec":
            return process_sys.modules[name]
        if name.startswith("truememory") or name.split(".")[0] in {"torch", "numpy", "hdbscan", "sqlite_vec"}:
            raise AssertionError("Unexpected application/native import: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def safe_exec(code: object, namespace: dict) -> None:
        namespace.setdefault("__builtins__", fixture_builtins)
        exec(code, namespace)

    fixture_builtins = dict(vars(builtins), __import__=safe_import, exec=safe_exec)

    def load(path: str) -> dict:
        path = Path(path)
        if path.name in loaded:
            return loaded[path.name]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Existing fixture loaders also read production source. Pin their
        # decoding in this harness on Python 3.10/Windows without editing it.
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "read_text" and not node.args
                    and not any(keyword.arg == "encoding" for keyword in node.keywords)):
                node.keywords.append(ast.keyword(arg="encoding", value=ast.Constant(value="utf-8")))
        namespace = {"__file__": str(path), "__name__": "synthetic_fixture_" + path.stem,
                     "__builtins__": fixture_builtins}
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
        loaded[path.name] = namespace
        return namespace

    return load


class TestFixtureIsolation(unittest.TestCase):
    def setUp(self) -> None:
        values = vars(sys).copy()
        values["modules"] = {}
        self.process_sys = types.SimpleNamespace(**values)
        load = fixture_loader(self.process_sys)
        self.routing = load(str(ROOT / "tests/test-maintenance-engine-routing-753.py"))
        self.adapters = self.routing["ADAPTERS"]
        self.storage = self.routing["STORAGE"]
        self.maintenance = self.routing["MAINTENANCE"]
        _, self.foreign_vector, _ = self.adapters["CLUSTERS"]["load_clustering"]()
        self.process_sys.modules["truememory.vector_search"] = self.foreign_vector
        load_stdlib = self.routing["stdlib_module"]
        style = load_stdlib("personality_style_vec", {})
        self.process_sys.modules["truememory.personality_style_vec"] = style
        self.enter_context(patch.dict(self.routing, {"stdlib_module": lambda name, modules:
            style if name == "personality_style_vec" else load_stdlib(name, modules)}))
        self.registry_before = self.process_sys.modules.copy()
        # Primitive loaders have their own import guard. Bind their sys view
        # explicitly so both modules start in the same synthetic process.
        self.enter_context(patch.object(self.maintenance, "sys", self.process_sys))
        self.foreign = types.ModuleType("synthetic_foreign_maintenance")
        self.foreign.__dict__.update(self.maintenance.__dict__)
        # Functions need their own globals to represent an unrelated worker.
        source = (ROOT / "truememory/maintenance.py").read_text(encoding="utf-8")
        exec(compile(source, "synthetic-foreign-maintenance", "exec"), self.foreign.__dict__)
        self.foreign.sys = self.process_sys

        self.uri = "file:synthetic-fixture-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
        self.connect = sqlite3.connect
        self.keeper = self.connect(self.uri, uri=True, check_same_thread=False)
        self.addCleanup(self.keeper.close)
        self.create = self.storage.create_db
        self.coordinator = self.maintenance.MaintenanceCoordinator(None)
        fixture_test = self

        class MemoryDirectory:
            name = ":memory:"

            def cleanup(self) -> None:
                pass

        class MemoryFixture(self.routing["RoutingFixture"]):
            def open_db(self) -> sqlite3.Connection:
                conn = fixture_test.open_memory()
                self.addCleanup(conn.close)
                return conn

        self.enter_context(patch.dict(self.adapters, {"tempfile": types.SimpleNamespace(
            TemporaryDirectory=lambda **kwargs: MemoryDirectory())}))
        self.enter_context(patch.object(self.storage, "create_db", self.open_memory))
        self.enter_context(patch.object(self.maintenance, "get_coordinator", lambda path: self.coordinator))
        self.fixture = MemoryFixture()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def enter_context(self, context: object) -> object:
        result = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return result

    def open_memory(self, path: object = None) -> sqlite3.Connection:
        sqlite_view = types.SimpleNamespace(**dict(vars(sqlite3), connect=lambda *args, **kwargs:
            self.connect(self.uri, uri=True, check_same_thread=False)))
        with patch.object(self.storage, "sqlite3", sqlite_view):
            return self.create(":memory:")

    def states(self, conn: sqlite3.Connection | None = None) -> dict:
        conn = conn or self.fixture.engine.conn
        specs = self.maintenance.engine_layer_specs(conn, self.coordinator)
        return self.maintenance.read_layer_states(conn, specs)

    def reopen(self) -> object:
        self.fixture.engine.close()
        reopened = self.fixture.new_engine()
        reopened._has_consolidation = True
        return reopened

    def test_fixture_does_not_publish_its_vector_into_process_registry(self) -> None:
        self.assertIs(self.process_sys.modules["truememory.vector_search"], self.foreign_vector)
        self.assertIs(self.maintenance.sys.modules["truememory.vector_search"], self.fixture.vector)
        self.assertIsNot(self.maintenance.sys.modules, self.process_sys.modules)
        self.assertEqual(self.process_sys.modules, self.registry_before)

    def test_interloper_cannot_defer_bootstrap_or_reset_reopen_threshold(self) -> None:
        foreign_conn = self.connect(":memory:", check_same_thread=False)
        self.fixture.conn.backup(foreign_conn)
        entered, release = threading.Event(), threading.Event()
        observed = []

        class BlockedSnapshot:
            def __getattr__(boundary, name: str) -> object:
                return getattr(foreign_conn, name)

            def execute(boundary, sql: str, *args: object) -> sqlite3.Cursor:
                if sql.startswith("SELECT type,name,rootpage,sql"):
                    entered.set()
                    if not release.wait(3):
                        raise AssertionError("Synthetic worker was not released")
                return foreign_conn.execute(sql, *args)

        worker_coordinator = self.foreign.MaintenanceCoordinator(None)

        def work(conn: sqlite3.Connection, cancel: threading.Event) -> object:
            observed.append(self.foreign.sys.modules["truememory.vector_search"])
            self.foreign._cluster_schedule_dependency(conn, versions=dict(self.fixture.versions))
            return self.foreign.MaintenanceReport((), "SKIPPED (synthetic)")

        with patch.object(self.foreign, "create_db", lambda path: BlockedSnapshot()), \
             patch.object(self.foreign, "_release_owner", lambda fd: None):
            worker = threading.Thread(target=worker_coordinator._run, args=(-1, work))
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                result = self.fixture.engine.consolidate()
                states = self.states()
            finally:
                release.set()
                worker.join(3)
                foreign_conn.close()
            self.assertFalse(worker.is_alive())
        self.assertEqual(worker_coordinator.status, ("success", None))
        self.assertNotIn("DEFERRED", result["cluster_messages"])
        self.assertEqual(observed, [self.foreign_vector])
        self.assertEqual(len(states), 8)
        self.assertTrue(all(state.attempted_insert_count == 0 for state in states.values()), states)
        self.assertTrue(all(state.outcome == "success_empty" for state in states.values()), states)
        self.fixture.add(24)
        before = self.maintenance.read_source_revision(self.fixture.conn)
        versions = self.coordinator.capability_snapshot()[1]
        reopened = self.reopen()
        with patch.object(self.coordinator, "request_layers") as request:
            reopened._maybe_auto_consolidate()
            request.assert_not_called()
            self.assertEqual(self.maintenance.read_source_revision(reopened.conn), before)
            self.assertEqual(self.coordinator.capability_snapshot()[1], versions)
            reopened.add("synthetic twenty fifth")
            request.assert_called_once_with(threshold=25)

    def test_intentionally_deferred_bootstrap_still_recovers_at_twenty_four(self) -> None:
        with self.fixture.vector._lock:
            result = self.fixture.engine.consolidate()
        self.assertEqual(result["cluster_messages"], "DEFERRED (ModelBusy)")
        self.assertIsNone(self.states()["clusters"].attempted_insert_count)
        self.fixture.add(24)
        reopened = self.reopen()
        with patch.object(self.coordinator, "request_layers") as request:
            reopened._maybe_auto_consolidate()
            request.assert_called_once_with(threshold=25)
        report = self.maintenance.run_engine_maintenance(
            reopened.conn, self.coordinator, threshold=25, prepare_extensions=False)
        self.assertEqual([item.layer for item in report.results if item.attempted], ["clusters"])
        states = self.states(reopened.conn)
        self.assertEqual(states["clusters"].attempted_insert_count, 24)
        self.assertTrue(all(state.attempted_insert_count == 0 for name, state in states.items() if name != "clusters"))


if __name__ == "__main__":
    unittest.main()
