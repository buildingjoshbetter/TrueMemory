"""Private style maintenance with stdlib source and synthetic in-memory SQLite."""

import ast
import builtins
import json
import runpy
import sqlite3
import sys
import threading
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ACCUMULATOR = runpy.run_path(str(Path(__file__).with_name("test-l0-style-accumulator-747.py")))
STYLE, STORAGE = ACCUMULATOR["STYLE"], ACCUMULATOR["STORAGE"]


def load_maintenance() -> types.ModuleType:
    modules = {"truememory.storage": STORAGE,
               "truememory._platform": ACCUMULATOR["load_stdlib_source"]("_platform")}

    def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                    fromlist: tuple = (), level: int = 0) -> object:
        if name in modules:
            return modules[name]
        if name.split(".", 1)[0] not in sys.stdlib_module_names:
            raise AssertionError("Non-stdlib import forbidden: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load_source(name: str) -> types.ModuleType:
        loaded = types.ModuleType("synthetic_style_boundary_" + name.replace("/", "_"))
        loaded.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
        source = ROOT / "truememory" / (name + ".py")
        missing = object()
        previous = sys.modules.get(loaded.__name__, missing)
        sys.modules[loaded.__name__] = loaded
        try:
            exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), loaded.__dict__)
        finally:
            if previous is missing:
                sys.modules.pop(loaded.__name__, None)
            else:
                sys.modules[loaded.__name__] = previous
        modules["truememory." + name.replace("/", ".")] = loaded
        return loaded

    module = load_source("maintenance")
    modules["psutil"] = types.SimpleNamespace()
    for name in ("rebuild_source", "embedding_target", "tier_config", "tier_switch/cache",
                 "tier_switch/job", "tier_switch/source", "tier_switch/activation", "tier_switch/serving",
                 "tier_switch/writer"):
        load_source(name)
    modules["truememory.tier_switch"] = types.SimpleNamespace(serving=modules["truememory.tier_switch.serving"])
    vector = types.ModuleType("synthetic_style_vector_identity")
    vector.__dict__.update(__builtins__=dict(vars(builtins), __import__=safe_import), sqlite3=sqlite3,
        EMBEDDING_MODEL="model2vec", _embedding_dim=256, _model=None, _lock=threading.Lock(),
        _frozen_embedding_target=None, _runtime_policy_tier=None, resolve_tier=lambda: "edge",
        _cfg_get_model_group=modules["truememory.tier_config"].get_model_group)
    tree = ast.parse((ROOT / "truememory/vector_search.py").read_text(encoding="utf-8"))
    resolvers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_active_tier_group", "_active_vec_table", "_active_sep_table"}]
    exec(compile(ast.Module(body=resolvers, type_ignores=[]), "actual-style-vector-resolvers", "exec"), vector.__dict__)
    modules["truememory.vector_search"] = vector
    modules["truememory"] = types.SimpleNamespace(vector_search=vector,
        reranker=types.SimpleNamespace(_model=None, get_current_reranker_name=lambda: "synthetic/reranker"))
    runtime = load_source("tier_switch/runtime")
    runtime.sys = types.SimpleNamespace(modules=modules)
    module._fixture_modules = modules

    def import_module(name: str) -> types.ModuleType:
        if name != "truememory.personality_style_vec":
            raise AssertionError("Unexpected style dependency: " + name)
        return STYLE

    module.importlib = types.SimpleNamespace(import_module=import_module)
    return module


MAINTENANCE = load_maintenance()


class PublicationFailure:
    """Invalidate cached statements before denying one real COMMIT or RELEASE."""

    def __init__(self, conn: sqlite3.Connection, *, borrowed: bool) -> None:
        self.conn, self.borrowed = conn, borrowed
        self.matches = 0
        self.denied = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.conn, name)

    def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
        match = sql.startswith("RELEASE truememory_layer_") if self.borrowed else sql == "COMMIT"
        if match:
            self.matches += 1
            if self.matches == (1 if self.borrowed else 2):
                self.denied += 1
                denied = ((sqlite3.SQLITE_SAVEPOINT, "RELEASE", sql.split()[1]) if self.borrowed
                          else (sqlite3.SQLITE_TRANSACTION, "COMMIT", None))

                def authorize(action: int, one: str | None, two: str | None, *_rest: object) -> int:
                    return sqlite3.SQLITE_DENY if (action, one, two) == denied else sqlite3.SQLITE_OK

                self.conn.set_authorizer(authorize)
                try:
                    return self.conn.execute(sql, *args)
                finally:
                    self.conn.set_authorizer(ACCUMULATOR["allow_authorized_action"])
        return self.conn.execute(sql, *args)


class LegacySqliteModule:
    """Python 3.10 lacks these primary result-code exports."""

    def __getattr__(self, name: str) -> object:
        if name in {"SQLITE_BUSY", "SQLITE_LOCKED", "SQLITE_INTERRUPT"}:
            raise AttributeError(name)
        return getattr(sqlite3, name)


class StyleMaintenanceTests(unittest.TestCase):
    def connection(self, cache: int = 100, *, uri: str | None = None) -> sqlite3.Connection:
        conn = sqlite3.connect(uri or ":memory:", uri=uri is not None, timeout=0, cached_statements=cache)
        self.addCleanup(conn.close)
        conn.executescript(STORAGE._SCHEMA_SQL)
        STORAGE._initialize_maintenance_tracking(conn)
        conn.execute("CREATE TABLE IF NOT EXISTS sentinel(value TEXT)")
        conn.commit()
        return conn

    def add(self, conn: sqlite3.Connection, text: str = "synthetic source", *, sender: str = "Synthetic",
            directive: int = 0) -> None:
        conn.execute("INSERT INTO messages(sender,content,timestamp,directive) VALUES (?,?,'2026-01-01',?)",
                     (sender, text, directive))
        conn.commit()

    def state(self, conn: sqlite3.Connection) -> object:
        spec = MAINTENANCE.style_layer_spec(conn)
        parameters = json.loads(spec.resolve_dependency().key)["parameters"]
        self.assertEqual(parameters, {"accumulator_version": STYLE._STYLE_ACCUMULATOR_VERSION,
            "hash_version": 2, "dimension": STYLE.DIM, "ngrams": list(STYLE.NGRAM_SIZES),
            "sum_format": STYLE._STYLE_SUM.format})
        return MAINTENANCE.read_layer_states(conn, (spec,))["style_vectors"]

    def output(self, conn: sqlite3.Connection) -> list[tuple]:
        return conn.execute("SELECT * FROM entity_style_vectors ORDER BY entity").fetchall()

    def checkpoint(self, conn: sqlite3.Connection) -> tuple | None:
        return MAINTENANCE._read_layer_row(conn, "style_vectors")

    def test_spec_and_ordinary_schema_do_not_activate_or_scan(self) -> None:
        conn = self.connection()
        source = MAINTENANCE.read_source_revision(conn)
        before = conn.total_changes

        def authorize(action: int, table: str | None, *_rest: object) -> int:
            if action == sqlite3.SQLITE_READ and table in {"messages", "entity_style_vectors"}:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(authorize)
        spec = MAINTENANCE.style_layer_spec(conn)
        state = MAINTENANCE.read_layer_states(conn, (spec,))[spec.layer]
        self.assertEqual(state.successful_coverage, "unverified")
        self.assertEqual(conn.total_changes, before)
        self.assertEqual(len(STORAGE._MAINTENANCE_LAYERS), 8)
        self.assertEqual(len(MAINTENANCE.all_layer_specs(conn)), 8)
        self.assertEqual(len(MAINTENANCE.maintenance_busy_result()), 9)
        self.assertEqual(conn.execute("SELECT count(*) FROM maintenance_layers").fetchone()[0], 8)
        MAINTENANCE.prepare_style_maintenance(conn)
        self.assertEqual(MAINTENANCE.read_source_revision(conn), source)
        self.assertEqual(conn.execute("SELECT count(*) FROM maintenance_layers").fetchone()[0], 9)
        self.assertIsNone(self.state(conn).successful_revision)
        before = conn.total_changes
        MAINTENANCE.prepare_style_maintenance(conn)
        self.assertEqual(conn.total_changes, before)
        conn.set_authorizer(ACCUMULATOR["allow_authorized_action"])

    def test_preparation_is_owned_or_part_of_caller_transaction(self) -> None:
        for borrowed in (False, True):
            conn = self.connection()
            if borrowed:
                conn.execute("INSERT INTO sentinel VALUES ('synthetic pending')")
            MAINTENANCE.prepare_style_maintenance(conn)
            self.assertEqual(conn.in_transaction, borrowed)
            if borrowed:
                conn.rollback()
                self.assertIsNone(self.checkpoint(conn))
                self.assertEqual(conn.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'truememory_style_output_%'").fetchone()[0], 0)
            else:
                self.assertIsNotNone(self.checkpoint(conn))

    def test_cached_preparation_commit_and_release_faults_do_not_enroll(self) -> None:
        for cache in (0, 5, 100):
            for borrowed in (False, True):
                with self.subTest(cache=cache, borrowed=borrowed):
                    conn = self.connection(cache)
                    if borrowed:
                        conn.execute("INSERT INTO sentinel VALUES ('synthetic pending')")
                    sql = "RELEASE SAVEPOINT truememory_style_tracking" if borrowed else "COMMIT"
                    action = ((sqlite3.SQLITE_SAVEPOINT, "RELEASE", "truememory_style_tracking") if borrowed
                              else (sqlite3.SQLITE_TRANSACTION, "COMMIT", None))
                    probe = ACCUMULATOR["StatementFailure"](conn, sql, action)
                    with self.assertRaises(sqlite3.DatabaseError):
                        MAINTENANCE.prepare_style_maintenance(probe)
                    self.assertEqual(probe.armed, [sql])
                    self.assertIsNone(self.checkpoint(conn))
                    self.assertEqual(conn.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'truememory_style_output_%'").fetchone()[0], 0)
                    self.assertEqual(conn.in_transaction, borrowed)
                    self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], int(borrowed))
                    conn.rollback()

    def test_batch_arithmetic_case_zero_directive_and_hash_are_unchanged(self) -> None:
        conn = self.connection()
        for sender, text, directive in (("Synthetic", "abc", 0), ("synthetic", "a", 0),
                                         ("Synthetic", "", 0), ("Other", "ignored", 1), ("", "ignored", 0)):
            self.add(conn, text, sender=sender, directive=directive)
        expected = STYLE.build_entity_style_vectors(conn)
        baseline = conn.execute("SELECT entity,vector,message_count,vector_sum,accumulator_version FROM entity_style_vectors").fetchall()
        token = MAINTENANCE.read_source_revision(conn)
        public_rows = conn.execute("SELECT * FROM maintenance_layers ORDER BY layer").fetchall()
        result = MAINTENANCE.run_style_maintenance(conn)
        self.assertEqual((result.outcome, result.output_count, result.coverage), ("success", 1, "complete"))
        self.assertEqual(conn.execute("SELECT entity,vector,message_count,vector_sum,accumulator_version FROM entity_style_vectors").fetchall(), baseline)
        self.assertEqual(STYLE.get_entity_style_vector(conn, "SYNTHETIC"), expected["synthetic"])
        self.assertEqual(conn.execute("SELECT value FROM metadata WHERE key='style_vec_hash_version'").fetchone(), ("2",))
        self.assertEqual(MAINTENANCE.read_source_revision(conn), token)
        self.assertEqual(conn.execute("SELECT * FROM maintenance_layers WHERE layer!='style_vectors' ORDER BY layer").fetchall(), public_rows)
        self.assertFalse(MAINTENANCE.run_style_maintenance(conn).attempted)
        self.assertEqual(self.state(conn).attempted_insert_count, 5)

    def test_empty_success_retirement_and_source_corrections(self) -> None:
        for statement in ("UPDATE messages SET content='synthetic changed'",
                          "UPDATE messages SET sender='Other'", "UPDATE messages SET directive=1", "DELETE FROM messages"):
            with self.subTest(statement=statement):
                conn = self.connection()
                self.add(conn)
                MAINTENANCE.run_style_maintenance(conn)
                conn.execute(statement)
                conn.commit()
                self.assertEqual(len(MAINTENANCE.plan_layers(conn, (MAINTENANCE.style_layer_spec(conn),))), 1)
                result = MAINTENANCE.run_style_maintenance(conn)
                self.assertIn(result.outcome, {"success", "success_empty"})
                self.assertEqual(self.state(conn).successful_revision, MAINTENANCE.read_source_revision(conn).revision)
        empty = self.connection()
        result = MAINTENANCE.run_style_maintenance(empty)
        self.assertEqual((result.outcome, result.output_count), ("success_empty", 0))
        self.assertEqual(self.state(empty).attempted_insert_count, 0)

    def test_output_changes_invalidate_failed_baseline_and_rollback_atomically(self) -> None:
        for statement in ("UPDATE entity_style_vectors SET message_count=999",
                          "DELETE FROM entity_style_vectors",
                          "INSERT INTO entity_style_vectors(entity,vector) VALUES ('other','[]')"):
            with self.subTest(statement=statement):
                conn = self.connection()
                self.add(conn)
                MAINTENANCE.run_style_maintenance(conn)
                with patch.object(STYLE, "compute_style_vector", side_effect=ValueError("synthetic")):
                    self.assertEqual(MAINTENANCE.run_style_maintenance(conn, force=True).outcome, "failed")
                self.assertEqual(MAINTENANCE.plan_layers(conn, (MAINTENANCE.style_layer_spec(conn),)), ())
                old, token = self.checkpoint(conn), MAINTENANCE.read_source_revision(conn)
                conn.execute(statement)
                self.assertIsNone(self.state(conn).attempted_revision)
                self.assertTrue(self.state(conn).full_rebuild_required)
                self.assertEqual(MAINTENANCE.read_source_revision(conn), token)
                conn.rollback()
                self.assertEqual(self.checkpoint(conn), old)
                conn.execute(statement)
                conn.commit()
                self.assertEqual(len(MAINTENANCE.plan_layers(conn, (MAINTENANCE.style_layer_spec(conn),))), 1)
                self.assertEqual(MAINTENANCE.run_style_maintenance(conn).outcome, "success")

    def test_trigger_repair_is_metadata_only_and_does_not_certify_v1_rows(self) -> None:
        conn = self.connection()
        self.add(conn)
        MAINTENANCE.run_style_maintenance(conn)
        conn.execute("DROP TRIGGER truememory_style_output_update")
        conn.execute("CREATE TRIGGER truememory_style_output_update AFTER UPDATE ON entity_style_vectors BEGIN SELECT 1; END")
        conn.commit()
        output, source = self.output(conn), MAINTENANCE.read_source_revision(conn)

        def authorize(action: int, table: str | None, *_rest: object) -> int:
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and table in {"messages", "entity_style_vectors"} else sqlite3.SQLITE_OK

        conn.set_authorizer(authorize)
        MAINTENANCE.prepare_style_maintenance(conn)
        self.assertIsNone(self.state(conn).attempted_revision)
        self.assertTrue(self.state(conn).full_rebuild_required)
        conn.set_authorizer(ACCUMULATOR["allow_authorized_action"])
        self.assertEqual(self.output(conn), output)
        self.assertEqual(MAINTENANCE.read_source_revision(conn), source)

    def test_future_custom_format_is_retained_and_low_level_still_has_its_old_contract(self) -> None:
        conn = self.connection()
        self.add(conn)
        conn.execute("INSERT INTO entity_style_vectors VALUES ('synthetic','[1]',9,'synthetic old',X'1234',9)")
        conn.execute("INSERT INTO metadata(key,value) VALUES ('style_vec_hash_version','1')")
        conn.commit()
        before = self.output(conn)
        result = MAINTENANCE.run_style_maintenance(conn)
        self.assertEqual((result.outcome, result.error_category), ("unavailable", "StyleAccumulatorUnsupported"))
        self.assertEqual(self.output(conn), before)
        self.assertEqual(conn.execute("SELECT value FROM metadata WHERE key='style_vec_hash_version'").fetchone(), ("1",))
        self.assertIsNone(self.state(conn).successful_revision)
        self.assertFalse(MAINTENANCE.run_style_maintenance(conn).attempted)
        STYLE.build_entity_style_vectors(conn)
        self.assertEqual(conn.execute("SELECT accumulator_version FROM entity_style_vectors").fetchone(), (1,))
        self.assertIsNone(self.state(conn).attempted_revision)

    def test_custom_future_default_is_guarded_even_without_rows(self) -> None:
        for with_source in (False, True):
            with self.subTest(with_source=with_source):
                conn = self.connection()
                conn.execute("DROP TABLE entity_style_vectors")
                conn.execute("CREATE TABLE entity_style_vectors(entity TEXT PRIMARY KEY,vector TEXT,message_count INTEGER,"
                             "updated_at TEXT,vector_sum BLOB,accumulator_version INTEGER NOT NULL DEFAULT 9)")
                conn.commit()
                if with_source:
                    self.add(conn)
                schema = conn.execute("PRAGMA table_info(entity_style_vectors)").fetchall()
                result = MAINTENANCE.run_style_maintenance(conn)
                self.assertEqual((result.outcome, result.error_category), ("unavailable", "StyleAccumulatorUnsupported"))
                self.assertEqual(conn.execute("PRAGMA table_info(entity_style_vectors)").fetchall(), schema)
                self.assertEqual(self.output(conn), [])
                self.assertIsNone(conn.execute("SELECT value FROM metadata WHERE key='style_vec_hash_version'").fetchone())

    def test_missing_output_trigger_cannot_certify_through_an_unprepared_spec(self) -> None:
        conn = self.connection()
        self.add(conn)
        MAINTENANCE.run_style_maintenance(conn)
        before = self.output(conn)
        conn.execute("DROP TRIGGER truememory_style_output_update")
        conn.commit()
        spec = MAINTENANCE.style_layer_spec(conn)
        self.assertFalse(spec.resolve_dependency().available)
        result = MAINTENANCE.run_layers(conn, (spec,), force=True)[0]
        self.assertEqual((result.outcome, result.error_category), ("unavailable", "StyleTrackingUnprepared"))
        self.assertEqual(self.output(conn), before)
        self.assertEqual(MAINTENANCE.run_style_maintenance(conn).outcome, "success")

    def test_borrowed_success_remains_pending_and_outer_rollback_restores_everything(self) -> None:
        conn = self.connection()
        self.add(conn)
        MAINTENANCE.run_style_maintenance(conn)
        before, checkpoint = self.output(conn), self.checkpoint(conn)
        conn.execute("INSERT INTO sentinel VALUES ('synthetic caller')")
        conn.execute("UPDATE messages SET content='synthetic changed'")
        result = MAINTENANCE.run_style_maintenance(conn, allow_caller_transaction=True)
        self.assertTrue(result.pending_caller_commit)
        self.assertEqual(result.outcome, "success")
        self.assertTrue(conn.in_transaction)
        conn.rollback()
        self.assertEqual(self.output(conn), before)
        self.assertEqual(self.checkpoint(conn), checkpoint)
        self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], 0)

    def test_legacy_and_invalid_sums_rebuild_from_source_without_double_counting(self) -> None:
        for version, total in ((0, None), (1, b"synthetic invalid sum")):
            with self.subTest(version=version):
                conn = self.connection()
                self.add(conn, "synthetic first")
                self.add(conn, "synthetic second")
                conn.execute("INSERT INTO entity_style_vectors VALUES ('synthetic','[1]',99,'synthetic old',?,?)",
                             (total, version))
                conn.commit()
                self.assertEqual(STYLE.get_entity_style_vector(conn, "Synthetic"), [1])
                self.assertEqual(MAINTENANCE.run_style_maintenance(conn).outcome, "success")
                self.assertEqual(conn.execute("SELECT message_count,accumulator_version,length(vector_sum) FROM entity_style_vectors").fetchone(), (2, 1, 2048))
                self.assertEqual(self.state(conn).attempted_insert_count, 2)

    def test_cached_commit_and_release_faults_retain_prior_output(self) -> None:
        for cache in (0, 5, 100):
            for borrowed in (False, True):
                with self.subTest(cache=cache, borrowed=borrowed):
                    conn = self.connection(cache)
                    self.add(conn)
                    MAINTENANCE.run_style_maintenance(conn)
                    before, prior = self.output(conn), self.state(conn)
                    conn.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
                    conn.commit()
                    if borrowed:
                        conn.execute("INSERT INTO sentinel VALUES ('synthetic caller')")
                    probe = PublicationFailure(conn, borrowed=borrowed)
                    result = MAINTENANCE.run_style_maintenance(probe, force=True, allow_caller_transaction=borrowed)
                    self.assertEqual((result.outcome, probe.denied), ("failed", 1))
                    self.assertEqual(self.output(conn), before)
                    self.assertEqual(self.state(conn).successful_revision, prior.successful_revision)
                    self.assertEqual(conn.execute("SELECT value FROM metadata WHERE key='style_vec_hash_version'").fetchone(), ("1",))
                    self.assertEqual(conn.in_transaction, borrowed)
                    if borrowed:
                        self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], 1)
                        conn.rollback()

    def test_verified_source_change_defers_without_consuming_attempt(self) -> None:
        conn = self.connection()
        self.add(conn)
        MAINTENANCE.run_style_maintenance(conn)
        before, checkpoint = self.output(conn), self.checkpoint(conn)
        compute = STYLE.compute_style_vector

        def changed(text: str) -> list[float]:
            conn.execute("UPDATE messages SET recipient='synthetic changed'")
            return compute(text)

        with patch.object(STYLE, "compute_style_vector", changed):
            result = MAINTENANCE.run_style_maintenance(conn, force=True)
        self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleSourceChanged"))
        self.assertEqual(self.output(conn), before)
        self.assertEqual(self.checkpoint(conn), checkpoint)
        self.assertFalse(conn.in_transaction)

    def test_cancellation_during_compute_and_after_writer_retains_generation(self) -> None:
        for phase in ("before", "compute", "writer"):
            with self.subTest(phase=phase):
                conn = self.connection()
                self.add(conn)
                MAINTENANCE.run_style_maintenance(conn)
                before, checkpoint = self.output(conn), self.checkpoint(conn)
                cancel = threading.Event()
                compute = STYLE.compute_style_vector

                def cancelled(text: str) -> list[float]:
                    if phase == "compute":
                        cancel.set()
                    return compute(text)

                def trace(sql: str) -> None:
                    if phase == "writer" and sql == "DELETE FROM entity_style_vectors":
                        cancel.set()

                if phase == "before":
                    cancel.set()
                conn.set_trace_callback(trace)
                with patch.object(STYLE, "compute_style_vector", cancelled):
                    result = MAINTENANCE.run_style_maintenance(conn, force=True, cancel=cancel, connection_owned=True)
                conn.set_trace_callback(None)
                self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleCancelled"))
                self.assertEqual(self.output(conn), before)
                self.assertEqual(self.checkpoint(conn), checkpoint)
                self.assertFalse(conn.in_transaction)

    def test_sql_capture_cancellation_disables_handler_before_rollback(self) -> None:
        conn = self.connection()
        self.add(conn)
        MAINTENANCE.run_style_maintenance(conn)
        before, checkpoint = self.output(conn), self.checkpoint(conn)
        cancel = threading.Event()

        class ProgressBoundary:
            def __getattr__(boundary, name: str) -> object:
                return getattr(conn, name)

            def set_progress_handler(boundary, handler: object, steps: int) -> None:
                conn.set_progress_handler(handler, 1 if handler is not None else 0)

            def execute(boundary, sql: str, *args: object) -> sqlite3.Cursor:
                if sql.startswith("SELECT sender, content, timestamp"):
                    cancel.set()
                return conn.execute(sql, *args)

        result = MAINTENANCE.run_style_maintenance(ProgressBoundary(), force=True, cancel=cancel, connection_owned=True)
        self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleCancelled"))
        self.assertFalse(conn.in_transaction)
        self.assertEqual(self.output(conn), before)
        self.assertEqual(self.checkpoint(conn), checkpoint)

    def test_borrowed_progress_handler_is_never_replaced(self) -> None:
        conn = self.connection()
        self.add(conn)
        MAINTENANCE.prepare_style_maintenance(conn)
        conn.execute("INSERT INTO sentinel VALUES ('synthetic caller')")
        calls = []
        conn.set_progress_handler(lambda: calls.append(1) or 0, 1)

        class BorrowedBoundary:
            def __getattr__(boundary, name: str) -> object:
                return getattr(conn, name)

            def set_progress_handler(boundary, *_args: object) -> None:
                raise AssertionError("Borrowed handler must remain untouched")

        result = MAINTENANCE.run_style_maintenance(BorrowedBoundary(), cancel=threading.Event(),
                                                  allow_caller_transaction=True, connection_owned=True)
        self.assertEqual(result.outcome, "success")
        self.assertTrue(calls)
        before = len(calls)
        conn.execute("SELECT count(*) FROM messages").fetchone()
        self.assertGreater(len(calls), before)
        conn.set_progress_handler(None, 0)
        conn.rollback()

    def test_clean_caller_connection_keeps_its_unknown_progress_handler_by_default(self) -> None:
        conn = self.connection()
        self.add(conn)
        calls = []
        conn.set_progress_handler(lambda: calls.append(1) or 0, 1)

        class CallerBoundary:
            def __getattr__(boundary, name: str) -> object:
                return getattr(conn, name)

            def set_progress_handler(boundary, *_args: object) -> None:
                raise AssertionError("Clean transaction does not prove connection ownership")

        result = MAINTENANCE.run_style_maintenance(CallerBoundary(), cancel=threading.Event())
        self.assertEqual(result.outcome, "success")
        before = len(calls)
        conn.execute("SELECT count(*) FROM messages").fetchone()
        self.assertGreater(len(calls), before)
        conn.set_progress_handler(None, 0)

    def test_refused_rollback_never_restores_a_deferred_checkpoint(self) -> None:
        conn = self.connection()
        self.add(conn)
        MAINTENANCE.run_style_maintenance(conn)
        cancel = threading.Event()
        compute = STYLE.compute_style_vector

        def cancelled(text: str) -> list[float]:
            cancel.set()
            return compute(text)

        class RefusedRollback:
            def __getattr__(boundary, name: str) -> object:
                return getattr(conn, name)

            def rollback(boundary) -> None:
                raise sqlite3.OperationalError("synthetic rollback refusal")

        with patch.object(STYLE, "compute_style_vector", cancelled):
            with self.assertRaises(MAINTENANCE._LayerRollbackFailed):
                MAINTENANCE.run_style_maintenance(RefusedRollback(), force=True, cancel=cancel)
        self.assertTrue(conn.in_transaction)
        self.assertEqual(self.state(conn).outcome, "running")
        conn.rollback()
        self.assertEqual(self.state(conn).outcome, "running")

    def test_stale_deferred_restore_cannot_overwrite_new_invalidation(self) -> None:
        conn = self.connection()
        self.add(conn)
        MAINTENANCE.run_style_maintenance(conn)
        with MAINTENANCE.maintenance_owner(None) as owner:
            previous = self.checkpoint(conn)
            conn.execute("UPDATE maintenance_layers SET outcome='running',run_generation=? WHERE layer='style_vectors'", (owner.generation,))
            conn.commit()
            running = self.checkpoint(conn)
            conn.execute("UPDATE entity_style_vectors SET message_count=111")
            conn.commit()
            invalidated = self.checkpoint(conn)
            with self.assertRaises(MAINTENANCE.MaintenanceBusyError):
                MAINTENANCE._restore_deferred_attempt(conn, "style_vectors", previous, running, owner)
            self.assertEqual(self.checkpoint(conn), invalidated)
            self.assertIsNone(self.state(conn).attempted_revision)

    def test_two_connections_observe_only_committed_style_output(self) -> None:
        uri = "file:synthetic-style-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
        first = self.connection(uri=uri)
        second = self.connection(uri=uri)
        self.add(first)
        MAINTENANCE.run_style_maintenance(first)
        self.assertEqual(self.output(first), self.output(second))
        self.assertEqual(self.checkpoint(first), self.checkpoint(second))
        reopened = load_maintenance()
        self.assertFalse(reopened.run_style_maintenance(second).attempted)

    def test_writer_contention_defers_only_after_confirmed_output_rollback(self) -> None:
        uri = "file:synthetic-style-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
        first = self.connection(uri=uri)
        second = self.connection(uri=uri)
        self.add(first)
        MAINTENANCE.run_style_maintenance(first)
        before, checkpoint = self.output(first), self.checkpoint(first)
        compute = STYLE.compute_style_vector
        rollbacks = []

        def contend(text: str) -> list[float]:
            second.execute("BEGIN IMMEDIATE")
            second.execute("INSERT INTO sentinel VALUES ('synthetic competing writer')")
            return compute(text)

        class WriterBoundary:
            def __getattr__(boundary, name: str) -> object:
                return getattr(first, name)

            def rollback(boundary) -> None:
                first.rollback()
                rollbacks.append(not first.in_transaction)
                second.rollback()

        with patch.object(STYLE, "compute_style_vector", contend):
            result = MAINTENANCE.run_style_maintenance(WriterBoundary(), force=True)
        self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleWriterBusy"))
        self.assertEqual(rollbacks, [True])
        self.assertEqual(self.output(first), before)
        self.assertEqual(self.checkpoint(first), checkpoint)

    def test_unexpected_sql_failure_stays_failed_and_modelbusy_default_stays_unchanged(self) -> None:
        conn = self.connection()
        self.add(conn)
        with patch.object(STYLE, "compute_style_vector", side_effect=sqlite3.OperationalError("synthetic unexpected")):
            result = MAINTENANCE.run_style_maintenance(conn)
        self.assertEqual((result.outcome, result.error_category), ("failed", "OperationalError"))
        self.assertEqual(MAINTENANCE.LayerDeferredError("synthetic").category, "ModelBusy")
        with self.assertRaises(ValueError):
            MAINTENANCE.LayerDeferredError("synthetic", category="Unreviewed")

    def test_existing_adapter_modelbusy_result_and_checkpoint_are_preserved(self) -> None:
        conn = self.connection()
        dependency = MAINTENANCE.make_layer_dependency(1)

        def busy(_conn: sqlite3.Connection) -> None:
            raise MAINTENANCE.LayerDeferredError("synthetic model owner")

        spec = MAINTENANCE.LayerSpec("summaries", "build_summaries", lambda: dependency, busy, lambda _conn: 0)
        before = MAINTENANCE._read_layer_row(conn, "summaries")
        restore = MAINTENANCE._restore_deferred_attempt

        def positional_restore(*args: object) -> None:
            self.assertEqual(len(args), 5)
            restore(*args)

        with patch.object(MAINTENANCE, "_restore_deferred_attempt", positional_restore):
            result = MAINTENANCE.run_layers(conn, (spec,))[0]
        self.assertEqual((result.outcome, result.error_category), ("deferred", "ModelBusy"))
        self.assertEqual(MAINTENANCE._read_layer_row(conn, "summaries"), before)

    def test_python310_errors_and_extended_codes_retain_specific_categories(self) -> None:
        cases = (
            ("synthetic unexpected", None, False, "failed", "OperationalError"),
            ("database is locked", None, False, "deferred", "StyleWriterBusy"),
            ("database table is locked: messages", None, False, "deferred", "StyleWriterBusy"),
            ("database table is locked: maintenance_layers", None, False, "deferred", "StyleWriterBusy"),
            ("interrupted", None, True, "deferred", "StyleCancelled"),
            ("synthetic busy snapshot", 517, False, "deferred", "StyleWriterBusy"),
            ("synthetic shared cache", 262, False, "deferred", "StyleWriterBusy"),
            ("database is locked", 10, False, "failed", "OperationalError"),
        )
        for message, code, cancelled, outcome, category in cases:
            with self.subTest(code=code, category=category, message=message):
                conn = self.connection()
                self.add(conn)
                MAINTENANCE.run_style_maintenance(conn)
                before = self.output(conn)
                cancel = threading.Event()
                error = sqlite3.OperationalError(message)
                self.assertFalse(hasattr(error, "sqlite_errorcode"))
                if code is not None:
                    error.sqlite_errorcode = code

                def fail(_text: str) -> list[float]:
                    if cancelled:
                        cancel.set()
                    raise error

                with patch.object(MAINTENANCE, "sqlite3", LegacySqliteModule()), \
                     patch.object(STYLE, "compute_style_vector", fail):
                    result = MAINTENANCE.run_style_maintenance(conn, force=True, cancel=cancel)
                self.assertEqual((result.outcome, result.error_category), (outcome, category))
                self.assertEqual(self.output(conn), before)
                self.assertFalse(conn.in_transaction)

    def test_running_diagnostic_contention_defers_without_recording_an_attempt(self) -> None:
        uri = "file:synthetic-style-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
        first = self.connection(uri=uri)
        second = self.connection(uri=uri)
        self.add(first)
        MAINTENANCE.run_style_maintenance(first)
        before, checkpoint = self.output(first), self.checkpoint(first)
        second.execute("BEGIN IMMEDIATE")
        second.execute("INSERT INTO sentinel VALUES ('synthetic writer stays active')")
        try:
            result = MAINTENANCE.run_style_maintenance(first, force=True)
            self.assertEqual((result.outcome, result.error_category, result.attempted),
                             ("deferred", "StyleWriterBusy", False))
            self.assertTrue(second.in_transaction)
            self.assertFalse(first.in_transaction)
            self.assertEqual(self.output(first), before)
            self.assertEqual(self.checkpoint(first), checkpoint)
        finally:
            second.rollback()
        self.assertEqual(MAINTENANCE.run_style_maintenance(first, force=True).outcome, "success")

    def test_restoration_contention_leaves_running_diagnostic_for_later_recovery(self) -> None:
        uri = "file:synthetic-style-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
        first = self.connection(uri=uri)
        second = self.connection(uri=uri)
        self.add(first)
        MAINTENANCE.run_style_maintenance(first)
        before, checkpoint = self.output(first), self.checkpoint(first)
        compute = STYLE.compute_style_vector

        def contend(text: str) -> list[float]:
            second.execute("BEGIN IMMEDIATE")
            second.execute("INSERT INTO sentinel VALUES ('synthetic writer stays active')")
            return compute(text)

        try:
            with patch.object(STYLE, "compute_style_vector", contend):
                result = MAINTENANCE.run_style_maintenance(first, force=True)
            self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleWriterBusy"))
            self.assertTrue(second.in_transaction)
            self.assertFalse(first.in_transaction)
            self.assertEqual(self.output(first), before)
            self.assertEqual(self.state(first).outcome, "running")
            self.assertNotEqual(self.checkpoint(first), checkpoint)
        finally:
            second.rollback()
        self.assertEqual(MAINTENANCE.run_style_maintenance(first).outcome, "success")
        self.assertEqual(self.state(first).outcome, "success")

    def test_actual_delete_interrupt_confirms_owned_automatic_rollback(self) -> None:
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                conn = self.connection()
                self.add(conn)
                MAINTENANCE.run_style_maintenance(conn)
                before, checkpoint = self.output(conn), self.checkpoint(conn)
                cancel = threading.Event()
                observations = []

                class InterruptBoundary:
                    handler_active = False

                    def __getattr__(boundary, name: str) -> object:
                        return getattr(conn, name)

                    def set_progress_handler(boundary, handler: object, steps: int) -> None:
                        boundary.handler_active = handler is not None
                        conn.set_progress_handler(handler, 1 if handler is not None else 0)

                    def execute(boundary, sql: str, *args: object) -> sqlite3.Cursor:
                        if sql == "DELETE FROM entity_style_vectors":
                            cancel.set()
                        elif cancel.is_set():
                            self.assertFalse(boundary.handler_active)
                        try:
                            return conn.execute(sql, *args)
                        except sqlite3.OperationalError as error:
                            if sql == "DELETE FROM entity_style_vectors":
                                observations.append((str(error), conn.in_transaction))
                                if legacy:
                                    old_error = sqlite3.OperationalError(str(error))
                                    self.assertFalse(hasattr(old_error, "sqlite_errorcode"))
                                    raise old_error from error
                            raise

                with patch.object(MAINTENANCE, "sqlite3", LegacySqliteModule() if legacy else sqlite3):
                    result = MAINTENANCE.run_style_maintenance(InterruptBoundary(), force=True,
                                                               cancel=cancel, connection_owned=True)
                self.assertEqual(observations, [("interrupted", False)])
                self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleCancelled"))
                self.assertFalse(conn.in_transaction)
                self.assertEqual(self.output(conn), before)
                self.assertEqual(self.checkpoint(conn), checkpoint)

    def test_unwitnessed_interrupt_cannot_claim_an_ended_transaction_rolled_back(self) -> None:
        conn = self.connection()
        self.add(conn)
        MAINTENANCE.run_style_maintenance(conn)
        cancel = threading.Event()

        def ended(_text: str) -> list[float]:
            conn.commit()
            cancel.set()
            error = sqlite3.OperationalError("interrupted")
            error.sqlite_errorcode = 9
            raise error

        with patch.object(STYLE, "compute_style_vector", ended):
            with self.assertRaises(MAINTENANCE._LayerRollbackFailed):
                MAINTENANCE.run_style_maintenance(conn, force=True, cancel=cancel, connection_owned=True)
        self.assertEqual(self.state(conn).outcome, "running")

    def test_first_enrollment_and_trigger_repair_contention_preserve_caller(self) -> None:
        for repair in (False, True):
            for borrowed in (False, True):
                with self.subTest(repair=repair, borrowed=borrowed):
                    uri = "file:synthetic-style-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
                    first = self.connection(uri=uri)
                    second = self.connection(uri=uri)
                    self.add(first)
                    if repair:
                        MAINTENANCE.run_style_maintenance(first)
                        first.execute("DROP TRIGGER truememory_style_output_update")
                        first.commit()
                    before, checkpoint = self.output(first), self.checkpoint(first)
                    first.execute("CREATE TEMP TABLE caller_pending(value TEXT)")
                    if borrowed:
                        first.execute("BEGIN")
                        first.execute("INSERT INTO caller_pending VALUES ('synthetic pending')")
                    second.execute("BEGIN IMMEDIATE")
                    second.execute("INSERT INTO sentinel VALUES ('synthetic persistent writer')")
                    try:
                        result = MAINTENANCE.run_style_maintenance(first, allow_caller_transaction=borrowed)
                        self.assertEqual((result.outcome, result.error_category, result.attempted),
                                         ("deferred", "StyleWriterBusy", False))
                        self.assertEqual(result.pending_caller_commit, borrowed)
                        self.assertTrue(second.in_transaction)
                        self.assertEqual(first.in_transaction, borrowed)
                        self.assertEqual(first.execute("SELECT count(*) FROM caller_pending").fetchone()[0], int(borrowed))
                        self.assertEqual(self.output(first), before)
                        self.assertEqual(self.checkpoint(first), checkpoint)
                        self.assertFalse(STORAGE._style_output_tracking_ready(first))
                    finally:
                        first.rollback()
                        second.rollback()
                    self.assertEqual(MAINTENANCE.run_style_maintenance(first).outcome, "success")

    def test_preparation_unknown_errors_or_unconfirmed_transaction_end_do_not_defer(self) -> None:
        for ended in (False, True):
            with self.subTest(ended=ended):
                conn = self.connection()
                self.add(conn)

                class UnsafePreparation:
                    def __getattr__(boundary, name: str) -> object:
                        return getattr(conn, name)

                    def execute(boundary, sql: str, *args: object) -> sqlite3.Cursor:
                        if (ended and sql == "COMMIT") or (not ended and sql == "BEGIN IMMEDIATE"):
                            if ended:
                                conn.commit()
                            raise sqlite3.OperationalError("database is locked" if ended else "synthetic unknown")
                        return conn.execute(sql, *args)

                with self.assertRaises(sqlite3.OperationalError):
                    MAINTENANCE.run_style_maintenance(UnsafePreparation())
                self.assertFalse(conn.in_transaction)
                self.assertEqual(self.checkpoint(conn) is not None, ended)

    def test_source_metadata_read_contention_before_enrollment_defers(self) -> None:
        uri = "file:synthetic-style-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
        first = self.connection(uri=uri)
        second = self.connection(uri=uri)
        self.add(first)
        second.execute("BEGIN IMMEDIATE")
        second.execute("INSERT INTO messages(sender,content) VALUES ('Synthetic','synthetic pending writer')")
        try:
            result = MAINTENANCE.run_style_maintenance(first)
            self.assertEqual((result.outcome, result.error_category, result.attempted),
                             ("deferred", "StyleWriterBusy", False))
            self.assertTrue(second.in_transaction)
            self.assertFalse(first.in_transaction)
            self.assertIsNone(self.checkpoint(first))
        finally:
            second.rollback()
        self.assertEqual(MAINTENANCE.run_style_maintenance(first).outcome, "success")

    def test_failed_preparation_savepoint_rollback_is_not_deferred(self) -> None:
        uri = "file:synthetic-style-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
        first = self.connection(uri=uri)
        second = self.connection(uri=uri)
        self.add(first)
        first.execute("BEGIN")
        second.execute("BEGIN IMMEDIATE")
        second.execute("INSERT INTO sentinel VALUES ('synthetic writer')")

        class RefusedSavepointRollback:
            def __getattr__(boundary, name: str) -> object:
                return getattr(first, name)

            def execute(boundary, sql: str, *args: object) -> sqlite3.Cursor:
                if sql == "ROLLBACK TO SAVEPOINT truememory_style_tracking":
                    raise sqlite3.OperationalError("synthetic rollback refused")
                return first.execute(sql, *args)

        try:
            with self.assertRaisesRegex(sqlite3.OperationalError, "synthetic rollback refused"):
                MAINTENANCE.run_style_maintenance(RefusedSavepointRollback(), allow_caller_transaction=True)
            self.assertTrue(first.in_transaction)
            self.assertTrue(second.in_transaction)
            self.assertIsNone(self.checkpoint(first))
        finally:
            first.rollback()
            second.rollback()

    def test_read_baseline_races_after_preparation_initial_read_and_cancellation_defer(self) -> None:
        for phase in ("initial", "layer", "cancelled"):
            with self.subTest(phase=phase):
                uri = "file:synthetic-style-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
                first = self.connection(uri=uri)
                second = self.connection(uri=uri)
                self.add(first)
                MAINTENANCE.run_style_maintenance(first)
                before, checkpoint = self.output(first), self.checkpoint(first)
                prepare = MAINTENANCE.prepare_style_maintenance
                read = MAINTENANCE.read_layer_states
                reads = []
                cancel = threading.Event()
                if phase == "cancelled":
                    cancel.set()

                def contend() -> None:
                    second.execute("BEGIN IMMEDIATE")
                    second.execute("INSERT INTO messages(sender,content) VALUES ('Synthetic','synthetic pending writer')")

                def prepared(conn: sqlite3.Connection, **kwargs: object) -> None:
                    prepare(conn, **kwargs)
                    if phase == "initial":
                        contend()

                def baseline(conn: sqlite3.Connection, specs: tuple) -> dict:
                    reads.append(1)
                    result = read(conn, specs)
                    if len(reads) == 1 and phase != "initial":
                        contend()
                    return result

                try:
                    with patch.object(MAINTENANCE, "prepare_style_maintenance", prepared), \
                         patch.object(MAINTENANCE, "read_layer_states", baseline):
                        result = MAINTENANCE.run_style_maintenance(first, force=True, cancel=cancel)
                    self.assertEqual((result.outcome, result.error_category, result.attempted),
                                     ("deferred", "StyleWriterBusy", False))
                    self.assertEqual(len(reads), 1 if phase == "initial" else 2)
                    self.assertTrue(second.in_transaction)
                    self.assertFalse(first.in_transaction)
                    self.assertEqual(self.output(first), before)
                    self.assertEqual(self.checkpoint(first), checkpoint)
                finally:
                    second.rollback()
                self.assertEqual(MAINTENANCE.run_style_maintenance(first, force=True).outcome, "success")

    def test_read_deferral_requires_known_lock_and_unchanged_caller_boundary(self) -> None:
        for borrowed in (False, True):
            for changed_boundary in (False, True):
                for locked in (False, True):
                    with self.subTest(borrowed=borrowed, changed_boundary=changed_boundary, locked=locked):
                        conn = self.connection()
                        self.add(conn)
                        MAINTENANCE.run_style_maintenance(conn)
                        conn.execute("CREATE TEMP TABLE caller_pending(value TEXT)")
                        if borrowed:
                            conn.execute("BEGIN")
                            conn.execute("INSERT INTO caller_pending VALUES ('synthetic pending')")

                        def failed_read(_conn: sqlite3.Connection, _specs: tuple) -> dict:
                            if changed_boundary:
                                if borrowed:
                                    conn.commit()
                                else:
                                    conn.execute("BEGIN")
                            raise sqlite3.OperationalError("database table is locked: maintenance_source_state"
                                                          if locked else "synthetic unexpected read")

                        with patch.object(MAINTENANCE, "read_layer_states", failed_read):
                            if locked and not changed_boundary:
                                result = MAINTENANCE.run_style_maintenance(conn, force=True, allow_caller_transaction=borrowed)
                                self.assertEqual((result.outcome, result.error_category, result.pending_caller_commit),
                                                 ("deferred", "StyleWriterBusy", borrowed))
                                self.assertEqual(conn.in_transaction, borrowed)
                                self.assertEqual(conn.execute("SELECT count(*) FROM caller_pending").fetchone()[0], int(borrowed))
                            else:
                                with self.assertRaises(sqlite3.OperationalError):
                                    MAINTENANCE.run_style_maintenance(conn, force=True, allow_caller_transaction=borrowed)
                        conn.rollback()


if __name__ == "__main__":
    unittest.main()
