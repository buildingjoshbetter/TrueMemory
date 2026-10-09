"""Atomic incremental style coverage with actual source and in-memory SQLite."""

import ast
import builtins
import json
import logging
import runpy
import sqlite3
import sys
import threading
import types
import unittest
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(Path(__file__).with_name("test-l0-style-maintenance-747.py")))
STORAGE, STYLE, MAINTENANCE = BASE["STORAGE"], BASE["STYLE"], BASE["MAINTENANCE"]
ACCUMULATOR = BASE["ACCUMULATOR"]


def load_add() -> types.FunctionType:
    rebuild = types.ModuleType("synthetic_incremental_rebuild")
    rebuild.__dict__.update(sqlite3=sqlite3, Iterator=Iterator, AbstractContextManager=AbstractContextManager,
                            contextmanager=contextmanager, nullcontext=nullcontext)
    source = ast.parse((ROOT / "truememory/rebuild_source.py").read_text(encoding="utf-8"))
    selected = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "rebuild_transaction"]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "synthetic-rebuild", "exec"), rebuild.__dict__)
    modules = {"truememory.personality_style_vec": STYLE, "truememory.rebuild_source": rebuild,
               "truememory.maintenance": MAINTENANCE,
               "truememory.tier_switch.runtime": MAINTENANCE._fixture_modules["truememory.tier_switch.runtime"],
               "truememory.tier_switch.writer": ACCUMULATOR["load_stdlib_source"]("tier_switch/writer")}

    def safe_import(name: str, globals: dict | None = None, locals: dict | None = None,
                    fromlist: tuple = (), level: int = 0) -> object:
        if name in modules:
            return modules[name]
        if name.split(".", 1)[0] not in sys.stdlib_module_names:
            raise AssertionError("Non-stdlib import forbidden: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    tree = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
    add = next(node for node in engine.body if isinstance(node, ast.FunctionDef) and node.name == "add")
    namespace = {"engine_operation": lambda function: function, "__builtins__": dict(vars(builtins), __import__=safe_import), "MAX_CONTENT_LENGTH": 50000,
                 "insert_message": STORAGE.insert_message, "_update_style_vec": STYLE.update_entity_style_vector_incremental,
                 "logger": logging.getLogger("synthetic-style-incremental")}
    validator = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_validate_add_content")
    exec(compile(ast.Module(body=[validator, add], type_ignores=[]), "actual-engine-add", "exec"), namespace)
    return namespace["add"]


ADD = load_add()


class BoundaryFailure:
    def __init__(self, conn: sqlite3.Connection, boundary: str) -> None:
        self.conn, self.boundary = conn, boundary
        self.denied = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.conn, name)

    def commit(self) -> None:
        self.execute("COMMIT")

    def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
        match = (sql == "COMMIT" if self.boundary == "commit" else
                 sql == "RELEASE SAVEPOINT rebuild_source" if self.boundary == "caller_release" else
                 sql.startswith("RELEASE truememory_style_coverage_"))
        if match and not self.denied:
            self.denied += 1
            action = ((sqlite3.SQLITE_TRANSACTION, "COMMIT", None) if sql == "COMMIT" else
                      (sqlite3.SQLITE_SAVEPOINT, "RELEASE", sql.split()[-1]))
            fault = ACCUMULATOR["StatementFailure"](self.conn, sql, action)
            return fault.execute(sql, *args)
        return self.conn.execute(sql, *args)


class StyleIncrementalTests(unittest.TestCase):
    def connection(self, *, bootstrap: bool = True, cache: int = 100) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:", cached_statements=cache)
        self.addCleanup(conn.close)
        conn.executescript(STORAGE._SCHEMA_SQL)
        STORAGE._initialize_maintenance_tracking(conn)
        conn.execute("CREATE TABLE sentinel(value TEXT)")
        conn.commit()
        if bootstrap:
            self.assertEqual(MAINTENANCE.run_style_maintenance(conn).outcome, "success_empty")
        return conn

    def engine(self, conn: sqlite3.Connection) -> types.SimpleNamespace:
        engine = types.SimpleNamespace(conn=conn, _has_vectors=False, _has_personality=False, _has_style_vec=True,
                                       _write_lock=threading.Lock(), _ensure_connection=lambda: None,
                                       _maybe_auto_consolidate=lambda: None)
        engine.add = types.MethodType(ADD, engine)
        return engine

    def checkpoint(self, conn: sqlite3.Connection) -> tuple | None:
        return MAINTENANCE._read_layer_row(conn, "style_vectors")

    def rows(self, conn: sqlite3.Connection) -> list[tuple]:
        return conn.execute("SELECT * FROM entity_style_vectors ORDER BY entity").fetchall()

    def state(self, conn: sqlite3.Connection) -> object:
        spec = MAINTENANCE.style_layer_spec(conn)
        return MAINTENANCE.read_layer_states(conn, (spec,))[spec.layer]

    def append(self, conn: sqlite3.Connection, previous: object, *, content: str = "synthetic ordinary",
               sender: str = "Synthetic", directive: bool = False, message_id: int | None = None,
               publish: object = None) -> object:
        if message_id is None:
            message_id = STORAGE.insert_message(conn, {"content": content, "sender": sender, "directive": directive})
        vector = STYLE.compute_style_vector(content)
        return MAINTENANCE._publish_style_append(
            conn, previous, message_id, sender=sender, content=content, directive=directive, precomputed=True,
            publish=publish or (lambda: STYLE.update_entity_style_vector_incremental(
                conn, sender, content, _pre_computed_vec=vector)),
        )

    def test_75_actual_adds_advance_raw_sums_and_attempts_without_any_full_build(self) -> None:
        conn = self.connection()
        engine = self.engine(conn)
        expected = {}
        original = STYLE._compute_entity_style_vectors
        with patch.object(STYLE, "_compute_entity_style_vectors", wraps=original) as builds:
            for index in range(75):
                sender = "" if index % 11 == 0 else ("Synthetic" if index % 2 else "SYNTHETIC")
                directive = index % 13 == 0
                text = "a" if index % 7 == 0 else "synthetic ordinary " + str(index)
                result = engine.add(text, sender=sender, directive=directive)
                self.assertEqual(result["id"], index + 1)
                if sender and not directive:
                    total, count = expected.setdefault(sender.lower(), [[0.0] * STYLE.DIM, 0])
                    vector = STYLE.compute_style_vector(text)
                    expected[sender.lower()][0] = [one + two for one, two in zip(total, vector)]
                    expected[sender.lower()][1] = count + 1
                state = self.state(conn)
                self.assertEqual((state.successful_revision, state.attempted_revision, state.attempted_insert_count),
                                 (index + 1, index + 1, index + 1))
                self.assertEqual((state.successful_coverage, state.full_rebuild_required), ("complete", False))
                self.assertEqual(MAINTENANCE.run_style_maintenance(conn, threshold=25).outcome, "current")
            self.assertEqual(builds.call_count, 0)
        self.assertEqual(self.state(conn).output_count, len(expected))
        for entity, (total, count) in expected.items():
            row = conn.execute("SELECT vector_sum,message_count,accumulator_version,vector FROM entity_style_vectors WHERE entity=?",
                               (entity,)).fetchone()
            self.assertEqual(row[:3], (STYLE._STYLE_SUM.pack(*total), count, 1))
            self.assertEqual(json.loads(row[3]), STYLE._style_profile_from_sum(total, count))
        self.assertFalse(conn.in_transaction)

    def test_new_entities_case_merging_duplicates_and_zero_vectors_count_once(self) -> None:
        conn = self.connection()
        engine = self.engine(conn)
        for sender, text in (("One", "a"), ("ONE", "ab"), ("One", "ab"), ("Two", "synthetic second")):
            engine.add(text, sender=sender)
        self.assertEqual(conn.execute("SELECT entity,message_count FROM entity_style_vectors ORDER BY entity").fetchall(),
                         [("one", 3), ("two", 1)])
        self.assertEqual(STYLE.get_entity_style_vector(conn, "one"), [0.0] * STYLE.DIM)
        self.assertEqual(self.state(conn).output_count, 2)

    def test_excluded_rows_advance_successful_empty_without_calling_updater(self) -> None:
        conn = self.connection()
        with patch.dict(ADD.__globals__, _update_style_vec=lambda *args, **kwargs: self.fail("Excluded profile write")):
            self.engine(conn).add("synthetic directive", sender="Synthetic", directive=True)
            self.engine(conn).add("synthetic anonymous", sender="")
        self.assertEqual(self.rows(conn), [])
        self.assertEqual((self.state(conn).outcome, self.state(conn).attempted_insert_count), ("success_empty", 2))

    def test_no_enrollment_and_no_rebuild_when_coverage_is_missing_or_dirty(self) -> None:
        for mode in ("missing", "dirty", "legacy", "running", "failed", "future", "hash", "triggers"):
            with self.subTest(mode=mode):
                conn = self.connection(bootstrap=mode != "missing")
                if mode == "dirty":
                    conn.execute("UPDATE maintenance_layers SET full_rebuild_required=1 WHERE layer='style_vectors'")
                elif mode == "legacy":
                    conn.execute("INSERT INTO entity_style_vectors(entity,vector) VALUES ('synthetic','[]')")
                elif mode in {"running", "failed"}:
                    conn.execute("UPDATE maintenance_layers SET outcome=? WHERE layer='style_vectors'", (mode,))
                elif mode == "future":
                    conn.execute("ALTER TABLE entity_style_vectors RENAME TO old_style")
                    conn.execute("CREATE TABLE entity_style_vectors(entity TEXT PRIMARY KEY, vector TEXT, message_count INTEGER, "
                                 "updated_at TEXT, vector_sum BLOB, accumulator_version INTEGER DEFAULT 9)")
                    for name, definition in STORAGE._STYLE_OUTPUT_TRIGGERS.items():
                        conn.execute("DROP TRIGGER " + name)
                        conn.execute(definition)
                    self.assertEqual(self.state(conn).successful_coverage, "complete")
                elif mode == "hash":
                    conn.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
                elif mode == "triggers":
                    conn.execute("DROP TRIGGER truememory_style_output_insert")
                conn.commit()
                before, checkpoint = self.rows(conn), self.checkpoint(conn)
                updater = Mock(side_effect=AssertionError("Unsafe profile write"))
                with patch.dict(ADD.__globals__, _update_style_vec=updater), \
                     patch.object(STYLE, "_compute_entity_style_vectors", side_effect=AssertionError("No fallback rebuild")) as builds:
                    self.engine(conn).add("synthetic pending", sender="Synthetic")
                updater.assert_not_called()
                builds.assert_not_called()
                self.assertEqual(self.rows(conn), before)
                self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)
                if mode == "missing":
                    self.assertIsNone(self.checkpoint(conn))
                    self.assertEqual(conn.execute("SELECT count(*) FROM maintenance_layers").fetchone()[0], 8)
                if mode == "failed":
                    self.assertEqual(self.checkpoint(conn), checkpoint)

    def test_incomplete_attempt_or_source_proof_cannot_advance_coverage(self) -> None:
        mutations = (
            "UPDATE maintenance_layers SET attempted_insert_count=NULL WHERE layer='style_vectors'",
            "UPDATE maintenance_layers SET successful_coverage='unverified' WHERE layer='style_vectors'",
            "UPDATE maintenance_layers SET output_count=-1 WHERE layer='style_vectors'",
            "UPDATE maintenance_layers SET builder_version=9 WHERE layer='style_vectors'",
            "UPDATE maintenance_source_state SET revision=1",
        )
        for sql in mutations:
            with self.subTest(sql=sql):
                conn = self.connection()
                conn.execute(sql)
                conn.commit()
                updater = Mock(side_effect=AssertionError("Incomplete proof"))
                with patch.dict(ADD.__globals__, _update_style_vec=updater):
                    self.engine(conn).add("synthetic pending", sender="Synthetic")
                updater.assert_not_called()
                self.assertEqual(self.rows(conn), [])
                self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)

    def test_precompute_failure_preserves_source_and_invalidates_current_attempt(self) -> None:
        conn = self.connection()
        with patch.object(STYLE, "compute_style_vector", side_effect=ValueError("synthetic failure")):
            self.engine(conn).add("synthetic retained source", sender="Synthetic")
        state = self.state(conn)
        self.assertEqual((state.outcome, state.error_category, state.attempted_revision),
                         ("pending", "StylePrecomputeFailed", None))
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)
        self.assertEqual(self.rows(conn), [])
        self.assertEqual(MAINTENANCE.run_style_maintenance(conn).outcome, "success")
        self.assertEqual(conn.execute("SELECT message_count FROM entity_style_vectors").fetchone()[0], 1)

    def test_source_insert_hash_trigger_cannot_certify_an_old_hash(self) -> None:
        conn = self.connection()
        engine = self.engine(conn)
        engine.add("synthetic baseline", sender="Synthetic")
        before = self.rows(conn)
        conn.execute("CREATE TRIGGER synthetic_source_hash AFTER INSERT ON messages BEGIN "
                     "UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'; END")
        conn.commit()
        engine.add("synthetic retained", sender="Synthetic")
        self.assertEqual(self.rows(conn), before)
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)
        self.assertEqual(conn.execute("SELECT value FROM metadata WHERE key='style_vec_hash_version'").fetchone()[0], "1")
        self.assertEqual((self.state(conn).outcome, self.state(conn).error_category), ("pending", "StyleHashPending"))

    def test_extra_main_and_temp_output_triggers_defer_without_deleting_other_profiles(self) -> None:
        for temporary, name in ((False, "synthetic_extra"), (True, "synthetic_extra"),
                                (True, "truememory_style_output_insert"), (False, "truememory_style_output_extra")):
            with self.subTest(temporary=temporary, name=name):
                conn = self.connection()
                engine = self.engine(conn)
                engine.add("synthetic other profile", sender="Other")
                before, checkpoint = self.rows(conn), self.checkpoint(conn)
                conn.execute("CREATE " + ("TEMP " if temporary else "") + "TRIGGER " + name
                             + " AFTER INSERT ON main.entity_style_vectors WHEN new.entity='synthetic' BEGIN "
                             "DELETE FROM entity_style_vectors WHERE entity='other'; END")
                conn.commit()
                updater = Mock(wraps=STYLE.update_entity_style_vector_incremental)
                with patch.dict(ADD.__globals__, _update_style_vec=updater):
                    engine.add("synthetic retained", sender="Synthetic")
                updater.assert_not_called()
                self.assertEqual(self.rows(conn), before)
                self.assertEqual(self.checkpoint(conn), checkpoint)
                self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)
                changes = conn.total_changes
                result = MAINTENANCE.run_style_maintenance(conn, force=True)
                self.assertEqual((result.outcome, result.error_category, result.attempted),
                                 ("deferred", "StyleTriggersUnsupported", False))
                spec = MAINTENANCE.style_layer_spec(conn)
                self.assertTrue(spec.resolve_dependency().deferred)
                self.assertEqual(MAINTENANCE.run_layers(conn, (spec,), force=True)[0].error_category,
                                 "StyleTriggersUnsupported")
                self.assertEqual(conn.total_changes, changes)

    def test_extra_checkpoint_triggers_cannot_run_while_reporting_pending(self) -> None:
        for temporary in (False, True):
            for action in ("failure", "delete_profile"):
                with self.subTest(temporary=temporary, action=action):
                    conn = self.connection()
                    engine = self.engine(conn)
                    engine.add("synthetic other profile", sender="Other")
                    before, checkpoint = self.rows(conn), self.checkpoint(conn)
                    change = ("UPDATE maintenance_layers SET outcome='failed',error_category='NewerFailure',"
                              "run_generation='newer-owner',attempted_insert_count=999 WHERE layer='style_vectors';"
                              if action == "failure" else "DELETE FROM entity_style_vectors WHERE entity='other';")
                    conn.execute("CREATE " + ("TEMP " if temporary else "")
                                 + "TRIGGER synthetic_checkpoint AFTER UPDATE OF outcome ON main.maintenance_layers "
                                 "WHEN new.layer='style_vectors' AND new.outcome='pending' BEGIN " + change + " END")
                    conn.commit()
                    with patch.object(STYLE, "compute_style_vector", side_effect=ValueError("synthetic precompute")):
                        engine.add("synthetic retained", sender="Synthetic")
                    self.assertEqual(self.rows(conn), before)
                    self.assertEqual(self.checkpoint(conn), checkpoint)
                    self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)
                    self.assertEqual(self.state(conn).dependency.error_category, "StyleTriggersUnsupported")

    def test_metadata_triggers_defer_before_preparation_or_rebuild_writes(self) -> None:
        for temporary in (False, True):
            for readiness in ("ready", "unenrolled", "repair"):
                for borrowed in (False, True):
                    with self.subTest(temporary=temporary, readiness=readiness, borrowed=borrowed):
                        conn = self.connection(bootstrap=False)
                        STORAGE.insert_message(conn, {"content": "synthetic source", "sender": "Synthetic", "directive": False})
                        conn.commit()
                        if readiness != "unenrolled":
                            self.assertEqual(MAINTENANCE.run_style_maintenance(conn).outcome, "success")
                        if readiness == "repair":
                            conn.execute("DROP TRIGGER truememory_style_output_update")
                        conn.execute("CREATE " + ("TEMP " if temporary else "")
                                     + "TRIGGER synthetic_metadata AFTER UPDATE OF value ON main.metadata "
                                     "WHEN new.key='style_vec_hash_version' BEGIN DELETE FROM entity_style_vectors; END")
                        conn.commit()
                        before, checkpoint = self.rows(conn), self.checkpoint(conn)
                        if borrowed:
                            conn.execute("INSERT INTO sentinel VALUES ('synthetic caller')")
                        changes = conn.total_changes
                        sql = []
                        conn.set_trace_callback(sql.append)
                        result = MAINTENANCE.run_style_maintenance(conn, force=True, allow_caller_transaction=borrowed)
                        conn.set_trace_callback(None)
                        self.assertEqual((result.outcome, result.error_category, result.attempted),
                                         ("deferred", "StyleTriggersUnsupported", False))
                        self.assertFalse(any(statement.lstrip().upper().startswith(
                            ("INSERT", "UPDATE", "DELETE", "REPLACE", "BEGIN", "SAVEPOINT")) for statement in sql))
                        self.assertEqual(conn.total_changes, changes)
                        self.assertEqual(self.rows(conn), before)
                        self.assertEqual(self.checkpoint(conn), checkpoint)
                        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)
                        self.assertEqual(conn.in_transaction, borrowed)
                        self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(),
                                         [("synthetic caller",)] if borrowed else [])
                        conn.rollback()
                        self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [])
                        self.assertEqual(self.rows(conn), before)
                        self.assertEqual(self.checkpoint(conn), checkpoint)

    def test_metadata_triggers_skip_incremental_publication_and_pending_diagnostics(self) -> None:
        for temporary in (False, True):
            with self.subTest(temporary=temporary):
                conn = self.connection()
                engine = self.engine(conn)
                engine.add("synthetic baseline", sender="Synthetic")
                before, checkpoint = self.rows(conn), self.checkpoint(conn)
                conn.execute("CREATE " + ("TEMP " if temporary else "")
                             + "TRIGGER synthetic_metadata AFTER UPDATE OF value ON main.metadata "
                             "WHEN new.key='style_vec_hash_version' BEGIN DELETE FROM entity_style_vectors; END")
                conn.commit()
                updater = Mock(wraps=STYLE.update_entity_style_vector_incremental)
                with patch.dict(ADD.__globals__, _update_style_vec=updater):
                    engine.add("synthetic retained", sender="Synthetic")
                updater.assert_not_called()
                self.assertEqual(self.rows(conn), before)
                self.assertEqual(self.checkpoint(conn), checkpoint)
                self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)
                self.assertEqual(self.state(conn).dependency.error_category, "StyleTriggersUnsupported")

    def test_metadata_trigger_installed_during_compute_is_rejected_before_publication(self) -> None:
        for temporary in (False, True):
            with self.subTest(temporary=temporary):
                conn = self.connection()
                self.engine(conn).add("synthetic baseline", sender="Synthetic")
                before, checkpoint = self.rows(conn), self.checkpoint(conn)
                compute = STYLE._compute_entity_style_vectors

                def installed_after_compute(*args: object, **kwargs: object) -> object:
                    value = compute(*args, **kwargs)
                    conn.execute("CREATE " + ("TEMP " if temporary else "")
                                 + "TRIGGER synthetic_late_metadata AFTER UPDATE OF value ON main.metadata "
                                 "WHEN new.key='style_vec_hash_version' BEGIN DELETE FROM entity_style_vectors; END")
                    return value

                with patch.object(STYLE, "_compute_entity_style_vectors", installed_after_compute), \
                     patch.object(STYLE, "_publish_style_vectors", wraps=STYLE._publish_style_vectors) as publish:
                    result = MAINTENANCE.run_style_maintenance(conn, force=True)
                publish.assert_not_called()
                self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleTriggersUnsupported"))
                self.assertEqual(self.rows(conn), before)
                self.assertEqual(self.checkpoint(conn), checkpoint)
                self.assertFalse(MAINTENANCE._style_extra_triggers(conn))
                self.assertFalse(conn.in_transaction)

    def test_source_trigger_newer_failure_is_not_overwritten_by_append_reporting(self) -> None:
        conn = self.connection()
        conn.executescript("""CREATE TRIGGER synthetic_checkpoint AFTER UPDATE OF outcome ON maintenance_layers
            WHEN new.layer='style_vectors' AND new.outcome='pending' BEGIN
            UPDATE maintenance_layers SET outcome='failed',error_category='NewerFailure',
                run_generation='newer-owner',attempted_insert_count=999 WHERE layer='style_vectors'; END;
            CREATE TRIGGER synthetic_source AFTER INSERT ON messages BEGIN
            UPDATE maintenance_layers SET outcome='pending' WHERE layer='style_vectors'; END;""")
        self.engine(conn).add("synthetic retained", sender="Synthetic")
        state = self.state(conn)
        self.assertEqual((state.outcome, state.error_category, state.run_generation, state.attempted_insert_count),
                         ("failed", "NewerFailure", "newer-owner", 999))
        self.assertEqual(self.rows(conn), [])

    def test_preparation_rejects_unsupported_triggers_before_enrollment_or_repair(self) -> None:
        for bootstrap in (False, True):
            for temporary in (False, True):
                with self.subTest(bootstrap=bootstrap, temporary=temporary):
                    conn = self.connection(bootstrap=bootstrap)
                    if bootstrap:
                        conn.execute("DROP TRIGGER truememory_style_output_update")
                    conn.execute("CREATE " + ("TEMP " if temporary else "")
                                 + "TRIGGER synthetic_preparation AFTER UPDATE ON main.maintenance_layers BEGIN "
                                 "INSERT INTO sentinel VALUES ('unexpected checkpoint mutation'); END")
                    conn.commit()
                    before = self.checkpoint(conn)
                    changes = conn.total_changes
                    result = MAINTENANCE.run_style_maintenance(conn)
                    self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleTriggersUnsupported"))
                    self.assertEqual(conn.total_changes, changes)
                    self.assertEqual(self.checkpoint(conn), before)
                    self.assertFalse(conn.in_transaction)

    def test_preparation_rechecks_trigger_admission_inside_its_original_writer_boundary(self) -> None:
        for borrowed in (False, True):
            conn = self.connection(bootstrap=False)
            if borrowed:
                conn.execute("INSERT INTO sentinel VALUES ('synthetic caller')")
            installed = []

            class InstallBeforeAdmission:
                def __getattr__(self, name: str) -> object:
                    return getattr(conn, name)

                def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                    if sql in {"BEGIN IMMEDIATE", "SAVEPOINT truememory_style_tracking"} and not installed:
                        installed.append(True)
                        conn.execute("CREATE TEMP TRIGGER synthetic_preparation AFTER UPDATE ON main.maintenance_layers BEGIN "
                                     "INSERT INTO sentinel VALUES ('unexpected checkpoint mutation'); END")
                    return conn.execute(sql, *args)

            result = MAINTENANCE.run_style_maintenance(InstallBeforeAdmission(), allow_caller_transaction=borrowed)
            self.assertEqual(result.error_category, "StyleTriggersUnsupported")
            self.assertEqual(installed, [True])
            self.assertEqual(conn.in_transaction, borrowed)
            self.assertIsNone(self.checkpoint(conn))
            self.assertEqual(conn.execute("SELECT value FROM sentinel").fetchall(),
                             [("synthetic caller",)] if borrowed else [])
            conn.rollback()

    def test_trigger_installed_after_baseline_cannot_fire_on_running_diagnostic(self) -> None:
        conn = self.connection()
        before = self.checkpoint(conn)
        read = MAINTENANCE.read_layer_states
        reads = []

        def installed_after_read(*args: object, **kwargs: object) -> object:
            state = read(*args, **kwargs)
            reads.append(True)
            if len(reads) == 2:
                conn.execute("CREATE TEMP TRIGGER synthetic_late_checkpoint AFTER UPDATE ON main.maintenance_layers "
                             "BEGIN INSERT INTO sentinel VALUES ('unexpected diagnostic write'); END")
            return state

        with patch.object(MAINTENANCE, "read_layer_states", installed_after_read), \
             patch.object(STYLE, "_compute_entity_style_vectors", wraps=STYLE._compute_entity_style_vectors) as builds:
            result = MAINTENANCE.run_style_maintenance(conn, force=True)
        builds.assert_not_called()
        self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleTriggersUnsupported"))
        self.assertEqual(self.checkpoint(conn), before)
        self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [])
        self.assertFalse(conn.in_transaction)

    def test_trigger_installed_after_rollback_cannot_fire_on_diagnostic_restoration(self) -> None:
        conn = self.connection()
        installed = []

        class InstallAfterRollback:
            def __getattr__(self, name: str) -> object:
                return getattr(conn, name)

            def rollback(self) -> None:
                conn.rollback()
                if not installed:
                    installed.append(True)
                    conn.execute("CREATE TEMP TRIGGER synthetic_late_restore AFTER UPDATE ON main.maintenance_layers "
                                 "BEGIN INSERT INTO sentinel VALUES ('unexpected restoration'); END")

        with patch.object(STYLE, "_compute_entity_style_vectors",
                          side_effect=MAINTENANCE.LayerDeferredError("synthetic busy", category="StyleWriterBusy")):
            result = MAINTENANCE.run_style_maintenance(InstallAfterRollback(), force=True)
        self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleTriggersUnsupported"))
        self.assertEqual(self.state(conn).outcome, "running")
        self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [])
        self.assertFalse(conn.in_transaction)
        conn.execute("DROP TRIGGER synthetic_late_restore")
        self.assertEqual(MAINTENANCE.run_style_maintenance(conn).outcome, "success_empty")

    def test_tracked_output_admission_checks_triggers_before_any_profile_write(self) -> None:
        conn = self.connection()
        before = self.checkpoint(conn)
        compute = STYLE._compute_entity_style_vectors

        def installed_after_compute(*args: object, **kwargs: object) -> object:
            value = compute(*args, **kwargs)
            conn.execute("CREATE TEMP TRIGGER synthetic_late_output AFTER DELETE ON main.entity_style_vectors "
                         "BEGIN INSERT INTO sentinel VALUES ('unexpected output publication'); END")
            return value

        with patch.object(STYLE, "_compute_entity_style_vectors", installed_after_compute), \
             patch.object(STYLE, "_publish_style_vectors", wraps=STYLE._publish_style_vectors) as publication:
            result = MAINTENANCE.run_style_maintenance(conn, force=True)
        publication.assert_not_called()
        self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleTriggersUnsupported"))
        self.assertEqual(self.checkpoint(conn), before)
        self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [])
        self.assertFalse(conn.in_transaction)

    def test_unsupported_attempt_diagnostic_reports_deferral_without_writing(self) -> None:
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable):
                conn = self.connection()
                before = self.checkpoint(conn)
                installed = []

                def install() -> None:
                    if not installed:
                        installed.append(True)
                        conn.execute("CREATE TEMP TRIGGER synthetic_late_attempt AFTER UPDATE ON main.maintenance_layers "
                                     "BEGIN INSERT INTO sentinel VALUES ('unexpected attempt write'); END")

                class InstallAfterFailure:
                    def __getattr__(self, name: str) -> object:
                        return getattr(conn, name)

                    def rollback(self) -> None:
                        conn.rollback()
                        install()

                if unavailable:
                    conn.execute("DROP TRIGGER truememory_style_output_update")
                    spec = MAINTENANCE.style_layer_spec(conn)
                    read = MAINTENANCE.read_layer_states
                    reads = []

                    def installed_after_read(*args: object, **kwargs: object) -> object:
                        states = read(*args, **kwargs)
                        reads.append(True)
                        if len(reads) == 2:
                            install()
                        return states

                    with patch.object(MAINTENANCE, "read_layer_states", installed_after_read):
                        result, = MAINTENANCE.run_layers(conn, (spec,), force=True)
                    self.assertEqual(self.checkpoint(conn), before)
                    self.assertFalse(result.attempted)
                else:
                    with patch.object(STYLE, "_compute_entity_style_vectors", side_effect=ValueError("synthetic failure")):
                        result = MAINTENANCE.run_style_maintenance(InstallAfterFailure(), force=True)
                    self.assertEqual(self.state(conn).outcome, "running")
                self.assertEqual(installed, [True])
                self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleTriggersUnsupported"))
                self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [])
                self.assertFalse(conn.in_transaction)

    def test_updater_metadata_and_checkpoint_mutations_rollback_only_style_work(self) -> None:
        for change in ("hash", "ready_trigger", "extra_main", "extra_temp", "checkpoint_trigger", "checkpoint", "format",
                       "metadata_main", "metadata_temp"):
            with self.subTest(change=change):
                conn = self.connection()
                engine = self.engine(conn)
                engine.add("synthetic baseline", sender="Synthetic")
                before = self.rows(conn)
                publish = STYLE.update_entity_style_vector_incremental

                def changed(*args: object, **kwargs: object) -> None:
                    publish(*args, **kwargs)
                    if change == "hash":
                        conn.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
                    elif change == "ready_trigger":
                        conn.execute("DROP TRIGGER truememory_style_output_insert")
                    elif change == "checkpoint":
                        conn.execute("UPDATE maintenance_layers SET outcome='failed',error_category='NewerFailure',"
                                     "run_generation='newer-owner',attempted_insert_count=999 WHERE layer='style_vectors'")
                    elif change == "format":
                        conn.execute("ALTER TABLE entity_style_vectors ADD COLUMN synthetic_format TEXT")
                    else:
                        table = ("metadata" if change.startswith("metadata_") else
                                 "maintenance_layers" if change == "checkpoint_trigger" else "entity_style_vectors")
                        conn.execute("CREATE " + ("TEMP " if change in {"extra_temp", "metadata_temp"} else "")
                                     + "TRIGGER synthetic_during_append AFTER UPDATE ON main." + table
                                     + " BEGIN INSERT INTO sentinel VALUES ('synthetic trigger'); END")

                with patch.dict(ADD.__globals__, _update_style_vec=changed):
                    engine.add("synthetic retained", sender="Synthetic")
                self.assertEqual(self.rows(conn), before)
                self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)
                self.assertEqual(self.state(conn).outcome, "pending")
                self.assertIsNone(self.state(conn).attempted_revision)
                self.assertEqual(conn.execute("SELECT value FROM metadata WHERE key='style_vec_hash_version'").fetchone()[0], "2")
                self.assertFalse(MAINTENANCE._style_extra_triggers(conn))
                self.assertTrue(STORAGE._style_output_tracking_ready(conn))

    def test_post_success_metadata_and_provenance_changes_cannot_certify_publication(self) -> None:
        for change in ("hash", "checkpoint", "extra_trigger"):
            with self.subTest(change=change):
                conn = self.connection()
                engine = self.engine(conn)
                engine.add("synthetic baseline", sender="Synthetic")
                before = self.rows(conn)
                record = MAINTENANCE.record_layer_success_in_transaction

                def changed(*args: object, **kwargs: object) -> None:
                    record(*args, **kwargs)
                    if change == "hash":
                        conn.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
                    elif change == "checkpoint":
                        conn.execute("UPDATE maintenance_layers SET outcome='failed',error_category='NewerFailure',"
                                     "run_generation='newer-owner',attempted_insert_count=999 WHERE layer='style_vectors'")
                    else:
                        conn.execute("CREATE TEMP TRIGGER synthetic_after_success AFTER INSERT ON main.entity_style_vectors "
                                     "BEGIN DELETE FROM entity_style_vectors WHERE entity='synthetic'; END")

                with patch.object(MAINTENANCE, "record_layer_success_in_transaction", changed):
                    engine.add("synthetic retained", sender="Synthetic")
                self.assertEqual(self.rows(conn), before)
                self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)
                self.assertEqual(self.state(conn).outcome, "pending")
                self.assertIsNone(self.state(conn).attempted_revision)

    def test_pending_category_compare_and_swap_preserves_a_newer_whole_row(self) -> None:
        conn = self.connection()
        changed = []

        class ChangeBeforeCategoryWrite:
            def __getattr__(self, name: str) -> object:
                return getattr(conn, name)

            def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                if sql.startswith("UPDATE maintenance_layers SET error_category=") and not changed:
                    changed.append(True)
                    conn.execute("UPDATE maintenance_layers SET outcome='failed',error_category='NewerFailure',"
                                 "run_generation='newer-owner',attempted_insert_count=999 WHERE layer='style_vectors'")
                return conn.execute(sql, *args)

        with patch.object(STYLE, "compute_style_vector", side_effect=ValueError("synthetic precompute")):
            self.engine(ChangeBeforeCategoryWrite()).add("synthetic retained", sender="Synthetic")
        self.assertEqual(changed, [True])
        state = self.state(conn)
        self.assertEqual((state.outcome, state.error_category, state.run_generation, state.attempted_insert_count),
                         ("failed", "NewerFailure", "newer-owner", 999))
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)

    def test_replayed_proof_never_counts_one_source_row_twice(self) -> None:
        conn = self.connection()
        conn.execute("BEGIN IMMEDIATE")
        previous = MAINTENANCE._capture_style_append(conn)
        self.assertEqual(self.append(conn, previous).outcome, "success")
        before, checkpoint = self.rows(conn), self.checkpoint(conn)
        result = self.append(conn, previous, message_id=1)
        self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleCoverageChanged"))
        self.assertEqual(self.rows(conn), before)
        self.assertEqual(self.checkpoint(conn), checkpoint)
        conn.commit()

    def test_exact_append_rejects_correction_extra_insert_epoch_and_low_id_reuse(self) -> None:
        for mode in ("correction", "extra", "epoch", "reuse", "wrong_id", "wrong_content", "counters", "nonappend"):
            with self.subTest(mode=mode):
                conn = self.connection()
                self.engine(conn).add("synthetic baseline", sender="Synthetic")
                before = self.rows(conn)
                conn.execute("BEGIN IMMEDIATE")
                previous = MAINTENANCE._capture_style_append(conn)
                new_id = STORAGE.insert_message(conn, {"content": "synthetic ordinary", "sender": "Synthetic"})
                if mode == "correction":
                    conn.execute("UPDATE messages SET content='synthetic correction' WHERE id=1")
                elif mode == "extra":
                    STORAGE.insert_message(conn, {"content": "synthetic extra", "sender": "Synthetic"})
                elif mode == "epoch":
                    conn.execute("UPDATE maintenance_source_state SET epoch='synthetic-new-epoch'")
                elif mode == "reuse":
                    conn.execute("DELETE FROM messages WHERE id=1")
                    conn.execute("INSERT INTO messages(id,content,sender) VALUES (1,'synthetic replacement','Synthetic')")
                elif mode == "counters":
                    conn.execute("UPDATE maintenance_source_state SET insert_count=insert_count+1")
                elif mode == "nonappend":
                    conn.execute("UPDATE maintenance_source_state SET nonappend_revision=revision")
                result = self.append(conn, previous, message_id=1 if mode == "wrong_id" else new_id,
                                     content="synthetic different" if mode == "wrong_content" else "synthetic ordinary",
                                     publish=lambda: self.fail("Unproven append reached updater"))
                self.assertEqual(result.outcome, "deferred")
                self.assertEqual(self.rows(conn), before)
                conn.rollback()

    def test_unique_schema_and_insert_below_high_watermark_are_not_exact_appends(self) -> None:
        for unique in (False, True):
            conn = self.connection()
            if unique:
                conn.execute("CREATE UNIQUE INDEX synthetic_unique ON messages(content)")
            conn.execute("BEGIN IMMEDIATE")
            previous = MAINTENANCE._capture_style_append(conn)
            conn.execute("INSERT INTO messages(id,content,sender) VALUES (?, 'synthetic ordinary','Synthetic')",
                         (1 if unique else 0,))
            result = self.append(conn, previous, message_id=1 if unique else 0,
                                 publish=lambda: self.fail("Nonappend reached updater"))
            self.assertEqual(result.error_category, "StyleAppendUnproven")
            self.assertEqual(self.rows(conn), [])
            conn.rollback()

    def test_newer_output_invalidation_survives_old_append_proof(self) -> None:
        conn = self.connection()
        conn.execute("BEGIN IMMEDIATE")
        previous = MAINTENANCE._capture_style_append(conn)
        new_id = STORAGE.insert_message(conn, {"content": "synthetic ordinary", "sender": "Synthetic"})
        conn.execute("INSERT INTO entity_style_vectors(entity,vector) VALUES ('other','[]')")
        checkpoint = self.checkpoint(conn)
        result = self.append(conn, previous, message_id=new_id, publish=lambda: self.fail("Stale proof"))
        self.assertEqual(result.error_category, "StyleCoverageChanged")
        self.assertEqual(self.checkpoint(conn), checkpoint)
        conn.rollback()

    def test_running_worker_cannot_restore_over_newly_missed_foreground_append(self) -> None:
        conn = self.connection()
        before = self.checkpoint(conn)
        with MAINTENANCE.maintenance_owner(None) as owner:
            conn.execute("UPDATE maintenance_layers SET outcome='running',run_generation=? WHERE layer='style_vectors'",
                         (owner.generation,))
            conn.commit()
            running = self.checkpoint(conn)
            self.engine(conn).add("synthetic foreground append", sender="Synthetic")
            invalidated = self.checkpoint(conn)
            self.assertEqual(self.state(conn).outcome, "pending")
            with self.assertRaises(MAINTENANCE.MaintenanceBusyError):
                MAINTENANCE._restore_deferred_attempt(conn, "style_vectors", before, running, owner,
                                                     protect_rollback=True)
            self.assertEqual(self.checkpoint(conn), invalidated)
        self.assertEqual(MAINTENANCE.run_style_maintenance(conn).outcome, "success")
        self.assertEqual(conn.execute("SELECT message_count FROM entity_style_vectors").fetchone()[0], 1)

    def test_publication_error_rolls_back_profile_but_keeps_source_and_pending(self) -> None:
        for error in (ValueError("synthetic failure"), sqlite3.OperationalError("synthetic SQL failure"),
                      sqlite3.OperationalError("database is locked")):
            with self.subTest(error=type(error).__name__, locked=str(error) == "database is locked"):
                conn = self.connection()
                engine = self.engine(conn)
                engine.add("synthetic baseline", sender="Synthetic")
                before = self.rows(conn)
                publish = STYLE.update_entity_style_vector_incremental

                def failed(*args: object, **kwargs: object) -> None:
                    publish(*args, **kwargs)
                    raise error

                with patch.dict(ADD.__globals__, _update_style_vec=failed):
                    engine.add("synthetic retained", sender="Synthetic")
                self.assertEqual(self.rows(conn), before)
                self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)
                state = self.state(conn)
                self.assertEqual((state.outcome, state.successful_coverage, state.attempted_revision),
                                 ("pending", "unverified", None))
                self.assertEqual(state.error_category, "StyleWriterBusy" if str(error) == "database is locked" else type(error).__name__)

    def test_source_mutation_inside_updater_is_rolled_back_without_erasing_append(self) -> None:
        conn = self.connection()
        publish = STYLE.update_entity_style_vector_incremental

        def changed(*args: object, **kwargs: object) -> None:
            publish(*args, **kwargs)
            conn.execute("UPDATE messages SET content='synthetic unapproved correction' WHERE id=1")

        with patch.dict(ADD.__globals__, _update_style_vec=changed):
            self.engine(conn).add("synthetic retained", sender="Synthetic")
        self.assertEqual(conn.execute("SELECT content FROM messages").fetchone()[0], "synthetic retained")
        self.assertEqual(self.rows(conn), [])
        self.assertEqual(self.state(conn).error_category, "StyleSourceChanged")

    def test_provenance_failure_rolls_back_the_successful_raw_sum_append(self) -> None:
        conn = self.connection()
        engine = self.engine(conn)
        engine.add("synthetic baseline", sender="Synthetic")
        before = self.rows(conn)
        record = MAINTENANCE.record_layer_success_in_transaction

        def failed(*args: object, **kwargs: object) -> None:
            record(*args, **kwargs)
            raise sqlite3.OperationalError("synthetic provenance failure")

        with patch.object(MAINTENANCE, "record_layer_success_in_transaction", failed):
            engine.add("synthetic retained", sender="Synthetic")
        self.assertEqual(self.rows(conn), before)
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 2)
        self.assertEqual((self.state(conn).outcome, self.state(conn).attempted_revision), ("pending", None))

    def test_failed_pending_write_reports_failure_without_claiming_durable_invalidation(self) -> None:
        conn = self.connection()
        conn.execute("BEGIN IMMEDIATE")
        previous = MAINTENANCE._capture_style_append(conn)
        checkpoint = self.checkpoint(conn)
        new_id = STORAGE.insert_message(conn, {"content": "synthetic ordinary", "sender": "Synthetic"})
        with patch.object(MAINTENANCE, "_STYLE_OUTPUT_INVALIDATE_SQL", "synthetic invalid SQL"):
            result = MAINTENANCE._publish_style_append(
                conn, previous, new_id, sender="Synthetic", content="synthetic ordinary", directive=False,
                precomputed=False, publish=lambda: self.fail("No precomputed vector"),
            )
        self.assertEqual((result.outcome, result.error_category, result.coverage, result.pending_caller_commit),
                         ("failed", "OperationalError", "unverified", True))
        self.assertEqual(self.checkpoint(conn), checkpoint)
        state = self.state(conn)
        self.assertEqual(MAINTENANCE.layer_freshness(state.source, state, state.dependency), "append_pending")
        conn.commit()
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)

    def test_read_capture_failure_preserves_source_and_does_not_claim_success(self) -> None:
        unusual_error = type("Synthetic Invalid Category", (sqlite3.OperationalError,), {})
        for error in (sqlite3.OperationalError("database table is locked: maintenance_layers"),
                      sqlite3.OperationalError("synthetic capture failure"), unusual_error("synthetic detail")):
            conn = self.connection()
            read = MAINTENANCE.read_layer_states
            failures = []

            def unavailable(*args: object, **kwargs: object) -> object:
                if not failures:
                    failures.append(True)
                    raise error
                return read(*args, **kwargs)

            with patch.object(MAINTENANCE, "read_layer_states", unavailable):
                self.engine(conn).add("synthetic retained", sender="Synthetic")
            self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)
            self.assertEqual(self.rows(conn), [])
            self.assertEqual(self.state(conn).successful_coverage, "unverified")
            self.assertRegex(self.state(conn).error_category, r"\A[A-Za-z][A-Za-z0-9_]{0,63}\Z")

    def test_borrowed_append_success_and_failure_remain_in_caller_transaction(self) -> None:
        for failed in (False, True):
            conn = self.connection()
            before = self.checkpoint(conn)
            conn.execute("INSERT INTO sentinel VALUES ('synthetic pending')")
            updater = STYLE.update_entity_style_vector_incremental
            with patch.dict(ADD.__globals__, _update_style_vec=(lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("synthetic")))
                            if failed else updater):
                self.engine(conn).add("synthetic caller source", sender="Synthetic")
            self.assertTrue(conn.in_transaction)
            self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)
            conn.rollback()
            self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], 0)
            self.assertEqual(self.rows(conn), [])
            self.assertEqual(self.checkpoint(conn), before)

    def test_cached_source_commit_and_caller_release_faults_restore_whole_add(self) -> None:
        for cache in (0, 5, 100):
            for borrowed in (False, True):
                with self.subTest(cache=cache, borrowed=borrowed):
                    conn = self.connection(cache=cache)
                    if borrowed:
                        conn.execute("BEGIN")
                        self.engine(conn).add("synthetic warm release", sender="Synthetic")
                        conn.rollback()
                    before = self.checkpoint(conn)
                    source = MAINTENANCE.read_source_revision(conn)
                    if borrowed:
                        conn.execute("INSERT INTO sentinel VALUES ('synthetic caller')")
                    fault = BoundaryFailure(conn, "caller_release" if borrowed else "commit")
                    with self.assertRaises(sqlite3.DatabaseError):
                        self.engine(fault).add("synthetic rejected publication", sender="Synthetic")
                    self.assertEqual(fault.denied, 1)
                    self.assertEqual(conn.in_transaction, borrowed)
                    self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
                    self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], int(borrowed))
                    self.assertEqual(self.rows(conn), [])
                    self.assertEqual(self.checkpoint(conn), before)
                    self.assertEqual(MAINTENANCE.read_source_revision(conn), source)

    def test_style_savepoint_release_fault_preserves_add_and_invalidates_coverage(self) -> None:
        for cache in (0, 5, 100):
            conn = self.connection(cache=cache)
            fault = BoundaryFailure(conn, "style_release")
            self.engine(fault).add("synthetic retained", sender="Synthetic")
            self.assertEqual(fault.denied, 1)
            self.assertFalse(conn.in_transaction)
            self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)
            self.assertEqual(self.rows(conn), [])
            self.assertEqual((self.state(conn).outcome, self.state(conn).attempted_revision), ("pending", None))

    def test_rejected_rollback_is_not_swallowed_as_successful_source_add(self) -> None:
        conn = self.connection()

        class RefuseRollback:
            def __getattr__(self, name: str) -> object:
                return getattr(conn, name)

            def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                if sql.startswith("ROLLBACK TO truememory_style_coverage_"):
                    raise sqlite3.OperationalError("synthetic rollback refused")
                return conn.execute(sql, *args)

        with patch.dict(ADD.__globals__, _update_style_vec=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("synthetic"))):
            with self.assertRaises(MAINTENANCE._LayerRollbackFailed):
                self.engine(RefuseRollback()).add("synthetic failed source", sender="Synthetic")
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
        self.assertEqual(self.rows(conn), [])

    def test_cleanup_release_failure_preserves_borrowed_caller_and_rejects_add(self) -> None:
        conn = self.connection()
        before = self.checkpoint(conn)
        conn.execute("INSERT INTO sentinel VALUES ('synthetic caller')")

        class RefuseRelease:
            def __getattr__(self, name: str) -> object:
                return getattr(conn, name)

            def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                if sql.startswith("RELEASE truememory_style_coverage_"):
                    raise sqlite3.OperationalError("synthetic release refused")
                return conn.execute(sql, *args)

        with self.assertRaises(MAINTENANCE._LayerRollbackFailed):
            self.engine(RefuseRelease()).add("synthetic rejected", sender="Synthetic")
        self.assertTrue(conn.in_transaction)
        self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
        self.assertEqual(self.rows(conn), [])
        self.assertEqual(self.checkpoint(conn), before)
        conn.rollback()

    def test_ended_transaction_cannot_be_reported_as_a_safe_style_failure(self) -> None:
        conn = self.connection()

        def ended(*args: object, **kwargs: object) -> None:
            conn.rollback()
            raise sqlite3.OperationalError("database is locked")

        with patch.dict(ADD.__globals__, _update_style_vec=ended):
            with self.assertRaises(MAINTENANCE._LayerRollbackFailed):
                self.engine(conn).add("synthetic rejected", sender="Synthetic")
        self.assertFalse(conn.in_transaction)
        self.assertEqual(conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)

    def test_add_queries_use_only_indexed_source_and_profile_point_reads(self) -> None:
        conn = self.connection()
        statements = []
        conn.set_trace_callback(statements.append)
        self.engine(conn).add("synthetic bounded", sender="Synthetic")
        conn.set_trace_callback(None)
        reads = [sql for sql in statements if sql.startswith("SELECT") and
                 ("FROM messages " in sql or "FROM entity_style_vectors " in sql)]
        self.assertEqual(len(reads), 3)
        for sql in reads:
            self.assertIn(" WHERE ", sql)
            plans = conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
            self.assertTrue(all("SEARCH" in row[3] for row in plans), plans)
        self.assertFalse(any("COUNT(" in sql.upper() or "MAX(" in sql.upper() for sql in statements if sql.startswith("SELECT")))


if __name__ == "__main__":
    unittest.main()
