"""Actual private style routing with synthetic SQLite and stdlib-only loaders.

Run the InMemoryRouting and CoordinatorRouting classes for local checks.
FileOpenerRouting requires the isolated full-suite host and synthetic files.
"""

import os
import runpy
import sqlite3
import tempfile
import threading
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(Path(__file__).with_name("test-l0-style-incremental-747.py")))
ROUTING = runpy.run_path(str(Path(__file__).with_name("test-maintenance-engine-routing-753.py")))
STORAGE, STYLE, MAINTENANCE = BASE["STORAGE"], BASE["STYLE"], BASE["MAINTENANCE"]


def public_specs(conn: sqlite3.Connection, *_args: object, **_kwargs: object) -> tuple:
    layers = ("clusters", "summaries", "contradictions", "structured_facts", "surprise", "episodes", "landmarks", "dunbar")
    return tuple(MAINTENANCE.LayerSpec(layer, key, lambda: MAINTENANCE.make_layer_dependency(1),
        lambda _conn: None, lambda _conn: 0, connection=conn)
        for layer, key in zip(layers, MAINTENANCE._MAINTENANCE_RESULT_KEYS))


class InMemoryRouting(unittest.TestCase):
    def connection(self, *, bootstrap: bool = False) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.executescript(STORAGE._SCHEMA_SQL)
        STORAGE._initialize_maintenance_tracking(conn)
        conn.execute("CREATE TABLE sentinel(value TEXT)")
        conn.commit()
        if bootstrap:
            self.assertEqual(MAINTENANCE.run_style_maintenance(conn).outcome, "success_empty")
        return conn

    def engine(self, conn: sqlite3.Connection) -> object:
        modules = MAINTENANCE._fixture_modules
        modules["truememory.personality_style_vec"] = STYLE
        modules["truememory.rebuild_source"] = ROUTING["stdlib_module"]("rebuild_source", modules)
        module = ROUTING["load_engine"](modules)
        module._HAS_STYLE_VEC = True
        module._update_style_vec = STYLE.update_entity_style_vector_incremental
        engine = module.TrueMemoryEngine(":memory:")
        engine.conn, engine.ready = conn, True
        engine._has_consolidation = False
        engine._has_style_vec = True
        engine._maintenance_coordinator = MAINTENANCE.MaintenanceCoordinator(None)
        engine._maintenance_handle = conn
        engine._engine_test_module = module
        return engine

    def add(self, conn: sqlite3.Connection, *, sender: str = "Synthetic") -> None:
        STORAGE.insert_message(conn, {"content": "synthetic source", "sender": sender, "directive": False})
        conn.commit()

    def checkpoint(self, conn: sqlite3.Connection) -> tuple | None:
        return MAINTENANCE._read_layer_row(conn, "style_vectors")

    def rows(self, conn: sqlite3.Connection) -> list:
        return conn.execute("SELECT * FROM entity_style_vectors ORDER BY entity").fetchall()

    def no_native(self) -> object:
        original = MAINTENANCE.importlib
        return patch.object(MAINTENANCE, "importlib", types.SimpleNamespace(import_module=lambda name:
            original.import_module(name) if name == "truememory.personality_style_vec" else
            self.fail("Unexpected dependency preparation")))

    def test_style_only_dispatch_precedes_and_excludes_public_preparation(self) -> None:
        conn = self.connection()
        self.add(conn)
        coordinator = MAINTENANCE.MaintenanceCoordinator(None)
        with self.no_native(), patch.object(MAINTENANCE, "_prepare_worker_extensions", side_effect=AssertionError), \
             patch.object(MAINTENANCE, "engine_layer_specs", side_effect=AssertionError):
            report = MAINTENANCE.run_engine_maintenance(conn, coordinator, include_layers=False, include_style=True)
        self.assertEqual(report.results, ())
        self.assertTrue(report.preferences.startswith("SKIPPED"))
        self.assertEqual(report.style_result.outcome, "success")
        self.assertEqual(MAINTENANCE.maintenance_report_status(report), ("success", None))
        self.assertEqual(MAINTENANCE.observe_style(conn).health["freshness"], "current")

    def test_combined_style_publishes_before_unrelated_native_failure(self) -> None:
        conn = self.connection()
        self.add(conn)
        with patch.object(MAINTENANCE, "_prepare_worker_extensions", side_effect=RuntimeError("synthetic extension")):
            with self.assertRaises(RuntimeError):
                MAINTENANCE.run_engine_maintenance(conn, MAINTENANCE.MaintenanceCoordinator(None), include_style=True)
        self.assertEqual(MAINTENANCE.observe_style(conn).health["freshness"], "current")
        self.assertEqual(len(self.rows(conn)), 1)

    def test_bounded_readiness_and_health_do_not_read_source_or_profiles(self) -> None:
        conn = self.connection()
        def authorize(action: int, table: str | None, *_args: object) -> int:
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and table in {"messages", "entity_style_vectors"} else sqlite3.SQLITE_OK
        conn.set_authorizer(authorize)
        before = conn.total_changes
        with self.no_native():
            observation = MAINTENANCE.observe_style(conn)
            health = self.engine(conn).get_style_health()
        self.assertTrue(observation.eligible)
        self.assertEqual(health["coverage"], "unverified")
        self.assertEqual(conn.total_changes, before)
        self.assertFalse(conn.in_transaction)
        conn.set_authorizer(BASE["ACCUMULATOR"]["allow_authorized_action"])

    def test_style_plans_with_consolidation_disabled_and_submits_outside_lock(self) -> None:
        engine = self.engine(self.connection())
        calls = []
        def requested(**kwargs: object) -> bool:
            self.assertFalse(engine._write_lock.locked())
            self.assertFalse(engine.conn.in_transaction)
            calls.append(kwargs)
            return False
        with patch.object(engine._maintenance_coordinator, "request_layers", requested), \
             patch.object(MAINTENANCE, "engine_layer_specs", side_effect=AssertionError):
            engine._maybe_auto_consolidate()
        self.assertEqual(calls, [{"threshold": 25, "include_layers": False, "include_style": True}])

    def test_75_actual_adds_make_no_style_request_or_full_rebuild(self) -> None:
        conn = self.connection(bootstrap=True)
        engine = self.engine(conn)
        with patch.object(engine._maintenance_coordinator, "request_layers") as requested, \
             patch.object(STYLE, "_compute_entity_style_vectors", wraps=STYLE._compute_entity_style_vectors) as builds:
            for index in range(75):
                engine.add("synthetic append " + str(index), sender="Synthetic")
            requested.assert_not_called()
            builds.assert_not_called()
        self.assertEqual(conn.execute("SELECT message_count FROM entity_style_vectors").fetchone(), (75,))
        self.assertEqual(MAINTENANCE.observe_style(conn).state.attempted_insert_count, 75)

    def test_uncovered_append_threshold_is_24_then_25_and_corrections_are_immediate(self) -> None:
        conn = self.connection(bootstrap=True)
        for _ in range(24):
            self.add(conn)
        self.assertFalse(MAINTENANCE.observe_style(conn).eligible)
        self.assertEqual(MAINTENANCE.observe_style(conn).health["pending_reason"], "awaiting_threshold")
        self.add(conn)
        self.assertTrue(MAINTENANCE.observe_style(conn).eligible)
        self.assertEqual(MAINTENANCE.run_routed_style(conn).outcome, "success")
        conn.execute("UPDATE messages SET content='synthetic correction' WHERE id=1")
        conn.commit()
        self.assertTrue(MAINTENANCE.observe_style(conn).eligible)

    def test_hash_migration_and_owned_trigger_repair_make_current_rows_eligible(self) -> None:
        for change in ("hash", "trigger", "coverage"):
            with self.subTest(change=change):
                conn = self.connection(bootstrap=True)
                if change == "hash":
                    conn.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
                elif change == "trigger":
                    conn.execute("DROP TRIGGER truememory_style_output_update")
                else:
                    conn.execute("UPDATE maintenance_layers SET successful_coverage='unverified' WHERE layer='style_vectors'")
                conn.commit()
                self.assertTrue(MAINTENANCE.observe_style(conn).eligible)
                self.assertEqual(MAINTENANCE.run_routed_style(conn).outcome, "success_empty")
                self.assertFalse(MAINTENANCE.observe_style(conn).eligible)
                self.assertEqual(MAINTENANCE.observe_style(conn).health["coverage"], "complete")

    def test_failed_hash_migration_keeps_attempt_baseline_and_suppresses_reopen_retry(self) -> None:
        conn = self.connection(bootstrap=True)
        conn.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
        conn.commit()
        with patch.object(STYLE, "_compute_entity_style_vectors", side_effect=ValueError("synthetic failure")):
            result = MAINTENANCE.run_routed_style(conn)
        self.assertEqual(result.outcome, "failed")
        before = self.checkpoint(conn)
        self.assertFalse(MAINTENANCE.observe_style(conn).eligible)
        self.assertEqual(MAINTENANCE.run_routed_style(conn).outcome, "untrusted")
        self.assertEqual(self.checkpoint(conn), before)

    def test_failed_attempts_retry_at_25_appends_without_repeated_worker_opens(self) -> None:
        for mode in ("initial", "current", "future"):
            conn = self.connection(bootstrap=mode == "current")
            self.add(conn)
            if mode == "current":
                MAINTENANCE.run_routed_style(conn, force=True)
                conn.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
            elif mode == "future":
                conn.execute("DROP TABLE entity_style_vectors")
                conn.execute("CREATE TABLE entity_style_vectors(entity TEXT PRIMARY KEY,vector TEXT,message_count INTEGER,"
                             "updated_at TEXT,vector_sum BLOB,accumulator_version INTEGER NOT NULL DEFAULT 9)")
            conn.commit()
            if mode == "future":
                failed = MAINTENANCE.run_routed_style(conn)
            else:
                with patch.object(STYLE, "_compute_entity_style_vectors", side_effect=ValueError("synthetic build failure")):
                    failed = MAINTENANCE.run_routed_style(conn)
            self.assertEqual(failed.outcome, "unavailable" if mode == "future" else "failed")
            engine = self.engine(conn)
            coordinator = MAINTENANCE.MaintenanceCoordinator(ROOT / "synthetic-worker-not-created.sqlite")
            engine._maintenance_coordinator = coordinator
            events = []
            class Opened:
                row_factory = None
                def __getattr__(self, name: str) -> object:
                    return getattr(conn, name)
                def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                    if sql == "PRAGMA quick_check(1)":
                        events.append("check")
                    return conn.execute(sql, *args)
                def close(self) -> None:
                    events.append("close")
            class ImmediateThread:
                def __init__(self, *, target: object, **_kwargs: object) -> None:
                    self.target = target
                def start(self) -> None:
                    self.target()
            def connect(*_args: object, **_kwargs: object) -> object:
                events.append("open")
                return Opened()
            previous = 0
            with patch.object(MAINTENANCE.sqlite3, "connect", connect), \
                 patch.object(MAINTENANCE.threading, "Thread", ImmediateThread), \
                 patch.object(MAINTENANCE, "_acquire_owner", return_value=123), \
                 patch.object(MAINTENANCE, "_release_owner"):
                for count in (0, 1, 24, 25):
                    for _ in range(count - previous):
                        self.add(conn)
                    previous = count
                    with self.subTest(mode=mode, appends=count):
                        observed = MAINTENANCE.observe_style(conn)
                        if count < 25:
                            baseline = self.checkpoint(conn)
                            self.assertFalse(MAINTENANCE.run_routed_style(conn).attempted)
                            self.assertEqual(self.checkpoint(conn), baseline)
                        before_wakes = len(events)
                        for _ in range(3):
                            engine._maybe_auto_consolidate()
                        self.assertEqual(events[before_wakes:], ["open", "check", "close"] if count == 25 else [])
                        self.assertEqual(observed.eligible, count == 25)
                        self.assertEqual(observed.eligible, MAINTENANCE._eligible(observed.state, 25, False))
                        if count == 25:
                            result = coordinator._last_report.style_result
                            self.assertTrue(result.attempted)
                            self.assertEqual(result.outcome, "unavailable" if mode == "future" else "success")
                            self.assertFalse(MAINTENANCE.observe_style(conn).eligible)

    def test_failed_attempt_correction_or_dependency_repair_retries_before_25(self) -> None:
        for change in ("correction", "dependency"):
            with self.subTest(change=change):
                conn = self.connection()
                self.add(conn)
                with patch.object(STYLE, "_compute_entity_style_vectors", side_effect=ValueError("synthetic failure")):
                    self.assertEqual(MAINTENANCE.run_routed_style(conn).outcome, "failed")
                if change == "correction":
                    conn.execute("UPDATE messages SET content='synthetic correction' WHERE id=1")
                else:
                    conn.execute("DROP TRIGGER truememory_style_output_update")
                conn.commit()
                self.assertTrue(MAINTENANCE.observe_style(conn).eligible)
                self.assertEqual(MAINTENANCE.run_routed_style(conn).outcome, "success")
                self.assertFalse(MAINTENANCE.observe_style(conn).eligible)

    def test_future_default_remains_unverified_and_guarded_without_reopen_loop(self) -> None:
        conn = self.connection(bootstrap=True)
        conn.execute("DROP TABLE entity_style_vectors")
        conn.execute("CREATE TABLE entity_style_vectors(entity TEXT PRIMARY KEY,vector TEXT,message_count INTEGER,"
                     "updated_at TEXT,vector_sum BLOB,accumulator_version INTEGER NOT NULL DEFAULT 9)")
        conn.commit()
        MAINTENANCE.prepare_style_maintenance(conn)
        observation = MAINTENANCE.observe_style(conn)
        self.assertTrue(observation.eligible)
        self.assertFalse(observation.health["format_ready"])
        self.assertEqual(observation.health["coverage"], "unverified")
        schema = conn.execute("PRAGMA table_info(entity_style_vectors)").fetchall()
        result = MAINTENANCE.run_routed_style(conn)
        self.assertEqual((result.outcome, result.error_category), ("unavailable", "StyleAccumulatorUnsupported"))
        self.assertEqual(conn.execute("PRAGMA table_info(entity_style_vectors)").fetchall(), schema)
        self.assertEqual(self.rows(conn), [])
        self.assertFalse(MAINTENANCE.observe_style(conn).eligible)

    def test_hash_invalidation_writer_contention_defers_after_confirmed_rollback(self) -> None:
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                original = self.connection(bootstrap=True)
                original.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
                original.commit()
                uri = "file:synthetic-style-routing-" + uuid.uuid4().hex + "?mode=memory&cache=shared"
                conn = sqlite3.connect(uri, uri=True, timeout=0)
                contender = sqlite3.connect(uri, uri=True, timeout=0)
                self.addCleanup(conn.close)
                self.addCleanup(contender.close)
                original.backup(conn)
                before = self.checkpoint(conn)
                if borrowed:
                    conn.execute("BEGIN")
                observe = MAINTENANCE.observe_style
                def contend(*args: object, **kwargs: object) -> object:
                    result = observe(*args, **kwargs)
                    contender.execute("BEGIN IMMEDIATE")
                    contender.execute("INSERT INTO sentinel VALUES ('synthetic writer')")
                    return result
                with patch.object(MAINTENANCE, "observe_style", contend):
                    result = MAINTENANCE.run_routed_style(conn, allow_caller_transaction=borrowed)
                self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleWriterBusy"))
                self.assertEqual(conn.in_transaction, borrowed)
                self.assertEqual(self.checkpoint(conn), before)
                contender.rollback()
                result = MAINTENANCE.run_routed_style(conn, allow_caller_transaction=borrowed)
                self.assertEqual(result.outcome, "success_empty")
                if borrowed:
                    conn.rollback()
                    self.assertEqual(self.checkpoint(conn), before)

    def test_hash_invalidation_preserves_newer_failure_instead_of_forced_overwrite(self) -> None:
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                conn = self.connection(bootstrap=True)
                conn.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
                conn.commit()
                if borrowed:
                    conn.execute("BEGIN")
                expected = []
                observe = MAINTENANCE.observe_style
                def replace(*args: object, **kwargs: object) -> object:
                    result = observe(*args, **kwargs)
                    conn.execute("UPDATE maintenance_layers SET outcome='failed',error_category='NewerFailure',"
                                 "run_generation='newer-owner',attempted_insert_count=999 WHERE layer='style_vectors'")
                    if not borrowed:
                        conn.commit()
                    expected.append(self.checkpoint(conn))
                    return result
                with patch.object(MAINTENANCE, "observe_style", replace), \
                     patch.object(STYLE, "_compute_entity_style_vectors", side_effect=AssertionError):
                    result = MAINTENANCE.run_routed_style(conn, force=True, allow_caller_transaction=borrowed)
                self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleSourceChanged"))
                self.assertEqual(self.checkpoint(conn), expected[0])
                self.assertEqual(conn.in_transaction, borrowed)

    def test_hash_invalidation_does_not_hide_unknown_errors_or_lost_transactions(self) -> None:
        for boundary in ("unknown", "ended", "rollback"):
            with self.subTest(boundary=boundary):
                conn = self.connection(bootstrap=True)
                conn.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
                conn.commit()
                before = self.checkpoint(conn)
                class Fault:
                    def __getattr__(self, name: str) -> object:
                        return getattr(conn, name)

                    def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                        if sql == "UPDATE maintenance_layers SET layer=layer WHERE 0":
                            if boundary == "ended":
                                conn.rollback()
                            raise sqlite3.OperationalError("synthetic unknown" if boundary == "unknown" else "database is locked")
                        return conn.execute(sql, *args)

                    def rollback(self) -> None:
                        if boundary == "rollback":
                            raise sqlite3.OperationalError("synthetic rollback failure")
                        conn.rollback()

                expected = sqlite3.OperationalError if boundary == "unknown" else MAINTENANCE._LayerRollbackFailed
                # The observation's read-only rollback must remain ordinary;
                # inject only at the subsequent writer boundary.
                with patch.object(MAINTENANCE, "observe_style", return_value=MAINTENANCE.observe_style(conn)):
                    with self.assertRaises(expected):
                        MAINTENANCE.run_routed_style(Fault())
                conn.rollback()
                self.assertEqual(self.checkpoint(conn), before)

    def test_unsupported_triggers_do_not_schedule_or_mutate_on_any_guarded_table(self) -> None:
        for table in ("entity_style_vectors", "maintenance_layers", "metadata"):
            for temporary in (False, True):
                with self.subTest(table=table, temporary=temporary):
                    conn = self.connection()
                    conn.execute("CREATE " + ("TEMP " if temporary else "") + "TRIGGER synthetic_unsupported AFTER INSERT ON main."
                                 + table + " BEGIN INSERT INTO sentinel VALUES ('unexpected mutation'); END")
                    conn.commit()
                    engine = self.engine(conn)
                    before = conn.total_changes
                    with patch.object(engine._maintenance_coordinator, "request_layers") as requested:
                        engine._maybe_auto_consolidate()
                    requested.assert_not_called()
                    report = MAINTENANCE.run_engine_maintenance(conn, engine._maintenance_coordinator,
                                                               include_layers=False, include_style=True)
                    self.assertEqual(report.style_result.error_category, "StyleTriggersUnsupported")
                    self.assertEqual(report.style_result.outcome, "deferred")
                    self.assertEqual(conn.total_changes, before)
                    self.assertIsNone(self.checkpoint(conn))

    def test_source_readiness_rejects_missing_changed_shadowed_and_unready_tracking(self) -> None:
        for change in ("missing", "changed", "shadow", "unready", "token", "schema"):
            with self.subTest(change=change):
                conn = self.connection()
                if change == "missing":
                    conn.execute("DROP TRIGGER messages_maintenance_ai")
                elif change == "changed":
                    conn.execute("DROP TRIGGER messages_maintenance_ai")
                    conn.execute("CREATE TRIGGER messages_maintenance_ai AFTER INSERT ON messages BEGIN SELECT 1; END")
                elif change == "shadow":
                    conn.execute("CREATE TEMP TRIGGER messages_maintenance_ai AFTER INSERT ON main.messages BEGIN SELECT 1; END")
                elif change == "unready":
                    conn.execute("UPDATE maintenance_source_state SET tracking_ready=0")
                elif change == "token":
                    conn.execute("DELETE FROM maintenance_source_state")
                else:
                    conn.execute("ALTER TABLE messages RENAME COLUMN id TO synthetic_id")
                conn.commit()
                before = conn.total_changes
                observation = MAINTENANCE.observe_style(conn)
                self.assertFalse(observation.eligible)
                self.assertEqual(observation.health["error_category"], "StyleSourceUnavailable")
                self.assertEqual(MAINTENANCE.run_routed_style(conn).outcome, "unavailable")
                self.assertEqual(conn.total_changes, before)

    def test_canonical_source_rechecked_under_preparation_and_diagnostic_writer(self) -> None:
        for enrolled in (False, True):
            for borrowed in (False, True):
                with self.subTest(enrolled=enrolled, borrowed=borrowed):
                    conn = self.connection(bootstrap=enrolled)
                    self.add(conn)
                    if enrolled:
                        MAINTENANCE.run_routed_style(conn, force=True)
                    baseline, output = self.checkpoint(conn), self.rows(conn)
                    if borrowed:
                        conn.execute("INSERT INTO sentinel VALUES ('caller pending')")
                    observe = MAINTENANCE.observe_style
                    def invalidate(*args: object, **kwargs: object) -> object:
                        result = observe(*args, **kwargs)
                        conn.execute("DROP TRIGGER messages_maintenance_ai")
                        if not borrowed:
                            conn.commit()
                        return result
                    before = conn.total_changes
                    with patch.object(MAINTENANCE, "observe_style", invalidate):
                        result = MAINTENANCE.run_routed_style(conn, force=True, allow_caller_transaction=borrowed)
                    self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleSourceChanged"))
                    self.assertEqual(conn.total_changes, before)
                    self.assertEqual(self.checkpoint(conn), baseline)
                    self.assertEqual(self.rows(conn), output)
                    self.assertEqual(conn.in_transaction, borrowed)
                    self.assertEqual(MAINTENANCE.observe_style(conn).health["error_category"], "StyleSourceUnavailable")
                    if borrowed:
                        self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [("caller pending",)])
                        conn.rollback()
                        self.assertTrue(MAINTENANCE.observe_style(conn).health["available"] if enrolled else
                                        MAINTENANCE.observe_style(conn).eligible)

    def test_canonical_source_rechecked_before_output_and_rollback_remains_required(self) -> None:
        for boundary in ("publication", "rollback_failure"):
            with self.subTest(boundary=boundary):
                conn = self.connection(bootstrap=True)
                self.add(conn)
                MAINTENANCE.run_routed_style(conn, force=True)
                baseline, output = self.checkpoint(conn), self.rows(conn)
                compute = STYLE._compute_entity_style_vectors
                def invalidate(*args: object, **kwargs: object) -> object:
                    result = compute(*args, **kwargs)
                    conn.execute("DROP TRIGGER messages_maintenance_ai")
                    return result
                class Connection:
                    def __getattr__(self, name: str) -> object:
                        return getattr(conn, name)
                    def rollback(self) -> None:
                        missing = conn.execute("SELECT 1 FROM sqlite_master WHERE name='messages_maintenance_ai'").fetchone() is None
                        if boundary == "rollback_failure" and missing:
                            raise sqlite3.OperationalError("synthetic rollback failure")
                        conn.rollback()
                with patch.object(STYLE, "_compute_entity_style_vectors", invalidate):
                    if boundary == "rollback_failure":
                        with self.assertRaises(MAINTENANCE._LayerRollbackFailed):
                            MAINTENANCE.run_routed_style(Connection(), force=True)
                        conn.rollback()
                    else:
                        result = MAINTENANCE.run_routed_style(Connection(), force=True)
                        self.assertEqual((result.outcome, result.error_category), ("deferred", "StyleSourceChanged"))
                        self.assertEqual(self.checkpoint(conn), baseline)
                self.assertEqual(self.rows(conn), output)
                self.assertFalse(conn.in_transaction)

    def test_current_style_rechecks_source_in_every_baseline_snapshot(self) -> None:
        for boundary in ("observation", "first_baseline", "cancelled_baseline"):
            for borrowed in (False, True):
                with self.subTest(boundary=boundary, borrowed=borrowed):
                    conn = self.connection(bootstrap=True)
                    self.add(conn)
                    MAINTENANCE.run_routed_style(conn, force=True)
                    baseline, output = self.checkpoint(conn), self.rows(conn)
                    self.assertEqual(MAINTENANCE.observe_style(conn).health["freshness"], "current")
                    if borrowed:
                        conn.execute("INSERT INTO sentinel VALUES ('caller pending')")
                    cancel = threading.Event()
                    original = (MAINTENANCE.observe_style if boundary == "observation" else
                                MAINTENANCE._read_layer_run_baseline)
                    changed = False
                    def invalidate(*args: object, **kwargs: object) -> object:
                        nonlocal changed
                        result = original(*args, **kwargs)
                        if not changed:
                            changed = True
                            conn.execute("DROP TRIGGER messages_maintenance_ai")
                            if not borrowed:
                                conn.commit()
                            if boundary == "cancelled_baseline":
                                cancel.set()
                        return result
                    def read_only(action: int, table: str | None, *_args: object) -> int:
                        if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE} and table in {
                            "messages", "entity_style_vectors", "maintenance_layers", "metadata",
                        }:
                            return sqlite3.SQLITE_DENY
                        return sqlite3.SQLITE_OK
                    attribute = "observe_style" if boundary == "observation" else "_read_layer_run_baseline"
                    before = conn.total_changes
                    conn.set_authorizer(read_only)
                    try:
                        with patch.object(MAINTENANCE, attribute, invalidate), \
                             patch.object(STYLE, "_compute_entity_style_vectors", side_effect=AssertionError):
                            result = MAINTENANCE.run_routed_style(conn, force=False, cancel=cancel,
                                                                 allow_caller_transaction=borrowed)
                    finally:
                        conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                    self.assertTrue(changed)
                    self.assertEqual((result.outcome, result.error_category, result.coverage, result.attempted),
                                     ("deferred", "StyleSourceChanged", "unverified", False))
                    self.assertEqual(conn.total_changes, before)
                    self.assertEqual(self.checkpoint(conn), baseline)
                    self.assertEqual(self.rows(conn), output)
                    self.assertEqual(conn.in_transaction, borrowed)
                    self.assertEqual(MAINTENANCE.observe_style(conn).health["error_category"], "StyleSourceUnavailable")
                    if borrowed:
                        self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [("caller pending",)])
                        conn.rollback()
                        self.assertEqual(MAINTENANCE.observe_style(conn).health["freshness"], "current")

    def test_current_style_baseline_is_read_only_and_preserves_caller_snapshot(self) -> None:
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                conn = self.connection(bootstrap=True)
                baseline = self.checkpoint(conn)
                if borrowed:
                    conn.execute("INSERT INTO sentinel VALUES ('caller pending')")
                observed = []
                validate = MAINTENANCE._validate_routed_style_source
                def in_snapshot(connection: sqlite3.Connection) -> None:
                    observed.append(connection.in_transaction)
                    validate(connection)
                def read_only(action: int, table: str | None, *_args: object) -> int:
                    if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}:
                        return sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_READ and table in {"messages", "entity_style_vectors"}:
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK
                before = conn.total_changes
                conn.set_authorizer(read_only)
                try:
                    with patch.object(MAINTENANCE, "_validate_routed_style_source", in_snapshot), \
                         patch.object(STYLE, "_compute_entity_style_vectors", side_effect=AssertionError):
                        result = MAINTENANCE.run_routed_style(conn, allow_caller_transaction=borrowed)
                finally:
                    conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                self.assertEqual((result.outcome, result.coverage, result.attempted), ("current", "complete", False))
                self.assertTrue(observed)
                self.assertTrue(all(observed))
                self.assertEqual(conn.total_changes, before)
                self.assertEqual(self.checkpoint(conn), baseline)
                self.assertEqual(conn.in_transaction, borrowed)
                if borrowed:
                    self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [("caller pending",)])
                    conn.rollback()

    def test_late_complete_readiness_loss_defers_without_an_attempt_and_can_recover(self) -> None:
        changes = {"hash": "StyleHashPending", "format": "StyleAccumulatorUnsupported",
                   "output_trigger": "StyleTrackingUnprepared", "checkpoint": "StyleTrackingUnprepared"}
        changes.update({f"{schema}_{table}": "StyleTriggersUnsupported" for schema in ("main", "temp")
                        for table in ("entity_style_vectors", "maintenance_layers", "metadata")})
        for change, category in changes.items():
            boundaries = ("observation", "preparation", "first_baseline", "cancelled_baseline") if change in {
                "hash", "format",
            } else ("preparation", "first_baseline", "cancelled_baseline")
            for boundary in boundaries:
                for borrowed in (False, True):
                    with self.subTest(change=change, boundary=boundary, borrowed=borrowed):
                        conn = self.connection(bootstrap=True)
                        self.add(conn)
                        self.assertEqual(MAINTENANCE.run_routed_style(conn, force=True).outcome, "success")
                        baseline, output = self.checkpoint(conn), self.rows(conn)
                        if borrowed:
                            conn.execute("INSERT INTO sentinel VALUES ('caller pending')")
                        cancel = threading.Event()
                        attribute = {"observation": "observe_style", "preparation": "_prepare_routed_style_maintenance"}.get(
                            boundary, "_read_layer_run_baseline")
                        original = getattr(MAINTENANCE, attribute)
                        changed = []
                        def read_only(action: int, table: str | None, *_args: object) -> int:
                            if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE} and table in {
                                "messages", "entity_style_vectors", "maintenance_layers", "metadata",
                            }:
                                return sqlite3.SQLITE_DENY
                            if action == sqlite3.SQLITE_READ and table in {"messages", "entity_style_vectors"}:
                                return sqlite3.SQLITE_DENY
                            return sqlite3.SQLITE_OK
                        def invalidate(*args: object, **kwargs: object) -> object:
                            result = original(*args, **kwargs)
                            if not changed:
                                if change == "hash":
                                    conn.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
                                elif change == "format":
                                    conn.execute("ALTER TABLE entity_style_vectors DROP COLUMN accumulator_version")
                                    conn.execute("ALTER TABLE entity_style_vectors ADD COLUMN accumulator_version INTEGER DEFAULT 9")
                                elif change == "output_trigger":
                                    conn.execute("DROP TRIGGER truememory_style_output_update")
                                elif change == "checkpoint":
                                    conn.execute("DELETE FROM maintenance_layers WHERE layer='style_vectors'")
                                else:
                                    schema, table = change.split("_", 1)
                                    conn.execute(f"CREATE {'TEMP ' if schema == 'temp' else ''}TRIGGER extra_proof "
                                                 f"AFTER UPDATE ON main.{table} BEGIN SELECT 1; END")
                                if not borrowed:
                                    conn.commit()
                                changed.append((self.checkpoint(conn), self.rows(conn), conn.total_changes))
                                if boundary == "cancelled_baseline":
                                    cancel.set()
                                conn.set_authorizer(read_only)
                            return result
                        try:
                            with patch.object(MAINTENANCE, attribute, invalidate), self.no_native(), \
                                 patch.object(STYLE, "_compute_entity_style_vectors", side_effect=AssertionError):
                                result = MAINTENANCE.run_routed_style(conn, cancel=cancel,
                                                                     allow_caller_transaction=borrowed)
                        finally:
                            conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                        self.assertEqual(len(changed), 1)
                        self.assertEqual((result.outcome, result.error_category, result.coverage, result.attempted),
                                         ("deferred", category, "unverified", False))
                        self.assertEqual((self.checkpoint(conn), self.rows(conn), conn.total_changes), changed[0])
                        self.assertEqual(conn.in_transaction, borrowed)
                        if change in {"hash", "output_trigger", "checkpoint"}:
                            if not borrowed:
                                self.assertTrue(MAINTENANCE.observe_style(conn).eligible)
                            repaired = MAINTENANCE.run_routed_style(conn, allow_caller_transaction=borrowed)
                            self.assertEqual((repaired.outcome, repaired.coverage), ("success", "complete"))
                            self.assertTrue(repaired.attempted)
                        elif change == "format":
                            failed = MAINTENANCE.run_routed_style(conn, allow_caller_transaction=borrowed)
                            self.assertEqual((failed.outcome, failed.error_category, failed.attempted),
                                             ("unavailable", "StyleAccumulatorUnsupported", True))
                            attempt = self.checkpoint(conn)
                            self.assertFalse(MAINTENANCE.run_routed_style(conn, allow_caller_transaction=borrowed).attempted)
                            self.assertEqual(self.checkpoint(conn), attempt)
                        if borrowed:
                            self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [("caller pending",)])
                            conn.rollback()
                            self.assertEqual((self.checkpoint(conn), self.rows(conn)), (baseline, output))

    def test_failed_builder_with_old_complete_coverage_keeps_its_retry_budget(self) -> None:
        for change, failure in ((change, failure) for change in ("hash", "format")
                                for failure in (ValueError, MAINTENANCE.LayerUnavailableError)):
            for borrowed in (False, True):
                with self.subTest(change=change, failure=failure.__name__, borrowed=borrowed):
                    conn = self.connection(bootstrap=True)
                    self.add(conn)
                    self.assertEqual(MAINTENANCE.run_routed_style(conn, force=True).outcome, "success")
                    with patch.object(STYLE, "_compute_entity_style_vectors", side_effect=failure("synthetic failure")):
                        self.assertEqual(MAINTENANCE.run_routed_style(conn, force=True).outcome,
                                         "failed" if failure is ValueError else "unavailable")
                    baseline = self.checkpoint(conn)
                    self.assertEqual(baseline[MAINTENANCE._LAYER_ROW_COLUMNS.index("successful_coverage")], "complete")
                    if change == "hash":
                        conn.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
                    else:
                        conn.execute("ALTER TABLE entity_style_vectors DROP COLUMN accumulator_version")
                        conn.execute("ALTER TABLE entity_style_vectors ADD COLUMN accumulator_version INTEGER DEFAULT 9")
                    conn.commit()
                    if borrowed:
                        conn.execute("INSERT INTO sentinel VALUES ('caller pending')")
                    previous = 0
                    for count in (0, 1, 24, 25):
                        for _ in range(count - previous):
                            STORAGE.insert_message(conn, {"content": "synthetic append", "sender": "Synthetic"})
                        if not borrowed:
                            conn.commit()
                        previous = count
                        result = MAINTENANCE.run_routed_style(conn, allow_caller_transaction=borrowed)
                        self.assertEqual(result.attempted, count == 25)
                        if count < 25:
                            self.assertEqual(result.coverage, "unverified")
                            self.assertEqual(result.error_category, failure.__name__)
                            self.assertEqual(self.checkpoint(conn), baseline)
                        else:
                            self.assertEqual(result.outcome, "success" if change == "hash" else "unavailable")
                        self.assertEqual(conn.in_transaction, borrowed)
                    if borrowed:
                        self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [("caller pending",)])
                        conn.rollback()
                        self.assertEqual(self.checkpoint(conn), baseline)

    def test_retry_state_reload_preserves_known_unverified_failure_coverage(self) -> None:
        coverage_index = MAINTENANCE._LAYER_ROW_COLUMNS.index("successful_coverage")
        for change in ("intact", "hash", "format"):
            for failure in (ValueError, MAINTENANCE.LayerUnavailableError):
                for borrowed in (False, True):
                    for force in (False, True):
                        with self.subTest(change=change, failure=failure.__name__, borrowed=borrowed, force=force):
                            conn = self.connection(bootstrap=True)
                            self.add(conn)
                            self.assertEqual(MAINTENANCE.run_routed_style(conn, force=True).outcome, "success")
                            with patch.object(STYLE, "_compute_entity_style_vectors", side_effect=failure("synthetic failure")):
                                prior = MAINTENANCE.run_routed_style(conn, force=True)
                            self.assertEqual(prior.coverage, "complete")
                            baseline = self.checkpoint(conn)
                            if change == "hash":
                                conn.execute("UPDATE metadata SET value='1' WHERE key='style_vec_hash_version'")
                            elif change == "format":
                                conn.execute("ALTER TABLE entity_style_vectors DROP COLUMN accumulator_version")
                                conn.execute("ALTER TABLE entity_style_vectors ADD COLUMN accumulator_version INTEGER DEFAULT 9")
                            conn.commit()
                            output = self.rows(conn)
                            if borrowed:
                                conn.execute("INSERT INTO sentinel VALUES ('caller pending')")
                            if not force:
                                for _ in range(25):
                                    STORAGE.insert_message(conn, {"content": "synthetic append", "sender": "Synthetic"})
                                if not borrowed:
                                    conn.commit()
                            compute = STYLE._compute_entity_style_vectors
                            with patch.object(STYLE, "_compute_entity_style_vectors", wraps=compute,
                                              side_effect=failure("synthetic failure") if change == "intact" else None):
                                result = MAINTENANCE.run_routed_style(conn, force=force, allow_caller_transaction=borrowed)
                            expected = ("success" if change == "hash" else "unavailable" if change == "format"
                                        or failure is MAINTENANCE.LayerUnavailableError else "failed")
                            self.assertEqual((result.outcome, result.coverage, result.attempted),
                                             (expected, "unverified" if change == "format" else "complete", True))
                            self.assertEqual(self.checkpoint(conn)[coverage_index], "complete")
                            self.assertEqual(conn.in_transaction, borrowed)
                            if change != "hash":
                                self.assertEqual(self.rows(conn), output)
                            if borrowed:
                                self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [("caller pending",)])
                                conn.rollback()
                                self.assertEqual(self.checkpoint(conn), baseline)
                                self.assertEqual(self.rows(conn), output)

    def test_canonical_style_rejects_preexisting_and_late_temp_table_shadows(self) -> None:
        for table in ("messages", "maintenance_source_state", "maintenance_layers", "metadata", "entity_style_vectors"):
            for kind in ("TABLE", "VIEW"):
                for late in (False, True):
                    for borrowed in (False, True):
                        with self.subTest(table=table, kind=kind, late=late, borrowed=borrowed):
                            conn = self.connection(bootstrap=True)
                            baseline = self.checkpoint(conn)
                            if borrowed:
                                conn.execute("INSERT INTO sentinel VALUES ('caller pending')")
                            def shadow() -> None:
                                conn.execute(f"CREATE TEMP {kind} {table.upper()} AS SELECT * FROM main.{table}")
                                if not borrowed:
                                    conn.commit()
                            prepare = MAINTENANCE._prepare_routed_style_maintenance
                            def after_prepare(*args: object, **kwargs: object) -> None:
                                prepare(*args, **kwargs)
                                shadow()
                            if not late:
                                shadow()
                            before = conn.total_changes
                            with patch.object(STYLE, "_compute_entity_style_vectors", side_effect=AssertionError), self.no_native():
                                if late:
                                    with patch.object(MAINTENANCE, "_prepare_routed_style_maintenance", after_prepare):
                                        result = MAINTENANCE.run_routed_style(conn, allow_caller_transaction=borrowed)
                                else:
                                    result = MAINTENANCE.run_routed_style(conn, allow_caller_transaction=borrowed)
                            self.assertEqual((result.error_category, result.coverage, result.attempted),
                                             ("StyleSourceChanged" if late else "StyleSourceUnavailable", "unverified", False))
                            self.assertEqual(conn.total_changes, before)
                            self.assertEqual(conn.execute("SELECT " + ",".join(MAINTENANCE._LAYER_ROW_COLUMNS)
                                + " FROM main.maintenance_layers WHERE layer='style_vectors'").fetchone(), baseline)
                            self.assertEqual(conn.in_transaction, borrowed)
                            if borrowed:
                                self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [("caller pending",)])
                                conn.rollback()

    def test_canonical_style_rejects_case_variant_source_trigger_shadows(self) -> None:
        for late in (False, True):
            for borrowed in (False, True):
                with self.subTest(late=late, borrowed=borrowed):
                    conn = self.connection(bootstrap=True)
                    baseline = self.checkpoint(conn)
                    if borrowed:
                        conn.execute("INSERT INTO sentinel VALUES ('caller pending')")
                    def shadow() -> None:
                        conn.execute("CREATE TEMP TRIGGER MESSAGES_MAINTENANCE_AI AFTER INSERT ON main.messages BEGIN SELECT 1; END")
                        if not borrowed:
                            conn.commit()
                    prepare = MAINTENANCE._prepare_routed_style_maintenance
                    def after_prepare(*args: object, **kwargs: object) -> None:
                        prepare(*args, **kwargs)
                        shadow()
                    if not late:
                        shadow()
                    before = conn.total_changes
                    with patch.object(STYLE, "_compute_entity_style_vectors", side_effect=AssertionError):
                        if late:
                            with patch.object(MAINTENANCE, "_prepare_routed_style_maintenance", after_prepare):
                                result = MAINTENANCE.run_routed_style(conn, allow_caller_transaction=borrowed)
                        else:
                            result = MAINTENANCE.run_routed_style(conn, allow_caller_transaction=borrowed)
                    self.assertEqual((result.error_category, result.coverage, result.attempted),
                                     ("StyleSourceChanged" if late else "StyleSourceUnavailable", "unverified", False))
                    self.assertEqual(conn.total_changes, before)
                    self.assertEqual(self.checkpoint(conn), baseline)
                    self.assertEqual(conn.in_transaction, borrowed)
                    if borrowed:
                        self.assertEqual(conn.execute("SELECT * FROM sentinel").fetchall(), [("caller pending",)])
                        conn.rollback()

    def test_current_style_read_rollback_failure_is_not_a_safe_deferral(self) -> None:
        for failure in ("raises", "no_op"):
            with self.subTest(failure=failure):
                conn = self.connection(bootstrap=True)
                baseline = self.checkpoint(conn)
                armed = False
                observe = MAINTENANCE.observe_style
                def observed(*args: object, **kwargs: object) -> object:
                    nonlocal armed
                    result = observe(*args, **kwargs)
                    armed = True
                    return result
                class Connection:
                    def __getattr__(self, name: str) -> object:
                        return getattr(conn, name)
                    def rollback(self) -> None:
                        if armed:
                            if failure == "raises":
                                raise sqlite3.OperationalError("synthetic read rollback failure")
                            return
                        conn.rollback()
                with patch.object(MAINTENANCE, "observe_style", observed):
                    with self.assertRaises(MAINTENANCE._LayerRollbackFailed):
                        MAINTENANCE.run_routed_style(Connection())
                self.assertTrue(conn.in_transaction)
                conn.rollback()
                self.assertEqual(self.checkpoint(conn), baseline)

    def test_in_memory_pending_and_caller_transaction_do_not_start_workers(self) -> None:
        engine = self.engine(self.connection())
        with patch.object(MAINTENANCE.threading, "Thread", side_effect=AssertionError):
            engine._maybe_auto_consolidate()
        self.assertEqual(engine._maintenance_coordinator.status[0], "pending_in_memory")
        engine.conn.execute("INSERT INTO sentinel VALUES ('caller')")
        with patch.object(engine._maintenance_coordinator, "request_layers") as requested:
            engine._maybe_auto_consolidate()
        requested.assert_not_called()
        self.assertEqual(engine.get_style_health()["pending_reason"], "pending_caller_commit")
        engine.conn.rollback()

    def test_borrowed_success_rollback_preserves_failure_and_original_handle(self) -> None:
        conn = self.connection(bootstrap=True)
        coordinator = MAINTENANCE.MaintenanceCoordinator(None)
        failure = MAINTENANCE.LayerResult("style_vectors", "style_vectors", "failed", 0, 0.0, "SyntheticFailure", True, "unverified")
        token = coordinator._begin_observation()
        old_report = MAINTENANCE.MaintenanceReport((), "SKIPPED", failure)
        coordinator._finish_observation(token, old_report, "failed", "SyntheticFailure")
        before = self.checkpoint(conn)
        conn.execute("INSERT INTO sentinel VALUES ('caller')")
        with MAINTENANCE.maintenance_observation(coordinator, borrowed=True) as reports:
            report = MAINTENANCE.run_engine_maintenance(conn, coordinator, force=True, include_layers=False,
                                                       include_style=True, allow_caller_transaction=True)
            reports.append(report)
        self.assertEqual(MAINTENANCE.maintenance_report_status(report)[0], "pending")
        self.assertIs(coordinator._last_report, old_report)
        self.assertEqual(MAINTENANCE.style_health(conn, coordinator)["last_error"], "SyntheticFailure")
        self.assertTrue(conn.in_transaction)
        conn.rollback()
        self.assertEqual(self.checkpoint(conn), before)
        self.assertEqual(coordinator.status, ("failed", "SyntheticFailure"))
        self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone(), (0,))

    def test_historical_success_cannot_hide_new_invalidation_and_committed_success_clears_health(self) -> None:
        conn = self.connection(bootstrap=True)
        self.add(conn)
        coordinator = MAINTENANCE.MaintenanceCoordinator(None)
        with MAINTENANCE.maintenance_observation(coordinator) as reports:
            reports.append(MAINTENANCE.run_engine_maintenance(conn, coordinator, force=True, include_layers=False, include_style=True))
        conn.execute("DELETE FROM entity_style_vectors")
        conn.commit()
        self.assertNotEqual(MAINTENANCE.style_health(conn, coordinator)["freshness"], "current")
        token = coordinator._begin_observation()
        coordinator._finish_observation(token, None, "failed", "SyntheticFailure")
        self.assertEqual(MAINTENANCE.run_routed_style(conn).outcome, "success")
        self.assertNotIn("last_error", MAINTENANCE.style_health(conn, coordinator))

    def test_explicit_consolidate_preserves_nine_keys_and_original_connection(self) -> None:
        conn = self.connection()
        engine = self.engine(conn)
        importer = types.SimpleNamespace(import_module=lambda name: STYLE if name == "truememory.personality_style_vec"
                                         else types.SimpleNamespace(extract_preferences=lambda _conn: None))
        with patch.object(MAINTENANCE, "engine_layer_specs", public_specs), patch.object(MAINTENANCE, "importlib", importer), \
             patch.object(MAINTENANCE, "_installed_dependency", return_value=None):
            result = engine.consolidate()
            with patch.object(engine, "get_clustering_health", return_value={"status": "synthetic"}):
                stats = engine.get_stats()
        self.assertEqual(set(result), set(MAINTENANCE._MAINTENANCE_RESULT_KEYS))
        self.assertEqual(len(result), 9)
        self.assertEqual(len(STORAGE._MAINTENANCE_LAYERS), 8)
        self.assertEqual(len(stats["maintenance"]["layers"]), 8)
        self.assertNotIn("style_vectors", stats["maintenance"]["layers"])
        self.assertEqual(stats["maintenance"]["style"]["outcome"], "success_empty")
        self.assertIs(engine.conn, conn)
        self.assertEqual(engine._maintenance_coordinator._last_report.style_result.outcome, "success_empty")
        self.assertFalse(conn.in_transaction)

    def test_explicit_style_failure_changes_aggregate_and_private_health(self) -> None:
        conn = self.connection()
        engine = self.engine(conn)
        importer = types.SimpleNamespace(import_module=lambda name: STYLE if name == "truememory.personality_style_vec"
                                         else types.SimpleNamespace(extract_preferences=lambda _conn: None))
        with patch.object(MAINTENANCE, "engine_layer_specs", public_specs), patch.object(MAINTENANCE, "importlib", importer), \
             patch.object(MAINTENANCE, "_installed_dependency", return_value=None), \
             patch.object(STYLE, "_compute_entity_style_vectors", side_effect=ValueError("synthetic failure")):
            result = engine.consolidate()
        self.assertEqual(len(result), 9)
        self.assertEqual(engine._maintenance_coordinator.status, ("failed", "ValueError"))
        self.assertEqual(engine.get_style_health()["outcome"], "failed")

    def test_owned_manual_preserves_foreground_temp_rejection_on_dedicated_handle(self) -> None:
        conn = self.connection()
        self.add(conn)
        conn.execute("CREATE TEMP TRIGGER synthetic_foreground AFTER INSERT ON main.metadata BEGIN SELECT 1; END")
        conn.commit()
        engine = self.engine(conn)
        coordinator = MAINTENANCE.MaintenanceCoordinator(ROOT / "synthetic-manual-not-created.sqlite")
        engine._maintenance_coordinator = coordinator
        dedicated = self.connection()
        conn.backup(dedicated)
        self.assertFalse(MAINTENANCE._style_extra_triggers(dedicated))
        events = []
        class Owned:
            def __getattr__(self, name: str) -> object:
                return getattr(dedicated, name)
            def close(self) -> None:
                events.append("close")
        importer = types.SimpleNamespace(import_module=lambda name: STYLE if name == "truememory.personality_style_vec"
                                         else types.SimpleNamespace(extract_preferences=lambda _conn: None))
        with patch.object(engine._engine_test_module, "create_db", return_value=Owned()), \
             patch.object(MAINTENANCE, "engine_layer_specs", public_specs), patch.object(MAINTENANCE, "importlib", importer), \
             patch.object(MAINTENANCE, "_prepare_worker_extensions", return_value=None), \
             patch.object(MAINTENANCE, "_installed_dependency", return_value=None), \
             patch.object(MAINTENANCE, "_acquire_owner", return_value=123), \
             patch.object(MAINTENANCE, "_release_owner", side_effect=lambda _fd: events.append("release")):
            result = engine.consolidate()
        self.assertEqual(len(result), 9)
        self.assertEqual(len(coordinator._last_report.results), 8)
        self.assertEqual(coordinator._last_report.style_result.error_category, "StyleTriggersUnsupported")
        self.assertEqual(coordinator._last_report.style_result.outcome, "deferred")
        self.assertEqual(events, ["close", "release"])
        self.assertEqual(self.rows(dedicated), [])
        self.assertIsNone(self.checkpoint(dedicated))

    def test_delete_normalizes_only_the_style_entity_key(self) -> None:
        conn = self.connection(bootstrap=True)
        engine = self.engine(conn)
        engine.add("synthetic source", sender="Synthetic")
        self.assertEqual(conn.execute("SELECT entity FROM entity_style_vectors").fetchone(), ("synthetic",))
        conn.executemany("INSERT INTO entity_profiles(entity) VALUES (?)", [("Synthetic",), ("synthetic",)])
        conn.executemany("INSERT INTO summaries(entity,summary) VALUES (?,'synthetic summary')", [("Synthetic",), ("synthetic",)])
        conn.commit()
        namespace = STORAGE.__dict__
        original = namespace["__builtins__"]["__import__"]
        def safe_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "truememory.tier_config":
                return types.SimpleNamespace(VALID_TIER_GROUPS=())
            return original(name, *args, **kwargs)
        with patch.dict(namespace["__builtins__"], __import__=safe_import):
            self.assertTrue(STORAGE.delete_message(conn, 1))
        self.assertEqual(self.rows(conn), [])
        self.assertEqual(conn.execute("SELECT entity FROM entity_profiles").fetchall(), [("synthetic",)])
        self.assertEqual(conn.execute("SELECT entity FROM summaries").fetchall(), [("synthetic",)])
        self.assertTrue(MAINTENANCE.observe_style(conn).eligible)

    def test_actual_bulk_style_stage_runs_once_and_reports_borrowed_success(self) -> None:
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                conn = self.connection()
                engine = self.engine(conn)
                module = engine._engine_test_module
                module._HAS_CONSOLIDATION = False
                def load(connection: sqlite3.Connection, _path: Path) -> int:
                    STORAGE.insert_message(connection, {"content": "synthetic bulk", "sender": "Synthetic", "directive": False})
                    if not borrowed:
                        connection.commit()
                    return 1
                with patch.object(module, "create_db", return_value=conn), \
                     patch.object(module, "load_messages_from_file", load), \
                     patch.object(engine._maintenance_coordinator, "observe_clustering_outcome"), \
                     patch.object(STYLE, "_compute_entity_style_vectors", wraps=STYLE._compute_entity_style_vectors) as builds:
                    stats = engine.ingest("synthetic-inline-boundary")
                    self.assertEqual(builds.call_count, 1)
                    self.assertTrue(stats["build_style_vectors"].startswith("1 vectors"))
                    self.assertEqual("pending caller commit" in stats["build_style_vectors"], borrowed)
                    if not borrowed:
                        engine.add("synthetic after bulk", sender="Synthetic")
                        self.assertEqual(builds.call_count, 1)
                        self.assertEqual(conn.execute("SELECT message_count FROM entity_style_vectors").fetchone(), (2,))
                if borrowed:
                    self.assertIsNone(engine._maintenance_coordinator._last_report)
                    conn.rollback()
                    self.assertIsNone(self.checkpoint(conn))
                    self.assertEqual(self.rows(conn), [])

    def test_deprecated_open_schedules_without_synchronous_style_build_or_hash_claim(self) -> None:
        conn = self.connection()
        probe = self.connection()
        engine = self.engine(conn)
        module = engine._engine_test_module
        engine.db_path = (ROOT / "synthetic-existing-not-created.sqlite")
        exists = Path.exists
        closed_owners = []
        owner_os = types.SimpleNamespace(getpid=os.getpid, fspath=os.fspath, close=closed_owners.append)
        with patch.object(Path, "exists", lambda path: True if path == engine.db_path else exists(path)), \
             patch.object(module.sqlite3, "connect", side_effect=[probe, conn]) as opened, \
             patch.object(MAINTENANCE, "try_file_lock", return_value=719), \
             patch.object(MAINTENANCE, "os", owner_os), \
             patch.object(engine, "_purge_legacy_entity_profile_summaries"), \
             patch.object(engine, "_maybe_auto_consolidate") as planned, \
             patch.object(STYLE, "_compute_entity_style_vectors", side_effect=AssertionError):
            with self.assertWarns(DeprecationWarning):
                self.assertIs(engine.open(rebuild_vectors=False), engine)
        planned.assert_called_once()
        self.assertIs(engine.conn, conn)
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(closed_owners, [719])
        with self.assertRaises(sqlite3.ProgrammingError):
            probe.execute("SELECT 1")
        self.assertIsNone(conn.execute("SELECT value FROM metadata WHERE key='style_vec_hash_version'").fetchone())
        self.assertIsNone(self.checkpoint(conn))

    def test_manual_pending_result_and_health_do_not_clear_prior_completed_failure(self) -> None:
        conn = self.connection(bootstrap=True)
        engine = self.engine(conn)
        coordinator = engine._maintenance_coordinator
        token = coordinator._begin_observation()
        coordinator._finish_observation(token, MAINTENANCE.MaintenanceReport((), "SKIPPED"), "failed", "PriorFailure")
        conn.execute("INSERT INTO sentinel VALUES ('caller')")
        personality = types.SimpleNamespace(extract_preferences=lambda _conn: None)
        importer = types.SimpleNamespace(import_module=lambda name: STYLE if name == "truememory.personality_style_vec"
                                         else personality)
        with patch.object(MAINTENANCE, "engine_layer_specs", public_specs), patch.object(MAINTENANCE, "importlib", importer), \
             patch.object(MAINTENANCE, "_installed_dependency", return_value=None), \
             patch.object(personality, "extract_preferences",
                          side_effect=AssertionError("unsupported preference work")) as preferences:
            result = engine.consolidate()
        preferences.assert_not_called()
        self.assertEqual(len(result), 9)
        self.assertEqual(set(result), set(MAINTENANCE._MAINTENANCE_RESULT_KEYS))
        self.assertTrue(all("pending caller commit" in result[key]
                            for key in MAINTENANCE._MAINTENANCE_RESULT_KEYS if key != "extract_preferences"), result)
        self.assertEqual(result["extract_preferences"], "UNAVAILABLE (ScheduledPreferencesUnsupported)")
        self.assertEqual(coordinator.status, ("failed", "PriorFailure"))
        self.assertEqual(engine.get_style_health()["last_error"], "PriorFailure")
        conn.rollback()

    def test_routed_owned_commit_and_borrowed_release_failures_keep_previous_output(self) -> None:
        for borrowed in (False, True):
            for cache in (0, 5, 100):
                with self.subTest(borrowed=borrowed, cache=cache):
                    conn = sqlite3.connect(":memory:", cached_statements=cache)
                    self.addCleanup(conn.close)
                    conn.executescript(STORAGE._SCHEMA_SQL)
                    STORAGE._initialize_maintenance_tracking(conn)
                    conn.commit()
                    self.add(conn)
                    MAINTENANCE.run_routed_style(conn)
                    before, checkpoint = self.rows(conn), self.checkpoint(conn)
                    if borrowed:
                        conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-caller','pending')")
                    wrapper = BASE["BASE"]["PublicationFailure"](conn, borrowed=borrowed)
                    result = MAINTENANCE.run_routed_style(wrapper, force=True, allow_caller_transaction=borrowed)
                    self.assertEqual(result.outcome, "failed")
                    self.assertEqual(wrapper.denied, 1)
                    self.assertEqual(self.rows(conn), before)
                    self.assertEqual(self.checkpoint(conn)[3:7], checkpoint[3:7])
                    self.assertEqual(conn.in_transaction, borrowed)
                    if borrowed:
                        self.assertEqual(conn.execute("SELECT value FROM metadata WHERE key='synthetic-caller'").fetchone(), ("pending",))
                        conn.rollback()
                        self.assertEqual(self.checkpoint(conn), checkpoint)

    def test_cancellation_after_style_compute_prevents_public_stages(self) -> None:
        conn = self.connection(bootstrap=True)
        before = self.checkpoint(conn)
        cancel = threading.Event()
        compute = STYLE._compute_entity_style_vectors
        def stop_after_compute(*args: object, **kwargs: object) -> object:
            result = compute(*args, **kwargs)
            cancel.set()
            return result
        with patch.object(STYLE, "_compute_entity_style_vectors", stop_after_compute), \
             patch.object(MAINTENANCE, "_prepare_worker_extensions", side_effect=AssertionError):
            report = MAINTENANCE.run_engine_maintenance(conn, MAINTENANCE.MaintenanceCoordinator(None),
                                                       include_style=True, force=True, cancel=cancel)
        self.assertEqual(report.results, ())
        self.assertEqual(report.style_result.error_category, "StyleCancelled")
        self.assertEqual(self.checkpoint(conn), before)
        self.assertFalse(conn.in_transaction)

    def test_health_does_not_connect_when_disconnected_or_foreground_busy(self) -> None:
        engine = self.engine(self.connection())
        with patch.object(engine, "_ensure_connection", side_effect=AssertionError), \
             patch.object(engine, "_maybe_auto_consolidate", side_effect=AssertionError):
            with engine._write_lock:
                self.assertEqual(engine.get_style_health()["pending_reason"], "foreground_busy")
            engine.conn = None
            self.assertEqual(engine.get_style_health()["pending_reason"], "not_connected")

    def test_integrity_opener_faults_clear_handler_and_close_before_returning(self) -> None:
        for failure in ("busy", "cancel", "io", "unknown", "corrupt", "empty", "config"):
            with self.subTest(failure=failure):
                conn = self.connection()
                cancel = threading.Event()
                events = []
                handlers = []
                class Connection:
                    row_factory = None
                    def __getattr__(self, name: str) -> object:
                        return getattr(conn, name)
                    def close(self) -> None:
                        events.append("close")
                    def set_progress_handler(self, callback: object, interval: int) -> None:
                        handlers.append(callback)
                        events.append("clear" if callback is None else "install")
                    def execute(self, sql: str, *args: object) -> object:
                        if failure == "config" and sql == "PRAGMA foreign_keys=ON":
                            raise sqlite3.OperationalError("synthetic configuration")
                        if sql == "PRAGMA quick_check(1)":
                            if failure == "cancel":
                                cancel.set()
                                self_check = handlers[-1]()
                                if self_check != 1:
                                    raise AssertionError("Owned cancellation handler did not stop")
                                raise sqlite3.OperationalError("interrupted")
                            if failure in {"corrupt", "empty"}:
                                return types.SimpleNamespace(fetchone=lambda: None if failure == "empty" else ("synthetic corruption",))
                            raise sqlite3.OperationalError({"busy": "database is locked", "io": "disk I/O error",
                                                            "unknown": "synthetic unexpected"}[failure])
                        return conn.execute(sql, *args)
                with patch.object(MAINTENANCE.sqlite3, "connect", return_value=Connection()):
                    expected = MAINTENANCE._StyleOpenRejected if failure in {"busy", "cancel"} else STORAGE.DatabaseOpenError
                    with self.assertRaises(expected) as raised:
                        MAINTENANCE._open_style_database((ROOT / "synthetic-not-created.sqlite"), cancel)
                self.assertEqual(events[-1], "close")
                if failure != "config":
                    self.assertEqual(events, ["install", "clear", "close"])
                if failure in {"busy", "cancel"}:
                    self.assertEqual(raised.exception.result.error_category, "StyleWriterBusy" if failure == "busy" else "StyleCancelled")

    def test_integrity_opener_preflight_forbids_general_setup_and_profile_mutation(self) -> None:
        conn = self.connection(bootstrap=True)
        self.add(conn)
        MAINTENANCE.run_routed_style(conn, force=True)
        conn.execute("DELETE FROM maintenance_layers WHERE layer='landmarks'")
        conn.execute("CREATE TRIGGER synthetic_initializer BEFORE INSERT ON maintenance_layers BEGIN "
                     "UPDATE entity_style_vectors SET message_count=message_count+1; END")
        conn.commit()
        class Connection:
            row_factory = None
            def __getattr__(self, name: str) -> object:
                return getattr(conn, name)
            def close(self) -> None:
                pass
            def execute(self, sql: str, *args: object) -> object:
                if sql == "PRAGMA quick_check(1)":
                    raise AssertionError("Unsupported opener cannot claim integrity checking")
                return conn.execute(sql, *args)
        before = conn.total_changes
        with patch.object(MAINTENANCE.sqlite3, "connect", return_value=Connection()), \
             patch.object(MAINTENANCE, "create_db", side_effect=AssertionError):
            with self.assertRaises(MAINTENANCE._StyleOpenRejected):
                MAINTENANCE._open_style_database((ROOT / "synthetic-not-created.sqlite"))
        self.assertEqual(conn.total_changes, before)
        self.assertEqual(conn.execute("SELECT message_count FROM entity_style_vectors").fetchone(), (1,))
        # Actual general initialization is the negative control: all eight
        # BEFORE INSERT triggers run, even for rows ignored by their conflict.
        STORAGE._initialize_maintenance_tracking(conn)
        self.assertEqual(conn.execute("SELECT message_count FROM entity_style_vectors").fetchone(), (9,))


class CoordinatorRouting(unittest.TestCase):
    connection = InMemoryRouting.connection
    no_native = InMemoryRouting.no_native

    def test_started_worker_binding_failure_reports_error_and_preserves_newer_evidence(self) -> None:
        for newer in (False, True):
            with self.subTest(newer=newer):
                coordinator = MAINTENANCE.MaintenanceCoordinator(ROOT / "synthetic-binding-not-created.sqlite")
                old_report = MAINTENANCE.MaintenanceReport((), "SKIPPED previous")
                newer_report = MAINTENANCE.MaintenanceReport((), "SKIPPED newer")
                coordinator._last_report = old_report
                def fail_identity() -> None:
                    if newer:
                        token = coordinator._begin_observation()
                        coordinator._finish_observation(token, newer_report, "failed", "NewerManual")
                    raise OSError("synthetic identity failure")
                with patch.object(MAINTENANCE, "_acquire_owner", return_value=123), \
                     patch.object(MAINTENANCE, "_release_owner") as released, \
                     patch.object(MAINTENANCE.uuid, "uuid4", fail_identity), \
                     patch.object(MAINTENANCE, "create_db", side_effect=AssertionError) as opened:
                    self.assertTrue(coordinator.request(lambda *_args: self.fail("Binding failure ran work")))
                    self.assertTrue(coordinator.wait(2))
                opened.assert_not_called()
                released.assert_called_once_with(123)
                self.assertEqual(coordinator.status, ("failed", "NewerManual" if newer else "OSError"))
                self.assertIs(coordinator._last_report, newer_report if newer else old_report)
                self.assertFalse(coordinator._active)
                self.assertIsNone(coordinator._thread)

    def test_mixed_notifications_union_flags_and_keep_latest_threshold(self) -> None:
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                coordinator = MAINTENANCE.MaintenanceCoordinator((ROOT / "synthetic-not-created.sqlite"))
                coordinator._active = True
                requests = [(True, False), (False, True)]
                if reverse:
                    requests.reverse()
                for index, (layers, style) in enumerate(requests):
                    self.assertFalse(coordinator.request_layers(threshold=24 + index, include_layers=layers, include_style=style))
                self.assertEqual(coordinator._pending, MAINTENANCE.MaintenanceRequest(2, 25, True, True))
                with self.assertRaises(ValueError):
                    coordinator.request_layers(include_layers=False, include_style=False)

    def test_busy_admission_merges_consumed_kind_with_newer_pending_kind(self) -> None:
        coordinator = MAINTENANCE.MaintenanceCoordinator((ROOT / "synthetic-not-created.sqlite"))
        def busy(_path: Path, on_busy: object) -> None:
            coordinator.request_layers(threshold=31, include_layers=False, include_style=True)
            on_busy()
        with patch.object(MAINTENANCE, "_acquire_owner", busy):
            self.assertFalse(coordinator.request_layers(threshold=25))
        self.assertEqual(coordinator._pending, MAINTENANCE.MaintenanceRequest(2, 31, True, True))

    def test_worker_start_failure_unions_real_new_wake_without_self_retry(self) -> None:
        for newer in (False, True):
            with self.subTest(newer=newer):
                coordinator = MAINTENANCE.MaintenanceCoordinator((ROOT / "synthetic-not-created.sqlite"))
                coordinator._active = True
                def start() -> None:
                    if newer:
                        coordinator.request_layers(threshold=30, include_layers=False, include_style=True)
                    raise RuntimeError("synthetic start")
                with patch.object(MAINTENANCE, "_acquire_owner", return_value=123), \
                     patch.object(MAINTENANCE, "_release_owner") as released, \
                     patch.object(MAINTENANCE.threading, "Thread", return_value=types.SimpleNamespace(start=start)), \
                     patch.object(coordinator, "_launch_layers") as successor:
                    with self.assertRaises(RuntimeError):
                        coordinator._launch(lambda *_args: None, MAINTENANCE.MaintenanceRequest(0, 25))
                released.assert_called_once_with(123)
                self.assertEqual(successor.call_count, int(newer))
                if newer:
                    self.assertEqual(successor.call_args.args[0], MAINTENANCE.MaintenanceRequest(1, 30, True, True))

    def test_observation_tokens_reject_late_completion_and_borrowed_results(self) -> None:
        coordinator = MAINTENANCE.MaintenanceCoordinator(None)
        one, two = coordinator._begin_observation(), coordinator._begin_observation()
        newer = MAINTENANCE.MaintenanceReport((), "SKIPPED")
        coordinator._finish_observation(two, newer, "failed", "NewerFailure")
        coordinator._finish_observation(one, None, "success", None)
        self.assertEqual(coordinator.status, ("failed", "NewerFailure"))
        self.assertIs(coordinator._last_report, newer)
        three = coordinator._begin_observation()
        coordinator._finish_observation(three, MAINTENANCE.MaintenanceReport((), "SKIPPED"), "success", None, borrowed=True)
        self.assertIs(coordinator._last_report, newer)

    def test_launched_opener_is_immutable_and_style_worker_never_initializes_general_schema(self) -> None:
        coordinator = MAINTENANCE.MaintenanceCoordinator((ROOT / "synthetic-not-created.sqlite"))
        with patch.object(coordinator, "_launch", return_value=True) as launch:
            coordinator._launch_layers(MAINTENANCE.MaintenanceRequest(1, 25, False, True))
        work = launch.call_args.args[0]
        self.assertEqual(launch.call_args.kwargs, {"style_only": True})
        coordinator._pending = MAINTENANCE.MaintenanceRequest(2, 30, True, False)
        conn = self.connection()
        events = []
        class Connection:
            def __getattr__(self, name: str) -> object:
                return getattr(conn, name)
            def close(self) -> None:
                events.append("close")
        with patch.object(MAINTENANCE, "_open_style_database", return_value=Connection()) as opener, \
             patch.object(MAINTENANCE, "create_db", side_effect=AssertionError), \
             patch.object(MAINTENANCE, "_release_owner", side_effect=lambda _fd: events.append("release")), \
             patch.object(coordinator, "_launch_layers") as successor, self.no_native():
            coordinator._run(123, work, style_only=True)
        opener.assert_called_once()
        self.assertEqual(events, ["close", "release"])
        self.assertEqual(coordinator._last_report.style_result.outcome, "success_empty")
        successor.assert_called_once_with(MAINTENANCE.MaintenanceRequest(2, 30, True, False))

    def test_late_worker_cleanup_cannot_replace_newer_manual_failure(self) -> None:
        coordinator = MAINTENANCE.MaintenanceCoordinator(ROOT / "synthetic-worker-cleanup-not-created.sqlite")
        conn = self.connection()
        events = []
        class Connection:
            def __getattr__(self, name: str) -> object:
                return getattr(conn, name)
            def close(self) -> None:
                events.append("close")
                token = coordinator._begin_observation()
                coordinator._finish_observation(token, None, "failed", "NewerFailure")
        exists = Path.exists
        with patch.object(Path, "exists", lambda path: False if path == coordinator.path else exists(path)), \
             patch.object(MAINTENANCE, "create_db", return_value=Connection()), \
             patch.object(MAINTENANCE, "_release_owner", side_effect=lambda _fd: events.append("release")):
            coordinator._run(123, lambda *_args: MAINTENANCE.MaintenanceReport((), "SKIPPED"))
        self.assertEqual(events, ["close", "release"])
        self.assertEqual(coordinator.status, ("failed", "NewerFailure"))

    def test_setup_and_close_failures_retain_report_and_release_after_cleanup(self) -> None:
        for boundary in ("setup", "close"):
            with self.subTest(boundary=boundary):
                coordinator = MAINTENANCE.MaintenanceCoordinator(ROOT / "synthetic-worker-failure-not-created.sqlite")
                old = MAINTENANCE.MaintenanceReport((), "SKIPPED")
                coordinator._last_report = old
                conn = self.connection()
                events = []
                class Connection:
                    def __getattr__(self, name: str) -> object:
                        return getattr(conn, name)
                    def close(self) -> None:
                        events.append("close")
                        raise sqlite3.OperationalError("synthetic close")
                exists = Path.exists
                with patch.object(Path, "exists", lambda path: False if path == coordinator.path else exists(path)), \
                     patch.object(MAINTENANCE, "create_db", side_effect=RuntimeError("synthetic setup") if boundary == "setup" else None,
                                  return_value=Connection()), \
                     patch.object(MAINTENANCE, "_release_owner", side_effect=lambda _fd: events.append("release")):
                    coordinator._run(123, lambda *_args: MAINTENANCE.MaintenanceReport((), "SKIPPED new"))
                self.assertIs(coordinator._last_report, old)
                self.assertEqual(coordinator.status, ("failed", "RuntimeError" if boundary == "setup" else "OperationalError"))
                self.assertEqual(events, ["release"] if boundary == "setup" else ["close", "release"])

    def test_unstarted_failure_cannot_overwrite_manual_observation_after_owner_release(self) -> None:
        coordinator = MAINTENANCE.MaintenanceCoordinator(ROOT / "synthetic-start-not-created.sqlite")
        newer = MAINTENANCE.MaintenanceReport((), "SKIPPED newer")
        class Unstarted:
            def __init__(self, **_kwargs: object) -> None:
                pass
            def start(self) -> None:
                raise RuntimeError("synthetic start failure")
        def released(_fd: int) -> None:
            token = coordinator._begin_observation()
            coordinator._finish_observation(token, newer, "failed", "NewerManual")
        with patch.object(MAINTENANCE, "_acquire_owner", return_value=123), \
             patch.object(MAINTENANCE, "_release_owner", released), \
             patch.object(MAINTENANCE.threading, "Thread", Unstarted):
            with self.assertRaisesRegex(RuntimeError, "synthetic start failure"):
                coordinator.request_layers(include_layers=False, include_style=True)
        self.assertEqual(coordinator.status, ("failed", "NewerManual"))
        self.assertIs(coordinator._last_report, newer)
        self.assertFalse(coordinator._active)
        self.assertTrue(coordinator.wait(0))


class FileOpenerRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="synthetic-style-routing-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.sqlite"
        conn = STORAGE.create_db(self.path)
        conn.close()

    def test_uri_filenames_settings_and_missing_file(self) -> None:
        for suffix in ("hash#mark", "question?mark", "percent%mark", "space mark", "unicode-é"):
            with self.subTest(suffix=suffix):
                if os.name == "nt" and "?" in suffix:
                    self.skipTest("Question marks are not valid Windows filenames")
                path = self.path.with_name(suffix + ".sqlite")
                original = STORAGE.create_db(path)
                journal = original.execute("PRAGMA journal_mode").fetchone()
                original.close()
                conn = MAINTENANCE._open_style_database(path)
                try:
                    self.assertIsNone(conn.row_factory)
                    self.assertEqual(conn.execute("PRAGMA busy_timeout").fetchone(), (10000,))
                    self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone(), (1,))
                    self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone(), journal)
                    self.assertEqual(MAINTENANCE.run_routed_style(conn, connection_owned=True).outcome, "success_empty")
                finally:
                    conn.close()
        missing = self.path.with_name("missing.sqlite")
        with self.assertRaises(STORAGE.DatabaseOpenError):
            MAINTENANCE._open_style_database(missing)
        self.assertFalse(missing.exists())

    def test_unsupported_guard_precedes_general_initializer_side_effects(self) -> None:
        conn = sqlite3.connect(self.path)
        conn.execute("CREATE TABLE sentinel(value TEXT)")
        conn.execute("DELETE FROM maintenance_layers WHERE layer='landmarks'")
        conn.execute("CREATE TRIGGER synthetic_extra BEFORE INSERT ON maintenance_layers BEGIN INSERT INTO sentinel VALUES ('unexpected'); END")
        conn.commit()
        before = conn.total_changes
        with self.assertRaises(MAINTENANCE._StyleOpenRejected) as raised:
            MAINTENANCE._open_style_database(self.path)
        self.assertEqual(raised.exception.result.error_category, "StyleTriggersUnsupported")
        self.assertEqual(conn.total_changes, before)
        self.assertEqual(conn.execute("SELECT count(*) FROM sentinel").fetchone(), (0,))
        self.assertIsNone(conn.execute("SELECT 1 FROM maintenance_layers WHERE layer='landmarks'").fetchone())
        conn.close()

    def test_missing_source_readiness_and_corruption_never_enroll_style(self) -> None:
        for change in ("missing", "changed", "unready", "token", "schema", "metadata"):
            with self.subTest(change=change):
                path = self.path.with_name("synthetic-" + change + ".sqlite")
                conn = STORAGE.create_db(path)
                if change in {"missing", "changed"}:
                    conn.execute("DROP TRIGGER messages_maintenance_ai")
                    if change == "changed":
                        conn.execute("CREATE TRIGGER messages_maintenance_ai AFTER INSERT ON messages BEGIN SELECT 1; END")
                elif change == "unready":
                    conn.execute("UPDATE maintenance_source_state SET tracking_ready=0")
                elif change == "token":
                    conn.execute("DELETE FROM maintenance_source_state")
                elif change == "schema":
                    conn.execute("ALTER TABLE messages RENAME COLUMN id TO synthetic_id")
                else:
                    conn.execute("DROP TABLE metadata")
                conn.commit()
                conn.close()
                with self.assertRaises(MAINTENANCE._StyleOpenRejected) as raised:
                    MAINTENANCE._open_style_database(path)
                self.assertEqual(raised.exception.result.error_category, "StyleSourceUnavailable")
                conn = sqlite3.connect(path)
                self.assertIsNone(conn.execute("SELECT 1 FROM maintenance_layers WHERE layer='style_vectors'").fetchone())
                self.assertEqual(conn.execute("SELECT count(*) FROM entity_style_vectors").fetchone(), (0,))
                conn.close()
        damaged = self.path.with_name("synthetic-corrupt.sqlite")
        damaged.write_bytes(b"synthetic invalid database")
        with self.assertRaises(STORAGE.DatabaseOpenError):
            MAINTENANCE._open_style_database(damaged)


if __name__ == "__main__":
    unittest.main()
