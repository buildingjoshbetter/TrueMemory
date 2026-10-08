"""Durable maintenance scheduling and publication on synthetic SQLite only."""

import ast
import runpy
import sqlite3
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
_loader = runpy.run_path(str(Path(__file__).with_name("test-maintenance-source-revision-753.py")))
STORAGE, MAINTENANCE = _loader["STORAGE"], _loader["MAINTENANCE"]


class RunnerFixture(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="synthetic-runner-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "synthetic.sqlite"
        self.conn = self.open_db()
        self.conn.execute("CREATE TABLE synthetic_output(layer TEXT, value INTEGER)")
        self.calls = []
        self.dependency = MAINTENANCE.make_layer_dependency(1, {"algorithm": "synthetic"})

    def open_db(self) -> sqlite3.Connection:
        conn = STORAGE.create_db(self.path)
        conn.execute("PRAGMA busy_timeout=200")
        self.addCleanup(conn.close)
        return conn

    def add(self, count: int, conn: sqlite3.Connection | None = None) -> None:
        conn = conn or self.conn
        conn.executemany("INSERT INTO messages(content,timestamp) VALUES ('synthetic source','2026-01-01')",
                         [()] * count)
        conn.commit()

    def spec(self, layer: str = "summaries", action=None):
        def build(conn: sqlite3.Connection) -> None:
            self.calls.append(layer)
            conn.execute("DELETE FROM synthetic_output WHERE layer=?", (layer,))
            count = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
            if count:
                conn.execute("INSERT INTO synthetic_output VALUES (?,?)", (layer, count))
            if action is not None:
                action(conn)

        def output_count(conn: sqlite3.Connection) -> int:
            return conn.execute("SELECT count(*) FROM synthetic_output WHERE layer=?", (layer,)).fetchone()[0]

        return MAINTENANCE.LayerSpec(layer, "synthetic_" + layer, lambda: self.dependency, build, output_count)

    def state(self, spec=None, conn: sqlite3.Connection | None = None):
        spec = spec or self.spec()
        return MAINTENANCE.read_layer_states(conn or self.conn, (spec,))[spec.layer]

    def run_layer(self, spec=None, **kwargs):
        return MAINTENANCE.run_layers(self.conn, (spec or self.spec(),), **kwargs)[0]

    def output(self, conn: sqlite3.Connection | None = None) -> list[tuple]:
        return (conn or self.conn).execute("SELECT * FROM synthetic_output ORDER BY layer").fetchall()


class TestRunnerEligibility(RunnerFixture):
    def test_initial_empty_success_is_durable_and_skipped_after_reopen(self) -> None:
        result = self.run_layer()
        self.assertEqual((result.outcome, result.output_count), ("success_empty", 0))
        reopened = self.open_db()
        self.assertEqual(MAINTENANCE.plan_layers(reopened, (self.spec(),)), ())
        self.assertEqual(MAINTENANCE.layer_freshness(self.state().source, self.state(), self.dependency), "current")
        self.assertEqual(self.run_layer().attempted, False)
        self.assertEqual(self.calls, ["summaries"])

    def test_historical_zero_insert_count_gets_first_attempt(self) -> None:
        self.add(3)
        self.conn.execute("UPDATE maintenance_source_state SET revision=0,insert_count=0,nonappend_revision=0")
        self.conn.commit()
        self.assertEqual(self.run_layer().outcome, "success")
        self.assertEqual(self.output(), [("summaries", 3)])
        self.assertEqual(self.state().attempted_insert_count, 0)

    def test_threshold_is_24_then_25_across_connections(self) -> None:
        self.run_layer()
        for _ in range(24):
            self.add(1, self.open_db())
        self.assertEqual(MAINTENANCE.plan_layers(self.open_db(), (self.spec(),)), ())
        self.add(1, self.open_db())
        self.assertEqual(len(MAINTENANCE.plan_layers(self.open_db(), (self.spec(),))), 1)
        self.assertEqual(self.run_layer().outcome, "success")
        self.assertEqual(self.state().attempted_insert_count, 25)
        self.add(24)
        self.assertFalse(self.run_layer().attempted)
        self.assertEqual(self.calls, ["summaries", "summaries"])

    def test_rollback_does_not_advance_threshold(self) -> None:
        self.run_layer()
        self.conn.executemany("INSERT INTO messages(content) VALUES ('synthetic rolled back')", [()] * 25)
        self.conn.rollback()
        self.assertEqual(MAINTENANCE.plan_layers(self.conn, (self.spec(),)), ())
        self.assertEqual(self.state().source.insert_count, 0)

    def test_update_delete_and_replacement_wake_below_threshold(self) -> None:
        self.add(1)
        self.run_layer()
        for sql in (
            "UPDATE messages SET content='synthetic correction' WHERE id=1",
            "INSERT OR REPLACE INTO messages(id,content) VALUES (1,'synthetic replacement')",
            "DELETE FROM messages WHERE id=1",
        ):
            with self.subTest(operation=sql.split()[0]):
                self.conn.execute("PRAGMA recursive_triggers=OFF")
                self.conn.execute(sql)
                self.conn.commit()
                self.assertTrue(self.run_layer().attempted)
        self.assertEqual(self.state().source.correction_count, 2)

    def test_untrusted_existing_success_and_abandoned_state_attempt_once(self) -> None:
        self.add(1)
        self.run_layer()
        self.conn.execute("UPDATE maintenance_layers SET full_rebuild_required=1 WHERE layer='summaries'")
        self.conn.commit()
        self.assertTrue(self.run_layer().attempted)
        self.conn.execute("UPDATE maintenance_layers SET outcome='running',run_generation='synthetic-dead-owner' WHERE layer='summaries'")
        self.conn.commit()
        self.assertEqual(len(MAINTENANCE.plan_layers(self.conn, (self.spec(),))), 1)
        self.assertTrue(self.run_layer().attempted)
        self.assertFalse(self.run_layer().attempted)

    def test_failure_is_suppressed_for_exact_input_and_manual_force_retries(self) -> None:
        def fail(conn: sqlite3.Connection) -> None:
            raise ValueError("synthetic source-bearing error")

        spec = self.spec(action=fail)
        result = self.run_layer(spec)
        self.assertEqual((result.outcome, result.error_category), ("failed", "ValueError"))
        self.assertFalse(self.run_layer(spec).attempted)
        self.assertTrue(self.run_layer(spec, force=True).attempted)
        self.assertEqual(self.calls, ["summaries", "summaries"])
        self.assertNotIn("source-bearing", str(self.state(spec)))

    def _assert_initial_terminal_attempt_respects_threshold(self, unavailable: bool) -> None:
        def fail(conn: sqlite3.Connection) -> None:
            raise ValueError("synthetic failure")

        if unavailable:
            self.dependency = MAINTENANCE.make_layer_dependency(1, available=False, error_category="ImportError")
        spec = self.spec(action=fail)
        outcome = "unavailable" if unavailable else "failed"
        self.assertEqual(self.run_layer(spec).outcome, outcome)
        self.assertIsNone(self.state(spec).successful_epoch)
        self.assertTrue(self.state(spec).full_rebuild_required)
        for added, total in ((1, 1), (23, 24)):
            with self.subTest(insert_count=total):
                self.add(added, self.open_db())
                self.assertEqual(MAINTENANCE.plan_layers(self.open_db(), (spec,)), ())
                self.assertFalse(self.run_layer(spec).attempted)
                self.assertEqual(self.state(spec).attempted_insert_count, 0)
        self.add(1, self.open_db())
        self.assertEqual(len(MAINTENANCE.plan_layers(self.open_db(), (spec,))), 1)
        self.assertTrue(self.run_layer(spec).attempted)
        self.assertEqual(self.state(spec).attempted_insert_count, 25)
        self.assertFalse(self.run_layer(spec).attempted)

        self.conn.execute("UPDATE messages SET content='synthetic correction' WHERE id=1")
        self.conn.commit()
        self.assertTrue(self.run_layer(spec).attempted)
        self.assertFalse(self.run_layer(spec).attempted)
        self.dependency = MAINTENANCE.make_layer_dependency(
            2, available=not unavailable, error_category="ImportError" if unavailable else None,
        )
        self.assertTrue(self.run_layer(spec).attempted)
        self.assertFalse(self.run_layer(spec).attempted)
        self.assertTrue(self.run_layer(spec, force=True).attempted)
        self.assertEqual(self.calls, [] if unavailable else ["summaries"] * 5)

    def test_initial_failure_waits_for_25_appends_but_corrections_dependency_and_force_wake(self) -> None:
        self._assert_initial_terminal_attempt_respects_threshold(unavailable=False)

    def test_initial_unavailable_waits_for_25_appends_but_corrections_dependency_and_force_wake(self) -> None:
        self._assert_initial_terminal_attempt_respects_threshold(unavailable=True)

    def test_unavailable_attempt_uses_captured_source_not_later_current(self) -> None:
        self.dependency = MAINTENANCE.make_layer_dependency(1, available=False, error_category="ImportError")
        writer = self.open_db()
        original = MAINTENANCE._record_attempt
        captured = []

        def record(conn, state, *args):
            captured.append(state.source)
            self.add(1, writer)
            return original(conn, state, *args)

        with patch.object(MAINTENANCE, "_record_attempt", record):
            self.assertEqual(self.run_layer().outcome, "unavailable")
        state = self.state()
        self.assertEqual((state.attempted_revision, state.attempted_insert_count), (0, 0))
        self.assertEqual(state.source.revision, 1)
        self.assertEqual(captured[0].revision, 0)
        self.assertEqual(self.calls, [])

    def test_unchanged_unavailable_is_suppressed_and_restored_dependency_retries(self) -> None:
        self.dependency = MAINTENANCE.make_layer_dependency(1, available=False, error_category="ImportError")
        self.assertEqual(self.run_layer().outcome, "unavailable")
        self.assertFalse(self.run_layer().attempted)
        self.dependency = MAINTENANCE.make_layer_dependency(1)
        self.assertEqual(self.run_layer().outcome, "success_empty")
        self.assertEqual(self.calls, ["summaries"])

    def test_new_source_epoch_invalidates_old_success_without_new_inserts(self) -> None:
        self.add(1)
        self.run_layer()
        old = self.state()
        self.conn.execute("DROP TRIGGER messages_maintenance_au")
        STORAGE._initialize_maintenance_tracking(self.conn)
        self.assertNotEqual(self.state().source.epoch, old.source.epoch)
        self.assertEqual(self.state().source.insert_count, 0)
        self.assertTrue(self.run_layer().attempted)

    def test_dependency_change_retries_and_failed_new_version_preserves_old_success(self) -> None:
        self.add(1)
        self.run_layer()
        old = self.state()
        self.dependency = MAINTENANCE.make_layer_dependency(2, {"algorithm": "synthetic"})

        def fail(conn: sqlite3.Connection) -> None:
            raise ValueError("synthetic failure")

        self.assertEqual(self.run_layer(self.spec(action=fail)).outcome, "failed")
        state = self.state()
        self.assertEqual(state.successful_dependency, old.successful_dependency)
        self.assertEqual(state.attempted_dependency, self.dependency.key)
        self.assertEqual(MAINTENANCE.layer_freshness(state.source, state, self.dependency), "dependency_pending")
        self.assertFalse(self.run_layer().attempted)

    def test_forced_failure_does_not_hide_a_still_current_previous_success(self) -> None:
        self.add(1)
        self.run_layer()
        previous = self.output()

        def fail(conn: sqlite3.Connection) -> None:
            raise ValueError("synthetic failure")

        self.run_layer(self.spec(action=fail), force=True)
        state = self.state()
        self.assertEqual(state.outcome, "failed")
        self.assertEqual(MAINTENANCE.layer_freshness(state.source, state, self.dependency), "current")
        self.assertEqual(self.output(), previous)

    def test_failure_does_not_rerun_or_erase_successful_sibling(self) -> None:
        self.add(1)
        good = self.spec()

        def fail(conn: sqlite3.Connection) -> None:
            raise ValueError("synthetic failure")

        bad = self.spec("landmarks", fail)
        result = MAINTENANCE.run_layers(self.conn, (good, bad))
        self.assertEqual([item.outcome for item in result], ["success", "failed"])
        self.assertEqual(self.output(), [("summaries", 1)])
        self.assertTrue(all(not item.attempted for item in MAINTENANCE.run_layers(self.open_db(), (good, bad))))


class TestRunnerTransactions(RunnerFixture):
    def test_running_diagnostic_is_committed_without_writer_during_compute(self) -> None:
        self.add(1)
        writer = self.open_db()
        spec = self.spec()
        observed = []

        def build(conn: sqlite3.Connection) -> None:
            observed.append(writer.execute("SELECT outcome FROM maintenance_layers WHERE layer='summaries'").fetchone()[0])
            writer.execute("INSERT INTO metadata(key,value) VALUES ('synthetic_writer','committed')")
            writer.commit()
            spec.build(conn)

        result = self.run_layer(spec._replace(build=build))
        self.assertEqual(observed, ["running"])
        self.assertEqual(result.outcome, "failed")  # Stale WAL snapshot cannot publish.
        self.assertEqual(self.output(), [])
        self.assertIsNone(self.state().successful_revision)

    def test_changed_source_is_pending_and_cannot_publish_stale_output(self) -> None:
        self.add(1)
        self.run_layer()
        writer = self.open_db()
        spec = self.spec()

        def build(conn: sqlite3.Connection) -> None:
            writer.execute("UPDATE messages SET content='synthetic correction' WHERE id=1")
            writer.commit()
            spec.build(conn)

        result = self.run_layer(spec._replace(build=build), force=True)
        state = self.state()
        self.assertEqual(result.outcome, "failed")
        self.assertEqual(state.attempted_revision, 1)
        self.assertEqual(state.source.revision, 2)
        self.assertEqual(state.successful_revision, 1)
        self.assertEqual(len(MAINTENANCE.plan_layers(self.conn, (spec,))), 1)

    def test_dependency_change_after_output_write_rolls_back_output_and_checkpoint(self) -> None:
        self.add(1)
        self.run_layer()
        old = self.state()
        previous = self.output()

        def change(conn: sqlite3.Connection) -> None:
            self.dependency = MAINTENANCE.make_layer_dependency(2)

        self.assertEqual(self.run_layer(self.spec(action=change), force=True).outcome, "failed")
        self.assertEqual(self.output(), previous)
        self.assertEqual(self.state().successful_dependency, old.successful_dependency)

    def test_dependency_is_rechecked_after_checkpoint_before_commit(self) -> None:
        self.add(1)
        original = MAINTENANCE.record_layer_success_in_transaction

        def publish(*args, **kwargs) -> None:
            original(*args, **kwargs)
            self.dependency = MAINTENANCE.make_layer_dependency(2)

        with patch.object(MAINTENANCE, "record_layer_success_in_transaction", publish):
            self.assertEqual(self.run_layer().outcome, "failed")
        self.assertEqual(self.output(), [])
        self.assertIsNone(self.state().successful_revision)

    def test_successful_output_and_provenance_become_visible_together(self) -> None:
        self.add(1)
        self.run_layer()
        self.add(1)
        reader = self.open_db()
        observed = []
        original = MAINTENANCE.record_layer_success_in_transaction

        def publish(*args, **kwargs) -> None:
            original(*args, **kwargs)
            observed.append((self.output(reader), self.state(conn=reader).successful_revision))

        with patch.object(MAINTENANCE, "record_layer_success_in_transaction", publish):
            self.run_layer(force=True)
        self.assertEqual(observed, [([("summaries", 1)], 1)])
        self.assertEqual((self.output(reader), self.state(conn=reader).successful_revision), ([("summaries", 2)], 2))

    def test_output_commit_failure_preserves_previous_success(self) -> None:
        self.add(1)
        self.run_layer()
        old = self.state()
        self.add(1)
        denied = []

        def authorizer(action: int, operation: str | None, *unused: str | None) -> int:
            if action == sqlite3.SQLITE_TRANSACTION and operation == "COMMIT" and not denied:
                denied.append(True)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        try:
            # Install after the running diagnostic commits; this also
            # invalidates SQLite's cached prepared COMMIT statement.
            result = self.run_layer(self.spec(action=lambda conn: conn.set_authorizer(authorizer)), force=True)
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertEqual(result.outcome, "failed")
        self.assertEqual(denied, [True])
        self.assertEqual(self.output(), [("summaries", 1)])
        self.assertEqual(self.state().successful_revision, old.successful_revision)
        self.assertFalse(self.conn.in_transaction)

    def test_checkpoint_write_failure_cannot_publish_output(self) -> None:
        self.add(1)
        self.conn.execute(
            "CREATE TRIGGER synthetic_checkpoint_failure BEFORE UPDATE OF successful_revision ON maintenance_layers "
            "BEGIN SELECT RAISE(ABORT,'synthetic checkpoint failure'); END"
        )
        self.assertEqual(self.run_layer().outcome, "failed")
        self.assertEqual(self.output(), [])
        self.assertIsNone(self.state().successful_revision)

    def test_change_after_commit_remains_pending_without_an_immediate_loop(self) -> None:
        self.add(1)
        writer = self.open_db()
        actual = self.conn
        commits = []

        class AfterCommit:
            def __getattr__(self, name: str) -> object:
                return getattr(actual, name)

            def execute(self, sql: str, parameters: tuple = ()):
                result = actual.execute(sql, parameters)
                if sql == "COMMIT":
                    commits.append(True)
                    if len(commits) == 2:
                        writer.execute("INSERT INTO messages(content) VALUES ('synthetic next generation')")
                        writer.commit()
                return result

        result = MAINTENANCE.run_layers(AfterCommit(), (self.spec(),))
        self.assertEqual(result[0].outcome, "success")
        self.assertEqual(self.calls, ["summaries"])
        state = self.state()
        self.assertEqual((state.successful_revision, state.source.revision), (1, 2))
        self.assertEqual(MAINTENANCE.layer_freshness(state.source, state, self.dependency), "append_pending")

    def test_failure_status_commit_failure_is_surfaced(self) -> None:
        denied = []

        def authorizer(action: int, operation: str | None, *unused: str | None) -> int:
            if action == sqlite3.SQLITE_TRANSACTION and operation == "COMMIT" and not denied:
                denied.append(True)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        def fail(conn: sqlite3.Connection) -> None:
            conn.set_authorizer(authorizer)
            raise ValueError("synthetic failure")

        try:
            with self.assertRaises(sqlite3.DatabaseError):
                self.run_layer(self.spec(action=fail))
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertEqual(self.state().outcome, "running")
        self.assertEqual(denied, [True])
        self.assertFalse(self.conn.in_transaction)
        self.assertTrue(self.run_layer().attempted)

    def test_cancellation_rolls_back_and_leaves_retryable_abandoned_state(self) -> None:
        self.add(1)
        cancel = threading.Event()
        result = self.run_layer(self.spec(action=lambda conn: cancel.set()), cancel=cancel)
        self.assertEqual(result.outcome, "abandoned")
        self.assertEqual(self.output(), [])
        self.assertIsNone(self.state().successful_revision)
        self.assertTrue(self.run_layer().attempted)

    def test_keyboard_interrupt_rolls_back_and_propagates(self) -> None:
        self.add(1)

        def interrupt(conn: sqlite3.Connection) -> None:
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.run_layer(self.spec(action=interrupt))
        self.assertEqual(self.state().outcome, "abandoned")
        self.assertEqual(self.output(), [])

    def test_active_caller_transaction_is_rejected_without_committing_it(self) -> None:
        self.conn.execute("INSERT INTO messages(content) VALUES ('synthetic pending')")
        with self.assertRaises(RuntimeError):
            self.run_layer()
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.state().source.insert_count, 0)

    def test_nested_worker_owner_is_reused_and_other_thread_reports_busy(self) -> None:
        errors = []

        def contender() -> None:
            try:
                MAINTENANCE.run_layers(self.open_db(), (self.spec(),))
            except MAINTENANCE.MaintenanceBusyError:
                errors.append("busy")

        with MAINTENANCE.maintenance_owner(self.path):
            self.assertTrue(self.run_layer().attempted)
            thread = threading.Thread(target=contender)
            thread.start()
            thread.join(3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["busy"])


class TestSharedCheckpointAndReaders(RunnerFixture):
    def test_checkpoint_requires_transaction_and_does_not_commit_caller(self) -> None:
        source = MAINTENANCE.read_source_revision(self.conn)
        arguments = dict(layer="entity_profiles", dependency=self.dependency, source=source, output_count=0)
        with self.assertRaises(RuntimeError):
            MAINTENANCE.record_layer_success_in_transaction(self.conn, **arguments)
        self.conn.execute("BEGIN")
        MAINTENANCE.record_layer_success_in_transaction(self.conn, **arguments)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.state(self.spec("entity_profiles")).successful_revision, 0)
        self.conn.rollback()
        self.assertIsNone(self.state(self.spec("entity_profiles")).successful_revision)

    def test_checkpoint_rejects_a_stale_captured_source(self) -> None:
        source = MAINTENANCE.read_source_revision(self.conn)
        self.add(1)
        self.conn.execute("BEGIN")
        with self.assertRaises(sqlite3.OperationalError):
            MAINTENANCE.record_layer_success_in_transaction(self.conn, layer="summaries",
                dependency=self.dependency, source=source, output_count=0)
        self.conn.rollback()
        self.assertIsNone(self.state().successful_revision)

    def test_reader_context_keeps_row_and_freshness_in_one_snapshot(self) -> None:
        self.add(1)
        self.run_layer()
        writer = self.open_db()
        with MAINTENANCE.layer_read_snapshot(self.conn, "summaries", self.dependency) as state:
            self.add(1, writer)
            MAINTENANCE.run_layers(writer, (self.spec(),), force=True)
            self.assertEqual(MAINTENANCE.layer_freshness(state.source, state, self.dependency), "current")
            self.assertEqual(self.output(), [("summaries", 1)])
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.output(), [("summaries", 2)])

    def test_read_context_preserves_caller_transaction(self) -> None:
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic_pending','retained')")
        with MAINTENANCE.layer_read_snapshot(self.conn, "summaries", self.dependency):
            self.assertTrue(self.conn.in_transaction)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertIsNone(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic_pending'").fetchone())

    def test_failed_reader_releases_only_its_owned_snapshot(self) -> None:
        with self.assertRaises(ValueError):
            with MAINTENANCE.layer_read_snapshot(self.conn, "summaries", self.dependency):
                raise ValueError("synthetic reader failure")
        self.assertFalse(self.conn.in_transaction)

    def test_correction_pending_is_distinct_from_append_pending(self) -> None:
        self.add(1)
        self.run_layer()
        self.add(1)
        state = self.state()
        self.assertEqual(MAINTENANCE.layer_freshness(state.source, state, self.dependency), "append_pending")
        self.conn.execute("UPDATE messages SET sender='synthetic-other' WHERE id=1")
        self.conn.commit()
        state = self.state()
        self.assertEqual(MAINTENANCE.layer_freshness(state.source, state, self.dependency), "correction_pending")


class TestRunnerMigrationAndAdapters(RunnerFixture):
    def test_additive_migration_preserves_epoch_counters_and_existing_provenance(self) -> None:
        self.add(1)
        self.run_layer()
        token = MAINTENANCE.read_source_revision(self.conn)
        success = self.state().successful_dependency
        self.conn.execute("ALTER TABLE maintenance_layers RENAME TO synthetic_previous_layers")
        self.conn.execute(
            "CREATE TABLE maintenance_layers AS SELECT layer,builder_version,outcome,successful_epoch,"
            "successful_revision,successful_dependency,attempted_epoch,attempted_revision,attempted_dependency,"
            "full_rebuild_required,output_count,run_generation,error_category FROM synthetic_previous_layers"
        )
        self.conn.execute("CREATE UNIQUE INDEX synthetic_layer_key ON maintenance_layers(layer)")
        self.conn.commit()
        STORAGE._initialize_maintenance_tracking(self.conn)
        self.assertEqual(MAINTENANCE.read_source_revision(self.conn), token)
        self.assertEqual(self.state().successful_dependency, success)
        self.assertIsNone(self.state().attempted_insert_count)
        schema_version = self.conn.execute("PRAGMA schema_version").fetchone()[0]
        STORAGE._initialize_maintenance_tracking(self.conn)
        self.assertEqual(self.conn.execute("PRAGMA schema_version").fetchone()[0], schema_version)

    def test_six_real_nonvector_builders_compose_without_native_imports(self) -> None:
        modules = {}
        for name in ("consolidation", "predictive", "temporal"):
            module = types.ModuleType("synthetic_runner_" + name)
            path = ROOT / "truememory" / (name + ".py")
            tree = ast.parse(path.read_text())
            tree.body = [node for node in tree.body if not (
                isinstance(node, ast.ImportFrom) and node.module in {"truememory.storage", "truememory.fts_search"}
            )]
            exec(compile(tree, str(path), "exec"), module.__dict__)
            modules["truememory." + name] = module
        self.conn.executemany(
            "INSERT INTO messages(content,sender,recipient,timestamp) VALUES (?,?,?,?)",
            [("synthetic team launched Cedar", "synthetic-sender", "synthetic-recipient", "2026-01-01"),
             ("synthetic team switched from Cedar to Birch", "synthetic-sender", "synthetic-recipient", "2026-01-02")],
        )
        self.conn.commit()
        boundary = types.SimpleNamespace(import_module=lambda name: modules[name])
        with patch.object(MAINTENANCE, "importlib", boundary):
            specs = MAINTENANCE.nonvector_layer_specs()
            result = MAINTENANCE.run_layers(self.conn, specs)
            self.assertEqual(len(result), 6)
            self.assertTrue(all(item.outcome in {"success", "success_empty"} for item in result), result)
            self.assertTrue(all(not item.attempted for item in MAINTENANCE.run_layers(self.open_db(), specs)))
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT outcome FROM maintenance_layers WHERE layer='clusters'").fetchone(), ("pending",))


if __name__ == "__main__":
    unittest.main()
