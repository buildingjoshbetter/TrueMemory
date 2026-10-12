"""Durable selected-job admission using only synthetic in-memory databases."""

from __future__ import annotations

import dataclasses
import runpy
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = runpy.run_path(str(Path(__file__).with_name("test-tier-job-identity-795.py")))
CONNECT = BASE["CONNECT"]
CONNECTION = BASE["MockFileConnection"]
Cancelled = BASE["Cancelled"]


class FaultConnection(CONNECTION):
    fault_sql: str | None = None
    fault: BaseException | None = None
    fail_commit = False
    retain_commit = False
    selection_payload_reads = 0

    def execute(self, sql: str, parameters: tuple = ()):
        if sql.startswith("SELECT job_id, tier"):
            self.selection_payload_reads += 1
        if self.fault_sql is not None and self.fault_sql in sql:
            raise self.fault
        return super().execute(sql, parameters)

    def commit(self) -> None:
        if self.fail_commit:
            raise sqlite3.OperationalError("synthetic marker commit failure")
        if not self.retain_commit:
            super().commit()


class TestTierJobBinding(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = BASE["TestTierJobIdentity"]()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.api = self.fixture.api
        self.conn = self.fixture.conn
        self.target = self.fixture.target
        self.table = self.api._SELECTION_TABLE

    def select(self, **kwargs):
        return self.api.select_tier_job(self.conn, self.target, **kwargs)

    def row(self, conn=None):
        return (conn or self.conn).execute(f"SELECT * FROM {self.table}").fetchone()

    def clone(self, *, factory=CONNECTION):
        conn = CONNECT(":memory:", factory=factory)
        self.conn.backup(conn)
        self.addCleanup(conn.close)
        return conn

    def test_selection_commits_exact_marker_without_changing_source_identity(self) -> None:
        previous = self.fixture.capture()
        before = self.conn.total_changes
        job = self.select()
        self.assertEqual(self.conn.total_changes - before, 1)
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(job.target, self.target)
        self.assertEqual(job.source_schema_signature, previous.source_schema_signature)
        self.assertEqual(job.source_epoch, previous.source_epoch)
        self.assertEqual(self.row(), (1, job.job_id, "selected", "base", "qwen3_256", 256,
                                      "basepro", job.tracker, job.source_epoch, job.source_schema_signature))
        self.api.check_tier_job(self.conn, job)
        self.api.check_selected_tier_job(self.conn, job)
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)

    def test_repeated_read_checks_do_not_change_source_plan_write_fence(self) -> None:
        job = self.select()
        changes, version = self.conn.total_changes, self.conn.execute("PRAGMA data_version").fetchone()[0]
        for _ in range(3):
            self.api.check_selected_tier_job(self.conn, job)
        self.assertEqual(self.conn.total_changes, changes)
        self.assertEqual(self.conn.execute("PRAGMA data_version").fetchone()[0], version)
        self.assertFalse(self.conn.in_transaction)

    def test_live_job_cannot_be_replaced_even_with_its_id(self) -> None:
        job = self.select()
        before = self.row()
        for kwargs in ({}, {"previous_job_id": job.job_id}, {"previous_job_id": "0" * 32}):
            with self.assertRaises(self.api.TierJobSelectionError):
                self.select(**kwargs)
            self.assertEqual(self.row(), before)
            self.assertFalse(self.conn.in_transaction)

    def test_terminal_replacement_requires_exact_previous_id(self) -> None:
        old = self.select()
        self.assertTrue(self.api.end_selected_tier_job(self.conn, old, outcome="cancelled"))
        for kwargs in ({}, {"previous_job_id": "0" * 32}):
            with self.assertRaises(self.api.TierJobSelectionError):
                self.select(**kwargs)
        new = self.select(previous_job_id=old.job_id)
        self.assertNotEqual(old.job_id, new.job_id)
        self.api.check_selected_tier_job(self.conn, new)
        self.assertFalse(self.api.end_selected_tier_job(self.conn, old, outcome="failed"))
        self.assertEqual(self.row()[1:3], (new.job_id, "selected"))

    def test_termination_compares_full_identity_and_is_not_completion(self) -> None:
        job = self.select()
        mismatched = dataclasses.replace(job, target=self.fixture.modules["embedding_target"].EmbeddingTarget(
            "pro", "qwen3_256", 256, "basepro",
        ))
        self.assertFalse(self.api.end_selected_tier_job(self.conn, mismatched, outcome="failed"))
        with self.assertRaises(self.api.TierJobSelectionError):
            self.api.end_selected_tier_job(self.conn, job, outcome="completed")
        self.assertEqual(self.row()[2], "selected")
        self.assertTrue(self.api.end_selected_tier_job(self.conn, job, outcome="failed"))
        self.assertFalse(self.api.end_selected_tier_job(self.conn, job, outcome="cancelled"))
        self.assertEqual(self.row()[2], "failed")
        with self.assertRaises(self.api.TierJobSelectionError):
            self.api.check_selected_tier_job(self.conn, job)

    def test_preselection_clone_reopen_is_rejected_after_stale_connection_selection(self) -> None:
        # The clone shares the source epoch but predates the marker write through
        # the accepted connection. File observations intentionally cannot tell.
        clone = self.clone()
        job = self.select()
        self.fixture.reopen_hooks()
        with patch.object(self.api.sqlite3, "connect", return_value=clone):
            with self.assertRaises(self.api.TierJobSelectionError):
                with self.api.open_selected_tier_job(job):
                    self.fail("Pre-selection clone must not reach model admission")
        self.assertTrue(clone.closed)
        self.assertEqual(self.fixture.events[-1], "release")
        self.api.check_selected_tier_job(self.conn, job)

    def test_clone_with_a_previously_selected_id_cannot_impersonate_new_selection(self) -> None:
        old = self.select()
        clone = self.clone()
        self.api.end_selected_tier_job(self.conn, old, outcome="cancelled")
        job = self.select(previous_job_id=old.job_id)
        self.fixture.reopen_hooks()
        with patch.object(self.api.sqlite3, "connect", return_value=clone):
            with self.assertRaises(self.api.TierJobSelectionError):
                with self.api.open_selected_tier_job(job):
                    self.fail("Old cloned selection must not yield")
        self.assertTrue(clone.closed)

    def test_postselection_marker_copy_is_explicitly_outside_identity_guarantee(self) -> None:
        job = self.select()
        clone = self.clone()
        self.fixture.reopen_hooks()
        with patch.object(self.api.sqlite3, "connect", return_value=clone):
            with self.api.open_selected_tier_job(job) as opened:
                self.assertIs(opened, clone)
                self.assertFalse(opened.in_transaction)
        self.assertTrue(clone.closed)

    def test_owned_selected_reopen_validates_before_yield_and_releases_on_cancellation(self) -> None:
        job = self.select()
        reopened = self.fixture.reopen_hooks()
        sentinel = Cancelled("synthetic cancellation")
        with self.assertRaises(Cancelled) as caught:
            with self.api.open_selected_tier_job(job) as conn:
                self.assertFalse(conn.in_transaction)
                self.api.check_selected_tier_job(conn, job)
                raise sentinel
        self.assertIs(caught.exception, sentinel)
        self.assertTrue(reopened[0].closed)
        self.assertEqual(self.fixture.events[-1], "release")

    def test_borrowed_transaction_never_installs_or_mutates_marker(self) -> None:
        job = self.fixture.capture()
        self.conn.execute("BEGIN")
        before = self.conn.total_changes
        for call in (lambda: self.select(), lambda: self.api.check_selected_tier_job(self.conn, job),
                     lambda: self.api.end_selected_tier_job(self.conn, job, outcome="failed")):
            with self.assertRaises(self.api.TierJobIdentityError):
                call()
            self.assertTrue(self.conn.in_transaction)
            self.assertEqual(self.conn.total_changes, before)
        self.conn.rollback()
        self.assertIsNone(self.conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.table,)).fetchone())

    def test_wrong_schema_view_temp_shadow_and_side_effects_refuse(self) -> None:
        for ddl in (f"CREATE TABLE {self.table}(singleton INTEGER PRIMARY KEY, job_id TEXT)",
                    f"CREATE VIEW {self.table} AS SELECT 1 AS singleton",
                    f"CREATE TEMP TABLE {self.table.upper()}(value TEXT)"):
            conn = self.clone()
            conn.execute(ddl)
            with self.assertRaises(self.api.TierJobIdentityError):
                self.api.select_tier_job(conn, self.target)
            self.assertFalse(conn.in_transaction)
        job = self.select()
        self.conn.execute(f"CREATE TRIGGER synthetic_marker_side_effect AFTER UPDATE ON {self.table} BEGIN SELECT 1; END")
        before = self.row()
        for call in (lambda: self.api.check_selected_tier_job(self.conn, job),
                     lambda: self.api.end_selected_tier_job(self.conn, job, outcome="failed")):
            with self.assertRaises(self.api.TierJobSelectionError):
                call()
            self.assertEqual(self.row(), before)

    def test_invalid_and_oversized_stored_values_fail_before_value_materialization(self) -> None:
        job = self.select()
        for field, value in (("model_id", "x" * 513), ("source_epoch", "x" * 129),
                             ("dimension", "not-an-integer"), ("job_id", "z" * 32),
                             ("state", "completed"), ("source_schema_signature", "z" * 64)):
            conn = self.clone()
            conn.execute("PRAGMA ignore_check_constraints=ON")
            conn.execute(f"UPDATE {self.table} SET {field}=?", (value,))
            conn.commit()
            with self.assertRaises(self.api.TierJobSelectionError):
                self.api.check_selected_tier_job(conn, job)
            self.assertFalse(conn.in_transaction)

    def test_extra_singleton_and_changed_target_refuse(self) -> None:
        job = self.select()
        conn = self.clone()
        conn.execute("PRAGMA ignore_check_constraints=ON")
        columns = ", ".join((*self.api._SELECTION_FIELDS, "state"))
        conn.execute(f"INSERT INTO {self.table} SELECT 2, {columns} FROM {self.table}")
        conn.commit()
        with self.assertRaises(self.api.TierJobSelectionError):
            self.api.check_selected_tier_job(conn, job)
        changed = dataclasses.replace(job, target=self.fixture.modules["embedding_target"].EmbeddingTarget(
            "custom", "synthetic/encoder", 384, "custom",
        ))
        with self.assertRaises(self.api.TierJobSelectionError):
            self.api.check_selected_tier_job(self.conn, changed)

    def test_embedded_nul_cannot_hide_an_oversized_payload_from_admission(self) -> None:
        job = self.select()
        for field, bound in (("model_id", 512), ("source_epoch", 128), ("source_schema_signature", 64)):
            conn = self.clone(factory=FaultConnection)
            conn.execute("PRAGMA ignore_check_constraints=ON")
            conn.execute(f"UPDATE {self.table} SET {field}=?", ("a\0" + "x" * (4 * bound),))
            conn.commit()
            self.assertEqual(conn.execute(f"SELECT length({field}) FROM {self.table}").fetchone()[0], 1)
            with self.assertRaises(self.api.TierJobSelectionError):
                self.api.check_selected_tier_job(conn, job)
            self.assertEqual(conn.selection_payload_reads, 0)
            self.assertFalse(conn.in_transaction)

    def test_marker_byte_limit_accepts_multibyte_model_identity_at_character_bound(self) -> None:
        target = self.fixture.modules["embedding_target"].EmbeddingTarget("custom", "\U00010400" * 512, 384, "custom")
        job = self.api.select_tier_job(self.conn, target)
        self.api.check_selected_tier_job(self.conn, job)
        self.assertEqual(len(job.target.model_id.encode("utf-8")), 2048)
        self.assertEqual(len(job.target.model_id), 512)

    def test_source_epoch_changed_between_capture_and_writer_refuses_without_marker(self) -> None:
        original = self.api.capture_tier_job
        def replace_after_capture(conn, target):
            job = original(conn, target)
            conn.execute("UPDATE truememory_rebuild_source_v1 SET epoch='synthetic-replacement'")
            conn.commit()
            return job
        with patch.object(self.api, "capture_tier_job", replace_after_capture):
            with self.assertRaises(self.api.TierJobSelectionError):
                self.select()
        self.assertIsNone(self.conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.table,)).fetchone())
        self.assertFalse(self.conn.in_transaction)

    def test_temp_or_path_changed_after_capture_is_rechecked_inside_writer(self) -> None:
        original = self.api.capture_tier_job
        for change in ("temp", "path"):
            conn = self.clone()
            def change_after_capture(database, target):
                job = original(database, target)
                if change == "temp":
                    database.execute("CREATE TEMP TABLE MESSAGES(id INTEGER PRIMARY KEY, content TEXT)")
                else:
                    database.fake_path = str(BASE["SYNTHETIC_PATH"].with_name("other.db"))
                return job
            with patch.object(self.api, "capture_tier_job", change_after_capture):
                with self.assertRaises(self.api.TierJobSelectionError):
                    self.api.select_tier_job(conn, self.target)
            self.assertFalse(conn.in_transaction)
            self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.table,)).fetchone())

    def test_invalid_previous_id_and_oversized_identity_make_no_marker_writes(self) -> None:
        for value in (True, 1, "invalid", "0" * 33):
            with self.assertRaises(self.api.TierJobSelectionError):
                self.select(previous_job_id=value)
        target = self.fixture.modules["embedding_target"].EmbeddingTarget("custom", "x" * 513, 384, "custom")
        with self.assertRaises(self.api.TierJobSelectionError):
            self.api.select_tier_job(self.conn, target)
        self.assertIsNone(self.conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.table,)).fetchone())
        self.assertFalse(self.conn.in_transaction)

    def test_marker_state_change_between_admission_and_read_snapshot_refuses(self) -> None:
        job = self.select()
        original = self.api.check_tier_job
        def cancel_after_identity(database, captured):
            original(database, captured)
            database.execute(f"UPDATE {self.table} SET state='cancelled'")
            database.commit()
        with patch.object(self.api, "check_tier_job", cancel_after_identity):
            with self.assertRaises(self.api.TierJobSelectionError):
                self.api.check_selected_tier_job(self.conn, job)
        self.assertFalse(self.conn.in_transaction)

    def test_retained_commit_rolls_back_and_never_returns_selected_descriptor(self) -> None:
        conn = self.clone(factory=FaultConnection)
        conn.retain_commit = True
        with self.assertRaises(self.api.TierJobSelectionError):
            self.api.select_tier_job(conn, self.target)
        self.assertFalse(conn.in_transaction)
        self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.table,)).fetchone())

    def test_source_change_after_marker_insert_rolls_back_entire_selection(self) -> None:
        original = self.api._selection_source
        calls = []
        def replace_before_final_check(database, job):
            calls.append(1)
            if len(calls) == 2:
                database.execute("UPDATE truememory_rebuild_source_v1 SET epoch='synthetic-raced-epoch'")
            original(database, job)
        old_epoch = self.fixture.capture().source_epoch
        with patch.object(self.api, "_selection_source", replace_before_final_check):
            with self.assertRaises(self.api.TierJobSelectionError):
                self.select()
        self.assertEqual(self.fixture.capture().source_epoch, old_epoch)
        self.assertIsNone(self.conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.table,)).fetchone())

    def test_selection_errors_do_not_include_source_or_target_details(self) -> None:
        conn = self.clone(factory=FaultConnection)
        conn.fault_sql = "INSERT INTO main.truememory_tier_selected"
        conn.fault = sqlite3.OperationalError("synthetic sensitive-path target text")
        with self.assertRaises(self.api.TierJobSelectionError) as caught:
            self.api.select_tier_job(conn, self.target)
        self.assertNotIn("sensitive", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_current_source_changes_do_not_rebind_the_marker_to_a_generation(self) -> None:
        job = self.select()
        before = self.row()
        self.conn.execute("INSERT INTO messages VALUES (1,'synthetic appended content')")
        self.conn.commit()
        self.api.check_selected_tier_job(self.conn, job)
        self.assertEqual(self.row(), before)
        fields = {row[1] for row in self.conn.execute(f"PRAGMA table_info({self.table})")}
        self.assertNotIn("generation", fields)
        self.assertNotIn("last_message_id", fields)
        self.assertNotIn("count", fields)

    def test_schema_creation_insert_commit_and_cancellation_rollback_together(self) -> None:
        for stage in ("CREATE TABLE truememory_tier_selected", "INSERT INTO main.truememory_tier_selected", "commit", "cancel"):
            conn = self.clone(factory=FaultConnection)
            if stage == "commit":
                conn.fail_commit = True
            else:
                conn.fault_sql = "INSERT INTO main.truememory_tier_selected" if stage == "cancel" else stage
                conn.fault = Cancelled("synthetic") if stage == "cancel" else sqlite3.OperationalError("synthetic failure")
            expected = Cancelled if stage == "cancel" else self.api.TierJobSelectionError
            with self.assertRaises(expected):
                self.api.select_tier_job(conn, self.target)
            self.assertFalse(conn.in_transaction)
            self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.table,)).fetchone())

    def test_terminal_commit_failure_preserves_selected_marker(self) -> None:
        job = self.select()
        conn = self.clone(factory=FaultConnection)
        conn.fail_commit = True
        with self.assertRaises(self.api.TierJobSelectionError):
            self.api.end_selected_tier_job(conn, job, outcome="failed")
        self.assertFalse(conn.in_transaction)
        self.assertEqual(self.row(conn)[2], "selected")

    def test_replacement_commit_failure_retains_previous_terminal_selection(self) -> None:
        old = self.select()
        self.api.end_selected_tier_job(self.conn, old, outcome="cancelled")
        conn = self.clone(factory=FaultConnection)
        conn.fail_commit = True
        before = self.row(conn)
        with self.assertRaises(self.api.TierJobSelectionError):
            self.api.select_tier_job(conn, self.target, previous_job_id=old.job_id)
        self.assertEqual(self.row(conn), before)
        self.assertFalse(conn.in_transaction)

    def test_cancellation_after_insert_rolls_back_marker_and_preserves_exception(self) -> None:
        original = self.api._selection_source
        sentinel = Cancelled("synthetic after-insert cancellation")
        checks = []
        def cancel_after_insert(database, job):
            checks.append(1)
            if len(checks) == 2:
                raise sentinel
            original(database, job)
        with patch.object(self.api, "_selection_source", cancel_after_insert):
            with self.assertRaises(Cancelled) as caught:
                self.select()
        self.assertIs(caught.exception, sentinel)
        self.assertFalse(self.conn.in_transaction)
        self.assertIsNone(self.conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.table,)).fetchone())

    def test_selected_read_rollback_uncertainty_closes_owned_connection(self) -> None:
        job = self.select()
        reopened = self.fixture.reopen_hooks()
        original = self.api._selection_row
        def retain_read_snapshot(database):
            row = original(database)
            database.rollback_failure = "retained"
            return row
        with patch.object(self.api, "_selection_row", retain_read_snapshot):
            with self.assertRaises(self.api.TierJobSelectionError):
                with self.api.open_selected_tier_job(job):
                    self.fail("Uncertain selected read must not yield")
        self.assertTrue(reopened[0].closed)
        self.assertEqual(self.fixture.events[-1], "release")

    def test_uncertain_rollback_stops_without_additional_status_write(self) -> None:
        job = self.select()
        for failure in ("retained", sqlite3.OperationalError("synthetic rollback failure")):
            conn = self.clone(factory=FaultConnection)
            conn.rollback_failure = failure
            # The identity read itself cannot release its snapshot. No terminal
            # write may follow this uncertainty.
            before = conn.total_changes
            with self.assertRaises(self.api.TierJobIdentityError):
                self.api.end_selected_tier_job(conn, job, outcome="failed")
            self.assertTrue(conn.in_transaction)
            self.assertEqual(conn.total_changes, before)
            conn.rollback_failure = None
            conn.rollback()

    def test_writer_rollback_failure_does_not_retry_or_publish_another_status(self) -> None:
        conn = self.clone(factory=FaultConnection)
        original = self.api.capture_tier_job
        def fail_writer_after_capture(database, target):
            job = original(database, target)
            database.rollback_failure = "retained"
            database.fault_sql = "INSERT INTO main.truememory_tier_selected"
            database.fault = sqlite3.OperationalError("synthetic write failure")
            return job
        with patch.object(self.api, "capture_tier_job", fail_writer_after_capture):
            with self.assertRaises(self.api.TierJobSelectionError):
                self.api.select_tier_job(conn, self.target)
        self.assertTrue(conn.in_transaction)
        self.assertEqual(conn.total_changes, 0)
        conn.rollback_failure = None
        conn.rollback()
        self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.table,)).fetchone())

    def test_selections_on_shared_memory_connection_serialize_without_stealing(self) -> None:
        uri = "file:synthetic-tier-binding?mode=memory&cache=shared"
        first = CONNECT(uri, uri=True, factory=CONNECTION)
        second = CONNECT(uri, uri=True, factory=CONNECTION)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        self.conn.backup(first)
        job = self.api.select_tier_job(first, self.target)
        with self.assertRaises(self.api.TierJobSelectionError):
            self.api.select_tier_job(second, self.target)
        self.api.check_selected_tier_job(second, job)
        self.assertTrue(self.api.end_selected_tier_job(second, job, outcome="cancelled"))
        next_job = self.api.select_tier_job(first, self.target, previous_job_id=job.job_id)
        self.assertFalse(self.api.end_selected_tier_job(second, job, outcome="failed"))
        self.api.check_selected_tier_job(first, next_job)


if __name__ == "__main__":
    unittest.main()
