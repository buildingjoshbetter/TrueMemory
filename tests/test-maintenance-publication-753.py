"""Synthetic SQLite checks for composable summary and landmark publication."""

import ast
import json
import re
import sqlite3
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_modules() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    modules = []
    for name in ("storage", "consolidation", "temporal"):
        module = types.ModuleType("synthetic_publication_" + name)
        path = ROOT / "truememory" / (name + ".py")
        tree = ast.parse(path.read_text(), filename=str(path))
        # Avoid package/native startup. All production computation and SQL run
        # unchanged; these imported search helpers are unused by these builders.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom)
            and node.module in {"truememory.storage", "truememory.fts_search"}
        )]
        exec(compile(tree, str(path), "exec"), module.__dict__)
        modules.append(module)
    return tuple(modules)


STORAGE, CONSOLIDATION, TEMPORAL = load_modules()


class PublicationFixture(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="synthetic-publication-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "synthetic.sqlite"
        self.conn = self.open_db()

    def open_db(self) -> sqlite3.Connection:
        conn = STORAGE.create_db(self.path)
        conn.execute("PRAGMA busy_timeout = 200")
        self.addCleanup(conn.close)
        return conn

    def seed(self) -> None:
        self.conn.executemany(
            "INSERT INTO messages(content,sender,recipient,timestamp) VALUES (?,?,?,?)",
            [(f"synthetic note {i:02d} is recorded here", "synthetic-sender", "synthetic-recipient",
              f"2026-01-{i + 1:02d}") for i in range(12)],
        )
        self.conn.execute(
            "INSERT INTO messages(content,sender,recipient,timestamp) VALUES "
            "('synthetic team launched Cedar and moved to Birch', 'synthetic-sender', 'synthetic-recipient', '2026-02-01')"
        )
        self.conn.execute(
            "INSERT INTO summaries(period,summary) VALUES ('monthly','synthetic previous summary')"
        )
        self.conn.execute(
            "INSERT INTO landmark_events(event_name,timestamp) VALUES ('synthetic previous event','2025-01-01')"
        )
        self.conn.commit()

    def snapshot(self, table: str, conn: sqlite3.Connection | None = None) -> list[tuple]:
        return (conn or self.conn).execute(f"SELECT * FROM {table} ORDER BY id").fetchall()

    def landmark_compute_hook(self, action):
        called = False

        def findall(pattern: str, content: str) -> list[str]:
            nonlocal called
            if not called:
                called = True
                action()
            return re.findall(pattern, content)

        return patch.object(TEMPORAL, "re", types.SimpleNamespace(
            compile=re.compile, IGNORECASE=re.IGNORECASE, findall=findall,
        ))

    def builders(self) -> tuple[tuple[str, object, str], ...]:
        return (
            ("summaries", CONSOLIDATION.build_summaries, "summaries"),
            ("structured", CONSOLIDATION.build_structured_facts, "summaries"),
            ("contradictions", CONSOLIDATION.detect_contradictions, "fact_timeline"),
            ("landmarks", TEMPORAL.detect_landmark_events, "landmark_events"),
        )


class TestSummaryPublication(PublicationFixture):
    def test_owned_periods_only_and_empty_source_clears_them(self) -> None:
        for period in ("monthly", "entity_monthly", "structured_fact", "entity_profile", "synthetic_other", None):
            self.conn.execute("INSERT INTO summaries(period,summary) VALUES (?, 'synthetic prior')", (period,))
        self.conn.commit()
        preserved = self.conn.execute(
            "SELECT * FROM summaries WHERE period NOT IN ('monthly','entity_monthly') OR period IS NULL ORDER BY id"
        ).fetchall()
        self.assertEqual(CONSOLIDATION.build_summaries(self.conn), 0)
        self.assertEqual(self.snapshot("summaries"), preserved)
        self.assertFalse(self.conn.in_transaction)

    def test_directive_only_input_is_successful_empty(self) -> None:
        self.seed()
        self.conn.execute("UPDATE messages SET directive = 1")
        self.conn.commit()
        self.assertEqual(CONSOLIDATION.build_summaries(self.conn), 0)
        self.assertEqual(self.snapshot("summaries"), [])

    def test_selection_content_dates_and_other_producers_are_preserved(self) -> None:
        self.seed()
        self.conn.execute("INSERT INTO summaries(period,summary) VALUES ('structured_fact','synthetic fact')")
        self.conn.commit()
        preserved = self.conn.execute("SELECT * FROM summaries WHERE period = 'structured_fact'").fetchall()
        self.assertEqual(CONSOLIDATION.build_summaries(self.conn), 3)
        rows = self.conn.execute(
            "SELECT period,start_date,end_date,entity,summary,key_facts,message_ids "
            "FROM summaries WHERE period IN ('monthly','entity_monthly') ORDER BY period,start_date"
        ).fetchall()
        january = [f"synthetic note {i:02d} is recorded here" for i in range(12)]
        self.assertEqual(rows[0][:6], (
            "entity_monthly", "2026-01-01", "2026-01-05", "synthetic-sender",
            "\n".join(january[:5]), "[]",
        ))
        self.assertEqual(json.loads(rows[0][6]), [1, 2, 3, 4, 5])
        self.assertEqual(rows[1][:6], (
            "monthly", "2026-01-01", "2026-01-08", "",
            "\n".join("[synthetic-sender] " + value for value in january[:8]), "[]",
        ))
        self.assertEqual(set(json.loads(rows[1][6])), set(range(1, 9)))
        self.assertEqual(rows[2][:6], (
            "monthly", "2026-02-01", "2026-02-01", "",
            "[synthetic-sender] synthetic team launched Cedar and moved to Birch", "[]",
        ))
        self.assertEqual(self.conn.execute("SELECT * FROM summaries WHERE period = 'structured_fact'").fetchall(), preserved)

    def test_summary_computation_does_not_hold_writer(self) -> None:
        self.seed()
        writer = self.open_db()
        original = CONSOLIDATION._message_salience
        observed = []

        def score(message: dict) -> float:
            if not observed:
                self.assertFalse(self.conn.in_transaction)
                writer.execute("INSERT INTO metadata(key,value) VALUES ('synthetic_writer','committed')")
                writer.commit()
                observed.append(True)
            return original(message)

        with patch.object(CONSOLIDATION, "_message_salience", score):
            CONSOLIDATION.build_summaries(self.conn)
        self.assertEqual(observed, [True])

    def test_summary_compute_failure_preserves_complete_previous_output(self) -> None:
        self.seed()
        previous = self.snapshot("summaries")
        with patch.object(CONSOLIDATION, "_message_salience", side_effect=RuntimeError("synthetic computation failure")):
            with self.assertRaises(RuntimeError):
                CONSOLIDATION.build_summaries(self.conn)
        self.assertEqual(self.snapshot("summaries"), previous)
        self.assertFalse(self.conn.in_transaction)

    def test_outer_read_snapshot_rejects_concurrent_source_change(self) -> None:
        self.seed()
        writer = self.open_db()
        previous = self.snapshot("summaries")
        original = CONSOLIDATION._message_salience
        changed = []
        self.conn.execute("BEGIN")
        self.conn.execute("SELECT revision FROM maintenance_source_state").fetchone()

        def score(message: dict) -> float:
            if not changed:
                writer.execute("UPDATE messages SET content='synthetic correction' WHERE id=1")
                writer.commit()
                changed.append(True)
            return original(message)

        with patch.object(CONSOLIDATION, "_message_salience", score):
            with self.assertRaises(sqlite3.OperationalError):
                CONSOLIDATION.build_summaries(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(changed, [True])
        self.assertEqual(self.snapshot("summaries"), previous)

    def test_summary_midpublication_failure_does_not_delete_other_producers(self) -> None:
        self.seed()
        self.conn.execute("INSERT INTO summaries(period,summary) VALUES ('structured_fact','synthetic preserved')")
        self.conn.execute(
            "CREATE TRIGGER synthetic_summary_failure BEFORE INSERT ON summaries "
            "WHEN new.period = 'entity_monthly' BEGIN SELECT RAISE(ABORT,'synthetic insertion failure'); END"
        )
        self.conn.commit()
        previous = self.snapshot("summaries")
        with self.assertRaises(sqlite3.IntegrityError):
            CONSOLIDATION.build_summaries(self.conn)
        self.conn.commit()
        self.assertEqual(self.snapshot("summaries"), previous)


class TestLandmarkPublication(PublicationFixture):
    def test_patterns_priority_context_entities_and_empty_publication(self) -> None:
        self.seed()
        self.assertEqual(TEMPORAL.detect_landmark_events(self.conn), 1)
        row = self.conn.execute(
            "SELECT event_name,timestamp,event_type,related_entities,source_message_id FROM landmark_events"
        ).fetchone()
        self.assertEqual(row, (
            "hetic team launched Cedar and moved to Birch", "2026-02-01", "move",
            '["synthetic-sender", "synthetic-recipient", "Cedar", "Birch"]', 13,
        ))
        self.conn.execute("UPDATE messages SET timestamp = ''")
        self.conn.commit()
        self.assertEqual(TEMPORAL.detect_landmark_events(self.conn), 0)
        self.assertEqual(self.snapshot("landmark_events"), [])

    def test_compute_failure_and_cancellation_preserve_previous_output(self) -> None:
        self.seed()
        previous = self.snapshot("landmark_events")
        for failure in (RuntimeError("synthetic computation failure"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                def fail() -> None:
                    raise failure
                with self.landmark_compute_hook(fail):
                    with self.assertRaises(type(failure)):
                        TEMPORAL.detect_landmark_events(self.conn)
                self.conn.commit()
                self.assertEqual(self.snapshot("landmark_events"), previous)

    def test_concurrent_source_changes_reject_output_without_blocking_writer(self) -> None:
        self.seed()
        writer = self.open_db()
        mutations = (
            "UPDATE messages SET content='synthetic changed' WHERE id=13",
            "UPDATE messages SET sender='synthetic-other' WHERE id=13",
            "UPDATE messages SET recipient='synthetic-other' WHERE id=13",
            "UPDATE messages SET timestamp='2026-03-01' WHERE id=13",
            "UPDATE messages SET id=113 WHERE id=13",
            "DELETE FROM messages WHERE id=13",
            "INSERT INTO messages(content,timestamp) VALUES ('synthetic launched Elm','2026-03-01')",
        )
        for sql in mutations:
            with self.subTest(mutation=sql.split()[0:4]):
                # Each variation uses a separate committed fixture generation.
                self.conn.execute("DELETE FROM messages")
                self.conn.execute("DELETE FROM summaries")
                self.conn.execute("DELETE FROM landmark_events")
                self.conn.execute("DELETE FROM sqlite_sequence WHERE name='messages'")
                self.conn.commit()
                self.seed()
                previous = self.snapshot("landmark_events")

                def mutate() -> None:
                    self.assertFalse(self.conn.in_transaction)
                    writer.execute(sql)
                    writer.commit()

                with self.landmark_compute_hook(mutate):
                    with self.assertRaisesRegex(sqlite3.OperationalError, "source changed"):
                        TEMPORAL.detect_landmark_events(self.conn)
                self.conn.commit()
                self.assertEqual(self.snapshot("landmark_events"), previous)

    def test_unrelated_concurrent_write_does_not_reject_stable_source(self) -> None:
        self.seed()
        writer = self.open_db()

        def mutate() -> None:
            writer.execute("INSERT INTO metadata(key,value) VALUES ('synthetic_writer','committed')")
            writer.commit()

        with self.landmark_compute_hook(mutate):
            self.assertEqual(TEMPORAL.detect_landmark_events(self.conn), 1)

    def test_caller_stale_snapshot_is_not_restarted_or_committed(self) -> None:
        self.seed()
        writer = self.open_db()
        previous = self.snapshot("landmark_events")
        self.conn.execute("BEGIN")
        self.conn.execute("SELECT count(*) FROM messages").fetchone()

        def mutate() -> None:
            writer.execute("UPDATE messages SET content='synthetic changed' WHERE id=13")
            writer.commit()

        with self.landmark_compute_hook(mutate):
            with self.assertRaises(sqlite3.OperationalError):
                TEMPORAL.detect_landmark_events(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.snapshot("landmark_events"), previous)

    def test_midpublication_failure_rolls_back_even_if_caller_later_commits(self) -> None:
        self.seed()
        self.conn.execute("INSERT INTO messages(content,timestamp) VALUES ('synthetic launched Elm','2026-03-01')")
        self.conn.execute(
            "CREATE TRIGGER synthetic_landmark_failure BEFORE INSERT ON landmark_events "
            "WHEN new.source_message_id=14 BEGIN SELECT RAISE(ABORT,'synthetic insertion failure'); END"
        )
        self.conn.commit()
        previous = self.snapshot("landmark_events")
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic_caller','retained')")
        with self.assertRaises(sqlite3.IntegrityError):
            TEMPORAL.detect_landmark_events(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.conn.commit()
        self.assertEqual(self.snapshot("landmark_events"), previous)
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic_caller'").fetchone(), ("retained",))


class TestPublicationTransactions(PublicationFixture):
    def test_readers_see_complete_old_generation_until_publication(self) -> None:
        self.seed()
        reader = self.open_db()
        for table, builder in (("summaries", CONSOLIDATION.build_summaries), ("landmark_events", TEMPORAL.detect_landmark_events)):
            with self.subTest(table=table):
                previous = self.snapshot(table, reader)
                observed = []

                def observe() -> int:
                    observed.append(self.snapshot(table, reader))
                    return 0

                self.conn.create_function("synthetic_observe", 0, observe)
                self.conn.execute(
                    f"CREATE TRIGGER synthetic_reader AFTER INSERT ON {table} "
                    "BEGIN SELECT synthetic_observe(); END"
                )
                builder(self.conn)
                self.assertTrue(observed)
                self.assertTrue(all(rows == previous for rows in observed))
                self.assertNotEqual(self.snapshot(table, reader), previous)
                self.conn.execute("DROP TRIGGER synthetic_reader")

    def test_every_builder_retains_caller_transaction_and_rollback(self) -> None:
        self.seed()
        for name, builder, table in self.builders():
            with self.subTest(builder=name):
                previous = self.snapshot(table)
                self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic_caller','pending')")
                builder(self.conn)
                self.assertTrue(self.conn.in_transaction)
                self.conn.rollback()
                self.assertEqual(self.snapshot(table), previous)
                self.assertIsNone(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic_caller'").fetchone())

    def test_final_release_failure_rolls_back_standalone_and_caller_owned(self) -> None:
        self.seed()
        for name, builder, table in self.builders():
            for caller_owned in (False, True):
                with self.subTest(builder=name, caller_owned=caller_owned):
                    self.conn.rollback()
                    self.conn.execute("DELETE FROM metadata WHERE key='synthetic_caller'")
                    self.conn.commit()
                    previous = self.snapshot(table)
                    if caller_owned:
                        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic_caller','retained')")
                    denied = []

                    def authorizer(action: int, operation: str | None, *unused: str | None) -> int:
                        if action == sqlite3.SQLITE_SAVEPOINT and operation == "RELEASE" and not denied:
                            denied.append(True)
                            return sqlite3.SQLITE_DENY
                        return sqlite3.SQLITE_OK

                    self.conn.set_authorizer(authorizer)
                    try:
                        with self.assertRaises(sqlite3.DatabaseError):
                            builder(self.conn)
                    finally:
                        self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                    self.assertEqual(denied, [True])
                    self.assertEqual(self.conn.in_transaction, caller_owned)
                    self.conn.commit()
                    self.assertEqual(self.snapshot(table), previous)
                    if caller_owned:
                        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic_caller'").fetchone(), ("retained",))
                        self.conn.execute("DELETE FROM metadata WHERE key='synthetic_caller'")
                        self.conn.commit()

    def test_actual_outermost_commit_constraint_failure_restores_output(self) -> None:
        self.seed()
        self.conn.execute("CREATE TABLE synthetic_parent(id INTEGER PRIMARY KEY)")
        self.conn.execute(
            "CREATE TABLE synthetic_child(parent_id INTEGER REFERENCES synthetic_parent(id) DEFERRABLE INITIALLY DEFERRED)"
        )
        for table, builder in (("summaries", CONSOLIDATION.build_summaries), ("landmark_events", TEMPORAL.detect_landmark_events)):
            with self.subTest(table=table):
                self.conn.rollback()
                self.conn.execute("DROP TRIGGER IF EXISTS synthetic_commit_failure")
                self.conn.execute(
                    f"CREATE TRIGGER synthetic_commit_failure AFTER INSERT ON {table} "
                    "BEGIN INSERT INTO synthetic_child VALUES (1); END"
                )
                previous = self.snapshot(table)
                with self.assertRaises(sqlite3.IntegrityError):
                    builder(self.conn)
                self.assertFalse(self.conn.in_transaction)
                self.assertEqual(self.snapshot(table), previous)
                self.assertEqual(self.conn.execute("SELECT * FROM synthetic_child").fetchall(), [])
                self.conn.execute("DROP TRIGGER synthetic_commit_failure")

    def test_failed_rollback_does_not_release_and_commit_failed_output(self) -> None:
        self.seed()
        for table, builder in (("summaries", CONSOLIDATION.build_summaries), ("landmark_events", TEMPORAL.detect_landmark_events)):
            with self.subTest(table=table):
                self.conn.rollback()
                self.conn.execute("DROP TRIGGER IF EXISTS synthetic_write_failure")
                self.conn.execute(
                    f"CREATE TRIGGER synthetic_write_failure BEFORE INSERT ON {table} "
                    "BEGIN SELECT RAISE(ABORT,'synthetic write failure'); END"
                )
                previous = self.snapshot(table)
                observed = []

                def authorizer(action: int, operation: str | None, *unused: str | None) -> int:
                    if action == sqlite3.SQLITE_SAVEPOINT:
                        observed.append(operation)
                        if operation == "ROLLBACK":
                            return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK

                self.conn.set_authorizer(authorizer)
                try:
                    with self.assertRaises(sqlite3.DatabaseError):
                        builder(self.conn)
                finally:
                    self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                self.assertTrue(self.conn.in_transaction)
                self.assertNotIn("RELEASE", observed)
                self.conn.rollback()
                self.assertEqual(self.snapshot(table), previous)
                self.conn.execute("DROP TRIGGER synthetic_write_failure")

    def test_cancellation_inside_shared_savepoint_restores_previous_output(self) -> None:
        self.seed()
        previous = self.snapshot("summaries")
        with self.assertRaises(KeyboardInterrupt):
            with CONSOLIDATION._consolidation_write(self.conn, "synthetic_cancel"):
                self.conn.execute("DELETE FROM summaries")
                raise KeyboardInterrupt()
        self.conn.commit()
        self.assertEqual(self.snapshot("summaries"), previous)


if __name__ == "__main__":
    unittest.main()
