"""Surprise-index transaction regressions using SQLite and synthetic messages."""
from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
import sqlite3
import tempfile
import threading
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "truememory" / "predictive.py"


def load_predictive() -> types.ModuleType:
    # Execute the production stdlib module without importing the package,
    # whose initialization can load optional native retrieval dependencies.
    module = types.ModuleType("synthetic_predictive_750")
    exec(compile(ast.parse(SOURCE.read_text()), str(SOURCE), "exec"), module.__dict__)
    return module


class SurpriseIndexTransactions(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="tm-surprise-750-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "synthetic.sqlite"
        self.conn = sqlite3.connect(self.path, timeout=0.5, check_same_thread=False)
        self.addCleanup(self.conn.close)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                timestamp TEXT DEFAULT ''
            );
            CREATE TABLE notes (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE surprise_scores (
                message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
                surprise REAL NOT NULL DEFAULT 0.0,
                fact_count INTEGER NOT NULL DEFAULT 0,
                new_fact_count INTEGER NOT NULL DEFAULT 0
            );
        """)
        self.conn.executemany(
            "INSERT INTO messages(content,timestamp) VALUES (?,?)",
            [
                ("Meridian launched in Austin and raised $1.5M.", "2026-02-02"),
                ("Meridian raised $1.5M.", "2026-02-03"),
                ("okay", "2026-02-03"),
            ],
        )
        self.conn.commit()
        self.predictive = load_predictive()
        self.predictive.build_surprise_index(self.conn)

    def other(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=0.5)
        connection.execute("PRAGMA foreign_keys=ON")
        self.addCleanup(connection.close)
        return connection

    def stored(self, conn: sqlite3.Connection | None = None) -> list[tuple]:
        return (conn or self.conn).execute(
            "SELECT message_id,surprise,fact_count,new_fact_count "
            "FROM surprise_scores ORDER BY message_id"
        ).fetchall()

    def reference(self) -> tuple[dict[int, float], list[tuple]]:
        rows = self.conn.execute(
            "SELECT id,content,timestamp FROM messages ORDER BY timestamp,id"
        ).fetchall()
        expected, stored = {}, []
        for index, (msg_id, content, _timestamp) in enumerate(rows):
            # Independent prefix definition of chronological surprise. Small
            # fixtures can recompute the whole prefix instead of sharing the
            # production accumulator implementation.
            prefix_facts = set().union(*(
                self.predictive.extract_facts(previous[1]) for previous in rows[:index]
            ))
            facts = self.predictive.extract_facts(content)
            score = self.predictive.compute_surprise_score(content, prefix_facts)
            expected[msg_id] = score
            stored.append((msg_id, round(score, 4), len(facts), len(facts - prefix_facts)))
        return expected, sorted(stored)

    def test_chronological_reference_after_add_update_delete_and_backdate(self) -> None:
        mutations = [
            ("SELECT 1", ()),
            ("INSERT INTO messages(content,timestamp) VALUES (?,?)",
             ("Orion moved to Denver and raised $2M.", "2026-03-01")),
            ("UPDATE messages SET content=? WHERE id=1",
             ("Meridian launched in Boston and raised $3M.",)),
            ("DELETE FROM messages WHERE id=2", ()),
            ("INSERT INTO messages(content,timestamp) VALUES (?,?)",
             ("Orion raised $2M.", "2026-01-01")),
            ("UPDATE messages SET timestamp=? WHERE id=1", ("2025-12-01",)),
        ]
        for sql, values in mutations:
            with self.subTest(sql=sql):
                self.conn.execute(sql, values)
                self.conn.commit()
                expected, stored = self.reference()
                returned = self.predictive.build_surprise_index(self.conn)
                self.assertEqual(list(returned), list(expected))
                self.assertEqual(returned, expected)
                self.assertEqual(self.stored(), stored)

    def test_full_precision_return_and_four_decimal_storage(self) -> None:
        with patch.object(self.predictive, "compute_surprise_score", return_value=0.123456789):
            scores = self.predictive.build_surprise_index(self.conn)
        self.assertTrue(all(score == 0.123456789 for score in scores.values()))
        self.assertTrue(all(row[1] == 0.1235 for row in self.stored()))

    def during_compute(
        self, action: Callable[[sqlite3.Connection], None],
    ) -> tuple[list[object], bool]:
        entered, release = threading.Event(), threading.Event()
        outcomes: list[object] = []
        original = self.predictive.compute_surprise_score
        paused = False

        def score(content: str, facts: set) -> float:
            nonlocal paused
            if not paused:
                paused = True
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("Synthetic compute watchdog expired")
            return original(content, facts)

        def run() -> None:
            try:
                outcomes.append(self.predictive.build_surprise_index(self.conn))
            except BaseException as error:
                outcomes.append(error)

        with patch.object(self.predictive, "compute_surprise_score", side_effect=score):
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(entered.wait(2), "Production compute path was not entered")
                held_transaction = self.conn.in_transaction
                action(self.other())
            finally:
                release.set()
                worker.join(3)
                self.assertFalse(worker.is_alive(), "Synthetic worker did not stop")
        self.assertEqual(len(outcomes), 1)
        return outcomes, held_transaction

    def test_foreground_writer_completes_while_compute_is_paused(self) -> None:
        def write(other: sqlite3.Connection) -> None:
            other.execute("INSERT INTO notes VALUES (1,'synthetic foreground write')")
            other.commit()

        isolation = self.conn.isolation_level
        outcomes, held_transaction = self.during_compute(write)
        self.assertFalse(held_transaction)
        self.assertIsInstance(outcomes[0], dict)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0], 1)
        self.assertEqual(self.conn.isolation_level, isolation)

    def test_concurrent_update_delete_and_backdated_insert_reject_stale_output(self) -> None:
        mutations = [
            ("UPDATE messages SET content=? WHERE id=1", ("Corrected synthetic content",)),
            ("UPDATE messages SET timestamp=? WHERE id=1", ("2024-01-01",)),
            ("DELETE FROM messages WHERE id=2", ()),
            ("INSERT INTO messages(content,timestamp) VALUES (?,?)",
             ("Backdated synthetic evidence", "2020-01-01")),
            ("INSERT INTO messages(content,timestamp) VALUES (?,?)",
             ("Later synthetic evidence", "2030-01-01")),
        ]
        for sql, values in mutations:
            with self.subTest(sql=sql):
                previous: list[tuple] = []

                def write(other: sqlite3.Connection) -> None:
                    other.execute(sql, values)
                    other.commit()
                    previous.extend(self.stored(other))

                outcomes, held_transaction = self.during_compute(write)
                self.assertFalse(held_transaction)
                self.assertIsInstance(outcomes[0], sqlite3.OperationalError)
                self.assertIn("source changed", str(outcomes[0]))
                self.assertIn("retry", str(outcomes[0]))
                self.conn.commit()  # The engine's later commit cannot publish failed output.
                self.assertEqual(self.stored(), previous)
                expected, stored = self.reference()
                self.assertEqual(self.predictive.build_surprise_index(self.conn), expected)
                self.assertEqual(self.stored(), stored)

    def test_mid_compute_failure_preserves_previous_index_after_commit(self) -> None:
        previous = self.stored()
        original = self.predictive.compute_surprise_score
        calls = 0

        def fail(content: str, facts: set) -> float:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Synthetic compute failure")
            return original(content, facts)

        with patch.object(self.predictive, "compute_surprise_score", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "Synthetic compute failure"):
                self.predictive.build_surprise_index(self.conn)
        self.conn.commit()
        self.assertEqual(self.stored(), previous)

    def fail_second_insert(self) -> None:
        self.conn.execute("""
            CREATE TEMP TRIGGER fail_surprise_insert BEFORE INSERT ON surprise_scores
            WHEN NEW.message_id = 2
            BEGIN SELECT RAISE(ABORT, 'synthetic insert failure'); END
        """)

    def test_mid_write_failure_restores_complete_previous_index(self) -> None:
        previous = self.stored()
        self.fail_second_insert()
        with self.assertRaisesRegex(sqlite3.IntegrityError, "synthetic insert failure"):
            self.predictive.build_surprise_index(self.conn)
        self.assertFalse(self.conn.in_transaction)
        self.conn.commit()
        self.assertEqual(self.stored(), previous)

    def test_final_source_validation_holds_writer_ownership(self) -> None:
        connection = self.conn
        other = self.other()
        other.execute("PRAGMA busy_timeout=20")
        observed = []

        class Witness:
            source_reads = 0

            def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                if sql.startswith("SELECT id, content, timestamp FROM messages"):
                    self.source_reads += 1
                    if self.source_reads == 2:
                        try:
                            other.execute("UPDATE messages SET content='Stale writer' WHERE id=1")
                        except sqlite3.OperationalError:
                            observed.append("writer blocked")
                        else:
                            other.rollback()
                            observed.append("writer admitted")
                return connection.execute(sql, *args)

            def executemany(self, sql: str, values: list[tuple]) -> sqlite3.Cursor:
                return connection.executemany(sql, values)

        self.predictive.build_surprise_index(Witness())
        self.assertEqual(observed, ["writer blocked"])
        other.execute("UPDATE messages SET content='Writer resumes' WHERE id=1")
        other.commit()

    def test_mid_write_cancellation_rolls_back_before_caller_commit(self) -> None:
        connection = self.conn
        previous = self.stored()

        class Interrupted:
            def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                return connection.execute(sql, *args)

            def executemany(self, sql: str, values: list[tuple]) -> sqlite3.Cursor:
                connection.execute(sql, values[0])
                raise KeyboardInterrupt("Synthetic cancellation")

        with self.assertRaisesRegex(KeyboardInterrupt, "Synthetic cancellation"):
            self.predictive.build_surprise_index(Interrupted())
        self.conn.commit()
        self.assertEqual(self.stored(), previous)

    def test_release_denial_restores_old_index_before_later_commit(self) -> None:
        previous = self.stored()
        self.conn.execute("UPDATE messages SET content='okay' WHERE id=1")
        self.conn.commit()
        self.assertNotEqual(self.reference()[1], previous)
        for caller_owned in (False, True):
            with self.subTest(caller_owned=caller_owned):
                if caller_owned:
                    self.conn.execute("INSERT INTO notes VALUES (1,'caller release test')")
                releases = []

                def deny_first_release(
                    action: int, operation: str | None, name: str | None,
                    _database: str | None, _source: str | None,
                ) -> int:
                    if (action == sqlite3.SQLITE_SAVEPOINT and operation == "RELEASE"
                            and name == "truememory_surprise_publish"):
                        releases.append(operation)
                        if len(releases) == 1:
                            return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK

                self.conn.set_authorizer(deny_first_release)
                try:
                    with self.assertRaises(sqlite3.DatabaseError):
                        self.predictive.build_surprise_index(self.conn)
                finally:
                    # Disabling with None is supported only on Python 3.11+.
                    self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                self.assertEqual(len(releases), 2)
                self.assertEqual(self.conn.in_transaction, caller_owned)
                self.assertEqual(self.stored(), previous)
                self.conn.commit()
                other = self.other()
                self.assertEqual(self.stored(other), previous)
                self.assertEqual(other.execute("SELECT COUNT(*) FROM notes").fetchone()[0], int(caller_owned))

    def test_release_cancellation_restores_old_index_before_later_commit(self) -> None:
        connection = self.conn
        previous = self.stored()
        self.conn.execute("UPDATE messages SET content='okay' WHERE id=1")
        self.conn.commit()
        self.assertNotEqual(self.reference()[1], previous)
        for caller_owned in (False, True):
            with self.subTest(caller_owned=caller_owned):
                if caller_owned:
                    self.conn.execute("INSERT INTO notes VALUES (1,'caller cancellation test')")

                class InterruptedRelease:
                    releases = 0

                    def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                        if sql == "RELEASE SAVEPOINT truememory_surprise_publish":
                            self.releases += 1
                            if self.releases == 1:
                                raise KeyboardInterrupt("Synthetic release cancellation")
                        return connection.execute(sql, *args)

                    def executemany(self, sql: str, values: list[tuple]) -> sqlite3.Cursor:
                        return connection.executemany(sql, values)

                interrupted = InterruptedRelease()
                with self.assertRaisesRegex(KeyboardInterrupt, "Synthetic release cancellation"):
                    self.predictive.build_surprise_index(interrupted)
                self.assertEqual(interrupted.releases, 2)
                self.assertEqual(self.conn.in_transaction, caller_owned)
                self.assertEqual(self.stored(), previous)
                self.conn.commit()
                other = self.other()
                self.assertEqual(self.stored(other), previous)
                self.assertEqual(other.execute("SELECT COUNT(*) FROM notes").fetchone()[0], int(caller_owned))

    def test_caller_transaction_success_remains_uncommitted_and_can_rollback(self) -> None:
        previous = self.stored()
        self.conn.execute("INSERT INTO notes VALUES (1,'uncommitted caller work')")
        self.conn.execute("UPDATE messages SET content='Changed caller evidence' WHERE id=1")
        self.predictive.build_surprise_index(self.conn)
        self.assertTrue(self.conn.in_transaction)
        other = self.other()
        self.assertEqual(other.execute("SELECT COUNT(*) FROM notes").fetchone()[0], 0)
        self.assertEqual(self.stored(other), previous)
        self.conn.rollback()
        self.assertEqual(self.stored(), previous)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0], 0)

    def test_caller_transaction_failure_keeps_unrelated_writes_for_caller_commit(self) -> None:
        previous = self.stored()
        self.fail_second_insert()
        self.conn.execute("INSERT INTO notes VALUES (1,'caller owns this write')")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "synthetic insert failure"):
            self.predictive.build_surprise_index(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.stored(), previous)
        self.conn.commit()
        other = self.other()
        self.assertEqual(other.execute("SELECT COUNT(*) FROM notes").fetchone()[0], 1)
        self.assertEqual(self.stored(other), previous)

    def test_stale_caller_wal_snapshot_is_not_restarted_or_committed(self) -> None:
        self.conn.execute("BEGIN")
        source = self.conn.execute("SELECT content FROM messages WHERE id=1").fetchone()
        other = self.other()
        other.execute("UPDATE messages SET content='Concurrent corrected evidence' WHERE id=1")
        other.commit()
        previous = self.stored(other)
        with self.assertRaises(sqlite3.OperationalError):
            self.predictive.build_surprise_index(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=1").fetchone(), source)
        self.conn.rollback()
        self.assertEqual(self.stored(other), previous)
        expected, stored = self.reference()
        self.assertEqual(self.predictive.build_surprise_index(self.conn), expected)
        self.assertEqual(self.stored(), stored)

    def test_empty_corpus_clears_existing_orphan_scores(self) -> None:
        self.conn.execute("DELETE FROM messages")
        self.conn.commit()
        self.conn.execute("PRAGMA foreign_keys=OFF")
        self.conn.execute("INSERT INTO surprise_scores VALUES (999,0.5,1,1)")
        self.conn.commit()
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.assertEqual(self.predictive.build_surprise_index(self.conn), {})
        self.assertEqual(self.stored(), [])

    def test_schema_creation_nests_in_caller_transaction(self) -> None:
        self.conn.execute("DROP TABLE surprise_scores")
        self.conn.execute("INSERT INTO notes VALUES (1,'caller schema test')")
        self.predictive.build_surprise_index(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.other().execute("SELECT COUNT(*) FROM notes").fetchone()[0], 0)
        self.conn.rollback()
        self.assertIsNone(self.conn.execute(
            "SELECT name FROM sqlite_master WHERE name='surprise_scores'"
        ).fetchone())

    def test_schema_helper_never_commits_unrelated_caller_work(self) -> None:
        self.conn.execute("INSERT INTO notes VALUES (1,'caller helper test')")
        self.predictive._ensure_surprise_table(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.other().execute("SELECT COUNT(*) FROM notes").fetchone()[0], 0)
        self.conn.rollback()


if __name__ == "__main__":
    unittest.main()
