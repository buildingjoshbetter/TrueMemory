"""Real SQLite regressions, with only the two stdlib production modules loaded."""

import ast
import sqlite3
import tempfile
import threading
import types
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_modules() -> tuple[types.ModuleType, types.ModuleType]:
    storage = types.ModuleType("synthetic_episode_storage")
    temporal = types.ModuleType("synthetic_episode_temporal")
    for module, name in ((storage, "storage"), (temporal, "temporal")):
        path = ROOT / "truememory" / f"{name}.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        # Avoid package initialization and optional native dependencies. All
        # production functions and SQL in these two modules remain unchanged.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == "truememory.storage"
        )]
        if name == "temporal":
            module._row_to_dict = storage._row_to_dict
            module.select_message_cols = storage.select_message_cols
        exec(compile(tree, str(path), "exec"), module.__dict__)
    return storage, temporal


STORAGE, TEMPORAL = load_modules()

LEGACY_TRIGGER = """
CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content, sender, recipient, category, modality)
    VALUES (new.id, new.content, new.sender, new.recipient, new.category, new.modality);
END;
"""

AUDIT_SQL = """
CREATE TABLE synthetic_write_audit (event TEXT, row_id INTEGER);
CREATE TRIGGER synthetic_message_update AFTER UPDATE ON messages BEGIN
    INSERT INTO synthetic_write_audit VALUES ('message_update', new.id);
END;
CREATE TRIGGER synthetic_episode_insert AFTER INSERT ON episodes BEGIN
    INSERT INTO synthetic_write_audit VALUES ('episode_insert', new.id);
END;
CREATE TRIGGER synthetic_episode_update AFTER UPDATE ON episodes BEGIN
    INSERT INTO synthetic_write_audit VALUES ('episode_update', new.id);
END;
CREATE TRIGGER synthetic_episode_delete AFTER DELETE ON episodes BEGIN
    INSERT INTO synthetic_write_audit VALUES ('episode_delete', old.id);
END;
CREATE TRIGGER synthetic_fts_insert AFTER INSERT ON messages_fts_content BEGIN
    INSERT INTO synthetic_write_audit VALUES ('fts_insert', new.id);
END;
CREATE TRIGGER synthetic_fts_delete AFTER DELETE ON messages_fts_content BEGIN
    INSERT INTO synthetic_write_audit VALUES ('fts_delete', old.id);
END;
CREATE TRIGGER synthetic_fts_update AFTER UPDATE ON messages_fts_content BEGIN
    INSERT INTO synthetic_write_audit VALUES ('fts_update', new.id);
END;
"""


class EpisodeFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-episodes-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.sqlite"
        self.conn = self.open_db()
        self.conn.executescript(AUDIT_SQL)

    def open_db(self) -> sqlite3.Connection:
        conn = STORAGE.create_db(self.path)
        self.addCleanup(conn.close)
        return conn

    def add(self, timestamps: list[str | None]) -> list[int]:
        ids = []
        for timestamp in timestamps:
            cursor = self.conn.execute(
                "INSERT INTO messages(content, sender, recipient, timestamp, category, modality) "
                "VALUES ('synthetic cedar', 'senderold', 'recipientold', ?, 'categoryold', 'modalityold')",
                (timestamp,),
            )
            ids.append(cursor.lastrowid)
        self.conn.commit()
        self.clear_audit()
        return ids

    def clear_audit(self) -> None:
        self.conn.execute("DELETE FROM synthetic_write_audit")
        self.conn.commit()

    def audit(self) -> dict[str, int]:
        return dict(self.conn.execute(
            "SELECT event, count(*) FROM synthetic_write_audit GROUP BY event"
        ))

    def search(self, query: str, conn: sqlite3.Connection | None = None) -> list[int]:
        return [row[0] for row in (conn or self.conn).execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? ORDER BY rowid", (query,)
        )]

    def snapshot(self) -> tuple[list[tuple], list[tuple], list[tuple]]:
        return (
            self.conn.execute("SELECT * FROM messages ORDER BY id").fetchall(),
            self.conn.execute("SELECT * FROM episodes ORDER BY id").fetchall(),
            self.conn.execute("SELECT rowid, * FROM messages_fts ORDER BY rowid").fetchall(),
        )

    def groups(self) -> list[tuple[int, tuple[int, ...]]]:
        return [(row[0], tuple(member[0] for member in self.conn.execute(
            "SELECT id FROM messages WHERE episode_id = ? ORDER BY id", (row[0],)
        ))) for row in self.conn.execute("SELECT id FROM episodes ORDER BY id").fetchall()]

    def start_worker(self, action: Callable[[], object]) -> tuple[threading.Thread, list[object]]:
        outcomes = []

        def run() -> None:
            try:
                outcomes.append(action())
            except Exception as error:  # Return worker failures to the asserting test.
                outcomes.append(error)

        worker = threading.Thread(target=run)
        worker.start()
        self.addCleanup(worker.join, 3)
        return worker, outcomes


class TestFtsUpdateWrites(EpisodeFixture):
    def test_metadata_and_equal_indexed_values_do_not_rewrite_fts(self) -> None:
        ids = self.add(["2026-01-01T00:00:00"])
        self.conn.execute(
            "UPDATE messages SET episode_id = 7, timestamp = '2026-01-02', "
            "emotional_valence = 0.5, directive = 1, metadata = '{\"synthetic\": true}'"
        )
        self.conn.execute(
            "UPDATE messages SET id = id, content = content, sender = sender, "
            "recipient = recipient, category = category, modality = modality"
        )
        self.assertEqual(self.audit(), {"message_update": 2})
        self.assertEqual(self.search("cedar"), ids)

    def test_each_indexed_value_updates_real_search_results(self) -> None:
        ids = self.add(["2026-01-01"])
        for column in ("content", "sender", "recipient", "category", "modality"):
            with self.subTest(column=column):
                old = "cedar" if column == "content" else f"{column}old"
                new = f"{column}new"
                self.clear_audit()
                self.conn.execute(f"UPDATE messages SET {column} = ?", (new,))
                self.assertEqual(self.search(f"{column}:{old}"), [])
                self.assertEqual(self.search(f"{column}:{new}"), ids)
                self.assertEqual(self.audit(), {
                    "fts_delete": 1, "fts_insert": 1, "message_update": 1,
                })

    def test_nullable_indexed_values_use_null_safe_comparison(self) -> None:
        ids = self.add(["2026-01-01"])
        for column in ("sender", "recipient", "category", "modality"):
            with self.subTest(column=column):
                self.conn.execute(f"UPDATE messages SET {column} = NULL")
                self.assertEqual(self.search(f"{column}:{column}old"), [])
                self.clear_audit()
                self.conn.execute(f"UPDATE messages SET {column} = NULL")
                self.assertEqual(self.audit(), {"message_update": 1})
                self.clear_audit()
                self.conn.execute(f"UPDATE messages SET {column} = ?", (f"{column}new",))
                self.assertEqual(self.search(f"{column}:{column}new"), ids)
                self.assertEqual(self.audit().get("fts_insert"), 1)

    def test_id_and_all_rowid_aliases_keep_fts_rowids_current(self) -> None:
        self.add(["2026-01-01"])
        for new_id, alias in enumerate(("id", "rowid", "_rowid_", "oid"), 101):
            with self.subTest(alias=alias):
                self.clear_audit()
                self.conn.execute(f"UPDATE messages SET {alias} = ?", (new_id,))
                self.assertEqual(self.search("cedar"), [new_id])
                self.assertEqual(self.conn.execute(
                    "SELECT rowid FROM messages_fts"
                ).fetchall(), [(new_id,)])
                self.assertEqual(self.audit().get("fts_delete"), 1)

    def install_legacy(self) -> None:
        self.conn.execute("DROP TRIGGER messages_au")
        self.conn.execute(LEGACY_TRIGGER)
        self.conn.commit()

    def test_existing_database_migration_preserves_data_and_is_idempotent(self) -> None:
        self.add(["2026-01-01"])
        self.install_legacy()
        before = self.snapshot()
        version = self.conn.execute("PRAGMA schema_version").fetchone()[0]
        opened = self.open_db()
        self.assertEqual(opened.execute("PRAGMA schema_version").fetchone()[0], version + 2)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.audit(), {})
        reopened = self.open_db()
        self.assertEqual(reopened.execute("PRAGMA schema_version").fetchone()[0], version + 2)
        reopened.execute("UPDATE messages SET episode_id = 5")
        reopened.commit()
        self.assertEqual(self.audit(), {"message_update": 1})
        self.assertEqual(self.search("cedar", reopened), [1])

    def test_failed_migration_restores_old_trigger_and_caller_transaction(self) -> None:
        self.add(["2026-01-01"])
        self.install_legacy()
        before = self.snapshot()
        old_sql = self.conn.execute("SELECT sql FROM sqlite_master WHERE name = 'messages_au'").fetchone()[0]

        def deny_create(action: int, name: str, *unused: str | None) -> int:
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_TRIGGER and name == "messages_au" else sqlite3.SQLITE_OK

        for caller_owned in (False, True):
            with self.subTest(caller_owned=caller_owned):
                if caller_owned:
                    self.conn.execute("INSERT INTO metadata(key, value) VALUES ('synthetic-uncommitted', 'pending')")
                self.conn.set_authorizer(deny_create)
                try:
                    with self.assertRaises(sqlite3.DatabaseError):
                        STORAGE._migrate_messages_fts_trigger(self.conn)
                finally:
                    # Disabling with None is supported only on Python 3.11+.
                    self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                self.assertEqual(self.conn.in_transaction, caller_owned)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(self.conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name = 'messages_au'"
                ).fetchone()[0], old_sql)
                if caller_owned:
                    self.assertEqual(self.conn.execute(
                        "SELECT value FROM metadata WHERE key = 'synthetic-uncommitted'"
                    ).fetchone(), ("pending",))
                    self.conn.rollback()

    def test_simultaneous_migrators_recheck_after_writer_lock(self) -> None:
        self.install_legacy()
        second = sqlite3.connect(self.path, check_same_thread=False)
        self.addCleanup(second.close)
        second.execute("PRAGMA busy_timeout = 2000")
        entered = threading.Event()
        release = threading.Event()
        second_started = threading.Event()
        traces = [[], []]

        def first_trace(statement: str) -> None:
            traces[0].append(statement)
            if statement.startswith("DROP TRIGGER"):
                entered.set()
                release.wait(2)

        def second_trace(statement: str) -> None:
            traces[1].append(statement)
            if statement == "BEGIN IMMEDIATE":
                second_started.set()

        self.conn.set_trace_callback(first_trace)
        second.set_trace_callback(second_trace)
        first, first_results = self.start_worker(lambda: STORAGE._migrate_messages_fts_trigger(self.conn))
        self.addCleanup(release.set)
        self.assertTrue(entered.wait(2))
        other, other_results = self.start_worker(lambda: STORAGE._migrate_messages_fts_trigger(second))
        self.assertTrue(second_started.wait(2))
        release.set()
        first.join(2)
        other.join(2)
        self.assertEqual(first_results, [None])
        self.assertEqual(other_results, [None])
        self.assertEqual(sum(sql.startswith("DROP TRIGGER") for sql in traces[0]), 1)
        self.assertFalse(any(sql.startswith("DROP TRIGGER") for sql in traces[1]))


class TestEpisodeDeltas(EpisodeFixture):
    def test_first_pass_assigns_once_and_unchanged_pass_writes_nothing(self) -> None:
        ids = self.add([f"2026-01-01T{hour:02}:00:00" for hour in range(12)])
        self.assertEqual(TEMPORAL.detect_episodes(self.conn), 1)
        self.assertEqual(self.audit(), {"episode_insert": 1, "message_update": 12})
        self.assertEqual(self.search("cedar"), ids)
        before = self.snapshot()
        self.clear_audit()
        changes = self.conn.total_changes
        self.assertEqual(TEMPORAL.detect_episodes(self.conn), 1)
        self.assertEqual(self.conn.total_changes, changes)
        self.assertEqual(self.audit(), {})
        self.assertEqual(self.snapshot(), before)

    def test_original_gap_timezone_invalid_and_empty_semantics(self) -> None:
        ids = self.add([
            None, "", "0000", "2026-01-01T00:00:00+05:00",
            "2026-01-01T06:00:00Z", "2026-01-01T12:00:01-05:00",
            "2026-01-01T18:00:01", "not-a-time", "zzz",
        ])
        self.assertEqual(TEMPORAL.detect_episodes(self.conn), 2)
        self.assertEqual([members for _, members in self.groups()], [tuple(ids[2:5]), tuple(ids[5:])])
        self.assertEqual(self.conn.execute(
            "SELECT id FROM messages WHERE episode_id IS NULL ORDER BY id"
        ).fetchall(), [(ids[0],), (ids[1],)])

    def test_addition_changes_only_affected_group_and_preserves_other_id(self) -> None:
        ids = self.add(["2026-01-01T00:00:00", "2026-01-01T01:00:00", "2026-01-03T00:00:00"])
        TEMPORAL.detect_episodes(self.conn)
        old = self.groups()
        added = self.add(["2026-01-01T02:00:00"])[0]
        TEMPORAL.detect_episodes(self.conn)
        new = self.groups()
        self.assertIn(old[1], new)
        self.assertNotIn(old[0][0], [eid for eid, _ in new])
        self.assertIn(tuple(ids[:2] + [added]), [members for _, members in new])
        self.assertEqual(self.audit(), {"episode_delete": 1, "episode_insert": 1, "message_update": 3})

    def test_merge_and_split_use_new_ids(self) -> None:
        ids = self.add(["2026-01-01T00:00:00", "2026-01-01T12:00:00"])
        TEMPORAL.detect_episodes(self.conn)
        old_ids = {eid for eid, _ in self.groups()}
        bridge = self.add(["2026-01-01T06:00:00"])[0]
        self.assertEqual(TEMPORAL.detect_episodes(self.conn), 1)
        merged_id = self.groups()[0][0]
        self.assertNotIn(merged_id, old_ids)
        self.assertEqual(self.groups()[0][1], tuple(ids + [bridge]))
        self.clear_audit()
        self.assertEqual(TEMPORAL.detect_episodes(self.conn, gap_hours=5), 3)
        self.assertTrue(all(eid not in old_ids | {merged_id} for eid, _ in self.groups()))
        self.assertEqual(self.audit(), {"episode_delete": 1, "episode_insert": 3, "message_update": 3})

    def test_deleted_member_does_not_reuse_incomplete_old_episode(self) -> None:
        ids = self.add(["2026-01-01T00:00:00", "2026-01-01T01:00:00"])
        TEMPORAL.detect_episodes(self.conn)
        old_id = self.groups()[0][0]
        self.conn.execute("DELETE FROM messages WHERE id = ?", (ids[1],))
        self.clear_audit()
        TEMPORAL.detect_episodes(self.conn)
        self.assertNotEqual(self.groups()[0][0], old_id)
        self.assertEqual(self.groups()[0][1], (ids[0],))
        self.assertEqual(self.audit(), {"episode_delete": 1, "episode_insert": 1, "message_update": 1})

    def test_timestamp_change_with_same_members_updates_only_bounds(self) -> None:
        self.add(["2026-01-01T00:00:00", "2026-01-01T01:00:00"])
        TEMPORAL.detect_episodes(self.conn)
        old = self.groups()
        self.conn.execute("UPDATE messages SET timestamp = '2026-01-01T02:00:00' WHERE id = 2")
        self.clear_audit()
        TEMPORAL.detect_episodes(self.conn)
        self.assertEqual(self.groups(), old)
        self.assertEqual(self.audit(), {"episode_update": 1})
        self.assertEqual(self.conn.execute("SELECT end_time FROM episodes").fetchone(), ("2026-01-01T02:00:00",))

    def test_nonempty_summary_invalidated_once_even_with_same_member_ids(self) -> None:
        self.add(["2026-01-01"])
        TEMPORAL.detect_episodes(self.conn)
        old = self.groups()
        self.conn.execute("UPDATE episodes SET summary = 'synthetic derived summary'")
        self.conn.execute("UPDATE messages SET content = 'synthetic maple'")
        self.clear_audit()
        TEMPORAL.detect_episodes(self.conn)
        self.assertEqual(self.groups(), old)
        self.assertEqual(self.conn.execute("SELECT summary FROM episodes").fetchone(), ("",))
        self.assertEqual(self.audit(), {"episode_update": 1})
        self.clear_audit()
        TEMPORAL.detect_episodes(self.conn)
        self.assertEqual(self.audit(), {})

    def test_zero_eligible_rows_remove_stale_assignments_without_deleting_messages(self) -> None:
        ids = self.add(["2026-01-01", "2026-01-02"])
        TEMPORAL.detect_episodes(self.conn)
        self.conn.execute("UPDATE messages SET timestamp = CASE id WHEN 1 THEN '' ELSE NULL END")
        self.clear_audit()
        self.assertEqual(TEMPORAL.detect_episodes(self.conn), 0)
        self.assertEqual(self.groups(), [])
        self.assertEqual(self.search("cedar"), ids)
        self.assertEqual(self.audit(), {"episode_delete": 2, "message_update": 2})
        self.clear_audit()
        TEMPORAL.detect_episodes(self.conn)
        self.assertEqual(self.audit(), {})

    def test_list_row_factory_does_not_create_false_metadata_deltas(self) -> None:
        self.add(["2026-01-01"])
        TEMPORAL.detect_episodes(self.conn)
        self.clear_audit()
        self.conn.row_factory = lambda cursor, row: list(row)
        TEMPORAL.detect_episodes(self.conn)
        self.assertEqual(self.audit(), {})


class TestEpisodeTransactions(EpisodeFixture):
    def test_caller_owns_commit_and_can_rollback_all_changes(self) -> None:
        self.add(["2026-01-01"])
        reader = self.open_db()
        before = self.snapshot()
        self.conn.execute("INSERT INTO metadata(key, value) VALUES ('synthetic-caller', 'pending')")
        self.assertEqual(TEMPORAL.detect_episodes(self.conn), 1)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(reader.execute("SELECT count(*) FROM episodes").fetchone(), (0,))
        self.assertIsNone(reader.execute("SELECT value FROM metadata WHERE key = 'synthetic-caller'").fetchone())
        self.conn.rollback()
        self.assertEqual(self.snapshot(), before)
        self.assertIsNone(self.conn.execute("SELECT value FROM metadata WHERE key = 'synthetic-caller'").fetchone())
        TEMPORAL.detect_episodes(self.conn)
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(reader.execute("SELECT count(*) FROM episodes").fetchone(), (1,))

    def test_assignment_failure_rolls_back_partial_writes_and_preserves_caller_work(self) -> None:
        self.add(["2026-01-01T00:00:00", "2026-01-01T01:00:00"])
        before = self.snapshot()
        self.conn.executescript("""
            CREATE TRIGGER synthetic_fail_assignment BEFORE UPDATE OF episode_id ON messages
            WHEN new.id = 2 BEGIN SELECT RAISE(ABORT, 'synthetic assignment failure'); END;
        """)
        for caller_owned in (False, True):
            with self.subTest(caller_owned=caller_owned):
                if caller_owned:
                    self.conn.execute("INSERT INTO metadata(key, value) VALUES ('synthetic-caller', 'pending')")
                with self.assertRaisesRegex(sqlite3.IntegrityError, "synthetic assignment failure"):
                    TEMPORAL.detect_episodes(self.conn)
                self.assertEqual(self.conn.in_transaction, caller_owned)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(self.audit(), {})
                if caller_owned:
                    self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key = 'synthetic-caller'").fetchone(), ("pending",))
                    self.conn.rollback()

    def test_episode_delete_failure_restores_old_assignments_and_ids(self) -> None:
        self.add(["2026-01-01"])
        TEMPORAL.detect_episodes(self.conn)
        self.add(["2026-01-01T01:00:00"])
        before = self.snapshot()
        self.conn.executescript("""
            CREATE TRIGGER synthetic_fail_delete BEFORE DELETE ON episodes
            BEGIN SELECT RAISE(ABORT, 'synthetic deletion failure'); END;
        """)
        with self.assertRaisesRegex(sqlite3.IntegrityError, "synthetic deletion failure"):
            TEMPORAL.detect_episodes(self.conn)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.audit(), {})
        self.assertFalse(self.conn.in_transaction)

    def test_writer_commit_during_grouping_retries_fresh_snapshot(self) -> None:
        self.add(["2026-01-01"])
        writer = self.open_db()
        reader = self.open_db()
        entered = threading.Event()
        release = threading.Event()
        calls = []
        group = TEMPORAL._group_episode_rows

        def paused_group(rows: list[tuple], gap: float) -> list[list[tuple]]:
            calls.append(len(rows))
            if len(calls) == 1:
                entered.set()
                release.wait(2)
            return group(rows, gap)

        with patch.object(TEMPORAL, "_group_episode_rows", paused_group):
            worker, outcomes = self.start_worker(lambda: TEMPORAL.detect_episodes(self.conn))
            self.addCleanup(release.set)
            self.assertTrue(entered.wait(2))
            writer.execute("INSERT INTO messages(content, timestamp) VALUES ('synthetic maple', '2026-01-01T01:00:00')")
            writer.commit()
            self.assertEqual(self.search("maple", reader), [2])
            release.set()
            worker.join(2)
        self.assertEqual(outcomes, [1])
        self.assertEqual(calls, [1, 2])
        self.assertEqual(self.groups()[0][1], (1, 2))
        self.assertFalse(self.conn.in_transaction)

    def test_caller_snapshot_conflict_is_not_restarted_or_committed(self) -> None:
        self.add(["2026-01-01"])
        writer = self.open_db()
        self.conn.execute("BEGIN")
        self.conn.execute("SELECT count(*) FROM messages").fetchone()
        writer.execute("INSERT INTO messages(content, timestamp) VALUES ('synthetic maple', '2026-01-01T01:00:00')")
        writer.commit()
        with self.assertRaises(sqlite3.OperationalError):
            TEMPORAL.detect_episodes(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone(), (1,))
        self.assertEqual(self.conn.execute("SELECT count(*) FROM episodes").fetchone(), (0,))
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone(), (2,))

    def test_snapshot_retries_are_bounded_at_three_attempts(self) -> None:
        self.add(["2026-01-01"])
        writer = self.open_db()
        group = TEMPORAL._group_episode_rows
        calls = []

        def conflicting_group(rows: list[tuple], gap: float) -> list[list[tuple]]:
            calls.append(1)
            writer.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES ('synthetic-writer', ?)", (str(len(calls)),))
            writer.commit()
            return group(rows, gap)

        with patch.object(TEMPORAL, "_group_episode_rows", conflicting_group):
            with self.assertRaises(sqlite3.OperationalError):
                TEMPORAL.detect_episodes(self.conn)
        self.assertEqual(len(calls), 3)
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.groups(), [])
        self.assertEqual(self.audit(), {})

    def test_python310_snapshot_retry_without_extended_error_attribute(self) -> None:
        self.add(["2026-01-01"])
        writer = self.open_db()
        reconcile = TEMPORAL._reconcile_episodes
        calls = []

        def legacy_error(conn: sqlite3.Connection, gap: float) -> int:
            calls.append(1)
            if len(calls) == 1:
                conn.execute("SELECT count(*) FROM messages").fetchone()
                writer.execute("INSERT INTO messages(content, timestamp) VALUES ('synthetic maple', '2026-01-01T01:00:00')")
                writer.commit()
                # Python 3.10 errors lack sqlite_errorcode. The changed
                # connection data_version must justify retrying this error.
                raise sqlite3.OperationalError("database is locked")
            return reconcile(conn, gap)

        with patch.object(TEMPORAL, "_reconcile_episodes", legacy_error):
            self.assertEqual(TEMPORAL.detect_episodes(self.conn), 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.groups()[0][1], (1, 2))

    def test_ordinary_lock_errors_without_changed_snapshot_do_not_retry(self) -> None:
        self.add(["2026-01-01"])
        for error_code in (None, 5):  # SQLITE_BUSY; also works on Python 3.10.
            with self.subTest(error_code=error_code):
                error = sqlite3.OperationalError("database is locked")
                if error_code is not None:
                    error.sqlite_errorcode = error_code
                with patch.object(TEMPORAL, "_reconcile_episodes", side_effect=error) as reconcile:
                    with self.assertRaises(sqlite3.OperationalError):
                        TEMPORAL.detect_episodes(self.conn)
                self.assertEqual(reconcile.call_count, 1)
                self.assertFalse(self.conn.in_transaction)

    def test_wal_search_reader_proceeds_while_episode_writer_is_held(self) -> None:
        ids = self.add(["2026-01-01"])
        reader = self.open_db()
        entered = threading.Event()
        release = threading.Event()
        real_conn = self.conn

        class PausedAfterWrite:
            def __getattr__(self, name: str) -> object:
                return getattr(real_conn, name)

            def execute(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
                cursor = real_conn.execute(sql, parameters)
                if sql.startswith("INSERT INTO episodes"):
                    entered.set()
                    release.wait(2)
                return cursor

        worker, outcomes = self.start_worker(lambda: TEMPORAL.detect_episodes(PausedAfterWrite()))
        self.addCleanup(release.set)
        self.assertTrue(entered.wait(2))
        search_worker, search_outcomes = self.start_worker(lambda: self.search("cedar", reader))
        search_worker.join(1)
        self.assertFalse(search_worker.is_alive(), "WAL reader waited for the episode writer")
        self.assertEqual(search_outcomes, [ids])
        release.set()
        worker.join(2)
        self.assertEqual(outcomes, [1])


if __name__ == "__main__":
    unittest.main()
