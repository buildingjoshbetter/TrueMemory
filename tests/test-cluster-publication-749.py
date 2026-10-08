"""SQLite publication tests with explicit synthetic native-library boundaries."""

import ast
import builtins
import math
import sqlite3
import struct
import tempfile
import threading
import types
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


class SyntheticArray:
    """Only the numeric operations exercised by the production cluster builder."""

    def __init__(self, values: list) -> None:
        self.values = values

    def __iter__(self) -> Iterator[object]:
        return iter(SyntheticArray(value) if isinstance(value, list) else value for value in self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, scalar: float) -> list[bool]:
        return [value == scalar for row in self.values for value in row]

    def __setitem__(self, mask: list[bool], scalar: float) -> None:
        for row, selected in zip(self.values, mask):
            if selected:
                row[0] = scalar

    def __truediv__(self, other: "SyntheticArray") -> "SyntheticArray":
        return SyntheticArray([[f32(value / scale[0]) for value in row]
                               for row, scale in zip(self.values, other.values)])

    def astype(self, dtype: str) -> "SyntheticArray":
        return SyntheticArray([f32(value) for value in self.values])

    def tolist(self) -> list:
        return self.values


class SyntheticNumpy:
    float32 = "float32"
    ndarray = SyntheticArray

    @staticmethod
    def array(values: list, dtype: str | None = None) -> SyntheticArray:
        return SyntheticArray([f32(value) for value in values])

    @staticmethod
    def stack(values: list[SyntheticArray]) -> SyntheticArray:
        return SyntheticArray([value.tolist() for value in values])

    @staticmethod
    def mean(values: list[SyntheticArray], axis: int) -> SyntheticArray:
        return SyntheticArray([sum(column) / len(values) for column in zip(*(value.tolist() for value in values))])

    class linalg:
        @staticmethod
        def norm(values: SyntheticArray, axis: int, keepdims: bool) -> SyntheticArray:
            return SyntheticArray([[f32(math.sqrt(sum(value * value for value in row)))] for row in values.tolist()])


# The parent verification can inject actual NumPy here on the test host. Local
# execution never imports native libraries or the application package.
NUMPY = SyntheticNumpy()


def load_clustering() -> tuple[types.ModuleType, types.ModuleType, types.SimpleNamespace]:
    vector = types.ModuleType("synthetic_vector_search")
    vector._lock = threading.Lock()
    vector.EMBEDDING_MODEL = "synthetic-model-a"
    vector._embedding_dim = 2
    vector._cfg_get_model_group = lambda model: "basepro" if model == "synthetic-model-b" else "edge"
    tree = ast.parse((ROOT / "truememory/vector_search.py").read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"_active_tier_group", "_active_vec_table"}]
    vector.sqlite3 = sqlite3
    exec(compile(ast.Module(body=selected, type_ignores=[]), "synthetic-vector-resolver", "exec"), vector.__dict__)

    boundary = types.SimpleNamespace(labels=[0, 0, -1, 1, 1], fit=None, parameters=[], inputs=[], missing=False)

    class SyntheticHdbscan:
        def __init__(self, **parameters: object) -> None:
            boundary.parameters.append(parameters)

        def fit_predict(self, values: SyntheticArray) -> list[int]:
            boundary.inputs.append(values.tolist())
            if boundary.fit is not None:
                boundary.fit()
            return boundary.labels

    def synthetic_import(
        name: str, globals: dict | None = None, locals: dict | None = None,
        fromlist: tuple[str, ...] = (), level: int = 0,
    ) -> object:
        if name == "numpy":
            return NUMPY
        if name == "hdbscan":
            if boundary.missing:
                raise ModuleNotFoundError("synthetic missing hdbscan", name="hdbscan")
            return types.SimpleNamespace(HDBSCAN=SyntheticHdbscan)
        if name == "truememory":
            return types.SimpleNamespace(vector_search=vector)
        if name == "truememory.vector_search":
            return vector
        return builtins.__import__(name, globals, locals, fromlist, level)

    module = types.ModuleType("synthetic_clustering")
    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=synthetic_import)
    path = ROOT / "truememory/clustering.py"
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module, vector, boundary


SCHEMA = """
CREATE TABLE messages (
 id INTEGER PRIMARY KEY, content TEXT, sender TEXT, recipient TEXT,
 timestamp TEXT, category TEXT, modality TEXT
);
CREATE TABLE vec_messages_edge (embedding BLOB);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
CREATE TABLE vector_cache_registry (
 tier_group TEXT PRIMARY KEY, vec_table TEXT, sep_table TEXT,
 last_embedded_id INTEGER, vector_count INTEGER, model_name TEXT,
 embedding_dim INTEGER, last_updated REAL, created REAL
);
"""


class ClusterFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.vector, self.boundary = load_clustering()
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-clusters-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.sqlite"
        self.conn = self.open_db()
        self.conn.executescript(SCHEMA + self.module._CLUSTER_SCHEMA)
        self.seed()

    def open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        self.addCleanup(conn.close)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 1000")
        return conn

    def seed(self) -> None:
        for table in ("message_clusters", "cluster_centroids", "vec_messages_edge", "messages", "metadata", "vector_cache_registry"):
            self.conn.execute(f"DELETE FROM {table}")
        vectors = [(1, 0), (0, 1), (0, 0), (2, 2), (4, 0)]
        for mid, (vector, category) in enumerate(zip(vectors, ["session-z", "session-a", "noise", None, "session-z"]), 1):
            self.conn.execute(
                "INSERT INTO messages VALUES (?, 'synthetic source', 'synthetic-sender', 'synthetic-recipient', '2026-01-01', ?, 'text')",
                (mid, category),
            )
            self.conn.execute("INSERT INTO vec_messages_edge(rowid, embedding) VALUES (?, ?)", (mid, struct.pack("2f", *vector)))
        self.conn.execute("INSERT INTO metadata VALUES ('embed_model', 'synthetic-model-a', 'synthetic-stamp')")
        self.conn.execute("INSERT INTO metadata VALUES ('embed_dim', '2', 'synthetic-stamp')")
        self.conn.execute("INSERT INTO vector_cache_registry VALUES ('edge', 'vec_messages_edge', 'vec_messages_sep_edge', 5, 5, 'synthetic-model-a', 2, 1.0, 1.0)")
        self.conn.execute("INSERT INTO message_clusters VALUES (1, 7, 0)")
        self.conn.execute("INSERT INTO cluster_centroids VALUES (7, ?, 1, 'previous-session', 'synthetic previous summary')", (struct.pack("2f", 9, 9),))
        self.conn.commit()

    def output(self, conn: sqlite3.Connection | None = None) -> tuple[list[tuple], list[tuple]]:
        db = conn or self.conn
        return (
            db.execute("SELECT * FROM message_clusters ORDER BY message_id").fetchall(),
            db.execute("SELECT * FROM cluster_centroids ORDER BY cluster_id").fetchall(),
        )

    def start_worker(self, action: Callable[[], object]) -> tuple[threading.Thread, list[object]]:
        outcomes = []

        def run() -> None:
            try:
                outcomes.append(action())
            except Exception as error:  # Forward thread failures to the test.
                outcomes.append(error)

        worker = threading.Thread(target=run)
        worker.start()
        self.addCleanup(worker.join, 3)
        return worker, outcomes


class TestClusterComputation(ClusterFixture):
    def test_reference_parameters_normalization_assignments_and_centroids(self) -> None:
        self.assertEqual(self.module.cluster_messages(self.conn, min_cluster_size=3, min_samples=2), 2)
        self.assertEqual(self.boundary.parameters, [{
            "min_cluster_size": 3, "min_samples": 2, "metric": "euclidean", "cluster_selection_method": "eom",
        }])
        assignments, centroids = self.output()
        self.assertEqual(assignments, [(1, 0, 0), (2, 0, 0), (3, -1, 1), (4, 1, 0), (5, 1, 0)])
        self.assertEqual(centroids, [
            (0, struct.pack("2f", 0.5, 0.5), 2, "session-a, session-z", ""),
            (1, struct.pack("2f", 3, 1), 2, "session-z", ""),
        ])
        self.assertEqual(self.boundary.inputs[0][:3], [[1, 0], [0, 1], [0, 0]])
        for actual in self.boundary.inputs[0][3]:
            self.assertAlmostEqual(actual, 1 / math.sqrt(2), places=6)
        self.assertFalse(self.conn.in_transaction)

    def test_compute_failure_leaves_previous_cache_after_caller_commit(self) -> None:
        before = self.output()

        def fail() -> None:
            self.assertFalse(self.conn.in_transaction)
            raise RuntimeError("synthetic compute failure")

        self.boundary.fit = fail
        with self.assertRaisesRegex(RuntimeError, "synthetic compute failure"):
            self.module.cluster_messages(self.conn)
        self.conn.commit()
        self.assertEqual(self.output(), before)
        self.assertFalse(self.conn.in_transaction)

    def test_embedding_read_failure_preserves_previous_cache(self) -> None:
        before = self.output()
        with patch.object(self.module, "_get_all_embeddings", side_effect=ValueError("synthetic embedding failure")):
            with self.assertRaisesRegex(ValueError, "synthetic embedding failure"):
                self.module.cluster_messages(self.conn)
        self.conn.commit()
        self.assertEqual(self.output(), before)
        self.assertFalse(self.conn.in_transaction)

    def test_centroid_serialization_failure_never_opens_publication(self) -> None:
        before = self.output()
        serialize = self.module._serialize_f32
        calls = []

        def fail_second(vector: SyntheticArray) -> bytes:
            self.assertFalse(self.conn.in_transaction)
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("synthetic centroid failure")
            return serialize(vector)

        with patch.object(self.module, "_serialize_f32", fail_second):
            with self.assertRaisesRegex(ValueError, "synthetic centroid failure"):
                self.module.cluster_messages(self.conn)
        self.conn.commit()
        self.assertEqual(self.output(), before)

    def test_incomplete_hdbscan_labels_cannot_publish_partial_cache(self) -> None:
        before = self.output()
        self.boundary.labels = [0]
        with self.assertRaisesRegex(ValueError, "incomplete assignment"):
            self.module.cluster_messages(self.conn)
        self.assertEqual(self.output(), before)

    def test_missing_dependency_remains_an_error_even_with_empty_vectors(self) -> None:
        self.conn.execute("DELETE FROM vec_messages_edge")
        self.conn.commit()
        before = self.output()
        self.boundary.missing = True
        with self.assertRaises(ModuleNotFoundError):
            self.module.cluster_messages(self.conn)
        self.assertEqual(self.output(), before)
        self.assertFalse(self.conn.in_transaction)

    def test_empty_completed_vector_input_atomically_clears_old_cache(self) -> None:
        self.conn.execute("DELETE FROM vec_messages_edge")
        self.conn.commit()
        self.assertEqual(self.module.cluster_messages(self.conn), 0)
        self.assertEqual(self.output(), ([], []))
        self.assertEqual(self.boundary.parameters, [])

    def test_empty_snapshot_cannot_clear_cache_after_concurrent_vector_addition(self) -> None:
        self.conn.execute("DELETE FROM vec_messages_edge")
        self.conn.commit()
        writer = self.open_db()
        before = self.output()
        extract = self.module._get_all_embeddings

        def insert_after_read(conn: sqlite3.Connection) -> tuple[list[int], SyntheticArray]:
            result = extract(conn)
            writer.execute("INSERT INTO vec_messages_edge(rowid, embedding) VALUES (1, ?)", (struct.pack("2f", 1, 0),))
            writer.commit()
            return result

        with patch.object(self.module, "_get_all_embeddings", insert_after_read):
            with self.assertRaisesRegex(RuntimeError, "changed during computation"):
                self.module.cluster_messages(self.conn)
        self.assertEqual(self.output(), before)

    def test_missing_vector_table_is_not_successful_empty_input(self) -> None:
        self.conn.execute("DROP TABLE vec_messages_edge")
        self.conn.commit()
        before = self.output()
        with self.assertRaises(sqlite3.OperationalError):
            self.module.cluster_messages(self.conn)
        self.assertEqual(self.output(), before)

    def test_in_progress_vector_index_does_not_clear_old_cache(self) -> None:
        self.conn.execute("DELETE FROM vec_messages_edge")
        self.conn.execute("INSERT INTO metadata VALUES ('vec_build_state:vec_messages_edge', 'in_progress', 'synthetic-stamp')")
        self.conn.commit()
        before = self.output()
        with self.assertRaisesRegex(RuntimeError, "rebuild is in progress"):
            self.module.cluster_messages(self.conn)
        self.assertEqual(self.output(), before)

    def test_orphan_vector_is_rejected_without_resurrecting_assignment(self) -> None:
        self.conn.execute("DELETE FROM messages WHERE id = 2")
        self.conn.commit()
        before = self.output()
        with self.assertRaisesRegex(RuntimeError, "without source messages"):
            self.module.cluster_messages(self.conn)
        self.assertEqual(self.output(), before)

    def test_legacy_vector_table_without_identity_metadata_remains_supported(self) -> None:
        self.conn.execute("ALTER TABLE vec_messages_edge RENAME TO vec_messages")
        self.conn.execute("DROP TABLE vector_cache_registry")
        self.conn.execute("DROP TABLE metadata")
        self.conn.commit()
        self.assertEqual(self.module.cluster_messages(self.conn), 2)


class TestClusterSourceFence(ClusterFixture):
    def test_source_and_vector_space_changes_reject_publication(self) -> None:
        writer = self.open_db()
        changes = {
            "new_source_without_vector": "INSERT INTO messages(id, content, category) VALUES (6, 'synthetic addition', 'new-session')",
            "source_edit": "UPDATE messages SET content = 'synthetic replacement' WHERE id = 2",
            "source_delete": "DELETE FROM messages WHERE id = 2",
            "category_edit": "UPDATE messages SET category = 'changed-session' WHERE id = 2",
            "timestamp_edit": "UPDATE messages SET timestamp = '2026-02-01' WHERE id = 2",
            "vector_edit": "UPDATE vec_messages_edge SET embedding = X'000080400000803F' WHERE rowid = 2",
            "raw_signed_zero": "UPDATE vec_messages_edge SET embedding = X'000000800000803F' WHERE rowid = 2",
            "vector_delete": "DELETE FROM vec_messages_edge WHERE rowid = 2",
            "vector_insert": "INSERT INTO vec_messages_edge(rowid, embedding) VALUES (6, X'0000803F00000000')",
            "stored_model": "UPDATE metadata SET value = 'synthetic-model-new' WHERE key = 'embed_model'",
            "stored_dimension": "UPDATE metadata SET value = '4' WHERE key = 'embed_dim'",
            "stored_identity_stamp": "UPDATE metadata SET updated_at = 'new-stamp' WHERE key = 'embed_model'",
            "cache_model": "UPDATE vector_cache_registry SET model_name = 'synthetic-model-new'",
            "cache_dimension": "UPDATE vector_cache_registry SET embedding_dim = 4",
            "cache_identity_stamp": "UPDATE vector_cache_registry SET created = 2.0",
            "rebuild_in_progress": "INSERT INTO metadata VALUES ('vec_build_state:vec_messages_edge', 'in_progress', 'synthetic-stamp')",
        }
        for name, statement in changes.items():
            with self.subTest(change=name):
                self.seed()
                before = self.output()

                def change_source() -> None:
                    writer.execute(statement)
                    writer.commit()

                self.boundary.fit = change_source
                with self.assertRaisesRegex(RuntimeError, "changed during computation|rebuild is in progress"):
                    self.module.cluster_messages(self.conn)
                self.conn.commit()
                self.assertEqual(self.output(), before)

    def test_active_table_change_rejects_identical_vector_bytes(self) -> None:
        writer = self.open_db()
        writer.execute("CREATE TABLE vec_alternative (embedding BLOB)")
        writer.execute("INSERT INTO vec_alternative(rowid, embedding) SELECT rowid, embedding FROM vec_messages_edge")
        writer.commit()
        before = self.output()

        def switch_table() -> None:
            writer.execute("UPDATE vector_cache_registry SET vec_table = 'vec_alternative'")
            writer.commit()

        self.boundary.fit = switch_table
        with self.assertRaisesRegex(RuntimeError, "changed during computation"):
            self.module.cluster_messages(self.conn)
        self.assertEqual(self.output(), before)

    def test_runtime_model_and_dimension_changes_reject_publication(self) -> None:
        for field, new_value in (("EMBEDDING_MODEL", "synthetic-model-new"), ("_embedding_dim", 4)):
            with self.subTest(field=field):
                before = self.output()

                def change_model() -> None:
                    with self.vector._lock:
                        setattr(self.vector, field, new_value)

                self.boundary.fit = change_model
                with self.assertRaisesRegex(RuntimeError, "changed during computation"):
                    self.module.cluster_messages(self.conn)
                self.assertEqual(self.output(), before)

    def test_table_definition_change_rejects_identical_vector_bytes(self) -> None:
        writer = self.open_db()
        before = self.output()

        def change_schema() -> None:
            writer.execute("ALTER TABLE vec_messages_edge ADD COLUMN synthetic_marker TEXT")
            writer.commit()

        self.boundary.fit = change_schema
        with self.assertRaisesRegex(RuntimeError, "changed during computation"):
            self.module.cluster_messages(self.conn)
        self.assertEqual(self.output(), before)

    def test_unrelated_metadata_write_does_not_reject_current_source(self) -> None:
        writer = self.open_db()

        def unrelated_write() -> None:
            writer.execute("INSERT INTO metadata VALUES ('synthetic-unrelated', 'new', 'synthetic-stamp')")
            writer.commit()

        self.boundary.fit = unrelated_write
        self.assertEqual(self.module.cluster_messages(self.conn), 2)

    def test_fingerprint_and_embedding_reads_share_one_snapshot(self) -> None:
        writer = self.open_db()
        before = self.output()
        extract = self.module._get_all_embeddings
        captured = []

        def concurrent_read(conn: sqlite3.Connection) -> tuple[list[int], SyntheticArray]:
            writer.execute("UPDATE vec_messages_edge SET embedding = ? WHERE rowid = 1", (struct.pack("2f", 9, 9),))
            writer.execute("UPDATE messages SET category = 'changed-session' WHERE id = 1")
            writer.commit()
            result = extract(conn)
            captured.append(result[1].tolist())
            return result

        with patch.object(self.module, "_get_all_embeddings", concurrent_read):
            with self.assertRaisesRegex(RuntimeError, "changed during computation"):
                self.module.cluster_messages(self.conn)
        self.assertEqual(captured[0][0], [1, 0])
        self.assertEqual(self.output(), before)

    def test_second_writer_can_commit_while_compute_is_paused(self) -> None:
        writer = self.open_db()
        entered = threading.Event()
        release = threading.Event()

        def pause() -> None:
            entered.set()
            release.wait(2)

        self.boundary.fit = pause
        worker, outcomes = self.start_worker(lambda: self.module.cluster_messages(self.conn))
        self.addCleanup(release.set)
        self.assertTrue(entered.wait(2))
        self.assertFalse(self.conn.in_transaction)
        writer.execute("INSERT INTO metadata VALUES ('synthetic-unrelated', 'new', 'synthetic-stamp')")
        writer.commit()
        release.set()
        worker.join(2)
        self.assertEqual(outcomes, [2])


class TestClusterPublication(ClusterFixture):
    def test_mid_centroid_failure_rolls_back_both_tables_even_if_caller_commits(self) -> None:
        before = self.output()
        self.conn.executescript("""
            CREATE TRIGGER synthetic_centroid_failure BEFORE INSERT ON cluster_centroids
            WHEN new.cluster_id = 1 BEGIN SELECT RAISE(ABORT, 'synthetic centroid insert failure'); END;
        """)
        with self.assertRaisesRegex(sqlite3.IntegrityError, "synthetic centroid insert failure"):
            self.module.cluster_messages(self.conn)
        self.conn.commit()
        self.assertEqual(self.output(), before)
        self.assertFalse(self.conn.in_transaction)
        self.assertTrue(self.vector._lock.acquire(blocking=False))
        self.vector._lock.release()

    def test_caller_transaction_retains_unrelated_work_and_publication_until_rollback(self) -> None:
        reader = self.open_db()
        before = self.output()
        self.conn.execute("INSERT INTO metadata VALUES ('synthetic-caller', 'pending', 'synthetic-stamp')")
        self.assertEqual(self.module.cluster_messages(self.conn), 2)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.output(reader), before)
        self.assertIsNone(reader.execute("SELECT value FROM metadata WHERE key = 'synthetic-caller'").fetchone())
        self.conn.rollback()
        self.assertEqual(self.output(), before)
        self.assertIsNone(self.conn.execute("SELECT value FROM metadata WHERE key = 'synthetic-caller'").fetchone())

    def test_publication_failure_rolls_back_savepoint_but_preserves_caller_write(self) -> None:
        before = self.output()
        self.conn.executescript("""
            CREATE TRIGGER synthetic_assignment_failure BEFORE INSERT ON message_clusters
            WHEN new.message_id = 2 BEGIN SELECT RAISE(ABORT, 'synthetic assignment failure'); END;
        """)
        self.conn.execute("INSERT INTO metadata VALUES ('synthetic-caller', 'pending', 'synthetic-stamp')")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "synthetic assignment failure"):
            self.module.cluster_messages(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.output(), before)
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key = 'synthetic-caller'").fetchone(), ("pending",))
        self.conn.rollback()

    def test_caller_read_snapshot_conflict_is_not_retried_or_committed(self) -> None:
        writer = self.open_db()
        before = self.output()
        self.conn.execute("BEGIN")
        self.conn.execute("SELECT count(*) FROM messages").fetchone()

        def write() -> None:
            writer.execute("UPDATE messages SET category = 'changed-session' WHERE id = 2")
            writer.commit()

        self.boundary.fit = write
        with self.assertRaises(sqlite3.OperationalError):
            self.module.cluster_messages(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.output(), before)
        self.assertEqual(len(self.boundary.inputs), 1)
        self.conn.rollback()

    def test_schema_creation_does_not_commit_caller_transaction(self) -> None:
        self.conn.execute("DROP TABLE message_clusters")
        self.conn.execute("DROP TABLE cluster_centroids")
        self.conn.commit()
        self.conn.execute("INSERT INTO metadata VALUES ('synthetic-caller', 'pending', 'synthetic-stamp')")
        self.module.cluster_messages(self.conn)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertIsNone(self.conn.execute("SELECT name FROM sqlite_master WHERE name = 'message_clusters'").fetchone())
        self.assertIsNone(self.conn.execute("SELECT value FROM metadata WHERE key = 'synthetic-caller'").fetchone())

    def test_busy_model_lock_fails_without_waiting_and_releases_writer(self) -> None:
        writer = self.open_db()
        before = self.output()
        self.boundary.fit = lambda: self.vector._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "Embedding model is busy"):
                self.module.cluster_messages(self.conn)
            self.assertFalse(self.conn.in_transaction)
            self.assertEqual(self.output(), before)
            writer.execute("INSERT INTO metadata VALUES ('synthetic-unrelated', 'new', 'synthetic-stamp')")
            writer.commit()
        finally:
            self.vector._lock.release()

    def test_model_lock_is_held_through_publication_commit(self) -> None:
        commits = []
        real_conn = self.conn
        vector = self.vector

        class CheckCommit:
            def __getattr__(self, name: str) -> object:
                return getattr(real_conn, name)

            def execute(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
                if sql == "COMMIT":
                    commits.append(vector._lock.locked())
                return real_conn.execute(sql, parameters)

        self.module.cluster_messages(CheckCommit())
        self.assertEqual(commits, [True, True])
        self.assertFalse(self.vector._lock.locked())

    def test_commit_and_release_failure_roll_back_before_later_caller_commit(self) -> None:
        before = self.output()
        for caller_owned in (False, True):
            with self.subTest(caller_owned=caller_owned):
                denied = []
                actions = []
                if caller_owned:
                    self.conn.execute("INSERT INTO metadata VALUES ('synthetic-caller', 'pending', 'synthetic-stamp')")

                def deny_terminal_once(action: int, operation: str, *unused: str | None) -> int:
                    if action in (sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT):
                        actions.append(operation)
                    target = "RELEASE" if caller_owned else "COMMIT"
                    if operation == target and action in (sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT) and not denied:
                        denied.append(target)
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK

                # Install after read transaction completion. This also expires
                # SQLite's cached prepared statements and authorization checks.
                self.boundary.fit = lambda: self.conn.set_authorizer(deny_terminal_once)
                try:
                    with self.assertRaises(sqlite3.DatabaseError):
                        self.module.cluster_messages(self.conn)
                finally:
                    # Disabling with None is supported only on Python 3.11+.
                    self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
                self.assertEqual(denied, ["RELEASE" if caller_owned else "COMMIT"])
                self.assertIn("ROLLBACK", actions)
                if caller_owned:
                    self.assertEqual(actions[-2:], ["ROLLBACK", "RELEASE"])
                self.assertEqual(self.conn.in_transaction, caller_owned)
                self.assertFalse(self.vector._lock.locked())
                self.conn.commit()
                self.assertEqual(self.output(), before)
                if caller_owned:
                    self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key = 'synthetic-caller'").fetchone(), ("pending",))

    def test_cancellation_during_publication_rolls_back_and_releases_model_lock(self) -> None:
        before = self.output()
        real_conn = self.conn

        class CancelAfterAssignment:
            def __getattr__(self, name: str) -> object:
                return getattr(real_conn, name)

            def executemany(self, sql: str, parameters: list[tuple]) -> sqlite3.Cursor:
                cursor = real_conn.executemany(sql, parameters)
                if sql.startswith("INSERT INTO message_clusters"):
                    raise KeyboardInterrupt("synthetic cancellation")
                return cursor

        with self.assertRaises(KeyboardInterrupt):
            self.module.cluster_messages(CancelAfterAssignment())
        self.conn.commit()
        self.assertEqual(self.output(), before)
        self.assertFalse(self.conn.in_transaction)
        self.assertFalse(self.vector._lock.locked())

    def test_reader_sees_old_assignments_and_centroids_until_atomic_commit(self) -> None:
        reader = self.open_db()
        before = self.output()
        entered = threading.Event()
        release = threading.Event()

        def pause() -> int:
            entered.set()
            release.wait(2)
            return 0

        self.conn.create_function("synthetic_pause", 0, pause)
        self.conn.executescript("""
            CREATE TRIGGER synthetic_pause_insert BEFORE INSERT ON cluster_centroids
            WHEN new.cluster_id = 1 BEGIN SELECT synthetic_pause(); END;
        """)
        worker, outcomes = self.start_worker(lambda: self.module.cluster_messages(self.conn))
        self.addCleanup(release.set)
        self.assertTrue(entered.wait(2))
        reader.execute("BEGIN")
        self.assertEqual(self.output(reader), before)
        release.set()
        worker.join(2)
        self.assertEqual(outcomes, [2])
        self.assertEqual(self.output(reader), before)
        reader.commit()
        self.assertEqual(self.output(reader), self.output())
        self.assertNotEqual(self.output(reader), before)


if __name__ == "__main__":
    unittest.main()
