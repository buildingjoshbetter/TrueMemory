"""Exact clustered retrieval with synthetic SQLite data and fake native boundaries."""
from __future__ import annotations

import ast
import builtins
import heapq
import math
import random
import sqlite3
import struct
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class SyntheticArray(list):
    def astype(self, dtype: object) -> "SyntheticArray":
        return SyntheticArray(struct.unpack("f", struct.pack("f", value))[0] for value in self)


class SyntheticNumpy:
    float32 = "float32"
    ndarray = SyntheticArray

    @staticmethod
    def array(values: object, dtype: object = None) -> SyntheticArray:
        return SyntheticArray(values).astype(dtype)

    @staticmethod
    def dot(left: list, right: list) -> float:
        if len(left) != len(right):
            raise ValueError("synthetic shape mismatch")
        return sum(a * b for a, b in zip(left, right))

    class linalg:
        @staticmethod
        def norm(values: list) -> float:
            return math.sqrt(sum(value * value for value in values))


# Remote verification may inject real NumPy. The default imports no framework
# or application package and never constructs a real embedding model.
NUMPY = SyntheticNumpy()

# Frozen implementation at 9b2b6ff, including its actual SQL and stable sort.
REFERENCE_SOURCE = r'''
def reference_search_clustered(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    top_clusters: int = 3,
    include_directives: bool = False,
) -> list[dict]:
    """
    Two-stage clustered search: find top clusters, then search within them.

    1. Embed the query.
    2. Find the *top_clusters* most similar cluster centroids.
    3. Retrieve messages from those clusters.
    4. Rank by vector similarity within the selected clusters.

    This can surface results that flat search misses by focusing on
    contextually coherent message groups.

    Args:
        conn:         Open database connection.
        query:        The search query.
        limit:        Maximum results to return.
        top_clusters: Number of top clusters to search within.

    Returns:
        List of result dicts sorted by similarity.
    """
    from truememory.vector_search import get_model
    from truememory.mps_utils import encode_with_model_ownership

    model = get_model()
    query_vec = encode_with_model_ownership(model, [query])[0].astype(np.float32)

    # Check if clusters exist
    try:
        cluster_count = conn.execute(
            "SELECT COUNT(*) FROM cluster_centroids"
        ).fetchone()[0]
    except Exception:
        return []

    if cluster_count == 0:
        return []

    # Find top clusters by centroid similarity
    centroids = conn.execute(
        "SELECT cluster_id, centroid, message_count FROM cluster_centroids"
    ).fetchall()

    cluster_scores = []
    for cid, centroid_blob, msg_count in centroids:
        dim = len(centroid_blob) // 4
        centroid = np.array(struct.unpack(f"{dim}f", centroid_blob), dtype=np.float32)
        # Cosine similarity
        sim = np.dot(query_vec, centroid) / (
            np.linalg.norm(query_vec) * np.linalg.norm(centroid) + 1e-9
        )
        cluster_scores.append((cid, float(sim), msg_count))

    cluster_scores.sort(key=lambda x: x[1], reverse=True)
    selected_clusters = [c[0] for c in cluster_scores[:top_clusters]]

    if not selected_clusters:
        return []

    # Get message IDs from selected clusters
    placeholders = ",".join("?" * len(selected_clusters))
    cluster_msg_rows = conn.execute(
        f"SELECT message_id FROM message_clusters WHERE cluster_id IN ({placeholders})",
        selected_clusters,
    ).fetchall()

    cluster_msg_ids = {r[0] for r in cluster_msg_rows}
    if not cluster_msg_ids:
        return []

    # Get full messages with their embeddings
    id_placeholders = ",".join("?" * len(cluster_msg_ids))
    msg_ids_list = list(cluster_msg_ids)

    directive_filter = (
        "" if include_directives
        else " AND (m.directive = 0 OR m.directive IS NULL)"
    )
    messages = conn.execute(
        f"""SELECT m.id, m.content, m.sender, m.recipient, m.timestamp,
                   m.category, m.modality, m.directive
            FROM messages m
            WHERE m.id IN ({id_placeholders}){directive_filter}""",
        msg_ids_list,
    ).fetchall()

    if not messages:
        return []

    # Resolve vec table once outside the loop (not per-message)
    from truememory.vector_search import _active_vec_table
    _vec_tbl = _active_vec_table(conn)

    # Batch-fetch all embeddings in one query (issue #584) instead of
    # one SELECT per message.  Chunk into groups of 500 to stay within
    # SQLite's parameter limits.
    _CHUNK = 500
    emb_map: dict[int, np.ndarray] = {}
    all_msg_ids = [msg[0] for msg in messages]
    for chunk_start in range(0, len(all_msg_ids), _CHUNK):
        chunk = all_msg_ids[chunk_start : chunk_start + _CHUNK]
        ph = ",".join("?" * len(chunk))
        try:
            rows = conn.execute(
                f"SELECT rowid, embedding FROM {_vec_tbl} WHERE rowid IN ({ph})",
                chunk,
            ).fetchall()
            for rid, blob in rows:
                dim = len(blob) // 4
                emb_map[rid] = np.array(
                    struct.unpack(f"{dim}f", blob), dtype=np.float32
                )
        except Exception:
            logger.debug("Batch embedding fetch failed for chunk", exc_info=True)

    # Pre-compute query norm once
    _query_norm = np.linalg.norm(query_vec) + 1e-9

    # Score each message by vector similarity to query
    results = []
    for msg in messages:
        msg_id = msg[0]
        msg_vec = emb_map.get(msg_id)
        if msg_vec is not None:
            sim = float(np.dot(query_vec, msg_vec) / (
                _query_norm * (np.linalg.norm(msg_vec) + 1e-9)
            ))
        else:
            sim = 0.0

        results.append({
            "id": msg_id,
            "content": msg[1],
            "sender": msg[2],
            "recipient": msg[3],
            "timestamp": msg[4],
            "category": msg[5],
            "modality": msg[6],
            "directive": bool(msg[7]),
            "score": sim,
            "source": "clustered",
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]
'''


def load_clustering() -> tuple[types.ModuleType, types.SimpleNamespace]:
    boundary = types.SimpleNamespace(query=[1.0, 0.0, 0.0, 0.0], before_encode=None, encodes=0)
    vector = types.ModuleType("synthetic_vector_search")
    vector._active_tier_group = lambda: "basepro"
    vector.sqlite3 = sqlite3
    source = (ROOT / "truememory/vector_search.py").read_text()
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
             and node.name == "_active_vec_table"]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "synthetic-vector-table", "exec"), vector.__dict__)
    model = object()
    vector.get_model = lambda: model

    def encode(actual_model: object, texts: list[str]) -> list:
        assert actual_model is model and texts == ["synthetic query"]
        boundary.encodes += 1
        if boundary.before_encode:
            boundary.before_encode()
        return [NUMPY.array(boundary.query, dtype=NUMPY.float32)]

    def synthetic_import(name: str, globals: dict | None = None, locals: dict | None = None,
                         fromlist: tuple = (), level: int = 0) -> object:
        if name == "numpy":
            return NUMPY
        if name == "truememory.vector_search":
            return vector
        if name == "truememory.mps_utils":
            return types.SimpleNamespace(encode_with_model_ownership=encode)
        if name == "truememory" or name.startswith("truememory."):
            raise AssertionError("unexpected application import")
        return builtins.__import__(name, globals, locals, fromlist, level)

    module = types.ModuleType("synthetic_clustering")
    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=synthetic_import)
    path = ROOT / "truememory/clustering.py"
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    exec(compile(REFERENCE_SOURCE, "frozen-clustered-reference-9b2b6ff", "exec"), module.__dict__)
    return module, boundary


SCHEMA = """
CREATE TABLE messages (
 id INTEGER PRIMARY KEY, content TEXT, sender TEXT, recipient TEXT,
 timestamp TEXT, category TEXT, modality TEXT, directive INTEGER
);
CREATE TABLE cluster_centroids (cluster_id INTEGER PRIMARY KEY, centroid BLOB, message_count INTEGER);
CREATE TABLE message_clusters (message_id INTEGER PRIMARY KEY, cluster_id INTEGER);
CREATE INDEX idx_cluster_id ON message_clusters(cluster_id);
CREATE TABLE vec_messages (embedding BLOB);
CREATE TABLE vector_cache_registry (tier_group TEXT PRIMARY KEY, vec_table TEXT);
INSERT INTO vector_cache_registry VALUES ('basepro', 'vec_messages');
"""


def blob(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


class ObservedCursor:
    def __init__(self, cursor: sqlite3.Cursor, owner: "ObservedConnection", kind: str) -> None:
        self.cursor, self.owner, self.kind = cursor, owner, kind

    def _record(self, row: tuple | None) -> tuple | None:
        if row is not None:
            self.owner.rows[self.kind] = self.owner.rows.get(self.kind, 0) + 1
        return row

    def __iter__(self) -> "ObservedCursor":
        return self

    def __next__(self) -> tuple:
        return self._record(next(self.cursor))

    def fetchone(self) -> tuple | None:
        return self._record(self.cursor.fetchone())

    def fetchall(self) -> list[tuple]:
        rows = self.cursor.fetchall()
        self.owner.max_fetchall[self.kind] = max(self.owner.max_fetchall.get(self.kind, 0), len(rows))
        return [self._record(row) for row in rows]

    def close(self) -> None:
        self.cursor.close()
        self.owner.closed += 1


class ObservedConnection:
    def __init__(self, conn: sqlite3.Connection, maximum_parameters: int = 500) -> None:
        self.conn = conn
        self.maximum_parameters = maximum_parameters
        self.parameter_counts = []
        self.rows = {}
        self.max_fetchall = {}
        self.before_vectors = None
        self.before_content = None
        self.closed = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.conn, name)

    def execute(self, sql: str, parameters: tuple | list = ()) -> ObservedCursor:
        self.parameter_counts.append(len(parameters))
        if len(parameters) > self.maximum_parameters:
            raise sqlite3.OperationalError("synthetic SQL parameter limit")
        normalized = " ".join(sql.lower().split())
        kind = "other"
        if normalized.startswith("select") and "content" in normalized:
            kind = "content"
            if self.before_content:
                self.before_content()
        elif normalized.startswith("select rowid, embedding"):
            kind = "vectors"
            if self.before_vectors:
                callback, self.before_vectors = self.before_vectors, None
                callback()
        elif normalized.startswith("select distinct m.id"):
            kind = "candidate_ids"
        return ObservedCursor(self.conn.execute(sql, parameters), self, kind)


class ClusteredSearchFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.boundary = load_clustering()
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-cluster-search-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.sqlite"
        self.conn = self.open_db()
        self.conn.executescript(SCHEMA)

    def open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=1000")
        self.addCleanup(conn.close)
        return conn

    def seed(self, count: int = 40, clusters: int = 5, *, random_seed: int = 0,
             all_ties: bool = False, directives: bool = True, content_size: int = 32) -> None:
        rng = random.Random(random_seed)
        for table in ("messages", "message_clusters", "cluster_centroids", "vec_messages"):
            self.conn.execute(f"DELETE FROM {table}")
        centroids = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0] * 4]
        for cid in range(clusters):
            self.conn.execute("INSERT INTO cluster_centroids VALUES (?,?,?)", (cid, blob(centroids[cid % 5]), count))
        ids = [index * 7 + 3 for index in range(count)]
        rng.shuffle(ids)
        for mid in ids:
            vector = [1.0, 0.0, 0.0, 0.0] if all_ties else [float(rng.randrange(-3, 4)) for _ in range(4)]
            directive = (None, 0, 0, 0, 1)[mid % 5] if directives else 0
            self.conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)", (
                mid, "synthetic-" + "x" * content_size, "synthetic-sender", "synthetic-recipient",
                "2026-01-01", None if mid % 3 else "synthetic-category", "text", directive,
            ))
            self.conn.execute("INSERT INTO message_clusters VALUES (?,?)", (mid, mid % clusters))
            if all_ties or mid % 11:
                self.conn.execute("INSERT INTO vec_messages(rowid,embedding) VALUES (?,?)", (mid, blob(vector)))
        self.conn.commit()

    def compare(self, **kwargs: object) -> list[dict]:
        expected = self.module.reference_search_clustered(self.conn, "synthetic query", **kwargs)
        result = self.module.search_clustered(self.conn, "synthetic query", **kwargs)
        self.assertEqual(result, expected)
        return result


class TestExactClusteredSearch(ClusteredSearchFixture):
    def test_randomized_results_match_frozen_sql_reference(self) -> None:
        for seed in range(15):
            self.seed(61, random_seed=seed)
            self.boundary.query = [float(random.Random(seed + axis).randrange(-3, 4)) for axis in range(4)]
            for top_clusters in (1, 3, 0, -1, 8):
                for limit in (0, 1, 17, 100, -7, -100):
                    with self.subTest(seed=seed, top_clusters=top_clusters, limit=limit):
                        self.compare(limit=limit, top_clusters=top_clusters, include_directives=bool(seed % 2))

    def test_equal_scores_keep_original_primary_key_order(self) -> None:
        self.seed(63, all_ties=True, directives=False)
        for table, key in (("messages", "id"), ("message_clusters", "message_id"), ("vec_messages", "rowid")):
            self.conn.execute(f"UPDATE {table} SET {key}=-{key} WHERE {key}<40")
        self.conn.commit()
        results = self.compare(limit=31)
        self.assertEqual([row["id"] for row in results], sorted(row["id"] for row in results))
        self.assertEqual(len({row["score"] for row in results}), 1)

    def test_missing_vectors_zero_vectors_and_negative_scores(self) -> None:
        self.seed(4, clusters=1, all_ties=True, directives=False)
        self.conn.execute("DELETE FROM vec_messages WHERE rowid=3")
        self.conn.execute("UPDATE vec_messages SET embedding=? WHERE rowid=10", (blob([0.0] * 4),))
        self.conn.execute("UPDATE vec_messages SET embedding=? WHERE rowid=17", (blob([-1.0, 0.0, 0.0, 0.0]),))
        self.conn.commit()
        rows = self.compare(limit=100)
        self.assertEqual([row["id"] for row in rows], [24, 3, 10, 17])
        self.assertEqual([row["score"] for row in rows[1:3]], [0.0, 0.0])
        self.assertLess(rows[-1]["score"], 0)

    def test_directive_filter_and_opt_in_match(self) -> None:
        self.seed(50, clusters=1, all_ties=True)
        self.assertTrue(all(not row["directive"] for row in self.compare(limit=100)))
        self.assertTrue(any(row["directive"] for row in self.compare(limit=100, include_directives=True)))

    def test_missing_vector_table_keeps_zero_score_fallback(self) -> None:
        self.seed(20, clusters=1, directives=False)
        self.conn.execute("UPDATE vector_cache_registry SET vec_table='missing_synthetic_vectors'")
        self.conn.commit()
        rows = self.compare(limit=7)
        self.assertEqual([row["score"] for row in rows], [0.0] * 7)

    def test_partial_corrupt_vector_chunk_keeps_previous_fallback_behavior(self) -> None:
        self.seed(25, clusters=1, directives=False)
        self.conn.execute("UPDATE vec_messages SET embedding=X'00' WHERE rowid=24")
        self.conn.commit()
        self.compare(limit=20)

    def test_no_clusters_returns_empty_after_encoding(self) -> None:
        self.assertEqual(self.compare(), [])
        self.assertEqual(self.boundary.encodes, 2)
        self.assertFalse(self.conn.in_transaction)

    def test_zero_query_preserves_all_score_ties(self) -> None:
        self.seed(20, clusters=1)
        self.boundary.query = [0.0] * 4
        self.compare(limit=10)


class TestClusteredSearchBounds(ClusteredSearchFixture):
    def test_large_cluster_bounds_vectors_heap_and_full_content(self) -> None:
        self.seed(2107, clusters=1, directives=False, content_size=4096)
        observed = ObservedConnection(self.conn)
        heap_peak = 0
        arrays_live = 0
        arrays_peak = 0
        original_array = NUMPY.array

        def release_array() -> None:
            nonlocal arrays_live
            arrays_live -= 1

        def array(*args: object, **kwargs: object) -> object:
            nonlocal arrays_live, arrays_peak
            result = original_array(*args, **kwargs)
            arrays_live += 1
            arrays_peak = max(arrays_peak, arrays_live)
            weakref.finalize(result, release_array)
            return result

        def push(heap: list, item: tuple) -> None:
            nonlocal heap_peak
            heapq.heappush(heap, item)
            heap_peak = max(heap_peak, len(heap))

        def replace(heap: list, item: tuple) -> None:
            nonlocal heap_peak
            heapq.heapreplace(heap, item)
            heap_peak = max(heap_peak, len(heap))

        self.module.heapq = types.SimpleNamespace(merge=heapq.merge, heappush=push, heapreplace=replace)
        with patch.object(NUMPY, "array", array):
            result = self.module.search_clustered(observed, "synthetic query", limit=100)
        self.assertEqual(len(result), 100)
        self.assertEqual(observed.rows["content"], 100)
        self.assertEqual(observed.rows["candidate_ids"], 2107)
        self.assertLessEqual(max(observed.parameter_counts), 500)
        self.assertLessEqual(observed.max_fetchall["vectors"], 500)
        self.assertEqual(heap_peak, 100)
        self.assertLessEqual(arrays_peak, 505)
        self.assertGreater(arrays_peak, 100)
        self.assertFalse(self.conn.in_transaction)

    def test_low_sqlite_variable_limit_and_many_selected_clusters(self) -> None:
        self.seed(619, clusters=619, all_ties=True, directives=False)
        expected = self.module.reference_search_clustered(self.conn, "synthetic query", limit=580, top_clusters=619)
        cap = 17 if hasattr(self.conn, "setlimit") else 500
        if hasattr(self.conn, "setlimit"):
            old_limit = self.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, cap)
            self.addCleanup(self.conn.setlimit, sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, old_limit)
        observed = ObservedConnection(self.conn, maximum_parameters=cap)
        result = self.module.search_clustered(observed, "synthetic query", limit=580, top_clusters=619)
        self.assertEqual(result, expected)
        self.assertLessEqual(max(observed.parameter_counts), cap)
        self.assertEqual(observed.rows["content"], 580)
        self.assertGreater(observed.closed, 1)

    def test_negative_limit_materializes_actual_requested_output_only(self) -> None:
        self.seed(31, clusters=1, all_ties=True, directives=False)
        observed = ObservedConnection(self.conn)
        result = self.module.search_clustered(observed, "synthetic query", limit=-7)
        self.assertEqual(result, self.module.reference_search_clustered(self.conn, "synthetic query", limit=-7))
        self.assertEqual(observed.rows["content"], 31 - 7)

    def test_legacy_duplicate_memberships_do_not_duplicate_candidates(self) -> None:
        self.seed(31, clusters=3, all_ties=True, directives=False)
        self.conn.execute("ALTER TABLE message_clusters RENAME TO old_memberships")
        self.conn.execute("CREATE TABLE message_clusters(message_id INTEGER,cluster_id INTEGER)")
        self.conn.execute("INSERT INTO message_clusters SELECT * FROM old_memberships")
        self.conn.execute("INSERT INTO message_clusters SELECT * FROM old_memberships")
        self.conn.commit()
        self.compare(limit=100)


class TestClusteredSearchSnapshots(ClusteredSearchFixture):
    def test_concurrent_changes_do_not_mix_scores_and_winner_content(self) -> None:
        self.seed(30, clusters=1, all_ties=True, directives=False)
        expected = self.module.reference_search_clustered(self.conn, "synthetic query", limit=5)
        writer = self.open_db()
        observed = ObservedConnection(self.conn)

        def change() -> None:
            self.assertTrue(self.conn.in_transaction)
            writer.execute("UPDATE messages SET content='synthetic replacement',directive=1 WHERE id=3")
            writer.execute("DELETE FROM messages WHERE id=10")
            writer.execute("UPDATE vec_messages SET embedding=? WHERE rowid=17", (blob([-1.0, 0.0, 0.0, 0.0]),))
            writer.execute("UPDATE message_clusters SET cluster_id=99 WHERE message_id=24")
            writer.execute("UPDATE cluster_centroids SET centroid=?", (blob([-1.0, 0.0, 0.0, 0.0]),))
            writer.commit()

        observed.before_vectors = change
        result = self.module.search_clustered(observed, "synthetic query", limit=5)
        self.assertEqual(result, expected)
        self.assertFalse(self.conn.in_transaction)
        self.assertNotEqual(self.module.search_clustered(self.conn, "synthetic query", limit=5), expected)

    def test_caller_transaction_is_not_committed(self) -> None:
        self.seed(15, clusters=1, all_ties=True, directives=False)
        reader = self.open_db()
        before = reader.execute("SELECT content FROM messages WHERE id=3").fetchone()
        self.conn.execute("UPDATE messages SET content='synthetic pending' WHERE id=3")
        rows = self.module.search_clustered(self.conn, "synthetic query", limit=1)
        self.assertEqual(rows[0]["content"], "synthetic pending")
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(reader.execute("SELECT content FROM messages WHERE id=3").fetchone(), before)
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=3").fetchone(), before)

    def test_encoding_finishes_before_owned_snapshot_begins(self) -> None:
        self.seed(10, clusters=1)
        self.boundary.before_encode = lambda: self.assertFalse(self.conn.in_transaction)
        self.module.search_clustered(self.conn, "synthetic query")

    def test_materialization_failure_closes_owned_transaction(self) -> None:
        self.seed(10, clusters=1)
        observed = ObservedConnection(self.conn)

        def fail() -> None:
            raise RuntimeError("synthetic read failure")

        observed.before_content = fail
        with self.assertRaisesRegex(RuntimeError, "synthetic read failure"):
            self.module.search_clustered(observed, "synthetic query")
        self.assertFalse(self.conn.in_transaction)
        self.assertGreater(observed.closed, 0)

    def test_cancellation_keeps_caller_work_and_closes_candidate_cursors(self) -> None:
        self.seed(10, clusters=1, all_ties=True, directives=False)
        self.conn.execute("UPDATE messages SET content='synthetic pending' WHERE id=3")
        observed = ObservedConnection(self.conn)

        def cancel() -> None:
            raise KeyboardInterrupt("synthetic cancellation")

        observed.before_vectors = cancel
        with self.assertRaises(KeyboardInterrupt):
            self.module.search_clustered(observed, "synthetic query")
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=3").fetchone(), ("synthetic pending",))
        self.assertGreater(observed.closed, 0)
        self.conn.rollback()

    def test_caller_materialization_failure_preserves_pending_work(self) -> None:
        self.seed(10, clusters=1, all_ties=True, directives=False)
        self.conn.execute("UPDATE messages SET content='synthetic pending' WHERE id=3")
        observed = ObservedConnection(self.conn)

        def fail() -> None:
            raise RuntimeError("synthetic read failure")

        observed.before_content = fail
        with self.assertRaisesRegex(RuntimeError, "synthetic read failure"):
            self.module.search_clustered(observed, "synthetic query")
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=3").fetchone(), ("synthetic pending",))
        self.conn.rollback()

    def test_nonfinite_scores_fail_without_identifying_source_rows(self) -> None:
        self.seed(10, clusters=1, all_ties=True, directives=False)
        for target in ("centroid", "embedding", "query"):
            with self.subTest(target=target):
                self.seed(10, clusters=1, all_ties=True, directives=False)
                self.boundary.query = [1.0, 0.0, 0.0, 0.0]
                invalid = [float("nan"), 0.0, 0.0, 0.0]
                if target == "query":
                    self.boundary.query = invalid
                else:
                    table = "cluster_centroids" if target == "centroid" else "vec_messages"
                    self.conn.execute(f"UPDATE {table} SET {target}=?", (blob(invalid),))
                    self.conn.commit()
                with self.assertRaises(ValueError) as raised:
                    self.module.search_clustered(self.conn, "synthetic query")
                self.assertEqual(str(raised.exception), "Clustered search similarity is nonfinite")
                self.assertFalse(self.conn.in_transaction)


def benchmark_clustered_search(variant: str, clusters: int) -> dict:
    """Prepared Linux/real-NumPy benchmark; invoke one case per fresh process."""
    import gc
    import hashlib
    import json
    import resource
    import statistics
    import sys
    import time
    import tracemalloc

    if isinstance(NUMPY, SyntheticNumpy) or sys.platform != "linux":
        raise RuntimeError("Benchmark requires explicit real NumPy on the isolated Linux test host")
    if variant not in ("reference", "current") or clusters not in (3, 30):
        raise ValueError("Benchmark requires a frozen variant and cluster count")
    fixture = ClusteredSearchFixture()
    fixture.setUp()
    try:
        fixture.seed(10000, clusters=clusters, random_seed=41, directives=False, content_size=4096)
        for table, key, column in (("vec_messages", "rowid", "embedding"),
                                   ("cluster_centroids", "cluster_id", "centroid")):
            for identifier, vector in fixture.conn.execute(f"SELECT {key},{column} FROM {table}").fetchall():
                fixture.conn.execute(f"UPDATE {table} SET {column}=? WHERE {key}=?", (vector + bytes(252 * 4), identifier))
        fixture.conn.commit()
        fixture.boundary.query = [1.0] + [0.0] * 255
        search = (fixture.module.reference_search_clustered if variant == "reference"
                  else fixture.module.search_clustered)
        samples = []
        fingerprints = set()
        for _ in range(7):
            gc.collect()
            observed = ObservedConnection(fixture.conn, maximum_parameters=10000)
            tracemalloc.start()
            try:
                started = time.perf_counter()
                result = search(observed, "synthetic query", limit=100, top_clusters=3)
                elapsed = time.perf_counter() - started
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            assert len(result) == 100
            fingerprints.add(hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest())
            samples.append({"seconds": elapsed, "python_peak_bytes": peak,
                            "full_message_rows": observed.rows["content"],
                            "maximum_parameters": max(observed.parameter_counts)})
            del result
        assert len(fingerprints) == 1
        return {"variant": variant, "clusters": clusters, "rows": 10000, "dimensions": 256,
                "limit": 100, "top_clusters": 3, "repetitions": 7,
                "source_sha256": hashlib.sha256((ROOT / "truememory/clustering.py").read_bytes()).hexdigest(),
                "reference_commit": "9b2b6ff", "numpy_version": NUMPY.__version__,
                "result_sha256": fingerprints.pop(), "samples": samples,
                "median_seconds": statistics.median(sample["seconds"] for sample in samples),
                "p95_seconds": sorted(sample["seconds"] for sample in samples)[-1],
                "process_high_water_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "memory_note": "Python allocation peak is not all native memory; process HWM includes fixture setup",
                "model_execution": False, "synthetic_only": True}
    finally:
        fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
