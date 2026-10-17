"""
TrueMemory Scene Clustering
=========================

Groups conversation messages into coherent "scenes" or episodes using
HDBSCAN clustering on message embeddings.  Provides two-stage retrieval:
first identify relevant clusters, then search within them.

This is inspired by EverMemOS's scene-clustering approach but implemented
on top of TrueMemory's existing SQLite + Model2Vec infrastructure.

Usage::

    from truememory.clustering import cluster_messages, search_clustered

    cluster_messages(conn)
    results = search_clustered(conn, "What job did Jordan get?", limit=10)

Dependencies:
    - hdbscan (``pip install hdbscan``)
    - numpy
    - truememory.vector_search (for embeddings)
"""

from __future__ import annotations

import hashlib
import heapq
import logging
import math
import sqlite3
import struct
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from itertools import islice
from types import ModuleType

import numpy as np

from truememory.memory_budget import LocalMemoryPolicy
from truememory.storage import begin_layer_publication

_LOCAL_MEMORY_POLICY = LocalMemoryPolicy()

logger = logging.getLogger(__name__)


class ClusterModelBusyError(RuntimeError):
    """Model ownership is temporarily unavailable; no input version changed."""


class ClusterSourceChangedError(RuntimeError):
    """Messages or vectors changed while clustering computed; retry later."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CLUSTER_SCHEMA = """
CREATE TABLE IF NOT EXISTS message_clusters (
    message_id   INTEGER PRIMARY KEY REFERENCES messages(id),
    cluster_id   INTEGER NOT NULL,
    noise        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cluster_centroids (
    cluster_id   INTEGER PRIMARY KEY,
    centroid     BLOB NOT NULL,
    message_count INTEGER DEFAULT 0,
    session_range TEXT DEFAULT '',
    summary      TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_cluster_id ON message_clusters(cluster_id);
"""


def _init_cluster_tables(conn: sqlite3.Connection) -> None:
    """Create tables inside the publication transaction without committing it."""
    for statement in _CLUSTER_SCHEMA.split(";"):
        if statement.strip():
            conn.execute(statement)


@contextmanager
def _cluster_transaction(conn: sqlite3.Connection, *, write: bool = False) -> Iterator[None]:
    owned = not conn.in_transaction
    if owned:
        conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
    else:
        conn.execute("SAVEPOINT truememory_clusters")
    completed = False
    try:
        if write and not owned:
            # Acquire the writer before validating a caller's read snapshot.
            # A stale WAL snapshot fails here; no source row is changed.
            conn.execute("UPDATE messages SET id = id WHERE 0")
        yield
        conn.execute("COMMIT" if owned else "RELEASE truememory_clusters")
        completed = True
    finally:
        if not completed and conn.in_transaction:
            if owned:
                conn.rollback()
            else:
                conn.execute("ROLLBACK TO truememory_clusters")
                conn.execute("RELEASE truememory_clusters")


def _cluster_dependency_identity(conn: sqlite3.Connection, vector_search: ModuleType) -> tuple[tuple, ...]:
    """Read cheap identity under the model lock and a coherent SQLite snapshot."""
    group = vector_search._active_tier_group()
    vec_table = vector_search._active_vec_table(conn)
    identity = [("model", vector_search.EMBEDDING_MODEL, vector_search._embedding_dim, group, vec_table)]
    definitions = conn.execute(
        "SELECT type, name, rootpage, sql FROM sqlite_master "
        "WHERE name IN (?, 'messages', 'metadata', 'vector_cache_registry') ORDER BY name",
        (vec_table,),
    ).fetchall()
    identity.append(("schema",))
    identity.extend(tuple(row) for row in definitions)
    tables = {row[1] for row in definitions}
    identity.append(("registry",))
    if "vector_cache_registry" in tables:
        for row in conn.execute("SELECT * FROM vector_cache_registry WHERE tier_group = ?", (group,)):
            identity.append(tuple(row))
    identity.append(("metadata",))
    if "metadata" in tables:
        build_key = f"vec_build_state:{vec_table}"
        for row in conn.execute(
            "SELECT key, value, updated_at FROM metadata "
            "WHERE key IN ('embed_model', 'embed_dim', ?, ?) ORDER BY key",
            (build_key, f"vec_source_v1:{vec_table}"),
        ):
            if row[0] == build_key and row[1] == "in_progress":
                raise RuntimeError("Clustering requires a completed vector index; rebuild is in progress")
            identity.append(tuple(row))
    return tuple(identity)


def cluster_publication_guard(conn: sqlite3.Connection) -> AbstractContextManager[Callable[[], None]]:
    """Capture now; later hold model identity through the runner's terminal write.

    The caller must run cluster_messages in its owned transaction before entering
    this guard. Its raw-input fence and retained SQLite writer protect source and
    vector bytes; this guard closes the runtime-model gap after nested RELEASE.
    """
    if conn.in_transaction:
        raise RuntimeError("Cluster publication capture requires a clean transaction boundary")
    from truememory import vector_search

    with vector_search._lock:
        with _cluster_transaction(conn):
            expected = _cluster_dependency_identity(conn, vector_search)

    def validate() -> None:
        if _cluster_dependency_identity(conn, vector_search) != expected:
            raise RuntimeError("Clustering dependency changed before outer commit")

    @contextmanager
    def hold() -> Iterator[Callable[[], None]]:
        if not conn.in_transaction:
            raise RuntimeError("Cluster publication requires the runner's active transaction")
        # Ensure writer admission before taking the nonblocking model lock.
        conn.execute("UPDATE messages SET id=id WHERE 0")
        acquired = vector_search._lock.acquire(blocking=False)
        if not acquired:
            raise ClusterModelBusyError("Embedding model is busy before cluster commit")
        try:
            yield validate
        finally:
            vector_search._lock.release()

    return hold()


def _cluster_source_state(
    conn: sqlite3.Connection, vector_search: ModuleType, *, collect_categories: bool = False,
    watermark: int | None = None,
) -> tuple[bytes, dict[int, str | None]]:
    """Fingerprint raw inputs under the model lock and a database snapshot."""
    digest = hashlib.sha256()

    def add(row: tuple) -> None:
        digest.update(struct.pack(">Q", len(row)))
        for value in row:
            if value is None:
                tag, data = b"n", b""
            elif isinstance(value, bytes):
                tag, data = b"b", value
            elif isinstance(value, str):
                tag, data = b"s", value.encode("utf-8")
            elif isinstance(value, int):
                tag, data = b"i", str(value).encode("ascii")
            elif isinstance(value, float):
                tag, data = b"f", struct.pack(">d", value)
            else:
                raise TypeError("Unsupported SQLite value in clustering source")
            digest.update(tag)
            digest.update(struct.pack(">Q", len(data)))
            digest.update(data)

    identity = _cluster_dependency_identity(conn, vector_search)
    for row in identity:
        add(row)
    vec_table = identity[0][-1]
    add(("vectors",))
    quoted_table = '"' + vec_table.replace('"', '""') + '"'
    # Vector rowids are message ids, so one watermark bounds both inputs.
    bound = "" if watermark is None else " WHERE rowid <= ?"
    parameters = () if watermark is None else (watermark,)
    for row in conn.execute(f"SELECT rowid, embedding FROM {quoted_table}{bound} ORDER BY rowid", parameters):
        add(tuple(row))
    add(("messages",))
    categories = {}
    for row in conn.execute(
        "SELECT id, content, sender, recipient, timestamp, category, modality FROM messages"
        + bound.replace("rowid", "id") + " ORDER BY id", parameters,
    ):
        add(tuple(row))
        if collect_categories:
            categories[row[0]] = row[5]
    return digest.digest(), categories


# ---------------------------------------------------------------------------
# Embedding extraction helpers
# ---------------------------------------------------------------------------

def _get_all_embeddings(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """
    Extract all message embeddings from the active vec_messages table.

    Returns:
        Tuple of (message_ids, embeddings_array) where embeddings_array
        is shape (n_messages, dim).
    """
    from truememory.vector_search import _active_vec_table
    vec_table = _active_vec_table(conn)
    rows = conn.execute(
        f"SELECT rowid, embedding FROM {vec_table} ORDER BY rowid"
    ).fetchall()

    if not rows:
        return [], np.array([])

    ids = []
    vecs = []
    for row_id, blob in rows:
        ids.append(row_id)
        dim = len(blob) // 4  # float32 = 4 bytes
        vec = np.array(struct.unpack(f"{dim}f", blob), dtype=np.float32)
        vecs.append(vec)

    return ids, np.stack(vecs)


def _serialize_f32(vector: np.ndarray) -> bytes:
    """Serialize a float32 vector to raw bytes."""
    return struct.pack(f"{len(vector)}f", *vector.tolist())


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_messages(
    conn: sqlite3.Connection,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    *,
    cluster_selection_method: str = "eom",
) -> int:
    """
    Cluster messages using HDBSCAN on their embeddings.

    Messages that don't fit into any cluster are marked as noise
    (cluster_id = -1).

    Args:
        conn:             Open database connection with vec_messages built.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        min_samples:      Minimum samples for HDBSCAN core points.
        cluster_selection_method: Explicit HDBSCAN selection policy, eom or leaf.

    Returns:
        Number of clusters found (excluding noise).
    """
    if type(min_cluster_size) is not int or not 2 <= min_cluster_size <= 2147483647:
        raise ValueError("min_cluster_size must be an integer from 2 through 2147483647")
    if type(min_samples) is not int or not 1 <= min_samples <= 2147483647:
        raise ValueError("min_samples must be an integer from 1 through 2147483647")
    if type(cluster_selection_method) is not str or cluster_selection_method not in {"eom", "leaf"}:
        raise ValueError("cluster_selection_method must be eom or leaf")

    import hdbscan
    from truememory import vector_search

    # Standalone callers can wait before opening a read transaction. A caller
    # transaction may already own SQLite's writer, so never wait inside it.
    if not vector_search._lock.acquire(blocking=not conn.in_transaction):
        raise ClusterModelBusyError("Embedding model is busy before clustering snapshot")
    try:
        with _cluster_transaction(conn):
            source_state, categories = _cluster_source_state(conn, vector_search, collect_categories=True)
            msg_ids, embeddings = _get_all_embeddings(conn)
    finally:
        vector_search._lock.release()
    if any(mid not in categories for mid in msg_ids):
        raise RuntimeError("Clustering found vectors without source messages; rebuild the vector index")

    labels = []
    if msg_ids:
        # Normalize for cosine-like clustering.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",  # on normalized vectors ≈ cosine
            cluster_selection_method=cluster_selection_method,
        )
        try:
            labels = clusterer.fit_predict(normed)
        except ValueError:
            if len(msg_ids) > min_samples:
                raise
            # No point can have min_samples neighbours, so none is a core point
            # and all are noise until the store grows. HDBSCAN raises instead
            # ("k must be less than or equal to the number of training
            # points"): hdbscan 0.8.42 on one vector, builds that do not clamp
            # min_samples to the input on up to min_samples vectors.
            labels = [-1] * len(msg_ids)
        if len(labels) != len(msg_ids):
            raise ValueError("Clustering returned an incomplete assignment set")

    # Store cluster assignments
    rows = []
    for msg_id, label in zip(msg_ids, labels):
        is_noise = 1 if label == -1 else 0
        rows.append((msg_id, int(label), is_noise))

    # Build centroids
    cluster_vecs = defaultdict(list)
    cluster_msg_ids = defaultdict(list)
    for msg_id, label, emb in zip(msg_ids, labels, embeddings):
        if label >= 0:
            cluster_vecs[label].append(emb)
            cluster_msg_ids[label].append(msg_id)

    centroids = []
    for cid, vecs in cluster_vecs.items():
        centroid = np.mean(vecs, axis=0).astype(np.float32)
        sessions = {categories[mid] for mid in cluster_msg_ids[cid] if categories[mid]}
        session_range = ", ".join(sorted(sessions))
        centroids.append((int(cid), _serialize_f32(centroid), len(vecs), session_range))

    model_lock_acquired = False
    # A maintenance build may publish over messages appended since its
    # snapshot; they stay unclustered until the next build.
    watermark = begin_layer_publication(conn)
    try:
        with _cluster_transaction(conn, write=True):
            # Never hold SQLite's writer lock while waiting for native model
            # load or a tier change. Keep the model stable through commit.
            model_lock_acquired = vector_search._lock.acquire(blocking=False)
            if not model_lock_acquired:
                raise ClusterModelBusyError("Embedding model is busy; retry clustering after model loading or tier switching")
            current_state, _ = _cluster_source_state(conn, vector_search, watermark=watermark)
            if current_state != source_state:
                raise ClusterSourceChangedError("Clustering source changed during computation; retry consolidation")
            _init_cluster_tables(conn)
            conn.execute("DELETE FROM message_clusters")
            conn.execute("DELETE FROM cluster_centroids")
            conn.executemany(
                "INSERT INTO message_clusters (message_id, cluster_id, noise) VALUES (?, ?, ?)", rows,
            )
            conn.executemany(
                "INSERT INTO cluster_centroids (cluster_id, centroid, message_count, session_range) "
                "VALUES (?, ?, ?, ?)", centroids,
            )
    finally:
        if model_lock_acquired:
            vector_search._lock.release()
    return len(cluster_vecs)


# ---------------------------------------------------------------------------
# Cluster-scoped search
# ---------------------------------------------------------------------------

def search_clustered(
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

    configured_memory = "_LOCAL_MEMORY_POLICY" in globals()
    if configured_memory:
        from truememory.memory_budget import (
            ProfileKey, budgeted_encode_with_model_ownership, validate_local_policy,
        )
        validate_local_policy(globals()["_LOCAL_MEMORY_POLICY"])
    model = get_model()
    if configured_memory:
        values = budgeted_encode_with_model_ownership(
            model, [query], ProfileKey("cluster query embedding", "local_embedding", input_item_count=1),
        )
    else:
        values = encode_with_model_ownership(model, [query])
    query_vec = values[0].astype(np.float32)

    # Keep centroids, memberships, vectors and winner content in one snapshot.
    # Encoding completes before we open an owned read transaction.
    with _cluster_transaction(conn):
        return _search_clustered_snapshot(conn, query_vec, limit, top_clusters, include_directives)


def _cluster_query_chunk_size(conn: sqlite3.Connection) -> int:
    getlimit = getattr(conn, "getlimit", None)
    size = min(500, getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)) if getlimit else 500
    if size < 1:
        raise ValueError("Clustered search requires SQL query parameters")
    return size


def _cluster_candidate_ids(
    conn: sqlite3.Connection, selected_clusters: list[int], include_directives: bool, chunk_size: int,
) -> Iterator[int]:
    directive_filter = "" if include_directives else " AND (m.directive = 0 OR m.directive IS NULL)"
    cursors = []
    try:
        for start in range(0, len(selected_clusters), chunk_size):
            chunk = selected_clusters[start:start + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            cursors.append(conn.execute(
                f"SELECT DISTINCT m.id FROM messages m "
                f"JOIN message_clusters c ON c.message_id = m.id "
                f"WHERE c.cluster_id IN ({placeholders}){directive_filter} ORDER BY m.id", chunk,
            ))
        # The old primary-key IN lookup delivered ascending message IDs. Keep
        # that tie order across chunks and deduplicate legacy membership rows.
        previous = None
        streams = (map(lambda row: row[0], cursor) for cursor in cursors)
        for message_id in heapq.merge(*streams):
            if message_id != previous:
                yield message_id
            previous = message_id
    finally:
        for cursor in cursors:
            cursor.close()


def _search_clustered_snapshot(
    conn: sqlite3.Connection, query_vec: np.ndarray, limit: int,
    top_clusters: int, include_directives: bool,
) -> list[dict]:
    try:
        cluster_count = conn.execute("SELECT COUNT(*) FROM cluster_centroids").fetchone()[0]
    except Exception:
        return []
    if cluster_count == 0:
        return []

    centroids = conn.execute(
        "SELECT cluster_id, centroid, message_count FROM cluster_centroids"
    ).fetchall()
    cluster_scores = []
    for cid, centroid_blob, msg_count in centroids:
        dim = len(centroid_blob) // 4
        centroid = np.array(struct.unpack(f"{dim}f", centroid_blob), dtype=np.float32)
        sim = float(np.dot(query_vec, centroid) / (
            np.linalg.norm(query_vec) * np.linalg.norm(centroid) + 1e-9
        ))
        if not math.isfinite(sim):
            raise ValueError("Clustered search similarity is nonfinite")
        cluster_scores.append((cid, sim, msg_count))
    cluster_scores.sort(key=lambda item: item[1], reverse=True)
    selected_clusters = [item[0] for item in cluster_scores[:top_clusters]]
    if not selected_clusters:
        return []

    chunk_size = _cluster_query_chunk_size(conn)
    keep = limit
    if limit < 0:
        with closing(_cluster_candidate_ids(conn, selected_clusters, include_directives, chunk_size)) as candidates:
            keep = max(0, sum(1 for _ in candidates) + limit)
    if keep == 0:
        return []

    from truememory.vector_search import _active_vec_table
    vec_table = _active_vec_table(conn)
    quoted_table = '"' + vec_table.replace('"', '""') + '"'
    query_norm = np.linalg.norm(query_vec) + 1e-9
    winners: list[tuple[float, int, int]] = []
    with closing(_cluster_candidate_ids(conn, selected_clusters, include_directives, chunk_size)) as candidates:
        while chunk := list(islice(candidates, chunk_size)):
            placeholders = ",".join("?" * len(chunk))
            embeddings = {}
            try:
                rows = conn.execute(
                    f"SELECT rowid, embedding FROM {quoted_table} WHERE rowid IN ({placeholders})", chunk,
                ).fetchall()
                for message_id, blob in rows:
                    dim = len(blob) // 4
                    embeddings[message_id] = np.array(struct.unpack(f"{dim}f", blob), dtype=np.float32)
            except Exception:
                logger.debug("Batch embedding fetch failed for chunk", exc_info=True)
            for message_id in chunk:
                vector = embeddings.get(message_id)
                sim = 0.0 if vector is None else float(np.dot(query_vec, vector) / (
                    query_norm * (np.linalg.norm(vector) + 1e-9)
                ))
                if not math.isfinite(sim):
                    raise ValueError("Clustered search similarity is nonfinite")
                candidate = (sim, -message_id, message_id)
                if len(winners) < keep:
                    heapq.heappush(winners, candidate)
                elif candidate > winners[0]:
                    heapq.heapreplace(winners, candidate)

    winners.sort(reverse=True)
    messages = {}
    for start in range(0, len(winners), chunk_size):
        ids = [item[2] for item in winners[start:start + chunk_size]]
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, content, sender, recipient, timestamp, category, modality, directive "
            f"FROM messages WHERE id IN ({placeholders})", ids,
        )
        for row in rows:
            messages[row[0]] = row
    results = []
    for score, _, message_id in winners:
        msg = messages[message_id]
        results.append({
            "id": message_id, "content": msg[1], "sender": msg[2], "recipient": msg[3],
            "timestamp": msg[4], "category": msg[5], "modality": msg[6],
            "directive": bool(msg[7]), "score": score, "source": "clustered",
        })
    return results


def get_cluster_info(conn: sqlite3.Connection) -> list[dict]:
    """
    Get summary information about all clusters.

    Returns:
        List of cluster info dicts with cluster_id, message_count,
        session_range.
    """
    try:
        rows = conn.execute(
            "SELECT cluster_id, message_count, session_range, summary "
            "FROM cluster_centroids ORDER BY cluster_id"
        ).fetchall()
    except Exception:
        return []

    return [
        {
            "cluster_id": r[0],
            "message_count": r[1],
            "session_range": r[2],
            "summary": r[3],
        }
        for r in rows
    ]
