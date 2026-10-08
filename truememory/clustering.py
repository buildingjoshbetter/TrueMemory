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
import logging
import sqlite3
import struct
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from types import ModuleType

import numpy as np

logger = logging.getLogger(__name__)


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
            raise RuntimeError("Embedding model is busy before cluster commit")
        try:
            yield validate
        finally:
            vector_search._lock.release()

    return hold()


def _cluster_source_state(
    conn: sqlite3.Connection, vector_search: ModuleType, *, collect_categories: bool = False,
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
    for row in conn.execute(f"SELECT rowid, embedding FROM {quoted_table} ORDER BY rowid"):
        add(tuple(row))
    add(("messages",))
    categories = {}
    for row in conn.execute(
        "SELECT id, content, sender, recipient, timestamp, category, modality FROM messages ORDER BY id"
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
) -> int:
    """
    Cluster messages using HDBSCAN on their embeddings.

    Messages that don't fit into any cluster are marked as noise
    (cluster_id = -1).

    Args:
        conn:             Open database connection with vec_messages built.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        min_samples:      Minimum samples for HDBSCAN core points.

    Returns:
        Number of clusters found (excluding noise).
    """
    import hdbscan
    from truememory import vector_search

    # Model loading may also hold this lock. Wait before opening our read
    # transaction, and release both before any clustering computation.
    with vector_search._lock:
        with _cluster_transaction(conn):
            source_state, categories = _cluster_source_state(conn, vector_search, collect_categories=True)
            msg_ids, embeddings = _get_all_embeddings(conn)
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
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(normed)
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
    try:
        with _cluster_transaction(conn, write=True):
            # Never hold SQLite's writer lock while waiting for native model
            # load or a tier change. Keep the model stable through commit.
            model_lock_acquired = vector_search._lock.acquire(blocking=False)
            if not model_lock_acquired:
                raise RuntimeError("Embedding model is busy; retry clustering after model loading or tier switching")
            current_state, _ = _cluster_source_state(conn, vector_search)
            if current_state != source_state:
                raise RuntimeError("Clustering source changed during computation; retry consolidation")
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
