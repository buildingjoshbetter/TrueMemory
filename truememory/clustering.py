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

import sqlite3
import struct
from collections import defaultdict

import numpy as np


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


def _init_cluster_tables(conn: sqlite3.Connection):
    """Create clustering tables if they don't exist."""
    conn.executescript(_CLUSTER_SCHEMA)
    conn.commit()


# ---------------------------------------------------------------------------
# Embedding extraction helpers
# ---------------------------------------------------------------------------

def _get_all_embeddings(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """
    Extract all message embeddings from vec_messages.

    Returns:
        Tuple of (message_ids, embeddings_array) where embeddings_array
        is shape (n_messages, dim).
    """
    rows = conn.execute(
        "SELECT rowid, embedding FROM vec_messages ORDER BY rowid"
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

    _init_cluster_tables(conn)

    # Clear existing clusters
    conn.execute("DELETE FROM message_clusters")
    conn.execute("DELETE FROM cluster_centroids")

    # Get embeddings
    msg_ids, embeddings = _get_all_embeddings(conn)
    if len(msg_ids) == 0:
        conn.commit()
        return 0

    # Normalize for cosine-like clustering
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # on normalized vectors ≈ cosine
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(normed)

    # Store cluster assignments
    rows = []
    for msg_id, label in zip(msg_ids, labels):
        is_noise = 1 if label == -1 else 0
        rows.append((msg_id, int(label), is_noise))

    conn.executemany(
        "INSERT INTO message_clusters (message_id, cluster_id, noise) VALUES (?, ?, ?)",
        rows,
    )

    # Build centroids
    cluster_vecs = defaultdict(list)
    cluster_msg_ids = defaultdict(list)
    for msg_id, label, emb in zip(msg_ids, labels, embeddings):
        if label >= 0:
            cluster_vecs[label].append(emb)
            cluster_msg_ids[label].append(msg_id)

    for cid, vecs in cluster_vecs.items():
        centroid = np.mean(vecs, axis=0).astype(np.float32)
        # Get session range for this cluster
        placeholders = ",".join("?" * len(cluster_msg_ids[cid]))
        sessions = conn.execute(
            f"SELECT DISTINCT category FROM messages WHERE id IN ({placeholders})",
            cluster_msg_ids[cid],
        ).fetchall()
        session_range = ", ".join(sorted(s[0] for s in sessions if s[0]))

        conn.execute(
            "INSERT INTO cluster_centroids (cluster_id, centroid, message_count, session_range) "
            "VALUES (?, ?, ?, ?)",
            (int(cid), _serialize_f32(centroid), len(vecs), session_range),
        )

    conn.commit()
    n_clusters = len(cluster_vecs)
    return n_clusters


# ---------------------------------------------------------------------------
# Cluster-scoped search
# ---------------------------------------------------------------------------

def search_clustered(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    top_clusters: int = 3,
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

    model = get_model()
    query_vec = model.encode([query])[0].astype(np.float32)

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

    messages = conn.execute(
        f"""SELECT m.id, m.content, m.sender, m.recipient, m.timestamp,
                   m.category, m.modality
            FROM messages m
            WHERE m.id IN ({id_placeholders})""",
        msg_ids_list,
    ).fetchall()

    if not messages:
        return []

    # Score each message by vector similarity to query
    results = []
    for msg in messages:
        msg_id = msg[0]
        # Get embedding
        try:
            emb_row = conn.execute(
                "SELECT embedding FROM vec_messages WHERE rowid = ?", (msg_id,)
            ).fetchone()
            if emb_row:
                dim = len(emb_row[0]) // 4
                msg_vec = np.array(
                    struct.unpack(f"{dim}f", emb_row[0]), dtype=np.float32
                )
                sim = float(np.dot(query_vec, msg_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(msg_vec) + 1e-9
                ))
            else:
                sim = 0.0
        except Exception:
            sim = 0.0

        results.append({
            "id": msg_id,
            "content": msg[1],
            "sender": msg[2],
            "recipient": msg[3],
            "timestamp": msg[4],
            "category": msg[5],
            "modality": msg[6],
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
