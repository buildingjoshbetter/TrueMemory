"""
TrueMemory Vector Search
======================

Semantic search using Model2Vec (potion-base-8M, 256-dim) embeddings stored
in a sqlite-vec virtual table.  Provides nearest-neighbor retrieval based on
cosine distance so that queries like "networking problems" can surface results
about ECONNREFUSED even when no keywords overlap.

Usage::

    from truememory.storage import create_db, load_messages_from_file
    from truememory.vector_search import init_vec_table, build_vectors, search_vector

    conn = create_db("truememory.db")
    load_messages_from_file(conn, "synthetic_v2_messages.json")
    init_vec_table(conn)
    build_vectors(conn)
    results = search_vector(conn, "networking problems", limit=5)

Dependencies:
    - model2vec (``pip install model2vec``)
    - sqlite-vec (``pip install sqlite-vec``)
    - numpy
"""

from __future__ import annotations

import os
import struct
import sqlite3
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

# Public tier names → internal model identifiers
_TIER_ALIASES = {
    "base": "model2vec",
    "pro": "qwen3",
}

_MODEL_DIMS = {
    "model2vec": 256,
    "minilm": 384,
    "bge-small": 384,
    "qwen3": 1024,
}


def _resolve_model_name(name: str) -> str:
    """Resolve a public tier name (base/pro) or internal name to an internal name."""
    return _TIER_ALIASES.get(name.lower(), name)


_raw_env = os.environ.get("TRUEMEMORY_EMBED_MODEL", "base")
EMBEDDING_MODEL = _resolve_model_name(_raw_env)

_model = None
_embedding_dim: int = _MODEL_DIMS.get(EMBEDDING_MODEL, 256)
_lock = threading.Lock()


def set_embedding_model(name: str) -> None:
    """Switch the embedding model. Must be called BEFORE init_vec_table/build_vectors.

    Accepts tier names ("base", "pro") or internal model names.
    """
    global EMBEDDING_MODEL, _model, _embedding_dim
    with _lock:
        _model = None  # Force reload
        EMBEDDING_MODEL = _resolve_model_name(name)
        _embedding_dim = get_embedding_dim(EMBEDDING_MODEL)


def get_embedding_dim(name: str | None = None) -> int:
    """Return the embedding dimension for a given model name."""
    name = _resolve_model_name(name) if name else EMBEDDING_MODEL
    return _MODEL_DIMS.get(name, 256)


def get_model():
    """Lazy-load the embedding model (singleton)."""
    global _model, _embedding_dim
    if _model is not None:
        return _model  # Fast path, no lock needed
    with _lock:
        if _model is not None:
            return _model  # Another thread loaded it
        resolved = EMBEDDING_MODEL
        if resolved == "model2vec":
            from model2vec import StaticModel
            _model = StaticModel.from_pretrained("minishlab/potion-base-8M", force_download=False)
            _embedding_dim = 256
        elif resolved == "minilm":
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            _embedding_dim = 384
        elif resolved == "bge-small":
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            _embedding_dim = 384
        elif resolved == "qwen3":
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
            _embedding_dim = 1024
        else:
            from model2vec import StaticModel
            _model = StaticModel.from_pretrained("minishlab/potion-base-8M", force_download=False)
            _embedding_dim = 256
    return _model


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_f32(vector) -> bytes:
    """
    Serialize a float vector to raw little-endian bytes for sqlite-vec.

    Accepts any array-like (list, tuple, numpy array).  Returns a
    ``struct``-packed blob of 32-bit floats.
    """
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    return struct.pack(f"{len(vector)}f", *vector)


# ---------------------------------------------------------------------------
# Table initialization
# ---------------------------------------------------------------------------

def init_vec_table(conn: sqlite3.Connection) -> None:
    """
    Initialize the sqlite-vec extension and create both vector tables.

    Creates two virtual tables:
    - ``vec_messages`` -- completion embeddings (semantic similarity)
    - ``vec_messages_sep`` -- separation embeddings (with metadata for
      distinguishing similar messages)

    Both store 256-dimensional float32 embeddings keyed by ``rowid``
    which must correspond to ``messages.id``.

    This function is idempotent -- calling it on an already-initialized
    database is safe (``CREATE VIRTUAL TABLE IF NOT EXISTS``).

    Args:
        conn: An open SQLite connection (from :func:`truememory.storage.create_db`).
    """
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    dim = _embedding_dim
    # Completion embeddings (semantic similarity)
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages "
        f"USING vec0(embedding float[{dim}])"
    )
    # Separation embeddings (uniqueness, with metadata)
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages_sep "
        f"USING vec0(embedding float[{dim}])"
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Batch embedding & storage
# ---------------------------------------------------------------------------

_BATCH_SIZE = 100


def build_vectors(
    conn: sqlite3.Connection,
    messages: list[dict] | None = None,
) -> int:
    """
    Embed messages and store their vectors in the ``vec_messages`` table.

    If *messages* is ``None`` the function reads every row from the
    ``messages`` table.  Otherwise it uses the supplied list (each dict must
    have an ``"id"`` and ``"content"`` key).

    Embedding is performed in batches of 100 for efficiency.  Existing rows
    in ``vec_messages`` are cleared before inserting to keep the vector index
    in sync with the messages table.

    Args:
        conn:     Open database connection with sqlite-vec already loaded
                  (call :func:`init_vec_table` first).
        messages: Optional pre-fetched list of message dicts.  When omitted,
                  messages are read from the database.

    Returns:
        Number of vectors inserted.
    """
    # Fetch messages from DB if not provided.
    if messages is None:
        rows = conn.execute(
            "SELECT id, content FROM messages ORDER BY id"
        ).fetchall()
        messages = [{"id": row[0], "content": row[1]} for row in rows]

    if not messages:
        return 0

    # Clear existing vectors for a clean rebuild.
    conn.execute("DELETE FROM vec_messages")

    model = get_model()
    total = 0

    for start in range(0, len(messages), _BATCH_SIZE):
        batch = messages[start : start + _BATCH_SIZE]
        texts = [m["content"] for m in batch]
        ids = [m["id"] for m in batch]

        embeddings = model.encode(texts)  # shape (len(batch), 256)

        for msg_id, embedding in zip(ids, embeddings):
            conn.execute(
                "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
                (msg_id, serialize_f32(embedding)),
            )

        total += len(batch)

    conn.commit()
    return total


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def search_vector(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
) -> list[dict]:
    """
    Search for messages by vector similarity.

    Steps:
        1. Embed the *query* string using Model2Vec.
        2. Query the ``vec_messages`` virtual table for the nearest neighbours
           (cosine distance -- lower means more similar).
        3. Join with the ``messages`` table to retrieve full message data.
        4. Normalize distance scores into a 0--1 similarity score (higher is
           better) using ``score = 1 / (1 + distance)``.

    The normalization ``1 / (1 + d)`` maps distance **0** to score **1.0** and
    gracefully degrades toward **0** as distance grows, without ever going
    negative.  This is preferable to a raw ``1 - d`` clamp because cosine
    distances from sqlite-vec can exceed 1.0 in edge cases.

    Args:
        conn:  Open database connection with sqlite-vec loaded and vectors
               built (see :func:`init_vec_table`, :func:`build_vectors`).
        query: Natural-language search string.
        limit: Maximum number of results to return.

    Returns:
        List of result dicts sorted by descending similarity, each containing:
        ``id``, ``content``, ``sender``, ``recipient``, ``timestamp``,
        ``category``, ``modality``, ``score``.
    """
    model = get_model()
    query_embedding = model.encode([query])[0]  # shape (256,)

    query_blob = serialize_f32(query_embedding)

    # sqlite-vec requires k=? in WHERE clause for KNN queries.
    # We do the KNN search first, then JOIN with messages for full data.
    rows = conn.execute(
        """
        SELECT v.rowid, v.distance,
               m.content, m.sender, m.recipient,
               m.timestamp, m.category, m.modality
        FROM (
            SELECT rowid, distance
            FROM vec_messages
            WHERE embedding MATCH ? AND k = ?
        ) v
        JOIN messages m ON m.id = v.rowid
        ORDER BY v.distance
        """,
        (query_blob, limit),
    ).fetchall()

    results: list[dict] = []
    for row in rows:
        distance = row[1]
        score = 1.0 / (1.0 + distance)

        results.append(
            {
                "id": row[0],
                "content": row[2],
                "sender": row[3],
                "recipient": row[4],
                "timestamp": row[5],
                "category": row[6],
                "modality": row[7],
                "score": round(score, 6),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Separation embeddings (B2: dual embedding support)
# ---------------------------------------------------------------------------

def build_separation_vectors(
    conn: sqlite3.Connection,
    messages: list[dict] | None = None,
) -> int:
    """
    Build separation embeddings: ``"{sender} to {recipient} on {date}: {content}"``.

    Separation embeddings encode metadata (sender, recipient, date) alongside
    content so that messages from the same person on the same topic are
    distinguished from each other, improving retrieval precision when many
    similar messages exist.

    If *messages* is ``None`` the function reads every row from the
    ``messages`` table.  Otherwise it uses the supplied list (each dict must
    have ``"id"``, ``"content"``, ``"sender"``, ``"recipient"``, and
    ``"timestamp"`` keys).

    Args:
        conn:     Open database connection with sqlite-vec loaded
                  (call :func:`init_vec_table` first).
        messages: Optional pre-fetched list of message dicts.

    Returns:
        Number of separation vectors inserted.
    """
    if messages is None:
        rows = conn.execute(
            "SELECT id, content, sender, recipient, timestamp FROM messages ORDER BY id"
        ).fetchall()
        messages = [
            {"id": r[0], "content": r[1], "sender": r[2], "recipient": r[3], "timestamp": r[4]}
            for r in rows
        ]

    if not messages:
        return 0

    conn.execute("DELETE FROM vec_messages_sep")

    model = get_model()
    total = 0

    for start in range(0, len(messages), _BATCH_SIZE):
        batch = messages[start : start + _BATCH_SIZE]

        # Build separation texts with metadata
        texts = []
        ids = []
        for m in batch:
            sep_text = (
                f"{m.get('sender', '?')} to {m.get('recipient', '?')} "
                f"on {m.get('timestamp', '?')[:10]}: {m['content']}"
            )
            texts.append(sep_text)
            ids.append(m["id"])

        embeddings = model.encode(texts)

        for msg_id, embedding in zip(ids, embeddings):
            conn.execute(
                "INSERT INTO vec_messages_sep(rowid, embedding) VALUES (?, ?)",
                (msg_id, serialize_f32(embedding)),
            )

        total += len(batch)

    conn.commit()
    return total


def embed_single(conn: sqlite3.Connection, message_id: int, content: str) -> None:
    """
    Embed a single message and insert it into ``vec_messages``.

    This is the incremental counterpart to :func:`build_vectors` — it embeds
    one message at a time (~5ms with Model2Vec) for use with the production
    ``add()`` API.

    Args:
        conn:       Open database connection with sqlite-vec loaded
                    (call :func:`init_vec_table` first).
        message_id: The ``messages.id`` of the row being embedded.
        content:    The text to embed.
    """
    model = get_model()
    embedding = model.encode([content])[0]  # shape (dim,)
    conn.execute(
        "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
        (message_id, serialize_f32(embedding)),
    )
    # Caller is responsible for committing


def search_vector_separation(
    conn: sqlite3.Connection,
    query: str,
    sender: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """
    Search using separation embeddings.

    Separation embeddings include sender/recipient/date metadata, so they
    distinguish between otherwise-identical messages from different people
    or time periods.  Optionally prefix the query with a sender name for
    sender-aware retrieval.

    Args:
        conn:   Open database connection with sqlite-vec loaded and
                separation vectors built.
        query:  Natural-language search string.
        sender: Optional sender name to prepend to the query for
                sender-aware matching.
        limit:  Maximum number of results to return.

    Returns:
        List of result dicts sorted by descending similarity.
    """
    model = get_model()

    # Build query with optional sender context
    if sender:
        query_text = f"{sender}: {query}"
    else:
        query_text = query

    query_embedding = model.encode([query_text])[0]
    query_blob = serialize_f32(query_embedding)

    rows = conn.execute(
        """
        SELECT v.rowid, v.distance,
               m.content, m.sender, m.recipient,
               m.timestamp, m.category, m.modality
        FROM (
            SELECT rowid, distance
            FROM vec_messages_sep
            WHERE embedding MATCH ? AND k = ?
        ) v
        JOIN messages m ON m.id = v.rowid
        ORDER BY v.distance
        """,
        (query_blob, limit),
    ).fetchall()

    results: list[dict] = []
    for row in rows:
        distance = row[1]
        score = 1.0 / (1.0 + distance)
        results.append({
            "id": row[0],
            "content": row[2],
            "sender": row[3],
            "recipient": row[4],
            "timestamp": row[5],
            "category": row[6],
            "modality": row[7],
            "score": round(score, 6),
        })

    return results
