"""
TrueMemory Vector Search
========================

Tier-aware semantic search backed by a sqlite-vec virtual table. The
embedding model is resolved from the active tier when :func:`get_model`
is first called — the module itself imports cleanly with only cheap
constants computed:

    edge → Model2Vec potion-base-8M @ 256d (CPU, ~30MB)
    base → Qwen3-Embedding-0.6B @ 256d Matryoshka (GPU recommended, ~1.5GB)
    pro  → Qwen3-Embedding-0.6B @ 256d Matryoshka (GPU recommended, ~1.5GB)

The active tier comes from the ``TRUEMEMORY_EMBED_MODEL`` env var
(``edge`` / ``base`` / ``pro``) or ``~/.truememory/config.json``; MCP-server
callers may also invoke :func:`set_embedding_model` at runtime. Cosine-
distance nearest-neighbour search surfaces queries like "networking
problems" against stored "ECONNREFUSED" messages without keyword overlap.

Usage::

    from truememory.storage import create_db
    from truememory.vector_search import init_vec_table, build_vectors, search_vector

    conn = create_db("truememory.db")
    # ... insert messages ...
    init_vec_table(conn)
    build_vectors(conn)
    results = search_vector(conn, "networking problems", limit=5)

Dependencies (all included in ``pip install truememory``):
    - model2vec — edge tier embeddings
    - sentence-transformers — base / pro tier embeddings + reranker
    - sqlite-vec — vector search extension
    - numpy
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import struct
import sqlite3
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TrueMemoryMigrationError(Exception):
    """Raised when the stored DB was built with a different embedder than the
    currently-configured tier.

    Upgrading between tiers that use different embedding dimensions (e.g.
    v0.3.0 Pro @ 1024d → v0.4.0 Pro @ 256d) or different embedders at the
    same dimension (e.g. Model2Vec 256d → Qwen3 256d) requires re-embedding
    the stored messages. See the CHANGELOG for migration steps.
    """

# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

# Public tier names → internal model identifiers (v0.4.0 paper-aligned Edge/Base/Pro)
_TIER_ALIASES = {
    "edge": "model2vec",
    "base": "qwen3_256",
    "pro": "qwen3_256",
}

_MODEL_DIMS = {
    "model2vec": 256,
    "minilm": 384,
    "bge-small": 384,
    "qwen3_256": 256,
}

# v0.4.0 breaking change: the old internal name "qwen3" (1024d native) is gone.
# Callers must migrate to "pro" (tier alias) or "qwen3_256" (internal name).
_REMOVED_MODELS = {"qwen3"}


def _resolve_model_name(name: str) -> str:
    """Resolve a public tier name (edge/base/pro) or internal model name.

    Raises:
        ValueError: if *name* refers to a model removed in v0.4.0.
    """
    lowered = name.lower()
    if lowered in _REMOVED_MODELS:
        raise ValueError(
            f"Embedding model {name!r} was removed in TrueMemory v0.4.0. "
            f"Migrate to 'pro' (tier alias) or 'qwen3_256' (internal name) — "
            f"the paper-aligned Qwen3-Embedding-0.6B @ 256d Matryoshka config."
        )
    return _TIER_ALIASES.get(lowered, name)


_raw_env = os.environ.get("TRUEMEMORY_EMBED_MODEL", "edge")
EMBEDDING_MODEL = _resolve_model_name(_raw_env)

_model = None
_embedding_dim: int = _MODEL_DIMS.get(EMBEDDING_MODEL, 256)
_lock = threading.Lock()

_MODEL_TO_GROUP = {
    "model2vec": "edge",
    "qwen3_256": "basepro",
    "minilm": "basepro",
    "bge-small": "basepro",
}
_VALID_GROUPS = frozenset({"edge", "basepro"})


def _active_tier_group() -> str:
    """Map the current embedding model to its tier group."""
    return _MODEL_TO_GROUP.get(EMBEDDING_MODEL, "basepro")


def _active_vec_table(conn: sqlite3.Connection) -> str:
    """Return the active vec_messages table name for the current tier."""
    try:
        group = _active_tier_group()
        row = conn.execute(
            "SELECT vec_table FROM vector_cache_registry WHERE tier_group = ?",
            (group,),
        ).fetchone()
        if row:
            return row[0]
    except sqlite3.OperationalError:
        pass
    return "vec_messages"


def _active_sep_table(conn: sqlite3.Connection) -> str:
    """Return the active vec_messages_sep table name for the current tier."""
    try:
        group = _active_tier_group()
        row = conn.execute(
            "SELECT sep_table FROM vector_cache_registry WHERE tier_group = ?",
            (group,),
        ).fetchone()
        if row:
            return row[0]
    except sqlite3.OperationalError:
        pass
    return "vec_messages_sep"


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


def unload_model() -> None:
    """Release the embedding model from memory."""
    global _model
    with _lock:
        _model = None


def get_model():
    """Lazy-load the embedding model (singleton).

    When the shared model server is enabled (default), returns a proxy
    that routes inference to the server process. Falls back to local
    loading if the server is unavailable.
    """
    global _model, _embedding_dim
    if _model is not None:
        return _model  # Fast path, no lock needed
    with _lock:
        if _model is not None:
            return _model  # Another thread loaded it

        from truememory.model_client import use_model_server, get_embedding_proxy
        if use_model_server():
            try:
                proxy = get_embedding_proxy(tier=EMBEDDING_MODEL)
                _model = proxy
                return _model
            except Exception:
                pass  # Fall through to local loading

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
        elif resolved == "qwen3_256":
            from sentence_transformers import SentenceTransformer
            import sys as _sys
            _mkwargs = {}
            if _sys.platform == "darwin":
                _mkwargs["attn_implementation"] = "eager"
            _model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                truncate_dim=256,
                model_kwargs=_mkwargs or None,
            )
            _embedding_dim = 256
        else:
            from model2vec import StaticModel
            _model = StaticModel.from_pretrained("minishlab/potion-base-8M", force_download=False)
            _embedding_dim = 256
    return _model


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Embedder-identity metadata # ---------------------------------------------------------------------------


def _ensure_metadata_table(conn: sqlite3.Connection) -> None:
    """Idempotently create the key/value metadata table.

    storage._SCHEMA_SQL also declares this table, but engine.open() connects
    to existing v0.3.0 DBs without running that script, so a runtime-safe
    CREATE IF NOT EXISTS is needed before any metadata read/write.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metadata ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT"
        ")"
    )


def _write_embedder_metadata(conn: sqlite3.Connection) -> None:
    """Record `(embed_model, embed_dim)` so later opens can detect drift."""
    _ensure_metadata_table(conn)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value, updated_at) VALUES (?, ?, ?)",
        ("embed_model", EMBEDDING_MODEL, now),
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value, updated_at) VALUES (?, ?, ?)",
        ("embed_dim", str(_embedding_dim), now),
    )
    conn.commit()


def _read_embedder_metadata(
    conn: sqlite3.Connection,
) -> tuple[str | None, int | None]:
    """Return `(stored_model, stored_dim)` or `(None, None)` if not recorded."""
    _ensure_metadata_table(conn)
    rows = conn.execute(
        "SELECT key, value FROM metadata WHERE key IN ('embed_model', 'embed_dim')"
    ).fetchall()
    stored = {r[0]: r[1] for r in rows}
    model = stored.get("embed_model")
    dim_str = stored.get("embed_dim")
    try:
        dim = int(dim_str) if dim_str else None
    except ValueError:
        dim = None
    return model, dim


def _detect_existing_vec_dim(
    conn: sqlite3.Connection, table_name: str | None = None
) -> int | None:
    """Parse the dimension declared in an existing vector table schema."""
    name = table_name or _active_vec_table(conn)
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone()
    if not row or not row[0]:
        if name != "vec_messages":
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='vec_messages'"
            ).fetchone()
            if not row or not row[0]:
                return None
        else:
            return None
    match = re.search(r"float\[(\d+)\]", row[0])
    return int(match.group(1)) if match else None


def _migration_hint() -> str:
    return (
        "Re-embed via `truememory_configure(tier=...)` (re-encodes existing "
        "memories with the new model) or delete the DB (e.g. "
        "`~/.truememory/memories.db`) to start fresh."
    )


def _check_embedder_compatibility(conn: sqlite3.Connection) -> None:
    """Guard against silent dim- or model-mismatch when a vec table exists.

    Called from :func:`init_vec_table`. Skips when no vector table is found
    so that explicit re-embed flows (truememory_configure dropping + rebuilding)
    aren't blocked by stale metadata from the previous embedder.
    """
    existing_dim = _detect_existing_vec_dim(conn)
    if existing_dim is None:
        return  # fresh DB or dropped-for-re-embed — nothing to protect

    stored_model, _stored_dim = _read_embedder_metadata(conn)
    current_dim = _embedding_dim

    if existing_dim != current_dim:
        stored_hint = (
            f" (stored embed_model={stored_model!r})"
            if stored_model
            else " (no metadata — likely a legacy v0.3.0 DB)"
        )
        raise TrueMemoryMigrationError(
            f"Database has a {existing_dim}d vec_messages table but the "
            f"current embedder produces {current_dim}d vectors{stored_hint}. "
            f"This commonly happens when upgrading between tiers with "
            f"different embedding dimensions (e.g. v0.3.0 Pro @ 1024d → "
            f"v0.4.0 Pro @ 256d). " + _migration_hint()
        )

    if stored_model is not None and stored_model != EMBEDDING_MODEL:
        raise TrueMemoryMigrationError(
            f"Database was built with embed_model={stored_model!r}; current "
            f"is {EMBEDDING_MODEL!r}. Matching dims ({current_dim}d) would "
            f"otherwise mask a silent vector-space mismatch. " + _migration_hint()
        )

    if stored_model is None:
        logger.warning(
            "vec_messages exists without embedder metadata (legacy v0.3.0-style "
            "DB). Current embedder=%r at %dd. If you have switched embedding "
            "models since ingestion, re-embed via truememory_configure() — "
            "otherwise new vectors will carry the %r marker going forward.",
            EMBEDDING_MODEL, current_dim, EMBEDDING_MODEL,
        )


def _check_rebuild_allowed(conn: sqlite3.Connection) -> None:
    """Refuse a silent auto-rebuild if metadata names a different embedder.

    Called from :func:`TrueMemoryEngine.open` when `vec_messages` is missing
    and `rebuild_vectors=True`. The intent of that path is bootstrap — but if
    the DB has metadata, an implicit rebuild with the current (possibly
    different) model would silently re-encode against a different vector
    space. Force the user to route through `truememory_configure`.
    """
    stored_model, _ = _read_embedder_metadata(conn)
    if stored_model is not None and stored_model != EMBEDDING_MODEL:
        raise TrueMemoryMigrationError(
            f"Refusing silent auto-rebuild: DB metadata says "
            f"embed_model={stored_model!r} but current is {EMBEDDING_MODEL!r}. "
            f"Call truememory_configure() to re-embed explicitly, or delete "
            f"the DB to start fresh."
        )


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

def init_vec_table(
    conn: sqlite3.Connection, *, tier_group: str | None = None
) -> None:
    """
    Initialize the sqlite-vec extension and create both vector tables.

    Creates two virtual tables for embeddings keyed by ``rowid``
    matching ``messages.id``. When *tier_group* is provided the tables
    are named ``vec_messages_{tier_group}`` / ``vec_messages_sep_{tier_group}``
    for the tier-switch cache system; otherwise the active table names
    are resolved from the vector cache registry (falling back to the
    generic ``vec_messages`` / ``vec_messages_sep``).

    This function is idempotent.

    Args:
        conn: An open SQLite connection (from :func:`truememory.storage.create_db`).
        tier_group: Optional explicit tier group name ("edge" or "basepro").
    """
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    _ensure_metadata_table(conn)

    if tier_group:
        if tier_group not in _VALID_GROUPS:
            raise ValueError(f"Invalid tier_group: {tier_group!r}")
        vec_name = f"vec_messages_{tier_group}"
        sep_name = f"vec_messages_sep_{tier_group}"
    else:
        _check_embedder_compatibility(conn)
        vec_name = _active_vec_table(conn)
        sep_name = _active_sep_table(conn)

    dim = _embedding_dim
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {vec_name} "
        f"USING vec0(embedding float[{dim}])"
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {sep_name} "
        f"USING vec0(embedding float[{dim}])"
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Batch embedding & storage
# ---------------------------------------------------------------------------

_BATCH_SIZE_CPU = 100
_BATCH_SIZE_MPS = int(os.environ.get("TRUEMEMORY_MPS_BATCH_SIZE", "16"))


def _get_batch_size() -> int:
    """Return batch size appropriate for the current device."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return _BATCH_SIZE_MPS
    except Exception:
        pass
    return _BATCH_SIZE_CPU


def _flush_mps_cache() -> None:
    """Release MPS GPU memory after each batch to prevent accumulation."""
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
    except Exception:
        pass
    import gc
    gc.collect()


def build_vectors(
    conn: sqlite3.Connection,
    messages: list[dict] | None = None,
    *,
    table_name: str | None = None,
) -> int:
    """
    Embed messages and store their vectors in a vector table.

    If *messages* is ``None`` the function reads every row from the
    ``messages`` table.  Otherwise it uses the supplied list (each dict must
    have an ``"id"`` and ``"content"`` key).

    Batch size adapts to the device: 100 on CPU (Edge tier), 16 on MPS
    (Base/Pro tier) to keep GPU memory under ~4GB on 8GB machines.
    MPS cache is flushed between batches to prevent memory accumulation.

    Args:
        conn:       Open database connection with sqlite-vec already loaded
                    (call :func:`init_vec_table` first).
        messages:   Optional pre-fetched list of message dicts.
        table_name: Target vector table. Defaults to the active table for
                    the current tier group.

    Returns:
        Number of vectors inserted.
    """
    if messages is None:
        rows = conn.execute(
            "SELECT id, content FROM messages ORDER BY id"
        ).fetchall()
        messages = [{"id": row[0], "content": row[1]} for row in rows]

    if not messages:
        return 0

    tbl = table_name or _active_vec_table(conn)
    conn.execute(f"DELETE FROM {tbl}")

    model = get_model()
    batch_size = _get_batch_size()
    total = 0

    try:
        import torch
        no_grad = torch.no_grad()
    except ImportError:
        from contextlib import nullcontext
        no_grad = nullcontext()

    with no_grad:
        for start in range(0, len(messages), batch_size):
            batch = messages[start : start + batch_size]
            texts = [m["content"] for m in batch]
            ids = [m["id"] for m in batch]

            embeddings = model.encode(texts, show_progress_bar=False)

            conn.executemany(
                f"INSERT INTO {tbl}(rowid, embedding) VALUES (?, ?)",
                [(mid, serialize_f32(emb)) for mid, emb in zip(ids, embeddings)],
            )

            total += len(batch)
            del embeddings
            _flush_mps_cache()

    conn.commit()
    _write_embedder_metadata(conn)
    return total


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def search_vector(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    _query_blob: bytes | None = None,
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
        _query_blob: Pre-computed serialized embedding (skip encoding if given).

    Returns:
        List of result dicts sorted by descending similarity, each containing:
        ``id``, ``content``, ``sender``, ``recipient``, ``timestamp``,
        ``category``, ``modality``, ``score``.
    """
    limit = max(1, min(limit, 4096))

    if _query_blob is None:
        model = get_model()
        query_embedding = model.encode([query])[0]
        _query_blob = serialize_f32(query_embedding)

    query_blob = _query_blob

    tbl = _active_vec_table(conn)
    rows = conn.execute(
        f"""
        SELECT v.rowid, v.distance,
               m.content, m.sender, m.recipient,
               m.timestamp, m.category, m.modality
        FROM (
            SELECT rowid, distance
            FROM {tbl}
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


def search_vector_raw(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """Search by vector similarity, returning cosine similarity scores.

    Unlike :func:`search_vector` which returns ``1/(1+distance)``, this
    function converts sqlite-vec's cosine distance to cosine similarity:
    ``cos_sim = max(0, 1 - distance)``. This is used by the encoding
    gate where the paper equation (1) requires ``n_t = 1 - cos_sim``.
    """
    model = get_model()
    query_embedding = model.encode([query])[0]
    query_blob = serialize_f32(query_embedding)

    tbl = _active_vec_table(conn)
    rows = conn.execute(
        f"""
        SELECT v.rowid, v.distance,
               m.content, m.sender, m.recipient,
               m.timestamp, m.category, m.modality
        FROM (
            SELECT rowid, distance
            FROM {tbl}
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
        cos_sim = max(0.0, min(1.0, 1.0 - distance))

        results.append(
            {
                "id": row[0],
                "content": row[2],
                "sender": row[3],
                "recipient": row[4],
                "timestamp": row[5],
                "category": row[6],
                "modality": row[7],
                "score": round(cos_sim, 6),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Separation embeddings (B2: dual embedding support)
# ---------------------------------------------------------------------------

def build_separation_vectors(
    conn: sqlite3.Connection,
    messages: list[dict] | None = None,
    *,
    table_name: str | None = None,
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
        conn:       Open database connection with sqlite-vec loaded
                    (call :func:`init_vec_table` first).
        messages:   Optional pre-fetched list of message dicts.
        table_name: Target separation vector table. Defaults to the active
                    table for the current tier group.

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

    tbl = table_name or _active_sep_table(conn)
    conn.execute(f"DELETE FROM {tbl}")

    model = get_model()
    batch_size = _get_batch_size()
    total = 0

    try:
        import torch
        no_grad = torch.no_grad()
    except ImportError:
        from contextlib import nullcontext
        no_grad = nullcontext()

    with no_grad:
        for start in range(0, len(messages), batch_size):
            batch = messages[start : start + batch_size]

            texts = []
            ids = []
            for m in batch:
                sep_text = _build_sep_text(
                    m.get("sender", "?"), m.get("recipient", "?"),
                    m.get("timestamp", "?"), m["content"],
                )
                texts.append(sep_text)
                ids.append(m["id"])

            embeddings = model.encode(texts, show_progress_bar=False)

            conn.executemany(
                f"INSERT INTO {tbl}(rowid, embedding) VALUES (?, ?)",
                [(mid, serialize_f32(emb)) for mid, emb in zip(ids, embeddings)],
            )

            total += len(batch)
            del embeddings
            _flush_mps_cache()

    conn.commit()
    _write_embedder_metadata(conn)
    return total


def _build_sep_text(sender: str, recipient: str, timestamp: str, content: str) -> str:
    return (
        f"{sender or '?'} to {recipient or '?'} "
        f"on {(timestamp or '?')[:10]}: {content}"
    )


def embed_single(conn: sqlite3.Connection, message_id: int, content: str) -> None:
    """
    Embed a single message and insert into ``vec_messages`` and ``vec_messages_sep``.

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
    vec_tbl = _active_vec_table(conn)
    conn.execute(
        f"INSERT INTO {vec_tbl}(rowid, embedding) VALUES (?, ?)",
        (message_id, serialize_f32(embedding)),
    )

    try:
        row = conn.execute(
            "SELECT sender, recipient, timestamp FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        if row:
            sep_text = _build_sep_text(row[0], row[1], row[2], content)
            sep_embedding = model.encode([sep_text])[0]
            sep_tbl = _active_sep_table(conn)
            conn.execute(
                f"INSERT INTO {sep_tbl}(rowid, embedding) VALUES (?, ?)",
                (message_id, serialize_f32(sep_embedding)),
            )
    except Exception:
        # WARNING (not DEBUG): a missing separation vector means every future
        # sender-aware search will silently fail to find this message. The
        # row is in messages + vec_messages but invisible to the separation
        # pipeline — silent search-quality degradation per stored memory.
        logger.warning(
            "Failed to create separation vector for message %d — "
            "sender-aware search will not surface this row",
            message_id, exc_info=True,
        )
    # Caller is responsible for committing


def search_vector_separation(
    conn: sqlite3.Connection,
    query: str,
    sender: str | None = None,
    limit: int = 10,
    _query_blob: bytes | None = None,
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
        _query_blob: Pre-computed serialized embedding (skip encoding if given).
                     Ignored when *sender* is set (different query text).

    Returns:
        List of result dicts sorted by descending similarity.
    """
    limit = max(1, min(limit, 4096))

    if sender:
        model = get_model()
        query_text = f"{sender}: {query}"
        query_embedding = model.encode([query_text])[0]
        query_blob = serialize_f32(query_embedding)
    elif _query_blob is not None:
        query_blob = _query_blob
    else:
        model = get_model()
        query_embedding = model.encode([query])[0]
        query_blob = serialize_f32(query_embedding)

    sep_tbl = _active_sep_table(conn)
    rows = conn.execute(
        f"""
        SELECT v.rowid, v.distance,
               m.content, m.sender, m.recipient,
               m.timestamp, m.category, m.modality
        FROM (
            SELECT rowid, distance
            FROM {sep_tbl}
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


# ---------------------------------------------------------------------------
# Legacy migration (generic → tier-specific table names)
# ---------------------------------------------------------------------------

def migrate_legacy_vec_tables(conn: sqlite3.Connection) -> bool:
    """One-time migration: copy generic vec tables to tier-specific names.

    Detects whether the old generic ``vec_messages`` / ``vec_messages_sep``
    tables exist and copies their data into the tier-specific tables for
    the correct tier group (determined from stored metadata, defaulting to
    edge since pre-tier-switch versions used Model2Vec).  Populates the
    ``vector_cache_registry`` so future operations use the new names.

    Always registers the edge group for legacy tables so switch-back works.

    Returns True if migration occurred, False if nothing to migrate.
    """
    from truememory.tier_switch.cache import VectorCacheRegistry

    has_old = conn.execute(
        "SELECT name FROM sqlite_master WHERE name='vec_messages' AND type='table'"
    ).fetchone()
    if not has_old:
        return False

    stored_model, _ = _read_embedder_metadata(conn)
    group = _MODEL_TO_GROUP.get(stored_model, "edge")
    if group not in _VALID_GROUPS:
        group = "edge"

    new_vec = f"vec_messages_{group}"
    new_sep = f"vec_messages_sep_{group}"

    count = conn.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0]
    if count == 0:
        conn.execute("DROP TABLE IF EXISTS vec_messages")
        conn.execute("DROP TABLE IF EXISTS vec_messages_sep")
        conn.commit()
        return True

    dim = _detect_existing_vec_dim(conn, "vec_messages") or 256

    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {new_vec} "
        f"USING vec0(embedding float[{dim}])"
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {new_sep} "
        f"USING vec0(embedding float[{dim}])"
    )

    conn.execute(f"INSERT INTO {new_vec} SELECT * FROM vec_messages")
    try:
        conn.execute(f"INSERT INTO {new_sep} SELECT * FROM vec_messages_sep")
    except sqlite3.OperationalError:
        pass

    conn.execute("DROP TABLE vec_messages")
    try:
        conn.execute("DROP TABLE vec_messages_sep")
    except sqlite3.OperationalError:
        pass

    max_id = conn.execute(
        f"SELECT MAX(rowid) FROM {new_vec}"
    ).fetchone()[0] or 0

    model_map = {"edge": "potion-base-8M", "basepro": "Qwen3-Embedding-0.6B"}
    VectorCacheRegistry.set(
        conn,
        group,
        vec_table=new_vec,
        sep_table=new_sep,
        last_embedded_id=max_id,
        vector_count=count,
        model_name=model_map.get(group, "potion-base-8M"),
        embedding_dim=dim,
    )

    logger.info(
        "Migrated %d vectors from vec_messages → %s (group=%s)",
        count, new_vec, group,
    )
    return True
