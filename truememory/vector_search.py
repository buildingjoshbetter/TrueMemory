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
import math
import os
import re
import struct
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from truememory._platform import _env_int
from truememory.storage import _deserialize_metadata, select_message_cols

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

# Centralized tier config -- single source of truth lives in tier_config.py.
# Legacy symbols (_TIER_ALIASES, _MODEL_DIMS, _MODEL_TO_GROUP) are kept as
# thin delegates so that existing callers (model_server.py etc.) don't break.
from truememory.tier_config import (
    TIERS as _TIERS,
    MODEL_DIMS as _MODEL_DIMS,
    MODEL_TO_GROUP as _MODEL_TO_GROUP,
    VALID_TIER_GROUPS as _VALID_GROUPS,
    get_embed_model as _cfg_get_embed_model,
    get_embed_dim_for_model as _cfg_get_embed_dim_for_model,
    get_model_group as _cfg_get_model_group,
)

# Backward-compat alias -- model_server.py imports this symbol.
_TIER_ALIASES = {t: cfg["embed_model"] for t, cfg in _TIERS.items()}

# v0.4.0 breaking change: the old internal name "qwen3" (1024d native) is gone.
# Callers must migrate to "pro" (tier alias) or "qwen3_256" (internal name).
_REMOVED_MODELS = {"qwen3"}


def _resolve_model_name(name: str) -> str:
    """Resolve a public tier name (edge/base/pro/custom) or internal model name.

    Raises:
        ValueError: if *name* refers to a model removed in v0.4.0.
    """
    lowered = name.strip().lower()
    if lowered in _REMOVED_MODELS:
        raise ValueError(
            f"Embedding model {name!r} was removed in TrueMemory v0.4.0. "
            f"Migrate to 'pro' (tier alias) or 'qwen3_256' (internal name) -- "
            f"the paper-aligned Qwen3-Embedding-0.6B @ 256d Matryoshka config."
        )
    if lowered == "custom":
        try:
            return _cfg_get_embed_model("custom")
        except ValueError:
            logger.warning("Custom tier resolution failed; falling back to model2vec.")
            return "model2vec"
    # Known public tier (edge/base/pro, case-insensitively normalized e.g.
    # "PRO" -> "pro") or known internal model name -> resolve directly.
    if lowered in _TIER_ALIASES:
        return _TIER_ALIASES[lowered]
    if lowered in _MODEL_DIMS:
        return lowered
    # M-88 (#640): an unknown/garbage tier string must NOT be silently treated
    # as a custom HuggingFace model id (which, with CUSTOM_ALLOW_DOWNLOAD=1,
    # triggers an arbitrary model download). Only honour a value that clearly
    # looks like a HF repo id (contains "/"); otherwise fall back to a safe
    # default and log so the misconfiguration is visible.
    if "/" in lowered:
        return name
    logger.warning(
        "Unknown tier/model %r is not a known tier (edge/base/pro/custom) or "
        "model name and is not a HuggingFace repo id; falling back to "
        "model2vec (edge).",
        name,
    )
    return "model2vec"


def resolve_tier() -> str:
    """Resolve the active tier: env var -> config.json -> ``'edge'``.

    Canonical tier resolver for the entire codebase.  Every call site
    that formerly did ``os.environ.get("TRUEMEMORY_EMBED_MODEL", "edge")``
    should call this instead so that ``~/.truememory/config.json`` is
    honoured when the env var is absent.
    """
    env = os.environ.get("TRUEMEMORY_EMBED_MODEL", "").strip().lower()
    if env:
        return env
    try:
        import json
        from pathlib import Path
        _cfg = Path.home() / ".truememory" / "config.json"
        if _cfg.exists():
            tier = json.loads(_cfg.read_text(encoding="utf-8")).get("tier", "")
            if tier:
                return tier.strip().lower()
    except Exception:
        pass
    return "edge"

_raw_env = resolve_tier()
EMBEDDING_MODEL = _resolve_model_name(_raw_env)

_model = None
_embedding_dim: int = _MODEL_DIMS.get(EMBEDDING_MODEL, 256)
_lock = threading.Lock()
_model_generation = 0


def _active_tier_group() -> str:
    """Map the current embedding model to its tier group."""
    return _cfg_get_model_group(EMBEDDING_MODEL)


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
    global EMBEDDING_MODEL, _model, _embedding_dim, _model_generation
    with _lock:
        _model_generation += 1
        _model = None  # Force reload
        EMBEDDING_MODEL = _resolve_model_name(name)
        _embedding_dim = get_embedding_dim(EMBEDDING_MODEL)


def get_embedding_dim(name: str | None = None) -> int:
    """Return the embedding dimension for a given model name."""
    if name and name.lower().strip() == "custom":
        from truememory.tier_config import get_embed_dim
        return get_embed_dim("custom")
    name = _resolve_model_name(name) if name else EMBEDDING_MODEL
    return _cfg_get_embed_dim_for_model(name)


def unload_model() -> None:
    """Release the embedding model from memory."""
    global _model, _model_generation
    with _lock:
        _model_generation += 1
        _model = None


def get_model():
    """Lazy-load the embedding model (singleton).

    When the shared model server is enabled (default), returns a proxy
    that routes inference to the server process, including after idle exit.
    Local loading requires explicit TRUEMEMORY_NO_MODEL_SERVER=1.
    """
    global _model, _embedding_dim
    if _model is not None:
        return _model  # Fast path, no lock needed
    with _lock:
        if _model is not None:
            return _model  # Another thread loaded it

        from truememory.model_client import use_model_server, get_embedding_proxy
        if use_model_server():
            _model = get_embedding_proxy(tier=EMBEDDING_MODEL)
            return _model

        from truememory.mps_utils import ensure_mps_memory_budget, resolve_device

        resolved = EMBEDDING_MODEL
        if resolved == "model2vec":
            from model2vec import StaticModel
            _model = StaticModel.from_pretrained("minishlab/potion-base-8M", force_download=False)
            _embedding_dim = 256
        elif resolved == "minilm":
            device = resolve_device(None)
            ensure_mps_memory_budget(device)
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            _embedding_dim = 384
        elif resolved == "bge-small":
            device = resolve_device(None)
            ensure_mps_memory_budget(device)
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
            _embedding_dim = 384
        elif resolved == "qwen3_256":
            device = resolve_device(None)
            ensure_mps_memory_budget(device)
            from sentence_transformers import SentenceTransformer
            import sys as _sys
            _mkwargs = {}
            if _sys.platform == "darwin":
                _mkwargs["attn_implementation"] = "eager"
            _model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                truncate_dim=256,
                model_kwargs=_mkwargs or None,
                device=device,
            )
            _embedding_dim = 256
        elif resolved not in _MODEL_DIMS:
            # Custom tier: load arbitrary SentenceTransformer model.
            # Requires TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD=1 (enforced by
            # tier_config.resolve_custom_tier at config time).
            if os.environ.get("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "").strip() != "1":
                logger.warning(
                    "Custom model %r requested without "
                    "TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD=1 -- "
                    "falling back to model2vec.",
                    resolved,
                )
                from model2vec import StaticModel
                _model = StaticModel.from_pretrained(
                    "minishlab/potion-base-8M", force_download=False
                )
                _embedding_dim = 256
            else:
                from truememory.tier_config import resolve_custom_tier
                cfg = resolve_custom_tier()
                custom_dim = cfg["embed_dim"]
                device = resolve_device(None)
                ensure_mps_memory_budget(device)
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(
                    resolved, truncate_dim=custom_dim,
                    trust_remote_code=False,
                    device=device,
                )
                _embedding_dim = custom_dim
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


def _write_embedder_metadata_no_commit(conn: sqlite3.Connection) -> None:
    """Stage `(embed_model, embed_dim)` rows WITHOUT committing.

    Lets callers (build_vectors / build_separation_vectors) commit the embedder
    metadata in the same transaction as clearing the in-progress marker, so the
    table is never "trusted" under stale metadata (issue #647).
    """
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


def _write_embedder_metadata(conn: sqlite3.Connection) -> None:
    """Record `(embed_model, embed_dim)` so later opens can detect drift."""
    _write_embedder_metadata_no_commit(conn)
    conn.commit()


# ---------------------------------------------------------------------------
# Build-state marker (issue #647)
# ---------------------------------------------------------------------------
#
# A vec0 table that exists with rows looks "built" to engine.open() — but a run
# killed mid-embed (or one that hit a None content / NaN embedding) leaves a
# partial or empty table that must NOT be trusted. We persist an in-progress
# marker in the SAME transaction as the destructive DELETE and clear it only
# with the final commit (alongside the embedder metadata). Any table whose
# marker is still ``in_progress`` is treated as NOT built and rebuilt.
#
# This is deliberately separate from #631's ``*_cos_stage`` staging tables:
# that machinery makes the *L2→cosine re-norm* (no re-embed) crash-safe; this
# marker makes the *re-embed* (build_vectors / build_separation_vectors) crash-
# safe. They compose — a build leaves the marker; a cosine migration leaves a
# stage table; neither touches the other's state.

_BUILD_STATE_KEY_PREFIX = "vec_build_state:"


def _build_state_key(table_name: str) -> str:
    return f"{_BUILD_STATE_KEY_PREFIX}{table_name}"


def _mark_build_in_progress(conn: sqlite3.Connection, table_name: str) -> None:
    """Record that *table_name* is mid-build. Caller controls the commit so the
    marker lands in the same transaction as the DELETE that wipes the table."""
    _ensure_metadata_table(conn)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value, updated_at) VALUES (?, ?, ?)",
        (_build_state_key(table_name), "in_progress", now),
    )


def _clear_build_in_progress(conn: sqlite3.Connection, table_name: str) -> None:
    """Clear the in-progress marker for *table_name*. Caller controls commit."""
    _ensure_metadata_table(conn)
    conn.execute(
        "DELETE FROM metadata WHERE key = ?", (_build_state_key(table_name),)
    )


def _build_in_progress(conn: sqlite3.Connection, table_name: str) -> bool:
    """True if *table_name* has an uncleared in-progress build marker."""
    try:
        _ensure_metadata_table(conn)
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (_build_state_key(table_name),),
        ).fetchone()
    except sqlite3.OperationalError:
        return False
    return bool(row) and row[0] == "in_progress"


def vectors_are_built(
    conn: sqlite3.Connection, table_name: str | None = None
) -> bool:
    """True if the active (or named) vec table is present, readable, and not
    mid-build.

    A table left ``in_progress`` by an interrupted build is treated as NOT
    built so engine.open() rebuilds it rather than trusting partial/empty data
    (issue #647, M-21). Also returns False if the table is missing/unreadable.
    """
    tbl = table_name or _active_vec_table(conn)
    if _build_in_progress(conn, tbl):
        return False
    try:
        conn.execute(f"SELECT 1 FROM {tbl} LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return False
    return True


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

    Raises ValueError if any element is NaN or Inf.
    """
    if isinstance(vector, np.ndarray):
        if not np.all(np.isfinite(vector)):
            raise ValueError("Embedding contains NaN or Inf values")
        vector = vector.tolist()
    else:
        if any(math.isnan(v) or math.isinf(v) for v in vector):
            raise ValueError("Embedding contains NaN or Inf values")
    return struct.pack(f"{len(vector)}f", *vector)


# ---------------------------------------------------------------------------
# Cosine-metric helpers (issue #631)
# ---------------------------------------------------------------------------

# sqlite-vec stores vectors with an L2 distance function by default. We declare
# ``distance_metric=cosine`` on every vec0 table so that ``v.distance`` is a
# true cosine distance and ``cos_sim = 1 - distance`` is numerically honest.
#
# NOTE (sqlite-vec 0.1.9 syntax): ``distance_metric`` is a *column-level*
# option (space-separated inside the column declaration), NOT a comma-separated
# table option. ``vec0(embedding float[256] distance_metric=cosine)`` is valid;
# ``vec0(embedding float[256], distance_metric=cosine)`` raises
# "Unknown table option".
_VEC_DISTANCE_METRIC = "cosine"


def _vec0_column_decl(dim: int) -> str:
    """Return the vec0 embedding-column declaration with the cosine metric."""
    return f"embedding float[{dim}] distance_metric={_VEC_DISTANCE_METRIC}"


def _table_uses_cosine(conn: sqlite3.Connection, table_name: str) -> bool:
    """True if *table_name* exists and its DDL declares the cosine metric.

    Used to detect old-format (L2-default) vec tables that need rebuilding.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ? AND type='table'",
        (table_name,),
    ).fetchone()
    if not row or not row[0]:
        return False
    return "distance_metric=cosine" in row[0].replace(" ", "").replace("'", "")


def _normalize_for_cosine(emb):
    """L2-normalize *emb* so cosine distance is meaningful, or return None.

    Cosine distance is only well-defined once vectors are unit-normalized.
    Edge-tier (potion) vectors already arrive unit-norm, but Base/Pro (Qwen3
    @ truncate_dim=256, Matryoshka) vectors do NOT — truncating a normalized
    1024d vector to 256d leaves ``||v|| ~= 0.55``. Normalizing at write time
    keeps cosine ordering correct across every tier.

    A zero (or non-finite) vector has undefined cosine and would poison the
    cosine table with NaN distances (sqlite-vec returns ``distance=NULL`` for
    such rows). Those are skipped by returning ``None`` (C2-8 guard).
    """
    arr = np.asarray(emb, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return None
    return arr / norm


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
            raise ValueError(
                f"Invalid tier_group: {tier_group!r}. "
                f"Valid: {sorted(_VALID_GROUPS)}"
            )
        vec_name = f"vec_messages_{tier_group}"
        sep_name = f"vec_messages_sep_{tier_group}"
    else:
        _check_embedder_compatibility(conn)
        vec_name = _active_vec_table(conn)
        sep_name = _active_sep_table(conn)

    dim = _embedding_dim
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {vec_name} "
        f"USING vec0({_vec0_column_decl(dim)})"
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {sep_name} "
        f"USING vec0({_vec0_column_decl(dim)})"
    )
    conn.commit()

    # Upgrade any pre-#631 L2-default tables to cosine in place. Idempotent:
    # a no-op once the active tables already declare distance_metric=cosine.
    migrate_to_cosine_metric(conn)


# ---------------------------------------------------------------------------
# Batch embedding & storage
# ---------------------------------------------------------------------------

_BATCH_SIZE_CPU = 100
_BATCH_SIZE_MPS = _env_int("TRUEMEMORY_MPS_BATCH_SIZE", 16, lo=1)


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
    from truememory.mps_utils import flush_mps_cache
    flush_mps_cache()


def _is_mps_oom(exc: Exception) -> bool:
    """Return True if the exception is an MPS out-of-memory error."""
    from truememory.mps_utils import is_mps_oom
    return is_mps_oom(exc)


def _encode_with_mps_fallback(model, texts, **kwargs):
    """Encode texts with MPS OOM fallback. Thread-safe device transitions."""
    from truememory.mps_utils import encode_with_mps_fallback
    return encode_with_mps_fallback(model, texts, **kwargs)


_BUILD_VECTORS_TXN_BATCH = 100
"""Number of embedding batches to accumulate before committing.

Each embedding batch is ``_get_batch_size()`` messages (100 on CPU, 16 on
MPS). Serialize a finite group before acquiring an owned DB writer transaction;
publish the group and consumed-input checkpoint in one commit.
"""


def _capture_rebuild_model() -> tuple[object, tuple[int, str, int]]:
    with _lock:
        before = (_model_generation, EMBEDDING_MODEL)
    model = get_model()
    with _lock:
        if before != (_model_generation, EMBEDDING_MODEL):
            raise RuntimeError("Embedding model changed while loading rebuild model")
        return model, (_model_generation, EMBEDDING_MODEL, _embedding_dim)


@contextmanager
def _rebuild_model_fence(identity: tuple[int, str, int]) -> Iterator[None]:
    # A writer must never wait for a model load that needs this state lock.
    if not _lock.acquire(blocking=False):
        raise RuntimeError("Embedding model is busy; retry rebuild publication")
    try:
        if identity != (_model_generation, EMBEDDING_MODEL, _embedding_dim):
            raise RuntimeError("Embedding model changed during rebuild")
        yield
    finally:
        _lock.release()


class VectorPublicationChanged(RuntimeError):
    """Precomputed vectors no longer match the active publication target."""


@contextmanager
def _foreground_model_fence(identity: tuple[int, str, int]) -> Iterator[None]:
    if not _lock.acquire(blocking=False):
        raise VectorPublicationChanged("Embedding model is busy; retry vector publication")
    try:
        if identity != (_model_generation, EMBEDDING_MODEL, _embedding_dim):
            raise VectorPublicationChanged("Embedding model changed; retry vector publication")
        yield
    finally:
        _lock.release()


def _validate_foreground_vector_target(
    conn: sqlite3.Connection, identity: tuple[int, str, int], message_id: int,
) -> tuple[str, str]:
    """Validate while the caller owns both the database writer and model fence."""
    from truememory.rebuild_source import RebuildSourceChanged, load_manifest, manifest_key

    tables = (_active_vec_table(conn), _active_sep_table(conn))
    for table in tables:
        manifest = load_manifest(conn, manifest_key((table,)))
        if manifest is not None and (manifest.model != identity[1] or manifest.dimension != identity[2]):
            raise VectorPublicationChanged("Vector target uses a different embedding model; reopen with the active tier")
        if not _build_in_progress(conn, table):
            continue
        if manifest is None:
            raise VectorPublicationChanged("Vector target is rebuilding without matching source identity")
        if manifest.source is None:
            # Explicit caller input has no database prefix certificate.
            raise VectorPublicationChanged("Vector target is rebuilding explicit input; retry after completion")
        high = manifest.source.high_id
        if high is None or message_id > high:
            continue
        try:
            manifest.source.check(conn)
        except RebuildSourceChanged:
            # A corrected source invalidates this generation. Publish the
            # correction normally; the rebuild cannot later certify it.
            continue
        raise VectorPublicationChanged("Vector row belongs to an active rebuild; retry after completion")
    return tables


@contextmanager
def _foreground_vector_publication(
    conn: sqlite3.Connection, identity: tuple[int, str, int], message_id: int,
) -> Iterator[tuple[str, str]]:
    from truememory.rebuild_source import rebuild_transaction

    with rebuild_transaction(conn, write=True, publication_fence=_foreground_model_fence(identity)):
        yield _validate_foreground_vector_target(conn, identity, message_id)


def _build_streamed_vectors(
    conn: sqlite3.Connection, messages: list[dict] | None, *, table_name: str | None,
    txn_batch: int | None, separation: bool,
) -> int:
    from truememory.maintenance import connection_database_path, maintenance_owner
    from truememory.rebuild_source import (
        RebuildManifest, RebuildSourceChanged, capture_source, input_fingerprint,
        ensure_rebuild_tracking, load_manifest, manifest_key, rebuild_transaction, save_manifest,
    )

    if messages is not None and not messages:
        return 0
    table = table_name or (_active_sep_table(conn) if separation else _active_vec_table(conn))
    quoted_table = '"' + table.replace('"', '""') + '"'
    key = manifest_key((table,))
    digest = input_fingerprint(messages, separation=separation) if messages is not None else None
    with maintenance_owner(connection_database_path(conn)):
        source_empty = False
        if messages is None:
            ensure_rebuild_tracking(conn)
            with closing(conn.execute("SELECT 1 FROM messages LIMIT 1")) as cursor:
                source_empty = cursor.fetchone() is None
        if source_empty:
            identity = (_model_generation, EMBEDDING_MODEL, _embedding_dim)
            with rebuild_transaction(conn, write=True, publication_fence=_rebuild_model_fence(identity)):
                empty_source = capture_source(conn)
                if empty_source.total == 0:
                    # An empty range needs no native model allocation. Recheck
                    # under the writer so an append cannot slip through clear.
                    conn.execute(f"DELETE FROM {quoted_table}")
                    _ensure_metadata_table(conn)
                    empty = RebuildManifest.new(identity[1], identity[2], (table,), source=empty_source)
                    empty = replace(empty, complete=True,
                                    schema_version=conn.execute("PRAGMA schema_version").fetchone()[0])
                    save_manifest(conn, key, empty)
                    _clear_build_in_progress(conn, table)
                    return 0
        model, identity = _capture_rebuild_model()
        with rebuild_transaction(conn, write=True, publication_fence=_rebuild_model_fence(identity)):
            manifest = load_manifest(conn, key)
            resumable = (manifest is not None and not manifest.complete
                         and manifest.model == identity[1] and manifest.dimension == identity[2]
                         and manifest.targets == (table,) and manifest.input_digest == digest
                         and manifest.schema_version == conn.execute("PRAGMA schema_version").fetchone()[0]
                         and (manifest.source is not None) == (messages is None))
            if resumable and manifest.source is not None:
                try:
                    manifest.source.check(conn)
                except RebuildSourceChanged:
                    resumable = False
            if not resumable:
                source = capture_source(conn) if messages is None else None
                manifest = RebuildManifest.new(identity[1], identity[2], (table,), source=source,
                                               input_digest=digest, total=len(messages) if messages is not None else 0)
                conn.execute(f"DELETE FROM {quoted_table}")
                _mark_build_in_progress(conn, table)
                manifest = replace(manifest, schema_version=conn.execute("PRAGMA schema_version").fetchone()[0])
                save_manifest(conn, key, manifest)

        batch_size = _get_batch_size()
        commit_every = max(1, txn_batch if txn_batch is not None else _BUILD_VECTORS_TXN_BATCH)
        total = 0
        try:
            import torch
            no_grad = torch.no_grad()
        except ImportError:
            from contextlib import nullcontext
            no_grad = nullcontext()

        with no_grad:
            while manifest.consumed < manifest.total:
                next_manifest = manifest
                rows_to_insert = []
                for _ in range(commit_every):
                    if next_manifest.consumed >= next_manifest.total:
                        break
                    if manifest.source is not None:
                        batch = manifest.source.page(conn, next_manifest.cursor, batch_size, include_metadata=separation)
                    else:
                        batch = messages[next_manifest.consumed:next_manifest.consumed + batch_size]
                    if not batch:
                        raise RebuildSourceChanged("Rebuild source ended before its captured row count")
                    ids = [message["id"] for message in batch]
                    texts = ([_build_sep_text(message.get("sender", "?"), message.get("recipient", "?"),
                                              message.get("timestamp", "?"), message["content"]) for message in batch]
                             if separation else [message["content"] for message in batch])
                    embeddings = _encode_with_mps_fallback(model, texts, show_progress_bar=False)
                    if len(embeddings) != len(ids):
                        raise ValueError("Embedding count does not match the rebuild batch")
                    valid = 0
                    for mid, embedding in zip(ids, embeddings):
                        normed = _normalize_for_cosine(embedding)
                        if normed is None:
                            logger.warning("Skipping undefined cosine embedding during rebuild")
                            continue
                        rows_to_insert.append((mid, serialize_f32(normed)))
                        valid += 1
                    next_manifest = next_manifest.advance(ids[-1], len(batch), valid)
                    del embeddings, batch, texts, ids
                    _flush_mps_cache()
                with rebuild_transaction(conn, write=True, publication_fence=_rebuild_model_fence(identity)):
                    manifest.check(conn, key)
                    if rows_to_insert:
                        conn.executemany(f"INSERT INTO {quoted_table}(rowid, embedding) VALUES (?, ?)", rows_to_insert)
                    save_manifest(conn, key, next_manifest)
                total += next_manifest.outputs - manifest.outputs
                manifest = next_manifest
                del rows_to_insert

        with rebuild_transaction(conn, write=True, publication_fence=_rebuild_model_fence(identity)):
            manifest.check(conn, key)
            save_manifest(conn, key, replace(manifest, complete=True))
            _clear_build_in_progress(conn, table)
            _write_embedder_metadata_no_commit(conn)
        return total


def build_vectors(
    conn: sqlite3.Connection,
    messages: list[dict] | None = None,
    *,
    table_name: str | None = None,
    txn_batch: int | None = None,
) -> int:
    """
    Embed messages and store their vectors in a vector table.

    If *messages* is ``None`` the function reads every row from the
    ``messages`` table.  Otherwise it uses the supplied list (each dict must
    have an ``"id"`` and ``"content"`` key).

    Database inputs are read in bounded ID pages. A source revision and model
    fence protect every publication. Resume uses consumed input, including
    skipped invalid vectors, rather than the maximum stored vector ID.

    Serialize *txn_batch* embedding batches (default 100) before acquiring an
    owned writer transaction. Caller-owned transactions remain caller-owned;
    their existing locks cannot be released by this function.

    Args:
        conn:       Open database connection with sqlite-vec already loaded
                    (call :func:`init_vec_table` first).
        messages:   Optional pre-fetched list of message dicts.
        table_name: Target vector table. Defaults to the active table for
                    the current tier group.
        txn_batch:  Number of embedding batches per commit.  Defaults to
                    ``_BUILD_VECTORS_TXN_BATCH``.

    Returns:
        Number of vectors inserted.
    """
    return _build_streamed_vectors(conn, messages, table_name=table_name, txn_batch=txn_batch,
                                   separation=False)



# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def search_vector(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    _query_blob: bytes | None = None,
    include_directives: bool = False,
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
        query_embedding = _encode_with_mps_fallback(model, [query])[0]
        _query_blob = serialize_f32(query_embedding)

    query_blob = _query_blob

    # Over-fetch when excluding directives so we still return enough results
    # after filtering.  sqlite-vec MATCH doesn't support WHERE on joined
    # columns, so we post-filter instead.
    fetch_limit = limit * 2 if not include_directives else limit

    tbl = _active_vec_table(conn)
    rows = conn.execute(
        f"""
        SELECT v.rowid, v.distance,
               {select_message_cols(conn, alias='m')}
        FROM (
            SELECT rowid, distance
            FROM {tbl}
            WHERE embedding MATCH ? AND k = ?
        ) v
        JOIN messages m ON m.id = v.rowid
        ORDER BY v.distance
        """,
        (query_blob, fetch_limit),
    ).fetchall()

    results: list[dict] = []
    for row in rows:
        # Exclude directives by default
        if not include_directives and row[9]:
            continue

        distance = row[1]
        score = 1.0 / (1.0 + distance)

        results.append(
            {
                "id": row[0],
                "content": row[3],
                "sender": row[4],
                "recipient": row[5],
                "timestamp": row[6],
                "category": row[7],
                "modality": row[8],
                "directive": bool(row[9]),
                "metadata": _deserialize_metadata(row[10]),
                "score": round(score, 6),
            }
        )

    return results[:limit]


def search_vector_raw(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 5,
    include_directives: bool = False,
) -> list[dict]:
    """Search by vector similarity, returning cosine similarity scores.

    Unlike :func:`search_vector` which returns ``1/(1+distance)``, this
    function converts sqlite-vec's cosine distance to cosine similarity:
    ``cos_sim = max(0, 1 - distance)``. This is used by the encoding
    gate where the paper equation (1) requires ``n_t = 1 - cos_sim``.
    """
    model = get_model()
    query_embedding = _encode_with_mps_fallback(model, [query])[0]
    query_blob = serialize_f32(query_embedding)

    fetch_limit = limit * 2 if not include_directives else limit

    tbl = _active_vec_table(conn)
    rows = conn.execute(
        f"""
        SELECT v.rowid, v.distance,
               {select_message_cols(conn, alias='m')}
        FROM (
            SELECT rowid, distance
            FROM {tbl}
            WHERE embedding MATCH ? AND k = ?
        ) v
        JOIN messages m ON m.id = v.rowid
        ORDER BY v.distance
        """,
        (query_blob, fetch_limit),
    ).fetchall()

    results: list[dict] = []
    for row in rows:
        if not include_directives and row[9]:
            continue

        distance = row[1]
        cos_sim = max(0.0, min(1.0, 1.0 - distance))

        results.append(
            {
                "id": row[0],
                "content": row[3],
                "sender": row[4],
                "recipient": row[5],
                "timestamp": row[6],
                "category": row[7],
                "modality": row[8],
                "directive": bool(row[9]),
                "metadata": _deserialize_metadata(row[10]),
                "score": round(cos_sim, 6),
            }
        )

    return results[:limit]


# ---------------------------------------------------------------------------
# Separation embeddings (B2: dual embedding support)
# ---------------------------------------------------------------------------

def build_separation_vectors(
    conn: sqlite3.Connection,
    messages: list[dict] | None = None,
    *,
    table_name: str | None = None,
    txn_batch: int | None = None,
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
    return _build_streamed_vectors(conn, messages, table_name=table_name, txn_batch=txn_batch,
                                   separation=True)


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
    model, identity = _capture_rebuild_model()
    embedding = _encode_with_mps_fallback(model, [content])[0]
    normed = _normalize_for_cosine(embedding)
    if normed is None:
        logger.warning("Skipping undefined cosine embedding during incremental publication")
        return
    completion = serialize_f32(normed)
    separation = None
    try:
        row = conn.execute(
            "SELECT sender, recipient, timestamp FROM messages WHERE id = ?", (message_id,),
        ).fetchone()
        if row:
            sep_text = _build_sep_text(row[0], row[1], row[2], content)
            sep_embedding = _encode_with_mps_fallback(model, [sep_text])[0]
            sep_normed = _normalize_for_cosine(sep_embedding)
            if sep_normed is not None:
                separation = serialize_f32(sep_normed)
    except Exception:
        logger.warning("Failed to prepare separation vector during incremental publication", exc_info=True)
    with _foreground_vector_publication(conn, identity, message_id) as (vec_tbl, sep_tbl):
        conn.execute(f"INSERT INTO {vec_tbl}(rowid, embedding) VALUES (?, ?)", (message_id, completion))
        if separation is not None:
            try:
                conn.execute(f"INSERT INTO {sep_tbl}(rowid, embedding) VALUES (?, ?)", (message_id, separation))
            except Exception:
                logger.warning("Failed to create separation vector during incremental publication", exc_info=True)
        _write_embedder_metadata_no_commit(conn)


def search_vector_separation(
    conn: sqlite3.Connection,
    query: str,
    sender: str | None = None,
    limit: int = 10,
    _query_blob: bytes | None = None,
    include_directives: bool = False,
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
        query_embedding = _encode_with_mps_fallback(model, [query_text])[0]
        query_blob = serialize_f32(query_embedding)
    elif _query_blob is not None:
        query_blob = _query_blob
    else:
        model = get_model()
        query_embedding = _encode_with_mps_fallback(model, [query])[0]
        query_blob = serialize_f32(query_embedding)

    fetch_limit = limit * 2 if not include_directives else limit

    sep_tbl = _active_sep_table(conn)
    rows = conn.execute(
        f"""
        SELECT v.rowid, v.distance,
               {select_message_cols(conn, alias='m')}
        FROM (
            SELECT rowid, distance
            FROM {sep_tbl}
            WHERE embedding MATCH ? AND k = ?
        ) v
        JOIN messages m ON m.id = v.rowid
        ORDER BY v.distance
        """,
        (query_blob, fetch_limit),
    ).fetchall()

    results: list[dict] = []
    for row in rows:
        if not include_directives and row[9]:
            continue

        distance = row[1]
        score = 1.0 / (1.0 + distance)
        results.append({
            "id": row[0],
            "content": row[3],
            "sender": row[4],
            "recipient": row[5],
            "timestamp": row[6],
            "category": row[7],
            "modality": row[8],
            "directive": bool(row[9]),
            "metadata": _deserialize_metadata(row[10]),
            "score": round(score, 6),
        })

    return results[:limit]


# ---------------------------------------------------------------------------
# Cosine-metric migration (issue #631)
# ---------------------------------------------------------------------------

# Every vec0 table that might hold user vectors. Tier-specific names cover the
# tier-switch cache; the generic names cover legacy / pre-tier-switch DBs.
_KNOWN_VEC_TABLES = (
    "vec_messages",
    "vec_messages_sep",
    "vec_messages_edge",
    "vec_messages_sep_edge",
    "vec_messages_basepro",
    "vec_messages_sep_basepro",
)


def _rebuild_table_as_cosine(conn: sqlite3.Connection, table_name: str) -> int:
    """Rebuild one L2-default vec0 table in place as a cosine table.

    Re-inserts the EXISTING stored vectors (L2-normalized so cosine is
    meaningful) — no re-embedding. Zero-norm / non-finite vectors are dropped
    so they can't poison the cosine table with NULL distances (C2-8).

    vec0 virtual tables cannot be renamed (``ALTER TABLE RENAME`` leaves their
    shadow ``*_rowids`` / ``*_chunks`` tables behind), so the rebuild recreates
    the table under its original name. Crash-safety comes from a *plain* SQLite
    staging table ``{table}_cos_stage`` that holds the normalized vectors before
    the vec0 table is dropped: if the process dies mid-rebuild, the next run
    detects the staging table and finishes the swap rather than losing data.

    Returns the number of vectors carried over.
    """
    dim = _detect_existing_vec_dim(conn, table_name)
    if dim is None:
        return 0

    stage = f"{table_name}_cos_stage"
    stage_done = f"{stage}_done"
    fmt = f"{dim}f"

    # Phase 1: copy normalized vectors into a durable plain staging table.
    # D1-2 crash-safety: a separate "done" marker row is written in the SAME
    # commit as the staged rows. An interrupted stage (crash mid-copy) leaves an
    # empty/partial stage with NO done marker, so resume can tell it apart from a
    # complete stage and re-stage from the still-intact original instead of
    # swapping an empty stage over it and losing every vector.
    conn.execute(f"DROP TABLE IF EXISTS {stage}")
    conn.execute(f"DROP TABLE IF EXISTS {stage_done}")
    conn.execute(
        f"CREATE TABLE {stage} (rowid INTEGER PRIMARY KEY, embedding BLOB)"
    )
    conn.execute(f"CREATE TABLE {stage_done} (done INTEGER PRIMARY KEY)")
    rows = conn.execute(
        f"SELECT rowid, embedding FROM {table_name}"
    ).fetchall()
    skipped = 0
    staged = 0
    for rowid, blob in rows:
        if blob is None or len(blob) != dim * 4:
            skipped += 1
            continue
        normed = _normalize_for_cosine(struct.unpack(fmt, blob))
        if normed is None:
            skipped += 1
            continue
        conn.execute(
            f"INSERT INTO {stage}(rowid, embedding) VALUES (?, ?)",
            (rowid, serialize_f32(normed)),
        )
        staged += 1
    # Mark staging complete in the SAME transaction as the staged rows, so the
    # marker is durable iff every row is.
    conn.execute(f"INSERT INTO {stage_done}(done) VALUES (1)")
    conn.commit()  # staged rows + completeness marker durable before we touch the original

    # Phase 2: replace the vec0 table with a cosine one and refill it.
    carried = _finish_cosine_swap(conn, table_name, dim)

    if skipped:
        logger.warning(
            "Cosine migration for %s dropped %d zero-norm/invalid vector(s)",
            table_name, skipped,
        )
    logger.info(
        "Rebuilt %s as cosine (%d vectors carried over)", table_name, carried
    )
    return carried


def _finish_cosine_swap(
    conn: sqlite3.Connection, table_name: str, dim: int
) -> int:
    """Drop the old vec0 table, recreate it as cosine, refill from staging.

    Assumes ``{table_name}_cos_stage`` exists and is populated. Idempotent /
    resumable: safe to call after a crash that left the staging table behind.
    """
    stage = f"{table_name}_cos_stage"
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(
        f"CREATE VIRTUAL TABLE {table_name} USING vec0({_vec0_column_decl(dim)})"
    )
    carried = 0
    staged_rows = conn.execute(
        f"SELECT rowid, embedding FROM {stage}"
    ).fetchall()
    if staged_rows:
        conn.executemany(
            f"INSERT INTO {table_name}(rowid, embedding) VALUES (?, ?)",
            staged_rows,
        )
        carried = len(staged_rows)
    conn.execute(f"DROP TABLE IF EXISTS {stage}")
    conn.execute(f"DROP TABLE IF EXISTS {stage}_done")
    return carried


def migrate_to_cosine_metric(conn: sqlite3.Connection) -> bool:
    """Upgrade any old-format (L2-default) vec0 tables to cosine in place.

    Detects tables whose DDL lacks ``distance_metric=cosine`` and rebuilds each
    one, re-inserting the existing stored vectors (L2-normalized). This is the
    #631 fix for existing user DBs: the vectors are unchanged, only the table's
    distance function and per-row norm change so that ``cos_sim = 1 - distance``
    becomes numerically honest.

    Idempotent: a no-op once every present table already declares cosine.
    Returns True if at least one table was rebuilt.
    """
    # Ensure the extension is loaded (callers from engine.open already load it,
    # but init_vec_table is the common entry and loads it just above).
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:
        # If the extension cannot load, there are no vec0 tables to migrate.
        return False

    migrated = False
    for table in _KNOWN_VEC_TABLES:
        stage = f"{table}_cos_stage"
        has_stage = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ? AND type='table'",
            (stage,),
        ).fetchone()
        if has_stage:
            # A prior run was interrupted between staging and swap. Only finish
            # the swap if the staging completed durably (the "done" marker is
            # present); otherwise the stage is empty/partial and the ORIGINAL
            # table is still intact, so drop the partial stage and re-stage from
            # scratch below rather than swapping an empty stage over the original
            # and losing every vector (D1-2).
            stage_done = f"{stage}_done"
            stage_complete = bool(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name = ? AND type='table'",
                    (stage_done,),
                ).fetchone()
                and conn.execute(f"SELECT 1 FROM {stage_done} LIMIT 1").fetchone()
            )
            if stage_complete:
                dim = (
                    _detect_existing_vec_dim(conn, table)
                    or _detect_existing_vec_dim(conn, stage)
                    or _embedding_dim
                )
                _finish_cosine_swap(conn, table, dim)
                conn.commit()
                migrated = True
                continue
            # Incomplete stage from an interrupted copy — discard it and fall
            # through to a fresh rebuild from the intact original table.
            conn.execute(f"DROP TABLE IF EXISTS {stage}")
            conn.execute(f"DROP TABLE IF EXISTS {stage_done}")
            conn.commit()

        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ? AND type='table'",
            (table,),
        ).fetchone()
        if not exists:
            continue
        if _table_uses_cosine(conn, table):
            continue
        _rebuild_table_as_cosine(conn, table)
        migrated = True

    if migrated:
        conn.commit()
    return migrated


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
        has_tiered = conn.execute(
            "SELECT name FROM sqlite_master WHERE name=? AND type='table'",
            (new_vec,),
        ).fetchone()
        if not has_tiered:
            return False
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
        f"USING vec0({_vec0_column_decl(dim)})"
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {new_sep} "
        f"USING vec0({_vec0_column_decl(dim)})"
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
