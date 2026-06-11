"""
TrueMemory Engine
===============

The main orchestrator for the TrueMemory 6-layer memory system.  Ties together
all retrieval layers into a single ingest/search pipeline with graceful
degradation — if any module is missing or fails, the engine falls back to
whatever layers are available.

Layers:
    L0  Personality Engram   — entity profiles, communication patterns, preferences
    L1  Working Memory       — deferred (not needed for benchmark)
    L2  Episodic             — FTS5 keyword search + temporal filtering
    L3  Semantic             — Model2Vec vector search + RRF hybrid fusion
    L4  Salience Guard       — noise filtering + entity boosting
    L5  Consolidation        — summaries, contradiction resolution, predictive coding

Design principles:
    1. Graceful degradation  — partial failures never block the whole pipeline.
    2. Measurable            — every step is timed so we can see which layers add value.
    3. Composable            — each layer can be toggled on/off for A/B testing.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import warnings
import time
import sqlite3
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────────
# Core modules (always available)
# ───────────────────────────────────────────────────────────────────────────
from truememory.storage import (
    DEFAULT_BUSY_TIMEOUT_MS,
    create_db, load_messages_from_file, get_message_count,
    insert_message, delete_message, update_message, get_message,
)
from truememory.fts_search import search_fts

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
# Table-name allowlist — validated before any f-string SQL interpolation
# to prevent table-name injection in delete_all / forget.
# ───────────────────────────────────────────────────────────────────────────

_ALLOWED_TABLES = frozenset({
    "messages", "vec_messages", "vec_messages_sep", "entity_profiles",
    "entity_style_vectors", "entity_relationships", "fact_timeline",
    "summaries", "episodes", "landmark_events", "causal_edges",
    "surprise_scores", "message_clusters", "cluster_centroids",
    "messages_fts",
    # Tier-group vector tables (added for tier-switch cache system)
    "vec_messages_edge", "vec_messages_sep_edge",
    "vec_messages_basepro", "vec_messages_sep_basepro",
})

_ALLOWED_COLUMNS = frozenset({
    "source_message_id", "cause_msg_id", "effect_msg_id", "message_id",
})

_SQLITE_IN_CHUNK = 500


def _resolve_vec_tables(conn: sqlite3.Connection) -> tuple[str, str]:
    """Resolve the active tier's (vec, sep) table names for delete/update.

    On tier-group-cache DBs the live tables are ``vec_messages_basepro`` /
    ``vec_messages_edge`` (etc.); the flat ``vec_messages`` name may not
    exist post-migration, so hardcoding it left vectors orphaned on
    delete/update.

    Delegates to ``_active_vec_table`` / ``_active_sep_table`` which
    return ``"vec_messages"`` / ``"vec_messages_sep"`` when the
    ``vector_cache_registry`` table does not exist (pre-migration DBs).

    Raises:
        ValueError: If the resolved table name is not in ``_ALLOWED_TABLES``.
        Any exception propagated from the underlying SQL lookups.

    Callers should catch exceptions and fall back to the legacy table
    names if this function fails unexpectedly.

    Returns:
        Tuple of ``(vec_table_name, sep_table_name)``.
    """
    from truememory.vector_search import _active_vec_table, _active_sep_table
    vec = _active_vec_table(conn)
    sep = _active_sep_table(conn)
    for name in (vec, sep):
        if name not in _ALLOWED_TABLES:
            raise ValueError(f"Resolved vector table name {name!r} is not in _ALLOWED_TABLES")
    return vec, sep


# All known tier-group vector table names. Used by delete_all full-wipe
# to ensure ALL tiers are cleared, not just the currently active one.
_ALL_VEC_TABLES = (
    "vec_messages", "vec_messages_sep",
    "vec_messages_edge", "vec_messages_sep_edge",
    "vec_messages_basepro", "vec_messages_sep_basepro",
)


def _delete_in_chunks(conn, table: str, col: str, ids: list[int], chunk_size: int = _SQLITE_IN_CHUNK) -> None:
    if not ids:
        return
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        conn.execute(f"DELETE FROM {table} WHERE {col} IN ({placeholders})", chunk)

# ───────────────────────────────────────────────────────────────────────────
# Optional modules — each import is wrapped so missing deps don't break
# the engine.  Capability flags track what's available at runtime.
# ───────────────────────────────────────────────────────────────────────────

_HAS_VECTOR = False
try:
    from truememory.vector_search import (
        init_vec_table,
        build_vectors,
        build_separation_vectors,
        TrueMemoryMigrationError,
        _check_rebuild_allowed,
        resolve_tier,
    )
    _HAS_VECTOR = True
except (ImportError, ModuleNotFoundError):
    pass


# module-level tracker for sqlite-vec load failures. On platforms
# without a sqlite-vec wheel (Linux ARM musl, some BSDs, some sandboxed
# runtimes) the extension load fails and search silently falls back to
# FTS5-only. truememory_stats.health (F07) surfaces this via
# get_vectors_load_error() so users can see the degradation.
_vectors_load_error: str | None = None


def get_vectors_load_error() -> str | None:
    """Return the last sqlite-vec load failure message, or None if healthy.

    Consumed by ``truememory_stats.health`` to surface the degraded-search
    state. Module-level because ``sqlite_vec.load`` state is process-wide.
    """
    return _vectors_load_error

_HAS_HYBRID = False
try:
    from truememory.hybrid import search_hybrid
    _HAS_HYBRID = True
except (ImportError, ModuleNotFoundError):
    pass

_HAS_TEMPORAL = False
try:
    from truememory.temporal import detect_temporal_intent, search_temporal, detect_episodes, detect_landmark_events
    _HAS_TEMPORAL = True
except (ImportError, ModuleNotFoundError):
    pass

_HAS_SALIENCE = False
try:
    from truememory.salience import apply_salience_guard
    _HAS_SALIENCE = True
except (ImportError, ModuleNotFoundError):
    pass

_HAS_PERSONALITY = False
try:
    from truememory.personality import (
        build_entity_profiles,
        extract_preferences,
        search_personality,
        PERSONALITY_ASPECTS,
        build_dunbar_hierarchy,
    )
    _HAS_PERSONALITY = True
except (ImportError, ModuleNotFoundError):
    pass

_HAS_CONSOLIDATION = False
try:
    from truememory.consolidation import (
        build_summaries,
        detect_contradictions,
        search_contradictions,
        search_consolidated,
        build_entity_summary_sheets,
        build_structured_facts,
    )
    _HAS_CONSOLIDATION = True
except (ImportError, ModuleNotFoundError):
    pass

_HAS_PREDICTIVE = False
try:
    from truememory.predictive import build_surprise_index
    _HAS_PREDICTIVE = True
except (ImportError, ModuleNotFoundError):
    pass

_HAS_QUERY_CLASSIFIER = False
try:
    from truememory.query_classifier import classify_query, get_search_mode
    _HAS_QUERY_CLASSIFIER = True
except (ImportError, ModuleNotFoundError):
    pass

_HAS_RERANKER = False
try:
    import truememory.reranker  # noqa: F401
    _HAS_RERANKER = True
except (ImportError, ModuleNotFoundError):
    pass

_HAS_HYDE = False
try:
    from truememory.hyde import hyde_search
    _HAS_HYDE = True
except (ImportError, ModuleNotFoundError):
    pass

_HAS_CLUSTERING = False
try:
    from truememory.clustering import cluster_messages, search_clustered
    _HAS_CLUSTERING = True
except (ImportError, ModuleNotFoundError):
    pass

_HAS_STYLE_VEC = False
try:
    from truememory.personality_style_vec import (
        build_entity_style_vectors,
        update_entity_style_vector_incremental as _update_style_vec,
    )
    _HAS_STYLE_VEC = True
except (ImportError, ModuleNotFoundError):
    pass


# ───────────────────────────────────────────────────────────────────────────
# Helper
# ───────────────────────────────────────────────────────────────────────────

def _has_personality_intent(query: str) -> bool:
    """
    Return True if the query is genuinely asking about personality, preferences,
    habits, communication style, or character — as opposed to factual recall.

    This prevents personality profile results (score=1.0) from dominating
    factual queries like "What is Jordan's half marathon time?"
    """
    lower = query.lower()

    # Strong personality signals — these phrases almost always indicate
    # a personality question regardless of other content.
    strong_signals = [
        "kind of person", "what type of person", "personality",
        "describe.*as a person", "communication style", "how does.*communicate",
        "how does.*text", "how does.*talk", "what are.*fears",
        "what are.*hobbies", "what does.*like to eat",
        "what does.*like to do", "favorite food", "favourite food",
        "daily routine", "morning routine", "night routine",
        "what are.*traits", "character traits",
        "what are.*values", "what matters to",
        "what are.*insecurities", "what worries",
        "what are.*habits", "how does.*greet",
        "what are.*interests", "free time",
    ]
    for signal in strong_signals:
        if re.search(signal, lower):
            return True

    # Check against personality aspect keywords — require at least 2 matches
    # to avoid false positives on single common words.
    if _HAS_PERSONALITY:
        total_keyword_hits = 0
        for aspect, config in PERSONALITY_ASPECTS.items():
            hits = sum(1 for kw in config["keywords"] if kw in lower)
            total_keyword_hits += hits
        if total_keyword_hits >= 2:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════
# TrueMemoryEngine
# ═══════════════════════════════════════════════════════════════════════════

class TrueMemoryEngine:
    """
    The TrueMemory 6-layer memory system.

    Layers:
        L0  Personality Engram  (entity profiles, communication patterns)
        L1  Working Memory      (deferred — not needed for benchmark)
        L2  Episodic            (FTS5 + temporal filtering)
        L3  Semantic            (vector search + RRF hybrid fusion)
        L4  Salience Guard      (noise filtering + entity boosting)
        L5  Consolidation       (summaries + contradiction resolution + predictive coding)
    """

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def __init__(self, db_path: str | Path = None,
                 alpha_surprise: float | None = None):
        """Initialize engine with an optional database path.

        If *db_path* is omitted the database defaults to
        ``<project_root>/truememory.db``.

        :param alpha_surprise: Optional override for the MEMORIST-L5
            surprise rerank boost coefficient. When set, takes priority
            over ``TRUEMEMORY_ALPHA_SURPRISE`` env var and the default
            of 0.2. Range [0, ~2.0]; Modal alpha sweep (2026-04-26)
            found α=0.2 is the empirical peak. See
            ``_working/memorist/l5_predictive/REPORT.md``.
        """
        self.db_path = Path(db_path) if db_path else Path(__file__).parent.parent / "truememory.db"
        self.conn: sqlite3.Connection | None = None
        self.ready = False
        self.stats: dict = {}
        self._write_lock = threading.Lock()
        self._init_lock = threading.Lock()

        # L5 surprise rerank boost coefficient. None = resolve from env
        # var / default at call-time via _get_alpha_surprise().
        self._alpha_surprise_override = alpha_surprise

        # Capability flags (set during ingest)
        self._has_vectors = False
        self._has_hybrid = _HAS_HYBRID
        self._has_temporal = _HAS_TEMPORAL
        self._has_salience = _HAS_SALIENCE
        self._has_personality = _HAS_PERSONALITY
        self._has_consolidation = _HAS_CONSOLIDATION
        self._has_predictive = _HAS_PREDICTIVE
        self._has_reranker = _HAS_RERANKER
        self._has_hyde = _HAS_HYDE
        self._has_clustering = _HAS_CLUSTERING
        self._has_style_vec = _HAS_STYLE_VEC

        # Coordination flag for background NaN re-embed thread (#485).
        self._nan_migration_in_progress = False

        # Auto-consolidation: run L5 consolidation every N adds (#498)
        self._adds_since_consolidation = 0
        self._auto_consolidate_threshold = int(
            os.environ.get("TRUEMEMORY_AUTO_CONSOLIDATE_EVERY", "25")
        )
        self._consolidation_thread: threading.Thread | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Auto-connect (production API)
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_connection(self) -> None:
        """Open (or create) the database and load extensions if needed.

        Called automatically by :meth:`add`, :meth:`search`, and other
        public methods so users never have to call ``ingest()`` or
        ``open()`` manually for simple CRUD workflows.
        """
        if self.conn is not None:
            return

        with self._init_lock:
            if self.conn is not None:
                return

            # Create parent directory if using a real path
            db_str = str(self.db_path)
            if db_str != ":memory:":
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self.conn = create_db(self.db_path)

            # Load sqlite-vec extension
            if _HAS_VECTOR:
                try:
                    import sqlite_vec
                    self.conn.enable_load_extension(True)
                    sqlite_vec.load(self.conn)
                    self.conn.enable_load_extension(False)
                    init_vec_table(self.conn)
                    try:
                        from truememory.vector_search import migrate_legacy_vec_tables
                        migrate_legacy_vec_tables(self.conn)
                    except Exception:
                        logger.debug("Legacy vec table migration skipped", exc_info=True)
                    self._has_vectors = True
                except Exception as exc:
                    global _vectors_load_error
                    _vectors_load_error = f"{type(exc).__name__}: {exc}"
                    logger.warning("Failed to load sqlite-vec — FTS-only mode: %s", exc)
                    self._has_vectors = False

            self._has_hybrid = _HAS_HYBRID and self._has_vectors

            # Qwen3 NaN fix: macOS SDPA kernel produces NaN embeddings.
            # Re-embed once for Base/Pro users on macOS.
            import sys as _sys
            if _sys.platform == "darwin" and self._has_vectors:
                try:
                    _embed_model = resolve_tier()
                    if _embed_model in ("base", "pro", "qwen3_256"):
                        _tables = {r[0] for r in self.conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()}
                        if "metadata" in _tables:
                            _row = self.conn.execute(
                                "SELECT value FROM metadata WHERE key = 'qwen3_nan_fix_applied'"
                            ).fetchone()
                            if _row is None:
                                _msg_count = self.conn.execute(
                                    "SELECT COUNT(*) FROM messages"
                                ).fetchone()[0] if "messages" in _tables else 0
                                if _msg_count > 0:
                                    def _bg_reembed(db_path):
                                        import sqlite3 as _sql
                                        _conn = None
                                        try:
                                            _conn = _sql.connect(str(db_path), check_same_thread=False)
                                            _conn.execute("PRAGMA journal_mode=WAL")
                                            _conn.execute("PRAGMA busy_timeout=%d" % DEFAULT_BUSY_TIMEOUT_MS)
                                            _conn.execute("PRAGMA synchronous=NORMAL")
                                            _conn.execute("PRAGMA foreign_keys=ON")
                                            # Issue #499: load sqlite-vec on the
                                            # background thread's connection so
                                            # vec virtual tables are available.
                                            import sqlite_vec as _sv
                                            _conn.enable_load_extension(True)
                                            _sv.load(_conn)
                                            _conn.enable_load_extension(False)
                                            from truememory.vector_search import (
                                                build_vectors as _bv,
                                                build_separation_vectors as _bsv,
                                                init_vec_table as _ivt,
                                            )
                                            _conn.execute("DROP TABLE IF EXISTS vec_messages")
                                            _conn.execute("DROP TABLE IF EXISTS vec_messages_sep")
                                            _conn.commit()
                                            _ivt(_conn)
                                            _bv(_conn)
                                            _bsv(_conn)
                                            # Issue #485: only set the flag AFTER
                                            # successful re-embed so a failed
                                            # thread allows retry on next init.
                                            _conn.execute(
                                                "INSERT OR REPLACE INTO metadata "
                                                "(key, value) VALUES (?, ?)",
                                                ("qwen3_nan_fix_applied", "1"),
                                            )
                                            _conn.commit()
                                            logger.warning(
                                                "Qwen3 NaN fix: re-embedded %d vectors "
                                                "in background", _msg_count,
                                            )
                                        except Exception:
                                            logger.warning(
                                                "Qwen3 NaN background migration failed — "
                                                "will retry on next startup",
                                                exc_info=True,
                                            )
                                        finally:
                                            if _conn is not None:
                                                try:
                                                    _conn.close()
                                                except Exception:
                                                    pass
                                    _db = self.db_path
                                    if str(_db) != ":memory:":
                                        self._nan_migration_in_progress = True
                                        def _bg_wrapper(db_path):
                                            try:
                                                _bg_reembed(db_path)
                                            finally:
                                                self._nan_migration_in_progress = False
                                        _t = threading.Thread(
                                            target=_bg_wrapper, args=(_db,),
                                            daemon=True,
                                        )
                                        _t.start()
                                        logger.info(
                                            "Qwen3 NaN fix: re-embedding %d vectors "
                                            "in background thread", _msg_count,
                                        )
                                    else:
                                        from truememory.vector_search import (
                                            build_vectors as _bv,
                                            build_separation_vectors as _bsv,
                                            init_vec_table as _ivt,
                                        )
                                        self.conn.execute("DROP TABLE IF EXISTS vec_messages")
                                        self.conn.execute("DROP TABLE IF EXISTS vec_messages_sep")
                                        self.conn.commit()
                                        _ivt(self.conn)
                                        _bv(self.conn)
                                        _bsv(self.conn)
                                        self.conn.execute(
                                            "INSERT OR REPLACE INTO metadata "
                                            "(key, value) VALUES (?, ?)",
                                            ("qwen3_nan_fix_applied", "1"),
                                        )
                                        self.conn.commit()
                                        logger.warning(
                                            "Qwen3 NaN fix: re-embedded all vectors",
                                        )
                                else:
                                    # No messages — just set the flag to skip
                                    # future checks.
                                    self.conn.execute(
                                        "INSERT OR REPLACE INTO metadata "
                                        "(key, value) VALUES (?, ?)",
                                        ("qwen3_nan_fix_applied", "1"),
                                    )
                                    self.conn.commit()
                except Exception:
                    logger.warning(
                        "Qwen3 NaN migration failed — run "
                        "'truememory-ingest upgrade-tier base --force' to fix manually",
                        exc_info=True,
                    )

            self.ready = True
            self._maybe_startup_consolidate()

    def _maybe_startup_consolidate(self) -> None:
        """Trigger background consolidation on startup if data looks stale."""
        if not self._has_consolidation or self.conn is None:
            return
        try:
            msg_count = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            if msg_count < self._auto_consolidate_threshold:
                return
            cluster_count = self.conn.execute(
                "SELECT COUNT(*) FROM message_clusters"
            ).fetchone()[0]
            if cluster_count == 0:
                self._consolidation_thread = threading.Thread(
                    target=self._bg_consolidate,
                    daemon=True,
                    name="startup-consolidate",
                )
                self._consolidation_thread.start()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    # Production CRUD API
    # ──────────────────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        sender: str = "",
        recipient: str = "",
        timestamp: str = "",
        category: str = "",
        metadata: dict | None = None,
        directive: bool = False,
    ) -> dict:
        """Store a single memory.

        Auto-connects to the database if not already connected. Unlike
        :meth:`ingest`, this appends to existing data.

        Args:
            content:   The memory text.
            sender:    Who said it (maps to ``user_id`` in the simple API).
            recipient: Who it was said to.
            timestamp: ISO-8601 timestamp string.
            category:  Optional grouping label.
            metadata:  Reserved for future use.

        Returns:
            Dict with ``id`` and the stored fields.
        """
        if not isinstance(content, str):
            raise TypeError(f"content must be a string, got {type(content).__name__}")

        self._ensure_connection()

        pre_embedding = None
        pre_sep_embedding = None
        if self._has_vectors:
            try:
                from truememory.vector_search import (
                    get_model, _encode_with_mps_fallback, _build_sep_text,
                )
                model = get_model()
                pre_embedding = _encode_with_mps_fallback(model, [content])[0]
                sep_text = _build_sep_text(sender, recipient, timestamp, content)
                pre_sep_embedding = _encode_with_mps_fallback(model, [sep_text])[0]
            except Exception:
                logger.debug("Failed to pre-compute embedding during add()", exc_info=True)

        pre_style_vec = None
        if self._has_style_vec and sender:
            try:
                from truememory.personality_style_vec import compute_style_vector
                pre_style_vec = compute_style_vector(content)
            except Exception:
                logger.debug("Failed to pre-compute style vector during add()", exc_info=True)

        with self._write_lock:
            msg = {
                "content": content,
                "sender": sender,
                "recipient": recipient,
                "timestamp": timestamp,
                "category": category,
                "modality": "",
                "directive": directive,
            }
            new_id = insert_message(self.conn, msg)

            if pre_embedding is not None:
                try:
                    from truememory.vector_search import (
                        serialize_f32, _active_vec_table, _active_sep_table,
                        _write_embedder_metadata,
                    )
                    vec_tbl = _active_vec_table(self.conn)
                    self.conn.execute(
                        f"INSERT INTO {vec_tbl}(rowid, embedding) VALUES (?, ?)",
                        (new_id, serialize_f32(pre_embedding)),
                    )
                    if pre_sep_embedding is not None:
                        sep_tbl = _active_sep_table(self.conn)
                        self.conn.execute(
                            f"INSERT INTO {sep_tbl}(rowid, embedding) VALUES (?, ?)",
                            (new_id, serialize_f32(pre_sep_embedding)),
                        )
                    _write_embedder_metadata(self.conn)
                except Exception:
                    logger.warning("Failed to store embedding for message %s during add()", new_id, exc_info=True)

            # Incrementally update entity profile.
            # Skip for directives (#637 M-08): directive content must never
            # pollute entity_profiles / style vectors, or it would surface via
            # the personality `profile` / `style_vec` supplement sources.
            if self._has_personality and sender and not directive:
                try:
                    from truememory.personality import update_entity_profile_incremental
                    update_entity_profile_incremental(self.conn, sender, content, recipient)
                except Exception:
                    logger.debug("Failed to update entity profile for %s during add()", sender, exc_info=True)

            # Store pre-computed style vector (DB write only, computation happened outside lock)
            if pre_style_vec is not None and not directive:
                try:
                    _update_style_vec(self.conn, sender, content, _pre_computed_vec=pre_style_vec)
                except Exception:
                    logger.debug("Failed to update style vector for %s during add()", sender, exc_info=True)

            self.conn.commit()

        self._maybe_auto_consolidate()

        return {
            "id": new_id,
            "content": content,
            "sender": sender,
            "recipient": recipient,
            "timestamp": timestamp,
            "category": category,
            "directive": directive,
        }

    def _maybe_auto_consolidate(self) -> None:
        """Trigger background consolidation after threshold adds."""
        self._adds_since_consolidation += 1
        if self._adds_since_consolidation < self._auto_consolidate_threshold:
            return
        if not self._has_consolidation:
            return
        if (self._consolidation_thread is not None
                and self._consolidation_thread.is_alive()):
            return
        self._adds_since_consolidation = 0
        self._consolidation_thread = threading.Thread(
            target=self._bg_consolidate,
            daemon=True,
            name="auto-consolidate",
        )
        self._consolidation_thread.start()

    def _bg_consolidate(self) -> None:
        """Run consolidation in a background thread with its own connection."""
        try:
            self.consolidate()
        except Exception:
            logger.debug("Auto-consolidation failed", exc_info=True)

    def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID.

        Returns True if deleted, False if not found.
        """
        self._ensure_connection()
        with self._write_lock:
            return delete_message(self.conn, memory_id)

    def delete_all(self, user_id: str | None = None) -> bool:
        """Delete all memories, optionally filtered by user.

        Handles deletion from ALL tables in the schema: messages,
        messages_fts, entity_profiles, fact_timeline, summaries,
        episodes, landmark_events, causal_edges, entity_relationships,
        surprise_scores, message_clusters, cluster_centroids,
        and vector tables (vec_messages, vec_messages_sep).

        Args:
            user_id: If provided, only delete this user's memories and
                     related data.  If None, deletes everything.

        Returns:
            True if any rows were deleted from messages.
        """
        if user_id is not None and not isinstance(user_id, str):
            raise TypeError(f"user_id must be a string or None, got {type(user_id).__name__}")
        if isinstance(user_id, str) and not user_id.strip():
            raise ValueError("user_id cannot be an empty string")

        self._ensure_connection()

        with self._write_lock:
            if user_id is not None:
                # Get message IDs and episode IDs for this user before deleting
                msg_ids = [
                    row[0] for row in self.conn.execute(
                        "SELECT id FROM messages WHERE sender = ?", (user_id,)
                    ).fetchall()
                ]
                episode_ids = list({
                    row[0] for row in self.conn.execute(
                        "SELECT DISTINCT episode_id FROM messages WHERE sender = ? AND episode_id IS NOT NULL",
                        (user_id,),
                    ).fetchall()
                })

                # Delete child rows BEFORE parent to avoid FK violations
                if msg_ids:
                    for table, col in [
                        ("fact_timeline", "source_message_id"),
                        ("landmark_events", "source_message_id"),
                        ("causal_edges", "cause_msg_id"),
                        ("causal_edges", "effect_msg_id"),
                        ("surprise_scores", "message_id"),
                        ("message_clusters", "message_id"),
                    ]:
                        if table not in _ALLOWED_TABLES:
                            raise ValueError(f"Invalid table name: {table}")
                        if col not in _ALLOWED_COLUMNS:
                            raise ValueError(f"Invalid column name: {col}")
                        try:
                            _delete_in_chunks(self.conn, table, col, msg_ids)
                        except Exception:
                            logger.warning("Failed to clean %s for user %s", table, user_id, exc_info=True)

                # Clean entity profile for this user (normalize to lowercase #467)
                try:
                    self.conn.execute(
                        "DELETE FROM entity_profiles WHERE entity = ?", (user_id.lower(),)
                    )
                except Exception:
                    logger.warning("Failed to clean entity_profiles for user %s", user_id, exc_info=True)

                # Clean entity style vectors for this user (normalize to lowercase #467)
                try:
                    self.conn.execute(
                        "DELETE FROM entity_style_vectors WHERE entity = ?", (user_id.lower(),)
                    )
                except Exception:
                    logger.warning("Failed to clean entity_style_vectors for user %s", user_id, exc_info=True)

                # Clean entity relationships involving this user (normalize to lowercase #467)
                try:
                    self.conn.execute(
                        "DELETE FROM entity_relationships WHERE entity_a = ? OR entity_b = ?",
                        (user_id.lower(), user_id.lower()),
                    )
                except Exception:
                    logger.warning("Failed to clean entity_relationships for user %s", user_id, exc_info=True)

                # Clean summaries scoped to this user
                try:
                    self.conn.execute(
                        "DELETE FROM summaries WHERE entity = ?", (user_id.lower(),)
                    )
                except Exception:
                    logger.warning("Failed to clean summaries for user %s", user_id, exc_info=True)

                if episode_ids:
                    try:
                        _delete_in_chunks(self.conn, "episodes", "id", episode_ids)
                    except Exception:
                        logger.warning("Failed to clean episodes for user %s", user_id, exc_info=True)

                # Clean vector tables for deleted message IDs
                if msg_ids:
                    try:
                        vec_tbl, sep_tbl = _resolve_vec_tables(self.conn)
                    except Exception:
                        logger.warning("_resolve_vec_tables failed in delete_all(user_id=%s); falling back to legacy table names", user_id, exc_info=True)
                        vec_tbl, sep_tbl = "vec_messages", "vec_messages_sep"
                    # dict.fromkeys deduplicates in case vec_tbl == sep_tbl (shouldn't happen, but defensive)
                    for vec_table in dict.fromkeys((vec_tbl, sep_tbl)):
                        try:
                            _delete_in_chunks(self.conn, vec_table, "rowid", msg_ids)
                        except Exception:
                            logger.warning("Failed to clean %s for user %s", vec_table, user_id, exc_info=True)

                # Remove orphaned cluster_centroids (clusters with no
                # remaining message_clusters rows after user deletion).
                try:
                    self.conn.execute(
                        "DELETE FROM cluster_centroids WHERE cluster_id NOT IN "
                        "(SELECT DISTINCT cluster_id FROM message_clusters)"
                    )
                except Exception:
                    logger.warning("Failed to clean cluster_centroids for user %s", user_id, exc_info=True)

                # Delete parent rows AFTER all child FK references are gone
                cursor = self.conn.execute(
                    "DELETE FROM messages WHERE sender = ?", (user_id,)
                )
                deleted = cursor.rowcount > 0

            else:
                # Full wipe — delete child tables BEFORE messages to avoid FK violations
                for table in (
                    "entity_profiles",
                    "entity_style_vectors",
                    "fact_timeline",
                    "summaries",
                    "episodes",
                    "landmark_events",
                    "causal_edges",
                    "entity_relationships",
                    "surprise_scores",
                    "message_clusters",
                    "cluster_centroids",
                ):
                    if table not in _ALLOWED_TABLES:
                        raise ValueError(f"Invalid table name: {table}")
                    try:
                        self.conn.execute(f"DELETE FROM {table}")
                    except Exception:
                        logger.warning("Failed to clear table %s during delete_all", table, exc_info=True)

                # Clear ALL known vector tables across all tiers.
                for vec_table in _ALL_VEC_TABLES:
                    try:
                        self.conn.execute(f"DELETE FROM {vec_table}")
                    except Exception:
                        logger.debug("Failed to clear %s during delete_all (table may not exist)", vec_table, exc_info=True)

                cursor = self.conn.execute("DELETE FROM messages")
                deleted = cursor.rowcount > 0

            # Rebuild FTS index
            try:
                self.conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
            except Exception:
                logger.warning("Failed to rebuild FTS index during delete_all", exc_info=True)

            self.conn.commit()
            return deleted

    def consolidate(self) -> dict[str, str]:
        """Run all consolidation layers (L0-L5) under the write lock.

        Returns timing stats for each step.
        """
        import time as _time

        self._ensure_connection()
        stats: dict[str, str] = {}

        try:
            from truememory.consolidation import (
                build_summaries,
                detect_contradictions,
                build_structured_facts,
            )
        except (ImportError, ModuleNotFoundError):
            stats["consolidation"] = "SKIPPED (module not available)"
            return stats

        try:
            from truememory.predictive import build_surprise_index
        except (ImportError, ModuleNotFoundError):
            build_surprise_index = None

        try:
            from truememory.temporal import detect_episodes, detect_landmark_events
        except (ImportError, ModuleNotFoundError):
            detect_episodes = None
            detect_landmark_events = None

        try:
            from truememory.personality import build_dunbar_hierarchy
        except (ImportError, ModuleNotFoundError):
            build_dunbar_hierarchy = None

        _cluster_messages = None
        if _HAS_CLUSTERING and self._has_vectors:
            try:
                from truememory.clustering import cluster_messages as _cm
                _cluster_messages = _cm
            except (ImportError, ModuleNotFoundError):
                pass

        _extract_preferences = None
        if _HAS_PERSONALITY:
            try:
                from truememory.personality import extract_preferences as _ep
                _extract_preferences = _ep
            except (ImportError, ModuleNotFoundError):
                pass

        with self._write_lock:
            if _cluster_messages:
                try:
                    t0 = _time.time()
                    n = _cluster_messages(self.conn)
                    stats["cluster_messages"] = f"{n} clusters in {_time.time() - t0:.3f}s"
                    self._has_clustering = True
                except Exception as exc:
                    stats["cluster_messages"] = f"ERROR: {exc}"

            if _extract_preferences:
                try:
                    t0 = _time.time()
                    _extract_preferences(self.conn)
                    stats["extract_preferences"] = f"{_time.time() - t0:.3f}s"
                except Exception as exc:
                    stats["extract_preferences"] = f"ERROR: {exc}"

            try:
                t0 = _time.time()
                build_summaries(self.conn)
                stats["build_summaries"] = f"{_time.time() - t0:.3f}s"
            except Exception as exc:
                stats["build_summaries"] = f"ERROR: {exc}"

            try:
                t0 = _time.time()
                detect_contradictions(self.conn)
                stats["detect_contradictions"] = f"{_time.time() - t0:.3f}s"
            except Exception as exc:
                stats["detect_contradictions"] = f"ERROR: {exc}"

            try:
                t0 = _time.time()
                n = build_structured_facts(self.conn)
                stats["structured_facts"] = f"{n} facts in {_time.time() - t0:.3f}s"
            except Exception as exc:
                stats["structured_facts"] = f"ERROR: {exc}"

            if build_surprise_index:
                try:
                    t0 = _time.time()
                    build_surprise_index(self.conn)
                    stats["build_surprise_index"] = f"{_time.time() - t0:.3f}s"
                except Exception as exc:
                    stats["build_surprise_index"] = f"ERROR: {exc}"

            if detect_episodes:
                try:
                    t0 = _time.time()
                    ep = detect_episodes(self.conn)
                    stats["detect_episodes"] = f"{ep} episodes in {_time.time() - t0:.3f}s"
                except Exception as exc:
                    stats["detect_episodes"] = f"ERROR: {exc}"

            if detect_landmark_events:
                try:
                    t0 = _time.time()
                    lm = detect_landmark_events(self.conn)
                    stats["detect_landmarks"] = f"{lm} events in {_time.time() - t0:.3f}s"
                except Exception as exc:
                    stats["detect_landmarks"] = f"ERROR: {exc}"

            if build_dunbar_hierarchy:
                try:
                    t0 = _time.time()
                    primary = None
                    try:
                        row = self.conn.execute(
                            "SELECT sender, COUNT(*) as cnt FROM messages "
                            "WHERE sender != '' AND sender IS NOT NULL "
                            "GROUP BY sender ORDER BY cnt DESC LIMIT 1"
                        ).fetchone()
                        if row and row[0] and row[0].strip():
                            primary = row[0]
                    except Exception:
                        pass
                    result = build_dunbar_hierarchy(self.conn, primary_entity=primary)
                    n_rel = len(result) if isinstance(result, dict) else result
                    stats["dunbar_hierarchy"] = f"{n_rel} relationships in {_time.time() - t0:.3f}s"
                except Exception as exc:
                    stats["dunbar_hierarchy"] = f"ERROR: {exc}"

            self.conn.commit()

        self._has_consolidation = True
        return stats

    def update(self, memory_id: int, content: str | None = None, **fields) -> dict | None:
        """Update a memory.

        Re-embeds the message if ``content`` changes.

        Args:
            memory_id: The memory to update.
            content:   New content text (optional).
            **fields:  Other fields: ``sender``, ``recipient``,
                       ``timestamp``, ``category``.

        Returns:
            Updated memory dict, or None if not found.
        """
        self._ensure_connection()

        pre_embedding = None
        pre_sep_embedding = None
        if content is not None and self._has_vectors:
            try:
                from truememory.vector_search import (
                    get_model, _encode_with_mps_fallback, _build_sep_text,
                )
                model = get_model()
                pre_embedding = _encode_with_mps_fallback(model, [content])[0]
                row = self.conn.execute(
                    "SELECT sender, recipient, timestamp FROM messages WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                if row:
                    sender_val = fields.get("sender", row[0])
                    recipient_val = fields.get("recipient", row[1])
                    timestamp_val = fields.get("timestamp", row[2])
                    sep_text = _build_sep_text(sender_val, recipient_val, timestamp_val, content)
                    pre_sep_embedding = _encode_with_mps_fallback(model, [sep_text])[0]
            except Exception:
                logger.debug("Failed to pre-compute embedding during update()", exc_info=True)

        with self._write_lock:
            if content is not None:
                fields["content"] = content

            ok = update_message(self.conn, memory_id, **fields)
            if not ok:
                return None

            if pre_embedding is not None:
                try:
                    from truememory.vector_search import (
                        serialize_f32, _active_vec_table, _active_sep_table,
                        _write_embedder_metadata,
                    )
                    vec_tbl = _active_vec_table(self.conn)
                    try:
                        self.conn.execute(f"DELETE FROM {vec_tbl} WHERE rowid = ?", (memory_id,))
                    except Exception:
                        logger.debug("Failed to delete old vector embedding for message %d", memory_id, exc_info=True)
                    self.conn.execute(
                        f"INSERT INTO {vec_tbl}(rowid, embedding) VALUES (?, ?)",
                        (memory_id, serialize_f32(pre_embedding)),
                    )
                    if pre_sep_embedding is not None:
                        sep_tbl = _active_sep_table(self.conn)
                        try:
                            self.conn.execute(f"DELETE FROM {sep_tbl} WHERE rowid = ?", (memory_id,))
                        except Exception:
                            logger.debug("Failed to delete old sep vector for message %d", memory_id, exc_info=True)
                        self.conn.execute(
                            f"INSERT INTO {sep_tbl}(rowid, embedding) VALUES (?, ?)",
                            (memory_id, serialize_f32(pre_sep_embedding)),
                        )
                    _write_embedder_metadata(self.conn)
                except Exception:
                    logger.warning("Vector embedding failed for message %d", memory_id, exc_info=True)

            self.conn.commit()

        return self.get(memory_id)

    def get(self, memory_id: int) -> dict | None:
        """Retrieve a single memory by ID."""
        self._ensure_connection()
        return get_message(self.conn, memory_id)

    def get_all(self, limit: int = 100, offset: int = 0, user_id: str | None = None) -> list[dict]:
        """List memories with pagination.

        Args:
            limit:   Max memories to return.
            offset:  Number of rows to skip.
            user_id: Filter by sender (optional).

        Returns:
            List of memory dicts.
        """
        self._ensure_connection()
        from truememory.storage import _SELECT_COLS, _row_to_dict

        if user_id:
            rows = self.conn.execute(
                f"SELECT {_SELECT_COLS} FROM messages WHERE sender = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset),
            ).fetchall()
        else:
            rows = self.conn.execute(
                f"SELECT {_SELECT_COLS} FROM messages ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

        return [_row_to_dict(r) for r in rows]

    # ──────────────────────────────────────────────────────────────────────
    # Open existing database (for benchmarking / testing)
    # ──────────────────────────────────────────────────────────────────────

    def open(self, rebuild_vectors: bool = True) -> "TrueMemoryEngine":
        """Open an existing database for search (no ingestion).

        Connects to the database at ``self.db_path`` and detects which
        optional layers are available based on the tables present.
        Optionally rebuilds vector indexes if missing.

        Returns self for chaining: ``engine = TrueMemoryEngine(path).open()``
        """
        warnings.warn(
            "open() is deprecated — production code uses _ensure_connection()/create_db()",
            DeprecationWarning,
            stacklevel=2,
        )

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = None  # Use default tuple rows
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-64000")
        self.conn.execute("PRAGMA mmap_size=268435456")

        # Detect available tables
        tables = {
            row[0]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        # ── MEMORIST-L4 migration: purge legacy entity_profile summary rows ─
        # As of 2026-04-24, build_entity_summary_sheets is disabled by
        # default (see consolidate() step 12). Existing v0.5.0 databases
        # may contain `period='entity_profile'` rows that continue to be
        # surfaced by search_consolidated (no period filter). Delete them
        # once on open() so upgraders get the research's measured +5.3pt
        # lift on day one rather than after their next consolidation run.
        # Idempotent — re-running is a cheap no-op.
        #
        # Skipped if the user explicitly re-enables via
        # TRUEMEMORY_ENTITY_SHEETS=1 (the next consolidate() will rewrite
        # these rows, so deleting them here is pointless).
        # Skip if migration flag already set (idempotency + perf:
        # avoids a full-table scan of `summaries` on every open()).
        if "metadata" in tables:
            try:
                _cur = self.conn.execute(
                    "SELECT value FROM metadata WHERE key = ?",
                    ("l4_entity_profile_migration_done",),
                )
                _row = _cur.fetchone()
                _cur.close()
                _migration_done = _row is not None and _row[0] == "1"
            except Exception:
                _migration_done = False
        else:
            _migration_done = False

        _entity_sheets_re_enabled = (
            os.environ.get("TRUEMEMORY_ENTITY_SHEETS", "")
            .strip().lower() in {"1", "true", "yes", "on"}
        )

        if (
            "summaries" in tables
            and not _migration_done
            and not _entity_sheets_re_enabled
        ):
            try:
                cur = self.conn.execute(
                    "DELETE FROM summaries WHERE period = 'entity_profile'"
                )
                deleted = cur.rowcount
                cur.close()
                if deleted > 0:
                    self.conn.commit()
                    logger.info(
                        "MEMORIST-L4 migration: purged %d legacy "
                        "entity_profile summary rows (disabled by default; "
                        "set TRUEMEMORY_ENTITY_SHEETS=1 to re-enable)",
                        deleted,
                    )
                # Record the migration as done so subsequent opens skip
                # the scan (even if 0 rows were deleted).
                if "metadata" in tables:
                    try:
                        self.conn.execute(
                            "INSERT OR REPLACE INTO metadata (key, value) "
                            "VALUES (?, ?)",
                            ("l4_entity_profile_migration_done", "1"),
                        )
                        self.conn.commit()
                    except Exception:
                        logger.debug(
                            "failed to record l4 migration flag",
                            exc_info=True,
                        )
            except Exception:
                logger.warning(
                    "MEMORIST-L4 entity_profile migration failed; "
                    "legacy rows may remain. Set TRUEMEMORY_ENTITY_SHEETS=1 "
                    "to revert to legacy behavior if this is blocking.",
                    exc_info=True,
                )

        # Style vector hash migration: Python's hash() was replaced with a
        # stable hashlib-based hash in v0.6.3. Existing style vectors were
        # computed with the old non-deterministic hash and must be rebuilt.
        if "entity_style_vectors" in tables and "metadata" in tables:
            try:
                row = self.conn.execute(
                    "SELECT value FROM metadata WHERE key = 'style_vec_hash_version'"
                ).fetchone()
                if row is None or row[0] != "2":
                    if _HAS_STYLE_VEC:
                        from truememory.personality_style_vec import build_entity_style_vectors
                        build_entity_style_vectors(self.conn)
                        logger.info("Style vectors rebuilt with stable hash (one-time migration)")
                    self.conn.execute(
                        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                        ("style_vec_hash_version", "2"),
                    )
                    self.conn.commit()
            except Exception:
                logger.debug("Style vector migration failed", exc_info=True)

        # Load sqlite-vec extension if available.
        # upgrade DEBUG → WARNING and track failure in a
        # module-level state so ``truememory_stats.health`` can report
        # "search is FTS-only because sqlite-vec failed to load".
        global _vectors_load_error
        if _HAS_VECTOR:
            try:
                import sqlite_vec
                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                _vectors_load_error = None
            except Exception as e:
                _vectors_load_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    "sqlite-vec unavailable (%s); falling back to FTS-only "
                    "search for this process. See "
                    "https://github.com/buildingjoshbetter/TrueMemory for "
                    "platform notes.",
                    _vectors_load_error,
                )

        # Check for vector tables — rebuild if missing.
        # if metadata names a different embedder, refuse silent
        # rebuild; route the user through truememory_configure() instead.
        self._has_vectors = False
        if _HAS_VECTOR:
            from truememory.vector_search import _active_vec_table
            vec_tbl = _active_vec_table(self.conn)
            try:
                self.conn.execute(f"SELECT 1 FROM {vec_tbl} LIMIT 1").fetchone()
                self._has_vectors = True
            except Exception:
                logger.warning(
                    "Vector table %r missing or unreadable; attempting rebuild "
                    "with current model=%s",
                    vec_tbl,
                    resolve_tier(),
                )
                if rebuild_vectors:
                    _check_rebuild_allowed(self.conn)  # raises on model drift
                    try:
                        init_vec_table(self.conn)
                        n = build_vectors(self.conn)
                        self._has_vectors = n > 0
                        logger.info(
                            "Vector table %r rebuilt with %d vectors (model=%s)",
                            vec_tbl, n, resolve_tier(),
                        )
                    except TrueMemoryMigrationError:
                        raise
                    except Exception:
                        logger.exception("Vector table rebuild failed")
                        self._has_vectors = False

        self._has_hybrid = _HAS_HYBRID and self._has_vectors
        self._has_clustering = _HAS_CLUSTERING and "message_clusters" in tables

        # Count messages for stats
        try:
            count = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            self.stats["message_count"] = count
        except Exception:
            logger.debug("Failed to count messages in open()", exc_info=True)
            self.stats["message_count"] = 0

        self.ready = True
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────────────────────────────────

    def ingest(self, data_path: str | Path) -> dict:
        """
        Full ingestion pipeline.

        Steps:
            1. Create database with schema
            2. Load messages from JSON file
            3. Build vector embeddings (Model2Vec -> sqlite-vec)
            4. Build entity profiles (L0)
            5. Extract preferences (L0)
            6. Build summaries (L5)
            7. Detect contradictions (L5)
            8. Build surprise index (predictive coding)

        Each step is wrapped in ``try/except`` so partial failures never
        block the whole pipeline.  If vector search is not installed the
        engine still works with FTS5 only.

        Returns:
            Dict mapping step names to timing strings (or error messages).
        """
        data_path = Path(data_path)
        stats: dict = {}

        # ── 1. Create database ────────────────────────────────────────────
        try:
            t0 = time.time()
            # Remove stale database so every run is clean.
            if self.db_path.exists() and str(self.db_path) != ":memory:":
                self.db_path.unlink()
            self.conn = create_db(self.db_path)
            stats["create_db"] = f"{time.time() - t0:.3f}s"
        except Exception as exc:
            stats["create_db"] = f"ERROR: {exc}"
            logger.warning("create_db failed", exc_info=True)
            return stats  # Can't continue without a database.

        # ── 2. Load messages ──────────────────────────────────────────────
        try:
            t0 = time.time()
            msg_count = load_messages_from_file(self.conn, data_path)
            elapsed = time.time() - t0
            stats["load_messages"] = f"{msg_count} messages in {elapsed:.3f}s"
            self.stats["message_count"] = msg_count
        except Exception as exc:
            stats["load_messages"] = f"ERROR: {exc}"
            logger.warning("load_messages failed", exc_info=True)
            self.stats["message_count"] = 0

        # ── 3. Build vector embeddings ────────────────────────────────────
        if _HAS_VECTOR:
            try:
                t0 = time.time()
                init_vec_table(self.conn)
                vec_count = build_vectors(self.conn)
                elapsed = time.time() - t0
                stats["build_vectors"] = f"{vec_count} vectors in {elapsed:.3f}s"
                self._has_vectors = True
            except Exception as exc:
                stats["build_vectors"] = f"ERROR: {exc}"
                logger.debug("build_vectors failed", exc_info=True)
                self._has_vectors = False
        else:
            stats["build_vectors"] = "SKIPPED (sqlite-vec or model2vec not installed)"
            self._has_vectors = False

        # Update hybrid capability — needs vectors to be useful.
        self._has_hybrid = _HAS_HYBRID and self._has_vectors

        # ── 3b. Build scene clusters ───────────────────────────────────────
        if _HAS_CLUSTERING and self._has_vectors:
            try:
                t0 = time.time()
                n_clusters = cluster_messages(self.conn)
                stats["build_clusters"] = f"{n_clusters} clusters in {time.time() - t0:.3f}s"
                self._has_clustering = True
            except Exception as exc:
                stats["build_clusters"] = f"ERROR: {exc}"
                logger.debug("build_clusters failed", exc_info=True)
                self._has_clustering = False
        else:
            stats["build_clusters"] = "SKIPPED (clustering or vectors not available)"
            self._has_clustering = False

        # ── 4. Build entity profiles (L0) ─────────────────────────────────
        if _HAS_PERSONALITY:
            try:
                t0 = time.time()
                build_entity_profiles(self.conn)
                stats["build_profiles"] = f"{time.time() - t0:.3f}s"
            except Exception as exc:
                stats["build_profiles"] = f"ERROR: {exc}"
                logger.debug("build_profiles failed", exc_info=True)
                self._has_personality = False
        else:
            stats["build_profiles"] = "SKIPPED (personality module not available)"

        # ── 4.5. Build entity style vectors (L0 char-n-gram) ─────────────
        if _HAS_STYLE_VEC:
            try:
                t0 = time.time()
                style_vecs = build_entity_style_vectors(self.conn)
                stats["build_style_vectors"] = f"{len(style_vecs)} vectors in {time.time() - t0:.3f}s"
                self._has_style_vec = True
            except Exception as exc:
                stats["build_style_vectors"] = f"ERROR: {exc}"
                logger.debug("build_style_vectors failed", exc_info=True)
                self._has_style_vec = False
        else:
            stats["build_style_vectors"] = "SKIPPED (personality_style_vec module not available)"

        # ── 5. Extract preferences (L0) ───────────────────────────────────
        if _HAS_PERSONALITY:
            try:
                t0 = time.time()
                extract_preferences(self.conn)
                stats["extract_preferences"] = f"{time.time() - t0:.3f}s"
            except Exception as exc:
                stats["extract_preferences"] = f"ERROR: {exc}"
                logger.debug("extract_preferences failed", exc_info=True)
        else:
            stats["extract_preferences"] = "SKIPPED (personality module not available)"

        # ── 6. Build summaries (L5) ───────────────────────────────────────
        if _HAS_CONSOLIDATION:
            try:
                t0 = time.time()
                build_summaries(self.conn)
                stats["build_summaries"] = f"{time.time() - t0:.3f}s"
            except Exception as exc:
                stats["build_summaries"] = f"ERROR: {exc}"
                logger.debug("build_summaries failed", exc_info=True)
                self._has_consolidation = False
        else:
            stats["build_summaries"] = "SKIPPED (consolidation module not available)"

        # ── 7. Detect contradictions (L5) ─────────────────────────────────
        if _HAS_CONSOLIDATION:
            try:
                t0 = time.time()
                detect_contradictions(self.conn)
                stats["detect_contradictions"] = f"{time.time() - t0:.3f}s"
            except Exception as exc:
                stats["detect_contradictions"] = f"ERROR: {exc}"
                logger.debug("detect_contradictions failed", exc_info=True)
        else:
            stats["detect_contradictions"] = "SKIPPED (consolidation module not available)"

        # ── 8. Build surprise index (predictive coding) ───────────────────
        if _HAS_PREDICTIVE:
            try:
                t0 = time.time()
                build_surprise_index(self.conn)
                stats["build_surprise_index"] = f"{time.time() - t0:.3f}s"
            except Exception as exc:
                stats["build_surprise_index"] = f"ERROR: {exc}"
                logger.debug("build_surprise_index failed", exc_info=True)
                self._has_predictive = False
        else:
            stats["build_surprise_index"] = "SKIPPED (predictive module not available)"

        # ── 9. Build separation vectors (B2) ─────────────────────────────
        if _HAS_VECTOR:
            try:
                t0 = time.time()
                sep_count = build_separation_vectors(self.conn)
                stats["build_separation_vectors"] = f"{sep_count} vectors in {time.time() - t0:.3f}s"
            except Exception as exc:
                stats["build_separation_vectors"] = f"ERROR: {exc}"
                logger.debug("build_separation_vectors failed", exc_info=True)
        else:
            stats["build_separation_vectors"] = "SKIPPED (vector module not available)"

        # ── 10. Detect episodes (B1) ─────────────────────────────────────
        if _HAS_TEMPORAL:
            try:
                t0 = time.time()
                ep_count = detect_episodes(self.conn)
                stats["detect_episodes"] = f"{ep_count} episodes in {time.time() - t0:.3f}s"
            except Exception as exc:
                stats["detect_episodes"] = f"ERROR: {exc}"
                logger.debug("detect_episodes failed", exc_info=True)
        else:
            stats["detect_episodes"] = "SKIPPED (temporal module not available)"

        # ── 11. Detect landmark events (E3) ──────────────────────────────
        if _HAS_TEMPORAL:
            try:
                t0 = time.time()
                lm_count = detect_landmark_events(self.conn)
                stats["detect_landmarks"] = f"{lm_count} events in {time.time() - t0:.3f}s"
            except Exception as exc:
                stats["detect_landmarks"] = f"ERROR: {exc}"
                logger.debug("detect_landmarks failed", exc_info=True)
        else:
            stats["detect_landmarks"] = "SKIPPED (temporal module not available)"

        # ── 12. Build entity summary sheets (B3) ─────────────────────────
        # Disabled 2026-04-24 per MEMORIST-L4 research.
        # See CHANGELOG v0.6.0 for rationale.
        # The function wrote `summaries` rows with period='entity_profile'
        # that saturated top-1 retrieval by keyword match and leaked
        # superseded facts into contradiction scoring. Disabling produced
        # +5.3 pts on the L4 composite probe metric (Pareto-dominant,
        # see REPORT.md §3 Table "D1 vs C1" and §10.7 Ablation 2).
        #
        # Escape hatch: set TRUEMEMORY_ENTITY_SHEETS=1 to re-enable this
        # function. Users who regress on real-world workloads can revert
        # without a code patch. Intended to be removed in a future release
        # once long-horizon production telemetry confirms the disable.
        _entity_sheets_enabled = (
            os.environ.get("TRUEMEMORY_ENTITY_SHEETS", "")
            .strip().lower() in {"1", "true", "yes", "on"}
        )
        if _HAS_CONSOLIDATION and _entity_sheets_enabled:
            try:
                t0 = time.time()
                sheet_count = build_entity_summary_sheets(self.conn)
                stats["entity_summary_sheets"] = f"{sheet_count} sheets in {time.time() - t0:.3f}s (re-enabled via TRUEMEMORY_ENTITY_SHEETS=1)"
            except Exception as exc:
                stats["entity_summary_sheets"] = f"ERROR: {exc}"
                logger.debug("entity_summary_sheets failed", exc_info=True)
        elif _HAS_CONSOLIDATION:
            stats["entity_summary_sheets"] = "DISABLED (MEMORIST-L4; set TRUEMEMORY_ENTITY_SHEETS=1 to re-enable)"
        else:
            stats["entity_summary_sheets"] = "SKIPPED (consolidation module not available)"

        # ── 13. Build structured facts (B4) ──────────────────────────────
        if _HAS_CONSOLIDATION:
            try:
                t0 = time.time()
                fact_count = build_structured_facts(self.conn)
                stats["structured_facts"] = f"{fact_count} facts in {time.time() - t0:.3f}s"
            except Exception as exc:
                stats["structured_facts"] = f"ERROR: {exc}"
                logger.debug("structured_facts failed", exc_info=True)
        else:
            stats["structured_facts"] = "SKIPPED (consolidation module not available)"

        # ── 14. Build Dunbar hierarchy (E2) ──────────────────────────────
        if _HAS_PERSONALITY:
            try:
                t0 = time.time()
                primary = None
                try:
                    row = self.conn.execute(
                        "SELECT sender, COUNT(*) as cnt FROM messages "
                        "WHERE sender != '' AND sender IS NOT NULL "
                        "GROUP BY sender ORDER BY cnt DESC LIMIT 1"
                    ).fetchone()
                    if row and row[0] and row[0].strip():
                        primary = row[0]
                except Exception as exc:
                    logger.debug("Failed to detect primary entity for Dunbar: %s", exc)
                dunbar_result = build_dunbar_hierarchy(self.conn, primary_entity=primary)
                dunbar_count = len(dunbar_result) if isinstance(dunbar_result, dict) else dunbar_result
                stats["dunbar_hierarchy"] = f"{dunbar_count} relationships in {time.time() - t0:.3f}s"
            except Exception as exc:
                stats["dunbar_hierarchy"] = f"ERROR: {exc}"
                logger.debug("dunbar_hierarchy failed", exc_info=True)
        else:
            stats["dunbar_hierarchy"] = "SKIPPED (personality module not available)"

        # ── Record capabilities ───────────────────────────────────────────
        self.stats["capabilities"] = {
            "fts5": True,
            "vector_search": self._has_vectors,
            "hybrid_rrf": self._has_hybrid,
            "temporal": self._has_temporal,
            "salience": self._has_salience,
            "personality": self._has_personality,
            "style_vec": self._has_style_vec,
            "consolidation": self._has_consolidation,
            "predictive": self._has_predictive,
            "reranker": self._has_reranker,
            "hyde": self._has_hyde,
            "clustering": self._has_clustering,
        }

        self.ready = True
        return stats

    # ──────────────────────────────────────────────────────────────────────
    # Search — full 6-layer pipeline
    # ──────────────────────────────────────────────────────────────────────

    def search_vectors_raw(self, query: str, limit: int = 5) -> list[dict] | None:
        """Pure vector cosine similarity search, or None if unavailable.

        Returns results with ``score`` as cosine similarity in [0, 1].
        Returns None (not empty list) when vectors aren't available,
        so callers can distinguish "no vectors" from "no matches."
        """
        self._ensure_connection()
        if not self._has_vectors:
            return None
        try:
            from truememory.vector_search import search_vector_raw
            return search_vector_raw(self.conn, query, limit=limit)
        except Exception:
            return None

    def search(self, query: str, limit: int = 10, _skip_surprise_boost: bool = False, _skip_reranker: bool = False, _skip_salience_guard: bool = False, include_directives: bool = False) -> list[dict]:
        """
        Main search pipeline.

        1. Classify query and determine search mode (spotlight vs diffuse).
        2. Try hybrid search (FTS5 + vector + RRF) with adaptive weights.
           Falls back to FTS5-only if vector search is unavailable.
        3. Scent trail -- follow entity/term trails from top results.
        4. Quality self-check -- detect uniformly low scores, trigger fallback.
        5. Check for temporal intent -- apply temporal filtering with
           cross-source feedback (re-scope vector search to detected window).
        6. Check for personality query -- supplement with personality search.
        7. Check for contradiction -- supplement with fact timeline.
        8. Apply salience guard with mode-aware threshold.
        8.5. L5 surprise rerank boost.
        8.6. Cross-encoder reranking (skipped when called from search_agentic).
        9. Return top *limit* results.

        Each result dict contains: ``id``, ``content``, ``sender``,
        ``recipient``, ``timestamp``, ``category``, ``modality``, ``score``,
        ``source`` (which retrieval method found it).

        Every step is wrapped in ``try/except`` with fallbacks.  If any layer
        fails, the search still returns results from the layers that worked.
        """
        self._ensure_connection()
        if not self.conn:
            return []

        results: list[dict] = []
        source_label = "fts"

        # ── 0. Classify query (A2/A5) ────────────────────────────────────
        query_info = {
            "query_type": "general",
            "weights": {"fts": 1.0, "vec": 1.0},
            "confidence": 0.5,
        }
        search_mode = "spotlight"
        if _HAS_QUERY_CLASSIFIER:
            try:
                query_info = classify_query(query)
                search_mode = get_search_mode(query)
            except Exception:
                logger.debug("Query classification failed in search()", exc_info=True)

        fts_w = query_info["weights"].get("fts", 1.0)
        vec_w = query_info["weights"].get("vec", 1.0)

        # ── 1. Primary retrieval with adaptive weights (A1) ──────────────
        if self._has_hybrid:
            try:
                results = search_hybrid(
                    self.conn, query, limit=limit * 3,
                    fts_weight=fts_w, vec_weight=vec_w,
                    include_directives=include_directives,
                )
                source_label = "hybrid"
            except Exception:
                logger.debug("Hybrid search failed, falling back to FTS5", exc_info=True)
                results = []

        if not results:
            try:
                results = search_fts(self.conn, query, limit=limit * 3, include_directives=include_directives)
                source_label = "fts"
                # Normalize FTS results to match the hybrid output shape.
                for r in results:
                    if "source" not in r:
                        r["source"] = "fts"
            except Exception:
                logger.debug("FTS search failed in search()", exc_info=True)
                results = []

        # ── Sender diversity check ─────────────────────────────────────
        # In conversations with few unique senders (2-3 person chats like
        # LoCoMo), scent trail and quality self-check add noise rather than
        # value because every follow-up search returns messages from the same
        # small set of speakers.
        _unique_senders = {r.get("sender", "").lower() for r in results if r.get("sender")}
        _unique_senders.discard("")
        _high_diversity = len(_unique_senders) > 5

        # ── 2. Scent trail (A3) — only when sender diversity is high ─────
        if _high_diversity and len(results) >= 3:
            try:
                results = self._scent_trail(query, results, limit)
            except Exception:
                logger.debug("Scent trail failed in search()", exc_info=True)

        # ── 3. Retrieval quality self-check (A4) — only with high diversity
        if _high_diversity:
            try:
                results = self._quality_self_check(query, results, limit)
            except Exception:
                logger.debug("Quality self-check failed in search()", exc_info=True)

        # ── 4. Temporal filtering with cross-source feedback (A1) ─────────
        if self._has_temporal:
            try:
                intent = detect_temporal_intent(query)
                if intent.get("has_temporal"):
                    # search_temporal takes the query + existing results and
                    # filters/re-ranks them using the detected time window.
                    if source_label == "hybrid":
                        temporal_results = search_temporal(
                            self.conn, query,
                            hybrid_results=results,
                            limit=limit * 2,
                            include_directives=include_directives,
                        )
                    else:
                        temporal_results = search_temporal(
                            self.conn, query,
                            fts_results=results,
                            limit=limit * 2,
                            include_directives=include_directives,
                        )
                    if temporal_results:
                        # Tag new results with temporal source and merge.
                        existing_ids = {r["id"] for r in results}
                        for tr in temporal_results:
                            if "source" not in tr:
                                tr["source"] = "temporal"
                            if tr["id"] not in existing_ids:
                                results.append(tr)
                                existing_ids.add(tr["id"])
                            else:
                                # Boost score of temporally-relevant existing results.
                                for r in results:
                                    if r["id"] == tr["id"]:
                                        r["score"] = r.get("score", 0) * 1.3
                                        if r.get("source") and "temporal" not in r["source"]:
                                            r["source"] = r["source"] + "+temporal"
                                        break

                        # If the intent calls for chronological order, re-sort
                        # by timestamp (trajectory queries).
                        if intent.get("is_trajectory") or intent.get("sort_by_time"):
                            results.sort(key=lambda r: r.get("timestamp", ""))

                    # Cross-source feedback (A1): if temporal found a date
                    # window, re-scope vector search to that window for
                    # additional candidates.
                    if self._has_hybrid and intent.get("after") and intent.get("before"):
                        try:
                            from truememory.fts_search import search_fts_in_range
                            range_results = search_fts_in_range(
                                self.conn, query,
                                after=intent["after"],
                                before=intent["before"],
                                limit=limit,
                                include_directives=include_directives,
                            )
                            if range_results:
                                existing_ids = {r.get("id") for r in results if r.get("id")}
                                # Rescoped rows arrive in [0,1] FTS space and
                                # would dominate the RRF-scored pool (cap
                                # ~0.05). Rescale them to the local max so
                                # they compete fairly (#633 M-10).
                                _resc_max = max(
                                    (r.get("score", r.get("rrf_score", 0)) for r in results),
                                    default=0.05,
                                )
                                for rr in range_results:
                                    if rr.get("id") and rr["id"] not in existing_ids:
                                        rr["source"] = "temporal_rescoped"
                                        _rs = rr.get("score", 1.0)
                                        if not isinstance(_rs, (int, float)):
                                            _rs = 1.0
                                        rr["score"] = _resc_max * 0.8 * max(0.0, min(_rs, 1.0))
                                        results.append(rr)
                                        existing_ids.add(rr["id"])
                        except Exception:
                            logger.debug("Temporal cross-source rescope failed in search()", exc_info=True)

            except Exception:
                logger.debug("Temporal filtering failed in search()", exc_info=True)

        # ── 5. Personality supplementation ────────────────────────────────
        # Only inject personality/profile results when the query is actually
        # about personality, preferences, habits, or character.  Otherwise
        # profile results (score=1.0) dominate factual queries.
        if self._has_personality and _has_personality_intent(query):
            try:
                # Only forward include_directives when explicitly enabled so
                # we stay compatible with callers/mocks using the older
                # 3-arg signature (the default-False path already excludes
                # directives via search_personality's own candidate filter).
                _pers_kwargs = {"include_directives": True} if include_directives else {}
                personality_results = search_personality(
                    self.conn, query, limit=5, **_pers_kwargs,
                )
                if personality_results:
                    existing_ids = {r.get("id") for r in results if r.get("id")}
                    max_existing = max(
                        (r.get("score", 0) for r in results), default=0.05
                    )
                    _l0_scale_raw = os.environ.get("TRUEMEMORY_L0_SCORE_SCALE")
                    try:
                        _l0_scale = float(_l0_scale_raw) if _l0_scale_raw is not None else 0.9
                    except (ValueError, TypeError):
                        _l0_scale = 0.9
                    _l0_scale = max(0.0, min(_l0_scale, 1.0))
                    for pr in personality_results:
                        if "source" not in pr:
                            pr["source"] = "personality"
                        if pr.get("source") in ("profile", "style_vec", "fts"):
                            pr["score"] = max_existing * (0.8 if pr["source"] == "profile" else _l0_scale)
                        pr_id = pr.get("id")
                        if pr_id and pr_id in existing_ids:
                            continue
                        results.append(pr)
                        if pr_id:
                            existing_ids.add(pr_id)
            except Exception:
                logger.debug("Personality supplementation failed in search()", exc_info=True)

        # ── 6. Contradiction / fact timeline ──────────────────────────────
        if self._has_consolidation:
            try:
                contradiction_results = search_contradictions(self.conn, query)
                if contradiction_results:
                    # Use .get(): supplement rows (personality/profile) may
                    # lack an "id" key — bracket access raised KeyError that
                    # the blanket except silently swallowed, discarding ALL
                    # contradiction supplements on personality queries
                    # (#630 M-05).
                    existing_ids = {r.get("id") for r in results if r.get("id")}
                    # Score contradictions competitively so they survive
                    # the top-K slice and reach the reranker (#581).
                    max_existing = max(
                        (r.get("score", r.get("rrf_score", 0)) for r in results),
                        default=0.01,
                    )
                    for cr in contradiction_results:
                        cr["source"] = cr.get("source", "contradiction")
                        # Ensure 'content' key exists so the salience
                        # guard can score the row (#581 bug-1).
                        if "content" not in cr:
                            cr["content"] = cr.get(
                                "current_fact",
                                cr.get("text", cr.get("memory", "")),
                            )
                        # Assign a competitive score so contradictions
                        # are not sliced off before reranking (#581 bug-2).
                        if "score" not in cr:
                            cr["score"] = max_existing * 0.8
                        if cr.get("id") and cr["id"] not in existing_ids:
                            results.append(cr)
                            existing_ids.add(cr["id"])
            except Exception:
                logger.debug("Contradiction search failed in search()", exc_info=True)

            try:
                consolidated = search_consolidated(self.conn, query, limit=3)
                if consolidated:
                    # .get() guard (#630 M-05): see contradiction block above.
                    existing_ids = {r.get("id") for r in results if r.get("id")}
                    # Consolidated rows carry RAW integer keyword-overlap
                    # scores (e.g. relevance*2) that dwarf RRF scores
                    # (~0.0167); any overlap >= 2 would outrank every
                    # organic hit. Rescale them to the local pool before
                    # appending so they compete fairly (#633 M-09).
                    _cons_max = max(
                        (r.get("score", r.get("rrf_score", 0)) for r in results),
                        default=0.05,
                    )
                    _raw_max = max(
                        (s.get("score", 0) for s in consolidated
                         if isinstance(s.get("score"), (int, float))),
                        default=0.0,
                    )
                    for sr in consolidated:
                        sr["source"] = sr.get("source", "summary")
                        _raw = sr.get("score", 0)
                        if not isinstance(_raw, (int, float)) or _raw_max <= 0:
                            _rel = 1.0
                        else:
                            _rel = _raw / _raw_max
                        sr["score"] = _cons_max * 0.8 * max(0.0, min(_rel, 1.0))
                        if sr.get("id") and sr["id"] not in existing_ids:
                            results.append(sr)
                            existing_ids.add(sr["id"])
            except Exception:
                logger.debug("Consolidated search failed in search()", exc_info=True)

        # ── 7. Salience guard with mode-aware threshold (A5) ──────────────
        # Skipped when called from search_agentic() which applies its own
        # salience guard after entity boosting to avoid filtering out
        # low-salience entity-matched rows before the boost can rescue them.
        if self._has_salience and results and not _skip_salience_guard:
            try:
                _sal_override = os.environ.get("TRUEMEMORY_MIN_SALIENCE")
                if _sal_override is not None:
                    try:
                        min_sal = float(_sal_override)
                    except (ValueError, TypeError):
                        min_sal = 0.02 if search_mode == "diffuse" else 0.05
                else:
                    min_sal = 0.02 if search_mode == "diffuse" else 0.05
                results = apply_salience_guard(
                    results, query, conn=self.conn, min_salience=min_sal,
                )
            except Exception:
                logger.debug("Salience guard failed in search()", exc_info=True)

        # ── 7.5 L5 surprise rerank boost (MEMORIST-L5) ──
        # Skipped when called from search_agentic() which applies its own
        # boost after merging all result sources.
        if not _skip_surprise_boost:
            results = self._apply_surprise_boost(results)

        # ── 8.6 Cross-encoder reranking ──────────────────────────────────
        # Skipped when called from search_agentic() which applies its own
        # reranking after merging all result sources.
        if not _skip_reranker and self._has_reranker and len(results) > 1:
            try:
                from truememory.reranker import rerank_with_modality_fusion
                # Re-sort by score desc BEFORE the candidate slice so that
                # supplements appended at the tail (contradiction / summary /
                # temporal_rescoped) are not sliced off before the reranker
                # ever sees them. search_agentic already re-sorts; this makes
                # the main path consistent (#633 M-11, restores #581).
                results.sort(
                    key=lambda r: (
                        -(r.get("score", r.get("rrf_score", r.get("raw_score", 0))) or 0),
                        str(r.get("id", "")),
                    )
                )
                results = rerank_with_modality_fusion(
                    query, results[:limit * 3],
                    top_k=limit,
                    rrf_weight=0.4,
                    rerank_weight=0.6,
                )
            except Exception:
                logger.debug("Cross-encoder rerank failed in search()", exc_info=True)

        # ── 9. Ensure all results have required fields and trim ───────────
        cleaned: list[dict] = []
        seen_ids: set = set()
        seen_content: set = set()

        for r in results:
            # Exclude directives unless explicitly requested (#588)
            if not include_directives and r.get("directive"):
                continue

            content = r.get("content", "")
            rid = r.get("id")

            # Deduplicate by id (if present) and by content prefix
            content_key = content[:200]
            if rid and rid in seen_ids:
                continue
            if content_key in seen_content:
                continue

            # Resolve the best available score.  Some layers return
            # different score keys (rrf_score, raw_score, entity_boost).
            score = r.get("score", r.get("rrf_score", r.get("raw_score", 0)))
            # Clamp negative BM25 raw scores to 0 so they sort correctly.
            if isinstance(score, (int, float)) and score < 0:
                score = 0.0

            cleaned.append({
                # Preserve None for id-less supplement rows (#630 M-67).
                # Rewriting to 0 made #606's id-keyed RRF collapse multiple
                # distinct id-less rows into one fabricated id=0 document.
                "id": rid,
                "content": content,
                "sender": r.get("sender", ""),
                "recipient": r.get("recipient", ""),
                "timestamp": r.get("timestamp", ""),
                "category": r.get("category", ""),
                "modality": r.get("modality", ""),
                "directive": r.get("directive", False),
                "score": score,
                "source": r.get("source", source_label),
            })

            if rid:
                seen_ids.add(rid)
            seen_content.add(content_key)

        # Sort by score descending, then by str(id) for determinism.
        # str() is type-stable even when ids mix int message ids with
        # "summary_N" / None supplement ids (#630 M-01).
        cleaned.sort(key=lambda d: (-d["score"], str(d.get("id", ""))))
        return cleaned[:limit]

    # ──────────────────────────────────────────────────────────────────────
    # Search — agentic multi-round retrieval
    # ──────────────────────────────────────────────────────────────────────

    def search_agentic(
        self,
        query: str,
        limit: int = 10,
        max_rounds: int = 2,
        llm_fn=None,
        use_hyde: bool = True,
        use_reranker: bool = True,
        use_clustering: bool = True,
        reranker_device: str | None = None,
        max_per_session: int = 0,
        use_llm_reranker: bool = False,
        include_directives: bool = False,
    ) -> list[dict]:
        """
        Agentic retrieval with sufficiency checking and multi-query generation.

        This is the full retrieval pipeline designed to maximize recall:

        Round 1:
            1. Standard 6-layer search (primary results — keeps its own scoring)
            2. HyDE-enhanced search (if llm_fn available) — fused via RRF
            3. Cluster-scoped search — supplemental only (added to pool, not fused)

        Sufficiency check:
            - If top-5 results have high scores and good diversity, skip round 2.

        Round 2 (if not sufficient and llm_fn available):
            1. Generate 2-3 refined sub-queries
            2. Run each through 6-layer pipeline
            3. Merge new results into the pool

        Final (if use_reranker):
            - Cross-encoder reranking on the candidate pool
            - Return top *limit* results

        Args:
            query:           The search query.
            limit:           Maximum results to return.
            max_rounds:      Maximum retrieval rounds (default 2).
            llm_fn:          Callable for HyDE generation and query refinement.
            use_hyde:        Whether to use HyDE (requires llm_fn).
            use_reranker:    Whether to apply cross-encoder reranking.
            use_clustering:  Whether to include cluster-scoped search.
            reranker_device: Device for cross-encoder.

        Returns:
            List of result dicts sorted by relevance.
        """
        self._ensure_connection()
        if not self.conn:
            return []

        # ── Round 1: Primary retrieval ────────────────────────────────────
        # Standard 6-layer search is our anchor — it keeps its own scoring.
        # When reranker is enabled, pull a much larger candidate pool so the
        # cross-encoder can surface evidence from deeper in the ranking.
        if use_reranker and self._has_reranker:
            candidate_pool = max(limit * 8, 100)  # Large pool for reranking
        else:
            candidate_pool = limit * 3
        primary_results = self.search(query, limit=candidate_pool, _skip_surprise_boost=True, _skip_reranker=True, _skip_salience_guard=True, include_directives=include_directives)

        # If HyDE available, run a parallel search and fuse with RRF
        if use_hyde and self._has_hyde and self._has_hybrid and llm_fn:
            try:
                hyde_results = hyde_search(
                    self.conn, query, llm_fn=llm_fn, limit=candidate_pool,
                    include_directives=include_directives,
                )
                if hyde_results:
                    from truememory.hybrid import reciprocal_rank_fusion
                    primary_results = reciprocal_rank_fusion(
                        [primary_results, hyde_results]
                    )
            except Exception:
                logger.debug("HyDE search failed in search_agentic()", exc_info=True)

        # ── Score normalization (issue #584) ─────────────────────────────
        # Normalize primary results to [0, 1] before merging supplements so
        # all sources compete in the same score space.
        from truememory.agentic_search import normalize_scores

        normalize_scores(primary_results)

        # Cluster search: add as supplement, preserving cluster/diversity
        # ordering.  Cluster scores (cosine 0-1) live in a different space
        # than RRF scores; we normalize them independently and tag each
        # result with its diversity position so downstream sorts can
        # preserve the cluster-sampling order.
        if use_clustering and self._has_clustering:
            try:
                cluster_results = search_clustered(
                    self.conn, query, limit=limit, top_clusters=3,
                    include_directives=include_directives,
                )
                if cluster_results and primary_results:
                    # Normalize cluster scores to [0, 1] independently
                    normalize_scores(cluster_results)

                    existing_ids = {r.get("id") for r in primary_results if r.get("id")}
                    for idx, cr in enumerate(cluster_results):
                        cid = cr.get("id")
                        if cid and cid not in existing_ids:
                            # Preserve diversity position from clustering
                            cr["_cluster_position"] = idx
                            cr["source"] = cr.get("source", "") + "+cluster_supp"
                            primary_results.append(cr)
                            existing_ids.add(cid)
            except Exception:
                logger.debug("Cluster search failed in search_agentic()", exc_info=True)

        # ── Entity-focused search ───────────────────────────────────────
        # Extract person names from the query and search specifically within
        # their messages.  This addresses single-hop keyword mismatch where
        # the question mentions a person but uses different vocabulary than
        # the evidence (e.g., "What did X research?" vs "Researching adoption
        # agencies...").
        #
        # Strategy: BOOST existing primary results that also appear in entity
        # search (overlap = strong signal), and add genuinely new results
        # with independently normalized scores so they compete fairly.
        #
        # Fix #582: Use _entity_boosted flag to ensure boost is applied
        # exactly once.  The flag is checked before boosting and set after,
        # so repeated calls (e.g. from refined-query merges) are idempotent.
        try:
            entity_results = self._entity_focused_search(query, limit * 2)
            if entity_results and primary_results:
                # Detect the query's temporal window so the entity boost and
                # entity_new rows do not invert in-window temporal ordering
                # by appending out-of-window rows at up to 1.0 (#633 M-68).
                _ent_after = _ent_before = None
                try:
                    from truememory.temporal import (
                        detect_temporal_intent,
                        _validate_iso_date,
                        _exclusive_upper_bound,
                    )
                    _ent_intent = detect_temporal_intent(query)
                    _ent_after = _validate_iso_date(_ent_intent.get("after"))
                    _eb = _validate_iso_date(_ent_intent.get("before"))
                    _ent_before = _exclusive_upper_bound(_eb) if _eb else None
                except Exception:
                    logger.debug("Entity temporal-window detection failed", exc_info=True)

                def _in_window(row: dict) -> bool:
                    # No window detected -> every row is "in window".
                    if _ent_after is None and _ent_before is None:
                        return True
                    ts = row.get("timestamp") or ""
                    if not ts:
                        return False
                    if _ent_after is not None and ts < _ent_after:
                        return False
                    if _ent_before is not None and ts >= _ent_before:
                        return False
                    return True

                # Normalize entity scores to [0, 1] independently
                normalize_scores(entity_results)

                entity_ids = {r.get("id") for r in entity_results if r.get("id")}

                # Boost primary results that overlap with entity search.
                # Gate the 1.5x boost on window membership so out-of-window
                # evidence is not promoted over in-window rows (#633 M-68).
                for pr in primary_results:
                    pid = pr.get("id")
                    if pid and pid in entity_ids and not pr.get("_entity_boosted"):
                        if _in_window(pr):
                            pr["score"] = pr.get("score", pr.get("rrf_score", 0)) * 1.5
                            pr["_entity_boosted"] = True
                            if "entity_boost" not in pr.get("source", ""):
                                pr["source"] = pr.get("source", "") + "+entity_boost"

                existing_ids = {r.get("id") for r in primary_results if r.get("id")}
                added = 0
                for er in entity_results:
                    eid = er.get("id")
                    if eid and eid not in existing_ids:
                        er["source"] = er.get("source", "") + "+entity_new"
                        # Only in-window entity_new rows are exempt from the
                        # salience floor and keep their full normalized score;
                        # out-of-window rows are demoted so they cannot
                        # outrank in-window temporal evidence (#633 M-68).
                        if _in_window(er):
                            er["_entity_boosted"] = True
                        else:
                            er["score"] = er.get("score", 0) * 0.1
                        primary_results.append(er)
                        existing_ids.add(eid)
                        added += 1
                        if added >= limit:
                            break

                # Re-sort primary results only (cluster supplements keep
                # their _cluster_position and are appended after primary).
                primary_results.sort(
                    key=lambda d: (-d.get("score", 0), d.get("id", 0))
                )
        except Exception:
            logger.debug("Entity-focused search failed in search_agentic()", exc_info=True)

        # ── Salience guard (deferred from search() for #582) ─────────────
        # Applied HERE instead of inside search() so that entity-boosted
        # rows survive the salience floor.  Entity-matched rows (flagged
        # with _entity_boosted) are exempt from the salience floor to
        # prevent the guard from discarding low-salience entity evidence.
        if self._has_salience and primary_results:
            try:
                _sal_override = os.environ.get("TRUEMEMORY_MIN_SALIENCE")
                if _sal_override is not None:
                    try:
                        min_sal = float(_sal_override)
                    except (ValueError, TypeError):
                        min_sal = 0.05
                else:
                    min_sal = 0.05
                primary_results = apply_salience_guard(
                    primary_results, query, conn=self.conn,
                    min_salience=min_sal,
                    entity_rescue_ids=frozenset(
                        r.get("id") for r in primary_results
                        if r.get("_entity_boosted") and r.get("id")
                    ),
                )
            except Exception:
                logger.debug("Salience guard failed in search_agentic()", exc_info=True)

        # ── Sufficiency check ─────────────────────────────────────────────
        is_sufficient = self._check_sufficiency(primary_results[:5])

        # ── Round 2: Refined queries (if not sufficient) ──────────────────
        if not is_sufficient and max_rounds >= 2 and llm_fn:
            refined_queries = self._generate_refined_queries(
                query, primary_results[:5], llm_fn,
            )

            existing_ids = {r.get("id") for r in primary_results if r.get("id")}
            for rq in refined_queries:
                try:
                    rq_results = self.search(rq, limit=limit, _skip_surprise_boost=True, _skip_reranker=True, include_directives=include_directives)
                    normalize_scores(rq_results)
                    for rr in rq_results:
                        rid = rr.get("id")
                        if rid and rid not in existing_ids:
                            rr["source"] = rr.get("source", "") + "+refined"
                            rr["score"] = rr.get("score", 0) * 0.9
                            primary_results.append(rr)
                            existing_ids.add(rid)
                        elif rid and rid in existing_ids:
                            # Boost existing result that also appears in refined
                            for pr in primary_results:
                                if pr.get("id") == rid:
                                    pr["score"] = pr.get("score", 0) * 1.15
                                    break
                except Exception:
                    logger.debug("Refined query search failed in search_agentic()", exc_info=True)

            # Re-sort after adding refined results
            primary_results.sort(
                key=lambda d: (-d.get("score", 0), d.get("id", 0))
            )

        # ── L5 surprise rerank boost (MEMORIST-L5, applied BEFORE cross-encoder) ──
        # Per ISSUES.md Issue #1 Scope: "join surprise_scores after RRF/L3
        # and before cross-encoder rerank". Boost mutates score in place;
        # cross-encoder then folds the boosted RRF into fused_score so the
        # signal propagates through LLM reranker downstream (rerank_with_llm
        # would otherwise overwrite a post-rerank boost).
        primary_results = self._apply_surprise_boost(primary_results)

        # ── Cross-encoder reranking (modality-aware) ──────────────────────
        if use_reranker and self._has_reranker and len(primary_results) > 1:
            try:
                from truememory.reranker import rerank_with_modality_fusion
                final_results = rerank_with_modality_fusion(
                    query, primary_results[:limit * 5],
                    top_k=limit * 2 if (max_per_session > 0 or use_llm_reranker) else limit,
                    rrf_weight=0.4,
                    rerank_weight=0.6,
                    device=reranker_device,
                )
                # ── LLM reranking (optional, after cross-encoder) ──────────
                if use_llm_reranker and llm_fn and len(final_results) > limit:
                    try:
                        from truememory.reranker import rerank_with_llm
                        _llm_cap = min(limit * 2, 50)
                        final_results = rerank_with_llm(
                            query, final_results[:_llm_cap],
                            llm_fn=llm_fn, top_k=limit,
                        )
                    except Exception:
                        logger.debug("LLM reranking failed in search_agentic()", exc_info=True)
                return self._clean_results(final_results, limit, max_per_session=max_per_session,
                                           include_directives=include_directives)
            except Exception:
                logger.warning("Cross-encoder rerank failed in search_agentic()", exc_info=True)

        # ── LLM reranking without cross-encoder ──────────────────────────
        if use_llm_reranker and llm_fn and len(primary_results) > limit:
            try:
                from truememory.reranker import rerank_with_llm
                _llm_cap = min(limit * 2, 50)
                primary_results = rerank_with_llm(
                    query, primary_results[:_llm_cap],
                    llm_fn=llm_fn, top_k=limit,
                )
            except Exception:
                logger.debug("LLM reranking (standalone) failed in search_agentic()", exc_info=True)

        return self._clean_results(primary_results, limit, max_per_session=max_per_session,
                                   include_directives=include_directives)

    # ── L5 surprise rerank boost (MEMORIST-L5 wiring, 2026-04-24) ──────
    # Multiplies the reranked `score` field by (1 + α · surprise) for
    # message-backed rows, then re-sorts. Source-gated so non-message rows
    # (summaries, personality profiles, contradictions) are not boosted —
    # their `id` values do NOT reference `messages.id`, so joining on
    # `surprise_scores.message_id` would silently mismatch and rewrite
    # unrelated rows' scores.
    #
    # Default α=0.2 per Modal alpha sweep (2026-04-26): 5-point sweep
    # (0, 0.1, 0.15, 0.2, 0.3) × 3 seeds each found α=0.2 is the
    # empirical peak at 93.20% mean LoCoMo (vs 93.00% at α=0, 93.07%
    # at α=0.3). Cross-validated on GPUBox (RTX 5090).
    #
    # Precedence: constructor arg > env var > 0.2.
    #
    # See ``_working/memorist/l5_predictive/REPORT.md`` §1, §10 for
    # rationale and ``ISSUES.md`` for the follow-up validation plan.

    _warned_no_surprise = False

    @staticmethod
    def _source_is_blocked(source: str | None) -> bool:
        from truememory.l5_boost import source_is_blocked
        return source_is_blocked(source)

    def _get_alpha_surprise(self) -> float:
        from truememory.l5_boost import get_alpha_surprise
        return get_alpha_surprise(
            getattr(self, "_alpha_surprise_override", None),
        )

    def _apply_surprise_boost(self, results: list[dict]) -> list[dict]:
        from truememory.l5_boost import apply_surprise_boost
        return apply_surprise_boost(
            self.conn, results,
            alpha_override=getattr(self, "_alpha_surprise_override", None),
        )

    def _check_sufficiency(self, top_results: list[dict]) -> bool:
        from truememory.agentic_search import check_sufficiency
        return check_sufficiency(top_results)

    def _generate_refined_queries(self, original_query: str, top_results: list[dict], llm_fn) -> list[str]:
        from truememory.agentic_search import generate_refined_queries
        return generate_refined_queries(original_query, top_results, llm_fn)

    def _entity_focused_search(self, query: str, limit: int) -> list[dict]:
        from truememory.agentic_search import entity_focused_search
        return entity_focused_search(self.conn, query, limit)

    def _clean_results(self, results: list[dict], limit: int, max_per_session: int = 0,
                       include_directives: bool = False) -> list[dict]:
        from truememory.agentic_search import clean_results
        return clean_results(results, limit, max_per_session,
                             include_directives=include_directives)

    def _scent_trail(self, query: str, results: list[dict], limit: int) -> list[dict]:
        from truememory.search_quality import scent_trail
        return scent_trail(self.conn, query, results, limit)

    def _quality_self_check(self, query: str, results: list[dict], limit: int) -> list[dict]:
        from truememory.search_quality import quality_self_check
        return quality_self_check(self.conn, query, results, limit)

    # ──────────────────────────────────────────────────────────────────────
    # Search — simple FTS5-only (for benchmarking)
    # ──────────────────────────────────────────────────────────────────────

    def search_simple(self, query: str, limit: int = 10) -> list[dict]:
        """
        Simple FTS5-only search (no vector, no layers).

        Useful for comparison benchmarks to measure the value added by
        each layer beyond raw keyword search.
        """
        if not self.conn:
            return []

        try:
            results = search_fts(self.conn, query, limit=limit)
        except Exception:
            return []

        cleaned: list[dict] = []
        for r in results:
            cleaned.append({
                "id": r.get("id", 0),
                "content": r.get("content", ""),
                "sender": r.get("sender", ""),
                "recipient": r.get("recipient", ""),
                "timestamp": r.get("timestamp", ""),
                "category": r.get("category", ""),
                "modality": r.get("modality", ""),
                "directive": r.get("directive", False),
                "score": r.get("score", 0),
                "source": "fts",
            })

        return cleaned

    # ──────────────────────────────────────────────────────────────────────
    # Stats / teardown
    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return ingestion and search statistics."""
        self._ensure_connection()
        stats = dict(self.stats)

        # Add live DB stats if connected.
        if self.conn:
            try:
                stats["message_count"] = get_message_count(self.conn)
                dc = self.conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE directive = 1"
                ).fetchone()
                stats["directive_count"] = dc[0] if dc else 0
            except Exception:
                logger.debug("Failed to get message count in get_stats()", exc_info=True)

            try:
                db_path = Path(self.db_path)
                if db_path.exists() and str(db_path) != ":memory:":
                    stats["db_size_kb"] = round(db_path.stat().st_size / 1024, 1)
            except Exception:
                logger.debug("Failed to get db size in get_stats()", exc_info=True)

        return stats

    def close(self):
        """Close database connection and join background threads."""
        if self._consolidation_thread is not None and self._consolidation_thread.is_alive():
            try:
                self._consolidation_thread.join(timeout=5)
            except Exception:
                pass
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                logger.debug("Failed to close database connection", exc_info=True)
            self.conn = None
            self.ready = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        status = "ready" if self.ready else "not ready"
        msg_count = self.stats.get("message_count", "?")
        return f"<TrueMemoryEngine db={self.db_path.name} msgs={msg_count} {status}>"
