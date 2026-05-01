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
    )
    _HAS_VECTOR = True
except (ImportError, ModuleNotFoundError):
    pass


# Hunter F08: module-level tracker for sqlite-vec load failures. On platforms
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

_QUERY_STOP_WORDS = frozenset({
    "what", "did", "does", "do", "how", "where", "when", "who",
    "which", "why", "is", "are", "was", "were", "has", "have",
    "had", "would", "could", "should", "will", "can", "the",
    "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "about", "their", "they", "them", "his", "her", "its",
})


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
                self._has_vectors = True
            except Exception:
                logger.debug("Failed to load sqlite-vec extension", exc_info=True)
                self._has_vectors = False

        self._has_hybrid = _HAS_HYBRID and self._has_vectors
        self.ready = True

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
        self._ensure_connection()

        msg = {
            "content": content,
            "sender": sender,
            "recipient": recipient,
            "timestamp": timestamp,
            "category": category,
            "modality": "",
        }
        new_id = insert_message(self.conn, msg)

        # Embed the message for vector search
        if self._has_vectors:
            try:
                from truememory.vector_search import embed_single
                embed_single(self.conn, new_id, content)
            except Exception:
                logger.debug("Failed to embed message %s during add()", new_id, exc_info=True)

        # Incrementally update entity profile
        if self._has_personality and sender:
            try:
                from truememory.personality import update_entity_profile_incremental
                update_entity_profile_incremental(self.conn, sender, content, recipient)
            except Exception:
                logger.debug("Failed to update entity profile for %s during add()", sender, exc_info=True)

        # Incrementally update style vector (L0 char-n-gram)
        if self._has_style_vec and sender:
            try:
                _update_style_vec(self.conn, sender, content)
            except Exception:
                logger.debug("Failed to update style vector for %s during add()", sender, exc_info=True)

        # Persist vector embedding and any profile updates
        self.conn.commit()

        return {
            "id": new_id,
            "content": content,
            "sender": sender,
            "recipient": recipient,
            "timestamp": timestamp,
            "category": category,
        }

    def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID.

        Returns True if deleted, False if not found.
        """
        self._ensure_connection()
        return delete_message(self.conn, memory_id)

    def delete_all(self, user_id: str | None = None) -> bool:
        """Delete all memories, optionally filtered by user.

        Handles deletion from ALL tables in the schema: messages,
        messages_fts, entity_profiles, fact_timeline, summaries,
        episodes, landmark_events, causal_edges, entity_relationships,
        and vector tables (vec_messages, vec_messages_sep).

        Args:
            user_id: If provided, only delete this user's memories and
                     related data.  If None, deletes everything.

        Returns:
            True if any rows were deleted from messages.
        """
        self._ensure_connection()

        if user_id:
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

            cursor = self.conn.execute(
                "DELETE FROM messages WHERE sender = ?", (user_id,)
            )
            deleted = cursor.rowcount > 0

            # Clean up related tables scoped to this user
            if msg_ids:
                placeholders = ",".join("?" * len(msg_ids))

                for table, col in [
                    ("fact_timeline", "source_message_id"),
                    ("landmark_events", "source_message_id"),
                    ("causal_edges", "cause_msg_id"),
                    ("causal_edges", "effect_msg_id"),
                ]:
                    try:
                        self.conn.execute(
                            f"DELETE FROM {table} WHERE {col} IN ({placeholders})",
                            msg_ids,
                        )
                    except Exception:
                        logger.debug("Failed to clean %s for user %s", table, user_id, exc_info=True)

            # Clean entity profile for this user
            try:
                self.conn.execute(
                    "DELETE FROM entity_profiles WHERE entity = ?", (user_id,)
                )
            except Exception:
                logger.debug("Failed to clean entity_profiles for user %s", user_id, exc_info=True)

            # Clean entity style vectors for this user
            try:
                self.conn.execute(
                    "DELETE FROM entity_style_vectors WHERE entity = ?", (user_id,)
                )
            except Exception:
                logger.debug("Failed to clean entity_style_vectors for user %s", user_id, exc_info=True)

            # Clean entity relationships involving this user
            try:
                self.conn.execute(
                    "DELETE FROM entity_relationships WHERE entity_a = ? OR entity_b = ?",
                    (user_id, user_id),
                )
            except Exception:
                logger.debug("Failed to clean entity_relationships for user %s", user_id, exc_info=True)

            # Clean summaries scoped to this user
            try:
                self.conn.execute(
                    "DELETE FROM summaries WHERE entity = ?", (user_id,)
                )
            except Exception:
                logger.debug("Failed to clean summaries for user %s", user_id, exc_info=True)

            # Clean episodes linked to this user's messages
            if episode_ids:
                ep_placeholders = ",".join("?" * len(episode_ids))
                try:
                    self.conn.execute(
                        f"DELETE FROM episodes WHERE id IN ({ep_placeholders})",
                        episode_ids,
                    )
                except Exception:
                    logger.debug("Failed to clean episodes for user %s", user_id, exc_info=True)

            # Clean vector tables for deleted message IDs
            if msg_ids:
                for vec_table in ("vec_messages", "vec_messages_sep"):
                    try:
                        self.conn.execute(
                            f"DELETE FROM {vec_table} WHERE rowid IN ({placeholders})",
                            msg_ids,
                        )
                    except Exception:
                        logger.debug("Failed to clean %s for user %s", vec_table, user_id, exc_info=True)

        else:
            # Full wipe of all tables
            cursor = self.conn.execute("DELETE FROM messages")
            deleted = cursor.rowcount > 0

            for table in (
                "entity_profiles",
                "entity_style_vectors",
                "fact_timeline",
                "summaries",
                "episodes",
                "landmark_events",
                "causal_edges",
                "entity_relationships",
            ):
                try:
                    self.conn.execute(f"DELETE FROM {table}")
                except Exception:
                    logger.debug("Failed to clear table %s during delete_all", table, exc_info=True)

            # Clear vector tables
            for vec_table in ("vec_messages", "vec_messages_sep"):
                try:
                    self.conn.execute(f"DELETE FROM {vec_table}")
                except Exception:
                    logger.debug("Failed to clear %s during delete_all", vec_table, exc_info=True)

        # Rebuild FTS index
        try:
            self.conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        except Exception:
            logger.debug("Failed to rebuild FTS index during delete_all", exc_info=True)

        self.conn.commit()
        return deleted

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
        if content is not None:
            fields["content"] = content

        ok = update_message(self.conn, memory_id, **fields)
        if not ok:
            return None

        # Re-embed if content changed
        if content is not None and self._has_vectors:
            try:
                from truememory.vector_search import embed_single
                # Remove old embedding, insert new
                try:
                    self.conn.execute("DELETE FROM vec_messages WHERE rowid = ?", (memory_id,))
                except Exception:
                    logger.debug("Failed to delete old vector embedding for message %d", memory_id, exc_info=True)
                embed_single(self.conn, memory_id, content)
            except Exception:
                logger.warning("Vector embedding failed for message %d", memory_id, exc_info=True)

        # Persist vector embedding and any profile updates
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
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = None  # Use default tuple rows
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")

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

        # Load sqlite-vec extension if available.
        # Hunter F08: upgrade DEBUG → WARNING and track failure in a
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
        # Hunter F32: if metadata names a different embedder, refuse silent
        # rebuild; route the user through truememory_configure() instead.
        self._has_vectors = False
        if _HAS_VECTOR:
            try:
                self.conn.execute("SELECT COUNT(*) FROM vec_messages").fetchone()
                self._has_vectors = True
            except Exception:
                logger.warning(
                    "vec_messages table missing; attempting rebuild with "
                    "current model=%s",
                    os.environ.get("TRUEMEMORY_EMBED_MODEL", "edge"),
                )
                if rebuild_vectors:
                    _check_rebuild_allowed(self.conn)  # raises on model drift
                    try:
                        init_vec_table(self.conn)
                        n = build_vectors(self.conn)
                        self._has_vectors = n > 0
                        logger.info(
                            "vec_messages rebuilt with %d vectors (model=%s)",
                            n,
                            os.environ.get("TRUEMEMORY_EMBED_MODEL", "edge"),
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

    def search(self, query: str, limit: int = 10, _skip_surprise_boost: bool = False) -> list[dict]:
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
                )
                source_label = "hybrid"
            except Exception:
                logger.debug("Hybrid search failed, falling back to FTS5", exc_info=True)
                results = []

        if not results:
            try:
                results = search_fts(self.conn, query, limit=limit * 3)
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
                        )
                    else:
                        temporal_results = search_temporal(
                            self.conn, query,
                            fts_results=results,
                            limit=limit * 2,
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
                    if self._has_hybrid and intent.get("start_date") and intent.get("end_date"):
                        try:
                            from truememory.fts_search import search_fts_in_range
                            range_results = search_fts_in_range(
                                self.conn, query,
                                after=intent["start_date"],
                                before=intent["end_date"],
                                limit=limit,
                            )
                            if range_results:
                                existing_ids = {r.get("id") for r in results if r.get("id")}
                                for rr in range_results:
                                    if rr.get("id") and rr["id"] not in existing_ids:
                                        rr["source"] = "temporal_rescoped"
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
                personality_results = search_personality(self.conn, query, limit=5)
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
                    existing_ids = {r["id"] for r in results}
                    for cr in contradiction_results:
                        cr["source"] = cr.get("source", "contradiction")
                        if cr.get("id") and cr["id"] not in existing_ids:
                            results.append(cr)
                            existing_ids.add(cr["id"])
            except Exception:
                logger.debug("Contradiction search failed in search()", exc_info=True)

            try:
                consolidated = search_consolidated(self.conn, query, limit=3)
                if consolidated:
                    existing_ids = {r["id"] for r in results}
                    for sr in consolidated:
                        sr["source"] = sr.get("source", "summary")
                        if sr.get("id") and sr["id"] not in existing_ids:
                            results.append(sr)
                            existing_ids.add(sr["id"])
            except Exception:
                logger.debug("Consolidated search failed in search()", exc_info=True)

        # ── 7. Salience guard with mode-aware threshold (A5) ──────────────
        if self._has_salience and results:
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

        # ── 8. Ensure all results have required fields and trim ───────────
        cleaned: list[dict] = []
        seen_ids: set = set()
        seen_content: set = set()

        for r in results:
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
                "id": rid if rid else 0,
                "content": content,
                "sender": r.get("sender", ""),
                "recipient": r.get("recipient", ""),
                "timestamp": r.get("timestamp", ""),
                "category": r.get("category", ""),
                "modality": r.get("modality", ""),
                "score": score,
                "source": r.get("source", source_label),
            })

            if rid:
                seen_ids.add(rid)
            seen_content.add(content_key)

        # Sort by score descending, then by id for determinism.
        cleaned.sort(key=lambda d: (-d["score"], d["id"]))
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
        primary_results = self.search(query, limit=candidate_pool, _skip_surprise_boost=True)

        # If HyDE available, run a parallel search and fuse with RRF
        if use_hyde and self._has_hyde and self._has_hybrid and llm_fn:
            try:
                hyde_results = hyde_search(
                    self.conn, query, llm_fn=llm_fn, limit=candidate_pool,
                )
                if hyde_results:
                    from truememory.hybrid import reciprocal_rank_fusion
                    primary_results = reciprocal_rank_fusion(
                        [primary_results, hyde_results]
                    )
            except Exception:
                logger.debug("HyDE search failed in search_agentic()", exc_info=True)

        # Cluster search: add as low-priority supplement (never displace primary)
        # Cluster scores (cosine 0-1) are in a different range than RRF scores
        # (~0.01-0.03), so we must rescale cluster scores to be below the
        # lowest primary result to ensure they only fill empty slots.
        if use_clustering and self._has_clustering:
            try:
                cluster_results = search_clustered(
                    self.conn, query, limit=limit, top_clusters=3,
                )
                if cluster_results and primary_results:
                    # Find the minimum score in primary results
                    min_primary = min(
                        r.get("score", r.get("rrf_score", 0))
                        for r in primary_results
                    )
                    existing_ids = {r.get("id") for r in primary_results if r.get("id")}
                    for cr in cluster_results:
                        cid = cr.get("id")
                        if cid and cid not in existing_ids:
                            # Scale to just below primary results
                            cr["score"] = min_primary * 0.5
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
        # search (overlap = strong signal), and add genuinely new results at
        # a competitive score level so they can displace weak primary results.
        try:
            entity_results = self._entity_focused_search(query, limit * 2)
            if entity_results and primary_results:
                entity_ids = {r.get("id") for r in entity_results if r.get("id")}

                # Boost primary results that overlap with entity search
                for pr in primary_results:
                    pid = pr.get("id")
                    if pid and pid in entity_ids:
                        pr["score"] = pr.get("score", pr.get("rrf_score", 0)) * 1.5
                        if "entity_boost" not in pr.get("source", ""):
                            pr["source"] = pr.get("source", "") + "+entity_boost"

                # Add genuinely new entity results at median primary score
                primary_scores = [
                    r.get("score", r.get("rrf_score", 0)) for r in primary_results
                ]
                primary_scores.sort(reverse=True)
                # Use median score so new entity results can compete for top-k
                median_idx = len(primary_scores) // 2
                median_score = primary_scores[median_idx] if primary_scores else 0.01

                existing_ids = {r.get("id") for r in primary_results if r.get("id")}
                added = 0
                for er in entity_results:
                    eid = er.get("id")
                    if eid and eid not in existing_ids:
                        er["score"] = median_score * 0.9
                        er["source"] = er.get("source", "") + "+entity_new"
                        primary_results.append(er)
                        existing_ids.add(eid)
                        added += 1
                        if added >= limit:
                            break

                # Re-sort after boosting
                primary_results.sort(
                    key=lambda d: (-d.get("score", d.get("rrf_score", 0)), d.get("id", 0))
                )
        except Exception:
            logger.debug("Entity-focused search failed in search_agentic()", exc_info=True)

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
                    rq_results = self.search(rq, limit=limit, _skip_surprise_boost=True)
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
                key=lambda d: (-d.get("score", d.get("rrf_score", 0)), d.get("id", 0))
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
                return self._clean_results(final_results, limit, max_per_session=max_per_session)
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

        return self._clean_results(primary_results, limit, max_per_session=max_per_session)

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

    _SURPRISE_BOOST_SOURCE_BLOCKLIST = frozenset({
        "personality", "profile", "style_vec", "summary", "contradiction",
    })
    # IN-clause parameter chunk size. SQLite default is 999 variables —
    # keep a healthy margin for any other bound params in the query.
    _SURPRISE_IN_CHUNK = 500

    def _source_is_blocked(self, source: str | None) -> bool:
        """True if any '+'-separated segment of `source` is in the
        blocklist. Handles composed labels like 'personality+refined'
        produced by the agentic refined-query loop.
        """
        if not source:
            return False
        return any(
            seg in self._SURPRISE_BOOST_SOURCE_BLOCKLIST
            for seg in source.split("+")
        )

    _DEFAULT_ALPHA_SURPRISE = 0.2

    def _get_alpha_surprise(self) -> float:
        """Resolve alpha_surprise per MEMORIST-L5 precedence:
        constructor arg > TRUEMEMORY_ALPHA_SURPRISE env var > 0.2.

        Sanitizes against non-finite values (inf, -inf, nan) and
        TypeError/ValueError. Negative values are clamped to 0.
        """
        import math
        # Constructor override path
        alpha = getattr(self, "_alpha_surprise_override", None)
        if alpha is not None:
            try:
                a = float(alpha)
            except (TypeError, ValueError):
                return self._DEFAULT_ALPHA_SURPRISE
            if math.isnan(a) or math.isinf(a):
                return self._DEFAULT_ALPHA_SURPRISE
            return max(0.0, a)
        # Env-var path
        env = os.environ.get("TRUEMEMORY_ALPHA_SURPRISE")
        if env:
            try:
                a = float(env)
            except ValueError:
                logger.warning(
                    "Invalid TRUEMEMORY_ALPHA_SURPRISE=%r; using default", env,
                )
                return self._DEFAULT_ALPHA_SURPRISE
            if math.isnan(a) or math.isinf(a):
                logger.warning(
                    "Non-finite TRUEMEMORY_ALPHA_SURPRISE=%r; using default", env,
                )
                return self._DEFAULT_ALPHA_SURPRISE
            return max(0.0, a)
        return self._DEFAULT_ALPHA_SURPRISE

    def _apply_surprise_boost(self, results: list[dict]) -> list[dict]:
        """Apply L5 surprise multiplicative boost to message-backed rows.

        Mutates ``r["score"]`` (the canonical field that
        ``rerank_with_modality_fusion`` sorts on) so re-sort is coherent.
        Non-message rows and rows without a surprise score are left
        untouched. When ``alpha_surprise == 0.0`` this function is a
        no-op that preserves result order byte-for-byte.
        """
        if not results:
            return results
        alpha = self._get_alpha_surprise()
        if alpha <= 0.0:
            return results  # exact no-op; identical order preserved

        # Collect message-backed row IDs. Non-message rows carry a
        # `source` that indicates a different origin table. Sources can
        # be composed via "+" (e.g. "personality+refined" from the
        # round-2 refined-queries loop); check every segment against
        # the blocklist so composite labels don't escape.
        message_rows = [
            r for r in results
            if r.get("id") is not None
            and not self._source_is_blocked(r.get("source"))
        ]
        if not message_rows:
            return results

        ids = [r["id"] for r in message_rows]
        surprise_map: dict[int, float] = {}
        try:
            # Chunk to stay under SQLite's 999-variable IN limit.
            for i in range(0, len(ids), self._SURPRISE_IN_CHUNK):
                chunk = ids[i : i + self._SURPRISE_IN_CHUNK]
                placeholders = ",".join("?" * len(chunk))
                cur = self.conn.execute(
                    f"SELECT message_id, surprise FROM surprise_scores "
                    f"WHERE message_id IN ({placeholders})",
                    chunk,
                )
                surprise_map.update(dict(cur.fetchall()))
        except sqlite3.OperationalError as exc:
            # Most likely surprise_scores table doesn't exist yet (cold
            # DB before first consolidate). Surface at WARNING once per
            # process so silent no-ops are visible.
            logger.warning(
                "L5 surprise boost unavailable: %s (run consolidate first)",
                exc,
            )
            return results
        except Exception:
            logger.warning(
                "L5 surprise boost failed; returning unboosted results",
                exc_info=True,
            )
            return results

        if not surprise_map:
            return results  # no scored messages; nothing to boost

        # Apply multiplicative boost on the canonical `score` field
        # that rerank_with_modality_fusion set to the fused_score.
        for r in message_rows:
            s = surprise_map.get(r["id"], 0.0)
            if s > 0.0:
                base = r.get("score", r.get("rerank_score", r.get("rrf_score", 0.0)))
                r["score"] = base * (1.0 + alpha * float(s))

        # Re-sort by the same canonical field.
        results = sorted(
            results,
            key=lambda r: r.get("score", 0.0),
            reverse=True,
        )
        return results

    def _check_sufficiency(self, top_results: list[dict]) -> bool:
        """
        Check if retrieval results are sufficient (no need for round 2).

        Considers:
        - Average score of top results
        - Content diversity (unique content prefixes)
        """
        if not top_results or len(top_results) < 3:
            return False

        scores = [r.get("score", r.get("rrf_score", 0)) for r in top_results]
        avg_score = sum(scores) / len(scores)

        # Check content diversity
        unique_prefixes = {r.get("content", "")[:100] for r in top_results}

        # Sufficient if scores are high and results are diverse
        return avg_score > 0.02 and len(unique_prefixes) >= 3

    def _generate_refined_queries(
        self,
        original_query: str,
        top_results: list[dict],
        llm_fn,
    ) -> list[str]:
        """
        Generate 2-3 refined sub-queries based on the original query and
        initial retrieval results.

        For list-type questions (What X does Y do? Where has Y been?), we
        generate queries targeting DIFFERENT instances that may not overlap
        with already-retrieved results.
        """
        context_snippets = []
        for r in top_results[:5]:
            content = r.get("content", "")[:150]
            _sender = r.get("sender", "")
            if content:
                context_snippets.append(content)

        context_str = "\n".join(context_snippets)

        # Detect list-type questions
        q_lower = original_query.lower()
        list_patterns = [
            "what ", "which ", "where has", "where did",
            "what activities", "what books", "what movies",
            "what hobbies", "what does", "what do",
            "what items", "what places", "what foods",
            "what sports", "what songs", "what bands",
        ]
        is_list_question = any(q_lower.startswith(p) or p in q_lower for p in list_patterns)

        if is_list_question:
            prompt = f"""Given this question about a conversation: "{original_query}"

Here are results already found:
{context_str}

This question likely has MULTIPLE answers scattered across different parts of the conversation.
Generate 3 search queries to find ADDITIONAL answers not yet covered. Each query should:
- Use different vocabulary than what's already found
- Search for the same topic from a different angle
- Target different time periods or contexts

Return ONLY the queries, one per line:"""
        else:
            prompt = f"""Given this question about a conversation: "{original_query}"

And these partially relevant results:
{context_str}

Generate 2-3 alternative search queries that might find the missing information. Each query should approach the question from a different angle — try different keywords, focus on specific entities or events mentioned, or rephrase the question.

Return ONLY the queries, one per line, no numbering or explanation:"""

        try:
            response = llm_fn(prompt)
            lines = [
                line.strip().strip('"').strip("'").strip("-").strip()
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            return lines[:3]
        except Exception:
            return []

    # ──────────────────────────────────────────────────────────────────────
    # Entity-focused search helper
    # ──────────────────────────────────────────────────────────────────────

    def _entity_focused_search(self, query: str, limit: int) -> list[dict]:
        """
        Extract person names from the query and search specifically within
        their messages using FTS by sender.

        This helps single-hop questions like "What did Caroline research?"
        by finding messages FROM Caroline that match "research", rather than
        searching all 419 messages where common query words dilute relevance.

        Also extracts key content terms and does a focused FTS search with
        just those terms (dropping stop words like "what/did/how/does").
        """
        if not self.conn:
            return []

        results = []

        # ── 1. Extract person names ──────────────────────────────────────
        # Look for capitalized words that could be names.
        # Also check against known senders in the database.
        known_senders = set()
        try:
            rows = self.conn.execute(
                "SELECT DISTINCT sender FROM messages"
            ).fetchall()
            known_senders = {r[0].lower() for r in rows if r[0]}
        except Exception:
            logger.debug("Failed to fetch known senders for entity search", exc_info=True)

        # Find words in query that match known senders
        query_words = query.split()
        matched_senders = []
        for word in query_words:
            clean = word.strip("'\"?.,!").lower()
            # Check if word (or word with possessive removed) matches a sender
            for sender in known_senders:
                if clean == sender.lower() or clean.rstrip("'s") == sender.lower():
                    matched_senders.append(sender)

        # Also try multi-word sender names
        query_lower = query.lower()
        for sender in known_senders:
            if len(sender) > 2 and sender.lower() in query_lower:
                if sender not in matched_senders:
                    matched_senders.append(sender)

        # ── 2. Search within matched senders ──────────────────────────────
        if matched_senders:
            # Build a focused query by removing the sender name and stop words
            stop_words = _QUERY_STOP_WORDS

            content_words = []
            for word in query_words:
                clean = word.strip("'\"?.,!").lower().rstrip("'s")
                if (clean not in stop_words
                    and len(clean) > 2
                    and clean not in {s.lower() for s in matched_senders}):
                    content_words.append(word.strip("'\"?.,!"))

            if content_words:
                focused_query = " ".join(content_words)
            else:
                focused_query = query

            try:
                from truememory.fts_search import search_fts_by_sender
                for sender in matched_senders[:2]:  # Max 2 senders
                    # Find the original-case sender name
                    orig_sender = sender
                    for s_row in self.conn.execute(
                        "SELECT DISTINCT sender FROM messages"
                    ).fetchall():
                        if s_row[0] and s_row[0].lower() == sender.lower():
                            orig_sender = s_row[0]
                            break

                    sender_results = search_fts_by_sender(
                        self.conn, focused_query, orig_sender, limit=limit
                    )
                    for r in sender_results:
                        r["source"] = "entity_sender"
                    results.extend(sender_results)
            except Exception:
                logger.debug("FTS by sender search failed in entity search", exc_info=True)

        # ── 3. Focused content-only search ────────────────────────────────
        # Search with just the key content terms, dropping question structure
        if not matched_senders:
            stop_words = _QUERY_STOP_WORDS
            content_words = [
                w.strip("'\"?.,!") for w in query_words
                if w.strip("'\"?.,!").lower() not in stop_words
                and len(w.strip("'\"?.,!")) > 2
            ]
            if len(content_words) >= 2 and content_words != query_words:
                focused_query = " ".join(content_words)
                try:
                    focused_results = search_fts(self.conn, focused_query, limit=limit)
                    for r in focused_results:
                        r["source"] = "entity_focused"
                    results.extend(focused_results)
                except Exception:
                    logger.debug("Focused content search failed in entity search", exc_info=True)

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Result cleaning helper
    # ──────────────────────────────────────────────────────────────────────

    def _clean_results(
        self, results: list[dict], limit: int,
        max_per_session: int = 0,
    ) -> list[dict]:
        """Deduplicate and clean a result list.

        Args:
            results:         Raw results to clean.
            limit:           Maximum results to return.
            max_per_session: If >0, cap results per session/category for diversity.
                             When the cap is hit, extra results are deferred and
                             fill any remaining slots after the diversity pass.
        """
        cleaned: list[dict] = []
        seen_ids: set = set()
        seen_content: set = set()

        for r in results:
            content = r.get("content", "")
            rid = r.get("id")

            content_key = content[:200]
            if rid and rid in seen_ids:
                continue
            if content_key in seen_content:
                continue

            score = r.get("score", r.get("rrf_score", r.get("fused_score", r.get("rerank_score", 0))))
            if isinstance(score, (int, float)) and score < 0:
                score = 0.0

            cleaned.append({
                "id": rid if rid else 0,
                "content": content,
                "sender": r.get("sender", ""),
                "recipient": r.get("recipient", ""),
                "timestamp": r.get("timestamp", ""),
                "category": r.get("category", ""),
                "modality": r.get("modality", ""),
                "score": score,
                "source": r.get("source", "agentic"),
            })

            if rid:
                seen_ids.add(rid)
            seen_content.add(content_key)

        cleaned.sort(key=lambda d: (-d["score"], d["id"]))

        # ── Session-diversity enforcement ───────────────────────────────
        if max_per_session > 0 and len(cleaned) > limit:
            diverse: list[dict] = []
            deferred: list[dict] = []
            session_counts: dict[str, int] = {}

            for r in cleaned:
                sess = r.get("category", "unknown")
                count = session_counts.get(sess, 0)
                if count < max_per_session:
                    diverse.append(r)
                    session_counts[sess] = count + 1
                else:
                    deferred.append(r)

            # Fill remaining slots with deferred (best-scored first)
            if len(diverse) < limit:
                diverse.extend(deferred[: limit - len(diverse)])

            return diverse[:limit]

        return cleaned[:limit]

    # ──────────────────────────────────────────────────────────────────────
    # Search helpers — scent trail and quality self-check
    # ──────────────────────────────────────────────────────────────────────

    def _scent_trail(self, query: str, results: list[dict], limit: int) -> list[dict]:
        """
        Follow entity/term trails from top results to find related messages.

        After initial retrieval, extract key entities and proper nouns from the
        top-3 results.  Run targeted follow-up searches scoped to those
        entities (hop 1) and combined trail terms (hop 2).  Merge new results
        back in with slightly discounted scores.  Capped at 2 hops.

        Args:
            query:   The original search query.
            results: Current result list (must have >= 3 entries).
            limit:   The caller's requested limit (used for sizing).

        Returns:
            Augmented result list with scent-trail discoveries appended.
        """
        if not results or len(results) < 3:
            return results

        # Extract entities and key terms from top-3 results
        trail_entities: set[str] = set()
        trail_terms: set[str] = set()

        for r in results[:3]:
            content = r.get("content", "")
            sender = r.get("sender", "").lower()
            recipient = r.get("recipient", "").lower()

            if sender:
                trail_entities.add(sender)
            if recipient:
                trail_entities.add(recipient)

            # Extract proper nouns from content
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            common = {
                "The", "This", "That", "What", "When", "Where", "How",
                "But", "And", "Just", "Not",
            }
            for noun in proper_nouns:
                if noun not in common and len(noun) > 2:
                    trail_terms.add(noun.lower())

        if not trail_entities and not trail_terms:
            return results

        existing_ids = {r.get("id") for r in results if r.get("id")}
        new_results: list[dict] = []

        # Hop 1: search for messages from/about trail entities
        for entity in list(trail_entities)[:3]:
            try:
                from truememory.fts_search import search_fts_by_sender
                entity_results = search_fts_by_sender(self.conn, query, entity, limit=5)
                for er in entity_results:
                    if er.get("id") and er["id"] not in existing_ids:
                        er["source"] = er.get("source", "scent_trail")
                        new_results.append(er)
                        existing_ids.add(er["id"])
            except Exception:
                logger.debug("Scent trail hop 1 failed for entity %s", entity, exc_info=True)

        # Hop 2: search with combined trail terms
        if trail_terms:
            trail_query = " ".join(list(trail_terms)[:5])
            try:
                term_results = search_fts(self.conn, trail_query, limit=5)
                for tr in term_results:
                    if tr.get("id") and tr["id"] not in existing_ids:
                        tr["source"] = "scent_trail"
                        # Slightly lower score since these are follow-up hits
                        tr["score"] = tr.get("score", 0) * 0.7
                        new_results.append(tr)
                        existing_ids.add(tr["id"])
            except Exception:
                logger.debug("Scent trail hop 2 failed", exc_info=True)

        results.extend(new_results)
        return results

    def _quality_self_check(self, query: str, results: list[dict], limit: int) -> list[dict]:
        """
        Check if results are uniformly low quality and trigger fallback if so.

        If the top-5 results all have scores below 0.04 with a range < 0.005,
        the retrieval likely failed to find anything truly relevant.  In that
        case, fall back to broader single-term FTS searches to cast a wider
        net.

        Args:
            query:   The original search query.
            results: Current result list.
            limit:   The caller's requested limit.

        Returns:
            Original or augmented result list.
        """
        if not results or len(results) < 5:
            return results

        top5_scores = [r.get("score", r.get("rrf_score", 0)) for r in results[:5]]

        # Check for uniformly low scores
        max_score = max(top5_scores) if top5_scores else 0
        min_score = min(top5_scores) if top5_scores else 0
        score_range = max_score - min_score

        if max_score < 0.04 and score_range < 0.005:
            # Results are uniformly poor -- try broader search
            try:
                # Fallback: broader FTS with individual terms
                words = [w for w in query.lower().split() if len(w) > 3]
                if words:
                    existing_ids = {r.get("id") for r in results if r.get("id")}
                    for word in words[:3]:
                        try:
                            broad_results = search_fts(self.conn, word, limit=10)
                            for br in broad_results:
                                if br.get("id") and br["id"] not in existing_ids:
                                    br["source"] = "fallback_broad"
                                    br["score"] = br.get("score", 0) * 0.5
                                    results.append(br)
                                    existing_ids.add(br["id"])
                        except Exception:
                            logger.debug("Broad FTS fallback failed for term in quality self-check", exc_info=True)
            except Exception:
                logger.debug("Quality self-check fallback failed", exc_info=True)

        return results

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
                "score": r.get("score", 0),
                "source": "fts",
            })

        return cleaned

    # ──────────────────────────────────────────────────────────────────────
    # Stats / teardown
    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return ingestion and search statistics."""
        stats = dict(self.stats)

        # Add live DB stats if connected.
        if self.conn:
            try:
                stats["message_count"] = get_message_count(self.conn)
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
        """Close database connection."""
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
