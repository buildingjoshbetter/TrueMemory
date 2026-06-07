"""
TrueMemory Storage Layer
======================

Core data storage using SQLite with WAL mode. Manages the full database schema
and message CRUD operations for the TrueMemory 6-layer memory system.

Schema overview:
    messages        - Core message store (content, sender, recipient, timestamps)
    messages_fts    - FTS5 virtual table for full-text search (auto-synced via triggers)
    entity_profiles - L0 Personality Engram (per-entity traits, style, topics)
    fact_timeline   - L5 contradiction tracking (supersedable facts)
    summaries       - L5 consolidated summaries (daily/weekly/monthly/entity)
"""

import json
import logging
import sqlite3
import time
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Core messages table
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    sender TEXT DEFAULT '',
    recipient TEXT DEFAULT '',
    timestamp TEXT DEFAULT '',
    category TEXT DEFAULT '',
    modality TEXT DEFAULT '',
    episode_id INTEGER DEFAULT NULL,
    emotional_valence REAL DEFAULT 0.0,
    embedding_separation BLOB DEFAULT NULL
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, sender, recipient, category, modality,
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, sender, recipient, category, modality)
    VALUES (new.id, new.content, new.sender, new.recipient, new.category, new.modality);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content, sender, recipient, category, modality)
    VALUES (new.id, new.content, new.sender, new.recipient, new.category, new.modality);
END;

-- Entity profiles (L0 Personality Engram)
CREATE TABLE IF NOT EXISTS entity_profiles (
    entity TEXT PRIMARY KEY,
    message_count INTEGER DEFAULT 0,
    traits TEXT DEFAULT '{}',
    communication_style TEXT DEFAULT '{}',
    topics TEXT DEFAULT '[]',
    relationships TEXT DEFAULT '{}',
    updated_at TEXT
);

-- Entity style vectors (L0 char-n-gram profiles, MEMORIST-L0)
CREATE TABLE IF NOT EXISTS entity_style_vectors (
    entity TEXT PRIMARY KEY,
    vector TEXT,
    message_count INTEGER DEFAULT 0,
    updated_at TEXT
);

-- Fact timeline (L5 contradiction tracking)
CREATE TABLE IF NOT EXISTS fact_timeline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    fact TEXT NOT NULL,
    source_message_id INTEGER,
    timestamp TEXT,
    superseded_by INTEGER,
    entity_scope TEXT DEFAULT '',
    valid_from TEXT DEFAULT '',
    valid_to TEXT DEFAULT '',
    FOREIGN KEY (source_message_id) REFERENCES messages(id)
);

-- Consolidated summaries (L5)
CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period TEXT,
    start_date TEXT,
    end_date TEXT,
    entity TEXT DEFAULT '',
    summary TEXT NOT NULL,
    key_facts TEXT DEFAULT '[]',
    message_ids TEXT DEFAULT '[]',
    created_at TEXT
);

-- Episode boundaries (B1: 6-hour gap heuristic grouping)
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    message_count INTEGER DEFAULT 0,
    summary TEXT DEFAULT ''
);

-- Landmark events (E3: job changes, moves, launches, breakups)
CREATE TABLE IF NOT EXISTS landmark_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT DEFAULT '',
    related_entities TEXT DEFAULT '[]',
    source_message_id INTEGER,
    FOREIGN KEY (source_message_id) REFERENCES messages(id)
);

-- Causal edges (D2: forward chains and backward cause scanning)
CREATE TABLE IF NOT EXISTS causal_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cause_msg_id INTEGER NOT NULL,
    effect_msg_id INTEGER NOT NULL,
    relationship TEXT DEFAULT '',
    confidence REAL DEFAULT 0.0,
    FOREIGN KEY (cause_msg_id) REFERENCES messages(id),
    FOREIGN KEY (effect_msg_id) REFERENCES messages(id)
);

-- Entity relationships (E2: Dunbar hierarchy)
CREATE TABLE IF NOT EXISTS entity_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_a TEXT NOT NULL,
    entity_b TEXT NOT NULL,
    relationship_type TEXT DEFAULT '',
    strength REAL DEFAULT 0.0,
    dunbar_layer TEXT DEFAULT '',
    last_interaction TEXT DEFAULT ''
);

-- Embedder identity / schema version (prevents silent quality
-- collapse when a tier switch produces matching dims but different vector
-- spaces — e.g. Model2Vec 256d → Qwen3 256d). Writers: build_vectors,
-- build_separation_vectors. Readers: init_vec_table, engine.open().
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT
);

-- Surprise scores for L5 predictive layer (MEMORIST)
CREATE TABLE IF NOT EXISTS surprise_scores (
    message_id INTEGER PRIMARY KEY,
    surprise    REAL NOT NULL DEFAULT 0.0,
    fact_count  INTEGER NOT NULL DEFAULT 0,
    new_fact_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

-- Clustering (HDBSCAN episode clusters)
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

CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender);
CREATE INDEX IF NOT EXISTS idx_cluster_id ON message_clusters(cluster_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_episode_id ON messages(episode_id);
CREATE INDEX IF NOT EXISTS idx_messages_category ON messages(category);
CREATE INDEX IF NOT EXISTS idx_fact_timeline_subject ON fact_timeline(subject);
CREATE INDEX IF NOT EXISTS idx_summaries_entity ON summaries(entity);
CREATE INDEX IF NOT EXISTS idx_summaries_period ON summaries(period);
CREATE INDEX IF NOT EXISTS idx_entity_relationships_a ON entity_relationships(entity_a);
CREATE INDEX IF NOT EXISTS idx_landmark_events_timestamp ON landmark_events(timestamp);

-- Vector cache registry (tier-switch: tracks per-tier-group vector table state)
CREATE TABLE IF NOT EXISTS vector_cache_registry (
    tier_group TEXT PRIMARY KEY,
    vec_table TEXT NOT NULL,
    sep_table TEXT NOT NULL,
    last_embedded_id INTEGER DEFAULT 0,
    vector_count INTEGER DEFAULT 0,
    model_name TEXT,
    embedding_dim INTEGER DEFAULT 256,
    last_updated REAL,
    created REAL
);

-- Rebuild status (tier-switch: tracks async rebuild progress)
CREATE TABLE IF NOT EXISTS rebuild_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tier_group TEXT NOT NULL,
    target_tier TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    action TEXT,
    total_messages INTEGER DEFAULT 0,
    processed_messages INTEGER DEFAULT 0,
    progress_pct REAL DEFAULT 0,
    eta_seconds REAL DEFAULT 0,
    batch_size INTEGER DEFAULT 0,
    throughput_ips REAL DEFAULT 0,
    ram_pct REAL DEFAULT 0,
    pressure REAL DEFAULT 0,
    error TEXT,
    started_at REAL,
    completed_at REAL,
    backup_path TEXT,
    last_heartbeat REAL
);
CREATE INDEX IF NOT EXISTS idx_rebuild_status_active
    ON rebuild_status(tier_group, status);
"""


# single source of truth for the sqlite busy_timeout pragma.
# Pre-fix, create_db used 5000ms and pipeline._set_busy_timeout used
# 10_000ms — same DB, asymmetric lock-wait behaviour that surfaced as
# sporadic "database is locked" errors under concurrent ingest + MCP
# search load. Both paths now import this constant.
DEFAULT_BUSY_TIMEOUT_MS = 10_000


# ---------------------------------------------------------------------------
# Database creation
# ---------------------------------------------------------------------------

_EXPECTED_COLUMNS = {
    "recipient": "TEXT DEFAULT ''",
    "category": "TEXT DEFAULT ''",
    "modality": "TEXT DEFAULT ''",
    "episode_id": "INTEGER DEFAULT NULL",
    "emotional_valence": "REAL DEFAULT 0.0",
    "embedding_separation": "BLOB DEFAULT NULL",
}


def _backup_database(db_path: Path) -> Path | None:
    """Create a complete backup of the database including WAL/SHM files.

    Returns the backup path on success, None on failure.
    """
    import shutil
    import uuid

    suffix = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    backup = Path(f"{db_path}.backup-pre-migration-{suffix}")

    if backup.exists():
        return None

    try:
        shutil.copy2(str(db_path), str(backup))
        for ext in ("-wal", "-shm"):
            wal_path = Path(f"{db_path}{ext}")
            if wal_path.exists():
                shutil.copy2(str(wal_path), str(backup) + ext)
        log.info("Legacy DB backup created: %s", backup)
        return backup
    except Exception as e:
        log.warning("Could not back up database before migration: %s", e)
        return None


def _migrate_messages_schema(conn: sqlite3.Connection, db_path: str | Path) -> None:
    """Add missing columns to an existing messages table (legacy DB upgrade).

    Safety measures:
    - Creates a complete backup (DB + WAL + SHM) before any changes
    - If backup fails, migration is skipped entirely
    - All ALTER TABLE statements run inside a single transaction
    - If any ALTER fails, the entire transaction is rolled back
    - Only runs if the messages table already exists and is missing columns
    """
    try:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
    except Exception:
        return
    if not existing:
        return

    missing = {col: typedef for col, typedef in _EXPECTED_COLUMNS.items() if col not in existing}
    if not missing:
        return

    if str(db_path) != ":memory:":
        backup = _backup_database(Path(str(db_path)))
        if backup is None:
            log.warning("Skipping legacy migration — backup failed")
            return

    try:
        conn.execute("BEGIN IMMEDIATE")
        for col, typedef in missing.items():
            conn.execute(f"ALTER TABLE messages ADD COLUMN {col} {typedef}")
            log.info("Migrated legacy DB: added column messages.%s", col)
        conn.execute("COMMIT")
    except Exception as e:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        log.warning("Legacy migration failed and was rolled back: %s", e)


def create_db(db_path: str | Path) -> sqlite3.Connection:
    """
    Create (or open) a TrueMemory database with the full schema.

    If the database has an older schema (pre-v0.5), missing columns are
    added automatically via ALTER TABLE. A timestamped backup is created
    before any migration.

    Enables WAL mode for concurrent read/write performance and executes all
    CREATE TABLE / CREATE VIRTUAL TABLE / CREATE TRIGGER statements.

    Args:
        db_path: Filesystem path for the SQLite database file.
                 Use \":memory:\" for an in-memory database.

    Returns:
        An open ``sqlite3.Connection`` with row_factory left at default
        (callers choose their own access pattern).
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA mmap_size=268435456")
    _migrate_messages_schema(conn, db_path)
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Message loading
# ---------------------------------------------------------------------------

def bulk_replace_messages(conn: sqlite3.Connection, messages: list[dict]) -> int:
    """
    Replace every row in ``messages`` with the provided list (destructive).

    **DESTRUCTIVE:** Clears all existing message data (messages + FTS index)
    before inserting, so the database reflects exactly the provided list
    afterwards. If you want to append without wiping, use
    :func:`insert_message` per-row.

    Each dict in *messages* should have at minimum a ``content`` key.
    Optional keys: ``sender``, ``recipient``, ``timestamp``, ``category``,
    ``modality``.

    Args:
        conn:     Open database connection (from :func:`create_db`).
        messages: List of message dicts to insert.

    Returns:
        Number of messages inserted.
    """
    conn.execute("DELETE FROM messages")
    # The DELETE trigger handles FTS cleanup row-by-row, but if the table was
    # freshly created (no rows yet) that is a no-op.  For safety, also rebuild
    # the FTS index after bulk delete.
    try:
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    except sqlite3.OperationalError:
        pass  # FTS table might already be clean

    conn.executemany(
        """INSERT INTO messages
           (content, sender, recipient, timestamp, category, modality)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (
                msg["content"],
                msg.get("sender", ""),
                msg.get("recipient", ""),
                msg.get("timestamp", ""),
                msg.get("category", ""),
                msg.get("modality", ""),
            )
            for msg in messages
        ],
    )

    conn.commit()
    return len(messages)


def load_messages(conn: sqlite3.Connection, messages: list[dict]) -> int:
    """Deprecated alias for :func:`bulk_replace_messages`.

    the original name ``load_messages`` paralleled
    ``insert_message`` (non-destructive) but actually WIPES the table
    before inserting. Callers writing ``load_messages(conn, [new_msg])``
    believing it appended silently destroyed their DB. The destructive
    semantics now live under ``bulk_replace_messages``; this alias is
    preserved for one release with a ``DeprecationWarning``.
    """
    import warnings
    warnings.warn(
        "`load_messages` is a deprecated alias for "
        "`bulk_replace_messages` (which makes the destructive semantics "
        "explicit). This alias will be removed in a future release — "
        "migrate to `bulk_replace_messages` for the same behaviour, or "
        "use `insert_message` per-row if you actually want to append.",
        DeprecationWarning,
        stacklevel=2,
    )
    return bulk_replace_messages(conn, messages)


def load_messages_from_file(conn: sqlite3.Connection, json_path: str | Path) -> int:
    """
    Load messages from a JSON file into the database (destructive — wipes first).

    The file must contain a JSON array of message objects (same format as
    :func:`bulk_replace_messages`).

    Args:
        conn:      Open database connection.
        json_path: Path to the JSON file.

    Returns:
        Number of messages inserted.
    """
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as f:
        messages = json.load(f)

    # Use the non-deprecated name internally so we don't emit our own
    # DeprecationWarning to users who never called `load_messages`.
    return bulk_replace_messages(conn, messages)


# ---------------------------------------------------------------------------
# Message retrieval (CRUD reads)
# ---------------------------------------------------------------------------

def _row_to_dict(row: tuple) -> dict:
    """Convert a raw row tuple to a message dict."""
    return {
        "id": row[0],
        "content": row[1],
        "sender": row[2],
        "recipient": row[3],
        "timestamp": row[4],
        "category": row[5],
        "modality": row[6],
    }


_SELECT_COLS = "id, content, sender, recipient, timestamp, category, modality"


def get_message(conn: sqlite3.Connection, msg_id: int) -> dict | None:
    """
    Retrieve a single message by its primary key.

    Args:
        conn:   Open database connection.
        msg_id: The message ID to look up.

    Returns:
        Message dict, or ``None`` if not found.
    """
    row = conn.execute(
        f"SELECT {_SELECT_COLS} FROM messages WHERE id = ?", (msg_id,)
    ).fetchone()
    return _row_to_dict(row) if row else None


def get_messages_by_sender(conn: sqlite3.Connection, sender: str) -> list[dict]:
    """
    Retrieve all messages from a specific sender, ordered by timestamp.

    Args:
        conn:   Open database connection.
        sender: Sender name to filter on (case-sensitive).

    Returns:
        List of message dicts.
    """
    rows = conn.execute(
        f"SELECT {_SELECT_COLS} FROM messages WHERE sender = ? ORDER BY timestamp",
        (sender,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_messages_in_range(
    conn: sqlite3.Connection,
    after: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """
    Retrieve messages within a timestamp range.

    Timestamps are compared as ISO-8601 strings (lexicographic ordering).
    Either bound may be omitted for an open-ended range.

    Args:
        conn:   Open database connection.
        after:  Inclusive lower bound (e.g. ``"2025-01-01"``).
        before: Inclusive upper bound (e.g. ``"2025-12-31"``).

    Returns:
        List of message dicts ordered by timestamp.
    """
    clauses: list[str] = []
    params: list[str] = []

    if after is not None:
        clauses.append("timestamp >= ?")
        params.append(after)
    if before is not None:
        clauses.append("timestamp <= ?")
        params.append(before)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT {_SELECT_COLS} FROM messages{where} ORDER BY timestamp",
        params,
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_all_senders(conn: sqlite3.Connection) -> list[str]:
    """
    Get a sorted list of unique sender names in the database.

    Returns:
        Sorted list of sender strings (empty strings excluded).
    """
    rows = conn.execute(
        "SELECT DISTINCT sender FROM messages WHERE sender != '' ORDER BY sender"
    ).fetchall()
    return [r[0] for r in rows]


def get_message_count(conn: sqlite3.Connection) -> int:
    """
    Return the total number of messages in the database.
    """
    row = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# Single-message insert / delete (production API)
# ---------------------------------------------------------------------------

def insert_message(conn: sqlite3.Connection, msg: dict) -> int:
    """
    Insert a single message without clearing existing data.

    Unlike :func:`bulk_replace_messages`, this appends to the database —
    it does NOT wipe existing messages.  The FTS5 INSERT trigger keeps the full-text
    index in sync automatically.

    Args:
        conn: Open database connection (from :func:`create_db`).
        msg:  Message dict.  Must have ``content``; optional keys:
              ``sender``, ``recipient``, ``timestamp``, ``category``,
              ``modality``.

    Returns:
        The new row's integer ID.
    """
    content = msg.get("content", "")
    if not content or not content.strip():
        raise ValueError("content cannot be empty or whitespace-only")
    cursor = conn.execute(
        """INSERT INTO messages
           (content, sender, recipient, timestamp, category, modality)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            msg["content"],
            msg.get("sender", ""),
            msg.get("recipient", ""),
            msg.get("timestamp", ""),
            msg.get("category", ""),
            msg.get("modality", ""),
        ),
    )
    return cursor.lastrowid


def delete_message(conn: sqlite3.Connection, msg_id: int) -> bool:
    """
    Delete a single message and its vector embedding.

    The FTS5 DELETE trigger automatically removes the full-text index entry.
    The vector embedding in ``vec_messages`` is also removed if the table
    exists.

    Args:
        conn:   Open database connection.
        msg_id: The message ID to delete.

    Returns:
        True if a row was deleted, False if the ID was not found.
    """
    cursor = conn.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
    deleted = cursor.rowcount > 0

    if deleted:
        for tbl in (
            "vec_messages", "vec_messages_edge", "vec_messages_basepro",
        ):
            try:
                conn.execute(f"DELETE FROM {tbl} WHERE rowid = ?", (msg_id,))
            except sqlite3.OperationalError:
                pass
        for tbl in (
            "vec_messages_sep", "vec_messages_sep_edge",
            "vec_messages_sep_basepro",
        ):
            try:
                conn.execute(f"DELETE FROM {tbl} WHERE rowid = ?", (msg_id,))
            except sqlite3.OperationalError:
                pass

        for tbl, col in (
            ("fact_timeline", "source_message_id"),
            ("landmark_events", "source_message_id"),
            ("surprise_scores", "message_id"),
        ):
            try:
                conn.execute(f"DELETE FROM {tbl} WHERE {col} = ?", (msg_id,))
            except sqlite3.OperationalError:
                pass
        for col in ("cause_msg_id", "effect_msg_id"):
            try:
                conn.execute(f"DELETE FROM causal_edges WHERE {col} = ?", (msg_id,))
            except sqlite3.OperationalError:
                pass

    conn.commit()
    return deleted


def update_message(conn: sqlite3.Connection, msg_id: int, **fields) -> bool:
    """
    Update fields on an existing message.

    Only the provided keyword arguments are changed.  Valid field names:
    ``content``, ``sender``, ``recipient``, ``timestamp``, ``category``,
    ``modality``.

    The AFTER UPDATE trigger on ``messages`` automatically keeps the
    FTS5 index in sync.

    Args:
        conn:    Open database connection.
        msg_id:  The message ID to update.
        **fields: Column names and new values.

    Returns:
        True if the row was updated, False if ID not found.
    """
    allowed = {"content", "sender", "recipient", "timestamp", "category", "modality"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [msg_id]
    cursor = conn.execute(f"UPDATE messages SET {set_clause} WHERE id = ?", values)

    # The AFTER UPDATE trigger on messages keeps FTS5 in sync automatically.
    conn.commit()
    return cursor.rowcount > 0
