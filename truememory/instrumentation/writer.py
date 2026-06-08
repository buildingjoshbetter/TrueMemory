"""
Telemetry writer — writes to a **separate** instrumentation.db.

CRITICAL: This module must NEVER import or reference TRUEMEMORY_DB_PATH,
TRUEMEMORY_DB, or memories.db.  All telemetry goes to instrumentation.db.

Includes:
- Automatic 7-day retention pruning (on install + every 1000 emits)
- Wall-clock fallback prune (if last prune was >24h ago)
- busy_timeout for multi-process safety
- Index on ts column for efficient pruning
- No raw content or query text stored
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

from truememory.instrumentation.log import dlog

_DEFAULT_INSTRUMENTATION_DB = Path.home() / ".truememory" / "instrumentation.db"

_RETAIN_DAYS = 7
_PRUNE_EVERY = 1000
_PRUNE_WALL_CLOCK_SEC = 86400  # 24 hours

# Module-level state
_emit_count = 0
_last_prune_time: float = 0.0
_conn_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_schema_created = False


def _resolve_db_path() -> str:
    """Resolve the instrumentation DB path.  NEVER returns memories.db."""
    override = os.environ.get("TRUEMEMORY_INSTRUMENTATION_DB", "").strip()
    return override or str(_DEFAULT_INSTRUMENTATION_DB)


def _get_connection() -> sqlite3.Connection:
    """Get or create the instrumentation DB connection (thread-safe)."""
    global _conn, _schema_created
    if _conn is not None and _schema_created:
        return _conn
    with _conn_lock:
        if _conn is not None and _schema_created:
            return _conn
        db_path = _resolve_db_path()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=1000")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                signal TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry(ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telemetry_signal ON telemetry(signal)"
        )
        conn.commit()
        _conn = conn
        _schema_created = True
        return conn


def emit(signal: str, data: dict | None = None) -> None:
    """Write a telemetry row.  Swallows all exceptions."""
    try:
        import json
        conn = _get_connection()
        now = time.time()
        payload = json.dumps(data or {}, default=str)
        conn.execute(
            "INSERT INTO telemetry (ts, signal, data) VALUES (?, ?, ?)",
            (now, signal, payload),
        )
        conn.commit()
        _prune_if_needed(conn)
    except Exception:
        dlog("emit failed for signal=%s", signal)


def _prune_if_needed(conn: sqlite3.Connection) -> None:
    """Prune old rows periodically (counter-based + wall-clock fallback)."""
    global _emit_count, _last_prune_time
    _emit_count += 1
    now = time.time()
    should_prune = (
        _emit_count % _PRUNE_EVERY == 0
        or (_last_prune_time > 0 and (now - _last_prune_time) > _PRUNE_WALL_CLOCK_SEC)
    )
    if not should_prune:
        return
    _do_prune(conn)


def _do_prune(conn: sqlite3.Connection) -> None:
    """Execute the actual prune."""
    global _last_prune_time
    cutoff = time.time() - (_RETAIN_DAYS * 86400)
    try:
        cur = conn.execute("DELETE FROM telemetry WHERE ts < ?", (cutoff,))
        deleted = cur.rowcount
        conn.commit()
        _last_prune_time = time.time()
        if deleted > 0:
            dlog("pruned %d telemetry rows older than %d days", deleted, _RETAIN_DAYS)
    except sqlite3.Error:
        pass


def prune_now() -> None:
    """Force an immediate prune.  Called on install()."""
    try:
        conn = _get_connection()
        _do_prune(conn)
    except Exception:
        pass


def close() -> None:
    """Close the instrumentation DB connection."""
    global _conn, _schema_created
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
            _schema_created = False


def reset() -> None:
    """Full reset for testing."""
    global _emit_count, _last_prune_time
    close()
    _emit_count = 0
    _last_prune_time = 0.0
