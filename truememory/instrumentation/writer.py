"""SQLite writer for the ``telemetry`` semantic-signal table.

This module adds a structured signal sink: a single ``telemetry`` table inside
the user's ``memories.db``. A dashboard layer can JOIN ``telemetry.memory_id``
to ``messages.id`` and aggregate per-signal counts in pure SQL — no log parsing
required.

Schema (kept stable so external dashboard mirrors can read it)::

    telemetry(
      id           INTEGER PRIMARY KEY,
      ts           REAL    NOT NULL,
      pid          INTEGER,
      signal       TEXT    NOT NULL,
      memory_id    INTEGER,
      value_num    REAL,
      value_text   TEXT,
      context_json TEXT
    )
    + index idx_telemetry_signal_ts ON telemetry(signal, ts)
    + index idx_telemetry_memory_id ON telemetry(memory_id)

Design constraints:

- **Independent connection.** The writer never reuses the MCP server's SQLite
  connection. The server holds its connection on the asyncio event-loop
  thread, and we cannot risk contention or cross-thread cursor use.
- **Schema migration on first write.** ``CREATE TABLE IF NOT EXISTS`` runs on
  the first ``emit()`` — cheap on the SQLite C layer and avoids a separate
  install step.
- **Exception-swallowing.** A telemetry-write failure must NEVER bring down the
  MCP server. Every error is logged via ``dlog`` and dropped.
- **Disabled by default.** Gated on ``TRUEMEMORY_INSTRUMENTATION`` (the same
  flag that gates the whole overlay). Production users are unaffected.
- **No surprise side effects.** Does not call ``PRAGMA journal_mode=WAL`` or
  ``BEGIN``/``COMMIT`` explicitly — relies on SQLite autocommit so the writer
  never interferes with the MCP server's transaction state.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from truememory.instrumentation.log import dlog, is_enabled

# Path resolution follows TrueMemory's own convention: TRUEMEMORY_DB_PATH > the
# legacy TRUEMEMORY_DB env var > ~/.truememory/memories.db. Resolved live on
# each open so test fixtures that set TRUEMEMORY_DB_PATH are honored.
_DEFAULT_DB_PATH = Path.home() / ".truememory" / "memories.db"

# Lazy module-level connection. We open on first emit() so a process that never
# emits never touches the DB. Re-opened automatically if the file moves between
# calls (rare; supports test fixtures).
_conn: sqlite3.Connection | None = None
_conn_path: str | None = None
_conn_lock = threading.Lock()
_schema_migrated = False

_TELEMETRY_DDL = """
CREATE TABLE IF NOT EXISTS telemetry (
  id           INTEGER PRIMARY KEY,
  ts           REAL    NOT NULL,
  pid          INTEGER,
  signal       TEXT    NOT NULL,
  memory_id    INTEGER,
  value_num    REAL,
  value_text   TEXT,
  context_json TEXT
);
"""

_TELEMETRY_INDICES = (
    "CREATE INDEX IF NOT EXISTS idx_telemetry_signal_ts ON telemetry(signal, ts);",
    "CREATE INDEX IF NOT EXISTS idx_telemetry_memory_id ON telemetry(memory_id);",
)


def _resolve_db_path() -> str:
    """Return the canonical memories.db path (env override or default)."""
    override = (
        os.environ.get("TRUEMEMORY_DB_PATH")
        or os.environ.get("TRUEMEMORY_DB")
        or ""
    ).strip()
    return override or str(_DEFAULT_DB_PATH)


def _get_conn() -> sqlite3.Connection | None:
    """Open (or reopen) the instrumentation-owned SQLite connection.

    Returns None if the DB file's parent directory does not exist — in that
    case the MCP server hasn't been started yet, so there's nothing to write
    telemetry for.
    """
    global _conn, _conn_path, _schema_migrated
    path = _resolve_db_path()
    if _conn is not None and _conn_path == path:
        return _conn

    with _conn_lock:
        # Re-check inside the lock — another thread may have opened it.
        if _conn is not None and _conn_path == path:
            return _conn
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
            _schema_migrated = False

        parent = Path(path).parent
        if not parent.exists():
            return None
        try:
            # check_same_thread=False so emit() can fire from any thread
            # (worker threads under asyncio.to_thread, the WAL-checkpoint hook
            # in patch.py, background drainers).
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.execute("PRAGMA busy_timeout=5000")
            _conn = conn
            _conn_path = path
            dlog(f"telemetry writer opened conn path={path}")
            return _conn
        except sqlite3.Error as exc:
            dlog(
                f"telemetry writer open FAILED path={path} "
                f"{type(exc).__name__}: {exc}"
            )
            return None


def _ensure_schema(conn: sqlite3.Connection) -> bool:
    """Run the DDL idempotently. Cached so we don't re-run on every emit."""
    global _schema_migrated
    if _schema_migrated:
        return True
    try:
        with conn:
            conn.execute(_TELEMETRY_DDL)
            for ddl in _TELEMETRY_INDICES:
                conn.execute(ddl)
        _schema_migrated = True
        dlog("telemetry writer schema migrated (CREATE TABLE/INDEX IF NOT EXISTS)")
        return True
    except sqlite3.Error as exc:
        dlog(
            f"telemetry writer schema migration FAILED "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def emit(
    signal: str,
    *,
    memory_id: int | None = None,
    value_num: float | None = None,
    value_text: str | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Append one row to the ``telemetry`` table.

    Args:
        signal: One of the documented signal names (``salience``, ``category``,
            ``gate_decision``, ``surprise``, ``search_distance``,
            ``memory_returned``, ``user_forget``) or a model-lifecycle signal
            (``preload_start``, ``preload_complete``, ``model_unload``,
            ``reranker_degraded``, ``instrumentation_start``). Free-form for
            future expansion.
        memory_id: Foreign key into ``messages.id`` when the signal relates to a
            specific memory. Nullable.
        value_num: Numeric payload — score, latency in ms, distance, rank.
            Nullable.
        value_text: Categorical payload — reason code, model name, requested
            text. Nullable.
        context: Arbitrary structured payload serialized as JSON into
            ``context_json``. Nullable. Use sparingly — column-typed payloads
            (``value_num`` / ``value_text``) query faster.

    No-op when ``TRUEMEMORY_INSTRUMENTATION`` is unset. Errors are swallowed.
    """
    if not is_enabled():
        return
    conn = _get_conn()
    if conn is None:
        return
    if not _ensure_schema(conn):
        return

    context_json = None
    if context:
        try:
            context_json = json.dumps(context, default=str, sort_keys=True)
        except (TypeError, ValueError) as exc:
            dlog(
                f"telemetry writer json encode failed signal={signal} "
                f"{type(exc).__name__}: {exc}"
            )
            context_json = None

    try:
        with conn:
            conn.execute(
                "INSERT INTO telemetry "
                "(ts, pid, signal, memory_id, value_num, value_text, context_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    os.getpid(),
                    signal,
                    memory_id,
                    value_num,
                    value_text,
                    context_json,
                ),
            )
    except sqlite3.Error as exc:
        dlog(
            f"telemetry writer emit FAILED signal={signal} "
            f"memory_id={memory_id} {type(exc).__name__}: {exc}"
        )


def close() -> None:
    """Close the instrumentation-owned connection. Called at process exit by an
    atexit hook ``install()`` registers. Safe to call multiple times."""
    global _conn, _conn_path, _schema_migrated
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
            _conn_path = None
            _schema_migrated = False
