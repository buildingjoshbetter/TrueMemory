from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

_CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
_CLAUDE_SESSIONS = Path.home() / ".claude" / "sessions"
_DASHBOARD_DB = Path.home() / ".truememory" / "dashboard.db"

_dash_conn: sqlite3.Connection | None = None


def get_dashboard_conn() -> sqlite3.Connection:
    global _dash_conn
    if _dash_conn is not None:
        return _dash_conn
    _DASHBOARD_DB.parent.mkdir(parents=True, exist_ok=True)
    _dash_conn = sqlite3.connect(str(_DASHBOARD_DB), timeout=15, check_same_thread=False)
    _dash_conn.execute("PRAGMA journal_mode=WAL")
    _dash_conn.execute("PRAGMA busy_timeout=15000")
    ensure_session_table(_dash_conn)
    return _dash_conn

_SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS dashboard_sessions (
    session_id TEXT PRIMARY KEY,
    project_dir TEXT,
    started_at TEXT,
    ended_at TEXT,
    message_count INTEGER DEFAULT 0,
    user_message_count INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    summary TEXT DEFAULT '',
    version TEXT DEFAULT '',
    jsonl_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_ds_started ON dashboard_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_ds_project ON dashboard_sessions(project_dir);
"""


def ensure_session_table(conn: sqlite3.Connection):
    for stmt in _SESSION_SCHEMA.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()


def get_session_count(conn: sqlite3.Connection) -> int:
    try:
        return conn.execute("SELECT COUNT(*) FROM dashboard_sessions").fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def index_sessions(conn: sqlite3.Connection, max_sessions: int = 0) -> int:
    ensure_session_table(conn)

    existing = set()
    for row in conn.execute("SELECT session_id FROM dashboard_sessions").fetchall():
        existing.add(row[0])

    session_meta: dict[str, dict] = {}
    if _CLAUDE_SESSIONS.is_dir():
        for f in _CLAUDE_SESSIONS.iterdir():
            if f.suffix == ".json":
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    sid = data.get("sessionId", "")
                    if sid:
                        session_meta[sid] = data
                except (json.JSONDecodeError, OSError):
                    continue

    indexed = 0
    if not _CLAUDE_PROJECTS.is_dir():
        return indexed

    for project_dir in sorted(_CLAUDE_PROJECTS.iterdir()):
        if not project_dir.is_dir():
            continue

        project_name = project_dir.name
        jsonl_files = sorted(project_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

        for jf in jsonl_files:
            session_id = jf.stem
            if session_id in existing:
                continue

            try:
                info = _parse_session_jsonl(jf, session_meta.get(session_id, {}))
            except Exception:
                continue

            info["project_dir"] = _decode_project_name(project_name)
            info["jsonl_path"] = str(jf)
            info["session_id"] = session_id

            conn.execute(
                """INSERT OR IGNORE INTO dashboard_sessions
                   (session_id, project_dir, started_at, ended_at,
                    message_count, user_message_count, word_count,
                    summary, version, jsonl_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    info["session_id"],
                    info["project_dir"],
                    info.get("started_at", ""),
                    info.get("ended_at", ""),
                    info.get("message_count", 0),
                    info.get("user_message_count", 0),
                    info.get("word_count", 0),
                    info.get("summary", ""),
                    info.get("version", ""),
                    info["jsonl_path"],
                ),
            )
            indexed += 1
            existing.add(session_id)

            if max_sessions and indexed >= max_sessions:
                conn.commit()
                return indexed

    conn.commit()
    return indexed


def _parse_session_jsonl(path: Path, meta: dict) -> dict:
    messages = []
    first_ts = None
    last_ts = None
    version = meta.get("version", "")
    user_msgs = 0
    word_count = 0
    summary_parts = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f):
            if line_no > 2000:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")
            ts = msg.get("timestamp", "")

            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            if not version:
                version = msg.get("version", "")

            if msg_type == "user":
                user_msgs += 1
                content = _extract_text(msg)
                if content:
                    word_count += len(content.split())
                    if len(summary_parts) < 2 and not _is_internal_message(content):
                        summary_parts.append(content[:150])

            elif msg_type == "assistant":
                content = _extract_text(msg)
                if content:
                    word_count += len(content.split())

            messages.append(msg_type)

    summary = " | ".join(summary_parts) if summary_parts else ""
    summary = _clean_summary(summary)

    started_at = first_ts or ""
    if not started_at and meta.get("startedAt"):
        from datetime import datetime, timezone
        try:
            started_at = datetime.fromtimestamp(
                meta["startedAt"] / 1000, tz=timezone.utc
            ).isoformat()
        except (ValueError, OSError):
            pass

    return {
        "started_at": started_at,
        "ended_at": last_ts or "",
        "message_count": len(messages),
        "user_message_count": user_msgs,
        "word_count": word_count,
        "summary": summary[:500],
        "version": version,
    }


def _extract_text(msg: dict) -> str:
    content = msg.get("content", msg.get("message", ""))
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        role_content = content.get("content", "")
        if isinstance(role_content, str):
            return role_content
        if isinstance(role_content, list):
            parts = []
            for block in role_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return " ".join(parts)
    return ""


_INTERNAL_PREFIXES = (
    "[[TRUEMEMORY_INTERNAL",
    "<command-message>",
    "<command-name>",
    "<local-command-stdout>",
    "[[TRUEMEMORY",
    "You are a memory extraction system",
    "/loop",
)

_INTERNAL_SUBSTRINGS = (
    "[[TRUEMEMORY",
    "truememory_store",
    "truememory_search",
    "<command-message>",
    "<command-name>",
    "<local-command-stdout>",
)


def _is_internal_message(content: str) -> bool:
    stripped = content.lstrip()
    for prefix in _INTERNAL_PREFIXES:
        if stripped.startswith(prefix):
            return True
    for sub in _INTERNAL_SUBSTRINGS:
        if sub in stripped:
            return True
    return False


def _clean_summary(text: str) -> str:
    import re
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _decode_project_name(name: str) -> str:
    if name == "-":
        return "Home"
    decoded = name.replace("-", "/")
    parts = [p for p in decoded.split("/") if p]
    return parts[-1] if parts else decoded


def load_transcript(jsonl_path: str) -> list[dict]:
    path = Path(jsonl_path)
    if not path.exists():
        return []

    messages = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")
            if msg_type not in ("user", "assistant"):
                continue

            text = _extract_text(msg)
            if not text or _is_internal_message(text):
                continue

            ts = msg.get("timestamp", "")
            uuid = msg.get("uuid", "")

            messages.append({
                "type": msg_type,
                "content": text[:5000],
                "timestamp": ts,
                "uuid": uuid,
            })

    return messages


def search_sessions(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    ensure_session_table(conn)
    words = query.lower().split()
    if not words:
        return []

    like_clauses = []
    params = []
    for w in words[:5]:
        like_clauses.append("(LOWER(summary) LIKE ? OR LOWER(project_dir) LIKE ?)")
        params.extend([f"%{w}%", f"%{w}%"])

    sql = f"""
        SELECT session_id, project_dir, started_at, ended_at,
               message_count, user_message_count, word_count, summary, version
        FROM dashboard_sessions
        WHERE {' AND '.join(like_clauses)}
        ORDER BY started_at DESC
        LIMIT ?
    """
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def list_sessions(
    conn: sqlite3.Connection,
    project: str = "",
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    ensure_session_table(conn)

    where_clauses = ["user_message_count > 0", "summary != ''"]
    params: list = []
    if project:
        where_clauses.append("project_dir = ?")
        params.append(project)

    where = "WHERE " + " AND ".join(where_clauses)

    total = conn.execute(f"SELECT COUNT(*) FROM dashboard_sessions {where}", params).fetchone()[0]

    sql = f"""
        SELECT session_id, project_dir, started_at, ended_at,
               message_count, user_message_count, word_count, summary, version
        FROM dashboard_sessions {where}
        ORDER BY started_at DESC
        LIMIT ? OFFSET ?
    """
    rows = conn.execute(sql, params + [limit, offset]).fetchall()
    return [_row_to_dict(r) for r in rows], total


def list_projects(conn: sqlite3.Connection) -> list[str]:
    ensure_session_table(conn)
    rows = conn.execute(
        "SELECT DISTINCT project_dir FROM dashboard_sessions ORDER BY project_dir"
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def _row_to_dict(row) -> dict:
    return {
        "session_id": row[0],
        "project_dir": row[1],
        "started_at": row[2],
        "ended_at": row[3],
        "message_count": row[4],
        "user_message_count": row[5],
        "word_count": row[6],
        "summary": row[7],
        "version": row[8],
    }
