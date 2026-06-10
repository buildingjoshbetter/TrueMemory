from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, HTTPException

from truememory.dashboard.server import session_index

router = APIRouter(prefix="/api", tags=["sessions"])


@router.get("/sessions")
def list_sessions(
    search: Optional[str] = Query(None),
    project: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    conn = session_index.get_dashboard_conn()

    count = session_index.get_session_count(conn)
    if count == 0:
        session_index.index_sessions(conn, max_sessions=500)

    if search:
        results = session_index.search_sessions(conn, search, limit=limit)
        return {"sessions": results, "total": len(results), "limit": limit, "offset": 0}

    sessions, total = session_index.list_sessions(
        conn, project=project or "", limit=limit, offset=offset
    )
    return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}


@router.get("/sessions/projects")
def list_projects():
    conn = session_index.get_dashboard_conn()
    return session_index.list_projects(conn)


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    conn = session_index.get_dashboard_conn()

    row = conn.execute(
        """SELECT session_id, project_dir, started_at, ended_at,
                  message_count, user_message_count, word_count,
                  summary, version, jsonl_path
           FROM dashboard_sessions WHERE session_id = ?""",
        (session_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": row[0], "project_dir": row[1],
        "started_at": row[2], "ended_at": row[3],
        "message_count": row[4], "user_message_count": row[5],
        "word_count": row[6], "summary": row[7],
        "version": row[8], "jsonl_path": row[9],
    }


@router.get("/sessions/{session_id}/transcript")
def get_transcript(session_id: str):
    conn = session_index.get_dashboard_conn()

    row = conn.execute(
        "SELECT jsonl_path FROM dashboard_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = session_index.load_transcript(row[0])
    return {"messages": messages, "count": len(messages)}


@router.post("/sessions/reindex")
def reindex_sessions():
    conn = session_index.get_dashboard_conn()
    indexed = session_index.index_sessions(conn, max_sessions=2000)
    total = session_index.get_session_count(conn)
    return {"indexed": indexed, "total": total}
