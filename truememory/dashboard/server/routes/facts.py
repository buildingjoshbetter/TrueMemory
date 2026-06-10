from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from truememory.dashboard.server.deps import get_engine

router = APIRouter(prefix="/api", tags=["facts"])


@router.get("/facts")
def list_facts(
    subject: Optional[str] = Query(None),
    show_superseded: bool = Query(False),
    limit: int = Query(200, ge=1, le=1000),
):
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    where_clauses = []
    params: list = []

    if subject:
        where_clauses.append("subject LIKE ?")
        params.append(f"%{subject}%")

    if not show_superseded:
        where_clauses.append("superseded_by IS NULL")

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    rows = conn.execute(
        f"""SELECT id, subject, fact, source_message_id, timestamp,
                  superseded_by, entity_scope, valid_from, valid_to
           FROM fact_timeline {where_sql}
           ORDER BY subject, timestamp DESC
           LIMIT ?""",
        params + [limit],
    ).fetchall()

    if not rows:
        # Synthetic facts fallback from memories
        import re

        mem_rows = conn.execute(
            """SELECT id, category, content, timestamp
               FROM messages
               WHERE category IN ('preference', 'decision', 'personal', 'technical', 'correction')
               ORDER BY category, id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

        facts = []
        for r in mem_rows:
            content = r[2]
            content = re.sub(r'^\[[\w_]+\]\s*', '', content)
            cat = r[1] or 'uncategorized'
            facts.append({
                "id": r[0],
                "subject": cat.capitalize(),
                "fact": content,
                "source_message_id": r[0],
                "timestamp": r[3],
                "superseded_by": None,
                "entity_scope": "",
                "valid_from": r[3] or "",
                "valid_to": "",
                "is_current": True,
            })

        subjects_set = sorted(set(f["subject"] for f in facts))
        return {
            "facts": facts,
            "total": len(facts),
            "subjects": subjects_set,
        }

    facts = [
        {
            "id": r[0], "subject": r[1], "fact": r[2],
            "source_message_id": r[3], "timestamp": r[4],
            "superseded_by": r[5], "entity_scope": r[6],
            "valid_from": r[7], "valid_to": r[8],
            "is_current": r[5] is None,
        }
        for r in rows
    ]

    subjects = conn.execute(
        "SELECT DISTINCT subject FROM fact_timeline ORDER BY subject"
    ).fetchall()

    return {
        "facts": facts,
        "total": len(facts),
        "subjects": [s[0] for s in subjects],
    }


@router.get("/facts/contradictions")
def get_contradictions():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    try:
        rows = conn.execute(
            """SELECT f1.id, f1.subject, f1.fact, f1.timestamp,
                      f2.id, f2.fact, f2.timestamp
               FROM fact_timeline f1
               JOIN fact_timeline f2 ON f1.subject = f2.subject
                    AND f1.id < f2.id
                    AND f1.superseded_by IS NULL
                    AND f2.superseded_by IS NULL
               ORDER BY f1.subject"""
        ).fetchall()
    except Exception:
        return []

    contradictions = []
    for r in rows:
        contradictions.append({
            "subject": r[1],
            "fact_a": {"id": r[0], "fact": r[2], "timestamp": r[3]},
            "fact_b": {"id": r[4], "fact": r[5], "timestamp": r[6]},
        })

    return contradictions
