from __future__ import annotations

import datetime
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from truememory.dashboard.server.deps import get_engine

router = APIRouter(prefix="/api", tags=["memories"])


class SearchBody(BaseModel):
    query: str
    limit: int = 50


class UpdateBody(BaseModel):
    content: str


class BulkDeleteBody(BaseModel):
    ids: list[int]


@router.get("/memories")
def list_memories(
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    sender: Optional[str] = Query(None),
    sort: str = Query("newest"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    engine = get_engine()

    if search:
        results = engine.search(search, limit=limit)
        if category:
            results = [r for r in results if r.get("category") == category]
        if sender:
            results = [r for r in results if r.get("sender") == sender]
        return {"memories": results, "total": len(results), "limit": limit, "offset": 0}

    engine._ensure_connection()
    conn = engine.conn

    where_clauses = []
    params: list = []
    if category:
        where_clauses.append("category = ?")
        params.append(category)
    if sender:
        where_clauses.append("sender = ?")
        params.append(sender)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    order = "id DESC" if sort == "newest" else "id ASC"

    count_sql = f"SELECT COUNT(*) FROM messages {where_sql}"
    total = conn.execute(count_sql, params).fetchone()[0]

    data_sql = f"""
        SELECT id, content, sender, recipient, timestamp, category, modality
        FROM messages {where_sql}
        ORDER BY {order}
        LIMIT ? OFFSET ?
    """
    rows = conn.execute(data_sql, params + [limit, offset]).fetchall()

    memories = [
        {
            "id": r[0], "content": r[1], "sender": r[2], "recipient": r[3],
            "timestamp": r[4], "category": r[5], "modality": r[6],
        }
        for r in rows
    ]
    return {"memories": memories, "total": total, "limit": limit, "offset": offset}


@router.get("/memories/senders")
def list_senders():
    engine = get_engine()
    engine._ensure_connection()
    from truememory.storage import get_all_senders
    senders = get_all_senders(engine.conn)
    return [s for s in senders if s]


@router.get("/memories/categories")
def list_categories():
    engine = get_engine()
    engine._ensure_connection()
    rows = engine.conn.execute(
        "SELECT DISTINCT category FROM messages WHERE category != '' ORDER BY category"
    ).fetchall()
    return [r[0] for r in rows]


@router.get("/memories/stats")
def memory_stats():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    week_ago = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
    ).isoformat()
    this_week = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE timestamp >= ?", (week_ago,)
    ).fetchone()[0]

    entities = conn.execute(
        """SELECT COUNT(*) FROM entity_profiles
           WHERE entity NOT LIKE '__test%'
           AND entity NOT IN ('test', 'test_user')
           AND LENGTH(entity) <= 50"""
    ).fetchone()[0]

    thirty_days_ago = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
    ).isoformat()
    daily_rows = conn.execute(
        """SELECT DATE(timestamp) as day, COUNT(*) as cnt
           FROM messages WHERE timestamp >= ?
           GROUP BY DATE(timestamp) ORDER BY day""",
        (thirty_days_ago,),
    ).fetchall()

    today = datetime.date.today()
    daily_map = {r[0]: r[1] for r in daily_rows}
    sparkline = []
    for i in range(30):
        d = (today - datetime.timedelta(days=29 - i)).isoformat()
        sparkline.append(daily_map.get(d, 0))

    cat_rows = conn.execute(
        "SELECT category, COUNT(*) FROM messages GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    categories = {r[0] or "(uncategorized)": r[1] for r in cat_rows}

    from truememory.dashboard.server.deps import get_config
    config = get_config()
    tier = config.get("tier", "edge")
    capabilities = {"fts5": True, "vector_search": True}
    if tier in ("base", "pro"):
        capabilities["reranker"] = True
    if tier == "pro":
        capabilities["hyde"] = True

    return {
        "total": total,
        "this_week": this_week,
        "entities": entities,
        "gate_pass_rate": None,
        "sparkline": sparkline,
        "categories": categories,
        "capabilities": capabilities,
    }


@router.post("/memories/search")
def search_memories(body: SearchBody):
    engine = get_engine()
    results = engine.search(body.query, limit=body.limit)
    return results


@router.get("/memories/{memory_id}")
def get_memory(memory_id: int):
    engine = get_engine()
    result = engine.get(memory_id)
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Memory not found")
    return result


@router.put("/memories/{memory_id}")
def update_memory(memory_id: int, body: UpdateBody):
    engine = get_engine()
    result = engine.update(memory_id, content=body.content)
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Memory not found")
    return result


@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: int):
    engine = get_engine()
    deleted = engine.delete(memory_id)
    return {"deleted": deleted}


@router.post("/memories/bulk-delete")
def bulk_delete(body: BulkDeleteBody):
    engine = get_engine()
    count = 0
    for mid in body.ids:
        if engine.delete(mid):
            count += 1
    return {"deleted": count, "total": len(body.ids)}
