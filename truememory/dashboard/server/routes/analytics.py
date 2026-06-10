from __future__ import annotations

import datetime

from fastapi import APIRouter

from truememory.dashboard.server.deps import get_engine

router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/analytics/growth")
def memory_growth():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    rows = conn.execute(
        """SELECT DATE(timestamp) as day, COUNT(*) as cnt
           FROM messages WHERE timestamp != ''
           GROUP BY DATE(timestamp)
           ORDER BY day"""
    ).fetchall()

    daily = [{"date": r[0], "count": r[1]} for r in rows]

    cumulative = []
    total = 0
    for d in daily:
        total += d["count"]
        cumulative.append({"date": d["date"], "count": d["count"], "cumulative": total})

    return cumulative


@router.get("/analytics/categories")
def category_distribution():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    rows = conn.execute(
        """SELECT category, COUNT(*) as cnt
           FROM messages GROUP BY category
           ORDER BY cnt DESC"""
    ).fetchall()

    result = []
    for r in rows:
        cat_name = r[0] or "(uncategorized)"
        if cat_name != "(uncategorized)":
            cat_name = cat_name.capitalize()
        result.append({"category": cat_name, "count": r[1]})
    return result


@router.get("/analytics/entities")
def top_entities():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    rows = conn.execute(
        """SELECT entity, message_count
           FROM entity_profiles
           ORDER BY message_count DESC
           LIMIT 20"""
    ).fetchall()

    # Filter test entities
    filtered = []
    for r in rows:
        if r[0].startswith("__test") or r[0] in ("test", "test_user"):
            continue
        if len(r[0]) > 50:
            continue
        filtered.append({"entity": r[0], "message_count": r[1]})

    # Case-insensitive merge
    merged: dict[str, dict] = {}
    for e in filtered:
        key = e["entity"].lower()
        if key in merged:
            merged[key]["message_count"] += e["message_count"]
        else:
            merged[key] = dict(e)
    entities = sorted(merged.values(), key=lambda x: x["message_count"], reverse=True)

    # Fallback to sender counts if fewer than 3 real entities
    if len(entities) < 3:
        sender_rows = conn.execute(
            """SELECT sender, COUNT(*) as cnt FROM messages
               WHERE sender != '' GROUP BY LOWER(sender)
               ORDER BY cnt DESC LIMIT 20"""
        ).fetchall()
        entities = [{"entity": r[0], "message_count": r[1]} for r in sender_rows]

    return entities


@router.get("/analytics/ingest")
def ingest_stats():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    now = datetime.datetime.now(datetime.timezone.utc)
    periods = {
        "7d": (now - datetime.timedelta(days=7)).isoformat(),
        "30d": (now - datetime.timedelta(days=30)).isoformat(),
        "all": "",
    }

    result = {}
    for label, since in periods.items():
        if since:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE timestamp >= ?", (since,)
            ).fetchone()[0]
        else:
            count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        result[label] = count

    daily_rows = conn.execute(
        """SELECT DATE(timestamp) as day, COUNT(*) as cnt
           FROM messages WHERE timestamp >= ?
           GROUP BY DATE(timestamp) ORDER BY day""",
        ((now - datetime.timedelta(days=30)).isoformat(),),
    ).fetchall()

    today = datetime.date.today()
    daily_map = {r[0]: r[1] for r in daily_rows}
    daily_rate = []
    for i in range(30):
        d = (today - datetime.timedelta(days=29 - i)).isoformat()
        daily_rate.append({"date": d, "count": daily_map.get(d, 0)})

    result["daily_rate"] = daily_rate
    return result


@router.get("/analytics/timeline")
def timeline_by_category():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    now = datetime.datetime.now(datetime.timezone.utc)
    since = (now - datetime.timedelta(days=30)).isoformat()

    rows = conn.execute(
        """SELECT DATE(timestamp) as day, category, COUNT(*) as cnt
           FROM messages WHERE timestamp >= ? AND category != ''
           GROUP BY DATE(timestamp), category
           ORDER BY day, category""",
        (since,),
    ).fetchall()

    result: dict[str, dict[str, int]] = {}
    for r in rows:
        day = r[0]
        if day not in result:
            result[day] = {}
        result[day][r[1]] = r[2]

    return result
