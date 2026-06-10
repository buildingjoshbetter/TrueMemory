from __future__ import annotations

from fastapi import APIRouter, HTTPException

from truememory.dashboard.server.deps import get_engine
from truememory.personality import (
    get_entity_profile,
    extract_preferences,
)

router = APIRouter(prefix="/api", tags=["entities"])


@router.get("/entities")
def list_entities():
    import json as _json

    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    rows = conn.execute(
        """SELECT entity, message_count, traits, topics, updated_at
           FROM entity_profiles ORDER BY message_count DESC"""
    ).fetchall()

    entities = []
    for r in rows:
        if r[0].startswith("__test") or r[0] in ("test", "test_user"):
            continue
        if len(r[0]) > 50:
            continue

        traits = []
        topics = []
        try:
            traits = _json.loads(r[2]) if r[2] else []
        except (_json.JSONDecodeError, TypeError):
            pass
        try:
            topics = _json.loads(r[3]) if r[3] else []
        except (_json.JSONDecodeError, TypeError):
            pass

        entities.append({
            "entity": r[0],
            "message_count": r[1],
            "traits": traits if isinstance(traits, list) else list(traits.keys()) if isinstance(traits, dict) else [],
            "topics": topics if isinstance(topics, list) else [],
            "updated_at": r[4] or "",
        })

    # Case-insensitive merge: combine entities with same lowercase name
    merged: dict[str, dict] = {}
    for e in entities:
        key = e["entity"].lower()
        if key in merged:
            merged[key]["message_count"] += e["message_count"]
            existing_traits = set(merged[key]["traits"])
            for t in e["traits"]:
                if t not in existing_traits:
                    merged[key]["traits"].append(t)
            existing_topics = set(merged[key]["topics"])
            for t in e["topics"]:
                if t not in existing_topics:
                    merged[key]["topics"].append(t)
            if e["updated_at"] > merged[key]["updated_at"]:
                merged[key]["updated_at"] = e["updated_at"]
        else:
            merged[key] = dict(e)
    entities = list(merged.values())

    mentioned = _extract_mentioned_entities(conn)
    entity_names = {e["entity"].lower() for e in entities}
    for m in mentioned:
        if m["entity"].lower() not in entity_names:
            entities.append(m)

    entities.sort(key=lambda e: e["message_count"], reverse=True)
    return entities


def _extract_mentioned_entities(conn) -> list[dict]:
    rows = conn.execute(
        """SELECT content, category FROM messages
           WHERE category IN ('relationship', 'personal')
           ORDER BY id DESC LIMIT 500"""
    ).fetchall()

    from collections import Counter
    name_counts: Counter = Counter()

    for content, _ in rows:
        text = content.lower()
        for marker in ("brother", "sister", "mom", "dad", "wife", "husband",
                        "girlfriend", "boyfriend", "partner", "friend",
                        "advisor", "cofounder", "co-founder"):
            if marker in text:
                name_counts[marker] += 1

        import re
        named = re.findall(
            r"\b(?:user(?:'s)?|josh(?:'s)?)\s+(?:brother|sister|friend|advisor|partner)\s+(?:is\s+)?([A-Z][a-z]+)",
            content,
        )
        for n in named:
            if len(n) > 2 and n.lower() not in ("the", "this", "that", "user"):
                name_counts[n] += 1

    entities = []
    for name, count in name_counts.most_common(20):
        if count >= 1:
            entities.append({
                "entity": name,
                "message_count": count,
                "traits": [],
                "topics": [],
                "updated_at": "",
            })
    return entities


@router.get("/entities/graph")
def entity_graph():
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    entity_rows = conn.execute(
        "SELECT entity, message_count FROM entity_profiles ORDER BY message_count DESC"
    ).fetchall()

    # Filter test entities from nodes
    raw_nodes = []
    for r in entity_rows:
        if r[0].startswith("__test") or r[0] in ("test", "test_user"):
            continue
        if len(r[0]) > 50:
            continue
        raw_nodes.append({"id": r[0], "message_count": r[1], "radius": max(12, min(40, r[1] // 5 + 10))})

    # Case-insensitive node merge
    merged_nodes: dict[str, dict] = {}
    for n in raw_nodes:
        key = n["id"].lower()
        if key in merged_nodes:
            merged_nodes[key]["message_count"] += n["message_count"]
            merged_nodes[key]["radius"] = max(12, min(40, merged_nodes[key]["message_count"] // 5 + 10))
        else:
            merged_nodes[key] = dict(n)
    nodes = list(merged_nodes.values())

    # Collect filtered entity IDs for edge filtering
    filtered_node_ids = {n["id"].lower() for n in nodes}

    edge_rows = conn.execute(
        """SELECT entity_a, entity_b, relationship_type, strength, dunbar_layer
           FROM entity_relationships"""
    ).fetchall()

    edges = []
    for r in edge_rows:
        src, tgt = r[0], r[1]
        # Filter edges where source or target is a filtered entity
        if src.lower() not in filtered_node_ids or tgt.lower() not in filtered_node_ids:
            continue
        edges.append({
            "source": src, "target": tgt,
            "relationship_type": r[2], "strength": r[3],
            "dunbar_layer": r[4],
        })

    if not edges and len(nodes) > 1:
        sender_counts = conn.execute(
            """SELECT sender, COUNT(*) as cnt FROM messages
               WHERE sender != '' GROUP BY sender ORDER BY cnt DESC LIMIT 20"""
        ).fetchall()

        entity_set = {n["id"].lower() for n in nodes}
        for s_row in sender_counts:
            sender = s_row[0]
            if sender.lower() in entity_set:
                for node in nodes:
                    if node["id"].lower() != sender.lower():
                        edges.append({
                            "source": sender,
                            "target": node["id"],
                            "relationship_type": "mentioned",
                            "strength": min(1.0, s_row[1] / 100),
                            "dunbar_layer": "",
                        })

    return {"nodes": nodes, "edges": edges}


@router.get("/entities/{entity_name}")
def get_entity(entity_name: str):
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn

    profile = get_entity_profile(conn, entity_name)
    if profile is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    recent = conn.execute(
        """SELECT id, content, timestamp, category FROM messages
           WHERE sender = ? ORDER BY id DESC LIMIT 10""",
        (entity_name,),
    ).fetchall()

    recent_memories = [
        {"id": r[0], "content": r[1][:200], "timestamp": r[2], "category": r[3]}
        for r in recent
    ]

    return {
        "profile": profile,
        "recent_memories": recent_memories,
    }


@router.get("/entities/{entity_name}/preferences")
def get_entity_preferences(entity_name: str):
    engine = get_engine()
    engine._ensure_connection()
    conn = engine.conn
    prefs = extract_preferences(conn, entity_name)
    return prefs
