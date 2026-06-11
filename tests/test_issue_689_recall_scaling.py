"""Regression lock: search_consolidated must not re-run the broad-OR FTS query
when there is no consolidated data (PERF-01 / issue #689).

Pre-fix: with empty summaries + fact_timeline (the default until consolidate()
runs, and always in FTS-only mode) search_consolidated fell through to an FTS
fallback that re-ran the same broad-OR query the primary search pipeline had
already run — doubling recall latency (~44% of search cost at scale).
Post-fix: it returns [] early without touching FTS.

FTS-only / no model loads.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import truememory.consolidation as consolidation
from truememory.consolidation import search_consolidated
from truememory.storage import create_db, insert_message


def test_empty_consolidation_skips_fts_fallback(tmp_path, monkeypatch):
    conn = create_db(tmp_path / "c.db")
    # Seed messages so FTS *would* return hits if the fallback ran.
    for i in range(5):
        insert_message(conn, {
            "content": f"the carbonsense journey milestone {i} funding round",
            "sender": "alice", "recipient": "bob",
            "timestamp": f"2026-0{i+1}-01T10:00:00Z", "category": "s", "modality": "conversation",
        })
    conn.commit()

    # summaries + fact_timeline are empty (never consolidated).
    assert conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM fact_timeline").fetchone()[0] == 0

    # Spy on the FTS fallback — it must NOT be called.
    called = {"n": 0}
    real = consolidation._fts_search

    def _spy(*a, **k):
        called["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(consolidation, "_fts_search", _spy)

    res = search_consolidated(conn, "summarize the carbonsense journey funding", limit=10)
    assert res == [], "search_consolidated should return [] when no consolidated data exists"
    assert called["n"] == 0, "the redundant FTS fallback must not run on empty consolidation tables"
    conn.close()


def test_with_consolidated_data_still_searches(tmp_path):
    """When a summary exists, search_consolidated still returns it (no regression)."""
    import json
    conn = create_db(tmp_path / "c2.db")
    conn.execute(
        "INSERT INTO summaries (period, entity, summary, key_facts, message_ids, created_at) "
        "VALUES ('all', 'alice', ?, ?, '[]', '2026-01-01T00:00:00Z')",
        ("alice raised a Series A funding round for carbonsense", json.dumps(["Series A"])),
    )
    conn.commit()
    res = search_consolidated(conn, "alice funding carbonsense", limit=10)
    assert any("Series A" in r.get("content", "") or "funding" in r.get("content", "") for r in res)
    conn.close()
