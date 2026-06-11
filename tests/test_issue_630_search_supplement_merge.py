"""Regression tests for issue #630 — id-hygiene bugs in engine.search's
supplement-merge step.

Three confirmed bugs in the path that merges personality / contradiction /
consolidated-summary supplements into the FTS+vector result list:

* **M-01** — the final ``cleaned.sort`` tie-broke on ``d["id"]``. Consolidation
  emits ``"summary_N"`` *string* ids (#606) while messages have *int* ids, so any
  score tie (guaranteed: FTS normalization pins the top hit at exactly 1.0)
  raised ``TypeError: '<' not supported between 'str' and 'int'``. Same bug in
  ``agentic_search.clean_results``. Fix: tie-break on ``str(d.get("id", ""))``.

* **M-05** — ``personality.search_personality`` emits rows with *no* ``"id"``
  key. ``engine.search`` built ``existing_ids`` with bracket access
  (``{r["id"] for r in results}``); the KeyError was swallowed by a blanket
  ``except`` so on EVERY personality-intent query ALL contradiction supplements
  and consolidated summaries were silently discarded. Fix: ``r.get("id")``.

* **M-67** — id-less rows were rewritten to ``id=0`` during cleaning, so #606's
  id-keyed RRF collapsed multiple distinct id-less rows into one fabricated
  ``id=0`` document. Fix: preserve ``None`` ids through cleaning.

These tests drive the real ``TrueMemoryEngine.search`` with an in-memory /
FTS-only engine and stub the supplement functions on the ``truememory.engine``
module namespace. No model loads.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture
def fts_engine(tmp_path, monkeypatch):
    """Small FTS-only engine: a handful of seeded messages, no vectors, no
    reranker, salience/surprise boosts disabled so we exercise the bare
    supplement-merge + clean + sort path."""
    monkeypatch.delenv("TRUEMEMORY_ALPHA_SURPRISE", raising=False)

    from truememory.engine import TrueMemoryEngine
    from truememory.storage import create_db

    db_path = tmp_path / "i630.db"
    conn = create_db(db_path)
    for i in range(1, 5):
        conn.execute(
            "INSERT INTO messages (content, sender, recipient, timestamp, "
            "category, modality) VALUES (?, 'alice', 'bob', ?, 'session_1', "
            "'conversation')",
            (f"alice favorite food number {i}", f"2026-0{i}-01T10:00:00Z"),
        )
    conn.commit()
    conn.close()

    eng = TrueMemoryEngine(db_path)
    eng.open(rebuild_vectors=False)
    # Force the bare path: no cross-encoder rerank, no salience filtering.
    eng._has_reranker = False
    eng._has_salience = False
    return eng


# Query that _has_personality_intent() recognizes (matches "favorite food").
_PERSONALITY_QUERY = "what is alice favorite food"


def test_summary_string_id_tie_does_not_raise_typeerror(fts_engine, monkeypatch):
    """M-01: a consolidated summary with a string id ("summary_7") tying at
    score 1.0 with int-id message rows must NOT raise TypeError, and results
    must come back sorted."""
    eng = fts_engine
    eng._has_consolidation = True

    monkeypatch.setattr("truememory.engine.search_contradictions", lambda *a, **k: [])

    def fake_consolidated(conn, query, limit=3):
        # String id + score exactly 1.0 to force a tie with the FTS top hit.
        return [{
            "id": "summary_7",
            "content": "alice tends to prefer spicy food",
            "score": 1.0,
            "source": "summary",
        }]

    monkeypatch.setattr("truememory.engine.search_consolidated", fake_consolidated)

    # Must not raise TypeError on the mixed-type tie-break.
    results = eng.search(_PERSONALITY_QUERY, limit=10)
    assert isinstance(results, list)
    # Sorted descending by score.
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # The string-id summary survived the merge.
    assert any(r.get("id") == "summary_7" for r in results)


def test_personality_intent_keeps_supplements(fts_engine, monkeypatch):
    """M-05: on a personality-intent query, id-less personality rows in the
    result list must NOT cause the contradiction/summary existing_ids
    comprehension to KeyError-and-discard the supplements."""
    eng = fts_engine
    eng._has_personality = True
    eng._has_consolidation = True

    # Personality results carry NO "id" key (this is what triggered the bug).
    def fake_personality(conn, query, limit=5):
        return [{
            "content": "alice communication style: casual",
            "sender": "alice",
            "recipient": "",
            "timestamp": "",
            "source": "profile",
            "score": 1.0,
        }]

    def fake_contradictions(conn, query, *a, **k):
        return [{
            "id": 9001,
            "content": "alice now prefers sushi (was pizza)",
            "current_fact": "alice now prefers sushi",
            "source": "contradiction",
        }]

    def fake_consolidated(conn, query, limit=3):
        return [{
            "id": "summary_3",
            "content": "alice summary: enjoys spicy food",
            "score": 0.5,
            "source": "summary",
        }]

    monkeypatch.setattr("truememory.engine.search_personality", fake_personality)
    monkeypatch.setattr("truememory.engine.search_contradictions", fake_contradictions)
    monkeypatch.setattr("truememory.engine.search_consolidated", fake_consolidated)

    results = eng.search(_PERSONALITY_QUERY, limit=10)
    ids = [r.get("id") for r in results]
    # Both supplements must survive — before the fix they were silently dropped.
    assert 9001 in ids, "contradiction supplement was discarded (M-05 regression)"
    assert "summary_3" in ids, "consolidated summary was discarded (M-05 regression)"


def test_idless_rows_do_not_collapse_to_id_zero(fts_engine, monkeypatch):
    """M-67: two distinct id-less supplement rows must remain two distinct
    rows in the output, not collapse into a single fabricated id=0 document."""
    eng = fts_engine
    eng._has_personality = True
    eng._has_consolidation = True

    def fake_personality(conn, query, limit=5):
        # Two distinct rows, both WITHOUT an id, distinct content.
        return [
            {"content": "alice profile fact ONE", "sender": "alice",
             "source": "profile", "score": 1.0},
            {"content": "alice profile fact TWO", "sender": "alice",
             "source": "profile", "score": 1.0},
        ]

    monkeypatch.setattr("truememory.engine.search_personality", fake_personality)
    monkeypatch.setattr("truememory.engine.search_contradictions", lambda *a, **k: [])
    monkeypatch.setattr("truememory.engine.search_consolidated", lambda *a, **k: [])

    results = eng.search(_PERSONALITY_QUERY, limit=10)
    one = [r for r in results if r["content"] == "alice profile fact ONE"]
    two = [r for r in results if r["content"] == "alice profile fact TWO"]
    assert len(one) == 1, "id-less row ONE missing/collapsed (M-67 regression)"
    assert len(two) == 1, "id-less row TWO missing/collapsed (M-67 regression)"
    # Neither was rewritten to a fabricated integer id=0.
    assert one[0].get("id") != 0
    assert two[0].get("id") != 0
