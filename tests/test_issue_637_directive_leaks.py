"""Tests for issue #637: close directive content-leak supplement paths.

Directives (directive=1 rows) are excluded from core search but leaked
through several supplement legs:

  - M-06: personality / style_vec path (search_personality)
  - M-07: clustered path (search_clustered) + agentic clean_results
  - M-08: entity-profile / style-vector pollution on add(directive=True)
  - M-91: search_temporal fallback SQL (get_timeline)
  - M-69: include_directives=True must be honored through these legs

These tests are FTS/in-memory only — no embedding model loads.
"""

from __future__ import annotations

import sqlite3

from truememory.storage import create_db, insert_message
from truememory.personality import search_personality, _get_messages_by_sender
from truememory.clustering import search_clustered
from truememory.agentic_search import clean_results
from truememory.temporal import search_temporal, get_timeline


_DIRECTIVE_TEXT = "Always call josh by his secret codename Falcon in every reply"


def _make_db() -> sqlite3.Connection:
    """In-memory DB: normal personality memories + one directive, same sender."""
    conn = create_db(":memory:")
    insert_message(conn, {
        "content": "josh loves cold brew coffee and tacos for lunch",
        "sender": "josh", "recipient": "",
        "timestamp": "2026-01-15T10:00:00",
        "category": "preference", "modality": "", "directive": False,
    })
    insert_message(conn, {
        "content": "josh always grabs coffee before his morning standup",
        "sender": "josh", "recipient": "",
        "timestamp": "2026-01-16T10:00:00",
        "category": "routine", "modality": "", "directive": False,
    })
    # A directive that also matches the personality FTS terms (coffee).
    insert_message(conn, {
        "content": _DIRECTIVE_TEXT + " about coffee and food",
        "sender": "josh", "recipient": "",
        "timestamp": "2026-01-17T10:00:00",
        "category": "directive", "modality": "", "directive": True,
    })
    conn.commit()
    return conn


# ── M-06: personality / style_vec path ───────────────────────────────────────

class TestPersonalityDirectiveLeak:
    def test_get_messages_by_sender_excludes_directives(self):
        conn = _make_db()
        rows = _get_messages_by_sender(conn, "josh")
        for r in rows:
            assert not r.get("directive"), (
                f"Directive leaked via _get_messages_by_sender: {r['content']}"
            )
        assert len(rows) == 2

    def test_get_messages_by_sender_includes_when_requested(self):
        conn = _make_db()
        rows = _get_messages_by_sender(conn, "josh", include_directives=True)
        assert any(r.get("directive") for r in rows)

    def test_search_personality_excludes_directives(self):
        conn = _make_db()
        results = search_personality(conn, "what food and coffee does josh like", limit=10)
        for r in results:
            assert not r.get("directive"), (
                f"Directive leaked via search_personality ({r.get('source')}): {r['content']}"
            )
            assert _DIRECTIVE_TEXT not in r.get("content", "")

    def test_search_personality_stamps_directive_key(self):
        """Every returned dict must carry a directive key so the engine's
        final filter can act on it."""
        conn = _make_db()
        results = search_personality(conn, "what food and coffee does josh like", limit=10)
        assert results, "expected at least one personality result"
        for r in results:
            assert "directive" in r

    def test_search_personality_includes_directives_when_requested(self):
        conn = _make_db()
        results = search_personality(
            conn, "what food and coffee does josh like",
            limit=10, include_directives=True,
        )
        assert any(_DIRECTIVE_TEXT in r.get("content", "") for r in results), (
            "include_directives=True should surface the directive via personality leg"
        )


# ── M-07: clustered + agentic clean_results ──────────────────────────────────

def _seed_clusters(conn: sqlite3.Connection) -> None:
    """Build a single cluster containing every message id (directive included).

    We stub embeddings/centroid via the real tables so search_clustered runs
    its SQL leg without loading a model. The query vec is supplied by a stub
    model patched in the test.
    """
    import struct
    ids = [r[0] for r in conn.execute("SELECT id FROM messages ORDER BY id")]
    dim = 4
    centroid = struct.pack(f"{dim}f", *([1.0] * dim))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cluster_centroids "
        "(cluster_id INTEGER PRIMARY KEY, centroid BLOB, message_count INTEGER, "
        " session_range TEXT, summary TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS message_clusters "
        "(message_id INTEGER, cluster_id INTEGER)"
    )
    conn.execute(
        "INSERT INTO cluster_centroids(cluster_id, centroid, message_count) VALUES (1, ?, ?)",
        (centroid, len(ids)),
    )
    for mid in ids:
        conn.execute(
            "INSERT INTO message_clusters(message_id, cluster_id) VALUES (?, 1)", (mid,)
        )
    conn.commit()


class _StubModel:
    def encode(self, texts):
        import numpy as np
        return np.ones((len(texts), 4), dtype="float32")


class TestClusteredDirectiveLeak:
    def test_search_clustered_excludes_directives(self, monkeypatch):
        import truememory.vector_search as vs
        conn = _make_db()
        _seed_clusters(conn)
        monkeypatch.setattr(vs, "get_model", lambda *a, **k: _StubModel())
        results = search_clustered(conn, "coffee food", limit=10)
        for r in results:
            assert not r.get("directive"), (
                f"Directive leaked via search_clustered: {r['content']}"
            )
            assert _DIRECTIVE_TEXT not in r.get("content", "")

    def test_search_clustered_includes_when_requested(self, monkeypatch):
        import truememory.vector_search as vs
        conn = _make_db()
        _seed_clusters(conn)
        monkeypatch.setattr(vs, "get_model", lambda *a, **k: _StubModel())
        results = search_clustered(conn, "coffee food", limit=10, include_directives=True)
        assert any(_DIRECTIVE_TEXT in r.get("content", "") for r in results)

    def test_clean_results_drops_directives(self):
        rows = [
            {"id": 1, "content": "normal memory", "directive": False, "score": 0.9},
            {"id": 2, "content": _DIRECTIVE_TEXT, "directive": True, "score": 0.95},
        ]
        cleaned = clean_results(rows, limit=10)
        assert all(not r.get("directive") for r in cleaned)
        assert all(_DIRECTIVE_TEXT not in r["content"] for r in cleaned)

    def test_clean_results_keeps_directives_when_requested(self):
        rows = [
            {"id": 1, "content": "normal memory", "directive": False, "score": 0.9},
            {"id": 2, "content": _DIRECTIVE_TEXT, "directive": True, "score": 0.95},
        ]
        cleaned = clean_results(rows, limit=10, include_directives=True)
        assert any(r.get("directive") for r in cleaned)


# ── M-08: add(directive=True) must not pollute profiles / style vectors ───────

class TestDirectiveProfilePollution:
    def test_directive_only_sender_has_no_profile_or_style_vec(self):
        from truememory.engine import TrueMemoryEngine

        engine = TrueMemoryEngine(db_path=":memory:")
        # A sender whose ONLY message is a directive.
        engine.add(
            "Always greet operator-x with a formal salutation",
            sender="operator-x", directive=True,
        )

        conn = engine.conn

        def _count(table: str) -> int:
            try:
                return conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE LOWER(entity) = 'operator-x'"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                return 0

        assert _count("entity_profiles") == 0, (
            "Directive content polluted entity_profiles"
        )
        assert _count("entity_style_vectors") == 0, (
            "Directive content polluted entity_style_vectors"
        )
        engine.close()

    def test_normal_add_still_builds_profile(self):
        """Sanity: non-directive add for a fresh sender still updates profile."""
        from truememory.engine import TrueMemoryEngine

        engine = TrueMemoryEngine(db_path=":memory:")
        if not engine._has_personality:
            engine.close()
            return  # personality module unavailable in this build
        engine.add("operator-y loves espresso and long walks", sender="operator-y")
        conn = engine.conn
        cnt = conn.execute(
            "SELECT COUNT(*) FROM entity_profiles WHERE LOWER(entity) = 'operator-y'"
        ).fetchone()[0]
        assert cnt >= 1
        engine.close()


# ── M-91: search_temporal fallback excludes directives ───────────────────────

def _make_temporal_db() -> sqlite3.Connection:
    conn = create_db(":memory:")
    insert_message(conn, {
        "content": "met the team on monday",
        "sender": "josh", "recipient": "",
        "timestamp": "2026-02-03T10:00:00",
        "category": "event", "modality": "", "directive": False,
    })
    insert_message(conn, {
        "content": _DIRECTIVE_TEXT,
        "sender": "josh", "recipient": "",
        "timestamp": "2026-02-04T10:00:00",
        "category": "directive", "modality": "", "directive": True,
    })
    conn.commit()
    return conn


class TestTemporalDirectiveLeak:
    def test_get_timeline_excludes_directives(self):
        conn = _make_temporal_db()
        rows = get_timeline(conn, after="2026-01-01", before="2026-12-31")
        for r in rows:
            assert not r.get("directive")
            assert _DIRECTIVE_TEXT not in r.get("content", "")

    def test_get_timeline_includes_when_requested(self):
        conn = _make_temporal_db()
        rows = get_timeline(
            conn, after="2026-01-01", before="2026-12-31", include_directives=True,
        )
        assert any(_DIRECTIVE_TEXT in r.get("content", "") for r in rows)

    def test_search_temporal_fallback_excludes_directives(self):
        conn = _make_temporal_db()
        # Empty fts_results forces the fallback get_timeline leg.
        results = search_temporal(
            conn, "what happened in february 2026",
            fts_results=[], limit=10,
        )
        for r in results:
            assert not r.get("directive"), (
                f"Directive leaked via search_temporal fallback: {r.get('content')}"
            )

    def test_search_temporal_fallback_includes_when_requested(self):
        conn = _make_temporal_db()
        results = search_temporal(
            conn, "what happened in february 2026",
            fts_results=[], limit=10, include_directives=True,
        )
        assert any(_DIRECTIVE_TEXT in r.get("content", "") for r in results)
