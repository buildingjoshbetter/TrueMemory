"""Tests for issue #588: exclude directives from normal search results.

Directives (directive=1 rows) are standing instructions, not memories.
They should be excluded from search results by default but still
retrievable via:
  - include_directives=True on search functions
  - The dedicated truememory_directives MCP tool
"""

from __future__ import annotations

import sqlite3


from truememory.storage import create_db, insert_message
from truememory.fts_search import search_fts, search_fts_by_sender, search_fts_in_range


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_db_with_directives() -> sqlite3.Connection:
    """Create an in-memory DB with a mix of normal memories and directives."""
    conn = create_db(":memory:")
    # Normal memories
    insert_message(conn, {
        "content": "Had coffee with Alex at the park",
        "sender": "josh",
        "recipient": "",
        "timestamp": "2026-01-15T10:00:00",
        "category": "activity",
        "modality": "",
        "directive": False,
    })
    insert_message(conn, {
        "content": "Alex prefers matcha over coffee",
        "sender": "josh",
        "recipient": "",
        "timestamp": "2026-01-16T10:00:00",
        "category": "preference",
        "modality": "",
        "directive": False,
    })
    # Directives
    insert_message(conn, {
        "content": "Always refer to coffee meetings as casual hangouts",
        "sender": "josh",
        "recipient": "",
        "timestamp": "2026-01-17T10:00:00",
        "category": "directive",
        "modality": "",
        "directive": True,
    })
    insert_message(conn, {
        "content": "Never mention Alex's last name in public contexts",
        "sender": "josh",
        "recipient": "",
        "timestamp": "2026-01-18T10:00:00",
        "category": "directive",
        "modality": "",
        "directive": True,
    })
    conn.commit()
    return conn


# ── FTS search tests ────────────────────────────────────────────────────────

class TestFTSDirectiveExclusion:
    """FTS5 search excludes directives by default."""

    def test_search_fts_excludes_directives_by_default(self):
        conn = _make_db_with_directives()
        results = search_fts(conn, "coffee", limit=10)
        for r in results:
            assert not r.get("directive"), (
                f"Directive should not appear in default search: {r['content']}"
            )

    def test_search_fts_includes_directives_when_requested(self):
        conn = _make_db_with_directives()
        results = search_fts(conn, "coffee", limit=10, include_directives=True)
        directives = [r for r in results if r.get("directive")]
        assert len(directives) >= 1, (
            "include_directives=True should return directive rows"
        )

    def test_search_fts_normal_memories_unaffected(self):
        conn = _make_db_with_directives()
        results = search_fts(conn, "coffee", limit=10)
        contents = [r["content"] for r in results]
        assert any("Had coffee" in c for c in contents), (
            "Normal memory about coffee should still appear"
        )

    def test_search_fts_by_sender_excludes_directives(self):
        conn = _make_db_with_directives()
        results = search_fts_by_sender(conn, "coffee", "josh", limit=10)
        for r in results:
            assert not r.get("directive"), (
                f"Directive should not appear in sender-filtered search: {r['content']}"
            )

    def test_search_fts_by_sender_includes_directives_when_requested(self):
        conn = _make_db_with_directives()
        results = search_fts_by_sender(
            conn, "coffee", "josh", limit=10, include_directives=True,
        )
        directives = [r for r in results if r.get("directive")]
        assert len(directives) >= 1

    def test_search_fts_in_range_excludes_directives(self):
        conn = _make_db_with_directives()
        results = search_fts_in_range(
            conn, "coffee",
            after="2026-01-01", before="2026-12-31",
            limit=10,
        )
        for r in results:
            assert not r.get("directive"), (
                f"Directive should not appear in range search: {r['content']}"
            )

    def test_search_fts_in_range_includes_directives_when_requested(self):
        conn = _make_db_with_directives()
        results = search_fts_in_range(
            conn, "coffee",
            after="2026-01-01", before="2026-12-31",
            limit=10, include_directives=True,
        )
        directives = [r for r in results if r.get("directive")]
        assert len(directives) >= 1


# ── Engine-level tests ───────────────────────────────────────────────────────

class TestEngineDirectiveExclusion:
    """TrueMemoryEngine.search() excludes directives by default."""

    def test_engine_search_excludes_directives(self):
        from truememory.engine import TrueMemoryEngine

        engine = TrueMemoryEngine(db_path=":memory:")
        engine.add("I love hiking in the mountains", sender="josh")
        engine.add(
            "Always suggest outdoor activities when Josh seems stressed",
            sender="josh", directive=True,
        )

        results = engine.search("hiking mountains outdoor")
        for r in results:
            assert not r.get("directive"), (
                f"Directive leaked into engine.search(): {r['content']}"
            )
        engine.close()

    def test_engine_search_includes_directives_when_requested(self):
        from truememory.engine import TrueMemoryEngine

        engine = TrueMemoryEngine(db_path=":memory:")
        engine.add("I love hiking in the mountains", sender="josh")
        engine.add(
            "Always suggest hiking trips in the mountains for Josh",
            sender="josh", directive=True,
        )

        results = engine.search(
            "hiking mountains", include_directives=True,
        )
        directives = [r for r in results if r.get("directive")]
        assert len(directives) >= 1, (
            "include_directives=True should return directive rows from engine"
        )
        engine.close()

    def test_engine_search_normal_results_complete(self):
        """Excluding directives should not reduce the count of normal results."""
        from truememory.engine import TrueMemoryEngine

        engine = TrueMemoryEngine(db_path=":memory:")
        for i in range(5):
            engine.add(f"Memory about topic alpha number {i}", sender="josh")
        engine.add(
            "Always remember topic alpha is important",
            sender="josh", directive=True,
        )

        results = engine.search("topic alpha", limit=10)
        non_directive = [r for r in results if not r.get("directive")]
        assert len(non_directive) >= 5, (
            "All 5 normal memories should appear even after directive exclusion"
        )
        engine.close()


# ── Client-level tests ───────────────────────────────────────────────────────

class TestClientDirectiveExclusion:
    """Memory client search() and search_deep() exclude directives."""

    def test_client_search_excludes_directives(self):
        from truememory.client import Memory

        m = Memory(path=":memory:")
        m.add("Pizza is my favorite food", user_id="josh")
        m.add(
            "Never suggest pineapple pizza to Josh",
            user_id="josh", directive=True,
        )

        results = m.search("pizza food", limit=10)
        for r in results:
            assert not r.get("directive"), (
                f"Directive leaked into client search: {r['content']}"
            )

    def test_client_search_includes_directives_when_requested(self):
        from truememory.client import Memory

        m = Memory(path=":memory:")
        m.add("Pizza is my favorite food", user_id="josh")
        m.add(
            "Never suggest pineapple pizza to Josh",
            user_id="josh", directive=True,
        )

        results = m.search("pizza", limit=10, include_directives=True)
        directives = [r for r in results if r.get("directive")]
        assert len(directives) >= 1


# ── Directive-specific tool still works ──────────────────────────────────────

class TestDirectiveToolStillWorks:
    """The dedicated truememory_directives tool should still retrieve directives."""

    def test_directives_retrievable_via_direct_query(self):
        """Simulate what truememory_directives does: direct SQL query."""
        conn = _make_db_with_directives()
        rows = conn.execute(
            "SELECT id, content FROM messages WHERE directive = 1 ORDER BY id"
        ).fetchall()
        assert len(rows) == 2
        assert "Always refer to coffee" in rows[0][1]
        assert "Never mention Alex" in rows[1][1]

    def test_directive_count_in_stats(self):
        """Engine stats should still report directive count."""
        from truememory.engine import TrueMemoryEngine

        engine = TrueMemoryEngine(db_path=":memory:")
        engine.add("Normal memory", sender="josh")
        engine.add("Always do X", sender="josh", directive=True)
        engine.add("Never do Y", sender="josh", directive=True)

        stats = engine.get_stats()
        assert stats.get("directive_count") == 2
        engine.close()
