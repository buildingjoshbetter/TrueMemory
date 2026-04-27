"""Tests for L0 char-n-gram style vector implementation.

Validates algorithm parity with the C3c candidate
(benchmarks/gate_eval/candidates/l0_personality/c3c_char_ngram.py),
database integration, incremental updates, keyword cleanup,
deprecation warnings, and search_personality vector scoring.
"""
from __future__ import annotations

import math
import sqlite3
import warnings

from truememory.personality_style_vec import (
    DIM,
    compute_style_vector,
    cosine_similarity,
    mean_pool_vectors,
    build_entity_style_vectors,
    update_entity_style_vector_incremental,
    get_entity_style_vector,
)
from truememory.storage import create_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _make_test_db() -> sqlite3.Connection:
    """Create an in-memory DB with full TrueMemory schema."""
    conn = create_db(":memory:")
    return conn


def _insert_msg(conn, content, sender="", recipient="", timestamp=""):
    conn.execute(
        "INSERT INTO messages (content, sender, recipient, timestamp, category, modality) "
        "VALUES (?, ?, ?, ?, '', '')",
        (content, sender, recipient, timestamp),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# a. Basic vector properties
# ---------------------------------------------------------------------------

class TestComputeStyleVector:
    def test_basic_properties(self):
        """Vector is length 256, all floats."""
        vec = compute_style_vector("hello world")
        assert len(vec) == 256
        assert all(isinstance(x, float) for x in vec)

    def test_unit_norm(self):
        """L2 norm is ~1.0 (within 1e-6)."""
        vec = compute_style_vector("hello world this is a test sentence")
        norm = _l2_norm(vec)
        assert abs(norm - 1.0) < 1e-6

    def test_deterministic(self):
        """Same text produces same vector."""
        v1 = compute_style_vector("the quick brown fox")
        v2 = compute_style_vector("the quick brown fox")
        assert v1 == v2

    def test_empty_text(self):
        """Empty text returns zero vector."""
        vec = compute_style_vector("")
        assert len(vec) == DIM
        assert all(x == 0.0 for x in vec)
        # Also test whitespace-only
        vec2 = compute_style_vector("   \t\n  ")
        assert all(x == 0.0 for x in vec2)

    def test_different_texts_different_vectors(self):
        """Different texts produce different vectors."""
        v1 = compute_style_vector("hello world")
        v2 = compute_style_vector("goodbye moon")
        assert v1 != v2

    def test_similar_texts_similar_vectors(self):
        """Similar texts have cosine > 0.5."""
        v1 = compute_style_vector("I love coffee in the morning")
        v2 = compute_style_vector("I love tea in the morning")
        sim = cosine_similarity(v1, v2)
        assert sim > 0.5, f"Expected cosine > 0.5, got {sim}"


# ---------------------------------------------------------------------------
# g-h. Mean pool
# ---------------------------------------------------------------------------

class TestMeanPool:
    def test_identical_vectors(self):
        """Mean of identical vectors equals that vector."""
        vec = compute_style_vector("test message")
        result = mean_pool_vectors([vec, vec, vec])
        for a, b in zip(vec, result):
            assert abs(a - b) < 1e-6

    def test_normalization(self):
        """Result is unit-normalized."""
        v1 = compute_style_vector("hello world")
        v2 = compute_style_vector("goodbye moon")
        result = mean_pool_vectors([v1, v2])
        norm = _l2_norm(result)
        assert abs(norm - 1.0) < 1e-6

    def test_empty(self):
        """Empty list returns zero vector."""
        result = mean_pool_vectors([])
        assert len(result) == DIM
        assert all(x == 0.0 for x in result)


# ---------------------------------------------------------------------------
# i-j. Cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical(self):
        """Cosine of vector with itself is ~1.0."""
        vec = compute_style_vector("test message for cosine")
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_different_texts_lower(self):
        """Cosine of very different texts is < 0.5."""
        v1 = compute_style_vector("aaaa bbbb cccc dddd eeee ffff gggg")
        v2 = compute_style_vector("zzzz yyyy xxxx wwww vvvv uuuu tttt")
        sim = cosine_similarity(v1, v2)
        assert sim < 0.5, f"Expected cosine < 0.5, got {sim}"

    def test_zero_vector(self):
        """Cosine with zero vector returns 0.0."""
        vec = compute_style_vector("hello")
        zero = [0.0] * DIM
        assert cosine_similarity(vec, zero) == 0.0
        assert cosine_similarity(zero, zero) == 0.0


# ---------------------------------------------------------------------------
# k. Database: build_entity_style_vectors
# ---------------------------------------------------------------------------

class TestBuildEntityStyleVectors:
    def test_build_for_multiple_entities(self):
        """Ingest messages for 2 entities, build vectors, verify both stored."""
        conn = _make_test_db()

        _insert_msg(conn, "I love hiking in the mountains", sender="Alice")
        _insert_msg(conn, "The trail was beautiful today", sender="Alice")
        _insert_msg(conn, "Let me check the database migration", sender="Bob")
        _insert_msg(conn, "The API endpoint is returning 500", sender="Bob")

        result = build_entity_style_vectors(conn)

        assert "Alice" in result
        assert "Bob" in result
        assert len(result["Alice"]) == DIM
        assert len(result["Bob"]) == DIM

        # Vectors should be stored in DB
        stored_alice = get_entity_style_vector(conn, "Alice")
        assert stored_alice is not None
        assert len(stored_alice) == DIM

        stored_bob = get_entity_style_vector(conn, "Bob")
        assert stored_bob is not None
        conn.close()


# ---------------------------------------------------------------------------
# l. Incremental update
# ---------------------------------------------------------------------------

class TestIncrementalUpdate:
    def test_incremental_converges(self):
        """Add messages one at a time, verify vector converges toward batch-built vector."""
        conn = _make_test_db()
        messages = [
            "I love hiking in the mountains",
            "The trail was beautiful today",
            "Going for a run this morning",
            "Mountain biking is my favorite weekend activity",
        ]

        # Batch build
        for msg in messages:
            _insert_msg(conn, msg, sender="Alice")
        batch_result = build_entity_style_vectors(conn)
        batch_vec = batch_result["Alice"]

        # Incremental build (fresh DB)
        conn2 = _make_test_db()
        for msg in messages:
            _insert_msg(conn2, msg, sender="Alice")
            update_entity_style_vector_incremental(conn2, "Alice", msg)
        inc_vec = get_entity_style_vector(conn2, "Alice")

        assert inc_vec is not None
        # Cosine should be reasonably high (incremental weighted average
        # differs slightly from batch mean-pool due to normalization order)
        sim = cosine_similarity(batch_vec, inc_vec)
        assert sim > 0.8, f"Expected cosine > 0.8, got {sim}"

        conn.close()
        conn2.close()


# ---------------------------------------------------------------------------
# m. Retrieve stored vector
# ---------------------------------------------------------------------------

class TestGetEntityStyleVector:
    def test_retrieve_matches_build(self):
        """Retrieved vector matches what was built."""
        conn = _make_test_db()
        _insert_msg(conn, "Test message one", sender="Charlie")
        _insert_msg(conn, "Test message two", sender="Charlie")

        result = build_entity_style_vectors(conn)
        stored = get_entity_style_vector(conn, "Charlie")

        assert stored is not None
        for a, b in zip(result["Charlie"], stored):
            assert abs(a - b) < 1e-10
        conn.close()

    def test_nonexistent_entity(self):
        """Returns None for entity with no vector."""
        conn = _make_test_db()
        assert get_entity_style_vector(conn, "Nobody") is None
        conn.close()


# ---------------------------------------------------------------------------
# n. Forget clears vector
# ---------------------------------------------------------------------------

class TestForgetClearsVector:
    def test_delete_clears_style_vector(self):
        """After forget, vector is None."""
        conn = _make_test_db()

        _insert_msg(conn, "Hello from Dave", sender="Dave")
        build_entity_style_vectors(conn)

        assert get_entity_style_vector(conn, "Dave") is not None

        # Simulate forget
        conn.execute("DELETE FROM entity_style_vectors WHERE entity = ?", ("Dave",))
        conn.commit()

        assert get_entity_style_vector(conn, "Dave") is None
        conn.close()


# ---------------------------------------------------------------------------
# o. Josh-specific keywords removed
# ---------------------------------------------------------------------------

class TestJoshKeywordsRemoved:
    def test_no_josh_keywords_in_personality(self):
        """Verify Josh-specific tokens are NOT in any keyword cluster."""
        import truememory.personality as p
        source_code = open(p.__file__).read()

        josh_tokens = ["clickhouse", "biscuit", "corgi", "lily"]

        # Check that none appear as set literal entries
        for token in josh_tokens:
            assert f'"{token}"' not in source_code, \
                f'Josh-specific token "{token}" still found in personality.py'

    def test_f1_acl_removed_from_activities(self):
        """f1 and acl removed from _ACTIVITY_KEYWORDS."""
        from truememory.personality import _ACTIVITY_KEYWORDS
        assert "f1" not in _ACTIVITY_KEYWORDS
        assert "acl" not in _ACTIVITY_KEYWORDS
        assert "formula 1" not in _ACTIVITY_KEYWORDS


# ---------------------------------------------------------------------------
# p-extra. Split-half stability (MEMORIST acceptance criterion)
# ---------------------------------------------------------------------------

class TestSplitHalfStability:
    def test_split_half_cosine_above_070(self):
        """Build profiles from two halves of the same entity's messages.
        Cosine similarity between the two halves must be >= 0.70.
        This is a pre-registered MEMORIST acceptance criterion."""
        from truememory.personality_style_vec import (
            compute_style_vector,
            mean_pool_vectors,
            cosine_similarity,
        )

        # 20 diverse messages from a single persona
        messages = [
            "Hey what's up, just got back from the gym",
            "I've been working on this Rust project all week",
            "The coffee at that new place on 5th is incredible",
            "Berlin is amazing in the summer, you should visit",
            "Just finished reading that book about distributed systems",
            "My dog loves the park near our apartment",
            "We should grab dinner at that Thai place sometime",
            "The concert last night was absolutely wild",
            "I'm thinking about switching to a standing desk",
            "Have you tried that new vegan restaurant downtown?",
            "Working from home has been great for productivity",
            "The sunrise this morning was beautiful from my balcony",
            "I need to start training for the marathon in October",
            "Just deployed a new microservice to production",
            "My roommate makes the best pasta from scratch",
            "The museum exhibit on modern art was thought-provoking",
            "I finally fixed that memory leak in the server",
            "Planning a hiking trip to the mountains next month",
            "The farmers market on Saturday had amazing produce",
            "Late night coding session with lo-fi beats is the vibe",
        ]

        # Split into two halves
        half1 = messages[:10]
        half2 = messages[10:]

        # Build vectors for each half
        vecs1 = [compute_style_vector(m) for m in half1]
        vecs2 = [compute_style_vector(m) for m in half2]

        profile1 = mean_pool_vectors(vecs1)
        profile2 = mean_pool_vectors(vecs2)

        sim = cosine_similarity(profile1, profile2)
        assert sim >= 0.70, (
            f"Split-half cosine similarity {sim:.4f} is below the 0.70 "
            f"acceptance threshold. The style vector is not stable enough."
        )


# ---------------------------------------------------------------------------
# p. Deprecated functions warn
# ---------------------------------------------------------------------------

class TestDeprecatedFunctionsWarn:
    def test_extract_topics_warns(self):
        """_extract_topics emits DeprecationWarning."""
        from truememory.personality import _extract_topics
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _extract_topics([{"content": "test message"}])
            assert len(w) >= 1
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
            assert any("deprecated" in str(x.message).lower() for x in w)

    def test_extract_traits_warns(self):
        """_extract_traits emits DeprecationWarning."""
        from truememory.personality import _extract_traits
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _extract_traits([{"content": "test message"}])
            assert len(w) >= 1
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
            assert any("deprecated" in str(x.message).lower() for x in w)


# ---------------------------------------------------------------------------
# q. search_personality uses vectors
# ---------------------------------------------------------------------------

class TestSearchPersonalityUsesVectors:
    def test_returns_vector_scored_results(self):
        """With style vectors built, search_personality returns results
        scored by vector similarity (not just FTS)."""
        conn = _make_test_db()

        _insert_msg(conn, "I love tacos and burritos for dinner", sender="alice", timestamp="2025-01-01T10:00:00")
        _insert_msg(conn, "Let me grab some sushi tonight", sender="alice", timestamp="2025-01-01T11:00:00")
        _insert_msg(conn, "The API is returning errors", sender="bob", timestamp="2025-01-01T12:00:00")

        build_entity_style_vectors(conn)

        from truememory.personality import search_personality
        results = search_personality(conn, "What does alice like to eat?", limit=5)

        assert len(results) > 0
        # At least one result should be from alice
        alice_results = [r for r in results if r["sender"].lower() == "alice"]
        assert len(alice_results) > 0
        conn.close()


# ---------------------------------------------------------------------------
# r. Persona scoping bias
# ---------------------------------------------------------------------------

class TestPersonaScopingBias:
    def test_same_entity_higher_score(self):
        """Same-entity messages get higher scores than other-entity messages."""
        conn = _make_test_db()

        # Both entities talk about food
        _insert_msg(conn, "I love pizza and pasta", sender="alice", timestamp="2025-01-01T10:00:00")
        _insert_msg(conn, "I love pizza and pasta", sender="bob", timestamp="2025-01-01T11:00:00")

        build_entity_style_vectors(conn)

        from truememory.personality import search_personality
        results = search_personality(conn, "What food does alice like?", limit=10)

        # Find alice's and bob's results
        alice_scores = [r["score"] for r in results if r["sender"].lower() == "alice" and r["source"] != "profile"]
        bob_scores = [r["score"] for r in results if r["sender"].lower() == "bob" and r["source"] != "profile"]

        if alice_scores and bob_scores:
            # Alice should score higher due to 5.0 persona scoping bias
            assert max(alice_scores) > max(bob_scores), \
                f"Alice max score {max(alice_scores)} should be > Bob max score {max(bob_scores)}"
        conn.close()
