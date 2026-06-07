"""Regression tests for issue #467: entity_profiles case-sensitivity.

SQLite TEXT PRIMARY KEY is case-sensitive by default, so 'Josh' and 'josh'
create two separate entity profiles, fragmenting entity data. These tests
verify that entity names are normalized to lowercase before insert/lookup
in both entity_profiles and entity_style_vectors.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np

from truememory.client import Memory
from truememory.storage import create_db
from truememory.personality import (
    build_entity_profiles,
    get_entity_profile,
    update_entity_profile_incremental,
)
from truememory.personality_style_vec import (
    build_entity_style_vectors,
    update_entity_style_vector_incremental,
    get_entity_style_vector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    """In-memory DB with full TrueMemory schema."""
    return create_db(":memory:")


def _insert_msg(conn, content, sender="", recipient="", timestamp=""):
    conn.execute(
        "INSERT INTO messages (content, sender, recipient, timestamp, category, modality) "
        "VALUES (?, ?, ?, ?, '', '')",
        (content, sender, recipient, timestamp),
    )
    conn.commit()


def _count_entity_rows(conn, table="entity_profiles"):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ---------------------------------------------------------------------------
# entity_profiles: build_entity_profiles (batch)
# ---------------------------------------------------------------------------

class TestBuildEntityProfilesCaseInsensitive:
    def test_mixed_case_senders_produce_single_row(self):
        """'Josh' and 'josh' messages must result in one entity_profiles row."""
        conn = _make_db()
        _insert_msg(conn, "Hello from Josh", sender="Josh")
        _insert_msg(conn, "Hello from josh", sender="josh")
        _insert_msg(conn, "Hello from JOSH", sender="JOSH")

        build_entity_profiles(conn)

        count = _count_entity_rows(conn)
        assert count == 1, (
            f"Expected 1 entity_profiles row, got {count} — "
            "case variants are creating separate rows"
        )

    def test_message_count_aggregated(self):
        """All case variants' messages must be counted together."""
        conn = _make_db()
        _insert_msg(conn, "msg1", sender="Josh")
        _insert_msg(conn, "msg2", sender="josh")
        _insert_msg(conn, "msg3", sender="JOSH")

        profiles = build_entity_profiles(conn)

        # All variants should map to the same lowercase key
        assert "josh" in profiles, f"Expected 'josh' key, got keys: {list(profiles.keys())}"
        assert profiles["josh"]["message_count"] == 3


# ---------------------------------------------------------------------------
# entity_profiles: incremental update
# ---------------------------------------------------------------------------

class TestIncrementalProfileCaseInsensitive:
    def test_incremental_updates_merge_case_variants(self):
        """update_entity_profile_incremental must merge 'Josh' and 'josh'."""
        conn = _make_db()
        update_entity_profile_incremental(conn, "Josh", "first message")
        update_entity_profile_incremental(conn, "josh", "second message")

        count = _count_entity_rows(conn)
        assert count == 1, (
            f"Expected 1 entity_profiles row after incremental updates, got {count}"
        )

    def test_get_entity_profile_case_insensitive(self):
        """get_entity_profile('JOSH') must find the profile stored as 'josh'."""
        conn = _make_db()
        update_entity_profile_incremental(conn, "Josh", "a message about coding")

        # Lookup with different case
        profile = get_entity_profile(conn, "josh")
        assert profile is not None, "Lookup with 'josh' found nothing"

        profile_upper = get_entity_profile(conn, "JOSH")
        assert profile_upper is not None, "Lookup with 'JOSH' found nothing"


# ---------------------------------------------------------------------------
# entity_style_vectors: batch build
# ---------------------------------------------------------------------------

class TestBuildStyleVectorsCaseInsensitive:
    def test_mixed_case_senders_one_vector_row(self):
        """'Josh' and 'josh' must produce one entity_style_vectors row."""
        conn = _make_db()
        _insert_msg(conn, "Hey there how are you doing today", sender="Josh")
        _insert_msg(conn, "Im doing great thanks for asking", sender="josh")

        build_entity_style_vectors(conn)

        count = _count_entity_rows(conn, table="entity_style_vectors")
        assert count == 1, (
            f"Expected 1 entity_style_vectors row, got {count}"
        )


# ---------------------------------------------------------------------------
# entity_style_vectors: incremental update
# ---------------------------------------------------------------------------

class TestIncrementalStyleVectorCaseInsensitive:
    def test_incremental_style_merges_case_variants(self):
        """Incremental style vector updates for 'Josh' and 'josh' must merge."""
        conn = _make_db()
        update_entity_style_vector_incremental(conn, "Josh", "hello world this is a test")
        update_entity_style_vector_incremental(conn, "josh", "another test message here")

        count = _count_entity_rows(conn, table="entity_style_vectors")
        assert count == 1, (
            f"Expected 1 entity_style_vectors row, got {count}"
        )

    def test_get_style_vector_case_insensitive(self):
        """get_entity_style_vector('JOSH') must find vector stored under 'josh'."""
        conn = _make_db()
        update_entity_style_vector_incremental(conn, "Josh", "testing the style vector")

        vec = get_entity_style_vector(conn, "josh")
        assert vec is not None, "Lookup with 'josh' found nothing"

        vec_upper = get_entity_style_vector(conn, "JOSH")
        assert vec_upper is not None, "Lookup with 'JOSH' found nothing"


# ---------------------------------------------------------------------------
# Integration: Memory client
# ---------------------------------------------------------------------------

class TestMemoryClientEntityCase:
    def test_add_with_different_case_user_ids(self, monkeypatch):
        """Memory.add with user_id 'Josh' and 'josh' must not fragment profiles."""
        # Mock the embedding model to avoid downloading real models
        mock_model = type("MockModel", (), {
            "encode": staticmethod(
                lambda texts, **kw: np.array(
                    [np.random.rand(256).astype(np.float32)] * len(texts)
                )
            ),
        })()

        mem = Memory(path=":memory:")
        engine = mem._engine

        # Patch the embedding model if vectors are enabled
        try:
            import truememory.vector_search as vs
            monkeypatch.setattr(vs, "_model", mock_model)
            monkeypatch.setattr(vs, "_embedding_dim", 256)
        except Exception:
            pass

        mem.add("I like coffee", user_id="Josh")
        mem.add("I also like tea", user_id="josh")

        # Check entity_profiles table directly
        rows = engine.conn.execute(
            "SELECT entity FROM entity_profiles"
        ).fetchall()
        entities = [r[0] for r in rows]

        assert len(entities) <= 1, (
            f"Expected at most 1 entity_profiles row, got {len(entities)}: {entities}"
        )
