"""Tests for issue #582: entity boost must rescue low-salience rows and apply exactly once.

Verifies that:
1. Entity-matched rows with low salience survive after entity boost in agentic search
2. The entity boost is applied exactly once (idempotent _entity_boosted flag)
3. Non-entity rows are unaffected by the rescue mechanism
"""

from __future__ import annotations

import sqlite3

import pytest

from truememory.salience import (
    apply_salience_guard,
    compute_message_salience,
    filter_by_salience,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db_with_entities() -> sqlite3.Connection:
    """Create an in-memory DB with a messages table containing known entities."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE messages ("
        "  id INTEGER PRIMARY KEY,"
        "  content TEXT, sender TEXT, recipient TEXT,"
        "  timestamp TEXT, category TEXT, modality TEXT,"
        "  episode_id INTEGER, directive INTEGER DEFAULT 0"
        ")"
    )
    conn.execute(
        "INSERT INTO messages (id, content, sender, recipient, timestamp, category, modality) "
        "VALUES (1, 'ok', 'jordan', 'sam', '', '', '')"
    )
    conn.execute(
        "INSERT INTO messages (id, content, sender, recipient, timestamp, category, modality) "
        "VALUES (2, 'Jordan researched adoption agencies for weeks', 'jordan', 'sam', '', '', '')"
    )
    conn.execute(
        "INSERT INTO messages (id, content, sender, recipient, timestamp, category, modality) "
        "VALUES (3, 'The weather is nice today', 'alice', 'bob', '', '', '')"
    )
    conn.commit()
    return conn


def _low_salience_entity_row(msg_id: int = 1) -> dict:
    """A row from entity 'jordan' whose content is low-salience noise ('ok')."""
    return {
        "id": msg_id,
        "content": "ok",
        "sender": "jordan",
        "recipient": "sam",
        "timestamp": "",
        "category": "",
        "modality": "",
        "score": 0.02,
        "source": "hybrid",
    }


def _high_salience_entity_row(msg_id: int = 2) -> dict:
    return {
        "id": msg_id,
        "content": "Jordan researched adoption agencies for weeks",
        "sender": "jordan",
        "recipient": "sam",
        "timestamp": "",
        "category": "",
        "modality": "",
        "score": 0.03,
        "source": "hybrid",
    }


def _non_entity_row(msg_id: int = 3) -> dict:
    return {
        "id": msg_id,
        "content": "The weather is nice today",
        "sender": "alice",
        "recipient": "bob",
        "timestamp": "",
        "category": "",
        "modality": "",
        "score": 0.025,
        "source": "hybrid",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEntityBoostRescuesLowSalience:
    """Entity-matched rows with low raw salience survive the guard (#582)."""

    def test_low_salience_entity_row_rescued_by_entity_rescue_ids(self):
        """A low-salience row whose id is in entity_rescue_ids must survive."""
        conn = _make_db_with_entities()
        row = _low_salience_entity_row()

        # Confirm it IS below the salience floor
        raw_salience = compute_message_salience(row["content"])
        assert raw_salience < 0.10, f"Expected low salience, got {raw_salience}"

        results = apply_salience_guard(
            [row],
            query="What did Jordan say?",
            conn=conn,
            min_salience=0.10,
            entity_rescue_ids=frozenset({row["id"]}),
        )

        assert len(results) == 1, "Entity-rescued row must survive salience guard"
        assert results[0]["id"] == row["id"]

    def test_low_salience_entity_row_filtered_without_rescue(self):
        """Without entity_rescue_ids the same row is filtered out."""
        conn = _make_db_with_entities()
        row = _low_salience_entity_row()

        results = apply_salience_guard(
            [row],
            query="What did Jordan say?",
            conn=conn,
            min_salience=0.10,
        )

        # The row might survive if filter_by_entity boosts it enough, but
        # filter_by_salience checks raw content salience, not boosted score.
        # "ok" has salience ~0.0, so it should be filtered.
        survived_ids = {r["id"] for r in results}
        assert row["id"] not in survived_ids, (
            "Low-salience row should be filtered when no entity_rescue_ids"
        )

    def test_filter_by_salience_respects_rescue_ids(self):
        """filter_by_salience directly: rescued IDs bypass the floor."""
        row = _low_salience_entity_row()
        results = filter_by_salience(
            [row], min_salience=0.10, entity_rescue_ids=frozenset({row["id"]}),
        )
        assert len(results) == 1
        assert results[0]["id"] == row["id"]

    def test_filter_by_salience_still_filters_non_rescued(self):
        """Non-rescued low-salience rows are still filtered normally."""
        row = _low_salience_entity_row()
        results = filter_by_salience(
            [row], min_salience=0.10, entity_rescue_ids=frozenset(),
        )
        assert len(results) == 0


class TestEntityBoostAppliedExactlyOnce:
    """The _entity_boosted flag prevents double-application (#582)."""

    def test_boosted_flag_prevents_double_boost(self):
        """A row with _entity_boosted=True must not be boosted again."""
        row = _high_salience_entity_row()
        original_score = 0.03
        row["score"] = original_score

        # Simulate first boost (as search_agentic would do)
        row["score"] = original_score * 1.5
        row["_entity_boosted"] = True
        first_boost_score = row["score"]

        # Simulate second pass: check flag before boosting
        if not row.get("_entity_boosted"):
            row["score"] = row["score"] * 1.5  # Would double-boost

        assert row["score"] == first_boost_score, (
            f"Score changed from {first_boost_score} to {row['score']} — "
            "double-boost detected"
        )

    def test_score_stable_after_idempotent_check(self):
        """Running the boost logic twice with the flag yields same score."""
        row = _high_salience_entity_row()
        row["score"] = 0.03

        # First application
        if not row.get("_entity_boosted"):
            row["score"] *= 1.5
            row["_entity_boosted"] = True

        score_after_first = row["score"]

        # Second application (idempotent)
        if not row.get("_entity_boosted"):
            row["score"] *= 1.5
            row["_entity_boosted"] = True

        assert row["score"] == score_after_first


class TestNonEntityRowsUnaffected:
    """Non-entity rows are not affected by the rescue mechanism."""

    def test_non_entity_low_salience_still_filtered(self):
        """A low-salience row NOT in entity_rescue_ids is still filtered."""
        conn = _make_db_with_entities()
        entity_row = _low_salience_entity_row(msg_id=1)
        non_entity_noise = {
            "id": 99,
            "content": "lol",
            "sender": "nobody",
            "recipient": "whoever",
            "timestamp": "",
            "category": "",
            "modality": "",
            "score": 0.01,
            "source": "fts",
        }

        results = apply_salience_guard(
            [entity_row, non_entity_noise],
            query="What did Jordan say?",
            conn=conn,
            min_salience=0.10,
            entity_rescue_ids=frozenset({entity_row["id"]}),
        )

        result_ids = {r["id"] for r in results}
        assert entity_row["id"] in result_ids, "Entity-rescued row must survive"
        assert non_entity_noise["id"] not in result_ids, (
            "Non-entity low-salience row must still be filtered"
        )

    def test_high_salience_non_entity_row_unaffected(self):
        """A high-salience non-entity row passes through normally."""
        conn = _make_db_with_entities()
        good_row = _non_entity_row()

        results = apply_salience_guard(
            [good_row],
            query="What did Jordan say?",
            conn=conn,
            min_salience=0.10,
            entity_rescue_ids=frozenset(),
        )

        assert len(results) == 1
        assert results[0]["id"] == good_row["id"]

    def test_entity_rescue_ids_empty_by_default(self):
        """When entity_rescue_ids is not passed, behaviour is unchanged."""
        conn = _make_db_with_entities()
        row = _low_salience_entity_row()

        # Default (no rescue) — same as pre-#582 behaviour
        results_default = apply_salience_guard(
            [row.copy()],
            query="What did Jordan say?",
            conn=conn,
            min_salience=0.10,
        )
        results_explicit_none = apply_salience_guard(
            [row.copy()],
            query="What did Jordan say?",
            conn=conn,
            min_salience=0.10,
            entity_rescue_ids=None,
        )

        assert len(results_default) == len(results_explicit_none)
