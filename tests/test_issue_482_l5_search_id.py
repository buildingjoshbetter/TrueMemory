"""Regression tests for issue #482: L5 search results silently dropped.

search_contradictions() and search_consolidated() return dicts without
an 'id' key, causing engine.py to silently discard all L5 results.
"""
from __future__ import annotations

import sqlite3

import pytest

from truememory.consolidation import search_contradictions, search_consolidated
from truememory.storage import create_db


def _make_db_with_facts():
    """Create an in-memory DB with fact_timeline and summaries data."""
    conn = create_db(":memory:")

    conn.execute(
        "INSERT INTO messages (id, content, sender, timestamp) "
        "VALUES (1, 'CarbonSense uses PostgreSQL', 'alice', '2025-01-01')"
    )
    conn.execute(
        "INSERT INTO messages (id, content, sender, timestamp) "
        "VALUES (2, 'CarbonSense migrated to ClickHouse', 'alice', '2025-06-01')"
    )

    conn.execute(
        "INSERT INTO fact_timeline (subject, fact, timestamp, source_message_id) "
        "VALUES ('CarbonSense database', 'PostgreSQL', '2025-01-01', 1)"
    )
    conn.execute(
        "INSERT INTO fact_timeline (subject, fact, timestamp, superseded_by, source_message_id) "
        "VALUES ('CarbonSense database', 'PostgreSQL', '2025-01-01', 2, 1)"
    )
    conn.execute(
        "INSERT INTO fact_timeline (subject, fact, timestamp, source_message_id) "
        "VALUES ('CarbonSense database', 'ClickHouse', '2025-06-01', 2)"
    )

    conn.execute(
        "INSERT INTO summaries (period, start_date, end_date, entity, summary, key_facts, message_ids) "
        "VALUES ('monthly', '2025-01-01', '2025-01-31', 'alice', "
        "'Alice discussed CarbonSense database migration', "
        "'[\"migrated from PostgreSQL to ClickHouse\"]', '[1,2]')"
    )
    conn.commit()
    return conn


class TestIssue482L5SearchId:
    """Verify L5 search results include 'id' key for engine merging."""

    def test_issue_482_contradiction_results_have_id(self):
        """search_contradictions() results must have an 'id' key."""
        conn = _make_db_with_facts()
        results = search_contradictions(conn, "What database does CarbonSense use?")
        assert len(results) > 0, "Expected contradiction results for CarbonSense query"
        for r in results:
            assert "id" in r, (
                f"Contradiction result missing 'id' key. Keys: {list(r.keys())}"
            )

    def test_issue_482_consolidated_results_have_id(self):
        """search_consolidated() results must have an 'id' key."""
        conn = _make_db_with_facts()
        results = search_consolidated(conn, "CarbonSense database migration")
        assert len(results) > 0, "Expected consolidated results for CarbonSense query"
        for r in results:
            assert "id" in r, (
                f"Consolidated result missing 'id' key. Keys: {list(r.keys())}"
            )

    def test_issue_482_l5_results_merged_into_search(self):
        """Integration: L5 results must appear in engine search output."""
        from unittest.mock import patch, MagicMock
        import numpy as np
        from truememory.client import Memory

        m = Memory(path=":memory:")
        fake_embedding = np.random.rand(256).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.array(
            [fake_embedding] * len(texts)
        )

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            m.add(content="CarbonSense uses PostgreSQL for analytics", user_id="alice")
            m.add(content="CarbonSense migrated to ClickHouse last month", user_id="alice")

        conn = m._engine.conn
        conn.execute(
            "INSERT INTO fact_timeline (subject, fact, timestamp, superseded_by, source_message_id) "
            "VALUES ('CarbonSense database', 'PostgreSQL', '2025-01-01', 2, 1)"
        )
        conn.execute(
            "INSERT INTO fact_timeline (subject, fact, timestamp, source_message_id) "
            "VALUES ('CarbonSense database', 'ClickHouse', '2025-06-01', 2)"
        )
        conn.execute(
            "INSERT INTO summaries (period, start_date, end_date, entity, summary, key_facts, message_ids) "
            "VALUES ('monthly', '2025-01-01', '2025-01-31', '', "
            "'CarbonSense database migration from PostgreSQL to ClickHouse', "
            "'[\"ClickHouse\"]', '[1,2]')"
        )
        conn.commit()

        m._engine._has_consolidation = True

        results = m.search("What database does CarbonSense use?")

        sources = [r.get("source", "") for r in results]
        has_l5 = any(
            s in ("contradiction", "summary", "fact_timeline")
            for s in sources
        )
        assert has_l5, (
            f"No L5 results in search output. Sources found: {sources}"
        )
