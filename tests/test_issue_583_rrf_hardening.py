"""
Tests for issue #583: RRF id-space hardening.

Covers:
  1. id=0 results are preserved (not collapsed or dropped)
  2. Mixed int/str ID tiebreaks don't crash and produce stable ordering
  3. HyDE fusion errors log a warning but don't drop all results
  4. Normal RRF behaviour is unchanged
"""

from __future__ import annotations

import logging

import pytest

from truememory.hybrid import reciprocal_rank_fusion


# ── helpers ───────────────────────────────────────────────────────────

def _make_doc(doc_id, content="test"):
    """Build a minimal result dict."""
    return {"id": doc_id, "content": content}


# ── 1. id=0 preserved ────────────────────────────────────────────────

class TestIdZeroPreserved:
    """id=0 is a valid SQLite rowid and must not be collapsed or dropped."""

    def test_single_list_id_zero(self):
        results = reciprocal_rank_fusion([[_make_doc(0, "zero"), _make_doc(1, "one")]])
        ids = {r["id"] for r in results}
        assert 0 in ids, "id=0 was dropped"
        assert 1 in ids

    def test_two_lists_id_zero_merges(self):
        """id=0 appearing in two lists should get a higher fused score."""
        list_a = [_make_doc(0, "a"), _make_doc(1, "b")]
        list_b = [_make_doc(0, "a"), _make_doc(2, "c")]
        results = reciprocal_rank_fusion([list_a, list_b])
        ids = [r["id"] for r in results]
        assert ids[0] == 0, "id=0 should rank first (appears in both lists)"
        assert len(results) == 3

    def test_id_zero_not_collapsed_with_other_zeros(self):
        """Multiple docs all with id=0 in separate lists should merge correctly."""
        list_a = [_make_doc(0, "version_a")]
        list_b = [_make_doc(0, "version_b_longer_fields")]
        results = reciprocal_rank_fusion([list_a, list_b])
        assert len(results) == 1  # same id, merged
        assert results[0]["rrf_score"] > 0


# ── 2. Type-stable tiebreaks ─────────────────────────────────────────

class TestTypeStableTiebreaks:
    """Mixed int/str IDs must not crash the sort and must produce stable order."""

    def test_mixed_int_str_ids_no_crash(self):
        """Tiebreaking between int and str IDs must not raise TypeError."""
        list_a = [_make_doc(1), _make_doc("abc")]
        list_b = [_make_doc("abc"), _make_doc(1)]
        # Both docs have the same RRF score — tiebreak must handle mixed types
        results = reciprocal_rank_fusion([list_a, list_b])
        ids = {str(r["id"]) for r in results}
        assert "1" in ids
        assert "abc" in ids

    def test_int_and_str_same_value_merge(self):
        """int(42) and str('42') refer to the same logical doc — they should merge."""
        list_a = [_make_doc(42, "int version")]
        list_b = [_make_doc("42", "str version")]
        results = reciprocal_rank_fusion([list_a, list_b])
        # After str-normalisation they should merge into one entry
        assert len(results) == 1
        assert results[0]["rrf_score"] > 0

    def test_deterministic_ordering(self):
        """Same inputs should always produce same output order."""
        lists = [[_make_doc(3), _make_doc(1), _make_doc(2)]]
        r1 = reciprocal_rank_fusion(lists)
        r2 = reciprocal_rank_fusion(lists)
        assert [r["id"] for r in r1] == [r["id"] for r in r2]


# ── 3. HyDE fusion error logging ─────────────────────────────────────

class TestHydeFusionErrorLogging:
    """HyDE errors must log a warning, not silently swallow."""

    def test_hyde_multi_search_logs_warning_on_search_failure(self, caplog):
        """hyde_multi_search should log.warning (not debug) when a HyDE search fails."""
        from truememory.hyde import hyde_multi_search
        import unittest.mock as mock

        call_count = {"n": 0}

        def _mock_search_hybrid(conn, query, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [_make_doc(1, "result")]
            raise RuntimeError("simulated HyDE search failure")

        def _mock_llm(prompt):
            return "A hypothetical conversation about the topic with details."

        # hyde_multi_search does `from truememory.hybrid import search_hybrid`
        # inside the function body, so we patch the source module.
        with mock.patch("truememory.hybrid.search_hybrid", side_effect=_mock_search_hybrid):
            with caplog.at_level(logging.WARNING, logger="truememory.hyde"):
                results = hyde_multi_search(
                    conn=None,
                    query="test query",
                    llm_fn=_mock_llm,
                    limit=5,
                )
        assert len(results) >= 1
        assert any("HyDE" in rec.message for rec in caplog.records), (
            "Expected a WARNING-level log about HyDE failure"
        )

    def test_hyde_search_logs_warning_on_fusion_failure(self, caplog):
        """hyde_search should log.warning when the HyDE search itself fails."""
        from truememory.hyde import hyde_search
        import unittest.mock as mock

        call_count = {"n": 0}

        def _mock_search(conn, query, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [_make_doc(1)]
            raise RuntimeError("boom")

        def _mock_llm(prompt):
            return "A hypothetical document with enough characters to pass."

        with mock.patch("truememory.hybrid.search_hybrid", side_effect=_mock_search):
            with caplog.at_level(logging.WARNING, logger="truememory.hyde"):
                results = hyde_search(
                    conn=None, query="test", llm_fn=_mock_llm, limit=5,
                )
        assert len(results) >= 1, "Should fall back to original results"
        assert any("HyDE" in r.message and r.levelno >= logging.WARNING for r in caplog.records)


# ── 4. Normal RRF behaviour unchanged ────────────────────────────────

class TestNormalRRFBehaviour:
    """Regression tests: standard RRF semantics are preserved."""

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_single_list(self):
        docs = [_make_doc(10), _make_doc(20), _make_doc(30)]
        results = reciprocal_rank_fusion([docs])
        assert len(results) == 3
        # First doc should have the highest score
        assert results[0]["id"] == 10

    def test_two_list_fusion(self):
        list_a = [_make_doc(1), _make_doc(2)]
        list_b = [_make_doc(2), _make_doc(3)]
        results = reciprocal_rank_fusion([list_a, list_b])
        # doc 2 appears in both lists — should be ranked first
        assert results[0]["id"] == 2

    def test_scores_are_positive(self):
        results = reciprocal_rank_fusion([[_make_doc(1)]])
        assert all(r["rrf_score"] > 0 for r in results)
        assert all(r["score"] == r["rrf_score"] for r in results)

    def test_k_parameter(self):
        docs = [_make_doc(1)]
        r60 = reciprocal_rank_fusion([docs], k=60)
        r10 = reciprocal_rank_fusion([docs], k=10)
        # Lower k means higher score for rank-1 doc
        assert r10[0]["rrf_score"] > r60[0]["rrf_score"]

    def test_none_id_skipped(self):
        """Docs with id=None should be silently skipped."""
        results = reciprocal_rank_fusion([[{"id": None, "content": "x"}, _make_doc(1)]])
        assert len(results) == 1
        assert results[0]["id"] == 1
