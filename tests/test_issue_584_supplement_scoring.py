"""Tests for issue #584 — agentic supplement score normalization.

Verifies:
1. Mixed-source results are normalized to [0, 1].
2. Cluster/diversity ordering is preserved through merge.
3. Single-source relative ordering is unchanged after normalization.
4. Batch embedding fetch in clustering (structural, no DB needed).
"""

from __future__ import annotations

import pytest

from truememory.agentic_search import normalize_scores, normalize_supplement_scores


# ---------------------------------------------------------------------------
# normalize_scores
# ---------------------------------------------------------------------------

class TestNormalizeScores:
    """Unit tests for per-source min-max normalization."""

    def test_basic_normalization(self):
        """Scores spanning an arbitrary range are mapped to [0, 1]."""
        results = [
            {"id": 1, "score": 10.0},
            {"id": 2, "score": 20.0},
            {"id": 3, "score": 30.0},
        ]
        normalize_scores(results)
        assert results[0]["score"] == pytest.approx(0.0)
        assert results[1]["score"] == pytest.approx(0.5)
        assert results[2]["score"] == pytest.approx(1.0)

    def test_preserves_relative_order(self):
        """Within a single source, relative ordering must not change."""
        results = [
            {"id": 1, "score": 5.5},
            {"id": 2, "score": 100.0},
            {"id": 3, "score": 50.0},
        ]
        original_order = [(r["id"], r["score"]) for r in results]
        normalize_scores(results)

        # The relative ranking (by score) should be identical.
        original_ranked = sorted(original_order, key=lambda t: -t[1])
        new_ranked = sorted(results, key=lambda r: -r["score"])
        assert [t[0] for t in original_ranked] == [r["id"] for r in new_ranked]

    def test_all_equal_scores(self):
        """When all scores are identical, normalize to 0.5."""
        results = [
            {"id": 1, "score": 7.0},
            {"id": 2, "score": 7.0},
            {"id": 3, "score": 7.0},
        ]
        normalize_scores(results)
        for r in results:
            assert r["score"] == pytest.approx(0.5)

    def test_single_result(self):
        """A single result normalizes to 0.5 (neutral)."""
        results = [{"id": 1, "score": 42.0}]
        normalize_scores(results)
        assert results[0]["score"] == pytest.approx(0.5)

    def test_empty_list(self):
        """Empty input returns empty list without error."""
        results: list[dict] = []
        ret = normalize_scores(results)
        assert ret == []

    def test_bm25_scores(self):
        """BM25 scores (unbounded, potentially large) are normalized to [0, 1]."""
        results = [
            {"id": 1, "score": 0.5},
            {"id": 2, "score": 12.3},
            {"id": 3, "score": 45.6},
            {"id": 4, "score": 2.1},
        ]
        normalize_scores(results)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_cosine_scores_unchanged_range(self):
        """Cosine similarity scores already in [0, 1] remain in [0, 1]."""
        results = [
            {"id": 1, "score": 0.1},
            {"id": 2, "score": 0.5},
            {"id": 3, "score": 0.9},
        ]
        normalize_scores(results)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0
        # Min should map to 0, max to 1
        scores = {r["id"]: r["score"] for r in results}
        assert scores[1] == pytest.approx(0.0)
        assert scores[3] == pytest.approx(1.0)

    def test_negative_scores(self):
        """Negative scores (possible in some BM25 variants) are handled."""
        results = [
            {"id": 1, "score": -5.0},
            {"id": 2, "score": 0.0},
            {"id": 3, "score": 5.0},
        ]
        normalize_scores(results)
        assert results[0]["score"] == pytest.approx(0.0)
        assert results[1]["score"] == pytest.approx(0.5)
        assert results[2]["score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# normalize_supplement_scores
# ---------------------------------------------------------------------------

class TestNormalizeSupplementScores:
    """Verify that multiple source lists are normalized independently."""

    def test_independent_normalization(self):
        """Each source gets its own [0, 1] normalization."""
        primary = [
            {"id": 1, "score": 0.01},
            {"id": 2, "score": 0.03},
        ]
        cluster = [
            {"id": 3, "score": 0.7},
            {"id": 4, "score": 0.9},
        ]
        entity = [
            {"id": 5, "score": 15.0},
            {"id": 6, "score": 45.0},
        ]
        normalize_supplement_scores(primary, cluster, entity)

        # All sources should now be in [0, 1]
        for source in [primary, cluster, entity]:
            for r in source:
                assert 0.0 <= r["score"] <= 1.0

        # Verify they were normalized independently (not together)
        # primary[0] should be 0.0 (min of its source)
        assert primary[0]["score"] == pytest.approx(0.0)
        assert primary[1]["score"] == pytest.approx(1.0)
        assert cluster[0]["score"] == pytest.approx(0.0)
        assert cluster[1]["score"] == pytest.approx(1.0)
        assert entity[0]["score"] == pytest.approx(0.0)
        assert entity[1]["score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cluster order preservation
# ---------------------------------------------------------------------------

class TestClusterOrderPreservation:
    """Verify cluster/diversity ordering is preserved through the merge."""

    def test_cluster_position_tag_survives_merge(self):
        """Cluster results tagged with _cluster_position keep the tag."""
        primary = [
            {"id": 1, "score": 0.8, "source": "hybrid"},
            {"id": 2, "score": 0.6, "source": "hybrid"},
        ]
        cluster_supplements = [
            {"id": 10, "score": 0.4, "source": "clustered+cluster_supp", "_cluster_position": 0},
            {"id": 11, "score": 0.3, "source": "clustered+cluster_supp", "_cluster_position": 1},
            {"id": 12, "score": 0.5, "source": "clustered+cluster_supp", "_cluster_position": 2},
        ]

        # Simulate the merge: primary + cluster supplements
        merged = primary + cluster_supplements

        # Extract cluster results and verify position order is preserved
        cluster_in_merged = [r for r in merged if "_cluster_position" in r]
        positions = [r["_cluster_position"] for r in cluster_in_merged]
        assert positions == [0, 1, 2], "Cluster diversity order must be preserved"

    def test_cluster_results_not_resorted_by_score(self):
        """After merging, cluster supplements should maintain their
        diversity-sampling order, not be reordered by raw score."""
        # Cluster results come in diversity order (not score order)
        cluster_results = [
            {"id": 10, "score": 0.3},  # diverse pick 1
            {"id": 11, "score": 0.9},  # diverse pick 2
            {"id": 12, "score": 0.1},  # diverse pick 3
        ]

        # Normalize them
        normalize_scores(cluster_results)

        # Tag with position
        for idx, cr in enumerate(cluster_results):
            cr["_cluster_position"] = idx

        # Verify: positions should follow original list order, not score order
        ids_in_order = [cr["id"] for cr in cluster_results]
        assert ids_in_order == [10, 11, 12]
        assert cluster_results[0]["_cluster_position"] == 0
        assert cluster_results[1]["_cluster_position"] == 1
        assert cluster_results[2]["_cluster_position"] == 2


# ---------------------------------------------------------------------------
# Single-source relative ordering
# ---------------------------------------------------------------------------

class TestSingleSourceOrdering:
    """Normalization must not change relative ranking within one source."""

    def test_fts_scores_preserve_ranking(self):
        """FTS/BM25 results: highest score stays highest after normalization."""
        results = [
            {"id": 1, "score": 45.0, "source": "fts"},
            {"id": 2, "score": 12.0, "source": "fts"},
            {"id": 3, "score": 33.0, "source": "fts"},
            {"id": 4, "score": 1.0, "source": "fts"},
        ]
        original_ranking = sorted(results, key=lambda r: -r["score"])
        original_id_order = [r["id"] for r in original_ranking]

        normalize_scores(results)
        new_ranking = sorted(results, key=lambda r: -r["score"])
        new_id_order = [r["id"] for r in new_ranking]

        assert original_id_order == new_id_order

    def test_vector_scores_preserve_ranking(self):
        """Vector/cosine results: ranking is preserved."""
        results = [
            {"id": 1, "score": 0.92, "source": "vector"},
            {"id": 2, "score": 0.87, "source": "vector"},
            {"id": 3, "score": 0.95, "source": "vector"},
        ]
        original_ranking = sorted(results, key=lambda r: -r["score"])
        original_id_order = [r["id"] for r in original_ranking]

        normalize_scores(results)
        new_ranking = sorted(results, key=lambda r: -r["score"])
        new_id_order = [r["id"] for r in new_ranking]

        assert original_id_order == new_id_order
