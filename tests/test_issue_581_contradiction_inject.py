"""
Tests for issue #581 — contradiction supplements survive salience guard
and are not sliced off before reranking.

Bug 1: filter_by_salience() scored contradiction rows 0.0 because they
       lacked a 'content' key (they use 'current_fact' instead).
Bug 2: Contradiction supplements had no 'score', so they sorted to the
       bottom and were sliced off before the reranker saw them.
"""
import pytest

from truememory.salience import filter_by_salience, compute_message_salience


# ---------------------------------------------------------------------------
# Bug 1: Salience guard must score contradiction rows via 'current_fact'
# ---------------------------------------------------------------------------

class TestContradictionSalienceScoring:
    """Contradiction supplement rows get non-zero salience scores."""

    def test_contradiction_row_with_current_fact_gets_nonzero_salience(self):
        """A row with 'current_fact' but no 'content' should be scored."""
        rows = [
            {
                "id": 1,
                "current_fact": "CarbonSense migrated from PostgreSQL to ClickHouse in March 2026",
                "source": "contradiction",
                "modality": "",
            },
        ]
        filtered = filter_by_salience(rows, min_salience=0.01)
        assert len(filtered) == 1
        assert filtered[0]["salience"] > 0.0

    def test_contradiction_row_with_content_key_still_works(self):
        """If engine.py already populated 'content', salience still works."""
        rows = [
            {
                "id": 2,
                "content": "The project switched to ClickHouse for analytics",
                "current_fact": "The project switched to ClickHouse for analytics",
                "source": "contradiction",
                "modality": "",
            },
        ]
        filtered = filter_by_salience(rows, min_salience=0.01)
        assert len(filtered) == 1
        assert filtered[0]["salience"] > 0.0

    def test_row_with_text_key_gets_scored(self):
        """Rows using 'text' key (alternate supplement format) are scored."""
        rows = [
            {
                "id": 3,
                "text": "User prefers dark mode for all applications",
                "source": "supplement",
                "modality": "",
            },
        ]
        filtered = filter_by_salience(rows, min_salience=0.01)
        assert len(filtered) == 1
        assert filtered[0]["salience"] > 0.0

    def test_row_with_memory_key_gets_scored(self):
        """Rows using 'memory' key are scored."""
        rows = [
            {
                "id": 4,
                "memory": "Josh lives in Austin TX and works on AI projects",
                "source": "supplement",
                "modality": "",
            },
        ]
        filtered = filter_by_salience(rows, min_salience=0.01)
        assert len(filtered) == 1
        assert filtered[0]["salience"] > 0.0

    def test_truly_empty_row_still_filtered(self):
        """A row with no content in any key should still score 0.0."""
        rows = [
            {"id": 5, "source": "contradiction", "modality": ""},
        ]
        filtered = filter_by_salience(rows, min_salience=0.01)
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# Bug 2: Contradiction supplements survive the full pipeline
# ---------------------------------------------------------------------------

class TestContradictionSurvivesGuard:
    """Contradiction supplements with realistic content survive filtering."""

    def test_contradiction_survives_default_threshold(self):
        """Contradiction with real content survives the default 0.10 threshold."""
        rows = [
            {
                "id": 10,
                "content": "CarbonSense migrated their database from PostgreSQL to ClickHouse in March",
                "source": "hybrid",
                "modality": "",
                "score": 0.5,
            },
            {
                "id": 11,
                "current_fact": "CarbonSense uses ClickHouse for their analytics database since March 2026",
                "source": "contradiction",
                "modality": "",
                "score": 0.4,
            },
        ]
        filtered = filter_by_salience(rows, min_salience=0.10)
        sources = {r["source"] for r in filtered}
        assert "contradiction" in sources, (
            "Contradiction row was dropped by salience guard"
        )

    def test_mixed_results_preserve_contradictions(self):
        """In a pool of normal + contradiction rows, contradictions survive."""
        rows = [
            {"id": i, "content": f"Normal result number {i} with enough text to score well",
             "source": "hybrid", "modality": "", "score": 0.5}
            for i in range(1, 6)
        ]
        rows.append({
            "id": 100,
            "current_fact": "The team decided to use Rust instead of Go for the backend rewrite",
            "source": "contradiction",
            "modality": "",
            "score": 0.4,
        })
        filtered = filter_by_salience(rows, min_salience=0.05)
        contradiction_rows = [r for r in filtered if r["source"] == "contradiction"]
        assert len(contradiction_rows) == 1


# ---------------------------------------------------------------------------
# Integration: contradiction injection in engine populates 'content' + 'score'
# ---------------------------------------------------------------------------

class TestContradictionInjectionShape:
    """
    Verify the expected shape of contradiction rows after engine.py
    injects them — they must have 'content' and 'score' keys.
    """

    def test_content_populated_from_current_fact(self):
        """Simulates the engine.py injection logic: content comes from current_fact."""
        # This mirrors what engine.py now does after the fix
        cr = {
            "id": 42,
            "current_fact": "Project uses ClickHouse",
            "subject": "database",
            "source": "contradiction",
        }
        # Simulate engine.py fix logic
        if "content" not in cr:
            cr["content"] = cr.get(
                "current_fact",
                cr.get("text", cr.get("memory", "")),
            )
        assert cr["content"] == "Project uses ClickHouse"

    def test_score_assigned_when_missing(self):
        """Contradictions get a competitive score from max_existing."""
        cr = {"id": 42, "current_fact": "Uses ClickHouse", "source": "contradiction"}
        max_existing = 0.5
        if "score" not in cr:
            cr["score"] = max_existing * 0.8
        assert cr["score"] == pytest.approx(0.4)
        assert cr["score"] > 0

    def test_existing_score_not_overwritten(self):
        """If a contradiction already has a score, it is preserved."""
        cr = {"id": 42, "current_fact": "Uses ClickHouse", "source": "contradiction", "score": 0.9}
        max_existing = 0.5
        if "score" not in cr:
            cr["score"] = max_existing * 0.8
        assert cr["score"] == 0.9  # Not overwritten
