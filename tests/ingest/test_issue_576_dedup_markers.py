"""Tests for marker-gated dedup arbitration (issue #576).

The >0.92 cosine short-circuit used to silently SKIP all high-similarity
candidates.  This dropped ~20% of genuine updates (e.g. dose changes,
rate changes) that embed close to the original fact.

After the fix, the short-circuit checks for update markers first.
If markers are present, the candidate is routed to arbitration (LLM or
heuristic) instead of being dropped.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from truememory.ingest.dedup import (
    DedupAction,
    _has_update_markers,
    _heuristic_dedup,
    check_duplicate,
)


# ── _has_update_markers unit tests ──────────────────────────────────────


class TestHasUpdateMarkers:
    """Verify the marker regex patterns detect genuine updates."""

    def test_changed_to(self):
        assert _has_update_markers("medication dose changed to 10mg")

    def test_changed_from(self):
        assert _has_update_markers("medication dose changed from 5mg to 10mg")

    def test_now_uses(self):
        assert _has_update_markers("now uses Python 3.12")

    def test_now_prefers(self):
        assert _has_update_markers("now prefers dark mode")

    def test_no_longer(self):
        assert _has_update_markers("no longer takes ibuprofen")

    def test_not_anymore(self):
        assert _has_update_markers("not anymore interested in React")

    def test_switched_to(self):
        assert _has_update_markers("switched to Vim from VS Code")

    def test_switched_from(self):
        assert _has_update_markers("switched from npm to bun")

    def test_instead_of(self):
        assert _has_update_markers("uses bun instead of npm")

    def test_moved_to(self):
        assert _has_update_markers("moved to Austin from SF")

    def test_replaced(self):
        assert _has_update_markers("replaced the old router")

    def test_updated(self):
        assert _has_update_markers("updated the medication schedule")

    def test_was_now(self):
        assert _has_update_markers("rate was 6.5% now 6.25%")

    def test_formerly(self):
        assert _has_update_markers("formerly lived in New York")

    def test_previously(self):
        assert _has_update_markers("previously used npm")

    def test_used_to(self):
        assert _has_update_markers("used to take 5mg daily")

    def test_number_change_arrow(self):
        assert _has_update_markers("dose 5mg -> 10mg")

    def test_number_change_to(self):
        assert _has_update_markers("rate 6.5 to 6.25")

    def test_number_change_unicode_arrow(self):
        assert _has_update_markers("weight 180 → 175 lbs")

    def test_since_date(self):
        assert _has_update_markers("vegetarian since January")

    def test_as_of(self):
        assert _has_update_markers("as of March 2026, lives in Austin")

    # ── Negative cases: no markers ──

    def test_plain_fact_no_markers(self):
        assert not _has_update_markers("likes coffee")

    def test_simple_preference(self):
        assert not _has_update_markers("prefers dark mode")

    def test_bare_statement(self):
        assert not _has_update_markers("works at Acme Corp")


# ── Integration: check_duplicate with mocked vector search ─────────────


def _make_memory(results: list[dict]):
    """Return a mock memory object with canned search_vectors results."""
    mem = MagicMock()
    mem.search_vectors.return_value = results
    return mem


class TestHighSimilarityShortCircuit:
    """The >0.92 path should SKIP true duplicates but UPDATE marker-bearing facts."""

    def test_true_duplicate_still_skipped(self):
        """'prefers dark mode' vs 'prefers dark mode' -> SKIP."""
        mem = _make_memory([
            {"id": 1, "content": "prefers dark mode", "score": 0.97},
        ])
        result = check_duplicate("prefers dark mode", mem)
        assert result.action == DedupAction.SKIP

    def test_medication_dose_change_not_skipped(self):
        """'dose changed from 5mg to 10mg' vs 'takes medication 5mg daily' -> UPDATE."""
        mem = _make_memory([
            {"id": 1, "content": "takes medication 5mg daily", "score": 0.95},
        ])
        result = check_duplicate("medication dose changed from 5mg to 10mg", mem)
        assert result.action == DedupAction.UPDATE, (
            f"Genuine dose change should UPDATE, got {result.action}: {result.reason}"
        )

    def test_mortgage_rate_change_not_skipped(self):
        """'mortgage rate is 6.25%' with number-change marker -> UPDATE."""
        mem = _make_memory([
            {"id": 1, "content": "mortgage rate is 6.5%", "score": 0.96},
        ])
        # The fact itself contains "6.5 to 6.25" style language
        result = check_duplicate(
            "mortgage rate changed to 6.25% (was 6.5%)", mem
        )
        assert result.action == DedupAction.UPDATE, (
            f"Rate change should UPDATE, got {result.action}: {result.reason}"
        )

    def test_minor_rewording_without_markers_skipped(self):
        """'likes coffee a lot' vs 'likes coffee' -> SKIP (no update marker)."""
        mem = _make_memory([
            {"id": 1, "content": "likes coffee", "score": 0.94},
        ])
        result = check_duplicate("likes coffee a lot", mem)
        assert result.action == DedupAction.SKIP, (
            f"Minor rewording without markers should SKIP, got {result.action}"
        )

    def test_no_longer_marker_triggers_update(self):
        """'no longer drinks coffee' vs 'likes coffee' -> UPDATE."""
        mem = _make_memory([
            {"id": 1, "content": "likes coffee", "score": 0.93},
        ])
        result = check_duplicate("no longer drinks coffee", mem)
        assert result.action == DedupAction.UPDATE, (
            f"'no longer' marker should trigger UPDATE, got {result.action}"
        )

    def test_switched_marker_triggers_update(self):
        """'switched to tea from coffee' -> UPDATE."""
        mem = _make_memory([
            {"id": 1, "content": "drinks coffee daily", "score": 0.93},
        ])
        result = check_duplicate("switched to tea from coffee", mem)
        assert result.action == DedupAction.UPDATE, (
            f"'switched to' marker should trigger UPDATE, got {result.action}"
        )


class TestHeuristicDedupMarkers:
    """The heuristic path's marker detection should also use the centralized function."""

    def test_changed_marker_at_moderate_similarity(self):
        result = _heuristic_dedup(
            "medication dose changed from 5mg to 10mg",
            "takes medication 5mg daily",
            existing_id=1,
            similarity=0.80,
        )
        assert result.action == DedupAction.UPDATE

    def test_now_marker_at_moderate_similarity(self):
        result = _heuristic_dedup(
            "now uses bun for package management",
            "uses npm for package management",
            existing_id=1,
            similarity=0.80,
        )
        assert result.action == DedupAction.UPDATE

    def test_no_marker_at_moderate_similarity_adds(self):
        """Without markers, moderate similarity -> ADD (distinct fact)."""
        result = _heuristic_dedup(
            "enjoys hiking on weekends",
            "enjoys swimming on weekends",
            existing_id=1,
            similarity=0.55,
        )
        assert result.action == DedupAction.ADD
