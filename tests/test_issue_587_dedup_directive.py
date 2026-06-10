"""Tests for issue #587: dedup must never UPDATE or SKIP against directive rows.

Directives are sacred standing instructions (directive=1). The dedup
pipeline must treat them as invisible — new memories that happen to be
similar to directives should always be stored as ADD, never UPDATE or SKIP.
"""

from unittest.mock import MagicMock

from truememory.ingest.dedup import DedupAction, check_duplicate


def _make_memory(results: list[dict]) -> MagicMock:
    """Create a mock memory object whose search_vectors returns *results*."""
    mem = MagicMock()
    mem.search_vectors.return_value = results
    return mem


# ── Directive guard tests ────────────────────────────────────────────────

def test_similar_directive_not_skipped():
    """A new memory similar to a directive must be ADDed, not SKIPped."""
    memory = _make_memory([
        {
            "id": 1,
            "content": "Always respond in formal English",
            "score": 0.95,  # Very high — would normally trigger SKIP
            "directive": True,
        },
    ])
    decision = check_duplicate(
        "Always respond in formal English",
        memory,
    )
    assert decision.action == DedupAction.ADD, (
        f"Expected ADD when only candidate is a directive, got {decision.action}: {decision.reason}"
    )


def test_identical_directive_not_updated():
    """Storing a memory identical to a directive must ADD, not UPDATE."""
    memory = _make_memory([
        {
            "id": 42,
            "content": "User prefers dark mode in all apps",
            "score": 0.99,
            "directive": True,
        },
    ])
    decision = check_duplicate(
        "User prefers dark mode in all apps",
        memory,
    )
    assert decision.action == DedupAction.ADD, (
        f"Expected ADD against directive, got {decision.action}: {decision.reason}"
    )


def test_directive_filtered_non_directive_still_deduped():
    """Non-directive candidates still participate in dedup normally."""
    memory = _make_memory([
        {
            "id": 1,
            "content": "Always respond in formal English",
            "score": 0.95,
            "directive": True,
        },
        {
            "id": 2,
            "content": "Always respond in formal English",
            "score": 0.93,
            "directive": False,
        },
    ])
    decision = check_duplicate(
        "Always respond in formal English",
        memory,
    )
    # The directive (id=1) is filtered out; the non-directive (id=2)
    # at 0.93 triggers the near-exact-match SKIP path.
    assert decision.action == DedupAction.SKIP, (
        f"Expected SKIP against non-directive near-exact match, got {decision.action}: {decision.reason}"
    )


def test_all_candidates_are_directives():
    """When all candidates are directives, dedup should ADD."""
    memory = _make_memory([
        {"id": 10, "content": "Directive A", "score": 0.80, "directive": True},
        {"id": 11, "content": "Directive B", "score": 0.70, "directive": True},
    ])
    decision = check_duplicate("Directive A revised", memory)
    assert decision.action == DedupAction.ADD, (
        f"Expected ADD when all candidates are directives, got {decision.action}"
    )


def test_non_directive_dedup_still_works():
    """Normal dedup between non-directives is unaffected by the guard."""
    memory = _make_memory([
        {
            "id": 5,
            "content": "Josh lives in Austin TX",
            "score": 0.96,
            "directive": False,
        },
    ])
    decision = check_duplicate(
        "Josh lives in Austin TX",
        memory,
    )
    assert decision.action == DedupAction.SKIP, (
        f"Expected SKIP for near-exact non-directive match, got {decision.action}: {decision.reason}"
    )


def test_no_directive_field_defaults_false():
    """Results missing the 'directive' key should not be filtered out."""
    memory = _make_memory([
        {
            "id": 99,
            "content": "Josh likes coffee",
            "score": 0.95,
            # No 'directive' key — should default to False (not filtered)
        },
    ])
    decision = check_duplicate("Josh likes coffee", memory)
    assert decision.action == DedupAction.SKIP, (
        f"Expected SKIP for near-exact match without directive key, got {decision.action}"
    )
