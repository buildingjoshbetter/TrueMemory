"""Tests for issue #632: score-space contract.

`Memory.search_vectors()` falls back to the full `search()` pipeline whose
scores are RELATIVELY normalized (FTS top hit pinned to 1.0; reranker fused
scores min-max pinned). Several consumers wrongly compared those relative
scores against ABSOLUTE cosine thresholds:

  - dedup.py:        SKIP when score > 0.92
  - user_prompt_submit.py: novelty SKIP when score > 0.85
  - encoding_gate.py: pays the full reranker pipeline per fact

Net bug: when the embedder is dead (FTS-only / degraded mode), any incoming
fact sharing ONE keyword with a stored memory gets a relative score near 1.0
and is silently dropped as a "duplicate".

The fix tags each result with ``score_space`` (``"cosine"`` vs
``"relative"``). Consumers apply absolute cosine thresholds ONLY to
cosine-space scores; otherwise they fall back to the scale-free word-overlap
heuristic.

These tests use plain mock memories — no embeddings, no model loads.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from truememory.ingest.dedup import DedupAction, check_duplicate


def _make_memory(results: list[dict]) -> MagicMock:
    mem = MagicMock()
    mem.search_vectors.return_value = results
    return mem


# ── (a) Core regression: relative score must NOT drop a non-duplicate ─────

def test_relative_score_does_not_drop_keyword_overlap_nonduplicate():
    """FTS-only mode: a relative score of 1.0 on a fact that shares only one
    keyword (and is NOT a true duplicate) must NOT be SKIPped."""
    memory = _make_memory([
        {
            "id": 1,
            "content": "User lives in Austin",
            # In FTS-only mode the top hit is pinned to 1.0 regardless of
            # true similarity — sharing the keyword "user" is enough.
            "score": 1.0,
            "score_space": "relative",
            "directive": False,
        },
    ])
    decision = check_duplicate(
        "User works at a startup",  # shares "user" only; distinct fact
        memory,
        config=None,  # no LLM -> heuristic fallback path
    )
    assert decision.action == DedupAction.ADD, (
        f"Relative 1.0 score must not trigger SKIP on a non-duplicate; "
        f"got {decision.action}: {decision.reason}"
    )


def test_relative_score_high_but_real_paraphrase_still_word_overlap_deduped():
    """Even in relative-score mode, a true rephrased duplicate (high word
    overlap) is still caught by the scale-free heuristic."""
    memory = _make_memory([
        {
            "id": 7,
            "content": "User prefers dark mode in all apps",
            "score": 1.0,
            "score_space": "relative",
            "directive": False,
        },
    ])
    decision = check_duplicate(
        "User prefers dark mode in all apps",  # identical -> substring SKIP
        memory,
        config=None,
    )
    assert decision.action == DedupAction.SKIP, (
        f"Identical content must still dedup via the heuristic; "
        f"got {decision.action}: {decision.reason}"
    )


# ── (b) Genuine cosine-space near-duplicate IS still deduped ──────────────

def test_cosine_space_near_duplicate_is_skipped():
    """A true cosine-space score > 0.92 is a real near-duplicate -> SKIP."""
    memory = _make_memory([
        {
            "id": 2,
            "content": "User lives in Austin, Texas",
            "score": 0.97,
            "score_space": "cosine",
            "directive": False,
        },
    ])
    decision = check_duplicate(
        "User resides in Austin TX",
        memory,
        config=None,
    )
    assert decision.action == DedupAction.SKIP, (
        f"Cosine-space 0.97 must SKIP as near-exact match; "
        f"got {decision.action}: {decision.reason}"
    )


def test_missing_score_space_defaults_to_cosine_back_compat():
    """Back-compat: results with no score_space tag (e.g. from a raw vector
    path that pre-dates the tag) are treated as cosine-space."""
    memory = _make_memory([
        {
            "id": 3,
            "content": "User lives in Austin, Texas",
            "score": 0.97,
            # no score_space key
            "directive": False,
        },
    ])
    decision = check_duplicate(
        "User resides in Austin TX",
        memory,
        config=None,
    )
    assert decision.action == DedupAction.SKIP


def test_cosine_space_low_similarity_added():
    """Cosine-space but low score -> distinct fact -> ADD."""
    memory = _make_memory([
        {
            "id": 4,
            "content": "User likes hiking",
            "score": 0.10,
            "score_space": "cosine",
            "directive": False,
        },
    ])
    decision = check_duplicate(
        "User drives a Tesla",
        memory,
        config=None,
    )
    assert decision.action == DedupAction.ADD


# ── (c) Novelty check in degraded mode uses word-overlap, not relative 1.0 ─

def test_novelty_check_word_overlap_helper():
    """The novelty fallback helper is scale-free word overlap."""
    from truememory.ingest.hooks.user_prompt_submit import _word_overlap

    # Identical -> 1.0
    assert _word_overlap("user likes dogs", "user likes dogs") == 1.0
    # One shared keyword out of many -> well below the 0.85 cutoff
    assert _word_overlap("user works at a startup", "user lives in austin") < 0.85
    # Empty -> 0.0
    assert _word_overlap("", "") == 0.0


def test_novelty_relative_one_keyword_overlap_not_dropped():
    """Simulate the novelty loop: a relative-space hit with score 1.0 but
    only one shared keyword must NOT be treated as a duplicate."""
    from truememory.ingest.hooks.user_prompt_submit import _word_overlap

    prompt = "User just adopted a golden retriever puppy named Biscuit"
    hit = {
        "content": "User lives in Austin",
        "score": 1.0,
        "score_space": "relative",
    }
    # Replicates the guarded novelty decision in _try_per_exchange_store.
    is_cosine = hit.get("score_space", "relative") == "cosine"
    dropped = (
        (is_cosine and hit["score"] > 0.85)
        or (not is_cosine and _word_overlap(prompt, hit["content"]) > 0.85)
    )
    assert dropped is False, "relative 1.0 with weak overlap must not drop the prompt"


def test_novelty_relative_true_duplicate_dropped():
    """A genuine duplicate in relative mode IS dropped via word overlap."""
    from truememory.ingest.hooks.user_prompt_submit import _word_overlap

    prompt = "User prefers dark mode in all apps"
    hit = {
        "content": "User prefers dark mode in all apps",
        "score": 1.0,
        "score_space": "relative",
    }
    is_cosine = hit.get("score_space", "relative") == "cosine"
    dropped = (
        (is_cosine and hit["score"] > 0.85)
        or (not is_cosine and _word_overlap(prompt, hit["content"]) > 0.85)
    )
    assert dropped is True


# ── client.py score_space tagging ─────────────────────────────────────────

def test_search_vectors_tags_cosine_when_raw_available():
    """When the raw vector path returns results, they are tagged cosine."""
    from truememory.client import Memory

    m = Memory.__new__(Memory)  # bypass __init__/DB
    m._engine = MagicMock()
    m._engine.search_vectors_raw.return_value = [
        {"id": 1, "content": "x", "score": 0.9},
    ]
    out = m.search_vectors("q", limit=3)
    assert out[0]["score_space"] == "cosine"


def test_search_vectors_falls_back_to_relative_when_no_vectors():
    """When raw vectors are unavailable, the search() fallback is tagged
    relative, never cosine."""
    from truememory.client import Memory

    m = Memory.__new__(Memory)
    m._engine = MagicMock()
    m._engine.search_vectors_raw.return_value = None
    m._engine.search.return_value = [
        {"id": 1, "content": "x", "score": 1.0, "sender": ""},
    ]
    out = m.search_vectors("q", limit=3)
    assert out[0]["score_space"] == "relative"


def test_search_tags_relative():
    """The full search() pipeline always yields relative-space scores."""
    from truememory.client import Memory

    m = Memory.__new__(Memory)
    m._engine = MagicMock()
    m._engine.search.return_value = [
        {"id": 1, "content": "x", "score": 1.0, "sender": ""},
    ]
    out = m.search("q", limit=3)
    assert out[0]["score_space"] == "relative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
