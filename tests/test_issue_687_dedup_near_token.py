"""Regression lock: dedup must not silently drop factually-distinct memories
that differ only by a number at high cosine similarity (C2-DEDUP / issue #687).

Pre-fix: a >0.92 cosine match (non-correction) returned DedupAction.SKIP BEFORE
the LLM-arbitration path, so on Base/Pro "Project deadline is October 15th" vs
"...16th" (cosine ~0.99) silently dropped the second fact. Post-fix: a >0.92
match routes to the LLM when a config is present; without one, a digit-run
divergence keeps both facts (ADD) and only true paraphrases fast-SKIP.

No model loads — drives check_duplicate with a stubbed search function.
"""


from truememory.ingest.dedup import (
    check_duplicate,
    DedupAction,
    _has_divergent_numbers,
)


class _FakeMemory:
    """Stub Memory whose search_vectors surfaces one candidate at a fixed cosine."""

    def __init__(self, top_content, score=0.99):
        self._top_content = top_content
        self._score = score

    def search_vectors(self, _query, limit=3):
        return [{"id": 7, "content": self._top_content, "score": self._score, "score_space": "cosine"}]


# ── the helper ───────────────────────────────────────────────────────────────

def test_divergent_numbers_helper():
    assert _has_divergent_numbers("deadline October 15th", "deadline October 16th") is True
    assert _has_divergent_numbers("paid $116", "paid $117") is True
    assert _has_divergent_numbers("version 1.2.0", "version 1.3.0") is True
    # same numbers / pure paraphrase -> not divergent
    assert _has_divergent_numbers("deadline is October 15th", "the deadline: October 15th") is False
    assert _has_divergent_numbers("likes coffee", "enjoys coffee") is False


# ── no-LLM (heuristic) path: divergent numbers must be kept ──────────────────

def test_near_token_numeric_fact_not_dropped_without_llm():
    decision = check_duplicate(
        "Project deadline is October 16th",
        memory=_FakeMemory("Project deadline is October 15th", score=0.99),
        config=None,
    )
    assert decision.action == DedupAction.ADD, (
        f"distinct-date fact was dropped: {decision.action} ({decision.reason})"
    )


def test_true_paraphrase_still_skipped_without_llm():
    """A genuine paraphrase (no numeric divergence) still fast-SKIPs."""
    decision = check_duplicate(
        "the user really likes coffee",
        memory=_FakeMemory("user likes coffee", score=0.99),
        config=None,
    )
    assert decision.action == DedupAction.SKIP


# ── LLM path: >0.92 routes to arbitration instead of silent SKIP ─────────────

def test_high_similarity_routes_to_llm_when_config_present(monkeypatch):
    """With an LLM config, a >0.92 match is arbitrated, not auto-SKIPped."""
    import truememory.ingest.dedup as dedup_mod

    called = {"n": 0}

    def _fake_llm(fact, existing, existing_id, config):
        called["n"] += 1
        return dedup_mod.DedupDecision(action=DedupAction.ADD, fact=fact, reason="llm: distinct")

    monkeypatch.setattr(dedup_mod, "_llm_dedup", _fake_llm)

    class _Cfg:  # any truthy LLMConfig-ish object
        pass

    decision = check_duplicate(
        "Project deadline is October 16th",
        memory=_FakeMemory("Project deadline is October 15th", score=0.99),
        config=_Cfg(),
    )
    assert called["n"] == 1, "the >0.92 match must reach LLM arbitration"
    assert decision.action == DedupAction.ADD
