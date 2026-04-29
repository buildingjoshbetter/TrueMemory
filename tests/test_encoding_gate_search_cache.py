"""Test that novelty search results are cached and reused by fallback PE."""


class CountingMemory:
    """Memory mock that counts search calls."""

    def __init__(self, score: float = 0.5, content: str = "existing fact"):
        self._score = score
        self._content = content
        self.search_count = 0

    def search(self, query, **kwargs):
        self.search_count += 1
        return [{"content": self._content, "score": self._score}]


def test_fallback_pe_reuses_novelty_search():
    """Fallback PE should reuse novelty search results, not make a second call.

    On main, _fallback_prediction_error() calls self._search() independently,
    wasting ~2-5ms per fact. The fix caches novelty results in
    _last_search_results and reuses them.
    """
    from truememory.ingest.encoding_gate import EncodingGate

    memory = CountingMemory(score=0.5)
    gate = EncodingGate(memory=memory)
    gate.evaluate("Some fact about Portland", "")

    # With score=0.5, novelty maps to ~0.50, which is in the mid-range
    # (not > 0.9 and not < 0.05), so PE will use the truememory path
    # or fallback. The similar_memory lookup also fires (0.1 < 0.5 < 0.7).
    #
    # Without caching: novelty search (1) + fallback PE search (1) +
    # similar_memory search (1) = 3 calls
    # With caching: novelty search (1) + PE reuses cache + similar reuses cache = 1 call
    assert memory.search_count == 1, (
        f"Expected 1 search call (novelty only), got {memory.search_count}. "
        f"Fallback PE and similar_memory should reuse cached novelty results."
    )
