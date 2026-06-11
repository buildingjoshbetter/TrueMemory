"""Regression lock: recall/dedup-only paths must skip the cross-encoder
reranker (PERF-03 / issue #690).

Pre-fix: the encoding-gate similar_memory lookup and the per-prompt novelty
dedup search omitted ``_skip_reranker``, so each paid the cross-encoder — a
full CrossEncoder model load in a cold hook subprocess — even though they only
read content/score, never ranked order.

No model loads — uses a stub Memory that records the _skip_reranker kwarg.
"""



class _RecordingMemory:
    """Records every search() call's kwargs; returns one bland result."""

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(kwargs)
        return [{"id": 1, "content": "a prior memory", "score": 0.5, "score_space": "cosine"}]

    # encoding_gate.similar_memory cache path may call search_vectors; provide it
    def search_vectors(self, query, limit=10):
        return [{"id": 1, "content": "a prior memory", "score": 0.5, "score_space": "cosine"}]


def test_encoding_gate_similar_memory_skips_reranker():
    from truememory.ingest.encoding_gate import EncodingGate
    mem = _RecordingMemory()
    gate = EncodingGate(memory=mem, user_id="u")
    # Drive the _search(fact, limit=1) similar-memory path directly.
    gate._search("some candidate fact", limit=1, skip_reranker=True)
    assert mem.calls, "search was not called"
    assert mem.calls[-1].get("_skip_reranker") is True, (
        "encoding-gate similar-memory search must pass _skip_reranker=True"
    )


def test_encoding_gate_source_passes_skip_reranker():
    """The _search call site for similar_memory (gate.py:316) passes skip_reranker."""
    import inspect
    from truememory.ingest import encoding_gate
    src = inspect.getsource(encoding_gate)
    # the limit=1 similar-memory lookup must request skip_reranker
    assert "self._search(fact, limit=1, skip_reranker=True)" in src, (
        "the similar_memory _search(limit=1) call must pass skip_reranker=True"
    )


def test_prompt_novelty_dedup_passes_skip_reranker():
    """The per-prompt novelty dedup search (user_prompt_submit) passes _skip_reranker."""
    import inspect
    from truememory.ingest.hooks import user_prompt_submit
    src = inspect.getsource(user_prompt_submit)
    assert "limit=3, _skip_reranker=True" in src, (
        "per-prompt novelty dedup search must pass _skip_reranker=True"
    )
