"""Test compression-based novelty scoring (issue #107).

Validates that the encoding gate's novelty signal uses gzip compression
cost against stored memories instead of cosine similarity inversion.
"""


class MockMemoryWithContent:
    """Returns search results with specific content for compression comparison."""

    def __init__(self, memories: list[str]):
        self._memories = memories

    def search(self, query, **kwargs):
        return [{"content": m, "score": 0.5} for m in self._memories]

    def search_vectors(self, query, limit=5):
        return self.search(query, limit=limit)


class MockMemoryEmpty:
    """Returns no search results (empty memory)."""

    def search(self, query, **kwargs):
        return []

    def search_vectors(self, query, limit=5):
        return []


def test_empty_memory_returns_max_novelty():
    """With no stored memories, everything is maximally novel."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryEmpty())
    decision = gate.evaluate("I just got a new job at Google", "personal")
    assert decision.novelty == 1.0


def test_redundant_message_scores_low_novelty():
    """A message that's already in memory should score low novelty."""
    from truememory.ingest.encoding_gate import EncodingGate

    memories = ["I work at Google as a software engineer"]
    gate = EncodingGate(memory=MockMemoryWithContent(memories))

    decision = gate.evaluate("I work at Google as a software engineer", "personal")
    assert decision.novelty < 0.5, (
        f"Redundant message should score low novelty, got {decision.novelty:.3f}"
    )


def test_novel_message_scores_high_novelty():
    """A message about a new topic should score high novelty."""
    from truememory.ingest.encoding_gate import EncodingGate

    memories = ["I work at Google as a software engineer"]
    gate = EncodingGate(memory=MockMemoryWithContent(memories))

    decision = gate.evaluate("We just adopted a golden retriever puppy named Scout", "personal")
    assert decision.novelty > 0.5, (
        f"Novel message should score high novelty, got {decision.novelty:.3f}"
    )


def test_noise_scores_low_novelty():
    """Short noise like 'ok' should score low — gzip compresses it trivially."""
    from truememory.ingest.encoding_gate import EncodingGate

    memories = ["I work at Google", "We moved to Portland last month"]
    gate = EncodingGate(memory=MockMemoryWithContent(memories))

    decision = gate.evaluate("ok", "")
    assert decision.novelty < 0.3, (
        f"Noise 'ok' should score low novelty (< 0.3), got {decision.novelty:.3f}"
    )


def test_novelty_uses_compression_not_cosine():
    """The novelty signal should use gzip compression, not cosine similarity.

    Regression test: cosine similarity scored 'ok' as HIGH novelty (0.95+)
    because it's semantically distant from everything. Compression scores
    it LOW because it's trivially short and adds no information.
    """
    from truememory.ingest.encoding_gate import EncodingGate

    memories = ["I work at Google", "We moved to Portland", "The salary is $150,000"]
    gate = EncodingGate(memory=MockMemoryWithContent(memories))

    noise_decision = gate.evaluate("ok", "")
    signal_decision = gate.evaluate("I just got engaged to Riley last weekend", "personal")

    assert signal_decision.novelty > noise_decision.novelty, (
        f"Signal ({signal_decision.novelty:.3f}) should score higher novelty "
        f"than noise ({noise_decision.novelty:.3f}). If noise scores higher, "
        f"the scorer is using cosine distance, not compression."
    )


def test_novelty_score_in_range():
    """Novelty scores must be in [0.05, 1.0]."""
    from truememory.ingest.encoding_gate import EncodingGate

    memories = ["Some existing memory about work and life"]
    gate = EncodingGate(memory=MockMemoryWithContent(memories))

    for msg in ["ok", "I got a new job", "A" * 500, "", "🎉🎉🎉"]:
        decision = gate.evaluate(msg, "")
        assert 0.0 <= decision.novelty <= 1.0, (
            f"Novelty {decision.novelty} out of range for: {msg!r}"
        )
