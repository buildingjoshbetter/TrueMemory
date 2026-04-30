"""Tests for the v044 embedding pair-difference PE scorer."""


class MockMemoryWithContent:
    """Returns search results with specific content."""

    def __init__(self, content: str, score: float = 0.5):
        self._content = content
        self._score = score

    def search(self, query, **kwargs):
        if self._content:
            return [{"content": self._content, "score": self._score}]
        return []

    def search_vectors(self, query, limit=5):
        return self.search(query)


class EmptyMemory:
    def search(self, query, **kwargs):
        return []

    def search_vectors(self, query, limit=5):
        return []


def test_noise_returns_zero():
    """Noise messages should always return PE=0 regardless of memory."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryWithContent("Alice lives in Seattle"))
    for noise in ["ok", "lol", "haha", "yeah", "cool", "thanks"]:
        decision = gate.evaluate(noise)
        assert decision.prediction_error == 0.0, (
            f"{noise!r} should have PE=0.0, got {decision.prediction_error}"
        )


def test_empty_memory_returns_zero():
    """With no memory, there's nothing to contradict — PE should be 0."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=EmptyMemory())
    decision = gate.evaluate("I moved to Portland", "personal")
    assert decision.prediction_error == 0.0, (
        f"PE with empty memory should be 0.0, got {decision.prediction_error}"
    )


def test_pe_in_valid_range():
    """PE should always be between 0 and 1."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryWithContent("Alice works at Google", score=0.8))
    messages = [
        "Alice works at Google",
        "Alice switched to Anthropic",
        "I moved to Portland",
        "The salary is $350k",
        "We broke up",
    ]
    for msg in messages:
        decision = gate.evaluate(msg)
        assert 0.0 <= decision.prediction_error <= 1.0, (
            f"{msg!r}: PE={decision.prediction_error} out of range"
        )


def test_contradicting_pair_higher_pe_than_consistent():
    """A message that contradicts memory should have higher PE than
    one that's consistent with it.

    This is the core v044 property: the (message, memory) pair embedding
    should diverge more from the (memory, memory) self-pair when the
    message says something different about the same topic.
    """
    from truememory.ingest.encoding_gate import EncodingGate

    memory_content = "Alice works at Google as a software engineer"

    gate_consistent = EncodingGate(
        memory=MockMemoryWithContent(memory_content, score=0.8)
    )
    decision_consistent = gate_consistent.evaluate(
        "Alice is a software engineer at Google"
    )

    gate_contradicting = EncodingGate(
        memory=MockMemoryWithContent(memory_content, score=0.8)
    )
    decision_contradicting = gate_contradicting.evaluate(
        "Alice quit Google and joined Anthropic as a researcher"
    )

    assert decision_contradicting.prediction_error >= decision_consistent.prediction_error, (
        f"Contradicting message should have PE >= consistent message. "
        f"Contradicting PE={decision_contradicting.prediction_error}, "
        f"Consistent PE={decision_consistent.prediction_error}"
    )


def test_unrelated_topic_low_pe():
    """A message about an unrelated topic should have low PE even if
    memory exists — unrelated topics have nothing to contradict."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(
        memory=MockMemoryWithContent("Alice works at Google", score=0.1)
    )
    decision = gate.evaluate("I love hiking in the mountains", "personal")
    assert decision.prediction_error < 0.3, (
        f"Unrelated topic should have low PE, got {decision.prediction_error}"
    )
