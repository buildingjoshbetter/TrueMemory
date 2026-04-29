"""Test that prediction error is decoupled from novelty (issue #110)."""


class MockMemoryWithScore:
    """Returns results with a specific score."""
    def __init__(self, score: float, content: str = "existing"):
        self._score = score
        self._content = content
    def search(self, query, **kwargs):
        if self._score > 0:
            return [{"content": self._content, "score": self._score}]
        return []
    def search_vectors(self, query, limit=5):
        return self.search(query)


def test_pe_not_clamped_at_high_novelty():
    """PE should NOT return a fixed 0.30 when novelty is high.

    Before this fix, lines 353-356 clamped PE to 0.30 whenever
    novelty > 0.9. Now PE computes its own value independently.
    """
    from truememory.ingest.encoding_gate import EncodingGate

    # Empty memory = novelty will be 1.0 (max)
    gate = EncodingGate(memory=MockMemoryWithScore(score=0.0))

    # Evaluate two very different messages — their PE should differ
    decision_noise = gate.evaluate("ok", "")
    decision_signal = gate.evaluate("I just got promoted to VP and my salary is now $350,000", "personal")

    # Before the fix, both would get PE=0.30 (clamped).
    # After the fix, they should differ because the surprise scorer
    # sees different content.
    # We don't assert exact values (they depend on the scorer),
    # but PE should no longer be identical for all high-novelty messages.
    assert decision_noise.prediction_error != 0.30 or decision_signal.prediction_error != 0.30, (
        f"PE should not be clamped to 0.30 for high-novelty messages. "
        f"noise PE={decision_noise.prediction_error}, signal PE={decision_signal.prediction_error}"
    )


def test_pe_not_forced_zero_at_low_novelty():
    """PE should NOT be forced to 0.0 when novelty is very low.

    Before this fix, lines 358-360 forced PE to 0.0 whenever
    novelty < 0.05. A near-duplicate that CONTRADICTS existing
    memory should still get high PE.
    """
    from truememory.ingest.encoding_gate import EncodingGate

    # High similarity = low novelty (~0.05)
    gate = EncodingGate(memory=MockMemoryWithScore(score=0.95, content="I work at Stripe"))

    # This message contradicts existing memory — should have nonzero PE
    # even though novelty is very low (near-duplicate text similarity)
    decision = gate.evaluate("I switched from Stripe to Anthropic", "personal")

    # The _looks_like_update heuristic should catch "switched from...to"
    # and return high PE. Before the fix, PE was forced to 0.0 for
    # low-novelty messages, so this contradiction would be invisible.
    # We just check PE is not forced to exactly 0.0
    assert decision.prediction_error > 0.0, (
        f"PE should not be forced to 0.0 for near-duplicates that contain updates. "
        f"Got PE={decision.prediction_error}"
    )
