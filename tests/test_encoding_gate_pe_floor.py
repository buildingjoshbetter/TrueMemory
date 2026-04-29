"""Test that the PE signal can reach zero for pure noise (no +0.1 floor)."""


class MockMemoryWithScore:
    """Returns results with a specific score."""
    def __init__(self, score: float, content: str = "existing"):
        self._score = score
        self._content = content
    def search(self, query, **kwargs):
        if self._score > 0:
            return [{"content": self._content, "score": self._score}]
        return []


def test_pe_no_floor_for_noise():
    """PE should be able to reach values below 0.10 for low-surprise messages.

    The old formula `surprise * 0.8 + 0.1` guaranteed PE >= 0.10 for all
    messages, adding 0.025 to every noise message's final score. The fix
    removes the +0.1 floor so pure noise can get PE closer to zero.
    """
    from truememory.ingest.encoding_gate import EncodingGate

    # Use a moderate novelty (0.1 < novelty < 0.9) to avoid the early returns.
    # Search score 0.50 -> novelty ~0.50 (in the mid-range).
    gate = EncodingGate(
        memory=MockMemoryWithScore(score=0.50, content="existing fact"),
        threshold=0.30,
    )
    decision = gate.evaluate("ok", "")

    # With the old formula (surprise * 0.8 + 0.1), minimum PE = 0.10.
    # With the new formula (surprise * 0.9), PE for noise ("ok") should be:
    # compute_surprise_score("ok", set()) returns 0.05 (noise match).
    # New PE: 0.05 * 0.9 = 0.045.
    # The PE should be below 0.10 (the old floor).
    assert decision.prediction_error < 0.10, (
        f"PE for noise message should be below 0.10 (old floor), "
        f"got {decision.prediction_error}"
    )
