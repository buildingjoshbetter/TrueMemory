"""Test that the encoding gate threshold uses >= (paper equation 4)."""

import pytest


class MockMemoryFixedScore:
    """Returns results with a controlled score to produce a known gate score.

    Provides both search() (hybrid fallback) and search_vectors() (cosine
    path) so the gate tests exercise the preferred cosine code path.
    """

    def __init__(self, score: float, content: str = "existing"):
        self._score = score
        self._content = content

    def search(self, query, **kwargs):
        if self._score > 0:
            return [{"content": self._content, "score": self._score}]
        return []

    def search_vectors(self, query, limit=5):
        """Return same results as search — gate prefers this method."""
        return self.search(query)


def test_threshold_boundary_gte():
    """Score exactly at threshold should pass the gate (>= per paper eq 4)."""
    from truememory.ingest.encoding_gate import EncodingGate

    # Use novelty-only weighting with empty memory (novelty=1.0)
    # and set threshold to match the expected score
    gate = EncodingGate(
        memory=MockMemoryFixedScore(score=0.0),  # empty results → novelty=1.0
        threshold=1.0,  # set threshold exactly at novelty=1.0
        w_novelty=1.0,
        w_salience=0.0,
        w_prediction_error=0.0,
    )
    decision = gate.evaluate("test fact", "")
    assert abs(decision.novelty - 1.0) < 0.01, f"Expected novelty ~1.0, got {decision.novelty}"
    # Paper equation (4): score >= threshold should encode (score=1.0 >= threshold=1.0)
    assert decision.should_encode is True, (
        f"Score {decision.encoding_score} at threshold {gate.threshold} should encode "
        f"(paper equation 4 uses >=, not >)"
    )


def test_docstring_mentions_gte():
    """Module docstring should say >= not > for the threshold."""
    import truememory.ingest.encoding_gate as mod
    docstring = mod.__doc__ or ""
    # The docstring should reflect the paper's >= comparison
    assert ">=" in docstring or "≥" in docstring or "> 0.30" not in docstring, (
        "Module docstring should use >= (matching paper equation 4), not >"
    )


@pytest.mark.parametrize("search_score,expected_novelty", [
    (0.0, 1.0),     # no match → fully novel
    (0.05, 0.95),   # 1.0 - 0.05
    (0.10, 0.90),   # 1.0 - 0.10
    (0.25, 0.75),   # 1.0 - 0.25
    (0.50, 0.50),   # 1.0 - 0.50
    (0.95, 0.05),   # 1.0 - 0.95 = 0.05 (floor)
    (1.0, 0.05),    # floor at 0.05
])
def test_linear_novelty_mapping(search_score, expected_novelty):
    """Verify the novelty = 1 - similarity inversion (paper eq 1)."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(
        memory=MockMemoryFixedScore(score=search_score),
        w_novelty=1.0,
        w_salience=0.0,
        w_prediction_error=0.0,
    )
    decision = gate.evaluate("test fact", "")
    assert abs(decision.novelty - expected_novelty) < 0.01, (
        f"At search_score={search_score}, expected novelty ~{expected_novelty}, "
        f"got {decision.novelty:.4f}"
    )


def test_novelty_monotonically_decreasing():
    """Higher search similarity should produce lower novelty."""
    from truememory.ingest.encoding_gate import EncodingGate

    prev_novelty = 2.0
    for score in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        gate = EncodingGate(
            memory=MockMemoryFixedScore(score=score),
            w_novelty=1.0, w_salience=0.0, w_prediction_error=0.0,
        )
        decision = gate.evaluate("test", "")
        assert decision.novelty <= prev_novelty, (
            f"Novelty should decrease as similarity increases: "
            f"score={score} gave novelty={decision.novelty}, "
            f"but previous (lower score) gave {prev_novelty}"
        )
        prev_novelty = decision.novelty
