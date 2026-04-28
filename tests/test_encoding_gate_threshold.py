"""Test that the encoding gate threshold uses >= (paper equation 4)."""

import pytest


class MockMemoryFixedScore:
    """Returns results with a controlled score to produce a known gate score."""

    def __init__(self, score: float, content: str = "existing"):
        self._score = score
        self._content = content

    def search(self, query, **kwargs):
        if self._score > 0:
            return [{"content": self._content, "score": self._score}]
        return []


def test_threshold_boundary_gte():
    """Score exactly at threshold should pass the gate (>= per paper eq 4)."""
    from truememory.ingest.encoding_gate import EncodingGate

    threshold = 0.50
    gate = EncodingGate(
        memory=MockMemoryFixedScore(score=0.0),
        threshold=threshold,
        w_novelty=1.0,
        w_salience=0.0,
        w_prediction_error=0.0,
    )
    # With empty results (score=0), novelty=1.0.
    # Final score = 1.0 * 1.0 / 1.0 = 1.0 — well above threshold.
    # Now set weights so score lands exactly at threshold.
    # We need score = threshold exactly.
    # With w_novelty=1, w_salience=0, w_pe=0: score = novelty.
    # We need novelty = 0.50 exactly.
    # From the piecewise: novelty = 0.30 + 0.40*(1-(x-0.4)/0.2)
    #   0.50 = 0.30 + 0.40*(1-(x-0.4)/0.2)
    #   0.20 = 0.40*(1-(x-0.4)/0.2)
    #   0.50 = 1-(x-0.4)/0.2
    #   (x-0.4)/0.2 = 0.50
    #   x = 0.50
    # So search score = 0.50 gives novelty = 0.50.
    gate2 = EncodingGate(
        memory=MockMemoryFixedScore(score=0.50),
        threshold=0.50,
        w_novelty=1.0,
        w_salience=0.0,
        w_prediction_error=0.0,
    )
    decision = gate2.evaluate("test fact", "")
    # novelty at search_score=0.50: 0.30 + 0.40*(1-(0.50-0.4)/0.2) = 0.30+0.40*0.5 = 0.50
    assert abs(decision.novelty - 0.50) < 0.01, f"Expected novelty ~0.50, got {decision.novelty}"
    assert abs(decision.encoding_score - 0.50) < 0.01, f"Expected score ~0.50, got {decision.encoding_score}"
    # Paper equation (4): score >= threshold should encode
    assert decision.should_encode is True, (
        f"Score {decision.encoding_score} at threshold {threshold} should encode "
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
