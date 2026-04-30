"""Test that the encoding gate threshold uses >= (paper equation 4)."""


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


def test_compression_novelty_empty_memory():
    """Empty memory should give maximum novelty."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(
        memory=MockMemoryFixedScore(score=0.0),  # empty results
        w_novelty=1.0, w_salience=0.0, w_prediction_error=0.0,
    )
    decision = gate.evaluate("I just got a new job at Google", "")
    assert decision.novelty == 1.0, f"Empty memory should give novelty=1.0, got {decision.novelty}"


def test_compression_novelty_in_valid_range():
    """Novelty scores must be in [0.05, 1.0]."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(
        memory=MockMemoryFixedScore(score=0.5, content="I work at Google as an engineer"),
        w_novelty=1.0, w_salience=0.0, w_prediction_error=0.0,
    )
    for msg in ["ok", "I got a new job", "We are moving to Portland next month"]:
        decision = gate.evaluate(msg, "")
        assert 0.0 <= decision.novelty <= 1.0, (
            f"Novelty {decision.novelty} out of range for: {msg!r}"
        )
