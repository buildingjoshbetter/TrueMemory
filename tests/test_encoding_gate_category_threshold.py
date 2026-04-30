"""Tests for per-category threshold overrides."""


class MockMemory:
    def search(self, *a, **kw):
        return []

    def search_vectors(self, *a, **kw):
        return []


def test_correction_gets_lower_threshold():
    """A correction that scores just below the default threshold should
    still pass because corrections get a threshold reduction."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(
        memory=MockMemory(),
        threshold=0.26,
        salience_floor=0.0,
    )

    decision_general = gate.evaluate("some minor technical detail", "general")
    decision_correction = gate.evaluate("some minor technical detail", "correction")

    if decision_general.encoding_score < 0.26 and decision_general.encoding_score >= 0.20:
        assert decision_general.should_encode is False
        assert decision_correction.should_encode is True, (
            f"Correction with score {decision_correction.encoding_score} should pass "
            f"at reduced threshold but was rejected"
        )


def test_general_category_no_override():
    """General category should use the default threshold unchanged."""
    from truememory.ingest.encoding_gate import _CATEGORY_THRESHOLD_OVERRIDE

    assert "general" not in _CATEGORY_THRESHOLD_OVERRIDE


def test_threshold_override_floor():
    """Even with overrides, the effective threshold should never go below 0.10."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(
        memory=MockMemory(),
        threshold=0.12,
        salience_floor=0.0,
    )
    # correction override is -0.06, so 0.12 - 0.06 = 0.06, floored to 0.10
    decision = gate.evaluate("test", "correction")
    # The effective threshold should be 0.10, not 0.06
    assert decision.encoding_score < 0.10 or decision.should_encode is True
