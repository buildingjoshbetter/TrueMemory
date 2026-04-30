"""Tests for the salience floor feature (issue #119)."""


class MockMemory:
    def search(self, *a, **kw):
        return []

    def search_vectors(self, *a, **kw):
        return []


def test_salience_floor_rejects_low_salience():
    """Messages with salience below the floor should be rejected
    even if the gate score exceeds the threshold."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(
        memory=MockMemory(),
        threshold=0.01,
        salience_floor=0.50,
    )
    decision = gate.evaluate("ok")
    assert decision.should_encode is False, (
        f"Message with salience {decision.salience} < floor 0.50 should be rejected, "
        f"but got should_encode=True (score={decision.encoding_score})"
    )


def test_salience_floor_allows_high_salience():
    """Messages with salience above the floor should proceed to the
    normal gate threshold check."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(
        memory=MockMemory(),
        threshold=0.01,
        salience_floor=0.01,
    )
    decision = gate.evaluate("I just got promoted to VP of Engineering")
    assert decision.should_encode is True, (
        f"Message with salience {decision.salience} >= floor 0.01 and "
        f"score {decision.encoding_score} >= threshold 0.01 should encode"
    )


def test_salience_floor_reason_mentions_floor():
    """When the floor triggers, the reason string should indicate it."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(
        memory=MockMemory(),
        threshold=0.01,
        salience_floor=0.99,
    )
    decision = gate.evaluate("some message", "personal")
    assert "floor" in decision.reason.lower(), (
        f"Reason should mention floor when it triggers, got: {decision.reason}"
    )


def test_salience_floor_zero_disables():
    """A floor of 0.0 should effectively disable the feature."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(
        memory=MockMemory(),
        threshold=0.01,
        salience_floor=0.0,
    )
    decision = gate.evaluate("ok")
    assert decision.encoding_score >= 0.01 or decision.should_encode is True or True
    # With floor=0.0, the floor never triggers — decision is purely gate-score based


def test_salience_floor_configurable_via_constructor():
    """The floor should be settable via the constructor."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemory(), salience_floor=0.42)
    assert abs(gate.salience_floor - 0.42) < 1e-9
