"""Tests for issue #585: encoding gate PE degradation and contradiction bypass."""

import unittest
from unittest.mock import patch, MagicMock


class MockMemory:
    """Minimal mock memory that returns search results."""

    def __init__(self, results=None):
        self._results = results or []

    def search(self, query, **kwargs):
        return self._results

    def search_vectors(self, query, limit=5):
        return self._results


class MockMemoryWithContent(MockMemory):
    def __init__(self, content="User lives in Seattle", score=0.5):
        super().__init__([{"content": content, "score": score}])


class EmptyMemory(MockMemory):
    def __init__(self):
        super().__init__([])


# ---------------------------------------------------------------------------
# PE unavailable -> gate degrades OPEN (all facts pass)
# ---------------------------------------------------------------------------

def test_pe_unavailable_all_pass():
    """When PE model fails to load, all memories should pass (open gate)."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryWithContent())

    # Simulate PE model failure
    with patch(
        "truememory.ingest.encoding_gate.EncodingGate._compute_prediction_error",
        side_effect=_fail_pe_and_mark(gate),
    ):
        pass

    # Directly mark PE as unavailable (simulating model load failure)
    gate._pe_available = False
    gate._pe_degradation_count = 1

    decision = gate.evaluate("Some random fact with low novelty")
    assert decision.should_encode is True, (
        f"PE degraded gate should pass everything, got should_encode={decision.should_encode}"
    )
    assert "pe-degraded" in decision.reason


def test_pe_model_load_failure_sets_flag():
    """When get_model() raises, _pe_available should become False."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryWithContent())
    # Ensure _last_search_results is populated so PE tries to load model
    gate._last_search_results = [{"content": "test memory", "score": 0.5}]

    with patch(
        "truememory.vector_search.get_model",
        side_effect=RuntimeError("model not found"),
    ):
        pe = gate._compute_prediction_error("User moved to Portland")

    assert pe == 0.0
    assert gate._pe_available is False
    assert gate._pe_degradation_count >= 1


def test_pe_unavailable_full_evaluate_passes():
    """Full evaluate() should pass facts when PE model is dead."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryWithContent())

    with patch(
        "truememory.vector_search.get_model",
        side_effect=RuntimeError("model not found"),
    ):
        # First evaluate triggers model load failure
        decision = gate.evaluate("User likes Python over JavaScript", "preference")

    # Gate should have degraded open
    assert gate._pe_available is False

    # Second evaluate should also pass (gate stays open)
    decision2 = gate.evaluate("User prefers dark mode", "preference")
    assert decision2.should_encode is True


# ---------------------------------------------------------------------------
# PE available + low score -> memory blocked (existing behavior preserved)
# ---------------------------------------------------------------------------

def test_pe_available_low_score_blocked():
    """With PE available, a low-scoring fact should still be blocked."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryWithContent(), threshold=0.90)
    # PE is available (default), high threshold ensures blocking
    decision = gate.evaluate("ok", "general")
    # "ok" is noise — should be blocked by salience floor or threshold
    assert decision.should_encode is False


# ---------------------------------------------------------------------------
# Contradictions always pass regardless of PE score
# ---------------------------------------------------------------------------

def test_contradiction_markers_always_pass():
    """Facts with contradiction markers should always pass the gate."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=EmptyMemory(), threshold=0.99)

    contradiction_facts = [
        "Actually, I moved to Portland not Seattle",
        "Correction: my name is spelled differently",
        "I no longer work at Google",
        "User switched to vim from emacs",
        "It's not Python but JavaScript that I prefer",
    ]

    for fact in contradiction_facts:
        decision = gate.evaluate(fact, "personal")
        assert decision.should_encode is True, (
            f"Contradiction should pass: {fact!r}, got should_encode={decision.should_encode}, "
            f"reason={decision.reason}"
        )
        assert "contradiction-bypass" in decision.reason


def test_correction_category_always_passes():
    """Facts with category='correction' should always pass the gate."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=EmptyMemory(), threshold=0.99)
    decision = gate.evaluate("User prefers tabs over spaces", "correction")
    assert decision.should_encode is True
    assert "contradiction-bypass" in decision.reason


def test_non_contradiction_can_be_blocked():
    """Non-contradiction facts should still be subject to normal gating."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryWithContent(), threshold=0.99)
    decision = gate.evaluate("The weather is nice today", "general")
    # With threshold=0.99, normal facts should be blocked
    assert decision.should_encode is False


# ---------------------------------------------------------------------------
# PE degradation counter
# ---------------------------------------------------------------------------

def test_pe_degradation_counter_increments():
    """Each PE failure should increment the degradation counter."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryWithContent())
    gate._last_search_results = [{"content": "test", "score": 0.5}]

    with patch(
        "truememory.vector_search.get_model",
        side_effect=RuntimeError("boom"),
    ):
        gate._compute_prediction_error("fact one")
        assert gate._pe_degradation_count == 1

        # Reset model to None so it tries again
        gate._embed_model = None
        gate._compute_prediction_error("fact two")
        assert gate._pe_degradation_count == 2


def test_pe_degradation_in_batch_summary():
    """log_batch_summary should include PE degradation info when present."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryWithContent())
    gate._pe_available = False
    gate._pe_degradation_count = 3

    # Run an evaluate so batch has data
    gate.evaluate("Some fact", "personal")

    summary = gate.log_batch_summary()
    assert summary["pe_degradation_count"] == 3
    assert summary["pe_available"] is False


def test_pe_degradation_absent_when_healthy():
    """log_batch_summary should NOT include PE degradation when PE is fine."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=EmptyMemory())
    gate.evaluate("Some fact", "personal")

    summary = gate.log_batch_summary()
    assert "pe_degradation_count" not in summary


# ---------------------------------------------------------------------------
# _is_contradiction unit tests
# ---------------------------------------------------------------------------

def test_is_contradiction_function():
    """Direct tests for _is_contradiction helper."""
    from truememory.ingest.encoding_gate import _is_contradiction

    # Should be True
    assert _is_contradiction("Actually, I live in Austin", "personal") is True
    assert _is_contradiction("Correction: wrong email", "general") is True
    assert _is_contradiction("I no longer use npm", "preference") is True
    assert _is_contradiction("It's not X but Y", "personal") is True
    assert _is_contradiction("any text", "correction") is True
    assert _is_contradiction("She switched to Android", "personal") is True

    # Should be False
    assert _is_contradiction("I like pizza", "personal") is False
    assert _is_contradiction("The meeting is at 3pm", "event") is False
    assert _is_contradiction("ok", "general") is False


# Helper for mocking
def _fail_pe_and_mark(gate):
    """Side effect that marks PE as failed."""
    def _inner(*args, **kwargs):
        gate._pe_available = False
        gate._pe_degradation_count += 1
        return 0.0
    return _inner
