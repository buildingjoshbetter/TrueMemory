"""Test that non-numeric env vars for gate weights don't crash the pipeline (#303)."""

import os


class MockMemoryEmpty:
    def search(self, *a, **kw):
        return []


def test_non_numeric_env_var_falls_back_to_default():
    """Setting TRUEMEMORY_GATE_W_NOVELTY to a non-numeric string should
    fall back to the default rather than raising ValueError."""
    os.environ["TRUEMEMORY_GATE_W_NOVELTY"] = "high"
    os.environ["TRUEMEMORY_GATE_W_SALIENCE"] = "abc"
    os.environ["TRUEMEMORY_GATE_W_PE"] = ""
    os.environ["TRUEMEMORY_GATE_SALIENCE_FLOOR"] = "not_a_number"
    try:
        from truememory.ingest.encoding_gate import EncodingGate
        gate = EncodingGate(memory=MockMemoryEmpty())
        assert abs(gate.w_novelty - 0.25) < 1e-9, f"Expected default 0.25, got {gate.w_novelty}"
        assert abs(gate.w_salience - 0.20) < 1e-9, f"Expected default 0.20, got {gate.w_salience}"
        assert abs(gate.w_prediction_error - 0.30) < 1e-9, f"Expected default 0.30, got {gate.w_prediction_error}"
        assert abs(gate.salience_floor - 0.10) < 1e-9, f"Expected default 0.10, got {gate.salience_floor}"
    finally:
        os.environ.pop("TRUEMEMORY_GATE_W_NOVELTY", None)
        os.environ.pop("TRUEMEMORY_GATE_W_SALIENCE", None)
        os.environ.pop("TRUEMEMORY_GATE_W_PE", None)
        os.environ.pop("TRUEMEMORY_GATE_SALIENCE_FLOOR", None)


def test_valid_numeric_env_var_still_works():
    """Valid numeric env vars should still be applied."""
    os.environ["TRUEMEMORY_GATE_W_NOVELTY"] = "0.50"
    try:
        from truememory.ingest.encoding_gate import EncodingGate
        gate = EncodingGate(memory=MockMemoryEmpty())
        assert abs(gate.w_novelty - 0.50) < 1e-9, f"Expected 0.50, got {gate.w_novelty}"
    finally:
        os.environ.pop("TRUEMEMORY_GATE_W_NOVELTY", None)
