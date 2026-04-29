"""Test that signal weights can be configured via env vars."""

import os


class MockMemoryEmpty:
    def search(self, *a, **kw):
        return []


def test_env_var_weights_applied():
    """TRUEMEMORY_GATE_W_* env vars should override default weights."""
    os.environ["TRUEMEMORY_GATE_W_NOVELTY"] = "0.10"
    os.environ["TRUEMEMORY_GATE_W_SALIENCE"] = "0.60"
    os.environ["TRUEMEMORY_GATE_W_PE"] = "0.30"
    try:
        from truememory.ingest.encoding_gate import EncodingGate
        gate = EncodingGate(memory=MockMemoryEmpty())
        assert abs(gate.w_novelty - 0.10) < 1e-9, f"Expected w_novelty=0.10, got {gate.w_novelty}"
        assert abs(gate.w_salience - 0.60) < 1e-9, f"Expected w_salience=0.60, got {gate.w_salience}"
        assert abs(gate.w_prediction_error - 0.30) < 1e-9, f"Expected w_pe=0.30, got {gate.w_prediction_error}"
    finally:
        os.environ.pop("TRUEMEMORY_GATE_W_NOVELTY", None)
        os.environ.pop("TRUEMEMORY_GATE_W_SALIENCE", None)
        os.environ.pop("TRUEMEMORY_GATE_W_PE", None)


def test_explicit_args_override_env():
    """Explicit constructor args should take precedence over env vars."""
    os.environ["TRUEMEMORY_GATE_W_NOVELTY"] = "0.99"
    try:
        from truememory.ingest.encoding_gate import EncodingGate
        gate = EncodingGate(memory=MockMemoryEmpty(), w_novelty=0.10)
        assert abs(gate.w_novelty - 0.10) < 1e-9, (
            f"Explicit arg should override env var: expected 0.10, got {gate.w_novelty}"
        )
    finally:
        os.environ.pop("TRUEMEMORY_GATE_W_NOVELTY", None)
