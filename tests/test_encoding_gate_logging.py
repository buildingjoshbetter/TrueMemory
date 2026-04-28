"""Test encoding gate batch logging and stats tracking."""


class MockMemoryEmpty:
    def search(self, *a, **kw):
        return []


def test_log_batch_summary_returns_stats():
    """log_batch_summary() should return evaluation stats."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryEmpty())
    gate.evaluate("I moved to Portland", "personal")
    gate.evaluate("ok", "")

    assert hasattr(gate, "log_batch_summary"), (
        "EncodingGate should have a log_batch_summary() method"
    )
    summary = gate.log_batch_summary()
    assert summary["evaluated"] == 2
    assert summary["passed"] + summary["blocked"] == 2


def test_batch_stats_cleared_on_reset():
    """reset_batch() should clear batch stats."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryEmpty())
    gate.evaluate("test fact", "")
    assert hasattr(gate, "_batch_scores"), (
        "EncodingGate should track _batch_scores"
    )
    assert len(gate._batch_scores) > 0
    gate.reset_batch()
    assert len(gate._batch_scores) == 0


def test_empty_batch_summary():
    """log_batch_summary() on empty batch should return evaluated=0."""
    from truememory.ingest.encoding_gate import EncodingGate

    gate = EncodingGate(memory=MockMemoryEmpty())
    summary = gate.log_batch_summary()
    assert summary["evaluated"] == 0
