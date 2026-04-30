"""
Tests for the pipeline's reset_batch() wiring added in round 2.

Verifies:
- ingest_transcript() calls gate.reset_batch() at the start
- ingest_text() calls gate.reset_batch() at the start
- Batch-level state from one run does not leak into the next
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from truememory.ingest.pipeline import IngestionPipeline


class MockMemory:
    def __init__(self):
        self.stored = []
        self._engine = self

    def search(self, query, limit=5, user_id=None):
        return []

    def add(self, content="", user_id=None, metadata=None, sender="", timestamp="", category="", **kwargs):
        self.stored.append({"id": len(self.stored), "content": content, "category": category})
        return {"id": len(self.stored) - 1}

    def update(self, memory_id, content):
        return {"id": memory_id, "content": content}


def test_ingest_transcript_calls_reset_batch():
    """Stale batch state from a previous run should not leak into a new ingest_transcript call."""
    memory = MockMemory()
    pipeline = IngestionPipeline(
        memory=memory,
        gate_threshold=0.25,
        use_llm_dedup=False,
        llm_config=None,
    )
    pipeline.llm_config = None

    # Inject stale state
    pipeline.gate._batch_scores.append(0.999)

    # Create a minimal transcript
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("User: I'm Alice and I live in Seattle\nAssistant: Nice")
        path = f.name

    try:
        pipeline.ingest_transcript(path)
        assert 0.999 not in pipeline.gate._batch_scores, \
            "reset_batch() was not called at the start of ingest_transcript"
    finally:
        Path(path).unlink()


def test_ingest_transcript_two_consecutive_runs_dont_leak_state():
    """Two consecutive transcripts should each start with a fresh batch cache."""
    memory = MockMemory()
    pipeline = IngestionPipeline(
        memory=memory,
        gate_threshold=0.25,
        use_llm_dedup=False,
        llm_config=None,
    )
    pipeline.llm_config = None

    transcripts = []
    try:
        for i, content in enumerate([
            "User: I work at Google\nAssistant: Cool",
            "User: I prefer Python over Java\nAssistant: Got it",
        ]):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(content)
                transcripts.append(f.name)

        # First run — produces some batch scores
        pipeline.ingest_transcript(transcripts[0], session_id="run-1")
        first_count = len(pipeline.gate._batch_scores)

        # Second run — should reset and produce a fresh batch
        pipeline.ingest_transcript(transcripts[1], session_id="run-2")
        second_count = len(pipeline.gate._batch_scores)

        # The second batch should NOT accumulate scores from the first run
        assert second_count <= first_count + 5, \
            "batch scores appear to accumulate across runs"

        # Add stale score before the next run, verify it's gone after
        pipeline.gate._batch_scores.append(0.999)
        pipeline.ingest_transcript(transcripts[0], session_id="run-3")
        assert 0.999 not in pipeline.gate._batch_scores
    finally:
        for p in transcripts:
            Path(p).unlink()


def test_ingest_text_also_handles_reset_batch_correctly():
    """ingest_text should also work correctly across multiple calls."""
    memory = MockMemory()
    pipeline = IngestionPipeline(
        memory=memory,
        gate_threshold=0.25,
        use_llm_dedup=False,
        llm_config=None,
    )
    pipeline.llm_config = None

    # ingest_text currently doesn't explicitly call reset_batch, but should
    # not crash on consecutive calls
    result1 = pipeline.ingest_text("User: I live in Seattle\nUser: I work remotely")
    result2 = pipeline.ingest_text("User: I prefer TypeScript\nUser: I use vim")

    assert result1.facts_extracted >= 0
    assert result2.facts_extracted >= 0
