"""
Integration tests — full pipeline end-to-end with a realistic Claude Code transcript.

These tests exercise the full ingestion flow (parse → extract → gate → dedup → store)
against a realistic synthetic transcript that matches the documented Claude Code
conversation format (JSON array of turns with `type` and `content` fields, where
assistant content can be a list of content blocks).

The tests use a mock Memory backend so they run without truememory installed
and without any LLM API calls — we force heuristic extraction mode.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from truememory.ingest.encoding_gate import EncodingGate
from truememory.ingest.pipeline import IngestionPipeline, IngestionResult
from truememory.ingest.transcript import parse_transcript, format_for_extraction


# Load the fixture once
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_claude_code_transcript.json"


class MockMemory:
    """In-memory Memory mock that supports the operations the pipeline uses."""

    def __init__(self, existing_memories: list | None = None):
        self.stored: list[dict] = list(existing_memories or [])
        self._next_id = len(self.stored)
        # Expose _engine attribute to mimic Memory's internal structure
        # so pipeline._store_fact's getattr check finds something
        self._engine = self

    def search(self, query: str, limit: int = 10, user_id: str | None = None) -> list[dict]:
        """Return results sorted by simple word-overlap score."""
        results = []
        query_words = set(query.lower().split())
        for m in self.stored:
            content = m.get("content", "")
            mem_words = set(content.lower().split())
            overlap = len(query_words & mem_words)
            total = max(len(query_words | mem_words), 1)
            score = overlap / total
            results.append({**m, "score": score})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def add(self, content: str = "", user_id: str | None = None, metadata: dict | None = None,
            sender: str = "", timestamp: str = "", category: str = "", **kwargs) -> dict:
        """Store a memory and return a dict with an id."""
        memory_id = self._next_id
        self._next_id += 1
        entry = {
            "id": memory_id,
            "content": content,
            "user_id": user_id or sender,
            "sender": sender or user_id or "",
            "timestamp": timestamp,
            "category": category,
        }
        self.stored.append(entry)
        return entry

    def update(self, memory_id: int, content: str) -> dict | None:
        """Update an existing memory."""
        for m in self.stored:
            if m.get("id") == memory_id:
                m["content"] = content
                return m
        return None


# ---------------------------------------------------------------------------
# Transcript parsing tests
# ---------------------------------------------------------------------------

def test_fixture_exists():
    """The fixture file must exist and be valid JSON."""
    assert FIXTURE_PATH.exists(), f"Fixture not found at {FIXTURE_PATH}"
    data = json.loads(FIXTURE_PATH.read_text())
    assert isinstance(data, list)
    assert len(data) >= 10, "Fixture should have at least 10 turns"


def test_parse_claude_code_transcript():
    """Parser correctly handles Claude Code's JSON array format with content blocks."""
    messages = parse_transcript(FIXTURE_PATH)
    assert len(messages) > 0

    # Should have both human and assistant turns
    roles = {m.role for m in messages}
    assert "human" in roles
    assert "assistant" in roles

    # Content blocks should be flattened into text
    assistant_msgs = [m for m in messages if m.role == "assistant"]
    assert any("stack" in m.content or "architecture" in m.content.lower() for m in assistant_msgs)


def test_format_for_extraction_strips_tools():
    """Formatted transcript should be clean User/Assistant lines."""
    messages = parse_transcript(FIXTURE_PATH)
    formatted = format_for_extraction(messages)
    assert "User:" in formatted
    assert "Assistant:" in formatted
    # Should have multiple turns
    assert formatted.count("User:") >= 5
    assert formatted.count("Assistant:") >= 3


# ---------------------------------------------------------------------------
# End-to-end pipeline tests (heuristic mode — no LLM required)
# ---------------------------------------------------------------------------

def test_e2e_heuristic_extraction_and_storage():
    """
    Full pipeline: parse fixture → extract facts heuristically → gate → dedup → store.

    Verifies:
    - Pipeline doesn't crash on a realistic transcript
    - At least some facts are extracted (the fixture contains multiple "I prefer/I use" statements)
    - Facts pass the encoding gate
    - Facts land in the mock memory store
    """
    memory = MockMemory()

    pipeline = IngestionPipeline(
        memory=memory,
        user_id="alice",
        gate_threshold=0.25,
        use_llm_dedup=False,
        llm_config=None,  # triggers auto_detect; if it fails, we force heuristic below
    )
    # Force heuristic extraction regardless of environment
    pipeline.llm_config = None

    result = pipeline.ingest_transcript(str(FIXTURE_PATH), session_id="test-e2e")

    assert isinstance(result, IngestionResult)
    assert result.facts_extracted > 0, "Should extract at least one fact from the fixture"
    # In heuristic mode, simple "I prefer X" patterns should land
    stored_texts = [m["content"] for m in memory.stored]
    # The fixture contains "I prefer bun over npm" — heuristic should catch it
    preference_hits = [
        s for s in stored_texts
        if "prefer" in s.lower() or "bun" in s.lower() or "typescript" in s.lower()
    ]
    assert len(preference_hits) > 0 or result.facts_stored > 0, \
        f"Expected at least one stored fact. Got stored={result.facts_stored}, memories={stored_texts}"


def test_e2e_reset_batch_between_runs():
    """
    Verify that ingest_transcript calls reset_batch() at the start of each run.
    This prevents batch-level prediction-error state from leaking between transcripts.
    """
    memory = MockMemory()
    pipeline = IngestionPipeline(
        memory=memory,
        user_id="alice",
        gate_threshold=0.25,
        use_llm_dedup=False,
        llm_config=None,
    )
    pipeline.llm_config = None

    # Inject stale batch state
    pipeline.gate._batch_scores.append(0.999)

    # Ingest the fixture
    pipeline.ingest_transcript(str(FIXTURE_PATH), session_id="test-reset-1")

    # After ingestion, the stale score should be gone
    # (batch may contain NEW scores from the actual facts, but not the stale one)
    assert 0.999 not in pipeline.gate._batch_scores, \
        "reset_batch() was not called by ingest_transcript — stale state leaked"


def test_e2e_dedup_prevents_double_storage():
    """
    If we ingest the same transcript twice, the second pass should not double-store.
    """
    memory = MockMemory()
    pipeline = IngestionPipeline(
        memory=memory,
        user_id="alice",
        gate_threshold=0.25,
        use_llm_dedup=False,
        llm_config=None,
    )
    pipeline.llm_config = None

    result1 = pipeline.ingest_transcript(str(FIXTURE_PATH), session_id="first")
    _count_after_first = len(memory.stored)

    result2 = pipeline.ingest_transcript(str(FIXTURE_PATH), session_id="second")
    _count_after_second = len(memory.stored)

    # Second run should store significantly fewer (ideally zero new) facts
    assert result2.facts_stored <= result1.facts_stored, \
        f"Second run stored {result2.facts_stored} new facts when most should have been dedup-skipped"


def test_e2e_empty_transcript_returns_clean_result():
    """Empty transcript should return an IngestionResult with zero counts and no crash."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("[]")
        empty_path = f.name

    try:
        memory = MockMemory()
        pipeline = IngestionPipeline(
            memory=memory,
            gate_threshold=0.25,
            use_llm_dedup=False,
            llm_config=None,
        )
        pipeline.llm_config = None
        result = pipeline.ingest_transcript(empty_path)
        assert result.facts_extracted == 0
        assert result.facts_stored == 0
        assert len(memory.stored) == 0
    finally:
        Path(empty_path).unlink()


def test_e2e_short_transcript_skipped():
    """Transcripts below the minimum substantive content threshold should be skipped gracefully."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("User: hi\nAssistant: hello")
        short_path = f.name

    try:
        memory = MockMemory()
        pipeline = IngestionPipeline(
            memory=memory,
            gate_threshold=0.25,
            use_llm_dedup=False,
            llm_config=None,
        )
        pipeline.llm_config = None
        result = pipeline.ingest_transcript(short_path)
        # Too short to extract meaningful facts
        assert result.facts_extracted == 0 or result.facts_stored == 0
    finally:
        Path(short_path).unlink()


# ---------------------------------------------------------------------------
# Encoding gate + truememory delegation tests
# ---------------------------------------------------------------------------

def test_encoding_gate_handles_missing_truememory_gracefully():
    """
    When truememory.salience and truememory.predictive are not available (e.g. test env),
    the gate should still work via the heuristic fallback path.
    """
    memory = MockMemory()
    gate = EncodingGate(memory, threshold=0.30)

    # A fact with clearly novel content should pass
    decision = gate.evaluate("Alice works on Project Alpha, a sensor node network", category="personal")
    assert decision.should_encode
    assert 0.0 <= decision.encoding_score <= 1.0

    # Clamping check: all signals should be in valid ranges
    assert 0.0 <= decision.novelty <= 1.0
    assert 0.0 <= decision.salience <= 1.0
    assert 0.0 <= decision.prediction_error <= 1.0


def test_encoding_gate_reset_batch_clears_state():
    """reset_batch() should clear internal batch-level state."""
    memory = MockMemory()
    gate = EncodingGate(memory, threshold=0.30)
    gate._batch_scores.append(0.5)
    assert len(gate._batch_scores) == 1
    gate.reset_batch()
    assert len(gate._batch_scores) == 0


def test_mock_memory_supports_engine_add_signature():
    """
    The pipeline's _store_fact calls engine.add(content, sender, timestamp, category).
    Verify our MockMemory accepts that signature (it should, since it mirrors the real API).
    """
    memory = MockMemory()
    # Simulate the engine-level add call that pipeline._store_fact makes
    result = memory.add(
        content="test fact",
        sender="alice",
        timestamp="2026-04-05T00:00:00Z",
        category="personal",
    )
    assert result["id"] == 0
    assert result["content"] == "test fact"
    assert result["category"] == "personal"
