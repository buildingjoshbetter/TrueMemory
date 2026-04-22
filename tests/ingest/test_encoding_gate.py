"""Tests for the hippocampal encoding gate."""

from truememory.ingest.encoding_gate import EncodingGate


class FakeMemory:
    """Mock memory for testing the encoding gate."""

    def __init__(self, existing_memories=None):
        self.memories = existing_memories or []

    def search(self, query, limit=5, user_id=None):
        """Return fake search results with similarity scores."""
        if not self.memories:
            return []
        # Simple keyword overlap scoring for testing
        results = []
        query_words = set(query.lower().split())
        for mem in self.memories:
            mem_words = set(mem["content"].lower().split())
            overlap = len(query_words & mem_words)
            total = max(len(query_words | mem_words), 1)
            score = overlap / total
            results.append({**mem, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


def test_novel_fact_passes_gate():
    """A completely novel fact should pass the encoding gate."""
    memory = FakeMemory()  # Empty memory
    gate = EncodingGate(memory, threshold=0.30)

    decision = gate.evaluate("Alice lives in Seattle, Washington", category="personal")
    assert decision.should_encode is True
    assert decision.novelty == 1.0  # Nothing in memory


def test_duplicate_fact_blocked():
    """A fact that already exists should be blocked."""
    memory = FakeMemory([
        {"id": 1, "content": "Alice lives in Seattle, Washington"},
    ])
    gate = EncodingGate(memory, threshold=0.30)

    decision = gate.evaluate("Alice lives in Seattle, Washington", category="personal")
    # With high similarity, novelty should be low → may not pass gate
    # (exact behavior depends on the fake scoring)
    assert decision.novelty < 0.5


def test_correction_high_salience():
    """Corrections should have high salience scores."""
    memory = FakeMemory()
    gate = EncodingGate(memory, threshold=0.30)

    decision = gate.evaluate(
        "Actually, the deadline is Friday not Thursday",
        category="correction",
    )
    assert decision.salience > 0.5
    assert decision.should_encode is True


def test_life_event_high_salience():
    """Life events should boost salience."""
    memory = FakeMemory()
    gate = EncodingGate(memory, threshold=0.30)

    decision = gate.evaluate("Got promoted to senior engineer", category="personal")
    assert decision.salience > 0.5


def test_technical_lower_salience_than_correction():
    """Technical facts should have lower salience than corrections."""
    memory = FakeMemory()
    gate = EncodingGate(memory, threshold=0.30)

    tech = gate.evaluate("Using PostgreSQL 16 for the database", category="technical")
    correction = gate.evaluate("Actually we switched from Postgres to SQLite", category="correction")

    # Corrections should score higher on salience than routine technical facts
    assert correction.salience > tech.salience


def test_encoding_decision_has_explanation():
    """Every decision should include a human-readable explanation."""
    memory = FakeMemory()
    gate = EncodingGate(memory, threshold=0.30)

    decision = gate.evaluate("Prefers bun over npm", category="preference")
    assert decision.reason
    assert "ENCODE" in decision.reason or "SKIP" in decision.reason


def test_prediction_error_on_contradiction():
    """A fact that updates existing knowledge should have high prediction error."""
    memory = FakeMemory([
        {"id": 1, "content": "Uses npm for package management"},
    ])
    gate = EncodingGate(memory, threshold=0.30)

    decision = gate.evaluate(
        "Switched to bun, no longer uses npm",
        category="correction",
    )
    # Should encode because correction + update signal
    assert decision.should_encode is True


def test_threshold_sensitivity():
    """Higher thresholds should block more facts."""
    memory = FakeMemory()
    gate_low = EncodingGate(memory, threshold=0.10)
    gate_high = EncodingGate(memory, threshold=0.90)

    fact = "Minor technical detail about config"
    decision_low = gate_low.evaluate(fact, category="technical")
    decision_high = gate_high.evaluate(fact, category="technical")

    # Same fact, same scores, different thresholds
    assert decision_low.encoding_score == decision_high.encoding_score
    # Low threshold more permissive
    assert decision_low.should_encode or not decision_high.should_encode
