"""Tests for fact extraction."""

import json

from truememory.ingest.extractor import (
    _parse_extraction_response,
    _salvage_partial_json,
    extract_facts_simple,
)


def test_parse_clean_json():
    """Test parsing a clean JSON array response."""
    response = '''[
        {"content": "Lives in Seattle, Washington", "category": "personal", "confidence": "high"},
        {"content": "Prefers dark mode", "category": "preference", "confidence": "medium"}
    ]'''

    facts = _parse_extraction_response(response, max_facts=50)
    assert len(facts) == 2
    assert facts[0].content == "Lives in Seattle, Washington"
    assert facts[0].category == "personal"
    assert facts[1].content == "Prefers dark mode"


def test_parse_markdown_wrapped_json():
    """Test parsing JSON wrapped in markdown code fences."""
    response = '''```json
[
    {"content": "Uses TypeScript", "category": "technical", "confidence": "high"}
]
```'''

    facts = _parse_extraction_response(response, max_facts=50)
    assert len(facts) == 1
    assert facts[0].content == "Uses TypeScript"


def test_parse_json_with_preamble():
    """Test parsing JSON preceded by LLM preamble text."""
    response = '''Here are the extracted facts:

[
    {"content": "Name is Alice", "category": "personal", "confidence": "high"}
]

These are the key facts from the conversation.'''

    facts = _parse_extraction_response(response, max_facts=50)
    assert len(facts) == 1
    assert facts[0].content == "Name is Alice"


def test_salvage_partial_json():
    """Test salvaging facts from malformed JSON."""
    text = '''[
        {"content": "Fact one", "category": "personal"},
        BROKEN ENTRY
        {"content": "Fact two", "category": "preference"}
    ]'''

    facts = _salvage_partial_json(text)
    assert len(facts) == 2


def test_empty_response():
    """Test handling of empty extraction response."""
    facts = _parse_extraction_response("[]", max_facts=50)
    assert len(facts) == 0


def test_max_facts_limit():
    """Test that max_facts parameter is respected."""
    response = json.dumps([
        {"content": f"Fact {i}", "category": "personal"}
        for i in range(100)
    ])

    facts = _parse_extraction_response(response, max_facts=10)
    assert len(facts) == 10


def test_simple_extractor_personal():
    """Test heuristic extractor catches personal facts."""
    transcript = "User: I'm a software engineer working at Google.\nAssistant: Great!"

    facts = extract_facts_simple(transcript)
    assert len(facts) >= 1
    assert any("software engineer" in f.content.lower() for f in facts)


def test_simple_extractor_preference():
    """Test heuristic extractor catches preferences."""
    transcript = "User: I prefer Python over JavaScript for backend work."

    facts = extract_facts_simple(transcript)
    assert len(facts) >= 1
    assert any("prefer" in f.content.lower() for f in facts)


def test_simple_extractor_no_noise():
    """Test heuristic extractor doesn't extract noise."""
    transcript = "User: ok\nAssistant: sounds good\nUser: thanks"

    facts = extract_facts_simple(transcript)
    assert len(facts) == 0


