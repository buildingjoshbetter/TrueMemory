"""Tests for issue #586 — heuristic extractor fixes.

Covers:
1. Role-scoping: only user lines are extracted (assistant text ignored)
2. Case-insensitive matching
3. Smart/curly quote normalisation
4. New categories: correction, temporal, technical
5. Category validation: invalid categories remapped to "general"
"""

from truememory.ingest.extractor import (
    ALLOWED_CATEGORIES,
    _extract_user_lines,
    _normalize_quotes,
    _parse_extraction_response,
    _validate_category,
    extract_facts_simple,
)


# ── 1. Role-scoping ─────────────────────────────────────────────────────


def test_only_user_lines_extracted():
    """Assistant text that matches patterns must NOT produce facts."""
    transcript = (
        "User: I prefer dark mode for my editor.\n\n"
        "Assistant: I'm a large language model. I prefer to help you.\n\n"
        "User: Thanks!"
    )
    facts = extract_facts_simple(transcript)
    # The user's "I prefer dark mode" should match, but the assistant's
    # "I'm a large language model" and "I prefer to help you" should not.
    assert any("dark mode" in f.content.lower() for f in facts)
    assert not any("large language model" in f.content.lower() for f in facts)
    assert not any("help you" in f.content.lower() for f in facts)


def test_assistant_personal_statement_ignored():
    """'I am' in assistant text must not be extracted as a personal fact."""
    transcript = (
        "User: What are you?\n\n"
        "Assistant: I am an AI assistant made by Anthropic.\n\n"
        "User: I am a software engineer."
    )
    facts = extract_facts_simple(transcript)
    assert any("software engineer" in f.content.lower() for f in facts)
    assert not any("AI assistant" in f.content for f in facts)


def test_extract_user_lines_helper():
    """_extract_user_lines keeps only User: blocks."""
    transcript = (
        "User: Hello, I'm Josh.\n\n"
        "Assistant: Nice to meet you!\n\n"
        "User: I live in Austin."
    )
    result = _extract_user_lines(transcript)
    assert "Josh" in result
    assert "Austin" in result
    assert "Nice to meet you" not in result


# ── 2. Case-insensitive matching ────────────────────────────────────────


def test_case_insensitive_personal():
    """Mixed-case 'I AM', 'i am', 'I Am' should all match."""
    for phrase in ["I AM a doctor", "i am a doctor", "I Am A Doctor"]:
        transcript = f"User: {phrase}."
        facts = extract_facts_simple(transcript)
        assert any("doctor" in f.content.lower() for f in facts), (
            f"Failed to match: {phrase}"
        )


def test_case_insensitive_preference():
    """'I PREFER', 'i prefer', etc. should all match."""
    for phrase in ["I PREFER vim", "i prefer vim", "I Prefer Vim"]:
        transcript = f"User: {phrase} over emacs."
        facts = extract_facts_simple(transcript)
        assert any("vim" in f.content.lower() for f in facts), (
            f"Failed to match: {phrase}"
        )


# ── 3. Smart-quote normalisation ────────────────────────────────────────


def test_smart_quotes_normalized():
    """Curly/smart quotes must be replaced before pattern matching."""
    # Uses curly apostrophe in "I’m"
    transcript = "User: I’m a designer from Berlin."
    facts = extract_facts_simple(transcript)
    assert any("designer" in f.content.lower() for f in facts)


def test_normalize_quotes_helper():
    assert _normalize_quotes("‘hello’") == "'hello'"
    assert _normalize_quotes("“world”") == '"world"'


def test_smart_double_quotes():
    """Double curly quotes should also be normalised."""
    transcript = "User: I prefer “dark mode” for coding."
    facts = extract_facts_simple(transcript)
    assert any("dark mode" in f.content.lower() for f in facts)


# ── 4. New categories: correction, temporal, technical ──────────────────


def test_correction_category_actually():
    transcript = "User: Actually, the deadline is Friday not Thursday."
    facts = extract_facts_simple(transcript)
    assert any(f.category == "correction" for f in facts)


def test_correction_category_no_its():
    transcript = "User: No, it's Python 3.12, not 3.11."
    facts = extract_facts_simple(transcript)
    assert any(f.category == "correction" for f in facts)


def test_correction_category_i_meant():
    transcript = "User: I meant the staging server, not production."
    facts = extract_facts_simple(transcript)
    assert any(f.category == "correction" for f in facts)


def test_temporal_category_date():
    transcript = "User: The deadline is on January 15."
    facts = extract_facts_simple(transcript)
    assert any(f.category == "temporal" for f in facts)


def test_temporal_category_next_week():
    transcript = "User: Next week I'll be on vacation in Hawaii."
    facts = extract_facts_simple(transcript)
    assert any(f.category == "temporal" for f in facts)


def test_temporal_category_planning():
    transcript = "User: I'm planning to migrate the database next month."
    facts = extract_facts_simple(transcript)
    assert any(f.category == "temporal" for f in facts)


def test_technical_category_using():
    transcript = "User: I'm using PostgreSQL for the backend."
    facts = extract_facts_simple(transcript)
    assert any(f.category == "technical" for f in facts)


def test_technical_category_configured():
    transcript = "User: I configured Nginx as a reverse proxy."
    facts = extract_facts_simple(transcript)
    assert any(f.category == "technical" for f in facts)


def test_technical_category_stack():
    transcript = "User: Our stack is React, FastAPI, and PostgreSQL."
    facts = extract_facts_simple(transcript)
    assert any(f.category == "technical" for f in facts)


# ── 5. Category validation ──────────────────────────────────────────────


def test_validate_category_known():
    for cat in ALLOWED_CATEGORIES:
        assert _validate_category(cat) == cat


def test_validate_category_unknown_falls_back():
    assert _validate_category("invented_category") == "general"
    assert _validate_category("") == "general"
    assert _validate_category("BANANA") == "general"


def test_validate_category_case_insensitive():
    assert _validate_category("Personal") == "personal"
    assert _validate_category("DECISION") == "decision"


def test_llm_parser_validates_categories():
    """_parse_extraction_response must clamp invalid categories to 'general'."""
    import json
    response = json.dumps([
        {"content": "Fact one", "category": "personal"},
        {"content": "Fact two", "category": "totally_made_up"},
        {"content": "Fact three", "category": "PREFERENCE"},
    ])
    facts = _parse_extraction_response(response, max_facts=50)
    assert facts[0].category == "personal"
    assert facts[1].category == "general"  # invalid → general
    assert facts[2].category == "preference"  # uppercased → normalised


def test_heuristic_extractor_never_emits_invalid_category():
    """All facts from extract_facts_simple must have valid categories."""
    transcript = (
        "User: I'm a designer. I prefer dark mode. Actually, it's React not Vue. "
        "Next week I'll deploy. I'm using Docker. I'm working on a new feature. "
        "I started a blog. I hired a contractor. I decided to use Rust."
    )
    facts = extract_facts_simple(transcript)
    assert facts, "expected some facts"
    for f in facts:
        assert f.category in ALLOWED_CATEGORIES, (
            f"Invalid category {f.category!r} in fact: {f.content!r}"
        )


# ── Regression: existing tests still pass ───────────────────────────────


def test_simple_extractor_personal():
    """Heuristic extractor catches personal facts (regression)."""
    transcript = "User: I'm a software engineer working at Google.\n\nAssistant: Great!"
    facts = extract_facts_simple(transcript)
    assert len(facts) >= 1
    assert any("software engineer" in f.content.lower() for f in facts)


def test_simple_extractor_preference():
    """Heuristic extractor catches preferences (regression)."""
    transcript = "User: I prefer Python over JavaScript for backend work."
    facts = extract_facts_simple(transcript)
    assert len(facts) >= 1
    assert any("prefer" in f.content.lower() for f in facts)


def test_simple_extractor_no_noise():
    """Heuristic extractor doesn't extract noise (regression)."""
    transcript = "User: ok\n\nAssistant: sounds good\n\nUser: thanks"
    facts = extract_facts_simple(transcript)
    assert len(facts) == 0
