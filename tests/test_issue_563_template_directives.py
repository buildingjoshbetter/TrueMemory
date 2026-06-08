"""Regression tests for issue #563: CLAUDE_TEMPLATE.md must mention directives."""

from pathlib import Path

TEMPLATE = Path(__file__).parent.parent / "truememory" / "ingest" / "CLAUDE_TEMPLATE.md"


def test_issue_563_template_mentions_directives():
    text = TEMPLATE.read_text()
    assert "directive" in text.lower(), "CLAUDE_TEMPLATE.md must mention directives"


def test_issue_563_template_auto_store_has_directive_true():
    text = TEMPLATE.read_text()
    auto_store_start = text.index("Auto-Store")
    next_section = text.index("##", auto_store_start + 1)
    auto_store = text[auto_store_start:next_section]
    assert "directive=True" in auto_store or "directive=true" in auto_store, (
        "Auto-Store section must mention directive=True"
    )


def test_issue_563_template_has_trigger_phrases():
    text = TEMPLATE.read_text()
    lower = text.lower()
    assert "always" in lower, "Template must include 'always' trigger phrase"
    assert "never" in lower, "Template must include 'never' trigger phrase"
