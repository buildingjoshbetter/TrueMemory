"""Comprehensive regression tests for directive discoverability across all 4 Claude-facing surfaces.

Covers:
1. CLAUDE_TEMPLATE.md — directive guidance in Auto-Store and Auto-Recall
2. MCP instructions — Directives section, storing cross-reference, trigger phrases, management
3. Tool schemas — directive parameter, docstrings, alwaysLoad metadata
4. Session start hook — directive ordering, deduplication, zero-directive behavior
"""

import inspect
import os
import tempfile
from pathlib import Path

TEMPLATE = Path(__file__).parent.parent / "truememory" / "ingest" / "CLAUDE_TEMPLATE.md"


# ── Surface 1: CLAUDE_TEMPLATE.md ──────────────────────────────────────────


def test_template_mentions_directives():
    text = TEMPLATE.read_text()
    assert "directive" in text.lower(), "CLAUDE_TEMPLATE.md must mention directives"


def test_template_auto_store_has_directive_guidance():
    text = TEMPLATE.read_text()
    auto_store_start = text.index("Auto-Store")
    next_section = text.index("##", auto_store_start + 1)
    auto_store = text[auto_store_start:next_section]
    assert "directive=True" in auto_store or "directive=true" in auto_store, (
        "Auto-Store section must mention directive=True"
    )


def test_template_has_trigger_phrases():
    text = TEMPLATE.read_text()
    lower = text.lower()
    assert "always" in lower, "Template must include 'always' trigger phrase"
    assert "never" in lower, "Template must include 'never' trigger phrase"


def test_template_auto_recall_mentions_directives():
    text = TEMPLATE.read_text()
    recall_start = text.index("Auto-Recall")
    next_section = text.index("##", recall_start + 1)
    recall = text[recall_start:next_section]
    assert "directive" in recall.lower(), (
        "Auto-Recall section must mention that directives are auto-injected"
    )


# ── Surface 2: MCP instructions ────────────────────────────────────────────


def _get_instructions():
    from truememory.mcp_server import mcp
    return mcp.instructions


def test_mcp_instructions_has_directives_section():
    instructions = _get_instructions()
    assert "Directives" in instructions, "MCP instructions must have Directives section"


def test_mcp_instructions_storing_cross_references_directives():
    instructions = _get_instructions()
    storing_start = instructions.index("Storing memories")
    storing_end = instructions.index("Recalling memories")
    storing = instructions[storing_start:storing_end]
    assert "directive" in storing.lower(), (
        "Storing memories section must cross-reference directives"
    )


def test_mcp_instructions_directive_trigger_phrases():
    instructions = _get_instructions()
    lower = instructions.lower()
    assert "always do" in lower
    assert "never do" in lower
    assert "from now on" in lower


def test_mcp_instructions_directive_management():
    instructions = _get_instructions()
    assert "truememory_forget" in instructions, (
        "Directives section must explain how to delete directives"
    )


def test_mcp_instructions_directive_contradiction():
    instructions = _get_instructions()
    lower = instructions.lower()
    assert "contradict" in lower or "conflict" in lower, (
        "Directives section must mention handling contradictions"
    )


# ── Surface 3: Tool schemas ────────────────────────────────────────────────


def test_truememory_store_has_directive_param():
    from truememory.mcp_server import truememory_store
    sig = inspect.signature(truememory_store)
    assert "directive" in sig.parameters
    assert sig.parameters["directive"].default is False


def test_truememory_store_docstring_documents_directive():
    from truememory.mcp_server import truememory_store
    assert truememory_store.__doc__ is not None
    assert "directive" in truememory_store.__doc__.lower()


def test_truememory_directives_exists():
    from truememory.mcp_server import truememory_directives
    assert callable(truememory_directives)
    assert truememory_directives.__doc__ is not None


def test_truememory_forget_exists():
    from truememory.mcp_server import truememory_forget
    assert callable(truememory_forget)


def test_truememory_forget_has_always_load():
    from truememory.mcp_server import mcp
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    forget_tool = tools.get("truememory_forget")
    assert forget_tool is not None, "truememory_forget tool not registered"
    meta = getattr(forget_tool, "meta", None) or {}
    assert meta.get("anthropic/alwaysLoad") is True, (
        f"truememory_forget must have alwaysLoad=True, got {meta}"
    )


# ── Surface 4: Session start hook ──────────────────────────────────────────


def test_session_start_directives_before_context():
    from truememory import Memory
    from truememory.ingest.hooks.session_start import recall_memories

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        m = Memory(path=db_path)
        m.add("Always use dark mode", user_id="testuser", directive=True)
        m.add("Likes Python", user_id="testuser")
        m.close()
        ctx = recall_memories({}, user_id="testuser", db_path=db_path)
        assert "<truememory-directives>" in ctx
        assert "<truememory-context>" in ctx
        assert ctx.index("<truememory-directives>") < ctx.index("<truememory-context>")
    finally:
        os.unlink(db_path)


def test_session_start_no_directive_duplication():
    from truememory import Memory
    from truememory.ingest.hooks.session_start import recall_memories

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        m = Memory(path=db_path)
        m.add("Always use dark mode", user_id="testuser", directive=True)
        m.add("Likes Python", user_id="testuser")
        m.close()
        ctx = recall_memories({}, user_id="testuser", db_path=db_path)
        if "<truememory-context>" in ctx:
            context_block = ctx.split("<truememory-context>")[1].split("</truememory-context>")[0]
            assert "dark mode" not in context_block, (
                "Directive content should not appear in context block"
            )
    finally:
        os.unlink(db_path)


def test_session_start_zero_directives_no_block():
    from truememory import Memory
    from truememory.ingest.hooks.session_start import recall_memories

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        m = Memory(path=db_path)
        m.add("Likes Python", user_id="testuser")
        m.close()
        ctx = recall_memories({}, user_id="testuser", db_path=db_path)
        assert "<truememory-directives>" not in ctx, (
            "Directives block should not appear when there are no directives"
        )
    finally:
        os.unlink(db_path)
