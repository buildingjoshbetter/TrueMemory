"""Regression tests for issues #564 and #565: MCP instructions directive discoverability."""


def _get_instructions():
    from truememory.mcp_server import mcp
    return mcp.instructions


def test_issue_564_storing_section_cross_references_directives():
    """The 'Storing memories' section must mention directives."""
    instructions = _get_instructions()
    storing_start = instructions.index("Storing memories")
    storing_end = instructions.index("Recalling memories")
    storing = instructions[storing_start:storing_end]
    assert "directive" in storing.lower(), (
        "Storing memories section must cross-reference directives"
    )


def test_issue_565_directive_trigger_from_now_on():
    instructions = _get_instructions()
    assert "from now on" in instructions.lower(), (
        "Directives section must include 'from now on' trigger phrase"
    )


def test_issue_565_directive_trigger_every_session():
    instructions = _get_instructions()
    assert "every session" in instructions.lower(), (
        "Directives section must include 'every session' trigger phrase"
    )


def test_issue_565_directive_management_forget():
    instructions = _get_instructions()
    directives_start = instructions.index("Directives")
    directives_section = instructions[directives_start:]
    assert "truememory_forget" in directives_section, (
        "Directives section must explain how to delete directives via truememory_forget"
    )


def test_issue_565_directive_management_contradict():
    instructions = _get_instructions()
    lower = instructions.lower()
    assert "contradict" in lower or "conflict" in lower, (
        "Directives section must mention handling contradictions/conflicts"
    )
