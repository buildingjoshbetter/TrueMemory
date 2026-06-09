"""Regression test for issue #566: truememory_forget must have alwaysLoad meta."""


def test_issue_566_forget_has_always_load():
    """truememory_forget must declare alwaysLoad so it is never deferred behind ToolSearch."""
    from truememory.mcp_server import truememory_forget

    tool_meta = getattr(truememory_forget, "_mcp_tool_meta", None)
    if tool_meta is None:
        # FastMCP stores metadata via the _tool_meta attribute or in the
        # tool's extra field.  Try the public .meta attribute on the Tool
        # object exposed by mcp.list_tools() instead.
        from truememory.mcp_server import mcp

        tools = {t.name: t for t in mcp._tool_manager.list_tools()}
        forget_tool = tools.get("truememory_forget")
        assert forget_tool is not None, "truememory_forget tool not registered"
        meta = getattr(forget_tool, "meta", None) or {}
        assert meta.get("anthropic/alwaysLoad") is True, (
            f"truememory_forget must have alwaysLoad=True in meta, got {meta}"
        )
    else:
        assert tool_meta.get("anthropic/alwaysLoad") is True
