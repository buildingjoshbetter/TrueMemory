"""Tests for the removed OpenClaw adapter.

The OpenClaw adapter was removed in June 2026 because the target product
(openclaw/openclaw) does not exist as described. These tests verify that:
1. The stub module is importable (no ImportError for existing code)
2. The adapter is NOT in the active registry
3. Install methods raise NotImplementedError
4. detect() and is_configured() return False
5. The _strip_json5_comments utility is preserved and functional
"""
from __future__ import annotations

import json

import pytest


def test_import_openclaw_adapter():
    """Stub module should still be importable."""
    from truememory.hooks.adapters.openclaw import OpenClawAdapter  # noqa: F401


def test_openclaw_not_in_registry():
    """Removed adapters must not appear in the active registry."""
    from truememory.hooks.registry import get_adapter
    assert get_adapter("openclaw") is None


def test_openclaw_detect_returns_false():
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    assert not OpenClawAdapter().detect()


def test_openclaw_is_configured_returns_false():
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    assert not OpenClawAdapter().is_configured()


def test_openclaw_verify_returns_false():
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    assert not OpenClawAdapter().verify()


def test_openclaw_install_mcp_raises():
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    with pytest.raises(NotImplementedError, match="does not exist"):
        OpenClawAdapter().install_mcp()


def test_openclaw_install_hooks_raises():
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    with pytest.raises(NotImplementedError, match="does not exist"):
        OpenClawAdapter().install_hooks()


def test_openclaw_uninstall_raises():
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    with pytest.raises(NotImplementedError, match="does not exist"):
        OpenClawAdapter().uninstall()


def test_openclaw_is_cli_adapter_subclass():
    """Stub must remain a CLIAdapter subclass for isinstance checks."""
    from truememory.hooks.adapters.base import CLIAdapter
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    adapter = OpenClawAdapter()
    assert isinstance(adapter, CLIAdapter)


def test_openclaw_stub_properties():
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    adapter = OpenClawAdapter()
    assert adapter.cli_id == "openclaw"
    assert "REMOVED" in adapter.name
    assert adapter.config_path.name == "openclaw.json"
    assert adapter.get_system_prompt_path() is None
    assert adapter.get_system_prompt_content() == ""


# -- JSON5 comment stripping (utility preserved from original adapter) --

def test_strip_json5_comments():
    from truememory.hooks.adapters.openclaw import _strip_json5_comments
    text = '// comment\n{"key": "val", // inline\n}'
    cleaned = _strip_json5_comments(text)
    data = json.loads(cleaned)
    assert data["key"] == "val"


def test_strip_json5_trailing_commas():
    from truememory.hooks.adapters.openclaw import _strip_json5_comments
    text = '{"a": 1, "b": 2,}'
    cleaned = _strip_json5_comments(text)
    data = json.loads(cleaned)
    assert data["a"] == 1
    assert data["b"] == 2
