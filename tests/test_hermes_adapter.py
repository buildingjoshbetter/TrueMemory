"""Tests for the removed Hermes Agent adapter.

The Hermes adapter was removed in June 2026 because the target product
(NousResearch/hermes-agent) does not exist. These tests verify that:
1. The stub module is importable (no ImportError for existing code)
2. The adapter is NOT in the active registry
3. Install methods raise NotImplementedError
4. detect() and is_configured() return False
"""
from __future__ import annotations

import pytest


def test_import_hermes_adapter():
    """Stub module should still be importable."""
    from truememory.hooks.adapters.hermes import HermesAdapter  # noqa: F401


def test_hermes_not_in_registry():
    """Removed adapters must not appear in the active registry."""
    from truememory.hooks.registry import get_adapter
    assert get_adapter("hermes") is None


def test_hermes_detect_returns_false():
    from truememory.hooks.adapters.hermes import HermesAdapter
    assert not HermesAdapter().detect()


def test_hermes_is_configured_returns_false():
    from truememory.hooks.adapters.hermes import HermesAdapter
    assert not HermesAdapter().is_configured()


def test_hermes_verify_returns_false():
    from truememory.hooks.adapters.hermes import HermesAdapter
    assert not HermesAdapter().verify()


def test_hermes_install_mcp_raises():
    from truememory.hooks.adapters.hermes import HermesAdapter
    with pytest.raises(NotImplementedError, match="does not exist"):
        HermesAdapter().install_mcp()


def test_hermes_install_hooks_raises():
    from truememory.hooks.adapters.hermes import HermesAdapter
    with pytest.raises(NotImplementedError, match="does not exist"):
        HermesAdapter().install_hooks()


def test_hermes_uninstall_raises():
    from truememory.hooks.adapters.hermes import HermesAdapter
    with pytest.raises(NotImplementedError, match="does not exist"):
        HermesAdapter().uninstall()


def test_hermes_is_cli_adapter_subclass():
    """Stub must remain a CLIAdapter subclass for isinstance checks."""
    from truememory.hooks.adapters.base import CLIAdapter
    from truememory.hooks.adapters.hermes import HermesAdapter
    adapter = HermesAdapter()
    assert isinstance(adapter, CLIAdapter)


def test_hermes_stub_properties():
    from truememory.hooks.adapters.hermes import HermesAdapter
    adapter = HermesAdapter()
    assert adapter.cli_id == "hermes"
    assert "REMOVED" in adapter.name
    assert adapter.config_path.name == "config.yaml"
    assert adapter.get_system_prompt_path() is None
    assert adapter.get_system_prompt_content() == ""
