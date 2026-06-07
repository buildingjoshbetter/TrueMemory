"""Tests for the Kimi CLI adapter (#183).

Validates MCP config, TOML hook registration, detection, and
config merge safety without network calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest



# -- Import tests --

def test_import_kimi_adapter():
    from truememory.hooks.adapters.kimi import KimiAdapter  # noqa: F401


def test_kimi_in_registry():
    from truememory.hooks.registry import get_adapter
    adapter = get_adapter("kimi")
    assert adapter is not None
    assert adapter.cli_id == "kimi"


# -- Instantiation --

def test_kimi_adapter_properties():
    from truememory.hooks.adapters.kimi import KimiAdapter
    adapter = KimiAdapter()
    assert adapter.name == "Kimi CLI"
    assert adapter.cli_id == "kimi"
    assert isinstance(adapter.config_path, Path)
    assert adapter.config_path.name == "config.toml"


def test_kimi_implements_all_abstract_methods():
    from truememory.hooks.adapters.base import CLIAdapter
    from truememory.hooks.adapters.kimi import KimiAdapter
    import inspect
    abstract_methods = {
        name for name, _ in inspect.getmembers(CLIAdapter)
        if getattr(getattr(CLIAdapter, name, None), "__isabstractmethod__", False)
    }
    adapter = KimiAdapter()
    for method_name in abstract_methods:
        assert hasattr(adapter, method_name), f"Missing: {method_name}"


# -- Detection --

def test_detect_false_no_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import kimi as kimi_mod
    monkeypatch.setattr(kimi_mod, "_KIMI_DIR", tmp_path / "nonexistent")
    from truememory.hooks.adapters.kimi import KimiAdapter
    adapter = KimiAdapter()
    monkeypatch.setattr("shutil.which", lambda x: None)
    assert not adapter.detect()


def test_detect_true_with_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import kimi as kimi_mod
    kimi_dir = tmp_path / ".kimi"
    kimi_dir.mkdir()
    monkeypatch.setattr(kimi_mod, "_KIMI_DIR", kimi_dir)
    from truememory.hooks.adapters.kimi import KimiAdapter
    adapter = KimiAdapter()
    assert adapter.detect()


# -- MCP config --

def test_install_mcp_creates_config(tmp_path, monkeypatch):
    from truememory.hooks.adapters import kimi as kimi_mod
    mcp_path = tmp_path / "mcp.json"
    monkeypatch.setattr(kimi_mod, "_MCP_CONFIG", mcp_path)
    from truememory.hooks.adapters.kimi import KimiAdapter
    adapter = KimiAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")

    data = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert "truememory" in data["mcpServers"]
    assert data["mcpServers"]["truememory"]["command"] == "/usr/bin/python3"
    assert data["mcpServers"]["truememory"]["args"] == ["-m", "truememory.mcp_server"]


def test_install_mcp_preserves_existing(tmp_path, monkeypatch):
    from truememory.hooks.adapters import kimi as kimi_mod
    mcp_path = tmp_path / "mcp.json"
    mcp_path.write_text(json.dumps({
        "mcpServers": {
            "other-server": {"command": "other", "args": ["arg"]}
        }
    }), encoding="utf-8")
    monkeypatch.setattr(kimi_mod, "_MCP_CONFIG", mcp_path)
    from truememory.hooks.adapters.kimi import KimiAdapter
    adapter = KimiAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")

    data = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert "truememory" in data["mcpServers"]
    assert "other-server" in data["mcpServers"]


# -- TOML hook config --

def test_install_hooks_creates_toml(tmp_path, monkeypatch):
    from truememory.hooks.adapters import kimi as kimi_mod
    hook_config = tmp_path / "config.toml"
    monkeypatch.setattr(kimi_mod, "_HOOK_CONFIG", hook_config)
    from truememory.hooks.adapters.kimi import KimiAdapter
    adapter = KimiAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    text = hook_config.read_text(encoding="utf-8")
    assert "[[hooks]]" in text
    assert 'event = "SessionStart"' in text
    assert 'event = "Stop"' in text
    assert 'event = "PreCompact"' in text
    assert "truememory" in text.lower()


def test_install_hooks_preserves_existing_toml(tmp_path, monkeypatch):
    from truememory.hooks.adapters import kimi as kimi_mod
    hook_config = tmp_path / "config.toml"
    hook_config.write_text(
        '[general]\ntheme = "dark"\n\n[[hooks]]\nevent = "MyCustomHook"\ncommand = "my-cmd"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(kimi_mod, "_HOOK_CONFIG", hook_config)
    from truememory.hooks.adapters.kimi import KimiAdapter
    adapter = KimiAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    text = hook_config.read_text(encoding="utf-8")
    assert 'theme = "dark"' in text
    assert 'event = "MyCustomHook"' in text
    assert 'event = "SessionStart"' in text


@pytest.mark.skipif(sys.platform == "win32", reason="TOML path handling differs on Windows")
def test_install_hooks_idempotent(tmp_path, monkeypatch):
    from truememory.hooks.adapters import kimi as kimi_mod
    hook_config = tmp_path / "config.toml"
    monkeypatch.setattr(kimi_mod, "_HOOK_CONFIG", hook_config)
    from truememory.hooks.adapters.kimi import KimiAdapter
    adapter = KimiAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")
    first_text = hook_config.read_text(encoding="utf-8")
    adapter.install_hooks(python_path="/usr/bin/python3")
    second_text = hook_config.read_text(encoding="utf-8")
    assert first_text == second_text


# -- Uninstall --

def test_uninstall_removes_entries(tmp_path, monkeypatch):
    from truememory.hooks.adapters import kimi as kimi_mod
    mcp_path = tmp_path / "mcp.json"
    hook_config = tmp_path / "config.toml"
    monkeypatch.setattr(kimi_mod, "_MCP_CONFIG", mcp_path)
    monkeypatch.setattr(kimi_mod, "_HOOK_CONFIG", hook_config)
    from truememory.hooks.adapters.kimi import KimiAdapter
    adapter = KimiAdapter()

    adapter.install_mcp(python_path="/usr/bin/python3")
    adapter.install_hooks(python_path="/usr/bin/python3")
    assert adapter.is_configured()

    adapter.uninstall()
    data = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert "truememory" not in data.get("mcpServers", {})

    text = hook_config.read_text(encoding="utf-8")
    assert "truememory" not in text.lower()


# -- is_configured --

def test_is_configured_false_clean(tmp_path, monkeypatch):
    from truememory.hooks.adapters import kimi as kimi_mod
    monkeypatch.setattr(kimi_mod, "_MCP_CONFIG", tmp_path / "mcp.json")
    monkeypatch.setattr(kimi_mod, "_HOOK_CONFIG", tmp_path / "config.toml")
    from truememory.hooks.adapters.kimi import KimiAdapter
    assert not KimiAdapter().is_configured()


# -- Build command --

def test_build_command_with_user_and_db():
    from truememory.hooks.adapters.kimi import KimiAdapter
    cmd = KimiAdapter._build_command(
        "/usr/bin/python3",
        Path("/path/to/session_start.py"),
        user_id="alice",
        db_path="/data/mem.db",
    )
    assert "/usr/bin/python3" in cmd
    assert "session_start.py" in cmd
    assert "--user" in cmd
    assert "alice" in cmd
    assert "--db" in cmd


# -- Timeout values (regression test for ms-vs-seconds bug) --

def test_hook_timeout_values_are_seconds(tmp_path, monkeypatch):
    """Kimi CLI interprets timeout as seconds; verify we don't emit milliseconds."""
    from truememory.hooks.adapters import kimi as kimi_mod
    hook_config = tmp_path / "config.toml"
    monkeypatch.setattr(kimi_mod, "_HOOK_CONFIG", hook_config)
    from truememory.hooks.adapters.kimi import KimiAdapter
    adapter = KimiAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    text = hook_config.read_text(encoding="utf-8")
    # SessionStart gets 10s, everything else gets 5s
    assert "timeout = 10" in text
    assert "timeout = 5" in text
    # Must NOT contain millisecond values
    assert "timeout = 10000" not in text
    assert "timeout = 5000" not in text


# -- Uninstall with blank lines in TOML (regression test) --

def test_uninstall_handles_blank_lines_in_hook_block(tmp_path, monkeypatch):
    """Hook blocks with internal blank lines must be fully removed."""
    from truememory.hooks.adapters import kimi as kimi_mod
    hook_config = tmp_path / "config.toml"
    mcp_path = tmp_path / "mcp.json"
    # Write a hook block that has a blank line inside it
    hook_config.write_text(
        '[[hooks]]\nevent = "SessionStart"\ncommand = "python truememory/ingest/hooks/session_start.py"\n\ntimeout = 10\n',
        encoding="utf-8",
    )
    mcp_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(kimi_mod, "_HOOK_CONFIG", hook_config)
    monkeypatch.setattr(kimi_mod, "_MCP_CONFIG", mcp_path)
    from truememory.hooks.adapters.kimi import KimiAdapter
    adapter = KimiAdapter()
    adapter.uninstall()

    text = hook_config.read_text(encoding="utf-8")
    assert "truememory" not in text.lower()
    assert "timeout" not in text


# -- Uninstall preserves non-TrueMemory content (regression test) --

def test_uninstall_preserves_unrelated_hooks_and_tables(tmp_path, monkeypatch):
    """Uninstall must not delete user hooks or unrelated TOML tables."""
    from truememory.hooks.adapters import kimi as kimi_mod
    hook_config = tmp_path / "config.toml"
    mcp_path = tmp_path / "mcp.json"
    hook_config.write_text(
        '[[hooks]]\nevent = "SessionStart"\n'
        'command = "python truememory/ingest/hooks/session_start.py"\n\n'
        'timeout = 10\n\n'
        '[[hooks]]\nevent = "Stop"\n'
        'command = "my-own-script.sh"\ntimeout = 30\n\n'
        '[settings]\ntheme = "dark"\n',
        encoding="utf-8",
    )
    mcp_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(kimi_mod, "_HOOK_CONFIG", hook_config)
    monkeypatch.setattr(kimi_mod, "_MCP_CONFIG", mcp_path)
    from truememory.hooks.adapters.kimi import KimiAdapter
    adapter = KimiAdapter()
    adapter.uninstall()

    text = hook_config.read_text(encoding="utf-8")
    # TrueMemory hook must be gone
    assert "truememory" not in text.lower()
    # User's own hook must survive
    assert "my-own-script.sh" in text
    assert 'event = "Stop"' in text
    # Unrelated TOML table must survive
    assert "[settings]" in text
    assert 'theme = "dark"' in text
