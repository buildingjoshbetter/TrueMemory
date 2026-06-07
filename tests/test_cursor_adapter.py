"""Tests for the Cursor adapter (#234).

Validates MCP config (mcp.json), hook config (hooks.json with version key),
detection, dual-file merge safety, and uninstall across both files.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path


# -- Import tests --

def test_import_cursor_adapter():
    from truememory.hooks.adapters.cursor import CursorAdapter  # noqa: F401


def test_cursor_in_registry():
    from truememory.hooks.registry import get_adapter
    adapter = get_adapter("cursor")
    assert adapter is not None
    assert adapter.cli_id == "cursor"


# -- Instantiation --

def test_cursor_adapter_properties():
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    assert adapter.name == "Cursor"
    assert adapter.cli_id == "cursor"
    assert isinstance(adapter.config_path, Path)
    assert adapter.config_path.name == "mcp.json"


def test_cursor_implements_all_abstract_methods():
    from truememory.hooks.adapters.base import CLIAdapter
    from truememory.hooks.adapters.cursor import CursorAdapter
    abstract_methods = {
        name for name, _ in inspect.getmembers(CLIAdapter)
        if getattr(getattr(CLIAdapter, name, None), "__isabstractmethod__", False)
    }
    adapter = CursorAdapter()
    for method_name in abstract_methods:
        assert hasattr(adapter, method_name), f"Missing: {method_name}"


# -- Detection --

def test_detect_false_no_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    monkeypatch.setattr(cursor_mod, "_CURSOR_DIR", tmp_path / "nonexistent")
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    monkeypatch.setattr("shutil.which", lambda x: None)
    assert not adapter.detect()


def test_detect_true_with_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir()
    monkeypatch.setattr(cursor_mod, "_CURSOR_DIR", cursor_dir)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    assert adapter.detect()


# -- MCP config (mcp.json) --

def test_install_mcp_creates_config(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    mcp_path = tmp_path / "mcp.json"
    monkeypatch.setattr(cursor_mod, "_MCP_CONFIG", mcp_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")

    data = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert "truememory" in data["mcpServers"]
    assert data["mcpServers"]["truememory"]["command"] == "/usr/bin/python3"
    assert data["mcpServers"]["truememory"]["args"] == ["-m", "truememory.mcp_server"]


def test_install_mcp_preserves_existing(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    mcp_path = tmp_path / "mcp.json"
    mcp_path.write_text(json.dumps({
        "mcpServers": {
            "other-server": {"command": "other", "args": ["arg"]}
        }
    }), encoding="utf-8")
    monkeypatch.setattr(cursor_mod, "_MCP_CONFIG", mcp_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")

    data = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert "truememory" in data["mcpServers"]
    assert "other-server" in data["mcpServers"]


def test_install_mcp_idempotent(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    mcp_path = tmp_path / "mcp.json"
    monkeypatch.setattr(cursor_mod, "_MCP_CONFIG", mcp_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")
    first = mcp_path.read_text(encoding="utf-8")
    adapter.install_mcp(python_path="/usr/bin/python3")
    second = mcp_path.read_text(encoding="utf-8")
    assert first == second


# -- Hook config (hooks.json) --

def test_install_hooks_creates_entries(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    hook_path = tmp_path / "hooks.json"
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", hook_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    data = json.loads(hook_path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    hooks = data["hooks"]
    assert "sessionStart" in hooks
    assert "stop" in hooks
    assert "beforeSubmitPrompt" in hooks
    assert "preCompact" in hooks
    assert len(hooks["sessionStart"]) == 1
    assert "truememory" in hooks["sessionStart"][0]["command"].lower()


def test_install_hooks_has_version_key(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    hook_path = tmp_path / "hooks.json"
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", hook_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    data = json.loads(hook_path.read_text(encoding="utf-8"))
    assert "version" in data
    assert data["version"] == 1


def test_install_hooks_preserves_existing(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    hook_path = tmp_path / "hooks.json"
    hook_path.write_text(json.dumps({
        "version": 1,
        "hooks": {
            "sessionStart": [
                {"command": "my-custom-hook", "timeout": 5000}
            ]
        },
    }), encoding="utf-8")
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", hook_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    data = json.loads(hook_path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert len(data["hooks"]["sessionStart"]) == 2
    assert data["hooks"]["sessionStart"][0]["command"] == "my-custom-hook"


def test_install_hooks_preserves_existing_version(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    hook_path = tmp_path / "hooks.json"
    hook_path.write_text(json.dumps({
        "version": 1,
        "customSetting": "keep-me",
    }), encoding="utf-8")
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", hook_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    data = json.loads(hook_path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["customSetting"] == "keep-me"


def test_install_hooks_idempotent(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    hook_path = tmp_path / "hooks.json"
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", hook_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")
    first = hook_path.read_text(encoding="utf-8")
    adapter.install_hooks(python_path="/usr/bin/python3")
    second = hook_path.read_text(encoding="utf-8")
    assert first == second


# -- Uninstall (must clean BOTH files) --

def test_uninstall_removes_from_both_files(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    mcp_path = tmp_path / "mcp.json"
    hook_path = tmp_path / "hooks.json"
    monkeypatch.setattr(cursor_mod, "_MCP_CONFIG", mcp_path)
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", hook_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()

    adapter.install_mcp(python_path="/usr/bin/python3")
    adapter.install_hooks(python_path="/usr/bin/python3")
    assert adapter.is_configured()

    adapter.uninstall()

    mcp_data = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert "truememory" not in mcp_data.get("mcpServers", {})

    hook_data = json.loads(hook_path.read_text(encoding="utf-8"))
    hooks = hook_data.get("hooks", {})
    for entries in hooks.values():
        for h in entries:
            assert "truememory" not in h.get("command", "").lower()


def test_uninstall_preserves_other_entries(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    mcp_path = tmp_path / "mcp.json"
    hook_path = tmp_path / "hooks.json"

    mcp_path.write_text(json.dumps({
        "mcpServers": {
            "other-server": {"command": "other"},
            "truememory": {"command": "python", "args": ["-m", "truememory.mcp_server"]},
        }
    }), encoding="utf-8")

    hook_path.write_text(json.dumps({
        "version": 1,
        "hooks": {
            "sessionStart": [
                {"command": "my-hook", "timeout": 5000},
                {"command": "/path/to/truememory/hooks/session_start.py", "timeout": 10000},
            ]
        },
    }), encoding="utf-8")

    monkeypatch.setattr(cursor_mod, "_MCP_CONFIG", mcp_path)
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", hook_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.uninstall()

    mcp_data = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert "other-server" in mcp_data["mcpServers"]
    assert "truememory" not in mcp_data["mcpServers"]

    hook_data = json.loads(hook_path.read_text(encoding="utf-8"))
    assert hook_data["version"] == 1
    assert len(hook_data["hooks"]["sessionStart"]) == 1
    assert hook_data["hooks"]["sessionStart"][0]["command"] == "my-hook"


# -- is_configured (checks BOTH files) --

def test_is_configured_false_clean(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    monkeypatch.setattr(cursor_mod, "_MCP_CONFIG", tmp_path / "mcp.json")
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", tmp_path / "hooks.json")
    from truememory.hooks.adapters.cursor import CursorAdapter
    assert not CursorAdapter().is_configured()


def test_is_configured_true_with_mcp_only(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    mcp_path = tmp_path / "mcp.json"
    monkeypatch.setattr(cursor_mod, "_MCP_CONFIG", mcp_path)
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", tmp_path / "hooks.json")
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")
    assert adapter.is_configured()


def test_is_configured_true_with_hooks_only(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    hook_path = tmp_path / "hooks.json"
    monkeypatch.setattr(cursor_mod, "_MCP_CONFIG", tmp_path / "mcp.json")
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", hook_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")
    assert adapter.is_configured()


# -- verify (requires BOTH) --

def test_verify_requires_both(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    mcp_path = tmp_path / "mcp.json"
    hook_path = tmp_path / "hooks.json"
    monkeypatch.setattr(cursor_mod, "_MCP_CONFIG", mcp_path)
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", hook_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")
    assert not adapter.verify()
    adapter.install_hooks(python_path="/usr/bin/python3")
    assert adapter.verify()


# -- System prompt --

def test_system_prompt_path():
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    path = adapter.get_system_prompt_path()
    assert path is not None
    assert path.name == ".cursorrules"


def test_system_prompt_content():
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    content = adapter.get_system_prompt_content()
    assert "TrueMemory" in content
    assert "truememory_search" in content


# -- Build command --

def test_build_command_with_user_and_db():
    from truememory.hooks.adapters.cursor import CursorAdapter
    cmd = CursorAdapter._build_command(
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


# -- Event names are camelCase --

def test_hook_events_are_camelcase(tmp_path, monkeypatch):
    from truememory.hooks.adapters import cursor as cursor_mod
    hook_path = tmp_path / "hooks.json"
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", hook_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    data = json.loads(hook_path.read_text(encoding="utf-8"))
    event_names = list(data["hooks"].keys())
    for name in event_names:
        assert name[0].islower(), f"Event {name!r} should be camelCase, not PascalCase"


def test_hook_uses_before_submit_prompt_not_user_prompt_submit(tmp_path, monkeypatch):
    """Regression: Cursor's event is beforeSubmitPrompt, not userPromptSubmit."""
    from truememory.hooks.adapters import cursor as cursor_mod
    hook_path = tmp_path / "hooks.json"
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", hook_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    data = json.loads(hook_path.read_text(encoding="utf-8"))
    hooks = data["hooks"]
    assert "beforeSubmitPrompt" in hooks, "Expected beforeSubmitPrompt event"
    assert "userPromptSubmit" not in hooks, (
        "userPromptSubmit is a Claude Code event name, not a Cursor event"
    )
    assert len(hooks["beforeSubmitPrompt"]) == 1
    assert "truememory" in hooks["beforeSubmitPrompt"][0]["command"].lower()


def test_hook_timeouts_are_in_seconds(tmp_path, monkeypatch):
    """Cursor hook timeouts are in seconds. Values like 5000 or 10000 are wrong
    (those would be interpreted as 5000/10000 seconds). Sane values are 1-120."""
    from truememory.hooks.adapters import cursor as cursor_mod
    hook_path = tmp_path / "hooks.json"
    monkeypatch.setattr(cursor_mod, "_HOOK_CONFIG", hook_path)
    from truememory.hooks.adapters.cursor import CursorAdapter
    adapter = CursorAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    data = json.loads(hook_path.read_text(encoding="utf-8"))
    for event_name, entries in data["hooks"].items():
        for entry in entries:
            timeout = entry.get("timeout")
            if timeout is not None:
                assert 1 <= timeout <= 120, (
                    f"Timeout for {event_name!r} is {timeout}; expected 1-120 seconds, "
                    f"not milliseconds"
                )
