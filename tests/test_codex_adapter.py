"""Tests for the Codex CLI adapter (#182).

Validates TOML-based MCP config, hook registration, detection,
config merge safety, and TOML section removal without network calls.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest


# -- Import tests --

def test_import_codex_adapter():
    from truememory.hooks.adapters.codex import CodexAdapter  # noqa: F401


def test_codex_in_registry():
    from truememory.hooks.registry import get_adapter
    adapter = get_adapter("codex")
    assert adapter is not None
    assert adapter.cli_id == "codex"


# -- Instantiation --

def test_codex_adapter_properties():
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    assert adapter.name == "Codex CLI"
    assert adapter.cli_id == "codex"
    assert isinstance(adapter.config_path, Path)
    assert adapter.config_path.name == "config.toml"


def test_codex_implements_all_abstract_methods():
    from truememory.hooks.adapters.base import CLIAdapter
    from truememory.hooks.adapters.codex import CodexAdapter
    abstract_methods = {
        name for name, _ in inspect.getmembers(CLIAdapter)
        if getattr(getattr(CLIAdapter, name, None), "__isabstractmethod__", False)
    }
    adapter = CodexAdapter()
    for method_name in abstract_methods:
        assert hasattr(adapter, method_name), f"Missing: {method_name}"


# -- Detection --

def test_detect_false_no_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    monkeypatch.setattr(codex_mod, "_CODEX_DIR", tmp_path / "nonexistent")
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    monkeypatch.setattr("shutil.which", lambda x: None)
    assert not adapter.detect()


def test_detect_true_with_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    monkeypatch.setattr(codex_mod, "_CODEX_DIR", codex_dir)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    assert adapter.detect()


# -- MCP config (TOML-based, not JSON) --

def test_install_mcp_creates_config(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")

    text = config_path.read_text(encoding="utf-8")
    assert "[mcp_servers.truememory]" in text
    assert '"/usr/bin/python3"' in text
    assert 'truememory.mcp_server' in text


def test_install_mcp_preserves_existing(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[general]\ntheme = "dark"\n\n[mcp_servers.other]\ncommand = "other"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")

    text = config_path.read_text(encoding="utf-8")
    assert '[mcp_servers.truememory]' in text
    assert '[mcp_servers.other]' in text
    assert 'theme = "dark"' in text


def test_install_mcp_idempotent(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")
    first_text = config_path.read_text(encoding="utf-8")
    adapter.install_mcp(python_path="/usr/bin/python3")
    second_text = config_path.read_text(encoding="utf-8")
    assert first_text == second_text


# -- Hook config --

def test_install_hooks_creates_correct_toml_structure(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    text = config_path.read_text(encoding="utf-8")
    # Correct format
    assert "[[hooks.SessionStart]]" in text
    assert "[[hooks.Stop]]" in text
    assert "[[hooks.UserPromptSubmit]]" in text
    assert "[[hooks.PreCompact]]" in text
    assert "[[hooks.SessionStart.hooks]]" in text
    assert 'type = "command"' in text
    assert "truememory" in text.lower()
    # Must NOT use the old flat format
    assert "[[hooks]]" not in text
    assert 'event = ' not in text


def test_install_hooks_timeout_is_seconds(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    text = config_path.read_text(encoding="utf-8")
    assert "timeout = 10" in text
    assert "timeout = 5" in text
    # Must NOT have millisecond values
    assert "10000" not in text
    assert "5000" not in text


def test_install_hooks_valid_toml(tmp_path, monkeypatch):
    """The generated config must parse as valid TOML with the nested hooks structure."""
    import tomllib

    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    text = config_path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    assert "hooks" in data
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    assert "SessionStart" in hooks
    assert "Stop" in hooks
    # Each event has a list of matcher groups, each with a "hooks" list
    for event_name in ("SessionStart", "Stop", "UserPromptSubmit", "PreCompact"):
        matcher_groups = hooks[event_name]
        assert isinstance(matcher_groups, list)
        assert len(matcher_groups) >= 1
        mg = matcher_groups[0]
        assert "hooks" in mg
        assert isinstance(mg["hooks"], list)
        hook_entry = mg["hooks"][0]
        assert hook_entry["type"] == "command"
        assert "truememory" in hook_entry["command"].lower()


def test_install_hooks_preserves_existing_correct_format(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[general]\ntheme = "dark"\n\n'
        '[[hooks.MyCustomHook]]\n\n'
        '[[hooks.MyCustomHook.hooks]]\n'
        'type = "command"\n'
        'command = "my-cmd"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    text = config_path.read_text(encoding="utf-8")
    assert 'theme = "dark"' in text
    assert "[[hooks.MyCustomHook]]" in text
    assert "[[hooks.SessionStart]]" in text


def test_install_hooks_migrates_legacy_format(tmp_path, monkeypatch):
    """Old [[hooks]] entries with event= keys should be cleaned up and replaced."""
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[general]\ntheme = "dark"\n\n'
        '[[hooks]]\n'
        'event = "SessionStart"\n'
        'command = "/usr/bin/python3 /path/to/truememory/session_start.py"\n'
        'timeout = 10000\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    text = config_path.read_text(encoding="utf-8")
    # Legacy format must be gone
    assert "[[hooks]]" not in text
    assert 'event = ' not in text
    # Correct format present
    assert "[[hooks.SessionStart]]" in text
    assert "[[hooks.SessionStart.hooks]]" in text


def test_install_mcp_and_hooks_produces_valid_toml(tmp_path, monkeypatch):
    """End-to-end: MCP + hooks together must be valid TOML."""
    import tomllib

    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")
    adapter.install_hooks(python_path="/usr/bin/python3")

    text = config_path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    assert "mcp_servers" in data
    assert "hooks" in data
    assert "truememory" in data["mcp_servers"]


@pytest.mark.skipif(sys.platform == "win32", reason="TOML path handling differs on Windows")
def test_install_hooks_idempotent(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")
    first_text = config_path.read_text(encoding="utf-8")
    adapter.install_hooks(python_path="/usr/bin/python3")
    second_text = config_path.read_text(encoding="utf-8")
    assert first_text == second_text


# -- MCP + hooks together (both in same file) --

def test_install_mcp_and_hooks_in_same_file(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")
    adapter.install_hooks(python_path="/usr/bin/python3")

    text = config_path.read_text(encoding="utf-8")
    assert "[mcp_servers.truememory]" in text
    assert "[[hooks.SessionStart]]" in text


# -- Uninstall --

def test_uninstall_removes_mcp_and_hooks(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()

    adapter.install_mcp(python_path="/usr/bin/python3")
    adapter.install_hooks(python_path="/usr/bin/python3")
    assert adapter.is_configured()

    adapter.uninstall()
    text = config_path.read_text(encoding="utf-8")
    assert "[mcp_servers.truememory]" not in text
    assert "truememory" not in text.lower()


def test_uninstall_preserves_other_entries(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[general]\ntheme = "dark"\n\n'
        '[[hooks.MyCustomHook]]\n\n'
        '[[hooks.MyCustomHook.hooks]]\n'
        'type = "command"\n'
        'command = "my-cmd"\n'
        'timeout = 5\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")
    adapter.install_hooks(python_path="/usr/bin/python3")
    adapter.uninstall()

    text = config_path.read_text(encoding="utf-8")
    assert 'theme = "dark"' in text
    assert "[[hooks.MyCustomHook]]" in text
    assert "[mcp_servers.truememory]" not in text


def test_uninstall_cleans_legacy_format(tmp_path, monkeypatch):
    """Uninstall must also remove old-format [[hooks]] blocks."""
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[general]\ntheme = "dark"\n\n'
        '[[hooks]]\n'
        'event = "SessionStart"\n'
        'command = "/usr/bin/python3 /path/to/truememory/session_start.py"\n'
        'timeout = 10000\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.uninstall()

    text = config_path.read_text(encoding="utf-8")
    assert 'theme = "dark"' in text
    assert "truememory" not in text.lower()
    assert "[[hooks]]" not in text


# -- MCP section removal (line-by-line TOML parsing) --

def test_remove_mcp_section_boundary():
    from truememory.hooks.adapters.codex import CodexAdapter
    text = (
        '[general]\ntheme = "dark"\n\n'
        '[mcp_servers.truememory]\n'
        'command = "/usr/bin/python3"\n'
        'args = ["-m", "truememory.mcp_server"]\n\n'
        '[mcp_servers.other]\n'
        'command = "other"\n'
    )
    result = CodexAdapter._remove_mcp_section(text)
    assert "[mcp_servers.truememory]" not in result
    assert "[mcp_servers.other]" in result
    assert 'theme = "dark"' in result
    assert 'command = "other"' in result


# -- Hook block removal (correct format) --

def test_remove_hook_blocks_correct_format():
    from truememory.hooks.adapters.codex import CodexAdapter
    text = (
        '[[hooks.SessionStart]]\n\n'
        '[[hooks.SessionStart.hooks]]\n'
        'type = "command"\n'
        'command = "/usr/bin/python3 /path/to/truememory/session_start.py"\n'
        'timeout = 10\n\n'
        '[[hooks.MyCustomHook]]\n\n'
        '[[hooks.MyCustomHook.hooks]]\n'
        'type = "command"\n'
        'command = "my-cmd"\n'
        'timeout = 5\n'
    )
    result = CodexAdapter._remove_hook_blocks(text)
    assert "truememory" not in result.lower()
    assert "[[hooks.MyCustomHook]]" in result
    assert 'command = "my-cmd"' in result


# -- Legacy hook block removal --

def test_remove_legacy_hook_blocks_with_blank_lines():
    """Old adapter had a bug where blank lines within a block would terminate collection."""
    from truememory.hooks.adapters.codex import CodexAdapter
    text = (
        '[[hooks]]\n'
        'event = "SessionStart"\n'
        '\n'
        'command = "/usr/bin/python3 /path/to/truememory/session_start.py"\n'
        'timeout = 10000\n\n'
        '[[hooks]]\n'
        'event = "MyCustomHook"\n'
        'command = "my-cmd"\n'
    )
    result = CodexAdapter._remove_legacy_hook_blocks(text)
    assert "truememory" not in result.lower()
    assert 'event = "MyCustomHook"' in result
    assert 'command = "my-cmd"' in result


# -- TOML escape helper --

def test_toml_escape():
    from truememory.hooks.adapters.codex import _toml_escape
    assert _toml_escape(r'C:\Users\test') == 'C:\\\\Users\\\\test'
    assert _toml_escape('say "hello"') == 'say \\"hello\\"'
    assert _toml_escape('/usr/bin/python3') == '/usr/bin/python3'


# -- _parse_existing_hooks --

def test_parse_existing_hooks_correct_format():
    from truememory.hooks.adapters.codex import CodexAdapter
    text = (
        '[[hooks.SessionStart]]\n\n'
        '[[hooks.SessionStart.hooks]]\n'
        'type = "command"\n'
        'command = "/usr/bin/python3 /path/to/truememory/session_start.py"\n'
        'timeout = 10\n\n'
        '[[hooks.Stop]]\n\n'
        '[[hooks.Stop.hooks]]\n'
        'type = "command"\n'
        'command = "/usr/bin/python3 /path/to/truememory/stop.py"\n'
        'timeout = 5\n'
    )
    result = CodexAdapter._parse_existing_hooks(text)
    assert isinstance(result, dict)
    assert "SessionStart" in result
    assert "Stop" in result
    mg = result["SessionStart"][0]
    assert "hooks" in mg
    assert mg["hooks"][0]["command"] == "/usr/bin/python3 /path/to/truememory/session_start.py"


def test_parse_existing_hooks_empty():
    from truememory.hooks.adapters.codex import CodexAdapter
    result = CodexAdapter._parse_existing_hooks("")
    assert result == {}


# -- _event_already_registered --

def test_event_already_registered_correct_format():
    from truememory.hooks.adapters.codex import CodexAdapter
    hooks = {
        "SessionStart": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "/usr/bin/python3 /path/to/truememory/session_start.py",
                        "timeout": 10,
                    }
                ]
            }
        ]
    }
    assert CodexAdapter._event_already_registered(hooks, "SessionStart")
    assert not CodexAdapter._event_already_registered(hooks, "Stop")


def test_event_already_registered_no_truememory():
    from truememory.hooks.adapters.codex import CodexAdapter
    hooks = {
        "SessionStart": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "/usr/bin/python3 /path/to/other/hook.py",
                        "timeout": 10,
                    }
                ]
            }
        ]
    }
    assert not CodexAdapter._event_already_registered(hooks, "SessionStart")


# -- is_configured --

def test_is_configured_false_clean(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", tmp_path / "config.toml")
    from truememory.hooks.adapters.codex import CodexAdapter
    assert not CodexAdapter().is_configured()


# -- verify --

@pytest.mark.skipif(sys.platform == "win32", reason="TOML path handling differs on Windows")
def test_verify_requires_both(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()

    adapter.install_mcp(python_path="/usr/bin/python3")
    assert not adapter.verify()

    adapter.install_hooks(python_path="/usr/bin/python3")
    assert adapter.verify()


# -- AGENTS.md system prompt --

def test_system_prompt_path():
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    path = adapter.get_system_prompt_path()
    assert path is not None
    assert path.name == "AGENTS.md"


def test_system_prompt_content():
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    content = adapter.get_system_prompt_content()
    assert len(content) > 0
    assert "TrueMemory" in content
    assert "truememory_search" in content


# -- Build command --

def test_build_command_with_user_and_db():
    from truememory.hooks.adapters.codex import CodexAdapter
    cmd = CodexAdapter._build_command(
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
