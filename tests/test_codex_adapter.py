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

def test_install_hooks_creates_toml(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    text = config_path.read_text(encoding="utf-8")
    assert "[[hooks]]" in text
    assert 'event = "SessionStart"' in text
    assert 'event = "Stop"' in text
    assert 'event = "UserPromptSubmit"' in text
    assert "truememory" in text.lower()
    assert 'event = "PreCompact"' in text


def test_install_hooks_preserves_existing_toml(tmp_path, monkeypatch):
    from truememory.hooks.adapters import codex as codex_mod
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[general]\ntheme = "dark"\n\n[[hooks]]\nevent = "MyCustomHook"\ncommand = "my-cmd"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(codex_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.codex import CodexAdapter
    adapter = CodexAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    text = config_path.read_text(encoding="utf-8")
    assert 'theme = "dark"' in text
    assert 'event = "MyCustomHook"' in text
    assert 'event = "SessionStart"' in text


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
    assert "[[hooks]]" in text
    assert 'event = "SessionStart"' in text


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
        '[[hooks]]\nevent = "MyCustomHook"\ncommand = "my-cmd"\ntimeout = 5000\n',
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
    assert 'event = "MyCustomHook"' in text
    assert "[mcp_servers.truememory]" not in text


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
