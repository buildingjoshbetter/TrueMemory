"""Tests for the Gemini CLI adapter (#233).

Validates MCP config, JSON hook registration, detection, and
config merge safety without network calls.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path


# -- Import tests --

def test_import_gemini_adapter():
    from truememory.hooks.adapters.gemini import GeminiAdapter  # noqa: F401


def test_gemini_in_registry():
    from truememory.hooks.registry import get_adapter
    adapter = get_adapter("gemini")
    assert adapter is not None
    assert adapter.cli_id == "gemini"


# -- Instantiation --

def test_gemini_adapter_properties():
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    assert adapter.name == "Gemini CLI"
    assert adapter.cli_id == "gemini"
    assert isinstance(adapter.config_path, Path)
    assert adapter.config_path.name == "settings.json"


def test_gemini_implements_all_abstract_methods():
    from truememory.hooks.adapters.base import CLIAdapter
    from truememory.hooks.adapters.gemini import GeminiAdapter
    abstract_methods = {
        name for name, _ in inspect.getmembers(CLIAdapter)
        if getattr(getattr(CLIAdapter, name, None), "__isabstractmethod__", False)
    }
    adapter = GeminiAdapter()
    for method_name in abstract_methods:
        assert hasattr(adapter, method_name), f"Missing: {method_name}"


# -- Detection --

def test_detect_false_no_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    monkeypatch.setattr(gemini_mod, "_GEMINI_DIR", tmp_path / "nonexistent")
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    monkeypatch.setattr("shutil.which", lambda x: None)
    assert not adapter.detect()


def test_detect_true_with_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir()
    monkeypatch.setattr(gemini_mod, "_GEMINI_DIR", gemini_dir)
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    assert adapter.detect()


# -- MCP config --

def test_install_mcp_creates_config(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    config_path = tmp_path / "settings.json"
    monkeypatch.setattr(gemini_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "truememory" in data["mcpServers"]
    assert data["mcpServers"]["truememory"]["command"] == "/usr/bin/python3"
    assert data["mcpServers"]["truememory"]["args"] == ["-m", "truememory.mcp_server"]


def test_install_mcp_preserves_existing(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps({
        "mcpServers": {
            "other-server": {"command": "other", "args": ["arg"]}
        },
        "theme": "dark",
    }), encoding="utf-8")
    monkeypatch.setattr(gemini_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "truememory" in data["mcpServers"]
    assert "other-server" in data["mcpServers"]
    assert data["theme"] == "dark"


def test_install_mcp_idempotent(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    config_path = tmp_path / "settings.json"
    monkeypatch.setattr(gemini_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")
    first = config_path.read_text(encoding="utf-8")
    adapter.install_mcp(python_path="/usr/bin/python3")
    second = config_path.read_text(encoding="utf-8")
    assert first == second


# -- Hook config --

def test_install_hooks_creates_entries(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    config_path = tmp_path / "settings.json"
    monkeypatch.setattr(gemini_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    hooks = data["hooks"]
    assert "SessionStart" in hooks
    assert "SessionEnd" in hooks
    assert "PreCompress" in hooks
    assert len(hooks["SessionStart"]) == 1
    assert "truememory" in hooks["SessionStart"][0]["command"].lower()


def test_install_hooks_preserves_existing(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps({
        "hooks": {
            "SessionStart": [
                {"command": "my-custom-hook", "timeout": 5000}
            ]
        },
        "theme": "dark",
    }), encoding="utf-8")
    monkeypatch.setattr(gemini_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["theme"] == "dark"
    assert len(data["hooks"]["SessionStart"]) == 2
    assert data["hooks"]["SessionStart"][0]["command"] == "my-custom-hook"


def test_install_hooks_idempotent(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    config_path = tmp_path / "settings.json"
    monkeypatch.setattr(gemini_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")
    first = config_path.read_text(encoding="utf-8")
    adapter.install_hooks(python_path="/usr/bin/python3")
    second = config_path.read_text(encoding="utf-8")
    assert first == second


# -- Uninstall --

def test_uninstall_removes_entries(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    config_path = tmp_path / "settings.json"
    monkeypatch.setattr(gemini_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()

    adapter.install_mcp(python_path="/usr/bin/python3")
    adapter.install_hooks(python_path="/usr/bin/python3")
    assert adapter.is_configured()

    adapter.uninstall()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "truememory" not in data.get("mcpServers", {})
    hooks = data.get("hooks", {})
    for entries in hooks.values():
        for h in entries:
            assert "truememory" not in h.get("command", "").lower()


def test_uninstall_preserves_other_entries(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps({
        "mcpServers": {
            "other-server": {"command": "other"},
            "truememory": {"command": "python", "args": ["-m", "truememory.mcp_server"]},
        },
        "hooks": {
            "SessionStart": [
                {"command": "my-hook", "timeout": 5000},
                {"command": "/path/to/truememory/hooks/session_start.py", "timeout": 10000},
            ]
        },
        "theme": "dark",
    }), encoding="utf-8")
    monkeypatch.setattr(gemini_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    adapter.uninstall()

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "other-server" in data["mcpServers"]
    assert "truememory" not in data["mcpServers"]
    assert data["theme"] == "dark"
    assert len(data["hooks"]["SessionStart"]) == 1
    assert data["hooks"]["SessionStart"][0]["command"] == "my-hook"


# -- is_configured --

def test_is_configured_false_clean(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    monkeypatch.setattr(gemini_mod, "_CONFIG_PATH", tmp_path / "settings.json")
    from truememory.hooks.adapters.gemini import GeminiAdapter
    assert not GeminiAdapter().is_configured()


def test_is_configured_true_after_install(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    config_path = tmp_path / "settings.json"
    monkeypatch.setattr(gemini_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")
    assert adapter.is_configured()


# -- verify --

def test_verify_requires_both(tmp_path, monkeypatch):
    from truememory.hooks.adapters import gemini as gemini_mod
    config_path = tmp_path / "settings.json"
    monkeypatch.setattr(gemini_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")
    assert not adapter.verify()
    adapter.install_hooks(python_path="/usr/bin/python3")
    assert adapter.verify()


# -- System prompt --

def test_system_prompt_path():
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    path = adapter.get_system_prompt_path()
    assert path is not None
    assert path.name == "GEMINI.md"


def test_system_prompt_content():
    from truememory.hooks.adapters.gemini import GeminiAdapter
    adapter = GeminiAdapter()
    content = adapter.get_system_prompt_content()
    assert "TrueMemory" in content
    assert "truememory_search" in content


# -- Build command --

def test_build_command_with_user_and_db():
    from truememory.hooks.adapters.gemini import GeminiAdapter
    cmd = GeminiAdapter._build_command(
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
