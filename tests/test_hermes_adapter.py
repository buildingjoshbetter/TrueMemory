"""Tests for the Hermes Agent adapter (#185).

Validates YAML MCP config, shell hook registration (hooks: key in
config.yaml), detection, and config merge safety without network calls.

Hermes uses a single config.yaml for both mcp_servers and hooks.
Hooks are list-of-dicts under each event name with command/matcher/timeout.
"""
from __future__ import annotations

from pathlib import Path

import yaml


# -- Import tests --

def test_import_hermes_adapter():
    from truememory.hooks.adapters.hermes import HermesAdapter  # noqa: F401


def test_hermes_in_registry():
    from truememory.hooks.registry import get_adapter
    adapter = get_adapter("hermes")
    assert adapter is not None
    assert adapter.cli_id == "hermes"


# -- Instantiation --

def test_hermes_adapter_properties():
    from truememory.hooks.adapters.hermes import HermesAdapter
    adapter = HermesAdapter()
    assert adapter.name == "Hermes Agent"
    assert adapter.cli_id == "hermes"
    assert isinstance(adapter.config_path, Path)
    assert adapter.config_path.name == "config.yaml"


def test_hermes_implements_all_abstract_methods():
    from truememory.hooks.adapters.base import CLIAdapter
    from truememory.hooks.adapters.hermes import HermesAdapter
    import inspect
    abstract_methods = {
        name for name, _ in inspect.getmembers(CLIAdapter)
        if getattr(getattr(CLIAdapter, name, None), "__isabstractmethod__", False)
    }
    adapter = HermesAdapter()
    for method_name in abstract_methods:
        assert hasattr(adapter, method_name), f"Missing: {method_name}"


# -- Detection --

def test_detect_false_no_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    monkeypatch.setattr(hermes_mod, "_HERMES_DIR", tmp_path / "nonexistent")
    from truememory.hooks.adapters.hermes import HermesAdapter
    adapter = HermesAdapter()
    monkeypatch.setattr("shutil.which", lambda x: None)
    assert not adapter.detect()


def test_detect_true_with_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir()
    monkeypatch.setattr(hermes_mod, "_HERMES_DIR", hermes_dir)
    from truememory.hooks.adapters.hermes import HermesAdapter
    assert HermesAdapter().detect()


# -- MCP config --

def test_install_mcp_creates_yaml(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(hermes_mod, "_CONFIG", config_path)
    from truememory.hooks.adapters.hermes import HermesAdapter
    HermesAdapter().install_mcp(python_path="/usr/bin/python3")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "truememory" in data["mcp_servers"]
    assert data["mcp_servers"]["truememory"]["command"] == "/usr/bin/python3"
    assert data["mcp_servers"]["truememory"]["args"] == ["-m", "truememory.mcp_server"]


def test_install_mcp_preserves_existing(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "mcp_servers": {"other-server": {"command": "other"}},
        "general": {"theme": "dark"},
    }), encoding="utf-8")
    monkeypatch.setattr(hermes_mod, "_CONFIG", config_path)
    from truememory.hooks.adapters.hermes import HermesAdapter
    HermesAdapter().install_mcp(python_path="/usr/bin/python3")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "truememory" in data["mcp_servers"]
    assert "other-server" in data["mcp_servers"]
    assert data["general"]["theme"] == "dark"


# -- Shell hooks (hooks: key in config.yaml) --

def test_install_hooks_creates_hooks_config(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(hermes_mod, "_CONFIG", config_path)
    from truememory.hooks.adapters.hermes import HermesAdapter
    HermesAdapter().install_hooks(python_path="/usr/bin/python3")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hooks = data["hooks"]

    # Should have 4 events registered
    assert len(hooks) == 4

    # Verify correct Hermes event names (from VALID_HOOKS)
    assert "on_session_start" in hooks
    assert "on_session_end" in hooks
    assert "pre_llm_call" in hooks
    assert "on_session_finalize" in hooks

    # Each event should map to a list of entries
    for event, entries in hooks.items():
        assert isinstance(entries, list), f"{event} should be a list"
        assert len(entries) >= 1
        entry = entries[0]
        assert isinstance(entry, dict)
        assert "command" in entry  # command is required
        assert "truememory" in entry["command"].lower()

    # Verify timeouts are set (in seconds, as per Hermes source)
    assert hooks["on_session_start"][0]["timeout"] == 30
    assert hooks["on_session_end"][0]["timeout"] == 30
    assert hooks["pre_llm_call"][0]["timeout"] == 10
    assert hooks["on_session_finalize"][0]["timeout"] == 15

    # Verify no invalid events used
    invalid_events = {"on_user_prompt", "on_pre_compact"}
    for event in invalid_events:
        assert event not in hooks, f"Invalid event {event} found"


def test_install_hooks_preserves_existing(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "hooks": {
            "pre_tool_call": [
                {"matcher": "terminal", "command": "my-guard.sh", "timeout": 5}
            ],
        },
        "mcp_servers": {"other": {"command": "x"}},
    }), encoding="utf-8")
    monkeypatch.setattr(hermes_mod, "_CONFIG", config_path)
    from truememory.hooks.adapters.hermes import HermesAdapter
    HermesAdapter().install_hooks(python_path="/usr/bin/python3")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Existing pre_tool_call hook should be preserved
    assert len(data["hooks"]["pre_tool_call"]) == 1
    assert data["hooks"]["pre_tool_call"][0]["command"] == "my-guard.sh"
    # TrueMemory hooks should be added
    assert "on_session_start" in data["hooks"]
    # Other config keys preserved
    assert "other" in data["mcp_servers"]


def test_install_hooks_idempotent(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(hermes_mod, "_CONFIG", config_path)
    from truememory.hooks.adapters.hermes import HermesAdapter
    adapter = HermesAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")
    first = config_path.read_text(encoding="utf-8")
    adapter.install_hooks(python_path="/usr/bin/python3")
    second = config_path.read_text(encoding="utf-8")
    assert first == second


def test_install_hooks_uses_single_config_file(tmp_path, monkeypatch):
    """Hooks and MCP should live in the same config.yaml file."""
    from truememory.hooks.adapters import hermes as hermes_mod
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(hermes_mod, "_CONFIG", config_path)
    from truememory.hooks.adapters.hermes import HermesAdapter
    adapter = HermesAdapter()
    adapter.install_mcp(python_path="/usr/bin/python3")
    adapter.install_hooks(python_path="/usr/bin/python3")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Both mcp_servers and hooks should be in the same file
    assert "mcp_servers" in data
    assert "hooks" in data
    assert "truememory" in data["mcp_servers"]
    assert "on_session_start" in data["hooks"]


# -- Uninstall --

def test_uninstall_removes_entries(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(hermes_mod, "_CONFIG", config_path)
    from truememory.hooks.adapters.hermes import HermesAdapter
    adapter = HermesAdapter()

    adapter.install_mcp(python_path="/usr/bin/python3")
    adapter.install_hooks(python_path="/usr/bin/python3")
    assert adapter.is_configured()

    adapter.uninstall()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "truememory" not in data.get("mcp_servers", {})

    # All truememory hook entries should be removed
    hooks = data.get("hooks", {})
    for event, entries in hooks.items():
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    assert "truememory" not in entry.get("command", "").lower()


# -- is_configured --

def test_is_configured_false_clean(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    monkeypatch.setattr(hermes_mod, "_CONFIG", tmp_path / "config.yaml")
    from truememory.hooks.adapters.hermes import HermesAdapter
    assert not HermesAdapter().is_configured()


# -- Build command --

def test_build_command_with_args():
    from truememory.hooks.adapters.hermes import HermesAdapter
    cmd = HermesAdapter._build_command(
        "/usr/bin/python3",
        Path("/path/to/session_start.py"),
        user_id="bob",
        db_path="/data/mem.db",
    )
    assert "/usr/bin/python3" in cmd
    assert "session_start.py" in cmd
    assert "--user" in cmd
    assert "bob" in cmd
