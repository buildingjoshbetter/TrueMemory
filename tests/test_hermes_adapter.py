"""Tests for the Hermes Agent adapter (#185).

Validates YAML MCP config, plugin hook registration, detection,
and config merge safety without network calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
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
    mcp_path = tmp_path / "config.yaml"
    monkeypatch.setattr(hermes_mod, "_MCP_CONFIG", mcp_path)
    from truememory.hooks.adapters.hermes import HermesAdapter
    HermesAdapter().install_mcp(python_path="/usr/bin/python3")

    data = yaml.safe_load(mcp_path.read_text(encoding="utf-8"))
    assert "truememory" in data["mcp_servers"]
    assert data["mcp_servers"]["truememory"]["command"] == "/usr/bin/python3"
    assert data["mcp_servers"]["truememory"]["args"] == ["-m", "truememory.mcp_server"]


def test_install_mcp_preserves_existing(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    mcp_path = tmp_path / "config.yaml"
    mcp_path.write_text(yaml.safe_dump({
        "mcp_servers": {"other-server": {"command": "other"}},
        "general": {"theme": "dark"},
    }), encoding="utf-8")
    monkeypatch.setattr(hermes_mod, "_MCP_CONFIG", mcp_path)
    from truememory.hooks.adapters.hermes import HermesAdapter
    HermesAdapter().install_mcp(python_path="/usr/bin/python3")

    data = yaml.safe_load(mcp_path.read_text(encoding="utf-8"))
    assert "truememory" in data["mcp_servers"]
    assert "other-server" in data["mcp_servers"]
    assert data["general"]["theme"] == "dark"


# -- Plugin hooks --

def test_install_hooks_creates_cli_config(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    cli_config = tmp_path / "cli-config.yaml"
    monkeypatch.setattr(hermes_mod, "_CLI_CONFIG", cli_config)
    from truememory.hooks.adapters.hermes import HermesAdapter
    HermesAdapter().install_hooks(python_path="/usr/bin/python3")

    data = yaml.safe_load(cli_config.read_text(encoding="utf-8"))
    plugins = data["plugins"]
    assert len(plugins) == 2

    names = {p["name"] for p in plugins}
    assert "truememory-session-start" in names
    assert "truememory-session-end" in names

    events = {p["event"] for p in plugins}
    assert "on_session_start" in events
    assert "on_session_end" in events


def test_install_hooks_preserves_existing(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    cli_config = tmp_path / "cli-config.yaml"
    cli_config.write_text(yaml.safe_dump({
        "plugins": [{"name": "my-plugin", "event": "custom", "command": "my-cmd"}],
    }), encoding="utf-8")
    monkeypatch.setattr(hermes_mod, "_CLI_CONFIG", cli_config)
    from truememory.hooks.adapters.hermes import HermesAdapter
    HermesAdapter().install_hooks(python_path="/usr/bin/python3")

    data = yaml.safe_load(cli_config.read_text(encoding="utf-8"))
    names = {p["name"] for p in data["plugins"]}
    assert "my-plugin" in names
    assert "truememory-session-start" in names


def test_install_hooks_idempotent(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    cli_config = tmp_path / "cli-config.yaml"
    monkeypatch.setattr(hermes_mod, "_CLI_CONFIG", cli_config)
    from truememory.hooks.adapters.hermes import HermesAdapter
    adapter = HermesAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")
    first = cli_config.read_text(encoding="utf-8")
    adapter.install_hooks(python_path="/usr/bin/python3")
    second = cli_config.read_text(encoding="utf-8")
    assert first == second


# -- Uninstall --

def test_uninstall_removes_entries(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    mcp_path = tmp_path / "config.yaml"
    cli_config = tmp_path / "cli-config.yaml"
    monkeypatch.setattr(hermes_mod, "_MCP_CONFIG", mcp_path)
    monkeypatch.setattr(hermes_mod, "_CLI_CONFIG", cli_config)
    from truememory.hooks.adapters.hermes import HermesAdapter
    adapter = HermesAdapter()

    adapter.install_mcp(python_path="/usr/bin/python3")
    adapter.install_hooks(python_path="/usr/bin/python3")
    assert adapter.is_configured()

    adapter.uninstall()
    mcp_data = yaml.safe_load(mcp_path.read_text(encoding="utf-8"))
    assert "truememory" not in mcp_data.get("mcp_servers", {})

    cli_data = yaml.safe_load(cli_config.read_text(encoding="utf-8"))
    tm_plugins = [
        p for p in cli_data.get("plugins", [])
        if "truememory" in p.get("name", "").lower()
    ]
    assert len(tm_plugins) == 0


# -- is_configured --

def test_is_configured_false_clean(tmp_path, monkeypatch):
    from truememory.hooks.adapters import hermes as hermes_mod
    monkeypatch.setattr(hermes_mod, "_MCP_CONFIG", tmp_path / "config.yaml")
    monkeypatch.setattr(hermes_mod, "_CLI_CONFIG", tmp_path / "cli-config.yaml")
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
