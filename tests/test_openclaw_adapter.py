"""Tests for the OpenClaw adapter (#184).

Validates JSON config merge for MCP (mcp.servers key), JS plugin installation,
detection, and config safety without network calls.
"""
from __future__ import annotations

import json


# -- Import tests --

def test_import_openclaw_adapter():
    from truememory.hooks.adapters.openclaw import OpenClawAdapter  # noqa: F401


def test_openclaw_in_registry():
    from truememory.hooks.registry import get_adapter
    adapter = get_adapter("openclaw")
    assert adapter is not None
    assert adapter.cli_id == "openclaw"


# -- Instantiation --

def test_openclaw_adapter_properties():
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    adapter = OpenClawAdapter()
    assert adapter.name == "OpenClaw"
    assert adapter.cli_id == "openclaw"
    assert adapter.config_path.name == "openclaw.json"


def test_openclaw_implements_all_abstract_methods():
    from truememory.hooks.adapters.base import CLIAdapter
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    import inspect
    abstract_methods = {
        name for name, _ in inspect.getmembers(CLIAdapter)
        if getattr(getattr(CLIAdapter, name, None), "__isabstractmethod__", False)
    }
    adapter = OpenClawAdapter()
    for method_name in abstract_methods:
        assert hasattr(adapter, method_name), f"Missing: {method_name}"


# -- Detection --

def test_detect_false_no_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import openclaw as oc_mod
    monkeypatch.setattr(oc_mod, "_OPENCLAW_DIR", tmp_path / "nonexistent")
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    monkeypatch.setattr("shutil.which", lambda x: None)
    assert not OpenClawAdapter().detect()


def test_detect_true_with_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import openclaw as oc_mod
    oc_dir = tmp_path / ".openclaw"
    oc_dir.mkdir()
    monkeypatch.setattr(oc_mod, "_OPENCLAW_DIR", oc_dir)
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    assert OpenClawAdapter().detect()


# -- MCP config --

def test_install_mcp_creates_config(tmp_path, monkeypatch):
    from truememory.hooks.adapters import openclaw as oc_mod
    config_path = tmp_path / "openclaw.json"
    monkeypatch.setattr(oc_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    OpenClawAdapter().install_mcp(python_path="/usr/bin/python3")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    # Should use nested mcp.servers, not top-level mcpServers
    assert "mcp" in data
    assert "servers" in data["mcp"]
    assert "truememory" in data["mcp"]["servers"]
    assert data["mcp"]["servers"]["truememory"]["command"] == "/usr/bin/python3"


def test_install_mcp_preserves_existing(tmp_path, monkeypatch):
    from truememory.hooks.adapters import openclaw as oc_mod
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(json.dumps({
        "mcp": {"servers": {"other": {"command": "x"}}},
        "settings": {"debug": True},
    }), encoding="utf-8")
    monkeypatch.setattr(oc_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    OpenClawAdapter().install_mcp(python_path="/usr/bin/python3")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "truememory" in data["mcp"]["servers"]
    assert "other" in data["mcp"]["servers"]
    assert data["settings"]["debug"] is True


def test_install_mcp_handles_json5_comments(tmp_path, monkeypatch):
    from truememory.hooks.adapters import openclaw as oc_mod
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(
        '// OpenClaw config\n{\n  "mcp": {"servers": {}}\n}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(oc_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    OpenClawAdapter().install_mcp(python_path="/usr/bin/python3")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "truememory" in data["mcp"]["servers"]


def test_install_mcp_migrates_legacy_top_level(tmp_path, monkeypatch):
    """If old top-level mcpServers entry exists, it should be cleaned up."""
    from truememory.hooks.adapters import openclaw as oc_mod
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(json.dumps({
        "mcpServers": {"truememory": {"command": "old"}},
    }), encoding="utf-8")
    monkeypatch.setattr(oc_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    OpenClawAdapter().install_mcp(python_path="/usr/bin/python3")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    # New correct location should exist
    assert "truememory" in data["mcp"]["servers"]
    # Old top-level mcpServers should be cleaned up
    assert "mcpServers" not in data or "truememory" not in data.get("mcpServers", {})


def test_has_mcp_detects_legacy_top_level(tmp_path, monkeypatch):
    """Detection should find entries at the old top-level mcpServers path too."""
    from truememory.hooks.adapters import openclaw as oc_mod
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(json.dumps({
        "mcpServers": {"truememory": {"command": "old"}},
    }), encoding="utf-8")
    monkeypatch.setattr(oc_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    assert OpenClawAdapter()._has_mcp_entry()


def test_has_mcp_detects_correct_path(tmp_path, monkeypatch):
    """Detection should find entries at the correct mcp.servers path."""
    from truememory.hooks.adapters import openclaw as oc_mod
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(json.dumps({
        "mcp": {"servers": {"truememory": {"command": "python3"}}},
    }), encoding="utf-8")
    monkeypatch.setattr(oc_mod, "_CONFIG_PATH", config_path)
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    assert OpenClawAdapter()._has_mcp_entry()


# -- Plugin install --

def test_install_hooks_creates_plugin(tmp_path, monkeypatch):
    from truememory.hooks.adapters import openclaw as oc_mod
    plugins_dir = tmp_path / "plugins"
    monkeypatch.setattr(oc_mod, "_PLUGINS_DIR", plugins_dir)
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    OpenClawAdapter().install_hooks(python_path="/usr/bin/python3")

    plugin_dir = plugins_dir / "truememory"
    assert plugin_dir.is_dir()
    assert (plugin_dir / "openclaw.plugin.json").exists()
    assert (plugin_dir / "package.json").exists()
    assert (plugin_dir / "index.js").exists()

    manifest = json.loads((plugin_dir / "openclaw.plugin.json").read_text(encoding="utf-8"))
    assert manifest["id"] == "truememory"
    assert manifest["name"] == "TrueMemory"
    # Uses activation.onStartup (not activationEvents)
    assert manifest["activation"]["onStartup"] is True
    # Must have configSchema (required by OpenClaw)
    assert "configSchema" in manifest
    # Should NOT have fabricated fields
    assert "activationEvents" not in manifest
    assert "main" not in manifest


def test_install_hooks_uses_register_api(tmp_path, monkeypatch):
    """index.js should use register(api) with api.on(), not activate(ctx)."""
    from truememory.hooks.adapters import openclaw as oc_mod
    plugins_dir = tmp_path / "plugins"
    monkeypatch.setattr(oc_mod, "_PLUGINS_DIR", plugins_dir)
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    OpenClawAdapter().install_hooks(python_path="/usr/bin/python3")

    js_content = (plugins_dir / "truememory" / "index.js").read_text(encoding="utf-8")
    # Should use register(api) pattern
    assert "register(api)" in js_content
    assert 'api.on("session_start"' in js_content
    assert 'api.on("session_end"' in js_content
    assert 'api.on("before_tool_call"' in js_content
    assert 'api.on("before_compaction"' in js_content
    # Should NOT use old fabricated activate(ctx) pattern
    assert "activate(ctx)" not in js_content
    assert "ctx.onSessionStart" not in js_content
    assert "ctx.onSessionEnd" not in js_content
    # Should use ESM imports, not CommonJS require
    assert "require(" not in js_content
    assert "module.exports" not in js_content


def test_install_hooks_package_json_esm(tmp_path, monkeypatch):
    """package.json should declare ESM module type and openclaw extensions."""
    from truememory.hooks.adapters import openclaw as oc_mod
    plugins_dir = tmp_path / "plugins"
    monkeypatch.setattr(oc_mod, "_PLUGINS_DIR", plugins_dir)
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    OpenClawAdapter().install_hooks(python_path="/usr/bin/python3")

    pkg = json.loads((plugins_dir / "truememory" / "package.json").read_text(encoding="utf-8"))
    assert pkg["type"] == "module"
    assert "openclaw" in pkg
    assert "./index.js" in pkg["openclaw"]["extensions"]


def test_install_hooks_idempotent(tmp_path, monkeypatch):
    from truememory.hooks.adapters import openclaw as oc_mod
    plugins_dir = tmp_path / "plugins"
    monkeypatch.setattr(oc_mod, "_PLUGINS_DIR", plugins_dir)
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    adapter = OpenClawAdapter()
    adapter.install_hooks(python_path="/usr/bin/python3")
    first_js = (plugins_dir / "truememory" / "index.js").read_text(encoding="utf-8")
    adapter.install_hooks(python_path="/usr/bin/python3")
    second_js = (plugins_dir / "truememory" / "index.js").read_text(encoding="utf-8")
    assert first_js == second_js


# -- Uninstall --

def test_uninstall_removes_entries(tmp_path, monkeypatch):
    from truememory.hooks.adapters import openclaw as oc_mod
    config_path = tmp_path / "openclaw.json"
    plugins_dir = tmp_path / "plugins"
    monkeypatch.setattr(oc_mod, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(oc_mod, "_PLUGINS_DIR", plugins_dir)
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    adapter = OpenClawAdapter()

    adapter.install_mcp(python_path="/usr/bin/python3")
    adapter.install_hooks(python_path="/usr/bin/python3")
    assert adapter.is_configured()

    adapter.uninstall()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "truememory" not in data.get("mcp", {}).get("servers", {})
    assert not (plugins_dir / "truememory").exists()


def test_uninstall_cleans_legacy_top_level(tmp_path, monkeypatch):
    """Uninstall should also remove entries from legacy top-level mcpServers."""
    from truememory.hooks.adapters import openclaw as oc_mod
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(json.dumps({
        "mcpServers": {"truememory": {"command": "old"}},
        "mcp": {"servers": {"truememory": {"command": "new"}}},
    }), encoding="utf-8")
    monkeypatch.setattr(oc_mod, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(oc_mod, "_PLUGINS_DIR", tmp_path / "plugins")
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    OpenClawAdapter().uninstall()

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "truememory" not in data.get("mcpServers", {})
    assert "truememory" not in data.get("mcp", {}).get("servers", {})


# -- is_configured --

def test_is_configured_false_clean(tmp_path, monkeypatch):
    from truememory.hooks.adapters import openclaw as oc_mod
    monkeypatch.setattr(oc_mod, "_CONFIG_PATH", tmp_path / "openclaw.json")
    monkeypatch.setattr(oc_mod, "_PLUGINS_DIR", tmp_path / "plugins")
    from truememory.hooks.adapters.openclaw import OpenClawAdapter
    assert not OpenClawAdapter().is_configured()


# -- JSON5 comment stripping --

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
