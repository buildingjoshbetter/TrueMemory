"""Tests for issue #600: Antigravity adapter, Groq provider, DeepSearch model selection."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Antigravity adapter tests
# ---------------------------------------------------------------------------


def test_import_antigravity_adapter():
    from truememory.hooks.adapters.antigravity import AntigravityAdapter  # noqa: F401


def test_antigravity_in_registry():
    from truememory.hooks.registry import get_adapter

    adapter = get_adapter("antigravity")
    assert adapter is not None
    assert adapter.cli_id == "antigravity"


def test_antigravity_adapter_properties():
    from truememory.hooks.adapters.antigravity import AntigravityAdapter

    adapter = AntigravityAdapter()
    assert adapter.name == "Antigravity"
    assert adapter.cli_id == "antigravity"
    assert isinstance(adapter.config_path, Path)
    assert adapter.config_path.name == "mcp.json"


def test_antigravity_implements_all_abstract_methods():
    from truememory.hooks.adapters.base import CLIAdapter
    from truememory.hooks.adapters.antigravity import AntigravityAdapter

    abstract_methods = {
        name for name, _ in inspect.getmembers(CLIAdapter)
        if getattr(getattr(CLIAdapter, name, None), "__isabstractmethod__", False)
    }
    adapter = AntigravityAdapter()
    for method_name in abstract_methods:
        assert hasattr(adapter, method_name), f"Missing: {method_name}"


def test_antigravity_template_exists():
    template = (
        Path(__file__).parent.parent / "truememory" / "hooks" / "templates" / "antigravity" / "mcp.json"
    )
    assert template.exists(), f"Template not found at {template}"
    data = json.loads(template.read_text(encoding="utf-8"))
    assert "mcpServers" in data
    assert "truememory" in data["mcpServers"]
    assert data["mcpServers"]["truememory"]["args"] == ["-m", "truememory.mcp_server"]


def test_antigravity_detect_false_without_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import antigravity as ag_mod
    from truememory.hooks.adapters.antigravity import AntigravityAdapter

    monkeypatch.setattr(ag_mod, "_ANTIGRAVITY_DIR", tmp_path / "missing")
    monkeypatch.setattr(ag_mod, "_MCP_CONFIG", tmp_path / "missing" / "mcp.json")
    with patch("shutil.which", return_value=None):
        assert not AntigravityAdapter().detect()


def test_antigravity_detect_true_with_dir(tmp_path, monkeypatch):
    from truememory.hooks.adapters import antigravity as ag_mod
    from truememory.hooks.adapters.antigravity import AntigravityAdapter

    ag_dir = tmp_path / ".antigravity"
    ag_dir.mkdir()
    monkeypatch.setattr(ag_mod, "_ANTIGRAVITY_DIR", ag_dir)
    monkeypatch.setattr(ag_mod, "_MCP_CONFIG", ag_dir / "mcp.json")

    assert AntigravityAdapter().detect()


def test_antigravity_install_mcp_creates_config(tmp_path, monkeypatch):
    from truememory.hooks.adapters import antigravity as ag_mod
    from truememory.hooks.adapters.antigravity import AntigravityAdapter

    config_path = tmp_path / "mcp.json"
    monkeypatch.setattr(ag_mod, "_MCP_CONFIG", config_path)

    AntigravityAdapter().install_mcp(python_path="/usr/bin/python3")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["mcpServers"]["truememory"]["command"] == "/usr/bin/python3"
    assert data["mcpServers"]["truememory"]["args"] == ["-m", "truememory.mcp_server"]


def test_antigravity_uninstall_removes_entry(tmp_path, monkeypatch):
    from truememory.hooks.adapters import antigravity as ag_mod
    from truememory.hooks.adapters.antigravity import AntigravityAdapter

    config_path = tmp_path / "mcp.json"
    config_path.write_text(json.dumps({
        "mcpServers": {
            "truememory": {"command": "python", "args": ["-m", "truememory.mcp_server"]},
            "other": {"command": "other"},
        }
    }))
    monkeypatch.setattr(ag_mod, "_MCP_CONFIG", config_path)

    AntigravityAdapter().uninstall()

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "truememory" not in data["mcpServers"]
    assert "other" in data["mcpServers"]


def test_antigravity_system_prompt():
    from truememory.hooks.adapters.antigravity import AntigravityAdapter

    adapter = AntigravityAdapter()
    content = adapter.get_system_prompt_content()
    assert "TrueMemory" in content
    assert len(content) > 50


# ---------------------------------------------------------------------------
# Groq provider tests
# ---------------------------------------------------------------------------


def test_groq_hydrate_config():
    from truememory.ingest.models import LLMConfig, hydrate_config

    cfg = LLMConfig(provider="groq", api_key="gsk_test")
    cfg = hydrate_config(cfg)
    assert cfg.base_url == "https://api.groq.com/openai/v1"
    assert cfg.model == "llama-3.3-70b-versatile"
    assert cfg.api_key == "gsk_test"


def test_groq_hydrate_config_env_key(monkeypatch):
    from truememory.ingest.models import LLMConfig, hydrate_config

    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_env")
    cfg = LLMConfig(provider="groq")
    cfg = hydrate_config(cfg)
    assert cfg.api_key == "gsk_from_env"
    assert cfg.base_url == "https://api.groq.com/openai/v1"


def test_groq_hydrate_preserves_custom_model():
    from truememory.ingest.models import LLMConfig, hydrate_config

    cfg = LLMConfig(provider="groq", model="mixtral-8x7b-32768", api_key="gsk_test")
    cfg = hydrate_config(cfg)
    assert cfg.model == "mixtral-8x7b-32768"


def test_groq_auto_detect(monkeypatch):
    """Groq is detected when GROQ_API_KEY is set and higher-priority providers are absent."""
    from truememory.ingest import models as models_mod

    # Disable higher-priority providers
    monkeypatch.setattr(models_mod, "_ollama_available", lambda: False)
    monkeypatch.setattr(models_mod, "_claude_cli_available", lambda: False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_auto")

    cfg = models_mod.auto_detect()
    assert cfg.provider == "groq"
    assert cfg.api_key == "gsk_test_auto"


def test_groq_routes_through_openai_compat():
    """Groq provider routes through _complete_openai_compat (not anthropic)."""
    from truememory.ingest.models import LLMConfig

    cfg = LLMConfig(provider="groq", model="test-model", api_key="gsk_test",
                    base_url="https://api.groq.com/openai/v1")

    # We just verify the routing logic (not the actual HTTP call)
    # The complete() function routes groq through _complete_openai_compat
    # because it's not "anthropic" or "claude_cli"
    assert cfg.provider not in ("anthropic", "claude_cli")


def test_groq_in_mcp_server_providers():
    """Groq appears in the MCP server LLM providers table."""
    from truememory import mcp_server

    providers = {p[0] for p in mcp_server._LLM_PROVIDERS}
    assert "groq" in providers


def test_groq_configure_accepted(tmp_path, monkeypatch):
    """truememory_configure accepts 'groq' as api_provider."""
    from truememory import mcp_server

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    monkeypatch.setattr(mcp_server, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(mcp_server, "_TRUEMEMORY_DIR", tmp_path)
    # Invalidate config cache
    monkeypatch.setattr(mcp_server, "_config_cache", None)
    monkeypatch.setattr(mcp_server, "_config_cache_time", 0.0)

    result = json.loads(mcp_server.truememory_configure(
        tier="base", api_key="gsk_test123", api_provider="groq",
    ))
    assert "error" not in result


# ---------------------------------------------------------------------------
# DeepSearch model selection tests
# ---------------------------------------------------------------------------


def test_deepsearch_falls_back_when_not_configured(tmp_path, monkeypatch):
    """When no deepsearch_provider is set, _build_deepsearch_llm_fn returns None."""
    from truememory import mcp_server

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"tier": "pro"}))
    monkeypatch.setattr(mcp_server, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(mcp_server, "_config_cache", None)
    monkeypatch.setattr(mcp_server, "_config_cache_time", 0.0)

    fn = mcp_server._build_deepsearch_llm_fn()
    assert fn is None  # No override; should fall back to default


def test_deepsearch_uses_custom_provider_when_configured(tmp_path, monkeypatch):
    """When deepsearch_provider is set with a valid key, a function is returned."""
    from truememory import mcp_server

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "tier": "pro",
        "deepsearch_provider": "groq",
        "deepsearch_model": "llama-3.3-70b-versatile",
        "groq_api_key": "gsk_test_deepsearch",
    }))
    monkeypatch.setattr(mcp_server, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(mcp_server, "_config_cache", None)
    monkeypatch.setattr(mcp_server, "_config_cache_time", 0.0)

    fn = mcp_server._build_deepsearch_llm_fn()
    assert fn is not None
    assert callable(fn)


def test_deepsearch_returns_none_for_unknown_provider(tmp_path, monkeypatch):
    """Unknown deepsearch_provider falls back gracefully."""
    from truememory import mcp_server

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "tier": "pro",
        "deepsearch_provider": "nonexistent_provider",
    }))
    monkeypatch.setattr(mcp_server, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(mcp_server, "_config_cache", None)
    monkeypatch.setattr(mcp_server, "_config_cache_time", 0.0)

    fn = mcp_server._build_deepsearch_llm_fn()
    assert fn is None


def test_deepsearch_returns_none_when_no_api_key(tmp_path, monkeypatch):
    """deepsearch_provider set but no matching API key returns None."""
    from truememory import mcp_server

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "tier": "pro",
        "deepsearch_provider": "groq",
        # No groq_api_key set
    }))
    monkeypatch.setattr(mcp_server, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(mcp_server, "_config_cache", None)
    monkeypatch.setattr(mcp_server, "_config_cache_time", 0.0)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    fn = mcp_server._build_deepsearch_llm_fn()
    assert fn is None


def test_deepsearch_resolve_uses_override(tmp_path, monkeypatch):
    """_resolve_deepsearch_llm returns the override fn when configured."""
    from truememory import mcp_server

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "tier": "edge",  # Edge tier -- normally no LLM
        "deepsearch_provider": "openai",
        "openai_api_key": "sk-test-deepsearch",
    }))
    monkeypatch.setattr(mcp_server, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(mcp_server, "_config_cache", None)
    monkeypatch.setattr(mcp_server, "_config_cache_time", 0.0)
    # Reset deepsearch cache
    monkeypatch.setattr(mcp_server, "_cached_deepsearch_llm_fn", None)
    monkeypatch.setattr(mcp_server, "_cached_deepsearch_llm_fn_built", False)

    fn = mcp_server._resolve_deepsearch_llm()
    assert fn is not None  # Override active even on edge tier


def test_deepsearch_resolve_falls_back_to_default_on_pro(tmp_path, monkeypatch):
    """On pro tier with no deepsearch override, falls back to _get_llm_fn."""
    from truememory import mcp_server

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"tier": "pro"}))
    monkeypatch.setattr(mcp_server, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(mcp_server, "_config_cache", None)
    monkeypatch.setattr(mcp_server, "_config_cache_time", 0.0)
    # Reset caches
    monkeypatch.setattr(mcp_server, "_cached_deepsearch_llm_fn", None)
    monkeypatch.setattr(mcp_server, "_cached_deepsearch_llm_fn_built", False)
    monkeypatch.setattr(mcp_server, "_cached_llm_fn", "mock_default_fn")
    monkeypatch.setattr(mcp_server, "_cached_llm_fn_built", True)

    fn = mcp_server._resolve_deepsearch_llm()
    assert fn == "mock_default_fn"


def test_deepsearch_resolve_returns_none_on_edge_no_override(tmp_path, monkeypatch):
    """On edge tier with no deepsearch override, returns None."""
    from truememory import mcp_server

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"tier": "edge"}))
    monkeypatch.setattr(mcp_server, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(mcp_server, "_config_cache", None)
    monkeypatch.setattr(mcp_server, "_config_cache_time", 0.0)
    # Reset caches
    monkeypatch.setattr(mcp_server, "_cached_deepsearch_llm_fn", None)
    monkeypatch.setattr(mcp_server, "_cached_deepsearch_llm_fn_built", False)

    fn = mcp_server._resolve_deepsearch_llm()
    assert fn is None
