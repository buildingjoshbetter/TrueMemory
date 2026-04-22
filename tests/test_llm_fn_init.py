"""Regression lock for Hunter F05 — `_build_llm_fn` must log + store
per-provider errors instead of silently returning None.

Prior behavior: three `try: ... except Exception: pass` blocks around
Anthropic / OpenRouter / OpenAI client construction. Any init failure
(bad key, import error, network hiccup) silently fell through to no-HyDE
mode and `_cached_llm_fn_built = True` locked None in for the process
lifetime — a paid Pro tier silently degraded to Base with no surface signal.
"""
from __future__ import annotations

import logging

import pytest


@pytest.fixture
def server(monkeypatch, tmp_path):
    """Scope `_CONFIG_PATH` into tmp_path and reset LLM error state between
    tests. Avoids reloading the module (which would pollute
    huggingface_hub's cached HF_HOME).
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".truememory").mkdir()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import truememory.mcp_server as ms
    monkeypatch.setattr(ms, "_TRUEMEMORY_DIR", home / ".truememory")
    monkeypatch.setattr(ms, "_CONFIG_PATH", home / ".truememory" / "config.json")
    # Reset F05 state so tests don't leak into each other
    ms._clear_all_llm_errors()
    ms._current_llm_provider_name = None
    yield ms
    # Teardown: clear state for the next test
    ms._clear_all_llm_errors()
    ms._current_llm_provider_name = None


def test_no_api_keys_returns_none_without_errors(server):
    fn = server._build_llm_fn()
    assert fn is None
    assert server._llm_last_error == {}


def test_anthropic_init_failure_stores_error_and_logs(server, monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")

    def _boom(api_key):
        raise RuntimeError("simulated anthropic init failure")

    monkeypatch.setattr(server, "_build_anthropic_llm", _boom)
    # Block OpenRouter + OpenAI so we're solely testing anthropic error surface
    monkeypatch.setattr(server, "_build_openrouter_llm", lambda api_key: None)
    monkeypatch.setattr(server, "_build_openai_llm", lambda api_key: None)

    with caplog.at_level(logging.WARNING, logger="truememory.mcp_server"):
        fn = server._build_llm_fn()

    # No other provider keys set → fn is None
    assert fn is None
    # Error captured and log-surfaced
    assert "anthropic" in server._llm_last_error
    assert "RuntimeError" in server._llm_last_error["anthropic"]
    assert any(
        "HyDE LLM init failed" in rec.message and "anthropic" in rec.message
        for rec in caplog.records
    )


def test_successful_provider_clears_that_providers_error(server, monkeypatch):
    """If Anthropic fails but OpenRouter succeeds, we return OpenRouter and
    leave Anthropic's error in the dict (so stats.health can surface it),
    but OpenRouter's entry is cleared."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")

    def _boom_anthropic(api_key):
        raise RuntimeError("simulated")

    def _ok_openrouter(api_key):
        def _fn(prompt: str) -> str:
            return "ok"
        return _fn

    monkeypatch.setattr(server, "_build_anthropic_llm", _boom_anthropic)
    monkeypatch.setattr(server, "_build_openrouter_llm", _ok_openrouter)

    fn = server._build_llm_fn()
    assert fn is not None
    assert fn("anything") == "ok"
    assert "anthropic" in server._llm_last_error  # surfaced in health
    assert "openrouter" not in server._llm_last_error  # cleared on success
    assert server._current_llm_provider_name == "openrouter"


def test_all_providers_fail_sets_none_and_records_all(server, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k2")
    monkeypatch.setenv("OPENAI_API_KEY", "k3")

    def _boom(api_key):
        raise RuntimeError(f"boom:{api_key}")

    monkeypatch.setattr(server, "_build_anthropic_llm", _boom)
    monkeypatch.setattr(server, "_build_openrouter_llm", _boom)
    monkeypatch.setattr(server, "_build_openai_llm", _boom)

    fn = server._build_llm_fn()
    assert fn is None
    assert set(server._llm_last_error.keys()) == {"anthropic", "openrouter", "openai"}
    assert server._current_llm_provider_name is None
