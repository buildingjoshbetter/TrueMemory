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

    Default: ``TRUEMEMORY_DISABLE_CLAUDE_CLI=1`` so legacy API-key tests
    don't accidentally pick up the priority-1 Claude CLI provider on
    machines where ``claude.exe`` is on PATH. Tests targeting the Claude
    CLI path explicitly delete this env var.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".truememory").mkdir()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TRUEMEMORY_DISABLE_CLAUDE_CLI", "1")
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


def test_importing_mcp_server_does_not_set_hf_offline():
    """Regression lock: importing `truememory.mcp_server` must NOT set
    `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` as a side effect. That was
    a perf shortcut for the MCP CLI entry point, but when module-level
    setdefault ran at import time it poisoned later tests / notebooks
    that expected online HF access (CI has no cached model2vec, so the
    first `build_vectors` call raised `OfflineModeIsEnabled`).
    """
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os\n"
                "assert 'HF_HUB_OFFLINE' not in os.environ\n"
                "import truememory.mcp_server  # noqa: F401\n"
                "assert 'HF_HUB_OFFLINE' not in os.environ, 'mcp_server set HF_HUB_OFFLINE at import'\n"
                "assert 'TRANSFORMERS_OFFLINE' not in os.environ, 'mcp_server set TRANSFORMERS_OFFLINE at import'\n"
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess failed: {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


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


# ---------------------------------------------------------------------------
# Claude CLI provider (priority 1, no API key — subscription auth)
# ---------------------------------------------------------------------------


def test_claude_cli_builder_returns_callable_when_on_path(server, monkeypatch):
    """`_build_claude_cli_llm` returns a `(prompt) -> str` closure when
    the `claude` binary is on PATH. The closure wraps
    `_complete_claude_cli` from the extraction-pipeline models module."""
    import truememory.ingest.models as models

    monkeypatch.setattr(models, "_claude_cli_available", lambda: True)
    monkeypatch.setattr(
        models, "_complete_claude_cli",
        lambda config, prompt, system: f"echo:{prompt}",
    )

    fn = server._build_claude_cli_llm("")
    assert callable(fn)
    assert fn("hi") == "echo:hi"


def test_claude_cli_builder_raises_when_not_on_path(server, monkeypatch):
    """`_build_claude_cli_llm` raises RuntimeError when `claude` isn't on
    PATH — lets `_build_llm_fn` fall through to the next provider."""
    import truememory.ingest.models as models

    monkeypatch.setattr(models, "_claude_cli_available", lambda: False)

    with pytest.raises(RuntimeError, match="claude CLI not on PATH"):
        server._build_claude_cli_llm("")


def test_build_llm_fn_returns_claude_cli_when_available_no_keys(server, monkeypatch):
    """Priority 1: with no API keys set and CLI on PATH, `_build_llm_fn`
    returns the Claude CLI closure without ever consulting api_key env vars."""
    monkeypatch.delenv("TRUEMEMORY_DISABLE_CLAUDE_CLI", raising=False)
    import truememory.ingest.models as models

    monkeypatch.setattr(models, "_claude_cli_available", lambda: True)
    monkeypatch.setattr(
        models, "_complete_claude_cli",
        lambda config, prompt, system: "ok",
    )

    fn = server._build_llm_fn()
    assert fn is not None
    assert fn("test") == "ok"
    assert server._current_llm_provider_name == "claude_cli"


def test_build_llm_fn_claude_cli_wins_over_openrouter_key(server, monkeypatch):
    """Priority 1 means Claude CLI wins even when OPENROUTER_API_KEY is set
    — this is the deliberate behavior change for users with both. Opt out
    via `TRUEMEMORY_DISABLE_CLAUDE_CLI=1`."""
    monkeypatch.delenv("TRUEMEMORY_DISABLE_CLAUDE_CLI", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
    import truememory.ingest.models as models

    monkeypatch.setattr(models, "_claude_cli_available", lambda: True)
    monkeypatch.setattr(
        models, "_complete_claude_cli",
        lambda config, prompt, system: "from-cli",
    )
    # OpenRouter builder would also succeed if reached — but it must NOT
    # be reached. Sentinel error if so.
    def _openrouter_must_not_be_called(api_key):
        raise AssertionError("OpenRouter builder called despite Claude CLI win")
    monkeypatch.setattr(server, "_build_openrouter_llm", _openrouter_must_not_be_called)

    fn = server._build_llm_fn()
    assert fn is not None
    assert fn("x") == "from-cli"
    assert server._current_llm_provider_name == "claude_cli"


def test_build_llm_fn_disable_claude_cli_env_falls_through_to_openrouter(server, monkeypatch):
    """`TRUEMEMORY_DISABLE_CLAUDE_CLI=1` (already set by fixture) skips the
    Claude CLI provider even when CLI is on PATH, falling through to
    OpenRouter (or whatever API key is set)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    import truememory.ingest.models as models

    # CLI is "available" but kill switch is on (from fixture)
    monkeypatch.setattr(models, "_claude_cli_available", lambda: True)

    def _ok_openrouter(api_key):
        def _fn(prompt: str) -> str:
            return "from-openrouter"
        return _fn
    monkeypatch.setattr(server, "_build_openrouter_llm", _ok_openrouter)

    fn = server._build_llm_fn()
    assert fn is not None
    assert fn("y") == "from-openrouter"
    assert server._current_llm_provider_name == "openrouter"


def test_build_llm_fn_claude_cli_init_error_falls_through(server, monkeypatch):
    """If `_build_claude_cli_llm` raises (e.g., CLI not on PATH on this
    machine), the loop records the error and falls through to the next
    provider. Mirrors the API-key provider error-handling pattern."""
    monkeypatch.delenv("TRUEMEMORY_DISABLE_CLAUDE_CLI", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    import truememory.ingest.models as models

    monkeypatch.setattr(models, "_claude_cli_available", lambda: False)

    def _ok_openrouter(api_key):
        def _fn(prompt: str) -> str:
            return "from-openrouter"
        return _fn
    monkeypatch.setattr(server, "_build_openrouter_llm", _ok_openrouter)

    fn = server._build_llm_fn()
    assert fn is not None
    assert fn("z") == "from-openrouter"
    assert server._current_llm_provider_name == "openrouter"
    assert "claude_cli" in server._llm_last_error
