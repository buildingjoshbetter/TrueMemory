"""The claude CLI extraction backend defaults to Haiku, not the CLI's default model.

Extraction runs up to 20 calls per session on the user's subscription; the CLI
default (usually Opus) exhausted it.
"""
from __future__ import annotations

from truememory.ingest import models


def test_claude_cli_defaults_to_haiku() -> None:
    config = models.hydrate_config(models.LLMConfig(provider="claude_cli"))
    assert config.provider == "claude_cli"
    assert config.model == "haiku"


def test_claude_cli_keeps_an_explicit_model() -> None:
    config = models.hydrate_config(models.LLMConfig(provider="claude-cli", model="sonnet"))
    assert config.model == "sonnet"


def test_cli_command_passes_the_model(monkeypatch) -> None:
    seen = {}

    class Done:
        returncode = 0
        stdout = '{"result": "ok", "is_error": false}'
        stderr = ""

    def run(cmd, **kwargs):
        seen["cmd"] = cmd
        return Done()

    monkeypatch.setattr(models, "_claude_cli_available", lambda: True)
    monkeypatch.setattr(models.subprocess, "run", run)
    config = models.hydrate_config(models.LLMConfig(provider="claude_cli"))
    assert models.complete(config, "prompt") == "ok"
    assert seen["cmd"][seen["cmd"].index("--model") + 1] == "haiku"
    # The user's MCP servers are not started for an extraction call.
    assert "--strict-mcp-config" in seen["cmd"] and "--mcp-config" not in seen["cmd"]
