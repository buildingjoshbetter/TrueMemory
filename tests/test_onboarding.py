"""Tests for conversational onboarding (#188).

Validates CLI detection display, --cli flag parsing, and
integration state tracking.
"""
from __future__ import annotations

from unittest.mock import patch


# -- Import tests --

def test_import_setup_cli_integrations():
    from truememory.ingest.cli import _setup_cli_integrations  # noqa: F401


def test_import_hooks_cli():
    from truememory.hooks.cli import install_cli, uninstall_cli, verify_cli  # noqa: F401


# -- CLI flag parsing --

def test_setup_accepts_cli_flag():
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "truememory.ingest.cli", "setup", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert "--cli" in result.stdout


# -- Registry integration --

def test_detect_installed_returns_adapters():
    from truememory.hooks.registry import detect_installed
    installed = detect_installed()
    assert isinstance(installed, list)


def test_get_adapter_known():
    from truememory.hooks.registry import get_adapter
    adapter = get_adapter("claude")
    assert adapter is not None
    assert adapter.cli_id == "claude"


def test_get_adapter_unknown_returns_none():
    from truememory.hooks.registry import get_adapter
    assert get_adapter("nonexistent") is None


# -- State tracking --

def test_mark_configured_and_load(tmp_path, monkeypatch):
    from truememory.hooks import registry
    state_file = tmp_path / "integrations.json"
    monkeypatch.setattr(registry, "STATE_FILE", state_file)

    registry.mark_configured("kimi")
    state = registry.load_state()
    assert "kimi" in state["configured"]
    assert "kimi" in state["configured_at"]


# -- _setup_cli_integrations with --cli flag --

def test_setup_cli_integrations_with_cli_flag(capsys):
    import argparse

    args = argparse.Namespace(cli="claude", non_interactive=True)
    config = {"user_id": ""}

    from truememory.ingest.cli import _setup_cli_integrations

    with patch("truememory.hooks.cli.install_cli", return_value=True):
        _setup_cli_integrations(args, config)

    captured = capsys.readouterr()
    assert "Claude Code" in captured.out


def test_setup_cli_integrations_unknown_cli(capsys):
    import argparse
    args = argparse.Namespace(cli="nonexistent", non_interactive=True)
    config = {}

    from truememory.ingest.cli import _setup_cli_integrations
    _setup_cli_integrations(args, config)

    captured = capsys.readouterr()
    assert "Unknown CLI" in captured.out


def test_setup_cli_integrations_multiple(capsys):
    import argparse
    args = argparse.Namespace(cli="claude,nonexistent", non_interactive=True)
    config = {"user_id": ""}

    from truememory.ingest.cli import _setup_cli_integrations

    with patch("truememory.hooks.cli.install_cli", return_value=True):
        _setup_cli_integrations(args, config)

    captured = capsys.readouterr()
    assert "Claude Code" in captured.out
    assert "Unknown CLI" in captured.out
