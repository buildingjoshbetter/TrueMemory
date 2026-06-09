"""Regression tests for PR 575 — ChatGPT Desktop adapter safety.

Covers the clobber and fabricated-success bugs found in review:
- uninstall() must never replace a corrupt-but-recoverable config with `{}`
- uninstall() must not rewrite a config that has no truememory entry
- install_mcp() must back up (never silently destroy) an unparseable config
- detect() must require the actual app, not just a config directory
- install_mcp() must refuse (no config write, no "configured" state) when the
  ChatGPT Desktop app is absent
- an explicit EXPERIMENTAL warning must be emitted whenever config is written
"""
from __future__ import annotations

import json

import pytest

CORRUPT_CONFIG = """\
{
  "mcpServers": {
    "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]},
  }
}
"""


def _patch_paths(monkeypatch, tmp_path, app_installed=True):
    from truememory.hooks.adapters import chatgpt as chatgpt_mod

    chatgpt_dir = tmp_path / "com.openai.chat"
    config_path = chatgpt_dir / "mcp.json"
    # raising=False so the same tests demonstrably FAIL against the pre-fix
    # adapter (which had no _app_installed gate) instead of erroring.
    monkeypatch.setattr(
        chatgpt_mod, "_app_installed", lambda: app_installed, raising=False
    )
    monkeypatch.setattr(chatgpt_mod, "_CHATGPT_DIR", chatgpt_dir)
    monkeypatch.setattr(chatgpt_mod, "_CONFIG_PATH", config_path)
    return config_path


def test_uninstall_corrupt_config_is_noop_with_warning(tmp_path, monkeypatch, capsys):
    """F2: uninstall over an unparseable config must leave it byte-identical
    and warn on stderr rather than silently no-op'ing."""
    from truememory.hooks.adapters.chatgpt import ChatGPTAdapter

    config_path = _patch_paths(monkeypatch, tmp_path)
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CORRUPT_CONFIG, encoding="utf-8")

    ChatGPTAdapter().uninstall()

    assert config_path.read_text(encoding="utf-8") == CORRUPT_CONFIG
    assert "leaving it untouched" in capsys.readouterr().err


def test_uninstall_foreign_servers_only_is_noop(tmp_path, monkeypatch):
    """uninstall must not rewrite/reformat a config with no truememory entry."""
    from truememory.hooks.adapters.chatgpt import ChatGPTAdapter

    config_path = _patch_paths(monkeypatch, tmp_path)
    config_path.parent.mkdir(parents=True)
    original = '{"mcpServers": {"github": {"command": "npx"}}}'
    config_path.write_text(original, encoding="utf-8")

    ChatGPTAdapter().uninstall()

    assert config_path.read_text(encoding="utf-8") == original


def test_uninstall_non_dict_config_is_noop_with_warning(tmp_path, monkeypatch, capsys):
    """uninstall must not crash or clobber when the config is valid JSON but not an object."""
    from truememory.hooks.adapters.chatgpt import ChatGPTAdapter

    config_path = _patch_paths(monkeypatch, tmp_path)
    config_path.parent.mkdir(parents=True)
    original = '["not", "a", "dict"]'
    config_path.write_text(original, encoding="utf-8")

    ChatGPTAdapter().uninstall()

    assert config_path.read_text(encoding="utf-8") == original
    assert "leaving it untouched" in capsys.readouterr().err


def test_install_backs_up_corrupt_config(tmp_path, monkeypatch, capsys):
    """F1: install over an unparseable config must back it up, never destroy it."""
    from truememory.hooks.adapters.chatgpt import ChatGPTAdapter

    config_path = _patch_paths(monkeypatch, tmp_path)
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CORRUPT_CONFIG, encoding="utf-8")

    ChatGPTAdapter().install_mcp(python_path="/usr/bin/python3")

    backups = list(config_path.parent.glob("mcp.json.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == CORRUPT_CONFIG

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["mcpServers"]["truememory"]["command"] == "/usr/bin/python3"

    err = capsys.readouterr().err
    assert "backed it up" in err


def test_detect_false_when_app_absent_even_with_config_dir(tmp_path, monkeypatch):
    """F4: a config directory alone (possibly self-created) must not count as detection."""
    from truememory.hooks.adapters.chatgpt import ChatGPTAdapter

    config_path = _patch_paths(monkeypatch, tmp_path, app_installed=False)
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}", encoding="utf-8")

    assert not ChatGPTAdapter().detect()


def test_install_refuses_when_app_absent(tmp_path, monkeypatch):
    """F4: install must not fabricate config dirs or claim success without the app."""
    from truememory.hooks.adapters.chatgpt import ChatGPTAdapter

    config_path = _patch_paths(monkeypatch, tmp_path, app_installed=False)

    with pytest.raises(RuntimeError, match="not found"):
        ChatGPTAdapter().install_mcp(python_path="/usr/bin/python3")

    assert not config_path.exists()
    assert not config_path.parent.exists()


def test_install_cli_reports_failure_when_app_absent(tmp_path, monkeypatch):
    """The setup flow must report failure (no mark_configured) when the app is absent."""
    from truememory.hooks import registry
    from truememory.hooks.cli import install_cli

    _patch_paths(monkeypatch, tmp_path, app_installed=False)
    state_file = tmp_path / "integrations.json"
    monkeypatch.setattr(registry, "STATE_FILE", state_file)

    assert install_cli("chatgpt") is False
    assert not state_file.exists()


def test_install_write_failure_leaves_original_intact(tmp_path, monkeypatch):
    """Atomic write: a failure at replace time must leave the original config
    byte-identical with no stray temp files behind."""
    from truememory.hooks.adapters import chatgpt as chatgpt_mod
    from truememory.hooks.adapters.chatgpt import ChatGPTAdapter

    config_path = _patch_paths(monkeypatch, tmp_path)
    config_path.parent.mkdir(parents=True)
    original = json.dumps({"mcpServers": {"github": {"command": "npx"}}})
    config_path.write_text(original, encoding="utf-8")

    real_atomic_write = chatgpt_mod._atomic_write

    def failing_replace_write(path, text):
        def boom(src, dst):
            raise OSError("disk full")

        real_replace = chatgpt_mod.os.replace
        chatgpt_mod.os.replace = boom
        try:
            real_atomic_write(path, text)
        finally:
            chatgpt_mod.os.replace = real_replace

    monkeypatch.setattr(chatgpt_mod, "_atomic_write", failing_replace_write)

    with pytest.raises(OSError):
        ChatGPTAdapter().install_mcp(python_path="/usr/bin/python3")

    assert config_path.read_text(encoding="utf-8") == original
    assert list(config_path.parent.glob("*.tmp")) == []


def test_backup_names_are_unique_and_exclusive(tmp_path, monkeypatch):
    """Backups carry pid + random suffix and are created with O_EXCL, so
    repeated installs over corrupt configs in the same second never
    overwrite a previous backup."""
    import os as os_mod

    from truememory.hooks.adapters.chatgpt import ChatGPTAdapter

    config_path = _patch_paths(monkeypatch, tmp_path)
    config_path.parent.mkdir(parents=True)

    adapter = ChatGPTAdapter()
    config_path.write_text(CORRUPT_CONFIG, encoding="utf-8")
    adapter.install_mcp(python_path="/usr/bin/python3")
    second_corrupt = CORRUPT_CONFIG.replace("github", "gitlab")
    config_path.write_text(second_corrupt, encoding="utf-8")
    adapter.install_mcp(python_path="/usr/bin/python3")

    backups = sorted(config_path.parent.glob("mcp.json.bak-*"))
    assert len(backups) == 2
    contents = {b.read_text(encoding="utf-8") for b in backups}
    assert contents == {CORRUPT_CONFIG, second_corrupt}
    for b in backups:
        assert f"-{os_mod.getpid()}-" in b.name


def test_no_stray_backup_on_refuse(tmp_path, monkeypatch):
    """The app-absent refusal must happen BEFORE any side effect: no backup
    file, no write, config byte-identical."""
    from truememory.hooks.adapters.chatgpt import ChatGPTAdapter

    config_path = _patch_paths(monkeypatch, tmp_path, app_installed=False)
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CORRUPT_CONFIG, encoding="utf-8")

    with pytest.raises(RuntimeError, match="not found"):
        ChatGPTAdapter().install_mcp(python_path="/usr/bin/python3")

    assert config_path.read_text(encoding="utf-8") == CORRUPT_CONFIG
    assert list(config_path.parent.glob("mcp.json.bak-*")) == []
    assert list(config_path.parent.glob("*.tmp")) == []


def test_install_refuses_when_backup_fails(tmp_path, monkeypatch):
    """If the corrupt config cannot be backed up, install must refuse and
    leave the original byte-identical (never clobber without a backup)."""
    from truememory.hooks.adapters import chatgpt as chatgpt_mod
    from truememory.hooks.adapters.chatgpt import ChatGPTAdapter

    config_path = _patch_paths(monkeypatch, tmp_path)
    config_path.parent.mkdir(parents=True)
    config_path.write_text(CORRUPT_CONFIG, encoding="utf-8")

    def backup_boom(path):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(chatgpt_mod, "_backup_config", backup_boom)

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        ChatGPTAdapter().install_mcp(python_path="/usr/bin/python3")

    assert config_path.read_text(encoding="utf-8") == CORRUPT_CONFIG
    assert list(config_path.parent.glob("*.tmp")) == []


def test_experimental_warning_emitted_on_install(tmp_path, monkeypatch, capsys):
    """When the adapter does run, it must tell the truth about ChatGPT support."""
    from truememory.hooks.adapters.chatgpt import ChatGPTAdapter

    _patch_paths(monkeypatch, tmp_path)

    ChatGPTAdapter().install_mcp(python_path="/usr/bin/python3")

    err = capsys.readouterr().err
    assert "EXPERIMENTAL" in err
    assert "does not currently load local MCP servers" in err
