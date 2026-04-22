"""Regression lock for Hunter F28 + F29 — platform compatibility.

F28: POSIX ``chmod(0o600)`` / ``chmod(0o700)`` silently no-op on Windows,
leaving `~/.truememory/config.json` (which stores API keys in plaintext)
readable by any local user. Both `mcp_server._save_config` and
`ingest/cli._save_truememory_config` now warn to stderr when persisting
an API key on Windows.

F29: `_setup_claude` previously hardcoded the macOS Claude Desktop path,
so Linux and Windows users got "Claude Desktop not detected" even when
it was installed. Resolution is now per-platform via
`_claude_desktop_config_path`.
"""
from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# F29 — per-platform Claude Desktop config path
# ---------------------------------------------------------------------------


def test_claude_desktop_path_macos(monkeypatch):
    import truememory.mcp_server as ms
    monkeypatch.setattr(ms.sys, "platform", "darwin")
    p = ms._claude_desktop_config_path()
    assert "Library/Application Support/Claude/claude_desktop_config.json" in str(p)


def test_claude_desktop_path_linux(monkeypatch):
    import truememory.mcp_server as ms
    monkeypatch.setattr(ms.sys, "platform", "linux")
    p = ms._claude_desktop_config_path()
    assert ".config/Claude/claude_desktop_config.json" in str(p)


def test_claude_desktop_path_linux_variant(monkeypatch):
    """Any non-darwin, non-win32 platform should resolve to the Linux path
    (the fall-through branch — covers OpenBSD, FreeBSD, etc.)."""
    import truememory.mcp_server as ms
    monkeypatch.setattr(ms.sys, "platform", "freebsd14")
    p = ms._claude_desktop_config_path()
    assert ".config/Claude/claude_desktop_config.json" in str(p)


def test_claude_desktop_path_windows_with_appdata(monkeypatch):
    import truememory.mcp_server as ms
    monkeypatch.setattr(ms.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", "C:\\Users\\test\\AppData\\Roaming")
    p = ms._claude_desktop_config_path()
    as_str = str(p)
    assert "Roaming" in as_str
    assert "Claude" in as_str
    assert "claude_desktop_config.json" in as_str


def test_claude_desktop_path_windows_without_appdata(monkeypatch):
    """If APPDATA is unset we fall back to the canonical Roaming path
    under the user's home directory."""
    import truememory.mcp_server as ms
    monkeypatch.setattr(ms.sys, "platform", "win32")
    monkeypatch.delenv("APPDATA", raising=False)
    p = ms._claude_desktop_config_path()
    as_str = str(p)
    assert "AppData" in as_str or "appdata" in as_str.lower()
    assert "Roaming" in as_str
    assert "claude_desktop_config.json" in as_str


# ---------------------------------------------------------------------------
# F28 — Windows permission warning in _save_config
# ---------------------------------------------------------------------------


def _tmp_config_dir(tmp_path, monkeypatch, ms):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".truememory").mkdir()
    monkeypatch.setattr(ms, "_TRUEMEMORY_DIR", home / ".truememory")
    monkeypatch.setattr(ms, "_CONFIG_PATH", home / ".truememory" / "config.json")
    return home


def test_save_config_warns_on_windows_when_api_key_present(
    tmp_path, monkeypatch, capsys
):
    import truememory.mcp_server as ms
    _tmp_config_dir(tmp_path, monkeypatch, ms)
    monkeypatch.setattr(ms.sys, "platform", "win32")

    ms._save_config({"tier": "pro", "anthropic_api_key": "sk-ant-fake"})

    captured = capsys.readouterr()
    assert "windows" in captured.err.lower()
    assert "config.json" in captured.err.lower()
    assert "environment variable" in captured.err.lower()


def test_save_config_silent_on_windows_without_api_key(
    tmp_path, monkeypatch, capsys
):
    """The warning fires only when a secret is being persisted — a bare
    tier-only config is not sensitive, so don't nag."""
    import truememory.mcp_server as ms
    _tmp_config_dir(tmp_path, monkeypatch, ms)
    monkeypatch.setattr(ms.sys, "platform", "win32")

    ms._save_config({"tier": "edge"})

    captured = capsys.readouterr()
    assert captured.err == ""


def test_save_config_silent_on_posix_with_api_key(
    tmp_path, monkeypatch, capsys
):
    """On macOS / Linux the chmod(0o600) is real; the warning must NOT
    fire there (would be false alarm)."""
    import truememory.mcp_server as ms
    _tmp_config_dir(tmp_path, monkeypatch, ms)
    monkeypatch.setattr(ms.sys, "platform", "darwin")

    ms._save_config({"tier": "pro", "anthropic_api_key": "sk-ant-fake"})

    captured = capsys.readouterr()
    assert captured.err == ""


def test_save_config_roundtrip_writes_file(tmp_path, monkeypatch):
    """Baseline: the save path still produces a readable JSON file on
    POSIX (no regression to the happy path)."""
    import truememory.mcp_server as ms
    home = _tmp_config_dir(tmp_path, monkeypatch, ms)
    monkeypatch.setattr(ms.sys, "platform", "darwin")

    ms._save_config({"tier": "base", "anthropic_api_key": "sk-ant-xyz"})

    cfg = home / ".truememory" / "config.json"
    assert cfg.exists()
    data = json.loads(cfg.read_text())
    assert data["tier"] == "base"
    assert data["anthropic_api_key"] == "sk-ant-xyz"


# ---------------------------------------------------------------------------
# F28 — duplicate in ingest/cli.py
# ---------------------------------------------------------------------------


def test_ingest_cli_save_warns_on_windows_when_api_key_present(
    tmp_path, monkeypatch, capsys
):
    from truememory.ingest import cli as ic
    cfg_path = tmp_path / ".truememory" / "config.json"
    monkeypatch.setattr(ic, "_TRUEMEMORY_CONFIG_PATH", cfg_path)
    monkeypatch.setattr(ic.sys, "platform", "win32")

    ic._save_truememory_config({"tier": "pro", "openrouter_api_key": "sk-or-fake"})

    captured = capsys.readouterr()
    assert "windows" in captured.err.lower()
    assert "environment variable" in captured.err.lower()


def test_ingest_cli_save_silent_on_posix(tmp_path, monkeypatch, capsys):
    from truememory.ingest import cli as ic
    cfg_path = tmp_path / ".truememory" / "config.json"
    monkeypatch.setattr(ic, "_TRUEMEMORY_CONFIG_PATH", cfg_path)
    monkeypatch.setattr(ic.sys, "platform", "linux")

    ic._save_truememory_config({"tier": "pro", "openrouter_api_key": "sk-or-fake"})

    captured = capsys.readouterr()
    assert captured.err == ""
