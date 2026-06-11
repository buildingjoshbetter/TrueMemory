"""Tests for issue #651: installer + adapter honesty.

Covers:
- M-29: install_hooks is idempotent (re-running does not duplicate hook entries).
- M-63: the alwaysLoad/MCP patch targets ~/.claude.json and writes atomically;
        subprocess failures are surfaced, not swallowed.
- M-62: hook-less adapters' generic prompt does NOT promise auto-loaded
        directives or SessionEnd transcript capture; hook-capable ones do.
- M-66: the antigravity adapter targets/verifies the documented mcp_config.json.

All filesystem access is redirected to tmp dirs / fake homes; the real
~/.claude and ~/.antigravity are never touched. No model loads.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# M-29: install_hooks idempotency
# ---------------------------------------------------------------------------

def _count_tm_hook_entries(settings: dict) -> dict[str, int]:
    """Count truememory hook entries per event in a settings dict."""
    counts: dict[str, int] = {}
    for event, entries in settings.get("hooks", {}).items():
        if not isinstance(entries, list):
            continue
        n = 0
        for h in entries:
            if not isinstance(h, dict):
                continue
            inner = h.get("hooks", [])
            if isinstance(inner, list) and any(
                isinstance(ih, dict) and "truememory" in str(ih.get("command", "")).lower()
                for ih in inner
            ):
                n += 1
            elif "truememory" in str(h.get("command", "")).lower():
                n += 1
        counts[event] = n
    return counts


def _make_claude_adapter(tmp_path: Path):
    from truememory.hooks.adapters.claude import ClaudeAdapter

    adapter = ClaudeAdapter()
    settings_path = tmp_path / ".claude" / "settings.json"
    # Patch config_path to point at the fake home.
    patcher = patch.object(
        type(adapter), "config_path",
        new=property(lambda self: settings_path),
    )
    return adapter, settings_path, patcher


def test_install_hooks_is_idempotent(tmp_path):
    """Running install_hooks twice must not duplicate hook entries (M-29)."""
    adapter, settings_path, patcher = _make_claude_adapter(tmp_path)
    with patcher:
        adapter.install_hooks(python_path="/usr/bin/python3")
        first = json.loads(settings_path.read_text())
        adapter.install_hooks(python_path="/usr/bin/python3")
        second = json.loads(settings_path.read_text())

    counts_first = _count_tm_hook_entries(first)
    counts_second = _count_tm_hook_entries(second)

    # Each of the 4 events should have exactly one truememory entry, both times.
    assert counts_first == counts_second
    assert counts_first  # non-empty
    for event, n in counts_second.items():
        assert n == 1, f"{event} has {n} truememory entries after 2 installs (expected 1)"


def test_install_hooks_commands_are_module_form(tmp_path):
    """Sanity: the bug was module-form commands vs .py-path dedup needle."""
    adapter, settings_path, patcher = _make_claude_adapter(tmp_path)
    with patcher:
        adapter.install_hooks(python_path="/usr/bin/python3")
    settings = json.loads(settings_path.read_text())
    cmds = [
        ih["command"]
        for entries in settings["hooks"].values()
        for h in entries
        for ih in h.get("hooks", [])
    ]
    assert cmds
    assert all("-m truememory.ingest.hooks" in c for c in cmds)
    # The dedup needle ".py" path must NOT appear (that was the broken needle).
    assert all("session_start.py" not in c for c in cmds)


def test_install_hooks_thrice_no_growth(tmp_path):
    """Three installs (the issue's repro) stay at one entry per event."""
    adapter, settings_path, patcher = _make_claude_adapter(tmp_path)
    with patcher:
        for _ in range(3):
            adapter.install_hooks(python_path="/usr/bin/python3")
        settings = json.loads(settings_path.read_text())
    for event, n in _count_tm_hook_entries(settings).items():
        assert n == 1, f"{event} grew to {n} entries after 3 installs"


# ---------------------------------------------------------------------------
# M-63: install_mcp targets ~/.claude.json + atomic write + subprocess check
# ---------------------------------------------------------------------------

def test_install_mcp_patches_claude_json_not_settings(tmp_path):
    """alwaysLoad must be written to ~/.claude.json (what the CLI reads)."""
    from truememory.hooks.adapters.claude import ClaudeAdapter

    adapter = ClaudeAdapter()
    settings_path = tmp_path / ".claude" / "settings.json"
    claude_json = tmp_path / ".claude.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # Simulate `claude mcp add` having written the server entry to ~/.claude.json.
    claude_json.write_text(json.dumps({
        "mcpServers": {"truememory": {"command": "py", "args": ["-m", "truememory.mcp_server"]}}
    }))
    settings_path.write_text(json.dumps({"existing": "keep"}))

    class _OK:
        returncode = 0
        stderr = b""

    with patch.object(type(adapter), "config_path", new=property(lambda self: settings_path)), \
         patch.object(type(adapter), "mcp_config_path", new=property(lambda self: claude_json)), \
         patch("subprocess.run", return_value=_OK()):
        adapter.install_mcp(python_path="py")

    patched = json.loads(claude_json.read_text())
    assert patched["mcpServers"]["truememory"]["alwaysLoad"] is True
    # settings.json must be left untouched (no truncation, no mcp added there).
    assert json.loads(settings_path.read_text()) == {"existing": "keep"}


def test_install_mcp_atomic_write_no_leftover_tmp(tmp_path):
    """The atomic patch must not leave a .tmp file behind."""
    from truememory.hooks.adapters.claude import ClaudeAdapter

    adapter = ClaudeAdapter()
    claude_json = tmp_path / ".claude.json"
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    claude_json.write_text(json.dumps({
        "mcpServers": {"truememory": {"command": "py"}}
    }))

    class _OK:
        returncode = 0
        stderr = b""

    with patch.object(type(adapter), "config_path", new=property(lambda self: settings_path)), \
         patch.object(type(adapter), "mcp_config_path", new=property(lambda self: claude_json)), \
         patch("subprocess.run", return_value=_OK()):
        adapter.install_mcp(python_path="py")

    leftover = list(tmp_path.glob(".*.tmp"))
    assert not leftover, f"left tmp files: {leftover}"


def test_install_mcp_skips_patch_when_add_fails(tmp_path, capsys):
    """A failed `claude mcp add` must be surfaced, not swallowed, and skip patch."""
    from truememory.hooks.adapters.claude import ClaudeAdapter

    adapter = ClaudeAdapter()
    claude_json = tmp_path / ".claude.json"
    settings_path = tmp_path / ".claude" / "settings.json"

    class _Fail:
        returncode = 1
        stderr = b"boom"

    with patch.object(type(adapter), "config_path", new=property(lambda self: settings_path)), \
         patch.object(type(adapter), "mcp_config_path", new=property(lambda self: claude_json)), \
         patch("subprocess.run", return_value=_Fail()):
        adapter.install_mcp(python_path="py")

    err = capsys.readouterr().err
    assert "claude mcp add" in err  # failure surfaced
    assert not claude_json.exists()  # no phantom patch


def test_atomic_write_json_replaces_in_place(tmp_path):
    from truememory.hooks.adapters.claude import ClaudeAdapter

    target = tmp_path / "cfg.json"
    target.write_text('{"old": true}')
    ClaudeAdapter._atomic_write_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"new": True}
    assert not list(tmp_path.glob(".*.tmp"))


# ---------------------------------------------------------------------------
# M-62: capability-parameterized generic prompt honesty
# ---------------------------------------------------------------------------

def test_hookless_prompt_omits_false_guarantees():
    """A hook-less adapter's prompt must not claim auto-load or SessionEnd."""
    from truememory.hooks.adapters.base import get_generic_system_prompt

    prompt = get_generic_system_prompt(has_hooks=False, has_session_start=False).lower()
    assert "injected automatically at the start of every session" not in prompt
    assert "sessionend" not in prompt
    assert "captures the full transcript" not in prompt
    assert "memory.md" not in prompt
    # It should still teach storing + directives.
    assert "truememory_store" in prompt
    assert "directive" in prompt
    # And it should tell a hook-less host to store manually.
    assert "store" in prompt and "yourself" in prompt


def test_hookful_prompt_keeps_autoload_guarantee():
    """A hook-capable adapter keeps the auto-load promise (template path)."""
    from truememory.hooks.adapters.base import get_generic_system_prompt

    prompt = get_generic_system_prompt(has_hooks=True, has_session_start=True).lower()
    # Template-derived prompt mentions auto-loaded directives / every session.
    assert "every session" in prompt or "session start" in prompt
    assert "truememory" in prompt


def test_antigravity_prompt_is_honest():
    """The real Antigravity adapter (hook-less) yields an honest prompt."""
    from truememory.hooks.adapters.antigravity import AntigravityAdapter

    adapter = AntigravityAdapter()
    assert adapter.has_hooks is False
    assert adapter.has_session_start is False
    prompt = adapter.get_system_prompt_content().lower()
    assert "sessionend" not in prompt
    assert "injected automatically at the start of every session" not in prompt
    assert "truememory_store" in prompt


def test_claude_adapter_declares_hooks():
    from truememory.hooks.adapters.claude import ClaudeAdapter

    a = ClaudeAdapter()
    assert a.has_hooks is True
    assert a.has_session_start is True


@pytest.mark.parametrize("cli_id,expected", [
    ("claude", True),
    ("hermes", True),
    ("kimi", True),
    ("cursor", True),
    ("gemini", True),
    ("codex", True),
    ("openclaw", True),
    ("antigravity", False),
    ("chatgpt", False),
])
def test_adapter_hook_capability_flags(cli_id, expected):
    """Capability flags must match whether the adapter installs real hooks."""
    from truememory.hooks.registry import get_adapter

    adapter = get_adapter(cli_id)
    assert adapter is not None
    assert adapter.has_hooks is expected


# ---------------------------------------------------------------------------
# M-66: antigravity targets/verifies the documented config filename
# ---------------------------------------------------------------------------

def test_antigravity_config_filename_is_mcp_config(tmp_path, monkeypatch):
    """The adapter must read/write the documented mcp_config.json filename."""
    from truememory.hooks.adapters import antigravity as ag_mod
    from truememory.hooks.adapters.antigravity import AntigravityAdapter

    config_path = tmp_path / ".antigravity" / "mcp_config.json"
    monkeypatch.setattr(ag_mod, "_MCP_CONFIG", config_path)

    adapter = AntigravityAdapter()
    assert adapter.config_path.name == "mcp_config.json"

    adapter.install_mcp(python_path="/usr/bin/python3")
    # Written to the documented filename.
    assert config_path.exists()
    assert config_path.name == "mcp_config.json"
    data = json.loads(config_path.read_text())
    assert "truememory" in data["mcpServers"]


def test_antigravity_verify_checks_documented_target(tmp_path, monkeypatch):
    """verify() must pass only when the documented config holds our entry."""
    from truememory.hooks.adapters import antigravity as ag_mod
    from truememory.hooks.adapters.antigravity import AntigravityAdapter

    config_path = tmp_path / ".antigravity" / "mcp_config.json"
    monkeypatch.setattr(ag_mod, "_MCP_CONFIG", config_path)
    adapter = AntigravityAdapter()

    # No config yet -> verify False.
    assert adapter.verify() is False

    # A config at the WRONG filename must NOT satisfy verify().
    config_path.parent.mkdir(parents=True, exist_ok=True)
    (config_path.parent / "mcp.json").write_text(json.dumps({
        "mcpServers": {"truememory": {"command": "py"}}
    }))
    assert adapter.verify() is False

    # Install at the documented filename -> verify True.
    adapter.install_mcp(python_path="py")
    assert adapter.verify() is True
