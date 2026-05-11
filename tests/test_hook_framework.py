"""Tests for the universal hook framework (#186).

Validates the adapter interface, core logic extraction, Claude adapter,
and CLI registry without network calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# -- Import tests --

def test_import_hooks_package():
    import truememory.hooks

def test_import_core():
    from truememory.hooks.core import recall_memories
    from truememory.hooks.core import buffer_message
    from truememory.hooks.core import prune_old_buffers
    from truememory.hooks.core import save_snapshot
    from truememory.hooks.core import run_background_ingestion

def test_import_base_adapter():
    from truememory.hooks.adapters.base import CLIAdapter

def test_import_claude_adapter():
    from truememory.hooks.adapters.claude import ClaudeAdapter

def test_import_registry():
    from truememory.hooks.registry import detect_installed
    from truememory.hooks.registry import detect_configured
    from truememory.hooks.registry import get_adapter


# -- CLIAdapter interface --

def test_claude_adapter_instantiation():
    from truememory.hooks.adapters.claude import ClaudeAdapter
    adapter = ClaudeAdapter()
    assert adapter.name == "Claude Code"
    assert adapter.cli_id == "claude"
    assert isinstance(adapter.config_path, Path)


def test_claude_adapter_implements_all_abstract_methods():
    from truememory.hooks.adapters.base import CLIAdapter
    from truememory.hooks.adapters.claude import ClaudeAdapter
    import inspect
    abstract_methods = {
        name for name, _ in inspect.getmembers(CLIAdapter)
        if getattr(getattr(CLIAdapter, name, None), "__isabstractmethod__", False)
    }
    adapter = ClaudeAdapter()
    for method_name in abstract_methods:
        assert hasattr(adapter, method_name), f"Missing method: {method_name}"


# -- Registry --

def test_get_adapter_claude():
    from truememory.hooks.registry import get_adapter
    adapter = get_adapter("claude")
    assert adapter is not None
    assert adapter.cli_id == "claude"


def test_get_adapter_unknown():
    from truememory.hooks.registry import get_adapter
    assert get_adapter("nonexistent") is None


def test_state_roundtrip(tmp_path, monkeypatch):
    from truememory.hooks import registry
    state_file = tmp_path / "integrations.json"
    monkeypatch.setattr(registry, "STATE_FILE", state_file)

    registry.mark_configured("claude")
    state = registry.load_state()
    assert "claude" in state["configured"]
    assert "claude" in state["configured_at"]

    registry.mark_unconfigured("claude")
    state = registry.load_state()
    assert "claude" not in state["configured"]


# -- Core logic --

def test_buffer_message(tmp_path, monkeypatch):
    from truememory.hooks import core
    monkeypatch.setattr(core, "BUFFER_DIR", tmp_path / "buffers")
    core.buffer_message("test-session-123", "Hello world test message")
    buffer_file = tmp_path / "buffers" / "test-session-123.jsonl"
    assert buffer_file.exists()
    data = json.loads(buffer_file.read_text(encoding="utf-8").strip())
    assert data["content"] == "Hello world test message"
    assert data["role"] == "user"


def test_buffer_message_sanitizes_session_id(tmp_path, monkeypatch):
    from truememory.hooks import core
    monkeypatch.setattr(core, "BUFFER_DIR", tmp_path / "buffers")
    core.buffer_message("../../etc/passwd", "exploit attempt")
    buffer_file = tmp_path / "buffers" / "etcpasswd.jsonl"
    assert buffer_file.exists()


def test_prune_old_buffers(tmp_path, monkeypatch):
    import time
    from truememory.hooks import core
    buf_dir = tmp_path / "buffers"
    buf_dir.mkdir()
    monkeypatch.setattr(core, "BUFFER_DIR", buf_dir)
    monkeypatch.setattr(core, "RETENTION_DAYS", 0)

    old_file = buf_dir / "old-session.jsonl"
    old_file.write_text("{}", encoding="utf-8")
    import os
    os.utime(old_file, (time.time() - 86400, time.time() - 86400))

    core.prune_old_buffers()
    assert not old_file.exists()


def test_recall_memories_empty_db():
    from truememory.hooks.core import recall_memories
    result = recall_memories({}, db_path=":memory:")
    assert result == ""


def test_sanitize_session_id():
    from truememory.hooks.core import _sanitize_session_id
    assert _sanitize_session_id("abc-123_def") == "abc-123_def"
    assert _sanitize_session_id("../../etc") == "etc"
    assert _sanitize_session_id("") == "unknown"


# -- Claude adapter is_configured --

def test_claude_is_configured_false_no_file(tmp_path, monkeypatch):
    from truememory.hooks.adapters.claude import ClaudeAdapter
    adapter = ClaudeAdapter()
    monkeypatch.setattr(
        type(adapter), "config_path",
        property(lambda self: tmp_path / "settings.json"),
    )
    assert not adapter.is_configured()


def test_claude_is_configured_true(tmp_path, monkeypatch):
    from truememory.hooks.adapters.claude import ClaudeAdapter
    adapter = ClaudeAdapter()
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "hooks": {
            "SessionStart": [{
                "matcher": "",
                "hooks": [{"type": "command", "command": "/usr/bin/python truememory/hooks/session_start.py"}],
            }]
        }
    }), encoding="utf-8")
    monkeypatch.setattr(
        type(adapter), "config_path",
        property(lambda self: settings_path),
    )
    assert adapter.is_configured()
