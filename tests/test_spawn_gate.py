"""Tests for the flock-based spawn gate that prevents ingest process avalanches.

The spawn gate serializes check-then-spawn decisions via a file lock so that
concurrent Stop/SessionStart hooks can't all see 0 active processes and all
spawn simultaneously.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def test_spawn_gate_yields_true_under_cap(tmp_path, monkeypatch):
    """When fewer than SPAWN_CAP processes are active, gate yields True."""
    from truememory.hooks import core

    monkeypatch.setattr(core, "SPAWN_CAP", 3)
    monkeypatch.setattr(core, "SPAWN_LOCK_PATH", tmp_path / ".spawn.lock")
    monkeypatch.setattr(core, "_count_active_ingest_processes", lambda: 1)

    with core.spawn_gate() as allowed:
        assert allowed is True


def test_spawn_gate_yields_false_at_cap(tmp_path, monkeypatch):
    """When at or above SPAWN_CAP, gate yields False."""
    from truememory.hooks import core

    monkeypatch.setattr(core, "SPAWN_CAP", 2)
    monkeypatch.setattr(core, "SPAWN_LOCK_PATH", tmp_path / ".spawn.lock")
    monkeypatch.setattr(core, "_count_active_ingest_processes", lambda: 2)

    with core.spawn_gate() as allowed:
        assert allowed is False


def test_spawn_gate_windows_fallback(tmp_path, monkeypatch):
    """On Windows (no fcntl), gate still works without a file lock."""
    from truememory.hooks import core

    monkeypatch.setattr(core, "_HAS_FCNTL", False)
    monkeypatch.setattr(core, "SPAWN_CAP", 5)
    monkeypatch.setattr(core, "_count_active_ingest_processes", lambda: 0)

    with core.spawn_gate() as allowed:
        assert allowed is True


def test_spawn_cap_env_var_unified(monkeypatch):
    """TRUEMEMORY_SPAWN_CAP takes precedence over TRUEMEMORY_INGEST_SPAWN_CAP."""
    monkeypatch.setenv("TRUEMEMORY_SPAWN_CAP", "7")
    monkeypatch.setenv("TRUEMEMORY_INGEST_SPAWN_CAP", "99")

    import importlib
    from truememory.hooks import core
    importlib.reload(core)
    try:
        assert core.SPAWN_CAP == 7
    finally:
        monkeypatch.delenv("TRUEMEMORY_SPAWN_CAP")
        monkeypatch.delenv("TRUEMEMORY_INGEST_SPAWN_CAP")
        importlib.reload(core)


def test_spawn_cap_fallback_env_var(monkeypatch):
    """When TRUEMEMORY_SPAWN_CAP is not set, falls back to TRUEMEMORY_INGEST_SPAWN_CAP."""
    monkeypatch.delenv("TRUEMEMORY_SPAWN_CAP", raising=False)
    monkeypatch.setenv("TRUEMEMORY_INGEST_SPAWN_CAP", "5")

    import importlib
    from truememory.hooks import core
    importlib.reload(core)
    try:
        assert core.SPAWN_CAP == 5
    finally:
        monkeypatch.delenv("TRUEMEMORY_INGEST_SPAWN_CAP")
        importlib.reload(core)


def test_drain_backlog_respects_spawn_cap(tmp_path, monkeypatch):
    """_drain_backlog must not spawn processes beyond the spawn cap."""
    from truememory.ingest.hooks import session_start as ss_mod
    from truememory.hooks import core as core_mod

    backlog = tmp_path / "backlog"
    backlog.mkdir()
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type": "human", "content": "test"}')

    for i in range(5):
        marker = backlog / f"session-{i}.json"
        marker.write_text(json.dumps({
            "transcript_path": str(transcript),
            "session_id": f"session-{i}",
        }))

    monkeypatch.setattr(ss_mod, "BACKLOG_DIR", backlog)

    spawn_count = {"n": 0}

    @contextmanager
    def _counting_gate():
        if spawn_count["n"] >= 2:
            yield False
        else:
            spawn_count["n"] += 1
            yield True

    monkeypatch.setattr(core_mod, "spawn_gate", _counting_gate)

    popen_calls = []
    def _mock_popen(*args, **kwargs):
        popen_calls.append(args)
        return type("P", (), {"pid": 0, "__enter__": lambda s: s, "__exit__": lambda *a: None})()

    monkeypatch.setattr(subprocess, "Popen", _mock_popen)

    ss_mod._drain_backlog()

    assert len(popen_calls) == 2, f"Expected exactly 2 spawns, got {len(popen_calls)}"
    remaining = list(backlog.glob("*.json"))
    assert len(remaining) == 3, f"Expected 3 remaining backlog items, got {len(remaining)}"
