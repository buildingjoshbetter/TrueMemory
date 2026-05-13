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
    monkeypatch.setattr(core, "SPAWN_PIDS_PATH", tmp_path / ".spawn_pids")
    # Write 1 fake live PID (current process, known alive)
    (tmp_path / ".spawn_pids").write_text(f"{os.getpid()}\n")

    with core.spawn_gate() as allowed:
        assert allowed is True


def test_spawn_gate_yields_false_at_cap(tmp_path, monkeypatch):
    """When at or above the dynamic cap, gate yields False."""
    from truememory.hooks import core

    monkeypatch.setattr(core, "_get_spawn_cap", lambda: 1)
    monkeypatch.setattr(core, "SPAWN_LOCK_PATH", tmp_path / ".spawn.lock")
    monkeypatch.setattr(core, "SPAWN_PIDS_PATH", tmp_path / ".spawn_pids")
    (tmp_path / ".spawn_pids").write_text(f"{os.getpid()}\n")

    with core.spawn_gate() as allowed:
        assert allowed is False


def test_spawn_gate_windows_fallback(tmp_path, monkeypatch):
    """On Windows (no fcntl), gate still works without a file lock."""
    from truememory.hooks import core

    monkeypatch.setattr(core, "_HAS_FCNTL", False)
    monkeypatch.setattr(core, "_get_spawn_cap", lambda: 5)
    monkeypatch.setattr(core, "_count_active_ingest_processes", lambda: 0)

    with core.spawn_gate() as allowed:
        assert allowed is True


def test_spawn_cap_env_var_override(monkeypatch):
    """TRUEMEMORY_SPAWN_CAP env var overrides dynamic tier-based cap."""
    from truememory.hooks import core

    import importlib
    monkeypatch.setenv("TRUEMEMORY_SPAWN_CAP", "7")
    importlib.reload(core)
    try:
        assert core._get_spawn_cap() == 7
    finally:
        monkeypatch.delenv("TRUEMEMORY_SPAWN_CAP")
        importlib.reload(core)


def test_spawn_cap_tier_aware(monkeypatch):
    """Edge tier gets higher cap than Base/Pro."""
    from truememory.hooks import core

    monkeypatch.setattr(core, "_SPAWN_CAP_OVERRIDE", "")
    monkeypatch.setattr(core, "_get_memory_available_ratio", lambda: 0.5)

    monkeypatch.setattr(core, "_get_current_tier", lambda: "edge")
    assert core._get_spawn_cap() == core._SPAWN_CAP_EDGE

    monkeypatch.setattr(core, "_get_current_tier", lambda: "base")
    assert core._get_spawn_cap() == core._SPAWN_CAP_GPU

    monkeypatch.setattr(core, "_get_current_tier", lambda: "pro")
    assert core._get_spawn_cap() == core._SPAWN_CAP_GPU


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
    monkeypatch.setattr(core_mod, "register_spawned_pid", lambda pid: None)

    ingest_calls = []
    def _mock_popen(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        result = type("P", (), {"pid": 99999, "__enter__": lambda s: s, "__exit__": lambda *a: None, "stdout": ""})()
        if isinstance(cmd, list) and "truememory.ingest.cli" in " ".join(str(c) for c in cmd):
            ingest_calls.append(args)
        return result

    monkeypatch.setattr(subprocess, "Popen", _mock_popen)

    ss_mod._drain_backlog()

    assert len(ingest_calls) == 2, f"Expected exactly 2 ingest spawns, got {len(ingest_calls)}"
    remaining = list(backlog.glob("*.json"))
    assert len(remaining) == 3, f"Expected 3 remaining backlog items, got {len(remaining)}"
