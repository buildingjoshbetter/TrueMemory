"""Tests for the dynamic spawn cap and flock-based spawn gate.

The spawn gate serializes check-then-spawn decisions via a file lock.
The dynamic cap adapts to tier (Edge=CPU, Base/Pro=GPU) and system health
(memory pressure, swap growth, ramp-up/ramp-down).
"""
from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Spawn gate tests
# ---------------------------------------------------------------------------


def test_spawn_gate_yields_true_under_cap(tmp_path, monkeypatch):
    """When fewer than cap processes are active, gate yields True."""
    from truememory.hooks import core

    monkeypatch.setattr(core, "_get_spawn_cap", lambda: 3)
    monkeypatch.setattr(core, "SPAWN_LOCK_PATH", tmp_path / ".spawn.lock")
    monkeypatch.setattr(core, "SPAWN_PIDS_PATH", tmp_path / ".spawn_pids")
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


# ---------------------------------------------------------------------------
# Env var override
# ---------------------------------------------------------------------------


def test_spawn_cap_env_var_override(monkeypatch):
    """TRUEMEMORY_SPAWN_CAP env var overrides everything."""
    from truememory.hooks import core

    monkeypatch.setenv("TRUEMEMORY_SPAWN_CAP", "7")
    assert core._get_spawn_cap() == 7


# ---------------------------------------------------------------------------
# Edge tier: CPU-based ceiling
# ---------------------------------------------------------------------------


def test_edge_ceiling_by_cpu_cores(tmp_path, monkeypatch):
    """Edge ceiling = physical_cores - 1, capped at 8."""
    from truememory.hooks import core

    monkeypatch.delenv("TRUEMEMORY_SPAWN_CAP", raising=False)
    monkeypatch.delenv("TRUEMEMORY_INGEST_SPAWN_CAP", raising=False)
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 90)
    monkeypatch.setattr(core, "_get_swap_used_gb", lambda: 0.0)
    monkeypatch.setattr(core, "_get_current_tier", lambda: "edge")
    monkeypatch.setattr(core, "_SPAWN_CAP_STATE_PATH", tmp_path / ".state")

    # 10-core Mac — ceiling = min(9, 8) = 8
    monkeypatch.setattr(core, "_get_physical_cores", lambda: 10)
    caps = []
    for _ in range(15):
        caps.append(core._get_spawn_cap())
    assert max(caps) == 8, f"Edge 10-core should max at 8, got {max(caps)}"

    # 4-core Air — ceiling = min(3, 8) = 3
    monkeypatch.setattr(core, "_get_physical_cores", lambda: 4)
    # Reset state
    if (tmp_path / ".state").exists():
        (tmp_path / ".state").unlink()
    caps = []
    for _ in range(10):
        caps.append(core._get_spawn_cap())
    assert max(caps) == 3, f"Edge 4-core should max at 3, got {max(caps)}"


# ---------------------------------------------------------------------------
# Base/Pro tier: unified memory ceiling
# ---------------------------------------------------------------------------


def test_base_ceiling_by_unified_memory(tmp_path, monkeypatch):
    """Base ceiling = (unified_memory - 2GB) / 1.0GB, capped at 6."""
    from truememory.hooks import core

    monkeypatch.delenv("TRUEMEMORY_SPAWN_CAP", raising=False)
    monkeypatch.delenv("TRUEMEMORY_INGEST_SPAWN_CAP", raising=False)
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 90)
    monkeypatch.setattr(core, "_get_swap_used_gb", lambda: 0.0)
    monkeypatch.setattr(core, "_get_current_tier", lambda: "base")
    monkeypatch.setattr(core, "_SPAWN_CAP_STATE_PATH", tmp_path / ".state")

    # 8GB Mac — (8-2)/1.0 = 6, capped at 6
    monkeypatch.setattr(core, "_get_total_memory_gb", lambda: 8)
    caps = []
    for _ in range(10):
        caps.append(core._get_spawn_cap())
    assert max(caps) == 6, f"Base 8GB should max at 6, got {max(caps)}"

    # 4GB Mac — (4-2)/1.0 = 2
    monkeypatch.setattr(core, "_get_total_memory_gb", lambda: 4)
    if (tmp_path / ".state").exists():
        (tmp_path / ".state").unlink()
    caps = []
    for _ in range(5):
        caps.append(core._get_spawn_cap())
    assert max(caps) == 2, f"Base 4GB should max at 2, got {max(caps)}"


# ---------------------------------------------------------------------------
# Memory pressure detection
# ---------------------------------------------------------------------------


def test_memory_free_pct_classification():
    """Free percentage maps to correct pressure levels."""
    from truememory.hooks.core import _classify_memory_pressure

    assert _classify_memory_pressure(90) == "normal"
    assert _classify_memory_pressure(50) == "normal"
    assert _classify_memory_pressure(40) == "normal"
    assert _classify_memory_pressure(39) == "warn"
    assert _classify_memory_pressure(20) == "warn"
    assert _classify_memory_pressure(15) == "warn"
    assert _classify_memory_pressure(14) == "critical"
    assert _classify_memory_pressure(5) == "critical"
    assert _classify_memory_pressure(0) == "critical"


def test_warn_halves_cap_after_hysteresis(tmp_path, monkeypatch):
    """Sustained warn (2+ consecutive) halves the cap."""
    from truememory.hooks import core

    monkeypatch.delenv("TRUEMEMORY_SPAWN_CAP", raising=False)
    monkeypatch.delenv("TRUEMEMORY_INGEST_SPAWN_CAP", raising=False)
    monkeypatch.setattr(core, "_get_current_tier", lambda: "edge")
    monkeypatch.setattr(core, "_get_physical_cores", lambda: 10)
    monkeypatch.setattr(core, "_get_swap_used_gb", lambda: 0.0)
    monkeypatch.setattr(core, "_SPAWN_CAP_STATE_PATH", tmp_path / ".state")

    # Ramp up to ceiling
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 90)
    for _ in range(15):
        core._get_spawn_cap()

    # Warn 1 — no action (hysteresis)
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 30)
    cap1 = core._get_spawn_cap()
    assert cap1 == 8, f"Single warn should not reduce, got {cap1}"

    # Warn 2 — halve
    cap2 = core._get_spawn_cap()
    assert cap2 == 4, f"Sustained warn should halve to 4, got {cap2}"


def test_critical_drops_to_floor(tmp_path, monkeypatch):
    """Critical pressure drops to floor immediately."""
    from truememory.hooks import core

    monkeypatch.delenv("TRUEMEMORY_SPAWN_CAP", raising=False)
    monkeypatch.delenv("TRUEMEMORY_INGEST_SPAWN_CAP", raising=False)
    monkeypatch.setattr(core, "_get_current_tier", lambda: "edge")
    monkeypatch.setattr(core, "_get_physical_cores", lambda: 10)
    monkeypatch.setattr(core, "_get_swap_used_gb", lambda: 0.0)
    monkeypatch.setattr(core, "_SPAWN_CAP_STATE_PATH", tmp_path / ".state")

    # Ramp up
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 90)
    for _ in range(15):
        core._get_spawn_cap()

    # Critical — immediate floor
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 10)
    cap = core._get_spawn_cap()
    assert cap == 1, f"Critical should drop to 1, got {cap}"


# ---------------------------------------------------------------------------
# Swap growth detection
# ---------------------------------------------------------------------------


def test_swap_growth_triggers_emergency(tmp_path, monkeypatch):
    """Growing swap (delta > 0.5GB) triggers emergency floor."""
    from truememory.hooks import core

    monkeypatch.delenv("TRUEMEMORY_SPAWN_CAP", raising=False)
    monkeypatch.delenv("TRUEMEMORY_INGEST_SPAWN_CAP", raising=False)
    monkeypatch.setattr(core, "_get_current_tier", lambda: "edge")
    monkeypatch.setattr(core, "_get_physical_cores", lambda: 10)
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 90)
    monkeypatch.setattr(core, "_SPAWN_CAP_STATE_PATH", tmp_path / ".state")

    # First tick: swap at 2GB (establishes baseline)
    monkeypatch.setattr(core, "_get_swap_used_gb", lambda: 2.0)
    core._get_spawn_cap()

    # Second tick: swap jumped to 3GB (growth > 0.5GB)
    monkeypatch.setattr(core, "_get_swap_used_gb", lambda: 3.0)
    cap = core._get_spawn_cap()
    assert cap == 1, f"Growing swap should drop to floor, got {cap}"


def test_stable_swap_does_not_trigger(tmp_path, monkeypatch):
    """Historical swap (not growing) should NOT trigger emergency."""
    from truememory.hooks import core

    monkeypatch.delenv("TRUEMEMORY_SPAWN_CAP", raising=False)
    monkeypatch.delenv("TRUEMEMORY_INGEST_SPAWN_CAP", raising=False)
    monkeypatch.setattr(core, "_get_current_tier", lambda: "edge")
    monkeypatch.setattr(core, "_get_physical_cores", lambda: 10)
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 90)
    monkeypatch.setattr(core, "_SPAWN_CAP_STATE_PATH", tmp_path / ".state")

    # Multiple ticks with stable 4GB swap — should ramp up normally
    monkeypatch.setattr(core, "_get_swap_used_gb", lambda: 4.0)
    caps = []
    for _ in range(10):
        caps.append(core._get_spawn_cap())
    assert max(caps) > 1, f"Stable swap should allow ramp-up, got max {max(caps)}"


# ---------------------------------------------------------------------------
# Ramp-up persistence across processes
# ---------------------------------------------------------------------------


def test_ramp_up_persists_across_calls(tmp_path, monkeypatch):
    """Cap state persists to disk so cascade processes inherit it."""
    from truememory.hooks import core

    monkeypatch.delenv("TRUEMEMORY_SPAWN_CAP", raising=False)
    monkeypatch.delenv("TRUEMEMORY_INGEST_SPAWN_CAP", raising=False)
    monkeypatch.setattr(core, "_get_current_tier", lambda: "edge")
    monkeypatch.setattr(core, "_get_physical_cores", lambda: 10)
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 90)
    monkeypatch.setattr(core, "_get_swap_used_gb", lambda: 0.0)
    monkeypatch.setattr(core, "_SPAWN_CAP_STATE_PATH", tmp_path / ".state")

    # Ramp up several ticks
    caps = []
    for _ in range(5):
        caps.append(core._get_spawn_cap())

    # Verify state was written and reflects ramp-up
    assert (tmp_path / ".state").exists(), "State file should be written"
    import json as _json
    state = _json.loads((tmp_path / ".state").read_text())

    # The persisted cap should match what the last call returned
    assert state["cap"] == caps[-1], f"State cap={state['cap']} should match last returned cap={caps[-1]}"

    # Next call should continue ramp-up from persisted state
    next_cap = core._get_spawn_cap()
    assert next_cap == caps[-1] + 1, f"Should ramp from {caps[-1]} to {caps[-1]+1}, got {next_cap}"
    assert next_cap <= 8, "Should never exceed Edge hard ceiling of 8"


def test_state_expires_after_timeout(tmp_path, monkeypatch):
    """Stale state file (e.g., after reboot) is ignored."""
    from truememory.hooks import core

    monkeypatch.delenv("TRUEMEMORY_SPAWN_CAP", raising=False)
    monkeypatch.delenv("TRUEMEMORY_INGEST_SPAWN_CAP", raising=False)
    monkeypatch.setattr(core, "_get_current_tier", lambda: "edge")
    monkeypatch.setattr(core, "_get_physical_cores", lambda: 10)
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 90)
    monkeypatch.setattr(core, "_get_swap_used_gb", lambda: 0.0)
    monkeypatch.setattr(core, "_SPAWN_CAP_STATE_PATH", tmp_path / ".state")

    # Write an old state
    import json as _json
    (tmp_path / ".state").write_text(_json.dumps({
        "cap": 8,
        "warn_count": 0,
        "last_swap_gb": 0.0,
        "timestamp": 0,  # epoch = very old
    }))

    # Should start from floor since state expired, then ramp to 2
    cap1 = core._get_spawn_cap()  # reads expired state → floor (1), saves, ramps to 2
    assert cap1 <= 2, f"Expired state should start near floor, got {cap1}"
    # Verify the state was reset (not still showing 8)
    state = _json_load(tmp_path / ".state")
    assert state["cap"] <= 2, f"State should be reset near floor, got {state['cap']}"


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------


def test_single_warn_does_not_reduce(tmp_path, monkeypatch):
    """A single warn reading should NOT reduce the cap."""
    from truememory.hooks import core

    monkeypatch.delenv("TRUEMEMORY_SPAWN_CAP", raising=False)
    monkeypatch.delenv("TRUEMEMORY_INGEST_SPAWN_CAP", raising=False)
    monkeypatch.setattr(core, "_get_current_tier", lambda: "edge")
    monkeypatch.setattr(core, "_get_physical_cores", lambda: 10)
    monkeypatch.setattr(core, "_get_swap_used_gb", lambda: 0.0)
    monkeypatch.setattr(core, "_SPAWN_CAP_STATE_PATH", tmp_path / ".state")

    # Ramp up to 8
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 90)
    for _ in range(15):
        core._get_spawn_cap()

    # Single warn
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 30)
    cap = core._get_spawn_cap()
    assert cap == 8, f"Single warn should not reduce, got {cap}"

    # Back to normal — warn count resets
    monkeypatch.setattr(core, "_get_memory_free_pct", lambda: 90)
    cap = core._get_spawn_cap()
    state = _json_load(tmp_path / ".state")
    assert state["warn_count"] == 0, "Warn count should reset"


def _json_load(path):
    import json as _json
    return _json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Drain backlog integration
# ---------------------------------------------------------------------------


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
