"""Regression locks for Hunter F21 + F33 — stop-hook safety.

F21: `_run_background_ingestion` now refuses to Popen when `SPAWN_CAP`
concurrent ingest processes are already live — pre-fix, N parallel Stop
hooks loaded N embedding models (~600MB each on Pro) and OOM-killed on
laptops.

F33: when Popen itself fails, the hook now writes a JSON marker to
`BACKLOG_DIR` instead of falling back to synchronous inline ingestion
(the old path blocked Claude Code's shutdown for 10–60s).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# F21 — spawn cap
# ---------------------------------------------------------------------------


def test_spawn_cap_blocks_when_at_cap(monkeypatch, tmp_path):
    """When `_count_active_ingest_processes()` reports >= SPAWN_CAP,
    `_run_background_ingestion` must NOT call Popen and must write a
    backlog marker explaining why."""
    from truememory.ingest.hooks import stop as stop_mod

    monkeypatch.setattr(stop_mod, "SPAWN_CAP", 2)
    monkeypatch.setattr(stop_mod, "_count_active_ingest_processes", lambda: 2)
    monkeypatch.setattr(stop_mod, "BACKLOG_DIR", tmp_path / "backlog")
    monkeypatch.setattr(stop_mod, "TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(stop_mod, "LOG_DIR", tmp_path / "logs")

    popen_calls = []

    def _record_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return type("DummyProc", (), {"pid": 0})()

    monkeypatch.setattr(subprocess, "Popen", _record_popen)

    stop_mod._run_background_ingestion(
        transcript_path="/fake/transcript.json",
        session_id="sess-at-cap",
        user_id="alice",
        db_path="/tmp/fake.db",
    )

    assert popen_calls == [], (
        "F21 regression: Popen was called despite cap reached"
    )
    markers = list((tmp_path / "backlog").glob("*.json"))
    assert len(markers) == 1
    data = json.loads(markers[0].read_text())
    assert data["session_id"] == "sess-at-cap"
    assert "spawn_cap_reached" in data["reason"]


def test_spawn_cap_allows_spawn_under_cap(monkeypatch, tmp_path):
    """Under the cap, Popen must be called and no backlog marker written."""
    from truememory.ingest.hooks import stop as stop_mod

    monkeypatch.setattr(stop_mod, "SPAWN_CAP", 4)
    monkeypatch.setattr(stop_mod, "_count_active_ingest_processes", lambda: 1)
    monkeypatch.setattr(stop_mod, "BACKLOG_DIR", tmp_path / "backlog")
    monkeypatch.setattr(stop_mod, "TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(stop_mod, "LOG_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir()

    popen_calls = []

    def _record_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return type("DummyProc", (), {"pid": 123})()

    monkeypatch.setattr(subprocess, "Popen", _record_popen)

    stop_mod._run_background_ingestion(
        transcript_path="/fake/transcript.json",
        session_id="sess-ok",
        user_id="alice",
        db_path="/tmp/fake.db",
    )

    assert len(popen_calls) == 1, "Popen must be called when under the cap"
    backlog = tmp_path / "backlog"
    assert not backlog.exists() or not list(backlog.glob("*.json"))


def test_count_active_ingest_processes_windows_noop(monkeypatch):
    """On Windows, `_count_active_ingest_processes` returns 0 so the cap
    check doesn't become a hard fence (pgrep is POSIX-only)."""
    from truememory.ingest.hooks import stop as stop_mod
    import sys as _sys

    monkeypatch.setattr(_sys, "platform", "win32")
    # Even if pgrep would exist, the win32 branch short-circuits
    assert stop_mod._count_active_ingest_processes() == 0


def test_count_active_ingest_processes_pgrep_missing(monkeypatch):
    """If pgrep isn't on PATH (sandboxed runtime), return 0 rather than
    crash the hook."""
    from truememory.ingest.hooks import stop as stop_mod
    import sys as _sys

    monkeypatch.setattr(_sys, "platform", "linux")

    def _no_pgrep(*args, **kwargs):
        raise FileNotFoundError("pgrep not found")

    monkeypatch.setattr(subprocess, "run", _no_pgrep)
    assert stop_mod._count_active_ingest_processes() == 0


# ---------------------------------------------------------------------------
# F33 — Popen failure → backlog marker, NOT inline ingestion
# ---------------------------------------------------------------------------


def test_popen_failure_queues_backlog_not_inline(monkeypatch, tmp_path):
    """When subprocess.Popen raises (disk full, permission denied, etc.),
    the hook must write a backlog marker and NOT fall back to inline
    ingestion — that path blocks Claude Code's shutdown."""
    from truememory.ingest.hooks import stop as stop_mod

    monkeypatch.setattr(stop_mod, "SPAWN_CAP", 99)
    monkeypatch.setattr(stop_mod, "_count_active_ingest_processes", lambda: 0)
    monkeypatch.setattr(stop_mod, "BACKLOG_DIR", tmp_path / "backlog")
    monkeypatch.setattr(stop_mod, "TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(stop_mod, "LOG_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir()

    def _popen_boom(*args, **kwargs):
        raise OSError("simulated: disk full for log file")

    monkeypatch.setattr(subprocess, "Popen", _popen_boom)

    inline_called = {"n": 0}

    def _spy_inline(*args, **kwargs):
        inline_called["n"] += 1
        return {}

    # If the old inline-fallback code path had survived, it would call
    # `truememory.ingest.ingest`. Monkeypatch that name on the import
    # site (truememory.ingest.__init__) so any call would trip the spy.
    import truememory.ingest as _ingest_pkg
    monkeypatch.setattr(_ingest_pkg, "ingest", _spy_inline, raising=False)

    # Must not raise, must not call inline ingest, must queue to backlog.
    stop_mod._run_background_ingestion(
        transcript_path="/fake/transcript.json",
        session_id="sess-popen-fail",
        user_id="alice",
        db_path="/tmp/fake.db",
    )

    assert inline_called["n"] == 0, (
        "F33 regression: inline ingestion was called after Popen failure; "
        "this blocks Claude Code shutdown"
    )
    markers = list((tmp_path / "backlog").glob("*.json"))
    assert len(markers) == 1
    data = json.loads(markers[0].read_text())
    assert data["session_id"] == "sess-popen-fail"
    assert "popen_failed" in data["reason"]
    assert "OSError" in data["reason"]


def test_backlog_write_failure_is_swallowed(monkeypatch, tmp_path):
    """Best-effort: if the backlog directory itself is unwritable (disk
    full, chmod 000), the hook logs an error but must NOT raise — the
    session's memories are just lost, and the primary contract (don't
    block Claude Code) is preserved."""
    from truememory.ingest.hooks import stop as stop_mod

    # Point BACKLOG_DIR at a path that will fail on mkdir (nested inside
    # a file, not a dir)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    monkeypatch.setattr(stop_mod, "BACKLOG_DIR", blocker / "subdir" / "backlog")

    # Should not raise
    stop_mod._queue_to_backlog(
        transcript_path="/fake/transcript.json",
        session_id="sess-backlog-fail",
        user_id="alice",
        db_path="/tmp/fake.db",
        reason="test",
    )


def test_backlog_marker_schema(monkeypatch, tmp_path):
    """Document the backlog marker schema — a later session_start drain
    path will rely on these keys."""
    from truememory.ingest.hooks import stop as stop_mod

    monkeypatch.setattr(stop_mod, "BACKLOG_DIR", tmp_path / "backlog")
    stop_mod._queue_to_backlog(
        transcript_path="/conv/transcript.json",
        session_id="sess-abc",
        user_id="alice",
        db_path="/tmp/alice.db",
        reason="test:reason",
    )

    marker = tmp_path / "backlog" / "sess-abc.json"
    assert marker.exists()
    data = json.loads(marker.read_text())
    required_keys = {
        "transcript_path", "session_id", "user_id",
        "db_path", "queued_at", "reason",
    }
    assert required_keys <= set(data.keys())
    assert data["session_id"] == "sess-abc"
    assert data["reason"] == "test:reason"
    # ISO-8601 timestamp
    assert "T" in data["queued_at"]


def test_empty_session_id_still_queues(monkeypatch, tmp_path):
    """If session_id is missing, still write a marker (under 'unknown.json')
    so we don't silently drop the memories."""
    from truememory.ingest.hooks import stop as stop_mod

    monkeypatch.setattr(stop_mod, "BACKLOG_DIR", tmp_path / "backlog")
    stop_mod._queue_to_backlog(
        transcript_path="/conv/transcript.json",
        session_id="",
        user_id="",
        db_path="",
        reason="no-session-id",
    )
    assert (tmp_path / "backlog" / "unknown.json").exists()


def test_spawn_cap_env_var_override(monkeypatch):
    """`TRUEMEMORY_INGEST_SPAWN_CAP` env var must control the cap."""
    monkeypatch.setenv("TRUEMEMORY_INGEST_SPAWN_CAP", "5")
    import importlib
    from truememory.ingest.hooks import stop as stop_mod
    importlib.reload(stop_mod)
    assert stop_mod.SPAWN_CAP == 5

    monkeypatch.setenv("TRUEMEMORY_INGEST_SPAWN_CAP", "1")
    importlib.reload(stop_mod)
    assert stop_mod.SPAWN_CAP == 1


def test_backlog_dir_env_var_override(monkeypatch, tmp_path):
    """`TRUEMEMORY_BACKLOG_DIR` env var must control the path."""
    override = tmp_path / "custom_backlog"
    monkeypatch.setenv("TRUEMEMORY_BACKLOG_DIR", str(override))
    import importlib
    from truememory.ingest.hooks import stop as stop_mod
    importlib.reload(stop_mod)
    assert Path(str(stop_mod.BACKLOG_DIR)) == override
