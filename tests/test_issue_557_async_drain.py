"""Tests for issue #557: _drain_backlog() runs async in background subprocess."""

import os
import subprocess
import sys
import time
from unittest import mock



def test_maintenance_spawns_background_subprocess(tmp_path, monkeypatch):
    """_run_maintenance_background() spawns a Popen child, not a blocking call."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".truememory" / "logs").mkdir(parents=True)

    spawned = {}

    class FakeProc:
        pid = 12345

    def fake_popen(cmd, **kwargs):
        spawned["cmd"] = cmd
        spawned["kwargs"] = kwargs
        return FakeProc()

    with mock.patch("subprocess.Popen", side_effect=fake_popen):
        from truememory.ingest.hooks.session_start import _run_maintenance_background
        _run_maintenance_background()

    assert "cmd" in spawned, "Popen was never called"
    assert spawned["cmd"][0] == sys.executable
    assert spawned["cmd"][1] == "-c"
    # The inline script should import and run both maintenance functions.
    script = spawned["cmd"][2]
    assert "_drain_backlog" in script
    assert "_scan_stale_sessions" in script
    # Must detach from parent session group so it survives hook exit.
    assert spawned["kwargs"].get("start_new_session") is hasattr(os, "setsid")
    # stdout/stderr must be redirected (not inherited) to avoid blocking.
    assert spawned["kwargs"].get("stdin") == subprocess.DEVNULL
    assert spawned["kwargs"]["stdout"] is not None
    # Child must be tagged so it doesn't recurse into spawning another child.
    env = spawned["kwargs"].get("env", {})
    assert env.get("TRUEMEMORY_MAINTENANCE_CHILD") == "1"


def test_maintenance_skipped_on_popen_failure(tmp_path, monkeypatch):
    """If Popen fails, maintenance is SKIPPED, not run inline (issue #644 / M-38).

    Spawn failures happen precisely when the system is unhealthy (fd
    exhaustion, OOM, disk full); running drain+scan synchronously here would
    block SessionStart and delay recall injection at the worst time. The next
    session retries maintenance via its own spawn.
    """
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".truememory" / "logs").mkdir(parents=True)

    calls = []

    with mock.patch("subprocess.Popen", side_effect=OSError("no such file")), \
         mock.patch("truememory.ingest.hooks.session_start._drain_backlog", side_effect=lambda: calls.append("drain")), \
         mock.patch("truememory.ingest.hooks.session_start._scan_stale_sessions", side_effect=lambda: calls.append("scan")):
        from truememory.ingest.hooks.session_start import _run_maintenance_background
        _run_maintenance_background()

    assert calls == [], "maintenance must NOT run inline on spawn failure (M-38)"


def test_session_start_returns_quickly(tmp_path, monkeypatch):
    """main() should return in < 1s even if drain would be slow."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    # Create the onboarded marker so it takes the recall path (which we stub).
    (tmp_path / ".truememory").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".truememory" / ".onboarded").touch()
    (tmp_path / ".truememory" / "logs").mkdir(parents=True, exist_ok=True)

    # Stub stdin for main().
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{}"))

    # Mock the background spawn to be instant (don't actually fork).
    with mock.patch("truememory.ingest.hooks.session_start._run_maintenance_background") as bg, \
         mock.patch("truememory.ingest.hooks.session_start.recall_memories", return_value=""):
        t0 = time.monotonic()
        from truememory.ingest.hooks.session_start import main
        main()
        elapsed = time.monotonic() - t0

    bg.assert_called_once()
    assert elapsed < 1.0, f"main() took {elapsed:.2f}s — should be < 1s"


def test_drain_still_completes_in_background(tmp_path, monkeypatch):
    """Integration: the spawned subprocess actually runs drain logic."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    # Create a sentinel file that the subprocess will create to prove it ran.
    sentinel = tmp_path / "drain_ran"
    script = f"""
import pathlib
pathlib.Path({str(sentinel)!r}).write_text("ok")
"""
    log_dir = tmp_path / ".truememory" / "logs"
    log_dir.mkdir(parents=True)
    log_file = open(log_dir / "test.log", "w")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        log_file.close()

    proc.wait(timeout=10)
    assert sentinel.exists(), "Background subprocess did not complete"
    assert sentinel.read_text() == "ok"


def test_main_calls_maintenance_not_drain_directly(monkeypatch, tmp_path):
    """main() must call _run_maintenance_background, not _drain_backlog directly."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".truememory").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".truememory" / ".onboarded").touch()
    (tmp_path / ".truememory" / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{}"))

    with mock.patch("truememory.ingest.hooks.session_start._run_maintenance_background") as bg, \
         mock.patch("truememory.ingest.hooks.session_start._drain_backlog") as drain, \
         mock.patch("truememory.ingest.hooks.session_start._scan_stale_sessions") as scan, \
         mock.patch("truememory.ingest.hooks.session_start.recall_memories", return_value=""):
        from truememory.ingest.hooks.session_start import main
        main()

    bg.assert_called_once()
    drain.assert_not_called()
    scan.assert_not_called()
