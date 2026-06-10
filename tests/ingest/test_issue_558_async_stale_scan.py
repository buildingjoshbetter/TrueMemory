"""Tests for issue #558 — async stale-session scan.

The stale-session scanner (_scan_stale_sessions) walks every project dir under
~/.claude/projects/, which is expensive with 27K+ transcript files.  It must
run in a detached background process so it doesn't block SessionStart.

Covers:
  - _spawn_stale_scan launches a subprocess with start_new_session=True
  - main() calls _spawn_stale_scan (not _scan_stale_sessions directly)
  - --scan-stale flag runs _scan_stale_sessions synchronously (the subprocess entry point)
  - session_start returns quickly (not blocked by the scan)
"""
from __future__ import annotations

import io
import os
import subprocess
import sys


from truememory.ingest.hooks import session_start as ss


class TestSpawnStaleScan:
    """_spawn_stale_scan must launch a detached background process."""

    def test_spawn_calls_popen_with_start_new_session(self, monkeypatch):
        """Verify Popen is called with start_new_session=True and the
        correct command-line arguments."""
        captured = {}

        class FakePopen:
            def __init__(self, cmd, **kwargs):
                captured["cmd"] = cmd
                captured["kwargs"] = kwargs
                self.pid = 12345

        monkeypatch.setattr(subprocess, "Popen", FakePopen)

        ss._spawn_stale_scan()

        assert captured, "_spawn_stale_scan did not call Popen"
        assert captured["cmd"] == [
            sys.executable, "-m", "truememory.ingest.hooks.session_start", "--scan-stale",
        ]
        assert captured["kwargs"]["start_new_session"] is hasattr(os, "setsid")
        assert captured["kwargs"]["stdin"] == subprocess.DEVNULL

    def test_spawn_failure_does_not_raise(self, monkeypatch):
        """If Popen fails, _spawn_stale_scan logs but does not propagate."""

        def _boom(*args, **kwargs):
            raise OSError("spawn failed")

        monkeypatch.setattr(subprocess, "Popen", _boom)
        # Must not raise
        ss._spawn_stale_scan()


class TestMainDoesNotCallScanDirectly:
    """main() must use _run_maintenance_background, never _scan_stale_sessions directly."""

    def test_main_calls_maintenance_not_scan(self, monkeypatch):
        """main() must call _run_maintenance_background (background), not
        _scan_stale_sessions (synchronous)."""
        calls = {"maintenance": 0, "scan": 0}

        monkeypatch.setattr(ss, "_run_maintenance_background", lambda: calls.__setitem__("maintenance", calls["maintenance"] + 1))
        monkeypatch.setattr(ss, "_scan_stale_sessions", lambda: calls.__setitem__("scan", calls["scan"] + 1))
        monkeypatch.setattr(ss, "_drain_backlog", lambda: None)
        monkeypatch.setattr(ss, "_is_first_run", lambda: True)
        monkeypatch.setattr(ss, "_check_for_update", lambda: "")
        monkeypatch.setattr(ss, "_check_email_needed", lambda: "")
        monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
        monkeypatch.setattr("sys.stdout", io.StringIO())
        monkeypatch.delenv("TRUEMEMORY_EXTRACTION", raising=False)

        ss.main()

        assert calls["maintenance"] == 1, "main() should call _run_maintenance_background"
        assert calls["scan"] == 0, "main() should NOT call _scan_stale_sessions directly"


class TestScanStaleFlag:
    """--scan-stale is the subprocess entry point for the background scan."""

    def test_scan_stale_flag_runs_scan_and_returns(self, monkeypatch):
        """When --scan-stale is passed, main() runs _scan_stale_sessions
        synchronously and returns without doing recall or drain."""
        calls = {"scan": 0, "drain": 0, "recall": 0}

        monkeypatch.setattr(ss, "_scan_stale_sessions", lambda: calls.__setitem__("scan", calls["scan"] + 1))
        monkeypatch.setattr(ss, "_drain_backlog", lambda: calls.__setitem__("drain", calls["drain"] + 1))
        monkeypatch.setattr(ss, "recall_memories", lambda *a, **k: calls.__setitem__("recall", calls["recall"] + 1) or "")
        monkeypatch.delenv("TRUEMEMORY_EXTRACTION", raising=False)

        # Simulate --scan-stale on the command line
        monkeypatch.setattr("sys.argv", ["session_start.py", "--scan-stale"])
        monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
        captured = io.StringIO()
        monkeypatch.setattr("sys.stdout", captured)

        ss.main()

        assert calls["scan"] == 1, "--scan-stale should run the scan"
        assert calls["drain"] == 0, "--scan-stale should NOT drain backlog"
        assert calls["recall"] == 0, "--scan-stale should NOT do recall"
        assert captured.getvalue() == "", "--scan-stale should produce no stdout"


class TestSessionStartReturnsQuickly:
    """SessionStart must not be blocked by the stale scan."""

    def test_session_start_does_not_wait_for_scan(self, monkeypatch):
        """The main() function should return without waiting for the
        background scan process.  We verify by checking that Popen is
        called but .wait()/.communicate() is never called."""
        waited = {"value": False}

        class FakePopen:
            def __init__(self, cmd, **kwargs):
                self.pid = 99999

            def wait(self, *a, **k):
                waited["value"] = True
                return 0

            def communicate(self, *a, **k):
                waited["value"] = True
                return (b"", b"")

        monkeypatch.setattr(subprocess, "Popen", FakePopen)
        monkeypatch.setattr(ss, "_drain_backlog", lambda: None)
        monkeypatch.setattr(ss, "_is_first_run", lambda: True)
        monkeypatch.setattr(ss, "_check_for_update", lambda: "")
        monkeypatch.setattr(ss, "_check_email_needed", lambda: "")
        monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
        monkeypatch.setattr("sys.stdout", io.StringIO())
        monkeypatch.delenv("TRUEMEMORY_EXTRACTION", raising=False)

        ss.main()

        assert not waited["value"], "main() must not wait on the background scan process"
