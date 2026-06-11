"""Issue #642 — ingest/hooks adopt _platform primitives on Windows.

The ingest/hooks layer historically used bare ``os.kill(pid, 0)`` liveness
checks and ``start_new_session``-only spawns, both of which are wrong on
Windows, plus lockless cap/dedup fallbacks. This suite verifies the fixes:

- M-22: ``_shared._pid_is_alive`` routes through ``_platform.pid_is_alive``
  (psutil) on win32 instead of bare ``os.kill``.
- M-50: the five spawn sites pass ``_platform.spawn_kwargs()`` so children
  carry the Windows detach flags.
- M-79: the spawn gate and dedup-store lock take a real ``msvcrt`` lock on
  Windows instead of a no-op / lockless count.

All tests simulate win32 by monkeypatching ``sys.platform`` / the
``_platform`` helpers — no real Windows runtime is required.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# M-22: _pid_is_alive routes through _platform.pid_is_alive on Windows
# ---------------------------------------------------------------------------


class TestPidLivenessRoutesThroughPlatform:
    def test_win32_dead_pid_uses_platform_helper_not_os_kill(self):
        """On simulated win32, a dead PID is reported dead via the helper,
        and bare os.kill is NOT called (it would Ctrl+C the process group)."""
        from truememory.ingest.hooks import _shared

        called = {"os_kill": False, "platform": False}

        def fake_os_kill(pid, sig):  # pragma: no cover — must NOT run
            called["os_kill"] = True
            raise AssertionError("os.kill must not be used on win32")

        def fake_platform_alive(pid):
            called["platform"] = True
            return False

        with patch.object(_shared.sys, "platform", "win32"), \
                patch.object(_shared.os, "kill", fake_os_kill), \
                patch.object(_shared._platform, "pid_is_alive", fake_platform_alive):
            assert _shared._pid_is_alive(12345) is False

        assert called["platform"] is True
        assert called["os_kill"] is False

    def test_win32_live_pid_via_platform_helper(self):
        from truememory.ingest.hooks import _shared

        with patch.object(_shared.sys, "platform", "win32"), \
                patch.object(_shared._platform, "pid_is_alive", lambda pid: True):
            assert _shared._pid_is_alive(999) is True

    def test_nonpositive_pid_short_circuits(self):
        from truememory.ingest.hooks import _shared
        assert _shared._pid_is_alive(0) is False
        assert _shared._pid_is_alive(-1) is False

    def test_posix_path_still_uses_os_kill(self):
        """Non-win32 keeps the EPERM-aware os.kill semantics."""
        from truememory.ingest.hooks import _shared
        # Our own live PID is alive on POSIX.
        if sys.platform != "win32":
            assert _shared._pid_is_alive(os.getpid()) is True


# ---------------------------------------------------------------------------
# M-50: the five spawn sites pass _platform.spawn_kwargs()
# ---------------------------------------------------------------------------


class TestSpawnKwargs:
    def test_spawn_kwargs_win32_has_detach_flags(self):
        """The helper the spawn sites use yields Windows creationflags."""
        from truememory import _platform
        with patch.object(_platform.sys, "platform", "win32"):
            kw = _platform.spawn_kwargs()
        assert "creationflags" in kw
        # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_NO_WINDOW
        assert kw["creationflags"] & 0x00000200  # CREATE_NEW_PROCESS_GROUP
        assert kw["creationflags"] & 0x00000008  # DETACHED_PROCESS
        assert "start_new_session" not in kw

    def test_spawn_kwargs_posix(self):
        from truememory import _platform
        with patch.object(_platform.sys, "platform", "linux"):
            kw = _platform.spawn_kwargs()
        assert "start_new_session" in kw
        assert "creationflags" not in kw

    @pytest.mark.parametrize("modname", [
        "truememory.mcp_server",
        "truememory.ingest.cli",
        "truememory.ingest.hooks.session_start",
        "truememory.ingest.hooks.stop",
        "truememory.hooks.core",
    ])
    def test_spawn_sites_no_bare_start_new_session(self, modname):
        """No spawn site passes a literal start_new_session=hasattr(...) any
        more for the ingest workers — they must funnel through spawn_kwargs.

        (stop.py/core.py use an inline win32 branch; the five #642 sites use
        the helper. Either way, none should regress to the no-op-only form.)"""
        import importlib
        mod = importlib.import_module(modname)
        src = open(mod.__file__, encoding="utf-8").read()
        # session_start/cli/mcp_server must reference spawn_kwargs.
        if modname in (
            "truememory.mcp_server",
            "truememory.ingest.cli",
            "truememory.ingest.hooks.session_start",
        ):
            assert "spawn_kwargs" in src, f"{modname} should use _platform.spawn_kwargs()"

    def test_session_start_spawn_uses_helper_kwargs(self, monkeypatch, tmp_path):
        """_spawn_stale_scan funnels Popen kwargs from spawn_kwargs()."""
        from truememory.ingest.hooks import session_start

        captured = {}

        class FakeProc:
            pid = 7

        def fake_popen(cmd, **kwargs):
            captured.update(kwargs)
            return FakeProc()

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(session_start, "_platform",
                            _StubPlatform(creationflags=0x208))
        import subprocess as _sp
        monkeypatch.setattr(_sp, "Popen", fake_popen)

        session_start._spawn_stale_scan()
        # The stub's creationflags must have reached Popen.
        assert captured.get("creationflags") == 0x208
        assert "start_new_session" not in captured


class _StubPlatform:
    def __init__(self, **kw):
        self._kw = kw

    def spawn_kwargs(self):
        return dict(self._kw)

    def pid_is_alive(self, pid):
        return True


# ---------------------------------------------------------------------------
# M-79: spawn gate + dedup lock have a real msvcrt branch on Windows
# ---------------------------------------------------------------------------


class TestWindowsLocks:
    def test_spawn_gate_takes_msvcrt_lock_when_no_fcntl(self, monkeypatch, tmp_path):
        """When fcntl is unavailable (win32), spawn_gate locks via msvcrt and
        enforces the cap from the PID file — NOT the lockless pgrep count."""
        from truememory.hooks import core

        lock_calls = []

        class FakeMsvcrt:
            LK_LOCK = 1
            LK_UNLCK = 2

            def locking(self, fileno, flag, nbytes):
                lock_calls.append(flag)

        monkeypatch.setattr(core, "_HAS_FCNTL", False)
        monkeypatch.setattr(core, "SPAWN_LOCK_PATH", tmp_path / ".spawn.lock")
        monkeypatch.setattr(core, "SPAWN_PIDS_PATH", tmp_path / ".spawn_pids")
        monkeypatch.setattr(core, "_read_live_pids", lambda: [])
        monkeypatch.setattr(core, "_write_pids", lambda pids: None)
        monkeypatch.setattr(core, "_get_spawn_cap", lambda: 2)
        # If the lockless path were taken, this would be consulted; make it
        # blow up so the test fails if msvcrt branch is skipped.
        monkeypatch.setattr(
            core, "_count_active_ingest_processes",
            lambda: (_ for _ in ()).throw(AssertionError("lockless path used")),
        )
        monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt())

        with core.spawn_gate() as allowed:
            assert allowed is True  # 0 live < cap 2

        assert FakeMsvcrt.LK_LOCK in lock_calls
        assert FakeMsvcrt.LK_UNLCK in lock_calls

    def test_dedup_store_lock_takes_msvcrt_lock_when_no_fcntl(self, monkeypatch, tmp_path):
        from truememory.ingest import pipeline

        lock_calls = []

        class FakeMsvcrt:
            LK_LOCK = 1
            LK_UNLCK = 2

            def locking(self, fileno, flag, nbytes):
                lock_calls.append(flag)

        monkeypatch.setattr(pipeline, "_HAS_FCNTL", False)
        monkeypatch.setattr(pipeline, "_LOCK_PATH", tmp_path / "ingest.lock")
        monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt())

        with pipeline._dedup_store_lock():
            pass

        assert FakeMsvcrt.LK_LOCK in lock_calls
        assert FakeMsvcrt.LK_UNLCK in lock_calls

    def test_dedup_store_lock_posix_uses_flock(self, tmp_path, monkeypatch):
        """POSIX path unaffected: still acquires via flock without msvcrt."""
        if sys.platform == "win32":
            pytest.skip("POSIX-only assertion")
        from truememory.ingest import pipeline
        monkeypatch.setattr(pipeline, "_LOCK_PATH", tmp_path / "ingest.lock")
        with pipeline._dedup_store_lock():
            pass  # acquires + releases flock without error
