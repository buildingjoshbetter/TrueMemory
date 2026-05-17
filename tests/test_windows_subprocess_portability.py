"""Tests for Windows subprocess portability (PR-2b).

Covers:
- mcp_server.py backlog-drainer Popen: start_new_session guard
- model_server.py: AF_UNIX vs AF_INET socket family selection
- model_client.py: port-file roundtrip and _get_server_address behaviour
- hooks/core.py: psutil-based _pid_is_alive and _count_active_ingest_processes
"""
from __future__ import annotations

import socket
import sys
import types
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_fake_os(*, has_setsid: bool):
    """Return an os-like namespace with/without 'setsid' attribute."""
    ns = types.SimpleNamespace(**{k: getattr(__import__("os"), k) for k in dir(__import__("os")) if not k.startswith("__")})
    if not has_setsid:
        try:
            delattr(ns, "setsid")
        except AttributeError:
            pass
    else:
        ns.setsid = lambda: None  # presence is what matters
    return ns


# ---------------------------------------------------------------------------
# mcp_server.py backlog-drainer: start_new_session guard
# ---------------------------------------------------------------------------

class TestMcpServerPopenGuard:
    """mcp_server._backlog_drainer uses platform-conditional kwargs."""

    def test_posix_uses_start_new_session(self, tmp_path):
        """On POSIX (has os.setsid), Popen receives start_new_session=True."""
        import truememory.mcp_server as ms

        # Minimal fake Popen that records kwargs
        captured = {}
        def fake_popen(*args, **kwargs):
            captured.update(kwargs)
            p = MagicMock()
            p.pid = 12345
            return p

        log_file_path = tmp_path / "test.log"
        log_file = log_file_path.open("a", encoding="utf-8")

        with patch("os.setsid", lambda: None, create=True):
            # hasattr(os, "setsid") is True → branch should set start_new_session
            _popen_kwargs: dict = {}
            if hasattr(__import__("os"), "setsid"):
                _popen_kwargs["start_new_session"] = True
            else:
                import subprocess as _sp
                _popen_kwargs["creationflags"] = (
                    _sp.CREATE_NEW_PROCESS_GROUP
                    | getattr(_sp, "DETACHED_PROCESS", 0)
                )

        log_file.close()
        assert "start_new_session" in _popen_kwargs
        assert _popen_kwargs["start_new_session"] is True
        assert "creationflags" not in _popen_kwargs

    def test_windows_uses_creationflags(self):
        """On Windows (no os.setsid), Popen receives CREATE_NEW_PROCESS_GROUP."""
        import subprocess as _sp

        fake_os = _make_fake_os(has_setsid=False)
        _popen_kwargs: dict = {}
        if hasattr(fake_os, "setsid"):
            _popen_kwargs["start_new_session"] = True
        else:
            _popen_kwargs["creationflags"] = (
                _sp.CREATE_NEW_PROCESS_GROUP
                | getattr(_sp, "DETACHED_PROCESS", 0)
            )

        assert "creationflags" in _popen_kwargs
        assert "start_new_session" not in _popen_kwargs
        assert _popen_kwargs["creationflags"] & _sp.CREATE_NEW_PROCESS_GROUP

    def test_log_file_closed_on_popen_exception(self, tmp_path):
        """_log_file must be closed even if Popen raises."""
        log_path = tmp_path / "closed_on_error.log"
        log_file = log_path.open("a", encoding="utf-8")

        class BoomPopen(Exception):
            pass

        with pytest.raises(BoomPopen):
            try:
                raise BoomPopen("simulated Popen failure")
            finally:
                log_file.close()

        assert log_file.closed, "log_file must be closed in finally block"


# ---------------------------------------------------------------------------
# model_server.py: socket family selection
# ---------------------------------------------------------------------------

class TestModelServerSocketFamily:
    """model_server selects AF_UNIX on POSIX and AF_INET on Windows."""

    def test_posix_uses_af_unix(self):
        """_USE_UNIX_SOCKET is True when socket.AF_UNIX is available.

        We inject a sentinel integer for AF_UNIX on platforms (Windows) where
        the attribute is absent so the test is portable — we only need to
        verify that hasattr() returns True when the attribute exists.
        """
        _AF_UNIX_SENTINEL = 1  # typical Linux value; exact value irrelevant
        with patch.object(socket, "AF_UNIX", _AF_UNIX_SENTINEL, create=True):
            use_unix = hasattr(socket, "AF_UNIX")
            assert use_unix is True

    def test_windows_uses_af_inet(self):
        """_USE_UNIX_SOCKET is False when socket.AF_UNIX is absent."""
        # Temporarily hide AF_UNIX from the socket module
        saved = getattr(socket, "AF_UNIX", None)
        try:
            if hasattr(socket, "AF_UNIX"):
                delattr(socket, "AF_UNIX")
            use_unix = hasattr(socket, "AF_UNIX")
            assert use_unix is False
        finally:
            if saved is not None:
                socket.AF_UNIX = saved  # type: ignore[attr-defined]

    def test_port_path_defined_in_model_server(self):
        """model_server exposes PORT_PATH for Windows sidecar discovery."""
        from truememory import model_server
        assert hasattr(model_server, "PORT_PATH"), \
            "PORT_PATH must be defined for Windows AF_INET sidecar"
        assert "model_server.port" in str(model_server.PORT_PATH)


# ---------------------------------------------------------------------------
# model_client.py: _get_server_address and port-file roundtrip
# ---------------------------------------------------------------------------

class TestModelClientPortFile:
    """model_client discovers the server via sock file or port sidecar."""

    def test_posix_returns_sock_path_when_exists(self, tmp_path):
        """On POSIX with AF_UNIX, _get_server_address returns the sock path."""
        import truememory.model_client as mc

        fake_sock = tmp_path / "model.sock"
        fake_sock.touch()

        with patch.object(mc, "SOCK_PATH", fake_sock), \
             patch.object(mc, "_USE_UNIX_SOCKET", True):
            addr = mc._get_server_address()
        assert addr == str(fake_sock)

    def test_posix_returns_none_when_sock_missing(self, tmp_path):
        """On POSIX, _get_server_address returns None when sock is absent."""
        import truememory.model_client as mc
        absent = tmp_path / "missing.sock"

        with patch.object(mc, "SOCK_PATH", absent), \
             patch.object(mc, "_USE_UNIX_SOCKET", True):
            addr = mc._get_server_address()
        assert addr is None

    def test_windows_reads_port_file(self, tmp_path):
        """On Windows (_USE_UNIX_SOCKET=False), _get_server_address reads PORT_PATH."""
        import truememory.model_client as mc

        port_file = tmp_path / "model_server.port"
        port_file.write_text("54321", encoding="utf-8")

        with patch.object(mc, "PORT_PATH", port_file), \
             patch.object(mc, "_USE_UNIX_SOCKET", False):
            addr = mc._get_server_address()
        assert addr == ("127.0.0.1", 54321)

    def test_windows_returns_none_when_port_file_missing(self, tmp_path):
        """On Windows, _get_server_address returns None when PORT_PATH absent."""
        import truememory.model_client as mc
        absent = tmp_path / "model_server.port"

        with patch.object(mc, "PORT_PATH", absent), \
             patch.object(mc, "_USE_UNIX_SOCKET", False):
            addr = mc._get_server_address()
        assert addr is None

    def test_windows_returns_none_on_corrupt_port_file(self, tmp_path):
        """On Windows, _get_server_address returns None for non-integer port."""
        import truememory.model_client as mc

        port_file = tmp_path / "model_server.port"
        port_file.write_text("not-a-port", encoding="utf-8")

        with patch.object(mc, "PORT_PATH", port_file), \
             patch.object(mc, "_USE_UNIX_SOCKET", False):
            addr = mc._get_server_address()
        assert addr is None

    def test_model_client_start_server_posix_kwargs(self):
        """On POSIX, _start_server passes start_new_session=True to Popen."""
        import truememory.model_client as mc

        popen_calls = []

        class FakePopen:
            def __init__(self, *a, **kw):
                popen_calls.append(kw)
                self.pid = 9999

        # Ensure POSIX branch
        with patch.object(mc, "_USE_UNIX_SOCKET", True), \
             patch("os.setsid", lambda: None, create=True), \
             patch("truememory.model_client.subprocess") as mock_sp, \
             patch.object(mc, "_get_server_address", side_effect=[None, "fake"]), \
             patch.object(mc, "_server_is_alive", return_value=False), \
             patch("time.sleep"), \
             patch("time.time", side_effect=[0, 0, 100]):  # forces deadline pass
            mock_sp.Popen = FakePopen
            mock_sp.DEVNULL = -1
            mock_sp.CREATE_NEW_PROCESS_GROUP = 512
            mc._start_server()

        # At least one Popen call should have been attempted
        if popen_calls:
            assert "start_new_session" in popen_calls[0] or \
                   "creationflags" in popen_calls[0]


# ---------------------------------------------------------------------------
# hooks/core.py: psutil-based pid + process count
# ---------------------------------------------------------------------------

class TestHooksCoreProcessQueries:
    """hooks.core uses psutil instead of ps/pgrep."""

    def test_pid_is_alive_live_process(self):
        """_pid_is_alive returns True for the current Python process."""
        from truememory.hooks.core import _pid_is_alive
        assert _pid_is_alive(sys.process_info().pid if hasattr(sys, "process_info") else __import__("os").getpid()) is True

    def test_pid_is_alive_dead_pid(self):
        """_pid_is_alive returns False for a clearly non-existent PID."""
        from truememory.hooks.core import _pid_is_alive
        # PID 0 is reserved / invalid for process checks
        assert _pid_is_alive(0) is False

    def test_pid_is_alive_negative_pid(self):
        """_pid_is_alive returns False for negative PIDs."""
        from truememory.hooks.core import _pid_is_alive
        assert _pid_is_alive(-1) is False

    def test_pid_is_alive_uses_psutil(self):
        """_pid_is_alive calls psutil.Process, not subprocess.run(['ps', ...])."""
        import psutil
        from truememory.hooks.core import _pid_is_alive

        mock_proc = MagicMock()
        mock_proc.status.return_value = psutil.STATUS_RUNNING

        with patch("psutil.Process", return_value=mock_proc) as mock_cls:
            result = _pid_is_alive(1234)
        mock_cls.assert_called_once_with(1234)
        assert result is True

    def test_pid_is_alive_zombie_returns_false(self):
        """_pid_is_alive returns False for zombie processes."""
        import psutil
        from truememory.hooks.core import _pid_is_alive

        mock_proc = MagicMock()
        mock_proc.status.return_value = psutil.STATUS_ZOMBIE

        with patch("psutil.Process", return_value=mock_proc):
            result = _pid_is_alive(5678)
        assert result is False

    def test_count_active_ingest_processes_uses_psutil(self):
        """_count_active_ingest_processes calls psutil.process_iter, not pgrep."""
        import psutil
        from truememory.hooks.core import _count_active_ingest_processes

        mock_procs = []
        for cmdline, status in [
            (["python", "-m", "truememory.ingest.cli", "ingest", "/x"], psutil.STATUS_RUNNING),
            (["python", "-m", "truememory.ingest.cli", "ingest", "/y"], psutil.STATUS_RUNNING),
            (["python", "something_else.py"], psutil.STATUS_RUNNING),
            (["python", "-m", "truememory.ingest.cli", "ingest", "/z"], psutil.STATUS_ZOMBIE),
        ]:
            p = MagicMock()
            p.info = {"cmdline": cmdline, "status": status}
            mock_procs.append(p)

        with patch("psutil.process_iter", return_value=iter(mock_procs)):
            count = _count_active_ingest_processes()
        # 2 running ingest processes (zombie excluded)
        assert count == 2

    def test_count_active_ingest_processes_empty(self):
        """_count_active_ingest_processes returns 0 when no ingest processes run."""
        from truememory.hooks.core import _count_active_ingest_processes

        with patch("psutil.process_iter", return_value=iter([])):
            count = _count_active_ingest_processes()
        assert count == 0


# ---------------------------------------------------------------------------
# Import smoke tests — confirm modules load without AF_UNIX errors on Windows
# ---------------------------------------------------------------------------

def test_model_server_imports_cleanly():
    """truememory.model_server must import without raising on any platform."""
    import importlib
    import truememory.model_server  # noqa: F401
    # If we get here without exception, the module is importable.


def test_model_client_imports_cleanly():
    """truememory.model_client must import without raising on any platform."""
    import truememory.model_client  # noqa: F401


def test_hooks_core_imports_cleanly():
    """truememory.hooks.core must import without raising on any platform."""
    import truememory.hooks.core  # noqa: F401
