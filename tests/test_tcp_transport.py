"""Tests for Windows TCP loopback transport (PR #411).

Covers HMAC authentication, port/token file I/O, oversized request
rejection, platform detection, and cleanup behaviour.  All tests run
on POSIX by temporarily patching _USE_UNIX to False.
"""
from __future__ import annotations

import hmac
import os
import pickle
import secrets
import socket
import struct
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from truememory._platform import pid_is_alive, spawn_kwargs, _USE_UNIX


# ---------------------------------------------------------------------------
# _platform module tests
# ---------------------------------------------------------------------------

class TestPlatform:
    def test_pid_is_alive_for_self(self):
        assert pid_is_alive(os.getpid()) is True

    def test_pid_is_alive_for_dead_pid(self):
        assert pid_is_alive(99999999) is False

    def test_spawn_kwargs_returns_dict(self):
        kw = spawn_kwargs()
        assert isinstance(kw, dict)
        if os.name != "nt":
            assert "start_new_session" in kw

    def test_use_unix_is_bool(self):
        assert isinstance(_USE_UNIX, bool)


# ---------------------------------------------------------------------------
# Port / token file helpers (model_client)
# ---------------------------------------------------------------------------

class TestPortTokenFiles:
    def test_read_port_valid(self, tmp_path):
        port_file = tmp_path / "model_server.port"
        port_file.write_text("12345")
        with patch("truememory.model_client.PORT_PATH", port_file):
            from truememory.model_client import _read_port
            assert _read_port() == 12345

    def test_read_port_missing(self, tmp_path):
        with patch("truememory.model_client.PORT_PATH", tmp_path / "no_such_file"):
            from truememory.model_client import _read_port
            assert _read_port() is None

    def test_read_port_invalid(self, tmp_path):
        port_file = tmp_path / "model_server.port"
        port_file.write_text("0")
        with patch("truememory.model_client.PORT_PATH", port_file):
            from truememory.model_client import _read_port
            assert _read_port() is None

    def test_read_token_valid(self, tmp_path):
        token_file = tmp_path / "model_server.token"
        token = secrets.token_bytes(32)
        token_file.write_text(token.hex())
        with patch("truememory.model_client.TOKEN_PATH", token_file):
            from truememory.model_client import _read_token
            assert _read_token() == token

    def test_read_token_wrong_length(self, tmp_path):
        token_file = tmp_path / "model_server.token"
        token_file.write_text("abcd")
        with patch("truememory.model_client.TOKEN_PATH", token_file):
            from truememory.model_client import _read_token
            assert _read_token() is None

    def test_read_token_missing(self, tmp_path):
        with patch("truememory.model_client.TOKEN_PATH", tmp_path / "nope"):
            from truememory.model_client import _read_token
            assert _read_token() is None


# ---------------------------------------------------------------------------
# HMAC TCP authentication (mini integration test)
# ---------------------------------------------------------------------------

class TestTCPAuth:
    """Spin up a real ModelServer.handle_client on a TCP socket and test auth."""

    @pytest.fixture()
    def server_and_token(self):
        from truememory.model_server import ModelServer
        srv = ModelServer()
        token = secrets.token_bytes(32)
        srv._token = token

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        yield srv, listener, port, token
        listener.close()

    def _make_request_bytes(self, req: dict) -> bytes:
        data = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
        header = struct.pack(">I", len(data))
        return header + data

    def test_valid_token_accepted(self, server_and_token):
        srv, listener, port, token = server_and_token
        req_bytes = self._make_request_bytes({"action": "ping"})

        def client():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", port))
            sock.sendall(token + req_bytes)
            resp_header = sock.recv(4)
            sock.close()
            return resp_header

        t = threading.Thread(target=client)
        t.start()

        conn, _ = listener.accept()
        with patch("truememory.model_server._USE_UNIX", False):
            srv.handle_client(conn)
        t.join(timeout=5)

    def test_invalid_token_rejected(self, server_and_token):
        srv, listener, port, token = server_and_token
        bad_token = secrets.token_bytes(32)

        closed = threading.Event()

        def client():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", port))
            sock.sendall(bad_token)
            try:
                data = sock.recv(4)
                if not data:
                    closed.set()
            except Exception:
                closed.set()
            sock.close()

        t = threading.Thread(target=client)
        t.start()

        conn, _ = listener.accept()
        with patch("truememory.model_server._USE_UNIX", False):
            srv.handle_client(conn)
        t.join(timeout=5)
        assert closed.is_set(), "Server should close connection on bad token"


# ---------------------------------------------------------------------------
# Oversized request rejection
# ---------------------------------------------------------------------------

class TestOversizedRequest:
    def test_rejects_oversized_payload(self):
        from truememory.model_server import ModelServer, _MAX_MESSAGE_SIZE
        srv = ModelServer()

        client_sock, server_sock = socket.socketpair()
        try:
            fake_length = _MAX_MESSAGE_SIZE + 1
            header = struct.pack(">I", fake_length)
            client_sock.sendall(header)

            srv.handle_client(server_sock)

            try:
                data = client_sock.recv(1)
                assert data == b"", "Server should close after oversized request"
            except (ConnectionResetError, BrokenPipeError):
                pass
        finally:
            client_sock.close()
            server_sock.close()


# ---------------------------------------------------------------------------
# Cleanup idempotency
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_double_cleanup_is_safe(self, tmp_path):
        from truememory.model_server import ModelServer
        srv = ModelServer()

        with patch("truememory.model_server.SOCK_PATH", tmp_path / "model.sock"), \
             patch("truememory.model_server.PID_PATH", tmp_path / "model_server.pid"), \
             patch("truememory.model_server.PORT_PATH", tmp_path / "model_server.port"), \
             patch("truememory.model_server.TOKEN_PATH", tmp_path / "model_server.token"):

            (tmp_path / "model_server.pid").write_text("99999")
            srv._cleanup()
            srv._cleaned_up = False
            srv._cleanup()

    def test_atomic_write_text(self, tmp_path):
        from truememory.model_server import ModelServer
        target = tmp_path / "test.txt"
        ModelServer._atomic_write_text(target, "hello", mode=0o600)
        assert target.read_text() == "hello"
        assert oct(target.stat().st_mode & 0o777) == "0o600"
