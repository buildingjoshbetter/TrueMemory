"""Regression tests for issue #458: unbounded message size DoS on model server socket.

The model server reads a 4-byte big-endian length header and previously
allocated that many bytes without validation. A malformed or malicious
message could request up to 4GB allocation, causing OOM/DoS.
"""
from __future__ import annotations

import json
import socket
import struct
import tempfile
import threading
import time
from pathlib import Path

import pytest

_SKIP_NO_UNIX = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="Unix domain sockets not available on this platform",
)


class TestIssue458SocketBounds:
    """Verify that oversized messages are rejected before allocation."""

    def test_issue_458_max_message_size_constant_exists(self):
        """Both server and client must define _MAX_MESSAGE_SIZE."""
        from truememory import model_server, model_client

        assert hasattr(model_server, "_MAX_MESSAGE_SIZE"), (
            "model_server missing _MAX_MESSAGE_SIZE — DoS via unbounded allocation"
        )
        assert hasattr(model_client, "_MAX_MESSAGE_SIZE"), (
            "model_client missing _MAX_MESSAGE_SIZE — DoS via unbounded response"
        )
        assert model_server._MAX_MESSAGE_SIZE <= 100 * 1024 * 1024, (
            f"Server MAX_MESSAGE_SIZE too large ({model_server._MAX_MESSAGE_SIZE})"
        )

    def test_issue_458_server_bounds_check_in_handle_client(self):
        """model_server.py handle_client must check length before allocating."""
        import inspect
        from truememory.model_server import ModelServer

        source = inspect.getsource(ModelServer.handle_client)
        assert "_MAX_MESSAGE_SIZE" in source or "MAX_MESSAGE" in source, (
            "handle_client has no message size check — unbounded allocation DoS"
        )

    def test_issue_458_client_bounds_check_in_send_request(self):
        """model_client.py _send_request must check response length."""
        import inspect
        from truememory.model_client import _send_request

        source = inspect.getsource(_send_request)
        assert "_MAX_MESSAGE_SIZE" in source or "MAX_MESSAGE" in source, (
            "_send_request has no response size check — unbounded allocation DoS"
        )

    @_SKIP_NO_UNIX
    def test_issue_458_oversized_message_rejected(self):
        """Server must close connection when length header exceeds limit."""
        from truememory.model_server import ModelServer, _MAX_MESSAGE_SIZE

        server = ModelServer()

        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "test.sock"

            srv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv_sock.bind(str(sock_path))
            srv_sock.listen(1)
            srv_sock.settimeout(5.0)

            response_received = [False]
            connection_closed = [False]

            def run_server():
                try:
                    conn, _ = srv_sock.accept()
                    server.handle_client(conn)
                except socket.timeout:
                    pass
                finally:
                    srv_sock.close()

            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()

            time.sleep(0.1)

            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(5.0)
            try:
                client.connect(str(sock_path))
                oversized_length = _MAX_MESSAGE_SIZE + 1
                header = struct.pack(">I", oversized_length)
                client.sendall(header)

                try:
                    data = client.recv(4096)
                    if not data:
                        connection_closed[0] = True
                    else:
                        response_received[0] = True
                except (ConnectionResetError, BrokenPipeError, socket.timeout):
                    connection_closed[0] = True
            finally:
                client.close()

            server_thread.join(timeout=5.0)

            assert connection_closed[0] or not response_received[0], (
                "Server processed an oversized message instead of rejecting it — "
                "DoS via unbounded allocation is possible"
            )

    @_SKIP_NO_UNIX
    def test_issue_458_valid_sized_message_accepted(self):
        """Messages under the limit must still be processed normally."""
        from truememory.model_server import ModelServer

        server = ModelServer()

        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "test.sock"

            srv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv_sock.bind(str(sock_path))
            srv_sock.listen(1)
            srv_sock.settimeout(5.0)

            response_holder = [None]

            def run_server():
                try:
                    conn, _ = srv_sock.accept()
                    server.handle_client(conn)
                except socket.timeout:
                    pass
                finally:
                    srv_sock.close()

            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()

            time.sleep(0.1)

            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(5.0)
            try:
                client.connect(str(sock_path))
                request = json.dumps({"op": "ping"}).encode("utf-8")
                header = struct.pack(">I", len(request))
                client.sendall(header + request)

                resp_header = b""
                while len(resp_header) < 4:
                    chunk = client.recv(4 - len(resp_header))
                    if not chunk:
                        break
                    resp_header += chunk

                if len(resp_header) == 4:
                    resp_len = struct.unpack(">I", resp_header)[0]
                    resp_data = b""
                    while len(resp_data) < resp_len:
                        chunk = client.recv(resp_len - len(resp_data))
                        if not chunk:
                            break
                        resp_data += chunk
                    if resp_data:
                        response_holder[0] = json.loads(resp_data)
            finally:
                client.close()

            server_thread.join(timeout=5.0)

            assert response_holder[0] is not None, (
                "Valid-sized ping request was not processed — bounds check is too aggressive"
            )
            assert response_holder[0].get("ok") is True
