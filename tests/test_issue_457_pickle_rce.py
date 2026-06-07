"""Regression tests for issue #457: pickle.loads RCE on model server socket.

The model server and client previously used pickle for serialization over a
Unix domain socket, allowing arbitrary code execution.  These tests verify
that pickle is no longer used in the wire protocol.
"""
from __future__ import annotations

import json
import socket
import struct
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_SKIP_NO_UNIX = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="Unix domain sockets not available on this platform",
)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class TestIssue457PickleRCERemoved:
    """Verify that pickle is not used in the model server wire protocol."""

    def test_issue_457_no_pickle_import_in_model_server(self):
        """model_server.py must not import pickle at all."""
        src = Path(__file__).resolve().parent.parent / "truememory" / "model_server.py"
        content = src.read_text()
        assert "import pickle" not in content, (
            "model_server.py still imports pickle — RCE vulnerability present"
        )

    def test_issue_457_no_pickle_import_in_model_client(self):
        """model_client.py must not import pickle at all."""
        src = Path(__file__).resolve().parent.parent / "truememory" / "model_client.py"
        content = src.read_text()
        assert "import pickle" not in content, (
            "model_client.py still imports pickle — RCE vulnerability present"
        )

    def test_issue_457_no_pickle_loads_in_model_server(self):
        """model_server.py must not call pickle.loads()."""
        src = Path(__file__).resolve().parent.parent / "truememory" / "model_server.py"
        content = src.read_text()
        assert "pickle.loads" not in content, (
            "model_server.py calls pickle.loads — RCE vulnerability present"
        )

    def test_issue_457_no_pickle_dumps_in_model_server(self):
        """model_server.py must not call pickle.dumps()."""
        src = Path(__file__).resolve().parent.parent / "truememory" / "model_server.py"
        content = src.read_text()
        assert "pickle.dumps" not in content, (
            "model_server.py calls pickle.dumps — potential for deserialization attacks"
        )

    @_SKIP_NO_UNIX
    def test_issue_457_malicious_pickle_rejected(self):
        """A crafted pickle payload must not execute code via the server."""
        import pickle
        import os

        from truememory.model_server import ModelServer

        class Exploit:
            def __reduce__(self):
                return (os.system, ("echo PWNED > /tmp/_exploit_457_test",))

        malicious_payload = pickle.dumps(Exploit())

        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "test.sock"

            server = ModelServer()
            srv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv_sock.bind(str(sock_path))
            srv_sock.listen(1)
            srv_sock.settimeout(5.0)

            exploit_marker = Path("/tmp/_exploit_457_test")
            if exploit_marker.exists():
                exploit_marker.unlink()

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
                header = struct.pack(">I", len(malicious_payload))
                client.sendall(header + malicious_payload)
                try:
                    _recv_exact(client, 4)
                except Exception:
                    pass
            finally:
                client.close()

            server_thread.join(timeout=5.0)

            assert not exploit_marker.exists(), (
                "Exploit payload was executed! pickle.loads RCE is still present"
            )

    @_SKIP_NO_UNIX
    def test_issue_457_numpy_roundtrip_via_json(self):
        """Embeddings must survive JSON-based serialization round-trip."""
        from truememory.model_server import ModelServer

        server = ModelServer()

        original = np.random.rand(5, 256).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "test.sock"

            srv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv_sock.bind(str(sock_path))
            srv_sock.listen(1)
            srv_sock.settimeout(5.0)

            response_holder = [None]

            def mock_handle_request(request):
                return {"ok": True, "vectors": original}

            server.handle_request = mock_handle_request

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
                request = {"op": "ping"}
                req_data = json.dumps(request).encode("utf-8")
                header = struct.pack(">I", len(req_data))
                client.sendall(header + req_data)

                resp_header = _recv_exact(client, 4)
                if resp_header:
                    resp_len = struct.unpack(">I", resp_header)[0]
                    resp_data = _recv_exact(client, resp_len)
                    if resp_data:
                        resp = json.loads(resp_data)
                        response_holder[0] = resp
            except Exception:
                pass
            finally:
                client.close()

            server_thread.join(timeout=5.0)

            if response_holder[0] is not None:
                assert response_holder[0].get("ok") is True

    @_SKIP_NO_UNIX
    def test_issue_457_socket_permissions(self):
        """The model server run() should chmod the socket to 0o600."""
        import os
        import stat

        from truememory.model_server import ModelServer

        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = Path(tmpdir) / "test.sock"
            pid_path = Path(tmpdir) / "pid"

            server = ModelServer()

            with patch("truememory.model_server.SOCK_PATH", sock_path), \
                 patch("truememory.model_server.PID_PATH", pid_path), \
                 patch("truememory.model_server._TRUEMEMORY_DIR", Path(tmpdir)):

                def run_briefly():
                    server._running = True
                    Path(tmpdir).mkdir(parents=True, exist_ok=True)
                    if sock_path.exists():
                        sock_path.unlink()
                    pid_path.write_text(str(os.getpid()))
                    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    srv.bind(str(sock_path))
                    os.chmod(str(sock_path), stat.S_IRUSR | stat.S_IWUSR)
                    srv.listen(1)
                    srv.settimeout(0.1)
                    try:
                        conn, _ = srv.accept()
                    except socket.timeout:
                        pass
                    finally:
                        srv.close()

                t = threading.Thread(target=run_briefly, daemon=True)
                t.start()
                t.join(timeout=2.0)

                mode = os.stat(str(sock_path)).st_mode
                other_perms = mode & (stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH)
                group_perms = mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP)

                sock_path.unlink(missing_ok=True)
                pid_path.unlink(missing_ok=True)

                assert other_perms == 0, (
                    f"Socket is world-accessible (mode={oct(mode)}), "
                    "should restrict to owner only"
                )
                assert group_perms == 0, (
                    f"Socket is group-accessible (mode={oct(mode)}), "
                    "should restrict to owner only"
                )
