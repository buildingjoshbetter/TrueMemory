"""Client for the shared model server.

Provides drop-in replacements for get_model() and get_reranker() that
route inference to the shared model_server process over a domain/TCP socket.
Auto-starts the server on first request if not running.

Falls back to local model loading if the server cannot be reached.
Set TRUEMEMORY_NO_MODEL_SERVER=1 to force local loading.

Socket discovery
----------------
POSIX (Linux/macOS):  connects to ~/.truememory/model.sock (AF_UNIX).
Windows:              reads the TCP port from ~/.truememory/model_server.port
                      then connects to 127.0.0.1:<port> (AF_INET).
"""

import logging
import os
import pickle
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_TRUEMEMORY_DIR = Path.home() / ".truememory"
SOCK_PATH = _TRUEMEMORY_DIR / "model.sock"
PORT_PATH = _TRUEMEMORY_DIR / "model_server.port"   # Windows sidecar
PID_PATH = _TRUEMEMORY_DIR / "model_server.pid"

# Use AF_UNIX where available (POSIX), fall back to AF_INET TCP on Windows.
_USE_UNIX_SOCKET = hasattr(socket, "AF_UNIX")

_HEADER_FMT = ">I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

_SERVER_START_TIMEOUT = 30.0
_REQUEST_TIMEOUT = 120.0


def _get_server_address() -> str | tuple | None:
    """Return the socket address the server is listening on, or None.

    POSIX: returns the SOCK_PATH string (AF_UNIX).
    Windows: reads PORT_PATH and returns ("127.0.0.1", port) (AF_INET).
    Returns None if the sidecar file is absent / unreadable.
    """
    if _USE_UNIX_SOCKET:
        return str(SOCK_PATH) if SOCK_PATH.exists() else None
    try:
        port = int(PORT_PATH.read_text(encoding="utf-8").strip())
        return ("127.0.0.1", port)
    except (OSError, ValueError):
        return None


def _server_is_alive() -> bool:
    if not PID_PATH.exists():
        return False
    try:
        pid = int(PID_PATH.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError, OSError):
        return False


def _start_server() -> bool:
    """Start the model server as a detached subprocess."""
    _TRUEMEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if _USE_UNIX_SOCKET:
        if SOCK_PATH.exists() and not _server_is_alive():
            SOCK_PATH.unlink(missing_ok=True)
    else:
        if PORT_PATH.exists() and not _server_is_alive():
            PORT_PATH.unlink(missing_ok=True)
    if PID_PATH.exists() and not _server_is_alive():
        PID_PATH.unlink(missing_ok=True)

    if _server_is_alive():
        return True

    log.info("Starting model server...")
    try:
        # start_new_session is POSIX-only (setsid). On Windows use
        # CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS instead.
        _popen_kwargs: dict = {}
        if hasattr(os, "setsid"):
            _popen_kwargs["start_new_session"] = True
        else:
            _popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )
        subprocess.Popen(
            [sys.executable, "-m", "truememory.model_server"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_popen_kwargs,
        )
    except Exception as e:
        log.warning("Failed to start model server: %s", e)
        return False

    # Wait for the server to write its sidecar file (sock or port).
    deadline = time.time() + _SERVER_START_TIMEOUT
    while time.time() < deadline:
        if _get_server_address() is not None:
            time.sleep(0.2)
            return True
        time.sleep(0.1)

    log.warning("Model server did not start within %.0fs", _SERVER_START_TIMEOUT)
    return False


def _send_request(request: dict) -> dict:
    """Send a request to the model server and return the response."""
    addr = _get_server_address()
    if addr is None:
        raise FileNotFoundError("Model server socket/port file not found")

    if _USE_UNIX_SOCKET:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(_REQUEST_TIMEOUT)
    try:
        sock.connect(addr)
        data = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        header = struct.pack(_HEADER_FMT, len(data))
        sock.sendall(header + data)

        resp_header = _recv_exact(sock, _HEADER_SIZE)
        if not resp_header:
            raise ConnectionError("Server closed connection")
        resp_len = struct.unpack(_HEADER_FMT, resp_header)[0]
        resp_data = _recv_exact(sock, resp_len)
        if not resp_data:
            raise ConnectionError("Incomplete response")
        return pickle.loads(resp_data)
    finally:
        sock.close()


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _request_with_autostart(request: dict) -> dict:
    """Send request, auto-starting server if needed."""
    try:
        return _send_request(request)
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        pass

    if not _start_server():
        raise ConnectionError("Cannot start model server")

    return _send_request(request)


class EmbeddingProxy:
    """Drop-in replacement for the embedding model with .encode() method."""

    def __init__(self, tier: str = ""):
        self._tier = tier

    def encode(self, texts, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        resp = _request_with_autostart({
            "op": "embed",
            "texts": list(texts),
            "tier": self._tier,
        })
        if not resp.get("ok"):
            raise RuntimeError(f"Model server error: {resp.get('error', 'unknown')}")
        return resp["vectors"]


class RerankerProxy:
    """Drop-in replacement for CrossEncoder with .predict() method."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name

    def predict(self, pairs, **kwargs) -> np.ndarray:
        resp = _request_with_autostart({
            "op": "rerank",
            "pairs": list(pairs),
            "model_name": self._model_name,
        })
        if not resp.get("ok"):
            raise RuntimeError(f"Model server error: {resp.get('error', 'unknown')}")
        return resp["scores"]


def use_model_server() -> bool:
    """Check if the model server should be used.

    Returns True only if:
    1. TRUEMEMORY_NO_MODEL_SERVER is not set
    2. The server sidecar file exists (SOCK_PATH on POSIX, PORT_PATH on Windows)

    Processes that want to ensure the server is running should call
    ensure_server_running() first (e.g., during MCP server startup).
    """
    if os.environ.get("TRUEMEMORY_NO_MODEL_SERVER", "") == "1":
        return False
    return _get_server_address() is not None and _server_is_alive()


def ensure_server_running() -> bool:
    """Start the model server if it's not already running.

    Call from MCP server startup or CLI to enable the shared model server.
    Returns True if server is running after this call.
    """
    if os.environ.get("TRUEMEMORY_NO_MODEL_SERVER", "") == "1":
        return False
    if _server_is_alive() and _get_server_address() is not None:
        return True
    return _start_server()


def get_embedding_proxy(tier: str = "") -> EmbeddingProxy:
    """Get an embedding proxy connected to the model server."""
    return EmbeddingProxy(tier=tier)


def get_reranker_proxy(model_name: str | None = None) -> RerankerProxy:
    """Get a reranker proxy connected to the model server."""
    return RerankerProxy(model_name=model_name)


def ping() -> bool:
    """Check if model server is reachable."""
    try:
        resp = _send_request({"op": "ping"})
        return resp.get("ok", False)
    except Exception:
        return False
