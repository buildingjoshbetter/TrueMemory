"""Client for the shared model server.

Provides drop-in replacements for get_model() and get_reranker() that
route inference to the shared model_server process over a Unix domain
socket (POSIX) or HMAC-authenticated TCP loopback (Windows).

Auto-starts the server on first request if not running.

Falls back to local model loading if the server cannot be reached.
Set TRUEMEMORY_NO_MODEL_SERVER=1 to force local loading.
"""

import base64
import json
import logging
import os
import platform
import plistlib
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from truememory._platform import (
    _LOOPBACK_HOST,
    _USE_UNIX,
    pid_is_alive,
    spawn_kwargs,
)

log = logging.getLogger(__name__)

_TRUEMEMORY_DIR = Path.home() / ".truememory"
SOCK_PATH = _TRUEMEMORY_DIR / "model.sock"
PID_PATH = _TRUEMEMORY_DIR / "model_server.pid"
PORT_PATH = _TRUEMEMORY_DIR / "model_server.port"
TOKEN_PATH = _TRUEMEMORY_DIR / "model_server.token"

_HEADER_FMT = ">I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# Issue #646 (M-53): wire protocol version. Must match
# ``model_server.PROTOCOL_VERSION``. On mismatch / non-JSON the client raises
# a clear ConnectionError instead of an inscrutable UnicodeDecodeError.
PROTOCOL_VERSION = 1


class ProtocolMismatchError(ConnectionError):
    """Raised when the server speaks an incompatible/foreign protocol."""

_SERVER_START_TIMEOUT = 30.0
_REQUEST_TIMEOUT = 120.0

# Issue #577: process-wide deadline override for model-server requests.
# None keeps the legacy 120s default. Latency-sensitive processes (Claude
# Code hooks) set a short deadline via set_request_timeout() so a contended
# server fast-fails and the caller's FTS-only fallback can trigger.
_default_request_timeout: float | None = None


def set_request_timeout(timeout: float | None) -> None:
    """Set a process-wide deadline (seconds) for model-server requests.

    When set, every request issued by this process without an explicit
    per-call ``timeout=`` fast-fails with :class:`TimeoutError` once the
    deadline expires, instead of absorbing the full autostart + retry
    cycle. Pass ``None`` to restore the legacy 120s behavior.
    """
    global _default_request_timeout
    _default_request_timeout = timeout


def _json_object_hook(obj):
    """Decode base64-encoded numpy arrays from JSON."""
    if "__ndarray__" in obj:
        data = base64.b64decode(obj["__ndarray__"])
        return np.frombuffer(data, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj

_APP_BUNDLE_PATH = _TRUEMEMORY_DIR / "TrueMemory.app"
_APP_EXECUTABLE = _APP_BUNDLE_PATH / "Contents" / "MacOS" / "TrueMemory"
_LSREGISTER = (
    "/System/Library/Frameworks/CoreServices.framework"
    "/Frameworks/LaunchServices.framework/Support/lsregister"
)


def _ensure_app_bundle() -> str | None:
    """Create a macOS .app bundle so Activity Monitor shows our icon.

    Returns the path to the .app executable, or None on failure.
    """
    if platform.system() != "Darwin":
        return None

    real_python = os.path.realpath(sys.executable)

    if _APP_EXECUTABLE.exists():
        try:
            if os.path.samefile(_APP_EXECUTABLE, real_python):
                return str(_APP_EXECUTABLE)
        except OSError:
            pass

    try:
        if _APP_BUNDLE_PATH.exists():
            shutil.rmtree(_APP_BUNDLE_PATH)

        contents = _APP_BUNDLE_PATH / "Contents"
        macos_dir = contents / "MacOS"
        resources_dir = contents / "Resources"
        macos_dir.mkdir(parents=True)
        resources_dir.mkdir(parents=True)

        os.link(real_python, _APP_EXECUTABLE)

        # @executable_path/../lib/libpython*.dylib needs this symlink
        python_root = Path(real_python).parent.parent
        lib_dir = python_root / "lib"
        if lib_dir.exists():
            os.symlink(lib_dir, contents / "lib")

        try:
            from importlib.resources import files
            icon_data = files("truememory.assets").joinpath("AppIcon.icns").read_bytes()
            (resources_dir / "AppIcon.icns").write_bytes(icon_data)
        except Exception:
            pass

        plist = {
            "CFBundleExecutable": "TrueMemory",
            "CFBundleIconFile": "AppIcon",
            "CFBundleIdentifier": "network.sauron.truememory",
            "CFBundleName": "TrueMemory",
            "CFBundleDisplayName": "TrueMemory",
            "CFBundlePackageType": "APPL",
            "LSBackgroundOnly": True,
            "LSUIElement": True,
        }
        with open(contents / "Info.plist", "wb") as f:
            plistlib.dump(plist, f)

        if os.path.exists(_LSREGISTER):
            subprocess.run(
                [_LSREGISTER, "-f", str(_APP_BUNDLE_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )

        return str(_APP_EXECUTABLE)
    except OSError as e:
        if e.errno == 18:
            log.debug("Cannot hardlink across devices, skipping app bundle")
        else:
            log.debug("Failed to create app bundle: %s", e)
        return None
    except Exception as e:
        log.debug("Failed to create app bundle: %s", e)
        return None


def _server_is_alive() -> bool:
    if not PID_PATH.exists():
        return False
    try:
        pid = int(PID_PATH.read_text().strip())
        return pid_is_alive(pid)
    except (ValueError, OSError):
        return False


def _read_port() -> int | None:
    """Read the TCP port written by the model server (Windows transport)."""
    try:
        port = int(PORT_PATH.read_text().strip())
        if not (1 <= port <= 65535):
            return None
        return port
    except (FileNotFoundError, ValueError, OSError):
        return None


def _read_token() -> bytes | None:
    """Read the HMAC token written by the model server (Windows transport)."""
    try:
        token = bytes.fromhex(TOKEN_PATH.read_text().strip())
        if len(token) != 32:
            return None
        return token
    except (FileNotFoundError, ValueError, OSError):
        return None


def _server_ready() -> bool:
    """Return True when the transport endpoint exists.

    On POSIX this checks for the Unix socket file; on Windows it checks
    for the port file that the server writes after binding.
    """
    if _USE_UNIX:
        return SOCK_PATH.exists()
    return PORT_PATH.exists() and TOKEN_PATH.exists()


def _start_server(wait_timeout: float | None = None) -> bool:
    """Start the model server as a detached subprocess.

    *wait_timeout* caps how long to wait for the spawned server to become
    ready (defaults to ``_SERVER_START_TIMEOUT``). Deadline-bound callers
    (issue #577) pass their remaining budget: the spawn still happens, so
    the server warms up for subsequent requests even when this call
    returns False.
    """
    if os.environ.get("TRUEMEMORY_NO_MODEL_SERVER", "") == "1":
        return False
    _TRUEMEMORY_DIR.mkdir(parents=True, exist_ok=True)

    alive = _server_is_alive()
    if not alive:
        # Clean up all stale artefacts in one pass.
        for p in (SOCK_PATH, PID_PATH, PORT_PATH, TOKEN_PATH):
            p.unlink(missing_ok=True)
    else:
        return True

    log.info("Starting model server...")

    popen_extra = spawn_kwargs()

    app_exe = _ensure_app_bundle()
    if app_exe:
        cmd = [app_exe, "-m", "truememory.model_server"]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
    else:
        cmd = [sys.executable, "-m", "truememory.model_server"]
        env = None

    try:
        _stderr_path = _TRUEMEMORY_DIR / "model_server.stderr"
        _stderr_fh = open(_stderr_path, "a")
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=_stderr_fh,
                env=env,
                **popen_extra,
            )
        finally:
            _stderr_fh.close()
    except Exception as e:
        log.warning("Failed to start model server: %s", e)
        if app_exe:
            try:
                _stderr_fh2 = open(_stderr_path, "a")
                try:
                    subprocess.Popen(
                        [sys.executable, "-m", "truememory.model_server"],
                        stdout=subprocess.DEVNULL,
                        stderr=_stderr_fh2,
                        **popen_extra,
                    )
                finally:
                    _stderr_fh2.close()
            except Exception as e2:
                log.warning("Fallback launch also failed: %s", e2)
                return False
        else:
            return False

    wait = _SERVER_START_TIMEOUT
    if wait_timeout is not None:
        wait = min(wait, max(wait_timeout, 0.0))
    deadline = time.time() + wait
    while time.time() < deadline:
        if _server_ready():
            time.sleep(0.2)
            return True
        time.sleep(0.1)

    log.warning("Model server did not start within %.0fs", wait)
    return False


def _remaining(deadline: float | None) -> float | None:
    """Seconds left until *deadline* (a time.monotonic() value), or None.

    Raises :class:`TimeoutError` if the deadline has already passed (M-76).
    """
    if deadline is None:
        return None
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("model server request deadline exceeded")
    return left


def _connect(
    deadline: float | None = None, timeout: float | None = None
) -> socket.socket:
    """Open a connection to the model server (Unix or TCP).

    *deadline* is an absolute ``time.monotonic()`` value bounding the TOTAL
    request, not just this op (issue #646, M-76). *timeout* is a legacy
    relative-seconds alias (converted to a deadline) kept for callers/tests
    that predate the total-deadline change.
    """
    if deadline is None and timeout is not None:
        deadline = time.monotonic() + timeout
    if _USE_UNIX:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_remaining(deadline) if deadline is not None else _REQUEST_TIMEOUT)
        try:
            sock.connect(str(SOCK_PATH))
        except OSError:
            sock.close()
            raise
        return sock

    # --- TCP loopback (Windows) ---
    port = _read_port()
    if port is None:
        raise ConnectionError("Model server port file not found")
    token = _read_token()
    if token is None:
        raise ConnectionError("Model server token file not found")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(_remaining(deadline) if deadline is not None else _REQUEST_TIMEOUT)
    try:
        sock.connect((_LOOPBACK_HOST, port))
        sock.sendall(token)
    except OSError:
        sock.close()
        raise
    return sock


def _send_request(request: dict, timeout: float | None = None) -> dict:
    """Send a request to the model server and return the response.

    *timeout* bounds the TOTAL request (connect + send + recv), not each
    socket op (issue #646, M-76): the old per-op settimeout could blow ~4x
    the intended budget in the worst case. A single ``time.monotonic()``
    deadline is computed once and the remaining budget is re-applied before
    each blocking op.

    The deadline is also shipped to the server in the payload (M-44) so an
    already-expired request fails cheaply server-side, before the global
    lock and a full encode.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    sock = _connect(deadline)
    try:
        payload = request
        if timeout is not None and "deadline" not in payload:
            # Server checks this wall-clock epoch deadline before encode (M-44).
            payload = {**request, "deadline": time.time() + max(_remaining(deadline), 0.0)}
        data = json.dumps(payload).encode("utf-8")
        header = struct.pack(_HEADER_FMT, len(data))
        sock.settimeout(_remaining(deadline) if deadline is not None else _REQUEST_TIMEOUT)
        sock.sendall(header + data)

        sock.settimeout(_remaining(deadline) if deadline is not None else _REQUEST_TIMEOUT)
        resp_header = _recv_exact(sock, _HEADER_SIZE)
        if not resp_header:
            raise ConnectionError("Server closed connection")
        resp_len = struct.unpack(_HEADER_FMT, resp_header)[0]
        if resp_len > _MAX_MESSAGE_SIZE:
            raise ConnectionError(f"Response too large: {resp_len} bytes")
        sock.settimeout(_remaining(deadline) if deadline is not None else _REQUEST_TIMEOUT)
        resp_data = _recv_exact(sock, resp_len)
        if not resp_data:
            raise ConnectionError("Incomplete response")
        return _decode_response(resp_data)
    finally:
        sock.close()


def _decode_response(resp_data: bytes) -> dict:
    """Decode a server response, detecting a foreign/stale protocol (M-53).

    A stale pickle-era daemon (or any non-JSON producer) returns bytes that
    aren't valid UTF-8 JSON; previously that surfaced as an inscrutable
    UnicodeDecodeError. Detect it and raise a clear, actionable error. A
    well-formed JSON response carrying an incompatible ``protocol`` version
    is rejected the same way.
    """
    try:
        resp = json.loads(resp_data, object_hook=_json_object_hook)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
        raise ProtocolMismatchError(
            "protocol mismatch — old model server running; restart it"
        ) from e
    if not isinstance(resp, dict):
        raise ProtocolMismatchError(
            "protocol mismatch — old model server running; restart it"
        )
    proto = resp.get("protocol")
    if proto is not None and proto != PROTOCOL_VERSION:
        raise ProtocolMismatchError(
            "protocol mismatch — old model server running; restart it"
        )
    return resp


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _deadline_error(timeout: float) -> TimeoutError:
    return TimeoutError(
        f"Model server request exceeded {timeout:.1f}s deadline"
    )


def _request_with_autostart(request: dict, timeout: float | None = None) -> dict:
    """Send request, auto-starting server if needed.

    *timeout* is a per-request deadline in seconds; ``None`` uses the
    process default (see :func:`set_request_timeout`, issue #577) which in
    turn defaults to the legacy ``_REQUEST_TIMEOUT``. When a deadline is in
    effect, expiry raises :class:`TimeoutError` immediately (fast-fail)
    instead of silently absorbing the autostart + full-timeout retry cycle —
    deadline-bound callers (hook recall) have their own FTS-only fallback.
    """
    if timeout is None:
        timeout = _default_request_timeout
    has_deadline = timeout is not None
    started = time.monotonic()

    try:
        return _send_request(request, timeout=timeout)
    except ProtocolMismatchError:
        # A stale/foreign server is bound (M-53). Restarting won't help until
        # it's killed — surface the clear, actionable error rather than spin
        # through an autostart retry (which can't bind anyway).
        raise
    except TimeoutError:
        # socket.timeout is TimeoutError on Python >= 3.10.
        if has_deadline:
            raise _deadline_error(timeout) from None
        # Legacy path (no deadline): fall through to autostart + retry.
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        pass

    remaining: float | None = None
    if has_deadline:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise _deadline_error(timeout)

    if not _start_server(wait_timeout=remaining):
        raise ConnectionError("Cannot start model server")

    if has_deadline:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise _deadline_error(timeout)
        try:
            return _send_request(request, timeout=remaining)
        except TimeoutError:
            raise _deadline_error(timeout) from None

    return _send_request(request)


class EmbeddingProxy:
    """Drop-in replacement for the embedding model with .encode() method."""

    def __init__(self, tier: str = ""):
        self._tier = tier

    def encode(self, texts, timeout: float | None = None, **kwargs) -> np.ndarray:
        """Embed *texts* via the model server.

        *timeout* is an optional per-call deadline in seconds (issue #577);
        on expiry a :class:`TimeoutError` is raised (fast-fail, no autostart
        retry). ``None`` uses the process default / legacy 120s.
        """
        if isinstance(texts, str):
            texts = [texts]
        resp = _request_with_autostart({
            "op": "embed",
            "texts": list(texts),
            "tier": self._tier,
        }, timeout=timeout)
        if not resp.get("ok"):
            raise RuntimeError(f"Model server error: {resp.get('error', 'unknown')}")
        return resp["vectors"]


class RerankerProxy:
    """Drop-in replacement for CrossEncoder with .predict() method."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name

    def predict(self, pairs, timeout: float | None = None, **kwargs) -> np.ndarray:
        """Rerank *pairs* via the model server.

        *timeout* is an optional per-call deadline in seconds (issue #577);
        see :meth:`EmbeddingProxy.encode`.
        """
        resp = _request_with_autostart({
            "op": "rerank",
            "pairs": list(pairs),
            "model_name": self._model_name,
        }, timeout=timeout)
        if not resp.get("ok"):
            raise RuntimeError(f"Model server error: {resp.get('error', 'unknown')}")
        return resp["scores"]


def use_model_server() -> bool:
    """Check if the model server should be used.

    Returns True only if:
    1. TRUEMEMORY_NO_MODEL_SERVER is not set
    2. The server endpoint exists (server is running)

    Processes that want to ensure the server is running should call
    ensure_server_running() first (e.g., during MCP server startup).
    """
    if os.environ.get("TRUEMEMORY_NO_MODEL_SERVER", "") == "1":
        return False
    return _server_ready() and _server_is_alive()


def ensure_server_running() -> bool:
    """Start the model server if it's not already running.

    Call from MCP server startup or CLI to enable the shared model server.
    Returns True if server is running after this call.
    """
    if os.environ.get("TRUEMEMORY_NO_MODEL_SERVER", "") == "1":
        return False
    if _server_is_alive() and _server_ready():
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
