"""Client for the shared model server.

Provides drop-in replacements for get_model() and get_reranker() that
route inference to the shared model_server process over a Unix domain
socket (POSIX) or HMAC-authenticated TCP loopback (Windows).

Auto-starts the server on first request if not running.

An unavailable server raises an availability error without loading local copies.
Set TRUEMEMORY_NO_MODEL_SERVER=1 at process startup for explicit local loading.
"""

import base64
import json
import logging
import os
import platform
import plistlib
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from truememory.embedding_target import EmbeddingTarget

from truememory._platform import (
    _LOOPBACK_HOST,
    _MODEL_SERVER_GENERATION_ENV,
    _USE_UNIX,
    pid_is_alive,
    process_birth,
    read_start_claim,
    spawn_kwargs,
    try_file_lock,
)

log = logging.getLogger(__name__)

_TRUEMEMORY_DIR = Path.home() / ".truememory"
SOCK_PATH = _TRUEMEMORY_DIR / "model.sock"
PID_PATH = _TRUEMEMORY_DIR / "model_server.pid"
PORT_PATH = _TRUEMEMORY_DIR / "model_server.port"
TOKEN_PATH = _TRUEMEMORY_DIR / "model_server.token"
LOCK_PATH = _TRUEMEMORY_DIR / "model_server.lock"
START_LOCK_PATH = _TRUEMEMORY_DIR / "model_server.start.lock"
START_STATE_PATH = _TRUEMEMORY_DIR / "model_server.start.json"

_HEADER_FMT = ">I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# Issue #646 (M-53): wire protocol version. Must match
# ``model_server.PROTOCOL_VERSION``. On mismatch / non-JSON the client raises
# a clear ConnectionError instead of an inscrutable UnicodeDecodeError.
PROTOCOL_VERSION = 1


class ProtocolMismatchError(ConnectionError):
    """Raised when the server speaks an incompatible/foreign protocol."""


class ServingIdentityMismatchError(ProtocolMismatchError):
    """A selected request was rejected for effective identity or shape drift."""


class ModelServerBusyError(RuntimeError):
    """Capacity rejection: do not restart the daemon or load another model."""

    retry_after_ms = 250

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


def _ensure_app_bundle(deadline: float | None = None) -> str | None:
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

    # Cosmetic setup must not consume a short recall/startup deadline.
    if deadline is not None and _remaining(deadline) < 10:
        return None

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
                timeout=min(10, _remaining(deadline)) if deadline is not None else 10,
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


def _probe_server(deadline: float) -> bool:
    """Validate transport/auth/protocol without renewing the caller's budget."""
    if not _server_ready():
        return False
    try:
        response = _send_request({"op": "ping"}, timeout=min(0.2, _remaining(deadline)))
    except ModelServerBusyError:
        return True  # A capacity response still proves an existing daemon.
    except ProtocolMismatchError:
        raise
    except OSError:
        return False
    if not response.get("ok"):
        raise ProtocolMismatchError("Model endpoint does not accept the readiness ping")
    return True


def _write_start_claim(claim: dict) -> None:
    temporary = START_STATE_PATH.with_name(START_STATE_PATH.name + "." + secrets.token_hex(8) + ".tmp")
    fd = os.open(str(temporary), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(claim, stream)
        os.replace(temporary, START_STATE_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def _launch_is_pending(claim: dict | None) -> bool:
    if claim is None:
        return False
    identity = claim["child"] or claim["launcher"]
    return process_birth(identity["pid"]) == identity["born"]


def _spawn_server(generation: str, deadline: float) -> subprocess.Popen:
    _remaining(deadline)
    app_exe = _ensure_app_bundle(deadline)
    env = os.environ.copy()
    env[_MODEL_SERVER_GENERATION_ENV] = generation
    if app_exe:
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
    executables = [app_exe, sys.executable] if app_exe else [sys.executable]
    for index, executable in enumerate(executables):
        _remaining(deadline)
        try:
            with (_TRUEMEMORY_DIR / "model_server.stderr").open("a") as stderr:
                return subprocess.Popen(
                    [executable, "-m", "truememory.model_server"],
                    stdout=subprocess.DEVNULL, stderr=stderr, env=env, **spawn_kwargs(),
                )
        except OSError:
            if index == len(executables) - 1:
                raise
    raise RuntimeError("No model-server executable available")


def _start_server(wait_timeout: float | None = None) -> bool:
    """Coordinate one launch generation and wait within one monotonic budget.

    A short-lived launch lock protects the claim and Popen; the child owns a
    separate lifetime bind lock. A timed-out caller leaves its live child
    claimed, so concurrent or later clients wait instead of spawning copies.
    """
    if not use_model_server():
        return False
    wait = _SERVER_START_TIMEOUT if wait_timeout is None else min(_SERVER_START_TIMEOUT, max(wait_timeout, 0.0))
    if wait <= 0:
        return False
    deadline = time.monotonic() + wait
    launched = False
    try:
        _TRUEMEMORY_DIR.mkdir(parents=True, exist_ok=True)
        try:
            _TRUEMEMORY_DIR.chmod(0o700)
        except OSError:
            pass
        while time.monotonic() < deadline:
            if _probe_server(deadline):
                return True
            _remaining(deadline)
            gate = try_file_lock(START_LOCK_PATH)
            if gate is not None:
                try:
                    if _probe_server(deadline):
                        return True
                    _remaining(deadline)
                    bind = try_file_lock(LOCK_PATH)
                    if bind is not None:
                        claim = None
                        try:
                            # Never trust a PID file or remove endpoint files.
                            # A held bind lock is the daemon's ownership proof.
                            previous = read_start_claim(START_STATE_PATH)
                            if not _launch_is_pending(previous):
                                if launched:
                                    return False  # At most one launch per caller.
                                born = process_birth(os.getpid())
                                if born is None:
                                    raise OSError("Cannot identify startup owner")
                                claim = {"generation": secrets.token_hex(16),
                                         "launcher": {"pid": os.getpid(), "born": born}, "child": None}
                                _write_start_claim(claim)
                        finally:
                            os.close(bind)
                        # Release bind ownership BEFORE launching the child.
                        if claim is not None:
                            try:
                                child = _spawn_server(claim["generation"], deadline)
                                launched = True
                                child_born = process_birth(child.pid)
                                if child_born is None:
                                    raise OSError("Model server exited before startup")
                                claim["child"] = {"pid": child.pid, "born": child_born}
                                _write_start_claim(claim)
                            except (OSError, TimeoutError):
                                # Invalidate this generation; a late child must
                                # not publish after its launch failed.
                                START_STATE_PATH.unlink(missing_ok=True)
                                raise
                finally:
                    os.close(gate)
            time.sleep(min(0.05, _remaining(deadline)))
    except ProtocolMismatchError:
        raise
    except TimeoutError:
        pass
    except (OSError, ValueError) as error:
        log.warning("Shared model-server startup unavailable (%s)", type(error).__name__)
    log.warning("Shared model server unavailable within %.2fs; retry or inspect model_server.stderr", wait)
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
        try:
            sock.sendall(header + data)
        except OSError:
            # A bounded server may reject the header before our full send ends.
            # Preserve that explicit error without mistaking overload for death.
            recovery_deadline = time.monotonic() + 0.25
            if deadline is not None:
                recovery_deadline = min(recovery_deadline, deadline)
            try:
                _receive_response(sock, recovery_deadline)
            except ModelServerBusyError:
                raise
            except (OSError, ValueError):
                pass
            raise
        return _receive_response(sock, deadline)
    finally:
        sock.close()


def _receive_response(sock: socket.socket, deadline: float | None) -> dict:
    response_deadline = deadline if deadline is not None else time.monotonic() + _REQUEST_TIMEOUT
    resp_header = _recv_exact(sock, _HEADER_SIZE, response_deadline)
    if not resp_header:
        raise ConnectionError("Server closed connection")
    resp_len = struct.unpack(_HEADER_FMT, resp_header)[0]
    if resp_len > _MAX_MESSAGE_SIZE:
        raise ConnectionError(f"Response too large: {resp_len} bytes")
    resp_data = _recv_exact(sock, resp_len, response_deadline)
    if not resp_data:
        raise ConnectionError("Incomplete response")
    return _decode_response(resp_data)


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
    if resp.get("error_code") == "server_busy":
        raise ModelServerBusyError("Model server busy: request capacity exhausted")
    return resp


def _recv_exact(sock: socket.socket, n: int, deadline: float | None = None) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        if deadline is not None:
            sock.settimeout(_remaining(deadline))
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
        if has_deadline and time.monotonic() - started >= timeout:
            raise _deadline_error(timeout)
        raise ConnectionError(
            "Shared model server unavailable; retry the request or inspect "
            "~/.truememory/model_server.stderr"
        )

    if has_deadline:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise _deadline_error(timeout)
        try:
            return _send_request(request, timeout=remaining)
        except TimeoutError:
            raise _deadline_error(timeout) from None

    return _send_request(request)


def _batch_request(request: dict, kwargs: dict) -> dict:
    """Use an additive operation so old daemons cannot ignore an explicit limit."""
    if "batch_size" not in kwargs:
        return request
    batch_size = kwargs["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise ValueError("batch_size must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    return {**request, "op": request["op"] + "_batched", "batch_size": int(batch_size)}


def _check_model_response(response: dict, request: dict) -> None:
    if response.get("error_code") == "serving_identity_mismatch":
        raise ServingIdentityMismatchError("Model server rejected the selected serving identity")
    if response.get("ok"):
        return
    if response.get("error_code") == "server_busy":
        raise ModelServerBusyError("Model server busy: request capacity exhausted")
    if response.get("error") == f"Unknown op: {request['op']}" and "batch_size" in request:
        raise ProtocolMismatchError(
            "Model server does not support bounded batches; restart it after upgrading"
        )
    raise RuntimeError(f"Model server error: {response.get('error', 'unknown')}")


class EmbeddingProxy:
    """Drop-in replacement for the embedding model with .encode() method."""

    def __init__(self, tier: str = ""):
        self._tier = tier

    def encode(self, texts, timeout: float | None = None, **kwargs) -> np.ndarray:
        """Embed *texts* via the model server.

        *timeout* is an optional per-call deadline in seconds (issue #577);
        on expiry a :class:`TimeoutError` is raised (fast-fail, no autostart
        retry). ``None`` uses the process default / legacy 120s.

        ``batch_size`` is a positive integer ceiling on items per inference
        call. Older daemons must be restarted to support explicit ceilings.
        """
        if isinstance(texts, str):
            texts = [texts]
        request = _batch_request({
            "op": "embed",
            "texts": list(texts),
            "tier": self._tier,
        }, kwargs)
        resp = _request_with_autostart(request, timeout=timeout)
        _check_model_response(resp, request)
        return resp["vectors"]


class PreparedEmbeddingProxy:
    """Strict target requests; no fallback to the ordinary embedding protocol."""

    def __init__(self, target: "EmbeddingTarget") -> None:
        self.target = target

    def _request(self, operation: str, timeout: float | None, **fields: object) -> dict:
        from truememory.embedding_target import EmbeddingTarget, EmbeddingTargetError

        request = {"op": operation, "target": self.target.to_wire(), **fields}
        response = _request_with_autostart(request, timeout=timeout)
        if response.get("error") == f"Unknown op: {operation}":
            raise ProtocolMismatchError(
                "Model server does not support certified embedding targets; restart it after upgrading"
            )
        _check_model_response(response, request)
        try:
            effective = EmbeddingTarget.from_wire(response.get("target"))
        except EmbeddingTargetError as exc:
            raise ProtocolMismatchError("Model server omitted a valid embedding target receipt") from exc
        if effective != self.target:
            raise ProtocolMismatchError("Model server returned a different embedding target")
        return response

    def prepare(self, timeout: float | None = None) -> None:
        self._request("prepare_embed_target_v1", timeout)

    def encode(
        self, texts: str | list[str], *, timeout: float | None = None, batch_size: int = 32,
    ) -> np.ndarray:
        from truememory.embedding_target import check_target_vectors

        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        response = self._request("embed_target_v1", timeout, texts=texts, batch_size=batch_size)
        vectors = response["vectors"]
        check_target_vectors(self.target, vectors, len(texts))
        return vectors


def _check_serving_protocol(response: dict) -> None:
    if type(response.get("protocol")) is not int or response["protocol"] != PROTOCOL_VERSION:
        raise ProtocolMismatchError("Model server omitted the selected-serving protocol; restart it after upgrading")


def _serving_reranker_name(value: object) -> str:
    import re

    if (type(value) is not str or len(value.encode("utf-8")) > 512
            or re.fullmatch(r"[\w][\w.\-]*(/[\w][\w.\-]*)?", value) is None):
        raise ValueError("Selected reranking requires an explicit model identity")
    return value


class CertifiedEmbeddingProxy(EmbeddingProxy):
    """Built-in serving with a fresh identity receipt on every fast/main result.

    Construction configures identity only; it does not contact the daemon or
    prove model readiness. Custom targets use PreparedEmbeddingProxy instead.
    """

    def __init__(self, target: "EmbeddingTarget") -> None:
        from truememory.embedding_target import EmbeddingTarget, EmbeddingTargetError

        self.target = EmbeddingTarget.from_wire(target.to_wire())
        if self.target.tier == "custom":
            raise EmbeddingTargetError("Custom serving requires the prepared target protocol")
        super().__init__(tier=self.target.model_id)

    def encode(self, texts, timeout: float | None = None, **kwargs) -> np.ndarray:
        from truememory.embedding_target import EmbeddingTarget, EmbeddingTargetError, check_target_vectors

        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        request = _batch_request({
            "op": "embed", "texts": texts, "tier": self.target.model_id,
            "expected_target": self.target.to_wire(),
        }, kwargs)
        response = _request_with_autostart(request, timeout=timeout)
        _check_model_response(response, request)
        _check_serving_protocol(response)
        try:
            effective = EmbeddingTarget.from_wire(response.get("target"))
        except EmbeddingTargetError as exc:
            raise ProtocolMismatchError("Model server omitted a valid serving target receipt") from exc
        if effective != self.target:
            raise ProtocolMismatchError("Model server returned a different serving target")
        vectors = response["vectors"]
        check_target_vectors(self.target, vectors, len(texts))
        return vectors


class RerankerProxy:
    """Drop-in replacement for CrossEncoder with .predict() method."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name

    def predict(self, pairs, timeout: float | None = None, **kwargs) -> np.ndarray:
        """Rerank *pairs* via the model server.

        *timeout* is an optional per-call deadline in seconds (issue #577);
        see :meth:`EmbeddingProxy.encode` for deadlines and ``batch_size``.
        """
        request = _batch_request({
            "op": "rerank",
            "pairs": list(pairs),
            "model_name": self._model_name,
        }, kwargs)
        resp = _request_with_autostart(request, timeout=timeout)
        _check_model_response(resp, request)
        return resp["scores"]


class CertifiedRerankerProxy(RerankerProxy):
    """Explicit reranker receipt per response, without eager native readiness."""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name=_serving_reranker_name(model_name))

    def predict(self, pairs, timeout: float | None = None, **kwargs) -> np.ndarray:
        request = _batch_request({
            "op": "rerank", "pairs": list(pairs), "model_name": self._model_name,
            "expected_model_name": self._model_name,
        }, kwargs)
        response = _request_with_autostart(request, timeout=timeout)
        _check_model_response(response, request)
        _check_serving_protocol(response)
        receipt = response.get("reranker")
        if (type(receipt) is not dict or set(receipt) != {"version", "model_name"}
                or type(receipt["version"]) is not int or receipt["version"] != 1
                or type(receipt["model_name"]) is not str or receipt["model_name"] != self._model_name):
            raise ProtocolMismatchError("Model server omitted the exact serving reranker receipt")
        scores = response.get("scores")
        if (not isinstance(scores, np.ndarray) or not scores.shape
                or scores.shape[0] != len(request["pairs"])):
            raise ProtocolMismatchError("Model server returned a different serving reranker result count")
        return scores


def use_model_server() -> bool:
    """Return sharing policy, independently of endpoint readiness.

    Configure local mode before loading models. Existing cached model lifetime
    is unchanged; changing environment variables is not a live mode switch.
    """
    return os.environ.get("TRUEMEMORY_NO_MODEL_SERVER", "") != "1"


def ensure_server_running() -> bool:
    """Start the model server if it's not already running.

    Call from MCP server startup or CLI to enable the shared model server.
    Returns True if server is running after this call.
    """
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
