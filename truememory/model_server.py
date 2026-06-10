"""Shared model server — loads embedding + reranker models once for all processes.

Run as: python -m truememory.model_server
Or auto-started by model_client on first request.

Transport:
  - POSIX: AF_UNIX socket at ~/.truememory/model.sock
  - Windows: TCP loopback (127.0.0.1) on an ephemeral port written to
    ~/.truememory/model_server.port, authenticated via HMAC token stored
    in ~/.truememory/model_server.token (chmod 0o600).

Auto-exits after idle timeout (default 300s, configurable via
TRUEMEMORY_MODEL_SERVER_IDLE env var).
"""

import atexit
import os

try:
    import psutil
except ImportError:
    psutil = None


def _set_mps_memory_cap():
    """Set MPS memory cap and BLAS thread limits BEFORE torch is imported."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
        return
    if psutil is not None:
        total_gb = psutil.virtual_memory().total / (1024**3)
        ratio = min(0.08, 2.5 / total_gb) if total_gb >= 16 else 0.19
        ratio = str(max(ratio, 1.5 / total_gb))
    else:
        ratio = "0.19"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = ratio
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")


_set_mps_memory_cap()

import base64  # noqa: E402
import gc  # noqa: E402
import hmac  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import secrets  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import stat  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from truememory._platform import _LOOPBACK_HOST, _USE_UNIX, pid_is_alive  # noqa: E402

log = logging.getLogger(__name__)

_TRUEMEMORY_DIR = Path.home() / ".truememory"
SOCK_PATH = _TRUEMEMORY_DIR / "model.sock"
PID_PATH = _TRUEMEMORY_DIR / "model_server.pid"
PORT_PATH = _TRUEMEMORY_DIR / "model_server.port"
TOKEN_PATH = _TRUEMEMORY_DIR / "model_server.token"
IDLE_TIMEOUT = int(os.environ.get("TRUEMEMORY_MODEL_SERVER_IDLE", "300"))

_HEADER_FMT = ">I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB


def _json_default(obj):
    """Encode numpy arrays as base64 for safe JSON serialization."""
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj, dtype=np.float32)
        return {
            "__ndarray__": base64.b64encode(arr.tobytes()).decode("ascii"),
            "dtype": "float32",
            "shape": list(arr.shape),
        }
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


_ALLOWED_DTYPES = frozenset({"float32", "float64", "float16", "int32", "int64"})


def _json_object_hook(obj):
    """Decode base64-encoded numpy arrays from JSON."""
    if "__ndarray__" in obj:
        dtype_str = obj["dtype"]
        if dtype_str not in _ALLOWED_DTYPES:
            raise ValueError(f"Disallowed dtype: {dtype_str}")
        data = base64.b64decode(obj["__ndarray__"])
        return np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(obj["shape"])
    return obj

# Maximum request payload size (100 MB) — reject before allocating memory.
_HMAC_TOKEN_BYTES = 32


class ModelServer:
    """Serves embedding and reranking over a Unix domain socket (POSIX)
    or HMAC-authenticated TCP loopback (Windows)."""

    _SUSTAINED_THRESHOLD = 10
    _SUSTAINED_WINDOW = 30

    def __init__(self):
        self._embed_model = None
        self._embed_tier: str | None = None
        self._reranker = None
        self._reranker_name: str | None = None
        self._lock = threading.Lock()
        # Issue #577: idle-tracking gets its own lock so the single-text
        # fast lane (and the idle checker) never block on the global
        # request lock while a batch encode is in flight.
        self._activity_lock = threading.Lock()
        self._last_activity = time.time()
        self._running = True
        self._embed_timestamps: list[float] = []
        self._throttler = None
        self._throttler_active = False
        self._token: bytes | None = None  # HMAC token for TCP transport
        self._bound_port: int | None = None  # TCP port (Windows only)
        # Issue #577: models that hit an MPS OOM are degraded to CPU for the
        # lifetime of this server process ("embed" / "rerank"). Mutated only
        # while holding self._lock; read lock-free (string membership).
        self._sticky_cpu: set[str] = set()

    def _mark_sticky_cpu(self, kind: str) -> bool:
        """Permanently degrade *kind* ("embed"/"rerank") to CPU after an
        MPS OOM (issue #577). Re-promoting to MPS after recovery guaranteed
        the next OOM (the pool never drops below the watermark cap), so the
        degradation is sticky for the server's lifetime.

        Caller must hold ``self._lock``. Returns True on the first marking
        (which is logged loudly, once).
        """
        if kind in self._sticky_cpu:
            return False
        self._sticky_cpu.add(kind)
        log.error(
            "MPS OOM in the %s path — degrading the %s model to CPU for the "
            "lifetime of this model server (no MPS re-promotion). Restart "
            "the server to try MPS again, or set TRUEMEMORY_DEVICE=cpu to "
            "make CPU permanent.",
            kind, kind,
        )
        return True

    def _embed_device(self) -> str | None:
        """Device for embed model loads: sticky-CPU > TRUEMEMORY_DEVICE >
        framework auto-selection (None)."""
        if "embed" in self._sticky_cpu:
            return "cpu"
        from truememory.mps_utils import resolve_device
        return resolve_device(None)

    @staticmethod
    def _resolve_embed_model_id(tier: str) -> str:
        """Resolve a tier name to the internal embedding model ID."""
        from truememory.vector_search import EMBEDDING_MODEL, set_embedding_model

        if tier and tier != EMBEDDING_MODEL:
            set_embedding_model(tier)

        resolved = EMBEDDING_MODEL if not tier else tier
        # Resolve tier -> internal model ID via centralized tier_config.
        # _TIER_ALIASES is still exported by vector_search for compat.
        from truememory.vector_search import _TIER_ALIASES
        model_id = _TIER_ALIASES.get(resolved, resolved)

        # Custom tier: resolve via tier_config
        if resolved == "custom":
            try:
                from truememory.tier_config import get_embed_model
                model_id = get_embed_model("custom")
            except (ValueError, ImportError) as e:
                log.warning("Custom tier resolution failed (%s); falling back to model2vec.", e)
                model_id = "model2vec"
        return model_id

    @staticmethod
    def _build_embed_model(model_id: str, device: str | None):
        """Construct an embedding model. ``device=None`` lets the framework
        pick (SentenceTransformer auto-selects; model2vec is CPU-only)."""
        if model_id == "model2vec":
            from model2vec import StaticModel
            return StaticModel.from_pretrained(
                "minishlab/potion-base-8M", force_download=False
            )
        if model_id == "qwen3_256":
            from sentence_transformers import SentenceTransformer
            mkwargs = {}
            if sys.platform == "darwin":
                mkwargs["attn_implementation"] = "eager"
            return SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                truncate_dim=256,
                model_kwargs=mkwargs or None,
                device=device,
            )
        if model_id not in ("model2vec", "minilm", "bge-small", "qwen3_256"):
            # Custom model: require explicit opt-in for arbitrary downloads
            if os.environ.get("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "").strip() != "1":
                log.warning(
                    "Custom model %r requested without "
                    "TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD=1 -- "
                    "falling back to model2vec.",
                    model_id,
                )
                from model2vec import StaticModel
                return StaticModel.from_pretrained(
                    "minishlab/potion-base-8M", force_download=False
                )
            from sentence_transformers import SentenceTransformer
            from truememory.tier_config import resolve_custom_tier
            cfg = resolve_custom_tier()
            custom_dim = cfg["embed_dim"]
            return SentenceTransformer(
                model_id, truncate_dim=custom_dim,
                trust_remote_code=False,
                device=device,
            )
        from model2vec import StaticModel
        return StaticModel.from_pretrained(
            "minishlab/potion-base-8M", force_download=False
        )

    def _get_embed_model(self, tier: str):
        if self._embed_model is not None and self._embed_tier == tier:
            return self._embed_model

        model_id = self._resolve_embed_model_id(tier)
        self._embed_model = self._build_embed_model(model_id, self._embed_device())
        self._embed_tier = tier
        log.info("Loaded embedding model for tier=%s", tier)
        return self._embed_model

    def _get_reranker(self, model_name: str | None = None):
        from truememory.reranker import get_current_reranker_name
        name = model_name or get_current_reranker_name()

        if self._reranker is not None and self._reranker_name == name:
            return self._reranker

        from sentence_transformers import CrossEncoder
        from truememory.mps_utils import auto_detect_device, resolve_device
        if "rerank" in self._sticky_cpu:
            device = "cpu"
        else:
            device = resolve_device(auto_detect_device())

        self._reranker = CrossEncoder(name, device=device)
        self._reranker_name = name
        log.info("Loaded reranker model=%s device=%s", name, device)
        return self._reranker

    def handle_request(self, request: dict) -> dict:
        with self._activity_lock:
            self._last_activity = time.time()
        op = request.get("op")

        if op == "ping":
            return {"ok": True}

        if op == "embed":
            texts = request["texts"]
            tier = request.get("tier", "")

            now = time.time()
            with self._lock:
                self._embed_timestamps.append(now)
                self._embed_timestamps = [
                    t for t in self._embed_timestamps
                    if now - t < self._SUSTAINED_WINDOW
                ]
                should_activate = (
                    len(self._embed_timestamps) >= self._SUSTAINED_THRESHOLD
                    and not self._throttler_active
                )

            if should_activate:
                self._activate_throttler()

            if self._throttler_active and self._throttler:
                self._throttler.before_batch()

            encode_start = time.time()
            oom = False
            with self._lock:
                model = self._get_embed_model(tier)
                try:
                    vectors = model.encode(texts, show_progress_bar=False)
                except RuntimeError as exc:
                    from truememory.mps_utils import is_mps_oom, flush_mps_cache
                    if not is_mps_oom(exc):
                        raise
                    # Issue #577: sticky-CPU degradation. The device move
                    # happens under the lock (transitions must not race other
                    # encodes) but the expensive full re-encode runs OUTSIDE
                    # the lock so other clients are not starved for 23-112s.
                    # The model is never re-promoted to MPS: the old
                    # re-promotion left the MPS pool above its watermark cap
                    # and guaranteed the next OOM (retry storms).
                    self._mark_sticky_cpu("embed")
                    flush_mps_cache()
                    if hasattr(model, "to"):
                        model.to("cpu")
                    oom = True
            if oom:
                log.warning(
                    "MPS OOM during encoding — retrying on CPU outside the "
                    "request lock"
                )
                vectors = model.encode(texts, show_progress_bar=False)
            encode_time = time.time() - encode_start

            if self._throttler_active and self._throttler:
                self._throttler.after_batch(len(texts), encode_time)
                if self._throttler.should_flush_cache():
                    self._flush_mps_cache()

            with self._lock:
                should_deactivate = self._throttler_active and len(self._embed_timestamps) < 3
            if should_deactivate:
                self._deactivate_throttler()

            return {"ok": True, "vectors": np.asarray(vectors, dtype=np.float32)}

        if op == "rerank":
            pairs = request["pairs"]
            model_name = request.get("model_name")
            oom = False
            with self._lock:
                reranker = self._get_reranker(model_name)
                try:
                    scores = reranker.predict(
                        pairs, batch_size=64, show_progress_bar=False
                    )
                except RuntimeError as exc:
                    from truememory.mps_utils import is_mps_oom, flush_mps_cache
                    if not is_mps_oom(exc):
                        raise
                    # Issue #577: the rerank path previously had NO OOM
                    # handler — a raw MPS OOM reached the client (3/3 rerank
                    # deaths in the baseline probe). Same sticky-CPU policy
                    # as embed: drop the MPS-loaded model and reload on CPU.
                    self._mark_sticky_cpu("rerank")
                    flush_mps_cache()
                    self._reranker = None
                    self._reranker_name = None
                    oom = True
            if oom:
                with self._lock:
                    reranker = self._get_reranker(model_name)  # sticky → CPU
                # Retry outside the lock — rerank batches are the expensive
                # part; other clients must not starve behind the recovery.
                scores = reranker.predict(
                    pairs, batch_size=64, show_progress_bar=False
                )
            return {"ok": True, "scores": np.asarray(scores, dtype=np.float32)}

        return {"ok": False, "error": f"Unknown op: {op}"}

    def _activate_throttler(self):
        """Start adaptive throttling for sustained workload."""
        try:
            from truememory.tier_switch.throttler import DynamicThrottler
        except ImportError:
            log.warning("Cannot import DynamicThrottler — running without throttling")
            return
        # Issue #577: honor sticky-CPU degradation and TRUEMEMORY_DEVICE in
        # the throttler's device pick (it tunes MPS-specific behavior).
        if "embed" in self._sticky_cpu:
            device = "cpu"
        else:
            from truememory.mps_utils import resolve_device
            device = resolve_device(None)
            if device is None:
                device = "cpu"
                try:
                    import torch
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        device = "mps"
                except ImportError:
                    pass
        self._throttler = DynamicThrottler(device=device)
        self._throttler_active = True
        log.info(
            "Sustained workload detected (%d requests in %ds) — throttler activated",
            len(self._embed_timestamps), self._SUSTAINED_WINDOW,
        )

    def _deactivate_throttler(self):
        """Stop adaptive throttling — workload ended."""
        self._throttler = None
        self._throttler_active = False
        self._embed_timestamps.clear()
        log.info("Workload ended — throttler deactivated")

    def _flush_mps_cache(self):
        """Flush MPS cache — only called when throttler says to."""
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
        except Exception:
            pass
        gc.collect()

    _CLIENT_TIMEOUT = 30.0  # seconds; caps how long any single client can block

    def handle_client(self, conn: socket.socket):
        try:
            conn.settimeout(self._CLIENT_TIMEOUT)

            # --- HMAC authentication for TCP transport ---
            if not _USE_UNIX:
                if self._token is None:
                    # Fail closed: TCP transport must always have a token.
                    log.warning("TCP client rejected: no HMAC token configured")
                    conn.close()
                    return
                token_bytes = self._recv_exact(conn, _HMAC_TOKEN_BYTES)
                if token_bytes is None:
                    # Connection closed before sending token (e.g. idle-timeout
                    # dummy connection).  Drop silently -- not a real auth failure.
                    conn.close()
                    return
                if not hmac.compare_digest(token_bytes, self._token):
                    log.warning("TCP client failed HMAC authentication")
                    conn.close()
                    return

            header = self._recv_exact(conn, _HEADER_SIZE)
            if not header:
                return
            length = struct.unpack(_HEADER_FMT, header)[0]
            if length > _MAX_MESSAGE_SIZE:
                log.warning(
                    "Rejecting oversized request (%d bytes, max %d)",
                    length,
                    _MAX_MESSAGE_SIZE,
                )
                conn.close()
                return
            data = self._recv_exact(conn, length)
            if not data:
                return

            request = json.loads(data, object_hook=_json_object_hook)
            response = self.handle_request(request)
            self._send_response(conn, response)
        except Exception as e:
            try:
                self._send_response(conn, {"ok": False, "error": str(e)})
            except Exception:
                pass
        finally:
            conn.close()

    def _recv_exact(self, conn: socket.socket, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _send_response(self, conn: socket.socket, response: dict):
        data = json.dumps(response, default=_json_default).encode("utf-8")
        if len(data) > _MAX_MESSAGE_SIZE:
            data = json.dumps({"ok": False, "error": "Response too large"}).encode("utf-8")
        header = struct.pack(_HEADER_FMT, len(data))
        conn.sendall(header + data)

    def _idle_checker(self):
        while self._running:
            time.sleep(60)
            if not self._running:
                break
            with self._activity_lock:
                last = self._last_activity
            elapsed = time.time() - last
            if elapsed >= IDLE_TIMEOUT:
                log.info(
                    "Idle timeout (%.0fs), shutting down model server", elapsed
                )
                self._running = False
                # Send a dummy connection to unblock accept().
                try:
                    if _USE_UNIX:
                        dummy = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        dummy.connect(str(SOCK_PATH))
                    else:
                        port = self._bound_port
                        if port:
                            dummy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            dummy.connect((_LOOPBACK_HOST, port))
                        else:
                            break
                    dummy.close()
                except Exception:
                    pass
                break

    @staticmethod
    def _atomic_write_text(path: Path, text: str, mode: int = 0o644) -> None:
        """Write *text* to *path* atomically via a temp file + rename.

        *mode* is applied to the temp file **before** the rename so the
        target is never visible with default permissions (eliminates the
        TOCTOU window for sensitive files like the HMAC token).
        """
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        # Create with restricted permissions from the start.
        fd = os.open(str(tmp), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        try:
            os.replace(str(tmp), str(path))
        except OSError:
            # os.replace can fail on Windows if another process holds the
            # file open.  Fall back to direct write with restricted perms.
            fd2 = os.open(str(path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
            with os.fdopen(fd2, "w") as f2:
                f2.write(text)
            tmp.unlink(missing_ok=True)

    def run(self):
        _TRUEMEMORY_DIR.mkdir(parents=True, exist_ok=True)

        PID_PATH.write_text(str(os.getpid()))

        if _USE_UNIX:
            if SOCK_PATH.exists():
                SOCK_PATH.unlink()
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(str(SOCK_PATH))
            os.chmod(str(SOCK_PATH), stat.S_IRUSR | stat.S_IWUSR)
            transport_desc = f"sock={SOCK_PATH}"
        else:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.bind((_LOOPBACK_HOST, 0))
            self._bound_port = srv.getsockname()[1]
            self._token = secrets.token_bytes(_HMAC_TOKEN_BYTES)
            self._atomic_write_text(TOKEN_PATH, self._token.hex(), mode=0o600)
            self._atomic_write_text(PORT_PATH, str(self._bound_port), mode=0o600)
            transport_desc = f"tcp={_LOOPBACK_HOST}:{self._bound_port}"
        srv.listen(16)
        srv.settimeout(2.0)

        idle_thread = threading.Thread(target=self._idle_checker, daemon=True)
        idle_thread.start()

        log.info(
            "Model server started: pid=%d %s idle_timeout=%ds",
            os.getpid(), transport_desc, IDLE_TIMEOUT,
        )

        try:
            while self._running:
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not self._running:
                    conn.close()
                    break
                t = threading.Thread(
                    target=self.handle_client, args=(conn,), daemon=True
                )
                t.start()
        finally:
            srv.close()
            self._cleanup()

    def _cleanup(self):
        if getattr(self, "_cleaned_up", False):
            return
        self._cleaned_up = True
        for p in (SOCK_PATH, PID_PATH, PORT_PATH, TOKEN_PATH):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        self._embed_model = None
        self._reranker = None
        self._token = None
        gc.collect()
        log.info("Model server stopped")


def _handle_signal(signum, frame):
    log.info("Received signal %d, shutting down", signum)
    sys.exit(0)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [model_server] %(levelname)s %(message)s",
    )

    try:
        import setproctitle
        setproctitle.setproctitle("TrueMemory")
    except ImportError:
        pass

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_signal)

    if PID_PATH.exists():
        try:
            old_pid = int(PID_PATH.read_text().strip())
            if pid_is_alive(old_pid):
                log.error("Model server already running (pid=%d)", old_pid)
                sys.exit(1)
        except (ValueError, OSError):
            pass
        PID_PATH.unlink(missing_ok=True)
        if SOCK_PATH.exists():
            SOCK_PATH.unlink(missing_ok=True)
        PORT_PATH.unlink(missing_ok=True)
        TOKEN_PATH.unlink(missing_ok=True)

    server = ModelServer()

    # Ensure cleanup runs even on unhandled exit.
    atexit.register(server._cleanup)

    server.run()


if __name__ == "__main__":
    main()
