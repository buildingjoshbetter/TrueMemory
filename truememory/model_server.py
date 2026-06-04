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

import gc  # noqa: E402
import hmac  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import secrets  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
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

# Maximum request payload size (100 MB) — reject before allocating memory.
MAX_REQUEST_SIZE = 100 * 1024 * 1024

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
        self._last_activity = time.time()
        self._running = True
        self._embed_timestamps: list[float] = []
        self._throttler = None
        self._throttler_active = False
        self._token: bytes | None = None  # HMAC token for TCP transport
        self._bound_port: int | None = None  # TCP port (Windows only)

    def _get_embed_model(self, tier: str):
        if self._embed_model is not None and self._embed_tier == tier:
            return self._embed_model

        from truememory.vector_search import EMBEDDING_MODEL, set_embedding_model

        if tier and tier != EMBEDDING_MODEL:
            set_embedding_model(tier)

        resolved = EMBEDDING_MODEL if not tier else tier
        from truememory.vector_search import _TIER_ALIASES
        model_id = _TIER_ALIASES.get(resolved, resolved)

        if model_id == "model2vec":
            from model2vec import StaticModel
            self._embed_model = StaticModel.from_pretrained(
                "minishlab/potion-base-8M", force_download=False
            )
        elif model_id == "qwen3_256":
            from sentence_transformers import SentenceTransformer
            mkwargs = {}
            if sys.platform == "darwin":
                mkwargs["attn_implementation"] = "eager"
            self._embed_model = SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                truncate_dim=256,
                model_kwargs=mkwargs or None,
            )
        else:
            from model2vec import StaticModel
            self._embed_model = StaticModel.from_pretrained(
                "minishlab/potion-base-8M", force_download=False
            )

        self._embed_tier = tier
        log.info("Loaded embedding model for tier=%s", tier)
        return self._embed_model

    def _get_reranker(self, model_name: str | None = None):
        from truememory.reranker import get_current_reranker_name
        name = model_name or get_current_reranker_name()

        if self._reranker is not None and self._reranker_name == name:
            return self._reranker

        from sentence_transformers import CrossEncoder
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except ImportError:
            pass

        self._reranker = CrossEncoder(name, device=device)
        self._reranker_name = name
        log.info("Loaded reranker model=%s device=%s", name, device)
        return self._reranker

    def handle_request(self, request: dict) -> dict:
        self._last_activity = time.time()
        op = request.get("op")

        if op == "ping":
            return {"ok": True}

        if op == "embed":
            texts = request["texts"]
            tier = request.get("tier", "")

            now = time.time()
            self._embed_timestamps.append(now)
            self._embed_timestamps = [
                t for t in self._embed_timestamps
                if now - t < self._SUSTAINED_WINDOW
            ]

            if (len(self._embed_timestamps) >= self._SUSTAINED_THRESHOLD
                    and not self._throttler_active):
                self._activate_throttler()

            if self._throttler_active and self._throttler:
                self._throttler.before_batch()

            encode_start = time.time()
            with self._lock:
                model = self._get_embed_model(tier)
                vectors = model.encode(texts, show_progress_bar=False)
            encode_time = time.time() - encode_start

            if self._throttler_active and self._throttler:
                self._throttler.after_batch(len(texts), encode_time)
                if self._throttler.should_flush_cache():
                    self._flush_mps_cache()

            if self._throttler_active and len(self._embed_timestamps) < 3:
                self._deactivate_throttler()

            return {"ok": True, "vectors": np.asarray(vectors, dtype=np.float32)}

        if op == "rerank":
            pairs = request["pairs"]
            model_name = request.get("model_name")
            with self._lock:
                reranker = self._get_reranker(model_name)
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

    def handle_client(self, conn: socket.socket):
        try:
            # --- HMAC authentication for TCP transport ---
            if not _USE_UNIX and self._token is not None:
                token_bytes = self._recv_exact(conn, _HMAC_TOKEN_BYTES)
                if token_bytes is None or not hmac.compare_digest(
                    token_bytes, self._token
                ):
                    log.warning("TCP client failed HMAC authentication")
                    conn.close()
                    return

            header = self._recv_exact(conn, _HEADER_SIZE)
            if not header:
                return
            length = struct.unpack(_HEADER_FMT, header)[0]

            # Reject oversized requests before allocating memory.
            if length > MAX_REQUEST_SIZE:
                log.warning(
                    "Rejecting oversized request (%d bytes, max %d)",
                    length,
                    MAX_REQUEST_SIZE,
                )
                conn.close()
                return

            data = self._recv_exact(conn, length)
            if not data:
                return

            request = pickle.loads(data)
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
        data = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        header = struct.pack(_HEADER_FMT, len(data))
        conn.sendall(header + data)

    def _idle_checker(self):
        while self._running:
            time.sleep(60)
            if not self._running:
                break
            elapsed = time.time() - self._last_activity
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
    def _atomic_write_text(path: Path, text: str) -> None:
        """Write *text* to *path* atomically via a temp file + rename."""
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text)
        try:
            os.replace(str(tmp), str(path))
        except OSError:
            # os.replace can fail on Windows if another process holds the
            # file open.  Fall back to direct write.
            path.write_text(text)
            tmp.unlink(missing_ok=True)

    def run(self):
        _TRUEMEMORY_DIR.mkdir(parents=True, exist_ok=True)

        PID_PATH.write_text(str(os.getpid()))

        if _USE_UNIX:
            # ----- POSIX: AF_UNIX transport (unchanged behaviour) -----
            if SOCK_PATH.exists():
                SOCK_PATH.unlink()

            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(str(SOCK_PATH))
            transport_desc = f"sock={SOCK_PATH}"
        else:
            # ----- Windows: TCP loopback transport -----
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.bind((_LOOPBACK_HOST, 0))  # ephemeral port
            self._bound_port = srv.getsockname()[1]

            # Write port file so the client can discover us.
            self._atomic_write_text(PORT_PATH, str(self._bound_port))
            try:
                os.chmod(PORT_PATH, 0o600)
            except OSError:
                pass  # chmod may not be meaningful on all Windows builds

            # Generate and persist HMAC auth token.
            self._token = secrets.token_bytes(_HMAC_TOKEN_BYTES)
            self._atomic_write_text(TOKEN_PATH, self._token.hex())
            try:
                os.chmod(TOKEN_PATH, 0o600)
            except OSError:
                pass

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
        # Stale PID file -- clean up leftover artefacts.
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
