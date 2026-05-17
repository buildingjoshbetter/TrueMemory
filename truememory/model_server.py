"""Shared model server — loads embedding + reranker models once for all processes.

Run as: python -m truememory.model_server
Or auto-started by model_client on first request.

On POSIX systems (Linux/macOS) the server listens on a Unix domain socket:
    ~/.truememory/model.sock

On Windows, AF_UNIX is unavailable.  The server instead binds a TCP socket on
loopback (127.0.0.1, OS-assigned port) and writes the chosen port to a sidecar
file so the client can discover it:
    ~/.truememory/model_server.port

Auto-exits after idle timeout (default 300s, configurable via
TRUEMEMORY_MODEL_SERVER_IDLE env var).
"""

import gc
import logging
import os
import pickle
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_TRUEMEMORY_DIR = Path.home() / ".truememory"
SOCK_PATH = _TRUEMEMORY_DIR / "model.sock"
PORT_PATH = _TRUEMEMORY_DIR / "model_server.port"   # Windows sidecar: holds TCP port
PID_PATH = _TRUEMEMORY_DIR / "model_server.pid"
IDLE_TIMEOUT = int(os.environ.get("TRUEMEMORY_MODEL_SERVER_IDLE", "300"))

# Use AF_UNIX where available (POSIX), fall back to AF_INET TCP on Windows.
_USE_UNIX_SOCKET = hasattr(socket, "AF_UNIX")

_HEADER_FMT = ">I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class ModelServer:
    """Serves embedding and reranking over a domain/TCP socket."""

    def __init__(self):
        self._embed_model = None
        self._embed_tier: str | None = None
        self._reranker = None
        self._reranker_name: str | None = None
        self._lock = threading.Lock()
        self._last_activity = time.time()
        self._running = True
        # Populated in run() — used by idle_checker to send a dummy wakeup.
        self._bound_addr: str | tuple = ""

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
            with self._lock:
                model = self._get_embed_model(tier)
                vectors = model.encode(texts, show_progress_bar=False)
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

    def handle_client(self, conn: socket.socket):
        try:
            header = self._recv_exact(conn, _HEADER_SIZE)
            if not header:
                return
            length = struct.unpack(_HEADER_FMT, header)[0]
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
                # Send a dummy connection to unblock srv.accept().
                try:
                    if _USE_UNIX_SOCKET:
                        dummy = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        dummy.connect(str(SOCK_PATH))
                    else:
                        addr = self._bound_addr  # ("127.0.0.1", port)
                        dummy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        dummy.connect(addr)
                    dummy.close()
                except Exception:
                    pass
                break

    def run(self):
        _TRUEMEMORY_DIR.mkdir(parents=True, exist_ok=True)

        PID_PATH.write_text(str(os.getpid()))

        if _USE_UNIX_SOCKET:
            # POSIX path — Unix domain socket
            if SOCK_PATH.exists():
                SOCK_PATH.unlink()
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(str(SOCK_PATH))
            self._bound_addr = str(SOCK_PATH)
            log.info(
                "Model server started (AF_UNIX): pid=%d sock=%s idle_timeout=%ds",
                os.getpid(), SOCK_PATH, IDLE_TIMEOUT,
            )
        else:
            # Windows path — loopback TCP, OS-assigned port
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", 0))
            port = srv.getsockname()[1]
            PORT_PATH.write_text(str(port), encoding="utf-8")
            self._bound_addr = ("127.0.0.1", port)
            log.info(
                "Model server started (AF_INET): pid=%d port=%d idle_timeout=%ds",
                os.getpid(), port, IDLE_TIMEOUT,
            )

        srv.listen(16)
        srv.settimeout(2.0)

        idle_thread = threading.Thread(target=self._idle_checker, daemon=True)
        idle_thread.start()

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
                t = threading.Thread(target=self.handle_client, args=(conn,), daemon=True)
                t.start()
        finally:
            srv.close()
            self._cleanup()

    def _cleanup(self):
        if _USE_UNIX_SOCKET and SOCK_PATH.exists():
            SOCK_PATH.unlink(missing_ok=True)
        if not _USE_UNIX_SOCKET and PORT_PATH.exists():
            PORT_PATH.unlink(missing_ok=True)
        if PID_PATH.exists():
            PID_PATH.unlink(missing_ok=True)
        self._embed_model = None
        self._reranker = None
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

    # Register SIGTERM and SIGINT on all platforms.
    # SIGHUP is POSIX-only — guard before registering.
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_signal)

    if PID_PATH.exists():
        try:
            old_pid = int(PID_PATH.read_text().strip())
            os.kill(old_pid, 0)
            log.error("Model server already running (pid=%d)", old_pid)
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            PID_PATH.unlink(missing_ok=True)
            if SOCK_PATH.exists():
                SOCK_PATH.unlink(missing_ok=True)
            if PORT_PATH.exists():
                PORT_PATH.unlink(missing_ok=True)

    server = ModelServer()
    server.run()


if __name__ == "__main__":
    main()
