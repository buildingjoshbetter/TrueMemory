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
from dataclasses import dataclass  # noqa: E402
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

LOCK_PATH = _TRUEMEMORY_DIR / "model_server.lock"

# Issue #646 (M-53): protocol/version handshake. Bumped whenever the wire
# format changes incompatibly. The client echoes this back on mismatch.
PROTOCOL_VERSION = 1

_HEADER_FMT = ">I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB


def _safe_to_cleanup_artifacts() -> bool:
    """Return True when this process may remove the shared artifacts (M-20).

    ``_cleanup`` must NOT unlink a live successor's socket/pid/port/token:
    after a crash a fresh server can already hold the bind lock and have
    rewritten PID_PATH, and tearing down its files lets concurrent hooks
    cycle servers indefinitely.

    It is safe to clean when PID_PATH names *this* process, or when it names
    nothing live (missing file, unparseable, or a dead PID — a stale crash
    artifact). The ONLY case we refuse is PID_PATH naming a *different live*
    process — the successor that now owns the artifacts.
    """
    try:
        on_disk = int(PID_PATH.read_text().strip())
    except (ValueError, OSError):
        return True  # missing / unparseable — nothing live owns it
    if on_disk == os.getpid():
        return True
    # A different PID is recorded — only refuse if it's actually alive.
    return not pid_is_alive(on_disk)


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


@dataclass(frozen=True)
class _EmbedState:
    """Immutable snapshot of the loaded embedding model and its identity.

    Issue #577 (panel round 2): the server stores this in a SINGLE attribute
    (``ModelServer._embed_state``) assigned only while holding the global
    request lock. Lock-free readers (the single-text fast lane) read one
    reference — Python reference assignment is atomic, so a reader can never
    observe a torn ``(model, tier, model_id)`` triple.
    """

    model: object
    tier: str
    model_id: str


class ModelServer:
    """Serves embedding and reranking over a Unix domain socket (POSIX)
    or HMAC-authenticated TCP loopback (Windows)."""

    _SUSTAINED_THRESHOLD = 10
    _SUSTAINED_WINDOW = 30
    # Requests with at most this many texts take the single-text fast lane
    # (issue #577): hook recall queries must never queue behind ingestion
    # batches or OOM recovery.
    _FAST_LANE_MAX_TEXTS = 1

    def __init__(self):
        # Loaded embedding model + identity as ONE immutable snapshot,
        # assigned only under self._lock; the fast lane reads the single
        # reference lock-free (issue #577, panel round 2).
        self._embed_state: _EmbedState | None = None
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
        # Issue #577: dedicated CPU encoder for the single-text fast lane,
        # loaded lazily on the first *contended* single-text request.
        # Cached by MODEL ID (not tier string) so it always tracks the main
        # path's actual loaded identity.
        self._fast_encoder = None
        self._fast_model_id: str | None = None
        self._fast_lock = threading.Lock()
        # Issue #646 (M-20): exclusive bind-lock fd, held for the process
        # lifetime; released in _cleanup. None until run() acquires it.
        self._lock_fd: int | None = None
        # Issue #646 (M-74): in-flight request counter so idle shutdown never
        # kills a request mid-encode. Guarded by _activity_lock.
        self._inflight = 0

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
        self._write_status_file()
        return True

    def _write_status_file(self) -> None:
        """Persist sticky-CPU state so truememory_status can read it (issue #592)."""
        try:
            status_path = Path.home() / ".truememory" / "model_server.status"
            status_path.write_text(json.dumps({
                "sticky_cpu": sorted(self._sticky_cpu),
                "pid": os.getpid(),
                "updated": time.time(),
            }))
        except Exception:
            pass  # best-effort; don't crash the server

    def _embed_device(self) -> str | None:
        """Device for embed model loads: sticky-CPU > TRUEMEMORY_DEVICE >
        framework auto-selection (None)."""
        if "embed" in self._sticky_cpu:
            return "cpu"
        from truememory.mps_utils import resolve_device
        return resolve_device(None)

    def _recover_embed_oom_locked(self, model) -> None:
        """The single recovery path for an embed MPS OOM (issue #577).

        Caller MUST hold ``self._lock``. Marks the embed path sticky-CPU
        (loud log once), flushes the MPS cache, and moves the model to CPU —
        all atomically in the caller's lock hold. Retry placement belongs to
        the caller: single texts re-encode inside the lock (ms-scale);
        batches re-encode after releasing it so other clients don't starve.
        """
        from truememory.mps_utils import flush_mps_cache
        self._mark_sticky_cpu("embed")
        flush_mps_cache()
        if hasattr(model, "to"):
            model.to("cpu")

    @staticmethod
    def _peek_embed_model_id(tier: str) -> str:
        """Resolve a tier name to the internal embedding model ID as a PURE
        READ — no mutation of vector_search globals (issue #577 panel: the
        fast lane must never call ``set_embedding_model``, which force-unloads
        the process-local model singleton and rewrites ``EMBEDDING_MODEL`` /
        ``_embedding_dim`` while the main path may be mid-encode)."""
        from truememory.vector_search import EMBEDDING_MODEL, _TIER_ALIASES

        resolved = EMBEDDING_MODEL if not tier else tier
        # Resolve tier -> internal model ID via centralized tier_config.
        # _TIER_ALIASES is still exported by vector_search for compat.
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
    def _resolve_embed_model_id(tier: str) -> str:
        """Resolve a tier name to the internal embedding model ID, keeping
        vector_search's globals in sync (main-path behavior, pre-#577)."""
        from truememory.vector_search import EMBEDDING_MODEL, set_embedding_model

        if tier and tier != EMBEDDING_MODEL:
            set_embedding_model(tier)

        return ModelServer._peek_embed_model_id(tier)

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

    # Model IDs the fast lane may rebuild with guaranteed vector-space
    # parity: ONLY ids with an explicit, deterministic branch in
    # _build_embed_model (fixed dim, no config reads). "minilm"/"bge-small"
    # have NO explicit server branch (legacy fall-through to model2vec), and
    # custom models re-read config.json at build time — the fast lane
    # declines all of those and falls through to the main path.
    _FAST_LANE_SAFE_MODEL_IDS = frozenset({"model2vec", "qwen3_256"})

    def _get_embed_model(self, tier: str):
        state = self._embed_state
        if state is not None and state.tier == tier:
            return state.model

        model_id = self._resolve_embed_model_id(tier)
        model = self._build_embed_model(model_id, self._embed_device())
        # ONE atomic reference assignment of an immutable snapshot — the
        # fast lane can never observe a torn (model, tier, model_id) triple.
        self._embed_state = _EmbedState(model=model, tier=tier, model_id=model_id)
        log.info("Loaded embedding model for tier=%s (model=%s)", tier, model_id)
        return model

    def _get_fast_encoder(self, tier: str):
        """CPU-resident encoder for the single-text fast lane (issue #577).

        Loaded lazily on the first single-text request that finds the global
        lock busy, then kept for the server's lifetime. Caller must hold
        ``self._fast_lock``.

        Vector-space parity (panel rounds 1-2): the fast encoder is ONLY
        ever built from the main path's actual loaded identity, taken from
        one atomic read of the ``_embed_state`` snapshot. The fast lane
        never resolves tiers itself — no global reads, no config reads, no
        drift. Returns ``None`` (decline → caller falls through to the main
        locked path) when there is no snapshot for this tier yet, or the
        loaded model has no explicit deterministic builder branch.
        """
        state = self._embed_state  # single atomic reference read
        if state is None or state.tier != tier:
            # No main model loaded for this tier yet — declining means the
            # one-time load happens exactly once, on the main path, with
            # its usual global sync.
            return None

        if state.model_id not in self._FAST_LANE_SAFE_MODEL_IDS:
            return None

        if self._fast_encoder is not None and self._fast_model_id == state.model_id:
            return self._fast_encoder

        self._fast_encoder = self._build_embed_model(state.model_id, "cpu")
        self._fast_model_id = state.model_id
        log.info(
            "Fast-lane CPU encoder loaded (tier=%s, model=%s) — single-text "
            "requests no longer queue behind batch work", tier, state.model_id,
        )
        return self._fast_encoder

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

    def _handle_fast_embed(self, texts: list, tier: str) -> dict | None:
        """Single-text fast lane (issue #577).

        Tries the main path without blocking; when the global lock is busy
        (a batch encode or OOM recovery is in progress) the text is encoded
        on a dedicated CPU encoder OUTSIDE the global lock, so hook recall
        queries never wait on ingestion work. Fast-lane requests skip the
        sustained-workload bookkeeping entirely — a burst of small hook
        queries must not trip the throttler's batch=1 ramp (finding C-7).

        Returns a response dict, or None to fall through to the normal
        (locked) path.
        """
        if self._lock.acquire(blocking=False):
            try:
                model = self._get_embed_model(tier)
                try:
                    vectors = model.encode(texts, show_progress_bar=False)
                except RuntimeError as exc:
                    from truememory.mps_utils import is_mps_oom
                    if not is_mps_oom(exc):
                        raise
                    # Same recovery path as the batch handler, atomic in
                    # this lock hold; a single text on CPU is ms-scale, so
                    # the retry stays under the lock too.
                    self._recover_embed_oom_locked(model)
                    vectors = model.encode(texts, show_progress_bar=False)
                return {"ok": True, "vectors": np.asarray(vectors, dtype=np.float32)}
            finally:
                self._lock.release()

        try:
            with self._fast_lock:
                model = self._get_fast_encoder(tier)
                if model is None:
                    # Vector-space parity with the main path cannot be
                    # guaranteed (custom model) — decline the fast lane.
                    return None
                vectors = model.encode(texts, show_progress_bar=False)
            return {"ok": True, "vectors": np.asarray(vectors, dtype=np.float32)}
        except Exception:
            log.warning(
                "Fast-lane CPU encode failed — falling back to the main "
                "embed path", exc_info=True,
            )
            return None

    def handle_request(self, request: dict) -> dict:
        # Issue #646 (M-74): count this request as in-flight and stamp
        # activity at START *and* completion. The in-flight counter keeps the
        # idle checker from shutting the server down mid-encode; re-stamping
        # at completion means a long encode that finishes near the idle
        # horizon isn't reaped the instant it returns.
        with self._activity_lock:
            self._last_activity = time.time()
            self._inflight += 1
        try:
            return self._handle_request_inner(request)
        finally:
            with self._activity_lock:
                self._inflight -= 1
                self._last_activity = time.time()

    def _handle_request_inner(self, request: dict) -> dict:
        op = request.get("op")

        if op == "ping":
            return {"ok": True}

        # Issue #646 (M-44): server-side deadline. The client ships an
        # absolute monotonic-equivalent deadline as wall-clock epoch seconds
        # ("deadline"). If it has already passed, fail cheap BEFORE acquiring
        # the global lock or running a full encode — the client has already
        # abandoned the request and fallen back to FTS-only.
        deadline = request.get("deadline")
        if deadline is not None:
            try:
                if time.time() >= float(deadline):
                    return {"ok": False, "error": "deadline exceeded before encode"}
            except (TypeError, ValueError):
                pass

        if op == "embed":
            texts = request["texts"]
            tier = request.get("tier", "")

            # Single-text fast lane (issue #577): hook recall queries must
            # never queue behind batch ingestion work or OOM recovery.
            if len(texts) <= self._FAST_LANE_MAX_TEXTS:
                fast = self._handle_fast_embed(texts, tier)
                if fast is not None:
                    return fast

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
                with self._lock:
                    self._activate_throttler()

            # Issue #646 (M-43): capture the throttler to a local under the
            # lock. Reading self._throttler_active then self._throttler
            # separately raced _deactivate_throttler nulling the attribute
            # between the two reads (AttributeError on before_batch()).
            with self._lock:
                throttler = self._throttler if self._throttler_active else None
            if throttler is not None:
                throttler.before_batch()

            encode_start = time.time()
            # `vectors` is assigned on exactly one of two paths: inside the
            # locked try (retry_on_cpu stays False), or by the post-lock
            # retry (retry_on_cpu True). Any other outcome raises.
            vectors = None
            retry_on_cpu = False
            with self._lock:
                model = self._get_embed_model(tier)
                try:
                    vectors = model.encode(texts, show_progress_bar=False)
                except RuntimeError as exc:
                    from truememory.mps_utils import is_mps_oom
                    if not is_mps_oom(exc):
                        raise
                    # Issue #577: sticky-CPU degradation via the single
                    # shared recovery path, atomic in this lock hold. The
                    # model is never re-promoted to MPS (the old re-promotion
                    # left the MPS pool above its watermark cap and
                    # guaranteed the next OOM — retry storms). Only the
                    # expensive full re-encode runs OUTSIDE the lock, so
                    # other clients are not starved for 23-112s.
                    self._recover_embed_oom_locked(model)
                    retry_on_cpu = True
            if retry_on_cpu:
                log.warning(
                    "MPS OOM during encoding — retrying on CPU outside the "
                    "request lock"
                )
                vectors = model.encode(texts, show_progress_bar=False)
            encode_time = time.time() - encode_start

            # M-43: re-capture under the lock for the same reason as above.
            with self._lock:
                throttler = self._throttler if self._throttler_active else None
            if throttler is not None:
                throttler.after_batch(len(texts), encode_time)
                if throttler.should_flush_cache():
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
                    # Mark + reload happen ATOMICALLY under this same lock
                    # hold so the retry below always runs on the locally
                    # reloaded CPU instance (panel round 1, item 1).
                    self._mark_sticky_cpu("rerank")
                    flush_mps_cache()
                    self._reranker = None
                    self._reranker_name = None
                    reranker = self._get_reranker(model_name)  # sticky → CPU
                    oom = True
            if oom:
                # Retry outside the lock — rerank batches are the expensive
                # part; other clients must not starve behind the recovery.
                scores = reranker.predict(
                    pairs, batch_size=64, show_progress_bar=False
                )
            return {"ok": True, "scores": np.asarray(scores, dtype=np.float32)}

        return {"ok": False, "error": f"Unknown op: {op}"}

    def _activate_throttler(self):
        """Start adaptive throttling for sustained workload.

        Caller must hold ``self._lock`` (issue #646, M-43): activate and
        deactivate mutate ``_throttler`` / ``_throttler_active`` together and
        must not interleave with the lock-free readers in ``handle_request``.
        """
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
        """Stop adaptive throttling — workload ended.

        Caller must hold ``self._lock`` (issue #646, M-43).
        """
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
        # Issue #646 (M-53): tag every response with the protocol version so
        # a newer client can detect a version mismatch instead of choking on
        # an unexpected payload shape.
        if "protocol" not in response:
            response = {**response, "protocol": PROTOCOL_VERSION}
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
                inflight = self._inflight
            elapsed = time.time() - last
            # Issue #646 (M-74): never idle-shut-down while a request is
            # mid-flight, even if its start timestamp is older than the idle
            # horizon (a long batch encode can outlast IDLE_TIMEOUT).
            if elapsed >= IDLE_TIMEOUT and inflight == 0:
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

    @staticmethod
    def _restrict_acl_windows(path: Path) -> None:
        """Best-effort: restrict *path*'s ACL to the current user (Windows).

        ``os.open(..., 0o600)`` does not restrict Windows ACLs, so a TCP
        fallback token written with mode 0o600 can still be world-readable.
        Use ``icacls`` to remove inherited ACEs and grant only the current
        user full control. On any failure, warn (mirroring the config.json
        permission warning) rather than crash — the loopback bind + HMAC
        still gate access, this only hardens the at-rest token file.
        """
        user = os.environ.get("USERNAME")
        if not user:
            return
        try:
            import subprocess
            subprocess.run(
                ["icacls", str(path), "/inheritance:r",
                 "/grant:r", f"{user}:F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except Exception:
            print(
                "truememory: warning — could not restrict ACL on "
                f"{path}; the model-server HMAC token may be readable by "
                "other local users on this machine. On a shared machine, "
                "ensure ~/.truememory is not world-readable.",
                file=sys.stderr,
            )

    def _acquire_bind_lock(self):
        """Take an exclusive lock BEFORE binding (issue #646, M-20).

        The previous design wrote PID_PATH then bound, leaving a multi-second
        TOCTOU window (PID written before imports finish) during which a
        second concurrent starter would also bind and the two servers would
        cycle each other's artifacts indefinitely. Holding an OS-level
        exclusive lock for the process lifetime makes "only one server binds"
        atomic. On POSIX we use ``flock``; on Windows ``msvcrt.locking``. The
        fd is kept open (and stored on ``self``) until the process exits.

        Returns the lock fd, or raises ``RuntimeError`` if another live
        server holds the lock.
        """
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise RuntimeError("another model server holds the bind lock")
        return fd

    def run(self):
        # M-89: keep ~/.truememory owner-only (0700) — it holds memories/PII.
        _TRUEMEMORY_DIR.mkdir(parents=True, exist_ok=True)
        try:
            _TRUEMEMORY_DIR.chmod(0o700)
        except OSError:
            pass

        # Exclusive bind lock BEFORE touching socket/pid artifacts (M-20).
        self._lock_fd = self._acquire_bind_lock()

        if _USE_UNIX:
            # We hold the exclusive lock, so any socket file here is stale
            # (a crashed predecessor); safe to remove now that no live peer
            # can own it.
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
            # M-80: the 0o600 mode above is a no-op on Windows — os.open only
            # honors the read-only bit, so the HMAC token can be world-readable
            # to other local users. Restrict the ACL to the current user via
            # icacls; warn (mirroring the config.json perm warning) if that
            # fails so a shared-machine operator is not silently exposed.
            if sys.platform == "win32":
                self._restrict_acl_windows(TOKEN_PATH)
            transport_desc = f"tcp={_LOOPBACK_HOST}:{self._bound_port}"
        # PID written only AFTER a successful bind — never advertises a
        # not-yet-listening server (M-20). We own the artifacts from here.
        PID_PATH.write_text(str(os.getpid()))
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
        # Only remove artifacts THIS process owns (issue #646, M-20). After a
        # crash a fresh server may already hold the lock and have rewritten
        # PID_PATH; unlinking its live socket/token here would let concurrent
        # hooks cycle servers indefinitely.
        if _safe_to_cleanup_artifacts():
            for p in (SOCK_PATH, PID_PATH, PORT_PATH, TOKEN_PATH):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        # Release the bind lock (lock fd is process-owned, always safe).
        lock_fd = getattr(self, "_lock_fd", None)
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass
            self._lock_fd = None
        # Reset the embed snapshot and fast-lane identity so no stale model
        # identity survives a stop (issue #577, panel round 2).
        self._embed_state = None
        self._reranker = None
        self._reranker_name = None
        self._fast_encoder = None
        self._fast_model_id = None
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

    # Fast pre-check: a live PID means a server is (probably) already up.
    # This is advisory only — the authoritative guard is the exclusive bind
    # lock taken inside run() (issue #646, M-20), which closes the TOCTOU
    # window this check alone left open. Stale artifacts from a crashed
    # predecessor are reclaimed under the lock in run(), not here.
    if PID_PATH.exists():
        try:
            old_pid = int(PID_PATH.read_text().strip())
            if pid_is_alive(old_pid):
                log.error("Model server already running (pid=%d)", old_pid)
                sys.exit(1)
        except (ValueError, OSError):
            pass

    server = ModelServer()

    # Ensure cleanup runs even on unhandled exit.
    atexit.register(server._cleanup)

    try:
        server.run()
    except RuntimeError as e:
        # Lost the bind-lock race to a concurrent starter (M-20). Exit
        # cleanly without touching the winner's artifacts.
        log.error("Model server start aborted: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
