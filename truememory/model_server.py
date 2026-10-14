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

def _set_mps_memory_cap():
    """Keep thread defaults early; MPS byte calibration occurs at model load."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")


_set_mps_memory_cap()

import base64  # noqa: E402
import gc  # noqa: E402
import hmac  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import secrets  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import stat  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from _thread import LockType  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from concurrent.futures import Future, ThreadPoolExecutor  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING, Callable  # noqa: E402

import numpy as np  # noqa: E402

if TYPE_CHECKING:
    from truememory.embedding_target import EmbeddingTarget

try:
    import psutil  # noqa: E402
except ImportError:
    psutil = None

from truememory._platform import (  # noqa: E402
    _LOOPBACK_HOST, _MODEL_SERVER_GENERATION_ENV, _USE_UNIX, _env_int,
    pid_is_alive, read_start_claim, try_file_lock,
)

log = logging.getLogger(__name__)

_TRUEMEMORY_DIR = Path.home() / ".truememory"
SOCK_PATH = _TRUEMEMORY_DIR / "model.sock"
PID_PATH = _TRUEMEMORY_DIR / "model_server.pid"
PORT_PATH = _TRUEMEMORY_DIR / "model_server.port"
TOKEN_PATH = _TRUEMEMORY_DIR / "model_server.token"
IDLE_TIMEOUT = _env_int("TRUEMEMORY_MODEL_SERVER_IDLE", 300, lo=0)

LOCK_PATH = _TRUEMEMORY_DIR / "model_server.lock"
START_STATE_PATH = _TRUEMEMORY_DIR / "model_server.start.json"

# Issue #646 (M-53): protocol/version handshake. Bumped whenever the wire
# format changes incompatibly. The client echoes this back on mismatch.
PROTOCOL_VERSION = 1

_HEADER_FMT = ">I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# Embedding uses SentenceTransformer's default; reranking keeps the server's 64.
# These cap items per call, not tokens, queued payloads, or total process memory.
_EMBED_BATCH_LIMIT = 32
_RERANK_BATCH_LIMIT = 64


class _ResultTooLarge(ValueError):
    """The complete float32 result cannot fit the existing response frame."""


class _ServingIdentityMismatch(ValueError):
    """Explicit selected identity rejection, distinct from native availability."""

    serving_identity_mismatch = True


class _ProcessMemoryRefused(RuntimeError):
    """Current process memory cannot admit another model stage."""


def _array_metadata(shape: tuple, encoded: str = "") -> dict:
    return {"__ndarray__": encoded, "dtype": "float32", "shape": list(shape)}


def _array_base64_size(shape: tuple) -> int:
    raw_bytes = math.prod(shape) * 4
    return 4 * ((raw_bytes + 2) // 3)


def _result_wire_size(shape: tuple, field: str) -> int:
    envelope = {"ok": True, field: _array_metadata(shape), "protocol": PROTOCOL_VERSION}
    return len(json.dumps(envelope).encode("utf-8")) + _array_base64_size(shape)


def _check_result_size(shape: tuple, field: str) -> None:
    size = _result_wire_size(shape, field)
    if size > _MAX_MESSAGE_SIZE:
        raise _ResultTooLarge(
            f"Complete {field} response requires {size} bytes; the limit is "
            f"{_MAX_MESSAGE_SIZE} bytes. Split the request into fewer inputs."
        )


def _response_wire_size(response: dict) -> int:
    """Measure array JSON without making float32, bytes or base64 copies."""
    array_bytes = 0

    def metadata_only(obj: object) -> dict:
        nonlocal array_bytes
        if isinstance(obj, np.ndarray):
            array_bytes += _array_base64_size(obj.shape)
            return _array_metadata(obj.shape)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    envelope_bytes = len(json.dumps(response, default=metadata_only).encode("utf-8"))
    return envelope_bytes + array_bytes


def _admission_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not raw.isascii() or not raw.isdecimal() or not minimum <= int(raw) <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return int(raw)


def _physical_memory_bytes() -> int:
    if psutil is None:
        raise ValueError("Automatic model-server RSS budget requires a physical-memory sensor")
    try:
        physical = psutil.virtual_memory().total
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Automatic model-server physical-memory measurement is unavailable") from exc
    if isinstance(physical, bool) or not isinstance(physical, int) or not 0 < physical <= 2**63 - 1:
        raise ValueError("Physical memory must be a positive signed64 integer byte count")
    return physical


def _automatic_rss_budget_bytes(physical: int | None = None) -> int:
    """Finite sampled admission threshold, without a native peak guarantee."""
    if physical is None:
        physical = _physical_memory_bytes()
    # Base/Pro on CPU keep the runtime (0.6 GB), Qwen (1.2 GB, or 2.4 GB in
    # float32 on macOS) and the reranker (0.6 GB) resident. A quarter of an
    # 8 GB host cannot hold that, so small hosts may use up to half their
    # memory, never more than 4 GiB on macOS or 3 GiB elsewhere.
    budget = physical // 4
    if budget > 0:
        floor = (4 if sys.platform == "darwin" else 3) * 1024**3
        budget = min(8 * 1024**3, max(budget, min(physical // 2, floor)))
    if budget <= 0:
        raise ValueError("Automatic model-server RSS budget must be positive")
    return budget


def _batch_limit(value: object, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("batch_size must be a positive integer")
    return min(value, maximum)


def _checked_batch(
    values: object, count: int, total: int, field: str,
    *, admit: Callable[[tuple, int], None] | None = None,
    before_materialize: Callable[[], None] | None = None,
) -> np.ndarray:
    if admit is not None and isinstance(values, np.ndarray) and values.ndim:
        cast_bytes = 0 if values.dtype == np.dtype(np.float32) else math.prod(values.shape) * 4
        admit((total, *values.shape[1:]), cast_bytes)
    if not isinstance(values, np.ndarray) and before_materialize is not None:
        before_materialize()
    batch = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=np.float32)
    if batch.ndim == 0 or batch.shape[0] != count:
        raise ValueError("Model output count does not match the input microbatch")
    _check_result_size((total, *batch.shape[1:]), field)
    if admit is not None:
        admit((total, *batch.shape[1:]), 0)
    return np.asarray(batch, dtype=np.float32)


def _store_batch_result(
    result: np.ndarray | None, values: object, offset: int, count: int, total: int,
    field: str = "vectors", *, indices: np.ndarray | None = None,
    admit: Callable[[tuple, int], None] | None = None,
    before_materialize: Callable[[], None] | None = None,
) -> np.ndarray:
    """Check the complete output before allocating it; keep every float32 row."""
    if admit is None and before_materialize is None:
        batch = _checked_batch(values, count, total, field)
    else:
        batch = _checked_batch(values, count, total, field, admit=admit,
                               before_materialize=before_materialize)
    if result is None:
        result = np.empty((total, *batch.shape[1:]), dtype=np.float32)
    elif result.shape[1:] != batch.shape[1:]:
        raise ValueError("Model output dimensions changed between microbatches")
    if indices is None:
        result[offset:offset + count] = batch
    else:
        result[indices] = batch
    return result


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
        return _array_metadata(arr.shape, base64.b64encode(arr.tobytes()).decode("ascii"))
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


_ALLOWED_DTYPES = frozenset({"float32", "float64", "float16", "int32", "int64"})
_decoded_array_groups = itertools.count()


def _json_object_hook(obj):
    """Decode base64-encoded numpy arrays from JSON."""
    if "__ndarray__" in obj:
        dtype_str = obj["dtype"]
        if dtype_str not in _ALLOWED_DTYPES:
            raise ValueError(f"Disallowed dtype: {dtype_str}")
        from truememory.memory_budget import ProfileKey, current_session
        memory = current_session()
        encoded = obj["__ndarray__"]
        if memory is not None and type(encoded) is str:
            memory.charge(ProfileKey("request ndarray decode", "model_server",
                                     native_tensor_shape=(len(encoded),)),
                          known_bytes=len(encoded) + 3 * ((len(encoded) + 3) // 4),
                          group=f"request decoded array:{next(_decoded_array_groups)}")
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
    generation: object = None
    prepared_identity: tuple[str, int, str] | None = None

    def __post_init__(self) -> None:
        if self.generation is None:
            object.__setattr__(self, "generation", object())


class _RequestDeadlineExceeded(TimeoutError):
    """Stop this request without triggering model fallback or another retry."""


@dataclass
class _TransportRequest:
    length: int
    queue_expires_at: float
    stopped: threading.Event
    frame_expires_at: float = math.inf
    started: bool = False
    claimed: bool = False
    withdrawn: bool = False


@dataclass(frozen=True)
class _RequestDeadline:
    expires_at: float | None = None
    transport: _TransportRequest | None = None

    @classmethod
    def from_wall_clock(cls, value: object) -> "_RequestDeadline":
        if value is None:
            return cls()
        try:
            wall_deadline = float(value)
        except (TypeError, ValueError, OverflowError):
            return cls()  # Preserve legacy handling of malformed deadlines.
        if math.isnan(wall_deadline) or wall_deadline == math.inf:
            return cls()
        remaining = wall_deadline - time.time()
        return cls(time.monotonic() + remaining)

    def remaining(self) -> float | None:
        expires_at = self.expires_at
        if self.transport is not None:
            if self.transport.stopped.is_set():
                raise _RequestDeadlineExceeded("model server is shutting down")
            if expires_at is None and not self.transport.started:
                expires_at = self.transport.queue_expires_at
        if expires_at is None:
            return None
        remaining = expires_at - time.monotonic()
        if remaining <= 0:
            raise _RequestDeadlineExceeded("deadline exceeded before encode")
        return remaining

    def check(self) -> None:
        self.remaining()

    def start_inference(self) -> None:
        self.check()
        if self.transport is not None:
            self.transport.started = True

    @contextmanager
    def locked(self, lock: LockType) -> Iterator[None]:
        remaining = self.remaining()
        if self.transport is not None:
            while not lock.acquire(timeout=min(0.1, remaining) if remaining is not None else 0.1):
                remaining = self.remaining()
        elif remaining is None:
            lock.acquire()
        elif not lock.acquire(timeout=min(remaining, threading.TIMEOUT_MAX)):
            raise _RequestDeadlineExceeded("deadline exceeded before encode")
        try:
            # A successful wakeup can still occur after the request expired.
            self.check()
            yield
        finally:
            lock.release()


class ModelServer:
    """Serves embedding and reranking over a Unix domain socket (POSIX)
    or HMAC-authenticated TCP loopback (Windows)."""

    _SUSTAINED_THRESHOLD = 10
    _SUSTAINED_WINDOW = 30
    # Requests with at most this many texts take the single-text fast lane
    # (issue #577): hook recall queries must never queue behind ingestion
    # batches or OOM recovery.
    _FAST_LANE_MAX_TEXTS = 1

    def __init__(self, *, automatic_rss_budget: bool = False) -> None:
        if not isinstance(automatic_rss_budget, bool):
            raise ValueError("automatic_rss_budget must be a bool")
        # Installed main opts in; explicit settings and direct construction retain their contract.
        # The installed server skips the fast-lane CPU copy on hosts with 8 GiB or
        # less: beside the MPS cap it does not fit.
        self._small_host = False
        if automatic_rss_budget and os.environ.get("TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB") is None:
            physical = _physical_memory_bytes()
            self._max_rss_bytes = _automatic_rss_budget_bytes(physical)
            self._small_host = physical <= 8 * 1024**3
        else:
            self._max_rss_bytes = _admission_setting(
                "TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB", 0, 0, 2**31 - 1,
            ) * 1048576
        self._max_handlers = _admission_setting("TRUEMEMORY_MODEL_SERVER_MAX_HANDLERS", 16, 1, 128)
        self._max_request_bytes = _admission_setting(
            "TRUEMEMORY_MODEL_SERVER_MAX_REQUEST_BYTES", 32 * 1024**2, _MAX_MESSAGE_SIZE, 1024**3,
        )
        self._header_timeout = _admission_setting(
            "TRUEMEMORY_MODEL_SERVER_HEADER_TIMEOUT_MS", 1000, 100, 30000,
        ) / 1000
        self._frame_timeout = _admission_setting(
            "TRUEMEMORY_MODEL_SERVER_FRAME_TIMEOUT_MS", 30000, 100, 120000,
        ) / 1000
        self._queue_timeout = 120.0
        self._admission_lock = threading.Lock()
        self._staging_lock = threading.Lock()
        self._staging: socket.socket | None = None
        self._clients: dict[socket.socket, _TransportRequest] = {}
        self._reserved_request_bytes = 0
        self._admission_high_water = {"handlers": 0, "request_bytes": 0}
        self._busy_rejections = 0
        self._stopped = threading.Event()
        self._transport_context = threading.local()
        self._memory_owner = None
        if automatic_rss_budget:
            from truememory.memory_budget import get_process_registry
            self._memory_owner = get_process_registry().register_owner(
                self._max_rss_bytes, kind="model_server",
                usage_reader=lambda: psutil.Process().memory_info().rss if psutil is not None else None,
            )
        # Submission requires a slot+bytes lease; at most max_handlers futures
        # can be running or waiting in this executor.
        try:
            self._workers = ThreadPoolExecutor(max_workers=self._max_handlers, thread_name_prefix="model-client")
        except BaseException:
            if self._memory_owner is not None:
                self._memory_owner.close()
            raise
        # Loaded embedding model + identity as ONE immutable snapshot,
        # assigned only under self._lock; the fast lane reads the single
        # reference lock-free (issue #577, panel round 2).
        self._embed_state: _EmbedState | None = None
        self._reranker = None
        self._reranker_name: str | None = None
        self._lock = threading.Lock()
        # Only snapshot/cache publication, never model work or destruction.
        # Main state -> residency; fast -> residency; never the reverse.
        self._residency_lock = threading.Lock()
        # Main models stay exclusively owned through loading, recovery and
        # CPU retries, even while the state lock is released (issue #738).
        # Lock order: inference -> state; fast inference owns only _fast_lock.
        self._inference_lock = threading.Lock()
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
        # A generation distinguishes A -> B -> A from an unchanged residency.
        self._fast_encoder = None
        self._fast_model_id: str | None = None
        self._fast_generation: object = None
        self._fast_lock = threading.Lock()
        # CPU constructions are admitted and built one at a time, so two
        # builds cannot both pass a size check sized for one. Innermost lock.
        self._construction_lock = threading.Lock()
        # Issue #646 (M-20): exclusive bind-lock fd, held for the process
        # lifetime; released in _cleanup. None until run() acquires it.
        self._lock_fd: int | None = None
        # Issue #646 (M-74): in-flight request counter so idle shutdown never
        # kills a request mid-encode. Guarded by _activity_lock.
        self._inflight = 0

    def _check_process_memory(
        self, stage: str, deadline: _RequestDeadline | None = None,
    ) -> None:
        """Sample current RSS under the caller's model owner, never peak RSS."""
        if deadline is not None:
            deadline.check()
        budget = self._max_rss_bytes
        if budget == 0:
            if getattr(self, "_memory_owner", None) is not None:
                self._memory_stage(stage, deadline)
            return
        try:
            usage = psutil.Process().memory_info().rss if psutil is not None else None
        except Exception:
            usage = None
        # Sensor work consumes the same request budget. Expiry takes precedence
        # over a memory refusal and cannot admit another native stage.
        if deadline is not None:
            deadline.check()
        if isinstance(usage, bool) or not isinstance(usage, int) or usage <= 0:
            raise _ProcessMemoryRefused(
                f"Model server busy: current process RSS unavailable before {stage}"
            )
        if usage >= budget:
            raise _ProcessMemoryRefused(
                f"Model server busy: process RSS budget exhausted before {stage}"
            )
        if getattr(self, "_memory_owner", None) is not None and not stage.endswith(" completion"):
            self._memory_stage(stage, deadline)

    def _memory_scope(self):
        from contextlib import nullcontext
        owner = getattr(self, "_memory_owner", None)
        if owner is None:
            return nullcontext(None)
        from truememory.memory_budget import InvalidAllocationPlan, StaleBudgetOwner, allocation_scope
        try:
            owner.set_limit(self._max_rss_bytes)
        except (InvalidAllocationPlan, StaleBudgetOwner, ValueError) as error:
            raise _ProcessMemoryRefused(f"Model server busy: {error}") from error
        return allocation_scope(owner)

    def _memory_stage(
        self, stage: str, deadline: _RequestDeadline | None = None, *,
        known_bytes: int = 0, shape: tuple[int, ...] | None = None,
        wire_signature: str | None = None, model_identity: tuple[str, int, str] | None = None,
        device: str | None = None, input_count: int | None = None,
    ) -> None:
        owner = getattr(self, "_memory_owner", None)
        if owner is None:
            return
        from truememory.memory_budget import (
            InvalidAllocationPlan, MemoryAdmissionRefused, ProfileKey, StaleBudgetOwner, current_session,
        )
        if current_session() is None:
            # A standalone sampled guard performs no allocation itself.
            return
        deadline = deadline or getattr(self._transport_context, "memory_deadline", None)
        try:
            owner.set_limit(self._max_rss_bytes)
            key = ProfileKey(stage, "model_server", native_tensor_shape=shape,
                             wire_signature=wire_signature, model_identity=model_identity,
                             device=device, input_item_count=input_count)
            with self._memory_scope() as memory:
                memory.charge(key, known_bytes=known_bytes,
                              deadline=deadline.check if deadline is not None else None)
        except (MemoryAdmissionRefused, InvalidAllocationPlan, StaleBudgetOwner, ValueError) as error:
            raise _ProcessMemoryRefused(f"Model server busy: {error}") from error

    def _reserve_result_assembly(self, shape: tuple, cast_bytes: int = 0) -> None:
        self._memory_stage("result assembly", known_bytes=math.prod(shape) * 4, shape=shape)
        if cast_bytes:
            self._memory_stage("result cast", known_bytes=cast_bytes, shape=(cast_bytes,))

    def _result_admitter(self) -> Callable[[tuple, int], None] | None:
        return self._reserve_result_assembly if getattr(self, "_memory_owner", None) is not None else None

    def _before_result_materialization(self) -> None:
        self._memory_stage("result materialize")

    def _result_materializer(self) -> Callable[[], None] | None:
        return self._before_result_materialization if getattr(self, "_memory_owner", None) is not None else None

    def _result_admission_options(self, *, materialize: bool = True) -> dict[str, object]:
        if getattr(self, "_memory_owner", None) is None:
            return {}
        options = {"admit": self._result_admitter()}
        if materialize:
            options["before_materialize"] = self._result_materializer()
        return options

    def _retain_memory_response(self, memory: object, response: dict) -> None:
        if memory is None:
            return
        from truememory.memory_budget import InvalidAllocationPlan, MemoryAdmissionRefused, StaleBudgetOwner
        try:
            for value in response.values():
                if isinstance(value, np.ndarray):
                    memory.retain_buffer(value)
        except (MemoryAdmissionRefused, InvalidAllocationPlan, StaleBudgetOwner, ValueError) as error:
            raise _ProcessMemoryRefused(f"Model server busy: {error}") from error

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

    def _recover_embed_oom_locked(
        self, model: object, deadline: _RequestDeadline | None = None,
    ) -> None:
        """The single recovery path for an embed MPS OOM (issue #577).

        Caller MUST hold ``self._lock``. Marks the embed path sticky-CPU
        (loud log once), flushes the MPS cache, and moves the model to CPU —
        all atomically in the caller's lock hold. The caller retains main
        inference ownership throughout recovery and retry. Batch retries
        release the state lock so bookkeeping and the fast lane stay responsive.
        """
        @contextmanager
        def memory_scope() -> Iterator[object | None]:
            if getattr(self, "_memory_owner", None) is None:
                yield None
            else:
                with self._memory_scope() as memory:
                    yield memory

        with memory_scope() as _memory:
            from truememory.mps_utils import flush_mps_cache
            self._mark_sticky_cpu("embed")
            flush_mps_cache()
            if deadline is not None:
                self._check_embed_recovery_deadline_locked(model, deadline)
            try:
                self._check_process_memory("embed CPU transfer", deadline)
                state = self._embed_state
                model_id = state.model_id if state is not None and state.model is model else None
                self._memory_stage("embed CPU transfer", deadline,
                                   known_bytes=self._MODEL_WEIGHT_BYTES.get(model_id, 0))
            except (_ProcessMemoryRefused, _RequestDeadlineExceeded):
                # A refused transfer leaves the failed accelerator instance behind.
                # Invalidate it so a cache hit cannot bypass sticky CPU selection.
                state = self._embed_state
                if state is not None and state.model is model:
                    self._publish_embed_state(None)
                raise
            if deadline is not None:
                self._check_embed_recovery_deadline_locked(model, deadline)
            if hasattr(model, "to"):
                try:
                    model.to("cpu")
                except BaseException:
                    # Failed or cancelled transfers cannot certify this instance
                    # as CPU-ready. Preserve the original error and sticky choice.
                    state = self._embed_state
                    if state is not None and state.model is model:
                        self._publish_embed_state(None)
                    raise
                from truememory.mps_utils import prefer_float32_on_cpu
                prefer_float32_on_cpu(model)
                self._release_stage("embed CPU transfer")
                # The moved weights' accelerator storage is free only now.
                flush_mps_cache()

    def _check_embed_recovery_deadline_locked(
        self, model: object, deadline: _RequestDeadline,
    ) -> None:
        """Record the OOM even if its request cannot recover; caller holds the lock."""
        self._mark_sticky_cpu("embed")
        try:
            deadline.check()
        except _RequestDeadlineExceeded:
            # Cached models bypass device resolution. Drop this failed instance
            # so the next request loads on CPU without moving it for dead work.
            state = self._embed_state
            if state is not None and state.model is model:
                self._publish_embed_state(None)
            raise

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
        """Resolve this request without changing the daemon's default model."""
        if tier and tier.strip().lower() == "qwen3":
            # Retain the removed-model error formerly raised by the setter,
            # without adopting its broader normalization or changing globals.
            from truememory.vector_search import _resolve_model_name
            _resolve_model_name(tier)
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
            from truememory.mps_utils import ensure_mps_memory_budget
            ensure_mps_memory_budget(device)
            from sentence_transformers import SentenceTransformer
            mkwargs = {}
            if sys.platform == "darwin":
                mkwargs["attn_implementation"] = "eager"
            return SentenceTransformer(
                "Qwen/Qwen3-Embedding-0.6B",
                truncate_dim=256,
                config_kwargs={"use_cache": False},
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
            from truememory.tier_config import resolve_custom_tier
            cfg = resolve_custom_tier()
            custom_dim = cfg["embed_dim"]
            from truememory.mps_utils import ensure_mps_memory_budget
            ensure_mps_memory_budget(device)
            from sentence_transformers import SentenceTransformer
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

    # Resident bytes charged before a known model is built, so admission refuses
    # a construction that cannot fit instead of refusing it after weights load.
    # Qwen3-Embedding-0.6B is bfloat16 on disk but float32 under transformers 4.
    _MODEL_WEIGHT_BYTES = {
        "qwen3_256": 2_400_000_000,
        "model2vec": 32_000_000,
        "Alibaba-NLP/gte-reranker-modernbert-base": 600_000_000,
        "cross-encoder/ms-marco-MiniLM-L-6-v2": 92_000_000,
    }

    def _resolve_embed_cache(self, tier: str) -> tuple[str, _EmbedState | None]:
        """Return the requested identity and its compatible cached snapshot."""
        state = self._embed_state
        if state is not None and state.tier == tier:
            # A queued empty-tier request must follow the load it waited for,
            # even if the default changed mid-load. Custom keys likewise keep
            # the identity actually built rather than re-reading their config.
            return state.model_id, state

        model_id = self._resolve_embed_model_id(tier)
        if state is not None and state.model_id == model_id and (
            model_id in self._FAST_LANE_SAFE_MODEL_IDS or state.tier == tier
        ):
            return model_id, state
        return model_id, None

    def _preflight_embed_result(self, tier: str, count: int) -> None:
        """Caller holds the state lock; mirror the actual server constructors."""
        if not count:
            return
        model_id, state = self._resolve_embed_cache(tier)
        cached = state is not None
        known = model_id in ("model2vec", "qwen3_256", "minilm", "bge-small")
        fallback = not cached and os.environ.get("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "").strip() != "1"
        if known or fallback:
            # Legacy minilm/bge-small fall through to model2vec here, not
            # the 384-dimensional models used by the local loader.
            _check_result_size((count, 256), "vectors")
            if getattr(self, "_memory_owner", None) is not None:
                self._reserve_result_assembly((count, 256))
        # A custom truncate_dim is only an upper bound. Even ST's dimension
        # getter can return that bound when native width is unknown; check
        # the actual first slice before allocating the complete result.

    def _embed_global_order(
        self, model: object, texts: list, limit: int, deadline: _RequestDeadline,
    ) -> np.ndarray | None:
        """Match modern ST's native32 order only for the known Qwen model."""
        deadline.check()
        state = self._embed_state
        if (limit != 32 or len(texts) <= limit or state is None or state.model is not model
                or state.model_id != "qwen3_256" or not all(isinstance(text, str) for text in texts)):
            return None
        length = getattr(model, "_input_length", None)
        flatten = getattr(model, "_can_flatten_inputs", None)
        if not callable(length) or not callable(flatten) or flatten():
            return None
        deadline.check()
        lengths = [-length(text) for text in texts]
        deadline.check()
        order = np.argsort(lengths)
        deadline.check()
        return order

    @staticmethod
    def _embed_slice_indices(
        model: object, texts: list, order: np.ndarray, offset: int, limit: int,
        deadline: _RequestDeadline,
    ) -> np.ndarray:
        deadline.check()
        indices = order[offset:offset + limit]
        # Each encode sorts again. Invert its permutation within equal-length
        # groups so its native rows keep the exact global order, including ties.
        lengths = [-model._input_length(texts[index]) for index in indices]
        deadline.check()
        inner = np.argsort(lengths)
        deadline.check()
        inverse = np.argsort(inner)
        deadline.check()
        return indices[inverse]

    def _rerank_global_order(
        self, model: object, pairs: list, limit: int, deadline: _RequestDeadline,
    ) -> np.ndarray | None:
        """Match modern CrossEncoder's native 64-row policy for ModernBERT."""
        deadline.check()
        if (limit != 64 or len(pairs) <= limit or self._reranker is not model
                or self._reranker_name != "Alibaba-NLP/gte-reranker-modernbert-base"
                or not all(isinstance(pair, (list, tuple)) and len(pair) == 2
                           and all(isinstance(text, str) for text in pair) for pair in pairs)):
            return None
        length = getattr(model, "_input_length", None)
        flatten = getattr(model, "_can_flatten_inputs", None)
        if not callable(length) or not callable(flatten) or flatten():
            return None
        deadline.check()
        lengths = [-length(pair) for pair in pairs]
        deadline.check()
        order = np.argsort(lengths)
        deadline.check()
        return order

    @staticmethod
    def _rerank_slice_indices(
        model: object, pairs: list, order: np.ndarray, offset: int, limit: int,
        deadline: _RequestDeadline,
    ) -> np.ndarray:
        deadline.check()
        indices = order[offset:offset + limit]
        # Predict sorts each supplied batch again, including equal-length pairs.
        lengths = [-model._input_length(pairs[index]) for index in indices]
        deadline.check()
        inner = np.argsort(lengths)
        deadline.check()
        inverse = np.argsort(inner)
        deadline.check()
        return indices[inverse]

    @staticmethod
    def _preflight_rerank_result(
        model_name: str | None, count: int, *, admit: Callable[[tuple, int], None] | None = None,
    ) -> None:
        from truememory.reranker import get_current_reranker_name
        name = model_name or get_current_reranker_name()
        if count and name in (
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "Alibaba-NLP/gte-reranker-modernbert-base",
        ):
            _check_result_size((count,), "scores")
            if admit is not None:
                admit((count,), 0)
        # Custom CrossEncoder constructors preserve num_labels=None; their
        # output may contain several labels per pair. Inspect its real shape.

    def _retire_stale_fast_encoder(self) -> None:
        """Drop an idle stale clone; an active owner retries this on release."""
        if not self._fast_lock.acquire(blocking=False):
            return
        release_fast = True
        try:
            with self._residency_lock:
                state = self._embed_state
                if self._fast_encoder is None or (
                    state is not None and self._fast_generation is state.generation
                ):
                    # Unlock before publication can change the generation, so
                    # a following invalidation cannot miss an idle stale cache.
                    self._fast_lock.release()
                    release_fast = False
                    return
                retired = self._fast_encoder
                self._fast_encoder = None
                self._fast_model_id = None
                self._fast_generation = None
            # Destruction stays outside residency, but before another fast load.
            state = None
            del retired
        finally:
            if release_fast:
                self._fast_lock.release()

    def _publish_embed_state(self, state: _EmbedState | None) -> None:
        """Caller owns the main state lock; retire without waiting for fast work."""
        with self._residency_lock:
            retired = self._embed_state
            self._embed_state = state
        del retired
        self._retire_stale_fast_encoder()

    @staticmethod
    def _discard_models() -> None:
        """Free dropped models (reference cycles) and their accelerator cache."""
        from truememory.mps_utils import flush_mps_cache
        flush_mps_cache()

    def _construct_model(
        self, kind: str, stage: str, device: str | None, deadline: _RequestDeadline | None,
        build: Callable[[str | None], object], *, model_identity: tuple[str, int, str] | None = None,
        known_bytes: int = 0,
    ) -> tuple[object, str | None]:
        """Build within the memory budget on *device*, else on CPU.

        An MPS OOM while building degrades *kind* to CPU for the server
        lifetime, as an inference OOM does (issue #577), so a model that
        cannot fit is not rebuilt and refused on every request.
        """
        # Dropped models are reference cycles; free them before admitting this one.
        self._discard_models()
        selected = device
        while True:
            resident = selected
            if resident is None:
                from truememory.mps_utils import auto_detect_device
                resident = auto_detect_device()
            # Accelerator weights are not in sampled RSS; only CPU builds are charged.
            charged = known_bytes if resident == "cpu" else 0
            with self._construction_slot(charged, deadline):
                # Admit with the full size, then return the reservation: the
                # weights reach sampled RSS as they load.
                self._memory_stage(stage, deadline, known_bytes=charged,
                                   model_identity=model_identity, device=selected)
                self._release_stage(stage)
                failed = False
                try:
                    model = build(selected)
                except RuntimeError as exc:
                    from truememory.mps_utils import is_mps_oom
                    if selected == "cpu" or not is_mps_oom(exc):
                        raise
                    failed = True
                if not failed:
                    from truememory.mps_utils import prefer_float32_on_cpu
                    prefer_float32_on_cpu(model)
                    try:
                        self._check_process_memory(stage + " completion", deadline)
                    except BaseException:
                        model = None
                        self._discard_models()
                        raise
                    self._release_stage(stage)
                    return model, selected
            # Outside the handler, so the failed build's frames are released.
            self._mark_sticky_cpu(kind)
            self._discard_models()
            selected = "cpu"
            self._check_process_memory(stage, deadline)

    def _construction_slot(self, charged: int, deadline: _RequestDeadline | None):
        if not charged:
            from contextlib import nullcontext
            return nullcontext()
        return self._construction_lock if deadline is None else deadline.locked(self._construction_lock)

    def _release_stage(self, stage: str) -> None:
        """Return a finished stage's reservation; its bytes are now in sampled RSS."""
        owner = getattr(self, "_memory_owner", None)
        if owner is None:
            return
        from truememory.memory_budget import current_session
        session = current_session()
        if session is not None and session.owner is owner:
            session.release(stage)

    def _get_embed_model(self, tier: str, deadline: _RequestDeadline | None = None):
        with self._memory_scope() as _memory:
            model_id, state = self._resolve_embed_cache(tier)
            if state is not None:
                if state.tier != tier:
                    self._publish_embed_state(_EmbedState(
                        model=state.model, tier=tier, model_id=model_id, generation=state.generation,
                        prepared_identity=state.prepared_identity,
                    ))
                return state.model

            # Memory-first replacement: failed construction leaves the cache empty.
            self._publish_embed_state(None)
            self._check_process_memory("embed construction", deadline)
            device = self._embed_device()
            if deadline is not None:
                deadline.check()
            model, _ = self._construct_model(
                "embed", "embed construction", device, deadline,
                lambda selected: self._build_embed_model(model_id, selected),
                model_identity=((model_id, 256, "legacy")
                                if model_id in self._FAST_LANE_SAFE_MODEL_IDS else None),
                known_bytes=self._MODEL_WEIGHT_BYTES.get(model_id, 0),
            )
            # ONE atomic reference assignment of an immutable snapshot — the
            # fast lane can never observe a torn (model, tier, model_id) triple.
            self._publish_embed_state(_EmbedState(model=model, tier=tier, model_id=model_id))
            log.info("Loaded embedding model for tier=%s (model=%s)", tier, model_id)
            return model

    def _get_prepared_embed_model(self, target: "EmbeddingTarget", deadline: _RequestDeadline) -> object:
        with self._memory_scope() as _memory:
            from truememory.embedding_target import build_target_model

            target.check_configuration()
            deadline.check()
            state = self._embed_state
            if state is not None and (
                state.prepared_identity == target.identity
                or (state.model_id == target.model_id and target.model_id in self._FAST_LANE_SAFE_MODEL_IDS)
            ):
                return state.model
            # A legacy custom label is not evidence that its constructor did not
            # fall back. Only strict construction can establish that identity.
            state = None
            self._publish_embed_state(None)
            self._check_process_memory("prepared embed construction", deadline)
            device = self._embed_device()
            deadline.check()
            model, _ = self._construct_model(
                "embed", "prepared embed construction", device, deadline,
                lambda selected: build_target_model(target, selected, check=deadline.check),
                model_identity=target.identity, known_bytes=self._MODEL_WEIGHT_BYTES.get(target.model_id, 0),
            )
            self._publish_embed_state(_EmbedState(
                model=model, tier=target.tier, model_id=target.model_id,
                prepared_identity=target.identity,
            ))
            return model

    def _get_fast_encoder(
        self, tier: str, deadline: _RequestDeadline | None = None, *, expected_target: "EmbeddingTarget | None" = None,
    ):
        """CPU-resident encoder for the single-text fast lane (issue #577).

        Loaded lazily on the first single-text request that finds the global
        lock busy, then cached for that main residency. Caller must hold
        ``self._fast_lock`` through inference and retire stale cache on release.

        Vector-space parity (panel rounds 1-2): the fast encoder is ONLY
        ever built from the main path's actual loaded identity, taken from
        one atomic read of the ``_embed_state`` snapshot. The fast lane
        never resolves tiers itself — no global reads, no config reads, no
        drift. Returns ``None`` (decline → caller falls through to the main
        locked path) when there is no snapshot for this tier yet, or the
        loaded model has no explicit deterministic builder branch.
        """
        with self._memory_scope() as _memory:
            state = self._embed_state  # single atomic reference read
            if state is None or state.tier != tier:
                # No main model loaded for this tier yet — declining means the
                # one-time load happens exactly once, on the main path,
                # without changing the daemon's default selection.
                return None

            if state.model_id not in self._FAST_LANE_SAFE_MODEL_IDS:
                return None

            model_id, generation = state.model_id, state.generation
            if getattr(self, "_small_host", False):
                # A second CPU copy (2.4 GB in float32 on macOS) does not fit beside
                # the MPS cap on an 8 GiB host; wait for the main model instead.
                raise _ProcessMemoryRefused("Model server busy: no room for a fast-lane copy on this host")
            if expected_target is not None and model_id != expected_target.model_id:
                from truememory.embedding_target import EmbeddingTargetError
                raise EmbeddingTargetError("Fast encoder residency does not match the serving target")
            # A slow CPU constructor must not retain the old main model snapshot.
            state = None
            if self._fast_encoder is not None and self._fast_generation is generation:
                if expected_target is not None and self._fast_model_id != model_id:
                    from truememory.embedding_target import EmbeddingTargetError
                    raise EmbeddingTargetError("Fast encoder cache does not match the serving target")
                return self._fast_encoder

            self._fast_encoder = None
            self._fast_model_id = None
            self._fast_generation = None
            self._check_process_memory("fast embed construction", deadline)
            model, _ = self._construct_model(
                "embed", "fast embed construction", "cpu", deadline,
                lambda selected: self._build_embed_model(model_id, selected),
                model_identity=(model_id, 256, "legacy"), known_bytes=self._MODEL_WEIGHT_BYTES.get(model_id, 0),
            )
            # A changed generation, including A -> B -> A, must not publish this
            # construction. Its already-captured request may still finish on it.
            if self._residency_lock.acquire(blocking=False):
                try:
                    state = self._embed_state
                    if state is not None and state.generation is generation:
                        self._fast_encoder = model
                        self._fast_model_id = model_id
                        self._fast_generation = generation
                finally:
                    self._residency_lock.release()
                    state = None
            log.info(
                "Fast-lane CPU encoder loaded (tier=%s, model=%s) — single-text "
                "requests no longer queue behind batch work", tier, model_id,
            )
            return model

    def _get_reranker(
        self, model_name: str | None = None, deadline: _RequestDeadline | None = None,
    ):
        from contextlib import nullcontext
        scope = self._memory_scope() if getattr(self, "_memory_owner", None) is not None else nullcontext(None)
        with scope as _memory:
            from truememory.reranker import get_current_reranker_name
            name = model_name or get_current_reranker_name()

            if self._reranker is not None and self._reranker_name == name:
                return self._reranker

            self._reranker = None
            self._reranker_name = None
            self._check_process_memory("reranker construction", deadline)
            from truememory.mps_utils import auto_detect_device, ensure_mps_memory_budget, resolve_device
            if "rerank" in self._sticky_cpu:
                device = "cpu"
            else:
                device = resolve_device(auto_detect_device())

            def build(selected: str | None) -> object:
                ensure_mps_memory_budget(selected)
                from sentence_transformers import CrossEncoder
                if deadline is not None:
                    deadline.check()
                return CrossEncoder(name, device=selected)

            model, device = self._construct_model(
                "rerank", "reranker construction", device, deadline, build,
                model_identity=(name, 0, "reranker"), known_bytes=self._MODEL_WEIGHT_BYTES.get(name, 0),
            )
            self._reranker = model
            self._reranker_name = name
            log.info("Loaded reranker model=%s device=%s", name, device)
            return self._reranker

    def _check_serving_embed_model(self, model: object, target: "EmbeddingTarget | None") -> None:
        if target is None:
            return
        from truememory.embedding_target import EmbeddingTargetError

        state = self._embed_state
        if state is None or state.model is not model or state.model_id != target.model_id:
            raise EmbeddingTargetError("Effective encoder does not match the serving target")

    @staticmethod
    def _serving_embed_response(vectors: object, count: int, target: "EmbeddingTarget | None") -> dict:
        if target is None:
            return {"ok": True, "vectors": vectors}
        from truememory.embedding_target import check_target_vectors

        check_target_vectors(target, vectors, count)
        return {"ok": True, "vectors": vectors, "target": target.to_wire(), "protocol": PROTOCOL_VERSION}

    @staticmethod
    def _serving_reranker_name(value: object) -> str:
        import re

        if (type(value) is not str or len(value.encode("utf-8")) > 512
                or re.fullmatch(r"[\w][\w.\-]*(/[\w][\w.\-]*)?", value) is None):
            raise _ServingIdentityMismatch("Selected reranking requires an explicit model identity")
        return value

    def _check_serving_reranker(self, model: object, expected: str | None) -> None:
        if expected is not None and (self._reranker is not model or self._reranker_name != expected):
            raise _ServingIdentityMismatch("Effective reranker does not match the serving target")

    def _handle_fast_embed(
        self, texts: list, tier: str, deadline: _RequestDeadline | None = None,
        *, expected_target: "EmbeddingTarget | None" = None,
    ) -> dict | None:
        """Single-text fast lane (issue #577).

        Tries the main path without blocking; when its inference owner is busy
        (a batch encode or OOM recovery is in progress) the text is encoded
        on a dedicated CPU encoder OUTSIDE the global lock, so hook recall
        queries never wait on ingestion work. Fast-lane requests skip the
        sustained-workload bookkeeping entirely — a burst of small hook
        queries must not trip the throttler's batch=1 ramp (finding C-7).

        Returns a response dict, or None to fall through to the normal
        (locked) path.
        """
        deadline = deadline or _RequestDeadline()
        result_options = (self._result_admission_options()
                          if getattr(self, "_memory_owner", None) is not None else {})
        deadline.check()
        if self._inference_lock.acquire(blocking=False):
            try:
                # No inference is running, so the state lock is only held for
                # bookkeeping. Wait briefly rather than build a second encoder.
                remaining = deadline.remaining()
                wait = 0.25 if remaining is None else max(0.0, min(0.25, remaining))
                if self._lock.acquire(timeout=wait):
                    model = None
                    try:
                        deadline.check()
                        deadline.start_inference()
                        self._preflight_embed_result(tier, len(texts))
                        deadline.check()
                        model = self._get_embed_model(tier, deadline=deadline)
                        self._check_serving_embed_model(model, expected_target)
                        self._check_process_memory("embed slice", deadline)
                        deadline.check()
                        recover = False
                        try:
                            vectors = model.encode(texts, batch_size=1, show_progress_bar=False)
                        except RuntimeError as exc:
                            from truememory.mps_utils import is_mps_oom
                            if not is_mps_oom(exc):
                                raise
                            recover = True
                        if recover:
                            # Leave the exception scope before recovery so the
                            # failed forward's traceback no longer owns tensors.
                            self._check_embed_recovery_deadline_locked(model, deadline)
                            self._recover_embed_oom_locked(model, deadline)
                            self._check_process_memory("embed CPU retry", deadline)
                            deadline.check()
                            vectors = model.encode(texts, batch_size=1, show_progress_bar=False)
                        return self._serving_embed_response(
                            _checked_batch(vectors, len(texts), len(texts), "vectors", **result_options),
                            len(texts), expected_target,
                        )
                    finally:
                        model = None
                        self._lock.release()
            finally:
                self._inference_lock.release()

        was_started = deadline.transport.started if deadline.transport is not None else False
        fast_owned = False
        wait_for_main = False
        try:
            try:
                with deadline.locked(self._fast_lock):
                    fast_owned = True
                    model = None
                    try:
                        deadline.start_inference()
                        try:
                            model = (self._get_fast_encoder(tier, deadline=deadline) if expected_target is None else
                                     self._get_fast_encoder(tier, deadline=deadline, expected_target=expected_target))
                        except _ProcessMemoryRefused:
                            # Only the second copy does not fit; a process over budget stays refused.
                            self._check_process_memory("fast embed fallback", deadline)
                            wait_for_main = True
                        deadline.check()
                        if model is None:
                            if deadline.transport is not None:
                                deadline.transport.started = was_started
                            if not wait_for_main:
                                # Custom identities decline rather than drift.
                                return None
                        else:
                            self._check_process_memory("fast embed slice", deadline)
                            deadline.check()
                            vectors = model.encode(texts, batch_size=1, show_progress_bar=False)
                            if expected_target is not None:
                                response = self._serving_embed_response(
                                    _checked_batch(vectors, len(texts), len(texts), "vectors", **result_options),
                                    len(texts), expected_target,
                                )
                    finally:
                        model = None
            finally:
                if fast_owned:
                    self._retire_stale_fast_encoder()
            if wait_for_main:
                # Wait briefly for the main model outside the fast-lane lock, so
                # concurrent queries do not queue behind each other. A long batch
                # stays refused so hook recall falls back to keyword search promptly.
                remaining = deadline.remaining()
                wait = 1.0 if remaining is None else max(0.0, min(1.0, remaining))
                if not self._inference_lock.acquire(timeout=wait):
                    raise _ProcessMemoryRefused("Model server busy: the main model is serving a long batch")
                self._inference_lock.release()
                return None
            return (response if expected_target is not None else
                    {"ok": True, "vectors": _checked_batch(vectors, len(texts), len(texts), "vectors",
                                                          **result_options)})
        except (_RequestDeadlineExceeded, _ResultTooLarge, _ProcessMemoryRefused):
            raise
        except Exception as exc:
            if expected_target is not None:
                from truememory.embedding_target import EmbeddingTargetError
                if isinstance(exc, EmbeddingTargetError):
                    # An identity/shape rejection cannot be relabelled by a
                    # later fallback. Other fast-lane failures still retry.
                    raise
            if deadline.transport is not None:
                deadline.transport.started = was_started
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
            if getattr(self, "_memory_owner", None) is None:
                return self._handle_request_inner(request)
            with self._memory_scope() as memory:
                response = self._handle_request_inner(request)
                self._retain_memory_response(memory, response)
                return response
        except _RequestDeadlineExceeded as exc:
            return {"ok": False, "error": str(exc)}
        except _ResultTooLarge as exc:
            return {"ok": False, "error": str(exc), "error_code": "result_too_large"}
        except _ProcessMemoryRefused as exc:
            return {"ok": False, "error": str(exc), "error_code": "server_busy", "retry_after_ms": 250}
        finally:
            self._transport_context.memory_deadline = None
            with self._activity_lock:
                self._inflight -= 1
                self._last_activity = time.time()

    def _handle_request_inner(self, request: dict) -> dict:
        op = request.get("op")
        enrolled = getattr(self, "_memory_owner", None) is not None
        result_options = self._result_admission_options() if enrolled else {}
        preflight_options = self._result_admission_options(materialize=False) if enrolled else {}

        # Convert the wire epoch once. Queueing, loading and recovery consume
        # one monotonic budget; wall-clock adjustments cannot renew it.
        deadline = _RequestDeadline.from_wall_clock(request.get("deadline"))
        deadline = _RequestDeadline(deadline.expires_at, getattr(self._transport_context, "request", None))
        self._transport_context.memory_deadline = deadline
        deadline.check()

        if op == "ping":
            return {"ok": True}

        if op in ("embed", "embed_batched", "prepare_embed_target_v1", "embed_target_v1"):
            target = None
            expected_target = None
            preparing = op == "prepare_embed_target_v1"
            if op in ("prepare_embed_target_v1", "embed_target_v1"):
                from truememory.embedding_target import (
                    EmbeddingTarget, TARGET_PROBE_TEXT, check_target_vectors,
                )
                target = EmbeddingTarget.from_wire(request.get("target"))
                target.check_configuration()
                texts = [TARGET_PROBE_TEXT] if preparing else request["texts"]
                if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                    raise ValueError("Prepared embedding inputs must be a list of strings")
                empty_target_input = not texts
                if empty_target_input:
                    texts = [TARGET_PROBE_TEXT]
                tier = target.tier
            else:
                texts = request["texts"]
                tier = request.get("tier", "")
                if "expected_target" in request:
                    from truememory.embedding_target import EmbeddingTarget, EmbeddingTargetError, check_target_vectors
                    expected_target = EmbeddingTarget.from_wire(request["expected_target"])
                    if expected_target.tier == "custom" or tier != expected_target.model_id:
                        raise EmbeddingTargetError("Selected serving requires its canonical built-in model identity")
                    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                        raise EmbeddingTargetError("Selected embedding inputs must be a list of strings")
            try:
                batch_limit = _batch_limit(request.get("batch_size", _EMBED_BATCH_LIMIT),
                                           _EMBED_BATCH_LIMIT)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}

            # Single-text fast lane (issue #577): hook recall queries must
            # never queue behind batch ingestion work or OOM recovery.
            if target is None and len(texts) <= self._FAST_LANE_MAX_TEXTS:
                fast = (self._handle_fast_embed(texts, tier, deadline) if expected_target is None else
                        self._handle_fast_embed(texts, tier, deadline, expected_target=expected_target))
                if fast is not None:
                    return fast

            now = time.time()
            with deadline.locked(self._lock):
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
                with deadline.locked(self._lock):
                    self._activate_throttler()

            limit, throttler = self._request_batch_limit(batch_limit, deadline)
            with deadline.locked(self._inference_lock):
                model = None
                try:
                    encode_start = time.monotonic()
                    vectors = None
                    offset = 0
                    order = None
                    while offset < len(texts) or vectors is None:
                        retry = None
                        with deadline.locked(self._lock):
                            if model is None:
                                deadline.start_inference()
                                if target is None:
                                    self._preflight_embed_result(tier, len(texts))
                                else:
                                    _check_result_size((len(texts), target.dimension), "vectors")
                                    if enrolled:
                                        self._reserve_result_assembly((len(texts), target.dimension))
                                deadline.check()
                                model = (self._get_embed_model(tier, deadline=deadline) if target is None
                                         else self._get_prepared_embed_model(target, deadline))
                                self._check_serving_embed_model(model, expected_target)
                                order = self._embed_global_order(model, texts, limit, deadline)
                            while offset < len(texts) or vectors is None:
                                deadline.check()
                                indices = None if order is None else self._embed_slice_indices(
                                    model, texts, order, offset, limit, deadline,
                                )
                                batch = (texts[offset:offset + limit] if indices is None else
                                         [texts[index] for index in indices])
                                self._check_process_memory("embed slice", deadline)
                                deadline.check()
                                recover = False
                                try:
                                    values = model.encode(batch, batch_size=limit, show_progress_bar=False)
                                except RuntimeError as exc:
                                    from truememory.mps_utils import is_mps_oom
                                    if not is_mps_oom(exc):
                                        raise
                                    recover = True
                                if recover:
                                    self._check_embed_recovery_deadline_locked(model, deadline)
                                    self._recover_embed_oom_locked(model, deadline)
                                    retry = batch
                                    break
                                if target is not None or expected_target is not None:
                                    check_target_vectors(target or expected_target, values, len(batch))
                                vectors = _store_batch_result(
                                    vectors, values, offset, len(batch), len(texts), indices=indices,
                                    **result_options,
                                )
                                offset += len(batch)
                        if retry is not None:
                            # Release only the state lock; inference ownership
                            # still covers the retry and remaining slices.
                            self._check_process_memory("embed CPU retry", deadline)
                            deadline.check()
                            log.warning("MPS OOM during encoding; retrying microbatch on CPU")
                            values = model.encode(retry, batch_size=limit, show_progress_bar=False)
                            if target is not None or expected_target is not None:
                                check_target_vectors(target or expected_target, values, len(retry))
                            vectors = _store_batch_result(
                                vectors, values, offset, len(retry), len(texts), indices=indices,
                                **result_options,
                            )
                            offset += len(retry)
                    self._after_request_batches(throttler, len(texts), encode_start, deadline)

                    with deadline.locked(self._lock):
                        should_deactivate = self._throttler_active and len(self._embed_timestamps) < 3
                        if should_deactivate:
                            self._deactivate_throttler()

                    if target is not None:
                        return ({"ok": True, "target": target.to_wire()} if preparing else
                                {"ok": True, "target": target.to_wire(),
                                 "vectors": vectors[:0] if empty_target_input else vectors})
                    return self._serving_embed_response(vectors, len(texts), expected_target)
                finally:
                    model = None

        if op in ("rerank", "rerank_batched"):
            pairs = request["pairs"]
            model_name = request.get("model_name")
            expected_name = None
            if "expected_model_name" in request:
                expected_name = self._serving_reranker_name(request["expected_model_name"])
                if model_name != expected_name:
                    raise _ServingIdentityMismatch("Selected reranking requires its explicit model identity")
            try:
                batch_limit = _batch_limit(request.get("batch_size", _RERANK_BATCH_LIMIT),
                                           _RERANK_BATCH_LIMIT)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            limit, throttler = self._request_batch_limit(batch_limit, deadline)
            with deadline.locked(self._inference_lock):
                reranker = None
                try:
                    predict_start = time.monotonic()
                    scores = None
                    offset = 0
                    order = None
                    recovery_name = model_name
                    while offset < len(pairs) or scores is None:
                        retry = None
                        with deadline.locked(self._lock):
                            if reranker is None:
                                deadline.start_inference()
                                self._preflight_rerank_result(model_name, len(pairs), **preflight_options)
                                deadline.check()
                                reranker = self._get_reranker(model_name, deadline=deadline)
                                self._check_serving_reranker(reranker, expected_name)
                                order = self._rerank_global_order(reranker, pairs, limit, deadline)
                                if order is not None:
                                    recovery_name = self._reranker_name
                            while offset < len(pairs) or scores is None:
                                deadline.check()
                                indices = None if order is None else self._rerank_slice_indices(
                                    reranker, pairs, order, offset, limit, deadline,
                                )
                                batch = (pairs[offset:offset + limit] if indices is None else
                                         [pairs[index] for index in indices])
                                self._check_process_memory("reranker slice", deadline)
                                deadline.check()
                                recover = False
                                try:
                                    values = reranker.predict(batch, batch_size=limit, show_progress_bar=False)
                                except RuntimeError as exc:
                                    from truememory.mps_utils import is_mps_oom, flush_mps_cache
                                    if not is_mps_oom(exc):
                                        raise
                                    recover = True
                                if recover:
                                    self._mark_sticky_cpu("rerank")
                                    self._reranker = None
                                    self._reranker_name = None
                                    # Neither the failed traceback nor this
                                    # local may retain the obsolete accelerator
                                    # model while its CPU replacement is built.
                                    reranker = None
                                    deadline.check()
                                    flush_mps_cache()
                                    deadline.check()
                                    reranker = self._get_reranker(recovery_name, deadline=deadline)
                                    self._check_serving_reranker(reranker, expected_name)
                                    retry = batch
                                    break
                                if indices is None:
                                    scores = _store_batch_result(scores, values, offset, len(batch), len(pairs),
                                                                 "scores", **result_options)
                                else:
                                    scores = _store_batch_result(
                                        scores, values, offset, len(batch), len(pairs), "scores", indices=indices,
                                        **result_options,
                                    )
                                offset += len(batch)
                        if retry is not None:
                            self._check_process_memory("reranker CPU retry", deadline)
                            deadline.check()
                            values = reranker.predict(retry, batch_size=limit, show_progress_bar=False)
                            if indices is None:
                                scores = _store_batch_result(scores, values, offset, len(retry), len(pairs),
                                                             "scores", **result_options)
                            else:
                                scores = _store_batch_result(
                                    scores, values, offset, len(retry), len(pairs), "scores", indices=indices,
                                    **result_options,
                                )
                            offset += len(retry)
                    self._after_request_batches(throttler, len(pairs), predict_start, deadline)
                    return ({"ok": True, "scores": scores} if expected_name is None else
                            {"ok": True, "scores": scores, "protocol": PROTOCOL_VERSION,
                             "reranker": {"version": 1, "model_name": expected_name}})
                finally:
                    reranker = None

        return {"ok": False, "error": f"Unknown op: {op}"}

    def _request_batch_limit(
        self, limit: int, deadline: _RequestDeadline,
    ) -> tuple[int, object | None]:
        # Preserve once-per-request sensing outside the model state lock.
        # Per-slice sensing needs a scheduler.
        # Capture the active instance under the lock, preserving #646's race fix.
        with deadline.locked(self._lock):
            throttler = self._throttler if self._throttler_active else None
        if throttler is not None and not getattr(throttler, "adaptive_applicable", True):
            # No applicable sensor policy is not healthy ramp evidence. Keep
            # the existing caller/server hard bound without MPS slow-start or
            # pacing. Recheck on each request for newly published MPS state.
            deadline.check()
            return limit, None
        if throttler is not None:
            deadline.check()
            safe_limit, _ = throttler.before_batch()
            deadline.check()
            limit = _batch_limit(safe_limit, limit)
        return limit, throttler

    def _after_request_batches(
        self, throttler: object | None, count: int, started: float,
        deadline: _RequestDeadline,
    ) -> None:
        deadline.check()
        from truememory.mps_utils import trim_mps_cache
        trim_mps_cache()
        if throttler is not None:
            throttler.after_batch(count, time.monotonic() - started)
            if throttler.should_flush_cache():
                deadline.check()
                self._flush_mps_cache()

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
            from truememory.mps_utils import get_mps_memory_budget
            import torch
            gc.collect()
            if get_mps_memory_budget() is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
        except Exception:
            gc.collect()

    _CLIENT_TIMEOUT = 30.0
    _REJECT_TIMEOUT = 0.25

    def _admit_client(self, conn: socket.socket) -> _TransportRequest | None:
        # run() has one accept/staging slot. Direct synchronous users cannot
        # create a second staging queue behind it.
        if not self._staging_lock.acquire(blocking=False):
            conn.close()
            return None
        admitted = None
        accepted_at = time.monotonic()
        try:
            with self._admission_lock:
                if self._stopped.is_set():
                    return None
                self._staging = conn
            frame_deadline = accepted_at + self._frame_timeout
            header_deadline = min(accepted_at + self._header_timeout, frame_deadline)
            if not _USE_UNIX:
                if self._token is None:
                    return None
                token = self._recv_exact(conn, _HMAC_TOKEN_BYTES, header_deadline, charge=False)
                if token is None or not hmac.compare_digest(token, self._token):
                    return None
            header = self._recv_exact(conn, _HEADER_SIZE, header_deadline, charge=False)
            if header is None:
                return None
            length = struct.unpack(_HEADER_FMT, header)[0]
            if not 0 < length <= _MAX_MESSAGE_SIZE:
                log.warning("Rejecting invalid request frame length (%d bytes)", length)
                return None
            with self._admission_lock:
                if self._stopped.is_set():
                    return None
                if (len(self._clients) >= self._max_handlers
                        or self._reserved_request_bytes + length > self._max_request_bytes):
                    self._busy_rejections += 1
                else:
                    admitted = _TransportRequest(
                        length, accepted_at + self._queue_timeout, self._stopped,
                        frame_deadline,
                    )
                    self._clients[conn] = admitted
                    self._reserved_request_bytes += length
                    self._admission_high_water["handlers"] = max(
                        self._admission_high_water["handlers"], len(self._clients),
                    )
                    self._admission_high_water["request_bytes"] = max(
                        self._admission_high_water["request_bytes"], self._reserved_request_bytes,
                    )
            if admitted is None:
                self._reject_client(conn, length, "server_busy", "Model server busy: request capacity exhausted")
            return admitted
        except (OSError, TimeoutError):
            return None
        except _ProcessMemoryRefused as error:
            # Memory pressure refuses this connection, never the accept loop.
            self._reject_client(conn, 0, "server_busy", str(error))
            return None
        finally:
            with self._admission_lock:
                self._staging = None
            self._staging_lock.release()
            if admitted is None:
                conn.close()

    def _reject_client(self, conn: socket.socket, length: int, code: str, message: str) -> None:
        expires_at = time.monotonic() + self._REJECT_TIMEOUT
        try:
            conn.settimeout(self._REJECT_TIMEOUT)
            response = {"ok": False, "error_code": code, "error": message}
            if code == "server_busy":
                response["retry_after_ms"] = 250
            self._send_response(conn, response)
            conn.shutdown(socket.SHUT_WR)
            # Legacy clients send before reading. Drain without retaining their
            # body so closing unread input does not discard the busy response.
            while length > 0:
                remaining = expires_at - time.monotonic()
                if remaining <= 0:
                    break
                conn.settimeout(remaining)
                chunk = conn.recv(min(length, 4096))
                if not chunk:
                    break
                length -= len(chunk)
        except (OSError, _ProcessMemoryRefused):
            pass

    def _dispatch_client(self, conn: socket.socket) -> Future | None:
        admitted = self._admit_client(conn)
        if admitted is None:
            return None
        try:
            return self._workers.submit(self._run_admitted_client, conn, admitted)
        except (RuntimeError, OSError, MemoryError):
            # submit() may enqueue before thread.start fails. Withdraw unclaimed
            # leases and disable the executor rather than leaving orphan work.
            self._stop_clients()
            raise

    def handle_client(self, conn: socket.socket) -> None:
        admitted = self._admit_client(conn)
        if admitted is not None:
            self._run_admitted_client(conn, admitted)

    def _run_admitted_client(self, conn: socket.socket, admitted: _TransportRequest) -> None:
        with self._admission_lock:
            if admitted.withdrawn:
                return
            admitted.claimed = True
        try:
            with self._memory_scope() as _memory:
                self._serve_client(conn, admitted)
        finally:
            # The serving frame and exception traceback have returned. No
            # decoded request, raw body, or response survives credit release.
            self._release_client(conn)

    def _release_client(self, conn: socket.socket) -> None:
        try:
            conn.close()
        finally:
            with self._admission_lock:
                admitted = self._clients.pop(conn, None)
                if admitted is not None:
                    self._reserved_request_bytes -= admitted.length
            with self._activity_lock:
                self._last_activity = time.time()

    def _serve_client(self, conn: socket.socket, admitted: _TransportRequest) -> None:
        data = request = response = None
        try:
            data = self._recv_exact(conn, admitted.length, admitted.frame_expires_at)
            if data is None:
                return
            import hashlib
            self._memory_stage("request materialize", known_bytes=0, shape=(len(data),),
                               wire_signature=hashlib.sha256(data).hexdigest())
            request = json.loads(data, object_hook=_json_object_hook)
            data = None
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object")
            self._transport_context.request = admitted
            response = self.handle_request(request)
            conn.settimeout(self._CLIENT_TIMEOUT)
            self._send_response(conn, response)
        except Exception as error:
            try:
                conn.settimeout(self._REJECT_TIMEOUT)
                failure = {"ok": False, "error": str(error)}
                identity_error = getattr(type(error), "serving_identity_mismatch", False) is True
                if isinstance(request, dict) and (
                    "expected_target" in request or request.get("op") in ("prepare_embed_target_v1", "embed_target_v1")
                ):
                    from truememory.embedding_target import EmbeddingTargetError
                    identity_error = identity_error or isinstance(error, EmbeddingTargetError)
                if identity_error:
                    failure["error_code"] = "serving_identity_mismatch"
                elif isinstance(error, _ProcessMemoryRefused):
                    failure["error_code"] = "server_busy"
                    failure["retry_after_ms"] = 250
                self._send_response(conn, failure)
            except Exception:
                # The connection is already failing. Keep serialization errors
                # inside this boundary so their tracebacks do not retain input.
                pass
        finally:
            self._transport_context.request = None
            self._transport_context.memory_deadline = None
            data = request = response = None

    def _stop_clients(self) -> None:
        self._running = False
        self._stopped.set()
        with self._admission_lock:
            connections = list(self._clients)
            if self._staging is not None:
                connections.append(self._staging)
            for conn, admitted in list(self._clients.items()):
                if not admitted.claimed:
                    admitted.withdrawn = True
                    del self._clients[conn]
                    self._reserved_request_bytes -= admitted.length
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._workers.shutdown(wait=False, cancel_futures=True)
        for conn in connections:
            with self._admission_lock:
                retained = conn in self._clients
            if not retained:
                conn.close()

    def _recv_exact(
        self, conn: socket.socket, n: int, expires_at: float | None = None, *, charge: bool = True,
    ) -> bytes | None:
        with self._memory_scope() as _memory:
            # The accept thread reads fixed-size headers only. Sampled RSS from
            # concurrent inference must not refuse, or stop, the accept loop.
            if charge:
                self._memory_stage("frame receive", _RequestDeadline(expires_at),
                                   known_bytes=2 * n + min(n, 65536), shape=(n,))
            buf = bytearray(n)
            offset = 0
            while offset < n:
                if self._stopped.is_set():
                    return None
                if expires_at is not None:
                    remaining = expires_at - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("request frame receive timed out: deadline exceeded")
                    # Cross-thread shutdown does not reliably interrupt recv on
                    # every platform. Poll shutdown without extending the frame budget.
                    conn.settimeout(min(remaining, 0.1))
                try:
                    chunk = conn.recv(min(n - offset, 65536))
                except TimeoutError:
                    if expires_at is None:
                        raise
                    continue
                if not chunk:
                    return None
                buf[offset:offset + len(chunk)] = chunk
                offset += len(chunk)
            return bytes(buf)

    def _send_response(self, conn: socket.socket, response: dict):
        # Issue #646 (M-53): tag every response with the protocol version so
        # a newer client can detect a version mismatch instead of choking on
        # an unexpected payload shape.
        with self._memory_scope() as _memory:
            if "protocol" not in response:
                response = {**response, "protocol": PROTOCOL_VERSION}
            # Error and busy replies are small. Refusing them under memory
            # pressure would drop the connection, which clients treat as a
            # dead server and answer by launching another one.
            charged = response.get("ok") is not False
            if charged:
                self._memory_stage("response metadata")
            wire_bytes = _response_wire_size(response)
            if wire_bytes > _MAX_MESSAGE_SIZE:
                response = {"ok": False, "error": "Response too large", "protocol": PROTOCOL_VERSION}
                wire_bytes = _response_wire_size(response)
            if charged and getattr(self, "_memory_owner", None) is not None:
                payload_bytes = sum(math.prod(value.shape) * 4 for value in response.values()
                                    if isinstance(value, np.ndarray))
                encoded_bytes = sum(_array_base64_size(value.shape) for value in response.values()
                                    if isinstance(value, np.ndarray))
                # Original ndarray storage has a separate retained-result ticket.
                copies = 2 * payload_bytes + 2 * encoded_bytes + 3 * wire_bytes + _HEADER_SIZE
                self._memory_stage("response serialize", known_bytes=copies, shape=(wire_bytes,))
            data = json.dumps(response, default=_json_default).encode("utf-8")
            header = struct.pack(_HEADER_FMT, len(data))
            conn.sendall(header + data)

    def _idle_checker(self):
        while self._running:
            time.sleep(60)
            if not self._running:
                break
            with self._admission_lock:
                admitted = bool(self._clients) or self._staging is not None
                with self._activity_lock:
                    last = self._last_activity
                    inflight = self._inflight
                    elapsed = time.time() - last
                    should_stop = elapsed >= IDLE_TIMEOUT and inflight == 0 and not admitted
                    if should_stop:
                        self._running = False
                        self._stopped.set()
            # Issue #646 (M-74): never idle-shut-down while a request is
            # mid-flight, even if its start timestamp is older than the idle
            # horizon (a long batch encode can outlast IDLE_TIMEOUT).
            if should_stop:
                log.info(
                    "Idle timeout (%.0fs), shutting down model server", elapsed
                )
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
        fd = try_file_lock(LOCK_PATH)
        if fd is None:
            raise RuntimeError("another model server holds the bind lock")
        return fd

    def _validate_launch_generation_locked(self) -> None:
        """A superseded managed child must not publish endpoint artifacts."""
        generation = os.environ.get(_MODEL_SERVER_GENERATION_ENV)
        if generation is None:
            return  # Explicit CLI launch still uses the lifetime bind lock.
        try:
            claim = read_start_claim(START_STATE_PATH)
        except OSError as error:
            os.close(self._lock_fd)
            self._lock_fd = None
            raise RuntimeError("Cannot verify model-server launch generation") from error
        if claim is None or claim["generation"] != generation:
            os.close(self._lock_fd)
            self._lock_fd = None
            raise RuntimeError("model-server launch was superseded; retry the request")

    def run(self):
        if getattr(self, "_cleaned_up", False):
            raise RuntimeError("a stopped model server cannot be restarted in the same instance")
        # M-89: keep ~/.truememory owner-only (0700) — it holds memories/PII.
        _TRUEMEMORY_DIR.mkdir(parents=True, exist_ok=True)
        try:
            _TRUEMEMORY_DIR.chmod(0o700)
        except OSError:
            pass

        # Exclusive bind lock BEFORE touching socket/pid artifacts (M-20).
        self._lock_fd = self._acquire_bind_lock()
        self._validate_launch_generation_locked()

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
        if "TRUEMEMORY_REQUIRE_PROCESS_MEMORY_CAP" in os.environ:
            from truememory_memory_bootstrap import record_server_ready
            record_server_ready()

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
                self._dispatch_client(conn)
        finally:
            srv.close()
            self._cleanup()

    def _cleanup(self):
        if getattr(self, "_cleaned_up", False):
            return
        self._cleaned_up = True
        self._stop_clients()
        with self._admission_lock:
            with self._activity_lock:
                retained_work = bool(self._clients) or self._inflight > 0
        if retained_work:
            # Native inference cannot be preempted. Keep model references and
            # bind ownership until OS exit so another daemon cannot overlap
            # that work. Cleanup remains nonblocking and the instance cannot
            # be restarted; stale endpoint files are reclaimed by the next
            # bind owner after this process exits.
            log.info("Shutdown retains model-server ownership until process exit")
            return
        # Keep bind ownership until every cached model has been released,
        # including cycles collected below. If teardown fails, retain the
        # lock until process exit instead of overlapping a successor's load.
        self._embed_state = None
        self._reranker = None
        self._reranker_name = None
        self._fast_encoder = None
        self._fast_model_id = None
        self._fast_generation = None
        self._token = None
        gc.collect()
        if getattr(self, "_memory_owner", None) is not None:
            self._memory_owner.close()
        # Only remove artifacts THIS process owns (issue #646, M-20). After a
        # crash a fresh server may already hold the lock and have rewritten
        # PID_PATH; unlinking its live socket/token here would let concurrent
        # hooks cycle servers indefinitely.
        lock_fd = getattr(self, "_lock_fd", None)
        if lock_fd is not None and _safe_to_cleanup_artifacts():
            for p in (SOCK_PATH, PID_PATH, PORT_PATH, TOKEN_PATH):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        # Release the bind lock (lock fd is process-owned, always safe).
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass
            self._lock_fd = None
        log.info("Model server stopped")


def _handle_signal(signum, frame):
    log.info("Received signal %d, shutting down", signum)
    sys.exit(0)


def main():
    if "TRUEMEMORY_REQUIRE_PROCESS_MEMORY_CAP" in os.environ:
        from truememory_memory_bootstrap import require_child_scope
        require_child_scope()
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

    # A PID file can outlive its process and point at an unrelated live PID.
    # Only run()'s exclusive bind lock decides endpoint ownership.
    server = ModelServer(automatic_rss_budget=True)

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
