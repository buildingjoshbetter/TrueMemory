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
    pid_is_alive, read_start_claim, release_file_lock, try_file_lock,
)

log = logging.getLogger(__name__)

_TRUEMEMORY_DIR = Path.home() / ".truememory"
SOCK_PATH = _TRUEMEMORY_DIR / "model.sock"
PID_PATH = _TRUEMEMORY_DIR / "model_server.pid"
PORT_PATH = _TRUEMEMORY_DIR / "model_server.port"
TOKEN_PATH = _TRUEMEMORY_DIR / "model_server.token"
IDLE_TIMEOUT = _env_int("TRUEMEMORY_MODEL_SERVER_IDLE", 300, lo=0)
# After TERM, in-flight model work gets this long to reach a stopping point.
_TERM_GRACE_SECONDS = 5.0

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


def _on_cpu(model: object) -> bool:
    device = getattr(model, "device", None)
    return getattr(device, "type", device) == "cpu"


class _FastLaneNotReady(_ProcessMemoryRefused):
    """The retired fast-lane clone is reloading; take the bounded fallback."""

    fast_lane_reloading = True


class _MpsBuildRefused(_ProcessMemoryRefused):
    """A construction allowed only on MPS found no room there, or ran out of it."""

    def __init__(self, message: str, *, oom: bool) -> None:
        super().__init__(message)
        self.mps_oom = oom


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
    # memory, never more than 4 GiB on macOS or 3 GiB elsewhere. The macOS
    # ceiling is 6 GiB: the MPS cap comes on top, and a measured soak peaked
    # at 5.6 GB, so more room would only let a degraded CPU path grow.
    budget = physical // 4
    if budget > 0:
        darwin = sys.platform == "darwin"
        floor = (4 if darwin else 3) * 1024**3
        budget = min((6 if darwin else 8) * 1024**3, max(budget, min(physical // 2, floor)))
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
        self._reranker_device: str | None = None
        # name -> (model, device, monotonic time it was switched out); under _lock.
        self._parked_rerankers: dict[str, tuple[object, str | None, float]] = {}
        # Monotonic time the current reranker was last requested; under _lock.
        self._reranker_last_used = 0.0
        # An MPS model released without inference ownership leaves its cache
        # for the next inference owner to empty.
        self._mps_flush_pending = False
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
        # Issue #577: models that hit an MPS OOM are degraded to CPU ("embed" /
        # "rerank:<name>") until they return to MPS or are released. Mutated only
        # while holding self._lock; read lock-free (string membership).
        self._sticky_cpu: set[str] = set()
        # Issue #577: dedicated CPU encoder for the single-text fast lane,
        # loaded lazily on the first *contended* single-text request.
        # A generation distinguishes A -> B -> A from an unchanged residency.
        self._fast_encoder = None
        self._fast_model_id: str | None = None
        self._fast_generation: object = None
        self._fast_lock = threading.Lock()
        # The clone costs 2.4 GB in float32 but only serves single queries while
        # batch work holds the main model. Release it after this many seconds
        # without such contention (0 keeps it); a retired residency then
        # rebuilds in the background so no query waits on the reload.
        self._fast_idle_seconds = _admission_setting("TRUEMEMORY_FAST_LANE_IDLE_SECONDS", 600, 0, 2**31 - 1)
        # Monotonic time of the last contention; every clone publication stamps it.
        self._fast_last_used = 0.0
        self._fast_retired_generation: object = None
        # At most one background rebuild; set and cleared under _residency_lock.
        self._fast_rebuild: threading.Thread | None = None
        # (monotonic time until which reloads wait, RSS when the last one was refused)
        self._fast_rebuild_refusal: tuple[float, int | None] = (0.0, None)
        # Embedding sticky CPU is retried on MPS after a backoff (see _repromote_embed_locked).
        self._embed_cpu_since: float | None = None
        self._embed_ooms = 0
        self._embed_repromote_deferred = False
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
        """Degrade *kind* ("embed" or "rerank:<name>") to CPU after an MPS OOM
        (issue #577). The model is not moved back while it serves; embedding
        returns to MPS through a fresh load after a backoff
        (``_repromote_embed_locked``), and a reranker when its CPU copy is released.

        Caller must hold ``self._lock``. Returns True on the first marking
        (which is logged loudly, once).
        """
        if kind in self._sticky_cpu:
            return False
        self._sticky_cpu.add(kind)
        if kind == "embed":
            import time as clock  # this method is also loaded alone by source tests
            self._embed_cpu_since = clock.monotonic()
            self._embed_ooms = getattr(self, "_embed_ooms", 0) + 1
        if kind.startswith("rerank:"):
            # One reranker that does not fit says nothing about the others.
            log.error("MPS OOM with reranker %s — serving it on CPU until it is "
                      "released; other models keep MPS.", kind.partition(":")[2])
            self._write_status_file()
            return True
        log.error(
            "MPS OOM in the %s path — degrading the %s model to CPU. MPS is "
            "retried after %ds; set TRUEMEMORY_DEVICE=cpu to make CPU permanent.",
            kind, kind, self._embed_repromote_after() if kind == "embed" and hasattr(self, "_embed_repromote_after") else 0,
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
            # The CPU copy about to be made duplicates any fast-lane clone.
            # Release the clone first, so the transfer is admitted against
            # memory that is actually free.
            self._release_fast_clone()
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
                # A fast-lane clone now duplicates the CPU main model. A clone in
                # use is released by its owner when the query finishes.
                self._retire_stale_fast_encoder()

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
        "BAAI/bge-reranker-v2-m3": 2_272_000_000,
    }
    # Bytes each model holds on MPS, in its checkpoint dtype.
    _MPS_WEIGHT_BYTES = {
        "qwen3_256": 1_200_000_000,
        "Alibaba-NLP/gte-reranker-modernbert-base": 600_000_000,
        "cross-encoder/ms-marco-MiniLM-L-6-v2": 92_000_000,
        "BAAI/bge-reranker-v2-m3": 2_272_000_000,
    }
    # A switched-out reranker stays resident (parked) so alternating between the
    # default and the deep-search reranker reloads neither. It leaves MPS for a
    # newcomer only after this long unused, so alternation never thrashes, and
    # is released after the longer period.
    _RERANKER_EVICT_IDLE_SECONDS = 60
    _PARKED_RERANKER_IDLE_SECONDS = 600
    # The current reranker is released after this long without a request.
    _RERANKER_IDLE_SECONDS = 600
    # A refused fast-lane reload is not retried for this long (see _start_fast_rebuild).
    _FAST_REBUILD_BACKOFF_SECONDS = 60
    # After an embedding MPS OOM the model runs on CPU; MPS is retried after this
    # long, doubled for every further OOM up to the cap.
    _EMBED_REPROMOTE_SECONDS = 600
    _EMBED_REPROMOTE_MAX_FACTOR = 16
    # Loading the MPS copy stages its weights in CPU memory first.
    _EMBED_REPROMOTE_ROOM_BYTES = 1_200_000_000

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

    def _retire_stale_fast_encoder(self, *, idle_at: float | None = None) -> bool:
        """Drop an idle stale clone; an active owner retries this on release.

        With *idle_at* (a monotonic time), a current clone is dropped too once
        it has gone the idle period without contention and no batch holds the
        main model. Its residency then rebuilds in the background. A clone of
        a main model that is itself on CPU duplicates its weights and is
        dropped at once; single queries then use the main model.
        """
        if not self._fast_lock.acquire(blocking=False):
            return False
        release_fast = True
        try:
            with self._residency_lock:
                state = self._embed_state
                current = state is not None and self._fast_generation is state.generation
                idle = (idle_at is not None and current and self._fast_idle_seconds > 0
                        and idle_at >= self._fast_last_used + self._fast_idle_seconds
                        and not self._inference_lock.locked())
                # The CPU main model lent to single queries during its return
                # to MPS is not a duplicate of itself.
                redundant = (current and self._fast_encoder is not state.model
                             and self._duplicates_cpu_main(state))
                if self._fast_encoder is None or (current and not idle and not redundant):
                    # Unlock before publication can change the generation, so
                    # a following invalidation cannot miss an idle stale cache.
                    self._fast_lock.release()
                    release_fast = False
                    return False
                if idle and not redundant:
                    self._fast_retired_generation = self._fast_generation
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
        if idle or redundant:
            # The clone is a reference cycle; collect it so its float32 weights
            # return to the system now rather than at the next full collection.
            self._collect_models()
            if redundant:
                log.info("Fast-lane CPU encoder released: the main embedding model is on CPU")
            else:
                log.info("Fast-lane CPU encoder released after %ds without contention",
                         self._fast_idle_seconds)
        return True

    def _embed_repromote_after(self) -> int:
        """Seconds on CPU before embedding retries MPS: doubles per OOM, capped."""
        factor = min(2 ** max(0, getattr(self, "_embed_ooms", 1) - 1), self._EMBED_REPROMOTE_MAX_FACTOR)
        return self._EMBED_REPROMOTE_SECONDS * factor

    def _embed_repromote_due(self, now: float | None = None) -> bool:
        """Whether embedding, on CPU since an MPS OOM, has waited out its backoff."""
        since = getattr(self, "_embed_cpu_since", None)
        if since is None or "embed" not in self._sticky_cpu:
            return False
        now = time.monotonic() if now is None else now
        # Compare against the deadline, not the elapsed time: ``since + period``
        # is the exact moment the backoff ends, with no float rounding gap.
        if now < since + self._embed_repromote_after():
            return False
        from truememory.mps_utils import resolve_device
        return resolve_device(None) != "cpu"  # TRUEMEMORY_DEVICE=cpu keeps CPU.

    def _repromote_embed(self, now: float | None = None) -> bool:
        """Retirer tick: return embedding to MPS when no request owns the model."""
        if not self._embed_repromote_due(now):
            return False
        if not self._inference_lock.acquire(blocking=False):
            return False  # That owner's next embed request swaps instead.
        try:
            if not self._lock.acquire(blocking=False):
                return False
            try:
                # No request owns this swap; only server shutdown stops it.
                deadline = _RequestDeadline(None, _TransportRequest(0, math.inf, self._stopped, started=True))
                return self._repromote_embed_locked(deadline, now=now)
            finally:
                self._lock.release()
        finally:
            self._inference_lock.release()

    def _repromote_embed_locked(self, deadline: _RequestDeadline, *, now: float | None = None) -> bool:
        """Swap a CPU embedding model whose backoff has passed back to MPS.

        Caller owns the inference and state locks, so no batch is running on
        the model. Its MPS copy is loaded fresh under MPS admission while the
        CPU model serves single queries through the fast lane; publishing the
        MPS copy retires the CPU one. Without RSS room for the load the swap
        waits for a later request; a refused or failed MPS load keeps CPU and
        restarts the backoff.
        """
        if not self._embed_repromote_due(now):
            return False
        state = self._embed_state
        if state is None or not _on_cpu(state.model):
            # Nothing on CPU to move: the next load is admitted on MPS.
            state = None
            self._end_embed_sticky()
            return True
        model_id, tier, generation = state.model_id, state.tier, state.generation
        if model_id not in self._FAST_LANE_SAFE_MODEL_IDS or model_id not in self._MPS_WEIGHT_BYTES:
            # Only a built-in model is rebuilt beside its CPU copy; any other
            # model is reloaded by its next request.
            state = None
            self._publish_embed_state(None)
            self._end_embed_sticky()
            self._discard_models()
            return True
        if not self._embed_repromote_room():
            state = None
            if not self._embed_repromote_deferred:
                self._embed_repromote_deferred = True
                log.info("Embedding stays on CPU until %.1f GB of the RSS budget is free to load it on MPS",
                         self._EMBED_REPROMOTE_ROOM_BYTES / 1e9)
            return False
        from truememory.mps_utils import auto_detect_device, resolve_device
        device = resolve_device(None) or auto_detect_device()
        if device == "cpu":
            state = None
            return False
        with self._fast_lock:
            self._publish_fast_encoder(state.model, model_id, generation, blocking=True)
        state = None
        model = None
        try:
            with self._memory_scope():
                model, _ = self._construct_model(
                    "embed", "embed construction", device, deadline,
                    lambda selected: self._build_embed_model(model_id, selected),
                    model_identity=(model_id, 256, "legacy"), known_bytes=self._MODEL_WEIGHT_BYTES.get(model_id, 0),
                    mps_bytes=self._MPS_WEIGHT_BYTES[model_id], cpu_fallback=False,
                    make_room=lambda shortfall: self._evict_parked_rerankers(shortfall, min_idle=0),
                )
            self._publish_embed_state(_EmbedState(model=model, tier=tier, model_id=model_id))
        except _MpsBuildRefused as error:
            self._embed_cpu_since = time.monotonic()
            if error.mps_oom:
                self._embed_ooms += 1
            log.info("Embedding stays on CPU: %s; MPS is retried after %ds", error, self._embed_repromote_after())
            return False
        except _ProcessMemoryRefused as error:
            log.info("Embedding return to MPS deferred: %s", error)
            return False
        except _RequestDeadlineExceeded:
            raise
        except Exception:
            self._embed_cpu_since = time.monotonic()
            log.warning("Embedding MPS load failed; staying on CPU, retried after %ds",
                        self._embed_repromote_after(), exc_info=True)
            return False
        finally:
            model = None
            self._end_cpu_embed_loan(generation)
        self._end_embed_sticky()
        # Inference is owned here, so emptying the MPS cache races no work.
        self._discard_models()
        return True

    def _end_cpu_embed_loan(self, generation: object) -> None:
        """Take the lent CPU model back once no single query is encoding on it."""
        with self._fast_lock:
            with self._residency_lock:
                if self._fast_generation is generation:
                    self._fast_encoder = None
                    self._fast_model_id = None
                    self._fast_generation = None

    def _end_embed_sticky(self) -> None:
        """Caller holds the state lock."""
        self._sticky_cpu.discard("embed")
        self._embed_cpu_since = None
        self._embed_repromote_deferred = False
        self._write_status_file()
        log.info("Embedding returns to MPS (OOM count %d)", self._embed_ooms)

    def _embed_repromote_room(self) -> bool:
        budget = self._max_rss_bytes
        if not budget:
            return True
        usage = self._resident_bytes()
        return usage is not None and budget - usage >= self._EMBED_REPROMOTE_ROOM_BYTES

    @staticmethod
    def _resident_bytes() -> int | None:
        try:
            usage = psutil.Process().memory_info().rss if psutil is not None else None
        except Exception:
            return None
        return usage if type(usage) is int and usage > 0 else None

    def _retire_idle_fast_encoder(self, now: float | None = None) -> bool:
        return self._retire_stale_fast_encoder(idle_at=time.monotonic() if now is None else now)

    def _idle_retirer(self) -> None:
        """Release the idle fast-lane copy and idle rerankers; retry MPS for embedding."""
        periods = [period for period in (self._fast_idle_seconds, self._PARKED_RERANKER_IDLE_SECONDS,
                                         self._RERANKER_IDLE_SECONDS) if period]
        if not periods:
            return
        interval = min(60.0, max(1.0, min(periods) / 4))
        while not self._stopped.wait(interval):
            for retire in (self._retire_idle_fast_encoder, self._retire_parked_rerankers, self._repromote_embed):
                try:
                    retire()
                except Exception:
                    log.warning("Idle model release failed; retrying later", exc_info=True)

    def _publish_embed_state(self, state: _EmbedState | None) -> None:
        """Caller owns the main state lock; retire without waiting for fast work."""
        with self._residency_lock:
            retired = self._embed_state
            self._embed_state = state
        del retired
        self._retire_stale_fast_encoder()

    def _discard_models(self) -> None:
        """Free dropped models (reference cycles) and their accelerator cache.

        Caller owns the main inference lock, so this also empties a cache whose
        flush was deferred.
        """
        self._mps_flush_pending = False
        from truememory.mps_utils import flush_mps_cache
        flush_mps_cache()

    @staticmethod
    def _collect_models() -> None:
        """Free dropped CPU models without touching the accelerator.

        Emptying the MPS cache while another thread commits MPS work aborts
        the process, so paths outside the inference lock only collect.
        """
        import gc as collector
        collector.collect()

    def _construct_model(
        self, kind: str, stage: str, device: str | None, deadline: _RequestDeadline | None,
        build: Callable[[str | None], object], *, model_identity: tuple[str, int, str] | None = None,
        known_bytes: int = 0, flush_accelerator: bool = True, sticky: str | None = None,
        mps_bytes: int = 0, make_room: Callable[[int], bool] | None = None, cpu_fallback: bool = True,
    ) -> tuple[object, str | None]:
        """Build within the memory budget on *device*, else on CPU.

        An MPS build is admitted first: its *mps_bytes* plus room for one
        forward pass must fit beside the resident models, after *make_room*
        frees what it may; otherwise the model is placed on CPU. An MPS OOM
        while building marks *sticky* (default *kind*) CPU, as an inference
        OOM does (issue #577), so a model that cannot fit is not rebuilt and
        refused on every request. Without *cpu_fallback* either case raises
        ``_MpsBuildRefused`` instead. Callers that do not own the main
        inference lock pass ``flush_accelerator=False``.

        A finished build is returned even when *deadline* passed meanwhile:
        the deadline belongs to the request, the model to the cache.
        """
        discard = self._discard_models if flush_accelerator else self._collect_models
        # Dropped models are reference cycles; free them before admitting this one.
        discard()
        selected = device
        while True:
            resident = selected
            if resident is None:
                from truememory.mps_utils import auto_detect_device
                resident = auto_detect_device()
            label = model_identity[0] if model_identity else stage
            if resident == "mps" and mps_bytes and not self._mps_admits(label, mps_bytes, make_room):
                if not cpu_fallback:
                    raise _MpsBuildRefused(f"no MPS room for {label}", oom=False)
                selected = resident = "cpu"
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
                        self._check_process_memory(stage + " completion")
                    except BaseException:
                        model = None
                        discard()
                        raise
                    self._release_stage(stage)
                    return model, selected
            # Outside the handler, so the failed build's frames are released.
            if not cpu_fallback:
                self._discard_models()
                raise _MpsBuildRefused(f"MPS ran out of memory loading {label}", oom=True)
            self._mark_sticky_cpu(sticky or kind)
            self._discard_models()
            selected = "cpu"
            self._check_process_memory(stage, deadline)

    def _mps_admits(self, label: str, mps_bytes: int, make_room: Callable[[int], bool] | None) -> bool:
        """Whether *mps_bytes* and room for inference fit the MPS budget now.

        Caller owns the main inference lock: no other MPS work is running.
        Before the first MPS model configures the budget nothing is measured,
        and an unmeasured build still falls back to CPU on an OOM.
        """
        from truememory import mps_utils
        measure = getattr(mps_utils, "mps_free_bytes", None)
        free = measure() if callable(measure) else None
        if free is None:
            return True
        need = mps_bytes + mps_utils.MPS_ACTIVATION_RESERVE_BYTES
        if need > free and make_room is not None and make_room(need - free):
            self._discard_models()
            free = measure()
            if free is None:
                return True
        if need <= free:
            return True
        log.info("%s placed on CPU: it needs %.2f GiB of MPS with room for inference, %.2f GiB is free",
                 label, need / 1024**3, max(free, 0) / 1024**3)
        return False

    def _parked_reranker_table(self) -> dict[str, tuple[object, str | None, float]]:
        table = getattr(self, "_parked_rerankers", None)
        if table is None:
            table = self._parked_rerankers = {}
        return table

    def _evict_parked_rerankers(self, shortfall: int, *, min_idle: float) -> bool:
        """Drop parked MPS rerankers, least recently used first, if that frees *shortfall*."""
        now = time.monotonic()
        candidates = sorted(
            (since, name) for name, (_model, device, since) in self._parked_reranker_table().items()
            if device != "cpu" and now - since >= min_idle and name in self._MPS_WEIGHT_BYTES
        )
        if sum(self._MPS_WEIGHT_BYTES[name] for _, name in candidates) < shortfall:
            return False
        freed = 0
        for _, name in candidates:
            self._drop_parked_reranker(name)
            freed += self._MPS_WEIGHT_BYTES[name]
            log.info("Released idle reranker %s from MPS for a new model", name)
            if freed >= shortfall:
                break
        return True

    def _drop_parked_reranker(self, name: str) -> None:
        """Forget a parked reranker; a CPU placement is retried on MPS at its next load."""
        self._parked_reranker_table().pop(name, None)
        key = f"rerank:{name}"
        if key in self._sticky_cpu:
            self._sticky_cpu.discard(key)
            self._write_status_file()

    def _retire_parked_rerankers(self, now: float | None = None) -> bool:
        """Retirer tick: release idle rerankers; the state lock is enough."""
        now = time.monotonic() if now is None else now
        if not self._lock.acquire(blocking=False):
            return False
        try:
            owner = self._inference_lock.acquire(blocking=False)
            try:
                return bool(self._release_idle_rerankers_locked(now, owns_inference=owner))
            finally:
                if owner:
                    self._inference_lock.release()
        finally:
            self._lock.release()

    def _release_idle_rerankers_locked(self, now: float, *, owns_inference: bool) -> list[str]:
        """Drop rerankers parked, or current but unrequested, past their idle period.

        Caller holds the state lock, which is all dropping needs, so a batch
        on another model does not hold the release back. Emptying the MPS
        cache needs inference ownership; without it the flush waits for the
        next inference owner. A CPU model needs no flush.
        """
        table = self._parked_reranker_table()
        released = [name for name, entry in table.items() if now >= entry[2] + self._PARKED_RERANKER_IDLE_SECONDS]
        accelerator = any(table[name][1] != "cpu" for name in released)
        for name in released:
            self._drop_parked_reranker(name)
        name = self._reranker_name
        if (self._reranker is not None and name is not None
                and now >= getattr(self, "_reranker_last_used", now) + self._RERANKER_IDLE_SECONDS):
            accelerator = accelerator or getattr(self, "_reranker_device", None) != "cpu"
            self._reranker = None
            self._reranker_name = None
            self._reranker_device = None
            self._drop_parked_reranker(name)  # its CPU placement is retried on MPS
            released.append(name)
        flush = accelerator or getattr(self, "_mps_flush_pending", False)
        if owns_inference and flush:
            self._discard_models()
        elif flush:
            self._mps_flush_pending = True
        if released and not (owns_inference and flush):
            # The models are reference cycles; free them now.
            self._collect_models()
        if released:
            log.info("Released idle rerankers: %s", ", ".join(released))
        return released

    def _before_inference_locked(
        self, deadline: _RequestDeadline, *, swap_embedding: bool = False, reranking: bool = False,
    ) -> None:
        """Caller owns the inference and state locks before its first piece.

        Idle rerankers are released and a deferred MPS cache emptied here, so
        continuous batches do not hold them back. An embed batch first swaps a
        due CPU embedding model to MPS; a single query never pays that load
        and leaves it to the next batch or retirer tick.
        """
        now = time.monotonic()
        if reranking:
            self._reranker_last_used = now  # this request is using the current reranker
        self._release_idle_rerankers_locked(now, owns_inference=True)
        if swap_embedding:
            self._repromote_embed_locked(deadline)

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
                mps_bytes=self._MPS_WEIGHT_BYTES.get(model_id, 0),
                make_room=lambda shortfall: self._evict_parked_rerankers(shortfall, min_idle=0),
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
                mps_bytes=self._MPS_WEIGHT_BYTES.get(target.model_id, 0),
                make_room=lambda shortfall: self._evict_parked_rerankers(shortfall, min_idle=0),
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
        lock busy, then cached for that main residency until it goes unused
        for ``TRUEMEMORY_FAST_LANE_IDLE_SECONDS``. A released residency reloads
        in the background and raises ``_FastLaneNotReady`` meanwhile. Caller
        must hold ``self._fast_lock`` through inference and retire stale cache
        on release.

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
            # During its return to MPS the CPU main model itself serves here.
            lent = self._fast_encoder is not None and self._fast_encoder is state.model
            if self._duplicates_cpu_main(state) and not lent:
                # A CPU main model serves single queries itself; a second CPU
                # copy would only duplicate 2.4 GB of identical weights.
                state = None
                self._drop_redundant_fast_encoder_locked()
                raise _ProcessMemoryRefused("Model server busy: the main model serves single queries on CPU")
            if getattr(self, "_small_host", False) and not lent:
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
            if self._fast_retired_generation is generation:
                # This residency's clone was released while idle. Reload it in
                # the background; this query takes the bounded fallback meanwhile.
                self._start_fast_rebuild(model_id, generation)
                raise _FastLaneNotReady("Model server busy: the fast-lane copy is reloading")

            self._fast_encoder = None
            self._fast_model_id = None
            self._fast_generation = None
            model = self._build_fast_encoder(model_id, deadline)
            self._publish_fast_encoder(model, model_id, generation, blocking=False)
            log.info(
                "Fast-lane CPU encoder loaded (tier=%s, model=%s) — single-text "
                "requests no longer queue behind batch work", tier, model_id,
            )
            return model

    # A clone of a CPU main model is kept only when it is this small (model2vec);
    # the float32 Qwen copy duplicates 2.4 GB.
    _CLONE_DUPLICATE_LIMIT_BYTES = 256 * 1024**2

    def _duplicates_cpu_main(self, state: _EmbedState) -> bool:
        return (_on_cpu(state.model)
                and self._MODEL_WEIGHT_BYTES.get(state.model_id, 0) > self._CLONE_DUPLICATE_LIMIT_BYTES)

    def _release_fast_clone(self) -> None:
        """Drop the clone unless a query is using it; that query's owner drops it."""
        if not self._fast_lock.acquire(blocking=False):
            return
        try:
            self._drop_redundant_fast_encoder_locked()
        finally:
            self._fast_lock.release()

    def _drop_redundant_fast_encoder_locked(self) -> None:
        """Caller owns ``_fast_lock``: drop a clone that duplicates a CPU main model."""
        with self._residency_lock:
            retired = self._fast_encoder
            self._fast_encoder = None
            self._fast_model_id = None
            self._fast_generation = None
        if retired is not None:
            retired = None
            self._collect_models()
            log.info("Fast-lane CPU encoder released: the main embedding model is on CPU")

    def _build_fast_encoder(self, model_id: str, deadline: _RequestDeadline | None) -> object:
        """Construct a CPU clone; the caller's allocation session charges its size."""
        self._check_process_memory("fast embed construction", deadline)
        model, _ = self._construct_model(
            "embed", "fast embed construction", "cpu", deadline,
            lambda selected: self._build_embed_model(model_id, selected),
            model_identity=(model_id, 256, "legacy"), known_bytes=self._MODEL_WEIGHT_BYTES.get(model_id, 0),
            flush_accelerator=False,
        )
        return model

    def _publish_fast_encoder(self, model: object, model_id: str, generation: object, *, blocking: bool) -> bool:
        """Cache a clone for its residency; the caller owns ``_fast_lock``.

        A changed generation, including A -> B -> A, must not publish this
        construction. Its already-captured request may still finish on it.
        """
        if not self._residency_lock.acquire(blocking=blocking):
            return False
        try:
            state = self._embed_state
            if state is None or state.generation is not generation:
                return False
            self._fast_encoder = model
            self._fast_model_id = model_id
            self._fast_generation = generation
            self._fast_last_used = time.monotonic()
            if self._fast_retired_generation is generation:
                self._fast_retired_generation = None
            return True
        finally:
            self._residency_lock.release()
            state = None

    def _start_fast_rebuild(self, model_id: str, generation: object) -> None:
        """Start at most one background reload of a released clone.

        After a refused reload, contended queries do not retry the admission
        and its collections for a minute, unless resident memory has since
        dropped by the clone's size.
        """
        refused_until, refused_rss = self._fast_rebuild_refusal
        if time.monotonic() < refused_until:
            usage = self._resident_bytes()
            room = self._MODEL_WEIGHT_BYTES.get(model_id, 0)
            if usage is None or refused_rss is None or refused_rss - usage < room:
                return
        with self._residency_lock:
            if self._fast_rebuild is not None or self._stopped.is_set():
                return
            thread = threading.Thread(target=self._rebuild_fast_encoder, args=(model_id, generation),
                                      name="fast-lane-rebuild", daemon=True)
            self._fast_rebuild = thread
        # Counted as in-flight work: idle shutdown and cleanup wait for the load.
        with self._activity_lock:
            self._inflight += 1
        try:
            thread.start()
        except RuntimeError:
            with self._residency_lock:
                self._fast_rebuild = None
            with self._activity_lock:
                self._inflight -= 1
            log.warning("Cannot start the fast-lane reload; single queries keep the bounded fallback",
                        exc_info=True)

    def _rebuild_fast_encoder(self, model_id: str, generation: object) -> None:
        # No request owns this load; only server shutdown stops it.
        deadline = _RequestDeadline(None, _TransportRequest(0, math.inf, self._stopped, started=True))
        model = None
        built = published = False
        try:
            with self._memory_scope():
                model = self._build_fast_encoder(model_id, deadline)
                built = True
            with deadline.locked(self._fast_lock):
                published = self._publish_fast_encoder(model, model_id, generation, blocking=True)
            if published:
                log.info("Fast-lane CPU encoder reloaded in the background (model=%s)", model_id)
        except _ProcessMemoryRefused as error:
            self._fast_rebuild_refusal = (time.monotonic() + self._FAST_REBUILD_BACKOFF_SECONDS,
                                          self._resident_bytes())
            log.info("Fast-lane reload not admitted: %s", error)
        except _RequestDeadlineExceeded as error:
            log.info("Fast-lane reload not admitted: %s", error)
        except Exception:
            log.warning("Fast-lane reload failed; single queries keep the bounded fallback", exc_info=True)
        finally:
            model = None
            if built and not published:
                self._collect_models()
            with self._residency_lock:
                self._fast_rebuild = None
            with self._activity_lock:
                self._inflight -= 1

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

            # Park the previous reranker instead of dropping it, so switching
            # between the default and the deep-search reranker reloads neither.
            parked_rerankers = self._parked_reranker_table()
            previous, previous_name = self._reranker, self._reranker_name
            self._reranker = None
            self._reranker_name = None
            if previous is not None and previous_name is not None:
                parked_rerankers[previous_name] = (previous, getattr(self, "_reranker_device", None),
                                                   time.monotonic())
            previous = None
            self._reranker_device = None
            parked = parked_rerankers.pop(name, None)
            if parked is not None:
                self._reranker, self._reranker_device = parked[0], parked[1]
                self._reranker_name = name
                return self._reranker

            from truememory.mps_utils import auto_detect_device, ensure_mps_memory_budget, resolve_device
            sticky = f"rerank:{name}"

            def build(selected: str | None) -> object:
                ensure_mps_memory_budget(selected)
                from sentence_transformers import CrossEncoder
                if deadline is not None:
                    deadline.check()
                model = CrossEncoder(name, device=selected)
                from truememory._rerank_budget import cap_reranker_length
                cap_reranker_length(model)
                return model

            def construct() -> tuple[object, str | None]:
                self._check_process_memory("reranker construction", deadline)
                device = "cpu" if sticky in self._sticky_cpu else resolve_device(auto_detect_device())
                return self._construct_model(
                    "rerank", "reranker construction", device, deadline, build,
                    model_identity=(name, 0, "reranker"), known_bytes=self._MODEL_WEIGHT_BYTES.get(name, 0),
                    sticky=sticky, mps_bytes=self._MPS_WEIGHT_BYTES.get(name, 0),
                    make_room=lambda shortfall: self._evict_parked_rerankers(
                        shortfall, min_idle=self._RERANKER_EVICT_IDLE_SECONDS),
                )

            try:
                model, device = construct()
            except _ProcessMemoryRefused:
                if not parked_rerankers:
                    raise
                # Parked rerankers hold memory; release them and admit once more.
                for parked_name in list(parked_rerankers):
                    self._drop_parked_reranker(parked_name)
                self._discard_models()
                model, device = construct()
            self._reranker = model
            self._reranker_name = name
            self._reranker_device = device
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
        *, expected_target: "EmbeddingTarget | None" = None, serve_reload: bool = True,
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

        # Contention: this query found the main model busy. It keeps the clone.
        self._fast_last_used = time.monotonic()
        was_started = deadline.transport.started if deadline.transport is not None else False
        fast_owned = False
        wait_for_main = False
        reloading = False
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
                        except _ProcessMemoryRefused as error:
                            # The second copy does not fit or is reloading; a process
                            # over budget stays refused.
                            reloading = serve_reload and getattr(error, "fast_lane_reloading", False)
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
                ends = time.monotonic() + wait
                while not self._inference_lock.acquire(timeout=min(0.05, wait) if reloading else wait):
                    state = self._embed_state
                    if (reloading and state is not None and self._fast_encoder is not None
                            and self._fast_generation is state.generation):
                        # The reload finished within the same bounded wait; serve from it once.
                        state = None
                        return self._handle_fast_embed(texts, tier, deadline, expected_target=expected_target,
                                                       serve_reload=False)
                    state = None
                    wait = ends - time.monotonic()
                    if wait <= 0:
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
                                self._before_inference_locked(
                                    deadline, swap_embedding=len(texts) > self._FAST_LANE_MAX_TEXTS)
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
                    steps = None
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
                                self._before_inference_locked(deadline, reranking=True)
                                order = self._rerank_global_order(reranker, pairs, limit, deadline)
                                if order is not None:
                                    recovery_name = self._reranker_name
                                slices = None
                                if getattr(reranker, "tokenizer", None) is not None and (
                                        type(getattr(reranker, "max_seq_length", None)) is int
                                        or hasattr(reranker, "max_length")):
                                    from truememory._rerank_budget import (
                                        budgeted_slices, pair_token_lengths, rerank_token_budget,
                                    )
                                    lengths = pair_token_lengths(reranker, pairs)
                                    slices = (None if lengths is None else
                                              budgeted_slices(lengths, limit, rerank_token_budget()))
                                if slices is not None:
                                    # Long pairs: smaller longest-first batches bound attention scratch.
                                    order = np.array([index for group in slices for index in group], dtype=np.intp)
                                    steps, start = {}, 0
                                    for group in slices:
                                        steps[start] = len(group)
                                        start += len(group)
                                    recovery_name = self._reranker_name
                            while offset < len(pairs) or scores is None:
                                deadline.check()
                                step = limit if steps is None else steps[offset]
                                if steps is not None:
                                    indices = order[offset:offset + step]
                                else:
                                    indices = None if order is None else self._rerank_slice_indices(
                                        reranker, pairs, order, offset, step, deadline,
                                    )
                                batch = (pairs[offset:offset + step] if indices is None else
                                         [pairs[index] for index in indices])
                                self._check_process_memory("reranker slice", deadline)
                                deadline.check()
                                recover = False
                                try:
                                    values = reranker.predict(batch, batch_size=step, show_progress_bar=False)
                                except RuntimeError as exc:
                                    from truememory.mps_utils import is_mps_oom, flush_mps_cache
                                    if not is_mps_oom(exc):
                                        raise
                                    recover = True
                                if recover:
                                    # Only this reranker moves to CPU; parked ones keep MPS.
                                    failed_name = self._reranker_name or recovery_name
                                    if failed_name is None:
                                        from truememory.reranker import get_current_reranker_name
                                        failed_name = get_current_reranker_name()
                                    self._mark_sticky_cpu(f"rerank:{failed_name}")
                                    self._reranker = None
                                    self._reranker_name = None
                                    self._reranker_device = None
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
                            values = reranker.predict(retry, batch_size=len(retry) or limit, show_progress_bar=False)
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
            release_file_lock(self._lock_fd)
            self._lock_fd = None
            raise RuntimeError("Cannot verify model-server launch generation") from error
        if claim is None or claim["generation"] != generation:
            release_file_lock(self._lock_fd)
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
        threading.Thread(target=self._idle_retirer, name="idle-model-retirer", daemon=True).start()

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
        self._release_resources()

    def _finish_after_signal(self, grace: float | None = None) -> None:
        """Exit promptly after a stop signal, abandoning native work past *grace*.

        Admission stops at once (``_cleanup`` already ran). Requests stop at
        their next slice boundary; a native forward pass cannot be interrupted,
        so after *grace* seconds the process exits without waiting for it. The
        next server reclaims the endpoint files under its bind lock.
        """
        grace = _TERM_GRACE_SECONDS if grace is None else grace
        self._stop_clients()
        expires = time.monotonic() + grace
        while True:
            with self._admission_lock:
                with self._activity_lock:
                    busy = bool(self._clients) or self._inflight > 0
            if not busy:
                self._release_resources()
                return
            if time.monotonic() >= expires:
                break
            time.sleep(0.05)
        log.warning("Model work still running %.0fs after the stop signal; exiting without it", grace)
        logging.shutdown()
        os._exit(0)

    def _release_resources(self) -> None:
        """Drop models, then endpoint artifacts and the bind lock, once."""
        if getattr(self, "_resources_released", False):
            return
        self._resources_released = True
        # Keep bind ownership until every cached model has been released,
        # including cycles collected below. If teardown fails, retain the
        # lock until process exit instead of overlapping a successor's load.
        self._embed_state = None
        self._reranker = None
        self._reranker_name = None
        self._parked_reranker_table().clear()
        self._fast_encoder = None
        self._fast_model_id = None
        self._fast_generation = None
        self._fast_retired_generation = None
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
                release_file_lock(lock_fd)
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
        try:
            server.run()
        except (SystemExit, KeyboardInterrupt):
            # TERM/INT/HUP: never wait unboundedly for a native forward pass.
            server._finish_after_signal()
            raise
    except RuntimeError as e:
        # Lost the bind-lock race to a concurrent starter (M-20). Exit
        # cleanly without touching the winner's artifacts.
        log.error("Model server start aborted: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
