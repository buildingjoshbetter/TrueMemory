"""Shared MPS OOM detection and fallback utilities.

Consolidates MPS out-of-memory handling into a single module, replacing
inconsistent string-matching logic previously duplicated across
vector_search.py and model_server.py.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
from _thread import LockType
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_device_lock = threading.Lock()


@dataclass
class _ModelOwnership:
    lock: LockType = field(default_factory=threading.Lock)
    users: int = 0


# Entries live only while a caller owns or waits for that exact instance. The
# caller keeps the model alive, so its id cannot be reused during this interval.
_model_owners: dict[int, _ModelOwnership] = {}


@contextmanager
def _own_model(model: object) -> Iterator[None]:
    key = id(model)
    with _device_lock:
        owner = _model_owners.get(key)
        if owner is None:
            owner = _ModelOwnership()
            _model_owners[key] = owner
        owner.users += 1
    try:
        with owner.lock:
            yield
    finally:
        with _device_lock:
            owner.users -= 1
            if owner.users == 0:
                del _model_owners[key]


_VALID_DEVICE_VALUES = ("cpu", "mps", "cuda", "auto")


def auto_detect_device() -> str:
    """Auto-detect the best torch device: cuda -> mps -> cpu.

    This is the detection order every load site used before issue #577;
    call sites that want explicit detection pass this as ``default_auto``
    to :func:`resolve_device`.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def resolve_device(default_auto: str | None = None) -> str | None:
    """Resolve the inference device, honoring ``TRUEMEMORY_DEVICE`` (issue #577).

    ``TRUEMEMORY_DEVICE`` accepts ``cpu`` | ``mps`` | ``cuda`` | ``auto``:

    * ``cpu`` — always honored (the escape hatch for MPS OOM retry storms
      on memory-constrained Macs).
    * ``mps`` / ``cuda`` — honored when the accelerator is actually
      available; otherwise a warning is logged and resolution falls back
      to *default_auto*.
    * ``auto``, unset — *default_auto*.
    * anything else — warning + *default_auto*.

    *default_auto* is the call site's auto behavior: pass
    ``auto_detect_device()`` to keep explicit cuda→mps→cpu detection, or
    ``None`` to let the framework (e.g. SentenceTransformer) pick its own
    device.
    """
    raw = os.environ.get("TRUEMEMORY_DEVICE", "").strip().lower()
    if not raw or raw == "auto":
        return default_auto
    if raw not in _VALID_DEVICE_VALUES:
        logger.warning(
            "Invalid TRUEMEMORY_DEVICE=%r (expected cpu|mps|cuda|auto) — "
            "falling back to auto device selection.",
            raw,
        )
        return default_auto
    if raw == "cpu":
        return "cpu"
    try:
        import torch
        if raw == "cuda" and torch.cuda.is_available():
            return "cuda:0"
        if (
            raw == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"
    except ImportError:
        pass
    logger.warning(
        "TRUEMEMORY_DEVICE=%s requested but that device is not available — "
        "falling back to auto device selection.",
        raw,
    )
    return default_auto


def is_mps_oom(exc: Exception) -> bool:
    """Return True if the exception is an MPS out-of-memory error."""
    msg = str(exc)
    msg_lower = msg.lower()
    return (
        ("mps" in msg_lower and "out of memory" in msg_lower)
        or "mps backend out of memory" in msg_lower
        or ("mps" in msg_lower and "allocated" in msg_lower and "exceed" in msg_lower)
    )


def flush_mps_cache() -> None:
    """Flush MPS/CUDA cache and run garbage collection."""
    try:
        import torch
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def encode_with_model_ownership(model: object, texts: object, **kwargs: object) -> object:
    """Serialize local inference while preserving the caller's recovery policy."""
    from truememory.model_client import EmbeddingProxy

    if isinstance(model, EmbeddingProxy):
        return model.encode(texts, **kwargs)
    with _own_model(model):
        return model.encode(texts, **kwargs)


def encode_with_mps_fallback(model: object, texts: object, **kwargs: object) -> object:
    """Own a local model through inference and recovery; keep recovered models on CPU."""
    from truememory.model_client import EmbeddingProxy

    if isinstance(model, EmbeddingProxy):
        # The daemon owns native inference. Serializing its proxy here would
        # prevent recall from using the daemon's independent fast lane.
        return model.encode(texts, **kwargs)

    with _own_model(model):
        try:
            return model.encode(texts, **kwargs)
        except RuntimeError as exc:
            if not is_mps_oom(exc):
                raise
            logger.warning("MPS OOM during local encoding; retrying on CPU without MPS re-promotion")
        # Leave the exception block first so the failed forward traceback can
        # release its tensors. Move live weights before flushing freed storage.
        if hasattr(model, "to"):
            model.to("cpu")
        flush_mps_cache()
        return model.encode(texts, **kwargs)
