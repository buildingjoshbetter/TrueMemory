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

logger = logging.getLogger(__name__)

_device_lock = threading.Lock()

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


def encode_with_mps_fallback(model, texts, **kwargs):
    """Encode texts, falling back to CPU if MPS runs out of memory.

    Thread-safe: acquires _device_lock around model.to() transitions.
    Restores model to MPS after successful CPU fallback.
    """
    try:
        return model.encode(texts, **kwargs)
    except RuntimeError as exc:
        if not is_mps_oom(exc):
            raise
        logger.warning("MPS OOM during encoding — flushing cache and retrying on CPU")
        flush_mps_cache()
        if hasattr(model, "to"):
            with _device_lock:
                model.to("cpu")
            try:
                result = model.encode(texts, **kwargs)
            finally:
                try:
                    import torch
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        with _device_lock:
                            model.to("mps")
                except Exception:
                    pass
            return result
        return model.encode(texts, **kwargs)
