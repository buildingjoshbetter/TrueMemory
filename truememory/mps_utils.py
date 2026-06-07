"""Shared MPS OOM detection and fallback utilities.

Consolidates MPS out-of-memory handling into a single module, replacing
inconsistent string-matching logic previously duplicated across
vector_search.py and model_server.py.
"""
from __future__ import annotations

import gc
import logging
import threading

logger = logging.getLogger(__name__)

_device_lock = threading.Lock()


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
