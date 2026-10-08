"""Shared MPS OOM detection and fallback utilities.

Consolidates MPS out-of-memory handling into a single module, replacing
inconsistent string-matching logic previously duplicated across
vector_search.py and model_server.py.
"""
from __future__ import annotations

import gc
import logging
import math
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

_GIB = 1024**3
_MPS_HIGH_ENV = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
_MPS_LOW_ENV = "PYTORCH_MPS_LOW_WATERMARK_RATIO"
_mps_budget_lock = threading.Lock()


class MPSBudgetError(RuntimeError):
    """MPS model loading cannot proceed with an unverified allocation policy."""


@dataclass(frozen=True)
class MPSMemoryBudget:
    """HIGH setter result; LOW records bootstrap intent, not verified native state."""

    intended_bytes: int | None
    recommended_bytes: int
    effective_bytes: int | None
    fraction: float
    requested_low_watermark_ratio: float
    source: str
    enforcement: str


_mps_budget: MPSMemoryBudget | None = None
_mps_budget_failure: str | None = None


def get_mps_memory_budget() -> MPSMemoryBudget | None:
    """Read this process's successfully configured policy without importing torch."""
    return _mps_budget


def _positive_bytes(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MPSBudgetError(f"{name} must report a positive integer byte count")
    return value


def _default_mps_budget_bytes(physical_bytes: int) -> int:
    physical_bytes = _positive_bytes(physical_bytes, "Physical memory")
    target = min(0.08 * physical_bytes, 2.5 * _GIB) if physical_bytes >= 16 * _GIB else 0.19 * physical_bytes
    return int(max(target, 1.5 * _GIB))


def _watermark_ratio(name: str, value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as exc:
        raise MPSBudgetError(f"{name} must be a finite ratio between 0 and 2") from exc
    if not math.isfinite(ratio) or not 0 <= ratio <= 2:
        raise MPSBudgetError(f"{name} must be a finite ratio between 0 and 2")
    return ratio


def ensure_mps_memory_budget(device: str | None) -> MPSMemoryBudget | None:
    """Configure the allocator once, before constructing this process's MPS models.

    ``None`` preserves framework auto-selection at the constructor; detection
    here only determines whether MPS calibration is needed. CPU/CUDA callers
    do not touch the MPS allocator or its environment settings.
    """
    selected = auto_detect_device() if device is None else str(device)
    if selected.split(":", 1)[0] != "mps":
        return None

    global _mps_budget, _mps_budget_failure
    with _mps_budget_lock:
        if _mps_budget is not None:
            return _mps_budget
        if _mps_budget_failure is not None:
            raise MPSBudgetError(_mps_budget_failure)
        try:
            high_text = os.environ.get(_MPS_HIGH_ENV)
            high = _watermark_ratio(_MPS_HIGH_ENV, high_text) if high_text is not None else None
            low = _watermark_ratio(_MPS_LOW_ENV, os.environ.get(_MPS_LOW_ENV, "0.0"))
            bootstrap_high = high if high is not None else 1.7
            if low > (2.0 if bootstrap_high == 0 else bootstrap_high):
                raise MPSBudgetError("MPS low watermark exceeds the initial high watermark")

            # recommended_max_memory() itself initializes the allocator. Keep
            # the existing low=0 policy before that query; no public low setter
            # exists. Explicit operator values always take precedence. An
            # embedding host may already have initialized LOW; this setting
            # records bootstrap intent and cannot verify that native state.
            os.environ.setdefault(_MPS_LOW_ENV, "0.0")
            import torch

            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise MPSBudgetError("MPS is unavailable for allocation-budget calibration")
            recommended = getattr(torch.mps, "recommended_max_memory", None)
            setter = getattr(torch.mps, "set_per_process_memory_fraction", None)
            if not callable(recommended) or not callable(setter):
                raise MPSBudgetError(
                    "PyTorch lacks the public MPS allocation-budget APIs; MPS requires "
                    "PyTorch>=2.5. Upgrade to a compatible build if available"
                )

            intended = None
            if high is None:
                import psutil
                intended = _default_mps_budget_bytes(psutil.virtual_memory().total)
            recommended_bytes = _positive_bytes(recommended(), "MPS recommended memory")
            if high is not None:
                fraction = high
            else:
                assert intended is not None
                fraction = intended / recommended_bytes
            if not math.isfinite(fraction) or not 0 <= fraction <= 2 or (fraction == 0 and high is None):
                raise MPSBudgetError("Intended MPS byte budget is outside PyTorch's supported ratio range")
            if fraction != 0 and low > fraction:
                raise MPSBudgetError("MPS low watermark exceeds the calibrated high watermark")
            effective = int(fraction * recommended_bytes) if fraction else None
            if fraction and not effective:
                raise MPSBudgetError("MPS allocation budget rounds down to zero bytes")
            if high is not None:
                intended = effective
            budget = MPSMemoryBudget(
                intended, recommended_bytes, effective, fraction, low,
                "operator" if high is not None else "automatic",
                "enforced" if fraction else "disabled",
            )
            setter(float(fraction))
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
            detail = str(exc) if isinstance(exc, MPSBudgetError) else type(exc).__name__
            _mps_budget_failure = (
                "Cannot configure MPS allocation budget: " + detail
                + ". Correct the policy or select TRUEMEMORY_DEVICE=cpu, then restart this process."
            )
            raise MPSBudgetError(_mps_budget_failure) from exc

        # A partially initialized allocator is never advertised as configured.
        _mps_budget = budget
        logger.info("MPS allocation budget: source=%s intended_bytes=%s recommended_bytes=%d effective_bytes=%s fraction=%.8g enforcement=%s",
                    budget.source, budget.intended_bytes, budget.recommended_bytes,
                    budget.effective_bytes, budget.fraction, budget.enforcement)
        return budget


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
        if get_mps_memory_budget() is not None and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
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
