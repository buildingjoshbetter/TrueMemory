"""
Structured telemetry signals.

Each signal function formats metadata-only data (no raw content, no query
text, no PII) and delegates to ``writer.emit()``.

Privacy contract: telemetry rows must contain ONLY:
- Method names, durations, result counts
- Numeric scores (salience, novelty, prediction_error, cosine distances)
- Reason codes (machine-readable strings)
- Memory IDs (integers)
- Category labels

NEVER store: raw memory content, search query text, user_id values,
content hashes, or any PII.
"""

from __future__ import annotations

from truememory.instrumentation import writer


def emit_timing(method: str, duration_ms: float, n_results: int = 0,
                extra: dict | None = None) -> None:
    """Emit a method timing signal."""
    data = {
        "method": method,
        "duration_ms": round(duration_ms, 1),
        "n_results": n_results,
    }
    if extra:
        data.update(extra)
    writer.emit("timing", data)


def emit_gate_decision(
    should_encode: bool,
    encoding_score: float,
    novelty: float,
    salience: float,
    prediction_error: float,
    reason_code: str,
    category: str = "",
) -> None:
    """Emit an encoding gate decision signal."""
    writer.emit("gate_decision", {
        "should_encode": should_encode,
        "encoding_score": round(encoding_score, 4),
        "novelty": round(novelty, 4),
        "salience": round(salience, 4),
        "prediction_error": round(prediction_error, 4),
        "reason_code": reason_code,
        "category": category,
    })


def emit_search_distance(
    spread: float,
    top_score: float,
    min_score: float,
    mean_score: float,
    n_results: int,
) -> None:
    """Emit search quality metrics (no query text stored)."""
    writer.emit("search_distance", {
        "spread": round(spread, 4),
        "top_score": round(top_score, 4),
        "min_score": round(min_score, 4),
        "mean_score": round(mean_score, 4),
        "n_results": n_results,
    })


def emit_memory_returned(memory_id: int, rank: int) -> None:
    """Emit a memory-returned signal (tracks which memories surface)."""
    writer.emit("memory_returned", {
        "memory_id": memory_id,
        "rank": rank,
    })


def emit_surprise(score: float) -> None:
    """Emit native prediction_error as the surprise telemetry signal."""
    writer.emit("surprise", {
        "score": round(score, 4),
    })


def emit_category(category: str) -> None:
    """Emit the category of an added memory."""
    writer.emit("category", {
        "category": category,
    })


def emit_salience(score: float) -> None:
    """Emit a salience score signal."""
    writer.emit("salience", {
        "score": round(score, 4),
    })


def emit_model_lifecycle(event: str, model_name: str = "",
                         duration_ms: float = 0.0) -> None:
    """Emit model load/unload events."""
    writer.emit("model_lifecycle", {
        "event": event,
        "model_name": model_name,
        "duration_ms": round(duration_ms, 1),
    })
