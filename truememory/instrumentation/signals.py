"""Telemetry signal emitters + auxiliary diagnostic helpers.

Every ``emit_*`` here is gated on ``is_enabled()`` (so it is a no-op when the
overlay is disabled) and writes one row to the ``telemetry`` table via
``writer.emit``. The monkey-patches in ``patch.py`` call these from inside the
wrapped engine / MCP-server methods.

Two groups:

1. **Model-lifecycle signals** — ``preload_start``, ``preload_complete``,
   ``model_unload``, ``reranker_degraded``. Render a per-day model lifecycle
   view (preload → ready → degraded → unloaded → re-preload).
2. **Per-memory semantic signals** — ``salience``, ``category``,
   ``gate_decision``, ``surprise``, ``search_distance``, ``memory_returned``,
   ``user_forget``. These power the dashboard's data lanes.

Auxiliary helpers (cold/warm model flag, WAL-checkpoint snapshot, HF Hub
network intercept, spawner classification) are diagnostic-only — they write to
the instrumentation log via ``dlog``, not the telemetry table.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from truememory.instrumentation.log import dlog, is_enabled
from truememory.instrumentation import writer

_model_warm = False
_model_warm_lock = threading.Lock()


def is_model_cold() -> bool:
    return not _model_warm


def mark_model_warm() -> None:
    global _model_warm
    with _model_warm_lock:
        _model_warm = True


def log_spawner_classification() -> None:
    """Best-effort identification of the spawning process (the MCP host vs a
    sub-process it launched).

    Uses psutil if available; falls back to logging just PIDs. Diagnostic
    only — never written to the telemetry table.
    """
    if not is_enabled():
        return
    pid = os.getpid()
    ppid = os.getppid()
    try:
        import psutil
        try:
            parent = psutil.Process(ppid)
            parent_name = parent.name()
            try:
                grandparent = parent.parent()
                grandparent_name = grandparent.name() if grandparent else "(none)"
                grandparent_pid = grandparent.pid if grandparent else 0
            except Exception:
                grandparent_name = "(unavailable)"
                grandparent_pid = 0
            dlog(
                f"spawner pid={pid} ppid={ppid} parent='{parent_name}' "
                f"gppid={grandparent_pid} grandparent='{grandparent_name}'"
            )
        except psutil.NoSuchProcess:
            dlog(f"spawner pid={pid} ppid={ppid} (parent process gone)")
    except ImportError:
        dlog(f"spawner pid={pid} ppid={ppid} (psutil unavailable)")
    except Exception as exc:
        dlog(f"spawner classification failed: {type(exc).__name__}: {exc}")


def log_wal_checkpoint(conn) -> None:
    """Run PRAGMA wal_checkpoint(PASSIVE) and log result. Called after
    engine.add commit. Diagnostic only."""
    if not is_enabled():
        return
    try:
        cursor = conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        row = cursor.fetchone()
        if row:
            busy, pages_moved, pages_remaining = row[0], row[1], row[2]
            dlog(
                f"wal_checkpoint busy={busy} pages_moved={pages_moved} "
                f"pages_remaining={pages_remaining}"
            )
    except Exception as exc:
        dlog(f"wal_checkpoint failed: {type(exc).__name__}: {exc}")


class _HFHubDiagHandler(logging.Handler):
    """Forwards WARNING+ records from huggingface_hub to dlog.

    Triggers when HF_HUB_OFFLINE=1 is bypassed somehow and the loader tries to
    reach the network. Useful for catching race conditions between preload
    threads and config-mutation calls.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            dlog(f"[hf_hub] {record.levelname}: {msg}")
        except Exception:
            pass


def install_hf_hub_intercept() -> None:
    """Attach the HF Hub diagnostic handler. Idempotent — checks for prior
    install."""
    if not is_enabled():
        return
    try:
        hub_logger = logging.getLogger("huggingface_hub")
        for h in hub_logger.handlers:
            if isinstance(h, _HFHubDiagHandler):
                return
        handler = _HFHubDiagHandler()
        handler.setLevel(logging.WARNING)
        hub_logger.addHandler(handler)
        dlog("hf_hub intercept attached (WARNING+ -> dlog)")
    except Exception as exc:
        dlog(f"hf_hub intercept install failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Model-lifecycle signals (preload / degraded / idle)
# ---------------------------------------------------------------------------
#
# Every emit_* writes BOTH a dlog line (human-readable ops trace) AND a
# telemetry row (structured query surface). Belt + suspenders: a future
# dashboard query can JOIN on memory_id, and when the telemetry table is empty
# the human can still grep the instrumentation log for the same event.


def emit_preload_start() -> None:
    """Fires when ``_preload_models`` begins its work (embedding + reranker
    preload threads spawn)."""
    if not is_enabled():
        return
    dlog("preload_start")
    writer.emit(
        "preload_start",
        value_text="reranker+embedding",
        context={"pid": os.getpid()},
    )


def emit_preload_complete(model: str, load_ms: float) -> None:
    """Fires when ``get_reranker()`` (or the embedding model getter) returns
    from a cold load."""
    if not is_enabled():
        return
    dlog(f"preload_complete model={model!r} load_ms={load_ms:.0f}")
    writer.emit(
        "preload_complete",
        value_num=load_ms,
        value_text=model,
    )


def emit_reranker_degraded(reason: str) -> None:
    """Fires when the reranker degrades — e.g. the shared model-server proxy
    fails and forces an expensive local fallback load."""
    if not is_enabled():
        return
    dlog(f"reranker_degraded reason={reason!r}")
    writer.emit(
        "reranker_degraded",
        value_text=reason,
    )


def emit_model_unload(model: str, idle_seconds: float | None = None) -> None:
    """Fires when the MCP server's idle timer unloads a model to free RAM."""
    if not is_enabled():
        return
    dlog(f"model_unload model={model!r} idle_seconds={idle_seconds}")
    writer.emit(
        "model_unload",
        value_num=idle_seconds,
        value_text=model,
    )


# ---------------------------------------------------------------------------
# Per-memory semantic signals (power the dashboard's data lanes)
# ---------------------------------------------------------------------------


def emit_salience(memory_id: int | None, score: float) -> None:
    """Fires once per candidate fact with the computed salience score [0, 1]."""
    if not is_enabled():
        return
    writer.emit(
        "salience",
        memory_id=memory_id,
        value_num=score,
    )


def emit_category(memory_id: int, category: str) -> None:
    """Fires once per stored memory with the auto-detected category."""
    if not is_enabled():
        return
    writer.emit(
        "category",
        memory_id=memory_id,
        value_text=category,
    )


def emit_gate_decision(
    memory_id: int | None,
    score: float,
    outcome: str,
    *,
    reason_code: str | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Fires per encoding-gate decision (pass OR reject)."""
    if not is_enabled():
        return
    writer.emit(
        "gate_decision",
        memory_id=memory_id,
        value_num=score,
        value_text=reason_code or outcome,
        context=context,
    )


def emit_search_distance(
    query_text: str,
    *,
    spread: float,
    top_score: float,
    min_score: float,
    mean_score: float,
    n_results: int,
) -> None:
    """Fires once per search call with a retrieval-quality summary.

    ``value_num`` holds the score SPREAD (top - bottom) of the returned result
    set — the informative signal. The top result's raw score is pinned to ~1.0
    by the reranker's per-call min-max normalization, so spread, not top, is
    what tracks retrieval confidence: wide spread = the retriever cleanly
    separated the best hit from the tail; narrow spread = a flat, low-confidence
    result set. Spread stays comparable across calls regardless of whether the
    reranked path (normalized ~1.0 scores) or the rerank-skipped path (raw RRF
    ~0.03 scores) produced the result, because it is a difference of two scores
    on the same per-call scale.

    The raw ``top``/``min``/``mean`` absolutes are stashed in context_json so
    the dashboard can still plot them directly when desired.

    query_text is truncated to 60 chars before storage (never log full query
    text — avoids leaking PII to the telemetry table).
    """
    if not is_enabled():
        return
    writer.emit(
        "search_distance",
        value_num=spread,
        context={
            "query_preview": (query_text or "")[:60],
            "n_results": n_results,
            "top": round(top_score, 6),
            "min": round(min_score, 6),
            "mean": round(mean_score, 6),
        },
    )


def emit_memory_returned(memory_id: int, rank: int, query_text: str) -> None:
    """Fires per returned memory in a search response. Rank is 0-based."""
    if not is_enabled():
        return
    writer.emit(
        "memory_returned",
        memory_id=memory_id,
        value_num=float(rank),
        context={"query_preview": (query_text or "")[:60]},
    )


def emit_surprise(memory_id: int | None, score: float) -> None:
    """Fires once per candidate fact with the surprise score [0, 1].

    ⚠ This is the INSTRUMENTATION LAYER'S novelty-ratio heuristic (see
    ``surprise.compute_surprise``), DISTINCT from TrueMemory's native
    consolidation-based ``surprise_scores``. The signal keeps the name
    ``surprise`` because that is the column the dashboard reads.
    """
    if not is_enabled():
        return
    writer.emit(
        "surprise",
        memory_id=memory_id,
        value_num=score,
    )


def emit_user_forget(memory_id: int | None, requested_text: str, match_count: int) -> None:
    """Fires per explicit delete call. Powers the audit lane."""
    if not is_enabled():
        return
    writer.emit(
        "user_forget",
        memory_id=memory_id,
        value_num=float(match_count),
        value_text=(requested_text or "")[:200],
    )
