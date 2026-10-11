"""
TrueMemory Cross-Encoder Reranker
===============================

Reranks retrieval results using a cross-encoder model that jointly encodes
(query, document) pairs for more accurate relevance scoring than embedding-
based similarity alone.

The reranker model is resolved per tier via ``get_reranker_name_for_tier``:
Edge → ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (22M, CPU-friendly, paper
§2.0 Edge reranker), Base / Pro → ``Alibaba-NLP/gte-reranker-modernbert-base``
(149M, GPU recommended). Callers with explicit needs can override by passing
``model_name=...`` to ``get_reranker``. Can optionally use GPU if available.

Usage::

    from truememory.reranker import rerank

    results = search_hybrid(conn, query, limit=50)
    reranked = rerank(query, results, top_k=10)

Dependencies:
    - sentence-transformers (``pip install sentence-transformers``)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

_model = None
_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model_certified = False
_lock = threading.Lock()
_inference_lock = threading.Lock()  # Protects concurrent model.predict() calls

# ---------------------------------------------------------------------------
# Tier-aware reranker resolution (v0.4.0 paper S2.0)
# ---------------------------------------------------------------------------
#
# Edge uses the lightweight MiniLM cross-encoder (22M params, CPU-friendly).
# Base and Pro use gte-reranker-modernbert-base (149M, GPU recommended) --
# required to reach the 92.0% / 93.0% LoCoMo targets for those tiers.
#
# The active tier is cached in _active_tier. It's seeded lazily on first
# get_current_reranker_name() call (from TRUEMEMORY_EMBED_MODEL env var or
# ~/.truememory/config.json), and can be updated at runtime via
# set_active_tier() -- the MCP server calls this on truememory_configure.
#
# The canonical mapping lives in tier_config.TIERS; this dict is a
# backward-compat delegate for callers that access _TIER_RERANKERS directly.
from truememory.tier_config import TIERS as _TIERS, get_reranker as _cfg_get_reranker

_TIER_RERANKERS = {t: cfg["reranker"] for t, cfg in _TIERS.items()}

_active_tier: str | None = None  # None = not yet resolved; resolved lazily
_frozen_reranker_id: str | None = None


def get_reranker_name_for_tier(tier: str) -> str:
    """Pure mapping from tier name ("edge" / "base" / "pro" / "custom") to reranker HF ID.

    Case-insensitive. Unknown or empty tier names fall back to the Edge
    default (MiniLM). Does not load any model -- use ``get_reranker`` for that.
    Supports the "custom" tier via tier_config.get_reranker().
    """
    if not tier:
        return _TIER_RERANKERS["edge"]
    t = tier.lower()
    if t == "custom":
        try:
            return _cfg_get_reranker("custom")
        except ValueError as e:
            log.warning("Custom tier reranker resolution failed (%s); using edge default.", e)
            return _TIER_RERANKERS["edge"]
    return _TIER_RERANKERS.get(t, _TIER_RERANKERS["edge"])


def _resolve_tier_from_env_and_config() -> str:
    """Read the active tier from TRUEMEMORY_EMBED_MODEL env var, then from
    ~/.truememory/config.json. Safe on missing / malformed config / any OS
    error — returns "edge" as the ultimate fallback.

    This is called at most once per process (cached in _active_tier) unless
    set_active_tier() is called explicitly.
    """
    import os
    _KNOWN_TIERS = {"edge", "base", "pro", "custom"}
    env = os.environ.get("TRUEMEMORY_EMBED_MODEL", "").strip().lower()
    if env in _KNOWN_TIERS:
        return env
    try:
        from pathlib import Path
        import json
        cfg_path = Path.home() / ".truememory" / "config.json"
        if cfg_path.exists():
            # utf-8-sig strips a leading BOM (#640). isinstance guard: a
            # valid-JSON non-object config (null / "x" / [..]) would make
            # data.get raise AttributeError, crashing lazy tier resolution
            # (M-25c / #640) — treat it as malformed and fall back to Edge.
            data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                tier = (data.get("tier") or "").strip().lower()
                if tier in _KNOWN_TIERS:
                    return tier
    except (json.JSONDecodeError, OSError, AttributeError, TypeError) as e:
        # Previously a bare `except Exception: pass`
        # that hid corrupt config.json exactly like mcp_server._load_config
        # did. Log a warning so the user knows the fallback fired — the MCP
        # server's _load_config (called earlier in the startup sequence)
        # handles the corrupt-file rename.
        import sys as _sys
        print(
            f"truememory: could not read tier from config.json ({type(e).__name__}: "
            f"{e}); falling back to Edge. Run `truememory-mcp --setup` to fix.",
            file=_sys.stderr,
        )
    return "edge"


def set_active_tier(tier: str) -> None:
    """Update the cached active tier.

    Called by the MCP server's ``truememory_configure`` when the user changes
    tier at runtime so subsequent ``get_reranker(model_name=None)`` calls
    resolve to the new tier's reranker. Empty / unknown tier falls back
    to "edge".
    """
    global _active_tier, _frozen_reranker_id
    from truememory.tier_switch.serving import exclusive_activation
    with exclusive_activation():
        _frozen_reranker_id = None
        if not tier:
            _active_tier = "edge"
            return
        t = tier.strip().lower()
        _active_tier = t if t in ("edge", "base", "pro", "custom") else "edge"


def get_current_reranker_name() -> str:
    """Return the reranker HF model ID for the currently-active tier.

    Resolves lazily on first call via _resolve_tier_from_env_and_config;
    subsequent calls use the cached _active_tier until set_active_tier()
    is called.
    """
    global _active_tier
    import sys
    runtime = sys.modules.get("truememory.tier_switch.runtime")
    operation = runtime.current_runtime_operation() if runtime is not None else None
    if operation is not None:
        return operation.reranker_id
    if _frozen_reranker_id is not None:
        return _frozen_reranker_id
    if _active_tier is None:
        _active_tier = _resolve_tier_from_env_and_config()
    return get_reranker_name_for_tier(_active_tier)


def apply_reranker_policy(tier: str, model_name: str, *, deadline: float | None = None,
                          cancelled: threading.Event | None = None) -> None:
    """Publish policy identity without loading an absent native model."""
    global _model, _model_name, _model_certified, _active_tier, _frozen_reranker_id
    from truememory.tier_switch.runtime import _reranker_id
    from truememory.tier_switch.serving import exclusive_activation, _control
    from truememory.model_client import CertifiedRerankerProxy, RerankerProxy
    _reranker_id(model_name)
    control = _control(None, deadline, cancelled)
    with exclusive_activation(deadline=deadline, cancelled=cancelled), _reranker_load_lock(control):
        if _model_name != model_name:
            _model, _model_certified = None, False
        elif isinstance(_model, RerankerProxy):
            _model, _model_certified = CertifiedRerankerProxy(model_name), True
        _model_name, _active_tier, _frozen_reranker_id = model_name, tier, model_name


def apply_frozen_reranker(
    tier: str, model_name: str, *, deadline: float | None = None,
    cancelled: threading.Event | None = None,
) -> None:
    """Publish an explicit default only after its existing singleton is ready."""
    global _active_tier, _frozen_reranker_id
    from truememory.tier_switch.serving import exclusive_activation
    with exclusive_activation(deadline=deadline, cancelled=cancelled):
        # Release before loading a distinct reranker. An acknowledgement belongs
        # to the runtime bridge and is not published on this partial boundary.
        if _model_name != model_name:
            unload_reranker()
        get_reranker(model_name=model_name, deadline=deadline, cancelled=cancelled, certified=True)
        _active_tier = tier
        _frozen_reranker_id = model_name


def unload_reranker(
    should_unload: "Callable[[], bool] | None" = None,
) -> bool:
    """Release the reranker model from memory.

    Args:
        should_unload: Optional predicate evaluated inside ``_lock`` before
            clearing the model. When provided and it returns ``False``, the
            unload is skipped (a concurrent search arrived while we were
            waiting on the lock). Pass ``None`` for unconditional unload.

    Returns:
        ``True`` if the model was actually unloaded, ``False`` if skipped.
    """
    global _model
    from truememory.tier_switch.serving import exclusive_activation
    with exclusive_activation(), _lock:
        if should_unload is not None and not should_unload():
            return False
        if _model is None:
            return False
        _model = None
        return True


@contextmanager
def _reranker_load_lock(control: object) -> Iterator[None]:
    from truememory.tier_switch.serving import _check, _wait_seconds
    acquired = False
    try:
        while not acquired:
            _check(control)
            seconds = _wait_seconds(control)
            acquired = _lock.acquire() if seconds is None else _lock.acquire(timeout=seconds)
        _check(control)
        yield
    finally:
        if acquired:
            _lock.release()


def get_reranker(
    model_name: str | None = None, device: str | None = None, *,
    deadline: float | None = None, cancelled: threading.Event | None = None, certified: bool = False,
):
    """
    Lazy-load the cross-encoder reranker (singleton).

    When the shared model server is enabled (default), returns a proxy
    that routes inference to the server even when its endpoint is absent.
    Local loading requires explicit TRUEMEMORY_NO_MODEL_SERVER=1.

    Args:
        model_name: HuggingFace model ID.  If None (the default), resolves via
                    ``get_current_reranker_name()`` to the tier-correct model
                    (Edge → MiniLM, Base/Pro → gte-reranker-modernbert-base).
                    Pass an explicit name for overrides (bench scripts, custom
                    rerankers).
        device:     Device string (``"cpu"``, ``"cuda:0"``, etc.).
                    If None, auto-detects.

    Returns:
        A ``sentence_transformers.CrossEncoder`` instance or RerankerProxy.
    """
    global _model, _model_name, _model_certified
    from truememory.tier_switch.serving import _control, _check
    control = _control(None, deadline, cancelled)
    _check(control)
    if type(certified) is not bool:
        raise TypeError("Reranker certification must be boolean")

    name = model_name or get_current_reranker_name()
    from truememory.tier_switch.runtime import current_runtime_operation, TierRuntimeError
    operation = current_runtime_operation()
    if operation is not None and name != operation.reranker_id:
        raise TierRuntimeError("Reranker override must be captured before serving admission")
    if operation is not None and operation.selection is not None:
        certified = True
    cached_model, cached_name = _model, _model_name
    if cached_model is not None and name == cached_name and (not certified or _model_certified):
        return cached_model
    from truememory.tier_switch.runtime import require_model_load_allowed
    require_model_load_allowed()
    from contextlib import nullcontext
    from truememory.tier_switch.serving import exclusive_activation
    with exclusive_activation(deadline=deadline, cancelled=cancelled) if operation is None else nullcontext():
        with _reranker_load_lock(control):
            cached_model, cached_name = _model, _model_name
            if cached_model is not None and name == cached_name and (not certified or _model_certified):
                return cached_model

            _model = cached_model = None  # Release a distinct cached slot before construction.
            from truememory.model_client import use_model_server, get_reranker_proxy
            if use_model_server():
                _check(control)
                if certified:
                    from truememory.model_client import CertifiedRerankerProxy
                    proxy = CertifiedRerankerProxy(name)
                else:
                    proxy = get_reranker_proxy(model_name=name)
                _model = proxy
                _model_name = name
                _model_certified = certified
                return proxy

            if device is None:
                # Issue #577: honor TRUEMEMORY_DEVICE (cpu|mps|cuda|auto) before
                # auto-detection. An explicit device= parameter wins over the env.
                from truememory.mps_utils import auto_detect_device, resolve_device
                device = resolve_device(auto_detect_device())

            from truememory.mps_utils import ensure_mps_memory_budget
            ensure_mps_memory_budget(device)
            from sentence_transformers import CrossEncoder
            _check(control)
            _model = CrossEncoder(name, device=device)
            _model_name = name
            _model_certified = True
            result = _model
            return result


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


def _normalize_and_fuse(
    reranked: list[dict],
    rerank_weight: float,
    rrf_weight: float,
    top_k: int,
) -> list[dict]:
    """Normalize rerank + original scores to [0,1] and fuse."""
    if not reranked:
        return []
    rerank_scores = [r["rerank_score"] for r in reranked]
    rr_min, rr_max = min(rerank_scores), max(rerank_scores)
    # Degenerate case (all rerank scores identical): map to neutral 0.5 like
    # normalize_scores, not 0.0, so a uniform-rerank pool does not collapse
    # the rerank component to zero (#633 M-77).
    rr_degenerate = rr_max <= rr_min
    rr_range = rr_max - rr_min if rr_max > rr_min else 1.0

    orig_scores = [r.get("score", r.get("rrf_score", 0)) for r in reranked]
    orig_min, orig_max = min(orig_scores), max(orig_scores)
    orig_degenerate = orig_max <= orig_min
    orig_range = orig_max - orig_min if orig_max > orig_min else 1.0

    for r in reranked:
        norm_rerank = 0.5 if rr_degenerate else (r["rerank_score"] - rr_min) / rr_range
        norm_orig = (
            0.5 if orig_degenerate
            else (r.get("score", r.get("rrf_score", 0)) - orig_min) / orig_range
        )
        r["fused_score"] = rerank_weight * norm_rerank + rrf_weight * norm_orig
        r["score"] = r["fused_score"]

    reranked.sort(key=lambda r: r["fused_score"], reverse=True)
    return reranked[:top_k]


def rerank(
    query: str,
    results: list[dict],
    top_k: int = 10,
    model_name: str | None = None,
    device: str | None = None,
    batch_size: int = 64,
) -> list[dict]:
    """
    Rerank a list of retrieval results using a cross-encoder.

    The cross-encoder scores each (query, document.content) pair and returns
    the top *top_k* results sorted by cross-encoder score descending.

    Args:
        query:      The search query.
        results:    List of result dicts (must have ``"content"`` key).
        top_k:      Number of results to return after reranking.
        model_name: Optional override for the cross-encoder model.
        device:     Optional device string.
        batch_size: Batch size for prediction.

    Returns:
        Top *top_k* results sorted by cross-encoder score, each with an added
        ``"rerank_score"`` key.
    """
    if not results:
        return []

    if len(results) <= 1:
        return results[:top_k]

    model = get_reranker(model_name=model_name, device=device)

    # Build (query, content) pairs
    pairs = [(query, r.get("content", "")) for r in results]

    # Score all pairs (locked for thread safety with parallel queries)
    with _inference_lock:
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    # Attach scores and sort
    scored = []
    for r, score in zip(results, scores):
        entry = dict(r)  # shallow copy
        entry["rerank_score"] = float(score)
        scored.append(entry)

    scored.sort(key=lambda r: r["rerank_score"], reverse=True)
    return scored[:top_k]


def rerank_with_fusion(
    query: str,
    results: list[dict],
    top_k: int = 10,
    rrf_weight: float = 0.3,
    rerank_weight: float = 0.7,
    **kwargs,
) -> list[dict]:
    """
    Rerank results and fuse cross-encoder scores with original RRF scores.

    This blends the original retrieval ranking with the cross-encoder's
    assessment, preventing the reranker from completely overriding useful
    signals from keyword/vector search.

    Args:
        query:          The search query.
        results:        List of result dicts.
        top_k:          Number of results to return.
        rrf_weight:     Weight for original RRF/retrieval score.
        rerank_weight:  Weight for cross-encoder score.

    Returns:
        Top *top_k* results sorted by fused score.
    """
    if not results:
        return []

    reranked = rerank(query, results, top_k=len(results), **kwargs)
    return _normalize_and_fuse(reranked, rerank_weight, rrf_weight, top_k)


# ---------------------------------------------------------------------------
# Modality-aware reranking
# ---------------------------------------------------------------------------

def rerank_with_modality_fusion(
    query: str,
    results: list[dict],
    top_k: int = 10,
    rrf_weight: float = 0.3,
    rerank_weight: float = 0.7,
    **kwargs,
) -> list[dict]:
    """
    Rerank with cross-encoder, then apply modality-aware score adjustments.

    Episodes and facts are summaries; their scores are adjusted based on
    question type:
    - Detail questions (specific names, dates, numbers) → prefer raw messages
    - Synthesis questions (explain, describe, why) → boost episodes/facts
    - General questions → no modality adjustment
    """
    if not results:
        return []

    reranked = rerank(query, results, top_k=len(results), **kwargs)
    question_type = _classify_question_type(query)

    for r in reranked:
        modality = r.get("modality", "conversation")

        if question_type == "detail":
            if modality in ("episode", "fact"):
                r["rerank_score"] = r["rerank_score"] * 0.7
        elif question_type == "synthesis":
            if modality in ("episode", "fact"):
                r["rerank_score"] = r["rerank_score"] * 1.2

    return _normalize_and_fuse(reranked, rerank_weight, rrf_weight, top_k)


def _classify_question_type(query: str) -> str:
    """
    Classify question as 'detail', 'synthesis', or 'general'.

    Detail: specific facts, names, dates, numbers
    Synthesis: explanations, summaries, reasoning
    """
    import re
    q = query.lower().strip()

    detail_patterns = [
        r"\bwhat date\b", r"\bwhat time\b", r"\bwhat is the name\b",
        r"\bhow many\b", r"\bhow much\b", r"\bwhat number\b",
        r"\bwhat.*address\b", r"\bwhat.*phone\b", r"\bwhat.*email\b",
        r"\bwhen did\b", r"\bwhen was\b", r"\bwhen is\b",
        r"\bwhere did\b", r"\bwhere was\b", r"\bwhere is\b",
        r"\bwho is\b", r"\bwho was\b", r"\bwho did\b",
    ]

    synthesis_patterns = [
        r"^explain\b", r"^describe\b", r"^summarize\b",
        r"\bwhat kind of\b", r"\bwhat type of\b",
        r"^why\b", r"\bhow does\b", r"\bhow do\b",
        r"\bwhat fields\b", r"\bwhat activities\b",
        r"\brelationship\b", r"\blikely\b",
    ]

    if any(re.search(p, q) for p in detail_patterns):
        return "detail"
    if any(re.search(p, q) for p in synthesis_patterns):
        return "synthesis"
    return "general"


# ---------------------------------------------------------------------------
# LLM-based reranking
# ---------------------------------------------------------------------------

_RERANK_PROMPT = """Given the question below, rate each document's relevance from 0-10.
0 = completely irrelevant, 10 = directly answers the question or contains key evidence.

Question: {query}

Documents:
{documents}

For each document, output ONLY a line like "D1: 8" (document number: score).
Output ALL {n} scores, one per line:"""


def rerank_with_llm(
    query: str,
    results: list[dict],
    llm_fn,
    top_k: int = 15,
) -> list[dict]:
    """
    Rerank results using an LLM judge for relevance scoring.

    Much more accurate than cross-encoder models for conversational
    content because the LLM understands context, paraphrasing, and
    can reason about relevance.

    Args:
        query:   The search query.
        results: Candidate results to rerank.
        llm_fn:  Callable that takes a prompt and returns text.
        top_k:   Number of results to return.

    Returns:
        Top *top_k* results sorted by LLM-assigned relevance score.
    """
    if not results or len(results) <= top_k:
        return results[:top_k]

    # Build document list for prompt (truncate long content)
    doc_lines = []
    for i, r in enumerate(results):
        content = r.get("content", "")[:200]
        _sender = r.get("sender", "")
        doc_lines.append(f"D{i+1}: {content}")

    documents = "\n".join(doc_lines)
    prompt = _RERANK_PROMPT.format(
        query=query, documents=documents, n=len(results),
    )

    try:
        response = llm_fn(prompt)
        # Parse scores from "D1: 8" format
        import re
        scores = {}
        for line in response.strip().split("\n"):
            m = re.match(r'D(\d+)\s*:\s*(\d+)', line.strip())
            if m:
                idx = int(m.group(1)) - 1  # 0-based
                score = int(m.group(2))
                if 0 <= idx < len(results):
                    scores[idx] = score

        # Assign scores to results
        scored = []
        for i, r in enumerate(results):
            entry = dict(r)
            entry["llm_rerank_score"] = scores.get(i, 0)
            entry["score"] = scores.get(i, 0)
            scored.append(entry)

        scored.sort(key=lambda r: (-r["llm_rerank_score"], -r.get("rrf_score", 0)))
        return scored[:top_k]

    except Exception as e:
        log.warning("LLM reranking failed: %s — returning original order", e)
        return results[:top_k]
