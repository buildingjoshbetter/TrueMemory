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
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

_model = None
_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_lock = threading.Lock()
_inference_lock = threading.Lock()  # Protects concurrent model.predict() calls

# ---------------------------------------------------------------------------
# Tier-aware reranker resolution (v0.4.0 paper §2.0)
# ---------------------------------------------------------------------------
#
# Edge uses the lightweight MiniLM cross-encoder (22M params, CPU-friendly).
# Base and Pro use gte-reranker-modernbert-base (149M, GPU recommended) —
# required to reach the 91.5% / 91.8% LoCoMo targets for those tiers.
#
# The active tier is cached in _active_tier. It's seeded lazily on first
# get_current_reranker_name() call (from TRUEMEMORY_EMBED_MODEL env var or
# ~/.truememory/config.json), and can be updated at runtime via
# set_active_tier() — the MCP server calls this on truememory_configure.
_TIER_RERANKERS = {
    "edge": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "base": "Alibaba-NLP/gte-reranker-modernbert-base",
    "pro": "Alibaba-NLP/gte-reranker-modernbert-base",
}

_active_tier: str | None = None  # None = not yet resolved; resolved lazily


def get_reranker_name_for_tier(tier: str) -> str:
    """Pure mapping from tier name ("edge" / "base" / "pro") to reranker HF ID.

    Case-insensitive. Unknown or empty tier names fall back to the Edge
    default (MiniLM). Does not load any model — use ``get_reranker`` for that.
    """
    if not tier:
        return _TIER_RERANKERS["edge"]
    return _TIER_RERANKERS.get(tier.lower(), _TIER_RERANKERS["edge"])


def _resolve_tier_from_env_and_config() -> str:
    """Read the active tier from TRUEMEMORY_EMBED_MODEL env var, then from
    ~/.truememory/config.json. Safe on missing / malformed config / any OS
    error — returns "edge" as the ultimate fallback.

    This is called at most once per process (cached in _active_tier) unless
    set_active_tier() is called explicitly.
    """
    import os
    env = os.environ.get("TRUEMEMORY_EMBED_MODEL", "").strip().lower()
    if env in ("edge", "base", "pro"):
        return env
    try:
        from pathlib import Path
        import json
        cfg_path = Path.home() / ".truememory" / "config.json"
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text())
            tier = (data.get("tier") or "").strip().lower()
            if tier in ("edge", "base", "pro"):
                return tier
    except (json.JSONDecodeError, OSError) as e:
        # Hunter F04 duplicate: previously a bare `except Exception: pass`
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
    global _active_tier
    if not tier:
        _active_tier = "edge"
        return
    t = tier.strip().lower()
    _active_tier = t if t in ("edge", "base", "pro") else "edge"


def get_current_reranker_name() -> str:
    """Return the reranker HF model ID for the currently-active tier.

    Resolves lazily on first call via _resolve_tier_from_env_and_config;
    subsequent calls use the cached _active_tier until set_active_tier()
    is called.
    """
    global _active_tier
    if _active_tier is None:
        _active_tier = _resolve_tier_from_env_and_config()
    return get_reranker_name_for_tier(_active_tier)


def get_reranker(model_name: str | None = None, device: str | None = None):
    """
    Lazy-load the cross-encoder reranker (singleton).

    Args:
        model_name: HuggingFace model ID.  If None (the default), resolves via
                    ``get_current_reranker_name()`` to the tier-correct model
                    (Edge → MiniLM, Base/Pro → gte-reranker-modernbert-base).
                    Pass an explicit name for overrides (bench scripts, custom
                    rerankers).
        device:     Device string (``"cpu"``, ``"cuda:0"``, etc.).
                    If None, auto-detects.

    Returns:
        A ``sentence_transformers.CrossEncoder`` instance.
    """
    global _model, _model_name

    name = model_name or get_current_reranker_name()
    if _model is not None and name == _model_name:
        return _model  # Fast path, no lock needed
    with _lock:
        if _model is not None and name == _model_name:
            return _model  # Another thread loaded it

        from sentence_transformers import CrossEncoder

        if device is None:
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        _model = CrossEncoder(name, device=device)
        _model_name = name
        return _model


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
    rr_range = rr_max - rr_min if rr_max > rr_min else 1.0

    orig_scores = [r.get("score", r.get("rrf_score", 0)) for r in reranked]
    orig_min, orig_max = min(orig_scores), max(orig_scores)
    orig_range = orig_max - orig_min if orig_max > orig_min else 1.0

    for r in reranked:
        norm_rerank = (r["rerank_score"] - rr_min) / rr_range
        norm_orig = (r.get("score", r.get("rrf_score", 0)) - orig_min) / orig_range
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
