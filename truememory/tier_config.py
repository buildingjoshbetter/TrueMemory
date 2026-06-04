"""Centralized tier configuration -- single source of truth.

Every tier-aware module (vector_search, reranker, model_server,
tier_switch/cache) imports from here instead of maintaining its own
mapping dicts.  This eliminates the class of bugs where a new tier is
added to one module but forgotten in another.

Named ``tier_config`` (not ``config``) to avoid shadowing the heavily-used
local variable pattern ``config = _load_config()`` throughout mcp_server.py.
"""

from __future__ import annotations

import copy
import logging
import os
import re

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in tier definitions
# ---------------------------------------------------------------------------

TIERS: dict[str, dict] = {
    "edge": {
        "embed_model": "model2vec",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "embed_dim": 256,
        "tier_group": "edge",
        "model_name": "potion-base-8M",  # for vector_cache_registry
    },
    "base": {
        "embed_model": "qwen3_256",
        "reranker": "Alibaba-NLP/gte-reranker-modernbert-base",
        "embed_dim": 256,
        "tier_group": "basepro",
        "model_name": "Qwen3-Embedding-0.6B",
    },
    "pro": {
        "embed_model": "qwen3_256",
        "reranker": "Alibaba-NLP/gte-reranker-modernbert-base",
        "embed_dim": 256,
        "tier_group": "basepro",
        "model_name": "Qwen3-Embedding-0.6B",
    },
}

MODEL_DIMS: dict[str, int] = {
    "model2vec": 256,
    "minilm": 384,
    "bge-small": 384,
    "qwen3_256": 256,
}

VALID_TIER_GROUPS: frozenset[str] = frozenset({"edge", "basepro", "custom"})

MODEL_TO_GROUP: dict[str, str] = {
    "model2vec": "edge",
    "qwen3_256": "basepro",
    "minilm": "basepro",
    "bge-small": "basepro",
}

# Regex for validating HuggingFace model IDs:
# org/model-name or just model-name, alphanumeric + dots/hyphens/underscores
_HF_MODEL_ID_RE = re.compile(r"^[\w][\w.\-]*(\/[\w][\w.\-]*)?$")

# ---------------------------------------------------------------------------
# Custom tier resolution (Phase 2)
# ---------------------------------------------------------------------------


def resolve_custom_tier() -> dict:
    """Build custom tier config from ``TRUEMEMORY_CUSTOM_*`` env vars.

    Requires ``TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD=1`` as an explicit opt-in
    to acknowledge that arbitrary HuggingFace models will be downloaded.

    Raises:
        ValueError: if required env vars are missing or invalid.
    """
    if os.environ.get("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "").strip() != "1":
        raise ValueError(
            "Custom tier requires TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD=1 "
            "to acknowledge arbitrary model downloads."
        )

    # Do NOT lowercase the embed model -- HF IDs are case-sensitive
    # (e.g. "Qwen/Qwen3-Embedding-0.6B")
    embed = os.environ.get("TRUEMEMORY_CUSTOM_EMBED_MODEL", "").strip()
    if not embed:
        raise ValueError(
            "TRUEMEMORY_CUSTOM_EMBED_MODEL must be set for custom tier"
        )

    # Validate model ID format to prevent shell injection / path traversal
    if not _HF_MODEL_ID_RE.fullmatch(embed):
        raise ValueError(
            f"Invalid model ID format: {embed!r}. "
            f"Expected HuggingFace format like 'org/model-name' or 'model-name'."
        )

    reranker = os.environ.get(
        "TRUEMEMORY_CUSTOM_RERANKER",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ).strip()
    if reranker and not _HF_MODEL_ID_RE.fullmatch(reranker):
        raise ValueError(
            f"Invalid reranker model ID format: {reranker!r}."
        )

    raw_dim = os.environ.get("TRUEMEMORY_CUSTOM_EMBED_DIM", "256").strip()
    try:
        dim = int(raw_dim)
    except (ValueError, TypeError):
        raise ValueError(
            f"TRUEMEMORY_CUSTOM_EMBED_DIM must be an integer, got {raw_dim!r}"
        )

    if dim < 1 or dim > 4096:
        raise ValueError(
            f"TRUEMEMORY_CUSTOM_EMBED_DIM must be 1-4096, got {dim}"
        )

    return {
        "embed_model": embed,
        "reranker": reranker or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "embed_dim": dim,
        "tier_group": "custom",
        "model_name": embed,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tier_config(tier: str) -> dict:
    """Return a *copy* of the config dict for the given tier.

    Routes "custom" to :func:`resolve_custom_tier`.  Raises
    :class:`ValueError` for truly unknown tier names (does NOT silently
    fall back to edge -- that masks configuration errors).

    Returns a shallow copy so callers cannot mutate the global TIERS dict.
    """
    t = tier.lower().strip()
    if t == "custom":
        return resolve_custom_tier()
    if t not in TIERS:
        raise ValueError(
            f"Unknown tier: {tier!r}. Valid tiers: {sorted(TIERS)} + ['custom']"
        )
    return copy.copy(TIERS[t])


def get_embed_model(tier: str) -> str:
    """Return the embedding model ID for a tier."""
    return get_tier_config(tier)["embed_model"]


def get_reranker(tier: str) -> str:
    """Return the reranker HF model ID for a tier."""
    return get_tier_config(tier)["reranker"]


def get_embed_dim(tier: str) -> int:
    """Return the embedding dimension for a tier."""
    return get_tier_config(tier)["embed_dim"]


def get_embed_dim_for_model(model_name: str) -> int:
    """Return the embedding dimension for an internal model name.

    For known built-in models, returns from MODEL_DIMS.
    For unknown models (custom tier), attempts to resolve the custom
    tier config and returns the configured dimension if the model matches.
    """
    if model_name in MODEL_DIMS:
        return MODEL_DIMS[model_name]
    # Check if this is a custom tier model
    try:
        cfg = resolve_custom_tier()
        if model_name == cfg["embed_model"]:
            return cfg["embed_dim"]
    except ValueError:
        pass
    return 256


def get_tier_group(tier: str) -> str:
    """Return the tier group for a tier (edge / basepro / custom)."""
    return get_tier_config(tier)["tier_group"]


def get_model_group(model_name: str) -> str:
    """Return the tier group for a model name.

    Checks MODEL_TO_GROUP first, then falls back to "custom" for
    any model loaded via the custom tier path.
    """
    return MODEL_TO_GROUP.get(model_name, "custom")


def get_model_name_for_group(group: str) -> str:
    """Return the canonical model display name for a tier group."""
    _GROUP_MODEL_NAMES = {
        "edge": "potion-base-8M",
        "basepro": "Qwen3-Embedding-0.6B",
    }
    return _GROUP_MODEL_NAMES.get(group, "")


def is_custom_tier(tier: str) -> bool:
    """Check if a tier string refers to the custom tier."""
    return tier.lower().strip() == "custom"
