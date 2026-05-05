"""
TrueMemory Configuration
========================

Centralized configuration for memory tiers, embedding models, and rerankers.
Allows for easy customization of the memory pipeline via standard tiers or
a 'custom' tier that reads from environment variables.
"""

import os
import logging

log = logging.getLogger(__name__)

# Default tier mappings (v0.4.0 paper-aligned)
DEFAULT_TIERS = {
    "edge": {
        "embed_model": "model2vec",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "embed_dim": 256,
    },
    "base": {
        "embed_model": "qwen3_256",
        "reranker": "Alibaba-NLP/gte-reranker-modernbert-base",
        "embed_dim": 256,
    },
    "pro": {
        "embed_model": "qwen3_256",
        "reranker": "Alibaba-NLP/gte-reranker-modernbert-base",
        "embed_dim": 256,
    },
}

# Legacy model dimensions (for backward compatibility or explicit model selection)
MODEL_DIMS = {
    "model2vec": 256,
    "minilm": 384,
    "bge-small": 384,
    "qwen3_256": 256,
}


def get_tier_config(tier_name: str | None = None) -> dict:
    """
    Resolve the configuration for a given tier name.

    If tier_name is 'custom', reads from environment variables:
      - TRUEMEMORY_CUSTOM_EMBED_MODEL
      - TRUEMEMORY_CUSTOM_RERANKER
      - TRUEMEMORY_CUSTOM_EMBED_DIM

    Otherwise, returns the configuration from DEFAULT_TIERS.
    If tier_name is None, defaults to the TRUEMEMORY_EMBED_MODEL env var or 'edge'.
    """
    if tier_name is None:
        tier_name = os.environ.get("TRUEMEMORY_EMBED_MODEL", "edge")

    lowered = tier_name.lower().strip()

    if lowered == "custom":
        custom_embed = os.environ.get("TRUEMEMORY_CUSTOM_EMBED_MODEL", "model2vec")
        custom_reranker = os.environ.get(
            "TRUEMEMORY_CUSTOM_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        try:
            custom_dim = int(os.environ.get("TRUEMEMORY_CUSTOM_EMBED_DIM", "256"))
        except ValueError:
            log.warning("Invalid TRUEMEMORY_CUSTOM_EMBED_DIM; falling back to 256")
            custom_dim = 256

        return {
            "embed_model": custom_embed,
            "reranker": custom_reranker,
            "embed_dim": custom_dim,
        }

    # Return default or fallback to edge
    return DEFAULT_TIERS.get(lowered, DEFAULT_TIERS["edge"])


def get_embedding_dim(model_name: str) -> int:
    """Return the dimension for a specific internal model name."""
    # Check MODEL_DIMS first, then fallback to config resolution if it's a tier name
    if model_name in MODEL_DIMS:
        return MODEL_DIMS[model_name]

    # If it's a tier name, resolve its config
    if model_name in DEFAULT_TIERS or model_name == "custom":
        return get_tier_config(model_name)["embed_dim"]

    # Final fallback for arbitrary HF IDs
    return 256
