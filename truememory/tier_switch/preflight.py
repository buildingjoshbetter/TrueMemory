"""Find a switch's target models before any switch work starts.

With downloads disabled, a missing reranker otherwise surfaced only at
activation, after every memory had been re-embedded.
"""

from __future__ import annotations

import os
from pathlib import Path

_TRUE = {"1", "true", "yes", "on"}


def downloads_disabled() -> bool:
    return any(os.environ.get(name, "").strip().lower() in _TRUE
               for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def cached_snapshot(repository: str) -> bool:
    """A complete local snapshot exists; a local model directory counts."""
    if Path(repository).expanduser().is_dir():
        return True
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repository, local_files_only=True)
    except Exception:  # Not cached, an invalid id, or no hub client: all unloadable offline.
        return False
    return True


def missing_models(tier: str) -> list[str]:
    """Model repositories the tier needs that cannot load while downloads are disabled."""
    if not downloads_disabled():
        return []
    from truememory.embedding_target import EmbeddingTarget, model_repository
    from truememory.tier_config import get_tier_config

    needed = (model_repository(EmbeddingTarget.capture(tier).model_id), get_tier_config(tier)["reranker"])
    return [repository for repository in dict.fromkeys(needed) if not cached_snapshot(repository)]
