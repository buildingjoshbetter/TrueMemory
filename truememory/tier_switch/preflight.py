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


# What every loader here reads: model2vec, SentenceTransformer and CrossEncoder
# all need the config, the weights (single file or a sharded index) and a
# tokenizer. A snapshot directory alone can be an interrupted download.
_CONFIG = "config.json"
_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
_INDEXES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_TOKENIZERS = ("tokenizer.json", "vocab.txt", "tokenizer.model", "vocab.json")


def cached_snapshot(repository: str) -> bool:
    """The files a loader needs are present: in the hub cache or a local model directory."""
    local = Path(repository).expanduser()
    if local.is_dir():
        def present(name: str) -> str | None:
            return str(local / name) if (local / name).is_file() else None
    else:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return False

        def present(name: str) -> str | None:
            try:
                found = try_to_load_from_cache(repository, name)
            except Exception:  # An invalid id cannot load offline either.
                return None
            return found if isinstance(found, str) else None

    if present(_CONFIG) is None or not any(present(name) for name in _TOKENIZERS):
        return False
    if any(present(name) for name in _WEIGHTS):
        return True
    for index in _INDEXES:
        found = present(index)
        if found is None:
            continue
        try:
            import json
            with open(found, encoding="utf-8") as stream:
                shards = set(json.load(stream)["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False
        return bool(shards) and all(present(shard) for shard in shards)
    return False


def missing_models(tier: str) -> list[str]:
    """Model repositories the tier needs that cannot load while downloads are disabled."""
    if not downloads_disabled():
        return []
    from truememory.embedding_target import EmbeddingTarget, model_repository
    from truememory.tier_config import get_tier_config

    needed = (model_repository(EmbeddingTarget.capture(tier).model_id), get_tier_config(tier)["reranker"])
    return [repository for repository in dict.fromkeys(needed) if not cached_snapshot(repository)]
