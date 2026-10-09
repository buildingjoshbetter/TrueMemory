"""Explicit embedding identity for inactive target preparation.

This module imports no model framework. A descriptor identifies an embedding
space, not resident weights, database readiness, or an active configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import re
import sys


TARGET_PROBE_TEXT = "TrueMemory embedding target validation."


class EmbeddingTargetError(ValueError):
    """The requested target cannot be certified without changing its identity."""


@dataclass(frozen=True)
class EmbeddingTarget:
    tier: str
    model_id: str
    dimension: int
    tier_group: str

    def __post_init__(self) -> None:
        builtins = {
            "edge": ("model2vec", 256, "edge"),
            "base": ("qwen3_256", 256, "basepro"),
            "pro": ("qwen3_256", 256, "basepro"),
        }
        if not isinstance(self.tier, str) or self.tier not in (*builtins, "custom"):
            raise EmbeddingTargetError("Prepared targets require an explicit public tier")
        if (not isinstance(self.model_id, str)
                or re.fullmatch(r"[\w][\w.\-]*(/[\w][\w.\-]*)?", self.model_id) is None):
            raise EmbeddingTargetError("Invalid prepared embedding model identity")
        if type(self.dimension) is not int or not 1 <= self.dimension <= 4096:
            raise EmbeddingTargetError("Prepared embedding dimension must be an integer from 1 to 4096")
        if self.tier in builtins:
            if self.identity != builtins[self.tier]:
                raise EmbeddingTargetError("Prepared target does not match its built-in tier")
        elif self.tier_group != "custom":
            raise EmbeddingTargetError("Custom prepared targets require the custom table group")
        if self.model_id in {"minilm", "bge-small", "qwen3"}:
            raise EmbeddingTargetError("This legacy model identity cannot be certified by prepared targets")
        if self.model_id in {"model2vec", "qwen3_256"} and self.dimension != 256:
            raise EmbeddingTargetError("This built-in embedding model requires 256 dimensions")

    @property
    def identity(self) -> tuple[str, int, str]:
        return self.model_id, self.dimension, self.tier_group

    @property
    def tables(self) -> tuple[str, str]:
        return f"vec_messages_{self.tier_group}", f"vec_messages_sep_{self.tier_group}"

    @classmethod
    def capture(cls, tier: str) -> EmbeddingTarget:
        from truememory.tier_config import get_tier_config

        if not isinstance(tier, str):
            raise EmbeddingTargetError("Prepared targets require an explicit public tier")
        tier = tier.strip().lower()
        config = get_tier_config(tier)
        return cls(tier, config["embed_model"], config["embed_dim"], config["tier_group"])

    def check_configuration(self) -> None:
        if self != self.capture(self.tier):
            raise EmbeddingTargetError("Prepared target configuration changed; capture a new target")

    def to_wire(self) -> dict:
        return {"version": 1, "tier": self.tier, "model_id": self.model_id,
                "dimension": self.dimension, "tier_group": self.tier_group}

    @classmethod
    def from_wire(cls, value: object) -> EmbeddingTarget:
        if (not isinstance(value, dict)
                or set(value) != {"version", "tier", "model_id", "dimension", "tier_group"}
                or type(value["version"]) is not int or value["version"] != 1):
            raise EmbeddingTargetError("Unsupported prepared target descriptor")
        return cls(value["tier"], value["model_id"], value["dimension"], value["tier_group"])


def check_target_vectors(target: EmbeddingTarget, vectors: object, count: int) -> None:
    if getattr(vectors, "shape", None) != (count, target.dimension):
        raise EmbeddingTargetError("Actual embedding output does not match the prepared target dimension")


def build_target_model(
    target: EmbeddingTarget, device: str | None, *, check: Callable[[], None] | None = None,
) -> object:
    """Construct from captured arguments, never the legacy custom fallback."""
    def admit() -> None:
        # Revalidate permission after ownership waits and imports, but never
        # replace captured constructor arguments with newly read settings.
        target.check_configuration()
        if check is not None:
            check()

    admit()
    if target.model_id == "model2vec":
        from model2vec import StaticModel

        admit()
        return StaticModel.from_pretrained("minishlab/potion-base-8M", force_download=False)

    from truememory.mps_utils import ensure_mps_memory_budget
    ensure_mps_memory_budget(device)
    from sentence_transformers import SentenceTransformer

    if target.model_id == "qwen3_256":
        model_kwargs = {"attn_implementation": "eager"} if sys.platform == "darwin" else None
        admit()
        return SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B", truncate_dim=256,
            model_kwargs=model_kwargs, device=device,
        )
    admit()
    return SentenceTransformer(
        target.model_id, truncate_dim=target.dimension,
        trust_remote_code=False, device=device,
    )
