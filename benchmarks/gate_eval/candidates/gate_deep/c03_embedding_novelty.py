"""C03: Embedding Novelty Detection.

Uses sentence-transformers to compute embedding similarity between
new messages and stored messages. High similarity = redundant = drop.
"""
from __future__ import annotations

import numpy as np
from collections import deque

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import SentenceTransformer
    print("  [c03_embedding] Loading all-MiniLM-L6-v2...", flush=True)
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class C03EmbeddingNovelty:
    name = "c03_embedding_novelty"
    tier = "base"

    def __init__(self, max_stored: int = 200, novelty_threshold: float = 0.50):
        self.max_stored = max_stored
        self.novelty_threshold = novelty_threshold
        self._stored_embeddings: deque[np.ndarray] = deque(maxlen=max_stored)

    def importance_score(self, message: dict, context: list[dict] | None = None) -> float:
        text = (message.get("content") or "").strip()
        if not text:
            return 0.0
        if len(text) < 4:
            return 0.0

        model = _get_model()
        emb = model.encode(text, normalize_embeddings=True)

        if not self._stored_embeddings:
            return 1.0

        max_sim = max(
            _cosine_sim(emb, stored) for stored in self._stored_embeddings
        )
        novelty = 1.0 - max(0.0, min(1.0, max_sim))
        return novelty

    def should_encode(self, message: dict, context: list[dict] | None = None) -> bool:
        text = (message.get("content") or "").strip()
        if not text or len(text) < 4:
            return False

        model = _get_model()
        emb = model.encode(text, normalize_embeddings=True)

        if not self._stored_embeddings:
            self._stored_embeddings.append(emb)
            return True

        max_sim = max(
            _cosine_sim(emb, stored) for stored in self._stored_embeddings
        )
        novelty = 1.0 - max(0.0, min(1.0, max_sim))
        keep = novelty >= self.novelty_threshold
        if keep:
            self._stored_embeddings.append(emb)
        return keep

    def reset(self):
        self._stored_embeddings.clear()
