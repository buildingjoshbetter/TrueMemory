"""C3 — frozen style embedder.

Primary model: `AnnaWegmann/Style-Embedding` (RoBERTa-base, 125M params,
768-dim embeddings, content-independent style via conversation-controlled
triplet training).

If `sentence-transformers` is not installed the candidate logs a warning
and returns profiles that are all-zero vectors (sentinel for "model
unavailable"). See result JSON `candidate_unavailable` flag.

Consumption at retrieval: cosine similarity between query-encoded vector
and candidate-message encoded vector, with the persona's mean style
vector used as a bias — `score = persona_bias * sim(query, candidate)`
where `persona_bias = 1 + cos(candidate_vec, profile_vec)`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gate_eval.candidates.l0_personality._base import (
    L0Candidate,
    Profile,
    RerankResult,
)

_HF_MODEL_ID = "AnnaWegmann/Style-Embedding"


def _cos(u, v) -> float:
    num = sum(a * b for a, b in zip(u, v))
    na = math.sqrt(sum(a * a for a in u))
    nb = math.sqrt(sum(b * b for b in v))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


class C3StyleEmbedder(L0Candidate):
    name = "c3_style_embedder_wegmann"
    tier = "base"
    consumes_at_retrieval = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._available = None

    def _load_model(self):
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_HF_MODEL_ID)
            self._available = True
        except Exception as exc:
            self._model = None
            self._available = False
            print(f"[C3] model unavailable: {exc}", file=sys.stderr)
        return self._available

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not self._load_model():
            # Sentinel: 768-d zero vectors
            return [[0.0] * 768 for _ in texts]
        embs = self._model.encode(texts, show_progress_bar=False,
                                  convert_to_numpy=False)
        return [list(map(float, e)) for e in embs]

    def build_profile(self, persona_id: str,
                      messages: list[dict]) -> Profile:
        if not messages:
            return Profile(persona_id=persona_id, data={},
                           bytes_estimate=2,
                           readable_summary="(empty)")
        texts = [m["text"] for m in messages]
        embs = self._encode(texts)
        # Mean-pool
        if embs and embs[0]:
            dim = len(embs[0])
            mean = [sum(e[i] for e in embs) / len(embs) for i in range(dim)]
        else:
            mean = []
        data = {
            "style_vector": mean,
            "message_count": len(messages),
            "model": _HF_MODEL_ID,
            "available": bool(self._available),
        }
        # bytes estimate: 768 floats × 4 bytes ≈ 3 KB
        bytes_est = len(mean) * 4 + 64
        summary = (f"{persona_id} · style_vec({len(mean)}d) · "
                   f"model={'ok' if self._available else 'UNAVAILABLE'}")
        return Profile(
            persona_id=persona_id, data=data,
            bytes_estimate=bytes_est,
            readable_summary=summary,
        )

    def score_for_personalization(self, query: str,
                                  active_profile: Profile,
                                  candidate_messages: list[dict]
                                  ) -> list[RerankResult]:
        profile_vec = active_profile.data.get("style_vector", [])
        if not profile_vec or not self._load_model():
            # Degrade to persona-filter + lexical (D0-like) fallback
            q_words = set(query.lower().split())
            results = []
            for msg in candidate_messages:
                same = msg.get("source_persona_id", "") == active_profile.persona_id
                base = 5.0 if same else 0.0
                overlap = len(q_words & set(msg["text"].lower().split()))
                results.append(RerankResult(
                    message_text=msg["text"],
                    source_persona_id=msg.get("source_persona_id", ""),
                    score=base + 0.01 * overlap,
                    metadata={"degraded": True},
                ))
            results.sort(key=lambda r: r.score, reverse=True)
            return results

        texts = [msg["text"] for msg in candidate_messages]
        [q_vec] = self._encode([query])
        cand_vecs = self._encode(texts)
        results = []
        for msg, v in zip(candidate_messages, cand_vecs):
            same = msg.get("source_persona_id", "") == active_profile.persona_id
            sim_query = _cos(q_vec, v)
            sim_profile = _cos(profile_vec, v)
            # Persona-scoping bias as in C5 — keeps candidates comparable.
            base = 5.0 if same else 0.0
            score = base + 0.5 * sim_query + 0.5 * sim_profile
            results.append(RerankResult(
                message_text=msg["text"],
                source_persona_id=msg.get("source_persona_id", ""),
                score=score,
                metadata={"sim_query": sim_query, "sim_profile": sim_profile},
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
