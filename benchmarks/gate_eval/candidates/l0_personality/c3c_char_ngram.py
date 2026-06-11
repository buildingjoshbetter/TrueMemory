"""C3c — char-n-gram hashing embedder (proxy for C3 when sentence-transformers
is unavailable in the local environment).

Not the actual Wegmann/LUAR model — but it IS a real learned-style vector
in the sense that the representation comes from the persona's own writing
statistics, not from hand-tuned keyword lists. Uses char-(3,4,5)-grams
hashed into a fixed-dim bag-of-ngrams vector, L2-normalized, mean-pooled
across the persona's messages.

Purpose: let the Phase 9 local sweep produce a comparison of a
"non-hand-tuned, persona-derived dense vector" against C1/C5 baselines,
even when the full HF model isn't available. Phase 11 (Modal) would
swap in the real Wegmann model.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gate_eval.candidates.l0_personality._base import (
    L0Candidate, Profile, RerankResult,
)

DIM = 256
NGRAM_SIZES = (3, 4, 5)


def _ngrams(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    out = []
    for n in NGRAM_SIZES:
        for i in range(len(text) - n + 1):
            out.append(text[i:i + n])
    return out


def _hash_vec(text: str) -> list[float]:
    vec = [0.0] * DIM
    for ng in _ngrams(text):
        h = hash(ng) % DIM
        vec[h] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _mean(vecs: list[list[float]]) -> list[float]:
    if not vecs:
        return [0.0] * DIM
    dim = len(vecs[0])
    out = [0.0] * dim
    for v in vecs:
        for i in range(dim):
            out[i] += v[i]
    out = [x / len(vecs) for x in out]
    norm = math.sqrt(sum(x * x for x in out))
    return [x / norm for x in out] if norm > 0 else out


def _cos(u, v) -> float:
    num = sum(a * b for a, b in zip(u, v))
    na = math.sqrt(sum(a * a for a in u))
    nb = math.sqrt(sum(b * b for b in v))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


class C3CharNgram(L0Candidate):
    name = "c3c_char_ngram_proxy"
    tier = "edge"
    consumes_at_retrieval = True

    def build_profile(self, persona_id: str,
                      messages: list[dict]) -> Profile:
        if not messages:
            return Profile(persona_id=persona_id, data={},
                           bytes_estimate=2, readable_summary="(empty)")
        vecs = [_hash_vec(m["text"]) for m in messages]
        mean_vec = _mean(vecs)
        data = {
            "style_vector": mean_vec,
            "message_count": len(messages),
            "model": f"char_ngram_hash_dim{DIM}_n{NGRAM_SIZES}",
        }
        return Profile(
            persona_id=persona_id, data=data,
            bytes_estimate=DIM * 4 + 64,
            readable_summary=f"{persona_id} · char-ngram vec({DIM}d)",
        )

    def score_for_personalization(self, query: str,
                                  active_profile: Profile,
                                  candidate_messages: list[dict]
                                  ) -> list[RerankResult]:
        profile_vec = active_profile.data.get("style_vector", [])
        q_vec = _hash_vec(query)
        results = []
        for msg in candidate_messages:
            same = msg.get("source_persona_id", "") == active_profile.persona_id
            cv = _hash_vec(msg["text"])
            sim_query = _cos(q_vec, cv)
            sim_profile = _cos(profile_vec, cv) if profile_vec else 0.0
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
