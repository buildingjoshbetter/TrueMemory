"""Candidate 4b — EMBEDDING PREDICTION ERROR (KNN variant).

Surprise = 1 − max_cos(v_t, v_{m' ∈ prior_window_k}) — the same metric
as the proxy oracle. Calibration vs proxy must therefore be ~1.0 by
construction, which is a sanity-check; the interesting number is its
retrieval lift over the baseline when consumed.

Warning: because this candidate IS the proxy oracle, its calibration
score is not informative. The report calls this out.
"""

from __future__ import annotations

from benchmarks.gate_eval.candidates.l5_predictive._base import L5Candidate


class L5EmbedPEKnn(L5Candidate):
    name = "l5_embed_pe_knn"
    tier = "all"

    def __init__(self, **kwargs):
        kwargs.setdefault("alpha_surprise", 0.3)
        kwargs.setdefault("window", 500)
        super().__init__(**kwargs)
        self._prior = []  # list of (unit_embedding)
        try:
            from model2vec import StaticModel
            self._encoder = StaticModel.from_pretrained("minishlab/potion-base-32M")
        except ImportError:
            self._encoder = None
        import numpy as np
        self._np = np

    def score(self, msg: str, context: list[str]) -> float:
        if self._encoder is None:
            return 0.5
        emb = self._encoder.encode([msg], show_progress_bar=False)[0]
        emb = self._np.asarray(emb, dtype=self._np.float32)
        emb = emb / (self._np.linalg.norm(emb) + 1e-9)
        window = self.config.get("window", 500)

        if not self._prior:
            self._prior.append(emb)
            return 1.0

        prior_slice = self._prior[-window:]
        prior_stack = self._np.stack(prior_slice, axis=0)
        sims = prior_stack @ emb
        max_sim = float(sims.max())
        max_sim = max(-1.0, min(1.0, max_sim))
        surprise = 1.0 - max_sim
        self._prior.append(emb)
        return float(max(0.0, min(1.0, surprise)))
