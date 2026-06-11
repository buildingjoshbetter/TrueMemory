"""Candidate 4 — EMBEDDING PREDICTION ERROR (EMA variant).

Compute per-message `v_t = embed(msg_t)`. Predict `v̂_t = EMA(v_{<t})`
with decay λ. Surprise = 1 − cos(v_t, v̂_t), clipped to [0, 1].

Matches the paper's `π_t = ‖v_t − v̂_t‖₂` formulation but uses cosine
distance (scale-invariant) rather than L2 so it's directly comparable
to the proxy oracle.

Tier: all (reuses Model2Vec at Edge, Qwen3 at Base/Pro). This
implementation uses Model2Vec for harness simplicity.
"""

from __future__ import annotations

from benchmarks.gate_eval.candidates.l5_predictive._base import L5Candidate


class L5EmbedPEEma(L5Candidate):
    name = "l5_embed_pe_ema"
    tier = "all"

    def __init__(self, **kwargs):
        kwargs.setdefault("alpha_surprise", 0.3)
        kwargs.setdefault("ema_lambda", 0.85)
        super().__init__(**kwargs)
        self._ema = None
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
        norm = self._np.linalg.norm(emb) + 1e-9
        emb_unit = emb / norm

        if self._ema is None:
            self._ema = emb_unit.copy()
            return 1.0

        # Surprise BEFORE update (prediction error)
        ema_norm = self._np.linalg.norm(self._ema) + 1e-9
        ema_unit = self._ema / ema_norm
        cos_sim = float(emb_unit @ ema_unit)
        cos_sim = max(-1.0, min(1.0, cos_sim))
        surprise = (1.0 - cos_sim) / 2.0  # map [-1, 1] → [1, 0]

        # Update EMA
        lam = self.config.get("ema_lambda", 0.85)
        self._ema = lam * self._ema + (1 - lam) * emb_unit
        return float(max(0.0, min(1.0, surprise)))
