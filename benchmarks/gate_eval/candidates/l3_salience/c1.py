"""C1 — Baseline: current additive scorer (v0.5.0 compute_message_salience)."""
from __future__ import annotations

from truememory.salience import compute_message_salience


class Candidate:
    name = "C1"
    tier = "all"
    model_ids = []

    def fit(self, messages_train):
        pass

    def score(self, msg) -> float:
        return float(compute_message_salience(
            msg.get("content", "") or "",
            msg.get("modality", "") or "",
        ))
