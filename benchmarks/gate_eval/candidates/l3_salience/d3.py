"""D3 — Reversed C1 (1 - C1): diagnostic."""
from __future__ import annotations

from truememory.salience import compute_message_salience


class Candidate:
    name = "D3"
    tier = "all"
    model_ids = []

    def fit(self, messages_train):
        pass

    def score(self, msg) -> float:
        return 1.0 - float(compute_message_salience(
            msg.get("content", "") or "",
            msg.get("modality", "") or "",
        ))
