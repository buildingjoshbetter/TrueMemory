"""D2 — All-ones diagnostic baseline."""
from __future__ import annotations


class Candidate:
    name = "D2"
    tier = "all"
    model_ids = []

    def fit(self, messages_train):
        pass

    def score(self, msg) -> float:
        return 1.0
