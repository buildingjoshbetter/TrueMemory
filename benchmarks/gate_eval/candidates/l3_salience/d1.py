"""D1 — Random diagnostic baseline."""
from __future__ import annotations

import random


class Candidate:
    name = "D1"
    tier = "all"
    model_ids = []

    def __init__(self):
        self._rng = random.Random(42)

    def fit(self, messages_train):
        self._rng = random.Random(42)  # reset per fold for determinism

    def score(self, msg) -> float:
        return self._rng.random()
