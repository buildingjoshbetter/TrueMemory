"""D1: Random 50% drop — lower bound for any intelligent gate."""
from __future__ import annotations

import random


class D01RandomDrop:
    name = "d01_random_drop"
    tier = "all"

    def __init__(self, drop_rate: float = 0.50, seed: int = 42):
        self.drop_rate = drop_rate
        self._rng = random.Random(seed)

    def should_encode(self, message: dict, context: list[dict] | None = None) -> bool:
        return self._rng.random() >= self.drop_rate

    def importance_score(self, message: dict, context: list[dict] | None = None) -> float:
        return self._rng.random()
