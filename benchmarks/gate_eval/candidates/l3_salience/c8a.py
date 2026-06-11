"""C8a — Gzip compression ratio as single-feature salience.

Sign convention (after empirical test, Tick 11): **HIGH ratio = LOW
salience.** Pre-tick hypothesis was wrong: short noisy messages have
high gzip ratios because the gzip header overwhelms tiny payloads.
Long, structured, information-rich messages compress *better* (low
ratio). Flipped sign to match observed AUC > 0.5.
"""
from __future__ import annotations

import math

from ._features import gzip_ratio


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


class Candidate:
    name = "C8a"
    tier = "all"
    model_ids = []

    def fit(self, messages_train):
        pass

    def score(self, msg) -> float:
        text = msg.get("content", "") or ""
        if not text:
            return 0.0
        r = gzip_ratio(text)
        # Negative slope: lower ratio (better compression) = higher salience.
        # Centered at r ≈ 0.65 (where short noisy text typically lives).
        z = 4.0 * (0.65 - r)
        return float(_sigmoid(z))
