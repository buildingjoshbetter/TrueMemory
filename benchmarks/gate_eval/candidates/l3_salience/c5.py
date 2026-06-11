"""C5 — Minimal-3 factor set: length + num + arousal, additive clamped."""
from __future__ import annotations

from ._features import extract_features


class Candidate:
    name = "C5"
    tier = "all"
    model_ids = []

    def fit(self, messages_train):
        pass

    def score(self, msg) -> float:
        f = extract_features(msg)
        # Indices: 2=length, 3=num, 11=arou
        s = 0.3 + 0.3 * f[2] + 0.2 * f[3] + 0.2 * f[11]
        return max(0.0, min(1.0, s))
