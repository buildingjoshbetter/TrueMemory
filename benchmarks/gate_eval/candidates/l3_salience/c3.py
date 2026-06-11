"""C3 — Log-linear (sigmoid) with hand-weights.

Weights derived by re-interpreting additive deltas as log-odds units.
"""
from __future__ import annotations

import math

from ._features import extract_features


_WEIGHTS = {
    "f_noise": -4.0,
    "f_emoji": -2.5,
    "f_length": 4.0,
    "f_num":   1.5,
    "f_money": 2.0,
    "f_date":  1.5,
    "f_mod":   1.5,
    "f_nl":    0.5,
    "f_bul":   0.5,
    "f_excl":  0.8,
    "f_caps":  1.0,
    "f_arou":  2.0,
    "f_life":  3.0,
}
# b chosen so all-zeros input yields sigmoid(b) ≈ 0.30 (C1 base).
_BIAS = math.log(0.3 / 0.7)  # = -0.847
_FEATURE_NAMES = tuple(_WEIGHTS.keys())


def _sigmoid(x: float) -> float:
    if x >= 0:
        ez = math.exp(-x)
        return 1.0 / (1.0 + ez)
    ez = math.exp(x)
    return ez / (1.0 + ez)


class Candidate:
    name = "C3"
    tier = "all"
    model_ids = []

    def fit(self, messages_train):
        pass

    def score(self, msg) -> float:
        f = dict(zip(_FEATURE_NAMES, extract_features(msg)))
        z = _BIAS
        for name, val in f.items():
            z += _WEIGHTS[name] * val
        return float(_sigmoid(z))
