"""C2 — Multiplicative hand-weights: S = Π(1 + α_i·f_i) normalized."""
from __future__ import annotations

from ._features import extract_features

# Hand-tuned alphas roughly reflecting current code's sign + magnitude.
# Negatives for noise/emoji (implemented as (1 - |α|·f)); positives for
# substantive factors.
_ALPHAS = {
    "f_noise": -0.8,  # strong reject
    "f_emoji": -0.5,
    "f_length": 0.8,
    "f_num":   0.4,
    "f_money": 0.5,
    "f_date":  0.4,
    "f_mod":   0.3,
    "f_nl":    0.15,
    "f_bul":   0.15,
    "f_excl":  0.2,
    "f_caps":  0.2,
    "f_arou":  0.6,
    "f_life":  1.0,
}
_BASE = 0.3
_FEATURE_NAMES = (
    "f_noise", "f_emoji", "f_length", "f_num", "f_money", "f_date",
    "f_mod", "f_nl", "f_bul", "f_excl", "f_caps", "f_arou", "f_life",
)

# Normalization: pre-compute the maximum theoretical product so we can
# scale back into [0, 1]. Positive alphas: all-ones-input product.
_MAX_PRODUCT = _BASE
for name in _FEATURE_NAMES:
    a = _ALPHAS[name]
    if a > 0:
        _MAX_PRODUCT *= (1 + a)


class Candidate:
    name = "C2"
    tier = "all"
    model_ids = []

    def fit(self, messages_train):
        pass

    def score(self, msg) -> float:
        f = dict(zip(_FEATURE_NAMES, extract_features(msg)))
        prod = _BASE
        for name, val in f.items():
            a = _ALPHAS[name]
            if a >= 0:
                prod *= (1 + a * val)
            else:
                # Negative alpha: multiplicative attenuation.
                prod *= max(0.0, 1 + a * val)
        # Normalize to [0, 1].
        return float(min(1.0, max(0.0, prod / _MAX_PRODUCT)))
