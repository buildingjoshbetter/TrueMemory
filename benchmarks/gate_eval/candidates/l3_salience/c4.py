"""C4 — Learned logistic regression on retrieval-utility labels.

Per-fold train within the harness's LOCO-CV loop (the harness passes
train messages in via fit()).
"""
from __future__ import annotations

import numpy as np

from sklearn.linear_model import LogisticRegression

from ._features import FEATURE_NAMES, extract_features


class Candidate:
    name = "C4"
    tier = "all"
    model_ids = []

    def __init__(self):
        self._model: LogisticRegression | None = None

    def fit(self, messages_train: list[dict]) -> None:
        X = np.asarray([extract_features(m) for m in messages_train], dtype=float)
        y = np.asarray([m["utility_binary"] for m in messages_train], dtype=int)
        if y.sum() == 0 or y.sum() == len(y):
            # Degenerate fold — fall back to a fixed 0.5 output via a fake model.
            self._model = None
            return
        self._model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            solver="lbfgs",
        )
        self._model.fit(X, y)

    def score(self, msg: dict) -> float:
        if self._model is None:
            return 0.5
        x = np.asarray(extract_features(msg), dtype=float).reshape(1, -1)
        p = float(self._model.predict_proba(x)[0, 1])
        return p

    @property
    def feature_names(self):
        return FEATURE_NAMES

    @property
    def coefficients(self):
        if self._model is None:
            return None
        return dict(zip(FEATURE_NAMES, self._model.coef_[0].tolist()))
