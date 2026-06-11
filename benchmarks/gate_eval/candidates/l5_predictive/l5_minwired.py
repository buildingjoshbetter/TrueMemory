"""Candidate 2 — MINIMAL WIRING.

Identical scoring to Candidate 1 (shipped fact-fingerprint), but
consumes surprise at retrieval via multiplicative rerank with
alpha_surprise > 0. Measures the lift of "wire the signal we already
have."
"""

from __future__ import annotations

from benchmarks.gate_eval.candidates.l5_predictive._base import L5Candidate


class L5Minwired(L5Candidate):
    name = "l5_minwired"
    tier = "all"

    def __init__(self, **kwargs):
        kwargs.setdefault("alpha_surprise", 0.3)  # mid-range default
        super().__init__(**kwargs)
        self._accum_facts: set[str] = set()
        try:
            from truememory.predictive import compute_surprise_score
            self._scorer = compute_surprise_score
        except Exception:
            self._scorer = None

    def score(self, msg: str, context: list[str]) -> float:
        if self._scorer is None:
            return 0.5
        s = self._scorer(msg, self._accum_facts)
        try:
            from truememory.predictive import extract_facts
            self._accum_facts.update(extract_facts(msg))
        except Exception:
            pass
        return float(s)
