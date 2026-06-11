"""Candidate 1 — BASELINE (unwired).

Uses the shipped fact-fingerprint scorer (truememory.predictive) but
does NOT consume it at retrieval (alpha_surprise=0). Measures pure
retrieval substrate performance with L5 as a dead signal, matching
v0.5.0's actual behavior.
"""

from __future__ import annotations

from benchmarks.gate_eval.candidates.l5_predictive._base import L5Candidate


class L5Unwired(L5Candidate):
    name = "l5_unwired"
    tier = "all"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # alpha_surprise defaults to 0 (inherited) — no consumption.
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
        # Update accumulator with facts from this msg to match
        # build_surprise_index's chronological behavior.
        try:
            from truememory.predictive import extract_facts
            self._accum_facts.update(extract_facts(msg))
        except Exception:
            pass
        return float(s)
