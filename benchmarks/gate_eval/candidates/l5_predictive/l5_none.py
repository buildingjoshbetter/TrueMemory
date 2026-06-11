"""Candidate 11 — NO L5 AT ALL.

Every message scores 0.5 (constant). alpha_surprise = 0. The retrieval
pipeline is identical to the baseline with no surprise-coding layer.
This is the Cao-critique null: if the lift isn't there, ship this.
"""

from __future__ import annotations

from benchmarks.gate_eval.candidates.l5_predictive._base import L5Candidate


class L5None(L5Candidate):
    name = "l5_none"
    tier = "all"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # alpha_surprise = 0 inherited

    def score(self, msg: str, context: list[str]) -> float:
        return 0.5
