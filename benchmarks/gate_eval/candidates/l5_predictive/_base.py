"""L5Candidate base — simple score() + config interface.

An L5Candidate implements:
  - `name: str` — result-JSON key
  - `tier: str` — "edge" | "base" | "pro" | "all"
  - `score(msg: str, context: list[str]) -> float` — surprise in [0, 1]

Retrieval-time reranking is applied by the harness using
`config.alpha_surprise` (default 0 for unwired candidates, >0 for
candidates that actually consume the signal).

Set `_is_l5_candidate = True` on subclasses so `discover_l5_candidates`
picks them up.
"""

from __future__ import annotations


class L5Candidate:
    name: str = "abstract"
    tier: str = "all"
    _is_l5_candidate: bool = True

    def __init__(self, **kwargs):
        self.config = dict(kwargs)
        # Default: unwired. Candidates that do rerank set this.
        self.config.setdefault("alpha_surprise", 0.0)

    def score(self, msg: str, context: list[str]) -> float:
        raise NotImplementedError
