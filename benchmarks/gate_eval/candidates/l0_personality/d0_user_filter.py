"""D0 — user-filtered L1 retrieval only, no distilled L0.

The HARD baseline (per L0_PREREGISTRATION §1 and CANDIDATES.md).
Mimics Mem0's "flat user-tagged facts + user filter" approach.

Any Tier A candidate that does not beat D0 on the primary composite
MUST NOT ship — it's bringing L0 complexity without L0 value.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gate_eval.candidates.l0_personality._base import (
    L0Candidate,
    Profile,
    RerankResult,
)


class D0UserFilter(L0Candidate):
    name = "d0_user_filter_only"
    tier = "edge"
    consumes_at_retrieval = True  # consumes the persona_id filter

    def build_profile(self, persona_id: str,
                      messages: list[dict]) -> Profile:
        return Profile(
            persona_id=persona_id,
            data={"note": "no L0 distillation; retrieval filters by persona_id"},
            bytes_estimate=0,
            readable_summary=f"(D0) no distilled profile for {persona_id}",
        )

    def score_for_personalization(self, query: str,
                                  active_profile: Profile,
                                  candidate_messages: list[dict]
                                  ) -> list[RerankResult]:
        """Score 1.0 for matches to active persona, 0.0 otherwise.
        Lexical overlap between query and candidate text breaks ties.
        """
        q_words = set(query.lower().split())
        results = []
        for msg in candidate_messages:
            same_user = msg.get("source_persona_id", "") == active_profile.persona_id
            base = 1.0 if same_user else 0.0
            overlap = len(q_words & set(msg["text"].lower().split()))
            score = base + 0.01 * overlap  # break ties within same-user class
            results.append(RerankResult(
                message_text=msg["text"],
                source_persona_id=msg.get("source_persona_id", ""),
                score=score,
                metadata={"same_user": same_user},
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
