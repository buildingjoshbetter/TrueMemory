"""D1 — no L0 at all, no user filter either.

Floor diagnostic. If anything beats this, L0 (or at least user-scoping)
helps. Mostly here as a sanity check that our harness isn't accidentally
scoring nulls.
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


class D1NoL0(L0Candidate):
    name = "d1_no_l0"
    tier = "edge"
    consumes_at_retrieval = False

    def build_profile(self, persona_id: str,
                      messages: list[dict]) -> Profile:
        return Profile(
            persona_id=persona_id,
            data={},
            bytes_estimate=0,
            readable_summary=f"(D1) no profile; no user filter",
        )

    def score_for_personalization(self, query: str,
                                  active_profile: Profile,
                                  candidate_messages: list[dict]
                                  ) -> list[RerankResult]:
        """Pure lexical overlap between query and candidate. No persona signal."""
        q_words = set(query.lower().split())
        results = []
        for msg in candidate_messages:
            overlap = len(q_words & set(msg["text"].lower().split()))
            results.append(RerankResult(
                message_text=msg["text"],
                source_persona_id=msg.get("source_persona_id", ""),
                score=float(overlap),
                metadata={},
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
