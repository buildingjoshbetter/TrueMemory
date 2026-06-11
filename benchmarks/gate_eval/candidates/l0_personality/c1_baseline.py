"""C1 — current shipping hand-tuned L0 (baseline).

Wraps truememory/personality.py without modification. Acts as the
reference point every other candidate must beat on the primary
composite (per L0_PREREGISTRATION §4 null band).

Limitations (already documented in FORMALIZATION.md §2):
  * Emoji via Unicode block ranges
  * Formality via ~20 casual-keyword counts
  * Topics via 10 static keyword clusters
  * Traits via 10 trait-indicator keyword sets
  * Proper nouns via \\b[A-Z][a-z]+\\b regex
  * Single-sender attribution (bug surfaced by Phase 1 Card 6)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gate_eval.candidates.l0_personality._base import (
    L0Candidate,
    Profile,
    RerankResult,
)
from truememory.personality import (  # type: ignore
    PERSONALITY_ASPECTS,
    _assess_formality,
    _detect_emoji,
    _extract_topics,
    _extract_traits,
    _find_typical_greeting,
)


class C1HandTuned(L0Candidate):
    name = "c1_baseline_hand_tuned"
    tier = "edge"
    consumes_at_retrieval = True

    def build_profile(self, persona_id: str,
                      messages: list[dict]) -> Profile:
        msg_dicts = [{"content": m["text"], "timestamp": m.get("timestamp", "")}
                     for m in messages]
        if not msg_dicts:
            prof = Profile(persona_id=persona_id, data={}, bytes_estimate=2,
                           readable_summary="(empty)")
            return prof

        lengths = [len(m["content"]) for m in msg_dicts]
        avg_length = sum(lengths) / len(lengths)
        uses_emoji = any(_detect_emoji(m["content"]) for m in msg_dicts)
        formality = _assess_formality(msg_dicts)
        greeting = _find_typical_greeting(msg_dicts)
        topics = _extract_topics(msg_dicts)
        traits = _extract_traits(msg_dicts)

        data = {
            "message_count": len(msg_dicts),
            "communication_style": {
                "avg_length": round(avg_length, 1),
                "uses_emoji": uses_emoji,
                "formality": formality,
                "typical_greeting": greeting,
            },
            "topics": topics,
            "traits": traits,
        }
        raw = json.dumps(data, ensure_ascii=False)
        summary = (f"{persona_id} · {formality} formality · "
                   f"emoji={'y' if uses_emoji else 'n'} · "
                   f"topics={topics[:3]} · traits={traits[:3]}")
        return Profile(
            persona_id=persona_id,
            data=data,
            bytes_estimate=len(raw.encode("utf-8")),
            readable_summary=summary,
        )

    def score_for_personalization(self, query: str,
                                  active_profile: Profile,
                                  candidate_messages: list[dict]
                                  ) -> list[RerankResult]:
        """Score each candidate message by (a) whether the persona's
        topics/traits overlap with words in the candidate, (b) whether
        the candidate's formality matches the persona's, (c) fallback
        to zero signal if no overlap.
        """
        prof_data = active_profile.data
        prof_topics = set(w for t in prof_data.get("topics", [])
                          for w in t.lower().split("/"))
        prof_traits = set(prof_data.get("traits", []))
        prof_formality = (prof_data.get("communication_style", {})
                          .get("formality", "mixed"))
        prof_uses_emoji = (prof_data.get("communication_style", {})
                           .get("uses_emoji", False))

        # Aspect detection on the query (match current search_personality)
        query_lower = query.lower()
        detected_aspect = "personality"
        best = 0
        for aspect, cfg in PERSONALITY_ASPECTS.items():
            score = sum(1 for kw in cfg["keywords"] if kw in query_lower)
            if score > best:
                best = score
                detected_aspect = aspect
        aspect_fts_terms = set(PERSONALITY_ASPECTS.get(detected_aspect, {})
                               .get("fts_terms", []))

        results = []
        for msg in candidate_messages:
            text = msg["text"]
            low = text.lower()
            words = set(low.split())
            score = 0.0
            # Topic overlap
            score += 0.3 * len(words & prof_topics)
            # Aspect keyword hit in the candidate
            score += 0.4 * len(words & aspect_fts_terms)
            # Formality match
            cand_formality = _assess_formality([{"content": text}])
            if cand_formality == prof_formality:
                score += 0.2
            # Emoji match
            if _detect_emoji(text) == prof_uses_emoji:
                score += 0.1
            results.append(RerankResult(
                message_text=text,
                source_persona_id=msg.get("source_persona_id", ""),
                score=score,
                metadata={"aspect": detected_aspect},
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
