"""
Memory Deduplication Pipeline
=============================

Adapted from Mem0's two-stage pattern for preventing memory bloat
while handling evolving facts.

For each extracted fact that passes the encoding gate:
1. Search existing memories for similar content
2. If high similarity found, decide: ADD, UPDATE, or SKIP
3. For updates, the old memory is superseded (not deleted)

The dedup stage runs AFTER the encoding gate. The gate decides IF something
is worth encoding. Dedup decides HOW to encode it — as a new memory or
as an update to an existing one.

This is the "memory consolidation" analog from Complementary Learning
Systems theory (McClelland, McNaughton & O'Reilly, 1995): new episodes
are compared against existing semantic knowledge, and either added as
new episodes or integrated into existing knowledge structures.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from truememory.ingest.extractor import _find_first_balanced
from truememory.ingest.models import LLMConfig, LLMError, complete

log = logging.getLogger(__name__)


class DedupAction(Enum):
    ADD = "add"       # Store as new memory
    UPDATE = "update" # Update existing memory
    SKIP = "skip"     # Duplicate, don't store


@dataclass
class DedupDecision:
    """Result of the deduplication check."""
    action: DedupAction
    fact: str                        # The fact to store (may be refined)
    existing_id: int | None = None   # ID of existing memory to update (if UPDATE)
    existing_content: str = ""       # Content of the existing memory
    reason: str = ""                 # Explanation


DEDUP_PROMPT = """\
[[TRUEMEMORY_INTERNAL_EXTRACTION]]
You are comparing a NEW fact against an EXISTING memory to decide what to do.

NEW FACT: {new_fact}
EXISTING MEMORY: {existing}

Decide ONE action:
- "add" if the new fact contains genuinely different information worth storing separately
- "update" if the new fact is a newer/better version of the same information (supersedes it)
- "skip" if the new fact is essentially the same as the existing memory (redundant)

If "update", also provide the merged/updated content that combines the best of both.

Return JSON: {{"action": "add|update|skip", "reason": "brief explanation", "merged": "updated content if action=update, else empty"}}"""


def check_duplicate(
    fact: str,
    memory,
    user_id: str = "",
    config: LLMConfig | None = None,
    similarity_threshold: float = 0.15,
) -> DedupDecision:
    """
    Check if a fact duplicates an existing memory.

    Two-stage approach:
    1. Vector search for similar memories (fast, cheap)
    2. If a plausible candidate exists, use LLM to decide ADD/UPDATE/SKIP
       (accurate, costs one LLM call per candidate pair)

    If no LLM config is provided, falls back to heuristic-only dedup.

    Why the threshold is so low (0.15): Model2Vec's 256-d embeddings
    produce compressed similarity distributions, so even near-paraphrases
    often score 0.2-0.4 — not 0.7+. A higher threshold (the previous 0.6
    default) silently skipped the LLM dedup stage and let paraphrased
    duplicates accumulate. When LLM dedup is available, we'd rather pay
    one LLM call than miss a real duplicate.

    Args:
        fact: The candidate fact to check.
        memory: TrueMemory Memory instance.
        user_id: User scope for memory search.
        config: LLM config for semantic dedup (optional).
        similarity_threshold: Score below which a candidate is treated as
            unrelated (heuristic path only). When ``config`` is provided,
            any non-empty search result is sent to the LLM.
    """
    # Stage 1: Vector search for similar memories (lightweight cosine only)
    try:
        results = memory.search_vectors(fact, limit=3) or []
    except Exception as e:
        log.warning("Dedup search failed: %s", e)
        return DedupDecision(action=DedupAction.ADD, fact=fact, reason="search failed, defaulting to add")

    if not results:
        return DedupDecision(action=DedupAction.ADD, fact=fact, reason="no existing memories")

    top = results[0]
    top_score = top.get("score", 0)
    top_content = top.get("content", "")
    top_id = top.get("id")

    # Very high similarity — likely near-exact duplicate, skip without LLM
    if top_score > 0.92:
        return DedupDecision(
            action=DedupAction.SKIP,
            fact=fact,
            existing_id=top_id,
            existing_content=top_content,
            reason=f"near-exact match ({top_score:.2f})",
        )

    # When LLM dedup is available, send the top candidate to the LLM
    # regardless of absolute similarity score — the LLM is the canonical
    # authority on paraphrase equivalence. The embedding score is only
    # used as a first-pass candidate retriever, not a filter.
    if config:
        return _llm_dedup(fact, top_content, top_id, config)

    # Heuristic-only path (no LLM): use the threshold to avoid
    # false positives from unrelated nearest neighbours.
    if top_score < similarity_threshold:
        return DedupDecision(action=DedupAction.ADD, fact=fact, reason=f"low similarity ({top_score:.2f})")

    return _heuristic_dedup(fact, top_content, top_id, top_score)


def _llm_dedup(
    fact: str,
    existing: str,
    existing_id: int | None,
    config: LLMConfig,
) -> DedupDecision:
    """Use LLM to make a nuanced dedup decision."""
    prompt = DEDUP_PROMPT.format(new_fact=fact, existing=existing)

    try:
        response = complete(config, prompt)
    except LLMError as e:
        log.warning("LLM dedup failed (%s): %s — falling back to heuristic", config.provider, e)
        return _heuristic_dedup(fact, existing, existing_id, similarity=0.7)
    except Exception as e:
        log.exception("Unexpected error in LLM dedup: %s — falling back to heuristic", e)
        return _heuristic_dedup(fact, existing, existing_id, similarity=0.7)

    # Parse response
    import json

    response = response.strip()
    response = re.sub(r"^```(?:json)?\s*\n?", "", response)
    response = re.sub(r"\n?```\s*$", "", response)

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # The LLM may return ``{"action": "update", "merged": {...}}`` —
        # the nested object breaks the naive ``\{[^{}]+\}`` regex that
        # used to live here. Reuse the extractor's balanced-bracket walker
        # so nested JSON is handled correctly.
        extracted = _find_first_balanced(response, "{", "}")
        if extracted:
            try:
                data = json.loads(extracted)
            except json.JSONDecodeError:
                return DedupDecision(action=DedupAction.ADD, fact=fact, reason="failed to parse LLM response")
        else:
            return DedupDecision(action=DedupAction.ADD, fact=fact, reason="no JSON in LLM response")

    action_str = data.get("action", "add").lower().strip()
    reason = data.get("reason", "")
    merged = data.get("merged", "")

    if action_str == "skip":
        return DedupDecision(
            action=DedupAction.SKIP,
            fact=fact,
            existing_id=existing_id,
            existing_content=existing,
            reason=reason,
        )
    elif action_str == "update":
        return DedupDecision(
            action=DedupAction.UPDATE,
            fact=merged or fact,
            existing_id=existing_id,
            existing_content=existing,
            reason=reason,
        )
    else:
        return DedupDecision(action=DedupAction.ADD, fact=fact, reason=reason)


def _word_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word sets."""
    words_a = set(re.findall(r'\w+', a.lower()))
    words_b = set(re.findall(r'\w+', b.lower()))
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


def _heuristic_dedup(
    fact: str,
    existing: str,
    existing_id: int | None,
    similarity: float,
) -> DedupDecision:
    """Heuristic dedup when no LLM is available."""
    fact_norm = fact.lower().strip()
    existing_norm = existing.lower().strip()

    # Substring containment — one is a subset of the other
    if fact_norm in existing_norm:
        return DedupDecision(
            action=DedupAction.SKIP,
            fact=fact,
            existing_id=existing_id,
            existing_content=existing,
            reason="new fact is subset of existing memory",
        )

    if existing_norm in fact_norm:
        return DedupDecision(
            action=DedupAction.UPDATE,
            fact=fact,
            existing_id=existing_id,
            existing_content=existing,
            reason="new fact expands on existing memory",
        )

    # Word-overlap check — catches rephrased duplicates that substring
    # matching misses. If >60% of words are shared, these facts are
    # about the same thing. The newer one supersedes the older.
    jaccard = _word_overlap(fact_norm, existing_norm)
    if jaccard > 0.60:
        if len(fact_norm) >= len(existing_norm):
            return DedupDecision(
                action=DedupAction.UPDATE,
                fact=fact,
                existing_id=existing_id,
                existing_content=existing,
                reason=f"rephrased duplicate (word overlap {jaccard:.0%})",
            )
        else:
            return DedupDecision(
                action=DedupAction.SKIP,
                fact=fact,
                existing_id=existing_id,
                existing_content=existing,
                reason=f"shorter restatement of existing (word overlap {jaccard:.0%})",
            )

    # High embedding similarity + update markers — likely a correction
    if similarity > 0.75:
        for marker in ["no longer", "not anymore", "now", "switched", "changed", "moved", "actually"]:
            if marker in fact_norm:
                return DedupDecision(
                    action=DedupAction.UPDATE,
                    fact=fact,
                    existing_id=existing_id,
                    existing_content=existing,
                    reason=f"appears to update existing fact (marker: {marker})",
                )

    # Default: add as separate memory
    return DedupDecision(
        action=DedupAction.ADD,
        fact=fact,
        existing_id=existing_id,
        existing_content=existing,
        reason=f"similar but distinct (score={similarity:.2f})",
    )
