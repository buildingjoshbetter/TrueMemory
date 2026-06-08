"""Search-quality helpers — scent trail and quality self-check.

Extracted from engine.py (issue #137) to reduce god-class complexity.
All functions take an explicit ``conn`` parameter instead of ``self``.
"""

from __future__ import annotations

import logging
import re
import sqlite3

from truememory.fts_search import search_fts, search_fts_by_sender

logger = logging.getLogger(__name__)


def scent_trail(
    conn: sqlite3.Connection,
    query: str,
    results: list[dict],
    limit: int,
) -> list[dict]:
    """Follow entity/term trails from top results to find related messages.

    Extracts key entities and proper nouns from the top-3 results, runs
    targeted follow-up searches (hop 1 = per-entity, hop 2 = combined
    trail terms), and merges discoveries back with discounted scores.
    """
    if not results or len(results) < 3:
        return results

    trail_entities: set[str] = set()
    trail_terms: set[str] = set()

    for r in results[:3]:
        content = r.get("content", "")
        sender = r.get("sender", "").lower()
        recipient = r.get("recipient", "").lower()

        if sender:
            trail_entities.add(sender)
        if recipient:
            trail_entities.add(recipient)

        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        common = {
            "The", "This", "That", "What", "When", "Where", "How",
            "But", "And", "Just", "Not",
        }
        for noun in proper_nouns:
            if noun not in common and len(noun) > 2:
                trail_terms.add(noun.lower())

    if not trail_entities and not trail_terms:
        return results

    existing_ids = {r.get("id") for r in results if r.get("id")}
    new_results: list[dict] = []

    for entity in list(trail_entities)[:3]:
        try:
            entity_results = search_fts_by_sender(conn, query, entity, limit=5)
            for er in entity_results:
                if er.get("id") and er["id"] not in existing_ids:
                    er["source"] = er.get("source", "scent_trail")
                    new_results.append(er)
                    existing_ids.add(er["id"])
        except Exception:
            logger.debug("Scent trail hop 1 failed for entity %s", entity, exc_info=True)

    if trail_terms:
        trail_query = " ".join(list(trail_terms)[:5])
        try:
            term_results = search_fts(conn, trail_query, limit=5)
            for tr in term_results:
                if tr.get("id") and tr["id"] not in existing_ids:
                    tr["source"] = "scent_trail"
                    tr["score"] = tr.get("score", 0) * 0.7
                    new_results.append(tr)
                    existing_ids.add(tr["id"])
        except Exception:
            logger.debug("Scent trail hop 2 failed", exc_info=True)

    results.extend(new_results)
    return results


def quality_self_check(
    conn: sqlite3.Connection,
    query: str,
    results: list[dict],
    limit: int,
) -> list[dict]:
    """Fallback when top-5 results are uniformly low quality.

    If all top-5 scores are below 0.04 with range < 0.005, runs broader
    single-term FTS searches to cast a wider net.
    """
    if not results or len(results) < 5:
        return results

    top5_scores = [r.get("score", r.get("rrf_score", 0)) for r in results[:5]]

    max_score = max(top5_scores) if top5_scores else 0
    min_score = min(top5_scores) if top5_scores else 0
    score_range = max_score - min_score

    if max_score < 0.04 and score_range < 0.005:
        try:
            words = [w for w in query.lower().split() if len(w) > 3]
            if words:
                existing_ids = {r.get("id") for r in results if r.get("id")}
                for word in words[:3]:
                    try:
                        broad_results = search_fts(conn, word, limit=10)
                        for br in broad_results:
                            if br.get("id") and br["id"] not in existing_ids:
                                br["source"] = "fallback_broad"
                                br["score"] = br.get("score", 0) * 0.5
                                results.append(br)
                                existing_ids.add(br["id"])
                    except Exception:
                        logger.debug(
                            "Broad FTS fallback failed for term in quality self-check",
                            exc_info=True,
                        )
        except Exception:
            logger.debug("Quality self-check fallback failed", exc_info=True)

    return results
