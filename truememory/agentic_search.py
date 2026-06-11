"""Agentic search helpers — sufficiency, refined queries, entity search, cleaning.

Extracted from engine.py (issue #137) to reduce god-class complexity.
All functions take an explicit ``conn`` parameter (where needed) instead of ``self``.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Sequence

from truememory.fts_search import search_fts, search_fts_by_sender

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score normalization (issue #584)
# ---------------------------------------------------------------------------

def normalize_scores(results: list[dict], *, key: str = "score") -> list[dict]:
    """Min-max normalize *key* to [0, 1] in place.

    If all scores are equal (or the list has <= 1 element), every score is
    set to 0.5 so downstream ranking treats them as neutral.

    Returns *results* for chaining.
    """
    if len(results) <= 1:
        for r in results:
            r[key] = 0.5
        return results

    scores = [r.get(key, 0) for r in results]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span == 0:
        for r in results:
            r[key] = 0.5
        return results

    for r in results:
        r[key] = (r.get(key, 0) - lo) / span
    return results


def normalize_supplement_scores(
    primary: list[dict],
    *supplements: Sequence[list[dict]],
    key: str = "score",
) -> None:
    """Normalize each source list independently to [0, 1].

    ``primary`` is normalized first. Each supplement list is then
    independently normalized so every source competes fairly in the
    merged pool.  The lists are mutated in place.
    """
    normalize_scores(primary, key=key)
    for supp in supplements:
        if supp:
            normalize_scores(supp, key=key)


QUERY_STOP_WORDS = frozenset({
    "what", "did", "does", "do", "how", "where", "when", "who",
    "which", "why", "is", "are", "was", "were", "has", "have",
    "had", "would", "could", "should", "will", "can", "the",
    "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "about", "their", "they", "them", "his", "her", "its",
})


def check_sufficiency(top_results: list[dict]) -> bool:
    """True if retrieval results are good enough to skip round 2."""
    if not top_results or len(top_results) < 3:
        return False

    scores = [r.get("score", r.get("rrf_score", 0)) for r in top_results]
    avg_score = sum(scores) / len(scores)

    unique_prefixes = {r.get("content", "")[:100] for r in top_results}
    return avg_score > 0.02 and len(unique_prefixes) >= 3


def generate_refined_queries(
    original_query: str,
    top_results: list[dict],
    llm_fn,
) -> list[str]:
    """Generate 2-3 refined sub-queries using *llm_fn* for round-2 retrieval."""
    context_snippets = []
    for r in top_results[:5]:
        content = r.get("content", "")[:150]
        if content:
            context_snippets.append(content)

    context_str = "\n".join(context_snippets)

    q_lower = original_query.lower()
    list_patterns = [
        "what ", "which ", "where has", "where did",
        "what activities", "what books", "what movies",
        "what hobbies", "what does", "what do",
        "what items", "what places", "what foods",
        "what sports", "what songs", "what bands",
    ]
    is_list_question = any(q_lower.startswith(p) or p in q_lower for p in list_patterns)

    if is_list_question:
        prompt = f"""Given this question about a conversation: "{original_query}"

Here are results already found:
{context_str}

This question likely has MULTIPLE answers scattered across different parts of the conversation.
Generate 3 search queries to find ADDITIONAL answers not yet covered. Each query should:
- Use different vocabulary than what's already found
- Search for the same topic from a different angle
- Target different time periods or contexts

Return ONLY the queries, one per line:"""
    else:
        prompt = f"""Given this question about a conversation: "{original_query}"

And these partially relevant results:
{context_str}

Generate 2-3 alternative search queries that might find the missing information. Each query should approach the question from a different angle — try different keywords, focus on specific entities or events mentioned, or rephrase the question.

Return ONLY the queries, one per line, no numbering or explanation:"""

    try:
        response = llm_fn(prompt)
        lines = [
            line.strip().strip('"').strip("'").strip("-").strip()
            for line in response.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        return lines[:3]
    except Exception:
        return []


def entity_focused_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
) -> list[dict]:
    """Extract person names from *query* and search within their messages."""
    if not conn:
        return []

    results = []

    known_senders = set()
    try:
        rows = conn.execute(
            "SELECT DISTINCT sender FROM messages"
        ).fetchall()
        known_senders = {r[0].lower() for r in rows if r[0]}
    except Exception:
        logger.debug("Failed to fetch known senders for entity search", exc_info=True)

    query_words = query.split()
    matched_senders = []
    for word in query_words:
        clean = word.strip("'\"?.,!").lower()
        for sender in known_senders:
            if clean == sender.lower() or clean.rstrip("'s") == sender.lower():
                matched_senders.append(sender)

    query_lower = query.lower()
    for sender in known_senders:
        if len(sender) > 2 and sender.lower() in query_lower:
            if sender not in matched_senders:
                matched_senders.append(sender)

    if matched_senders:
        stop_words = QUERY_STOP_WORDS
        content_words = []
        for word in query_words:
            clean = word.strip("'\"?.,!").lower().rstrip("'s")
            if (clean not in stop_words
                and len(clean) > 2
                and clean not in {s.lower() for s in matched_senders}):
                content_words.append(word.strip("'\"?.,!"))

        if content_words:
            focused_query = " ".join(content_words)
        else:
            focused_query = query

        try:
            for sender in matched_senders[:2]:
                orig_sender = sender
                for s_row in conn.execute(
                    "SELECT DISTINCT sender FROM messages"
                ).fetchall():
                    if s_row[0] and s_row[0].lower() == sender.lower():
                        orig_sender = s_row[0]
                        break

                sender_results = search_fts_by_sender(
                    conn, focused_query, orig_sender, limit=limit
                )
                for r in sender_results:
                    r["source"] = "entity_sender"
                results.extend(sender_results)
        except Exception:
            logger.debug("FTS by sender search failed in entity search", exc_info=True)

    if not matched_senders:
        stop_words = QUERY_STOP_WORDS
        content_words = [
            w.strip("'\"?.,!") for w in query_words
            if w.strip("'\"?.,!").lower() not in stop_words
            and len(w.strip("'\"?.,!")) > 2
        ]
        if len(content_words) >= 2 and content_words != query_words:
            focused_query = " ".join(content_words)
            try:
                focused_results = search_fts(conn, focused_query, limit=limit)
                for r in focused_results:
                    r["source"] = "entity_focused"
                results.extend(focused_results)
            except Exception:
                logger.debug("Focused content search failed in entity search", exc_info=True)

    return results


def clean_results(
    results: list[dict],
    limit: int,
    max_per_session: int = 0,
) -> list[dict]:
    """Deduplicate, clean, and optionally enforce session-diversity on results."""
    cleaned: list[dict] = []
    seen_ids: set = set()
    seen_content: set = set()

    for r in results:
        content = r.get("content", "")
        rid = r.get("id")

        content_key = content[:200]
        if rid and rid in seen_ids:
            continue
        if content_key in seen_content:
            continue

        score = r.get("score", r.get("rrf_score", r.get("fused_score", r.get("rerank_score", 0))))
        if isinstance(score, (int, float)) and score < 0:
            score = 0.0

        cleaned.append({
            # Preserve None for id-less rows (#630 M-67): rewriting to 0
            # collapsed distinct supplement rows under id-keyed RRF.
            "id": rid,
            "content": content,
            "sender": r.get("sender", ""),
            "recipient": r.get("recipient", ""),
            "timestamp": r.get("timestamp", ""),
            "category": r.get("category", ""),
            "modality": r.get("modality", ""),
            "directive": r.get("directive", False),
            "score": score,
            "source": r.get("source", "agentic"),
        })

        if rid:
            seen_ids.add(rid)
        seen_content.add(content_key)

    # Tie-break on str(id): type-stable across int / "summary_N" / None
    # supplement ids (#630 M-01).
    cleaned.sort(key=lambda d: (-d["score"], str(d.get("id", ""))))

    if max_per_session > 0 and len(cleaned) > limit:
        diverse: list[dict] = []
        deferred: list[dict] = []
        session_counts: dict[str, int] = {}

        for r in cleaned:
            sess = r.get("category", "unknown")
            count = session_counts.get(sess, 0)
            if count < max_per_session:
                diverse.append(r)
                session_counts[sess] = count + 1
            else:
                deferred.append(r)

        if len(diverse) < limit:
            diverse.extend(deferred[: limit - len(diverse)])

        return diverse[:limit]

    return cleaned[:limit]
