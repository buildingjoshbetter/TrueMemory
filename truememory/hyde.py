"""
TrueMemory HyDE (Hypothetical Document Embeddings)
=================================================

Implements HyDE query enhancement: given a question, generate a hypothetical
answer/conversation snippet that *would* contain the answer, then embed both
the original query and the hypothetical document for retrieval.  This bridges
the semantic gap between question phrasing and answer phrasing.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
Labels" (2022).

Usage::

    from truememory.hyde import hyde_search

    results = hyde_search(conn, "What job did Jordan get?", llm_fn=my_llm)

Dependencies:
    - truememory.hybrid (for search_hybrid + reciprocal_rank_fusion)
    - An LLM function for hypothetical doc generation (optional — falls back
      to original query if unavailable)
"""

from __future__ import annotations

import logging
import sqlite3

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

HYDE_PROMPT_CONVERSATION = """Given this question about a past conversation: "{query}"

Write a realistic snippet (2-3 sentences) from that conversation that would contain the answer. Include specific details, names, dates, and context that would naturally appear in a real conversation. Write it as dialogue between two people.

Conversation snippet:"""

HYDE_PROMPT_FACTUAL = """Given this question: "{query}"

Write a short passage (2-3 sentences) that would directly answer this question with specific facts, names, dates, and details.

Passage:"""


# ---------------------------------------------------------------------------
# Hypothetical document generation
# ---------------------------------------------------------------------------

def generate_hypothetical_doc(
    query: str,
    llm_fn=None,
    prompt_style: str = "conversation",
) -> str | None:
    """
    Generate a hypothetical document/conversation snippet that would answer
    the given query.

    Args:
        query:        The search query.
        llm_fn:       Callable that takes a prompt string and returns a string.
                      If None, returns None (caller should fall back to
                      original query only).
        prompt_style: ``"conversation"`` (default) for chat-style hypothetical,
                      ``"factual"`` for passage-style.

    Returns:
        A hypothetical document string, or None if llm_fn is unavailable.
    """
    if llm_fn is None:
        return None

    if prompt_style == "factual":
        prompt = HYDE_PROMPT_FACTUAL.format(query=query)
    else:
        prompt = HYDE_PROMPT_CONVERSATION.format(query=query)

    try:
        result = llm_fn(prompt)
        if result and len(result.strip()) > 10:
            return result.strip()
    except Exception as e:
        log.debug("HyDE hypothetical doc generation failed: %s", e)

    return None


def generate_multi_hypothetical_docs(
    query: str,
    llm_fn=None,
    n: int = 2,
) -> list[str]:
    """
    Generate multiple hypothetical documents for diversity.

    Args:
        query:  The search query.
        llm_fn: Callable that takes a prompt string and returns a string.
        n:      Number of hypothetical docs to generate.

    Returns:
        List of hypothetical document strings (may be fewer than n if some
        generations fail).
    """
    docs = []
    styles = ["conversation", "factual"]
    for i in range(n):
        style = styles[i % len(styles)]
        doc = generate_hypothetical_doc(query, llm_fn=llm_fn, prompt_style=style)
        if doc:
            docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# HyDE-enhanced search
# ---------------------------------------------------------------------------

def hyde_search(
    conn: sqlite3.Connection,
    query: str,
    llm_fn=None,
    limit: int = 10,
    candidate_pool: int = 30,
    fts_weight: float = 1.0,
    vec_weight: float = 1.0,
) -> list[dict]:
    """
    Search with both original query and HyDE-enhanced embedding.

    Runs two parallel searches — one with the original query, one with the
    hypothetical document — and fuses results with RRF.  If llm_fn is
    unavailable, falls back to standard hybrid search.

    Args:
        conn:           Open database connection.
        query:          The search query.
        llm_fn:         Callable for hypothetical doc generation.
        limit:          Max results to return.
        candidate_pool: Candidates to pull from each search.
        fts_weight:     Weight for FTS in hybrid search.
        vec_weight:     Weight for vector in hybrid search.

    Returns:
        Fused result list sorted by relevance.
    """
    from truememory.hybrid import search_hybrid, reciprocal_rank_fusion

    # Standard search with original query
    original_results = search_hybrid(
        conn, query, limit=candidate_pool,
        fts_weight=fts_weight, vec_weight=vec_weight,
    )

    # Generate hypothetical document
    hyp_doc = generate_hypothetical_doc(query, llm_fn=llm_fn)

    if not hyp_doc:
        # No LLM available — return original results
        return original_results[:limit]

    # Search with hypothetical document
    hyde_results = search_hybrid(
        conn, hyp_doc, limit=candidate_pool,
        fts_weight=fts_weight, vec_weight=vec_weight,
    )

    # Tag sources
    for r in original_results:
        r.setdefault("source", "hybrid")
    for r in hyde_results:
        r["source"] = r.get("source", "hybrid") + "+hyde"

    # Fuse with RRF
    fused = reciprocal_rank_fusion([original_results, hyde_results])
    return fused[:limit]


def hyde_multi_search(
    conn: sqlite3.Connection,
    query: str,
    llm_fn=None,
    limit: int = 10,
    candidate_pool: int = 30,
    n_hypothetical: int = 2,
    fts_weight: float = 1.0,
    vec_weight: float = 1.0,
) -> list[dict]:
    """
    Search with original query + multiple HyDE hypothetical documents.

    Generates *n_hypothetical* diverse hypothetical documents and fuses
    all search results with RRF for broader coverage.

    Args:
        conn:           Open database connection.
        query:          The search query.
        llm_fn:         Callable for hypothetical doc generation.
        limit:          Max results to return.
        candidate_pool: Candidates per search.
        n_hypothetical: Number of hypothetical docs.
        fts_weight:     Weight for FTS.
        vec_weight:     Weight for vector.

    Returns:
        Fused result list.
    """
    from truememory.hybrid import search_hybrid, reciprocal_rank_fusion

    all_result_lists = []

    # Original query search
    original_results = search_hybrid(
        conn, query, limit=candidate_pool,
        fts_weight=fts_weight, vec_weight=vec_weight,
    )
    all_result_lists.append(original_results)

    # Generate and search with hypothetical docs
    hyp_docs = generate_multi_hypothetical_docs(query, llm_fn=llm_fn, n=n_hypothetical)
    for doc in hyp_docs:
        try:
            hyde_results = search_hybrid(
                conn, doc, limit=candidate_pool,
                fts_weight=fts_weight, vec_weight=vec_weight,
            )
            all_result_lists.append(hyde_results)
        except Exception as e:
            log.debug("HyDE hybrid search failed for hypothetical doc: %s", e)

    if len(all_result_lists) == 1:
        return original_results[:limit]

    fused = reciprocal_rank_fusion(all_result_lists)
    return fused[:limit]
