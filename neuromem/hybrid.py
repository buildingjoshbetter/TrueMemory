"""
Neuromem Hybrid Search
======================

Reciprocal Rank Fusion (RRF) combining FTS5 keyword search and Model2Vec
vector search into a single ranked result list.

RRF is a simple, parameter-free fusion algorithm that merges multiple ranked
lists by assigning each document a score based on its rank position in each
list::

    rrf_score(d) = SUM_i  1 / (k + rank_i(d))

where *k* = 60 (the standard constant that prevents high-ranked documents
from dominating).  Documents appearing in **both** lists always outscore
documents found by only one retriever, making RRF naturally self-balancing.

Usage::

    from neuromem.storage import create_db, load_messages_from_file
    from neuromem.vector_search import init_vec_table, build_vectors
    from neuromem.hybrid import search_hybrid

    conn = create_db("neuromem.db")
    load_messages_from_file(conn, "synthetic_v2_messages.json")
    init_vec_table(conn)
    build_vectors(conn)

    results = search_hybrid(conn, "networking problems", limit=10)
    for r in results:
        print(f"[{r['source']:4s}] score={r['score']:.5f}  {r['content'][:80]}")

Dependencies:
    - neuromem.fts_search  (FTS5 keyword search)
    - neuromem.vector_search (Model2Vec semantic search)
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (generic)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF assigns each document a score of ``1 / (k + rank)`` from each list it
    appears in, then sums those scores.  Documents that appear in more lists
    (or at higher ranks) receive higher fused scores.

    The algorithm is retriever-agnostic -- it works with any number of ranked
    lists of any provenance as long as each result dict contains an ``"id"``
    key.

    Args:
        result_lists: List of ranked result lists.  Each inner list must be
                      sorted by descending relevance, and each result dict
                      must contain at least an ``"id"`` key.
        k:            RRF smoothing constant (default **60**).  Higher values
                      reduce the influence of top-ranked documents.

    Returns:
        Merged list sorted by descending RRF score.  Each result dict retains
        its original fields and gains two additional keys:

        - ``rrf_score`` -- the raw RRF score.
        - ``score`` -- alias for ``rrf_score`` (for API consistency).
    """
    # Accumulate scores and keep the best copy of each document.
    scores: dict[int, float] = defaultdict(float)
    best_doc: dict[int, dict] = {}

    for result_list in result_lists:
        for rank_0, doc in enumerate(result_list):
            doc_id = doc["id"]
            rank_1 = rank_0 + 1  # RRF uses 1-based ranks
            scores[doc_id] += 1.0 / (k + rank_1)

            # Keep the copy with the most fields (or the first seen).
            if doc_id not in best_doc or len(doc) > len(best_doc[doc_id]):
                best_doc[doc_id] = doc

    # Build fused result list.
    fused: list[dict] = []
    for doc_id, rrf_score in scores.items():
        entry = dict(best_doc[doc_id])  # shallow copy
        entry["rrf_score"] = round(rrf_score, 8)
        entry["score"] = entry["rrf_score"]
        fused.append(entry)

    # Sort by RRF score descending (tie-break by id for determinism).
    fused.sort(key=lambda d: (-d["rrf_score"], d["id"]))
    return fused


# ---------------------------------------------------------------------------
# Hybrid search (FTS5 + vector with RRF)
# ---------------------------------------------------------------------------

# Number of candidates to pull from each retriever before fusion.
# Pulling many more than *limit* ensures RRF has rich lists to fuse.
_CANDIDATE_POOL = 200


def search_hybrid(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    fts_weight: float = 1.0,
    vec_weight: float = 1.0,
) -> list[dict]:
    """
    Hybrid search combining FTS5 keyword search, Model2Vec completion vectors,
    and Model2Vec separation vectors with 3-list Reciprocal Rank Fusion.

    Steps:
        1. Run FTS5 full-text search for the top candidates.
        2. Run vector (completion) similarity search for the top candidates.
        3. Optionally run separation vector search as a third source.
        4. Fuse all lists with weighted RRF.
        5. Annotate each result with provenance metadata (``source``,
           ``fts_rank``, ``vec_rank``).
        6. Return the top *limit* results.

    The optional *fts_weight* and *vec_weight* parameters scale each
    retriever's RRF contribution.  Separation vectors receive 80% of
    *vec_weight*.  With equal weights (the default), a document found by
    **all three** retrievers always outranks one found by only one.

    Args:
        conn:       Open database connection with FTS5 tables populated and
                    sqlite-vec vectors built.
        query:      Natural-language search string.
        limit:      Maximum results to return (default 10).
        fts_weight: Multiplier for FTS5 RRF scores (default 1.0).
        vec_weight: Multiplier for vector RRF scores (default 1.0).

    Returns:
        List of result dicts sorted by descending hybrid score, each with:
        ``id``, ``content``, ``sender``, ``recipient``, ``timestamp``,
        ``category``, ``modality``, ``score``, ``rrf_score``, ``fts_rank``
        (int or ``None``), ``vec_rank`` (int or ``None``), ``source``
        (e.g. ``"fts"``, ``"vec"``, ``"fts+vec"``, ``"fts+vec+sep"``).
    """
    from neuromem.fts_search import search_fts
    from neuromem.vector_search import search_vector

    # Try to import separation search
    _has_sep = False
    try:
        from neuromem.vector_search import search_vector_separation
        # Check if the table exists and has data
        try:
            conn.execute("SELECT COUNT(*) FROM vec_messages_sep").fetchone()
            _has_sep = True
        except Exception:
            pass
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # 1. Retrieve candidates from all engines.
    # ------------------------------------------------------------------
    fts_results = search_fts(conn, query, limit=_CANDIDATE_POOL)
    vec_results = search_vector(conn, query, limit=_CANDIDATE_POOL)

    sep_results: list[dict] = []
    if _has_sep:
        try:
            # Check sender diversity — in 2-3 person conversations, separation
            # embeddings all share the same sender/recipient prefix, which
            # creates a uniform (unhelpful) ranking that dilutes the signal
            # from FTS + completion vectors.
            unique_senders_row = conn.execute(
                "SELECT COUNT(DISTINCT sender) FROM messages WHERE sender != ''"
            ).fetchone()
            sender_count = unique_senders_row[0] if unique_senders_row else 0
            if sender_count > 5:
                sep_results = search_vector_separation(conn, query, limit=_CANDIDATE_POOL)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 2. Build rank maps for provenance tracking.
    # ------------------------------------------------------------------
    fts_ranks: dict[int, int] = {
        doc["id"]: rank + 1 for rank, doc in enumerate(fts_results)
    }
    vec_ranks: dict[int, int] = {
        doc["id"]: rank + 1 for rank, doc in enumerate(vec_results)
    }
    sep_ranks: dict[int, int] = {
        doc["id"]: rank + 1 for rank, doc in enumerate(sep_results)
    } if sep_results else {}

    # ------------------------------------------------------------------
    # 3. 3-list weighted RRF fusion.
    # ------------------------------------------------------------------
    k = 60
    scores: dict[int, float] = defaultdict(float)
    best_doc: dict[int, dict] = {}

    for rank_0, doc in enumerate(fts_results):
        doc_id = doc["id"]
        scores[doc_id] += fts_weight * (1.0 / (k + rank_0 + 1))
        if doc_id not in best_doc or len(doc) > len(best_doc[doc_id]):
            best_doc[doc_id] = doc

    for rank_0, doc in enumerate(vec_results):
        doc_id = doc["id"]
        scores[doc_id] += vec_weight * (1.0 / (k + rank_0 + 1))
        if doc_id not in best_doc or len(doc) > len(best_doc[doc_id]):
            best_doc[doc_id] = doc

    sep_weight = vec_weight * 0.8  # slightly lower weight for separation
    for rank_0, doc in enumerate(sep_results):
        doc_id = doc["id"]
        scores[doc_id] += sep_weight * (1.0 / (k + rank_0 + 1))
        if doc_id not in best_doc or len(doc) > len(best_doc[doc_id]):
            best_doc[doc_id] = doc

    # ------------------------------------------------------------------
    # 4. Assemble results with provenance metadata.
    # ------------------------------------------------------------------
    fused: list[dict] = []
    for doc_id, rrf_score in scores.items():
        entry = dict(best_doc[doc_id])  # shallow copy

        in_fts = doc_id in fts_ranks
        in_vec = doc_id in vec_ranks
        in_sep = doc_id in sep_ranks

        entry["rrf_score"] = round(rrf_score, 8)
        entry["score"] = entry["rrf_score"]
        entry["fts_rank"] = fts_ranks.get(doc_id)
        entry["vec_rank"] = vec_ranks.get(doc_id)

        sources = []
        if in_fts:
            sources.append("fts")
        if in_vec:
            sources.append("vec")
        if in_sep:
            sources.append("sep")
        entry["source"] = "+".join(sources) if sources else "unknown"

        fused.append(entry)

    # Sort by RRF score descending (tie-break by id for determinism).
    fused.sort(key=lambda d: (-d["rrf_score"], d["id"]))

    return fused[:limit]
