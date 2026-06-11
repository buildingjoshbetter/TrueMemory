"""C1+RelFloor — current L4 with proposed fix L4-FIX1.

Identical to C1 baseline at consolidate-time. Retrieve-time:
monkey-patch `search_contradictions` to gate fact_timeline rows
by requiring match_score + fact_match ≥ 2 (instead of the current
match_score > 0 or fact_match > 0).

This tests the Phase 10 §10.9 fix hypothesis directly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from benchmarks.gate_eval.candidates.l4_consolidation._base import (
    ConsolidateTelemetry, L4Candidate, RetrievalResult,
)
from benchmarks.gate_eval.candidates.l4_consolidation.c1_baseline import (
    C1Baseline,
)


def _relevance_gated_search_contradictions(conn, query, floor=2):
    """Like truememory.consolidation.search_contradictions but with a
    minimum relevance floor (match_score + fact_match ≥ floor) before
    emitting a row."""
    stop_words = {
        "what", "which", "where", "when", "how", "does", "did", "is",
        "are", "was", "were", "the", "a", "an", "of", "for", "to",
        "in", "on", "at", "by", "with", "and", "or", "but", "not",
        "its", "it", "do", "has", "have", "had", "use", "used",
        "using", "current", "currently", "now", "today",
    }
    query_words = [
        w.lower().strip("?.,!\"'")
        for w in query.split()
        if w.lower().strip("?.,!\"'") not in stop_words and len(w) > 2
    ]
    if not query_words:
        return []

    results: list[dict] = []
    subjects = conn.execute("SELECT DISTINCT subject FROM fact_timeline").fetchall()

    for (subject,) in subjects:
        subject_lower = subject.lower()
        match_score = sum(
            1 for w in query_words
            if w in subject_lower or subject_lower in w
        )
        facts = conn.execute(
            "SELECT id, fact, timestamp, superseded_by, source_message_id "
            "FROM fact_timeline WHERE subject = ? ORDER BY timestamp",
            (subject,),
        ).fetchall()
        fact_match = sum(
            1 for _, fact, _, _, _ in facts
            for w in query_words
            if w in fact.lower()
        )

        # L4-FIX1: relevance floor.
        if (match_score + fact_match) < floor:
            continue

        if match_score > 0 or fact_match > 0:
            history = [
                {"id": r[0], "fact": r[1], "timestamp": r[2],
                 "superseded": r[3] is not None,
                 "source_message_id": r[4]}
                for r in facts
            ]
            current = [h for h in history if not h["superseded"]]
            latest = current[-1] if current else history[-1] if history else None
            if latest is None:
                continue
            results.append({
                "subject": subject,
                "current_fact": latest["fact"],
                "current_timestamp": latest["timestamp"],
                "source_message_id": latest["source_message_id"],
                "history": history,
                "relevance": match_score + fact_match,
            })

    results.sort(key=lambda r: r["relevance"], reverse=True)
    return results


class C1RelFloor(L4Candidate):
    name = "c1_relfloor"
    tier = "edge"

    def consolidate(self, db_path: str | Path) -> ConsolidateTelemetry:
        # Identical to C1 baseline consolidation
        tel = C1Baseline().consolidate(db_path)
        tel.notes = "C1 baseline consolidate + L4-FIX1 retrieval-gate at search-time"
        return tel

    def retrieve_augmented(self, query: str, db_path: str | Path,
                           k: int = 10) -> list[RetrievalResult]:
        """Override retrieval to use the relevance-floor-gated
        contradiction search. Mirrors truememory.consolidation.search_consolidated
        but substitutes the gated contradiction function."""
        import json
        import re
        from truememory.consolidation import _fts_search, _build_safe_fts_query

        conn = sqlite3.connect(str(db_path))
        results_out: list[RetrievalResult] = []
        try:
            lower_query = query.lower()
            query_words_gen = set(
                w.lower().strip("?.,!\"'")
                for w in query.split()
                if len(w) > 3
            )

            # Summaries path (identical to search_consolidated)
            target_entity = None
            for (entity,) in conn.execute(
                "SELECT DISTINCT entity FROM summaries WHERE entity != ''"
            ).fetchall():
                if entity and entity.lower() in lower_query:
                    target_entity = entity.lower()
                    break

            summaries = conn.execute(
                "SELECT id, period, start_date, end_date, entity, summary, key_facts, message_ids "
                "FROM summaries ORDER BY start_date"
            ).fetchall()
            scored_summaries = []
            for row in summaries:
                s_text = row[5]
                s_lower = s_text.lower()
                overlap = sum(1 for w in query_words_gen if w in s_lower)
                if overlap > 0:
                    scored_summaries.append({
                        "content": s_text,
                        "score": float(overlap),
                        "period": row[1],
                        "entity": row[4],
                        "source": "summary",
                    })
            scored_summaries.sort(key=lambda r: r["score"], reverse=True)

            # Gated fact_timeline path
            contra_results = _relevance_gated_search_contradictions(conn, query, floor=2)
            contra_out = []
            for cr in contra_results[:5]:
                history_text = "\n".join(
                    f"{h['timestamp']}: {h['fact']}"
                    + (" (superseded)" if h["superseded"] else " (current)")
                    for h in cr["history"]
                )
                contra_out.append({
                    "content": f"[Fact Timeline: {cr['subject']}]\n"
                               f"Current: {cr['current_fact']}\n"
                               f"History:\n{history_text}",
                    "score": float(cr["relevance"] * 2),
                    "source": "fact_timeline",
                })

            # Merge
            merged = scored_summaries + contra_out
            merged.sort(key=lambda r: r["score"], reverse=True)

            for r in merged[:k]:
                results_out.append(RetrievalResult(
                    content=r["content"],
                    score=r["score"],
                    source=r["source"],
                ))

            # FTS backfill
            need = max(0, k - len(results_out))
            if need > 0:
                fts_terms = [w for w in query_words_gen if len(w) > 3]
                if fts_terms:
                    fts_q = _build_safe_fts_query(list(fts_terms)[:8])
                    for r in _fts_search(conn, fts_q, limit=need):
                        results_out.append(RetrievalResult(
                            content=r["content"],
                            score=float(r.get("score", 0)),
                            source="messages",
                        ))
        finally:
            conn.close()
        return results_out[:k]
