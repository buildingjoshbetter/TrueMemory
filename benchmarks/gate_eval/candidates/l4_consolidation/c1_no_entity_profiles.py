"""C1_NoEntityProfiles — isolate root cause of D2>C1 contradiction win.

Exactly C1 baseline EXCEPT skip `build_entity_summary_sheets`.
Keeps regex contradictions, monthly summaries, and structured facts.

If this matches D2's contradiction accuracy, the root cause of the
D2 win is entity_profile summaries (not fact_timeline noise as
Phase 10 initially hypothesized).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from benchmarks.gate_eval.candidates.l4_consolidation._base import (
    ConsolidateTelemetry, L4Candidate, RetrievalResult,
)


def _bytes(p):
    try:
        return Path(p).stat().st_size
    except OSError:
        return 0


class C1NoEntityProfiles(L4Candidate):
    name = "c1_no_entity_profiles"
    tier = "edge"

    def consolidate(self, db_path: str | Path) -> ConsolidateTelemetry:
        from truememory.consolidation import (
            build_summaries, detect_contradictions,
            build_structured_facts,
        )

        before = _bytes(db_path)
        conn = sqlite3.connect(str(db_path))
        try:
            t0 = time.time()
            detect_contradictions(conn)
            build_summaries(conn)
            # SKIP build_entity_summary_sheets
            build_structured_facts(conn)
            wall = time.time() - t0
            rows = {
                "summaries": conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0],
                "fact_timeline": conn.execute("SELECT COUNT(*) FROM fact_timeline").fetchone()[0],
            }
        finally:
            conn.close()
        return ConsolidateTelemetry(
            wall_clock_s=wall,
            rows_written=rows,
            bytes_added=_bytes(db_path) - before,
            notes="C1 baseline minus build_entity_summary_sheets (isolates entity-profile summaries as root cause)",
        )

    def retrieve_augmented(self, query: str, db_path: str | Path,
                           k: int = 10) -> list[RetrievalResult]:
        from truememory.consolidation import search_consolidated
        from truememory.fts_search import search_fts
        conn = sqlite3.connect(str(db_path))
        results: list[RetrievalResult] = []
        try:
            for r in search_consolidated(conn, query, limit=k):
                results.append(RetrievalResult(
                    content=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                    source=r.get("source", "summary"),
                ))
            need = max(0, k - len(results))
            if need > 0:
                for r in search_fts(conn, query, limit=need):
                    results.append(RetrievalResult(
                        content=r.get("content", ""),
                        score=float(r.get("score", 0.0)),
                        source="messages",
                    ))
        finally:
            conn.close()
        return results[:k]
