"""C1 — Baseline: current TrueMemory L4 (SHA 9b7af17).

Runs the full existing consolidate stack verbatim from
`truememory/consolidation.py`: regex contradiction detection,
calendar-month extractive summaries, entity profile sheets,
structured-fact roster + locations.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from benchmarks.gate_eval.candidates.l4_consolidation._base import (
    ConsolidateTelemetry, L4Candidate, RetrievalResult,
)


def _db_bytes(db_path: Path | str) -> int:
    try:
        return Path(db_path).stat().st_size
    except OSError:
        return 0


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except sqlite3.Error:
        return 0


class C1Baseline(L4Candidate):
    name = "c1_baseline"
    tier = "edge"  # also runs unchanged on base/pro

    def consolidate(self, db_path: str | Path) -> ConsolidateTelemetry:
        from truememory.consolidation import (
            build_summaries, detect_contradictions,
            build_entity_summary_sheets, build_structured_facts,
        )

        bytes_before = _db_bytes(db_path)
        conn = sqlite3.connect(str(db_path))
        try:
            t0 = time.time()
            detect_contradictions(conn)
            build_summaries(conn)
            build_entity_summary_sheets(conn)
            build_structured_facts(conn)
            wall = time.time() - t0
            rows = {
                "summaries": _row_count(conn, "summaries"),
                "fact_timeline": _row_count(conn, "fact_timeline"),
            }
        finally:
            conn.close()

        return ConsolidateTelemetry(
            wall_clock_s=wall,
            rows_written=rows,
            bytes_added=_db_bytes(db_path) - bytes_before,
            notes="C1 baseline — regex + calendar + entity profiles + structured facts",
        )

    def retrieve_augmented(self, query: str, db_path: str | Path,
                           k: int = 10) -> list[RetrievalResult]:
        from truememory.consolidation import search_consolidated
        from truememory.fts_search import search_fts

        conn = sqlite3.connect(str(db_path))
        results: list[RetrievalResult] = []
        try:
            # Consolidated hits first (summaries + fact_timeline)
            for r in search_consolidated(conn, query, limit=k):
                results.append(RetrievalResult(
                    content=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                    source=r.get("source", "summary"),
                    metadata={"period": r.get("period"), "entity": r.get("entity", "")},
                ))
            # Backfill with FTS message hits to fill up to k
            need = max(0, k - len(results))
            if need > 0:
                fts_hits = search_fts(conn, query, limit=need)
                for r in fts_hits:
                    results.append(RetrievalResult(
                        content=r.get("content", ""),
                        score=float(r.get("score", 0.0)),
                        source="messages",
                        metadata={"sender": r.get("sender", ""),
                                  "timestamp": r.get("timestamp", "")},
                    ))
        finally:
            conn.close()
        return results[:k]
