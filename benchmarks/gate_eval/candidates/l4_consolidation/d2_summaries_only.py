"""D2 — Summaries-only diagnostic.

Runs build_summaries but DISABLES contradiction detection,
entity-profile sheets, and structured facts. Isolates the
contribution of the summaries-table path by itself.
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


class D2SummariesOnly(L4Candidate):
    name = "d2_summaries_only"
    tier = "edge"

    def consolidate(self, db_path: str | Path) -> ConsolidateTelemetry:
        from truememory.consolidation import build_summaries

        before = _bytes(db_path)
        conn = sqlite3.connect(str(db_path))
        try:
            # Wipe other L4 tables for clean isolation
            for table in ("fact_timeline",):
                try:
                    conn.execute(f"DELETE FROM {table}")
                except sqlite3.Error:
                    pass
            conn.commit()

            t0 = time.time()
            build_summaries(conn)
            wall = time.time() - t0
            n = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
        finally:
            conn.close()

        return ConsolidateTelemetry(
            wall_clock_s=wall,
            rows_written={"summaries": n, "fact_timeline": 0},
            bytes_added=_bytes(db_path) - before,
            notes="D2 — summaries only; no contradiction detection / no entity profiles",
        )

    def retrieve_augmented(self, query: str, db_path: str | Path,
                           k: int = 10) -> list[RetrievalResult]:
        from truememory.consolidation import search_consolidated
        from truememory.fts_search import search_fts

        conn = sqlite3.connect(str(db_path))
        results: list[RetrievalResult] = []
        try:
            for r in search_consolidated(conn, query, limit=k):
                if r.get("source") == "fact_timeline":
                    continue  # D2 suppresses fact_timeline path
                results.append(RetrievalResult(
                    content=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                    source=r.get("source", "summary"),
                    metadata={"period": r.get("period"), "entity": r.get("entity", "")},
                ))
            need = max(0, k - len(results))
            if need > 0:
                for r in search_fts(conn, query, limit=need):
                    results.append(RetrievalResult(
                        content=r.get("content", ""),
                        score=float(r.get("score", 0.0)),
                        source="messages",
                        metadata={"sender": r.get("sender", "")},
                    ))
        finally:
            conn.close()
        return results[:k]
