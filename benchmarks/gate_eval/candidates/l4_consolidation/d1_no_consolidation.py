"""D1 — No consolidation diagnostic.

Disables all L4 writers. Retrieval consults only messages / messages_fts —
no summaries or fact_timeline. If D1 does not measurably cost accuracy,
L4's consolidation framing is decorative.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from benchmarks.gate_eval.candidates.l4_consolidation._base import (
    ConsolidateTelemetry, L4Candidate, RetrievalResult,
)


class D1NoConsolidation(L4Candidate):
    name = "d1_no_consolidation"
    tier = "edge"

    def consolidate(self, db_path: str | Path) -> ConsolidateTelemetry:
        # Deliberate no-op: the ingest path (L0-L3) already wrote whatever
        # it writes; L4 does nothing here. Clear any pre-existing L4 rows
        # so the storage-baseline measurement is clean.
        conn = sqlite3.connect(str(db_path))
        try:
            for table in ("summaries", "fact_timeline"):
                try:
                    conn.execute(f"DELETE FROM {table}")
                except sqlite3.Error:
                    pass
            conn.commit()
        finally:
            conn.close()
        return ConsolidateTelemetry(
            wall_clock_s=0.0,
            rows_written={"summaries": 0, "fact_timeline": 0},
            bytes_added=0,
            notes="D1 — no-op; L4 writers disabled",
        )

    def retrieve_augmented(self, query: str, db_path: str | Path,
                           k: int = 10) -> list[RetrievalResult]:
        from truememory.fts_search import search_fts

        conn = sqlite3.connect(str(db_path))
        results: list[RetrievalResult] = []
        try:
            fts_hits = search_fts(conn, query, limit=k)
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
        return results
