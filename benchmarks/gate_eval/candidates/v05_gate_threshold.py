"""Candidate #2 — v05_gate_threshold (heuristic gate at configurable threshold).

Parameterizes the v0.5.0 EncodingGate at τ ∈ {0.10, 0.20, 0.30, 0.40, 0.50}.
Per Finding 1, the existing 3-signal weighted sum is architecturally
non-functional on cold-start data — at τ=0.30 it drops 0% of LoCoMo
messages; at τ=0.50 still 0%. This candidate exists to *confirm* that
empirically across the sweep, not to win.

Phase 9 will instantiate this candidate with each of the 5 thresholds
and write 5 result JSONs (one per τ).
"""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.gate_eval.candidates._base import Candidate, IngestTelemetry, RetrievalResult


class V05GateThreshold(Candidate):
    name = "v05_gate_threshold"

    def __init__(self, threshold: float = 0.30, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = float(threshold)
        self._pipeline = None
        self._memory = None

    def _ensure_pipeline(self, db_path: str | Path) -> None:
        if self._pipeline is not None:
            return
        from truememory import Memory
        from truememory.ingest.pipeline import IngestionPipeline

        os.environ.pop("ANTHROPIC_API_KEY", None)
        self._memory = Memory(path=str(db_path))
        self._pipeline = IngestionPipeline(
            memory=self._memory,
            gate_threshold=self.threshold,
            use_llm_dedup=True,
        )

    def ingest(self, session_messages: list[dict], db_path: str | Path,
               session_id: int = 0) -> IngestTelemetry:
        self._ensure_pipeline(db_path)
        lines = []
        for m in session_messages:
            label = "User" if m["role"] in ("user", "human") else "Assistant"
            lines.append(f"{label}: {m['content']}")
        transcript = "\n\n".join(lines)
        result = self._pipeline.ingest_text(transcript, session_id=str(session_id))
        return IngestTelemetry(
            n_messages_in=len(session_messages),
            n_facts_extracted=result.facts_extracted,
            n_kept=result.facts_stored + result.facts_updated,
            n_dropped=result.facts_skipped_gate,
            n_updated=result.facts_updated,
            n_skipped_dedup=result.facts_skipped_dedup,
            drop_reasons={"gate": result.facts_skipped_gate, "dedup": result.facts_skipped_dedup},
            extra={"threshold": self.threshold},
        )

    def retrieve(self, query: str, db_path: str | Path,
                 k: int = 10) -> list[RetrievalResult]:
        self._ensure_pipeline(db_path)
        rows = self._memory.search(query, limit=k)
        return [
            RetrievalResult(
                content=r.get("content", ""),
                score=float(r.get("score", 0.0) or 0.0),
                metadata={"id": r.get("id"), "category": r.get("category")},
            )
            for r in rows
        ]
