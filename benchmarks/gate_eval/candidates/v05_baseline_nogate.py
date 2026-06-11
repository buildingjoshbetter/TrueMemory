"""Candidate #1 — v05_baseline_nogate.

Reproduces v0.5.0 IngestionPipeline behavior with the gate effectively
disabled (`gate_threshold=0.0`). This is the control: every candidate is
compared against this number.

Per the truememory paper §6.1 Table 1, this configuration gives 91.5%
on LoCoMo. SHORT_HORIZON_200 should reproduce ~91% (subset variance band).
"""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.gate_eval.candidates._base import Candidate, IngestTelemetry, RetrievalResult


class V05BaselineNogate(Candidate):
    name = "v05_baseline_nogate"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Lazy-loaded — first ingest creates the pipeline
        self._pipeline = None
        self._memory = None

    def _ensure_pipeline(self, db_path: str | Path) -> None:
        if self._pipeline is not None:
            return
        # Defer the truememory imports until needed so the harness can
        # introspect the candidate without paying the model-load cost.
        from truememory import Memory
        from truememory.ingest.pipeline import IngestionPipeline

        self._memory = Memory(path=str(db_path))
        # Force-disable ANTHROPIC_API_KEY for any Claude-CLI extraction
        # path the pipeline takes — keeps the experiment using OAuth
        # rather than the bad env-var key (same trick as the synthetic
        # dataset generator).
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self._pipeline = IngestionPipeline(
            memory=self._memory,
            gate_threshold=0.0,        # gate effectively off
            use_llm_dedup=True,
        )

    def ingest(self, session_messages: list[dict], db_path: str | Path,
               session_id: int = 0) -> IngestTelemetry:
        self._ensure_pipeline(db_path)

        # Format the session as a transcript string the extractor wants
        # (matches `truememory.ingest.transcript.format_for_extraction`).
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
