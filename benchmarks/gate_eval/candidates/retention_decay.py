"""Candidate #7 — retention_decay_only (MemoryBank-style Ebbinghaus retention).

Phase 6 prediction:
  SHORT (LoCoMo): ≈ 91.5% — within-conversation queries are short enough
                    that decay barely matters.
  LONG  (synthetic): ≈ 85% — old un-rehearsed content correctly demoted;
                    recently-discussed content correctly amplified. Strong
                    gain on time-gap > 14 days.

Implementation:
  - Ingest behaves identically to v05_baseline_nogate (no gate — store
    every extracted fact). The candidate's distinctive contribution is
    the retrieval-time retention scoring.
  - Each stored row carries an `_strength` value in the ranking pipeline,
    initialized to 1.0 at ingest. We don't add a column to the SQLite
    schema (Constraint C4 — no schema migration); instead the strength is
    derived at retrieval time from `(timestamp_age_days, retrieval_count)`
    using the Ebbinghaus formula `R = exp(-Δt/S)`.
  - On every retrieval hit, we record an in-memory rehearsal-strengthening
    counter per row id, persisted across queries within the same harness
    run. Restart-persistence would require a column; out of scope for
    Phase 9.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.gate_eval.candidates._base import Candidate, IngestTelemetry, RetrievalResult


# Ebbinghaus default decay parameters (Phase 1 #20 + ACT-R #19 calibration):
#   half-life ~24 hours for un-rehearsed content;
#   each retrieval doubles the strength.
DEFAULT_S0 = 24.0          # initial strength in hours (R(24h) ≈ 0.37)
DEFAULT_REHEARSAL_BOOST = 2.0
DEFAULT_RETRIEVAL_THRESHOLD = 0.05  # rows below this R get demoted


class RetentionDecayOnly(Candidate):
    name = "retention_decay_only"

    def __init__(self,
                 s0_hours: float = DEFAULT_S0,
                 rehearsal_boost: float = DEFAULT_REHEARSAL_BOOST,
                 retrieval_threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
                 **kwargs):
        super().__init__(s0_hours=s0_hours, rehearsal_boost=rehearsal_boost,
                          retrieval_threshold=retrieval_threshold, **kwargs)
        self.s0_hours = s0_hours
        self.rehearsal_boost = rehearsal_boost
        self.retrieval_threshold = retrieval_threshold
        self._pipeline = None
        self._memory = None
        # In-memory per-row strength state (persists for the run only)
        self._strength: dict[int, float] = defaultdict(lambda: 1.0)
        self._retrieval_counts: dict[int, int] = defaultdict(int)

    def _ensure_pipeline(self, db_path: str | Path) -> None:
        if self._pipeline is not None:
            return
        from truememory import Memory
        from truememory.ingest.pipeline import IngestionPipeline

        os.environ.pop("ANTHROPIC_API_KEY", None)
        self._memory = Memory(path=str(db_path))
        self._pipeline = IngestionPipeline(
            memory=self._memory,
            gate_threshold=0.0,
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

        # Initialize new rows' strength
        engine = getattr(self._memory, "_engine", None)
        if engine is not None and getattr(engine, "conn", None) is not None:
            try:
                rows = engine.conn.execute(
                    "SELECT id FROM messages WHERE id NOT IN ({})".format(
                        ",".join(str(k) for k in self._strength.keys()) or "0"
                    )
                ).fetchall()
                for (rid,) in rows:
                    self._strength[rid] = 1.0
            except Exception:
                pass

        return IngestTelemetry(
            n_messages_in=len(session_messages),
            n_facts_extracted=result.facts_extracted,
            n_kept=result.facts_stored + result.facts_updated,
            n_dropped=result.facts_skipped_gate,
            n_updated=result.facts_updated,
            n_skipped_dedup=result.facts_skipped_dedup,
            drop_reasons={"gate": result.facts_skipped_gate, "dedup": result.facts_skipped_dedup},
            extra={"s0_hours": self.s0_hours, "rehearsal_boost": self.rehearsal_boost},
        )

    def _retention_score(self, row_id: int, age_hours: float) -> float:
        """Ebbinghaus retention `R = exp(-Δt/S)` with rehearsal boost."""
        s = self.s0_hours * (self.rehearsal_boost ** self._retrieval_counts[row_id])
        return math.exp(-age_hours / max(s, 1e-6))

    def retrieve(self, query: str, db_path: str | Path,
                 k: int = 10) -> list[RetrievalResult]:
        self._ensure_pipeline(db_path)
        # Pull more than k so we have room to rerank by retention
        rows = self._memory.search(query, limit=max(k * 3, 30))

        engine = getattr(self._memory, "_engine", None)
        now = datetime.now(timezone.utc)

        rescored = []
        for r in rows:
            rid = r.get("id")
            if rid is None:
                rescored.append((r, float(r.get("score", 0.0) or 0.0)))
                continue
            # Get the row's timestamp
            ts_str = r.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                age_hours = (now - ts).total_seconds() / 3600.0
            except (ValueError, AttributeError):
                age_hours = 0.0

            retention = self._retention_score(rid, age_hours)
            base = float(r.get("score", 0.0) or 0.0)
            combined = base * retention
            rescored.append((r, combined))

        rescored.sort(key=lambda x: x[1], reverse=True)
        top = rescored[:k]

        # Rehearsal-strengthening: every row that surfaces in top-k gets a count++
        for r, _ in top:
            rid = r.get("id")
            if rid is not None:
                self._retrieval_counts[rid] += 1

        return [
            RetrievalResult(
                content=r.get("content", ""),
                score=float(score),
                metadata={"id": r.get("id"), "category": r.get("category"),
                          "retention": round(score / max(float(r.get("score", 1.0) or 1.0), 1e-6), 3)},
            )
            for r, score in top
        ]
