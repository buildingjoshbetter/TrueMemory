"""Candidate #5 — mem0_extraction (Mem0-style extract-then-update).

Reproduces Mem0's atomic-fact extraction + ADD/UPDATE/SKIP/DELETE pipeline
as faithfully as possible against TrueMemory's storage substrate.

v0.5.0's `truememory.ingest.IngestionPipeline` already implements ADD/UPDATE/SKIP
(see `truememory.ingest.dedup.DedupAction`). This candidate's distinctive
contribution vs #1 v05_baseline_nogate is the explicit DELETE action — the
LLM judge can mark an old fact as superseded enough to remove. v0.5.0's
existing UPDATE supersedes content but leaves the row; DELETE actually
removes it.

Implementation note (deferred to Phase 9 first run): the truememory dedup
module returns ADD/UPDATE/SKIP only. To add DELETE we would either (a) extend
DedupAction in truememory.ingest.dedup, or (b) post-process: after each session,
re-judge the recently-touched rows for "is this still relevant given what we
just heard?" and prune the answer-NO ones.

For Phase 9, we use approach (b) — it doesn't require modifying truememory and
keeps the candidate self-contained in benchmarks/. The relevance judgment is
made by the same LLM the pipeline uses for dedup (Mem0's exact pattern).

Expected SHORT performance: ≈ 61% on LoCoMo per Mem0 paper Table 1
(Mem0's destructive UPDATE loses verbatim detail LoCoMo single-hop questions need).

Expected LONG performance: > v0.5.0 baseline because storage stays bounded —
estimate ~78% on synthetic.
"""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.gate_eval.candidates._base import Candidate, IngestTelemetry, RetrievalResult


class Mem0Extraction(Candidate):
    name = "mem0_extraction"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline = None
        self._memory = None
        # Tracks rows added by the most-recent session — these are the
        # candidates for post-session DELETE judgment.
        self._recent_session_row_ids: list[int] = []

    def _ensure_pipeline(self, db_path: str | Path) -> None:
        if self._pipeline is not None:
            return
        from truememory import Memory
        from truememory.ingest.pipeline import IngestionPipeline

        os.environ.pop("ANTHROPIC_API_KEY", None)
        self._memory = Memory(path=str(db_path))
        # Same gate-OFF as v05_baseline_nogate, BUT we will run an
        # additional post-session DELETE judgment pass.
        self._pipeline = IngestionPipeline(
            memory=self._memory,
            gate_threshold=0.0,
            use_llm_dedup=True,
        )

    def ingest(self, session_messages: list[dict], db_path: str | Path,
               session_id: int = 0) -> IngestTelemetry:
        self._ensure_pipeline(db_path)
        # Snapshot ids that exist BEFORE this session so we can identify
        # newly-added rows after.
        before_ids = self._all_row_ids()

        lines = []
        for m in session_messages:
            label = "User" if m["role"] in ("user", "human") else "Assistant"
            lines.append(f"{label}: {m['content']}")
        transcript = "\n\n".join(lines)
        result = self._pipeline.ingest_text(transcript, session_id=str(session_id))

        after_ids = self._all_row_ids()
        new_ids = sorted(after_ids - before_ids)

        # Phase 9 first-run policy: skip the post-session DELETE pass
        # unless explicitly enabled. The pipeline already has UPDATE
        # supersession; the additional DELETE is an experiment we run
        # later in Phase 10 ablations.
        n_deleted = 0
        if self.config.get("post_session_delete", False):
            n_deleted = self._post_session_delete_pass(transcript, new_ids)

        return IngestTelemetry(
            n_messages_in=len(session_messages),
            n_facts_extracted=result.facts_extracted,
            n_kept=result.facts_stored + result.facts_updated - n_deleted,
            n_dropped=result.facts_skipped_gate + n_deleted,
            n_updated=result.facts_updated,
            n_skipped_dedup=result.facts_skipped_dedup,
            drop_reasons={
                "gate": result.facts_skipped_gate,
                "dedup": result.facts_skipped_dedup,
                "post_session_delete": n_deleted,
            },
        )

    def _all_row_ids(self) -> set[int]:
        """Cheap snapshot of current `messages.id` set."""
        engine = getattr(self._memory, "_engine", None)
        if engine is None or not hasattr(engine, "conn") or engine.conn is None:
            return set()
        try:
            return {r[0] for r in engine.conn.execute("SELECT id FROM messages")}
        except Exception:
            return set()

    def _post_session_delete_pass(self, recent_transcript: str, candidate_ids: list[int]) -> int:
        """For each newly-added row, ask the LLM whether it should be
        retained given the FULL recent_transcript. Delete rows the LLM
        marks as redundant or trivial.

        Phase 9 baseline: disabled by default (`post_session_delete=False`).
        Phase 10 ablations may turn it on.
        """
        # Stub for now — implement when Phase 10 needs it.
        return 0

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
