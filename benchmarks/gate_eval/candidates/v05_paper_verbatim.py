"""Candidate #1b — v05_paper_verbatim (THE paper's actual baseline).

Phase 9 empirical finding (2026-04-23):
The original Phase 6 Candidate #1 (v05_baseline_nogate) ran the
`truememory.ingest.IngestionPipeline` against LoCoMo data and scored
1.5% — because IngestionPipeline runs LLM fact-extraction designed for
free-form Claude-Code Stop-hook transcripts, which is exactly the wrong
thing to do with already-cleaned LoCoMo dialogue turns.

The paper's actual reported 91.5% is produced by a different code path:
`truememory.engine.TrueMemoryEngine.ingest(json_path)` — verbatim storage
of pre-shaped message dicts. See `benchmarks/locomo/scripts/bench_truememory_pro.py`
lines 235-239 for the canonical ingestion pattern.

This candidate mirrors that path so the harness can produce the paper's
baseline number locally.

Important architectural note this exposed: TrueMemory has TWO ingestion
modes that should be characterized as distinct candidates in the
MEMORIST taxonomy. The Phase 6 list assumed they were the same, which
the empirical run refuted. A note has been added to the report's
methodology section.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from benchmarks.gate_eval.candidates._base import Candidate, IngestTelemetry, RetrievalResult


class V05PaperVerbatim(Candidate):
    """Mirror of the paper's ingest path: verbatim storage via engine.ingest()."""

    name = "v05_paper_verbatim"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine = None

    def _ensure_engine(self, db_path: str | Path) -> None:
        if self._engine is not None:
            return
        # Defer the import for harness `--list` performance
        from truememory.engine import TrueMemoryEngine

        os.environ.pop("ANTHROPIC_API_KEY", None)
        self._engine = TrueMemoryEngine(db_path=str(db_path))

    def ingest(self, session_messages: list[dict], db_path: str | Path,
               session_id: int = 0) -> IngestTelemetry:
        self._ensure_engine(db_path)
        # Shape the messages exactly like the paper's bench script does:
        # `{content, sender, recipient, timestamp, category, modality}`.
        # We don't have real timestamps for synthetic harness messages, so
        # fall back to the session_id as a proxy ordinal (still chronological).
        msg_dicts = []
        for i, m in enumerate(session_messages):
            msg_dicts.append({
                "content": m.get("content", ""),
                "sender": m.get("role", "user"),
                "recipient": "self",
                "timestamp": m.get("timestamp", f"session_{session_id}_{i:04d}"),
                "category": str(session_id),
                "modality": "conversation",
            })

        # The engine.ingest path takes a JSON file path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(msg_dicts, f)
            tmp_path = f.name

        try:
            self._engine.ingest(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return IngestTelemetry(
            n_messages_in=len(session_messages),
            n_facts_extracted=len(session_messages),  # 1:1 — no extraction
            n_kept=len(session_messages),
            n_dropped=0,
            n_updated=0,
            n_skipped_dedup=0,
            drop_reasons={},
            extra={"path": "engine.ingest_verbatim"},
        )

    def retrieve(self, query: str, db_path: str | Path,
                 k: int = 10) -> list[RetrievalResult]:
        self._ensure_engine(db_path)
        # Paper uses search_agentic with HyDE + reranker for the Pro tier.
        # For Phase 9 local sweep we use the engine's plain hybrid search
        # (no HyDE — that requires an LLM call per query, defer to Phase 11
        # Modal for the published-numbers run).
        try:
            rows = self._engine.search(query, limit=k)
        except AttributeError:
            # Older API
            rows = self._engine.search_hybrid(query, limit=k) if hasattr(self._engine, "search_hybrid") else []

        return [
            RetrievalResult(
                content=r.get("content", "") if isinstance(r, dict) else str(r),
                score=float((r.get("score", 0.0) if isinstance(r, dict) else 0.0) or 0.0),
                metadata={"id": r.get("id") if isinstance(r, dict) else None,
                           "category": r.get("category") if isinstance(r, dict) else None},
            )
            for r in rows
        ]
