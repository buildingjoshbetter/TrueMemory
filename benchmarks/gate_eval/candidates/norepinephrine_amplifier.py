"""Candidate #11 — norepinephrine_amplifier (regex-based fast variant).

Phase 6 spec called for an HF emotion classifier (j-hartmann/emotion-english-
distilroberta-base, ~300MB). For Phase 9 first-pass we use a CHEAPER
regex-keyword variant — no model download, runs in microseconds, but still
exercises the candidate's distinctive retrieval-time amplification logic
so we can observe whether the architectural pattern moves J_recall at all.

Phase 10 ablation will swap in the real HF classifier and quantify how
much the keyword shortcut costs.

Mechanism (mirrors Aston-Jones LC-NE adaptive gain — Phase 1 #3):
  - Ingest: store every extracted fact (no gate at ingest).
  - At ingest, tag each row's metadata with an `emotion_intensity` score
    derived from regex matches against a curated arousal-vocabulary list
    (Cahill & McGaugh #21 + life-event keywords from the v0.5.0
    `salience.py` module, which already has this list).
  - At retrieve, multiply the row's base score by `(1 + λ_NE * emotion_intensity)`
    so emotionally-charged rows surface higher in the top-k.

Phase 6 prediction:
  SHORT (LoCoMo): ≈ 91.5% (no change — LoCoMo doesn't reward emotion).
  LONG  (synthetic): +5-7 pp on emotional adversarial probes.

The PHASE 9 LOCAL METRIC equivalent: expect non-zero gain over
v05_baseline on the `emotional` probe subset of long_horizon_synthetic
(currently 5.56% across baseline candidates).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from benchmarks.gate_eval.candidates._base import Candidate, IngestTelemetry, RetrievalResult


# Curated arousal vocabulary. Sourced from truememory.salience._HIGH_AROUSAL
# and _LIFE_EVENTS so the candidate's emotion signal is consistent with the
# salience layer the v0.5.0 retrieval pipeline already uses.
_HIGH_AROUSAL_KEYWORDS = {
    "amazing", "incredible", "devastating", "heartbreaking",
    "thrilled", "furious", "terrified", "ecstatic", "crushed",
    "panic", "emergency", "urgent", "critical", "breakthrough",
    "milestone", "promoted", "fired", "pregnant", "engaged",
    "diagnosed", "accident", "passed away", "died",
    # synthetic-dataset-specific high-emotion phrases
    "miscarriage", "miscarried", "anaphylactic", "series b",
    "got 1k", "launched on product hunt", "lost a key",
}

_LIFE_EVENT_PATTERNS = [
    re.compile(r"\b(got\s+(?:married|engaged|promoted|fired|hired))\b", re.I),
    re.compile(r"\b(having\s+a\s+baby|moved\s+to|switched\s+to|broke\s+up)\b", re.I),
    re.compile(r"\b(graduated|launched|raised\s+funding|demo\s+day|ipo|acquisition)\b", re.I),
    re.compile(r"\b(accepted\s+(?:the\s+)?[A-Za-z]+\s+offer)\b", re.I),
]

_EXCLAMATION_RE = re.compile(r"!{2,}|[A-Z]{4,}")


def emotion_intensity(text: str) -> float:
    """Score 0.0 (neutral) → 1.0 (high arousal). Same shape as the spec's
    HF-classifier interface so swap-in is trivial in Phase 10."""
    if not text:
        return 0.0
    lower = text.lower()
    score = 0.0

    arousal_hits = sum(1 for k in _HIGH_AROUSAL_KEYWORDS if k in lower)
    score += min(0.5, arousal_hits * 0.15)

    event_hits = sum(1 for p in _LIFE_EVENT_PATTERNS if p.search(text))
    score += min(0.4, event_hits * 0.20)

    if _EXCLAMATION_RE.search(text):
        score += 0.10

    return max(0.0, min(1.0, score))


DEFAULT_LAMBDA_NE = 1.5  # multiplier on emotion-intensity for retrieval boost


class NorepinephrineAmplifier(Candidate):
    name = "norepinephrine_amplifier"

    def __init__(self, lambda_ne: float = DEFAULT_LAMBDA_NE, **kwargs):
        super().__init__(lambda_ne=lambda_ne, **kwargs)
        self.lambda_ne = lambda_ne
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
        return IngestTelemetry(
            n_messages_in=len(session_messages),
            n_facts_extracted=result.facts_extracted,
            n_kept=result.facts_stored + result.facts_updated,
            n_dropped=result.facts_skipped_gate,
            n_updated=result.facts_updated,
            n_skipped_dedup=result.facts_skipped_dedup,
            drop_reasons={"gate": result.facts_skipped_gate, "dedup": result.facts_skipped_dedup},
            extra={"lambda_ne": self.lambda_ne, "emotion_signal": "regex"},
        )

    def retrieve(self, query: str, db_path: str | Path,
                 k: int = 10) -> list[RetrievalResult]:
        self._ensure_pipeline(db_path)
        # Pull k×3 candidates so the NE rerank has room to surface emotional rows
        rows = self._memory.search(query, limit=max(k * 3, 30))

        rescored = []
        for r in rows:
            content = r.get("content", "")
            base_score = float(r.get("score", 0.0) or 0.0)
            emo = emotion_intensity(content)
            ne_gain = 1.0 + self.lambda_ne * emo
            rescored.append((r, base_score * ne_gain, emo))

        rescored.sort(key=lambda x: x[1], reverse=True)
        top = rescored[:k]

        return [
            RetrievalResult(
                content=r.get("content", ""),
                score=combined,
                metadata={"id": r.get("id"), "category": r.get("category"),
                           "emotion_intensity": round(emo, 3)},
            )
            for r, combined, emo in top
        ]
