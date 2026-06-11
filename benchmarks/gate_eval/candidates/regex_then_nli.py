"""Candidate #4 — regex_then_nli (two-stage encoding gate).

Phase 14 (addendum) implementation. Promotes the prototype at
`/tmp/two_stage_gate_prototype.py` to the harness Candidate interface.

Architecture:
  - Stage 1: regex prefilter (free, microseconds) drops obvious chit-chat.
  - Stage 2: zero-shot NLI classifier (HF, 10-30ms CPU) scores
    p_substantive on remaining messages; drops below threshold.
  - Survivors are stored VERBATIM via engine.ingest() (the same path
    v05_paper_verbatim uses), so the gate's effect is isolated from
    extraction-pipeline confounds.

Why verbatim storage:
  The Phase 14 addendum wants to measure storage reduction and
  chitchat-drop-rate at the gate level, not the extraction level.
  Wiring the gate into IngestionPipeline would mix gate decisions with
  LLM extraction decisions and confound the measurement. Verbatim
  storage isolates the gate's contribution.

Parameters:
  - nli_model: HF model ID for the Stage 2 classifier
  - threshold: float in [0, 1], p_substantive cutoff
  - regex_profile: "off" / "standard" / "aggressive"
    - "off"        — Stage 1 disabled; everything goes to Stage 2 (= single_stage_nli)
    - "standard"   — _CHITCHAT_EXACT + _CHITCHAT_REGEX from prototype
    - "aggressive" — adds short-msg drops + filler-phrase drops

Telemetry exposed via IngestTelemetry.extra:
  - drop_rate_by_stage: {"stage1_regex": N, "stage2_nli": N, "kept": N}
  - p_substantive_distribution: histogram over the NLI scores observed
  - per-message decisions: list of {text, p_substantive, decision, stage}
    (truncated to 200 entries to keep memory bounded)

The shared `_nli_score_cache` ensures that re-running the same
(model × regex_profile × message) tuple at a different threshold is
free — Stage 2 inference is deterministic so we can sweep thresholds
without re-inferencing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.gate_eval.candidates._base import Candidate, IngestTelemetry, RetrievalResult


# ---------------------------------------------------------------------------
# Stage 1 — regex prefilter (extends prototype's set with aggressive profile)
# ---------------------------------------------------------------------------

_CHITCHAT_EXACT = {
    "lol", "lmao", "rofl", "haha", "hehe", "haha!",
    "ok", "okay", "k", "kk",
    "yeah", "yep", "yup", "yea", "yes",
    "no", "nope", "nah",
    "sure", "sure thing", "cool", "nice", "nice!",
    "thanks", "thank you", "ty", "thx",
    "np", "you're welcome", "welcome",
    "hi", "hey", "hello", "yo",
    "bye", "goodbye", "gn", "good night", "good morning",
    "wow", "omg", "damn", "dang",
    "same", "yeah same", "same here",
    "got it", "understood", "noted",
    "sounds good", "sounds great",
}

_CHITCHAT_REGEX = re.compile(
    r"^(lo+l|ha+|he+he*|lmao+|ikr+|rofl|wow+|omg+|hmm+|uh+|ah+)[!.?]*$",
    re.IGNORECASE,
)
_PUNCT_ONLY = re.compile(r"^[\W_]+$")

# Aggressive profile additions
_FILLER_PHRASES = re.compile(
    r"^(thanks for|thank you for|appreciate|hope you|happy (monday|tuesday|wed|thurs|fri|sat|sun))\b",
    re.IGNORECASE,
)


def stage1_decision(text: str, profile: str = "standard") -> tuple[bool, str]:
    """Returns (is_chitchat, reason)."""
    if profile == "off":
        return False, ""
    s = (text or "").strip()
    if not s:
        return True, "empty"
    if len(s) < 4:
        return True, "too-short"
    lower = s.lower()
    if lower in _CHITCHAT_EXACT:
        return True, "chitchat-exact"
    if _CHITCHAT_REGEX.fullmatch(lower):
        return True, "chitchat-regex"
    if _PUNCT_ONLY.fullmatch(s):
        return True, "punct-only"
    if profile == "aggressive":
        if len(s) < 12 and lower not in _CHITCHAT_EXACT:
            # Short messages of unknown type — likely conversational fluff
            return True, "aggressive-short"
        if _FILLER_PHRASES.search(s):
            return True, "aggressive-filler"
    return False, ""


# ---------------------------------------------------------------------------
# Stage 2 — NLI classifier with shared on-disk score cache
# ---------------------------------------------------------------------------

_CANDIDATE_LABELS = [
    "substantive personal information, preference, decision, or factual statement",
    "conversational filler, pleasantry, acknowledgment, or small talk",
]
_HYPOTHESIS_TEMPLATE = "This message is {}."

NLI_CACHE_DIR = Path.home() / ".cache" / "memorist_nli_scores"


def _nli_cache_path(model_name: str) -> Path:
    safe = model_name.replace("/", "_")
    return NLI_CACHE_DIR / f"{safe}.jsonl"


_NLI_PIPELINES: dict[str, object] = {}  # model_name → loaded pipeline (process-level)


def _ensure_nli_pipeline(model_name: str, device: str | int = -1):
    if model_name in _NLI_PIPELINES:
        return _NLI_PIPELINES[model_name]
    from transformers import pipeline
    print(f"  [nli] loading {model_name} ...", flush=True)
    clf = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )
    _NLI_PIPELINES[model_name] = clf
    return clf


def _msg_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


def stage2_score(text: str, model_name: str, device: str | int = -1) -> float:
    """Returns p_substantive (probability the message is substantive).

    Uses an on-disk JSONL cache keyed on (model_name, msg_hash) so repeat
    queries are free. Cache is append-only — first run for a (model,
    msg) pair pays the inference cost; subsequent runs at any threshold
    look up.
    """
    NLI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _nli_cache_path(model_name)
    msg_hash = _msg_hash(text)

    # Check in-memory cache first
    if not hasattr(stage2_score, "_loaded_caches"):
        stage2_score._loaded_caches = {}
    if model_name not in stage2_score._loaded_caches:
        cache = {}
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        cache[rec["h"]] = rec["p"]
                    except Exception:
                        continue
        stage2_score._loaded_caches[model_name] = cache

    cache = stage2_score._loaded_caches[model_name]
    if msg_hash in cache:
        return cache[msg_hash]

    # Cache miss — run inference
    clf = _ensure_nli_pipeline(model_name, device=device)
    result = clf(
        text,
        candidate_labels=_CANDIDATE_LABELS,
        hypothesis_template=_HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    top_label = result["labels"][0]
    top_score = float(result["scores"][0])
    if top_label == _CANDIDATE_LABELS[0]:
        p_subst = top_score
    else:
        p_subst = 1.0 - top_score

    cache[msg_hash] = p_subst
    # Append to disk
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"h": msg_hash, "p": p_subst}) + "\n")

    return p_subst


def cache_stats() -> dict:
    """Return per-model cache entry counts."""
    NLI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for jf in NLI_CACHE_DIR.glob("*.jsonl"):
        try:
            with open(jf) as f:
                n = sum(1 for line in f if line.strip())
            out[jf.stem] = n
        except OSError:
            continue
    return out


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "MoritzLaurer/roberta-base-zeroshot-v2.0-c"


class RegexThenNli(Candidate):
    name = "regex_then_nli"

    def __init__(
        self,
        nli_model: str = DEFAULT_MODEL,
        threshold: float = 0.55,
        regex_profile: str = "standard",
        device: str | int = -1,
        max_per_msg_records: int = 200,
        **kwargs,
    ):
        super().__init__(
            nli_model=nli_model, threshold=threshold,
            regex_profile=regex_profile, **kwargs,
        )
        self.nli_model = nli_model
        self.threshold = float(threshold)
        self.regex_profile = regex_profile
        self.device = device
        self.max_per_msg_records = max_per_msg_records
        self._engine = None
        # Per-run accumulators for explainability
        self._per_msg_records: list[dict] = []
        self._stage1_drop_reasons: Counter = Counter()
        self._stage2_p_hist: list[float] = []  # all p_substantive values seen at Stage 2

    def _ensure_engine(self, db_path: str | Path):
        if self._engine is not None:
            return
        from truememory.engine import TrueMemoryEngine
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self._engine = TrueMemoryEngine(db_path=str(db_path))

    def evaluate_message(self, text: str) -> tuple[bool, str, float]:
        """Apply both stages. Returns (keep, stage_label, p_substantive)."""
        s1_drop, s1_reason = stage1_decision(text, profile=self.regex_profile)
        if s1_drop:
            self._stage1_drop_reasons[s1_reason] += 1
            return False, f"stage1:{s1_reason}", 0.0
        # Stage 2
        p = stage2_score(text, self.nli_model, device=self.device)
        self._stage2_p_hist.append(p)
        keep = p >= self.threshold
        stage = f"stage2:p={p:.3f}"
        return keep, stage, p

    def ingest(self, session_messages: list[dict], db_path: str | Path,
               session_id: int = 0) -> IngestTelemetry:
        self._ensure_engine(db_path)

        n_in = len(session_messages)
        n_kept = 0
        kept_dicts: list[dict] = []
        per_msg_decisions: list[dict] = []

        t0 = time.perf_counter()
        for i, m in enumerate(session_messages):
            text = m.get("content", "")
            t_msg0 = time.perf_counter()
            keep, stage, p = self.evaluate_message(text)
            elapsed_ms = (time.perf_counter() - t_msg0) * 1000.0

            if len(self._per_msg_records) < self.max_per_msg_records:
                self._per_msg_records.append({
                    "session_id": session_id,
                    "msg_idx": i,
                    "text": text,
                    "p_substantive": p,
                    "decision": "KEEP" if keep else "DROP",
                    "stage": stage,
                    "elapsed_ms": round(elapsed_ms, 2),
                })

            if keep:
                n_kept += 1
                kept_dicts.append({
                    "content": text,
                    "sender": m.get("role", "user"),
                    "recipient": "self",
                    "timestamp": m.get("timestamp", f"session_{session_id}_{i:04d}"),
                    "category": str(session_id),
                    "modality": "conversation",
                })

        ingest_wall = time.perf_counter() - t0

        # Bulk-write survivors verbatim via engine.ingest (json file path)
        if kept_dicts:
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(kept_dicts, f)
                tmp_path = f.name
            try:
                self._engine.ingest(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        n_dropped = n_in - n_kept

        return IngestTelemetry(
            n_messages_in=n_in,
            n_facts_extracted=n_in,  # no extraction here
            n_kept=n_kept,
            n_dropped=n_dropped,
            n_updated=0,
            n_skipped_dedup=0,
            drop_reasons={
                "stage1_total": sum(self._stage1_drop_reasons.values()),
                "stage2_dropped": n_dropped - sum(self._stage1_drop_reasons.values()),
                **{f"s1:{k}": v for k, v in self._stage1_drop_reasons.items()},
            },
            extra={
                "nli_model": self.nli_model,
                "threshold": self.threshold,
                "regex_profile": self.regex_profile,
                "ingest_wall_clock_s_session": round(ingest_wall, 3),
                "n_stage2_inferences_session": len(self._stage2_p_hist),
            },
        )

    def retrieve(self, query: str, db_path: str | Path,
                 k: int = 10) -> list[RetrievalResult]:
        self._ensure_engine(db_path)
        rows = self._engine.search(query, limit=k)
        return [
            RetrievalResult(
                content=r.get("content", "") if isinstance(r, dict) else str(r),
                score=float((r.get("score", 0.0) if isinstance(r, dict) else 0.0) or 0.0),
                metadata={"id": r.get("id") if isinstance(r, dict) else None,
                          "category": r.get("category") if isinstance(r, dict) else None},
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Explainability writers
    # ------------------------------------------------------------------

    def export_run_telemetry(self) -> dict:
        """Aggregate run-level telemetry for the harness to embed in result JSON."""
        from statistics import mean, median
        p_hist = self._stage2_p_hist
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist = defaultdict(int)
        for p in p_hist:
            for i in range(len(bins) - 1):
                if bins[i] <= p < bins[i + 1]:
                    hist[f"[{bins[i]:.1f},{bins[i+1]:.1f})"] += 1
                    break
            else:
                if p >= 1.0:
                    hist["[1.0,1.0]"] += 1
        return {
            "stage1_drop_reasons": dict(self._stage1_drop_reasons),
            "n_stage2_inferences": len(p_hist),
            "p_substantive_min": min(p_hist) if p_hist else None,
            "p_substantive_mean": round(mean(p_hist), 4) if p_hist else None,
            "p_substantive_median": round(median(p_hist), 4) if p_hist else None,
            "p_substantive_max": max(p_hist) if p_hist else None,
            "p_substantive_histogram": dict(hist),
            "per_message_records_truncated_to": len(self._per_msg_records),
        }

    def write_samples_md(self, out_path: Path, n_kept: int = 20,
                         n_dropped_s1: int = 20, n_dropped_s2: int = 20,
                         n_borderline: int = 10) -> Path:
        """Render per-cell sample explainability page."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        kept = [r for r in self._per_msg_records if r["decision"] == "KEEP"]
        d_s1 = [r for r in self._per_msg_records if r["decision"] == "DROP" and r["stage"].startswith("stage1")]
        d_s2 = [r for r in self._per_msg_records if r["decision"] == "DROP" and r["stage"].startswith("stage2")]
        # Borderline: stage2 messages within ±0.05 of threshold
        borderline = [
            r for r in self._per_msg_records
            if r["stage"].startswith("stage2")
            and abs(r["p_substantive"] - self.threshold) < 0.05
        ]

        def _table(rows: list[dict], header: str) -> list[str]:
            lines = [f"### {header}", "", "| p_subst | decision | stage | text |", "|---:|---|---|---|"]
            for r in rows:
                txt = (r["text"] or "").replace("|", "\\|").replace("\n", " ")[:140]
                lines.append(f"| {r['p_substantive']:.3f} | {r['decision']} | {r['stage']} | {txt} |")
            lines.append("")
            return lines

        out = [
            f"# Two-Stage Gate samples — `{self.nli_model}` τ={self.threshold} regex={self.regex_profile}",
            "",
            f"NLI model: `{self.nli_model}`. Threshold: {self.threshold}. Regex profile: `{self.regex_profile}`.",
            "",
            "Per-cell explainability samples per Phase 14 addendum spec §3.",
            "",
        ]
        out += _table(kept[:n_kept], f"Stage-2 KEPT (sample of {min(n_kept, len(kept))} of {len(kept)} kept)")
        out += _table(d_s1[:n_dropped_s1], f"Stage-1 regex DROPPED (sample of {min(n_dropped_s1, len(d_s1))} of {len(d_s1)} dropped by regex)")
        out += _table(d_s2[:n_dropped_s2], f"Stage-2 NLI DROPPED (sample of {min(n_dropped_s2, len(d_s2))} of {len(d_s2)} dropped by NLI)")
        out += _table(borderline[:n_borderline], f"Borderline (within ±0.05 of threshold; sample of {min(n_borderline, len(borderline))} of {len(borderline)})")

        # Stage-1 reason distribution
        if self._stage1_drop_reasons:
            out += ["### Stage-1 regex pattern hit distribution", "", "| reason | count |", "|---|---:|"]
            for reason, n in self._stage1_drop_reasons.most_common():
                out.append(f"| {reason} | {n} |")
            out.append("")

        out_path.write_text("\n".join(out), encoding="utf-8")
        return out_path
