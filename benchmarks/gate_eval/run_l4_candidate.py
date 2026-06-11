"""MEMORIST-L4 gate-eval harness — Phase 8.

Runs one L4 candidate against long_horizon_synthetic.json (with L4 probes),
produces a result-JSON under benchmarks/gate_eval/results/l4_consolidation/.

Execution model:
  1. Build a fresh SQLite DB.
  2. Ingest all sessions (chronological) via the standard TrueMemory
     API — same for every candidate. Measures baseline storage bytes.
  3. Call candidate.consolidate(db_path) — measures wall-clock, bytes
     added, rows written.
  4. For each L4 probe (generalization, contradiction, rehearsal,
     near_duplicate), call candidate.retrieve_augmented(question) and
     score per the probe-type rules in L4_README.md.
  5. Emit result JSON.

Usage:
    python benchmarks/gate_eval/run_l4_candidate.py \\
        --candidate c1_baseline \\
        --seed 42

Candidates discovered from
  benchmarks/gate_eval/candidates/l4_consolidation/*.py

No network-required probe authoring. OPENAI_API_KEY optional for C2/C4;
without it they fall back to extractive / C3 respectively.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "benchmarks" / "gate_eval" / "datasets"
CANDIDATES_DIR = REPO_ROOT / "benchmarks" / "gate_eval" / "candidates" / "l4_consolidation"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "gate_eval" / "results" / "l4_consolidation"

sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gate_eval.candidates.l4_consolidation._base import (  # noqa: E402
    L4Candidate,
)


def discover_candidates() -> dict[str, type[L4Candidate]]:
    found: dict[str, type[L4Candidate]] = {}
    for py in CANDIDATES_DIR.glob("*.py"):
        if py.stem.startswith("_"):
            continue
        modname = f"benchmarks.gate_eval.candidates.l4_consolidation.{py.stem}"
        mod = importlib.import_module(modname)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and issubclass(obj, L4Candidate) and obj is not L4Candidate:
                found[obj.name] = obj
    return found


def ingest_dataset(db_path: Path, dataset: dict) -> dict:
    """Standard TrueMemory ingest of the long_horizon_synthetic sessions.

    One Memory instance; each message is an individual add() call with
    synthesized ISO timestamp from session day_offset + turn index.
    Baseline for all candidates — NOT candidate-specific.
    """
    from truememory.engine import TrueMemoryEngine

    engine = TrueMemoryEngine(db_path=db_path)

    sessions = sorted(dataset["sessions"], key=lambda s: s["session_id"])
    base_date = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)

    n_msgs = 0
    for sess in sessions:
        persona = sess["persona_name"]
        day_offset = sess.get("day_offset", sess["session_id"])
        for i, turn in enumerate(sess["transcript"]):
            content = turn.get("content", "")
            if not content or not content.strip():
                continue
            ts = base_date + datetime.timedelta(days=day_offset, minutes=i * 2)
            role = turn.get("role", "user")
            sender = persona if role == "user" else "assistant"
            recipient = "assistant" if role == "user" else persona
            engine.add(
                content=content,
                sender=sender,
                recipient=recipient,
                timestamp=ts.isoformat(),
            )
            n_msgs += 1

    # Explicitly close to flush
    try:
        engine.close()
    except Exception:
        pass

    return {"n_messages": n_msgs, "n_sessions": len(sessions)}


_CLEAN_RE = re.compile(r"[^a-z0-9]+")


def _clean(s: str) -> str:
    return _CLEAN_RE.sub(" ", s.lower()).strip()


def _contains_keywords(content: str, gold: str, min_overlap: int = 2) -> bool:
    """True if content contains ≥min_overlap of the >3-char tokens from gold."""
    gold_tokens = [t for t in _clean(gold).split() if len(t) > 3]
    if not gold_tokens:
        return False
    content_clean = _clean(content)
    overlap = sum(1 for t in gold_tokens if t in content_clean)
    return overlap >= min(min_overlap, len(gold_tokens))


def score_generalization(probe: dict, results: list) -> dict:
    """Generalization gold = list of facts. Score = fraction of gold_facts
    that appear in top-k retrieved content. 'correct' if ≥ 50% covered."""
    gold_facts = probe.get("gold_facts", [])
    if not gold_facts:
        # Fall back to gold_answer keywords
        gold_answer = probe.get("gold_answer", "")
        hit = any(_contains_keywords(r.content, gold_answer, 2) for r in results)
        return {"covered": int(hit), "total": 1, "fraction": float(hit), "correct": int(hit)}
    all_content = " ".join((r.content or "") for r in results)
    covered = sum(1 for f in gold_facts if _contains_keywords(all_content, f, 2))
    fraction = covered / len(gold_facts)
    return {
        "covered": covered,
        "total": len(gold_facts),
        "fraction": fraction,
        "correct": int(fraction >= 0.5),
    }


def score_contradiction(probe: dict, results: list) -> dict:
    """Contradiction: correct iff new_fact keywords appear in top-3 AND
    superseded_fact is not the *primary* (top-1) result."""
    gold = probe.get("gold_answer", probe.get("new_fact", ""))
    forbidden = probe.get("superseded_fact_should_NOT_be_returned", "")

    top3 = results[:3]
    top1 = results[:1]
    new_in_top3 = any(_contains_keywords(r.content, gold, 2) for r in top3) if gold else False
    forbidden_in_top1 = (
        any(_contains_keywords(r.content, forbidden, 2) for r in top1)
        if forbidden else False
    )
    correct = new_in_top3 and not forbidden_in_top1
    return {
        "new_in_top3": int(new_in_top3),
        "forbidden_in_top1": int(forbidden_in_top1),
        "correct": int(correct),
    }


def score_rehearsal(probe: dict, results: list) -> dict:
    """Rehearsal: correct iff gold_answer keywords in top-3."""
    gold = probe.get("gold_answer", "")
    hit = any(_contains_keywords(r.content, gold, 2) for r in results[:3])
    return {"correct": int(hit), "N": probe.get("N", 1)}


def score_near_duplicate(probe: dict, results: list) -> dict:
    """Near-duplicate: correct iff gold_answer keywords in top-k."""
    gold = probe.get("gold_answer", "")
    hit = any(_contains_keywords(r.content, gold, 2) for r in results[:10])
    return {"correct": int(hit)}


SCORERS = {
    "generalization": score_generalization,
    "contradiction": score_contradiction,
    "rehearsal": score_rehearsal,
    "near_duplicate": score_near_duplicate,
}


def run_l4_probes(candidate: L4Candidate, db_path: Path, l4_probes: dict,
                  k: int = 10) -> dict:
    """Run all L4 probe types, compute per-type metrics."""
    out: dict[str, dict] = {}
    for ptype, scorer in SCORERS.items():
        probes = l4_probes.get(ptype, [])
        per_probe = []
        for p in probes:
            try:
                results = candidate.retrieve_augmented(p["question"], db_path, k=k)
            except Exception as e:
                per_probe.append({"probe_id": p["probe_id"], "error": str(e)[:100],
                                  "correct": 0})
                continue
            score = scorer(p, results)
            per_probe.append({"probe_id": p["probe_id"], **score})
        if not per_probe:
            out[ptype] = {"n": 0, "accuracy": 0.0, "per_probe": []}
            continue
        n_correct = sum(s.get("correct", 0) for s in per_probe)
        out[ptype] = {
            "n": len(per_probe),
            "n_correct": n_correct,
            "accuracy": round(n_correct / len(per_probe), 4),
            "per_probe": per_probe,
        }

    # Rehearsal correlation: accuracy by N bucket
    reh = out.get("rehearsal", {}).get("per_probe", [])
    by_n: dict[int, list[int]] = {}
    for s in reh:
        by_n.setdefault(s.get("N", 1), []).append(s.get("correct", 0))
    n_buckets = {}
    for N, hits in by_n.items():
        n_buckets[str(N)] = {
            "n": len(hits),
            "accuracy": round(sum(hits) / max(len(hits), 1), 4),
        }
    out["rehearsal"]["by_N"] = n_buckets

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--dataset", default="long_horizon_synthetic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--output", default=None)
    ap.add_argument("--skip-ingest-cache", action="store_true",
                    help="Always rebuild DB from scratch (default: reuse cache if present)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = discover_candidates()
    if args.candidate not in candidates:
        print(f"Unknown candidate: {args.candidate}", file=sys.stderr)
        print(f"Available: {sorted(candidates.keys())}", file=sys.stderr)
        sys.exit(1)

    dataset_path = DATASETS_DIR / f"{args.dataset}.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    l4_probes = dataset.get("l4_probes")
    if not l4_probes:
        print(f"ERROR: dataset {args.dataset} has no l4_probes key. "
              f"Run benchmarks/gate_eval/datasets/build_l4_probes.py first.",
              file=sys.stderr)
        sys.exit(1)

    cand_cls = candidates[args.candidate]
    cand = cand_cls(seed=args.seed)

    # DB caching: same dataset + same ingest code produces the same baseline DB.
    # We cache it under /tmp/l4_baseline_<hash>.db and copy for each run.
    cache_key = f"l4_baseline_{args.dataset}_{args.seed}"
    cache_db = Path(tempfile.gettempdir()) / f"{cache_key}.db"

    with tempfile.TemporaryDirectory(prefix=f"l4_{args.candidate}_") as td:
        db_path = Path(td) / "truememory.db"

        if cache_db.exists() and not args.skip_ingest_cache:
            import shutil
            shutil.copy2(cache_db, db_path)
            ingest_info = {"cached": True, "cache_path": str(cache_db)}
            ingest_wall = 0.0
        else:
            ingest_t0 = time.time()
            ingest_info = ingest_dataset(db_path, dataset)
            ingest_wall = time.time() - ingest_t0
            try:
                import shutil
                shutil.copy2(db_path, cache_db)
            except Exception:
                pass

        baseline_bytes = db_path.stat().st_size

        print(f"[{args.candidate}] baseline storage: {baseline_bytes:,} bytes")

        # Run candidate consolidation
        try:
            tel = cand.consolidate(db_path)
        except Exception as e:
            import traceback
            print(f"CONSOLIDATE FAILED: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(2)

        print(f"[{args.candidate}] consolidate wall-clock: {tel.wall_clock_s:.2f}s, "
              f"+{tel.bytes_added:,} bytes, rows={tel.rows_written}")

        # Run L4 probes
        probe_t0 = time.time()
        probe_results = run_l4_probes(cand, db_path, l4_probes, k=args.k)
        probe_wall = time.time() - probe_t0

        result = {
            "meta": {
                "candidate": args.candidate,
                "dataset": args.dataset,
                "seed": args.seed,
                "k": args.k,
                "ingest_ms": round(ingest_wall * 1000, 1),
                "consolidation_wall_clock_s": round(tel.wall_clock_s, 2),
                "baseline_storage_bytes": baseline_bytes,
                "bytes_added_by_consolidation": tel.bytes_added,
                "storage_multiplier": round(
                    (baseline_bytes + tel.bytes_added) / max(baseline_bytes, 1), 3),
                "rows_written": tel.rows_written,
                "notes": tel.notes,
                "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "probe_wall_clock_s": round(probe_wall, 2),
                "ingest_cached": ingest_info.get("cached", False),
            },
            "probes": probe_results,
        }

        # Aggregate headline metrics
        result["headline"] = {
            "generalization_accuracy": probe_results.get("generalization", {}).get("accuracy", 0),
            "contradiction_accuracy": probe_results.get("contradiction", {}).get("accuracy", 0),
            "rehearsal_accuracy": probe_results.get("rehearsal", {}).get("accuracy", 0),
            "near_duplicate_accuracy": probe_results.get("near_duplicate", {}).get("accuracy", 0),
            "J_composite_unweighted": round(
                probe_results.get("generalization", {}).get("accuracy", 0)
                * probe_results.get("contradiction", {}).get("accuracy", 0),
                4,
            ),
        }

        # Write result
        output = args.output or str(
            RESULTS_DIR / f"{args.candidate}__{args.dataset}__seed{args.seed}.json"
        )
        Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[{args.candidate}] wrote {output}")
        print(f"[{args.candidate}] headline: {result['headline']}")

    try:
        cand.cleanup()
    except Exception:
        pass


if __name__ == "__main__":
    main()
