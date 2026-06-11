"""MEMORIST gate-eval harness — Phase 8.

Uniform driver that runs one candidate × one dataset and emits a
result-JSON in the schema specified by MEMORIST_SPEC.md Phase 8.

Usage:
    python benchmarks/gate_eval/run_candidate.py \\
        --candidate v05_baseline_nogate \\
        --dataset short_horizon_200 \\
        --output benchmarks/gate_eval/results/v05_baseline_nogate__short_horizon_200.json

Datasets:
    short_horizon_200          → benchmarks/gate_eval/datasets/short_horizon_200.json
    long_horizon_synthetic     → benchmarks/gate_eval/datasets/long_horizon_synthetic.json

Candidates:
    Discovered by scanning benchmarks/gate_eval/candidates/*.py for any
    class whose `name` matches the --candidate argument.

Per Phase 0 / Phase 6, the answer model + judge protocol are held FIXED
across all (candidate × dataset) runs:
    answer_model: openai/gpt-4.1-mini, temperature 0
    judge_model:  openai/gpt-4o-mini majority-of-3, temperature 0

For local sweeps (Phase 9) we report retrieval-precision-style metrics
that don't require the answer-model API call (precision@k against the
gold-evidence message ids for short_horizon_200; against the planted
fact text for long_horizon_synthetic). The full answer-model + judge
loop is reserved for Phase 11 Modal runs.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "benchmarks" / "gate_eval" / "datasets"
CANDIDATES_DIR = REPO_ROOT / "benchmarks" / "gate_eval" / "candidates"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "gate_eval" / "results"

sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gate_eval.candidates._base import Candidate  # noqa: E402


# Install the shared extraction cache as soon as the harness loads. All
# candidates that use truememory.ingest.extract_facts (most of them) will
# transparently hit cached results when re-running. Set MEMORIST_NO_CACHE=1
# to disable.
if not os.environ.get("MEMORIST_NO_CACHE"):
    try:
        from benchmarks.gate_eval import _extract_cache
        _extract_cache.install()
    except Exception as _e:
        print(f"  warning: extract-cache install failed: {_e}", file=sys.stderr)


def discover_candidates() -> dict[str, type[Candidate]]:
    """Walk candidates/*.py and return a {name: cls} map."""
    found: dict[str, type[Candidate]] = {}
    for py in CANDIDATES_DIR.glob("*.py"):
        if py.stem.startswith("_"):
            continue
        modname = f"benchmarks.gate_eval.candidates.{py.stem}"
        mod = importlib.import_module(modname)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and issubclass(obj, Candidate) and obj is not Candidate:
                found[obj.name] = obj
    return found


def load_dataset(name: str) -> dict:
    path = DATASETS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def db_size_bytes(db_path: Path) -> int:
    try:
        return db_path.stat().st_size
    except OSError:
        return 0


def db_row_count(db_path: Path, table: str) -> int:
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
            return cur.fetchone()[0]
    except sqlite3.Error:
        return 0


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def run_short_horizon(candidate: Candidate, dataset: dict, db_path: Path,
                       skip_ingest: bool = False) -> dict:
    """Run candidate against short_horizon_200.

    Ingest each LoCoMo conversation as one big "session", then query each
    of the 200 questions and compute precision@k against the evidence-
    message ids (heuristic — Mem0/Zep-style extraction loses id linkage,
    so we score on whether the EVIDENCE TEXT appears in the top-k content).
    """
    convs = {c["conv_idx"]: c for c in dataset["convs"]}

    # ---- Ingest phase ----
    ingest_t0 = time.time()
    per_msg_times: list[float] = []
    ingest_summary = {"sessions": 0, "messages_in": 0, "kept": 0, "dropped": 0}

    if skip_ingest:
        # Re-running after a fix; the DB is already populated. Mark all
        # ingest counters as -1 (sentinel: "not measured this run").
        ingest_summary = {"sessions": -1, "messages_in": -1, "kept": -1, "dropped": -1}
    else:
        for conv in dataset["convs"]:
            # Flatten LoCoMo's session_X structure into one message stream
            sessions: list[dict] = []
            cdata = conv["conversation"]
            for k in sorted(cdata.keys()):
                if k.startswith("session_") and not k.endswith("_date_time"):
                    for turn in cdata[k]:
                        text = turn.get("text", "")
                        speaker = turn.get("speaker", "human")
                        role = "user" if speaker.lower() in ("user", "human") else "assistant"
                        sessions.append({"role": role, "content": text})

            msg_t0 = time.time()
            tel = candidate.ingest(sessions, db_path=db_path, session_id=conv["conv_idx"])
            per_msg_times.append((time.time() - msg_t0) / max(len(sessions), 1))
            ingest_summary["sessions"] += 1
            ingest_summary["messages_in"] += tel.n_messages_in
            ingest_summary["kept"] += tel.n_kept
            ingest_summary["dropped"] += tel.n_dropped

    ingest_wall = time.time() - ingest_t0

    # ---- Retrieval phase ----
    retr_t0 = time.time()
    n_total = 0
    # Track precision at multiple k thresholds so retrieve-time reranking
    # candidates that don't change top-k membership but DO change top-k
    # ordering still surface a measurable signal.
    n_at = {1: 0, 3: 0, 10: 0}
    by_category = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}
    K = 10

    for qa in dataset["qa"]:
        n_total += 1
        cat = qa.get("category", 0)
        if cat in by_category:
            by_category[cat][1] += 1

        try:
            results = candidate.retrieve(qa["question"], db_path=db_path, k=K)
        except Exception as e:
            print(f"  retrieve failed for q{n_total}: {e}", file=sys.stderr)
            continue

        # Coerce to string — LoCoMo has a small number of int answers (e.g. years)
        raw_answer = qa.get("answer")
        gold_answer = ("" if raw_answer is None else str(raw_answer)).strip().lower()
        if not gold_answer:
            continue

        # Compute hit-position so we can score @1, @3, @10 simultaneously.
        # Use the same punctuation-stripping clean function so the metric
        # is consistent between datasets.
        import re as _re
        _clean = lambda s: _re.sub(r"[^a-z0-9 ]+", "", s.lower())
        gold_clean = _clean(gold_answer)
        hit_position = -1
        for i, r in enumerate(results[:K]):
            if gold_clean in _clean(r.content or ""):
                hit_position = i + 1
                break

        if hit_position > 0:
            n_at[10] += 1
            if hit_position <= 3:
                n_at[3] += 1
            if hit_position == 1:
                n_at[1] += 1
            if cat in by_category:
                by_category[cat][0] += 1
    n_evidence_recovered = n_at[10]

    retr_wall = time.time() - retr_t0

    return {
        "ingest": {
            "total_sessions": ingest_summary["sessions"],
            "total_messages": ingest_summary["messages_in"],
            "kept": ingest_summary["kept"],
            "dropped": ingest_summary["dropped"],
            "drop_rate_pct": round(100 * ingest_summary["dropped"] / max(ingest_summary["messages_in"], 1), 2),
            "ingest_wall_clock_s": round(ingest_wall, 2),
            "per_message_ms_avg": round(1000 * sum(per_msg_times) / max(len(per_msg_times), 1), 2),
        },
        "retrieval": {
            "total_qs": n_total,
            "gold_in_topk": n_evidence_recovered,
            "precision_at_k_pct": round(100 * n_evidence_recovered / max(n_total, 1), 2),
            "precision_at_1_pct": round(100 * n_at[1] / max(n_total, 1), 2),
            "precision_at_3_pct": round(100 * n_at[3] / max(n_total, 1), 2),
            "precision_at_10_pct": round(100 * n_at[10] / max(n_total, 1), 2),
            "k": K,
            "by_category": {
                str(c): {
                    "recovered": v[0],
                    "total": v[1],
                    "pct": round(100 * v[0] / max(v[1], 1), 2),
                }
                for c, v in by_category.items()
            },
            "wall_clock_s": round(retr_wall, 2),
        },
        "storage": {
            "db_size_bytes": db_size_bytes(db_path),
            "rows_in_messages": db_row_count(db_path, "messages"),
        },
    }


def run_long_horizon(candidate: Candidate, dataset: dict, db_path: Path,
                      skip_ingest: bool = False) -> dict:
    """Run candidate against long_horizon_synthetic.

    Process sessions in chronological order; after each session, run any
    queries that target a session ≤ current session. Score on whether the
    gold answer text appears in the top-k retrieved content.
    """
    sessions_by_id = {s["session_id"]: s for s in dataset["sessions"]}
    sorted_sids = sorted(sessions_by_id.keys())

    queries_by_after = {}
    for q in dataset["retrieval_queries"]:
        queries_by_after.setdefault(q["asked_after_session"], []).append(q)

    ingest_t0 = time.time()
    ingest_summary = {"sessions": 0, "messages_in": 0, "kept": 0, "dropped": 0}
    per_msg_times: list[float] = []
    K = 10
    by_probe = {}  # probe_type → [recovered, total]
    by_gap = {}    # time_gap_days bucket → [recovered, total]
    n_total = 0
    n_recovered = 0
    n_at = {1: 0, 3: 0, 10: 0}

    for sid in sorted_sids:
        sess = sessions_by_id[sid]
        if not skip_ingest:
            msg_t0 = time.time()
            tel = candidate.ingest(sess["transcript"], db_path=db_path, session_id=sid)
            per_msg_times.append((time.time() - msg_t0) / max(len(sess["transcript"]), 1))
            ingest_summary["sessions"] += 1
            ingest_summary["messages_in"] += tel.n_messages_in
            ingest_summary["kept"] += tel.n_kept
            ingest_summary["dropped"] += tel.n_dropped

        # Run queries scheduled to fire "after" this session
        for q in queries_by_after.get(sid, []):
            n_total += 1
            probe = q.get("probe_type", "baseline")
            gap = q.get("time_gap_days", 0)
            by_probe.setdefault(probe, [0, 0])[1] += 1
            by_gap.setdefault(gap, [0, 0])[1] += 1

            try:
                results = candidate.retrieve(q["question"], db_path=db_path, k=K)
            except Exception as e:
                print(f"  retrieve failed for q{q.get('query_id', '?')}: {e}", file=sys.stderr)
                continue
            raw_gold = q.get("gold_answer")
            gold = ("" if raw_gold is None else str(raw_gold)).strip().lower()
            if not gold:
                continue

            # For noise probes, the gold answer is N/A — being NOT recovered is success
            if probe == "noise":
                recovered_text = any(
                    "n/a" not in (r.content or "").lower() and gold[:20] in (r.content or "").lower()
                    for r in results
                )
                # success = NOT recovered (count as @1/@3/@10 success uniformly)
                if not recovered_text:
                    n_recovered += 1
                    n_at[1] += 1
                    n_at[3] += 1
                    n_at[10] += 1
                    by_probe[probe][0] += 1
                    by_gap[gap][0] += 1
            else:
                # For positive probes, find the BEST hit position so we can
                # score @1, @3, @10 simultaneously. Strip punctuation from
                # keywords so "(anaphylactic)" matches "anaphylactic" and
                # "allergies:" matches "allergy" (close but not exact —
                # but at least we don't fail on colon vs no-colon).
                import re as _re
                _clean = lambda s: _re.sub(r"[^a-z0-9]+", "", s.lower())
                fact_keywords = [_clean(w) for w in gold.split() if len(w) > 3][:3]
                fact_keywords = [k for k in fact_keywords if k]  # drop empties
                hit_pos = -1
                if fact_keywords:
                    for i, r in enumerate(results[:K]):
                        content_clean = _clean(r.content or "")
                        if all(kw in content_clean for kw in fact_keywords):
                            hit_pos = i + 1
                            break
                if hit_pos > 0:
                    n_recovered += 1
                    n_at[10] += 1
                    if hit_pos <= 3:
                        n_at[3] += 1
                    if hit_pos == 1:
                        n_at[1] += 1
                    by_probe[probe][0] += 1
                    by_gap[gap][0] += 1

    ingest_wall = time.time() - ingest_t0

    return {
        "ingest": {
            "total_sessions": ingest_summary["sessions"],
            "total_messages": ingest_summary["messages_in"],
            "kept": ingest_summary["kept"],
            "dropped": ingest_summary["dropped"],
            "drop_rate_pct": round(100 * ingest_summary["dropped"] / max(ingest_summary["messages_in"], 1), 2),
            "ingest_wall_clock_s": round(ingest_wall, 2),
            "per_message_ms_avg": round(1000 * sum(per_msg_times) / max(len(per_msg_times), 1), 2),
        },
        "retrieval": {
            "total_qs": n_total,
            "recovered": n_recovered,
            "precision_at_k_pct": round(100 * n_recovered / max(n_total, 1), 2),
            "precision_at_1_pct": round(100 * n_at[1] / max(n_total, 1), 2),
            "precision_at_3_pct": round(100 * n_at[3] / max(n_total, 1), 2),
            "precision_at_10_pct": round(100 * n_at[10] / max(n_total, 1), 2),
            "k": K,
            "by_probe_type": {
                p: {"recovered": v[0], "total": v[1], "pct": round(100 * v[0] / max(v[1], 1), 2)}
                for p, v in by_probe.items()
            },
            "by_time_gap_days": {
                str(g): {"recovered": v[0], "total": v[1], "pct": round(100 * v[0] / max(v[1], 1), 2)}
                for g, v in by_gap.items()
            },
        },
        "storage": {
            "db_size_bytes": db_size_bytes(db_path),
            "rows_in_messages": db_row_count(db_path, "messages"),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", help="Candidate name (e.g. v05_baseline_nogate)")
    parser.add_argument("--dataset", choices=("short_horizon_200", "long_horizon_synthetic"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None,
                        help="DB path (default: ~/.cache/memorist_dbs/<candidate>__<dataset>.db)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete the cached DB and re-ingest from scratch")
    parser.add_argument("--list", action="store_true", help="List discovered candidates and exit")
    args = parser.parse_args()

    candidates = discover_candidates()
    if args.list:
        for name, cls in sorted(candidates.items()):
            print(f"  {name:<40s} ({cls.__module__}.{cls.__name__})")
        return

    if not args.candidate or not args.dataset:
        parser.error("--candidate and --dataset are required unless --list")
    if args.candidate not in candidates:
        print(f"Unknown candidate: {args.candidate}", file=sys.stderr)
        print(f"Available: {sorted(candidates.keys())}", file=sys.stderr)
        sys.exit(2)

    cand = candidates[args.candidate]()
    dataset = load_dataset(args.dataset)

    # Persist DBs at a stable cache location so re-runs (e.g. after fixing
    # a harness bug) can skip the expensive re-ingestion phase. Pass --db
    # explicitly to override; pass --reset to force a fresh DB.
    if args.db:
        db_path = args.db
    else:
        cache_root = Path.home() / ".cache" / "memorist_dbs"
        cache_root.mkdir(parents=True, exist_ok=True)
        db_path = cache_root / f"{args.candidate}__{args.dataset}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reset and db_path.exists():
        db_path.unlink()
    skip_ingest = db_path.exists()
    if skip_ingest:
        print(f"  reusing cached DB: {db_path} (use --reset to force re-ingest)", file=sys.stderr)

    if args.dataset == "short_horizon_200":
        result_body = run_short_horizon(cand, dataset, db_path, skip_ingest=skip_ingest)
    else:
        result_body = run_long_horizon(cand, dataset, db_path, skip_ingest=skip_ingest)

    payload = {
        "candidate": args.candidate,
        "dataset": args.dataset,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "db_path": str(db_path),
        **result_body,
    }

    out = args.output or RESULTS_DIR / f"{args.candidate}__{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"  precision@k = {result_body['retrieval'].get('precision_at_k_pct', 0)}%")
    print(f"  ingest: {result_body['ingest']['kept']} kept / {result_body['ingest']['total_messages']} msgs")

    cand.cleanup()


if __name__ == "__main__":
    main()
