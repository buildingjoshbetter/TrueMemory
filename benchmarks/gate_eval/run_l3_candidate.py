"""
L3 Salience candidate harness.

Loads a labeled dataset built by `build_l3_labels.py`, runs a candidate
scorer, computes AUC / precision@k / Kendall's tau / ECE / latency, and
emits a result JSON under `results/l3_salience/`.

Usage:
    python benchmarks/gate_eval/run_l3_candidate.py \
        --candidate C1 \
        --dataset short \
        [--folds loco] \
        [--out results/l3_salience/C1_short.json]

Supports LOCO-CV (leave-one-conversation-out) on short-horizon; LOSO-CV
(leave-one-session-out) limited to Alex-Park sessions on long-horizon
per pre-registration amendment T9-A1 (see JOURNAL.md).

Anti-circularity guard: if a candidate declares a model-id and that
model-id collides with the downstream reranker model, harness aborts.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
RESULTS = HERE / "results" / "l3_salience"

# Pre-registered frozen artifact — downstream reranker. Candidate models
# MUST NOT match this ID, per COUPLING_CONTRACT.md §D.
DOWNSTREAM_RERANKER = "Alibaba-NLP/gte-reranker-modernbert-base"


def _load(which: str) -> tuple[list[dict], dict, dict]:
    """Return (messages, meta, cv_spec) for one of 'short' or 'long'."""
    if which == "short":
        path = DATASETS / "l3_short_horizon_200_labels.json"
        doc = json.loads(path.read_text())
        messages = doc["messages"]
        meta = doc["_meta"]
        conv_ids = sorted({m["conv_id"] for m in messages})
        cv = {"scheme": "LOCO", "folds": conv_ids}
    elif which == "long":
        path = DATASETS / "l3_long_horizon_synthetic_labels.json"
        doc = json.loads(path.read_text())
        messages_all = doc["messages"]
        # Restrict to Alex Park per pre-reg amendment T9-A1.
        messages = [m for m in messages_all if m["conv_id"] == "Alex Park"]
        meta = {**doc["_meta"], "restricted_to": "Alex Park"}
        session_ids = sorted({m["session_id"] for m in messages})
        cv = {"scheme": "LOSO", "folds": session_ids}
    else:
        raise ValueError(f"unknown dataset: {which}")
    return messages, meta, cv


def _get_candidate(name: str):
    mod = importlib.import_module(f"benchmarks.gate_eval.candidates.l3_salience.{name.lower()}")
    cls = getattr(mod, "Candidate")
    return cls()


def _assert_no_circularity(cand) -> None:
    model_ids = getattr(cand, "model_ids", []) or []
    if DOWNSTREAM_RERANKER in model_ids:
        raise SystemExit(
            f"ANTI-CIRCULARITY ABORT: candidate '{cand.name}' uses the "
            f"downstream reranker '{DOWNSTREAM_RERANKER}' as its source. "
            f"Update the candidate to use a different model."
        )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC with ties averaged. O(n log n)."""
    order = np.argsort(s)
    y_sorted = y[order]
    # Rank handling: ranks average over ties.
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-based average rank
        ranks[i:j + 1] = avg_rank
        i = j + 1
    n_pos = int(y_sorted.sum())
    n_neg = int(len(y_sorted) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(ranks[y_sorted == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    if k <= 0 or k > len(s):
        return float("nan")
    order = np.argsort(-s)
    topk = y[order][:k]
    return float(topk.sum() / k)


def _kendall_tau_b(y: np.ndarray, s: np.ndarray) -> float:
    """Approximate Kendall's tau-b via scipy if available; else fall back."""
    try:
        from scipy.stats import kendalltau
        t, _ = kendalltau(s, y)
        return float(t) if not math.isnan(t) else 0.0
    except Exception:
        return float("nan")


def _ece(y: np.ndarray, s: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error on equal-frequency bins."""
    n = len(s)
    if n == 0:
        return float("nan")
    order = np.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        bin_s = s_sorted[lo:hi]
        bin_y = y_sorted[lo:hi]
        if len(bin_s) == 0:
            continue
        mean_conf = float(bin_s.mean())
        mean_acc = float(bin_y.mean())
        ece += (hi - lo) / n * abs(mean_conf - mean_acc)
    return float(ece)


def _welch_mean_ci(xs: list[float]) -> tuple[float, float, float]:
    """Return (mean, se, 95%CI-half-width) for a list of floats."""
    a = np.asarray([x for x in xs if not math.isnan(x)])
    if len(a) < 2:
        return (float(a.mean()) if len(a) else float("nan"), float("nan"), float("nan"))
    mean = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(len(a)))
    # Conservative z-based 95% CI half-width.
    return (mean, se, 1.96 * se)


# --------------------------------------------------------------------------
# Run one fold
# --------------------------------------------------------------------------

def _run_fold(cand, messages_train: list[dict], messages_test: list[dict]) -> dict:
    # Fit on train (no-op for most).
    cand.fit(messages_train)

    # Score test.
    scores = np.empty(len(messages_test), dtype=float)
    t0 = time.perf_counter()
    for i, m in enumerate(messages_test):
        scores[i] = cand.score(m)
    total_ms = (time.perf_counter() - t0) * 1000.0
    per_msg_ms = total_ms / max(1, len(messages_test))

    y = np.asarray([m["utility_binary"] for m in messages_test], dtype=int)
    s = np.asarray(scores, dtype=float)

    return {
        "n": int(len(messages_test)),
        "n_pos": int(y.sum()),
        "auc": _roc_auc(y, s),
        "p5": _precision_at_k(y, s, 5),
        "p10": _precision_at_k(y, s, 10),
        "p20": _precision_at_k(y, s, 20),
        "p50": _precision_at_k(y, s, 50),
        "tau": _kendall_tau_b(y, s),
        "ece": _ece(y, s),
        "ms_per_msg": per_msg_ms,
        "total_ms": total_ms,
    }


# --------------------------------------------------------------------------
# CV driver
# --------------------------------------------------------------------------

def run(candidate_name: str, dataset: str, out_path: Path | None = None) -> dict:
    cand = _get_candidate(candidate_name)
    _assert_no_circularity(cand)

    messages, meta, cv = _load(dataset)
    folds = cv["folds"]
    fold_key = "conv_id" if cv["scheme"] == "LOCO" else "session_id"

    print(f"Running {candidate_name} on {dataset}: {len(messages)} msgs, "
          f"{len(folds)}-fold {cv['scheme']}-CV")

    per_fold = []
    for i, f in enumerate(folds):
        train = [m for m in messages if m[fold_key] != f]
        test = [m for m in messages if m[fold_key] == f]
        print(f"  fold {i+1}/{len(folds)}: fold={f} "
              f"(train={len(train)}, test={len(test)}, "
              f"pos={sum(m['utility_binary'] for m in test)})",
              end="", flush=True)
        r = _run_fold(cand, train, test)
        r["fold"] = f
        per_fold.append(r)
        print(f" → auc={r['auc']:.3f} p10={r['p10']:.3f} ms/msg={r['ms_per_msg']:.3f}")

    agg = {}
    for k in ("auc", "p5", "p10", "p20", "p50", "tau", "ece", "ms_per_msg"):
        mean, se, ci = _welch_mean_ci([r[k] for r in per_fold])
        agg[f"{k}_mean"] = mean
        agg[f"{k}_se"] = se
        agg[f"{k}_ci95"] = ci

    result = {
        "candidate": candidate_name,
        "dataset": dataset,
        "cv_scheme": cv["scheme"],
        "n_folds": len(folds),
        "dataset_meta": meta,
        "per_fold": per_fold,
        "aggregate": agg,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=float))
        print(f"Wrote {out_path}")

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="candidate id (e.g. C1, C4)")
    ap.add_argument("--dataset", required=True, choices=["short", "long"])
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON path (default: results/l3_salience/{cand}_{dataset}.json)")
    args = ap.parse_args()

    out = args.out or (RESULTS / f"{args.candidate}_{args.dataset}.json")
    run(args.candidate, args.dataset, out_path=out)
    return 0


if __name__ == "__main__":
    # Make project root importable as 'benchmarks.gate_eval.candidates...'
    sys.path.insert(0, str(HERE.parent.parent))
    sys.exit(main())
