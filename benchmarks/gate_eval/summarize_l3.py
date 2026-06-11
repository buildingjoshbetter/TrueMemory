"""Summarize L3 sweep results.

Reads all results/l3_salience/*_{dataset}.json files, prints a table
with mean ± CI for AUC, p@10, ECE, ms/msg, and Welch's t-test p-value
vs C1 baseline (per-fold paired comparison).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "l3_salience"


def _welch(a: list[float], b: list[float]) -> float:
    """Welch's two-sample t-test p-value (two-tailed)."""
    try:
        from scipy.stats import ttest_ind
        t, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return float(p)
    except Exception:
        return float("nan")


def _paired_t(deltas: list[float]) -> float:
    """One-sample t-test p-value vs 0 (paired t)."""
    try:
        from scipy.stats import ttest_1samp
        clean = [d for d in deltas if not math.isnan(d)]
        if len(clean) < 2:
            return float("nan")
        t, p = ttest_1samp(clean, 0.0)
        return float(p)
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="short")
    ap.add_argument("--baseline", default="C1")
    args = ap.parse_args()

    files = sorted(RES.glob(f"*_{args.dataset}.json"))
    if not files:
        print(f"No results found for dataset={args.dataset}")
        return 1

    results = {}
    for fp in files:
        d = json.loads(fp.read_text())
        results[d["candidate"]] = d

    if args.baseline not in results:
        print(f"Baseline {args.baseline} not found in results.")
        return 1

    base_folds = {r["fold"]: r for r in results[args.baseline]["per_fold"]}

    rows = []
    for cid, d in sorted(results.items()):
        agg = d["aggregate"]
        # Per-fold deltas vs baseline (paired by fold).
        deltas_auc = []
        deltas_p10 = []
        for r in d["per_fold"]:
            br = base_folds.get(r["fold"])
            if not br:
                continue
            if not (math.isnan(r["auc"]) or math.isnan(br["auc"])):
                deltas_auc.append(r["auc"] - br["auc"])
            if not (math.isnan(r["p10"]) or math.isnan(br["p10"])):
                deltas_p10.append(r["p10"] - br["p10"])
        p_auc = _paired_t(deltas_auc) if cid != args.baseline else None
        p_p10 = _paired_t(deltas_p10) if cid != args.baseline else None
        rows.append({
            "id": cid,
            "auc": agg["auc_mean"],
            "auc_ci": agg["auc_ci95"],
            "d_auc": (agg["auc_mean"] - results[args.baseline]["aggregate"]["auc_mean"]),
            "p_auc": p_auc,
            "p10": agg["p10_mean"],
            "d_p10": (agg["p10_mean"] - results[args.baseline]["aggregate"]["p10_mean"]),
            "p_p10": p_p10,
            "ece": agg["ece_mean"],
            "ms": agg["ms_per_msg_mean"],
        })

    print(f"\nDataset: {args.dataset}    Baseline: {args.baseline}    Folds: "
          f"{results[args.baseline]['n_folds']}\n")
    print(f"{'cand':<6} {'AUC':>7} {'±CI':>6} {'ΔAUC':>7} {'p_AUC':>7}  "
          f"{'p@10':>5} {'Δp@10':>7} {'p_p10':>7}  {'ECE':>5}  {'ms/msg':>7}")
    print("-" * 88)
    for r in rows:
        ci = f"{r['auc_ci']:.3f}" if not math.isnan(r['auc_ci']) else "  -  "
        p_auc_s = f"{r['p_auc']:.3f}" if r['p_auc'] is not None and not math.isnan(r['p_auc']) else "  -  "
        p_p10_s = f"{r['p_p10']:.3f}" if r['p_p10'] is not None and not math.isnan(r['p_p10']) else "  -  "
        ece_s = f"{r['ece']:.3f}" if not math.isnan(r['ece']) else "  -  "
        print(f"{r['id']:<6} {r['auc']:.3f}  {ci}  {r['d_auc']:+.3f}  {p_auc_s:>6}  "
              f"{r['p10']:.3f}  {r['d_p10']:+.3f}  {p_p10_s:>6}  {ece_s}  {r['ms']:7.4f}")

    # Highlight winners by null band (ΔAUC ≥ 0.03 ∧ Δp@10 ≥ 0.01 ∧ p < 0.1)
    print("\nNull-band check (ΔAUC ≥ 0.03 ∧ Δp@10 ≥ 0.01 ∧ Welch p < 0.1):")
    base_auc = results[args.baseline]["aggregate"]["auc_mean"]
    for r in rows:
        if r["id"] == args.baseline:
            continue
        winner = (
            r["d_auc"] >= 0.03
            and r["d_p10"] >= 0.01
            and r["p_auc"] is not None
            and not math.isnan(r["p_auc"])
            and r["p_auc"] < 0.1
        )
        print(f"  {r['id']:<6} → {'WINNER ✓' if winner else 'null'}  "
              f"(ΔAUC={r['d_auc']:+.3f}, Δp@10={r['d_p10']:+.3f}, p={r['p_auc']})")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
