#!/usr/bin/env python3
"""
Phase 4 & 5: Correlation analysis + Three-signal ablation for PE sweep.

Reads pe_100_sweep.json, computes pairwise correlations among top 30,
then tests three-signal gate combinations (novelty + salience + PE).
"""

import json
import math
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def pearson_r(x, y):
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(max(0, sum((xi - mx) ** 2 for xi in x) / (n - 1)))
    sy = math.sqrt(max(0, sum((yi - my) ** 2 for yi in y) / (n - 1)))
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    return cov / (sx * sy)


def compute_auc(scores_signal, scores_noise):
    labels = [1] * len(scores_signal) + [0] * len(scores_noise)
    scores = list(scores_signal) + list(scores_noise)
    paired = sorted(zip(scores, labels), reverse=True)
    tp = fp = tp_prev = fp_prev = 0
    auc = 0.0
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    prev_score = None
    for score, label in paired:
        if score != prev_score and prev_score is not None:
            auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
            tp_prev = tp
            fp_prev = fp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
    return auc / (n_pos * n_neg)


def main():
    sweep_path = RESULTS_DIR / "pe_100_sweep.json"
    with open(sweep_path) as f:
        sweep = json.load(f)

    all_scores = sweep["all_scores"]
    novelty_scores = sweep["novelty_scores"]
    salience_scores = sweep["salience_scores"]
    results = sweep["results"]

    # Build labels from the benchmark
    bench_path = Path(__file__).parent / "datasets" / "gate_benchmark.json"
    with open(bench_path) as f:
        bench = json.load(f)

    labels = []
    cats = []
    for conv in bench["conversations"]:
        for msg in conv["messages"]:
            cat = msg["category"]
            cats.append(cat)
            if cat.startswith("S"):
                labels.append("S")
            elif cat.startswith("N"):
                labels.append("N")
            else:
                labels.append("B")

    # =====================================================================
    # PHASE 4: Correlation Analysis
    # =====================================================================
    print("=" * 80)
    print("PHASE 4: Correlation Analysis")
    print("=" * 80)

    top30_names = [r["variant"] for r in results[:30]]
    top30_scores = {name: all_scores[name] for name in top30_names}

    # Pairwise correlation matrix
    corr_matrix = {}
    for i, name1 in enumerate(top30_names):
        corr_matrix[name1] = {}
        for name2 in top30_names:
            r = pearson_r(top30_scores[name1], top30_scores[name2])
            corr_matrix[name1][name2] = round(r, 4)

    # Correlation with novelty and salience
    novelty_corr = {}
    salience_corr = {}
    for name in top30_names:
        novelty_corr[name] = round(pearson_r(top30_scores[name], novelty_scores), 4)
        salience_corr[name] = round(pearson_r(top30_scores[name], salience_scores), 4)

    # Identify candidates: HIGH AUC + LOW correlation with N/S
    candidates = []
    for r in results[:30]:
        name = r["variant"]
        r_n = abs(novelty_corr[name])
        r_s = abs(salience_corr[name])
        if r_n < 0.4 and r_s < 0.4:
            candidates.append({
                "variant": name,
                "auc": r["auc"],
                "r_novelty": novelty_corr[name],
                "r_salience": salience_corr[name],
                "s4_recall": r["s4_recall_03"],
                "n_fp_rate": r["n_fp_rate_03"],
            })

    print(f"\nTop 30 variants analyzed")
    print(f"Candidates (AUC > thresh, |r_nov| < 0.4, |r_sal| < 0.4): {len(candidates)}")
    for c in candidates[:10]:
        print(f"  {c['variant']}: AUC={c['auc']:.3f}, r_nov={c['r_novelty']:.3f}, r_sal={c['r_salience']:.3f}")

    corr_output = {
        "top30_pairwise": corr_matrix,
        "novelty_correlation": novelty_corr,
        "salience_correlation": salience_corr,
        "independent_candidates": candidates,
    }
    corr_path = RESULTS_DIR / "pe_correlation.json"
    with open(corr_path, "w") as f:
        json.dump(corr_output, f, indent=2)
    print(f"\nCorrelation results saved to {corr_path}")

    # =====================================================================
    # PHASE 5: Three-Signal Ablation
    # =====================================================================
    print("\n" + "=" * 80)
    print("PHASE 5: Three-Signal Ablation")
    print("=" * 80)

    # Use top 10 PE variants for ablation
    top10_pe = [r["variant"] for r in results[:10]]

    weight_configs = [
        ("equal", 0.33, 0.33, 0.33),
        ("novelty_heavy", 0.50, 0.25, 0.25),
        ("salience_heavy", 0.25, 0.50, 0.25),
        ("pe_heavy", 0.25, 0.25, 0.50),
        ("paper_default", 0.40, 0.35, 0.25),
    ]

    # Two-signal baselines
    baselines = {
        "novelty_only": (1.0, 0.0, 0.0),
        "salience_only": (0.0, 1.0, 0.0),
        "nov_sal": (0.50, 0.50, 0.0),
        "nov_sal_paper": (0.55, 0.45, 0.0),
    }

    ablation_results = {}

    for pe_name in top10_pe:
        pe_scores = all_scores[pe_name]
        pe_result = {"variant": pe_name}
        pe_result["pe_auc"] = results[[r["variant"] for r in results].index(pe_name)]["auc"]

        # Baselines (no PE)
        baseline_metrics = {}
        for bl_name, (wn, ws, wpe) in baselines.items():
            combo = [
                wn * n + ws * s + wpe * p
                for n, s, p in zip(novelty_scores, salience_scores, pe_scores)
            ]
            sig = [c for c, l in zip(combo, labels) if l == "S"]
            noi = [c for c, l in zip(combo, labels) if l == "N"]
            auc = compute_auc(sig, noi)

            s4 = [c for c, cat in zip(combo, cats) if cat == "S4"]
            s4_recall = sum(1 for s in s4 if s >= 0.3) / max(len(s4), 1)

            n_all = [c for c, l in zip(combo, labels) if l == "N"]
            n_fp = sum(1 for s in n_all if s >= 0.3) / max(len(n_all), 1)

            baseline_metrics[bl_name] = {
                "weights": (wn, ws, wpe),
                "auc": round(auc, 4),
                "s4_recall": round(s4_recall, 4),
                "n_fp_rate": round(n_fp, 4),
            }

        pe_result["baselines"] = baseline_metrics

        # Three-signal combinations
        combo_metrics = {}
        for config_name, wn, ws, wpe in weight_configs:
            combo = [
                wn * n + ws * s + wpe * p
                for n, s, p in zip(novelty_scores, salience_scores, pe_scores)
            ]
            sig = [c for c, l in zip(combo, labels) if l == "S"]
            noi = [c for c, l in zip(combo, labels) if l == "N"]
            auc = compute_auc(sig, noi)

            s4 = [c for c, cat in zip(combo, cats) if cat == "S4"]
            s4_recall = sum(1 for s in s4 if s >= 0.3) / max(len(s4), 1)

            n_all = [c for c, l in zip(combo, labels) if l == "N"]
            n_fp = sum(1 for s in n_all if s >= 0.3) / max(len(n_all), 1)

            combo_metrics[config_name] = {
                "weights": (wn, ws, wpe),
                "auc": round(auc, 4),
                "s4_recall": round(s4_recall, 4),
                "n_fp_rate": round(n_fp, 4),
            }

        pe_result["three_signal"] = combo_metrics

        # Compute improvement over best baseline
        best_bl_auc = max(b["auc"] for b in baseline_metrics.values())
        best_3s_auc = max(c["auc"] for c in combo_metrics.values())
        pe_result["improvement"] = round(best_3s_auc - best_bl_auc, 4)
        pe_result["best_baseline_auc"] = best_bl_auc
        pe_result["best_three_signal_auc"] = best_3s_auc

        ablation_results[pe_name] = pe_result

    # Print ablation results
    print(f"\n{'PE Variant':<16} {'PE AUC':>7} {'Best BL':>8} {'Best 3S':>8} {'Δ AUC':>7}")
    print("-" * 50)
    for name in top10_pe:
        r = ablation_results[name]
        delta_str = f"+{r['improvement']:.4f}" if r["improvement"] > 0 else f"{r['improvement']:.4f}"
        print(f"{name:<16} {r['pe_auc']:>7.4f} {r['best_baseline_auc']:>8.4f} {r['best_three_signal_auc']:>8.4f} {delta_str:>7}")

    # Print detailed breakdown for best PE variant
    best_pe = max(ablation_results.values(), key=lambda r: r["improvement"])
    print(f"\nBest PE variant for ablation: {best_pe['variant']} (Δ AUC: +{best_pe['improvement']:.4f})")
    print(f"\nBaseline comparisons:")
    for bl_name, bl in best_pe["baselines"].items():
        print(f"  {bl_name:<20} AUC={bl['auc']:.4f}  S4_R={bl['s4_recall']:.2f}  N_FP={bl['n_fp_rate']:.2f}")
    print(f"\nThree-signal combos (with {best_pe['variant']}):")
    for cfg_name, cfg in best_pe["three_signal"].items():
        print(f"  {cfg_name:<20} AUC={cfg['auc']:.4f}  S4_R={cfg['s4_recall']:.2f}  N_FP={cfg['n_fp_rate']:.2f}")

    ablation_path = RESULTS_DIR / "pe_ablation.json"
    with open(ablation_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nAblation results saved to {ablation_path}")

    return corr_output, ablation_results


if __name__ == "__main__":
    main()
