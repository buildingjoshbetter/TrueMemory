#!/usr/bin/env python3
"""
Phase 4: Correlation analysis of top novelty variants.
Phase 5: Combination sweep of uncorrelated high-AUC variants.
"""

import json
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"


def pearson_r(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
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


def run_correlation():
    data = json.load(open(RESULTS_DIR / "novelty_120_sweep.json"))
    results = data["results"]
    all_scores = data["all_scores"]

    # Load benchmark for labels
    bench = json.load(open(Path(__file__).parent / "datasets" / "gate_benchmark.json"))
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

    # Top 30 variants by AUC
    top30 = results[:30]
    top30_names = [r["variant"] for r in top30]

    print(f"Top 30 variants for correlation analysis:")
    for i, r in enumerate(top30):
        print(f"  {i+1}. {r['variant']} AUC={r['auc']:.3f}")

    # Phase 4: Pairwise correlation
    print("\n\n=== PHASE 4: CORRELATION ANALYSIS ===\n")

    corr_matrix = {}
    for i, n1 in enumerate(top30_names):
        for j, n2 in enumerate(top30_names):
            if i >= j:
                continue
            r = pearson_r(all_scores[n1], all_scores[n2])
            corr_matrix[f"{n1}|{n2}"] = round(r, 3)

    # Find uncorrelated pairs (|r| < 0.3)
    uncorrelated_pairs = []
    for pair, r in sorted(corr_matrix.items(), key=lambda x: abs(x[1])):
        n1, n2 = pair.split("|")
        a1 = next(r_["auc"] for r_ in results if r_["variant"] == n1)
        a2 = next(r_["auc"] for r_ in results if r_["variant"] == n2)
        if abs(r) < 0.3:
            uncorrelated_pairs.append((n1, n2, r, a1, a2))

    print(f"Uncorrelated pairs (|r| < 0.3): {len(uncorrelated_pairs)}")
    for n1, n2, r, a1, a2 in uncorrelated_pairs[:20]:
        print(f"  {n1} ({a1:.3f}) × {n2} ({a2:.3f})  r={r:.3f}")

    # Select top 5 most uncorrelated high-AUC variants
    # Strategy: pick variants that appear most often in uncorrelated pairs
    # and have high AUC
    from collections import Counter
    variant_uncorr_count = Counter()
    for n1, n2, r, a1, a2 in uncorrelated_pairs:
        variant_uncorr_count[n1] += 1
        variant_uncorr_count[n2] += 1

    # Score: AUC * uncorrelated_count
    variant_combo_score = {}
    for name in top30_names[:20]:  # Top 20
        auc = next(r_["auc"] for r_ in results if r_["variant"] == name)
        uncorr = variant_uncorr_count.get(name, 0)
        variant_combo_score[name] = auc * (1 + uncorr * 0.1)

    top5_combo = sorted(variant_combo_score.items(), key=lambda x: x[1], reverse=True)[:5]
    top10_combo = sorted(variant_combo_score.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 5 for combinations:")
    for name, score in top5_combo:
        auc = next(r_["auc"] for r_ in results if r_["variant"] == name)
        uncorr = variant_uncorr_count.get(name, 0)
        print(f"  {name}: AUC={auc:.3f} uncorr_pairs={uncorr} combo_score={score:.3f}")

    # Save correlation results
    corr_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "top30_variants": top30_names,
        "correlation_matrix": corr_matrix,
        "uncorrelated_pairs": [
            {"v1": n1, "v2": n2, "r": r, "auc1": a1, "auc2": a2}
            for n1, n2, r, a1, a2 in uncorrelated_pairs
        ],
        "top5_combo_candidates": [name for name, _ in top5_combo],
        "top10_combo_candidates": [name for name, _ in top10_combo],
    }
    with open(RESULTS_DIR / "novelty_correlation.json", "w") as f:
        json.dump(corr_output, f, indent=2)
    print(f"\nCorrelation results saved.")

    # Phase 5: Combination sweep
    print("\n\n=== PHASE 5: COMBINATION SWEEP ===\n")

    top5_names = [name for name, _ in top5_combo]
    top10_names = [name for name, _ in top10_combo]

    combo_results = []

    def eval_combo(scores_combined, combo_name, method):
        s_scores = [s for s, l in zip(scores_combined, labels) if l == "S"]
        n_scores = [s for s, l in zip(scores_combined, labels) if l == "N"]
        auc = compute_auc(s_scores, n_scores)

        s4_scores = [s for s, c in zip(scores_combined, cats) if c == "S4"]
        s4_recall = sum(1 for s in s4_scores if s >= 0.3) / max(len(s4_scores), 1)

        n_fp = sum(1 for s in n_scores if s >= 0.3) / max(len(n_scores), 1)

        # Per-S-category recall
        s_recall = {}
        for cat in ["S1", "S2", "S3", "S4", "S5"]:
            cat_s = [s for s, c in zip(scores_combined, cats) if c == cat]
            if cat_s:
                s_recall[cat] = round(sum(1 for s in cat_s if s >= 0.3) / len(cat_s), 4)

        combo_results.append({
            "combo": combo_name,
            "method": method,
            "auc": round(auc, 4),
            "s4_recall_03": round(s4_recall, 4),
            "n_fp_rate_03": round(n_fp, 4),
            "s_recall_03": s_recall,
        })
        return auc

    n_msgs = len(labels)

    # All pairs from top 10
    print(f"Testing pairs from top 10 ({len(list(combinations(top10_names, 2)))} pairs)...")
    for v1, v2 in combinations(top10_names, 2):
        s1 = np.array(all_scores[v1])
        s2 = np.array(all_scores[v2])

        # Mean
        combined = (s1 + s2) / 2
        eval_combo(combined.tolist(), f"{v1}+{v2}", "mean")

        # Geometric mean
        combined = np.sqrt(np.maximum(s1, 0.001) * np.maximum(s2, 0.001))
        eval_combo(combined.tolist(), f"{v1}+{v2}", "gmean")

        # Max
        combined = np.maximum(s1, s2)
        eval_combo(combined.tolist(), f"{v1}+{v2}", "max")

    # All triples from top 5
    print(f"Testing triples from top 5 ({len(list(combinations(top5_names, 3)))} triples)...")
    for combo_vars in combinations(top5_names, 3):
        scores_list = [np.array(all_scores[v]) for v in combo_vars]
        stacked = np.stack(scores_list)

        # Mean
        combined = stacked.mean(axis=0)
        name = "+".join(combo_vars)
        eval_combo(combined.tolist(), name, "mean")

        # Geometric mean
        combined = np.exp(np.mean(np.log(np.maximum(stacked, 0.001)), axis=0))
        eval_combo(combined.tolist(), name, "gmean")

        # Max
        combined = stacked.max(axis=0)
        eval_combo(combined.tolist(), name, "max")

    # 5-way combination of top 5
    print("Testing 5-way combination...")
    scores_list = [np.array(all_scores[v]) for v in top5_names]
    stacked = np.stack(scores_list)
    name = "+".join(top5_names)

    eval_combo(stacked.mean(axis=0).tolist(), name, "mean")
    eval_combo(np.exp(np.mean(np.log(np.maximum(stacked, 0.001)), axis=0)).tolist(), name, "gmean")
    eval_combo(stacked.max(axis=0).tolist(), name, "max")

    # Weighted combinations with AUC-weighted average
    print("Testing AUC-weighted combinations...")
    for combo_vars in combinations(top10_names, 2):
        s_list = [np.array(all_scores[v]) for v in combo_vars]
        aucs = [next(r_["auc"] for r_ in results if r_["variant"] == v) for v in combo_vars]
        total_auc = sum(aucs)
        weights = [a / total_auc for a in aucs]
        combined = sum(w * s for w, s in zip(weights, s_list))
        eval_combo(combined.tolist(), "+".join(combo_vars), "auc_weighted")

    # Logistic regression combinations (top 5)
    print("Testing logistic regression combinations...")
    for combo_vars in combinations(top5_names, 2):
        s_list = [np.array(all_scores[v]) for v in combo_vars]
        # Simple logistic fit: grid search for best weights
        y = np.array([1 if l == "S" else 0 for l in labels])
        mask = np.array([l != "B" for l in labels])
        y_filtered = y[mask]

        best_auc = 0
        best_w = None
        for w1 in np.arange(0.1, 1.0, 0.1):
            w2 = 1.0 - w1
            combined = w1 * s_list[0] + w2 * s_list[1]
            c_filtered = combined[mask]
            s_sig = c_filtered[y_filtered == 1]
            n_sig = c_filtered[y_filtered == 0]
            auc = compute_auc(s_sig.tolist(), n_sig.tolist())
            if auc > best_auc:
                best_auc = auc
                best_w = (w1, w2)

        if best_w:
            combined = best_w[0] * s_list[0] + best_w[1] * s_list[1]
            eval_combo(combined.tolist(), "+".join(combo_vars), f"best_weighted({best_w[0]:.1f},{best_w[1]:.1f})")

    # Sort by AUC
    combo_results.sort(key=lambda r: r["auc"], reverse=True)

    # Save
    combo_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_combinations": len(combo_results),
        "top5_candidates": top5_names,
        "top10_candidates": top10_names,
        "results": combo_results,
    }
    with open(RESULTS_DIR / "novelty_combo_sweep.json", "w") as f:
        json.dump(combo_output, f, indent=2)

    # Print top 20
    print(f"\n{'Rank':>4} {'AUC':>6} {'S4_R':>5} {'N_FP':>5} {'Method':<20} Combo")
    print("=" * 90)
    for i, r in enumerate(combo_results[:30]):
        print(f"{i+1:>4} {r['auc']:>6.3f} {r['s4_recall_03']:>5.2f} {r['n_fp_rate_03']:>5.2f} {r['method']:<20} {r['combo'][:50]}")

    print(f"\nTotal combinations tested: {len(combo_results)}")
    print(f"Best combination: {combo_results[0]['combo']} ({combo_results[0]['method']}) AUC={combo_results[0]['auc']:.4f}")

    # Compare best combo vs best individual
    best_individual = results[0]
    print(f"Best individual:  {best_individual['variant']} AUC={best_individual['auc']:.4f}")
    print(f"Improvement:      {(combo_results[0]['auc'] - best_individual['auc']):.4f}")


if __name__ == "__main__":
    run_correlation()
