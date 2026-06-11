#!/usr/bin/env python3
"""
Benchmark runner for the 100-variant prediction error sweep.

Key difference from novelty sweep: PE scores depend on the RELATIONSHIP
between the message and specific stored memories. The same message should
score differently depending on what's in memory.

Also computes correlation with shipped novelty (compression) and salience
(speech-act hybrid) scorers to ensure PE adds independent signal.
"""

import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.gate_eval.pe_sweep import ALL_VARIANTS, set_embedder
from benchmarks.gate_eval.novelty_sweep import (
    set_embedder as set_novelty_embedder,
)

# Shipped novelty scorer: PPM compression cost (variant_025)
from benchmarks.gate_eval.novelty_sweep import variant_025 as shipped_novelty

# Shipped salience scorer: speech-act hybrid (encoding_salience_d)
from truememory.ingest.encoding_salience import encoding_salience_d as shipped_salience

# Load embedder
from model2vec import StaticModel
model = StaticModel.from_pretrained("minishlab/potion-base-8M")
set_embedder(model)
set_novelty_embedder(model)
print(f"Embedder loaded: model2vec potion-base-8M, {len(ALL_VARIANTS)} PE variants")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_benchmark():
    path = Path(__file__).parent / "datasets" / "gate_benchmark.json"
    with open(path) as f:
        return json.load(f)


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


SPECIFIC_MESSAGES = [
    "ok", "lol", "we're pregnant", "I GOT IT", "dad passed away",
    "I said yes", "haha", "nice", "sounds good", "I DID IT",
    "I HAD A BABY", "I GOT INTO UCLA",
]


def run_sweep():
    data = load_benchmark()
    conversations = data["conversations"]

    variant_names = sorted(ALL_VARIANTS.keys())
    n_variants = len(variant_names)

    all_scores = {name: [] for name in variant_names}
    novelty_scores_all = []
    salience_scores_all = []
    all_labels = []
    all_cats = []
    msg_specific_scores = {name: {} for name in variant_names}
    variant_times = {name: 0.0 for name in variant_names}
    variant_counts = {name: 0 for name in variant_names}
    variant_errors = {name: [] for name in variant_names}

    total_msgs = 0

    for conv in conversations:
        messages = conv["messages"]
        conv_id = conv["conversation_id"]
        print(f"\nProcessing {conv_id}: {len(messages)} messages")

        memory_contents: list[str] = []
        memory_embeddings_list: list[np.ndarray] = []

        for msg_idx, msg in enumerate(messages):
            content = msg["content"]
            category = msg["category"]
            is_signal = msg["is_signal"]

            total_msgs += 1

            if memory_embeddings_list:
                memory_embeddings = np.array(memory_embeddings_list)
            else:
                memory_embeddings = None

            # Score with shipped novelty (for correlation)
            try:
                nov_score = shipped_novelty(content, memory_contents, memory_embeddings)
                nov_score = max(0.0, min(1.0, float(nov_score)))
            except Exception:
                nov_score = 0.5
            novelty_scores_all.append(nov_score)

            # Score with shipped salience (for correlation)
            try:
                sal_score = shipped_salience(content)
                sal_score = max(0.0, min(1.0, float(sal_score)))
            except Exception:
                sal_score = 0.5
            salience_scores_all.append(sal_score)

            # Score with all PE variants
            for name in variant_names:
                fn = ALL_VARIANTS[name]
                t0 = time.perf_counter()
                try:
                    score = fn(content, memory_contents, memory_embeddings)
                    score = max(0.0, min(1.0, float(score)))
                except Exception as e:
                    score = 0.5
                    if len(variant_errors[name]) < 3:
                        variant_errors[name].append(f"{conv_id}/{msg_idx}: {e}")
                elapsed = time.perf_counter() - t0
                variant_times[name] += elapsed
                variant_counts[name] += 1

                all_scores[name].append(score)

                if content in SPECIFIC_MESSAGES:
                    msg_specific_scores[name][content] = round(score, 4)

            # Record label
            if category.startswith("S"):
                all_labels.append("S")
            elif category.startswith("N"):
                all_labels.append("N")
            else:
                all_labels.append("B")
            all_cats.append(category)

            # Add signal messages to memory
            if is_signal:
                memory_contents.append(content)
                emb = model.encode([content])[0]
                memory_embeddings_list.append(emb)

            if (msg_idx + 1) % 100 == 0:
                print(f"  {msg_idx + 1}/{len(messages)} msgs, memory: {len(memory_contents)}")

    print(f"\nTotal messages scored: {total_msgs}")

    # Compute metrics per variant
    results = []
    for name in variant_names:
        fn = ALL_VARIANTS[name]
        desc = fn.__doc__ or ""
        desc = desc.strip().split("\n")[0] if desc else name
        scores = all_scores[name]

        scores_signal = [s for s, l in zip(scores, all_labels) if l == "S"]
        scores_noise = [s for s, l in zip(scores, all_labels) if l == "N"]

        auc = compute_auc(scores_signal, scores_noise)

        cat_recall = {}
        for cat in sorted(set(all_cats)):
            cat_scores = [s for s, c in zip(scores, all_cats) if c == cat]
            if cat_scores:
                recall = sum(1 for s in cat_scores if s >= 0.3) / len(cat_scores)
                cat_recall[cat] = round(recall, 4)

        s4_scores = [s for s, c in zip(scores, all_cats) if c == "S4"]
        s4_recall = sum(1 for s in s4_scores if s >= 0.3) / max(len(s4_scores), 1)

        n_scores = [s for s, l in zip(scores, all_labels) if l == "N"]
        n_fp = sum(1 for s in n_scores if s >= 0.3) / max(len(n_scores), 1)

        s_cat_recall = {}
        for cat in ["S1", "S2", "S3", "S4", "S5"]:
            cat_s = [s for s, c in zip(scores, all_cats) if c == cat]
            if cat_s:
                s_cat_recall[cat] = round(
                    sum(1 for s in cat_s if s >= 0.3) / len(cat_s), 4
                )

        ms_per_msg = (variant_times[name] / max(variant_counts[name], 1)) * 1000

        # Correlation with shipped novelty and salience
        r_novelty = pearson_r(scores, novelty_scores_all)
        r_salience = pearson_r(scores, salience_scores_all)

        result = {
            "variant": name,
            "description": desc,
            "auc": round(auc, 4),
            "s4_recall_03": round(s4_recall, 4),
            "n_fp_rate_03": round(n_fp, 4),
            "s_recall_03": s_cat_recall,
            "cat_recall_03": cat_recall,
            "scores": msg_specific_scores[name],
            "ms_per_msg": round(ms_per_msg, 4),
            "r_novelty": round(r_novelty, 4),
            "r_salience": round(r_salience, 4),
            "correlated_with_novelty": abs(r_novelty) > 0.6,
            "correlated_with_salience": abs(r_salience) > 0.6,
            "errors": variant_errors[name] if variant_errors[name] else None,
        }
        results.append(result)

    results.sort(key=lambda r: r["auc"], reverse=True)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_variants": len(results),
        "num_messages": total_msgs,
        "num_signal": sum(1 for l in all_labels if l == "S"),
        "num_noise": sum(1 for l in all_labels if l == "N"),
        "num_borderline": sum(1 for l in all_labels if l == "B"),
        "results": results,
        "all_scores": {name: [round(s, 4) for s in scores] for name, scores in all_scores.items()},
        "novelty_scores": [round(s, 4) for s in novelty_scores_all],
        "salience_scores": [round(s, 4) for s in salience_scores_all],
    }

    out_path = RESULTS_DIR / "pe_100_sweep.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print leaderboard
    print("\n" + "=" * 110)
    print(f"{'Rank':>4} {'Variant':<16} {'AUC':>6} {'S4_R':>5} {'N_FP':>5} {'r_nov':>6} {'r_sal':>6} {'ms':>6}  Description")
    print("=" * 110)
    for i, r in enumerate(results[:30]):
        flag = ""
        if r["correlated_with_novelty"]:
            flag += " [NOV!]"
        if r["correlated_with_salience"]:
            flag += " [SAL!]"
        desc = r["description"][:40] + flag
        print(
            f"{i+1:>4} {r['variant']:<16} {r['auc']:>6.3f} "
            f"{r['s4_recall_03']:>5.2f} {r['n_fp_rate_03']:>5.2f} "
            f"{r['r_novelty']:>6.3f} {r['r_salience']:>6.3f} "
            f"{r['ms_per_msg']:>6.2f}  {desc}"
        )

    # Print specific message scores for top 5
    print("\n\nSpecific message scores (top 5 variants):")
    print(f"{'Message':<25}", end="")
    for r in results[:5]:
        print(f" {r['variant']:>14}", end="")
    print()
    for msg in SPECIFIC_MESSAGES:
        print(f"{msg:<25}", end="")
        for r in results[:5]:
            s = r["scores"].get(msg, "—")
            if isinstance(s, float):
                print(f" {s:>14.3f}", end="")
            else:
                print(f" {str(s):>14}", end="")
        print()

    return output


if __name__ == "__main__":
    run_sweep()
