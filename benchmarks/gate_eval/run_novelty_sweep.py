#!/usr/bin/env python3
"""
Benchmark runner for the 120-variant novelty sweep.

Key difference from salience sweep: memory state builds incrementally
per conversation. Each message is scored against previously-stored
messages, not in isolation.
"""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.gate_eval.novelty_sweep import ALL_VARIANTS, set_embedder

# Load embedder
from model2vec import StaticModel
model = StaticModel.from_pretrained("minishlab/potion-base-8M")
set_embedder(model)
print(f"Embedder loaded: model2vec potion-base-8M, {len(ALL_VARIANTS)} variants")


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


SPECIFIC_MESSAGES = [
    "ok", "lol", "we're pregnant", "I GOT IT", "dad passed away",
    "I said yes", "haha", "nice", "sounds good", "I DID IT",
    "I HAD A BABY", "I GOT INTO UCLA",
]

RESULTS_DIR = Path(__file__).parent / "results"


def run_sweep():
    data = load_benchmark()
    conversations = data["conversations"]

    n_variants = len(ALL_VARIANTS)
    all_scores = {name: [] for name, _, _ in ALL_VARIANTS}
    all_labels = []  # 'S' or 'N' for each message
    all_cats = []
    msg_specific_scores = {name: {} for name, _, _ in ALL_VARIANTS}
    variant_times = {name: 0.0 for name, _, _ in ALL_VARIANTS}
    variant_counts = {name: 0 for name, _, _ in ALL_VARIANTS}
    variant_errors = {name: [] for name, _, _ in ALL_VARIANTS}

    total_msgs = 0

    for conv in conversations:
        messages = conv["messages"]
        conv_id = conv["conversation_id"]
        print(f"\nProcessing {conv_id}: {len(messages)} messages")

        # Per-conversation memory state
        memory_contents: list[str] = []
        memory_embeddings_list: list[np.ndarray] = []

        for msg_idx, msg in enumerate(messages):
            content = msg["content"]
            category = msg["category"]
            is_signal = msg["is_signal"]

            total_msgs += 1

            # Build memory embedding matrix
            if memory_embeddings_list:
                memory_embeddings = np.array(memory_embeddings_list)
            else:
                memory_embeddings = None

            # Pre-compute embedding for current message (shared by all variants)
            # Variants will re-compute this internally — but the embedding model
            # caches, so it's fast.

            # Score with all variants
            for name, desc, fn in ALL_VARIANTS:
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

                # Check specific messages
                if content in SPECIFIC_MESSAGES:
                    msg_specific_scores[name][content] = round(score, 4)

            # Record label
            if category.startswith("S"):
                all_labels.append("S")
            elif category.startswith("N"):
                all_labels.append("N")
            else:
                all_labels.append("B")  # borderline
            all_cats.append(category)

            # Add signal messages to memory (simulating correct gate behavior)
            if is_signal:
                memory_contents.append(content)
                emb = model.encode([content])[0]
                memory_embeddings_list.append(emb)

            if (msg_idx + 1) % 100 == 0:
                print(f"  {msg_idx + 1}/{len(messages)} messages processed, memory size: {len(memory_contents)}")

    print(f"\nTotal messages scored: {total_msgs}")

    # Compute metrics per variant
    results = []
    for name, desc, fn in ALL_VARIANTS:
        scores = all_scores[name]

        # Split into signal vs noise (exclude borderline)
        scores_signal = [s for s, l in zip(scores, all_labels) if l == "S"]
        scores_noise = [s for s, l in zip(scores, all_labels) if l == "N"]

        auc = compute_auc(scores_signal, scores_noise)

        # Per-category recall at threshold 0.3
        cat_recall = {}
        for cat in sorted(set(all_cats)):
            cat_scores = [s for s, c in zip(scores, all_cats) if c == cat]
            if cat_scores:
                recall = sum(1 for s in cat_scores if s >= 0.3) / len(cat_scores)
                cat_recall[cat] = round(recall, 4)

        # S4 recall
        s4_scores = [s for s, c in zip(scores, all_cats) if c == "S4"]
        s4_recall = sum(1 for s in s4_scores if s >= 0.3) / max(len(s4_scores), 1)

        # N* false positive rate
        n_scores = [s for s, l in zip(scores, all_labels) if l == "N"]
        n_fp = sum(1 for s in n_scores if s >= 0.3) / max(len(n_scores), 1)

        # Per-subcategory S recall
        s_cat_recall = {}
        for cat in ["S1", "S2", "S3", "S4", "S5"]:
            cat_s = [s for s, c in zip(scores, all_cats) if c == cat]
            if cat_s:
                s_cat_recall[cat] = round(sum(1 for s in cat_s if s >= 0.3) / len(cat_s), 4)

        ms_per_msg = (variant_times[name] / max(variant_counts[name], 1)) * 1000

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
            "errors": variant_errors[name] if variant_errors[name] else None,
        }
        results.append(result)

    # Sort by AUC descending
    results.sort(key=lambda r: r["auc"], reverse=True)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_variants": len(results),
        "num_messages": total_msgs,
        "num_signal": sum(1 for l in all_labels if l == "S"),
        "num_noise": sum(1 for l in all_labels if l == "N"),
        "num_borderline": sum(1 for l in all_labels if l == "B"),
        "results": results,
        "all_scores": {name: scores for name, scores in all_scores.items()},
    }

    out_path = RESULTS_DIR / "novelty_120_sweep.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print leaderboard
    print("\n" + "=" * 90)
    print(f"{'Rank':>4} {'Variant':<16} {'AUC':>6} {'S4_R':>5} {'N_FP':>5} {'ok':>5} {'IGOTIT':>6} {'ms':>6}  Description")
    print("=" * 90)
    for i, r in enumerate(results[:30]):
        ok_score = r["scores"].get("ok", "—")
        igotit_score = r["scores"].get("I GOT IT", "—")
        ok_str = f"{ok_score:.2f}" if isinstance(ok_score, float) else ok_score
        ig_str = f"{igotit_score:.2f}" if isinstance(igotit_score, float) else igotit_score
        print(f"{i+1:>4} {r['variant']:<16} {r['auc']:>6.3f} {r['s4_recall_03']:>5.2f} {r['n_fp_rate_03']:>5.2f} {ok_str:>5} {ig_str:>6} {r['ms_per_msg']:>6.2f}  {r['description'][:45]}")

    return output


if __name__ == "__main__":
    run_sweep()
