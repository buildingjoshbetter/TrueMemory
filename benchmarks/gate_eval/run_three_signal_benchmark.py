#!/usr/bin/env python3
"""
Three-signal gate benchmark on GateLoCoMo.

Runs the full encoding gate with shipped signals:
- Novelty: compression-based (v025)
- Salience: speech-act hybrid (encoding_salience_d)
- PE: cross-encoder embedding difference (v044)

Reports per-category breakdown (S1-S5, N1-N5, B1-B3) and overall AUC.
Uses paper-default weights (0.40/0.35/0.25).
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model2vec import StaticModel

model = StaticModel.from_pretrained("minishlab/potion-base-8M")

from benchmarks.gate_eval.novelty_sweep import set_embedder as set_novelty_embedder
from benchmarks.gate_eval.novelty_sweep import variant_025 as shipped_novelty
from benchmarks.gate_eval.pe_sweep_v2 import set_embedder as set_pe_embedder
from benchmarks.gate_eval.pe_sweep_v2 import variant_044 as shipped_pe
from truememory.ingest.encoding_salience import encoding_salience_d as shipped_salience

set_novelty_embedder(model)
set_pe_embedder(model)

# Gate weights (paper default)
W_N = 0.40
W_S = 0.35
W_PE = 0.25
THRESHOLD = 0.30


def load_benchmark():
    path = Path(__file__).parent / "datasets" / "gate_benchmark.json"
    with open(path) as f:
        return json.load(f)


def compute_auc(scores_signal, scores_noise):
    labels = [1] * len(scores_signal) + [0] * len(scores_noise)
    scores = list(scores_signal) + list(scores_noise)
    paired = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    auc = 0.0
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp_prev = fp_prev = 0
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
    return auc / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.5


def main():
    print("Three-Signal Gate Benchmark on GateLoCoMo")
    print("=" * 70)
    print(f"Weights: novelty={W_N}, salience={W_S}, PE={W_PE}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Novelty: compression (v025)")
    print(f"Salience: speech-act hybrid (encoding_salience_d)")
    print(f"PE: cross-encoder embedding diff (v044)")
    print()

    benchmark = load_benchmark()

    # Per-category tracking
    category_scores = defaultdict(list)  # category -> list of gate scores
    category_signals = defaultdict(list)  # novelty, salience, pe per category
    all_signal_scores = []
    all_noise_scores = []

    total_msgs = 0
    t0 = time.time()

    for conv in benchmark["conversations"]:
        memory_contents = []
        memory_embeddings = None

        for msg in conv["messages"]:
            content = msg["content"]
            category = msg.get("category", "?")

            if memory_contents:
                memory_embeddings = model.encode(memory_contents)
            else:
                memory_embeddings = None

            # Score each signal
            try:
                novelty = float(shipped_novelty(content, memory_contents, memory_embeddings))
            except Exception:
                novelty = 0.5
            salience = float(shipped_salience(content))
            try:
                pe = float(shipped_pe(content, memory_contents, memory_embeddings))
            except Exception:
                pe = 0.0

            # Gate score
            total_w = W_N + W_S + W_PE
            gate_score = (W_N * novelty + W_S * salience + W_PE * pe) / total_w
            gate_score = max(0.0, min(1.0, gate_score))

            category_scores[category].append(gate_score)
            category_signals[category].append((novelty, salience, pe, gate_score))

            if category.startswith("S"):
                all_signal_scores.append(gate_score)
            elif category.startswith("N"):
                all_noise_scores.append(gate_score)

            # Build memory with signal messages
            if category.startswith("S"):
                memory_contents.append(content)

            total_msgs += 1

    elapsed = time.time() - t0
    print(f"Scored {total_msgs} messages in {elapsed:.1f}s ({elapsed/total_msgs*1000:.1f}ms/msg)\n")

    # Overall AUC
    overall_auc = compute_auc(all_signal_scores, all_noise_scores)

    # Per-category stats
    print("=" * 70)
    print("PER-CATEGORY BREAKDOWN")
    print("=" * 70)
    print(f"{'Category':<10} {'Count':<7} {'Avg Gate':<10} {'Avg N':<8} {'Avg S':<8} {'Avg PE':<8} {'Encode%':<10}")
    print("-" * 70)

    sorted_cats = sorted(category_scores.keys())
    signal_cats = [c for c in sorted_cats if c.startswith("S")]
    noise_cats = [c for c in sorted_cats if c.startswith("N")]
    border_cats = [c for c in sorted_cats if c.startswith("B")]

    for group_name, cats in [("SIGNAL", signal_cats), ("NOISE", noise_cats), ("BORDERLINE", border_cats)]:
        print(f"\n  {group_name}:")
        group_scores = []
        for cat in cats:
            scores = category_scores[cat]
            signals = category_signals[cat]
            avg_gate = sum(scores) / len(scores)
            avg_n = sum(s[0] for s in signals) / len(signals)
            avg_s = sum(s[1] for s in signals) / len(signals)
            avg_pe = sum(s[2] for s in signals) / len(signals)
            encode_pct = sum(1 for s in scores if s >= THRESHOLD) / len(scores) * 100
            group_scores.extend(scores)

            print(f"  {cat:<10} {len(scores):<7} {avg_gate:<10.3f} {avg_n:<8.3f} {avg_s:<8.3f} {avg_pe:<8.3f} {encode_pct:<10.1f}%")

        if group_scores:
            avg_group = sum(group_scores) / len(group_scores)
            encode_group = sum(1 for s in group_scores if s >= THRESHOLD) / len(group_scores) * 100
            print(f"  {'TOTAL':<10} {len(group_scores):<7} {avg_group:<10.3f} {'':8} {'':8} {'':8} {encode_group:<10.1f}%")

    # S4 specific recall
    s4_scores = category_scores.get("S4", [])
    s4_recall = sum(1 for s in s4_scores if s >= THRESHOLD) / len(s4_scores) * 100 if s4_scores else 0

    # N* FP rate
    n_fp = sum(1 for s in all_noise_scores if s >= THRESHOLD) / len(all_noise_scores) * 100 if all_noise_scores else 0

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"  Gate AUC (S* vs N*):     {overall_auc:.4f}")
    print(f"  S* encode rate:          {sum(1 for s in all_signal_scores if s >= THRESHOLD) / len(all_signal_scores) * 100:.1f}%")
    print(f"  N* false positive rate:  {n_fp:.1f}%")
    print(f"  S4 recall (life events): {s4_recall:.1f}%")
    print(f"  Total signal messages:   {len(all_signal_scores)}")
    print(f"  Total noise messages:    {len(all_noise_scores)}")

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  No PE baseline:          ~0.792 AUC")
    print(f"  v1 keyword PE:           ~0.813 AUC")
    print(f"  This run (v044 PE):       {overall_auc:.4f} AUC")

    # Save results
    results = {
        "config": {
            "weights": {"novelty": W_N, "salience": W_S, "pe": W_PE},
            "threshold": THRESHOLD,
            "novelty_scorer": "compression_v025",
            "salience_scorer": "encoding_salience_d",
            "pe_scorer": "v044_embedding_diff",
        },
        "overall": {
            "auc": round(overall_auc, 4),
            "signal_encode_rate": round(sum(1 for s in all_signal_scores if s >= THRESHOLD) / len(all_signal_scores), 4),
            "noise_fp_rate": round(n_fp / 100, 4),
            "s4_recall": round(s4_recall / 100, 4),
        },
        "per_category": {},
    }

    for cat in sorted_cats:
        scores = category_scores[cat]
        signals = category_signals[cat]
        results["per_category"][cat] = {
            "count": len(scores),
            "avg_gate_score": round(sum(scores) / len(scores), 4),
            "avg_novelty": round(sum(s[0] for s in signals) / len(signals), 4),
            "avg_salience": round(sum(s[1] for s in signals) / len(signals), 4),
            "avg_pe": round(sum(s[2] for s in signals) / len(signals), 4),
            "encode_rate": round(sum(1 for s in scores if s >= THRESHOLD) / len(scores), 4),
        }

    out_path = Path(__file__).parent / "results" / "three_signal_gate_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
