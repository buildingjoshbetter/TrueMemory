#!/usr/bin/env python3
"""
Joint weight + threshold sweep for the three-signal encoding gate.

Tests all combinations of:
- Novelty weight: 0.25 to 0.60 (step 0.05)
- Salience weight: 0.15 to 0.45 (step 0.05)
- PE weight: 0.05 to 0.30 (step 0.05)
- Threshold: 0.20 to 0.40 (step 0.02)
- Salience floor: 0.05, 0.08, 0.10, 0.12, 0.15

Constraint: weights are normalized so they sum to 1.0.

Uses the shipped scorers:
- Novelty: compression (v025)
- Salience: encoding_salience_d (with expanded noise set)
- PE: v044 embedding pair-difference
"""

import json
import sys
import time
from collections import defaultdict
from itertools import product
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


def score_all_messages(benchmark):
    """Score every message with all three signals. Returns per-message scores."""
    msg_scores = {}

    for conv in benchmark["conversations"]:
        memory_contents = []
        memory_embeddings = None

        for msg in conv["messages"]:
            msg_id = msg.get("id", "")
            content = msg["content"]
            category = msg.get("category", "")

            if memory_contents:
                memory_embeddings = model.encode(memory_contents)
            else:
                memory_embeddings = None

            try:
                novelty = float(shipped_novelty(content, memory_contents, memory_embeddings))
            except Exception:
                novelty = 0.5
            salience = float(shipped_salience(content))
            try:
                pe = float(shipped_pe(content, memory_contents, memory_embeddings))
            except Exception:
                pe = 0.0

            msg_scores[msg_id] = {
                "novelty": novelty,
                "salience": salience,
                "pe": pe,
                "category": category,
            }

            if category.startswith("S"):
                memory_contents.append(content)

    return msg_scores


def run_sweep(msg_scores):
    """Grid search over weights, thresholds, and salience floors."""

    # Categorize messages
    signal_ids = [mid for mid, s in msg_scores.items() if s["category"].startswith("S")]
    noise_ids = [mid for mid, s in msg_scores.items() if s["category"].startswith("N")]
    s4_ids = [mid for mid, s in msg_scores.items() if s["category"] == "S4"]

    # Per-subcategory
    subcats = defaultdict(list)
    for mid, s in msg_scores.items():
        subcats[s["category"]].append(mid)

    # Weight grid
    n_weights = [round(x, 2) for x in np.arange(0.25, 0.65, 0.05)]
    s_weights = [round(x, 2) for x in np.arange(0.15, 0.50, 0.05)]
    p_weights = [round(x, 2) for x in np.arange(0.05, 0.35, 0.05)]
    thresholds = [round(x, 2) for x in np.arange(0.20, 0.42, 0.02)]
    sal_floors = [0.05, 0.08, 0.10, 0.12, 0.15]

    results = []
    total = 0

    for w_n, w_s, w_p in product(n_weights, s_weights, p_weights):
        total_w = w_n + w_s + w_p
        if total_w < 0.01:
            continue

        # Precompute gate scores for this weight combo
        gate_scores = {}
        for mid, s in msg_scores.items():
            gate_scores[mid] = (w_n * s["novelty"] + w_s * s["salience"] + w_p * s["pe"]) / total_w

        for threshold in thresholds:
            for sal_floor in sal_floors:
                # Compute decisions
                signal_gate = []
                noise_gate = []
                s4_pass = 0
                n_pass = 0

                for mid in signal_ids:
                    gs = gate_scores[mid]
                    sal = msg_scores[mid]["salience"]
                    passes = sal >= sal_floor and gs >= threshold
                    signal_gate.append(gs if sal >= sal_floor else 0.0)
                    if passes and msg_scores[mid]["category"] == "S4":
                        s4_pass += 1

                for mid in noise_ids:
                    gs = gate_scores[mid]
                    sal = msg_scores[mid]["salience"]
                    noise_gate.append(gs if sal >= sal_floor else 0.0)
                    if sal >= sal_floor and gs >= threshold:
                        n_pass += 1

                # AUC on gate scores (with floor applied)
                auc = compute_auc(signal_gate, noise_gate)

                s4_recall = s4_pass / len(s4_ids) if s4_ids else 0
                n_fp = n_pass / len(noise_ids) if noise_ids else 0
                s_encode = sum(1 for mid in signal_ids
                              if msg_scores[mid]["salience"] >= sal_floor
                              and gate_scores[mid] >= threshold) / len(signal_ids)

                # F1
                precision = s_encode / (s_encode + n_fp) if (s_encode + n_fp) > 0 else 0
                recall = s_encode
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                results.append({
                    "w_n": w_n, "w_s": w_s, "w_p": w_p,
                    "threshold": threshold, "sal_floor": sal_floor,
                    "auc": round(auc, 4),
                    "s4_recall": round(s4_recall, 3),
                    "n_fp_rate": round(n_fp, 3),
                    "s_encode_rate": round(s_encode, 3),
                    "f1": round(f1, 4),
                })
                total += 1

    return results, total


def main():
    print("Joint Weight + Threshold Sweep")
    print("=" * 70)

    benchmark = load_benchmark()

    print("Scoring all messages...")
    t0 = time.time()
    msg_scores = score_all_messages(benchmark)
    print(f"  Scored {len(msg_scores)} messages in {time.time() - t0:.1f}s")

    print("\nRunning sweep...")
    t0 = time.time()
    results, total = run_sweep(msg_scores)
    elapsed = time.time() - t0
    print(f"  Tested {total} configurations in {elapsed:.1f}s")

    # Sort by different criteria
    by_auc = sorted(results, key=lambda x: x["auc"], reverse=True)
    by_f1 = sorted(results, key=lambda x: x["f1"], reverse=True)

    # Filter: S4 recall > 80%
    by_auc_s4 = [r for r in by_auc if r["s4_recall"] >= 0.80]
    by_f1_s4 = [r for r in by_f1 if r["s4_recall"] >= 0.80]

    print("\n" + "=" * 100)
    print("TOP 15 BY AUC (S4 recall >= 80%)")
    print("=" * 100)
    print(f"{'Rank':<5} {'AUC':<8} {'F1':<8} {'S4':<6} {'S%':<6} {'NFP':<6} {'N':<6} {'S':<6} {'PE':<6} {'Thr':<6} {'Floor'}")
    print("-" * 100)
    for i, r in enumerate(by_auc_s4[:15]):
        print(f"{i+1:<5} {r['auc']:<8.4f} {r['f1']:<8.4f} {r['s4_recall']:<6.2f} "
              f"{r['s_encode_rate']:<6.2f} {r['n_fp_rate']:<6.2f} "
              f"{r['w_n']:<6.2f} {r['w_s']:<6.2f} {r['w_p']:<6.2f} "
              f"{r['threshold']:<6.2f} {r['sal_floor']}")

    print("\n" + "=" * 100)
    print("TOP 15 BY F1 (S4 recall >= 80%)")
    print("=" * 100)
    print(f"{'Rank':<5} {'AUC':<8} {'F1':<8} {'S4':<6} {'S%':<6} {'NFP':<6} {'N':<6} {'S':<6} {'PE':<6} {'Thr':<6} {'Floor'}")
    print("-" * 100)
    for i, r in enumerate(by_f1_s4[:15]):
        print(f"{i+1:<5} {r['auc']:<8.4f} {r['f1']:<8.4f} {r['s4_recall']:<6.2f} "
              f"{r['s_encode_rate']:<6.2f} {r['n_fp_rate']:<6.2f} "
              f"{r['w_n']:<6.2f} {r['w_s']:<6.2f} {r['w_p']:<6.2f} "
              f"{r['threshold']:<6.2f} {r['sal_floor']}")

    # Also show best unrestricted
    print("\n" + "=" * 100)
    print("TOP 5 BY AUC (unrestricted)")
    print("=" * 100)
    for i, r in enumerate(by_auc[:5]):
        print(f"{i+1:<5} AUC={r['auc']:.4f} F1={r['f1']:.4f} S4={r['s4_recall']:.2f} "
              f"S%={r['s_encode_rate']:.2f} NFP={r['n_fp_rate']:.2f} "
              f"N={r['w_n']:.2f} S={r['w_s']:.2f} PE={r['w_p']:.2f} "
              f"thr={r['threshold']:.2f} floor={r['sal_floor']}")

    # Old baseline comparison
    print("\n" + "=" * 100)
    print("COMPARISON TO PREVIOUS CONFIGS")
    print("=" * 100)
    print(f"  Previous (no floor, 0.40/0.35/0.25, thr=0.30): AUC ~0.796, S4=83.3%, NFP=25.4%")
    if by_auc_s4:
        best = by_auc_s4[0]
        print(f"  Best AUC (S4>=80%): AUC={best['auc']:.4f}, S4={best['s4_recall']:.1%}, NFP={best['n_fp_rate']:.1%}")
        print(f"    Config: N={best['w_n']:.2f} S={best['w_s']:.2f} PE={best['w_p']:.2f} "
              f"thr={best['threshold']:.2f} floor={best['sal_floor']}")
    if by_f1_s4:
        best_f1 = by_f1_s4[0]
        print(f"  Best F1 (S4>=80%):  F1={best_f1['f1']:.4f}, AUC={best_f1['auc']:.4f}, "
              f"S4={best_f1['s4_recall']:.1%}, NFP={best_f1['n_fp_rate']:.1%}")
        print(f"    Config: N={best_f1['w_n']:.2f} S={best_f1['w_s']:.2f} PE={best_f1['w_p']:.2f} "
              f"thr={best_f1['threshold']:.2f} floor={best_f1['sal_floor']}")

    # Save
    output = {
        "total_configs": total,
        "top_15_auc_s4_filtered": by_auc_s4[:15],
        "top_15_f1_s4_filtered": by_f1_s4[:15],
        "top_5_auc_unrestricted": by_auc[:5],
        "previous_baseline": {
            "weights": [0.40, 0.35, 0.25],
            "threshold": 0.30,
            "sal_floor": "none",
            "auc": 0.796,
            "s4_recall": 0.833,
            "n_fp_rate": 0.254,
        },
    }

    out_path = RESULTS_DIR / "weight_threshold_sweep.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
