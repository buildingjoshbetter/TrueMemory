#!/usr/bin/env python3
"""
Model-free PE combination sweep.

Tests all pairs, triples, and quads of the top 10 model-free PE variants
(from both v1 and v2 sweeps) using mean, max, and AUC-weighted combination
methods. Then runs three-signal gate ablation for the best combos.

Total configs: ~1,125 combinations.
"""

import json
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Load embedding model (shared by v1 and v2 variants)
# ---------------------------------------------------------------------------
from model2vec import StaticModel

model = StaticModel.from_pretrained("minishlab/potion-base-8M")

# Set embedders for both sweep modules
from benchmarks.gate_eval.pe_sweep import set_embedder as set_v1_embedder
from benchmarks.gate_eval.pe_sweep_v2 import set_embedder as set_v2_embedder

set_v1_embedder(model)
set_v2_embedder(model)

# ---------------------------------------------------------------------------
# Import the 10 model-free variants
# ---------------------------------------------------------------------------
from benchmarks.gate_eval.pe_sweep import (
    variant_020 as v1_020,
    variant_033 as v1_033,
    variant_070 as v1_070,
)
from benchmarks.gate_eval.pe_sweep_v2 import (
    variant_011 as v2_011,
    variant_024 as v2_024,
    variant_044 as v2_044,
    variant_052 as v2_052,
    variant_070 as v2_070,
    variant_075 as v2_075,
    variant_081 as v2_081,
)

# Shipped novelty scorer: PPM compression cost (variant_025)
from benchmarks.gate_eval.novelty_sweep import (
    set_embedder as set_novelty_embedder,
    variant_025 as shipped_novelty,
)

set_novelty_embedder(model)

# Shipped salience scorer
from truememory.ingest.encoding_salience import encoding_salience_d as shipped_salience

VARIANTS = {
    "v044": (v2_044, 0.745),
    "v1_070": (v1_070, 0.626),
    "v1_033": (v1_033, 0.609),
    "v1_020": (v1_020, 0.580),
    "v2_070": (v2_070, 0.574),
    "v2_011": (v2_011, 0.553),
    "v2_075": (v2_075, 0.547),
    "v2_052": (v2_052, 0.533),
    "v2_024": (v2_024, 0.533),
    "v2_081": (v2_081, 0.521),
}

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
    prev_score = None
    tp_prev = fp_prev = 0
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
    """Score every message with every variant. Returns {variant_name: {msg_id: score}}."""
    all_scores = {name: {} for name in VARIANTS}
    novelty_scores = {}
    salience_scores = {}

    for conv in benchmark["conversations"]:
        conv_id = conv["conversation_id"]
        memory_contents = []
        memory_embeddings = None

        for msg in conv["messages"]:
            msg_id = msg.get("id", f"{conv_id}_{msg.get('message_id', '')}")
            content = msg["content"]

            # Compute embeddings for memory
            if memory_contents:
                memory_embeddings = model.encode(memory_contents)
            else:
                memory_embeddings = None

            # Score with each PE variant
            for name, (fn, _auc) in VARIANTS.items():
                try:
                    score = fn(content, memory_contents, memory_embeddings)
                    all_scores[name][msg_id] = float(score)
                except Exception:
                    all_scores[name][msg_id] = 0.0

            # Score with shipped novelty and salience
            try:
                novelty_scores[msg_id] = float(shipped_novelty(content, memory_contents, memory_embeddings))
            except Exception:
                novelty_scores[msg_id] = 0.5
            salience_scores[msg_id] = float(shipped_salience(content))

            # Add signal messages to memory
            cat = msg.get("category", "")
            if cat.startswith("S"):
                memory_contents.append(content)

    return all_scores, novelty_scores, salience_scores


def get_signal_noise_ids(benchmark):
    """Get message IDs for signal (S*) and noise (N*) categories."""
    signal_ids = []
    noise_ids = []
    for conv in benchmark["conversations"]:
        conv_id = conv["conversation_id"]
        for msg in conv["messages"]:
            msg_id = msg.get("id", f"{conv_id}_{msg.get('message_id', '')}")
            cat = msg.get("category", "")
            if cat.startswith("S"):
                signal_ids.append(msg_id)
            elif cat.startswith("N"):
                noise_ids.append(msg_id)
    return signal_ids, noise_ids


def combine_scores(score_dicts, method, aucs=None):
    """Combine multiple variant score dicts into one using the given method."""
    all_msg_ids = set()
    for d in score_dicts:
        all_msg_ids.update(d.keys())

    combined = {}
    for msg_id in all_msg_ids:
        values = [d.get(msg_id, 0.0) for d in score_dicts]
        if method == "mean":
            combined[msg_id] = sum(values) / len(values)
        elif method == "max":
            combined[msg_id] = max(values)
        elif method == "weighted" and aucs is not None:
            total_w = sum(aucs)
            if total_w > 0:
                combined[msg_id] = sum(v * w for v, w in zip(values, aucs)) / total_w
            else:
                combined[msg_id] = sum(values) / len(values)
    return combined


def compute_gate_auc(novelty_scores, salience_scores, pe_scores, weights, signal_ids, noise_ids, threshold=0.30):
    """Compute three-signal gate AUC."""
    w_n, w_s, w_pe = weights
    total_w = w_n + w_s + w_pe
    if total_w == 0:
        total_w = 1.0

    signal_gate = []
    noise_gate = []

    for msg_id in signal_ids:
        n = novelty_scores.get(msg_id, 0.5)
        s = salience_scores.get(msg_id, 0.5)
        p = pe_scores.get(msg_id, 0.0)
        gate = (w_n * n + w_s * s + w_pe * p) / total_w
        signal_gate.append(gate)

    for msg_id in noise_ids:
        n = novelty_scores.get(msg_id, 0.5)
        s = salience_scores.get(msg_id, 0.5)
        p = pe_scores.get(msg_id, 0.0)
        gate = (w_n * n + w_s * s + w_pe * p) / total_w
        noise_gate.append(gate)

    auc = compute_auc(signal_gate, noise_gate)

    # S4 recall and N FP rate at threshold
    s4_above = sum(1 for s in signal_gate if s >= threshold)
    s4_recall = s4_above / len(signal_gate) if signal_gate else 0
    n_above = sum(1 for s in noise_gate if s >= threshold)
    n_fp = n_above / len(noise_gate) if noise_gate else 0

    return auc, s4_recall, n_fp


def main():
    print("Loading benchmark...")
    benchmark = load_benchmark()
    signal_ids, noise_ids = get_signal_noise_ids(benchmark)
    print(f"  {len(signal_ids)} signal, {len(noise_ids)} noise messages")

    print("\nScoring all messages with 10 model-free variants...")
    t0 = time.time()
    all_scores, novelty_scores, salience_scores = score_all_messages(benchmark)
    elapsed = time.time() - t0
    print(f"  Scoring complete in {elapsed:.1f}s")

    # Compute individual AUCs
    print("\nIndividual variant AUCs:")
    individual_aucs = {}
    for name in VARIANTS:
        sig = [all_scores[name].get(mid, 0.0) for mid in signal_ids]
        noi = [all_scores[name].get(mid, 0.0) for mid in noise_ids]
        auc = compute_auc(sig, noi)
        individual_aucs[name] = auc
        print(f"  {name}: {auc:.4f}")

    # N+S baseline
    ns_auc, ns_s4, ns_fp = compute_gate_auc(
        novelty_scores, salience_scores,
        {mid: 0.0 for mid in list(novelty_scores.keys())},
        (0.55, 0.45, 0.0), signal_ids, noise_ids,
    )
    print(f"\nN+S baseline: AUC {ns_auc:.4f}, S4 recall {ns_s4:.3f}, N FP {ns_fp:.3f}")

    # ---------------------------------------------------------------------------
    # Test all combinations
    # ---------------------------------------------------------------------------
    variant_names = list(VARIANTS.keys())
    methods = ["mean", "max", "weighted"]
    weight_configs = [
        (0.50, 0.25, 0.25),
        (0.40, 0.35, 0.25),
        (0.33, 0.33, 0.33),
        (0.45, 0.30, 0.25),
        (0.35, 0.35, 0.30),
    ]

    results = []
    total_combos = 0

    for combo_size in [2, 3, 4]:
        combos = list(combinations(variant_names, combo_size))
        print(f"\nTesting {len(combos)} {combo_size}-way combinations × {len(methods)} methods...")

        for combo in combos:
            score_dicts = [all_scores[name] for name in combo]
            aucs = [individual_aucs[name] for name in combo]

            for method in methods:
                combined = combine_scores(score_dicts, method, aucs if method == "weighted" else None)

                # PE-only AUC
                sig = [combined.get(mid, 0.0) for mid in signal_ids]
                noi = [combined.get(mid, 0.0) for mid in noise_ids]
                pe_auc = compute_auc(sig, noi)

                results.append({
                    "variants": list(combo),
                    "size": combo_size,
                    "method": method,
                    "pe_auc": round(pe_auc, 4),
                })
                total_combos += 1

    print(f"\nTotal combos tested: {total_combos}")

    # Sort by PE AUC
    results.sort(key=lambda x: x["pe_auc"], reverse=True)

    # Print top 30
    print("\n" + "=" * 90)
    print("TOP 30 MODEL-FREE PE COMBINATIONS")
    print("=" * 90)
    print(f"{'Rank':<5} {'AUC':<8} {'Method':<10} {'Variants'}")
    print("-" * 90)
    for i, r in enumerate(results[:30]):
        print(f"{i+1:<5} {r['pe_auc']:<8.4f} {r['method']:<10} {' + '.join(r['variants'])}")

    # ---------------------------------------------------------------------------
    # Three-signal gate ablation for top 10 combos
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("THREE-SIGNAL GATE ABLATION (top 10 model-free combos)")
    print("=" * 90)

    ablation_results = []
    seen_combos = set()

    for r in results[:30]:
        combo_key = tuple(sorted(r["variants"])) + (r["method"],)
        if combo_key in seen_combos:
            continue
        if len(ablation_results) >= 10:
            break
        seen_combos.add(combo_key)

        score_dicts = [all_scores[name] for name in r["variants"]]
        aucs_for_combo = [individual_aucs[name] for name in r["variants"]]
        combined = combine_scores(
            score_dicts, r["method"],
            aucs_for_combo if r["method"] == "weighted" else None,
        )

        best_gate_auc = 0
        best_config = None
        all_configs = []

        for weights in weight_configs:
            gate_auc, s4, nfp = compute_gate_auc(
                novelty_scores, salience_scores, combined,
                weights, signal_ids, noise_ids,
            )
            config_result = {
                "weights": list(weights),
                "gate_auc": round(gate_auc, 4),
                "s4_recall": round(s4, 3),
                "n_fp_rate": round(nfp, 3),
            }
            all_configs.append(config_result)
            if gate_auc > best_gate_auc:
                best_gate_auc = gate_auc
                best_config = config_result

        ablation_entry = {
            "variants": r["variants"],
            "method": r["method"],
            "pe_auc": r["pe_auc"],
            "best_gate_auc": round(best_gate_auc, 4),
            "best_weights": best_config["weights"],
            "best_s4_recall": best_config["s4_recall"],
            "best_n_fp_rate": best_config["n_fp_rate"],
            "all_configs": all_configs,
        }
        ablation_results.append(ablation_entry)

        print(f"\n  {' + '.join(r['variants'])} ({r['method']})")
        print(f"  PE AUC: {r['pe_auc']:.4f}")
        for cfg in all_configs:
            w = cfg["weights"]
            marker = " <<<" if cfg["gate_auc"] == best_gate_auc else ""
            print(f"    N={w[0]:.2f} S={w[1]:.2f} PE={w[2]:.2f} → Gate AUC {cfg['gate_auc']:.4f}  "
                  f"S4={cfg['s4_recall']:.3f}  NFP={cfg['n_fp_rate']:.3f}{marker}")

    # ---------------------------------------------------------------------------
    # Save everything
    # ---------------------------------------------------------------------------
    output = {
        "individual_aucs": {k: round(v, 4) for k, v in individual_aucs.items()},
        "ns_baseline": {"auc": round(ns_auc, 4), "s4_recall": round(ns_s4, 3), "n_fp_rate": round(ns_fp, 3)},
        "top_50_combos": results[:50],
        "ablation_top_10": ablation_results,
        "total_combos_tested": total_combos,
    }

    out_path = RESULTS_DIR / "pe_modelfree_combos.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"N+S baseline (no PE):          Gate AUC {ns_auc:.4f}")
    if ablation_results:
        best = max(ablation_results, key=lambda x: x["best_gate_auc"])
        print(f"Best model-free combo:         Gate AUC {best['best_gate_auc']:.4f}  "
              f"({' + '.join(best['variants'])}, {best['method']})")
        print(f"  Weights: N={best['best_weights'][0]:.2f} S={best['best_weights'][1]:.2f} PE={best['best_weights'][2]:.2f}")
        print(f"  S4 recall: {best['best_s4_recall']:.3f}")
        print(f"  N FP rate: {best['best_n_fp_rate']:.3f}")
    print(f"v2 NLI combo (v003+v044):      Gate AUC 0.839  (for reference)")
    print(f"v1 best (v033 keywords):       Gate AUC 0.813  (for reference)")


if __name__ == "__main__":
    main()
