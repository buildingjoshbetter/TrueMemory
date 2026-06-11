#!/usr/bin/env python3
"""
Benchmark four incremental optimizations on GateLoCoMo.

Tests each optimization individually and cumulatively:
1. Per-category threshold (corrections/decisions get lower threshold)
2. Entity density floor (reject messages with zero info markers)
3. Length fast path (short noise / long signal shortcuts)
4. Conversation position (greetings at session start penalized)

Uses the shipped scorers with the updated weights from PR #122.
"""

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model2vec import StaticModel
model = StaticModel.from_pretrained("minishlab/potion-base-8M")

from benchmarks.gate_eval.novelty_sweep import set_embedder as sn
from benchmarks.gate_eval.novelty_sweep import variant_025 as shipped_novelty
from benchmarks.gate_eval.pe_sweep_v2 import set_embedder as sp
from benchmarks.gate_eval.pe_sweep_v2 import variant_044 as shipped_pe
from truememory.ingest.encoding_salience import encoding_salience_d as shipped_salience

sn(model)
sp(model)

W_N, W_S, W_PE = 0.25, 0.20, 0.30
THRESHOLD = 0.26
SAL_FLOOR = 0.10


def load_benchmark():
    path = Path(__file__).parent / "datasets" / "gate_benchmark.json"
    with open(path) as f:
        return json.load(f)


def compute_auc(signal_scores, noise_scores):
    labels = [1] * len(signal_scores) + [0] * len(noise_scores)
    scores = list(signal_scores) + list(noise_scores)
    paired = sorted(zip(scores, labels), reverse=True)
    tp = fp = auc = 0.0
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp_prev = fp_prev = 0
    prev_score = None
    for score, label in paired:
        if score != prev_score and prev_score is not None:
            auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
            tp_prev, fp_prev = tp, fp
        tp += label
        fp += 1 - label
        prev_score = score
    auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
    return auc / (n_pos * n_neg)


_INFO_MARKER_RE = re.compile(r'[A-Z]{2,}|\d|[$€£]')
_PROPER_NOUN_RE = re.compile(r'\b[A-Z][a-z]+\b')

# Category-specific threshold overrides
_CATEGORY_THRESHOLDS = {
    "correction": 0.20,
    "decision": 0.22,
    "personal": 0.24,
    "relationship": 0.22,
}


def score_messages(benchmark):
    """Score every message and return structured data."""
    messages = []
    for conv in benchmark["conversations"]:
        memory_contents = []
        memory_embeddings = None
        msg_index_in_session = 0
        current_session = None

        for msg in conv["messages"]:
            content = msg["content"]
            category = msg.get("category", "")
            session = msg.get("session", "")

            if session != current_session:
                msg_index_in_session = 0
                current_session = session
            else:
                msg_index_in_session += 1

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

            total_w = W_N + W_S + W_PE
            gate_score = (W_N * novelty + W_S * salience + W_PE * pe) / total_w

            has_info_markers = bool(_INFO_MARKER_RE.search(content))
            has_proper_nouns = bool(_PROPER_NOUN_RE.search(content))
            length = len(content.strip())

            messages.append({
                "content": content,
                "category": category,
                "gate_score": gate_score,
                "salience": salience,
                "novelty": novelty,
                "pe": pe,
                "length": length,
                "has_info_markers": has_info_markers,
                "has_proper_nouns": has_proper_nouns,
                "msg_index_in_session": msg_index_in_session,
                "lm_category": msg.get("notes", "").lower(),
            })

            if category.startswith("S"):
                memory_contents.append(content)

    return messages


def apply_decision(msg, opts):
    """Apply the gate decision with optional optimizations."""
    sal = msg["salience"]
    score = msg["gate_score"]
    cat = msg["category"].lower() if msg.get("category") else ""
    length = msg["length"]
    idx = msg["msg_index_in_session"]

    # Salience floor (always on — from PR #122)
    if sal < SAL_FLOOR:
        return False

    threshold = THRESHOLD

    # Opt 1: Per-category threshold
    if opts.get("cat_threshold"):
        # Map benchmark categories to LLM extractor categories
        # S3 = corrections, S2 = decisions, S1 = personal, etc.
        if cat.startswith("s3") or "correction" in msg.get("lm_category", ""):
            threshold = _CATEGORY_THRESHOLDS.get("correction", THRESHOLD)
        elif cat.startswith("s2") or "decision" in msg.get("lm_category", ""):
            threshold = _CATEGORY_THRESHOLDS.get("decision", THRESHOLD)
        elif "relationship" in msg.get("lm_category", ""):
            threshold = _CATEGORY_THRESHOLDS.get("relationship", THRESHOLD)

    # Opt 2: Entity density floor
    if opts.get("entity_floor"):
        if length < 20 and not msg["has_info_markers"] and not msg["has_proper_nouns"]:
            return False

    # Opt 3: Length fast path
    if opts.get("length_path"):
        if length < 8 and sal < 0.15:
            return False
        if length > 120:
            threshold = max(0.18, threshold - 0.04)

    # Opt 4: Conversation position
    if opts.get("conv_position"):
        if idx <= 1 and sal < 0.20:
            return False

    return score >= threshold


def evaluate_config(messages, opts, label):
    """Run a config and print results."""
    signal = [m for m in messages if m["category"].startswith("S")]
    noise = [m for m in messages if m["category"].startswith("N")]
    s4 = [m for m in messages if m["category"] == "S4"]

    signal_decisions = [apply_decision(m, opts) for m in signal]
    noise_decisions = [apply_decision(m, opts) for m in noise]

    s_encode = sum(signal_decisions) / len(signal) if signal else 0
    n_fp = sum(noise_decisions) / len(noise) if noise else 0
    s4_recall = sum(apply_decision(m, opts) for m in s4) / len(s4) if s4 else 0

    # AUC using gate scores (with floor applied)
    sig_scores = [m["gate_score"] if m["salience"] >= SAL_FLOOR else 0.0 for m in signal]
    noi_scores = [m["gate_score"] if m["salience"] >= SAL_FLOOR else 0.0 for m in noise]
    auc = compute_auc(sig_scores, noi_scores)

    # Per subcategory
    subcats = defaultdict(list)
    for m in messages:
        if m["category"]:
            subcats[m["category"]].append(apply_decision(m, opts))

    return {
        "label": label,
        "auc": round(auc, 4),
        "s_encode": round(s_encode, 4),
        "n_fp": round(n_fp, 4),
        "s4_recall": round(s4_recall, 4),
        "per_cat": {cat: round(sum(decs) / len(decs), 3) if decs else 0
                    for cat, decs in sorted(subcats.items())},
    }


def print_result(r, prev=None):
    """Print one result row."""
    delta_s = f" ({r['s_encode'] - prev['s_encode']:+.1%})" if prev else ""
    delta_n = f" ({r['n_fp'] - prev['n_fp']:+.1%})" if prev else ""
    delta_auc = f" ({r['auc'] - prev['auc']:+.4f})" if prev else ""

    print(f"\n  {r['label']}")
    print(f"    AUC: {r['auc']:.4f}{delta_auc}   S encode: {r['s_encode']:.1%}{delta_s}   "
          f"N FP: {r['n_fp']:.1%}{delta_n}   S4: {r['s4_recall']:.1%}")

    cats = r["per_cat"]
    signal_cats = sorted(c for c in cats if c.startswith("S"))
    noise_cats = sorted(c for c in cats if c.startswith("N"))

    print(f"    Signal:  ", end="")
    for c in signal_cats:
        print(f"{c}={cats[c]:.0%}  ", end="")
    print()
    print(f"    Noise:   ", end="")
    for c in noise_cats:
        print(f"{c}={cats[c]:.0%}  ", end="")
    print()


def main():
    print("Incremental Optimization Benchmark")
    print("=" * 70)

    benchmark = load_benchmark()
    print("Scoring messages...")
    t0 = time.time()
    messages = score_messages(benchmark)
    print(f"  {len(messages)} messages scored in {time.time()-t0:.1f}s\n")

    # Baseline (PR #122 config)
    baseline = evaluate_config(messages, {}, "Baseline (PR #122)")
    print_result(baseline)

    # Each optimization individually
    configs = [
        ({"cat_threshold": True}, "1. Per-category threshold"),
        ({"entity_floor": True}, "2. Entity density floor"),
        ({"length_path": True}, "3. Length fast path"),
        ({"conv_position": True}, "4. Conversation position"),
    ]

    prev = baseline
    individual_results = []
    for opts, label in configs:
        r = evaluate_config(messages, opts, label)
        print_result(r, baseline)
        individual_results.append(r)

    # Cumulative
    print("\n" + "=" * 70)
    print("CUMULATIVE (stacking one at a time)")
    print("=" * 70)

    cumulative_opts = {}
    prev = baseline
    print_result(baseline)
    for opts, label in configs:
        cumulative_opts.update(opts)
        r = evaluate_config(messages, dict(cumulative_opts), f"+ {label}")
        print_result(r, prev)
        prev = r

    # All four together
    all_opts = {k: True for opt, _ in configs for k, v in opt.items()}
    final = evaluate_config(messages, all_opts, "ALL FOUR combined")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print_result(baseline)
    print_result(final, baseline)

    # Save
    output = {
        "baseline": baseline,
        "individual": individual_results,
        "all_combined": final,
    }
    out_path = Path(__file__).parent / "results" / "incremental_optimizations.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
