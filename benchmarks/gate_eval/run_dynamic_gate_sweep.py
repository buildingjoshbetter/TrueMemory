#!/usr/bin/env python3
"""
Dynamic GateLoCoMo: Per-category weight profile sweep.

Tests whether per-category weight profiles improve the encoding gate beyond
static weights. Sweeps profile counts (1, 2, 3, 5, 7) and compares to
existing category-specific mechanisms (threshold overrides, salience boost).

Usage:
    .venv/bin/python3 benchmarks/gate_eval/run_dynamic_gate_sweep.py
"""

from __future__ import annotations

import gzip
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_PATH = PROJECT_ROOT / "benchmarks" / "gate_eval" / "datasets" / "gate_benchmark.json"
DYNAMIC_BENCHMARK_PATH = PROJECT_ROOT / "benchmarks" / "gate_eval" / "datasets" / "gate_benchmark_dynamic.json"
SCORES_PATH = PROJECT_ROOT / "benchmarks" / "gate_eval" / "results" / "dynamic_gate_scores.json"
SWEEP_PATH = PROJECT_ROOT / "benchmarks" / "gate_eval" / "results" / "dynamic_profile_sweep.json"
AUDIT_PATH = PROJECT_ROOT / "benchmarks" / "gate_eval" / "results" / "dynamic_noise_audit.json"
COMPARE_PATH = PROJECT_ROOT / "benchmarks" / "gate_eval" / "results" / "dynamic_vs_existing.json"
REPORT_PATH = PROJECT_ROOT / "_working" / "memorist" / "gate_functional" / "DYNAMIC_WEIGHTING_REPORT.md"
RUSTLE_PATH = PROJECT_ROOT / "_working" / "memorist" / "gate_functional" / "DYNAMIC_WEIGHTING_RUSTLE.md"


# ============================================================================
# PHASE 1: Category classifiers
# ============================================================================

def classify_content_only(content: str) -> str:
    lower = content.lower()
    if any(w in lower for w in ["actually", "no longer", "not anymore", "switched",
                                 "changed", "turns out", "correction", "i was wrong"]):
        return "correction"
    if any(w in lower for w in ["dating", "boyfriend", "girlfriend", "engaged", "married",
                                 "broke up", "seeing someone", "wife", "husband", "partner"]):
        return "relationship"
    if any(w in lower for w in ["prefer", "love", "hate", "favorite", "fav", "best",
                                 "worst", "rather", "can't stand"]):
        return "preference"
    if any(w in lower for w in ["decided", "going to", "gonna", "planning to", "committed",
                                 "booked", "signed up", "enrolled"]):
        return "decision"
    if any(w in lower for w in ["monday", "tuesday", "wednesday", "thursday", "friday",
                                 "saturday", "sunday", "january", "february", "march",
                                 "april", "may", "june", "july", "august", "september",
                                 "october", "november", "december", "next week", "next month",
                                 "next year", "tomorrow", "yesterday", "tonight"]):
        return "temporal"
    if any(w in lower for w in ["lives in", "live in", "work at", "works at", "moved to",
                                 "years old", "name is", "i am a", "i'm a"]):
        return "personal"
    if any(w in lower for w in ["deploy", "stack", "database", "api", "server", "code",
                                 "repo", "branch", "commit"]):
        return "technical"
    return "general"


def classify_notes_assisted(content: str, notes: str) -> str:
    notes_lower = (notes or "").lower()
    if any(w in notes_lower for w in ["correction", "update", "corrects", "revises"]):
        return "correction"
    if any(w in notes_lower for w in ["preference", "favorite", "likes", "dislikes", "opinion"]):
        return "preference"
    if any(w in notes_lower for w in ["relationship", "dating", "partner", "engaged",
                                       "married", "romantic"]):
        return "relationship"
    if any(w in notes_lower for w in ["decision", "commitment", "committed", "chose",
                                       "plan", "intent"]):
        return "decision"
    if any(w in notes_lower for w in ["date", "timeline", "deadline", "schedule",
                                       "timing", "when"]):
        return "temporal"
    if any(w in notes_lower for w in ["fact", "personal", "identity", "location",
                                       "job", "salary", "name"]):
        return "personal"
    if any(w in notes_lower for w in ["technical", "stack", "architecture", "tool"]):
        return "technical"
    return classify_content_only(content)


def phase1_annotate():
    print("\n" + "=" * 70)
    print("PHASE 1: Annotate GateLoCoMo with extractor categories")
    print("=" * 70)

    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    cat_a_counts = Counter()
    cat_b_counts = Counter()
    cross_tab_a = defaultdict(Counter)
    cross_tab_b = defaultdict(Counter)

    for conv in benchmark["conversations"]:
        for msg in conv["messages"]:
            content = msg["content"]
            notes = msg.get("notes", "")
            cat = msg["category"]

            cat_a = classify_content_only(content)
            cat_b = classify_notes_assisted(content, notes)

            msg["extractor_category_a"] = cat_a
            msg["extractor_category_b"] = cat_b

            cat_a_counts[cat_a] += 1
            cat_b_counts[cat_b] += 1

            label_prefix = cat[:2] if len(cat) >= 2 else cat
            cross_tab_a[label_prefix][cat_a] += 1
            cross_tab_b[label_prefix][cat_b] += 1

    # Print cross-tabulations
    all_ext_cats = sorted(set(list(cat_a_counts.keys()) + list(cat_b_counts.keys())))
    bench_cats = ["S1", "S2", "S3", "S4", "S5", "N1", "N2", "N3", "N4", "N5", "B1", "B2", "B3"]

    for label, cross_tab, name in [
        ("Classifier A (content-only)", cross_tab_a, "A"),
        ("Classifier B (notes-assisted)", cross_tab_b, "B"),
    ]:
        print(f"\n{label}:")
        header = f"{'Cat':>4} | " + " | ".join(f"{c:>10}" for c in all_ext_cats) + " | Total"
        print(header)
        print("-" * len(header))
        for bc in bench_cats:
            row = cross_tab[bc]
            total = sum(row.values())
            cells = " | ".join(f"{row.get(c, 0):>10}" for c in all_ext_cats)
            print(f"{bc:>4} | {cells} | {total:>5}")

    # Validation
    print("\n--- Validation ---")
    s3_total = sum(cross_tab_a["S3"].values())
    s3_corr_a = cross_tab_a["S3"].get("correction", 0)
    s3_corr_b = cross_tab_b["S3"].get("correction", 0)
    print(f"S3 → correction: A={s3_corr_a}/{s3_total} ({s3_corr_a/max(1,s3_total)*100:.0f}%), "
          f"B={s3_corr_b}/{s3_total} ({s3_corr_b/max(1,s3_total)*100:.0f}%)")

    s4_total = sum(cross_tab_a["S4"].values())
    s4_pers_rel_a = cross_tab_a["S4"].get("personal", 0) + cross_tab_a["S4"].get("relationship", 0)
    s4_pers_rel_b = cross_tab_b["S4"].get("personal", 0) + cross_tab_b["S4"].get("relationship", 0)
    print(f"S4 → personal+relationship: A={s4_pers_rel_a}/{s4_total} ({s4_pers_rel_a/max(1,s4_total)*100:.0f}%), "
          f"B={s4_pers_rel_b}/{s4_total} ({s4_pers_rel_b/max(1,s4_total)*100:.0f}%)")

    n1_total = sum(cross_tab_a["N1"].values())
    n1_gen_a = cross_tab_a["N1"].get("general", 0)
    n1_gen_b = cross_tab_b["N1"].get("general", 0)
    print(f"N1 → general: A={n1_gen_a}/{n1_total} ({n1_gen_a/max(1,n1_total)*100:.0f}%), "
          f"B={n1_gen_b}/{n1_total} ({n1_gen_b/max(1,n1_total)*100:.0f}%)")

    gen_a = cat_a_counts.get("general", 0)
    gen_b = cat_b_counts.get("general", 0)
    print(f"Total 'general': A={gen_a}, B={gen_b}")

    DYNAMIC_BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DYNAMIC_BENCHMARK_PATH, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\nSaved annotated benchmark to {DYNAMIC_BENCHMARK_PATH}")

    return benchmark


# ============================================================================
# PHASE 2: Score all messages
# ============================================================================

# --- Novelty scorer (variant_025) ---

def _gz_len(text: str) -> int:
    return len(gzip.compress(text.encode()))


def score_novelty_025(content: str, memory_contents: list[str]) -> float:
    if not memory_contents:
        return 1.0
    corpus = " ".join(memory_contents[-30:])
    c_corpus = _gz_len(corpus)
    c_combined = _gz_len(corpus + " " + content)
    cost = c_combined - c_corpus
    c_msg = _gz_len(content)
    if c_msg < 1:
        return 0.0
    return max(0.0, min(1.0, cost / (c_msg + 10)))


# --- Salience scorer (encoding_salience_d) ---

def score_salience_d(content: str) -> float:
    from truememory.ingest.encoding_salience import encoding_salience_d
    return encoding_salience_d(content, "")


# --- PE scorer (variant_044) ---

_NOISE_EXACT_PE = frozenset({
    "ok", "okay", "k", "kk", "yes", "yeah", "yep", "yup", "ya", "yea",
    "no", "nah", "nope", "lol", "lmao", "lmfao", "haha", "hahaha", "heh",
    "omg", "omfg", "wtf", "nice", "cool", "dope", "sick", "lit", "fire",
    "thanks", "thx", "ty", "thank you", "got it", "gotcha",
    "sounds good", "sounds great", "bet", "word", "sure", "for sure",
    "same", "mood", "idk", "idc", "np", "no problem",
    "gn", "goodnight", "good night", "gm", "good morning", "brb", "ttyl",
    "damn", "dude", "bro", "ugh", "wow", "yikes", "ooh", "oof",
    "true", "facts", "right", "exactly", "totally", "absolutely",
    "lmao dead", "im dead", "crying", "screaming", "yo", "heyyy", "hey",
    "hi", "hello", "sup", "what's up",
})


def _is_noise_pe(text: str) -> bool:
    return text.lower().strip().rstrip("!?.… ") in _NOISE_EXACT_PE or len(text.strip()) < 3


def _cosine_sim(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b)) / (na * nb)


def score_pe_044(content: str, memory_contents: list[str],
                 memory_embeddings: np.ndarray, embedder) -> float:
    if _is_noise_pe(content) or not memory_contents or memory_embeddings is None:
        return 0.0
    emb = embedder.encode([content])[0]
    sims = np.dot(memory_embeddings / np.linalg.norm(memory_embeddings, axis=1, keepdims=True),
                  emb / max(np.linalg.norm(emb), 1e-10))
    idx = int(np.argmax(sims))
    sim = float(sims[idx])
    if sim < 0.2:
        return 0.0
    mem = memory_contents[idx]
    pair_emb = embedder.encode([content + " [SEP] " + mem])[0]
    self_emb = embedder.encode([mem + " [SEP] " + mem])[0]
    dist = 1.0 - _cosine_sim(pair_emb, self_emb)
    return max(0.0, min(1.0, dist))


def phase2_score(benchmark: dict):
    print("\n" + "=" * 70)
    print("PHASE 2: Score all messages with incremental memory")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    all_scores = []
    t0 = time.time()

    for conv in benchmark["conversations"]:
        conv_id = conv["conversation_id"]
        memory_contents: list[str] = []
        memory_embeddings_list: list[np.ndarray] = []
        memory_embeddings: np.ndarray | None = None

        print(f"  Scoring {conv_id} ({len(conv['messages'])} messages)...")

        for i, msg in enumerate(conv["messages"]):
            content = msg["content"]

            # Score novelty
            novelty = score_novelty_025(content, memory_contents)

            # Score salience (category-independent)
            salience = score_salience_d(content)

            # Score PE
            pe = score_pe_044(content, memory_contents, memory_embeddings, embedder)

            entry = {
                "id": msg["id"],
                "conversation_id": conv_id,
                "content": content,
                "category": msg["category"],
                "is_signal": msg.get("is_signal", False),
                "extractor_category_a": msg.get("extractor_category_a", "general"),
                "extractor_category_b": msg.get("extractor_category_b", "general"),
                "novelty": round(novelty, 4),
                "salience": round(salience, 4),
                "pe": round(pe, 4),
            }
            all_scores.append(entry)

            # Add signal messages to memory for subsequent scoring
            if msg.get("is_signal", False):
                memory_contents.append(content)
                emb = embedder.encode([content])[0]
                memory_embeddings_list.append(emb)
                memory_embeddings = np.array(memory_embeddings_list)

            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(conv['messages'])} scored")

    elapsed = time.time() - t0
    print(f"\n  Scored {len(all_scores)} messages in {elapsed:.1f}s")

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_PATH, "w") as f:
        json.dump({"scores": all_scores, "total": len(all_scores)}, f, indent=2)
    print(f"  Saved to {SCORES_PATH}")

    return all_scores


# ============================================================================
# PHASE 3: Profile count sweep
# ============================================================================

SALIENCE_FLOOR = 0.10

PROFILE_CONFIGS = {
    "config_1": {
        "name": "1 profile (baseline)",
        "profiles": {
            "default": {"w_n": 0.25, "w_s": 0.20, "w_pe": 0.30, "threshold": 0.26,
                        "categories": ["correction", "temporal", "personal", "preference",
                                       "decision", "relationship", "technical", "general"]},
        }
    },
    "config_2": {
        "name": "2 profiles",
        "profiles": {
            "high-value": {"w_n": 0.20, "w_s": 0.35, "w_pe": 0.20, "threshold": 0.22,
                           "categories": ["correction", "decision", "relationship"]},
            "default": {"w_n": 0.25, "w_s": 0.20, "w_pe": 0.30, "threshold": 0.26,
                        "categories": ["personal", "preference", "temporal", "technical", "general"]},
        }
    },
    "config_3": {
        "name": "3 profiles",
        "profiles": {
            "update": {"w_n": 0.40, "w_s": 0.15, "w_pe": 0.20, "threshold": 0.22,
                       "categories": ["correction", "temporal"]},
            "importance": {"w_n": 0.20, "w_s": 0.35, "w_pe": 0.20, "threshold": 0.24,
                           "categories": ["decision", "relationship", "personal"]},
            "default": {"w_n": 0.25, "w_s": 0.20, "w_pe": 0.30, "threshold": 0.26,
                        "categories": ["preference", "technical", "general"]},
        }
    },
    "config_4": {
        "name": "5 profiles",
        "profiles": {
            "correction": {"w_n": 0.20, "w_s": 0.15, "w_pe": 0.40, "threshold": 0.20,
                           "categories": ["correction"]},
            "temporal": {"w_n": 0.40, "w_s": 0.15, "w_pe": 0.20, "threshold": 0.22,
                         "categories": ["temporal"]},
            "relationship": {"w_n": 0.20, "w_s": 0.40, "w_pe": 0.15, "threshold": 0.22,
                             "categories": ["relationship"]},
            "decision+personal": {"w_n": 0.25, "w_s": 0.30, "w_pe": 0.20, "threshold": 0.24,
                                  "categories": ["decision", "personal"]},
            "default": {"w_n": 0.25, "w_s": 0.20, "w_pe": 0.30, "threshold": 0.26,
                        "categories": ["preference", "technical", "general"]},
        }
    },
    "config_5": {
        "name": "7 profiles",
        "profiles": {
            "correction": {"w_n": 0.20, "w_s": 0.15, "w_pe": 0.40, "threshold": 0.20,
                           "categories": ["correction"]},
            "temporal": {"w_n": 0.40, "w_s": 0.15, "w_pe": 0.20, "threshold": 0.22,
                         "categories": ["temporal"]},
            "personal": {"w_n": 0.25, "w_s": 0.30, "w_pe": 0.20, "threshold": 0.24,
                         "categories": ["personal"]},
            "preference": {"w_n": 0.25, "w_s": 0.30, "w_pe": 0.20, "threshold": 0.24,
                           "categories": ["preference"]},
            "decision": {"w_n": 0.20, "w_s": 0.35, "w_pe": 0.20, "threshold": 0.22,
                         "categories": ["decision"]},
            "relationship": {"w_n": 0.20, "w_s": 0.40, "w_pe": 0.15, "threshold": 0.22,
                             "categories": ["relationship"]},
            "technical": {"w_n": 0.30, "w_s": 0.20, "w_pe": 0.25, "threshold": 0.26,
                          "categories": ["technical"]},
            "general": {"w_n": 0.25, "w_s": 0.20, "w_pe": 0.30, "threshold": 0.26,
                        "categories": ["general"]},
        }
    },
}


# Category → Salience boost (from encoding_gate.py)
CATEGORY_SALIENCE_BOOST = {
    "correction": 0.40, "decision": 0.30, "personal": 0.25,
    "preference": 0.25, "relationship": 0.20, "temporal": 0.15,
    "technical": 0.10, "general": 0.05,
}

# Category → Threshold override delta (from encoding_gate.py)
CATEGORY_THRESHOLD_OVERRIDE = {
    "correction": -0.06, "decision": -0.04, "relationship": -0.04,
}


def evaluate_gate(score_entry: dict, w_n: float, w_s: float, w_pe: float,
                  threshold: float, salience_floor: float = SALIENCE_FLOOR,
                  salience_boost: float = 0.0, threshold_delta: float = 0.0) -> tuple[bool, float]:
    """Evaluate a single message through the gate with given weights."""
    novelty = score_entry["novelty"]
    salience = score_entry["salience"] + salience_boost
    salience = max(0.0, min(1.0, salience))
    pe = score_entry["pe"]

    total_w = w_n + w_s + w_pe
    if total_w <= 0:
        total_w = 1.0
    raw = novelty * w_n + salience * w_s + pe * w_pe
    gate_score = max(0.0, min(1.0, raw / total_w))

    if salience < salience_floor:
        return False, gate_score

    effective_threshold = max(0.10, threshold + threshold_delta)
    return gate_score >= effective_threshold, gate_score


def compute_auc(scores: list[dict], classifier_key: str, config: dict,
                salience_floor: float = SALIENCE_FLOOR,
                use_threshold_override: bool = False,
                use_salience_boost: bool = False) -> float:
    """Compute AUC by sweeping thresholds from 0 to 1."""
    signal_scores = []
    noise_scores = []

    for s in scores:
        cat = s["category"]
        if cat.startswith("B"):
            continue

        ext_cat = s[classifier_key]
        profile = _get_profile(config, ext_cat)

        sal_boost = CATEGORY_SALIENCE_BOOST.get(ext_cat, 0.05) if use_salience_boost else 0.0
        thr_delta = CATEGORY_THRESHOLD_OVERRIDE.get(ext_cat, 0.0) if use_threshold_override else 0.0

        _, gate_score = evaluate_gate(s, profile["w_n"], profile["w_s"], profile["w_pe"],
                                       profile["threshold"], salience_floor,
                                       sal_boost, thr_delta)

        if cat.startswith("S"):
            signal_scores.append(gate_score)
        elif cat.startswith("N"):
            noise_scores.append(gate_score)

    if not signal_scores or not noise_scores:
        return 0.5

    # Wilcoxon-Mann-Whitney AUC estimate
    auc = 0.0
    for ss in signal_scores:
        for ns in noise_scores:
            if ss > ns:
                auc += 1.0
            elif ss == ns:
                auc += 0.5
    return auc / (len(signal_scores) * len(noise_scores))


def _get_profile(config: dict, ext_category: str) -> dict:
    """Look up the profile for a given extractor category."""
    for profile_name, profile in config["profiles"].items():
        if ext_category in profile["categories"]:
            return profile
    return config["profiles"].get("default", list(config["profiles"].values())[0])


def evaluate_config(scores: list[dict], classifier_key: str, config: dict,
                    salience_floor: float = SALIENCE_FLOOR,
                    use_threshold_override: bool = False,
                    use_salience_boost: bool = False) -> dict:
    """Evaluate a full config on all scored messages."""
    total_s = 0
    encoded_s = 0
    total_n = 0
    encoded_n = 0
    s4_total = 0
    s4_encoded = 0
    per_subcat = defaultdict(lambda: {"total": 0, "encoded": 0})
    decisions = {}

    for s in scores:
        cat = s["category"]
        if cat.startswith("B"):
            continue

        ext_cat = s[classifier_key]
        profile = _get_profile(config, ext_cat)

        sal_boost = CATEGORY_SALIENCE_BOOST.get(ext_cat, 0.05) if use_salience_boost else 0.0
        thr_delta = CATEGORY_THRESHOLD_OVERRIDE.get(ext_cat, 0.0) if use_threshold_override else 0.0

        should_encode, gate_score = evaluate_gate(
            s, profile["w_n"], profile["w_s"], profile["w_pe"],
            profile["threshold"], salience_floor, sal_boost, thr_delta)

        decisions[s["id"]] = {"encode": should_encode, "score": gate_score, "ext_cat": ext_cat}

        per_subcat[cat]["total"] += 1
        if should_encode:
            per_subcat[cat]["encoded"] += 1

        if cat.startswith("S"):
            total_s += 1
            if should_encode:
                encoded_s += 1
            if cat == "S4":
                s4_total += 1
                if should_encode:
                    s4_encoded += 1
        elif cat.startswith("N"):
            total_n += 1
            if should_encode:
                encoded_n += 1

    s_encode = encoded_s / max(1, total_s)
    n_fp = encoded_n / max(1, total_n)
    s4_recall = s4_encoded / max(1, s4_total)

    sub_rates = {}
    for sc, vals in sorted(per_subcat.items()):
        sub_rates[sc] = round(vals["encoded"] / max(1, vals["total"]), 3)

    return {
        "s_encode": round(s_encode, 4),
        "n_fp": round(n_fp, 4),
        "s4_recall": round(s4_recall, 4),
        "s_total": total_s,
        "s_encoded": encoded_s,
        "n_total": total_n,
        "n_encoded": encoded_n,
        "s4_total": s4_total,
        "s4_encoded": s4_encoded,
        "per_subcat": sub_rates,
        "decisions": decisions,
    }


def sweep_profile(scores, classifier_key, base_config, profile_name, profile):
    """Sweep a single profile's parameters and return best variant."""
    if profile_name == "default" or profile_name == "general":
        return None

    dominant_key = max(["w_n", "w_s", "w_pe"], key=lambda k: profile[k])
    base_val = profile[dominant_key]
    base_thr = profile["threshold"]

    weight_vals = [max(0.05, base_val + d) for d in [-0.10, -0.05, 0.0, 0.05, 0.10]]
    weight_vals = sorted(set(min(0.60, v) for v in weight_vals))
    thr_vals = [max(0.10, base_thr + d) for d in [-0.04, -0.02, 0.0, 0.02, 0.04]]
    thr_vals = sorted(set(v for v in thr_vals))
    floor_vals = [0.05, 0.08, 0.10]

    best = None
    best_metric = -1.0

    for wv in weight_vals:
        for tv in thr_vals:
            for fv in floor_vals:
                trial_config = deepcopy(base_config)
                trial_profile = trial_config["profiles"][profile_name]
                trial_profile[dominant_key] = wv
                # Re-normalize other weights
                remaining = 1.0 - wv
                other_keys = [k for k in ["w_n", "w_s", "w_pe"] if k != dominant_key]
                other_sum = sum(profile[k] for k in other_keys)
                if other_sum > 0:
                    for k in other_keys:
                        trial_profile[k] = remaining * (profile[k] / other_sum)
                else:
                    for k in other_keys:
                        trial_profile[k] = remaining / len(other_keys)
                trial_profile["threshold"] = tv

                result = evaluate_config(scores, classifier_key, trial_config, fv)
                metric = result["s_encode"] - result["n_fp"]
                if result["s4_recall"] < 0.80:
                    metric -= 0.5  # Penalty for low S4 recall

                if metric > best_metric:
                    best_metric = metric
                    best = {
                        "profile_name": profile_name,
                        "dominant_key": dominant_key,
                        "dominant_val": round(wv, 3),
                        "threshold": round(tv, 3),
                        "salience_floor": round(fv, 3),
                        "w_n": round(trial_profile["w_n"], 3),
                        "w_s": round(trial_profile["w_s"], 3),
                        "w_pe": round(trial_profile["w_pe"], 3),
                        "s_encode": result["s_encode"],
                        "n_fp": result["n_fp"],
                        "s4_recall": result["s4_recall"],
                        "metric": round(best_metric, 4),
                    }

    return best


def phase3_sweep(scores: list[dict]):
    print("\n" + "=" * 70)
    print("PHASE 3: Profile count sweep")
    print("=" * 70)

    baseline_result_a = evaluate_config(scores, "extractor_category_a", PROFILE_CONFIGS["config_1"])
    baseline_result_b = evaluate_config(scores, "extractor_category_b", PROFILE_CONFIGS["config_1"])

    print(f"\n  Baseline (config_1):")
    print(f"    Classifier A: S*={baseline_result_a['s_encode']:.3f}, N*FP={baseline_result_a['n_fp']:.3f}, "
          f"S4={baseline_result_a['s4_recall']:.3f}")
    print(f"    Classifier B: S*={baseline_result_b['s_encode']:.3f}, N*FP={baseline_result_b['n_fp']:.3f}, "
          f"S4={baseline_result_b['s4_recall']:.3f}")

    all_results = {}

    for config_key, config in PROFILE_CONFIGS.items():
        print(f"\n  --- {config['name']} ---")

        for clf_key, clf_name in [("extractor_category_a", "A"), ("extractor_category_b", "B")]:
            # Hand-picked evaluation
            result = evaluate_config(scores, clf_key, config)
            auc = compute_auc(scores, clf_key, config)

            print(f"    [{clf_name}] S*={result['s_encode']:.3f}, N*FP={result['n_fp']:.3f}, "
                  f"S4={result['s4_recall']:.3f}, AUC={auc:.3f}")

            # Count decision flips vs baseline
            baseline_decisions = baseline_result_a["decisions"] if clf_name == "A" else baseline_result_b["decisions"]
            flips = 0
            for msg_id, dec in result["decisions"].items():
                if msg_id in baseline_decisions:
                    if dec["encode"] != baseline_decisions[msg_id]["encode"]:
                        flips += 1

            # Sweep non-default profiles
            best_swept = {}
            for profile_name, profile in config["profiles"].items():
                swept = sweep_profile(scores, clf_key, config, profile_name, profile)
                if swept:
                    best_swept[profile_name] = swept

            # Apply best swept params and re-evaluate
            if best_swept:
                swept_config = deepcopy(config)
                for pn, sv in best_swept.items():
                    swept_config["profiles"][pn]["w_n"] = sv["w_n"]
                    swept_config["profiles"][pn]["w_s"] = sv["w_s"]
                    swept_config["profiles"][pn]["w_pe"] = sv["w_pe"]
                    swept_config["profiles"][pn]["threshold"] = sv["threshold"]

                swept_result = evaluate_config(scores, clf_key, swept_config)
                swept_auc = compute_auc(scores, clf_key, swept_config)
                print(f"    [{clf_name}] SWEPT: S*={swept_result['s_encode']:.3f}, "
                      f"N*FP={swept_result['n_fp']:.3f}, S4={swept_result['s4_recall']:.3f}, "
                      f"AUC={swept_auc:.3f}")
            else:
                swept_result = result
                swept_auc = auc
                swept_config = config

            all_results[f"{config_key}_{clf_name}"] = {
                "config_key": config_key,
                "config_name": config["name"],
                "classifier": clf_name,
                "hand_picked": {
                    "s_encode": result["s_encode"],
                    "n_fp": result["n_fp"],
                    "s4_recall": result["s4_recall"],
                    "auc": round(auc, 4),
                    "per_subcat": result["per_subcat"],
                    "flips": flips,
                },
                "swept": {
                    "s_encode": swept_result["s_encode"],
                    "n_fp": swept_result["n_fp"],
                    "s4_recall": swept_result["s4_recall"],
                    "auc": round(swept_auc, 4),
                    "per_subcat": swept_result["per_subcat"],
                    "best_params": best_swept,
                },
                "decisions": {k: {"encode": v["encode"], "score": round(v["score"], 4)}
                              for k, v in swept_result["decisions"].items()},
            }

    with open(SWEEP_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved sweep results to {SWEEP_PATH}")

    return all_results


# ============================================================================
# PHASE 4: Noise leak audit
# ============================================================================

def phase4_audit(scores: list[dict], sweep_results: dict):
    print("\n" + "=" * 70)
    print("PHASE 4: Noise leak audit")
    print("=" * 70)

    baseline_a = sweep_results["config_1_A"]["decisions"]
    baseline_b = sweep_results["config_1_B"]["decisions"]
    baseline_n_fp_a = sweep_results["config_1_A"]["swept"]["n_fp"]
    baseline_n_fp_b = sweep_results["config_1_B"]["swept"]["n_fp"]

    audit = {}

    for key, result in sweep_results.items():
        if key.startswith("config_1"):
            continue

        clf = result["classifier"]
        baseline = baseline_a if clf == "A" else baseline_b
        baseline_nfp = baseline_n_fp_a if clf == "A" else baseline_n_fp_b

        leaks = []
        for s in scores:
            if not s["category"].startswith("N"):
                continue
            msg_id = s["id"]
            if msg_id not in result["decisions"] or msg_id not in baseline:
                continue

            new_enc = result["decisions"][msg_id]["encode"]
            old_enc = baseline[msg_id]["encode"]

            if new_enc and not old_enc:
                leaks.append({
                    "id": msg_id,
                    "content": s["content"][:60],
                    "category": s["category"],
                    "ext_cat": s[f"extractor_category_{'a' if clf == 'A' else 'b'}"],
                    "new_score": result["decisions"][msg_id]["score"],
                    "old_score": baseline[msg_id]["score"],
                })

        n_fp = result["swept"]["n_fp"]
        disqualified = (n_fp - baseline_nfp) > 0.02

        audit[key] = {
            "config_name": result["config_name"],
            "classifier": clf,
            "total_leaks": len(leaks),
            "leaks": leaks[:30],
            "n_fp": n_fp,
            "baseline_n_fp": baseline_nfp,
            "delta_n_fp": round(n_fp - baseline_nfp, 4),
            "disqualified": disqualified,
        }

        status = "DISQUALIFIED" if disqualified else "OK"
        print(f"  {key}: {len(leaks)} leaks, N*FP delta={n_fp - baseline_nfp:+.4f} [{status}]")

    with open(AUDIT_PATH, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n  Saved audit to {AUDIT_PATH}")

    return audit


# ============================================================================
# PHASE 5: Comparison to existing mechanisms
# ============================================================================

def phase5_compare(scores: list[dict], sweep_results: dict):
    print("\n" + "=" * 70)
    print("PHASE 5: Comparison to existing mechanisms")
    print("=" * 70)

    # Find best non-baseline config from classifier B
    best_key = None
    best_metric = -1.0
    for key, result in sweep_results.items():
        if result["classifier"] != "B":
            continue
        if key == "config_1_B":
            continue
        metric = result["swept"]["s_encode"] - result["swept"]["n_fp"]
        if result["swept"]["s4_recall"] >= 0.80 and metric > best_metric:
            best_metric = metric
            best_key = key

    if not best_key:
        best_key = "config_2_B"

    best_config_key = sweep_results[best_key]["config_key"]
    best_config = deepcopy(PROFILE_CONFIGS[best_config_key])
    # Apply swept params
    if sweep_results[best_key]["swept"]["best_params"]:
        for pn, sv in sweep_results[best_key]["swept"]["best_params"].items():
            best_config["profiles"][pn]["w_n"] = sv["w_n"]
            best_config["profiles"][pn]["w_s"] = sv["w_s"]
            best_config["profiles"][pn]["w_pe"] = sv["w_pe"]
            best_config["profiles"][pn]["threshold"] = sv["threshold"]

    clf_key = "extractor_category_b"
    baseline_config = PROFILE_CONFIGS["config_1"]

    setups = {}

    # Setup A: Baseline only
    result_a = evaluate_config(scores, clf_key, baseline_config)
    auc_a = compute_auc(scores, clf_key, baseline_config)
    setups["A_baseline"] = {
        "label": "Baseline only",
        "s_encode": result_a["s_encode"],
        "n_fp": result_a["n_fp"],
        "s4_recall": result_a["s4_recall"],
        "auc": round(auc_a, 4),
    }

    # Setup B: Existing mechanisms (threshold override + salience boost)
    result_b = evaluate_config(scores, clf_key, baseline_config,
                               use_threshold_override=True, use_salience_boost=True)
    auc_b = compute_auc(scores, clf_key, baseline_config,
                        use_threshold_override=True, use_salience_boost=True)
    setups["B_existing"] = {
        "label": "Existing mechanisms",
        "s_encode": result_b["s_encode"],
        "n_fp": result_b["n_fp"],
        "s4_recall": result_b["s4_recall"],
        "auc": round(auc_b, 4),
    }

    # Setup C: Dynamic weights only
    result_c = evaluate_config(scores, clf_key, best_config)
    auc_c = compute_auc(scores, clf_key, best_config)
    setups["C_dynamic"] = {
        "label": f"Dynamic weights only ({best_config_key})",
        "s_encode": result_c["s_encode"],
        "n_fp": result_c["n_fp"],
        "s4_recall": result_c["s4_recall"],
        "auc": round(auc_c, 4),
        "config_used": best_config_key,
    }

    # Setup D: All three
    result_d = evaluate_config(scores, clf_key, best_config,
                               use_threshold_override=True, use_salience_boost=True)
    auc_d = compute_auc(scores, clf_key, best_config,
                        use_threshold_override=True, use_salience_boost=True)
    setups["D_all_three"] = {
        "label": "All three combined",
        "s_encode": result_d["s_encode"],
        "n_fp": result_d["n_fp"],
        "s4_recall": result_d["s4_recall"],
        "auc": round(auc_d, 4),
    }

    print()
    print(f"  {'Setup':<30} {'AUC':>6} {'S*':>7} {'N*FP':>7} {'S4':>6}")
    print("  " + "-" * 60)
    for key, s in setups.items():
        print(f"  {s['label']:<30} {s['auc']:>6.3f} {s['s_encode']:>7.3f} "
              f"{s['n_fp']:>7.3f} {s['s4_recall']:>6.3f}")

    with open(COMPARE_PATH, "w") as f:
        json.dump({"best_dynamic_config": best_config_key, "setups": setups}, f, indent=2)
    print(f"\n  Saved comparison to {COMPARE_PATH}")

    return setups


# ============================================================================
# PHASE 6: Report
# ============================================================================

def phase6_report(scores, sweep_results, audit, comparison):
    print("\n" + "=" * 70)
    print("PHASE 6: Writing report")
    print("=" * 70)

    # Rebuild cross-tabulation from scores
    cross_tab_a = defaultdict(Counter)
    cross_tab_b = defaultdict(Counter)
    for s in scores:
        cat = s["category"]
        label = cat[:2] if len(cat) >= 2 else cat
        cross_tab_a[label][s["extractor_category_a"]] += 1
        cross_tab_b[label][s["extractor_category_b"]] += 1

    all_ext_cats = sorted(set(
        c for ct in [cross_tab_a, cross_tab_b] for row in ct.values() for c in row.keys()
    ))
    bench_cats = ["S1", "S2", "S3", "S4", "S5", "N1", "N2", "N3", "N4", "N5", "B1", "B2", "B3"]

    def _xtab(cross_tab):
        lines = []
        header = f"| {'Cat':>4} | " + " | ".join(f"{c:>10}" for c in all_ext_cats) + " | Total |"
        sep = "|" + "-" * (len(header) - 2) + "|"
        lines.append(header)
        lines.append(sep)
        for bc in bench_cats:
            row = cross_tab[bc]
            total = sum(row.values())
            cells = " | ".join(f"{row.get(c, 0):>10}" for c in all_ext_cats)
            lines.append(f"| {bc:>4} | {cells} | {total:>5} |")
        return "\n".join(lines)

    # Results table
    config_keys = ["config_1", "config_2", "config_3", "config_4", "config_5"]
    results_table_lines = []
    results_table_lines.append("| Config | Clf | AUC (hp) | S* (hp) | N*FP (hp) | S4 (hp) | AUC (sw) | S* (sw) | N*FP (sw) | S4 (sw) |")
    results_table_lines.append("|--------|-----|----------|---------|-----------|---------|----------|---------|-----------|---------|")
    for ck in config_keys:
        for clf in ["A", "B"]:
            key = f"{ck}_{clf}"
            if key not in sweep_results:
                continue
            r = sweep_results[key]
            hp = r["hand_picked"]
            sw = r["swept"]
            results_table_lines.append(
                f"| {r['config_name'][:20]:20} | {clf} | {hp['auc']:.3f} | {hp['s_encode']:.3f} | "
                f"{hp['n_fp']:.3f} | {hp['s4_recall']:.3f} | {sw['auc']:.3f} | {sw['s_encode']:.3f} | "
                f"{sw['n_fp']:.3f} | {sw['s4_recall']:.3f} |"
            )

    # Noise leak summary
    noise_lines = []
    for key, a in sorted(audit.items()):
        status = "**DISQUALIFIED**" if a["disqualified"] else "OK"
        noise_lines.append(f"- **{a['config_name']} [{a['classifier']}]**: {a['total_leaks']} leaks, "
                           f"N*FP delta={a['delta_n_fp']:+.4f} — {status}")
        if a["leaks"]:
            for leak in a["leaks"][:5]:
                noise_lines.append(f"  - `{leak['content']}` ({leak['category']}/{leak['ext_cat']}) "
                                   f"old={leak['old_score']:.3f} new={leak['new_score']:.3f}")

    # Classifier sensitivity
    sensitivity_lines = []
    for ck in config_keys:
        key_a = f"{ck}_A"
        key_b = f"{ck}_B"
        if key_a in sweep_results and key_b in sweep_results:
            auc_a = sweep_results[key_a]["swept"]["auc"]
            auc_b = sweep_results[key_b]["swept"]["auc"]
            gap = abs(auc_a - auc_b)
            fragile = "FRAGILE" if gap > 0.03 else "stable"
            name = sweep_results[key_a]["config_name"]
            sensitivity_lines.append(f"- **{name}**: AUC(A)={auc_a:.3f}, AUC(B)={auc_b:.3f}, "
                                     f"gap={gap:.3f} — {fragile}")

    # Marginal improvement curve data
    curve_lines = []
    curve_lines.append("| Profiles | S* (A) | S* (B) | N*FP (A) | N*FP (B) |")
    curve_lines.append("|----------|--------|--------|----------|----------|")
    for ck in config_keys:
        key_a = f"{ck}_A"
        key_b = f"{ck}_B"
        if key_a in sweep_results and key_b in sweep_results:
            ra = sweep_results[key_a]["swept"]
            rb = sweep_results[key_b]["swept"]
            n_profiles = len(PROFILE_CONFIGS[ck]["profiles"])
            curve_lines.append(f"| {n_profiles} | {ra['s_encode']:.3f} | {rb['s_encode']:.3f} | "
                               f"{ra['n_fp']:.3f} | {rb['n_fp']:.3f} |")

    # Top 20 decision changes
    baseline_decisions_b = sweep_results["config_1_B"]["decisions"]
    best_dynamic_key = None
    best_dynamic_metric = -1.0
    for key, r in sweep_results.items():
        if r["classifier"] != "B" or key == "config_1_B":
            continue
        metric = r["swept"]["s_encode"] - r["swept"]["n_fp"]
        if metric > best_dynamic_metric:
            best_dynamic_metric = metric
            best_dynamic_key = key

    changes = []
    if best_dynamic_key:
        dynamic_decisions = sweep_results[best_dynamic_key]["decisions"]
        for s in scores:
            if s["category"].startswith("B"):
                continue
            msg_id = s["id"]
            if msg_id in baseline_decisions_b and msg_id in dynamic_decisions:
                old_score = baseline_decisions_b[msg_id]["score"]
                new_score = dynamic_decisions[msg_id]["score"]
                delta = new_score - old_score
                changes.append({
                    "id": msg_id,
                    "content": s["content"][:60],
                    "category": s["category"],
                    "ext_cat_b": s["extractor_category_b"],
                    "old_score": old_score,
                    "new_score": new_score,
                    "delta": delta,
                    "old_encode": baseline_decisions_b[msg_id]["encode"],
                    "new_encode": dynamic_decisions[msg_id]["encode"],
                })

    changes.sort(key=lambda x: x["delta"], reverse=True)
    top_helped = changes[:10]
    top_hurt = changes[-10:]

    change_lines = []
    change_lines.append("### Messages helped most (biggest positive score delta)")
    change_lines.append("| Content | Cat | ExtCat | Old | New | Delta |")
    change_lines.append("|---------|-----|--------|-----|-----|-------|")
    for c in top_helped:
        change_lines.append(f"| `{c['content'][:40]}` | {c['category']} | {c['ext_cat_b']} | "
                            f"{c['old_score']:.3f} | {c['new_score']:.3f} | {c['delta']:+.3f} |")

    change_lines.append("")
    change_lines.append("### Messages hurt most (biggest negative score delta or noise leak)")
    change_lines.append("| Content | Cat | ExtCat | Old | New | Delta |")
    change_lines.append("|---------|-----|--------|-----|-----|-------|")
    for c in top_hurt:
        change_lines.append(f"| `{c['content'][:40]}` | {c['category']} | {c['ext_cat_b']} | "
                            f"{c['old_score']:.3f} | {c['new_score']:.3f} | {c['delta']:+.3f} |")

    # Comparison table
    comp_lines = []
    comp_lines.append("| Setup | AUC | S* Encode | N* FP | S4 Recall |")
    comp_lines.append("|-------|-----|-----------|-------|-----------|")
    for key, s in comparison.items():
        comp_lines.append(f"| {s['label']:<30} | {s['auc']:.3f} | {s['s_encode']:.3f} | "
                          f"{s['n_fp']:.3f} | {s['s4_recall']:.3f} |")

    # Recommendation
    baseline_s_enc_b = sweep_results["config_1_B"]["swept"]["s_encode"]
    baseline_n_fp_b = sweep_results["config_1_B"]["swept"]["n_fp"]
    existing_s_enc = comparison.get("B_existing", {}).get("s_encode", baseline_s_enc_b)
    existing_n_fp = comparison.get("B_existing", {}).get("n_fp", baseline_n_fp_b)

    best_s_enc = -1.0
    best_config_name = "none"
    best_n_fp = 0.0
    best_s4 = 0.0
    for key, r in sweep_results.items():
        if r["classifier"] != "B" or key == "config_1_B":
            continue
        if r["swept"]["s4_recall"] >= 0.80 and r["swept"]["s_encode"] > best_s_enc:
            best_s_enc = r["swept"]["s_encode"]
            best_config_name = r["config_name"]
            best_n_fp = r["swept"]["n_fp"]
            best_s4 = r["swept"]["s4_recall"]

    s_enc_gain = best_s_enc - baseline_s_enc_b
    n_fp_delta = best_n_fp - baseline_n_fp_b
    beats_existing = best_s_enc > existing_s_enc

    ship_criteria = [
        ("S* encode gain >= 3%", s_enc_gain >= 0.03),
        ("N* FP delta <= 2%", n_fp_delta <= 0.02),
        ("S4 recall >= 80%", best_s4 >= 0.80),
        ("Beats existing mechanisms", beats_existing),
    ]

    recommendation_lines = []
    all_pass = all(c[1] for c in ship_criteria)
    for desc, passed in ship_criteria:
        marker = "PASS" if passed else "FAIL"
        recommendation_lines.append(f"- [{marker}] {desc}")

    if all_pass:
        recommendation_lines.append(f"\n**Recommendation:** Ship {best_config_name} with swept parameters.")
    else:
        recommendation_lines.append(f"\n**Recommendation:** Do NOT ship dynamic weighting. "
                                     f"The existing simpler mechanisms are sufficient.")

    report = f"""# Dynamic Weighting Report

## 1. Category Annotation Distribution

### Classifier A (content-only)

{_xtab(cross_tab_a)}

### Classifier B (notes-assisted)

{_xtab(cross_tab_b)}

## 2. Full Results Table

{chr(10).join(results_table_lines)}

## 3. Noise Leak Audit

{chr(10).join(noise_lines)}

## 4. Classifier Sensitivity

{chr(10).join(sensitivity_lines)}

## 5. Marginal Improvement Curve

{chr(10).join(curve_lines)}

## 6. The 20 Biggest Decision Changes

{chr(10).join(change_lines)}

## 7. Comparison to Existing Mechanisms

{chr(10).join(comp_lines)}

**Key finding:** Existing mechanisms (threshold override + salience boost) achieve
S*={existing_s_enc:.3f} / N*FP={existing_n_fp:.3f}. Best dynamic config achieves
S*={best_s_enc:.3f} / N*FP={best_n_fp:.3f}.

## 8. Recommendation

Best dynamic config: **{best_config_name}** (S* gain: {s_enc_gain:+.3f}, N*FP delta: {n_fp_delta:+.3f})

{chr(10).join(recommendation_lines)}

## 9. Honest Assessment

Dynamic weighting adds a dict lookup per message — computationally cheap. But it introduces
{sum(len(p) for c in PROFILE_CONFIGS.values() for p in c['profiles'].values() if not isinstance(p, str))}
parameters across weight profiles that need re-tuning whenever a signal scorer changes.

The question: is the gain ({s_enc_gain:+.1%} S* encode) worth the ongoing maintenance cost?
The existing per-category threshold overrides and salience boosts achieve most of the category-specific
benefit with 3 parameters (threshold) + 8 parameters (boost) = 11 parameters, and they're intuitive.
Dynamic weight profiles add ~21 more parameters for a marginal improvement.
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"  Report written to {REPORT_PATH}")

    return report


# ============================================================================
# PHASE 7: Rustle-the-feathers
# ============================================================================

def phase7_rustle(scores, sweep_results, comparison):
    print("\n" + "=" * 70)
    print("PHASE 7: Rustle-the-feathers")
    print("=" * 70)

    rustle_sections = []

    # --- 1. Overfitting Skeptic ---
    print("  1. Overfitting skeptic (conversation split)...")

    conv_ids = sorted(set(s["conversation_id"] for s in scores))
    split_a_convs = set(conv_ids[:2])
    split_b_convs = set(conv_ids[2:])

    split_a_scores = [s for s in scores if s["conversation_id"] in split_a_convs]
    split_b_scores = [s for s in scores if s["conversation_id"] in split_b_convs]

    config_keys = ["config_1", "config_2", "config_3", "config_4", "config_5"]
    rankings_a = []
    rankings_b = []

    for ck in config_keys:
        config = PROFILE_CONFIGS[ck]
        ra = evaluate_config(split_a_scores, "extractor_category_b", config)
        rb = evaluate_config(split_b_scores, "extractor_category_b", config)
        metric_a = ra["s_encode"] - ra["n_fp"]
        metric_b = rb["s_encode"] - rb["n_fp"]
        rankings_a.append((ck, metric_a))
        rankings_b.append((ck, metric_b))

    rankings_a.sort(key=lambda x: x[1], reverse=True)
    rankings_b.sort(key=lambda x: x[1], reverse=True)

    rank_a = {ck: i for i, (ck, _) in enumerate(rankings_a)}
    rank_b = {ck: i for i, (ck, _) in enumerate(rankings_b)}

    n = len(config_keys)
    d_sq_sum = sum((rank_a[ck] - rank_b[ck]) ** 2 for ck in config_keys)
    spearman = 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1))

    overfitting_lines = []
    overfitting_lines.append(f"Split A (conv 1-2): {', '.join(f'{ck}={m:.3f}' for ck, m in rankings_a)}")
    overfitting_lines.append(f"Split B (conv 3-5): {', '.join(f'{ck}={m:.3f}' for ck, m in rankings_b)}")
    overfitting_lines.append(f"Spearman rank correlation: {spearman:.3f}")
    overfitting_lines.append(f"{'PASS' if spearman >= 0.7 else 'FAIL'}: threshold is 0.7")

    rustle_sections.append(("The Overfitting Skeptic", "\n".join(overfitting_lines)))
    print(f"    Spearman r={spearman:.3f}")

    # --- 2. Mis-categorization Skeptic ---
    print("  2. Mis-categorization skeptic (confusion matrix)...")

    confusion_matrix = {
        ("correction", "personal"): 0.30,
        ("personal", "correction"): 0.30,
        ("correction", "general"): 0.20,
        ("general", "correction"): 0.20,
        ("decision", "personal"): 0.25,
        ("personal", "decision"): 0.25,
        ("temporal", "general"): 0.20,
        ("general", "temporal"): 0.20,
        ("relationship", "personal"): 0.15,
        ("personal", "relationship"): 0.15,
    }
    default_confusion = 0.05

    # Find best dynamic config
    best_key = None
    best_metric = -1.0
    for key, r in sweep_results.items():
        if r["classifier"] != "B" or key == "config_1_B":
            continue
        m = r["swept"]["s_encode"] - r["swept"]["n_fp"]
        if r["swept"]["s4_recall"] >= 0.80 and m > best_metric:
            best_metric = m
            best_key = key

    if not best_key:
        best_key = "config_2_B"

    best_ck = sweep_results[best_key]["config_key"]
    best_config = deepcopy(PROFILE_CONFIGS[best_ck])
    if sweep_results[best_key]["swept"]["best_params"]:
        for pn, sv in sweep_results[best_key]["swept"]["best_params"].items():
            best_config["profiles"][pn]["w_n"] = sv["w_n"]
            best_config["profiles"][pn]["w_s"] = sv["w_s"]
            best_config["profiles"][pn]["w_pe"] = sv["w_pe"]
            best_config["profiles"][pn]["threshold"] = sv["threshold"]

    all_cats = ["correction", "temporal", "personal", "preference", "decision",
                "relationship", "technical", "general"]

    trial_aucs = []
    trial_s_encs = []
    trial_n_fps = []

    for seed in range(10):
        rng = random.Random(seed)
        confused_scores = deepcopy(scores)
        for s in confused_scores:
            orig_cat = s["extractor_category_b"]
            if rng.random() < 0.5:
                candidates = [(to_cat, confusion_matrix.get((orig_cat, to_cat), default_confusion))
                              for to_cat in all_cats if to_cat != orig_cat]
                total_prob = sum(p for _, p in candidates)
                if total_prob > 0:
                    r = rng.random() * total_prob
                    cum = 0.0
                    for to_cat, p in candidates:
                        cum += p
                        if r <= cum:
                            s["extractor_category_b"] = to_cat
                            break

        result = evaluate_config(confused_scores, "extractor_category_b", best_config)
        auc = compute_auc(confused_scores, "extractor_category_b", best_config)
        trial_aucs.append(auc)
        trial_s_encs.append(result["s_encode"])
        trial_n_fps.append(result["n_fp"])

    mean_auc = sum(trial_aucs) / len(trial_aucs)
    std_auc = (sum((a - mean_auc) ** 2 for a in trial_aucs) / len(trial_aucs)) ** 0.5
    mean_s = sum(trial_s_encs) / len(trial_s_encs)
    mean_n = sum(trial_n_fps) / len(trial_n_fps)

    miscat_lines = []
    miscat_lines.append(f"Config tested: {best_ck}")
    miscat_lines.append(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    miscat_lines.append(f"Mean S* encode: {mean_s:.4f}")
    miscat_lines.append(f"Mean N* FP: {mean_n:.4f}")
    miscat_lines.append(f"AUC std: {std_auc:.4f}")
    miscat_lines.append(f"{'PASS' if std_auc < 0.02 else 'FAIL'}: threshold is std < 0.02")

    rustle_sections.append(("The Mis-categorization Skeptic", "\n".join(miscat_lines)))
    print(f"    AUC std={std_auc:.4f}")

    # --- 3. Simplicity Advocate ---
    print("  3. Simplicity advocate...")

    # Alt 1: Just add temporal to threshold override
    alt1_config = deepcopy(PROFILE_CONFIGS["config_1"])
    alt1_override = dict(CATEGORY_THRESHOLD_OVERRIDE)
    alt1_override["temporal"] = -0.06
    result_alt1 = evaluate_config(scores, "extractor_category_b", alt1_config,
                                  use_threshold_override=True)
    # Override with custom temporal
    # Need to recompute with the extra temporal override
    alt1_decisions = {}
    alt1_s = 0
    alt1_s_enc = 0
    alt1_n = 0
    alt1_n_enc = 0
    for s in scores:
        cat = s["category"]
        if cat.startswith("B"):
            continue
        ext_cat = s["extractor_category_b"]
        profile = alt1_config["profiles"]["default"]
        thr_delta = alt1_override.get(ext_cat, 0.0)
        should_encode, gate_score = evaluate_gate(s, profile["w_n"], profile["w_s"], profile["w_pe"],
                                                   profile["threshold"], SALIENCE_FLOOR, 0.0, thr_delta)
        if cat.startswith("S"):
            alt1_s += 1
            if should_encode:
                alt1_s_enc += 1
        elif cat.startswith("N"):
            alt1_n += 1
            if should_encode:
                alt1_n_enc += 1
    alt1_s_rate = alt1_s_enc / max(1, alt1_s)
    alt1_n_rate = alt1_n_enc / max(1, alt1_n)

    # Alt 2: Just 2 profiles
    result_alt2 = evaluate_config(scores, "extractor_category_b", PROFILE_CONFIGS["config_2"])
    auc_alt2 = compute_auc(scores, "extractor_category_b", PROFILE_CONFIGS["config_2"])

    # Alt 3: Existing mechanisms
    result_alt3 = evaluate_config(scores, "extractor_category_b", PROFILE_CONFIGS["config_1"],
                                  use_threshold_override=True, use_salience_boost=True)

    baseline_s = sweep_results["config_1_B"]["swept"]["s_encode"]
    best_dynamic_s = sweep_results[best_key]["swept"]["s_encode"]
    gain_dynamic = best_dynamic_s - baseline_s

    simplicity_lines = []
    simplicity_lines.append(f"| Alternative | S* | N*FP | LOC changed | AUC gain/LOC |")
    simplicity_lines.append(f"|------------|------|------|-------------|--------------|")
    simplicity_lines.append(f"| +temporal threshold | {alt1_s_rate:.3f} | {alt1_n_rate:.3f} | 1 | "
                            f"{(alt1_s_rate - baseline_s):.4f} |")
    simplicity_lines.append(f"| 2 profiles | {result_alt2['s_encode']:.3f} | {result_alt2['n_fp']:.3f} | ~20 | "
                            f"{(result_alt2['s_encode'] - baseline_s) / 20:.4f} |")
    simplicity_lines.append(f"| Existing mechanisms | {result_alt3['s_encode']:.3f} | {result_alt3['n_fp']:.3f} | 0 (shipped) | N/A |")
    simplicity_lines.append(f"| Best dynamic ({best_ck}) | {best_dynamic_s:.3f} | "
                            f"{sweep_results[best_key]['swept']['n_fp']:.3f} | ~50 | "
                            f"{gain_dynamic / 50:.4f} |")

    one_line_gets_80pct = (alt1_s_rate - baseline_s) >= 0.80 * gain_dynamic if gain_dynamic > 0 else True
    simplicity_lines.append(f"\n1-line change gets {((alt1_s_rate - baseline_s) / max(0.001, gain_dynamic) * 100):.0f}% "
                            f"of dynamic's benefit. {'YES, 80%+ threshold met.' if one_line_gets_80pct else 'No, less than 80%.'}")

    rustle_sections.append(("The Simplicity Advocate", "\n".join(simplicity_lines)))
    print(f"    1-line change: S*={alt1_s_rate:.3f}")

    # --- 4. Production Skeptic ---
    print("  4. Production skeptic (extracted fact simulation)...")

    # Find the 20 biggest-impact messages from changes analysis
    baseline_decisions_b = sweep_results["config_1_B"]["decisions"]
    dynamic_decisions = sweep_results[best_key]["decisions"]

    impact_msgs = []
    for s in scores:
        if s["category"].startswith("B"):
            continue
        msg_id = s["id"]
        if msg_id in baseline_decisions_b and msg_id in dynamic_decisions:
            old_score = baseline_decisions_b[msg_id]["score"]
            new_score = dynamic_decisions[msg_id]["score"]
            if abs(new_score - old_score) > 0.01:
                impact_msgs.append({
                    "id": msg_id,
                    "content": s["content"],
                    "category": s["category"],
                    "ext_cat": s["extractor_category_b"],
                    "old_score": old_score,
                    "new_score": new_score,
                    "delta": abs(new_score - old_score),
                    "novelty": s["novelty"],
                    "salience": s["salience"],
                    "pe": s["pe"],
                })
    impact_msgs.sort(key=lambda x: x["delta"], reverse=True)
    top_20 = impact_msgs[:20]

    prod_lines = []
    prod_lines.append("Extracted facts typically have higher salience (reformulated as clear statements).")
    prod_lines.append("The extractor turns `'we're thinking June next year'` into `'Planning wedding for June next year'`.")
    prod_lines.append("")
    prod_lines.append("Simulating extraction reformulation on top 20 impact messages:")
    prod_lines.append("")

    baseline_handles = 0
    for msg in top_20:
        # Simulate extraction: extracted facts are typically longer, more
        # structured, and have higher salience scores. Estimate +0.15 salience.
        sim_salience = min(1.0, msg["salience"] + 0.15)
        sim_entry = {
            "novelty": msg["novelty"],
            "salience": sim_salience,
            "pe": msg["pe"],
        }
        baseline_profile = PROFILE_CONFIGS["config_1"]["profiles"]["default"]
        should_encode, score = evaluate_gate(sim_entry, baseline_profile["w_n"],
                                              baseline_profile["w_s"], baseline_profile["w_pe"],
                                              baseline_profile["threshold"])
        if should_encode:
            baseline_handles += 1
        prod_lines.append(f"  - `{msg['content'][:50]}` → salience {msg['salience']:.2f}→{sim_salience:.2f} "
                          f"→ baseline {'ENCODE' if should_encode else 'SKIP'}")

    prod_lines.append(f"\nBaseline handles {baseline_handles}/20 ({baseline_handles/20*100:.0f}%) "
                      f"after extraction reformulation.")
    prod_lines.append("If >80%, dynamic weighting is solving a problem that extraction already solves.")

    rustle_sections.append(("The Production Skeptic", "\n".join(prod_lines)))
    print(f"    Baseline handles {baseline_handles}/20 after simulated extraction")

    # Write rustle report
    rustle_md = "# Dynamic Weighting — Rustle-the-Feathers\n\n"
    for title, content in rustle_sections:
        rustle_md += f"## {title}\n\n{content}\n\n---\n\n"

    # Summary
    pass_count = 0
    total_checks = 4
    checks = [
        ("Overfitting (Spearman >= 0.7)", spearman >= 0.7),
        ("Mis-categorization (AUC std < 0.02)", std_auc < 0.02),
        ("Simplicity (1-line < 80% of benefit)", not one_line_gets_80pct),
        ("Production (baseline handles < 80%)", baseline_handles / 20 < 0.80),
    ]
    rustle_md += "## Summary\n\n"
    for desc, passed in checks:
        marker = "PASS" if passed else "FAIL"
        if passed:
            pass_count += 1
        rustle_md += f"- [{marker}] {desc}\n"
    rustle_md += f"\n**{pass_count}/{total_checks} checks passed.**\n"

    with open(RUSTLE_PATH, "w") as f:
        f.write(rustle_md)
    print(f"  Rustle report written to {RUSTLE_PATH}")

    return rustle_md


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()

    # Phase 1
    benchmark = phase1_annotate()

    # Phase 2
    scores = phase2_score(benchmark)

    # Phase 3
    sweep_results = phase3_sweep(scores)

    # Phase 4
    audit = phase4_audit(scores, sweep_results)

    # Phase 5
    comparison = phase5_compare(scores, sweep_results)

    # Phase 6
    phase6_report(scores, sweep_results, audit, comparison)

    # Phase 7
    phase7_rustle(scores, sweep_results, comparison)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE. Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
