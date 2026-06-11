#!/usr/bin/env python3
"""Gate Deep Research — Sweep Runner.

Evaluates gate candidates on message-level filtering decisions and
measures impact on retrieval precision. Unlike run_candidate.py which
operates at the full pipeline level (ingest → retrieve), this runner
operates at the gate level:

1. Load dataset messages
2. For each candidate × threshold: run gate on all messages
3. Filter to kept messages only
4. Ingest kept messages via TrueMemory engine
5. Run retrieval queries and measure p@k

This isolates the gate's effect from extraction/dedup confounds.

Usage:
    python benchmarks/gate_eval/run_gate_deep_sweep.py
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATASETS_DIR = REPO_ROOT / "benchmarks" / "gate_eval" / "datasets"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "gate_eval" / "results" / "gate_deep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_short_horizon():
    path = DATASETS_DIR / "short_horizon_200.json"
    return json.loads(path.read_text())


def load_long_horizon():
    path = DATASETS_DIR / "long_horizon_synthetic.json"
    return json.loads(path.read_text())


def extract_messages_short(dataset: dict) -> list[dict]:
    """Flatten LoCoMo conversations into a flat message list."""
    messages = []
    for conv in dataset["convs"]:
        cdata = conv["conversation"]
        for k in sorted(cdata.keys()):
            if k.startswith("session_") and not k.endswith("_date_time"):
                for turn in cdata[k]:
                    messages.append({
                        "content": turn.get("text", ""),
                        "sender": turn.get("speaker", "human"),
                        "recipient": "other",
                        "timestamp": f"conv{conv['conv_idx']}_{k}",
                        "category": "",
                        "modality": "conversation",
                        "conv_idx": conv["conv_idx"],
                    })
    return messages


def extract_messages_long(dataset: dict) -> list[dict]:
    """Flatten long_horizon sessions into a flat message list."""
    messages = []
    for sess in dataset["sessions"]:
        for i, turn in enumerate(sess["transcript"]):
            messages.append({
                "content": turn.get("content", ""),
                "sender": turn.get("role", "user"),
                "recipient": "other",
                "timestamp": f"s{sess['session_id']}_{i:04d}",
                "category": "",
                "modality": "conversation",
                "session_id": sess["session_id"],
            })
    return messages


def identify_chitchat(messages: list[dict]) -> set[int]:
    """Heuristic: identify obvious chitchat messages by index."""
    chitchat_idx = set()
    chitchat_patterns = {
        "lol", "lmao", "haha", "ok", "okay", "yeah", "yep", "sure",
        "thanks", "thank you", "hi", "hey", "hello", "bye", "wow",
        "omg", "nice", "cool", "got it", "sounds good", "np",
    }
    for i, m in enumerate(messages):
        text = (m.get("content") or "").strip().lower()
        if not text or len(text) < 4:
            chitchat_idx.add(i)
        elif text in chitchat_patterns:
            chitchat_idx.add(i)
    return chitchat_idx


def identify_utility_messages(messages: list[dict], qa_list: list[dict],
                                dataset_type: str) -> set[int]:
    """Identify messages that contain gold answers (utility-positive)."""
    utility_idx = set()
    clean = lambda s: re.sub(r"[^a-z0-9 ]+", "", s.lower())

    if dataset_type == "short":
        for qa in qa_list:
            gold = clean(str(qa.get("answer", "")))
            if not gold:
                continue
            for i, m in enumerate(messages):
                content = clean(m.get("content", ""))
                if gold[:30] in content:
                    utility_idx.add(i)
    elif dataset_type == "long":
        for qa in qa_list:
            gold = str(qa.get("gold_answer", "")).lower()
            keywords = [clean(w) for w in gold.split() if len(w) > 3][:3]
            keywords = [k for k in keywords if k]
            if not keywords:
                continue
            for i, m in enumerate(messages):
                content = clean(m.get("content", ""))
                if all(kw in content for kw in keywords):
                    utility_idx.add(i)
    return utility_idx


def run_gate_on_messages(gate, messages: list[dict], threshold: float | None = None):
    """Run gate on all messages and return scores + decisions."""
    scores = []
    decisions = []
    latencies = []

    for i, msg in enumerate(messages):
        context = messages[max(0, i-3):i] if i > 0 else None
        t0 = time.perf_counter()
        score = gate.importance_score(msg, context)
        elapsed = (time.perf_counter() - t0) * 1000.0

        if threshold is not None:
            keep = score >= threshold
        else:
            keep = gate.should_encode(msg, context)

        scores.append(score)
        decisions.append(keep)
        latencies.append(elapsed)

    return scores, decisions, latencies


def compute_retrieval_precision(
    kept_messages: list[dict],
    qa_list: list[dict],
    dataset_type: str,
    db_path: Path,
) -> dict:
    """Ingest kept messages and measure retrieval precision."""
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("OPENROUTER_API_KEY", None)

    from truememory.engine import TrueMemoryEngine
    engine = TrueMemoryEngine(db_path=str(db_path))

    # Ingest all kept messages
    if kept_messages:
        ingest_data = []
        for m in kept_messages:
            ingest_data.append({
                "content": m.get("content", ""),
                "sender": m.get("sender", "user"),
                "recipient": "self",
                "timestamp": m.get("timestamp", ""),
                "category": "",
                "modality": "conversation",
            })

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(ingest_data, f)
            tmp = f.name
        try:
            engine.ingest(tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # Query and measure precision
    K = 10
    n_total = 0
    n_at = {1: 0, 3: 0, 10: 0}
    clean_fn = lambda s: re.sub(r"[^a-z0-9 ]+", "", s.lower())

    for qa in qa_list:
        n_total += 1
        question = qa.get("question", "")

        try:
            results = engine.search(question, limit=K)
        except Exception:
            continue

        if dataset_type == "short":
            gold = clean_fn(str(qa.get("answer", "")))
            if not gold:
                continue
            hit_pos = -1
            for i, r in enumerate(results[:K]):
                content = clean_fn(r.get("content", "") if isinstance(r, dict) else str(r))
                if gold[:30] in content:
                    hit_pos = i + 1
                    break
        else:
            gold = str(qa.get("gold_answer", "")).lower()
            keywords = [clean_fn(w) for w in gold.split() if len(w) > 3][:3]
            keywords = [k for k in keywords if k]
            if not keywords:
                continue
            probe = qa.get("probe_type", "baseline")
            if probe == "noise":
                continue  # skip noise probes for simplicity
            hit_pos = -1
            for i, r in enumerate(results[:K]):
                content = clean_fn(r.get("content", "") if isinstance(r, dict) else str(r))
                if all(kw in content for kw in keywords):
                    hit_pos = i + 1
                    break

        if hit_pos > 0:
            n_at[10] += 1
            if hit_pos <= 3:
                n_at[3] += 1
            if hit_pos == 1:
                n_at[1] += 1

    return {
        "total_qs": n_total,
        "p_at_1": round(100 * n_at[1] / max(n_total, 1), 2),
        "p_at_3": round(100 * n_at[3] / max(n_total, 1), 2),
        "p_at_10": round(100 * n_at[10] / max(n_total, 1), 2),
    }


def evaluate_candidate(
    gate,
    gate_name: str,
    messages: list[dict],
    qa_list: list[dict],
    dataset_type: str,
    dataset_name: str,
    thresholds: list[float],
    chitchat_idx: set[int],
    utility_idx: set[int],
    skip_retrieval: bool = False,
) -> list[dict]:
    """Evaluate a gate candidate at multiple thresholds."""
    results = []

    # First get scores for all messages (gate-specific computation)
    print(f"  Scoring {len(messages)} messages with {gate_name}...", flush=True)
    scores, _, latencies = run_gate_on_messages(gate, messages)

    for tau in thresholds:
        decisions = [s >= tau for s in scores]
        n_kept = sum(decisions)
        n_dropped = len(messages) - n_kept
        drop_rate = round(100 * n_dropped / max(len(messages), 1), 2)

        # Chitchat analysis
        chitchat_dropped = sum(1 for i in chitchat_idx if not decisions[i])
        chitchat_total = len(chitchat_idx)
        chitchat_drop_rate = round(100 * chitchat_dropped / max(chitchat_total, 1), 2)

        # False positive analysis (utility messages incorrectly dropped)
        utility_dropped = sum(1 for i in utility_idx if not decisions[i])
        utility_total = len(utility_idx)
        false_positive_rate = round(100 * utility_dropped / max(utility_total, 1), 2)

        # Latency stats
        import statistics
        lat_sorted = sorted(latencies)
        p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0
        p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if lat_sorted else 0
        p99 = lat_sorted[int(len(lat_sorted) * 0.99)] if lat_sorted else 0

        result = {
            "candidate": gate_name,
            "dataset": dataset_name,
            "threshold": tau,
            "n_messages": len(messages),
            "n_kept": n_kept,
            "n_dropped": n_dropped,
            "drop_rate_pct": drop_rate,
            "chitchat_total": chitchat_total,
            "chitchat_dropped": chitchat_dropped,
            "chitchat_drop_rate_pct": chitchat_drop_rate,
            "utility_total": utility_total,
            "utility_dropped": utility_dropped,
            "false_positive_rate_pct": false_positive_rate,
            "latency_p50_ms": round(p50, 3),
            "latency_p95_ms": round(p95, 3),
            "latency_p99_ms": round(p99, 3),
        }

        # Retrieval precision (expensive — skip for speed)
        if not skip_retrieval and n_kept > 0:
            kept_msgs = [m for m, d in zip(messages, decisions) if d]
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "test.db"
                try:
                    prec = compute_retrieval_precision(
                        kept_msgs, qa_list, dataset_type, db_path
                    )
                    result.update(prec)
                except Exception as e:
                    result["retrieval_error"] = str(e)

        results.append(result)
        print(
            f"    τ={tau:.2f}: drop={drop_rate}% kept={n_kept} "
            f"chitchat_drop={chitchat_drop_rate}% FP={false_positive_rate}% "
            f"p95={p95:.1f}ms",
            flush=True,
        )

    return results


def main():
    print("=" * 70)
    print("GATE DEEP RESEARCH — Candidate Sweep")
    print("=" * 70)

    # Import candidates
    from benchmarks.gate_eval.candidates.gate_deep.d00_store_all import D00StoreAll
    from benchmarks.gate_eval.candidates.gate_deep.d01_random_drop import D01RandomDrop
    from benchmarks.gate_eval.candidates.gate_deep.d02_length_only import D02LengthOnly
    from benchmarks.gate_eval.candidates.gate_deep.d03_regex_only import D03RegexOnly
    from benchmarks.gate_eval.candidates.gate_deep.c02_compression import C02Compression

    # Thresholds for candidates with continuous scores
    thresholds_continuous = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    thresholds_binary = [0.50]  # For binary candidates

    candidates = [
        ("d00_store_all", D00StoreAll(), thresholds_binary),
        ("d01_random_50", D01RandomDrop(drop_rate=0.50), thresholds_binary),
        ("d02_length_20", D02LengthOnly(min_length=20), thresholds_binary),
        ("d03_regex", D03RegexOnly(), thresholds_binary),
        ("c02_compression", C02Compression(context_window=30), thresholds_continuous),
    ]

    # Try NLI (may fail if transformers not installed)
    try:
        from benchmarks.gate_eval.candidates.gate_deep.c01_nli_zeroshot import C01NliZeroshot
        candidates.append(
            ("c01_nli_zeroshot", C01NliZeroshot(threshold=0.50), thresholds_continuous)
        )
    except ImportError as e:
        print(f"  Skipping C01 NLI: {e}")

    # Try embedding novelty
    try:
        from benchmarks.gate_eval.candidates.gate_deep.c03_embedding_novelty import C03EmbeddingNovelty
        candidates.append(
            ("c03_embedding_novelty", C03EmbeddingNovelty(), thresholds_continuous)
        )
    except ImportError as e:
        print(f"  Skipping C03 Embedding: {e}")

    # Try hybrid cascade
    try:
        from benchmarks.gate_eval.candidates.gate_deep.c08_hybrid_cascade import C08HybridCascade
        candidates.append(
            ("c08_hybrid_edge", C08HybridCascade(use_nli=False, use_context=False), thresholds_binary)
        )
        candidates.append(
            ("c08_hybrid_base", C08HybridCascade(use_nli=True, use_context=False), thresholds_continuous)
        )
    except ImportError as e:
        print(f"  Skipping C08 Hybrid: {e}")

    # Load datasets
    print("\nLoading datasets...")
    short_ds = load_short_horizon()
    long_ds = load_long_horizon()

    short_msgs = extract_messages_short(short_ds)
    long_msgs = extract_messages_long(long_ds)
    print(f"  short_horizon: {len(short_msgs)} messages, {len(short_ds['qa'])} queries")
    print(f"  long_horizon:  {len(long_msgs)} messages, {len(long_ds['retrieval_queries'])} queries")

    # Identify chitchat and utility messages
    short_chitchat = identify_chitchat(short_msgs)
    long_chitchat = identify_chitchat(long_msgs)
    short_utility = identify_utility_messages(short_msgs, short_ds["qa"], "short")
    long_utility = identify_utility_messages(
        long_msgs, long_ds["retrieval_queries"], "long"
    )
    print(f"  short chitchat: {len(short_chitchat)}, utility: {len(short_utility)}")
    print(f"  long  chitchat: {len(long_chitchat)}, utility: {len(long_utility)}")

    all_results = []

    for cand_name, gate, thresholds in candidates:
        print(f"\n{'='*50}")
        print(f"Candidate: {cand_name}")
        print(f"{'='*50}")

        # Reset stateful candidates
        if hasattr(gate, "reset"):
            gate.reset()

        # Short horizon
        print(f"\n  Dataset: short_horizon_200")
        results = evaluate_candidate(
            gate, cand_name, short_msgs, short_ds["qa"],
            "short", "short_horizon_200", thresholds,
            short_chitchat, short_utility,
            skip_retrieval=True,
        )
        all_results.extend(results)

        # Reset and run long horizon
        if hasattr(gate, "reset"):
            gate.reset()

        print(f"\n  Dataset: long_horizon_synthetic")
        results = evaluate_candidate(
            gate, cand_name, long_msgs, long_ds["retrieval_queries"],
            "long", "long_horizon_synthetic", thresholds,
            long_chitchat, long_utility,
            skip_retrieval=True,
        )
        all_results.extend(results)

    # Save results
    out_path = RESULTS_DIR / "sweep_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n\nResults saved to {out_path}")
    print(f"Total cells: {len(all_results)}")

    return all_results


if __name__ == "__main__":
    main()
