#!/usr/bin/env python3
"""Gate weight/threshold sweep — 200 configs × 2000 messages × 200 questions.

Evaluates encoding gate signal weights (novelty, salience, prediction_error)
and thresholds to find optimal gate settings that maximize signal retention
while dropping noise. Pure retrieval evaluation — no answer generation,
no judge, no API calls.

Usage:
    .venv/bin/python3 benchmarks/gate_eval/run_gate_weight_sweep.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy L5 surprise warnings
logging.getLogger("truememory.engine").setLevel(logging.ERROR)

BENCHMARK_PATH = PROJECT_ROOT / "benchmarks" / "gate_eval" / "datasets" / "gate_benchmark.json"
RESULTS_PATH = PROJECT_ROOT / "benchmarks" / "gate_eval" / "results" / "gate_weight_sweep.json"

WEIGHT_CONFIGS = [
    {"w_n": 0.40, "w_s": 0.35, "w_p": 0.25, "label": "default"},
    {"w_n": 0.33, "w_s": 0.34, "w_p": 0.33, "label": "equal"},
    {"w_n": 0.60, "w_s": 0.25, "w_p": 0.15, "label": "novelty_heavy"},
    {"w_n": 0.70, "w_s": 0.20, "w_p": 0.10, "label": "novelty_dominant"},
    {"w_n": 0.20, "w_s": 0.55, "w_p": 0.25, "label": "salience_heavy"},
    {"w_n": 0.15, "w_s": 0.60, "w_p": 0.25, "label": "salience_dominant"},
    {"w_n": 0.10, "w_s": 0.70, "w_p": 0.20, "label": "salience_max"},
    {"w_n": 0.25, "w_s": 0.25, "w_p": 0.50, "label": "pe_heavy"},
    {"w_n": 0.20, "w_s": 0.20, "w_p": 0.60, "label": "pe_dominant"},
    {"w_n": 0.05, "w_s": 0.55, "w_p": 0.40, "label": "novelty_suppressed"},
    {"w_n": 0.00, "w_s": 0.60, "w_p": 0.40, "label": "no_novelty"},
    {"w_n": 0.00, "w_s": 1.00, "w_p": 0.00, "label": "salience_only"},
    {"w_n": 0.50, "w_s": 0.50, "w_p": 0.00, "label": "no_pe"},
    {"w_n": 0.30, "w_s": 0.70, "w_p": 0.00, "label": "ns_salience_lean"},
    {"w_n": 0.50, "w_s": 0.50, "w_p": 0.00, "label": "ns_balanced"},
    {"w_n": 0.70, "w_s": 0.30, "w_p": 0.00, "label": "ns_novelty_lean"},
    {"w_n": 0.50, "w_s": 0.00, "w_p": 0.50, "label": "np_balanced"},
    {"w_n": 0.70, "w_s": 0.00, "w_p": 0.30, "label": "np_novelty_lean"},
    {"w_n": 0.00, "w_s": 0.50, "w_p": 0.50, "label": "sp_balanced"},
    {"w_n": 0.00, "w_s": 0.70, "w_p": 0.30, "label": "sp_salience_lean"},
]

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def load_benchmark() -> dict:
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def flatten_messages(benchmark: dict) -> list[dict]:
    messages = []
    for conv in benchmark["conversations"]:
        for msg in conv["messages"]:
            messages.append(msg)
    return messages


def config_key(label: str, threshold: float) -> str:
    return f"{label}@{threshold:.2f}"


def load_completed() -> dict[str, dict]:
    if not RESULTS_PATH.exists():
        return {}
    try:
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        return {config_key(r["label"], r["threshold"]): r for r in data.get("results", [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_results(completed: dict[str, dict], benchmark: dict) -> None:
    sorted_results = sorted(
        completed.values(),
        key=lambda r: (-r["p_at_10"], -r["noise_drop_rate"]),
    )
    output = {
        "sweep_date": "2026-04-29",
        "benchmark": f"GateLoCoMo v{benchmark['version']}",
        "total_configs": len(sorted_results),
        "total_messages": benchmark["total_messages"],
        "total_questions": benchmark["total_questions"],
        "tier": "pro",
        "hyde": False,
        "results": sorted_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)


def run_single_config(
    messages: list[dict],
    questions: list[dict],
    weight_config: dict,
    threshold: float,
    signal_ids: set[str],
    noise_ids: set[str],
) -> dict:
    from truememory import Memory
    from truememory.ingest.encoding_gate import EncodingGate

    tmpdir = tempfile.mkdtemp(prefix="gate_sweep_")
    db_path = os.path.join(tmpdir, "sweep.db")

    try:
        t0 = time.time()

        memory = Memory(path=db_path)
        engine = memory._engine

        gate = EncodingGate(
            memory=memory,
            threshold=threshold,
            w_novelty=weight_config["w_n"],
            w_salience=weight_config["w_s"],
            w_prediction_error=weight_config["w_p"],
        )

        bench_to_engine: dict[str, int] = {}
        dropped_ids: set[str] = set()

        for msg in messages:
            decision = gate.evaluate(msg["content"], msg.get("category", ""))
            if decision.should_encode:
                result = engine.add(
                    content=msg["content"],
                    sender=msg.get("speaker", ""),
                    recipient=msg.get("recipient", ""),
                    timestamp=msg.get("timestamp", ""),
                    category=msg.get("category", ""),
                )
                engine_id = result.get("id")
                if engine_id is not None:
                    bench_to_engine[msg["id"]] = engine_id
            else:
                dropped_ids.add(msg["id"])

        messages_kept = len(bench_to_engine)
        messages_dropped = len(dropped_ids)

        signal_kept = len(signal_ids - dropped_ids)
        signal_total = len(signal_ids)
        noise_dropped = len(noise_ids & dropped_ids)
        noise_total = len(noise_ids)

        engine_to_bench = {v: k for k, v in bench_to_engine.items()}

        hits = 0
        for q in questions:
            try:
                search_results = engine.search(q["question"], limit=10, _skip_surprise_boost=True)
            except Exception:
                search_results = []

            result_bench_ids = set()
            for r in search_results:
                eid = r.get("id")
                if eid is not None and eid in engine_to_bench:
                    result_bench_ids.add(engine_to_bench[eid])

            evidence_ids = set(q["evidence_messages"])
            if result_bench_ids & evidence_ids:
                hits += 1

        memory.close()

        elapsed = time.time() - t0
        p_at_10 = hits / len(questions) if questions else 0.0
        signal_retention = signal_kept / signal_total if signal_total else 1.0
        noise_drop_rate = noise_dropped / noise_total if noise_total else 0.0
        drop_rate = messages_dropped / (messages_kept + messages_dropped) if (messages_kept + messages_dropped) else 0.0

        return {
            "label": weight_config["label"],
            "w_novelty": weight_config["w_n"],
            "w_salience": weight_config["w_s"],
            "w_pe": weight_config["w_p"],
            "threshold": threshold,
            "messages_kept": messages_kept,
            "messages_dropped": messages_dropped,
            "drop_rate": round(drop_rate, 4),
            "signal_kept": signal_kept,
            "signal_total": signal_total,
            "signal_retention": round(signal_retention, 4),
            "noise_dropped": noise_dropped,
            "noise_total": noise_total,
            "noise_drop_rate": round(noise_drop_rate, 4),
            "p_at_10": round(p_at_10, 4),
            "questions_hit": hits,
            "questions_total": len(questions),
            "elapsed_s": round(elapsed, 1),
        }

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_gate_off_baseline(
    messages: list[dict],
    questions: list[dict],
    signal_ids: set[str],
    noise_ids: set[str],
) -> dict:
    from truememory import Memory

    tmpdir = tempfile.mkdtemp(prefix="gate_sweep_baseline_")
    db_path = os.path.join(tmpdir, "baseline.db")

    try:
        t0 = time.time()
        memory = Memory(path=db_path)
        engine = memory._engine

        bench_to_engine: dict[str, int] = {}
        for msg in messages:
            result = engine.add(
                content=msg["content"],
                sender=msg.get("speaker", ""),
                recipient=msg.get("recipient", ""),
                timestamp=msg.get("timestamp", ""),
                category=msg.get("category", ""),
            )
            engine_id = result.get("id")
            if engine_id is not None:
                bench_to_engine[msg["id"]] = engine_id

        engine_to_bench = {v: k for k, v in bench_to_engine.items()}

        hits = 0
        for q in questions:
            try:
                search_results = engine.search(q["question"], limit=10, _skip_surprise_boost=True)
            except Exception:
                search_results = []

            result_bench_ids = set()
            for r in search_results:
                eid = r.get("id")
                if eid is not None and eid in engine_to_bench:
                    result_bench_ids.add(engine_to_bench[eid])

            evidence_ids = set(q["evidence_messages"])
            if result_bench_ids & evidence_ids:
                hits += 1

        memory.close()

        elapsed = time.time() - t0
        p_at_10 = hits / len(questions) if questions else 0.0

        return {
            "label": "gate_off",
            "w_novelty": 0.0,
            "w_salience": 0.0,
            "w_pe": 0.0,
            "threshold": 0.0,
            "messages_kept": len(bench_to_engine),
            "messages_dropped": 0,
            "drop_rate": 0.0,
            "signal_kept": len(signal_ids),
            "signal_total": len(signal_ids),
            "signal_retention": 1.0,
            "noise_dropped": 0,
            "noise_total": len(noise_ids),
            "noise_drop_rate": 0.0,
            "p_at_10": round(p_at_10, 4),
            "questions_hit": hits,
            "questions_total": len(questions),
            "elapsed_s": round(elapsed, 1),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def print_summary(completed: dict[str, dict]) -> None:
    print("\n" + "=" * 70)
    print("[5/5] TOP 10 CONFIGS BY p@10")
    print("=" * 70)

    results = sorted(completed.values(), key=lambda r: (-r["p_at_10"], -r["noise_drop_rate"]))

    header = f"{'Rank':<5} {'Label':<22} {'w_n':>5} {'w_s':>5} {'w_p':>5} {'Thr':>5} {'p@10':>6} {'SigRet':>7} {'NsDrop':>7} {'Drop%':>6}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results[:10], 1):
        print(
            f"{i:<5} {r['label']:<22} "
            f"{r['w_novelty']:>5.2f} {r['w_salience']:>5.2f} {r['w_pe']:>5.2f} "
            f"{r['threshold']:>5.2f} {r['p_at_10']:>6.4f} "
            f"{r['signal_retention']:>6.1%} {r['noise_drop_rate']:>6.1%} "
            f"{r['drop_rate']:>5.1%}"
        )

    baseline = next((r for r in results if r["label"] == "gate_off"), None)
    if baseline:
        print(f"\nBaseline (gate OFF): p@10={baseline['p_at_10']:.4f}")


def main():
    print("=" * 70)
    print("Gate Weight/Threshold Sweep")
    print("=" * 70)

    print("\n[1/5] Loading benchmark data...")
    benchmark = load_benchmark()
    messages = flatten_messages(benchmark)
    questions = benchmark["questions"]
    print(f"  Messages: {len(messages)}")
    print(f"  Questions: {len(questions)}")
    print(f"  Version: {benchmark['version']}")

    signal_ids = {m["id"] for m in messages if m.get("category", "").startswith("S")}
    noise_ids = {m["id"] for m in messages if m.get("category", "").startswith("N")}
    borderline_ids = {m["id"] for m in messages if m.get("category", "").startswith("B")}
    print(f"  Signal: {len(signal_ids)}, Noise: {len(noise_ids)}, Borderline: {len(borderline_ids)}")

    print("\n[2/5] Loading Pro tier models (one-time)...")
    from truememory.vector_search import set_embedding_model
    set_embedding_model("pro")
    from truememory.reranker import get_reranker, set_active_tier
    set_active_tier("pro")
    get_reranker(model_name="Alibaba-NLP/gte-reranker-modernbert-base")
    print("  Embedding: Qwen3 256d")
    print("  Reranker: gte-reranker-modernbert-base")

    print("\n[3/5] Running sweep...")
    total_configs = len(WEIGHT_CONFIGS) * len(THRESHOLDS)
    print(f"  Total configs: {total_configs}")

    completed = load_completed()
    if completed:
        print(f"  Resuming: {len(completed)} configs already done")

    baseline_key = config_key("gate_off", 0.0)
    if baseline_key not in completed:
        print("\n  Running baseline (gate OFF)...")
        baseline_result = run_gate_off_baseline(messages, questions, signal_ids, noise_ids)
        completed[baseline_key] = baseline_result
        save_results(completed, benchmark)
        print(f"  Baseline p@10={baseline_result['p_at_10']:.4f} ({baseline_result['elapsed_s']:.1f}s)")

    config_num = 0
    sweep_start = time.time()
    for wc in WEIGHT_CONFIGS:
        for thresh in THRESHOLDS:
            config_num += 1
            key = config_key(wc["label"], thresh)
            if key in completed:
                continue

            result = run_single_config(messages, questions, wc, thresh, signal_ids, noise_ids)
            completed[key] = result
            save_results(completed, benchmark)

            done = len([r for r in completed.values() if r["label"] != "gate_off"])
            elapsed_total = time.time() - sweep_start
            eta = (elapsed_total / max(done, 1)) * (total_configs - done)

            print(
                f"  [{config_num}/{total_configs}] {wc['label']} @ {thresh:.2f}: "
                f"p@10={result['p_at_10']:.4f} "
                f"sig_ret={result['signal_retention']:.0%} "
                f"ns_drop={result['noise_drop_rate']:.0%} "
                f"kept={result['messages_kept']} "
                f"({result['elapsed_s']:.0f}s) "
                f"ETA={eta/3600:.1f}h",
                flush=True,
            )

    print(f"\n[4/5] Sweep complete. {len(completed)} results saved to {RESULTS_PATH}")

    print_summary(completed)


if __name__ == "__main__":
    main()
