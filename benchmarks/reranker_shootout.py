#!/usr/bin/env python3
"""
Reranker Shootout — Benchmark multiple rerankers on the same candidate pool.

Tests speed and result quality across models using neuromem's actual memory DB.
Outputs a markdown comparison table.
"""

import json
import sys
import time
from pathlib import Path

# Add neuromem to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromem.engine import NeuromemEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = Path.home() / ".neuromem" / "memories.db"
OUTPUT_PATH = Path.home() / "Desktop" / "RERANKER_SHOOTOUT.md"

# 10 diverse test queries
QUERIES = [
    "user preferences and style",
    "rustling the feathers analysis pattern",
    "GPU box setup RTX 5090",
    "blogging medium substack AI memory",
    "neuromem benchmark results locomo",
    "paradox pi node camera",
    "project sunrise dashboard",
    "skippy texting imessage",
    "network topology switches router",
    "interrogation mode calibration",
]

# Rerankers to test (model_name for sentence_transformers CrossEncoder)
RERANKERS = [
    {
        "name": "ms-marco-MiniLM-L-6-v2",
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "params": "22M",
        "type": "cross-encoder",
    },
    {
        "name": "ms-marco-MiniLM-L-12-v2",
        "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "params": "33M",
        "type": "cross-encoder",
    },
    {
        "name": "bge-reranker-v2-m3",
        "model": "BAAI/bge-reranker-v2-m3",
        "params": "568M",
        "type": "cross-encoder",
    },
    {
        "name": "mxbai-rerank-large-v1",
        "model": "mixedbread-ai/mxbai-rerank-large-v1",
        "params": "560M",
        "type": "cross-encoder",
    },
]

TOP_K = 10  # Results to return after reranking
CANDIDATE_POOL = 50  # Candidates to pull before reranking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_candidates(engine, query, limit=CANDIDATE_POOL):
    """Get raw candidates from the 6-layer pipeline (no reranking)."""
    results = engine.search(query, limit=limit)
    return results


def rerank_with_model(query, candidates, model_name, device="cpu", batch_size=64):
    """Rerank candidates using a specific cross-encoder model."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device)
    pairs = [(query, r.get("content", "")) for r in candidates]

    t0 = time.time()
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    t1 = time.time()

    # Attach scores and sort
    for i, r in enumerate(candidates):
        r["rerank_score"] = float(scores[i])

    ranked = sorted(candidates, key=lambda x: -x["rerank_score"])
    return ranked[:TOP_K], t1 - t0


def format_result(r, max_len=80):
    """Format a single result for display."""
    content = r.get("content", "")[:max_len]
    score = r.get("rerank_score", r.get("score", 0))
    return f"[{score:.3f}] {content}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("RERANKER SHOOTOUT")
    print("=" * 70)
    print(f"DB: {DB_PATH}")
    print(f"Queries: {len(QUERIES)}")
    print(f"Rerankers: {len(RERANKERS)}")
    print(f"Candidate pool: {CANDIDATE_POOL}")
    print(f"Top-k output: {TOP_K}")
    print()

    # Init engine
    engine = NeuromemEngine(db_path=str(DB_PATH))
    engine._ensure_connection()
    msg_count = engine.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    print(f"Memories in DB: {msg_count}")
    print()

    # Phase 1: Get candidates for all queries (without reranking)
    print("--- Phase 1: Generating candidate pools ---")
    candidates_by_query = {}
    for q in QUERIES:
        candidates = get_candidates(engine, q, limit=CANDIDATE_POOL)
        candidates_by_query[q] = candidates
        print(f"  \"{q[:40]}...\": {len(candidates)} candidates")

    # Also get baseline (no reranking) results
    baseline_results = {}
    for q in QUERIES:
        baseline_results[q] = candidates_by_query[q][:TOP_K]

    # Phase 2: Run each reranker
    all_results = {}  # {reranker_name: {query: (results, time)}}

    for rr in RERANKERS:
        name = rr["name"]
        model = rr["model"]
        params = rr["params"]
        print()
        print(f"--- Testing: {name} ({params}) ---")

        # Cold start: load model
        t_cold_start = time.time()
        try:
            from sentence_transformers import CrossEncoder
            _ = CrossEncoder(model, device="cpu")
            t_cold_end = time.time()
            cold_start = t_cold_end - t_cold_start
            print(f"  Model loaded in {cold_start:.2f}s")
        except Exception as e:
            print(f"  FAILED to load: {e}")
            all_results[name] = None
            continue

        # Warm runs
        query_results = {}
        total_time = 0
        for q in QUERIES:
            cands = candidates_by_query[q]
            if not cands:
                query_results[q] = ([], 0)
                continue

            # Deep copy candidates so rerank scores don't bleed
            import copy
            cands_copy = copy.deepcopy(cands)

            ranked, elapsed = rerank_with_model(q, cands_copy, model)
            query_results[q] = (ranked, elapsed)
            total_time += elapsed
            print(f"  \"{q[:35]}...\": {len(ranked)} results in {elapsed:.3f}s")

        all_results[name] = {
            "params": params,
            "cold_start": cold_start,
            "total_time": total_time,
            "avg_time": total_time / len(QUERIES),
            "queries": query_results,
        }
        print(f"  TOTAL: {total_time:.2f}s | AVG: {total_time/len(QUERIES):.3f}s/query")

    engine.close()

    # Phase 3: Build comparison report
    print()
    print("--- Generating report ---")

    md = []
    md.append("# Reranker Shootout Results")
    md.append("")
    md.append(f"**Date:** 2026-04-07")
    md.append(f"**DB:** {msg_count} memories")
    md.append(f"**Candidate pool:** {CANDIDATE_POOL} per query")
    md.append(f"**Top-k output:** {TOP_K}")
    md.append(f"**Device:** CPU (Apple Silicon)")
    md.append("")

    # Speed comparison table
    md.append("## Speed Comparison")
    md.append("")
    md.append("| Reranker | Params | Cold Start | Avg/Query | Total (10q) | Status |")
    md.append("|---|---|---|---|---|---|")

    # Baseline row
    md.append(f"| No reranker (baseline) | — | — | ~0ms | ~0ms | OK |")

    for rr in RERANKERS:
        name = rr["name"]
        data = all_results.get(name)
        if data is None:
            md.append(f"| {name} | {rr['params']} | — | — | — | FAILED |")
        else:
            md.append(
                f"| {name} | {data['params']} | {data['cold_start']:.2f}s | "
                f"{data['avg_time']:.3f}s | {data['total_time']:.2f}s | OK |"
            )

    # Quality comparison — show top 3 results per query per reranker
    md.append("")
    md.append("## Quality Comparison (Top 3 Results Per Query)")
    md.append("")

    for q in QUERIES:
        md.append(f"### Query: \"{q}\"")
        md.append("")

        # Baseline
        md.append("**No reranker (baseline):**")
        for r in baseline_results[q][:3]:
            md.append(f"- {format_result(r)}")
        if not baseline_results[q]:
            md.append("- (no results)")
        md.append("")

        # Each reranker
        for rr in RERANKERS:
            name = rr["name"]
            data = all_results.get(name)
            if data is None:
                md.append(f"**{name}:** FAILED")
                md.append("")
                continue

            results, elapsed = data["queries"].get(q, ([], 0))
            md.append(f"**{name}** ({elapsed:.3f}s):")
            for r in results[:3]:
                md.append(f"- {format_result(r)}")
            if not results:
                md.append("- (no results)")
            md.append("")

    # Per-query timing table
    md.append("## Per-Query Timing (seconds)")
    md.append("")
    header = "| Query | " + " | ".join(rr["name"] for rr in RERANKERS) + " |"
    sep = "|---|" + "|".join("---" for _ in RERANKERS) + "|"
    md.append(header)
    md.append(sep)

    for q in QUERIES:
        row = f"| {q[:35]}... | "
        cells = []
        for rr in RERANKERS:
            name = rr["name"]
            data = all_results.get(name)
            if data is None:
                cells.append("FAIL")
            else:
                _, elapsed = data["queries"].get(q, ([], 0))
                cells.append(f"{elapsed:.3f}")
        row += " | ".join(cells) + " |"
        md.append(row)

    # Recommendation
    md.append("")
    md.append("## Recommendation")
    md.append("")

    # Find fastest and best
    valid = {k: v for k, v in all_results.items() if v is not None}
    if valid:
        fastest = min(valid.items(), key=lambda x: x[1]["avg_time"])
        slowest = max(valid.items(), key=lambda x: x[1]["avg_time"])
        md.append(f"- **Fastest:** {fastest[0]} at {fastest[1]['avg_time']:.3f}s/query")
        md.append(f"- **Slowest:** {slowest[0]} at {slowest[1]['avg_time']:.3f}s/query")
        md.append(f"- **Speedup:** {slowest[1]['avg_time'] / fastest[1]['avg_time']:.1f}x")
        md.append("")
        md.append("Review the quality comparison above to determine if the fastest model's")
        md.append("result quality is acceptable compared to the larger models.")

    # Write output
    output = "\n".join(md)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"Report written to {OUTPUT_PATH}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
