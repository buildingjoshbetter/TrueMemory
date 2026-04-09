#!/usr/bin/env python3
"""
Reranker Mega-Shootout — 20 models, speed + quality comparison.

Tests every viable cross-encoder reranker from ~4M to ~600M params.
Same candidate pool, same queries, head-to-head.
"""

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromem.engine import NeuromemEngine

DB_PATH = Path.home() / ".neuromem" / "memories.db"
OUTPUT_PATH = Path.home() / "Desktop" / "RERANKER_SHOOTOUT_20.md"

# 10 diverse test queries
QUERIES = [
    "user preferences",
    "rustling the feathers analysis",
    "GPU box RTX 5090",
    "blogging substack medium",
    "neuromem benchmark locomo",
    "paradox pi node camera",
    "project sunrise dashboard",
    "skippy texting imessage",
    "network topology router",
    "interrogation mode",
]

# 20 rerankers sorted by parameter count (ascending)
RERANKERS = [
    # ~4M - Tiny
    {"name": "TinyBERT-L2-v2", "model": "cross-encoder/ms-marco-TinyBERT-L2-v2", "params": "4.4M"},
    # ~16M - Very Small
    {"name": "MiniLM-L2-v2", "model": "cross-encoder/ms-marco-MiniLM-L-2-v2", "params": "16M"},
    # ~19M
    {"name": "MiniLM-L4-v2", "model": "cross-encoder/ms-marco-MiniLM-L-4-v2", "params": "19M"},
    # ~22M - Current standard
    {"name": "MiniLM-L6-v2", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "params": "22M"},
    # ~33M
    {"name": "MiniLM-L12-v2", "model": "cross-encoder/ms-marco-MiniLM-L-12-v2", "params": "33M"},
    # ~100M
    {"name": "mxbai-rerank-xsmall", "model": "mixedbread-ai/mxbai-rerank-xsmall-v1", "params": "100M"},
    # ~109M
    {"name": "bge-reranker-base", "model": "BAAI/bge-reranker-base", "params": "109M"},
    # ~110M
    {"name": "electra-base", "model": "cross-encoder/ms-marco-electra-base", "params": "110M"},
    # ~149M
    {"name": "granite-reranker", "model": "ibm-granite/granite-embedding-reranker-english-r2", "params": "149M"},
    # ~149M
    {"name": "gte-modernbert-base", "model": "Alibaba-NLP/gte-reranker-modernbert-base", "params": "149M"},
    # ~150M
    {"name": "ModernBERT-base-gooaq", "model": "tomaarsen/reranker-ModernBERT-base-gooaq-bce", "params": "150M"},
    # ~150M
    {"name": "ModernBERT-base-msmarco", "model": "tomaarsen/reranker-msmarco-ModernBERT-base-lambdaloss", "params": "150M"},
    # ~200M
    {"name": "mxbai-rerank-base", "model": "mixedbread-ai/mxbai-rerank-base-v1", "params": "200M"},
    # ~278M
    {"name": "bge-reranker-v2-m3", "model": "BAAI/bge-reranker-v2-m3", "params": "278M"},
    # ~280M
    {"name": "jina-reranker-v2-base", "model": "jinaai/jina-reranker-v2-base-multilingual", "params": "~280M"},
    # ~305M
    {"name": "gte-multilingual-reranker", "model": "Alibaba-NLP/gte-multilingual-reranker-base", "params": "~305M"},
    # ~395M
    {"name": "ModernBERT-large-gooaq", "model": "tomaarsen/reranker-ModernBERT-large-gooaq-bce", "params": "395M"},
    # ~335M
    {"name": "bge-reranker-large", "model": "BAAI/bge-reranker-large", "params": "335M"},
    # ~560M
    {"name": "mxbai-rerank-large", "model": "mixedbread-ai/mxbai-rerank-large-v1", "params": "560M"},
    # ~600M
    {"name": "Qwen3-Reranker-0.6B", "model": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls", "params": "600M"},
]

TOP_K = 10
CANDIDATE_POOL = 100  # Realistic pool size


def get_candidates(engine, query, limit=CANDIDATE_POOL):
    """Get raw candidates from the 6-layer pipeline."""
    return engine.search(query, limit=limit)


def rerank_candidates(query, candidates, model, batch_size=64):
    """Rerank using a preloaded model. Returns (results, inference_time)."""
    if not candidates:
        return [], 0.0
    pairs = [(query, r.get("content", "")) for r in candidates]
    t0 = time.time()
    try:
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    except Exception as e:
        return candidates[:TOP_K], 0.0
    t1 = time.time()
    for i, r in enumerate(candidates):
        r["rerank_score"] = float(scores[i])
    ranked = sorted(candidates, key=lambda x: -x["rerank_score"])
    return ranked[:TOP_K], t1 - t0


def main():
    print("=" * 70)
    print("RERANKER MEGA-SHOOTOUT — 20 MODELS")
    print("=" * 70)

    engine = NeuromemEngine(db_path=str(DB_PATH))
    engine._ensure_connection()
    msg_count = engine.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    print(f"DB: {msg_count} memories | Candidate pool: {CANDIDATE_POOL} | Top-k: {TOP_K}")
    print(f"Models: {len(RERANKERS)}")
    print()

    # Phase 1: Get candidates
    print("--- Phase 1: Building candidate pools ---")
    candidates_by_query = {}
    for q in QUERIES:
        c = get_candidates(engine, q, limit=CANDIDATE_POOL)
        candidates_by_query[q] = c
        print(f"  \"{q}\": {len(c)} candidates")
    engine.close()

    # Phase 2: Test each reranker
    results_all = {}

    for i, rr in enumerate(RERANKERS):
        name = rr["name"]
        model_id = rr["model"]
        params = rr["params"]
        print()
        print(f"[{i+1}/{len(RERANKERS)}] {name} ({params}) — {model_id}")

        # Load model
        t_load_start = time.time()
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_id, device="cpu")
            t_load_end = time.time()
            load_time = t_load_end - t_load_start
            print(f"  Loaded in {load_time:.2f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            results_all[name] = {"status": "FAILED", "error": str(e)[:100], "params": params}
            continue

        # Run queries
        query_results = {}
        total_infer = 0
        for q in QUERIES:
            cands = copy.deepcopy(candidates_by_query[q])
            ranked, elapsed = rerank_candidates(q, cands, model)
            query_results[q] = {"results": ranked, "time": elapsed, "count": len(ranked)}
            total_infer += elapsed

        avg_infer = total_infer / len(QUERIES) if QUERIES else 0
        print(f"  Total inference: {total_infer:.2f}s | Avg: {avg_infer:.3f}s/query")

        results_all[name] = {
            "status": "OK",
            "params": params,
            "model": model_id,
            "load_time": load_time,
            "total_infer": total_infer,
            "avg_infer": avg_infer,
            "queries": query_results,
        }

        # Free model memory
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        import gc
        gc.collect()

    # Phase 3: Generate report
    print()
    print("--- Generating report ---")

    md = []
    md.append("# Reranker Mega-Shootout: 20 Models Compared")
    md.append("")
    md.append(f"**Date:** 2026-04-07")
    md.append(f"**DB:** {msg_count} memories")
    md.append(f"**Candidate pool:** {CANDIDATE_POOL} per query")
    md.append(f"**Queries:** {len(QUERIES)}")
    md.append(f"**Device:** CPU (Apple Silicon)")
    md.append("")

    # Main speed table
    md.append("## Speed Comparison (sorted by params)")
    md.append("")
    md.append("| # | Model | Params | Load Time | Avg/Query | Total (10q) | Status |")
    md.append("|---|---|---|---|---|---|---|")

    for i, rr in enumerate(RERANKERS):
        name = rr["name"]
        data = results_all.get(name, {})
        if data.get("status") == "FAILED":
            md.append(f"| {i+1} | {name} | {rr['params']} | — | — | — | FAILED |")
        elif data.get("status") == "OK":
            md.append(
                f"| {i+1} | {name} | {data['params']} | {data['load_time']:.1f}s | "
                f"**{data['avg_infer']:.3f}s** | {data['total_infer']:.2f}s | OK |"
            )

    # Quality comparison — top 3 results for each query, each model
    md.append("")
    md.append("## Quality Comparison (Top 3 Per Query)")
    md.append("")
    md.append("Only showing models that loaded successfully.")
    md.append("")

    ok_models = [rr for rr in RERANKERS if results_all.get(rr["name"], {}).get("status") == "OK"]

    for q in QUERIES:
        md.append(f"### \"{q}\"")
        md.append("")
        for rr in ok_models:
            name = rr["name"]
            data = results_all[name]
            qdata = data["queries"].get(q, {})
            results = qdata.get("results", [])
            elapsed = qdata.get("time", 0)
            md.append(f"**{name}** ({elapsed:.3f}s):")
            if not results:
                md.append("- (no results)")
            else:
                for r in results[:3]:
                    content = r.get("content", "")[:80]
                    score = r.get("rerank_score", 0)
                    md.append(f"- [{score:.3f}] {content}")
            md.append("")

    # Per-query timing matrix
    md.append("## Per-Query Timing Matrix (seconds)")
    md.append("")
    # Only show OK models
    header = "| Query | " + " | ".join(rr["name"][:15] for rr in ok_models) + " |"
    sep = "|---|" + "|".join("---" for _ in ok_models) + "|"
    md.append(header)
    md.append(sep)
    for q in QUERIES:
        cells = []
        for rr in ok_models:
            data = results_all[rr["name"]]
            t = data["queries"].get(q, {}).get("time", 0)
            cells.append(f"{t:.3f}")
        md.append(f"| {q[:25]} | " + " | ".join(cells) + " |")

    # Speed vs quality summary
    md.append("")
    md.append("## Speed Tiers")
    md.append("")
    md.append("| Tier | Models | Avg/Query | Use Case |")
    md.append("|---|---|---|---|")

    instant = [rr["name"] for rr in ok_models if results_all[rr["name"]]["avg_infer"] < 0.1]
    fast = [rr["name"] for rr in ok_models if 0.1 <= results_all[rr["name"]]["avg_infer"] < 1.0]
    medium = [rr["name"] for rr in ok_models if 1.0 <= results_all[rr["name"]]["avg_infer"] < 10.0]
    slow = [rr["name"] for rr in ok_models if results_all[rr["name"]]["avg_infer"] >= 10.0]

    if instant:
        md.append(f"| Instant (<100ms) | {', '.join(instant)} | <0.1s | Default search |")
    if fast:
        md.append(f"| Fast (100ms-1s) | {', '.join(fast)} | 0.1-1s | Standard search |")
    if medium:
        md.append(f"| Medium (1-10s) | {', '.join(medium)} | 1-10s | Enhanced search |")
    if slow:
        md.append(f"| Slow (10s+) | {', '.join(slow)} | 10s+ | Deep search only |")

    md.append("")
    md.append("## Next Step")
    md.append("")
    md.append("Pick the top 3 candidates (best quality from each speed tier) and run them")
    md.append("through the full LoCoMo benchmark to get actual accuracy numbers.")

    output = "\n".join(md)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(f"Report: {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
