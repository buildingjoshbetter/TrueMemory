# LoCoMo Benchmark Results

Evaluation of 8 memory systems on the [LoCoMo](https://arxiv.org/abs/2312.17487) benchmark: 10 multi-session conversations, 1540 questions across 4 categories (single-hop, multi-hop, temporal reasoning, open-domain).

## Leaderboard

<p align="center">
  <img src="../../assets/charts/leaderboard-bar.png" alt="LoCoMo 8-System Comparison" />
</p>

<p align="center">
  <img src="../../assets/charts/category-grouped-bars.png" alt="Performance by Category" />
</p>

## Evaluation Pipeline

Every system was evaluated with the same pipeline to ensure fair comparison:

1. **Retrieval** -- Each system ingests the conversation and retrieves context for each question using its own retrieval method.
2. **Answer generation** -- `openai/gpt-4.1-mini` via OpenRouter generates an answer from the retrieved context. `temperature=0`, `max_tokens=200`.
3. **LLM judging** -- `openai/gpt-4o-mini` via OpenRouter judges whether the generated answer matches the gold answer. `temperature=0`, `max_tokens=10`. Run 3 times per question; majority vote decides correctness.

The answer model, judge model, prompts, temperature, and voting scheme are identical across all 8 systems. Only the retrieval layer differs.

<p align="center">
  <img src="../../assets/charts/eval-pipeline.png" alt="Evaluation Pipeline" />
</p>

## How to Reproduce

### Prerequisites

1. A [Modal](https://modal.com) account (free tier works for most systems).
2. An [OpenRouter](https://openrouter.ai) API key for answer generation and judging.
3. For Supermemory: an additional Supermemory API key.

### Setup

```bash
# Create the Modal secret with your OpenRouter key
modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-...

# For Supermemory only
modal secret create supermemory-key SUPERMEMORY_API_KEY=sm_...
```

### Run Individual Systems

Each script in `scripts/` is self-contained with zero local imports:

```bash
# Full run (10 conversations, 1540 questions)
modal run --detach scripts/bench_bm25.py

# Smoke test (1 conversation, 5 questions)
modal run --detach scripts/bench_bm25.py --smoke

# Download results from Modal Volume
modal volume get locomo-results / ./results --force
```

See `scripts/README.md` for details on each script.

### Verify Scores

```bash
python3 scripts/verify_scores.py
```

Recomputes accuracy from the raw JSON result files with zero dependencies beyond Python stdlib.

## File Structure

```
benchmarks/locomo/
  README.md                  # This file
  BENCHMARK_RESULTS.md       # Full technical report (latency, cost, architecture)
  requirements.txt           # Python dependencies grouped by system
  data/
    locomo10.json            # LoCoMo dataset (10 conversations, 1540 questions)
  results/
    bm25_v2_run1.json        # BM25 results
    engram_v2_run1.json      # Engram results
    evermemos_v2_run1.json   # EverMemOS results
    mem0_v2_run1.json        # Mem0 results
    truememory_edge_v0.4.0.json    # TrueMemory Edge results (90.1% target — Phase 6)
    truememory_base_v0.4.0.json    # TrueMemory Base results (91.5% target — Phase 6)
    truememory_pro_v0.4.0.json     # TrueMemory Pro results (91.8% target — Phase 6)
    rag_v2_run1.json         # RAG (ChromaDB) results
    supermemory_v2_run1.json # Supermemory results
  scripts/
    README.md                # Script documentation
    bench_bm25.py            # BM25 keyword baseline
    bench_engram.py          # Engram memory system
    bench_evermemos.py       # EverMemOS (pre-built retrieval)
    bench_mem0.py            # Mem0 LLM-extracted memory
    bench_truememory_edge.py   # TrueMemory Edge tier (90.1%, CPU)
    bench_truememory_base.py   # TrueMemory Base tier (91.5%, T4 GPU recommended, HyDE off)
    bench_truememory_pro.py    # TrueMemory Pro tier (91.8%, T4 GPU, +HyDE)
    bench_rag.py             # ChromaDB RAG baseline
    bench_supermemory.py     # Supermemory cloud API
    modal_benchmark.py       # Development runner (see individual scripts instead)
    verify_scores.py         # Score verification tool
```

## Full Details

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for the complete technical report including per-category breakdowns, latency analysis, cost breakdown, hardware requirements, and retrieval architecture comparison.
