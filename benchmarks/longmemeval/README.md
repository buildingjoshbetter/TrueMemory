# LongMemEval Benchmark Results

Evaluation of TrueMemory Pro and 5 competitor systems on [LongMemEval](https://arxiv.org/abs/2410.10813), 500 questions across 6 question types testing long-term interactive memory.

Two dataset variants are evaluated:
- **Oracle** (`longmemeval_oracle.json`): each question includes only the relevant haystack sessions
- **Strict** (`longmemeval_s.json`): each question includes distractor sessions mixed with relevant ones (harder)

The paper reports the strict (_s) variant. The oracle variant is included for completeness.

## Results

### Strict variant (_s) — reported in the paper

| System | Accuracy | Correct | Answer model |
|--------|----------|---------|-------------|
| TM Pro (3-run mean) | **87.8%** | 439 | gpt-4.1-mini |
| RAG (ChromaDB) | 87.0% | 435 | gpt-4.1-mini |
| BM25 | 81.6% | 408 | gpt-4.1-mini |
| Engram | 82.2% | 411 | gpt-4.1-mini |
| Mem0 | 66.0% | 330 | gpt-4.1-mini |
| Supermemory | 15.8% | 79 | gpt-4.1-mini |

Individual TM Pro runs (_s): 86.6%, 88.6%, 88.2%

### Oracle variant

| System | Accuracy | Correct | Answer model |
|--------|----------|---------|-------------|
| TM Pro (3-run mean) | **92.0%** | 460 | gpt-4.1-mini |
| RAG (ChromaDB) | 91.8% | 459 | gpt-4.1-mini |
| BM25 | 90.0% | 450 | gpt-4.1-mini |
| Engram | 86.0% | 430 | gpt-4.1-mini |
| Mem0 | 64.0% | 320 | gpt-4.1-mini |
| Supermemory | 15.8% | 79 | gpt-4.1-mini |

Individual TM Pro runs (oracle): 92.2%, 91.8%, 92.0%

## Evaluation Config

| Parameter | Value |
|-----------|-------|
| Answer model | `openai/gpt-4.1-mini` via OpenRouter |
| Judge model | `openai/gpt-4o-mini` via OpenRouter |
| Judge runs per question | 3 (majority vote) |
| Answer temperature | 0 |
| Judge temperature | 0 |
| Retrieval top-k | 100 |
| Search method | `search_agentic` (HyDE + reranker) |

## How to Reproduce

```bash
modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-...

# Strict variant (reported in paper)
modal run --detach bench_truememory_pro_s.py --run-id 1

# Oracle variant
modal run --detach bench_truememory_pro.py --run-id 1

# Download results
modal volume get longmemeval-results / ./results --force
```

## File Structure

```
benchmarks/longmemeval/
  README.md                                          # This file
  bench_truememory_pro.py                             # Oracle variant benchmark script
  bench_truememory_pro_s.py                           # Strict variant benchmark script
  data/
    longmemeval_oracle.json                           # Oracle dataset (500 questions)
    longmemeval_s.json                                # Strict dataset (500 questions)
  results/
    truememory_pro_longmemeval_run1.json               # Oracle run 1 (92.2%)
    truememory_pro_longmemeval_run2.json               # Oracle run 2 (91.8%)
    truememory_pro_longmemeval_run3.json               # Oracle run 3 (92.0%)
    truememory_pro_longmemeval_s_run1.json             # Strict run 1 (86.6%)
    truememory_pro_longmemeval_s_run2.json             # Strict run 2 (88.6%)
    truememory_pro_longmemeval_s_run3.json             # Strict run 3 (88.2%)
    bm25_run1.json                                     # BM25 baseline
    engram_run1.json                                   # Engram baseline
    mem0_run1.json                                     # Mem0
    rag_run1.json                                      # RAG (ChromaDB)
    supermemory_run1.json                              # Supermemory
```
