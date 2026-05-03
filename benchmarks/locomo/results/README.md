# LoCoMo Benchmark Results

Raw result JSON files from all 8 systems evaluated on the LoCoMo benchmark. Each file contains the full per-question detail arrays: generated answers, gold answers, judge votes, latency, and category labels.

## Competitor Systems (single run each)

| File | System | Score |
|------|--------|-------|
| `bm25_v2_run1.json` | BM25 keyword baseline | 80.5% |
| `engram_v2_run1.json` | Engram memory | 84.5% |
| `evermemos_v2_run1.json` | EverMemOS (pre-computed retrieval) | 94.5% |
| `mem0_v2_run1.json` | Mem0 LLM-extracted memory | 61.4% |
| `rag_v2_run1.json` | RAG / ChromaDB | 86.2% |
| `supermemory_v2_run1.json` | Supermemory cloud API | 65.4% |

## TrueMemory v0.6.0 (3-run validated)

| File | Tier | Score |
|------|------|-------|
| `truememory_edge_v060_run1.json` | Edge | 89.9% |
| `truememory_edge_v060_run2.json` | Edge | 89.5% |
| `truememory_edge_v060_run3.json` | Edge | 89.5% |
| `truememory_base_v060_run2.json` | Base | 92.1% |
| `truememory_base_v060_run3.json` | Base | 92.2% |
| `truememory_pro_v060_run1.json` | Pro | 92.8% |
| `truememory_pro_v060_run2.json` | Pro | 93.1% |
| `truememory_pro_v060_run3.json` | Pro | 93.1% |

### 3-run means

| Tier | Mean | ±std |
|------|------|------|
| Edge | 89.6% | ±0.19 |
| Base | 92.2% | ±0.07 (2 runs; run 1 pending) |
| Pro | 93.0% | ±0.14 |

## Verification

Recompute all scores from scratch:

```bash
python3 ../scripts/verify_scores.py
```
