# Research Paper Parameter Alignment — Validated Baselines

These are the research-paper-derived parameters that MUST be verified against the current codebase.
Any drift from these values is a potential regression that may have caused benchmark score loss.

---

## Critical Parameters (Benchmark-Affecting)

### Search Pipeline

| Parameter | Research Value | File | Variable/Location | Notes |
|-----------|---------------|------|-------------------|-------|
| alpha_surprise | 0.2 | `engine.py:1784` | `_DEFAULT_ALPHA_SURPRISE` | L5 predictive coding boost coefficient. Boosts results with high surprise scores. α=0.2 is the empirical peak from Modal alpha sweep (2026-04-26). NOT related to RRF weights. |
| RRF k constant | 60 | `hybrid.py` | `k` param in `reciprocal_rank_fusion()` | Standard RRF constant |
| Vector dim | 256 | `vector_search.py` | Matryoshka truncation | Both Edge (Model2Vec) and Base/Pro (Qwen3) use 256d |
| Edge embed model | `minishlab/potion-base-8M` | `vector_search.py` | Model loading | Model2Vec, 256d output |
| Base/Pro embed model | `Qwen/Qwen3-Embedding-0.6B` | `vector_search.py` | Model loading | sentence-transformers, 256d Matryoshka |
| Edge reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `reranker.py` | Model loading | Lightweight, 22M params |
| Base/Pro reranker | `Alibaba-NLP/gte-reranker-modernbert-base` | `reranker.py` | Model loading | Heavier, better quality |
| Deep search reranker | `BAAI/bge-reranker-v2-m3` | `reranker.py` or `engine.py` | search_deep() path | 568M params, maximum recall |
| Search top_k default | 10 | `engine.py` | `search()` | Default result count |
| Deep search top_k | 500 | `engine.py` | `search_deep()` | Candidate pool for deep search |
| Multi-round retrieval | 2–3 rounds | `engine.py` | `search_deep()` | Multiple retrieval passes |

### Encoding Gate

| Parameter | Research Value | File | Variable/Location | Notes |
|-----------|---------------|------|-------------------|-------|
| Gate threshold | 0.30 | `encoding_gate.py:152` | threshold | Combined gate score threshold. Memories below this are filtered out. |
| w_novelty | 0.25 | `encoding_gate.py:166` | weight | Novelty component weight in gate score |
| w_salience | 0.20 | `encoding_gate.py:168` | weight | Salience component weight in gate score |
| w_pe | 0.30 | `encoding_gate.py:170` | weight | Predictive encoding component weight |
| Novelty signal | gzip NCD compression | `encoding_gate.py:255` | novelty computation | Normalized Compression Distance via gzip (AUC 0.788). NOT cosine similarity. |
| Salience spotlight min | 0.05 | `engine.py:1423` | min_sal | Minimum salience for spotlight search mode |
| Salience diffuse min | 0.02 | `engine.py:1421` | min_sal | Minimum salience for diffuse search mode |

### Benchmark Baselines (3-run mean)

| Benchmark | Score | Std Dev | Config | Date Validated |
|-----------|-------|---------|--------|----------------|
| LoCoMo | 93.20% | ±0.35 | alpha=0.2, threshold 0.05/0.02 | 2026-05-03 |
| LoCoMo (alt) | 93.07% | ±0.15 | alpha=0.3, threshold 0.05/0.02 | 2026-04-30 |
| LongMemEval | 87.8% | — | Pro tier | 2026-05-05 |
| BEAM-1M | 76.6% | — | Pro tier | 2026-05-05 |
| Single-run peak | 93.6% | — | scale=0.3 | — |

### Answer Model
| Parameter | Value | Notes |
|-----------|-------|-------|
| Answer model | gpt-4.1-mini | Used for all benchmarks |
| Temperature | 0 | Deterministic answers |

---

## Secondary Parameters

### Personality System (L0)

| Parameter | Research Value | File | Notes |
|-----------|---------------|------|-------|
| Dunbar hierarchy | 5/15/50/150 | `personality.py` | Dunbar social brain numbers |
| Style vector dim | Varies | `personality_style_vec.py` | Computed from writing patterns |
| Entity resolution threshold | — | `personality.py` | Alias matching confidence |

### Salience (L3)

| Parameter | Research Value | File | Notes |
|-----------|---------------|------|-------|
| Salience weight | Research-defined | `salience.py` | Weight in final scoring |
| Entity detection model | — | `salience.py` | NER or pattern-based |

### Consolidation (L4/L5)

| Parameter | Research Value | File | Notes |
|-----------|---------------|------|-------|
| Contradiction threshold | — | `consolidation.py` | When to flag contradiction |
| Timeline resolution | — | `consolidation.py` | Granularity of timeline |
| Entity sheets | Disabled? | `consolidation.py` | test_l4_entity_sheets_disabled exists |

### Predictive Coding

| Parameter | Research Value | File | Notes |
|-----------|---------------|------|-------|
| Surprise threshold | — | `predictive.py` | When information is "surprising" |
| Noise floor | — | `predictive.py` | Below this = noise |

---

## Validation Commands

### Quick Parameter Check
```bash
cd /Users/j/Desktop/TrueMemory

# Check alpha value
grep -n "alpha" truememory/hybrid.py | head -20

# Check RRF k value
grep -n "k=" truememory/hybrid.py | head -20

# Check embedding dim
grep -n "256\|dimension\|dim" truememory/vector_search.py | head -20

# Check model names
grep -n "potion-base\|Qwen3\|ms-marco\|modernbert\|bge-reranker" truememory/*.py | head -20

# Check encoding gate thresholds
grep -n "0.05\|0.02\|threshold" truememory/ingest/encoding_gate.py | head -20

# Check novelty uses cosine (NOT RRF)
grep -n "cosine\|rrf\|novelty" truememory/ingest/encoding_gate.py | head -20

# Check deep search top_k
grep -n "500\|top_k" truememory/engine.py | head -20
```

### Full LoCoMo Regression Test
```bash
cd /Users/j/Desktop/TrueMemory
# Requires benchmark data at ~/Desktop/TrueMemory_benchmarks/data/locomo10.json
# Run 3 times, compute mean and std dev
for i in 1 2 3; do
  .venv/bin/python benchmarks/run_locomo.py --config main 2>&1 | tail -5
done
```

---

## Known Regressions to Watch

1. **Novelty signal using RRF instead of cosine** — Primary encoding gate bug from prior audit (memory #1276)
2. **Alpha drift** — Research says 0.3, but best score was at 0.2 — need to verify which is in code
3. **Threshold overrides** — Category-specific thresholds may override the validated 0.05/0.02
4. **Reranker model mismatch** — Tier-specific models may have been swapped during recent refactors
5. **Entity sheets disabled** — Test file suggests L4 entity sheets are disabled; was this intentional?
6. **PE decoupling** — Predictive encoding may have been re-coupled during recent changes
7. **Score normalization** — Recent performance PRs may have changed normalization behavior
8. **SQLite tuning** — PR #223 changed SQLite pragmas; may affect benchmark reproducibility
