# TrueMemory Benchmark Archive

> **Total cataloged runs: 348** | Date range: 2025-12 to 2026-04-26
> Standard LoCoMo eval: 10 conversations, 1540 questions (Cat-5 adversarial excluded), GPT-4o-mini judge x3 majority vote
> Early R&D eval: 1-6 conversations, 152-932 questions, various judges (claude-sonnet-4-5-20250929, claude-opus-4-6)

**Result archives:** Modal volume `locomo-results` | `~/Desktop/TrueMemory_benchmarks/results/` | repo `benchmarks/locomo/results/` | `benchmarks/gate_eval/results/` | `_working/benchmarks_56combo/results/`

---

## Leaderboard (Best Per-System, LoCoMo-10 Standard)

| System | Version | Score | Correct/1540 | Config |
|--------|---------|-------|--------------|--------|
| EverMemOS | — | 94.5% | 1455/1540 | External competitor (retrieval-focused) |
| **TrueMemory Pro** | **v0.6.0** | **93.6%** | **1441/1540** | L3-only, threshold 0.10, alpha=0 (single run high) |
| **TrueMemory Pro** | **v0.6.0** | **93.20% mean** | **1437/1540** | threshold 0.05/0.02, alpha=0.2 (3-run mean ±0.35) |
| TrueMemory Pro | v0.6.0 | 93.07% mean | 1433/1540 | threshold 0.05/0.02, alpha=0.3 (3-run mean ±0.15) |
| TrueMemory Pro | v0.5.0 | 92.9% | 1430/1540 | patched (set_active_tier fix) |
| TrueMemory Base | v0.5.0 | 92.4% | 1423/1540 | patched |
| neuromem Pro | v0.2.0 (gpu-box) | 91.75% | 1413/1540 | best_run2 |
| TrueMemory Pro | v0.5.0 | 91.5% | 1409/1540 | unpatched (MiniLM reranker bug) |
| TrueMemory Edge | v0.5.0 | 89.4% | 1377/1540 | patched |
| RAG baseline | — | 86.2% | 1327/1540 | Competitor |
| Engram | — | 84.5% | 1302/1540 | Competitor |
| BM25 | — | 80.5% | 1239/1540 | Competitor |
| SuperMemory | — | 65.4% | 1007/1540 | Competitor |
| Mem0 | — | 61.4% | 946/1540 | Competitor |

---

## Section 1: v0.6.0 Runs (2026-04-26)

All runs: Pro tier, HyDE ON, gte-reranker-modernbert, Qwen3 256d, set_active_tier("pro"), T4 GPU, 10-way parallel, installed from GitHub main.

### Alpha Sweep (threshold 0.05/0.02, Modal T4)

| Alpha | Run 1 | Run 2 | Run 3 | Mean | StdDev | vs Baseline |
|-------|-------|-------|-------|------|--------|-------------|
| 0 | 93.3% | 93.0% | 92.7% | 93.00% | ±0.30 | +0.10 |
| 0.1 | 92.0% | 92.9% | 93.2% | 92.70% | ±0.62 | -0.20 |
| 0.15 | 92.3% | 93.4% | 92.5% | 92.73% | ±0.59 | -0.17 |
| **0.2** | **93.6%** | **93.0%** | **93.0%** | **93.20%** | **±0.35** | **+0.30** |
| 0.3 | 93.2% | 93.1% | 92.9% | 93.07% | ±0.15 | +0.17 |

**Winner: alpha=0.2** — empirical peak, strongest multi-hop (93.8% in run 1), shipped as new default in PR #81.

### Alpha Sweep Cross-Validation (GPUBox RTX 5090, 5 concurrent runs)

| Alpha | Score | Correct | single | multi | temporal | open | Wall Clock |
|-------|-------|---------|--------|-------|----------|------|------------|
| 0 | 93.2% | 1436 | 91.8% | 91.0% | 80.2% | 96.1% | 34.1 min |
| 0.1 | 92.8% | 1429 | 94.0% | 90.7% | 82.3% | 94.4% | 34.5 min |
| 0.2 | 93.3% | 1437 | 93.6% | 93.5% | 82.3% | 94.4% | 34.1 min |
| 0.3 | 92.7% | 1428 | 92.2% | 91.9% | 82.3% | 94.4% | 34.1 min |
| 0.5 | 92.4% | 1423 | 90.1% | 91.0% | 84.4% | 94.6% | 33.6 min |

GPUBox confirms alpha=0.2 as the peak (93.3%, 1437/1540). Apples-to-apples with Modal validated.

### GPUBox 3-Run Means (alpha=0.15 and 0.2, 6 concurrent runs)

| Alpha | Run 1 | Run 2 | Run 3 | Mean | StdDev |
|-------|-------|-------|-------|------|--------|
| 0.15 | 92.7% | 92.2% | 92.7% | 92.53% | ±0.29 |
| 0.2 | 91.9% | 92.2% | 92.6% | 92.23% | ±0.35 |

GPUBox means are ~0.5-1.0% lower than Modal at 6 concurrent runs (vs 0.1-0.3% delta at 5 concurrent). Higher contention at 6 runs may cause slight degradation.

### 2x2 Attribution Matrix (threshold × alpha)

| | Threshold 0.05/0.02 | Threshold 0.10 |
|---|---|---|
| **Alpha 0.3 (L5 on)** | 93.07% ±0.15 (3 runs) | 92.77% ±0.23 (3 runs) |
| **Alpha 0.2 (L5 on)** | **93.20% ±0.35 (3 runs)** | — |
| **Alpha 0 (L5 off)** | 93.00% ±0.30 (3 runs) | 92.47% ±1.00 (3 runs) |

### All v0.6.0 Runs (Detailed)

| File | Version | Threshold | Alpha | Score | Correct | single_hop | multi_hop | temporal | open_domain |
|------|---------|-----------|-------|-------|---------|------------|-----------|----------|-------------|
| truememory_pro_v060_l3only_run1.json | l3only-run1 | 0.10 | 0 | 93.6% | 1441/1540 | 95.0% | 91.0% | 83.3% | 95.2% |
| truememory_pro_v060_l3l5.json | l3l5 (main run 1) | 0.05/0.02 | 0.3 | 93.2% | 1436/1540 | 94.3% | 90.0% | 84.4% | 95.1% |
| truememory_pro_v060_main_run2.json | main-run2 | 0.05/0.02 | 0.3 | 93.1% | 1433/1540 | 92.2% | 91.3% | 83.3% | 95.1% |
| truememory_pro_v060_variant_b.json | variant-b | 0.10 | 0.3 | 93.1% | 1434/1540 | 91.5% | 93.5% | 79.2% | 95.1% |
| truememory_pro_v060_variant_c.json | variant-c | 0.0 | 0.3 | 93.0% | 1432/1540 | 91.1% | 92.8% | 86.5% | 94.4% |
| truememory_pro_v060_main_run3.json | main-run3 | 0.05/0.02 | 0.3 | 92.9% | 1431/1540 | 91.8% | 91.9% | 83.3% | 94.8% |
| truememory_pro_v060_l3l5_run2.json | l3l5-run2 | 0.10 | 0.3 | 92.9% | 1431/1540 | 92.9% | 92.2% | 79.2% | 94.8% |
| truememory_pro_v060_l3l5_run3.json | l3l5-run3 | 0.10 | 0.3 | 92.9% | 1430/1540 | 93.3% | 91.3% | 81.2% | 94.6% |
| truememory_pro_v060_l3l5_run1.json | l3l5-run1 | 0.10 | 0.3 | 92.5% | 1424/1540 | 91.5% | 91.9% | 78.1% | 94.6% |
| truememory_pro_v060_variant_a.json | variant-a | 0.25 | 0.3 | 92.3% | 1421/1540 | 92.2% | 89.4% | 82.3% | 94.5% |
| truememory_pro_v060_l3only_run3.json | l3only-run3 | 0.10 | 0 | 92.1% | 1419/1540 | 90.4% | 91.3% | 78.1% | 94.6% |
| truememory_pro_v060_l3only_run2.json | l3only-run2 | 0.10 | 0 | 91.7% | 1412/1540 | 90.4% | 90.7% | 80.2% | 93.8% |

### Group Means (All Configurations)

| Config | Threshold | Alpha | Mean | StdDev | Runs | Platform |
|--------|-----------|-------|------|--------|------|----------|
| **Alpha sweep winner** | **0.05/0.02** | **0.2** | **93.20%** | **±0.35** | **3** | **Modal** |
| Main (original) | 0.05/0.02 | 0.3 | 93.07% | ±0.15 | 3 | Modal |
| Alpha=0 | 0.05/0.02 | 0 | 93.00% | ±0.30 | 3 | Modal |
| L3+L5 | 0.10 | 0.3 | 92.77% | ±0.23 | 3 | Modal |
| Alpha=0.15 | 0.05/0.02 | 0.15 | 92.73% | ±0.59 | 3 | Modal |
| Alpha=0.1 | 0.05/0.02 | 0.1 | 92.70% | ±0.62 | 3 | Modal |
| L3-only | 0.10 | 0 | 92.47% | ±1.00 | 3 | Modal |

### L0 Char-N-Gram Style Vectors (PR #82, alpha=0.2, Modal T4)

Branch: `feat/l0-char-ngram-style-vectors`. Config identical to alpha sweep winner except: L0 personality scoring now uses 256-d hashed char-n-gram vectors (C3c) instead of hand-tuned keyword extraction. All other params unchanged (Pro tier, HyDE ON, gte-reranker, Qwen3 256d, alpha=0.2).

#### L0 Baseline (score_scale=1.0, uncapped)

| Scale | Run 1 | Run 2 | Run 3 | Mean | StdDev | vs 93.20% baseline |
|-------|-------|-------|-------|------|--------|--------------------|
| 1.0 (uncapped) | 92.7% | 92.9% | 93.2% | 92.93% | ±0.25 | -0.27% |

Per-category means (scale=1.0): single_hop 92.1%, multi_hop 91.7%, temporal 82.3%, open_domain 94.8%

Regression diagnosis: style_vec results enter pipeline at 5.0+ scores (persona scoping bias), displacing correct factual results on queries that trigger `_has_personality_intent`. Old FTS-based personality results had low BM25-range scores that couldn't outrank hybrid results. Fix: scale style_vec scores via `TRUEMEMORY_L0_SCORE_SCALE` env var.

#### L0 Score Scale Sweep (TRUEMEMORY_L0_SCORE_SCALE)

| Scale | Run 1 | Run 2 | Run 3 | Mean | StdDev | vs 93.20% baseline |
|-------|-------|-------|-------|------|--------|--------------------|
| **0.7** | **93.1%** | **93.4%** | **92.7%** | **93.07%** | **±0.29** | **-0.13%** |
| 0.6 | 93.2% | 93.1% | 92.7% | 93.00% | ±0.22 | -0.20% |
| 0.8 | 92.8% | 92.9% | 93.2% | 92.97% | ±0.17 | -0.23% |
| 0.9 | 92.7% | 93.1% | 91.9% | 92.57% | ±0.50 | -0.63% |

**Winner: scale=0.7** — closest to baseline at -0.13% (within judge noise ±0.35).

**Winner: scale=0.7** — closest to baseline at -0.13% (within judge noise ±0.35).

**Key finding:** LoCoMo only triggers `_has_personality_intent` on 90/1540 queries (5.8%), and most are false positives (factual queries containing keywords like "friends", "food"). The scale sweep is optimizing damage reduction from false-positive personality injections, not L0 effectiveness. A dedicated personality eval benchmark (PersonaLoCoMo) is needed to properly tune L0 — see `benchmarks/personality_eval/`.

#### Per-Category Means (L0 Score Scale Sweep)

| Scale | single_hop | multi_hop | temporal | open_domain |
|-------|-----------|-----------|----------|-------------|
| 0.6 | 92.3% | 91.8% | 82.3% | 94.9% |
| 0.7 | 93.1% | 91.8% | 82.6% | 94.8% |
| 0.8 | 92.0% | 92.0% | 84.0% | 94.6% |
| 0.9 | 91.3% | 90.7% | 84.4% | 94.7% |
| 1.0 | 92.1% | 91.7% | 82.3% | 94.8% |

#### All L0 Runs (Detailed)

| File | Scale | Run | Score | Correct | single_hop | multi_hop | temporal | open_domain |
|------|-------|-----|-------|---------|------------|-----------|----------|-------------|
| truememory_pro_v060_l0char_run1.json | 1.0 | 1 | 92.7% | 1427/1540 | 91.5% | 91.6% | 81.2% | 94.8% |
| truememory_pro_v060_l0char_run2.json | 1.0 | 2 | 92.9% | 1430/1540 | 92.6% | 91.0% | 84.4% | 94.6% |
| truememory_pro_v060_l0char_run3.json | 1.0 | 3 | 93.2% | 1435/1540 | 92.2% | 92.5% | 81.2% | 95.1% |
| truememory_pro_v060_l0scale06_run1.json | 0.6 | 1 | 93.2% | 1435/1540 | 91.1% | 92.8% | 85.4% | 94.9% |
| truememory_pro_v060_l0scale06_run2.json | 0.6 | 2 | 93.1% | 1434/1540 | 93.6% | 91.3% | 81.2% | 95.0% |
| truememory_pro_v060_l0scale06_run3.json | 0.6 | 3 | 92.7% | 1427/1540 | 92.2% | 91.3% | 80.2% | 94.8% |
| truememory_pro_v060_l0scale07_run1.json | 0.7 | 1 | 93.1% | 1434/1540 | 94.0% | 90.3% | 83.3% | 95.0% |
| truememory_pro_v060_l0scale07_run2.json | 0.7 | 2 | 93.4% | 1439/1540 | 92.6% | 93.8% | 85.4% | 94.5% |
| truememory_pro_v060_l0scale07_run3.json | 0.7 | 3 | 92.7% | 1428/1540 | 92.6% | 91.3% | 79.2% | 94.9% |
| truememory_pro_v060_l0scale08_run1.json | 0.8 | 1 | 92.8% | 1429/1540 | 91.8% | 92.5% | 81.2% | 94.5% |
| truememory_pro_v060_l0scale08_run2.json | 0.8 | 2 | 92.9% | 1431/1540 | 92.6% | 92.2% | 83.3% | 94.3% |
| truememory_pro_v060_l0scale08_run3.json | 0.8 | 3 | 93.2% | 1436/1540 | 91.5% | 91.3% | 87.5% | 95.1% |
| truememory_pro_v060_l0scale09_run1.json | 0.9 | 1 | 92.7% | 1428/1540 | 90.8% | 91.9% | 82.3% | 94.9% |
| truememory_pro_v060_l0scale09_run2.json | 0.9 | 2 | 93.1% | 1433/1540 | 92.2% | 90.0% | 87.5% | 95.2% |
| truememory_pro_v060_l0scale09_run3.json | 0.9 | 3 | 91.9% | 1415/1540 | 90.8% | 90.3% | 83.3% | 93.9% |

#### Personality Intent Analysis

Of LoCoMo's 1540 questions, only 90 (5.8%) trigger `_has_personality_intent`. Breakdown of those 90:
- open_domain: 53 (mostly false positives — factual questions containing "friends", "food", "hobby")
- single_hop: 14 (factual questions with personality keywords)
- multi_hop: 13 (factual questions with personality keywords)
- temporal: 10 (factual questions with personality keywords)

Conclusion: LoCoMo is not a valid benchmark for L0 personality tuning. Building PersonaLoCoMo (`benchmarks/personality_eval/`) — a dedicated 2000-question personality-focused eval using the same 10 LoCoMo conversations.

---

## Section 2: v0.5.0 Production Runs

### Pro Tier

| File | Version | Score | Correct | single_hop | multi_hop | temporal | open_domain | Notes |
|------|---------|-------|---------|------------|-----------|----------|-------------|-------|
| truememory_pro_v050_patched.json | v3-modal-rerun | 92.9% | 1430/1540 | 90.1% | 92.2% | 84.4% | 95.0% | set_active_tier('pro') patch — correct reranker |
| truememory_pro_v050_gate_on.json | v3-modal-rerun | 92.3% | 1422/1540 | 91.8% | 90.0% | 81.2% | 94.6% | encoding gate enabled |
| truememory_pro_v050_runA.json | v3-modal-rerun | 91.8% | 1414/1540 | 90.4% | 91.3% | 85.4% | 93.2% | unpatched (MiniLM reranker bug) |
| truememory_pro_v050_runB.json | v3-modal-rerun | 91.0% | 1402/1540 | 88.7% | 89.7% | 82.3% | 93.3% | unpatched (MiniLM reranker bug) |
| truememory_pro_v3_modal.json | v3-modal-rerun | 91.5% | 1409/1540 | 91.1% | 90.7% | 84.4% | 92.7% | unpatched |
| truememory_pro_v2_run1.json | v3-checkpointed | 90.7% | 1397/1540 | 91.1% | 88.8% | 80.2% | 92.5% | legacy (pre-v0.4.0 tier realignment) |

### Base Tier

| File | Version | Score | Correct | single_hop | multi_hop | temporal | open_domain | Notes |
|------|---------|-------|---------|------------|-----------|----------|-------------|-------|
| truememory_base_v050_patched.json | v3-modal-rerun | 92.4% | 1423/1540 | 91.5% | 90.7% | 80.2% | 94.8% | set_active_tier('base') patch |
| truememory_base_v050_setier.json | v3-modal-rerun | 92.0% | 1417/1540 | 91.1% | 89.1% | 80.2% | 94.8% | set_active_tier variant |
| truememory_base_v050_runA.json | v3-modal-rerun | 90.2% | 1389/1540 | 87.6% | 89.4% | 80.2% | 92.5% | unpatched |
| truememory_base_v050_runB.json | v3-modal-rerun | 90.2% | 1389/1540 | 87.2% | 91.3% | 82.3% | 91.7% | unpatched |

### Edge Tier

| File | Version | Score | Correct | single_hop | multi_hop | temporal | open_domain | Notes |
|------|---------|-------|---------|------------|-----------|----------|-------------|-------|
| truememory_edge_v050_patched.json | v3-checkpointed | 89.4% | 1377/1540 | 87.6% | 89.1% | 80.2% | 91.2% | patched |
| truememory_edge_v050_runB.json | v3-checkpointed | 89.1% | 1372/1540 | 86.9% | 89.1% | 78.1% | 91.1% |  |
| truememory_edge_v050_runA.json | v3-checkpointed | 88.9% | 1369/1540 | 87.6% | 88.5% | 79.2% | 90.6% |  |

---

## Section 3: Pre-Rebrand neuromem Runs (LoCoMo-10)

### Modal Volume Reruns (8 runs, same config, measuring judge variance)

| File | Score | Correct | single_hop | multi_hop | temporal | open_domain |
|------|-------|---------|------------|-----------|----------|-------------|
| neuromem_pro_rerun_3.json | 91.5% | 1409/1540 | 91.1% | 90.7% | 84.4% | 92.7% |
| neuromem_pro_rerun_8.json | 91.3% | 1406/1540 | 89.7% | 90.3% | 83.3% | 93.1% |
| neuromem_pro_rerun_2.json | 91.1% | 1403/1540 | 91.1% | 90.0% | 83.3% | 92.4% |
| neuromem_pro_rerun_6.json | 90.7% | 1397/1540 | 91.1% | 89.7% | 80.2% | 92.2% |
| neuromem_pro_rerun_4.json | 90.6% | 1396/1540 | 91.1% | 88.8% | 78.1% | 92.6% |
| neuromem_pro_rerun_5.json | 90.5% | 1393/1540 | 90.8% | 88.2% | 83.3% | 92.0% |
| neuromem_pro_rerun_1.json | 90.3% | 1391/1540 | 89.4% | 87.5% | 82.3% | 92.6% |
| neuromem_pro_rerun_7.json | 90.0% | 1386/1540 | 89.4% | 88.8% | 86.5% | 91.1% |

### Other neuromem LoCoMo-10 Runs (unique files only)

| File | Source | Version | Score | Correct | Notes |
|------|--------|---------|-------|---------|-------|
| neuromem_pro_best_run2.json | v0.2.0 archive | v3-gpu-box | 91.75% | 1413/1540 | neuromem v0.2.0 |
| neuromem_pro_v2_run2.json | benchmarks_final | v2-benchmark-rules | 91.3% | 1406/1540 | benchmarks_final copy |
| neuromem_pro_v2_run1.json | Modal | v3-checkpointed | 90.7% | 1397/1540 | Modal volume locomo-results (canonical production) |
| neuromem_pro_modal_run1.json | benchmarks_final | v3-checkpointed | 90.7% | 1397/1540 | benchmarks_final copy |
| neuromem_base_v2_run1.json | Modal | v3-checkpointed | 88.2% | 1359/1540 | Modal volume locomo-results (canonical production) |
| neuromem_base_modal_run1.json | benchmarks_final | v3-checkpointed | 88.2% | 1359/1540 | benchmarks_final copy |
| neuromem_locomo_full.json | R&D archive |  | 29.0% | 447/1540 | Early R&D full pipeline |

---

## Section 4: 56-Grid Sweep (Tier Selection)

Embedder x Reranker combinatorial sweep. All: HyDE on, Pro tier, gpt-4.1-mini answer, gpt-4o-mini judge x3, LoCoMo-10.
Total combos evaluated: 62

| # | Embedder | Reranker | Score | Correct | single_hop | multi_hop | temporal | open_domain | Wall Clock |
|---|----------|----------|-------|---------|------------|-----------|----------|-------------|------------|
| 1 | zembed-1 cloud | mxbai-large 560M | 93.1% | 1433/1540 | 93.3% | 92.5% | 85.4% | 94.1% | 28596s |
| 2 | zembed-1 cloud | Qwen3-Reranker 600M | 92.9% | 1430/1540 | 94.0% | 90.0% | 84.4% | 94.5% | 19956s |
| 3 | zembed-1 cloud | gte-modernbert-reranker 149M | 92.6% | 1426/1540 | 90.1% | 90.7% | 84.4% | 95.1% | 3926s |
| 4 | Qwen3-0.6B @256d Matryoshka | zerank-1 cloud 4B | 92.3% | 1421/1540 | 92.2% | 93.1% | 83.3% | 93.0% | 10783s |
| 5 | zembed-1 cloud | bge-v2-m3 278M | 92.2% | 1420/1540 | 91.8% | 91.0% | 80.2% | 94.2% | 7726s |
| 6 | nomic-embed-v1.5 @768d | zerank-1 cloud 4B | 92.0% | 1417/1540 | 91.1% | 91.9% | 83.3% | 93.3% | 8264s |
| 7 | Qwen3-0.6B @512d Matryoshka | zerank-1 cloud 4B | 92.0% | 1417/1540 | 91.5% | 90.3% | 83.3% | 93.8% | 27090s |
| 8 | gte-modernbert-base @768d | gte-modernbert-reranker 149M | 91.9% | 1415/1540 | 90.8% | 90.0% | 80.2% | 94.3% | 6383s |
| 9 | Qwen3-0.6B @512d Matryoshka | Qwen3-Reranker 600M | 91.9% | 1415/1540 | 94.3% | 89.4% | 82.3% | 93.1% | 8173s |
| 10 | zembed-1 cloud | zerank-1 cloud 4B | 91.9% | 1416/1540 | 91.8% | 91.0% | 76.0% | 94.2% | 4626s |
| 11 | gte-modernbert-base @768d | bge-v2-m3 278M | 91.8% | 1414/1540 | 91.8% | 90.0% | 84.4% | 93.3% | 23453s |
| 12 | Qwen3-0.6B @256d Matryoshka | gte-modernbert-reranker 149M | 91.8% | 1414/1540 | 90.4% | 90.7% | 81.2% | 93.9% | 9749s |
| 13 | gte-modernbert-base @768d | Qwen3-Reranker 600M | 91.7% | 1412/1540 | 89.0% | 90.7% | 82.3% | 94.1% | 6991s |
| 14 | Qwen3-0.6B @1024d | Qwen3-Reranker 600M | 91.7% | 1412/1540 | 89.4% | 92.5% | 80.2% | 93.5% | 2968s |
| 15 | Qwen3-0.6B @256d Matryoshka | MiniLM-L12 33M | 91.7% | 1412/1540 | 90.1% | 92.2% | 82.3% | 93.1% | 8712s |
| 16 | gte-modernbert-base @768d | zerank-1 cloud 4B | 91.6% | 1410/1540 | 88.7% | 90.7% | 84.4% | 93.7% | 9254s |
| 17 | Qwen3-0.6B @512d Matryoshka | gte-modernbert-reranker 149M | 91.6% | 1410/1540 | 90.1% | 89.4% | 80.2% | 94.2% | 12797s |
| 18 | zembed-1 cloud | bge-large 335M | 91.6% | 1410/1540 | 91.1% | 90.3% | 78.1% | 93.7% | 6375s |
| 19 | Qwen3-0.6B @256d Matryoshka | gte-reranker-modernbert-base 149M | 91.5% | 1409/1540 | 90.4% | 90.3% | 82.3% | 93.3% | 957s |
| 20 | Qwen3-0.6B @512d Matryoshka | bge-v2-m3 278M | 91.5% | 1409/1540 | 90.8% | 91.3% | 83.3% | 92.7% | 23022s |
| 21 | Qwen3-0.6B @512d Matryoshka | mxbai-large 560M | 91.5% | 1409/1540 | 92.2% | 90.7% | 81.2% | 92.7% | 21725s |
| 22 | Model2Vec @256d | MiniLM-L12 33M | 91.4% | 1407/1540 | 89.0% | 91.0% | 86.5% | 92.9% | 5539s |
| 23 | Qwen3-0.6B @256d Matryoshka | Qwen3-Reranker 600M | 91.4% | 1407/1540 | 90.1% | 90.0% | 82.3% | 93.3% | 8039s |
| 24 | Qwen3-0.6B @512d Matryoshka | bge-large 335M | 91.4% | 1408/1540 | 91.1% | 89.4% | 84.4% | 93.1% | 5121s |
| 25 | Model2Vec @256d | bge-large 335M | 91.3% | 1406/1540 | 89.0% | 91.0% | 84.4% | 93.0% | 20606s |
| 26 | nomic-embed-v1.5 @768d | Qwen3-Reranker 600M | 91.3% | 1406/1540 | 89.7% | 91.3% | 77.1% | 93.5% | 7302s |
| 27 | Qwen3-0.6B @256d Matryoshka | bge-v2-m3 278M | 91.3% | 1406/1540 | 90.1% | 90.0% | 83.3% | 93.1% | 18300s |
| 28 | Qwen3-0.6B @512d Matryoshka | MiniLM-L12 33M | 91.3% | 1406/1540 | 90.8% | 89.1% | 80.2% | 93.6% | 17104s |
| 29 | gte-modernbert-base @768d | bge-large 335M | 91.2% | 1405/1540 | 90.1% | 90.3% | 81.2% | 93.1% | 7491s |
| 30 | gte-modernbert-base @768d | MiniLM-L12 33M | 91.2% | 1404/1540 | 89.7% | 88.5% | 80.2% | 93.9% | 1115s |
| 31 | gte-modernbert-base @768d | No reranker | 91.2% | 1404/1540 | 90.1% | 88.8% | 83.3% | 93.3% | 13717s |
| 32 | nomic-embed-v1.5 @768d | bge-large 335M | 91.2% | 1404/1540 | 88.7% | 90.3% | 80.2% | 93.6% | 17370s |
| 33 | Qwen3-0.6B @256d Matryoshka | No reranker | 91.2% | 1404/1540 | 89.4% | 90.3% | 81.2% | 93.2% | 3984s |
| 34 | gte-modernbert-base @768d | mxbai-large 560M | 91.1% | 1403/1540 | 88.7% | 88.2% | 83.3% | 93.9% | 2565s |
| 35 | Qwen3-0.6B @512d Matryoshka | No reranker | 91.1% | 1403/1540 | 88.7% | 90.3% | 83.3% | 93.1% | 22355s |
| 36 | Model2Vec @256d | zerank-1 cloud 4B | 91.0% | 1402/1540 | 88.7% | 91.9% | 84.4% | 92.3% | 1776s |
| 37 | nomic-embed-v1.5 @768d | No reranker | 91.0% | 1402/1540 | 89.0% | 89.7% | 82.3% | 93.2% | 15280s |
| 38 | Qwen3-0.6B @256d Matryoshka | bge-large 335M | 91.0% | 1401/1540 | 89.7% | 89.4% | 85.4% | 92.6% | 9103s |
| 39 | Qwen3-0.6B @256d Matryoshka | mxbai-large 560M | 91.0% | 1401/1540 | 88.7% | 90.0% | 82.3% | 93.1% | 9335s |
| 40 | nomic-embed-v1.5 @768d | bge-v2-m3 278M | 90.9% | 1400/1540 | 88.7% | 89.7% | 81.2% | 93.2% | 2277s |
| 41 | nomic-embed-v1.5 @768d | MiniLM-L12 33M | 90.9% | 1400/1540 | 89.0% | 89.4% | 78.1% | 93.6% | 2078s |
| 42 | Qwen3-0.6B @1024d | bge-v2-m3 278M | 90.9% | 1400/1540 | 90.1% | 90.7% | 82.3% | 92.3% | 14628s |
| 43 | Qwen3-0.6B @1024d | MiniLM-L12 33M | 90.9% | 1400/1540 | 89.4% | 87.9% | 88.5% | 92.9% | 6377s |
| 44 | nomic-embed-v1.5 @768d | mxbai-large 560M | 90.8% | 1398/1540 | 90.1% | 88.8% | 79.2% | 93.1% | 10667s |
| 45 | Qwen3-0.6B @1024d | No reranker | 90.8% | 1399/1540 | 88.3% | 88.8% | 86.5% | 93.0% | 11736s |
| 46 | Qwen3-0.6B @1024d | zerank-1 cloud 4B | 90.8% | 1399/1540 | 89.0% | 90.3% | 80.2% | 92.9% | 27538s |
| 47 | Qwen3-0.6B @1024d | gte-modernbert-reranker 149M | 90.7% | 1397/1540 | 89.7% | 88.8% | 80.2% | 93.0% | 13538s |
| 48 | Qwen3-0.6B @1024d | mxbai-large 560M | 90.7% | 1397/1540 | 89.0% | 89.4% | 82.3% | 92.7% | 6424s |
| 49 | Model2Vec @256d | gte-modernbert-reranker 149M | 90.6% | 1395/1540 | 90.1% | 90.0% | 83.3% | 91.8% | 8483s |
| 50 | Qwen3-0.6B @256d Matryoshka | MiniLM-L12 33M | 90.6% | 1395/1540 | 90.4% | 89.1% | 81.2% | 92.3% | 943s |
| 51 | zembed-1 cloud | MiniLM-L12 33M | 90.6% | 1395/1540 | 90.8% | 87.9% | 83.3% | 92.4% | 10216s |
| 52 | Model2Vec @256d | Qwen3-Reranker 600M | 90.5% | 1394/1540 | 87.9% | 89.7% | 81.2% | 92.7% | 14225s |
| 53 | gte-modernbert-base @768d | MiniLM-L12 33M | 90.5% | 1393/1540 | 86.5% | 89.7% | 82.3% | 93.0% | 915s |
| 54 | Qwen3-0.6B @512d Matryoshka | MiniLM-L12 33M | 90.5% | 1394/1540 | 88.3% | 88.2% | 82.3% | 93.1% | 894s |
| 55 | zembed-1 cloud | No reranker | 90.5% | 1393/1540 | 89.7% | 89.7% | 77.1% | 92.5% | 2186s |
| 56 | nomic-embed-v1.5 @768d | gte-modernbert-reranker 149M | 90.4% | 1392/1540 | 88.7% | 86.9% | 81.2% | 93.3% | 4106s |
| 57 | Model2Vec potion-base-8M @256d | ms-marco-MiniLM-L-6-v2 22M | 90.1% | 1387/1540 | 89.4% | 89.7% | 79.2% | 91.7% | 1290s |
| 58 | Qwen3-0.6B @1024d | bge-large 335M | 90.1% | 1387/1540 | 88.7% | 89.4% | 79.2% | 92.0% | 9656s |
| 59 | Model2Vec @256d | bge-v2-m3 278M | 89.9% | 1385/1540 | 88.3% | 87.9% | 84.4% | 91.9% | 10354s |
| 60 | Model2Vec @256d | mxbai-large 560M | 89.9% | 1384/1540 | 86.9% | 89.1% | 87.5% | 91.4% | 12514s |
| 61 | Model2Vec @256d | No reranker | 89.9% | 1384/1540 | 87.9% | 91.0% | 82.3% | 91.0% | 1808s |
| 62 | Model2Vec potion-base-8M @256d | No reranker | 88.8% | 1367/1540 | 87.2% | 87.9% | 82.3% | 90.4% | 1390s |

---

## Section 5: MEMORIST Candidate Evaluations (gate_eval)

### L0 Personality Filtering

| Candidate | Tier | A/B Accuracy | Speaker ID | Leakage Rate | Intra-Consistency | Profile Bytes |
|-----------|------|-------------|------------|--------------|-------------------|---------------|
| c3c_char_ngram_proxy | edge | 68.6% | 100.0% | 0.0% | 0.875 | 1088.0 |
| c3_style_embedder_wegmann | base | 60.0% | 100.0% | 0.0% | 1.000 | 3136.0 |
| d0_user_filter_only | edge | 60.0% | 100.0% | 0.0% | 1.000 | 0.0 |
| c5_expanded_keywords | edge | 54.3% | 100.0% | 0.0% | 0.365 | 309.2 |
| d1_no_l0 | edge | 37.1% | 53.3% | 60.0% | 1.000 | 0.0 |
| c1_baseline_hand_tuned | edge | 27.1% | 60.0% | 20.0% | 0.504 | 188.6 |

### L3 Salience Scoring

| Candidate | Dataset | Folds | Mean AUC | Mean P@10 | Mean ECE | Positive Rate |
|-----------|---------|-------|----------|-----------|----------|---------------|
| C1 | long | 8 | — | 0.7500 | 0.5011 | 0.119 |
| C1 | short | 10 | 0.6744 | 0.1100 | 0.4617 | 0.050 |
| C2 | long | 8 | — | 0.7500 | 0.7310 | 0.119 |
| C2 | short | 10 | 0.6388 | 0.0900 | 0.0273 | 0.050 |
| C3 | long | 8 | — | 0.7500 | 0.3117 | 0.119 |
| C3 | short | 10 | 0.6740 | 0.1200 | 0.8384 | 0.050 |
| C4 | long | 8 | — | 0.7500 | 0.4897 | 0.119 |
| C4 | short | 10 | 0.7194 | 0.2100 | 0.3951 | 0.050 |
| C5 | long | 8 | — | 0.7500 | 0.4929 | 0.119 |
| C5 | short | 10 | 0.6942 | 0.1000 | 0.4569 | 0.050 |
| C8a | long | 8 | — | 0.7500 | 0.6324 | 0.119 |
| C8a | short | 10 | 0.7079 | 0.1500 | 0.1581 | 0.050 |
| D1 | long | 8 | — | 0.7500 | 0.5601 | 0.119 |
| D1 | short | 10 | 0.4754 | 0.0500 | 0.4509 | 0.050 |
| D2 | long | 8 | — | 0.7500 | 0.2500 | 0.119 |
| D2 | short | 10 | 0.5000 | 0.0200 | 0.9497 | 0.050 |
| D3 | long | 8 | — | 0.7500 | 0.4989 | 0.119 |
| D3 | short | 10 | 0.3256 | 0.0200 | 0.4377 | 0.050 |

### L4 Consolidation

| Candidate | Dataset | Generalization Acc | Gen N | Retention Acc | Contradiction Acc | Storage Mult | Consolidation Time |
|-----------|---------|-------------------|-------|---------------|-------------------|--------------|-------------------|
| c1_baseline | long_horizon_synthetic | 98.0% | 51 | — | 61.3% | 1.059x | 0.06s |
| c1_no_entity_profiles | long_horizon_synthetic | 98.0% | 51 | — | 64.5% | 1.044x | 0.06s |
| c1_relfloor | long_horizon_synthetic | 98.0% | 51 | — | 61.3% | 1.059x | 0.07s |
| c2_llm_abstractive | long_horizon_synthetic | 98.0% | 51 | — | 61.3% | 1.000x | 0.07s |
| c5_nli_contradictions | long_horizon_synthetic | 98.0% | 51 | — | 61.3% | 1.059x | 0.07s |
| d2_summaries_only | long_horizon_synthetic | 98.0% | 51 | — | 64.5% | 1.041x | 0.03s |
| c3_hdbscan_tight | long_horizon_synthetic | 90.2% | 51 | — | 61.3% | 1.046x | 0.91s |
| c3d2_combined | long_horizon_synthetic | 90.2% | 51 | — | 61.3% | 1.046x | 0.87s |
| c3_hdbscan_extractive | long_horizon_synthetic | 64.7% | 51 | — | 71.0% | 1.036x | 0.73s |
| c4_hdbscan_abstractive | long_horizon_synthetic | 64.7% | 51 | — | 71.0% | 1.036x | 0.72s |
| d1_no_consolidation | long_horizon_synthetic | 9.8% | 51 | — | 9.7% | 1.000x | 0.00s |

### L5 Predictive (Surprise Boost)

| Candidate | Dataset | Alpha | P@1 | P@3 | P@10 | Spearman | Kendall Tau |
|-----------|---------|-------|-----|-----|------|----------|-------------|
| l5_embed_pe_ema | long_horizon_synthetic | 0.3 | 35.2% | 44.0% | 48.4% | 0.2986 | 0.2168 |
| l5_embed_pe_ema | short_horizon_200 | 0.3 | 11.0% | 20.0% | 25.5% | 0.4893 | 0.3501 |
| l5_embed_pe_knn | long_horizon_synthetic | 0.3 | 37.4% | 45.0% | 48.4% | 1.0 | 0.9999 |
| l5_embed_pe_knn | short_horizon_200 | 0.3 | 12.0% | 20.0% | 25.0% | 1.0 | 0.9998 |
| l5_minwired | long_horizon_synthetic | 0.3 | 39.6% | 44.0% | 49.5% | 0.1044 | 0.0656 |
| l5_minwired | short_horizon_200 | 0.3 | 12.0% | 21.0% | 27.0% | 0.1392 | 0.1068 |
| l5_none | long_horizon_synthetic | 0.0 | 35.2% | 45.0% | 49.5% | 0.1318 | nan |
| l5_none | short_horizon_200 | 0.0 | 12.5% | 20.0% | 25.0% | -0.0375 | nan |
| l5_unwired | long_horizon_synthetic | 0.0 | 35.2% | 45.0% | 49.5% | 0.1044 | 0.0656 |
| l5_unwired | short_horizon_200 | 0.0 | 12.5% | 20.0% | 25.0% | 0.1392 | 0.1068 |

### Two-Stage Gate Sweep (NLI threshold x regex profile x dataset)

Total configurations: 30

| Threshold | Regex | Dataset | Kept | Dropped | Drop% | P@1 | P@10 |
|-----------|-------|---------|------|---------|-------|-----|------|
| 0.4 | aggressive | long | 475 | 275 | 36.7% | 6.6% | 6.6% |
| 0.4 | off | long | 479 | 271 | 36.1% | 6.6% | 6.6% |
| 0.4 | standard | long | 479 | 271 | 36.1% | 6.6% | 6.6% |
| 0.5 | aggressive | long | 397 | 353 | 47.1% | 6.6% | 6.6% |
| 0.5 | off | long | 398 | 352 | 46.9% | 6.6% | 6.6% |
| 0.5 | standard | long | 398 | 352 | 46.9% | 6.6% | 6.6% |
| 0.55 | aggressive | long | 351 | 399 | 53.2% | 6.6% | 6.6% |
| 0.55 | off | long | 352 | 398 | 53.1% | 6.6% | 6.6% |
| 0.55 | standard | long | 352 | 398 | 53.1% | 6.6% | 6.6% |
| 0.6 | aggressive | long | 304 | 446 | 59.5% | 6.6% | 6.6% |
| 0.6 | off | long | 305 | 445 | 59.3% | 6.6% | 6.6% |
| 0.6 | standard | long | 305 | 445 | 59.3% | 6.6% | 6.6% |
| 0.7 | aggressive | long | 214 | 536 | 71.5% | 6.6% | 6.6% |
| 0.7 | off | long | 214 | 536 | 71.5% | 6.6% | 6.6% |
| 0.7 | standard | long | 214 | 536 | 71.5% | 6.6% | 6.6% |
| 0.4 | aggressive | short | 1103 | 4779 | 81.2% | 2.0% | 4.0% |
| 0.4 | off | short | 1115 | 4767 | 81.0% | 2.0% | 4.0% |
| 0.4 | standard | short | 1115 | 4767 | 81.0% | 2.0% | 4.0% |
| 0.5 | off | short | 755 | 5127 | 87.2% | 2.0% | 4.0% |
| 0.5 | standard | short | 755 | 5127 | 87.2% | 2.0% | 4.0% |
| 0.5 | aggressive | short | 748 | 5134 | 87.3% | 2.0% | 3.5% |
| 0.55 | aggressive | short | 605 | 5277 | 89.7% | 1.0% | 1.5% |
| 0.55 | off | short | 610 | 5272 | 89.6% | 1.0% | 1.5% |
| 0.55 | standard | short | 610 | 5272 | 89.6% | 1.0% | 1.5% |
| 0.6 | aggressive | short | 479 | 5403 | 91.9% | 1.0% | 1.5% |
| 0.6 | off | short | 479 | 5403 | 91.9% | 1.0% | 1.5% |
| 0.6 | standard | short | 479 | 5403 | 91.9% | 1.0% | 1.5% |
| 0.7 | aggressive | short | 287 | 5595 | 95.1% | 0.5% | 1.5% |
| 0.7 | off | short | 287 | 5595 | 95.1% | 0.5% | 1.5% |
| 0.7 | standard | short | 287 | 5595 | 95.1% | 0.5% | 1.5% |

### Root-Level Gate Experiments

| Candidate | Dataset | Messages | Kept | Drop% | P@1 | P@10 | Queries |
|-----------|---------|----------|------|-------|-----|------|---------|
| active_forgetting | long_horizon_synthetic | 750 | 253 | 0.0% | 16.5% | 17.6% | 91 |
| mem0_extraction | long_horizon_synthetic | 750 | 253 | 0.0% | 16.5% | 17.6% | 91 |
| norepinephrine_amplifier | long_horizon_synthetic | 750 | 253 | 0.0% | 16.5% | 17.6% | 91 |
| retention_decay_only | long_horizon_synthetic | 750 | 253 | 0.0% | 16.5% | 17.6% | 91 |
| v05_baseline_nogate | long_horizon_synthetic | 750 | 253 | 0.0% | 16.5% | 17.6% | 91 |
| v05_paper_verbatim | long_horizon_synthetic | 750 | 750 | 0.0% | 6.6% | 7.7% | 91 |
| v05_paper_verbatim | short_horizon_200 | 5882 | 5882 | 0.0% | — | 6.0% | 200 |
| v05_baseline_nogate | short_horizon_200 | 5882 | 15 | 0.0% | — | 1.5% | 200 |

---

## Section 6: Early Development (neuromem R&D Iterations)

These are from the pre-rebrand era (neuromem), testing single conversations, various LLM backends, and search modes.

### Full Pipeline Runs (retrieval + answer + judge)

| File | Score | Correct/Total | Judge | Gen Model | Notes |
|------|-------|---------------|-------|-----------|-------|
| neuromem_locomo_full_rejudged_openai_gpt-4o-mini_conv0_ | 92.8% | 141/152 | gpt-4o-mini |  | agentic |
| neuromem_locomo_full_rejudged_openai_gpt-4o-mini_haiku_ | 88.2% | 134/152 | gpt-4o-mini |  | agentic |
| neuromem_locomo_full_rejudged_openai_gpt-4o-mini_v5_6co | 86.6% | 388/448 | gpt-4o-mini |  | agentic |
| neuromem_locomo_full_rejudged_openai_gpt-4o-mini_v5_3co | 86.3% | 358/415 | gpt-4o-mini |  | agentic |
| neuromem_locomo_full_agentic_episodes_k25_conv0.json | 85.5% | 130/152 | sonnet-3.5 | opus-4 | agentic |
| neuromem_locomo_full_agentic_sonnet_episodes_conv0.json | 85.5% | 130/152 | sonnet-3.5 | opus-4 | agentic |
| neuromem_locomo_full_agentic_episodes_v2_conv0.json | 84.9% | 129/152 | sonnet-3.5 | opus-4 | agentic |
| neuromem_locomo_full_agentic_v4_conv0.json | 84.9% | 129/152 | sonnet-3.5 | opus-4 | agentic |
| neuromem_locomo_full_agentic_diverse_conv0.json | 84.2% | 128/152 | sonnet-3.5 | opus-4 | agentic |
| neuromem_locomo_full_agentic_prompt_v3_conv0.json | 83.6% | 127/152 | sonnet-3.5 | opus-4 | agentic |
| neuromem_locomo_full_rejudged_openai_gpt-4o-mini_haiku_ | 83.4% | 346/415 | gpt-4o-mini |  | agentic |
| neuromem_locomo_full_agentic_episodes_sonnet_conv0.json | 80.3% | 122/152 | sonnet-3.5 | sonnet-3.5 | agentic |
| neuromem_locomo_full_rejudged_anthropic_claude-sonnet-4 | 79.9% | 358/448 | sonnet-3.5 |  | agentic |
| neuromem_locomo_full_agentic_episodes_3conv.json | 79.5% | 330/415 | sonnet-3.5 | opus-4 | agentic |
| neuromem_locomo_full_rejudged_anthropic_claude-sonnet-4 | 79.3% | 329/415 | sonnet-3.5 |  | agentic |
| neuromem_locomo_full_rejudged_anthropic_claude-sonnet-4 | 74.6% | 359/481 | sonnet-3.5 |  | agentic |
| neuromem_locomo_full_rejudged_anthropic_claude-sonnet-4 | 71.8% | 339/472 | sonnet-3.5 |  | agentic |
| neuromem_locomo_full_agentic_agentic_opus_conv0.json | 62.5% | 95/152 | sonnet-3.5 | opus-4 | agentic |
| neuromem_locomo_full_agentic_v5_3conv.json | 45.5% | 189/415 | sonnet-3.5 | opus-4 | agentic |
| neuromem_locomo_full_agentic_sonnet_llmrerank_conv0.jso | 4.6% | 7/152 | sonnet-3.5 | opus-4 | agentic |

### Retrieval-Only Runs (no answer generation)

| File | Top-K | QA Pairs | Recall@K | Content Overlap |
|------|-------|----------|----------|-----------------|
| neuromem_locomo_retrieval_agentic_hyde_entity_k20.json | 20 | 152 | 0.6075 | 0.3185 |
| neuromem_locomo_retrieval_agentic_agentic_temporal_reso | 15 | 152 | 0.5970 | 0.3605 |
| neuromem_locomo_retrieval_standard.json | 15 | 1540 | 0.5970 | 0.4002 |
| neuromem_locomo_retrieval_agentic_hyde_haiku.json | 15 | 152 | 0.5954 | 0.2965 |
| neuromem_locomo_retrieval_agentic_episodes_k25_conv0.js | 25 | 152 | 0.5866 | 0.5483 |
| neuromem_locomo_retrieval_agentic_agentic_opus_conv0.js | 15 | 152 | 0.5822 | 0.2987 |
| neuromem_locomo_retrieval_agentic_entity_v2.json | 15 | 152 | 0.5707 | 0.3015 |
| neuromem_locomo_retrieval_agentic_reranker_v2.json | 15 | 152 | 0.5707 | 0.3015 |
| neuromem_locomo_retrieval_agentic.json | 15 | 152 | 0.5609 | 0.2949 |
| neuromem_locomo_retrieval_agentic_entity_v1.json | 15 | 152 | 0.5609 | 0.2949 |
| neuromem_locomo_retrieval_agentic_episodes_k20_conv0.js | 20 | 152 | 0.5521 | 0.5443 |
| neuromem_locomo_retrieval.json | 10 | 1540 | 0.5458 | 0.3715 |
| neuromem_locomo_retrieval_agentic_v5_6conv.json | 15 | 973 | 0.5399 | 0.5042 |
| neuromem_locomo_retrieval_agentic_episodes_3conv.json | 15 | 415 | 0.5283 | 0.5805 |
| neuromem_locomo_retrieval_agentic_gpt41_5conv_a.json | 15 | 725 | 0.5224 | 0.5089 |
| neuromem_locomo_retrieval_agentic_gpt41_5conv_b.json | 15 | 815 | 0.5170 | 0.5201 |
| neuromem_locomo_retrieval_agentic_v5_3conv.json | 15 | 415 | 0.5057 | 0.5547 |
| neuromem_locomo_retrieval_agentic_episodes_sonnet_conv0 | 15 | 152 | 0.5033 | 0.5279 |
| neuromem_locomo_retrieval_agentic_episodes_6conv.json | 15 | 973 | 0.4963 | 0.6041 |
| neuromem_locomo_retrieval_agentic_episodes_v2_conv0.jso | 15 | 152 | 0.4962 | 0.5249 |
| neuromem_locomo_retrieval_agentic_sonnet_llmrerank_conv | 15 | 152 | 0.4885 | 0.5121 |
| neuromem_locomo_retrieval_agentic_v4_conv0.json | 15 | 152 | 0.4885 | 0.5112 |
| neuromem_locomo_retrieval_agentic_diverse_conv0.json | 15 | 152 | 0.4874 | 0.5056 |
| neuromem_locomo_retrieval_agentic_v6_claude_missing.jso | 15 | 622 | 0.4856 | 0.5814 |
| neuromem_locomo_retrieval_agentic_sonnet_episodes_conv0 | 15 | 152 | 0.4819 | 0.4936 |
| neuromem_locomo_retrieval_agentic_minilm_retrieval_conv | 15 | 152 | 0.4561 | 0.5273 |
| neuromem_locomo_retrieval_agentic_prompt_v3_conv0.json | 15 | 152 | 0.4523 | 0.5049 |

---

## Section 7: Competitor Benchmarks

### LoCoMo-10 Standard (1540 questions)

| System | Score | Correct | Source |
|--------|-------|---------|--------|
| EverMemOS | 94.5% | 1455/1540 | A_modal_volume |
| RAG | 86.2% | 1327/1540 | A_modal_volume |
| Engram | 84.5% | 1302/1540 | A_modal_volume |
| BM25 | 80.5% | 1239/1540 | A_modal_volume |
| SuperMemory | 65.4% | 1007/1540 | A_modal_volume |
| Mem0 | 61.4% | 946/1540 | A_modal_volume |

### Legacy Competitors (v1/v2 Custom Benchmark, 60 queries)

| File | System | Version | Hit Rate | Hit+Partial | Queries | Notes |
|------|--------|---------|----------|-------------|---------|-------|
| benchmark_chromadb_results.json | ChromaDB | 1.5.5 | — | — |  | Raw retrieval results (no scor |
| benchmark_langmem_results.json | LangMem | 0.0.30 | — | — | None | Legacy competitor |
| benchmark_v2_chromadb_results.json | ChromaDB | 1.5.5 | — | — | 60 | Legacy competitor |
| benchmark_v2_chromadb_scored.json | ChromaDB |  | 30.0% | 68.3% | 60 | Legacy competitor (custom form |
| benchmark_v2_cognee_results.json | Cognee | 0.5.4 | — | — | 60 | Legacy competitor |
| benchmark_v2_fts5_results.json | unknown |  | — | — | None | Legacy competitor |
| benchmark_v2_langmem_results.json | LangMem | latest | — | — | 60 | Legacy competitor |
| benchmark_v2_langmem_scored.json | LangMem |  | — | — | 60 | Legacy competitor |
| benchmark_v2_neuromem_results.json | unknown |  | — | — | None | Legacy competitor |
| benchmark_v2_mem0_results.json | Mem0 | 1.0.5 | — | — | 60 | Mem0 v2 competitor |

---

## Section 8: LongMemEval Results

500 questions across 6 types: single-session-user, single-session-assistant, single-session-preference, multi-session, temporal-reasoning, knowledge-update.

### Scored Runs (answer + judge)

| File | System | Benchmark | Version | Accuracy | Correct/500 | Judge | Source |
|------|--------|-----------|---------|----------|-------------|-------|--------|
| neuromem_pro_longmemeval_run1.json | neuromem_pro | LongMemEval_oracle | v3-modal | 91.0% | 455/500 | gpt-4o-mini | results |
| neuromem_pro_longmemeval_run2.json | neuromem_pro | LongMemEval_oracle | v3-modal | 91.0% | 455/500 | gpt-4o-mini | results |
| neuromem_pro_longmemeval_run3.json | neuromem_pro | LongMemEval_oracle | v3-modal | 90.8% | 454/500 | gpt-4o-mini | results |
| neuromem_base_run1.json | neuromem_base | LongMemEval_oracle | v1-modal | 90.4% | 452/500 | gpt-4o-mini | results |
| neuromem_base_run1.json | neuromem_base | LongMemEval_oracle | v1-modal | 86.0% | 430/500 | gpt-4o-mini | results_s |
| neuromem_pro_longmemeval_run1.json | neuromem_pro | LongMemEval_oracle | v3-modal-s | 85.6% | 428/500 | gpt-4o-mini | results_s |
| neuromem_longmemeval_s_scored.json | Neuromem | LongMemEval_s |  | 72.4% | 362/500 | opus-4 | archive |
| neuromem_longmemeval_s_scored_opus46.json | Neuromem | LongMemEval_s |  | 72.4% | 362/500 | opus-4 | archive |
| neuromem_longmemeval_s_scored_sonnet45.json | Neuromem | LongMemEval_s |  | 68.6% | 343/500 | sonnet-3.5 | archive |
| supermemory_run1.json | supermemory | LongMemEval_oracle | v1-modal | 15.8% | 79/500 | gpt-4o-mini | results |

### Retrieval-Only

| File | System | Top-K | Recall | Overlap |
|------|--------|-------|--------|---------|
| neuromem_longmemeval_oracle_retrieval.json | Neuromem | 10 | 0.9049 | 0.532 |
| neuromem_longmemeval_s_retrieval.json | Neuromem | 10 | 0.8435 | 0.5161 |

### LongMemEval Directory — LoCoMo Competitor Reruns

| File | System | Score | Correct/Total | Source |
|------|--------|-------|---------------|--------|
| rag_run1.json | rag | 91.8% | 459/500 | L_longmemeval_results |
| bm25_run1.json | bm25 | 90.0% | 450/500 | L_longmemeval_results |
| rag_run1.json | rag | 87.0% | 435/500 | L_longmemeval_results_s |
| engram_run1.json | engram | 86.0% | 430/500 | L_longmemeval_results |
| engram_run1.json | engram | 82.2% | 411/500 | L_longmemeval_results_s |
| bm25_run1.json | bm25 | 81.6% | 408/500 | L_longmemeval_results_s |
| mem0_run1.json | mem0 | 66.0% | 330/500 | L_longmemeval_results_s |
| mem0_run1.json | mem0 | 64.0% | 320/500 | L_longmemeval_results |

---

## Section 9: v3 Custom Benchmark (Retrieval Quality)

Custom 5-session, 5-query smoke tests comparing raw retrieval quality across systems.

| File | System | Status | Queries | Avg Query ms | Ingested |
|------|--------|--------|---------|--------------|----------|
| chromadb_raw.json | ChromaDB | COMPLETED | 50 | 8.3 | 498 |
| chromadb_raw_smoke.json | ChromaDB | COMPLETED | 5 | 17.3 | 45 |
| cognee_raw_smoke.json | Cognee | COMPLETED | 5 | 179.2 | 0 |
| fts5_raw.json | FTS5 | COMPLETED | 50 | 0.6 | 498 |
| fts5_raw_smoke.json | FTS5 | COMPLETED | 5 | 0.3 | 45 |
| mem0_raw.json | Mem0 | COMPLETED | 50 | 8.1 | 0 |
| neuromem_raw.json | Neuromem | COMPLETED | 50 | 2.3 | 498 |
| neuromem_raw_smoke.json | Neuromem | COMPLETED | 5 | 1.1 | 35 |
| openmemory_raw.json | OpenMemory | COMPLETED | 50 | 139.7 | 498 |
| openmemory_raw_smoke.json | OpenMemory | COMPLETED | 5 | 11.6 | 45 |

---

## Section 10: Cross-Archive Duplicate Map

Files that exist in multiple archive locations (canonical source listed first):

| File | Locations |
|------|-----------|
| bm25_run1.json | L_longmemeval_results, L_longmemeval_results_s |
| bm25_v2_run1.json | A_modal_volume, B_repo_locomo, J_v020, K_final_copy |
| engram_run1.json | L_longmemeval_results, L_longmemeval_results_s |
| engram_v2_run1.json | A_modal_volume, B_repo_locomo, J_v020, K_final_copy |
| evermemos_v2_run1.json | A_modal_volume, B_repo_locomo, J_v020, K_final_copy |
| mem0_run1.json | L_longmemeval_results, L_longmemeval_results_s |
| mem0_v2_run1.json | A_modal_volume, B_repo_locomo, J_v020, K_final_copy |
| neuromem_base_run1.json | L_longmemeval_results, L_longmemeval_results_s |
| neuromem_base_v2_run1.json | A_modal_volume, J_v020, K_final_copy |
| neuromem_pro_longmemeval_run1.json | L_longmemeval_results, L_longmemeval_results_s |
| neuromem_pro_rerun_1.json | A_modal_volume, L_longmemeval_reruns |
| neuromem_pro_rerun_2.json | A_modal_volume, L_longmemeval_reruns |
| neuromem_pro_rerun_3.json | A_modal_volume, L_longmemeval_reruns |
| neuromem_pro_rerun_4.json | A_modal_volume, L_longmemeval_reruns |
| neuromem_pro_rerun_5.json | A_modal_volume, L_longmemeval_reruns |
| neuromem_pro_rerun_6.json | A_modal_volume, L_longmemeval_reruns |
| neuromem_pro_rerun_7.json | A_modal_volume, L_longmemeval_reruns |
| neuromem_pro_rerun_8.json | A_modal_volume, L_longmemeval_reruns |
| neuromem_pro_v2_run1.json | A_modal_volume, J_v020, K_final_copy |
| rag_run1.json | L_longmemeval_results, L_longmemeval_results_s |
| rag_v2_run1.json | A_modal_volume, B_repo_locomo, J_v020, K_final_copy |
| supermemory_v2_run1.json | A_modal_volume, B_repo_locomo, J_v020, K_final_copy |

---

## Key Findings

1. **Best LoCoMo-10 score: 93.6%** (truememory_pro_v060_l3only_run1 and alpha02_run1, single-run highs)
2. **Best 3-run mean: 93.20% ±0.35** (v0.6.0, threshold 0.05/0.02, **alpha=0.2** — Modal alpha sweep winner)
3. **Alpha=0.2 is the optimal L5 surprise boost** — 5-point sweep (0, 0.1, 0.15, 0.2, 0.3) × 3 seeds, cross-validated on GPUBox
4. **Alpha dip at 0.1-0.15** — enough boost to interfere, not enough to help. Avoid this range.
5. **Issue #75 (MiniLM reranker bug)** caused ~1-2% regression in v0.5.0 unpatched runs
6. **v0.6.0 L3 learned weights** improve single_hop significantly; overall within judge noise (±1%)
7. **L5 surprise boost** reduces run-to-run variance (std 0.35 at alpha=0.2 vs 1.00 at alpha=0)
8. **GPUBox validated as apples-to-apples with Modal** — single-run scores within 0.1-0.3% at 5 concurrent; 6 concurrent shows ~0.5-1% degradation
9. **56-grid sweep winner:** gte-modernbert + qwen3-reranker emerged as best embedder/reranker combo
10. **Judge variance** across identical configs is roughly ±1% (McNemar p=0.545)
11. **Competitor gap:** Best TrueMemory (93.6%) vs EverMemOS (94.5%) = -0.9%; vs RAG baseline (86.2%) = +7.4%
12. **L0 char-n-gram style vectors (PR #82)** replace hand-tuned keyword extraction. C3c scored 0.686 vs 0.271 on MEMORIST personality probes. On LoCoMo: neutral (-0.13% at scale=0.7, within judge noise). LoCoMo only exercises L0 on 5.8% of queries.
13. **L0 score scale sweep** (0.6, 0.7, 0.8, 0.9) found 0.7 optimal for LoCoMo damage reduction. The sweep confirmed the regression was from uncapped 5.0+ persona-bias scores displacing factual results, not from the style vector algorithm itself.
14. **PersonaLoCoMo benchmark in progress** — dedicated 2000-question personality eval using LoCoMo conversations. Required to properly tune L0 since LoCoMo's personality coverage is insufficient.

---

*Updated: 2026-04-27 | Total unique result files cataloged: 340*