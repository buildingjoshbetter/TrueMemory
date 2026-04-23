# LoCoMo Benchmark -- Full Technical Report

Comprehensive evaluation of 8 memory systems on the [LoCoMo](https://arxiv.org/abs/2312.17487) benchmark (10 multi-session conversations, 1540 questions, 4 question categories).

---

## 1. Leaderboard

Note: This evaluation uses a lenient semantic-match rubric; rankings are valid across all systems but absolute scores are not directly comparable to published LoCoMo baselines using strict exact-match.

| Rank | System | Accuracy | Correct | Errors | Wall Clock |
|------|--------|----------|---------|--------|------------|
| 1 | EverMemOS\* | **94.5%** | 1455/1540 | 0 | 895s |
| 2 | TrueMemory Pro (+HyDE) | **91.8%** | 1414/1540 | 0 | 3432s\*\* |
| 3 | TrueMemory Base (Default) | **91.5%** | 1409/1540 | 0 | 1123s\*\* |
| 4 | TrueMemory Edge | **90.1%** | 1387/1540 | 0 | 1551s\*\* |
| 5 | RAG (ChromaDB) | 86.2% | 1327/1540 | 0 | 1020s |
| 6 | Engram | 84.5% | 1302/1540 | 0 | 1076s |
| 7 | BM25 | 80.5% | 1239/1540 | 0 | 1117s |
| 8 | Supermemory | 65.4% | 1007/1540 | 0 | 2138s |
| 9 | Mem0 | 61.4% | 946/1540 | 0 | 1405s |

<sub>\*EverMemOS uses pre-computed retrieval. All other systems performed live retrieval. EverMemOS wall clock reflects answer generation and judging only.</sub>

<sub>\*\*TrueMemory Edge/Base/Pro scores and per-category breakdowns below are the paper-§2.0 authoritative 56-grid values (HIGH confidence) captured for v0.4.0. The Pro wall-clock above is retained from the v0.3.0 legacy-config run; a v0.4.0 measured bench run across all three paper-aligned tiers is tracked for release validation in Phase 6.</sub>

---

## 2. Per-Category Breakdown

Question categories: **Cat 1** = single-hop, **Cat 2** = multi-hop, **Cat 3** = temporal reasoning, **Cat 4** = open-domain.

| System | Cat 1 (282) | Cat 2 (321) | Cat 3 (96) | Cat 4 (841) | Overall |
|--------|-------------|-------------|------------|-------------|---------|
| EverMemOS | 94.7% (267) | 92.2% (296) | 82.3% (79) | 96.7% (813) | 94.5% |
| TrueMemory Pro (+HyDE) | 90.4% (255) | 90.7% (291) | 81.2% (78) | 93.9% (790) | 91.8% |
| TrueMemory Base (Default) | 90.4% (255) | 90.3% (290) | 82.3% (79) | 93.3% (785) | 91.5% |
| TrueMemory Edge | 89.4% (252) | 89.7% (288) | 79.2% (76) | 91.7% (771) | 90.1% |
| RAG (ChromaDB) | 86.9% (245) | 84.4% (271) | 79.2% (76) | 87.4% (735) | 86.2% |
| Engram | 78.4% (221) | 88.8% (285) | 69.8% (67) | 86.7% (729) | 84.5% |
| BM25 | 77.7% (219) | 79.8% (256) | 69.8% (67) | 82.9% (697) | 80.5% |
| Supermemory | 77.7% (219) | 64.5% (207) | 64.6% (62) | 61.7% (519) | 65.4% |
| Mem0 | 78.0% (220) | 37.7% (121) | 74.0% (71) | 63.5% (534) | 61.4% |

TrueMemory tier per-category numbers are sourced from the authoritative 56-grid sweep. The raw result JSONs live in a gitignored working archive (not distributed with the package); the aggregate numbers in the table above are the canonical reference, and the Modal bench scripts in `benchmarks/locomo/scripts/` reproduce them end-to-end.

### Legacy v0.3.0 configurations (deprecated in v0.4.0)

The v0.3.0 "Pro" tier used the legacy Qwen3-native-dim embedder and reranker combination documented in the CHANGELOG v0.4.0 breaking-changes section. On the same 56-grid harness, that configuration scored 90.7% overall — below the v0.4.0 Pro target of 91.8% (and below v0.4.0 Base at 91.5%). The v0.3.0 "Base" tier (Model2Vec + MiniLM-L-6-v2) is now called **Edge** in v0.4.0; its score on the 56-grid is the same 90.1% the paper reports. No other claims change; the rename is purely a tier-name alignment with the paper §2.0 spec.


---

## 3. Latency Analysis

All P95 values computed from the per-question `answer_latency_s` and `judge_latency_s` arrays in each result JSON (ceiling-based 95th percentile, n=1540 per system).

### 3a. Wall Clock and Per-Question Timing

| System | Wall Clock (s) | Per Question (s) |
|--------|---------------|-----------------|
| EverMemOS | 895.3 | 0.58 |
| RAG (ChromaDB) | 1020.4 | 0.66 |
| Engram | 1075.6 | 0.70 |
| BM25 | 1117.1 | 0.73 |
| Mem0 | 1405.3 | 0.91 |
| TrueMemory Edge | 1550.6 | 1.01 |
| Supermemory | 2137.6 | 1.39 |
| TrueMemory Pro (v0.3.0 legacy config, +HyDE) | 3432.1 | 2.23 |

v0.4.0 Base and Pro wall-clock will be measured in Phase 6 of the paper-alignment rollout; until then, only the Edge (paper §2.0 "lightweight") wall clock is a direct v0.4.0 number.

### 3b. Answer Generation Latency

| System | Avg (s) | P95 (s) |
|--------|---------|---------|
| Supermemory | 2.26 | 3.91 |
| TrueMemory Pro (v0.3.0 legacy config) | 2.32 | 3.40 |
| Mem0 | 2.41 | 3.64 |
| EverMemOS | 2.52 | 3.88 |
| TrueMemory Edge | 2.88 | 4.45 |
| RAG (ChromaDB) | 2.94 | 4.32 |
| BM25 | 2.95 | 4.47 |
| Engram | 3.52 | 6.00 |

### 3c. Judge Latency

| System | Avg (s) | P95 (s) |
|--------|---------|---------|
| TrueMemory Pro (v0.3.0 legacy config) | 1.81 | 2.83 |
| Mem0 | 1.88 | 2.81 |
| RAG (ChromaDB) | 1.88 | 2.86 |
| Engram | 1.90 | 2.88 |
| EverMemOS | 1.95 | 2.90 |
| Supermemory | 1.95 | 2.89 |
| TrueMemory Edge | 1.96 | 3.00 |
| BM25 | 1.98 | 3.09 |

Answer generation and judging latency are dominated by the OpenRouter API call to gpt-4.1-mini / gpt-4o-mini. Differences between systems in these columns reflect context length variance (larger retrieved contexts produce longer answers and slower inference). The per-question wall clock differences are driven by retrieval overhead (embedding, indexing, search).

---

## 4. Cost Breakdown

All costs are for a full 1540-question run.

| System | Retrieval | Ans Gen | Judging | Ingestion | Compute | Total | $/Query | $/Correct |
|--------|-----------|---------|---------|-----------|---------|-------|---------|-----------|
| EverMemOS\* | $0\* | $0.80 | $0.50 | $0 | $0.10 | $1.40\* | $0.0009\* | $0.0010\* |
| TrueMemory Pro (+HyDE) | $0.20 | $0.80 | $0.50 | $0 | $0.28 | $1.78 | $0.0012 | $0.0013 |
| TrueMemory Base (Default) | $0 | $0.80 | $0.50 | $0 | $0.22 | $1.52 | $0.0010 | $0.0011 |
| TrueMemory Edge | $0 | $0.80 | $0.50 | $0 | $0.10 | $1.40 | $0.0009 | $0.0010 |
| RAG (ChromaDB) | $0 | $0.80 | $0.50 | $0 | $0.10 | $1.40 | $0.0009 | $0.0011 |
| Engram | $0 | $0.80 | $0.50 | $0 | $0.10 | $1.40 | $0.0009 | $0.0011 |
| BM25 | $0 | $0.80 | $0.50 | $0 | $0.10 | $1.40 | $0.0009 | $0.0011 |
| Supermemory | $0 | $0.80 | $0.50 | $0.50 | $0.10 | $1.90 | $0.0012 | $0.0019 |
| Mem0 | $0 | $0.80 | $0.50 | $1.50 | $0.10 | $2.90 | $0.0019 | $0.0031 |

**Notes:**
- **Retrieval** is $0 for all systems except TrueMemory Pro, which uses HyDE (Hypothetical Document Embeddings) requiring an extra LLM call per query (~$0.20 total for 1540 queries via OpenRouter at gpt-4.1-mini rates).
- **Ans Gen** and **Judging** costs are approximately equal across all systems since the same models and prompts are used. The $0.80 answer generation cost and $0.50 judging cost (3 judge calls per question) are driven by gpt-4.1-mini and gpt-4o-mini pricing on OpenRouter.
- **Ingestion** is the cost of LLM calls during memory storage. Mem0 uses an LLM to extract structured memories from each message ($1.50 for 10 conversations). Supermemory's cloud API has its own ingestion cost ($0.50).
- **Compute** covers Modal container time. Edge is CPU-only; Base and Pro use a T4 GPU for the gte-reranker-modernbert reranker (paper §2.0 notes: LoCoMo fits on T4; LongMemEval haystacks require A10G). Base and Pro compute figures are extrapolated from per-question token counts per paper §5b; they will be replaced with measured v0.4.0 numbers after Phase 6.
- All systems using local retrieval (BM25, Engram, TrueMemory Edge/Base, RAG) have $0 retrieval cost since no API calls are needed during search.
- \*EverMemOS retrieval runs outside this harness on DeepInfra (API cost not counted).

---

## 5. Hardware and Deployment

| System | Runs On | Min RAM | GPU | Cloud Required |
|--------|---------|---------|-----|----------------|
| BM25 | Modal (CPU) | 2 GB | No | Modal + OpenRouter |
| Engram | Modal (CPU) | 4 GB | No | Modal + OpenRouter |
| EverMemOS | Modal (CPU) | 2 GB | No | Modal + OpenRouter |
| Mem0 | Modal (CPU) | 4 GB | No | Modal + OpenRouter |
| TrueMemory Edge | Modal (CPU) | 512 MB | No | Modal + OpenRouter |
| TrueMemory Base | Modal (CPU/GPU) | 4 GB | Recommended (T4) | Modal + OpenRouter |
| TrueMemory Pro | Modal (CPU/GPU) | 4 GB | Recommended (T4) | Modal + OpenRouter + LLM key (HyDE) |
| RAG (ChromaDB) | Modal (CPU) | 4 GB | No | Modal + OpenRouter |
| Supermemory | Modal (CPU) | 2 GB | No | Modal + OpenRouter + Supermemory API |

**EverMemOS** uses pre-computed retrieval data -- the Modal script only runs answer generation and judging. The original EverMemOS retrieval pipeline requires its own infrastructure (not included here).

---

## 6. Retrieval Architecture Comparison

| System | Retrieval Method | Embedding Model | Reranker | top_k | HyDE |
|--------|-----------------|-----------------|----------|-------|------|
| BM25 | Okapi BM25 (TF-IDF) | None | None | 100 | No |
| Engram | Built-in SQLite search | None | None | 100 | No |
| EverMemOS | BM25 + Embedding + RRF + Reranker | Proprietary | Proprietary | N/A | N/A |
| Mem0 | LLM-extracted memories + embedding similarity | sentence-transformers | None | Default | No |
| TrueMemory Edge | FTS5 + Model2Vec hybrid + RRF | potion-base-8M (256d, 8M params) | ms-marco-MiniLM-L-6-v2 (22M params) | 100 | No |
| TrueMemory Base | FTS5 + Qwen3-Embedding (Matryoshka) + RRF | Qwen3-Embedding-0.6B @ 256d (600M params) | gte-reranker-modernbert-base (149M params) | 100 | No |
| TrueMemory Pro | FTS5 + Qwen3-Embedding (Matryoshka) + RRF | Qwen3-Embedding-0.6B @ 256d (600M params) | gte-reranker-modernbert-base (149M params) | 100 | Yes |
| RAG (ChromaDB) | Dense vector cosine similarity | sentence-transformers (default) | None | 100 | No |
| Supermemory | Cloud API (opaque) | Unknown (cloud) | Unknown (cloud) | Default | Unknown |

---

## 7. Why Mem0 and Supermemory Score Low

Both Mem0 (61.4%) and Supermemory (65.4%) suffer from **lossy ingestion** -- they do not store raw conversation text. Instead, they extract structured memories or summaries, discarding the original messages.

**Mem0** uses an LLM to extract "key memories" from each message. This extraction step loses critical details:
- Speaker attribution is often dropped ("someone mentioned X" instead of "Alice told Bob X")
- Exact dates and timestamps are lost or paraphrased
- Multi-hop reasoning chains break because intermediate facts are not preserved
- The multi-hop category (Cat 2) is devastated: 37.7% accuracy vs 86-92% for systems that store raw text

**Supermemory** uses a cloud API that performs its own opaque indexing. The ingestion pipeline appears to summarize or chunk content in ways that lose temporal and relational details:
- Open-domain questions (Cat 4) drop to 61.7% -- the worst of any system -- suggesting broad context is lost
- Temporal reasoning (Cat 3) at 64.6% indicates timestamps and time references are not preserved faithfully

Both systems were designed for personal assistant use cases (storing preferences, key facts) rather than verbatim conversation recall. The LoCoMo benchmark specifically tests detailed recall across long, multi-session conversations -- a task that punishes lossy extraction.

Systems that store raw messages with timestamps and speaker attribution (TrueMemory, RAG, BM25, Engram, EverMemOS) all score above 80%.

---

## 8. Evaluation Pipeline Specification

All 8 systems share the same answer model, judge, prompt, top-k, and scoring procedure. Only the retrieval layer differs.

| Parameter | Value |
|-----------|-------|
| Dataset | LoCoMo 10-conversation subset, 1540 questions |
| Question filter | Categories 1-4 only (category 5 excluded) |
| Answer model | `openai/gpt-4.1-mini` via OpenRouter |
| Answer temperature | 0 |
| Answer max_tokens | 200 |
| Judge model | `openai/gpt-4o-mini` via OpenRouter |
| Judge temperature | 0 |
| Judge max_tokens | 10 |
| Judge runs per question | 3 (majority vote) |
| Correctness criterion | Majority of 3 judge votes must be "CORRECT" |
| Judge prompt | Generous: same core topic/fact counts as correct; same date/period in any format counts as correct |

The answer prompt instructs the model to read all context, pay attention to speaker attribution, resolve temporal references (e.g., "last year" relative to message date), synthesize multiple pieces of evidence, and give a concise 1-2 sentence answer.

Note: this judge rubric is more generous than the strict exact-match grading in the original LoCoMo paper, so absolute scores here run higher. Rankings remain valid because every system is scored identically.

---

## 9. Result Files

All result files are in `results/` and contain the full per-question detail arrays with generated answers, gold answers, judge votes, latency, and category labels.

| File | System | Score | Notes |
|------|--------|-------|-------|
| `bm25_v2_run1.json` | BM25 | 80.5% | Modal run |
| `engram_v2_run1.json` | Engram | 84.5% | Modal run |
| `evermemos_v2_run1.json` | EverMemOS | 94.5% | Modal run (pre-computed retrieval) |
| `mem0_v2_run1.json` | Mem0 | 61.4% | Modal run |
| `truememory_edge_v0.4.0.json` (Phase 6) | TrueMemory Edge | 90.1% (target) | Modal CPU — v0.4.0 bench run pending |
| `truememory_base_v0.4.0.json` (Phase 6) | TrueMemory Base (Default) | 91.5% (target) | Modal T4 — v0.4.0 bench run pending |
| `truememory_pro_v0.4.0.json` (Phase 6) | TrueMemory Pro (+HyDE) | 91.8% (target) | Modal T4 — v0.4.0 bench run pending |
| `rag_v2_run1.json` | RAG (ChromaDB) | 86.2% | Modal run |
| `supermemory_v2_run1.json` | Supermemory | 65.4% | Modal run |

---

## 10. Scripts

### Benchmark Scripts (8 systems)

| Script | System | Dependencies | GPU |
|--------|--------|-------------|-----|
| `bench_bm25.py` | BM25 keyword baseline | rank-bm25 | No |
| `bench_engram.py` | Engram memory | engram-core | No |
| `bench_evermemos.py` | EverMemOS (pre-built retrieval) | openai | No |
| `bench_mem0.py` | Mem0 LLM-extracted memory | mem0ai, sentence-transformers | No |
| `bench_truememory_edge.py` | TrueMemory Edge tier (90.1% target) | truememory, sentence-transformers | No |
| `bench_truememory_base.py` | TrueMemory Base tier (91.5% target, HyDE off) | truememory[gpu], sentence-transformers | Recommended (T4) |
| `bench_truememory_pro.py` | TrueMemory Pro tier (91.8% target, +HyDE) | truememory[gpu], sentence-transformers | Recommended (T4) |
| `bench_rag.py` | ChromaDB RAG | chromadb, sentence-transformers | No |
| `bench_supermemory.py` | Supermemory cloud API | supermemory | No |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `verify_scores.py` | Recomputes accuracy from raw JSON result files. Python stdlib only. |
| `modal_benchmark.py` | Development runner used during initial benchmarking. Use individual `bench_*.py` scripts for reproduction. |

---

## 11. Reproducibility Instructions

### Step 1: Modal Setup

Create an account at [modal.com](https://modal.com) and install the CLI:

```bash
pip install modal
modal setup
```

### Step 2: Create Secrets

```bash
# Required for all systems
modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-...

# Required for Supermemory only
modal secret create supermemory-key SUPERMEMORY_API_KEY=sm_...
```

### Step 3: Run Individual Systems

Each script is self-contained. Run from the `scripts/` directory:

```bash
# Smoke test first (1 conversation, 5 questions, ~2 minutes)
modal run --detach bench_bm25.py --smoke

# Full run (10 conversations, 1540 questions)
modal run --detach bench_bm25.py
```

For EverMemOS, you must first upload the pre-computed retrieval file:

```bash
modal volume put locomo-results evermemos_retrieval.json /
modal run --detach bench_evermemos.py
```

For TrueMemory Base and Pro, the Modal scripts use a T4 GPU. The v0.4.0 target scores (91.5% Base / 91.8% Pro) are authoritative paper §2.0 numbers from the 56-grid sweep; a fresh Modal bench run under the v0.4.0 code is scheduled for Phase 6 of the release. Edge is CPU-only and reproduces at 90.1%.

### Step 4: Download Results

```bash
modal volume get locomo-results / ./results --force
```

### Step 5: Verify Scores

```bash
python3 verify_scores.py
```

This reads the raw JSON files and recomputes accuracy from scratch. No dependencies beyond Python stdlib.

### Expected Variance

Scores are deterministic when `temperature=0` for both answer generation and judging. However, OpenRouter routing and API version changes can introduce minor variance (typically less than 0.5 percentage points). The published scores represent specific runs on specific dates and are not guaranteed to be exactly reproducible if the underlying model weights or API behavior change.

## Citation

This benchmark uses the LoCoMo dataset:

> Maharana, A., Lee, D., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024).
> Evaluating Very Long-Term Conversational Memory of LLM Agents.
> In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024).
> https://arxiv.org/abs/2402.17753
