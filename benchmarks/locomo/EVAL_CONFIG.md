# Evaluation Configuration

Configuration used for all LoCoMo benchmark runs.

<p align="center">
  <img src="../../assets/charts/eval-pipeline.png" alt="Evaluation Pipeline" />
</p>

## Answer Generation

| Parameter | Value |
|-----------|-------|
| Model | `openai/gpt-4.1-mini` |
| Temperature | 0 |
| Max Tokens | 200 |

## Judging

| Parameter | Value |
|-----------|-------|
| Model | `openai/gpt-4o-mini` |
| Temperature | 0 |
| Max Tokens | 10 |
| Runs per question | 3 (majority vote) |

## Dataset

| Parameter | Value |
|-----------|-------|
| Dataset | LoCoMo v1, 10 conversations, 1,540 questions |
| Category 1 (single-hop) | 282 questions |
| Category 2 (multi-hop) | 321 questions |
| Category 3 (temporal) | 96 questions |
| Category 4 (open-domain) | 841 questions |
| Category 5 (adversarial) | Excluded per standard practice |

## Preprocessing

| Parameter | Value |
|-----------|-------|
| Temporal resolution | Relative dates resolved to absolutes during ingestion |

## Infrastructure

| Parameter | Value |
|-----------|-------|
| Compute platform | Modal serverless (Python 3.11) |
| API routing | OpenRouter |
| Recommended reproduction platform | Modal (free credits available at [modal.com](https://modal.com)) |

## TrueMemory tier configurations

TrueMemory v0.6.0 ships three tiers. All three share the same 6-layer pipeline (FTS5 + dense + RRF + cross-encoder reranker + temporal/salience layers); the tiers differ only in embedder, reranker, and whether HyDE query expansion is used. Base and Pro share the same embedder + reranker. Only HyDE toggles between them.

| Tier | Embedder | Reranker | HyDE | top_k | LoCoMo (3-run mean) | Hardware |
|------|----------|----------|------|-------|---------------------|----------|
| Edge | Model2Vec potion-base-8M @ 256d (8M params) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M) | off | 100 | 89.6% | CPU only, 512 MB RAM |
| Base | `Qwen/Qwen3-Embedding-0.6B` @ 256d Matryoshka (600M) | `Alibaba-NLP/gte-reranker-modernbert-base` (149M) | off | 100 | 92.0% | CPU or GPU (T4 recommended), 4 GB RAM |
| Pro (+HyDE) | `Qwen/Qwen3-Embedding-0.6B` @ 256d Matryoshka (600M) | `Alibaba-NLP/gte-reranker-modernbert-base` (149M) | on (gpt-4.1-mini via OpenRouter) | 100 | 93.0% | CPU or GPU (T4 recommended), 4 GB RAM, LLM API key |

Per-tier bench scripts: `bench_truememory_edge.py`, `bench_truememory_base.py`, `bench_truememory_pro.py`.
