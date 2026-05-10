# Tier Selection Guide

TrueMemory ships with three tiers. All models are pre-installed. Switch anytime.

## Comparison

| | Edge | Base | Pro |
|---|------|------|-----|
| **LoCoMo accuracy** | 89.6% | 92.0% | 93.0% |
| **Embedder** | Model2Vec potion-base-8M (8M params) | Qwen3-Embedding-0.6B (600M params) | Qwen3-Embedding-0.6B |
| **Reranker** | MiniLM-L-6-v2 (22M) | gte-reranker-modernbert (149M) | gte-reranker-modernbert |
| **HyDE** | off | off | on |
| **Requires API key** | no | no | yes |
| **Best for** | low-resource machines, CPU-only | most users | maximum accuracy |

## When to use each

**Edge** — You're on a machine with limited RAM (<4GB), or you want the fastest possible search with no GPU. Accuracy is still strong at 89.6%.

**Base** — The recommended default. 92.0% accuracy with Qwen3 embeddings and a gte-reranker. No API key needed. Works offline.

**Pro** — Maximum accuracy at 93.0%. Adds HyDE query expansion, which uses an LLM to generate hypothetical answers and search with those. Requires an API key (Anthropic, OpenRouter, or OpenAI).

## Switching tiers

From the terminal:
```bash
truememory-ingest upgrade-tier base
truememory-ingest upgrade-tier pro
truememory-ingest upgrade-tier edge
```

Or via Claude:
```
"Switch TrueMemory to Pro tier"
```

Switching re-embeds all existing memories with the new model. This takes a few seconds for small databases, longer for large ones. No data is lost.

## API keys for Pro

Pro needs an LLM API key for HyDE query expansion. Supported providers:

| Provider | Key format | Set via |
|----------|-----------|---------|
| Anthropic | `sk-ant-...` | `ANTHROPIC_API_KEY` env var or setup wizard |
| OpenRouter | `sk-or-...` | `OPENROUTER_API_KEY` env var or setup wizard |
| OpenAI | `sk-...` | `OPENAI_API_KEY` env var or setup wizard |

Without an API key, Pro works identically to Base (HyDE is silently skipped).
