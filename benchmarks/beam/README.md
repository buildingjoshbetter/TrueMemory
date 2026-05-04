# BEAM Benchmark Results

Evaluation of TrueMemory Pro on [BEAM](https://arxiv.org/abs/2510.27246) (Beyond a Million Tokens), testing memory at conversation scales that exceed any model's context window.

## Results

### BEAM-1M (3-run validated mean)

35 conversations at 1M+ tokens each, 20 questions per conversation = 700 total.

| System | Accuracy | Correct | Answer model |
|--------|----------|---------|-------------|
| TM Pro (3-run mean) | **76.6%** | 536 | gpt-4.1-mini |
| Hindsight (published) | 73.9% | — | proprietary |

Individual runs: 75.0%, 78.1%, 76.6%

Per-category breakdown (3-run means):

| Category | Score |
|----------|-------|
| Preference following | 97.1% |
| Contradiction resolution | 91.4% |
| Information extraction | 91.4% |
| Summarization | 89.5% |
| Instruction following | 84.8% |
| Abstention | 82.4% |
| Knowledge update | 77.6% |
| Multi-session reasoning | 67.1% |
| Temporal reasoning | 64.8% |
| Event ordering | 19.5% |

### BEAM-10M (preliminary, single run)

10 conversations at 10M tokens each (~20,000 messages per conversation), 20 questions each = 200 total.

| System | Accuracy | Correct | GPU |
|--------|----------|---------|-----|
| TM Pro (single run) | **65.0%** | 130 | A100 80GB |

Per-category breakdown:

| Category | Score |
|----------|-------|
| Contradiction resolution | 90.0% |
| Knowledge update | 90.0% |
| Preference following | 85.0% |
| Summarization | 85.0% |
| Information extraction | 80.0% |
| Instruction following | 65.0% |
| Multi-session reasoning | 60.0% |
| Abstention | 60.0% |
| Temporal reasoning | 30.0% |
| Event ordering | 5.0% |

## Evaluation Config

| Parameter | BEAM-1M | BEAM-10M |
|-----------|---------|----------|
| Dataset | `Mohammadta/BEAM`, split `1M` | `Mohammadta/BEAM-10M`, split `10M` |
| Answer model | `openai/gpt-4.1-mini` | `openai/gpt-4.1-mini` |
| Answer max tokens | 500 | 500 |
| Judge model | `openai/gpt-4o-mini` | `openai/gpt-4o-mini` |
| Judge voting | 3x majority vote | 3x majority vote |
| Temperature | 0 | 0 |
| Retrieval top-k | 100 | 100 |
| GPU | T4 | A100 80GB |

## How to Reproduce

```bash
modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-...

# BEAM 1M
modal run --detach bench_truememory_pro_beam1m.py           # Full (35 convs)
modal run --detach bench_truememory_pro_beam1m.py --smoke    # Smoke (10 convs)

# BEAM 10M
modal run --detach bench_truememory_pro_beam10m.py           # Full (10 convs)
modal run --detach bench_truememory_pro_beam10m.py --smoke    # Smoke (3 convs)

# Download results
modal volume get locomo-results / ./results --force
```

## File Structure

```
benchmarks/beam/
  README.md                                    # This file
  bench_truememory_pro_beam1m.py                # BEAM 1M benchmark script
  bench_truememory_pro_beam10m.py               # BEAM 10M benchmark script
  truememory_pro_beam1m_run1.json               # 1M run 1 (75.0%)
  truememory_pro_beam1m_run2.json               # 1M run 2 (78.1%)
  truememory_pro_beam1m_run3.json               # 1M run 3 (76.6%)
  truememory_pro_beam10m_run1.json              # 10M run 1 (65.0%)
```
