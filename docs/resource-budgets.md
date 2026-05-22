# Resource Budgets by Tier

TrueMemory's memory footprint varies by tier. These budgets were established
in v0.7.0 after extensive benchmarking on Apple Silicon Macs.

## Architecture

All tiers use a **shared model server** (`truememory-model-server`) that loads
models once and serves all MCP sessions via a Unix domain socket. Each MCP
session is a lightweight proxy (~80 MB) that delegates model inference to the
shared server.

## Per-Tier Budgets

| Tier | Model Server | Each MCP Session | 5 Sessions Total |
|------|-------------|-----------------|------------------|
| **Edge** | ~500 MB | ~80 MB | ~900 MB |
| **Base** | ~1.5 GB | ~80 MB | ~1.9 GB |
| **Pro** | ~1.5 GB | ~80 MB | ~1.9 GB |

## MPS Watermark

The PyTorch MPS memory watermark prevents over-allocation. The actual code
in `model_server.py`:

```python
ratio = min(0.08, 2.5 / total_gb) if total_gb >= 16 else 0.19
ratio = max(ratio, 1.5 / total_gb)  # never below 1.5 GB ceiling
```

Two rules: (1) machines under 16 GB get ratio 0.19 as a floor to avoid
crashing PyTorch, (2) a final clamp ensures no machine ever gets a ceiling
below 1.5 GB regardless of RAM size.

| Machine RAM | MPS Ceiling | Note |
|-------------|------------|------|
| 8 GB | 1.5 GB | Floor ratio (0.19) |
| 12 GB | 2.3 GB | Floor ratio (0.19) |
| 16 GB | 1.5 GB | Clamped up from 1.3 GB |
| 18 GB | 1.5 GB | Clamped up from 1.4 GB |
| 24 GB | 1.9 GB | Standard (0.08) |
| 32 GB | 2.5 GB | Capped at 2.5 GB |
| 48 GB | 2.5 GB | Capped at 2.5 GB |
| 64 GB | 2.5 GB | Capped at 2.5 GB |
| 96+ GB | 2.5 GB | Capped at 2.5 GB |

No machine gets below 1.5 GB. The curve is monotonically non-decreasing
from 16 GB upward.

Users can override via `PYTORCH_MPS_HIGH_WATERMARK_RATIO` environment variable.

## What Consumes Memory

- **PyTorch runtime**: ~800 MB (loaded by model server)
- **Embedding model** (Base/Pro): Qwen3-Embedding-0.6B ~600 MB
- **Embedding model** (Edge): model2vec/potion-base-8M ~30 MB
- **Reranker** (Base/Pro): gte-reranker-modernbert-base ~300 MB on MPS
- **Reranker** (Edge): ms-marco-MiniLM-L-6-v2 ~22 MB
- **MPS GPU workspace**: varies by watermark ratio
- **Each MCP session**: Python + SQLite + protocol handling ~80 MB

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | auto | MPS memory ceiling as fraction of RAM |
| `TRUEMEMORY_MODEL_SERVER_IDLE` | 300 | Seconds before idle model server exits |
| `TRUEMEMORY_NO_MODEL_SERVER` | 0 | Set to 1 to disable shared model server |
| `TRUEMEMORY_MAX_RSS_MB` | 0 | Reported in stats (informational) |
