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
ratio = str(min(0.08, 2.5 / total_gb)) if total_gb >= 16 else "0.19"
```

Machines with 16+ GB RAM use ratio 0.08 (or lower to cap at 2.5 GB).
Machines under 16 GB use ratio 0.19 as a floor — without this, an 8 GB
machine would get a 0.64 GB ceiling which crashes PyTorch (minimum ~1.0 GB
needed for the reranker model + workspace).

| Machine RAM | Ratio | MPS Ceiling | Note |
|-------------|-------|------------|------|
| 8 GB | 0.19 | 1.5 GB | Floor ratio — 0.08 would crash |
| 12 GB | 0.19 | 2.3 GB | Floor ratio — 0.08 would be 0.96 GB (too tight) |
| 16 GB | 0.08 | 1.3 GB | Standard ratio starts here |
| 18 GB | 0.08 | 1.4 GB | |
| 24 GB | 0.08 | 1.9 GB | |
| 32 GB | 0.078 | 2.5 GB | Capped at 2.5 GB |
| 48 GB | 0.052 | 2.5 GB | Capped at 2.5 GB |
| 64 GB | 0.039 | 2.5 GB | Capped at 2.5 GB |
| 96+ GB | <0.03 | 2.5 GB | Capped at 2.5 GB |

The 8/12 GB ceilings are higher than 16 GB because they need a higher ratio
to stay above the ~1.0 GB crash floor. This is intentional — the alternative
is PyTorch OOM on small machines.

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
