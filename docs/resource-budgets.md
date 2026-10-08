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

## Model-server request admission

The server bounds accepted work before reading a request body or decoding its
JSON. These are transport and backlog limits, not a process RSS limit. Decoded
Python objects, response serialization, model weights and native workspaces are
outside the byte accounting below.

| Variable | Default | Valid range | Meaning |
|----------|---------|-------------|---------|
| `TRUEMEMORY_MODEL_SERVER_MAX_HANDLERS` | 16 | 1..128 | Maximum admitted requests and worker threads |
| `TRUEMEMORY_MODEL_SERVER_MAX_REQUEST_BYTES` | 33554432 | 10485760..1073741824 | Aggregate advertised frame bytes reserved by admitted requests |
| `TRUEMEMORY_MODEL_SERVER_HEADER_TIMEOUT_MS` | 1000 | 100..30000 | Total time for authentication and frame header |
| `TRUEMEMORY_MODEL_SERVER_FRAME_TIMEOUT_MS` | 30000 | 100..120000 | Total authentication, header and body receive time from connection acceptance |

Values must be ASCII decimal integers in the stated ranges. Invalid values stop
server startup with an explicit configuration error; zero does not mean unlimited.
The existing per-frame limit remains 10485760 bytes (10 MiB).

With defaults, the server has at most 16 admitted requests and one accept-loop
staging socket: `16 + 1 = 17` accepted sockets. The operating system's listen
backlog is separate. Reserved request lengths satisfy
`sum(frame_bytes) <= 33554432`. Three maximum frames reserve
`3 * 10485760 = 31457280` bytes; a fourth would require
`4 * 10485760 = 41943040 > 33554432` and is rejected. Credits remain reserved
through receiving, decoding, queued inference, computation and response sending.
They return only after that work ends and its payload references are dropped.

Authentication and protocol version 1 are unchanged. After authentication, an
exhausted slot or byte budget returns an error with `error_code: server_busy`
and a fixed `retry_after_ms: 250` hint. The hint does not schedule a retry.
New clients raise `ModelServerBusyError` without retrying, restarting the daemon,
or loading a local model copy. Legacy clients can read the usual framed error;
a stalled sender may instead encounter a bounded transport failure. A rejected
body is briefly drained using a 4096-byte scratch buffer, with a 250 ms total limit.
Receive deadlines apply to the whole phase, so sending occasional bytes cannot
keep a slot indefinitely. Header and frame deadlines both start at connection
acceptance: a header that consumes 1 second leaves 29 seconds of the default
30-second frame budget. The client exception also exposes the fixed retry hint.

The legacy frame header contains neither operation nor deadline. Caller deadlines
are therefore checked immediately after bounded JSON decoding, before waiting for
inference or loading a model. Requests with no effective caller deadline have a
separate 120-second queue ceiling measured from transport acceptance. Once model
work starts, that queue ceiling does not impose a computation timeout. Caller
deadlines still apply while queued and between batches; a present, valid caller
deadline replaces the legacy queue ceiling, including when it is longer than
120 seconds. Slot and byte credits remain bounded in either case. Internal
admission state never comes from the request JSON.

Admitted single-text embedding requests retain the existing CPU fast path during
main-model contention. There is no reserved query capacity: at saturation even a
query or ping can receive busy. Shutdown closes transport sockets and wakes queued
lock waits. Active native inference cannot be preempted and retains its credits
until it returns; graceful interpreter exit also waits for active pool workers.
A valid client may half-close its write side while awaiting a response, so EOF
after a complete frame alone does not cancel queued work. Its caller deadline or
the legacy queue ceiling still bounds the wait.

## Complete-result allocation and recovery lifetime

The model server now checks whether a complete inference result can fit the
existing 10 MiB response frame before allocating its full result array. This is
an exact output and transport check, not a limit on total process memory.

For a float32 result with shape `S`, the arithmetic is:

- Raw array bytes: `R = 4 * product(S)`.
- Base64 bytes: `E = 4 * ceil(R / 3)`.
- JSON envelope bytes: `J`, measured using the actual field name, shape, dtype,
  protocol version and JSON formatting, with an empty base64 value.
- A successful response requires `E + J <= 10485760`.

For 256-dimensional embeddings, 7679 inputs require
`R = 4 * 7679 * 256 = 7863296`, `E = 10484396`, and `J = 101` bytes:
`10484396 + 101 = 10484497 <= 10485760`. At 7680 inputs,
`R = 7864320`, `E = 10485760`, and `J = 101`:
`10485760 + 101 = 10485861 > 10485760`. The second request is rejected before
loading its known built-in embedding model. A smaller inference microbatch
cannot make that complete response fit. The caller must split the request into
separate requests; successful responses still contain every input in order.

The built-in embedding constructors and their existing model2vec fallbacks have
known 256-dimensional output. The built-in scalar rerankers have one float32
score per input. These shapes are checked before model loading. An opted-in
custom embedding model's configured `truncate_dim` is an upper bound, and its
dimension getter can report that bound when its native width is unknown. Custom
rerankers may return multiple labels per pair. These unknown shapes are checked
against the actual first native result, before allocating the complete result
or converting that native array to float32. Every later slice must retain its
output dimensions and expected input count. Empty outputs retain their native
rank. Custom model loading and its first forward pass can still allocate memory
before this check; they are not covered by an estimated native-memory budget.

The response serializer independently measures actual array shapes before
making contiguous float32, raw-byte, base64 or full JSON copies. Oversized
results return an error rather than a partial result. Embedding models,
rerankers, precision, retrieval depth and wire format are unchanged.

Shared-model MPS recovery now leaves the failed exception scope before flushing
caches or starting CPU recovery. This releases the failed forward traceback's
references to temporary workspaces. Reranker recovery also drops both the cached
and local obsolete model references before constructing its CPU replacement.
Main inference ownership remains held through recovery and the failed-slice
retry. Sticky CPU state and deadline checks still apply, and completed slices
are not recomputed. This releases Python references; it does not guarantee that
a native allocator immediately returns pages to the operating system.

Model weights, native activation/workspace peaks, CPU transfer demand, normal
model replacement overlap, fast-encoder residency and serializer copies for
accepted results still require measured whole-process budget calibration.
This change introduces no new process-memory threshold, token estimator,
automatic recycling policy or claim that native allocation cannot overshoot.
