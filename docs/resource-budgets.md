# Resource budgets

## Scope and model ownership

The shared model server normally owns embedding and reranking models for all
MCP sessions. A contended single-query request can also load its existing CPU
fast encoder. A client loads models locally only when sharing is explicitly disabled.
Server unavailability is reported without loading local copies. Each process that actually constructs an MPS model configures
its own MPS allocator budget; a proxy does not initialize a local allocator.
Cache cleanup skips an unconfigured MPS allocator so CPU-only or proxy work
cannot consume allocator startup settings before a later MPS model load.

The budget below limits allocations managed by the PyTorch MPS allocator.
It does not cap CPU tensors, Python objects, resident memory, physical footprint,
swap, or the sum of multiple processes. CPU fallback has no MPS protection.
Whole-process admission, cancellation, and recovery limits remain separate
work tracked in #297. No fixed per-tier or per-session footprint is guaranteed.

Sharing is the default policy even when the endpoint is missing after cold
start, idle exit or a crash. Embedding and reranker getters return lightweight
proxies; endpoint absence or proxy failure never silently constructs local model
copies. Set `TRUEMEMORY_NO_MODEL_SERVER=1` before loading models for explicit local
mode. Existing model/device selection and cached-model lifetime are unchanged;
changing the environment is not a live mode switch.

### Startup and ownership

Clients coordinate startup with `model_server.start.lock` and an atomic
`model_server.start.json` generation record. Only one launch is started while
that process identity remains live. A caller timing out leaves the pending
child available to later callers. Child identity includes process creation time,
so a reused PID cannot keep a dead launch alive. A crashed launcher can be
replaced; a superseded managed child cannot publish an endpoint. These records
contain process coordination metadata, not memory contents.

The daemon's separate lifetime `model_server.lock` is the authority for binding
and reclaiming endpoint files. Clients do not remove socket, PID, port or token
files. A starter that never acquired bind ownership cannot remove a winner's
files. A PID file alone does not prove readiness. Readiness requires a bounded
protocol response, including the existing Windows loopback token exchange.
An authenticated busy response proves presence; a foreign protocol is surfaced
without restarting it or loading local models.

Startup lock waits, optional macOS app registration and readiness probes share
one monotonic budget: at most 30 seconds, further capped by an explicit caller's
remaining request time. Cosmetic app construction is skipped for short budgets.
Expired callers do not start new children. Existing legacy 120-second request
timeouts and the single autostart/retry contract remain unchanged. Startup or
inference unavailability is reported to the caller; inspect
`~/.truememory/model_server.stderr` and retry. A live but stalled launch is not
repeatedly replaced or signaled automatically.

Shutdown closes admission without claiming to preempt native inference. If
admitted or in-flight work still retains models, cleanup keeps its bind lock,
endpoint artifacts and model references until process exit. It does not wait
indefinitely inside cleanup, and that server instance cannot be restarted.
The next bind owner reclaims stale files after the old process exits. This
prevents a replacement daemon from loading models while the old daemon's native
work is still draining.

## One byte budget and one denominator

Let `P` be physical memory in bytes, `G = 1024**3`, and `R` be the bytes returned
by `torch.mps.recommended_max_memory()`. TrueMemory preserves its existing
intended default allocation policy, expressed explicitly in bytes:

```python
B = int(max(1.5 * G, min(0.08 * P, 2.5 * G) if P >= 16 * G else 0.19 * P))
fraction = B / R
torch.mps.set_per_process_memory_fraction(float(fraction))
effective_bytes = int(fraction * R)
```

`R` is Metal's recommended working-set size, which can differ from physical
RAM. PyTorch applies the ratio to `R`. The previous physical-RAM ratio could
therefore enforce a different byte limit than intended. The public setter
avoids exporting a generated HIGH ratio into child processes as if it were an
operator override. Float conversion can change the effective byte count by
rounding; telemetry reports that effective count.

| Physical RAM | Intended MPS budget | Calculation in GiB |
|---|---|---|
| 8 GiB | 1.52 GiB | max(1.5, 0.19 × 8) |
| 12 GiB | 2.28 GiB | max(1.5, 0.19 × 12) |
| 16 GiB | 1.5 GiB | max(1.5, min(1.28, 2.5)) |
| 18 GiB | 1.5 GiB | max(1.5, min(1.44, 2.5)) |
| 24 GiB | 1.92 GiB | max(1.5, min(1.92, 2.5)) |
| 32 GiB and above | 2.5 GiB | max(1.5, min(0.08 × RAM, 2.5)) |

For example, a 2.5 GiB budget with `R = 20 GiB` needs `2.5 / 20 = 0.125`.
With `R = 25 GiB`, it needs `2.5 / 25 = 0.1`. Both enforce the same intended
bytes. The throttler uses the effective bytes for memory level and growth:
warning at `used / effective >= 0.85`, critical at `>= 0.95`. For exactly
2.5 GiB, those boundaries are 2.125 GiB and 2.375 GiB. Physical RAM still
selects batch-size profiles; it is no longer a second MPS headroom denominator.

## Initialization and operator settings

All MPS transformer factories initialize this policy before constructing a
model, including explicit local embedding and reranking. CPU, CUDA,
Model2Vec, and proxy paths do not configure MPS. A process lock serializes
initialization, and an immutable snapshot is published only after the public
setter succeeds. Logs and throttler metrics expose intended, recommended,
and effective bytes, the policy source, and enforcement status.

| Setting | Default | Meaning |
|---|---|---|
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | Derived `B / R` via public setter | An explicit finite ratio from 0 through 2 is preserved. Positive values use `ratio × R` bytes; 0 explicitly disables the hard limit. |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO` | `0.0` | Existing policy retained before the first allocator-touching query. Zero disables adaptive commit and allocator garbage collection. Native calibration of this choice is still pending. |
| `TRUEMEMORY_DEVICE` | auto | Explicit `cpu` bypasses MPS setup; existing device selection rules apply. |
| `TRUEMEMORY_MODEL_SERVER_IDLE` | 300 | Seconds before idle model server exit. |
| `TRUEMEMORY_NO_MODEL_SERVER` | 0 | Set to 1 to request local model loading. |

Explicit LOW is also preserved, validated as finite and within 0 through 2,
and must not exceed a positive effective HIGH. With HIGH 0, PyTorch permits
LOW through 2. Settings are captured at first initialization; change them
before launch and restart to apply a different policy. Code embedding
TrueMemory should initialize this policy before any unrelated MPS allocation.
External host code may already have initialized the allocator. The public API
can set HIGH afterward, but it cannot read or change effective LOW. The snapshot
therefore labels LOW as `requested_low_watermark_ratio`, a bootstrap intention,
not verified native enforcement. Tests establish ordering for TrueMemory's
owned entrypoints; they cannot establish the earlier state of an external host.

Missing public APIs, invalid or unavailable byte metrics, conflicting ratios,
an unsupported derived ratio, or a failing setter stop MPS model construction
with a configuration error. Failed setup is not retried in the same process
because allocator initialization may already have occurred; the error directs
the operator to correct the policy or select CPU and restart. No success
snapshot or invented effective byte count is reported.

MPS budgeting requires PyTorch 2.5 or newer, which introduced
[`recommended_max_memory()`](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/mps/__init__.py#L119-L126).
Package metadata requires `torch>=2.5,<3` when `sys_platform == 'darwin'` and
`platform_machine == 'arm64'`. The general `torch>=2.0,<3` requirement remains
for other environments, including CPU and CUDA use. Older Intel Mac installations
can select `TRUEMEMORY_DEVICE=cpu` and restart; a compatible newer PyTorch wheel
is not guaranteed to exist for that platform. A missing MPS API stops MPS model
loading and reports the required version and the explicit CPU escape.

For unconfigured or explicitly unlimited MPS, the throttler treats memory
headroom and growth as unknown, so those readings cannot authorize a ramp.
It still reacts to known thermal pressure. Once configured, MPS telemetry
describes the process allocator, including when one model falls back to CPU
and another remains on MPS. MPS pressure can still back off shared admissions;
it never establishes a CPU memory limit. A newly initialized MPS allocator
invalidates an in-progress sample taken before that initialization.

## Model-fit and thermal calibration still required

This correction establishes units and enforcement setup. It does not show
that the retained budget accommodates every supported model or input. Models,
precision, dimensions, retrieval depth, and rerankers remain unchanged:
Edge uses Potion and MiniLM; Base and Pro use Qwen3-Embedding-0.6B and
gte-reranker-modernbert-base. Custom tiers retain their selected models.

Native calibration must measure both resident models, the optional CPU fast
encoder, bounded embedding and reranking workloads, idle periods, and sticky
CPU recovery. Record model identity and dtype, token counts, batch sizes,
MPS current and driver allocations, recommended/effective bytes, process RSS
and physical footprint, swap change, latency, CPU time, and thermal pressure.
These memory measurements overlap and must not be added together. Verify
retrieval quality and fit before changing the retained byte policy or LOW.
Linux or CUDA tests cannot establish MPS behavior or Mac thermal performance.

## PyTorch references

- [Recommended working-set bytes](https://docs.pytorch.org/docs/stable/generated/torch.mps.recommended_max_memory.html)
- [Public allocation-fraction setter](https://docs.pytorch.org/docs/stable/generated/torch.mps.set_per_process_memory_fraction.html)
- [HIGH and LOW watermark semantics](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)
- [Allocator initialization](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/mps/MPSAllocator.mm)

The public contracts were checked against PyTorch 2.12 documentation and the
installed 2.14.1 Python API source used for isolated validation. Native MPS
initialization and model-fit measurements remain a separate validation gate.

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

## Tier rebuild batch durability

The tier-switch worker computes and serializes both completion and separation
embeddings before starting its batch write transaction. Each output must contain
one vector per input message. Model choice, dimensions, precision and the existing
encoding ownership wrapper remain unchanged. Completion output arrays are released
after serialization; the two serialized batches remain in memory until publication.
This is one throttled batch of staging, not a corpus-wide staging operation or a
new process-memory bound.

The vector pair and an existing cache registry's progress checkpoint commit in one
transaction. A failure in either encode, serialization, insertion, checkpoint or
commit leaves the previous committed batch intact. OOM retry uses the same input
offset after rollback, so it cannot retry on top of partial completion vectors.
Exceptions used for cancellation also roll back before propagating. A cancel flag
retains the existing behavior of finishing the current batch before stopping.

Full-rebuild clearing removes both target tables' rows and resets an existing
checkpoint in one transaction. A failed clear reports failure without proceeding
to encoding. A target with no registry row retains the existing full-restart
behavior; this change does not invent model metadata or create a new resume policy.

`run()` requires its manager-owned connection to have no active caller transaction
before initialization. The batch helper uses a savepoint if explicitly called
inside a caller transaction and does not commit unrelated caller work. Status
writes use a separate transaction after the batch checkpoint is committed. Status
failure rolls back its own changes and cannot cause a committed batch to be retried.
Database rollback failure still requires caller recovery; no transaction helper
can promise successful rollback after SQLite itself rejects that operation.

The regression suite uses synthetic encoders with ordinary SQLite tables and the
same cases against real sqlite-vec tables when the extension is available. These
checks establish paired-write and retry behavior, not native MPS OOM behavior,
thermal improvement or model-performance measurements. Whole-rebuild streaming
and corpus loading remain separate work.
