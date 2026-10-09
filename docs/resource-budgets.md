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

## Adaptive policy applicability in the model daemon

The daemon applies adaptive MPS/thermal admission when MPS is selected, a
process MPS budget has been published, or the platform is Darwin. Darwin
remains applicable before model loading, and a CPU fallback does not bypass
monitoring when another model can still use MPS. Unknown required sensors
remain conservative; an empty set of required sensors is never healthy ramp
evidence. Applicability is checked for each request.

On Linux/Windows CPU or CUDA with none of those policies, the daemon keeps
the validated caller/server batch bound and skips adaptive slow-start, pacing
and adaptive cache flushing for that request. The hard ceilings remain 32
embedding inputs and 64 reranker pairs; a caller requesting 8 remains bounded
by 8. For 100 pairs, `ceil(100 / 8) = 13` normal microbatches replace
the erroneous 100 singleton microbatches. Failed-slice retries can add native
invocations. Actual runtimes still require measurement. Request deadlines, inference ownership and output/admission
limits continue to apply.

This corrects the daemon regression observed in the first native comparison.
It does not change the tier worker's adaptive/OOM policy, models, precision
or retrieval depth. The initial comparison's numerical failures and latency
regressions remain recorded until separate native validation resolves them.
Linux/CUDA behavior does not establish MPS memory or thermal performance.

## Server target identity prerequisite (#795)

An explicit embedding request no longer changes the daemon's default model,
dimension or generation. The same raw tier keeps its loaded snapshot, including
an empty tier whose default changed while its initial load was in progress.
Main and fast requests retain that captured identity. A cache miss or a
different tier request resolves the current selection; changing the global
default alone does not invalidate an already published empty-tier snapshot.
`base`, `pro` and `qwen3_256` reuse the same loaded Qwen3 instance; `edge` and
`model2vec` reuse the same loaded Model2Vec instance. These are the two
deterministic identities already allowed by the fast lane. Alias updates
publish one immutable cache snapshot. The later resident-replacement checkpoint
supersedes the original failure behavior: distinct loads release the old cache
before construction, and a failed replacement leaves that cache empty.

The removed `qwen3` name still raises the existing migration error, including
case and whitespace variants. This validation does not mutate the default or
adopt broader alias normalization from the active-model setter.

Explicit custom and other configuration-dependent cache entries retain their
existing raw-key reuse policy, including a custom entry that loaded a built-in
identity or Model2Vec fallback before its configuration changed. They gain no
new identity or dimension attestation, and the legacy server fallback for `minilm` and `bge-small`
remains unchanged. Result-size preflight uses the same cache-selection rule
as loading. Fast-lane inference continues to follow its captured loaded-model
snapshot, without resolving current global settings.

This correction preserves inference/state lock ownership, request deadlines,
native batch ceilings and sticky CPU recovery. It does not bound the lifetime
of separately retained old model references: the previous main instance can
still be alive while a distinct replacement is constructed, and the optional
fast CPU instance remains separately owned. A managed preparation lifetime and
an explicit protocol guarantee for older daemons are subsequent prerequisites.
No tier-switch caller, local loader, target table or active configuration is
changed by this checkpoint; issue #795 remains open.

### Qwen global input order within bounded calls

Multi-batch embedding requests for the cached `qwen3_256` model now preserve
modern SentenceTransformers' global length ordering when the resolved effective
batch limit is exactly 32, every input is a string, `_input_length` and
`_can_flatten_inputs` are available, and input
flattening is disabled. The verified SentenceTransformers 6.1.0 implementation
measures string characters before prompt preprocessing. The server leaves
prompt resolution, instruction processing, tokenization, dtype and the model's
encode path unchanged. Legacy interfaces, other model identities, non-string
inputs and flattened modes retain the existing bounded slice path; this
checkpoint does not certify their global-order parity or change reranking.

The server sorts occurrence indices once. For each slice it compensates for
the inner encode call's equal-length permutation, then scatters returned rows
directly to their original positions. Duplicate text values keep distinct
indices. Single-input requests and requests fitting one effective batch keep
their existing call path. The effective limit remains fixed once per request,
bounded by caller, server and applicable adaptive ceilings. Omitted limits and
oversized limits clamped to 32 use global ordering when the adaptive ceiling
also preserves 32 and the other eligibility conditions hold. Smaller effective
limits, including explicit eight and
adaptive reduction to eight, retain the prior contiguous slice membership.
This preserves the verified native32 default ordering without extending it to
layouts with unresolved numerical differences. Explicit eight remains capped
at eight; retaining its prior behavior does not waive its failed numeric gate.

Output preflight runs before ordering work. For the known 256-dimensional
float32 result, 7,679 rows require 10,484,497 response bytes; 7,680 require
10,485,861 bytes, exceeding the existing 10,485,760-byte frame limit. Planning
adds an O(N) index array and temporary length keys, plus O(batch size) slice
indices. It creates neither a full-request token tensor nor a second complete
output array. Existing frame/admission limits remain in force; these are not
a total process-memory guarantee.

Deadlines are checked before and after ordering and before each encode and
retry. Main inference ownership spans planning, all slices and recovery. An
OOM retry retains the exact failed slice and occurrence indices; completed
rows are retained, and the cursor advances only after a successful result
write. The existing CPU recovery retry releases only the state lock.

Stdlib AST/fake tests cover native input membership and order, ties,
duplicates, Unicode, empty strings, default prompts, scatter, bounds, deadline
expiry and recovery. Removing global ordering, tie compensation or scatter
fails independent negative controls. Equal native shapes alone are not parity
evidence. Actual ordered token features, raw numeric gates, latency and native
memory still require the bounded native diagnostic before acceptance.

### ModernBERT reranker global order at the 64-row limit

Reranking now preserves modern CrossEncoder's global raw-pair length order
only for the cached `Alibaba-NLP/gte-reranker-modernbert-base` model, plain
two-string pairs, available `_input_length` and `_can_flatten_inputs` helpers,
disabled input flattening, an effective limit of 64, and more than 64 pairs.
The inspected SentenceTransformers 6.1.0 implementation sorts by the combined
query and document character lengths before prompt preprocessing. Its inner
sort and score restoration remain active. Default prompts, tokenizer behavior,
activation, model identity and dtype are unchanged. Older interfaces, custom
models, other input forms, flattened modes, requests fitting one batch and
lower caller or adaptive limits retain their previous bounded slice path.

Occurrence indices are sorted once. Each supplied slice compensates for the
inner equal-length permutation; scores scatter directly into their original
positions, including separate occurrences of duplicate pairs. The plan adds
O(N) indices and temporary length keys, with O(batch size) slice indices. It
does not create a corpus token tensor or a second complete score array.
Output preflight, once-per-request admission and all caller, server and
adaptive ceilings remain in force.

Deadline checks bracket planning and slice sorting and precede each prediction
and retry. The inference owner spans the plan, slices and recovery. On OOM,
the failed slice and its indices stay fixed, completed scores remain in place,
and the cursor advances after successful publication. For this planned path,
CPU replacement uses the actual resolved model name even when the request
omitted it; a changed default cannot replace that model during retry.

Stdlib tests exercise actual server methods with synthetic CrossEncoder
ordering, cyclic ties, duplicate occurrences, Unicode, empty strings, prompts,
caps, deadlines and later-slice OOM. Mutations removing global order, inverse
tie compensation or scatter fail the output and membership checks. These
tests do not establish native numeric or performance acceptance. The 64-row
reference still requires ordered token-feature, raw-score and timing checks
on the pinned runtime; lower-limit numeric failures are not waived.

Unplanned reranker calls retain the positional result-writer interface on
success and OOM retry. Only planned slices supply occurrence indices. The
regression checks both explicit-eight and default single-batch calls with a
positional-only writer wrapper, including a later failed slice and its retry.

### Resident model replacement and stale CPU clones (R13 / #297)

Distinct embedding and reranker replacements release the obsolete cached
instance before starting the new constructor. This is a memory-first policy:
if construction fails, the old warm cache is gone and a later request for its
identity must reload it. Compatible aliases still reuse the same instance.
Main inference ownership continues to cover loading, inference and recovery.

Each main embedding residency has an opaque generation. Alias retags preserve
it; unloading and reloading the same identity creates a new generation. Fast
CPU construction captures only the resolved identity and generation, without
retaining the main model snapshot. It drops any obsolete CPU cache before
construction. A clone built across a generation change cannot publish into the
current cache, including A -> B -> A changes. Its already-captured request may
finish on that identity, after which the uncached clone is released.

Main publication tries to retire an idle stale fast cache without waiting for
the fast owner. Active fast inference keeps its instance until completion or
failure; the owner retries retirement after releasing its slot. A short
residency metadata lock closes publication and retirement races. Its order is
main state -> residency or fast -> residency, never the reverse. Constructors,
inference and model destruction run outside that metadata lock. Fast cache
publication uses a nonblocking attempt; metadata contention leaves a transient
clone for the captured request. Models are never moved as part of retirement.

This checkpoint limits obsolete Python cache references. It does not establish
a whole-process byte budget, bound native allocator caches, eliminate active
main/CPU overlap, change OOM recovery, or calibrate peak memory or heat. Model
identities, dtypes, dimensions, batch limits, queue limits and result caps are
unchanged. Stdlib tests exercise the actual loaders and handlers with weakrefs,
constructor and inference barriers, failed replacement, deadlines and identity
generation changes. Native memory and whole-process admission remain separate
validation work.

### Cooperative process RSS admission checkpoint

The shared model daemon accepts `TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB` as an
integer MiB setting, from 0 through 2,147,483,647. Its default is 0, which
disables this admission policy and performs no RSS measurements. This separate
setting does not change MCP's historical `ru_maxrss` reporting or the existing
`TRUEMEMORY_MAX_RSS_MB` setting. A finite deployment default remains uncalibrated.

For a nonzero setting, the byte budget is B = configured MiB * 1,048,576.
Each admission reads current process RSS U from `psutil.Process().memory_info()`.
U < B admits the stage; U >= B refuses it. For example, a synthetic 1 MiB budget
admits 1,048,575 bytes and refuses both 1,048,576 and 1,048,577 bytes. An unavailable
sensor, sensor exception, missing value, boolean, noninteger or nonpositive RSS
also refuses when enabled. A later current reading below B can admit work even
after a higher historical peak. RSS is not added to GPU allocator or physical
footprint measurements, which can overlap it.

Checks occur under the existing main inference/state or fast encoder owner:
after obsolete cache references are released and before model construction,
after construction before cache publication, before each encode/predict slice,
before embedding CPU transfer and before CPU retries. Recovery records sticky
CPU selection before possible refusal. A refused embedding transfer invalidates
the failed accelerator cache, so the next request cannot bypass CPU selection
through a cache hit. Reranker recovery retains its release-before-replacement
behavior. Refused requests return `server_busy` with the existing 250 ms retry
hint, without partial vectors or scores. Fast-lane refusal propagates directly
instead of falling through to main inference. Ownership release, deadlines,
fast generation fencing, models, dtypes, dimensions, result ordering and
existing request/batch/result limits are preserved.

RSS sensor work consumes the request's existing monotonic deadline. The daemon
checks that deadline before and after sampling, and again after device setup
before main constructors. If sensing both expires the request and returns an
unavailable or overbudget value, deadline expiry takes precedence. No constructor,
transfer or inference retry starts for that expired request. Expiry while
admitting an embedding CPU transfer still invalidates its failed MPS cache and
retains sticky CPU selection. Disabled admission performs no sensor reads and
retains the constructor deadline checks.

This is opt-in sampled admission, not a hard memory ceiling or a reservation.
Concurrent main and fast lanes can each pass a sample, and one constructor,
transfer, native call or result allocation can grow beyond B before the next
sample. Native allocator caches and unified-memory accounting remain outside
this policy. Synthetic tests cover thresholds, invalid sensors, constructor and
slice crossings, both CPU recovery routes, refusal without fallback, and zero
sensor reads when disabled. Finite-default calibration, workspace reservations,
system headroom, bounded recovery and actual Mac/MPS memory and thermal gates
remain pending. This checkpoint does not establish that the original 26.24 GB
Mac symptom is fixed.

### Failed embedding transfer retention checkpoint

If embedding recovery's CPU transfer fails or is cancelled, the daemon drops
the cached embedding snapshot only when it still names that failed instance.
The original exception propagates unchanged, with no extra load, retry or
partial successful result. Sticky CPU selection remains set; the next admitted
request rebuilds the same requested model on CPU. An unrelated replacement
snapshot survives. Existing generation fencing retires idle obsolete fast
clones and lets active captured work finish before retirement.

Synthetic tests cover ordinary errors, memory errors and cancellation, retained
identity, ownership release and weak references after exception tracebacks are
released. This bounds failed Python cache retention, not native allocator
reclamation, CPU-transfer peak memory or the original Mac memory incident.

### Certified inactive embedding targets (#795)

`EmbeddingTarget.capture(tier)` freezes the requested public tier, effective
model ID, dimension and table group. Its exact completion/separation table pair
is derived from that validated group. `prepare_embedding_target(target)` yields
an encode-only context lease without changing the active `vector_search` model,
dimension or generation, config, environment, metadata or registry. Daemon cache
residency generations still change when resident models are replaced. Call preparation outside SQLite
transactions; this API has no connection argument and cannot discover an
unrelated transaction held by its caller. No tier-switch caller adopts it yet.

Shared preparation uses additive `prepare_embed_target_v1` and
`embed_target_v1` operations. Older daemons must reject these operations; the
client raises a protocol error and never falls back to ordinary embedding.
Every successful response carries the exact descriptor. Preparation encodes one fixed
synthetic validation string under the existing inference/state owners, batch
policy, cooperative RSS admission, deadline and CPU-recovery path. Every actual
output slice must have the captured width, including CPU retries. Model dimension
getters and configured truncation bounds alone cannot certify a target. Empty
encode requests also validate the descriptor and a probe, returning shape
`(0, dimension)` only after successful certification.

The daemon uses its existing single main cache and release-before-load policy;
prepared requests do not add a target cache or fast CPU clone. Deterministic
built-ins can reuse the same resident weights. Legacy custom cache labels are
insufficient evidence because their constructor may have fallen back. Strict
custom construction uses frozen arguments and rechecks configuration/download
permission after ownership waits and before construction. It never relabels a
fallback as the requested model. Legacy `minilm`, `bge-small` and removed `qwen3`
identities are rejected by this new API; their ordinary loader behavior is
unchanged. A receipt certifies the embedding identity at that operation, not
persistent residency, a source generation or database readiness. Later requests
may reload after eviction or reject changed configuration.

Explicit local mode permits one outstanding preparation lease per process. It
borrows a matching active built-in model, or owns at most one additional target.
Construction and inference do not hold the active-model state lock. Local
encoding shares the existing exact-instance ownership registry; optional
deadlines cover state, lease and model-owner admission and are checked again
before native work. The ownership helper's ordinary callers retain unbounded
waiting. Cleanup unregisters users even after timeout. Local target encoding
does not introduce another OOM-retry policy. Closing rejects new calls before
waiting for active encoding, clears its model reference, then releases the
preparation slot. An interrupted closer can retry, and active encoding drains
the pending close itself. Same-target leases do not get independent native
owners. This is a reference-count policy, not a byte bound or a guarantee that
external references and exception tracebacks have disappeared.

`init_prepared_target_tables(conn, target)` creates or validates only the exact
named pair in `main`. It rejects temporary objects shadowing either name and
requires the canonical `vec0` float dimension and cosine metric. Existing table
contents are neither cleared nor certified. A failed pair rolls back together;
an existing caller transaction remains open under a savepoint. No active-table
migration or active metadata write occurs. Future publishers must retain main
schema qualification or recheck shadowing and must separately attest source,
target generation and content identity.

Synthetic tests use stub models and in-memory SQLite transaction controls. They
do not measure native loading, MPS memory, allocator release or latency. The
tier manager still retains its complete source list; bounded source enrollment,
truthful activation and the original Mac memory incident remain unresolved.
