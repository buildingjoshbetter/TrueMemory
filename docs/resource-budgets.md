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

## Episode and full-text index maintenance

The `messages_au` trigger replaces an FTS row only when its message ID or an
indexed value changes: content, sender, recipient, category or modality. The
comparison is NULL-safe and also detects updates through SQLite's rowid aliases.
Changes to episode IDs, timestamps and other unindexed metadata leave FTS alone.
Opening an older database replaces the unconditional trigger atomically once;
subsequent opens check its definition without repeating trigger DDL or rebuilding
the index. Concurrent openers recheck the definition after obtaining the writer
lock. A failed replacement restores the previous trigger.

Episode detection keeps the existing timestamp ordering and parsing: timestamps
are sorted lexically, timezone suffixes are stripped before computing gaps, and
only a gap strictly greater than six hours starts another group by default.
Invalid timestamps retain the previous grouping behavior. Empty and NULL
timestamps are ineligible. This change does not correct timezone semantics.

Exact member sets retain their episode IDs when the stored member count also
matches. An addition, deletion, merge or split creates new IDs for the affected
groups; unaffected groups retain theirs. Changed bounds or counts update episode
metadata, and only changed message assignments are written. Obsolete episodes
and assignments on ineligible messages are cleared, including when no eligible
messages remain. Source messages and their full-text search results are preserved.

Nonempty derived episode summaries are cleared, as in the previous detector.
Equal member IDs cannot prove that source content was unchanged, so retaining
those summaries would risk stale derived text. Empty summaries cause no write.
No source content, revision counter or dirty-ID queue is added to episode metadata.

Detection reads one deferred SQLite transaction snapshot before writing its
changes. With no caller transaction, it owns the final commit and retries a stale
WAL snapshot at most twice: `1 initial attempt + 2 retries = 3 attempts`. Other
errors propagate after rollback. Inside a caller transaction it uses a savepoint,
never commits unrelated writes, and leaves snapshot conflicts for the caller to
resolve. Connections must not be used concurrently by multiple callers. WAL
readers can continue searching while episode changes await commit; another writer
still shares SQLite's single writer lock.

The synthetic 12-message regression checks `12` first-pass message assignments,
`0` FTS replacements from those assignments, and `0` row writes on the next
unchanged pass. It also checks real FTS results, transactional rollback and reader
progress during a held writer transaction. These are correctness and write-count
checks, not measured production latency or memory improvements. Detection still
reads the eligible timestamps, existing assignments and episode metadata on every
pass, with work and Python bookkeeping proportional to the corpus size. Database
size, WAL growth and duration on larger corpora still need measurement; scheduling
and database-wide maintenance coordination remain separate work.

## Surprise-index transaction lifetime

Surprise rebuilding reads messages in the existing `timestamp, id` order and
computes scores and accumulated fact counts before opening its publication
transaction. Foreground writers can proceed during that computation. A caller
that already owns a write transaction keeps that transaction and its locks;
the builder never commits or restarts it.

Publication uses a savepoint. It acquires SQLite writer ownership with a
no-row delete, then compares the exact ordered source IDs, content and
timestamps with the computation's input. These are all source fields used by
surprise scoring. A concurrent update, deletion, timestamp change or insertion
rejects stale output with `sqlite3.OperationalError` and a retry message.
A stale caller-owned WAL snapshot instead fails at SQLite's write upgrade;
the caller retains responsibility for restarting its own transaction. There
is no automatic retry loop or new cadence threshold.

Only a matching source snapshot permits the complete clear and bulk insert.
Both writes share the savepoint, so computation failure, partial insertion
failure and cancellation preserve the previous scores even if the caller
later commits unrelated work. Successful empty input clears obsolete scores.
An outer caller transaction remains open; without one, releasing the
savepoint commits the complete generation. The table-creation helper also
leaves caller-owned work uncommitted.

Returned scores retain their full precision. Stored scores retain four-decimal
rounding, and fact counts and chronological accumulation are unchanged. This
change still computes the complete corpus and validates that complete source
under writer ownership. It does not add incremental surprise semantics,
maintenance scheduling, revision schema or a claim of bounded corpus memory.

## Cluster cache publication

Clustering reads embeddings, source fields and categories from one SQLite
snapshot. For an owned transaction, it closes that read snapshot before vector
normalization, HDBSCAN, centroid arithmetic and serialization. These computations
open no writer transaction. HDBSCAN parameters, normalized inputs, noise label
`-1`, centroid arithmetic and sorted distinct session categories are unchanged.

Publication obtains SQLite writer ownership, validates the input state again,
then replaces assignments and centroids in one transaction. Validation includes
raw vector bytes and row IDs; source IDs, content, sender, recipient, timestamp,
category and modality; active vector table/schema; the selected cache registry
row; embedder metadata/build marker; and runtime model, dimension and tier group.
A typed, length-prefixed SHA-256 fingerprint allows streaming validation without
retaining another full vector matrix. It is a content fingerprint, not a durable
source-generation counter. Unrelated metadata updates do not invalidate it.

If inputs changed, the vector index is marked in progress, a vector has no source
message, or the existing model lock is busy at publication, clustering raises an
error and retains the previous cache. There is no automatic retry. A genuinely
empty, completed vector table clears both cache tables atomically. Missing
HDBSCAN remains an error, including for empty input.

Publication tries the model lock without waiting after acquiring the database
writer lock. It holds that model lock through an owned commit or caller savepoint
release, then releases it on success or failure. Native model loading cannot make
publication wait while it holds the writer lock. Computation, insertion, commit,
savepoint-release and cancellation failures roll back this operation's changes,
so a later caller commit cannot publish an incomplete replacement. Table creation
also participates in publication instead of implicitly committing caller work.

An existing caller transaction remains caller-owned. Its preexisting writer lock,
if any, necessarily remains held during computation; this operation does not
commit it to release that lock. A stale caller WAL snapshot raises rather than
being restarted. The caller is responsible for keeping source and model state
consistent until its later commit. The fence establishes consistency at
publication, not absolute freshness through arbitrary later caller changes.

Snapshot capture and publication validation still scan the input rows; validation
under the writer lock is proportional to the corpus size. Embeddings and normalized
vectors still occupy their existing NumPy arrays. Synthetic SQLite tests establish
rollback, writer progress during paused computation and atomic reader views.
Actual HDBSCAN runtime, validation duration, WAL growth, native peak memory and
retrieval quality on representative corpora remain release measurement gates.

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

## Maintenance coordination foundation

The storage schema now records transactional source revision primitives and
initial pending layer states. This foundation does not yet replace engine
scheduling or run builders automatically. The existing threshold remains 25.
It does not establish that the current engine already coalesces maintenance.

`read_source_revision(conn)` reads an immutable token from the caller's SQLite
snapshot without committing: epoch, revision, insert count, correction count,
historical maximum message ID and last nonappend revision. A caller sees its own
uncommitted changes; rollback restores every counter. Use committed snapshots
when issuing a durable freshness claim. Existing rows bootstrap with their
maximum ID and zero historical event counts, since their mutation history is
unknown. Missing or replaced tracking triggers start a new epoch.

Insertions increment revision and insert count. Deletions and real updates to ID,
content, sender, recipient, timestamp, category, modality, directive or metadata
increment revision and correction count. Comparisons are NULL-safe. Equal-value
updates and derived episode/emotional/separation fields do not self-dirty. Rowid
aliases are included. No message text, deleted content or per-event queue is added.

An insertion above the historical maximum advances that maximum. An insertion
into a lower ID hole, a replacement at an existing ID, a source update or a delete
advances the nonappend revision. The maximum never decreases within an epoch.
`new.is_append_only_since(old)` certifies that the earlier ID range is unchanged;
it does not certify chronological order. A higher ID may have an older or tied
timestamp. Tokens from different bootstrap epochs cannot certify continuity.
Degraded legacy schemas remain open with tracking unavailable until their source
columns can be migrated; they cannot issue a misleading revision token.

`maintenance_owner(path)` provides synchronous ownership, shared with
`get_coordinator(path).request(work)`. Canonical paths and symlink aliases share
one in-process claim and one nonblocking process-owned file lock beside the
database. A nested synchronous operation on the same worker thread reuses its
outer ownership token. Other threads or processes receive busy. The lock file is
not deleted to signal completion. Process death releases the real lock; forked
children close inherited owner descriptors and must obtain their own coordinator.
Live database replacement through another path and hard-link aliases are not
supported concurrency mechanisms.

Workers open and close their own connections. Ownership remains held through
work and connection teardown, including cancellation and interrupted thread
startup. Cancellation signals a phase boundary; it cannot preempt native work.
A callback must finish its transaction explicitly; unfinished writes are rolled
back and reported failed. Error status stores only a bounded exception category,
not exception text, source content or database paths.

Private `:memory:` databases cannot be reopened by a separate worker.
Asynchronous requests stay explicitly `pending_in_memory` without creating a
thread or file. Synchronous callers use the original connection and remain
responsible for its serialization. The engine's existing manual path is unchanged
in this foundation checkpoint. Per-layer freshness enforcement, scheduler routing,
vector dependency generations and measured maintenance performance remain later
integration work; initially pending rows are not success claims.

## Composable summary and landmark publication

Monthly summary refresh owns only `monthly` and `entity_monthly` rows. Empty
eligible input removes those periods while preserving structured facts, opt-in
entity-profile sheets and other summary producers. Sentence selection, salience,
entity thresholds, dates and returned counts retain the existing rules.

The shared consolidation savepoint now covers its terminal `RELEASE` as well as
the writes. A failed outermost commit or nested release rolls back that builder's
changes before cleanup releases the savepoint. If SQLite rejects rollback, the
error propagates without attempting a release that could commit failed output;
the connection's owner must recover or close that transaction. This applies to
summaries, structured facts and contradictions without committing caller work.

Landmark detection collects and computes its complete event rows before writer
admission. It then compares the exact ordered source IDs, content, sender,
recipient and timestamps while holding the writer, before replacing events.
Concurrent source changes reject publication. A stale caller-owned WAL snapshot
raises at write upgrade and remains the caller's transaction. Pattern order,
one-event-per-message selection, context clipping and related-entity limits are
unchanged. Empty source clears previous events atomically. Write, release and
cancellation failures preserve the previous complete output; later caller commit
cannot expose a partially replaced generation after successful rollback.

These builders can compose inside an owned transaction, but this checkpoint does
not enable the scheduler or claim durable layer freshness. Summary builders still
need the runner's source snapshot to fence concurrent changes during computation.
Landmarks still stage the eligible corpus and validate it under writer ownership;
no corpus-memory or constant-time publication bound is claimed.

The future layer runner must commit any `running` diagnostic in a separate short
transaction before opening its read snapshot. It must not update attempt metadata
and retain that writer transaction during computation. The actual attempted
source/dependency token is recorded at terminal publication or, after rollback,
in a separate failure-status transaction. Successful output and successful
provenance must commit atomically. A cluster run must retain validated runtime
model ownership through that outer commit; an inner builder savepoint release
alone does not supply that guarantee.

## Durable nonvector layer runner

`run_layers` now provides explicit, database-owned execution for summaries,
structured facts, contradictions, surprise, episodes and landmarks. It requires
a dedicated connection with no active caller transaction. Existing engine startup,
automatic/manual consolidation and retrieval paths are not routed through it yet;
this checkpoint neither disables nor certifies clusters or Dunbar relationships.

Each layer records its last attempted source epoch/revision, dependency and insert
count separately from its successful provenance. Dependency keys include their own
builder version, so a failed new-version attempt cannot relabel an old success.
Missing or untrusted layers get an initial attempt even when historical input has
zero tracked inserts. Empty input can establish durable `success_empty`.

After that initial attempt, pure append work becomes eligible at
`current.insert_count - attempted_insert_count >= 25`. Twenty-four committed
inserts remain pending; the twenty-fifth qualifies across connection and engine
lifetimes. Rolled-back inserts count zero. A changed epoch, dependency or newer
nonappend revision permits a new attempt; the latter covers edits, deletes and
same-ID replacements. An unchanged failed/unavailable attempt is suppressed until
explicit force. A failed or unavailable initial attempt also establishes the
25-insert baseline even though successful output remains untrusted. Each selected
layer is attempted at most once per run. New changes
remain pending and do not start an immediate retry loop.

Running diagnostics commit before the layer's deferred source snapshot begins.
The builder computes without a diagnostic write transaction held open. Output,
successful provenance and the terminal attempt token/count publish in one outer
commit. Source snapshot upgrades and dependency checks reject stale publication.
Failure rolls back output first, then records only the actual captured attempted
token and a bounded error category in a separate transaction. A status-write
failure propagates. Cancellation rolls back unpublished output and leaves an
abandoned attempt eligible for a later run; active work retains ownership.

`record_layer_success_in_transaction` is also available to future standalone full
builders. It requires an existing transaction and matching captured source token,
creates no schema, and never commits caller work. The builder remains responsible
for coherent source capture and completing its output before calling it.
`read_layer_states` reads source and checkpoint rows together. `layer_freshness`
distinguishes exact current output from append, correction and dependency pending
states. The last attempt's failure remains visible even when an older successful
generation is still current. `layer_read_snapshot` lets a read-only getter read
its output and provenance in one SQLite snapshot; it closes only its own read
transaction and leaves an existing caller transaction untouched.

The additive attempt-count migration preserves source epochs, counters and
tracking triggers. It does not invent counts for old checkpoint rows. These
contracts provide no wall-clock freshness guarantee, native memory bound or
measured throughput improvement. Routing production entry points and completing
vector, Dunbar and profile dependency contracts remain subsequent work.
Other direct output writers must participate in the same publication contract
before those checkpoints can certify production freshness; arbitrary external
SQL against derived tables is not covered by source revision tracking.

## Dunbar generated relationship publication

Dunbar now records the exact relationship IDs it creates in an additive ownership
ledger. A generation ID and fingerprint bind each ID to its persisted fields,
including SQL NULL values and types. A refresh deletes only verified generated
rows. Existing unowned contacts and every unrelated relationship producer remain
untouched; matching values do not establish ownership. Empty source, an absent
primary, or a changed primary retires the previous verified generated generation.
The returned hierarchy, SQL aggregation/tie behavior, case normalization,
frequency thresholds and three-decimal rounding remain unchanged.

Insert, changed-field update and delete triggers relinquish ownership before an
external edit or ID reuse can inherit it. The insert trigger also covers SQLite
`INSERT OR REPLACE` with recursive triggers and foreign keys disabled. Generated
inserts write their ownership record after the relationship insert. No-op updates
retain ownership. If a tracking trigger is missing or replaced, repair preserves
all relationship rows and invalidates the uncertain ledger; fingerprints cannot
prove continuity through an untracked interval. Repair is transactional and an
unchanged installation performs no DDL or ledger writes.

For an independent call, the builder captures a coherent source/ownership read
snapshot, ends that transaction, and computes before taking writer ownership.
Publication compares streaming fingerprints of the relevant message IDs,
senders, recipients and timestamps, schema, relationship rows and ownership rows.
A concurrent change rejects publication without an automatic retry. Source text
is neither copied into the ledger nor staged for this fence. These checks add
scans and do not claim constant-time work or a measured throughput improvement.

Output and its ledger publish atomically, including empty cleanup. A borrowed
transaction uses savepoints and remains rollbackable, including any ledger or
trigger migration. COMMIT/RELEASE failures roll back unpublished work; a failed
rollback never proceeds to a cleanup release. A caller that already owns a writer
retains its own locking and final commit responsibility. A later caller commit
cannot be certified against changes outside that retained snapshot.

`read_dunbar_coverage` reads ownership and relationship rows in one snapshot. It
reports managed rows, unowned contacts and invalid ownership, with `partial`,
`managed_only` or `untracked` coverage. This is an ownership description, not
source freshness. Ambiguous legacy rows are preserved and remain visible as
partial coverage. This checkpoint neither adopts those rows nor routes the
maintenance scheduler or changes profile/style/summary-sheet builders.

## Cluster ownership through the runner's outer commit

`LayerSpec` accepts an optional publication-guard factory. Existing six nonvector
specs leave it unset. The runner invokes the factory after committing its running
diagnostic and before opening the attempt read transaction. A factory captures
identity and returns an unentered context; it must not load a model or retain a
database transaction across that return.

After builder output, the success checkpoint and the final dependency resolver,
the runner enters that context. The context yields a validator, which runs only
after its release has been registered in an `ExitStack` outside the transaction.
Ownership therefore remains held through outer COMMIT or rollback, including a
failed identity validation, cancellation or failed COMMIT. Dependency resolvers
run before guard entry and cannot recursively acquire its non-reentrant lock.
Trusted guards release resources without suppressing transaction failures.

The cluster guard captures model name, embedding dimension, tier group, active
table/schema identity, the complete relevant registry row and embed/build metadata
in a short coherent read snapshot under the model lock. If present, the whole
`vec_source_v1:<table>` descriptor participates in identity comparison, so a
changed generation or committed progress within that generation is distinguishable.
The descriptor comparison is not a complete all-writer vector-freshness proof.
The existing in-progress index marker still rejects clustering.

At final publication, the guard takes SQLite writer admission first and acquires
the model lock nonblocking. A busy lock or changed identity rejects output and
its checkpoint. Successful validation retains the lock through the runner's
terminal write. Initial model-lock waiting occurs before the attempt snapshot
and outside any writer transaction. No model load or blocking model-lock wait is
introduced under SQLite writer ownership.

This guard is intended for the existing `cluster_messages` builder: its raw-vector
and source fingerprint checks already run before it publishes, and its nested
RELEASE leaves the runner's writer held. Other connections cannot change those
inputs before outer COMMIT. The runner's remaining writes affect only provenance;
trusted adapters must not modify vector/source inputs after the builder returns.
The outer guard therefore adds no third raw-vector/source scan. It does not
replace the builder's raw-input fence, certify arbitrary external vector writes,
or make a borrowed caller's later COMMIT safe. This checkpoint adds no cluster
adapter, engine routing or coverage migration.

## Eight-layer adapters and borrowed publication

`all_layer_specs(conn)` assembles clusters, monthly/entity-monthly summaries,
contradictions, structured facts, surprise, episodes, landmarks and Dunbar in
the existing consolidation order. Specs belong to that connection; using them
on another connection is rejected before any builder runs. The original
`nonvector_layer_specs()` API retains its six adapters and defaults. Preferences
remain a separate nonpersisted call. Profiles, styles, entity sheets and engine
scheduling are not enrolled by this checkpoint.

Successful output and its checkpoint now include `successful_coverage` in the
same transaction. Allowed values are `complete`, `legacy_contacts_unowned`,
`vector_generation_unverified` and `unverified`. Historical rows migrate to
`unverified` without changing source epochs, counters, tracking triggers or
successful provenance. Failed/unavailable attempts retain the previous successful
coverage, just as they retain its output. Coverage describes that publication,
not a later arbitrary external edit of a derived table. Source freshness and
execution outcome remain separate; partial coverage does not disable retrieval
or cause an automatic repeat of otherwise unchanged work.

Dunbar's primary selection uses the existing sender count/tie query inside the
runner's source snapshot. Counts describe verified generated relationships;
unowned contacts are preserved and reported as `legacy_contacts_unowned`.
`read_dunbar_coverage` remains the coherent live ownership inspection API.
Cluster counts exclude noise, so successful all-noise clustering has zero
centroids and `success_empty` while retaining its noise assignments. Missing
dependencies, missing vector tables and in-progress rebuilds are unavailable,
never successful empty output. A present rebuild descriptor must decode as an
object with format `version` exactly integer `1` and `complete` exactly boolean
`true`; malformed, unsupported or incomplete descriptors are unavailable.
Its SQLite `schema_version` is not the format version. A missing descriptor
retains compatibility with `vector_generation_unverified` coverage. This narrow
envelope check is not a full rebuild/source certificate. Runtime dependency import
failure also records an unavailable attempt against the captured source/dependency,
without a retry loop. Explicit force can retry after repair.

Cluster scheduling uses a different identity from its publication fence. Its
stable key binds runtime model/dimension/tier, active table/schema, relevant
registry mappings and model fields, embed metadata values, build state and the
whole rebuild descriptor, plus dependency versions and unchanged HDBSCAN
parameters. Ordinary metadata timestamps and registry progress/count fields are
excluded: `insert_count - attempted_insert_count >= 25` must still mean 25
committed appends, rather than one metadata rewrite. Full identity, including
those excluded values, is still checked by the publication guard. Initial empty
or historical input attempts once; corrections, dependency changes and explicit
force remain independent reasons to run. Failed initial attempts also retain
their durable 25-insert baseline.

Temporary model-lock contention is a live `deferred` result with `ModelBusy`,
not a dependency-version change or a persisted unavailable attempt. Even forced
work cannot bypass that ownership boundary. An initial busy probe performs no
diagnostic write; planning omits it until another safe boundary can observe the
runtime. `layer_freshness` reports `dependency_deferred` during that interval.
If contention occurs after computation starts, the result has `attempted=True`
to report actual work, but the durable attempt counters and previous successful
coverage remain unchanged. No timer or automatic retry is added.

An owned runner restores its entire prior checkpoint row, or its prior absence,
only after confirming output rollback. Restoration requires the same process,
ownership token, run generation and complete unchanged `running` diagnostic;
it cannot overwrite a replacement worker's state. Failure to roll back, loss of
ownership, or failed restoration raises instead of reporting clean deferral.
A cancellation racing with a busy result restores that baseline and stops the
remaining layers. Borrowed runs have no precompute diagnostic to restore and
roll back only their layer savepoint. Missing packages, invalid descriptors and
other declared dependency loss still produce durable unavailable attempts.
Only the explicit `ClusterModelBusyError` from the builder/publication boundary
is translated to deferral; it remains a `RuntimeError` subclass for existing
direct callers.

The cheap cluster probe reads package metadata, existing runtime state and small
SQLite metadata records. It does not import native packages, load models, scan
source/vector rows or wait for the model lock. Execution requires the caller's
connection to have its vector extension loaded and the existing vector runtime
initialized. Actual clustering and guard imports occur only when execution is
attempted. Initial builder acquisition is nonblocking whenever the connection
already has a transaction, including the interval after a successful probe, so
a caller retaining SQLite's writer never waits for another model owner there.
Standalone builder and owned-guard initial capture can still wait for model
ownership before opening their SQLite transaction; the nonblocking probe does
not promise that all execution is wait-free. Tuple and `sqlite3.Row` factories produce the
same scheduling identity and rebuild-state checks. Whole rebuild descriptors
still do not certify every foreground
vector writer or arbitrary vector SQL; published cluster coverage therefore
remains `vector_generation_unverified` even when source freshness is current.

`run_layers` continues to require a clean transaction by default. The explicit
`allow_caller_transaction=True` option uses protected per-layer savepoints when
the caller already has a transaction. It never commits unrelated caller writes,
including success/failure metadata. No `running` diagnostic is written before
borrowed computation. A caller that already holds a writer, or retains earlier
layer output, retains that writer until its own terminal transaction. The runner
does not promise short writer ownership for this caller-controlled mode.

Borrowed results carry `pending_caller_commit=True`; other connections cannot see
their output/checkpoint until the caller commits, and caller rollback removes
both. Failed RELEASE rolls back the layer before recording its failed attempt.
If rollback itself is rejected, execution raises immediately without further
diagnostic writes or cleanup release; the caller must roll back that transaction.
Cancellation remains cooperative and preserves caller ownership.

Cluster borrowing has an explicit separate guard: initial model acquisition is
nonblocking, full identity is rechecked under writer/model ownership, and model
ownership lasts through the runner's savepoint RELEASE or rollback. The lock is
released before returning. This does not certify the runtime model at the
caller's later outer COMMIT. Other adapters with an owned publication guard must
supply an explicit borrowed guard or report unavailable. This checkpoint does
not route engines or ingest, change algorithms or establish a measured latency,
memory or throughput improvement.

## Standalone rebuild source streaming (issue #756, first checkpoint)

`build_vectors()` and `build_separation_vectors()` read database-backed inputs
in ascending-ID pages, using the existing native batch size. They capture a
fixed upper ID, row count and A10 source revision, including its epoch. Each
page read ends before inference. Publication checks the source revision,
model generation, target manifest and database schema version under writer
ownership. Writer acquisition precedes the nonblocking model fence; that
fence remains held through the terminal COMMIT or SAVEPOINT RELEASE. A source correction, deletion, lower-ID insertion, epoch change or
schema change rejects the unfinished generation. An unrelated schema change
is conservatively invalidating. New high-ID source rows remain outside the
captured range.

A versioned metadata manifest contains model/dimension/target identity,
revision counters, the last **consumed input ID**, consumed count and inserted
vector count. It contains no source text. Skipped zero/nonfinite vectors still
advance consumed input, so restart cannot infer progress from `MAX(rowid)`.
A saved manifest without matching identity and revision is restarted once on
the next explicit call; the builder does not run an automatic restart loop.
Legacy boolean markers alone are insufficient resume evidence.

The native batch size and `txn_batch` durability cadence remain configurable.
Each group is encoded and serialized before its owned writer transaction.
Serialized payload is bounded by
`max(1, txn_batch) * batch_size * dimension * 4` bytes per resident index,
plus Python objects, one input page, the current native output and allocator
memory. Defaults at 256 dimensions give:

- CPU: `100 * 100 * 256 * 4 = 10,240,000` bytes (`9.765625 MiB`).
- MPS: `100 * 16 * 256 * 4 = 1,638,400` bytes (`1.5625 MiB`).
- Custom 384-dimensional CPU example: `100 * 100 * 384 * 4 = 15,360,000`
  bytes (`14.6484375 MiB`).

Completion and separation standalone calls stage their groups separately.
This bound is independent of corpus rows with fixed controls, but can increase
staging for small corpora compared with immediate insertion. It is not a
whole-process memory cap. A single record is not truncated and can itself be
large. Native model residency and thermal behavior are not measured by the
synthetic regression tests.

One maintenance owner covers the complete operation. Same-thread nesting in
an existing A10 maintenance worker reuses its owner. A caller's open SQLite
transaction is preserved with savepoints; the rebuild cannot release locks
that the caller already owns. Such callers also retain control of durability.
Otherwise, no rebuild-owned read or writer transaction remains open during
inference. Short page snapshots avoid pinning a job-long WAL reader; they do
not promise globally bounded WAL when other readers exist.

Explicit `messages` lists preserve caller order and text. A streamed digest
binds restart to that exact list without making another corpus-sized copy;
the consumed list position preserves out-of-order IDs. The caller's allocated
list is outside the database paging bound. Concurrent mutation of a supplied
list is not supported. Database-backed calls reuse the canonical source
tracker only with the exact built-in `messages` definition and verified tracking.
Older connections, including deprecated
`engine.open()` and the minimal five-column separation schema, instead install
a rebuild-only bridge when canonical tracking is unavailable. Completion also
supports a legacy table containing only `id INTEGER PRIMARY KEY` and `content`.

The bridge adds one namespaced state table and three source triggers. It does
not add columns to `messages`, migrate other storage features, or mark the
canonical maintenance tracker ready. Installation and repair own the writer
before model loading or clearing vectors; caller transactions use savepoints.
The token explicitly identifies its tracker, epoch, covered source fields,
table schema and index definitions. Page/publication checks verify that source
schema and the exact trigger definitions; missing or altered tracking starts
a new epoch on the next call, invalidating untracked intervals. These checks
read schema metadata and scalar counters, not corpus text. Read-only legacy
connections and unsupported source schemas fail before model allocation.
Bridge updates compare each covered value's SQLite storage type and bytes,
so custom `NOCASE`/`RTRIM` collations cannot hide changed encoder input.
Other table definitions retain the bridge even after a full storage migration
reports canonical tracking ready. This conservative definition whitelist avoids
inferring equality semantics from arbitrary SQL; compatible custom definitions
remain usable with the additional bridge counters.

Same-ID replacement invalidates a bridge generation even with recursive
triggers disabled. Custom schemas with unique indexes, including indexed primary keys that
are not rowid aliases, use a conservative bridge: every successful insert or update invalidates the
captured prefix. This includes expression and partial unique indexes, where
`OR REPLACE` can silently delete another row without a delete trigger. Such
schemas remain supported, but concurrent high-ID appends cannot preserve a
running generation. They retain this bridge even if the canonical tracker
reports ready, because its append certificate does not cover those hidden
replacement deletions. There is no repeated automatic restart loop.

When the trusted built-in definition and canonical tracking become suitable,
the next rebuild changes tracker
identity and starts a fresh generation; it atomically retires only the bridge's
owned triggers. Its inactive state table remains. The bridge never certifies
the readiness of other maintenance layers. Normal canonical schemas need no
bridge table, triggers or additional writer commit during setup.

Foreground engine add/update and `embed_single` bind inference to the active
model generation. Same-model appends above an in-progress database range are
published normally and cannot advance its consumed cursor. Initial clear and
source capture share one writer transaction, so a committed earlier append
joins the captured range. A correction within the range publishes its current
vector and invalidates the rebuild's source fence. A late model mismatch is
reported categorically: add rolls back its uncommitted source row; update's
existing storage helper has already committed its source, so its error says
to retry vector publication. Explicit-input or unattested in-progress targets
reject concurrent publication rather than silently reporting success.

Foreground publication acquires the SQLite writer before its nonblocking
model fence, then validates the target manifest and retains both owners
through the owned commit or caller savepoint release. Inference finishes
before either is acquired. Add source insertion shares that transaction;
rejection rolls back only the new add, preserving unrelated pending caller
writes. Successful add and `embed_single` calls inside an existing caller
transaction also leave durability to that caller. Update retains its existing
source-helper commit behavior; the subsequent vector publication is fenced
separately and cannot claim the source update was rolled back.

This checkpoint does **not** remove the tier-switch manager/worker's retained
message lists or solve tier configuration publication across SQLite, config
file and process state. Issue #756 remains open for that next checkpoint.
No global embedder-compatibility authority, models, precision, dimensions,
reranker selection, retrieval depth or deployment defaults change here.

### Legacy rebuild value precision

The rebuild bridge also compares values with explicit `BINARY` equality.
SQLite's conversion of a REAL value to BLOB formats it as text and can lose
precision, so the byte and storage-type checks alone cannot certify an
unchanged source. Numeric comparison supplements both existing checks.

SQL numeric equality cannot distinguish stored positive and negative REAL
zero. A source UPDATE involving a covered REAL zero therefore invalidates
the generation conservatively, including assigning the same zero again.
For ordinary source columns, the trigger listens only to covered source
fields and unshadowed `rowid` aliases. Unrelated derived-only updates do not
invalidate solely because a source field contains zero. When a covered
source field is generated, the trigger must listen to all updates because
its dependencies can be other columns. Custom UNIQUE schemas retain their
existing all-update invalidation for hidden replacement deletions.

Changed trigger definitions fail the existing identity check. The next
writer-owned repair starts a fresh epoch before model loading or clearing
the target, so manifests captured under the earlier comparison cannot
resume. Synthetic SQLite tests cover precise REAL metadata rendered into
separation text, signed zero, collations, storage types, rowid aliases,
generated fields, repair failures and caller rollback. These tests do not
measure native model behavior or add a process memory budget.

### Canonical source comparisons and FTS replacement cleanup

Canonical maintenance tracking and FTS synchronization compare storage types
and values with explicit `BINARY` collation. Declared `NOCASE` or `RTRIM`
collations cannot hide source edits, and adjacent REAL values do not depend on
SQLite's lossy REAL-to-text conversion. SQL equality cannot distinguish retained
positive and negative REAL zero, so an explicit source UPDATE involving REAL
zero invalidates conservatively, including assigning the same zero again.

For ordinary columns, comparison triggers listen to their covered fields and
unshadowed `rowid`, `_rowid_`, and `oid` aliases. Default derived-only updates
therefore remain clean. A generated source field can depend on any column, so
its comparison trigger listens to all updates; REAL zero in such a field can
conservatively invalidate an otherwise-derived update. The rebuild bridge keeps
its separate, existing schema contract and canonical-reuse restriction.

Any UNIQUE index on messages selects conservative source tracking dynamically:
every INSERT fences append-only reuse, and every UPDATE is a correction. This
includes ordinary, expression, partial and composite UNIQUE indexes. It also
includes writes that do not delete a conflicting row. Default-schema inserts
remain append-only, and their derived-only updates remain clean. Hidden deletes
from `INSERT OR REPLACE` or `UPDATE OR REPLACE` cannot escape this fence when
`recursive_triggers` is disabled.

Separate FTS cleanup triggers run only when messages has a UNIQUE index. They
remove indexed row IDs absent from messages. Their trigger-level guard avoids a
content scan on the default schema. A custom UNIQUE schema pays an O(N) orphan
scan for each inserted or updated message, where N is its FTS row count. Source
certification does not depend on FTS completeness or trigger execution order.

These canonical guarantees require `messages.id` to be an ordinary INTEGER
PRIMARY KEY rowid alias. An indexed `INTEGER PRIMARY KEY DESC` can retain NULL
and REAL IDs; a WITHOUT ROWID INTEGER primary key can retain REAL IDs. FTS may
allocate an unrelated rowid for NULL or reject a REAL ID. Unsupported IDs leave
canonical source tracking unavailable, and FTS repair is skipped with a warning;
the database remains readable. Missing source or indexed fields likewise retain
the degraded legacy path. This checkpoint does not change the rebuild bridge's
broader ID support or establish FTS synchronization for these unsupported schemas.

Changed source-trigger definitions rotate the epoch and mark maintenance layers
pending. Changed or missing owned FTS definitions are repaired and the index is
refilled from current messages in one transaction. The refill repairs stale,
missing and orphan indexed rows. It happens once per definition repair, not on
ordinary repeated opens. A caller transaction retains its writes and ownership;
rollback restores both trigger definitions and indexed contents. The runtime
must provide FTS5, `table_xinfo`, and table-valued `pragma_index_list`.

Synthetic SQLite 3.50.4 progress-handler measurements were identical at 100 and
10,000 source rows: 289 virtual-machine steps for append, 232 for a derived-only
update, and 15 for revision read. The row-count ratio is 10,000 / 100 = 100;
each measured step ratio is 1. These counts cover the tested SQL paths, not
native inference, overall database-open cost, memory use, or UNIQUE schemas.
An authorizer-based regression also rejects any source/FTS content read during
ordinary repeated tracking initialization and FTS migration.

Interval trust remains a separate unresolved contract. A writer can drop
tracking triggers, mutate messages, and recreate identical definitions without
the current definition checks exposing that interval. Arbitrary direct FTS
mutations can also defeat synchronization. This checkpoint adds neither a
schema-cookie fence nor a readiness gate to FTS search, and does not certify
either kind of external mutation.

### Foreground model admission before writer ownership

Owned foreground add and incremental vector publication now acquire the model
identity fence before opening the SQLite write transaction. Add first owns its
engine write lock, so another mutation on that same engine cannot hold the
engine lock while waiting for a model fence held by the queued add. The model
fence validates the captured generation and remains held through commit or
rollback. There is one model acquisition, with no nested acquisition of the
nonreentrant model lock.

This permits an owned foreground call to wait for a concurrent cluster source
snapshot without holding SQLite's writer or failing a valid vector publication.
Model loading and inference finish before owned writer admission. A model
generation change during loading, encoding or admission still rejects stale
publication; add and update propagate those typed errors instead of silently
falling back to a source-only write. Update retains its existing distinction:
capture failure precedes its source write, while a later vector-publication
failure can occur after the source update committed.

Caller transactions use nonblocking model capture and publication admission.
Add and update classify the connection during model capture under their engine
write lock, so another same-engine mutation's transaction cannot be mistaken
for borrowed caller work. They release that lock before encoding.
Capture accepts only a cached model observed together with its identity under
the model lock. Busy ownership or an unloaded model produces a typed retry
error before any additional source write; the caller's transaction remains
intact. Cached-model inference can still run inside a caller-held transaction.
The publication fence protects the nested savepoint release, not a later outer
commit controlled by that caller. Streamed rebuilds retain nonblocking model
fences after writer admission and use the same cached-only borrowed capture.

SQLite and event regressions cover actual clustering snapshot contention,
same-engine mutation ordering, stale generations, borrowed sentinels, atomic
add publication, and model release after admission/publication/commit failures.
This checkpoint changes admission ordering rather than bounding wait duration:
cluster snapshots may still hold the model lock while scanning the corpus.
Native integration and latency remain separate acceptance checks.

### Engine maintenance routing

Engine startup, committed mutations and later safe API boundaries now use the
same durable eight-layer planner. The default append trigger remains
`insert_count - attempted_insert_count >= 25`; 24 committed appends do not
qualify and the 25th does, including across engine instances and reopen.
`TRUEMEMORY_AUTO_CONSOLIDATE_EVERY` retains its positive-integer override.
Initial historical or empty input attempts once. Corrections, source epochs and
dependency changes follow their separate eligibility rules. A successful empty
cluster result no longer starts every layer again on each open. Source edits
committed through another handle become visible at the next safe API boundary;
there is no idle polling timer. Uncommitted caller changes do not schedule work.

Foreground scheduling reads eight checkpoint records, source counters and small
dependency metadata. It does not scan source/vector rows, import native
frameworks, wait for the model lock or perform consolidation in `add()`. The
foreground write lock is acquired nonblocking for this probe. Package versions
are cached in three records and refreshed on connection setup and worker/manual
setup. One extension-failure record is bound to the probed model/table identity
and version snapshot. Actual worker extension loading supplies that evidence;
metadata alone is not proof of availability. Epoch checks prevent a late worker
from overwriting newer shared capability evidence, while that worker still uses
its own actual failure for its current attempt. A transient model-busy probe
cannot erase an existing extension-failure baseline.

Each canonical file has at most one active owner and one pending notification
generation. A notification stores only a generation and threshold, not an
engine, caller connection or source content. Multiple foreground notifications
replace that single pending slot. A same-process owner's release services an
existing pending notification; it creates no notification of its own. A pending
coordinator behind a local owner is retained only until that exact owner's
release or cancellation, so closing the last engine does not lose the wake.
Cross-process contention reports busy/pending and needs a later API boundary;
this checkpoint has no automatic cross-process release notification. Deferred
model ownership and failed attempts do not schedule their own retry loops.

Workers create and close their own file connection. `engine.close()` detaches
and closes only the foreground connection; it does not cancel or join a shared
worker. Cancellation clears queued notifications but retains ownership until
active computation reaches a cooperative boundary and its connection closes.
Thread-start failure closes an unclaimed owner descriptor; interruption after
the target started leaves teardown with that exact worker and cannot cancel a
later successor. Child processes discard inherited owner descriptors, pending
references and registry state. Daemon threads do not guarantee completion when
the process exits; durable abandoned-attempt recovery remains necessary.

Explicit `consolidate()` forces one pass with the same eight result keys plus
the existing nonpersisted preference extraction result. A busy owner produces
nine explicit `BUSY` entries and does not queue a hidden forced pass. A file
call without an existing transaction uses a dedicated connection. An existing
caller transaction uses the original connection's protected savepoints and
reports `pending caller commit`; its writes and maintenance output remain
rollbackable together. `:memory:` automatic work reports pending without opening
an empty replacement database or creating a worker. Explicit manual maintenance
uses the original in-memory handle. Caller-held writer and later outer-commit
limitations from the borrowed-runner contract still apply.

Successful manual summary/cluster publication restores the calling engine's
corresponding capability flags, including successful empty output. Failed or
unavailable layers never enable those flags or clear a previously usable
capability. The existing automatic-consolidation opt-out remains until explicit
manual repair. Cluster retrieval still requires live vector availability.
`get_stats()` exposes bounded coordinator and per-layer outcome, freshness,
coverage and categorical error state; it does not expose source text, paths or
exception messages. Unverified vector-generation and legacy-contact coverage
remain explicit limitations, rather than complete success claims.

This routing checkpoint preserves builder algorithms and the A17 foreground
vector-publication guards. Ingest database replacement, profile/style getter
freshness and custom-schema source tracking remain separate work. Synthetic
SQLite and event tests establish scheduling and ownership behavior; native
latency, throughput and memory acceptance still require measurement.
