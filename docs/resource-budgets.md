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
## Atomic L0 full generations

Full personality-profile and character-style builders now read a coherent source
snapshot, compute and serialize their complete output, then validate the exact
relevant source values and source schema under SQLite writer ownership. A changed
source raises a categorical `sqlite3.OperationalError`; neither builder retries
internally. Publication replaces only its fully owned table. Vanished senders are
removed, and successful empty input clears the previous output.

Ordinary messages contribute to both full builders, matching the incremental
add path's directive exclusion. Existing schemas without a `directive` column
continue to treat all messages as ordinary. No maintenance tables are required
for these public builders. Profile source order remains timestamp order; style
source order remains sender and timestamp order. The existing formulas,
thresholds, Python lowercase identity keys and 256-dimensional style algorithm
remain unchanged. Generated update timestamps are incidental to the generation.

When the builder owns its transaction, its read transaction ends before CPU work
and its write transaction begins only for validation and replacement. A caller's
existing transaction is retained instead, with savepoints around each phase. The
builder never commits or restarts unrelated caller work. Any writer lock already
held by the caller remains held during computation; the builder cannot release
that lock on the caller's behalf. A stale WAL reader fails at write upgrade.

Write, cancellation and terminal COMMIT/RELEASE failures roll back the attempted
replacement. The previous complete generation survives a later caller commit
after successful rollback. If SQLite also rejects rollback, cleanup propagates
that failure without releasing partial output; the connection owner must recover
or close the transaction. This is not a guarantee that a caller can ignore a
failed rollback and safely commit.

These are full recomputations. They retain source rows and computed output, and
style computation still retains one entity's per-message 256-element vectors.
Validation streams the source again while holding the writer; it is linear in
the read source size, not constant-time. No memory or latency improvement is
claimed without measurement. They perform no embedding-model work.

This is the first #748 foundation stage. It does not register L0 maintenance
checkpoints, suppress stale cache reads, change incremental updaters, schedule
repair after source mutations or alter opt-in entity-profile summary sheets.
Those integration steps remain necessary to resolve the complete invalidation
issue. An A10 runner can wrap a builder in its own source transaction and commit
the successful output/checkpoint together once that shared contract is wired.

### L0 style accumulator format, checkpoint 1

Style profiles retain the unnormalized sum of their per-message, unit-length
character n-gram vectors. The `vector_sum` column stores 256 little-endian
float64 components: `256 * 8 = 2048` bytes per entity. The existing JSON `vector`
column remains the normalized mean exposed to readers. `message_count` counts
contributing messages, including messages whose style vector is zero. Each
append reads and writes one entity and performs O(256) arithmetic. Batch
computation retains the existing source snapshot and arithmetic order and uses
one 256-component sum while computing each entity's messages.

`accumulator_version = 1` identifies this format. Additive schema migration
leaves existing rows at version 0 with a NULL sum; it neither reconstructs lost
magnitude from the old normalized profile nor scans source messages. Repeated
schema initialization performs only the fixed table-column check. These
derived-table schema changes do not advance message source revision counters.
Batch publication creates or upgrades the format only after source validation,
then atomically replaces profiles, sums, counts and versions in the same
transaction. Failed publication preserves the previous generation and any
caller-owned writes.

Appending to a legacy, unsupported-version or invalid accumulator raises
`StyleVectorRebuildRequired` with a bounded reason (`legacy`,
`unsupported_version` or `invalid`). The previous profile remains readable.
An append uses writer ownership before reading its sum and a savepoint around
schema and row changes. It never commits: a successful standalone call leaves
its transaction pending; an existing caller keeps final commit and rollback
ownership. The engine's existing best-effort style catch can therefore retain
its source/vector add without partially changing the style profile.

Empty entity/message incremental calls remain no-ops. Batch builds retain
their ordinary-message filter, case folding and existing empty-source behavior;
directives remain excluded by the engine's add caller. Getters, style dimension,
embedding and reranking models, native dtypes, and retrieval result caps are
unchanged. The format does not certify freshness after unrelated source edits.

This is a preparatory checkpoint, not production activation. Existing legacy
profiles defer style appends until an explicit successful style rebuild. The
production migration bridge and its pending/failed status must be integrated
before release; otherwise legacy profiles could remain stale while source adds
continue. This checkpoint adds no engine routing, automatic rebuild, model
load, corpus scan on open, or user-facing maintenance status. Synthetic
arithmetic and transaction checks do not establish retrieval quality or native
performance acceptance.


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

### Clustering health visibility (#720)

Clustering health reuses the scheduler's dependency identity and persisted
attempt/provenance row. It reports availability, outcome, freshness, coverage,
output count and categorical errors separately. A successful empty or all-noise
result remains `success_empty` with zero output; a missing dependency cannot
become an empty success. Existing unverified vector-generation coverage remains
visible and does not certify complete health.

The engine accessor does not connect or initialize a coordinator. It attempts
the foreground lock without waiting, then reads the existing connection under
one read snapshot. Model ownership is also tried without waiting. Foreground
and model contention report deferred observation; a disconnected engine reports
unknown readiness. An existing caller transaction is preserved and reported as
pending caller commit. Source mutations, pending work and failed attempts retain
their distinct outcomes and prior published counts.

Package inspection uses at most six top-level spec lookups and three distribution
records per observation. It imports no native module and loads no model.
Distribution metadata
failure can report dependency unavailability, but installation guidance requires
a confirmed absent HDBSCAN module. A cached worker extension failure is reused
only while its dependency identity matches. Health reads do not change durable
attempts, capability epochs or scheduling, and do not create a worker or database.

Actual maintenance and bulk-ingest failures emit a categorical warning once per
failure transition in the shared process coordinator. Repeated status reads and
foreground notifications do not log warnings. Committed successful clustering,
including empty output, clears that process failure. A borrowed savepoint release
does not prove the caller's outer commit and preserves prior process failures
through rollback. There is no outer-commit callback; a subsequent committed
successful maintenance pass clears the retained evidence. A process failure is not a durable
cross-process diagnostic; persisted scheduler outcomes remain independently
visible. The optional clustering extra, eight persisted layers, nine manual
result keys, builder algorithms and preference stage are unchanged.

### Private style maintenance preparation (#747)

Explicit style maintenance can enroll a private `style_vectors` checkpoint.
Ordinary database initialization and spec construction do not enroll it or
scan profiles. The eight public adapters and nine consolidation result keys
remain unchanged; explicit enrollment adds a ninth physical checkpoint row.
Fixed output INSERT/UPDATE/DELETE triggers invalidate only that private row
and clear its attempt baseline. Trigger repair invalidates provenance using
metadata without scanning messages or profiles. Caller rollback also rolls
back enrollment, trigger installation and invalidation.

Tracked rebuilds retain the existing 256-dimensional character n-gram
arithmetic, sender ordering, case merging and ordinary-message filter. They
capture source before computation, acquire writer ownership before checking
source and future accumulator versions, then publish profiles, raw sums,
hash version 2 and complete provenance in the runner's final transaction.
Unsupported future formats are retained. Valid legacy getter output remains
readable; getters never rebuild. Low-level standalone builders retain their
existing SQLite-fenced contract and do not gain global coordination or the
tracked future-format guard.

Style work checks cooperative cancellation during capture, computation and
publication. SQLite progress-handler installation additionally requires the
explicit `connection_owned=True` opt-in and an owned runner transaction; a
clean transaction alone does not establish ownership of the connection. That
handler is disabled before rollback or commit. Other connections keep their
caller's unknown handler, and borrowed transactions retain final commit
ownership. Verified stale source, cancellation and SQLite writer
contention defer without establishing a failure baseline after confirmed
rollback; unexpected SQL failures remain failures. Failed publication retains
the previous output and successful provenance.

Initial enrollment and trigger repair can also defer on lock contention:
the preparation boundary must establish that no writes started or confirm
its owned rollback or caller savepoint rollback. Caller work stays pending;
unknown errors and an unconfirmed transaction end do not become deferral.
Read-only baseline probes before a running diagnostic, including the repeated
layer probe and cancelled-result probe, can defer only for a known lock error
with unchanged transaction ownership. Unavailable observations report an
unknown count and unverified coverage; no attempt baseline is advanced.
Lock contention before the running diagnostic leaves its prior checkpoint
unchanged. If a writer instead blocks restoration after a deferred build, the
running diagnostic remains persisted; a later owner can recover it after the
writer releases. The deferred result does not claim that restoration succeeded.
An owned SQLite interruption can attest automatic rollback only when this
operation's installed callback returned cancellation and SQLite reported the
interrupt code, or the exact legacy interrupt message on Python 3.10. The
callback is cleared before cleanup. Merely observing an ended transaction
does not establish rollback. Error categories also support Python 3.10's
missing SQLite code attributes without hiding unexpected SQL errors.

This checkpoint does not activate engine routing, automatic migration,
incremental coverage, style status or coordinator style-only jobs. Those
remaining migration steps are required before release. Synthetic transaction
proofs do not establish native performance or retrieval quality.

### Incremental style coverage checkpoint

Single-memory adds can now advance an explicitly enrolled, current style
checkpoint. They capture complete successful provenance and the attempt
baseline before insertion, inside the existing source writer transaction.
The source epoch, correction count and nonappend revision must stay unchanged;
revision and insert count must each increase by exactly 1. The inserted ID
must exceed the preceding high watermark and equal the new high watermark.
The inserted source row and unchanged style checkpoint are verified before
the existing raw-sum updater runs. Profile reads use entity-key lookups;
source verification uses the inserted primary key. No full source or profile
scan, schema enrollment or fallback rebuild runs on add.

Hash version 2, accumulator schema, exact owned trigger definitions and the
dependency identity are checked again after source insertion, after the raw
append and after successful provenance publication. An ordinary profile write
must produce the exact invalidated projection of the captured checkpoint;
excluded inputs must leave that checkpoint unchanged. The final successful
row must match the intended source, output count and both baselines exactly.
Changed metadata or provenance rolls back style publication instead of
certifying it. Pending category writes compare every checkpoint field, so
they cannot overwrite a newer failure, owner or attempt count.

Incremental and tracked style work conservatively defer schemas with any
extra MAIN trigger on `entity_style_vectors`, any MAIN trigger on
`maintenance_layers` or `metadata`, or any TEMP trigger on these tables. Only the three
fixed owned output trigger names with their exact definitions permit append
coverage; a similar name or a TEMP shadow is insufficient. These checks read
schema metadata and never scan source rows or profiles. Unsupported schemas
also skip pending checkpoint writes, since those writes could themselves
invoke an unsupported trigger. Preparation, running diagnostics, tracked
output admission and deferred restoration recheck under writer ownership
before mutating affected rows. The private dependency remains deferred with
`StyleTriggersUnsupported`; future coordinator/status routing must preserve
that status rather than certify or retry unsupported work as successful.
Metadata triggers are included because publishing the hash marker can invoke
profile mutations after a rebuild has written its output.

The raw sum and both provenance baselines publish in one nested savepoint.
An existing entity keeps the output count; a new entity adds 1. Directive or
empty-sender rows advance source coverage without touching profiles. Short
texts with zero vectors still increment message counts. Identical content in
two different source rows counts twice; replaying an already consumed append
proof cannot count the same row again. The public updater, getter and batch
arithmetic are unchanged. Incremental and batch accumulation orders may have
different floating-point rounding; this checkpoint introduces no tolerance.

Missing, dirty, legacy, future-format or running coverage skips unsafe profile
updates while retaining the source add. A newly missed append invalidates only
its captured successful/running checkpoint and clears its attempt baseline;
it cannot overwrite a newer invalidation. An existing failed build retains its
retry baseline. Failed precomputation or publication leaves bounded categorical
evidence, and unexpected SQL errors are failures rather than writer deferrals.
If writing that pending evidence also fails, the result reports failure and
does not claim durable invalidation; source freshness still shows the uncovered
append. Normal source commit owns the final publication. Caller transactions
retain their final commit or rollback, and savepoint rollback/release failure
escapes rather than reporting a successful add.

The synthetic gate starts with one tracked empty bootstrap, performs 75 actual
engine adds and probes tracked eligibility after every add. The required number
of additional full builds is 0; successful and attempted insert baselines both
advance by 75. This checkpoint does not activate startup migration, style-only
coordinator jobs, consolidation/bulk style routing or bounded public style
health. Those routing steps remain required before release.

### Style routing checkpoint

Startup, cached-connection and committed-add notifications now plan private
style work independently of optional clustering. Style readiness reads only
fixed source/checkpoint metadata, schema declarations and trigger definitions.
It never reads source contents or profile values, loads a model, enrolls schema
or waits for the foreground engine lock. A compatible complete/current hash2
checkpoint needs no work; an uncovered append uses the existing threshold of
25 committed inserts. Thus 24 uncovered inserts do not schedule a rebuild and
25 do. Corrections, missing provenance, wrong hash markers and supported owned
trigger repair are immediately eligible. A failed attempt at the same source
and dependency retains its retry baseline. The synthetic 75-add gate starts
from a tracked empty bootstrap and requires 0 additional full builds and
0 style worker requests.

Failed and unavailable attempts retain that same 25-insert retry budget even
when hash2 or complete coverage is missing. At 0, 1 and 24 subsequent appends,
repeated notifications produce 0 worker opens and 0 integrity scans; at 25,
one retry advances the attempt baseline. Source corrections and changed
dependencies remain immediately eligible. This also bounds retries for an
unsupported future accumulator format.

The coordinator retains one active worker and one immutable pending request.
Coalescing unions requested work kinds and retains the latest threshold; it
does not turn a busy or failed attempt into an automatic retry loop. Style-only
jobs open an existing canonical database using an escaped `mode=rw` URI. They
retain tuple rows, foreign keys, the 10000 ms busy timeout and existing local
cache/synchronization settings. They do not create a database, set its journal
mode, initialize general schema, load extensions or repair canonical source
tracking. Unsupported MAIN/TEMP triggers on style profiles, checkpoints or
metadata return `deferred` with `StyleTriggersUnsupported`, without enrollment,
diagnostic writes or repeated scheduling. Missing or incompatible canonical
source readiness is `unavailable`; normal schema initialization remains the
repair route. The existing public-layer and combined opener still uses its
general initialization contract.

After safe metadata preflight, the style-only worker runs `quick_check(1)`
before style writes. This is a background database scan, not a bounded
foreground probe. Corrupt, failed or cancelled checks do not claim a successful
open. Only an exclusively owned connection receives its cancellation handler;
the handler is cleared before cleanup or style execution. Existing SQLite busy
waits can delay cancellation. The worker closes its handle before releasing
canonical ownership, including on setup and execution failure.

The deprecated `open()` route now schedules guarded work instead of rebuilding
style synchronously or independently claiming hash2. Explicit consolidation
and bulk ingestion use the same private tracked route; bulk publishes style
once and preserves its existing result key. Successful manual and worker
observations are ordered within the process and published after dedicated
connection cleanup. Borrowed work remains pending caller commit, leaves its
original handle open and never clears earlier completed failure evidence.
Rollback restores prior output and provenance. Conditional hash/format
invalidation compares the entire observed checkpoint under writer ownership;
a newer row survives, and lock deferral requires confirmed rollback.

Owned manual execution carries an unready foreground style observation into
its private report instead of bypassing TEMP-trigger restrictions on a new
connection. Public-layer work retains its existing initialization behavior.
Canonical style preparation validates source readiness under `BEGIN IMMEDIATE`
or an upgraded caller savepoint before enrollment. Diagnostic, restoration
and output paths revalidate at their explicit writer admission points, so a
currently missing or changed source trigger cannot publish certified output.
These checks preserve protected rollback and the standalone low-level contract.
An unstarted worker records its failure before releasing ownership; later
lifecycle cleanup cannot overwrite a newer manual observation.

Public contracts remain eight maintenance layers and nine consolidation
result keys. Private style results affect aggregate status and appear separately
in `maintenance.style` health. This bounded observation does not connect or
schedule work. It distinguishes missing/unverified provenance, current complete
coverage, successful empty output, failures, unsupported readiness and pending
caller commits. Current database invalidation overrides historical success;
a genuinely committed current checkpoint can supersede historical failure.
Future declared accumulator formats remain unverified until guarded inspection;
health does not scan rows to infer per-row format compatibility. Mixed-case
message deletion normalizes only the style aggregate key to lowercase.

Exact current source-trigger definitions establish structural readiness only.
They cannot detect a prior drop/change/recreate interval in which changes went
untracked. Matching definitions cannot promote missing or unverified provenance
or clear a known invalidation. These routes retain the existing uninterrupted
canonical-tracking assumption; they add no interval-attestation mechanism or
native performance/retrieval-quality claim.

Canonical style no-op decisions validate source readiness in the same read
snapshot as their checkpoint and revision. Owned probes roll back only their
own read transaction; borrowed probes retain the caller's transaction. They
perform no source/profile scan, writer admission, rebuild or enrollment. A
trigger lost after the initial scheduling observation returns unverified
`StyleSourceChanged`, including at the per-layer and cancelled-result reads.
Read rollback failure remains an error. Worker observation ordering begins
before ownership binding, so a binding failure reports its category while
preserving a newer manual observation.

Fixtures that exclusively test public consolidation or explicit vector rebuilds
disable their unrelated style capability before opening or adding. Production
style startup remains independent of the public consolidation flag. The public
threshold still requires an established durable attempt: 24 later inserts
produce no request, and the 25th makes the layer eligible.

Canonical complete-coverage reads also validate hash2, the declared accumulator
format, exact owned output triggers and enrollment, and unsupported MAIN/TEMP
triggers in the checkpoint's read snapshot. TEMP tables or views that shadow
the five canonical source, checkpoint, metadata and output table names are
unsupported. These checks use schema and keyed metadata reads; they scan no
source or profile rows and acquire no writer. A readiness change after
preparation returns unverified coverage without a durable attempted baseline,
so the next wake can repair supported enrollment or perform hash migration.
Cancelled result reads use the same proof. Initial unverified migrations still
run normally. A real failed builder keeps its error and insert-count baseline;
if old complete coverage loses its metadata proof, only the returned coverage
is downgraded. Its 25-insert retry and explicit force behavior are preserved.
That downgrade survives the output-state reload if a retry fails. Successful
rebuilds publish their newly verified coverage.

### Bounded tier rebuild OOM recovery checkpoint

The rebuild worker leaves the failed batch's exception scope before backoff and
cache cleanup, retaining only an OOM classification flag. Failed forward and
write tracebacks no longer remain actively referenced by that handler during
cleanup. Existing OOM classification is unchanged, including the terminal
handling of a bare `MemoryError()` without a recognized message.

Recovery tracks the actual source-batch length at the unchanged offset, rather
than the throttler's configured size. A local effective ceiling never increases
until a completion/separation pair and its checkpoint commit. Each halving phase
allows at most two failed attempts, including one cleanup retry at singleton
size. Two failures then halve the phase ceiling, or terminate at one. A naturally
smaller batch can skip a phase only when its size is at most half that phase's
ceiling; smaller sensor decrements do not renew retry credits. For initial failed
length N, at most `2 * (floor(log2(N)) + 1)` attempts can fail at that offset:
N=1 permits 2, N=8 permits 8, and N=16 permits 10. These count paired batch
attempts; an attempt can run both completion and separation inference.

A configured size of 32 with only 8 remaining rows therefore retries at most
8 rows before forcing 4, even if later configured sizes are 16 or 8. Singleton
exhaustion reports failure with an instruction to free memory before retrying.
Previously committed pairs and progress survive. Recovery state resets after
committed progress; successful normal batching and ordering are unchanged.
Cancellation and the existing 9000-second timeout are rechecked after recovery
hooks and recovery admission before another native batch begins.

If a failed batch or full-table clear leaves the worker's owned connection in
a transaction, the failure propagates before OOM classification, cleanup, retry
or status writes. An owned status-write rollback that leaves a transaction open
also propagates. The manager must abort and close that connection; failure status
cannot safely be published through it and may retain the previous running state.
Ordinary clean status rollback remains best-effort, and borrowed status calls
retain their existing savepoint and caller-transaction semantics.

This bounds repeated nonprogress, not one native call's duration, allocator bytes,
the complete source list retained by the worker, or whole-process memory. Native
Mac/MPS calibration and the original memory incident remain unresolved.

### Tier source adapter: bounded checkpoints for an inactive pair

`tier_switch/source.py` supplies model-free plan, initialize, read, publish-prefix,
and finish operations. It is not yet called by the tier manager or worker, so it
does not remove their retained source list or change serving activation. Callers
must own an inactive completion/separation pair, validate its exact model and
schema beforehand, install source tracking and metadata before planning, and
hold the canonical maintenance lease plus exclusive cooperating pair-writer
ownership. Database identity on background reopen and snapshot consistency are
caller obligations. The adapter does not establish these preconditions.

Full initialization accepts only an already empty pair and never clears tables.
A registry row, matching row count, generic manifest, or schema alone cannot
certify existing vector contents. Resume and delta require both a strict source
manifest and this adapter's receipt. The receipt binds the model, dimension,
target names, schema generation, consumed source range, and an ordered SHA256
chain over signed 64-bit row IDs and both serialized vector blobs. Restart
recomputes that chain once in pages of 64 rows. Dimensions are limited to 1..4096;
each stored blob must contain exactly `4 * dimension` bytes. Two vector pages
therefore contain at most `2 * 64 * 4096 * 4 = 2,097,152` bytes of serialized
payload, excluding Python objects, SQLite buffers and transient hash input.
Receipts detect changed bytes under cooperating ownership; they do not authenticate
data against a writer that can forge both metadata and vectors.

Restart verification takes O(N) time in one read snapshot and can retain WAL
pages until that snapshot ends. Source pages default to 64 rows and honor the
explicit caller row limit. A row limit is not a text-byte bound: one message can
still be large. Reads finish before encoding. Every encoded prefix publishes both
target rows, the consumed cursor and the receipt in one owned transaction. The
returned immutable pending suffix supports shrinking retries without skipping
inputs. Failed database writes require replanning because SQLite total_changes
also counts rolled-back writes. Borrowed transactions are refused untouched.

The plan pins its connection, data_version and total_changes. Any intervening
write, including a legitimate concurrent tail append or unrelated status write,
requires explicit replanning and bounded restart verification. TEMP shadows and
side-effect triggers on pair/metadata tables are refused. Source corrections,
deletions and low-ID insertions invalidate captured provenance; a separately
prepared empty pair is then required. Negative and zero IDs remain valid cursors.
Finish distinguishes complete captured-range coverage from whole-source coverage
at its own transaction snapshot. Neither result activates a target or promises
future freshness. Final concurrent-tail caller adoption, native memory budgets,
and the original Mac memory incident remain unresolved.

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
publish one immutable cache snapshot. A failed distinct load preserves the
previous snapshot.

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
