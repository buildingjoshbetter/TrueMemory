# Resource budgets

## Scope and model ownership

The shared model server normally owns embedding and reranking models for all
MCP sessions. A contended single-query request can also load its existing CPU
fast encoder. If the server is unavailable or disabled, a client can load
models locally. Each process that actually constructs an MPS model configures
its own MPS allocator budget; a proxy does not initialize a local allocator.
Cache cleanup skips an unconfigured MPS allocator so CPU-only or proxy work
cannot consume allocator startup settings before a later MPS model load.

The budget below limits allocations managed by the PyTorch MPS allocator.
It does not cap CPU tensors, Python objects, resident memory, physical footprint,
swap, or the sum of multiple processes. CPU fallback has no MPS protection.
Whole-process admission, cancellation, and recovery limits remain separate
work tracked in #297. No fixed per-tier or per-session footprint is guaranteed.

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
model, including standalone embedding and reranking fallbacks. CPU, CUDA,
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
