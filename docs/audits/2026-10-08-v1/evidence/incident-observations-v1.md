# Live incident observations

Recorded during the 2026-10-08 audit. Local operational evidence, not a controlled benchmark. No personal memory contents are included. Earlier sampling values below were observed in tool output during this session; full original vmmap/sample dumps were not saved in this directory.

## User evidence

- Activity Monitor screenshot at approximately 04:01 local: TrueMemory process 72887 displayed 26.24 GB.
- Physical memory: 32.00 GB. System compressed memory: 11.09 GB. System swap used: 12.52 GB.
- These are distinct accounting categories. Process footprint, RSS, compressed memory, and swap must not be added together as independent physical allocations.

## Read-only process inspection

- 04:03:54 local, vmmap process 72887: physical footprint 40.2 G; peak 78.4 G; MALLOC_LARGE 36.2 G, of which 28.8 G swapped; IOAccelerator 1.3 G.
- Approximately 04:08, process sample: footprint 41.1 G; peak 78.4 G; active CPU scaled-dot-product/flash-attention float kernels and threads waiting on locks. The sample does not identify which model or request created the retained allocation.
- Approximately 04:20, process CPU 64.5%, RSS 12,293,936 KiB; system swap 14,180.56 MiB.
- Approximately 04:29, process elapsed time 14:39:59, CPU 95.4%, RSS 12,794,768 KiB. RSS conversion: 12,794,768 / 1,048,576 = approximately 12.202 GiB. Physical RAM from sysctl: 34,359,738,368 bytes / 1,073,741,824 = 32 GiB. System swap: 15,035.75 MiB / 1024 = approximately 14.683 GiB.
- pmset reported no recorded thermal or performance warning. That is not a temperature measurement and does not establish that the machine was cool.

## Runtime and logs

- The installed runtime is an editable checkout of the original repository, whose source HEAD was e7f1fd7. Clean audit checkout is 063e5b8844af735a52fde886217a5d26a0f13064. The intervening main commits concern dependencies/workflows; source defects must still be tied to exact paths and commits.
- Package metadata: truememory 0.7.6.1, source version 0.7.6.2; torch 2.12.0, transformers 5.11.0, sentence-transformers 5.5.1, numpy 2.4.6, hdbscan 0.8.44.
- Model-server status: embedder and reranker in sticky CPU fallback after earlier MPS failures.
- Previous-day logs: server started 13:49:21; Qwen model loaded 13:49:33; MPS OOM at 13:49:53; CPU fallback. Reranker loaded on MPS 14:08:19; OOM 14:08:20; CPU load 14:08:22.
- Read-only corpus aggregate: 9,942 memory rows; mean content length approximately 233.666 characters; maximum 4,140 characters; zero rows longer than 8,192 characters. This does not measure query lengths, padded batches, token lengths, or intermediate tensors.

## Interpretation boundaries

- Confirmed: excessive process footprint, swap pressure, substantial CPU activity, prior MPS OOM, and both models reporting CPU fallback.
- Not established: which exact request or tensor retained the tens of gigabytes, a CPU allocator leak versus retained live tensors, the relative contribution of reranking versus embedding, or an independently measured temperature/power delta.
- MPS allocator watermarks are not a whole-process memory ceiling and do not cap CPU-side allocations.
- Do not run load tests on this already pressured process. Capture a bounded baseline in an isolated, recovered environment before real inference comparisons.
- This audit performed read-only inspection and synthetic control-flow probes. It did not restart the user's service, modify the live database, change global settings, or deploy production fixes.
