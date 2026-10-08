# TrueMemory audit evidence, 2026-10-08

Read [the dossier](audit-dossier-v3.md) for the plan, evidence limits, claims matrix, granular issue specifications, and release gates. This branch contains audit documentation and synthetic probes only. No production fix or release is claimed.

## Reproduce the narrow findings

The source under test is commit `063e5b8844af735a52fde886217a5d26a0f13064`. The audit branch adds documentation to that source. From its repository root:

```sh
python3 -B docs/audits/2026-10-08-v1/evidence/architecture-repro-v2.py .
python3 -B docs/audits/2026-10-08-v1/evidence/contracts-repro-v2.py .
python3 -B docs/audits/2026-10-08-v1/evidence/parent-repro-v2.py .
python3 -B docs/audits/2026-10-08-v1/evidence/runtime-repro-v2.py .
```

The first three programs use Python's standard library. The runtime program also needs NumPy and an authenticated `gh` executable; it reads PR729 source pinned at `693ac64f4bb9d806c555e2030d8444e133d1cff7` for two explicitly unmerged findings. Use an existing environment with these requirements. The probes never import PyTorch or load model weights and do not read user conversations or the live database.

The selected production functions run with synthetic collaborators. Architecture probes use temporary SQLite databases; the vector retry probe substitutes ordinary unique-rowid tables for sqlite-vec and does not prove native extension recovery. Host schema fixtures are not live host dispatch tests. Deadline/concurrency probes bound their waits and use fake inference. Failed assertions after a fix are expected; these programs capture the audited broken behavior and must be converted into proper regression tests during implementation.

Version 2 scripts change only source-root portability and temporary-directory placement from their local v1 counterparts. Original v1 receipts are preserved. Architecture receipt v1 contains repeated intermediate runs; count unique probe names, not output lines. Public-copy verification receipts v2, when present, record reruns of the portable scripts.

## Evidence boundaries

Do not run a hardware stress test on an already pressured process. Full model, platform, quality, and hardware validation are future release gates, not results of this documentation branch. No promise of zero heat or attribution of the entire live memory footprint is made.

## Dossier revision 3

Revision 3 corrects reproduction command working directories and requires numeric soak duration, drift, idle recovery, and comparison criteria to be frozen before testing fixes. Prior dossier versions remain available for audit history.
