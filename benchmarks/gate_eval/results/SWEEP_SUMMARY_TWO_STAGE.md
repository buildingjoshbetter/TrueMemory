# Phase 14 — Two-Stage Gate Sweep Summary

Auto-generated from `two_stage_sweep/*.json`. Last update: 2026-04-23 06:10:24.

Storage Δ % is computed against the verbatim baseline (v05_paper_verbatim).

| NLI model | τ | regex | dataset | drop% | kept/total | p@1 | p@3 | p@10 | chitchat-drop% | p95 ms | db bytes |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| roberta-base-zeroshot-v2.0-c | 0.4 | aggressive | long | 36.67 | 475/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.018 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.5 | aggressive | long | 47.07 | 397/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.016 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.55 | aggressive | long | 53.2 | 351/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.014 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.6 | aggressive | long | 59.47 | 304/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.015 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.7 | aggressive | long | 71.47 | 214/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.013 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.4 | off | long | 36.13 | 479/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.014 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.5 | off | long | 46.93 | 398/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.014 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.55 | off | long | 53.07 | 352/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.019 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.6 | off | long | 59.33 | 305/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.028 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.7 | off | long | 71.47 | 214/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.039 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.4 | standard | long | 36.13 | 479/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.023 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.5 | standard | long | 46.93 | 398/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.017 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.55 | standard | long | 53.07 | 352/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.013 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.6 | standard | long | 59.33 | 305/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.015 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.7 | standard | long | 71.47 | 214/750 | 6.59 | 6.59 | 6.59 | 100.0 | 0.015 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.4 | aggressive | short | 81.25 | 1103/5882 | 2.0 | 3.0 | 4.0 | 100.0 | 0.005 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.5 | aggressive | short | 87.28 | 748/5882 | 2.0 | 2.5 | 3.5 | 100.0 | 0.005 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.55 | aggressive | short | 89.71 | 605/5882 | 1.0 | 1.0 | 1.5 | 100.0 | 0.005 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.6 | aggressive | short | 91.86 | 479/5882 | 1.0 | 1.0 | 1.5 | 100.0 | 0.004 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.7 | aggressive | short | 95.12 | 287/5882 | 0.5 | 1.0 | 1.5 | 100.0 | 0.005 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.4 | off | short | 81.04 | 1115/5882 | 2.0 | 3.0 | 4.0 | 100.0 | 347.361 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.5 | off | short | 87.16 | 755/5882 | 2.0 | 2.5 | 4.0 | 100.0 | 0.004 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.55 | off | short | 89.63 | 610/5882 | 1.0 | 1.0 | 1.5 | 100.0 | 0.004 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.6 | off | short | 91.86 | 479/5882 | 1.0 | 1.0 | 1.5 | 100.0 | 0.005 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.7 | off | short | 95.12 | 287/5882 | 0.5 | 1.0 | 1.5 | 100.0 | 0.006 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.4 | standard | short | 81.04 | 1115/5882 | 2.0 | 3.0 | 4.0 | 100.0 | 0.011 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.5 | standard | short | 87.16 | 755/5882 | 2.0 | 2.5 | 4.0 | 100.0 | 0.005 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.55 | standard | short | 89.63 | 610/5882 | 1.0 | 1.0 | 1.5 | 100.0 | 0.004 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.6 | standard | short | 91.86 | 479/5882 | 1.0 | 1.0 | 1.5 | 100.0 | 0.004 | 4096 |
| roberta-base-zeroshot-v2.0-c | 0.7 | standard | short | 95.12 | 287/5882 | 0.5 | 1.0 | 1.5 | 100.0 | 0.004 | 4096 |
