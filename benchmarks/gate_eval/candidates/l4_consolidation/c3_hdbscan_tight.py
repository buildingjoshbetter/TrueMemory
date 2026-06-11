"""C3_tight — HDBSCAN extractive with min_cluster_size=3, min_samples=2.

Phase 10 ablation per Phase9_Sweep.md Finding 3: default HDBSCAN
collapsed 750 msgs into 2 huge clusters. Tighter params force finer
granularity.
"""

from __future__ import annotations

from benchmarks.gate_eval.candidates.l4_consolidation.c3_hdbscan_extractive import (
    C3HDBSCANExtractive,
)


class C3HDBSCANTight(C3HDBSCANExtractive):
    name = "c3_hdbscan_tight"
    tier = "base"

    def __init__(self, **kwargs):
        kwargs.setdefault("min_cluster_size", 3)
        kwargs.setdefault("min_samples", 2)
        super().__init__(**kwargs)
