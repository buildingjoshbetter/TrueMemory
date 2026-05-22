"""Smoke tests for the MEMORIST gate-eval harness scaffolding.

These tests verify the harness's structural plumbing (candidate
discovery, dataset loading, result-JSON shape) WITHOUT actually
running an LLM-driven candidate end-to-end (that's reserved for
Phase 9 which will exercise the full pipeline against the real
datasets).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_candidate_discovery_finds_v05_baseline():
    from benchmarks.gate_eval.run_candidate import discover_candidates

    cands = discover_candidates()
    assert "v05_baseline_nogate" in cands, f"Expected v05_baseline_nogate in {sorted(cands)}"
    assert "v05_gate_threshold" in cands, f"Expected v05_gate_threshold in {sorted(cands)}"


def test_short_horizon_dataset_shape():
    """The committed dataset must round-trip with the documented shape."""
    path = REPO_ROOT / "benchmarks" / "gate_eval" / "datasets" / "short_horizon_200.json"
    if not path.exists():
        # Allow the test to skip when the dataset hasn't been built (CI etc.)
        import pytest
        pytest.skip(f"{path.name} not present (run build_short_horizon_200.py first)")

    data = json.loads(path.read_text())
    assert "qa" in data and "convs" in data and "meta" in data
    assert len(data["qa"]) == 200, f"Expected 200 QA, got {len(data['qa'])}"
    assert data["meta"]["n_actual"] == 200
    assert data["meta"]["seed"] == 42
    # Every QA must reference an included conv
    conv_idxs = {c["conv_idx"] for c in data["convs"]}
    for qa in data["qa"]:
        assert qa["conv_idx"] in conv_idxs
        assert qa["category"] in (1, 2, 3, 4)


def test_candidate_base_interface():
    """The Candidate ABC enforces the three required methods."""
    from benchmarks.gate_eval.candidates._base import Candidate

    # Cannot instantiate the abstract base directly
    try:
        Candidate()
    except TypeError:
        pass
    else:
        raise AssertionError("Candidate base should be uninstantiable (ABC)")


def test_v05_baseline_does_not_load_truememory_at_construction():
    """First-load cost should be deferred so harness `--list` is fast."""
    from benchmarks.gate_eval.candidates.v05_baseline_nogate import V05BaselineNogate

    c = V05BaselineNogate()
    assert c._pipeline is None
    assert c._memory is None
    assert c.name == "v05_baseline_nogate"


if __name__ == "__main__":
    test_candidate_discovery_finds_v05_baseline()
    test_short_horizon_dataset_shape()
    test_candidate_base_interface()
    test_v05_baseline_does_not_load_truememory_at_construction()
    print("All harness smoke tests passed.")
