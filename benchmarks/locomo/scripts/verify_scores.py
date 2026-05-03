#!/usr/bin/env python3
"""
Verify published LoCoMo benchmark scores.

Reads ALL result JSONs in results/ and recomputes accuracy from scratch.
No dependencies beyond Python stdlib.

Usage:
    python3 verify_scores.py
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Expected scores per result file (filename -> claimed accuracy %)
EXPECTED_SCORES = {
    "bm25_v2_run1.json": 80.5,
    "engram_v2_run1.json": 84.5,
    "evermemos_v2_run1.json": 94.5,
    "mem0_v2_run1.json": 61.4,
    "rag_v2_run1.json": 86.2,
    "supermemory_v2_run1.json": 65.4,
    "truememory_edge_v060_run1.json": 89.9,
    "truememory_edge_v060_run2.json": 89.5,
    "truememory_edge_v060_run3.json": 89.5,
    "truememory_base_v060_run2.json": 92.1,
    "truememory_base_v060_run3.json": 92.2,
    "truememory_pro_v060_run1.json": 92.8,
    "truememory_pro_v060_run2.json": 93.1,
    "truememory_pro_v060_run3.json": 93.1,
}


def verify_file(filepath: Path, claimed_score: float = None):
    """Verify a single result JSON against its claimed score."""
    if not filepath.exists():
        print(f"  NOT FOUND: {filepath.name}")
        return False

    with open(filepath) as f:
        data = json.load(f)

    total = data.get("total_questions", 0)
    details = data.get("details", [])

    if not details:
        print(f"  {filepath.name}: no 'details' array found")
        return False

    correct = sum(1 for q in details if q.get("correct") is True)
    acc = correct / total * 100 if total > 0 else 0

    print(f"  {filepath.name}: {correct}/{total} ({acc:.2f}%)")

    # Per-category breakdown
    by_cat = {}
    for q in details:
        cat = q.get("category", "unknown")
        by_cat.setdefault(cat, {"correct": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if q.get("correct"):
            by_cat[cat]["correct"] += 1

    for cat in sorted(by_cat.keys(), key=lambda x: str(x)):
        c = by_cat[cat]["correct"]
        t = by_cat[cat]["total"]
        print(f"    Cat {cat}: {c}/{t} ({c / t * 100:.1f}%)")

    if claimed_score is not None:
        print(f"\n  Recomputed: {acc:.2f}%  (claimed: {claimed_score}%)")
        if abs(acc - claimed_score) < 0.5:
            print("  VERIFIED")
            return True
        else:
            print(f"  MISMATCH -- expected ~{claimed_score}%, got {acc:.2f}%")
            return False
    else:
        print(f"\n  Recomputed: {acc:.2f}%  (no claimed score)")
        return True


def main():
    print("=" * 60)
    print("TRUEMEMORY BENCHMARK VERIFICATION")
    print("=" * 60)

    # Verify all .json files in results/
    all_ok = True
    result_files = sorted(RESULTS_DIR.glob("*.json"))

    if not result_files:
        print("\nNo result files found in results/")
        return

    for filepath in result_files:
        claimed = EXPECTED_SCORES.get(filepath.name)
        label = f" (claimed: {claimed}%)" if claimed else ""
        print(f"\n{filepath.name}{label}")
        print("-" * 40)
        ok = verify_file(filepath, claimed)
        if not ok:
            all_ok = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Files checked: {len(result_files)}")
    print(f"  Files with expected scores: {sum(1 for f in result_files if f.name in EXPECTED_SCORES)}")
    if all_ok:
        print("  Status: ALL VERIFIED")
    else:
        print("  Status: SOME MISMATCHES DETECTED")


if __name__ == "__main__":
    main()
