"""
Build the SHORT_HORIZON_200 evaluation dataset from LoCoMo.

Phase 7.1 of the MEMORIST research session.

Samples 200 questions from `benchmarks/locomo/data/locomo10.json` proportionally
across the 4 numeric categories the TrueMemory paper reports on (1, 2, 3, 4).
Category 5 (open_domain / adversarial) is excluded — that's the 1,986 - 1,540
discrepancy noted in the journal: the paper reports on 1,540 questions, which
equals the four-category total (282 + 321 + 96 + 841).

Proportional distribution applied to N=200:
    Cat 1 (single_hop?)  282/1540 = 18.3% → 37
    Cat 2 (temporal?)    321/1540 = 20.8% → 42
    Cat 3 (?)             96/1540 =  6.2% → 12
    Cat 4 (multi_hop?)   841/1540 = 54.6% → 109
                                    ─────────
                                    200

We don't need to know the human-readable category names to do the sampling
(the QA harness uses the numeric IDs anyway), but we record them for
traceability.

Output: `benchmarks/gate_eval/datasets/short_horizon_200.json` with shape:
    {
        "qa": [200 entries with conv_id ↔ conversation linkage],
        "convs": [unique LoCoMo conversations the 200 questions reference],
        "meta": {seed, built_at, src, category_distribution, ...}
    }

This file is committed and IMMUTABLE from the moment it lands.
"""

from __future__ import annotations

import datetime
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
N_TARGET = 200
INCLUDED_CATEGORIES = (1, 2, 3, 4)  # exclude cat 5 per TrueMemory paper convention

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json"
OUT = REPO_ROOT / "benchmarks" / "gate_eval" / "datasets" / "short_horizon_200.json"


def main() -> None:
    rng = random.Random(SEED)

    src_data = json.loads(SRC.read_text(encoding="utf-8"))

    # Index every QA with its source conv. The LoCoMo file is a list of 10
    # conversations; each has `qa`, `conversation`, `sample_id`, etc.
    pool: list[dict] = []
    for conv_idx, conv in enumerate(src_data):
        sample_id = conv.get("sample_id", f"conv_{conv_idx}")
        for qa_idx, qa in enumerate(conv["qa"]):
            cat = qa.get("category")
            if cat not in INCLUDED_CATEGORIES:
                continue
            pool.append({
                "conv_id": sample_id,
                "conv_idx": conv_idx,
                "qa_idx": qa_idx,
                "question": qa["question"],
                "answer": qa.get("answer", ""),
                "evidence": qa.get("evidence", []),
                "category": cat,
            })

    # Distribution check
    cat_counts = Counter(qa["category"] for qa in pool)
    total = sum(cat_counts.values())
    assert total == 1540, (
        f"Expected 1,540 questions in cats {INCLUDED_CATEGORIES}, got {total}. "
        "LoCoMo dataset shape may have changed; re-verify before continuing."
    )

    # Proportional sampling — round to nearest int, then top-up the largest
    # bucket to land on exactly N_TARGET.
    by_cat = defaultdict(list)
    for qa in pool:
        by_cat[qa["category"]].append(qa)

    # Deterministic shuffle within each category (so re-running gives the
    # same 200 questions).
    for cat in by_cat:
        rng.shuffle(by_cat[cat])

    target_per_cat = {
        cat: round(N_TARGET * cat_counts[cat] / total)
        for cat in INCLUDED_CATEGORIES
    }
    # Adjust to land on exactly N_TARGET — assign the rounding remainder to
    # the largest bucket (cat 4) to minimize representation error.
    delta = N_TARGET - sum(target_per_cat.values())
    if delta != 0:
        biggest = max(INCLUDED_CATEGORIES, key=lambda c: cat_counts[c])
        target_per_cat[biggest] += delta

    sampled: list[dict] = []
    for cat in INCLUDED_CATEGORIES:
        n = target_per_cat[cat]
        sampled.extend(by_cat[cat][:n])

    assert len(sampled) == N_TARGET, f"Got {len(sampled)}, expected {N_TARGET}"

    # Collect the unique conversations referenced by the 200 questions and
    # serialize them (so the dataset is self-contained — no need to ship
    # locomo10.json alongside).
    referenced_conv_idxs = sorted({qa["conv_idx"] for qa in sampled})
    convs_out: list[dict] = []
    for conv_idx in referenced_conv_idxs:
        conv = src_data[conv_idx]
        convs_out.append({
            "conv_idx": conv_idx,
            "sample_id": conv.get("sample_id", f"conv_{conv_idx}"),
            "conversation": conv["conversation"],
            "session_summary": conv.get("session_summary", {}),
        })

    out_payload = {
        "qa": sampled,
        "convs": convs_out,
        "meta": {
            "seed": SEED,
            "n_target": N_TARGET,
            "n_actual": len(sampled),
            "src": str(SRC.relative_to(REPO_ROOT)),
            "src_total_qa": sum(len(c["qa"]) for c in src_data),
            "src_eligible_qa": total,
            "included_categories": list(INCLUDED_CATEGORIES),
            "category_target_distribution": target_per_cat,
            "category_actual_distribution": dict(Counter(q["category"] for q in sampled)),
            "n_referenced_convs": len(referenced_conv_idxs),
            "referenced_conv_ids": [c["sample_id"] for c in convs_out],
            "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "phase": "memorist_phase_7_1",
            "immutable": True,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"  n_qa: {len(sampled)}  n_convs: {len(convs_out)}")
    print(f"  category distribution: {out_payload['meta']['category_actual_distribution']}")


if __name__ == "__main__":
    main()
