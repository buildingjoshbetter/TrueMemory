"""Per-persona breakdown for L0 result JSONs.

Re-runs each candidate against the human probes but reports A/B accuracy
per persona so we can see whether a candidate's aggregate number masks
persona-specific wins or losses.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gate_eval.run_l0_candidate import (  # noqa: E402
    build_message_pool,
    discover_candidates,
    load_datasets,
    mixed_pool,
    score_query_result,
)

OUT = REPO_ROOT / "benchmarks" / "gate_eval" / "results" / "l0_personality" / "_per_persona.json"


def run_per_persona(candidate_names: list[str], top_k: int = 5) -> dict:
    ds = load_datasets()
    queries = ds["queries"]["queries"]
    pool_by_persona = build_message_pool(None, ds["probes"], "human")
    pool = mixed_pool(pool_by_persona)
    personas = sorted(pool_by_persona)
    out: dict = {"candidates": {}}
    cands = discover_candidates()
    for name in candidate_names:
        if name not in cands:
            print(f"skip {name}: not found", file=sys.stderr)
            continue
        cand = cands[name]()
        profiles = {pid: cand.build_profile(pid, msgs)
                    for pid, msgs in pool_by_persona.items()}
        by_persona: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0,
                                                             "by_type": defaultdict(lambda: [0, 0])})
        for q in queries:
            pid = q["persona_id"]
            if q["query_type"] not in ("A", "B"):
                continue
            res = cand.score_for_personalization(q["text"], profiles[pid], pool)
            j = score_query_result(q, res, k=top_k)
            rec = by_persona[pid]
            rec["total"] += 1
            rec["by_type"][q["query_type"]][1] += 1
            if j["correct"]:
                rec["correct"] += 1
                rec["by_type"][q["query_type"]][0] += 1
        by_persona_clean = {}
        for pid in personas:
            rec = by_persona.get(pid, {"correct": 0, "total": 0, "by_type": {}})
            by_persona_clean[pid] = {
                "a_b_accuracy": rec["correct"] / max(1, rec["total"]),
                "correct": rec["correct"], "total": rec["total"],
                "by_type": {t: {"correct": v[0], "total": v[1],
                                "acc": v[0] / max(1, v[1])}
                             for t, v in rec["by_type"].items()},
            }
        out["candidates"][name] = by_persona_clean
        cand.cleanup()
    OUT.write_text(json.dumps(out, indent=2))
    return out


def print_table(data: dict) -> None:
    cands = list(data["candidates"].keys())
    personas = sorted(next(iter(data["candidates"].values())).keys())
    print(f"{'candidate':<34}", end="")
    for p in personas:
        print(f"{p:>8}", end="")
    print("  avg")
    print("-" * (34 + 8 * len(personas) + 6))
    for name in cands:
        print(f"{name:<34}", end="")
        accs = []
        for p in personas:
            acc = data["candidates"][name][p]["a_b_accuracy"]
            accs.append(acc)
            print(f"{acc:>8.3f}", end="")
        print(f"  {sum(accs)/len(accs):.3f}")


if __name__ == "__main__":
    names = ["d1_no_l0", "d0_user_filter_only", "c1_baseline_hand_tuned",
             "c5_expanded_keywords", "c3c_char_ngram_proxy",
             "c3_style_embedder_wegmann"]
    data = run_per_persona(names)
    print_table(data)
    print(f"\nWrote {OUT}")
