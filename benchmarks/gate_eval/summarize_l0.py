"""Aggregate all L0 result JSONs into a summary table + per-persona breakdown.

Reads:  benchmarks/gate_eval/results/l0_personality/*.json
Writes: stdout table + benchmarks/gate_eval/results/l0_personality/_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

RESULTS = Path(__file__).resolve().parent / "results" / "l0_personality"
SUMMARY = RESULTS / "_summary.json"

# Null-band thresholds from L0_PREREGISTRATION §4
LIFT_FLOOR = 0.05
CONSISTENCY_FLOOR = 0.70
SLOP_CEIL = 0.20


def load_all() -> list[dict]:
    out = []
    for p in RESULTS.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            out.append(json.loads(p.read_text()))
        except Exception as exc:
            print(f"skip {p.name}: {exc}")
    return out


def summarize(results: list[dict]) -> dict:
    # Find D0 as the hard-baseline comparator
    d0 = next((r for r in results if r["meta"]["candidate"] == "d0_user_filter_only"), None)
    d1 = next((r for r in results if r["meta"]["candidate"] == "d1_no_l0"), None)

    def get_hab(r): return r.get("human_probe_scores", {}).get("a_b_accuracy")
    d0_hab = get_hab(d0) if d0 else None
    d1_hab = get_hab(d1) if d1 else None

    rows = []
    for r in results:
        m = r["meta"]
        hp = r.get("human_probe_scores", {})
        sp = r.get("synthetic_scores", {})
        hab = get_hab(r)
        sab = sp.get("a_b_accuracy") if not sp.get("skipped") else None
        consistency = r.get("intra_persona_consistency")
        slop_gap = r.get("slop_transfer_gap")
        lift_vs_d0 = (hab - d0_hab) if (hab is not None and d0_hab is not None) else None
        lift_vs_d1 = (hab - d1_hab) if (hab is not None and d1_hab is not None) else None

        # Null-band pass / fail (against D0 baseline)
        null_band_pass = (
            lift_vs_d0 is not None
            and lift_vs_d0 >= LIFT_FLOOR
            and (consistency is None or consistency >= CONSISTENCY_FLOOR)
            and (slop_gap is None or slop_gap <= SLOP_CEIL)
        )

        rows.append({
            "candidate": m["candidate"],
            "tier": m["tier"],
            "human_a_b_acc": hab,
            "synthetic_a_b_acc": sab,
            "lift_vs_d0": lift_vs_d0,
            "lift_vs_d1": lift_vs_d1,
            "slop_gap": slop_gap,
            "intra_consistency": consistency,
            "speaker_id_acc_human": hp.get("speaker_id_accuracy"),
            "leakage_fail_rate_human": hp.get("leakage_failure_rate"),
            "profile_bytes_avg": r.get("profile_bytes_avg"),
            "profile_readable": r.get("profile_readable"),
            # Partial null-band per REPORT §7.1: only 2 of 4 pre-reg
            # conditions (lift, consistency) are measurable in-session;
            # conditions 3 (slop_gap) and 4 (Welch p) require the
            # deferred synthetic corpus + LOCO-CV.
            "null_band_partial_vs_d0": null_band_pass,
        })
    rows.sort(key=lambda r: (r["candidate"]))

    # Per-persona breakdown is reconstructable from the per_query_sample (caps at 10)
    # — for the full table we'd need to re-run with a flag. For now, capture the
    # per-query sample we have.
    samples = {r["meta"]["candidate"]: r.get("human_probe_scores", {}).get("per_query_sample", [])
                for r in results}

    return {"rows": rows, "per_query_samples": samples,
            "baselines": {"d0_hab": d0_hab, "d1_hab": d1_hab},
            "thresholds": {
                "lift_floor": LIFT_FLOOR,
                "consistency_floor": CONSISTENCY_FLOOR,
                "slop_ceiling": SLOP_CEIL,
            }}


def print_table(summary: dict) -> None:
    print(f"{'candidate':<32} {'tier':<5} {'A/B':>6} {'Δ_D0':>7} "
          f"{'Δ_D1':>7} {'cons':>6} {'lkg':>6} {'bytes':>6} {'null-band':>10}")
    print("-" * 95)
    for r in summary["rows"]:
        def f(x, fmt=".3f"):
            if x is None:
                return "   —  "
            return format(x, fmt)
        print(f"{r['candidate']:<32} {r['tier']:<5} "
              f"{f(r['human_a_b_acc']):>6} "
              f"{f(r['lift_vs_d0'], '+.3f'):>7} "
              f"{f(r['lift_vs_d1'], '+.3f'):>7} "
              f"{f(r['intra_consistency']):>6} "
              f"{f(r['leakage_fail_rate_human']):>6} "
              f"{r['profile_bytes_avg'] or 0:>6.0f} "
              f"{'partial' if r['null_band_partial_vs_d0'] else 'fail':>10}")
    b = summary["baselines"]
    print(f"\nBaselines: D0={b['d0_hab']:.3f}  D1={b['d1_hab']:.3f}")
    print(f"Null band: lift≥{summary['thresholds']['lift_floor']} "
          f"& consistency≥{summary['thresholds']['consistency_floor']} "
          f"& slop_gap≤{summary['thresholds']['slop_ceiling']}")


def main() -> int:
    results = load_all()
    if not results:
        print("No result JSONs found.")
        return 1
    summary = summarize(results)
    print_table(summary)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
