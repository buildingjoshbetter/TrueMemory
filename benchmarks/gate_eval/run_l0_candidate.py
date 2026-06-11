"""MEMORIST-L0 test harness — Phase 8.

Runs one L0 candidate against the multi-persona + human-probes datasets.
Emits a result JSON under benchmarks/gate_eval/results/l0_personality/.

Primary metrics (per L0_PREREGISTRATION §1, §2, §3):
  * personalization_lift_synthetic  — top-1 accuracy on A/B queries,
    synthetic corpus, relative to D1 floor.
  * personalization_lift_human      — same but on human-probe corpus.
    PRIMARY GATE.
  * slop_transfer_gap               — |synthetic - human|. PRIMARY.
  * intra_persona_consistency       — cosine-or-slot-agreement between
    profile built from first-half vs second-half of persona's corpus.
  * speaker_id_accuracy             — Type-C query accuracy.
  * cross_persona_leakage           — on Type-D queries: what fraction
    of top-k results come from must_not_surface_personas?
  * cold_start_curve                — accuracy at N=5,20,100,500,2000 messages.
  * profile_bytes                   — mean bytes/profile.
  * profile_readable                — 1 if candidate emits readable_summary.

Usage:
    python benchmarks/gate_eval/run_l0_candidate.py \\
        --candidate c1_baseline_hand_tuned \\
        --seed 42

Without --live-transcripts, falls back to human-probes only (synthetic
corpus is loaded if present, otherwise skipped with a note in the result JSON).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import pkgutil
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = REPO_ROOT / "benchmarks" / "gate_eval" / "datasets"
RESULTS = REPO_ROOT / "benchmarks" / "gate_eval" / "results" / "l0_personality"
CANDIDATES_PKG = "benchmarks.gate_eval.candidates.l0_personality"

sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gate_eval.candidates.l0_personality._base import (  # noqa: E402
    L0Candidate,
    Profile,
    RerankResult,
)


# ───────────────────────────── candidate discovery ──────────────────────────────
def discover_candidates() -> dict[str, type[L0Candidate]]:
    pkg_path = REPO_ROOT / "benchmarks" / "gate_eval" / "candidates" / "l0_personality"
    found: dict[str, type[L0Candidate]] = {}
    for m in pkgutil.iter_modules([str(pkg_path)]):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{CANDIDATES_PKG}.{m.name}")
        for attr in dir(mod):
            cls = getattr(mod, attr)
            if (isinstance(cls, type) and issubclass(cls, L0Candidate)
                    and cls is not L0Candidate):
                found[cls.name] = cls
    return found


# ─────────────────────────── dataset loading ────────────────────────────────────
def load_datasets() -> dict:
    ds: dict[str, Any] = {}
    ds["personas"] = json.loads((DATASETS / "l0_personas.json").read_text())
    ds["queries"] = json.loads((DATASETS / "l0_queries.json").read_text())
    ds["probes"] = json.loads((DATASETS / "l0_human_probes.json").read_text())
    synth = DATASETS / "l0_multi_persona_synthetic.json"
    if synth.exists():
        ds["synthetic"] = json.loads(synth.read_text())
    else:
        ds["synthetic"] = None
    return ds


# ─────────────────────── flatten per-persona message pools ──────────────────────
def build_message_pool(synthetic: dict | None,
                       probes: dict,
                       pool_kind: str) -> dict[str, list[dict]]:
    """Return {persona_id: [{"text","timestamp","source_persona_id"}]}."""
    pool: dict[str, list[dict]] = defaultdict(list)
    if pool_kind == "synthetic" and synthetic:
        for session in synthetic.get("sessions", []):
            pid = session["persona_id"]
            for idx, msg_text in enumerate(session["messages"]):
                pool[pid].append({
                    "text": msg_text,
                    "timestamp": f"{session['session_id']}_{idx:03d}",
                    "source_persona_id": pid,
                })
    elif pool_kind == "human":
        for probe in probes["probes"]:
            pid = probe["persona_id"]
            pool[pid].append({
                "text": probe["text"],
                "timestamp": probe["probe_id"],
                "source_persona_id": pid,
            })
    return dict(pool)


def mixed_pool(per_persona: dict[str, list[dict]]) -> list[dict]:
    """Merge all personas' messages into one pool (for retrieval scoring)."""
    out: list[dict] = []
    for msgs in per_persona.values():
        out.extend(msgs)
    return out


# ─────────────────────────────── scoring ─────────────────────────────────────────
def score_query_result(query: dict, results: list[RerankResult],
                       k: int = 5) -> dict:
    """Judge one query against the top-k reranked candidates.

    Returns {correct: bool, leakage_rate: float|None, notes: str}.
    """
    topk = results[:k]
    q_type = query["query_type"]

    if q_type in ("A", "B"):
        must_not = [x.lower() for x in query.get("must_not_contain", [])]
        gold_keywords = [x.lower() for x in query.get("gold_keywords", [])]
        gold_persona = query["persona_id"]
        # Relevance-aware top-k scoring: a query is correct iff ≥1 of the
        # top-k results satisfies (a) from gold persona, (b) contains no
        # must_not_contain, (c) contains ≥1 gold_keyword if any gold_keywords
        # are defined (otherwise persona-filter-only). This avoids the D0
        # tautology — if the top-k contains N messages from the gold persona
        # but none are topically relevant, the query is marked wrong.
        if not topk:
            return {"correct": False, "notes": "no results"}
        for r in topk:
            if r.source_persona_id != gold_persona:
                continue
            body = r.message_text.lower()
            if any(bad in body for bad in must_not):
                continue
            if gold_keywords and not any(kw in body for kw in gold_keywords):
                continue
            # First in-topk result that passes all filters wins.
            return {"correct": True,
                    "notes": f"matched rank-{topk.index(r)}"}
        return {"correct": False,
                "notes": "no in-topk result passed persona+content filters"}

    if q_type == "C":
        gold = query["gold_persona"]
        if not topk:
            return {"correct": False, "notes": "no results"}
        pred = topk[0].source_persona_id
        return {"correct": pred == gold, "notes": f"predicted {pred} gold {gold}"}

    if q_type == "D":
        forbidden = set(query.get("must_not_surface_personas", []))
        leaked = sum(1 for r in topk if r.source_persona_id in forbidden)
        rate = leaked / max(1, len(topk))
        # "correct" = zero leakage in top-k
        return {"correct": leaked == 0,
                "leakage_rate": rate,
                "notes": f"{leaked}/{len(topk)} leaked"}

    return {"correct": False, "notes": f"unknown type {q_type}"}


# ────────────────────── intra-persona consistency ────────────────────────────────
def profile_similarity(a: Profile, b: Profile) -> float:
    """Agreement between two profiles of the same persona.

    If both have a `style_vector`, cosine. Otherwise slot-agreement
    (Jaccard over stringified items in the data dict's lists).
    """
    if "style_vector" in a.data and "style_vector" in b.data:
        va = a.data["style_vector"]
        vb = b.data["style_vector"]
        if len(va) == len(vb):
            num = sum(x * y for x, y in zip(va, vb))
            na = math.sqrt(sum(x * x for x in va))
            nb = math.sqrt(sum(x * x for x in vb))
            if na > 0 and nb > 0:
                return num / (na * nb)
    # Slot agreement fallback: Jaccard over stringified leaf values
    def leaves(d: Any) -> set[str]:
        out = set()
        if isinstance(d, dict):
            for k, v in d.items():
                for child in leaves(v):
                    out.add(f"{k}={child}")
        elif isinstance(d, (list, tuple)):
            for item in d:
                out |= leaves(item)
        else:
            out.add(str(d))
        return out
    sa = leaves(a.data)
    sb = leaves(b.data)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ────────────────────────── cold-start curve ────────────────────────────────────
def cold_start_eval(candidate: L0Candidate,
                    per_persona: dict[str, list[dict]],
                    queries: list[dict],
                    candidate_pool: list[dict],
                    Ns: list[int]) -> dict[int, float]:
    """For each N, rebuild profiles from first-N messages per persona and
    measure A/B-query accuracy on the resulting profiles.
    """
    out = {}
    for N in Ns:
        profiles = {
            pid: candidate.build_profile(pid, msgs[:N])
            for pid, msgs in per_persona.items()
        }
        correct = 0
        total = 0
        for q in queries:
            if q["query_type"] not in ("A", "B"):
                continue
            pid = q["persona_id"]
            res = candidate.score_for_personalization(
                q["text"], profiles[pid], candidate_pool)
            out_score = score_query_result(q, res)
            if out_score["correct"]:
                correct += 1
            total += 1
        out[N] = correct / max(1, total)
    return out


# ───────────────────────────────── run ──────────────────────────────────────────
def run_candidate(candidate_name: str, seed: int = 42,
                  top_k: int = 5) -> dict:
    random.seed(seed)
    datasets = load_datasets()
    candidates = discover_candidates()
    if candidate_name not in candidates:
        raise SystemExit(
            f"Unknown candidate '{candidate_name}'. Available: {sorted(candidates)}"
        )
    cls = candidates[candidate_name]
    candidate = cls()

    personas = {p["id"]: p for p in datasets["personas"]["personas"]}
    queries = datasets["queries"]["queries"]

    # Build both pools (synthetic is optional until transcripts are generated)
    synth_pool = build_message_pool(datasets["synthetic"],
                                    datasets["probes"], "synthetic")
    human_pool = build_message_pool(None, datasets["probes"], "human")
    synth_cand_pool = mixed_pool(synth_pool) if synth_pool else []
    human_cand_pool = mixed_pool(human_pool)

    t_build0 = time.time()

    # Full profiles (from human probes only if synthetic is absent)
    full_source = synth_pool if synth_pool else human_pool
    full_profiles = {pid: candidate.build_profile(pid, msgs)
                     for pid, msgs in full_source.items()}
    t_build = time.time() - t_build0

    # ── scoring ──
    def score_pool(pool: list[dict], label: str) -> dict:
        if not pool:
            return {"skipped": True, "note": f"{label} pool empty"}
        t0 = time.time()
        per_query = []
        a_b_correct = 0
        a_b_total = 0
        c_correct = 0
        c_total = 0
        d_leaked = 0
        d_total = 0
        leakage_rates = []
        for q in queries:
            pid = q["persona_id"]
            profile = full_profiles.get(pid)
            if profile is None:
                continue
            res = candidate.score_for_personalization(
                q["text"], profile, pool)
            j = score_query_result(q, res, k=top_k)
            per_query.append({"query_id": q["query_id"],
                              "query_type": q["query_type"],
                              "correct": j["correct"],
                              "notes": j.get("notes")})
            if q["query_type"] in ("A", "B"):
                a_b_total += 1
                if j["correct"]:
                    a_b_correct += 1
            elif q["query_type"] == "C":
                c_total += 1
                if j["correct"]:
                    c_correct += 1
            elif q["query_type"] == "D":
                d_total += 1
                if not j["correct"]:
                    d_leaked += 1
                if "leakage_rate" in j:
                    leakage_rates.append(j["leakage_rate"])
        t = time.time() - t0
        return {
            "skipped": False,
            "pool_size": len(pool),
            "retrieval_wall_s": round(t, 3),
            "a_b_accuracy": a_b_correct / max(1, a_b_total),
            "speaker_id_accuracy": c_correct / max(1, c_total),
            "leakage_failure_rate": d_leaked / max(1, d_total),
            "mean_leakage_rate_top_k": (sum(leakage_rates) / max(1, len(leakage_rates))),
            "counts": {"A_B": a_b_total, "C": c_total, "D": d_total},
            "per_query_sample": per_query[:10],
        }

    human_scores = score_pool(human_cand_pool, "human")
    synth_scores = score_pool(synth_cand_pool, "synthetic")

    # Slop transfer gap (primary)
    slop_gap = None
    if not human_scores.get("skipped") and not synth_scores.get("skipped"):
        slop_gap = abs(synth_scores["a_b_accuracy"] - human_scores["a_b_accuracy"])

    # Intra-persona consistency
    consistency = []
    for pid, msgs in full_source.items():
        if len(msgs) < 4:
            continue
        mid = len(msgs) // 2
        a = candidate.build_profile(pid, msgs[:mid])
        b = candidate.build_profile(pid, msgs[mid:])
        consistency.append(profile_similarity(a, b))
    intra_persona_consistency = (sum(consistency) / len(consistency)
                                  if consistency else None)

    # Profile bytes + readability
    bytes_avg = (sum(p.bytes_estimate for p in full_profiles.values())
                 / max(1, len(full_profiles)))
    readable = all(p.readable_summary for p in full_profiles.values())

    # Cold-start curve (only if we have enough messages)
    cold_start = None
    if full_source:
        max_pool = max(len(v) for v in full_source.values())
        Ns = [n for n in (5, 20, 100, 500, 2000) if n <= max_pool]
        if Ns:
            cold_start = cold_start_eval(
                candidate, full_source, queries,
                mixed_pool(full_source), Ns)

    result = {
        "meta": {
            "candidate": candidate_name,
            "tier": candidate.tier,
            "seed": seed,
            "top_k": top_k,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "commit": "9b7af17",
        },
        "build_wall_s": round(t_build, 3),
        "profile_bytes_avg": round(bytes_avg, 1),
        "profile_readable": bool(readable),
        "intra_persona_consistency": intra_persona_consistency,
        "slop_transfer_gap": slop_gap,
        "human_probe_scores": human_scores,
        "synthetic_scores": synth_scores,
        "cold_start_curve": cold_start,
        "profile_summaries": {pid: p.readable_summary
                              for pid, p in full_profiles.items()},
    }

    candidate.cleanup()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", help="Candidate name (see --list)")
    parser.add_argument("--list", action="store_true",
                        help="List discovered candidates and exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.list:
        cands = discover_candidates()
        for name in sorted(cands):
            print(f"  {name}  ({cands[name].tier})")
        return 0
    if not args.candidate:
        parser.error("--candidate required (or pass --list to enumerate)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    result = run_candidate(args.candidate, seed=args.seed, top_k=args.top_k)
    out_path = Path(args.output) if args.output else \
        RESULTS / f"{args.candidate}__seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")
    print(f"  human A/B accuracy: {result['human_probe_scores'].get('a_b_accuracy')}")
    print(f"  synthetic A/B acc:  {result['synthetic_scores'].get('a_b_accuracy', 'n/a')}")
    print(f"  slop gap:           {result['slop_transfer_gap']}")
    print(f"  intra-persona cos:  {result['intra_persona_consistency']}")
    print(f"  profile bytes avg:  {result['profile_bytes_avg']}")
    print(f"  profile readable:   {result['profile_readable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
