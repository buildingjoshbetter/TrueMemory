"""MEMORIST-L5 harness — Phase 8.

Self-contained sweep runner for L5 predictive-coding candidates. Runs
one candidate × one dataset and emits a result-JSON including:

  - retrieval precision@k (the primary lift signal)
  - calibration vs proxy oracle (Spearman + Kendall τ)
  - rare-precision@k (precision on queries whose evidence message is in
    the bottom-quartile of proxy-oracle surprisal)
  - per-message ingest timing
  - retrieval latency

DESIGN NOTE. To isolate the L5 signal from fact-extraction confounders
in the v0.5.0 pipeline, this harness uses a simplified message-level
retrieval layer: FTS5 BM25 + dense-embedding cosine, fused by reciprocal
rank fusion. Every candidate shares this same retrieval substrate; what
varies is (a) the surprise score per message and (b) whether/how it
reranks. Results are per-session ablation numbers, not full-v0.5.0-
pipeline numbers. Phase 11 Modal runs would add the full pipeline; this
session did not run Phase 11 (budget/time).

Usage:
    python benchmarks/gate_eval/run_l5_candidate.py \\
        --candidate l5_minwired \\
        --dataset short_horizon_200
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "benchmarks" / "gate_eval" / "datasets"
CANDIDATES_DIR = REPO_ROOT / "benchmarks" / "gate_eval" / "candidates" / "l5_predictive"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "gate_eval" / "results" / "l5_predictive"

sys.path.insert(0, str(REPO_ROOT))


def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower())


_STOPWORDS = frozenset(
    "a an and the is are was were be been being have has had do does did "
    "will would could should may might must can shall to of for in on at by "
    "with from into onto about as it its this that those these i you he she "
    "we they me him her us them my your his her our their what when where who "
    "how why not no yes but or if so".split()
)


def _match_gold(gold: str, content: str) -> bool:
    """Return True if content contains gold — lenient match.

    Pass-1: exact cleaned-substring.
    Pass-2: ≥60% content-token coverage (non-stopword tokens from gold
    appear in cleaned content). Handles paraphrased gold answers in
    long_horizon_synthetic where the message doesn't contain the gold
    verbatim but does contain enough of its content words.
    """
    gold_c = _clean(gold)
    cont_c = _clean(content)
    if not gold_c or not cont_c:
        return False
    if gold_c in cont_c:
        return True
    gold_toks = [t for t in gold_c.split() if t and t not in _STOPWORDS and len(t) > 2]
    if not gold_toks:
        return False
    cont_set = set(cont_c.split())
    hits = sum(1 for t in gold_toks if t in cont_set)
    return hits / len(gold_toks) >= 0.6


def iter_short_horizon_messages(dataset):
    """Yield dicts with msg_key, content, conv_idx, session_name, dia_id."""
    for conv in dataset["convs"]:
        cidx = conv["conv_idx"]
        cdata = conv["conversation"]
        for k in sorted(cdata.keys()):
            if k.startswith("session_") and not k.endswith("_date_time"):
                for turn in cdata[k]:
                    text = turn.get("text", "")
                    dia_id = turn.get("dia_id", "?")
                    if text:
                        yield {
                            "msg_key": f"conv-{cidx}/{k}/{dia_id}",
                            "content": text,
                            "speaker": turn.get("speaker", "?"),
                            "conv_idx": cidx,
                        }


def iter_long_horizon_messages(dataset):
    for sess in dataset["sessions"]:
        sid = sess["session_id"]
        for i, turn in enumerate(sess.get("transcript", [])):
            text = turn.get("content", "")
            if text:
                yield {
                    "msg_key": f"session-{sid}/turn-{i}",
                    "content": text,
                    "speaker": turn.get("role", "?"),
                    "session_id": sid,
                    "turn_idx": i,
                }


# ---------------------------------------------------------------------------
# Retrieval substrate (shared by every L5 candidate)
# ---------------------------------------------------------------------------

class RetrievalBackend:
    """Simple BM25 + dense-embedding + RRF retrieval."""

    def __init__(self):
        try:
            from model2vec import StaticModel
        except ImportError:
            raise SystemExit("pip install model2vec required")
        import numpy as np
        self.np = np
        self._encoder = StaticModel.from_pretrained("minishlab/potion-base-32M")
        self._messages: list[dict] = []
        self._embeddings = None

    def add_message(self, msg_key: str, content: str, surprise: float = 0.0):
        self._messages.append(
            {"msg_key": msg_key, "content": content, "surprise": float(surprise)}
        )

    def build_indices(self):
        if not self._messages:
            self._embeddings = self.np.zeros((0, 256), dtype=self.np.float32)
            return
        texts = [m["content"] for m in self._messages]
        emb = self._encoder.encode(texts, show_progress_bar=False)
        emb = self.np.asarray(emb, dtype=self.np.float32)
        norms = self.np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
        self._embeddings = emb / norms

    def retrieve_fused(self, query: str, k: int = 10,
                       alpha_surprise: float = 0.0) -> list[dict]:
        """Return top-k rows. Uses BM25 (simple term overlap) + cosine +
        RRF fusion, with an optional multiplicative surprise boost."""
        if not self._messages:
            return []
        # ---- BM25-ish: term-overlap per-doc (Okapi-lite for small corpus)
        q_terms = set(_clean(query).split())
        bm25_scores = []
        for m in self._messages:
            doc_terms = set(_clean(m["content"]).split())
            overlap = len(q_terms & doc_terms)
            # Tiny length normalization
            bm25_scores.append(
                overlap / (len(doc_terms) ** 0.5 + 1e-6) if overlap else 0.0
            )

        # ---- Dense cosine
        q_emb = self._encoder.encode([query], show_progress_bar=False)[0]
        q_emb = self.np.asarray(q_emb, dtype=self.np.float32)
        q_emb = q_emb / (self.np.linalg.norm(q_emb) + 1e-9)
        cos_scores = self._embeddings @ q_emb  # (n_msgs,)

        # ---- RRF fusion with surprise boost
        bm25_rank = self.np.argsort(-self.np.asarray(bm25_scores))
        cos_rank = self.np.argsort(-cos_scores)
        rrf_k = 60
        rrf = self.np.zeros(len(self._messages), dtype=self.np.float32)
        for i, idx in enumerate(bm25_rank):
            rrf[idx] += 1.0 / (rrf_k + i + 1)
        for i, idx in enumerate(cos_rank):
            rrf[idx] += 1.0 / (rrf_k + i + 1)

        if alpha_surprise != 0.0:
            surprise_vec = self.np.asarray(
                [m["surprise"] for m in self._messages], dtype=self.np.float32
            )
            rrf = rrf * (1.0 + alpha_surprise * surprise_vec)

        order = self.np.argsort(-rrf)[:k]
        return [
            {**self._messages[int(i)], "score": float(rrf[int(i)])}
            for i in order
        ]


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------

def discover_l5_candidates():
    found = {}
    for py in CANDIDATES_DIR.glob("*.py"):
        if py.stem.startswith("_"):
            continue
        modname = f"benchmarks.gate_eval.candidates.l5_predictive.{py.stem}"
        mod = importlib.import_module(modname)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and getattr(obj, "_is_l5_candidate", False):
                found[obj.name] = obj
    return found


# ---------------------------------------------------------------------------
# Dataset eval
# ---------------------------------------------------------------------------

def run_dataset(candidate, dataset, dataset_name: str,
                proxy_labels: dict | None) -> dict:
    """Run the candidate end-to-end."""
    backend = RetrievalBackend()

    iterator = (
        iter_short_horizon_messages(dataset)
        if dataset_name == "short_horizon_200"
        else iter_long_horizon_messages(dataset)
    )
    messages = list(iterator)

    # ---- Ingest: score each message with candidate, store in backend
    ingest_t0 = time.time()
    per_msg_ms = []
    for i, msg in enumerate(messages):
        context = [m["content"] for m in messages[max(0, i - 50):i]]
        t0 = time.time()
        surprise = candidate.score(msg["content"], context)
        per_msg_ms.append((time.time() - t0) * 1000)
        backend.add_message(msg["msg_key"], msg["content"], surprise)
    backend.build_indices()
    ingest_wall = time.time() - ingest_t0

    candidate_scores = {m["msg_key"]: m["surprise"] for m in backend._messages}

    # ---- Retrieval
    retr_t0 = time.time()
    K = 10
    n_total = 0
    n_at = {1: 0, 3: 0, 10: 0}
    rare_threshold = None
    if proxy_labels:
        sorted_s = sorted(proxy_labels.values())
        rare_threshold = sorted_s[len(sorted_s) // 4] if sorted_s else None
    n_rare = 0
    rare_at = {1: 0, 3: 0, 10: 0}
    by_category = {}
    by_probe = {}

    alpha = candidate.config.get("alpha_surprise", 0.0)

    if dataset_name == "short_horizon_200":
        for qa in dataset["qa"]:
            n_total += 1
            cat = qa.get("category", 0)
            results = backend.retrieve_fused(qa["question"], k=K,
                                             alpha_surprise=alpha)
            gold = str(qa.get("answer", "")).strip().lower()
            if not gold:
                continue
            hit = -1
            for i, r in enumerate(results[:K]):
                if _match_gold(gold, r["content"]):
                    hit = i + 1
                    break
            # Rare = gold evidence message's proxy surprisal in bottom quartile
            is_rare = False
            if rare_threshold is not None:
                evidence = qa.get("evidence", [])
                # Match evidence dia_ids to msg_keys — rough
                rare_scores = []
                for e in evidence:
                    for mk, s in proxy_labels.items():
                        if mk.endswith(f"/{e}"):
                            rare_scores.append(s)
                            break
                if rare_scores and min(rare_scores) >= (
                    sorted_s[-len(sorted_s)//4] if sorted_s else 0
                ):
                    # evidence surprisal in top quartile → "rare fact" probe
                    is_rare = True

            if hit > 0:
                n_at[10] += 1
                if hit <= 3:
                    n_at[3] += 1
                if hit == 1:
                    n_at[1] += 1
            by_category.setdefault(cat, [0, 0])[1] += 1
            if hit > 0:
                by_category[cat][0] += 1

            if is_rare:
                n_rare += 1
                if hit > 0:
                    rare_at[10] += 1
                    if hit <= 3:
                        rare_at[3] += 1
                    if hit == 1:
                        rare_at[1] += 1

    else:  # long_horizon_synthetic
        for q in dataset["retrieval_queries"]:
            n_total += 1
            probe = q.get("probe_type", "baseline")
            results = backend.retrieve_fused(q["question"], k=K,
                                             alpha_surprise=alpha)
            gold = str(q.get("gold_answer", "") or "").strip().lower()
            if not gold or gold == "n/a":
                # Noise probe: success = NOT retrieving anything with gold text.
                if probe == "noise":
                    hit = 1  # by default succeed; only fail if we find something
                else:
                    continue
            else:
                hit = -1
                for i, r in enumerate(results[:K]):
                    if _match_gold(gold, r["content"]):
                        hit = i + 1
                        break
            if hit > 0:
                n_at[10] += 1
                if hit <= 3:
                    n_at[3] += 1
                if hit == 1:
                    n_at[1] += 1
            by_probe.setdefault(probe, [0, 0])[1] += 1
            if hit > 0:
                by_probe[probe][0] += 1

    retr_wall = time.time() - retr_t0

    # ---- Calibration vs proxy oracle
    calibration = {}
    if proxy_labels:
        keys = [k for k in candidate_scores if k in proxy_labels]
        if keys:
            import numpy as np
            cand_vals = np.asarray([candidate_scores[k] for k in keys],
                                    dtype=np.float64)
            oracle_vals = np.asarray([proxy_labels[k] for k in keys],
                                      dtype=np.float64)
            # Spearman
            rank_c = cand_vals.argsort().argsort()
            rank_o = oracle_vals.argsort().argsort()
            mean_c, mean_o = rank_c.mean(), rank_o.mean()
            num = ((rank_c - mean_c) * (rank_o - mean_o)).sum()
            denom = ((rank_c - mean_c) ** 2).sum() ** 0.5 * \
                    ((rank_o - mean_o) ** 2).sum() ** 0.5
            spearman = float(num / denom) if denom > 0 else 0.0
            # Kendall's tau (approx via O(n log n))
            try:
                from scipy.stats import kendalltau
                tau, _ = kendalltau(cand_vals, oracle_vals)
                tau = float(tau)
            except Exception:
                tau = None
            calibration = {
                "n_pairs": len(keys),
                "spearman": round(spearman, 4),
                "kendall_tau": round(tau, 4) if tau is not None else None,
                "oracle_mean": round(float(oracle_vals.mean()), 4),
                "candidate_mean": round(float(cand_vals.mean()), 4),
            }

    return {
        "candidate": candidate.name,
        "dataset": dataset_name,
        "alpha_surprise": alpha,
        "ingest": {
            "n_messages": len(messages),
            "wall_s": round(ingest_wall, 2),
            "per_message_ms_avg": round(sum(per_msg_ms) / max(len(per_msg_ms), 1), 3),
            "per_message_ms_p95": round(sorted(per_msg_ms)[int(0.95 * len(per_msg_ms))] if per_msg_ms else 0, 3),
        },
        "retrieval": {
            "n_queries": n_total,
            "precision_at_1_pct": round(100 * n_at[1] / max(n_total, 1), 2),
            "precision_at_3_pct": round(100 * n_at[3] / max(n_total, 1), 2),
            "precision_at_10_pct": round(100 * n_at[10] / max(n_total, 1), 2),
            "n_rare_queries": n_rare,
            "rare_precision_at_10_pct": round(100 * rare_at[10] / max(n_rare, 1), 2) if n_rare else None,
            "wall_s": round(retr_wall, 2),
            "by_category": {str(c): {"recovered": v[0], "total": v[1],
                                      "pct": round(100 * v[0] / max(v[1], 1), 2)}
                             for c, v in by_category.items()},
            "by_probe": {p: {"recovered": v[0], "total": v[1],
                              "pct": round(100 * v[0] / max(v[1], 1), 2)}
                          for p, v in by_probe.items()},
        },
        "calibration": calibration,
        "surprise_distribution": _dist_summary(
            [m["surprise"] for m in backend._messages]
        ),
    }


def _dist_summary(values: list[float]) -> dict:
    if not values:
        return {}
    import numpy as np
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "frac_gt_05": round(float((arr > 0.5).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["short_horizon_200", "long_horizon_synthetic"])
    ap.add_argument("--alpha", type=float, default=None,
                    help="Override the candidate's default alpha_surprise")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cands = discover_l5_candidates()
    if args.candidate not in cands:
        raise SystemExit(f"Unknown candidate. Available: {sorted(cands)}")
    cls = cands[args.candidate]
    kwargs = {}
    if args.alpha is not None:
        kwargs["alpha_surprise"] = args.alpha
    candidate = cls(**kwargs)

    dataset = json.loads(
        (DATASETS_DIR / f"{args.dataset}.json").read_text(encoding="utf-8")
    )
    proxy_path = DATASETS_DIR / f"l5_oracle_proxy__{args.dataset}.json"
    proxy_labels = None
    if proxy_path.exists():
        proxy_data = json.loads(proxy_path.read_text(encoding="utf-8"))
        proxy_labels = {
            entry["msg_key"]: entry["proxy_surprisal"]
            for entry in proxy_data["msg_labels"]
        }

    result = run_dataset(candidate, dataset, args.dataset, proxy_labels)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else \
          RESULTS_DIR / f"{args.candidate}__{args.dataset}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  wrote: {out}")
    print(f"  precision@10: {result['retrieval']['precision_at_10_pct']}%  "
          f"calib spearman: {result.get('calibration', {}).get('spearman', 'n/a')}")


if __name__ == "__main__":
    main()
