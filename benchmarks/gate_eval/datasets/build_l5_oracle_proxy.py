"""Build the embedding-based proxy oracle surprisal labels.

Per L5_README.md Layer-1 spec: for each message in the dataset, compute
`proxy_surprisal(m) = 1 − max_cos(v_m, v_{m' ∈ prior_window_500})`.

Output: `l5_oracle_proxy__<dataset>.json` with:
    {
      "dataset": str,
      "model": "model2vec:potion-base-32M",
      "window": 500,
      "msg_labels": [
        {"msg_key": "conv-0/session_1/D1:1", "proxy_surprisal": 0.87,
         "content": "..."},
        ...
      ]
    }

Where `msg_key` uniquely identifies a message across the dataset. For
short_horizon the key format is `<conv_idx>/<session_k>/<dia_id>`. For
long_horizon the key format is `<session_id>/<turn_idx>`.

Usage:
    python benchmarks/gate_eval/datasets/build_l5_oracle_proxy.py \\
        --dataset short_horizon_200
    python benchmarks/gate_eval/datasets/build_l5_oracle_proxy.py \\
        --dataset long_horizon_synthetic

Runtime budget: ~30 s per dataset on CPU with Model2Vec. Zero API spend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT))


def iter_short_horizon(dataset: dict):
    """Yield (msg_key, content) tuples in chronological order."""
    for conv in dataset["convs"]:
        cidx = conv["conv_idx"]
        cdata = conv["conversation"]
        for k in sorted(cdata.keys()):
            if k.startswith("session_") and not k.endswith("_date_time"):
                for turn in cdata[k]:
                    text = turn.get("text", "")
                    dia_id = turn.get("dia_id", "?")
                    if text:
                        yield (f"conv-{cidx}/{k}/{dia_id}", text)


def iter_long_horizon(dataset: dict):
    for sess in dataset["sessions"]:
        sid = sess["session_id"]
        for i, turn in enumerate(sess.get("transcript", [])):
            text = turn.get("content", "")
            if text:
                yield (f"session-{sid}/turn-{i}", text)


def compute_proxy_surprisal(messages, window: int = 500):
    """Compute 1 − max_cos(v_m, v_prior_window) for each message."""
    try:
        from model2vec import StaticModel
    except ImportError:
        raise SystemExit(
            "model2vec not installed. This script uses the shipped Model2Vec "
            "embedder. Run: pip install model2vec"
        )

    import numpy as np

    model = StaticModel.from_pretrained("minishlab/potion-base-32M")
    texts = [m[1] for m in messages]
    if not texts:
        return []

    # Embed in one batch; Model2Vec is fast enough on CPU
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    # Normalize for cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
    embeddings = embeddings / norms

    surprisals = []
    for i in range(len(texts)):
        if i == 0:
            # First message: no prior → define surprisal = 1.0 (maximally novel)
            surprisals.append(1.0)
            continue
        lo = max(0, i - window)
        prior = embeddings[lo:i]
        # cos = dot product on normalized vectors
        sims = prior @ embeddings[i]
        max_sim = float(sims.max())
        # Clip max_sim to [-1, 1] just in case of FP drift
        max_sim = max(-1.0, min(1.0, max_sim))
        surprisals.append(round(1.0 - max_sim, 4))
    return surprisals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["short_horizon_200", "long_horizon_synthetic"])
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    input_path = DATASETS_DIR / f"{args.dataset}.json"
    if not input_path.exists():
        raise SystemExit(f"Dataset not found: {input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))

    iterator = (
        iter_short_horizon(data)
        if args.dataset == "short_horizon_200"
        else iter_long_horizon(data)
    )
    messages = list(iterator)
    print(f"  dataset: {args.dataset}; messages: {len(messages)}", file=sys.stderr)

    surprisals = compute_proxy_surprisal(messages, window=args.window)

    out = {
        "dataset": args.dataset,
        "model": "model2vec:minishlab/potion-base-32M",
        "metric": "1 - max_cos(v_m, v_prior_window)",
        "window": args.window,
        "n_messages": len(messages),
        "msg_labels": [
            {"msg_key": key, "proxy_surprisal": s, "content_preview": text[:80]}
            for (key, text), s in zip(messages, surprisals)
        ],
    }

    out_path = Path(args.output) if args.output else DATASETS_DIR / f"l5_oracle_proxy__{args.dataset}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  wrote: {out_path}", file=sys.stderr)
    print(f"  mean proxy surprisal: {sum(surprisals)/max(len(surprisals),1):.3f}",
          file=sys.stderr)
    print(f"  high-surprise (>0.5) msgs: "
          f"{sum(1 for s in surprisals if s > 0.5)}/{len(surprisals)}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
