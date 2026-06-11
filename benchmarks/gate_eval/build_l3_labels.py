"""
Build per-message retrieval-utility labels for L3 evaluation.

Two datasets, two extractors (different gold-evidence shapes):

  short_horizon_200.json
      Direct LoCoMo per-message evidence: qa[i]['evidence'] = ['D{s}:{m}', ...]
      Each message has dia_id matching one of these tags exactly.
      → Trivial join.

  long_horizon_synthetic.json
      Session-level grounding: query['gold_from_session'] = S, query['gold_answer'] = "..."
      No per-message tags.
        - First pass: keyword-overlap (≥3 content words) between message.content
          and gold_answer.
        - Fallback (when overlap is sparse): mark all messages in gold_from_session
          as utility-positive (session-level loose label). Justified by the
          synthetic dataset's session-level grounding — answers are paraphrased
          across multiple turns, not concentrated in a single message.

Output schema (per message):
    {
      "msg_id":       int,                # row index in flat-list
      "conv_id":      str,                # conversation/session identifier
      "dia_id":       str | None,         # if available
      "ts":           int,                # message ordinal within conv (sec proxy)
      "speaker":      str,
      "content":      str,
      "modality":     str,                # "text" by default
      "utility_binary":  0 | 1,
      "utility_weighted": float,          # rank-aware weighted sum
      "supporting_qids": list[int],       # which queries this msg supports
    }

Saves:
  benchmarks/gate_eval/datasets/l3_short_horizon_200_labels.json
  benchmarks/gate_eval/datasets/l3_long_horizon_synthetic_labels.json

Plus per-dataset summary stats (positive rate, per-conversation positive
counts) appended to the JSON under "_meta".
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"

_STOP = frozenset(
    "a an the of in on at to for and or but with from by is are was were "
    "be been being it this that these those i you he she we they me him her "
    "us them my your his its our their as if then so do does did will would "
    "can could should may might must shall not no yes ok okay".split()
)


def _content_words(text: str) -> set[str]:
    """Lowercase content words >=4 chars, stopwords removed."""
    toks = re.findall(r"[A-Za-z][A-Za-z']+", text.lower())
    return {t for t in toks if len(t) >= 4 and t not in _STOP}


# --------------------------------------------------------------------------
# Short-horizon labeler — direct LoCoMo evidence join
# --------------------------------------------------------------------------

def label_short_horizon(path: Path) -> dict[str, Any]:
    src = json.loads(path.read_text())
    qa = src["qa"]
    convs = src["convs"]

    by_conv = {c["sample_id"]: c for c in convs}
    out_messages: list[dict[str, Any]] = []
    msg_id = 0

    # Build per-message records first.
    # Key by (conv_id, dia_id) for fast evidence lookup.
    msg_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for c in convs:
        conv_id = c["sample_id"]
        conv = c["conversation"]
        # Iterate sessions in order to get a per-conv ordinal timestamp.
        session_keys = sorted(
            [k for k in conv.keys() if re.fullmatch(r"session_\d+", k)],
            key=lambda x: int(x.split("_")[1]),
        )
        ord_ts = 0
        for sk in session_keys:
            for m in conv[sk]:
                ord_ts += 1
                rec = {
                    "msg_id": msg_id,
                    "conv_id": conv_id,
                    "dia_id": m.get("dia_id"),
                    "ts": ord_ts,
                    "speaker": m.get("speaker", ""),
                    "content": m.get("text", m.get("clean_text", "")),
                    "modality": "text",
                    "utility_binary": 0,
                    "utility_weighted": 0.0,
                    "supporting_qids": [],
                }
                out_messages.append(rec)
                msg_by_key[(conv_id, rec["dia_id"])] = rec
                msg_id += 1

    # Now apply evidence labels from QA.
    for qi, q in enumerate(qa):
        cid = q["conv_id"]
        ev = q.get("evidence", [])
        if not ev:
            continue
        weight = 1.0 / len(ev)
        for tag in ev:
            key = (cid, tag)
            rec = msg_by_key.get(key)
            if rec is None:
                continue
            rec["utility_binary"] = 1
            rec["utility_weighted"] += weight
            rec["supporting_qids"].append(qi)

    # Summary stats
    n_total = len(out_messages)
    n_pos = sum(1 for m in out_messages if m["utility_binary"])
    by_conv_pos = {}
    for m in out_messages:
        by_conv_pos.setdefault(m["conv_id"], [0, 0])
        by_conv_pos[m["conv_id"]][0] += 1
        if m["utility_binary"]:
            by_conv_pos[m["conv_id"]][1] += 1

    return {
        "_meta": {
            "source": str(path.name),
            "labeler": "loCoMo_evidence_direct_join",
            "n_messages": n_total,
            "n_positive": n_pos,
            "positive_rate": n_pos / n_total if n_total else 0.0,
            "by_conv": {k: {"n": v[0], "pos": v[1]} for k, v in by_conv_pos.items()},
        },
        "messages": out_messages,
    }


# --------------------------------------------------------------------------
# Long-horizon labeler — keyword-overlap with gold_answer
# --------------------------------------------------------------------------

MIN_OVERLAP = 3  # content-word overlap required


def label_long_horizon(path: Path) -> dict[str, Any]:
    src = json.loads(path.read_text())
    sessions = src["sessions"]
    queries = src["retrieval_queries"]

    # Per-session ordered messages
    session_by_id = {s["session_id"]: s for s in sessions}

    out_messages: list[dict[str, Any]] = []
    # key: (session_id, ordinal_in_session)
    msg_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    msg_id = 0

    for s in sessions:
        sid = s["session_id"]
        for ord_in_session, m in enumerate(s.get("transcript", [])):
            rec = {
                "msg_id": msg_id,
                "conv_id": s["persona_name"],  # group by persona for LOCO-CV
                "session_id": sid,
                "ord_in_session": ord_in_session,
                "ts": s["day_offset"] * 1000 + ord_in_session,
                "speaker": m.get("role", ""),
                "content": m.get("content", ""),
                "modality": "text",
                "utility_binary": 0,
                "utility_weighted": 0.0,
                "supporting_qids": [],
            }
            out_messages.append(rec)
            msg_by_key[(sid, ord_in_session)] = rec
            msg_id += 1

    # Apply keyword-overlap labels.
    n_q_with_match = 0
    n_q_total = 0
    for q in queries:
        n_q_total += 1
        gold_session = q.get("gold_from_session")
        gold_answer = q.get("gold_answer", "")
        if not gold_session or not gold_answer:
            continue
        gold_words = _content_words(gold_answer)
        if len(gold_words) < MIN_OVERLAP:
            # Fall back to substring match on the literal answer.
            gold_lower = gold_answer.lower()
            sess = session_by_id.get(gold_session)
            if not sess:
                continue
            matched_any = False
            matched = []
            for ord_in_session, m in enumerate(sess.get("transcript", [])):
                if gold_lower in m.get("content", "").lower():
                    matched.append((sid, ord_in_session))
                    matched_any = True
            if matched_any:
                n_q_with_match += 1
                weight = 1.0 / max(1, len(matched))
                for (sid, oid) in matched:
                    key = (gold_session, oid)
                    rec = msg_by_key.get(key)
                    if rec:
                        rec["utility_binary"] = 1
                        rec["utility_weighted"] += weight
                        rec["supporting_qids"].append(q["query_id"])
            continue

        sess = session_by_id.get(gold_session)
        if not sess:
            continue
        # First pass: per-message keyword-overlap.
        matched: list[int] = []
        for ord_in_session, m in enumerate(sess.get("transcript", [])):
            mwords = _content_words(m.get("content", ""))
            if len(mwords & gold_words) >= MIN_OVERLAP:
                matched.append(ord_in_session)
        # Fallback: session-level loose label.
        if not matched:
            matched = list(range(len(sess.get("transcript", []))))
        if matched:
            n_q_with_match += 1
            weight = 1.0 / len(matched)
            for oid in matched:
                key = (gold_session, oid)
                rec = msg_by_key.get(key)
                if rec:
                    rec["utility_binary"] = 1
                    rec["utility_weighted"] += weight
                    rec["supporting_qids"].append(q["query_id"])

    n_total = len(out_messages)
    n_pos = sum(1 for m in out_messages if m["utility_binary"])
    by_conv_pos = {}
    for m in out_messages:
        by_conv_pos.setdefault(m["conv_id"], [0, 0])
        by_conv_pos[m["conv_id"]][0] += 1
        if m["utility_binary"]:
            by_conv_pos[m["conv_id"]][1] += 1

    return {
        "_meta": {
            "source": str(path.name),
            "labeler": f"keyword_overlap_min{MIN_OVERLAP}_or_substring_fallback",
            "n_messages": n_total,
            "n_positive": n_pos,
            "positive_rate": n_pos / n_total if n_total else 0.0,
            "n_queries_total": n_q_total,
            "n_queries_with_match": n_q_with_match,
            "by_conv": {k: {"n": v[0], "pos": v[1]} for k, v in by_conv_pos.items()},
        },
        "messages": out_messages,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    short_in = DATASETS / "short_horizon_200.json"
    long_in = DATASETS / "long_horizon_synthetic.json"

    short_out = DATASETS / "l3_short_horizon_200_labels.json"
    long_out = DATASETS / "l3_long_horizon_synthetic_labels.json"

    print(f"[1/2] Labeling {short_in.name} ...")
    short = label_short_horizon(short_in)
    short_out.write_text(json.dumps(short, indent=2))
    sm = short["_meta"]
    print(f"  → {sm['n_messages']} msgs, {sm['n_positive']} positive "
          f"({sm['positive_rate']:.2%})")

    print(f"[2/2] Labeling {long_in.name} ...")
    lng = label_long_horizon(long_in)
    long_out.write_text(json.dumps(lng, indent=2))
    lm = lng["_meta"]
    print(f"  → {lm['n_messages']} msgs, {lm['n_positive']} positive "
          f"({lm['positive_rate']:.2%}) — "
          f"{lm['n_queries_with_match']}/{lm['n_queries_total']} queries matched")

    print("\nPer-conv positive distribution (short):")
    for cid, info in sm["by_conv"].items():
        print(f"  {cid}: {info['pos']:>3} / {info['n']:>4}")

    print("\nPer-persona positive distribution (long):")
    for cid, info in lm["by_conv"].items():
        print(f"  {cid}: {info['pos']:>3} / {info['n']:>4}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
