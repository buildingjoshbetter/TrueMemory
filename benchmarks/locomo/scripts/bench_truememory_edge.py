#!/usr/bin/env python3
"""
LoCoMo Benchmark — TrueMemory Edge Tier
======================================
TrueMemory Edge tier using Model2Vec (potion-base-8M, 256-dim) embeddings
and cross-encoder reranker (ms-marco-MiniLM-L-6-v2, 22M params).
No HyDE, no GPU required. CPU-only, ~30M total params. Paper §2.0 target: 90.1%.

This is a fully self-contained Modal script. No local imports required.

Dependencies: truememory, sentence-transformers
Eval: openai/gpt-4.1-mini (answers) + openai/gpt-4o-mini (judge) via OpenRouter

Usage:
    modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-...

    modal run --detach bench_truememory_edge.py          # Full run (10 convs, 1540 Qs)
    modal run --detach bench_truememory_edge.py --smoke  # Smoke test (1 conv, 5 Qs)

    modal volume get locomo-results / ./results --force
"""
# ruff: noqa: E701, E702, E722
# This bench script uses a deliberately terse one-line-per-statement style
# to keep the Modal-shipped source compact. Style rules above are silenced
# for the file; correctness rules still apply.

import json
import modal
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Modal Setup ──────────────────────────────────────────────────────────

app = modal.App("locomo-truememory-edge")
vol = modal.Volume.from_name("locomo-results", create_if_missing=True)
VM = "/results"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

img = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai>=1.0", "truememory", "sentence-transformers"))

# ── Eval Config (IDENTICAL across all systems) ──────────────────────────

ANSWER_MODEL = "openai/gpt-4.1-mini"
ANSWER_MAX_TOKENS = 200
ANSWER_TEMPERATURE = 0
JUDGE_MODEL = "openai/gpt-4o-mini"
JUDGE_MAX_TOKENS = 10
JUDGE_TEMPERATURE = 0
NUM_JUDGE_RUNS = 3

# ── Shared Functions ─────────────────────────────────────────────────────

def mkc():
    import openai
    return openai.OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                         base_url=OPENROUTER_BASE_URL, timeout=60.0)

def _retry(fn, retries=5):
    for i in range(retries + 1):
        try: return fn()
        except Exception as e:
            if i >= retries or not any(k in str(e).lower() for k in
                ["connection","timeout","429","502","503","504","rate_limit"]): raise
            time.sleep(2 * (2**i))

ANSWER_PROMPT = """You are answering questions about personal conversations between friends.
You have been given retrieved conversation excerpts as context.

INSTRUCTIONS:
1. Read ALL context carefully — the answer may be spread across multiple excerpts
2. Look for specific names, dates, numbers, and details
3. Pay attention to who said what (speaker attribution matters)
4. For time questions, look for date mentions and temporal references
   - If someone says "last year" and the message is from 2023, that means 2022
   - If someone says "yesterday" on 2023-08-25, that means 2023-08-24
5. If multiple pieces of evidence exist, synthesize them
6. Give a concise, specific answer (1-2 sentences max)
7. If the context genuinely doesn't contain the answer, say "Not enough information"

Context:
{context}

Question: {question}

Think step by step, then give your final answer:"""

JUDGE_SYS = "You are a strict answer grader. Output ONLY valid JSON."
JUDGE_USR = """Determine if the generated answer is CORRECT or WRONG compared to the gold answer.
Be generous: if the generated answer mentions the same core topic/fact, mark CORRECT.
For time questions: same date/period in any format counts as CORRECT.

Question: {question}
Gold answer: {gold}
Generated answer: {generated}

Output ONLY: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""

def _verdict(c):
    c = c.strip()
    m = re.search(r'\{[^{}]*"label"\s*:\s*"([^"]*)"[^{}]*\}', c, re.IGNORECASE)
    if m: return m.group(1).strip().upper() == "CORRECT"
    return "CORRECT" in c.upper() and "WRONG" not in c.upper()

def gen_answer(client, ctx, q):
    def _c():
        return client.chat.completions.create(
            model=ANSWER_MODEL, max_tokens=ANSWER_MAX_TOKENS, temperature=ANSWER_TEMPERATURE,
            messages=[{"role":"user","content":ANSWER_PROMPT.format(context=ctx, question=q)}]
        ).choices[0].message.content
    try: return _retry(_c)
    except Exception as e: return f"ERROR: {e}"

def judge_one(client, q, gold, gen):
    """Judge a single answer. Returns (correct_bool, votes_list). Skips if answer is ERROR."""
    if gen.startswith("ERROR:"):
        return False, [False, False, False]
    up = JUDGE_USR.format(question=q, gold=gold, generated=gen)
    votes = []
    for _ in range(NUM_JUDGE_RUNS):
        def _j():
            return client.chat.completions.create(
                model=JUDGE_MODEL, max_tokens=JUDGE_MAX_TOKENS, temperature=JUDGE_TEMPERATURE,
                messages=[{"role":"system","content":JUDGE_SYS},{"role":"user","content":up}]
            ).choices[0].message.content
        try: votes.append(_verdict(_retry(_j)))
        except: votes.append(False)
    return sum(votes) > len(votes)/2, votes

def score_results(details):
    """Compute scores from pre-judged details."""
    cats = {1:"single_hop", 2:"multi_hop", 3:"temporal", 4:"open_domain"}
    by_cat = {}
    for cid, cn in cats.items():
        items = [d for d in details if d["category"] == cid]
        if items:
            c = sum(1 for d in items if d["correct"])
            by_cat[cn] = {"correct":c, "total":len(items), "accuracy":round(c/len(items)*100,1)}
    tc = sum(1 for d in details if d["correct"])
    return {"j_score": round(tc/len(details)*100,1) if details else 0,
            "total_correct":tc, "total_questions":len(details),
            "num_judge_runs":NUM_JUDGE_RUNS, "by_category":by_cat}

# ── LoCoMo Parsing ──────────────────────────────────────────────────────

def _pdt(s):
    for f in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y"):
        try: return datetime.strptime(s.strip(), f)
        except: pass
    return None

def _rtime(text, ds):
    dt = _pdt(ds)
    if not dt: return text
    for pat, d in [(r'\byesterday\b',1),(r'\blast week\b',7),(r'\blast month\b',30),
                   (r'\blast year\b',365),(r'\btwo years ago\b',730),(r'\ba year ago\b',365),
                   (r'\ba month ago\b',30),(r'\ba week ago\b',7),(r'\brecently\b',7)]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            d_str = (dt - timedelta(days=d)).strftime("%B %d, %Y")
            text = text[:m.start()] + f"{m.group(0)} (approximately {d_str})" + text[m.end():]
    return text

def parse_conv(conv):
    msgs, c = [], conv["conversation"]
    sa, sb = c["speaker_a"], c["speaker_b"]
    for sk in sorted([k for k in c if k.startswith("session_") and not k.endswith("_date_time")],
                     key=lambda k: int(k.split("_")[1])):
        ds = c.get(f"{sk}_date_time", "")
        sdt = _pdt(ds)
        for i, t in enumerate(c[sk]):
            sp = t["speaker"]
            ct = _rtime(t["text"], ds) if ds else t["text"]
            ts = (sdt+timedelta(seconds=30*i)).strftime("%Y-%m-%dT%H:%M:%S") if sdt else ds
            msgs.append({"content":ct,"speaker":sp,"recipient":sb if sp==sa else sa,
                         "timestamp":ts,"session":f"session_{sk.split('_')[1]}"})
    return msgs

def get_qa(conv): return [q for q in conv["qa"] if q["category"] != 5]
def fmsg(m): return f"[{m['timestamp']}] {m['speaker']} to {m['recipient']}: {m['content']}"

# ── TrueMemory Context Formatting ─────────────────────────────────────────

def _nm_format_ctx(results):
    """Format TrueMemory results with metadata — matches v2 scripts exactly."""
    parts = []
    for r in results:
        sender = r.get("sender", "?")
        ts = r.get("timestamp", "")
        cat = r.get("category", "")
        modality = r.get("modality", "")
        meta = f"[{sender}"
        if ts: meta += f" | {ts}"
        if cat: meta += f" | {cat}"
        if modality and modality != "conversation": meta += f" | {modality}"
        meta += "]"
        parts.append(f"{meta} {r['content']}")
    return "\n\n".join(parts)

# ── TrueMemory Edge Retrieval ─────────────────────────────────────────────

def retrieve_truememory_edge(conv_data, conv_idx):
    from truememory.vector_search import set_embedding_model
    set_embedding_model("edge")
    from truememory.engine import TrueMemoryEngine
    from truememory.reranker import get_reranker
    import tempfile
    get_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    msgs = parse_conv(conv_data)
    _tmp_db_file = tempfile.NamedTemporaryFile(suffix=".db", prefix=f"edge_{conv_idx}_", delete=False)
    tmp_db = _tmp_db_file.name
    _tmp_db_file.close()
    engine = TrueMemoryEngine(db_path=tmp_db)
    # Ingest
    import json as _json
    _tmp_json_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp_json = _tmp_json_file.name
    _tmp_json_file.close()
    msg_dicts = [{"content":m["content"],"sender":m["speaker"],"recipient":m["recipient"],
                  "timestamp":m["timestamp"],"category":m["session"],"modality":"conversation"}
                 for m in msgs]
    with open(tmp_json,"w") as f: _json.dump(msg_dicts, f)
    engine.ingest(tmp_json)
    # Retrieve
    results = []
    for qa in get_qa(conv_data):
        sr = engine.search_agentic(qa["question"], limit=100, use_hyde=False, use_reranker=True)
        ctx = _nm_format_ctx(sr)
        results.append((qa["question"], qa["category"], qa["answer"], ctx or "No results found."))
    engine.close()
    os.unlink(tmp_db)
    for ext in ("-wal","-shm"):
        try: os.unlink(tmp_db + ext)
        except: pass
    os.unlink(tmp_json)
    return results

# ── Worker ───────────────────────────────────────────────────────────────

def _bench_conv(conv_data, conv_idx, smoke=False):
    """Run one conversation through retrieval -> answer -> judge pipeline."""
    sid = conv_data.get("sample_id", f"conv_{conv_idx}")
    qas = get_qa(conv_data)
    n_qs = 5 if smoke else len(qas)

    print(f"  [truememory_edge] Conv {conv_idx} ({sid}): {len(parse_conv(conv_data))} msgs, {n_qs} Qs")

    # Step 1: Retrieve
    t0 = time.time()
    try:
        all_retrieved = retrieve_truememory_edge(conv_data, conv_idx)
    except Exception as e:
        print(f"    RETRIEVAL FAILED: {e}")
        return [{"question":q["question"],"category":q["category"],"gold_answer":q["answer"],
                 "generated_answer":f"ERROR: retrieval failed: {e}","correct":False,
                 "judge_votes":[False]*3,"num_retrieved":0,"conversation_id":sid}
                for q in qas[:n_qs]]
    print(f"    Retrieved in {time.time()-t0:.1f}s")

    # Step 2: Answer + Judge (with error guard + latency tracking)
    client = mkc()
    details = []
    for i, (question, category, gold, ctx) in enumerate(all_retrieved[:n_qs]):
        t_ans = time.time()
        answer = gen_answer(client, ctx, question)
        answer_latency = time.time() - t_ans

        t_jdg = time.time()
        correct, votes = judge_one(client, question, gold, answer)
        judge_latency = time.time() - t_jdg

        details.append({"question":question, "category":category, "gold_answer":gold,
                        "generated_answer":answer, "correct":correct, "judge_votes":votes,
                        "num_retrieved": ctx.count("[Result") + ctx.count("[Memory"),
                        "conversation_id":sid,
                        "answer_latency_s": round(answer_latency, 2),
                        "judge_latency_s": round(judge_latency, 2)})
        if (i+1) % 25 == 0:
            c = sum(1 for d in details if d["correct"])
            print(f"    {i+1}/{n_qs}: {c}/{i+1} ({c/(i+1)*100:.0f}%)")

    retrieval_time = time.time() - t0
    c = sum(1 for d in details if d["correct"])
    print(f"    Conv {conv_idx} done: {c}/{len(details)} correct, retrieval={retrieval_time:.1f}s")
    return details

@app.function(image=img, secrets=[modal.Secret.from_name("openrouter-key")],
              timeout=14400, memory=8192)
def worker(conv_data, conv_idx, smoke=False):
    return _bench_conv(conv_data, conv_idx, smoke)

# ── Orchestrator ─────────────────────────────────────────────────────────

@app.function(image=img, secrets=[modal.Secret.from_name("openrouter-key")],
              timeout=28800, memory=2048, volumes={VM: vol})
def orchestrate(locomo_data: list, smoke: bool = False):
    """Run TrueMemory Edge, checkpointing after each conversation."""
    system = "truememory_edge"
    ckpt_path = f"{VM}/{system}_checkpoint.json"
    result_path = f"{VM}/{system}_v2_run1.json"
    n_convs = 1 if smoke else len(locomo_data)
    mode = "SMOKE TEST" if smoke else "FULL RUN"

    run_start = time.time()
    print(f"\n{'='*60}")
    print(f"{system.upper()} — {mode} ({n_convs} conversations)")
    print(f"{'='*60}")

    # Resume from checkpoint
    all_details = []
    done_convs = set()
    try:
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        all_details = ckpt.get("details", [])
        done_convs = set(ckpt.get("done_convs", []))
        print(f"  Resuming: {len(done_convs)} convs done, {len(all_details)} answers")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Process conversations (parallel spawn, sequential collect)
    pending = {}
    for ci in range(n_convs):
        if ci in done_convs:
            print(f"  Conv {ci}: SKIPPED (checkpoint)")
            continue
        pending[ci] = worker.spawn(locomo_data[ci], ci, smoke)

    for ci, handle in pending.items():
        try:
            conv_details = handle.get()
            all_details.extend(conv_details)
            done_convs.add(ci)
            # Checkpoint
            with open(ckpt_path, "w") as f:
                json.dump({"details": all_details, "done_convs": sorted(done_convs)}, f)
            vol.commit()
            c = sum(1 for d in conv_details if d["correct"])
            print(f"  Conv {ci} saved: {c}/{len(conv_details)} correct (checkpoint: {len(done_convs)}/{n_convs})")
        except Exception as e:
            print(f"  Conv {ci} FAILED: {e}")

    if not all_details:
        print(f"  {system}: NO RESULTS")
        return {"system": system, "error": "no results"}

    # Compute final scores
    scores = score_results(all_details)
    print(f"\n  {system.upper()}: {scores['j_score']}% ({scores['total_correct']}/{scores['total_questions']})")
    for cn, cd in scores["by_category"].items():
        print(f"    {cn:15s}: {cd['accuracy']}% ({cd['correct']}/{cd['total']})")

    # Compute latency stats
    total_time = time.time() - run_start
    ans_lats = [d.get("answer_latency_s",0) for d in all_details if d.get("answer_latency_s")]
    jdg_lats = [d.get("judge_latency_s",0) for d in all_details if d.get("judge_latency_s")]
    timing = {
        "total_wall_clock_s": round(total_time, 1),
        "avg_answer_latency_s": round(sum(ans_lats)/len(ans_lats), 2) if ans_lats else 0,
        "avg_judge_latency_s": round(sum(jdg_lats)/len(jdg_lats), 2) if jdg_lats else 0,
        "p95_answer_latency_s": round(sorted(ans_lats)[int(len(ans_lats)*0.95)] if ans_lats else 0, 2),
        "p95_judge_latency_s": round(sorted(jdg_lats)[int(len(jdg_lats)*0.95)] if jdg_lats else 0, 2),
    }
    print(f"\n  Timing: {timing}")

    # Save final result
    result = {
        "system": system, "version": "v3-checkpointed", "run": 1,
        "answer_model": ANSWER_MODEL, "answer_max_tokens": ANSWER_MAX_TOKENS,
        "answer_temperature": ANSWER_TEMPERATURE, "judge_model": JUDGE_MODEL,
        "judge_max_tokens": JUDGE_MAX_TOKENS, "judge_temperature": JUDGE_TEMPERATURE,
        "smoke_test": smoke, "timing": timing, **scores, "details": all_details,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print(f"  Saved: {result_path}")

    # Clean up checkpoint
    try: os.remove(ckpt_path); vol.commit()
    except: pass

    return {"system": system, "j_score": scores["j_score"],
            "total": scores["total_questions"], "correct": scores["total_correct"]}

# ── Entry Point ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(smoke: bool = False, dataset: str = None):
    if dataset is None:
        for c in [Path(__file__).parent.parent / "data" / "locomo10.json"]:
            if c.exists(): dataset = str(c); break
    if not dataset:
        print("ERROR: locomo10.json not found"); sys.exit(1)
    with open(dataset) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"LoCoMo Benchmark — TrueMemory Edge (CPU) — {'SMOKE TEST' if smoke else 'FULL RUN'}")
    print(f"{'='*60}")
    print(f"  Mode:     {'1 conv x 5 Qs' if smoke else '10 convs x 1540 Qs'}")
    print(f"  Answer:   {ANSWER_MODEL} via OpenRouter")
    print(f"  Judge:    {JUDGE_MODEL} via OpenRouter")
    print("  Volume:   locomo-results")
    print(f"{'='*60}\n")

    orchestrate.spawn(data, smoke)
    print("  Launched on Modal. Results save to Volume 'locomo-results'.")
    print("  Download: modal volume get locomo-results / ./results --force")
