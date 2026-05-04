#!/usr/bin/env python3
"""
LongMemEval Benchmark — TrueMemory Pro Tier (A100 GPU, Modal)
==========================================================
TrueMemory Pro with Qwen3-Embedding-0.6B, gte-reranker-modernbert-base,
and HyDE via OpenRouter. Matches LoCoMo evaluation methodology exactly.

Dataset: LongMemEval oracle (500 questions, 6 types)
Eval: openai/gpt-4.1-mini (answers) + openai/gpt-4o-mini (judge, 3-vote) via OpenRouter

Usage:
    modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-...

    modal run --detach bench_truememory_pro.py                    # Full (500 Qs)
    modal run --detach bench_truememory_pro.py --smoke            # Smoke (20 Qs)
    modal run --detach bench_truememory_pro.py --run-id 2         # Run 2

    modal volume get longmemeval-results / ./results --force
"""

import json, modal, os, re, sys, time
from datetime import datetime
from pathlib import Path

# ── Modal Setup ──────────────────────────────────────────────────────────

app = modal.App("longmemeval-truememory-pro-s")
vol = modal.Volume.from_name("longmemeval-results", create_if_missing=True)
VM = "/results"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

img = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai>=1.0", "truememory[gpu]>=0.6.0", "sentence-transformers"))

# ── Eval Config (IDENTICAL to LoCoMo methodology) ────────────────────────

ANSWER_MODEL = "openai/gpt-4.1-mini"
ANSWER_MAX_TOKENS = 200
ANSWER_TEMPERATURE = 0
JUDGE_MODEL = "openai/gpt-4o-mini"
JUDGE_MAX_TOKENS = 10
JUDGE_TEMPERATURE = 0
NUM_JUDGE_RUNS = 3
TOP_K = 100

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

ANSWER_PROMPT = """You are a memory assistant answering questions about past conversations.
You have been given retrieved conversation excerpts as context.

INSTRUCTIONS:
1. Read ALL context carefully — the answer may be spread across multiple excerpts
2. Look for specific names, dates, numbers, and details
3. Pay attention to who said what (speaker attribution matters)
4. For time questions, look for date mentions and temporal references
5. If someone's preference or situation CHANGED over time, give the LATEST information
6. If multiple pieces of evidence exist, synthesize them
7. Give a concise, specific answer (1-2 sentences max)
8. If the context genuinely doesn't contain the answer, say "I don't have that information in our conversation history"

Current date: {question_date}

Context:
{context}

Question: {question}

Think step by step, then give your final answer:"""

JUDGE_SYS = "You are a strict answer grader. Output ONLY valid JSON."
JUDGE_USR = """Determine if the generated answer is CORRECT or WRONG compared to the gold answer.
Be generous: if the generated answer mentions the same core topic/fact, mark CORRECT.
For time questions: same date/period in any format counts as CORRECT.
For preferences: if the core preference is captured, mark CORRECT.
For knowledge updates: the response should reflect the LATEST information.

Question: {question}
Gold answer: {gold}
Generated answer: {generated}

Output ONLY: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""

def _verdict(c):
    c = c.strip()
    m = re.search(r'\{[^{}]*"label"\s*:\s*"([^"]*)"[^{}]*\}', c, re.IGNORECASE)
    if m: return m.group(1).strip().upper() == "CORRECT"
    return "CORRECT" in c.upper() and "WRONG" not in c.upper()

def gen_answer(client, ctx, q, question_date):
    def _c():
        return client.chat.completions.create(
            model=ANSWER_MODEL, max_tokens=ANSWER_MAX_TOKENS, temperature=ANSWER_TEMPERATURE,
            messages=[{"role":"user","content":ANSWER_PROMPT.format(
                context=ctx, question=q, question_date=question_date)}]
        ).choices[0].message.content
    try: return _retry(_c)
    except Exception as e: return f"ERROR: {e}"

def judge_one(client, q, gold, gen):
    if gen.startswith("ERROR:"):
        return False, [False] * NUM_JUDGE_RUNS
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

def judge_abstention(hypothesis):
    """Abstention questions use keyword matching (standard for LongMemEval)."""
    refusal = ["don't have", "no information", "not mentioned", "not available",
               "cannot find", "no record", "wasn't discussed", "not discussed",
               "don't recall", "no conversation", "not in our"]
    return any(p in hypothesis.lower() for p in refusal)

# ── Data Conversion ──────────────────────────────────────────────────────

def sessions_to_messages(question_data):
    messages = []
    sessions = question_data.get("haystack_sessions", [])
    dates = question_data.get("haystack_dates", [])
    for si, session in enumerate(sessions):
        session_date = dates[si] if si < len(dates) else ""
        for ti, turn in enumerate(session):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            sender = "user" if role == "user" else "assistant"
            recipient = "assistant" if role == "user" else "user"
            ts = session_date
            messages.append({
                "content": content, "sender": sender, "recipient": recipient,
                "timestamp": ts, "category": f"session_{si}", "modality": "conversation",
            })
    return messages

def _tm_format_ctx(results):
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

def _tm_make_hyde_fn():
    client = mkc()
    def _call(prompt):
        resp = client.chat.completions.create(
            model="openai/gpt-4.1-mini", max_tokens=300, temperature=0.3,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    return _call

# ── Worker (processes a batch of questions) ──────────────────────────────

def _bench_batch(questions, smoke=False):
    from truememory.vector_search import set_embedding_model
    set_embedding_model("pro")
    from truememory.engine import TrueMemoryEngine
    from truememory.reranker import get_reranker, set_active_tier
    set_active_tier("pro")
    import tempfile

    get_reranker(model_name="Alibaba-NLP/gte-reranker-modernbert-base")
    llm_fn = _tm_make_hyde_fn()
    client = mkc()
    results = []

    for qi, qdata in enumerate(questions):
        qid = qdata["question_id"]
        is_abstention = qid.endswith("_abs")
        messages = sessions_to_messages(qdata)

        if not messages:
            results.append({
                "question_id": qid, "question_type": qdata["question_type"],
                "correct": False, "hypothesis": "NO MESSAGES",
                "gold": qdata["answer"], "judge_votes": [False]*3,
            })
            continue

        tmp_db = tempfile.mktemp(suffix=".db", prefix=f"lme_{qi}_")
        tmp_json = tempfile.mktemp(suffix=".json", prefix="lme_msgs_")
        try:
            # Ingest
            with open(tmp_json, "w") as f:
                json.dump(messages, f)
            engine = TrueMemoryEngine(db_path=tmp_db)
            engine.ingest(tmp_json)

            # Retrieve with agentic search (HyDE + reranker)
            sr = engine.search_agentic(qdata["question"], limit=TOP_K,
                                       llm_fn=llm_fn, use_hyde=True,
                                       use_reranker=True)
            ctx = _tm_format_ctx(sr)

            # Generate
            question_date = qdata.get("question_date", "")
            hypothesis = gen_answer(client, ctx or "No results found.",
                                    qdata["question"], question_date)

            # Judge
            if is_abstention:
                correct = judge_abstention(hypothesis)
                votes = [correct] * 3
            else:
                correct, votes = judge_one(client, qdata["question"],
                                           qdata["answer"], hypothesis)

            results.append({
                "question_id": qid, "question_type": qdata["question_type"],
                "correct": correct, "hypothesis": hypothesis,
                "gold": qdata["answer"], "judge_votes": votes,
                "num_retrieved": len(sr), "num_messages": len(messages),
            })

            engine.close()
        except Exception as e:
            results.append({
                "question_id": qid, "question_type": qdata["question_type"],
                "correct": False, "hypothesis": f"ERROR: {e}",
                "gold": qdata["answer"], "judge_votes": [False]*3,
            })
            try: engine.close()
            except: pass
        finally:
            for p in [tmp_db, tmp_db+"-wal", tmp_db+"-shm", tmp_json]:
                try: os.unlink(p)
                except: pass

        if (qi+1) % 10 == 0:
            c = sum(1 for r in results if r["correct"])
            print(f"    [{qi+1}/{len(questions)}] {c}/{qi+1} ({c/(qi+1)*100:.0f}%)")

    return results

@app.function(image=img, secrets=[modal.Secret.from_name("openrouter-key")],
              timeout=14400, memory=8192, gpu="A10G")
def worker(questions, batch_idx, smoke=False):
    print(f"  [Worker {batch_idx}] Processing {len(questions)} questions")
    return _bench_batch(questions, smoke)

# ── Orchestrator ─────────────────────────────────────────────────────────

@app.function(image=img, secrets=[modal.Secret.from_name("openrouter-key")],
              timeout=28800, memory=2048, volumes={VM: vol})
def orchestrate(dataset: list, run_id: int = 1, smoke: bool = False):
    system = "truememory_pro_longmemeval_s"
    ckpt_path = f"{VM}/{system}_r{run_id}_checkpoint.json"
    result_path = f"{VM}/{system}_run{run_id}.json"

    if smoke:
        dataset = dataset[:20]
    n_qs = len(dataset)
    mode = f"{'SMOKE' if smoke else 'FULL'} — Run {run_id}"

    run_start = time.time()
    print(f"\n{'='*60}")
    print(f"LONGMEMEVAL — NEUROMEM PRO — {mode} ({n_qs} questions)")
    print(f"{'='*60}")

    # Resume from checkpoint
    all_results = []
    done_ids = set()
    try:
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        all_results = ckpt.get("results", [])
        done_ids = {r["question_id"] for r in all_results}
        print(f"  Resuming: {len(done_ids)}/{n_qs} done")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Filter out already-done questions
    remaining = [q for q in dataset if q["question_id"] not in done_ids]
    print(f"  Remaining: {len(remaining)} questions")

    if remaining:
        # Split into batches of ~50 for parallelism (10 workers)
        batch_size = max(1, len(remaining) // 10)
        batches = []
        for i in range(0, len(remaining), batch_size):
            batches.append(remaining[i:i+batch_size])

        # Spawn workers
        handles = {}
        for bi, batch in enumerate(batches):
            handles[bi] = worker.spawn(batch, bi, smoke)

        # Collect results
        for bi, handle in handles.items():
            try:
                batch_results = handle.get()
                all_results.extend(batch_results)
                done_ids.update(r["question_id"] for r in batch_results)
                # Checkpoint
                with open(ckpt_path, "w") as f:
                    json.dump({"results": all_results}, f)
                vol.commit()
                c = sum(1 for r in batch_results if r["correct"])
                print(f"  Batch {bi} done: {c}/{len(batch_results)} correct "
                      f"(total: {len(done_ids)}/{n_qs})")
            except Exception as e:
                print(f"  Batch {bi} FAILED: {e}")

    if not all_results:
        print("  NO RESULTS")
        return {"error": "no results"}

    # Score
    total = len(all_results)
    correct = sum(1 for r in all_results if r["correct"])
    acc = round(correct / total * 100, 1) if total else 0

    # Per-type
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r["question_type"]].append(r)

    type_scores = {}
    print(f"\n{'='*60}")
    print(f"RESULTS — Run {run_id}: {correct}/{total} ({acc}%)")
    print(f"{'='*60}")
    for qt in sorted(by_type):
        items = by_type[qt]
        c = sum(1 for r in items if r["correct"])
        a = round(c / len(items) * 100, 1) if items else 0
        print(f"  {qt:35s}: {c}/{len(items)} ({a}%)")
        type_scores[qt] = {"correct": c, "total": len(items), "accuracy": a}

    total_time = time.time() - run_start
    result = {
        "system": "truememory_pro", "benchmark": "LongMemEval_s",
        "version": "v3-modal", "run": run_id,
        "answer_model": ANSWER_MODEL, "answer_max_tokens": ANSWER_MAX_TOKENS,
        "judge_model": JUDGE_MODEL, "num_judge_runs": NUM_JUDGE_RUNS,
        "smoke_test": smoke,
        "overall_accuracy": acc, "total_correct": correct,
        "total_questions": total, "by_type": type_scores,
        "elapsed_s": round(total_time, 1),
        "details": all_results,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print(f"\n  Saved: {result_path}")

    try: os.remove(ckpt_path); vol.commit()
    except: pass

    return {"run": run_id, "accuracy": acc, "correct": correct, "total": total}

# ── Entry Point ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(smoke: bool = False, run_id: int = 1, dataset: str = None):
    if dataset is None:
        for c in [Path(__file__).parent / "data" / "longmemeval_s.json"]:
            if c.exists(): dataset = str(c); break
    if not dataset:
        print("ERROR: longmemeval_s.json not found"); sys.exit(1)
    with open(dataset) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"LongMemEval — TrueMemory Pro (A100) — {'SMOKE' if smoke else 'FULL'} Run {run_id}")
    print(f"{'='*60}")
    print(f"  Questions: {len(data)} ({'20 smoke' if smoke else 'all 500'})")
    print(f"  Answer:    {ANSWER_MODEL} via OpenRouter")
    print(f"  Judge:     {JUDGE_MODEL} ({NUM_JUDGE_RUNS}-vote majority)")
    print(f"  Search:    search_agentic(limit={TOP_K}, hyde=True, reranker=True)")
    print(f"{'='*60}\n")

    result = orchestrate.remote(data, run_id, smoke)
    print(f"\n  Result: {result}")
    print(f"  Download: modal volume get longmemeval-results / ./results --force")
