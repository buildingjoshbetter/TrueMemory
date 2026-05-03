#!/usr/bin/env python3
"""
BEAM 10M Benchmark — TrueMemory Pro
===================================================================
BEAM (Beyond a Million Tokens) evaluates 10 memory abilities across
multi-session conversations at 10M token scale.

10 conversations in the 10M split (Mohammadta/BEAM-10M), 20 questions each = 200 total.
Smoke test: first 3 conversations = 60 questions.

Dependencies: truememory[gpu], sentence-transformers, datasets
Eval: openai/gpt-4.1-mini (answers) + openai/gpt-4o-mini (judge) via OpenRouter

Usage:
    modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-...

    modal run --detach bench_truememory_pro_beam10m.py --smoke   # 3 convs
    modal run --detach bench_truememory_pro_beam10m.py           # All 10 convs

    modal volume get locomo-results / ./results --force
"""

import ast, json, modal, os, re, sys, time, tempfile
from pathlib import Path

app = modal.App("beam-truememory-pro-10m")
vol = modal.Volume.from_name("locomo-results", create_if_missing=True)
VM = "/results"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

img = (modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("openai>=1.0",
                 "truememory[gpu] @ git+https://github.com/buildingjoshbetter/TrueMemory.git@main",
                 "sentence-transformers", "datasets"))

ANSWER_MODEL = "openai/gpt-4.1-mini"
ANSWER_MAX_TOKENS = 500
ANSWER_TEMPERATURE = 0
JUDGE_MODEL = "openai/gpt-4o-mini"
JUDGE_MAX_TOKENS = 50
JUDGE_TEMPERATURE = 0

def mkc():
    import openai
    return openai.OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                         base_url=OPENROUTER_BASE_URL, timeout=120.0)

def _retry(fn, retries=5):
    for i in range(retries + 1):
        try: return fn()
        except Exception as e:
            if i >= retries or not any(k in str(e).lower() for k in
                ["connection","timeout","429","502","503","504","rate_limit"]): raise
            time.sleep(2 * (2**i))

ANSWER_PROMPT = """You are answering questions about a long multi-session conversation.
You have been given retrieved conversation excerpts as context.

INSTRUCTIONS:
1. Read ALL context carefully — the answer may be spread across multiple excerpts
2. Look for specific names, dates, numbers, technical details, and preferences
3. Pay attention to temporal ordering — later messages may update earlier ones
4. For time questions, calculate carefully using the timestamps provided
5. If asked about preferences or instructions the user gave, look for explicit statements
6. If the context genuinely doesn't contain the answer, say "Not enough information"
7. Give a thorough, detailed answer — BEAM rewards completeness

Context:
{context}

Question: {question}

Think step by step, then give your final answer:"""

JUDGE_PROMPT = """You are evaluating whether a generated answer correctly addresses the question based on the ideal response.

Question: {question}
Ideal Response: {ideal}
Generated Answer: {generated}

Score the answer:
- If the generated answer captures the key facts from the ideal response, output "CORRECT"
- If it misses critical information or contradicts the ideal, output "WRONG"
- Be generous with phrasing differences — focus on factual accuracy

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

def judge_one(client, q, ideal, gen):
    if gen.startswith("ERROR:"):
        return False, [False, False, False]
    up = JUDGE_PROMPT.format(question=q, ideal=ideal, generated=gen)
    votes = []
    for _ in range(3):
        def _j():
            return client.chat.completions.create(
                model=JUDGE_MODEL, max_tokens=JUDGE_MAX_TOKENS, temperature=JUDGE_TEMPERATURE,
                messages=[{"role":"user","content":up}]
            ).choices[0].message.content
        try: votes.append(_verdict(_retry(_j)))
        except: votes.append(False)
    return sum(votes) > len(votes)/2, votes


def _extract_questions(probing_questions_str):
    """Parse probing questions from the dataset. Returns list of (question, ideal_answer, category)."""
    pq = ast.literal_eval(probing_questions_str)
    questions = []
    for category, qs in pq.items():
        for q in qs:
            question_text = q.get("question", "")
            ideal = (q.get("ideal_response") or q.get("ideal_answer")
                     or q.get("answer") or q.get("ideal_summary") or "")
            if not ideal and q.get("expected_compliance"):
                ideal = q["expected_compliance"]
            questions.append({
                "question": question_text,
                "ideal": ideal,
                "category": category,
                "difficulty": q.get("difficulty", ""),
                "rubric": q.get("rubric", ""),
            })
    return questions


@app.function(image=img, secrets=[modal.Secret.from_name("openrouter-key")],
              timeout=14400, memory=8192, gpu="T4")
def worker(conv_data: dict, conv_idx: int):
    """Process one BEAM conversation: ingest → retrieve → answer → judge."""
    from truememory.vector_search import set_embedding_model
    set_embedding_model("pro")
    from truememory.engine import TrueMemoryEngine
    from truememory.reranker import get_reranker, set_active_tier
    set_active_tier("pro")
    get_reranker(model_name="Alibaba-NLP/gte-reranker-modernbert-base")

    conv_id = conv_data.get("conversation_id", conv_idx)
    chat_sessions = conv_data["chat"]
    questions = _extract_questions(conv_data["probing_questions"])

    # Flatten 10M sessions: each session is a dict with plan-N keys
    # containing batches with turns containing message dicts
    flat_msgs = []
    for session in chat_sessions:
        if isinstance(session, dict):
            for key in sorted(session.keys(), key=lambda k: int(k.split('-')[1]) if '-' in k and k.split('-')[1].isdigit() else 0):
                plan = session[key]
                if not isinstance(plan, list):
                    continue
                for batch in plan:
                    if not isinstance(batch, dict):
                        continue
                    for turn in batch.get("turns", []):
                        for msg in turn:
                            if isinstance(msg, dict):
                                flat_msgs.append(msg)
        elif isinstance(session, list):
            for msg in session:
                if isinstance(msg, dict):
                    flat_msgs.append(msg)

    print(f"  [beam-pro] Conv {conv_idx} (id={conv_id}): "
          f"{len(chat_sessions)} sessions, {len(flat_msgs)} msgs, "
          f"{len(questions)} Qs")

    # Step 1: Ingest all messages
    t0 = time.time()
    _tmp = tempfile.NamedTemporaryFile(suffix=".db", prefix=f"beam_{conv_idx}_", delete=False)
    tmp_db = _tmp.name
    _tmp.close()

    engine = TrueMemoryEngine(tmp_db)
    engine._ensure_connection()

    msg_count = 0
    for msg in flat_msgs:
        content = msg.get("content", "")
        role = msg.get("role", "user")
        time_anchor = msg.get("time_anchor", "")
        sender = "user" if role == "user" else "assistant"
        ts = time_anchor or f"2024-01-01T00:00:{msg_count:02d}"

        engine.add(content=content, sender=sender,
                   recipient="assistant" if sender == "user" else "user",
                   timestamp=ts, category="chat")
        msg_count += 1

    ingest_time = time.time() - t0
    print(f"    Ingested {msg_count} messages in {ingest_time:.1f}s")

    # Step 2: Retrieve + Answer + Judge
    client = mkc()
    details = []

    for i, q in enumerate(questions):
        t_ret = time.time()
        try:
            sr = engine.search_agentic(q["question"], limit=100,
                                       use_hyde=True, use_reranker=True)
        except Exception as e:
            sr = []
            print(f"    Search failed for Q{i}: {e}")

        ctx_parts = []
        for j, r in enumerate(sr[:50]):
            content = r.get("content", "")
            ts = r.get("timestamp", "")
            sender = r.get("sender", "")
            ctx_parts.append(f"[Result {j+1}] [{ts}] {sender}: {content}")
        ctx = "\n\n".join(ctx_parts) if ctx_parts else "No relevant memories found."
        retrieval_time = time.time() - t_ret

        t_ans = time.time()
        answer = gen_answer(client, ctx, q["question"])
        answer_latency = time.time() - t_ans

        t_jdg = time.time()
        correct, votes = judge_one(client, q["question"], q["ideal"], answer)
        judge_latency = time.time() - t_jdg

        details.append({
            "question": q["question"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "ideal_answer": q["ideal"][:500],
            "generated_answer": answer,
            "correct": correct,
            "judge_votes": votes,
            "num_retrieved": len(sr),
            "conversation_id": conv_id,
            "retrieval_latency_s": round(retrieval_time, 2),
            "answer_latency_s": round(answer_latency, 2),
            "judge_latency_s": round(judge_latency, 2),
        })

        if (i+1) % 5 == 0:
            c = sum(1 for d in details if d["correct"])
            print(f"    {i+1}/{len(questions)}: {c}/{i+1} ({c/(i+1)*100:.0f}%)")

    # Cleanup
    try: os.unlink(tmp_db)
    except: pass

    c = sum(1 for d in details if d["correct"])
    print(f"    Conv {conv_idx} done: {c}/{len(details)} correct "
          f"({c/len(details)*100:.1f}%), ingest={ingest_time:.1f}s")
    return details


@app.function(image=img, secrets=[modal.Secret.from_name("openrouter-key")],
              timeout=28800, memory=2048, volumes={VM: vol})
def orchestrate(smoke: bool = False):
    """Load BEAM 10M, spawn workers, collect results."""
    from datasets import load_dataset
    ds = load_dataset("Mohammadta/BEAM-10M", split="10M")

    system = "truememory_pro_beam10m"
    ckpt_path = f"{VM}/{system}_checkpoint.json"
    result_path = f"{VM}/{system}_run1.json"
    n_convs = 3 if smoke else len(ds)
    mode = "SMOKE TEST (3 convs)" if smoke else f"FULL RUN ({len(ds)} convs)"

    run_start = time.time()
    print(f"\n{'='*60}")
    print(f"BEAM 10M — TRUEMEMORY PRO — {mode}")
    print(f"{'='*60}")
    print(f"  Conversations: {n_convs}")
    print(f"  Questions per conv: 20")
    print(f"  Total questions: {n_convs * 20}")
    print(f"  Answer:   {ANSWER_MODEL} via OpenRouter")
    print(f"  Judge:    {JUDGE_MODEL} via OpenRouter")

    # Resume from checkpoint
    all_details = []
    done_convs = set()
    try:
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        all_details = ckpt.get("details", [])
        done_convs = set(ckpt.get("done_convs", []))
        print(f"  Resuming: {len(done_convs)} convs done")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Spawn workers
    pending = {}
    for ci in range(n_convs):
        if ci in done_convs:
            print(f"  Conv {ci}: SKIPPED (checkpoint)")
            continue
        conv_data = {
            "conversation_id": ds[ci]["conversation_id"],
            "chat": ds[ci]["chat"],
            "probing_questions": ds[ci]["probing_questions"],
        }
        pending[ci] = worker.spawn(conv_data, ci)

    for ci, handle in pending.items():
        try:
            conv_details = handle.get()
            all_details.extend(conv_details)
            done_convs.add(ci)
            with open(ckpt_path, "w") as f:
                json.dump({"details": all_details, "done_convs": sorted(done_convs)}, f)
            vol.commit()
            c = sum(1 for d in conv_details if d["correct"])
            print(f"  Conv {ci} saved: {c}/{len(conv_details)} correct "
                  f"(checkpoint: {len(done_convs)}/{n_convs})")
        except Exception as e:
            print(f"  Conv {ci} FAILED: {e}")

    # Final results
    total_time = time.time() - run_start

    cats = {}
    for d in all_details:
        cat = d["category"]
        cats.setdefault(cat, {"correct": 0, "total": 0})
        cats[cat]["total"] += 1
        if d["correct"]:
            cats[cat]["correct"] += 1

    tc = sum(1 for d in all_details if d["correct"])
    result = {
        "system": system,
        "benchmark": "BEAM-1M",
        "version": "v1",
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "smoke_test": smoke,
        "j_score": round(tc/len(all_details)*100, 1) if all_details else 0,
        "total_correct": tc,
        "total_questions": len(all_details),
        "by_category": {cat: {**v, "accuracy": round(v["correct"]/v["total"]*100, 1)}
                        for cat, v in sorted(cats.items())},
        "total_time_s": round(total_time, 1),
        "details": all_details,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()

    print(f"\n  BEAM 10M — TRUEMEMORY PRO: {result['j_score']}% ({tc}/{len(all_details)})")
    for cat, v in sorted(cats.items()):
        print(f"    {cat:30s}: {v['correct']}/{v['total']} "
              f"({v['correct']/v['total']*100:.1f}%)")
    print(f"\n  Time: {total_time:.0f}s")
    print(f"  Saved: {result_path}")

    try: os.remove(ckpt_path); vol.commit()
    except: pass


@app.local_entrypoint()
def main(smoke: bool = False):
    orchestrate.remote(smoke=smoke)
