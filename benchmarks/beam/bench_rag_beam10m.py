#!/usr/bin/env python3
"""
BEAM 10M Benchmark — ChromaDB RAG Baseline
==========================================
Dense vector retrieval using ChromaDB with sentence-transformers embeddings.
Same eval pipeline as TrueMemory BEAM benchmark. No reranker, no HyDE.

10 conversations in the 10M split (Mohammadta/BEAM-10M), 20 questions each = 200 total.
Smoke test: first 10 conversations = 200 questions.

Dependencies: chromadb, sentence-transformers, datasets
Eval: openai/gpt-4.1-mini (answers) + openai/gpt-4o-mini (judge) via OpenRouter

Usage:
    modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-...

    modal run --detach bench_rag_beam10m.py --smoke   # 10 convs
    modal run --detach bench_rag_beam10m.py           # All 35 convs

    modal volume get locomo-results / ./results --force
"""

import ast, json, modal, os, re, sys, time
from pathlib import Path

app = modal.App("beam-rag-10m")
vol = modal.Volume.from_name("locomo-results", create_if_missing=True)
VM = "/results"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

img = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai>=1.0", "chromadb", "sentence-transformers", "datasets"))

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
            })
    return questions


@app.function(image=img, secrets=[modal.Secret.from_name("openrouter-key")],
              timeout=14400, memory=8192)
def worker(conv_data: dict, conv_idx: int):
    """Process one BEAM conversation: ChromaDB ingest → retrieve → answer → judge."""
    import chromadb

    conv_id = conv_data.get("conversation_id", conv_idx)
    chat_sessions = conv_data["chat"]
    questions = _extract_questions(conv_data["probing_questions"])

    # Flatten 10M sessions: each session is a dict with plan-N keys
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

    print(f"  [rag] Conv {conv_idx} (id={conv_id}): "
          f"{len(chat_sessions)} sessions, {len(flat_msgs)} msgs, "
          f"{len(questions)} Qs", flush=True)

    # Step 1: Ingest into ChromaDB
    t0 = time.time()
    client_db = chromadb.Client()
    col = client_db.create_collection(name=f"beam_{conv_idx}", metadata={"hnsw:space": "cosine"})

    docs = []
    ids = []
    msg_count = 0
    for msg in flat_msgs:
        content = msg.get("content", "")
        role = msg.get("role", "user")
        time_anchor = msg.get("time_anchor", "")
        doc = f"[{role} | {time_anchor}] {content}"
        docs.append(doc)
        ids.append(f"m_{msg_count}")
        msg_count += 1
        if msg_count % 5000 == 0:
            print(f"    Conv {conv_idx}: prepared {msg_count}/{len(flat_msgs)} ({msg_count/len(flat_msgs)*100:.0f}%)", flush=True)

    # ChromaDB batch add (max 5000 per batch)
    for s in range(0, len(docs), 5000):
        col.add(documents=docs[s:s+5000], ids=ids[s:s+5000])
        print(f"    Conv {conv_idx}: indexed {min(s+5000, len(docs))}/{len(docs)}", flush=True)

    ingest_time = time.time() - t0
    print(f"    Ingested {msg_count} messages in {ingest_time:.1f}s", flush=True)

    # Step 2: Retrieve + Answer + Judge
    api_client = mkc()
    details = []

    for i, q in enumerate(questions):
        t_ret = time.time()
        try:
            results = col.query(query_texts=[q["question"]], n_results=100)
            retrieved_docs = results["documents"][0] if results["documents"] else []
            ctx = "\n\n".join(f"[Result {j+1}] {d}" for j, d in enumerate(retrieved_docs))
            num_retrieved = len(retrieved_docs)
        except Exception as e:
            ctx = "No results found."
            num_retrieved = 0
            print(f"    Retrieval error Q{i}: {e}", flush=True)
        retrieval_time = time.time() - t_ret

        t_ans = time.time()
        answer = gen_answer(api_client, ctx or "No results found.", q["question"])
        answer_time = time.time() - t_ans

        t_jdg = time.time()
        correct, votes = judge_one(api_client, q["question"], q["ideal"], answer)
        judge_time = time.time() - t_jdg

        details.append({
            "question": q["question"],
            "category": q["category"],
            "difficulty": q.get("difficulty", ""),
            "ideal_answer": q["ideal"],
            "generated_answer": answer,
            "correct": correct,
            "judge_votes": votes,
            "num_retrieved": num_retrieved,
            "conversation_id": conv_id,
            "retrieval_latency_s": round(retrieval_time, 2),
            "answer_latency_s": round(answer_time, 2),
            "judge_latency_s": round(judge_time, 2),
        })

        if (i+1) % 5 == 0:
            c = sum(1 for d in details if d["correct"])
            print(f"    {i+1}/{len(questions)}: {c}/{i+1} ({c/(i+1)*100:.0f}%)", flush=True)

    c = sum(1 for d in details if d["correct"])
    print(f"    Conv {conv_idx} done: {c}/{len(details)} correct "
          f"({c/len(details)*100:.1f}%), ingest={ingest_time:.1f}s", flush=True)
    return details


@app.function(image=img, secrets=[modal.Secret.from_name("openrouter-key")],
              timeout=43200, memory=4096, volumes={VM: vol})
def orchestrate(smoke: bool = False):
    """Load BEAM 10M, spawn workers, collect results."""
    import threading
    from datasets import load_dataset
    ds = load_dataset("Mohammadta/BEAM-10M", split="10M")

    system = "rag_beam10m"
    ckpt_path = f"{VM}/{system}_checkpoint.json"
    result_path = f"{VM}/{system}_run1.json"
    n_convs = 3 if smoke else len(ds)
    mode = "SMOKE TEST (3 convs)" if smoke else f"FULL RUN ({len(ds)} convs)"

    run_start = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"BEAM 10M — RAG (ChromaDB) — {mode}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Conversations: {n_convs}", flush=True)
    print(f"  Questions per conv: 20", flush=True)
    print(f"  Total questions: {n_convs * 20}", flush=True)
    print(f"  Answer:   {ANSWER_MODEL} via OpenRouter", flush=True)
    print(f"  Judge:    {JUDGE_MODEL} via OpenRouter", flush=True)

    # Heartbeat thread
    heartbeat_stop = threading.Event()
    done_convs = set()
    def heartbeat():
        while not heartbeat_stop.is_set():
            heartbeat_stop.wait(60)
            if not heartbeat_stop.is_set():
                elapsed = time.time() - run_start
                print(f"  [heartbeat] {elapsed:.0f}s elapsed, {len(done_convs)}/{n_convs} convs done", flush=True)
    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    hb_thread.start()

    # Resume from checkpoint
    all_details = []
    try:
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        all_details = ckpt.get("details", [])
        done_convs.update(ckpt.get("done_convs", []))
        print(f"  Resuming: {len(done_convs)} convs done", flush=True)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Spawn workers
    pending = {}
    for ci in range(n_convs):
        if ci in done_convs:
            print(f"  Conv {ci}: SKIPPED (checkpoint)", flush=True)
            continue
        conv_data = {
            "conversation_id": ds[ci].get("conversation_id", ci),
            "chat": ds[ci]["chat"],
            "probing_questions": ds[ci]["probing_questions"],
        }
        pending[ci] = worker.spawn(conv_data, ci)
    print(f"  Spawned {len(pending)} workers", flush=True)

    for ci, handle in pending.items():
        try:
            print(f"  Waiting for Conv {ci}...", flush=True)
            conv_details = handle.get()
            all_details.extend(conv_details)
            done_convs.add(ci)
            with open(ckpt_path, "w") as f:
                json.dump({"details": all_details, "done_convs": sorted(done_convs)}, f)
            vol.commit()
            c = sum(1 for d in conv_details if d["correct"])
            print(f"  Conv {ci} saved: {c}/{len(conv_details)} correct "
                  f"(checkpoint: {len(done_convs)}/{n_convs})", flush=True)
        except Exception as e:
            print(f"  Conv {ci} FAILED: {e}", flush=True)

    heartbeat_stop.set()

    if not all_details:
        print(f"  NO RESULTS", flush=True)
        return {"error": "no results"}

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
        "benchmark": "BEAM-10M",
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

    print(f"\n  BEAM 10M — RAG (ChromaDB): {result['j_score']}% ({tc}/{len(all_details)})", flush=True)
    for cat, v in sorted(cats.items()):
        print(f"    {cat:30s}: {v['correct']}/{v['total']} "
              f"({v['correct']/v['total']*100:.1f}%)", flush=True)
    print(f"\n  Time: {total_time:.0f}s", flush=True)
    print(f"  Saved: {result_path}", flush=True)

    try: os.remove(ckpt_path); vol.commit()
    except: pass

    return {"system": system, "j_score": result["j_score"],
            "total": result["total_questions"], "correct": tc}


@app.local_entrypoint()
def main(smoke: bool = False):
    result = orchestrate.remote(smoke=smoke)
    print(f"\n  Final result: {result}")
