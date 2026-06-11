#!/usr/bin/env python3
"""PersonaLoCoMo Benchmark Generator — generates personality evaluation questions from LoCoMo conversations."""

import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import openai

DATA_DIR = Path("/Users/j/Desktop/TrueMemory/benchmarks/personality_eval/data")
CHECKPOINT_PATH = DATA_DIR / "personality_eval.json"
LOCOMO_PATH = Path("/Users/j/Desktop/TrueMemory/benchmarks/locomo/data/locomo10.json")

CATEGORIES = [
    "food_and_drink",
    "hobbies_and_interests",
    "communication_style",
    "personality_and_character",
    "daily_life_and_routines",
    "relationships_and_people",
    "emotions_and_stress",
    "life_changes_over_time",
    "how_they_compare",
    "practical_recommendations",
]

CATEGORY_DESCRIPTIONS = {
    "food_and_drink": "What they eat, cook, drink, where they go. Dietary patterns, favorite spots, cooking habits.",
    "hobbies_and_interests": "What they do for fun, what they're passionate about, what they watch/read/play.",
    "communication_style": "How they text/talk. Formal? Casual? Emoji-heavy? Brief or long-winded? How they show excitement vs frustration.",
    "personality_and_character": 'What kind of person are they? The "tell me about X" category.',
    "daily_life_and_routines": "Day-to-day patterns, exercise habits, work schedule, how they spend a typical week.",
    "relationships_and_people": "People in their life — family, friends, coworkers, pets. How those relationships work.",
    "emotions_and_stress": "What makes them happy, anxious, excited, frustrated. How they cope. What weighs on them.",
    "life_changes_over_time": "How has this person evolved? New interests, life events, attitude shifts. Requires understanding personality OVER TIME.",
    "how_they_compare": "Side-by-side comparisons between the two speakers. Tests whether the system can DISTINGUISH between two people.",
    "practical_recommendations": "Questions requiring personality understanding to give useful advice. Most like real assistant interactions.",
}

client = openai.OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    timeout=180.0,
)

MODEL = "openai/gpt-4.1-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 16000
MAX_CONTEXT_CHARS = 324000  # ~108k tokens * 3 chars/token


def load_locomo():
    with open(LOCOMO_PATH) as f:
        return json.load(f)


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        if data.get("categories") == CATEGORIES:
            return data
    return {
        "benchmark": "personalocomo",
        "version": "1.0",
        "source": "locomo10",
        "generator_model": MODEL,
        "generation_date": "2026-04-27",
        "target_total": 2000,
        "categories": CATEGORIES,
        "character_sketches": {},
        "done_convs": [],
        "partial_calls": {},
        "total_questions": 0,
        "questions": [],
    }


def save_checkpoint(checkpoint):
    checkpoint["total_questions"] = len(checkpoint["questions"])
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"  Checkpoint saved: {checkpoint['total_questions']} questions, done_convs={checkpoint['done_convs']}")


def format_conversation(conv_data):
    c = conv_data["conversation"]
    sessions = sorted(
        [k for k in c if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda s: int(s.split("_")[1]),
    )
    lines = []
    for s in sessions:
        num = s.split("_")[1]
        date_key = f"{s}_date_time"
        date = c.get(date_key, "unknown date")
        lines.append(f"\n--- Session {num} ({date}) ---")
        for msg in c[s]:
            lines.append(f"{msg['speaker']}: {msg['text']}")
    return "\n".join(lines), sessions


def truncate_conversation(conv_text, sessions, conv_data):
    if len(conv_text) <= MAX_CONTEXT_CHARS:
        return conv_text, []

    c = conv_data["conversation"]
    total = len(sessions)
    keep_front = int(total * 0.4)
    keep_back = int(total * 0.4)
    drop_start = keep_front
    drop_end = total - keep_back

    dropped = [s.split("_")[1] for s in sessions[drop_start:drop_end]]

    lines = []
    for i, s in enumerate(sessions):
        if drop_start <= i < drop_end:
            if i == drop_start:
                lines.append(f"\n[Sessions {dropped[0]}-{dropped[-1]} omitted for context limits — {keep_front + keep_back} sessions retained]")
            continue
        num = s.split("_")[1]
        date_key = f"{s}_date_time"
        date = c.get(date_key, "unknown date")
        lines.append(f"\n--- Session {num} ({date}) ---")
        for msg in c[s]:
            lines.append(f"{msg['speaker']}: {msg['text']}")

    return "\n".join(lines), dropped


def get_locomo_questions(conv_data):
    questions = []
    for qa in conv_data.get("qa", []):
        q = qa.get("question", "")
        if q:
            questions.append(q)
    return questions


def generate_character_sketches(conv_data, conv_text):
    c = conv_data["conversation"]
    speaker_a = c["speaker_a"]
    speaker_b = c["speaker_b"]

    prompt = f"""Read this conversation between {speaker_a} and {speaker_b}. Write a 5-sentence character sketch for EACH speaker that captures:
- Their personality and temperament
- Their interests and what they care about
- Their communication style
- What's going on in their life
- How they've changed over the conversation

Output ONLY a JSON object: {{"speaker_a_name": "{speaker_a}", "speaker_a_sketch": "...", "speaker_b_name": "{speaker_b}", "speaker_b_sketch": "..."}}

CONVERSATION:
{conv_text[:MAX_CONTEXT_CHARS]}"""

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def build_generation_prompt(speaker_a, speaker_b, sketch_a, sketch_b, categories, count, locomo_qs, conv_text):
    cat_desc = "\n".join(f"- {cat} ({count_per_cat} questions): {CATEGORY_DESCRIPTIONS[cat]}" for cat, count_per_cat in categories)

    locomo_bullet = "\n".join(f"- {q}" for q in locomo_qs) if locomo_qs else "(none)"

    return f"""You are helping build a personality benchmark. Read this conversation between {speaker_a} and {speaker_b}, then generate personality questions that sound like a real person asking their AI assistant about people in their life.

CHARACTER SKETCHES (from Phase 1 analysis):
{speaker_a}: {sketch_a}
{speaker_b}: {sketch_b}

Generate exactly {count} questions across these categories:
{cat_desc}

Each question must be a JSON object with these fields:
- "id": string like "conv{{N}}_{{category}}_{{NNN}}" (e.g. "conv0_food_and_drink_001")
- "question": the natural-sounding question
- "category": one of the category names above
- "target_entity": name of the person the question is primarily about (or "both" for comparison questions)
- "conversation_idx": the conversation index number
- "speaker_a": "{speaker_a}"
- "speaker_b": "{speaker_b}"
- "gold_answer": a natural, conversational answer (2-4 sentences) citing specific evidence with session numbers
- "evidence_sessions": array of session numbers (integers) where evidence is found
- "difficulty": "easy", "medium", or "hard"

Rules:
- Questions must sound NATURAL — like someone talking to their phone, not a researcher
- Every answer must be grounded in actual message content (cite session numbers)
- Gold answers should read like a friend explaining someone to you — conversational, specific, 2-4 sentences
- Cover both speakers roughly equally
- Mix difficulties: ~30% easy, ~40% medium, ~30% hard
- For "how_they_compare" questions, always discuss BOTH people with evidence
- For "practical_recommendations" questions, actually give the recommendation
- Do NOT use synthetic phrases like "Based on the conversation", "Analyze", "Evaluate", "Assess", "Speaker A/B", "the entity", "the participant"
- Do NOT start any gold_answer with "I don't know" or "Not enough information"

IMPORTANT: Do NOT duplicate any of these existing LoCoMo questions for this conversation:
{locomo_bullet}

Output ONLY a JSON array of question objects. No other text.

CONVERSATION:
{conv_text}"""


def parse_llm_response(text, conv_idx):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        questions = json.loads(text)
        if isinstance(questions, list):
            return questions
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    print(f"  WARNING: Failed to parse LLM response for conv {conv_idx}")
    print(f"  Response start: {text[:200]}")
    return []


def word_overlap(q1, q2):
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def dedup_against_locomo(questions, locomo_qs):
    if not locomo_qs:
        return questions, 0
    kept = []
    removed = 0
    for q in questions:
        max_overlap = max(word_overlap(q["question"], lq) for lq in locomo_qs)
        if max_overlap > 0.70:
            removed += 1
        else:
            kept.append(q)
    return kept, removed


def spot_check_gold_answers(questions, conv_data, sample_size=10):
    if len(questions) < sample_size:
        sample_size = len(questions)
    sample = random.sample(questions, sample_size)
    c = conv_data["conversation"]

    sessions_map = {}
    for key in c:
        if key.startswith("session_") and not key.endswith("date_time"):
            num = int(key.split("_")[1])
            sessions_map[num] = c[key]

    failures = 0
    for q in sample:
        evidence = q.get("evidence_sessions", [])
        if not evidence:
            continue
        for sess_num in evidence:
            if sess_num not in sessions_map:
                print(f"  SPOT CHECK FAIL: {q['id']} cites session {sess_num} which doesn't exist")
                failures += 1
                break

    return failures


def generate_conversation(conv_idx, conv_data, checkpoint, locomo_data):
    c = conv_data["conversation"]
    speaker_a = c["speaker_a"]
    speaker_b = c["speaker_b"]

    sessions = sorted(
        [k for k in c if k.startswith("session_") and not k.endswith("date_time")],
        key=lambda s: int(s.split("_")[1]),
    )
    msg_count = sum(len(c[s]) for s in sessions)

    print(f"\n{'='*60}")
    print(f"CONVERSATION {conv_idx}: {speaker_a} & {speaker_b}")
    print(f"  {len(sessions)} sessions, {msg_count} messages")
    print(f"{'='*60}")

    conv_text, all_sessions = format_conversation(conv_data)
    conv_text, dropped = truncate_conversation(conv_text, all_sessions, conv_data)
    if dropped:
        print(f"  Truncated: dropped sessions {dropped[0]}-{dropped[-1]} for context limits")

    locomo_qs = get_locomo_questions(conv_data)
    print(f"  Existing LoCoMo questions to avoid: {len(locomo_qs)}")

    # Phase 1: Character sketches
    print(f"  Generating character sketches...")
    try:
        sketches = generate_character_sketches(conv_data, conv_text)
        sketch_a = sketches.get("speaker_a_sketch", "")
        sketch_b = sketches.get("speaker_b_sketch", "")
        checkpoint["character_sketches"][f"conv_{conv_idx}"] = {
            speaker_a: sketch_a,
            speaker_b: sketch_b,
        }
        print(f"  {speaker_a}: {sketch_a[:100]}...")
        print(f"  {speaker_b}: {sketch_b[:100]}...")
    except Exception as e:
        print(f"  ERROR generating sketches: {e}")
        sketch_a = f"{speaker_a} is a speaker in this conversation."
        sketch_b = f"{speaker_b} is a speaker in this conversation."

    # Phase 2: Generate questions
    use_6_calls = msg_count >= 500
    if use_6_calls:
        call_plan = [
            (["food_and_drink", "hobbies_and_interests"], {cat: 20 for cat in ["food_and_drink", "hobbies_and_interests"]}),
            (["communication_style", "personality_and_character"], {cat: 20 for cat in ["communication_style", "personality_and_character"]}),
            (["daily_life_and_routines", "relationships_and_people"], {cat: 20 for cat in ["daily_life_and_routines", "relationships_and_people"]}),
            (["emotions_and_stress", "life_changes_over_time"], {cat: 20 for cat in ["emotions_and_stress", "life_changes_over_time"]}),
            (["how_they_compare"], {"how_they_compare": 20}),
            (["practical_recommendations"], {"practical_recommendations": 20}),
        ]
    else:
        call_plan = [
            (["food_and_drink", "hobbies_and_interests", "communication_style"], {cat: 20 for cat in ["food_and_drink", "hobbies_and_interests", "communication_style"]}),
            (["personality_and_character", "daily_life_and_routines", "relationships_and_people"], {cat: 20 for cat in ["personality_and_character", "daily_life_and_routines", "relationships_and_people"]}),
            (["emotions_and_stress", "life_changes_over_time", "how_they_compare"], {cat: 20 for cat in ["emotions_and_stress", "life_changes_over_time", "how_they_compare"]}),
            (["practical_recommendations"], {"practical_recommendations": 20}),
        ]

    partial_key = f"conv_{conv_idx}"
    done_calls = checkpoint.get("partial_calls", {}).get(partial_key, [])
    all_questions = []

    for call_idx, (cats, cat_counts) in enumerate(call_plan):
        if call_idx in done_calls:
            print(f"  Call {call_idx+1}/{len(call_plan)}: skipping (already done)")
            continue

        count = sum(cat_counts.values())
        cat_with_counts = [(cat, cat_counts[cat]) for cat in cats]
        print(f"  Call {call_idx+1}/{len(call_plan)}: {cats} ({count} questions)...")

        prompt = build_generation_prompt(
            speaker_a, speaker_b, sketch_a, sketch_b,
            cat_with_counts, count, locomo_qs, conv_text
        )

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content
            questions = parse_llm_response(text, conv_idx)

            if len(questions) < count * 0.5:
                print(f"  Under-generation ({len(questions)}/{count}), retrying...")
                time.sleep(2)
                resp = client.chat.completions.create(
                    model=MODEL,
                    temperature=0.1,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.choices[0].message.content
                questions = parse_llm_response(text, conv_idx)

            for q in questions:
                q["conversation_idx"] = conv_idx
                q["speaker_a"] = speaker_a
                q["speaker_b"] = speaker_b

            all_questions.extend(questions)
            print(f"    Got {len(questions)} questions")

            if "partial_calls" not in checkpoint:
                checkpoint["partial_calls"] = {}
            if partial_key not in checkpoint["partial_calls"]:
                checkpoint["partial_calls"][partial_key] = []
            checkpoint["partial_calls"][partial_key].append(call_idx)

        except Exception as e:
            print(f"  ERROR on call {call_idx+1}: {e}")
            time.sleep(5)
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.choices[0].message.content
                questions = parse_llm_response(text, conv_idx)
                for q in questions:
                    q["conversation_idx"] = conv_idx
                    q["speaker_a"] = speaker_a
                    q["speaker_b"] = speaker_b
                all_questions.extend(questions)
                print(f"    Retry got {len(questions)} questions")
            except Exception as e2:
                print(f"  RETRY FAILED: {e2}")

        time.sleep(1)

    # Fix IDs
    cat_counters = Counter()
    for q in all_questions:
        cat = q.get("category", "unknown")
        cat_counters[cat] += 1
        q["id"] = f"conv{conv_idx}_{cat}_{cat_counters[cat]:03d}"

    # Dedup against LoCoMo
    all_questions, removed = dedup_against_locomo(all_questions, locomo_qs)
    print(f"  LoCoMo dedup: removed {removed} questions")

    # Spot check
    failures = spot_check_gold_answers(all_questions, conv_data)
    print(f"  Spot check: {failures} failures out of min(10, {len(all_questions)})")
    if failures > 2:
        print(f"  WARNING: High spot-check failure rate — consider regenerating")

    # Add to checkpoint
    checkpoint["questions"].extend(all_questions)
    if conv_idx not in checkpoint["done_convs"]:
        checkpoint["done_convs"].append(conv_idx)
    if partial_key in checkpoint.get("partial_calls", {}):
        del checkpoint["partial_calls"][partial_key]
    save_checkpoint(checkpoint)

    print(f"  DONE: {len(all_questions)} questions for conv {conv_idx}")
    return all_questions


def main():
    if len(sys.argv) > 1:
        target_conv = int(sys.argv[1])
    else:
        target_conv = None

    locomo_data = load_locomo()
    checkpoint = load_checkpoint()

    print(f"PersonaLoCoMo Generator")
    print(f"Existing: {len(checkpoint['questions'])} questions, done_convs={checkpoint['done_convs']}")

    if target_conv is not None:
        convs_to_do = [target_conv] if target_conv not in checkpoint["done_convs"] else []
    else:
        convs_to_do = [i for i in range(10) if i not in checkpoint["done_convs"]]

    if not convs_to_do:
        print("All conversations done!")
        return

    print(f"Conversations to generate: {convs_to_do}")

    for conv_idx in convs_to_do:
        generate_conversation(conv_idx, locomo_data[conv_idx], checkpoint, locomo_data)

    print(f"\nFinal: {len(checkpoint['questions'])} total questions across {len(checkpoint['done_convs'])} conversations")


if __name__ == "__main__":
    main()
