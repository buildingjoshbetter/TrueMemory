#!/usr/bin/env python3
"""
Generate personality-focused evaluation questions from LoCoMo conversations.

Uses an LLM to read each conversation and produce ~150 personality questions
per conversation across 6 categories, with gold answers grounded in the
actual message content.

Categories:
    1. preferences    — "What does X like to eat/do/watch?"
    2. communication  — "How does X communicate? Formal or casual?"
    3. traits         — "What kind of person is X?"
    4. routines       — "What's X's daily routine?"
    5. relationships  — "Who is X closest to? How do they interact?"
    6. discrimination — "Who is more health-conscious, X or Y?"

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python generate_questions.py [--conv-idx 0] [--output personality_eval.json]
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import re

LOCOMO_PATH = Path(__file__).parent.parent / "locomo" / "data" / "locomo10.json"
OUTPUT_PATH = Path(__file__).parent / "data" / "personality_eval.json"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-4.1-mini"

CATEGORIES = [
    "preferences",
    "communication",
    "traits",
    "routines",
    "relationships",
    "discrimination",
]

GENERATION_PROMPT = """You are generating personality evaluation questions for a memory system benchmark.

You have been given a conversation between {speaker_a} and {speaker_b} spanning {num_sessions} sessions and {num_messages} messages.

Your task: generate exactly {target_count} personality-focused questions with gold answers, based ONLY on what is explicitly stated or strongly implied in the conversation.

## Categories (generate roughly equal numbers of each)

1. **preferences** — What does this person like/dislike/prefer? (food, activities, entertainment, etc.)
2. **communication** — How does this person communicate? (formal/casual, emoji usage, message length, greeting style)
3. **traits** — What personality traits does this person exhibit? (anxious, ambitious, caring, analytical, etc.)
4. **routines** — What are this person's habits and routines? (morning routine, exercise, work patterns)
5. **relationships** — How does this person relate to others mentioned? (family, friends, colleagues)
6. **discrimination** — Comparative questions between {speaker_a} and {speaker_b}. ("Who is more X?", "How do they differ in Y?")

## Rules

- Every question MUST be answerable from the conversation content provided
- Gold answers must cite specific evidence from messages (quote or paraphrase)
- Include questions about BOTH speakers, roughly equally
- Discrimination questions must compare the two speakers
- Questions should range from easy (directly stated) to hard (requires synthesizing across multiple messages)
- Do NOT make up information not present in the conversation
- Keep gold answers concise but specific (1-3 sentences)

## Output format

Return a JSON array of objects:
```json
[
  {{
    "question": "What kind of food does {speaker_a} enjoy?",
    "category": "preferences",
    "target_entity": "{speaker_a}",
    "gold_answer": "{speaker_a} enjoys Italian food, specifically mentioning loving pasta and pizza in sessions 3 and 7.",
    "difficulty": "easy",
    "evidence_sessions": [3, 7]
  }},
  ...
]
```

Return ONLY the JSON array, no other text.

## Conversation

{conversation_text}"""


def parse_conversation(conv):
    """Parse a LoCoMo conversation into readable text."""
    c = conv["conversation"]
    sa, sb = c["speaker_a"], c["speaker_b"]

    sessions = sorted(
        [k for k in c if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    text_parts = []
    for sk in sessions:
        session_num = sk.split("_")[1]
        ds = c.get(f"{sk}_date_time", "")
        text_parts.append(f"\n=== Session {session_num} ({ds}) ===\n")
        for t in c[sk]:
            sp = t["speaker"]
            text_parts.append(f"  {sp}: {t['text']}")

    return sa, sb, len(sessions), sum(len(c[s]) for s in sessions), "\n".join(text_parts)


def generate_questions_for_conv(conv_data, conv_idx, target_per_conv=150):
    """Generate personality questions for one conversation using OpenRouter."""
    import openai

    client = openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        timeout=120.0,
    )

    sa, sb, num_sessions, num_messages, conv_text = parse_conversation(conv_data)

    # Truncate conversation if too long (keep first and last portions)
    if len(conv_text) > 80000:
        half = 38000
        conv_text = conv_text[:half] + "\n\n... [middle truncated for length] ...\n\n" + conv_text[-half:]

    prompt = GENERATION_PROMPT.format(
        speaker_a=sa,
        speaker_b=sb,
        num_sessions=num_sessions,
        num_messages=num_messages,
        target_count=target_per_conv,
        conversation_text=conv_text,
    )

    print(f"  Conv {conv_idx}: {sa} & {sb} ({num_sessions} sessions, {num_messages} msgs)")
    print(f"  Prompt length: {len(prompt)} chars")

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=16000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.choices[0].message.content.strip()

            # Extract JSON from response
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*\n?", "", content)
                content = re.sub(r"\n?```\s*$", "", content)

            questions = json.loads(content)

            # Tag with conversation metadata
            for q in questions:
                q["conversation_idx"] = conv_idx
                q["speaker_a"] = sa
                q["speaker_b"] = sb

            print(f"  Generated {len(questions)} questions")
            return questions

        except (json.JSONDecodeError, Exception) as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(5)
            continue

    print(f"  FAILED to generate questions for conv {conv_idx}")
    return []


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv-idx", type=int, default=None,
                        help="Process a single conversation (0-9)")
    parser.add_argument("--target", type=int, default=150,
                        help="Target questions per conversation")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY")
        sys.exit(1)

    with open(LOCOMO_PATH) as f:
        data = json.load(f)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    all_questions = []
    done_convs = set()
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        all_questions = existing.get("questions", [])
        done_convs = set(existing.get("done_convs", []))
        print(f"Resuming: {len(done_convs)} convs done, {len(all_questions)} questions")

    convs_to_process = [args.conv_idx] if args.conv_idx is not None else range(len(data))

    for ci in convs_to_process:
        if ci in done_convs:
            print(f"Conv {ci}: SKIPPED (already done)")
            continue

        print(f"\n--- Generating conv {ci} ---")
        questions = generate_questions_for_conv(data[ci], ci, args.target)
        all_questions.extend(questions)
        done_convs.add(ci)

        # Checkpoint after each conversation
        result = {
            "benchmark": "personality_eval",
            "source": "locomo10",
            "generator_model": MODEL,
            "target_per_conv": args.target,
            "categories": CATEGORIES,
            "done_convs": sorted(done_convs),
            "total_questions": len(all_questions),
            "questions": all_questions,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved checkpoint: {len(all_questions)} total questions")

    # Final summary
    from collections import Counter
    cat_counts = Counter(q.get("category", "?") for q in all_questions)
    entity_counts = Counter(q.get("target_entity", "?") for q in all_questions)
    print(f"\n=== DONE: {len(all_questions)} questions across {len(done_convs)} conversations ===")
    print(f"By category: {dict(cat_counts)}")
    print(f"By entity: {dict(entity_counts)}")


if __name__ == "__main__":
    main()
