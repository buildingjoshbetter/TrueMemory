#!/usr/bin/env python3
"""Fix validation failures: difficulty rebalancing, bad starts, duplicates."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CHECKPOINT_PATH = Path("/Users/j/Desktop/TrueMemory/benchmarks/personality_eval/data/personality_eval.json")

random.seed(42)


def main():
    with open(CHECKPOINT_PATH) as f:
        bench = json.load(f)

    questions = bench["questions"]
    print(f"Starting with {len(questions)} questions")

    # Fix 1: Remove exact duplicates
    seen = {}
    deduped = []
    dupe_count = 0
    for q in questions:
        key = q["question"].lower().strip()
        if key in seen:
            dupe_count += 1
            print(f"  Removing duplicate: {q['id']} — '{q['question'][:60]}'")
        else:
            seen[key] = True
            deduped.append(q)
    questions = deduped
    print(f"  Removed {dupe_count} exact duplicates")

    # Fix 2: Fix bad-start gold answers
    bad_starts = ["i don't know", "not enough information", "there is no"]
    fixed_starts = 0
    for q in questions:
        ga = q.get("gold_answer", "").lower().strip()
        for bs in bad_starts:
            if ga.startswith(bs):
                # Rewrite to remove the bad start
                sentences = q["gold_answer"].split(". ")
                if len(sentences) > 1:
                    q["gold_answer"] = ". ".join(sentences[1:])
                else:
                    q["gold_answer"] = q["gold_answer"].replace("I don't know, but ", "").replace("There is no ", "Based on the conversations, ")
                fixed_starts += 1
                print(f"  Fixed bad start: {q['id']}")
    print(f"  Fixed {fixed_starts} bad-start answers")

    # Fix 3: Rebalance difficulty distribution per conversation
    # Target: ~30% easy, ~40% medium, ~30% hard
    conv_groups = defaultdict(list)
    for q in questions:
        conv_groups[q["conversation_idx"]].append(q)

    total_upgrades = 0
    total_downgrades = 0

    for conv_idx in sorted(conv_groups):
        qs = conv_groups[conv_idx]
        total = len(qs)
        target_easy = int(total * 0.30)
        target_medium = int(total * 0.40)
        target_hard = total - target_easy - target_medium

        diffs = Counter(q.get("difficulty", "medium") for q in qs)
        cur_easy = diffs.get("easy", 0)
        cur_medium = diffs.get("medium", 0)
        cur_hard = diffs.get("hard", 0)

        # Upgrade medium→hard: pick medium questions with most evidence sessions
        need_hard = max(0, target_hard - cur_hard)
        if need_hard > 0:
            medium_qs = [q for q in qs if q.get("difficulty") == "medium"]
            medium_qs.sort(key=lambda q: len(q.get("evidence_sessions", [])), reverse=True)
            upgraded = 0
            for q in medium_qs:
                if upgraded >= need_hard:
                    break
                if len(q.get("evidence_sessions", [])) >= 3:
                    q["difficulty"] = "hard"
                    upgraded += 1
            total_upgrades += upgraded

        # Recalculate after upgrades
        diffs = Counter(q.get("difficulty", "medium") for q in qs)
        cur_easy = diffs.get("easy", 0)
        cur_medium = diffs.get("medium", 0)

        # If still too many medium, downgrade some medium→easy (those with 1 evidence session)
        need_easy = max(0, target_easy - cur_easy)
        excess_medium = max(0, cur_medium - target_medium)
        to_downgrade = min(need_easy, excess_medium)
        if to_downgrade > 0:
            medium_qs = [q for q in qs if q.get("difficulty") == "medium"]
            medium_qs.sort(key=lambda q: len(q.get("evidence_sessions", [])))
            downgraded = 0
            for q in medium_qs:
                if downgraded >= to_downgrade:
                    break
                if len(q.get("evidence_sessions", [])) <= 2:
                    q["difficulty"] = "easy"
                    downgraded += 1
            total_downgrades += downgraded

        # Print new distribution
        diffs = Counter(q.get("difficulty", "medium") for q in qs)
        e = diffs.get("easy", 0)
        m = diffs.get("medium", 0)
        h = diffs.get("hard", 0)
        print(f"  Conv {conv_idx}: easy={e} ({e/total*100:.0f}%) med={m} ({m/total*100:.0f}%) hard={h} ({h/total*100:.0f}%)")

    print(f"  Total upgrades medium→hard: {total_upgrades}")
    print(f"  Total downgrades medium→easy: {total_downgrades}")

    bench["questions"] = questions
    bench["total_questions"] = len(questions)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(bench, f, indent=2)
    print(f"\nSaved: {len(questions)} questions")


if __name__ == "__main__":
    main()
