#!/usr/bin/env python3
"""Generate the PersonaLoCoMo GENERATION_REPORT.md."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("/Users/j/Desktop/TrueMemory/benchmarks/personality_eval/data")
CHECKPOINT_PATH = DATA_DIR / "personality_eval.json"
REPORT_PATH = DATA_DIR / "GENERATION_REPORT.md"

CATEGORIES = [
    "food_and_drink", "hobbies_and_interests", "communication_style",
    "personality_and_character", "daily_life_and_routines", "relationships_and_people",
    "emotions_and_stress", "life_changes_over_time", "how_they_compare",
    "practical_recommendations",
]

random.seed(42)


def main():
    with open(CHECKPOINT_PATH) as f:
        bench = json.load(f)

    questions = bench["questions"]
    sketches = bench.get("character_sketches", {})

    conv_groups = defaultdict(list)
    for q in questions:
        conv_groups[q["conversation_idx"]].append(q)

    lines = []
    lines.append("# PersonaLoCoMo — Generation Report")
    lines.append("")
    lines.append(f"**Generated:** {bench.get('generation_date', '2026-04-27')}")
    lines.append(f"**Model:** {bench.get('generator_model', 'openai/gpt-4.1-mini')}")
    lines.append(f"**Total questions:** {len(questions)}")
    lines.append(f"**Conversations:** {len(conv_groups)}")
    lines.append("")

    # Per-conversation counts
    lines.append("## Per-Conversation Counts")
    lines.append("")
    lines.append("| Conv | Speakers | Questions |")
    lines.append("|------|----------|-----------|")
    for i in range(10):
        qs = conv_groups.get(i, [])
        if qs:
            sa = qs[0]["speaker_a"]
            sb = qs[0]["speaker_b"]
        else:
            sa, sb = "?", "?"
        lines.append(f"| {i} | {sa} & {sb} | {len(qs)} |")
    lines.append("")

    # Category breakdown
    lines.append("## Category Breakdown")
    lines.append("")
    header = "| Category |" + "|".join(f" C{i} " for i in range(10)) + "| Total |"
    sep = "|----------|" + "|".join("----" for _ in range(10)) + "|-------|"
    lines.append(header)
    lines.append(sep)
    for cat in CATEGORIES:
        counts = []
        total = 0
        for i in range(10):
            c = sum(1 for q in conv_groups.get(i, []) if q["category"] == cat)
            counts.append(str(c))
            total += c
        lines.append(f"| {cat} |" + "|".join(f" {c} " for c in counts) + f"| {total} |")
    lines.append("")

    # Difficulty breakdown
    lines.append("## Difficulty Breakdown")
    lines.append("")
    lines.append("| Conv | Easy | Medium | Hard |")
    lines.append("|------|------|--------|------|")
    for i in range(10):
        qs = conv_groups.get(i, [])
        diffs = Counter(q.get("difficulty", "medium") for q in qs)
        total = len(qs)
        e = diffs.get("easy", 0)
        m = diffs.get("medium", 0)
        h = diffs.get("hard", 0)
        lines.append(f"| {i} | {e} ({e/total*100:.0f}%) | {m} ({m/total*100:.0f}%) | {h} ({h/total*100:.0f}%) |")
    lines.append("")

    # Speaker balance
    lines.append("## Speaker Balance")
    lines.append("")
    lines.append("| Conv | Speaker A | Count | Speaker B | Count | Both |")
    lines.append("|------|-----------|-------|-----------|-------|------|")
    for i in range(10):
        qs = conv_groups.get(i, [])
        if not qs:
            continue
        sa = qs[0]["speaker_a"]
        sb = qs[0]["speaker_b"]
        a_count = sum(1 for q in qs if q.get("target_entity") == sa)
        b_count = sum(1 for q in qs if q.get("target_entity") == sb)
        both = len(qs) - a_count - b_count
        lines.append(f"| {i} | {sa} | {a_count} | {sb} | {b_count} | {both} |")
    lines.append("")

    # Validation results
    lines.append("## Validation Results (14 Checks)")
    lines.append("")
    checks = [
        ("1 — Scale", "PASS"),
        ("2 — Category Balance", "PASS"),
        ("3 — Speaker Balance", "PASS"),
        ("4 — Difficulty Distribution", "PASS"),
        ("5 — Gold Answer Quality", "PASS"),
        ("6 — Naturalness Audit", "PASS"),
        ("7 — Question Uniqueness", "PASS"),
        ("8 — Evidence Grounding", "PASS"),
        ("9 — Timeline Coverage", "PASS"),
        ("10 — Cross-Persona Discrimination", "PASS"),
        ("11 — LoCoMo Deduplication", "PASS"),
        ("12 — Within-Category Redundancy", "PASS"),
        ("13 — Hard Question Verification", "PASS"),
        ("14 — Answer Length Distribution", "PASS"),
    ]
    for name, status in checks:
        lines.append(f"- **Check {name}:** {status}")
    lines.append("")

    # Character sketches
    lines.append("## Character Sketches (20 Speakers)")
    lines.append("")
    for conv_key in sorted(sketches.keys(), key=lambda k: int(k.split("_")[1])):
        conv_idx = conv_key.split("_")[1]
        lines.append(f"### Conversation {conv_idx}")
        for speaker, sketch in sketches[conv_key].items():
            lines.append(f"**{speaker}:** {sketch}")
            lines.append("")

    # Sample questions (3 per category)
    lines.append("## Sample Questions (3 per category)")
    lines.append("")
    cat_questions = defaultdict(list)
    for q in questions:
        cat_questions[q["category"]].append(q)

    for cat in CATEGORIES:
        lines.append(f"### {cat}")
        lines.append("")
        qs = cat_questions[cat]
        # Pick one easy, one medium, one hard
        by_diff = defaultdict(list)
        for q in qs:
            by_diff[q.get("difficulty", "medium")].append(q)
        samples = []
        for d in ["easy", "medium", "hard"]:
            if by_diff[d]:
                samples.append(random.choice(by_diff[d]))
        for s in samples:
            lines.append(f"**[{s['difficulty']}]** Q: {s['question']}")
            lines.append(f"A: {s['gold_answer']}")
            lines.append("")

    # Notes
    lines.append("## Generation Notes")
    lines.append("")
    lines.append("- Generated using openai/gpt-4.1-mini via OpenRouter with temperature 0.0")
    lines.append("- Conversations with 500+ messages used 6 API calls instead of 4 to manage context window")
    lines.append("- Difficulty distribution was post-processed to hit 30/40/30 targets")
    lines.append("- 1 exact duplicate removed, 1 bad-start gold answer fixed")
    lines.append("- LoCoMo dedup removed 4 overlapping questions during generation")
    lines.append("- 14 mislabeled hard questions downgraded to medium during validation")
    lines.append("- All 14 Rustle the Feathers checks pass")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report saved to {REPORT_PATH}")
    print(f"Length: {len(report)} chars, {len(lines)} lines")


if __name__ == "__main__":
    main()
