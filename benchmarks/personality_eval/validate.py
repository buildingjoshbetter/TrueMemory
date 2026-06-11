#!/usr/bin/env python3
"""PersonaLoCoMo Validation — Rustle the Feathers (14 checks)."""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

CHECKPOINT_PATH = Path("/Users/j/Desktop/TrueMemory/benchmarks/personality_eval/data/personality_eval.json")
LOCOMO_PATH = Path("/Users/j/Desktop/TrueMemory/benchmarks/locomo/data/locomo10.json")

CATEGORIES = [
    "food_and_drink", "hobbies_and_interests", "communication_style",
    "personality_and_character", "daily_life_and_routines", "relationships_and_people",
    "emotions_and_stress", "life_changes_over_time", "how_they_compare",
    "practical_recommendations",
]

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def load_data():
    with open(CHECKPOINT_PATH) as f:
        bench = json.load(f)
    with open(LOCOMO_PATH) as f:
        locomo = json.load(f)
    return bench, locomo


def word_overlap(q1, q2):
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def get_session_count(conv_data):
    c = conv_data["conversation"]
    return len([k for k in c if k.startswith("session_") and not k.endswith("date_time")])


def check_1_scale(questions):
    print("\n" + "="*60)
    print("CHECK 1 — Scale")
    total = len(questions)
    conv_counts = Counter(q["conversation_idx"] for q in questions)

    passed = True
    if total < 1800:
        print(f"  FAIL: total {total} < 1800")
        passed = False
    else:
        print(f"  OK: total {total} >= 1800")

    for i in range(10):
        c = conv_counts.get(i, 0)
        status = "OK" if c >= 170 else "FAIL"
        if c < 170:
            passed = False
        print(f"  Conv {i}: {c} questions [{status}]")

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def check_2_category_balance(questions):
    print("\n" + "="*60)
    print("CHECK 2 — Category Balance")
    matrix = defaultdict(lambda: defaultdict(int))
    for q in questions:
        matrix[q["conversation_idx"]][q["category"]] += 1

    passed = True
    header = f"{'Conv':>6}" + "".join(f"{c[:8]:>10}" for c in CATEGORIES)
    print(f"  {header}")
    for i in range(10):
        row = f"{i:>6}"
        for cat in CATEGORIES:
            count = matrix[i].get(cat, 0)
            if count < 12 or count > 28:
                passed = False
            row += f"{count:>10}"
        print(f"  {row}")

    missing = []
    for i in range(10):
        for cat in CATEGORIES:
            if matrix[i].get(cat, 0) == 0:
                missing.append((i, cat))
                passed = False
    if missing:
        print(f"  Missing categories: {missing}")

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def check_3_speaker_balance(questions):
    print("\n" + "="*60)
    print("CHECK 3 — Speaker Balance")
    passed = True

    conv_groups = defaultdict(list)
    for q in questions:
        conv_groups[q["conversation_idx"]].append(q)

    for conv_idx in sorted(conv_groups):
        qs = conv_groups[conv_idx]
        speaker_a = qs[0]["speaker_a"]
        speaker_b = qs[0]["speaker_b"]

        a_count = sum(1 for q in qs if q.get("target_entity") == speaker_a)
        b_count = sum(1 for q in qs if q.get("target_entity") == speaker_b)
        both_count = sum(1 for q in qs if q.get("target_entity") in ("both", speaker_a + " and " + speaker_b, speaker_b + " and " + speaker_a))
        other = len(qs) - a_count - b_count - both_count

        total_individual = a_count + b_count
        if total_individual > 0:
            a_pct = a_count / total_individual * 100
        else:
            a_pct = 50
        status = "OK" if 40 <= a_pct <= 60 else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"  Conv {conv_idx}: {speaker_a}={a_count} ({a_pct:.0f}%), {speaker_b}={b_count} ({100-a_pct:.0f}%), both={both_count}, other={other} [{status}]")

    # Check comparison questions reference both speakers
    compare_qs = [q for q in questions if q["category"] == "how_they_compare"]
    both_in_q = 0
    for q in compare_qs:
        text = q["question"].lower()
        sa = q["speaker_a"].lower()
        sb = q["speaker_b"].lower()
        if sa in text and sb in text:
            both_in_q += 1
    if compare_qs:
        rate = both_in_q / len(compare_qs) * 100
        print(f"  Comparison questions with both speakers in text: {both_in_q}/{len(compare_qs)} ({rate:.0f}%)")
        if rate < 80:
            passed = False

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def check_4_difficulty(questions):
    print("\n" + "="*60)
    print("CHECK 4 — Difficulty Distribution")
    passed = True
    conv_groups = defaultdict(list)
    for q in questions:
        conv_groups[q["conversation_idx"]].append(q)

    for conv_idx in sorted(conv_groups):
        qs = conv_groups[conv_idx]
        total = len(qs)
        diffs = Counter(q.get("difficulty", "unknown") for q in qs)
        easy_pct = diffs.get("easy", 0) / total * 100
        med_pct = diffs.get("medium", 0) / total * 100
        hard_pct = diffs.get("hard", 0) / total * 100

        e_ok = 25 <= easy_pct <= 35
        m_ok = 35 <= med_pct <= 45
        h_ok = 25 <= hard_pct <= 35
        status = "OK" if (e_ok and m_ok and h_ok) else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"  Conv {conv_idx}: easy={easy_pct:.0f}% med={med_pct:.0f}% hard={hard_pct:.0f}% [{status}]")

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def check_5_gold_quality(questions):
    print("\n" + "="*60)
    print("CHECK 5 — Gold Answer Quality")
    passed = True
    short = 0
    no_detail = 0
    bad_start = 0
    bad_starts_list = ["i don't know", "not enough information", "there is no"]

    for q in questions:
        ga = q.get("gold_answer", "")
        if len(ga) < 30:
            short += 1
        ga_lower = ga.lower().strip()
        if any(ga_lower.startswith(bs) for bs in bad_starts_list):
            bad_start += 1

    print(f"  Short answers (<30 chars): {short}")
    print(f"  Bad starts: {bad_start}")
    if short > 0 or bad_start > 0:
        passed = False

    # Sample 5 per conversation
    conv_groups = defaultdict(list)
    for q in questions:
        conv_groups[q["conversation_idx"]].append(q)

    print(f"\n  Sample gold answers:")
    for conv_idx in sorted(conv_groups):
        qs = conv_groups[conv_idx]
        sample = random.sample(qs, min(5, len(qs)))
        for s in sample[:2]:
            print(f"    Conv {conv_idx} [{s['category']}]: Q: {s['question'][:80]}")
            print(f"      A: {s['gold_answer'][:120]}...")

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def check_6_naturalness(questions):
    print("\n" + "="*60)
    print("CHECK 6 — Naturalness Audit")
    passed = True
    synthetic_patterns = [
        r"based on the conversation",
        r"according to the messages",
        r"\banalyze\b",
        r"\bevaluate\b",
        r"\bassess\b",
        r"\bspeaker a\b",
        r"\bspeaker b\b",
        r"\bthe entity\b",
        r"\bthe participant\b",
        r"\btemporal\b",
        r"\bvalence\b",
        r"\bdistribution\b",
        r"\bspectrum\b",
        r"sessions?\s+\d+\s+(through|to)\s+\d+",
    ]

    conv_groups = defaultdict(list)
    for q in questions:
        conv_groups[q["conversation_idx"]].append(q)

    flagged = []
    for conv_idx in sorted(conv_groups):
        qs = conv_groups[conv_idx]
        sample = random.sample(qs, min(50, len(qs)))
        for q in sample:
            text = q["question"].lower()
            for pat in synthetic_patterns:
                if re.search(pat, text):
                    flagged.append((conv_idx, q["id"], q["question"][:80], pat))
                    break

    for conv_idx, qid, qtxt, pat in flagged:
        print(f"  FLAGGED conv {conv_idx} {qid}: '{qtxt}' (matched: {pat})")

    if flagged:
        passed = False
        print(f"  Total flagged: {len(flagged)}")
    else:
        print(f"  No synthetic patterns found in sample (0 flagged)")

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed, flagged


def check_7_uniqueness(questions):
    print("\n" + "="*60)
    print("CHECK 7 — Question Uniqueness")
    passed = True

    seen = set()
    exact_dupes = 0
    for q in questions:
        text = q["question"].lower().strip()
        if text in seen:
            exact_dupes += 1
        seen.add(text)

    print(f"  Exact duplicates: {exact_dupes}")
    if exact_dupes > 0:
        passed = False

    # Near-duplicates (sample-based for speed)
    near_dupes = 0
    all_texts = [q["question"] for q in questions]
    if len(all_texts) > 500:
        sample_idx = random.sample(range(len(all_texts)), 500)
    else:
        sample_idx = list(range(len(all_texts)))

    for i in range(len(sample_idx)):
        for j in range(i+1, len(sample_idx)):
            ov = word_overlap(all_texts[sample_idx[i]], all_texts[sample_idx[j]])
            if ov > 0.85:
                near_dupes += 1
                if near_dupes <= 5:
                    print(f"  Near-dupe ({ov:.2f}): '{all_texts[sample_idx[i]][:60]}' vs '{all_texts[sample_idx[j]][:60]}'")

    print(f"  Near-duplicates found (>85% overlap, sample of {len(sample_idx)}): {near_dupes}")
    if near_dupes > 0:
        passed = False

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def check_8_evidence(questions, locomo):
    print("\n" + "="*60)
    print("CHECK 8 — Evidence Grounding")
    passed = True

    conv_groups = defaultdict(list)
    for q in questions:
        conv_groups[q["conversation_idx"]].append(q)

    for conv_idx in sorted(conv_groups):
        qs = conv_groups[conv_idx]
        c = locomo[conv_idx]["conversation"]
        max_session = len([k for k in c if k.startswith("session_") and not k.endswith("date_time")])

        has_evidence = sum(1 for q in qs if q.get("evidence_sessions") and len(q["evidence_sessions"]) > 0)
        pct = has_evidence / len(qs) * 100

        invalid_refs = 0
        for q in qs:
            for s in q.get("evidence_sessions", []):
                if s < 1 or s > max_session:
                    invalid_refs += 1

        status = "OK" if pct >= 85 else "FAIL"
        if status == "FAIL":
            passed = False
        extra = f" (invalid refs: {invalid_refs})" if invalid_refs else ""
        print(f"  Conv {conv_idx}: {has_evidence}/{len(qs)} ({pct:.0f}%) grounded{extra} [{status}]")

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def check_9_timeline_coverage(questions, locomo):
    print("\n" + "="*60)
    print("CHECK 9 — Timeline Coverage")
    passed = True

    conv_groups = defaultdict(list)
    for q in questions:
        conv_groups[q["conversation_idx"]].append(q)

    for conv_idx in sorted(conv_groups):
        qs = conv_groups[conv_idx]
        max_session = get_session_count(locomo[conv_idx])
        all_sessions = set(range(1, max_session + 1))

        referenced = set()
        for q in qs:
            for s in q.get("evidence_sessions", []):
                referenced.add(s)

        coverage = len(referenced & all_sessions) / len(all_sessions) * 100
        status = "OK" if coverage >= 50 else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"  Conv {conv_idx}: {len(referenced & all_sessions)}/{len(all_sessions)} sessions ({coverage:.0f}%) [{status}]")

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def check_10_comparison_quality(questions):
    print("\n" + "="*60)
    print("CHECK 10 — Cross-Persona Discrimination Quality")
    passed = True

    compare_qs = [q for q in questions if q["category"] == "how_they_compare"]
    print(f"  Total comparison questions: {len(compare_qs)}")

    both_in_q = 0
    both_in_a = 0
    for q in compare_qs:
        sa = q["speaker_a"].lower()
        sb = q["speaker_b"].lower()
        qtext = q["question"].lower()
        atext = q.get("gold_answer", "").lower()

        if sa in qtext and sb in qtext:
            both_in_q += 1
        if sa in atext and sb in atext:
            both_in_a += 1

    if compare_qs:
        q_rate = both_in_q / len(compare_qs) * 100
        a_rate = both_in_a / len(compare_qs) * 100
        print(f"  Both speakers in question: {both_in_q}/{len(compare_qs)} ({q_rate:.0f}%)")
        print(f"  Both speakers in answer: {both_in_a}/{len(compare_qs)} ({a_rate:.0f}%)")
        if q_rate < 80 or a_rate < 80:
            passed = False

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def check_11_locomo_dedup(questions, locomo):
    print("\n" + "="*60)
    print("CHECK 11 — LoCoMo Deduplication")
    passed = True
    total_removed = 0

    conv_groups = defaultdict(list)
    for q in questions:
        conv_groups[q["conversation_idx"]].append(q)

    to_remove = set()
    for conv_idx in sorted(conv_groups):
        locomo_qs = [qa.get("question", "") for qa in locomo[conv_idx].get("qa", [])]
        qs = conv_groups[conv_idx]
        removed = 0
        for q in qs:
            for lq in locomo_qs:
                if word_overlap(q["question"], lq) > 0.70:
                    to_remove.add(q["id"])
                    removed += 1
                    break
        if removed:
            print(f"  Conv {conv_idx}: {removed} overlapping questions found")
        total_removed += removed

    print(f"  Total to remove: {total_removed}")
    if total_removed > 0:
        passed = False

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed, to_remove


def check_12_within_category_redundancy(questions):
    print("\n" + "="*60)
    print("CHECK 12 — Within-Category Redundancy")
    passed = True
    total_redundant = 0
    to_remove = set()

    conv_cat_groups = defaultdict(list)
    for q in questions:
        conv_cat_groups[(q["conversation_idx"], q["category"])].append(q)

    for (conv_idx, cat), qs in sorted(conv_cat_groups.items()):
        redundant = 0
        for i in range(len(qs)):
            if qs[i]["id"] in to_remove:
                continue
            for j in range(i+1, len(qs)):
                if qs[j]["id"] in to_remove:
                    continue
                if word_overlap(qs[i]["question"], qs[j]["question"]) > 0.75:
                    # Remove the lower-difficulty one
                    di = DIFFICULTY_ORDER.get(qs[i].get("difficulty", "easy"), 0)
                    dj = DIFFICULTY_ORDER.get(qs[j].get("difficulty", "easy"), 0)
                    if di <= dj:
                        to_remove.add(qs[i]["id"])
                    else:
                        to_remove.add(qs[j]["id"])
                    redundant += 1

        if redundant:
            print(f"  Conv {conv_idx} / {cat}: {redundant} redundant pairs")
        total_redundant += redundant

    print(f"  Total redundant: {total_redundant}")
    if total_redundant > 0:
        passed = False

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed, to_remove


def check_13_hard_verification(questions):
    print("\n" + "="*60)
    print("CHECK 13 — Hard Question Verification")
    passed = True

    hard_qs = [q for q in questions if q.get("difficulty") == "hard"]
    sample = random.sample(hard_qs, min(20, len(hard_qs)))

    mislabeled = 0
    downgrade_ids = []
    for q in sample:
        evidence = q.get("evidence_sessions", [])
        if len(evidence) < 3:
            mislabeled += 1
            downgrade_ids.append(q["id"])
            if mislabeled <= 5:
                print(f"  Mislabeled: {q['id']} — only {len(evidence)} evidence sessions")

    print(f"  Mislabeled hard questions (of {len(sample)} sampled): {mislabeled}")
    if mislabeled > 5:
        passed = False

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed, downgrade_ids


def check_14_answer_length(questions):
    print("\n" + "="*60)
    print("CHECK 14 — Answer Length Distribution")
    passed = True

    lengths = [len(q.get("gold_answer", "")) for q in questions]
    too_short = sum(1 for l in lengths if l < 30)
    too_long = sum(1 for l in lengths if l > 500)
    avg = sum(lengths) / len(lengths)
    med = sorted(lengths)[len(lengths)//2]

    print(f"  Avg length: {avg:.0f} chars")
    print(f"  Median: {med} chars")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
    print(f"  Too short (<30): {too_short}")
    print(f"  Too long (>500): {too_long}")

    if too_short > 0 or too_long > len(questions) * 0.1:
        passed = False

    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    random.seed(42)
    bench, locomo = load_data()
    questions = bench["questions"]
    print(f"PersonaLoCoMo Validation — {len(questions)} questions")

    results = {}
    results["check_1"] = check_1_scale(questions)
    results["check_2"] = check_2_category_balance(questions)
    results["check_3"] = check_3_speaker_balance(questions)
    results["check_4"] = check_4_difficulty(questions)
    results["check_5"] = check_5_gold_quality(questions)
    p6, flagged_6 = check_6_naturalness(questions)
    results["check_6"] = p6
    results["check_7"] = check_7_uniqueness(questions)
    results["check_8"] = check_8_evidence(questions, locomo)
    results["check_9"] = check_9_timeline_coverage(questions, locomo)
    results["check_10"] = check_10_comparison_quality(questions)
    p11, to_remove_11 = check_11_locomo_dedup(questions, locomo)
    results["check_11"] = p11
    p12, to_remove_12 = check_12_within_category_redundancy(questions)
    results["check_12"] = p12
    p13, downgrade_ids = check_13_hard_verification(questions)
    results["check_13"] = p13
    results["check_14"] = check_14_answer_length(questions)

    # Apply fixes
    all_remove = to_remove_11 | to_remove_12
    if all_remove:
        print(f"\nRemoving {len(all_remove)} questions (dedup + redundancy)...")
        questions = [q for q in questions if q["id"] not in all_remove]
        bench["questions"] = questions
        bench["total_questions"] = len(questions)

    # Downgrade mislabeled hard questions
    if downgrade_ids:
        print(f"Downgrading {len(downgrade_ids)} mislabeled hard questions to medium...")
        id_set = set(downgrade_ids)
        for q in bench["questions"]:
            if q["id"] in id_set:
                q["difficulty"] = "medium"

    if all_remove or downgrade_ids:
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(bench, f, indent=2)
        print(f"Updated checkpoint: {len(bench['questions'])} questions")

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    all_pass = True
    for check, passed in sorted(results.items()):
        status = "PASS" if passed else "FAIL"
        print(f"  {check}: {status}")
        if not passed:
            all_pass = False
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print(f"  Final question count: {len(bench['questions'])}")


if __name__ == "__main__":
    main()
