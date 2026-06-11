#!/usr/bin/env python3
"""Phase 10: Assemble final benchmark file and run validation checks."""

import json
from collections import Counter
from pathlib import Path
from datetime import datetime

base = Path("benchmarks/gate_eval/datasets")

# ─── Load all conversations ─────────────────────────────────────────────
conversations = []
all_messages = []
all_msg_ids = set()

conv_files = [
    ("conv1", ["gate_benchmark_conv1_part1.json", "gate_benchmark_conv1_part2.json"]),
    ("conv2", ["gate_benchmark_conv2.json"]),
    ("conv3", ["gate_benchmark_conv3.json"]),
    ("conv4", ["gate_benchmark_conv4.json"]),
    ("conv5", ["gate_benchmark_conv5.json"]),
]

conv_data = {}
for conv_id, files in conv_files:
    msgs = []
    for f in files:
        with open(base / f) as fh:
            data = json.load(fh)
            msgs.extend(data["messages"])
    conv_data[conv_id] = msgs
    all_messages.extend(msgs)
    for m in msgs:
        all_msg_ids.add(m["id"])
    conversations.append({
        "conversation_id": conv_id,
        "message_count": len(msgs),
        "speakers": list(set(m["speaker"] for m in msgs)),
    })

# ─── Load questions ──────────────────────────────────────────────────────
with open(base / "gate_benchmark_questions.json") as f:
    q_data = json.load(f)
questions = q_data["questions"]

# ─── Build final benchmark ───────────────────────────────────────────────
total_msgs = len(all_messages)
cat_counts = Counter(m["category"] for m in all_messages)
noise_total = sum(v for k, v in cat_counts.items() if k.startswith("N"))
signal_total = sum(v for k, v in cat_counts.items() if k.startswith("S"))
border_total = sum(v for k, v in cat_counts.items() if k.startswith("B"))

benchmark = {
    "benchmark": "GateLoCoMo",
    "version": "1.0",
    "total_messages": total_msgs,
    "total_questions": len(questions),
    "num_conversations": 5,
    "noise_ratio_actual": round(noise_total / total_msgs, 3),
    "signal_ratio_actual": round(signal_total / total_msgs, 3),
    "borderline_ratio_actual": round(border_total / total_msgs, 3),
    "category_distribution": dict(sorted(cat_counts.items())),
    "conversations": [],
    "questions": questions,
}

for conv_id, msgs in conv_data.items():
    cc = Counter(m["category"] for m in msgs)
    n = sum(v for k, v in cc.items() if k.startswith("N"))
    sp = Counter(m["speaker"] for m in msgs)
    benchmark["conversations"].append({
        "conversation_id": conv_id,
        "message_count": len(msgs),
        "noise_pct": round(n / len(msgs) * 100, 1),
        "speakers": {k: v for k, v in sp.items()},
        "messages": msgs,
    })

with open(base / "gate_benchmark.json", "w") as f:
    json.dump(benchmark, f, indent=2, ensure_ascii=False)

# ─── Validation ──────────────────────────────────────────────────────────
report = []
passes = 0
fails = 0

def check(name, condition, detail=""):
    global passes, fails
    if condition:
        passes += 1
        report.append(f"✅ {name}: PASS" + (f" — {detail}" if detail else ""))
    else:
        fails += 1
        report.append(f"❌ {name}: FAIL" + (f" — {detail}" if detail else ""))

# 1. Total messages ~2000 (±50)
check("Total messages ~2000", 1950 <= total_msgs <= 2050, f"{total_msgs}")

# 2. Per-conversation noise ratio (±10% of target since we already validated)
noise_targets = {"conv1": 50, "conv2": 45, "conv3": 55, "conv4": 40, "conv5": 50}
for conv_id, target in noise_targets.items():
    msgs = conv_data[conv_id]
    n = sum(1 for m in msgs if m["category"].startswith("N"))
    pct = n / len(msgs) * 100
    check(f"  {conv_id} noise ≈{target}%", abs(pct - target) <= 10, f"{pct:.1f}%")

# 3. All evidence_messages exist
missing_evidence = 0
for q in questions:
    for eid in q["evidence_messages"]:
        if eid not in all_msg_ids:
            missing_evidence += 1
check("All evidence_messages exist", missing_evidence == 0, f"{missing_evidence} missing")

# 4. No duplicate message IDs
check("No duplicate message IDs", len(all_msg_ids) == total_msgs,
      f"{total_msgs - len(all_msg_ids)} duplicates")

# 5. No duplicate questions (by question text)
q_texts = [q["question"] for q in questions]
q_uniq = len(set(q_texts))
check("No duplicate questions", q_uniq == len(questions), f"{len(questions) - q_uniq} duplicates")

# 6. All 5 conversations represented in questions
q_convs = set(q["conversation_id"] for q in questions)
check("All 5 conversations in questions", len(q_convs) == 5, f"found {len(q_convs)}")

# 7. Questions per conversation (~40 each)
q_conv_counts = Counter(q["conversation_id"] for q in questions)
for conv_id in ["conv1", "conv2", "conv3", "conv4", "conv5"]:
    c = q_conv_counts.get(conv_id, 0)
    check(f"  {conv_id} questions ~40", 30 <= c <= 50, f"{c}")

# 8. Evidence references signal messages
signal_evidence = 0
total_evidence_refs = 0
for q in questions:
    for eid in q["evidence_messages"]:
        total_evidence_refs += 1
        for m in all_messages:
            if m["id"] == eid and m["is_signal"]:
                signal_evidence += 1
                break
if total_evidence_refs > 0:
    pct = signal_evidence / total_evidence_refs * 100
    check("≥80% evidence refs are signal", pct >= 80, f"{pct:.1f}%")

# 9. Speaker balance
for conv_id, msgs in conv_data.items():
    sp = Counter(m["speaker"] for m in msgs)
    max_pct = max(v / len(msgs) * 100 for v in sp.values())
    check(f"  {conv_id} speaker balance <65%", max_pct <= 65, f"max={max_pct:.1f}%")

# 10. Timestamps chronological within sessions
chrono_ok = True
for conv_id, msgs in conv_data.items():
    sessions = {}
    for m in msgs:
        sessions.setdefault(m["session"], []).append(m)
    for sess, sess_msgs in sessions.items():
        ts = [m["timestamp"] for m in sess_msgs]
        if ts != sorted(ts):
            chrono_ok = False
check("Timestamps chronological within sessions", chrono_ok)

# 11. Session dates chronological within conversations
dates_ok = True
for conv_id, msgs in conv_data.items():
    sessions = {}
    for m in msgs:
        sessions.setdefault(m["session"], m["session_date"])
    dates = list(sessions.values())
    if dates != sorted(dates):
        dates_ok = False
check("Session dates chronological", dates_ok)

# 12. No empty evidence
empty_q = sum(1 for q in questions if not q["evidence_messages"])
check("No questions with empty evidence", empty_q == 0, f"{empty_q} empty")

# 13. Sample naturalness check
import random
random.seed(42)
sample = random.sample(all_messages, min(20, len(all_messages)))
synthetic_markers = ["I acknowledge", "Affirmative", "I concur", "As a side note", "For reference"]
synthetic_found = 0
for m in sample:
    for marker in synthetic_markers:
        if marker.lower() in m["content"].lower():
            synthetic_found += 1
check("Naturalness spot check (20 random)", synthetic_found == 0,
      f"{synthetic_found} synthetic-sounding")

# ─── Write report ────────────────────────────────────────────────────────
report_text = f"""# GateLoCoMo Benchmark Validation Report

Generated: {datetime.now().isoformat()[:19]}

## Summary

| Metric | Value |
|--------|-------|
| Total messages | {total_msgs} |
| Total questions | {len(questions)} |
| Conversations | 5 |
| Overall noise | {noise_total} ({noise_total/total_msgs*100:.1f}%) |
| Overall signal | {signal_total} ({signal_total/total_msgs*100:.1f}%) |
| Overall borderline | {border_total} ({border_total/total_msgs*100:.1f}%) |

## Category Distribution

| Category | Count | Percentage |
|----------|-------|-----------|
"""
for cat in sorted(cat_counts.keys()):
    report_text += f"| {cat} | {cat_counts[cat]} | {cat_counts[cat]/total_msgs*100:.1f}% |\n"

report_text += f"""
## Per-Conversation Stats

| Conv | Messages | Noise% | Target |
|------|----------|--------|--------|
"""
for conv_id, target in noise_targets.items():
    msgs = conv_data[conv_id]
    n = sum(1 for m in msgs if m["category"].startswith("N"))
    report_text += f"| {conv_id} | {len(msgs)} | {n/len(msgs)*100:.1f}% | {target}% |\n"

report_text += f"""
## Questions Distribution

| Conv | Questions |
|------|-----------|
"""
for conv_id in ["conv1", "conv2", "conv3", "conv4", "conv5"]:
    report_text += f"| {conv_id} | {q_conv_counts.get(conv_id, 0)} |\n"

report_text += f"""
## Validation Checks ({passes} passed, {fails} failed)

"""
for line in report:
    report_text += f"{line}\n"

report_text += f"""
## Sample Messages (naturalness check)

"""
for m in sample[:10]:
    report_text += f"- [{m['category']}] {m['speaker']}: \"{m['content'][:80]}\"\n"

with open(base / "GATE_BENCHMARK_REPORT.md", "w") as f:
    f.write(report_text)

print(f"\n{'='*60}")
print(f"GATE BENCHMARK ASSEMBLY COMPLETE")
print(f"{'='*60}")
print(f"Messages: {total_msgs}")
print(f"Questions: {len(questions)}")
print(f"Noise: {noise_total/total_msgs*100:.1f}% | Signal: {signal_total/total_msgs*100:.1f}% | Border: {border_total/total_msgs*100:.1f}%")
print(f"\nValidation: {passes} passed, {fails} failed")
for line in report:
    print(f"  {line}")
print(f"\nFiles written:")
print(f"  {base / 'gate_benchmark.json'}")
print(f"  {base / 'GATE_BENCHMARK_REPORT.md'}")
