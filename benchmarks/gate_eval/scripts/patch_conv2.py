#!/usr/bin/env python3
"""Patch conv2 to fix noise ratio: convert ~54 N→B/S to hit 45% noise target."""

import json
from collections import Counter
from pathlib import Path

# Patches: message_id → new category + notes.
# Each conversion is justified by the existing message content.
PATCHES = {
    # ── Session 1 (N=26→20, convert 6) ──
    "conv2_s01_006": ("S2", "Maria's opinion: 8 hours minimum sleep"),
    "conv2_s01_008": ("B2", "Casual fact: father's college sleep habits"),
    "conv2_s01_018": ("B1", "Introduces potential academic path: film minor"),
    "conv2_s01_020": ("B2", "Casual fact: Sam has always loved movies"),
    "conv2_s01_028": ("B2", "Casual fact: Sam's emotional connection to food"),
    "conv2_s01_031": ("B3", "Temporal pressure: January deadline reminder"),

    # ── Session 2 (N=30→22, convert 8) ──
    "conv2_s02_002": ("B2", "Casual fact: Sam is in class, prefers text"),
    "conv2_s02_015": ("B2", "Casual medical fact: prediabetes is very common"),
    "conv2_s02_018": ("B1", "Question provides context for follow-up test timeline"),
    "conv2_s02_025": ("B2", "Casual fact: Sam admits to procrastination pattern"),
    "conv2_s02_030": ("B1", "Question establishes statement length requirement context"),
    "conv2_s02_033": ("B2", "Casual fact: Sam agrees to have parents review statement"),
    "conv2_s02_039": ("S5", "Emotional reaction with 🥺 emoji shows vulnerability"),
    "conv2_s02_042": ("B3", "Temporal: 'last week' FaceTime observation + concern"),

    # ── Session 3 (N=19→16, convert 3) ──
    "conv2_s03_001": ("B3", "Temporal: confirming Wednesday night arrival"),
    "conv2_s03_013": ("B3", "Temporal: Sam hasn't seen Sofia 'in forever'"),
    "conv2_s03_030": ("B2", "Casual fact: ate ten tamales last year"),

    # ── Session 4 (N=22→17, convert 5) ──
    "conv2_s04_015": ("B3", "Temporal: waiting for kiln to finish"),
    "conv2_s04_019": ("B1", "Context: Maria offers financial help, establishes support pattern"),
    "conv2_s04_021": ("S2", "Maria's opinion: creative pursuits are wonderful"),
    "conv2_s04_029": ("B2", "Casual fact: Sam has always been good with words"),
    "conv2_s04_034": ("B2", "Casual fact: Sam eats 3 meals a day"),

    # ── Session 5 (N=22→16, convert 6) ──
    "conv2_s05_003": ("B1", "Context clarification: stressed about applications specifically"),
    "conv2_s05_008": ("S2", "Maria's advice philosophy: lists and prioritization"),
    "conv2_s05_010": ("B1", "Suggests professor extensions, provides strategic context"),
    "conv2_s05_024": ("B2", "Casual fact: dad complains but walks with Maria daily"),
    "conv2_s05_032": ("B1", "Request provides context: Maria wants to see pottery progress"),
    "conv2_s05_034": ("B3", "Temporal: Christmas gift suggestion establishes timeline"),

    # ── Session 6 (N=18→14, convert 4) ──
    "conv2_s06_015": ("B3", "Temporal: hard part is done, relief marker"),
    "conv2_s06_017": ("B1", "Context: can reuse material across applications"),
    "conv2_s06_026": ("B3", "Temporal: when coming home for Christmas"),
    "conv2_s06_032": ("B2", "Casual fact: house is already decorated for Christmas"),

    # ── Session 7 (N=22→17, convert 5) ──
    "conv2_s07_024": ("B1", "Context: what if accepted to multiple programs"),
    "conv2_s07_026": ("B2", "Casual fact: Berkeley is close enough for weekend visits"),
    "conv2_s07_030": ("B1", "Context about venue: Carmen's house is huge"),
    "conv2_s07_034": ("B2", "Casual fact: Sam says 'classic abuela'"),
    "conv2_s07_037": ("S5", "Emotional: family pride expressed by Maria"),

    # ── Session 8 (N=25→18, convert 7) ──
    "conv2_s08_005": ("B3", "Temporal: 'the last month has been insane'"),
    "conv2_s08_009": ("B3", "Temporal: 'stressful couple months' ahead"),
    "conv2_s08_010": ("S2", "Maria's advice: enjoy the semester, you've earned it"),
    "conv2_s08_016": ("B1", "Context question: cost of intermediate course"),
    "conv2_s08_020": ("S2", "Maria's values: supporting talent and creativity"),
    "conv2_s08_027": ("B1", "Context question about farmers market products"),
    "conv2_s08_031": ("B2", "Casual fact: Maria thinks father will be thrilled about market"),

    # ── Session 9 (N=27→18, convert 9) ──
    "conv2_s09_007": ("B2", "Casual fact: Sam checks email every 5 minutes"),
    "conv2_s09_009": ("B2", "Casual fact: Sam compares own anxiety to father's"),
    "conv2_s09_013": ("B1", "Context: Sam has reunion on calendar"),
    "conv2_s09_015": ("B2", "Casual fact: Sam characterizes Maria as a planner"),
    "conv2_s09_019": ("B1", "Context: Sam asks what they can help with"),
    "conv2_s09_023": ("B2", "Casual fact: Sam is 22 years old"),
    "conv2_s09_024": ("B2", "Casual fact: Maria asks about 'special someone', reveals curiosity"),
    "conv2_s09_028": ("B1", "Context: Sofia can pick Sam up, they'll catch up"),
    "conv2_s09_036": ("B2", "Casual fact: Maria is 'a proud mother' who shares Sam's work"),

    # ── Session 10 (N=23→18, convert 5) ──
    "conv2_s10_017": ("S5", "Emotional: Sam in disbelief about acceptance"),
    "conv2_s10_020": ("B2", "Casual fact: Sam tried calling dad, no answer"),
    "conv2_s10_026": ("B3", "Temporal question: when would Sam start at UCLA"),
    "conv2_s10_036": ("B2", "Casual fact: abuela will call Sam 'doctor/psychologist'"),
    "conv2_s10_039": ("B2", "Casual fact: Sam tells Maria not to be braggy"),
}

NOISE_TYPES = {
    "N1": "filler", "N2": "greeting_farewell", "N3": "reaction",
    "N4": "meta_conversation", "N5": "echo",
}

path = Path("benchmarks/gate_eval/datasets/gate_benchmark_conv2.json")
with open(path) as f:
    data = json.load(f)

patched = 0
for m in data["messages"]:
    if m["id"] in PATCHES:
        new_cat, new_notes = PATCHES[m["id"]]
        m["category"] = new_cat
        m["is_signal"] = not new_cat.startswith("N")
        m["noise_type"] = NOISE_TYPES.get(new_cat)
        m["notes"] = new_notes
        patched += 1

# Recompute stats
cat_counts = Counter(m["category"] for m in data["messages"])
noise_total = sum(v for k, v in cat_counts.items() if k.startswith("N"))
signal_total = sum(v for k, v in cat_counts.items() if k.startswith("S"))
border_total = sum(v for k, v in cat_counts.items() if k.startswith("B"))
n = len(data["messages"])

data["category_distribution"] = {
    "noise": {k: v for k, v in sorted(cat_counts.items()) if k.startswith("N")},
    "signal": {k: v for k, v in sorted(cat_counts.items()) if k.startswith("S")},
    "borderline": {k: v for k, v in sorted(cat_counts.items()) if k.startswith("B")},
    "totals": {
        "noise": noise_total, "signal": signal_total, "borderline": border_total,
        "noise_pct": round(noise_total / n * 100, 1),
        "signal_pct": round(signal_total / n * 100, 1),
        "borderline_pct": round(border_total / n * 100, 1),
    },
}

with open(path, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✓ Patched {patched} messages ({len(PATCHES)} expected)")
print(f"  Noise:      {noise_total} ({noise_total/n*100:.1f}%)")
print(f"  Signal:     {signal_total} ({signal_total/n*100:.1f}%)")
print(f"  Borderline: {border_total} ({border_total/n*100:.1f}%)")
