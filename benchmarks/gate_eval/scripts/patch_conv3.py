#!/usr/bin/env python3
"""Patch conv3 to fix noise/borderline ratio: convert ~35 N→B/S to hit 55% noise."""

import json
from collections import Counter
from pathlib import Path

PATCHES = {
    # Session 1 (convert 3)
    "conv3_s01_003": ("B2", "Casual fact: great sleep quality"),
    "conv3_s01_014": ("B3", "Temporal: meeting around 4pm"),
    "conv3_s01_022": ("B3", "Temporal: dinner at 7"),

    # Session 2 (convert 3)
    "conv3_s02_007": ("B2", "Casual fact: only ate granola bar at 2pm"),
    "conv3_s02_017": ("B1", "Context: offers support for potential career change"),
    "conv3_s02_019": ("B1", "Context: Casey needs comfort, sets up Netflix routine"),

    # Session 3 (convert 4)
    "conv3_s03_011": ("B1", "Context: affordability concern about trip"),
    "conv3_s03_013": ("B1", "Context: references $890 flight price"),
    "conv3_s03_021": ("B3", "Temporal: requesting time off in March"),
    "conv3_s03_025": ("B2", "Casual characterization: Dev is a planner"),

    # Session 4 (convert 4)
    "conv3_s04_007": ("B1", "Context question: has Gerald done this before"),
    "conv3_s04_016": ("S5", "Emotional: considering career change"),
    "conv3_s04_021": ("S5", "Emotional: gratitude expressed with vulnerability"),
    "conv3_s04_023": ("S5", "Emotional: happy crying"),

    # Session 5 (convert 4)
    "conv3_s05_002": ("B2", "Casual fact: Casey loves Dev's planning style"),
    "conv3_s05_021": ("S5", "Emotional: moved by Dev's financial offer"),
    "conv3_s05_022": ("S5", "Emotional: Dev wants trip to be special"),
    "conv3_s05_027": ("S5", "Emotional: anticipation about Japan trip"),

    # Session 6 (convert 3)
    "conv3_s06_009": ("B2", "Casual fact: Casey has always had a good eye for design"),
    "conv3_s06_013": ("B1", "Context: evaluating bootcamp cost"),
    "conv3_s06_023": ("B1", "Context: Dev offers zero-pressure support"),

    # Session 7 (convert 3)
    "conv3_s07_015": ("S2", "Design preference: herringbone subway tile"),
    "conv3_s07_024": ("B2", "Casual fact: Casey getting excited about renovation"),
    "conv3_s07_027": ("B2", "Casual fact: acknowledges adulting milestone"),

    # Session 8 (convert 2)
    "conv3_s08_002": ("B1", "Context question: clarifying from-scratch ramen process"),

    # Session 9 (convert 2)
    "conv3_s09_015": ("B1", "Context: financial plan during bootcamp"),

    # Session 10 (convert 1)
    # Session 10 is already high-noise by design, only convert 1

    # Session 11 (convert 2)
    "conv3_s11_006": ("B3", "Temporal: about to share specific details"),

    # Session 12 (convert 2)
    "conv3_s12_016": ("B1", "Context: references earlier contractor quotes"),

    # Session 14 (convert 2)
    "conv3_s14_027": ("B1", "Context: setting up serious topic transition"),

    # N3 conversions (convert 3)
    "conv3_s03_015": ("B2", "Casual fact: expected $2k cost, surprised by lower price"),
    "conv3_s05_010": ("B1", "Context question about Kyoto duration"),
    "conv3_s06_004": ("B1", "Context: surprised by Casey's design interest"),

    # N5 conversions (convert 2)
    "conv3_s06_007": ("B3", "Temporal: Casey spent a weekend on design projects"),
    "conv3_s04_006": ("B2", "Casual fact: error was just a comma"),
}

NOISE_TYPES = {
    "N1": "filler", "N2": "greeting_farewell", "N3": "reaction",
    "N4": "meta_conversation", "N5": "echo",
}

path = Path("benchmarks/gate_eval/datasets/gate_benchmark_conv3.json")
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

cat_counts = Counter(m["category"] for m in data["messages"])
n = len(data["messages"])
noise_total = sum(v for k, v in cat_counts.items() if k.startswith("N"))
signal_total = sum(v for k, v in cat_counts.items() if k.startswith("S"))
border_total = sum(v for k, v in cat_counts.items() if k.startswith("B"))

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

print(f"✓ Patched {patched}/{len(PATCHES)} messages")
print(f"  Noise:      {noise_total} ({noise_total/n*100:.1f}%)")
print(f"  Signal:     {signal_total} ({signal_total/n*100:.1f}%)")
print(f"  Borderline: {border_total} ({border_total/n*100:.1f}%)")
