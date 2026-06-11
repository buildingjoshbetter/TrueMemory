#!/usr/bin/env python3
"""Smart patcher for conv4 and conv5 — convert N→B based on content heuristics."""

import json, re
from collections import Counter
from pathlib import Path

NOISE_TYPES = {
    "N1": "filler", "N2": "greeting_farewell", "N3": "reaction",
    "N4": "meta_conversation", "N5": "echo",
}

TEMPORAL_WORDS = re.compile(
    r'\b(last|next|week|month|ago|tomorrow|yesterday|monday|tuesday|wednesday|'
    r'thursday|friday|saturday|sunday|january|february|march|april|may|june|'
    r'july|august|september|october|november|december|tonight|morning|afternoon|'
    r'evening|years?|months?|days?|hours?|minutes?|soon|recently|since|before|after)\b',
    re.I
)

FACT_INDICATORS = re.compile(
    r'(\$\d|#\d|\d+\s*(miles?|km|lbs?|kg|percent|hours?|minutes?|months?|weeks?)|'
    r'\b(actually|remember|btw|by the way)\b)',
    re.I
)


def classify_borderline(content):
    """Heuristic to pick best B category for a message."""
    if TEMPORAL_WORDS.search(content):
        return "B3", "Temporal marker in casual context"
    if FACT_INDICATORS.search(content):
        return "B2", "Casual embedded fact"
    if len(content) > 40:
        return "B2", "Casual fact in conversational context"
    return "B1", "Context-enabling detail"


def patch_file(filepath, target_noise_pct, max_conversions):
    with open(filepath) as f:
        data = json.load(f)

    msgs = data["messages"]
    n = len(msgs)

    # Current noise count
    current_noise = sum(1 for m in msgs if m["category"].startswith("N"))
    target_noise = int(n * target_noise_pct / 100)
    needed = current_noise - target_noise
    needed = min(needed, max_conversions)

    if needed <= 0:
        print(f"  {filepath}: noise already at target ({current_noise/n*100:.1f}%)")
        return

    # Find candidates: N1 messages with content > 25 chars (more likely to contain info)
    candidates = []
    for m in msgs:
        if m["category"] == "N1" and len(m["content"]) > 25:
            candidates.append(m)

    # Also include N4/N5 with content > 30 chars
    for m in msgs:
        if m["category"] in ("N4", "N5") and len(m["content"]) > 30:
            candidates.append(m)

    # Sort by content length (longer = more likely to have embedded facts)
    candidates.sort(key=lambda m: len(m["content"]), reverse=True)

    converted = 0
    for m in candidates:
        if converted >= needed:
            break
        new_cat, new_notes = classify_borderline(m["content"])
        m["category"] = new_cat
        m["is_signal"] = True
        m["noise_type"] = None
        m["notes"] = new_notes
        converted += 1

    # Recompute stats
    cc = Counter(m["category"] for m in msgs)
    noise = sum(v for k, v in cc.items() if k.startswith("N"))
    signal = sum(v for k, v in cc.items() if k.startswith("S"))
    border = sum(v for k, v in cc.items() if k.startswith("B"))

    data["category_distribution"] = {
        "noise": {k: v for k, v in sorted(cc.items()) if k.startswith("N")},
        "signal": {k: v for k, v in sorted(cc.items()) if k.startswith("S")},
        "borderline": {k: v for k, v in sorted(cc.items()) if k.startswith("B")},
        "totals": {
            "noise": noise, "signal": signal, "borderline": border,
            "noise_pct": round(noise / n * 100, 1),
            "signal_pct": round(signal / n * 100, 1),
            "borderline_pct": round(border / n * 100, 1),
        },
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  {Path(filepath).name}: converted {converted} N→B")
    print(f"    Noise:  {noise} ({noise/n*100:.1f}%) [target {target_noise_pct}%]")
    print(f"    Signal: {signal} ({signal/n*100:.1f}%)")
    print(f"    Border: {border} ({border/n*100:.1f}%)")


print("Patching conv4 (target 40% noise)...")
patch_file("benchmarks/gate_eval/datasets/gate_benchmark_conv4.json", 40, 45)

print("\nPatching conv5 (target 50% noise)...")
patch_file("benchmarks/gate_eval/datasets/gate_benchmark_conv5.json", 50, 35)
