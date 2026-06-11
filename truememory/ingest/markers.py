"""
Shared contradiction / update marker vocabulary (issue #649)
============================================================

A correction ("Correction: X", "that's incorrect", "no longer Y") must be
treated coherently by BOTH the encoding gate and the dedup stage. Before
this module the gate's contradiction vocabulary (``_CONTRADICTION_MARKERS``)
and dedup's update vocabulary (``_UPDATE_MARKER_PATTERNS``) diverged, so a
correction could pass the gate (recognised as a contradiction) and then be
silently SKIPped by dedup (NOT recognised as an update) before LLM
arbitration ever ran. M-13.

This module is the single source of truth. The gate consumes
:data:`UPDATE_MARKERS` for substring/startswith matching; dedup consumes
:data:`UPDATE_MARKER_PATTERNS` (the same vocabulary compiled to regex, plus
number/date-change patterns that have no plain-substring equivalent).
"""

from __future__ import annotations

import re

# Plain-substring markers — words/phrases that signal a correction or a
# fact update. Used by the encoding gate (startswith / " marker " match)
# and compiled into word-boundary regexes for dedup below.
#
# Keep these lowercase and free of leading/trailing punctuation where a
# word-boundary regex is meaningful; entries with trailing punctuation
# (e.g. "correction:") are matched as substrings by the gate.
UPDATE_MARKERS: tuple[str, ...] = (
    "actually",
    "correction:",
    "correction -",
    "no longer",
    "not anymore",
    "changed to",
    "changed from",
    "switched to",
    "switched from",
    "moved to",
    "used to be",
    "used to",
    "instead of",
    "wrong about",
    "was wrong",
    "is wrong",
    "not true",
    "isn't true",
    "that's incorrect",
    "that is incorrect",
    "updated",
    "replaced",
    "formerly",
    "previously",
)


def _compile_markers() -> list[re.Pattern[str]]:
    """Compile :data:`UPDATE_MARKERS` plus number/date-change patterns.

    Word-boundary anchored so "actually" matches but "actualization" does
    not. Markers ending in punctuation (``correction:``) are matched as a
    leading-boundary substring since ``\\b`` does not sit before ``:``.
    """
    patterns: list[re.Pattern[str]] = []
    for marker in UPDATE_MARKERS:
        if marker[-1].isalnum():
            patterns.append(re.compile(rf"\b{re.escape(marker)}\b", re.IGNORECASE))
        else:
            patterns.append(re.compile(rf"\b{re.escape(marker)}", re.IGNORECASE))
    # Structural change patterns with no plain-substring equivalent.
    patterns.extend([
        # "now is/uses/prefers/lives/works/takes/runs/has ..."
        re.compile(
            r"\bnow\s+(?:is|uses?|prefers?|lives?|works?|takes?|runs?|has)\b",
            re.IGNORECASE,
        ),
        # "was ... now ..."
        re.compile(r"\bwas\b.*\bnow\b", re.IGNORECASE),
        # Number-change patterns  ("5mg to 10mg", "6.5% -> 6.25%")
        re.compile(r"\d[\d.]*[%a-zA-Z]*\s*(?:to|->|-->|=>|→)\s*\d[\d.]*", re.IGNORECASE),
        # Date-change patterns  ("since 2024", "as of March")
        re.compile(r"\b(?:since|as\s+of|starting|effective)\s+\w+", re.IGNORECASE),
    ])
    return patterns


# Regex form for dedup (and any caller that needs full-text scanning).
UPDATE_MARKER_PATTERNS: list[re.Pattern[str]] = _compile_markers()


def has_update_markers(content: str) -> bool:
    """Return True if *content* contains correction/update language.

    Single shared predicate used by both the gate and dedup so the two
    stages never disagree about what counts as a correction (#649).
    """
    for pattern in UPDATE_MARKER_PATTERNS:
        if pattern.search(content):
            return True
    return False
