"""
Shared feature extraction for L3 candidates C1-C5 and C8a.

Produces the 13 continuous features documented in
CANDIDATES.md §F, plus a convenience "all_zeros" check used by
the hand-weight fallbacks.
"""
from __future__ import annotations

import gzip
import math
import re

_NOISE_EXACT = frozenset({
    "ok", "okay", "k", "kk", "yes", "yeah", "yep", "yup", "ya", "yea",
    "no", "nah", "nope", "lol", "lmao", "lmfao", "haha", "hahaha", "heh",
    "omg", "omfg", "wtf", "nice", "cool", "dope", "sick", "lit", "fire",
    "thanks", "thx", "ty", "thank you", "got it", "gotcha",
    "sounds good", "sounds great", "bet", "word", "sure", "for sure",
    "same", "mood", "idk", "idc", "np", "no problem", "gn", "goodnight",
    "good night", "gm", "good morning", "brb", "ttyl",
    "?", "??", "???", "!", "!!", "!!!",
})

_EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF"
    r"\U00002702-\U000027B0"
    r"\U000024C2-\U0001F251"
    r"\U0001f900-\U0001f9FF"
    r"\U0001fa00-\U0001fa6f"
    r"\U0001fa70-\U0001faff]",
    re.UNICODE,
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")
_DATE_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"\s+\d{1,2}",
    re.IGNORECASE,
)
_HIGH_SIGNAL_MODALITIES = frozenset({
    "ocr", "email", "calendar", "note", "health_data", "strava",
    "receipt", "document", "bank_statement", "vet_record", "cap_table",
    "home_inspection",
})
_CAPS_WORDS_RE = re.compile(r"\b[A-Z]{3,}\b")
_BULLET_RE = re.compile(r"^[-*•]\s", re.MULTILINE)

_HIGH_AROUSAL = frozenset({
    "amazing", "incredible", "devastating", "heartbreaking",
    "thrilled", "furious", "terrified", "ecstatic", "crushed",
    "panic", "emergency", "urgent", "critical", "breakthrough",
    "milestone", "promoted", "fired", "pregnant", "engaged",
    "diagnosed", "accident", "passed away", "died",
})
_LIFE_EVENTS = frozenset({
    "got married", "got engaged", "having a baby", "got promoted",
    "got fired", "broke up", "moved to", "graduated", "launched",
    "raised funding", "demo day", "ipo", "acquisition",
})


FEATURE_NAMES = (
    "f_noise", "f_emoji", "f_length", "f_num", "f_money", "f_date",
    "f_mod", "f_nl", "f_bul", "f_excl", "f_caps", "f_arou", "f_life",
)


def extract_features(msg: dict) -> list[float]:
    """Return a length-13 feature vector for one message dict."""
    text = msg.get("content", "") or ""
    mod = msg.get("modality", "") or ""
    text_stripped = text.strip()
    text_lower = text_stripped.lower().strip("!?.… ")

    f_noise = 1.0 if text_lower in _NOISE_EXACT else 0.0
    # Emoji fraction
    if text_stripped:
        emoji_chars = sum(1 for _ in _EMOJI_RE.finditer(text_stripped))
        f_emoji = min(1.0, emoji_chars / max(1, len(text_stripped)))
    else:
        f_emoji = 0.0
    f_length = math.log1p(len(text_stripped)) / 7.0  # log(1+1000) / 7 ≈ 1
    f_num = math.log1p(len(_NUMBER_RE.findall(text_stripped))) / 3.0
    f_money = min(1.0, len(_MONEY_RE.findall(text_stripped)) / 2.0)
    f_date = min(1.0, len(_DATE_RE.findall(text_stripped)) / 2.0)
    f_mod = 1.0 if mod.lower() in _HIGH_SIGNAL_MODALITIES else 0.0
    f_nl = 1.0 if ("\n" in text_stripped and len(text_stripped) > 50) else 0.0
    f_bul = 1.0 if _BULLET_RE.search(text_stripped) else 0.0
    excl_count = text_stripped.count("!")
    f_excl = min(1.0, excl_count / 3.0)
    caps_n = len(_CAPS_WORDS_RE.findall(text_stripped))
    f_caps = min(1.0, caps_n / 5.0)
    arou_hits = sum(1 for w in _HIGH_AROUSAL if w in text_lower)
    f_arou = min(1.0, arou_hits / 3.0)
    life_hits = sum(1 for e in _LIFE_EVENTS if e in text_lower)
    f_life = min(1.0, life_hits / 2.0)

    return [
        f_noise, f_emoji, f_length, f_num, f_money, f_date,
        f_mod, f_nl, f_bul, f_excl, f_caps, f_arou, f_life,
    ]


def gzip_ratio(text: str) -> float:
    """`gzip_len / original_len`. Constant on empty; in (0, ~10) bounds."""
    if not text:
        return 0.0
    b = text.encode("utf-8")
    comp = gzip.compress(b, compresslevel=6)
    return len(comp) / max(1, len(b))
