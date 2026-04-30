"""
Encoding Salience Scorer
========================

Scores messages for ENCODING importance ("is this worth remembering?"),
NOT retrieval utility ("would this help answer a query?"). The retrieval
scorer lives in truememory/salience.py (L3) and is unchanged.

Five variants are implemented. The winner is selected empirically on
GateLoCoMo and wired into encoding_gate.py's _compute_salience().
"""

from __future__ import annotations

import json
import re
from math import exp, log
from pathlib import Path

from truememory.salience import (
    _HIGH_AROUSAL,
    _LIFE_EVENTS,
    _NOISE_EXACT,
    _EMOJI_PATTERN,
    _EMOJI_RE,
    _MONEY_PATTERN,
    _extract_features,
    compute_message_salience,
)


_COMMON_WORDS = frozenset({
    "i", "a", "the", "is", "it", "in", "to", "and", "of", "for", "on", "at",
    "by", "an", "or", "so", "if", "my", "we", "he", "she", "do", "no", "up",
    "be", "but", "not", "you", "all", "can", "her", "was", "one", "our", "out",
    "are", "has", "his", "how", "its", "let", "may", "new", "now", "old", "see",
    "way", "who", "did", "get", "got", "had", "him", "own", "say", "too", "use",
    "oh", "hey", "hi", "ok", "yeah", "yes", "like", "just", "that", "this",
    "with", "have", "from", "they", "been", "said", "each", "when", "what",
    "your", "will", "than", "them", "then", "some", "time", "very", "make",
    "also", "into", "only", "come", "made", "well", "back", "much", "more",
    "about", "would", "could", "after", "first", "other", "these", "which",
    "those", "here", "there", "even", "still", "down", "off", "over", "such",
    "take", "find", "give", "most", "tell", "think", "help", "every", "last",
    "long", "great", "little", "own", "right", "going", "know", "want",
    "actually", "really", "literally", "honestly", "though", "because",
    "already", "probably", "definitely", "seriously",
})

_UPDATE_VERBS = frozenset({
    "switched", "changed", "moved", "quit", "started", "enrolled",
    "promoted", "graduated", "launched", "resigned", "transferred",
    "hired", "fired", "accepted", "declined", "submitted",
})

_COMMITMENT_PATTERNS = frozenset({
    "said yes", "said no", "i'm in", "we're in", "i quit", "i did it",
    "i got it", "i got in", "i made it", "i passed", "i failed",
    "we're pregnant", "i'm pregnant", "she's pregnant",
    "i'm engaged", "we're engaged", "i'm married", "we're married",
    "i enrolled", "i applied", "i submitted", "i accepted",
    "i declined", "i resigned", "i'm leaving", "i'm moving",
    "it's booked", "it's done", "it's official", "it's over",
    "i had a baby", "had a baby", "having a baby",
    "seeing someone", "broke up", "breaking up",
    "got the job", "got the offer", "got accepted", "got rejected",
    "got promoted", "got fired", "got hired", "got laid off",
    "gave my notice", "two weeks notice", "gave notice",
    "passed away", "passed on",
})

# Category salience boost (same mapping as encoding_gate.py)
_CATEGORY_SALIENCE_BOOST = {
    "correction": 0.40,
    "decision": 0.30,
    "personal": 0.25,
    "preference": 0.25,
    "relationship": 0.20,
    "temporal": 0.15,
    "technical": 0.10,
    "general": 0.05,
}


# ---------------------------------------------------------------------------
# Variant A: Re-weighted L3 features (trained on GateLoCoMo)
# ---------------------------------------------------------------------------

_ENCODING_WEIGHTS_PATH = Path(__file__).parent.parent / "data" / "encoding_salience_weights.json"
_ENC_A_WEIGHTS: tuple[float, ...] | None = None
_ENC_A_BIAS: float | None = None

try:
    with open(_ENCODING_WEIGHTS_PATH) as _f:
        _enc_data = json.load(_f)
        _ENC_A_WEIGHTS = tuple(_enc_data["variant_a"]["weights"])
        _ENC_A_BIAS = float(_enc_data["variant_a"]["bias"])
        del _enc_data
except (FileNotFoundError, KeyError):
    pass


def encoding_salience_a(content: str, category: str = "") -> float:
    if not content or not content.strip():
        return 0.0
    if _ENC_A_WEIGHTS is None or _ENC_A_BIAS is None:
        return encoding_salience_c(content, category)
    features = _extract_features(content, "chat")
    logit = sum(w * f for w, f in zip(_ENC_A_WEIGHTS, features)) + _ENC_A_BIAS
    return 1.0 / (1.0 + exp(-logit))


# ---------------------------------------------------------------------------
# Variant B: Short-message boost
# ---------------------------------------------------------------------------

def _has_info_markers(content: str) -> bool:
    text = content.strip()
    text_lower = text.lower()
    if re.search(r'\d', text):
        return True
    if '$' in text:
        return True
    words = text.split()
    if any(w.isupper() and len(w) > 1 for w in words):
        return True
    if any(w[0:1].isupper() and w.lower() not in _COMMON_WORDS and len(w) > 1 for w in words):
        return True
    if any(phrase in text_lower for phrase in _HIGH_AROUSAL):
        return True
    if any(phrase in text_lower for phrase in _LIFE_EVENTS):
        return True
    return False


def encoding_salience_b(content: str, category: str = "") -> float:
    if not content or not content.strip():
        return 0.0
    base = compute_message_salience(content, "chat")
    length = max(1, len(content.strip()))
    if length < 50 and _has_info_markers(content):
        return min(1.0, base * (50.0 / length))
    return base


# ---------------------------------------------------------------------------
# Variant C: Rule-based importance scorer (no length dependency)
# ---------------------------------------------------------------------------

def encoding_salience_c(content: str, category: str = "") -> float:
    if not content or not content.strip():
        return 0.0

    text = content.strip()
    text_lower = text.lower()

    # Noise override first
    noise_stripped = text_lower.strip("!?.… ")
    if noise_stripped in _NOISE_EXACT:
        return 0.02

    # Emoji-only
    if _EMOJI_PATTERN.match(text):
        return 0.03

    score = 0.10

    # ALL-CAPS words
    caps_words = [w for w in text.split() if w.isupper() and len(w) > 1]
    score += min(0.50, len(caps_words) * 0.20)

    # Commitment / announcement patterns
    commit_hits = sum(1 for p in _COMMITMENT_PATTERNS if p in text_lower)
    score += min(0.50, commit_hits * 0.35)

    # Life event phrases
    life_hits = sum(1 for phrase in _LIFE_EVENTS if phrase in text_lower)
    score += min(0.50, life_hits * 0.30)

    # High-arousal words
    arousal_hits = sum(1 for w in _HIGH_AROUSAL if w in text_lower)
    score += min(0.50, arousal_hits * 0.30)

    # Numbers and money
    numbers = len(re.findall(r'\d+', text))
    money = len(_MONEY_PATTERN.findall(text))
    score += min(0.30, (numbers + money * 2) * 0.10)

    # Proper nouns
    proper = [w for w in text.split() if w[0:1].isupper() and w.lower() not in _COMMON_WORDS and len(w) > 1]
    score += min(0.30, len(proper) * 0.10)

    # Exclamation marks
    excl = text.count("!")
    score += min(0.15, excl * 0.05)

    # Update verbs
    if any(v in text_lower for v in _UPDATE_VERBS):
        score += 0.15

    # Category boost
    cat_boost = _CATEGORY_SALIENCE_BOOST.get((category or "").strip().lower(), 0.05)
    score += cat_boost

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Variant D: Hybrid — L3 for long messages, speech-act scorer for short
# ---------------------------------------------------------------------------

# Speech act patterns for short-message scoring (v23).
# Classifies messages by linguistic function rather than content,
# making it length-independent. Validated via 50-variant sweep on
# GateLoCoMo: AUC 0.726 on short messages (<50 chars), 96% S4 recall.

_NOISE_EXACT_V23 = frozenset({
    "ok", "okay", "k", "kk", "yes", "yeah", "yep", "yup", "ya", "yea",
    "no", "nah", "nope", "lol", "lmao", "lmfao", "haha", "hahaha", "heh",
    "omg", "omfg", "wtf", "nice", "cool", "dope", "sick", "lit", "fire",
    "thanks", "thx", "ty", "thank you", "got it", "gotcha",
    "sounds good", "sounds great", "bet", "word", "sure", "for sure",
    "same", "mood", "idk", "idc", "np", "no problem",
    "gn", "goodnight", "good night", "gm", "good morning", "brb", "ttyl",
    "damn", "dude", "bro", "ugh", "wow", "yikes", "ooh", "oof",
    "true", "facts", "right", "exactly", "totally", "absolutely",
    "lmao dead", "im dead", "crying", "screaming",
    # Reactions to someone else's news (N2 noise, issue #118)
    "that's great", "thats great", "that's awesome", "thats awesome",
    "that's amazing", "thats amazing", "that's crazy", "thats crazy",
    "that's insane", "thats insane", "that's wild", "thats wild",
    "that's so cool", "thats so cool",
    "congratulations", "congrats", "happy for you", "so happy for you",
    "proud of you", "so proud of you", "good for you",
    "no way", "are you serious", "oh my god", "oh my gosh",
    "i can't believe it", "i cant believe it", "shut up",
    "that's wonderful", "thats wonderful", "that's fantastic",
    "love that", "love it", "so cool", "so sick",
    "good luck", "you got this", "go for it", "let's go", "lets go",
    "aww", "aw", "yay", "woohoo", "woo",
})

_COMMITMENT_RE = re.compile(
    r"\b(?:"
    r"i\s+(?:got|did|made|found|built|started|quit|left|joined|enrolled|"
    r"accepted|submitted|finished|completed|signed|bought|sold|moved|"
    r"said|told|asked|proposed|created|launched|shipped|published|"
    r"passed|graduated|earned|won|lost|broke|fixed)"
    r"|i'm\s+(?:pregnant|engaged|leaving|moving|starting|quitting|"
    r"going\s+to|seeing\s+someone|having\s+a)"
    r"|we're\s+(?:pregnant|engaged|moving|having|getting|doing)"
    r"|i\s+have\s+(?:a\s+baby|cancer|diabetes|a\s+new)"
    r"|she\s+(?:promoted|said\s+yes|agreed|accepted)"
    r"|he\s+(?:proposed|said\s+yes|agreed|accepted)"
    r"|it's\s+(?:booked|official|confirmed|done|over|happening)"
    r"|i\s+gave\s+(?:my\s+(?:two\s+weeks|notice))"
    r"|(?:all|both)\s+(?:three|four|five)?\s*(?:apps?|applications?)\s+submitted"
    r")\b",
    re.IGNORECASE,
)


def _speech_act_score(content: str) -> float:
    """Classify short messages by speech act. Length-independent."""
    lower = content.lower().strip()
    if lower in _NOISE_EXACT_V23:
        return 0.02
    if content.strip().endswith("?") or lower.startswith((
        "what ", "how ", "why ", "where ", "when ", "who ", "which ",
        "do you", "are you", "is it", "can you", "could you",
    )):
        return 0.2
    if _COMMITMENT_RE.search(lower):
        return 0.8
    if any(p in lower for p in _COMMITMENT_PATTERNS):
        return 0.7
    if (
        re.search(r"\b(?:no longer|not anymore|instead|correction)\b", lower)
        or any(v in lower for v in _UPDATE_VERBS)
        or ("actually" in lower and re.search(r"\bnot\b", lower))
    ):
        return 0.6
    if re.match(r"^(?:hey|hi|hello|yo|sup|what's up|howdy)", lower):
        return 0.05
    if re.match(r"^(?:haha|lol|lmao|omg|wow|damn|ugh|yikes)", lower):
        return 0.08
    words = re.findall(r"[a-zA-Z]+", content)
    if len(words) >= 5:
        return 0.5
    return 0.25


def encoding_salience_d(content: str, category: str = "") -> float:
    if not content or not content.strip():
        return 0.0
    length = len(content.strip())
    if length <= 50:
        return _speech_act_score(content)
    else:
        return compute_message_salience(content, "chat")


# ---------------------------------------------------------------------------
# Variant E: Fine-tuned classifier with encoding-specific features
# ---------------------------------------------------------------------------

_ENC_E_WEIGHTS: tuple[float, ...] | None = None
_ENC_E_BIAS: float | None = None

try:
    with open(_ENCODING_WEIGHTS_PATH) as _f:
        _enc_data_e = json.load(_f)
        _ENC_E_WEIGHTS = tuple(_enc_data_e["variant_e"]["weights"])
        _ENC_E_BIAS = float(_enc_data_e["variant_e"]["bias"])
        del _enc_data_e
except (FileNotFoundError, KeyError):
    pass


def _extract_encoding_features(content: str, category: str = "") -> tuple[float, ...]:
    text = content.strip()
    text_lower = text.lower()
    noise_stripped = text_lower.strip("!?.… ")

    f_length = log(1 + len(text)) / 7.0
    f_noise = 1.0 if noise_stripped in _NOISE_EXACT else 0.0

    caps_words = [w for w in text.split() if w.isupper() and len(w) > 1]
    f_caps_words = min(1.0, len(caps_words) / 5.0)

    life_hits = sum(1 for phrase in _LIFE_EVENTS if phrase in text_lower)
    f_life_events = min(1.0, life_hits / 2.0)

    arousal_hits = sum(1 for w in _HIGH_AROUSAL if w in text_lower)
    f_arousal = min(1.0, arousal_hits / 3.0)

    commit_hits = sum(1 for p in _COMMITMENT_PATTERNS if p in text_lower)
    f_commitment = min(1.0, commit_hits / 2.0)

    numbers = len(re.findall(r'\d+', text))
    f_numbers = min(1.0, numbers / 3.0)

    money = len(_MONEY_PATTERN.findall(text))
    f_money = min(1.0, money / 2.0)

    proper = [w for w in text.split() if w[0:1].isupper() and w.lower() not in _COMMON_WORDS and len(w) > 1]
    f_proper_nouns = min(1.0, len(proper) / 5.0)

    f_exclamation = min(1.0, text.count("!") / 3.0)
    f_question = 1.0 if text.rstrip().endswith("?") else 0.0

    f_update_verb = 1.0 if any(v in text_lower for v in _UPDATE_VERBS) else 0.0

    has_number = numbers > 0
    has_proper = len(proper) > 0
    has_caps = len(caps_words) > 0
    f_short_dense = 1.0 if (len(text) < 30 and (has_number or has_proper or has_caps)) else 0.0

    if text:
        emoji_chars = sum(1 for _ in _EMOJI_RE.finditer(text))
        non_space = len(text.replace(" ", ""))
        f_emoji_only = 1.0 if (non_space > 0 and emoji_chars / non_space > 0.8) else 0.0
    else:
        f_emoji_only = 0.0

    cat_boost = _CATEGORY_SALIENCE_BOOST.get((category or "").strip().lower(), 0.05)
    f_category_boost = cat_boost

    return (
        f_length, f_noise, f_caps_words, f_life_events, f_arousal,
        f_commitment, f_numbers, f_money, f_proper_nouns, f_exclamation,
        f_question, f_update_verb, f_short_dense, f_emoji_only, f_category_boost,
    )


def encoding_salience_e(content: str, category: str = "") -> float:
    if not content or not content.strip():
        return 0.0
    if _ENC_E_WEIGHTS is None or _ENC_E_BIAS is None:
        return encoding_salience_c(content, category)
    features = _extract_encoding_features(content, category)
    logit = sum(w * f for w, f in zip(_ENC_E_WEIGHTS, features)) + _ENC_E_BIAS
    return 1.0 / (1.0 + exp(-logit))
