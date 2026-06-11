"""D3: Regex-only filter — Phase 14 Stage 1 regex without NLI."""
from __future__ import annotations

import re

_CHITCHAT_EXACT = {
    "lol", "lmao", "rofl", "haha", "hehe", "haha!",
    "ok", "okay", "k", "kk",
    "yeah", "yep", "yup", "yea", "yes",
    "no", "nope", "nah",
    "sure", "sure thing", "cool", "nice", "nice!",
    "thanks", "thank you", "ty", "thx",
    "np", "you're welcome", "welcome",
    "hi", "hey", "hello", "yo",
    "bye", "goodbye", "gn", "good night", "good morning",
    "wow", "omg", "damn", "dang",
    "same", "yeah same", "same here",
    "got it", "understood", "noted",
    "sounds good", "sounds great",
}

_CHITCHAT_REGEX = re.compile(
    r"^(lo+l|ha+|he+he*|lmao+|ikr+|rofl|wow+|omg+|hmm+|uh+|ah+)[!.?]*$",
    re.IGNORECASE,
)
_PUNCT_ONLY = re.compile(r"^[\W_]+$")


class D03RegexOnly:
    name = "d03_regex_only"
    tier = "all"

    def should_encode(self, message: dict, context: list[dict] | None = None) -> bool:
        return self.importance_score(message, context) > 0.0

    def importance_score(self, message: dict, context: list[dict] | None = None) -> float:
        text = (message.get("content") or "").strip()
        if not text:
            return 0.0
        if len(text) < 4:
            return 0.0
        lower = text.lower()
        if lower in _CHITCHAT_EXACT:
            return 0.0
        if _CHITCHAT_REGEX.fullmatch(lower):
            return 0.0
        if _PUNCT_ONLY.fullmatch(text):
            return 0.0
        return 1.0
