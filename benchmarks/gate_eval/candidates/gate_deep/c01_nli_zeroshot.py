"""C01: Zero-Shot NLI Classification — refined Phase 14 approach.

Uses MoritzLaurer's zero-shot NLI classifiers to score messages as
"substantive" vs "conversational filler". Optionally prepends the
D03 regex prefilter.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

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

_CANDIDATE_LABELS = [
    "substantive personal information, preference, decision, or factual statement",
    "conversational filler, pleasantry, acknowledgment, or small talk",
]
_HYPOTHESIS_TEMPLATE = "This message is {}."

CACHE_DIR = Path.home() / ".cache" / "gate_deep_nli"

_pipeline = None
_cache: dict[str, float] = {}
_cache_loaded = False


def _load_cache(model_name: str) -> None:
    global _cache, _cache_loaded
    if _cache_loaded:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_name.replace("/", "_")
    path = CACHE_DIR / f"{safe}.jsonl"
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    _cache[rec["h"]] = rec["p"]
                except Exception:
                    pass
    _cache_loaded = True


def _save_to_cache(model_name: str, msg_hash: str, score: float) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_name.replace("/", "_")
    path = CACHE_DIR / f"{safe}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps({"h": msg_hash, "p": score}) + "\n")


def _msg_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


def _get_pipeline(model_name: str):
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from transformers import pipeline
    print(f"  [c01_nli] Loading {model_name}...", flush=True)
    _pipeline = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1,
    )
    return _pipeline


def _nli_score(text: str, model_name: str) -> float:
    _load_cache(model_name)
    h = _msg_hash(text)
    if h in _cache:
        return _cache[h]

    clf = _get_pipeline(model_name)
    result = clf(
        text,
        candidate_labels=_CANDIDATE_LABELS,
        hypothesis_template=_HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    top_label = result["labels"][0]
    top_score = float(result["scores"][0])
    p_subst = top_score if top_label == _CANDIDATE_LABELS[0] else 1.0 - top_score

    _cache[h] = p_subst
    _save_to_cache(model_name, h, p_subst)
    return p_subst


def _regex_is_chitchat(text: str) -> bool:
    s = text.strip()
    if not s or len(s) < 4:
        return True
    lower = s.lower()
    if lower in _CHITCHAT_EXACT:
        return True
    if _CHITCHAT_REGEX.fullmatch(lower):
        return True
    if _PUNCT_ONLY.fullmatch(s):
        return True
    return False


class C01NliZeroshot:
    name = "c01_nli_zeroshot"
    tier = "base"

    def __init__(
        self,
        model_name: str = "MoritzLaurer/roberta-base-zeroshot-v2.0-c",
        threshold: float = 0.50,
        use_regex_prefilter: bool = True,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.use_regex_prefilter = use_regex_prefilter

    def importance_score(self, message: dict, context: list[dict] | None = None) -> float:
        text = (message.get("content") or "").strip()
        if not text:
            return 0.0
        if self.use_regex_prefilter and _regex_is_chitchat(text):
            return 0.0
        return _nli_score(text, self.model_name)

    def should_encode(self, message: dict, context: list[dict] | None = None) -> bool:
        return self.importance_score(message, context) >= self.threshold
