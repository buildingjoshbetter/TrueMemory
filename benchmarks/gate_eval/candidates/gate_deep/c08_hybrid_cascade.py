"""C08: Hybrid Cascade Gate.

Multi-stage gate: regex → length+compression → NLI → context check.
Edge tier uses only stages 1-2. Base adds NLI. Pro adds context.
"""
from __future__ import annotations

import gzip
import re
from collections import deque

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
_HAS_ENTITY = re.compile(
    r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\$\d+|@\w+|\d+%|\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b)",
)
_HAS_NUMBER = re.compile(r"\b\d{2,}\b")


def _compressed_len(text: str) -> int:
    return len(gzip.compress(text.encode("utf-8", errors="replace")))


def _ncd(x: str, y: str) -> float:
    if not x or not y:
        return 1.0
    cx = _compressed_len(x)
    cy = _compressed_len(y)
    cxy = _compressed_len(x + " " + y)
    denominator = max(cx, cy)
    if denominator == 0:
        return 1.0
    return (cxy - min(cx, cy)) / denominator


class C08HybridCascade:
    name = "c08_hybrid_cascade"
    tier = "base"

    def __init__(
        self,
        nli_threshold: float = 0.50,
        ncd_dedup_threshold: float = 0.10,
        min_length: int = 15,
        use_nli: bool = True,
        use_context: bool = False,
        nli_model: str = "MoritzLaurer/roberta-base-zeroshot-v2.0-c",
        context_window: int = 20,
    ):
        self.nli_threshold = nli_threshold
        self.ncd_dedup_threshold = ncd_dedup_threshold
        self.min_length = min_length
        self.use_nli = use_nli
        self.use_context = use_context
        self.nli_model = nli_model
        self._recent: deque[str] = deque(maxlen=context_window)
        self._nli_module = None

    def _stage1_regex(self, text: str) -> tuple[bool, str]:
        """Stage 1: Regex prefilter. Returns (is_chitchat, reason)."""
        s = text.strip()
        if not s:
            return True, "empty"
        if len(s) < 4:
            return True, "too-short"
        lower = s.lower()
        if lower in _CHITCHAT_EXACT:
            return True, "chitchat-exact"
        if _CHITCHAT_REGEX.fullmatch(lower):
            return True, "chitchat-regex"
        if _PUNCT_ONLY.fullmatch(s):
            return True, "punct-only"
        return False, ""

    def _stage2_length_compression(self, text: str) -> tuple[str, float]:
        """Stage 2: Length + NCD dedup. Returns (decision, ncd_score).
        decision: 'drop', 'pass', 'uncertain'
        """
        s = text.strip()
        if len(s) < self.min_length:
            return "drop", 0.0

        if self._recent:
            context_str = " ".join(self._recent)
            ncd = _ncd(context_str, s)
            if ncd < self.ncd_dedup_threshold:
                return "drop", ncd
        return "pass", 1.0

    def _stage3_nli(self, text: str) -> tuple[str, float]:
        """Stage 3: NLI classification. Returns (decision, p_substantive)."""
        if self._nli_module is None:
            from benchmarks.gate_eval.candidates.gate_deep.c01_nli_zeroshot import _nli_score
            self._nli_module = _nli_score

        p = self._nli_module(text, self.nli_model)
        if p >= 0.70:
            return "encode", p
        if p < 0.30:
            return "drop", p
        return "uncertain", p

    def _stage4_context(self, text: str, context: list[dict] | None) -> str:
        """Stage 4: Context-aware check for borderline messages."""
        if _HAS_ENTITY.search(text) or _HAS_NUMBER.search(text):
            return "encode"

        if context and len(context) >= 1:
            prev = (context[-1].get("content") or "").strip()
            if prev.endswith("?"):
                return "encode"

        return "drop"

    def importance_score(self, message: dict, context: list[dict] | None = None) -> float:
        text = (message.get("content") or "").strip()
        if not text:
            return 0.0

        is_chitchat, _ = self._stage1_regex(text)
        if is_chitchat:
            return 0.0

        decision, _ = self._stage2_length_compression(text)
        if decision == "drop":
            return 0.1

        if self.use_nli:
            decision, p = self._stage3_nli(text)
            if decision == "encode":
                return p
            if decision == "drop":
                return p
            if self.use_context:
                ctx_decision = self._stage4_context(text, context)
                return 0.6 if ctx_decision == "encode" else 0.2
            return p

        return 0.7

    def should_encode(self, message: dict, context: list[dict] | None = None) -> bool:
        text = (message.get("content") or "").strip()
        if not text:
            return False

        is_chitchat, _ = self._stage1_regex(text)
        if is_chitchat:
            return False

        decision, _ = self._stage2_length_compression(text)
        if decision == "drop":
            return False

        if self.use_nli:
            decision, p = self._stage3_nli(text)
            if decision == "encode":
                self._recent.append(text)
                return True
            if decision == "drop":
                return False
            if self.use_context:
                ctx = self._stage4_context(text, context)
                if ctx == "encode":
                    self._recent.append(text)
                    return True
                return False
            keep = p >= self.nli_threshold
            if keep:
                self._recent.append(text)
            return keep

        self._recent.append(text)
        return True

    def reset(self):
        self._recent.clear()
