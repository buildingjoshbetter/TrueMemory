"""C02: Compression-Based Gating (NCD).

Uses gzip Normalized Compression Distance to measure how much new
information a message adds relative to recent stored messages.
Zero dependencies (stdlib gzip). Edge-tier safe (<1ms per message).
"""
from __future__ import annotations

import gzip
from collections import deque


def _compressed_len(text: str) -> int:
    return len(gzip.compress(text.encode("utf-8", errors="replace")))


def _ncd(x: str, y: str) -> float:
    """Normalized Compression Distance between two strings."""
    if not x or not y:
        return 1.0
    cx = _compressed_len(x)
    cy = _compressed_len(y)
    cxy = _compressed_len(x + " " + y)
    denominator = max(cx, cy)
    if denominator == 0:
        return 1.0
    return (cxy - min(cx, cy)) / denominator


class C02Compression:
    name = "c02_compression"
    tier = "all"

    def __init__(self, context_window: int = 30, ncd_threshold: float = 0.50):
        self.context_window = context_window
        self.ncd_threshold = ncd_threshold
        self._recent: deque[str] = deque(maxlen=context_window)

    def importance_score(self, message: dict, context: list[dict] | None = None) -> float:
        text = (message.get("content") or "").strip()
        if not text:
            return 0.0
        if len(text) < 4:
            return 0.0

        if not self._recent:
            return 1.0

        context_str = " ".join(self._recent)
        ncd = _ncd(context_str, text)
        return ncd

    def should_encode(self, message: dict, context: list[dict] | None = None) -> bool:
        score = self.importance_score(message, context)
        keep = score >= self.ncd_threshold
        if keep:
            text = (message.get("content") or "").strip()
            if text:
                self._recent.append(text)
        return keep

    def reset(self):
        self._recent.clear()
