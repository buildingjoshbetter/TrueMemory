"""D2: Length-only filter — drop messages under N characters."""
from __future__ import annotations


class D02LengthOnly:
    name = "d02_length_only"
    tier = "all"

    def __init__(self, min_length: int = 20):
        self.min_length = min_length

    def should_encode(self, message: dict, context: list[dict] | None = None) -> bool:
        return len((message.get("content") or "").strip()) >= self.min_length

    def importance_score(self, message: dict, context: list[dict] | None = None) -> float:
        text = (message.get("content") or "").strip()
        length = len(text)
        if length < self.min_length:
            return 0.0
        return min(1.0, length / 200.0)
