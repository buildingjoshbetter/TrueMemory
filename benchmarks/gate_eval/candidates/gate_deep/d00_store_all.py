"""D0: Store Everything — baseline gate that encodes all messages."""
from __future__ import annotations


class D00StoreAll:
    name = "d00_store_all"
    tier = "all"

    def should_encode(self, message: dict, context: list[dict] | None = None) -> bool:
        return True

    def importance_score(self, message: dict, context: list[dict] | None = None) -> float:
        return 1.0
