"""L5 Surprise Boost — predictive-coding surprise scoring.

Extracted from engine.py (issue #137) to reduce god-class complexity.
All functions take an explicit ``conn`` parameter instead of ``self``.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3

logger = logging.getLogger(__name__)

SURPRISE_BOOST_SOURCE_BLOCKLIST = frozenset({
    "personality", "profile", "style_vec", "summary", "contradiction",
})

SURPRISE_IN_CHUNK = 500

DEFAULT_ALPHA_SURPRISE = 0.2

_warned_no_surprise = False


def source_is_blocked(source: str | None) -> bool:
    """True if any '+'-separated segment of *source* is in the blocklist."""
    if not source:
        return False
    return any(
        seg in SURPRISE_BOOST_SOURCE_BLOCKLIST
        for seg in source.split("+")
    )


def get_alpha_surprise(override: float | None = None) -> float:
    """Resolve alpha_surprise: *override* > env var > 0.2.

    Sanitizes non-finite values and clamps negatives to 0.
    """
    if override is not None:
        try:
            a = float(override)
        except (TypeError, ValueError):
            return DEFAULT_ALPHA_SURPRISE
        if math.isnan(a) or math.isinf(a):
            return DEFAULT_ALPHA_SURPRISE
        return max(0.0, a)

    env = os.environ.get("TRUEMEMORY_ALPHA_SURPRISE")
    if env:
        try:
            a = float(env)
        except ValueError:
            logger.warning(
                "Invalid TRUEMEMORY_ALPHA_SURPRISE=%r; using default", env,
            )
            return DEFAULT_ALPHA_SURPRISE
        if math.isnan(a) or math.isinf(a):
            logger.warning(
                "Non-finite TRUEMEMORY_ALPHA_SURPRISE=%r; using default", env,
            )
            return DEFAULT_ALPHA_SURPRISE
        return max(0.0, a)

    return DEFAULT_ALPHA_SURPRISE


def apply_surprise_boost(
    conn: sqlite3.Connection,
    results: list[dict],
    alpha_override: float | None = None,
) -> list[dict]:
    """Apply L5 surprise multiplicative boost to message-backed rows.

    Mutates ``r["score"]`` so downstream re-sort is coherent.
    When ``alpha == 0`` this is an exact no-op.
    """
    global _warned_no_surprise

    if not results:
        return results
    alpha = get_alpha_surprise(alpha_override)
    if alpha <= 0.0:
        return results

    message_rows = [
        r for r in results
        if r.get("id") is not None
        and not source_is_blocked(r.get("source"))
    ]
    if not message_rows:
        return results

    ids = [r["id"] for r in message_rows]
    surprise_map: dict[int, float] = {}
    try:
        for i in range(0, len(ids), SURPRISE_IN_CHUNK):
            chunk = ids[i : i + SURPRISE_IN_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            cur = conn.execute(
                f"SELECT message_id, surprise FROM surprise_scores "
                f"WHERE message_id IN ({placeholders})",
                chunk,
            )
            surprise_map.update(dict(cur.fetchall()))
    except sqlite3.OperationalError as exc:
        if not _warned_no_surprise:
            logger.warning(
                "L5 surprise boost unavailable: %s (run consolidate first)",
                exc,
            )
            _warned_no_surprise = True
        return results
    except Exception:
        logger.warning(
            "L5 surprise boost failed; returning unboosted results",
            exc_info=True,
        )
        return results

    if not surprise_map:
        return results

    for r in message_rows:
        s = surprise_map.get(r["id"], 0.0)
        if s > 0.0:
            base = r.get("score", r.get("rerank_score", r.get("rrf_score", 0.0)))
            r["score"] = base * (1.0 + alpha * float(s))

    results = sorted(
        results,
        key=lambda r: r.get("score", 0.0),
        reverse=True,
    )
    return results
