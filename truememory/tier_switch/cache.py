"""Vector cache registry — tracks per-tier-group vector table state.

Enables instant switch-back by keeping old vector tables and only
re-embedding the delta (new messages since last switch).
"""

import logging
import sqlite3
import time

import psutil

log = logging.getLogger(__name__)

_MODEL_NAMES = {
    "edge": "potion-base-8M",
    "basepro": "Qwen3-Embedding-0.6B",
}


def tier_group(tier: str) -> str:
    """Map a tier name to its embedding model group."""
    if tier == "edge":
        return "edge"
    if tier in ("base", "pro"):
        return "basepro"
    raise ValueError(f"Unknown tier: {tier!r}")


def model_name_for_group(group: str) -> str:
    """Return the canonical embedding model name for a tier group."""
    return _MODEL_NAMES.get(group, "")


class VectorCacheRegistry:
    """CRUD operations for the vector_cache_registry table."""

    @staticmethod
    def get(conn: sqlite3.Connection, group: str) -> dict | None:
        """Fetch cache entry for a tier group, or None."""
        row = conn.execute(
            "SELECT tier_group, vec_table, sep_table, last_embedded_id, "
            "vector_count, model_name, embedding_dim, last_updated, created "
            "FROM vector_cache_registry WHERE tier_group = ?",
            (group,),
        ).fetchone()
        if not row:
            return None
        return {
            "tier_group": row[0],
            "vec_table": row[1],
            "sep_table": row[2],
            "last_embedded_id": row[3],
            "vector_count": row[4],
            "model_name": row[5],
            "embedding_dim": row[6],
            "last_updated": row[7],
            "created": row[8],
        }

    @staticmethod
    def set(
        conn: sqlite3.Connection,
        group: str,
        *,
        vec_table: str | None = None,
        sep_table: str | None = None,
        last_embedded_id: int = 0,
        vector_count: int = 0,
        model_name: str | None = None,
        embedding_dim: int = 256,
    ) -> None:
        """Insert or replace a cache entry."""
        now = time.time()
        vt = vec_table or f"vec_messages_{group}"
        st = sep_table or f"vec_messages_sep_{group}"
        mn = model_name or model_name_for_group(group)
        conn.execute(
            "INSERT OR REPLACE INTO vector_cache_registry "
            "(tier_group, vec_table, sep_table, last_embedded_id, "
            "vector_count, model_name, embedding_dim, last_updated, created) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (group, vt, st, last_embedded_id, vector_count, mn, embedding_dim,
             now, now),
        )
        conn.commit()

    @staticmethod
    def update_progress(
        conn: sqlite3.Connection,
        group: str,
        last_embedded_id: int,
        vector_count: int,
    ) -> None:
        """Update the progress fields after a batch commit."""
        conn.execute(
            "UPDATE vector_cache_registry "
            "SET last_embedded_id = ?, vector_count = ?, last_updated = ? "
            "WHERE tier_group = ?",
            (last_embedded_id, vector_count, time.time(), group),
        )
        conn.commit()

    @staticmethod
    def delete(conn: sqlite3.Connection, group: str) -> None:
        """Remove a cache entry (used with --force)."""
        conn.execute(
            "DELETE FROM vector_cache_registry WHERE tier_group = ?",
            (group,),
        )
        conn.commit()


def get_transition_action(
    from_tier: str, to_tier: str, force: bool = False
) -> str:
    """Determine what action is needed for a tier switch.

    Returns: "noop", "config_only", "delta", or "full_rebuild"
    """
    if from_tier == to_tier and not force:
        return "noop"
    if from_tier == to_tier and force:
        return "full_rebuild"

    from_group = tier_group(from_tier)
    to_group = tier_group(to_tier)

    if from_group == to_group:
        return "config_only"

    return "delta_or_full"


def resolve_rebuild_action(
    conn: sqlite3.Connection,
    to_group: str,
    force: bool = False,
) -> str:
    """Resolve whether a rebuild is delta or full based on cache state.

    Call this after get_transition_action returns "delta_or_full".
    Returns: "delta" or "full_rebuild"
    """
    if force:
        return "full_rebuild"

    cache = VectorCacheRegistry.get(conn, to_group)
    expected_model = model_name_for_group(to_group)

    if not cache:
        return "full_rebuild"
    if cache["model_name"] and cache["model_name"] != expected_model:
        return "full_rebuild"
    if cache["last_embedded_id"] <= 0:
        return "full_rebuild"

    return "delta"


def get_messages_to_embed(
    conn: sqlite3.Connection,
    to_group: str,
    force: bool = False,
) -> tuple[list[dict], bool]:
    """Determine which messages need embedding for the target tier.

    Returns: (messages_list, is_full_rebuild)
    """
    action = resolve_rebuild_action(conn, to_group, force)

    if action == "full_rebuild":
        rows = conn.execute(
            "SELECT id, content, sender, recipient, timestamp "
            "FROM messages ORDER BY id"
        ).fetchall()
        messages = [
            dict(zip(
                ["id", "content", "sender", "recipient", "timestamp"], r
            ))
            for r in rows
        ]
        return messages, True

    cache = VectorCacheRegistry.get(conn, to_group)
    last_id = cache["last_embedded_id"] if cache else 0
    current_max = (
        conn.execute("SELECT MAX(id) FROM messages").fetchone()[0] or 0
    )

    if last_id >= current_max:
        return [], False

    rows = conn.execute(
        "SELECT id, content, sender, recipient, timestamp "
        "FROM messages WHERE id > ? ORDER BY id",
        (last_id,),
    ).fetchall()
    messages = [
        dict(zip(["id", "content", "sender", "recipient", "timestamp"], r))
        for r in rows
    ]
    return messages, False


def preflight_ram_check(target_group: str) -> tuple[bool, str]:
    """Check if the machine has enough RAM for the target tier."""
    if target_group == "edge":
        return True, ""

    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)

    if total_gb < 8:
        return False, (
            f"Base/Pro tier requires at least 8GB total RAM. "
            f"This machine has {total_gb:.1f}GB. Use Edge tier."
        )

    if available_gb < 2.0:
        return False, (
            f"Insufficient available RAM ({available_gb:.1f}GB). "
            f"Close some applications or use Edge tier."
        )

    return True, ""
