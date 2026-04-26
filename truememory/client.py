"""
TrueMemory Client — Simple Memory API
====================================

A Mem0-compatible interface that wraps :class:`TrueMemoryEngine` for the
simplest possible developer experience::

    from truememory import Memory

    m = Memory()                          # ~/.truememory/memories.db
    m.add("Prefers dark mode", user_id="alex")
    results = m.search("preferences", user_id="alex")
    m.delete(results[0]["id"])

The ``user_id`` parameter maps to the ``sender`` field internally,
keeping the API simple while leveraging existing per-sender filtering.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from truememory.engine import TrueMemoryEngine

_DEFAULT_DB = Path.home() / ".truememory" / "memories.db"


class Memory:
    """
    High-level memory interface for AI agents.

    Args:
        path: Database file path.  Defaults to ``~/.truememory/memories.db``.
              Use ``":memory:"`` for an in-memory database (testing).
        alpha_surprise: Optional L5 surprise rerank boost coefficient.
              Multiplies each message's post-rerank score by
              ``(1 + alpha_surprise * surprise)``. Default ``None``
              resolves to the ``TRUEMEMORY_ALPHA_SURPRISE`` env var
              (or 0.3). Set explicitly to ``0`` to disable.
              See MEMORIST-L5 research for rationale.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        alpha_surprise: float | None = None,
    ):
        if path is None:
            path = _DEFAULT_DB
        db_path = Path(path) if str(path) != ":memory:" else path
        self._engine = TrueMemoryEngine(
            db_path=db_path,
            alpha_surprise=alpha_surprise,
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Store a memory.

        Args:
            content:  The text to remember.
            user_id:  Owner of this memory (optional).
            metadata: Reserved for future use.

        Returns:
            Dict with ``id``, ``content``, ``user_id``, ``created_at``.
            When ``content`` is empty / whitespace-only the memory is
            NOT stored — a warning is issued and a skip-marker is
            returned (``id`` is ``None``, ``created_at`` is ``None``).
        """
        # Hunter F38: skip empty / whitespace-only content. Callers
        # passing through user-generated text (parsed transcripts,
        # partial JSON) used to pollute the DB with useless rows that
        # inflated `stats().message_count`. Warn rather than raise so
        # batch callers can continue; the skip-marker lets them detect.
        if not content or not content.strip():
            import warnings
            warnings.warn(
                "Memory.add called with empty or whitespace-only content; "
                "skipping (no row inserted).",
                stacklevel=2,
            )
            return {
                "id": None,
                "content": content,
                "user_id": user_id or "",
                "created_at": None,
            }
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result = self._engine.add(
            content=content,
            sender=user_id or "",
            timestamp=now,
        )
        result["user_id"] = user_id or ""
        result["created_at"] = now
        return result

    def search(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search memories using the full 6-layer pipeline.

        Args:
            query:   Natural-language search string.
            user_id: Filter results to this user (optional).
            limit:   Max results.

        Returns:
            List of result dicts sorted by relevance.
        """
        results = self._engine.search(query, limit=limit * 3 if user_id else limit)

        if user_id:
            results = [r for r in results if r.get("sender", "") == user_id]

        for r in results:
            r["user_id"] = r.get("sender", "")

        return results[:limit]

    def search_deep(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
        llm_fn=None,
    ) -> list[dict]:
        """Agentic multi-round search (slower, higher accuracy).

        Args:
            query:   Natural-language search string.
            user_id: Filter results to this user (optional).
            limit:   Max results.
            llm_fn:  Callable for HyDE / query refinement (optional).

        Returns:
            List of result dicts sorted by relevance.
        """
        results = self._engine.search_agentic(
            query, limit=limit * 3 if user_id else limit, llm_fn=llm_fn,
        )

        if user_id:
            results = [r for r in results if r.get("sender", "") == user_id]

        for r in results:
            r["user_id"] = r.get("sender", "")

        return results[:limit]

    def get(self, memory_id: int) -> dict | None:
        """Retrieve a single memory by ID."""
        result = self._engine.get(memory_id)
        if result:
            result["user_id"] = result.get("sender", "")
        return result

    def get_all(
        self,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List all memories with optional user filtering."""
        results = self._engine.get_all(limit=limit, offset=offset, user_id=user_id)
        for r in results:
            r["user_id"] = r.get("sender", "")
        return results

    def update(self, memory_id: int, content: str) -> dict | None:
        """Update a memory's content.

        Returns the updated memory dict, or None if not found.
        """
        result = self._engine.update(memory_id, content=content)
        if result:
            result["user_id"] = result.get("sender", "")
        return result

    def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID."""
        return self._engine.delete(memory_id)

    def delete_all(self, user_id: str | None = None) -> bool:
        """Delete all memories, optionally filtered by user.

        Args:
            user_id: If provided, only delete this user's memories.
                     If None, deletes ALL memories.

        Returns:
            True if any rows were deleted.
        """
        return self._engine.delete_all(user_id=user_id)

    def stats(self) -> dict:
        """Return memory system statistics."""
        return self._engine.get_stats()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self):
        """Close the database connection."""
        self._engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return f"<Memory db={self._engine.db_path}>"
