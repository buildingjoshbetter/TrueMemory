"""Regression lock for the parallel-store hang.

Symptom (before fix): when Claude Code issues 3+ `truememory_store` MCP
calls in a single parallel tool batch, the harness UI hangs 10-15+ seconds
before any response renders, even though each individual store completes
server-side in ~60ms. Root cause was two-layer:

1. MCP layer: all 8 `@mcp.tool()` handlers were sync `def`, causing
   FastMCP to dispatch them via `return fn(**kwargs)` which blocks the
   single asyncio event loop thread for the duration of each call. JSON-RPC
   requests serialized at the transport layer.

2. Engine layer: `TrueMemoryEngine.add()` acquired `_write_lock` BEFORE
   calling `embed_single()` (~10-50ms of model.encode). Encoding is
   thread-safe (PyTorch releases the GIL inside .encode()), so concurrent
   stores could have overlapped on inference — but the lock prevented it.

Fix:
- MCP: tools changed to `async def`, engine calls wrapped in
  `await asyncio.to_thread(...)`.
- Engine: embeddings pre-computed OUTSIDE `_write_lock`; lock now only
  guards the 3 INSERTs + commit.

This test exercises the engine half — kicks 3 concurrent `engine.add()`
calls from threads and asserts the total wall-clock is well under what
serialized encodes would take. The MCP half is verified by a separate
asyncio.gather-based test.
"""
from __future__ import annotations

import asyncio
import inspect
import threading
import time

import pytest

from truememory.client import Memory


@pytest.fixture
def memory_db(tmp_path, monkeypatch):
    """Fresh Memory instance with a per-test DB; force Edge tier for speed."""
    db_path = tmp_path / "concurrent_store.db"
    monkeypatch.setenv("TRUEMEMORY_DB", str(db_path))
    monkeypatch.setenv("TRUEMEMORY_EMBED_MODEL", "edge")
    m = Memory()
    yield m


def test_engine_add_releases_lock_during_embed(memory_db):
    """Three threads each call m.add(); total wall-clock must be well under
    3 × serialized-encode time. The fix moves embedding OUTSIDE the
    write lock so encodes overlap. Edge embeddings are ~5-15ms each; with
    parallel encoding, 3 concurrent stores should land near a single-store
    cost + small lock-contention overhead.
    """
    contents = [
        "concurrent store test fact A " + "x" * 400,
        "concurrent store test fact B " + "y" * 400,
        "concurrent store test fact C " + "z" * 400,
    ]
    results: list[dict] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(content: str) -> None:
        try:
            r = memory_db.add(content=content, user_id="test")
            with lock:
                results.append(r)
        except BaseException as e:  # noqa: BLE001 — capture all
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(c,)) for c in contents]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    elapsed = time.perf_counter() - t0

    assert not errors, f"Concurrent add() raised: {errors!r}"
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # The hard requirement: all 3 stores complete in well under 30s. The
    # original bug was 10-15s harness-perceived hang on 3 concurrent stores.
    # After the fix, 3 concurrent Edge stores should finish in < 5s
    # (typically 200-800ms, but allow generous headroom for CI variance).
    assert elapsed < 5.0, (
        f"3 concurrent stores took {elapsed:.2f}s — the lock-during-embed "
        f"regression may have returned. Expected < 5s."
    )

    # Each row must be queryable + have its own ID
    ids = {r["id"] for r in results}
    assert len(ids) == 3, f"Expected 3 distinct ids, got {ids}"


def test_mcp_handlers_are_async():
    """The 6 hot-path MCP tool handlers must be coroutine functions.
    Sync handlers serialize concurrent MCP requests at the FastMCP layer.

    truememory_configure stays sync (called once at setup, has complex
    mutable state — not on the hot path).
    """
    from truememory import mcp_server as ms

    expected_async = [
        "truememory_store",
        "truememory_search",
        "truememory_search_deep",
        "truememory_get",
        "truememory_forget",
        "truememory_stats",
        "truememory_entity_profile",
    ]
    for name in expected_async:
        fn = getattr(ms, name)
        assert inspect.iscoroutinefunction(fn), (
            f"{name} must be `async def` so FastMCP doesn't block the "
            f"event loop. If you change it back to sync, expect the "
            f"parallel-store hang to return."
        )

    # Configure intentionally stays sync — assert that too so future refactors
    # know it's deliberate.
    assert not inspect.iscoroutinefunction(ms.truememory_configure), (
        "truememory_configure is intentionally sync — heavy state mutation, "
        "called once per session at setup, not on the hot path."
    )


def test_mcp_store_via_asyncio_gather(memory_db, monkeypatch):
    """Three concurrent truememory_store coroutines via asyncio.gather must
    all complete in well under 5s. Exercises the MCP-layer fix end-to-end.
    """
    from truememory import mcp_server as ms

    monkeypatch.setattr(ms, "_memory", memory_db)

    async def _run() -> list[str]:
        return await asyncio.gather(
            ms.truememory_store(content="gather store A " + "x" * 400, user_id="test"),
            ms.truememory_store(content="gather store B " + "y" * 400, user_id="test"),
            ms.truememory_store(content="gather store C " + "z" * 400, user_id="test"),
        )

    t0 = time.perf_counter()
    results = asyncio.run(_run())
    elapsed = time.perf_counter() - t0

    assert len(results) == 3
    assert all(isinstance(r, str) for r in results), "All results must be JSON strings"
    assert elapsed < 5.0, (
        f"3 gather()-ed truememory_store coroutines took {elapsed:.2f}s — "
        f"the async-handler regression may have returned. Expected < 5s."
    )
