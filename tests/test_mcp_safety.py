"""Regression locks for Hunter F22, F30, F31 — mcp_server.py concurrency /
safety cleanups.

F22: `_get_llm_fn` uses double-checked locking so concurrent first-callers
don't both build the LLM client.

F30: `_parallel_search._run_query` uses `with Memory()` context manager so
KeyboardInterrupt between construction and `try:` still closes the DB.

F31: `truememory_configure` pops HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
inside a try/finally so any raise between pop and restore doesn't leak
offline-mode-disabled state for the process lifetime.
"""
from __future__ import annotations

import json
import threading
import time


# ---------------------------------------------------------------------------
# F22 — double-checked locking on _get_llm_fn
# ---------------------------------------------------------------------------


def test_get_llm_fn_builds_exactly_once_under_concurrency(monkeypatch):
    """10 threads call `_get_llm_fn` simultaneously; `_build_llm_fn` must
    be called exactly once. Without F22's lock, the pre-fix code races
    and calls it up to 10 times."""
    import truememory.mcp_server as ms

    # Reset state so the fast path doesn't bypass the build on first call
    monkeypatch.setattr(ms, "_cached_llm_fn", None)
    monkeypatch.setattr(ms, "_cached_llm_fn_built", False)

    call_count = {"n": 0}
    build_start_barrier = threading.Event()

    def _slow_build():
        # Block until all threads are queued — forces the race window wide
        build_start_barrier.wait(timeout=5)
        call_count["n"] += 1
        time.sleep(0.01)  # widen the window further
        def _fn(prompt: str) -> str:
            return "ok"
        return _fn

    monkeypatch.setattr(ms, "_build_llm_fn", _slow_build)

    results = [None] * 10
    threads = []

    def _worker(i):
        results[i] = ms._get_llm_fn()

    for i in range(10):
        t = threading.Thread(target=_worker, args=(i,))
        threads.append(t)
        t.start()

    # All threads now queued in `_get_llm_fn`; release them into the build
    build_start_barrier.set()
    for t in threads:
        t.join(timeout=5)

    assert call_count["n"] == 1, (
        f"F22 regression: _build_llm_fn was called {call_count['n']} times "
        f"across 10 concurrent first-callers; expected exactly 1"
    )
    # All threads must see the same cached function
    assert all(r is results[0] for r in results)


def test_get_llm_fn_fast_path_skips_lock(monkeypatch):
    """Once built, subsequent calls must NOT re-acquire the lock (fast
    path). Confirm by monkeypatching the lock to raise if acquired."""
    import truememory.mcp_server as ms

    def _build():
        def _fn(prompt: str) -> str:
            return "ok"
        return _fn

    monkeypatch.setattr(ms, "_cached_llm_fn", None)
    monkeypatch.setattr(ms, "_cached_llm_fn_built", False)
    monkeypatch.setattr(ms, "_build_llm_fn", _build)

    # Prime the cache
    first = ms._get_llm_fn()
    assert first is not None

    # Now poison the lock — if the fast path works, this is never touched
    class _PoisonedLock:
        def __enter__(self):
            raise AssertionError("F22 fast-path regression: lock acquired on cached call")
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(ms, "_llm_cache_lock", _PoisonedLock())
    # Must not raise — fast path bypasses the lock
    second = ms._get_llm_fn()
    assert second is first


# ---------------------------------------------------------------------------
# F30 — _run_query uses context manager
# ---------------------------------------------------------------------------


def test_parallel_search_run_query_uses_context_manager():
    """Source-level check: the `_run_query` helper inside
    `_parallel_search` must use `with Memory(...) as ...:` (NOT
    `Memory(...); try/finally m.close()`). The pre-F30 form leaks a
    sqlite FD if KeyboardInterrupt arrives between construction and
    the `try:` — a narrow race that shows up in stress tests and on
    Ctrl-C during a long search."""
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "truememory" / "mcp_server.py"
    text = src.read_text()
    # Find the _run_query body — must contain the context-manager form
    idx = text.find("def _run_query(")
    assert idx != -1, "_run_query helper not found in mcp_server.py"
    body = text[idx : idx + 400]
    assert "with Memory(" in body, (
        "F30 regression: _run_query must use `with Memory(...)` context "
        "manager for interrupt-safe cleanup"
    )
    # And must NOT fall back to the explicit try/finally m.close() form
    assert "thread_m.close()" not in body, (
        "F30 regression: _run_query should not use explicit close() anymore"
    )


# ---------------------------------------------------------------------------
# F31 — HF offline-mode restore in try/finally
# ---------------------------------------------------------------------------


def test_configure_restores_hf_offline_on_success(monkeypatch, tmp_path):
    """Happy path: after a successful `truememory_configure`,
    HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are set to "1"."""
    import truememory.mcp_server as ms

    home = tmp_path / "home"
    home.mkdir()
    (home / ".truememory").mkdir()
    db_path = tmp_path / "memories.db"
    monkeypatch.setenv("TRUEMEMORY_DB", str(db_path))
    monkeypatch.setattr(ms, "_TRUEMEMORY_DIR", home / ".truememory")
    monkeypatch.setattr(ms, "_CONFIG_PATH", home / ".truememory" / "config.json")
    monkeypatch.setattr(ms, "_DB_PATH", str(db_path))
    monkeypatch.setattr(ms, "_memory", None)

    # Stub sentence_transformers + model-switch calls so we don't need GPU
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            import types
            return types.ModuleType("sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import truememory.vector_search as vs
    import truememory.reranker as rr
    monkeypatch.setattr(vs, "set_embedding_model", lambda tier: None)
    monkeypatch.setattr(rr, "set_active_tier", lambda tier: None)
    monkeypatch.setattr(ms, "_set_reranker", lambda name: None)

    # Clear env first
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    import os as _os
    result_json = ms.truememory_configure(tier="edge")
    _ = json.loads(result_json)
    assert _os.environ.get("HF_HUB_OFFLINE") == "1"
    assert _os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_configure_restores_hf_offline_on_set_embedding_model_raise(
    monkeypatch, tmp_path
):
    """Failure path: if `set_embedding_model` raises mid-configure, the
    finally must still restore HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE so
    subsequent searches don't silently hit the network."""
    import truememory.mcp_server as ms

    home = tmp_path / "home"
    home.mkdir()
    (home / ".truememory").mkdir()
    db_path = tmp_path / "memories.db"
    monkeypatch.setenv("TRUEMEMORY_DB", str(db_path))
    monkeypatch.setattr(ms, "_TRUEMEMORY_DIR", home / ".truememory")
    monkeypatch.setattr(ms, "_CONFIG_PATH", home / ".truememory" / "config.json")
    monkeypatch.setattr(ms, "_DB_PATH", str(db_path))
    monkeypatch.setattr(ms, "_memory", None)

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            import types
            return types.ModuleType("sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    import truememory.vector_search as vs

    def _boom(tier):
        raise ValueError("simulated: model removed")

    monkeypatch.setattr(vs, "set_embedding_model", _boom)

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    # truememory_configure wraps the tier switch; ValueError propagates
    # out per Python semantics after finally runs.
    import os as _os
    import pytest
    with pytest.raises(ValueError):
        ms.truememory_configure(tier="pro")

    # Critical F31 assertion: offline mode was RESTORED despite the raise
    assert _os.environ.get("HF_HUB_OFFLINE") == "1", (
        "F31 regression: set_embedding_model raised and offline mode was "
        "left disabled; subsequent searches will silently hit HF Hub"
    )
    assert _os.environ.get("TRANSFORMERS_OFFLINE") == "1"
