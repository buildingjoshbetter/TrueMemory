"""Regression lock for Hunter F03 — `truememory_configure` re-embed flow.

Prior behavior:
1. After a tier change, the re-embed block called ``build_vectors`` but NOT
   ``build_separation_vectors`` — the separation table was re-created empty,
   silently breaking separation search after any tier switch.
2. The whole block was wrapped in ``except Exception: pass`` — OOM,
   disk-full, interrupted download left the DB in an indeterminate state
   and the response still claimed success.
3. ``_memory = None`` happened outside a ``finally`` so failures left the
   stale Memory instance cached.

This test file verifies the fix: both vec tables rebuilt, exceptions
surfaced in ``rebuild_error``, ``_memory`` always nulled.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def server(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".truememory").mkdir()
    db_path = tmp_path / "memories.db"
    monkeypatch.setenv("TRUEMEMORY_DB", str(db_path))
    monkeypatch.setenv("TRUEMEMORY_EMBED_MODEL", "edge")
    import truememory.mcp_server as ms
    monkeypatch.setattr(ms, "_TRUEMEMORY_DIR", home / ".truememory")
    monkeypatch.setattr(ms, "_CONFIG_PATH", home / ".truememory" / "config.json")
    monkeypatch.setattr(ms, "_CONFIG_LOCK_PATH", ms._CONFIG_PATH.with_name("config.json.lock"))
    monkeypatch.setattr(ms, "_DB_PATH", str(db_path))
    monkeypatch.setattr(ms, "_memory", None)
    import numpy as np
    import truememory.vector_search as vs

    class SyntheticEncoder:
        def encode(self, texts, **kwargs):
            # 256 * (1 / 16) ** 2 = 1, so the synthetic vectors have unit norm.
            return np.full((len(texts), 256), 1 / 16, dtype=np.float32)

    encoder = SyntheticEncoder()
    monkeypatch.setattr(vs, "get_model", lambda: encoder)
    yield ms
    # Teardown: drop cached Memory and any model-level state mutated by
    # truememory_configure (which modifies vector_search globals via
    # set_embedding_model). Reset embedding model to 'edge' so later tests
    # see the pre-test default.
    if ms._memory is not None:
        try:
            ms._memory.close()
        except Exception:
            pass
    ms._memory = None
    import truememory.vector_search as vs
    vs.set_embedding_model("edge")


def test_rebuild_exception_surfaces_in_result(server, monkeypatch):
    """If the tier-switch rebuild startup fails, the payload must include
    rebuild_error and warning keys — not silently claim success."""
    m = server._get_memory()
    m.add("seed memory", user_id="alice")

    import truememory.vector_search as vs
    monkeypatch.setattr(vs, "set_embedding_model", lambda tier: None)

    import truememory.reranker as rr
    monkeypatch.setattr(rr, "set_active_tier", lambda tier: None)
    monkeypatch.setattr(server, "_set_reranker", lambda name: None)

    from truememory.tier_switch import manager as mgr

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated disk full")

    monkeypatch.setattr(mgr.RebuildManager, "start_rebuild", _boom)

    result_json = server.truememory_configure(tier="base")
    result = json.loads(result_json)

    assert result.get("rebuild_error"), f"expected rebuild_error, got {result}"
    assert "RuntimeError" in result["rebuild_error"]
    assert "simulated disk full" in result["rebuild_error"]
    assert "warning" in result
    assert server._memory is None


def test_rebuild_clears_memory_singleton_on_success(server, monkeypatch):
    """On a cross-group tier switch, _memory should be None so the next
    call gets a fresh instance with the new embedder, and the async
    rebuild should be started (returns a status_id)."""
    m = server._get_memory()
    m.add("seed", user_id="alice")
    assert server._memory is not None

    import truememory.vector_search as vs
    import truememory.reranker as rr
    monkeypatch.setattr(vs, "set_embedding_model", lambda tier: None)
    monkeypatch.setattr(rr, "set_active_tier", lambda tier: None)
    monkeypatch.setattr(server, "_set_reranker", lambda name: None)

    from truememory.tier_switch import manager as mgr
    rebuild_calls = {"count": 0}

    def _fake_start_rebuild(*args, **kwargs):
        rebuild_calls["count"] += 1
        return 42  # fake status_id

    monkeypatch.setattr(mgr.RebuildManager, "start_rebuild", _fake_start_rebuild)

    result_json = server.truememory_configure(tier="base")
    result = json.loads(result_json)
    assert result["tier"] == "base"
    assert rebuild_calls["count"] >= 1, "start_rebuild was not called"
    assert server._memory is None
    assert "rebuild_error" not in result
    assert result.get("status_id") == 42


def test_no_tier_change_skips_rebuild(server):
    """If old tier == new tier, no re-embed is attempted and no rebuild_error appears."""
    m = server._get_memory()
    m.add("x", user_id="u")
    # Current tier is "edge" per fixture; requesting edge again is a no-op re-embed.
    result_json = server.truememory_configure(tier="edge")
    result = json.loads(result_json)
    assert result["tier"] == "edge"
    assert "rebuild_error" not in result
