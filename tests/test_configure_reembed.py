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
    monkeypatch.setattr(ms, "_DB_PATH", str(db_path))
    monkeypatch.setattr(ms, "_memory", None)
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
    """If build_vectors raises mid re-embed, the payload must include
    rebuild_error and warning keys — not silently claim success."""
    # Seed a memory so the re-embed branch actually runs
    m = server._get_memory()
    m.add("seed memory", user_id="alice")

    # Make build_vectors blow up during the tier switch
    import truememory.vector_search as vs

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated disk full")

    monkeypatch.setattr(vs, "build_vectors", _boom)

    # Switch tier (edge → base) which triggers re-embed. truememory_configure
    # imports sentence_transformers to validate the tier has its deps — stub
    # the import so this test doesn't require the [gpu] extra installed.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            # Pretend it imports fine — return a stub module
            import types
            return types.ModuleType("sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result_json = server.truememory_configure(tier="base")
    result = json.loads(result_json)

    assert result.get("rebuild_error"), f"expected rebuild_error, got {result}"
    assert "RuntimeError" in result["rebuild_error"]
    assert "simulated disk full" in result["rebuild_error"]
    assert "warning" in result
    # And _memory must be nulled regardless of failure
    assert server._memory is None


def test_rebuild_clears_memory_singleton_on_success(server, monkeypatch):
    """On a successful same-dim tier switch, _memory should be None so the
    next call gets a fresh instance with the new embedder."""
    m = server._get_memory()
    m.add("seed", user_id="alice")
    assert server._memory is not None

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            import types
            return types.ModuleType("sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Mock the model-switch calls that would require downloading models
    import truememory.vector_search as vs
    import truememory.reranker as rr
    monkeypatch.setattr(vs, "set_embedding_model", lambda tier: None)
    monkeypatch.setattr(rr, "set_active_tier", lambda tier: None)
    monkeypatch.setattr(server, "_set_reranker", lambda name: None)

    # Keep vectors rebuild stubbed to skip the model load, just note it ran
    calls = {"main": 0, "sep": 0}

    def _fake_build_vectors(conn, messages=None):
        calls["main"] += 1
        return 1

    def _fake_build_sep(conn, messages=None):
        calls["sep"] += 1
        return 1

    monkeypatch.setattr(vs, "build_vectors", _fake_build_vectors)
    monkeypatch.setattr(vs, "build_separation_vectors", _fake_build_sep)
    monkeypatch.setattr(vs, "init_vec_table", lambda conn: None)

    result_json = server.truememory_configure(tier="pro")
    result = json.loads(result_json)
    assert result["tier"] == "pro"
    # The critical F03 behavior: both vec tables rebuilt, not just the main.
    assert calls["main"] >= 1, "build_vectors was not called"
    assert calls["sep"] >= 1, "build_separation_vectors was not called — F03 regression"
    assert server._memory is None
    assert "rebuild_error" not in result


def test_no_tier_change_skips_rebuild(server):
    """If old tier == new tier, no re-embed is attempted and no rebuild_error appears."""
    m = server._get_memory()
    m.add("x", user_id="u")
    # Current tier is "edge" per fixture; requesting edge again is a no-op re-embed.
    result_json = server.truememory_configure(tier="edge")
    result = json.loads(result_json)
    assert result["tier"] == "edge"
    assert "rebuild_error" not in result
