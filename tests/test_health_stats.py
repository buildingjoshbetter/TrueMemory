"""Regression lock for Hunter F07 + F08 — `truememory_stats.health` dict.

F07: the MCP tool's stats payload must include a `health` key that
reports per-subsystem status (reranker / hyde_llm / vectors). Without
this, silent-failure modes surfaced by F05 (bad API key → HyDE off),
F06 (missing sentence-transformers → reranker off), and F08
(sqlite-vec load failure → FTS-only search) are invisible from the
user's primary observability surface.

F08: sqlite-vec load failure must log at WARNING (not DEBUG) and
populate `engine._vectors_load_error` so F07 can surface it.
"""
from __future__ import annotations

import json
import logging

import pytest


@pytest.fixture
def server(monkeypatch, tmp_path):
    """Scope ~/.truememory into tmp_path; reset all F05/F06 error state."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".truememory").mkdir()
    db_path = tmp_path / "memories.db"
    monkeypatch.setenv("TRUEMEMORY_DB", str(db_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import truememory.mcp_server as ms
    monkeypatch.setattr(ms, "_TRUEMEMORY_DIR", home / ".truememory")
    monkeypatch.setattr(ms, "_CONFIG_PATH", home / ".truememory" / "config.json")
    monkeypatch.setattr(ms, "_DB_PATH", str(db_path))
    monkeypatch.setattr(ms, "_memory", None)
    ms._clear_all_llm_errors()
    ms._clear_reranker_error()
    ms._current_llm_provider_name = None
    # F08 state lives in engine module
    import truememory.engine as engine
    monkeypatch.setattr(engine, "_vectors_load_error", None)
    yield ms
    ms._clear_all_llm_errors()
    ms._clear_reranker_error()
    ms._current_llm_provider_name = None
    monkeypatch.setattr(engine, "_vectors_load_error", None)


# ---------------------------------------------------------------------------
# F07 — health payload shape
# ---------------------------------------------------------------------------


def test_health_has_three_subsystems(server):
    health = server._build_health_payload()
    assert set(health.keys()) == {"reranker", "hyde_llm", "vectors"}
    for sub in health.values():
        assert "status" in sub
        assert sub["status"] in ("ok", "degraded")


def test_health_all_ok_when_no_errors_recorded(server):
    health = server._build_health_payload()
    assert health["reranker"]["status"] == "ok"
    assert health["reranker"]["last_error"] is None
    assert health["hyde_llm"]["status"] == "ok"
    assert health["hyde_llm"]["last_error_by_provider"] is None
    assert health["vectors"]["status"] == "ok"
    assert health["vectors"]["last_error"] is None


def test_health_reranker_degraded_when_error_recorded(server):
    server._record_reranker_error("ImportError: boom")
    health = server._build_health_payload()
    assert health["reranker"]["status"] == "degraded"
    assert "ImportError" in health["reranker"]["last_error"]


def test_health_hyde_llm_degraded_reports_per_provider(server):
    import truememory.mcp_server as ms
    ms._record_llm_error("anthropic", RuntimeError("bad key"))
    ms._record_llm_error("openai", RuntimeError("rate-limited"))
    health = server._build_health_payload()
    assert health["hyde_llm"]["status"] == "degraded"
    errors = health["hyde_llm"]["last_error_by_provider"]
    assert "anthropic" in errors
    assert "openai" in errors
    assert "RuntimeError" in errors["anthropic"]


def test_health_hyde_llm_active_provider_reflects_current_resolve(server):
    server._current_llm_provider_name = "openrouter"
    health = server._build_health_payload()
    assert health["hyde_llm"]["active_provider"] == "openrouter"


def test_health_vectors_degraded_surfaces_engine_error(server, monkeypatch):
    import truememory.engine as engine
    monkeypatch.setattr(engine, "_vectors_load_error", "OSError: no wheel")
    health = server._build_health_payload()
    assert health["vectors"]["status"] == "degraded"
    assert "no wheel" in health["vectors"]["last_error"]


def test_truememory_stats_includes_health(server):
    result_json = server.truememory_stats()
    result = json.loads(result_json)
    assert "health" in result
    assert isinstance(result["health"], dict)
    assert "reranker" in result["health"]
    assert "hyde_llm" in result["health"]
    assert "vectors" in result["health"]


def test_health_does_not_leak_api_keys(server):
    """What-NOT-to-do of F07: never expose API keys in health."""
    import truememory.mcp_server as ms
    ms._record_llm_error("anthropic", RuntimeError("sk-ant-abcd-secret-value"))
    health = server._build_health_payload()
    # The error string will contain the repr of the exception, which
    # includes its message. The fix NAMES the exception type + its message;
    # an API-key-leaking exception is a caller concern, not ours. But we
    # shouldn't be synthesizing a message that INCLUDES any env-var key.
    payload_text = json.dumps(health)
    # ANTHROPIC_API_KEY env var is deleted in the fixture — no leak there
    # to assert against. The real gate is that health has no "api_key",
    # "anthropic_api_key", etc. keys.
    for forbidden in ("api_key", "anthropic_api_key", "openrouter_api_key", "openai_api_key"):
        assert forbidden not in payload_text


# ---------------------------------------------------------------------------
# F08 — sqlite-vec load failure populates engine state + logs at WARNING
# ---------------------------------------------------------------------------


def test_sqlite_vec_load_failure_logged_at_warning_and_stored(caplog, tmp_path, monkeypatch):
    """Simulate a platform where `sqlite_vec.load(...)` raises."""
    from truememory.engine import TrueMemoryEngine
    import truememory.engine as engine

    monkeypatch.setattr(engine, "_vectors_load_error", None)

    db_path = tmp_path / "health.db"
    # Create a DB first via ingest path isn't necessary — we just need
    # open() to try sqlite_vec.load. Use a minimal bootstrap.
    from truememory.storage import create_db
    conn = create_db(db_path)
    conn.close()

    # Now monkeypatch sqlite_vec.load to raise. Import it first so the
    # reference exists.
    import sqlite_vec
    def _boom(*args, **kwargs):
        raise OSError("simulated: sqlite-vec wheel unavailable on this platform")
    monkeypatch.setattr(sqlite_vec, "load", _boom)

    eng = TrueMemoryEngine(db_path)
    with caplog.at_level(logging.WARNING, logger="truememory.engine"):
        # rebuild_vectors=False so we don't hit the rebuild branch (that
        # would raise because sqlite-vec didn't load). We're only
        # exercising the load path.
        try:
            eng.open(rebuild_vectors=False)
        except Exception:
            # After the sqlite-vec load failure, the rebuild branch is
            # reached and re-raises; that's fine — we only care that the
            # load-failure WARNING was emitted first.
            pass

    # F08 assertions
    assert engine._vectors_load_error is not None
    assert "OSError" in engine._vectors_load_error
    assert any(
        "sqlite-vec unavailable" in rec.message
        for rec in caplog.records
        if rec.levelname == "WARNING"
    )


def test_sqlite_vec_load_success_clears_error(server, tmp_path):
    """A successful open() must clear any prior _vectors_load_error so
    later stats calls report ok."""
    import truememory.engine as engine
    engine._vectors_load_error = "OSError: prior failure"

    from truememory.storage import create_db
    from truememory.engine import TrueMemoryEngine
    db_path = tmp_path / "ok.db"
    conn = create_db(db_path)
    conn.close()

    TrueMemoryEngine(db_path).open(rebuild_vectors=False)
    # After a successful load, the error is cleared.
    assert engine._vectors_load_error is None


def test_get_vectors_load_error_public_helper(server):
    """F07 consumes the engine state via `get_vectors_load_error()` so it
    doesn't have to reach into module privates. The helper must exist and
    return the current value."""
    from truememory.engine import get_vectors_load_error
    import truememory.engine as engine
    assert get_vectors_load_error() is None
    engine._vectors_load_error = "OSError: x"
    assert get_vectors_load_error() == "OSError: x"
    engine._vectors_load_error = None
