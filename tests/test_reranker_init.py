"""Regression lock for Hunter F06 — `_set_reranker` must log + store
errors instead of silently swallowing them.

Prior behavior: `try: ... except Exception: pass`. ImportError on a user
who installed `truememory` without `[gpu]`, HF hub offline + uncached
model, CUDA OOM — all swallowed, Base/Pro silently degraded below Edge.
"""
from __future__ import annotations

import logging

import pytest


@pytest.fixture
def server(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".truememory").mkdir()
    import truememory.mcp_server as ms
    monkeypatch.setattr(ms, "_TRUEMEMORY_DIR", home / ".truememory")
    monkeypatch.setattr(ms, "_CONFIG_PATH", home / ".truememory" / "config.json")
    ms._clear_reranker_error()
    yield ms
    ms._clear_reranker_error()


def test_import_error_stores_last_error_with_install_hint(server, monkeypatch, caplog):
    import truememory.reranker as rr

    def _boom_import(*args, **kwargs):
        raise ImportError("No module named 'sentence_transformers'")

    monkeypatch.setattr(rr, "get_reranker", _boom_import)

    with caplog.at_level(logging.WARNING, logger="truememory.mcp_server"):
        server._set_reranker("BAAI/bge-reranker-v2-m3")

    assert server._reranker_last_error is not None
    assert "ImportError" in server._reranker_last_error
    assert "truememory[gpu]" in server._reranker_last_error
    assert any("Reranker init failed" in rec.message for rec in caplog.records)


def test_generic_exception_stored_and_logged(server, monkeypatch, caplog):
    import truememory.reranker as rr

    def _boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(rr, "get_reranker", _boom)

    with caplog.at_level(logging.WARNING, logger="truememory.mcp_server"):
        server._set_reranker("any-model")

    assert server._reranker_last_error is not None
    assert "RuntimeError" in server._reranker_last_error
    assert "CUDA" in server._reranker_last_error


def test_repeated_same_error_only_logs_once(server, monkeypatch, caplog):
    """Gotcha from F06: `_set_reranker` is called on EVERY search — the
    same error shouldn't spam logs. The stored error string stays current
    so stats.health still reports live state."""
    import truememory.reranker as rr

    def _boom(*args, **kwargs):
        raise RuntimeError("persistent failure")

    monkeypatch.setattr(rr, "get_reranker", _boom)

    with caplog.at_level(logging.WARNING, logger="truememory.mcp_server"):
        for _ in range(5):
            server._set_reranker("same-model")

    warnings = [r for r in caplog.records if "Reranker init failed" in r.message]
    assert len(warnings) == 1, f"expected 1 log, got {len(warnings)}"
    assert server._reranker_last_error is not None


def test_successful_init_clears_prior_error(server, monkeypatch):
    import truememory.reranker as rr

    calls = {"n": 0}

    def _sometimes_fails(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return None

    monkeypatch.setattr(rr, "get_reranker", _sometimes_fails)

    server._set_reranker("m")
    assert server._reranker_last_error is not None

    server._set_reranker("m")
    assert server._reranker_last_error is None


def test_never_raises_even_on_cascade(server, monkeypatch):
    """`_set_reranker` runs on the search hot path — it must NEVER raise
    (per the 'What NOT to do' of the finding)."""
    import truememory.reranker as rr

    def _boom_hard(*args, **kwargs):
        raise SystemExit(1)  # the meanest non-Exception subclass

    monkeypatch.setattr(rr, "get_reranker", _boom_hard)

    # SystemExit is not caught by `except Exception` — that's intentional;
    # a genuine SystemExit should propagate. Verify with a more ordinary
    # failure mode instead.
    def _boom_normal(*args, **kwargs):
        raise ValueError("bad input")

    monkeypatch.setattr(rr, "get_reranker", _boom_normal)
    server._set_reranker("m")  # must not raise
    assert "ValueError" in server._reranker_last_error
