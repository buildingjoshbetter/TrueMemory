"""Regression lock for issue #425 — truememory_store empty content.

Pre-fix, `truememory_store("")` passed empty/whitespace content straight to
`Memory.add()`, which returns a skip-marker record (`{"id": null, ...}`).
`json.dumps`-ing that record handed the calling agent a success-shaped payload
with `id:null`, silently hiding the fact that nothing was stored.

Post-fix, `truememory_store` rejects empty / whitespace-only content up front
and returns an explicit `{"error": ...}` object matching the existing error
shape used elsewhere in the tool (content-too-large, metadata-too-large).
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def server(monkeypatch, tmp_path):
    """Scope the DB + ~/.truememory into tmp_path so the tool runs against an
    isolated store (mirrors tests/test_health_stats.py)."""
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
    yield ms
    monkeypatch.setattr(ms, "_memory", None)


@pytest.mark.parametrize("bad_content", [None, "", "   ", "\t", "\n", "  \t\n  "])
def test_store_empty_returns_error_object(server, bad_content):
    """None / empty / whitespace-only content must return an {"error": ...}
    object, never a success-shaped record with id:null and never an
    AttributeError from calling .strip() on None (issue #425)."""
    result = json.loads(server.truememory_store(bad_content))
    assert "error" in result, (
        f"#425 regression: content={bad_content!r} did not return an error object; got {result!r}"
    )
    # Must NOT look like a stored record.
    assert "id" not in result, (
        f"#425 regression: content={bad_content!r} returned a record-shaped payload: {result!r}"
    )


def test_store_none_content_returns_error_not_attributeerror(server):
    """content=None must be rejected up front with an {"error": ...} object.
    The declared type is `str`, but an MCP client / agent can pass null; the
    guard must check `content is None` BEFORE any .strip() call so this never
    raises AttributeError (issue #425)."""
    result = json.loads(server.truememory_store(None))
    assert "error" in result, f"#425 regression: content=None did not error; got {result!r}"
    assert "id" not in result, f"#425 regression: content=None returned record-shaped payload: {result!r}"


def test_store_empty_does_not_insert_a_row(server):
    """The rejected store must not touch the database — message_count stays 0."""
    result = json.loads(server.truememory_store(""))
    assert "error" in result
    m = server._get_memory()
    assert m.stats().get("message_count", 0) == 0


def test_store_valid_content_still_succeeds(server):
    """Non-empty content is unaffected: returns a record with a real integer id
    (guards against the fix over-rejecting)."""
    result = json.loads(server.truememory_store("Prefers Python over JavaScript"))
    assert "error" not in result, f"valid store unexpectedly errored: {result!r}"
    assert result.get("id") is not None
    assert isinstance(result["id"], int)


def test_store_whitespace_padded_real_content_is_stored(server):
    """Content that is real text with surrounding whitespace is NOT empty and
    must still be stored (mirrors the F38 'don't over-trim' contract)."""
    result = json.loads(server.truememory_store("  real fact  "))
    assert "error" not in result, f"padded real content errored: {result!r}"
    assert result.get("id") is not None
