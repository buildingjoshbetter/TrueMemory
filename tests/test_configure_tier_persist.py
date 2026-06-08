"""Regression lock for #399 — `truememory_configure` tier persistence.

Prior behavior:
    A Base<->Pro tier switch takes the ``config_only`` transition path (the
    two tiers share the qwen3_256 embedding space, so no re-embed is needed).
    That branch was a literal ``pass``, and the only ``_save_config`` call in
    ``truememory_configure`` runs solely when ``api_key``/``email`` is given
    and never sets ``config["tier"]``. So a pure Base<->Pro switch was never
    persisted: after a restart the resolved tier reverted, splitting the
    runtime tier from the on-disk config (and potentially the embedding model).

This test verifies the fix: a ``config_only`` switch writes the new tier to
``config.json``. The cross-group (``delta_or_full``) path intentionally still
defers tier persistence to RebuildManager (covered in test_configure_reembed),
so it is not asserted here.
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
    if ms._memory is not None:
        try:
            ms._memory.close()
        except Exception:
            pass
    ms._memory = None
    import truememory.vector_search as vs
    vs.set_embedding_model("edge")


def _no_op_model(server, monkeypatch):
    """Stub out the model/reranker side effects so the test stays unit-level."""
    import truememory.vector_search as vs
    import truememory.reranker as rr
    monkeypatch.setattr(vs, "set_embedding_model", lambda tier: None)
    monkeypatch.setattr(rr, "set_active_tier", lambda tier: None)
    monkeypatch.setattr(server, "_set_reranker", lambda name: None)


def test_config_only_switch_persists_tier(server, monkeypatch):
    """A Base->Pro (config_only) switch must persist the new tier to config.json."""
    _no_op_model(server, monkeypatch)
    # Start from a persisted Base config.
    server._save_config({"tier": "base"})

    # Force the config_only transition (Base<->Pro share embeddings).
    from truememory.tier_switch import cache as tcache
    monkeypatch.setattr(tcache, "get_transition_action", lambda old, new: "config_only")

    result = json.loads(server.truememory_configure(tier="pro"))
    assert result["tier"] == "pro"

    persisted = json.loads(server._CONFIG_PATH.read_text(encoding="utf-8"))
    assert persisted["tier"] == "pro", f"tier was not persisted to config.json: {persisted}"


def test_config_only_switch_survives_reload(server, monkeypatch):
    """After a config_only switch, _load_config() reflects the new tier
    (simulating what tier resolution reads on the next process start)."""
    _no_op_model(server, monkeypatch)
    server._save_config({"tier": "pro"})

    from truememory.tier_switch import cache as tcache
    monkeypatch.setattr(tcache, "get_transition_action", lambda old, new: "config_only")

    json.loads(server.truememory_configure(tier="base"))
    assert server._load_config().get("tier") == "base"


def test_config_only_persist_does_not_clobber_other_keys(server, monkeypatch):
    """Persisting the tier must not drop unrelated keys already in config.json
    (api keys, email, future fields). _save_config writes the full _load_config
    dict, so pre-existing keys must survive the tier write."""
    _no_op_model(server, monkeypatch)
    server._save_config({
        "tier": "base",
        "anthropic_api_key": "sk-keep-me",
        "email": "user@example.com",
        "_sentinel": "preserved",
    })

    from truememory.tier_switch import cache as tcache
    monkeypatch.setattr(tcache, "get_transition_action", lambda old, new: "config_only")

    json.loads(server.truememory_configure(tier="pro"))
    persisted = json.loads(server._CONFIG_PATH.read_text(encoding="utf-8"))
    assert persisted["tier"] == "pro"
    assert persisted["anthropic_api_key"] == "sk-keep-me"
    assert persisted["email"] == "user@example.com"
    assert persisted["_sentinel"] == "preserved"


def test_real_base_to_pro_routes_through_config_only_and_persists(server, monkeypatch):
    """Lock the routing half of the bug: a genuine Base->Pro switch (no
    monkeypatch of get_transition_action) must route through config_only
    (Base and Pro share the qwen3_256 group) and persist the tier."""
    _no_op_model(server, monkeypatch)
    server._save_config({"tier": "base"})

    # Sanity: the real transition logic classifies Base<->Pro as config_only.
    from truememory.tier_switch.cache import get_transition_action
    assert get_transition_action("base", "pro") == "config_only"

    result = json.loads(server.truememory_configure(tier="pro"))
    assert result["tier"] == "pro"
    assert result.get("note", "").lower().startswith("tier switched instantly") or \
        "instant" in result.get("note", "").lower()
    assert server._load_config().get("tier") == "pro"
