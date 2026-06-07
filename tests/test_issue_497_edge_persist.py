"""Regression tests for #497 — First-run Edge tier selection never persists.

Bug:
    When a new user selects Edge tier during first-run setup via
    ``truememory_configure``, the tier is never written to ``config.json``.
    On the next MCP session ``setup_required`` is still ``True``, creating
    an infinite setup loop for Edge users.

Root cause:
    ``truememory_configure`` only called ``_save_config`` in two places:
    1. ``if api_key or email`` (line 837-838) — skipped for Edge with no key/email.
    2. Inside the ``old_tier != tier`` branch — skipped because ``old_tier``
       defaults to ``"edge"`` when no tier is persisted, so Edge->Edge is a no-op.

Fix:
    Always persist ``config["tier"] = tier`` and call ``_save_config(config)``
    unconditionally before the tier-switch logic, so every tier — including
    Edge on first run — is written to disk.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def server(monkeypatch, tmp_path):
    """Provide an isolated mcp_server module with a fresh temp config dir."""
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
    """Stub out model/reranker side effects so the test stays unit-level."""
    import truememory.vector_search as vs
    import truememory.reranker as rr

    monkeypatch.setattr(vs, "set_embedding_model", lambda tier: None)
    monkeypatch.setattr(rr, "set_active_tier", lambda tier: None)
    monkeypatch.setattr(server, "_set_reranker", lambda name: None)


def test_issue_497_edge_tier_persists(server, monkeypatch):
    """First-run Edge tier selection must be written to config.json.

    Simulates a brand-new user (no config.json exists) choosing Edge tier.
    After truememory_configure(tier="edge"), the tier MUST be persisted on
    disk so subsequent sessions see ``"tier": "edge"`` in config.json.
    """
    _no_op_model(server, monkeypatch)
    # Precondition: no config file exists (first run).
    assert not server._CONFIG_PATH.exists()

    result = json.loads(server.truememory_configure(tier="edge"))
    assert result["status"] == "configured"
    assert result["tier"] == "edge"

    # The tier must be persisted to config.json.
    assert server._CONFIG_PATH.exists(), "config.json was not created"
    persisted = json.loads(server._CONFIG_PATH.read_text(encoding="utf-8"))
    assert persisted.get("tier") == "edge", (
        f"Edge tier was not persisted to config.json: {persisted}"
    )


def test_issue_497_setup_not_repeated(server, monkeypatch):
    """After configuring Edge tier, setup_required must be False on next stats call.

    This is the user-visible symptom: without the fix, truememory_stats
    still returns ``"setup_required": true`` because ``"tier"`` is absent
    from config.json, trapping the user in an infinite setup loop.
    """
    _no_op_model(server, monkeypatch)
    # First run: configure edge tier.
    server.truememory_configure(tier="edge")

    # Simulate a fresh session: reload config from disk.
    stats = json.loads(server.truememory_stats())
    assert stats.get("setup_required") is not True, (
        "setup_required is still True after configuring Edge tier — "
        "infinite setup loop detected"
    )
    assert stats.get("tier_configured") is True, (
        "tier_configured should be True after configure()"
    )
    assert stats["tier"] == "edge"


def test_issue_497_base_pro_also_persist(server, monkeypatch):
    """Control test: Base and Pro tiers also persist on first run (no prior config)."""
    _no_op_model(server, monkeypatch)

    for tier_name in ("base", "pro"):
        # Reset: delete config file to simulate first run.
        if server._CONFIG_PATH.exists():
            server._CONFIG_PATH.unlink()

        result = json.loads(server.truememory_configure(tier=tier_name))
        assert result["status"] == "configured"
        assert result["tier"] == tier_name

        assert server._CONFIG_PATH.exists(), (
            f"config.json not created for {tier_name} tier"
        )
        persisted = json.loads(server._CONFIG_PATH.read_text(encoding="utf-8"))
        assert persisted.get("tier") == tier_name, (
            f"{tier_name} tier was not persisted to config.json: {persisted}"
        )
