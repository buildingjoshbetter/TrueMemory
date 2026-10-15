"""truememory_configure must not hold the MCP event loop while maintenance is busy elsewhere."""

from __future__ import annotations

import json
import threading
import time

import pytest


@pytest.fixture
def server(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / ".truememory").mkdir(parents=True)
    db_path = tmp_path / "memories.db"
    monkeypatch.setenv("TRUEMEMORY_DB", str(db_path))
    monkeypatch.setenv("TRUEMEMORY_EMBED_MODEL", "edge")
    import truememory.mcp_server as ms
    monkeypatch.setattr(ms, "_TRUEMEMORY_DIR", home / ".truememory")
    monkeypatch.setattr(ms, "_CONFIG_PATH", home / ".truememory" / "config.json")
    monkeypatch.setattr(ms, "_CONFIG_LOCK_PATH", ms._CONFIG_PATH.with_name("config.json.lock"))
    monkeypatch.setattr(ms, "_DB_PATH", str(db_path))
    monkeypatch.setattr(ms, "_memory", None)
    yield ms
    ms._memory = None
    import truememory.vector_search as vs
    vs.set_embedding_model("edge")


def test_busy_maintenance_returns_pending_and_stages_in_the_background(server, monkeypatch):
    # Review v5 L-v5-6: the tool waited up to 60 s for another process's
    # maintenance pass on the event loop, so the server answered nothing.
    from truememory.maintenance import MaintenanceBusyError
    from truememory.tier_switch.manager import RebuildManager
    calls, staged = [], threading.Event()

    class Manager:
        _last_outcome = _served_tier = _last_action = None

        def start_rebuild(self, *, target_tier, db_path, config_path, config_lock_path, staging_seconds=None):
            calls.append(staging_seconds)
            if staging_seconds is not None:
                raise MaintenanceBusyError("synthetic: the short wait ran out")
            if len(calls) == 1:  # A caller that never asked for a short wait blocks for the window.
                time.sleep(3)
                raise MaintenanceBusyError("synthetic: the full window ran out")
            staged.set()
            return 7

    server._save_config({"tier": "edge"})
    monkeypatch.setattr(RebuildManager, "get_instance", staticmethod(lambda: Manager()))
    started = time.monotonic()
    result = json.loads(server.truememory_configure(tier="base"))
    elapsed = time.monotonic() - started
    assert elapsed < 1.5, elapsed
    assert result["status"] == "pending", result
    assert "starts in the background" in result["note"]
    assert staged.wait(10)
    assert calls == [server._MCP_STAGING_SECONDS, None]
