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


def _staging_manager(outcome):
    from truememory.maintenance import MaintenanceBusyError
    done = threading.Event()

    class Manager:
        _last_outcome = _served_tier = _last_action = None

        def start_rebuild(self, *, target_tier, db_path, config_path, config_lock_path, staging_seconds=None):
            if staging_seconds is not None:
                raise MaintenanceBusyError("synthetic: the short wait ran out")
            try:
                return outcome()
            finally:
                done.set()

        def get_status(self, status_id=0):
            return {"status": "no_rebuild_found"}

    return Manager(), done


def _wait_for_staging(server, state):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        report = server._staging_report()
        if report is not None and report["state"] == state:
            return report
        time.sleep(0.01)
    raise AssertionError(server._staging_report())


@pytest.mark.parametrize("error", ["busy", "preflight"])
def test_a_failed_background_staging_is_reported_with_its_time_and_reason(server, monkeypatch, error):
    # Review v6 L-v6-2: the staging thread only logged its failure, and
    # truememory_status showed "no_rebuild_found" with no trace of the request.
    from truememory.maintenance import MaintenanceBusyError
    from truememory.tier_switch.manager import RebuildManager

    def fail():
        if error == "busy":
            raise MaintenanceBusyError("synthetic: maintenance stayed busy for the whole window")
        raise RuntimeError("synthetic: base models are not downloaded: synthetic-embedder")

    manager, done = _staging_manager(fail)
    server._save_config({"tier": "edge"})
    monkeypatch.setattr(server, "_background_staging", None)
    monkeypatch.setattr(RebuildManager, "get_instance", staticmethod(lambda: manager))
    assert json.loads(server.truememory_configure(tier="base"))["status"] == "pending"
    assert done.wait(10)
    report = _wait_for_staging(server, "failed")
    expected = ("MaintenanceBusyError: synthetic: maintenance stayed busy" if error == "busy"
                else "RuntimeError: synthetic: base models are not downloaded")
    assert report["error"].startswith(expected), report
    assert report["requested_tier"] == "base"
    assert time.strptime(report["failed_at"], "%Y-%m-%dT%H:%M:%SZ")
    status = json.loads(server.truememory_status())
    assert status["rebuild"] == {"status": "no_rebuild_found"}
    assert status["switch_staging"] == report

    # The next configure call reports it, then supersedes it.
    manager.start_rebuild = lambda **options: 9
    again = json.loads(server.truememory_configure(tier="base"))
    assert again["previous_background_staging"] == report
    assert "switch_staging" not in json.loads(server.truememory_status())


def test_a_started_background_staging_reports_its_status_id(server, monkeypatch):
    from truememory.tier_switch.manager import RebuildManager
    manager, done = _staging_manager(lambda: 7)
    server._save_config({"tier": "edge"})
    monkeypatch.setattr(server, "_background_staging", None)
    monkeypatch.setattr(RebuildManager, "get_instance", staticmethod(lambda: manager))
    assert json.loads(server.truememory_configure(tier="base"))["status"] == "pending"
    assert done.wait(10)
    report = _wait_for_staging(server, "started")
    assert (report["requested_tier"], report["status_id"]) == ("base", 7)
    assert "error" not in report
    assert json.loads(server.truememory_status())["switch_staging"] == report
