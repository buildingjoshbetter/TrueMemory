"""The MCP server waits briefly for another process's maintenance pass before opening.

A fresh open refuses at once (MaintenanceBusyError) while another process owns
maintenance. Hooks keep that fail-fast behavior; the long-lived MCP server
waits a bounded time, so a tool call no longer fails just because an ingest
worker was running a pass.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from truememory import maintenance, mcp_server
from truememory._platform import release_file_lock, try_file_lock
from truememory.maintenance import MaintenanceBusyError


class Engine:
    def __init__(self, busy: int = 0) -> None:
        self.busy = busy
        self.calls = 0
        self.conn = None

    def _ensure_connection(self) -> None:
        self.calls += 1
        if self.calls <= self.busy:
            raise MaintenanceBusyError("Maintenance is already running for this database")
        self.conn = object()


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    sleeps = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(mcp_server.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(mcp_server.time, "sleep", sleep)
    return types.SimpleNamespace(now=now, sleeps=sleeps)


@pytest.fixture
def held(tmp_path: Path, monkeypatch):
    """The maintenance lock held through another descriptor, as another process holds it."""
    database = tmp_path / "memories.db"
    database.write_bytes(b"")
    monkeypatch.setattr(mcp_server, "_DB_PATH", str(database))
    fd = try_file_lock(Path(str(maintenance.canonical_database_path(database)) + ".maintenance.lock"))
    assert fd is not None
    state = {"fd": fd}
    yield state
    if state["fd"] is not None:
        release_file_lock(state["fd"])


def test_a_tool_call_waits_out_another_process_pass_and_opens_lazily(monkeypatch, clock, held) -> None:
    engine = Engine()
    monkeypatch.setattr(mcp_server, "_memory", types.SimpleNamespace(_engine=engine))
    real_sleep = mcp_server.time.sleep

    def sleep_then_release(seconds: float) -> None:
        real_sleep(seconds)
        if len(clock.sleeps) == 3:  # The other process's pass ends.
            release_file_lock(held["fd"])
            held["fd"] = None

    monkeypatch.setattr(mcp_server.time, "sleep", sleep_then_release)
    memory = mcp_server._get_memory()
    assert memory._engine is engine
    assert len(clock.sleeps) == 3
    assert engine.calls == 0, "the store still opens inside the operation"


def test_the_wait_is_bounded(monkeypatch, clock, held) -> None:
    engine = Engine()
    monkeypatch.setattr(mcp_server, "_memory", types.SimpleNamespace(_engine=engine))
    mcp_server._get_memory()  # Returns; the operation itself reports the busy error.
    assert 0 < clock.now[0] - 1000.0 <= mcp_server._OPEN_BUSY_WAIT_SECONDS
    assert engine.calls == 0


def test_an_open_store_or_a_free_lock_never_waits(monkeypatch, clock, tmp_path) -> None:
    engine = Engine()
    engine.conn = object()
    monkeypatch.setattr(mcp_server, "_memory", types.SimpleNamespace(_engine=engine))
    mcp_server._get_memory()
    monkeypatch.setattr(mcp_server, "_DB_PATH", str(tmp_path / "memories.db"))
    monkeypatch.setattr(mcp_server, "_memory", types.SimpleNamespace(_engine=Engine()))
    mcp_server._get_memory()  # No lock file: nothing to wait for.
    assert clock.sleeps == []


def test_the_startup_open_retries_until_the_pass_ends(clock) -> None:
    engine = Engine(busy=3)
    assert mcp_server._open_waiting_for_maintenance(types.SimpleNamespace(_engine=engine), 300) is True
    assert engine.calls == 4 and engine.conn is not None


def test_the_startup_open_does_not_retry_other_failures(clock) -> None:
    calls = []

    class Failing(Engine):
        def _ensure_connection(self) -> None:
            calls.append(1)
            raise RuntimeError("synthetic open failure")

    assert mcp_server._open_waiting_for_maintenance(types.SimpleNamespace(_engine=Failing()), 300) is False
    assert calls == [1] and clock.sleeps == []
