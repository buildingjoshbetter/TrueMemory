"""Tests for the #401 self-only parent-death watcher.

The MCP server normally exits when its stdio client disconnects, but an
abrupt parent death (no stdin EOF) can leave it lingering and holding a
memories.db connection. The watcher self-terminates ONLY this process when its
launching parent dies (reparent to init, ppid==1). It never signals sibling
processes, so it cannot kill a live concurrent session's MCP server. Orphaning
is detected as a transition from a non-1 initial parent to ppid==1, so a server
launched directly by init/launchd/container init is never falsely killed.
"""
from __future__ import annotations

import os
import sys
import threading

import pytest

import truememory.mcp_server as ms


@pytest.mark.skipif(sys.platform == "win32", reason="orphan/reparent is POSIX-only")
def test_is_orphaned_true_on_transition_to_init(monkeypatch):
    """Started under a real parent, now reparented to init -> orphaned."""
    monkeypatch.setattr(os, "getppid", lambda: 1)
    assert ms._is_orphaned(initial_ppid=4321) is True


@pytest.mark.skipif(sys.platform == "win32", reason="orphan/reparent is POSIX-only")
def test_is_orphaned_false_when_parent_alive(monkeypatch):
    monkeypatch.setattr(os, "getppid", lambda: 4321)
    assert ms._is_orphaned(initial_ppid=4321) is False


@pytest.mark.skipif(sys.platform == "win32", reason="orphan/reparent is POSIX-only")
def test_is_orphaned_false_when_launched_under_init(monkeypatch):
    """If we were ALREADY a child of init at startup (initial_ppid==1), a
    current ppid of 1 is NOT an orphan transition -> must not report orphaned."""
    monkeypatch.setattr(os, "getppid", lambda: 1)
    assert ms._is_orphaned(initial_ppid=1) is False


def test_watcher_is_noop_during_extraction(monkeypatch):
    """Extraction subprocesses must not start the watcher."""
    monkeypatch.setenv("TRUEMEMORY_EXTRACTION", "1")
    before = threading.active_count()
    ms._start_parent_death_watcher(poll_interval=0.01)
    assert threading.active_count() == before, "watcher thread should not start under extraction"


@pytest.mark.skipif(sys.platform == "win32", reason="watcher is a no-op on Windows")
def test_watcher_not_started_when_launched_under_init(monkeypatch):
    """Launched directly by init (ppid==1 at startup): reparent detection is
    impossible, so the watcher must not start (avoids false-positive exit)."""
    monkeypatch.delenv("TRUEMEMORY_EXTRACTION", raising=False)
    monkeypatch.setattr(os, "getppid", lambda: 1)
    before = threading.active_count()
    ms._start_parent_death_watcher(poll_interval=0.01)
    assert threading.active_count() == before, "watcher must not start when launched under init"


@pytest.mark.skipif(sys.platform == "win32", reason="watcher is a no-op on Windows")
def test_watcher_starts_thread_when_parent_alive(monkeypatch):
    """With a real (non-init) parent the watcher starts a daemon thread that
    does not exit the process while the parent stays alive."""
    monkeypatch.delenv("TRUEMEMORY_EXTRACTION", raising=False)
    monkeypatch.setattr(os, "getppid", lambda: 4321)  # non-init parent, stays alive
    before = threading.active_count()
    ms._start_parent_death_watcher(poll_interval=0.01)
    import time as _t
    _t.sleep(0.05)
    assert threading.active_count() >= before + 1
    watcher = [t for t in threading.enumerate() if t.name == "parent-death-watcher"]
    assert watcher and watcher[0].daemon
