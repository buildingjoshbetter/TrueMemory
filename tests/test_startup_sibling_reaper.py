"""Tests for the #423 startup sibling reaper (CONSERVATIVE, guarded).

The self-only #401 watcher exits only the current process when its own parent
dies; it never reaps OTHER ("sibling") MCP servers from prior sessions that
keep a memories.db connection open and recreate `database is locked`
contention. The #423 reaper terminates ONLY sibling servers that are provably
abandoned: a different process, bound to the same memories.db, whose parent is
dead (reparented to init, ppid==1). A sibling with a living parent is a live
concurrent session and must never be touched.

These tests exercise the PURE selection predicate over mocked process lists and
the opt-in/guard behaviour of the reaper. Nothing is ever actually killed:
os.kill is monkeypatched to a recorder.
"""
from __future__ import annotations

import os
import sys

import pytest

import truememory.mcp_server as ms


DB = "/home/u/.truememory/memories.db"
OTHER_DB = "/home/u/.truememory/other.db"


def _proc(pid, ppid, *, name="TrueMemory MCP", cmdline=None, db=DB):
    return {
        "pid": pid,
        "ppid": ppid,
        "name": name,
        "cmdline": cmdline if cmdline is not None else ["python", "-m", "truememory.mcp_server"],
        "db": db,
    }


# --- pure selection predicate -------------------------------------------------

def test_selects_orphaned_sibling_same_db():
    """A different MCP process, same DB, parent dead (ppid==1) -> reaped."""
    procs = [_proc(2222, ppid=1)]
    assert ms._select_orphaned_siblings(procs, self_pid=1000, self_db=DB) == [2222]


def test_never_selects_self():
    """Even orphaned + same DB, the current process is never selected."""
    procs = [_proc(1000, ppid=1)]
    assert ms._select_orphaned_siblings(procs, self_pid=1000, self_db=DB) == []


def test_never_selects_live_sibling():
    """A sibling whose parent is ALIVE (ppid != 1) is a live session -> skip.

    This is the core safety guarantee: live concurrent sessions are untouched.
    """
    procs = [_proc(2222, ppid=4321)]  # real, living parent
    assert ms._select_orphaned_siblings(procs, self_pid=1000, self_db=DB) == []


def test_never_selects_different_db():
    """Orphaned MCP server bound to a DIFFERENT db is a different lock domain."""
    procs = [_proc(2222, ppid=1, db=OTHER_DB)]
    assert ms._select_orphaned_siblings(procs, self_pid=1000, self_db=DB) == []


def test_never_selects_unknown_db():
    """If a candidate's DB couldn't be resolved (""), it is not selected."""
    procs = [_proc(2222, ppid=1, db="")]
    assert ms._select_orphaned_siblings(procs, self_pid=1000, self_db=DB) == []


def test_never_selects_non_mcp_process():
    """A non-TrueMemory process is ignored even if orphaned w/ same db field."""
    procs = [_proc(2222, ppid=1, name="bash", cmdline=["bash"])]
    assert ms._select_orphaned_siblings(procs, self_pid=1000, self_db=DB) == []


def test_db_paths_compared_after_realpath(tmp_path):
    """Equivalent paths (symlink/.. forms) resolve to the same DB and match."""
    real = tmp_path / "memories.db"
    real.write_text("")
    weird = str(tmp_path / "sub" / ".." / "memories.db")
    procs = [_proc(2222, ppid=1, db=weird)]
    out = ms._select_orphaned_siblings(procs, self_pid=1000, self_db=str(real))
    assert out == [2222]


def test_mixed_list_selects_only_orphans():
    procs = [
        _proc(1000, ppid=1),               # self -> skip
        _proc(2222, ppid=1),               # orphan, same db -> reap
        _proc(3333, ppid=9999),            # live sibling -> skip
        _proc(4444, ppid=1, db=OTHER_DB),  # orphan, other db -> skip
        _proc(5555, ppid=1, name="bash", cmdline=["bash"]),  # non-mcp -> skip
    ]
    assert ms._select_orphaned_siblings(procs, self_pid=1000, self_db=DB) == [2222]


# --- _proc_is_truememory_mcp predicate ---------------------------------------

@pytest.mark.parametrize("name,cmdline,expected", [
    ("TrueMemory MCP", [], True),
    ("python3.12", ["python", "-m", "truememory.mcp_server"], True),
    ("truememory-mcp", [], True),
    ("bash", ["bash", "-l"], False),
    ("python3.12", ["python", "-m", "some.other.module"], False),
    ("", None, False),
])
def test_proc_is_truememory_mcp(name, cmdline, expected):
    assert ms._proc_is_truememory_mcp(name, cmdline) is expected


# --- guard / opt-in behaviour of the reaper ----------------------------------

def test_reaper_noop_without_optin(monkeypatch):
    """Default-safe: with TRUEMEMORY_REAP_SIBLINGS unset, reaper does nothing.

    We assert it never even enumerates processes by making _collect_mcp_procs
    raise if called.
    """
    monkeypatch.delenv("TRUEMEMORY_REAP_SIBLINGS", raising=False)
    monkeypatch.delenv("TRUEMEMORY_EXTRACTION", raising=False)
    monkeypatch.setattr(ms, "_collect_mcp_procs", lambda: (_ for _ in ()).throw(AssertionError("should not enumerate")))
    assert ms._reap_orphaned_siblings(db_path=DB) == []


def test_reaper_noop_under_extraction(monkeypatch):
    monkeypatch.setenv("TRUEMEMORY_REAP_SIBLINGS", "1")
    monkeypatch.setenv("TRUEMEMORY_EXTRACTION", "1")
    monkeypatch.setattr(ms, "_collect_mcp_procs", lambda: (_ for _ in ()).throw(AssertionError("should not enumerate")))
    assert ms._reap_orphaned_siblings(db_path=DB) == []


@pytest.mark.skipif(sys.platform == "win32", reason="reaper is a no-op on Windows")
def test_reaper_kills_only_orphan_when_optin(monkeypatch):
    """End-to-end (mocked): opt-in, signals ONLY the orphaned sibling pid."""
    monkeypatch.setenv("TRUEMEMORY_REAP_SIBLINGS", "1")
    monkeypatch.delenv("TRUEMEMORY_EXTRACTION", raising=False)
    monkeypatch.setattr(ms.os, "getpid", lambda: 1000)
    procs = [
        _proc(1000, ppid=1),     # self -> never
        _proc(2222, ppid=1),     # orphan -> reap
        _proc(3333, ppid=4321),  # live sibling -> never
    ]
    monkeypatch.setattr(ms, "_collect_mcp_procs", lambda: procs)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(ms.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    reaped = ms._reap_orphaned_siblings(db_path=DB)
    assert reaped == [2222]
    assert killed == [(2222, ms.signal.SIGTERM)]


@pytest.mark.skipif(sys.platform == "win32", reason="reaper is a no-op on Windows")
def test_reaper_tolerates_already_gone(monkeypatch):
    """A target that exits between selection and kill is skipped, not raised."""
    monkeypatch.setenv("TRUEMEMORY_REAP_SIBLINGS", "1")
    monkeypatch.delenv("TRUEMEMORY_EXTRACTION", raising=False)
    monkeypatch.setattr(ms.os, "getpid", lambda: 1000)
    monkeypatch.setattr(ms, "_collect_mcp_procs", lambda: [_proc(2222, ppid=1)])

    def _boom(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(ms.os, "kill", _boom)
    assert ms._reap_orphaned_siblings(db_path=DB) == []
