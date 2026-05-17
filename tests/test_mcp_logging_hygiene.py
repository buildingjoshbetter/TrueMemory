"""Regression locks for PR-5 — mcp_server.py logging + Popen file-handle
cleanup hygiene.

Two narrow additive fixes:

1. ``_parallel_search`` logs per-query failures at DEBUG instead of
   silently swallowing them. An operator triaging "search quality
   dropped" had zero breadcrumb that any query failed; now there's a
   trace at DEBUG so the noise stays out of WARNING but the signal is
   recoverable with one log-level bump.

2. ``_drain_batch_from_backlog`` wraps the Popen call in try/finally so
   the spawn log_file handle gets closed even if Popen raises (resource
   error, fd exhaustion, ASR block, etc.). A bare ``_log_file.close()``
   AFTER Popen leaked the handle on every spawn failure.

Both changes are purely additive — happy-path behaviour is unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fix 1 — _parallel_search logs per-query failures at DEBUG
# ---------------------------------------------------------------------------


def test_parallel_search_logs_query_failures_at_debug(monkeypatch, caplog):
    """When a query raises inside the ThreadPoolExecutor batch, the
    failure should be logged at DEBUG. Pre-fix, the exception was
    swallowed with a bare ``pass`` — operators had zero trace.

    The batch must still complete and return the successful queries'
    merged results — DEBUG logging must not change control flow.
    """
    import truememory.mcp_server as ms

    # Stub _get_memory so we don't need a real DB connection.
    fake_memory = MagicMock()
    fake_memory._engine.db_path = ":memory:"
    monkeypatch.setattr(ms, "_get_memory", lambda: fake_memory)

    # Stub the per-thread Memory() context so one query raises and the
    # other returns valid results.
    class _FakeMemory:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def search_deep(self, q, **kwargs):
            if q == "raises":
                raise RuntimeError("simulated query failure")
            return [{"id": 1, "score": 0.9, "content": "ok"}]

    monkeypatch.setattr(ms, "Memory", _FakeMemory)

    with caplog.at_level(logging.DEBUG, logger="truememory.mcp_server"):
        results = ms._parallel_search(
            queries=["raises", "ok"],
            user_id="",
            internal_limit=10,
            llm_fn=None,
            output_limit=5,
        )

    # Successful query result must still come through.
    assert any(r.get("id") == 1 for r in results), (
        "Successful query results must survive a failing sibling query."
    )

    # The failure must have left a DEBUG breadcrumb.
    debug_messages = [
        rec.message for rec in caplog.records
        if rec.levelno == logging.DEBUG and "parallel search query failed" in rec.message
    ]
    assert debug_messages, (
        "Per-query failure must log at DEBUG with the 'parallel search "
        "query failed' prefix; pre-fix there was no log call at all."
    )
    assert any("simulated query failure" in m for m in debug_messages), (
        "The exception detail should make it into the log message."
    )


# ---------------------------------------------------------------------------
# Fix 2 — backlog drainer closes log file even when Popen raises
# ---------------------------------------------------------------------------


def test_backlog_drainer_closes_log_file_when_popen_raises(monkeypatch, tmp_path):
    """When `subprocess.Popen` raises inside `_drain_batch_from_backlog`,
    the spawn log_file handle must still be closed. Pre-fix, the
    `_log_file.close()` AFTER the Popen call never ran on Popen
    failure, leaking the FD on every spawn error.

    Implementation note: `_drain_batch_from_backlog` does late imports
    from `truememory.ingest.hooks._shared` and `truememory.hooks.core`.
    On Windows, the former tries `import fcntl` at module top, which
    raises `ModuleNotFoundError` until agent-A's #345 lands. We inject
    a fake `_shared` into `sys.modules` BEFORE importing `mcp_server`
    to dodge that pre-existing Windows-only bug — keeps this test
    cross-platform without waiting on #345.
    """
    import sys
    import types

    fake_shared = types.ModuleType("truememory.ingest.hooks._shared")
    fake_shared.cleanup_stale_processing = lambda d: None
    fake_shared.check_extraction_budget = lambda: True
    fake_shared.record_stale_processing_pid = lambda *a, **kw: None
    fake_shared._safe_session_id = lambda s: s
    sys.modules.setdefault("truememory.ingest.hooks._shared", fake_shared)

    import truememory.mcp_server as ms
    import truememory.hooks.core as core
    import json

    # Set HOME to tmp_path so the backlog drainer uses an isolated dir.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    backlog_dir = tmp_path / ".truememory" / "backlog"
    backlog_dir.mkdir(parents=True)

    # Create one backlog marker pointing at a fake transcript that exists.
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text("{}\n", encoding="utf-8")

    marker_path = backlog_dir / "test-session.json"
    marker_path.write_text(json.dumps({
        "transcript_path": str(transcript_path),
        "session_id": "test-session",
        "user_id": "",
        "db_path": "",
    }), encoding="utf-8")

    # Track all opened files so we can assert they get closed.
    opened_files = []
    original_open = open

    def _spy_open(path, *args, **kwargs):
        f = original_open(path, *args, **kwargs)
        opened_files.append(f)
        return f

    # Make Popen always raise to exercise the failure path.
    def _fake_popen(*args, **kwargs):
        raise OSError(24, "Too many open files (simulated)")

    # spawn_gate context manager yields `True` (slot allowed)
    class _AllowedGate:
        def __enter__(self):
            return True

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(core, "spawn_gate", lambda: _AllowedGate())
    monkeypatch.setattr(core, "register_spawned_pid", lambda pid: None)

    # Patch open + Popen at the module level used by _drain_batch_from_backlog.
    monkeypatch.setattr("builtins.open", _spy_open)
    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    # Run the drainer. It should swallow the Popen failure via the
    # outer try/except and continue, but the inner try/finally must
    # have closed the log file handle.
    ms._drain_batch_from_backlog([marker_path])

    # Restore real open before pytest tries to use it for output capture.
    monkeypatch.setattr("builtins.open", original_open)

    # At least one log file was opened (the one for our marker).
    log_files = [f for f in opened_files if str(getattr(f, "name", "")).endswith(".log")]
    assert log_files, (
        "Expected at least one .log file to be opened by the drainer "
        "for our test session."
    )

    # Every log file we opened must now be closed.
    for f in log_files:
        assert f.closed, (
            f"Log file {f.name} was left open after Popen raised — "
            f"the try/finally cleanup didn't run. Pre-fix, the bare "
            f"`_log_file.close()` after Popen leaked this handle."
        )
