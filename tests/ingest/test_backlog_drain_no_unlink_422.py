"""Regression locks for issue #422 — backlog drain must not unlink the
``.processing`` claim on spawn.

Pre-fix bug: ``_drain_backlog`` (and the cascade / MCP drainers) removed the
``.processing`` claim marker immediately after ``Popen`` returned — i.e. when
the ingest worker was *spawned*, not when it *succeeded*. A worker that exits
non-zero (crash, OOM, embed-model error) therefore left no claim behind, so the
stale-``.processing`` watcher (``cleanup_stale_processing``) had nothing to
recover and the session's memories were silently dropped.

Post-fix contract:
  1. The drainer leaves the ``.processing`` claim in place after spawning.
  2. The ingest CLI deletes the claim on confirmed success
     (``clear_backlog_processing``).
  3. A dead worker (non-zero exit) leaves the claim, and
     ``cleanup_stale_processing`` restores it to ``.json`` so the session is
     re-queued rather than dropped.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def _gate_allows():
    yield True


def _write_marker(backlog: Path, session_id: str, transcript: Path) -> Path:
    backlog.mkdir(parents=True, exist_ok=True)
    marker = backlog / f"{session_id}.json"
    marker.write_text(
        json.dumps(
            {
                "transcript_path": str(transcript),
                "session_id": session_id,
                "user_id": "",
                "db_path": "",
            }
        ),
        encoding="utf-8",
    )
    return marker


def test_drain_leaves_processing_claim_after_spawn(monkeypatch, tmp_path):
    """After spawning a worker, the drainer must NOT unlink the .processing
    claim. Pre-fix this assertion fails because the claim was unlinked on spawn.
    """
    from truememory.ingest.hooks import session_start as ss
    from truememory.hooks import core as core_mod

    backlog = tmp_path / "backlog"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("x" * 100, encoding="utf-8")
    _write_marker(backlog, "sess-422-a", transcript)

    monkeypatch.setattr(ss, "BACKLOG_DIR", backlog)

    # Force budget + spawn gate to allow exactly one spawn.
    from truememory.ingest.hooks import _shared as shared_mod
    monkeypatch.setattr(shared_mod, "check_extraction_budget", lambda: True)
    monkeypatch.setattr(core_mod, "spawn_gate", _gate_allows)
    monkeypatch.setattr(core_mod, "register_spawned_pid", lambda pid: None)

    class _DummyProc:
        pid = os.getpid()  # alive PID so stale-cleanup leaves it be

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _DummyProc())

    ss._drain_backlog()

    claim = backlog / "sess-422-a.processing"
    json_marker = backlog / "sess-422-a.json"
    # The claim must still exist (worker hasn't confirmed success yet).
    assert claim.exists(), "drainer unlinked .processing on spawn (issue #422)"
    # And it must not have reverted to .json while the worker is alive.
    assert not json_marker.exists()


def test_clear_backlog_processing_removes_claim_on_success(tmp_path, monkeypatch):
    """The success helper removes the .processing claim for a session."""
    from truememory.ingest.hooks import _shared as shared_mod

    backlog = tmp_path / "backlog"
    backlog.mkdir(parents=True)
    claim = backlog / "sess-success.processing"
    claim.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(shared_mod, "BACKLOG_DIR", backlog)

    assert shared_mod.clear_backlog_processing("sess-success") is True
    assert not claim.exists()
    # Idempotent / safe when there's nothing to remove.
    assert shared_mod.clear_backlog_processing("sess-success") is False
    assert shared_mod.clear_backlog_processing("") is False
    assert shared_mod.clear_backlog_processing("unknown") is False


def test_crashed_worker_is_requeued_not_dropped(monkeypatch, tmp_path):
    """End-to-end recovery: a spawned worker that exits non-zero leaves the
    .processing claim, and cleanup_stale_processing restores it to .json so the
    session is re-queued.

    Pre-fix the claim would already be gone (unlinked on spawn), so this
    recovery is impossible and the session is silently lost.
    """
    from truememory.ingest.hooks import session_start as ss
    from truememory.ingest.hooks import _shared as shared_mod
    from truememory.hooks import core as core_mod

    backlog = tmp_path / "backlog"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("x" * 100, encoding="utf-8")
    _write_marker(backlog, "sess-crash", transcript)

    monkeypatch.setattr(ss, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(shared_mod, "check_extraction_budget", lambda: True)
    monkeypatch.setattr(core_mod, "spawn_gate", _gate_allows)
    monkeypatch.setattr(core_mod, "register_spawned_pid", lambda pid: None)

    # Simulate a worker that has already died (non-zero exit). Use a PID that
    # is guaranteed dead so the liveness check in cleanup treats it as crashed.
    dead_pid = _find_dead_pid()

    class _DeadProc:
        pid = dead_pid

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _DeadProc())

    ss._drain_backlog()

    claim = backlog / "sess-crash.processing"
    assert claim.exists(), "claim must survive spawn so the crash is recoverable"

    # Age the claim past the 30-minute stale threshold so the watcher acts.
    old = time.time() - (shared_mod._STALE_PROCESSING_THRESHOLD + 60)
    os.utime(claim, (old, old))

    # The worker is dead and the claim is stale → it must be restored to .json.
    shared_mod.cleanup_stale_processing(backlog)

    json_marker = backlog / "sess-crash.json"
    assert json_marker.exists(), "crashed worker's session must be re-queued (issue #422)"
    assert not claim.exists()


def _find_dead_pid() -> int:
    """Return a PID that is not currently alive."""
    for candidate in range(999999, 990000, -1):
        try:
            os.kill(candidate, 0)
        except OSError:
            return candidate
    return 999999
