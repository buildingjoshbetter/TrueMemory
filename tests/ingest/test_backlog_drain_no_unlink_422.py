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
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _allow_tmp_transcripts(tmp_path, monkeypatch):
    # BLAST OFF v3 (A1-2): the stop hook + backlog drain now require
    # transcript_path to resolve inside an allowed transcripts root. These
    # tests stage transcripts under tmp_path, so point the allowlist there.
    monkeypatch.setenv("TRUEMEMORY_TRANSCRIPT_DIR", str(tmp_path))


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


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill(pid, 0) liveness check is Unix-only")
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


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX os.kill EPERM/ESRCH semantics; win32 uses psutil via _platform.pid_is_alive",
)
def test_pid_alive_treats_eperm_as_alive_esrch_as_dead(monkeypatch):
    """The liveness check must not treat a live-but-EPERM process as dead.

    cleanup_stale_processing relies on _pid_is_alive to decide whether a
    crashed worker's claim can be reclaimed. os.kill(pid, 0) raises
    PermissionError (EPERM) for a process owned by another user that is very
    much alive — that must read as alive. Only ProcessLookupError / ESRCH
    means the process is genuinely gone.
    """
    import errno as _errno

    from truememory.ingest.hooks import _shared as shared_mod

    def _raise_eperm(pid, sig):
        raise PermissionError(_errno.EPERM, "Operation not permitted")

    def _raise_esrch(pid, sig):
        raise ProcessLookupError(_errno.ESRCH, "No such process")

    def _raise_bare_eperm_oserror(pid, sig):
        raise OSError(_errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(shared_mod.os, "kill", _raise_eperm)
    assert shared_mod._pid_is_alive(4242) is True, "EPERM process must be alive"

    monkeypatch.setattr(shared_mod.os, "kill", _raise_esrch)
    assert shared_mod._pid_is_alive(4242) is False, "ESRCH process must be dead"

    monkeypatch.setattr(shared_mod.os, "kill", _raise_bare_eperm_oserror)
    assert shared_mod._pid_is_alive(4242) is True, "bare OSError(EPERM) must be alive"


def test_sanitized_session_id_claim_filename_roundtrips(monkeypatch, tmp_path):
    """A session_id that requires sanitization (contains '/' and ':') must
    round-trip: the ``.processing`` claim filename the drainer writes (derived
    from the ``.json`` marker stem produced by stop._queue_to_backlog) must be
    exactly the filename clear_backlog_processing reconstructs from the raw
    session_id. Otherwise the CLI's success cleanup would target the wrong path
    and the claim would never be removed.
    """
    from truememory.ingest.hooks import session_start as ss
    from truememory.ingest.hooks import _shared as shared_mod
    from truememory.ingest.hooks import stop as stop_mod
    from truememory.hooks import core as core_mod

    raw_session_id = "proj/foo:bar baz-1"  # has '/', ':', and a space
    safe_id = shared_mod._safe_session_id(raw_session_id)
    # Sanitization actually changes the id (so this is a meaningful case).
    assert safe_id != raw_session_id
    assert "/" not in safe_id and ":" not in safe_id

    backlog = tmp_path / "backlog"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("x" * 100, encoding="utf-8")

    # The drainer/stop and the CLI cleanup must agree on the backlog dir.
    monkeypatch.setattr(ss, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(shared_mod, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(stop_mod, "BACKLOG_DIR", backlog)

    # Queue exactly the way the real Stop hook does, so the .json marker stem is
    # produced by the production naming path (stop._sanitize_session_id).
    stop_mod._queue_to_backlog(
        str(transcript), raw_session_id, "", "", reason="test",
    )
    json_marker = backlog / f"{safe_id}.json"
    assert json_marker.exists(), "queue_to_backlog must name the marker by sanitized id"

    # Allow exactly one spawn with an alive PID so the claim is left in place.
    monkeypatch.setattr(shared_mod, "check_extraction_budget", lambda: True)
    monkeypatch.setattr(core_mod, "spawn_gate", _gate_allows)
    monkeypatch.setattr(core_mod, "register_spawned_pid", lambda pid: None)

    class _DummyProc:
        pid = os.getpid()

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _DummyProc())

    ss._drain_backlog()

    # The drainer renamed .json -> .processing using the SAME sanitized stem.
    claim = backlog / f"{safe_id}.processing"
    assert claim.exists(), "drainer must write claim named by sanitized session id"

    # The CLI reconstructs the claim path purely from the raw session_id. It
    # must resolve to exactly the file the drainer wrote.
    assert shared_mod.clear_backlog_processing(raw_session_id, backlog) is True
    assert not claim.exists(), (
        "clear_backlog_processing must remove the claim the drainer wrote "
        "for a session_id that required sanitization"
    )


def test_cli_ingest_session_argparse_dest_is_session():
    """The CLI success path calls ``clear_backlog_processing(args.session)``.

    That only works if argparse stores ``--session`` under the dest
    ``session`` (the default for a ``--session`` flag). If the dest were ever
    renamed to ``session_id`` the success cleanup would raise
    ``AttributeError`` and the .processing claim would be left to the stale
    watcher, wasting a re-extraction of an already-ingested session. This test
    locks the attribute name end-to-end: it builds the real parser and checks
    that parsing ``--session`` populates ``args.session`` with the *raw*
    (unsanitized) value, which is exactly what gets passed back through to
    ``clear_backlog_processing``.
    """
    import argparse

    from truememory.ingest import cli as cli_mod

    raw_session_id = "proj/foo:bar baz-1"

    # Drive the actual parser construction the CLI uses, so a future rename of
    # the argparse dest is caught here rather than only at runtime.
    parser = argparse.ArgumentParser(prog="truememory-ingest")
    sub = parser.add_subparsers(dest="command")
    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("transcript")
    p_ingest.add_argument("--session", default="")
    args = parser.parse_args(["ingest", "t.jsonl", "--session", raw_session_id])

    assert hasattr(args, "session"), "argparse dest for --session must be 'session'"
    assert not hasattr(args, "session_id"), "no 'session_id' dest should exist"
    # The raw value must survive unchanged to clear_backlog_processing.
    assert args.session == raw_session_id

    # And the CLI source must actually read args.session (not args.session_id)
    # in the success-path cleanup that calls clear_backlog_processing.
    src = Path(cli_mod.__file__).read_text(encoding="utf-8")
    assert "clear_backlog_processing(args.session)" in src
    assert "args.session_id" not in src
