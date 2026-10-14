"""Regression lock for the MCP backlog drainer after completion (issue #422).

The MCP drainer used to spawn ``truememory.ingest.cli ingest <live transcript>``
and leave the ``.processing`` claim for the CLI to clear. Completion removed
that clearing, so every claim was restored by ``cleanup_stale_processing``
after 30 minutes and re-ingested forever. The #422 guarantee (a worker that
exits non-zero must not drop the session) now lives in the completion store:

  1. A valid marker is admitted as an immutable capture and then retired, so
     nothing remains for the stale watcher to resurrect.
  2. The capture is dispatched through completion (``--automatic`` with a
     completion key), whose bounded attempts survive a dead worker.
  3. SessionStart conversion and the MCP drainer never both process a marker.
  4. The M-90 transcript allowlist still applies; rejected markers are
     quarantined instead of spawned.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import types
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from truememory.ingest import completion
from truememory.ingest.hooks import _shared

_POPEN = subprocess.Popen


@contextmanager
def _allow():
    yield True


def _private_write(path: Path, value: dict) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle)


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    projects = home / ".claude" / "projects" / "-synthetic"
    projects.mkdir(parents=True)
    backlog = home / ".truememory" / "backlog"
    backlog.mkdir(parents=True)
    os.chmod(home / ".truememory", 0o700)
    os.chmod(backlog, 0o700)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TRUEMEMORY_BACKLOG_DIR", str(backlog))
    monkeypatch.delenv("TRUEMEMORY_TRANSCRIPT_DIR", raising=False)
    from truememory.hooks import core
    monkeypatch.setattr(_shared, "check_extraction_budget", lambda: True)
    monkeypatch.setattr(_shared, "refund_extraction_budget", lambda: None)
    monkeypatch.setattr(core, "spawn_gate", _allow)
    monkeypatch.setattr(core, "register_spawned_pid", lambda pid: None)
    process = _POPEN([sys.executable, "-c", "pass"])
    process.wait()
    spawned: list[list[str]] = []
    # The spawned worker exits before claiming, like a crashed CLI.
    worker = types.SimpleNamespace(pid=process.pid)

    def popen(command, **kwargs):
        spawned.append(list(command))
        return worker

    monkeypatch.setattr(completion.subprocess, "Popen", popen)
    return types.SimpleNamespace(home=home, projects=projects, backlog=backlog, spawned=spawned)


def _marker(env, transcript: Path | None = None) -> tuple[str, Path, Path]:
    session = str(uuid.uuid4())
    if transcript is None:
        transcript = env.projects / f"{session}.jsonl"
        lines = []
        for index in range(6):
            lines.append(json.dumps({"type": "user", "message": {"role": "user", "content": f"synthetic question {index}"}}))
            lines.append(json.dumps({"type": "assistant", "message": {"role": "assistant",
                                     "content": [{"type": "text", "text": f"synthetic answer {index}"}]}}))
        transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    marker = env.backlog / f"{session}.json"
    _private_write(marker, {"transcript_path": str(transcript), "transcript_host": "claude", "session_id": session,
                            "user_id": "", "db_path": "", "reason": "stale_session_recovery"})
    return session, transcript, marker


def test_mcp_drain_adopts_retires_and_never_respawns_the_marker(env):
    import truememory.mcp_server as ms

    session, transcript, marker = _marker(env)
    ms._drain_batch_from_backlog([marker])

    assert not marker.exists()
    assert not marker.with_suffix(".processing").exists()
    assert len(env.spawned) == 1
    command = env.spawned[0]
    assert "--automatic" in command and "--completion-key" in command
    assert command[command.index("--transcript-host") + 1] == "claude"
    assert command[command.index("--session") + 1] == session
    capture_path = Path(command[command.index("ingest") + 1])
    assert capture_path != transcript
    assert capture_path.read_bytes() == transcript.read_bytes()

    # Pre-fix the surviving claim was restored after 30 minutes and spawned
    # again on every tick; now nothing is left to restore.
    _shared.cleanup_stale_processing(env.backlog)
    ms._drain_batch_from_backlog(sorted(env.backlog.glob("*.json")))
    assert len(env.spawned) == 1


def test_dead_worker_is_retried_by_completion_not_dropped(env):
    import truememory.mcp_server as ms

    _, _, marker = _marker(env)
    ms._drain_batch_from_backlog([marker])
    key = env.spawned[0][env.spawned[0].index("--completion-key") + 1]
    completion.drain("", "", "claude")
    assert len(env.spawned) == 2
    second = env.spawned[1]
    assert second[second.index("--completion-key") + 1] == key
    assert second[second.index("--completion-attempt") + 1] == "2"


def test_rejected_transcript_is_quarantined_not_spawned(env):
    import truememory.mcp_server as ms

    outside = env.home / "not_a_transcript.txt"
    outside.write_text("synthetic private note", encoding="utf-8")
    _, _, marker = _marker(env, outside)
    ms._drain_batch_from_backlog([marker])
    assert env.spawned == []
    assert not marker.exists()
    assert marker.with_suffix(".corrupt").exists()
    assert not list((env.home / ".truememory").rglob("captures/*"))


def test_removed_transcript_retires_its_marker_without_admission(env):
    import truememory.mcp_server as ms

    _, transcript, marker = _marker(env)
    transcript.unlink()
    ms._drain_batch_from_backlog([marker])
    assert env.spawned == []
    assert not marker.exists()


def test_conversion_and_mcp_drainer_never_both_process_a_marker(env):
    import truememory.mcp_server as ms

    _, _, marker = _marker(env)
    completion.convert_legacy_pointers("", "", "claude")
    assert not marker.exists() and len(env.spawned) == 1
    ms._drain_batch_from_backlog([marker])
    assert len(env.spawned) == 1

    _, _, contested = _marker(env)
    barrier = threading.Barrier(4)
    adopted = []

    def adopt():
        barrier.wait()
        adopted.append(completion.adopt_legacy_pointer(contested))

    threads = [threading.Thread(target=adopt) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(capture is not None for capture in adopted) == 1
    assert not contested.exists()


def test_foreign_scope_and_live_claims_are_retained(env):
    _, _, marker = _marker(env)
    assert completion.adopt_legacy_pointer(marker, ("other-account", "", "claude")) is None
    assert completion.adopt_legacy_pointer(marker, ("", "", "codex")) is None
    assert marker.exists()
    claimed = marker.with_suffix(".processing")
    marker.rename(claimed)
    value = json.loads(claimed.read_text(encoding="utf-8"))
    claimed.unlink()
    _private_write(claimed, {**value, "claimed_pid": os.getpid()})
    assert completion.adopt_legacy_pointer(claimed) is None
    assert claimed.exists()
    assert env.spawned == []
