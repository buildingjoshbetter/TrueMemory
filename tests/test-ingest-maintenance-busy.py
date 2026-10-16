"""Synthetic controls for background ingestion while maintenance is busy.

A fresh database open fails at once with MaintenanceBusyError while another
process owns maintenance, so hooks never block a prompt. A background worker
used to inherit that: it extracted its facts, failed on the first store, exited
with "ERROR: Maintenance is already running for this database" and left its
backlog claim behind with a dead pid. These controls hold real cross-process
maintenance ownership in a child process and run the worker against a real
database:

  1. The worker waits out busy maintenance and stores the facts it extracted
     once.
  2. When the bounded wait expires, the extraction is kept, the completion
     attempt is given back, and the backlog claim returns to the queue with an
     attempt count. The next attempt stores the kept facts without extracting.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
import types
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from truememory.ingest import completion
from truememory.ingest.hooks import _shared

_POPEN = subprocess.Popen
_HOLDER = """
import sys
from truememory.maintenance import maintenance_owner
with maintenance_owner(sys.argv[1]):
    print("held", flush=True)
    sys.stdin.readline()
"""


@contextmanager
def _allow():
    yield True


class _Holder:
    """Another process that owns maintenance for the database until released."""

    def __init__(self, database: Path) -> None:
        self.process = _POPEN([sys.executable, "-c", _HOLDER, str(database)], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True)
        assert self.process.stdout.readline().strip() == "held"

    def release(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.write("\n")
            self.process.stdin.close()
            self.process.wait(timeout=30)


def _dead_pid() -> int:
    process = _POPEN([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


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
    monkeypatch.setenv("TRUEMEMORY_GATE_ENABLED", "0")
    monkeypatch.delenv("TRUEMEMORY_TRANSCRIPT_DIR", raising=False)
    monkeypatch.setattr(_shared, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(_shared, "RECALL_CACHE_PATH", home / ".truememory" / "recall_cache.json")
    monkeypatch.setattr(_shared, "check_extraction_budget", lambda: True)
    monkeypatch.setattr(_shared, "refund_extraction_budget", lambda: None)
    monkeypatch.setattr(_shared, "mark_session_extracted", lambda *args, **kwargs: None)
    from truememory.hooks import core
    monkeypatch.setattr(core, "spawn_gate", _allow)
    monkeypatch.setattr(core, "register_spawned_pid", lambda pid: None)
    spawned: list[list[str]] = []
    worker = types.SimpleNamespace(pid=_dead_pid())
    monkeypatch.setattr(completion.subprocess, "Popen", lambda command, **kwargs: spawned.append(list(command)) or worker)

    # The provider is synthetic: extraction is counted, and dedup never calls out.
    import truememory.model_client as model_client
    from truememory.ingest import dedup, extractor, pipeline
    from truememory.ingest.models import LLMConfig, LLMError
    monkeypatch.setattr(model_client, "ensure_server_running", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "auto_detect", lambda: LLMConfig(provider="anthropic", model="synthetic",
                                                                   api_key="synthetic"))

    def unavailable(*args, **kwargs):
        raise LLMError("synthetic provider is offline")

    monkeypatch.setattr(dedup, "complete", unavailable)
    extracted: list[str] = []
    facts = ("Synthetic user keeps the release checklist in a paper notebook",
             "Synthetic user schedules deploys only on Tuesday mornings")

    def extract(conversation, config, **kwargs):
        extracted.append(conversation)
        return extractor.ExtractionOutcome([extractor.ExtractedFact(content=text) for text in facts], "complete",
                                           total_chunks=1, attempted_chunks=1, successful_chunks=1)

    monkeypatch.setattr(extractor, "extract_facts_outcome", extract)
    database = home / ".truememory" / "memories.db"
    return types.SimpleNamespace(home=home, projects=projects, backlog=backlog, spawned=spawned, database=database,
                                 extracted=extracted, facts=facts)


def _session(env) -> tuple[str, Path]:
    session = str(uuid.uuid4())
    path = env.projects / f"{session}.jsonl"
    lines = []
    for index in range(6):
        lines.append(json.dumps({"type": "user", "message": {"role": "user", "content":
                                 f"synthetic question {index} about release planning and deploy days"}}))
        lines.append(json.dumps({"type": "assistant", "message": {"role": "assistant", "content":
                                 [{"type": "text", "text": f"synthetic answer {index}"}]}}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return session, path


def _args(env, session: str, path: Path) -> argparse.Namespace:
    return argparse.Namespace(transcript=str(path), session=session, user="", db=str(env.database), threshold=0.3,
                              trace=None, provider="auto", model="", verbose=False, automatic=True,
                              completion_key="", completion_attempt=0, transcript_host="claude")


def _claim(env, session: str, path: Path, pid: int) -> Path:
    """A backlog claim as the SessionStart drainer leaves it for the worker it spawned."""
    claim = env.backlog / f"{session}.processing"
    _shared._atomic_write_text(claim, json.dumps({
        "transcript_path": str(path), "transcript_host": "claude", "session_id": session, "user_id": "",
        "db_path": str(env.database), "reason": "stale_session_recovery", "claimed_pid": pid}))
    return claim


def _stored(env) -> list[str]:
    if not env.database.exists():
        return []
    with sqlite3.connect(env.database) as conn:
        return [row[0] for row in conn.execute("SELECT content FROM messages ORDER BY id")]


def _entry(env, path: Path, session: str) -> tuple[object, dict | None]:
    capture = completion.admit_capture(str(path), session, "", str(env.database), "claude")
    state = json.loads((capture.directory / "state.json").read_text(encoding="utf-8"))["captures"]
    return capture, state.get(capture.key)


def test_busy_maintenance_waits_and_stores_the_facts_extracted_once(env, caplog):
    # Pre-fix: the worker extracted, failed the first store with
    # MaintenanceBusyError, printed "ERROR: Maintenance is already running
    # for this database" and exited 3 without storing anything.
    from truememory.ingest import cli
    session, path = _session(env)
    holder = _Holder(env.database)
    timer = threading.Timer(1.0, holder.release)
    timer.start()
    try:
        with caplog.at_level(logging.INFO):
            cli._run_ingest(_args(env, session, path))
    finally:
        timer.cancel()
        holder.release()
    assert len(env.extracted) == 1
    assert _stored(env) == list(env.facts)
    assert any("Maintenance is running for this database" in record.getMessage() for record in caplog.records)
    capture, entry = _entry(env, path, session)
    assert completion.has_receipt(capture)
    assert not (capture.directory / "extracted").exists() or not any((capture.directory / "extracted").iterdir())


def test_expired_wait_requeues_the_claim_and_the_next_attempt_reuses_the_extraction(env, monkeypatch):
    # Pre-fix: the worker exited 3, its claim stayed .processing with a dead
    # pid, and the retry paid for the whole extraction again.
    from truememory.ingest import cli, pipeline
    monkeypatch.setattr(pipeline, "MAINTENANCE_WAIT_SECONDS", 1.0, raising=False)
    monkeypatch.setattr(completion, "DEFER_SECONDS", 0, raising=False)
    session, path = _session(env)
    claim = _claim(env, session, path, os.getpid())
    holder = _Holder(env.database)
    try:
        started = time.monotonic()
        with pytest.raises(SystemExit) as exited:
            cli._run_ingest(_args(env, session, path))
        assert time.monotonic() - started < 30
    finally:
        holder.release()
    assert exited.value.code == 5
    assert len(env.extracted) == 1
    assert _stored(env) == []
    assert not claim.exists()
    queued = json.loads(claim.with_suffix(".json").read_text(encoding="utf-8"))
    assert queued["attempts"] == 1 and queued["requeued"] == "maintenance_busy" and "claimed_pid" not in queued
    capture, entry = _entry(env, path, session)
    # The attempt is given back uncounted, and no outcome is recorded for it.
    assert (entry["status"], entry["attempts"], entry["deferred"], entry["outcomes"]) == ("retry", 0, 1, [])
    assert completion.load_extraction(capture) is not None

    cli._run_ingest(_args(env, session, path))
    assert len(env.extracted) == 1, "the retry extracted the same bytes again"
    assert _stored(env) == list(env.facts)
    assert completion.has_receipt(capture)
    assert not (capture.directory / "extracted" / (capture.key + ".json")).exists()


def test_hooks_still_fail_fast_and_keep_the_extraction_on_the_error(env):
    from truememory.ingest import ingest
    from truememory.maintenance import MaintenanceBusyError
    session, path = _session(env)
    holder = _Holder(env.database)
    try:
        started = time.monotonic()
        with pytest.raises(MaintenanceBusyError) as raised:
            ingest(str(path), db_path=str(env.database), session_id=session, require_complete=True)
        assert time.monotonic() - started < 5
    finally:
        holder.release()
    assert [fact["content"] for fact in raised.value.extraction["facts"]] == list(env.facts)


def test_a_deferred_attempt_is_redispatched_under_its_number_with_a_new_log(env, monkeypatch):
    monkeypatch.setattr(completion, "DEFER_SECONDS", 0, raising=False)
    monkeypatch.setattr(completion, "MAX_DEFERRALS", 1, raising=False)
    session, path = _session(env)
    capture = completion.admit_capture(str(path), session, "", str(env.database), "claude")
    for round_ in range(2):
        assert completion.dispatch(capture)
        attempt = int(env.spawned[-1][env.spawned[-1].index("--completion-attempt") + 1])
        assert attempt == 1
        with completion.worker_claim(capture, attempt) as allowed:
            assert allowed
            # Past MAX_DEFERRALS the attempt is spent like any other.
            assert completion.defer_attempt(capture) is (round_ == 0)
    entry = json.loads((capture.directory / "state.json").read_text(encoding="utf-8"))["captures"][capture.key]
    assert entry["deferred"] == 1 and entry["attempts"] == 1
    assert [outcome["category"] for outcome in entry["outcomes"]] == ["worker_unproven"]
    assert sorted(log.name for log in (capture.directory / "logs").iterdir()) == [
        capture.key + "-1-1.log", capture.key + "-1.log"]
