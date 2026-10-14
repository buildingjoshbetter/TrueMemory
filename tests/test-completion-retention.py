"""Synthetic controls for bounded completion retention and terminal outcomes.

Each SessionEnd retains one immutable copy of the transcript. These controls
pin the retention contract: settled bytes are released (receipts stay), a
growing session supersedes its own queued prefix, short sessions are never
retained, quota refusals leave a visible backlog pointer, and outcomes that the
same bytes would reproduce are not re-extracted five times.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import types
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from truememory.ingest import completion
from truememory.ingest.hooks import stop


def _turns(count: int, pad: str = "") -> list[str]:
    lines = []
    for index in range(count):
        lines.append(json.dumps({"type": "user", "message": {"role": "user", "content": f"synthetic question {index} {pad}"}}))
        lines.append(json.dumps({"type": "assistant", "message": {"role": "assistant",
                                 "content": [{"type": "text", "text": f"synthetic answer {index}"}]}}))
    return lines


_POPEN = subprocess.Popen


def _dead_pid() -> int:
    process = _POPEN([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


@contextmanager
def _allow():
    yield True


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    projects = home / ".claude" / "projects" / "-synthetic"
    projects.mkdir(parents=True)
    backlog = home / ".truememory" / "backlog"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TRUEMEMORY_BACKLOG_DIR", str(backlog))
    monkeypatch.delenv("TRUEMEMORY_TRANSCRIPT_DIR", raising=False)
    monkeypatch.setattr(stop, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(stop, "TRACE_DIR", home / ".truememory" / "traces")
    monkeypatch.setattr(stop, "LOG_DIR", home / ".truememory" / "logs")
    from truememory.hooks import core
    from truememory.ingest.hooks import _shared
    monkeypatch.setattr(_shared, "check_extraction_budget", lambda: True)
    monkeypatch.setattr(_shared, "refund_extraction_budget", lambda: None)
    monkeypatch.setattr(_shared, "mark_session_extracted", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "spawn_gate", _allow)
    monkeypatch.setattr(core, "register_spawned_pid", lambda pid: None)
    spawned: list[list[str]] = []
    worker = types.SimpleNamespace(pid=os.getpid())

    def popen(command, **kwargs):
        spawned.append(list(command))
        return worker

    monkeypatch.setattr(completion.subprocess, "Popen", popen)
    return types.SimpleNamespace(home=home, projects=projects, backlog=backlog, spawned=spawned, worker=worker,
                                 root=home / ".truememory" / "ingest-completion-v1")


def _transcript(env, lines: list[str], session: str | None = None) -> tuple[str, Path]:
    session = session or str(uuid.uuid4())
    path = env.projects / f"{session}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return session, path


def _session_end(env, monkeypatch, session: str, path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["stop", "--host", "claude", "--event", "SessionEnd"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(
        {"session_id": session, "transcript_path": str(path), "hook_event_name": "SessionEnd"})))
    stop.main()


def _account(env) -> Path:
    return completion.account_root("", "", env.root)


def _state(env) -> dict:
    return json.loads((_account(env) / "state.json").read_text(encoding="utf-8"))["captures"]


def _captures(env) -> list[Path]:
    return sorted((_account(env) / "captures").iterdir())


def _complete(capture, attempt: int | None = None) -> None:
    """Finish the dispatched (or an inline) attempt as a successful worker would."""
    attempt = _state_of(capture)["attempts"] if attempt is None else attempt
    with completion.worker_claim(capture, attempt) as allowed:
        assert allowed
        coverage = {"parser_status": "complete", "extractor_status": "complete", "extractor_calls": 1,
                    "stable": True, "provenance_observed": True, "parser_errors": [], "extractor_errors": [],
                    "malformed_records": 0, "unrecognized_records": 0,
                    "source_version": {name: capture.metadata[name] for name in ("sha256", "byte_count")}}
        result = types.SimpleNamespace(coverage=coverage, facts_extracted=0, facts_stored=0, facts_updated=0,
                                       facts_skipped_gate=0, facts_skipped_dedup=0, trace=[])
        assert completion.record_receipt(capture, result)


def _state_of(capture) -> dict:
    return json.loads((capture.directory / "state.json").read_text(encoding="utf-8"))["captures"][capture.key]


def _capture_for(env, key: str):
    return completion._capture_from(_account(env), key, _state(env)[key])


def test_completed_sessions_release_bytes_so_the_quota_never_fills(env, monkeypatch):
    # Pre-fix: every SessionEnd copy was retained forever and the fifth 15.7 MB
    # session was dropped with no spawn and no marker once 64 MiB filled.
    lines = _turns(6, "x" * 2000)
    size = len("\n".join(lines) + "\n")
    monkeypatch.setattr(completion, "MAX_ACCOUNT_BYTES", 2 * size + 100)
    for index in range(8):
        session, path = _transcript(env, lines)
        _session_end(env, monkeypatch, session, path)
        assert len(env.spawned) == index + 1, "a session was dropped instead of admitted"
        key = env.spawned[-1][env.spawned[-1].index("--completion-key") + 1]
        _complete(_capture_for(env, key))
    assert len(list((_account(env) / "receipts").iterdir())) == 8
    # Receipted copies are released as soon as their receipt is recorded.
    assert _captures(env) == []
    assert not list(env.backlog.glob("*.json"))


def test_quota_refusal_queues_a_visible_pointer_that_is_adopted_once_room_returns(env, monkeypatch, caplog):
    lines = _turns(6, "y" * 2000)
    size = len("\n".join(lines) + "\n")
    monkeypatch.setattr(completion, "MAX_ACCOUNT_BYTES", 2 * size + 100)
    live = []
    for _ in range(2):
        session, path = _transcript(env, lines)
        _session_end(env, monkeypatch, session, path)
        live.append(env.spawned[-1][env.spawned[-1].index("--completion-key") + 1])
    refused, refused_path = _transcript(env, lines)
    with caplog.at_level("WARNING"):
        _session_end(env, monkeypatch, refused, refused_path)
    assert len(env.spawned) == 2
    assert "capture_account_quota" in caplog.text
    marker = env.backlog / f"{refused}.json"
    queued = json.loads(marker.read_text(encoding="utf-8"))
    assert queued["reason"] == "completion_backpressure:capture_account_quota"
    assert queued["transcript_host"] == "claude"

    # Still full: adoption retains the pointer instead of dropping it.
    completion.convert_legacy_pointers("", "", "claude")
    assert marker.exists() and len(env.spawned) == 2

    for key in live:
        _complete(_capture_for(env, key))
    completion.convert_legacy_pointers("", "", "claude")
    assert not marker.exists()
    assert len(env.spawned) == 3
    assert env.spawned[-1][env.spawned[-1].index("--session") + 1] == refused
    assert len(_captures(env)) == 1


def test_growing_session_supersedes_its_queued_prefix_but_not_a_rewrite(env):
    lines = _turns(6)
    session, path = _transcript(env, lines)
    first = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    path.write_text("\n".join(lines + _turns(2, "later")) + "\n", encoding="utf-8")
    second = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    state = _state(env)
    assert state[first.key]["status"] == "superseded"
    assert state[first.key]["superseded_by"] == second.key
    assert state[first.key]["released"] is True
    assert not first.path.exists()
    assert state[second.key]["status"] == "queued"
    assert completion.dispatch(first) == 0

    # A rewrite that does not begin with the retained bytes keeps both versions.
    path.write_text("\n".join(_turns(7, "rewritten")) + "\n", encoding="utf-8")
    third = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    state = _state(env)
    assert state[second.key]["status"] == "queued"
    assert second.path.exists() and third.path.exists()

    # Another session's capture is never superseded.
    other, other_path = _transcript(env, lines)
    completion.admit_capture(str(other_path), other, "", "", "claude", root=env.root)
    assert _state(env)[third.key]["status"] == "queued"


def test_claimed_prefix_is_not_superseded_while_its_worker_holds_the_claim(env):
    lines = _turns(6)
    session, path = _transcript(env, lines)
    first = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    attempt = completion.reserve_inline_attempt(first)
    with completion.worker_claim(first, attempt) as allowed:
        assert allowed
        path.write_text("\n".join(lines + _turns(1, "later")) + "\n", encoding="utf-8")
        completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
        assert _state(env)[first.key]["status"] == "running"
        assert first.path.exists()


def test_short_session_end_retains_nothing(env, monkeypatch):
    session, path = _transcript(env, _turns(2))
    _session_end(env, monkeypatch, session, path)
    assert env.spawned == []
    assert not env.root.exists() or not list(env.root.rglob("*.jsonl"))


def test_short_capture_from_another_path_settles_and_is_released(env):
    session, path = _transcript(env, _turns(2))
    short = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    assert completion.dispatch(short) == 0
    entry = _state(env)[short.key]
    assert entry["status"] == "exhausted"
    assert [item["category"] for item in entry["outcomes"]] == ["below_minimum_messages"]
    other, other_path = _transcript(env, _turns(6))
    completion.admit_capture(str(other_path), other, "", "", "claude", root=env.root)
    assert not short.path.exists()
    assert _state(env)[short.key]["released"] is True
    assert env.spawned == []


def _heuristic_attempt(capture, now: float) -> None:
    attempt = completion.reserve_inline_attempt(capture)
    assert attempt
    with completion.worker_claim(capture, attempt) as allowed:
        assert allowed
        coverage = {"parser_status": "complete", "extractor_status": "heuristic", "extractor_calls": 0,
                    "stable": True, "provenance_observed": True, "parser_errors": [],
                    "malformed_records": 0, "unrecognized_records": 0,
                    "source_version": {name: capture.metadata[name] for name in ("sha256", "byte_count")}}
        result = types.SimpleNamespace(coverage=coverage, facts_extracted=1, facts_stored=1, facts_updated=0,
                                       facts_skipped_gate=0, facts_skipped_dedup=0, trace=[{"action": "stored"}])
        assert completion.record_receipt(capture, result) is False


def test_heuristic_outcome_retries_once_after_a_pause_then_settles(env, monkeypatch):
    # Heuristic extraction reflects the worker's provider detection (Ollama
    # probe, claude on PATH, keys), which can be transient: never settle it on
    # one observation, and never retry it more than once.
    session, path = _transcript(env, _turns(6))
    capture = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    now = time.time()
    _heuristic_attempt(capture, now)
    entry = _state(env)[capture.key]
    assert entry["status"] == "retry"
    assert entry["next_attempt_at"] >= now + completion.ENVIRONMENT_RETRY_SECONDS - 1
    assert capture.path.exists(), "the copy stays for the spaced retry"
    assert completion.reserve_inline_attempt(capture) == 0
    assert completion.dispatch(capture) == 0
    later = now + completion.ENVIRONMENT_RETRY_SECONDS + 1
    monkeypatch.setattr(completion.time, "time", lambda: later)
    _heuristic_attempt(capture, later)
    entry = _state(env)[capture.key]
    assert entry["status"] == "exhausted" and entry["attempts"] == 2
    assert not capture.path.exists()
    assert completion.reserve_inline_attempt(capture) == 0
    assert completion.dispatch(capture) == 0
    assert env.spawned == []


@pytest.mark.parametrize(("coverage", "expected"), [
    ({"parser_status": "partial", "extractor_status": "complete"}, True),
    ({"parser_status": "malformed", "extractor_status": "unobserved"}, True),
    ({"parser_status": "complete", "extractor_status": "heuristic"}, False),
    ({"parser_status": "complete", "extractor_status": "limited", "extractor_errors": ["chunk_limit"]}, True),
    ({"parser_status": "unreadable", "extractor_status": "unobserved"}, False),
    ({"parser_status": "unreadable", "extractor_status": "complete"}, False),
    ({"parser_status": "complete", "extractor_status": "limited", "extractor_errors": ["fact_limit"]}, False),
    ({"parser_status": "complete", "extractor_status": "partial", "extractor_errors": ["provider_error"]}, False),
    ({"parser_status": "complete", "extractor_status": "failed"}, False),
    (None, False),
])
def test_only_outcomes_the_same_bytes_reproduce_are_terminal(coverage, expected):
    assert completion._reproducible(coverage) is expected
    assert completion._environmental(coverage) is (coverage == {"parser_status": "complete",
                                                               "extractor_status": "heuristic"})


def test_state_entries_are_bounded_and_receipts_keep_evicted_bytes_settled(env, monkeypatch):
    monkeypatch.setattr(completion, "MAX_CAPTURES", 3)
    admitted = []
    for _ in range(7):
        session, path = _transcript(env, _turns(6))
        capture = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
        _complete(capture, completion.reserve_inline_attempt(capture))
        admitted.append((session, path, capture))
    assert len(_state(env)) <= 3
    assert len(list((_account(env) / "receipts").iterdir())) == 7
    assert _captures(env) == []
    assert len([path for path in _account(env).glob("*.lock") if path.name != "state.lock"]) <= 3
    session, path, first = admitted[0]
    assert first.key not in _state(env)
    again = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    assert again.key == first.key
    assert completion.has_receipt(again)
    assert not again.path.exists(), "receipted bytes were copied again"
    assert again.key not in _state(env)
    assert completion.reserve_inline_attempt(again) == 0
    assert completion.dispatch(again) == 0


def test_lost_final_worker_is_settled_and_released(env):
    session, path = _transcript(env, _turns(6))
    capture = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    with completion._lock(capture.directory / "state.lock"):
        state = completion._state(capture.directory)
        state["captures"][capture.key].update(attempts=completion.MAX_ATTEMPTS, status="pending", worker_pid=_dead_pid())
        completion._save_state(capture.directory, state)
    other, other_path = _transcript(env, _turns(6))
    completion.admit_capture(str(other_path), other, "", "", "claude", root=env.root)
    entry = _state(env)[capture.key]
    assert entry["status"] == "exhausted"
    assert entry["outcomes"][-1]["category"] == "worker_lost"
    assert entry["released"] is True
    assert not capture.path.exists()


def test_orphan_bytes_from_an_interrupted_admission_are_removed(env):
    session, path = _transcript(env, _turns(6))
    kept = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    orphan = kept.directory / "captures" / ("a" * 64 + ".jsonl")
    fd = os.open(orphan, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, b"synthetic orphan")
    os.close(fd)
    other, other_path = _transcript(env, _turns(6))
    added = completion.admit_capture(str(other_path), other, "", "", "claude", root=env.root)
    assert not orphan.exists()
    assert kept.path.exists() and added.path.exists()


def test_a_pointer_rewritten_during_adoption_is_kept(env, monkeypatch):
    env.backlog.mkdir(parents=True, exist_ok=True)
    session, path = _transcript(env, _turns(6))
    stop._queue_to_backlog(str(path), session, "", "", reason="synthetic", transcript_host="claude")
    pointer = env.backlog / f"{session}.json"
    real_admit = completion.admit_capture

    def admit(*args, **kwargs):
        capture = real_admit(*args, **kwargs)
        # A SessionEnd refused by backpressure queues the newest bytes meanwhile.
        stop._queue_to_backlog(str(path), session, "", "", reason="newer", transcript_host="claude")
        return capture

    monkeypatch.setattr(completion, "admit_capture", admit)
    assert completion.adopt_legacy_pointer(pointer) is not None
    assert json.loads(pointer.read_text())["reason"] == "newer"
