"""Synthetic controls for large transcripts in completion and parsing.

Sources above the copy limit are referenced as a hash-verified prefix instead
of being dropped or copied; the hook and the worker parse in bounded chunks.
The copy limit is patched down so these controls stay small.
"""
from __future__ import annotations

import hashlib
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

from truememory.ingest import completion, transcript
from truememory.ingest.hooks import stop

_POPEN = subprocess.Popen


def _records(turns: int, payload: int = 0, *, extras: bool = False) -> list[dict]:
    records = []
    for index in range(turns):
        records.append({"type": "user", "timestamp": f"2026-10-14T00:{index % 60:02d}:00Z",
                        "message": {"role": "user", "content": f"synthetic question {index}"}})
        records.append({"type": "assistant", "timestamp": f"2026-10-14T00:{index % 60:02d}:01Z",
                        "message": {"role": "assistant", "content": [
                            {"type": "thinking", "thinking": "private"},
                            {"type": "text", "text": f"<truememory-recall>echo</truememory-recall>answer {index} " + "a" * 700},
                            {"type": "tool_use", "id": f"t{index}", "name": "Read", "input": {"path": "x"}}]}})
        records.append({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{index}", "content": "r" * payload or "ok"}]}})
        if extras:
            records.append({"type": "attachment", "attachment": {"type": "hook_success"}})
            records.append({"type": "queue-operation", "operation": "enqueue", "content": f"synthetic question {index}"})
            records.append({"type": "queue-operation", "operation": "enqueue", "content": f"withdrawn {index}"})
            records.append({"type": "summary", "summary": "synthetic"})
            records.append({"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "look"}, {"type": "image", "source": {"type": "base64", "data": "AA=="}}]}})
    return records


def _write(path: Path, records: list[dict]) -> bytes:
    data = ("\n".join(json.dumps(record) for record in records) + "\n").encode("utf-8")
    path.write_bytes(data)
    return data


@pytest.mark.parametrize("extras", [False, True])
def test_compact_stream_renders_extraction_identically(tmp_path, extras):
    path = tmp_path / "session.jsonl"
    _write(path, _records(12, payload=3000, extras=extras) + [None] + ["text"])
    path.write_bytes(path.read_bytes() + b"{broken\n")
    full = transcript.parse_transcript_outcome(path)
    compact = transcript.parse_transcript_outcome(path, compact=True)
    assert (compact.status, compact.total_records, compact.malformed_records, compact.unrecognized_records,
            compact.error_categories) == (full.status, full.total_records, full.malformed_records,
                                         full.unrecognized_records, full.error_categories)
    assert [(m.role, m.timestamp) for m in compact.messages] == [(m.role, m.timestamp) for m in full.messages]
    assert transcript.format_for_extraction(compact.messages) == transcript.format_for_extraction(full.messages)
    assert [transcript.format_for_extraction([m]) for m in compact.messages] == \
        [transcript.format_for_extraction([m]) for m in full.messages]
    assert compact.file_version.sha256 == full.file_version.sha256
    assert all(not m.content for m in compact.messages if m.role == "tool_result")


def test_stream_offsets_and_line_splitting_cover_chunk_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "_READ_CHUNK", 7)
    monkeypatch.setattr(completion, "_CHUNK", 7)
    path = tmp_path / "session.jsonl"
    data = _write(path, _records(5, payload=40))
    outcome = transcript.parse_transcript_outcome(path, compact=True)
    assert outcome.complete and outcome.total_records == 15
    fd = os.open(path, os.O_RDONLY)
    try:
        offsets = {0, 1, 6, 7, 8, 50, len(data)}
        total, digest, prefixes = completion._stream(fd, len(data), offsets=offsets)
    finally:
        os.close(fd)
    assert (total, digest) == (len(data), hashlib.sha256(data).hexdigest())
    assert prefixes == {offset: hashlib.sha256(data[:offset]).hexdigest() for offset in offsets}


def test_human_count_stops_reading_once_the_threshold_is_met(tmp_path, monkeypatch):
    path = tmp_path / "session.jsonl"
    _write(path, _records(6) + _records(40, payload=200_000))
    reads = []
    real_open = Path.open

    class Counted:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def read(self, size=-1):
            data = self.handle.read(size)
            reads.append(len(data))
            return data

    with monkeypatch.context() as patched:
        patched.setattr(Path, "open", lambda self, *args, **kwargs: Counted(real_open(self, *args, **kwargs)))
        assert transcript.has_human_messages(path, 5)
    assert sum(reads) <= transcript._READ_CHUNK < path.stat().st_size
    assert not transcript.has_human_messages(path, 47)
    assert transcript.has_human_messages(path, 46)


def test_prefix_parse_of_a_growing_source_is_stable_and_proven(tmp_path):
    path = tmp_path / "session.jsonl"
    data = _write(path, _records(6))
    with path.open("ab") as handle:
        handle.write(b'{"type": "user", "message": {"role": "user", "content": "later"}}\n')
    outcome = transcript.parse_transcript_outcome(path, compact=True, byte_limit=len(data))
    assert outcome.complete
    assert outcome.file_version.stable
    assert outcome.file_version.sha256 == hashlib.sha256(data).hexdigest()
    assert all(m.content != "later" for m in outcome.messages)
    unlimited = transcript.parse_transcript_outcome(path, compact=True)
    assert unlimited.file_version.byte_count > len(data)


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
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.home() on Windows
    monkeypatch.setenv("TRUEMEMORY_BACKLOG_DIR", str(backlog))
    monkeypatch.delenv("TRUEMEMORY_TRANSCRIPT_DIR", raising=False)
    monkeypatch.setattr(stop, "BACKLOG_DIR", backlog)
    monkeypatch.setattr(stop, "TRACE_DIR", home / ".truememory" / "traces")
    monkeypatch.setattr(stop, "LOG_DIR", home / ".truememory" / "logs")
    monkeypatch.setattr(completion, "MAX_CAPTURE_BYTES", 4096)
    from truememory.hooks import core
    from truememory.ingest.hooks import _shared
    monkeypatch.setattr(_shared, "check_extraction_budget", lambda: True)
    monkeypatch.setattr(_shared, "refund_extraction_budget", lambda: None)
    monkeypatch.setattr(_shared, "mark_session_extracted", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "spawn_gate", _allow)
    monkeypatch.setattr(core, "register_spawned_pid", lambda pid: None)
    spawned: list[list[str]] = []
    monkeypatch.setattr(completion.subprocess, "Popen",
                        lambda command, **kwargs: spawned.append(list(command)) or types.SimpleNamespace(pid=os.getpid()))
    return types.SimpleNamespace(home=home, projects=projects, spawned=spawned,
                                 root=home / ".truememory" / "ingest-completion-v1")


def _session(env, records: list[dict]) -> tuple[str, Path, bytes]:
    session = str(uuid.uuid4())
    path = env.projects / f"{session}.jsonl"
    return session, path, _write(path, records)


def _end(env, monkeypatch, session: str, path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["stop", "--host", "claude", "--event", "SessionEnd"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"session_id": session, "transcript_path": str(path)})))
    stop.main()


def _state(capture) -> dict:
    return json.loads((capture.directory / "state.json").read_text(encoding="utf-8"))["captures"]


def test_large_session_end_is_referenced_not_dropped_or_copied(env, monkeypatch):
    session, path, data = _session(env, _records(8, payload=2000))
    assert len(data) > completion.MAX_CAPTURE_BYTES
    _end(env, monkeypatch, session, path)
    assert len(env.spawned) == 1
    command = env.spawned[0]
    assert Path(command[command.index("ingest") + 1]) == path.resolve()
    key = command[command.index("--completion-key") + 1]
    account = completion.account_root("", "", env.root)
    entry = json.loads((account / "state.json").read_text(encoding="utf-8"))["captures"][key]
    assert entry["metadata"]["storage"] == "source_prefix"
    assert entry["metadata"]["byte_count"] == len(data)
    assert list((account / "captures").iterdir()) == []
    assert list((account / "incoming").iterdir()) == []


def test_worker_proves_the_prefix_while_the_source_keeps_growing(env):
    session, path, data = _session(env, _records(8, payload=2000))
    capture = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    with path.open("ab") as handle:
        handle.write(b'{"type": "user", "message": {"role": "user", "content": "appended later"}}\n')
    loaded = completion.load_capture(capture.key, "", "", "claude", root=env.root)
    attempt = completion.reserve_inline_attempt(loaded)
    with completion.worker_claim(loaded, attempt) as allowed:
        assert allowed
        parsed = transcript.parse_transcript_outcome(loaded.path, compact=True,
                                                     byte_limit=loaded.metadata["byte_count"])
        coverage = {"parser_status": parsed.status, "extractor_status": "complete", "extractor_calls": 1,
                    "stable": parsed.file_version.stable, "provenance_observed": True,
                    "parser_errors": list(parsed.error_categories), "extractor_errors": [],
                    "malformed_records": parsed.malformed_records, "unrecognized_records": parsed.unrecognized_records,
                    "source_version": {"sha256": parsed.file_version.sha256, "byte_count": parsed.file_version.byte_count,
                                       "inode": parsed.file_version.before.inode}}
        result = types.SimpleNamespace(coverage=coverage, facts_extracted=0, facts_stored=0, facts_updated=0,
                                       facts_skipped_gate=0, facts_skipped_dedup=0, trace=[])
        assert completion.record_receipt(loaded, result)
    assert completion.has_receipt(capture)
    assert _state(capture)[capture.key]["released"] is True
    assert path.exists(), "a referenced source is never removed"


def test_removed_replaced_or_truncated_source_settles_with_a_recorded_outcome(env):
    session, path, data = _session(env, _records(8, payload=2000))
    capture = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    assert completion.dispatch(capture) > 0
    path.write_bytes(data[:len(data) // 2])
    with pytest.raises(completion.CompletionError):
        completion.load_capture(capture.key, "", "", "claude", root=env.root)
    entry = _state(capture)[capture.key]
    assert entry["status"] == "exhausted"
    assert entry["outcomes"][-1]["category"] == "source_prefix_changed"
    assert completion.dispatch(capture) == 0

    # A replaced or removed file is caught by dispatch before any worker starts.
    for change in ("replace", "remove"):
        other, other_path, other_data = _session(env, _records(8, payload=2000))
        referenced = completion.admit_capture(str(other_path), other, "", "", "claude", root=env.root)
        if change == "replace":
            replacement = other_path.with_suffix(".replacement")
            replacement.write_bytes(other_data)
            os.replace(replacement, other_path)
        else:
            other_path.unlink()
        assert completion.dispatch(referenced) == 0
        assert _state(referenced)[referenced.key]["outcomes"][-1]["category"] == "source_prefix_changed"

    # A parse of a different file than the one referenced never completes.
    late, late_path, _ = _session(env, _records(8, payload=2000))
    late_capture = completion.admit_capture(str(late_path), late, "", "", "claude", root=env.root)
    parsed = transcript.parse_transcript_outcome(late_path, compact=True,
                                                 byte_limit=late_capture.metadata["byte_count"])
    coverage = {"parser_status": "complete", "extractor_status": "complete", "extractor_calls": 1, "stable": True,
                "provenance_observed": True, "parser_errors": [], "extractor_errors": [], "malformed_records": 0,
                "unrecognized_records": 0, "source_version": {"sha256": parsed.file_version.sha256,
                                                              "byte_count": parsed.file_version.byte_count,
                                                              "inode": parsed.file_version.before.inode + 1}}
    result = types.SimpleNamespace(coverage=coverage, facts_extracted=0, facts_stored=0, facts_updated=0,
                                   facts_skipped_gate=0, facts_skipped_dedup=0, trace=[])
    assert not completion.strict_complete(result, late_capture)


def test_unreadable_source_is_retried_not_settled(env, monkeypatch):
    session, path, _ = _session(env, _records(8, payload=2000))
    capture = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    real = completion._source_state

    def denied(target, fd=None):
        if fd is None and Path(target) == capture.path:
            raise PermissionError("synthetic transient denial")
        return real(target, fd)

    monkeypatch.setattr(completion, "_source_state", denied)
    assert completion.dispatch(capture) == 0
    assert _state(capture)[capture.key]["status"] == "queued"
    assert completion.dispatch(capture) == 0
    monkeypatch.setattr(completion, "_source_state", real)
    assert completion.dispatch(capture) > 0
    monkeypatch.setattr(completion, "_source_state", denied)
    with pytest.raises(completion.CompletionError):
        completion.load_capture(capture.key, "", "", "claude", root=env.root)
    entry = _state(capture)[capture.key]
    assert entry["status"] == "retry"
    assert entry["outcomes"][-1]["category"] == "source_unavailable"


def test_growing_large_session_supersedes_its_queued_prefix(env):
    session, path, data = _session(env, _records(8, payload=2000))
    first = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    with path.open("ab") as handle:
        handle.write(("\n".join(json.dumps(record) for record in _records(2, payload=2000)) + "\n").encode())
    second = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    state = _state(first)
    assert state[first.key]["status"] == "superseded"
    assert state[second.key]["status"] == "queued"
    assert path.exists()


def test_copied_admission_streams_through_a_private_partial(env):
    session, path, data = _session(env, _records(1))
    assert len(data) <= completion.MAX_CAPTURE_BYTES
    capture = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    assert "storage" not in capture.metadata
    assert capture.path.read_bytes() == data
    assert list((capture.directory / "incoming").iterdir()) == []
    stale = capture.directory / "incoming" / "abandoned.partial"
    fd = os.open(stale, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    old = time.time() - 7200
    os.utime(stale, (old, old))
    fresh = capture.directory / "incoming" / "in-flight.partial"
    fd = os.open(fresh, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    other, other_path, _ = _session(env, _records(1))
    completion.admit_capture(str(other_path), other, "", "", "claude", root=env.root)
    assert not stale.exists() and fresh.exists()


def test_cli_parses_exactly_the_referenced_prefix(env, monkeypatch):
    import argparse
    from truememory.ingest import cli
    session, path, data = _session(env, _records(8, payload=2000))
    capture = completion.admit_capture(str(path), session, "", "", "claude", root=env.root)
    calls = []
    monkeypatch.setattr(cli, "ingest", lambda **kwargs: calls.append(kwargs) or types.SimpleNamespace(
        coverage=None, facts_extracted=0, facts_stored=0, facts_updated=0, facts_skipped_gate=0,
        facts_skipped_dedup=0, trace=[], facts_encoded=0, elapsed_seconds=0.0, facts_provenance_updated=0))
    monkeypatch.setattr(cli, "_print_result", lambda result: None)
    import truememory.model_client as model_client
    monkeypatch.setattr(model_client, "ensure_server_running", lambda *args, **kwargs: None)
    monkeypatch.setattr(completion, "record_receipt", lambda *args: False)
    args = argparse.Namespace(transcript=str(path), user="", db=str(env.home / "m.db"), threshold=0.3, trace=None,
                              provider="auto", model="", session=session, verbose=False)
    with pytest.raises(SystemExit):
        cli._execute_ingest(args, capture)
    assert calls[0]["byte_limit"] == len(data)


def test_session_end_hook_never_reads_a_large_source(env, monkeypatch):
    session, path, data = _session(env, _records(2, payload=200_000))
    assert len(data) > completion.MAX_CAPTURE_BYTES
    reads = []
    real_stream = completion._stream
    monkeypatch.setattr(completion, "_stream", lambda fd, count, **kwargs: reads.append(count) or real_stream(fd, count, **kwargs))
    monkeypatch.setattr(transcript._Stream, "records", lambda self, handle: (_ for _ in ()).throw(
        AssertionError("the hook parsed a large source")))
    _end(env, monkeypatch, session, path)
    # Two human turns would be declined at the copy size; at this size the
    # session is substantive and the worker decides.
    assert len(env.spawned) == 1
    assert reads == [0]
    command = env.spawned[0]
    key = command[command.index("--completion-key") + 1]
    metadata = json.loads((completion.account_root("", "", env.root) / "state.json").read_text())["captures"][key]["metadata"]
    assert "sha256" not in metadata and metadata["byte_count"] == len(data)


def _codex_rollout(turns: int, output: int, *, plan: bool = False) -> list[dict]:
    def response(role: str, text: str, turn: str) -> dict:
        return {"type": "response_item", "timestamp": "2026-10-14T00:00:00Z",
                "payload": {"type": "message", "role": role, "id": "i",
                            "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
                            "internal_chat_message_metadata_passthrough": {"turn_id": turn}}}
    records = [{"type": "session_meta", "payload": {"id": "s", "cwd": "/synthetic"}}]
    for index in range(turns):
        turn = f"t{index}"
        records.append({"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn,
                                                         "collaboration_mode_kind": "plan" if plan else "default"}})
        records.append({"type": "turn_context", "payload": {"turn_id": turn, "cwd": "/synthetic",
                        "collaboration_mode": {"mode": "plan" if plan else "default", "settings": {"model": "m"}}}})
        records.append(response("user", f"question {index}", turn))
        records.append({"type": "event_msg", "payload": {"type": "user_message", "message": f"question {index}",
                                                         "images": [], "audio": None}})
        for call in range(3):
            records.append({"type": "response_item", "payload": {"type": "function_call", "name": "shell",
                                                                 "arguments": "{}", "call_id": f"c{call}"}})
            records.append({"type": "response_item", "payload": {"type": "function_call_output", "call_id": f"c{call}",
                                                                 "output": "o" * output}})
        answer = f"<proposed_plan>\nstep {index}\n</proposed_plan>" if plan else f"answer {index}"
        records.append({"type": "event_msg", "payload": {"type": "agent_message", "message": answer}})
        records.append(response("assistant", answer, turn))
        records.append({"type": "event_msg", "payload": {"type": "task_complete", "turn_id": turn,
                                                         "last_agent_message": answer}})
    return records


@pytest.mark.parametrize("plan", [False, True])
def test_codex_compact_keeps_only_what_reconciliation_reads(tmp_path, plan):
    path = tmp_path / "rollout.jsonl"
    rows = _codex_rollout(6, 20_000, plan=plan)
    _write(path, rows)
    full = transcript.parse_transcript_outcome(path)
    compact = transcript.parse_transcript_outcome(path, compact=True)
    assert (compact.status, compact.total_records, compact.malformed_records, compact.unrecognized_records,
            compact.error_categories) == (full.status, full.total_records, full.malformed_records,
                                         full.unrecognized_records, full.error_categories)
    assert transcript.format_for_extraction(compact.messages) == transcript.format_for_extraction(full.messages)
    records = transcript._Records(allow_strings=False, compact=True)
    for row in rows:
        records.add(row)
    assert all(set(record.payload) <= set(transcript._CODEX_RECONCILED) for _, record in records.native)
    # Tool calls and outputs are not retained at all.
    assert len(records.native) == len(rows) - 6 * 6
