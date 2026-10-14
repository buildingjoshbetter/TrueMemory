"""Synthetic controls for parsing large transcripts in bounded memory.

The hook counts human turns only until its threshold, and the worker streams
JSONL record by record, keeping only what extraction reads.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from truememory.ingest import transcript


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
