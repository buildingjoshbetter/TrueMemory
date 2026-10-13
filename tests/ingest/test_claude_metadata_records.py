"""Claude Code session metadata records do not make transcript coverage partial."""
from __future__ import annotations

import json
from pathlib import Path

from truememory.ingest.transcript import parse_transcript_outcome

FIXTURE = Path(__file__).parent / "fixtures" / "sample_real_claude_code_transcript.jsonl"


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "session.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_real_format_fixture_is_complete():
    outcome = parse_transcript_outcome(FIXTURE)
    assert outcome.status == "complete"
    assert outcome.malformed_records == 0
    assert [message.role for message in outcome.messages] == [
        "human", "assistant", "assistant", "tool_result", "human", "assistant"]


def test_claude_21_metadata_records_are_not_conversation(tmp_path):
    prompt = "Synthetic fixture: the blue widget ships on Tuesday."
    records = [
        {"type": "permission-mode", "permissionMode": "default"},
        {"type": "queue-operation", "operation": "enqueue", "content": prompt},
        {"type": "attachment", "attachment": {"type": "hook_success", "hookEvent": "SessionStart", "content": ""}},
        {"type": "attachment", "attachment": {"type": "date", "date": "2026-10-13"}},
        {"type": "atis-latch", "atis": ""},
        {"type": "user", "message": {"role": "user", "content": prompt}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "Noted."}]}},
        {"type": "last-prompt", "lastPrompt": prompt},
        {"type": "cost-state", "totalCostUSD": 0},
    ]
    outcome = parse_transcript_outcome(_write(tmp_path, records))
    assert outcome.status == "complete"
    assert outcome.malformed_records == 0 and outcome.unrecognized_records == 0
    assert [(message.role, message.content) for message in outcome.messages] == [
        ("human", prompt), ("assistant", "Noted.")]


def test_unlisted_metadata_type_still_counts_as_malformed(tmp_path):
    records = [
        {"type": "user", "message": {"role": "user", "content": "Synthetic fixture prompt."}},
        {"type": "future-metadata", "value": 1},
    ]
    outcome = parse_transcript_outcome(_write(tmp_path, records))
    assert outcome.status == "partial"
    assert outcome.malformed_records == 1


def test_unknown_attachment_type_is_not_assumed_benign(tmp_path):
    records = [
        {"type": "user", "message": {"role": "user", "content": "Synthetic fixture prompt."}},
        {"type": "attachment", "attachment": {"type": "queued_command", "prompt": "Synthetic queued text."}},
    ]
    outcome = parse_transcript_outcome(_write(tmp_path, records))
    assert outcome.status == "partial"
    assert outcome.malformed_records == 1


def test_attachment_with_top_level_text_is_still_captured(tmp_path):
    records = [
        {"type": "user", "message": {"role": "user", "content": "Synthetic fixture prompt."}},
        {"type": "attachment", "attachment": {"type": "date"}, "content": "Synthetic attached text."},
    ]
    outcome = parse_transcript_outcome(_write(tmp_path, records))
    assert "Synthetic attached text." in [message.content for message in outcome.messages]


def test_unrecorded_queued_prompt_is_still_captured(tmp_path):
    records = [
        {"type": "user", "message": {"role": "user", "content": "Synthetic fixture prompt."}},
        {"type": "queue-operation", "operation": "enqueue", "content": "Synthetic prompt never dequeued."},
    ]
    outcome = parse_transcript_outcome(_write(tmp_path, records))
    assert "Synthetic prompt never dequeued." in [message.content for message in outcome.messages]
