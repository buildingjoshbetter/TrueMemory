"""
Transcript Parser
=================

Parses Claude Code conversation transcripts into structured messages.
Handles multiple formats:
- JSONL (one JSON object per line)
- JSON array of message objects
- Plain text with role markers

Claude Code stores transcripts as JSON arrays of conversation turns,
each with type/role, content, and optional tool metadata.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
import stat
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator

log = logging.getLogger(__name__)


@dataclass
class Message:
    """A single conversation turn."""
    role: str           # "human", "assistant", "system", "tool_use", "tool_result"
    content: str        # The text content
    timestamp: str = "" # ISO timestamp if available
    tool_name: str = "" # Tool name for tool_use messages


@dataclass(frozen=True)
class FileState:
    """File identity without a path or transcript content."""
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileState:
        return cls(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@dataclass(frozen=True)
class TranscriptFileVersion:
    """Identity of the bytes read, not a later completion-time stat."""
    byte_count: int
    sha256: str
    before: FileState
    after: FileState
    path_after: FileState | None

    @property
    def stable(self) -> bool:
        return self.before == self.after == self.path_after and self.byte_count == self.after.size


@dataclass(frozen=True)
class TranscriptOutcome:
    """Coverage evidence; only complete outcomes include all recognized input.

    Status is complete, partial, salvaged, malformed or unreadable. Messages
    remain available for inspection even when legacy parsing salvaged input.
    This API does not change the ingestion pipeline's completion policy.
    """
    messages: list[Message] = field(repr=False)
    status: str
    total_records: int = 0
    malformed_records: int = 0
    error_categories: tuple[str, ...] = ()
    file_version: TranscriptFileVersion | None = None
    unrecognized_records: int = 0

    @property
    def complete(self) -> bool:
        return self.status == "complete"


# Pinned Codex rollout grammar: openai/codex 0b755b1945bf4a31560df2ce1469aeb044dccbc9.
# These are exact exclusions, never prefix rules or recursive payload discovery.
_CODEX_CONTEXT = {
    "session_meta", "turn_context", "compacted", "token_usage_record", "world_state",
    "retained_context", "security_risk_score", "inter_agent_communication",
    "inter_agent_communication_metadata",
}
_CODEX_RESPONSE_CONTEXT = {
    "additional_tools", "agent_message", "reasoning", "local_shell_call", "function_call",
    "function_call_output", "custom_tool_call", "custom_tool_call_output", "tool_search_call",
    "tool_search_output", "web_search_call", "image_generation_call", "compaction",
    "compaction_summary", "configuration_update", "compaction_trigger", "context_compaction",
}
_CODEX_EVENT_CONTEXT = {
    "token_count", "agent_reasoning", "agent_reasoning_raw_content", "context_compacted",
    "exec_command_begin", "exec_command_end", "exec_command_output_delta",
    "mcp_tool_call_begin", "mcp_tool_call_end", "dynamic_tool_call_request",
    "dynamic_tool_call_response", "patch_apply_begin", "patch_apply_end", "patch_apply_updated",
    "web_search_begin", "web_search_end", "view_image_tool_call", "image_generation_begin",
    "image_generation_end", "stream_error", "stream_info", "error", "warning",
}
_CODEX_ITEM_CONTEXT = {
    "FunctionCallOutput", "HookPrompt", "Reasoning", "CommandExecution", "DynamicToolCall",
    "CollabAgentToolCall", "SubAgentActivity", "WebSearch", "ImageView", "ImageGeneration",
    "EnteredReviewMode", "ExitedReviewMode", "FileChange", "McpToolCall", "ContextCompaction",
}
_CODEX_SCAFFOLD_KINDS = {
    "agents_md.instructions", "hooks.additional_context", "guardian.retained_instructions",
}
_CODEX_PHASES = {"commentary", "partial_answer", "final_answer"}
_CODEX_START = {"task_started", "turn_started"}
_CODEX_END = {"task_complete", "turn_complete"}
_CODEX_ANY = object()
_CODEX_WHITE_SPACE = "\t\n\v\f\r \u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
_CodexCitation = tuple[tuple[tuple[str, int, int, str], ...], tuple[str, ...]]


@dataclass
class _CodexRecord:
    kind: str
    message: Message | None = field(default=None, repr=False)
    role: str = ""
    projection: str | None = field(default=None, repr=False)
    plan_projection: str | None = field(default=None, repr=False)
    projection_rewritten: bool = field(default=False, repr=False)
    memory_citation: _CodexCitation | None = field(default=None, repr=False)
    plan_body: str | None = field(default=None, repr=False)
    plan_event: bool = field(default=False, repr=False)
    frozen_plan_mode: bool = field(default=False, repr=False)
    turn_id: str | None = field(default=None, repr=False)
    item_id: str | None = field(default=None, repr=False)
    phase: str | None = None
    payload: dict = field(default_factory=dict, repr=False)
    malformed: bool = False
    unrecognized: bool = False
    errors: set[str] = field(default_factory=set)

    def reject(self, category: str, *, malformed: bool = False) -> None:
        self.errors.add(category)
        if malformed:
            self.malformed = True
        else:
            self.unrecognized = True


def _codex_attribution(value: dict) -> tuple[str, bool]:
    """Return absent/user/excluded/unknown, validating even legacy selectors."""
    for key, expected in (("harness_injected", bool), ("source_tool_namespace", str)):
        if key in value and value[key] is not None and type(value[key]) is not expected:
            return "unknown", True
    if "provenance" in value:
        provenance = value["provenance"]
        if not isinstance(provenance, dict) or not isinstance(provenance.get("type"), str):
            return "unknown", True
        kind = provenance["type"]
        if kind == "developer_instructions":
            if type(provenance.get("from_additional_requirements")) is not bool:
                return "unknown", True
        if kind == "harness" and type(provenance.get("hook", False)) is not bool:
            return "unknown", True
        if kind == "tool" and provenance.get("namespace") is not None:
            if not isinstance(provenance["namespace"], str):
                return "unknown", True
        if kind == "user":
            return "user", False
        if kind in {"developer_instructions", "skills", "agents_md", "harness", "tool"}:
            return "excluded", False
        return "unknown", False
    if value.get("source_tool_namespace") is not None or value.get("harness_injected") is True:
        return "excluded", False
    # Empty historical metadata and harness_injected:false are not positive
    # provenance. They use only the documented older role/text compatibility.
    return "absent", False


def _codex_optional_string(record: _CodexRecord, payload: dict, key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        record.reject("codex_invalid_metadata", malformed=True)
        return None
    return value or None


def _codex_strip_citations(text: str) -> tuple[str, list[str]]:
    """Pinned literal, nonnested citation projection, including open-at-EOF."""
    opening, closing = "<oai-mem-citation>", "</oai-mem-citation>"
    visible: list[str] = []
    bodies: list[str] = []
    offset = 0
    while True:
        start = text.find(opening, offset)
        if start < 0:
            visible.append(text[offset:])
            break
        visible.append(text[offset:start])
        body_start = start + len(opening)
        end = text.find(closing, body_start)
        if end < 0:
            bodies.append(text[body_start:])
            break
        bodies.append(text[body_start:end])
        offset = end + len(closing)
    return "".join(visible), bodies


def _codex_plan_parts(text: str) -> tuple[str, str | None, bool]:
    """Pinned whole-LF-line tags; all surviving characters remain exact."""
    visible: list[str] = []
    active = False
    saw_delimiter = False
    plan_body: list[str] | None = None
    lines = text.split("\n")
    for index, body in enumerate(lines):
        line = body + ("\n" if index + 1 < len(lines) else "")
        slug = body.strip(_CODEX_WHITE_SPACE)
        if slug in ("<proposed_plan>", "</proposed_plan>"):
            saw_delimiter = True
        if slug == "<proposed_plan>" and not active:
            active = True
            plan_body = []
        elif slug == "</proposed_plan>" and active:
            active = False
        elif not active:
            visible.append(line)
        elif plan_body is not None:
            plan_body.append(line)
    return "".join(visible), None if plan_body is None else "".join(plan_body), saw_delimiter


def _codex_strip_plan(text: str) -> tuple[str, bool]:
    visible, _, saw_delimiter = _codex_plan_parts(text)
    return visible, saw_delimiter


def _codex_rust_lines(text: str) -> Iterator[str]:
    """Rust lines split only at LF, stripping its preceding CR."""
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if index + 1 == len(lines) and not line:
            continue
        yield line[:-1] if index + 1 < len(lines) and line.endswith("\r") else line


def _codex_u32(text: str) -> int | None:
    text = text.strip(_CODEX_WHITE_SPACE)
    if text.startswith("+"):
        text = text[1:]
    if not text or any(character not in "0123456789" for character in text):
        return None
    digits = text.lstrip("0") or "0"
    if len(digits) > 10 or (len(digits) == 10 and digits > "4294967295"):
        return None
    return int(digits)


def _codex_citation_body_block(text: str, opening: str, closing: str) -> str | None:
    start = text.find(opening)
    if start < 0:
        return None
    start += len(opening)
    end = text.find(closing, start)
    return None if end < 0 else text[start:end]


def _codex_derive_citation(bodies: list[str]) -> _CodexCitation | None:
    entries: list[tuple[str, int, int, str]] = []
    rollout_ids: list[str] = []
    seen: set[str] = set()
    for body in bodies:
        block = _codex_citation_body_block(body, "<citation_entries>", "</citation_entries>")
        if block is not None:
            for line in _codex_rust_lines(block):
                line = line.strip(_CODEX_WHITE_SPACE)
                if "|note=[" not in line or not line.endswith("]"):
                    continue
                location, note = line.rsplit("|note=[", 1)
                if ":" not in location:
                    continue
                path, span = location.rsplit(":", 1)
                if "-" not in span:
                    continue
                start, end = span.split("-", 1)
                first, last = _codex_u32(start), _codex_u32(end)
                if first is not None and last is not None:
                    entries.append((path.strip(_CODEX_WHITE_SPACE), first, last,
                                    note[:-1].strip(_CODEX_WHITE_SPACE)))
        ids = _codex_citation_body_block(body, "<rollout_ids>", "</rollout_ids>")
        if ids is None:
            ids = _codex_citation_body_block(body, "<thread_ids>", "</thread_ids>")
        if ids is not None:
            for line in _codex_rust_lines(ids):
                value = line.strip(_CODEX_WHITE_SPACE)
                if value and value not in seen:
                    seen.add(value)
                    rollout_ids.append(value)
    return (tuple(entries), tuple(rollout_ids)) if entries or rollout_ids else None


def _codex_citation_object(value: object) -> _CodexCitation | None:
    if not isinstance(value, dict) or set(value) != {"entries", "rolloutIds"}:
        return None
    entries, ids = value["entries"], value["rolloutIds"]
    if not isinstance(entries, list) or not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        return None
    result: list[tuple[str, int, int, str]] = []
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"path", "lineStart", "lineEnd", "note"}:
            return None
        if not isinstance(item["path"], str) or not isinstance(item["note"], str):
            return None
        if any(type(item[key]) is not int or not 0 <= item[key] <= 4294967295
               for key in ("lineStart", "lineEnd")):
            return None
        result.append((item["path"], item["lineStart"], item["lineEnd"], item["note"]))
    return tuple(result), tuple(ids)


def _codex_decode_citation(record: _CodexRecord, value: object) -> None:
    if value is not None:
        record.memory_citation = _codex_citation_object(value)
        if record.memory_citation is None:
            record.reject("codex_unsupported_projection")


def _codex_context_mode(payload: dict) -> str | None:
    mode = payload.get("collaboration_mode")
    if not isinstance(mode, dict) or set(mode) != {"mode", "settings"}:
        return None
    if mode["mode"] not in ("plan", "default"):
        return None
    settings = mode["settings"]
    if not isinstance(settings, dict) or set(settings) - {"model", "reasoning_effort", "developer_instructions"}:
        return None
    if not isinstance(settings.get("model"), str):
        return None
    effort, instructions = settings.get("reasoning_effort"), settings.get("developer_instructions")
    if effort is not None and (not isinstance(effort, str) or not effort):
        return None
    if instructions is not None and not isinstance(instructions, str):
        return None
    return mode["mode"]


def _decode_codex_record(entry: dict) -> _CodexRecord | None:
    """Decode exact native envelopes without interpreting arbitrary nested prose."""
    kind = entry.get("type")
    native = kind in _CODEX_CONTEXT if isinstance(kind, str) else False
    native |= kind in ("response_item", "event_msg", "realtime_item")
    if not native:
        generic_kind = kind or entry.get("role")
        if ("payload" not in entry
                or (isinstance(generic_kind, str) and generic_kind in _NON_CONVERSATION_TYPES)
                or generic_kind in ("user", "human", "assistant", "tool_use", "tool_result", "system")
                or (not kind and "role" in entry and "content" in entry)):
            return None
    record = _CodexRecord(kind=kind if isinstance(kind, str) else "unknown")
    if not isinstance(kind, str):
        record.reject("codex_invalid_record", malformed=True)
        return record
    if not isinstance(entry.get("timestamp", ""), str):
        record.reject("codex_invalid_metadata", malformed=True)
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        record.reject("codex_invalid_record", malformed=True)
        return record
    record.payload = payload
    if kind in _CODEX_CONTEXT:
        return record
    if kind == "event_msg":
        return _decode_codex_event(record)
    if kind != "response_item":
        record.reject("codex_unrecognized_record")
        return record
    variant = payload.get("type")
    if not isinstance(variant, str):
        record.reject("codex_invalid_record", malformed=True)
        return record
    if variant in _CODEX_RESPONSE_CONTEXT:
        return record
    if variant != "message":
        record.reject("codex_unrecognized_record")
        return record
    role, content = payload.get("role"), payload.get("content")
    if not isinstance(role, str) or not isinstance(content, list):
        record.reject("codex_invalid_record", malformed=True)
        return record
    if role in ("system", "developer", "tool"):
        return record
    if role not in ("user", "assistant"):
        record.reject("codex_unrecognized_record")
        return record
    record.role = "human" if role == "user" else "assistant"
    outer, metadata = entry.get("metadata"), payload.get("internal_chat_message_metadata_passthrough")
    if any(value is not None and not isinstance(value, dict) for value in (outer, metadata)):
        record.reject("codex_invalid_metadata", malformed=True)
        return record
    outer, metadata = outer or {}, metadata or {}
    excluded = False
    for key in ("compaction_output", "inherited_user_message"):
        if key in outer:
            if type(outer[key]) is not bool:
                record.reject("codex_invalid_metadata", malformed=True)
                return record
            excluded |= outer[key]
    if "delivered_assistant_message" in outer:
        marker = outer["delivered_assistant_message"]
        if not isinstance(marker, str):
            record.reject("codex_invalid_metadata", malformed=True)
        elif marker.startswith("codex:code-mode-delivery:v1:incomplete:"):
            record.reject("codex_delivery_unavailable")
        elif marker != "codex:code-mode-delivery:v1:complete":
            record.reject("codex_unrecognized_record")
        return record
    if excluded:
        return record
    record.turn_id = _codex_optional_string(record, metadata, "turn_id")
    record.item_id = _codex_optional_string(record, payload, "id")
    record.phase = _codex_optional_string(record, payload, "phase")
    if record.phase is not None and record.phase not in _CODEX_PHASES:
        record.reject("codex_unrecognized_record")
    annotations, kinds = metadata.get("content_item_metadata"), metadata.get("content_item_kinds")
    for values, value_type in ((annotations, dict), (kinds, str)):
        if values is not None and (not isinstance(values, list) or len(values) != len(content)
                                   or any(not isinstance(value, value_type) for value in values)):
            record.reject("codex_invalid_metadata", malformed=True)
            return record
    parts: list[str] = []
    output_parts: list[str] = []
    projection_allowed = True
    for index, block in enumerate(content):
        attribution, bad = _codex_attribution(annotations[index]) if annotations is not None else ("absent", False)
        block_kind = kinds[index] if kinds is not None else ""
        if bad:
            record.reject("codex_invalid_metadata", malformed=True)
            projection_allowed = False
            continue
        if attribution == "excluded":
            # Proven attribution already covers generated context, including
            # new fragment kinds. Kind fallback cannot re-authorize it.
            projection_allowed = False
            continue
        if block_kind in _CODEX_SCAFFOLD_KINDS:
            if attribution == "user":
                record.reject("codex_unrecognized_provenance")
            projection_allowed = False
            continue
        if block_kind not in ("", "unknown"):
            record.reject("codex_unrecognized_provenance")
            projection_allowed = False
            continue
        if attribution == "unknown" or (attribution == "user" and role != "user"):
            record.reject("codex_unrecognized_provenance")
            projection_allowed = False
            continue
        if not isinstance(block, dict) or not isinstance(block.get("type"), str):
            record.reject("codex_invalid_content", malformed=True)
            continue
        if block["type"] not in ("input_text", "output_text"):
            record.reject("codex_unrecognized_content")
            continue
        if not isinstance(block.get("text"), str):
            record.reject("codex_invalid_content", malformed=True)
            continue
        parts.append(block["text"])
        if block["type"] == "output_text":
            output_parts.append(block["text"])
        if role == "user" and block["type"] != "input_text":
            projection_allowed = False
    public_text = "\n".join(part for part in parts if part).strip()
    if public_text:
        timestamp = entry.get("timestamp", "")
        record.message = Message(record.role, public_text, timestamp if isinstance(timestamp, str) else "")
    if projection_allowed and not record.malformed and not record.unrecognized:
        raw = "".join(parts)
        record.projection = raw
        if role == "assistant":
            record.projection, citations = _codex_strip_citations(raw)
            record.projection_rewritten = bool(citations)
            record.memory_citation = _codex_derive_citation(citations)
            output = "".join(output_parts)
            if "<proposed_plan>" in output:
                _, plan_body, _ = _codex_plan_parts(output)
                if plan_body is not None:
                    record.plan_body, _ = _codex_strip_citations(plan_body)
            if any(tag in record.projection for tag in ("<proposed_plan>", "</proposed_plan>")):
                plan_projection, saw_delimiter = _codex_strip_plan(record.projection)
                if saw_delimiter:
                    # A private candidate awaits positive same-turn mode.
                    record.plan_projection = plan_projection
    return record


def _decode_codex_event(record: _CodexRecord) -> _CodexRecord:
    payload = record.payload
    kind = payload.get("type")
    if not isinstance(kind, str):
        record.reject("codex_invalid_record", malformed=True)
        return record
    record.kind = kind
    if kind in _CODEX_START | _CODEX_END | {"turn_aborted"}:
        record.turn_id = _codex_optional_string(record, payload, "turn_id")
        if record.turn_id is None:
            record.reject("codex_unsupported_lifecycle")
        if kind in _CODEX_END:
            summary = payload.get("last_agent_message")
            if summary is not None:
                if not isinstance(summary, str) or not summary.strip(_CODEX_WHITE_SPACE):
                    record.reject("codex_unmatched_terminal")
                else:
                    record.projection = summary
        return record
    if kind in _CODEX_EVENT_CONTEXT:
        return record
    if kind in ("item_started", "item_completed"):
        item = payload.get("item")
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            record.reject("codex_invalid_record", malformed=True)
            return record
        item_kind = item["type"]
        if item_kind == "Plan":
            record.plan_event = True
            record.turn_id = _codex_optional_string(record, payload, "turn_id")
            record.item_id = _codex_optional_string(record, item, "id")
            record.projection = item.get("text") if isinstance(item.get("text"), str) else None
            permitted = {"type", "thread_id", "turn_id", "item", "started_at_ms", "completed_at_ms"}
            if (set(item) != {"type", "id", "text"}
                    or set(payload) - permitted or record.turn_id is None
                    or record.item_id != record.turn_id + "-plan" or record.projection is None
                    or not isinstance(payload.get("thread_id"), str) or not payload["thread_id"]):
                record.reject("codex_unsupported_projection")
            for key in ("started_at_ms", "completed_at_ms"):
                if key in payload and not (key == "started_at_ms" and payload[key] is None):
                    value = payload[key]
                    if type(value) is not int or not -(2**63) <= value < 2**63:
                        record.reject("codex_unsupported_projection")
            if kind == "item_started" and (record.projection != "" or "completed_at_ms" in payload
                    or type(payload.get("started_at_ms")) is not int):
                record.reject("codex_unsupported_projection")
            return record
        if item_kind in _CODEX_ITEM_CONTEXT:
            return record
        if item_kind not in ("UserMessage", "AgentMessage"):
            record.reject("codex_unrecognized_record")
            return record
        if kind == "item_started":
            return record
        record.turn_id = _codex_optional_string(record, payload, "turn_id")
        record.item_id = _codex_optional_string(record, item, "id")
        if record.turn_id is None or record.item_id is None:
            record.reject("codex_unmatched_conversation_event")
        record.role = "human" if item_kind == "UserMessage" else "assistant"
        record.phase = _codex_optional_string(record, item, "phase")
        content = item.get("content")
        expected_type = "text" if record.role == "human" else "Text"
        if not isinstance(content, list) or any(
            not isinstance(block, dict) or block.get("type") != expected_type
            or not isinstance(block.get("text"), str) for block in content
        ):
            record.reject("codex_unmatched_conversation_event")
        elif record.role == "assistant" and len(content) != 1:
            record.reject("codex_unsupported_projection")
        else:
            record.projection = "".join(block["text"] for block in content)
        if any(item.get(key) not in (None, []) for key in ("delivery", "questions")):
            record.reject("codex_unsupported_projection")
        if record.role == "assistant":
            _codex_decode_citation(record, item.get("memory_citation"))
        elif item.get("memory_citation") not in (None, []):
            record.reject("codex_unsupported_projection")
        return record
    if kind not in ("user_message", "agent_message"):
        record.reject("codex_unrecognized_record")
        return record
    record.role = "human" if kind == "user_message" else "assistant"
    record.phase = _codex_optional_string(record, payload, "phase")
    if not isinstance(payload.get("message"), str):
        record.reject("codex_unmatched_conversation_event")
    else:
        record.projection = payload["message"]
    # These native event fields carry additional conversation unavailable to
    # the plain-text decoder. Empty producer defaults do not add evidence.
    media = ("images", "image_details", "file_ids", "file_id_details", "image_order", "local_images",
             "local_image_details", "audio", "local_audio", "delivery", "questions")
    if any(payload.get(key) not in (None, []) for key in media):
        record.reject("codex_unmatched_conversation_event")
    if record.role == "assistant":
        _codex_decode_citation(record, payload.get("memory_citation"))
    elif payload.get("memory_citation") not in (None, []):
        record.reject("codex_unmatched_conversation_event")
    return record


def _codex_echo_index(credits: list[tuple[int, _CodexRecord]], turn_id: str | None) -> dict[tuple, list[int]]:
    """Eight ordinary entries; citation evidence adds eight exact selectors."""
    index: dict[tuple, list[int]] = {}
    for position, (_, candidate) in enumerate(credits):
        projections: tuple = (candidate.projection,)
        if candidate.memory_citation is not None:
            projections += ((candidate.projection, candidate.memory_citation),)
        for projection in projections:
            for phase in (candidate.phase, _CODEX_ANY):
                for turn in (turn_id or candidate.turn_id, _CODEX_ANY):
                    for item in (candidate.item_id, _CODEX_ANY):
                        key = (candidate.role, projection, phase, turn, item)
                        index.setdefault(key, []).append(position)
    return index


def _codex_next_echo(index: dict[tuple, list[int]], event: _CodexRecord, cursor: int) -> int | None:
    """Find the next compatible occurrence with at most two binary searches."""
    projection: str | tuple | None = event.projection
    if event.memory_citation is not None:
        projection = (event.projection, event.memory_citation)
    prefix = (event.role, projection, event.phase if event.phase is not None else _CODEX_ANY,
              event.turn_id if event.turn_id is not None else _CODEX_ANY)
    # Missing canonical IDs are compatible with a supplied event ID. An event
    # without an ID queries all occurrences, not only those missing an ID.
    item_keys = (_CODEX_ANY,) if event.item_id is None else (event.item_id, None)
    matches: list[int] = []
    for item in item_keys:
        positions = index.get((*prefix, item), ())
        offset = bisect_left(positions, cursor)
        if offset < len(positions):
            matches.append(positions[offset])
    return min(matches) if matches else None


def _reconcile_codex(records: list[tuple[int, _CodexRecord]]) -> None:
    """Reconcile supplied-byte evidence; events never create canonical turns."""
    scopes: dict[int, list[tuple[int, _CodexRecord]]] = {}
    scope_ids: dict[int, str | None] = {}
    scope_modes: dict[int, str | None] = {}
    explicit_modes: set[int] = set()
    rewritten_scopes: set[int] = set()
    invalid_scopes: set[int] = set()
    terminals: list[tuple[int, int, _CodexRecord]] = []
    scope = 0
    active = False
    closed_ids: set[str] = set()
    for index, record in records:
        if record.kind == "session_meta":
            scope += 1
            active = False
            closed_ids.clear()
            continue
        if record.kind in _CODEX_START:
            overlapping = active
            if overlapping:
                invalid_scopes.add(scope)
                record.reject("codex_unsupported_lifecycle")
            scope += 1
            active = True
            scope_ids[scope] = record.turn_id
            mode = record.payload.get("collaboration_mode_kind")
            scope_modes[scope] = mode if mode in ("plan", "default") else None
            if "collaboration_mode_kind" in record.payload:
                explicit_modes.add(scope)
            if overlapping or record.turn_id is None or record.malformed:
                invalid_scopes.add(scope)
            scopes.setdefault(scope, [])
            continue
        if record.kind in _CODEX_END | {"turn_aborted"}:
            if active:
                if record.turn_id != scope_ids.get(scope):
                    invalid_scopes.add(scope)
                    record.reject("codex_unsupported_lifecycle")
                terminals.append((index, scope, record))
                active = False
            elif record.turn_id in closed_ids or record.turn_id is None:
                record.reject("codex_unsupported_lifecycle")
            else:
                # No start in this supplied segment: only explicit response
                # turn metadata can explain a nonnull terminal summary.
                terminals.append((index, scope, record))
            if record.turn_id is not None:
                closed_ids.add(record.turn_id)
            scope += 1
            continue
        if active and record.turn_id is not None and record.turn_id != scope_ids.get(scope):
            record.reject("codex_unsupported_lifecycle")
        scopes.setdefault(scope, []).append((index, record))

    for group, items in scopes.items():
        mode = scope_modes.get(group)
        conflicting_mode = False
        mode_conflicts: list[_CodexRecord] = []
        plan_records: list[_CodexRecord] = []
        for _, record in items:
            if record.kind == "turn_context" and record.payload.get("collaboration_mode") is not None:
                if group not in scope_ids or record.payload.get("turn_id") != scope_ids.get(group):
                    continue
                context_mode = _codex_context_mode(record.payload)
                if context_mode is None:
                    mode_conflicts.append(record)
                    conflicting_mode = True
                elif mode is not None and context_mode != mode:
                    mode_conflicts.append(record)
                    conflicting_mode = True
                elif group not in explicit_modes:
                    mode = context_mode
            if record.kind == "response_item" and record.plan_projection is not None:
                plan_records.append(record)
                if mode == "plan" and not conflicting_mode and group not in invalid_scopes:
                    record.projection = record.plan_projection
                    record.projection_rewritten = True
                else:
                    record.projection = None
                    record.reject("codex_unsupported_projection")
            record.frozen_plan_mode = mode == "plan" and not conflicting_mode and group not in invalid_scopes
        if conflicting_mode:
            for _, record in items:
                record.frozen_plan_mode = False
            if plan_records or any(record.plan_event for _, record in items):
                for record in mode_conflicts:
                    record.reject("codex_unsupported_projection")
                for record in plan_records:
                    record.projection = None
                    record.reject("codex_unsupported_projection")
        if any(record.projection_rewritten for _, record in items):
            rewritten_scopes.add(group)
        if group in invalid_scopes:
            for _, record in items:
                if record.role:
                    record.reject("codex_unsupported_lifecycle")
        credits = [(index, record) for index, record in items if record.kind == "response_item"
                   and record.projection is not None and not record.malformed and not record.unrecognized]
        credit_index = _codex_echo_index(credits, scope_ids.get(group))
        plan_index: dict[tuple, list[int]] = {}
        for position, (_, candidate) in enumerate(items):
            if (candidate.kind == "response_item" and candidate.role == "assistant"
                    and candidate.frozen_plan_mode and candidate.plan_body is not None
                    and not candidate.malformed and not candidate.unrecognized):
                key = (scope_ids.get(group), candidate.plan_body)
                plan_index.setdefault(key, []).append(position)
        plan_cursor = 0
        plan_started = False
        plan_completed = False
        for _, plan in items:
            if not plan.plan_event or plan.malformed or plan.unrecognized:
                continue
            if plan.kind == "item_started":
                if (plan_started or plan_completed or not plan.frozen_plan_mode or group not in scope_ids
                        or not plan_index or plan.turn_id != scope_ids.get(group)):
                    plan.reject("codex_unsupported_projection")
                else:
                    plan_started = True
                continue
            positions = plan_index.get((plan.turn_id, plan.projection), ())
            offset = bisect_left(positions, plan_cursor)
            if not plan.frozen_plan_mode or group not in scope_ids or offset == len(positions):
                plan.reject("codex_unsupported_projection")
            else:
                plan_cursor = positions[offset] + 1
                plan_completed = True
        cursors: dict[str, int] = {}
        for _, event in items:
            if event.kind not in ("user_message", "agent_message", "item_completed") or not event.role:
                continue
            if event.malformed or event.unrecognized:
                continue
            position = _codex_next_echo(credit_index, event, cursors.get(event.role, 0))
            if position is None:
                event.reject("codex_unmatched_conversation_event")
                if group in rewritten_scopes or event.memory_citation is not None:
                    event.reject("codex_unsupported_projection")
            else:
                cursors[event.role] = position + 1
    for index, group, terminal in terminals:
        if terminal.projection is None or terminal.malformed or terminal.unrecognized:
            continue
        # Terminal evidence is independent of echo credit consumption. Only
        # earlier canonical occurrences in this lifecycle may explain it.
        candidates = scopes.get(group, [])
        explicit_ids = {candidate.turn_id for before, candidate in candidates
                        if before < index and candidate.kind == "response_item" and candidate.role
                        and candidate.turn_id is not None}
        ambiguous = group not in scope_ids and explicit_ids != {terminal.turn_id}
        if ambiguous or group in invalid_scopes or not any(
            before < index and candidate.kind == "response_item" and candidate.role == "assistant"
            and candidate.projection == terminal.projection and not candidate.malformed and not candidate.unrecognized
            and terminal.turn_id == (scope_ids.get(group) or candidate.turn_id)
            for before, candidate in reversed(candidates)
        ):
            terminal.reject("codex_unmatched_terminal")
            if group in rewritten_scopes:
                terminal.reject("codex_unsupported_projection")


def _transcript_path(source: str | Path) -> Path | None:
    if isinstance(source, Path):
        return source
    try:
        candidate = Path(source)
        # exists()/is_file() suppress access errors on Python 3.14. A single
        # stat keeps unreadable sources distinct from ordinary inline text.
        if stat.S_ISREG(candidate.stat().st_mode):
            return candidate
    except ValueError:
        pass
    except OSError as error:
        if (error.errno not in (errno.ENOENT, errno.ENOTDIR, errno.ENAMETOOLONG)
                and getattr(error, "winerror", None) != 123):
            raise
    return None


def parse_transcript_outcome(source: str | Path) -> TranscriptOutcome:
    """Parse with source coverage, retaining the legacy API separately.

    Explicit Paths are always files, including missing paths. String path
    detection follows the existing parser; other strings are inline content.
    Diagnostics contain categories and counts only. Stable means these
    observations agreed; it does not lock the source against later changes.
    """
    try:
        path = _transcript_path(source)
    except OSError:
        return TranscriptOutcome([], "unreadable", error_categories=("source_unreadable",))

    version = None
    errors: list[str] = []
    if path is not None:
        try:
            with path.open("rb") as handle:
                before = FileState.from_stat(os.fstat(handle.fileno()))
                data = handle.read()
                after = FileState.from_stat(os.fstat(handle.fileno()))
        except (OSError, ValueError):
            return TranscriptOutcome([], "unreadable", error_categories=("source_unreadable",))
        try:
            # Windows stat() and fstat() can give ctime different meanings.
            # Reopen the pathname to check identity using the same API.
            with path.open("rb") as identity_handle:
                path_after = FileState.from_stat(os.fstat(identity_handle.fileno()))
        except OSError:
            path_after = None
        version = TranscriptFileVersion(len(data), hashlib.sha256(data).hexdigest(), before, after, path_after)
        if not version.stable:
            errors.append("source_changed_during_read")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
            errors.append("invalid_utf8")
    else:
        text = source

    parsed = _parse_transcript_text_outcome(text.strip(), jsonl=path is not None and path.suffix.lower() == ".jsonl")
    errors.extend(parsed.error_categories)
    status = parsed.status
    if errors and status == "complete":
        status = "partial" if parsed.messages else "malformed"
    return TranscriptOutcome(parsed.messages, status, parsed.total_records, parsed.malformed_records,
                             tuple(errors), version, parsed.unrecognized_records)


def _covered_content_blocks(blocks: list, *, nested: bool = False) -> tuple[list, bool, bool]:
    """Retain supported text without certifying silently omitted payloads."""
    covered: list = []
    malformed = unrecognized = False
    for block in blocks:
        if isinstance(block, str):
            covered.append(block)
            continue
        if not isinstance(block, dict) or not isinstance(block.get("type"), str):
            malformed = True
            continue
        kind = block["type"]
        if kind == "thinking":
            # Deliberately excluded private reasoning is not lost conversation.
            covered.append(block)
        elif kind == "text":
            if isinstance(block.get("text"), str):
                covered.append(block)
            else:
                malformed = True
        elif kind == "tool_use" and not nested:
            if isinstance(block.get("name", ""), str):
                covered.append(block)
            else:
                malformed = True
        elif kind == "tool_result" and not nested:
            content = block.get("content")
            if isinstance(content, list):
                children, bad, unknown = _covered_content_blocks(content, nested=True)
                malformed |= bad
                unrecognized |= unknown
                covered.append({**block, "content": children})
            elif content is None or isinstance(content, str):
                covered.append(block)
            else:
                malformed = True
        else:
            # Images/audio and unknown block types need an explicit future
            # decoder. Their text-looking keys are not a supported transcript.
            unrecognized = True
    return covered, malformed, unrecognized


def _parse_transcript_text_outcome(text: str, *, jsonl: bool = False) -> TranscriptOutcome:
    if not text:
        return TranscriptOutcome([], "complete")
    if not jsonl and not text.startswith(("[", "{")):
        messages = _parse_plain_text(text)
        return TranscriptOutcome(messages, "complete", len(messages))
    if text.startswith("["):
        try:
            entries = json.loads(text)
        except json.JSONDecodeError:
            messages = _parse_plain_text(text)
            return TranscriptOutcome(messages, "salvaged", 1, 1, ("invalid_json_array",))
        allow_strings = True
    else:
        entries = []
        allow_strings = False
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                entries.append(None)
        # None also represents a syntactically valid but unusable JSONL null;
        # both count as one malformed record in the loop below.

    messages: list[Message] = []
    malformed = unrecognized = 0
    native_records: list[tuple[int, _CodexRecord]] = []
    prompts = _recorded_prompts(entries)
    for index, entry in enumerate(entries):
        if isinstance(entry, str) and allow_strings:
            messages.append(Message(role="unknown", content=entry))
            continue
        if not isinstance(entry, dict):
            malformed += 1
            continue
        native = _decode_codex_record(entry)
        if native is not None:
            native_records.append((index, native))
            if native.message is not None:
                messages.append(native.message)
            continue
        top_type = entry.get("type") or entry.get("role") or "unknown"
        if not isinstance(top_type, str):
            malformed += 1
            continue
        if isinstance(top_type, str) and top_type in _NON_CONVERSATION_TYPES:
            continue
        if _claude_context_record(entry, prompts):
            continue
        # The supported older tool-call format has name/input instead of
        # content. Keep its existing _extract_message behavior intact.
        if top_type != "tool_use":
            inner = entry.get("message")
            payload = inner if isinstance(inner, dict) else entry
            if "content" not in payload or not isinstance(payload["content"], (str, list)):
                malformed += 1
                continue
            if isinstance(payload["content"], list):
                blocks, bad, unknown = _covered_content_blocks(payload["content"])
                malformed += int(bad)
                unrecognized += int(unknown)
                payload = {**payload, "content": blocks}
                entry = {**entry, "message": payload} if isinstance(inner, dict) else payload
        try:
            message = _extract_message(entry)
        except (TypeError, AttributeError, ValueError):
            malformed += 1
            continue
        if message is not None:
            messages.append(message)
    _reconcile_codex(native_records)
    malformed += sum(record.malformed for _, record in native_records)
    unrecognized += sum(record.unrecognized for _, record in native_records)
    status = ("partial" if messages else "malformed") if malformed else "partial" if unrecognized else "complete"
    errors = (("malformed_records",) if malformed else ()) + (("unrecognized_content",) if unrecognized else ())
    errors += tuple(sorted({category for _, record in native_records for category in record.errors}))
    return TranscriptOutcome(messages, status, len(entries), malformed, errors,
                             unrecognized_records=unrecognized)


def parse_transcript(source: str | Path) -> list[Message]:
    """
    Parse a transcript file or string into structured messages.

    Handles Claude Code transcript format automatically. The `source`
    parameter may be:
    - A `Path` object (always treated as a file path)
    - A string that exists as a file (loaded as a file)
    - A string that doesn't exist as a file (treated as inline content)

    This avoids the fragile "len < 500 = path" heuristic from earlier
    versions, which mis-classified long absolute paths as content.

    Once we commit to path mode (a stat identifies a regular file),
    any read failure (``PermissionError``, ``OSError``) is treated as a
    real error and returns an empty list — **never** silently fall back
    to interpreting the path string itself as content. Silently re-parsing
    a path as content produced fake ``Message(role='unknown', content='/tmp/...')``
    entries that could poison the memory store. See Bug #1 in
    EDGE_CASE_REPORT.md for the reproduction.
    """
    text: str

    try:
        path = _transcript_path(source)
    except OSError:
        log.error("Cannot read transcript", extra={"error_category": "source_unreadable"})
        return []

    if path is not None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return []
        except (OSError, ValueError):
            log.error("Cannot read transcript", extra={"error_category": "source_unreadable"})
            return []
    else:
        text = source

    text = text.strip()
    if not text:
        return []

    # Try JSON array first (most common Claude Code format)
    if text.startswith("["):
        return _parse_json_array(text)

    # Try JSONL
    if text.startswith("{"):
        return _parse_jsonl(text)

    # Fall back to plain text
    return _parse_plain_text(text)


def _parse_json_array(text: str) -> list[Message]:
    """Parse a JSON array of conversation turns."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Failed to parse transcript as JSON array, falling back to plain text")
        return _parse_plain_text(text)

    messages = []
    for entry in data:
        if isinstance(entry, dict):
            msg = _extract_message(entry)
            if msg:
                messages.append(msg)
        elif isinstance(entry, str):
            messages.append(Message(role="unknown", content=entry))

    return messages


def _parse_jsonl(text: str) -> list[Message]:
    """Parse JSONL (one JSON object per line).

    Malformed lines are skipped but counted. If any were skipped we emit a
    single summary warning so users aren't surprised when a partially
    corrupted transcript yields fewer facts than expected. Previously these
    were silently dropped with no signal at all.
    """
    messages = []
    total_lines = 0
    malformed = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        total_lines += 1
        try:
            entry = json.loads(line)
            msg = _extract_message(entry)
            if msg:
                messages.append(msg)
        except json.JSONDecodeError:
            malformed += 1
            continue
    if malformed:
        log.warning(
            "Skipped %d malformed JSONL line(s) out of %d total while parsing transcript",
            malformed,
            total_lines,
        )
    return messages


_NON_CONVERSATION_TYPES = {
    "file-history-snapshot",
    "progress",
    "summary",
    "system",
    # Claude Code 2.1 session metadata without conversation text. last-prompt
    # only repeats the leaf user record. Unlisted future types still count as
    # malformed.
    "atis-latch",
    "cost-state",
    "last-prompt",
    "permission-mode",
}

# Claude Code 2.1 attachment types observed carrying host context, not user
# text. Other attachment types keep their previous parsing and coverage.
_CLAUDE_CONTEXT_ATTACHMENTS = frozenset({
    "agent_listing_delta", "auto_mode", "date", "deferred_tools_delta",
    "environment", "hook_success", "model", "prompt_snapshot",
    "remote_session_change", "session_context", "skill_listing",
    "total_tokens_reminder",
})


def _recorded_prompts(entries: list[object]) -> set[str]:
    prompts: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "user" or not isinstance(entry.get("message"), dict):
            continue
        content = entry["message"].get("content")
        if isinstance(content, str):
            prompts.add(content)
        elif isinstance(content, list):
            texts = [block.get("text") for block in content
                     if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)]
            prompts.update(texts)
            prompts.add("".join(texts))
    return prompts


def _claude_context_record(entry: dict, prompts: set[str]) -> bool:
    """True when a Claude Code record holds host context or an already recorded prompt."""
    kind = entry.get("type")
    if kind == "attachment":
        attachment = entry.get("attachment")
        return ("content" not in entry and isinstance(attachment, dict)
                and isinstance(attachment.get("type"), str)
                and attachment["type"] in _CLAUDE_CONTEXT_ATTACHMENTS)
    if kind == "queue-operation":
        content = entry.get("content")
        return content is None or (isinstance(content, str) and content in prompts)
    return False


def _extract_message(entry: dict) -> Message | None:
    """Extract a Message from a conversation turn dict.

    Handles two shapes:

    1. **Simplified shape** (tests + some external tools):
       ``{"type": "human"|"assistant", "content": <str|list>, "timestamp": ...}``

    2. **Real Claude Code on-disk schema**:
       ``{"type": "user"|"assistant", "message": {"role": ..., "content": <str|list>}, "timestamp": ...}``
       where assistant content is a list of blocks that can include
       ``thinking`` (skipped), ``text``, ``tool_use``, and user content can
       include ``tool_result`` blocks.

    Non-conversation entries like ``file-history-snapshot`` and ``progress``
    are filtered out.
    """
    native = _decode_codex_record(entry)
    if native is not None:
        return native.message

    # Handle different field names across formats
    top_type = entry.get("type") or entry.get("role") or "unknown"

    # Filter out non-conversation entry types (file-history-snapshot, progress, etc.)
    if top_type in _NON_CONVERSATION_TYPES:
        return None

    timestamp = entry.get("timestamp", "")

    # Real Claude Code format nests the message under "message"; unwrap it
    # if present so we read content from the right place. The outer "type"
    # ("user"/"assistant") is the authoritative role — the inner role
    # field is redundant but we fall back to it if the outer is missing.
    inner = entry.get("message")
    if isinstance(inner, dict):
        raw_content = inner.get("content", "")
        role = top_type if top_type != "unknown" else inner.get("role", "unknown")
    else:
        raw_content = entry.get("content", "")
        role = top_type

    content = ""
    tool_name = ""
    # Track whether the block list is *entirely* tool_result — in Claude Code's
    # on-disk schema, tool results come back to the model as `type: "user"`
    # entries whose content is a list of tool_result blocks. These are model
    # plumbing, not user conversation, and should be tagged `tool_result` so
    # format_for_extraction filters them out instead of feeding JSON noise
    # to the LLM.
    has_only_tool_results = False

    if isinstance(raw_content, str):
        content = raw_content
    elif isinstance(raw_content, list):
        # Claude API format: list of content blocks
        parts = []
        block_types_seen: set[str] = set()
        for block in raw_content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype:
                    block_types_seen.add(btype)
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "thinking":
                    # Internal chain-of-thought — not part of the conversation
                    # the user would see; skip so we don't leak reasoning
                    # into fact extraction.
                    continue
                elif btype == "tool_use":
                    tool_name = block.get("name", "")
                    parts.append(f"[tool: {tool_name}]")
                elif btype == "tool_result":
                    tr_content = block.get("content")
                    if isinstance(tr_content, list):
                        # tool_result content can itself be a list of blocks
                        for tb in tr_content:
                            if isinstance(tb, dict) and tb.get("type") == "text":
                                parts.append(tb.get("text", ""))
                            elif isinstance(tb, str):
                                parts.append(tb)
                    elif isinstance(tr_content, str):
                        parts.append(tr_content)
                    else:
                        parts.append(str(block.get("output", "")))
            elif isinstance(block, str):
                parts.append(block)
        content = "\n".join(p for p in parts if p)
        # If every block in a user-turn is a tool_result, re-tag the whole
        # message as tool_result so downstream filtering skips it.
        if role in ("user", "human") and block_types_seen == {"tool_result"}:
            has_only_tool_results = True

    # Handle top-level tool_use entries (rare, older format)
    if role == "tool_use":
        tool_name = entry.get("name", "")
        content = f"[tool: {tool_name}] {json.dumps(entry.get('input', {}))}"

    if not content.strip():
        return None

    # Re-tag pure tool-result user turns so they're filtered downstream
    if has_only_tool_results:
        role = "tool_result"
    # Normalize role: "user" and "human" are the same thing downstream
    elif role == "user":
        role = "human"

    return Message(
        role=role,
        content=content.strip(),
        timestamp=timestamp,
        tool_name=tool_name,
    )


def _parse_plain_text(text: str) -> list[Message]:
    """Parse plain text transcript with role markers."""
    messages = []
    current_role = "unknown"
    current_lines = []

    for line in text.splitlines():
        # Detect role markers
        role_match = re.match(r"^(Human|User|Assistant|System|Claude)\s*:\s*(.*)", line, re.IGNORECASE)
        if role_match:
            # Save previous message
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    messages.append(Message(role=current_role, content=content))
                current_lines = []

            role_name = role_match.group(1).lower()
            current_role = "human" if role_name in ("human", "user") else "assistant"
            remaining = role_match.group(2).strip()
            if remaining:
                current_lines.append(remaining)
        else:
            current_lines.append(line)

    # Don't forget the last message
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            messages.append(Message(role=current_role, content=content))

    return messages


# TrueMemory injects context into the live conversation via XML-wrapped
# blocks (the session_start / user_prompt_submit hooks emit
# <truememory-recall>, <truememory-context>, <truememory-directives>,
# <truememory-update>, <truememory-email-request>, <truememory-first-run>,
# ...). When that injected text lands back in the transcript and is fed to
# the extractor, the truncated near-duplicate memories get re-extracted as
# *new* memories — an echo-amplification loop that grows generational copies
# of the same fact (issue #652, M-19). We strip every <truememory-...>...
# </truememory-...> wrapper (and any stray unclosed opener) before extraction
# so our own injected context can never be mined back into the store.
_TRUEMEMORY_BLOCK_RE = re.compile(
    r"<truememory-[a-z0-9-]+\b[^>]*>.*?</truememory-[a-z0-9-]+>",
    re.IGNORECASE | re.DOTALL,
)
_TRUEMEMORY_STRAY_TAG_RE = re.compile(
    r"</?truememory-[a-z0-9-]+\b[^>]*>",
    re.IGNORECASE,
)


def strip_truememory_blocks(text: str) -> str:
    """Remove TrueMemory's own injected ``<truememory-*>`` context blocks.

    These blocks are injected into the conversation by the recall hooks
    (``session_start`` / ``user_prompt_submit``). Left in the transcript they
    would be re-extracted as fresh memories, creating an echo loop of
    truncated near-duplicates (issue #652). We drop whole wrapped blocks
    first, then sweep any stray unbalanced opener/closer tags that survived
    (e.g. a block split across a chunk boundary).
    """
    if "<truememory-" not in text.lower():
        return text
    cleaned = _TRUEMEMORY_BLOCK_RE.sub("", text)
    cleaned = _TRUEMEMORY_STRAY_TAG_RE.sub("", cleaned)
    return cleaned


def format_for_extraction(messages: list[Message]) -> str:
    """
    Format parsed messages into a clean transcript for LLM extraction.
    Filters out tool calls and system messages — focuses on human conversation.

    TrueMemory's own injected ``<truememory-*>`` recall/context blocks are
    stripped from each message before formatting so they can't be re-mined
    back into the store (issue #652, M-19).
    """
    lines = []
    for msg in messages:
        if msg.role in ("tool_use", "tool_result", "system"):
            continue
        role_label = "User" if msg.role == "human" else "Assistant"
        # Drop any TrueMemory-injected context blocks before we consider
        # length / truncation, so echoed recall never reaches the extractor.
        content = strip_truememory_blocks(msg.content)
        # Truncate very long assistant responses (code output, etc.)
        if msg.role == "assistant" and len(content) > 500:
            content = content[:500] + "... [truncated]"
        # Stripping a block may leave a message empty — skip it entirely so we
        # don't emit a bare "User:" / "Assistant:" label with no content.
        if not content.strip():
            continue
        lines.append(f"{role_label}: {content}")

    return "\n\n".join(lines)
