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
from dataclasses import dataclass, field
from pathlib import Path

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
    for entry in entries:
        if isinstance(entry, str) and allow_strings:
            messages.append(Message(role="unknown", content=entry))
            continue
        if not isinstance(entry, dict):
            malformed += 1
            continue
        top_type = entry.get("type") or entry.get("role") or "unknown"
        if not isinstance(top_type, str):
            malformed += 1
            continue
        if isinstance(top_type, str) and top_type in _NON_CONVERSATION_TYPES:
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
    status = ("partial" if messages else "malformed") if malformed else "partial" if unrecognized else "complete"
    errors = (("malformed_records",) if malformed else ()) + (("unrecognized_content",) if unrecognized else ())
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
        log.error("Transcript source unreadable", extra={"error_category": "source_unreadable"})
        return []

    if path is not None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return []
        except (OSError, ValueError):
            log.error("Transcript source unreadable", extra={"error_category": "source_unreadable"})
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
}


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
