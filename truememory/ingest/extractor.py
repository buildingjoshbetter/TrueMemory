"""
Fact Extractor
==============

LLM-based atomic fact extraction from conversation transcripts.
This is "deep processing" (Craik & Lockhart, 1972) — extracting meaning
and connections, not just surface text. Deeper processing produces
stronger, more retrievable memories.

Extracts:
- Personal facts (name, location, age, job, relationships)
- Preferences (likes, dislikes, style choices, tool preferences)
- Decisions (chose X over Y, committed to Z)
- Corrections (user corrected a previous assumption)
- Temporal facts (events with dates, deadlines, plans)
- Technical context (architecture decisions, configs, stack choices)

Each fact is atomic — one clear statement per fact. This mirrors how
the hippocampus encodes individual episodes rather than entire experiences.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from truememory.ingest.models import LLMConfig, LLMError, complete

log = logging.getLogger(__name__)


EXTRACTION_SYSTEM = """\
You are a memory extraction system. Your job is to extract atomic facts \
from conversations that should be remembered for future interactions.

You extract ONLY durable, reusable information — things that would be \
useful to recall in a future conversation days or weeks from now."""

EXTRACTION_PROMPT = """\
Given this conversation transcript, extract atomic facts worth remembering.

EXTRACT:
- Personal facts (name, location, age, job, relationships)
- Preferences (likes, dislikes, style preferences, tool/tech choices)
- Decisions made (chose X over Y, committed to Z, agreed on approach)
- Corrections (user corrected an assumption — these are high-value)
- Temporal facts (events with dates, deadlines, upcoming plans)
- Technical context (project architecture, stack decisions, configurations)
- Relationship details (who someone is, how they relate to others)

DO NOT EXTRACT:
- Transient debugging details (error messages, stack traces, temp fixes)
- Code snippets or implementation specifics (those live in the codebase)
- Greetings, pleasantries, filler ("hey", "sounds good", "thanks")
- Things obvious from the codebase or git history
- The assistant's own suggestions or explanations (only USER-stated facts)

For each fact, provide:
- "content": A clear, atomic statement. Write as a fact, not a quote.
  Good: "Prefers bun over npm"
  Bad: "The user mentioned they like bun"
- "category": One of: personal, preference, decision, correction, temporal, technical, relationship
- "confidence": high, medium, or low
- "source_role": "user" (stated by user) or "inferred" (implied by context)

Return a JSON array of objects. If no facts are worth extracting, return [].

TRANSCRIPT:
{transcript}

Extract atomic facts as JSON array:"""


@dataclass
class ExtractedFact:
    """An atomic fact extracted from a conversation."""
    content: str
    category: str = "general"
    confidence: str = "medium"
    source_role: str = "user"


# Chunking configuration for long transcripts.
#
# Per-chunk character budget: sized to fit comfortably inside an 8K–16K
# token context window after system prompt overhead. 20_000 chars ≈ ~5K
# tokens of English text, leaving headroom for the extraction prompt,
# system prompt, and the JSON response budget.
_CHUNK_CHAR_BUDGET = 20_000

# Hard cap on how many chunks we'll run per transcript before we log a
# warning and stop. At 20 chunks × ~5s per LLM call that's already ~100s
# of extraction latency; beyond this the user should split their
# transcript upstream. Callers can tune via the ``max_chunks`` argument.
_DEFAULT_MAX_CHUNKS = 20


def _chunk_transcript(transcript: str, budget: int = _CHUNK_CHAR_BUDGET) -> list[str]:
    """Split a transcript into ``budget``-sized chunks on message boundaries.

    The formatted transcript produced by :func:`format_for_extraction`
    separates messages with blank lines (``\\n\\n``), so we split on those
    to keep each chunk self-contained. If a single message is larger than
    the budget, it becomes its own (over-budget) chunk rather than being
    cut mid-sentence — the LLM will still read most of it and downstream
    dedup will catch any overlap with neighbouring chunks.
    """
    if len(transcript) <= budget:
        return [transcript]

    # Split on message boundaries; preserve the separator so we can
    # reassemble cleanly. format_for_extraction uses "\n\n" as separator.
    messages = transcript.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for msg in messages:
        # +2 accounts for the "\n\n" separator between messages.
        msg_len = len(msg) + 2
        if current and current_len + msg_len > budget:
            chunks.append("\n\n".join(current))
            current = [msg]
            current_len = msg_len
        else:
            current.append(msg)
            current_len += msg_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _dedupe_facts_by_content(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    """Drop facts with duplicate ``content`` (case-insensitive) preserving order.

    When the same fact is extracted from multiple overlapping chunks (e.g.
    the user's name restated across several sections), this collapses them
    before they hit the downstream encoding gate + dedup pipeline. That's
    cheap at the extractor layer and saves a Claude CLI subprocess per
    duplicate downstream.
    """
    seen: set[str] = set()
    out: list[ExtractedFact] = []
    for fact in facts:
        key = fact.content.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(fact)
    return out


def extract_facts(
    transcript: str,
    config: LLMConfig,
    max_facts: int = 50,
    max_chunks: int = _DEFAULT_MAX_CHUNKS,
) -> list[ExtractedFact]:
    """
    Extract atomic facts from a conversation transcript using an LLM.

    This is the "deep processing" stage — the LLM analyzes meaning,
    identifies important information, and produces clean atomic statements.

    For transcripts longer than ``_CHUNK_CHAR_BUDGET`` (~20K chars) we
    split on message boundaries and run the extractor on each chunk,
    merging and de-duplicating the resulting facts. This replaces the
    older behaviour which silently truncated to the first 30K chars and
    appended ``[transcript truncated]`` — for long conversations that
    dropped up to 95% of the content with zero signal to the user. See
    Bug #5 in PERF_REPORT.md.

    If chunking would produce more than ``max_chunks`` chunks, we emit a
    warning and process only the first ``max_chunks`` (which bounds worst-
    case LLM cost). The warning includes the drop ratio so callers know
    exactly how much content was skipped.

    Args:
        transcript: Formatted conversation text.
        config: LLM configuration for the extraction model.
        max_facts: Maximum facts to return per transcript (after merging).
        max_chunks: Maximum chunks to process. Defaults to 20, which at
            ~20K chars/chunk covers transcripts up to ~400K chars.

    Returns:
        List of ExtractedFact objects, deduplicated by content.
    """
    if not transcript.strip():
        return []

    chunks = _chunk_transcript(transcript, budget=_CHUNK_CHAR_BUDGET)

    if len(chunks) > max_chunks:
        total_chars = len(transcript)
        kept_chars = sum(len(c) for c in chunks[:max_chunks])
        dropped_pct = 100.0 * (1.0 - (kept_chars / max(total_chars, 1)))
        log.warning(
            "Transcript chunked into %d pieces but max_chunks=%d — processing "
            "first %d chunks only (%d/%d chars, %.1f%% of content dropped). "
            "Consider splitting the transcript upstream or raising max_chunks.",
            len(chunks), max_chunks, max_chunks, kept_chars, total_chars, dropped_pct,
        )
        chunks = chunks[:max_chunks]

    if len(chunks) > 1:
        log.info(
            "Transcript is %d chars; extracting across %d chunks (budget=%d/chunk)",
            len(transcript), len(chunks), _CHUNK_CHAR_BUDGET,
        )

    all_facts: list[ExtractedFact] = []
    for i, chunk in enumerate(chunks):
        prompt = EXTRACTION_PROMPT.format(transcript=chunk)
        try:
            response = complete(config, prompt, system=EXTRACTION_SYSTEM)
        except LLMError as e:
            log.error(
                "LLM extraction failed for chunk %d/%d (%s): %s",
                i + 1, len(chunks), config.provider, e,
            )
            # Keep whatever we've gathered from earlier chunks; don't let a
            # single mid-transcript failure wipe out the whole extraction.
            continue
        except Exception as e:
            log.exception(
                "Unexpected error during LLM extraction of chunk %d/%d: %s",
                i + 1, len(chunks), e,
            )
            continue

        chunk_facts = _parse_extraction_response(response, max_facts)
        if len(chunks) > 1:
            log.debug("Chunk %d/%d yielded %d facts", i + 1, len(chunks), len(chunk_facts))
        all_facts.extend(chunk_facts)

    # Deduplicate by content (case-insensitive) across chunks so the same
    # fact restated in multiple windows doesn't balloon the output list.
    merged = _dedupe_facts_by_content(all_facts)

    # Respect the caller's max_facts cap on the merged result.
    if len(merged) > max_facts:
        merged = merged[:max_facts]

    return merged


def _parse_extraction_response(response: str, max_facts: int) -> list[ExtractedFact]:
    """Parse the LLM's JSON response into ExtractedFact objects.

    Handles common LLM output variations:
    - Raw JSON array: `[...]`
    - Markdown-wrapped: `` ```json\n[...]\n``` ``
    - Markdown-wrapped with trailing prose: `` ```json\n[...]\n```\n\nExplanation... ``
    - JSON with preamble: `Here are the facts: [...]`
    - Object wrapper: `{"facts": [...]}` (unwrapped automatically)
    - Malformed responses (falls back to brace-matching salvage)
    """
    json_str = response.strip()

    # Strip leading markdown code fence if present
    json_str = re.sub(r"^```(?:json)?\s*\n?", "", json_str)

    # Walk the string to find a *balanced* JSON array or object. This is
    # more robust than regex because it handles:
    #   (1) trailing prose after the JSON ("```\n\nHere's the explanation...")
    #   (2) commentary before the JSON ("Here are the facts: [...]")
    #   (3) code fences embedded mid-response
    # and correctly respects string literals (braces inside strings).
    extracted = _find_first_balanced(json_str, "[", "]")
    if extracted is None:
        extracted = _find_first_balanced(json_str, "{", "}")
    if extracted is None:
        log.warning("No JSON array or object found in extraction response")
        return []
    json_str = extracted

    try:
        facts_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        log.warning("Failed to parse extraction JSON: %s", e)
        return _salvage_partial_json(json_str)

    # Unwrap common object wrappers: {"facts": [...]}, {"items": [...]}
    if isinstance(facts_data, dict):
        for key in ("facts", "items", "results", "data", "extracted"):
            if key in facts_data and isinstance(facts_data[key], list):
                facts_data = facts_data[key]
                break
        else:
            # No list found inside the object
            log.warning("Extraction response is an object without a known list key")
            return []

    if not isinstance(facts_data, list):
        log.warning("Extraction response is not a list after unwrap")
        return []

    facts = []
    for item in facts_data[:max_facts]:
        if not isinstance(item, dict):
            continue
        content = item.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if not content:
            continue
        facts.append(ExtractedFact(
            content=content,
            category=str(item.get("category", "general")),
            confidence=str(item.get("confidence", "medium")),
            source_role=str(item.get("source_role", "user")),
        ))

    log.info("Extracted %d facts from transcript", len(facts))
    return facts


def _find_first_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """Find the first balanced ``open_ch...close_ch`` span in ``text``.

    Walks the string tracking depth and respecting JSON string literals
    (including escaped quotes) so braces/brackets inside strings don't
    throw off the match. Returns ``None`` if no balanced span exists.

    This is used to extract a JSON array or object from an LLM response
    that may have leading commentary, trailing prose, or code fences.
    """
    start = text.find(open_ch)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _salvage_partial_json(text: str) -> list[ExtractedFact]:
    """Salvage facts from malformed JSON using brace-balanced extraction.

    Unlike a naive regex, this walks the string and matches balanced braces,
    correctly handling nested objects like {"content": "x", "metadata": {...}}.
    """
    facts = []
    i = 0
    n = len(text)

    while i < n:
        # Find the next opening brace
        start = text.find("{", i)
        if start < 0:
            break

        # Walk forward until we find the matching close brace
        depth = 0
        in_string = False
        escape = False
        end = -1
        for j in range(start, n):
            ch = text[j]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break

        if end < 0:
            break

        chunk = text[start:end + 1]
        try:
            item = json.loads(chunk)
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str) and content.strip():
                    facts.append(ExtractedFact(
                        content=content.strip(),
                        category=str(item.get("category", "general")),
                        confidence=str(item.get("confidence", "medium")),
                        source_role=str(item.get("source_role", "user")),
                    ))
        except json.JSONDecodeError:
            pass

        i = end + 1

    if facts:
        log.info("Salvaged %d facts from malformed JSON", len(facts))
    return facts


def extract_facts_simple(transcript: str) -> list[ExtractedFact]:
    """
    Extract facts without an LLM — regex/heuristic fallback.

    This is the "System 1" fast path. Less accurate but zero-cost,
    useful when no LLM is available or for real-time buffering.

    Looks for:
    - "I am/I'm [X]" patterns (personal facts)
    - "I prefer/like/hate [X]" patterns (preferences)
    - "My [X] is [Y]" patterns (personal facts)
    - Statements with proper nouns and verbs (decisions, events)
    """
    facts = []

    # Personal identity patterns
    for pattern, category in [
        (r"(?:I am|I'm|i am|i'm)\s+(?:a\s+)?(.{3,60}?)(?:\.|,|!|\n|$)", "personal"),
        (r"(?:my name is|i'm called|call me)\s+(\w+)", "personal"),
        (r"(?:I live|i live|I'm from|i'm from|I'm in|i'm in)\s+(.{3,40}?)(?:\.|,|!|\n|$)", "personal"),
        (r"(?:I work|i work)\s+(?:at|for|as)\s+(.{3,40}?)(?:\.|,|!|\n|$)", "personal"),
    ]:
        for match in re.finditer(pattern, transcript):
            facts.append(ExtractedFact(
                content=match.group(0).strip().rstrip(".,!"),
                category=category,
                confidence="medium",
                source_role="user",
            ))

    # Preference patterns
    for pattern in [
        r"(?:I prefer|i prefer|I like|i like|I love|i love)\s+(.{3,60}?)(?:\.|,|!|\n|$)",
        r"(?:I hate|i hate|I dislike|i dislike|I avoid|i avoid)\s+(.{3,60}?)(?:\.|,|!|\n|$)",
        r"(?:I always|i always|I never|i never|I usually|i usually)\s+(.{3,60}?)(?:\.|,|!|\n|$)",
    ]:
        for match in re.finditer(pattern, transcript):
            facts.append(ExtractedFact(
                content=match.group(0).strip().rstrip(".,!"),
                category="preference",
                confidence="low",
                source_role="user",
            ))

    return facts
