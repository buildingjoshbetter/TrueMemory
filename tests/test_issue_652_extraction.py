"""Tests for issue #652 — extraction quality fixes.

Covers:
1. M-65: a multi-paragraph user message keeps ALL paragraphs through the
   no-LLM (Edge-tier) heuristic extraction path — not just paragraph 1.
2. M-19: TrueMemory's own injected <truememory-recall>/<truememory-context>/
   <truememory-directives> blocks are stripped before extraction and do NOT
   become new memories (the echo-amplification loop is closed).

M-47 (reranker skip on hook call sites) was DEFERRED — the Memory.search
passthrough already exists (#632) and the remaining work is in hook files
owned by other agents, so it is out of scope for this PR.
"""

from truememory.ingest.extractor import (
    _extract_user_lines,
    extract_facts_simple,
)
from truememory.ingest.transcript import (
    Message,
    format_for_extraction,
    strip_truememory_blocks,
)


# ── M-65: multi-paragraph user messages survive role-scoping ────────────


def test_multi_paragraph_user_message_keeps_all_paragraphs():
    """Continuation paragraphs of a user message must be kept, not dropped.

    A single multi-paragraph user message is formatted with internal blank
    lines. _extract_user_lines splits on "\\n\\n", so only the first block
    carries the "User:" prefix; the rest are continuation blocks that must
    be attributed to the same user turn.
    """
    transcript = (
        "User: I prefer dark mode for my editor.\n\n"
        "I also like the bun runtime over npm.\n\n"
        "And my favorite language is Rust."
    )
    user_only = _extract_user_lines(transcript)
    assert "dark mode" in user_only
    assert "bun runtime" in user_only  # paragraph 2 survives
    assert "Rust" in user_only         # paragraph 3 survives


def test_multi_paragraph_facts_extracted_no_llm_path():
    """All paragraphs of a multi-paragraph user turn yield facts (Edge tier)."""
    transcript = (
        "User: I prefer dark mode for my editor.\n\n"
        "I prefer the bun runtime over npm.\n\n"
        "I use Rust for systems work."
    )
    facts = extract_facts_simple(transcript)
    joined = " ".join(f.content.lower() for f in facts)
    assert "dark mode" in joined
    assert "bun runtime" in joined  # paragraph 2 — lost before the M-65 fix
    assert "rust" in joined         # paragraph 3 — lost before the M-65 fix


def test_continuation_after_assistant_not_attributed_to_user():
    """A continuation block after an Assistant turn must NOT count as user."""
    transcript = (
        "User: I like dark mode.\n\n"
        "Assistant: Here is a long answer.\n\n"
        "I am a large language model and I prefer helping.\n\n"
        "User: Thanks."
    )
    user_only = _extract_user_lines(transcript)
    assert "dark mode" in user_only
    # The assistant's continuation paragraph must be excluded.
    assert "large language model" not in user_only


# ── M-19: strip injected TrueMemory blocks before extraction ────────────


def test_strip_truememory_blocks_removes_recall_wrapper():
    text = (
        "User: what do you remember about me?\n"
        "<truememory-recall>\n"
        "- Prefers dark mode (truncated near-dup...)\n"
        "- Uses bun over npm\n"
        "</truememory-recall>"
    )
    cleaned = strip_truememory_blocks(text)
    assert "truememory-recall" not in cleaned
    assert "Prefers dark mode" not in cleaned
    assert "what do you remember" in cleaned


def test_strip_truememory_blocks_removes_context_and_directives():
    text = (
        "<truememory-context>\nstale cached fact\n</truememory-context>\n"
        "real content\n"
        "<truememory-directives>\nalways do X\n</truememory-directives>"
    )
    cleaned = strip_truememory_blocks(text)
    assert "truememory-context" not in cleaned
    assert "truememory-directives" not in cleaned
    assert "stale cached fact" not in cleaned
    assert "always do X" not in cleaned
    assert "real content" in cleaned


def test_strip_truememory_blocks_handles_stray_tags():
    """A block split across a chunk boundary leaves a lone tag — sweep it."""
    text = "User: hi\n<truememory-recall>\ndangling injected line"
    cleaned = strip_truememory_blocks(text)
    assert "<truememory-recall>" not in cleaned


def test_format_for_extraction_strips_injected_recall():
    """Injected recall in message content must not reach the extractor."""
    messages = [
        Message(
            role="human",
            content=(
                "actually my name is Josh.\n"
                "<truememory-recall>\n"
                "- User prefers dark mode\n"
                "- User uses bun over npm\n"
                "</truememory-recall>"
            ),
        ),
    ]
    formatted = format_for_extraction(messages)
    assert "truememory-recall" not in formatted
    assert "prefers dark mode" not in formatted.lower()
    assert "Josh" in formatted


def test_echo_loop_closed_no_new_memories_from_recall():
    """Injected recall block must not be re-extracted as fresh facts.

    Before the fix, the truncated near-duplicate memories inside the recall
    block would be mined back out as new facts (generational copy growth).
    """
    messages = [
        Message(
            role="human",
            content=(
                "<truememory-recall>\n"
                "I prefer dark mode for my editor.\n"
                "I prefer bun over npm.\n"
                "My favorite color is teal.\n"
                "</truememory-recall>"
            ),
        ),
    ]
    formatted = format_for_extraction(messages)
    facts = extract_facts_simple(formatted)
    # None of the echoed "preferences" should have become facts.
    joined = " ".join(f.content.lower() for f in facts)
    assert "dark mode" not in joined
    assert "bun over npm" not in joined
    assert "teal" not in joined


def test_real_user_facts_still_extracted_alongside_stripped_recall():
    """Stripping recall must not suppress genuine new user facts."""
    messages = [
        Message(
            role="human",
            content=(
                "<truememory-recall>\nI prefer dark mode.\n</truememory-recall>\n"
                "actually, I prefer light mode now."
            ),
        ),
    ]
    formatted = format_for_extraction(messages)
    facts = extract_facts_simple(formatted)
    joined = " ".join(f.content.lower() for f in facts)
    assert "light mode" in joined  # the genuine new statement survives
    assert "dark mode" not in joined  # the echoed old one is gone
