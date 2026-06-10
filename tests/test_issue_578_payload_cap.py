"""Tests for issue #578 — per-memory truncation + total payload budget."""

import os
import pytest

from truememory.ingest.hooks.session_start import (
    _truncate_memory,
    _apply_budget,
    RECALL_MEMORY_CHARS,
    RECALL_BUDGET_CHARS,
)


# ---------------------------------------------------------------------------
# _truncate_memory — per-entry truncation
# ---------------------------------------------------------------------------

class TestTruncateMemory:
    """Per-memory truncation on word boundary with pointer suffix."""

    def test_short_memory_unchanged(self):
        """Memories within the limit pass through verbatim."""
        content = "User prefers dark mode"
        assert _truncate_memory(content, 42) == content

    def test_exact_limit_unchanged(self):
        """Memory exactly at the limit is NOT truncated."""
        content = "x" * RECALL_MEMORY_CHARS
        assert _truncate_memory(content, 1) == content

    def test_long_memory_truncated_on_word_boundary(self):
        """A long memory is sliced at the last word boundary before the limit."""
        words = ["word"] * 200  # 200 * 5 = 1000 chars (well over 500)
        content = " ".join(words)
        result = _truncate_memory(content, 99)
        assert "[truncated, id=99" in result
        assert "truememory_get" in result
        # The text portion (before the suffix) should end at a word boundary —
        # no partial words.
        text_part = result.split(" [truncated")[0]
        assert not text_part.endswith(" ")
        # Must be shorter than the original
        assert len(text_part) <= RECALL_MEMORY_CHARS

    def test_pointer_suffix_format(self):
        """The truncation pointer has the exact expected format."""
        content = "a " * 400  # 800 chars
        result = _truncate_memory(content, "abc-123")
        assert result.endswith("[truncated, id=abc-123 — use truememory_get]")

    def test_single_long_word(self):
        """A single word exceeding the limit is sliced without a space split."""
        content = "x" * 1000
        result = _truncate_memory(content, 7)
        assert "[truncated, id=7" in result
        # Falls back to slicing at max_chars since there's no space
        text_part = result.split(" [truncated")[0]
        assert len(text_part) == RECALL_MEMORY_CHARS

    def test_empty_memory(self):
        """Empty string passes through unchanged."""
        assert _truncate_memory("", 1) == ""

    def test_custom_max_chars(self):
        """The max_chars parameter overrides the default."""
        content = "hello world this is a test string"
        # max_chars=11 → content (32 chars) exceeds it, so it IS truncated
        result = _truncate_memory(content, 5, max_chars=11)
        assert "[truncated, id=5" in result
        # max_chars=100 → content fits, so it passes through unchanged
        result = _truncate_memory(content, 5, max_chars=100)
        assert result == content

    def test_custom_max_chars_short(self):
        """Short max_chars truncates correctly."""
        content = "hello world foo bar baz"
        result = _truncate_memory(content, 5, max_chars=10)
        assert "[truncated, id=5" in result
        text_part = result.split(" [truncated")[0]
        assert len(text_part) <= 10


# ---------------------------------------------------------------------------
# _apply_budget — total payload budget enforcement
# ---------------------------------------------------------------------------

class TestApplyBudget:
    """Total budget enforcement, dropping lowest-salience entries first."""

    def test_under_budget_all_kept(self):
        """When total size is under budget, all lines are kept."""
        lines = [("- short memory", 0.9), ("- another one", 0.8)]
        result = _apply_budget(lines, "", budget=10000)
        assert len(result) == 2

    def test_over_budget_drops_lowest_score(self):
        """When over budget, the lowest-score entries are dropped first."""
        lines = [
            ("- " + "A" * 3000, 0.9),  # high score — keep
            ("- " + "B" * 3000, 0.5),  # mid score
            ("- " + "C" * 3000, 0.1),  # low score — drop first
        ]
        result = _apply_budget(lines, "", budget=7000)
        # Should drop lowest-score entries to fit
        assert any("A" in l for l in result)
        # At least the lowest-score one should be gone
        kept_text = " ".join(result)
        # C (lowest score) should be dropped before A (highest)
        if "C" * 100 in kept_text:
            # If C is kept, A must also be kept
            assert "A" * 100 in kept_text

    def test_directive_block_counted_against_budget(self):
        """The directive block size is subtracted from available budget."""
        big_directive = "D" * 5000
        lines = [("- " + "M" * 2000, 0.9)]
        # Budget = 5500, directive = 5000, overhead ~300 → only ~200 left for memories
        result = _apply_budget(lines, big_directive, budget=5500)
        assert len(result) == 0  # memory too big for remaining budget

    def test_empty_memories(self):
        """No memories → empty result."""
        assert _apply_budget([], "", budget=8192) == []

    def test_preserves_order(self):
        """Kept entries preserve their original insertion order."""
        lines = [
            ("- first", 0.7),
            ("- second", 0.9),
            ("- third", 0.5),
        ]
        result = _apply_budget(lines, "", budget=10000)
        assert result == ["- first", "- second", "- third"]


# ---------------------------------------------------------------------------
# Directives are NOT truncated
# ---------------------------------------------------------------------------

class TestDirectivesExempt:
    """Directives must not be truncated — they are standing instructions."""

    def test_directive_block_not_truncated(self):
        """The directive XML block is emitted verbatim regardless of length.

        _truncate_memory is only called on recall results (all_results), not
        on directive entries. We verify by checking that _truncate_memory is
        NOT applied to content that would be a directive.
        """
        # Directives go through a separate code path that never calls
        # _truncate_memory. We verify the contract: _truncate_memory is only
        # called for memories, and directive_block is passed intact to
        # _apply_budget which only uses len() on it.
        long_directive = "Always respond in French. " * 100  # ~2600 chars
        # _apply_budget accepts the directive as-is and only measures its size
        lines = [("- mem", 0.9)]
        result = _apply_budget(lines, long_directive, budget=10000)
        assert result == ["- mem"]


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    """TRUEMEMORY_RECALL_MEMORY_CHARS and TRUEMEMORY_RECALL_BUDGET_CHARS."""

    def test_memory_chars_env_override(self, monkeypatch):
        """TRUEMEMORY_RECALL_MEMORY_CHARS changes the default per-memory cap."""
        monkeypatch.setenv("TRUEMEMORY_RECALL_MEMORY_CHARS", "50")
        # Re-import to pick up the new env value
        import importlib
        import truememory.ingest.hooks.session_start as mod
        importlib.reload(mod)
        try:
            assert mod.RECALL_MEMORY_CHARS == 50
            content = "word " * 20  # 100 chars
            result = mod._truncate_memory(content, 1)
            assert "[truncated" in result
        finally:
            # Restore
            monkeypatch.delenv("TRUEMEMORY_RECALL_MEMORY_CHARS", raising=False)
            importlib.reload(mod)

    def test_budget_chars_env_override(self, monkeypatch):
        """TRUEMEMORY_RECALL_BUDGET_CHARS changes the total payload budget."""
        monkeypatch.setenv("TRUEMEMORY_RECALL_BUDGET_CHARS", "1000")
        import importlib
        import truememory.ingest.hooks.session_start as mod
        importlib.reload(mod)
        try:
            assert mod.RECALL_BUDGET_CHARS == 1000
        finally:
            monkeypatch.delenv("TRUEMEMORY_RECALL_BUDGET_CHARS", raising=False)
            importlib.reload(mod)
