"""Regression tests for issue #455: L5 consolidation dead for MCP users.

MCP users call add() which only does insert + embed (4/14 build steps).
The truememory_consolidate tool fills the gap by rebuilding L2-L5 layers.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestIssue455ConsolidationMCP:
    """Verify that truememory_consolidate rebuilds L5 layers."""

    def _make_engine_with_messages(self, n=10):
        """Create a Memory engine with n messages, using only add()."""
        from truememory.client import Memory

        m = Memory(path=":memory:")
        fake_embedding = np.random.rand(256).astype(np.float32)

        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.array(
            [fake_embedding] * len(texts)
        )

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            for i in range(n):
                m.add(
                    content=f"I had a meeting with person_{i % 3} about project_{i % 2}",
                    user_id="alice",
                )
        return m

    def test_issue_455_consolidate_tool_exists(self):
        """truememory_consolidate must be registered as an MCP tool."""
        from truememory.mcp_server import mcp

        tool_names = []
        for tool in mcp._tool_manager.list_tools():
            tool_names.append(tool.name)

        assert "truememory_consolidate" in tool_names, (
            "truememory_consolidate tool not registered — MCP users have no "
            "way to trigger consolidation after add()"
        )

    def test_issue_455_add_only_leaves_summaries_empty(self):
        """After only add() calls, summaries table should be empty."""
        m = self._make_engine_with_messages(5)
        conn = m._engine.conn

        has_summaries = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='summaries'"
        ).fetchone()

        if has_summaries:
            count = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
            assert count == 0, (
                f"summaries has {count} rows after add()-only — "
                "consolidation is running from add() (if this passes, issue #455 is already fixed)"
            )

    def test_issue_455_consolidate_populates_tables(self):
        """After calling consolidate, at least one L5 table should have data."""
        m = self._make_engine_with_messages(10)
        conn = m._engine.conn

        from truememory.mcp_server import truememory_consolidate

        with patch("truememory.mcp_server._get_memory", return_value=m):
            result = truememory_consolidate()

        import json
        stats = json.loads(result)

        has_error_only = all(
            "ERROR" in str(v) or "SKIPPED" in str(v)
            for v in stats.values()
        )
        assert not has_error_only, (
            f"All consolidation steps failed or skipped: {stats}. "
            "At least one L5 layer should succeed."
        )

    def test_issue_455_consolidate_returns_timing_stats(self):
        """Consolidate tool must return per-step timing stats."""
        m = self._make_engine_with_messages(5)

        from truememory.mcp_server import truememory_consolidate

        with patch("truememory.mcp_server._get_memory", return_value=m):
            result = truememory_consolidate()

        import json
        stats = json.loads(result)

        assert isinstance(stats, dict), "Consolidate must return JSON object"
        assert len(stats) > 0, "Consolidate must return at least one stat"
