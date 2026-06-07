"""Regression tests for issue #463: HyDE not gated to Pro tier.

HyDE query expansion should ONLY fire for Pro-tier users. Edge and Base
users with API keys configured should NOT trigger HyDE (it adds latency
and unexpected API costs).
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestIssue463HydeTierGate:
    """Verify HyDE is gated to Pro tier only."""

    def _make_memory(self):
        from truememory.client import Memory

        m = Memory(path=":memory:")
        fake_embedding = np.random.rand(256).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.array(
            [fake_embedding] * len(texts)
        )
        with patch("truememory.vector_search.get_model", return_value=mock_model):
            for i in range(5):
                m.add(content=f"Test memory {i}", user_id="alice")
        return m, mock_model

    def test_issue_463_edge_tier_no_hyde(self):
        """Edge-tier search must NOT invoke HyDE even with llm_fn available."""
        from truememory.mcp_server import truememory_search

        m, mock_model = self._make_memory()
        fake_llm = MagicMock(return_value="hypothetical document")

        with (
            patch("truememory.mcp_server._get_memory", return_value=m),
            patch("truememory.mcp_server._get_llm_fn", return_value=fake_llm),
            patch("truememory.vector_search.get_model", return_value=mock_model),
            patch("truememory.reranker._active_tier", "edge"),
            patch("truememory.mcp_server._load_config", return_value={"tier": "edge"}),
        ):
            truememory_search(query="test memory", user_id="alice", limit=5)

        fake_llm.assert_not_called(), (
            "HyDE fired on Edge tier — should only fire on Pro"
        )

    def test_issue_463_base_tier_no_hyde(self):
        """Base-tier search must NOT invoke HyDE even with llm_fn available."""
        from truememory.mcp_server import truememory_search

        m, mock_model = self._make_memory()
        fake_llm = MagicMock(return_value="hypothetical document")

        with (
            patch("truememory.mcp_server._get_memory", return_value=m),
            patch("truememory.mcp_server._get_llm_fn", return_value=fake_llm),
            patch("truememory.vector_search.get_model", return_value=mock_model),
            patch("truememory.reranker._active_tier", "base"),
            patch("truememory.mcp_server._load_config", return_value={"tier": "base"}),
        ):
            truememory_search(query="test memory", user_id="alice", limit=5)

        fake_llm.assert_not_called(), (
            "HyDE fired on Base tier — should only fire on Pro"
        )

    def test_issue_463_pro_tier_allows_hyde(self):
        """Pro-tier search should pass llm_fn through (HyDE allowed)."""
        from truememory.mcp_server import truememory_search

        m, mock_model = self._make_memory()
        fake_llm = MagicMock(return_value="hypothetical document")

        with (
            patch("truememory.mcp_server._get_memory", return_value=m),
            patch("truememory.mcp_server._get_llm_fn", return_value=fake_llm),
            patch("truememory.vector_search.get_model", return_value=mock_model),
            patch("truememory.reranker._active_tier", "pro"),
            patch("truememory.mcp_server._load_config", return_value={"tier": "pro"}),
        ):
            result = truememory_search(query="test memory", user_id="alice", limit=5)

        assert isinstance(json.loads(result), list)

    def test_issue_463_search_deep_also_gated(self):
        """truememory_search_deep must also gate HyDE to Pro only."""
        from truememory.mcp_server import truememory_search_deep

        m, mock_model = self._make_memory()
        fake_llm = MagicMock(return_value="hypothetical document")

        with (
            patch("truememory.mcp_server._get_memory", return_value=m),
            patch("truememory.mcp_server._get_llm_fn", return_value=fake_llm),
            patch("truememory.vector_search.get_model", return_value=mock_model),
            patch("truememory.reranker._active_tier", "edge"),
            patch("truememory.mcp_server._load_config", return_value={"tier": "edge"}),
        ):
            truememory_search_deep(query="test memory", user_id="alice", limit=5)

        fake_llm.assert_not_called(), (
            "HyDE fired on Edge tier via search_deep — should only fire on Pro"
        )
