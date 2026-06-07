"""Regression tests for issue #460: warm search 15x slower than cold search.

Root cause: embed_single() (used by add()) does not write embedder metadata.
On the next process startup, _check_embedder_compatibility logs the "without
embedder metadata" warning, and the Qwen3 NaN fix path sees no flag and
re-embeds ALL vectors — making the "warm" path far slower than cold.

Fix: embed_single() must call _write_embedder_metadata() so subsequent
opens skip the NaN re-embed.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestIssue460WarmSearch:
    """Verify embedder metadata is written after embed_single."""

    def _make_mock_model(self):
        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.array(
            [np.random.rand(256).astype(np.float32)] * len(texts)
        )
        return mock_model

    def test_issue_460_embed_single_writes_metadata(self):
        """After add() via embed_single, embedder metadata must exist."""
        from truememory.client import Memory
        from truememory.vector_search import _read_embedder_metadata

        m = Memory(path=":memory:")
        mock_model = self._make_mock_model()

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            m.add(content="test memory", user_id="alice")

        stored_model, stored_dim = _read_embedder_metadata(m._engine.conn)
        assert stored_model is not None, (
            "embed_single did not write embed_model metadata — warm search "
            "will trigger unnecessary Qwen3 NaN re-embed on next startup"
        )
        assert stored_dim is not None, (
            "embed_single did not write embed_dim metadata"
        )

    def test_issue_460_no_reembed_on_warm_connect(self):
        """Second engine connecting to same DB must NOT trigger re-embed."""
        import tempfile, sqlite3
        from pathlib import Path
        from truememory.client import Memory
        from truememory.vector_search import _read_embedder_metadata

        mock_model = self._make_mock_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with patch("truememory.vector_search.get_model", return_value=mock_model):
                m1 = Memory(path=str(db_path))
                m1.add(content="first memory", user_id="alice")
                m1.add(content="second memory", user_id="alice")

            stored_model, _ = _read_embedder_metadata(m1._engine.conn)
            assert stored_model is not None, "metadata should be set after add()"

            encode_call_count = [0]
            original_encode = mock_model.encode

            def counting_encode(texts, **kw):
                encode_call_count[0] += len(texts)
                return np.array(
                    [np.random.rand(256).astype(np.float32)] * len(texts)
                )

            mock_model.encode = counting_encode

            with patch("truememory.vector_search.get_model", return_value=mock_model):
                m2 = Memory(path=str(db_path))
                m2.search("first memory")

            assert encode_call_count[0] <= 2, (
                f"Second engine re-encoded {encode_call_count[0]} texts — "
                "warm search is re-embedding all vectors due to missing metadata"
            )

    def test_issue_460_metadata_survives_multiple_adds(self):
        """Metadata must persist across multiple add() calls."""
        from truememory.client import Memory
        from truememory.vector_search import _read_embedder_metadata

        m = Memory(path=":memory:")
        mock_model = self._make_mock_model()

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            for i in range(5):
                m.add(content=f"memory {i}", user_id="alice")

        stored_model, stored_dim = _read_embedder_metadata(m._engine.conn)
        assert stored_model is not None
        assert stored_dim is not None
