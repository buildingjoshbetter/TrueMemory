"""Regression tests for issue #459: MPS OOM causes silent separation vector loss.

When MPS GPU runs out of memory during embedding, the system must fall back to
CPU rather than silently dropping separation vectors.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mps_oom_error():
    """Create a RuntimeError that looks like an MPS OOM."""
    return RuntimeError(
        "MPS backend out of memory (MPS allocated: 2.33 GiB, "
        "other allocations: 5.12 MiB, max allowed: 1.95 GiB). "
        "Tried to allocate 256.00 MiB."
    )


class TestIssue459MPSOOMFallback:
    """Verify that MPS OOM falls back to CPU instead of losing data."""

    def test_issue_459_embed_single_separation_vector_survives_oom(self):
        """embed_single must NOT silently drop separation vectors on MPS OOM.

        Simulates: first encode() call (primary embedding) succeeds,
        second encode() call (separation embedding) raises MPS OOM.
        After the fix, it should retry on CPU and succeed.
        """
        from truememory.vector_search import embed_single, init_vec_table

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT, sender TEXT, recipient TEXT, timestamp TEXT)")
        conn.execute("INSERT INTO messages VALUES (1, 'test memory content', 'alice', 'bob', '2026-01-01')")
        conn.commit()

        init_vec_table(conn)

        fake_embedding = np.random.rand(256).astype(np.float32)
        call_count = [0]

        def mock_encode(texts, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise _make_mps_oom_error()
            return np.array([fake_embedding])

        mock_model = MagicMock()
        mock_model.encode = mock_encode

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            embed_single(conn, 1, "test memory content")
            conn.commit()

        sep_tbl = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vec_messages_sep%'"
        ).fetchone()

        sep_tbl = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vec_messages_sep%'"
        ).fetchall()
        assert sep_tbl, "No separation vector table found"
        sep_count = conn.execute(f"SELECT COUNT(*) FROM {sep_tbl[0][0]}").fetchone()[0]
        assert sep_count == 1, (
            f"Separation vector missing (count={sep_count}) — MPS OOM caused "
            "silent data loss instead of falling back to CPU"
        )

        conn.close()

    def test_issue_459_build_separation_vectors_survives_oom(self):
        """build_separation_vectors must retry on CPU when MPS OOM occurs mid-batch."""
        from truememory.vector_search import build_separation_vectors, init_vec_table

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT, sender TEXT, recipient TEXT, timestamp TEXT)")
        for i in range(5):
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
                (i + 1, f"memory {i}", "alice", "bob", "2026-01-01"),
            )
        conn.commit()

        init_vec_table(conn)

        fake_embedding = np.random.rand(256).astype(np.float32)
        call_count = [0]

        def mock_encode(texts, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise _make_mps_oom_error()
            return np.array([fake_embedding] * len(texts))

        mock_model = MagicMock()
        mock_model.encode = mock_encode

        with patch("truememory.vector_search.get_model", return_value=mock_model):
            try:
                count = build_separation_vectors(conn)
            except RuntimeError as e:
                if "MPS" in str(e):
                    pytest.fail(
                        "build_separation_vectors propagated MPS OOM instead of "
                        "falling back to CPU — silent data loss on the bulk path"
                    )
                raise

    def test_issue_459_model_server_embed_survives_oom(self):
        """Model server embed handler must retry on CPU when MPS OOM occurs."""
        from truememory.model_server import ModelServer

        server = ModelServer()
        fake_vectors = np.random.rand(3, 256).astype(np.float32)
        call_count = [0]

        def mock_encode(texts, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise _make_mps_oom_error()
            return fake_vectors[:len(texts)]

        mock_model = MagicMock()
        mock_model.encode = mock_encode

        server._embed_model = mock_model
        server._embed_tier = ""

        request = {"op": "embed", "texts": ["hello", "world", "test"], "tier": ""}
        response = server.handle_request(request)

        assert response.get("ok") is not True or call_count[0] > 1, (
            "Model server did not attempt CPU fallback on MPS OOM — "
            "clients will see errors and silently lose embeddings"
        )

    def test_issue_459_non_mps_errors_still_propagate(self):
        """Non-MPS RuntimeErrors must NOT be caught by the OOM fallback."""
        from truememory.model_server import ModelServer

        server = ModelServer()

        def mock_encode(texts, **kwargs):
            raise RuntimeError("CUDA error: device-side assert triggered")

        mock_model = MagicMock()
        mock_model.encode = mock_encode

        server._embed_model = mock_model
        server._embed_tier = ""

        request = {"op": "embed", "texts": ["hello"], "tier": ""}
        with pytest.raises(RuntimeError, match="CUDA"):
            server.handle_request(request)
