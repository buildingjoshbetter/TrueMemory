"""Regression tests for issue #466: temporal key name mismatch.

engine.py:1429 checks intent.get("start_date") and intent.get("end_date")
but detect_temporal_intent() returns "after" and "before".  This causes
temporal cross-source feedback (A1 rescoping) to be silently skipped,
meaning temporally-scoped FTS results never merge into the final results.
"""
from __future__ import annotations

import inspect
import sqlite3
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestIssue466TemporalKeyMismatch:
    """Verify temporal search key names match between producer and consumer."""

    def test_detect_temporal_intent_returns_after_before_keys(self):
        """detect_temporal_intent must return 'after'/'before', not 'start_date'/'end_date'."""
        from truememory.temporal import detect_temporal_intent

        intent = detect_temporal_intent("what happened in January 2026?")
        assert intent["has_temporal"] is True, "Should detect temporal intent"
        assert "after" in intent, "Intent must have 'after' key"
        assert "before" in intent, "Intent must have 'before' key"
        # The mismatched keys must NOT exist
        assert "start_date" not in intent, (
            "Intent should not have 'start_date' — engine.py must use 'after'"
        )
        assert "end_date" not in intent, (
            "Intent should not have 'end_date' — engine.py must use 'before'"
        )

    def test_engine_uses_correct_intent_keys_for_rescope(self):
        """engine.py must reference intent['after']/intent['before'] in the
        cross-source feedback block, not intent['start_date']/intent['end_date'].

        This is a structural test that inspects the source code to catch
        the exact mismatch described in issue #466.
        """
        from truememory.engine import TrueMemoryEngine

        source = inspect.getsource(TrueMemoryEngine.search)

        # The cross-source feedback block must NOT use start_date / end_date
        assert 'intent["start_date"]' not in source and 'intent.get("start_date")' not in source, (
            'engine.py search() still references intent["start_date"] or '
            'intent.get("start_date") — must use intent["after"] / intent.get("after")'
        )
        assert 'intent["end_date"]' not in source and 'intent.get("end_date")' not in source, (
            'engine.py search() still references intent["end_date"] or '
            'intent.get("end_date") — must use intent["after"] / intent.get("before")'
        )

    def test_temporal_rescoped_results_appear_in_search(self):
        """Cross-source temporal feedback must produce 'temporal_rescoped' results.

        This is the core regression test: store messages in a known date range,
        search with a temporal query, and verify that temporal_rescoped results
        appear.  Before the fix, the cross-source block was silently skipped
        because intent["start_date"] was always None.
        """
        from truememory.client import Memory

        m = Memory(path=":memory:")
        fake_embedding = np.random.rand(256).astype(np.float32)

        mock_model = MagicMock()
        mock_model.encode = lambda texts, **kw: np.array(
            [fake_embedding] * len(texts)
        )

        engine = m._engine

        # Store messages in January 2026 with explicit timestamps
        with patch("truememory.vector_search.get_model", return_value=mock_model):
            for i in range(5):
                day = f"2026-01-{10 + i:02d}T10:00:00"
                engine.add(
                    content=f"Had a meeting about project alpha on day {i}",
                    sender="alice",
                    timestamp=day,
                )

            # Also store some messages outside the range
            for i in range(3):
                day = f"2025-06-{10 + i:02d}T10:00:00"
                engine.add(
                    content=f"Old meeting about project alpha in June {i}",
                    sender="alice",
                    timestamp=day,
                )

        # Ensure hybrid is available for the cross-source feedback guard
        engine._has_hybrid = True
        engine._has_temporal = True

        # Patch search_fts_in_range to return a known result with a unique ID
        # that would NOT be in the main results — this lets us detect whether
        # the cross-source feedback block actually fires.
        rescoped_result = {
            "id": 99999,
            "content": "rescoped result from temporal cross-source",
            "sender": "alice",
            "recipient": "",
            "timestamp": "2026-01-15T12:00:00",
            "score": 0.8,
            "source": "fts",
        }

        with patch("truememory.vector_search.get_model", return_value=mock_model), \
             patch("truememory.fts_search.search_fts_in_range", return_value=[rescoped_result]) as mock_fts_range:
            results = engine.search("project alpha in January 2026", limit=10)

        # The cross-source feedback block should have called search_fts_in_range
        # and added the rescoped result with source="temporal_rescoped".
        # Before the fix, search_fts_in_range is never called because
        # intent.get("start_date") is always None.
        sources = [r.get("source", "") for r in results]
        result_ids = [r.get("id") for r in results]

        assert mock_fts_range.called, (
            "search_fts_in_range was never called — the temporal cross-source "
            "feedback block is not firing. This is the #466 key mismatch: "
            "engine.py checks intent.get('start_date')/intent.get('end_date') "
            "but detect_temporal_intent returns 'after'/'before'."
        )
        assert 99999 in result_ids, (
            f"Rescoped result (id=99999) not found in search results (ids={result_ids}). "
            "The temporal cross-source feedback path produced results but they were not merged."
        )
        rescoped_sources = [r.get("source", "") for r in results if r.get("id") == 99999]
        assert any("temporal_rescoped" in s for s in rescoped_sources), (
            f"Rescoped result has wrong source tag: {rescoped_sources}. "
            "Expected 'temporal_rescoped'."
        )
