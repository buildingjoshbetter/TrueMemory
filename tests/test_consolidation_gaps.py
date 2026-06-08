"""Regression tests for consolidation gaps (clustering + preferences).

Verifies that consolidate() calls cluster_messages and extract_preferences —
features that were previously only available via ingest().
"""
from __future__ import annotations

import sqlite3
import tempfile
import os
from unittest.mock import patch, MagicMock

import pytest

from truememory.engine import TrueMemoryEngine
from truememory.storage import create_db


def _make_engine_with_messages(n=20):
    """Create an engine with messages in a temp DB (no vectors needed)."""
    td = tempfile.mkdtemp()
    db = os.path.join(td, "test.db")
    conn = create_db(db)
    for i in range(n):
        conn.execute(
            "INSERT INTO messages (content, sender, recipient, timestamp, category, modality) "
            "VALUES (?, ?, '', '', '', '')",
            (f"Message {i} about topic {i % 5}", "alice" if i % 2 == 0 else "bob"),
        )
    conn.commit()
    conn.close()

    eng = TrueMemoryEngine(db_path=db)
    eng._ensure_connection()
    return eng, td


def test_consolidate_calls_cluster_messages():
    """consolidate() must call cluster_messages when clustering is available."""
    eng, td = _make_engine_with_messages()
    eng._has_vectors = True

    with patch("truememory.engine._HAS_CLUSTERING", True), \
         patch("truememory.clustering.cluster_messages", return_value=3) as mock_cm:
        result = eng.consolidate()

    assert "cluster_messages" in result
    assert "ERROR" not in result["cluster_messages"]
    assert "3 clusters" in result["cluster_messages"]
    mock_cm.assert_called_once_with(eng.conn)
    eng.close()


def test_consolidate_calls_extract_preferences():
    """consolidate() must call extract_preferences when personality is available."""
    eng, td = _make_engine_with_messages()

    with patch("truememory.engine._HAS_PERSONALITY", True), \
         patch("truememory.personality.extract_preferences", return_value={}) as mock_ep:
        result = eng.consolidate()

    assert "extract_preferences" in result
    assert "ERROR" not in result["extract_preferences"]
    mock_ep.assert_called_once_with(eng.conn)
    eng.close()


def test_consolidate_skips_clustering_without_vectors():
    """consolidate() must skip clustering when vectors are unavailable."""
    eng, td = _make_engine_with_messages()
    eng._has_vectors = False

    result = eng.consolidate()
    assert "cluster_messages" not in result
    eng.close()


def test_consolidate_handles_cluster_error_gracefully():
    """consolidate() must catch and report clustering errors."""
    eng, td = _make_engine_with_messages()
    eng._has_vectors = True

    with patch("truememory.engine._HAS_CLUSTERING", True), \
         patch("truememory.clustering.cluster_messages", side_effect=RuntimeError("boom")):
        result = eng.consolidate()

    assert "cluster_messages" in result
    assert "ERROR" in result["cluster_messages"]
    eng.close()


def test_auto_consolidation_threshold_lowered():
    """Default auto-consolidation threshold is 25 (was 100)."""
    eng, td = _make_engine_with_messages()
    assert eng._auto_consolidate_threshold == 25
    eng.close()


def test_startup_consolidation_triggers_when_clusters_empty():
    """On startup, if messages exist but clusters are empty, consolidate fires."""
    eng, td = _make_engine_with_messages(n=30)

    with patch.object(eng, "_bg_consolidate") as mock_bg:
        eng._has_consolidation = True
        eng._maybe_startup_consolidate()

    assert mock_bg.called or eng._consolidation_thread is not None
    eng.close()


def test_startup_consolidation_skips_when_few_messages():
    """Don't consolidate on startup if fewer messages than threshold."""
    eng, td = _make_engine_with_messages(n=5)

    consolidate_called = False
    orig = eng._bg_consolidate
    def spy():
        nonlocal consolidate_called
        consolidate_called = True
        return orig()

    eng._bg_consolidate = spy
    eng._has_consolidation = True
    eng._maybe_startup_consolidate()
    assert not consolidate_called
    eng.close()
