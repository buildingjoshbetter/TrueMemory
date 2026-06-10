"""Tests for recall cache-TTL (issue #559).

The 5-query recall search is the main cost of the SessionStart /
UserPromptSubmit hooks. This cache stores results to a file so
subsequent calls within the TTL skip the full search pipeline.

Covers:
  - Fresh cache hit returns cached context (queries skipped)
  - Expired cache triggers full queries
  - Cache miss (no file) triggers full queries
  - New memory store invalidates the cache
  - Multi-DB isolation via cache key
  - TTL=0 disables caching
  - Corrupt cache file is handled gracefully
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from truememory.ingest.hooks import _shared


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the recall cache at an isolated temp file."""
    cache_path = tmp_path / "recall_cache.json"
    monkeypatch.setattr(_shared, "RECALL_CACHE_PATH", cache_path)
    monkeypatch.setattr(_shared, "RECALL_CACHE_TTL", 300.0)
    return cache_path


class TestCacheHit:
    """Fresh cache within TTL returns cached results."""

    def test_round_trip(self, isolated_cache):
        context = "<truememory-context>\n- Likes Python\n</truememory-context>"
        _shared.set_recall_cache(context, "", "")
        assert _shared.get_recall_cache("", "") == context

    def test_cache_returns_exact_context(self, isolated_cache):
        context = "<truememory-context>\n- Prefers dark mode\n- Lives in Austin\n</truememory-context>"
        _shared.set_recall_cache(context, "", "")
        result = _shared.get_recall_cache("", "")
        assert result == context

    def test_fresh_cache_skips_queries(self, isolated_cache):
        """When cache is fresh, get_recall_cache returns non-None."""
        _shared.set_recall_cache("cached", "", "")
        assert _shared.get_recall_cache("", "") is not None


class TestCacheExpired:
    """Expired cache returns None so full queries run."""

    def test_expired_cache_returns_none(self, isolated_cache):
        _shared.set_recall_cache("old-context", "", "")
        # Backdate the timestamp to be past TTL
        data = json.loads(isolated_cache.read_text(encoding="utf-8"))
        key = _shared._recall_cache_key("", "")
        data[key]["timestamp"] = time.time() - 400  # 400s > 300s TTL
        isolated_cache.write_text(json.dumps(data), encoding="utf-8")
        assert _shared.get_recall_cache("", "") is None

    def test_just_expired_returns_none(self, isolated_cache):
        _shared.set_recall_cache("borderline", "", "")
        data = json.loads(isolated_cache.read_text(encoding="utf-8"))
        key = _shared._recall_cache_key("", "")
        data[key]["timestamp"] = time.time() - 300  # exactly at TTL
        isolated_cache.write_text(json.dumps(data), encoding="utf-8")
        assert _shared.get_recall_cache("", "") is None

    def test_just_under_ttl_returns_cached(self, isolated_cache):
        _shared.set_recall_cache("still-fresh", "", "")
        data = json.loads(isolated_cache.read_text(encoding="utf-8"))
        key = _shared._recall_cache_key("", "")
        data[key]["timestamp"] = time.time() - 299  # 1s before TTL
        isolated_cache.write_text(json.dumps(data), encoding="utf-8")
        assert _shared.get_recall_cache("", "") == "still-fresh"


class TestCacheMiss:
    """No cache file returns None so full queries run."""

    def test_no_file_returns_none(self, isolated_cache):
        assert not isolated_cache.exists()
        assert _shared.get_recall_cache("", "") is None

    def test_wrong_key_returns_none(self, isolated_cache):
        _shared.set_recall_cache("context", "/path/a.db", "")
        assert _shared.get_recall_cache("/path/b.db", "") is None


class TestCacheInvalidation:
    """New memory store deletes the cache."""

    def test_invalidate_deletes_cache_file(self, isolated_cache):
        _shared.set_recall_cache("context", "", "")
        assert isolated_cache.exists()
        _shared.invalidate_recall_cache()
        assert not isolated_cache.exists()

    def test_invalidate_specific_db_preserves_others(self, isolated_cache):
        _shared.set_recall_cache("context-a", "/path/a.db", "")
        _shared.set_recall_cache("context-b", "/path/b.db", "")
        _shared.invalidate_recall_cache("/path/a.db", "")
        assert _shared.get_recall_cache("/path/a.db", "") is None
        assert _shared.get_recall_cache("/path/b.db", "") == "context-b"

    def test_invalidate_no_db_deletes_all(self, isolated_cache):
        _shared.set_recall_cache("context-a", "/path/a.db", "")
        _shared.set_recall_cache("context-b", "/path/b.db", "")
        _shared.invalidate_recall_cache()
        assert not isolated_cache.exists()

    def test_invalidate_nonexistent_is_noop(self, isolated_cache):
        # Should not raise
        _shared.invalidate_recall_cache()
        _shared.invalidate_recall_cache("/no/such.db")

    def test_get_returns_none_after_invalidation(self, isolated_cache):
        _shared.set_recall_cache("context", "", "")
        _shared.invalidate_recall_cache()
        assert _shared.get_recall_cache("", "") is None


class TestMultiDB:
    """Cache keys isolate different DB paths and user IDs."""

    def test_different_db_paths(self, isolated_cache):
        _shared.set_recall_cache("ctx-a", "/a.db", "")
        _shared.set_recall_cache("ctx-b", "/b.db", "")
        assert _shared.get_recall_cache("/a.db", "") == "ctx-a"
        assert _shared.get_recall_cache("/b.db", "") == "ctx-b"

    def test_different_user_ids(self, isolated_cache):
        _shared.set_recall_cache("ctx-alice", "", "alice")
        _shared.set_recall_cache("ctx-bob", "", "bob")
        assert _shared.get_recall_cache("", "alice") == "ctx-alice"
        assert _shared.get_recall_cache("", "bob") == "ctx-bob"

    def test_default_key(self, isolated_cache):
        assert _shared._recall_cache_key("", "") == "default:"
        assert _shared._recall_cache_key("/a.db", "alice") == "/a.db:alice"


class TestTTLDisabled:
    """TTL=0 disables caching entirely."""

    def test_ttl_zero_disables_set(self, isolated_cache, monkeypatch):
        monkeypatch.setattr(_shared, "RECALL_CACHE_TTL", 0)
        _shared.set_recall_cache("context", "", "")
        assert not isolated_cache.exists()

    def test_ttl_zero_disables_get(self, isolated_cache, monkeypatch):
        # Write a cache with normal TTL first
        _shared.set_recall_cache("context", "", "")
        assert isolated_cache.exists()
        # Now disable
        monkeypatch.setattr(_shared, "RECALL_CACHE_TTL", 0)
        assert _shared.get_recall_cache("", "") is None

    def test_ttl_negative_disables(self, isolated_cache, monkeypatch):
        monkeypatch.setattr(_shared, "RECALL_CACHE_TTL", -1.0)
        _shared.set_recall_cache("context", "", "")
        assert not isolated_cache.exists()
        assert _shared.get_recall_cache("", "") is None


class TestCorruptCache:
    """Corrupt cache files are handled gracefully."""

    def test_invalid_json(self, isolated_cache):
        isolated_cache.write_text("not valid json", encoding="utf-8")
        assert _shared.get_recall_cache("", "") is None

    def test_wrong_type(self, isolated_cache):
        isolated_cache.write_text('"just a string"', encoding="utf-8")
        assert _shared.get_recall_cache("", "") is None

    def test_missing_timestamp(self, isolated_cache):
        data = {"default:": {"context": "hello"}}
        isolated_cache.write_text(json.dumps(data), encoding="utf-8")
        assert _shared.get_recall_cache("", "") is None

    def test_set_overwrites_corrupt(self, isolated_cache):
        isolated_cache.write_text("garbage", encoding="utf-8")
        _shared.set_recall_cache("fresh", "", "")
        assert _shared.get_recall_cache("", "") == "fresh"


class TestEmptyContext:
    """Empty string context is cached normally (no queries returned results)."""

    def test_empty_string_cached(self, isolated_cache):
        _shared.set_recall_cache("", "", "")
        # Empty string is a valid cache entry (means 0 results last time)
        result = _shared.get_recall_cache("", "")
        assert result == ""
