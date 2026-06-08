"""Regression tests for config-group issues #502–#505.

#502 (M5): _load_config() reads disk on every search — 2x JSON parse per query.
           Cache the config in memory with a timestamp; only re-read after staleness.

#503 (M6): Config write is not atomic — a crash during write can corrupt config.json.
           Use atomic write pattern: write to temp file, then os.replace().

#504 (M7): Config write race between MCP and tier switch.
           Use fcntl.flock() or threading.Lock around config writes.

#505 (M8): truememory_configure reports ``hyde_search: enabled`` regardless of tier.
           Should check whether the tier actually supports HyDE (Pro only).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import pytest


@pytest.fixture
def server(monkeypatch, tmp_path):
    """Provide an isolated mcp_server module with a fresh temp config dir."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".truememory").mkdir()
    db_path = tmp_path / "memories.db"
    monkeypatch.setenv("TRUEMEMORY_DB", str(db_path))
    monkeypatch.setenv("TRUEMEMORY_EMBED_MODEL", "edge")
    import truememory.mcp_server as ms

    monkeypatch.setattr(ms, "_TRUEMEMORY_DIR", home / ".truememory")
    monkeypatch.setattr(ms, "_CONFIG_PATH", home / ".truememory" / "config.json")
    monkeypatch.setattr(ms, "_DB_PATH", str(db_path))
    monkeypatch.setattr(ms, "_memory", None)
    # Reset config cache so each test starts clean
    ms._config_cache = None
    ms._config_cache_mtime = 0.0
    ms._config_cache_time = 0.0
    yield ms
    if ms._memory is not None:
        try:
            ms._memory.close()
        except Exception:
            pass
    ms._memory = None
    import truememory.vector_search as vs

    vs.set_embedding_model("edge")


def _no_op_model(server, monkeypatch):
    """Stub out model/reranker side effects so the test stays unit-level."""
    import truememory.vector_search as vs
    import truememory.reranker as rr

    monkeypatch.setattr(vs, "set_embedding_model", lambda tier: None)
    monkeypatch.setattr(rr, "set_active_tier", lambda tier: None)
    monkeypatch.setattr(server, "_set_reranker", lambda name: None)


# -----------------------------------------------------------------------
# M5 (#502) — Config caching
# -----------------------------------------------------------------------


class TestIssue502ConfigCache:
    """_load_config() should cache in memory and not re-read disk every call."""

    def test_load_config_returns_cached_copy(self, server, monkeypatch):
        """Two rapid _load_config() calls should return the same data without
        re-reading from disk the second time when mtime hasn't changed."""
        cfg = {"tier": "base", "anthropic_api_key": "sk-test-12345"}
        server._CONFIG_PATH.write_text(json.dumps(cfg), encoding="utf-8")
        # Reset cache
        server._config_cache = None
        server._config_cache_mtime = 0.0
        server._config_cache_time = 0.0

        result1 = server._load_config()

        # Second call within staleness window, file NOT modified on disk
        # (mtime unchanged) — should use cache, no disk read
        read_count = 0
        original_read_text = type(server._CONFIG_PATH).read_text

        def counting_read(self_path, *args, **kwargs):
            nonlocal read_count
            read_count += 1
            return original_read_text(self_path, *args, **kwargs)

        monkeypatch.setattr(type(server._CONFIG_PATH), "read_text", counting_read)

        result2 = server._load_config()
        assert result1 == result2, (
            "_load_config() did not return cached result within staleness window"
        )
        assert read_count == 0, (
            f"_load_config() re-read disk {read_count} times within staleness window"
        )
        assert result1["tier"] == "base"

    def test_load_config_refreshes_after_staleness(self, server, monkeypatch):
        """After the staleness window, _load_config() should re-read disk."""
        cfg = {"tier": "base"}
        server._CONFIG_PATH.write_text(json.dumps(cfg), encoding="utf-8")
        server._config_cache = None
        server._config_cache_mtime = 0.0
        server._config_cache_time = 0.0

        result1 = server._load_config()
        assert result1["tier"] == "base"

        # Write a new config
        server._CONFIG_PATH.write_text(
            json.dumps({"tier": "pro"}), encoding="utf-8"
        )
        # Force staleness by backdating the cache time
        server._config_cache_time = time.monotonic() - 10.0
        server._config_cache_mtime = 0.0  # force mtime mismatch

        result2 = server._load_config()
        assert result2["tier"] == "pro", (
            "_load_config() did not refresh after staleness window expired"
        )


# -----------------------------------------------------------------------
# M6 (#503) — Atomic config write
# -----------------------------------------------------------------------


class TestIssue503AtomicWrite:
    """_save_config() must use atomic write (temp file + os.replace)."""

    def test_save_config_is_atomic(self, server, monkeypatch):
        """After _save_config(), the file should contain valid JSON even if
        we simulate observing it mid-write. We verify by checking that
        os.replace (or os.rename) is used instead of direct write."""
        calls = []
        original_replace = os.replace

        def tracking_replace(src, dst):
            calls.append((src, dst))
            return original_replace(src, dst)

        monkeypatch.setattr(os, "replace", tracking_replace)

        server._save_config({"tier": "base"})
        assert len(calls) >= 1, (
            "_save_config() did not use os.replace() for atomic write"
        )
        # The destination should be the config path
        assert str(calls[0][1]) == str(server._CONFIG_PATH)

    def test_save_config_produces_valid_json(self, server):
        """After _save_config(), the file must contain valid JSON."""
        server._save_config({"tier": "pro", "anthropic_api_key": "sk-test"})
        data = json.loads(server._CONFIG_PATH.read_text(encoding="utf-8"))
        assert data["tier"] == "pro"
        assert data["anthropic_api_key"] == "sk-test"

    def test_save_config_no_partial_write_on_crash(self, server, monkeypatch):
        """If the write to temp file fails, the original config should survive."""
        # Write initial config
        server._save_config({"tier": "base"})

        # Make json.dump raise an error to simulate a crash during write.
        # _save_config uses os.fdopen + json.dump, so we intercept json.dump.
        original_json_dump = json.dump
        call_count = 0

        def failing_dump(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 0:
                raise OSError("Simulated disk full")
            return original_json_dump(*args, **kwargs)

        monkeypatch.setattr(json, "dump", failing_dump)

        with pytest.raises(OSError, match="Simulated disk full"):
            server._save_config({"tier": "pro", "corrupt": True})

        # Restore json.dump so we can read the file
        monkeypatch.setattr(json, "dump", original_json_dump)

        # Original config should still be intact
        data = json.loads(server._CONFIG_PATH.read_text(encoding="utf-8"))
        assert data["tier"] == "base"
        assert "corrupt" not in data


# -----------------------------------------------------------------------
# M7 (#504) — Config write lock
# -----------------------------------------------------------------------


class TestIssue504WriteLock:
    """Concurrent _save_config() calls must be serialized with a lock."""

    def test_concurrent_saves_are_serialized(self, server, monkeypatch):
        """Two threads calling _save_config() simultaneously must not
        interleave their writes — the final file must contain one complete
        config, not a mix."""
        results = []
        barrier = threading.Barrier(2)

        def writer(tier_name, idx):
            barrier.wait()
            server._save_config({"tier": tier_name, "writer": idx})
            results.append(idx)

        t1 = threading.Thread(target=writer, args=("base", 1))
        t2 = threading.Thread(target=writer, args=("pro", 2))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # File must be valid JSON (no interleaved writes)
        data = json.loads(server._CONFIG_PATH.read_text(encoding="utf-8"))
        assert data["tier"] in ("base", "pro"), (
            f"Config file is corrupt after concurrent writes: {data}"
        )
        # Both writers must have completed
        assert len(results) == 2

    def test_save_config_lock_attribute_exists(self, server):
        """The module should have a config write lock."""
        assert hasattr(server, "_config_write_lock"), (
            "_config_write_lock not found — M7 lock not implemented"
        )


# -----------------------------------------------------------------------
# M8 (#505) — HyDE status should reflect tier, not just key presence
# -----------------------------------------------------------------------


class TestIssue505HydeTierCheck:
    """truememory_configure must only report hyde_search=enabled for Pro tier."""

    def test_hyde_disabled_for_edge_with_key(self, server, monkeypatch):
        """Edge tier should report hyde_search as disabled even when an API key
        is present, because Edge does not support HyDE."""
        _no_op_model(server, monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key-12345")

        result = json.loads(server.truememory_configure(tier="edge"))
        assert result["status"] == "configured"
        assert "enabled" not in result["hyde_search"], (
            f"Edge tier should not report hyde_search=enabled: {result['hyde_search']}"
        )

    def test_hyde_disabled_for_base_with_key(self, server, monkeypatch):
        """Base tier should report hyde_search as disabled even when an API key
        is present, because Base does not support HyDE."""
        _no_op_model(server, monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key-12345")

        result = json.loads(server.truememory_configure(tier="base"))
        assert result["status"] == "configured"
        assert "enabled" not in result["hyde_search"], (
            f"Base tier should not report hyde_search=enabled: {result['hyde_search']}"
        )

    def test_hyde_enabled_for_pro_with_key(self, server, monkeypatch):
        """Pro tier with an API key should report hyde_search=enabled."""
        _no_op_model(server, monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key-12345")

        result = json.loads(server.truememory_configure(tier="pro"))
        assert result["status"] == "configured"
        assert result["hyde_search"] == "enabled", (
            f"Pro tier with API key should report hyde_search=enabled: {result['hyde_search']}"
        )

    def test_hyde_disabled_for_pro_without_key(self, server, monkeypatch):
        """Pro tier without an API key should report hyde_search as disabled."""
        _no_op_model(server, monkeypatch)
        # Ensure no API keys in env
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = json.loads(server.truememory_configure(tier="pro"))
        assert result["status"] == "configured"
        assert "enabled" not in result["hyde_search"], (
            f"Pro tier without API key should not report hyde_search=enabled: {result['hyde_search']}"
        )
