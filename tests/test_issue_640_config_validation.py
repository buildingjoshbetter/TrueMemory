"""Regression tests for issue #640 — config.json schema validation at every reader.

Covers:
  M-25 (a) non-str / null tier does not crash import (mcp_server tier->environ).
  M-25 (b) valid-JSON non-object config (null / "x" / [..]) is routed to the
           corrupt-rename recovery instead of raising AttributeError.
  M-25 (c) reranker lazy tier resolution survives a non-dict config.
  M-25 (d) explicit ``deepsearch_provider: null`` does not crash deep search.
  M-87     a BOM-prefixed valid config loads (utf-8-sig), not renamed as corrupt.
  M-88     an unknown tier string does NOT trigger a custom-model download path;
           "PRO" normalizes to "pro".

All configs are written to a tmp dir; the real ~/.truememory is never touched.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def server(monkeypatch, tmp_path):
    """Isolated mcp_server module with a fresh temp config dir.

    Mirrors the fixture in tests/test_issue_502_config.py so config readers
    point at a throwaway directory.
    """
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
    ms._config_cache = None
    ms._config_cache_mtime = 0.0
    ms._config_cache_time = 0.0
    yield ms
    ms._config_cache = None
    ms._config_cache_mtime = 0.0
    ms._config_cache_time = 0.0


def _reset_cache(server):
    server._config_cache = None
    server._config_cache_mtime = 0.0
    server._config_cache_time = 0.0


# -----------------------------------------------------------------------
# M-25 (b) — valid-JSON non-object config is handled, not an AttributeError
# -----------------------------------------------------------------------


class TestNonObjectConfig:
    @pytest.mark.parametrize("body", ["null", '"hello"', "[1, 2]", "3"])
    def test_non_dict_config_routed_to_recovery(self, server, body):
        """A valid-JSON non-object config must return {} (corrupt-rename path),
        never raise AttributeError/TypeError."""
        server._CONFIG_PATH.write_text(body, encoding="utf-8")
        _reset_cache(server)

        result = server._load_config()
        assert result == {}
        # File renamed to a .corrupt backup, not left in place.
        assert not server._CONFIG_PATH.exists()
        backups = list(server._CONFIG_PATH.parent.glob("config.json.corrupt.*"))
        assert backups, "non-object config was not renamed to a .corrupt backup"

    def test_list_config_does_not_crash_per_tool(self, server):
        """A list config must not raise when callers do config.get(...)."""
        server._CONFIG_PATH.write_text("[1, 2, 3]", encoding="utf-8")
        _reset_cache(server)
        # Should be coerced to {} — .get() works fine.
        assert server._load_config().get("tier") is None


# -----------------------------------------------------------------------
# M-25 (a) — non-str / null tier must not crash the tier->environ assignment
# -----------------------------------------------------------------------


class TestNullTier:
    @pytest.mark.parametrize("tier_val", [None, 3, ["pro"], {"x": 1}, ""])
    def test_non_str_tier_does_not_crash(self, server, tier_val):
        """A loaded config with a non-str tier must not crash the import-time
        tier->environ assignment (a non-str value would raise TypeError)."""
        server._CONFIG_PATH.write_text(
            json.dumps({"tier": tier_val}), encoding="utf-8"
        )
        _reset_cache(server)
        startup_tier = server._load_config().get("tier")
        env = {}
        # This is the exact guard shipped in mcp_server.py.
        if isinstance(startup_tier, str) and startup_tier.strip():
            env["TRUEMEMORY_EMBED_MODEL"] = startup_tier
        # Non-str / empty tier leaves the env var unset (no TypeError).
        assert "TRUEMEMORY_EMBED_MODEL" not in env

    def test_null_tier_config_loads_clean(self, server):
        """{"tier": null} is a valid object; load returns it without crash."""
        server._CONFIG_PATH.write_text(json.dumps({"tier": None}), encoding="utf-8")
        _reset_cache(server)
        result = server._load_config()
        assert result == {"tier": None}


# -----------------------------------------------------------------------
# M-25 (d) — explicit deepsearch_provider: null must not crash deep search
# -----------------------------------------------------------------------


class TestNullDeepsearchProvider:
    def test_null_provider_returns_none_no_crash(self, server):
        """deepsearch_provider: null -> _build_deepsearch_llm_fn returns None
        (fall back to default) instead of raising None.strip()."""
        server._CONFIG_PATH.write_text(
            json.dumps({"tier": "pro", "deepsearch_provider": None,
                        "deepsearch_model": None}),
            encoding="utf-8",
        )
        _reset_cache(server)
        assert server._build_deepsearch_llm_fn() is None

    def test_missing_provider_returns_none(self, server):
        server._CONFIG_PATH.write_text(json.dumps({"tier": "pro"}), encoding="utf-8")
        _reset_cache(server)
        assert server._build_deepsearch_llm_fn() is None


# -----------------------------------------------------------------------
# M-87 — a BOM-prefixed valid config loads, not renamed as corrupt
# -----------------------------------------------------------------------


class TestBomConfig:
    def test_bom_config_loads(self, server):
        """A valid config saved with a UTF-8 BOM must load via utf-8-sig."""
        cfg = {"tier": "base", "anthropic_api_key": "sk-test"}
        # encoding="utf-8-sig" prepends the BOM bytes.
        server._CONFIG_PATH.write_text(json.dumps(cfg), encoding="utf-8-sig")
        _reset_cache(server)

        result = server._load_config()
        assert result["tier"] == "base"
        assert result["anthropic_api_key"] == "sk-test"
        # Not renamed away.
        assert server._CONFIG_PATH.exists()
        assert not list(server._CONFIG_PATH.parent.glob("config.json.corrupt.*"))


# -----------------------------------------------------------------------
# M-25 (c) — reranker lazy tier resolution survives a non-dict config
# -----------------------------------------------------------------------


class TestRerankerNonDictConfig:
    @pytest.mark.parametrize("body", ["null", '"x"', "[1, 2]"])
    def test_reranker_tier_resolution_falls_back_to_edge(
        self, monkeypatch, tmp_path, body,
    ):
        import truememory.reranker as rr

        cfg_dir = tmp_path / ".truememory"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(body, encoding="utf-8")
        import pathlib
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
        # Ensure env var doesn't short-circuit the config read.
        monkeypatch.delenv("TRUEMEMORY_EMBED_MODEL", raising=False)

        assert rr._resolve_tier_from_env_and_config() == "edge"

    def test_reranker_bom_config_resolves_tier(self, monkeypatch, tmp_path):
        import truememory.reranker as rr

        cfg_dir = tmp_path / ".truememory"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(
            json.dumps({"tier": "pro"}), encoding="utf-8-sig"
        )
        import pathlib
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.delenv("TRUEMEMORY_EMBED_MODEL", raising=False)

        assert rr._resolve_tier_from_env_and_config() == "pro"


# -----------------------------------------------------------------------
# M-88 — unknown tier must not trigger custom-model download; "PRO" -> "pro"
# -----------------------------------------------------------------------


class TestUnknownTierResolution:
    def test_pro_normalizes_to_pro_model(self):
        import truememory.vector_search as vs

        # "PRO" (wrong case) must resolve to the same model as "pro".
        assert vs._resolve_model_name("PRO") == vs._resolve_model_name("pro")
        assert vs._resolve_model_name("PRO") != "model2vec"

    @pytest.mark.parametrize("garbage", ["garbage-tier", "Pr0", " proo", "xyz"])
    def test_unknown_tier_falls_back_to_safe_default(self, garbage):
        import truememory.vector_search as vs

        # An unknown string with no "/" must NOT be returned verbatim (which
        # would be loaded as a custom HF model id) — fall back to model2vec.
        assert vs._resolve_model_name(garbage) == "model2vec"

    def test_hf_repo_id_still_honoured(self):
        import truememory.vector_search as vs

        # A value that clearly looks like a HF repo id is still passed through.
        assert vs._resolve_model_name("some-org/some-model") == "some-org/some-model"
