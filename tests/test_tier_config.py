"""Tests for truememory.tier_config — centralized tier configuration."""

from __future__ import annotations

import os
import pytest

from truememory.tier_config import (
    TIERS,
    MODEL_DIMS,
    MODEL_TO_GROUP,
    VALID_TIER_GROUPS,
    get_tier_config,
    get_embed_model,
    get_reranker,
    get_embed_dim,
    get_embed_dim_for_model,
    get_tier_group,
    get_model_group,
    get_model_name_for_group,
    is_custom_tier,
    resolve_custom_tier,
)


# ---------------------------------------------------------------------------
# Built-in tiers
# ---------------------------------------------------------------------------


class TestBuiltinTiers:
    def test_edge_config(self):
        cfg = get_tier_config("edge")
        assert cfg["embed_model"] == "model2vec"
        assert cfg["embed_dim"] == 256
        assert cfg["tier_group"] == "edge"
        assert cfg["reranker"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_base_config(self):
        cfg = get_tier_config("base")
        assert cfg["embed_model"] == "qwen3_256"
        assert cfg["embed_dim"] == 256
        assert cfg["tier_group"] == "basepro"

    def test_pro_config(self):
        cfg = get_tier_config("pro")
        assert cfg["embed_model"] == "qwen3_256"
        assert cfg["embed_dim"] == 256
        assert cfg["tier_group"] == "basepro"

    def test_case_insensitive(self):
        assert get_tier_config("Edge")["embed_model"] == "model2vec"
        assert get_tier_config("BASE")["embed_model"] == "qwen3_256"
        assert get_tier_config("Pro")["embed_model"] == "qwen3_256"

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="Unknown tier"):
            get_tier_config("nonexistent")

    def test_get_tier_config_returns_copy(self):
        """Ensure callers cannot mutate the global TIERS dict."""
        cfg1 = get_tier_config("edge")
        cfg1["embed_dim"] = 9999
        cfg2 = get_tier_config("edge")
        assert cfg2["embed_dim"] == 256


class TestHelpers:
    def test_get_embed_model(self):
        assert get_embed_model("edge") == "model2vec"
        assert get_embed_model("base") == "qwen3_256"
        assert get_embed_model("pro") == "qwen3_256"

    def test_get_reranker(self):
        assert get_reranker("edge") == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert get_reranker("base") == "Alibaba-NLP/gte-reranker-modernbert-base"

    def test_get_embed_dim(self):
        assert get_embed_dim("edge") == 256
        assert get_embed_dim("base") == 256

    def test_get_embed_dim_for_model(self):
        assert get_embed_dim_for_model("model2vec") == 256
        assert get_embed_dim_for_model("minilm") == 384
        assert get_embed_dim_for_model("unknown_model") == 256  # default

    def test_get_tier_group(self):
        assert get_tier_group("edge") == "edge"
        assert get_tier_group("base") == "basepro"
        assert get_tier_group("pro") == "basepro"

    def test_get_model_group(self):
        assert get_model_group("model2vec") == "edge"
        assert get_model_group("qwen3_256") == "basepro"
        assert get_model_group("unknown_model") == "custom"

    def test_get_model_name_for_group(self):
        assert get_model_name_for_group("edge") == "potion-base-8M"
        assert get_model_name_for_group("basepro") == "Qwen3-Embedding-0.6B"
        assert get_model_name_for_group("custom") == ""

    def test_is_custom_tier(self):
        assert is_custom_tier("custom") is True
        assert is_custom_tier("Custom") is True
        assert is_custom_tier("edge") is False
        assert is_custom_tier("pro") is False


class TestValidGroups:
    def test_contains_custom(self):
        assert "custom" in VALID_TIER_GROUPS
        assert "edge" in VALID_TIER_GROUPS
        assert "basepro" in VALID_TIER_GROUPS


# ---------------------------------------------------------------------------
# Custom tier
# ---------------------------------------------------------------------------


class TestCustomTier:
    def test_custom_tier_requires_opt_in(self):
        """Custom tier raises ValueError without TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD=1."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_MODEL", "org/my-model")
            mp.delenv("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", raising=False)
            with pytest.raises(ValueError, match="TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD"):
                resolve_custom_tier()

    def test_custom_tier_rejects_false_opt_in(self):
        """TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD=0 must be rejected."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_MODEL", "org/my-model")
            mp.setenv("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "0")
            with pytest.raises(ValueError, match="TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD"):
                resolve_custom_tier()

    def test_custom_tier_rejects_false_string_opt_in(self):
        """TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD=false must be rejected."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_MODEL", "org/my-model")
            mp.setenv("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "false")
            with pytest.raises(ValueError, match="TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD"):
                resolve_custom_tier()

    def test_custom_tier_requires_embed_model(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "1")
            mp.delenv("TRUEMEMORY_CUSTOM_EMBED_MODEL", raising=False)
            with pytest.raises(ValueError, match="TRUEMEMORY_CUSTOM_EMBED_MODEL"):
                resolve_custom_tier()

    def test_custom_tier_valid(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "1")
            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_MODEL", "Alibaba-NLP/gte-base-en-v1.5")
            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_DIM", "768")
            mp.setenv("TRUEMEMORY_CUSTOM_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")

            cfg = resolve_custom_tier()
            assert cfg["embed_model"] == "Alibaba-NLP/gte-base-en-v1.5"
            assert cfg["embed_dim"] == 768
            assert cfg["tier_group"] == "custom"
            assert cfg["reranker"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_custom_tier_via_get_tier_config(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "1")
            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_MODEL", "org/model-name")

            cfg = get_tier_config("custom")
            assert cfg["tier_group"] == "custom"
            assert cfg["embed_model"] == "org/model-name"

    def test_custom_tier_invalid_model_id(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "1")
            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_MODEL", "../../../etc/passwd")
            with pytest.raises(ValueError, match="Invalid model ID"):
                resolve_custom_tier()

    def test_custom_tier_dim_bounds(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "1")
            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_MODEL", "org/model")
            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_DIM", "99999")
            with pytest.raises(ValueError, match="1-4096"):
                resolve_custom_tier()

    def test_custom_tier_dim_non_integer(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "1")
            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_MODEL", "org/model")
            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_DIM", "abc")
            with pytest.raises(ValueError, match="must be an integer"):
                resolve_custom_tier()

    def test_custom_tier_dim_boundaries(self):
        """Boundary values: dim=1 and dim=4096 should be valid."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD", "1")
            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_MODEL", "org/model")

            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_DIM", "1")
            cfg = resolve_custom_tier()
            assert cfg["embed_dim"] == 1

            mp.setenv("TRUEMEMORY_CUSTOM_EMBED_DIM", "4096")
            cfg = resolve_custom_tier()
            assert cfg["embed_dim"] == 4096
