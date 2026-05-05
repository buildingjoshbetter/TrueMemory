import os
import pytest
from truememory import config
from truememory.vector_search import _resolve_model_name
from truememory.reranker import get_reranker_name_for_tier

def test_custom_tier_env_vars(monkeypatch):
    monkeypatch.setenv("TRUEMEMORY_CUSTOM_EMBED_MODEL", "custom-embed")
    monkeypatch.setenv("TRUEMEMORY_CUSTOM_RERANKER", "custom-reranker")
    monkeypatch.setenv("TRUEMEMORY_CUSTOM_EMBED_DIM", "512")
    
    conf = config.get_tier_config("custom")
    assert conf["embed_model"] == "custom-embed"
    assert conf["reranker"] == "custom-reranker"
    assert conf["embed_dim"] == 512

def test_custom_tier_resolution(monkeypatch):
    monkeypatch.setenv("TRUEMEMORY_EMBED_MODEL", "custom")
    monkeypatch.setenv("TRUEMEMORY_CUSTOM_EMBED_MODEL", "my-model")
    monkeypatch.setenv("TRUEMEMORY_CUSTOM_RERANKER", "my-reranker")
    
    assert _resolve_model_name("custom") == "my-model"
    assert get_reranker_name_for_tier("custom") == "my-reranker"

def test_standard_tiers():
    assert config.get_tier_config("edge")["embed_model"] == "model2vec"
    assert config.get_tier_config("base")["embed_model"] == "qwen3_256"
    assert config.get_tier_config("pro")["embed_model"] == "qwen3_256"

def test_fallback_logic():
    # Unknown tier falls back to edge
    assert config.get_tier_config("unknown")["embed_model"] == "model2vec"
    
def test_embedding_dim_resolution():
    assert config.get_embedding_dim("model2vec") == 256
    assert config.get_embedding_dim("minilm") == 384
    
    # Tier name resolution
    assert config.get_embedding_dim("edge") == 256
    assert config.get_embedding_dim("base") == 256
