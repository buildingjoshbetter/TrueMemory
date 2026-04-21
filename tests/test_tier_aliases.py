"""Tier alias resolution — v0.4.0 paper-aligned Edge/Base/Pro.

These tests assert the library-level contract only; they do NOT load the heavy
Qwen3 model. Aliases resolve to internal names and dimensions; the actual
SentenceTransformer download happens lazily in ``get_model()``.
"""
from __future__ import annotations

import pytest

from truememory import vector_search


@pytest.fixture(autouse=True)
def _restore_default_model():
    """Restore module state after each test so other suites see the default."""
    original_name = vector_search.EMBEDDING_MODEL
    original_dim = vector_search._embedding_dim
    yield
    vector_search._model = None
    vector_search.EMBEDDING_MODEL = original_name
    vector_search._embedding_dim = original_dim


def test_set_edge_resolves_to_model2vec_256d():
    vector_search.set_embedding_model("edge")
    assert vector_search.EMBEDDING_MODEL == "model2vec"
    assert vector_search.get_embedding_dim() == 256


def test_set_base_resolves_to_qwen3_256d():
    vector_search.set_embedding_model("base")
    assert vector_search.EMBEDDING_MODEL == "qwen3_256"
    assert vector_search.get_embedding_dim() == 256


def test_set_pro_resolves_to_qwen3_256d():
    vector_search.set_embedding_model("pro")
    assert vector_search.EMBEDDING_MODEL == "qwen3_256"
    assert vector_search.get_embedding_dim() == 256


def test_set_qwen3_raises_value_error():
    """Breaking change: the old internal name ``qwen3`` (1024d native) is gone."""
    with pytest.raises(ValueError, match=r"pro|qwen3_256"):
        vector_search.set_embedding_model("qwen3")


def test_qwen3_256_internal_name_works():
    vector_search.set_embedding_model("qwen3_256")
    assert vector_search.EMBEDDING_MODEL == "qwen3_256"
    assert vector_search.get_embedding_dim() == 256
