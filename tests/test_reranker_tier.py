"""Reranker tier resolution — v0.4.0 paper-aligned Edge/Base/Pro.

Locks the contract that:
  - Edge tier → cross-encoder/ms-marco-MiniLM-L-6-v2 (22M, CPU-friendly)
  - Base tier → Alibaba-NLP/gte-reranker-modernbert-base (149M, GPU recommended)
  - Pro  tier → Alibaba-NLP/gte-reranker-modernbert-base (149M, GPU recommended)
  - Unknown / empty tier → Edge default (safe fallback)

These tests do NOT load the actual CrossEncoder model. Everything here is
string mapping + the set_active_tier / get_current_reranker_name plumbing.
The singleton lives in truememory.reranker; the autouse fixture resets the
_active_tier cache after each test so state does not leak.
"""
from __future__ import annotations

import pytest

from truememory import reranker
from truememory.reranker import (
    get_current_reranker_name,
    get_reranker_name_for_tier,
    set_active_tier,
)


EDGE_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GTE_RERANKER = "Alibaba-NLP/gte-reranker-modernbert-base"


@pytest.fixture(autouse=True)
def _reset_active_tier():
    """Restore module state after each test so suites do not leak state."""
    original = reranker._active_tier
    yield
    reranker._active_tier = original


# --- Pure string mapping -----------------------------------------------------


def test_edge_tier_maps_to_minilm():
    assert get_reranker_name_for_tier("edge") == EDGE_RERANKER


def test_base_tier_maps_to_gte_reranker_modernbert():
    assert get_reranker_name_for_tier("base") == GTE_RERANKER


def test_pro_tier_maps_to_gte_reranker_modernbert():
    assert get_reranker_name_for_tier("pro") == GTE_RERANKER


def test_unknown_tier_falls_back_to_edge_default():
    assert get_reranker_name_for_tier("somethingelse") == EDGE_RERANKER


def test_empty_tier_falls_back_to_edge_default():
    assert get_reranker_name_for_tier("") == EDGE_RERANKER


def test_tier_mapping_is_case_insensitive():
    assert get_reranker_name_for_tier("BASE") == GTE_RERANKER
    assert get_reranker_name_for_tier("Pro") == GTE_RERANKER
    assert get_reranker_name_for_tier("Edge") == EDGE_RERANKER


# --- set_active_tier / get_current_reranker_name plumbing --------------------


def test_set_active_tier_pro_makes_current_resolve_to_gte():
    set_active_tier("pro")
    assert get_current_reranker_name() == GTE_RERANKER


def test_set_active_tier_base_makes_current_resolve_to_gte():
    set_active_tier("base")
    assert get_current_reranker_name() == GTE_RERANKER


def test_set_active_tier_edge_makes_current_resolve_to_minilm():
    set_active_tier("edge")
    assert get_current_reranker_name() == EDGE_RERANKER


def test_set_active_tier_empty_falls_back_to_edge():
    set_active_tier("")
    assert get_current_reranker_name() == EDGE_RERANKER


def test_set_active_tier_unknown_falls_back_to_edge():
    set_active_tier("something_weird")
    assert get_current_reranker_name() == EDGE_RERANKER


# --- Lazy resolution from env var (cold-start path) --------------------------


def test_cold_start_env_var_base_resolves_to_gte(monkeypatch):
    reranker._active_tier = None
    monkeypatch.setenv("TRUEMEMORY_EMBED_MODEL", "base")
    assert get_current_reranker_name() == GTE_RERANKER


def test_cold_start_env_var_pro_resolves_to_gte(monkeypatch):
    reranker._active_tier = None
    monkeypatch.setenv("TRUEMEMORY_EMBED_MODEL", "pro")
    assert get_current_reranker_name() == GTE_RERANKER


def test_cold_start_env_var_edge_resolves_to_minilm(monkeypatch):
    reranker._active_tier = None
    monkeypatch.setenv("TRUEMEMORY_EMBED_MODEL", "edge")
    assert get_current_reranker_name() == EDGE_RERANKER
