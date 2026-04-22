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


# --- Regression locks for the two contracts that DEFINE the fix --------------
# These tests pin the exact source-level invariants that caused the pre-PR-#2
# bug. Without them, a future "cleanup" could silently revert the fix: the
# string-mapping tests above would still pass because they don't exercise the
# get_reranker / truememory_configure call-through paths. These two asserts
# make that silent-regression impossible.


def test_get_reranker_default_routes_via_tier_resolution():
    """Change A.3 regression lock.

    When `get_reranker` is called with `model_name=None`, its default path
    MUST resolve via `get_current_reranker_name()` — NOT the stale
    `_model_name` literal. Python-API callers (engine.search_agentic →
    rerank_with_modality_fusion → rerank → get_reranker(model_name=None))
    depend on this to reach the tier-correct reranker; reverting this line
    silently re-introduces the pre-PR-#2 bug where Base/Pro users got the
    MiniLM reranker.

    Guarded by source inspection so the check is deterministic and does not
    require loading the CrossEncoder model.
    """
    import inspect

    src = inspect.getsource(reranker.get_reranker)
    assert "model_name or get_current_reranker_name()" in src, (
        "get_reranker's default path must read "
        "`name = model_name or get_current_reranker_name()`. "
        "Reverting to `_model_name` re-introduces the Base/Pro MiniLM bug."
    )
    assert "model_name or _model_name" not in src, (
        "Stale default path detected: get_reranker(None) falls through to "
        "the _model_name literal instead of the tier-resolved name."
    )


def test_truememory_configure_propagates_tier_to_reranker_module():
    """Change B.4 regression lock.

    `truememory_configure` must call `reranker.set_active_tier(tier)` AND
    pre-load the tier's reranker via `_set_reranker(_current_reranker())`.
    Without the first call, changing tier via MCP leaves the reranker module
    stuck on the old tier. Without the second, the first post-configure
    search pays a cold-start that defeats the whole "configure is the right
    moment to warm the new model" design.

    Guarded by source inspection for the same reason as the test above —
    calling `truememory_configure` in a unit test would require mocking
    sentence_transformers, the Memory singleton, and the re-embed path. Too
    much surface for a contract check.
    """
    import inspect

    from truememory import mcp_server

    src = inspect.getsource(mcp_server.truememory_configure)
    assert "_set_active_tier(tier)" in src, (
        "truememory_configure must call set_active_tier(tier) (imported as "
        "_set_active_tier) after saving the new tier to config, so subsequent "
        "get_reranker(model_name=None) calls resolve to the new tier's reranker."
    )
    assert "_set_reranker(_current_reranker())" in src, (
        "truememory_configure must pre-load the tier's reranker so the first "
        "post-configure search does not pay a cold-start (~70ms-3s)."
    )
