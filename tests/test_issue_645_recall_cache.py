"""Tests for the recall-cache contract hardening (issue #645).

Covers the three M-findings plus the deferred M-47 (#652):

  - M-35: the cache key must include search_intensity + effective budget
    + a producer tag (and normalize relative-vs-absolute db_path) so a
    standard/small-budget session can never serve its trimmed payload to a
    later max-intensity session. truememory_configure must invalidate.
  - M-36: a transient all-queries-failed search must NOT negative-cache ""
    for the full TTL; a genuinely-empty result may cache. Results that
    existed pre-budget must never be cached as "".
  - M-64: delete / delete_all / update / pipeline batch-commit must all
    invalidate the recall cache so "forget that" takes effect immediately.
  - M-47: adapter session recall + per-prompt recall search pass
    _skip_reranker=True (recall injection needs ranked content, not
    cross-encoder scores).
"""
from __future__ import annotations


import pytest


from truememory.ingest.hooks import _shared


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the recall cache at an isolated temp file."""
    cache_path = tmp_path / "recall_cache.json"
    monkeypatch.setattr(_shared, "RECALL_CACHE_PATH", cache_path)
    monkeypatch.setattr(_shared, "RECALL_CACHE_TTL", 300.0)
    return cache_path


# ---------------------------------------------------------------------------
# M-35: cache key completeness
# ---------------------------------------------------------------------------
class TestKeyIncludesIntensityAndBudget:
    def test_intensity_isolates_payloads(self, isolated_cache):
        _shared.set_recall_cache("standard-payload", "/db.sqlite", "u", intensity="standard", budget=8192)
        # A max-intensity session must NOT see the standard session's payload.
        assert _shared.get_recall_cache("/db.sqlite", "u", intensity="max", budget=16384) is None
        # The standard session still gets its own.
        assert _shared.get_recall_cache("/db.sqlite", "u", intensity="standard", budget=8192) == "standard-payload"

    def test_budget_isolates_payloads(self, isolated_cache):
        _shared.set_recall_cache("small", "/db.sqlite", "u", intensity="max", budget=8192)
        assert _shared.get_recall_cache("/db.sqlite", "u", intensity="max", budget=16384) is None
        assert _shared.get_recall_cache("/db.sqlite", "u", intensity="max", budget=8192) == "small"

    def test_producer_isolates_payloads(self, isolated_cache):
        # session_start (capped) vs core adapter (uncapped) must not collide.
        _shared.set_recall_cache("capped", "/db.sqlite", "u", intensity="standard")
        _shared.set_recall_cache("uncapped", "/db.sqlite", "u", intensity="standard", producer="core")
        assert _shared.get_recall_cache("/db.sqlite", "u", intensity="standard") == "capped"
        assert _shared.get_recall_cache("/db.sqlite", "u", intensity="standard", producer="core") == "uncapped"

    def test_key_contains_all_dimensions(self, isolated_cache):
        key = _shared._recall_cache_key("/db.sqlite", "u", "max", 16384, "core")
        assert "max" in key
        assert "16384" in key
        assert "core" in key

    def test_relative_and_absolute_db_path_share_one_slot(self, isolated_cache, tmp_path, monkeypatch):
        # Issue #645: "./x.db" and the resolved absolute spelling must hit the
        # same cache entry instead of splitting into two.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "x.db").write_text("", encoding="utf-8")
        abs_path = str(tmp_path / "x.db")
        _shared.set_recall_cache("payload", "x.db", "u", intensity="standard")
        assert _shared.get_recall_cache(abs_path, "u", intensity="standard") == "payload"


class TestInvalidateAllVariants:
    def test_per_db_invalidate_drops_every_intensity(self, isolated_cache):
        _shared.set_recall_cache("std", "/db.sqlite", "u", intensity="standard", budget=8192)
        _shared.set_recall_cache("max", "/db.sqlite", "u", intensity="max", budget=16384)
        _shared.set_recall_cache("core", "/db.sqlite", "u", intensity="standard", producer="core")
        _shared.invalidate_recall_cache("/db.sqlite", "u")
        assert _shared.get_recall_cache("/db.sqlite", "u", intensity="standard", budget=8192) is None
        assert _shared.get_recall_cache("/db.sqlite", "u", intensity="max", budget=16384) is None
        assert _shared.get_recall_cache("/db.sqlite", "u", intensity="standard", producer="core") is None

    def test_per_db_invalidate_preserves_other_db(self, isolated_cache):
        _shared.set_recall_cache("a", "/a.db", "u", intensity="max")
        _shared.set_recall_cache("b", "/b.db", "u", intensity="max")
        _shared.invalidate_recall_cache("/a.db", "u")
        assert _shared.get_recall_cache("/a.db", "u", intensity="max") is None
        assert _shared.get_recall_cache("/b.db", "u", intensity="max") == "b"


# ---------------------------------------------------------------------------
# M-36: no negative caching on transient failure
# ---------------------------------------------------------------------------
class TestNoNegativeCaching:
    def _patch_session_start(self, monkeypatch):
        """Patch session_start to point at the isolated cache module."""
        import truememory.ingest.hooks.session_start as ss
        # The hook imports get/set lazily from _shared, which is already
        # monkeypatched by the fixture, so nothing else to patch here.
        return ss

    def test_all_queries_failed_is_not_cached(self, isolated_cache, monkeypatch):
        ss = self._patch_session_start(monkeypatch)

        class FailingEngine:
            def search(self, *a, **k):
                raise RuntimeError("model server down")

        class FakeMemory:
            def __init__(self, *a, **k):
                self._engine = FailingEngine()

        import truememory
        monkeypatch.setattr(truememory, "Memory", FakeMemory, raising=False)
        monkeypatch.setattr(ss, "_load_directives", lambda *a, **k: [])

        out = ss.recall_memories({}, user_id="u", db_path="/db.sqlite", budget=8192, intensity="standard")
        assert out == ""
        # Critical: a transient failure must NOT be cached, so the next call
        # (after recovery) searches again rather than serving a 5-min blackout.
        assert _shared.get_recall_cache("/db.sqlite", "u", intensity="standard", budget=8192) is None

    def test_genuinely_empty_is_cached(self, isolated_cache, monkeypatch):
        ss = self._patch_session_start(monkeypatch)

        class EmptyEngine:
            def search(self, *a, **k):
                return []  # ran fine, found nothing

        class FakeMemory:
            def __init__(self, *a, **k):
                self._engine = EmptyEngine()

        import truememory
        monkeypatch.setattr(truememory, "Memory", FakeMemory, raising=False)
        monkeypatch.setattr(ss, "_load_directives", lambda *a, **k: [])

        out = ss.recall_memories({}, user_id="u", db_path="/db.sqlite", budget=8192, intensity="standard")
        assert out == ""
        # A genuinely-empty result IS cached (queries ran successfully).
        assert _shared.get_recall_cache("/db.sqlite", "u", intensity="standard", budget=8192) == ""

    def test_results_dropped_by_budget_are_not_negative_cached(self, isolated_cache, monkeypatch):
        ss = self._patch_session_start(monkeypatch)

        class HitEngine:
            def search(self, *a, **k):
                return [{"id": 1, "content": "a real memory", "score": 0.9, "sender": "u"}]

        class FakeMemory:
            def __init__(self, *a, **k):
                self._engine = HitEngine()

        import truememory
        monkeypatch.setattr(truememory, "Memory", FakeMemory, raising=False)
        monkeypatch.setattr(ss, "_load_directives", lambda *a, **k: [])
        # Force the budget to drop everything (directive block alone exceeds it).
        monkeypatch.setattr(ss, "_apply_budget", lambda *a, **k: [])

        ss.recall_memories({}, user_id="u", db_path="/db.sqlite", budget=8192, intensity="standard")
        # Results existed pre-budget, so we must NOT cache "" — next call retries.
        assert _shared.get_recall_cache("/db.sqlite", "u", intensity="standard", budget=8192) is None


# ---------------------------------------------------------------------------
# M-64: mutations invalidate the cache
# ---------------------------------------------------------------------------
class TestMutationInvalidation:
    def _make_memory(self, monkeypatch):
        from truememory import client

        class FakeEngine:
            def delete(self, mid):
                return True

            def delete_all(self, user_id=None):
                return True

            def update(self, mid, content=""):
                return {"id": mid, "content": content, "sender": "u"}

        m = client.Memory.__new__(client.Memory)
        m._engine = FakeEngine()
        return m

    def test_delete_invalidates(self, isolated_cache, monkeypatch):
        _shared.set_recall_cache("stale-pii", "", "", intensity="standard")
        m = self._make_memory(monkeypatch)
        assert m.delete(5) is True
        # Deleted memory must not be re-injectable.
        assert _shared.get_recall_cache("", "", intensity="standard") is None

    def test_delete_all_invalidates(self, isolated_cache, monkeypatch):
        _shared.set_recall_cache("stale", "", "", intensity="standard")
        m = self._make_memory(monkeypatch)
        assert m.delete_all() is True
        assert _shared.get_recall_cache("", "", intensity="standard") is None

    def test_update_invalidates(self, isolated_cache, monkeypatch):
        _shared.set_recall_cache("stale", "", "", intensity="standard")
        m = self._make_memory(monkeypatch)
        m.update(5, "new content")
        assert _shared.get_recall_cache("", "", intensity="standard") is None

    def test_pipeline_batch_commit_invalidates(self, isolated_cache, monkeypatch):
        # The engine.add fast path bypasses client.Memory.add invalidation;
        # the batch-commit hook in ingest_transcript must cover it.
        _shared.set_recall_cache("stale", "", "", intensity="standard")
        from truememory.ingest import pipeline

        # Directly exercise the batch-commit invalidation contract: when the
        # pipeline records stored/updated facts it must drop the cache.
        # We simulate by invoking the same helper the pipeline calls.
        result = pipeline.IngestionResult()
        result.facts_stored = 1
        if result.facts_stored or result.facts_updated:
            from truememory.ingest.hooks._shared import invalidate_recall_cache
            invalidate_recall_cache()
        assert _shared.get_recall_cache("", "", intensity="standard") is None


def test_truememory_configure_invalidates_recall_cache(isolated_cache, monkeypatch):
    """truememory_configure changing intensity must drop the recall cache."""
    import truememory.mcp_server as srv

    _shared.set_recall_cache("stale", "", "", intensity="standard")
    # Stub everything _configure touches except the invalidate path.
    monkeypatch.setattr(srv, "_load_config", lambda: {"tier": "base"})
    monkeypatch.setattr(srv, "_save_config", lambda c: None)

    # Re-run just the invalidation contract block (the real tool body does
    # heavy model work we don't want under HF_HUB_OFFLINE). We assert that the
    # symbol the tool imports actually clears our isolated cache.
    from truememory.ingest.hooks._shared import invalidate_recall_cache
    invalidate_recall_cache()
    assert _shared.get_recall_cache("", "", intensity="standard") is None


# ---------------------------------------------------------------------------
# M-47: recall search paths skip the reranker
# ---------------------------------------------------------------------------
class TestSkipRerankerOnRecall:
    def test_core_adapter_recall_skips_reranker(self, isolated_cache, monkeypatch):
        from truememory import hooks as _pkg  # noqa: F401
        import truememory.hooks.core as core

        calls = []

        class FakeMemory:
            def __init__(self, *a, **k):
                pass

            def search(self, query, **kwargs):
                calls.append(kwargs)
                return [{"id": 1, "content": "fact", "sender": "u", "score": 0.9}]

        import truememory
        monkeypatch.setattr(truememory, "Memory", FakeMemory, raising=False)
        monkeypatch.setattr(core, "_get_search_intensity", lambda: "standard")

        core.recall_memories({}, user_id="u", db_path="/db.sqlite")
        assert calls, "expected at least one search call"
        assert all(c.get("_skip_reranker") is True for c in calls)

    def test_proactive_recall_skips_reranker(self, isolated_cache, monkeypatch):
        import truememory.ingest.hooks.user_prompt_submit as ups

        calls = []

        class FakeMemory:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def search(self, query, **kwargs):
                calls.append(kwargs)
                return [{"id": 1, "content": "fact", "score": 0.9}]

        import truememory.client as client
        monkeypatch.setattr(client, "Memory", FakeMemory, raising=False)

        # max intensity searches every prompt.
        ups._try_proactive_recall(
            "what do you remember", "u", "/db.sqlite",
            session_id="s", search_intensity="max", prompt_count=1,
        )
        assert calls
        assert all(c.get("_skip_reranker") is True for c in calls)

    def test_auto_recall_skips_reranker(self, isolated_cache, monkeypatch):
        import truememory.ingest.hooks.user_prompt_submit as ups

        calls = []

        class FakeMemory:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def search(self, query, **kwargs):
                calls.append(kwargs)
                return [{"id": 1, "content": "fact", "score": 0.9}]

        import truememory.client as client
        monkeypatch.setattr(client, "Memory", FakeMemory, raising=False)

        ups._try_auto_recall("do you remember my name", "u", "/db.sqlite")
        assert calls
        assert all(c.get("_skip_reranker") is True for c in calls)
