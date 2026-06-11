"""Tests for issue #635 — per-exchange store coordination.

Three clusters in ``truememory/ingest/hooks/user_prompt_submit.py``:

- M-39 (P2): the per-exchange store path used a bare ``m.add()``, bypassing
  the pipeline's ``_dedup_store_lock`` + ``check_duplicate`` critical section,
  so a fact stored per-exchange and later re-extracted (rephrased) by the Stop
  pipeline got double-stored. We assert the store now routes through
  ``check_duplicate`` and SKIPs when dedup says the fact already exists.
- M-40 (P2): ``_STORABLE_RE`` flagged bare filler ("no", "actually", "I'm")
  and questions/quotes. We assert it does NOT fire on those but still fires on
  genuine preference/fact statements.
- M-73 (P3): conversation-depth buffer-dir derivation + unlocked counter RMW.
  We assert depth reads from the same dir buffer_message() writes, and the
  counter increments monotonically.

All tests mock the model/embedding layer (no real model loads).
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ---------------------------------------------------------------------------
# Fakes — fully model-free (mirrors test_issue_634_per_exchange_store.py)
# ---------------------------------------------------------------------------

class _FakeDecision:
    def __init__(self, should_encode: bool):
        self.should_encode = should_encode


class _FakeGate:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, fact, category=""):
        return _FakeDecision(True)


class _FakeMemory:
    """Records add() calls; search()/search_vectors() return no dupes by default."""

    instances: list["_FakeMemory"] = []

    def __init__(self, path=None, **kwargs):
        self.added: list[str] = []
        self.closed = False
        self.search_results: list[dict] = []
        self.vector_results: list[dict] = []
        _FakeMemory.instances.append(self)

    def search(self, query, user_id=None, limit=10, **kwargs):
        return list(self.search_results)

    def search_vectors(self, query, limit=3, **kwargs):
        return list(self.vector_results)

    def add(self, content, user_id=None, **kwargs):
        self.added.append(content)
        return {"id": len(self.added), "content": content, "user_id": user_id or ""}

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    _FakeMemory.instances.clear()
    yield
    _FakeMemory.instances.clear()


@pytest.fixture
def patched(monkeypatch):
    """Patch the model-touching deps in the hook with fakes."""
    import truememory.client as client
    import truememory.ingest.encoding_gate as eg
    import truememory.model_client as mc

    monkeypatch.setattr(client, "Memory", _FakeMemory)
    monkeypatch.setattr(eg, "EncodingGate", _FakeGate)
    monkeypatch.setattr(mc, "set_request_timeout", lambda v: None)
    return {"FakeMemory": _FakeMemory}


# ---------------------------------------------------------------------------
# M-39: per-exchange store routes through check_duplicate / dedup lock
# ---------------------------------------------------------------------------

class TestPerExchangeRoutesThroughDedup:
    def test_store_calls_check_duplicate(self, patched, monkeypatch):
        """The store path must invoke the pipeline's check_duplicate."""
        import truememory.ingest.dedup as dedup
        from truememory.ingest.dedup import DedupAction, DedupDecision

        calls: list[str] = []

        def fake_check(fact, memory, **kwargs):
            calls.append(fact)
            return DedupDecision(action=DedupAction.ADD, fact=fact, reason="new")

        monkeypatch.setattr(dedup, "check_duplicate", fake_check)

        from truememory.ingest.hooks import user_prompt_submit as ups
        ups._try_per_exchange_store(
            "I prefer dark mode for all my editors and always use TypeScript",
            "sess-639", "", "", "max",
        )
        assert calls, "store path must call check_duplicate"
        adds = [c for m in patched["FakeMemory"].instances for c in m.added]
        assert len(adds) == 1

    def test_dedup_skip_prevents_double_store(self, patched, monkeypatch):
        """When dedup says SKIP (pipeline already has the fact), do not add."""
        import truememory.ingest.dedup as dedup
        from truememory.ingest.dedup import DedupAction, DedupDecision

        def fake_check(fact, memory, **kwargs):
            return DedupDecision(action=DedupAction.SKIP, fact=fact, reason="dupe")

        monkeypatch.setattr(dedup, "check_duplicate", fake_check)

        from truememory.ingest.hooks import user_prompt_submit as ups
        ups._try_per_exchange_store(
            "I prefer dark mode for all my editors and always use TypeScript",
            "sess-skip", "", "", "max",
        )
        adds = [c for m in patched["FakeMemory"].instances for c in m.added]
        assert adds == [], "a fact dedup marks as duplicate must not be re-stored"

    def test_dedup_lock_is_held(self, patched, monkeypatch):
        """The store must run inside the pipeline's _dedup_store_lock."""
        import truememory.ingest.pipeline as pipeline
        import truememory.ingest.dedup as dedup
        from truememory.ingest.dedup import DedupAction, DedupDecision
        import contextlib

        entered = {"lock": False}

        @contextlib.contextmanager
        def fake_lock():
            entered["lock"] = True
            yield

        monkeypatch.setattr(pipeline, "_dedup_store_lock", fake_lock)
        monkeypatch.setattr(
            dedup, "check_duplicate",
            lambda fact, memory, **kw: DedupDecision(
                action=DedupAction.ADD, fact=fact, reason="new"),
        )

        from truememory.ingest.hooks import user_prompt_submit as ups
        ups._try_per_exchange_store(
            "I always use TypeScript over JavaScript for new projects",
            "sess-lock", "", "", "max",
        )
        assert entered["lock"], "store must be wrapped in _dedup_store_lock"


# ---------------------------------------------------------------------------
# M-40: _STORABLE_RE / _detect_storable_content false positives
# ---------------------------------------------------------------------------

class TestStorableDetection:
    @pytest.mark.parametrize("text", [
        "no",
        "no.",
        "actually can you run the tests again",
        "do you prefer tabs or spaces?",
        "what is my favorite color?",
        'the error said "I\'m deprecated" in the log',
        "instead of that, what should we do?",
    ])
    def test_filler_and_questions_do_not_fire(self, text):
        from truememory.ingest.hooks import user_prompt_submit as ups
        assert ups._detect_storable_content(text) is False, f"should NOT fire: {text!r}"

    @pytest.mark.parametrize("text", [
        "I prefer dark mode for all my editors",
        "I always use TypeScript over JavaScript",
        "my email is josh@example.com for the account",
        "actually, I use bun instead of npm now",
        "I'm a founder based in Austin Texas",
        "we decided to ship the feature on Friday",
        "remember that I hate trailing whitespace",
    ])
    def test_genuine_statements_fire(self, text):
        from truememory.ingest.hooks import user_prompt_submit as ups
        assert ups._detect_storable_content(text) is True, f"should fire: {text!r}"

    def test_bare_im_does_not_fire(self):
        from truememory.ingest.hooks import user_prompt_submit as ups
        # bare "I'm" with no identity/state predicate is filler.
        assert ups._detect_storable_content("I'm not sure, can you check?") is False


# ---------------------------------------------------------------------------
# M-73: buffer-dir derivation + counter lock
# ---------------------------------------------------------------------------

class TestBufferDirAndCounter:
    def test_depth_reads_custom_buffer_dir(self, tmp_path, monkeypatch):
        """conversation depth must read the buffer from BUFFER_DIR, even when
        TRUEMEMORY_BUFFER_DIR is not a ``.../buffers`` path."""
        import importlib
        custom = tmp_path / "custom_dir"
        monkeypatch.setenv("TRUEMEMORY_BUFFER_DIR", str(custom))
        import truememory.ingest.hooks.user_prompt_submit as ups
        importlib.reload(ups)
        try:
            sid = "sess-depth"
            # Seed prior on-topic buffer entries via the module's own writer.
            for _ in range(3):
                ups.buffer_message(sid, "we are building the truememory ingestion pipeline today")
            # Current prompt shares >=3 meaningful words with prior entries.
            ups.buffer_message(sid, "the truememory ingestion pipeline needs building")
            assert ups._check_conversation_depth(
                sid, "the truememory ingestion pipeline needs building") is True
        finally:
            monkeypatch.delenv("TRUEMEMORY_BUFFER_DIR", raising=False)
            importlib.reload(ups)

    def test_counter_increments_monotonically(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv("TRUEMEMORY_BUFFER_DIR", str(tmp_path / "buffers"))
        import truememory.ingest.hooks.user_prompt_submit as ups
        importlib.reload(ups)
        try:
            sid = "sess-counter"
            seen = [ups._increment_prompt_count(sid) for _ in range(5)]
            assert seen == [1, 2, 3, 4, 5]
            assert ups._get_prompt_count(sid) == 5
        finally:
            monkeypatch.delenv("TRUEMEMORY_BUFFER_DIR", raising=False)
            importlib.reload(ups)
