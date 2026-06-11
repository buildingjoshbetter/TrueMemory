"""Tests for issue #634 — per-exchange store DOA.

Three regressions in ``truememory/ingest/hooks/user_prompt_submit.py``:

- M-04 (P0): the store path read ``decision.passed`` but ``EncodingDecision``
  only has ``should_encode``. The AttributeError was swallowed by a blanket
  ``except``, so enhanced/max stored ZERO rows. We assert a real row lands.
- M-17 (P1): ``_check_conversation_depth`` read the last buffer entries
  INCLUDING the just-buffered current prompt, so "enhanced" always matched
  itself and behaved identically to "max". We assert enhanced does NOT fire
  on a shallow/first-prompt exchange that max would store.
- M-18 (P1): the store path now arms the recall deadline before its model
  searches and closes the Memory instance. We assert the deadline is set.

All tests mock the model/embedding layer (no real model loads).
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ---------------------------------------------------------------------------
# Fakes — fully model-free
# ---------------------------------------------------------------------------

class _FakeDecision:
    """Mimics EncodingDecision: has should_encode, NOT passed."""

    def __init__(self, should_encode: bool):
        self.should_encode = should_encode


class _FakeGate:
    def __init__(self, *args, **kwargs):
        # default: encode everything
        pass

    def evaluate(self, fact, category=""):
        return _FakeDecision(True)


class _FakeMemory:
    """Records add() calls; search() returns no dupes by default."""

    instances: list["_FakeMemory"] = []

    def __init__(self, path=None, **kwargs):
        self.added: list[str] = []
        self.closed = False
        self.search_results: list[dict] = []
        _FakeMemory.instances.append(self)

    def search(self, query, user_id=None, limit=10, **kwargs):
        return list(self.search_results)

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
    """Patch the model-touching deps in the hook with fakes + record deadline."""
    import truememory.client as client
    import truememory.ingest.encoding_gate as eg

    monkeypatch.setattr(client, "Memory", _FakeMemory)
    monkeypatch.setattr(eg, "EncodingGate", _FakeGate)

    deadline_calls: list = []
    import truememory.model_client as mc
    monkeypatch.setattr(mc, "set_request_timeout", lambda v: deadline_calls.append(v))

    return {"deadline_calls": deadline_calls, "FakeMemory": _FakeMemory}


# ---------------------------------------------------------------------------
# M-04: a real row lands at store_intensity="max"
# ---------------------------------------------------------------------------

class TestStoreActuallyLands:
    def test_max_stores_a_row(self, patched):
        from truememory.ingest.hooks import user_prompt_submit as ups
        ups._try_per_exchange_store(
            "I prefer dark mode for all my editors and always use TypeScript",
            "sess-max", "", "", "max",
        )
        adds = [c for m in patched["FakeMemory"].instances for c in m.added]
        assert len(adds) == 1, f"expected exactly one stored row, got {adds}"

    def test_max_respects_should_encode_false(self, patched, monkeypatch):
        """When the gate says should_encode=False, nothing is stored."""
        import truememory.ingest.encoding_gate as eg

        class _NoGate:
            def __init__(self, *a, **k):
                pass

            def evaluate(self, fact, category=""):
                return _FakeDecision(False)

        monkeypatch.setattr(eg, "EncodingGate", _NoGate)
        from truememory.ingest.hooks import user_prompt_submit as ups
        ups._try_per_exchange_store(
            "I prefer dark mode for all my editors", "sess-noenc", "", "", "max",
        )
        adds = [c for m in patched["FakeMemory"].instances for c in m.added]
        assert adds == [], "should not store when gate rejects"

    def test_dupe_above_threshold_not_stored(self, patched):
        """A near-duplicate (score > 0.85) is skipped."""
        from truememory.ingest.hooks import user_prompt_submit as ups
        # Pre-load the next FakeMemory with a high-scoring dupe by patching
        # the class to seed search_results.
        orig_init = _FakeMemory.__init__

        def seeded_init(self, path=None, **kwargs):
            orig_init(self, path=path, **kwargs)
            self.search_results = [{"content": "x", "score": 0.95}]

        _FakeMemory.__init__ = seeded_init
        try:
            ups._try_per_exchange_store(
                "I prefer dark mode for all my editors", "sess-dupe", "", "", "max",
            )
        finally:
            _FakeMemory.__init__ = orig_init
        adds = [c for m in patched["FakeMemory"].instances for c in m.added]
        assert adds == [], "near-duplicate should not be stored"


# ---------------------------------------------------------------------------
# M-18: deadline armed + memory closed
# ---------------------------------------------------------------------------

class TestRequestDeadlineAndClose:
    def test_deadline_armed_before_store(self, patched):
        from truememory.ingest.hooks import user_prompt_submit as ups
        ups._try_per_exchange_store(
            "I always use TypeScript over JavaScript for new projects",
            "sess-dl", "", "", "max",
        )
        assert patched["deadline_calls"], "store path must arm the recall deadline"

    def test_memory_closed_after_store(self, patched):
        from truememory.ingest.hooks import user_prompt_submit as ups
        ups._try_per_exchange_store(
            "I always use TypeScript over JavaScript for new projects",
            "sess-close", "", "", "max",
        )
        assert patched["FakeMemory"].instances, "a Memory was created"
        assert all(m.closed for m in patched["FakeMemory"].instances), (
            "Memory instance must be closed (no leak)"
        )


# ---------------------------------------------------------------------------
# M-17: enhanced depth gate differs from max (no self-match)
# ---------------------------------------------------------------------------

class TestEnhancedDistinctFromMax:
    def _write_buffer(self, tmp_path, session_id, contents):
        buf_dir = tmp_path / "buffers"
        buf_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
        f = buf_dir / f"{safe_id}.jsonl"
        with f.open("w", encoding="utf-8") as fh:
            for c in contents:
                fh.write(json.dumps({"role": "user", "content": c}) + "\n")
        return buf_dir

    def test_first_prompt_does_not_self_match(self, tmp_path, monkeypatch):
        """A first-prompt exchange (only the current prompt in the buffer)
        must NOT pass the depth gate — previously it matched itself."""
        from truememory.ingest.hooks import user_prompt_submit as ups
        prompt = "I prefer dark mode for all my editors and tools"
        buf_dir = self._write_buffer(tmp_path, "sess-first", [prompt])
        monkeypatch.setattr(ups, "_PROMPT_COUNTER_DIR", buf_dir)
        monkeypatch.setenv("TRUEMEMORY_BUFFER_DIR", str(buf_dir))
        assert ups._check_conversation_depth("sess-first", prompt) is False

    def test_enhanced_skips_shallow_max_would_store(self, patched, tmp_path, monkeypatch):
        """On a shallow/first-prompt exchange, enhanced stores nothing while
        max stores a row — proving the two tiers genuinely differ."""
        from truememory.ingest.hooks import user_prompt_submit as ups
        prompt = "I prefer dark mode for all my editors and tools"

        # Buffer contains ONLY the current prompt (first-prompt scenario).
        buf_dir = self._write_buffer(tmp_path, "sess-shallow", [prompt])
        monkeypatch.setattr(ups, "_PROMPT_COUNTER_DIR", buf_dir)
        monkeypatch.setenv("TRUEMEMORY_BUFFER_DIR", str(buf_dir))

        ups._try_per_exchange_store(prompt, "sess-shallow", "", "", "enhanced")
        enhanced_adds = [c for m in patched["FakeMemory"].instances for c in m.added]
        assert enhanced_adds == [], "enhanced must NOT fire on shallow first prompt"

        patched["FakeMemory"].instances.clear()
        ups._try_per_exchange_store(prompt, "sess-shallow", "", "", "max")
        max_adds = [c for m in patched["FakeMemory"].instances for c in m.added]
        assert len(max_adds) == 1, "max SHOULD store the same shallow prompt"

    def test_enhanced_fires_on_deep_topic(self, patched, tmp_path, monkeypatch):
        """With prior on-topic history (>=3 shared words), enhanced fires."""
        from truememory.ingest.hooks import user_prompt_submit as ups
        prior = [
            "lets talk about database indexing strategy performance",
            "the database indexing performance strategy matters here",
            "I prefer database indexing performance strategy tuning",
        ]
        prompt = "I prefer database indexing performance strategy as default"
        buf_dir = self._write_buffer(tmp_path, "sess-deep", prior + [prompt])
        monkeypatch.setattr(ups, "_PROMPT_COUNTER_DIR", buf_dir)
        monkeypatch.setenv("TRUEMEMORY_BUFFER_DIR", str(buf_dir))

        assert ups._check_conversation_depth("sess-deep", prompt) is True
        ups._try_per_exchange_store(prompt, "sess-deep", "", "", "enhanced")
        adds = [c for m in patched["FakeMemory"].instances for c in m.added]
        assert len(adds) == 1, "enhanced should fire when topic depth is real"
