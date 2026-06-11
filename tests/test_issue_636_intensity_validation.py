"""Tests for issue #636 — intensity validation + recall flow coherence.

Covers four findings in the ingest hooks:

- M-16 (P1): intensity config failed OPEN to "max" on any invalid value. The
  exclusion-based dispatch (``!= "standard"`` / ``!= "enhanced"`` → max) meant
  ``"MAX"``, ``"Enhanced"``, garbage, and JSON ``null`` all enabled the most
  expensive every-prompt mode. Readers now normalize (lowercase + allowlist)
  and fail CLOSED to "standard". We assert that in BOTH user_prompt_submit and
  session_start readers.
- M-41 (P2): ``_try_proactive_recall`` consumed the one-shot #561 debounce
  marker, returned None, and ``main()`` then fell through to ``_try_auto_recall``
  which saw no marker and re-ran the exact first-prompt search #561 suppressed.
  We assert auto-recall does NOT fire when the marker was consumed.
- M-42 (P2): max intensity built a second ``Memory()`` in ``_try_auto_recall``
  when proactive recall found nothing — double engine init + double search. We
  assert a single Memory construction (and both instances closed).
- M-72 (P3): "max" was nullified by the 8KB payload cap. The recall budget now
  scales with intensity. We assert max gets a larger effective budget while an
  explicit env override stays authoritative.

All tests mock the model/embedding layer (no real model loads).
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ---------------------------------------------------------------------------
# Fakes — fully model-free (mirrors tests/test_issue_634_per_exchange_store.py)
# ---------------------------------------------------------------------------

class _FakeMemory:
    """Records construction/close; search() returns configurable results."""

    instances: list["_FakeMemory"] = []
    next_results: list[dict] = []

    def __init__(self, path=None, **kwargs):
        self.closed = False
        self.searches: list[str] = []
        self.search_results: list[dict] = list(_FakeMemory.next_results)
        _FakeMemory.instances.append(self)

    def search(self, query, user_id=None, limit=10, **kwargs):
        self.searches.append(query)
        return list(self.search_results)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture(autouse=True)
def _reset_fakes():
    _FakeMemory.instances.clear()
    _FakeMemory.next_results = []
    yield
    _FakeMemory.instances.clear()
    _FakeMemory.next_results = []


@pytest.fixture
def patched_memory(monkeypatch):
    """Patch Memory + deadline setter used by the recall paths."""
    import truememory.client as client
    monkeypatch.setattr(client, "Memory", _FakeMemory)
    import truememory.model_client as mc
    monkeypatch.setattr(mc, "set_request_timeout", lambda v: None)
    return _FakeMemory


def _set_marker(monkeypatch, consumed: bool):
    """Make consume_recall_injected return ``consumed`` exactly once.

    Tracks how many times it is actually called so tests can assert the marker
    is consumed exactly once across the whole prompt (M-41).
    """
    import truememory.ingest.hooks._shared as shared
    calls = {"n": 0}

    def fake_consume(session_id, within_seconds=None):
        calls["n"] += 1
        # one-shot: True on first call, False afterwards (mirrors real marker)
        return consumed and calls["n"] == 1

    monkeypatch.setattr(shared, "consume_recall_injected", fake_consume)
    return calls


# ---------------------------------------------------------------------------
# M-16: invalid intensity normalizes to "standard" (fail closed), not "max"
# ---------------------------------------------------------------------------

# Genuinely invalid values: not a recognized level even after lowercasing.
# (Case variants like "MAX"/"Enhanced" ARE valid and covered separately.)
_INVALID_VALUES = ["TURBO", "garbage", "", "  ", None, 1, True, ["max"]]


def _write_config(home, **fields):
    cfg = home / ".truememory" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"tier": "edge", **fields}), encoding="utf-8")
    return cfg


class TestM16UserPromptSubmitNormalization:
    @pytest.mark.parametrize("value", _INVALID_VALUES)
    def test_invalid_search_intensity_not_max(self, value, monkeypatch, tmp_path):
        import truememory.ingest.hooks.user_prompt_submit as ups
        home = tmp_path / "home"
        _write_config(home, search_intensity=value, store_intensity=value)
        monkeypatch.setattr(ups.Path, "home", staticmethod(lambda: home))
        search, store = ups._get_intensity_config()
        assert search == "standard", f"{value!r} must not enable an intensity mode"
        assert store == "standard"

    def test_uppercase_max_is_normalized(self, monkeypatch, tmp_path):
        import truememory.ingest.hooks.user_prompt_submit as ups
        home = tmp_path / "home"
        _write_config(home, search_intensity="MAX")
        monkeypatch.setattr(ups.Path, "home", staticmethod(lambda: home))
        search, _ = ups._get_intensity_config()
        # uppercase MAX is a *valid* level after lowercasing — not garbage
        assert search == "max"

    def test_valid_values_preserved(self, monkeypatch, tmp_path):
        import truememory.ingest.hooks.user_prompt_submit as ups
        home = tmp_path / "home"
        for v in ("standard", "enhanced", "max"):
            _write_config(home, search_intensity=v, store_intensity=v)
            monkeypatch.setattr(ups.Path, "home", staticmethod(lambda: home))
            search, store = ups._get_intensity_config()
            assert (search, store) == (v, v)

    def test_normalize_helper_direct(self):
        import truememory.ingest.hooks.user_prompt_submit as ups
        assert ups._normalize_intensity("MAX") == "max"
        assert ups._normalize_intensity("  Enhanced ") == "enhanced"
        assert ups._normalize_intensity("turbo") == "standard"
        assert ups._normalize_intensity(None) == "standard"
        assert ups._normalize_intensity(42) == "standard"


class TestM16SessionStartNormalization:
    @pytest.mark.parametrize("value", _INVALID_VALUES)
    def test_invalid_search_intensity_not_max(self, value, monkeypatch, tmp_path):
        import truememory.ingest.hooks.session_start as ss
        home = tmp_path / "home"
        _write_config(home, search_intensity=value)
        monkeypatch.setattr(ss.Path, "home", staticmethod(lambda: home))
        assert ss._get_search_intensity() == "standard"

    def test_uppercase_normalized(self, monkeypatch, tmp_path):
        import truememory.ingest.hooks.session_start as ss
        home = tmp_path / "home"
        _write_config(home, search_intensity="Enhanced")
        monkeypatch.setattr(ss.Path, "home", staticmethod(lambda: home))
        assert ss._get_search_intensity() == "enhanced"

    def test_normalize_helper_direct(self):
        import truememory.ingest.hooks.session_start as ss
        assert ss._normalize_intensity("MAX") == "max"
        assert ss._normalize_intensity("nonsense") == "standard"
        assert ss._normalize_intensity(None) == "standard"


# ---------------------------------------------------------------------------
# M-41: marker consumed by proactive → auto-recall must NOT fire
# ---------------------------------------------------------------------------

class TestM41NoDuplicateRecall:
    def test_proactive_debounced_skips_autorecall(self, patched_memory, monkeypatch):
        import truememory.ingest.hooks.user_prompt_submit as ups
        # marker present (consumed once)
        calls = _set_marker(monkeypatch, consumed=True)

        # Drive main() with a recall-intent prompt under max intensity.
        monkeypatch.setattr(ups, "_get_intensity_config", lambda: ("max", "standard"))
        monkeypatch.setattr(ups, "_increment_prompt_count", lambda sid: 1)
        monkeypatch.setattr(ups, "buffer_message", lambda *a, **k: None)
        monkeypatch.setattr(ups, "_prune_old_buffers", lambda: None)
        monkeypatch.setattr(ups, "_try_capture_email", lambda p: None)
        monkeypatch.setattr(ups, "_try_per_exchange_store", lambda *a, **k: None)

        out = _run_main(monkeypatch, ups, prompt="what is my favorite color?")

        # No Memory was ever constructed (both paths suppressed by the marker).
        assert _FakeMemory.instances == [], "debounced prompt must not search"
        # Marker consumed exactly once (not double-consumed by both paths).
        assert calls["n"] == 1
        # No recall context emitted.
        assert out == ""

    def test_direct_proactive_returns_searched_when_debounced(self, patched_memory):
        import truememory.ingest.hooks.user_prompt_submit as ups
        ctx, searched = ups._try_proactive_recall(
            "what is my favorite color?", "", "", "sess", "max", 1, debounced=True,
        )
        assert ctx is None
        assert searched is True, "debounced proactive must report it handled the prompt"

    def test_autorecall_skips_when_debounced(self, patched_memory):
        import truememory.ingest.hooks.user_prompt_submit as ups
        out = ups._try_auto_recall(
            "what is my favorite color?", "", "", "sess", debounced=True,
        )
        assert out is None
        assert _FakeMemory.instances == [], "debounced auto-recall must not search"


# ---------------------------------------------------------------------------
# M-42: empty proactive result must NOT cause a second Memory init / search
# ---------------------------------------------------------------------------

class TestM42NoDoubleInit:
    def test_max_empty_result_single_memory(self, patched_memory, monkeypatch):
        import truememory.ingest.hooks.user_prompt_submit as ups
        # no marker → proactive runs the real search, which finds nothing
        _set_marker(monkeypatch, consumed=False)
        _FakeMemory.next_results = []

        monkeypatch.setattr(ups, "_get_intensity_config", lambda: ("max", "standard"))
        monkeypatch.setattr(ups, "_increment_prompt_count", lambda sid: 1)
        monkeypatch.setattr(ups, "buffer_message", lambda *a, **k: None)
        monkeypatch.setattr(ups, "_prune_old_buffers", lambda: None)
        monkeypatch.setattr(ups, "_try_capture_email", lambda p: None)
        monkeypatch.setattr(ups, "_try_per_exchange_store", lambda *a, **k: None)

        out = _run_main(monkeypatch, ups, prompt="what is my favorite color?")

        # Exactly ONE Memory constructed (proactive). Auto-recall does not add
        # a second when proactive already searched this prompt (M-42).
        assert len(_FakeMemory.instances) == 1, (
            f"expected single Memory init, got {len(_FakeMemory.instances)}"
        )
        # And it was closed (no leak).
        assert all(m.closed for m in _FakeMemory.instances)
        assert out == ""

    def test_proactive_closes_memory_on_empty(self, patched_memory):
        import truememory.ingest.hooks.user_prompt_submit as ups
        _FakeMemory.next_results = []
        ctx, searched = ups._try_proactive_recall(
            "what is my favorite color?", "", "", "sess", "max", 1, debounced=False,
        )
        assert ctx is None
        assert searched is True
        assert len(_FakeMemory.instances) == 1
        assert _FakeMemory.instances[0].closed is True


# ---------------------------------------------------------------------------
# M-72: recall budget scales with intensity; env override authoritative
# ---------------------------------------------------------------------------

class TestM72BudgetScaling:
    def test_max_budget_larger_than_standard(self, monkeypatch):
        # Ensure no env override is in play for this assertion.
        monkeypatch.delenv("TRUEMEMORY_RECALL_BUDGET_CHARS", raising=False)
        import importlib
        import truememory.ingest.hooks.session_start as ss
        ss = importlib.reload(ss)
        assert ss._intensity_budget("max") > ss._intensity_budget("standard")
        assert ss._intensity_budget("enhanced") == ss._intensity_budget("standard")

    def test_env_override_is_authoritative(self, monkeypatch):
        monkeypatch.setenv("TRUEMEMORY_RECALL_BUDGET_CHARS", "12345")
        import importlib
        import truememory.ingest.hooks.session_start as ss
        ss = importlib.reload(ss)
        # With an explicit override, all intensities use the pinned budget.
        assert ss._intensity_budget("max") == 12345
        assert ss._intensity_budget("standard") == 12345
        # restore module to default env state for other tests
        monkeypatch.delenv("TRUEMEMORY_RECALL_BUDGET_CHARS", raising=False)
        importlib.reload(ss)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_main(monkeypatch, ups, prompt: str) -> str:
    """Run ups.main() feeding *prompt* on stdin; return emitted additionalContext."""
    import io
    import sys

    stdin_payload = json.dumps({"prompt": prompt, "session_id": "sess"})
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_payload))

    captured: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: captured.append(" ".join(str(x) for x in a)))
    monkeypatch.delenv("TRUEMEMORY_EXTRACTION", raising=False)

    ups.main()

    if not captured:
        return ""
    try:
        return json.loads(captured[0]).get("additionalContext", "")
    except (json.JSONDecodeError, IndexError):
        return ""
