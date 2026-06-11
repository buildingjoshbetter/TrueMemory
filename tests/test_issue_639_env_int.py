"""Issue #639: shared ``_env_int`` helper.

Covers two failure classes the helper kills:

* **M-27 (P1)** — an empty or non-numeric env var used in a module-level bare
  ``int(os.environ.get(...))`` crashed the whole hook/server at *import*,
  before any ``main()`` try/except could run.
* **M-59 (P2)** — negative/zero values that parse but misbehave (negative
  SQLite ``LIMIT`` = unlimited, ``AUTO_CONSOLIDATE_EVERY=0`` = consolidate on
  every add, zero/negative budgets that drop all memory).

The helper returns the default on garbage and clamps into ``[lo, hi]``.
"""

import importlib

import pytest

from truememory._platform import _env_int


# ---------------------------------------------------------------------------
# (a) default on unset / empty / non-numeric
# ---------------------------------------------------------------------------

def test_env_int_unset_returns_default(monkeypatch):
    monkeypatch.delenv("TM_TEST_639", raising=False)
    assert _env_int("TM_TEST_639", 42) == 42


@pytest.mark.parametrize("raw", ["", "  ", "abc", "1.5", "0x10", "1,000", "nan"])
def test_env_int_garbage_returns_default(monkeypatch, raw):
    monkeypatch.setenv("TM_TEST_639", raw)
    assert _env_int("TM_TEST_639", 7) == 7


def test_env_int_valid_value_parsed(monkeypatch):
    monkeypatch.setenv("TM_TEST_639", "123")
    assert _env_int("TM_TEST_639", 7) == 123


def test_env_int_negative_value_parsed_when_unclamped(monkeypatch):
    monkeypatch.setenv("TM_TEST_639", "-5")
    assert _env_int("TM_TEST_639", 7) == -5


# ---------------------------------------------------------------------------
# (b) clamping into [lo, hi]
# ---------------------------------------------------------------------------

def test_env_int_clamps_negative_to_lo(monkeypatch):
    monkeypatch.setenv("TM_TEST_639", "-5")
    assert _env_int("TM_TEST_639", 7, lo=0) == 0


def test_env_int_clamps_zero_up_to_lo_one(monkeypatch):
    # AUTO_CONSOLIDATE_EVERY=0 must not mean "every add".
    monkeypatch.setenv("TM_TEST_639", "0")
    assert _env_int("TM_TEST_639", 25, lo=1) == 1


def test_env_int_clamps_over_max_to_hi(monkeypatch):
    monkeypatch.setenv("TM_TEST_639", "999")
    assert _env_int("TM_TEST_639", 7, hi=10) == 10


def test_env_int_within_bounds_unchanged(monkeypatch):
    monkeypatch.setenv("TM_TEST_639", "5")
    assert _env_int("TM_TEST_639", 7, lo=0, hi=10) == 5


def test_env_int_default_is_not_clamped_caller_must_pass_sane_default(monkeypatch):
    # A garbage value falls through to the default verbatim (callers pass
    # in-bounds defaults), but a value that parses is clamped.
    monkeypatch.setenv("TM_TEST_639", "garbage")
    assert _env_int("TM_TEST_639", 50, lo=0, hi=100) == 50


# ---------------------------------------------------------------------------
# (c) a representative module imports cleanly with a garbage env var
#     (no ValueError at import, M-27)
# ---------------------------------------------------------------------------

def test_session_start_imports_with_garbage_env(monkeypatch):
    # Was: bare int(os.environ.get("TRUEMEMORY_RECALL_BUDGET_CHARS", "8192"))
    # crashed the entire SessionStart hook at import on an empty value.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("TRUEMEMORY_RECALL_BUDGET_CHARS", "")
    monkeypatch.setenv("TRUEMEMORY_DIRECTIVE_LIMIT", "abc")
    monkeypatch.setenv("TRUEMEMORY_RECALL_MEMORY_CHARS", "-100")

    import truememory.ingest.hooks.session_start as ss

    importlib.reload(ss)  # must not raise ValueError at import

    assert ss.RECALL_BUDGET_CHARS == 8192     # empty -> default
    assert ss.DIRECTIVE_LIMIT == 50           # garbage -> default
    assert ss.RECALL_MEMORY_CHARS == 0        # negative -> clamped to lo=0


def test_engine_imports_with_garbage_consolidate_env(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("TRUEMEMORY_AUTO_CONSOLIDATE_EVERY", "")

    import truememory.engine as eng

    importlib.reload(eng)  # must not raise at import
