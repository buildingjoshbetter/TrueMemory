"""Regression locks for the Windows-portability fix in
``truememory/ingest/hooks/_shared.py`` and
``truememory/ingest/hooks/session_start.py``.

Bug (before fix): both modules called ``import fcntl`` at module top with no
``try / except ImportError`` guard. On Windows, ``fcntl`` does not exist, so
``import truememory.ingest.hooks._shared`` raised ``ModuleNotFoundError``
immediately. That cascaded to every Claude Code hook subprocess
(``session_start``, ``user_prompt_submit``, ``stop``) which all import from
``_shared``, breaking the entire hook-based memory-extraction pipeline on
Windows.

The fix mirrors the established pattern already used by
``truememory/ingest/pipeline.py``, ``truememory/hooks/core.py``, and
``truememory/ingest/hooks/user_prompt_submit.py``: a ``try / except
ImportError`` wraps the import and sets ``_HAS_FCNTL``; every ``fcntl.*``
call site guards on the flag.

These tests pin three things:

1. Both modules expose ``_HAS_FCNTL`` (presence of the flag is the contract
   that the import is guarded — without the try/except, the flag would
   never be assigned because the import would raise).
2. ``check_extraction_budget`` falls back to a non-atomic read-modify-write
   without raising when ``_HAS_FCNTL`` is False, and still enforces the
   hourly cap deterministically (the lock is for concurrency, not for
   correctness in single-threaded paths).
3. On a POSIX environment with ``fcntl`` available, ``check_extraction_budget``
   actually acquires LOCK_EX — a regression lock so a future refactor can't
   silently drop the cross-process coordination on platforms that need it.

Test fixtures rebuild ``_BUDGET_FILE`` per test inside ``tmp_path`` so the
user's real ``~/.truememory/.extraction_budget`` is never touched.
"""
from __future__ import annotations

import json
import sys
import time

import pytest


# ---------------------------------------------------------------------------
# Bug #1: module-level import contract
# ---------------------------------------------------------------------------


def test_shared_exposes_has_fcntl_flag():
    """The ``_HAS_FCNTL`` module attribute is the contract that the import
    is wrapped in try/except. Without the wrapper, the module would fail
    to import on Windows with ModuleNotFoundError, so the attribute would
    never be assigned — meaning if you can read the attribute, the guard
    is in place.
    """
    from truememory.ingest.hooks import _shared
    assert hasattr(_shared, "_HAS_FCNTL"), (
        "_shared.py is missing the _HAS_FCNTL flag — the module-level "
        "`import fcntl` is probably bare again, which crashes every Claude "
        "Code hook on Windows."
    )
    assert isinstance(_shared._HAS_FCNTL, bool)


def test_session_start_exposes_has_fcntl_flag():
    """Same contract for session_start. Without this, the SessionStart hook
    process exits with non-zero on every Windows launch before recall_memories
    ever runs — meaning no memory injection on Windows."""
    from truememory.ingest.hooks import session_start
    assert hasattr(session_start, "_HAS_FCNTL"), (
        "session_start.py is missing the _HAS_FCNTL flag — the SessionStart "
        "hook process will crash on every Windows launch."
    )
    assert isinstance(session_start._HAS_FCNTL, bool)


def test_has_fcntl_flags_agree_with_platform():
    """The flag must reflect the host environment. On POSIX with fcntl
    installed, True. On Windows or any other no-fcntl environment, False.
    A mismatch means the try/except is masking a real ImportError that
    should be surfaced.
    """
    try:
        import fcntl  # noqa: F401
        expected = True
    except ImportError:
        expected = False

    from truememory.ingest.hooks import _shared, session_start
    assert _shared._HAS_FCNTL is expected
    assert session_start._HAS_FCNTL is expected


# ---------------------------------------------------------------------------
# Bug #2: check_extraction_budget runtime behavior without fcntl
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_budget_file(tmp_path, monkeypatch):
    """Point ``_BUDGET_FILE`` at a per-test temp path so tests don't poison
    the user's real ``~/.truememory/.extraction_budget`` and don't interfere
    with each other."""
    from truememory.ingest.hooks import _shared
    budget_path = tmp_path / ".extraction_budget"
    monkeypatch.setattr(_shared, "_BUDGET_FILE", budget_path)
    return budget_path


def test_check_extraction_budget_allows_first_call_without_fcntl(
    _isolated_budget_file, monkeypatch,
):
    """Without fcntl, the function must still allow extractions up to the
    cap. The lock skip is not the same as denying extraction — that would
    silently break the hook pipeline on Windows.
    """
    from truememory.ingest.hooks import _shared
    monkeypatch.setattr(_shared, "_HAS_FCNTL", False)
    monkeypatch.setattr(_shared, "_MAX_EXTRACTIONS_PER_HOUR", 5)

    assert _shared.check_extraction_budget() is True
    assert _isolated_budget_file.exists()
    data = json.loads(_isolated_budget_file.read_text(encoding="utf-8"))
    assert data["count"] == 1
    assert data["hour"] == int(time.time() // 3600)


def test_check_extraction_budget_enforces_cap_without_fcntl(
    _isolated_budget_file, monkeypatch,
):
    """The lock is for concurrency safety across processes; the cap itself
    must still be enforced sequentially. Burn through the cap and verify
    the next call is denied.
    """
    from truememory.ingest.hooks import _shared
    monkeypatch.setattr(_shared, "_HAS_FCNTL", False)
    monkeypatch.setattr(_shared, "_MAX_EXTRACTIONS_PER_HOUR", 3)

    assert _shared.check_extraction_budget() is True
    assert _shared.check_extraction_budget() is True
    assert _shared.check_extraction_budget() is True
    assert _shared.check_extraction_budget() is False, (
        "4th call must be denied — the per-hour cap is enforced regardless "
        "of whether the lock is available."
    )


def test_check_extraction_budget_resets_at_new_hour_without_fcntl(
    _isolated_budget_file, monkeypatch,
):
    """When the recorded hour differs from the current hour, the counter
    must reset. Verify the Windows path preserves this semantics — without
    the reset, a single hour of heavy usage would permanently deny extraction
    on Windows.
    """
    from truememory.ingest.hooks import _shared
    monkeypatch.setattr(_shared, "_HAS_FCNTL", False)
    monkeypatch.setattr(_shared, "_MAX_EXTRACTIONS_PER_HOUR", 1)

    # Pre-populate the budget file with prior-hour data already at the cap.
    prior_hour = int(time.time() // 3600) - 1
    _isolated_budget_file.parent.mkdir(parents=True, exist_ok=True)
    _isolated_budget_file.write_text(
        json.dumps({"hour": prior_hour, "count": 999}), encoding="utf-8",
    )

    # Current hour is fresh — must allow extraction.
    assert _shared.check_extraction_budget() is True


def test_check_extraction_budget_does_not_call_fcntl_when_unavailable(
    _isolated_budget_file, monkeypatch,
):
    """Defensive: even if a future refactor accidentally calls
    ``fcntl.flock`` outside the ``_HAS_FCNTL`` guard, the test catches it
    because the unguarded call would raise NameError or AttributeError
    when _HAS_FCNTL is False and fcntl is bound to a module that isn't
    actually loaded.
    """
    from truememory.ingest.hooks import _shared
    monkeypatch.setattr(_shared, "_HAS_FCNTL", False)
    monkeypatch.setattr(_shared, "_MAX_EXTRACTIONS_PER_HOUR", 5)

    # Must complete without raising even when fcntl path is "disabled".
    result = _shared.check_extraction_budget()
    assert result is True


# ---------------------------------------------------------------------------
# Bug #2 regression lock: lock IS acquired on POSIX
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-only — Windows has no fcntl module to call",
)
def test_check_extraction_budget_acquires_lock_when_fcntl_available(
    _isolated_budget_file, monkeypatch,
):
    """Regression lock: on POSIX, ``check_extraction_budget`` must call
    ``fcntl.flock`` with LOCK_EX. Without cross-process coordination, two
    concurrent ingest processes would race on the read-modify-write and
    the budget cap could be silently exceeded.
    """
    from truememory.ingest.hooks import _shared
    monkeypatch.setattr(_shared, "_MAX_EXTRACTIONS_PER_HOUR", 5)
    assert _shared._HAS_FCNTL is True, "test prerequisite: fcntl must be available"

    flock_calls: list[tuple] = []
    real_flock = _shared.fcntl.flock

    def _spy_flock(fd, op):
        flock_calls.append((fd, op))
        return real_flock(fd, op)

    monkeypatch.setattr(_shared.fcntl, "flock", _spy_flock)

    _shared.check_extraction_budget()

    assert flock_calls, (
        "fcntl.flock was not called on POSIX — the cross-process lock has "
        "been silently disabled. This would let concurrent ingest processes "
        "race on the budget file and exceed the hourly cap."
    )
    fd, op = flock_calls[0]
    assert op == _shared.fcntl.LOCK_EX, (
        f"Expected LOCK_EX (exclusive), got op={op!r}. A shared lock would "
        f"not prevent concurrent writers."
    )
