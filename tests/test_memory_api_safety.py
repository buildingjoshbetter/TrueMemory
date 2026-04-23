"""Regression locks for Hunter F35 + F38 — Memory API safety.

F35: `storage.py:create_db` and `pipeline._set_busy_timeout` both derive
their `PRAGMA busy_timeout` value from a single source of truth
(`storage.DEFAULT_BUSY_TIMEOUT_MS`) so the two code paths can't drift
apart (pre-fix: 5000ms in storage vs 10_000ms in pipeline, producing
asymmetric lock-wait under contention).

F38: `Memory.add("")` and `Memory.add("   \\t\\n  ")` must issue a warning
and return a skip-marker (id=None, created_at=None) — pre-fix they
silently inserted useless rows that polluted `stats().message_count`.
"""
from __future__ import annotations

import warnings

import pytest

from truememory import Memory


# ---------------------------------------------------------------------------
# F35 — PRAGMA busy_timeout unification
# ---------------------------------------------------------------------------


def test_busy_timeout_single_source_of_truth():
    """Both code paths must derive their default from
    `storage.DEFAULT_BUSY_TIMEOUT_MS`."""
    from truememory.storage import DEFAULT_BUSY_TIMEOUT_MS
    assert DEFAULT_BUSY_TIMEOUT_MS == 10_000, (
        "F35 regression: the shared constant changed; verify all callers "
        "still intend the same value"
    )


def test_create_db_uses_shared_constant(tmp_path):
    """The sqlite connection created by `create_db` must actually have
    `busy_timeout` set to the shared constant — not some stale literal."""
    from truememory.storage import DEFAULT_BUSY_TIMEOUT_MS, create_db
    db_path = tmp_path / "check.db"
    conn = create_db(db_path)
    result = conn.execute("PRAGMA busy_timeout").fetchone()
    conn.close()
    assert result[0] == DEFAULT_BUSY_TIMEOUT_MS, (
        f"F35 regression: create_db set busy_timeout={result[0]}, expected "
        f"DEFAULT_BUSY_TIMEOUT_MS={DEFAULT_BUSY_TIMEOUT_MS}"
    )


def test_pipeline_set_busy_timeout_defaults_to_shared_constant():
    """`pipeline._set_busy_timeout()` with no timeout argument must pull
    the shared constant. Pre-fix default was 10_000 as a literal; the
    fix removes the literal by defaulting the param to None + pulling
    the constant at call time."""
    import inspect
    from truememory.ingest.pipeline import _set_busy_timeout
    sig = inspect.signature(_set_busy_timeout)
    # Default should be None (pull constant at call time), NOT a literal int
    assert sig.parameters["timeout_ms"].default is None, (
        "F35 regression: pipeline._set_busy_timeout has a literal default, "
        "which lets it drift from storage.DEFAULT_BUSY_TIMEOUT_MS"
    )


def test_pipeline_set_busy_timeout_applies_shared_constant(tmp_path):
    """End-to-end: call `_set_busy_timeout(memory)` with default and
    verify the connection's pragma matches the shared constant."""
    from truememory.ingest.pipeline import _set_busy_timeout
    from truememory.storage import DEFAULT_BUSY_TIMEOUT_MS

    m = Memory(path=str(tmp_path / "m.db"))
    try:
        # Engine needs a live connection for the helper to write the pragma
        m._engine._ensure_connection()
        # Reset it to a different value so we can detect the write
        m._engine.conn.execute("PRAGMA busy_timeout=1")
        assert m._engine.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1

        _set_busy_timeout(m)  # no timeout_ms → shared constant
        observed = m._engine.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert observed == DEFAULT_BUSY_TIMEOUT_MS
    finally:
        m.close()


# ---------------------------------------------------------------------------
# F38 — Memory.add rejects empty / whitespace-only content
# ---------------------------------------------------------------------------


def test_add_empty_string_warns_and_skips():
    m = Memory(":memory:")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = m.add("")
        assert result["id"] is None
        assert result["created_at"] is None
        assert result["content"] == ""
        # Stats must NOT count the skipped add
        assert m.stats().get("message_count", 0) == 0
        # Warning was emitted
        messages = [str(w.message) for w in caught]
        assert any("empty" in msg.lower() for msg in messages)
    finally:
        m.close()


def test_add_whitespace_only_warns_and_skips():
    m = Memory(":memory:")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = m.add("   \t\n  ")
        assert result["id"] is None
        assert m.stats().get("message_count", 0) == 0
        assert any("whitespace" in str(w.message).lower() for w in caught)
    finally:
        m.close()


def test_add_real_content_still_works():
    """Regression: the new validation must NOT affect the happy path."""
    m = Memory(":memory:")
    try:
        result = m.add("Alice prefers dark mode")
        assert result["id"] is not None
        assert isinstance(result["id"], int)
        assert result["content"] == "Alice prefers dark mode"
        assert result["created_at"] is not None
        assert m.stats().get("message_count", 0) == 1
    finally:
        m.close()


def test_add_content_with_leading_trailing_whitespace_still_stored():
    """Content with real text + surrounding whitespace is NOT empty —
    it should be stored as-is (F38's 'What NOT to do': don't trim
    silently; surface the warning so callers can decide)."""
    m = Memory(":memory:")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = m.add("  real content  ")
        assert result["id"] is not None
        # Content stored as-passed, NOT trimmed
        assert result["content"] == "  real content  "
        assert m.stats().get("message_count", 0) == 1
        # No warning for non-empty content
        empty_warnings = [w for w in caught if "empty" in str(w.message).lower()]
        assert not empty_warnings
    finally:
        m.close()


def test_add_skip_marker_preserves_user_id():
    """The skip-marker should echo the `user_id` the caller passed so
    batch callers can correlate the skip to their input row."""
    m = Memory(":memory:")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = m.add("", user_id="alice")
        assert result["user_id"] == "alice"
        assert result["id"] is None
    finally:
        m.close()


def test_add_skip_marker_on_none_like_input():
    """None content is a programmer error but shouldn't crash."""
    m = Memory(":memory:")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # `not None` is True, so the branch fires without even calling .strip()
            result = m.add(None)  # type: ignore[arg-type]
        assert result["id"] is None
    finally:
        m.close()


def test_stats_reflects_only_real_adds():
    """Integration: mix valid and empty adds, confirm stats counts
    only the valid ones."""
    m = Memory(":memory:")
    try:
        m.add("first")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.add("")
            m.add("   ")
        m.add("second")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.add("\t\n")
        assert m.stats().get("message_count", 0) == 2
    finally:
        m.close()


@pytest.mark.parametrize(
    "bad_content",
    ["", " ", "\t", "\n", "\r\n", "   \t\n  ", " "],  # non-breaking space
)
def test_add_rejects_various_empty_forms(bad_content):
    """A broad parameterized regression — any `str.strip()` → `""` is
    treated as empty."""
    m = Memory(":memory:")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = m.add(bad_content)
        assert result["id"] is None, (
            f"F38 regression: content={bad_content!r} was stored instead "
            f"of being skipped"
        )
    finally:
        m.close()
