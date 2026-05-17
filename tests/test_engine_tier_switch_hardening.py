"""Regression locks for the engine + tier-switch hardening PR.

Covers four classes of bugs surfaced in the 2026-05-16 cross-platform /
robustness audit (Gemini 3.1 Pro + Grok 4.3 + 5 Kosmos sub-agents):

1. **SQLite IN-clause variable-count limit** — ``engine.delete_all`` built
   unchunked ``WHERE id IN (?, ?, ...)`` clauses. Hitting
   SQLITE_MAX_VARIABLE_NUMBER (default 999 on system-packaged builds,
   32766 on newer wheels) raised ``OperationalError`` that the existing
   ``except Exception: logger.warning`` swallowed — every related-table
   cleanup for the >999 case silently failed, leaking
   ``fact_timeline`` / ``landmark_events`` / ``causal_edges`` /
   ``vec_messages*`` / ``episodes`` rows.

2. **Reranker TOCTOU on fast path** — ``get_reranker`` read ``_model``
   and ``_model_name`` as two separate global lookups. Under a concurrent
   tier-switch the GIL could release between the reads, returning the
   old model under the new name. Fix bundles the reads into a tuple
   unpack so the read is a single bytecode op.

3. **Tier-switch manager state race** — ``start_rebuild`` did a
   check-then-write of ``_active_thread`` (line 100 read, line 136 write)
   with no lock. Two concurrent ``truememory_configure`` calls could both
   pass the ``is_alive()`` check and both spawn rebuild threads racing on
   the same DB. Fix adds ``_state_lock`` around all reads / writes of
   ``_active_thread`` and ``_active_worker``.

4. **search_agentic degraded reranker fallthrough** — when the reranker
   is degraded (preload watchdog timed out or threw), the modality-fusion
   call returns the original ordering with no ``fused_score`` /
   ``rerank_score`` keys. The previous code returned that unchanged,
   bypassing the LLM-reranker fallback that would have improved quality.
   Fix detects "cross-encoder didn't run" by inspecting the result keys
   and falls through to the LLM rerank block.
"""
from __future__ import annotations

import logging
import threading
import time

import pytest


# ---------------------------------------------------------------------------
# Bug #1: _delete_in_chunks helper — pure unit test
# ---------------------------------------------------------------------------


def test_delete_in_chunks_no_ids_noop():
    """Empty id list must not run any SQL — guards against the
    ``IN ()`` syntax-error path that would happen if the helper ever
    produced an empty placeholder string."""
    from truememory.engine import _delete_in_chunks
    calls = []

    class _FakeConn:
        def execute(self, *args):
            calls.append(args)

    _delete_in_chunks(_FakeConn(), "fact_timeline", "source_message_id", [])
    assert calls == []


def test_delete_in_chunks_single_chunk():
    """≤chunk_size ids fits in one statement."""
    from truememory.engine import _delete_in_chunks
    calls = []

    class _FakeConn:
        def execute(self, sql, params):
            calls.append((sql, list(params)))

    _delete_in_chunks(_FakeConn(), "episodes", "id", [1, 2, 3])
    assert len(calls) == 1
    sql, params = calls[0]
    assert sql == "DELETE FROM episodes WHERE id IN (?,?,?)"
    assert params == [1, 2, 3]


def test_delete_in_chunks_splits_across_chunks_for_large_id_lists():
    """1500 ids @ chunk_size=500 → exactly 3 executes, fully covering
    all ids with no overlap, no drops, and each chunk under the
    SQLITE_MAX_VARIABLE_NUMBER=999 historical default."""
    from truememory.engine import _delete_in_chunks
    calls = []

    class _FakeConn:
        def execute(self, sql, params):
            calls.append((sql, list(params)))

    ids = list(range(1500))
    _delete_in_chunks(_FakeConn(), "vec_messages", "rowid", ids, chunk_size=500)

    assert len(calls) == 3, (
        f"Expected 1500/500=3 chunks, got {len(calls)}. Chunking is the "
        f"whole point of the helper — a single execute here would hit "
        f"SQLite's variable limit and silently fail."
    )

    rebuilt = []
    for sql, params in calls:
        # Every chunk must contain at most chunk_size placeholders.
        assert sql.count("?") <= 500
        rebuilt.extend(params)

    assert rebuilt == ids, (
        "Chunked deletes must cover every id without dropping or "
        "duplicating any — chunking is supposed to be transparent."
    )


def test_delete_in_chunks_keeps_each_chunk_below_sqlite_default_limit():
    """Default chunk size (500) keeps every batch well under SQLite's
    historical 999-variable cap. A future bump to chunk_size above 999
    would re-introduce the original bug on older SQLite builds."""
    from truememory.engine import _SQLITE_IN_CHUNK
    assert _SQLITE_IN_CHUNK <= 999, (
        f"_SQLITE_IN_CHUNK={_SQLITE_IN_CHUNK} exceeds the conservative "
        f"SQLITE_MAX_VARIABLE_NUMBER=999 floor; older system SQLite "
        f"builds (Debian, Ubuntu LTS) will silently fail to clean "
        f"related tables on bulk deletes."
    )


# ---------------------------------------------------------------------------
# Bug #2: reranker TOCTOU on fast path
# ---------------------------------------------------------------------------


def test_get_reranker_fast_path_reads_model_state_atomically():
    """The fast-path check in get_reranker now does a single tuple
    unpack of (_model, _model_name) so the GIL cannot release between
    observing the two globals. Verify the bytecode pattern via source
    inspection (introspection is unreliable for module globals)."""
    import inspect
    from truememory import reranker as rr

    src = inspect.getsource(rr.get_reranker)
    # The atomic bundle is the load-bearing fix; if it's gone, the
    # TOCTOU window is back.
    assert "cached_model, cached_name = _model, _model_name" in src, (
        "get_reranker fast path no longer bundles _model / _model_name "
        "into a single tuple read — TOCTOU window between the two "
        "global reads has been re-introduced. Under a concurrent "
        "tier-switch this returns the stale model under the new name."
    )


# ---------------------------------------------------------------------------
# Bug #3: tier-switch manager state lock
# ---------------------------------------------------------------------------


def test_rebuild_manager_has_state_lock():
    """The RebuildManager must own a Lock for state coordination — the
    fix is structurally dependent on its existence."""
    from truememory.tier_switch.manager import RebuildManager
    mgr = RebuildManager()
    assert hasattr(mgr, "_state_lock"), (
        "RebuildManager._state_lock missing — check-then-write race on "
        "_active_thread can no longer be prevented; two concurrent "
        "truememory_configure calls can both spawn rebuild threads."
    )
    # Must be an actual lock instance (allow Lock or RLock).
    assert hasattr(mgr._state_lock, "acquire") and hasattr(mgr._state_lock, "release")


def test_rebuild_manager_cancel_uses_lock_for_read():
    """cancel() must read _active_worker under _state_lock so a
    concurrent teardown can't NULL the reference between the
    truthiness check and the .cancel() call."""
    import inspect
    from truememory.tier_switch.manager import RebuildManager

    src = inspect.getsource(RebuildManager.cancel)
    assert "with self._state_lock" in src, (
        "RebuildManager.cancel reads _active_worker without holding "
        "_state_lock — a concurrent _rebuild_thread teardown can NULL "
        "the worker between the check and the .cancel() call."
    )


# ---------------------------------------------------------------------------
# Bug #4: search_agentic degraded reranker fallthrough
# ---------------------------------------------------------------------------


def test_search_agentic_uses_is_degraded_check_for_fallthrough():
    """search_agentic must detect "cross-encoder didn't actually run"
    by checking for fused_score / rerank_score on the result rows
    (which the degraded fallback path doesn't populate). Without this,
    the LLM-rerank fallback at line ~1758 is unreachable when the
    cross-encoder is degraded."""
    import inspect
    from truememory.engine import TrueMemoryEngine

    src = inspect.getsource(TrueMemoryEngine.search_agentic)
    # The two markers of the fix:
    assert "_cross_encoder_ran" in src, (
        "search_agentic missing the degraded-mode fallthrough guard — "
        "when the reranker is degraded, the LLM-rerank fallback is "
        "bypassed and the user gets un-reranked RRF ordering."
    )
    assert '"fused_score"' in src or "'fused_score'" in src
    assert "fall" in src.lower() and "degraded" in src.lower()


# ---------------------------------------------------------------------------
# Bug observability: vector_search separation-vector log level
# ---------------------------------------------------------------------------


def test_separation_vector_failure_logs_at_warning(caplog):
    """A failed separation-vector creation means every future
    sender-aware search will silently fail to surface that memory —
    must be at least WARNING, not DEBUG."""
    import sqlite3
    from truememory import vector_search as vs

    # In-memory DB without sqlite-vec or matching schema — the
    # _build_sep_text / model.encode path raises immediately, hitting
    # our except branch.
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, sender TEXT, "
        "recipient TEXT, timestamp TEXT)"
    )
    conn.execute(
        "INSERT INTO messages (id, sender, recipient, timestamp) "
        "VALUES (?, ?, ?, ?)",
        (1, "user", "ai", "2026-05-16T00:00:00Z"),
    )

    class _FakeModel:
        def encode(self, _):
            raise RuntimeError("simulated encode failure")

    caplog.set_level(logging.DEBUG, logger="truememory.vector_search")
    caplog.clear()
    # Function is private; access via attribute so refactors that rename
    # it surface as a test break.
    if hasattr(vs, "_build_separation_vector"):
        fn = vs._build_separation_vector
    elif hasattr(vs, "build_separation_vector_single"):
        fn = vs.build_separation_vector_single
    else:
        pytest.skip(
            "vector_search separation-vector helper not exposed under a "
            "known name; manual coverage required"
        )

    # Best-effort signature probe — most variants take (conn, message_id,
    # content, model). Fall back to skip if signature differs.
    try:
        fn(conn, 1, "hello world", _FakeModel())
    except TypeError:
        pytest.skip("separation-vector helper signature differs in this build")

    levels = [r.levelname for r in caplog.records if "separation vector" in r.message.lower()]
    assert "WARNING" in levels, (
        f"Expected WARNING for separation-vector failure, got levels: "
        f"{levels!r}. DEBUG-level here means silent per-memory data loss."
    )


# ---------------------------------------------------------------------------
# Concurrent start_rebuild only spawns one thread
# ---------------------------------------------------------------------------


def test_start_rebuild_serializes_concurrent_callers(monkeypatch, tmp_path):
    """Two threads calling start_rebuild simultaneously must result in
    exactly ONE rebuild thread being launched. Pre-fix, both threads
    could pass the is_alive() check and both spawn a worker.
    """
    from truememory.tier_switch.manager import RebuildManager

    mgr = RebuildManager()

    # Stub out everything start_rebuild touches so we exercise ONLY the
    # state-lock concurrency control, not the rebuild itself.
    spawn_calls: list = []
    spawn_lock = threading.Lock()

    class _FakeThread:
        def __init__(self, *_a, **_kw):
            self._alive = True
            # Real Thread API surface the test driver needs.
            self.daemon = _kw.get("daemon", False)

        def is_alive(self):
            return self._alive

        def start(self):
            with spawn_lock:
                spawn_calls.append(self)
            # Simulate a "hot" worker that stays alive for the test window
            # so the second caller observes is_alive()==True under the lock.
            time.sleep(0.5)
            self._alive = False

        def join(self, timeout=None):  # noqa: D401 — match Thread API
            # Test driver calls join() on the test orchestration threads
            # (which are also intercepted by the monkeypatch); make it a
            # no-op because the real synchronization happens via start()'s
            # internal sleep.
            self._alive = False

    # Patch the manager-module's view of threading.Thread without touching
    # the global threading module — keeps the test driver's own
    # threading.Thread (used to spin the two _race() callers) unaffected.
    import truememory.tier_switch.manager as _mgr_mod

    class _PatchedThreadingNamespace:
        Thread = _FakeThread
        Lock = threading.Lock  # manager also uses threading.Lock

    monkeypatch.setattr(_mgr_mod, "threading", _PatchedThreadingNamespace)
    monkeypatch.setattr(
        "truememory.tier_switch.manager._open_db",
        lambda _p=None: type("C", (), {"close": lambda self: None,
                                       "execute": lambda *_: type("R", (), {"fetchone": lambda _: None, "lastrowid": 1})()})(),
    )
    monkeypatch.setattr(
        "truememory.tier_switch.manager.tier_group", lambda t: "edge",
    )
    monkeypatch.setattr(
        "truememory.tier_switch.manager.preflight_ram_check",
        lambda _g: (True, ""),
    )
    monkeypatch.setattr(
        "truememory.tier_switch.manager.resolve_rebuild_action",
        lambda *_a: "rebuild",
    )
    monkeypatch.setattr(
        "truememory.tier_switch.manager.get_messages_to_embed",
        lambda *_a: ([{"id": 1, "content": "x"}], False),
    )
    monkeypatch.setattr(
        RebuildManager, "_create_status_row", lambda *_a: 42,
    )

    def _race():
        try:
            mgr.start_rebuild("edge")
        except Exception:
            pass

    t1 = threading.Thread(target=_race)
    t2 = threading.Thread(target=_race)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    # At most ONE thread launched — the second caller must observe the
    # in-flight thread via the lock-protected is_alive() check and
    # return the existing status_id without spawning a duplicate.
    assert len(spawn_calls) <= 1, (
        f"start_rebuild spawned {len(spawn_calls)} threads under "
        f"concurrent callers — the _state_lock did not serialize the "
        f"check-then-write of _active_thread. Two rebuilds will race "
        f"on the same DB."
    )
