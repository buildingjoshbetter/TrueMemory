"""Regression tests for issue #648 (theme T6): vec compatibility surfacing +
tier-switch guards.

Covers:
  - M-12: a dim-mismatched vec table surfaces degradation in health
    (``_vectors_load_error`` set + stats.health reflects it) instead of a
    permanent silent FTS-only fallback.
  - M-46: ``TrueMemoryMigrationError`` propagates from the vec-load path
    instead of being swallowed as a generic "FTS-only mode" error.
  - M-78: cluster-supplement is gated on live vector health, so a
    same-dim/different-model DB does not compare query embeddings to stale
    centroids.
  - M-23: the tier-switch vec load is guarded — a sqlite without
    ``enable_load_extension`` raises a typed, actionable error rather than a
    raw ``AttributeError``.
  - M-51: a ``start_rebuild`` failure before the worker thread starts does NOT
    brick subsequent rebuilds (``_active_thread`` is not left pointing at the
    long-lived caller thread).
"""
from __future__ import annotations

import sqlite3
import threading
from unittest.mock import patch

import pytest

from tests.conftest import requires_sqlite_ext


@pytest.fixture(autouse=True)
def _reset_vectors_load_error():
    """``_vectors_load_error`` is module-global; reset around each test so the
    health assertions are not polluted by sibling tests."""
    import truememory.engine as eng
    prev = eng._vectors_load_error
    eng._vectors_load_error = None
    try:
        yield
    finally:
        eng._vectors_load_error = prev


def _load_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _make_mismatched_db(path: str, wrong_dim: int) -> None:
    """Create a DB whose vec_messages table declares ``wrong_dim`` (!= the
    current embedder dim) plus matching metadata, simulating a tier change
    that left an incompatible vector table behind."""
    from truememory.storage import create_db
    import truememory.vector_search as vs

    conn = create_db(path)
    _load_vec(conn)
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages "
        f"USING vec0({vs._vec0_column_decl(wrong_dim)})"
    )
    # Insert a row so vectors_are_built() / SELECT 1 succeeds (table is "built").
    conn.execute(
        "INSERT INTO vec_messages(rowid, embedding) VALUES (1, ?)",
        (sqlite3.Binary(b"\x00\x00\x80\x3f" * wrong_dim),),
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('embed_model', ?)",
        ("legacy_model_1024",),
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('embed_dim', ?)",
        (str(wrong_dim),),
    )
    conn.commit()
    conn.close()


@requires_sqlite_ext
class TestM12HealthSurfacing:
    """Dim mismatch must surface in health, not silently drop to FTS-only."""

    def test_open_exists_path_surfaces_dim_mismatch(self, tmp_path):
        import truememory.vector_search as vs
        from truememory.engine import TrueMemoryEngine, get_vectors_load_error
        from truememory.vector_search import TrueMemoryMigrationError

        wrong_dim = vs._embedding_dim + 256  # guaranteed mismatch
        db = tmp_path / "mismatch.db"
        _make_mismatched_db(str(db), wrong_dim)

        eng = TrueMemoryEngine(str(db))
        # The exists-path now runs _check_embedder_compatibility() instead of
        # trusting a bare SELECT 1, so a mismatch raises the migration error.
        with pytest.raises(TrueMemoryMigrationError):
            eng.open(rebuild_vectors=False)

        # Degradation is recorded so truememory_stats.health reports it rather
        # than claiming vectors are healthy.
        err = get_vectors_load_error()
        assert err is not None
        assert "TrueMemoryMigrationError" in err
        assert eng._has_vectors is False

    def test_health_payload_reports_degraded(self, tmp_path):
        import truememory.vector_search as vs
        from truememory.engine import TrueMemoryEngine
        from truememory.vector_search import TrueMemoryMigrationError

        wrong_dim = vs._embedding_dim + 256
        db = tmp_path / "mismatch2.db"
        _make_mismatched_db(str(db), wrong_dim)

        eng = TrueMemoryEngine(str(db))
        with pytest.raises(TrueMemoryMigrationError):
            eng.open(rebuild_vectors=False)

        from truememory.mcp_server import _build_health_payload
        payload = _build_health_payload()
        assert payload["vectors"]["status"] == "degraded"
        assert payload["vectors"]["last_error"] is not None


@requires_sqlite_ext
class TestM46MigrationErrorPropagates:
    """TrueMemoryMigrationError must not be swallowed as a generic FTS error."""

    def test_ensure_connection_propagates_migration_error(self, tmp_path):
        import truememory.vector_search as vs
        from truememory.engine import TrueMemoryEngine
        from truememory.vector_search import TrueMemoryMigrationError

        wrong_dim = vs._embedding_dim + 256
        db = tmp_path / "ensure.db"
        _make_mismatched_db(str(db), wrong_dim)

        eng = TrueMemoryEngine(str(db))
        # _ensure_connection() → init_vec_table() → _check_embedder_compatibility()
        # raises; the new typed except clause re-raises instead of logging it as
        # "Failed to load sqlite-vec — FTS-only mode".
        with pytest.raises(TrueMemoryMigrationError):
            eng._ensure_connection()

    def test_ensure_connection_generic_error_still_fts_only(self, tmp_path):
        """A non-migration failure during vec init still degrades gracefully to
        FTS-only (regression guard: we only special-cased the migration error)."""
        from truememory.engine import TrueMemoryEngine

        db = tmp_path / "generic.db"
        eng = TrueMemoryEngine(str(db))
        with patch(
            "truememory.engine.init_vec_table",
            side_effect=RuntimeError("boom — not a migration"),
        ):
            # Must not raise: generic failures keep the FTS-only fallback.
            eng._ensure_connection()
        assert eng._has_vectors is False


@requires_sqlite_ext
class TestM78ClusterGate:
    """Cluster-supplement must be skipped when vectors are unavailable."""

    def test_clustering_skipped_when_no_vectors(self, tmp_path):
        from truememory.engine import TrueMemoryEngine

        db = tmp_path / "cluster.db"
        eng = TrueMemoryEngine(str(db))
        eng._ensure_connection()

        # Simulate a same-dim/different-model DB where the clusters table exists
        # but vectors are not trustworthy.
        eng._has_clustering = True
        eng._has_vectors = False

        called = {"clustered": False}

        def _fake_clustered(*a, **k):
            called["clustered"] = True
            return []

        with patch("truememory.engine.search_clustered", _fake_clustered):
            eng.search_agentic(
                "anything", limit=3, use_hyde=False, use_reranker=False,
                use_clustering=True, llm_fn=None,
            )

        assert called["clustered"] is False, (
            "cluster-supplement ran against stale centroids despite "
            "_has_vectors=False"
        )


class TestM23TierSwitchGuard:
    """_open_db must raise a typed error (not AttributeError) when the sqlite
    cannot load extensions."""

    def test_open_db_no_extension_raises_typed_error(self, tmp_path):
        from truememory.tier_switch.manager import (
            _open_db,
            TierSwitchUnsupportedError,
        )

        db = tmp_path / "noext.db"

        # Simulate macOS system Python: enable_load_extension absent.
        class _NoExtConn:
            def __init__(self, real):
                self._real = real

            def execute(self, *a, **k):
                return self._real.execute(*a, **k)

            def close(self):
                return self._real.close()

            def __getattr__(self, name):
                if name == "enable_load_extension":
                    raise AttributeError(
                        "'sqlite3.Connection' object has no attribute "
                        "'enable_load_extension'"
                    )
                return getattr(self._real, name)

        real = sqlite3.connect(str(db))
        with patch("sqlite3.connect", return_value=_NoExtConn(real)):
            with pytest.raises(TierSwitchUnsupportedError) as exc:
                _open_db(db)
        # Actionable platform guidance, not a raw AttributeError.
        assert "sqlite-vec" in str(exc.value)
        assert "extension" in str(exc.value).lower()


class TestM51StartRebuildNotBricked:
    """A pre-thread failure in start_rebuild must not brick later rebuilds."""

    def test_pre_thread_failure_does_not_brick(self):
        from truememory.tier_switch.manager import (
            RebuildManager,
            TierSwitchUnsupportedError,
        )

        mgr = RebuildManager()

        # First call: _open_db fails before the worker thread starts (M-23).
        with patch(
            "truememory.tier_switch.manager._open_db",
            side_effect=TierSwitchUnsupportedError("no vec on this python"),
        ):
            with pytest.raises(TierSwitchUnsupportedError):
                mgr.start_rebuild("base", db_path=None)

        # The slot must be free: _active_thread not left pointing at the
        # long-lived caller thread, and the claim flag cleared.
        assert mgr._claimed is False
        assert not (mgr._active_thread and mgr._active_thread.is_alive()), (
            "start_rebuild left _active_thread alive after a pre-thread "
            "failure — subsequent rebuilds would be permanently no-op'd"
        )
        assert mgr._active_thread is not threading.current_thread()

        # Second call reaches _open_db again (proves it is not short-circuited).
        reached = {"open_db": False}

        def _fail_again(*a, **k):
            reached["open_db"] = True
            raise TierSwitchUnsupportedError("still no vec")

        with patch(
            "truememory.tier_switch.manager._open_db", side_effect=_fail_again
        ):
            with pytest.raises(TierSwitchUnsupportedError):
                mgr.start_rebuild("base", db_path=None)

        assert reached["open_db"] is True, (
            "second start_rebuild was short-circuited — manager is bricked"
        )
