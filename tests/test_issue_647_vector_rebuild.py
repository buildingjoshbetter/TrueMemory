"""Issue #647 — vector rebuild durability.

M-21 (P1): ``build_vectors`` used to ``DELETE FROM {tbl}; commit()`` *before*
any embedding, so a mid-run failure persisted an empty/partial table that
``engine.open()`` accepted as "built" — and embedder metadata was committed
separately, so a crash mid tier-migration could leave NEW vectors under OLD
metadata. Fix: an ``in_progress`` build-state marker written in the SAME
transaction as the DELETE and cleared with the final commit (alongside the
embedder metadata); a table left ``in_progress`` is treated as NOT built and
the build resumes from ``max(rowid)`` rather than re-wiping.

M-45 (P2): ``build_separation_vectors`` never got #619's batch-commit
treatment — it held one giant write transaction for the entire run. Fix:
mirror ``build_vectors``' batched commits + the same marker.

These tests construct embeddings by hand (mock model) — no model loads,
HF_HUB_OFFLINE-safe — and gate on the repo's sqlite-vec skip pattern.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _can_load_vec() -> bool:
    """Check if sqlite-vec can actually be loaded (not just importable)."""
    try:
        import sqlite_vec
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        return True
    except Exception:
        return False


_HAS_VEC = _can_load_vec()

pytestmark = pytest.mark.skipif(not _HAS_VEC, reason="sqlite-vec cannot load")


# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_issue_590_build_vectors_batch.py)
# ---------------------------------------------------------------------------


def _make_mock_model():
    """Deterministic 256-d vectors, text-seeded so identical text → identical vec."""
    mock = MagicMock()

    def _encode(texts, **kw):
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % (2**31))
            vecs.append(rng.randn(256).astype(np.float32))
        return np.array(vecs)

    mock.encode = _encode
    return mock


def _make_server(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".truememory").mkdir()
    db_path = tmp_path / "memories.db"
    monkeypatch.setenv("TRUEMEMORY_DB", str(db_path))
    monkeypatch.setenv("TRUEMEMORY_EMBED_MODEL", "edge")
    import truememory.mcp_server as ms

    monkeypatch.setattr(ms, "_TRUEMEMORY_DIR", home / ".truememory")
    monkeypatch.setattr(ms, "_CONFIG_PATH", home / ".truememory" / "config.json")
    monkeypatch.setattr(ms, "_DB_PATH", str(db_path))
    monkeypatch.setattr(ms, "_memory", None)
    return ms, db_path


def _seed_messages(ms, n: int) -> None:
    m = ms._get_memory()
    for i in range(n):
        m.add(f"memory number {i}", user_id="test")


class _CommitCountingConn:
    """Real connection wrapper that counts commit() calls (3.14-safe proxy)."""

    def __init__(self, real_conn: sqlite3.Connection):
        self._real = real_conn
        self.commit_count = 0

    def commit(self):
        self.commit_count += 1
        self._real.commit()

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def server(monkeypatch, tmp_path):
    mock_model = _make_mock_model()
    with patch("truememory.vector_search.get_model", return_value=mock_model):
        ms, _db = _make_server(monkeypatch, tmp_path)
        yield ms
    if ms._memory is not None:
        try:
            ms._memory.close()
        except Exception:
            pass
    ms._memory = None
    import truememory.vector_search as vs

    vs.set_embedding_model("edge")


# ---------------------------------------------------------------------------
# (a) Interrupted build → table marked in_progress → treated as NOT built
# ---------------------------------------------------------------------------


class TestInterruptedBuildNotTrusted:
    def test_marker_set_during_build_and_failure_leaves_it(self, server, monkeypatch):
        """A build that raises mid-run leaves the in_progress marker set, and
        the table is therefore reported as NOT built."""
        _seed_messages(server, 12)
        mem = server._get_memory()
        conn = mem._engine.conn
        import truememory.vector_search as vs

        monkeypatch.setattr(vs, "_get_batch_size", lambda: 3)

        call_count = {"n": 0}
        real_encode = vs._encode_with_mps_fallback

        def failing_encode(model, texts, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 3:
                raise RuntimeError("simulated embedding failure")
            return real_encode(model, texts, **kwargs)

        monkeypatch.setattr(vs, "_encode_with_mps_fallback", failing_encode)

        tbl = vs._active_vec_table(conn)
        with pytest.raises(RuntimeError, match="simulated embedding failure"):
            vs.build_vectors(conn, txn_batch=1)

        # Marker survives the crash -> table is mid-build.
        assert vs._build_in_progress(conn, tbl) is True
        # engine-side "is it built?" check must reject the partial table.
        assert vs.vectors_are_built(conn, tbl) is False

    def test_empty_table_with_marker_not_built(self, server):
        """A 0-row table left in_progress (the classic crash-before-first-batch
        case) must NOT be accepted as built."""
        _seed_messages(server, 3)
        mem = server._get_memory()
        conn = mem._engine.conn
        import truememory.vector_search as vs

        tbl = vs._active_vec_table(conn)
        conn.execute(f"DELETE FROM {tbl}")
        vs._mark_build_in_progress(conn, tbl)
        conn.commit()

        # Old behaviour: SELECT 1 ... LIMIT 1 succeeds -> "built". New: rejected.
        assert vs.vectors_are_built(conn, tbl) is False


# ---------------------------------------------------------------------------
# (b) Resume after interruption does not wipe already-written vectors
# ---------------------------------------------------------------------------


class TestResumeDoesNotWipe:
    def test_resume_keeps_prior_vectors_and_completes(self, server, monkeypatch):
        """After an interruption left N committed vectors + marker, a second
        build resumes from max(rowid): it does NOT re-wipe and ends with every
        message embedded exactly once."""
        _seed_messages(server, 12)
        mem = server._get_memory()
        conn = mem._engine.conn
        import truememory.vector_search as vs

        monkeypatch.setattr(vs, "_get_batch_size", lambda: 3)
        tbl = vs._active_vec_table(conn)

        # First run: fail on the 3rd embedding batch (rows 1-6 committed).
        call_count = {"n": 0}
        real_encode = vs._encode_with_mps_fallback

        def failing_encode(model, texts, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 3:
                raise RuntimeError("boom")
            return real_encode(model, texts, **kwargs)

        monkeypatch.setattr(vs, "_encode_with_mps_fallback", failing_encode)
        with pytest.raises(RuntimeError, match="boom"):
            vs.build_vectors(conn, txn_batch=1)

        survived = conn.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
        assert survived == 6
        max_before = conn.execute(f"SELECT MAX(rowid) FROM {tbl}").fetchone()[0]
        assert max_before == 6

        # Second run: encoding works again. Resume must NOT delete the 6 rows.
        monkeypatch.setattr(vs, "_encode_with_mps_fallback", real_encode)

        # Wrap the connection to assert the resume path issues NO DELETE
        # (sqlite3.Connection attrs are read-only in 3.14, so proxy instead of
        # patching .execute in place).
        class _ExecSpyConn:
            def __init__(self, real):
                self._real = real
                self.executed: list[str] = []

            def execute(self, sql, *a, **k):
                self.executed.append(sql)
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, name):
                return getattr(self._real, name)

        spy = _ExecSpyConn(conn)
        added = vs.build_vectors(spy, txn_batch=1)
        executed = spy.executed

        # Resumed run only embedded the remaining 6 messages.
        assert added == 6
        assert not any(
            sql.strip().upper().startswith(f"DELETE FROM {tbl.upper()}")
            for sql in executed
        ), "resume path must not DELETE already-written vectors"

        total = conn.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
        assert total == 12
        # rowids are unique (no duplicate re-embed of rows 1-6).
        distinct = conn.execute(
            f"SELECT count(DISTINCT rowid) FROM {tbl}"
        ).fetchone()[0]
        assert distinct == 12
        # Marker cleared on success.
        assert vs._build_in_progress(conn, tbl) is False
        assert vs.vectors_are_built(conn, tbl) is True


# ---------------------------------------------------------------------------
# (c) build_separation_vectors commits in batches (M-45)
# ---------------------------------------------------------------------------


class TestSeparationBatchCommits:
    def test_separation_does_not_hold_one_giant_txn(self, server, monkeypatch):
        """build_separation_vectors must commit multiple times (DELETE/marker +
        per-batch commits + final), not a single end-of-run commit."""
        _seed_messages(server, 12)
        mem = server._get_memory()
        real_conn = mem._engine.conn
        import truememory.vector_search as vs

        monkeypatch.setattr(vs, "_get_batch_size", lambda: 3)  # 4 batches

        wrapper = _CommitCountingConn(real_conn)
        count = vs.build_separation_vectors(wrapper, txn_batch=1)
        assert count == 12

        # DELETE/marker (1) + 4 batch commits + final clear/metadata (1) >= 6.
        assert wrapper.commit_count >= 6, (
            f"separation build held too few commits ({wrapper.commit_count}); "
            f"expected batched commits, not one giant transaction"
        )

    def test_separation_marker_cleared_on_success(self, server):
        _seed_messages(server, 5)
        mem = server._get_memory()
        conn = mem._engine.conn
        import truememory.vector_search as vs

        vs.build_separation_vectors(conn, txn_batch=1)
        sep_tbl = vs._active_sep_table(conn)
        assert vs._build_in_progress(conn, sep_tbl) is False


# ---------------------------------------------------------------------------
# (d) Marker cleared on success alongside embedder metadata
# ---------------------------------------------------------------------------


class TestMarkerClearedWithMetadata:
    def test_marker_cleared_and_metadata_written(self, server):
        _seed_messages(server, 5)
        mem = server._get_memory()
        conn = mem._engine.conn
        import truememory.vector_search as vs

        tbl = vs._active_vec_table(conn)
        n = vs.build_vectors(conn, txn_batch=1)
        assert n == 5

        # Marker gone, table trusted.
        assert vs._build_in_progress(conn, tbl) is False
        assert vs.vectors_are_built(conn, tbl) is True

        # Embedder metadata stamped (model + dim), so no NEW-vectors-under-OLD
        # -metadata window.
        model, dim = vs._read_embedder_metadata(conn)
        assert model == vs.EMBEDDING_MODEL
        assert dim == vs._embedding_dim

    def test_metadata_and_marker_clear_share_final_commit(self, server, monkeypatch):
        """The marker clear and embedder-metadata write must be committed
        together — never a window where vectors are present + trusted but
        metadata is stale."""
        _seed_messages(server, 5)
        mem = server._get_memory()
        real_conn = mem._engine.conn
        import truememory.vector_search as vs

        tbl = vs._active_vec_table(real_conn)
        # Stale metadata from a hypothetical previous embedder.
        vs._ensure_metadata_table(real_conn)
        real_conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value, updated_at) "
            "VALUES ('embed_model', 'stale-old-model', '')"
        )
        real_conn.commit()

        # After the final commit of a successful build, metadata is current AND
        # the marker is clear in the same observable state.
        vs.build_vectors(real_conn, txn_batch=1)
        model, _ = vs._read_embedder_metadata(real_conn)
        assert model == vs.EMBEDDING_MODEL
        assert vs._build_in_progress(real_conn, tbl) is False
