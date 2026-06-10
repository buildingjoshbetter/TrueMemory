"""Issue #590 — build_vectors must batch write transactions.

Verifies that ``build_vectors`` no longer holds a single write transaction
across the full corpus re-embedding.  Instead it commits every N embedding
batches, releasing the DB write lock between commits so other writers can
proceed.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import sqlite_vec  # noqa: F401
    _HAS_VEC = True
except ImportError:
    _HAS_VEC = False

pytestmark = pytest.mark.skipif(not _HAS_VEC, reason="sqlite-vec not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model():
    """Return a mock embedding model that produces deterministic 256-d vectors.

    Uses a text-seeded RNG so identical texts get identical vectors —
    enough for nearest-neighbour search to return the correct result.
    """
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
    """Return a (server_module, db_path) tuple with a minimal TrueMemory env."""
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
    """Add *n* messages to the database via the server module."""
    m = ms._get_memory()
    for i in range(n):
        m.add(f"memory number {i}", user_id="test")


class _CommitCountingConn:
    """Thin wrapper around a real sqlite3.Connection that counts commit() calls.

    sqlite3.Connection attributes are read-only in Python 3.14+, so we
    cannot monkeypatch ``conn.commit`` directly.  This proxy delegates
    everything to the underlying connection but intercepts ``commit``.
    """

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
# Tests
# ---------------------------------------------------------------------------


class TestAllVectorsBuilt:
    """After batching, every message must still get a vector."""

    def test_small_corpus(self, server):
        """Fewer messages than one embedding batch."""
        _seed_messages(server, 5)
        mem = server._get_memory()
        conn = mem._engine.conn
        import truememory.vector_search as vs

        count = vs.build_vectors(conn, txn_batch=2)
        assert count == 5

        rows = conn.execute(
            f"SELECT count(*) FROM {vs._active_vec_table(conn)}"
        ).fetchone()[0]
        assert rows == 5

    def test_multiple_batches_all_vectors_present(self, server, monkeypatch):
        """Force small embedding batch + txn_batch to exercise multi-commit
        path with a modest number of messages."""
        _seed_messages(server, 12)
        mem = server._get_memory()
        conn = mem._engine.conn
        import truememory.vector_search as vs

        # batch_size=3 => 4 embedding batches; txn_batch=2 => 2 commits
        monkeypatch.setattr(vs, "_get_batch_size", lambda: 3)

        count = vs.build_vectors(conn, txn_batch=2)
        assert count == 12

        rows = conn.execute(
            f"SELECT count(*) FROM {vs._active_vec_table(conn)}"
        ).fetchone()[0]
        assert rows == 12

    def test_search_still_works_after_batch_build(self, server):
        """Vectors built in batches must be searchable."""
        _seed_messages(server, 8)
        mem = server._get_memory()
        conn = mem._engine.conn
        import truememory.vector_search as vs

        vs.build_vectors(conn, txn_batch=1)
        results = vs.search_vector(conn, "memory number 3", limit=5)
        assert len(results) > 0
        # The closest result should mention "memory number 3"
        assert "memory number 3" in results[0]["content"]


class TestMultipleBatchesCommitted:
    """Verify that commit() is called multiple times, not once."""

    def test_commit_count_single_embed_batch(self, server):
        """Even with one embedding batch, we get multiple commits
        (DELETE + batch + metadata)."""
        _seed_messages(server, 5)
        mem = server._get_memory()
        real_conn = mem._engine.conn
        import truememory.vector_search as vs

        wrapper = _CommitCountingConn(real_conn)

        vs.build_vectors(wrapper, txn_batch=1)

        # DELETE commit (1) + 1 batch commit + metadata commit (1) = 3
        assert wrapper.commit_count >= 3, (
            f"Expected at least 3 commits (delete + batch + metadata), "
            f"got {wrapper.commit_count}"
        )

    def test_many_batches_many_commits(self, server, monkeypatch):
        """Force small embedding batch size so we get many batches, then
        verify commit is called proportionally."""
        _seed_messages(server, 12)
        mem = server._get_memory()
        real_conn = mem._engine.conn
        import truememory.vector_search as vs

        # Force batch_size=3 so 12 messages = 4 embedding batches
        monkeypatch.setattr(vs, "_get_batch_size", lambda: 3)

        wrapper = _CommitCountingConn(real_conn)

        # txn_batch=2 means commit every 2 embedding batches
        # 4 embedding batches / 2 = 2 txn commits + DELETE commit + metadata
        count = vs.build_vectors(wrapper, txn_batch=2)
        assert count == 12

        # DELETE (1) + 2 txn commits + metadata (1) = 4
        assert wrapper.commit_count >= 4, (
            f"Expected at least 4 commits, got {wrapper.commit_count}"
        )

    def test_txn_batch_1_commits_every_embed_batch(self, server, monkeypatch):
        """txn_batch=1 with 4 embedding batches should yield at least 6
        commits: 1 DELETE + 4 batch commits + 1 metadata."""
        _seed_messages(server, 12)
        mem = server._get_memory()
        real_conn = mem._engine.conn
        import truememory.vector_search as vs

        monkeypatch.setattr(vs, "_get_batch_size", lambda: 3)

        wrapper = _CommitCountingConn(real_conn)

        count = vs.build_vectors(wrapper, txn_batch=1)
        assert count == 12

        # DELETE (1) + 4 batch commits + metadata (1) = 6
        assert wrapper.commit_count >= 6, (
            f"Expected at least 6 commits, got {wrapper.commit_count}"
        )


class TestPartialFailureIsolation:
    """A failure mid-batch must not corrupt previously committed batches."""

    def test_prior_batches_survive_failure(self, server, monkeypatch):
        """If embedding fails partway through, already-committed vectors
        must remain in the table."""
        _seed_messages(server, 12)
        mem = server._get_memory()
        conn = mem._engine.conn
        import truememory.vector_search as vs

        # Force batch_size=3 so 12 messages = 4 embedding batches
        monkeypatch.setattr(vs, "_get_batch_size", lambda: 3)

        call_count = {"n": 0}
        real_encode = vs._encode_with_mps_fallback

        def failing_encode(model, texts, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 3:
                raise RuntimeError("simulated embedding failure")
            return real_encode(model, texts, **kwargs)

        monkeypatch.setattr(vs, "_encode_with_mps_fallback", failing_encode)

        # txn_batch=1 means commit after every embedding batch.
        # Batches 1-2 succeed and commit; batch 3 raises.
        with pytest.raises(RuntimeError, match="simulated embedding failure"):
            vs.build_vectors(conn, txn_batch=1)

        # The first 2 batches (6 vectors) should be committed and intact.
        tbl = vs._active_vec_table(conn)
        survived = conn.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
        assert survived == 6, (
            f"Expected 6 vectors from 2 committed batches, got {survived}"
        )
