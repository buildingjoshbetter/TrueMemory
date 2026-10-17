"""Legacy vector gaps get a backfill cursor from default processes, without stats().

Releases before note_vectorless stored memories without vectors and without
the ``vector_backfill_from`` cursor, so no backfill pass reached them; only a
stats() call seeded one. A default-mode Engine now seeds it once, off the
caller's thread; hooks (auto_maintenance=False) never scan. Synthetic stores,
deterministic mock encoder.
"""

import hashlib
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sqlite_vec

from truememory import maintenance, vector_search
from truememory.client import Memory
from truememory.engine import TrueMemoryEngine

MEMORIES = 120
GAPS = range(11, 81)  # 70 early ids: three bounded backfill passes.


def _encoder() -> MagicMock:
    def encode(texts, **kwargs):
        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            vector = np.random.RandomState(seed).rand(256).astype(np.float32)
            vectors.append(vector / np.linalg.norm(vector))
        return np.array(vectors)
    model = MagicMock()
    model.encode = encode
    return model


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _vectorless(path: Path) -> int:
    conn = _connect(path)
    try:
        # An Engine open moves the generic pair to the tier's cache names.
        table = conn.execute("SELECT name FROM sqlite_master WHERE name IN ('vec_messages', 'vec_messages_edge') "
                             "ORDER BY name LIMIT 1").fetchone()[0]
        return conn.execute(f"SELECT count(*) FROM messages m WHERE NOT EXISTS "
                            f"(SELECT 1 FROM {table} v WHERE v.rowid = m.id)").fetchone()[0]
    finally:
        conn.close()


def _cursor(path: Path) -> str | None:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key = 'vector_backfill_from'").fetchone()
        return None if row is None else row[0]
    finally:
        conn.close()


def _join_backfill(memory: Memory) -> None:
    thread = memory._engine._backfill_thread
    if thread is not None:
        thread.join(10)
        assert not thread.is_alive()


@pytest.fixture
def encoder():
    with patch("truememory.vector_search.get_model", return_value=_encoder()), \
            patch.object(maintenance.MaintenanceCoordinator, "request_layers", lambda self, **kwargs: False):
        yield


@pytest.fixture
def legacy_store(tmp_path: Path, encoder) -> Path:
    """Memories an older release stored without vectors and without a cursor."""
    path = tmp_path / "memories.db"
    seed = Memory(path=path)
    for index in range(MEMORIES):
        seed.add(content=f"synthetic memory {index} about project alpha", user_id="u1")
    seed.close()
    conn = _connect(path)
    for table in ("vec_messages", "vec_messages_sep"):
        conn.execute(f"DELETE FROM {table} WHERE rowid BETWEEN ? AND ?", (GAPS.start, GAPS.stop - 1))
    conn.execute("DELETE FROM metadata WHERE key = 'vector_backfill_from'")
    conn.commit()
    conn.close()
    assert (_vectorless(path), _cursor(path)) == (len(GAPS), None)
    return path


def test_default_process_fills_legacy_gaps_without_stats(legacy_store: Path, monkeypatch) -> None:
    def no_stats(*args, **kwargs):
        raise AssertionError("coverage must not depend on a stats() call")

    monkeypatch.setattr(TrueMemoryEngine, "get_stats", no_stats)
    monkeypatch.setattr(TrueMemoryEngine, "get_vector_coverage", no_stats)
    memory = Memory(path=legacy_store)
    try:
        for _ in range(10):
            memory.search("project alpha", user_id="u1", limit=3, _skip_reranker=True)
            _join_backfill(memory)
            if _vectorless(legacy_store) == 0:
                break
    finally:
        memory.close()
    assert _vectorless(legacy_store) == 0
    assert _cursor(legacy_store) is None  # The last pass reached the end and retired it.


def test_hook_memory_never_scans_for_gaps(legacy_store: Path, monkeypatch) -> None:
    scans = []
    real = vector_search._first_gap
    monkeypatch.setattr(vector_search, "_first_gap", lambda *args, **kwargs: scans.append(args) or real(*args, **kwargs))
    with Memory(path=legacy_store, auto_maintenance=False) as memory:
        memory.search("project alpha", user_id="u1", limit=3, _skip_reranker=True)
        memory.add(content="synthetic memory stored from a hook", user_id="u1")
        assert memory._engine._backfill_thread is None
    assert scans == []
    assert _cursor(legacy_store) is None


def test_seed_runs_once_per_engine_off_the_caller_thread(legacy_store: Path, monkeypatch) -> None:
    seeds = []
    real = vector_search.seed_backfill_cursor

    def seed(*args, **kwargs):
        seeds.append(threading.current_thread().name)
        return real(*args, **kwargs)

    monkeypatch.setattr(vector_search, "seed_backfill_cursor", seed)
    memory = Memory(path=legacy_store)
    try:
        for _ in range(3):
            memory.search("project alpha", user_id="u1", limit=3, _skip_reranker=True)
            _join_backfill(memory)
            memory.add(content="synthetic memory added later", user_id="u1")
            _join_backfill(memory)
    finally:
        memory.close()
    assert seeds == ["truememory-vector-backfill"]


def test_seed_scans_in_chunks_and_stops(legacy_store: Path, monkeypatch) -> None:
    from truememory.storage import create_db
    from truememory.tier_switch.runtime import legacy_rebuild_operation
    monkeypatch.setattr(vector_search, "_SEED_CHUNK", 8)
    holds = []

    class Lock:
        def __enter__(self):
            holds.append(1)

        def __exit__(self, *exc):
            return False

    conn = create_db(legacy_store)
    try:
        vector_search.init_vec_table(conn)
        with legacy_rebuild_operation(conn):
            stop = threading.Event()
            stop.set()
            assert vector_search.seed_backfill_cursor(conn, "vec_messages", write_lock=Lock(), stop=stop) is None
            assert _cursor(legacy_store) is None
            assert vector_search.seed_backfill_cursor(conn, "vec_messages", write_lock=Lock()) == GAPS.start
            # One hold for the setup, one per 8-id chunk up to the gap, one to record it.
            assert len(holds) == 1 + 1 + (GAPS.start // 8 + 1) + 1
            assert _cursor(legacy_store) == str(GAPS.start)
            assert vector_search.seed_backfill_cursor(conn, "vec_messages", write_lock=Lock()) is None  # Cursor exists.
    finally:
        conn.close()
