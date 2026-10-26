"""Legacy vector gaps get a backfill cursor from default processes, without stats().

Releases before note_vectorless stored memories without vectors and without
the ``vector_backfill_from`` cursor, so no backfill pass reached them; only a
stats() call seeded one. A default-mode Engine now seeds it once, off the
caller's thread; hooks (auto_maintenance=False) never scan. Synthetic stores,
deterministic mock encoder.
"""

import hashlib
import json
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
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


def test_open_requests_maintenance_before_the_seed_starts(legacy_store: Path, monkeypatch) -> None:
    # A seed started first once took the Engine's writer, so the open's wake
    # found it busy and was dropped (at MCP startup, the only wake).
    events = []
    monkeypatch.setattr(maintenance.MaintenanceCoordinator, "request_layers",
                        lambda self, **kwargs: events.append("wake") or False)
    real = TrueMemoryEngine._schedule_backfill

    def schedule(self, *, seed_gap: bool = False) -> None:
        events.append("seed" if seed_gap else "pass")
        real(self, seed_gap=seed_gap)

    monkeypatch.setattr(TrueMemoryEngine, "_schedule_backfill", schedule)
    memory = Memory(path=legacy_store)
    try:
        memory._engine._ensure_connection()  # The open's maintenance wake.
        _join_backfill(memory)
        assert memory._engine._maintenance_pending_reason != "foreground_busy"
    finally:
        memory.close()
    assert events[:2] == ["wake", "seed"]
    assert _cursor(legacy_store) == str(GAPS.start)


def test_background_work_runs_no_statement_on_the_engine_connection(legacy_store: Path) -> None:
    # Callers may use and close engine.conn directly (test_issue_650 and the
    # #684 reconnect tests do); a close under another thread's statement crashes.
    statements = []
    memory = Memory(path=legacy_store)
    engine = memory._engine
    try:
        engine._open_connection_handle()
        engine.conn.set_trace_callback(lambda sql: statements.append(threading.current_thread().name))
        engine._ensure_connection()  # The open's wake seeds the cursor.
        _join_backfill(memory)
        assert _cursor(legacy_store) == str(GAPS.start)
        memory.search("project alpha", user_id="u1", limit=3, _skip_reranker=True)  # And starts a pass.
        _join_backfill(memory)
        assert _vectorless(legacy_store) < len(GAPS)
    finally:
        memory.close()
    assert statements and "truememory-vector-backfill" not in statements


DIRECT_CLOSE = textwrap.dedent('''
    import hashlib, json, sqlite3, sys, threading, time
    from unittest.mock import MagicMock, patch
    import numpy as np
    from truememory import maintenance, vector_search
    from truememory.client import Memory

    mode, db = sys.argv[1], sys.argv[2]
    reached, closed, armed = threading.Event(), threading.Event(), threading.Event()

    def encode(texts, **kwargs):
        vectors = [np.random.RandomState(int(hashlib.md5(text.encode()).hexdigest()[:8], 16)).rand(256)
                   for text in texts]
        return np.array([vector / np.linalg.norm(vector) for vector in vectors], dtype=np.float32)

    def background():
        return armed.is_set() and threading.current_thread().name == "truememory-vector-backfill"

    def progress():
        # Work on the Engine's connection: hold one statement inside sqlite3_step.
        if background() and not reached.is_set():
            reached.set()
            time.sleep(1.0)
        return 0

    def held(real):
        # Work on a connection of its own: hold it until engine.conn is closed.
        def call(*args, **kwargs):
            if background() and not reached.is_set():
                reached.set()
                assert closed.wait(30)
            return real(*args, **kwargs)
        return call

    vector_search.seed_backfill_cursor = held(vector_search.seed_backfill_cursor)
    vector_search.backfill_missing_vectors = held(vector_search.backfill_missing_vectors)
    model = MagicMock()
    model.encode = encode
    with patch("truememory.vector_search.get_model", return_value=model), \\
            patch.object(maintenance.MaintenanceCoordinator, "request_layers", lambda self, **kwargs: False):
        memory = Memory(path=db)
        engine = memory._engine
        engine._open_connection_handle()
        engine.conn.set_progress_handler(progress, 50)
        if mode == "seed":
            armed.set()
        engine._ensure_connection()  # The open's wake starts the seed.
        if mode == "pass":
            engine._backfill_thread.join(30)
            armed.set()
            memory.search("project alpha", user_id="u1", limit=3, _skip_reranker=True)
        thread = engine._backfill_thread
        assert reached.wait(30)
        time.sleep(0.1)
        engine.conn.close()  # Directly, with no Engine lock.
        closed.set()
        thread.join(30)
        memory.close()
    conn = sqlite3.connect(db)
    cursor = conn.execute("SELECT value FROM metadata WHERE key = 'vector_backfill_from'").fetchone()
    print(json.dumps({"alive": thread.is_alive(), "cursor": cursor and cursor[0]}))
''')


@pytest.mark.parametrize(("mode", "cursor"), [("seed", str(GAPS.start)), ("pass", str(GAPS.start + 32))])
def test_direct_close_during_background_work_does_not_crash(legacy_store: Path, tmp_path: Path,
                                                            mode: str, cursor: str) -> None:
    # A child process: the crash this guards against is a segmentation fault.
    script = tmp_path / "child.py"
    script.write_text(DIRECT_CLOSE, encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), mode, str(legacy_store)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    # The seed recorded the first gap; one pass of 32 then moved the cursor past it.
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"alive": False, "cursor": cursor}


def test_close_fences_a_seed_without_the_last_close_wait(legacy_store: Path, monkeypatch) -> None:
    entered, release, seen = threading.Event(), threading.Event(), []
    real = vector_search.seed_backfill_cursor

    def seed(conn, table, **kwargs):
        seen.append(conn)
        entered.set()
        assert release.wait(10)
        return real(conn, table, **kwargs)

    monkeypatch.setattr(vector_search, "seed_backfill_cursor", seed)
    memory = Memory(path=legacy_store)
    memory._engine._ensure_connection()
    assert entered.wait(10)
    thread = memory._engine._backfill_thread
    started = time.monotonic()
    memory.close()
    closed = time.monotonic() - started
    release.set()
    thread.join(10)
    assert isinstance(seen[0], maintenance._FencedConnection)
    # Its own close fences it after the 1 s join; detach does not wait 2 s more.
    assert closed < 1.5
    assert _cursor(legacy_store) is None  # Its write after close() was refused.
