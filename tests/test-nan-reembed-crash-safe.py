"""The Qwen3 NaN re-embed never leaves the served vectors empty, and resumes.

It used to start with DROP TABLE (or the rebuild's DELETE) on the served pair,
so a default process that exited mid-run left vector search empty until a
later run finished. It now builds shadow tables and replaces the served rows
in one transaction. Synthetic stores, mock encoders, and a child process this
test starts and kills.
"""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sqlite_vec

from truememory import maintenance, vector_search
from truememory.client import Memory
from tests.conftest import requires_sqlite_ext

# These stores carry real vec0 tables; some interpreters cannot load extensions.
pytestmark = requires_sqlite_ext

MEMORIES = 60
OLD, NEW = np.eye(256, dtype=np.float32)[0], np.eye(256, dtype=np.float32)[1]


def _model(vector: np.ndarray, encoded: list, fail_on_call: int | None = None) -> MagicMock:
    def encode(texts, **kwargs):
        encoded.append(len(texts))
        if fail_on_call is not None and len(encoded) == fail_on_call:
            raise RuntimeError("synthetic interruption")
        return np.array([vector for _ in texts])
    model = MagicMock()
    model.encode = encode
    return model


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _served(path: Path, table: str) -> dict[int, np.ndarray]:
    conn = _connect(path)
    try:
        return {rowid: np.frombuffer(blob, dtype=np.float32) for rowid, blob in
                conn.execute(f"SELECT rowid, embedding FROM {table}")}
    finally:
        conn.close()


def _tables(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


@pytest.fixture
def store(tmp_path: Path) -> Path:
    path = tmp_path / "memories.db"
    with patch("truememory.vector_search.get_model", return_value=_model(OLD, [])), \
            patch.object(maintenance.MaintenanceCoordinator, "request_layers", lambda self, **kwargs: False):
        memory = Memory(path=path)
        for index in range(MEMORIES):
            memory.add(content=f"synthetic memory {index} about project alpha", user_id="u1")
        memory.close()
        with Memory(path=path, auto_maintenance=False) as memory:  # Moves the pair to the tier's cache names.
            memory.search("project alpha", user_id="u1", limit=3, _skip_reranker=True)
    return path


def _reembed(path: Path, **kwargs) -> int:
    from truememory.storage import create_db
    from truememory.tier_switch.runtime import legacy_rebuild_operation
    conn = create_db(path)
    try:
        vector_search.init_vec_table(conn)
        with legacy_rebuild_operation(conn):
            return vector_search.reembed_served_vectors(conn, **kwargs)
    finally:
        conn.close()


def test_interrupted_reembed_keeps_served_vectors_and_resumes(store: Path, monkeypatch) -> None:
    served = "vec_messages_edge"
    assert served in _tables(store) and len(_served(store, served)) == MEMORIES
    monkeypatch.setattr(vector_search, "_BUILD_VECTORS_TXN_BATCH", 1)
    monkeypatch.setattr(vector_search, "_get_batch_size", lambda: 10)

    first = []
    with patch("truememory.vector_search.get_model", return_value=_model(NEW, first, fail_on_call=3)):
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            _reembed(store, mark=("synthetic_reembed_done", "1"))
    # Two batches reached the shadow; the served pair still has every old vector.
    assert len(_served(store, served + "_reembed")) == 20
    assert all(np.allclose(vector, OLD) for vector in _served(store, served).values())
    assert len(_served(store, served)) == MEMORIES

    # A memory added meanwhile (an append) keeps the vector its add stored.
    with patch("truememory.vector_search.get_model", return_value=_model(NEW, [])), \
            patch.object(maintenance.MaintenanceCoordinator, "request_layers", lambda self, **kwargs: False):
        with Memory(path=store, auto_maintenance=False) as memory:
            appended = memory.add(content="synthetic memory added during the re-embed", user_id="u1")["id"]

    second = []
    with patch("truememory.vector_search.get_model", return_value=_model(NEW, second)):
        _reembed(store, mark=("synthetic_reembed_done", "1"))
    # Resumed: the rest of the first shadow; the second, captured after the append, covers it too.
    assert sum(second) == (MEMORIES - 20) + (MEMORIES + 1)
    vectors = _served(store, served)
    assert sorted(vectors) == list(range(1, MEMORIES + 1)) + [appended]
    assert all(np.allclose(vector, NEW) for vector in vectors.values())
    assert len(_served(store, "vec_messages_sep_edge")) == MEMORIES + 1
    assert not {served + "_reembed", "vec_messages_sep_edge_reembed"} & _tables(store)
    conn = sqlite3.connect(store)
    try:
        assert conn.execute("SELECT value FROM metadata WHERE key='synthetic_reembed_done'").fetchone() == ("1",)
        assert conn.execute("SELECT count(*) FROM metadata WHERE key LIKE 'vec_source_v1:%_reembed'").fetchone() == (0,)
    finally:
        conn.close()



def test_interrupted_swap_keeps_a_vector_for_every_memory_and_resumes(store: Path, monkeypatch) -> None:
    served = "vec_messages_edge"
    # Short batches; a swap in one transaction never reaches a third check.
    monkeypatch.setattr(vector_search, "_REEMBED_SWAP_BATCH", 16, raising=False)
    real = vector_search._complete_shadow
    checks = []

    def check(conn, shadow, identity):
        checks.append(shadow)
        # First the pre-build check, then one per swap batch: stop in the third batch.
        if shadow == served + "_reembed" and checks.count(shadow) == 4:
            raise RuntimeError("synthetic interruption")
        return real(conn, shadow, identity)

    monkeypatch.setattr(vector_search, "_complete_shadow", check)
    with patch("truememory.vector_search.get_model", return_value=_model(NEW, [])):
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            _reembed(store, mark=("synthetic_reembed_done", "1"))
    vectors = _served(store, served)
    assert sorted(vectors) == list(range(1, MEMORIES + 1))  # Every memory still has a vector.
    assert all(np.allclose(vectors[mid], NEW if mid < 32 else OLD) for mid in vectors)

    monkeypatch.setattr(vector_search, "_complete_shadow", real)
    rerun = []
    with patch("truememory.vector_search.get_model", return_value=_model(NEW, rerun)):
        _reembed(store, mark=("synthetic_reembed_done", "1"))
    assert rerun == []  # Both shadows were complete: only the swap continued.
    for table in (served, "vec_messages_sep_edge"):
        vectors = _served(store, table)
        assert sorted(vectors) == list(range(1, MEMORIES + 1))
        assert all(np.allclose(vector, NEW) for vector in vectors.values())
    assert not {served + "_reembed", "vec_messages_sep_edge_reembed"} & _tables(store)
    conn = sqlite3.connect(store)
    try:
        assert conn.execute("SELECT value FROM metadata WHERE key='synthetic_reembed_done'").fetchone() == ("1",)
        assert conn.execute("SELECT count(*) FROM metadata WHERE key LIKE 'vec_reembed_swapped:%'").fetchone() == (0,)
    finally:
        conn.close()

def test_swap_pauses_between_batches_with_the_writer_free(store: Path, monkeypatch) -> None:
    # Back-to-back batches starved writers in other processes (a hook add
    # failed "database is locked" after a 31 s swap): every pause must leave
    # the writer free for another connection to take at once.
    monkeypatch.setattr(vector_search, "_REEMBED_SWAP_BATCH", 16, raising=False)
    taken = []

    def pause(seconds):
        other = sqlite3.connect(store, timeout=0)
        try:
            other.execute("BEGIN IMMEDIATE")
            other.execute("ROLLBACK")
            taken.append(seconds)
        finally:
            other.close()

    monkeypatch.setattr(vector_search.time, "sleep", pause)
    with patch("truememory.vector_search.get_model", return_value=_model(NEW, [])):
        _reembed(store, mark=("synthetic_reembed_done", "1"))
    # Two served tables, MEMORIES / 16 batches each.
    assert len(taken) >= 2 * (MEMORIES // 16)
    assert all(seconds > 0 for seconds in taken)


CHILD = textwrap.dedent('''
    import json, os, sqlite3, sys, threading, time
    import numpy as np
    from unittest.mock import MagicMock, patch
    from truememory import maintenance, vector_search
    from truememory.client import Memory

    mode, db = sys.argv[1], sys.argv[2]
    encoded = []

    def encode(texts, **kwargs):
        if "_bg_wrapper" in threading.current_thread().name:
            encoded.append(len(texts))
            if mode == "hang" and len(encoded) == 3:
                open(sys.argv[3], "w").close()  # Mid-run: two batches are in the shadow.
                time.sleep(600)
        return np.array([np.eye(256, dtype=np.float32)[len(text) % 256] for text in texts])

    model = MagicMock()
    model.encode = encode
    vector_search._BUILD_VECTORS_TXN_BATCH = 1
    vector_search._get_batch_size = lambda: 100

    def served(conn):
        for name in ("vec_messages_basepro", "vec_messages"):
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone():
                return name

    with patch("truememory.vector_search.get_model", return_value=model), \\
            patch.object(maintenance.MaintenanceCoordinator, "request_layers", lambda self, **kwargs: False):
        if mode == "seed":
            memory = Memory(path=db)
            for index in range(400):
                memory.add(content=f"synthetic memory {index} about project alpha", user_id="u1")
            memory.close()
            conn = sqlite3.connect(db)
            conn.execute("DELETE FROM metadata WHERE key='qwen3_nan_fix_applied'")
            conn.commit()
            conn.close()
        elif mode == "check":
            with Memory(path=db, auto_maintenance=False) as memory:
                results = memory.search("project alpha", user_id="u1", limit=5, _skip_reranker=True)
                conn = memory._engine.conn
                table = served(conn)
                rows = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                query = vector_search.serialize_f32(np.eye(256, dtype=np.float32)[3])
                hits = conn.execute(f"SELECT rowid FROM {table} WHERE embedding MATCH ? AND k = 5", (query,)).fetchall()
            print(json.dumps({"table": table, "rows": rows, "knn_hits": len(hits), "results": len(results)}))
        else:
            memory = Memory(path=db)
            memory.search("project alpha", user_id="u1", limit=3, _skip_reranker=True)
            deadline = time.monotonic() + 60
            while memory._engine._nan_migration_in_progress and time.monotonic() < deadline:
                time.sleep(0.05)
            conn = memory._engine.conn
            flag = conn.execute("SELECT value FROM metadata WHERE key='qwen3_nan_fix_applied'").fetchone()
            table = served(conn)
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            print(json.dumps({"flag": flag and flag[0], "rows": conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0],
                              "shadows": sorted(name for name in names if name.endswith("_reembed")),
                              "encoded": sum(encoded)}))
            memory.close()
''')


def _child(script: Path, *args: str, timeout: float = 120) -> dict:
    env = dict(os.environ, TRUEMEMORY_EMBED_MODEL="base")
    result = subprocess.run([sys.executable, str(script), *args], env=env, capture_output=True, text=True,
                            timeout=timeout)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else {}


@pytest.mark.skipif(sys.platform != "darwin", reason="The Qwen3 NaN re-embed runs on macOS only")
def test_killed_reembed_leaves_vector_search_working_and_a_later_run_completes(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    db, marker = tmp_path / "memories.db", tmp_path / "mid-run"
    _child(script, "seed", str(db))

    env = dict(os.environ, TRUEMEMORY_EMBED_MODEL="base")
    child = subprocess.Popen([sys.executable, str(script), "hang", str(db), str(marker)], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 60
        while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "the re-embed never reached its third batch"
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait(30)

    checked = _child(script, "check", str(db))
    assert (checked["rows"], checked["knn_hits"]) == (400, 5), checked
    assert checked["results"] > 0

    done = _child(script, "complete", str(db))
    assert done["flag"] == "1" and done["rows"] == 400 and done["shadows"] == [], done
    assert done["encoded"] == 200 + 400  # Resumed: two of four main batches were already in the shadow.
