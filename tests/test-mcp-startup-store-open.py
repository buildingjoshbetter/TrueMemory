"""In shared-model-server mode the MCP server opens its store at startup, without models.

Local mode already opened it while preloading models. In shared mode the
Memory opened only on the first TrueMemory tool call, so a session without
one scheduled no maintenance pass or vector gap seed in this long-lived
process. Synthetic store; every model entry point is made to fail.
"""

import ast
import hashlib
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sqlite_vec

from truememory import maintenance
from truememory.client import Memory

ROOT = Path(__file__).resolve().parents[1]


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


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """30 memories whose layers were never attempted, 10 of them stored without vectors by an old release."""
    path = tmp_path / "memories.db"
    with patch("truememory.vector_search.get_model", return_value=_encoder()), \
            patch.object(maintenance.MaintenanceCoordinator, "request_layers", lambda self, **kwargs: False):
        memory = Memory(path=path)
        for index in range(30):
            memory.add(content=f"synthetic memory {index} about project alpha", user_id="u1")
        memory.close()
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    for table in ("vec_messages", "vec_messages_sep"):
        conn.execute(f"DELETE FROM {table} WHERE rowid BETWEEN 5 AND 14")
    conn.execute("DELETE FROM metadata WHERE key = 'vector_backfill_from'")
    conn.commit()
    conn.close()
    return path


def test_startup_open_wakes_maintenance_and_seeds_without_models(store: Path, monkeypatch) -> None:
    import truememory.mcp_server as server
    import truememory.model_client as model_client
    import truememory.reranker as reranker
    import truememory.vector_search as vector_search

    touched = []

    def refuse(name):
        def call(*args, **kwargs):
            touched.append(name)
            raise AssertionError(f"startup open touched {name}")
        return call

    monkeypatch.setattr(vector_search, "get_model", refuse("get_model"))
    monkeypatch.setattr(reranker, "get_reranker", refuse("get_reranker"))
    monkeypatch.setattr(model_client, "_send_request", refuse("model server request"))
    wakes = []
    monkeypatch.setattr(maintenance.MaintenanceCoordinator, "request_layers",
                        lambda self, **kwargs: wakes.append(kwargs) or False)
    monkeypatch.setattr(server, "_DB_PATH", str(store))
    monkeypatch.setattr(server, "_memory", None)

    server._open_store_for_maintenance()
    for thread in threading.enumerate():
        if thread.name == "truememory-store-open":
            thread.join(30)
    memory = server._memory
    try:
        assert memory is not None and memory._engine.conn is not None
        if memory._engine._backfill_thread is not None:
            memory._engine._backfill_thread.join(30)
        assert wakes, "the startup open must schedule maintenance"
    finally:
        if memory is not None:
            memory.close()
    assert touched == []
    conn = sqlite3.connect(store)
    try:
        assert conn.execute("SELECT value FROM metadata WHERE key = 'vector_backfill_from'").fetchone() == ("5",)
    finally:
        conn.close()


def test_shared_mode_startup_calls_the_store_open() -> None:
    tree = ast.parse((ROOT / "truememory" / "mcp_server.py").read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    branch = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
                  and isinstance(node.test.operand, ast.Call)
                  and getattr(node.test.operand.func, "id", None) == "use_model_server")
    local = {node.func.id for statement in branch.body for node in ast.walk(statement)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    shared = {node.func.id for statement in branch.orelse for node in ast.walk(statement)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "_preload_models" in local
    assert "_open_store_for_maintenance" in shared
