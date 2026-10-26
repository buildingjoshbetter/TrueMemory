"""Hooks leave background maintenance to long-lived processes.

A hook process opens the store for one prompt and exits. Maintenance it woke
would be cut off by its close() (or exit) and make that close() wait, so hook
Memories pass auto_maintenance=False. The MCP server's Memory and the ingest
pipeline's keep waking it. Synthetic stores and a deterministic mock encoder.
"""

import ast
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from truememory import maintenance
from truememory.client import Memory

ROOT = Path(__file__).resolve().parents[1]
# Every module a hook process runs. The ingest pipeline is not one: the Stop
# hook (and Hermes) start it as a separate worker process.
HOOK_DIRS = ("truememory/ingest/hooks", "truememory/hooks")
LONG_LIVED_MODULES = ("truememory/mcp_server.py", "truememory/ingest/pipeline.py")


def _store_calls(relative: str) -> list[ast.Call]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id in {"Memory", "TrueMemoryEngine"}]


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
def encoder():
    with patch("truememory.vector_search.get_model", return_value=_encoder()):
        yield


@pytest.fixture
def eligible(tmp_path: Path, encoder) -> Path:
    """A store whose layers were never attempted, so the next wake requests a pass."""
    path = tmp_path / "memories.db"
    with patch.object(maintenance.MaintenanceCoordinator, "request_layers", lambda self, **kwargs: False):
        seed = Memory(path=path)
        for index in range(30):
            seed.add(content=f"synthetic memory {index} about project alpha", user_id="u1")
        seed.close()
    return path


@pytest.fixture
def wakes(monkeypatch) -> list:
    calls = []

    def request_layers(self, **kwargs):
        calls.append(kwargs)
        return False  # Record the wake; start no worker.

    monkeypatch.setattr(maintenance.MaintenanceCoordinator, "request_layers", request_layers)
    return calls


@pytest.fixture
def hook(monkeypatch):
    from truememory.ingest.hooks import user_prompt_submit
    monkeypatch.setattr("truememory.model_client.set_request_timeout", lambda timeout: None)
    return user_prompt_submit


def test_every_hook_memory_leaves_maintenance_to_a_long_lived_process() -> None:
    found = []
    for directory in HOOK_DIRS:
        for module in sorted((ROOT / directory).rglob("*.py")):
            relative = module.relative_to(ROOT).as_posix()
            for call in _store_calls(relative):
                found.append(relative)
                assert any(keyword.arg == "auto_maintenance" and isinstance(keyword.value, ast.Constant)
                           and keyword.value.value is False for keyword in call.keywords), f"{relative}:{call.lineno}"
    assert {"truememory/ingest/hooks/user_prompt_submit.py", "truememory/ingest/hooks/session_start.py",
            "truememory/ingest/hooks/compact.py", "truememory/hooks/core.py"} <= set(found)


def test_long_lived_processes_keep_auto_maintenance() -> None:
    for relative in LONG_LIVED_MODULES:
        calls = _store_calls(relative)
        assert calls, relative
        for call in calls:
            assert not any(keyword.arg == "auto_maintenance" for keyword in call.keywords), f"{relative}:{call.lineno}"


def test_hook_recall_wakes_no_maintenance(eligible: Path, wakes: list, hook) -> None:
    context = hook._try_auto_recall("what do you remember about project alpha?", "u1", str(eligible))
    assert context is not None and "project alpha" in context
    assert wakes == []
    # Control: the same open from a default Memory wakes maintenance on this store.
    with Memory(path=eligible) as memory:
        memory.search("project alpha", user_id="u1", limit=3, _skip_reranker=True)
    assert wakes


def test_hook_store_wakes_no_maintenance(eligible: Path, wakes: list, hook, monkeypatch) -> None:
    closed = []
    real_close = Memory.close

    def close(self):
        closed.append(getattr(self._engine, "_auto_maintenance", True))
        real_close(self)

    monkeypatch.setattr(Memory, "close", close)
    hook._try_per_exchange_store("I prefer tango with lima on project kilo from now on",
                                 "synthetic-session", "u1", str(eligible), "max")
    assert closed == [False]  # The hook opened the store, without maintenance.
    assert wakes == []


def test_hook_close_has_no_worker_to_wait_for(eligible: Path) -> None:
    coordinator = maintenance.get_coordinator(eligible)
    with Memory(path=eligible, auto_maintenance=False) as memory:
        assert memory.search("project alpha", user_id="u1", limit=3, _skip_reranker=True)
        memory.add(content="synthetic memory stored from a hook", user_id="u1")
        assert not coordinator.snapshot()["active"]
        assert memory._engine._backfill_thread is None
    assert coordinator.wait(0)


def test_mcp_server_memory_still_wakes_maintenance(eligible: Path, wakes: list, monkeypatch) -> None:
    import truememory.mcp_server as server
    monkeypatch.setattr(server, "_DB_PATH", str(eligible))
    monkeypatch.setattr(server, "_memory", None)
    memory = server._get_memory()
    try:
        assert getattr(memory._engine, "_auto_maintenance", True)
        assert memory.search("project alpha", user_id="u1", limit=3, _skip_reranker=True)
    finally:
        memory.close()
    assert wakes, "the MCP server's Memory must schedule maintenance"


def test_ingest_worker_memory_still_wakes_maintenance(eligible: Path, wakes: list) -> None:
    from truememory.ingest.models import LLMConfig
    from truememory.ingest.pipeline import IngestionPipeline
    worker = IngestionPipeline(db_path=eligible, llm_config=LLMConfig(provider="ollama"), use_llm_dedup=False)
    try:
        assert getattr(worker.memory._engine, "_auto_maintenance", True)
        worker.memory.add(content="synthetic fact stored by the ingest worker", user_id="u1")
    finally:
        worker.memory.close()
    assert wakes, "the ingest worker's Memory must schedule maintenance"
