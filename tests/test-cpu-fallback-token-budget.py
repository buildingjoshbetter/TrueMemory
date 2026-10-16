"""After an MPS OOM, CPU pieces are sized against the RSS budget, not free system memory."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from truememory import memory_budget
from truememory import model_server as ms
from truememory import mps_utils
from truememory._embed_budget import EMBED_TOKEN_BUDGET
from truememory._rerank_budget import accelerator_token_budget, budget_batch_size

GIB = 1024 ** 3
RESIDENT = 3_560_000_000  # the soak's resident set beside a 4 GiB budget (16 GiB Mac)


class FakeRSS:
    def __init__(self, rss: int) -> None:
        self.rss = rss

    def Process(self) -> "FakeRSS":
        return self

    def virtual_memory(self) -> types.SimpleNamespace:
        # Plenty of free system memory: the old sizing saw no pressure at all.
        return types.SimpleNamespace(available=16 * GIB, total=16 * GIB)

    def memory_info(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(rss=self.rss)


class Embedder:
    """Qwen-shaped: 16 heads, word count stands in for tokens, MPS can OOM."""

    config = types.SimpleNamespace(num_attention_heads=16)

    def __init__(self, device: str) -> None:
        self.device = device
        self.max_seq_length = 2048
        self.calls: list[tuple[str, int, int]] = []
        self.oom_on_mps = True

    def tokenizer(self, texts, truncation, max_length):
        return {"input_ids": [[0] * min(len(text.split()), max_length) for text in texts]}

    def encode(self, texts, batch_size=32, show_progress_bar=False, **kwargs):
        longest = max(len(text.split()) for text in texts)
        self.calls.append((self.device, len(texts), longest))
        if self.device == "mps" and self.oom_on_mps:
            raise RuntimeError("MPS backend out of memory (MPS allocated: 3.9 GiB, max allowed: 4.0 GiB)")
        return np.array([[len(text.split()), index] for index, text in enumerate(texts)], dtype=np.float32)

    def to(self, device: str) -> "Embedder":
        self.device = device
        return self


@pytest.fixture
def rss_budget(monkeypatch):
    sensor = FakeRSS(RESIDENT)
    monkeypatch.setitem(sys.modules, "psutil", sensor)
    monkeypatch.setattr(memory_budget, "process_rss_limit", lambda: 4 * GIB, raising=False)
    return sensor


def _cpu_fit() -> int:
    return int((4 * GIB - RESIDENT) * 0.8) // (16 * 8)


def test_cpu_budget_is_the_rss_room_not_free_system_memory(rss_budget):
    model = Embedder("cpu")
    fit = accelerator_token_budget(model, EMBED_TOKEN_BUDGET)
    assert fit == _cpu_fit() == 4_593_545
    # The soak's first CPU piece was 11 texts of 1,718 tokens: 4.16 GB of attention.
    assert budget_batch_size(1718, 32, EMBED_TOKEN_BUDGET) == 11
    assert budget_batch_size(1718, 32, fit) == 1
    assert 1 * 1718 ** 2 * 16 * 8 < (4 * GIB - RESIDENT)


def test_cpu_budget_floors_at_one_text_and_keeps_room_when_plentiful(rss_budget, monkeypatch):
    model = Embedder("cpu")
    rss_budget.rss = 5 * GIB
    assert accelerator_token_budget(model, EMBED_TOKEN_BUDGET) == 1
    rss_budget.rss = 1 * GIB
    monkeypatch.setattr(memory_budget, "process_rss_limit", lambda: 8 * GIB)
    assert accelerator_token_budget(model, EMBED_TOKEN_BUDGET) == EMBED_TOKEN_BUDGET


def test_models_without_known_heads_keep_their_budget(rss_budget):
    model = types.SimpleNamespace(device="cpu")
    assert accelerator_token_budget(model, EMBED_TOKEN_BUDGET) == EMBED_TOKEN_BUDGET


def test_server_retries_the_oom_piece_and_the_rest_in_rss_sized_pieces(rss_budget, monkeypatch):
    monkeypatch.delenv("TRUEMEMORY_DEVICE", raising=False)
    monkeypatch.setattr(mps_utils, "auto_detect_device", lambda: "mps")
    monkeypatch.setattr(mps_utils, "ensure_mps_memory_budget", lambda device: None)
    # No configured MPS cap: the MPS pieces keep the plain 32 x 1,024^2 budget.
    monkeypatch.setattr(mps_utils, "get_mps_memory_budget", lambda: None)
    server = ms.ModelServer()
    server._write_status_file = lambda: None
    server._embed_device = lambda: "mps"
    model = Embedder("mps")
    monkeypatch.setattr(server, "_build_embed_model", lambda model_id, device: model)
    texts = [" ".join(["memory"] * 1718) for _ in range(22)] + [" ".join(["short"] * 20) for _ in range(10)]
    try:
        response = server.handle_request({"op": "embed", "texts": texts, "tier": "qwen3_256"})
    finally:
        server._workers.shutdown(wait=False)
    assert response["ok"] is True
    # Row order is preserved: each vector carries its own text's length.
    assert [int(row[0]) for row in response["vectors"]] == [1718] * 22 + [20] * 10
    mps_calls = [call for call in model.calls if call[0] == "mps"]
    cpu_calls = [call for call in model.calls if call[0] == "cpu"]
    assert mps_calls == [("mps", 11, 1718)], "The MPS piece is sized by the MPS budget"
    fit = _cpu_fit()
    assert cpu_calls and all(size * longest ** 2 <= fit or size == 1 for _, size, longest in cpu_calls)
    assert all(size == 1 for _, size, longest in cpu_calls if longest == 1718)
    assert sum(size for _, size, _ in cpu_calls) == 32
