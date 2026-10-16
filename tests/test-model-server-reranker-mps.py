"""Rerankers are admitted against the MPS budget; one model's OOM never moves the others."""
from __future__ import annotations

import sys
import time
import types
import weakref

import numpy as np
import pytest

from truememory import model_server as ms
from truememory import mps_utils

GIB = 1024 ** 3
MODERNBERT = "Alibaba-NLP/gte-reranker-modernbert-base"
BGE = "BAAI/bge-reranker-v2-m3"
MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Bytes each model holds on MPS as loaded (checkpoint dtype), like the server's table.
SIZES = {"embed": 1_200_000_000, MODERNBERT: 600_000_000, BGE: 2_272_000_000, MINILM: 92_000_000,
         "synthetic/oversized": 600_000_000}
# One forward pass of activations on top of the resident weights.
ACTIVATION = 1_000_000_000


class FakeMPS:
    """Live MPS bytes of fake models; going past the budget raises the real OOM text."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.live: dict[int, tuple[weakref.ref, int]] = {}

    def allocated(self) -> int:
        self.live = {key: value for key, value in self.live.items() if value[0]() is not None}
        return sum(size for _, size in self.live.values())

    def allocate(self, owner: object, size: int) -> None:
        if self.allocated() + size > self.budget:
            raise RuntimeError(
                f"MPS backend out of memory (MPS allocated: {self.allocated()} B, "
                f"max allowed: {self.budget} B). Tried to allocate {size} B on private pool."
            )
        self.live[id(owner)] = (weakref.ref(owner), size)


class Resident:
    """Any object holding MPS memory, such as Qwen's weights or a forward pass."""


class Reranker:
    def __init__(self, name: str, device: str | None, mps: FakeMPS) -> None:
        self.name, self.device, self.mps = name, device, mps
        self.cycle = self
        self.predictions = 0
        if device == "mps":
            mps.allocate(self, SIZES[name])

    def predict(self, pairs, batch_size=32, show_progress_bar=False, **kwargs):
        if self.device == "mps":
            # Activations need room beside every resident model.
            scratch = Resident()
            self.mps.allocate(scratch, ACTIVATION)
            del scratch
        self.predictions += 1
        return np.full(len(pairs), 0.5, dtype=np.float32)


@pytest.fixture
def mps_host(monkeypatch):
    mps = FakeMPS(4 * GIB)
    built: list[tuple[str, str | None]] = []

    def cross_encoder(name, device=None):
        built.append((name, device))
        if name == "synthetic/oversized" and device == "mps":
            raise RuntimeError("MPS backend out of memory (MPS allocated: 3.90 GiB, max allowed: 4.00 GiB)")
        return Reranker(name, device, mps)

    monkeypatch.delenv("TRUEMEMORY_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=cross_encoder))
    monkeypatch.setattr(mps_utils, "auto_detect_device", lambda: "mps")
    monkeypatch.setattr(mps_utils, "ensure_mps_memory_budget", lambda device: None)
    monkeypatch.setattr(mps_utils, "mps_free_bytes", lambda: mps.budget - mps.allocated(), raising=False)
    server = ms.ModelServer()
    server._write_status_file = lambda: None
    embed = Resident()
    mps.allocate(embed, SIZES["embed"])  # Qwen3-Embedding resident on MPS
    yield server, mps, built, embed
    server._workers.shutdown(wait=False)


def _rerank(server: ms.ModelServer, name: str) -> dict:
    return server.handle_request({"op": "rerank", "pairs": [["synthetic query", "synthetic memory"]] * 3,
                                  "model_name": name})


def _no_path_wide_rerank_degradation(server: ms.ModelServer) -> bool:
    return "rerank" not in server._sticky_cpu


def test_deep_reranker_beside_modernbert_keeps_modernbert_on_mps(mps_host):
    server, mps, built, _embed = mps_host
    assert _rerank(server, MODERNBERT)["ok"] is True
    modernbert = server._reranker
    assert modernbert.device == "mps"
    # bge-reranker-v2-m3 holds 2.27 GB: beside Qwen, ModernBERT and room for a
    # forward pass it cannot fit the 4 GiB budget, so only the newcomer goes to CPU.
    assert _rerank(server, BGE)["ok"] is True
    assert server._reranker.device == "cpu" and server._reranker_name == BGE
    assert built == [(MODERNBERT, "mps"), (BGE, "cpu")]
    assert _no_path_wide_rerank_degradation(server) and not server._sticky_cpu
    # ModernBERT stayed resident on MPS and serves again without a reload.
    assert _rerank(server, MODERNBERT)["ok"] is True
    assert server._reranker is modernbert and modernbert.device == "mps"
    assert modernbert.predictions == 2 and len(built) == 2
    assert _rerank(server, BGE)["ok"] is True and len(built) == 2


def test_construction_oom_is_scoped_to_the_failing_model(mps_host):
    server, mps, built, _embed = mps_host
    assert _rerank(server, "synthetic/oversized")["ok"] is True
    assert server._reranker.device == "cpu"
    assert _no_path_wide_rerank_degradation(server)
    assert server._sticky_cpu == {"rerank:synthetic/oversized"}
    assert _rerank(server, MODERNBERT)["ok"] is True
    assert server._reranker.device == "mps"
    assert built == [("synthetic/oversized", "mps"), ("synthetic/oversized", "cpu"), (MODERNBERT, "mps")]


def test_inference_oom_moves_only_that_reranker(mps_host, monkeypatch):
    server, mps, built, _embed = mps_host
    assert _rerank(server, MINILM)["ok"] is True
    assert server._reranker.device == "mps"
    assert _rerank(server, MODERNBERT)["ok"] is True
    modernbert = server._reranker

    def oom(*args, **kwargs):
        raise RuntimeError("MPS backend out of memory (MPS allocated: 3.9 GiB, max allowed: 4.0 GiB)")

    # A long batch on ModernBERT runs out of MPS room once.
    modernbert.predict = oom
    modernbert = None
    assert _rerank(server, MODERNBERT)["ok"] is True
    assert server._reranker.device == "cpu"
    assert server._sticky_cpu == {f"rerank:{MODERNBERT}"}
    assert _rerank(server, MINILM)["ok"] is True
    assert server._reranker.device == "mps" and built.count((MINILM, "mps")) == 1


def test_cpu_placement_is_released_when_idle_and_retried_on_mps(mps_host):
    server, mps, built, embed = mps_host
    assert _rerank(server, "synthetic/oversized")["ok"] is True
    assert _rerank(server, MODERNBERT)["ok"] is True
    # The CPU copy of the failed model is parked; past the idle period it is released.
    parked = weakref.ref(server._parked_rerankers["synthetic/oversized"][0])
    assert server._retire_parked_rerankers(now=time.monotonic() + server._PARKED_RERANKER_IDLE_SECONDS)
    assert parked() is None and "synthetic/oversized" not in server._parked_rerankers
    assert server._sticky_cpu == set()
    # Its next load tries MPS again instead of inheriting the old placement.
    assert _rerank(server, "synthetic/oversized")["ok"] is True
    assert built[-2:] == [("synthetic/oversized", "mps"), ("synthetic/oversized", "cpu")]


def test_idle_parked_reranker_is_evicted_only_when_that_makes_room(mps_host, monkeypatch):
    server, mps, built, embed = mps_host
    assert _rerank(server, MODERNBERT)["ok"] is True
    modernbert = weakref.ref(server._reranker)
    # A newcomer that fits once ModernBERT leaves MPS: 1.2 + 0.8 + reserve <= 4 GiB.
    monkeypatch.setitem(SIZES, "synthetic/medium", 800_000_000)
    monkeypatch.setitem(server._MPS_WEIGHT_BYTES, "synthetic/medium", 800_000_000)
    assert _rerank(server, "synthetic/medium")["ok"] is True
    # Recently used: no eviction, the newcomer goes to CPU instead of thrashing.
    assert server._reranker.device == "cpu" and modernbert() is not None
    server._reranker = None
    server._reranker_name = None
    entry = server._parked_rerankers[MODERNBERT]
    server._parked_rerankers[MODERNBERT] = (entry[0], entry[1], entry[2] - server._RERANKER_EVICT_IDLE_SECONDS)
    server._parked_rerankers.pop("synthetic/medium", None)
    entry = None
    assert _rerank(server, "synthetic/medium")["ok"] is True
    assert server._reranker.device == "mps" and modernbert() is None
