"""The fast-lane CPU clone is released when idle and rebuilt in the background."""
from __future__ import annotations

import threading
import time
import types
import weakref

import numpy as np
import pytest

from truememory import model_server as ms

MIB = 1024 ** 2


class FakeRSS:
    """Stands in for psutil; the test sets the sampled resident size."""

    def __init__(self, rss: int) -> None:
        self.rss = rss

    def virtual_memory(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(total=32 * 1024 ** 3)

    def Process(self) -> FakeRSS:
        return self

    def memory_info(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(rss=self.rss)


class Model:
    """A reference cycle, like sentence-transformers models."""

    def __init__(self, device: str | None, value: float = 1.0, encode_hook=None) -> None:
        self.device = device
        self.value = value
        self.encode_hook = encode_hook
        self.cycle = self

    def encode(self, texts, batch_size=32, show_progress_bar=False, **kwargs):
        if self.encode_hook is not None:
            self.encode_hook()
        return np.full((len(texts), 2), self.value, dtype=np.float32)


class Builds:
    """Records clone constructions; the test can hold any build open."""

    def __init__(self) -> None:
        self.devices: list[str | None] = []
        self.refs: list[weakref.ref] = []
        self.gate: threading.Event | None = None
        self.started = threading.Event()
        self.encode_hook = None

    def __call__(self, model_id: str, device: str | None) -> Model:
        self.devices.append(device)
        self.started.set()
        if self.gate is not None and not self.gate.wait(10):
            raise AssertionError("build gate timed out")
        model = Model(device, value=float(len(self.devices)), encode_hook=self.encode_hook)
        self.refs.append(weakref.ref(model))
        return model


def _close(server: ms.ModelServer) -> None:
    server._stopped.set()
    rebuild = server._fast_rebuild
    if rebuild is not None:
        rebuild.join(10)
    server._workers.shutdown(wait=False)
    if server._memory_owner is not None:
        server._memory_owner.close()


def _server(monkeypatch, *, idle: str | None = None, budgeted: bool = False):
    if idle is None:
        monkeypatch.delenv("TRUEMEMORY_FAST_LANE_IDLE_SECONDS", raising=False)
    else:
        monkeypatch.setenv("TRUEMEMORY_FAST_LANE_IDLE_SECONDS", idle)
    rss = None
    if budgeted:
        rss = FakeRSS(500 * MIB)
        monkeypatch.setattr(ms, "psutil", rss)
        monkeypatch.setenv("TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB", "3072")
    server = ms.ModelServer(automatic_rss_budget=budgeted)
    server._write_status_file = lambda: None
    builds = Builds()
    monkeypatch.setattr(server, "_build_embed_model", builds)
    server._publish_embed_state(ms._EmbedState(model=Model("mps", value=-1.0), tier="", model_id="qwen3_256"))
    return server, builds, rss


@pytest.fixture
def lane(monkeypatch):
    server, builds, _ = _server(monkeypatch)
    yield server, builds
    _close(server)


class Batch:
    """Holds the main inference owner, like a long embed batch or a rerank."""

    def __init__(self, server: ms.ModelServer) -> None:
        self.server = server
        self.release = threading.Event()
        held = threading.Event()

        def run() -> None:
            with server._inference_lock:
                held.set()
                self.release.wait(10)

        self.thread = threading.Thread(target=run)
        self.thread.start()
        assert held.wait(5)

    def end(self) -> None:
        self.release.set()
        self.thread.join(5)


def _contended(server: ms.ModelServer, text: str = "synthetic query") -> dict | None:
    with server._memory_scope():
        return server._handle_fast_embed([text], "", ms._RequestDeadline(time.monotonic() + 5))


def _settle(server: ms.ModelServer) -> None:
    """Wait for any background reload to finish."""
    rebuild = server._fast_rebuild
    if rebuild is not None:
        rebuild.join(5)
        assert not rebuild.is_alive()


def _first_clone(server: ms.ModelServer, builds: Builds) -> weakref.ref:
    batch = Batch(server)
    try:
        response = _contended(server)
    finally:
        batch.end()
    assert response is not None and response["ok"] is True
    assert builds.devices == ["cpu"] and server._fast_encoder is builds.refs[0]()
    return builds.refs[0]


def test_idle_clone_is_retired_and_its_memory_released(lane):
    server, builds = lane
    clone = _first_clone(server, builds)
    used = server._fast_last_used
    assert server._fast_idle_seconds == 600
    assert server._retire_idle_fast_encoder(now=used + 599) is False
    assert server._fast_encoder is clone()
    assert server._retire_idle_fast_encoder(now=used + 600) is True
    assert server._fast_encoder is None and server._fast_model_id is None
    assert clone() is None, "Retired clone must be collected, not left as a reference cycle"


def test_a_running_batch_counts_as_contention(lane):
    server, builds = lane
    clone = _first_clone(server, builds)
    batch = Batch(server)
    try:
        assert server._retire_idle_fast_encoder(now=server._fast_last_used + 3600) is False
        assert server._fast_encoder is clone()
    finally:
        batch.end()
    assert server._retire_idle_fast_encoder(now=server._fast_last_used + 3600) is True


def test_contention_after_retire_rebuilds_in_background_without_waiting(lane):
    server, builds = lane
    _first_clone(server, builds)
    assert server._retire_idle_fast_encoder(now=server._fast_last_used + 600)
    builds.gate = threading.Event()
    builds.started.clear()
    batch = Batch(server)
    try:
        started = time.monotonic()
        with pytest.raises(ms._ProcessMemoryRefused, match="long batch"):
            _contended(server)
        elapsed = time.monotonic() - started
        assert 0.9 <= elapsed < 2.0, elapsed
        assert builds.started.wait(5), "A contended query after retirement must start a rebuild"
        rebuild = server._fast_rebuild
        assert rebuild is not None and rebuild.is_alive()
        # A second contended query neither waits on nor duplicates the rebuild.
        with pytest.raises(ms._ProcessMemoryRefused):
            _contended(server)
        assert server._fast_rebuild is rebuild and builds.devices == ["cpu", "cpu"]
        builds.gate.set()
        rebuild.join(5)
        assert not rebuild.is_alive() and server._fast_rebuild is None
        assert server._fast_encoder is builds.refs[1]()
        served = _contended(server)
        assert served is not None and served["vectors"][0][0] == 2.0
        assert builds.devices == ["cpu", "cpu"]
    finally:
        builds.gate.set()
        batch.end()


def test_short_batch_after_retire_serves_from_the_main_model(lane):
    server, builds = lane
    _first_clone(server, builds)
    assert server._retire_idle_fast_encoder(now=server._fast_last_used + 600)
    builds.gate = threading.Event()
    builds.started.clear()
    batch = Batch(server)
    threading.Timer(0.2, batch.end).start()
    try:
        started = time.monotonic()
        response = server._handle_fast_embed(["synthetic query"], "", ms._RequestDeadline(time.monotonic() + 5))
        if response is None:
            # The bounded fallback hands the query back to the main locked path.
            response = server.handle_request({"op": "embed", "texts": ["synthetic query"], "tier": ""})
        assert response["ok"] is True and response["vectors"][0][0] == -1.0
        assert time.monotonic() - started < 1.5
        assert builds.started.wait(5) and server._fast_encoder is None
    finally:
        builds.gate.set()
        batch.end()


def test_rebuild_still_requires_room_for_the_clone(monkeypatch):
    server, builds, rss = _server(monkeypatch, budgeted=True)
    try:
        _first_clone(server, builds)
        assert server._retire_idle_fast_encoder(now=server._fast_last_used + 600)
        # 1,200 MiB resident plus a 2.4 GB CPU clone exceeds the 3,072 MiB budget.
        rss.rss = 1200 * MIB
        batch = Batch(server)
        try:
            with pytest.raises(ms._ProcessMemoryRefused):
                _contended(server)
            _settle(server)
        finally:
            batch.end()
        assert builds.devices == ["cpu"] and server._fast_encoder is None
        rss.rss = 500 * MIB
        batch = Batch(server)
        try:
            # With room, the reload is admitted and serves this same query.
            response = _contended(server)
            _settle(server)
        finally:
            batch.end()
        assert response["ok"] is True and response["vectors"][0][0] == 2.0
        assert builds.devices == ["cpu", "cpu"] and server._fast_encoder is builds.refs[1]()
    finally:
        _close(server)


def test_new_residency_after_retire_keeps_the_first_contention_build(lane):
    server, builds = lane
    _first_clone(server, builds)
    assert server._retire_idle_fast_encoder(now=server._fast_last_used + 600)
    server._publish_embed_state(ms._EmbedState(model=Model("mps", value=-3.0), tier="", model_id="qwen3_256"))
    batch = Batch(server)
    try:
        response = _contended(server)
    finally:
        batch.end()
    assert response["ok"] is True and response["vectors"][0][0] == 2.0
    assert server._fast_rebuild is None


@pytest.mark.parametrize("value, expected", [(None, 600), ("45", 45), ("0", 0)])
def test_idle_seconds_setting(monkeypatch, value, expected):
    server, builds, _ = _server(monkeypatch, idle=value)
    try:
        assert server._fast_idle_seconds == expected
        _first_clone(server, builds)
        retired = server._retire_idle_fast_encoder(now=server._fast_last_used + 10 ** 9)
        assert retired is (expected != 0)
        assert (server._fast_encoder is None) is (expected != 0)
    finally:
        _close(server)


@pytest.mark.parametrize("value", ["-5", "1.5", "ten", ""])
def test_malformed_idle_seconds_are_rejected_like_other_server_settings(monkeypatch, value):
    monkeypatch.setenv("TRUEMEMORY_FAST_LANE_IDLE_SECONDS", value)
    with pytest.raises(ValueError, match="TRUEMEMORY_FAST_LANE_IDLE_SECONDS"):
        ms.ModelServer()


def test_retirer_thread_drops_an_idle_clone(monkeypatch):
    server, builds, _ = _server(monkeypatch, idle="1")
    try:
        clone = _first_clone(server, builds)
        retirer = threading.Thread(target=server._idle_retirer)
        retirer.start()
        deadline = time.monotonic() + 5
        while server._fast_encoder is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server._fast_encoder is None and clone() is None
        server._stopped.set()
        retirer.join(5)
        assert not retirer.is_alive()
    finally:
        _close(server)


def test_never_retiring_keeps_the_clone_while_the_retirer_runs(monkeypatch):
    server, builds, _ = _server(monkeypatch, idle="0")
    monkeypatch.setattr(server, "_PARKED_RERANKER_IDLE_SECONDS", 4)
    try:
        clone = _first_clone(server, builds)
        server._fast_last_used -= 10 ** 6
        retirer = threading.Thread(target=server._idle_retirer)
        retirer.start()
        time.sleep(2.5)
        assert server._fast_encoder is clone()
        server._stopped.set()
        retirer.join(5)
        assert not retirer.is_alive()
    finally:
        _close(server)


def test_retire_does_not_take_a_clone_from_an_in_flight_request(lane):
    server, builds = lane
    clone = _first_clone(server, builds)
    encoding, release = threading.Event(), threading.Event()

    def hold() -> None:
        encoding.set()
        assert release.wait(5)

    clone().encode_hook = hold
    results = []
    batch = Batch(server)
    request = threading.Thread(target=lambda: results.append(_contended(server)))
    try:
        request.start()
        assert encoding.wait(5)
        # The batch is over; only the request's own use of the clone remains.
        batch.end()
        assert server._retire_idle_fast_encoder(now=server._fast_last_used + 3600) is False
        assert server._fast_encoder is clone()
    finally:
        release.set()
        request.join(5)
        batch.end()
    assert results and results[0]["ok"] is True and results[0]["vectors"][0][0] == 1.0
    clone().encode_hook = None
    # The finished request is fresh use; only a full idle period retires the clone.
    used = server._fast_last_used
    assert server._retire_idle_fast_encoder(now=used + 599) is False
    assert server._retire_idle_fast_encoder(now=used + 600) is True


def test_reload_finishing_within_the_bounded_wait_serves_the_query(lane):
    server, builds = lane
    _first_clone(server, builds)
    assert server._retire_idle_fast_encoder(now=server._fast_last_used + 600)
    builds.gate = threading.Event()
    builds.started.clear()
    threading.Timer(0.3, builds.gate.set).start()
    batch = Batch(server)
    try:
        started = time.monotonic()
        response = _contended(server)
        elapsed = time.monotonic() - started
    finally:
        builds.gate.set()
        batch.end()
        _settle(server)
    assert response is not None and response["ok"] is True
    assert response["vectors"][0][0] == 2.0 and elapsed < 1.0, elapsed


def test_fast_lane_paths_never_touch_the_accelerator_cache(lane, monkeypatch):
    server, builds = lane

    def flush():
        raise AssertionError("MPS cache emptied outside the inference lock")

    monkeypatch.setattr(server, "_discard_models", flush)
    clone = _first_clone(server, builds)
    assert server._retire_idle_fast_encoder(now=server._fast_last_used + 600)
    assert clone() is None
    batch = Batch(server)
    try:
        assert _contended(server)["ok"] is True
        _settle(server)
    finally:
        batch.end()
    assert server._fast_encoder is builds.refs[1]()


def test_cpu_float32_weights_live_in_returnable_mappings(monkeypatch):
    torch = pytest.importorskip("torch")
    import mmap

    from truememory import mps_utils

    mapped = []
    real = mmap.mmap

    def recording(*args, **kwargs):
        mapped.append(args[1] if len(args) > 1 else kwargs.get("length"))
        return real(*args, **kwargs)

    monkeypatch.setattr(mmap, "mmap", recording)
    monkeypatch.setattr(mps_utils.sys, "platform", "darwin")
    model = torch.nn.Sequential(torch.nn.Embedding(64, 8), torch.nn.Linear(8, 4)).to(torch.bfloat16)
    model.device = torch.device("cpu")  # as SentenceTransformer reports
    expected = [parameter.detach().float().clone() for parameter in model.parameters()]
    assert mps_utils.prefer_float32_on_cpu(model) is model
    parameters = list(model.parameters())
    assert [parameter.dtype for parameter in parameters] == [torch.float32] * 3
    assert all(torch.equal(parameter, value) for parameter, value in zip(parameters, expected))
    assert sorted(mapped) == sorted(parameter.numel() * 4 for parameter in parameters)
    assert model(torch.tensor([[1, 2]])).dtype == torch.float32


def test_fast_lane_never_empties_the_mps_cache_beside_a_running_batch(lane, monkeypatch):
    """Regression: torch.mps.empty_cache() on one thread while another thread
    synchronizes MPS work aborts the process ("commit an already committed
    command buffer"). The fast lane runs while a batch owns the main model."""
    import sys

    from truememory import mps_utils

    server, builds = lane
    emptied = []
    torch = types.SimpleNamespace(mps=types.SimpleNamespace(
        empty_cache=lambda: emptied.append(server._inference_lock.locked()),
    ))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(mps_utils, "_mps_budget", object())
    _first_clone(server, builds)
    assert server._retire_idle_fast_encoder(now=server._fast_last_used + 600)
    batch = Batch(server)
    try:
        assert _contended(server)["ok"] is True
        _settle(server)
    finally:
        batch.end()
    assert builds.devices == ["cpu", "cpu"]
    assert emptied == [], "MPS cache emptied while another thread owned MPS work"
