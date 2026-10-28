"""Memory pressure refuses work without stopping the server or stranding models."""
from __future__ import annotations

import socket
import struct
import sys
import threading
import time
import types
import weakref

import numpy as np
import pytest

from truememory import model_server as ms
from truememory import mps_utils

MIB = 1024 ** 2
MODERNBERT = "Alibaba-NLP/gte-reranker-modernbert-base"


class FakeRSS:
    """Stands in for psutil; the test sets the sampled resident size."""

    def __init__(self, rss: int, physical: int = 32 * 1024 ** 3) -> None:
        self.rss = rss
        self.physical = physical

    def virtual_memory(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(total=self.physical)

    def Process(self) -> FakeRSS:
        return self

    def memory_info(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(rss=self.rss)


class Model:
    """A reference cycle, like sentence-transformers models."""

    def __init__(self, device: str | None) -> None:
        self.device = device
        self.cycle = self

    def encode(self, texts, batch_size=32, show_progress_bar=False, **kwargs):
        return np.ones((len(texts), 2), dtype=np.float32)


def _close(server: ms.ModelServer) -> None:
    server._workers.shutdown(wait=False)
    if server._memory_owner is not None:
        server._memory_owner.close()


@pytest.fixture
def budgeted(monkeypatch):
    rss = FakeRSS(500 * MIB)
    monkeypatch.setattr(ms, "psutil", rss)
    monkeypatch.setenv("TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB", "3072")
    server = ms.ModelServer(automatic_rss_budget=True)
    server._write_status_file = lambda: None
    yield server, rss
    _close(server)


@pytest.fixture
def plain():
    server = ms.ModelServer()
    server._write_status_file = lambda: None
    yield server
    _close(server)


@pytest.mark.skipif(not ms._USE_UNIX, reason="Unix socket transport")
def test_header_reads_are_not_refused_by_memory_pressure(budgeted):
    server, rss = budgeted
    rss.rss = 4096 * MIB
    client, peer = socket.socketpair()
    try:
        body = b'{"op": "ping"}'
        client.sendall(struct.pack(ms._HEADER_FMT, len(body)) + body)
        client.shutdown(socket.SHUT_WR)
        client.settimeout(10)
        admitted = server._admit_client(peer)
        assert admitted is not None and admitted.length == len(body)
        server._serve_client(peer, admitted)
        header = client.recv(ms._HEADER_SIZE)
        response = client.recv(struct.unpack(ms._HEADER_FMT, header)[0])
        assert b'"server_busy"' in response
        assert not server._stopped.is_set()
    finally:
        client.close()
        peer.close()


def test_accept_refusal_rejects_one_connection_and_keeps_serving(plain, monkeypatch):
    def refuse(*args, **kwargs):
        raise ms._ProcessMemoryRefused("Model server busy: synthetic pressure")

    monkeypatch.setattr(plain, "_recv_exact", refuse)
    client, peer = socket.socketpair()
    try:
        assert plain._admit_client(peer) is None
        assert not plain._stopped.is_set()
        assert plain._staging_lock.acquire(blocking=False)
        plain._staging_lock.release()
    finally:
        client.close()
        peer.close()


def test_construction_is_admitted_with_model_size(budgeted, monkeypatch):
    server, rss = budgeted
    builds = []
    monkeypatch.setattr(server, "_build_embed_model", lambda model_id, device: builds.append(device) or Model(device))
    monkeypatch.setattr(mps_utils, "auto_detect_device", lambda: "cpu")
    # 1,200 MiB resident plus 2.4 GB of Qwen weights on CPU exceeds the 3,072 MiB budget.
    rss.rss = 1200 * MIB
    with server._lock, pytest.raises(ms._ProcessMemoryRefused):
        server._get_embed_model("base")
    assert builds == []
    rss.rss = 500 * MIB
    with server._lock:
        assert server._get_embed_model("base").device in (None, "cpu", "mps")
    assert len(builds) == 1


def test_refused_construction_releases_the_built_model(budgeted, monkeypatch):
    server, rss = budgeted
    built = []

    def build(model_id, device):
        model = Model(device)
        built.append(weakref.ref(model))
        rss.rss = 3500 * MIB
        return model

    monkeypatch.setattr(server, "_build_embed_model", build)
    with server._lock, pytest.raises(ms._ProcessMemoryRefused):
        server._get_embed_model("base")
    assert server._embed_state is None
    assert built and built[0]() is None


@pytest.mark.parametrize("kind", ["embed", "rerank"])
def test_construction_oom_degrades_to_cpu_once(plain, monkeypatch, kind):
    builds = []

    def make(device):
        builds.append(device)
        if device != "cpu":
            raise RuntimeError("MPS backend out of memory (MPS allocated: 1.40 GiB, max allowed: 1.50 GiB)")
        return Model(device)

    monkeypatch.delenv("TRUEMEMORY_DEVICE", raising=False)
    monkeypatch.setattr(mps_utils, "auto_detect_device", lambda: "mps")
    monkeypatch.setattr(mps_utils, "ensure_mps_memory_budget", lambda device: None)
    if kind == "embed":
        monkeypatch.setattr(plain, "_build_embed_model", lambda model_id, device: make(device))

        def load():
            return plain._get_embed_model("base")

        def forget():
            plain._publish_embed_state(None)
    else:
        monkeypatch.setitem(sys.modules, "sentence_transformers",
                            types.SimpleNamespace(CrossEncoder=lambda name, device=None: make(device)))

        def load():
            return plain._get_reranker(MODERNBERT)

        def forget():
            plain._reranker = None

    with plain._lock:
        assert load().device == "cpu"
    assert builds[-1] == "cpu" and builds[0] != "cpu"
    # A reranker that does not fit moves alone; the embedding path is one model.
    assert plain._sticky_cpu == ({"embed"} if kind == "embed" else {f"rerank:{MODERNBERT}"})
    forget()
    with plain._lock:
        assert load().device == "cpu"
    assert builds[2:] == ["cpu"]


def test_non_memory_construction_errors_still_raise(plain, monkeypatch):
    def broken(model_id, device):
        raise RuntimeError("synthetic missing weights")

    monkeypatch.setattr(plain, "_build_embed_model", broken)
    with plain._lock, pytest.raises(RuntimeError, match="missing weights"):
        plain._get_embed_model("base")
    assert "embed" not in plain._sticky_cpu


def test_flush_collects_models_before_emptying_the_cache(monkeypatch):
    order = []
    torch = types.SimpleNamespace(
        mps=types.SimpleNamespace(empty_cache=lambda: order.append("empty_cache")),
        cuda=types.SimpleNamespace(empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(mps_utils, "get_mps_memory_budget", lambda: object())
    monkeypatch.setattr(mps_utils, "gc", types.SimpleNamespace(collect=lambda: order.append("collect")))
    mps_utils.flush_mps_cache()
    assert order == ["collect", "empty_cache"]


def test_single_text_waits_for_bookkeeping_instead_of_cloning(plain, monkeypatch):
    builds = []
    monkeypatch.setattr(plain, "_build_embed_model", lambda model_id, device: builds.append(device) or Model(device))
    main = Model("cpu")
    plain._publish_embed_state(ms._EmbedState(model=main, tier="", model_id="model2vec"))
    held = threading.Event()

    def bookkeeping():
        with plain._lock:
            held.set()
            time.sleep(0.05)

    worker = threading.Thread(target=bookkeeping)
    worker.start()
    held.wait()
    response = plain._handle_fast_embed(["synthetic"], "")
    worker.join()
    assert response is not None and response["ok"] is True
    assert builds == [] and plain._fast_encoder is None


@pytest.mark.parametrize("platform, physical_gib, expected_gib", [
    ("linux", 4, 2), ("linux", 8, 3), ("linux", 12, 3), ("linux", 16, 4), ("linux", 32, 8), ("linux", 64, 8),
    ("darwin", 4, 2), ("darwin", 8, 4), ("darwin", 12, 4), ("darwin", 16, 4), ("darwin", 18, 4.5),
    ("darwin", 24, 6), ("darwin", 32, 6), ("darwin", 64, 6), ("darwin", 128, 6),
])
def test_small_hosts_keep_room_for_base_tier_models(monkeypatch, platform, physical_gib, expected_gib):
    from truememory import memory_budget
    sensor = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(total=physical_gib * 1024 ** 3))
    monkeypatch.setattr(ms, "psutil", sensor)
    monkeypatch.setitem(sys.modules, "psutil", sensor)
    monkeypatch.setattr(sys, "platform", platform)
    assert ms._automatic_rss_budget_bytes() == int(expected_gib * 1024 ** 3)
    assert memory_budget.automatic_budget_bytes() == int(expected_gib * 1024 ** 3)


class HalfPrecisionModel:
    def __init__(self, device: str, dtype) -> None:
        self.device = device
        self.dtype = dtype

    def parameters(self):
        return iter([types.SimpleNamespace(dtype=self.dtype)])

    def float(self):
        self.dtype = "float32"
        return self


@pytest.mark.parametrize("platform, device, expected", [
    ("darwin", "cpu", "float32"), ("darwin", "mps", "bfloat16"), ("linux", "cpu", "bfloat16"),
])
def test_cpu_models_on_macos_run_in_float32(monkeypatch, platform, device, expected):
    torch = types.SimpleNamespace(bfloat16="bfloat16", float16="float16")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(mps_utils.sys, "platform", platform)
    model = HalfPrecisionModel(device, "bfloat16")
    assert mps_utils.prefer_float32_on_cpu(model) is model
    assert model.dtype == expected


def test_static_models_are_left_alone(monkeypatch):
    monkeypatch.setattr(mps_utils.sys, "platform", "darwin")
    model = types.SimpleNamespace(device="cpu")
    assert mps_utils.prefer_float32_on_cpu(model) is model


def test_mps_cap_leaves_room_for_resident_models_and_a_long_batch():
    for physical_gib in (8, 16, 36, 64):
        assert mps_utils._default_mps_budget_bytes(physical_gib * 1024 ** 3) == 4 * 1024 ** 3


def _embed_request(server, monkeypatch, rss, *, resident: str, loaded_rss_mib: int):
    builds = []

    def build(model_id, device):
        builds.append(device)
        rss.rss = loaded_rss_mib * MIB
        return Model(device)

    monkeypatch.delenv("TRUEMEMORY_DEVICE", raising=False)
    monkeypatch.setattr(mps_utils, "auto_detect_device", lambda: resident)
    monkeypatch.setattr(server, "_build_embed_model", build)
    with server._memory_scope():
        response = server.handle_request({"op": "embed", "texts": ["synthetic a", "synthetic b"], "tier": "base"})
    return response, builds


def test_loading_request_is_not_refused_by_its_own_construction_charge(budgeted, monkeypatch):
    server, rss = budgeted
    # 2,900 MiB resident after a float32 CPU load fits the 3,072 MiB budget
    # only once the 2.4 GB construction reservation is returned.
    response, builds = _embed_request(server, monkeypatch, rss, resident="cpu", loaded_rss_mib=2900)
    assert response["ok"] is True, response
    assert builds == [None]


def test_accelerator_loads_are_not_charged_against_rss(budgeted, monkeypatch):
    server, rss = budgeted
    # MPS weights never appear in RSS, so 1,200 MiB + 2.4 GB must not refuse an MPS load.
    rss.rss = 1200 * MIB
    response, builds = _embed_request(server, monkeypatch, rss, resident="mps", loaded_rss_mib=1300)
    assert response["ok"] is True, response


def test_clone_without_room_waits_for_the_main_model(budgeted, monkeypatch):
    server, rss = budgeted
    builds = []
    monkeypatch.setattr(server, "_build_embed_model", lambda model_id, device: builds.append(device) or Model(device))
    server._publish_embed_state(ms._EmbedState(model=Model("cpu"), tier="", model_id="qwen3_256"))
    # 1,200 MiB resident plus a 2.4 GB CPU clone exceeds the 3,072 MiB budget.
    rss.rss = 1200 * MIB
    held = threading.Event()

    def short_slice():
        with server._inference_lock:
            held.set()
            time.sleep(0.2)

    worker = threading.Thread(target=short_slice)
    worker.start()
    held.wait()
    try:
        with server._memory_scope():
            assert server._handle_fast_embed(["synthetic"], "") is None
    finally:
        worker.join()
    assert builds == [] and server._fast_encoder is None


def test_session_release_returns_one_stage(budgeted):
    from truememory.memory_budget import ProfileKey
    server, _ = budgeted
    with server._memory_scope() as session:
        lease = session.charge(ProfileKey("synthetic stage", "model_server"), known_bytes=1000)
        assert lease.reserved_known_bytes >= 1000
        session.release("synthetic stage")
        assert lease.reserved_known_bytes == 0



def test_short_deadline_bounds_the_bookkeeping_wait(plain, monkeypatch):
    plain._publish_embed_state(ms._EmbedState(model=Model("cpu"), tier="", model_id="model2vec"))
    monkeypatch.setattr(plain, "_build_embed_model", lambda model_id, device: Model(device))
    held, done = threading.Event(), threading.Event()

    def bookkeeping():
        with plain._lock:
            held.set()
            done.wait(5)

    worker = threading.Thread(target=bookkeeping)
    worker.start()
    held.wait()
    try:
        started = time.monotonic()
        plain._handle_fast_embed(["synthetic"], "", ms._RequestDeadline(time.monotonic() + 0.05))
        assert time.monotonic() - started < 0.25
    except ms._RequestDeadlineExceeded:
        assert time.monotonic() - started < 0.25
    finally:
        done.set()
        worker.join()


def test_loading_model_does_not_reserve_its_size_against_concurrent_work(budgeted, monkeypatch):
    server, rss = budgeted
    monkeypatch.setattr(mps_utils, "auto_detect_device", lambda: "cpu")
    seen = []

    def build(model_id, device):
        # Half-loaded: 1,700 MiB resident. Another request is admitted on RSS alone.
        rss.rss = 1700 * MIB

        def concurrent():
            with server._memory_scope():
                try:
                    server._check_process_memory("synthetic concurrent slice")
                    seen.append("admitted")
                except ms._ProcessMemoryRefused:
                    seen.append("refused")

        worker = threading.Thread(target=concurrent)
        worker.start()
        worker.join()
        rss.rss = 2900 * MIB
        return Model(device)

    monkeypatch.setattr(server, "_build_embed_model", build)
    with server._lock:
        server._get_embed_model("base")
    assert seen == ["admitted"]


def test_cpu_constructions_are_built_one_at_a_time(plain, monkeypatch):
    monkeypatch.setattr(mps_utils, "auto_detect_device", lambda: "cpu")
    active, peak = [0], [0]

    def build(selected):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        active[0] -= 1
        return Model(selected)

    workers = [threading.Thread(target=plain._construct_model,
                                args=("embed", "synthetic construction", "cpu", None, build),
                                kwargs={"known_bytes": 1}) for _ in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert peak[0] == 1


def test_dropped_models_are_freed_before_each_construction(plain, monkeypatch):
    order = []
    monkeypatch.setattr(plain, "_discard_models", lambda: order.append("discard"))
    plain._construct_model("embed", "synthetic construction", "cpu", None,
                           lambda selected: order.append("build") or Model(selected))
    assert order[:2] == ["discard", "build"]


def test_fallback_waits_at_most_a_second_for_a_long_batch(budgeted, monkeypatch):
    server, rss = budgeted
    monkeypatch.setattr(server, "_build_embed_model", lambda model_id, device: Model(device))
    server._publish_embed_state(ms._EmbedState(model=Model("cpu"), tier="", model_id="qwen3_256"))
    rss.rss = 1200 * MIB
    release = threading.Event()

    def long_batch():
        with server._inference_lock:
            release.wait(5)

    worker = threading.Thread(target=long_batch)
    worker.start()
    started = time.monotonic()
    try:
        with server._memory_scope(), pytest.raises(ms._ProcessMemoryRefused):
            server._handle_fast_embed(["synthetic"], "", ms._RequestDeadline(time.monotonic() + 5))
        assert 0.9 <= time.monotonic() - started < 2.0
    finally:
        release.set()
        worker.join()


@pytest.mark.parametrize("physical_gib, small", [(8, True), (16, False)])
def test_small_hosts_never_build_a_fast_lane_copy(monkeypatch, physical_gib, small):
    rss = FakeRSS(500 * MIB, physical=physical_gib * 1024 ** 3)
    monkeypatch.setattr(ms, "psutil", rss)
    monkeypatch.delenv("TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB", raising=False)
    server = ms.ModelServer(automatic_rss_budget=True)
    try:
        assert server._small_host is small
        builds = []
        monkeypatch.setattr(server, "_build_embed_model", lambda model_id, device: builds.append(device) or Model(device))
        server._publish_embed_state(ms._EmbedState(model=Model("mps"), tier="", model_id="qwen3_256"))
        with server._fast_lock, server._memory_scope():
            if small:
                with pytest.raises(ms._ProcessMemoryRefused):
                    server._get_fast_encoder("")
            else:
                server._get_fast_encoder("")
        assert builds == ([] if small else ["cpu"])
    finally:
        _close(server)


def test_test_servers_never_read_the_host_size(plain):
    assert plain._small_host is False


def test_concurrent_fallback_waits_do_not_queue_behind_each_other(budgeted, monkeypatch):
    server, rss = budgeted
    monkeypatch.setattr(server, "_build_embed_model", lambda model_id, device: Model(device))
    server._publish_embed_state(ms._EmbedState(model=Model("cpu"), tier="", model_id="qwen3_256"))
    rss.rss = 1200 * MIB
    release = threading.Event()

    def long_batch():
        with server._inference_lock:
            release.wait(5)

    holder = threading.Thread(target=long_batch)
    holder.start()
    finished = []

    def query():
        started = time.monotonic()
        with server._memory_scope():
            try:
                server._handle_fast_embed(["synthetic"], "", ms._RequestDeadline(time.monotonic() + 5))
            except ms._ProcessMemoryRefused:
                finished.append(time.monotonic() - started)

    waiters = [threading.Thread(target=query) for _ in range(3)]
    try:
        for waiter in waiters:
            waiter.start()
        for waiter in waiters:
            waiter.join()
    finally:
        release.set()
        holder.join()
    assert len(finished) == 3 and max(finished) < 1.6


@pytest.mark.parametrize("cached_mib, emptied", [(600, True), (100, False)])
def test_long_batches_return_cached_mps_heaps(monkeypatch, cached_mib, emptied):
    calls = []
    torch = types.SimpleNamespace(mps=types.SimpleNamespace(
        driver_allocated_memory=lambda: (1700 + cached_mib) * MIB,
        current_allocated_memory=lambda: 1700 * MIB,
        empty_cache=lambda: calls.append("empty"),
    ))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(mps_utils, "_mps_budget", object())
    mps_utils.trim_mps_cache()
    assert (calls == ["empty"]) is emptied
