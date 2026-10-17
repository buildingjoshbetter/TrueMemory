"""After an MPS OOM the server stays lean, recovers MPS, and stops promptly."""
from __future__ import annotations

import os
import threading
import time
import types
import weakref

import numpy as np
import pytest

from truememory import model_server as ms
from truememory import mps_utils

MIB = 1024 ** 2


class FakeRSS:
    def __init__(self, rss: int) -> None:
        self.rss = rss

    def virtual_memory(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(total=32 * 1024 ** 3)

    def Process(self) -> FakeRSS:
        return self

    def memory_info(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(rss=self.rss)


class Model:
    """A reference cycle like sentence-transformers models; MPS runs can OOM."""

    def __init__(self, device: str | None, owner: "Builds") -> None:
        self.device = device
        self.owner = owner
        self.cycle = self

    def encode(self, texts, batch_size=32, show_progress_bar=False, **kwargs):
        self.owner.encodes.append(self.device)
        if self.device == "mps" and self.owner.oom:
            raise RuntimeError("MPS backend out of memory (MPS allocated: 3.9 GiB, max allowed: 4.0 GiB)")
        if self.owner.hold is not None:
            self.owner.holding.set()
            self.owner.hold.wait(10)
        return np.ones((len(texts), 2), dtype=np.float32)

    def to(self, device: str) -> "Model":
        self.device = device
        return self


class Builds:
    def __init__(self) -> None:
        self.devices: list[str | None] = []
        self.refs: list[weakref.ref] = []
        self.encodes: list[str | None] = []
        self.oom = False
        self.hold: threading.Event | None = None
        self.holding = threading.Event()

    def __call__(self, model_id: str, device: str | None) -> Model:
        self.devices.append(device)
        model = Model(device, self)
        self.refs.append(weakref.ref(model))
        return model


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("TRUEMEMORY_DEVICE", raising=False)
    monkeypatch.setattr(mps_utils, "auto_detect_device", lambda: "mps")
    monkeypatch.setattr(mps_utils, "ensure_mps_memory_budget", lambda device: None)
    instance = ms.ModelServer()
    instance._write_status_file = lambda: None
    instance._embed_device = lambda: "cpu" if "embed" in instance._sticky_cpu else "mps"
    builds = Builds()
    monkeypatch.setattr(instance, "_build_embed_model", builds)
    yield instance, builds
    instance._stopped.set()
    rebuild = instance._fast_rebuild
    if rebuild is not None:
        rebuild.join(5)
    instance._workers.shutdown(wait=False)


def _embed(server: ms.ModelServer, texts: list[str]) -> dict:
    return server.handle_request({"op": "embed", "texts": texts, "tier": "qwen3_256"})


def _contended(server: ms.ModelServer, deadline: float = 5) -> dict | None:
    return server._handle_fast_embed(["synthetic query"], "qwen3_256", ms._RequestDeadline(time.monotonic() + deadline))


class Batch:
    def __init__(self, server: ms.ModelServer, seconds: float = 10) -> None:
        self.release = threading.Event()
        held = threading.Event()

        def run() -> None:
            with server._inference_lock:
                held.set()
                self.release.wait(seconds)

        self.thread = threading.Thread(target=run)
        self.thread.start()
        assert held.wait(5)

    def end(self) -> None:
        self.release.set()
        self.thread.join(5)


def _degrade_with_clone(server: ms.ModelServer, builds: Builds) -> weakref.ref:
    """Main model on MPS with a cached CPU clone, then an embed MPS OOM."""
    assert _embed(server, ["load"])["ok"]
    batch = Batch(server)
    try:
        assert _contended(server)["ok"]
    finally:
        batch.end()
    assert builds.devices == ["mps", "cpu"]
    clone = builds.refs[1]
    assert server._fast_encoder is clone()
    builds.oom = True
    assert _embed(server, ["a", "b"])["ok"]
    builds.oom = False
    assert "embed" in server._sticky_cpu and server._embed_state.model.device == "cpu"
    return clone


# --- 1. a clone of a CPU main model is redundant -------------------------------------

def test_oom_to_cpu_releases_the_fast_lane_clone_at_once(server):
    instance, builds = server
    clone = _degrade_with_clone(instance, builds)
    assert instance._fast_encoder is None
    assert clone() is None, "Two CPU copies of the same weights stay resident"


def test_single_queries_use_the_cpu_main_model_without_a_new_clone(server):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    served = _contended(instance)  # no batch: the main CPU model serves it directly
    assert served["ok"] and builds.encodes[-1] == "cpu" and builds.devices == ["mps", "cpu"]
    batch = Batch(instance)
    started = time.monotonic()
    try:
        with pytest.raises(ms._ProcessMemoryRefused, match="long batch"):
            _contended(instance)
    finally:
        batch.end()
    assert time.monotonic() - started < 2.0
    response = instance.handle_request({"op": "embed", "texts": ["q"], "tier": "qwen3_256"})
    assert response["ok"] and builds.devices == ["mps", "cpu"]
    assert builds.encodes[-1] == "cpu"


def test_a_clone_in_use_during_recovery_is_released_when_its_query_ends(server):
    instance, builds = server
    assert _embed(instance, ["load"])["ok"]
    batch = Batch(instance)
    try:
        assert _contended(instance)["ok"]
    finally:
        batch.end()
    clone = builds.refs[1]
    builds.hold = threading.Event()
    results = []
    batch = Batch(instance)
    query = threading.Thread(target=lambda: results.append(_contended(instance)))
    query.start()
    try:
        assert builds.holding.wait(5)  # the query is encoding on the clone
        batch.end()
        main = instance._embed_state.model
        with instance._lock:
            instance._recover_embed_oom_locked(main)
        main = None
        assert instance._fast_encoder is clone(), "An active clone was taken from its query"
    finally:
        builds.hold.set()
        query.join(10)
        batch.end()
    assert results and results[0]["ok"]
    assert instance._fast_encoder is None and clone() is None


# --- 2. embedding sticky CPU recovers -------------------------------------------------

def test_embedding_returns_to_mps_after_its_backoff(server):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    since = instance._embed_cpu_since
    assert instance._embed_repromote_after() == 600
    assert instance._repromote_embed(now=since + 599) is False
    cpu_main = weakref.ref(instance._embed_state.model)
    assert instance._repromote_embed(now=since + 600) is True
    assert "embed" not in instance._sticky_cpu
    assert instance._embed_state.model.device == "mps" and builds.devices == ["mps", "cpu", "mps"]
    assert cpu_main() is None, "The float32 CPU copy must be released"
    assert instance._fast_encoder is None
    assert _embed(instance, ["after"])["ok"]
    assert builds.devices == ["mps", "cpu", "mps"], "The swap already loaded the MPS copy"
    assert builds.encodes[-1] == "mps"


def test_repeat_ooms_double_the_backoff(server):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    assert instance._repromote_embed(now=instance._embed_cpu_since + 600)
    builds.oom = True
    assert _embed(instance, ["a", "b"])["ok"]
    builds.oom = False
    assert instance._embed_ooms == 2 and instance._embed_repromote_after() == 1200
    assert instance._repromote_embed(now=instance._embed_cpu_since + 1199) is False
    assert instance._repromote_embed(now=instance._embed_cpu_since + 1200) is True
    instance._embed_ooms = 40
    assert instance._embed_repromote_after() == 600 * instance._EMBED_REPROMOTE_MAX_FACTOR


def test_steady_hook_traffic_does_not_hold_the_return_back(server, monkeypatch):
    """Hook queries arrive every few seconds in a working session; here every 50 ms."""
    instance, builds = server
    _degrade_with_clone(instance, builds)
    monkeypatch.setattr(instance, "_PARKED_RERANKER_IDLE_SECONDS", 4)  # a retirer tick every second
    instance._embed_cpu_since -= instance._embed_repromote_after()
    stop = threading.Event()
    served = []

    def hooks() -> None:
        while not stop.is_set():
            served.append(_embed(instance, ["synthetic hook query"])["ok"])
            time.sleep(0.05)

    traffic = threading.Thread(target=hooks)
    retirer = threading.Thread(target=instance._idle_retirer)
    traffic.start()
    retirer.start()
    try:
        ends = time.monotonic() + 5
        while "embed" in instance._sticky_cpu and time.monotonic() < ends:
            time.sleep(0.05)
    finally:
        stop.set()
        traffic.join(5)
        instance._stopped.set()
        retirer.join(5)
    assert "embed" not in instance._sticky_cpu, "Steady traffic kept embedding on CPU"
    assert instance._embed_state.model.device == "mps"
    assert len(served) > 5 and all(served)


def test_the_next_batch_swaps_before_its_first_piece(server):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    instance._embed_cpu_since -= instance._embed_repromote_after()
    encodes = len(builds.encodes)
    assert _embed(instance, ["a", "b"])["ok"]
    assert builds.encodes[encodes:] == ["mps"] and "embed" not in instance._sticky_cpu


def test_a_single_query_never_pays_the_swap(server):
    """A contended query that falls through to the main model still skips the MPS load."""
    instance, builds = server
    _degrade_with_clone(instance, builds)
    instance._embed_cpu_since -= instance._embed_repromote_after()
    batch = Batch(instance)
    results = []
    query = threading.Thread(target=lambda: results.append(_embed(instance, ["synthetic hook query"])))
    query.start()
    time.sleep(0.2)  # the query is waiting briefly for the main model
    batch.end()
    query.join(5)
    assert results and results[0]["ok"] and builds.encodes[-1] == "cpu"
    assert "embed" in instance._sticky_cpu and builds.devices == ["mps", "cpu"]
    assert _embed(instance, ["a", "b"])["ok"] and builds.encodes[-1] == "mps"


def test_a_running_batch_defers_the_retirer_to_its_next_request(server):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    later = instance._embed_cpu_since + 10_000
    batch = Batch(instance)
    try:
        assert instance._repromote_embed(now=later) is False
        assert "embed" in instance._sticky_cpu
    finally:
        batch.end()
    assert instance._repromote_embed(now=later) is True


def test_single_queries_are_served_while_the_mps_copy_loads(server, monkeypatch):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    cpu_main = weakref.ref(instance._embed_state.model)
    loading, finish = threading.Event(), threading.Event()

    def slow_build(model_id: str, device: str | None) -> Model:
        if device == "mps":
            loading.set()
            assert finish.wait(10)
        return builds(model_id, device)

    monkeypatch.setattr(instance, "_build_embed_model", slow_build)
    results = []
    swap = threading.Thread(target=lambda: results.append(
        instance._repromote_embed(now=instance._embed_cpu_since + 600)))
    swap.start()
    try:
        assert loading.wait(5)
        started = time.monotonic()
        responses = [_contended(instance, deadline=2) for _ in range(3)]
        assert all(response["ok"] for response in responses), "Single queries were busy during the swap"
        assert time.monotonic() - started < 2.0
        assert builds.encodes[-3:] == ["cpu"] * 3 and builds.devices == ["mps", "cpu"], "A clone was built"
    finally:
        finish.set()
        swap.join(10)
    assert results == [True]
    assert instance._fast_encoder is None and cpu_main() is None
    assert instance._embed_state.model.device == "mps"


def test_without_rss_room_the_swap_waits_for_a_later_request(server, monkeypatch):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    rss = FakeRSS(3500 * MIB)
    monkeypatch.setattr(ms, "psutil", rss)
    instance._max_rss_bytes = 4096 * MIB  # 596 MiB of room; the MPS load stages 1.2 GB on CPU
    later = instance._embed_cpu_since + 600
    assert instance._repromote_embed(now=later) is False
    assert "embed" in instance._sticky_cpu and builds.devices == ["mps", "cpu"]
    assert instance._fast_encoder is None
    rss.rss = 2048 * MIB
    assert instance._repromote_embed(now=later) is True
    assert instance._embed_state.model.device == "mps"


def test_a_failed_mps_load_keeps_cpu_and_restarts_the_backoff(server, monkeypatch):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    cpu_main = instance._embed_state.model

    def oom_build(model_id: str, device: str | None) -> Model:
        if device == "mps":
            raise RuntimeError("MPS backend out of memory (MPS allocated: 3.9 GiB, max allowed: 4.0 GiB)")
        return builds(model_id, device)

    monkeypatch.setattr(instance, "_build_embed_model", oom_build)
    before = time.monotonic()
    assert instance._repromote_embed(now=instance._embed_cpu_since + 600) is False
    assert instance._embed_state.model is cpu_main and cpu_main.device == "cpu"
    assert "embed" in instance._sticky_cpu and instance._embed_ooms == 2
    assert instance._embed_cpu_since >= before and instance._fast_encoder is None
    assert builds.devices == ["mps", "cpu"], "A failed MPS load must not build a second CPU copy"
    # No room on MPS at all: the same, without counting an OOM.
    monkeypatch.setattr(instance, "_build_embed_model", builds)
    monkeypatch.setattr(mps_utils, "mps_free_bytes", lambda: 100 * MIB, raising=False)
    assert instance._repromote_embed(now=instance._embed_cpu_since + 1200) is False
    assert instance._embed_state.model is cpu_main and instance._embed_ooms == 2
    assert builds.devices == ["mps", "cpu"] and instance._fast_encoder is None
    cpu_main = None
    batch = Batch(instance)
    try:
        with pytest.raises(ms._ProcessMemoryRefused, match="long batch"):
            _contended(instance, deadline=0.5)  # the lent model went back with the failed swap
    finally:
        batch.end()


def test_a_load_that_finishes_after_its_deadline_is_kept(server, monkeypatch):
    instance, builds = server

    def slow_build(model_id: str, device: str | None) -> Model:
        time.sleep(0.3)
        return builds(model_id, device)

    monkeypatch.setattr(instance, "_build_embed_model", slow_build)
    with pytest.raises(ms._RequestDeadlineExceeded):
        instance._handle_fast_embed(["cold start"], "qwen3_256", ms._RequestDeadline(time.monotonic() + 0.1))
    assert instance._embed_state is not None and builds.devices == ["mps"]
    assert _embed(instance, ["next hook"])["ok"]
    assert builds.devices == ["mps"], "The finished load was discarded with its request"


def test_explicit_cpu_device_is_never_repromoted(server, monkeypatch):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    monkeypatch.setenv("TRUEMEMORY_DEVICE", "cpu")
    assert instance._repromote_embed(now=instance._embed_cpu_since + 10 ** 6) is False
    assert "embed" in instance._sticky_cpu


# --- 4. TERM stops promptly --------------------------------------------------------------

def test_stop_signal_exits_within_the_grace_even_while_native_work_runs(server, monkeypatch):
    instance, builds = server
    exits = []

    def fake_exit(code):
        exits.append((code, time.monotonic()))
        raise SystemExit(code)

    monkeypatch.setattr(ms.os, "_exit", fake_exit)
    with instance._activity_lock:
        instance._inflight += 1  # a native forward pass that never reaches a checkpoint
    try:
        started = time.monotonic()
        with pytest.raises(SystemExit):
            instance._finish_after_signal(grace=0.5)
        assert exits and exits[0][0] == 0 and 0.45 <= exits[0][1] - started < 2.0
        assert instance._stopped.is_set()
        assert not getattr(instance, "_resources_released", False)
    finally:
        with instance._activity_lock:
            instance._inflight -= 1


def test_stop_signal_releases_models_once_work_drains(server, monkeypatch):
    instance, builds = server
    monkeypatch.setattr(ms.os, "_exit", lambda code: pytest.fail("exited although work drained"))
    assert _embed(instance, ["load"])["ok"]
    main = weakref.ref(instance._embed_state.model)
    with instance._activity_lock:
        instance._inflight += 1
    threading.Timer(0.2, lambda: setattr(instance, "_inflight", instance._inflight - 1)).start()
    started = time.monotonic()
    instance._finish_after_signal(grace=5)
    assert time.monotonic() - started < 2.0
    assert instance._embed_state is None and main() is None
    assert instance._stopped.is_set()


def test_main_routes_a_stop_signal_through_the_bounded_exit(monkeypatch):
    calls = []

    class Server:
        def __init__(self, **kwargs):
            pass

        def run(self):
            raise SystemExit(0)

        def _finish_after_signal(self):
            calls.append("finish")

        def _cleanup(self):
            calls.append("cleanup")

    monkeypatch.setattr(ms, "ModelServer", Server)
    monkeypatch.setattr(ms.signal, "signal", lambda *args: None)
    monkeypatch.setattr(ms.atexit, "register", lambda *args: None)
    monkeypatch.delenv("TRUEMEMORY_REQUIRE_PROCESS_MEMORY_CAP", raising=False)
    with pytest.raises(SystemExit):
        ms.main()
    assert calls == ["finish"]


# --- 5. a refused reload is not retried on every query --------------------------------------

def test_a_refused_reload_backs_off_instead_of_retrying_per_query(monkeypatch):
    rss = FakeRSS(500 * MIB)
    monkeypatch.setattr(ms, "psutil", rss)
    monkeypatch.setenv("TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB", "3072")
    monkeypatch.setattr(mps_utils, "auto_detect_device", lambda: "mps")
    instance = ms.ModelServer(automatic_rss_budget=True)
    instance._write_status_file = lambda: None
    builds = Builds()
    monkeypatch.setattr(instance, "_build_embed_model", builds)
    collections = []
    monkeypatch.setattr(instance, "_collect_models", lambda: collections.append(1))
    try:
        instance._publish_embed_state(ms._EmbedState(model=Model("mps", builds), tier="qwen3_256",
                                                     model_id="qwen3_256"))
        with instance._memory_scope():
            batch = Batch(instance)
            try:
                assert _contended(instance)["ok"]
            finally:
                batch.end()
        assert instance._retire_idle_fast_encoder(now=instance._fast_last_used + 600)
        rss.rss = 1200 * MIB  # 1,200 MiB + a 2.4 GB clone exceeds 3,072 MiB
        attempts = []
        real_start = instance._rebuild_fast_encoder
        monkeypatch.setattr(instance, "_rebuild_fast_encoder", lambda *a: attempts.append(1) or real_start(*a))
        collections.clear()
        for _ in range(4):
            with instance._memory_scope():
                batch = Batch(instance)
                try:
                    with pytest.raises(ms._ProcessMemoryRefused):
                        _contended(instance, deadline=0.2)
                finally:
                    batch.end()
            if instance._fast_rebuild is not None:
                instance._fast_rebuild.join(5)
        assert attempts == [1], "Every contended query retried the refused reload"
        assert sum(collections) <= 1, "A refused admission must not run extra collections"
        # Resident memory fell by more than the clone's size: retry at once.
        rss.rss = 100 * MIB
        instance._fast_rebuild_refusal = (instance._fast_rebuild_refusal[0], 3000 * MIB)
        with instance._memory_scope():
            batch = Batch(instance)
            try:
                _contended(instance)
            finally:
                batch.end()
        if instance._fast_rebuild is not None:
            instance._fast_rebuild.join(5)
        assert attempts == [1, 1]
    finally:
        instance._stopped.set()
        instance._workers.shutdown(wait=False)
        if instance._memory_owner is not None:
            instance._memory_owner.close()


def test_status_never_claims_lifetime_degradation(server, caplog):
    instance, builds = server
    with caplog.at_level("ERROR"), instance._lock:
        instance._mark_sticky_cpu("embed")
    assert "lifetime" not in caplog.text and "retried" in caplog.text
    assert os.environ.get("TRUEMEMORY_DEVICE") is None


def test_on_a_small_budget_the_clone_is_released_before_the_cpu_transfer(monkeypatch):
    """16 GiB Mac shape: 4 GiB RSS budget, a 2.4 GB clone and a 2.4 GB CPU transfer."""
    live: dict[int, weakref.ref] = {}

    class Resident:
        def __init__(self) -> None:
            self.rss = 700 * MIB

        def virtual_memory(self) -> types.SimpleNamespace:
            return types.SimpleNamespace(total=16 * 1024 ** 3)

        def Process(self) -> "Resident":
            return self

        def memory_info(self) -> types.SimpleNamespace:
            cpu_copies = sum(1 for ref in live.values() if ref() is not None and ref().device == "cpu")
            return types.SimpleNamespace(rss=self.rss + cpu_copies * 2_400_000_000)

    monkeypatch.setattr(ms, "psutil", Resident())
    monkeypatch.setenv("TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB", "4096")
    monkeypatch.delenv("TRUEMEMORY_DEVICE", raising=False)
    monkeypatch.setattr(mps_utils, "auto_detect_device", lambda: "mps")
    monkeypatch.setattr(mps_utils, "ensure_mps_memory_budget", lambda device: None)
    instance = ms.ModelServer(automatic_rss_budget=True)
    instance._write_status_file = lambda: None
    instance._embed_device = lambda: "cpu" if "embed" in instance._sticky_cpu else "mps"
    builds = Builds()

    def build(model_id, device):
        model = builds(model_id, device)
        live[id(model)] = weakref.ref(model)
        return model

    monkeypatch.setattr(instance, "_build_embed_model", build)
    try:
        with instance._memory_scope():
            assert _embed(instance, ["load"])["ok"]
            batch = Batch(instance)
            try:
                assert _contended(instance)["ok"]
            finally:
                batch.end()
        clone = builds.refs[1]
        assert clone() is not None
        builds.oom = True
        response = _embed(instance, ["a", "b"])
        builds.oom = False
        assert response["ok"] is True, response
        assert clone() is None and instance._embed_state.model.device == "cpu"
    finally:
        instance._stopped.set()
        instance._workers.shutdown(wait=False)
        if instance._memory_owner is not None:
            instance._memory_owner.close()
