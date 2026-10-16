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

def test_embedding_returns_to_mps_after_a_quiet_backoff(server, monkeypatch):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    since = instance._embed_cpu_since
    assert instance._embed_repromote_after() == 600
    instance._last_activity = time.time() - 3600
    assert instance._repromote_embed(now=since + 599) is False
    cpu_main = weakref.ref(instance._embed_state.model)
    assert instance._repromote_embed(now=since + 600) is True
    assert "embed" not in instance._sticky_cpu and instance._embed_state is None
    assert cpu_main() is None, "The float32 CPU copy must be released"
    assert _embed(instance, ["after"])["ok"]
    assert builds.devices[-1] == "mps" and instance._embed_state.model.device == "mps"


def test_repeat_ooms_double_the_backoff(server):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    instance._last_activity = time.time() - 3600
    assert instance._repromote_embed(now=instance._embed_cpu_since + 600)
    assert _embed(instance, ["reload"])["ok"]
    builds.oom = True
    assert _embed(instance, ["a", "b"])["ok"]
    builds.oom = False
    assert instance._embed_ooms == 2 and instance._embed_repromote_after() == 1200
    instance._last_activity = time.time() - 3600
    assert instance._repromote_embed(now=instance._embed_cpu_since + 1199) is False
    assert instance._repromote_embed(now=instance._embed_cpu_since + 1200) is True
    instance._embed_ooms = 40
    assert instance._embed_repromote_after() == 600 * instance._EMBED_REPROMOTE_MAX_FACTOR


def test_recent_work_or_a_running_batch_defers_the_return(server):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    later = instance._embed_cpu_since + 10_000
    instance._last_activity = time.time()
    assert instance._repromote_embed(now=later) is False
    instance._last_activity = time.time() - 3600
    batch = Batch(instance)
    try:
        assert instance._repromote_embed(now=later) is False
    finally:
        batch.end()
    assert instance._repromote_embed(now=later) is True


def test_explicit_cpu_device_is_never_repromoted(server, monkeypatch):
    instance, builds = server
    _degrade_with_clone(instance, builds)
    monkeypatch.setenv("TRUEMEMORY_DEVICE", "cpu")
    instance._last_activity = time.time() - 3600
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
