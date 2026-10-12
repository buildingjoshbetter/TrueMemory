"""Event-driven local ownership and recovery checks with synthetic models."""
from __future__ import annotations

import threading
import sqlite3
import sys
import types
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import numpy as np

from truememory import mps_utils
from truememory.model_client import EmbeddingProxy


@pytest.fixture(autouse=True)
def no_allocator_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps_utils, "flush_mps_cache", lambda: None)
    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", torch)


@pytest.fixture
def ownership_wait(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    waiting = threading.Event()
    factory = mps_utils._ModelOwnership

    class ObservedLock:
        def __init__(self) -> None:
            self.lock = threading.Lock()

        def __enter__(self) -> None:
            if self.lock.locked():
                waiting.set()
            self.lock.acquire()

        def __exit__(self, *_exc: object) -> None:
            self.lock.release()

    monkeypatch.setattr(mps_utils, "_ModelOwnership", lambda: factory(lock=ObservedLock()))
    return waiting


def test_same_instance_cannot_enter_during_cpu_retry(ownership_wait: threading.Event) -> None:
    retry_started = threading.Event()
    release_retry = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    moves: list[str] = []
    active = 0

    class Model:
        device = "mps"

        def encode(self, texts: list[str], **kwargs: object) -> list[list[int]]:
            nonlocal active
            active += 1
            try:
                if self.device == "mps":
                    raise RuntimeError("MPS backend out of memory")
                if texts == ["first"]:
                    retry_started.set()
                    assert release_retry.wait(3)
                else:
                    second_entered.set()
                return [[len(texts[0]), 2]]
            finally:
                active -= 1

        def to(self, device: str) -> None:
            assert active == 0, "device movement overlapped native inference"
            moves.append(device)
            self.device = device

    model = Model()

    def second() -> object:
        second_attempted.set()
        return mps_utils.encode_with_mps_fallback(model, ["second"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(mps_utils.encode_with_mps_fallback, model, ["first"])
        try:
            assert retry_started.wait(2)
            other = pool.submit(second)
            assert second_attempted.wait(2)
            assert ownership_wait.wait(2)
            assert not second_entered.is_set()
        finally:
            release_retry.set()
        assert first.result(timeout=2) == [[5, 2]]
        assert other.result(timeout=2) == [[6, 2]]
    assert moves == ["cpu"]
    assert model.device == "cpu"
    assert id(model) not in mps_utils._model_owners


def test_unrelated_models_remain_concurrent() -> None:
    barrier = threading.Barrier(2)

    class Model:
        def encode(self, texts: list[str], **kwargs: object) -> list[str]:
            barrier.wait(timeout=2)
            return texts

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(mps_utils.encode_with_mps_fallback, Model(), [str(index)])
            for index in range(2)
        ]
        assert [future.result(timeout=3) for future in futures] == [["0"], ["1"]]


def test_concurrent_oom_calls_recover_once_without_repromotion() -> None:
    start = threading.Barrier(8)

    class Model:
        device = "mps"

        def __init__(self) -> None:
            self.moves: list[str] = []
            self.oom_count = 0

        def encode(self, texts: list[str], **kwargs: object) -> list[str]:
            if self.device == "mps":
                self.oom_count += 1
                raise RuntimeError("MPS backend out of memory")
            return texts

        def to(self, device: str) -> None:
            self.moves.append(device)
            self.device = device

    model = Model()

    def run(index: int) -> object:
        start.wait(timeout=2)
        return mps_utils.encode_with_mps_fallback(model, [str(index)])

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run, index) for index in range(8)]
        assert [future.result(timeout=3) for future in futures] == [[str(i)] for i in range(8)]
    assert model.oom_count == 1
    assert model.moves == ["cpu"]


@pytest.mark.parametrize("failure", ["normal", "move", "retry"])
def test_failures_release_instance_ownership(failure: str) -> None:
    class Model:
        failing = True
        calls = 0

        def encode(self, texts: list[str], **kwargs: object) -> list[str]:
            self.calls += 1
            if self.failing:
                if failure == "normal" or self.calls > 1:
                    raise RuntimeError("synthetic inference failure")
                raise RuntimeError("MPS backend out of memory")
            return texts

        def to(self, device: str) -> None:
            if failure == "move" and self.failing:
                raise RuntimeError("synthetic move failure")

    model = Model()
    with pytest.raises(RuntimeError, match="synthetic"):
        mps_utils.encode_with_mps_fallback(model, ["failure"])
    assert id(model) not in mps_utils._model_owners
    model.failing = False
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(mps_utils.encode_with_mps_fallback, model, ["reused"])
        assert future.result(timeout=2) == ["reused"]
    assert id(model) not in mps_utils._model_owners


def test_release_traceback_and_move_before_allocator_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Allocation:
        pass

    class Model:
        device = "mps"
        failed_allocation: weakref.ReferenceType[Allocation] | None = None

        def encode(self, texts: list[str], **kwargs: object) -> list[str]:
            if self.device == "mps":
                allocation = Allocation()
                self.failed_allocation = weakref.ref(allocation)
                raise RuntimeError("MPS backend out of memory")
            events.append("retry")
            return texts

        def to(self, device: str) -> None:
            assert self.failed_allocation is not None
            assert self.failed_allocation() is None
            events.append("move-" + device)
            self.device = device

    monkeypatch.setattr(mps_utils, "flush_mps_cache", lambda: events.append("flush"))
    model = Model()
    assert mps_utils.encode_with_mps_fallback(model, ["synthetic"]) == ["synthetic"]
    assert events == ["move-cpu", "flush", "retry"]


def test_unhashable_nonweakrefable_model_and_kwargs_are_supported() -> None:
    class Model:
        __slots__ = ()
        __hash__ = None

        def encode(self, texts: list[str], **kwargs: object) -> tuple:
            return texts, kwargs

    model = Model()
    assert mps_utils.encode_with_mps_fallback(
        model, ["a", "a"], batch_size=2, show_progress_bar=False,
    ) == (["a", "a"], {"batch_size": 2, "show_progress_bar": False})
    assert id(model) not in mps_utils._model_owners


def test_proxy_requests_keep_server_concurrency() -> None:
    barrier = threading.Barrier(2)

    class Proxy(EmbeddingProxy):
        def encode(self, texts: list[str], **kwargs: object) -> list[str]:
            barrier.wait(timeout=2)
            return texts

    proxy = Proxy()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(mps_utils.encode_with_mps_fallback, proxy, [str(index)])
            for index in range(2)
        ]
        assert [future.result(timeout=3) for future in futures] == [["0"], ["1"]]
    assert id(proxy) not in mps_utils._model_owners


def test_proxy_errors_do_not_trigger_local_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Proxy(EmbeddingProxy):
        def encode(self, texts: list[str], **kwargs: object) -> list[str]:
            calls.append("request")
            raise RuntimeError("Model server error: MPS backend out of memory")

    monkeypatch.setattr(mps_utils, "flush_mps_cache", lambda: calls.append("flush"))
    with pytest.raises(RuntimeError, match="Model server error"):
        mps_utils.encode_with_mps_fallback(Proxy(), ["synthetic"])
    assert calls == ["request"]


@pytest.mark.parametrize("caller", ["clustering", "gate", "worker-primary", "worker-separation"])
def test_real_callers_share_ownership_with_fallback(
    caller: str, monkeypatch: pytest.MonkeyPatch, ownership_wait: threading.Event,
) -> None:
    from truememory import clustering, vector_search
    from truememory.ingest.encoding_gate import EncodingGate
    from truememory.tier_switch.worker import RebuildWorker

    entered = threading.Event()
    release = threading.Event()
    guard = threading.Lock()
    blocked_call = 2 if caller == "worker-separation" else 1

    class Model:
        device = "mps"
        active = 0
        peak = 0
        calls = 0

        def __init__(self) -> None:
            self.moves: list[str] = []

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            with guard:
                self.calls += 1
                call = self.calls
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                if call == blocked_call:
                    entered.set()
                    assert release.wait(3)
                if texts == ["contender"] and self.device == "mps":
                    raise RuntimeError("MPS backend out of memory")
                return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float32)
            finally:
                with guard:
                    self.active -= 1

        def to(self, device: str) -> None:
            assert self.active == 0
            self.moves.append(device)
            self.device = device

    model = Model()
    monkeypatch.setattr(vector_search, "get_model", lambda: model)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        conn.execute("CREATE TABLE cluster_centroids (id INTEGER)")
        conn.execute("CREATE TABLE vec_fixture (rowid INTEGER, embedding BLOB)")
        conn.execute("CREATE TABLE sep_fixture (rowid INTEGER, embedding BLOB)")
        gate = types.SimpleNamespace(
            _embed_model=model, _last_search_results=[{"content": "synthetic nearby fact"}],
            _pe_available=True, _pe_degradation_count=0,
        )
        worker = types.SimpleNamespace(conn=conn)

        def existing_call() -> object:
            if caller == "clustering":
                return clustering.search_clustered(conn, "synthetic query")
            if caller == "gate":
                return EncodingGate._compute_prediction_error(gate, "synthetic fact")
            return RebuildWorker._process_batch(
                worker, [{"id": 1, "content": "synthetic record"}], model,
                "vec_fixture", "sep_fixture", lambda vector: vector.tobytes(),
                lambda *_parts: "synthetic separation",
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(existing_call)
            try:
                assert entered.wait(2)
                other = pool.submit(mps_utils.encode_with_mps_fallback, model, ["contender"])
                assert ownership_wait.wait(2)
                assert model.moves == []
            finally:
                release.set()
            first.result(timeout=2)
            np.testing.assert_array_equal(other.result(timeout=2), [[9, 1]])
        assert model.peak == 1
        assert model.moves == ["cpu"]
        assert gate._pe_available
        if caller.startswith("worker"):
            for table, text in (("vec_fixture", "synthetic record"),
                                ("sep_fixture", "synthetic separation")):
                rows = conn.execute(f"SELECT rowid, embedding FROM {table}").fetchall()
                assert len(rows) == 1
                assert rows[0][0] == 1
                np.testing.assert_array_equal(np.frombuffer(rows[0][1], dtype=np.float32), [len(text), 1])
        assert id(model) not in mps_utils._model_owners
    finally:
        conn.close()


def test_ownership_only_entry_preserves_outer_oom_policy() -> None:
    class Model:
        def encode(self, texts: list[str], **kwargs: object) -> object:
            raise RuntimeError("MPS backend out of memory")

        def to(self, device: str) -> None:
            raise AssertionError("ownership-only caller must retain its retry policy")

    model = Model()
    with pytest.raises(RuntimeError, match="MPS backend out of memory"):
        mps_utils.encode_with_model_ownership(model, ["synthetic"])
    assert id(model) not in mps_utils._model_owners
