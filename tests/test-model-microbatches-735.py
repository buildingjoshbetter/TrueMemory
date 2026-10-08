"""Synthetic batch-control regressions, including real framed socket transport."""
from __future__ import annotations

import socket
import threading
import types
import unittest
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from truememory import model_client as mc
from truememory import model_server as ms


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now


def vectors(texts: list[str]) -> np.ndarray:
    return np.asarray([[len(text), sum(map(ord, text)), 0.5] for text in texts],
                      dtype=np.float64).reshape(len(texts), 3)


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list, int]] = []
        self.fail_calls: set[int] = set()
        self.after_call: Callable[[], None] = lambda: None
        self.moves: list[str] = []

    def record(self, kind: str, items: list, batch_size: int) -> None:
        self.calls.append((kind, list(items), batch_size))
        self.after_call()
        if len(self.calls) in self.fail_calls:
            raise RuntimeError("MPS backend out of memory")

    def encode(self, texts: list[str], *, batch_size: int = 32,
               show_progress_bar: bool) -> np.ndarray:
        assert show_progress_bar is False
        self.record("embed", texts, batch_size)
        return vectors(texts)

    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int,
                show_progress_bar: bool) -> np.ndarray:
        assert show_progress_bar is False
        self.record("rerank", pairs, batch_size)
        return np.asarray([len(pair[1]) / 10 for pair in pairs], dtype=np.float64)

    def to(self, device: str) -> None:
        self.moves.append(device)


class RecordingThrottler:
    def __init__(self, limits: list[int]) -> None:
        self.limits = limits
        self.before_count = 0
        self.after_counts: list[int] = []

    def before_batch(self) -> tuple[int, dict]:
        limit = self.limits[min(self.before_count, len(self.limits) - 1)]
        self.before_count += 1
        return limit, {}

    def after_batch(self, count: int, _seconds: float) -> None:
        self.after_counts.append(count)

    def should_flush_cache(self) -> bool:
        return False


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[ms.ModelServer, RecordingModel]:
    from truememory import mps_utils

    server = ms.ModelServer()
    model = RecordingModel()
    monkeypatch.setattr(server, "_get_embed_model", lambda _tier: model)
    monkeypatch.setattr(server, "_get_fast_encoder", lambda _tier: model)
    monkeypatch.setattr(server, "_get_reranker", lambda _name: model)
    monkeypatch.setattr(server, "_write_status_file", lambda: None)
    monkeypatch.setattr(server, "_flush_mps_cache", lambda: None)
    monkeypatch.setattr(mps_utils, "flush_mps_cache", lambda: None)
    return server, model


def payload(op: str, count: int, **kwargs: object) -> dict:
    texts = [f"synthetic-{i}" for i in range(count)]
    return {"op": op, "texts": texts,
            "pairs": [("synthetic-query", text) for text in texts], **kwargs}


def activate(server: ms.ModelServer, limits: list[int]) -> RecordingThrottler:
    throttler = RecordingThrottler(limits)
    server._throttler = throttler
    server._throttler_active = True
    return throttler


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("op", ["embed", "rerank"])
def test_sustained_unsupported_adaptive_policy_keeps_requested_bound(
    runtime: tuple, monkeypatch: pytest.MonkeyPatch, platform: str, device: str, op: str,
) -> None:
    from truememory.tier_switch import throttler as tm

    server, model = runtime
    clock = FakeClock()
    sleeps: list[float] = []
    clock.sleep = sleeps.append
    monkeypatch.setattr(tm, "time", clock)
    monkeypatch.setattr(tm, "sys", types.SimpleNamespace(platform=platform))
    throttler = tm.DynamicThrottler(device)
    monkeypatch.setattr(throttler, "_budget_snapshot", lambda: None)
    monkeypatch.setattr(throttler, "_read_all_channels", lambda: {
        name: {"status": "unsupported", "required": False}
        for name in ("mps_level", "growth_rate", "thermal")
    })
    server._throttler, server._throttler_active = throttler, True
    server._embed_timestamps = [ms.time.time()] * server._SUSTAINED_THRESHOLD
    for _ in range(300):
        response = server.handle_request(payload(op, 17, batch_size=8))
        assert response["ok"]
    assert [len(items) for _, items, _ in model.calls] == [8, 8, 1] * 300
    assert all(limit == 8 for _, _, limit in model.calls)
    assert sleeps == []
    assert throttler.items_processed == 0
    assert throttler.state_machine.good_streak == 0


@pytest.mark.parametrize("op,maximum", [("embed", 32), ("rerank", 64)])
def test_unsupported_policy_retains_server_cap_and_rechecks_applicability(
    runtime: tuple, op: str, maximum: int,
) -> None:
    server, model = runtime
    throttler = activate(server, [2])
    server._embed_timestamps = [ms.time.time()] * server._SUSTAINED_THRESHOLD
    throttler.adaptive_applicable = False
    assert server.handle_request(payload(op, maximum + 3, batch_size=maximum * 10))["ok"]
    assert [len(items) for _, items, _ in model.calls] == [maximum, 3]
    assert throttler.before_count == 0
    assert throttler.after_counts == []
    model.calls.clear()
    throttler.adaptive_applicable = True
    assert server.handle_request(payload(op, 5, batch_size=8))["ok"]
    assert [len(items) for _, items, _ in model.calls] == [2, 2, 1]
    assert throttler.before_count == 1
    assert throttler.after_counts == [5]


@pytest.mark.parametrize("op,key", [("embed", "vectors"), ("rerank", "scores")])
@pytest.mark.parametrize("requested,limits,expected", [
    (1, [8], [1, 1, 1, 1, 1]),
    (8, [2], [2, 2, 1]),
    (8, [2, 1, 2], [2, 2, 1]),
])
def test_each_actual_call_obeys_caller_and_captured_throttler_limit(
    runtime: tuple, op: str, key: str, requested: int,
    limits: list[int], expected: list[int],
) -> None:
    server, model = runtime
    throttler = activate(server, limits)
    request = payload(op, 5, batch_size=requested)
    response = server.handle_request(request)
    assert response["ok"]
    assert [len(items) for _, items, _ in model.calls] == expected
    assert all(len(items) <= limit <= requested for _, items, limit in model.calls)
    assert throttler.before_count == 1
    assert throttler.after_counts == [5]
    expected_values = vectors(request["texts"]) if op == "embed" else [
        len(pair[1]) / 10 for pair in request["pairs"]
    ]
    np.testing.assert_array_equal(response[key], np.asarray(expected_values, dtype=np.float32))
    assert response[key].dtype == np.float32


@pytest.mark.parametrize("op,key,maximum", [
    ("embed", "vectors", 32),
    ("rerank", "scores", 64),
])
@pytest.mark.parametrize("explicit", [False, True])
def test_server_cap_also_bounds_legacy_and_oversized_caller_requests(
    runtime: tuple, op: str, key: str, maximum: int, explicit: bool,
) -> None:
    server, model = runtime
    request = payload(op, maximum + 3)
    if explicit:
        request["batch_size"] = maximum * 10
    response = server.handle_request(request)
    assert response["ok"]
    assert [len(items) for _, items, _ in model.calls] == [maximum, 3]
    assert response[key].shape[0] == maximum + 3


@pytest.mark.parametrize("op,key", [("embed", "vectors"), ("rerank", "scores")])
def test_duplicates_mixed_lengths_and_multidimensional_scores_keep_order(
    runtime: tuple, monkeypatch: pytest.MonkeyPatch, op: str, key: str,
) -> None:
    server, model = runtime
    texts = ["fixture", "x" * 1000, "fixture", "", "longer fixture"]
    request = {"op": op, "texts": texts,
               "pairs": [("query", text) for text in texts], "batch_size": 2}
    if op == "rerank":
        def predict(pairs: list, **kwargs: object) -> np.ndarray:
            model.record("rerank", pairs, kwargs["batch_size"])
            return vectors([pair[1] for pair in pairs])

        monkeypatch.setattr(model, "predict", predict)
    response = server.handle_request(request)
    assert response["ok"]
    assert response[key].shape == (5, 3)
    np.testing.assert_array_equal(response[key], vectors(texts).astype(np.float32))


@pytest.mark.parametrize("op,key", [("embed", "vectors"), ("rerank", "scores")])
@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_single_inputs_preserve_model_output_shape(
    runtime: tuple, op: str, key: str, count: int,
) -> None:
    server, model = runtime
    response = server.handle_request(payload(op, count, batch_size=1))
    assert response["ok"]
    assert response[key].shape == ((count, 3) if op == "embed" else (count,))
    assert response[key].dtype == np.float32
    assert len(model.calls) == 1
    assert model.calls[0][2] == 1


def test_contended_single_text_fast_lane_keeps_limit(
    runtime: tuple,
) -> None:
    server, model = runtime
    with server._lock:
        response = server.handle_request(payload("embed", 1, batch_size=1))
    assert response["ok"]
    assert len(model.calls) == 1
    assert model.calls[0][2] == 1


@pytest.mark.parametrize("op,key", [("embed", "vectors"), ("rerank", "scores")])
def test_only_failed_microbatch_retries_and_completed_work_is_retained(
    runtime: tuple, op: str, key: str,
) -> None:
    server, model = runtime
    model.fail_calls.add(2)
    request = payload(op, 5, batch_size=2)
    response = server.handle_request(request)
    assert response["ok"]
    source = request["texts" if op == "embed" else "pairs"]
    assert [items for _, items, _ in model.calls] == [
        source[:2], source[2:4], source[2:4], source[4:],
    ]
    assert [limit for _, _, limit in model.calls] == [2, 2, 2, 2]
    assert server._sticky_cpu == {op}
    expected = vectors(source) if op == "embed" else [len(pair[1]) / 10 for pair in source]
    np.testing.assert_array_equal(response[key], np.asarray(expected, dtype=np.float32))


@pytest.mark.parametrize("op", ["embed", "rerank"])
def test_expired_between_batches_does_not_start_another_call(
    runtime: tuple, monkeypatch: pytest.MonkeyPatch, op: str,
) -> None:
    server, model = runtime
    clock = FakeClock()
    monkeypatch.setattr(ms, "time", clock)
    model.after_call = lambda: setattr(clock, "now", clock.now + 2)
    response = server.handle_request(payload(op, 5, batch_size=2, deadline=clock.now + 1))
    assert not response["ok"]
    assert "deadline" in response["error"]
    assert len(model.calls) == 1
    assert server._inflight == 0
    assert not server._lock.locked()


@pytest.mark.parametrize("op", ["embed", "rerank"])
def test_expiry_on_later_oom_skips_retry_and_remaining_inputs(
    runtime: tuple, monkeypatch: pytest.MonkeyPatch, op: str,
) -> None:
    server, model = runtime
    clock = FakeClock()
    monkeypatch.setattr(ms, "time", clock)
    model.fail_calls.add(2)

    def advance_on_failure() -> None:
        if len(model.calls) == 2:
            clock.now += 2

    model.after_call = advance_on_failure
    response = server.handle_request(payload(op, 5, batch_size=2, deadline=clock.now + 1))
    assert not response["ok"]
    assert "deadline" in response["error"]
    assert len(model.calls) == 2
    assert server._sticky_cpu == {op}
    assert model.moves == []


@pytest.mark.parametrize("op", ["embed", "rerank"])
def test_different_model_request_cannot_interleave_successful_microbatches(
    runtime: tuple, monkeypatch: pytest.MonkeyPatch, op: str,
) -> None:
    server, model = runtime
    alternate = RecordingModel()
    getter = "_get_embed_model" if op == "embed" else "_get_reranker"
    loads: list[str] = []
    premature_releases: list[int] = []
    competitor_waiting = threading.Event()
    primary_thread = threading.get_ident()

    class ObservingLock:
        def __init__(self) -> None:
            self.lock = threading.Lock()

        def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
            if threading.get_ident() != primary_thread:
                competitor_waiting.set()
            return self.lock.acquire(blocking, timeout)

        def release(self) -> None:
            if threading.get_ident() == primary_thread and 0 < len(model.calls) < 3:
                premature_releases.append(len(model.calls))
            self.lock.release()

    server._lock = ObservingLock()

    def load(identity: str) -> RecordingModel:
        loads.append(identity)
        return model if identity == "primary" else alternate

    monkeypatch.setattr(server, getter, load)
    identity_field = "tier" if op == "embed" else "model_name"
    responses: list[dict] = []
    errors: list[Exception] = []

    def compete() -> None:
        try:
            responses.append(server.handle_request(payload(
                op, 2, batch_size=2, **{identity_field: "alternate"},
            )))
        except Exception as error:
            errors.append(error)

    competitor = threading.Thread(target=compete)

    def interleave() -> None:
        if len(model.calls) == 1:
            competitor.start()
            assert competitor_waiting.wait(timeout=2)
        assert loads == ["primary"]

    model.after_call = interleave
    try:
        response = server.handle_request(payload(
            op, 5, batch_size=2, **{identity_field: "primary"},
        ))
    finally:
        if competitor.ident is not None:
            competitor.join(timeout=3)
    assert response["ok"]
    assert not competitor.is_alive()
    assert errors == []
    assert len(responses) == 1 and responses[0]["ok"]
    assert loads == ["primary", "alternate"]
    assert premature_releases == []
    assert len(model.calls) == 3
    assert len(alternate.calls) == 1


@pytest.mark.parametrize("op", ["embed", "rerank"])
def test_throttler_sampling_stays_outside_model_lock(
    runtime: tuple, op: str,
) -> None:
    server, _model = runtime

    class NonblockingThrottler(RecordingThrottler):
        def before_batch(self) -> tuple[int, dict]:
            acquired = server._lock.acquire(blocking=False)
            assert acquired, "sensor sampling must not hold the model state lock"
            server._lock.release()
            return super().before_batch()

    throttler = NonblockingThrottler([2])
    server._throttler = throttler
    server._throttler_active = True
    assert server.handle_request(payload(op, 5, batch_size=8))["ok"]
    assert throttler.before_count == 1


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "2", None, [], {}])
@pytest.mark.parametrize("op", ["embed", "rerank"])
def test_invalid_batch_size_fails_before_client_transport_or_server_model_loading(
    runtime: tuple, monkeypatch: pytest.MonkeyPatch, op: str, bad: object,
) -> None:
    server, model = runtime
    monkeypatch.setattr(mc, "_request_with_autostart", lambda *_a, **_kw: pytest.fail("transport"))
    proxy = mc.EmbeddingProxy().encode if op == "embed" else mc.RerankerProxy().predict
    with pytest.raises(ValueError, match="positive integer"):
        proxy(["synthetic"], batch_size=bad)
    response = server.handle_request(payload(op, 2, batch_size=bad))
    assert not response["ok"]
    assert "positive integer" in response["error"]
    assert model.calls == []


@pytest.mark.parametrize("op", ["embed", "rerank"])
def test_proxy_supports_numpy_integer_and_preserves_timeout_and_identity(
    monkeypatch: pytest.MonkeyPatch, op: str,
) -> None:
    requests: list[dict] = []

    def record(request: dict, timeout: float | None = None) -> dict:
        assert timeout == 0.75
        requests.append(request)
        return {"ok": True, "vectors": np.zeros((1, 3)), "scores": np.zeros(1)}

    monkeypatch.setattr(mc, "_request_with_autostart", record)
    if op == "embed":
        mc.EmbeddingProxy("synthetic-tier").encode("synthetic", batch_size=np.int64(2), timeout=0.75)
        assert requests[0]["texts"] == ["synthetic"]
        assert requests[0]["tier"] == "synthetic-tier"
    else:
        mc.RerankerProxy("synthetic-model").predict([("q", "d")], batch_size=np.int64(2), timeout=0.75)
        assert requests[0]["model_name"] == "synthetic-model"
    assert requests[0]["batch_size"] == 2
    assert type(requests[0]["batch_size"]) is int
    assert requests[0]["op"] == op + "_batched"


@pytest.mark.parametrize("op", ["embed", "rerank"])
@pytest.mark.parametrize("legacy_daemon", [False, True])
def test_framed_protocol_enforces_bounds_or_rejects_old_daemon_before_inference(
    runtime: tuple, monkeypatch: pytest.MonkeyPatch, op: str, legacy_daemon: bool,
) -> None:
    server, model = runtime
    client_socket, server_socket = socket.socketpair()
    monkeypatch.setattr(ms, "_USE_UNIX", True)
    monkeypatch.setattr(mc, "_connect", lambda _deadline: client_socket)
    monkeypatch.setattr(mc, "_start_server", lambda **_kw: pytest.fail("unexpected autostart"))
    if legacy_daemon:
        current_handler = server._handle_request_inner

        def legacy_handler(request: dict) -> dict:
            if request["op"] not in ("embed", "rerank", "ping"):
                return {"ok": False, "error": f"Unknown op: {request['op']}"}
            return current_handler(request)

        monkeypatch.setattr(server, "_handle_request_inner", legacy_handler)
    thread = threading.Thread(target=server.handle_client, args=(server_socket,))
    thread.start()
    request = payload(op, 5)
    method = mc.EmbeddingProxy().encode if op == "embed" else mc.RerankerProxy().predict
    try:
        inputs = request["texts" if op == "embed" else "pairs"]
        if legacy_daemon:
            with pytest.raises(mc.ProtocolMismatchError, match="restart it after upgrading"):
                method(inputs, batch_size=2, timeout=2)
            assert model.calls == []
        else:
            output = method(inputs, batch_size=2, timeout=2)
            assert output.shape == ((5, 3) if op == "embed" else (5,))
            assert output.dtype == np.float32
            assert [len(items) for _, items, _ in model.calls] == [2, 2, 1]
            expected = vectors(inputs) if op == "embed" else [len(pair[1]) / 10 for pair in inputs]
            np.testing.assert_array_equal(output, np.asarray(expected, dtype=np.float32))
    finally:
        client_socket.close()
        server_socket.close()
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_output_count_or_dimension_mismatch_cannot_return_partial_success() -> None:
    with pytest.raises(ValueError, match="count"):
        ms._store_batch_result(None, [[1, 2]], 0, 2, 3)
    result = ms._store_batch_result(None, [[1, 2]], 0, 1, 2)
    with pytest.raises(ValueError, match="dimensions"):
        ms._store_batch_result(result, [[1, 2, 3]], 1, 1, 2)


class OrderArray:
    """Small indexing fake; these tests can run without importing NumPy."""

    def __init__(self, values: list, shape: tuple | None = None) -> None:
        self.values = list(values)
        self.shape = shape or ((len(values), len(values[0])) if values and isinstance(values[0], list)
                               else (len(values),))
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[object]:
        return iter(self.values)

    def __getitem__(self, key: object) -> object:
        if isinstance(key, OrderArray):
            return OrderArray([self.values[index] for index in key], (len(key), *self.shape[1:]))
        if isinstance(key, slice):
            values = self.values[key]
            return OrderArray(values, (len(values), *self.shape[1:]))
        return self.values[key]

    def __setitem__(self, key: object, value: "OrderArray") -> None:
        indices = list(key) if isinstance(key, OrderArray) else list(range(len(self)))[key]
        assert len(indices) == len(value)
        for index, row in zip(indices, value):
            self.values[index] = row


class OrderNP:
    ndarray = OrderArray
    float32 = "float32"

    def __init__(self) -> None:
        self.sort_sizes: list[int] = []
        self.allocations: list[tuple] = []
        self.after_sort: Callable[[], None] = lambda: None

    def argsort(self, values: object) -> OrderArray:
        keys = list(values)
        self.sort_sizes.append(len(keys))
        # Deliberately unstable ties make a second local sort observable.
        indices = sorted(range(len(keys)), key=lambda index: (keys[index], -index))
        self.after_sort()
        return OrderArray(indices)

    def asarray(self, values: object, dtype: object = None) -> OrderArray:
        return values if isinstance(values, OrderArray) else OrderArray(list(values))

    def empty(self, shape: tuple, dtype: object = None) -> OrderArray:
        self.allocations.append(shape)
        return OrderArray([None] * shape[0], shape)


class OrderModel:
    """Model rows identify their entire native batch and their position in it."""

    def __init__(self, numpy: OrderNP) -> None:
        self.numpy = numpy
        self.calls: list[list] = []
        self.limits: list[int] = []
        self.native_batches: list[list[str]] = []
        self.fail_calls: set[int] = set()
        self.after_call: Callable[[], None] = lambda: None
        self.after_move: Callable[[], None] = lambda: None
        self.moves: list[str] = []
        self.default_prompt = "Instruct: retain the source\nQuery: "

    @staticmethod
    def _input_length(text: str) -> int:
        return len(text)

    def _can_flatten_inputs(self) -> bool:
        return False

    def encode(self, texts: list, *, batch_size: int, show_progress_bar: bool) -> OrderArray:
        assert show_progress_bar is False
        self.calls.append(list(texts))
        self.limits.append(batch_size)
        self.after_call()
        if len(self.calls) in self.fail_calls:
            raise RuntimeError("MPS backend out of memory")
        order = self.numpy.argsort([-len(text) for text in texts])
        rows = [None] * len(texts)
        for start in range(0, len(texts), batch_size):
            indices = order[start:start + batch_size]
            features = [self.default_prompt + str(texts[index]) for index in indices]
            self.native_batches.append(features)
            fingerprint = sum((index + 1) * sum(map(ord, text)) for index, text in enumerate(features))
            for position, index in enumerate(indices):
                rows[index] = [fingerprint, position, sum(map(ord, features[position]))]
        return OrderArray(rows, (len(texts), 3))

    def to(self, device: str) -> None:
        self.moves.append(device)
        self.after_move()


def load_order_runtime() -> dict:
    """Execute actual control methods with stdlib dependencies and array fakes."""
    import ast
    import builtins
    import json
    import logging
    import math
    from contextlib import contextmanager
    from dataclasses import dataclass
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "truememory/model_server.py"
    functions = {"_array_metadata", "_array_base64_size", "_result_wire_size", "_check_result_size",
                 "_batch_limit", "_checked_batch", "_store_batch_result"}
    classes = {"_EmbedState", "_RequestDeadline", "_RequestDeadlineExceeded", "_ResultTooLarge"}
    methods = {"handle_request", "_handle_request_inner", "_handle_fast_embed", "_embed_global_order",
               "_embed_slice_indices", "_preflight_embed_result", "_resolve_embed_cache",
               "_request_batch_limit", "_after_request_batches", "_recover_embed_oom_locked",
               "_check_embed_recovery_deadline_locked", "_rerank_global_order", "_rerank_slice_indices",
               "_preflight_rerank_result"}
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in classes:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "ModelServer":
            node.body = [part for part in node.body if isinstance(part, ast.Assign)
                         or isinstance(part, ast.FunctionDef) and part.name in methods]
            body.append(node)

    mps = types.SimpleNamespace(is_mps_oom=lambda error: "out of memory" in str(error),
                                flush_mps_cache=lambda: None)

    def safe_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "__future__":
            return builtins.__import__(name, *args, **kwargs)
        if name == "truememory.mps_utils":
            return mps
        if name == "truememory.reranker":
            return types.SimpleNamespace(get_current_reranker_name=lambda: "Alibaba-NLP/gte-reranker-modernbert-base")
        raise AssertionError("Unexpected runtime import: " + name)

    numpy, clock = OrderNP(), FakeClock()
    namespace = {"__name__": __name__, "np": numpy, "time": clock, "math": math, "json": json,
                 "threading": threading, "contextmanager": contextmanager, "dataclass": dataclass,
                 "log": logging.getLogger("synthetic-order"), "PROTOCOL_VERSION": 1,
                 "_MAX_MESSAGE_SIZE": 10 * 1024**2, "_EMBED_BATCH_LIMIT": 32, "_RERANK_BATCH_LIMIT": 64,
                 "__builtins__": dict(vars(builtins), __import__=safe_import), "mps": mps}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


class TestQwenGlobalOrder(unittest.TestCase):
    def runtime(self) -> tuple[dict, object, OrderModel]:
        module = load_order_runtime()
        server = object.__new__(module["ModelServer"])
        model = OrderModel(module["np"])
        server._embed_state = module["_EmbedState"](model, "base", "qwen3_256")
        server._lock = threading.Lock()
        server._inference_lock = threading.Lock()
        server._activity_lock = threading.Lock()
        server._fast_lock = threading.Lock()
        server._transport_context = types.SimpleNamespace()
        server._embed_timestamps = []
        server._throttler_active = False
        server._throttler = None
        server._inflight = 0
        server._sticky_cpu = set()
        server.loads = 0

        def load(_tier: str) -> OrderModel:
            server.loads += 1
            return model

        server._get_embed_model = load
        server._get_fast_encoder = load
        server._mark_sticky_cpu = server._sticky_cpu.add
        server._deactivate_throttler = lambda: None
        server._flush_mps_cache = lambda: None
        return module, server, model

    @staticmethod
    def texts(count: int) -> list[str]:
        samples = ["", "same", "same", "aa", "bb", "é", "東京", "x" * 100, "tail"]
        return [samples[(index * 5 + index // 9) % len(samples)] for index in range(count)]

    def request(self, server: object, texts: list, **kwargs: object) -> dict:
        return server.handle_request({"op": "embed", "tier": "base", "texts": texts, **kwargs})

    def reference(self, texts: list, limit: int) -> tuple[OrderArray, list[list[str]]]:
        model = OrderModel(OrderNP())
        values = model.encode(texts, batch_size=limit, show_progress_bar=False)
        return values, model.native_batches

    def contiguous_reference(self, texts: list, limit: int) -> tuple[OrderArray, list[list[str]]]:
        model = OrderModel(OrderNP())
        values = []
        for start in range(0, max(1, len(texts)), limit):
            values.extend(model.encode(texts[start:start + limit], batch_size=limit,
                                       show_progress_bar=False).values)
        return OrderArray(values, (len(texts), 3)), model.native_batches

    def test_native_membership_ties_duplicates_unicode_and_scatter(self) -> None:
        for count in (0, 1, 2, 8, 31, 32, 33, 63, 64, 65, 100):
            for requested in (1, 2, 8, 32, 100):
                with self.subTest(count=count, requested=requested):
                    module, server, model = self.runtime()
                    texts = self.texts(count)
                    limit = min(requested, 32)
                    reference = self.reference if limit == 32 else self.contiguous_reference
                    expected, batches = reference(texts, limit)
                    response = self.request(server, texts, batch_size=requested)
                    self.assertTrue(response["ok"])
                    self.assertEqual(response["vectors"].values, expected.values)
                    self.assertEqual(model.native_batches, batches)
                    self.assertTrue(all(len(call) <= bound <= limit for call, bound in zip(model.calls, model.limits)))
                    self.assertLessEqual(len(module["np"].allocations), 1)
                    if limit != 32:
                        self.assertEqual(model.calls, [texts[start:start + limit]
                                         for start in range(0, max(1, count), limit)])
                        self.assertEqual(module["np"].sort_sizes, [len(call) for call in model.calls])

    def test_default_request_preserves_native_32_policy(self) -> None:
        for count in (33, 65, 100):
            with self.subTest(count=count):
                _module, server, model = self.runtime()
                texts = self.texts(count)
                expected, batches = self.reference(texts, 32)
                response = self.request(server, texts)
                self.assertEqual(response["vectors"].values, expected.values)
                self.assertEqual(model.native_batches, batches)
                self.assertEqual(model.limits, [32] * len(model.calls))

    def test_single_and_one_effective_batch_do_not_plan_or_reorder_inputs(self) -> None:
        for count in (0, 1, 8, 32):
            module, server, model = self.runtime()
            texts = self.texts(count)
            self.assertTrue(self.request(server, texts)["ok"])
            self.assertEqual(model.calls, [texts])
            self.assertEqual(module["np"].sort_sizes, [count])

    def test_contended_single_input_retains_the_existing_fast_lane(self) -> None:
        module, server, model = self.runtime()
        model.after_call = lambda: self.assertTrue(server._fast_lock.locked() and not server._lock.locked())
        with server._inference_lock:
            response = self.request(server, ["synthetic query"])
        self.assertTrue(response["ok"])
        self.assertEqual(model.calls, [["synthetic query"]])
        self.assertEqual(model.limits, [1])
        self.assertEqual(module["np"].sort_sizes, [1])

    def test_unsupported_identity_interface_flattening_and_inputs_keep_bounded_slices(self) -> None:
        for mode in ("other-model", "legacy", "no-flatten-probe", "flattened", "non-string"):
            with self.subTest(mode=mode):
                module, server, model = self.runtime()
                texts = self.texts(33)
                if mode == "other-model":
                    server._embed_state = module["_EmbedState"](model, "base", "model2vec")
                elif mode == "legacy":
                    model._input_length = None
                elif mode == "no-flatten-probe":
                    model._can_flatten_inputs = None
                elif mode == "flattened":
                    model._can_flatten_inputs = lambda: True
                else:
                    texts[5] = ["synthetic", "pair"]
                self.assertTrue(self.request(server, texts, batch_size=32)["ok"])
                self.assertEqual(model.calls, [texts[start:start + 32] for start in range(0, len(texts), 32)])
                self.assertNotIn(33, module["np"].sort_sizes)

    def test_negative_controls_detect_missing_global_order_compensation_and_scatter(self) -> None:
        for missing in ("global-order", "tie-compensation", "scatter"):
            with self.subTest(missing=missing):
                module, server, model = self.runtime()
                texts = self.texts(65)
                expected, batches = self.reference(texts, 32)
                if missing == "global-order":
                    server._embed_global_order = lambda *_args: None
                elif missing == "tie-compensation":
                    server._embed_slice_indices = lambda _model, _texts, order, offset, limit, _deadline: order[offset:offset + limit]
                else:
                    store = module["_store_batch_result"]
                    module["_store_batch_result"] = lambda *args, **_kwargs: store(*args)
                response = self.request(server, texts, batch_size=32)
                self.assertNotEqual(response["vectors"].values, expected.values)
                if missing != "scatter":
                    self.assertNotEqual(model.native_batches, batches)

    def test_negative_control_detects_missing_effective_limit_guard(self) -> None:
        import ast
        from pathlib import Path
        from unittest.mock import patch

        path = Path(__file__).resolve().parents[1] / "truememory/model_server.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        server_class = next(node for node in tree.body
                            if isinstance(node, ast.ClassDef) and node.name == "ModelServer")
        method = next(node for node in server_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_embed_global_order")
        removed = 0
        for node in ast.walk(method):
            if not isinstance(node, ast.BoolOp):
                continue
            retained = []
            for value in node.values:
                if (isinstance(value, ast.Compare) and isinstance(value.left, ast.Name)
                        and value.left.id == "limit" and len(value.ops) == 1
                        and isinstance(value.ops[0], ast.NotEq)
                        and isinstance(value.comparators[0], ast.Constant)
                        and value.comparators[0].value == 32):
                    removed += 1
                else:
                    retained.append(value)
            node.values = retained
        self.assertEqual(removed, 1)
        with patch.object(Path, "read_text", return_value=ast.unparse(tree)):
            for adaptive in (False, True):
                with self.subTest(adaptive=adaptive):
                    _module, server, model = self.runtime()
                    if adaptive:
                        server._throttler = RecordingThrottler([8])
                        server._throttler_active = True
                    texts = self.texts(65)
                    expected, batches = self.contiguous_reference(texts, 8)
                    response = self.request(server, texts, **({} if adaptive else {"batch_size": 8}))
                    self.assertTrue(response["ok"])
                    self.assertNotEqual(response["vectors"].values, expected.values)
                    self.assertNotEqual(model.native_batches, batches)
                    self.assertNotEqual(model.calls, [texts[start:start + 8] for start in range(0, 65, 8)])

    def test_cyclic_ties_require_the_inverse_and_place_each_occurrence_once(self) -> None:
        from unittest.mock import patch

        placements: list[int] = []
        setitem = OrderArray.__setitem__

        def record_placement(array: OrderArray, key: object, value: OrderArray) -> None:
            indices = list(key) if isinstance(key, OrderArray) else list(range(len(array)))[key]
            placements.extend(indices)
            setitem(array, key, value)

        def cyclic_argsort(numpy: OrderNP, values: object) -> OrderArray:
            keys = list(values)
            numpy.sort_sizes.append(len(keys))
            groups: dict[int, list[int]] = {}
            for index, key in enumerate(keys):
                groups.setdefault(key, []).append(index)
            order: list[int] = []
            for key in sorted(groups):
                group = groups[key]
                # Reversing ties is its own inverse; cycles expose direction errors.
                order.extend(group[1:] + group[:1])
            numpy.after_sort()
            return OrderArray(order)

        with patch.object(OrderNP, "argsort", cyclic_argsort), \
                patch.object(OrderArray, "__setitem__", record_placement):
            for count in (31, 33, 63, 64, 65):
                for limit in (8, 32):
                    with self.subTest(count=count, limit=limit):
                        texts = self.texts(count)
                        reference = self.reference if limit == 32 else self.contiguous_reference
                        expected, batches = reference(texts, limit)
                        _module, server, model = self.runtime()
                        placements.clear()
                        response = self.request(server, texts, batch_size=limit)
                        self.assertTrue(response["ok"])
                        self.assertEqual(response["vectors"].values, expected.values)
                        self.assertEqual(model.native_batches, batches)
                        self.assertEqual(sorted(placements), list(range(count)))
                        if count <= limit or limit != 32:
                            continue

                        module, server, model = self.runtime()

                        def wrong_inverse(
                            model: object, texts: list, order: OrderArray,
                            offset: int, limit: int, deadline: object,
                        ) -> OrderArray:
                            deadline.check()
                            indices = order[offset:offset + limit]
                            inner = module["np"].argsort([-model._input_length(texts[index]) for index in indices])
                            deadline.check()
                            return indices[inner]

                        server._embed_slice_indices = wrong_inverse
                        placements.clear()
                        response = self.request(server, texts, batch_size=limit)
                        self.assertTrue(response["ok"])
                        self.assertEqual(sorted(placements), list(range(count)))
                        self.assertNotEqual(model.native_batches, batches)
                        self.assertNotEqual(response["vectors"].values, expected.values)

    def test_adaptive_limit_is_captured_once_and_cannot_raise_caller_cap(self) -> None:
        for limits, requested, effective in (([2, 1, 16], 8, 2), ([64], 8, 8),
                                             ([8, 2, 32], None, 8), ([8], 32, 8),
                                             ([8], 100, 8), ([64], 100, 32)):
            module, server, model = self.runtime()
            throttler = RecordingThrottler(limits)
            server._throttler, server._throttler_active = throttler, True
            texts = self.texts(65)
            reference = self.reference if effective == 32 else self.contiguous_reference
            expected, batches = reference(texts, effective)
            response = self.request(server, texts, **({} if requested is None else {"batch_size": requested}))
            self.assertEqual(response["vectors"].values, expected.values)
            self.assertEqual(model.native_batches, batches)
            self.assertEqual(model.limits, [effective] * len(model.calls))
            self.assertEqual(throttler.before_count, 1)
            self.assertEqual(throttler.after_counts, [65])
            if effective != 32:
                self.assertEqual(model.calls, [texts[start:start + effective] for start in range(0, 65, effective)])
                self.assertEqual(module["np"].sort_sizes, [len(call) for call in model.calls])

    def test_result_preflight_precedes_model_load_and_planning(self) -> None:
        module, server, model = self.runtime()
        response = self.request(server, [""] * 7680)
        self.assertEqual(response["error_code"], "result_too_large")
        self.assertEqual(server.loads, 0)
        self.assertEqual(model.calls, [])
        self.assertEqual(module["np"].sort_sizes, [])
        self.assertEqual(module["np"].allocations, [])

    def test_ordered_output_count_and_dimension_guards_remain_active(self) -> None:
        for mismatch in ("count", "dimensions"):
            _module, server, model = self.runtime()
            encode = model.encode

            def invalid(texts: list, **kwargs: object) -> OrderArray:
                values = encode(texts, **kwargs)
                if mismatch == "count":
                    return OrderArray(values.values[:-1], (len(values) - 1, 3))
                if len(model.calls) == 2:
                    return OrderArray([row[:2] for row in values], (len(values), 2))
                return values

            model.encode = invalid
            with self.assertRaisesRegex(ValueError, mismatch):
                self.request(server, self.texts(65), batch_size=32)
            self.assertFalse(server._inference_lock.locked())
            self.assertFalse(server._lock.locked())

    def test_expiry_before_planning_and_after_each_sort_stops_encode(self) -> None:
        for phase in ("entry", "model-load", "flatten-probe", "lengths", "global-sort", "inner-sort", "inverse-sort"):
            with self.subTest(phase=phase):
                module, server, model = self.runtime()
                clock, numpy = module["time"], module["np"]
                expires = clock.now + 1

                def expire() -> None:
                    clock.now += 2

                if phase == "entry":
                    expires = clock.now
                elif phase == "model-load":
                    server._get_embed_model = lambda _tier: (expire(), model)[1]
                elif phase == "flatten-probe":
                    model._can_flatten_inputs = lambda: (expire(), False)[1]
                elif phase == "lengths":
                    model._input_length = lambda text: (expire(), len(text))[1]
                else:
                    wanted = {"global-sort": 1, "inner-sort": 2, "inverse-sort": 3}[phase]
                    numpy.after_sort = lambda: expire() if len(numpy.sort_sizes) == wanted else None
                response = self.request(server, self.texts(65), batch_size=32, deadline=expires)
                self.assertFalse(response["ok"])
                self.assertIn("deadline", response["error"])
                self.assertEqual(model.calls, [])
                self.assertEqual(numpy.sort_sizes, [] if phase in ("entry", "model-load", "flatten-probe", "lengths")
                                 else [65, 32, 32][:wanted])
                self.assertFalse(server._inference_lock.locked())
                self.assertFalse(server._lock.locked())

    def test_expiry_after_slice_does_not_plan_or_encode_another_slice(self) -> None:
        module, server, model = self.runtime()
        expires = module["time"].now + 1
        model.after_call = lambda: setattr(module["time"], "now", expires + 1)
        response = self.request(server, self.texts(65), batch_size=32, deadline=expires)
        self.assertFalse(response["ok"])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(module["np"].sort_sizes, [65, 32, 32, 32])

    def test_oom_retries_exact_slice_with_frozen_cursor_and_exclusive_owner(self) -> None:
        for limit in (8, 32):
            with self.subTest(limit=limit):
                module, server, model = self.runtime()
                texts = self.texts(65)
                reference = self.reference if limit == 32 else self.contiguous_reference
                expected, batches = reference(texts, limit)
                model.fail_calls.add(2)
                state_locks = []

                def observe() -> None:
                    self.assertTrue(server._inference_lock.locked())
                    state_locks.append(server._lock.locked())

                model.after_call = observe
                model.after_move = lambda: self.assertTrue(server._lock.locked() and server._inference_lock.locked())
                response = self.request(server, texts, batch_size=limit)
                self.assertEqual(response["vectors"].values, expected.values)
                self.assertEqual(model.native_batches, batches)
                self.assertEqual(model.calls[1], model.calls[2])
                batches_count = (len(texts) + limit - 1) // limit
                self.assertEqual(len(model.calls), batches_count + 1)
                self.assertEqual(state_locks, [True, True, False] + [True] * (batches_count - 2))
                self.assertEqual(module["np"].sort_sizes.count(65), int(limit == 32))
                self.assertEqual(module["np"].allocations, [(65, 3)])
                self.assertEqual(model.moves, ["cpu"])
                self.assertEqual(server._sticky_cpu, {"embed"})

    def test_expiry_on_oom_or_during_recovery_never_retries(self) -> None:
        for limit in (8, 32):
            for phase in ("oom", "flush", "move"):
                with self.subTest(limit=limit, phase=phase):
                    module, server, model = self.runtime()
                    expires = module["time"].now + 1
                    model.fail_calls.add(2)

                    def expire() -> None:
                        module["time"].now = expires + 1

                    if phase == "oom":
                        model.after_call = lambda: expire() if len(model.calls) == 2 else None
                    elif phase == "flush":
                        module["mps"].flush_mps_cache = expire
                    else:
                        model.after_move = expire
                    response = self.request(server, self.texts(65), batch_size=limit, deadline=expires)
                    self.assertFalse(response["ok"])
                    self.assertEqual(len(model.calls), 2)
                    self.assertEqual(model.moves, ["cpu"] if phase == "move" else [])
                    self.assertEqual(server._sticky_cpu, {"embed"})
                    self.assertFalse(server._inference_lock.locked())
                    self.assertFalse(server._lock.locked())


class PairOrderNP(OrderNP):
    def argsort(self, values: object) -> OrderArray:
        keys = list(values)
        self.sort_sizes.append(len(keys))
        groups: dict[int, list[int]] = {}
        for index, key in enumerate(keys):
            groups.setdefault(key, []).append(index)
        order = []
        for key in sorted(groups):
            group = groups[key]
            order.extend(group[1:] + group[:1])
        self.after_sort()
        return OrderArray(order)


class PairOrderModel:
    """Predict's pinned ST 6.1 pair sorting and restoration, with synthetic scores.

    cross_encoder/model.py SHA256 c8688911838df2acd48550e80dce2743c1f31dd85187f98d688517af1ce575e8:
    lines 695-736 resolve the prompt, sort raw pair lengths, preprocess batches,
    and restore scores. Features and scores here expose membership and position.
    """

    def __init__(self, numpy: OrderNP) -> None:
        self.numpy = numpy
        self.calls: list[list] = []
        self.limits: list[int] = []
        self.native_batches: list[list[tuple[str, str]]] = []
        self.fail_calls: set[int] = set()
        self.after_call: Callable[[], None] = lambda: None
        self.default_prompt = "Synthetic instruction: "

    @staticmethod
    def _input_length(pair: tuple | list) -> int:
        return sum(len(str(text)) for text in pair)

    def _can_flatten_inputs(self) -> bool:
        return False

    def predict(self, pairs: list, *, batch_size: int, show_progress_bar: bool) -> OrderArray:
        assert show_progress_bar is False
        self.calls.append(list(pairs))
        self.limits.append(batch_size)
        self.after_call()
        if len(self.calls) in self.fail_calls:
            raise RuntimeError("MPS backend out of memory")
        order = self.numpy.argsort([-PairOrderModel._input_length(pair) for pair in pairs])
        rows = [None] * len(pairs)
        for start in range(0, len(pairs), batch_size):
            indices = order[start:start + batch_size]
            features = [(self.default_prompt + str(pairs[index][0]), str(pairs[index][1])) for index in indices]
            self.native_batches.append(features)
            fingerprint = sum((index + 1) * sum((offset + 1) * ord(char) for offset, char in enumerate(str(pair)))
                              for index, pair in enumerate(features))
            for position, index in enumerate(indices):
                rows[index] = float(fingerprint * 1000 + position)
        return OrderArray(rows, (len(pairs),))


class TestModernBertGlobalOrder(unittest.TestCase):
    identity = "Alibaba-NLP/gte-reranker-modernbert-base"

    def runtime(self) -> tuple[dict, object, PairOrderModel]:
        module = load_order_runtime()
        module["np"] = PairOrderNP()
        server = object.__new__(module["ModelServer"])
        model = PairOrderModel(module["np"])
        server._lock = threading.Lock()
        server._inference_lock = threading.Lock()
        server._activity_lock = threading.Lock()
        server._inflight = 0
        server._transport_context = types.SimpleNamespace()
        server._throttler_active, server._throttler = False, None
        server._sticky_cpu = set()
        server._reranker, server._reranker_name = None, None
        server._mark_sticky_cpu = server._sticky_cpu.add
        server.default_name = self.identity
        server.loads, server.models = [], []
        server.after_load = lambda _model: None

        def load(name: str | None) -> PairOrderModel:
            server.loads.append(name)
            loaded = model if not server.models else PairOrderModel(module["np"])
            loaded.default_prompt = model.default_prompt
            server.models.append(loaded)
            server._reranker, server._reranker_name = loaded, name or server.default_name
            server.after_load(loaded)
            return loaded

        server._get_reranker = load
        return module, server, model

    @staticmethod
    def pairs(count: int) -> list[tuple[str, str]]:
        samples = [("", ""), ("same", "same"), ("same", "same"), ("雪", "a"),
                   ("e\u0301", ""), ("q" * 100, "d"), ("q", "d" * 100), ("query", "doc")]
        return [samples[(index * 3 + index // 8) % len(samples)] for index in range(count)]

    def request(self, server: object, pairs: list, **kwargs: object) -> dict:
        return server.handle_request({"op": "rerank", "pairs": pairs, **kwargs})

    def reference(self, pairs: list, *, prompt: str = "Synthetic instruction: ") -> PairOrderModel:
        model = PairOrderModel(PairOrderNP())
        model.default_prompt = prompt
        model.output = model.predict(pairs, batch_size=64, show_progress_bar=False)
        return model

    def test_default_64_native_membership_cyclic_ties_duplicates_and_scatter(self) -> None:
        for count in (63, 64, 65, 100, 127, 128, 129):
            for prompt in ("", "Synthetic instruction: "):
                with self.subTest(count=count, prompt=prompt):
                    module, server, model = self.runtime()
                    model.default_prompt = prompt
                    pairs = self.pairs(count)
                    expected = self.reference(pairs, prompt=prompt)
                    response = self.request(server, pairs)
                    self.assertTrue(response["ok"])
                    self.assertEqual(response["scores"].values, expected.output.values)
                    self.assertEqual(model.native_batches, expected.native_batches)
                    self.assertEqual(model.limits, [64] * len(model.calls))
                    self.assertEqual(module["np"].allocations, [(count,)])

    def test_each_duplicate_occurrence_is_written_exactly_once(self) -> None:
        from unittest.mock import patch

        for count in (65, 100, 129):
            placements = []
            original = OrderArray.__setitem__

            def store(array: OrderArray, key: object, values: OrderArray) -> None:
                placements.extend(list(key) if isinstance(key, OrderArray) else list(range(len(array)))[key])
                original(array, key, values)

            _module, server, _model = self.runtime()
            with patch.object(OrderArray, "__setitem__", store):
                response = self.request(server, self.pairs(count))
            self.assertTrue(response["ok"])
            self.assertEqual(sorted(placements), list(range(count)))

    def test_negative_controls_reject_global_ties_inverse_scatter_and_document_only_lengths(self) -> None:
        for mutation in ("global", "compensation", "inverse", "scatter", "document-length"):
            with self.subTest(mutation=mutation):
                module, server, model = self.runtime()
                pairs = self.pairs(129)
                expected = self.reference(pairs)
                if mutation == "global":
                    server._rerank_global_order = lambda *_args: None
                elif mutation == "compensation":
                    server._rerank_slice_indices = lambda _model, _pairs, order, offset, limit, _deadline: order[offset:offset + limit]
                elif mutation == "inverse":
                    def wrong_inverse(model: object, pairs: list, order: OrderArray, offset: int,
                                      limit: int, deadline: object) -> OrderArray:
                        deadline.check()
                        indices = order[offset:offset + limit]
                        inner = module["np"].argsort([-model._input_length(pairs[index]) for index in indices])
                        return indices[inner]
                    server._rerank_slice_indices = wrong_inverse
                elif mutation == "scatter":
                    store = module["_store_batch_result"]
                    module["_store_batch_result"] = lambda *args, **_kwargs: store(*args)
                else:
                    model._input_length = lambda pair: len(pair[1])
                response = self.request(server, pairs)
                self.assertTrue(response["ok"])
                self.assertNotEqual(response["scores"].values, expected.output.values)
                if mutation != "scatter":
                    self.assertNotEqual(model.native_batches, expected.native_batches)

    def test_single_empty_one_batch_and_lower_caps_preserve_contiguous_calls(self) -> None:
        for count, requested in ((0, 64), (1, 64), (8, 64), (64, 64), (129, 1), (129, 8), (129, 32)):
            with self.subTest(count=count, requested=requested):
                module, server, model = self.runtime()
                pairs = self.pairs(count)
                self.assertTrue(self.request(server, pairs, batch_size=requested)["ok"])
                self.assertEqual(model.calls, [pairs[i:i + requested] for i in range(0, count, requested)] or [[]])
                self.assertEqual(module["np"].sort_sizes, [len(call) for call in model.calls])

    def test_unsupported_identity_interface_flattening_and_pairs_keep_old_path(self) -> None:
        for mode in ("custom", "stale-cache", "legacy", "missing-flatten", "flattened", "three-strings", "non-string"):
            with self.subTest(mode=mode):
                module, server, model = self.runtime()
                pairs = self.pairs(100)
                if mode == "custom":
                    server.default_name = "synthetic-custom"
                elif mode == "stale-cache":
                    server.after_load = lambda _model: setattr(server, "_reranker", object())
                elif mode == "legacy":
                    model._input_length = None
                elif mode == "missing-flatten":
                    model._can_flatten_inputs = None
                elif mode == "flattened":
                    model._can_flatten_inputs = lambda: True
                elif mode == "three-strings":
                    pairs[5] = ("q", "d", "extra")
                else:
                    pairs[5] = ("q", 17)
                self.assertTrue(self.request(server, pairs)["ok"])
                self.assertEqual(model.calls, [pairs[:64], pairs[64:]])
                self.assertEqual(module["np"].sort_sizes, [64, 36])

    def test_caller_server_and_once_per_request_adaptive_limits(self) -> None:
        for requested, samples, effective in ((8, [128], 8), (128, [128], 64), (64, [8, 64], 8), (128, [64, 1], 64)):
            module, server, model = self.runtime()
            throttler = RecordingThrottler(samples)
            server._throttler, server._throttler_active = throttler, True
            pairs = self.pairs(129)
            self.assertTrue(self.request(server, pairs, batch_size=requested)["ok"])
            self.assertEqual(model.limits, [effective] * len(model.calls))
            self.assertEqual(throttler.before_count, 1)
            self.assertEqual(throttler.after_counts, [129])
            if effective == 64:
                self.assertEqual(model.native_batches, self.reference(pairs).native_batches)
            else:
                self.assertEqual(model.calls, [pairs[i:i + effective] for i in range(0, len(pairs), effective)])

    def test_result_preflight_precedes_loading_and_ordering(self) -> None:
        module, server, model = self.runtime()
        module["_MAX_MESSAGE_SIZE"] = 200
        response = self.request(server, self.pairs(129))
        self.assertEqual(response["error_code"], "result_too_large")
        self.assertEqual(server.loads, [])
        self.assertEqual(model.calls, [])
        self.assertEqual(module["np"].sort_sizes, [])
        self.assertEqual(module["np"].allocations, [])

    def test_deadlines_before_work_and_after_each_sort_stop_prediction(self) -> None:
        for phase in ("entry", "load", "flatten", "length", "global", "inner", "inverse"):
            with self.subTest(phase=phase):
                module, server, model = self.runtime()
                clock, numpy = module["time"], module["np"]
                expires = clock.now + 1

                def expire() -> None:
                    clock.now = expires + 1

                if phase == "entry":
                    expires = clock.now
                elif phase == "load":
                    server.after_load = lambda _model: expire()
                elif phase == "flatten":
                    model._can_flatten_inputs = lambda: (expire(), False)[1]
                elif phase == "length":
                    model._input_length = lambda pair: (expire(), PairOrderModel._input_length(pair))[1]
                else:
                    wanted = {"global": 1, "inner": 2, "inverse": 3}[phase]
                    numpy.after_sort = lambda: expire() if len(numpy.sort_sizes) == wanted else None
                response = self.request(server, self.pairs(129), deadline=expires)
                self.assertFalse(response["ok"])
                self.assertIn("deadline", response["error"])
                self.assertEqual(model.calls, [])
                self.assertEqual(numpy.sort_sizes, [] if phase in ("entry", "load", "flatten", "length") else [129, 64, 64][:wanted])
                self.assertFalse(server._inference_lock.locked())
                self.assertFalse(server._lock.locked())

    def test_expiry_after_first_slice_stops_following_work(self) -> None:
        module, server, model = self.runtime()
        expires = module["time"].now + 1
        model.after_call = lambda: setattr(module["time"], "now", expires + 1)
        response = self.request(server, self.pairs(129), deadline=expires)
        self.assertFalse(response["ok"])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(module["np"].sort_sizes, [129, 64, 64, 64])

    def test_later_oom_retries_frozen_slice_with_captured_omitted_identity(self) -> None:
        module, server, model = self.runtime()
        pairs = self.pairs(129)
        expected = self.reference(pairs)
        model.fail_calls.add(2)
        locks = []

        def observe() -> None:
            self.assertTrue(server._inference_lock.locked())
            locks.append(server._lock.locked())
            if len(model.calls) == 2:
                server.default_name = "synthetic-different-reranker"

        model.after_call = observe

        def loaded(replacement: PairOrderModel) -> None:
            self.assertTrue(server._inference_lock.locked() and server._lock.locked())
            if replacement is not model:
                replacement.after_call = observe

        server.after_load = loaded
        response = self.request(server, pairs)
        replacement = server.models[1]
        self.assertEqual(server.loads, [None, self.identity])
        self.assertEqual(server._reranker_name, self.identity)
        self.assertEqual(model.calls[1], replacement.calls[0])
        self.assertEqual(model.native_batches + replacement.native_batches, expected.native_batches)
        self.assertEqual(response["scores"].values, expected.output.values)
        self.assertEqual(locks, [True, True, False, True])
        self.assertEqual(module["np"].sort_sizes.count(129), 1)
        self.assertEqual(module["np"].allocations, [(129,)])
        self.assertEqual(server._sticky_cpu, {"rerank"})

    def test_lower_limit_recovery_keeps_existing_resolution_behavior(self) -> None:
        _module, server, model = self.runtime()
        model.fail_calls.add(2)
        model.after_call = lambda: setattr(server, "default_name", "synthetic-other") if len(model.calls) == 2 else None
        self.assertTrue(self.request(server, self.pairs(17), batch_size=8)["ok"])
        self.assertEqual(server.loads, [None, None])
        self.assertEqual(server._reranker_name, "synthetic-other")

    def test_deadline_on_oom_or_during_replacement_prevents_retry(self) -> None:
        for phase in ("oom", "flush", "replacement"):
            module, server, model = self.runtime()
            expires = module["time"].now + 1
            model.fail_calls.add(2)

            def expire() -> None:
                module["time"].now = expires + 1

            if phase == "oom":
                model.after_call = lambda: expire() if len(model.calls) == 2 else None
            elif phase == "flush":
                module["mps"].flush_mps_cache = expire
            else:
                server.after_load = lambda replacement: expire() if replacement is not model else None
            response = self.request(server, self.pairs(129), deadline=expires)
            self.assertFalse(response["ok"])
            self.assertEqual(len(model.calls), 2)
            self.assertTrue(all(not replacement.calls for replacement in server.models[1:]))
            self.assertEqual(server.loads, [None, self.identity] if phase == "replacement" else [None])
            self.assertFalse(server._inference_lock.locked())
            self.assertFalse(server._lock.locked())

    def test_ordered_score_count_and_shape_validation_remain_active(self) -> None:
        for mismatch in ("count", "dimensions"):
            _module, server, model = self.runtime()
            predict = model.predict

            def invalid(pairs: list, **kwargs: object) -> OrderArray:
                result = predict(pairs, **kwargs)
                if mismatch == "count":
                    return OrderArray(result.values[:-1])
                if len(model.calls) == 2:
                    return OrderArray([[value, value] for value in result])
                return result

            model.predict = invalid
            with self.assertRaisesRegex(ValueError, mismatch):
                self.request(server, self.pairs(129))
            self.assertFalse(server._inference_lock.locked())
            self.assertFalse(server._lock.locked())
