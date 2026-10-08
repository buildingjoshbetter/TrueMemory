"""Synthetic batch-control regressions, including real framed socket transport."""
from __future__ import annotations

import socket
import threading
from collections.abc import Callable

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

    def encode(self, texts: list[str], *, batch_size: int,
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
    ("embed", "vectors", ms._EMBED_BATCH_LIMIT),
    ("rerank", "scores", ms._RERANK_BATCH_LIMIT),
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
