"""Synthetic deadline regressions; no models, sockets, or real clock waits."""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from truememory import model_server as ms


class FakeClock:
    def __init__(self) -> None:
        self.wall = 1000.0
        self.tick = 100.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.tick

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.tick += seconds


class FakeLock:
    """Inject a wait at one acquisition, including a successful late wakeup."""

    def __init__(
        self, clock: FakeClock, *, wait_at: int = 0, wait: float = 2.0,
        contended: bool = False, time_out: bool = False,
    ) -> None:
        self.clock = clock
        self.wait_at = wait_at
        self.wait = wait
        self.contended = contended
        self.time_out = time_out
        self.calls = 0
        self.timeouts: list[float] = []
        self.held = False

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if not blocking and self.contended:
            return False
        self.calls += 1
        self.timeouts.append(timeout)
        if self.calls == self.wait_at:
            self.clock.advance(self.wait)
            if self.time_out and timeout >= 0:
                return False
        self.held = True
        return True

    def release(self) -> None:
        assert self.held, "release without acquisition"
        self.held = False

    def __enter__(self) -> FakeLock:
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class FakeModel:
    def __init__(self, clock: FakeClock, *, oom: bool = False, delay: float = 0) -> None:
        self.clock = clock
        self.oom = oom
        self.delay = delay
        self.calls = 0

    def encode(self, texts: list[str], **_kwargs: object) -> list[list[float]]:
        self.calls += 1
        self.clock.advance(self.delay)
        if self.oom and self.calls == 1:
            raise RuntimeError("MPS backend out of memory")
        return [[1.0, 0.0] for _ in texts]

    def predict(self, pairs: list[tuple[str, str]], **_kwargs: object) -> list[float]:
        self.calls += 1
        self.clock.advance(self.delay)
        if self.oom and self.calls == 1:
            raise RuntimeError("MPS backend out of memory")
        return [0.5 for _ in pairs]


class TestIssue736RequestDeadlines(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        clock_patch = patch.object(ms, "time", self.clock)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.server = ms.ModelServer()
        self.server._lock = FakeLock(self.clock)
        self.server._fast_lock = FakeLock(self.clock)
        self.model = FakeModel(self.clock)
        self.loads: list[str] = []
        self.recoveries: list[str] = []

        def load(tier: str | None) -> FakeModel:
            self.loads.append(tier or "fixture")
            return self.model

        self.server._get_embed_model = load
        self.server._get_fast_encoder = load
        self.server._get_reranker = load
        self.server._recover_embed_oom_locked = lambda model, deadline=None: self.recoveries.append("embed")
        self.server._mark_sticky_cpu = lambda kind: self.recoveries.append(kind)
        self.server._flush_mps_cache = lambda: None
        mps = types.ModuleType("truememory.mps_utils")
        mps.is_mps_oom = lambda error: "MPS backend out of memory" in str(error)
        mps.flush_mps_cache = lambda: None
        mps_patch = patch.dict("sys.modules", {"truememory.mps_utils": mps})
        mps_patch.start()
        self.addCleanup(mps_patch.stop)

    def request(self, op: str = "embed", *, single: bool = False) -> dict:
        payload = {"op": op, "deadline": self.clock.wall + 1.0}
        if op == "embed":
            payload["texts"] = ["fixture-a"] if single else ["fixture-a", "fixture-b"]
        else:
            payload["pairs"] = [("fixture-query", "fixture-document")]
        return payload

    def assert_expired(self, response: dict) -> None:
        self.assertFalse(response["ok"])
        self.assertIn("deadline", response["error"])
        self.assertEqual(self.server._inflight, 0)
        self.assertFalse(self.server._lock.held)
        self.assertFalse(self.server._fast_lock.held)

    def test_embed_expires_during_each_pre_inference_lock_wait(self) -> None:
        for acquisition in (1, 2, 3):
            with self.subTest(acquisition=acquisition):
                self.server._lock = FakeLock(self.clock, wait_at=acquisition)
                response = self.server.handle_request(self.request())
                self.assert_expired(response)
                self.assertEqual(self.loads, [])
                self.assertEqual(self.model.calls, 0)
                self.assertTrue(all(0 < value <= 1.0 for value in self.server._lock.timeouts))

    def test_rerank_expires_during_lock_wait(self) -> None:
        self.server._lock = FakeLock(self.clock, wait_at=1)
        self.assert_expired(self.server.handle_request(self.request("rerank")))
        self.assertEqual(self.loads, [])
        self.assertEqual(self.model.calls, 0)

    def test_timed_out_lock_is_not_released_or_followed_by_model_work(self) -> None:
        self.server._lock = FakeLock(self.clock, wait_at=1, time_out=True)
        self.assert_expired(self.server.handle_request(self.request()))
        self.assertEqual(self.loads, [])
        self.assertEqual(self.server._lock.timeouts, [1.0])

    def test_fast_lock_expiry_does_not_fall_back_to_main(self) -> None:
        self.server._lock = FakeLock(self.clock, contended=True)
        self.server._fast_lock = FakeLock(self.clock, wait_at=1)
        self.assert_expired(self.server.handle_request(self.request(single=True)))
        self.assertEqual(self.loads, [])
        self.assertEqual(self.server._lock.calls, 0)

    def test_fast_lock_timeout_does_not_release_an_unowned_lock(self) -> None:
        self.server._lock = FakeLock(self.clock, contended=True)
        self.server._fast_lock = FakeLock(self.clock, wait_at=1, time_out=True)
        self.assert_expired(self.server.handle_request(self.request(single=True)))
        self.assertEqual(self.loads, [])
        self.assertEqual(self.server._lock.calls, 0)
        self.assertEqual(self.server._fast_lock.timeouts, [1.0])

    def test_lock_waits_spend_one_budget_without_renewal(self) -> None:
        class ElapsingLock(FakeLock):
            def acquire(inner_self, blocking: bool = True, timeout: float = -1) -> bool:
                result = super().acquire(blocking, timeout)
                self.clock.advance(0.4)
                return result

        self.server._lock = ElapsingLock(self.clock)
        self.assert_expired(self.server.handle_request(self.request()))
        self.assertEqual(self.loads, [])
        self.assertEqual(len(self.server._lock.timeouts), 3)
        for actual, expected in zip(self.server._lock.timeouts, [1.0, 0.6, 0.2]):
            self.assertAlmostEqual(actual, expected)

    def test_expiry_while_waiting_to_activate_throttler_skips_activation(self) -> None:
        self.server._SUSTAINED_THRESHOLD = 1
        activations: list[bool] = []
        self.server._activate_throttler = lambda: activations.append(True)
        self.server._lock = FakeLock(self.clock, wait_at=2)
        self.assert_expired(self.server.handle_request(self.request()))
        self.assertEqual(activations, [])
        self.assertEqual(self.loads, [])

    def test_throttle_delay_cannot_start_model_loading(self) -> None:
        class Throttler:
            def before_batch(inner_self) -> tuple[int, dict]:
                self.clock.advance(2.0)
                return 1, {}

            def after_batch(inner_self, *_args: object) -> None:
                pass

            def should_flush_cache(inner_self) -> bool:
                return False

        self.server._throttler = Throttler()
        self.server._throttler_active = True
        self.assert_expired(self.server.handle_request(self.request()))
        self.assertEqual(self.loads, [])
        self.assertEqual(self.model.calls, 0)

    def test_model_loading_consumes_the_same_budget(self) -> None:
        for op, single, contended in (("embed", False, False), ("embed", True, False),
                                      ("embed", True, True), ("rerank", False, False)):
            with self.subTest(op=op, single=single, contended=contended):
                self.server._lock = FakeLock(self.clock, contended=contended)

                def slow_load(_identity: str | None) -> FakeModel:
                    self.clock.advance(2.0)
                    return self.model

                self.server._get_embed_model = slow_load
                self.server._get_fast_encoder = slow_load
                self.server._get_reranker = slow_load
                self.assert_expired(self.server.handle_request(self.request(op, single=single)))
                self.assertEqual(self.model.calls, 0)

    def test_expired_oom_records_fault_and_next_request_loads_cpu(self) -> None:
        for op, single in (("embed", False), ("embed", True), ("rerank", False)):
            with self.subTest(op=op, single=single):
                self.server = ms.ModelServer()
                self.server._lock = FakeLock(self.clock)
                self.server._fast_lock = FakeLock(self.clock)
                self.server._write_status_file = lambda: None
                failed_model = FakeModel(self.clock, oom=True, delay=2.0)
                cpu_model = FakeModel(self.clock)
                devices: list[str | None] = []
                expensive_recovery: list[str] = []

                def build(_identity: str, device: str | None) -> FakeModel:
                    self.assertTrue(self.server._lock.held)
                    devices.append(device)
                    return cpu_model

                failed_model.to = lambda device: expensive_recovery.append("move")
                self.server._embed_state = ms._EmbedState(
                    model=failed_model, tier="", model_id="qwen3_256",
                ) if op == "embed" else None
                self.server._resolve_embed_model_id = lambda tier: "qwen3_256"
                self.server._build_embed_model = build
                self.server._reranker = failed_model if op == "rerank" else None
                self.server._reranker_name = "fixture" if op == "rerank" else None
                mps = sys.modules["truememory.mps_utils"]
                mps.flush_mps_cache = lambda: expensive_recovery.append("flush")
                mps.auto_detect_device = lambda: "mps"
                mps.resolve_device = lambda device: device
                reranker_module = types.ModuleType("truememory.reranker")
                reranker_module.get_current_reranker_name = lambda: "fixture"
                transformers = types.ModuleType("sentence_transformers")
                transformers.CrossEncoder = build

                with patch.dict("sys.modules", {
                    "truememory.reranker": reranker_module,
                    "sentence_transformers": transformers,
                }):
                    self.assert_expired(self.server.handle_request(self.request(op, single=single)))
                    self.assertEqual(failed_model.calls, 1)
                    self.assertEqual(self.server._sticky_cpu, {op})
                    self.assertIsNone(self.server._embed_state)
                    self.assertIsNone(self.server._reranker)
                    self.assertIsNone(self.server._reranker_name)
                    self.assertEqual(devices, [])
                    self.assertEqual(expensive_recovery, [])

                    # A fresh budget reloads on CPU, then reuses that instance.
                    for _ in range(2):
                        response = self.server.handle_request(self.request(op, single=single))
                        self.assertTrue(response["ok"])
                    self.assertEqual(devices, ["cpu"])
                    self.assertEqual(failed_model.calls, 1)
                    self.assertEqual(cpu_model.calls, 2)
                    self.assertEqual(expensive_recovery, [])
                    if op == "embed":
                        self.assertIs(self.server._embed_state.model, cpu_model)
                    else:
                        self.assertIs(self.server._reranker, cpu_model)

    def test_unexpired_embed_recovery_keeps_cached_cpu_instance(self) -> None:
        for single in (False, True):
            with self.subTest(single=single):
                self.server = ms.ModelServer()
                self.server._lock = FakeLock(self.clock)
                self.server._fast_lock = FakeLock(self.clock)
                self.server._write_status_file = lambda: None
                model = FakeModel(self.clock, oom=True)
                movements: list[str] = []
                model.to = lambda device: movements.append(device)
                self.server._embed_state = ms._EmbedState(
                    model=model, tier="", model_id="qwen3_256",
                )
                self.assertTrue(self.server.handle_request(self.request(single=single))["ok"])
                self.assertTrue(self.server.handle_request(self.request(single=single))["ok"])
                self.assertEqual(self.server._sticky_cpu, {"embed"})
                self.assertEqual(movements, ["cpu"])
                self.assertIs(self.server._embed_state.model, model)
                self.assertEqual(model.calls, 3)

    def test_embed_cache_flush_expiry_skips_native_move_and_invalidates_cache(self) -> None:
        for single in (False, True):
            with self.subTest(single=single):
                self.server = ms.ModelServer()
                self.server._lock = FakeLock(self.clock)
                self.server._fast_lock = FakeLock(self.clock)
                self.server._write_status_file = lambda: None
                model = FakeModel(self.clock, oom=True)
                movements: list[str] = []
                model.to = lambda device: movements.append(device)
                self.server._embed_state = ms._EmbedState(
                    model=model, tier="", model_id="qwen3_256",
                )
                sys.modules["truememory.mps_utils"].flush_mps_cache = lambda: self.clock.advance(2.0)
                self.assert_expired(self.server.handle_request(self.request(single=single)))
                self.assertEqual(self.server._sticky_cpu, {"embed"})
                self.assertIsNone(self.server._embed_state)
                self.assertEqual(movements, [])
                self.assertEqual(model.calls, 1)

    def test_embed_recovery_consumes_budget_before_retry(self) -> None:
        for single in (False, True):
            with self.subTest(single=single):
                self.model = FakeModel(self.clock, oom=True)
                self.server._recover_embed_oom_locked = lambda model, deadline=None: self.clock.advance(2.0)
                self.assert_expired(self.server.handle_request(self.request(single=single)))
                self.assertEqual(self.model.calls, 1)

    def test_rerank_reload_consumes_budget_before_retry(self) -> None:
        self.model = FakeModel(self.clock, oom=True)
        cpu_model = FakeModel(self.clock)

        def load(_name: str | None) -> FakeModel:
            if self.recoveries:
                self.clock.advance(2.0)
                return cpu_model
            return self.model

        self.server._get_reranker = load
        self.assert_expired(self.server.handle_request(self.request("rerank")))
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(cpu_model.calls, 0)

    def test_rerank_cache_flush_expiry_prevents_cpu_model_load(self) -> None:
        self.model = FakeModel(self.clock, oom=True)
        sys.modules["truememory.mps_utils"].flush_mps_cache = lambda: self.clock.advance(2.0)
        self.assert_expired(self.server.handle_request(self.request("rerank")))
        self.assertEqual(self.loads, ["fixture"])
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.recoveries, ["rerank"])
        self.assertIsNone(self.server._reranker)
        self.assertIsNone(self.server._reranker_name)

    def test_wall_clock_rollback_cannot_extend_accepted_budget(self) -> None:
        def slow_load(_tier: str) -> FakeModel:
            self.clock.tick += 2.0
            self.clock.wall -= 100.0
            return self.model

        self.server._get_embed_model = slow_load
        self.assert_expired(self.server.handle_request(self.request()))
        self.assertEqual(self.model.calls, 0)

    def test_wall_clock_jump_cannot_expire_unspent_monotonic_budget(self) -> None:
        def load(_tier: str) -> FakeModel:
            self.clock.wall += 100.0
            return self.model

        self.server._get_embed_model = load
        self.assertTrue(self.server.handle_request(self.request())["ok"])
        self.assertEqual(self.model.calls, 1)

    def test_legacy_request_without_deadline_keeps_blocking_semantics(self) -> None:
        request = self.request()
        del request["deadline"]
        self.server._lock = FakeLock(self.clock, wait_at=1)
        self.assertTrue(self.server.handle_request(request)["ok"])
        self.assertEqual(self.model.calls, 1)
        self.assertTrue(all(value == -1 for value in self.server._lock.timeouts))

    def test_legacy_malformed_or_unbounded_deadlines_remain_compatible(self) -> None:
        for value in (None, "invalid", {}, float("nan"), float("inf")):
            with self.subTest(value=value):
                request = self.request(single=True)
                request["deadline"] = value
                self.assertTrue(self.server.handle_request(request)["ok"])

    def test_distant_valid_deadline_cannot_overflow_native_lock_timeout(self) -> None:
        request = self.request()
        request["deadline"] = 1e100
        self.assertTrue(self.server.handle_request(request)["ok"])
        self.assertTrue(all(0 < value <= ms.threading.TIMEOUT_MAX
                            for value in self.server._lock.timeouts))

    def test_fast_model_failure_before_expiry_still_uses_main_fallback(self) -> None:
        self.server._lock = FakeLock(self.clock, contended=True)

        def fail(_tier: str) -> FakeModel:
            raise ValueError("synthetic model failure")

        self.server._get_fast_encoder = fail
        self.assertTrue(self.server.handle_request(self.request(single=True))["ok"])
        self.assertEqual(self.model.calls, 1)

    def test_successful_requests_keep_response_fields(self) -> None:
        for op, key in (("embed", "vectors"), ("rerank", "scores")):
            with self.subTest(op=op):
                response = self.server.handle_request(self.request(op))
                self.assertEqual(set(response), {"ok", key})
                self.assertTrue(response["ok"])
