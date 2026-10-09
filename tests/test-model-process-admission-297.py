"""Current-RSS stage admission with synthetic models and stdlib-only sensors."""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import threading
import types
import unittest
import weakref
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "truememory" / "model_server.py"
SOURCE_TEXT: str | None = None
SUPPORT_PATH = Path(__file__).with_name("test-model-allocation-preflight-297.py")
SPEC = importlib.util.spec_from_file_location("process_admission_array_support", SUPPORT_PATH)
SUPPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPPORT)
BUDGET = 1048576


def load_server() -> types.ModuleType:
    tree = ast.parse(SOURCE_TEXT if SOURCE_TEXT is not None else SOURCE.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(alias.name == "numpy" for alias in node.names)
        or isinstance(node, ast.ImportFrom) and (node.module or "").startswith("truememory.")
        or isinstance(node, ast.Try) and any(
            isinstance(child, ast.Import) and any(alias.name == "psutil" for alias in child.names)
            for child in node.body
        )
        or isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "_set_mps_memory_cap"
    )]
    module = types.ModuleType("synthetic_process_admission_server")
    module.__dict__.update({
        "np": SUPPORT.Numpy(), "psutil": None, "_USE_UNIX": True, "_LOOPBACK_HOST": "127.0.0.1",
        "_env_int": lambda name, default, **kwargs: default, "pid_is_alive": lambda pid: False,
    })
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), module.__dict__)
    module.log.disabled = True
    return module


class Model:
    def __init__(self, owner: TestProcessAdmission, identity: str, device: str) -> None:
        self.owner, self.identity, self.device = owner, identity, device

    def encode(self, texts: list, **kwargs: object) -> list:
        self.owner.calls.append(("embed", self.device, list(texts), kwargs))
        self.owner.inference_hook(self)
        return [[float(text), 0.5] for text in texts]

    def predict(self, pairs: list, **kwargs: object) -> list:
        self.owner.calls.append(("rerank", self.device, list(pairs), kwargs))
        self.owner.inference_hook(self)
        return [float(pair[1]) for pair in pairs]

    def to(self, device: str) -> None:
        self.owner.assertTrue(self.owner.server._inference_lock.locked())
        self.owner.assertTrue(self.owner.server._lock.locked())
        self.owner.moves.append(device)
        self.device = device
        self.owner.transfer_hook()


class TestProcessAdmission(unittest.TestCase):
    def setUp(self) -> None:
        package = types.ModuleType("truememory")
        package.__path__ = []
        vector = types.ModuleType("truememory.vector_search")
        vector.EMBEDDING_MODEL = "model2vec"
        vector._TIER_ALIASES = {"edge": "model2vec", "base": "qwen3_256"}
        reranker = types.ModuleType("truememory.reranker")
        reranker.get_current_reranker_name = lambda: "rank-a"
        self.mps = types.ModuleType("truememory.mps_utils")
        self.mps.auto_detect_device = lambda: "mps"
        self.mps.resolve_device = lambda device: device or "mps"
        self.mps.ensure_mps_memory_budget = lambda device: None
        self.mps.is_mps_oom = lambda error: "MPS backend out of memory" in str(error)
        self.mps.flush_mps_cache = Mock()
        transformers = types.ModuleType("sentence_transformers")
        transformers.CrossEncoder = lambda identity, device: self.build(identity, device)
        self.modules = patch.dict(sys.modules, {
            "truememory": package, "truememory.vector_search": vector,
            "truememory.reranker": reranker, "truememory.mps_utils": self.mps,
            "sentence_transformers": transformers,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.environment = patch.dict(os.environ, {"TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB": "1"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.ms = load_server()
        self.server = self.ms.ModelServer()
        self.addCleanup(self.server._workers.shutdown, wait=True, cancel_futures=True)
        self.server._write_status_file = lambda: None
        self.server._build_embed_model = self.build
        self.rss: object = BUDGET - 1
        self.readings: list[object] = []
        self.measurement_owners: list[tuple[bool, bool]] = []
        self.sensor = Mock(side_effect=self.measure)
        self.ms.psutil = types.SimpleNamespace(Process=lambda: types.SimpleNamespace(memory_info=self.sensor))
        self.builds: list[tuple[str, str]] = []
        self.calls: list[tuple] = []
        self.moves: list[str] = []
        self.refs: list[weakref.ReferenceType] = []
        self.build_hook: Callable[[Model], None] = lambda model: None
        self.inference_hook: Callable[[Model], None] = lambda model: None
        self.transfer_hook: Callable[[], None] = lambda: None

    def tearDown(self) -> None:
        self.assertFalse(self.server._lock.locked())
        self.assertFalse(self.server._inference_lock.locked())
        self.assertFalse(self.server._fast_lock.locked())
        self.assertEqual(self.server._inflight, 0)
        self.assertTrue(all(owned and not metadata for owned, metadata in self.measurement_owners))

    def measure(self) -> types.SimpleNamespace:
        self.measurement_owners.append((
            self.server._inference_lock.locked() or self.server._fast_lock.locked(),
            self.server._residency_lock.locked(),
        ))
        self.readings.append(self.rss)
        return types.SimpleNamespace(rss=self.rss)

    def build(self, identity: str, device: str) -> Model:
        self.assertTrue(self.server._inference_lock.locked() or self.server._fast_lock.locked())
        self.builds.append((identity, device))
        model = Model(self, identity, device)
        self.refs.append(weakref.ref(model))
        self.build_hook(model)
        return model

    def request(self, op: str = "embed", count: int = 3, **options: object) -> dict:
        request = {
            "op": op, "tier": "edge", "model_name": "rank-a", "batch_size": 2,
            "texts": [str(index) for index in range(count)],
            "pairs": [("query", str(index)) for index in range(count)],
        }
        request.update(options)
        return self.server.handle_request(request)

    def assert_busy(self, response: dict) -> None:
        self.assertFalse(response["ok"], response)
        self.assertEqual(response["error_code"], "server_busy")
        self.assertEqual(response["retry_after_ms"], 250)
        self.assertNotIn("vectors", response)
        self.assertNotIn("scores", response)

    def clear_models(self) -> None:
        self.server._embed_state = None
        self.server._reranker = None
        self.server._reranker_name = None
        self.server._fast_encoder = None
        self.server._fast_model_id = None
        self.server._fast_generation = None
        self.builds.clear()
        self.calls.clear()
        self.server._embed_timestamps.clear()

    def test_setting_uses_mib_and_rejects_malformed_values(self) -> None:
        self.assertEqual(self.server._max_rss_bytes, 1 * 1048576)
        for raw in ("-1", "1.0", "nan", "inf", "true", "", " 1", "١", str(2**31)):
            with self.subTest(raw=raw), patch.dict(os.environ, {"TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB": raw}):
                with self.assertRaises(ValueError):
                    self.ms.ModelServer()

    def test_cold_constructor_boundary_all_main_paths(self) -> None:
        for op, count in (("embed", 1), ("embed_batched", 3), ("rerank", 3), ("rerank_batched", 3)):
            for usage in (BUDGET - 1, BUDGET, BUDGET + 1):
                with self.subTest(op=op, count=count, usage=usage):
                    self.clear_models()
                    self.rss = usage
                    response = self.request(op, count)
                    if usage < BUDGET:
                        self.assertTrue(response["ok"])
                        self.assertEqual(len(self.builds), 1)
                    else:
                        self.assert_busy(response)
                        self.assertEqual(self.builds, [])
                        self.assertEqual(self.calls, [])

    def test_cached_slice_boundary_all_main_paths(self) -> None:
        for op, count in (("embed", 1), ("embed", 3), ("rerank", 3)):
            with self.subTest(op=op, count=count):
                self.clear_models()
                self.rss = BUDGET - 1
                self.assertTrue(self.request(op, count)["ok"])
                built = len(self.builds)
                self.calls.clear()
                for usage in (BUDGET, BUDGET + 1):
                    self.rss = usage
                    self.assert_busy(self.request(op, count))
                    self.assertEqual(self.calls, [])
                    self.assertEqual(len(self.builds), built)

    def test_invalid_and_missing_measurements_refuse_before_construction(self) -> None:
        for invalid in (None, True, False, 0, -1, "1048575", 1.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                # Each sensor case is independent of sustained-workload policy.
                self.server._embed_timestamps.clear()
                self.rss = invalid
                self.assert_busy(self.request())
                self.assertEqual(self.builds, [])
        for error in (OSError("synthetic sensor failure"), RuntimeError("synthetic sensor failure")):
            with self.subTest(error=type(error).__name__):
                self.server._embed_timestamps.clear()
                self.sensor.side_effect = error
                self.assert_busy(self.request())
                self.assertEqual(self.builds, [])
        self.ms.psutil = None
        self.assert_busy(self.request())
        self.assertEqual(self.builds, [])

    def test_constructor_crossing_refuses_before_publication_or_inference(self) -> None:
        for op, count in (("embed", 1), ("embed", 3), ("rerank", 3)):
            with self.subTest(op=op, count=count):
                self.clear_models()
                self.rss = BUDGET - 1
                self.build_hook = lambda model: setattr(self, "rss", BUDGET)
                self.assert_busy(self.request(op, count))
                self.assertEqual(len(self.builds), 1)
                self.assertEqual(self.calls, [])
                self.assertIsNone(self.server._embed_state)
                self.assertIsNone(self.server._reranker)
                self.assertIsNone(self.refs[-1]())

    def test_slice_crossing_never_returns_partial_output(self) -> None:
        for op in ("embed", "rerank"):
            with self.subTest(op=op):
                self.clear_models()
                self.rss = BUDGET - 1
                self.inference_hook = lambda model: setattr(self, "rss", BUDGET)
                self.assert_busy(self.request(op, count=5))
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(len(self.calls[0][2]), 2)

    def test_fast_constructor_and_cache_refusal_never_fall_through_to_main(self) -> None:
        for cached in (False, True):
            with self.subTest(cached=cached):
                self.clear_models()
                self.rss = BUDGET - 1
                self.assertTrue(self.request()["ok"])
                with self.server._inference_lock:
                    if cached:
                        self.assertTrue(self.request(count=1)["ok"])
                    self.rss = BUDGET
                    before_builds, before_calls = len(self.builds), len(self.calls)
                    with patch.object(self.server, "_request_batch_limit", side_effect=AssertionError("Main fallback")):
                        self.assert_busy(self.request(count=1))
                    self.assertEqual(len(self.builds), before_builds)
                    self.assertEqual(len(self.calls), before_calls)

    def test_fast_constructor_crossing_drops_unpublished_clone(self) -> None:
        self.assertTrue(self.request()["ok"])
        self.build_hook = lambda model: setattr(self, "rss", BUDGET)
        with self.server._inference_lock:
            with patch.object(self.server, "_request_batch_limit", side_effect=AssertionError("Main fallback")):
                self.assert_busy(self.request(count=1))
        self.assertIsNone(self.server._fast_encoder)
        self.assertIsNone(self.refs[-1]())
        self.assertEqual(len(self.calls), 2)

    def fail_mps(self, model: Model) -> None:
        if model.device == "mps":
            raise RuntimeError("MPS backend out of memory")

    def test_embed_refused_transfer_is_sticky_and_invalidates_failed_cache(self) -> None:
        for count in (1, 3):
            with self.subTest(count=count):
                self.clear_models()
                self.server._sticky_cpu.clear()
                self.rss = BUDGET - 1

                def fail(model: Model) -> None:
                    if model.device == "mps":
                        self.rss = BUDGET
                        self.fail_mps(model)

                self.inference_hook = fail
                self.assert_busy(self.request(count=count))
                self.assertIn("embed", self.server._sticky_cpu)
                self.assertIsNone(self.server._embed_state)
                self.assertIsNone(self.refs[-1]())
                self.assertEqual(self.moves, [])
                self.rss = BUDGET - 1
                self.assertTrue(self.request(count=count)["ok"])
                self.assertEqual(self.builds[-1], ("model2vec", "cpu"))

    def test_embed_transfer_crossing_refuses_cpu_retry(self) -> None:
        for count in (1, 3):
            with self.subTest(count=count):
                self.clear_models()
                self.server._sticky_cpu.clear()
                self.moves.clear()
                self.rss = BUDGET - 1
                self.inference_hook = self.fail_mps
                self.transfer_hook = lambda: setattr(self, "rss", BUDGET)
                self.assert_busy(self.request(count=count))
                self.assertIn("embed", self.server._sticky_cpu)
                self.assertEqual(self.moves, ["cpu"])
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(self.server._embed_state.model.device, "cpu")

    def test_rerank_refused_replacement_is_sticky_and_releases_failed_model(self) -> None:
        def fail(model: Model) -> None:
            if model.device == "mps":
                self.rss = BUDGET
                self.fail_mps(model)

        self.inference_hook = fail
        self.assert_busy(self.request("rerank"))
        self.assertIn("rerank", self.server._sticky_cpu)
        self.assertEqual(len(self.builds), 1)
        self.assertIsNone(self.server._reranker)
        self.assertIsNone(self.server._reranker_name)
        self.assertIsNone(self.refs[-1]())
        self.rss = BUDGET - 1
        self.assertTrue(self.request("rerank")["ok"])
        self.assertEqual(self.builds[-1], ("rank-a", "cpu"))

    def test_rerank_cpu_constructor_crossing_refuses_retry(self) -> None:
        self.inference_hook = self.fail_mps

        def constructed(model: Model) -> None:
            if model.device == "cpu":
                self.rss = BUDGET

        self.build_hook = constructed
        self.assert_busy(self.request("rerank"))
        self.assertEqual(self.builds, [("rank-a", "mps"), ("rank-a", "cpu")])
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(self.server._reranker)
        self.assertTrue(all(ref() is None for ref in self.refs))

    def test_rerank_retry_samples_after_constructor_admission(self) -> None:
        self.inference_hook = self.fail_mps
        original = self.server._get_reranker

        def loaded(name: str, deadline: object = None) -> Model:
            model = original(name, deadline=deadline)
            if model.device == "cpu":
                self.rss = BUDGET
            return model

        self.server._get_reranker = loaded
        self.assert_busy(self.request("rerank"))
        self.assertEqual(len(self.builds), 2)
        self.assertEqual(len(self.calls), 1)

    def test_admitted_recovery_retains_complete_order_and_batch_controls(self) -> None:
        for op in ("embed", "rerank"):
            with self.subTest(op=op):
                self.clear_models()
                self.server._sticky_cpu.clear()
                self.inference_hook = self.fail_mps
                response = self.request(op, count=5)
                self.assertTrue(response["ok"])
                self.assertEqual([len(call[2]) for call in self.calls], [2, 2, 2, 1])
                self.assertTrue(all(call[3] == {"batch_size": 2, "show_progress_bar": False}
                                    for call in self.calls))
                actual = response["vectors" if op == "embed" else "scores"]
                self.assertEqual(actual.dtype, "float32")
                self.assertEqual(actual.data, [0.0, 0.5, 1.0, 0.5, 2.0, 0.5, 3.0, 0.5, 4.0, 0.5]
                                 if op == "embed" else [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_current_usage_can_fall_after_historical_peak(self) -> None:
        resource = types.ModuleType("resource")
        resource.getrusage = Mock(return_value=types.SimpleNamespace(ru_maxrss=10 * BUDGET))
        with patch.dict(sys.modules, {"resource": resource}):
            self.rss = BUDGET + 1
            self.assert_busy(self.request())
            self.rss = BUDGET - 1
            self.assertTrue(self.request()["ok"])
        resource.getrusage.assert_not_called()
        self.assertIn(BUDGET + 1, self.readings)
        self.assertIn(BUDGET - 1, self.readings)

    def test_disabled_budget_performs_zero_sensor_reads_including_recovery(self) -> None:
        self.server._max_rss_bytes = 0
        self.ms.psutil = types.SimpleNamespace(Process=Mock(side_effect=AssertionError("Disabled sensor")))
        self.inference_hook = self.fail_mps
        self.assertTrue(self.request()["ok"])
        self.assertTrue(self.request("rerank")["ok"])
        with self.server._inference_lock:
            self.assertTrue(self.request(count=1)["ok"])
        self.assertEqual(self.readings, [])
        self.ms.psutil.Process.assert_not_called()

    def test_default_budget_is_disabled(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB", None)
            server = self.ms.ModelServer()
        try:
            self.assertEqual(server._max_rss_bytes, 0)
        finally:
            server._workers.shutdown(wait=True, cancel_futures=True)

    def test_replacement_releases_obsolete_model_before_measuring(self) -> None:
        self.assertTrue(self.request()["ok"])
        old = self.refs[-1]

        def measurement() -> types.SimpleNamespace:
            self.assertIsNone(old(), "Old residency retained at admission")
            return types.SimpleNamespace(rss=BUDGET - 1)

        self.sensor.side_effect = measurement
        self.assertTrue(self.request(tier="base")["ok"])

    def test_expired_request_never_samples_or_loads(self) -> None:
        response = self.request(deadline=1)
        self.assertFalse(response["ok"])
        self.assertIn("deadline", response["error"])
        self.assertEqual(self.readings, [])
        self.assertEqual(self.builds, [])

    def test_sensor_time_expiry_prevents_all_five_constructor_routes(self) -> None:
        routes = (("embed", 1, "cold"), ("embed", 3, "cold"), ("rerank", 3, "cold"),
                  ("embed", 1, "fast"), ("rerank", 3, "recovery"))
        for op, count, route in routes:
            for outcome in ("under", "equal", "over", "invalid", "error"):
                with self.subTest(op=op, count=count, route=route, outcome=outcome):
                    self.clear_models()
                    self.server._sticky_cpu.clear()
                    self.rss = BUDGET - 1
                    self.sensor.side_effect = self.measure
                    self.inference_hook = self.fail_mps if route == "recovery" else lambda model: None
                    if route == "fast":
                        self.assertTrue(self.request()["ok"])
                    self.calls.clear()
                    self.builds.clear()
                    tick, expired_samples = [1000.0], []
                    clock = types.SimpleNamespace(time=lambda: tick[0], monotonic=lambda: tick[0])

                    def measure() -> types.SimpleNamespace:
                        value = self.measure()
                        if route != "recovery" or "rerank" in self.server._sticky_cpu:
                            expired_samples.append(value)
                            tick[0] = 1002.0
                            if outcome == "error":
                                raise OSError("synthetic expired RSS read")
                            value.rss = {"under": BUDGET - 1, "equal": BUDGET,
                                         "over": BUDGET + 1, "invalid": None}[outcome]
                        return value

                    self.sensor.side_effect = measure
                    with patch.object(self.ms, "time", clock):
                        if route == "fast":
                            with self.server._inference_lock, patch.object(
                                self.server, "_request_batch_limit", side_effect=AssertionError("Main fallback"),
                            ):
                                response = self.request(op, count, deadline=1001.0)
                        else:
                            response = self.request(op, count, deadline=1001.0)
                    self.assertFalse(response["ok"])
                    self.assertIn("deadline", response["error"])
                    self.assertNotIn("error_code", response)
                    self.assertEqual(len(expired_samples), 1)
                    self.assertEqual(self.builds, [("rank-a", "mps")] if route == "recovery" else [])
                    self.assertEqual(len(self.calls), 1 if route == "recovery" else 0)
                    if route == "recovery":
                        self.assertIn("rerank", self.server._sticky_cpu)
                        self.assertIsNone(self.server._reranker)
                        self.assertIsNone(self.server._reranker_name)
                        self.assertIsNone(self.refs[-1]())
                    if route == "fast":
                        self.assertIsNone(self.server._fast_encoder)
                    self.assertFalse(self.server._lock.locked())
                    self.assertFalse(self.server._inference_lock.locked())
                    self.assertFalse(self.server._fast_lock.locked())
                    self.assertEqual(self.server._inflight, 0)

    def test_sensor_expiry_after_construction_cannot_publish_or_infer(self) -> None:
        for op, count, fast in (("embed", 1, False), ("embed", 3, False),
                                ("rerank", 3, False), ("embed", 1, True)):
            with self.subTest(op=op, count=count, fast=fast):
                self.clear_models()
                self.sensor.side_effect = self.measure
                if fast:
                    self.assertTrue(self.request()["ok"])
                self.calls.clear()
                self.builds.clear()
                tick = [1000.0]
                clock = types.SimpleNamespace(time=lambda: tick[0], monotonic=lambda: tick[0])

                def measure() -> types.SimpleNamespace:
                    value = self.measure()
                    if self.builds:
                        tick[0] = 1002.0
                    return value

                self.sensor.side_effect = measure
                with patch.object(self.ms, "time", clock):
                    if fast:
                        with self.server._inference_lock:
                            response = self.request(op, count, deadline=1001.0)
                    else:
                        response = self.request(op, count, deadline=1001.0)
                self.assertIn("deadline", response["error"])
                self.assertEqual(len(self.builds), 1)
                self.assertEqual(self.calls, [])
                self.assertIsNone(self.refs[-1]())
                self.assertIsNone(self.server._fast_encoder)
                if not fast:
                    self.assertIsNone(self.server._embed_state)
                    self.assertIsNone(self.server._reranker)

    def test_embed_transfer_sensor_expiry_invalidates_failed_accelerator(self) -> None:
        for count in (1, 3):
            with self.subTest(count=count):
                self.clear_models()
                self.server._sticky_cpu.clear()
                tick = [1000.0]
                clock = types.SimpleNamespace(time=lambda: tick[0], monotonic=lambda: tick[0])

                def measure() -> types.SimpleNamespace:
                    value = self.measure()
                    if "embed" in self.server._sticky_cpu:
                        tick[0] = 1002.0
                    return value

                self.sensor.side_effect = measure
                self.inference_hook = self.fail_mps
                with patch.object(self.ms, "time", clock):
                    response = self.request(count=count, deadline=1001.0)
                self.assertIn("deadline", response["error"])
                self.assertIn("embed", self.server._sticky_cpu)
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(self.moves, [])
                self.assertIsNone(self.server._embed_state)
                self.assertIsNone(self.refs[-1]())

    def test_disabled_admission_retains_deadline_checks_without_sensor_reads(self) -> None:
        self.server._max_rss_bytes = 0
        self.ms.psutil = types.SimpleNamespace(Process=Mock(side_effect=AssertionError("Disabled sensor")))
        for op, count in (("embed", 1), ("embed", 3), ("rerank", 3)):
            with self.subTest(op=op, count=count):
                self.clear_models()
                tick = [1000.0]
                clock = types.SimpleNamespace(time=lambda: tick[0], monotonic=lambda: tick[0])

                def expire_device(device: str | None) -> str:
                    tick[0] = 1002.0
                    return "mps"

                with patch.object(self.ms, "time", clock), patch.object(self.mps, "resolve_device", expire_device):
                    response = self.request(op, count, deadline=1001.0)
                self.assertIn("deadline", response["error"])
                self.assertEqual(self.builds, [])
                self.assertEqual(self.calls, [])
        self.ms.psutil.Process.assert_not_called()
        self.assertEqual(self.readings, [])

    def test_invalid_sensor_cases_do_not_activate_unrelated_throttler(self) -> None:
        with patch.object(self.server, "_activate_throttler", side_effect=AssertionError("Unrelated throttler")):
            self.test_invalid_and_missing_measurements_refuse_before_construction()

    def assert_failed_transfer(self, failure: BaseException, count: int) -> None:
        responses = []
        try:
            responses.append(self.request(count=count, tier="base"))
        except BaseException as caught:
            self.assertIs(caught, failure)
        else:
            self.fail("A failed transfer returned a response")
        self.assertEqual(responses, [])
        self.assertIn("embed", self.server._sticky_cpu)
        self.assertEqual(self.moves, ["cpu"])
        self.assertFalse(self.server._lock.locked())
        self.assertFalse(self.server._inference_lock.locked())

    def test_failed_cpu_transfer_drops_cache_and_next_request_rebuilds_same_identity(self) -> None:
        for count, fail_at in ((1, 1), (5, 2)):
            for error_type in (RuntimeError, MemoryError, ValueError, KeyboardInterrupt, SystemExit):
                with self.subTest(count=count, error=error_type.__name__):
                    self.clear_models()
                    self.server._sticky_cpu.clear()
                    self.moves.clear()
                    failure = error_type("synthetic failed CPU transfer")
                    workspaces = []

                    def fail_inference(model: Model) -> None:
                        if len(self.calls) == fail_at:
                            self.fail_mps(model)

                    def fail_transfer(model: Model, device: str) -> None:
                        self.assertEqual(model.identity, "qwen3_256")
                        self.assertEqual(model.device, "mps")
                        self.assertTrue(self.server._inference_lock.locked())
                        self.assertTrue(self.server._lock.locked())
                        self.moves.append(device)
                        workspace = SUPPORT.Workspace()
                        workspaces.append(weakref.ref(workspace))
                        raise failure

                    self.inference_hook = fail_inference
                    with patch.object(Model, "to", fail_transfer):
                        self.assert_failed_transfer(failure, count)
                    self.assertIsNone(self.server._embed_state)
                    self.assertEqual(len(self.calls), fail_at)
                    self.assertEqual(self.builds, [("qwen3_256", "mps")])
                    self.assertEqual(self.server._inflight, 0)
                    self.assertFalse(self.server._fast_lock.locked())
                    self.assertIsNotNone(workspaces[0]())
                    self.assertIsNotNone(self.refs[-1]())
                    failure.__traceback__ = None
                    self.assertIsNone(workspaces[0]())
                    self.assertIsNone(self.refs[-1]())

                    self.inference_hook = self.fail_mps
                    response = self.request(count=count, tier="base")
                    self.assertTrue(response["ok"])
                    self.assertEqual(self.builds, [("qwen3_256", "mps"), ("qwen3_256", "cpu")])
                    self.assertEqual(self.moves, ["cpu"])
                    self.assertEqual(response["vectors"].data,
                                     [value for index in range(count) for value in (float(index), 0.5)])
                    self.assertEqual(response["vectors"].dtype, "float32")

    def test_failed_transfer_preserves_an_unrelated_replacement_snapshot(self) -> None:
        for count in (1, 3):
            with self.subTest(count=count):
                self.clear_models()
                self.server._sticky_cpu.clear()
                self.moves.clear()
                failure = ValueError("synthetic transfer failure after replacement")
                replacement = Model(self, "qwen3_256", "cpu")
                replacement_state = self.ms._EmbedState(replacement, "base", "qwen3_256")

                def fail_transfer(model: Model, device: str) -> None:
                    self.moves.append(device)
                    self.server._publish_embed_state(replacement_state)
                    raise failure

                self.inference_hook = self.fail_mps
                with patch.object(Model, "to", fail_transfer):
                    self.assert_failed_transfer(failure, count)
                self.assertIs(self.server._embed_state, replacement_state)
                failed = self.refs[-1]
                failure.__traceback__ = None
                self.assertIsNone(failed())
                self.assertTrue(self.request(count=count, tier="base")["ok"])
                self.assertEqual(self.builds, [("qwen3_256", "mps")])
                self.assertIs(self.server._embed_state.model, replacement)

    def test_failed_transfer_retires_idle_fast_clone(self) -> None:
        self.assertTrue(self.request(tier="base")["ok"])
        main = self.refs[-1]
        with self.server._inference_lock:
            self.assertTrue(self.request(count=1, tier="base")["ok"])
        clone = self.refs[-1]
        self.assertEqual(self.builds, [("qwen3_256", "mps"), ("qwen3_256", "cpu")])
        failure = MemoryError("synthetic transfer allocation failure")

        def fail_transfer(model: Model, device: str) -> None:
            self.moves.append(device)
            raise failure

        self.inference_hook = self.fail_mps
        with patch.object(Model, "to", fail_transfer):
            self.assert_failed_transfer(failure, 3)
        self.assertIsNone(self.server._embed_state)
        self.assertIsNone(self.server._fast_encoder)
        self.assertIsNone(self.server._fast_model_id)
        self.assertIsNone(self.server._fast_generation)
        self.assertIsNone(clone())
        failure.__traceback__ = None
        self.assertIsNone(main())

    def test_failed_transfer_allows_active_fast_work_to_finish_without_stale_publication(self) -> None:
        for phase in ("construction", "inference"):
            with self.subTest(phase=phase):
                self.clear_models()
                self.server._sticky_cpu.clear()
                self.moves.clear()
                self.inference_hook = lambda model: None
                self.build_hook = lambda model: None
                self.assertTrue(self.request(tier="base")["ok"])
                main = self.refs[-1]
                entered, release = threading.Event(), threading.Event()
                outcomes, errors = [], []
                failure = ValueError("synthetic transfer failure beside active fast work")

                def wait_for_release() -> None:
                    entered.set()
                    self.assertTrue(release.wait(2), "Synthetic fast work was not released")

                def on_build(model: Model) -> None:
                    if model.device == "cpu" and phase == "construction":
                        wait_for_release()

                def on_inference(model: Model) -> None:
                    if model.device == "cpu" and phase == "inference":
                        wait_for_release()
                    elif model.device == "mps":
                        self.fail_mps(model)

                def run_fast() -> None:
                    try:
                        outcomes.append(self.request(count=1, tier="base"))
                    except BaseException as error:
                        errors.append(error)

                def fail_transfer(model: Model, device: str) -> None:
                    self.moves.append(device)
                    raise failure

                self.build_hook = on_build
                self.inference_hook = on_inference
                worker = threading.Thread(target=run_fast, daemon=True)
                try:
                    with self.server._inference_lock:
                        worker.start()
                        self.assertTrue(entered.wait(1), "Fast owner never entered its native stage")
                    clone = self.refs[-1]
                    with patch.object(Model, "to", fail_transfer):
                        self.assert_failed_transfer(failure, 3)
                    self.assertIsNone(self.server._embed_state)
                    self.assertTrue(self.server._fast_lock.locked())
                    self.assertTrue(worker.is_alive())
                    failure.__traceback__ = None
                    self.assertIsNone(main())
                    self.assertIsNotNone(clone())
                finally:
                    release.set()
                    if worker.ident is not None:
                        worker.join(2)
                    self.assertFalse(worker.is_alive(), "Fast owner remained blocked")
                self.assertEqual(errors, [])
                self.assertEqual(len(outcomes), 1)
                self.assertTrue(outcomes[0]["ok"])
                self.assertEqual(outcomes[0]["vectors"].data, [0.0, 0.5])
                self.assertIsNone(self.server._fast_encoder)
                self.assertIsNone(self.server._fast_model_id)
                self.assertIsNone(self.server._fast_generation)
                self.assertIsNone(clone())


if __name__ == "__main__":
    unittest.main()
