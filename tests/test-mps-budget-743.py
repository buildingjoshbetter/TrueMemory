"""Real control flow with synthetic allocator APIs; no torch or model imports."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from collections.abc import Iterator
import dataclasses
import logging
import re
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from packaging.requirements import Requirement


SOURCE = Path(__file__).resolve().parents[1] / "truememory"
GIB = 1024**3
HIGH = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
LOW = "PYTORCH_MPS_LOW_WATERMARK_RATIO"


def source_module(path: str, dependencies: dict | None = None,
                  functions: tuple[str, ...] | None = None) -> types.ModuleType:
    tree = ast.parse((SOURCE / path).read_text(encoding="utf-8"))
    if functions is not None:
        tree.body = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in functions]
    module = types.ModuleType("synthetic_743_" + path.replace("/", "_").replace(".", "_"))
    module.__dict__.update(dependencies or {})
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(ast.fix_missing_locations(tree), path, "exec"), module.__dict__)
    return module


class BudgetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fresh()

    def fresh(self, env: dict[str, str] | None = None) -> None:
        self.events: list[object] = []
        self.env = dict(env or {})
        self.budget = source_module("mps_utils.py")
        self.budget.os = types.SimpleNamespace(environ=self.env)
        self.recommended = Mock(side_effect=self.recommended_bytes)
        self.setter = Mock(side_effect=lambda ratio: self.events.append(("set", ratio)))
        self.physical = Mock(return_value=types.SimpleNamespace(total=32 * GIB))
        self.torch = types.ModuleType("torch")
        self.torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        self.torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
        self.torch.mps = types.SimpleNamespace(
            recommended_max_memory=self.recommended,
            set_per_process_memory_fraction=self.setter,
            driver_allocated_memory=Mock(return_value=0),
        )
        self.package = types.ModuleType("truememory")
        self.package.__path__ = []
        self.package.mps_utils = self.budget
        modules = {
            "torch": self.torch,
            "psutil": types.SimpleNamespace(virtual_memory=self.physical),
            "truememory": self.package,
            "truememory.mps_utils": self.budget,
        }
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def recommended_bytes(self) -> int:
        self.assertIn(LOW, self.env, "LOW must be bootstrapped before allocator initialization")
        self.events.append("recommended")
        return 20 * GIB


class TestMPSBudget(BudgetTestCase):
    def test_preserved_intended_byte_curve(self) -> None:
        for physical, expected in ((8, 1.52), (12, 2.28), (16, 1.5),
                                   (18, 1.5), (24, 1.92), (32, 2.5), (64, 2.5)):
            with self.subTest(physical=physical):
                self.assertEqual(self.budget._default_mps_budget_bytes(physical * GIB), int(expected * GIB))

    def test_differing_recommended_sizes_enforce_the_same_bytes(self) -> None:
        for recommended, fraction in ((20, 0.125), (25, 0.1)):
            with self.subTest(recommended=recommended):
                self.fresh()
                def query() -> int:
                    self.recommended_bytes()
                    return recommended * GIB
                self.recommended.side_effect = query
                policy = self.budget.ensure_mps_memory_budget("mps")
                self.assertEqual(policy.intended_bytes, int(2.5 * GIB))
                self.assertEqual(policy.effective_bytes, int(2.5 * GIB))
                self.assertEqual(policy.recommended_bytes, recommended * GIB)
                self.assertEqual(policy.fraction, fraction)
                self.assertEqual((policy.source, policy.enforcement), ("automatic", "enforced"))
                self.assertEqual(self.events, ["recommended", ("set", fraction)])
                self.assertEqual(self.env, {LOW: "0.0"})
                self.setter.assert_called_once_with(float(fraction))
                self.assertIs(self.budget.get_mps_memory_budget(), policy)
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    policy.effective_bytes = 1

    def test_explicit_operator_finite_and_unlimited_ratios_are_preserved(self) -> None:
        for high, low in (("0.2", "0.1"), ("2", "2"), ("0", "1.4"), ("0", "0")):
            with self.subTest(high=high, low=low):
                self.fresh({HIGH: high, LOW: low})
                policy = self.budget.ensure_mps_memory_budget("mps")
                expected = int(float(high) * 20 * GIB) if float(high) else None
                self.assertEqual(policy.intended_bytes, expected)
                self.assertEqual(policy.effective_bytes, expected)
                self.assertEqual(policy.source, "operator")
                self.assertEqual(policy.enforcement, "enforced" if expected else "disabled")
                self.assertEqual(policy.requested_low_watermark_ratio, float(low))
                self.assertEqual(self.env, {HIGH: high, LOW: low})
                self.physical.assert_not_called()
                self.setter.assert_called_once_with(float(high))

    def test_invalid_operator_values_stop_before_query_or_success_state(self) -> None:
        for key in (HIGH, LOW):
            for invalid in ("", "nan", "NaN", "inf", "-inf", "-0.1", "2.01", "synthetic-invalid"):
                with self.subTest(key=key, value=invalid):
                    self.fresh({key: invalid})
                    with self.assertRaisesRegex(self.budget.MPSBudgetError, "finite ratio"):
                        self.budget.ensure_mps_memory_budget("mps")
                    self.recommended.assert_not_called()
                    self.setter.assert_not_called()
                    self.assertIsNone(self.budget.get_mps_memory_budget())

    def test_low_high_conflicts_fail_before_model_allocation(self) -> None:
        for env, expected_queries in (({HIGH: "0.1", LOW: "0.2"}, 0),
                                      ({LOW: "1.8"}, 0), ({LOW: "0.2"}, 1)):
            with self.subTest(env=env):
                self.fresh(env)
                with self.assertRaisesRegex(self.budget.MPSBudgetError, "low watermark exceeds"):
                    self.budget.ensure_mps_memory_budget("mps")
                self.assertEqual(self.recommended.call_count, expected_queries)
                self.setter.assert_not_called()
                self.assertIsNone(self.budget.get_mps_memory_budget())

    def test_missing_public_apis_or_invalid_metrics_never_publish_success(self) -> None:
        for api in ("recommended_max_memory", "set_per_process_memory_fraction"):
            with self.subTest(api=api):
                self.fresh()
                delattr(self.torch.mps, api)
                with self.assertRaisesRegex(self.budget.MPSBudgetError, "public MPS") as raised:
                    self.budget.ensure_mps_memory_budget("mps")
                self.assertIn("PyTorch>=2.5", str(raised.exception))
                self.assertIn("compatible build if available", str(raised.exception))
                self.assertIn("TRUEMEMORY_DEVICE=cpu", str(raised.exception))
                self.assertIn("restart", str(raised.exception))
                self.assertIsNone(self.budget.get_mps_memory_budget())
        for metric in ("physical", "recommended"):
            for invalid in (0, -1, True, 4.5, float("nan"), None):
                with self.subTest(metric=metric, value=invalid):
                    self.fresh()
                    if metric == "physical":
                        self.physical.return_value = types.SimpleNamespace(total=invalid)
                    else:
                        self.recommended.side_effect = None
                        self.recommended.return_value = invalid
                    with self.assertRaisesRegex(self.budget.MPSBudgetError, "positive integer byte count"):
                        self.budget.ensure_mps_memory_budget("mps")
                    self.setter.assert_not_called()
                    self.assertIsNone(self.budget.get_mps_memory_budget())

    def test_failed_query_or_setter_is_terminal_without_false_success(self) -> None:
        for failing in ("query", "setter"):
            with self.subTest(failing=failing):
                self.fresh()
                target = self.recommended if failing == "query" else self.setter
                target.side_effect = RuntimeError("synthetic allocator fault")
                for _ in range(2):
                    with self.assertRaisesRegex(self.budget.MPSBudgetError, "restart this process"):
                        self.budget.ensure_mps_memory_budget("mps")
                    self.assertIsNone(self.budget.get_mps_memory_budget())
                self.assertEqual(target.call_count, 1)
                self.assertIsNone(self.budget.ensure_mps_memory_budget("cpu"))

    def test_unsupported_derived_fraction_is_not_silently_clamped(self) -> None:
        self.recommended.side_effect = None
        self.recommended.return_value = GIB
        with self.assertRaisesRegex(self.budget.MPSBudgetError, "supported ratio range"):
            self.budget.ensure_mps_memory_budget("mps")
        self.setter.assert_not_called()
        self.assertIsNone(self.budget.get_mps_memory_budget())

    def test_non_mps_selection_does_not_touch_allocator_or_environment(self) -> None:
        for device in ("cpu", "cuda", "cuda:0", "xpu"):
            with self.subTest(device=device):
                self.assertIsNone(self.budget.ensure_mps_memory_budget(device))
        self.torch.cuda.is_available = lambda: True
        self.assertIsNone(self.budget.ensure_mps_memory_budget(None))
        self.recommended.assert_not_called()
        self.setter.assert_not_called()
        self.physical.assert_not_called()
        self.assertEqual(self.env, {})
        self.torch.cuda.is_available = lambda: False
        self.assertIsNotNone(self.budget.ensure_mps_memory_budget(None))

    def test_concurrent_initialization_publishes_one_complete_snapshot(self) -> None:
        entered, release = threading.Event(), threading.Event()
        results, errors = [], []
        def setter(ratio: float) -> None:
            entered.set()
            if not release.wait(2):
                raise RuntimeError("synthetic setter did not receive release")
        self.setter.side_effect = setter
        def initialize() -> None:
            try:
                results.append(self.budget.ensure_mps_memory_budget("mps"))
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=initialize) for _ in range(8)]
        try:
            for thread in threads:
                thread.start()
            self.assertTrue(entered.wait(2))
            self.assertIsNone(self.budget.get_mps_memory_budget())
        finally:
            release.set()
            for thread in threads:
                thread.join(2)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8)
        self.assertTrue(all(result is results[0] for result in results))
        self.recommended.assert_called_once_with()
        self.setter.assert_called_once_with(0.125)


class TestFactoryCoverage(BudgetTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.configure_factories()

    def configure_factories(self) -> None:
        self.factories: list[tuple[str, dict]] = []
        self.proxy = object()
        self.use_server = Mock(return_value=False)
        def transformer(name: str, **kwargs: object) -> object:
            device = kwargs.get("device")
            if device in (None, "mps"):
                self.assertIsNotNone(self.budget.get_mps_memory_budget(), "constructor ran without budget")
            self.events.append("construct")
            self.factories.append((name, kwargs))
            return object()
        fake_client = types.SimpleNamespace(
            use_model_server=self.use_server,
            get_embedding_proxy=lambda **kwargs: self.proxy,
            get_reranker_proxy=lambda **kwargs: self.proxy,
        )
        serving = source_module("tier_switch/serving.py")
        runtime = types.SimpleNamespace(current_runtime_operation=lambda: None,
                                        require_model_load_allowed=lambda: None, TierRuntimeError=RuntimeError)
        fake_tiers = types.SimpleNamespace(resolve_custom_tier=lambda: {"embed_dim": 192})
        self.static = Mock(return_value=object())
        for name, module in {
            "sentence_transformers": types.SimpleNamespace(SentenceTransformer=transformer, CrossEncoder=transformer),
            "model2vec": types.SimpleNamespace(StaticModel=types.SimpleNamespace(from_pretrained=self.static)),
            "truememory.model_client": fake_client,
            "truememory.tier_switch.serving": serving,
            "truememory.tier_switch.runtime": runtime,
            "truememory.tier_config": fake_tiers,
            "truememory.reranker": types.SimpleNamespace(get_current_reranker_name=lambda: "synthetic/reranker"),
        }.items():
            patcher = patch.dict(sys.modules, {name: module})
            patcher.start()
            self.addCleanup(patcher.stop)

    def local_embed(self, model_id: str) -> types.ModuleType:
        return source_module("vector_search.py", {
            "_model": None, "_frozen_embedding_target": None, "_embedding_dim": -1, "EMBEDDING_MODEL": model_id,
            "_MODEL_DIMS": {"model2vec": 256, "minilm": 384, "bge-small": 384, "qwen3_256": 256},
            "_lock": threading.Lock(), "os": types.SimpleNamespace(environ=self.env),
            "logger": logging.getLogger("synthetic743"),
        }, ("get_model",))

    def server(self):
        tree = ast.parse((SOURCE / "model_server.py").read_text(encoding="utf-8"))
        server = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelServer")
        names = {"_build_embed_model", "_get_reranker", "_mark_sticky_cpu", "_flush_mps_cache",
                 "_check_process_memory"}
        server.body = [node for node in server.body if isinstance(node, ast.FunctionDef) and node.name in names]
        refusal = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == "_ProcessMemoryRefused")
        namespace = {"sys": sys, "os": types.SimpleNamespace(environ=self.env),
                     "log": logging.getLogger("synthetic743"), "gc": types.SimpleNamespace(collect=Mock())}
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, refusal, server], type_ignores=[])),
                     "model_server.py", "exec"), namespace)
        instance = namespace["ModelServer"]()
        instance._max_rss_bytes = 0
        instance._reranker = None
        instance._reranker_name = None
        instance._sticky_cpu = set()
        instance._throttler = None
        instance._write_status_file = Mock()
        return instance

    def local_reranker(self) -> types.ModuleType:
        return source_module("reranker.py", {
            "_model": None, "_model_name": None, "_model_certified": False,
            "contextmanager": contextmanager, "Iterator": Iterator,
            "_lock": threading.Lock(),
            "get_current_reranker_name": lambda: "synthetic/reranker",
            "log": logging.getLogger("synthetic743"),
        }, ("get_reranker", "_reranker_load_lock"))

    def test_all_local_embedding_factories_calibrate_before_constructor(self) -> None:
        for model_id, dim in (("minilm", 384), ("bge-small", 384), ("qwen3_256", 256),
                              ("synthetic/custom", 192)):
            for device in ("mps", "auto", "cpu", "cuda"):
                with self.subTest(model=model_id, device=device):
                    self.fresh({"TRUEMEMORY_DEVICE": device, "TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD": "1"})
                    self.configure_factories()
                    self.torch.cuda.is_available = lambda: device == "cuda"
                    loader = self.local_embed(model_id)
                    loaded = loader.get_model()
                    self.assertIs(loader.get_model(), loaded)
                    self.assertEqual(loader._embedding_dim, dim)
                    self.assertEqual(len(self.factories), 1)
                    self.assertEqual(self.factories[0][1]["device"],
                                     {"mps": "mps", "auto": None, "cpu": "cpu", "cuda": "cuda:0"}[device])
                    if device in ("mps", "auto"):
                        self.assertEqual(self.events, ["recommended", ("set", 0.125), "construct"])
                    else:
                        self.assertEqual(self.events, ["construct"])
                        self.assertNotIn(LOW, self.env)

    def test_server_embedding_factories_and_both_rerankers_share_one_budget(self) -> None:
        self.env["TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD"] = "1"
        server = self.server()
        for model_id in ("qwen3_256", "synthetic/custom"):
            server._build_embed_model(model_id, "mps")
        server._get_reranker("synthetic/reranker")
        self.local_reranker().get_reranker("synthetic/reranker", device="mps")
        self.assertEqual(len(self.factories), 4)
        self.assertEqual(self.events[:2], ["recommended", ("set", 0.125)])
        self.recommended.assert_called_once_with()
        self.setter.assert_called_once_with(0.125)

    def test_cpu_fast_encoder_sticky_reranker_and_explicit_cpu_skip_budget(self) -> None:
        server = self.server()
        server._build_embed_model("qwen3_256", "cpu")
        server._sticky_cpu.add("rerank")
        server._get_reranker("synthetic/reranker")
        self.local_reranker().get_reranker("synthetic/reranker", device="cpu")
        self.assertEqual([kwargs["device"] for _, kwargs in self.factories], ["cpu"] * 3)
        self.assertEqual(self.events, ["construct"] * 3)
        self.assertEqual(self.env, {})

    def test_proxy_and_static_factories_never_configure_local_mps(self) -> None:
        self.use_server.return_value = True
        self.assertIs(self.local_embed("qwen3_256").get_model(), self.proxy)
        self.assertIs(self.local_reranker().get_reranker(), self.proxy)
        self.use_server.return_value = False
        for model_id in ("model2vec", "synthetic/not-opted-in"):
            self.local_embed(model_id).get_model()
            self.server()._build_embed_model(model_id, "mps")
        self.recommended.assert_not_called()
        self.setter.assert_not_called()
        self.assertEqual(self.factories, [])
        self.assertEqual(self.static.call_count, 4)
        self.assertEqual(self.env, {})

    def test_failed_setup_prevents_all_transformer_construction_and_cache_publish(self) -> None:
        self.setter.side_effect = RuntimeError("synthetic failed setup")
        embed = self.local_embed("qwen3_256")
        rerank = self.local_reranker()
        server = self.server()
        for load in (embed.get_model, rerank.get_reranker,
                     lambda: server._build_embed_model("qwen3_256", "mps"), server._get_reranker):
            with self.assertRaises(self.budget.MPSBudgetError):
                load()
        self.assertEqual(self.factories, [])
        self.assertIsNone(embed._model)
        self.assertIsNone(rerank._model)
        self.assertIsNone(server._reranker)
        self.assertIsNone(self.budget.get_mps_memory_budget())
        self.setter.assert_called_once_with(0.125)

    def test_server_sticky_flags_do_not_switch_the_whole_shared_throttler_to_cpu(self) -> None:
        server = self.server()
        server._throttler = types.SimpleNamespace(set_device=Mock())
        self.assertTrue(server._mark_sticky_cpu("rerank"))
        server._throttler.set_device.assert_not_called()
        self.assertTrue(server._mark_sticky_cpu("embed"))
        self.assertFalse(server._mark_sticky_cpu("embed"))
        server._throttler.set_device.assert_not_called()

    def test_pre_model_cleanup_cannot_initialize_allocator_before_low_bootstrap(self) -> None:
        self.torch.mps.empty_cache = Mock()
        self.torch.mps.synchronize = Mock()
        self.torch.cuda.empty_cache = Mock()
        self.budget.gc = types.SimpleNamespace(collect=Mock())
        idle = source_module("mcp_server.py", {
            "_is_still_idle": lambda: True, "_get_rss_mb": lambda: 0,
            "log": logging.getLogger("synthetic743"), "gc": types.SimpleNamespace(collect=Mock()),
        }, ("_unload_models",))
        with patch.dict(sys.modules, {
            "truememory.vector_search": types.SimpleNamespace(unload_model=Mock()),
            "truememory.reranker": types.SimpleNamespace(unload_reranker=Mock(return_value=True)),
        }):
            for initialized in (False, True):
                if initialized:
                    self.budget.ensure_mps_memory_budget("mps")
                self.budget.flush_mps_cache()
                self.server()._flush_mps_cache()
                idle._unload_models()
                self.assertEqual(self.torch.mps.empty_cache.call_count, 3 if initialized else 0)
                self.assertEqual(self.torch.mps.synchronize.call_count, 2 if initialized else 0)
                if not initialized:
                    self.assertEqual(self.env, {})
                    self.recommended.assert_not_called()
        self.assertEqual(self.torch.cuda.empty_cache.call_count, 2)


class TestThrottlerBudget(BudgetTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sensors = source_module("tier_switch/sensors.py")
        self.state = source_module("tier_switch/state_machine.py")
        patcher = patch.dict(sys.modules, {
            "truememory.tier_switch.sensors": self.sensors,
            "truememory.tier_switch.state_machine": self.state,
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        self.module = source_module("tier_switch/throttler.py")
        self.now = 0.0
        self.clock = types.SimpleNamespace(monotonic=lambda: self.now, sleep=lambda seconds: None)
        for module in (self.sensors, self.state, self.module):
            module.time = self.clock
        self.module.read_thermal_pressure = lambda: {"status": "ok", "scheduler_limit": 100}

    def test_memory_and_growth_use_effective_bytes_not_physical_ram(self) -> None:
        policy = self.budget.ensure_mps_memory_budget("mps")
        throttler = self.module.DynamicThrottler("mps")
        self.assertEqual(throttler.mps_cap_gb, 2.5)
        for ratio, status in ((0.84, "ok"), (0.85, "warning"), (0.95, "critical")):
            with self.subTest(ratio=ratio):
                self.now += 20
                self.torch.mps.driver_allocated_memory.return_value = int(ratio * policy.effective_bytes)
                readings = throttler._read_all_channels()
                self.assertAlmostEqual(readings["mps_level"]["ratio"], ratio)
                self.assertEqual(readings["mps_level"]["status"], status)
        self.assertEqual(throttler.growth_tracker._cap_gb, 2.5)
        self.assertAlmostEqual(readings["growth_rate"]["slope_pct"], 10)
        self.assertEqual(throttler._build_metrics()["mps_budget"], {
            "scope": "process_mps_allocator",
            "intended_bytes": policy.intended_bytes, "recommended_bytes": 20 * GIB,
            "effective_bytes": policy.effective_bytes, "source": "automatic", "enforcement": "enforced",
        })

    def test_missing_and_unlimited_budget_cannot_invent_healthy_headroom(self) -> None:
        for unlimited in (False, True):
            with self.subTest(unlimited=unlimited):
                if unlimited:
                    self.env[HIGH] = "0"
                    self.budget.ensure_mps_memory_budget("mps")
                throttler = self.module.DynamicThrottler("mps")
                for _ in range(5):
                    self.now += 20
                    readings = throttler._read_all_channels()
                    throttler.state_machine.safety_check(readings)
                for channel in ("mps_level", "growth_rate"):
                    self.assertEqual(readings[channel]["status"], "unknown")
                    self.assertTrue(readings[channel].get("required", True))
                self.assertEqual(throttler.state_machine.good_streak, 0)
                self.assertIsNone(throttler.mps_cap_gb)
                self.assertIsNone(throttler.growth_tracker)
                self.assertIsNone(throttler._build_metrics()["mps_budget"]["effective_bytes"])
                self.torch.mps.driver_allocated_memory.assert_not_called()
                if not unlimited:
                    self.recommended.assert_not_called()
                    self.setter.assert_not_called()

    def test_budget_initialized_after_throttler_becomes_usable_on_next_sample(self) -> None:
        throttler = self.module.DynamicThrottler("mps")
        self.assertEqual(throttler._read_all_channels()["mps_level"]["status"], "unknown")
        self.budget.ensure_mps_memory_budget("mps")
        self.assertEqual(throttler._read_all_channels()["mps_level"]["status"], "ok")
        self.assertEqual(throttler.mps_cap_gb, 2.5)

    def test_mixed_device_daemon_keeps_monitoring_the_surviving_mps_model(self) -> None:
        policy = self.budget.ensure_mps_memory_budget("mps")
        tree = ast.parse((SOURCE / "model_server.py").read_text(encoding="utf-8"))
        server = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelServer")
        mark = next(node for node in server.body if isinstance(node, ast.FunctionDef) and node.name == "_mark_sticky_cpu")
        namespace = {"log": logging.getLogger("synthetic743")}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[mark], type_ignores=[])),
                     "model_server.py", "exec"), namespace)
        for fallback in ("embed", "rerank"):
            # CPU covers throttler reactivation after sticky embedding; MPS
            # covers a throttler activated before either model's fallback.
            for initial_device in ("cpu", "mps"):
                with self.subTest(fallback=fallback, initial_device=initial_device):
                    throttler = self.module.DynamicThrottler(initial_device)
                    owner = types.SimpleNamespace(
                        _sticky_cpu=set(), _throttler=throttler, _write_status_file=Mock(),
                        _embed_state=types.SimpleNamespace(model=types.SimpleNamespace(
                            device="cpu" if fallback == "embed" else "mps")),
                        _reranker=types.SimpleNamespace(device="cpu" if fallback == "rerank" else "mps"),
                    )
                    namespace["_mark_sticky_cpu"](owner, fallback)
                    self.torch.mps.driver_allocated_memory.return_value = int(0.95 * policy.effective_bytes)
                    readings = throttler._read_all_channels()
                    self.assertEqual(readings["mps_level"]["status"], "critical")
                    self.assertAlmostEqual(readings["mps_level"]["ratio"], 0.95)
                    throttler.batch_size = 8
                    throttler.state_machine.safety_check(readings)
                    self.assertEqual(throttler.batch_size, 4)
                    metrics = throttler._build_metrics()["mps_budget"]
                    self.assertEqual(metrics["scope"], "process_mps_allocator")
                    self.assertEqual(metrics["effective_bytes"], policy.effective_bytes)

    def test_cpu_and_cuda_never_claim_mps_protection_or_initialize_it(self) -> None:
        for device in ("cpu", "cuda:0"):
            with self.subTest(device=device):
                throttler = self.module.DynamicThrottler(device)
                readings = throttler._read_all_channels()
                for channel in ("mps_level", "growth_rate"):
                    self.assertEqual(readings[channel]["status"], "unsupported")
                    self.assertFalse(readings[channel]["required"])
                self.assertEqual(throttler._build_metrics()["mps_budget"]["enforcement"], "not_applicable")
                self.assertIsNone(throttler.mps_cap_gb)
        self.assertEqual(self.env, {})
        self.recommended.assert_not_called()

    def test_throttler_cleanup_skips_uninitialized_mps_and_preserves_cuda_cleanup(self) -> None:
        self.torch.mps.empty_cache = Mock()
        self.torch.mps.synchronize = Mock()
        self.torch.cuda.empty_cache = Mock()
        self.torch.cuda.is_available = lambda: True
        self.module.DynamicThrottler.flush_gpu_cache()
        self.torch.mps.empty_cache.assert_not_called()
        self.torch.cuda.empty_cache.assert_called_once_with()
        self.recommended.assert_not_called()
        self.assertEqual(self.env, {})
        self.budget.ensure_mps_memory_budget("mps")
        self.module.DynamicThrottler.flush_gpu_cache()
        self.torch.mps.empty_cache.assert_called_once_with()
        self.torch.mps.synchronize.assert_called_once_with()

    def test_new_mps_budget_invalidates_pending_cpu_only_sample(self) -> None:
        throttler = self.module.DynamicThrottler("cpu")
        throttler.state_machine.batch_count = 5
        throttler.state_machine.good_streak = 2
        throttler._samples.append((0, {}))
        entered, release = threading.Event(), threading.Event()
        errors = []
        def sensor() -> dict:
            entered.set()
            if not release.wait(2):
                raise RuntimeError("synthetic probe did not receive release")
            return {"scheduler_limit": 100, "status": "ok"}
        self.module.read_thermal_pressure = sensor
        def sample() -> None:
            try:
                throttler._sample_if_due()
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=sample)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            self.budget.ensure_mps_memory_budget("mps")
        finally:
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(throttler.state_machine.good_streak, 0)
        self.assertEqual(list(throttler._samples), [])
        self.assertIsNone(throttler._last_sample_time)
        self.assertEqual(throttler._last_readings, {})
        self.assertEqual(throttler._build_metrics()["mps_budget"]["scope"], "process_mps_allocator")
        self.assertEqual(throttler._read_all_channels()["mps_level"]["status"], "ok")


class TestTorchMetadata(unittest.TestCase):
    def test_apple_silicon_minimum_preserves_other_platforms_and_upper_bound(self) -> None:
        metadata = (SOURCE.parent / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r"(?ms)^dependencies\s*=\s*(\[.*?^\])", metadata)
        self.assertIsNotNone(match)
        # The string-only dependency array is also a Python literal. This
        # avoids a TOML dependency for the supported Python 3.10 test runner.
        requirements = [Requirement(value) for value in ast.literal_eval(match.group(1))]
        torch_requirements = [requirement for requirement in requirements if requirement.name == "torch"]
        for platform, machine in (("darwin", "arm64"), ("darwin", "x86_64"),
                                  ("linux", "x86_64"), ("linux", "aarch64"),
                                  ("win32", "AMD64"), ("win32", "ARM64")):
            apple_silicon = platform == "darwin" and machine == "arm64"
            environment = {"sys_platform": platform, "platform_machine": machine}
            active = [requirement for requirement in torch_requirements
                      if requirement.marker is None or requirement.marker.evaluate(environment)]
            with self.subTest(platform=platform, machine=machine):
                self.assertEqual(len(active), 2 if apple_silicon else 1)
                for version in ("2.0.0", "2.4.1", "2.5.0", "2.14.1", "1.13.1", "3.0.0"):
                    expected = version in ("2.5.0", "2.14.1") or (
                        not apple_silicon and version in ("2.0.0", "2.4.1")
                    )
                    with self.subTest(version=version):
                        self.assertEqual(all(version in requirement.specifier for requirement in active), expected)


class TestStartupEnvironment(unittest.TestCase):
    def test_startup_keeps_thread_defaults_without_exporting_generated_watermarks(self) -> None:
        for explicit in ({}, {HIGH: "0.2", LOW: "0.1", "OMP_NUM_THREADS": "3"}):
            with self.subTest(explicit=explicit):
                env = dict(explicit)
                startup = source_module("model_server.py", {"os": types.SimpleNamespace(environ=env)},
                                        ("_set_mps_memory_cap",))
                startup._set_mps_memory_cap()
                self.assertEqual({key: env[key] for key in (HIGH, LOW) if key in env},
                                 {key: explicit[key] for key in (HIGH, LOW) if key in explicit})
                self.assertEqual(env["OMP_NUM_THREADS"], explicit.get("OMP_NUM_THREADS", "1"))
                for key in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
                    self.assertEqual(env[key], "1")
        # MCP must leave calibration to whichever process actually owns models.
        tree = ast.parse((SOURCE / "mcp_server.py").read_text(encoding="utf-8"))
        self.assertFalse(any(isinstance(node, ast.Constant) and node.value in (HIGH, LOW)
                             for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
