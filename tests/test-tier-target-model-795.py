"""Server target identity regressions using real control flow and stdlib fakes."""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch
import weakref


SOURCE = Path(__file__).resolve().parents[1] / "truememory" / "model_server.py"


class Array:
    def __init__(self, rows: list[list[float]], shape: tuple[int, int]) -> None:
        self.rows = rows
        self.shape = shape
        self.ndim = 2

    def __setitem__(self, key: slice, value: Array) -> None:
        self.rows[key] = value.rows


class Numpy:
    ndarray = Array
    float32 = "float32"

    @staticmethod
    def asarray(values: Array, **_kwargs: object) -> Array:
        return values

    @staticmethod
    def empty(shape: tuple[int, int], **_kwargs: object) -> Array:
        return Array([[] for _ in range(shape[0])], shape)


class FakeModel:
    def __init__(self, identity: str, device: str | None) -> None:
        self.identity = identity
        self.device = device
        self.calls: list[int] = []
        self.before_encode = lambda: None
        self.oom_once = False

    def encode(self, texts: list[str], **kwargs: object) -> Array:
        self.before_encode()
        self.calls.append(len(texts))
        if self.oom_once:
            self.oom_once = False
            raise RuntimeError("MPS backend out of memory")
        if len(texts) > kwargs["batch_size"]:
            raise AssertionError("Native microbatch ceiling exceeded")
        return Array([[1.0] * 256 for _ in texts], (len(texts), 256))

    def to(self, device: str) -> None:
        self.device = device


def load_server() -> types.ModuleType:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    tree = ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *definitions,
    ], type_ignores=[]))
    module = types.ModuleType("synthetic_target_server_795")
    module.__dict__.update(
        dataclass=dataclass, contextmanager=contextmanager, ThreadPoolExecutor=ThreadPoolExecutor,
        np=Numpy(), json=json, math=math, os=os, sys=sys, threading=threading, time=time,
        log=logging.getLogger("synthetic-target-795"), _MAX_MESSAGE_SIZE=100 * 1024**2,
        PROTOCOL_VERSION=1, _EMBED_BATCH_LIMIT=32, _RERANK_BATCH_LIMIT=64,
    )
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(tree, str(SOURCE), "exec"), module.__dict__)
    return module


class TestTierTargetIdentity(unittest.TestCase):
    def setUp(self) -> None:
        package = types.ModuleType("truememory")
        package.__path__ = []
        self.vector = types.ModuleType("truememory.vector_search")
        self.vector._TIER_ALIASES = {"edge": "model2vec", "base": "qwen3_256", "pro": "qwen3_256"}
        self.vector.EMBEDDING_MODEL = "model2vec"
        self.vector._embedding_dim = 256
        self.vector._model_generation = 7
        self.vector._model = object()

        def set_model(value: str) -> None:
            self.vector.EMBEDDING_MODEL = self.vector._resolve_model_name(value)
            self.vector._embedding_dim = 256
            self.vector._model_generation += 1
            self.vector._model = None

        self.vector.set_embedding_model = Mock(side_effect=set_model)
        self.custom = types.ModuleType("truememory.tier_config")
        self.custom.get_embed_model = Mock(return_value="synthetic/encoder-one")
        vector_tree = ast.parse(SOURCE.with_name("vector_search.py").read_text(encoding="utf-8"))
        resolver_nodes = [node for node in vector_tree.body if (
            isinstance(node, ast.FunctionDef) and node.name == "_resolve_model_name"
            or isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "_REMOVED_MODELS"
                for target in node.targets
            )
        )]
        self.vector.__dict__.update(
            logger=logging.getLogger("synthetic-vector-795"),
            _MODEL_DIMS={"model2vec": 256, "qwen3_256": 256, "minilm": 384, "bge-small": 384},
            _cfg_get_embed_model=self.custom.get_embed_model,
        )
        exec(compile(ast.Module(body=resolver_nodes, type_ignores=[]), "synthetic-real-vector-resolver", "exec"),
             self.vector.__dict__)
        self.mps = types.ModuleType("truememory.mps_utils")
        self.mps.resolve_device = lambda _device: "cpu"
        self.mps.is_mps_oom = lambda error: "MPS backend out of memory" in str(error)
        self.mps.flush_mps_cache = Mock()
        modules = patch.dict(sys.modules, {
            "truememory": package, "truememory.vector_search": self.vector,
            "truememory.tier_config": self.custom, "truememory.mps_utils": self.mps,
        })
        modules.start()
        self.addCleanup(modules.stop)
        environment = patch.dict(os.environ, {"TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD": "1"})
        environment.start()
        self.addCleanup(environment.stop)
        self.ms = load_server()
        self.server = self.ms.ModelServer()
        self.addCleanup(self.server._workers.shutdown, wait=True)
        self.server._write_status_file = lambda: None
        self.server._SUSTAINED_THRESHOLD = 1000
        self.builds: list[tuple[str, str | None]] = []
        self.server._build_embed_model = self.build

    def build(self, identity: str, device: str | None) -> FakeModel:
        self.assertTrue(self.server._lock.locked())
        self.assertTrue(self.server._inference_lock.locked())
        self.builds.append((identity, device))
        return FakeModel(identity, device)

    def request(self, tier: str, count: int = 2, **extra: object) -> dict:
        return self.server.handle_request({
            "op": "embed_batched", "tier": tier, "texts": ["synthetic"] * count,
            "batch_size": 32, **extra,
        })

    def globals(self) -> tuple:
        return (self.vector.EMBEDDING_MODEL, self.vector._embedding_dim,
                self.vector._model_generation, self.vector._model)

    def test_all_builtin_alias_pairs_reuse_one_model_without_global_mutation(self) -> None:
        for aliases in (("edge", "model2vec"), ("base", "pro", "qwen3_256")):
            for first in aliases:
                for second in aliases:
                    with self.subTest(first=first, second=second):
                        self.server._embed_state = None
                        self.builds.clear()
                        before = self.globals()
                        self.assertTrue(self.request(first)["ok"])
                        snapshot = self.server._embed_state
                        self.assertTrue(self.request(second)["ok"])
                        self.assertIs(self.server._embed_state.model, snapshot.model)
                        self.assertEqual(self.server._embed_state.tier, second)
                        self.assertEqual(snapshot.tier, first)
                        self.assertEqual(len(self.builds), 1)
                        self.assertEqual(self.globals(), before)
                        self.vector.set_embedding_model.assert_not_called()

    def test_explicit_target_does_not_change_later_default_request(self) -> None:
        before = self.globals()
        for tier, identity in (("", "model2vec"), ("base", "qwen3_256"), ("", "model2vec")):
            self.assertTrue(self.request(tier)["ok"])
            self.assertEqual(self.server._embed_state.model_id, identity)
            self.assertEqual(self.globals(), before)
        self.assertEqual([identity for identity, _ in self.builds],
                         ["model2vec", "qwen3_256", "model2vec"])

    def test_main_default_snapshot_survives_global_change_until_invalidated(self) -> None:
        self.request("")
        old = self.server._embed_state.model
        self.vector.set_embedding_model("pro")
        self.assertTrue(self.request("")["ok"])
        self.assertIs(self.server._embed_state.model, old)
        self.assertEqual(self.server._embed_state.model_id, "model2vec")
        self.assertEqual(len(self.builds), 1)
        self.server._embed_state = None
        self.assertTrue(self.request("")["ok"])
        self.assertIsNot(self.server._embed_state.model, old)
        self.assertEqual(self.server._embed_state.model_id, "qwen3_256")
        self.vector.set_embedding_model.assert_called_once_with("pro")

    def test_failed_distinct_load_preserves_old_snapshot_and_global_state(self) -> None:
        self.request("edge")
        old = self.server._embed_state
        before = self.globals()
        self.server._build_embed_model = Mock(side_effect=RuntimeError("synthetic load failure"))
        with self.assertRaisesRegex(RuntimeError, "synthetic load failure"):
            self.request("pro")
        self.assertIs(self.server._embed_state, old)
        self.assertEqual(self.globals(), before)
        self.assertEqual(self.server._inflight, 0)
        self.assertFalse(self.server._lock.locked())
        self.assertFalse(self.server._inference_lock.locked())

    def test_removed_model_rejection_matches_real_resolver_without_mutation(self) -> None:
        self.request("base")
        before = self.globals()
        snapshot = self.server._embed_state
        builds = list(self.builds)
        for removed in self.vector._REMOVED_MODELS:
            for spelling in (removed, removed.upper(), "  " + removed + "  ", "\t" + removed.upper() + "\n"):
                with self.subTest(spelling=spelling):
                    with self.assertRaises(ValueError) as reference:
                        self.vector._resolve_model_name(spelling)
                    with self.assertRaises(ValueError) as direct:
                        self.server._resolve_embed_model_id(spelling)
                    self.assertEqual(str(direct.exception), str(reference.exception))
                    for count in (1, 2):
                        with self.assertRaises(ValueError) as request:
                            self.request(spelling, count=count)
                        self.assertEqual(str(request.exception), str(reference.exception))
                    self.assertIs(self.server._embed_state, snapshot)
                    self.assertEqual(self.globals(), before)
                    self.assertEqual(self.builds, builds)
                    self.assertFalse(self.server._lock.locked())
                    self.assertFalse(self.server._inference_lock.locked())
        self.vector.set_embedding_model.assert_not_called()

    def test_other_resolver_spellings_keep_existing_peek_behavior(self) -> None:
        before = self.globals()
        for spelling in ("base", "pro", "qwen3_256", "custom", " PRO ", "invalid-tier", "Synthetic/CaseSensitive"):
            with self.subTest(spelling=spelling):
                self.assertEqual(self.server._resolve_embed_model_id(spelling),
                                 self.server._peek_embed_model_id(spelling))
        self.assertEqual(self.globals(), before)
        self.vector.set_embedding_model.assert_not_called()

    def test_explicit_custom_cache_keeps_original_config_and_opt_in_semantics(self) -> None:
        self.request("custom")
        original = self.server._embed_state
        self.custom.get_embed_model.side_effect = ValueError("synthetic invalid custom settings")
        with patch.dict(os.environ, {"TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD": "0"}):
            self.assertTrue(self.request("custom")["ok"])
        self.assertIs(self.server._embed_state, original)
        self.assertEqual(len(self.builds), 1)
        self.custom.get_embed_model.side_effect = None
        self.custom.get_embed_model.return_value = "synthetic/encoder-two"
        self.assertTrue(self.request("custom")["ok"])
        self.assertIs(self.server._embed_state, original)
        # A different explicit key retains the legacy custom reload boundary.
        self.assertTrue(self.request("synthetic/encoder-one")["ok"])
        self.assertIsNot(self.server._embed_state.model, original.model)
        self.assertEqual(len(self.builds), 2)

    def test_legacy_fallback_identities_are_not_canonicalized_as_builtin_aliases(self) -> None:
        for tier in ("minilm", "bge-small", "edge"):
            self.assertTrue(self.request(tier)["ok"])
        self.assertEqual([identity for identity, _ in self.builds], ["minilm", "bge-small", "model2vec"])
        static = types.ModuleType("model2vec")
        static.StaticModel = types.SimpleNamespace(from_pretrained=Mock(return_value=object()))
        with patch.dict(sys.modules, {"model2vec": static}):
            for identity in ("minilm", "bge-small"):
                self.ms.ModelServer._build_embed_model(identity, "cpu")
        self.assertEqual(static.StaticModel.from_pretrained.call_count, 2)
        for call in static.StaticModel.from_pretrained.call_args_list:
            self.assertEqual(call.args, ("minishlab/potion-base-8M",))
            self.assertEqual(call.kwargs, {"force_download": False})

    def test_raw_custom_builtin_or_fallback_snapshot_survives_config_changes(self) -> None:
        for initial in ("qwen3_256", "model2vec", None):
            for opt_out in (False, True):
                with self.subTest(initial=initial, opt_out=opt_out):
                    self.server._embed_state = None
                    self.builds.clear()
                    self.custom.get_embed_model.return_value = initial
                    self.custom.get_embed_model.side_effect = (
                        ValueError("synthetic missing custom settings") if initial is None else None
                    )
                    self.assertTrue(self.request("custom")["ok"])
                    original = self.server._embed_state
                    self.assertEqual(original.model_id, initial or "model2vec")
                    self.custom.get_embed_model.reset_mock()
                    self.custom.get_embed_model.return_value = "synthetic/encoder-two"
                    self.custom.get_embed_model.side_effect = (
                        ValueError("synthetic custom opt-out") if opt_out else None
                    )
                    with patch.dict(os.environ, {"TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD": "0" if opt_out else "1"}):
                        with patch.object(self.ms, "_check_result_size") as check:
                            self.server._preflight_embed_result("custom", 2)
                            check.assert_called_once_with((2, 256), "vectors")
                        self.assertTrue(self.request("custom")["ok"])
                    self.assertIs(self.server._embed_state, original)
                    self.assertEqual(len(self.builds), 1)
                    self.custom.get_embed_model.assert_not_called()

    def test_preflight_uses_same_resolved_identity_and_custom_cache_boundary(self) -> None:
        self.request("")
        self.vector.EMBEDDING_MODEL = "synthetic/encoder-one"
        with patch.object(self.ms, "_check_result_size") as check:
            self.server._preflight_embed_result("", 2)
            check.assert_called_once_with((2, 256), "vectors")
            check.reset_mock()
            self.server._embed_state = None
            self.server._preflight_embed_result("", 2)
            check.assert_not_called()
        self.request("custom")
        with patch.dict(os.environ, {"TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD": "0"}):
            with patch.object(self.ms, "_check_result_size") as check:
                self.server._preflight_embed_result("custom", 2)
                check.assert_not_called()
                self.server._preflight_embed_result("synthetic/other", 2)
                check.assert_called_once_with((2, 256), "vectors")
        self.request("base")
        with patch.object(self.ms, "_check_result_size") as check:
            self.server._preflight_embed_result("pro", 2)
            check.assert_called_once_with((2, 256), "vectors")

    def test_fast_and_main_lanes_keep_captured_default_identity(self) -> None:
        self.vector.EMBEDDING_MODEL = "base"
        self.request("")
        snapshot = self.server._embed_state
        self.vector.EMBEDDING_MODEL = "edge"

        def fast_build(identity: str, device: str | None) -> FakeModel:
            self.assertTrue(self.server._fast_lock.locked())
            self.assertEqual((identity, device), ("qwen3_256", "cpu"))
            return FakeModel(identity, device)

        self.server._build_embed_model = fast_build
        with self.server._lock:
            response = self.server._handle_fast_embed(["synthetic"], "")
        self.assertTrue(response["ok"])
        self.assertIs(self.server._embed_state, snapshot)
        self.assertEqual(self.server._fast_model_id, "qwen3_256")
        self.server._build_embed_model = self.build
        self.request("")
        self.assertIs(self.server._embed_state, snapshot)
        # A different explicit key leaves the original raw-default cache.
        self.request("edge")
        self.request("")
        self.assertEqual(self.server._embed_state.model_id, "model2vec")

    def test_waiting_default_query_reuses_identity_captured_during_load(self) -> None:
        self.vector.EMBEDDING_MODEL = "base"
        build_started = threading.Event()
        release_build = threading.Event()
        query_waiting = threading.Event()
        responses: dict[str, dict] = {}
        failures: list[BaseException] = []
        main_lock = self.server._lock

        class ObservedLock:
            def acquire(inner_self, blocking: bool = True, timeout: float = -1) -> bool:
                if threading.current_thread().name == "synthetic-waiting-query" and blocking:
                    query_waiting.set()
                return main_lock.acquire(blocking=blocking, timeout=timeout)

            def release(inner_self) -> None:
                main_lock.release()

            def locked(inner_self) -> bool:
                return main_lock.locked()

        self.server._lock = ObservedLock()

        def build(identity: str, device: str | None) -> FakeModel:
            model = self.build(identity, device)
            if threading.current_thread().name == "synthetic-default-loader":
                build_started.set()
                if not release_build.wait(2):
                    raise AssertionError("Synthetic model load was not released")
            return model

        def request(key: str, count: int) -> None:
            try:
                responses[key] = self.request("", count=count, deadline=time.time() + 3)
            except BaseException as error:
                failures.append(error)

        self.server._build_embed_model = build
        loader = threading.Thread(name="synthetic-default-loader", target=request, args=("batch", 2))
        query = threading.Thread(name="synthetic-waiting-query", target=request, args=("query", 1))
        loader.start()
        try:
            self.assertTrue(build_started.wait(2))
            self.assertIsNone(self.server._embed_state)
            self.vector.EMBEDDING_MODEL = "edge"
            query.start()
            self.assertTrue(query_waiting.wait(2))
            self.assertNotIn("query", responses)
        finally:
            release_build.set()
            loader.join(5)
            if query.ident is not None:
                query.join(5)
        self.assertFalse(loader.is_alive())
        self.assertFalse(query.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(set(responses), {"batch", "query"})
        self.assertTrue(all(response["ok"] for response in responses.values()))
        self.assertEqual(self.builds, [("qwen3_256", "cpu")])
        self.assertEqual(self.server._embed_state.model.calls, [2, 1])
        self.assertEqual(self.server._embed_state.model_id, "qwen3_256")
        self.assertFalse(main_lock.locked())
        self.assertFalse(self.server._inference_lock.locked())

    def test_alias_publication_preserves_snapshot_for_fast_lane(self) -> None:
        self.request("base")
        snapshot = self.server._embed_state
        self.request("pro")
        with self.server._lock:
            self.server._fast_encoder = FakeModel("qwen3_256", "cpu")
            self.server._fast_model_id = "qwen3_256"
            self.assertTrue(self.server._handle_fast_embed(["synthetic"], "pro")["ok"])
        self.assertEqual(snapshot.tier, "base")
        self.assertIs(snapshot.model, self.server._embed_state.model)

    def test_expired_request_cannot_load_or_relabel_cached_alias(self) -> None:
        self.request("base")
        snapshot = self.server._embed_state
        response = self.request("pro", deadline=time.time() - 1)
        self.assertFalse(response["ok"])
        self.assertIn("deadline", response["error"])
        self.assertIs(self.server._embed_state, snapshot)
        self.assertEqual(len(self.builds), 1)

    def test_distinct_target_cannot_load_while_existing_main_inference_is_owned(self) -> None:
        self.request("edge")
        entered = threading.Event()
        release = threading.Event()
        results: list[dict] = []

        def blocked_encode() -> None:
            self.assertTrue(self.server._inference_lock.locked())
            entered.set()
            if not release.wait(2):
                raise AssertionError("Synthetic inference was not released")

        self.server._embed_state.model.before_encode = blocked_encode
        worker = threading.Thread(target=lambda: results.append(self.request("edge")))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            response = self.request("pro", deadline=time.time() + 0.05)
            self.assertFalse(response["ok"])
            self.assertIn("deadline", response["error"])
            self.assertEqual(self.builds, [("model2vec", "cpu")])
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["ok"])
        self.request("pro")
        self.assertEqual(self.builds[-1], ("qwen3_256", "cpu"))

    def test_alias_reuse_retains_native_batch_ceiling_and_sticky_recovery(self) -> None:
        self.request("base")
        model = self.server._embed_state.model
        model.oom_once = True
        self.assertTrue(self.request("pro", count=35)["ok"])
        self.assertEqual(model.calls, [2, 32, 32, 3])
        self.assertEqual(self.server._sticky_cpu, {"embed"})
        self.assertEqual(model.device, "cpu")
        self.assertEqual(len(self.builds), 1)

    def test_expired_oom_invalidation_reloads_current_default_on_sticky_cpu(self) -> None:
        now = [1000.0]
        clock = types.SimpleNamespace(time=lambda: now[0], monotonic=lambda: now[0])
        self.vector.EMBEDDING_MODEL = "base"
        self.mps.resolve_device = lambda _device: "mps"
        with patch.object(self.ms, "time", clock):
            self.request("")
            failed = self.server._embed_state.model
            failed.oom_once = True

            def expired_native_work() -> None:
                now[0] += 2
                self.vector.EMBEDDING_MODEL = "edge"

            failed.before_encode = expired_native_work
            response = self.request("", deadline=now[0] + 1)
            self.assertFalse(response["ok"])
            self.assertIn("deadline", response["error"])
            self.assertIsNone(self.server._embed_state)
            self.assertEqual(self.server._sticky_cpu, {"embed"})
            self.mps.flush_mps_cache.assert_not_called()
            self.assertEqual(failed.device, "mps")
            self.assertTrue(self.request("")["ok"])
        self.assertEqual(self.builds, [("qwen3_256", "mps"), ("model2vec", "cpu")])
        self.assertEqual(self.server._embed_state.model_id, "model2vec")
        self.assertEqual(self.server._embed_state.model.device, "cpu")

    def test_old_instance_can_still_be_retained_during_distinct_load(self) -> None:
        self.request("edge")
        old = weakref.ref(self.server._embed_state.model)
        observed: list[bool] = []

        def build(identity: str, device: str | None) -> FakeModel:
            observed.append(old() is not None)
            return self.build(identity, device)

        self.server._build_embed_model = build
        self.request("base")
        self.assertEqual(observed, [True])
        self.assertIsNone(old())


if __name__ == "__main__":
    unittest.main()
