"""Exact wire arithmetic and recovery lifetimes, without native dependencies."""
from __future__ import annotations

import ast
import base64
import json
import math
import os
import struct
import sys
import threading
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "truememory" / "model_server.py"


class Array:
    def __init__(self, shape: tuple, data: list | None = None, dtype: str = "float32") -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.data = data
        self.dtype = dtype

    def __setitem__(self, index: slice, batch: Array) -> None:
        stride = math.prod(self.shape[1:])
        self.data[index.start * stride:index.stop * stride] = batch.data

    def tobytes(self) -> bytes:
        if self.data is None:
            raise AssertionError("Shape-only output must not be materialized")
        return struct.pack("=" + "f" * len(self.data), *self.data)


class Numpy:
    ndarray = Array
    float32 = "float32"

    def __init__(self) -> None:
        self.allocations: list[tuple] = []
        self.conversions = 0

    def asarray(self, values: object, dtype: str = "float32") -> Array:
        self.conversions += 1
        if isinstance(values, Array):
            return values if values.dtype == dtype else Array(values.shape, values.data, dtype)

        def unpack(value: object) -> tuple[tuple, list]:
            if not isinstance(value, (list, tuple)):
                return (), [float(value)]
            if not value:
                return (0,), []
            children = [unpack(child) for child in value]
            if any(shape != children[0][0] for shape, _ in children):
                raise ValueError("Ragged synthetic array")
            return (len(value), *children[0][0]), [item for _, data in children for item in data]

        shape, data = unpack(values)
        return Array(shape, data, dtype)

    def empty(self, shape: tuple, dtype: str) -> Array:
        self.allocations.append(shape)
        if math.prod(shape) > 4096:
            raise AssertionError("Unexpected large full-result allocation")
        return Array(shape, [None] * math.prod(shape), dtype)

    def ascontiguousarray(self, values: object, dtype: str) -> Array:
        return self.asarray(values, dtype)


def load_source() -> types.ModuleType:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
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
    module = types.ModuleType("synthetic_allocation_model_server")
    module.__dict__.update({
        "np": Numpy(), "psutil": None, "_USE_UNIX": True, "_LOOPBACK_HOST": "127.0.0.1",
        "_env_int": lambda name, default, **kwargs: default, "pid_is_alive": lambda pid: False,
    })
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), module.__dict__)
    return module


class CountOnlyInputs:
    """Large cardinality without allocating a large fixture or doing inference."""
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, key: object) -> object:
        raise AssertionError("Oversized request reached a native input slice")


class Workspace:
    pass


class TestAllocationPreflight(unittest.TestCase):
    def setUp(self) -> None:
        package = types.ModuleType("truememory")
        package.__path__ = []
        self.mps = types.ModuleType("truememory.mps_utils")
        self.mps.is_mps_oom = lambda error: "MPS backend out of memory" in str(error)
        self.mps.flush_mps_cache = Mock()
        reranker = types.ModuleType("truememory.reranker")
        reranker.get_current_reranker_name = lambda: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        vector = types.ModuleType("truememory.vector_search")
        vector.EMBEDDING_MODEL = "model2vec"
        vector._TIER_ALIASES = {"edge": "model2vec", "base": "qwen3_256", "pro": "qwen3_256"}
        self.modules = patch.dict(sys.modules, {
            "truememory": package, "truememory.mps_utils": self.mps,
            "truememory.reranker": reranker, "truememory.vector_search": vector,
        })
        self.modules.start()
        self.environment = patch.dict(os.environ, {"TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD": "1"})
        self.environment.start()
        self.ms = load_source()
        self.server = self.ms.ModelServer()
        self.server._write_status_file = lambda: None
        self.server._get_embed_model = Mock(side_effect=AssertionError("Unexpected model load"))
        self.server._get_reranker = Mock(side_effect=AssertionError("Unexpected model load"))
        self.threads: list[threading.Thread] = []
        self.releases: list[threading.Event] = []

    def tearDown(self) -> None:
        for event in self.releases:
            event.set()
        for thread in self.threads:
            thread.join(2)
            self.assertFalse(thread.is_alive())
        self.server._workers.shutdown(wait=True, cancel_futures=True)
        self.environment.stop()
        self.modules.stop()

    @staticmethod
    def request(op: str, count: int = 5, **kwargs: object) -> dict:
        return {"op": op, "texts": [str(i) for i in range(count)],
                "pairs": [("synthetic", str(i)) for i in range(count)],
                "tier": "synthetic/custom", "model_name": "synthetic/reranker",
                "batch_size": 2, **kwargs}

    def test_wire_arithmetic_matches_real_base64_json_including_empty_and_multilabel(self) -> None:
        for field in ("vectors", "scores"):
            for shape in ((0,), (1,), (2,), (3,), (2, 3), (1, 2, 3)):
                with self.subTest(field=field, shape=shape):
                    array = Array(shape, [0.25] * math.prod(shape))
                    response = {"ok": True, field: array, "protocol": self.ms.PROTOCOL_VERSION}
                    encoded = json.dumps(response, default=self.ms._json_default).encode()
                    self.assertEqual(self.ms._result_wire_size(shape, field), len(encoded))
                    self.assertEqual(self.ms._response_wire_size(response), len(encoded))
                    self.assertEqual(self.ms._array_base64_size(shape),
                                     len(base64.b64encode(array.tobytes())))

    def test_exact_limit_is_accepted_and_one_byte_less_is_rejected(self) -> None:
        size = self.ms._result_wire_size((3, 2), "vectors")
        with patch.object(self.ms, "_MAX_MESSAGE_SIZE", size):
            self.ms._check_result_size((3, 2), "vectors")
        with patch.object(self.ms, "_MAX_MESSAGE_SIZE", size - 1):
            with self.assertRaisesRegex(self.ms._ResultTooLarge, f"requires {size} bytes"):
                self.ms._check_result_size((3, 2), "vectors")
        self.assertEqual(self.ms._MAX_MESSAGE_SIZE, 10485760)
        self.assertEqual(self.ms._result_wire_size((7679, 256), "vectors"), 10484497)
        self.assertEqual(self.ms._result_wire_size((7680, 256), "vectors"), 10485861)
        self.assertEqual(self.ms._result_wire_size((1966061,), "scores"), 10485758)
        self.ms._check_result_size((1966061,), "scores")
        with self.assertRaises(self.ms._ResultTooLarge):
            self.ms._check_result_size((1966062,), "scores")

    def test_known_embedding_constructors_reject_before_loading_or_slicing(self) -> None:
        for tier in ("edge", "base", "pro", "model2vec", "qwen3_256", "minilm", "bge-small"):
            with self.subTest(tier=tier):
                response = self.server.handle_request({"op": "embed", "tier": tier,
                                                       "texts": CountOnlyInputs(7680)})
                self.assertEqual(response["error_code"], "result_too_large")
                self.assertIn("Split the request", response["error"])
        self.server._get_embed_model.assert_not_called()
        self.assertEqual(self.ms.np.allocations, [])

    def test_unknown_model_without_opt_in_checks_actual_model2vec_fallback(self) -> None:
        with patch.dict(os.environ, {"TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD": "0"}):
            response = self.server.handle_request({"op": "embed", "tier": "synthetic/custom",
                                                   "texts": CountOnlyInputs(7680)})
        self.assertEqual(response["error_code"], "result_too_large")
        self.server._get_embed_model.assert_not_called()

    def test_builtin_scalar_rerankers_reject_before_loading_or_slicing(self) -> None:
        for name in (None, "cross-encoder/ms-marco-MiniLM-L-6-v2",
                     "Alibaba-NLP/gte-reranker-modernbert-base"):
            with self.subTest(name=name):
                response = self.server.handle_request({"op": "rerank", "model_name": name,
                                                       "pairs": CountOnlyInputs(1966062)})
                self.assertEqual(response["error_code"], "result_too_large")
        self.server._get_reranker.assert_not_called()

    def test_custom_native_shape_is_checked_before_float32_conversion_or_full_allocation(self) -> None:
        for op, field, method in (("embed", "vectors", "encode"), ("rerank", "scores", "predict")):
            with self.subTest(op=op):
                native = Array((2, 1000000), dtype="float64")
                model = types.SimpleNamespace(**{method: Mock(return_value=native)})
                setattr(self.server, "_get_embed_model" if op == "embed" else "_get_reranker",
                        Mock(return_value=model))
                conversions = self.ms.np.conversions
                response = self.server.handle_request(self.request(op))
                self.assertEqual(response["error_code"], "result_too_large")
                self.assertEqual(self.ms.np.conversions, conversions)
                self.assertEqual(self.ms.np.allocations, [])
                self.assertEqual(getattr(model, method).call_count, 1)

    def test_custom_truncation_upper_bound_and_changed_opt_in_do_not_reject_valid_output(self) -> None:
        model = types.SimpleNamespace(encode=lambda texts, **kwargs: [[float(t), 0.5] for t in texts],
                                      get_embedding_dimension=Mock(return_value=4096))
        self.server._embed_state = self.ms._EmbedState(model, "synthetic/custom", "synthetic/custom")
        self.server._get_embed_model = Mock(return_value=model)
        # Even a getter can report truncate_dim when the native width is unknown.
        with patch.object(self.ms, "_MAX_MESSAGE_SIZE", 200), \
                patch.dict(os.environ, {"TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD": "0",
                                        "TRUEMEMORY_CUSTOM_EMBED_DIM": "4096"}):
            response = self.server.handle_request(self.request("embed"))
        self.assertTrue(response["ok"])
        self.assertEqual(response["vectors"].shape, (5, 2))
        model.get_embedding_dimension.assert_not_called()

    def test_valid_microbatches_keep_all_rows_order_dimensions_and_float32(self) -> None:
        for op, field, method in (("embed", "vectors", "encode"), ("rerank", "scores", "predict")):
            calls = []

            def forward(items: list, **kwargs: object) -> Array:
                ids = [int(item if op == "embed" else item[1]) for item in items]
                calls.append(ids)
                return Array((len(ids), 2), [value for i in ids for value in (float(i), 0.5)], "float64")

            model = types.SimpleNamespace(**{method: forward})
            setattr(self.server, "_get_embed_model" if op == "embed" else "_get_reranker",
                    lambda _name: model)
            response = self.server.handle_request(self.request(op))
            self.assertTrue(response["ok"])
            self.assertEqual(calls, [[0, 1], [2, 3], [4]])
            self.assertEqual(response[field].shape, (5, 2))
            self.assertEqual(response[field].dtype, "float32")
            self.assertEqual(response[field].data, [value for i in range(5) for value in (float(i), 0.5)])

    def test_native_count_and_later_dimension_changes_remain_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "count"):
            self.ms._store_batch_result(None, Array((1, 2), [0, 0]), 0, 2, 3)
        result = self.ms._store_batch_result(None, Array((1, 2), [0, 0]), 0, 1, 2)
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            self.ms._store_batch_result(result, Array((1, 3), [0, 0, 0]), 1, 1, 2)
        self.assertEqual(self.ms.np.allocations, [(2, 2)])

    def test_empty_native_results_keep_their_existing_rank(self) -> None:
        model = types.SimpleNamespace(encode=lambda *args, **kwargs: Array((0,), []),
                                      predict=lambda *args, **kwargs: Array((0,), []))
        self.server._get_embed_model = lambda tier: model
        self.server._get_reranker = lambda name: model
        for op, field in (("embed", "vectors"), ("rerank", "scores")):
            response = self.server.handle_request(self.request(op, count=0))
            self.assertEqual(response[field].shape, (0,))

    def test_serializer_rejects_unknown_shape_before_array_conversion_or_base64(self) -> None:
        response = {"ok": True, "vectors": Array((2, 1000000), dtype="float64")}
        connection = Mock()
        with patch.object(self.ms.np, "ascontiguousarray", side_effect=AssertionError("Array copy")), \
                patch.object(self.ms.base64, "b64encode", side_effect=AssertionError("Base64 copy")):
            self.server._send_response(connection, response)
        wire = connection.sendall.call_args.args[0]
        self.assertEqual(struct.unpack(">I", wire[:4])[0], len(wire) - 4)
        self.assertLessEqual(len(wire) - 4, self.ms._MAX_MESSAGE_SIZE)
        self.assertEqual(json.loads(wire[4:]), {"ok": False, "error": "Response too large", "protocol": 1})

    def test_fast_native_output_rejection_does_not_fall_back_or_copy(self) -> None:
        model = types.SimpleNamespace(encode=Mock(return_value=Array((1, 2000000))))
        self.server._get_fast_encoder = Mock(return_value=model)
        self.server._inference_lock.acquire()
        try:
            response = self.server.handle_request(self.request("embed", count=1))
        finally:
            self.server._inference_lock.release()
        self.assertEqual(response["error_code"], "result_too_large")
        self.server._get_embed_model.assert_not_called()
        self.assertEqual(self.ms.np.conversions, 0)

    def test_expiry_during_preflight_still_prevents_model_loading(self) -> None:
        clock = [0.0]

        def preflight(tier: str, count: int) -> None:
            clock[0] = 2.0

        with patch.object(self.ms.time, "time", side_effect=lambda: clock[0]), \
                patch.object(self.ms.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(self.server, "_preflight_embed_result", side_effect=preflight):
            response = self.server.handle_request(self.request("embed", deadline=1.0))
        self.assertIn("deadline", response["error"])
        self.server._get_embed_model.assert_not_called()

    def test_embed_failed_workspace_is_gone_before_recovery_in_single_and_batch_paths(self) -> None:
        for count, fail_on in ((1, 1), (5, 2)):
            with self.subTest(count=count):
                refs, calls, moves = [], [], []
                recovered = threading.Event()
                testcase = self

                class Model:
                    def encode(self, texts: list, **kwargs: object) -> list:
                        calls.append(list(texts))
                        if len(calls) == fail_on:
                            workspace = Workspace()
                            refs.append(weakref.ref(workspace))
                            raise RuntimeError("MPS backend out of memory")
                        return [[float(text), 0.5] for text in texts]

                    def to(self, device: str) -> None:
                        testcase.assertIsNone(refs[-1]())
                        testcase.assertTrue(testcase.server._inference_lock.locked())
                        testcase.assertTrue(testcase.server._lock.locked())
                        moves.append(device)
                        recovered.set()

                self.server._get_embed_model = lambda tier: Model()
                self.mps.flush_mps_cache.side_effect = lambda: self.assertIsNone(refs[-1]())
                response = self.server.handle_request(self.request("embed", count=count))
                self.assertTrue(response["ok"])
                self.assertTrue(recovered.is_set())
                self.assertEqual(moves, ["cpu"])
                self.assertEqual(calls, [["0"], ["0"]] if count == 1 else
                                 [["0", "1"], ["2", "3"], ["2", "3"], ["4"]])
                self.assertIn("embed", self.server._sticky_cpu)

    def rerank_recovery_fixture(self, on_replacement: object = None) -> tuple[list, list, list]:
        workspaces, old_models, calls = [], [], []
        testcase = self

        class Model:
            def __init__(self, fail: bool) -> None:
                self.fail = fail

            def predict(self, pairs: list, **kwargs: object) -> list:
                ids = [pair[1] for pair in pairs]
                calls.append(ids)
                if self.fail and ids[0] == "2":
                    workspace = Workspace()
                    workspaces.append(weakref.ref(workspace))
                    raise RuntimeError("MPS backend out of memory")
                return [float(value) for value in ids]

        def get_model(name: str) -> Model:
            if self.server._reranker is None:
                replacing = "rerank" in self.server._sticky_cpu
                if replacing:
                    self.assertIsNone(workspaces[-1]())
                    self.assertIsNone(old_models[-1]())
                    self.assertTrue(self.server._inference_lock.locked())
                    self.assertTrue(self.server._lock.locked())
                    if on_replacement is not None:
                        on_replacement()
                self.server._reranker = Model(fail=not replacing)
                self.server._reranker_name = name
                if not replacing:
                    old_models.append(weakref.ref(self.server._reranker))
            return self.server._reranker

        def flush() -> None:
            testcase.assertIsNone(workspaces[-1]())
            testcase.assertIsNone(old_models[-1]())

        self.server._get_reranker = get_model
        self.mps.flush_mps_cache.side_effect = flush
        return workspaces, old_models, calls

    def test_rerank_releases_failed_workspace_and_old_model_before_cpu_constructor(self) -> None:
        workspaces, old_models, calls = self.rerank_recovery_fixture()
        response = self.server.handle_request(self.request("rerank"))
        self.assertTrue(response["ok"])
        self.assertIsNone(workspaces[0]())
        self.assertIsNone(old_models[0]())
        self.assertEqual(calls, [["0", "1"], ["2", "3"], ["2", "3"], ["4"]])
        self.assertEqual(response["scores"].data, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_cpu_constructor_failure_releases_ownership_and_keeps_failed_model_dropped(self) -> None:
        def fail() -> None:
            raise RuntimeError("Synthetic constructor failure")

        self.rerank_recovery_fixture(fail)
        with self.assertRaisesRegex(RuntimeError, "Synthetic constructor failure"):
            self.server.handle_request(self.request("rerank"))
        self.assertIsNone(self.server._reranker)
        self.assertFalse(self.server._lock.locked())
        self.assertFalse(self.server._inference_lock.locked())
        self.assertIn("rerank", self.server._sticky_cpu)

    def test_replacement_retains_ownership_while_ping_remains_responsive(self) -> None:
        entered, release, competitor_started, loaded = (threading.Event() for _ in range(4))
        self.releases.append(release)

        def replacement() -> None:
            entered.set()
            self.assertTrue(release.wait(2))

        self.rerank_recovery_fixture(replacement)
        outcomes = []

        def run(request: dict, started: threading.Event | None = None) -> None:
            if started is not None:
                started.set()
            try:
                outcomes.append(self.server.handle_request(request))
            except Exception as error:
                outcomes.append(error)

        def load_embed(tier: str) -> object:
            loaded.set()
            return types.SimpleNamespace(encode=lambda texts, **kwargs: [[0.5] for _ in texts])

        self.server._get_embed_model = load_embed
        owner = threading.Thread(target=run, args=(self.request("rerank"),))
        self.threads.append(owner)
        owner.start()
        self.assertTrue(entered.wait(1))
        competitor = threading.Thread(target=run, args=(self.request("embed", count=2), competitor_started))
        self.threads.append(competitor)
        competitor.start()
        self.assertTrue(competitor_started.wait(1))
        self.assertFalse(loaded.wait(0.05))
        self.assertTrue(self.server.handle_request({"op": "ping"})["ok"])
        release.set()
        for thread in self.threads:
            thread.join(2)
            self.assertFalse(thread.is_alive())
        self.assertTrue(loaded.is_set())
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(all(isinstance(result, dict) and result["ok"] for result in outcomes), outcomes)

    def test_expired_oom_drops_failed_cache_without_recovery_or_retry(self) -> None:
        for op in ("embed", "rerank"):
            with self.subTest(op=op):
                clock, workspaces, models = [0.0], [], []

                def fail(items: list, **kwargs: object) -> list:
                    workspace = Workspace()
                    workspaces.append(weakref.ref(workspace))
                    clock[0] = 2.0
                    raise RuntimeError("MPS backend out of memory")

                class Model:
                    encode = staticmethod(fail)
                    predict = staticmethod(fail)
                    to = Mock(side_effect=AssertionError("Expired CPU transfer"))

                def load(name: str) -> Model:
                    model = Model()
                    models.append(weakref.ref(model))
                    if op == "embed":
                        self.server._embed_state = self.ms._EmbedState(model, name, "synthetic/custom")
                    else:
                        self.server._reranker = model
                    return model

                setattr(self.server, "_get_embed_model" if op == "embed" else "_get_reranker", load)
                self.mps.flush_mps_cache.reset_mock()
                with patch.object(self.ms.time, "time", side_effect=lambda: clock[0]), \
                        patch.object(self.ms.time, "monotonic", side_effect=lambda: clock[0]):
                    response = self.server.handle_request(self.request(op, deadline=1.0))
                self.assertIn("deadline", response["error"])
                self.assertEqual(len(models), 1)
                self.assertIsNone(models[0]())
                self.assertIsNone(workspaces[0]())
                self.assertIn(op, self.server._sticky_cpu)
                self.mps.flush_mps_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
