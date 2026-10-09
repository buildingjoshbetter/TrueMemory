"""Selected response receipts through real control flow and stdlib model doubles."""
from __future__ import annotations

import dataclasses
from pathlib import Path
import runpy
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch


BASE = runpy.run_path(str(Path(__file__).with_name("test-prepared-target-795.py")))
Array = BASE["Array"]


class FirstBusyLock:
    """One nonblocking contention observation, followed by a real unlocked lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.first = True

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if not blocking and self.first:
            self.first = False
            return False
        return self.lock.acquire(blocking, timeout)

    def release(self) -> None:
        self.lock.release()

    def locked(self) -> bool:
        return self.lock.locked()


class TestServingReceipts(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = BASE["TestPreparedTargets"]()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.client = self.fixture.client
        self.client.PROTOCOL_VERSION = 1
        self.client.np.integer = int
        self.server = self.fixture.server
        self.ms = self.fixture.ms
        self.Target = self.fixture.Target
        self.edge = self.Target.capture("edge")
        self.base = self.Target.capture("base")
        self.calls: list[tuple[str, str, list, int]] = []
        self.builds: list[tuple[str, str | None]] = []
        self.width = 256
        self.before_encode = lambda: None
        self.before_predict = lambda: None
        self.build_hook = lambda: None
        self.embed_oom = False
        self.rerank_oom = False
        self.cpu_error = False
        owner = self

        class Model:
            def __init__(self, identity: str, device: str | None = None) -> None:
                self.identity, self.device = identity, device

            def encode(self, texts: list[str], **kwargs: object) -> Array:
                owner.before_encode()
                owner.calls.append(("embed", self.identity, list(texts), kwargs["batch_size"]))
                if owner.cpu_error and owner.server._fast_lock.locked():
                    owner.cpu_error = False
                    raise RuntimeError("synthetic transient CPU failure")
                if owner.embed_oom:
                    owner.embed_oom = False
                    raise RuntimeError("MPS backend out of memory")
                return Array([[float(text[1:])] * owner.width for text in texts], (len(texts), owner.width))

            def to(self, device: str) -> None:
                self.device = device

        class Reranker:
            def __init__(self, name: str, device: str | None = None) -> None:
                self.name, self.device = name, device
                owner.builds.append((name, device))

            def predict(self, pairs: list, **kwargs: object) -> Array:
                owner.before_predict()
                owner.calls.append(("rerank", self.name, list(pairs), kwargs["batch_size"]))
                if owner.rerank_oom:
                    owner.rerank_oom = False
                    raise RuntimeError("MPS backend out of memory")
                return Array([[float(pair[1][1:])] for pair in pairs], (len(pairs), 1))

        self.Model = Model

        def build(identity: str, device: str | None) -> Model:
            self.assertTrue(self.server._inference_lock.locked() or self.server._fast_lock.locked())
            self.build_hook()
            self.builds.append((identity, device))
            return Model(identity, device)

        self.server._build_embed_model = build
        self.fixture.mps.auto_detect_device = lambda: "cpu"
        self.fixture.mps.is_mps_oom = lambda exc: "MPS backend out of memory" in str(exc)
        modules = patch.dict(sys.modules, {
            "sentence_transformers": types.SimpleNamespace(CrossEncoder=Reranker),
            "truememory.reranker": types.SimpleNamespace(get_current_reranker_name=lambda: "synthetic/default"),
        })
        modules.start()
        self.addCleanup(modules.stop)

    def embed(self, count: int = 1, *, target=None, **fields: object) -> dict:
        target = self.edge if target is None else target
        return self.server.handle_request({
            "op": "embed", "tier": target.model_id,
            "expected_target": target.to_wire(), "texts": [f"s{i}" for i in range(count)], **fields,
        })

    def rerank(self, count: int = 1, **fields: object) -> dict:
        return self.server.handle_request({
            "op": "rerank", "model_name": "synthetic/reranker", "expected_model_name": "synthetic/reranker",
            "pairs": [["query", f"s{i}"] for i in range(count)], **fields,
        })

    def assert_released(self) -> None:
        self.assertFalse(self.server._inference_lock.locked())
        self.assertFalse(self.server._fast_lock.locked())
        self.assertFalse(self.server._lock.locked())
        self.assertEqual(self.server._inflight, 0)

    def test_constructors_are_lazy_and_embedding_is_a_remote_proxy(self) -> None:
        self.client._request_with_autostart = Mock(side_effect=AssertionError("unexpected request"))
        proxy = self.client.CertifiedEmbeddingProxy(self.edge)
        self.assertIsInstance(proxy, self.client.EmbeddingProxy)
        self.client.CertifiedRerankerProxy("synthetic/reranker")
        self.client._request_with_autostart.assert_not_called()
        self.assertEqual(self.builds, [])

    def test_certified_client_uses_existing_operation_batch_and_deadline(self) -> None:
        calls = []

        def send(request: dict, timeout=None) -> dict:
            calls.append((request, timeout))
            return self.server.handle_request(request)

        self.client._request_with_autostart = send
        proxy = self.client.CertifiedEmbeddingProxy(self.edge)
        output = proxy.encode(["s0", "s1", "s2"], timeout=7, batch_size=2, show_progress_bar=False)
        self.assertEqual(output.rows, [[float(i)] * 256 for i in range(3)])
        self.assertEqual(calls[0][0]["op"], "embed_batched")
        self.assertEqual(calls[0][0]["tier"], "model2vec")
        self.assertEqual(calls[0][0]["expected_target"], self.edge.to_wire())
        self.assertEqual(calls[0][1], 7)
        self.assertEqual([len(call[2]) for call in self.calls], [2, 1])
        self.assert_released()

    def test_each_response_rejects_same_width_identity_drift_after_success(self) -> None:
        proxy = self.client.CertifiedEmbeddingProxy(self.edge)
        output = proxy.encode("s0")
        self.assertEqual(output.shape, (1, 256))
        for protocol in (1, None):
            for receipt in (None, self.base.to_wire()):
                response = {"ok": True, "vectors": output, "target": receipt}
                if protocol is not None:
                    response["protocol"] = protocol
                self.client._request_with_autostart = Mock(return_value=response)
                with self.subTest(protocol=protocol, receipt=receipt), self.assertRaises(self.client.ProtocolMismatchError):
                    proxy.encode("s0")
                self.client._request_with_autostart.assert_called_once()

    def test_embedding_receipt_protocol_and_descriptor_types_are_strict(self) -> None:
        response = self.embed()
        mutations = [
            {"protocol": value} for value in (None, True, 1.0, "1", 2)
        ] + [{"target": {**self.edge.to_wire(), **change}} for change in (
            {"version": True}, {"version": 2}, {"dimension": True}, {"extra": 1}, {"tier": "base"},
        )]
        for mutation in mutations:
            self.client._request_with_autostart = Mock(return_value={**response, **mutation})
            with self.subTest(mutation=mutation), self.assertRaises(self.client.ProtocolMismatchError):
                self.client.CertifiedEmbeddingProxy(self.edge).encode("s0")
        self.client._request_with_autostart = Mock(return_value={**response, "vectors": Array([[1.0]], (1, 1))})
        with self.assertRaises(self.fixture.target_module.EmbeddingTargetError):
            self.client.CertifiedEmbeddingProxy(self.edge).encode("s0")

    def test_custom_and_noncanonical_requests_refuse_before_native(self) -> None:
        custom = self.Target.capture("custom")
        with self.assertRaises(ValueError):
            self.client.CertifiedEmbeddingProxy(custom)
        for fields in ({"tier": "edge"}, {"tier": "qwen3_256"}, {"expected_target": custom.to_wire()},
                       {"expected_target": None}, {"texts": [object()]}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.embed(**fields)
        self.assertEqual(self.builds, [])
        self.assertEqual(self.calls, [])
        self.assert_released()

    def test_direct_main_fast_receipt_preserves_sustained_bookkeeping(self) -> None:
        before = self.fixture.globals()
        response = self.embed()
        self.assertEqual(response["target"], self.edge.to_wire())
        self.assertEqual(response["protocol"], 1)
        self.assertEqual(self.server._embed_timestamps, [])
        self.assertEqual(self.fixture.globals(), before)
        self.assertEqual(len(self.calls), 1)
        self.assert_released()

    def test_cpu_fast_receipt_and_cache_reuse_while_main_owner_busy(self) -> None:
        self.embed()
        self.server._inference_lock.acquire()
        try:
            first = self.embed()
            second = self.embed()
        finally:
            self.server._inference_lock.release()
        self.assertEqual(first["target"], self.edge.to_wire())
        self.assertEqual(second["target"], self.edge.to_wire())
        self.assertEqual(self.builds, [("model2vec", "cpu"), ("model2vec", "cpu")])
        self.assertEqual(self.server._embed_timestamps, [])
        self.assert_released()

    def test_cold_fast_decline_preserves_main_fallback_and_receipt(self) -> None:
        self.server._inference_lock = FirstBusyLock()
        response = self.embed()
        self.assertEqual(response["target"], self.edge.to_wire())
        self.assertEqual(len(self.server._embed_timestamps), 1)
        self.assertEqual(len(self.calls), 1)
        self.assert_released()

    def test_ordinary_cpu_conversion_keeps_its_original_post_owner_placement(self) -> None:
        self.embed()
        checked = self.ms._checked_batch
        ownership = []

        def observe(*args):
            ownership.append(self.server._fast_lock.locked())
            return checked(*args)

        self.ms._checked_batch = observe
        self.server._inference_lock.acquire()
        try:
            ordinary = self.server.handle_request({"op": "embed", "tier": "model2vec", "texts": ["s0"]})
            selected = self.embed()
        finally:
            self.server._inference_lock.release()
        self.assertEqual(ownership, [False, True])
        self.assertEqual(ordinary["vectors"].rows, selected["vectors"].rows)
        self.assertNotIn("target", ordinary)
        self.assertEqual(selected["target"], self.edge.to_wire())
        self.assert_released()

    def test_transient_cpu_fast_failure_keeps_verified_main_fallback(self) -> None:
        self.embed()
        self.server._inference_lock = FirstBusyLock()
        self.cpu_error = True
        response = self.embed()
        self.assertEqual(response["target"], self.edge.to_wire())
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(self.server._embed_timestamps), 1)
        self.assert_released()

    def test_generation_changes_during_cpu_construction_do_not_relabel_output(self) -> None:
        self.embed()
        original_generation = self.server._embed_state.generation

        def advance() -> None:
            if self.server._fast_lock.locked():
                with self.server._lock:
                    self.server._publish_embed_state(self.ms._EmbedState(self.Model("qwen3_256"), "qwen3_256", "qwen3_256"))
                    self.server._publish_embed_state(self.ms._EmbedState(self.Model("model2vec"), "model2vec", "model2vec"))

        self.build_hook = advance
        self.server._inference_lock.acquire()
        try:
            response = self.embed()
        finally:
            self.server._inference_lock.release()
        self.assertEqual(response["target"], self.edge.to_wire())
        self.assertIsNot(self.server._embed_state.generation, original_generation)
        self.assertIsNone(self.server._fast_encoder)
        self.assert_released()

    def test_effective_main_and_fast_identity_mismatch_never_encodes_or_falls_back(self) -> None:
        for fast in (False, True):
            self.server._embed_state = self.ms._EmbedState(self.Model("qwen3_256"), "model2vec", "qwen3_256")
            if fast:
                self.server._inference_lock.acquire()
            try:
                with self.subTest(fast=fast), self.assertRaises(ValueError):
                    self.embed()
            finally:
                if fast:
                    self.server._inference_lock.release()
            self.assertEqual(self.calls, [])
            self.assertEqual(self.server._embed_timestamps, [])
            self.assert_released()

    def test_fast_cached_identity_mismatch_and_width_failure_never_fall_back(self) -> None:
        self.embed()
        self.server._inference_lock.acquire()
        try:
            self.embed()
            self.server._fast_model_id = "qwen3_256"
            with self.assertRaises(ValueError):
                self.embed()
            self.server._fast_model_id = "model2vec"
            self.width = 1
            with self.assertRaises(ValueError):
                self.embed()
        finally:
            self.server._inference_lock.release()
        self.assertEqual(self.server._embed_timestamps, [])
        self.assert_released()

    def test_empty_selected_embedding_requires_receipt_without_added_probe(self) -> None:
        response = self.embed(0)
        self.assertEqual(response["vectors"].shape, (0, 256))
        self.assertEqual(response["target"], self.edge.to_wire())
        self.assertEqual([call[2] for call in self.calls], [[]])
        self.client._request_with_autostart = Mock(return_value={"ok": True, "vectors": response["vectors"], "protocol": 1})
        with self.assertRaises(self.client.ProtocolMismatchError):
            self.client.CertifiedEmbeddingProxy(self.edge).encode([])

    def test_embedding_oom_fast_and_main_retry_keep_receipt_order_and_owner(self) -> None:
        for count in (1, 5):
            self.calls.clear()
            self.embed_oom = True
            response = self.embed(count, op="embed_batched", batch_size=2)
            self.assertEqual(response["target"], self.edge.to_wire())
            self.assertEqual(response["vectors"].rows, [[float(i)] * 256 for i in range(count)])
            self.assertEqual([len(call[2]) for call in self.calls], [1, 1] if count == 1 else [2, 2, 2, 1])
            self.assertIn("embed", self.server._sticky_cpu)
            self.assert_released()

    def test_every_main_slice_checks_width_and_never_returns_partial_output(self) -> None:
        def change_width() -> None:
            if self.calls:
                self.width = 1

        self.before_encode = change_width
        with self.assertRaises(ValueError):
            self.embed(5, op="embed_batched", batch_size=2)
        self.assertEqual([len(call[2]) for call in self.calls], [2, 2])
        self.assert_released()

    def test_admission_and_deadline_refusals_have_no_receipt_or_fallback(self) -> None:
        response = self.embed(deadline=1)
        self.assertFalse(response["ok"])
        self.assertNotIn("target", response)
        self.assertEqual(self.calls, [])
        self.embed()
        self.server._inference_lock.acquire()
        self.server._check_process_memory = Mock(side_effect=self.ms._ProcessMemoryRefused("synthetic capacity"))
        try:
            response = self.embed()
        finally:
            self.server._inference_lock.release()
        self.assertEqual(response["error_code"], "server_busy")
        self.assertNotIn("target", response)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.server._embed_timestamps, [])
        self.assert_released()

    def test_ordinary_clients_and_responses_remain_unattested(self) -> None:
        self.client._request_with_autostart = Mock(return_value={"ok": True, "vectors": "legacy", "scores": "legacy"})
        self.assertEqual(self.client.EmbeddingProxy("edge").encode("s0"), "legacy")
        self.assertNotIn("expected_target", self.client._request_with_autostart.call_args.args[0])
        self.assertEqual(self.client.RerankerProxy().predict([["q", "s0"]]), "legacy")
        self.assertNotIn("expected_model_name", self.client._request_with_autostart.call_args.args[0])
        response = self.server.handle_request({"op": "embed", "tier": "model2vec", "texts": ["s0"]})
        self.assertEqual(set(response), {"ok", "vectors"})
        response = self.server.handle_request({"op": "rerank", "pairs": [["q", "s0"]]})
        self.assertEqual(set(response), {"ok", "scores"})

    def test_reranker_every_response_certifies_explicit_name_and_bounded_order(self) -> None:
        proxy = self.client.CertifiedRerankerProxy("synthetic/reranker")
        response = proxy.predict([["q", f"s{i}"] for i in range(5)], timeout=3, batch_size=2)
        self.assertEqual(response.rows, [[float(i)] for i in range(5)])
        self.assertEqual([len(call[2]) for call in self.calls], [2, 2, 1])
        self.assertEqual(self.builds, [("synthetic/reranker", "cpu")])
        for receipt in (None, {"version": 1, "model_name": "synthetic/default"},
                        {"version": True, "model_name": "synthetic/reranker"},
                        {"version": 1, "model_name": "synthetic/reranker", "extra": 1}):
            self.client._request_with_autostart = Mock(return_value={"ok": True, "scores": response, "protocol": 1, "reranker": receipt})
            with self.subTest(receipt=receipt), self.assertRaises(self.client.ProtocolMismatchError):
                proxy.predict([["q", "s0"]])
            self.client._request_with_autostart.assert_called_once()
        self.assert_released()

    def test_reranker_receipt_requires_one_result_row_per_pair(self) -> None:
        proxy = self.client.CertifiedRerankerProxy("synthetic/reranker")
        pairs = [["q", "s0"], ["q", "s1"]]
        receipt = {"version": 1, "model_name": "synthetic/reranker"}
        for scores in (None, [0., 1.], Array([], ()), Array([[0.]], (1, 1)),
                       Array([[0.], [1.], [2.]], (3, 1))):
            self.client._request_with_autostart = Mock(return_value={
                "ok": True, "scores": scores, "protocol": 1, "reranker": receipt,
            })
            with self.subTest(shape=getattr(scores, "shape", None)), self.assertRaises(self.client.ProtocolMismatchError):
                proxy.predict(iter(pairs))
            self.client._request_with_autostart.assert_called_once()
        for shape in ((2,), (2, 1), (2, 2)):
            scores = Array([[0.], [1.]], shape)
            self.client._request_with_autostart = Mock(return_value={
                "ok": True, "scores": scores, "protocol": 1, "reranker": receipt,
            })
            with self.subTest(shape=shape):
                self.assertIs(proxy.predict(iter(pairs)), scores)
        self.assert_released()

    def test_empty_reranker_and_missing_protocol_are_strict(self) -> None:
        response = self.rerank(0)
        self.assertEqual(response["scores"].shape, (0, 1))
        self.assertEqual(response["reranker"], {"version": 1, "model_name": "synthetic/reranker"})
        self.assertEqual([call[2] for call in self.calls], [[]])
        for value in (None, True, 1.0, "1", 2):
            self.client._request_with_autostart = Mock(return_value={**response, "protocol": value})
            with self.subTest(protocol=value), self.assertRaises(self.client.ProtocolMismatchError):
                self.client.CertifiedRerankerProxy("synthetic/reranker").predict([])

    def test_reranker_explicit_identity_validation_and_effective_cache_mismatch(self) -> None:
        for name in (None, "", True, "x" * 513, "synthetic/model/other", "bad\nname"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.client.CertifiedRerankerProxy(name)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.rerank(expected_model_name=name)
        with self.assertRaises(ValueError):
            self.rerank(model_name=None)
        self.assertEqual(self.builds, [])
        real_get = self.server._get_reranker

        def wrong(name=None, deadline=None):
            model = real_get(name, deadline)
            self.server._reranker_name = "synthetic/different"
            return model

        self.server._get_reranker = wrong
        with self.assertRaises(ValueError):
            self.rerank()
        self.assertEqual(self.calls, [])
        self.assert_released()

    def test_reranker_cpu_replacement_is_revalidated_before_retry(self) -> None:
        self.rerank_oom = True
        response = self.rerank(5, op="rerank_batched", batch_size=2)
        self.assertEqual(response["reranker"], {"version": 1, "model_name": "synthetic/reranker"})
        self.assertEqual(response["scores"].rows, [[float(i)] for i in range(5)])
        self.assertEqual([len(call[2]) for call in self.calls], [2, 2, 2, 1])
        self.assertEqual([name for name, _device in self.builds], ["synthetic/reranker"] * 2)
        real_get = self.server._get_reranker

        def wrong_retry(name=None, deadline=None):
            replacing = self.server._reranker is None
            model = real_get(name, deadline)
            if replacing:
                self.server._reranker_name = "synthetic/other"
            return model

        self.server._get_reranker = wrong_retry
        self.rerank_oom = True
        before = len(self.calls)
        with self.assertRaises(ValueError):
            self.rerank()
        self.assertEqual(len(self.calls), before + 1)
        self.assert_released()

    def test_public_tier_receipt_does_not_collapse_base_and_pro(self) -> None:
        response = self.embed(target=self.base)
        model = self.server._embed_state.model
        pro = dataclasses.replace(self.base, tier="pro")
        second = self.embed(target=pro)
        self.assertEqual(response["target"], self.base.to_wire())
        self.assertEqual(second["target"], pro.to_wire())
        self.assertIs(self.server._embed_state.model, model)
        self.client._request_with_autostart = Mock(return_value=second)
        with self.assertRaises(self.client.ProtocolMismatchError):
            self.client.CertifiedEmbeddingProxy(self.base).encode("s0")


if __name__ == "__main__":
    unittest.main()
