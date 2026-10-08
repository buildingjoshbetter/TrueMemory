"""Synthetic, event-driven ownership regressions; no models or server sockets."""
from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

from truememory import model_server as ms


class ObservedLock:
    def __init__(self, attempted: threading.Event) -> None:
        self.lock = threading.Lock()
        self.attempted = attempted

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if blocking and threading.current_thread().name.startswith("competitor"):
            self.attempted.set()
        return self.lock.acquire(blocking, timeout)

    def release(self) -> None:
        self.lock.release()

    def locked(self) -> bool:
        return self.lock.locked()


class TrackedModel:
    def __init__(self, attempted: threading.Event) -> None:
        self.attempted = attempted
        self.started = threading.Event()
        self.release = threading.Event()
        self.guard = threading.Lock()
        self.calls: list[list] = []
        self.active = 0
        self.peak = 0
        self.errors: dict[int, Exception] = {}
        self.block_on: int | None = None
        self.moves: list[str] = []
        self.unsafe_moves = 0
        self.move_error: Exception | None = None

    def _call(self, items: list) -> None:
        with self.guard:
            self.calls.append(list(items))
            call = len(self.calls)
            self.active += 1
            self.peak = max(self.peak, self.active)
            if threading.current_thread().name.startswith("competitor"):
                self.attempted.set()
        try:
            if call in self.errors:
                raise self.errors[call]
            if call == self.block_on:
                self.started.set()
                if not self.release.wait(timeout=5):
                    raise AssertionError("Synthetic inference was not released")
        finally:
            with self.guard:
                self.active -= 1

    def encode(self, texts: list[str], **_kwargs: object) -> list[list[float]]:
        self._call(texts)
        return [[float(len(text)), 0.5] for text in texts]

    def predict(self, pairs: list[tuple[str, str]], **_kwargs: object) -> list[float]:
        self._call(pairs)
        return [float(len(document)) for _, document in pairs]

    def to(self, device: str) -> None:
        with self.guard:
            self.unsafe_moves += int(self.active > 0)
            self.moves.append(device)
        if self.move_error is not None:
            raise self.move_error


class TestIssue738InferenceOwnership(unittest.TestCase):
    def setUp(self) -> None:
        self.threads: list[threading.Thread] = []
        self.models: list[TrackedModel] = []
        self.reset_runtime()
        mps = types.ModuleType("truememory.mps_utils")
        mps.is_mps_oom = lambda error: "MPS backend out of memory" in str(error)
        mps.flush_mps_cache = lambda: None
        mps.auto_detect_device = lambda: "mps"
        mps.resolve_device = lambda device: device
        mps.ensure_mps_memory_budget = lambda device: None
        reranker = types.ModuleType("truememory.reranker")
        reranker.get_current_reranker_name = lambda: "primary"
        transformers = types.ModuleType("sentence_transformers")

        def cross_encoder(name: str, device: str | None) -> TrackedModel:
            self.loads.append(("rerank", name, device))
            return self.rerank_cpu if name == "primary" else self.alternate

        transformers.CrossEncoder = cross_encoder
        patcher = patch.dict(sys.modules, {
            "truememory.mps_utils": mps, "truememory.reranker": reranker,
            "sentence_transformers": transformers,
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.finish_threads)

    def reset_runtime(self) -> None:
        self.attempted = threading.Event()
        self.server = ms.ModelServer()
        self.server._inference_lock = ObservedLock(self.attempted)
        self.server._write_status_file = lambda: None
        self.loads: list[tuple[str, str, str | None]] = []
        self.embed = TrackedModel(self.attempted)
        self.rerank_mps = TrackedModel(self.attempted)
        self.rerank_cpu = TrackedModel(self.attempted)
        self.fast = TrackedModel(self.attempted)
        self.alternate = TrackedModel(self.attempted)
        self.models.extend((self.embed, self.rerank_mps, self.rerank_cpu,
                            self.fast, self.alternate))
        self.server._embed_state = ms._EmbedState(self.embed, "primary", "qwen3_256")
        self.server._reranker = self.rerank_mps
        self.server._reranker_name = "primary"
        self.server._resolve_embed_model_id = lambda tier: (
            "qwen3_256" if tier == "primary" else "synthetic-alternate"
        )

        def build(model_id: str, device: str | None) -> TrackedModel:
            self.loads.append(("embed", model_id, device))
            return self.fast if model_id == "qwen3_256" else self.alternate

        self.server._build_embed_model = build

    def finish_threads(self) -> None:
        for model in self.models:
            model.release.set()
        for thread in self.threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive(), "Synthetic worker did not finish")

    def request(self, op: str, count: int = 4, identity: str = "primary") -> dict:
        texts = [f"fixture-{index}" for index in range(count)]
        return {"op": op, "texts": texts, "pairs": [("fixture-query", text) for text in texts],
                "tier": identity, "model_name": identity, "batch_size": 2}

    def start(self, request: dict, name: str) -> tuple[threading.Thread, dict]:
        outcome: dict = {}

        def run() -> None:
            try:
                outcome["response"] = self.server.handle_request(request)
            except Exception as error:
                outcome["error"] = error

        thread = threading.Thread(target=run, name=name, daemon=True)
        self.threads.append(thread)
        thread.start()
        return thread, outcome

    def assert_finished(self, worker: tuple[threading.Thread, dict]) -> dict:
        thread, outcome = worker
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive(), "Request did not finish")
        self.assertNotIn("error", outcome)
        return outcome["response"]

    def block_retry(self, op: str) -> tuple[TrackedModel, tuple[threading.Thread, dict]]:
        failing = self.embed if op == "embed" else self.rerank_mps
        failing.errors[1] = RuntimeError("MPS backend out of memory")
        retrying = self.embed if op == "embed" else self.rerank_cpu
        retrying.block_on = 2 if op == "embed" else 1
        owner = self.start(self.request(op), "owner")
        self.assertTrue(retrying.started.wait(timeout=2), "CPU retry did not start")
        return retrying, owner

    def assert_retry_owned_with_responsive_fast_lane(self, op: str) -> None:
        retrying, owner = self.block_retry(op)
        competitor = self.start(self.request(op, count=2), "competitor-main")
        self.assertTrue(self.attempted.wait(timeout=2))
        self.assertEqual(retrying.peak, 1, "CPU retry overlapped same-instance inference")
        acquired = self.server._lock.acquire(blocking=False)
        self.assertTrue(acquired, "CPU retry retained the state lock")
        if acquired:
            self.server._lock.release()
        query = self.start(self.request("embed", count=1), "query")
        self.assertTrue(self.assert_finished(query)["ok"])
        self.assertEqual(len(self.fast.calls), 1, "Query bypassed the busy inference owner")
        self.assertTrue(owner[0].is_alive())
        self.assertTrue(competitor[0].is_alive())
        retrying.release.set()
        self.assertTrue(self.assert_finished(owner)["ok"])
        self.assertTrue(self.assert_finished(competitor)["ok"])
        self.assertEqual(retrying.peak, 1)
        self.assertEqual(retrying.unsafe_moves, 0)
        self.assertEqual(self.server._inflight, 0)

    def test_issue_738_embed_retry_owns_model_while_fast_query_completes(self) -> None:
        self.assert_retry_owned_with_responsive_fast_lane("embed")

    def test_issue_738_rerank_retry_owns_reloaded_model_while_fast_query_completes(self) -> None:
        self.assert_retry_owned_with_responsive_fast_lane("rerank")

    def assert_replacement_waits_for_owner(self, op: str) -> None:
        retrying, owner = self.block_retry(op)
        previous_loads = list(self.loads)
        competitor = self.start(self.request(op, count=2, identity="alternate"), "competitor-tier")
        self.assertTrue(self.attempted.wait(timeout=2))
        self.assertEqual(self.loads, previous_loads, "Replacement loaded during owned inference")
        self.assertEqual(self.alternate.calls, [])
        retrying.release.set()
        self.assertTrue(self.assert_finished(owner)["ok"])
        self.assertTrue(self.assert_finished(competitor)["ok"])
        self.assertEqual(len(self.loads), len(previous_loads) + 1)
        self.assertEqual(len(self.alternate.calls), 1)

    def test_issue_738_embedding_tier_replacement_waits_for_cpu_retry(self) -> None:
        self.assert_replacement_waits_for_owner("embed")

    def test_issue_738_reranker_replacement_waits_for_cpu_retry(self) -> None:
        self.assert_replacement_waits_for_owner("rerank")

    def test_issue_738_main_slot_also_serializes_different_model_kinds(self) -> None:
        retrying, owner = self.block_retry("embed")
        competitor = self.start(self.request("rerank", count=2), "competitor-rerank")
        self.assertTrue(self.attempted.wait(timeout=2))
        self.assertEqual(self.rerank_mps.calls, [])
        retrying.release.set()
        self.assertTrue(self.assert_finished(owner)["ok"])
        self.assertTrue(self.assert_finished(competitor)["ok"])

    def test_issue_738_fast_encoder_has_one_owner_and_one_cached_instance(self) -> None:
        self.fast.block_on = 1
        self.server._fast_lock = ObservedLock(self.attempted)
        self.server._inference_lock.acquire()
        try:
            first = self.start(self.request("embed", count=1), "first-fast")
            self.assertTrue(self.fast.started.wait(timeout=2))
            second = self.start(self.request("embed", count=1), "competitor-fast")
            self.assertTrue(self.attempted.wait(timeout=2))
            self.assertEqual(self.fast.peak, 1)
            self.assertEqual(self.loads, [("embed", "qwen3_256", "cpu")])
            self.fast.release.set()
            self.assertTrue(self.assert_finished(first)["ok"])
            self.assertTrue(self.assert_finished(second)["ok"])
            self.assertEqual(self.fast.peak, 1)
            self.assertEqual(len(self.fast.calls), 2)
            self.assertEqual(len(self.loads), 1)
        finally:
            self.server._inference_lock.release()

    def test_issue_738_deadline_waiter_cannot_release_owner_or_start_model_work(self) -> None:
        for op in ("embed", "rerank"):
            with self.subTest(op=op):
                self.reset_runtime()
                retrying, owner = self.block_retry(op)
                calls = len(retrying.calls)
                loads = list(self.loads)
                request = self.request(op, count=2)
                request["deadline"] = time.time() + 0.05
                waiter = self.start(request, "competitor-deadline")
                response = self.assert_finished(waiter)
                self.assertFalse(response["ok"])
                self.assertIn("deadline", response["error"])
                self.assertTrue(self.server._inference_lock.locked())
                self.assertEqual(len(retrying.calls), calls)
                self.assertEqual(self.loads, loads)
                self.assertTrue(owner[0].is_alive())
                retrying.release.set()
                self.assertTrue(self.assert_finished(owner)["ok"])
                self.assertEqual(self.server._inflight, 0)

    def test_issue_738_exceptions_release_inference_and_state_locks(self) -> None:
        for stage in ("embed-load", "embed-call", "embed-move", "embed-retry",
                      "single-call", "rerank-call", "rerank-retry", "rerank-load"):
            with self.subTest(stage=stage):
                self.reset_runtime()
                error = ValueError("synthetic failure")

                def fail(*_args: object, **_kwargs: object) -> None:
                    raise error

                op = "rerank" if stage.startswith("rerank") else "embed"
                if stage == "embed-load":
                    self.server._get_embed_model = fail
                elif stage in ("embed-call", "single-call"):
                    self.embed.errors[1] = error
                elif stage.startswith("embed"):
                    self.embed.errors[1] = RuntimeError("MPS backend out of memory")
                    if stage == "embed-move":
                        self.embed.move_error = error
                    else:
                        self.embed.errors[2] = error
                elif stage == "rerank-call":
                    self.rerank_mps.errors[1] = error
                else:
                    self.rerank_mps.errors[1] = RuntimeError("MPS backend out of memory")
                    if stage == "rerank-load":
                        sys.modules["sentence_transformers"].CrossEncoder = fail
                    else:
                        self.rerank_cpu.errors[1] = error
                with self.assertRaisesRegex(ValueError, "synthetic failure"):
                    self.server.handle_request(self.request(op, count=1 if stage == "single-call" else 2))
                for lock in (self.server._inference_lock, self.server._lock, self.server._fast_lock):
                    self.assertTrue(lock.acquire(blocking=False))
                    lock.release()
                self.assertEqual(self.server._inflight, 0)

    def test_issue_738_fast_failure_releases_fast_owner_before_waiting_for_main(self) -> None:
        self.fast.errors[1] = ValueError("synthetic fast failure")
        self.server._inference_lock.acquire()
        try:
            waiter = self.start(self.request("embed", count=1), "competitor-fallback")
            self.assertTrue(self.attempted.wait(timeout=2))
            self.assertTrue(self.server._fast_lock.acquire(blocking=False))
            self.server._fast_lock.release()
            self.assertTrue(waiter[0].is_alive())
        finally:
            self.server._inference_lock.release()
        self.assertTrue(self.assert_finished(waiter)["ok"])
        self.assertEqual(len(self.embed.calls), 1)

    def test_issue_738_busy_state_lock_does_not_pin_main_permit_during_fast_work(self) -> None:
        self.server._lock.acquire()
        try:
            query = self.start(self.request("embed", count=1), "query")
            self.assertTrue(self.assert_finished(query)["ok"])
            self.assertEqual(len(self.fast.calls), 1)
            self.assertFalse(self.server._inference_lock.locked())
        finally:
            self.server._lock.release()
