"""Actual model loaders and handlers with stdlib lifetimes and event barriers."""
from __future__ import annotations

import ast
import importlib.util
import sys
import threading
import types
import unittest
import weakref
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "truememory" / "model_server.py"
SOURCE_TEXT: str | None = None
SUPPORT_PATH = Path(__file__).with_name("test-model-allocation-preflight-297.py")
SPEC = importlib.util.spec_from_file_location("residency_array_support", SUPPORT_PATH)
SUPPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPPORT)


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
    module = types.ModuleType("synthetic_residency_server")
    module.__dict__.update({
        "np": SUPPORT.Numpy(), "psutil": None, "_USE_UNIX": True, "_LOOPBACK_HOST": "127.0.0.1",
        "_env_int": lambda name, default, **kwargs: default, "pid_is_alive": lambda pid: False,
    })
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), module.__dict__)
    module.log.disabled = True
    return module


class MetadataLock:
    """Observe ownership without mistaking another thread's scalar work for ours."""
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.owner: int | None = None

    def acquire(self, blocking: bool = True) -> bool:
        acquired = self.lock.acquire(blocking=blocking)
        if acquired:
            self.owner = threading.get_ident()
        return acquired

    def release(self) -> None:
        self.owner = None
        self.lock.release()

    def locked(self) -> bool:
        return self.lock.locked()

    def owned(self) -> bool:
        return self.owner == threading.get_ident()

    def __enter__(self) -> MetadataLock:
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class Model:
    def __init__(self, owner: TestModelResidency, identity: str, device: str) -> None:
        self.owner, self.identity, self.device = owner, identity, device
        self.value = {"model2vec": 11.0, "qwen3_256": 22.0, "rank-a": 31.0, "rank-b": 32.0}[identity]

    def encode(self, texts: list, **kwargs: object) -> list:
        self.owner.encodes.append((self.identity, self.device, list(texts)))
        self.owner.encode_hook(self)
        return [[self.value] for _ in texts]

    def predict(self, pairs: list, **kwargs: object) -> list:
        self.owner.encodes.append((self.identity, self.device, list(pairs)))
        return [self.value for _ in pairs]

    def to(self, device: str) -> None:
        raise AssertionError("Residency retirement must not move an active model")

    def __del__(self) -> None:
        lock = getattr(self.owner.server, "_residency_lock", None)
        self.owner.destructions.append((self.identity, self.device, lock is not None and lock.owned()))


class TestModelResidency(unittest.TestCase):
    def setUp(self) -> None:
        package = types.ModuleType("truememory")
        package.__path__ = []
        vector = types.ModuleType("truememory.vector_search")
        vector.EMBEDDING_MODEL = "model2vec"
        vector._TIER_ALIASES = {"edge": "model2vec", "base": "qwen3_256", "pro": "qwen3_256"}
        reranker = types.ModuleType("truememory.reranker")
        reranker.get_current_reranker_name = lambda: "rank-a"
        mps = types.ModuleType("truememory.mps_utils")
        mps.auto_detect_device = lambda: "cpu"
        mps.resolve_device = lambda device: device
        mps.ensure_mps_memory_budget = lambda device: None
        mps.is_mps_oom = lambda error: False
        transformers = types.ModuleType("sentence_transformers")
        transformers.CrossEncoder = lambda identity, device: self.build(identity, device)
        self.modules = patch.dict(sys.modules, {
            "truememory": package, "truememory.vector_search": vector,
            "truememory.reranker": reranker, "truememory.mps_utils": mps,
            "sentence_transformers": transformers,
        })
        self.modules.start()
        self.ms = load_server()
        self.server = self.ms.ModelServer()
        if hasattr(self.server, "_residency_lock"):
            self.server._residency_lock = MetadataLock()
        self.server._write_status_file = lambda: None
        self.server._embed_device = lambda: "main"
        self.server._build_embed_model = self.build
        self.refs: list[weakref.ReferenceType] = []
        self.builds: list[tuple[str, str]] = []
        self.encodes: list[tuple] = []
        self.destructions: list[tuple] = []
        self.build_hook: Callable[[str, str], None] = lambda identity, device: None
        self.ready_hook: Callable[[Model], None] = lambda model: None
        self.encode_hook: Callable[[Model], None] = lambda model: None
        self.events: list[threading.Event] = []
        self.threads: list[threading.Thread] = []
        self.failures: list[str] = []

    def tearDown(self) -> None:
        for event in self.events:
            event.set()
        for thread in self.threads:
            thread.join(2)
            self.assertFalse(thread.is_alive(), "Synthetic owner did not finish")
        self.server._workers.shutdown(wait=True, cancel_futures=True)
        self.server._embed_state = None
        self.server._reranker = None
        self.server._fast_encoder = None
        self.modules.stop()
        self.assertEqual(self.failures, [])
        self.assertFalse(any(locked for _, _, locked in self.destructions), self.destructions)

    def build(self, identity: str, device: str) -> Model:
        lock = getattr(self.server, "_residency_lock", None)
        self.assertFalse(lock is not None and lock.owned(), "Constructor owns residency metadata")
        self.build_hook(identity, device)
        model = Model(self, identity, device)
        self.refs.append(weakref.ref(model))
        self.builds.append((identity, device))
        self.ready_hook(model)
        return model

    def request(self, op: str = "embed", tier: str = "edge", name: str = "rank-a") -> dict:
        return self.server.handle_request({
            "op": op, "tier": tier, "texts": ["one", "two"],
            "pairs": [("query", "one"), ("query", "two")], "model_name": name,
        })

    def fast(self, tier: str = "edge", deadline: object = None) -> dict | None:
        return self.server._handle_fast_embed(["query"], tier, deadline)

    def main_while_owned(self, tier: str) -> None:
        with self.server._lock:
            self.server._get_embed_model(tier)

    def event(self) -> threading.Event:
        event = threading.Event()
        self.events.append(event)
        return event

    def start(self, action: Callable[[], object]) -> tuple[threading.Thread, list]:
        results: list = []

        def run() -> None:
            try:
                results.append(action())
            except Exception as exc:
                self.failures.append(f"{type(exc).__name__}: {exc}")

        thread = threading.Thread(target=run, daemon=True)
        self.threads.append(thread)
        thread.start()
        return thread, results

    def finish(self, thread: threading.Thread) -> None:
        thread.join(2)
        self.assertFalse(thread.is_alive(), "Owner waited for another lane")
        self.assertEqual(self.failures, [])

    def test_main_replacement_releases_old_before_actual_handler_constructor(self) -> None:
        self.assertEqual(self.request()["vectors"].data, [11.0, 11.0])
        old = self.refs[-1]

        def check(identity: str, device: str) -> None:
            self.assertIsNone(old(), "Obsolete main remains live during replacement")
            self.assertTrue(self.server._lock.locked())
            self.assertTrue(self.server._inference_lock.locked())

        self.build_hook = check
        self.assertEqual(self.request(tier="base")["vectors"].data, [22.0, 22.0])

    def test_reranker_replacement_releases_old_before_actual_handler_constructor(self) -> None:
        self.assertEqual(self.request("rerank")["scores"].data, [31.0, 31.0])
        old = self.refs[-1]
        self.build_hook = lambda identity, device: self.assertIsNone(
            old(), "Obsolete reranker remains live during replacement",
        )
        self.assertEqual(self.request("rerank", name="rank-b")["scores"].data, [32.0, 32.0])

    def test_failed_replacements_leave_empty_cache_and_later_reload_old_identity(self) -> None:
        for op in ("embed", "rerank"):
            with self.subTest(op=op):
                self.build_hook = lambda identity, device: None
                self.request(op)
                old = self.refs[-1]

                def fail(identity: str, device: str) -> None:
                    self.assertIsNone(old())
                    raise RuntimeError("synthetic construction failure")

                self.build_hook = fail
                with self.assertRaisesRegex(RuntimeError, "synthetic construction failure"):
                    self.request(op, tier="base", name="rank-b")
                self.assertIsNone(self.server._embed_state if op == "embed" else self.server._reranker)
                if op == "rerank":
                    self.assertIsNone(self.server._reranker_name)
                self.assertFalse(self.server._lock.locked())
                self.assertFalse(self.server._inference_lock.locked())
                self.build_hook = lambda identity, device: None
                self.assertTrue(self.request(op)["ok"])
                self.assertIsNotNone(self.refs[-1]())

    def test_idle_fast_clone_retires_before_distinct_main_constructor(self) -> None:
        self.request()
        main = self.refs[-1]
        with self.server._inference_lock:
            self.assertEqual(self.fast()["vectors"].data, [11.0])
            fast = self.refs[-1]

            def check(identity: str, device: str) -> None:
                self.assertIsNone(fast(), "Idle obsolete fast clone remains resident")
                self.assertIsNone(main())

            self.build_hook = check
            self.main_while_owned("base")
        self.assertIsNone(self.server._fast_encoder)
        self.assertIsNone(self.server._fast_model_id)

    def test_alias_retag_preserves_main_generation_and_fast_cache(self) -> None:
        self.request(tier="base")
        generation = self.server._embed_state.generation
        with self.server._inference_lock:
            self.assertTrue(self.fast("base")["ok"])
            clone = self.refs[-1]
            self.main_while_owned("pro")
            self.assertIs(self.server._embed_state.generation, generation)
            self.assertIs(self.server._fast_encoder, clone())
            self.assertEqual(self.fast("pro")["vectors"].data, [22.0])
        self.assertEqual(self.builds, [("qwen3_256", "main"), ("qwen3_256", "cpu")])

    def test_fast_construction_releases_snapshot_and_cannot_publish_after_a_b_a(self) -> None:
        for return_to_a in (False, True):
            with self.subTest(return_to_a=return_to_a):
                self.request()
                old = weakref.ref(self.server._embed_state.model)
                started, release = self.event(), self.event()

                def block(model: Model) -> None:
                    if model.device == "cpu":
                        started.set()
                        if not release.wait(2):
                            raise AssertionError("Constructor barrier timed out")

                self.ready_hook = block
                with self.server._inference_lock:
                    thread, results = self.start(self.fast)
                    self.assertTrue(started.wait(2))
                    clone = self.refs[-1]
                    self.main_while_owned("base")
                    self.assertIsNone(old(), "Fast constructor retained main snapshot")
                    if return_to_a:
                        self.main_while_owned("edge")
                    self.encode_hook = lambda model: self.assertIsNone(
                        self.server._fast_encoder, "Stale construction published as current cache",
                    )
                    release.set()
                    self.finish(thread)
                    self.assertIsNotNone(results[0], "Captured fast request lost its correct output")
                    self.assertEqual(results[0]["vectors"].data, [11.0])
                    self.assertIsNone(clone())
                    self.assertIsNone(self.server._fast_encoder)
                self.encode_hook = lambda model: None
                self.ready_hook = lambda model: None

    def test_active_stale_fast_request_finishes_or_fails_before_clone_retirement(self) -> None:
        for fail in (False, True):
            with self.subTest(fail=fail):
                self.request()
                started, release = self.event(), self.event()

                def block(model: Model) -> None:
                    if model.device == "cpu":
                        started.set()
                        if not release.wait(2):
                            raise AssertionError("Inference barrier timed out")
                        if fail:
                            raise RuntimeError("synthetic encode failure")

                self.encode_hook = block
                with self.server._inference_lock:
                    thread, results = self.start(self.fast)
                    self.assertTrue(started.wait(2))
                    clone = self.refs[-1]
                    self.main_while_owned("base")
                    self.assertIs(self.server._fast_encoder, clone())
                    self.assertIsNotNone(clone(), "Active clone evicted")
                    release.set()
                    self.finish(thread)
                    if fail:
                        self.assertEqual(results, [None])
                    else:
                        self.assertEqual(results[0]["vectors"].data, [11.0])
                    self.assertIsNone(self.server._fast_encoder)
                    self.assertIsNone(clone())
                self.encode_hook = lambda model: None

    def test_fast_replacement_drops_old_clone_before_constructor_even_on_failure(self) -> None:
        self.request()
        with self.server._inference_lock:
            self.assertTrue(self.fast()["ok"])
            old = self.refs[-1]
            # Hold the owner slot without inference while the main lane changes.
            with self.server._fast_lock:
                self.main_while_owned("base")
                self.assertIsNotNone(old())

                def fail(identity: str, device: str) -> None:
                    self.assertIsNone(old(), "Old CPU clone overlaps replacement constructor")
                    raise RuntimeError("synthetic CPU construction failure")

                self.build_hook = fail
                with self.assertRaisesRegex(RuntimeError, "synthetic CPU construction failure"):
                    self.server._get_fast_encoder("base")
        self.assertIsNone(self.server._fast_encoder)
        self.assertIsNone(self.server._fast_model_id)

    def test_fast_publication_contention_returns_uncached_correct_model(self) -> None:
        self.request()
        with self.server._fast_lock:
            # Contend only at publication; the constructor must run lock-free.
            self.ready_hook = lambda model: self.server._residency_lock.acquire()
            try:
                model = self.server._get_fast_encoder("edge")
                clone = weakref.ref(model)
                self.assertIsNone(self.server._fast_encoder)
                self.assertEqual(model.encode(["query"]), [[11.0]])
            finally:
                self.server._residency_lock.release()
            model = None
            self.assertIsNone(clone())

    def test_fast_constructor_and_inference_finish_while_both_main_locks_are_owned(self) -> None:
        self.request()
        with self.server._inference_lock, self.server._lock:
            thread, results = self.start(self.fast)
            self.finish(thread)
            self.assertEqual(results[0]["vectors"].data, [11.0])
            self.assertIs(self.server._fast_encoder, self.refs[-1]())
        self.assertEqual(self.builds, [("model2vec", "main"), ("model2vec", "cpu")])

    def test_generation_validation_and_cache_publication_exclude_main_invalidation(self) -> None:
        self.request()
        publishing, release_publication = self.event(), self.event()
        main_attempted, encoding, release_encode = self.event(), self.event(), self.event()
        reads, guarded = [], []
        testcase = self

        class ObservedServer(self.ms.ModelServer):
            def __getattribute__(inner_self, name: str) -> object:
                value = super().__getattribute__(name)
                if name == "_embed_state" and threading.current_thread().name == "residency-fast-publication":
                    reads.append(True)
                    if len(reads) == 2:
                        guarded.append(inner_self._residency_lock.owned())
                        publishing.set()
                        if not release_publication.wait(2):
                            raise AssertionError("Publication barrier timed out")
                return value

        self.server.__class__ = ObservedServer
        acquire = self.server._residency_lock.acquire

        def observe_acquire(blocking: bool = True) -> bool:
            if threading.current_thread().name == "residency-main-replacement":
                main_attempted.set()
            return acquire(blocking)

        self.server._residency_lock.acquire = observe_acquire

        def encode(model: Model) -> None:
            if model.device == "cpu":
                encoding.set()
                if not release_encode.wait(2):
                    raise AssertionError("Publication inference barrier timed out")

        self.encode_hook = encode

        def fast() -> dict | None:
            threading.current_thread().name = "residency-fast-publication"
            return testcase.fast()

        def main() -> dict:
            threading.current_thread().name = "residency-main-replacement"
            return testcase.request(tier="base")

        with self.server._inference_lock:
            fast_thread, fast_results = self.start(fast)
            self.assertTrue(publishing.wait(2))
            self.assertEqual(guarded, [True])
        main_thread, main_results = self.start(main)
        self.assertTrue(main_attempted.wait(2))
        self.assertEqual(self.server._embed_state.model_id, "model2vec")
        release_publication.set()
        self.assertTrue(encoding.wait(2))
        self.finish(main_thread)
        self.assertEqual(main_results[0]["vectors"].data, [22.0, 22.0])
        release_encode.set()
        self.finish(fast_thread)
        self.assertEqual(fast_results[0]["vectors"].data, [11.0])
        self.assertIsNone(self.server._fast_encoder)

    def test_fast_deadline_after_stale_construction_skips_encode_and_releases_clone(self) -> None:
        self.request()
        started, release = self.event(), self.event()
        clock = [1.0]
        deadline = self.ms._RequestDeadline(expires_at=2.0)

        def block(model: Model) -> None:
            if model.device == "cpu":
                started.set()
                if not release.wait(2):
                    raise AssertionError("Deadline barrier timed out")

        self.ready_hook = block

        def expire() -> str:
            try:
                self.fast(deadline=deadline)
            except self.ms._RequestDeadlineExceeded:
                return "expired"
            raise AssertionError("Expired fast request completed")

        with patch.object(self.ms.time, "monotonic", side_effect=lambda: clock[0]):
            with self.server._inference_lock:
                thread, results = self.start(expire)
                self.assertTrue(started.wait(2))
                clone = self.refs[-1]
                self.main_while_owned("base")
                clock[0] = 3.0
                release.set()
                self.finish(thread)
                self.assertEqual(results, ["expired"])
                self.assertIsNone(clone())
                self.assertIsNone(self.server._fast_encoder)
        self.assertEqual(len(self.encodes), 1)
        self.assertFalse(self.server._fast_lock.locked())

    def test_idle_retirement_unlocks_owner_before_allowing_new_publication(self) -> None:
        self.request()
        with self.server._inference_lock:
            self.assertTrue(self.fast()["ok"])
        real = self.server._fast_lock
        testcase = self

        class CheckedRelease:
            def acquire(self, blocking: bool = True) -> bool:
                testcase.assertFalse(testcase.server._residency_lock.locked())
                testcase.assertFalse(blocking)
                return real.acquire(blocking=blocking)

            def release(self) -> None:
                testcase.assertTrue(testcase.server._residency_lock.locked())
                real.release()

        self.server._fast_lock = CheckedRelease()
        self.server._retire_stale_fast_encoder()
        self.assertFalse(real.locked())


if __name__ == "__main__":
    if "--source-stdin" in sys.argv:
        sys.argv.remove("--source-stdin")
        SOURCE_TEXT = sys.stdin.read()
    unittest.main()
