"""Actual source checkpoints and worker flow with in-memory SQL and stub leases."""

import ast
import builtins
import gc
import math
import runpy
import sqlite3
import struct
import sys
import types
import unittest
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEST = runpy.run_path(str(Path(__file__).with_name("test-tier-rebuild-source-756.py")))
TABLES = ("vec_messages_custom", "vec_messages_sep_custom")


class Workspace:
    pass


class Vectors(list):
    pass


class TestStreamedWorker(unittest.TestCase):
    def setUp(self):
        self.modules = SOURCE_TEST["load_modules"]()
        self.source = self.modules["tier_switch.source"]
        self.tracking = self.modules["rebuild_source"]
        self.now = 0.0
        self.sizes = [64]
        self.admissions = []
        self.attempts = []
        self.forwards = []
        self.deadlines = []
        self.refs = []
        self.cleanup = []
        self.sql = []
        self.committed = []
        self.live = []
        self.prepared = self.closed = 0
        self.fail_stage = ""
        self.failed_attempts = {1}
        self.error_factory = lambda: RuntimeError("synthetic out of memory")
        self.rollback_mode = ""
        self.commit_after_success_timeout = False
        self.commit_after_success_oom = ""
        self.armed = False
        self.stage = ""
        self.before_hook = lambda: None
        self.after_hook = lambda: None
        self.prepare_hook = lambda: None
        self.encode_hook = lambda: None
        self.cleanup_hook = lambda kind: None
        self.write_hook = lambda: None
        self.finish_hook = lambda: None
        owner = self

        class Connection(sqlite3.Connection):
            tier_write = False

            def execute(self, sql, parameters=()):
                owner.sql.append(sql)
                result = super().execute(sql, parameters)
                if owner.armed:
                    if sql.startswith(f'INSERT INTO main."{TABLES[0]}"'):
                        self.tier_write = True
                        owner.write_hook()
                        owner.maybe_fail("insert_completion")
                    elif sql.startswith(f'INSERT INTO main."{TABLES[1]}"'):
                        owner.maybe_fail("insert_separation")
                    elif sql.startswith("INSERT OR REPLACE INTO main.metadata"):
                        owner.maybe_fail("checkpoint")
                return result

            def commit(self):
                owner.sql.append("COMMIT")
                wrote = self.tier_write
                if owner.armed and self.tier_write:
                    owner.maybe_fail("commit")
                super().commit()
                self.tier_write = False
                if owner.armed and wrote and owner.commit_after_success_timeout:
                    owner.now = 9001.0
                    raise TimeoutError("synthetic timeout after committed pair")
                if owner.armed and wrote and owner.commit_after_success_oom:
                    if owner.commit_after_success_oom == "cancel":
                        owner.worker.cancel()
                    elif owner.commit_after_success_oom == "timeout":
                        owner.now = 9001.0
                    raise RuntimeError("synthetic out of memory after committed pair")

            def rollback(self):
                owner.sql.append("ROLLBACK")
                if owner.rollback_mode == "dirty_raise":
                    raise MemoryError("synthetic out of memory during rollback")
                if owner.rollback_mode == "dirty_return":
                    return
                super().rollback()
                self.tier_write = False
                if owner.rollback_mode == "clean_raise":
                    raise MemoryError("synthetic out of memory after rollback")
                if owner.rollback_mode == "clean_timeout":
                    owner.now = 9001.0
                    raise TimeoutError("synthetic rollback timeout after cleanup")

        self.conn = sqlite3.connect(":memory:", factory=Connection)
        self.addCleanup(self.conn.close)
        self.conn.executescript("""
            CREATE TABLE messages(id INTEGER PRIMARY KEY, content TEXT, sender TEXT, recipient TEXT, timestamp TEXT);
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE vec_messages_custom(embedding BLOB);
            CREATE TABLE vec_messages_sep_custom(embedding BLOB);
            CREATE TABLE serving_pair(embedding BLOB);
            INSERT INTO serving_pair(rowid,embedding) VALUES (42,x'0102');
            CREATE TABLE rebuild_status(id INTEGER PRIMARY KEY, status TEXT);
            INSERT INTO rebuild_status VALUES (1,'untouched');
        """)
        self.tracking.ensure_rebuild_tracking(self.conn)

        class Throttler:
            batch_size = 64

            def before_batch(self):
                if len(owner.admissions) > 2000:
                    raise AssertionError("Unbounded synthetic loop")
                if owner.sizes:
                    self.batch_size = owner.sizes.pop(0)
                owner.admissions.append(self.batch_size)
                owner.before_hook()
                return self.batch_size, {"synthetic": True}

            def after_batch(self, count, duration):
                owner.committed.append(count)
                owner.after_hook()

            def on_oom(self):
                owner.observe_cleanup("backoff")

            def should_flush_cache(self):
                return False

            @staticmethod
            def flush_gpu_cache():
                owner.observe_cleanup("flush")

        registry = types.ModuleType("synthetic_registry")
        registry.VectorCacheRegistry = type("ForbiddenRegistry", (), {})
        throttler_module = types.ModuleType("synthetic_throttler")
        throttler_module.DynamicThrottler = Throttler
        self.modules["tier_switch.cache"] = registry
        self.modules["tier_switch.throttler"] = throttler_module
        self.torch = types.SimpleNamespace(no_grad=nullcontext, cuda=types.SimpleNamespace(OutOfMemoryError=type("CudaOOM", (RuntimeError,), {})))

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                return self.torch
            if name in ("numpy", "sqlite_vec", "sentence_transformers"):
                raise AssertionError("No native imports in streamed worker tests")
            if name.startswith("truememory."):
                short = name.removeprefix("truememory.")
                if short in self.modules:
                    return self.modules[short]
                raise AssertionError("Unexpected application import: " + name)
            return builtins.__import__(name, globals, locals, fromlist, level)

        def load(name):
            module = types.ModuleType("synthetic_streamed_" + name.replace(".", "_"))
            sys.modules[module.__name__] = module
            module.__dict__["__builtins__"] = dict(vars(builtins), __import__=safe_import)
            path = ROOT / "truememory" / (name.replace(".", "/") + ".py")
            exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
            self.modules[name] = module
            return module

        target_module = load("embedding_target")
        self.target = target_module.EmbeddingTarget("custom", "synthetic/worker", 2, "custom")
        self.vector = types.ModuleType("synthetic_streamed_vector")
        self.vector.__dict__.update(math=math, struct=struct, np=types.SimpleNamespace(ndarray=type("NativeArray", (), {})))
        tree = ast.parse((ROOT / "truememory/vector_search.py").read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"serialize_f32", "_build_sep_text"}]
        exec(compile(tree, "actual-serialization-and-separation", "exec"), self.vector.__dict__)
        serialize = self.vector.serialize_f32

        def serialized(value):
            self.maybe_fail("serialize_" + self.stage)
            return serialize(value)

        self.vector.serialize_f32 = serialized
        self.active_model = object()
        self.vector._model = self.active_model
        self.vector.EMBEDDING_MODEL = "serving-model"
        self.vector._embedding_dim = 99
        self.vector._model_generation = 17

        class Lease:
            def encode(self, texts, *, timeout, batch_size):
                owner.assertFalse(owner.conn.in_transaction)
                owner.assertGreater(timeout, 0)
                owner.assertEqual(batch_size, 32)
                owner.deadlines.append(timeout)
                owner.stage = "completion" if texts[0].startswith("content-") else "separation"
                if owner.stage == "completion":
                    owner.attempts.append(list(texts))
                owner.forwards.append((owner.stage, list(texts)))
                owner.encode_hook()
                owner.maybe_fail(owner.stage)
                values = Vectors([[3., 4.]] * len(texts))
                owner.refs.append(weakref.ref(values))
                return values

        @contextmanager
        def prepare(target, *, timeout):
            self.assertEqual(target, self.target)
            self.assertFalse(self.conn.in_transaction)
            self.assertGreater(timeout, 0)
            self.prepared += 1
            self.prepare_hook()
            try:
                yield Lease()
            finally:
                self.closed += 1

        self.vector.prepare_embedding_target = prepare
        self.modules["vector_search"] = self.vector
        self.worker_module = load("tier_switch.worker")
        self.worker_module.time = types.SimpleNamespace(monotonic=lambda: self.now, time=lambda: self.now)
        self.worker = self.worker_module.RebuildWorker(self.conn, "custom", "custom", Throttler(), status_id=1,
                                                       status_callback=lambda count, total, metrics: self.live.append((count, total, metrics)))
        self.worker._update_status = lambda *args, **kwargs: self.fail("Streamed worker wrote status")
        self.worker._process_batch = lambda *args, **kwargs: self.fail("Streamed worker invoked full-list batch path")
        self.worker._count_vectors = lambda *args, **kwargs: self.fail("Streamed worker recounted target vectors")

    def seed(self, ids):
        self.conn.executemany("INSERT INTO messages VALUES (?,?,?,?,?)", [
            (mid, f"content-{mid}", "sender", "recipient", "2026-01-02T03:04:05") for mid in ids
        ])
        self.conn.commit()

    def start(self):
        return self.source.initialize_tier_source(self.conn, self.source.plan_tier_source(
            self.conn, model_id=self.target.model_id, dimension=self.target.dimension, targets=TABLES,
        ))

    def run_worker(self, count=8, *, page_size=64, plan=None):
        if plan is None:
            self.seed(range(1, count + 1))
            plan = self.start()
        self.armed = True
        return self.worker.run_source(self.target, plan, page_size=page_size)

    def maybe_fail(self, stage):
        if stage == self.fail_stage and (self.failed_attempts is None or len(self.attempts) in self.failed_attempts):
            workspace = Workspace()
            self.refs.append(weakref.ref(workspace))
            raise self.error_factory()

    def observe_cleanup(self, kind):
        gc.collect()
        self.cleanup.append((kind, sys.exc_info()[0] is None, all(ref() is None for ref in self.refs)))
        self.cleanup_hook(kind)

    def ids(self):
        return tuple([row[0] for row in self.conn.execute(f'SELECT rowid FROM "{table}" ORDER BY rowid')] for table in TABLES)

    def test_order_metadata_exact_vectors_no_normalization_or_activation(self):
        self.seed((-8, 0, 3, 9))
        result = self.run_worker(plan=self.start(), page_size=3)
        self.assertEqual((result.status, result.processed, result.captured_range_complete, result.current_source_complete, result.activated),
                         ("complete", 4, True, True, False))
        self.assertEqual(self.ids(), ([-8, 0, 3, 9], [-8, 0, 3, 9]))
        self.assertEqual(self.forwards, [
            ("completion", ["content--8", "content-0", "content-3"]),
            ("separation", [f"sender to recipient on 2026-01-02: content-{mid}" for mid in (-8, 0, 3)]),
            ("completion", ["content-9"]), ("separation", ["sender to recipient on 2026-01-02: content-9"]),
        ])
        for table in TABLES:
            self.assertEqual({row[0] for row in self.conn.execute(f'SELECT embedding FROM "{table}"')}, {struct.pack("2f", 3., 4.)})
        self.assertIs(self.vector._model, self.active_model)
        self.assertEqual((self.vector.EMBEDDING_MODEL, self.vector._embedding_dim, self.vector._model_generation), ("serving-model", 99, 17))
        self.assertEqual(self.conn.execute("SELECT rowid,embedding FROM serving_pair").fetchone(), (42, b"\x01\x02"))
        self.assertEqual(self.conn.execute("SELECT status FROM rebuild_status").fetchone()[0], "untouched")
        self.assertTrue(all(not metrics["activated"] for _, _, metrics in self.live))
        self.assertEqual((self.prepared, self.closed), (1, 1))

    def test_long_source_retains_only_bounded_pending_pages_without_full_fetch(self):
        self.seed(range(1, 513))
        plan = self.start()
        pages = []
        weak_pages = []
        read = self.source.read_tier_page
        audit = self.source._audit_pair
        audits = []

        def bounded(conn, candidate, *, size=64):
            self.assertLessEqual(size, 7)
            gc.collect()
            self.assertLessEqual(sum(ref() is not None for ref in weak_pages), 1)
            page = read(conn, candidate, size=size)
            pages.append(len(page.rows))
            weak_pages.append(weakref.ref(page))
            return page

        self.source.read_tier_page = bounded
        self.source._audit_pair = lambda *args: (audits.append(True), audit(*args))[1]
        result = self.run_worker(plan=plan, page_size=7)
        self.assertEqual(result.processed, 512)
        self.assertLessEqual(max(pages), 7)
        self.assertLessEqual(max(map(len, self.attempts)), 7)
        self.assertEqual(audits, [])
        source_queries = [sql for sql in self.sql if sql.startswith("SELECT id, content")]
        self.assertTrue(source_queries)
        self.assertTrue(all("LIMIT ?" in sql for sql in source_queries))
        self.assertEqual(self.ids(), (list(range(1, 513)), list(range(1, 513))))

    def test_empty_input_still_prepares_certifies_and_finishes(self):
        result = self.run_worker(0)
        self.assertEqual((self.prepared, self.closed, self.attempts), (1, 1, []))
        self.assertEqual((result.status, result.processed, result.current_source_complete), ("complete", 0, True))

    def test_suffix_pair_failure_replans_exact_checkpoint_and_recovers(self):
        self.sizes = [2, 4]
        self.fail_stage = "insert_separation"
        self.failed_attempts = {2}
        plans = []
        original = self.source.plan_tier_source

        def replan(*args, **kwargs):
            plan = original(*args, **kwargs)
            plans.append((plan.manifest.generation, plan.manifest.cursor, plan.manifest.consumed))
            return plan

        self.seed(range(1, 7))
        plan = self.start()
        self.source.plan_tier_source = replan
        result = self.run_worker(plan=plan, page_size=6)
        self.assertEqual(result.processed, 6)
        self.assertEqual([batch[0] for batch in self.attempts], ["content-1", "content-3", "content-3"])
        self.assertEqual([len(batch) for batch in self.attempts], [2, 4, 4])
        self.assertEqual(plans, [(plan.manifest.generation, 2, 2)])
        self.assertEqual(self.ids(), (list(range(1, 7)), list(range(1, 7))))
        self.assertTrue(all(clear and released for _, clear, released in self.cleanup))

    def test_tracebacks_and_arrays_release_before_all_recovery_cleanup(self):
        for stage in ("completion", "separation", "serialize_completion", "serialize_separation", "insert_completion", "insert_separation", "checkpoint", "commit"):
            with self.subTest(stage=stage):
                case = TestStreamedWorker()
                case.setUp()
                try:
                    case.fail_stage = stage
                    result = case.run_worker(1)
                    self.assertEqual((result.status, result.processed), ("complete", 1))
                    self.assertEqual(case.cleanup, [("backoff", True, True), ("flush", True, True)])
                finally:
                    case.doCleanups()

    def test_finite_halving_uses_actual_failed_prefix_and_cannot_renew_credits(self):
        for admissions, expected in (
            ([32] * 20, [8, 8, 4, 4, 2, 2, 1, 1]),
            (list(range(16, 0, -1)), None),
            ([16, 32, 15, 32, 7, 32, 3, 32, 1, 32], None),
            ([16, 1, 32], [8, 1, 1]),
        ):
            with self.subTest(admissions=admissions):
                case = TestStreamedWorker()
                case.setUp()
                try:
                    case.sizes = admissions
                    case.fail_stage = "completion"
                    case.failed_attempts = None
                    result = case.run_worker(8)
                    counts = list(map(len, case.attempts))
                    self.assertEqual((result.status, result.processed), ("oom_exhausted", 0))
                    self.assertLessEqual(len(counts), 2 * (8).bit_length())
                    self.assertEqual(counts[-2:], [1, 1])
                    self.assertTrue(all(right <= left for left, right in zip(counts, counts[1:])))
                    if expected is not None:
                        self.assertEqual(counts, expected)
                finally:
                    case.doCleanups()

    def test_sql_oom_revalidation_does_not_reset_halving_credits(self):
        self.fail_stage = "insert_separation"
        self.failed_attempts = None
        result = self.run_worker(8)
        self.assertEqual((result.status, result.processed), ("oom_exhausted", 0))
        self.assertEqual(list(map(len, self.attempts)), [8, 8, 4, 4, 2, 2, 1, 1])
        self.assertEqual(self.ids(), ([], []))

    def test_oom_budget_resets_only_after_committed_pair(self):
        self.sizes = [8] * 20
        self.fail_stage = "completion"
        self.failed_attempts = {1, 2, 4, 5, 6, 7, 8, 9}
        result = self.run_worker(8)
        self.assertEqual((result.status, result.processed), ("oom_exhausted", 4))
        self.assertEqual(list(map(len, self.attempts)), [8, 8, 4, 4, 4, 2, 2, 1, 1])
        self.assertEqual(self.ids(), ([1, 2, 3, 4], [1, 2, 3, 4]))

    def test_rollback_uncertainty_never_classifies_or_runs_later_sql(self):
        for mode in ("dirty_raise", "clean_raise", "dirty_return"):
            with self.subTest(mode=mode):
                case = TestStreamedWorker()
                case.setUp()
                try:
                    case.seed((1,))
                    plan = case.start()
                    case.fail_stage = "insert_separation"
                    case.rollback_mode = mode
                    case.worker._is_oom_error = lambda error: self.fail("Classified an uncertain rollback")
                    with self.assertRaises((MemoryError, RuntimeError)):
                        case.run_worker(plan=plan)
                    self.assertEqual(case.sql[-1], "ROLLBACK")
                    self.assertEqual(case.cleanup, [])
                    self.assertEqual((case.prepared, case.closed), (1, 1))
                finally:
                    case.doCleanups()

    def test_bare_memory_error_is_not_reclassified_as_retryable(self):
        self.fail_stage = "separation"
        self.error_factory = MemoryError
        with self.assertRaises(MemoryError):
            self.run_worker(1)
        self.assertEqual(self.cleanup, [])
        self.assertEqual(self.ids(), ([], []))

    def test_expired_deadline_does_not_translate_uncertain_rollback_timeout(self):
        self.fail_stage = "insert_separation"
        self.rollback_mode = "clean_timeout"
        with self.assertRaisesRegex(TimeoutError, "rollback timeout after cleanup"):
            self.run_worker(1)
        self.assertEqual(self.sql[-1], "ROLLBACK")
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.cleanup, [])
        self.assertEqual((self.prepared, self.closed), (1, 1))

    def test_commit_then_timeout_propagates_instead_of_returning_old_cursor(self):
        self.seed((1,))
        plan = self.start()
        self.commit_after_success_timeout = True
        with self.assertRaisesRegex(TimeoutError, "timeout after committed pair"):
            self.run_worker(plan=plan)
        self.assertEqual(self.sql[-1], "ROLLBACK")
        self.assertEqual(self.ids(), ([1], [1]))
        durable = self.tracking.load_manifest(self.conn, self.tracking.manifest_key(TABLES))
        self.assertEqual((durable.cursor, durable.consumed), (1, 1))
        self.assertEqual(plan.manifest.consumed, 0)
        self.assertEqual(self.cleanup, [])
        self.assertEqual((self.prepared, self.closed), (1, 1))

    def test_terminal_singleton_oom_cannot_hide_successful_pair_commit(self):
        self.fail_stage = "completion"
        self.failed_attempts = {1}
        self.commit_after_success_oom = "exhausted"
        with self.assertRaisesRegex(self.source.TierSourceUntrusted, "same source checkpoint"):
            self.run_worker(1)
        self.assertEqual(list(map(len, self.attempts)), [1, 1])
        self.assertEqual(self.ids(), ([1], [1]))
        self.assertEqual(self.tracking.load_manifest(self.conn, self.tracking.manifest_key(TABLES)).consumed, 1)

    def test_cancel_or_deadline_after_ambiguous_commit_propagates_without_more_sql(self):
        for stop in ("cancel", "timeout"):
            with self.subTest(stop=stop):
                case = TestStreamedWorker()
                case.setUp()
                try:
                    case.commit_after_success_oom = stop
                    with self.assertRaisesRegex(case.source.TierSourceUntrusted, "durable checkpoint revalidation"):
                        case.run_worker(1)
                    self.assertEqual(case.sql[-1], "ROLLBACK")
                    self.assertEqual(case.cleanup, [])
                    self.assertEqual(case.ids(), ([1], [1]))
                    self.assertEqual(len(case.attempts), 1)
                finally:
                    case.doCleanups()

    def test_preparation_and_each_external_boundary_observe_cancellation(self):
        for point in ("before_prepare", "prepare", "admission", "completion", "separation", "after_batch", "callback"):
            with self.subTest(point=point):
                case = TestStreamedWorker()
                case.setUp()
                try:
                    if point == "before_prepare":
                        case.worker.cancel()
                    elif point == "prepare":
                        case.prepare_hook = case.worker.cancel
                    elif point == "admission":
                        case.before_hook = case.worker.cancel
                    elif point in ("completion", "separation"):
                        case.encode_hook = lambda: case.worker.cancel() if case.stage == point else None
                    elif point == "after_batch":
                        case.after_hook = case.worker.cancel
                    else:
                        case.worker.status_callback = lambda *args: case.worker.cancel()
                    result = case.run_worker(2)
                    self.assertEqual(result.status, "cancelled")
                    self.assertFalse(result.activated)
                    self.assertEqual(result.processed, 2 if point in ("after_batch", "callback") else 0)
                    expected_calls = {"before_prepare": 0, "prepare": 0, "admission": 0, "completion": 1, "separation": 2, "after_batch": 2, "callback": 2}
                    self.assertEqual(len(case.forwards), expected_calls[point])
                    self.assertEqual(case.closed, case.prepared)
                finally:
                    case.doCleanups()

    def test_cancel_inside_sql_call_reports_committed_cursor_without_next_native(self):
        self.write_hook = self.worker.cancel
        self.sizes = [1]
        result = self.run_worker(2)
        self.assertEqual((result.status, result.processed, result.plan.manifest.cursor), ("cancelled", 1, 1))
        self.assertEqual(self.ids(), ([1], [1]))
        self.assertEqual(len(self.forwards), 2)

    def test_cancellation_and_timeout_after_cleanup_prevent_retry(self):
        for stop in ("cancelled", "timeout"):
            for point in ("backoff", "flush", "admission"):
                with self.subTest(stop=stop, point=point):
                    case = TestStreamedWorker()
                    case.setUp()
                    try:
                        case.fail_stage = "completion"
                        case.failed_attempts = None

                        def stopped():
                            if stop == "cancelled":
                                case.worker.cancel()
                            else:
                                case.now = 9001.0

                        case.cleanup_hook = lambda kind: stopped() if kind == point else None
                        case.before_hook = lambda: stopped() if point == "admission" and len(case.admissions) == 2 else None
                        result = case.run_worker(1)
                        self.assertEqual(result.status, stop)
                        self.assertEqual(len(case.attempts), 1)
                        self.assertEqual(result.processed, 0)
                    finally:
                        case.doCleanups()

    def test_timeout_after_prepare_and_during_encode_blocks_followup(self):
        self.prepare_hook = lambda: setattr(self, "now", 9001.0)
        result = self.run_worker(1)
        self.assertEqual((result.status, len(self.forwards), self.closed), ("timeout", 0, 1))

    def test_timeout_during_completion_stops_before_separation_and_sql(self):
        self.encode_hook = lambda: setattr(self, "now", 9001.0)
        result = self.run_worker(1)
        self.assertEqual((result.status, len(self.forwards), result.processed), ("timeout", 1, 0))
        self.assertEqual(self.ids(), ([], []))
        self.assertEqual(self.closed, 1)

    def test_remaining_deadline_reaches_both_lease_calls(self):
        self.encode_hook = lambda: setattr(self, "now", self.now + 10.0)
        result = self.run_worker(1)
        self.assertEqual(self.deadlines, [9000.0, 8990.0])
        self.assertEqual(result.status, "complete")

    def test_finish_cancellation_preserves_completed_manifest_truth(self):
        finish = self.source.finish_tier_source

        def finish_then_cancel(*args):
            result = finish(*args)
            self.worker.cancel()
            return result

        self.source.finish_tier_source = finish_then_cancel
        result = self.run_worker(1)
        self.assertEqual((result.status, result.processed, result.plan.manifest.complete), ("cancelled", 1, True))
        self.assertTrue(result.captured_range_complete)
        self.assertTrue(result.current_source_complete)
        self.assertFalse(result.activated)

    def test_revalidation_cannot_resume_another_generation(self):
        self.seed((1,))
        plan = self.start()
        self.fail_stage = "insert_separation"
        original = self.source.plan_tier_source

        def switched(*args, **kwargs):
            from dataclasses import replace
            resumed = original(*args, **kwargs)
            return replace(resumed, manifest=replace(resumed.manifest, generation="0" * 32))

        self.source.plan_tier_source = switched
        with self.assertRaisesRegex(self.source.TierSourceUntrusted, "same source checkpoint"):
            self.run_worker(plan=plan)
        self.assertEqual(len(self.attempts), 1)
        self.assertEqual(self.ids(), ([], []))

    def test_lease_call_cancellation_witness_does_not_claim_in_call_interruption(self):
        admitted_native = []

        def simulated_internal_wait_then_native():
            self.worker.cancel()
            admitted_native.append("already-admitted-forward")

        self.encode_hook = simulated_internal_wait_then_native
        result = self.run_worker(1)
        self.assertEqual(admitted_native, ["already-admitted-forward"])
        self.assertEqual(len(self.forwards), 1)
        self.assertEqual((result.status, result.processed), ("cancelled", 0))
        self.assertEqual(self.ids(), ([], []))

    def test_borrowed_connection_and_identity_mismatch_fail_before_model_or_sql(self):
        plan = self.start()
        self.conn.execute("INSERT INTO metadata VALUES ('borrowed','pending')")
        before = len(self.sql)
        with self.assertRaisesRegex(RuntimeError, "clean owned"):
            self.worker.run_source(self.target, plan)
        self.assertEqual(len(self.sql), before)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.prepared, 0)
        self.conn.rollback()
        self.worker.target_group = "different"
        with self.assertRaisesRegex(ValueError, "identities"):
            self.worker.run_source(self.target, plan)
        self.assertEqual(self.prepared, 0)

    def test_source_change_after_native_failure_refuses_before_retry(self):
        self.fail_stage = "completion"

        def correct(kind):
            if kind == "backoff":
                self.conn.execute("UPDATE messages SET content='corrected' WHERE id=1")
                self.conn.commit()

        self.cleanup_hook = correct
        with self.assertRaises(self.tracking.RebuildSourceChanged):
            self.run_worker(1)
        self.assertEqual(len(self.attempts), 1)
        self.assertEqual(self.ids(), ([], []))

    def test_captured_completion_does_not_claim_new_tail_or_activation(self):
        self.seed((1, 2))
        self.start()
        self.seed((3,))
        plan = self.source.plan_tier_source(self.conn, model_id=self.target.model_id, dimension=2, targets=TABLES)
        result = self.run_worker(plan=plan)
        self.assertEqual((result.captured_range_complete, result.current_source_complete, result.activated), (True, False, False))
        self.assertEqual(self.ids(), ([1, 2], [1, 2]))


if __name__ == "__main__":
    unittest.main()
