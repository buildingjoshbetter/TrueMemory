"""Bounded rebuild recovery with synthetic models and in-memory SQLite only."""
from __future__ import annotations

import ast
import gc
import sqlite3
import sys
import types
import unittest
import weakref
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "truememory/tier_switch/worker.py"
SOURCE_TEXT: str | None = None
VEC = "vec_messages_base"
SEP = "vec_messages_sep_base"


def load_worker(registry: type, throttler: type) -> types.ModuleType:
    tree = ast.parse(SOURCE_TEXT if SOURCE_TEXT is not None else SOURCE.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("truememory.")
    )]
    module = types.ModuleType("synthetic_bounded_rebuild")
    module.VectorCacheRegistry = registry
    module.DynamicThrottler = throttler
    exec(compile(tree, str(SOURCE), "exec"), module.__dict__)
    module.log.disabled = True
    return module


class Workspace:
    pass


class TestRebuildOOMRecovery(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.addCleanup(self.conn.close)
        self.conn.executescript(f"""
            CREATE TABLE {VEC}(rowid INTEGER PRIMARY KEY, embedding BLOB);
            CREATE TABLE {SEP}(rowid INTEGER PRIMARY KEY, embedding BLOB);
            CREATE TABLE progress(last_id INTEGER, n INTEGER);
            INSERT INTO progress VALUES(0,0);
            CREATE TABLE notes(value TEXT);
            CREATE TABLE rebuild_status(
                id INTEGER PRIMARY KEY, status TEXT, processed_messages INTEGER,
                progress_pct REAL, eta_seconds REAL, batch_size INTEGER,
                throughput_ips REAL, ram_pct REAL, last_heartbeat REAL,
                completed_at REAL, error TEXT
            );
            INSERT INTO rebuild_status(id,status,processed_messages) VALUES(1,'running',0);
        """)
        self.now = 0.0
        self.admissions: list[int] = []
        self.attempts: list[list[str]] = []
        self.forwards: list[tuple[str, list[str]]] = []
        self.sql: list[str] = []
        self.refs: list[weakref.ReferenceType] = []
        self.cleanup: list[tuple[str, bool, bool]] = []
        self.completed: list[int] = []
        self.ooms = 0
        self.sizes: list[int] = [1]
        self.halve = False
        self.stage = ""
        self.fail_stage = ""
        self.failed_attempts: set[int] | None = {1}
        self.error_factory: Callable[[], Exception] = lambda: RuntimeError("synthetic out of memory")
        self.rollback_error: Exception | None = None
        self.before_hook: Callable[[], None] = lambda: None
        self.cleanup_hook: Callable[[str], None] = lambda kind: None
        self.status_failure = False
        self.status_failed = False
        owner = self

        class Registry:
            @staticmethod
            def update_progress(conn: object, group: str, last_id: int, count: int, commit: bool = False) -> None:
                owner.assertFalse(commit)
                conn.execute("UPDATE progress SET last_id=?,n=?", (last_id, count))

        class Throttler:
            batch_size = 1

            def before_batch(self) -> tuple[int, dict]:
                if len(owner.admissions) >= 96:
                    raise AssertionError("Synthetic admission call cap reached")
                if owner.sizes:
                    self.batch_size = owner.sizes.pop(0)
                owner.admissions.append(self.batch_size)
                owner.now += 0.01
                if len(owner.admissions) >= 32:
                    owner.now = 9001.0
                owner.before_hook()
                return self.batch_size, {}

            def on_oom(self) -> None:
                owner.ooms += 1
                owner.observe_cleanup("backoff")
                if owner.halve:
                    self.batch_size = max(1, self.batch_size // 2)

            def after_batch(self, count: int, duration: float) -> None:
                owner.completed.append(count)

            def should_flush_cache(self) -> bool:
                return False

            @staticmethod
            def flush_gpu_cache() -> None:
                owner.observe_cleanup("flush")

            def get_eta_seconds(self, remaining: int) -> float:
                return float(remaining)

            def get_throughput(self) -> float:
                return 1.0

        class Connection:
            def __getattr__(self, name: str) -> object:
                return getattr(owner.conn, name)

            def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
                owner.sql.append(sql)
                if sql.startswith("UPDATE rebuild_status") and owner.status_failure and not owner.status_failed:
                    owner.status_failed = True
                    owner.conn.execute(sql, *args)
                    raise sqlite3.OperationalError("synthetic status write failure")
                if sql == "COMMIT":
                    owner.maybe_fail("commit")
                cursor = owner.conn.execute(sql, *args)
                if sql.startswith("UPDATE progress"):
                    owner.maybe_fail("checkpoint")
                if sql == f"DELETE FROM {SEP}":
                    owner.maybe_fail("full_clear")
                return cursor

            def executemany(self, sql: str, rows: list[tuple]) -> sqlite3.Cursor:
                owner.sql.append(sql)
                stage = "insert_completion" if VEC in sql else "insert_separation"
                if owner.should_fail(stage):
                    owner.conn.execute(sql, rows[0])
                    owner.maybe_fail(stage)
                return owner.conn.executemany(sql, rows)

            def rollback(self) -> None:
                owner.sql.append("ROLLBACK")
                if owner.rollback_error is not None:
                    raise owner.rollback_error
                owner.conn.rollback()

        class Model:
            def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
                owner.assertEqual(kwargs, {"show_progress_bar": False})
                owner.assertFalse(owner.conn.in_transaction)
                owner.stage = "separation" if texts[0].startswith("sep:") else "completion"
                if owner.stage == "completion":
                    owner.attempts.append(list(texts))
                owner.forwards.append((owner.stage, list(texts)))
                owner.maybe_fail(owner.stage)
                return [[1.0, float(index)] for index in range(len(texts))]

        def serialize(row: list[float]) -> bytes:
            self.maybe_fail("serialize_" + self.stage)
            return b"synthetic vector"

        package = types.ModuleType("truememory")
        package.__path__ = []
        vector = types.ModuleType("truememory.vector_search")
        vector.get_model = Model
        vector.set_embedding_model = lambda tier: None
        vector.init_vec_table = lambda conn, tier_group: None
        vector.serialize_f32 = serialize
        vector._build_sep_text = lambda sender, recipient, timestamp, content: "sep:" + content
        ownership = types.ModuleType("truememory.mps_utils")
        ownership.encode_with_model_ownership = lambda model, texts, **kwargs: model.encode(texts, **kwargs)
        torch = types.ModuleType("torch")
        torch.no_grad = nullcontext
        torch.cuda = types.SimpleNamespace(OutOfMemoryError=type("CudaOOM", (RuntimeError,), {}))
        psutil = types.ModuleType("psutil")
        psutil.virtual_memory = lambda: types.SimpleNamespace(percent=0.0)
        modules = patch.dict(sys.modules, {"truememory": package, "truememory.vector_search": vector,
                            "truememory.mps_utils": ownership, "torch": torch, "psutil": psutil})
        modules.start()
        self.addCleanup(modules.stop)
        self.module = load_worker(Registry, Throttler)
        self.module.time = types.SimpleNamespace(time=lambda: self.now)
        self.throttler = Throttler()
        self.worker = self.module.RebuildWorker(Connection(), "base", "base", self.throttler, status_id=1)

    def should_fail(self, stage: str) -> bool:
        return stage == self.fail_stage and (
            self.failed_attempts is None or len(self.attempts) in self.failed_attempts
        )

    def maybe_fail(self, stage: str) -> None:
        if self.should_fail(stage):
            workspace = Workspace()
            self.refs.append(weakref.ref(workspace))
            raise self.error_factory()

    def observe_cleanup(self, kind: str) -> None:
        gc.collect()
        self.cleanup.append((kind, sys.exc_info()[0] is None, all(ref() is None for ref in self.refs)))
        self.cleanup_hook(kind)

    def run_worker(self, count: int, full: bool = False) -> tuple[bool, int]:
        return self.worker.run([{"id": index, "content": f"item-{index}"}
                                for index in range(1, count + 1)], full)

    def ids(self) -> tuple[list[int], list[int]]:
        return tuple([row[0] for row in self.conn.execute(f"SELECT rowid FROM {table} ORDER BY rowid")]
                     for table in (VEC, SEP))

    def status(self) -> tuple:
        return self.conn.execute("SELECT status,processed_messages,error FROM rebuild_status").fetchone()

    def seed_prior(self) -> None:
        for table in (VEC, SEP):
            self.conn.execute(f"INSERT INTO {table} VALUES(99,?)", (b"prior",))
        self.conn.execute("UPDATE progress SET last_id=99,n=1")
        self.conn.commit()

    def test_transient_oom_releases_each_stage_traceback_before_cleanup_and_recovers(self) -> None:
        for stage in ("completion", "separation", "serialize_completion", "serialize_separation",
                      "insert_completion", "insert_separation", "checkpoint", "commit"):
            with self.subTest(stage=stage):
                self.fail_stage = stage
                self.attempts.clear()
                self.refs.clear()
                self.cleanup.clear()
                self.ooms = 0
                for table in (VEC, SEP):
                    self.conn.execute(f"DELETE FROM {table}")
                self.conn.execute("UPDATE progress SET last_id=0,n=0")
                self.conn.commit()
                self.assertEqual(self.run_worker(1), (True, 1))
                self.assertEqual(self.ooms, 1)
                self.assertEqual(self.cleanup, [("backoff", True, True), ("flush", True, True)])
                self.assertEqual(self.ids(), ([1], [1]))
                self.assertEqual(self.conn.execute("SELECT * FROM progress").fetchone(), (1, 1))
                self.assertFalse(self.conn.in_transaction)

    def test_persistent_singleton_stops_after_cleanup_retry_with_prior_rows_intact(self) -> None:
        self.seed_prior()
        self.fail_stage = "separation"
        self.failed_attempts = None
        self.assertEqual(self.run_worker(1), (False, 0))
        self.assertEqual(self.attempts, [["item-1"], ["item-1"]])
        self.assertEqual(self.ooms, 2)
        self.assertEqual(self.ids(), ([99], [99]))
        self.assertEqual(self.conn.execute("SELECT * FROM progress").fetchone(), (99, 1))
        self.assertEqual(self.status(), ("failed", 0,
                         "Rebuild ran out of memory twice at batch size 1; retry after freeing memory."))
        self.assertTrue(all(clear and released for _, clear, released in self.cleanup))

    def test_tail_eight_configured_thirty_two_can_shrink_and_complete(self) -> None:
        self.sizes = [32, 16, 8, 4]
        self.fail_stage = "completion"
        self.failed_attempts = {1, 2}
        self.assertEqual(self.run_worker(8), (True, 8))
        self.assertEqual([len(batch) for batch in self.attempts], [8, 8, 4, 4])
        self.assertEqual(self.ids(), (list(range(1, 9)), list(range(1, 9))))

    def test_adversarial_admissions_cannot_renew_halving_credits(self) -> None:
        for sizes in ([16] * 20, list(range(16, 0, -1)), [16, 32, 15, 32, 7, 32, 3, 32, 1, 32]):
            with self.subTest(sizes=sizes):
                self.sizes = list(sizes)
                self.admissions.clear()
                self.attempts.clear()
                self.fail_stage = "completion"
                self.failed_attempts = None
                self.assertEqual(self.run_worker(16), (False, 0))
                counts = [len(batch) for batch in self.attempts]
                if sizes == [16] * 20:
                    self.assertEqual(counts, [16, 16, 8, 8, 4, 4, 2, 2, 1, 1])
                self.assertLessEqual(len(counts), 2 * (16).bit_length())
                self.assertEqual(counts[-2:], [1, 1])
                self.assertTrue(all(right <= left for left, right in zip(counts, counts[1:])))
                self.assertTrue(all(batch[0] == "item-1" for batch in self.attempts))

    def test_direct_sixteen_to_one_skip_retains_singleton_cleanup_retry(self) -> None:
        self.sizes = [16, 1, 32]
        self.fail_stage = "completion"
        self.failed_attempts = None
        self.assertEqual(self.run_worker(16), (False, 0))
        self.assertEqual([len(batch) for batch in self.attempts], [16, 1, 1])

    def test_recovery_budget_resets_only_after_committed_pair(self) -> None:
        self.sizes = [8] * 20
        self.fail_stage = "completion"
        self.failed_attempts = {1, 2, 4, 5, 6, 7, 8, 9}
        self.assertEqual(self.run_worker(8), (False, 4))
        self.assertEqual([len(batch) for batch in self.attempts], [8, 8, 4, 4, 4, 2, 2, 1, 1])
        self.assertEqual([batch[0] for batch in self.attempts], ["item-1"] * 3 + ["item-5"] * 6)
        self.assertEqual(self.completed, [4])
        self.assertEqual(self.ids(), ([1, 2, 3, 4], [1, 2, 3, 4]))
        self.assertEqual(self.conn.execute("SELECT * FROM progress").fetchone(), (4, 4))
        self.assertEqual(self.status()[:2], ("failed", 4))

    def test_successful_batches_keep_original_admissions_order_and_count(self) -> None:
        self.sizes = [3, 5, 2]
        self.assertEqual(self.run_worker(10), (True, 10))
        self.assertEqual([len(batch) for batch in self.attempts], [3, 5, 2])
        self.assertEqual([text for batch in self.attempts for text in batch], [f"item-{i}" for i in range(1, 11)])
        self.assertEqual(self.completed, [3, 5, 2])
        self.assertEqual(self.cleanup, [])

    def test_cancel_and_timeout_during_recovery_or_next_admission_prevent_native_retry(self) -> None:
        for event in ("cancel", "timeout"):
            for phase in ("backoff", "flush", "admission"):
                with self.subTest(event=event, phase=phase):
                    self.worker._cancelled = False
                    self.now = 0.0
                    self.attempts.clear()
                    self.admissions.clear()
                    self.fail_stage = "completion"
                    self.failed_attempts = None

                    def stop() -> None:
                        if event == "cancel":
                            self.worker.cancel()
                        else:
                            self.now = 9001.0

                    self.cleanup_hook = lambda kind: stop() if kind == phase else None
                    self.before_hook = lambda: stop() if phase == "admission" and len(self.admissions) == 2 else None
                    self.assertEqual(self.run_worker(1), (False, 0))
                    self.assertEqual(len(self.attempts), 1)
                    self.assertEqual(self.status()[0], "cancelled" if event == "cancel" else "timeout")

    def test_bare_memory_error_classification_remains_terminal(self) -> None:
        self.fail_stage = "insert_separation"
        self.error_factory = MemoryError
        self.assertEqual(self.run_worker(1), (False, 0))
        self.assertEqual(self.ooms, 0)
        self.assertEqual(self.cleanup, [])
        self.assertEqual(self.ids(), ([], []))
        self.assertFalse(self.conn.in_transaction)

    def test_message_classified_write_memory_error_retries_only_after_clean_rollback(self) -> None:
        self.fail_stage = "insert_separation"
        self.error_factory = lambda: MemoryError("synthetic out of memory")
        self.assertEqual(self.run_worker(1), (True, 1))
        self.assertEqual(self.ooms, 1)
        self.assertEqual(self.cleanup, [("backoff", True, True), ("flush", True, True)])
        self.assertEqual(self.ids(), ([1], [1]))
        self.assertFalse(self.conn.in_transaction)

    def test_uncertain_batch_or_full_clear_rollback_never_classifies_retries_or_publishes_status(self) -> None:
        for stage in ("insert_separation", "full_clear"):
            with self.subTest(stage=stage):
                self.conn.rollback()
                for table in (VEC, SEP):
                    self.conn.execute(f"DELETE FROM {table}")
                self.conn.commit()
                self.seed_prior()
                self.sql.clear()
                self.attempts.clear()
                self.fail_stage = stage
                self.failed_attempts = None
                self.rollback_error = MemoryError("synthetic out of memory during rollback")
                with patch.object(self.worker, "_is_oom_error", Mock(side_effect=AssertionError("Classified uncertain failure"))) as classify, \
                        patch.object(self.worker, "_update_status", Mock(side_effect=AssertionError("Wrote uncertain status"))) as status:
                    with self.assertRaises(MemoryError) as raised:
                        self.run_worker(1, full=stage == "full_clear")
                    self.assertIs(raised.exception, self.rollback_error)
                    classify.assert_not_called()
                    status.assert_not_called()
                self.assertTrue(self.conn.in_transaction)
                self.assertEqual(self.sql[-1], "ROLLBACK")
                self.assertEqual(len(self.attempts), 0 if stage == "full_clear" else 1)
                self.assertEqual(self.cleanup, [])
                self.conn.rollback()
                self.assertEqual(self.ids(), ([99], [99]))
                self.rollback_error = None

    def test_uncertain_status_rollback_stops_before_another_batch_or_false_complete(self) -> None:
        self.status_failure = True
        self.rollback_error = sqlite3.OperationalError("synthetic status rollback failure")
        with self.assertRaises(sqlite3.OperationalError) as raised:
            self.run_worker(2)
        self.assertIs(raised.exception, self.rollback_error)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.sql[-1], "ROLLBACK")
        self.assertEqual(len(self.attempts), 1)
        self.assertEqual(self.cleanup, [])
        self.conn.rollback()
        self.assertEqual(self.ids(), ([1], [1]))
        self.assertEqual(self.conn.execute("SELECT * FROM progress").fetchone(), (1, 1))
        self.assertEqual(self.status()[:2], ("running", 0))

    def test_clean_status_rollback_remains_best_effort(self) -> None:
        self.status_failure = True
        self.assertEqual(self.run_worker(2), (True, 2))
        self.assertEqual(len(self.attempts), 2)
        self.assertEqual(self.ooms, 0)
        self.assertEqual(self.status()[:2], ("complete", 2))
        self.assertFalse(self.conn.in_transaction)

    def test_borrowed_status_savepoint_preserves_caller_work(self) -> None:
        self.conn.execute("INSERT INTO notes VALUES('synthetic caller')")
        self.status_failure = True
        self.worker._update_status("failed", 0, 1)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT * FROM notes").fetchall(), [("synthetic caller",)])
        self.assertEqual(self.status()[:2], ("running", 0))
        self.assertFalse(any(sql == "COMMIT" for sql in self.sql))
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT * FROM notes").fetchall(), [])


if __name__ == "__main__":
    unittest.main()
