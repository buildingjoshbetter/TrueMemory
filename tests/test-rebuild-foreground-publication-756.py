"""Actual foreground methods and rebuild publication on synthetic databases."""

import ast
import logging
import runpy
import sqlite3
import threading
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = types.SimpleNamespace(**runpy.run_path(str(ROOT / "tests/test-rebuild-source-streaming-756.py")))


class TestForegroundPublication(unittest.TestCase):
    setUp = BASE.TestStreamingBuilders.setUp
    seed = BASE.TestStreamingBuilders.seed
    ids = BASE.TestStreamingBuilders.ids
    manifest = BASE.TestStreamingBuilders.manifest
    writer = BASE.TestStreamingBuilders.writer

    def engine(self, conn=None):
        tree = ast.parse((ROOT / "truememory/engine.py").read_text(encoding="utf-8"))
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrueMemoryEngine")
        methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in ("add", "update")]
        cls = ast.ClassDef(name="SyntheticEngine", bases=[], keywords=[], body=methods, decorator_list=[])
        namespace = dict(engine_operation=lambda function: function, __builtins__=self.vector.__dict__["__builtins__"],
                         logger=logging.getLogger(__name__), MAX_CONTENT_LENGTH=10000,
                         insert_message=self.modules["storage"].insert_message,
                         update_message=self.modules["storage"].update_message)
        validators = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_validate_add_content"]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[*validators, cls], type_ignores=[])), "actual-engine-boundaries", "exec"), namespace)
        engine = namespace["SyntheticEngine"]()
        engine.conn = conn or self.conn
        engine._write_lock = threading.Lock()
        engine._ensure_connection = lambda: None
        engine._maybe_auto_consolidate = lambda: None
        engine._has_vectors = True
        engine._has_personality = engine._has_style_vec = False
        engine.get = lambda mid: self.modules["storage"].get_message(engine.conn, mid)
        return engine

    def test_actual_add_above_active_range_survives_builder_completion(self):
        self.seed(range(1, 5))
        other = self.writer()
        engine = self.engine(other)
        inserted = []
        def encode(texts):
            if not inserted:
                inserted.append(True)
                item = engine.add("synthetic-late-append", sender="sender", recipient="recipient", timestamp="2026-01-02")
                self.assertEqual(item["id"], 5)
                inserted.append(item["id"])
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        self.assertEqual(self.vector.build_vectors(self.conn, txn_batch=1), 4)
        self.assertEqual(self.ids(), [1, 2, 3, 4, 5])
        self.assertEqual((self.manifest().cursor, self.manifest().consumed, self.manifest().source.high_id), (4, 4, 4))
        self.assertEqual(self.ids("vec_messages_sep"), [5])

    def test_actual_update_invalidates_build_but_keeps_corrected_vector(self):
        self.seed(range(1, 5))
        engine = self.engine(self.writer())
        updated = []
        def encode(texts):
            if not updated:
                updated.append(True)
                self.assertEqual(engine.update(1, "synthetic-correction")["content"], "synthetic-correction")
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaises(self.source.RebuildSourceChanged):
            self.vector.build_vectors(self.conn, txn_batch=1)
        self.assertEqual(self.ids(), [1])
        self.assertEqual(self.manifest().consumed, 0)
        self.assertTrue(self.vector._build_in_progress(self.conn, "vec_messages"))

    def test_late_model_change_rolls_back_add_source_and_does_not_write_vectors(self):
        engine = self.engine()
        def encode(texts):
            if len(self.calls) == 2:
                self.vector._model_generation += 1
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaisesRegex(self.vector.VectorPublicationChanged, "model changed"):
            engine.add("synthetic-stale-model")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)
        self.assertEqual(self.ids(), [])
        self.assertFalse(self.conn.in_transaction)

    def test_late_model_change_update_reports_already_committed_source(self):
        self.seed((1,))
        engine = self.engine()
        def encode(texts):
            if len(self.calls) == 2:
                self.vector._model_generation += 1
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaisesRegex(self.vector.VectorPublicationChanged, "Source update committed"):
            engine.update(1, "synthetic-committed-correction")
        self.assertEqual(self.conn.execute("SELECT content FROM messages WHERE id=1").fetchone()[0], "synthetic-committed-correction")
        self.assertEqual(self.ids(), [])

    def test_completed_foreign_model_manifest_rejects_old_process_write(self):
        self.seed((1,))
        self.vector.build_vectors(self.conn)
        manifest = self.manifest()
        from dataclasses import replace
        self.source.save_manifest(self.conn, self.source.manifest_key(("vec_messages",)), replace(manifest, model="different-synthetic-model"))
        self.conn.commit()
        with self.assertRaisesRegex(self.vector.VectorPublicationChanged, "different embedding model"):
            self.engine().add("synthetic-old-process")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 1)
        self.assertEqual(self.ids(), [1])

    def test_unattested_marker_rejects_add_without_silent_success(self):
        self.vector._mark_build_in_progress(self.conn, "vec_messages")
        self.conn.commit()
        with self.assertRaisesRegex(self.vector.VectorPublicationChanged, "without matching source"):
            self.engine().add("synthetic-unattested")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)

    def test_guard_releases_model_lock_when_sql_publication_raises(self):
        self.seed((1,))
        identity = (0, "synthetic-model", 2)
        with self.assertRaisesRegex(RuntimeError, "synthetic write failure"):
            with self.vector._foreground_vector_publication(self.conn, identity, 1):
                raise RuntimeError("synthetic write failure")
        self.assertTrue(self.vector._lock.acquire(blocking=False))
        self.vector._lock.release()

    def test_embed_single_finishes_both_encodes_before_owned_database_write(self):
        self.seed((1,))
        def encode(texts):
            self.assertFalse(self.conn.in_transaction)
            self.assertFalse(self.vector._lock.locked())
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        self.vector.embed_single(self.conn, 1, "synthetic-1")
        self.assertEqual(self.ids(), [1])
        self.assertEqual(self.ids("vec_messages_sep"), [1])
        self.assertFalse(self.conn.in_transaction)

    def test_embed_single_stale_model_is_rejected_before_either_insert(self):
        self.seed((1,))
        def encode(texts):
            if len(self.calls) == 2:
                self.vector._model_generation += 1
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaises(self.vector.VectorPublicationChanged):
            self.vector.embed_single(self.conn, 1, "synthetic-1")
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.ids("vec_messages_sep"), [])

    def _attempt_foreign_manifest_at_vector_insert(self, operation):
        self.seed((1,))
        other = self.writer()
        other.execute("PRAGMA busy_timeout = 0")
        manifest = self.source.RebuildManifest.new(
            "different-synthetic-model", 2, ("vec_messages",), total=0,
        )
        attempts = []
        ownership = []
        real = self.conn
        case = self

        class Probe:
            def __getattr__(self, name):
                return getattr(real, name)

            def execute(self, sql, parameters=()):
                if not attempts and sql.startswith(("INSERT INTO vec_messages(", "DELETE FROM vec_messages WHERE")):
                    ownership.append((real.in_transaction, case.vector._lock.locked()))
                    try:
                        case.source.save_manifest(other, case.source.manifest_key(("vec_messages",)), manifest)
                        other.commit()
                        attempts.append("committed")
                    except sqlite3.OperationalError as error:
                        case.assertIn("locked", str(error).lower())
                        attempts.append("blocked")
                    finally:
                        other.rollback()
                return real.execute(sql, parameters)

        probe = Probe()
        if operation == "embed":
            self.vector.embed_single(probe, 1, "synthetic-1")
            expected = [1]
        elif operation == "update":
            self.engine(probe).update(1, "synthetic-correction")
            expected = [1]
        else:
            self.engine(probe).add("synthetic-append")
            expected = [2]
        self.assertEqual(attempts, ["blocked"])
        self.assertEqual(ownership, [(True, True)])
        self.assertIsNone(self.manifest())
        self.assertEqual(self.ids(), expected)
        self.assertEqual(self.ids("vec_messages_sep"), expected)
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='embed_model'").fetchone()[0],
                         "synthetic-model")
        self.assertFalse(self.conn.in_transaction)

    def test_embed_single_excludes_foreign_manifest_between_validation_and_insert(self):
        self._attempt_foreign_manifest_at_vector_insert("embed")

    def test_update_excludes_foreign_manifest_after_source_helper_commits(self):
        self._attempt_foreign_manifest_at_vector_insert("update")

    def test_add_excludes_foreign_manifest_through_vector_publication(self):
        self._attempt_foreign_manifest_at_vector_insert("add")

    def test_rejected_add_preserves_unrelated_caller_write_and_transaction(self):
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-caller','pending')")
        def encode(texts):
            if len(self.calls) == 2:
                self.vector._model_generation += 1
            return [[1., 0.] for _ in texts]
        self.on_encode = encode
        with self.assertRaises(self.vector.VectorPublicationChanged):
            self.engine().add("synthetic-rejected")
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-caller'").fetchone()[0], "pending")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)
        self.conn.commit()
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-caller'").fetchone()[0], "pending")
        self.assertEqual(self.ids(), [])

    def test_foreign_target_rejection_rolls_back_only_new_add_after_source_insert(self):
        manifest = self.source.RebuildManifest.new("different-synthetic-model", 2, ("vec_messages",))
        self.source.save_manifest(self.conn, self.source.manifest_key(("vec_messages",)), manifest)
        self.conn.commit()
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-caller','pending')")
        with self.assertRaisesRegex(self.vector.VectorPublicationChanged, "different embedding model"):
            self.engine().add("synthetic-rejected")
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)
        self.conn.commit()
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-caller'").fetchone()[0], "pending")
        self.assertEqual(self.ids(), [])

    def test_successful_add_remains_inside_caller_transaction(self):
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-caller','pending')")
        item = self.engine().add("synthetic-pending-add")
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.ids(), [item["id"]])
        self.conn.rollback()
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.ids("vec_messages_sep"), [])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)

    def test_embed_single_keeps_caller_work_pending(self):
        self.seed((1,))
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-caller','pending')")
        self.vector.embed_single(self.conn, 1, "synthetic-1")
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(self.ids(), [1])
        self.conn.rollback()
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.ids("vec_messages_sep"), [])

    def _assert_foreground_writer_and_terminal_fence(self, operation, caller_owned):
        self.seed((1,))
        if caller_owned:
            self.conn.execute("BEGIN")
        real = self.conn
        case = self
        boundaries = []

        class Probe:
            def __getattr__(self, name):
                return getattr(real, name)

            def execute(self, sql, parameters=()):
                if sql == "BEGIN IMMEDIATE" or sql == "UPDATE messages SET id = id WHERE 0":
                    case.assertTrue(case.vector._lock.locked())
                    boundaries.append("writer")
                if sql == "RELEASE SAVEPOINT rebuild_source":
                    case.assertTrue(case.vector._lock.locked())
                    boundaries.append("release")
                return real.execute(sql, parameters)

            def commit(self):
                case.assertTrue(case.vector._lock.locked())
                boundaries.append("commit")
                real.commit()

        if operation == "embed":
            self.vector.embed_single(Probe(), 1, "synthetic-1")
        else:
            self.engine(Probe()).add("synthetic-append")
        self.assertEqual(boundaries, ["writer", "release" if caller_owned else "commit"])
        self.assertEqual(real.in_transaction, caller_owned)
        self.assertFalse(self.vector._lock.locked())

    def test_embed_owned_commit_stays_inside_model_fence(self):
        self._assert_foreground_writer_and_terminal_fence("embed", False)

    def test_embed_caller_fence_precedes_writer_upgrade_and_contains_release(self):
        self._assert_foreground_writer_and_terminal_fence("embed", True)

    def test_add_owned_commit_stays_inside_model_fence(self):
        self._assert_foreground_writer_and_terminal_fence("add", False)

    def test_add_caller_fence_precedes_writer_upgrade_and_contains_release(self):
        self._assert_foreground_writer_and_terminal_fence("add", True)

    def test_denied_add_savepoint_release_rolls_back_only_add(self):
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-caller','pending')")
        denied = []
        def authorize(action, first, second, _database, _source):
            if action == sqlite3.SQLITE_SAVEPOINT and first == "RELEASE" and second == "rebuild_source" and not denied:
                self.assertTrue(self.vector._lock.locked())
                denied.append(True)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        self.conn.set_authorizer(authorize)
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                self.engine().add("synthetic-denied-release")
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertEqual(denied, [True])
        self.assertTrue(self.conn.in_transaction)
        self.assertFalse(self.vector._lock.locked())
        self.conn.commit()
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-caller'").fetchone()[0], "pending")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.ids("vec_messages_sep"), [])

    def test_cancelled_add_rolls_back_only_add_and_releases_model_fence(self):
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-caller','pending')")
        real = self.conn
        class Probe:
            def __getattr__(self, name):
                return getattr(real, name)

            def execute(self, sql, parameters=()):
                if sql.startswith("INSERT INTO vec_messages_sep("):
                    raise KeyboardInterrupt("synthetic cancellation")
                return real.execute(sql, parameters)
        with self.assertRaises(KeyboardInterrupt):
            self.engine(Probe()).add("synthetic-cancelled")
        self.assertTrue(self.conn.in_transaction)
        self.assertFalse(self.vector._lock.locked())
        self.conn.commit()
        self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-caller'").fetchone()[0], "pending")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.ids("vec_messages_sep"), [])

    def test_failed_add_owned_commit_does_not_publish_on_later_commit(self):
        real = self.conn
        case = self
        class Probe:
            def __getattr__(self, name):
                return getattr(real, name)

            def commit(self):
                case.assertTrue(case.vector._lock.locked())
                raise sqlite3.OperationalError("synthetic commit failure")
        with self.assertRaisesRegex(sqlite3.OperationalError, "synthetic commit failure"):
            self.engine(Probe()).add("synthetic-uncommitted")
        self.assertFalse(self.conn.in_transaction)
        self.assertFalse(self.vector._lock.locked())
        self.conn.commit()
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0)
        self.assertEqual(self.ids(), [])
        self.assertEqual(self.ids("vec_messages_sep"), [])


if __name__ == "__main__":
    unittest.main()
