"""Eight real maintenance adapters with SQLite and synthetic native boundaries."""

import ast
import importlib
import runpy
import sqlite3
import struct
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
PRIMITIVES = runpy.run_path(str(TESTS / "test-maintenance-source-revision-753.py"))
STORAGE, MAINTENANCE = PRIMITIVES["STORAGE"], PRIMITIVES["MAINTENANCE"]
CLUSTERS = runpy.run_path(str(TESTS / "test-cluster-publication-749.py"))


def source_modules() -> dict[str, types.ModuleType]:
    modules = {}
    for name in ("consolidation", "predictive", "temporal", "personality"):
        module = types.ModuleType("synthetic_adapter_" + name)
        path = ROOT / "truememory" / (name + ".py")
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module in {"truememory.storage", "truememory.fts_search"}
        )]
        module._initialize_dunbar_ownership = STORAGE._initialize_dunbar_ownership
        module._dunbar_ownership_ready = STORAGE._dunbar_ownership_ready
        exec(compile(tree, str(path), "exec"), module.__dict__)
        modules["truememory." + name] = module
    return modules


class AdapterFixture(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="synthetic-adapters-")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "synthetic.sqlite"
        self.conn = self.open_db()
        self.modules = source_modules()
        self.cluster, self.vector, self.native = CLUSTERS["load_clustering"]()
        self.modules["truememory.clustering"] = self.cluster
        self.native.fit = lambda: setattr(self.native, "labels", [-1] * len(self.native.inputs[-1]))
        self.conn.execute("CREATE TABLE vec_messages_edge(embedding BLOB)")
        self.conn.execute(
            "INSERT INTO vector_cache_registry VALUES "
            "('edge','vec_messages_edge','vec_messages_sep_edge',0,0,'synthetic-model-a',2,0,0)"
        )
        self.conn.executemany("INSERT INTO metadata(key,value,updated_at) VALUES (?,?,'synthetic-0')", [
            ("embed_model", "synthetic-model-a"), ("embed_dim", "2"),
            ("vec_source_v1:vec_messages_edge", '{"version":1,"generation":"synthetic-a","cursor":0,"complete":true}'),
        ])
        self.conn.commit()
        self.enterContext(patch.dict(sys.modules, {"truememory.vector_search": self.vector}))
        self.enterContext(patch.object(MAINTENANCE, "importlib", types.SimpleNamespace(
            import_module=lambda name: self.modules[name], metadata=importlib.metadata, util=importlib.util,
        )))
        self.versions = {"numpy": "synthetic-1", "hdbscan": "synthetic-1", "sqlite_vec": "synthetic-1"}
        self.enterContext(patch.object(MAINTENANCE, "_installed_dependency", lambda name, dist: self.versions[name]))

    # TestCase.enterContext was added after the oldest supported Python.
    def enterContext(self, context):
        result = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return result

    def open_db(self) -> sqlite3.Connection:
        conn = STORAGE.create_db(self.path)
        conn.execute("PRAGMA busy_timeout=100")
        self.addCleanup(conn.close)
        return conn

    def add(self, count: int = 1, *, commit: bool = True) -> None:
        for _ in range(count):
            cursor = self.conn.execute(
                "INSERT INTO messages(content,sender,recipient,timestamp,category) VALUES "
                "('synthetic team launched Cedar','synthetic-primary','synthetic-contact','2026-01-01','synthetic-session')"
            )
            self.conn.execute("INSERT INTO vec_messages_edge(rowid,embedding) VALUES (?,?)", (cursor.lastrowid, struct.pack("2f", 1, 2)))
            self.conn.execute("UPDATE metadata SET updated_at=? WHERE key IN ('embed_model','embed_dim')", (str(cursor.lastrowid),))
            self.conn.execute("UPDATE vector_cache_registry SET last_embedded_id=?,vector_count=?,last_updated=?",
                              (cursor.lastrowid, cursor.lastrowid, cursor.lastrowid))
        if commit:
            self.conn.commit()

    def specs(self, conn: sqlite3.Connection | None = None):
        return MAINTENANCE.all_layer_specs(conn or self.conn)

    def spec(self, layer: str = "clusters"):
        return next(spec for spec in self.specs() if spec.layer == layer)

    def run_layer(self, layer: str = "clusters", **kwargs):
        return MAINTENANCE.run_layers(self.conn, (self.spec(layer),), **kwargs)[0]

    def state(self, layer: str = "clusters", conn: sqlite3.Connection | None = None):
        conn = conn or self.conn
        spec = next(spec for spec in self.specs(conn) if spec.layer == layer)
        return MAINTENANCE.read_layer_states(conn, (spec,))[layer]

    def output(self, conn: sqlite3.Connection | None = None) -> tuple:
        conn = conn or self.conn
        return (conn.execute("SELECT * FROM message_clusters ORDER BY message_id").fetchall(),
                conn.execute("SELECT * FROM cluster_centroids ORDER BY cluster_id").fetchall())


class TestEightAdapters(AdapterFixture):
    def test_all_eight_actual_builders_and_original_six_contract(self) -> None:
        self.add(5)
        self.native.fit = lambda: setattr(self.native, "labels", [0, 0, -1, 1, 1])
        with patch.object(self.modules["truememory.personality"], "build_entity_profiles", side_effect=AssertionError("not enrolled")), \
             patch.object(self.modules["truememory.personality"], "extract_preferences", side_effect=AssertionError("separate nonpersisted call")):
            results = MAINTENANCE.run_layers(self.conn, self.specs())
        self.assertEqual([item.result_key for item in results], [
            "cluster_messages", "build_summaries", "detect_contradictions", "structured_facts",
            "build_surprise_index", "detect_episodes", "detect_landmarks", "dunbar_hierarchy",
        ])
        self.assertTrue(all(item.outcome in {"success", "success_empty"} for item in results), results)
        self.assertEqual(results[0].output_count, 2)
        self.assertEqual(results[-1].output_count, 1)
        self.assertEqual(results[0].coverage, "vector_generation_unverified")
        self.assertTrue(all(item.coverage == "complete" for item in results[1:]))
        self.assertEqual(len(MAINTENANCE.nonvector_layer_specs()), 6)
        self.assertTrue(all(spec.publication_guard is None for spec in MAINTENANCE.nonvector_layer_specs()))
        self.assertFalse(self.conn.in_transaction)

    def test_empty_all_layers_checkpoint_once_across_reopen(self) -> None:
        results = MAINTENANCE.run_layers(self.conn, self.specs())
        self.assertTrue(all(item.outcome == "success_empty" for item in results), results)
        reopened = self.open_db()
        self.assertEqual(MAINTENANCE.plan_layers(reopened, self.specs(reopened)), ())
        self.assertTrue(all(not item.attempted for item in MAINTENANCE.run_layers(reopened, self.specs(reopened))))

    def test_all_noise_retains_assignments_as_success_empty(self) -> None:
        self.add(5)
        result = self.run_layer()
        self.assertEqual((result.outcome, result.output_count, result.coverage),
                         ("success_empty", 0, "vector_generation_unverified"))
        self.assertEqual(len(self.output()[0]), 5)
        self.assertTrue(all(row[1:] == (-1, 1) for row in self.output()[0]))

    def test_missing_hdbscan_and_runtime_import_failure_are_unavailable_with_siblings(self) -> None:
        self.add(1)
        self.versions["hdbscan"] = None
        results = MAINTENANCE.run_layers(self.conn, self.specs())
        self.assertEqual(results[0].outcome, "unavailable")
        self.assertTrue(all(item.outcome in {"success", "success_empty"} for item in results[1:]))
        self.versions["hdbscan"] = "synthetic-1"
        self.native.missing = True
        result = self.run_layer()
        self.assertEqual((result.outcome, result.error_category), ("unavailable", "LayerUnavailableError"))
        self.assertIsNone(self.state().successful_epoch)
        self.assertFalse(self.run_layer().attempted)

    def test_missing_vector_table_and_active_rebuild_do_not_certify_empty(self) -> None:
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('vec_build_state:vec_messages_edge','in_progress')")
        self.conn.commit()
        self.assertEqual(self.run_layer().outcome, "unavailable")
        self.assertIsNone(self.state().successful_epoch)
        self.conn.execute("UPDATE metadata SET value='complete' WHERE key='vec_build_state:vec_messages_edge'")
        self.conn.execute("ALTER TABLE vec_messages_edge RENAME TO synthetic_old_vectors")
        self.conn.commit()
        result = self.run_layer()
        self.assertEqual((result.outcome, result.error_category), ("unavailable", "VectorIndexMissing"))

    def test_manifest_absence_is_unverified_and_invalid_or_incomplete_is_unavailable(self) -> None:
        self.conn.execute("DELETE FROM metadata WHERE key='vec_source_v1:vec_messages_edge'")
        self.conn.commit()
        self.assertEqual(self.run_layer().coverage, "vector_generation_unverified")
        for descriptor in (b'synthetic bytes', 'synthetic invalid JSON', 'null', '[]', '{}', '{"version":2,"complete":true}',
                           '{"version":true,"complete":true}', '{"version":1,"complete":1}',
                           '{"version":1,"complete":"true"}', '{"version":1,"complete":false}'):
            with self.subTest(descriptor=descriptor):
                self.conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('vec_source_v1:vec_messages_edge',?)",
                                  (descriptor,))
                self.conn.commit()
                result = self.run_layer()
                self.assertEqual(result.outcome, "unavailable")
                self.assertEqual(result.coverage, "vector_generation_unverified")

    def test_dunbar_primary_ties_legacy_contacts_and_empty_retirement(self) -> None:
        self.add(3)
        self.conn.execute(
            "INSERT INTO entity_relationships(entity_a,entity_b,relationship_type) "
            "VALUES ('synthetic-primary','synthetic-contact','contact')"
        )
        self.conn.commit()
        result = self.run_layer("dunbar")
        self.assertEqual((result.output_count, result.coverage), (1, "legacy_contacts_unowned"))
        self.assertEqual(self.conn.execute("SELECT count(*) FROM entity_relationships").fetchone()[0], 2)
        self.conn.execute("DELETE FROM messages")
        self.conn.commit()
        result = self.run_layer("dunbar")
        self.assertEqual((result.outcome, result.output_count, result.coverage), ("success_empty", 0, "legacy_contacts_unowned"))
        self.assertEqual(self.conn.execute("SELECT count(*) FROM entity_relationships").fetchone()[0], 1)

    def test_actual_primary_selection_matches_existing_tied_sender_query(self) -> None:
        self.conn.executemany("INSERT INTO messages(content,sender,recipient) VALUES ('synthetic',?,?)", [
            ("synthetic-beta", "synthetic-contact"), ("synthetic-alpha", "synthetic-other"),
        ])
        self.conn.commit()
        expected = self.conn.execute(
            "SELECT sender,COUNT(*) AS cnt FROM messages WHERE sender != '' AND sender IS NOT NULL "
            "GROUP BY sender ORDER BY cnt DESC LIMIT 1"
        ).fetchone()[0].lower()
        self.run_layer("dunbar")
        self.assertEqual(self.conn.execute("SELECT DISTINCT entity_a FROM entity_relationships").fetchall(), [(expected,)])


class TestAdapterScheduling(AdapterFixture):
    def test_24_actual_appends_with_metadata_progress_rewrites_wait_for_25(self) -> None:
        self.run_layer()
        original_key = self.spec().resolve_dependency().key
        self.add(24)
        self.assertEqual(self.spec().resolve_dependency().key, original_key)
        self.assertFalse(self.run_layer().attempted)
        self.add(1)
        self.assertTrue(self.run_layer().attempted)
        self.assertEqual(self.state().attempted_insert_count, 25)

    def test_failed_and_unavailable_initial_cluster_attempts_use_25_insert_baseline(self) -> None:
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable):
                self.conn.execute("DELETE FROM maintenance_layers WHERE layer='clusters'")
                self.conn.commit()
                self.versions["hdbscan"] = None if unavailable else "synthetic-1"
                with patch.object(self.cluster, "cluster_messages", side_effect=ValueError("synthetic failure")):
                    result = self.run_layer()
                    self.assertEqual(result.outcome, "unavailable" if unavailable else "failed")
                    self.add(1)
                    self.assertFalse(self.run_layer().attempted)
                    self.add(23)
                    self.assertFalse(self.run_layer().attempted)
                    self.add(1)
                    self.assertTrue(self.run_layer().attempted)

    def test_correction_replace_dependency_and_force_wake_below_25(self) -> None:
        self.add(1)
        self.run_layer()
        for sql in ("UPDATE messages SET content='synthetic changed' WHERE id=1",
                    "INSERT OR REPLACE INTO messages(id,content) VALUES (1,'synthetic replacement')"):
            self.conn.execute(sql)
            self.conn.commit()
            self.assertTrue(self.run_layer().attempted)
        self.versions["hdbscan"] = "synthetic-2"
        self.assertTrue(self.run_layer().attempted)
        self.assertTrue(self.run_layer(force=True).attempted)
        self.assertFalse(self.run_layer().attempted)

    def test_manifest_progress_and_same_dimensional_model_identity_change_key(self) -> None:
        previous = self.spec().resolve_dependency().key
        self.conn.execute("UPDATE metadata SET value=? WHERE key='vec_source_v1:vec_messages_edge'",
                          ('{"version":1,"generation":"synthetic-a","cursor":1,"complete":true}',))
        self.conn.commit()
        current = self.spec().resolve_dependency().key
        self.assertNotEqual(current, previous)
        self.vector.EMBEDDING_MODEL = "synthetic-model-c"
        self.assertNotEqual(self.spec().resolve_dependency().key, current)

    def test_foreground_eligibility_never_imports_native_modules_or_waits_for_model(self) -> None:
        original = MAINTENANCE.importlib.import_module
        imported = []

        def safe_import(name):
            imported.append(name)
            self.assertNotIn(name, {"numpy", "hdbscan", "sqlite_vec", "truememory.clustering", "truememory.vector_search"})
            return original(name)

        with patch.object(MAINTENANCE.importlib, "import_module", safe_import):
            self.vector._lock.acquire()
            try:
                self.assertEqual(self.spec().resolve_dependency().error_category, "ModelBusy")
                planned = MAINTENANCE.plan_layers(self.conn, self.specs())
                self.assertEqual(len(planned), 7)
                self.assertNotIn("clusters", {spec.layer for spec in planned})
            finally:
                self.vector._lock.release()
        self.assertNotIn("truememory.clustering", imported)
        self.assertFalse(self.conn.in_transaction)

    def test_adapters_reject_wrong_connection_before_running_or_recording(self) -> None:
        other = self.open_db()
        with self.assertRaisesRegex(ValueError, "another connection"):
            MAINTENANCE.run_layers(other, self.specs())
        self.assertEqual(other.execute("SELECT DISTINCT outcome FROM maintenance_layers").fetchall(), [("pending",)])


class TestTransientModelContention(AdapterFixture):
    def row(self) -> tuple | None:
        row = self.conn.execute("SELECT * FROM maintenance_layers WHERE layer='clusters'").fetchone()
        return tuple(row) if row is not None else None

    def release_model(self) -> None:
        if self.vector._lock.locked():
            self.vector._lock.release()

    def block_during_compute(self, cancel: threading.Event | None = None) -> None:
        def fit():
            self.native.labels = [-1] * len(self.native.inputs[-1])
            self.vector._lock.acquire()
            if cancel is not None:
                cancel.set()
        self.native.fit = fit
        self.addCleanup(self.release_model)

    def test_initial_busy_and_force_never_write_or_fabricate_attempt(self) -> None:
        self.add(5)
        for absent in (False, True):
            if absent:
                self.conn.execute("DELETE FROM maintenance_layers WHERE layer='clusters'")
                self.conn.commit()
            old, changes = self.row(), self.conn.total_changes
            self.vector._lock.acquire()
            try:
                for force in (False, True):
                    with self.subTest(absent=absent, force=force):
                        result = self.run_layer(force=force)
                        self.assertEqual((result.outcome, result.error_category, result.attempted),
                                         ("deferred", "ModelBusy", False))
                        self.assertEqual(self.row(), old)
                        self.assertEqual(self.conn.total_changes, changes)
                        self.assertEqual(MAINTENANCE.plan_layers(self.conn, (self.spec(),), force=force), ())
                        self.assertEqual(MAINTENANCE.layer_freshness(
                            self.state().source, self.state(), self.spec().resolve_dependency()), "dependency_deferred")
            finally:
                self.vector._lock.release()
        self.assertEqual(self.run_layer().outcome, "success_empty")
        self.assertFalse(self.run_layer().attempted)

    def test_busy_after_one_append_preserves_twenty_five_append_baseline(self) -> None:
        self.add(5)
        self.run_layer()
        previous = self.row()
        self.add(1)
        self.assertFalse(self.run_layer().attempted)
        self.vector._lock.acquire()
        try:
            self.assertEqual(self.run_layer(force=True).outcome, "deferred")
        finally:
            self.vector._lock.release()
        self.assertEqual(self.row(), previous)
        self.assertFalse(self.run_layer().attempted)
        self.add(23)
        self.assertFalse(self.run_layer().attempted)
        self.add(1)
        self.assertTrue(self.run_layer().attempted)

    def test_mid_compute_busy_restores_every_prior_field_and_original_output(self) -> None:
        self.add(5)
        self.run_layer()
        self.conn.execute("UPDATE maintenance_layers SET builder_version=7,outcome='failed',"
                          "error_category='SyntheticFailure',run_generation='synthetic-old' WHERE layer='clusters'")
        self.conn.commit()
        previous, output = self.row(), self.output()
        self.add(1)
        self.block_during_compute()
        result = self.run_layer(force=True)
        self.release_model()
        self.assertEqual((result.outcome, result.attempted), ("deferred", True))
        self.assertEqual(self.row(), previous)
        self.assertEqual(self.output(), output)
        self.assertFalse(self.conn.in_transaction)
        self.assertFalse(self.run_layer().attempted)

    def test_mid_compute_busy_restores_absent_row_only_after_output_rollback(self) -> None:
        self.add(1)
        self.conn.execute("DELETE FROM maintenance_layers WHERE layer='clusters'")
        self.conn.commit()
        self.block_during_compute()
        original, observations = MAINTENANCE._restore_deferred_attempt, []

        def restore(*args):
            observations.append((self.conn.in_transaction, self.output()))
            return original(*args)

        with patch.object(MAINTENANCE, "_restore_deferred_attempt", restore):
            result = self.run_layer()
        self.release_model()
        self.assertEqual((result.outcome, result.attempted), ("deferred", True))
        self.assertEqual(observations, [(False, ([], []))])
        self.assertIsNone(self.row())
        self.native.fit = lambda: setattr(self.native, "labels", [-1])
        self.assertTrue(self.run_layer().attempted)

    def test_busy_at_postbuild_dependency_and_outer_guard_rolls_back_checkpoint(self) -> None:
        self.add(1)
        self.run_layer()
        self.add(1)
        old_row, old_output = self.row(), self.output()
        for boundary in ("dependency", "guard"):
            with self.subTest(boundary=boundary):
                spec, after_build = self.spec(), []

                def count(conn):
                    after_build.append(0)
                    return spec.count_output(conn)

                def resolve():
                    if after_build:
                        after_build[0] += 1
                        if boundary == "dependency" and after_build[0] == 1:
                            self.vector._lock.acquire()
                    dependency = spec.resolve_dependency()
                    if boundary == "guard" and after_build and after_build[0] == 2:
                        self.vector._lock.acquire()
                    return dependency

                try:
                    result = MAINTENANCE.run_layers(self.conn, (spec._replace(
                        count_output=count, resolve_dependency=resolve),), force=True)[0]
                finally:
                    self.release_model()
                self.assertEqual((result.outcome, result.attempted), ("deferred", True))
                self.assertEqual(self.row(), old_row)
                self.assertEqual(self.output(), old_output)
                self.assertFalse(self.run_layer().attempted)

    def test_plain_runtime_error_is_a_failed_attempt_not_a_busy_deferral(self) -> None:
        self.add(1)
        with patch.object(self.cluster, "cluster_messages", side_effect=RuntimeError("Embedding model is busy")):
            result = self.run_layer()
        self.assertEqual((result.outcome, result.error_category, result.attempted), ("failed", "RuntimeError", True))
        self.assertEqual(self.row()[2], "failed")

    def test_cancellation_racing_busy_restores_baseline_and_stops_siblings(self) -> None:
        self.add(1)
        previous = self.row()
        cancel = threading.Event()
        self.block_during_compute(cancel)
        results = MAINTENANCE.run_layers(self.conn, self.specs(), cancel=cancel)
        self.release_model()
        self.assertEqual(len(results), 1)
        self.assertEqual((results[0].outcome, results[0].attempted), ("deferred", True))
        self.assertEqual(self.row(), previous)
        self.assertEqual(self.output(), ([], []))

    def test_changed_diagnostic_is_not_overwritten_even_with_same_run_generation(self) -> None:
        self.add(1)
        self.block_during_compute()
        restore = MAINTENANCE._restore_deferred_attempt
        writer = self.open_db()

        def replace_diagnostic(*args):
            writer.execute("UPDATE maintenance_layers SET error_category='ExternalChange' WHERE layer='clusters'")
            writer.commit()
            return restore(*args)

        with patch.object(MAINTENANCE, "_restore_deferred_attempt", replace_diagnostic):
            with self.assertRaisesRegex(MAINTENANCE.MaintenanceBusyError, "diagnostic changed"):
                self.run_layer()
        self.assertEqual(self.conn.execute("SELECT error_category FROM maintenance_layers WHERE layer='clusters'").fetchone()[0],
                         "ExternalChange")
        self.assertEqual(self.output(), ([], []))
        self.assertFalse(self.conn.in_transaction)

    def test_changed_owner_or_generation_is_not_overwritten(self) -> None:
        self.add(1)
        self.block_during_compute()
        restore = MAINTENANCE._restore_deferred_attempt
        for kind in ("owner", "generation"):
            with self.subTest(kind=kind):
                def change_ownership(conn, layer, previous, running, owner):
                    if kind == "generation":
                        conn.execute("UPDATE maintenance_layers SET run_generation='synthetic-other' WHERE layer=?", (layer,))
                        conn.commit()
                        return restore(conn, layer, previous, running, owner)
                    owners = MAINTENANCE._thread_owners.owners
                    owners[owner.path] = owner._replace(generation="synthetic-other")
                    try:
                        return restore(conn, layer, previous, running, owner)
                    finally:
                        owners[owner.path] = owner

                with patch.object(MAINTENANCE, "_restore_deferred_attempt", change_ownership):
                    with self.assertRaises(MAINTENANCE.MaintenanceBusyError):
                        self.run_layer(force=True)
                self.release_model()
                self.assertEqual(self.row()[2], "running")
                if kind == "generation":
                    self.assertEqual(self.conn.execute("SELECT run_generation FROM maintenance_layers WHERE layer='clusters'").fetchone()[0],
                                     "synthetic-other")
                self.assertEqual(self.output(), ([], []))

    def test_failed_restoration_commit_propagates_and_leaves_running_diagnostic(self) -> None:
        self.add(1)
        self.block_during_compute()
        restore, rejected = MAINTENANCE._restore_deferred_attempt, []

        def fail_commit(*args):
            def authorize(action, first, *_args):
                if action == sqlite3.SQLITE_TRANSACTION and first == "COMMIT" and not rejected:
                    rejected.append(True)
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            self.conn.set_authorizer(authorize)
            return restore(*args)

        try:
            with patch.object(MAINTENANCE, "_restore_deferred_attempt", fail_commit):
                with self.assertRaises(sqlite3.DatabaseError):
                    self.run_layer()
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertEqual(rejected, [True])
        self.assertEqual(self.row()[2], "running")
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.output(), ([], []))

    def test_output_rollback_failure_never_restores_diagnostic(self) -> None:
        self.add(1)
        restore = MAINTENANCE._restore_deferred_attempt

        def fail(conn):
            conn.execute("INSERT INTO message_clusters(message_id,cluster_id,noise) VALUES (1,-1,1)")
            conn.set_authorizer(lambda action, first, *_args: sqlite3.SQLITE_DENY
                                if action == sqlite3.SQLITE_TRANSACTION and first == "ROLLBACK" else sqlite3.SQLITE_OK)
            raise self.cluster.ClusterModelBusyError("synthetic contention")

        try:
            with patch.object(self.cluster, "cluster_messages", fail), \
                 patch.object(MAINTENANCE, "_restore_deferred_attempt", wraps=restore) as restoration:
                with self.assertRaises(MAINTENANCE._LayerRollbackFailed):
                    self.run_layer()
                restoration.assert_not_called()
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
            self.conn.rollback()
        self.assertEqual(self.row()[2], "running")
        self.assertEqual(self.output(), ([], []))

    def test_borrowed_busy_rolls_back_only_layer_without_attempt_metadata(self) -> None:
        self.add(1, commit=False)
        previous = self.row()
        self.block_during_compute()
        result = self.run_layer(allow_caller_transaction=True)
        self.release_model()
        self.assertEqual((result.outcome, result.attempted, result.pending_caller_commit), ("deferred", True, True))
        self.assertEqual(self.row(), previous)
        self.assertEqual(self.output(), ([], []))
        self.assertTrue(self.conn.in_transaction)
        self.conn.commit()
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)
        self.assertEqual(self.output(), ([], []))

    def test_model_owner_arriving_after_probe_never_blocks_inside_transaction(self) -> None:
        self.add(1)
        self.run_layer()
        self.add(1)
        previous, output = self.row(), self.output()
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                if borrowed:
                    self.conn.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-caller','retained')")
                spec, attempts = self.spec(), []
                held = threading.Lock()

                class BoundaryLock:
                    armed = False

                    def acquire(lock, blocking=True):
                        if lock.armed:
                            attempts.append(blocking)
                            if blocking:
                                raise AssertionError("Blocking model acquisition while SQLite transaction is retained")
                        return held.acquire(blocking=blocking)

                    def release(lock):
                        held.release()

                    def __enter__(lock):
                        lock.acquire()
                        return lock

                    def __exit__(lock, *_args):
                        lock.release()

                boundary = BoundaryLock()

                def build(conn):
                    self.assertTrue(conn.in_transaction)
                    held.acquire()
                    boundary.armed = True
                    return spec.build(conn)

                with patch.object(self.vector, "_lock", boundary):
                    try:
                        result = MAINTENANCE.run_layers(self.conn, (spec._replace(build=build),),
                            force=True, allow_caller_transaction=borrowed)[0]
                        self.assertTrue(held.locked(), "The other model owner's lock must not be released")
                    finally:
                        held.release()
                self.assertEqual((result.outcome, result.attempted), ("deferred", True))
                self.assertEqual(attempts, [False])
                self.assertEqual(self.row(), previous)
                self.assertEqual(self.output(), output)
                self.assertEqual(self.conn.in_transaction, borrowed)
                if borrowed:
                    self.assertEqual(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-caller'").fetchone()[0], "retained")
                    self.conn.rollback()
                    self.assertIsNone(self.conn.execute("SELECT value FROM metadata WHERE key='synthetic-caller'").fetchone())

    def test_standalone_initial_capture_wait_policy_stays_outside_sqlite_transaction(self) -> None:
        self.add(1)
        attempts = []
        held = threading.Lock()
        conn = self.conn

        class RecordingLock:
            def acquire(lock, blocking=True):
                attempts.append((blocking, conn.in_transaction))
                return held.acquire(blocking=blocking)

            def release(lock):
                held.release()

            def __enter__(lock):
                lock.acquire()
                return lock

            def __exit__(lock, *_args):
                lock.release()

        with patch.object(self.vector, "_lock", RecordingLock()):
            self.assertEqual(self.cluster.cluster_messages(self.conn), 0)
        self.assertEqual(attempts, [(True, False), (False, True)])


class TestRowFactoryCompatibility(AdapterFixture):
    def test_row_factory_supports_real_cluster_builder_and_all_adapters(self) -> None:
        self.add(5)
        original_key = self.spec().resolve_dependency().key
        self.conn.row_factory = sqlite3.Row
        self.assertEqual(self.cluster.cluster_messages(self.conn), 0)
        dependency = self.spec().resolve_dependency()
        self.assertTrue(dependency.available, dependency)
        self.assertEqual(dependency.key, original_key)
        results = MAINTENANCE.run_layers(self.conn, self.specs())
        self.assertTrue(all(item.outcome in {"success", "success_empty"} for item in results), results)
        self.assertFalse(self.run_layer().attempted)
        self.assertTrue(all(not state.dependency.deferred for state in MAINTENANCE.read_layer_states(self.conn, self.specs()).values()))

    def test_row_factory_detects_explicit_rebuild_in_progress(self) -> None:
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("INSERT INTO metadata(key,value) VALUES ('vec_build_state:vec_messages_edge','in_progress')")
        self.conn.commit()
        dependency = self.spec().resolve_dependency()
        self.assertEqual((dependency.available, dependency.error_category, dependency.deferred),
                         (False, "VectorRebuildInProgress", False))
        self.assertEqual(self.run_layer().outcome, "unavailable")


class TestCoveragePublication(AdapterFixture):
    def test_failure_preserves_success_coverage_and_previous_output(self) -> None:
        self.add(1)
        self.run_layer()
        old_output, old_state = self.output(), self.state()
        with patch.object(self.cluster, "cluster_messages", side_effect=ValueError("synthetic failure")):
            result = self.run_layer(force=True)
        self.assertEqual(result.coverage, "vector_generation_unverified")
        self.assertEqual(self.output(), old_output)
        self.assertEqual(self.state().successful_coverage, old_state.successful_coverage)
        self.assertEqual(self.state().successful_dependency, old_state.successful_dependency)

    def test_output_coverage_checkpoint_visible_together_at_real_commit(self) -> None:
        self.add(1)
        reader = self.open_db()
        observations = []

        def trace(sql):
            if sql == "COMMIT" and self.vector._lock.locked():
                observations.append((self.output(reader), self.state(conn=reader).successful_coverage))

        spec = self.spec()
        original = spec.count_output

        def count(conn):
            conn.set_trace_callback(trace)
            return original(conn)

        result = MAINTENANCE.run_layers(self.conn, (spec._replace(count_output=count),))[0]
        self.conn.set_trace_callback(None)
        self.assertEqual(result.outcome, "success_empty")
        self.assertEqual(observations, [(([], []), "unverified")])
        self.assertEqual(self.state(conn=reader).successful_coverage, "vector_generation_unverified")
        self.assertEqual(len(self.output(reader)[0]), 1)

    def test_coverage_failure_rolls_back_generated_output_and_old_coverage(self) -> None:
        self.add(1)
        self.run_layer()
        previous = self.output(), self.state().successful_coverage
        spec = self.spec()._replace(read_coverage=lambda conn: "invalid")
        result = MAINTENANCE.run_layers(self.conn, (spec,), force=True)[0]
        self.assertEqual(result.outcome, "failed")
        self.assertEqual((self.output(), self.state().successful_coverage), previous)

    def test_metadata_timestamp_change_does_not_schedule_but_rejects_inflight_publication(self) -> None:
        self.add(1)
        spec = self.spec()
        original = spec.count_output

        def changed(conn):
            result = original(conn)
            conn.execute("UPDATE metadata SET updated_at='synthetic-new' WHERE key='embed_model'")
            return result

        result = MAINTENANCE.run_layers(self.conn, (spec._replace(count_output=changed),))[0]
        self.assertEqual(result.outcome, "failed")
        self.assertEqual(self.output(), ([], []))
        self.assertEqual(self.state().successful_coverage, "unverified")

    def test_coverage_migration_preserves_epoch_and_provenance_and_is_idempotent(self) -> None:
        self.add(1)
        self.run_layer()
        token, old = MAINTENANCE.read_source_revision(self.conn), self.state()
        self.conn.execute("ALTER TABLE maintenance_layers RENAME TO synthetic_old_layers")
        fields = [row[1] for row in self.conn.execute("PRAGMA table_info(synthetic_old_layers)") if row[1] != "successful_coverage"]
        self.conn.execute("CREATE TABLE maintenance_layers AS SELECT " + ",".join(fields) + " FROM synthetic_old_layers")
        self.conn.execute("CREATE UNIQUE INDEX synthetic_layer_name ON maintenance_layers(layer)")
        self.conn.commit()
        STORAGE._initialize_maintenance_tracking(self.conn)
        self.assertEqual(MAINTENANCE.read_source_revision(self.conn), token)
        self.assertEqual(self.state().successful_dependency, old.successful_dependency)
        self.assertEqual(self.state().successful_coverage, "unverified")
        version, changes = self.conn.execute("PRAGMA schema_version").fetchone()[0], self.conn.total_changes
        STORAGE._initialize_maintenance_tracking(self.conn)
        self.assertEqual((self.conn.execute("PRAGMA schema_version").fetchone()[0], self.conn.total_changes), (version, changes))

    def test_owned_commit_failure_preserves_old_output_and_coverage(self) -> None:
        self.add(1)
        self.run_layer()
        previous = self.output(), self.state().successful_coverage
        self.native.fit = lambda: setattr(self.native, "labels", [0])
        denied = []

        def authorize(action, first, *_args):
            if action == sqlite3.SQLITE_TRANSACTION and first == "COMMIT" and not denied:
                denied.append(True)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        spec = self.spec()
        original = spec.count_output

        def count(conn):
            conn.set_authorizer(authorize)
            return original(conn)

        try:
            result = MAINTENANCE.run_layers(self.conn, (spec._replace(count_output=count),), force=True)[0]
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertEqual((result.outcome, denied), ("failed", [True]))
        self.assertEqual((self.output(), self.state().successful_coverage), previous)
        self.assertFalse(self.vector._lock.locked())

    def test_coverage_read_snapshot_does_not_mix_concurrent_publications(self) -> None:
        self.add(1)
        self.run_layer("dunbar")
        writer = self.open_db()
        spec = self.spec("dunbar")
        with MAINTENANCE.layer_read_snapshot(self.conn, "dunbar", spec.resolve_dependency()) as state:
            writer.execute("INSERT INTO entity_relationships(entity_a,entity_b,relationship_type) "
                           "VALUES ('synthetic-other','synthetic-contact','contact')")
            writer.commit()
            writer_spec = next(item for item in self.specs(writer) if item.layer == "dunbar")
            result = MAINTENANCE.run_layers(writer, (writer_spec,), force=True)[0]
            self.assertEqual(result.coverage, "legacy_contacts_unowned")
            self.assertEqual(state.successful_coverage, "complete")
            self.assertEqual(self.modules["truememory.personality"].read_dunbar_coverage(self.conn)["status"], "managed_only")
        self.assertEqual(self.state("dunbar").successful_coverage, "legacy_contacts_unowned")


class TestBorrowedPublication(AdapterFixture):
    def test_coverage_migration_and_output_remain_inside_caller_transaction(self) -> None:
        self.conn.execute("ALTER TABLE maintenance_layers RENAME TO synthetic_old_layers")
        fields = [row[1] for row in self.conn.execute("PRAGMA table_info(synthetic_old_layers)") if row[1] != "successful_coverage"]
        self.conn.execute("CREATE TABLE maintenance_layers AS SELECT " + ",".join(fields) + " FROM synthetic_old_layers")
        self.conn.execute("CREATE UNIQUE INDEX synthetic_layer_name ON maintenance_layers(layer)")
        self.conn.commit()
        token = MAINTENANCE.read_source_revision(self.conn)
        self.add(1, commit=False)
        STORAGE._initialize_maintenance_tracking(self.conn)
        result = self.run_layer(allow_caller_transaction=True)
        self.assertTrue(result.pending_caller_commit)
        self.conn.rollback()
        self.assertNotIn("successful_coverage", {row[1] for row in self.conn.execute("PRAGMA table_info(maintenance_layers)")})
        self.assertEqual(MAINTENANCE.read_source_revision(self.conn), token)
        self.assertEqual(self.output(), ([], []))

    def test_default_rejects_caller_transaction_and_opt_in_keeps_all_work_rollbackable(self) -> None:
        self.add(2, commit=False)
        with self.assertRaisesRegex(RuntimeError, "clean transaction"):
            MAINTENANCE.run_layers(self.conn, self.specs())
        reader = self.open_db()
        results = MAINTENANCE.run_layers(self.conn, self.specs(), allow_caller_transaction=True)
        self.assertTrue(all(item.pending_caller_commit for item in results))
        self.assertTrue(all(item.outcome in {"success", "success_empty"} for item in results), results)
        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(reader.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
        self.assertEqual(self.state(conn=reader).successful_coverage, "unverified")
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
        self.assertEqual(self.output(), ([], []))
        self.assertEqual(self.state().successful_coverage, "unverified")

    def test_read_only_caller_does_not_take_writer_for_running_diagnostic(self) -> None:
        self.add(1)
        writer = self.open_db()
        observed = []

        def compute():
            self.native.labels = [-1]
            writer.execute("INSERT INTO metadata(key,value) VALUES ('synthetic-writer','yes')")
            writer.commit()
            observed.append(True)

        self.native.fit = compute
        self.conn.execute("BEGIN")
        # The writer deliberately makes the read snapshot stale. Publication
        # and its failure diagnostic cannot upgrade that snapshot either.
        with self.assertRaises(sqlite3.OperationalError):
            self.run_layer(allow_caller_transaction=True)
        self.assertEqual(observed, [True])
        self.assertEqual(self.output(), ([], []))
        self.conn.rollback()

    def test_failed_borrowed_layer_and_successful_sibling_preserve_unrelated_writes(self) -> None:
        self.add(1, commit=False)
        with patch.object(self.cluster, "cluster_messages", side_effect=ValueError("synthetic failure")):
            results = MAINTENANCE.run_layers(self.conn, self.specs(), allow_caller_transaction=True)
        self.assertEqual(results[0].outcome, "failed")
        self.assertTrue(all(item.outcome in {"success", "success_empty"} for item in results[1:]))
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()
        self.assertEqual(self.state().outcome, "pending")

    def test_cluster_savepoint_release_holds_model_lock_but_return_releases_it(self) -> None:
        self.add(1, commit=False)
        releases = []
        self.conn.set_trace_callback(lambda sql: releases.append(self.vector._lock.locked())
                                     if sql.startswith("RELEASE truememory_layer_") else None)
        result = self.run_layer(allow_caller_transaction=True)
        self.conn.set_trace_callback(None)
        self.assertEqual(releases, [True])
        self.assertTrue(result.pending_caller_commit)
        self.assertEqual(result.coverage, "vector_generation_unverified")
        self.assertFalse(self.vector._lock.locked())
        self.vector.EMBEDDING_MODEL = "synthetic-model-c"
        self.conn.commit()
        self.assertEqual(self.state().successful_coverage, "vector_generation_unverified")

    def test_successful_release_failure_rolls_back_output_before_caller_commit(self) -> None:
        self.add(1, commit=False)
        rejected = []

        def authorize(action, first, second, *_args):
            if action == sqlite3.SQLITE_SAVEPOINT and first == "RELEASE" and second.startswith("truememory_layer_") and not rejected:
                rejected.append(True)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        self.conn.set_authorizer(authorize)
        try:
            result = self.run_layer(allow_caller_transaction=True)
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertEqual(result.outcome, "failed")
        self.assertEqual(rejected, [True])
        self.conn.commit()
        self.assertEqual(self.output(), ([], []))
        self.assertEqual(self.state().successful_coverage, "unverified")
        self.assertEqual(self.conn.execute("SELECT count(*) FROM messages").fetchone()[0], 1)

    def test_rejected_rollback_never_continues_with_attempt_write_or_release(self) -> None:
        self.add(1, commit=False)
        seen = []

        def authorize(action, first, second, *_args):
            if action == sqlite3.SQLITE_SAVEPOINT:
                seen.append(first)
                if first == "ROLLBACK" and second.startswith("truememory_layer_"):
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        self.conn.set_authorizer(authorize)
        try:
            with patch.object(self.cluster, "cluster_messages", side_effect=ValueError("synthetic failure")):
                with self.assertRaises(MAINTENANCE._LayerRollbackFailed):
                    self.run_layer(allow_caller_transaction=True)
        finally:
            self.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        self.assertEqual(seen, ["BEGIN", "ROLLBACK"])
        self.assertFalse(self.vector._lock.locked())
        self.assertEqual(self.state().outcome, "pending")
        self.conn.rollback()

    def test_cancellation_and_keyboard_interrupt_keep_caller_transaction(self) -> None:
        self.add(1, commit=False)
        cancel = threading.Event()
        spec = self.spec()
        original = spec.count_output

        def count(conn):
            cancel.set()
            return original(conn)

        result = MAINTENANCE.run_layers(self.conn, (spec._replace(count_output=count),),
                                        allow_caller_transaction=True, cancel=cancel)[0]
        self.assertEqual(result.outcome, "abandoned")
        self.assertTrue(result.pending_caller_commit)
        self.assertEqual(self.output(), ([], []))
        with patch.object(self.cluster, "cluster_messages", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self.run_layer(allow_caller_transaction=True, force=True)
        self.assertTrue(self.conn.in_transaction)
        self.assertFalse(self.vector._lock.locked())
        self.conn.rollback()

    def test_generic_guard_requires_explicit_borrowed_contract(self) -> None:
        self.add(1, commit=False)
        spec = self.spec()._replace(borrowed_publication_guard=None)
        result = MAINTENANCE.run_layers(self.conn, (spec,), allow_caller_transaction=True)[0]
        self.assertEqual(result.outcome, "unavailable")
        self.assertEqual(self.output(), ([], []))
        self.conn.rollback()

    def test_borrowed_source_change_after_compute_rolls_back_layer_only(self) -> None:
        self.add(1, commit=False)
        spec = self.spec()
        original = spec.count_output

        def count(conn):
            result = original(conn)
            conn.execute("UPDATE messages SET content='synthetic stale result' WHERE id=1")
            return result

        result = MAINTENANCE.run_layers(self.conn, (spec._replace(count_output=count),),
                                        allow_caller_transaction=True)[0]
        self.assertEqual(result.outcome, "failed")
        self.assertEqual(self.output(), ([], []))
        self.assertEqual(self.conn.execute("SELECT content FROM messages").fetchone()[0], "synthetic team launched Cedar")
        self.assertEqual(self.state().successful_coverage, "unverified")
        self.conn.rollback()


if __name__ == "__main__":
    unittest.main()
