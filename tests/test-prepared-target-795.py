"""Prepared targets through real control flow, stdlib stubs and in-memory SQL."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
import gc
import math
import os
from pathlib import Path
import re
import runpy
import sqlite3
import sys
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch
import weakref


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEXTS: dict[str, str] = {}
BASE = runpy.run_path(str(Path(__file__).with_name("test-tier-target-model-795.py")))


def load(name: str, *, definitions: bool = False, namespace: dict | None = None) -> types.ModuleType:
    path = ROOT / "truememory" / f"{name}.py"
    tree = ast.parse(SOURCE_TEXTS.get(name, path.read_text()))
    if definitions:
        tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                     *[node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]]
    tree = ast.fix_missing_locations(tree)
    module = types.ModuleType(f"truememory.{name}")
    if namespace:
        module.__dict__.update(namespace)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


class Array(BASE["Array"]):
    def __getitem__(self, key: slice) -> Array:
        return Array(self.rows[key], (len(self.rows[key]), self.shape[1]))


class Numpy(BASE["Numpy"]):
    @staticmethod
    def empty(shape: tuple[int, int], **_kwargs: object) -> Array:
        return Array([[] for _ in range(shape[0])], shape)


class VecConnection:
    """Actual SQLite transactions, with only native vec0 DDL replaced by a table."""

    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.schemas: dict[str, str] = {}
        self.fail_table: str | None = None
        self.extensions: list[bool] = []

    @property
    def in_transaction(self) -> bool:
        return self.db.in_transaction

    def enable_load_extension(self, enabled: bool) -> None:
        self.extensions.append(enabled)

    def execute(self, sql: str, parameters: tuple = ()):
        match = re.fullmatch(r'CREATE VIRTUAL TABLE main."(\w+)" USING (vec0\(.+\))', sql)
        if match:
            name = match[1]
            if name == self.fail_table:
                raise sqlite3.OperationalError("synthetic second-table denial")
            result = self.db.execute(f'CREATE TABLE main."{name}"(embedding BLOB)')
            self.schemas[name] = f'CREATE VIRTUAL TABLE "{name}" USING {match[2]}'
            return result
        result = self.db.execute(sql, parameters)
        if sql == "SELECT type, sql FROM main.sqlite_master WHERE name = ?":
            row = result.fetchone()
            if row and parameters[0] in self.schemas:
                row = (row[0], self.schemas[parameters[0]])
            return types.SimpleNamespace(fetchone=lambda: row)
        return result

    def commit(self) -> None:
        self.db.commit()

    def rollback(self) -> None:
        self.db.rollback()

    def close(self) -> None:
        self.db.close()


class TestPreparedTargets(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = BASE["TestTierTargetIdentity"]()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.target_module = load("embedding_target")
        self.config = load("tier_config")
        environment = patch.dict("os.environ", {
            "TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD": "1",
            "TRUEMEMORY_CUSTOM_EMBED_MODEL": "synthetic/encoder",
            "TRUEMEMORY_CUSTOM_EMBED_DIM": "384",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.Target = self.target_module.EmbeddingTarget
        self.target = self.Target.capture("base")
        self.calls: list[tuple[str, dict]] = []
        self.encoded: list[tuple[str, list[str], dict]] = []
        self.width: int | None = None
        self.build_hook = lambda: None
        self.encode_hook = lambda: None
        self.refs: list[weakref.ReferenceType] = []
        owner = self

        class Model:
            def __init__(self, identity: str, **kwargs: object):
                owner.refs.append(weakref.ref(self))
                owner.build_hook()
                owner.calls.append((identity, kwargs))
                self.identity = identity
                self.dimension = kwargs.get("truncate_dim", 256)
                self.device = kwargs.get("device", "cpu")
                self.oom_once = False

            def encode(self, texts: list[str], **kwargs: object) -> Array:
                owner.encode_hook()
                owner.encoded.append((self.identity, list(texts), kwargs))
                if self.oom_once:
                    self.oom_once = False
                    raise RuntimeError("MPS backend out of memory")
                width = self.dimension if owner.width is None else owner.width
                return Array([[float(index)] * width for index in range(len(texts))], (len(texts), width))

            def to(self, device: str) -> None:
                self.device = device

        self.Model = Model
        self.mps = load("mps_utils")
        self.mps.ensure_mps_memory_budget = Mock()
        self.mps.resolve_device = lambda _device: "cpu"
        self.mps.flush_mps_cache = Mock()
        modules = patch.dict(sys.modules, {
            "model2vec": types.SimpleNamespace(StaticModel=types.SimpleNamespace(from_pretrained=Model)),
            "sentence_transformers": types.SimpleNamespace(SentenceTransformer=Model),
            "sqlite_vec": types.SimpleNamespace(load=Mock()),
        })
        modules.start()
        self.addCleanup(modules.stop)
        self.ms = self.fixture.ms
        self.ms.np = Numpy()
        if "model_server" in SOURCE_TEXTS:
            tree = ast.parse(SOURCE_TEXTS["model_server"])
            tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
            with patch.dict(sys.modules, {self.ms.__name__: self.ms}):
                exec(compile(tree, "synthetic-server-override", "exec"), self.ms.__dict__)
        self.server = self.ms.ModelServer()
        self.addCleanup(self.server._workers.shutdown, wait=True)
        self.server._write_status_file = lambda: None
        self.server._SUSTAINED_THRESHOLD = 1000
        self.client = load("model_client", definitions=True, namespace={"np": Numpy()})
        self.client._request_with_autostart = lambda request, timeout=None: self.server.handle_request(request)
        self.client.use_model_server = lambda: False
        self.vector = load("vector_search", definitions=True, namespace={
            "np": Numpy(), "threading": threading, "contextmanager": contextmanager,
            "database_operation": lambda function: function,
            "time": time, "math": math, "re": re, "sqlite3": sqlite3,
            "_lock": threading.Lock(), "_prepared_target_slot": threading.Lock(),
            "_model": None, "EMBEDDING_MODEL": "model2vec", "_embedding_dim": 256,
            "_model_generation": 71, "_VEC_DISTANCE_METRIC": "cosine",
        })
        self.vector._TIER_ALIASES = {"base": "qwen3_256", "pro": "qwen3_256", "edge": "model2vec"}
        self.assertTrue(hasattr(self.vector, "prepare_embedding_target"), "Prepared lease API missing")
        self.assertTrue(hasattr(self.client, "PreparedEmbeddingProxy"), "Certified client protocol missing")
        self.assertTrue(hasattr(self.server, "_get_prepared_embed_model"), "Certified server ownership missing")

    def request(self, *, target=None, prepare: bool = True, texts=None, **fields: object) -> dict:
        target = self.target if target is None else target
        request = {"op": "prepare_embed_target_v1" if prepare else "embed_target_v1",
                   "target": target.to_wire(), "batch_size": 8, **fields}
        if not prepare:
            request["texts"] = ["synthetic"] if texts is None else texts
        return self.server.handle_request(request)

    def globals(self) -> tuple:
        return (self.vector.EMBEDDING_MODEL, self.vector._embedding_dim,
                self.vector._model_generation, self.vector._model)

    def test_descriptor_is_frozen_strict_and_aliases_share_effective_identity(self) -> None:
        self.assertEqual(self.target.identity, self.Target.capture("pro").identity)
        self.assertEqual(self.target.tables, ("vec_messages_basepro", "vec_messages_sep_basepro"))
        self.assertEqual(self.Target.from_wire(self.target.to_wire()), self.target)
        with self.assertRaises(FrozenInstanceError):
            self.target.dimension = 4
        for mutation in ({"dimension": True}, {"version": True}, {"tier_group": "x; DROP TABLE x"},
                         {"model_id": "synthetic/other"}, {"extra": 1}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.Target.from_wire({**self.target.to_wire(), **mutation})

    def test_server_certifies_probe_and_every_batch_without_global_mutation(self) -> None:
        before = self.globals()
        response = self.request()
        self.assertTrue(response["ok"])
        self.assertEqual(response["target"], self.target.to_wire())
        self.assertEqual(self.encoded[0][1], [self.target_module.TARGET_PROBE_TEXT])
        result = self.request(prepare=False, texts=["synthetic"] * 19)
        self.assertEqual(result["vectors"].shape, (19, 256))
        self.assertEqual([len(entry[1]) for entry in self.encoded], [1, 8, 8, 3])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.globals(), before)
        self.assertFalse(self.server._inference_lock.locked())

    def test_builtin_alias_reuses_resident_model_without_second_constructor(self) -> None:
        self.request()
        resident = self.server._embed_state.model
        self.request(target=self.Target.capture("pro"))
        self.assertIs(self.server._embed_state.model, resident)
        self.assertEqual(len(self.calls), 1)

    def test_custom_legacy_fallback_is_replaced_and_frozen_args_are_used(self) -> None:
        target = self.Target.capture("custom")
        stale = self.Model("wrong-fallback")
        self.server._embed_state = self.ms._EmbedState(stale, "custom", target.model_id)
        stale_ref = weakref.ref(stale)
        stale = None
        self.build_hook = lambda: self.assertIsNone(stale_ref())
        response = self.request(target=target)
        self.assertTrue(response["ok"])
        identity, kwargs = self.calls[-1]
        self.assertEqual(identity, "synthetic/encoder")
        self.assertEqual(kwargs["truncate_dim"], 384)
        self.assertIs(kwargs["trust_remote_code"], False)

    def test_wrong_actual_width_cannot_certify_or_return_vectors(self) -> None:
        target = self.Target.capture("custom")
        self.width = 128
        for preparing in (True, False):
            with self.subTest(preparing=preparing), self.assertRaises(self.target_module.EmbeddingTargetError):
                self.request(target=target, prepare=preparing)
            self.assertFalse(self.server._inference_lock.locked())
            self.assertFalse(self.server._lock.locked())

    def test_custom_config_mismatch_and_revoked_permission_refuse_even_empty_input(self) -> None:
        target = self.Target.capture("custom")
        for setting in ({"TRUEMEMORY_CUSTOM_EMBED_DIM": "512"}, {"TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD": "0"}):
            with self.subTest(setting=setting), patch.dict("os.environ", setting), self.assertRaises(ValueError):
                self.request(target=target, prepare=False, texts=[])
        self.assertEqual(self.calls, [])

    def test_constructor_direct_call_rechecks_permission_and_deadline(self) -> None:
        target = self.Target.capture("custom")
        with patch.dict("os.environ", {"TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD": "0"}), self.assertRaises(ValueError):
            self.target_module.build_target_model(target, "cpu")
        with self.assertRaises(TimeoutError):
            self.target_module.build_target_model(target, "cpu", check=Mock(side_effect=TimeoutError("expired")))
        self.assertEqual(self.calls, [])

    def test_empty_encode_still_certifies_and_preserves_requested_width(self) -> None:
        proxy = self.client.PreparedEmbeddingProxy(self.target)
        result = proxy.encode([])
        self.assertEqual(result.shape, (0, 256))
        self.assertEqual(len(self.encoded), 1)
        self.width = 2
        with self.assertRaises(self.target_module.EmbeddingTargetError):
            proxy.encode([])

    def test_old_daemon_missing_or_mismatched_receipt_never_falls_back(self) -> None:
        proxy = self.client.PreparedEmbeddingProxy(self.target)
        for response in ({"ok": False, "error": "Unknown op: prepare_embed_target_v1"},
                         {"ok": True}, {"ok": True, "target": self.Target.capture("edge").to_wire()}):
            with self.subTest(response=response):
                request = Mock(return_value=response)
                self.client._request_with_autostart = request
                with self.assertRaises(self.client.ProtocolMismatchError):
                    proxy.prepare()
                self.assertEqual(request.call_count, 1)
                self.assertEqual(request.call_args.args[0]["op"], "prepare_embed_target_v1")

    def test_expired_or_rss_refused_preparation_starts_no_constructor(self) -> None:
        expired = self.request(deadline=time.time() - 1)
        self.assertFalse(expired["ok"])
        self.server._max_rss_bytes = 1
        self.ms.psutil = types.SimpleNamespace(Process=lambda: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=1)))
        refused = self.request()
        self.assertEqual(refused["error_code"], "server_busy")
        self.assertEqual(self.calls, [])
        self.assertFalse(self.server._inference_lock.locked())

    def test_expiry_during_rss_sensor_prevents_strict_constructor(self) -> None:
        clock = [1000.0]
        self.ms.time = types.SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0])
        self.server._max_rss_bytes = 2

        def sample():
            clock[0] = 1002.0
            return types.SimpleNamespace(rss=1)

        self.ms.psutil = types.SimpleNamespace(Process=lambda: types.SimpleNamespace(memory_info=sample))
        result = self.request(deadline=1001.0)
        self.assertFalse(result["ok"])
        self.assertIn("deadline", result["error"])
        self.assertEqual(self.calls, [])
        self.assertFalse(self.server._lock.locked())
        self.assertFalse(self.server._inference_lock.locked())

    def test_prepared_server_recovery_uses_sticky_cpu_and_one_resident_model(self) -> None:
        self.request()
        resident = self.server._embed_state.model
        resident.device = "mps"
        resident.oom_once = True
        result = self.request(prepare=False)
        self.assertTrue(result["ok"])
        self.assertIs(self.server._embed_state.model, resident)
        self.assertEqual(resident.device, "cpu")
        self.assertIn("embed", self.server._sticky_cpu)
        self.assertEqual(len(self.calls), 1)

    def test_local_lease_is_exclusive_and_never_changes_active_globals(self) -> None:
        before = self.globals()
        self.build_hook = lambda: self.assertFalse(self.vector._lock.locked())
        self.encode_hook = lambda: self.assertFalse(self.vector._lock.locked())
        with self.vector.prepare_embedding_target(self.target) as lease:
            self.assertEqual(lease.target, self.target)
            self.assertEqual(lease.encode(["synthetic"]).shape, (1, 256))
            with self.assertRaises(RuntimeError):
                with self.vector.prepare_embedding_target(self.target):
                    self.fail("Second lease admitted")
        lease.close()
        with self.assertRaises(RuntimeError):
            lease.encode(["synthetic"])
        self.assertEqual(self.globals(), before)
        self.assertFalse(self.vector._prepared_target_slot.locked())
        gc.collect()
        self.assertTrue(all(ref() is None for ref in self.refs))

    def test_shared_context_lease_uses_certified_protocol_without_local_slot(self) -> None:
        self.client.use_model_server = lambda: True
        before = self.globals()
        with self.vector.prepare_embedding_target(self.target) as lease:
            self.assertFalse(self.vector._prepared_target_slot.locked())
            self.assertEqual(lease.encode(["synthetic"] * 3, batch_size=2).shape, (3, 256))
            self.assertEqual([len(entry[1]) for entry in self.encoded], [1, 2, 1])
        self.assertIsNone(lease._encoder)
        self.assertIsNotNone(self.server._embed_state)
        self.assertEqual(self.globals(), before)

    def test_matching_local_active_builtin_is_borrowed_without_constructor(self) -> None:
        active = self.Model("Qwen/Qwen3-Embedding-0.6B", truncate_dim=256)
        self.vector._model = active
        self.vector.EMBEDDING_MODEL = "qwen3_256"
        before = self.globals()
        with self.vector.prepare_embedding_target(self.target) as lease:
            lease.encode(["synthetic"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.globals(), before)

    def test_local_constructor_and_probe_baseexception_release_slot(self) -> None:
        for stage in ("construction", "probe"):
            with self.subTest(stage=stage):
                error = KeyboardInterrupt("synthetic cancellation")
                failing = Mock(side_effect=error)
                self.build_hook = failing if stage == "construction" else lambda: None
                self.encode_hook = failing if stage == "probe" else lambda: None
                with self.assertRaises(KeyboardInterrupt) as caught:
                    with self.vector.prepare_embedding_target(self.target):
                        self.fail("Cancelled preparation returned a lease")
                self.assertIs(caught.exception, error)
                self.assertFalse(self.vector._prepared_target_slot.locked())
                self.assertFalse(self.vector._lock.locked())
                error.__traceback__ = None
                failing.side_effect = None
                self.build_hook = self.encode_hook = lambda: None
                gc.collect()
                self.assertTrue(all(ref() is None for ref in self.refs))

    def test_active_state_wait_uses_preparation_deadline(self) -> None:
        lock = Mock()
        lock.acquire.return_value = False
        self.vector._lock = lock
        with self.assertRaises(TimeoutError):
            with self.vector.prepare_embedding_target(self.target, timeout=0.25):
                self.fail("Busy active state admitted")
        self.assertGreater(lock.acquire.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(lock.acquire.call_args.kwargs["timeout"], 0.25)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.vector._prepared_target_slot.locked())

    def test_local_native_owner_and_registry_waits_respect_same_deadline(self) -> None:
        with self.vector.prepare_embedding_target(self.target) as lease:
            model = lease._encoder
            encoded = len(self.encoded)
            with self.mps._own_model(model):
                with self.assertRaises(TimeoutError):
                    lease.encode(["synthetic"], timeout=0.01)
                self.assertEqual(self.mps._model_owners[id(model)].users, 1)
            self.assertEqual(self.mps._model_owners, {})
            with self.mps._device_lock:
                with self.assertRaises(TimeoutError):
                    lease.encode(["synthetic"], timeout=0.01)
            self.assertEqual(self.mps._model_owners, {})
            self.assertEqual(len(self.encoded), encoded)
            self.assertFalse(lease._encode_lock.locked())
            self.assertEqual(lease.encode(["synthetic"]).shape, (1, 256))

    def test_legacy_model_owner_preserves_context_manager_lock_contract(self) -> None:
        entered: list[str] = []

        class ContextOnlyLock:
            def __enter__(self) -> None:
                entered.append("enter")

            def __exit__(self, *_exception: object) -> None:
                entered.append("exit")

        model = object()
        factory = self.mps._ModelOwnership
        with patch.object(self.mps, "_ModelOwnership", lambda: factory(lock=ContextOnlyLock())):
            with self.mps._own_model(model):
                self.assertEqual(entered, ["enter"])
                self.assertEqual(self.mps._model_owners[id(model)].users, 1)
            self.assertEqual(entered, ["enter", "exit"])
            self.assertEqual(self.mps._model_owners, {})

    def test_permission_revoked_during_admission_or_budget_setup_starts_no_constructor(self) -> None:
        target = self.Target.capture("custom")
        def revoke(limit: int, _deadline: object) -> tuple[int, None]:
            os.environ["TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD"] = "0"
            return limit, None

        self.server._request_batch_limit = revoke
        with self.assertRaises(ValueError):
            self.request(target=target)
        os.environ["TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD"] = "1"
        self.mps.ensure_mps_memory_budget.side_effect = lambda _device: os.environ.update(
            TRUEMEMORY_CUSTOM_ALLOW_DOWNLOAD="0",
        )
        with self.assertRaises(ValueError):
            self.target_module.build_target_model(target, "mps")
        self.assertEqual(self.calls, [])

    def test_late_slice_width_mismatch_returns_no_partial_output(self) -> None:
        def vary_width() -> None:
            if self.encoded:
                self.width = 128

        self.encode_hook = vary_width
        with self.assertRaises(self.target_module.EmbeddingTargetError):
            self.request(prepare=False, texts=["synthetic"] * 9)
        self.assertEqual([len(entry[1]) for entry in self.encoded], [8, 1])
        self.assertFalse(self.server._inference_lock.locked())

    def test_preparation_deadline_consumed_by_constructor_prevents_probe(self) -> None:
        clock = [100.0]
        self.vector.time = types.SimpleNamespace(monotonic=lambda: clock[0])
        self.build_hook = lambda: clock.__setitem__(0, 102.0)
        with self.assertRaises(TimeoutError):
            with self.vector.prepare_embedding_target(self.target, timeout=1.0):
                self.fail("Expired construction started a preparation probe")
        self.assertEqual(self.encoded, [])
        self.assertFalse(self.vector._prepared_target_slot.locked())

    def test_close_marks_closed_before_wait_and_holds_slot_until_encode_drains(self) -> None:
        entered, finish, closed = threading.Event(), threading.Event(), threading.Event()
        errors: list[BaseException] = []
        with self.vector.prepare_embedding_target(self.target) as lease:
            def encode_hook() -> None:
                entered.set()
                if not finish.wait(3):
                    raise AssertionError("Synthetic encode barrier timed out")

            self.encode_hook = encode_hook

            def encode() -> None:
                try:
                    lease.encode(["synthetic"])
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=encode)
            thread.start()
            self.assertTrue(entered.wait(3))
            original_close = lease._finish_close
            lease._finish_close = lambda: (original_close(), closed.set())
            closer = threading.Thread(target=lease.close)
            closer.start()
            deadline = time.monotonic() + 3
            while not lease._closed and time.monotonic() < deadline:
                closed.wait(0.001)
            try:
                self.assertTrue(lease._closed)
                self.assertFalse(closed.is_set())
                with self.assertRaises(RuntimeError):
                    lease.encode(["queued"])
                with self.assertRaises(RuntimeError):
                    with self.vector.prepare_embedding_target(self.target):
                        self.fail("Closing target released residency early")
            finally:
                finish.set()
                thread.join(3)
                closer.join(3)
            self.assertFalse(thread.is_alive())
            self.assertFalse(closer.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(closed.is_set())
            self.assertIsNone(lease._encoder)
            self.assertFalse(self.vector._prepared_target_slot.locked())

    def test_interrupted_close_can_be_retried_and_does_not_reopen_lease(self) -> None:
        with self.vector.prepare_embedding_target(self.target) as lease:
            actual = lease._encode_lock
            failing = Mock()
            failing.__enter__ = Mock(side_effect=KeyboardInterrupt("synthetic close cancellation"))
            failing.__exit__ = Mock(return_value=False)
            failing.acquire.return_value = False
            lease._encode_lock = failing
            with self.assertRaises(KeyboardInterrupt):
                lease.close()
            self.assertTrue(self.vector._prepared_target_slot.locked())
            with self.assertRaises(RuntimeError):
                lease.encode(["synthetic"])
            lease._encode_lock = actual
            lease.close()
            self.assertFalse(self.vector._prepared_target_slot.locked())
            self.assertIsNone(lease._encoder)

    def test_named_pair_is_idempotent_without_active_metadata_or_registry(self) -> None:
        conn = VecConnection()
        self.addCleanup(conn.close)
        before = self.globals()
        self.assertEqual(self.vector.init_prepared_target_tables(conn, self.target), self.target.tables)
        self.vector.init_prepared_target_tables(conn, self.target)
        names = {row[0] for row in conn.db.execute("SELECT name FROM sqlite_master")}
        self.assertEqual(names, set(self.target.tables))
        self.assertFalse(conn.in_transaction)
        self.assertEqual(self.globals(), before)
        self.assertEqual(conn.extensions, [True, False, True, False])

    def test_target_ddl_preserves_borrowed_transaction_and_rolls_back_failed_pair(self) -> None:
        for borrowed in (False, True):
            with self.subTest(borrowed=borrowed):
                conn = VecConnection()
                self.addCleanup(conn.close)
                conn.db.execute("CREATE TABLE unrelated(value TEXT)")
                if borrowed:
                    conn.db.execute("INSERT INTO unrelated VALUES('synthetic')")
                conn.fail_table = self.target.tables[1]
                with self.assertRaises(sqlite3.OperationalError):
                    self.vector.init_prepared_target_tables(conn, self.target)
                self.assertEqual(conn.in_transaction, borrowed)
                self.assertEqual(conn.db.execute("SELECT count(*) FROM unrelated").fetchone()[0], int(borrowed))
                self.assertIsNone(conn.db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.target.tables[0],)).fetchone())
                conn.fail_table = None
                self.vector.init_prepared_target_tables(conn, self.target)
                self.assertEqual(conn.in_transaction, borrowed)
                if borrowed:
                    conn.rollback()
                    self.assertIsNone(conn.db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.target.tables[0],)).fetchone())

    def test_target_schema_rejects_wrong_width_metric_ordinary_table_and_temp_shadow(self) -> None:
        for kind in ("width", "metric", "ordinary", "temp", "temp_upper", "temp_mixed"):
            with self.subTest(kind=kind):
                conn = VecConnection()
                self.addCleanup(conn.close)
                name = self.target.tables[0]
                if kind == "ordinary":
                    conn.db.execute(f"CREATE TABLE {name}(embedding BLOB)")
                else:
                    self.vector.init_prepared_target_tables(conn, self.target)
                    if kind == "width":
                        conn.schemas[name] = conn.schemas[name].replace("float[256]", "float[384]")
                    elif kind == "metric":
                        conn.schemas[name] = conn.schemas[name].replace("cosine", "l2")
                    else:
                        shadow = name.upper() if kind == "temp_upper" else name
                        if kind == "temp_mixed":
                            shadow = self.target.tables[1].title()
                        conn.db.execute(f"CREATE TEMP TABLE {shadow}(embedding BLOB)")
                with self.assertRaises(ValueError):
                    self.vector.init_prepared_target_tables(conn, self.target)
                self.assertFalse(conn.in_transaction)


@unittest.skipUnless(os.environ.get("TRUEMEMORY_PREPARED_NATIVE_SQLITE_TEST") == "1",
                     "Actual sqlite-vec is an explicit isolated native gate")
class TestNativePreparedTables(unittest.TestCase):
    def test_real_vec0_pair_owned_and_borrowed_transactions(self) -> None:
        # Run only in the isolated native test container, never in local checks.
        import sqlite_vec

        package = types.ModuleType("truememory")
        package.__path__ = []
        with patch.dict(sys.modules, {"truememory": package, "sqlite_vec": sqlite_vec}):
            target_module = load("embedding_target")
            load("tier_config")
            vector = load("vector_search", definitions=True, namespace={
                "np": Numpy(), "threading": threading, "contextmanager": contextmanager,
                "database_operation": lambda function: function,
                "time": time, "math": math, "re": re, "sqlite3": sqlite3,
                "_VEC_DISTANCE_METRIC": "cosine",
            })
            target = target_module.EmbeddingTarget.capture("base")
            for borrowed in (False, True):
                with self.subTest(borrowed=borrowed):
                    conn = sqlite3.connect(":memory:")
                    try:
                        conn.execute("CREATE TABLE unrelated(value TEXT)")
                        if borrowed:
                            conn.execute("INSERT INTO unrelated VALUES('synthetic')")
                        self.assertEqual(vector.init_prepared_target_tables(conn, target), target.tables)
                        self.assertEqual(vector.init_prepared_target_tables(conn, target), target.tables)
                        self.assertEqual(conn.in_transaction, borrowed)
                        for name in target.tables:
                            self.assertEqual(conn.execute(f'SELECT count(*) FROM main."{name}"').fetchone()[0], 0)
                        if borrowed:
                            conn.rollback()
                            self.assertIsNone(conn.execute("SELECT 1 FROM main.sqlite_master WHERE name=?",
                                                           (target.tables[0],)).fetchone())
                        else:
                            conn.execute(f"CREATE TEMP TABLE {target.tables[0]}(value TEXT)")
                            with self.assertRaises(ValueError):
                                vector.init_prepared_target_tables(conn, target)
                    finally:
                        conn.close()
            for mismatched in target.tables:
                with self.subTest(mismatched=mismatched):
                    conn = sqlite3.connect(":memory:")
                    try:
                        conn.enable_load_extension(True)
                        sqlite_vec.load(conn)
                        conn.enable_load_extension(False)
                        conn.execute(
                            f'CREATE VIRTUAL TABLE main."{mismatched}" '
                            'USING vec0(embedding float[384] distance_metric=cosine)'
                        )
                        conn.commit()
                        with self.assertRaises(ValueError):
                            vector.init_prepared_target_tables(conn, target)
                        self.assertFalse(conn.in_transaction)
                        for name in target.tables:
                            row = conn.execute("SELECT sql FROM main.sqlite_master WHERE name=?", (name,)).fetchone()
                            if name == mismatched:
                                self.assertIn("float[384]", row[0])
                            else:
                                self.assertIsNone(row)
                    finally:
                        conn.close()


if __name__ == "__main__":
    unittest.main()
