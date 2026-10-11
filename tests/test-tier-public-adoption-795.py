"""Public adoption orchestration with production AST, stubs and in-memory SQL."""

from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import dataclasses
import io
import json
import logging
import re
import sqlite3
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class Target:
    tier: str
    model_id: str
    dimension: int
    tier_group: str

    @classmethod
    def capture(cls, tier: str) -> Target:
        if tier not in {"edge", "base", "pro"}:
            raise ValueError("unsupported synthetic target")
        return cls(
            tier,
            "model2vec" if tier == "edge" else "qwen3_256",
            256,
            "edge" if tier == "edge" else "basepro",
        )

    @property
    def identity(self) -> tuple:
        return self.model_id, self.dimension, self.tier_group

    @property
    def tables(self) -> tuple[str, str]:
        return f"vec_messages_{self.tier_group}", f"vec_messages_sep_{self.tier_group}"


class ActivationIntent(types.SimpleNamespace):
    pass


class PolicyIntent(types.SimpleNamespace):
    pass


class LegacyTierPolicy(types.SimpleNamespace):
    pass


class FakeConnection:
    def __init__(self) -> None:
        self.closed = False
        self.in_transaction = False
        self.executed: list[str] = []

    def close(self) -> None:
        self.closed = True

    def execute(self, sql: str, parameters: tuple = ()) -> object:
        self.executed.append(sql)
        raise AssertionError("Unexpected SQL on stub: " + sql)


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection, *, write: bool):
    if conn.in_transaction:
        raise RuntimeError("borrowed transaction")
    conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
    try:
        yield
        conn.commit() if write else conn.rollback()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise


class Harness:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.config = {"tier": "edge", "independent": "keep"}
        self.guard = types.SimpleNamespace(tier="edge", generation="old")
        self.state = types.SimpleNamespace(
            selection=None, legacy_policy=None, intent=None
        )
        self.job = types.SimpleNamespace(
            job_id="a" * 32,
            target=Target.capture("base"),
            database_path=Path("/synthetic/store.db").absolute(),
            tracker="canonical-v1",
            source_epoch="epoch",
            source_schema_signature="b" * 64,
        )
        self.intent = ActivationIntent(
            job_id=self.job.job_id,
            target=self.job.target,
            tracker=self.job.tracker,
            source_epoch=self.job.source_epoch,
            source_schema_signature=self.job.source_schema_signature,
            expected_config=self.guard,
            status_id=17,
            state="staged",
        )
        self.marker = None
        self.result = LegacyTierPolicy(
            target=Target.capture("pro"), generation="new", config_acknowledged=False
        )
        self.modules = {
            "truememory.embedding_target": types.SimpleNamespace(
                EmbeddingTarget=Target
            ),
            "truememory.tier_config": types.SimpleNamespace(
                get_tier_config=lambda tier: {"reranker": "synthetic/reranker"}
            ),
            "truememory.maintenance": types.SimpleNamespace(
                maintenance_owner=self.owner
            ),
            "truememory.rebuild_source": types.SimpleNamespace(
                ensure_rebuild_tracking=lambda conn: self.events.append("tracking")
            ),
            "truememory.tier_switch.cache": types.SimpleNamespace(
                preflight_ram_check=lambda group: (True, ""),
                get_transition_action=lambda old, new: (
                    "config_only" if {old, new} == {"base", "pro"} else "delta_or_full"
                ),
            ),
            "truememory.tier_switch.throttler": types.SimpleNamespace(
                DynamicThrottler=lambda **kwargs: object()
            ),
            "truememory.tier_switch.worker": types.SimpleNamespace(
                RebuildWorker=Mock()
            ),
            "truememory.tier_switch.serving": types.SimpleNamespace(
                exclusive_activation=self.exclusive
            ),
            "truememory.tier_switch.activation": types.SimpleNamespace(
                ActivationIntent=ActivationIntent,
                LegacyTierPolicy=LegacyTierPolicy,
                read_activation_state=lambda conn: self.state,
                _state=lambda conn: self.state,
                commit_config_only_transition=self.commit_policy,
                stage_activation_intent=self.stage,
                acknowledge_config=lambda conn, **kwargs: self.events.append("ack"),
                acknowledge_policy_config=lambda conn, **kwargs: self.events.append(
                    "ack"
                ),
                commit_certified_selection=lambda *args: self.result,
                _transaction=transaction,
                _normalized=lambda sql: re.sub(
                    r"\s*([(),])\s*", r"\1", " ".join(sql.strip().rstrip(";").split())
                ).lower(),
                _schema_sql=lambda conn, name: conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name=?", (name,)
                ).fetchone()[0],
                _guard_pair=lambda conn, target, **kwargs: self.events.append(
                    ("pair", kwargs)
                ),
            ),
            "truememory.tier_switch.job": types.SimpleNamespace(
                read_selected_job_marker=lambda conn: self.marker,
                select_tier_job=self.select,
                check_selected_tier_job=lambda *args: self.events.append("check"),
                end_selected_tier_job=lambda *args, **kwargs: self.events.append(
                    ("end", kwargs)
                ),
                _cancel_policy_job_in_writer=lambda *args, **kwargs: self.events.append(
                    ("cancel", kwargs)
                ),
            ),
            "truememory.tier_switch.projection": types.SimpleNamespace(
                capture_config_guard=lambda **kwargs: self.guard,
                mirror_activation=self.mirror,
                patch_config_fields=self.patch_config,
                CONFIG_WRITE_LOCK=threading.RLock(),
                ConfigFileLock=lambda *args, **kwargs: contextlib.nullcontext(),
                _read_config=lambda path: self.config.copy(),
                _write_config=self.write_config,
            ),
            "truememory.tier_switch.runtime": types.SimpleNamespace(
                apply_frozen_selection=lambda result: self.events.append(
                    "apply_frozen"
                ),
                apply_runtime_policy=lambda result: self.events.append("apply_policy"),
                serving_operation=Mock(
                    side_effect=AssertionError("native readiness forbidden")
                ),
            ),
            "truememory.tier_switch.source": types.SimpleNamespace(
                read_tier_progress=lambda *args, **kwargs: None
            ),
            "truememory.vector_search": types.SimpleNamespace(
                init_prepared_target_tables=lambda *args: self.events.append("tables")
            ),
            "truememory.storage": types.SimpleNamespace(
                DEFAULT_BUSY_TIMEOUT_MS=1,
                _validate_db_path=lambda p: p,
                create_db=Mock(side_effect=AssertionError("initialization forbidden")),
            ),
        }
        self.modules["truememory"] = types.SimpleNamespace(
            vector_search=self.modules["truememory.vector_search"]
        )
        vector = self.modules["truememory.vector_search"]
        vector.resolve_tier = lambda: "edge"
        vector.EMBEDDING_MODEL = "model2vec"
        vector._embedding_dim = 256
        self.modules["truememory"].reranker = types.SimpleNamespace(
            get_current_reranker_name=lambda: "synthetic/reranker"
        )
        vector._model = None
        self.modules["truememory"].reranker._model = None

        def apply_embedding_policy(target, **kwargs):
            self.events.append("lazy_embedding")
            vector.EMBEDDING_MODEL = target.model_id
            vector._embedding_dim = target.dimension
            vector.resolve_tier = lambda: target.tier

        vector.apply_embedding_policy = apply_embedding_policy
        self.modules["truememory"].reranker.apply_reranker_policy = lambda *args: (
            self.events.append("lazy_reranker")
        )

        def set_embedding_model(tier):
            self.events.append("lazy_embedding")
            applied = Target.capture(tier)
            vector.EMBEDDING_MODEL = applied.model_id
            vector._embedding_dim = applied.dimension
            vector.resolve_tier = lambda: tier

        vector.set_embedding_model = set_embedding_model
        self.modules["truememory"].reranker.set_active_tier = lambda tier: self.events.append("lazy_reranker")
        self.manager_module = self.load("truememory/tier_switch/manager.py")
        self.manager_module.os = types.SimpleNamespace(environ={}, name="posix")
        self.modules["truememory.tier_switch.manager"] = self.manager_module
        self.manager = self.manager_module.RebuildManager()
        self.conn = FakeConnection()
        self.manager_module._open_db = lambda *args, **kwargs: self.conn
        self.manager_module._load_tier_extension = lambda conn: self.events.append(
            "extension"
        )
        self.manager_module._detect_device = lambda: "cpu"

    def importer(self, name, globals=None, locals=None, fromlist=(), level=0):
        if name in self.modules:
            return self.modules[name]
        if name.startswith("truememory") or name in {
            "torch",
            "numpy",
            "sqlite_vec",
            "sentence_transformers",
            "model2vec",
        }:
            raise AssertionError("Unstubbed/native import: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load(
        self, relative: str, names: set[str] | None = None, extra: dict | None = None
    ):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if names is not None:
            tree.body = [
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names
            ]
            for node in tree.body:
                node.decorator_list = []
        module = types.ModuleType("synthetic_public_adoption")
        module.__dict__.update(
            {
                "__builtins__": dict(vars(builtins), __import__=self.importer),
                "Path": Path,
                "sqlite3": sqlite3,
                "sys": sys,
                "threading": threading,
            }
        )
        if extra:
            module.__dict__.update(extra)
        exec(compile(tree, str(path), "exec"), module.__dict__)
        return module

    @contextlib.contextmanager
    def owner(self, path):
        self.events.append("owner_enter")
        try:
            yield
        finally:
            self.events.append("owner_exit")

    @contextlib.contextmanager
    def exclusive(self, **kwargs):
        self.events.append("exclusive_enter")
        try:
            yield
        finally:
            self.events.append("exclusive_exit")

    def write_config(self, path, config):
        self.config = config.copy()
        self.events.append("config")

    def patch_config(self, fields, **kwargs):
        self.config.update(fields)
        return self.config.copy()

    def mirror(self, result, **kwargs):
        self.events.append("mirror")

    def commit_policy(self, conn, **kwargs):
        self.events.append("policy_commit")
        return self.result

    def select(self, conn, target, **kwargs):
        self.events.append("select")
        return self.job

    def stage(self, conn, job, **kwargs):
        self.events.append("stage")
        return self.intent

    def prepare(self, old="edge", requested="base"):
        self.guard.tier = old
        self.manager._serving_identity = lambda *args: (
            Target.capture(old),
            Target.capture(old).tables,
        )
        self.manager._check_inactive_target = lambda *args: self.events.append(
            "inactive_check"
        )
        self.manager._guard_status = lambda *args: self.events.append("status_guard")
        self.manager._create_status_row = lambda *args, **kwargs: (
            self.events.append(("status", kwargs)) or 17
        )
        return self.manager._prepare_job(self.conn, requested, False)


class MemoryPath:
    def __truediv__(self, name: str) -> MemoryPath:
        return self

    @property
    def parent(self) -> MemoryPath:
        return self

    def mkdir(self, **kwargs) -> None:
        pass

    def write_text(self, content: str, **kwargs) -> None:
        pass


def configure_harness(h: Harness, manager: object):
    h.modules["truememory.tier_switch.manager"].RebuildManager = types.SimpleNamespace(
        get_instance=lambda: manager
    )
    return h.load(
        "truememory/mcp_server.py",
        {"truememory_configure"},
        {
            "json": json,
            "os": types.SimpleNamespace(environ={}),
            "log": logging.getLogger("synthetic_configure"),
            "_memory": object(),
            "_memory_lock": threading.Lock(),
            "_config_cache": None,
            "_load_config": lambda: h.config.copy(),
            "_CONFIG_PATH": Path("/synthetic/config.json"),
            "_CONFIG_LOCK_PATH": Path("/synthetic/config.json.lock"),
            "_DB_PATH": "/synthetic/store.db",
            "_VALID_INTENSITIES": ("standard", "enhanced", "max"),
            "_config_file_lock": lambda **kwargs: contextlib.nullcontext(),
            "_config_write_lock": threading.RLock(),
        },
    )


def cancel_cli_harness(h: Harness, manager: object):
    factory = Mock(return_value=manager)
    h.modules["truememory.tier_switch.manager"].RebuildManager = factory
    h.modules["truememory"].__version__ = "synthetic"
    module = h.load(
        "truememory/ingest/cli.py",
        {
            "main",
            "_positive_rebuild_status_id",
            "_rebuild_job_id",
            "_run_cancel_rebuild",
        },
        {"argparse": argparse, "re": re, "os": types.SimpleNamespace(environ={})},
    )
    return module, factory


def status_connection() -> sqlite3.Connection:
    tree = ast.parse((ROOT / "truememory/tier_switch/manager.py").read_text(encoding="utf-8"))
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_guard_status"
    )
    sql = next(
        n.value
        for n in ast.walk(method)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value.startswith("CREATE TABLE rebuild_status")
    )
    conn = sqlite3.connect(":memory:")
    conn.execute(sql)
    return conn


class PublicAdoptionTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def test_force_refuses_before_config_or_sql(self):
        h = self.h
        with self.assertRaises(h.manager_module.TierSwitchUnsupportedError):
            h.manager._prepare_job(h.conn, "base", True)
        self.assertEqual(h.events, [])
        self.assertEqual(h.conn.executed, [])

    def test_shared_prepare_orders_status_job_intent_before_source(self):
        h = self.h
        self.assertEqual(h.prepare(), (h.job, h.intent, 17))
        self.assertLess(h.events.index("inactive_check"), h.events.index("tracking"))
        self.assertLess(h.events.index("tables"), h.events.index("select"))
        self.assertLess(h.events.index("select"), h.events.index("stage"))
        self.assertEqual(h.conn.executed, [])

    def test_live_marker_refuses_without_writes(self):
        h = self.h
        h.marker = types.SimpleNamespace(state="selected", job_id="old")
        with self.assertRaises(h.manager_module.TierSwitchUnsupportedError):
            h.prepare()
        self.assertNotIn("tracking", h.events)
        self.assertNotIn("select", h.events)

    def test_same_space_policy_projects_without_source_or_native(self):
        h = self.h
        h.state.intent = PolicyIntent(expected_config=h.guard, intent_id="policy")
        self.assertIsNone(h.prepare("base", "pro"))
        self.assertEqual(
            [
                x
                for x in h.events
                if isinstance(x, str) and x not in {"exclusive_enter", "exclusive_exit"}
            ],
            ["policy_commit", "mirror", "apply_policy", "ack"],
        )
        self.assertEqual(h.conn.executed, [])
        self.assertEqual(h.manager._last_action, "config_only")

    def test_pending_same_tier_retries_projection(self):
        h = self.h
        h.state.legacy_policy = h.result
        h.state.intent = PolicyIntent(expected_config=h.guard, intent_id="policy")
        self.assertIsNone(h.prepare("pro", "pro"))
        self.assertIn("apply_policy", h.events)
        self.assertEqual(h.manager._last_outcome, "complete")

    def test_pending_other_tier_refuses(self):
        h = self.h
        h.state.legacy_policy = h.result
        h.state.intent = PolicyIntent(expected_config=h.guard, intent_id="policy")
        with self.assertRaises(h.manager_module.TierSwitchUnsupportedError):
            h.prepare("pro", "edge")
        self.assertNotIn("mirror", h.events)

    def test_acknowledged_same_tier_still_projects_runtime(self):
        h = self.h
        h.result.config_acknowledged = True
        h.state.legacy_policy = h.result
        self.assertIsNone(h.prepare("pro", "pro"))
        self.assertIn("apply_policy", h.events)

    def test_runtime_failure_after_mirror_remains_pending(self):
        h = self.h
        h.state.legacy_policy = h.result
        h.state.intent = PolicyIntent(expected_config=h.guard, intent_id="policy")
        h.modules["truememory.tier_switch.runtime"].apply_runtime_policy = Mock(
            side_effect=RuntimeError("runtime fault")
        )
        with self.assertRaisesRegex(RuntimeError, "runtime fault"):
            h.prepare("pro", "pro")
        self.assertEqual(h.manager._last_outcome, "activation_pending")
        self.assertNotIn("ack", h.events)

    def test_config_failure_never_applies_or_acknowledges(self):
        h = self.h
        h.state.intent = PolicyIntent(expected_config=h.guard, intent_id="policy")
        h.modules["truememory.tier_switch.projection"].mirror_activation = Mock(
            side_effect=RuntimeError("config fault")
        )
        with self.assertRaisesRegex(RuntimeError, "config fault"):
            h.prepare("base", "pro")
        self.assertEqual(h.manager._last_outcome, "activation_pending")
        self.assertNotIn("apply_policy", h.events)
        self.assertNotIn("ack", h.events)

    def test_commit_ambiguity_is_pending_and_issues_no_followup_sql(self):
        h = self.h
        h.modules[
            "truememory.tier_switch.activation"
        ].commit_config_only_transition = Mock(
            side_effect=sqlite3.OperationalError("commit")
        )
        with self.assertRaises(sqlite3.Error):
            h.prepare("base", "pro")
        self.assertEqual(h.manager._last_outcome, "activation_pending")
        self.assertNotIn("mirror", h.events)
        self.assertEqual(h.conn.executed, [])

    def test_thread_failure_discards_connection_without_terminal_retry(self):
        h = self.h
        h.manager._bootstrap_empty_configuration = lambda *args: False
        h.manager._prepare_job = lambda *args, **kwargs: (h.job, h.intent, 17)
        thread = Mock()
        thread.start.side_effect = RuntimeError("thread start")
        thread.is_alive.return_value = False
        with patch.object(h.manager_module.threading, "Thread", return_value=thread):
            with self.assertRaisesRegex(RuntimeError, "thread start"):
                h.manager.start_rebuild(
                    "base",
                    backup_path=Path("/synthetic/backup"),
                    db_path=h.job.database_path,
                )
        self.assertTrue(h.conn.closed)
        self.assertFalse(h.manager._claimed)
        self.assertIsNone(h.manager._active_thread)
        self.assertEqual(h.manager._last_outcome, "interrupted")
        self.assertEqual(h.conn.executed, [])

    def test_background_handoff_contains_only_immutable_identity(self):
        h = self.h
        h.manager._bootstrap_empty_configuration = lambda *args: False
        h.manager._prepare_job = lambda *args, **kwargs: (h.job, h.intent, 17)
        thread = Mock()
        thread.is_alive.return_value = False
        with patch.object(
            h.manager_module.threading, "Thread", return_value=thread
        ) as factory:
            self.assertEqual(
                h.manager.start_rebuild("base", db_path=h.job.database_path), 17
            )
        self.assertTrue(h.conn.closed)
        self.assertEqual(factory.call_args.kwargs["args"], (h.job, h.intent, 17))
        self.assertNotIn(h.conn, factory.call_args.kwargs["args"])

    def test_status_insert_fault_discards_connection(self):
        h = self.h
        h.manager._bootstrap_empty_configuration = lambda *args: False
        h.manager._prepare_job = Mock(
            side_effect=sqlite3.OperationalError("status insert")
        )
        with self.assertRaises(sqlite3.Error):
            h.manager.start_rebuild("base", db_path=h.job.database_path)
        self.assertTrue(h.conn.closed)
        self.assertFalse(h.manager._claimed)
        self.assertEqual(h.conn.executed, [])

    def test_new_store_only_persists_onboarding_config(self):
        h = self.h
        with patch.object(Path, "stat", side_effect=FileNotFoundError):
            self.assertTrue(
                h.manager._bootstrap_empty_configuration(
                    h.job.database_path, "pro", False
                )
            )
        self.assertEqual(h.config, {"tier": "pro", "independent": "keep"})
        self.assertNotIn("policy_commit", h.events)
        self.assertNotIn("apply_policy", h.events)
        self.assertNotIn("tier_activation_generation", h.config)

    def test_tagged_config_cannot_bootstrap_missing_store(self):
        h = self.h
        h.config["tier_activation_generation"] = "old"
        with (
            patch.object(Path, "stat", side_effect=FileNotFoundError),
            self.assertRaises(h.manager_module.TierSwitchUnsupportedError),
        ):
            h.manager._bootstrap_empty_configuration(h.job.database_path, "pro", False)
        self.assertEqual(h.config["tier"], "edge")

    def test_nonempty_schema_is_not_bootstrapped(self):
        h = self.h
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIEW present AS SELECT 1")
        with (
            patch.object(Path, "stat", return_value=object()),
            patch.object(h.manager_module.sqlite3, "connect", return_value=conn),
        ):
            self.assertFalse(
                h.manager._bootstrap_empty_configuration(
                    h.job.database_path, "pro", False
                )
            )
        self.assertEqual(h.config["tier"], "edge")

    def test_uninitialized_empty_schema_bootstraps_without_database_writes(self):
        h = self.h
        conn = sqlite3.connect(":memory:")
        queries = []
        conn.set_trace_callback(queries.append)
        with (
            patch.object(Path, "stat", return_value=object()),
            patch.object(h.manager_module.sqlite3, "connect", return_value=conn),
        ):
            self.assertTrue(
                h.manager._bootstrap_empty_configuration(
                    h.job.database_path, "pro", False
                )
            )
        self.assertEqual(len(queries), 1)
        self.assertTrue(queries[0].startswith("SELECT"))

    def test_existing_unsupported_database_refuses_before_pragmas(self):
        h = self.h
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE metadata(wrong BLOB)")
        queries = []
        conn.set_trace_callback(queries.append)
        h.modules["truememory.tier_switch.activation"].read_activation_state = Mock(
            side_effect=RuntimeError("unsupported metadata")
        )
        opener = h.load(
            "truememory/tier_switch/manager.py",
            {"_open_db"},
            {"TierSwitchUnsupportedError": RuntimeError},
        )
        with (
            patch.object(opener.sqlite3, "connect", return_value=conn),
            self.assertRaisesRegex(RuntimeError, "unsupported metadata"),
        ):
            opener._open_db(h.job.database_path, initialize=True)
        self.assertEqual(len(queries), 1)
        self.assertTrue(queries[0].startswith("SELECT"))
        h.modules["truememory.storage"].create_db.assert_not_called()

    def test_existing_opener_does_not_create_missing_database(self):
        h = self.h
        opener = h.load(
            "truememory/tier_switch/manager.py",
            {"_open_db"},
            {"TierSwitchUnsupportedError": RuntimeError},
        )
        with patch.object(
            opener.sqlite3, "connect", side_effect=sqlite3.OperationalError("missing")
        ) as connect:
            with self.assertRaises(sqlite3.Error):
                opener._open_db(h.job.database_path)
        self.assertIn("?mode=rw", connect.call_args.args[0])

    def test_cli_config_only_never_probes(self):
        h = self.h
        h.modules["truememory.tier_switch.manager"].RebuildManager = lambda: (
            types.SimpleNamespace(
                run_rebuild_sync=lambda tier: True, _last_action="config_only"
            )
        )
        cli = h.load("truememory/ingest/cli.py", {"_setup_model_readiness"})
        self.assertEqual(cli._setup_model_readiness("pro", "base"), [])
        h.modules[
            "truememory.tier_switch.runtime"
        ].serving_operation.assert_not_called()

    def test_cli_same_tier_invokes_authoritative_manager(self):
        h = self.h
        run = Mock(side_effect=RuntimeError("pending"))
        h.modules["truememory.tier_switch.manager"].RebuildManager = lambda: (
            types.SimpleNamespace(run_rebuild_sync=run)
        )
        cli = h.load("truememory/ingest/cli.py", {"_setup_model_readiness"})
        self.assertEqual(cli._setup_model_readiness("pro", "pro"), ["tier activation"])
        run.assert_called_once_with("pro")

    def test_both_completion_flags_required(self):
        h = self.h
        h.state.intent = h.intent
        plan = types.SimpleNamespace(
            action="resume", manifest=types.SimpleNamespace(consumed=1, total=1)
        )
        h.modules["truememory.tier_switch.source"].plan_tier_source = (
            lambda *args, **kwargs: plan
        )
        h.modules["truememory.tier_switch.source"].initialize_tier_source = Mock(
            side_effect=AssertionError("no init")
        )
        worker = Mock()
        worker.run_source.return_value = types.SimpleNamespace(
            status="complete",
            captured_range_complete=False,
            current_source_complete=True,
            plan=plan,
        )
        h.manager_module.RebuildWorker = lambda *args: worker
        h.manager._terminal_status = lambda *args: h.events.append("terminal")
        h.manager._finalize_rebuild = Mock(
            side_effect=AssertionError("cannot finalize")
        )
        self.assertFalse(h.manager._run_job(h.conn, h.job, h.intent, 17))
        h.manager._finalize_rebuild.assert_not_called()
        self.assertEqual(h.manager._last_outcome, "failed")

    def test_source_conflict_does_not_finalize_or_retry(self):
        h = self.h
        h.state.intent = h.intent
        planner = Mock(side_effect=RuntimeError("source changed"))
        h.modules["truememory.tier_switch.source"].plan_tier_source = planner
        h.modules["truememory.tier_switch.source"].initialize_tier_source = Mock()
        h.manager._finalize_rebuild = Mock()
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            h.manager._run_job(h.conn, h.job, h.intent, 17)
        self.assertEqual(planner.call_count, 1)
        h.manager._finalize_rebuild.assert_not_called()
        self.assertEqual(h.conn.executed, [])

    def test_local_cancel_rejects_wrong_status(self):
        h = self.h
        h.manager._active_worker = Mock()
        h.manager._active_status_id = 17
        with self.assertRaises(h.manager_module.TierSwitchUnsupportedError):
            h.manager.cancel(99)
        h.manager._active_worker.cancel.assert_not_called()
        self.assertFalse(h.manager._cancel_requested.is_set())

    def test_interrupted_cancel_requires_exact_job(self):
        h = self.h
        h.marker = types.SimpleNamespace(**vars(h.job), state="selected")
        h.state.intent = h.intent
        with self.assertRaises(h.manager_module.TierSwitchUnsupportedError):
            h.manager.cancel(17, expected_job_id="wrong")
        self.assertTrue(h.conn.closed)
        self.assertFalse(
            any(isinstance(x, tuple) and x[0] == "cancel" for x in h.events)
        )

    def test_interrupted_cancel_cas_uses_exact_marker(self):
        h = self.h
        h.marker = types.SimpleNamespace(**vars(h.job), state="selected")
        h.state.intent = h.intent
        conn = sqlite3.connect(":memory:")
        h.manager_module._open_db = lambda *args: conn
        result = h.manager.cancel(17, expected_job_id=h.job.job_id)
        self.assertEqual(result, {"status": "cancelled", "job_id": h.job.job_id})
        writes = [x for x in h.events if isinstance(x, tuple) and x[0] == "cancel"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][1]["job_id"], h.job.job_id)

    def test_status_backup_path_round_trip_real_sql(self):
        h = self.h
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        tree = ast.parse((ROOT / "truememory/tier_switch/manager.py").read_text(encoding="utf-8"))
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_guard_status"
        )
        sql = next(
            n.value
            for n in ast.walk(method)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.startswith("CREATE TABLE rebuild_status")
        )
        conn.execute(sql)
        sid = h.manager._create_status_row(
            conn,
            "basepro",
            "base",
            "streamed",
            0,
            backup_path=Path("/synthetic/backup"),
        )
        self.assertEqual(
            conn.execute(
                "SELECT backup_path FROM rebuild_status WHERE id=?", (sid,)
            ).fetchone(),
            (str(Path("/synthetic/backup")),),
        )
        self.assertFalse(conn.in_transaction)

    def test_status_trigger_refused_before_insert(self):
        h = self.h
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        tree = ast.parse((ROOT / "truememory/tier_switch/manager.py").read_text(encoding="utf-8"))
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_guard_status"
        )
        sql = next(
            n.value
            for n in ast.walk(method)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.startswith("CREATE TABLE rebuild_status")
        )
        conn.execute(sql)
        conn.execute(
            "CREATE TRIGGER side_effect AFTER INSERT ON rebuild_status BEGIN SELECT 1; END"
        )
        with self.assertRaises(h.manager_module.TierSwitchUnsupportedError):
            h.manager._create_status_row(conn, "basepro", "base", "streamed", 0)
        self.assertEqual(
            conn.execute("SELECT count(*) FROM rebuild_status").fetchone(), (0,)
        )

    def test_nonempty_inactive_pair_requires_receipt(self):
        h = self.h
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        for name in h.job.target.tables:
            conn.execute(f'CREATE TABLE "{name}"(embedding BLOB)')
        conn.execute("INSERT INTO vec_messages_basepro VALUES (x'01')")
        conn.commit()
        before = conn.total_changes
        with self.assertRaises(h.manager_module.TierSwitchUnsupportedError):
            h.manager._check_inactive_target(conn, h.job.target)
        self.assertEqual(conn.total_changes, before)

    def test_configure_queued_clears_memory_and_reports_served_tier(self):
        h = self.h
        manager = types.SimpleNamespace(
            start_rebuild=Mock(return_value=42), _last_outcome="preparing"
        )
        module = configure_harness(h, manager)
        with patch.object(Path, "home", return_value=MemoryPath()):
            result = json.loads(module.truememory_configure("base"))
        self.assertIsNone(module._memory)
        self.assertEqual(result["tier"], "base")
        self.assertEqual(result["served_tier"], "edge")
        self.assertEqual(result["status"], "building")
        self.assertEqual(result["status_id"], 42)
        self.assertEqual(h.config["tier"], "edge")

    def test_configure_failure_clears_memory_and_keeps_served_config(self):
        h = self.h
        manager = types.SimpleNamespace(
            start_rebuild=Mock(side_effect=RuntimeError("synthetic fault")),
            _last_outcome="interrupted",
        )
        module = configure_harness(h, manager)
        with (
            patch.object(Path, "home", return_value=MemoryPath()),
            self.assertLogs("synthetic_configure", "ERROR"),
        ):
            result = json.loads(module.truememory_configure("base"))
        self.assertIsNone(module._memory)
        self.assertEqual(result["served_tier"], "edge")
        self.assertEqual(result["status"], "switch_failed")
        self.assertIn("synthetic fault", result["rebuild_error"])
        self.assertEqual(h.config["tier"], "edge")

    def test_configure_same_tier_does_not_bypass_pending(self):
        h = self.h
        h.config["tier"] = "pro"
        manager = types.SimpleNamespace(
            start_rebuild=Mock(side_effect=RuntimeError("runtime pending")),
            _last_outcome="activation_pending",
        )
        module = configure_harness(h, manager)
        with (
            patch.object(Path, "home", return_value=MemoryPath()),
            self.assertLogs("synthetic_configure", "ERROR"),
        ):
            result = json.loads(
                module.truememory_configure("pro", search_intensity="enhanced")
            )
        self.assertEqual(result["status"], "activation_pending")
        self.assertEqual(manager.start_rebuild.call_count, 1)
        self.assertEqual(h.config["search_intensity"], "enhanced")
        self.assertIsNone(module._memory)

    def test_superseded_policy_status_is_not_running(self):
        h = self.h
        conn = status_connection()
        sid = h.manager._create_status_row(conn, "basepro", "base", "streamed", 0)
        h.manager_module._open_db = lambda *args: conn
        h.state.intent = PolicyIntent()
        result = h.manager.get_status(sid)
        self.assertEqual(result["status"], "superseded")
        self.assertFalse(result["activated"])

    def test_status_reader_bounds_text_before_python_materialization(self):
        h = self.h
        conn = status_connection()
        sid = h.manager._create_status_row(conn, "basepro", "base", "streamed", 0)
        conn.execute(
            "UPDATE rebuild_status SET error=?,backup_path=?",
            ("e" * 100000, "b" * 100000),
        )
        conn.commit()
        queries = []
        conn.set_trace_callback(queries.append)
        h.manager_module._open_db = lambda *args: conn
        result = h.manager.get_status(sid)
        self.assertIsNone(result["error"])
        self.assertIsNone(result["backup_path"])
        self.assertFalse(any("SELECT *" in query for query in queries))
        self.assertTrue(
            any("length(CAST(error AS BLOB))<=4096" in query for query in queries)
        )

    def test_orphan_marker_exposed_without_claiming_row_ownership(self):
        h = self.h
        conn = status_connection()
        sid = h.manager._create_status_row(conn, "basepro", "base", "streamed", 0)
        h.manager_module._open_db = lambda *args: conn
        h.marker = types.SimpleNamespace(job_id=h.job.job_id, state="selected")
        result = h.manager.get_status(sid)
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["selected_job_id"], h.job.job_id)
        self.assertNotIn("job_id", result)

    def test_status_sql_fault_returns_unknown_and_discards_connection(self):
        h = self.h
        h.manager._guard_status = Mock(
            side_effect=sqlite3.OperationalError("status fault")
        )
        conn = sqlite3.connect(":memory:")
        h.manager_module._open_db = lambda *args: conn
        result = h.manager.get_status(17)
        self.assertEqual(result, {"status": "unknown", "error": "OperationalError"})
        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_local_cancel_rejects_wrong_explicit_job(self):
        h = self.h
        h.manager._claimed = True
        h.manager._active_status_id = 17
        h.manager._active_job_id = h.job.job_id
        with self.assertRaises(h.manager_module.TierSwitchUnsupportedError):
            h.manager.cancel(17, expected_job_id="wrong")
        self.assertFalse(h.manager._cancel_requested.is_set())

    def test_public_force_refuses_before_database_open(self):
        h = self.h
        h.manager_module._open_db = Mock(
            side_effect=AssertionError("must not initialize")
        )
        with self.assertRaises(h.manager_module.TierSwitchUnsupportedError):
            h.manager.start_rebuild("base", force=True, db_path=h.job.database_path)
        h.manager_module._open_db.assert_not_called()
        self.assertFalse(h.manager._claimed)

    def test_bootstrap_unloaded_runtime_projects_lazy_identity(self):
        h = self.h
        with patch.object(Path, "stat", side_effect=FileNotFoundError):
            self.assertTrue(
                h.manager._bootstrap_empty_configuration(
                    h.job.database_path, "pro", False
                )
            )
            self.assertEqual(h.manager._last_outcome, "complete")
            self.assertEqual(h.manager._served_tier, "pro")
            self.assertTrue(
                h.manager._bootstrap_empty_configuration(
                    h.job.database_path, "pro", False
                )
            )
        self.assertEqual(h.manager._last_outcome, "complete")
        self.assertEqual(h.manager._served_tier, "pro")
        self.assertEqual(h.config["tier"], "pro")
        self.assertEqual(
            h.modules["truememory.vector_search"].EMBEDDING_MODEL, "qwen3_256"
        )
        self.assertEqual(h.events.count("lazy_embedding"), 1)
        self.assertEqual(h.events.count("lazy_reranker"), 1)
        self.assertIsNone(h.modules["truememory.vector_search"]._model)

    def test_configure_bootstrap_pending_reports_runtime_served_tier(self):
        h = self.h
        h.config["tier"] = "pro"
        manager = types.SimpleNamespace(
            start_rebuild=Mock(return_value=0),
            _last_outcome="activation_pending",
            _last_action="onboarding",
            _served_tier="edge",
        )
        module = configure_harness(h, manager)
        with patch.object(Path, "home", return_value=MemoryPath()):
            result = json.loads(module.truememory_configure("pro"))
        self.assertEqual(result["status"], "activation_pending")
        self.assertEqual(result["tier"], "pro")
        self.assertEqual(result["served_tier"], "edge")
        self.assertNotIn("note", result)

    def test_committed_tier_reported_even_when_config_mirror_fails(self):
        h = self.h
        h.modules["truememory.tier_switch.projection"].mirror_activation = Mock(
            side_effect=RuntimeError("mirror")
        )
        with self.assertRaisesRegex(RuntimeError, "mirror"):
            h.manager._project_result(h.conn, h.result, h.intent, policy=True)
        self.assertEqual(h.manager._served_tier, "pro")
        self.assertEqual(h.manager._last_outcome, "activation_pending")

    def test_ambiguous_public_request_discards_before_fresh_readback(self):
        h = self.h
        h.manager._bootstrap_empty_configuration = lambda *args: False
        fresh = FakeConnection()
        calls = []

        def opener(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return h.conn
            self.assertTrue(h.conn.closed)
            return fresh

        def failing_prepare(*args, **kwargs):
            h.manager._last_outcome = "activation_pending"
            raise sqlite3.OperationalError("commit ambiguity")

        h.manager_module._open_db = opener
        h.manager._prepare_job = failing_prepare
        h.state.legacy_policy = h.result
        with self.assertRaises(sqlite3.Error):
            h.manager.start_rebuild("pro", db_path=h.job.database_path)
        self.assertEqual(len(calls), 2)
        self.assertTrue(fresh.closed)
        self.assertEqual(h.manager._served_tier, "pro")
        self.assertEqual(h.conn.executed, [])

    def test_unknown_readback_does_not_claim_old_config_served(self):
        h = self.h
        h.config["tier"] = "base"
        manager = types.SimpleNamespace(
            start_rebuild=Mock(return_value=0),
            _last_outcome="activation_pending",
            _served_tier=None,
        )
        module = configure_harness(h, manager)
        with patch.object(Path, "home", return_value=MemoryPath()):
            result = json.loads(module.truememory_configure("pro"))
        self.assertIsNone(result["served_tier"])
        self.assertEqual(result["status"], "activation_pending")

    def test_bootstrap_loaded_mismatch_refuses_before_config(self):
        h = self.h
        h.modules["truememory.vector_search"]._model = object()
        with (
            patch.object(Path, "stat", side_effect=FileNotFoundError),
            self.assertRaises(h.manager_module.TierSwitchUnsupportedError),
        ):
            h.manager._bootstrap_empty_configuration(h.job.database_path, "pro", False)
        self.assertEqual(h.manager._served_tier, "edge")
        self.assertEqual(h.config["tier"], "edge")
        self.assertNotIn("config", h.events)
        self.assertNotIn("lazy_embedding", h.events)

    def test_bootstrap_loaded_reranker_refuses_before_config(self):
        h = self.h
        h.modules["truememory"].reranker._model = object()
        with (
            patch.object(Path, "stat", side_effect=FileNotFoundError),
            self.assertRaises(h.manager_module.TierSwitchUnsupportedError),
        ):
            h.manager._bootstrap_empty_configuration(h.job.database_path, "pro", False)
        self.assertEqual(h.config["tier"], "edge")
        self.assertNotIn("lazy_embedding", h.events)

    def test_bootstrap_lazy_projection_error_never_writes_config(self):
        h = self.h
        h.modules["truememory.vector_search"].set_embedding_model = Mock(
            side_effect=RuntimeError("projection fault")
        )
        with (
            patch.object(Path, "stat", side_effect=FileNotFoundError),
            self.assertRaisesRegex(RuntimeError, "projection fault"),
        ):
            h.manager._bootstrap_empty_configuration(h.job.database_path, "pro", False)
        self.assertEqual(h.manager._last_outcome, "activation_pending")
        self.assertEqual(h.config["tier"], "edge")
        self.assertNotIn("config", h.events)

    def test_bootstrap_config_failure_restores_proven_original_unloaded_identity(self):
        h = self.h
        h.modules["truememory.tier_switch.projection"]._write_config = Mock(
            side_effect=OSError("before replace")
        )
        with (
            patch.object(Path, "stat", side_effect=FileNotFoundError),
            self.assertRaisesRegex(OSError, "before replace"),
        ):
            h.manager._bootstrap_empty_configuration(h.job.database_path, "pro", False)
        self.assertEqual(h.config["tier"], "edge")
        self.assertEqual(
            h.modules["truememory.vector_search"].EMBEDDING_MODEL, "model2vec"
        )
        self.assertEqual(h.modules["truememory.vector_search"].resolve_tier(), "edge")
        self.assertEqual(h.manager._served_tier, "edge")
        self.assertIsNone(h.modules["truememory.vector_search"]._model)

    def test_bootstrap_post_replace_fault_preserves_matching_new_identity(self):
        h = self.h

        def replaced_then_failed(path, config):
            h.config = config.copy()
            raise OSError("after replace")

        h.modules[
            "truememory.tier_switch.projection"
        ]._write_config = replaced_then_failed
        with (
            patch.object(Path, "stat", side_effect=FileNotFoundError),
            self.assertRaisesRegex(OSError, "after replace"),
        ):
            h.manager._bootstrap_empty_configuration(h.job.database_path, "pro", False)
        self.assertEqual(h.config["tier"], "pro")
        self.assertEqual(
            h.modules["truememory.vector_search"].EMBEDDING_MODEL, "qwen3_256"
        )
        self.assertEqual(h.manager._served_tier, "pro")
        self.assertEqual(h.manager._last_outcome, "activation_pending")

    def test_unreadable_store_cannot_be_treated_as_absent(self):
        h = self.h
        with (
            patch.object(Path, "stat", side_effect=PermissionError),
            self.assertRaises(PermissionError),
        ):
            h.manager._bootstrap_empty_configuration(h.job.database_path, "pro", False)
        self.assertEqual(h.config["tier"], "edge")
        self.assertNotIn("lazy_embedding", h.events)

    def test_cancel_cli_dispatch_uses_only_exact_supplied_identity(self):
        h = self.h
        manager = types.SimpleNamespace(
            cancel=Mock(return_value={"status": "cancelled", "job_id": h.job.job_id})
        )
        module, factory = cancel_cli_harness(h, manager)
        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "truememory-ingest",
                    "cancel-rebuild",
                    "--status-id",
                    "17",
                    "--job-id",
                    h.job.job_id,
                    "--db",
                    "/synthetic/store.db",
                ],
            ),
            contextlib.redirect_stdout(output),
        ):
            module.main()
        factory.assert_called_once_with()
        manager.cancel.assert_called_once_with(
            17, expected_job_id=h.job.job_id, db_path=h.job.database_path
        )
        self.assertIn("Rebuild 17 cancelled", output.getvalue())
        self.assertIn("explicitly", output.getvalue())
        self.assertEqual(h.events, [])

    def test_cancel_cli_parser_requires_both_exact_ids_before_manager(self):
        h = self.h
        manager = types.SimpleNamespace(cancel=Mock())
        module, factory = cancel_cli_harness(h, manager)
        invalid = [[], ["--status-id", "17"], ["--job-id", h.job.job_id]]
        invalid += [
            ["--status-id", value, "--job-id", h.job.job_id]
            for value in (
                "0",
                "-1",
                "+17",
                " 17",
                "017",
                "1_7",
                "1.0",
                "9223372036854775808",
                "9" * 200,
            )
        ]
        invalid += [
            ["--status-id", "17", "--job-id", value]
            for value in ("a" * 31, "a" * 33, "A" * 32, "g" * 32, " " + h.job.job_id)
        ]
        for arguments in invalid:
            with (
                self.subTest(arguments=arguments),
                patch.object(
                    sys, "argv", ["truememory-ingest", "cancel-rebuild", *arguments]
                ),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as error:
                    module.main()
                self.assertEqual(error.exception.code, 2)
        factory.assert_not_called()
        manager.cancel.assert_not_called()

    def test_cancel_cli_direct_handler_rejects_bool_and_invalid_job(self):
        h = self.h
        manager = types.SimpleNamespace(cancel=Mock())
        module, factory = cancel_cli_harness(h, manager)
        for status_id, job_id in (
            (True, h.job.job_id),
            (0, h.job.job_id),
            (17, "A" * 32),
        ):
            with (
                self.subTest(status_id=status_id, job_id=job_id),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as error:
                    module._run_cancel_rebuild(
                        argparse.Namespace(status_id=status_id, job_id=job_id, db=None)
                    )
                self.assertEqual(error.exception.code, 2)
        factory.assert_not_called()

    def test_cancel_cli_accepts_maximum_positive_sqlite_status_without_substitution(
        self,
    ):
        h = self.h
        manager = types.SimpleNamespace(
            cancel=Mock(return_value={"status": "cancelled", "job_id": h.job.job_id})
        )
        module, _ = cancel_cli_harness(h, manager)
        with (
            patch.object(
                sys,
                "argv",
                [
                    "truememory-ingest",
                    "cancel-rebuild",
                    "--status-id",
                    "9223372036854775807",
                    "--job-id",
                    h.job.job_id,
                ],
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            module.main()
        manager.cancel.assert_called_once_with(
            9223372036854775807, expected_job_id=h.job.job_id, db_path=None
        )

    def test_cancel_cli_busy_failure_is_nonzero_and_never_retries(self):
        h = self.h
        manager = types.SimpleNamespace(
            cancel=Mock(side_effect=RuntimeError("maintenance owner busy"))
        )
        module, _ = cancel_cli_harness(h, manager)
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as error:
            module._run_cancel_rebuild(
                argparse.Namespace(status_id=17, job_id=h.job.job_id, db=None)
            )
        self.assertEqual(error.exception.code, 1)
        manager.cancel.assert_called_once_with(
            17, expected_job_id=h.job.job_id, db_path=None
        )
        self.assertIn("maintenance owner busy", output.getvalue())
        self.assertIn("truememory_status(status_id)", output.getvalue())

    def test_cancel_cli_nonterminal_or_mismatched_result_never_claims_cancelled(self):
        h = self.h
        for result in (
            {"status": "no_selected_job"},
            {"status": "cancellation_requested", "status_id": 17},
            {"status": "cancelled", "job_id": "b" * 32},
        ):
            manager = types.SimpleNamespace(cancel=Mock(return_value=result))
            module, _ = cancel_cli_harness(h, manager)
            output = io.StringIO()
            with (
                self.subTest(result=result),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as error:
                    module._run_cancel_rebuild(
                        argparse.Namespace(status_id=17, job_id=h.job.job_id, db=None)
                    )
                self.assertEqual(error.exception.code, 1)
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(manager.cancel.call_count, 1)

    def test_cancel_cli_help_names_existing_status_and_explicit_retry(self):
        h = self.h
        manager = types.SimpleNamespace(cancel=Mock())
        module, factory = cancel_cli_harness(h, manager)
        output = io.StringIO()
        with (
            patch.object(
                sys, "argv", ["truememory-ingest", "cancel-rebuild", "--help"]
            ),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaises(SystemExit) as error:
                module.main()
        self.assertEqual(error.exception.code, 0)
        self.assertIn("truememory_status(status_id)", output.getvalue())
        self.assertIn("--status-id", output.getvalue())
        self.assertIn("--job-id", output.getvalue())
        self.assertIn("Retry upgrade-tier explicitly", " ".join(output.getvalue().split()))
        factory.assert_not_called()

    def test_public_status_recovery_command_is_bound_to_matching_marker(self):
        h = self.h
        conn = status_connection()
        sid = h.manager._create_status_row(conn, "basepro", "base", "streamed", 0)
        h.intent.status_id = sid
        h.state.intent = h.intent
        h.marker = types.SimpleNamespace(**vars(h.job), state="selected")
        h.manager_module._open_db = lambda *args: conn
        result = h.manager.get_status(sid)
        self.assertEqual(result["job_id"], h.job.job_id)
        self.assertIn(
            f"cancel-rebuild --status-id {sid} --job-id {h.job.job_id}",
            result["recovery"],
        )
        self.assertIn("same database path", result["recovery"])

    def test_orphan_status_does_not_offer_bound_cli_cancellation(self):
        h = self.h
        conn = status_connection()
        sid = h.manager._create_status_row(conn, "basepro", "base", "streamed", 0)
        h.manager_module._open_db = lambda *args: conn
        h.marker = types.SimpleNamespace(job_id=h.job.job_id, state="selected")
        result = h.manager.get_status(sid)
        self.assertNotIn("job_id", result)
        self.assertNotIn("--status-id", result["recovery"])
        self.assertIn("does not prove", result["recovery"])

    def test_exact_job_with_wrong_positive_status_cannot_cancel_marker(self):
        h = self.h
        h.marker = types.SimpleNamespace(**vars(h.job), state="selected")
        h.state.intent = h.intent
        with self.assertRaises(h.manager_module.TierSwitchUnsupportedError):
            h.manager.cancel(18, expected_job_id=h.job.job_id)
        self.assertTrue(h.conn.closed)
        self.assertFalse(
            any(isinstance(item, tuple) and item[0] == "cancel" for item in h.events)
        )


class ActualLegacyConfigurationComposition(unittest.TestCase):
    """Compose actual lazy setters, manager and runtime without native imports."""

    def setUp(self):
        import os
        import runpy
        import uuid
        self.os = os
        self.ns = runpy.run_path(str(ROOT / "tests/test-tier-public-boundaries-795.py"))
        self.boundary = self.ns["TestPublicBoundaries"]()
        self.addCleanup(self.boundary.doCleanups)
        self.boundary.setUp()
        self.conn, self.api = self.boundary.conn, self.boundary.api
        self.selected_marker_row = self.conn.execute("SELECT * FROM truememory_tier_selected_job_v1").fetchone()
        self.conn.execute("DELETE FROM truememory_tier_selected_job_v1")
        self.conn.executemany("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                              (("embed_model", "model2vec"), ("embed_dim", "256")))
        self.conn.commit()
        self.cfg = self.boundary.load_application("tier_config")
        self.ns["enter_context"](self, patch.dict(os.environ, {"TRUEMEMORY_EMBED_MODEL": "edge"}))
        definitions, namespace = self.ns["definitions"], self.boundary.namespace
        self.vector = definitions("vector_search.py", {
            "_resolve_model_name", "set_embedding_model", "get_embedding_dim", "resolve_tier",
            "_active_tier_group", "_active_vec_table", "_active_sep_table",
        }, namespace(os=os, sqlite3=sqlite3, logger=logging.getLogger("synthetic"),
            _model=None, _lock=threading.Lock(), _model_generation=0,
            _frozen_embedding_target=None, _runtime_policy_tier=None,
            EMBEDDING_MODEL="model2vec", _embedding_dim=256, _REMOVED_MODELS={"qwen3"},
            _TIER_ALIASES={tier: config["embed_model"] for tier, config in self.cfg.TIERS.items()},
            _MODEL_DIMS=self.cfg.MODEL_DIMS, _cfg_get_embed_model=self.cfg.get_embed_model,
            _cfg_get_embed_dim_for_model=self.cfg.get_embed_dim_for_model,
            _cfg_get_model_group=self.cfg.get_model_group))
        self.reranker = definitions("reranker.py", {
            "set_active_tier", "get_current_reranker_name", "get_reranker_name_for_tier",
        }, namespace(_model=None, _active_tier="edge", _frozen_reranker_id=None,
            _TIER_RERANKERS={tier: config["reranker"] for tier, config in self.cfg.TIERS.items()},
            _cfg_get_reranker=self.cfg.get_reranker, log=logging.getLogger("synthetic")))
        self.boundary.fixture.vector, self.boundary.fixture.reranker = self.vector, self.reranker
        self.boundary.modules["vector_search"] = self.vector
        self.boundary.modules["reranker"] = self.reranker
        self.h = Harness()
        for name, module in self.boundary.modules.items():
            self.h.modules["truememory." + name] = module
        self.h.modules["truememory"] = types.SimpleNamespace(vector_search=self.vector, reranker=self.reranker)
        self.h.modules["truememory.tier_config"] = self.cfg
        self.h.manager_module.os = os
        self.projection = self.h.modules["truememory.tier_switch.projection"]
        guard = definitions("tier_switch/projection.py", {"_guard", "ConfigProjectionConflict"},
                            namespace(_GENERATION_KEY="tier_activation_generation"))
        self.projection._guard = guard._guard
        self.h.manager._config_path = Path("/synthetic/config.json")
        self.h.manager._config_lock_path = Path("/synthetic/config.lock")
        self.owner = definitions("maintenance.py", {
            "MaintenanceBusyError", "MaintenanceOwnership", "_acquire_owner", "_release_owner",
            "_bind_owner", "maintenance_owner",
        }, namespace(NamedTuple=__import__("typing").NamedTuple,
            contextmanager=contextlib.contextmanager, Path=Path,
            os=types.SimpleNamespace(getpid=os.getpid, close=lambda fd: None), uuid=uuid,
            canonical_database_path=lambda value: self.h.job.database_path,
            _registry_lock=threading.Lock(), _held_paths={}, _held_fds=set(),
            _owner_waiters={}, _coordinators={}, _thread_owners=threading.local(),
            try_file_lock=lambda path: 987))
        self.h.modules["truememory.maintenance"] = self.owner
        self.h.manager_module._open_db = Mock(side_effect=AssertionError("Unexpected writable/native open"))
        self.sql = []
        self.conn.set_trace_callback(self.sql.append)

    def connect_read_snapshot(self):
        connection = self.conn
        class OwnedHandle:
            closed = False
            def __getattr__(self, name):
                return getattr(connection, name)
            def close(self):
                self.closed = True
        handle = OwnedHandle()
        opener = Mock(return_value=handle)
        self.h.manager_module.sqlite3 = types.SimpleNamespace(connect=opener)
        self.ns["enter_context"](self, patch.object(Path, "stat", return_value=types.SimpleNamespace()))
        return handle, opener

    def assert_no_persistent_mutation(self):
        self.assertFalse(any(sql.lstrip().split(" ", 1)[0].upper() in {
            "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "REPLACE",
        } for sql in self.sql))
        self.assertNotIn("config", self.h.events)
        self.assertFalse(self.conn.in_transaction)
        self.h.manager_module._open_db.assert_not_called()
        self.assertEqual(self.boundary.fixture.loads, [])

    def test_actual_bootstrap_then_fresh_legacy_admission_stays_lazy(self):
        for tier in ("base", "pro", "edge"):
            with self.subTest(tier=tier), patch.object(Path, "stat", side_effect=FileNotFoundError):
                self.assertTrue(self.h.manager._bootstrap_empty_configuration(self.h.job.database_path, tier, False))
            with self.api.serving_operation(self.conn) as operation:
                self.assertEqual(operation.tier, tier)
                self.assertIsNone(operation.selection)
                self.assertIsNone(operation.policy)
                self.assertEqual(operation.reranker_id, self.cfg.get_tier_config(tier)["reranker"])
            self.assertEqual(self.vector.resolve_tier(), tier)
            self.assertIsNone(self.vector._runtime_policy_tier)
            self.assertIsNone(self.vector._frozen_embedding_target)
            self.assertIsNone(self.reranker._frozen_reranker_id)
            self.assertIsNone(self.vector._model)
            self.assertIsNone(self.reranker._model)
            self.assertEqual(self.boundary.fixture.loads, [])
            self.assertEqual(self.h.config["tier"], tier)

    def test_actual_bootstrap_refuses_controlled_markers_before_ready_or_config(self):
        target = self.boundary.modules["embedding_target"].EmbeddingTarget.capture("base")
        with self.boundary.fixture.gate.exclusive_activation():
            self.h.manager._apply_bootstrap_identity(target, self.cfg.get_tier_config("base")["reranker"])
        for marker, value in (("_runtime_policy_tier", "base"), ("_frozen_embedding_target", target),
                              ("_frozen_reranker_id", self.cfg.get_tier_config("base")["reranker"])):
            slot = self.reranker if marker == "_frozen_reranker_id" else self.vector
            for loaded in (False, True):
                with self.subTest(marker=marker, loaded=loaded):
                    self.vector._model = object() if loaded else None
                    self.reranker._model = object() if loaded else None
                    before = self.vector._model, self.reranker._model
                    setattr(slot, marker, value)
                    with patch.object(Path, "stat", side_effect=FileNotFoundError):
                        with self.assertRaisesRegex(self.h.manager_module.TierSwitchUnsupportedError, "controlled runtime"):
                            self.h.manager._bootstrap_empty_configuration(self.h.job.database_path, "base", False)
                    self.assertEqual((self.vector._model, self.reranker._model), before)
                    self.assertEqual(getattr(slot, marker), value)
                    self.assertEqual(self.h.config["tier"], "edge")
                    self.assertNotIn("config", self.h.events)
                    setattr(slot, marker, None)
        self.assertEqual(self.boundary.fixture.loads, [])

    def test_actual_bootstrap_preserves_loaded_ordinary_same_tier_certified_reranker(self):
        self.vector._model, self.reranker._model = object(), object()
        self.reranker._model_certified = True
        before = self.vector._model, self.reranker._model
        with patch.object(Path, "stat", side_effect=FileNotFoundError):
            self.assertTrue(self.h.manager._bootstrap_empty_configuration(self.h.job.database_path, "edge", False))
        self.assertEqual((self.vector._model, self.reranker._model), before)
        self.assertTrue(self.reranker._model_certified)
        self.assertIsNone(self.reranker._frozen_reranker_id)
        with self.api.serving_operation(self.conn) as operation:
            self.assertEqual(operation.tier, "edge")
        self.assertEqual(self.boundary.fixture.loads, [])

    def test_actual_bootstrap_config_fault_restores_old_legacy_admission(self):
        self.projection._write_config = Mock(side_effect=OSError("before replace"))
        with patch.object(Path, "stat", side_effect=FileNotFoundError), self.assertRaisesRegex(OSError, "before replace"):
            self.h.manager._bootstrap_empty_configuration(self.h.job.database_path, "pro", False)
        with self.api.serving_operation(self.conn) as operation:
            self.assertEqual(operation.tier, "edge")
        self.assertEqual(self.h.config["tier"], "edge")
        self.assertEqual(self.vector.EMBEDDING_MODEL, "model2vec")
        self.assertIsNone(self.vector._runtime_policy_tier)
        self.assertIsNone(self.reranker._frozen_reranker_id)
        self.assertEqual(self.boundary.fixture.loads, [])

    def test_actual_bootstrap_post_replace_fault_keeps_new_legacy_admission(self):
        def replace_then_fail(path, config):
            self.h.config = config.copy()
            raise OSError("after replace")
        self.projection._write_config = replace_then_fail
        with patch.object(Path, "stat", side_effect=FileNotFoundError), self.assertRaisesRegex(OSError, "after replace"):
            self.h.manager._bootstrap_empty_configuration(self.h.job.database_path, "pro", False)
        with self.api.serving_operation(self.conn) as operation:
            self.assertEqual(operation.tier, "pro")
        self.assertEqual(self.h.config["tier"], "pro")
        self.assertEqual(self.h.manager._last_outcome, "activation_pending")
        self.assertIsNone(self.vector._runtime_policy_tier)
        self.assertEqual(self.boundary.fixture.loads, [])

    def test_actual_noop_while_other_maintenance_owner_and_reader_are_held(self):
        held, release = threading.Event(), threading.Event()
        errors = []
        def maintenance():
            try:
                with self.owner.maintenance_owner(self.h.job.database_path), self.boundary.fixture.gate.operation_lease(self.api._legacy_key()):
                    held.set()
                    if not release.wait(3):
                        raise AssertionError("Synthetic owner release timed out")
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=maintenance)
        thread.start()
        try:
            self.assertTrue(held.wait(1))
            with self.assertRaises(self.owner.MaintenanceBusyError):
                with self.owner.maintenance_owner(self.h.job.database_path):
                    self.fail("Negative control unexpectedly acquired busy owner")
            self.vector._model, self.reranker._model = object(), object()
            before_models = self.vector._model, self.reranker._model
            writer = self.boundary.load_application("tier_switch.writer")
            captured = writer.capture_writer_selection(self.conn)
            handle, opener = self.connect_read_snapshot()
            self.assertEqual(self.h.manager.start_rebuild("edge", db_path=self.h.job.database_path), 0)
            self.assertEqual(self.h.manager._last_action, "noop")
            self.assertEqual(self.h.manager._last_outcome, "complete")
            self.assertTrue(handle.closed)
            self.assertEqual(opener.call_count, 1)
            self.assertTrue(opener.call_args.args[0].endswith("?mode=ro"))
            self.assertEqual((self.vector._model, self.reranker._model), before_models)
            self.assertEqual(writer.capture_writer_selection(self.conn), captured)
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                writer.require_writer_selection(self.conn, captured)
            finally:
                self.conn.rollback()
            self.assert_no_persistent_mutation()
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_actual_noop_sync_path_avoids_owner(self):
        self.connect_read_snapshot()
        self.owner.maintenance_owner = Mock(side_effect=AssertionError("Unexpected owner"))
        self.assertTrue(self.h.manager._run_rebuild_sync_inner("edge", db_path=self.h.job.database_path))
        self.owner.maintenance_owner.assert_not_called()
        self.assert_no_persistent_mutation()

    def test_actual_noop_generation_tagged_config_cannot_bypass_recovery(self):
        self.h.config["tier_activation_generation"] = "a" * 32
        handle, _ = self.connect_read_snapshot()
        self.assertFalse(self.h.manager._try_legacy_noop(self.h.job.database_path, "edge"))
        self.assertTrue(handle.closed)
        self.assert_no_persistent_mutation()

    def test_actual_noop_staged_intent_and_orphan_job_cannot_bypass_recovery(self):
        self.conn.execute("INSERT INTO truememory_tier_selected_job_v1 VALUES (" + ",".join("?" for _ in self.selected_marker_row) + ")",
                          self.selected_marker_row)
        self.conn.commit()
        self.boundary.fixture.fixture.build()
        self.connect_read_snapshot()
        self.sql.clear()
        self.assertFalse(self.h.manager._try_legacy_noop(self.h.job.database_path, "edge"))
        self.assert_no_persistent_mutation()
        self.conn.execute("DELETE FROM metadata WHERE key='tier_activation_v1'")
        self.conn.commit()
        self.sql.clear()
        self.assertFalse(self.h.manager._try_legacy_noop(self.h.job.database_path, "edge"))
        self.assert_no_persistent_mutation()

    def test_actual_noop_policy_and_selected_authority_cannot_bypass_recovery(self):
        self.boundary.seed_policy()
        self.connect_read_snapshot()
        self.sql.clear()
        self.assertFalse(self.h.manager._try_legacy_noop(self.h.job.database_path, "pro"))
        self.assert_no_persistent_mutation()

    def test_actual_noop_rejects_incoherent_embedding_without_writes(self):
        self.vector.EMBEDDING_MODEL = "qwen3_256"
        self.connect_read_snapshot()
        with self.assertRaisesRegex(self.h.manager_module.TierSwitchUnsupportedError, "Legacy model configuration"):
            self.h.manager._try_legacy_noop(self.h.job.database_path, "edge")
        self.assert_no_persistent_mutation()

    def test_actual_noop_rejects_incoherent_reranker_without_writes(self):
        self.reranker._active_tier = "base"
        self.connect_read_snapshot()
        with self.assertRaisesRegex(self.h.manager_module.TierSwitchUnsupportedError, "Legacy runtime identity"):
            self.h.manager._try_legacy_noop(self.h.job.database_path, "edge")
        self.assert_no_persistent_mutation()

    def test_concurrent_actual_configure_reports_preparing_until_status_allocated(self):
        h = Harness()
        entered, release = threading.Event(), threading.Event()
        failures = []
        def bootstrap(*args):
            entered.set()
            if not release.wait(3):
                raise AssertionError("Preparation release timed out")
            h.manager._last_outcome = "complete"
            return True
        h.manager._bootstrap_empty_configuration = bootstrap
        def first_call():
            try:
                h.manager.start_rebuild("pro", db_path=h.job.database_path)
            except BaseException as error:
                failures.append(error)
        thread = threading.Thread(target=first_call)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            module = configure_harness(h, h.manager)
            with patch.object(Path, "home", return_value=MemoryPath()):
                result = json.loads(module.truememory_configure("pro"))
            self.assertTrue(h.manager._claimed)
            self.assertEqual(h.manager._active_status_id, 0)
            self.assertEqual(h.manager._last_outcome, "preparing")
            self.assertEqual(result["status"], "preparing")
            self.assertIsNone(result["served_tier"])
            self.assertNotIn("status_id", result)
            self.assertNotIn("rebuild_error", result)
            self.assertIn("preparation", result["note"])
            self.assertNotIn("TrueMemory is ready", result["next_steps"])
            self.assertIsNone(module._memory)
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(h.events.count("owner_enter"), 1)

    def test_actual_noop_sql_fault_closes_only_owned_connection(self):
        handle, _ = self.connect_read_snapshot()
        original = self.conn.execute
        def execute(sql, *args):
            if "temp.sqlite_master" in sql:
                raise sqlite3.OperationalError("synthetic unreadable authority")
            return original(sql, *args)
        with patch.object(self.conn, "execute", side_effect=execute):
            with self.assertRaisesRegex(sqlite3.OperationalError, "synthetic unreadable authority"):
                self.h.manager._try_legacy_noop(self.h.job.database_path, "edge")
        self.assertTrue(handle.closed)
        self.assertIsNone(self.h.manager._last_action)
        self.assert_no_persistent_mutation()


if __name__ == "__main__":
    unittest.main()
