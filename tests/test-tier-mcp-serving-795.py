"""MCP policy capture and child cleanup with actual gates and synthetic peers."""
from __future__ import annotations

import ast
import builtins
import json
import logging
import runpy
import threading
import types
import unittest
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GATE = runpy.run_path(str(ROOT / "truememory/tier_switch/serving.py"))


class PolicyRejected(RuntimeError):
    pass


class TestMCPServing(unittest.TestCase):
    def setUp(self) -> None:
        self.events = []
        self.gate = GATE["ServingGate"]()
        self.tier = "pro"
        self.llm = object()
        self.deep_llm = None
        self.query_failure = None
        self.construction_failure = False
        self.maintenance_requests = []
        owner = self

        class Memory:
            def __init__(self, path="synthetic"):
                if owner.construction_failure:
                    raise OSError("synthetic open failure")
                self._engine = types.SimpleNamespace(db_path=path, conn=object(), _write_lock=threading.Lock())
                self._engine._open_connection_handle = lambda: owner.events.append(("open", self._engine.conn))
                self._engine._runtime_initialized = True
                self._engine.ready = True
                def request_maintenance():
                    owner.assert_drained()
                    owner.maintenance_requests.append(self._engine.conn)
                self._engine._maybe_auto_consolidate = request_maintenance
                self._engine._maybe_startup_consolidate = request_maintenance

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                owner.events.append(("close", self._engine.conn))

            def search_deep(self, query, **kwargs):
                owner.events.append(("query", query, kwargs))
                if owner.query_failure is not None:
                    raise owner.query_failure
                return [{"id": "shared", "score": 2}, {"id": query, "score": 1}]

        self.memory = Memory()

        @contextmanager
        def serving_operation(conn, *, connection_lock=None, reranker_id=None, _defer_maintenance=False):
            owner.events.append(("admit", conn, reranker_id))
            with owner.gate.operation_lease(("synthetic-generation",)) as lease:
                def fork_child():
                    child = lease.fork_child()
                    owner.events.append(("reserve", child))

                    @contextmanager
                    def join(*, conn):
                        owner.events.append(("join", conn))
                        with child.join():
                            yield

                    return types.SimpleNamespace(join=join, close=child.close)

                yield types.SimpleNamespace(tier=owner.tier, reranker_id=reranker_id or "synthetic/default",
                                            fork_child=fork_child, _defer_maintenance=_defer_maintenance)
            owner.events.append(("release", conn))

        def reject(error):
            if isinstance(error, PolicyRejected):
                raise error

        runtime = types.SimpleNamespace(serving_operation=serving_operation, raise_if_serving_rejection=reject)

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "truememory.tier_switch.runtime":
                return runtime
            return builtins.__import__(name, globals, locals, fromlist, level)

        runtime_tree = ast.parse((ROOT / "truememory/tier_switch/runtime.py").read_text(encoding="utf-8"))
        context_names = {"open_engine_for_operation", "engine_serving_operation"}
        context_definitions = [node for node in runtime_tree.body
                               if isinstance(node, ast.FunctionDef) and node.name in context_names]
        self.assertEqual({node.name for node in context_definitions}, context_names)
        context_namespace = dict(__builtins__=dict(vars(builtins), __import__=safe_import),
                                 contextmanager=contextmanager, serving_operation=serving_operation,
                                 current_operation=lambda conn: None)
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *context_definitions], type_ignores=[])),
                     "actual-public-context", "exec"), context_namespace)
        runtime.engine_serving_operation = context_namespace["engine_serving_operation"]

        names = {"_search_operation", "_parallel_search", "_resolve_deepsearch_llm",
                 "truememory_search", "truememory_search_deep"}
        tree = ast.parse((ROOT / "truememory/mcp_server.py").read_text(encoding="utf-8"))
        nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                node.decorator_list = [d for d in node.decorator_list
                                       if isinstance(d, ast.Name) and d.id == "contextmanager"]
                nodes.append(node)
        self.ns = dict(__builtins__=dict(vars(builtins), __import__=safe_import), contextmanager=contextmanager,
                       Iterator=Iterator, Memory=Memory, ThreadPoolExecutor=ThreadPoolExecutor, json=json, threading=threading,
                       log=logging.getLogger(__name__), _get_memory=lambda: self.memory,
                       _get_llm_fn=lambda: self.llm, _get_deepsearch_llm_fn=lambda: self.deep_llm,
                       _load_config=lambda: {"tier": "edge"}, _touch_search_time=lambda: None,
                       _set_reranker=lambda name: self.events.append(("reranker", name)),
                       _SEARCH_INTERNAL_LIMIT=100, _DEEP_INTERNAL_LIMIT=500,
                       _DEEP_RERANKER="BAAI/bge-reranker-v2-m3")
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     "actual-mcp-serving", "exec"), self.ns)

    def assert_drained(self) -> None:
        with self.gate.exclusive_activation(timeout=0.1):
            pass

    def test_outer_request_schedules_once_after_all_children_release(self) -> None:
        self.ns["truememory_search"]("first | second")
        self.assertEqual(self.maintenance_requests, [self.memory._engine.conn])
        self.assertEqual(self.events[-1][0], "release")
        self.assert_drained()

    def test_failed_child_schedules_once_after_request_cleanup(self) -> None:
        self.query_failure = PolicyRejected("synthetic rejection")
        with self.assertRaises(PolicyRejected):
            self.ns["truememory_search"]("first | second")
        self.assertEqual(self.maintenance_requests, [self.memory._engine.conn])
        self.assert_drained()

    def test_parallel_handles_open_serially_and_release_before_search(self) -> None:
        first_opened = threading.Event()
        both_constructed = threading.Event()
        second_opened = threading.Event()
        release = threading.Event()
        mutex = threading.Lock()
        constructed = []
        opened = []
        errors = []
        outputs = []
        owner = self
        original = self.ns["Memory"]

        class ConcurrentMemory(original):
            def __init__(self, path="synthetic"):
                super().__init__(path)
                with mutex:
                    constructed.append(self)
                    if len(constructed) == 2:
                        both_constructed.set()
                def open_handle():
                    with mutex:
                        opened.append(self)
                        number = len(opened)
                    if number == 1:
                        first_opened.set()
                        if not release.wait(2):
                            raise AssertionError("First open was not released")
                    else:
                        second_opened.set()
                self._engine._open_connection_handle = open_handle

            def search_deep(self, query, **kwargs):
                owner.assertTrue(second_opened.wait(2), "Opening lock was retained during search")
                return super().search_deep(query, **kwargs)

        self.ns["Memory"] = ConcurrentMemory
        def run_request():
            try:
                outputs.extend(json.loads(self.ns["truememory_search"]("first | second")))
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=run_request)
        worker.start()
        try:
            self.assertTrue(first_opened.wait(2))
            self.assertTrue(both_constructed.wait(2))
            self.assertFalse(second_opened.wait(0.05), "Sibling opened during the first maintenance owner")
            self.assertEqual(len(opened), 1)
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(opened), 2)
        self.assertEqual([row["id"] for row in outputs], ["shared", "first", "second"])
        self.assertEqual(self.maintenance_requests, [self.memory._engine.conn])
        self.assert_drained()

    def test_standard_uses_admitted_policy_despite_old_config(self) -> None:
        result = json.loads(self.ns["truememory_search"]("synthetic", limit=1))
        query = next(event for event in self.events if event[0] == "query")
        self.assertEqual(len(result), 1)
        self.assertEqual(query[2]["limit"], 100)
        self.assertIs(query[2]["llm_fn"], self.llm)
        self.assertEqual(self.events[0][0], "open")
        self.assertEqual(self.events[1][0], "admit")
        self.assertEqual(self.events[-1][0], "release")
        self.assert_drained()

    def test_deep_override_and_edge_explicit_llm_survive(self) -> None:
        self.tier = "edge"
        self.deep_llm = object()
        self.ns["truememory_search_deep"]("synthetic")
        query = next(event for event in self.events if event[0] == "query")
        self.assertEqual(self.events[1][2], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(query[2]["limit"], 500)
        self.assertIs(query[2]["llm_fn"], self.deep_llm)
        self.assertIn(("reranker", "BAAI/bge-reranker-v2-m3"), self.events)
        self.assert_drained()

    def test_deep_default_llm_is_pro_only(self) -> None:
        self.assertIsNone(self.ns["_resolve_deepsearch_llm"]("edge"))
        self.assertIs(self.ns["_resolve_deepsearch_llm"]("pro"), self.llm)

    def test_parallel_children_join_and_release_before_parent(self) -> None:
        result = json.loads(self.ns["truememory_search"]("first | second"))
        self.assertEqual([row["id"] for row in result], ["shared", "first", "second"])
        self.assertEqual(sum(e[0] == "reserve" for e in self.events), 2)
        self.assertEqual(sum(e[0] == "join" for e in self.events), 2)
        self.assertEqual(sum(e[0] == "close" for e in self.events), 2)
        self.assertEqual(self.events[-1][0], "release")
        self.assert_drained()

    def test_identity_rejection_cannot_return_partial_success(self) -> None:
        self.query_failure = PolicyRejected("synthetic stale selection")
        with self.assertRaises(PolicyRejected):
            self.ns["truememory_search"]("first | second")
        self.assert_drained()

    def test_failed_child_constructor_releases_unused_reservation(self) -> None:
        self.construction_failure = True
        self.assertEqual(json.loads(self.ns["truememory_search"]("first | second")), [])
        self.assertEqual(sum(e[0] == "reserve" for e in self.events), 2)
        self.assert_drained()

    def test_failed_submission_releases_every_reserved_child(self) -> None:
        original = ThreadPoolExecutor.submit
        calls = 0

        def submit(pool, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic submission failure")
            return original(pool, *args, **kwargs)

        with patch.object(ThreadPoolExecutor, "submit", submit):
            with self.assertRaisesRegex(RuntimeError, "submission"):
                self.ns["truememory_search"]("first | second")
        self.assert_drained()

    def test_input_caps_and_empty_input_are_preserved(self) -> None:
        self.assertEqual(self.ns["truememory_search"](" "), "[]")
        self.assertEqual(self.events, [])
        self.ns["truememory_search"]("", queries=[str(n) + "x" * 2100 for n in range(12)], limit=999)
        queries = [e for e in self.events if e[0] == "query"]
        self.assertEqual(len(queries), 10)
        self.assertTrue(all(len(e[1]) == 2000 for e in queries))
        self.assertTrue(all(e[2]["limit"] == 100 for e in queries))
        self.assert_drained()


if __name__ == "__main__":
    unittest.main()
