"""Synthetic native bridge controls using actual source and inert collaborators.

No package entry point, native host, model or database is imported. Fixture
trees, including failed attempts, are retained. Local runs use an announced
exclusive TRUEMEMORY_HERMES_TEST_ROOT; hosted runs retain a temporary tree.
"""
from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_source(path: Path, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


BRIDGE = load_source(ROOT / "truememory/ingest/hooks/hermes.py", "synthetic_hermes765")
TRANSCRIPT = load_source(ROOT / "truememory/ingest/transcript.py", "synthetic_transcript765")


@contextmanager
def collaborators(children: dict[str, types.ModuleType]) -> Iterator[None]:
    modules: dict[str, types.ModuleType] = {}
    for name in ("truememory", "truememory.ingest", "truememory.ingest.hooks", "truememory.hooks"):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
    modules.update(children)
    with patch.dict(sys.modules, modules):
        yield


def inert_shared() -> types.ModuleType:
    module = types.ModuleType("truememory.ingest.hooks._shared")
    module.check_extraction_budget = lambda: True
    module.refund_extraction_budget = lambda: None
    return module


def payload(session: str = "synthetic-session", turn: str = "synthetic-turn", *,
            history: list[dict[str, object]] | None = None, post: bool = False) -> dict[str, object]:
    extra: dict[str, object] = {
        "turn_id": turn, "task_id": "synthetic-task", "user_message": "I prefer synthetic blue widgets.",
        "conversation_history": history or [], "sender_id": "untrusted-other-account",
    }
    if post:
        extra["assistant_response"] = "Synthetic assistant reply."
    return {"session_id": session, "extra": extra, "event": "untrusted-event",
            "user_id": "untrusted-other-account", "transcript_path": "/never/read/native/input"}


class HermesNativeBridgeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        explicit = os.environ.get("TRUEMEMORY_HERMES_TEST_ROOT")
        cls.fixture_root = Path(explicit) if explicit else Path(tempfile.mkdtemp(prefix="hermes765-"))
        if explicit:
            cls.fixture_root.mkdir(parents=True, exist_ok=False)
        cls.fixture_root = cls.fixture_root.resolve(strict=True)

    def setUp(self) -> None:
        self.case = self.fixture_root / self._testMethodName
        self.case.mkdir()
        self.store = BRIDGE.Store(self.case / "owned", "synthetic-user", str(self.case / "synthetic.db"),
                                  "synthetic-session")

    def test_exact_envelope_identity_and_configured_account(self) -> None:
        key = self.store.capture(payload(), "pre_llm_call")
        state = self.store.state()
        snapshot = self.store.snapshot(next(iter(state["captures"])))
        self.assertEqual(snapshot["user"], "synthetic-user")
        self.assertEqual(snapshot["db"], str(self.case / "synthetic.db"))
        self.assertEqual(snapshot["native_user_message"], "I prefer synthetic blue widgets.")
        self.assertEqual(state["turns"][key]["ordinal"], 1)
        self.assertNotIn("transcript_path", snapshot)

    def test_same_turn_replay_and_equal_text_new_turn_session(self) -> None:
        first = self.store.capture(payload(), "pre_llm_call")
        self.assertEqual(first, self.store.capture(payload(), "pre_llm_call"))
        self.assertEqual(len(self.store.state()["captures"]), 1)
        second = self.store.capture(payload(turn="next-turn"), "pre_llm_call")
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.state()["turns"][second]["ordinal"], 2)
        other = BRIDGE.Store(self.store.root, self.store.user, self.store.db, "other-session")
        other_key = other.capture(payload(session="other-session"), "pre_llm_call")
        self.assertNotEqual(first, other_key)
        self.assertEqual(len(other.state()["captures"]), 1)

    def test_missing_invalid_identity_has_no_unknown_fallback(self) -> None:
        for value in (None, [], 1, "", " ", "unknown", "a\n", "x" * 1025):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(BRIDGE.BridgeError):
                    BRIDGE._identity(value, "session_id")
        event = payload()
        event["extra"]["turn_id"] = None
        with self.assertRaises(BRIDGE.BridgeError):
            self.store.capture(event, "pre_llm_call")
        self.assertEqual(self.store.state()["captures"], {})

    def test_capture_rejects_cross_session_payload(self) -> None:
        with self.assertRaises(BRIDGE.BridgeError):
            self.store.capture(payload(session="wrong-session"), "pre_llm_call")
        self.assertEqual(self.store.state()["captures"], {})

    def test_history_current_text_and_supported_parts_no_sidecars(self) -> None:
        event = payload(history=[{"role": "user", "content": "Older native user.",
                                  "api_content": "injected API sidecar"}])
        event["extra"]["user_message"] = [{"type": "text", "text": "Native current text."},
                                           {"type": "image_url", "image_url": "ignored"}]
        self.store.capture(event, "pre_llm_call")
        snapshot = self.store.snapshot(next(iter(self.store.state()["captures"])))
        self.assertEqual([row["content"] for row in snapshot["messages"]],
                         ["Older native user.", "Native current text."])
        self.assertEqual(snapshot["native_history"], event["extra"]["conversation_history"])
        self.assertNotIn("injected API sidecar", json.dumps(snapshot["messages"]))

    def test_repeated_history_text_retains_both_native_rows(self) -> None:
        current = payload()["extra"]["user_message"]
        event = payload(history=[{"role": "user", "content": current},
                                 {"role": "assistant", "content": "Earlier answer."},
                                 {"role": "user", "content": current}])
        self.store.capture(event, "pre_llm_call")
        snapshot = self.store.snapshot(next(iter(self.store.state()["captures"])))
        self.assertEqual(len(snapshot["messages"]), 3)
        self.assertEqual(sum(row["content"] == current for row in snapshot["messages"]), 2)

    def test_post_full_native_assistant_capture_and_end_flush(self) -> None:
        event = payload(post=True)
        event["extra"]["assistant_response"] = "Synthetic assistant " + "a" * 5000
        self.store.capture(event, "post_llm_call")
        name = next(iter(self.store.state()["captures"]))
        before = (self.store.path / name).read_bytes()
        snapshot = self.store.snapshot(name)
        self.assertEqual(snapshot["messages"][-1]["content"], event["extra"]["assistant_response"])
        self.store.flush()
        self.assertTrue(self.store.state()["captures"][name]["flush"])
        self.assertEqual((self.store.path / name).read_bytes(), before)

    def test_malformed_supported_text_and_nonlist_history_rejected(self) -> None:
        for value in (None, {"text": "wrong"}, [{"type": "text", "text": 7}], [{"type": "unknown"}]):
            event = payload()
            event["extra"]["user_message"] = value
            with self.assertRaises(BRIDGE.BridgeError):
                self.store.capture(event, "pre_llm_call")
        event = payload()
        event["extra"]["conversation_history"] = {}
        with self.assertRaises(BRIDGE.BridgeError):
            self.store.capture(event, "pre_llm_call")
        self.assertEqual(self.store.state()["captures"], {})

    def test_empty_current_text_does_not_recall_historical_input(self) -> None:
        event = payload(history=[{"role": "user", "content": "Historical recall question?"}])
        event["extra"]["user_message"] = ""
        self.store.capture(event, "pre_llm_call")
        snapshot = self.store.snapshot(next(iter(self.store.state()["captures"])))
        session_start = types.ModuleType("truememory.ingest.hooks.session_start")
        prompt = types.ModuleType("truememory.ingest.hooks.user_prompt_submit")
        with collaborators({session_start.__name__: session_start, prompt.__name__: prompt}):
            self.assertEqual(BRIDGE._recall(snapshot, 1), "")
        self.assertEqual(snapshot["messages"][-1]["content"], "")

    def test_input_is_bounded_malformed_numbers_and_roots_rejected(self) -> None:
        with patch.object(BRIDGE, "MAX_INPUT_BYTES", 20):
            with self.assertRaises(BRIDGE.BridgeError):
                BRIDGE._input(io.StringIO("x" * 21))
            with self.assertRaises(BRIDGE.BridgeError):
                BRIDGE._input(io.StringIO('"' + "é" * 11 + '"'))
        for text in ("[]", "null", '{"x":NaN}'):
            with self.assertRaises(BRIDGE.BridgeError):
                BRIDGE._input(io.StringIO(text))

    def test_quota_aggregate_retains_accepted_bytes_and_releases_locks(self) -> None:
        self.store.capture(payload(), "pre_llm_call")
        before = {path: path.read_bytes() for path in self.store.path.iterdir() if path.suffix == ".json"}
        second = BRIDGE.Store(self.store.root, self.store.user, self.store.db, "other-session")
        size, _count = self.store.usage()
        with patch.object(BRIDGE, "MAX_SCOPE_BYTES", size):
            with self.assertRaises(BRIDGE.BridgeError):
                second.capture(payload(session="other-session"), "pre_llm_call")
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        with BRIDGE._lock(self.store.root / "quota.lock") as allowed:
            self.assertTrue(allowed)
        with BRIDGE._lock(second.capture_lock) as allowed:
            self.assertTrue(allowed)
        self.assertEqual(second.state()["captures"], {})

    def test_context_status_quota_is_also_bounded(self) -> None:
        before = self.store.usage()[0]
        with patch.object(BRIDGE, "MAX_SCOPE_BYTES", before + 10):
            with self.assertRaises(BRIDGE.BridgeError):
                self.store.write(self.store.path / "result.json", {"context": "c" * 100})
        self.assertFalse((self.store.path / "result.json").exists())

    def test_owned_snapshot_hash_and_symlink_mismatch_rejected(self) -> None:
        self.store.capture(payload(), "pre_llm_call")
        name = next(iter(self.store.state()["captures"]))
        wrong = self.store.path / ("0" * 64 + ".json")
        wrong.write_bytes((self.store.path / name).read_bytes())
        with self.assertRaises(BRIDGE.BridgeError):
            self.store.snapshot(wrong.name)
        link = self.store.path / "link.json"
        link.symlink_to(self.store.path / name)
        with self.assertRaises(BRIDGE.BridgeError):
            BRIDGE._read_owned(link)
        with self.assertRaises(BRIDGE.BridgeError):
            self.store.snapshot("../../foreign.json")

    def test_private_directory_and_lock_symlinks_rejected(self) -> None:
        foreign = self.case / "foreign"
        foreign.mkdir()
        link = self.case / "linked-root"
        link.symlink_to(foreign, target_is_directory=True)
        with self.assertRaises(BRIDGE.BridgeError):
            BRIDGE.Store(link, "u", "db", "s")
        self.store.worker_lock.symlink_to(self.case / "foreign-lock")
        with self.assertRaises(BRIDGE.BridgeError):
            with BRIDGE._lock(self.store.worker_lock):
                self.fail("symlink lock admitted")

    def test_two_os_lock_owners_cannot_take_over_live_lock(self) -> None:
        lock = self.store.worker_lock
        with BRIDGE._lock(lock) as first:
            self.assertTrue(first)
            with BRIDGE._lock(lock) as second:
                self.assertFalse(second)
        with BRIDGE._lock(lock) as after_release:
            self.assertTrue(after_release)

    def test_child_process_cannot_reclaim_live_os_lock(self) -> None:
        source = ROOT / "truememory/ingest/hooks/hermes.py"
        code = ("import sys,types,pathlib; p=pathlib.Path(sys.argv[1]); "
                "m=types.ModuleType('lock765');m.__file__=str(p);sys.modules[m.__name__]=m;"
                "exec(compile(p.read_text(),str(p),'exec'),m.__dict__); "
                "lock=m._lock(pathlib.Path(sys.argv[2])); admitted=lock.__enter__(); "
                "lock.__exit__(None,None,None);sys.exit(7 if admitted else 0)")
        with BRIDGE._lock(self.store.worker_lock) as held:
            self.assertTrue(held)
            result = subprocess.run([sys.executable, "-B", "-c", code, str(source),
                                     str(self.store.worker_lock)], capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_simultaneous_sessions_share_account_quota(self) -> None:
        second = BRIDGE.Store(self.store.root, self.store.user, self.store.db, "other-session")
        def admit(store: BRIDGE.Store) -> bool:
            event = payload(session=store.session)
            event["extra"]["user_message"] = "Synthetic quota content " + "x" * 4000
            try:
                store.capture(event, "pre_llm_call")
                return True
            except BRIDGE.BridgeError:
                return False
        with patch.object(BRIDGE, "MAX_SCOPE_BYTES", 12000), ThreadPoolExecutor(max_workers=2) as pool:
            accepted = list(pool.map(admit, (self.store, second)))
        self.assertEqual(sum(accepted), 1)
        self.assertLessEqual(self.store.usage()[0], 12000)
        self.assertEqual(sum(len(store.state()["captures"]) for store in (self.store, second)), 1)

    def test_windows_lock_branch_and_release_selected_source(self) -> None:
        calls: list[tuple[int, int]] = []
        primitive = types.ModuleType("msvcrt")
        primitive.LK_NBLCK, primitive.LK_UNLCK = 11, 12
        primitive.locking = lambda _fd, op, count: calls.append((op, count))
        with patch.dict(sys.modules, {"msvcrt": primitive}), patch.object(sys, "platform", "win32"):
            with BRIDGE._lock(self.store.path / "windows.lock") as allowed:
                self.assertTrue(allowed)
        self.assertEqual(calls, [(11, 1), (12, 1)])

    def test_birth_mismatch_dead_and_access_denied_are_conservative(self) -> None:
        with patch.object(BRIDGE, "_birth", return_value=12.0):
            self.assertTrue(BRIDGE._alive({"pid": 123, "born": 12.0}))
            self.assertFalse(BRIDGE._alive({"pid": 123, "born": 11.0}))
        with patch.object(BRIDGE, "_birth", return_value=None):
            self.assertFalse(BRIDGE._alive({"pid": 123, "born": 12.0}))
        with patch.object(BRIDGE, "_birth", side_effect=PermissionError):
            self.assertTrue(BRIDGE._alive({"pid": 123, "born": 12.0}))

    def ready_context(self, context: str = "Synthetic recall") -> str:
        key = self.store.capture(payload(), "pre_llm_call")
        state = self.store.state()
        state["turns"][key]["context_state"] = "ready"
        self.store.write(self.store.path / f"{key}.context.json",
                         {"account": self.store.account, "turn": key, "context": context,
                          "sha256": BRIDGE._digest(context)})
        self.store.save(state)
        return key

    def test_context_native_output_and_full_flush_marker(self) -> None:
        key = self.ready_context()
        output = io.StringIO()
        self.assertTrue(BRIDGE._emit_context(self.store, key, output))
        self.assertEqual(json.loads(output.getvalue()), {"context": "Synthetic recall"})
        self.assertEqual(self.store.state()["turns"][key]["context_state"], "written")

    def test_failed_context_write_and_flush_preserve_ready_cache(self) -> None:
        key = self.ready_context()
        class BrokenWriter(io.StringIO):
            def flush(self) -> None:
                raise BrokenPipeError("synthetic failed flush")
        with self.assertRaises(BrokenPipeError):
            BRIDGE._emit_context(self.store, key, BrokenWriter())
        self.assertEqual(self.store.state()["turns"][key]["context_state"], "ready")
        self.assertTrue((self.store.path / f"{key}.context.json").is_file())
        class ShortWriter(io.StringIO):
            def write(self, text: str) -> int:
                return len(text) - 1
        with self.assertRaises(BRIDGE.BridgeError):
            BRIDGE._emit_context(self.store, key, ShortWriter())
        self.assertEqual(self.store.state()["turns"][key]["context_state"], "ready")

    def pipeline_fixture(self, status: str = "complete", stored: int = 1,
                         *, missing_api: bool = False, throw: bool = False,
                         heuristic: bool = False, empty: bool = False
                         ) -> tuple[dict[str, types.ModuleType], dict[str, object],
                                    list[tuple[tuple[object, ...], dict[str, object]]],
                                    types.ModuleType, object, object]:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        extractor = types.ModuleType("truememory.ingest.extractor")
        fact = object()
        fact_list = [] if empty else [fact]
        def outcome(*args: object, **kwargs: object) -> object:
            calls.append((args, kwargs))
            return types.SimpleNamespace(facts=fact_list, complete=status == "complete", status=status)
        if not missing_api:
            extractor.extract_facts_outcome = outcome
        pipeline = types.ModuleType("truememory.ingest.pipeline")
        def original(*_args: object, **_kwargs: object) -> list[object]:
            return fact_list
        pipeline.extract_facts = original
        pipeline.extract_facts_simple = lambda _text: fact_list
        config = object()
        class Pipeline:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
            def ingest_text(self, text: str, *, session_id: str) -> object:
                facts = pipeline.extract_facts_simple(text) if heuristic else pipeline.extract_facts(text, config)
                if throw:
                    raise RuntimeError("synthetic pipeline failure")
                return types.SimpleNamespace(facts_extracted=len(facts), facts_stored=stored,
                                              facts_updated=0, facts_skipped_gate=0,
                                              facts_skipped_dedup=0, facts_encoded=len(facts))
        pipeline.IngestionPipeline = Pipeline
        stop = types.ModuleType("truememory.ingest.hooks.stop")
        stop.GATE_THRESHOLD, stop.MIN_MESSAGES = 0.30, 5
        children = {extractor.__name__: extractor, pipeline.__name__: pipeline,
                    "truememory.ingest.transcript": TRANSCRIPT, stop.__name__: stop}
        snapshot = {"messages": [{"role": "human", "content": "Synthetic long user fact " * 3}]
                                + [{"role": "assistant", "content": "Synthetic answer"}] * 4,
                    "user": self.store.user, "db": self.store.db, "session_id": self.store.session}
        return children, snapshot, calls, pipeline, original, config

    def test_exact_single_outcome_default_call_and_terminal_accounting(self) -> None:
        children, snapshot, calls, pipeline, original, config = self.pipeline_fixture()
        with collaborators(children):
            result = BRIDGE._run_pipeline(snapshot)
        self.assertEqual(result["state"], "extraction_complete")
        self.assertEqual(result["extraction_status"], "complete")
        self.assertTrue(result["terminal_accounting_complete"])
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0][1], config)
        self.assertEqual(calls[0][1], {})
        self.assertIs(pipeline.extract_facts, original)

    def test_partial_limited_failed_and_storage_gap_never_complete(self) -> None:
        for status, stored in (("partial", 1), ("limited", 1), ("failed", 1), ("complete", 0)):
            children, snapshot, calls, pipeline, original, _config = self.pipeline_fixture(status, stored)
            with collaborators(children):
                result = BRIDGE._run_pipeline(snapshot)
            self.assertEqual(result["state"], "worker_returned")
            self.assertEqual(result["extraction_status"], status)
            self.assertEqual(len(calls), 1)
            self.assertIs(pipeline.extract_facts, original)

    def test_missing_outcome_api_retains_original_pipeline_and_unproven_coverage(self) -> None:
        children, snapshot, calls, pipeline, original, _config = self.pipeline_fixture(missing_api=True)
        with collaborators(children):
            result = BRIDGE._run_pipeline(snapshot)
        self.assertEqual(result["state"], "worker_returned")
        self.assertEqual(result["extraction_status"], "coverage_unproven")
        self.assertEqual(calls, [])
        self.assertIs(pipeline.extract_facts, original)

    def test_heuristic_return_cannot_invent_complete_extraction_coverage(self) -> None:
        children, snapshot, calls, pipeline, original, _config = self.pipeline_fixture(heuristic=True)
        with collaborators(children):
            result = BRIDGE._run_pipeline(snapshot)
        self.assertEqual(result["state"], "worker_returned")
        self.assertEqual(result["extraction_status"], "heuristic_returned")
        self.assertTrue(result["terminal_accounting_complete"])
        self.assertEqual(calls, [])
        self.assertIs(pipeline.extract_facts, original)

    def test_complete_empty_outcome_distinguished_from_failed_empty(self) -> None:
        for status, expected in (("complete", "extraction_complete"), ("failed", "worker_returned")):
            children, snapshot, calls, _pipeline, _original, _config = self.pipeline_fixture(status, 0, empty=True)
            with collaborators(children):
                result = BRIDGE._run_pipeline(snapshot)
            self.assertEqual(result["state"], expected)
            self.assertEqual(result["counts"]["facts_extracted"], 0)
            self.assertEqual(len(calls), 1)

    def test_outcome_binding_restored_on_exception_and_original_threshold(self) -> None:
        children, snapshot, calls, pipeline, original, _config = self.pipeline_fixture(throw=True)
        with collaborators(children):
            with self.assertRaises(RuntimeError):
                BRIDGE._run_pipeline(snapshot)
        self.assertEqual(len(calls), 1)
        self.assertIs(pipeline.extract_facts, original)
        snapshot["messages"] = snapshot["messages"][:4]
        with collaborators(children):
            self.assertEqual(BRIDGE._run_pipeline(snapshot)["state"], "deferred")
        self.assertEqual(len(calls), 1)

    def test_worker_success_releases_settled_snapshot_and_keeps_distinct_states(self) -> None:
        key = self.store.capture(payload(), "pre_llm_call")
        self.store.flush()
        state = self.store.state()
        state["worker"] = {"token": "synthetic-token", "state": "reserved"}
        self.store.save(state)
        captures = {name: (self.store.path / name).read_bytes() for name in state["captures"]}
        with collaborators({"truememory.ingest.hooks._shared": inert_shared()}), \
                patch.object(BRIDGE, "_owner", return_value={"pid": 99, "born": 12}), \
                patch.object(BRIDGE, "_recall", return_value="Synthetic bounded context"), \
                patch.object(BRIDGE, "_run_pipeline", return_value={"state": "extraction_complete"}):
            BRIDGE._worker(self.store, "synthetic-token")
        current = self.store.state()
        self.assertEqual(current["turns"][key]["context_state"], "ready")
        self.assertEqual(current["worker"]["state"], "returned")
        self.assertTrue(all(item["state"] == "extraction_complete" for item in current["captures"].values()))
        # Settled entries stay as evidence; their snapshot bytes are released.
        self.assertTrue(captures)
        self.assertTrue(all(current["captures"][name]["released"] for name in captures))
        self.assertFalse(any((self.store.path / name).exists() for name in captures))
        self.assertTrue((self.store.path / f"{key}.context.json").is_file())
        self.assertTrue(all(item["attempts"] == 1 for item in current["captures"].values()))

    def test_worker_failure_retry_is_bounded_and_releases_only_when_exhausted(self) -> None:
        self.store.capture(payload(post=True), "post_llm_call")
        before = next(iter(self.store.state()["captures"]))
        raw = (self.store.path / before).read_bytes()
        shared = inert_shared()
        with collaborators({shared.__name__: shared}), \
                patch.object(BRIDGE, "_owner", return_value={"pid": 99, "born": 12}), \
                patch.object(BRIDGE, "_run_pipeline", side_effect=RuntimeError("synthetic worker failure")):
            for attempt in range(BRIDGE.MAX_ATTEMPTS):
                state = self.store.state()
                state["worker"] = {"token": "synthetic-token", "state": "reserved"}
                state["captures"][before]["retry_at"] = 0
                self.store.save(state)
                BRIDGE._worker(self.store, "synthetic-token")
                self.assertEqual(self.store.state()["captures"][before]["attempts"], attempt + 1)
                if attempt + 1 < BRIDGE.MAX_ATTEMPTS:
                    self.assertEqual((self.store.path / before).read_bytes(), raw)
        state = self.store.state()
        self.assertEqual(state["captures"][before]["state"], "failed")
        self.assertGreater(state["captures"][before]["retry_at"], 0)
        self.assertFalse(BRIDGE._due(state["captures"][before]))
        self.assertTrue(state["captures"][before]["released"])
        self.assertFalse((self.store.path / before).exists())

    def test_uncertain_attempt_can_retry_to_observed_complete(self) -> None:
        self.store.capture(payload(post=True), "post_llm_call")
        name = next(iter(self.store.state()["captures"]))
        raw = (self.store.path / name).read_bytes()
        results = [{"state": "worker_returned", "extraction_status": "partial"},
                   {"state": "extraction_complete", "extraction_status": "complete"}]
        shared = inert_shared()
        with collaborators({shared.__name__: shared}), patch.object(BRIDGE, "_owner", return_value={}), \
                patch.object(BRIDGE, "_run_pipeline", side_effect=results):
            for attempt in range(2):
                state = self.store.state()
                state["worker"] = {"token": "synthetic-token", "state": "reserved"}
                state["captures"][name]["retry_at"] = 0
                self.store.save(state)
                BRIDGE._worker(self.store, "synthetic-token")
                self.assertEqual(self.store.state()["captures"][name]["attempts"], attempt + 1)
                if attempt == 0:
                    self.assertEqual((self.store.path / name).read_bytes(), raw)
        self.assertEqual(self.store.state()["captures"][name]["state"], "extraction_complete")
        self.assertFalse((self.store.path / name).exists())

    def test_intake_during_worker_is_retained_and_processed_with_separate_budget(self) -> None:
        self.store.capture(payload(post=True), "post_llm_call")
        state = self.store.state()
        state["worker"] = {"token": "synthetic-token", "state": "reserved"}
        self.store.save(state)
        calls: list[str] = []
        budgets: list[bool] = []
        shared = inert_shared()
        shared.check_extraction_budget = lambda: budgets.append(True) or True
        def ingest(snapshot: dict[str, object]) -> dict[str, str]:
            calls.append(snapshot["turn_id"])
            if len(calls) == 1:
                self.store.capture(payload(turn="late-turn", post=True), "post_llm_call")
            return {"state": "extraction_complete"}
        with collaborators({shared.__name__: shared}), patch.object(BRIDGE, "_owner", return_value={}), \
                patch.object(BRIDGE, "_run_pipeline", side_effect=ingest):
            BRIDGE._worker(self.store, "synthetic-token")
        self.assertEqual(calls, ["synthetic-turn", "late-turn"])
        self.assertEqual(len(budgets), 1)
        self.assertEqual(len(self.store.state()["captures"]), 2)
        self.assertTrue(all(item["state"] == "extraction_complete"
                            for item in self.store.state()["captures"].values()))

    def test_worker_oversize_context_is_failed_not_written(self) -> None:
        key = self.store.capture(payload(), "pre_llm_call")
        state = self.store.state()
        state["worker"] = {"token": "synthetic-token", "state": "reserved"}
        self.store.save(state)
        shared = inert_shared()
        with collaborators({shared.__name__: shared}), patch.object(BRIDGE, "_owner", return_value={}), \
                patch.object(BRIDGE, "_recall", return_value="x" * (BRIDGE.MAX_CONTEXT_BYTES + 1)):
            BRIDGE._worker(self.store, "synthetic-token")
        self.assertEqual(self.store.state()["turns"][key]["context_state"], "failed")
        self.assertEqual(self.store.state()["turns"][key]["context_attempts"], 1)
        self.assertFalse((self.store.path / f"{key}.context.json").exists())

    def test_context_timeout_preserves_pending_capture(self) -> None:
        key = self.store.capture(payload(), "pre_llm_call")
        output = io.StringIO()
        with patch.object(BRIDGE, "CONTEXT_WAIT_SECONDS", 0):
            self.assertFalse(BRIDGE._emit_context(self.store, key, output))
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(self.store.state()["turns"][key]["context_state"], "pending")
        self.assertEqual(len(self.store.state()["captures"]), 1)

    def test_first_turn_session_context_later_targeted_recall_has_no_legacy_side_effects(self) -> None:
        self.store.capture(payload(), "pre_llm_call")
        snapshot = self.store.snapshot(next(iter(self.store.state()["captures"])))
        calls: list[tuple[str, object]] = []
        session = types.ModuleType("truememory.ingest.hooks.session_start")
        session._intensity_budget = lambda _intensity: 8192
        session.recall_memories = lambda _input, **kwargs: calls.append(("first", kwargs)) or "first-context"
        prompt = types.ModuleType("truememory.ingest.hooks.user_prompt_submit")
        prompt._get_intensity_config = lambda: ("standard", "max")
        prompt._try_proactive_recall = lambda *args, **kwargs: (None, False)
        prompt._try_auto_recall = lambda *args, **kwargs: calls.append(("later", args)) or "later-context"
        with collaborators({session.__name__: session, prompt.__name__: prompt}):
            self.assertEqual(BRIDGE._recall(snapshot, 1), "first-context")
            self.assertEqual(BRIDGE._recall(snapshot, 2), "later-context")
        self.assertEqual(calls[0][1]["user_id"], self.store.user)
        self.assertEqual(calls[0][1]["db_path"], self.store.db)
        self.assertEqual(calls[0][1]["budget"], 8192)
        self.assertEqual(calls[1][1][0], snapshot["native_user_message"])

    def spawn_collaborators(self, allowed: bool = True) -> tuple[dict[str, types.ModuleType], list[str]]:
        events: list[str] = []
        core = types.ModuleType("truememory.hooks.core")
        @contextmanager
        def gate() -> Iterator[bool]:
            events.append("gate")
            yield allowed
        core.spawn_gate = gate
        core.register_spawned_pid = lambda _pid: events.append("register")
        shared = inert_shared()
        shared.check_extraction_budget = lambda: events.append("budget") or True
        shared.refund_extraction_budget = lambda: events.append("refund")
        return {core.__name__: core, shared.__name__: shared}, events

    def test_spawn_cap_and_launch_failure_preserve_captured_retry_state(self) -> None:
        self.store.capture(payload(), "pre_llm_call")
        children, events = self.spawn_collaborators(allowed=False)
        with collaborators(children), patch.object(BRIDGE, "_birth", return_value=12):
            BRIDGE._spawn(self.store)
        self.assertEqual(events, ["budget", "gate", "refund"])
        self.assertEqual(self.store.state()["worker"], {})
        children, events = self.spawn_collaborators()
        with collaborators(children), patch.object(BRIDGE, "_birth", return_value=12), \
                patch.object(BRIDGE.subprocess, "Popen", side_effect=OSError("synthetic spawn failure")):
            with self.assertRaises(OSError):
                BRIDGE._spawn(self.store)
        self.assertEqual(events, ["budget", "gate", "refund"])
        self.assertEqual(self.store.state()["worker"]["state"], "spawn_failed")
        self.assertTrue(BRIDGE._work_available(self.store.state()))

    def test_spawn_live_lock_and_live_birth_deny_without_burning_budget(self) -> None:
        self.store.capture(payload(), "pre_llm_call")
        children, events = self.spawn_collaborators()
        with collaborators(children), BRIDGE._lock(self.store.worker_lock) as held:
            self.assertTrue(held)
            BRIDGE._spawn(self.store)
        self.assertEqual(events, [])
        state = self.store.state()
        state["worker"] = {"state": "reserved", "child": {"pid": 99, "born": 12}, "launcher": None}
        self.store.save(state)
        with collaborators(children), patch.object(BRIDGE, "_birth", return_value=12):
            BRIDGE._spawn(self.store)
        self.assertEqual(events, [])
        self.assertEqual(self.store.state()["worker"]["state"], "reserved")

    def test_spawn_reused_pid_uses_fixed_worker_arguments_and_account(self) -> None:
        self.store.capture(payload(), "pre_llm_call")
        state = self.store.state()
        state["worker"] = {"state": "reserved", "child": {"pid": 99, "born": 11}, "launcher": None}
        self.store.save(state)
        children, events = self.spawn_collaborators()
        with collaborators(children), patch.object(BRIDGE, "_birth", return_value=12), \
                patch.object(BRIDGE.subprocess, "Popen", return_value=types.SimpleNamespace(pid=123)) as child:
            BRIDGE._spawn(self.store)
        argv = child.call_args.args[0]
        self.assertEqual(argv[2:4], ["--worker", self.store.key])
        self.assertIn(self.store.user, argv)
        self.assertIn(self.store.db, argv)
        self.assertEqual(events, ["budget", "gate", "register"])
        self.assertEqual(self.store.state()["worker"]["child"], {"pid": 123, "born": 12})
        self.assertEqual(len(self.store.state()["captures"]), 1)

    def test_main_argv_ignores_stdin_event_account_paths_and_finalize_uses_owned_capture(self) -> None:
        event = payload()
        with patch.dict(os.environ, {"TRUEMEMORY_HERMES_SPOOL_DIR": str(self.store.root)}), \
                patch.object(sys, "argv", ["hermes.py", "--event", "pre_llm_call", "--user", self.store.user,
                                          "--db", self.store.db]), \
                patch.object(sys, "stdin", io.StringIO(json.dumps(event))), \
                patch.object(BRIDGE, "_spawn") as spawn, patch.object(BRIDGE, "_emit_context") as emit:
            self.assertEqual(BRIDGE.main(), 0)
        self.assertEqual(spawn.call_args.args[0].account, self.store.account)
        self.assertEqual(emit.call_args.args[0].session, self.store.session)
        self.assertEqual(len(self.store.state()["captures"]), 1)
        with patch.dict(os.environ, {"TRUEMEMORY_HERMES_SPOOL_DIR": str(self.store.root)}), \
                patch.object(sys, "argv", ["hermes.py", "--event", "on_session_finalize", "--user", self.store.user,
                                          "--db", self.store.db]), \
                patch.object(sys, "stdin", io.StringIO(json.dumps({"session_id": self.store.session,
                                                                  "transcript_path": "/never/read"}))), \
                patch.object(BRIDGE, "_spawn"), patch.object(BRIDGE, "_emit_context") as emit:
            self.assertEqual(BRIDGE.main(), 0)
        self.assertFalse(emit.called)
        self.assertTrue(all(item["flush"] for item in self.store.state()["captures"].values()))

    def test_crash_claim_durable_attempt_and_live_lock_no_reclaim(self) -> None:
        self.store.capture(payload(post=True), "post_llm_call")
        state = self.store.state()
        state["worker"] = {"token": "synthetic-token", "state": "reserved"}
        self.store.save(state)
        shared = inert_shared()
        with collaborators({shared.__name__: shared}), patch.object(BRIDGE, "_owner", return_value={}):
            with BRIDGE._lock(self.store.worker_lock) as held:
                self.assertTrue(held)
                BRIDGE._worker(self.store, "synthetic-token")
                self.assertEqual(self.store.state()["worker"]["state"], "reserved")
            with patch.object(BRIDGE, "_run_pipeline", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    BRIDGE._worker(self.store, "synthetic-token")
        item = next(iter(self.store.state()["captures"].values()))
        self.assertEqual(item["state"], "processing")
        self.assertEqual(item["attempts"], 1)
        with BRIDGE._lock(self.store.worker_lock) as released:
            self.assertTrue(released)

    def test_owned_command_migration_preserves_foreign_entries(self) -> None:
        source = ROOT / "truememory/hooks/adapters/hermes.py"
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom)
                     or not (node.module or "").startswith("truememory")]
        namespace = {"__file__": str(source), "__name__": "synthetic_adapter765", "CLIAdapter": object,
                     "atomic_write_text": lambda path, text: path.write_text(text)}
        exec(compile(tree, str(source), "exec"), namespace)
        config = self.case / "config.json"
        namespace.update(_CONFIG=config, _yaml_safe_load=json.loads, _yaml_safe_dump=json.dumps)
        adapter = namespace["HermesAdapter"]()
        hooks = ROOT / "truememory/ingest/hooks"
        legacy = adapter._build_command(sys.executable, hooks / "user_prompt_submit.py", "old-user", "old-db")
        foreign = [{"command": "echo truememory"}, {"command": "helper --label truememory"},
                   {"command": legacy + " # foreign comment"}, {"command": legacy + " ; foreign"},
                   {"command": adapter._build_command(sys.executable, hooks / "stop.py")}]
        config.write_text(json.dumps({"hooks": {"pre_llm_call": [*foreign, {"command": legacy}]}}))
        adapter.install_hooks(sys.executable, "new-user", "new-db")
        installed = json.loads(config.read_text())
        entries = installed["hooks"]["pre_llm_call"]
        self.assertEqual(entries[:-1], foreign)
        self.assertIn("hermes.py", entries[-1]["command"])
        self.assertIn("--event pre_llm_call", entries[-1]["command"])
        self.assertIn("new-user", entries[-1]["command"])
        self.assertIn("new-db", entries[-1]["command"])
        first = config.read_bytes()
        adapter.install_hooks(sys.executable, "new-user", "new-db")
        self.assertEqual(config.read_bytes(), first)
        self.assertFalse(adapter.has_session_start)
        adapter._remove_hook_entries()
        self.assertEqual(json.loads(config.read_text())["hooks"]["pre_llm_call"], foreign)


if __name__ == "__main__":
    unittest.main()
