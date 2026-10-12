"""Synthetic observed-byte completion controls; filesystem cases run on GPUBox."""
from __future__ import annotations

import ast
import contextlib
import datetime
import enum
import hashlib
import importlib.util
import json
import logging
import os
import sqlite3
import stat
import sys
import tempfile
import time
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ingestion_completion_768_controls", ROOT / "truememory/ingest/completion.py")
assert SPEC is not None and SPEC.loader is not None
COMPLETION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPLETION
SPEC.loader.exec_module(COMPLETION)


def result(action: str | None = None, *, parser: str = "complete", extractor: str = "complete") -> types.SimpleNamespace:
    value = types.SimpleNamespace(facts_extracted=int(action is not None), facts_stored=int(action == "stored"),
        facts_updated=int(action == "updated"), facts_skipped_gate=int(action == "skipped_gate"),
        facts_skipped_dedup=int(action == "skipped_dedup"), facts_provenance_updated=0,
        trace=[] if action is None else [{"action": action}],
        coverage={"parser_status": parser, "extractor_status": extractor, "extractor_calls": 1,
                  "stable": True, "provenance_observed": True,
                  "parser_errors": [], "extractor_errors": [],
                  "malformed_records": 0, "unrecognized_records": 0,
                  "source_version": {"sha256": "0" * 64, "byte_count": 7}})
    return value


class InertCompletionControls(unittest.TestCase):
    def test_rich_empty_and_exact_terminal_partitions(self) -> None:
        for action in (None, "stored", "updated", "skipped_gate", "skipped_dedup"):
            self.assertTrue(COMPLETION.strict_complete(result(action)))
        metadata_only = result("skipped_dedup")
        metadata_only.facts_provenance_updated = 1
        self.assertTrue(COMPLETION.strict_complete(metadata_only))

    def test_partial_caps_heuristic_and_absent_outcomes_are_not_terminal(self) -> None:
        for status in ("partial", "salvaged", "malformed", "unreadable", "unknown"):
            self.assertFalse(COMPLETION.strict_complete(result(parser=status)))
        for status in ("partial", "limited", "failed", "heuristic", "unobserved", "unknown"):
            self.assertFalse(COMPLETION.strict_complete(result(extractor=status)))
        self.assertFalse(COMPLETION.strict_complete(types.SimpleNamespace()))

    def test_unknown_storage_and_inconsistent_accounting_do_not_complete(self) -> None:
        for action in ("storage_failed", "unknown", "stored"):
            value = result(action)
            value.trace[0]["storage_error"] = {"exception": "OperationalError"}
            self.assertFalse(COMPLETION.strict_complete(value))
        value = result("stored")
        value.facts_extracted = 2
        self.assertFalse(COMPLETION.strict_complete(value))
        value = result("stored")
        value.trace = [{"action": "updated"}]
        self.assertFalse(COMPLETION.strict_complete(value))
        value = result("stored")
        value.trace.append({"action": "stored"})
        self.assertFalse(COMPLETION.strict_complete(value))

    def test_bool_negative_and_missing_provenance_proof_are_rejected(self) -> None:
        for field_name in ("facts_extracted", "facts_stored", "facts_updated", "facts_skipped_gate", "facts_skipped_dedup"):
            value = result()
            setattr(value, field_name, True)
            self.assertFalse(COMPLETION.strict_complete(value))
            setattr(value, field_name, -1)
            self.assertFalse(COMPLETION.strict_complete(value))
        for field_name, replacement in (("extractor_calls", True), ("extractor_calls", 0),
                                         ("stable", False), ("provenance_observed", False)):
            value = result()
            value.coverage[field_name] = replacement
            self.assertFalse(COMPLETION.strict_complete(value))

    def test_complete_protocol_requires_typed_record_counts_and_errors(self) -> None:
        for name in ("malformed_records", "unrecognized_records"):
            for replacement in (False, None, "0", 1):
                value = result()
                value.coverage[name] = replacement
                self.assertFalse(COMPLETION.strict_complete(value))
            value = result()
            del value.coverage[name]
            self.assertFalse(COMPLETION.strict_complete(value))
        for name in ("parser_errors", "extractor_errors"):
            for replacement in (None, (), "", ["synthetic_partial"]):
                value = result()
                value.coverage[name] = replacement
                self.assertFalse(COMPLETION.strict_complete(value))
            value = result()
            del value.coverage[name]
            self.assertFalse(COMPLETION.strict_complete(value))

    def test_receipt_revalidates_entire_rich_completion_proof(self) -> None:
        capture = COMPLETION.Capture(Path("/synthetic"), "1" * 64,
            Path("/synthetic/capture.jsonl"), {"sha256": "0" * 64, "byte_count": 7})
        def receipt_value():
            value = result("stored")
            return {"version": 1, "capture_key": capture.key, "scope": capture.metadata,
                "coverage": "observed_bytes", "parser_status": "complete", "extractor_status": "complete",
                "proof": {"coverage": value.coverage, "counts": {name: getattr(value, name) for name in COMPLETION._COUNTS},
                          "actions": ["stored"]}}
        def observed(value):
            with patch.object(Path, "exists", return_value=True), \
                    patch.object(COMPLETION, "_read_private", return_value=json.dumps(value).encode()):
                return COMPLETION.has_receipt(capture)
        self.assertTrue(observed(receipt_value()))
        for name in ("parser_status", "extractor_status", "proof"):
            value = receipt_value()
            del value[name]
            self.assertFalse(observed(value))
        for name, replacement in (("parser_status", "partial"), ("extractor_status", "failed"),
                                   ("version", True), ("scope", {})):
            value = receipt_value()
            value[name] = replacement
            self.assertFalse(observed(value))
        for name, replacement in (("parser_status", "partial"), ("extractor_status", "failed"),
                                   ("provenance_observed", False), ("parser_errors", ["synthetic"]),
                                   ("malformed_records", True)):
            value = receipt_value()
            value["proof"]["coverage"][name] = replacement
            self.assertFalse(observed(value))
        value = receipt_value()
        value["proof"]["counts"]["facts_stored"] = 0
        self.assertFalse(observed(value))
        value = receipt_value()
        value["proof"]["actions"] = ["storage_failed"]
        self.assertFalse(observed(value))

    def test_capture_inventory_stops_at_first_excess_entry(self) -> None:
        visited = []
        def entries():
            for index in range(5005):
                visited.append(index)
                yield types.SimpleNamespace(path="/synthetic/capture-" + str(index))
        with patch.object(COMPLETION.os, "scandir", return_value=contextlib.nullcontext(entries())), \
                patch.object(COMPLETION, "_open_private", return_value=41) as opened, \
                patch.object(COMPLETION.os, "fstat", return_value=types.SimpleNamespace(st_size=1)), \
                patch.object(COMPLETION.os, "close"):
            with self.assertRaisesRegex(COMPLETION.CompletionError, "capture_count_limit"):
                COMPLETION._capture_inventory(Path("/synthetic/captures"))
        self.assertEqual(len(visited), COMPLETION.MAX_CAPTURES + 1)
        self.assertEqual(opened.call_count, COMPLETION.MAX_CAPTURES)

    def test_source_ownership_requires_regular_current_account_or_fails_closed(self) -> None:
        info = types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=1001)
        with patch.object(COMPLETION, "os", types.SimpleNamespace(name="posix", getuid=lambda: 1001)):
            COMPLETION._source_owned(info, 41)
            info.st_uid = 1002
            with self.assertRaisesRegex(COMPLETION.CompletionError, "source_not_owned"):
                COMPLETION._source_owned(info, 41)
        info.st_mode = stat.S_IFDIR | 0o700
        with self.assertRaisesRegex(COMPLETION.CompletionError, "source_not_regular"):
            COMPLETION._source_owned(info, 41)
        info.st_mode = stat.S_IFREG | 0o600
        with patch.object(COMPLETION, "os", types.SimpleNamespace(name="unverifiable")):
            with self.assertRaisesRegex(COMPLETION.CompletionError, "source_ownership_unavailable"):
                COMPLETION._source_owned(info, 41)

    def test_legacy_pointer_inventory_never_inspects_over_limit_or_retires(self) -> None:
        shared = types.ModuleType("truememory.ingest.hooks._shared")
        shared.backlog_transcript_source = lambda *args, **kwargs: None
        platform = types.ModuleType("truememory._platform")
        platform.pid_is_alive = lambda pid: True
        visited = []
        def entries():
            for index in range(5005):
                visited.append(index)
                yield types.SimpleNamespace(path="/synthetic/pointer-" + str(index) + ".txt")
        with patch.dict(sys.modules, {shared.__name__: shared, platform.__name__: platform}), \
                patch.object(Path, "exists", return_value=True), patch.object(Path, "is_symlink", return_value=False), \
                patch.object(Path, "unlink") as removed, \
                patch.object(COMPLETION.os, "scandir", return_value=contextlib.nullcontext(entries())), \
                patch.object(COMPLETION, "_open_private") as opened:
            COMPLETION.convert_legacy_pointers("synthetic-account", "/synthetic/db", "claude")
        self.assertEqual(len(visited), COMPLETION.MAX_CAPTURES)
        opened.assert_not_called()
        removed.assert_not_called()

    def test_legacy_conversion_requires_exact_scope_retains_pointer_and_avoids_live_workers(self) -> None:
        shared = types.ModuleType("truememory.ingest.hooks._shared")
        shared.backlog_transcript_source = lambda value, host: (value["transcript_path"], value["session_id"])
        platform = types.ModuleType("truememory._platform")
        platform.pid_is_alive = lambda pid: pid == 42
        baseline = {"transcript_path": "/synthetic/source.jsonl", "session_id": "synthetic/a",
                    "user_id": "synthetic-account", "db_path": "/synthetic/db", "transcript_host": "claude"}
        scenarios = [(baseline, ".json", True), ({**baseline, "user_id": "other-account"}, ".json", False),
                     ({**baseline, "db_path": "/synthetic/other-db"}, ".json", False),
                     ({**baseline, "transcript_host": "codex"}, ".json", False),
                     ({**baseline, "claimed_pid": 42}, ".processing", False),
                     (baseline, ".processing", False), ({**baseline, "claimed_pid": 43}, ".processing", True)]
        for missing in ("user_id", "db_path", "transcript_host"):
            value = dict(baseline)
            del value[missing]
            scenarios.append((value, ".json", False))
        for value, suffix, admitted in scenarios:
            raw = json.dumps(value).encode()
            info = types.SimpleNamespace(st_dev=1, st_ino=1, st_size=len(raw), st_mtime_ns=1, st_ctime_ns=1)
            handle = types.SimpleNamespace(read=lambda count: raw[:count], fileno=lambda: 41)
            entry = types.SimpleNamespace(path="/synthetic/pointer" + suffix)
            with patch.dict(sys.modules, {shared.__name__: shared, platform.__name__: platform}), \
                    patch.object(Path, "exists", return_value=True), patch.object(Path, "is_symlink", return_value=False), \
                    patch.object(Path, "unlink") as removed, \
                    patch.object(COMPLETION.os, "scandir", return_value=contextlib.nullcontext(iter([entry]))), \
                    patch.object(COMPLETION, "_open_private", return_value=41), \
                    patch.object(COMPLETION.os, "fdopen", side_effect=lambda *args: contextlib.nullcontext(handle)), \
                    patch.object(COMPLETION.os, "fstat", return_value=info), \
                    patch.object(COMPLETION, "_source_owned"), \
                    patch.object(COMPLETION, "admit_capture", return_value=object()) as capture, \
                    patch.object(COMPLETION, "dispatch", return_value=0) as dispatch:
                COMPLETION.convert_legacy_pointers("synthetic-account", "/synthetic/db", "claude")
            self.assertEqual(capture.call_count, int(admitted))
            self.assertEqual(dispatch.call_count, int(admitted))
            removed.assert_not_called()
            if admitted:
                self.assertEqual(capture.call_args.args,
                    ("/synthetic/source.jsonl", "synthetic/a", "synthetic-account", "/synthetic/db", "claude"))

    def test_automatic_maintenance_failure_never_falls_back_to_legacy_cleanup(self) -> None:
        source = ROOT / "truememory/ingest/hooks/session_start.py"
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_drain_backlog")
        namespace = {"os": os, "log": logging.getLogger(__name__)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), namespace)
        module = types.ModuleType("truememory.ingest.completion")
        def retained_failure(*args):
            raise COMPLETION.CompletionError("synthetic retained failure")
        module.drain = retained_failure
        calls = []
        module.convert_legacy_pointers = lambda *args: calls.append(args)
        with patch.dict(sys.modules, {module.__name__: module}), \
                patch.dict(os.environ, {"TRUEMEMORY_MAINTENANCE_CHILD": "1"}, clear=True):
            namespace["_drain_backlog"]("claude")
        self.assertEqual(calls, [])

    def test_windows_owner_queries_require_insufficient_buffer_and_release_resources(self) -> None:
        import ctypes
        from unittest.mock import Mock
        for query_error, size, expected in ((122, 16, True), (5, 16, False), (122, 65537, False)):
            kernel = types.SimpleNamespace(GetCurrentProcess=Mock(return_value=11),
                CloseHandle=Mock(return_value=True), LocalFree=Mock(return_value=None))
            def get_security(handle, kind, fields, owner, group, dacl, sacl, descriptor):
                owner._obj.value = 101
                descriptor._obj.value = 202
                return 0
            def open_token(process, access, token):
                token._obj.value = 23
                return True
            def token_information(token, kind, buffer, capacity, needed):
                needed._obj.value = size
                return bool(buffer)
            security = types.SimpleNamespace(GetSecurityInfo=Mock(side_effect=get_security),
                OpenProcessToken=Mock(side_effect=open_token), GetTokenInformation=Mock(side_effect=token_information),
                EqualSid=Mock(return_value=True), IsWellKnownSid=Mock(), GetAclInformation=Mock(), GetAce=Mock())
            with patch.object(ctypes, "WinDLL", side_effect=lambda name, **kwargs: security if name == "advapi32" else kernel, create=True), \
                    patch.object(ctypes, "set_last_error", create=True), \
                    patch.object(ctypes, "get_last_error", return_value=query_error, create=True):
                self.assertEqual(COMPLETION._windows_security(41), expected)
            self.assertEqual(kernel.CloseHandle.call_args.args[0].value, 23)
            self.assertEqual(kernel.LocalFree.call_args.args[0].value, 202)
            self.assertEqual(security.GetTokenInformation.call_count, 2 if expected else 1)
        info = types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=0)
        platform = types.ModuleType("msvcrt")
        platform.get_osfhandle = lambda fd: 41
        with patch.object(COMPLETION, "os", types.SimpleNamespace(name="nt", getuid=lambda: 0)), \
                patch.dict(sys.modules, {"msvcrt": platform}), patch.object(COMPLETION, "_windows_security", return_value=False):
            with self.assertRaisesRegex(COMPLETION.CompletionError, "source_ownership_unavailable"):
                COMPLETION._source_owned(info, 41)

    def test_pending_live_launcher_is_advisory_block_not_permission_to_take_attempt(self) -> None:
        capture = COMPLETION.Capture(Path("/synthetic"), "1" * 64, Path("/synthetic/capture.jsonl"), {})
        entry = {"attempts": 1, "next_attempt_at": 0, "status": "pending", "worker_pid": 42}
        platform = types.ModuleType("truememory._platform")
        platform.pid_is_alive = lambda pid: pid == 42
        with patch.dict(sys.modules, {platform.__name__: platform}), \
                patch.object(COMPLETION, "_lock", side_effect=lambda *args, **kwargs: contextlib.nullcontext(True)), \
                patch.object(COMPLETION, "has_receipt", return_value=False), \
                patch.object(COMPLETION, "_state", return_value={"captures": {capture.key: entry}}), \
                patch.object(COMPLETION, "_save_state") as saved:
            self.assertEqual(COMPLETION.reserve_inline_attempt(capture), 0)
        self.assertEqual(entry["attempts"], 1)
        saved.assert_not_called()

    def test_retained_state_complete_flag_is_not_a_receipt_authority(self) -> None:
        entry = {"metadata": {"host": "claude"}, "status": "complete"}
        capture = object()
        with patch.object(Path, "exists", return_value=True), \
                patch.object(COMPLETION, "_account_directory"), \
                patch.object(COMPLETION, "_lock", side_effect=lambda *args, **kwargs: contextlib.nullcontext(True)), \
                patch.object(COMPLETION, "_state", return_value={"captures": {"synthetic-key": entry}}), \
                patch.object(COMPLETION, "_capture_from", return_value=capture), \
                patch.object(COMPLETION, "dispatch", return_value=0) as dispatched:
            COMPLETION.drain("synthetic-account", "/synthetic/db", "claude")
        dispatched.assert_called_once_with(capture)

    def test_capture_exact_byte_version_not_size_only(self) -> None:
        metadata = {"sha256": "0" * 64, "byte_count": 7}
        capture = COMPLETION.Capture(Path("/synthetic"), "1" * 64, Path("/synthetic/capture.jsonl"), metadata)
        value = result()
        self.assertTrue(COMPLETION.strict_complete(value, capture))
        value.coverage["source_version"]["sha256"] = "2" * 64
        self.assertFalse(COMPLETION.strict_complete(value, capture))
        value.coverage["source_version"]["sha256"] = "0" * 64
        value.coverage["source_version"]["byte_count"] = 8
        self.assertFalse(COMPLETION.strict_complete(value, capture))

    def test_identity_keeps_exact_session_account_host_and_locator(self) -> None:
        baseline = ["account-a", "/synthetic/db", "claude", "synthetic/a", "/synthetic/source"]
        identity = COMPLETION._digest(baseline)
        for index, value in enumerate(("account-b", "/synthetic/other-db", "codex", "synthetica", "/synthetic/other-source")):
            changed = list(baseline)
            changed[index] = value
            self.assertNotEqual(identity, COMPLETION._digest(changed))
        with self.assertRaises(COMPLETION.CompletionError):
            COMPLETION._text("x" * 257, 256)
        self.assertEqual(COMPLETION.MAX_CAPTURE_BYTES, 16777216)
        self.assertEqual(COMPLETION.MAX_ACCOUNT_BYTES, 67108864)
        self.assertEqual(COMPLETION.MAX_CAPTURES, 1024)
        self.assertEqual(COMPLETION.MAX_STATE_BYTES, 8388608)
        self.assertEqual(COMPLETION.MAX_ATTEMPTS, 5)

    def test_failure_backoff_and_exhaustion_retain_outcomes(self) -> None:
        entry = {"attempts": 1, "outcomes": []}
        with patch.object(COMPLETION.time, "time", return_value=100):
            COMPLETION._outcome(entry, "storage_unproven")
        self.assertEqual(entry["next_attempt_at"], 102)
        self.assertEqual(entry["status"], "retry")
        entry["attempts"] = 5
        with patch.object(COMPLETION.time, "time", return_value=100):
            COMPLETION._outcome(entry, "worker_unproven")
        self.assertEqual(entry["next_attempt_at"], 132)
        self.assertEqual(entry["status"], "exhausted")
        self.assertEqual([item["category"] for item in entry["outcomes"]], ["storage_unproven", "worker_unproven"])


class InertPipelineCompletionControls(unittest.TestCase):
    def setUp(self) -> None:
        source = ROOT / "truememory/ingest/pipeline.py"
        names = {"IngestionResult", "_invalidate_recall_after_batch", "IngestionPipeline"}
        nodes = [node for node in ast.parse(source.read_text()).body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        action = enum.Enum("DedupAction", ["ADD", "UPDATE", "SKIP"])
        self.namespace = {"dataclass": dataclass, "field": field, "contextlib": contextlib, "Iterator": object,
            "time": time, "datetime": datetime, "sqlite3": sqlite3, "Path": Path, "os": os,
            "log": logging.getLogger(__name__), "_safe_log": str, "DedupAction": action,
            "_dedup_store_lock": contextlib.nullcontext,
            "format_for_extraction": lambda messages: "User: " + "synthetic durable source " * 4,
            "SourceSegment": lambda *values: values, "source_context": lambda *values: None,
            "fact_metadata": lambda *values: {}, "gate_observation": lambda *values: {},
            "ProvenanceError": type("SyntheticProvenanceError", (sqlite3.OperationalError,), {}),
            "source_timestamp": lambda metadata: "",
            "check_duplicate": lambda *args, **kwargs: types.SimpleNamespace(action=action.ADD, fact="Synthetic preference", reason="synthetic", existing_id=None)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])), str(source), "exec"), self.namespace)
        self.pipeline = self.namespace["IngestionPipeline"].__new__(self.namespace["IngestionPipeline"])
        self.pipeline._record_source_provenance = True
        self.pipeline.memory = types.SimpleNamespace()
        self.pipeline.user_id = "synthetic-account"
        self.pipeline.llm_config = object()
        self.pipeline.use_llm_dedup = False
        self.pipeline.gate_enabled = True
        decision = types.SimpleNamespace(should_encode=True, encoding_score=1, novelty=1, salience=1, prediction_error=1, reason="synthetic")
        self.pipeline.gate = types.SimpleNamespace(reset_batch=lambda: None, evaluate=lambda *args: decision)
        self.parser = types.ModuleType("truememory.ingest.transcript")
        self.parser_calls = []
        version = types.SimpleNamespace(stable=True, sha256="0" * 64, byte_count=7)
        message = types.SimpleNamespace(role="human", timestamp="", content="synthetic")
        self.parser.parse_transcript_outcome = lambda source: self.parser_calls.append(source) or types.SimpleNamespace(messages=[message], status="complete", file_version=version)
        self.extractor = types.ModuleType("truememory.ingest.extractor")
        self.extraction_calls = []
        self.extraction = types.SimpleNamespace(facts=[], status="complete")
        self.extractor.extract_facts_outcome = lambda *args: self.extraction_calls.append(args) or self.extraction
        self.runtime = types.ModuleType("truememory.tier_switch.runtime")
        self.runtime.fact_operation = lambda memory: contextlib.nullcontext()
        self.runtime.raise_if_serving_rejection = lambda error: None
        self.shared = types.ModuleType("truememory.ingest.hooks._shared")
        self.shared.invalidate_recall_cache = lambda: None
        self.modules = {self.parser.__name__: self.parser, self.extractor.__name__: self.extractor,
                        self.runtime.__name__: self.runtime, self.shared.__name__: self.shared}

    def test_rich_extraction_is_observed_once_and_successful_empty_is_complete(self) -> None:
        with patch.dict(sys.modules, self.modules):
            value = self.pipeline.ingest_transcript("synthetic.jsonl", "exact/session", require_complete=True)
        self.assertEqual(len(self.parser_calls), 1)
        self.assertEqual(len(self.extraction_calls), 1)
        self.assertTrue(COMPLETION.strict_complete(value))

    def test_partial_extraction_and_heuristic_do_not_complete(self) -> None:
        self.extraction.status = "limited"
        with patch.dict(sys.modules, self.modules):
            value = self.pipeline.ingest_transcript("synthetic.jsonl", require_complete=True)
        self.assertFalse(COMPLETION.strict_complete(value))
        self.pipeline.llm_config = None
        self.namespace["extract_facts_simple"] = lambda text: []
        with patch.dict(sys.modules, self.modules):
            value = self.pipeline.ingest_transcript("synthetic.jsonl", require_complete=True)
        self.assertFalse(COMPLETION.strict_complete(value))
        self.assertEqual(len(self.extraction_calls), 1)

    def test_commit_then_raise_keeps_fact_nonterminal(self) -> None:
        self.extraction.facts = [types.SimpleNamespace(content="Synthetic preference", category="preference", confidence=0.9)]
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE synthetic_fact (content TEXT)")
        def ambiguous_write(*args, **kwargs):
            connection.execute("INSERT INTO synthetic_fact VALUES (?)", ("Synthetic preference",))
            connection.commit()
            raise sqlite3.OperationalError("synthetic commit then raise")
        self.pipeline._store_fact = ambiguous_write
        try:
            with patch.dict(sys.modules, self.modules):
                value = self.pipeline.ingest_transcript("synthetic.jsonl", require_complete=True)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM synthetic_fact").fetchone()[0], 1)
            self.assertEqual(value.facts_stored, 0)
            self.assertEqual(value.trace[0]["action"], "storage_failed")
            self.assertFalse(COMPLETION.strict_complete(value))
        finally:
            connection.close()

    def test_initialized_source_add_requires_exact_positive_acknowledgement(self) -> None:
        self.extraction.facts = [types.SimpleNamespace(content="Synthetic preference", category="preference", confidence=0.9)]
        calls = []
        def add_result(acknowledgement):
            self.pipeline.memory = types.SimpleNamespace(_engine=types.SimpleNamespace(
                add=lambda **values: calls.append(values) or acknowledgement))
            with patch.dict(sys.modules, self.modules):
                return self.pipeline.ingest_transcript("synthetic.jsonl", "exact/session", require_complete=True)
        correct = {"id": 1, "content": "[preference] Synthetic preference", "sender": "synthetic-account",
                   "timestamp": "", "category": "preference", "metadata": {}}
        for acknowledgement in (None, {}, {**correct, "id": None}, {**correct, "id": 0},
                                {**correct, "id": True}, {**correct, "metadata": {"dropped": True}},
                                {**correct, "timestamp": "2025"}, {**correct, "content": "other"}):
            value = add_result(acknowledgement)
            self.assertEqual(value.facts_stored, 0)
            self.assertEqual(value.trace[0]["action"], "storage_failed")
            self.assertFalse(COMPLETION.strict_complete(value))
        value = add_result(correct)
        self.assertEqual(value.facts_stored, 1)
        self.assertTrue(COMPLETION.strict_complete(value))
        self.pipeline._record_source_provenance = False
        value = add_result(None)
        self.assertEqual(value.facts_stored, 1)
        self.assertFalse(COMPLETION.strict_complete(value))


class FilesystemCompletionControls(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="tm-synthetic-completion-")
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "private-completion"
        self.source = self.directory / "source.jsonl"
        self.data = b"\n".join(json.dumps({"role": "user", "content": "synthetic source turn"}).encode() for _ in range(6))
        self.source.write_bytes(self.data)
        self.shared = types.ModuleType("truememory.ingest.hooks._shared")
        self.shared.is_allowed_transcript = lambda source, host: Path(source).parent == self.directory
        self.shared.normalize_transcript_host = lambda host: host if host in ("claude", "codex", "gemini", "cursor") else ""
        self.modules = patch.dict(sys.modules, {self.shared.__name__: self.shared})
        self.modules.start()

    def tearDown(self) -> None:
        self.modules.stop()
        self.temporary.cleanup()

    def capture(self, session: str = "synthetic/a", user: str = "synthetic-account", db: str = "/synthetic/db"):
        return COMPLETION.admit_capture(str(self.source), session, user, db, "claude", root=self.root)

    def complete_result(self, capture):
        value = result()
        value.coverage["source_version"] = {name: capture.metadata[name] for name in ("sha256", "byte_count")}
        return value

    def test_capture_is_exact_idempotent_and_scoped_without_sanitization(self) -> None:
        first = self.capture()
        self.assertEqual(first.path.read_bytes(), self.data)
        self.assertEqual(first.key, self.capture().key)
        self.assertNotEqual(first.key, self.capture("synthetica").key)
        self.assertNotEqual(first.directory, self.capture(user="other-account").directory)
        self.assertNotEqual(first.directory, self.capture(db="/synthetic/other-db").directory)
        self.assertEqual(len(list((first.directory / "captures").iterdir())), 2)

    def test_same_size_revision_never_uses_old_receipt(self) -> None:
        first = self.capture()
        attempt = COMPLETION.reserve_inline_attempt(first)
        with COMPLETION.worker_claim(first, attempt) as allowed:
            self.assertTrue(allowed)
            self.assertTrue(COMPLETION.record_receipt(first, self.complete_result(first)))
        replacement = self.data.replace(b"source", b"change")
        self.assertEqual(len(replacement), len(self.data))
        self.source.write_bytes(replacement)
        second = self.capture()
        self.assertNotEqual(first.key, second.key)
        self.assertFalse(COMPLETION.has_receipt(second))
        self.assertTrue(COMPLETION.has_receipt(first))
        self.assertEqual(first.path.read_bytes(), self.data)

    def test_limits_backpressure_preserve_original_and_prior_capture(self) -> None:
        first = self.capture()
        with patch.object(COMPLETION, "MAX_CAPTURE_BYTES", len(self.data) - 1):
            with self.assertRaises(COMPLETION.CompletionError):
                self.capture("other-session")
        self.assertEqual(self.source.read_bytes(), self.data)
        self.assertEqual(first.path.read_bytes(), self.data)
        with patch.object(COMPLETION, "MAX_ACCOUNT_BYTES", len(self.data)):
            with self.assertRaises(COMPLETION.CompletionError):
                self.capture("other-session")
        self.assertEqual(len(list((first.directory / "captures").iterdir())), 1)

    def test_five_failed_attempts_retain_capture_outcomes_and_no_receipt(self) -> None:
        capture = self.capture()
        for number in range(1, 6):
            with patch.object(COMPLETION.time, "time", return_value=number * 1000):
                attempt = COMPLETION.reserve_inline_attempt(capture)
                self.assertEqual(attempt, number)
                with COMPLETION.worker_claim(capture, attempt) as allowed:
                    self.assertTrue(allowed)
                    self.assertFalse(COMPLETION.record_receipt(capture, result(extractor="partial")))
        with patch.object(COMPLETION.time, "time", return_value=10000):
            self.assertEqual(COMPLETION.reserve_inline_attempt(capture), 0)
        state = json.loads((capture.directory / "state.json").read_text())["captures"][capture.key]
        self.assertEqual(state["status"], "exhausted")
        self.assertEqual(len(state["outcomes"]), 5)
        self.assertEqual(capture.path.read_bytes(), self.data)
        self.assertFalse(COMPLETION.has_receipt(capture))

    def test_os_claim_is_exclusive_and_exception_does_not_complete(self) -> None:
        capture = self.capture()
        attempt = COMPLETION.reserve_inline_attempt(capture)
        with self.assertRaises(RuntimeError):
            with COMPLETION.worker_claim(capture, attempt) as first:
                self.assertTrue(first)
                with COMPLETION.worker_claim(capture, attempt) as second:
                    self.assertFalse(second)
                raise RuntimeError("synthetic commit ambiguity")
        self.assertFalse(COMPLETION.has_receipt(capture))
        state = json.loads((capture.directory / "state.json").read_text())["captures"][capture.key]
        self.assertEqual(state["outcomes"][0]["category"], "worker_unproven")
        self.assertEqual(capture.path.read_bytes(), self.data)

    def test_receipt_requires_owned_attempt_and_keeps_capture_without_it(self) -> None:
        capture = self.capture()
        with self.assertRaisesRegex(COMPLETION.CompletionError, "receipt_without_owned_attempt"):
            COMPLETION.record_receipt(capture, self.complete_result(capture))
        self.assertFalse(COMPLETION.has_receipt(capture))
        self.assertEqual(capture.path.read_bytes(), self.data)

    def test_symlink_capture_and_corrupt_state_are_rejected(self) -> None:
        capture = self.capture()
        link = self.directory / "linked.jsonl"
        try:
            link.symlink_to(self.source)
        except (OSError, NotImplementedError):
            self.skipTest("synthetic symlink unavailable")
        with self.assertRaises(COMPLETION.CompletionError):
            COMPLETION.admit_capture(str(link), "synthetic", "synthetic-account", "/synthetic/db", root=self.root)
        state = capture.directory / "state.json"
        state.write_text("invalid synthetic state")
        with self.assertRaises(COMPLETION.CompletionError):
            self.capture()
        self.assertEqual(capture.path.read_bytes(), self.data)


if __name__ == "__main__":
    unittest.main()
