"""Synthetic stdlib controls for native recall output, without package imports.

The decoders model documented schemas. They do not execute native host parsers
or prove that a native model received the context. Retrieval and capture are
inert collaborators; the actual hook main/parser and serializer execute.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import os
import shlex
import sys
import types
import unittest
from pathlib import Path
from typing import Callable, cast
from unittest.mock import patch

TEST_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = Path(os.environ.get("TRUEMEMORY_HOOK_CONTRACT_SOURCE_ROOT", str(TEST_ROOT)))
HOOK_DIR = SOURCE_ROOT / "truememory" / "ingest" / "hooks"
_OUTPUT_PATH = TEST_ROOT / "truememory" / "ingest" / "hooks" / "output.py"
_SPEC = importlib.util.spec_from_file_location("synthetic_hook_output_763", _OUTPUT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
output = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(output)

SYNTHETIC_CONTEXT = "<truememory-recall>synthetic blue lantern</truememory-recall>"
NATIVE_EVENTS = (
    ("cursor", "sessionStart"),
    ("gemini", "SessionStart"),
    ("gemini", "BeforeAgent"),
    ("codex", "SessionStart"),
    ("codex", "UserPromptSubmit"),
)


def decode_fixture(host: str, event: str, response: str) -> str | None:
    """Strict synthetic protocol fixture, independently spelled field paths."""
    data = json.loads(response)
    if host == "cursor":
        if event != "sessionStart" or set(data) != {"additional_context"}:
            return None
        context = data["additional_context"]
    elif host in {"gemini", "codex"}:
        events = {"SessionStart", "BeforeAgent"} if host == "gemini" else {"SessionStart", "UserPromptSubmit"}
        if event not in events or set(data) != {"hookSpecificOutput"}:
            return None
        specific = data["hookSpecificOutput"]
        if not isinstance(specific, dict) or set(specific) != {"hookEventName", "additionalContext"}:
            return None
        if specific["hookEventName"] != event:
            return None
        context = specific["additionalContext"]
    else:
        if event not in {"SessionStart", "UserPromptSubmit"} or set(data) != {"additionalContext"}:
            return None
        context = data["additionalContext"]
    return context if isinstance(context, str) else None


class RecordingStream(io.StringIO):
    def __init__(self, failure: str = "") -> None:
        super().__init__()
        self.failure = failure
        self.flushed = False

    def write(self, text: str) -> int:
        if self.failure == "write":
            raise BrokenPipeError("synthetic write refusal")
        if self.failure == "short":
            return super().write(text[:len(text) // 2])
        return super().write(text)

    def flush(self) -> None:
        if self.failure == "flush":
            raise BrokenPipeError("synthetic flush refusal")
        self.flushed = True


def load_functions(script: str, names: set[str]) -> dict[str, object]:
    source = HOOK_DIR / script
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise AssertionError("Missing production function requested by fixture")
    namespace: dict[str, object] = {
        "argparse": argparse, "os": os, "json": json, "sys": sys, "Path": Path,
        "HOOK_HOSTS": output.HOOK_HOSTS, "emit_context": output.emit_context,
        "supports_context": output.supports_context,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class HookHarness:
    def __init__(self) -> None:
        self.markers: set[str] = set()
        self.flush_at_mark: list[bool] = []
        self.calls: list[tuple[str, object]] = []
        self.first_run = False
        self.recall_context = SYNTHETIC_CONTEXT
        self.notice = ""
        self.intensity = "standard"
        self.admitted = False
        self.stream = RecordingStream()
        self.shared = types.ModuleType("truememory.ingest.hooks._shared")
        self.shared.mark_recall_injected = self.mark  # type: ignore[attr-defined]
        self.shared.consume_recall_injected = self.consume  # type: ignore[attr-defined]
        self.shared.is_allowed_transcript = self.admit  # type: ignore[attr-defined]
        self.shared.should_extract_session = lambda *_a: True  # type: ignore[attr-defined]
        self.shared.mark_session_extracted = lambda *_a, **_k: self.calls.append(("extracted", None))  # type: ignore[attr-defined]
        self.stop = types.ModuleType("truememory.ingest.hooks.stop")
        self.stop.TRACE_DIR = types.SimpleNamespace(mkdir=lambda **_k: None)  # type: ignore[attr-defined]
        self.stop.LOG_DIR = types.SimpleNamespace(mkdir=lambda **_k: None)  # type: ignore[attr-defined]
        self.stop._has_enough_messages = lambda *_a: self.calls.append(("read-transcript", _a)) or True  # type: ignore[attr-defined]
        self.stop._run_background_ingestion = self.background  # type: ignore[attr-defined]

    def mark(self, session: str) -> None:
        self.calls.append(("mark", session))
        self.flush_at_mark.append(self.stream.flushed)
        self.markers.add(session)

    def consume(self, session: str) -> bool:
        self.calls.append(("consume", session))
        found = session in self.markers
        self.markers.discard(session)
        return found

    def admit(self, transcript: str, *, host: str) -> bool:
        self.calls.append(("admit", (transcript, host)))
        return self.admitted

    def background(self, *args: object, transcript_host: str) -> int:
        self.calls.append(("spawn", (args, transcript_host)))
        return 21

    def run(
        self, script: str, host: str, event: str, *, failure: str = "",
        payload: dict[str, object] | None = None,
    ) -> str:
        namespace = load_functions(script, {"main"})
        self.stream = RecordingStream(failure)
        namespace.update({
            "_parse_args": lambda: argparse.Namespace(host=host, event=event, user="synthetic-user", db="", scan_stale=False),
            "_run_maintenance_background": lambda *_a: None,
            "_get_search_intensity": lambda: "standard",
            "_INTENSITY_MEMORY_LIMITS": {"standard": 25}, "_BASE_MEMORY_LIMIT": 25,
            "_intensity_budget": lambda *_a: 8192, "_is_first_run": lambda: self.first_run,
            "_first_run_context": lambda: "synthetic onboarding",
            "recall_memories": lambda *_a, **_k: self.recall_context,
            "_check_for_update": lambda: self.notice, "_check_email_needed": lambda: "",
            "log": types.SimpleNamespace(error=lambda *_a: self.calls.append(("hook-error", None))),
            "buffer_message": lambda *_a: self.calls.append(("buffer", None)),
            "_prune_old_buffers": lambda: None, "_try_capture_email": lambda *_a: None,
            "_get_intensity_config": lambda: (self.intensity, "standard"),
            "_increment_prompt_count": lambda *_a: 5,
            "_try_per_exchange_store": lambda *_a: self.calls.append(("store", None)),
            "_try_auto_recall": self.auto_recall, "_try_proactive_recall": self.proactive_recall,
        })
        data = payload if payload is not None else {"session_id": "synthetic-session", "prompt": "Recall synthetic color"}
        with patch.dict(sys.modules, {self.shared.__name__: self.shared, self.stop.__name__: self.stop}), patch.object(sys, "stdin", io.StringIO(json.dumps(data))), patch.object(sys, "stdout", self.stream), patch.dict(os.environ, {"TRUEMEMORY_EXTRACTION": ""}):
            cast(Callable[[], None], namespace["main"])()
        return self.stream.getvalue()

    def auto_recall(self, *_args: object, debounced: bool) -> str | None:
        self.calls.append(("auto", debounced))
        return None if debounced else SYNTHETIC_CONTEXT

    def proactive_recall(self, *_args: object, debounced: bool) -> tuple[str | None, bool]:
        self.calls.append(("proactive", debounced))
        return (None if debounced else SYNTHETIC_CONTEXT), True


class TestNativeOutput(unittest.TestCase):
    def test_official_event_response_fixtures(self) -> None:
        for host, event in NATIVE_EVENTS + (("claude", "SessionStart"), ("claude", "UserPromptSubmit")):
            with self.subTest(host=host, event=event):
                stream = RecordingStream()
                self.assertTrue(output.emit_context(SYNTHETIC_CONTEXT, host=host, event=event, stream=stream))
                self.assertTrue(stream.flushed)
                self.assertEqual(decode_fixture(host, event, stream.getvalue()), SYNTHETIC_CONTEXT)
                self.assertEqual(stream.getvalue().count("\n"), 1)

    def test_legacy_payload_rejected_by_native_fixtures(self) -> None:
        legacy = json.dumps({"additionalContext": SYNTHETIC_CONTEXT})
        for host, event in NATIVE_EVENTS:
            with self.subTest(host=host, event=event):
                self.assertIsNone(decode_fixture(host, event, legacy))

    def test_wrong_events_and_hosts_emit_nothing(self) -> None:
        for host, event in (("cursor", "beforeSubmitPrompt"), ("cursor", "SessionStart"), ("gemini", "UserPromptSubmit"), ("codex", "BeforeAgent"), ("codex", "Stop"), ("claude", "BeforeAgent"), ("unknown", "SessionStart"), ("codex ", "SessionStart")):
            with self.subTest(host=host, event=event):
                stream = RecordingStream()
                self.assertFalse(output.emit_context(SYNTHETIC_CONTEXT, host=host, event=event, stream=stream))
                self.assertEqual(stream.getvalue(), "")
                self.assertFalse(stream.flushed)

    def test_io_failures_do_not_report_emitted(self) -> None:
        for failure in ("write", "flush", "short"):
            with self.subTest(failure=failure):
                self.assertFalse(output.emit_context(SYNTHETIC_CONTEXT, host="codex", event="SessionStart", stream=RecordingStream(failure)))

    def test_empty_context_emits_nothing(self) -> None:
        stream = RecordingStream()
        self.assertFalse(output.emit_context("", host="codex", event="SessionStart", stream=stream))
        self.assertEqual(stream.getvalue(), "")

    def test_session_main_uses_native_contract_and_marks_after_flush(self) -> None:
        for host, event in (("cursor", "sessionStart"), ("gemini", "SessionStart"), ("codex", "SessionStart"), ("claude", "SessionStart")):
            with self.subTest(host=host):
                harness = HookHarness()
                response = harness.run("session_start.py", host, event)
                self.assertEqual(decode_fixture(host, event, response), SYNTHETIC_CONTEXT)
                self.assertEqual(harness.markers, {"synthetic-session"})
                self.assertEqual(harness.flush_at_mark, [True])
                self.assertEqual([call for call in harness.calls if call[0] == "mark"], [("mark", "synthetic-session")])

    def test_session_failed_emission_never_marks(self) -> None:
        for failure in ("write", "flush", "short"):
            with self.subTest(failure=failure):
                harness = HookHarness()
                harness.run("session_start.py", "codex", "SessionStart", failure=failure)
                self.assertEqual(harness.markers, set())
                self.assertFalse([call for call in harness.calls if call[0] == "mark"])
                response = harness.run("user_prompt_submit.py", "codex", "UserPromptSubmit")
                self.assertEqual(decode_fixture("codex", "UserPromptSubmit", response), SYNTHETIC_CONTEXT)

    def test_session_unsupported_event_never_marks(self) -> None:
        harness = HookHarness()
        self.assertEqual(harness.run("session_start.py", "cursor", "beforeSubmitPrompt"), "")
        self.assertEqual(harness.markers, set())

    def test_notices_and_onboarding_do_not_suppress_prompt_recall(self) -> None:
        for first_run in (False, True):
            with self.subTest(first_run=first_run):
                harness = HookHarness()
                harness.first_run = first_run
                harness.recall_context = ""
                harness.notice = "synthetic update"
                response = harness.run("session_start.py", "gemini", "SessionStart")
                context = decode_fixture("gemini", "SessionStart", response)
                self.assertIsInstance(context, str)
                self.assertIn("synthetic update", cast(str, context))
                self.assertEqual(harness.markers, set())
                response = harness.run("user_prompt_submit.py", "gemini", "BeforeAgent")
                self.assertEqual(decode_fixture("gemini", "BeforeAgent", response), SYNTHETIC_CONTEXT)

    def test_native_prompt_main_preserves_once_only_debounce(self) -> None:
        for host, event in (("gemini", "BeforeAgent"), ("codex", "UserPromptSubmit"), ("claude", "UserPromptSubmit")):
            for intensity in ("standard", "max"):
                with self.subTest(host=host, intensity=intensity):
                    harness = HookHarness()
                    harness.intensity = intensity
                    harness.run("session_start.py", host, "SessionStart")
                    self.assertEqual(harness.run("user_prompt_submit.py", host, event), "")
                    self.assertEqual(harness.markers, set())
                    response = harness.run("user_prompt_submit.py", host, event)
                    self.assertEqual(decode_fixture(host, event, response), SYNTHETIC_CONTEXT)
                    self.assertEqual(len([call for call in harness.calls if call[0] == "consume"]), 2)
                    if intensity == "max":
                        self.assertFalse([call for call in harness.calls if call[0] == "auto"])

    def test_cursor_prompt_capture_without_unsupported_recall(self) -> None:
        harness = HookHarness()
        harness.intensity = "max"
        response = harness.run("user_prompt_submit.py", "cursor", "beforeSubmitPrompt")
        self.assertEqual(response, "")
        self.assertIn(("buffer", None), harness.calls)
        self.assertIn(("store", None), harness.calls)
        self.assertFalse([call for call in harness.calls if call[0] in {"auto", "proactive", "consume"}])

    def test_prompt_emission_failure_does_not_escape_or_set_markers(self) -> None:
        for host, event in (("gemini", "BeforeAgent"), ("codex", "UserPromptSubmit"), ("claude", "UserPromptSubmit")):
            for failure in ("write", "flush", "short"):
                with self.subTest(host=host, failure=failure):
                    harness = HookHarness()
                    harness.run("user_prompt_submit.py", host, event, failure=failure)
                    self.assertEqual(harness.markers, set())
                    self.assertFalse([call for call in harness.calls if call[0] == "mark"])

    def test_short_first_prompt_consumes_only_its_marker(self) -> None:
        harness = HookHarness()
        harness.run("session_start.py", "codex", "SessionStart")
        data = {"session_id": "synthetic-session", "prompt": "hi"}
        self.assertEqual(harness.run("user_prompt_submit.py", "codex", "UserPromptSubmit", payload=data), "")
        self.assertEqual(harness.markers, set())
        self.assertEqual(len([call for call in harness.calls if call[0] == "consume"]), 1)
        response = harness.run("user_prompt_submit.py", "codex", "UserPromptSubmit")
        self.assertEqual(decode_fixture("codex", "UserPromptSubmit", response), SYNTHETIC_CONTEXT)

    def test_untrusted_stdin_cannot_select_host_or_event(self) -> None:
        harness = HookHarness()
        data = {"session_id": "synthetic-session", "host": "cursor", "hook_event_name": "beforeSubmitPrompt", "event": "BeforeAgent"}
        response = harness.run("session_start.py", "codex", "SessionStart", payload=data)
        self.assertEqual(decode_fixture("codex", "SessionStart", response), SYNTHETIC_CONTEXT)

    def test_cli_trusted_identity_defaults_and_overrides(self) -> None:
        for script, default in (("session_start.py", "SessionStart"), ("user_prompt_submit.py", "UserPromptSubmit")):
            parser = cast(Callable[[], argparse.Namespace], load_functions(script, {"_parse_args"})["_parse_args"])
            with patch.object(sys, "argv", [script]):
                args = parser()
                self.assertTrue(hasattr(args, "host"))
                self.assertTrue(hasattr(args, "event"))
                self.assertEqual((args.host, args.event), ("claude", default))
            with patch.object(sys, "argv", [script, "--host", "gemini", "--event", "BeforeAgent"]):
                args = parser()
                self.assertEqual((args.host, args.event), ("gemini", "BeforeAgent"))
            with patch.object(sys, "argv", [script, "--host", "invalid"]), patch.object(sys, "stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser()

    def test_transcript_rejection_precedes_path_read(self) -> None:
        harness = HookHarness()
        data = {"session_id": "synthetic-session", "prompt": "Recall synthetic color", "transcript_path": "/synthetic/untrusted.jsonl"}
        with patch.object(Path, "exists", side_effect=AssertionError("Unadmitted transcript stat")), patch.object(Path, "is_file", side_effect=AssertionError("Unadmitted transcript stat")):
            response = harness.run("user_prompt_submit.py", "codex", "UserPromptSubmit", payload=data)
        self.assertEqual(decode_fixture("codex", "UserPromptSubmit", response), SYNTHETIC_CONTEXT)
        self.assertIn(("admit", ("/synthetic/untrusted.jsonl", "codex")), harness.calls)

    def test_admitted_transcript_threads_trusted_identity_to_spawn(self) -> None:
        for host, event in (("cursor", "beforeSubmitPrompt"), ("gemini", "BeforeAgent"), ("codex", "UserPromptSubmit"), ("claude", "UserPromptSubmit")):
            with self.subTest(host=host):
                harness = HookHarness()
                harness.admitted = True
                data = {"session_id": "synthetic-session", "prompt": "Recall synthetic color", "transcript_path": "/synthetic/native.jsonl"}
                with patch.object(Path, "is_file", return_value=True):
                    harness.run("user_prompt_submit.py", host, event, payload=data)
                self.assertIn(("admit", ("/synthetic/native.jsonl", host)), harness.calls)
                self.assertIn(("spawn", (("/synthetic/native.jsonl", "synthetic-session", "synthetic-user", ""), host)), harness.calls)
                self.assertIn(("extracted", None), harness.calls)
                phases = [call[0] for call in harness.calls]
                self.assertLess(phases.index("admit"), phases.index("read-transcript"))


class VirtualPath:
    files: dict[str, str] = {}

    def __init__(self, path: object) -> None:
        self.path = str(path)

    @classmethod
    def home(cls) -> VirtualPath:
        return cls("/synthetic")

    @property
    def parent(self) -> VirtualPath:
        return VirtualPath(str(Path(self.path).parent))

    def __truediv__(self, part: str) -> VirtualPath:
        return VirtualPath(self.path + "/" + part)

    def __str__(self) -> str:
        return self.path

    def exists(self) -> bool:
        return self.path in self.files

    def mkdir(self, **_kwargs: object) -> None:
        return None

    def read_text(self, **_kwargs: object) -> str:
        return self.files[self.path]


class TestCursorInstallation(unittest.TestCase):
    def test_fresh_and_upgrade_install_native_identity_preserve_other_hooks(self) -> None:
        source = SOURCE_ROOT / "truememory" / "hooks" / "adapters" / "cursor.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        owned = {"_HOOK_EVENTS", "_STALE_EVENTS", "_TRUEMEMORY_MARKER", "_CURSOR_DIR", "_MCP_CONFIG", "_HOOK_CONFIG"}
        selected = [node for node in tree.body if (isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) and target.id in owned for target in node.targets)) or (isinstance(node, ast.ClassDef) and node.name == "CursorAdapter")]
        namespace: dict[str, object] = {
            "CLIAdapter": type("SyntheticAdapter", (), {}), "Path": VirtualPath,
            "json": json, "shlex": shlex, "sys": sys, "shutil": types.SimpleNamespace(),
            "__file__": "/synthetic/truememory/hooks/adapters/cursor.py",
            "atomic_write_text": lambda path, text: VirtualPath.files.__setitem__(str(path), text),
        }
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
        adapter_type = cast(type, namespace["CursorAdapter"])
        config_path = str(namespace["_HOOK_CONFIG"])
        third_party = {"command": "synthetic unrelated hook", "timeout": 17, "custom": "preserve"}
        for upgraded in (False, True):
            with self.subTest(upgraded=upgraded):
                VirtualPath.files = {}
                existing = {"version": 1, "custom": "preserve", "hooks": {"sessionStart": [third_party.copy()]}}
                if upgraded:
                    existing["hooks"]["sessionStart"].append({"command": "python /synthetic/truememory/session_start.py", "timeout": 1, "custom": "owned metadata"})
                VirtualPath.files[config_path] = json.dumps(existing)
                adapter = adapter_type()
                adapter.install_hooks(python_path="/synthetic/python path", user_id="synthetic user", db_path="/synthetic/db path")
                data = json.loads(VirtualPath.files[config_path])
                self.assertEqual(data["hooks"]["sessionStart"][0], third_party)
                self.assertEqual(data["custom"], "preserve")
                for event in ("sessionStart", "stop", "beforeSubmitPrompt", "preCompact"):
                    commands = [entry for entry in data["hooks"][event] if "truememory" in entry["command"]]
                    self.assertEqual(len(commands), 1)
                    parts = shlex.split(commands[0]["command"])
                    self.assertIn("--host", parts)
                    self.assertIn("--event", parts)
                    self.assertEqual(parts[parts.index("--host") + 1], "cursor")
                    self.assertEqual(parts[parts.index("--event") + 1], event)
                    self.assertEqual(parts[parts.index("--user") + 1], "synthetic user")
                    self.assertEqual(parts[parts.index("--db") + 1], "/synthetic/db path")
                if upgraded:
                    self.assertEqual(data["hooks"]["sessionStart"][1]["custom"], "owned metadata")
                first = VirtualPath.files[config_path]
                adapter.install_hooks(python_path="/synthetic/python path", user_id="synthetic user", db_path="/synthetic/db path")
                self.assertEqual(VirtualPath.files[config_path], first)


if __name__ == "__main__":
    unittest.main()
