"""Synthetic stdlib controls for trusted transcript admission (#761).

Run directly with Python. Production functions execute from their AST with
model, database and spawn collaborators replaced by inert boundary stubs.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import io
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
from typing import Iterator
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def source_module(name: str, relative_path: str, names: set[str] | None = None) -> types.ModuleType:
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    if names is not None:
        tree.body = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
        ]
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / relative_path)
    if names is not None:
        module.__dict__.update({
            "Path": Path, "json": json, "os": os, "sys": sys,
            "subprocess": subprocess, "argparse": argparse,
            "log": logging.getLogger(name), "logging": logging,
        })
    sys.modules[name] = module
    exec(compile(tree, module.__file__, "exec"), module.__dict__)
    return module


@contextmanager
def scoped_module_bindings(packages: dict[str, types.ModuleType], owned: set[str]) -> Iterator[None]:
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in owned}
    sys.modules.update(packages)
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@contextmanager
def allowed_gate():
    yield True


class TranscriptAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-transcripts-761-")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name) / "home"
        self.home.mkdir()
        self.home_patch = patch.object(Path, "home", return_value=self.home)
        self.home_patch.start()
        self.addCleanup(self.home_patch.stop)
        env = {key: value for key, value in os.environ.items() if key not in (
            "CODEX_HOME", "GEMINI_CLI_HOME", "SANDBOX", "TRUEMEMORY_TRANSCRIPT_DIR",
            "TRUEMEMORY_EXTRACTION", "TRUEMEMORY_MAINTENANCE_CHILD",
        )}
        self.env_patch = patch.dict(os.environ, env, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        packages: dict[str, types.ModuleType] = {}
        for name in (
            "truememory", "truememory.ingest", "truememory.ingest.hooks",
            "truememory.hooks", "truememory.hooks.adapters",
        ):
            module = types.ModuleType(name)
            module.__path__ = []
            packages[name] = module
        platform = types.ModuleType("truememory._platform")
        platform._env_int = lambda name, default, **kwargs: default
        platform.spawn_kwargs = lambda: {}
        platform.pid_is_alive = lambda pid: False
        packages["truememory"]._platform = platform
        packages["truememory._platform"] = platform
        core = types.ModuleType("truememory.hooks.core")
        core.spawn_gate = allowed_gate
        core.register_spawned_pid = lambda pid: None
        core._load_cap_state = lambda: {"cap": 2}
        packages["truememory.hooks.core"] = core
        owned = set(packages) | {
            "truememory.hooks.adapters.base", "truememory.hooks.adapters.codex",
            "truememory.hooks.adapters.gemini", "truememory.ingest.hooks._shared",
            "truememory.ingest.hooks.stop", "truememory.ingest.hooks.compact",
            "truememory.ingest.hooks.session_start", "truememory.ingest.cli",
            "truememory.ingest.transcript",
        }
        self.module_bindings = scoped_module_bindings(packages, owned)
        self.module_bindings.__enter__()
        self.addCleanup(self.module_bindings.__exit__, None, None, None)
        source_module("truememory.hooks.adapters.base", "truememory/hooks/adapters/base.py")
        self.shared = source_module(
            "truememory.ingest.hooks._shared", "truememory/ingest/hooks/_shared.py",
        )
        self.shared.check_extraction_budget = lambda: True
        self.shared.refund_extraction_budget = lambda: None
        self.cleanup_stale_processing = self.shared.cleanup_stale_processing
        self.shared.cleanup_stale_processing = lambda path: None
        self.shared.mark_session_extracted = lambda *args, **kwargs: None
        self.stop = source_module("truememory.ingest.hooks.stop", "truememory/ingest/hooks/stop.py")
        self.stop._writable_dirs_ok = lambda: True
        self.stop.TRACE_DIR.mkdir(parents=True)
        self.stop.LOG_DIR.mkdir(parents=True)
        self.compact = source_module("truememory.ingest.hooks.compact", "truememory/ingest/hooks/compact.py")
        self.session_start = source_module(
            "truememory.ingest.hooks.session_start", "truememory/ingest/hooks/session_start.py",
            {"_drain_backlog", "_run_maintenance_background"},
        )
        self.session_start.BACKLOG_DIR = self.shared.BACKLOG_DIR
        self.session_start._DRAIN_CAP = 3
        self.session_start._platform = platform
        self.cli = source_module("truememory.ingest.cli", "truememory/ingest/cli.py", {"_cascade_next"})
        self.parser = source_module("truememory.ingest.transcript", "truememory/ingest/transcript.py")

    def write(self, path: Path, text: str = "synthetic conversation") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def native_path(self, host: str, name: str = "session.jsonl") -> Path:
        roots = {
            "claude": self.home / ".claude/projects/workspace",
            "codex": self.home / ".codex/sessions/2026/10/10",
            "gemini": self.home / ".gemini/tmp/workspace/chats",
        }
        return roots[host] / name

    def marker(self, host: str, name: str, path: Path, **extra: object) -> Path:
        data = {"transcript_path": str(path), "transcript_host": host,
                "session_id": name, "user_id": "synthetic", "db_path": ""}
        data.update(extra)
        return self.write(self.shared.BACKLOG_DIR / f"{name}.json", json.dumps(data))

    def test_native_roots_are_isolated_by_installed_host(self) -> None:
        for source in ("claude", "codex", "gemini"):
            path = self.write(self.native_path(source))
            for host in ("claude", "codex", "gemini", "cursor", "invalid"):
                with self.subTest(source=source, host=host):
                    self.assertEqual(self.shared.is_allowed_transcript(str(path), host), source == host)

    def test_legacy_default_remains_claude(self) -> None:
        self.assertTrue(self.shared.is_allowed_transcript(str(self.native_path("claude"))))
        self.assertFalse(self.shared.is_allowed_transcript(str(self.native_path("codex"))))
        self.assertFalse(self.shared.is_allowed_transcript(str(self.native_path("gemini"))))

    def test_unrelated_home_and_gemini_runtime_paths_are_rejected(self) -> None:
        for path in (self.home / "notes.txt", self.home / ".gemini/settings.json",
                     self.home / ".gemini/tmp/workspace/logs.json",
                     self.home / ".gemini/tmp/workspace/memory/item.json",
                     self.home / ".gemini/tmp/chats/item.json"):
            with self.subTest(path=path):
                self.assertFalse(self.shared.is_allowed_transcript(str(path), "gemini"))

    def test_gemini_supports_nested_subagent_chats(self) -> None:
        self.assertTrue(self.shared.is_allowed_transcript(
            str(self.native_path("gemini") / "subagent/turn.jsonl"), "gemini",
        ))

    def test_codex_home_override_selects_only_sessions(self) -> None:
        custom = self.home / "custom-codex"
        os.environ["CODEX_HOME"] = str(custom)
        self.assertTrue(self.shared.is_allowed_transcript(str(custom / "sessions/date/turn.jsonl"), "codex"))
        self.assertFalse(self.shared.is_allowed_transcript(str(custom / "config.toml"), "codex"))
        self.assertFalse(self.shared.is_allowed_transcript(str(self.native_path("codex")), "codex"))
        self.assertFalse(self.shared.is_allowed_transcript(str(custom / "sessions/date/turn.jsonl"), "gemini"))

    def test_gemini_home_and_sandbox_select_only_documented_runtime(self) -> None:
        custom = self.home / "custom-gemini"
        os.environ["GEMINI_CLI_HOME"] = str(custom)
        path = custom / ".gemini/tmp/workspace/chats/turn.json"
        self.assertTrue(self.shared.is_allowed_transcript(str(path), "gemini"))
        self.assertFalse(self.shared.is_allowed_transcript(str(self.native_path("gemini")), "gemini"))
        os.environ["SANDBOX"] = "sandbox-exec"
        sandbox_path = custom / ".cache/.gemini/tmp/workspace/chats/turn.jsonl"
        self.assertTrue(self.shared.is_allowed_transcript(str(sandbox_path), "gemini"))
        self.assertFalse(self.shared.is_allowed_transcript(str(path), "gemini"))
        self.assertFalse(self.shared.is_allowed_transcript(str(sandbox_path), "codex"))

    def test_explicit_scoped_override_and_broad_home_rejection(self) -> None:
        custom = self.home / "synthetic-transcripts"
        os.environ["TRUEMEMORY_TRANSCRIPT_DIR"] = str(custom)
        self.assertTrue(self.shared.is_allowed_transcript(str(custom / "turn.json"), "cursor"))
        self.assertFalse(self.shared.is_allowed_transcript(str(custom / "turn.json"), "invalid"))
        for broad in (self.home, self.home.parent, Path("/")):
            os.environ["TRUEMEMORY_TRANSCRIPT_DIR"] = str(broad)
            self.assertFalse(self.shared.is_allowed_transcript(str(self.home / "notes.txt")))

    def test_path_shape_is_validated_without_coercion(self) -> None:
        for value in (None, 42, True, [], {}, "", " ", "bad\0path"):
            with self.subTest(value=value):
                self.assertFalse(self.shared.is_allowed_transcript(value, "codex"))
        for value in (None, [], {}, "CODEX", " codex", "codex;command"):
            self.assertEqual(self.shared.normalize_transcript_host(value), "")

    def test_traversal_and_root_itself_are_rejected(self) -> None:
        for host in ("claude", "codex", "gemini"):
            with self.subTest(host=host):
                self.assertFalse(self.shared.is_allowed_transcript(str(self.native_path(host).parent / "../outside.txt"), host))
        self.assertFalse(self.shared.is_allowed_transcript(str(self.home / ".codex/sessions"), "codex"))
        self.assertFalse(self.shared.is_allowed_transcript(str(self.home / ".gemini/tmp/workspace/chats"), "gemini"))

    def test_symlink_escapes_and_loops_are_rejected(self) -> None:
        outside = self.write(self.home / "outside.txt")
        for host in ("claude", "codex", "gemini"):
            path = self.native_path(host)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(outside)
            self.assertFalse(self.shared.is_allowed_transcript(str(path), host))
            loop = path.parent / "loop.jsonl"
            loop.symlink_to(loop)
            self.assertFalse(self.shared.is_allowed_transcript(str(loop), host))
        subtree = self.home / ".gemini/tmp/other/chats"
        subtree.parent.mkdir(parents=True)
        subtree.symlink_to(self.home)
        self.assertFalse(self.shared.is_allowed_transcript(str(subtree / "outside.txt"), "gemini"))

    def test_marker_shape_and_source_are_sanitized(self) -> None:
        path = str(self.native_path("codex"))
        data = {"transcript_path": path, "transcript_host": "codex", "session_id": "../synthetic:id"}
        self.assertEqual(self.shared.backlog_transcript_source(data, "codex"), (path, "syntheticid"))
        self.assertIsNone(self.shared.backlog_transcript_source(data, "gemini"))
        for key, value in (("transcript_host", []), ("transcript_host", "invalid"),
                           ("transcript_path", 42), ("user_id", []), ("db_path", "bad\0db"),
                           ("session_id", {})):
            bad = dict(data)
            bad[key] = value
            self.assertIsNone(self.shared.backlog_transcript_source(bad, "codex"))
        for bad in (None, [], 42, "text"):
            self.assertIsNone(self.shared.backlog_transcript_source(bad, "codex"))

    def test_legacy_marker_is_only_admitted_by_claude(self) -> None:
        data = {"transcript_path": str(self.native_path("claude")), "session_id": "synthetic"}
        self.assertIsNotNone(self.shared.backlog_transcript_source(data))
        self.assertIsNone(self.shared.backlog_transcript_source(data, "codex"))

    def test_stale_marker_shape_cannot_crash_source_admission(self) -> None:
        for i, data in enumerate(([], {"claimed_pid": []}, {"claimed_pid": True}, {"claimed_pid": -1})):
            path = self.write(self.shared.BACKLOG_DIR / f"malformed-{i}.processing", json.dumps(data))
            os.utime(path, (1, 1))
        self.cleanup_stale_processing(self.shared.BACKLOG_DIR)
        self.assertEqual(len(list(self.shared.BACKLOG_DIR.glob("*.corrupt"))), 4)
        self.assertEqual(list(self.shared.BACKLOG_DIR.glob("*.json")), [])

    def test_stop_correct_host_reaches_parser_once_and_stdin_identity_cannot_override(self) -> None:
        for host in ("codex", "gemini"):
            text = json.dumps([{"role": "user", "content": "synthetic turn"} for _ in range(5)])
            path = self.write(self.native_path(host), text)
            calls: list[tuple[str, str]] = []

            def extract(transcript: str, session: str, user: str, db: str, transcript_host: str) -> int:
                calls.append((transcript, transcript_host))
                self.assertEqual(len(self.parser.parse_transcript(Path(transcript))), 5)
                return 17

            with patch.object(self.stop, "_run_background_ingestion", side_effect=extract), \
                    patch.object(sys, "argv", ["stop.py", "--host", host]), \
                    patch.object(sys, "stdin", io.StringIO(json.dumps({
                        "transcript_path": str(path), "session_id": "synthetic",
                        "host": "claude", "transcript_host": "claude",
                    }))):
                self.stop.main()
            self.assertEqual(calls, [(str(path), host)])
            calls.clear()
            with patch.object(self.stop, "_run_background_ingestion", side_effect=extract), \
                    patch.object(sys, "argv", ["stop.py", "--host", "claude"]), \
                    patch.object(sys, "stdin", io.StringIO(json.dumps({
                        "transcript_path": str(path), "session_id": "synthetic", "host": host,
                    }))):
                self.stop.main()
            self.assertEqual(calls, [])

    def test_stop_malformed_input_never_probes_source(self) -> None:
        for payload in ([], 42, None, {"transcript_path": 42}, {"transcript_path": "bad\0path"}):
            with patch.object(self.stop, "_has_enough_messages", side_effect=AssertionError("unexpected read")), \
                    patch.object(sys, "argv", ["stop.py", "--host", "codex"]), \
                    patch.object(sys, "stdin", io.StringIO(json.dumps(payload))):
                self.stop.main()

    def test_compact_correct_host_snapshot_and_background_admission(self) -> None:
        path = self.write(self.native_path("gemini"), json.dumps([
            {"role": "user", "content": "synthetic turn"} for _ in range(5)
        ]))
        with patch.object(self.compact, "save_snapshot") as snapshot, \
                patch.object(self.stop, "_run_background_ingestion", return_value=17) as spawn, \
                patch.object(sys, "argv", ["compact.py", "--host", "gemini", "--event", "PreCompress"]), \
                patch.object(sys, "stdin", io.StringIO(json.dumps({"transcript_path": str(path), "session_id": "synthetic"}))):
            self.compact.main()
        snapshot.assert_called_once()
        self.assertEqual(spawn.call_args.kwargs["transcript_host"], "gemini")
        with patch.object(self.compact, "save_snapshot") as snapshot, \
                patch.object(sys, "argv", ["compact.py", "--host", "codex"]), \
                patch.object(sys, "stdin", io.StringIO(json.dumps({"transcript_path": str(path), "session_id": "synthetic"}))):
            self.compact.main()
        snapshot.assert_not_called()

    def test_background_spawn_is_admitted_and_carries_trusted_identity(self) -> None:
        path = self.write(self.native_path("codex"))
        with patch.object(subprocess, "Popen", return_value=types.SimpleNamespace(pid=17)) as spawn:
            self.assertEqual(self.stop._run_background_ingestion(str(path), "synthetic", "", "", "codex"), 17)
            cmd = spawn.call_args.args[0]
            self.assertEqual(cmd[cmd.index("--transcript-host") + 1], "codex")
            self.assertEqual(self.stop._run_background_ingestion(str(path), "synthetic", "", "", "gemini"), 0)
        self.assertEqual(spawn.call_count, 1)

    def test_queue_records_valid_source_and_refuses_wrong_host(self) -> None:
        path = self.write(self.native_path("gemini"))
        self.stop._queue_to_backlog(str(path), "../synthetic:id", "", "", "synthetic", "gemini")
        marker = self.shared.BACKLOG_DIR / "syntheticid.json"
        data = json.loads(marker.read_text())
        self.assertEqual(data["transcript_host"], "gemini")
        self.assertEqual(data["session_id"], "syntheticid")
        self.stop._queue_to_backlog(str(path), "wrong", "", "", "synthetic", "codex")
        self.assertFalse((self.shared.BACKLOG_DIR / "wrong.json").exists())

    def test_session_drain_preserves_other_host_and_filters_before_cap(self) -> None:
        for i in range(4):
            self.marker("gemini", f"a-other-{i}", self.write(self.native_path("gemini", f"{i}.json")))
        path = self.write(self.native_path("codex"))
        own = self.marker("codex", "z-own", path)
        with patch.object(subprocess, "Popen", return_value=types.SimpleNamespace(pid=17)) as spawn:
            self.session_start._drain_backlog("codex")
        self.assertEqual(spawn.call_count, 1)
        cmd = spawn.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--transcript-host") + 1], "codex")
        self.assertFalse(own.exists())
        self.assertTrue(own.with_suffix(".processing").exists())
        self.assertEqual(len(list(self.shared.BACKLOG_DIR.glob("a-other-*.json"))), 4)

    def test_session_drain_quarantines_invalid_source_without_spawn(self) -> None:
        marker = self.marker("codex", "bad", self.write(self.home / "notes.txt"))
        with patch.object(subprocess, "Popen") as spawn:
            self.session_start._drain_backlog("codex")
        spawn.assert_not_called()
        self.assertTrue(marker.with_suffix(".corrupt").exists())

    def test_cascade_uses_same_admission_and_spawns_at_most_one(self) -> None:
        other = self.marker("gemini", "a-other", self.write(self.native_path("gemini")))
        bad = self.marker("codex", "b-bad", self.write(self.home / "notes.txt"))
        for name in ("c-own", "d-own"):
            self.marker("codex", name, self.write(self.native_path("codex", f"{name}.jsonl")))
        with patch.object(subprocess, "Popen", return_value=types.SimpleNamespace(pid=17)) as spawn:
            self.cli._cascade_next("codex")
        self.assertEqual(spawn.call_count, 1)
        cmd = spawn.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--transcript-host") + 1], "codex")
        self.assertTrue(other.exists())
        self.assertTrue(bad.with_suffix(".corrupt").exists())
        self.assertTrue((self.shared.BACKLOG_DIR / "d-own.json").exists())

    def test_maintenance_child_retains_host_and_only_claude_scans_claude_archive(self) -> None:
        for host in ("claude", "codex", "gemini", "cursor"):
            with patch.object(subprocess, "Popen") as spawn:
                self.session_start._run_maintenance_background(host)
            script = spawn.call_args.args[0][2]
            self.assertIn(f"_drain_backlog({host!r})", script)
            self.assertEqual(script.endswith("_scan_stale_sessions()"), host == "claude")
        with patch.object(subprocess, "Popen") as spawn:
            self.session_start._run_maintenance_background("codex;invalid")
        spawn.assert_not_called()

    def test_codex_installer_refreshes_identity_and_preserves_external_hook(self) -> None:
        module = source_module("truememory.hooks.adapters.codex", "truememory/hooks/adapters/codex.py")
        module._CONFIG_PATH = self.home / ".codex/config.toml"
        before = ('[[hooks.Stop]]\n[[hooks.Stop.hooks]]\n'
                  'type = "command"\ncommand = "external-helper"\n'
                  '[[hooks.SessionStart]]\n[[hooks.SessionStart.hooks]]\n'
                  'type = "command"\ncommand = "python /truememory/ingest/hooks/session_start.py"\n'
                  'timeout = 10\n')
        self.write(module._CONFIG_PATH, before)
        adapter = module.CodexAdapter()
        adapter.install_hooks(python_path="/synthetic path/python", user_id="synthetic")
        first = module._CONFIG_PATH.read_text()
        self.assertIn('command = "external-helper"', first)
        hooks = adapter._parse_existing_hooks(first)
        for event in module._HOOK_EVENTS:
            commands = [hook["command"] for group in hooks[event] for hook in group["hooks"]
                        if "truememory" in hook.get("command", "")]
            self.assertEqual(len(commands), 1)
            parts = shlex.split(commands[0])
            self.assertEqual(parts[parts.index("--host") + 1], "codex")
            self.assertEqual(parts[parts.index("--event") + 1], event)
        adapter.install_hooks(python_path="/synthetic path/python", user_id="synthetic")
        self.assertEqual(module._CONFIG_PATH.read_text(), first)

    def test_gemini_installer_refreshes_identity_and_preserves_external_hook(self) -> None:
        module = source_module("truememory.hooks.adapters.gemini", "truememory/hooks/adapters/gemini.py")
        module._CONFIG_PATH = self.home / ".gemini/settings.json"
        external = {"hooks": [{"type": "command", "command": "external-helper"}]}
        self.write(module._CONFIG_PATH, json.dumps({"hooks": {"SessionStart": [external,
            {"command": "python /truememory/ingest/hooks/session_start.py", "timeout": 10000}]}}))
        adapter = module.GeminiAdapter()
        adapter.install_hooks(python_path="/synthetic path/python", user_id="synthetic")
        first = module._CONFIG_PATH.read_text()
        data = json.loads(first)
        self.assertEqual(data["hooks"]["SessionStart"][0], external)
        for event in module._HOOK_EVENTS:
            commands = [hook["command"] for group in data["hooks"][event] for hook in group["hooks"]
                        if "truememory" in hook["command"]]
            self.assertEqual(len(commands), 1)
            parts = shlex.split(commands[0])
            self.assertEqual(parts[parts.index("--host") + 1], "gemini")
            self.assertEqual(parts[parts.index("--event") + 1], event)
        adapter.install_hooks(python_path="/synthetic path/python", user_id="synthetic")
        self.assertEqual(module._CONFIG_PATH.read_text(), first)

    def test_codex_foreign_mentions_are_preserved_and_do_not_suppress_registration(self) -> None:
        module = source_module("truememory.hooks.adapters.codex", "truememory/hooks/adapters/codex.py")
        module._CONFIG_PATH = self.home / ".codex/config.toml"
        foreign_commands = (
            "/synthetic/foreign-policy --target truememory",
            "/synthetic/foreign-policy --target /pkg/truememory/ingest/hooks/session_start.py",
            "python /synthetic/foreign-policy.py --target truememory",
            "python -c 'print(\"truememory\")'",
            "echo python /pkg/truememory/ingest/hooks/session_start.py",
            "/synthetic/foreign-policy # python /pkg/truememory/ingest/hooks/session_start.py",
            "python /pkg/not-truememory/ingest/hooks/session_start.py",
            "python /pkg/truememory/ingest/hooks/session_start.py && /synthetic/foreign-policy",
            "python /pkg/truememory/session_start.py",
            "python /pkg/truememory/hooks/session_start.py",
        )
        for foreign in foreign_commands:
            with self.subTest(foreign=foreign):
                command_line = 'command = ' + json.dumps(foreign) + '\n'
                before = ('[[hooks.SessionStart]]\n[[hooks.SessionStart.hooks]]\n'
                          'type = "command"\n' + command_line + 'timeout = 21\n')
                self.write(module._CONFIG_PATH, before)
                adapter = module.CodexAdapter()
                adapter.install_hooks(python_path="/synthetic/python")
                first = module._CONFIG_PATH.read_text()
                self.assertIn(before, first)
                commands = [hook["command"] for group in adapter._parse_existing_hooks(first)["SessionStart"]
                            for hook in group["hooks"]]
                self.assertIn(foreign, commands)
                self.assertEqual(len(commands), 2)
                adapter.install_hooks(python_path="/synthetic/python")
                self.assertEqual(module._CONFIG_PATH.read_text(), first)

    def test_gemini_foreign_mentions_are_preserved_and_do_not_suppress_registration(self) -> None:
        module = source_module("truememory.hooks.adapters.gemini", "truememory/hooks/adapters/gemini.py")
        module._CONFIG_PATH = self.home / ".gemini/settings.json"
        foreign_commands = (
            "/synthetic/foreign-policy --target truememory",
            "/synthetic/foreign-policy --target /pkg/truememory/ingest/hooks/session_start.py",
            "python /synthetic/foreign-policy.py --target truememory",
            "python -c 'print(\"truememory\")'",
            "echo python /pkg/truememory/ingest/hooks/session_start.py",
            "/synthetic/foreign-policy # python /pkg/truememory/ingest/hooks/session_start.py",
            "python /pkg/not-truememory/ingest/hooks/session_start.py",
            "python /pkg/truememory/ingest/hooks/session_start.py && /synthetic/foreign-policy",
            "python /pkg/truememory/session_start.py",
            "python /pkg/truememory/hooks/session_start.py",
        )
        for foreign in foreign_commands:
            for nested in (True, False):
                with self.subTest(foreign=foreign, nested=nested):
                    hook = {"command": foreign, "timeout": 21}
                    definition = {"hooks": [hook]} if nested else hook
                    self.write(module._CONFIG_PATH, json.dumps({"hooks": {"SessionStart": [definition]}}))
                    adapter = module.GeminiAdapter()
                    adapter.install_hooks(python_path="/synthetic/python")
                    first = module._CONFIG_PATH.read_text()
                    definitions = json.loads(first)["hooks"]["SessionStart"]
                    self.assertEqual(definitions[0], definition)
                    self.assertEqual(len(definitions), 2)
                    adapter.install_hooks(python_path="/synthetic/python")
                    self.assertEqual(module._CONFIG_PATH.read_text(), first)

    def test_owned_hook_predicate_requires_actual_python_invocation(self) -> None:
        predicate = sys.modules["truememory.hooks.adapters.base"].is_owned_hook_command
        for platform in ("linux", "win32"):
            with patch.object(sys, "platform", platform):
                for stem in ("session_start", "user_prompt_submit", "stop", "compact"):
                    quoted_command = (
                        subprocess.list2cmdline([
                            r"C:\Program Files\Python\python.exe",
                            rf"C:\Synthetic Package\truememory\ingest\hooks\{stem}.py",
                            "--user", "synthetic user",
                        ]) if platform == "win32" else
                        f"'/pkg with space/python3.10' '/pkg with space/truememory/ingest/hooks/{stem}.py' --user ';'"
                    )
                    for command in (
                        f"python /pkg/truememory/ingest/hooks/{stem}.py",
                        quoted_command,
                        f"python -B -m truememory.ingest.hooks.{stem}",
                    ):
                        with self.subTest(platform=platform, command=command):
                            self.assertTrue(predicate(command))
                for command in (None, [], "", "python -c 'truememory.ingest.hooks.stop'",
                                "python -m foreign --target truememory.ingest.hooks.stop",
                                "foreign /pkg/truememory/ingest/hooks/stop.py", "python-policy /pkg/truememory/ingest/hooks/stop.py",
                                "python /pkg/truememory/ingest/hooks/stop.py; foreign", "python /pkg/truememory/ingest/hooks/stop.py\nforeign",
                                "python /pkg/truememory/ingest/hooks/stop.py > output", "python '/pkg/truememory/ingest/hooks/stop.py",
                                "python \"$(foreign)/truememory/ingest/hooks/stop.py\"", "python /pkg/truememory/ingest/hooks/not-a-hook.py"):
                    with self.subTest(platform=platform, command=command):
                        self.assertFalse(predicate(command))
        with patch.object(sys, "platform", "win32"):
            command = subprocess.list2cmdline([
                r"C:\Program Files\Python\python.exe",
                r"C:\Synthetic Package\truememory\ingest\hooks\stop.py",
                "--host", "codex", "--user", "synthetic user",
            ])
            self.assertTrue(predicate(command))
            self.assertFalse(predicate(r"C:\foreign.exe --target truememory"))

    def test_owned_module_invocations_are_upgraded_without_duplication(self) -> None:
        codex = source_module("truememory.hooks.adapters.codex", "truememory/hooks/adapters/codex.py")
        codex._CONFIG_PATH = self.home / ".codex/config.toml"
        self.write(codex._CONFIG_PATH, '[[hooks.SessionStart]]\n[[hooks.SessionStart.hooks]]\n'
                   'command = "python -m truememory.ingest.hooks.session_start"\n')
        adapter = codex.CodexAdapter()
        adapter.install_hooks(python_path="/synthetic/python")
        groups = adapter._parse_existing_hooks(codex._CONFIG_PATH.read_text())["SessionStart"]
        self.assertEqual(len(groups), 1)
        self.assertIn("--host codex --event SessionStart", groups[0]["hooks"][0]["command"])
        gemini = source_module("truememory.hooks.adapters.gemini", "truememory/hooks/adapters/gemini.py")
        gemini._CONFIG_PATH = self.home / ".gemini/settings.json"
        self.write(gemini._CONFIG_PATH, json.dumps({"hooks": {"SessionStart": [{"hooks": [
            {"command": "python -m truememory.ingest.hooks.session_start"}]}]}}))
        gemini.GeminiAdapter().install_hooks(python_path="/synthetic/python")
        groups = json.loads(gemini._CONFIG_PATH.read_text())["hooks"]["SessionStart"]
        self.assertEqual(len(groups), 1)
        self.assertIn("--host gemini --event SessionStart", groups[0]["hooks"][0]["command"])

    def test_cursor_extraction_uses_native_conversation_identity(self) -> None:
        os.environ["TRUEMEMORY_TRANSCRIPT_DIR"] = str(self.home / "synthetic-transcripts")
        path = self.write(self.home / "synthetic-transcripts/session.jsonl", json.dumps([
            {"role": "user", "content": "synthetic turn"} for _ in range(5)
        ]))
        for module, argv in ((self.stop, ["stop.py", "--host", "cursor", "--event", "stop"]),
                             (self.compact, ["compact.py", "--host", "cursor", "--event", "preCompact"])):
            for conversation in ("synthetic-cursor-a", "synthetic-cursor-b"):
                with self.subTest(module=module.__name__, conversation=conversation), \
                        patch.object(self.compact, "save_snapshot") as snapshot, \
                        patch.object(self.stop, "_run_background_ingestion", return_value=17) as spawn, \
                        patch.object(sys, "argv", argv), \
                        patch.object(sys, "stdin", io.StringIO(json.dumps({
                            "transcript_path": str(path), "conversation_id": conversation,
                        }))):
                    module.main()
                    self.assertEqual(spawn.call_args.args[1], conversation)
                    if module is self.compact:
                        self.assertEqual(snapshot.call_args.args[1], conversation)

    def test_hook_identity_is_host_scoped_and_preserves_each_caller_default(self) -> None:
        payload = {"conversation_id": "synthetic-native", "session_id": "synthetic-legacy", "host": "cursor"}
        self.assertEqual(self.shared.hook_session_id(payload, "cursor"), "synthetic-native")
        for host in ("claude", "codex", "gemini"):
            self.assertEqual(self.shared.hook_session_id(payload, host), "synthetic-legacy")
            self.assertEqual(self.shared.hook_session_id({"conversation_id": "synthetic-native"}, host), "unknown")
        for malformed in (None, {}, [], 42, "", " ", "bad\0id"):
            with self.subTest(malformed=malformed):
                payload = {"conversation_id": malformed, "session_id": "synthetic-legacy"}
                self.assertEqual(self.shared.hook_session_id(payload, "cursor"), "synthetic-legacy")
                self.assertEqual(self.shared.hook_session_id({"conversation_id": malformed}, "cursor"), "unknown")
                self.assertEqual(self.shared.hook_session_id({"conversation_id": malformed}, "cursor", default=""), "")
        for malformed in (None, [], 42, "text"):
            self.assertEqual(self.shared.hook_session_id(malformed, "cursor"), "unknown")
        self.assertEqual(self.shared.hook_session_id(payload, "invalid", default=""), "")

    def test_module_binding_scope_retains_unrelated_imports_and_restores_owned_keys(self) -> None:
        owned = "synthetic_owned_module_761"
        unrelated = "synthetic_later_import_761"
        previous = types.ModuleType(owned)
        replacement = types.ModuleType(owned)
        imported = types.ModuleType(unrelated)
        with scoped_module_bindings({owned: previous}, {owned, unrelated}):
            with scoped_module_bindings({owned: replacement}, {owned}):
                sys.modules[unrelated] = imported
                self.assertIs(sys.modules[owned], replacement)
            self.assertIs(sys.modules[owned], previous)
            self.assertIs(sys.modules[unrelated], imported)

    def test_spawn_fixtures_require_five_real_user_turns(self) -> None:
        path = self.native_path("gemini")
        self.write(path, json.dumps([{"role": "user", "content": "synthetic turn"}]))
        self.assertFalse(self.stop._has_enough_messages(str(path), 5))
        self.write(path, json.dumps([{"role": "user", "content": "synthetic turn"} for _ in range(5)]))
        self.assertTrue(self.stop._has_enough_messages(str(path), 5))


if __name__ == "__main__":
    unittest.main()
