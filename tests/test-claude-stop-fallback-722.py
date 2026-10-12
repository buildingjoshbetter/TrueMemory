"""Synthetic controls for supported Claude Stop fallback; no native host claim."""
from __future__ import annotations

import ast
import io
import json
import os
import shlex
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def adapter_source() -> dict:
    source = ROOT / "truememory/hooks/adapters/claude.py"
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_HOOK_MODULES" for target in node.targets)]
    namespace = {"__file__": str(source), "json": json, "os": os, "sys": sys, "Path": Path, "CLIAdapter": object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])), str(source), "exec"), namespace)
    return namespace


ADAPTER = adapter_source()


class InertStopFallbackControls(unittest.TestCase):
    def test_default_sessionend_and_explicit_opt_in(self) -> None:
        build = ADAPTER["claude_hook_settings"]
        with patch.dict(os.environ, {}, clear=True):
            default = build("/synthetic/python3", "synthetic-account", "/synthetic/db")
        self.assertIn("SessionEnd", default["hooks"])
        self.assertNotIn("Stop", default["hooks"])
        for value in ("0", "true", "yes", "garbage"):
            with patch.dict(os.environ, {"TRUEMEMORY_CLAUDE_STOP_FALLBACK": value}, clear=True):
                self.assertNotIn("Stop", build("python3")["hooks"])
        with patch.dict(os.environ, {"TRUEMEMORY_CLAUDE_STOP_FALLBACK": "1"}, clear=True):
            fallback = build("python3", "synthetic-account", "/synthetic/db")
        self.assertIn("SessionEnd", fallback["hooks"])
        self.assertIn("Stop", fallback["hooks"])
        for event, groups in fallback["hooks"].items():
            argv = shlex.split(groups[0]["hooks"][0]["command"])
            self.assertEqual(argv[argv.index("--host") + 1], "claude")
            self.assertEqual(argv[argv.index("--event") + 1], event)
            self.assertEqual(argv[argv.index("--user") + 1], "synthetic-account")
            self.assertEqual(argv[argv.index("--db") + 1], "/synthetic/db")

    def test_exact_module_and_script_ownership(self) -> None:
        owned = ADAPTER["_owned_hook_module"]
        directory = ROOT / "truememory/ingest/hooks"
        module = "truememory.ingest.hooks.stop"
        self.assertEqual(owned("python3 -m " + module, directory, "python3"), module)
        self.assertEqual(owned("python3 " + shlex.quote(str(directory / "stop.py")), directory, "python3"), module)
        for command in ("echo truememory", "echo -m " + module,
                        "python3 -m " + module + ".foreign", "python3 -m " + module + " && echo x",
                        "python3 /foreign/truememory/stop.py", "python3 -m foreign --name truememory"):
            self.assertIsNone(owned(command, directory, "python3"))

    def test_recursion_guards_do_not_read_source_or_mark_completion(self) -> None:
        source = ROOT / "truememory/ingest/hooks/stop.py"
        node = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = []
        namespace = {"_parse_args": lambda: types.SimpleNamespace(host="claude", user="", db=""),
                     "os": os, "sys": sys, "json": json, "Path": lambda *args: calls.append("source")}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), namespace)
        with patch.dict(os.environ, {"TRUEMEMORY_EXTRACTION": "1"}, clear=True), \
                patch.object(sys, "stdin", io.StringIO("not JSON")):
            namespace["main"]()
        self.assertEqual(calls, [])
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(sys, "stdin", io.StringIO(json.dumps({"stop_hook_active": True, "transcript_path": "/synthetic/source"}))):
            namespace["main"]()
        self.assertEqual(calls, [])

    def test_windows_builder_preserves_fixed_identity_and_quotes(self) -> None:
        with patch.object(sys, "platform", "win32"), patch.dict(os.environ, {}, clear=True):
            settings = ADAPTER["claude_hook_settings"]("C:/Synthetic Python/python.exe", "synthetic account", "C:/Synthetic DB/db")
        command = settings["hooks"]["SessionEnd"][0]["hooks"][0]["command"]
        self.assertIn('"C:/Synthetic Python/python.exe"', command)
        self.assertIn("--host claude --event SessionEnd", command)
        self.assertIn('"synthetic account"', command)
        with patch.object(sys, "platform", "win32"):
            owned = ADAPTER["_owned_hook_module"](command, ROOT / "truememory/ingest/hooks", "C:/Synthetic Python/python.exe")
        self.assertEqual(owned, "truememory.ingest.hooks.stop")


class FilesystemStopFallbackControls(unittest.TestCase):
    def test_install_preserves_foreign_and_viable_stop_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tm-synthetic-stop-") as temporary:
            target = Path(temporary) / "settings.json"
            foreign = {"type": "command", "command": "echo truememory foreign", "timeout": 8}
            viable = {"type": "command", "command": "python3 -m truememory.ingest.hooks.stop", "timeout": 17}
            target.write_text(json.dumps({"other_setting": "retained", "hooks": {"Stop": [foreign, viable]}}))
            adapter_type = ADAPTER["ClaudeAdapter"]
            with patch.object(adapter_type, "config_path", new=property(lambda self: target)), \
                    patch.dict(os.environ, {}, clear=True):
                adapter = adapter_type()
                adapter.install_hooks("python3", "synthetic-account", "/synthetic/db")
                first = json.loads(target.read_text())
                adapter.install_hooks("python3", "synthetic-account", "/synthetic/db")
                second = json.loads(target.read_text())
            self.assertEqual(first, second)
            self.assertEqual(second["other_setting"], "retained")
            commands = [hook for group in second["hooks"]["Stop"] for hook in group["hooks"]]
            self.assertIn(foreign, commands)
            owned = [hook for hook in commands if "-m truememory.ingest.hooks.stop" in hook["command"]]
            self.assertEqual(len(owned), 1)
            self.assertEqual(owned[0]["timeout"], 17)
            self.assertIn("--host claude --event Stop", owned[0]["command"])
            self.assertIn("SessionEnd", second["hooks"])

    def test_foreign_substring_does_not_suppress_owned_event(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tm-synthetic-foreign-") as temporary:
            target = Path(temporary) / "settings.json"
            foreign = {"matcher": "", "hooks": [{"type": "command", "command": "echo truememory"}]}
            target.write_text(json.dumps({"hooks": {"SessionEnd": [foreign], "Stop": [foreign]}}))
            with patch.object(ADAPTER["ClaudeAdapter"], "config_path", new=property(lambda self: target)), \
                    patch.dict(os.environ, {}, clear=True):
                ADAPTER["ClaudeAdapter"]().install_hooks("python3")
            settings = json.loads(target.read_text())
            self.assertEqual(settings["hooks"]["Stop"], [foreign])
            self.assertIn(foreign, settings["hooks"]["SessionEnd"])
            self.assertEqual(len(settings["hooks"]["SessionEnd"]), 2)


if __name__ == "__main__":
    unittest.main()
