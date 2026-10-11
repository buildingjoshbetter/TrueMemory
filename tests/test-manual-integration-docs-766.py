"""Manual setup parity with actual constructors, isolated files and stdlib only.

No TrueMemory package, native host, hook script, model or database is loaded.
Fixture directories are retained, including any failed installation output.
Hermes's YAML codec is stubbed with JSON (valid YAML); its documented YAML is
parsed separately with a deliberately restricted, fail-closed subset parser.
"""
from __future__ import annotations

import ast
import copy
import io
import json
import os
import re
import shlex
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Protocol, cast
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


ROOT = Path(__file__).resolve().parents[1]
EVENTS: dict[str, dict[str, tuple[str, int]]] = {
    "codex": {"SessionStart": ("session_start.py", 10), "Stop": ("stop.py", 5),
              "UserPromptSubmit": ("user_prompt_submit.py", 5), "PreCompact": ("compact.py", 5)},
    "cursor": {"sessionStart": ("session_start.py", 10), "stop": ("stop.py", 5),
               "beforeSubmitPrompt": ("user_prompt_submit.py", 5), "preCompact": ("compact.py", 5)},
    "gemini": {"SessionStart": ("session_start.py", 10000), "SessionEnd": ("stop.py", 5000),
               "BeforeAgent": ("user_prompt_submit.py", 5000), "PreCompress": ("compact.py", 5000)},
    "kimi": {"SessionStart": ("session_start.py", 10), "Stop": ("stop.py", 5),
             "UserPromptSubmit": ("user_prompt_submit.py", 5), "PreCompact": ("compact.py", 5)},
    "hermes": {"on_session_start": ("session_start.py", 30), "on_session_end": ("stop.py", 30),
               "pre_llm_call": ("user_prompt_submit.py", 10), "on_session_finalize": ("compact.py", 15)},
}
CLASS_NAMES = {"codex": "CodexAdapter", "cursor": "CursorAdapter", "gemini": "GeminiAdapter",
               "kimi": "KimiAdapter", "hermes": "HermesAdapter", "openclaw": "OpenClawAdapter"}


class Adapter(Protocol):
    def install_mcp(self, python_path: str | None = None) -> None: ...
    def install_hooks(self, python_path: str | None = None, user_id: str = "", db_path: str = "") -> None: ...
    def detect(self) -> bool: ...


def fenced(text: str, language: str) -> list[str]:
    return re.findall(rf"^```{re.escape(language)}\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL)


def toml_subset(text: str) -> dict[str, object]:
    """Parse the documented table/array-table and JSON-compatible scalar subset."""
    result: dict[str, object] = {}
    current = result
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            array_table = line.startswith("[[")
            bracket = "[[" if array_table else "["
            closing = "]]" if array_table else "]"
            if not line.endswith(closing):
                raise ValueError(f"Invalid TOML section: {line}")
            names = line[len(bracket):-len(closing)].split(".")
            if not all(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) for name in names):
                raise ValueError(f"Unsupported TOML section: {line}")
            current = result
            for index, name in enumerate(names):
                if index == len(names) - 1 and array_table:
                    collection = current.setdefault(name, [])
                    if not isinstance(collection, list):
                        raise ValueError(f"Conflicting TOML collection: {line}")
                    entry: dict[str, object] = {}
                    collection.append(entry)
                    current = entry
                else:
                    child = current.setdefault(name, {})
                    if isinstance(child, list) and child:
                        child = child[-1]
                    if not isinstance(child, dict):
                        raise ValueError(f"Conflicting TOML section: {line}")
                    current = child
        else:
            match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.+)", line)
            if match is None or match.group(1) in current:
                raise ValueError(f"Unsupported or duplicate TOML field: {line}")
            current[match.group(1)] = json.loads(match.group(2))
    return result


def parse_toml(text: str) -> dict[str, object]:
    return tomllib.loads(text) if tomllib is not None else toml_subset(text)


def yaml_subset(text: str) -> dict[str, object]:
    """Parse only indentation maps/lists and scalars used by the Hermes guide."""
    if "\t" in text:
        raise ValueError("YAML tabs are unsupported")
    rows = [(len(line) - len(line.lstrip()), line.strip())
            for line in text.splitlines() if line.strip()]

    def scalar(value: str) -> object:
        if re.fullmatch(r"[0-9]+", value):
            return int(value)
        if not value or any(token in value for token in ("&", "*", "{", "[", "#", "\"", "'")):
            raise ValueError(f"Unsupported YAML scalar: {value!r}")
        return value

    def pair(value: str) -> tuple[str, str]:
        match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*):(?: (.*))?", value)
        if match is None:
            raise ValueError(f"Unsupported YAML mapping: {value!r}")
        return match.group(1), match.group(2) or ""

    def block(index: int, indent: int) -> tuple[object, int]:
        is_list = rows[index][1].startswith("- ")
        result: object = [] if is_list else {}
        while index < len(rows) and rows[index][0] == indent:
            line = rows[index][1]
            if is_list:
                if not line.startswith("- "):
                    raise ValueError("Mixed YAML collection")
                item = line[2:]
                if re.match(r"[A-Za-z_][A-Za-z_0-9]*:", item):
                    key, value = pair(item)
                    entry = {key: scalar(value)}
                    index += 1
                    if index < len(rows) and rows[index][0] > indent:
                        extra, index = block(index, rows[index][0])
                        if not isinstance(extra, dict) or key in extra:
                            raise ValueError("Invalid YAML list mapping")
                        entry.update(extra)
                    cast(list[object], result).append(entry)
                else:
                    cast(list[object], result).append(scalar(item))
                    index += 1
            else:
                key, value = pair(line)
                target = cast(dict[str, object], result)
                if key in target:
                    raise ValueError(f"Duplicate YAML key: {key}")
                index += 1
                if value:
                    target[key] = scalar(value)
                elif index < len(rows) and rows[index][0] > indent:
                    target[key], index = block(index, rows[index][0])
                else:
                    raise ValueError(f"Missing YAML value: {key}")
        return result, index

    if not rows or rows[0][0] != 0:
        raise ValueError("Expected root YAML map")
    result, consumed = block(0, 0)
    if consumed != len(rows) or not isinstance(result, dict):
        raise ValueError("Unsupported YAML structure")
    return result


def merge(base: dict[str, object], additions: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(base)
    for key, value in additions.items():
        current = result.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            result[key] = merge(current, value)
        elif isinstance(value, list) and isinstance(current, list):
            result[key] = current + copy.deepcopy(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_adapter(source: Path, home: Path) -> Adapter:
    """Execute actual stdlib-only adapter source with the shared base stubbed."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom)
                         and (node.module or "").startswith("truememory"))]
    base_source = ROOT / "truememory/hooks/adapters/base.py"
    base_tree = ast.parse(base_source.read_text(encoding="utf-8"))
    atomic_node = next(node for node in base_tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "atomic_write_text")
    namespace: dict[str, object] = {"Path": Path, "os": os, "CLIAdapter": object,
                                    "__file__": str(source), "__name__": "synthetic_docs766"}
    exec(compile(ast.Module(body=[atomic_node], type_ignores=[]), str(base_source), "exec"), namespace)
    with patch("pathlib.Path.home", return_value=home):
        exec(compile(tree, str(source), "exec"), namespace)
    if source.stem == "hermes":
        namespace["_yaml_safe_load"] = json.loads
        namespace["_yaml_safe_dump"] = json.dumps
    adapter_class = cast(type[Adapter], namespace[CLASS_NAMES[source.stem]])
    return adapter_class()


def canonical(value: object) -> object:
    if isinstance(value, list):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        result = {key: canonical(item) for key, item in value.items()}
        command = value.get("command")
        if isinstance(command, str) and "ingest/hooks/" in command:
            parts = shlex.split(command)
            result["command"] = ["PYTHON", Path(parts[1]).name, *parts[2:]]
        return result
    return value


class ManualSetupParityTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture_root = Path(os.environ.get("TRUEMEMORY_DOCS766_FIXTURE_ROOT",
                                          str(ROOT / ".work/fixtures/manual-integration-docs-766-v1")))
        fixture_root.mkdir(parents=True, exist_ok=True)
        self.case = Path(tempfile.mkdtemp(prefix=f"{self._testMethodName}-", dir=fixture_root))
        self.home = self.case / "adapter-home"
        self.home.mkdir()
        self.python = self.case / "bin/python"
        self.python.parent.mkdir()
        self.python.write_text("#!/bin/sh\nexit 71\n", encoding="utf-8")
        self.python.chmod(0o700)

    def document(self, host: str) -> str:
        return (ROOT / f"docs/setup-{host}.md").read_text(encoding="utf-8").replace(
            "/path/to/python", str(self.python)).replace("/path/to/truememory", str(ROOT / "truememory"))

    def install(self, host: str, paths: dict[str, str]) -> Adapter:
        for relative, text in paths.items():
            target = self.home / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        executable = self.python.parent / host
        executable.write_text("#!/bin/sh\nexit 71\n", encoding="utf-8")
        executable.chmod(0o700)
        adapter = load_adapter(ROOT / f"truememory/hooks/adapters/{host}.py", self.home)
        with patch("shutil.which", return_value=str(executable)):
            self.assertTrue(adapter.detect())
        adapter.install_mcp(str(self.python))
        adapter.install_hooks(str(self.python))
        return adapter

    def assert_contract(self, host: str, additions: dict[str, object]) -> None:
        hooks = additions["hooks"]
        if host == "kimi":
            self.assertIsInstance(hooks, list)
            entries = {entry["event"]: entry for entry in cast(list[dict[str, object]], hooks)}
        else:
            self.assertIsInstance(hooks, dict)
            entries = {}
            for event, groups in cast(dict[str, list[dict[str, object]]], hooks).items():
                self.assertEqual(len(groups), 1)
                entry = groups[0]
                if host in {"codex", "gemini"}:
                    self.assertEqual(set(entry), {"hooks"})
                    self.assertEqual(len(cast(list[object], entry["hooks"])), 1)
                    entry = cast(list[dict[str, object]], entry["hooks"])[0]
                    self.assertEqual(entry["type"], "command")
                entries[event] = entry
        self.assertEqual(set(entries), set(EVENTS[host]))
        for event, (script, timeout) in EVENTS[host].items():
            entry = entries[event]
            self.assertEqual(entry["timeout"], timeout)
            tokens = shlex.split(cast(str, entry["command"]))
            self.assertEqual(tokens[:2], [str(self.python), str(ROOT / "truememory/ingest/hooks" / script)])
            if host in {"codex", "cursor", "gemini"}:
                self.assertEqual(tokens[2:], ["--host", host, "--event", event])
            else:
                self.assertEqual(tokens[2:], [])

    def assert_preserved(self, host: str, additions: dict[str, object], generated: dict[str, object],
                         seed: dict[str, object]) -> None:
        self.assert_contract(host, additions)
        expected = merge(seed, additions)
        manual = self.case / "manual-config.json"
        manual.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        self.assertEqual(canonical(json.loads(manual.read_text())), canonical(generated))
        self.assertEqual(generated["unrelated"], seed["unrelated"])

    def test_codex_nested_toml_groups_seconds_and_identity(self) -> None:
        additions = parse_toml(fenced(self.document("codex"), "toml")[0])
        seed_text = ('unrelated = "keep"\n[mcp_servers.other]\ncommand = "/usr/bin/printf"\n'
                     '[[hooks.SessionStart]]\n[[hooks.SessionStart.hooks]]\ntype = "command"\n'
                     'command = "/usr/bin/printf synthetic-other"\ntimeout = 37\n')
        self.install("codex", {".codex/config.toml": seed_text})
        generated = parse_toml((self.home / ".codex/config.toml").read_text())
        self.assert_preserved("codex", additions, generated, parse_toml(seed_text))

    def test_cursor_separate_json_seconds_and_identity(self) -> None:
        mcp, hooks = [json.loads(block) for block in fenced(self.document("cursor"), "json")]
        self.assertEqual(hooks["version"], 1)
        mcp_seed = {"unrelated": "keep", "mcpServers": {"other": {"command": "/usr/bin/printf"}}}
        hook_seed = {"unrelated": "keep", "version": 1, "hooks": {
            "sessionStart": [{"command": "/usr/bin/printf synthetic-other", "timeout": 37}]}}
        self.install("cursor", {".cursor/mcp.json": json.dumps(mcp_seed),
                                ".cursor/hooks.json": json.dumps(hook_seed)})
        self.assertEqual(json.loads((self.home / ".cursor/mcp.json").read_text()), merge(mcp_seed, mcp))
        self.assert_preserved("cursor", hooks, json.loads((self.home / ".cursor/hooks.json").read_text()),
                              hook_seed)
        text = self.document("cursor")
        self.assertIn("does not inject per-prompt recall", text)
        self.assertIn("additional_context", text)

    def test_gemini_nested_json_milliseconds_and_identity(self) -> None:
        additions = json.loads(fenced(self.document("gemini"), "json")[0])
        seed = {"unrelated": "keep", "mcpServers": {"other": {"command": "/usr/bin/printf"}},
                "hooks": {"SessionStart": [{"hooks": [{"type": "command", "name": "other",
                    "command": "/usr/bin/printf synthetic-other", "timeout": 37}]}]}}
        self.install("gemini", {".gemini/settings.json": json.dumps(seed)})
        self.assert_preserved("gemini", additions,
                              json.loads((self.home / ".gemini/settings.json").read_text()), seed)

    def test_kimi_flat_toml_seconds_and_legacy_home(self) -> None:
        document = self.document("kimi")
        mcp = json.loads(fenced(document, "json")[0])
        additions = parse_toml(fenced(document, "toml")[0])
        seed_text = ('unrelated = "keep"\n[[hooks]]\nevent = "SessionStart"\n'
                     'command = "/usr/bin/printf synthetic-other"\ntimeout = 37\n')
        mcp_seed = {"unrelated": "keep", "mcpServers": {"other": {"command": "/usr/bin/printf"}}}
        self.install("kimi", {".kimi/config.toml": seed_text, ".kimi/mcp.json": json.dumps(mcp_seed)})
        self.assertEqual(json.loads((self.home / ".kimi/mcp.json").read_text()), merge(mcp_seed, mcp))
        self.assert_preserved("kimi", additions,
                              parse_toml((self.home / ".kimi/config.toml").read_text()),
                              parse_toml(seed_text))
        self.assertIn("~/.kimi-code/", document)
        self.assertIn("unverified", document)

    def test_hermes_same_yaml_file_hooks_map_seconds(self) -> None:
        document = self.document("hermes")
        mcp, hooks = [yaml_subset(block) for block in fenced(document, "yaml")]
        additions = merge(mcp, hooks)
        seed = {"unrelated": "keep", "mcp_servers": {"other": {"command": "/usr/bin/printf"}},
                "hooks": {"on_session_start": [{"command": "/usr/bin/printf synthetic-other",
                                                  "timeout": 37}]}}
        self.install("hermes", {".hermes/config.yaml": json.dumps(seed)})
        self.assert_preserved("hermes", additions,
                              json.loads((self.home / ".hermes/config.yaml").read_text()), seed)
        self.assertFalse((self.home / ".hermes/cli-config.yaml").exists())
        self.assertNotIn("cli-config.yaml", document)
        self.assertIn("continuations and retries", document)

    def test_openclaw_copy_package_discovery_and_known_adapter_divergence(self) -> None:
        document = self.document("openclaw")
        additions = json.loads(fenced(document, "json")[0])
        seed = {"unrelated": "keep", "mcp": {"servers": {"other": {"command": "/usr/bin/printf"}}},
                "plugins": {"allow": ["other"], "entries": {"other": {"enabled": False}}}}
        self.install("openclaw", {".openclaw/openclaw.json": json.dumps(seed)})
        generated = json.loads((self.home / ".openclaw/openclaw.json").read_text())
        self.assertEqual(generated, merge(seed, additions))
        shell = next(block for block in fenced(document, "bash") if "<<'PY'" in block)
        snippet = shell.split("\n", 1)[1].rsplit("\nPY", 1)[0]
        tree = ast.parse(snippet)
        for node in tree.body:
            if isinstance(node, ast.Import):
                node.names = [alias for alias in node.names if alias.name != "truememory"]
        tree.body = [node for node in tree.body if not isinstance(node, ast.Import) or node.names]
        namespace: dict[str, object] = {"truememory": type("SyntheticPackage", (), {
            "__file__": str(ROOT / "truememory/__init__.py")})}
        manual_home = self.case / "manual-home"
        with patch("pathlib.Path.home", return_value=manual_home), redirect_stdout(io.StringIO()):
            exec(compile(tree, "documented-openclaw-copy", "exec"), namespace)
        package = manual_home / ".openclaw/extensions/truememory"
        files = {"openclaw.plugin.json", "package.json", "index.js"}
        self.assertEqual({path.name for path in package.iterdir()}, files)
        manifest = json.loads((package / "openclaw.plugin.json").read_text())
        metadata = json.loads((package / "package.json").read_text())
        self.assertEqual(manifest["id"], "truememory")
        self.assertEqual(metadata["type"], "module")
        self.assertEqual(metadata["openclaw"]["extensions"], ["./index.js"])
        template = ROOT / "truememory/hooks/templates/openclaw"
        for filename in files:
            self.assertEqual((package / filename).read_bytes(), (template / filename).read_bytes())
            self.assertTrue((self.home / ".openclaw/plugins/truememory" / filename).is_file())
        before = {name: (package / name).read_bytes() for name in files}
        with patch("pathlib.Path.home", return_value=manual_home), self.assertRaises(FileExistsError):
            exec(compile(tree, "documented-openclaw-copy", "exec"), namespace)
        self.assertEqual(before, {name: (package / name).read_bytes() for name in files})
        self.assertIn("adapter path and plugin runtime remain tracked", document)
        self.assertIn("does not verify the manual", document)

    def test_compatibility_distinguishes_config_from_native_acceptance(self) -> None:
        text = (ROOT / "docs/compatibility.md").read_text(encoding="utf-8")
        self.assertIn("Native tested versions", text)
        self.assertIn("none in this parity validation", text)
        self.assertIn("per-prompt recall injection is unsupported", text)
        self.assertIn("10000 / 60 = 166.67", text)
        self.assertIn("5000 / 60 = 83.33", text)
        self.assertIn("10000 / 1000 = 10", text)
        self.assertNotIn("~/.hermes/cli-config.yaml", text)
        self.assertNotIn("before_agent_run", text)
        self.assertNotIn("agent_end", text)

    def test_restricted_parsers_reject_obsolete_or_conflicting_structures(self) -> None:
        for host in ("codex", "kimi"):
            snippet = fenced(self.document(host), "toml")[0]
            self.assertEqual(toml_subset(snippet), parse_toml(snippet))
        with self.assertRaises(AssertionError):
            self.assert_contract("codex", {"hooks": [{"event": "SessionStart"}]})
        with self.assertRaises(AssertionError):
            self.assert_contract("gemini", {"hooks": {"SessionStart": [{"command": "old", "timeout": 10000}]}})
        with self.assertRaises(ValueError):
            toml_subset('hooks = 4\n[[hooks.SessionStart]]\n')
        with self.assertRaises(ValueError):
            yaml_subset("hooks:\n  on_session_start:\n    - command: one\n      command: two\n")
        with self.assertRaises(ValueError):
            yaml_subset("hooks:\n  on_session_start: *anchor\n")


if __name__ == "__main__":
    unittest.main()
