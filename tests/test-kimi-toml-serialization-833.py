"""Synthetic Kimi command/TOML roundtrips using actual adapter and writer AST."""
from __future__ import annotations

import ast
import __future__
import builtins
import json
import logging
import shlex
import subprocess
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from typing import Iterator, Protocol, cast
from unittest.mock import patch

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None

ROOT = Path(__file__).resolve().parents[1]
KIMI_SOURCE = ROOT / "truememory/hooks/adapters/kimi.py"
BASE_SOURCE = ROOT / "truememory/hooks/adapters/base.py"
EVENTS = {
    "SessionStart": ("session_start.py", 10),
    "Stop": ("stop.py", 5),
    "UserPromptSubmit": ("user_prompt_submit.py", 5),
    "PreCompact": ("compact.py", 5),
}
FOREIGN_TOML = (
    '# Synthetic foreign settings\ntheme = "plain"\n\n'
    '[[hooks]]\nevent = "SessionStart"\ncommand = "printf foreign-hook"\n'
    'timeout = 37\n\n[features]\ntelemetry = false\n'
)
FOREIGN_MCP = {"mcpServers": {"foreign": {"command": "synthetic-foreign"}}, "theme": "plain"}


@dataclass
class MemoryFS:
    platform: str
    files: dict[str, str] = field(default_factory=dict)
    directories: set[str] = field(default_factory=set)
    replace_calls: int = 0
    fail_next_replace: bool = False

    def path(self, value: str | MemoryPath) -> MemoryPath:
        if isinstance(value, MemoryPath):
            if value.fs is not self:
                raise AssertionError("Path belongs to another synthetic filesystem")
            return value
        pure = PureWindowsPath(value) if self.platform == "win32" else PurePosixPath(value)
        return MemoryPath(self, pure)

    def replace(self, source: MemoryPath, destination: MemoryPath) -> None:
        self.replace_calls += 1
        if self.fail_next_replace:
            self.fail_next_replace = False
            raise OSError("Synthetic atomic replace failure")
        self.files[str(destination)] = self.files.pop(str(source))


@dataclass(frozen=True)
class MemoryPath:
    fs: MemoryFS
    pure: PurePath

    def __str__(self) -> str:
        return str(self.pure)

    def __truediv__(self, component: str) -> MemoryPath:
        return MemoryPath(self.fs, self.pure / component)

    @property
    def parent(self) -> MemoryPath:
        return MemoryPath(self.fs, self.pure.parent)

    @property
    def name(self) -> str:
        return self.pure.name

    def with_name(self, name: str) -> MemoryPath:
        return MemoryPath(self.fs, self.pure.with_name(name))

    def mkdir(self, *, parents: bool = False, exist_ok: bool = False) -> None:
        if not parents or not exist_ok:
            raise AssertionError("Unexpected synthetic mkdir arguments")
        self.fs.directories.add(str(self))

    def exists(self) -> bool:
        return str(self) in self.fs.files or str(self) in self.fs.directories

    def is_dir(self) -> bool:
        return str(self) in self.fs.directories

    def read_text(self, *, encoding: str) -> str:
        if encoding != "utf-8":
            raise AssertionError("Unexpected encoding")
        try:
            return self.fs.files[str(self)]
        except KeyError as error:
            raise FileNotFoundError(str(self)) from error

    def write_text(self, text: str, *, encoding: str) -> int:
        if encoding != "utf-8":
            raise AssertionError("Unexpected encoding")
        text.encode(encoding)
        self.fs.files[str(self)] = text
        return len(text)

    def unlink(self) -> None:
        try:
            del self.fs.files[str(self)]
        except KeyError as error:
            raise FileNotFoundError(str(self)) from error


@dataclass
class MemoryPathFactory:
    fs: MemoryFS

    def __call__(self, value: str | MemoryPath) -> MemoryPath:
        return self.fs.path(value)

    def home(self) -> MemoryPath:
        home = r"C:\synthetic-home" if self.fs.platform == "win32" else "/synthetic-home"
        return self.fs.path(home)


class HookAdapter(Protocol):
    def install_hooks(self, python_path: str, user_id: str = "", db_path: str = "") -> None: ...
    def install_mcp(self, python_path: str) -> None: ...
    def verify(self) -> bool: ...
    def uninstall(self) -> None: ...
    def _parse_existing_hooks(self, text: str) -> list[dict]: ...


@dataclass
class Fixture:
    fs: MemoryFS
    adapter: HookAdapter
    hook_config: MemoryPath
    mcp_config: MemoryPath
    hooks_dir: MemoryPath

    def text(self) -> str:
        return self.hook_config.read_text(encoding="utf-8")


def load_fixture(platform: str) -> Fixture:
    """Execute source AST with memory-only paths and the actual atomic writer."""
    fs = MemoryFS(platform)
    path_factory = MemoryPathFactory(fs)
    source_file = (
        r'C:\synthetic modules "quoted"\truememory\hooks\adapters\kimi.py'
        if platform == "win32"
        else '/synthetic modules "quoted"/truememory/hooks/adapters/kimi.py'
    )
    namespace: dict[str, object] = {
        "__name__": "synthetic_kimi_adapter",
        "__file__": source_file,
        "Path": path_factory,
        "CLIAdapter": object,
        "json": json,
        "logging": logging,
        "shlex": shlex,
        "shutil": SimpleNamespace(which=lambda executable: None),
        "sys": SimpleNamespace(platform=platform, executable="synthetic-python"),
        "os": SimpleNamespace(getpid=lambda: 833, replace=fs.replace),
    }
    base_tree = ast.parse(BASE_SOURCE.read_text(encoding="utf-8"))
    writer = next(
        node for node in base_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "atomic_write_text"
    )
    writer_module = ast.Module(body=[writer], type_ignores=[])
    exec(compile(writer_module, str(BASE_SOURCE), "exec", flags=__future__.annotations.compiler_flag), namespace)
    adapter_tree = ast.parse(KIMI_SOURCE.read_text(encoding="utf-8"))
    adapter_tree.body = [
        node for node in adapter_tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    exec(compile(adapter_tree, str(KIMI_SOURCE), "exec", flags=__future__.annotations.compiler_flag), namespace)
    adapter_type = namespace["KimiAdapter"]
    if not isinstance(adapter_type, type):
        raise AssertionError("Actual Kimi adapter class was not loaded")
    hook_config = cast(MemoryPath, namespace["_HOOK_CONFIG"])
    mcp_config = cast(MemoryPath, namespace["_MCP_CONFIG"])
    fs.files[str(hook_config)] = FOREIGN_TOML
    fs.files[str(mcp_config)] = json.dumps(FOREIGN_MCP)
    hooks_dir = fs.path(source_file).parent.parent.parent / "ingest" / "hooks"
    return Fixture(fs, cast(HookAdapter, adapter_type()), hook_config, mcp_config, hooks_dir)


@contextmanager
def scoped_imports(*, no_tomllib: bool) -> Iterator[None]:
    original_import = builtins.__import__

    def checked_import(
        name: str, globals: dict | None = None, locals: dict | None = None,
        fromlist: tuple | list = (), level: int = 0,
    ) -> object:
        if name == "tomllib" and no_tomllib:
            raise ModuleNotFoundError("Synthetic Python 3.10 tomllib absence")
        if name == "truememory" or name.startswith("truememory."):
            raise AssertionError("Actual application import is forbidden in this fixture")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=checked_import):
        yield


def expected_commands(fixture: Fixture, python: str, user: str, db: str) -> list[str]:
    commands: list[str] = []
    for script, _ in EVENTS.values():
        parts = [python, str(fixture.hooks_dir / script)]
        if user:
            parts.extend(["--user", user])
        if db:
            parts.extend(["--db", db])
        commands.append(
            subprocess.list2cmdline(parts) if fixture.fs.platform == "win32"
            else " ".join(shlex.quote(part) for part in parts)
        )
    return commands


def strict_generated_commands(text: str) -> list[str]:
    """Validate emitted JSON-compatible TOML scalars without a TOML dependency."""
    commands: list[str] = []
    for line in text.split("\n"):
        if not line.startswith("command = "):
            continue
        scalar = line[len("command = "):]
        if any(ord(char) < 0x20 or ord(char) == 0x7F for char in scalar):
            raise AssertionError("Raw TOML-forbidden control character")
        decoded = json.loads(scalar)
        if not isinstance(decoded, str):
            raise AssertionError("Generated command must be a string")
        if any(0xD800 <= ord(char) <= 0xDFFF for char in decoded):
            raise AssertionError("Surrogate is not a TOML Unicode scalar")
        commands.append(decoded)
    return commands


class TestKimiTomlSerialization(unittest.TestCase):
    def assert_roundtrip(
        self, fixture: Fixture, python: str, user: str, db: str, *, fallback: bool,
    ) -> None:
        expected = expected_commands(fixture, python, user, db)
        with scoped_imports(no_tomllib=fallback):
            fixture.adapter.install_hooks(python, user, db)
            hooks = fixture.adapter._parse_existing_hooks(fixture.text())
            installed = fixture.text()
            replacements = fixture.fs.replace_calls
            fixture.adapter.install_hooks(python, user, db)
            self.assertEqual(fixture.text(), installed)
            self.assertEqual(fixture.fs.replace_calls, replacements)
        self.assertEqual([hook["command"] for hook in hooks[1:]], expected)
        self.assertEqual([hook["event"] for hook in hooks[1:]], list(EVENTS))
        self.assertEqual([int(hook["timeout"]) for hook in hooks[1:]], [10, 5, 5, 5])
        self.assertEqual(strict_generated_commands(fixture.text())[1:], expected)
        self.assertTrue(fixture.text().startswith(FOREIGN_TOML.rstrip() + "\n"))
        if tomllib is not None:
            parsed = tomllib.loads(fixture.text())
            self.assertEqual([hook["command"] for hook in parsed["hooks"][1:]], expected)
            self.assertEqual([hook["timeout"] for hook in parsed["hooks"][1:]], [10, 5, 5, 5])
            self.assertEqual(parsed["features"], {"telemetry": False})

    def test_native_and_fallback_commands_preserve_platform_quoting(self) -> None:
        for platform in ("linux", "win32"):
            for fallback in (False, True):
                with self.subTest(platform=platform, fallback=fallback):
                    fixture = load_fixture(platform)
                    python = r'C:\Synthetic Python "quoted"\python.exe' if platform == "win32" else '/synthetic Python "quoted"/python'
                    self.assert_roundtrip(
                        fixture, python, 'synthetic user "quoted" \\ trailing\\',
                        r'C:\synthetic db "quoted"\store.db' if platform == "win32" else '/synthetic db "quoted"/store.db',
                        fallback=fallback,
                    )

    def test_all_control_and_unicode_characters_roundtrip(self) -> None:
        characters = [chr(codepoint) for codepoint in range(32)] + [
            chr(0x7F), chr(0x85), chr(0x2028), chr(0x2029), chr(0xE9), chr(0x4E2D), chr(0x20000),
            '"', "'", "\\", "#",
        ]
        for platform in ("linux", "win32"):
            for fallback in (False, True):
                for character in characters:
                    with self.subTest(platform=platform, fallback=fallback, codepoint=ord(character)):
                        fixture = load_fixture(platform)
                        self.assert_roundtrip(
                            fixture, "synthetic-python", "synthetic" + character + "user",
                            "synthetic" + character + "db", fallback=fallback,
                        )

    def test_reinstall_verify_uninstall_preserve_foreign_configuration(self) -> None:
        for platform in ("linux", "win32"):
            for fallback in (False, True):
                with self.subTest(platform=platform, fallback=fallback):
                    fixture = load_fixture(platform)
                    with scoped_imports(no_tomllib=fallback):
                        fixture.adapter.install_mcp("synthetic-python")
                        fixture.adapter.install_hooks("synthetic-python", 'user "quoted"', "db\\path")
                        first = fixture.text()
                        first_mcp = fixture.mcp_config.read_text(encoding="utf-8")
                        replacements = fixture.fs.replace_calls
                        fixture.adapter.install_hooks("synthetic-python", 'user "quoted"', "db\\path")
                        self.assertEqual(fixture.text(), first)
                        self.assertEqual(fixture.fs.replace_calls, replacements)
                        self.assertEqual(fixture.mcp_config.read_text(encoding="utf-8"), first_mcp)
                        self.assertTrue(fixture.adapter.verify())
                        self.assertEqual(len(fixture.adapter._parse_existing_hooks(first)), 5)
                        fixture.adapter.uninstall()
                        self.assertFalse(fixture.adapter.verify())
                        remaining = fixture.adapter._parse_existing_hooks(fixture.text())
                    self.assertEqual(len(remaining), 1)
                    self.assertEqual(remaining[0]["command"], "printf foreign-hook")
                    self.assertTrue(fixture.text().startswith(FOREIGN_TOML))
                    self.assertNotIn("truememory", fixture.text())
                    self.assertEqual(json.loads(fixture.mcp_config.read_text(encoding="utf-8")), FOREIGN_MCP)

    def test_atomic_failure_preserves_original_and_retry_installs_once(self) -> None:
        for platform in ("linux", "win32"):
            for fallback in (False, True):
                with self.subTest(platform=platform, fallback=fallback):
                    fixture = load_fixture(platform)
                    original_files = dict(fixture.fs.files)
                    fixture.fs.fail_next_replace = True
                    with scoped_imports(no_tomllib=fallback):
                        with self.assertRaisesRegex(OSError, "Synthetic atomic replace failure"):
                            fixture.adapter.install_hooks("synthetic-python", "synthetic\\user", "synthetic\\db")
                    self.assertEqual(fixture.fs.files, original_files)
                    self.assertEqual(fixture.fs.replace_calls, 1)
                    self.assert_roundtrip(fixture, "synthetic-python", "synthetic\\user", "synthetic\\db", fallback=fallback)
                    self.assertEqual(fixture.fs.replace_calls, 2)
                    self.assertFalse(any(name.endswith(".tmp") for name in fixture.fs.files))

    def test_fallback_rejects_malformed_complete_strings(self) -> None:
        malformed = [
            r'"truememory\q"', r'"truememory\u12zz"', '"truememory" trailing',
            '"truememory" "second"', '"truememory',
        ]
        for scalar in malformed:
            with self.subTest(scalar=scalar):
                fixture = load_fixture("win32")
                text = '[[hooks]]\nevent = "SessionStart"\ncommand = ' + scalar + '\ntimeout = 10\n'
                with scoped_imports(no_tomllib=True):
                    self.assertEqual(fixture.adapter._parse_existing_hooks(text), [])

    def test_fallback_allows_complete_string_and_trailing_comment(self) -> None:
        fixture = load_fixture("win32")
        command = r'"C:\Synthetic Python\python.exe" truememory --user "quoted user"'
        text = '[[hooks]]\nevent = "SessionStart"\ncommand = ' + json.dumps(command) + ' # "synthetic comment"\ntimeout = 10\n'
        with scoped_imports(no_tomllib=True):
            hooks = fixture.adapter._parse_existing_hooks(text)
        self.assertEqual(len(hooks), 1)
        self.assertEqual(hooks[0]["command"], command)
        self.assertEqual(hooks[0]["event"], "SessionStart")
        self.assertEqual(hooks[0]["timeout"], "10")


if __name__ == "__main__":
    unittest.main(verbosity=2)
