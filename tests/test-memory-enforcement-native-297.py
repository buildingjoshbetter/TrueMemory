"""Synthetic bootstrap readback controls; real enforcement remains root-gated."""
from __future__ import annotations

import ast
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch



import os as _host_os
import sys as _host_sys
from pathlib import PurePosixPath, PureWindowsPath
from types import ModuleType as _FixtureModule

_HOST_PATH = Path
_HOST_REPOSITORY = _HOST_PATH(__file__).resolve().parents[1]


def _forbidden_target_effect(*args: object, **kwargs: object) -> object:
    raise AssertionError("Unmocked synthetic target effect is forbidden")


class _TargetPath(PurePosixPath):
    def __new__(cls, *args: object):
        if args:
            first = args[0]
            if isinstance(first, _HOST_PATH) or (
                isinstance(first, str) and PureWindowsPath(first).is_absolute()
            ):
                native = _HOST_PATH(*args)
                if (not native.is_absolute() or ".." in native.parts
                        or not native.is_relative_to(_HOST_REPOSITORY)):
                    raise AssertionError("Native fixture path outside repository")
                return native
        return super().__new__(cls, *args)

    lstat = _forbidden_target_effect
    stat = _forbidden_target_effect
    resolve = _forbidden_target_effect
    is_symlink = _forbidden_target_effect
    read_bytes = _forbidden_target_effect
    read_text = _forbidden_target_effect
    write_bytes = _forbidden_target_effect
    write_text = _forbidden_target_effect
    open = _forbidden_target_effect
    exists = _forbidden_target_effect
    mkdir = _forbidden_target_effect
    unlink = _forbidden_target_effect

    def absolute(self):
        if not self.is_absolute():
            raise AssertionError("Relative synthetic path has no native working directory")
        return self


def _target_os():
    module = _FixtureModule("synthetic_target_os")
    module.__dict__.update(vars(_host_os))
    module.name = "posix"
    # Linux fixture symbols preserve access comparisons, not native Windows flags.
    for name, value in {"O_RDONLY": 0, "O_WRONLY": 1, "O_RDWR": 2, "O_ACCMODE": 3,
                        "O_CREAT": 64, "O_EXCL": 128, "O_TRUNC": 512, "O_APPEND": 1024,
                        "O_NONBLOCK": 2048, "O_DIRECTORY": 65536,
                        "O_NOFOLLOW": 131072, "O_CLOEXEC": 524288}.items():
        setattr(module, name, value)
    for name in ("getuid", "geteuid", "getpgid", "killpg", "kill", "pread", "open",
                 "read", "write", "close", "fstat", "stat", "lstat", "fsync", "fdopen",
                 "lseek", "umask", "access", "mkdir", "rename", "replace", "unlink",
                 "rmdir", "listdir", "scandir", "fchmod", "dup", "pipe", "set_blocking"):
        setattr(module, name, _forbidden_target_effect)
    return module


_TARGET_OS = _target_os() if _host_sys.platform == "win32" else _host_os


def _prepare_synthetic_bootstrap(test):
    if _host_sys.platform != "win32":
        return
    stack = ExitStack()
    test.addCleanup(stack.close)
    name = "_memory_bootstrap_synthetic_297"
    synthetic = _FixtureModule(name)
    synthetic.__file__ = str(_SOURCE)
    stack.enter_context(patch.dict(sys.modules, {name: synthetic}))
    # Fresh original-source globals keep every existing method-specific mock effective.
    exec(compile(_SOURCE.read_bytes(), str(_SOURCE), "exec", dont_inherit=True), synthetic.__dict__)
    synthetic.os, synthetic.Path = _TARGET_OS, _TargetPath
    stack.enter_context(patch.object(sys.modules[__name__], "bootstrap", synthetic))
    stack.enter_context(patch.object(sys.modules[__name__], "os", _TARGET_OS))
    stack.enter_context(patch.object(sys.modules[__name__], "Path", _TargetPath))


_SOURCE = Path(__file__).resolve().parents[1] / "truememory" / "memory_bootstrap.py"
_SPEC = importlib.util.spec_from_file_location("_memory_bootstrap_author_297", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bootstrap
_SPEC.loader.exec_module(bootstrap)


def expectation() -> object:
    return bootstrap.LinuxScopeExpectation("/synthetic/owned", 123, 1000, 10, 20, 67108864, 0)


_HAS_SCOPE_FLAGS = all(hasattr(_TARGET_OS, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"))


class SyntheticReadbackControls(unittest.TestCase):
    def setUp(self) -> None:
        _prepare_synthetic_bootstrap(self)

    def test_bootstrap_imports_only_stdlib_and_does_not_activate(self) -> None:
        tree = ast.parse(_SOURCE.read_bytes())
        imports = {node.module.split(".")[0] for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.module is not None}
        imports |= {alias.name.split(".")[0] for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for alias in node.names}
        self.assertLessEqual(imports, {"__future__", "dataclasses", "os", "pathlib", "stat"})
        self.assertNotIn("truememory", imports)
        self.assertNotIn("torch", imports)
        with self.assertRaises(SystemExit):
            bootstrap.main()

    def test_finite_typed_expectations(self) -> None:
        baseline = expectation().__dict__
        for field, value in (("pid", True), ("uid", -1), ("memory_max", 0),
                             ("swap_max", "max"), ("inode", 2**63), ("scope", "relative")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                bootstrap.LinuxScopeExpectation(**(baseline | {field: value}))

    @unittest.skipUnless(_HAS_SCOPE_FLAGS, "Descriptor readback needs POSIX no-follow primitives")
    def test_readback_never_promotes_physical_or_lifetime_claim(self) -> None:
        files = {"memory.max": "67108864", "memory.swap.max": "0", "cgroup.procs": "123",
                 "memory.current": "1048576", "memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0"}
        fake_stat = SimpleNamespace(st_uid=1000, st_dev=10, st_ino=20)
        with patch.object(bootstrap.os, "open", return_value=17), \
             patch.object(bootstrap.os, "fstat", return_value=fake_stat), \
             patch.object(bootstrap.os, "close") as close, \
             patch.object(bootstrap, "_read_owned_file", side_effect=lambda fd, name: files[name]):
            result = bootstrap.inspect_owned_linux_scope(expectation())
        self.assertEqual(result.members, (123,))
        self.assertEqual(result.memory_current, 1048576)
        self.assertEqual(result.metric, "controller_file_readback_unverified")
        self.assertFalse(result.physical_hard_cap)
        self.assertFalse(result.lifetime_enforcement_proven)
        close.assert_called_once_with(17)

    @unittest.skipUnless(_HAS_SCOPE_FLAGS, "Descriptor readback needs POSIX no-follow primitives")
    def test_unrelated_scope_or_member_is_rejected_and_descriptor_closes(self) -> None:
        base = {"memory.max": "67108864", "memory.swap.max": "0", "cgroup.procs": "123",
                "memory.current": "0", "memory.events": "oom_kill 0"}
        for changed in ({"memory.max": "max"}, {"memory.max": "67108863"},
                        {"memory.swap.max": "1"}, {"cgroup.procs": "123\n456"},
                        {"cgroup.procs": ""}, {"memory.events": "bad"}):
            files = base | changed
            with self.subTest(changed=changed), \
                 patch.object(bootstrap.os, "open", return_value=17), \
                 patch.object(bootstrap.os, "fstat", return_value=SimpleNamespace(st_uid=1000, st_dev=10, st_ino=20)), \
                 patch.object(bootstrap.os, "close") as close, \
                 patch.object(bootstrap, "_read_owned_file", side_effect=lambda fd, name: files[name]):
                with self.assertRaises(bootstrap.EnforcementUnavailable):
                    bootstrap.inspect_owned_linux_scope(expectation())
                close.assert_called_once_with(17)
        with patch.object(bootstrap.os, "open", return_value=17), \
             patch.object(bootstrap.os, "fstat", return_value=SimpleNamespace(st_uid=0, st_dev=10, st_ino=20)), \
             patch.object(bootstrap.os, "close") as close:
            with self.assertRaises(bootstrap.EnforcementUnavailable):
                bootstrap.inspect_owned_linux_scope(expectation())
            close.assert_called_once_with(17)

    @unittest.skipUnless(_HAS_SCOPE_FLAGS, "Descriptor readback needs POSIX no-follow primitives")
    def test_nonregular_controller_file_uses_nonblocking_open_and_closes(self) -> None:
        with patch.object(bootstrap.os, "open", return_value=19) as opened, \
             patch.object(bootstrap.os, "fstat", return_value=SimpleNamespace(st_mode=stat.S_IFIFO)), \
             patch.object(bootstrap.os, "close") as closed, \
             patch.object(bootstrap.os, "read") as read:
            with self.assertRaises(bootstrap.EnforcementUnavailable):
                bootstrap._read_owned_file(17, "memory.max")
            self.assertTrue(opened.call_args.args[1] & os.O_NONBLOCK)
            self.assertTrue(opened.call_args.args[1] & os.O_NOFOLLOW)
            read.assert_not_called()
            closed.assert_called_once_with(19)

    @unittest.skipUnless(_HAS_SCOPE_FLAGS, "Descriptor readback needs POSIX no-follow primitives")
    def test_readback_is_bounded_and_closes_on_decode_failure(self) -> None:
        for chunks in ([b"\xff", b""], [b"x" * 4096] * 17):
            with self.subTest(kind=len(chunks)), \
                 patch.object(bootstrap.os, "open", return_value=19), \
                 patch.object(bootstrap.os, "fstat", return_value=SimpleNamespace(st_mode=stat.S_IFREG)), \
                 patch.object(bootstrap.os, "close") as close, \
                 patch.object(bootstrap.os, "read", side_effect=chunks):
                with self.assertRaises((UnicodeError, bootstrap.EnforcementUnavailable)):
                    bootstrap._read_owned_file(17, "memory.max")
                close.assert_called_once_with(19)


@unittest.skipUnless(sys.platform == "linux" and os.environ.get("TRUEMEMORY_GPU_MEMORY_READBACK") == "1",
                     "Actual owned Linux readback requires a separate root-approved scope receipt")
class OwnedLinuxReadbackWitness(unittest.TestCase):
    def test_explicit_owned_scope_readback_only(self) -> None:
        receipt_path = os.environ.get("TRUEMEMORY_GPU_MEMORY_READBACK_RECEIPT")
        self.assertIsNotNone(receipt_path, "Root must supply an explicit scope expectation receipt")
        raw = json.loads(Path(receipt_path).read_text())
        self.assertEqual(set(raw), {"scope", "pid", "uid", "device", "inode", "memory_max", "swap_max"})
        result = bootstrap.inspect_owned_linux_scope(bootstrap.LinuxScopeExpectation(**raw))
        self.assertFalse(result.physical_hard_cap)
        self.assertFalse(result.lifetime_enforcement_proven)


if __name__ == "__main__":
    unittest.main()
