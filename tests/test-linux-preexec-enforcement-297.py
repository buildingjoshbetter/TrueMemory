"""Synthetic public-contract controls; actual manager/kernel execution is separate."""
from __future__ import annotations

import ast
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
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


def _target_signal(original):
    module = _FixtureModule("synthetic_target_signal")
    module.__dict__.update(vars(original))
    module.SIGTERM, module.SIGKILL, module.SIGUSR1 = 15, 9, 10
    for name in ("signal", "alarm", "raise_signal", "pidfd_send_signal"):
        setattr(module, name, _forbidden_target_effect)
    return module


def _target_select(original):
    module = _FixtureModule("synthetic_target_select")
    module.__dict__.update(vars(original))
    module.poll = module.select = _forbidden_target_effect
    return module


def _prepare_target_module(module):
    if _host_sys.platform != "win32":
        return module
    module.os = _TARGET_OS
    module.Path = _TargetPath
    for name in ("RUN_ROOT", "SOURCE_ROOT", "RUNTIME_PACKET"):
        if hasattr(module, name):
            value = getattr(module, name)
            setattr(module, name, _TargetPath(value.as_posix()))
    if hasattr(module, "sys"):
        isolated = _FixtureModule("synthetic_target_sys")
        isolated.__dict__.update(vars(module.sys))
        module.sys = isolated
    if hasattr(module, "signal"):
        module.signal = _target_signal(module.signal)
    if hasattr(module, "select"):
        module.select = _target_select(module.select)
    return module


_TARGET_OS = _target_os() if _host_sys.platform == "win32" else _host_os
if _host_sys.platform == "win32":
    Path = _TargetPath


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "truememory_memory_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("synthetic_linux_preexec_author", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
_prepare_target_module(MODULE)


def identity(cap: int = 67108864) -> object:
    generation = "a" * 32
    return MODULE.LinuxInvocationIdentity(
        1, generation, "12345678-1234-1234-1234-123456789abc", 1000,
        "app-truememory" + generation + ".slice", "b" * 32, 29, 31, 37,
        "/user.slice/app.slice/app-truememory" + generation + ".slice",
        "truememory-memory" + generation + ".service", "c" * 32, 31, 41,
        "/user.slice/app.slice/app-truememory" + generation + ".slice/child.service",
        43, 47, 1000.5, cap, 0, "d" * 64, "e" * 64,
    )


def evidence(value: object) -> object:
    return MODULE.LinuxScopeEvidence(value, 0x63677270, "cgroup2", True, True, True,
               value.uid, value.main_pid, value.start_ticks, value.generation,
               value.manifest_sha256, (value.main_pid,), value.cap_bytes, 0, "max", "max", True)


def receipt(value: object) -> object:
    return MODULE.LinuxExitReceipt(1, value, 0, None, "success", 0, None,
               (("max", 0), ("oom", 0)), (("max", 1), ("oom", 1)), False,
               ((value.slice_relative_path, ()), (value.service_relative_path, ())),
               True, 5, "f" * 64, (("manifest-v1.json", "e" * 64),),
               "linux_cgroup_charged_domain", False, value.cap_bytes, False)


class InertAdapter:
    def __init__(self) -> None:
        self.identity = identity()
        self.record = MODULE.LinuxProvisionalLaunch(self.identity, object(), object(), object(), object())
        self.calls: list[str] = []
        self.deadlines: list[float] = []
        self.terminal = False

    def start(self, policy: object, generation: str, python_executable: str, deadline: float) -> object:
        self.calls.append("start")
        self.deadlines.append(deadline)
        return self.record

    def inspect(self, record: object, deadline: float) -> object:
        self.calls.append("inspect")
        self.deadlines.append(deadline)
        result = evidence(self.identity)
        return replace(result, members=()) if self.terminal else result

    def inspect_existing(self, policy: object, *, pid: int, birth_time: float, generation: str,
                         claim: dict[str, object], deadline: float) -> object:
        self.calls.append("inspect_existing")
        self.deadlines.append(deadline)
        return evidence(self.identity)

    def barrier(self, record: object, value: object, deadline: float) -> None:
        self.calls.append("barrier")
        self.deadlines.append(deadline)

    def capture(self, record: object, value: object, deadline: float) -> object:
        self.calls.append("capture")
        self.deadlines.append(deadline)
        return receipt(self.identity)

    def stop(self, record: object, value: object, deadline: float) -> None:
        self.calls.append("stop")
        self.deadlines.append(deadline)


class TestLinuxPreexecContract(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = InertAdapter()
        self.deadline = time.monotonic() + 10.0
        self.policy = MODULE.LinuxDaemonLaunchPolicy(1, 67108864, 0)

    def launch(self) -> object:
        with patch.object(MODULE, "get_linux_adapter", return_value=self.adapter):
            return MODULE.launch_owned_linux(self.policy, generation="a" * 32,
                      python_executable="/synthetic/python", deadline=self.deadline)

    def test_literal_flag_only(self) -> None:
        self.assertIs(MODULE.required_cap_selected({}), False)
        self.assertIs(MODULE.required_cap_selected({"TRUEMEMORY_REQUIRE_PROCESS_MEMORY_CAP": "1"}), True)
        for invalid in ("0", "", " 1", "1 ", 1, True, None):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                MODULE.required_cap_selected({"TRUEMEMORY_REQUIRE_PROCESS_MEMORY_CAP": invalid})

    def test_exact_budget_arithmetic(self) -> None:
        env = {"TRUEMEMORY_REQUIRE_PROCESS_MEMORY_CAP": "1"}
        self.assertEqual(MODULE.plan_required_linux_launch(env, physical_total=34359738368).cap_bytes, 8589934592)
        self.assertEqual(MODULE.plan_required_linux_launch(env, physical_total=1073741824).cap_bytes, 268435456)
        env["TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB"] = "64"
        self.assertEqual(MODULE.plan_required_linux_launch(env, physical_total=None).cap_bytes, 64 * 1048576)
        self.assertIs(MODULE.plan_required_linux_launch(env, physical_total=None).physical_hard_cap, False)
        self.assertIs(MODULE.plan_required_linux_launch(env, physical_total=None).dynamic_registry_cap, False)

    def test_invalid_budget_does_not_fall_back(self) -> None:
        for invalid in ("0", "00", "064", "+64", "64 ", "64.0", "-1", True, 64, str(2**63), "8796093022208"):
            env = {"TRUEMEMORY_REQUIRE_PROCESS_MEMORY_CAP": "1", "TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB": invalid}
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                MODULE.plan_required_linux_launch(env, physical_total=34359738368)
        for total in (None, 0, 1, 3, True, 1.5, 2**63):
            with self.subTest(total=total), self.assertRaises((ValueError, TypeError)):
                MODULE.plan_required_linux_launch({"TRUEMEMORY_REQUIRE_PROCESS_MEMORY_CAP": "1"}, physical_total=total)

    def test_constructor_subclasses_rejected(self) -> None:
        class PolicySubtype(MODULE.LinuxDaemonLaunchPolicy):
            pass
        with self.assertRaises(TypeError):
            PolicySubtype(1, 67108864, 0)
        with self.assertRaises(ValueError):
            replace(self.policy, cap_bytes=True)
        with self.assertRaises(ValueError):
            replace(self.adapter.identity, birth_time=47)

    def test_reflective_mutation_revalidated_before_adapter(self) -> None:
        object.__setattr__(self.policy, "cap_bytes", True)
        with patch.object(MODULE, "get_linux_adapter", return_value=self.adapter), self.assertRaises(ValueError):
            MODULE.launch_owned_linux(self.policy, generation="a" * 32,
                python_executable="/synthetic/python", deadline=self.deadline)
        self.assertEqual(self.adapter.calls, [])

    def test_deadline_generation_executable_refuse_before_adapter(self) -> None:
        for field, invalid in (("deadline", True), ("deadline", float("inf")),
                               ("deadline", time.monotonic() - 1), ("generation", "A" * 32),
                               ("python_executable", "relative/python")):
            arguments = {"generation": "a" * 32, "python_executable": "/synthetic/python", "deadline": self.deadline}
            arguments[field] = invalid
            with self.subTest(field=field, invalid=invalid), patch.object(MODULE, "get_linux_adapter", return_value=self.adapter), self.assertRaises((ValueError, TypeError, TimeoutError)):
                MODULE.launch_owned_linux(self.policy, **arguments)
        self.assertEqual(self.adapter.calls, [])

    def test_start_retains_all_resources_without_release(self) -> None:
        handle = self.launch()
        self.assertEqual(handle.pid, 43)
        self.assertIs(handle.record, self.adapter.record)
        for name in ("runner", "pidfd", "scope_fd", "endpoint"):
            self.assertIs(getattr(handle.record, name), getattr(self.adapter.record, name))
        self.assertEqual(self.adapter.calls, ["start", "inspect"])
        self.assertEqual(self.adapter.deadlines, [self.deadline, self.deadline])

    def test_claim_actual_pid_and_birth_before_barrier(self) -> None:
        handle = self.launch()
        claim = {"generation": "a" * 32, "launcher": {"pid": 101, "born": 2000.5}, "child": {"pid": 43, "born": 1000.5}}
        handle.release_after_claim(claim, deadline=self.deadline)
        self.assertEqual(self.adapter.calls, ["start", "inspect", "inspect", "barrier"])
        self.assertTrue(all(value == self.deadline for value in self.adapter.deadlines))
        self.assertTrue(handle._released)

    def test_wrong_claim_cannot_release(self) -> None:
        handle = self.launch()
        for field, invalid in (("pid", 101), ("pid", True), ("born", 47), ("born", 47.0)):
            child = {"pid": 43, "born": 1000.5}
            child[field] = invalid
            claim = {"generation": "a" * 32, "launcher": {"pid": 101, "born": 2000.5}, "child": child}
            with self.subTest(field=field, invalid=invalid), self.assertRaises((ValueError, TypeError)):
                handle.release_after_claim(claim, deadline=self.deadline)
        self.assertNotIn("barrier", self.adapter.calls)

    def test_foreign_evidence_refuses_return(self) -> None:
        bad = evidence(self.adapter.identity)
        object.__setattr__(bad, "filesystem_magic", 0x01021994)
        with patch.object(self.adapter, "inspect", return_value=bad), patch.object(MODULE, "get_linux_adapter", return_value=self.adapter), self.assertRaises(ValueError):
            self.launch()

    def test_terminal_receipt_before_stop(self) -> None:
        handle = self.launch()
        self.adapter.terminal = True
        actual = MODULE.stop_owned_exact(handle, deadline=self.deadline)
        self.assertEqual(actual.identity, self.adapter.identity)
        self.assertLess(self.adapter.calls.index("capture"), self.adapter.calls.index("stop"))
        self.assertIs(actual.physical_hard_cap, False)
        self.assertIs(actual.dynamic_registry_cap, False)
        self.assertIsNone(actual.memory_peak)

    def test_live_or_undurable_receipt_cannot_stop(self) -> None:
        handle = self.launch()
        self.adapter.terminal = True
        for field, invalid in (("populated", True), ("durable_before_stop", False), ("physical_hard_cap", True),
                               ("dynamic_registry_cap", True), ("kernel_cap_bytes", 1),
                               ("descendant_members", ((self.adapter.identity.slice_relative_path, (43,)),))):
            bad = receipt(self.adapter.identity)
            object.__setattr__(bad, field, invalid)
            with self.subTest(field=field), patch.object(self.adapter, "capture", return_value=bad), self.assertRaises((ValueError, RuntimeError)):
                MODULE.stop_owned_exact(handle, deadline=self.deadline)
        self.assertNotIn("stop", self.adapter.calls)

    def test_identity_drift_before_stop(self) -> None:
        handle = self.launch()
        self.adapter.terminal = True
        foreign = replace(self.adapter.identity, slice_invocation_id="f" * 32)
        with patch.object(self.adapter, "inspect", return_value=evidence(foreign)), self.assertRaises(RuntimeError):
            MODULE.stop_owned_exact(handle, deadline=self.deadline)
        self.assertNotIn("capture", self.adapter.calls)
        self.assertNotIn("stop", self.adapter.calls)

    def test_live_domain_cannot_capture_or_stop(self) -> None:
        handle = self.launch()
        with self.assertRaises(RuntimeError):
            MODULE.stop_owned_exact(handle, deadline=self.deadline)
        self.assertNotIn("capture", self.adapter.calls)
        self.assertNotIn("stop", self.adapter.calls)

    def test_counters_and_owned_paths_are_strict(self) -> None:
        original = receipt(self.adapter.identity)
        for field, invalid in (("events_after", (("max", 0), ("oom", -1))),
                               ("events_after", (("max", 1), ("max", 2))),
                               ("descendant_members", (("/foreign.slice", ()),)),
                               ("artifact_sha256", (("../foreign", "a" * 64),))):
            with self.subTest(field=field), self.assertRaises((ValueError, RuntimeError)):
                replace(original, **{field: invalid})

    def existing(self, **changes: object) -> object:
        values = {"pid": 43, "birth_time": 1000.5, "generation": "a" * 32,
                  "claim": {"generation": "a" * 32, "launcher": {"pid": 101, "born": 2000.5},
                            "child": {"pid": 43, "born": 1000.5}},
                  "deadline": self.deadline, "adapter": self.adapter}
        values.update(changes)
        return MODULE.inspect_existing_owned_linux(self.policy, **values)

    def test_readonly_join_has_no_launch_or_stop_authority(self) -> None:
        result = self.existing()
        self.assertIs(type(result), MODULE.LinuxScopeEvidence)
        self.assertEqual(self.adapter.calls, ["inspect_existing"])
        self.assertEqual(self.adapter.deadlines, [self.deadline])
        self.assertFalse(hasattr(result, "release_after_claim"))

    def test_readonly_join_rejects_inputs_before_observer(self) -> None:
        for name, invalid in (("pid", True), ("pid", 0), ("birth_time", 47), ("birth_time", float("nan")),
                              ("generation", "A" * 32), ("deadline", True), ("deadline", float("inf")),
                              ("deadline", time.monotonic() - 1), ("claim", None)):
            with self.subTest(name=name), self.assertRaises((ValueError, TypeError, TimeoutError)):
                self.existing(**{name: invalid})
        self.assertEqual(self.adapter.calls, [])

    def test_readonly_join_rejects_foreign_claim_before_observer(self) -> None:
        for name, invalid in (("pid", 101), ("pid", True), ("born", 47), ("born", 47.0)):
            child = {"pid": 43, "born": 1000.5}
            child[name] = invalid
            claim = {"generation": "a" * 32, "launcher": {"pid": 101, "born": 2000.5}, "child": child}
            with self.subTest(name=name), self.assertRaises((ValueError, TypeError)):
                self.existing(claim=claim)
        self.assertEqual(self.adapter.calls, [])

    def test_readonly_join_accepts_smaller_real_cap(self) -> None:
        self.policy = MODULE.LinuxDaemonLaunchPolicy(1, 134217728, 0)
        result = self.existing()
        self.assertEqual(result.cap_bytes, 67108864)
        self.assertIs(result.kernel_limit_preexec, True)

    def test_readonly_join_refuses_larger_real_cap(self) -> None:
        self.policy = MODULE.LinuxDaemonLaunchPolicy(1, 33554432, 0)
        with self.assertRaises(RuntimeError):
            self.existing()
        self.assertEqual(self.adapter.calls, ["inspect_existing"])

    def test_readonly_join_revalidates_reflective_evidence(self) -> None:
        for name, invalid in (("peer_pid", 101), ("members", (43, 101)), ("cap_bytes", 1),
                              ("swap_bytes", 1), ("leaf_memory_max", "67108864"), ("kernel_limit_preexec", False)):
            value = evidence(self.adapter.identity)
            object.__setattr__(value, name, invalid)
            with self.subTest(name=name), patch.object(self.adapter, "inspect_existing", return_value=value), self.assertRaises((ValueError, RuntimeError)):
                self.existing()
        self.assertNotIn("start", self.adapter.calls)
        self.assertNotIn("stop", self.adapter.calls)

    def test_readonly_join_callback_cannot_change_inputs(self) -> None:
        claim = {"generation": "a" * 32, "launcher": {"pid": 101, "born": 2000.5}, "child": {"pid": 43, "born": 1000.5}}
        def mutate(policy: object, **kwargs: object) -> object:
            claim["child"]["pid"] = 101
            return evidence(self.adapter.identity)
        with patch.object(self.adapter, "inspect_existing", side_effect=mutate), self.assertRaises((ValueError, RuntimeError)):
            self.existing(claim=claim)

    def test_json_duplicate_keys_and_nonfinite_refused(self) -> None:
        for raw in (b'{"pid":43,"pid":101}', b'{"birth":NaN}', b'{"birth":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                MODULE._strict_json(raw)

    def test_same_pid_exec_preserves_ordinary_final_argv(self) -> None:
        source = MODULE_PATH.read_text()
        self.assertIn('[python_executable, "-I", "-S", str(bootstrap)', source)
        self.assertIn('[manifest["interpreter"]["path"], "-m", "truememory.model_server"]', source)
        self.assertIn('os.execve(manifest["interpreter"]["path"]', source)
        self.assertNotIn('importlib.import_module("site").main()', source)
        self.assertIn('"preexec_gate_passed": True', source)
        self.assertIn('"application-ready"', source)

    def test_original_snapshot_restores_reserved_name_values(self) -> None:
        original = {"TRUEMEMORY_REQUIRE_PROCESS_MEMORY_CAP": "1", "TRUEMEMORY_MODEL_SERVER_GENERATION": "a" * 32,
                    "TRUEMEMORY_CAP_CHILD": "1", "INVOCATION_ID": "original-synthetic-value",
                    "PYTHONHASHSEED": "43", "PYTHONMALLOC": "malloc"}
        packet = MODULE._original_environment_packet(original, generation="a" * 32, manifest_sha256="e" * 64)
        with patch.dict(MODULE.os.environ, {"INVOCATION_ID": "manager-synthesized-value"}):
            restored = MODULE._original_environment(packet["forwarded_environment"], generation="a" * 32,
                                                     names=sorted(original))
        self.assertEqual(restored, original)
        self.assertEqual(restored["INVOCATION_ID"], "original-synthetic-value")
        self.assertEqual(restored["PYTHONHASHSEED"], "43")

    def test_environment_snapshot_exact_complete_frame_bound(self) -> None:
        original = {"TRUEMEMORY_REQUIRE_PROCESS_MEMORY_CAP": "1", "TRUEMEMORY_MODEL_SERVER_GENERATION": "a" * 32,
                    "TRUEMEMORY_CAP_CHILD": "1", "SYNTHETIC": "value"}
        packet = MODULE._original_environment_packet(original, generation="a" * 32, manifest_sha256="e" * 64)
        self.assertLessEqual(len(MODULE._canonical(packet)) + 4, 65536)
        oversized = {**original, "SYNTHETIC": "synthetic-private-value" * 5000}
        with self.assertRaises(RuntimeError) as caught:
            MODULE._original_environment_packet(oversized, generation="a" * 32, manifest_sha256="e" * 64)
        self.assertNotIn("synthetic-private-value", str(caught.exception))
        for invalid in (None, 43, "invalid\x00value"):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, RuntimeError)):
                MODULE._original_environment({**original, "SYNTHETIC": invalid}, generation="a" * 32)
        with self.assertRaises(RuntimeError):
            MODULE._original_environment(original, generation="a" * 32, names=sorted(original) + ["MISSING"])

    def test_original_snapshot_stays_out_of_retained_manifest(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text())
        adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LinuxAdapter")
        start = next(node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == "start")
        assignment = next(node for node in ast.walk(start) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "manifest" for target in node.targets))
        self.assertIsInstance(assignment.value, ast.Dict)
        keys = {key.value for key in assignment.value.keys if isinstance(key, ast.Constant)}
        self.assertIn("forwarded_names", keys)
        self.assertNotIn("forwarded_environment", keys)
        self.assertNotIn("environment", keys)
        self.assertNotIn("environment_sha256", keys)

    def test_packaging_and_protected_capsule(self) -> None:
        project = (ROOT / "pyproject.toml").read_text()
        self.assertIn('truememory-model-server = "truememory.model_server:main"', project)
        self.assertIn('truememory-model-server-capped = "truememory_memory_bootstrap:main"', project)
        self.assertIn('[tool.hatch.build.targets.wheel.force-include]', project)
        self.assertIn('"truememory_memory_bootstrap.py" = "truememory_memory_bootstrap.py"', project)
        self.assertEqual(hashlib.sha256((ROOT / "truememory/memory_bootstrap.py").read_bytes()).hexdigest(),
                         "a33c0f4bf25bfa590390f50296623455491333ace98da97e4a3511e6aee50c5d")
        tree = ast.parse(MODULE_PATH.read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in imports:
            names = [node.module] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names]
            self.assertFalse(any(name is not None and name.startswith("truememory") for name in names))


class TestLinuxBackendSixFixes(unittest.TestCase):
    def test_preexec_packet_requires_exact_boolean(self) -> None:
        expected = {"generation": "a" * 32, "manifest_sha256": "e" * 64, "preexec_gate_passed": True}
        self.assertEqual(MODULE._literal_packet(dict(expected), expected, "preexec"), expected)
        for invalid in (1, 1.0, "true", None, False):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, RuntimeError)):
                MODULE._literal_packet({**expected, "preexec_gate_passed": invalid}, expected, "preexec")

    def test_final_ack_requires_exact_boolean(self) -> None:
        expected = {"generation": "a" * 32, "manifest_sha256": "e" * 64, "gate_passed": True}
        self.assertEqual(MODULE._literal_packet(dict(expected), expected, "ack"), expected)
        for invalid in (1, 1.0, "true", None, False):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, RuntimeError)):
                MODULE._literal_packet({**expected, "gate_passed": invalid}, expected, "ack")

    def test_ready_packet_requires_exact_scalar_types_and_keys(self) -> None:
        value = identity()
        expected = {"version": 1, "generation": value.generation, "manifest_sha256": value.manifest_sha256,
                    "pid": value.main_pid, "kernel_cap_bytes": value.cap_bytes, "metric": "linux_cgroup_charged_domain",
                    "physical_hard_cap": False, "dynamic_registry_cap": False}
        self.assertEqual(MODULE._literal_packet(dict(expected), expected, "ready"), expected)
        for name, invalid in (("version", True), ("version", 1.0), ("pid", 43.0),
                              ("kernel_cap_bytes", 67108864.0), ("physical_hard_cap", 0),
                              ("dynamic_registry_cap", 0), ("physical_hard_cap", "false")):
            with self.subTest(name=name, invalid=invalid), self.assertRaises((ValueError, RuntimeError)):
                MODULE._literal_packet({**expected, name: invalid}, expected, "ready")
        missing = dict(expected)
        del missing["version"]
        for invalid in (None, [], missing, {**expected, "extra": 1}, {**expected, 1: "extra"}):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, RuntimeError)):
                MODULE._literal_packet(invalid, expected, "ready")

    def test_pruned_limits_only_allowed_for_empty_terminal_evidence(self) -> None:
        value = identity()
        terminal = replace(evidence(value), members=(), leaf_memory_max="unavailable_pruned", leaf_swap_max="unavailable_pruned")
        self.assertIs(MODULE._validate_evidence(terminal, value, terminal=True), terminal)
        with self.assertRaises((ValueError, RuntimeError)):
            MODULE._validate_evidence(terminal, value)
        for changes in ({"members": (value.main_pid,)}, {"leaf_swap_max": "max"},
                        {"leaf_memory_max": "max"}, {"leaf_memory_max": None}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, RuntimeError)):
                replace(terminal, **changes)

    def inspect_terminal(self, *, control_group: str | None = None, leaf: str = "pruned",
                         exec_pid: int = 43, members: tuple[int, ...] = (), exited: bool = True) -> tuple[object, list[str]]:
        value = identity()
        trace: list[str] = []
        scope = MODULE._NativeScope(29, "/synthetic", 70, 71, (), pidfd_device=101, pidfd_inode=103)
        runner = MODULE._NativeRunner(object(), object(), {})
        endpoint = MODULE._NativeEndpoint(SimpleNamespace(fileno=lambda: 73), SimpleNamespace(fileno=lambda: 74),
                         value.main_pid, value.uid, {"generation": value.generation, "manifest_sha256": value.manifest_sha256,
                         "pid": value.main_pid, "start_ticks": value.start_ticks, "bootstrap_sha256": value.bootstrap_sha256},
                         107, 109, 113, 127)
        record = MODULE.LinuxProvisionalLaunch(value, runner, 72, scope, endpoint)
        metadata = {70: (31, 37), 71: (31, 41), 72: (101, 103), 73: (107, 109),
                    74: (113, 127), 75: (31, 37), 76: (31, 999 if leaf == "replacement" else 41)}

        def fstat(descriptor: int) -> object:
            device, inode = metadata[descriptor]
            return SimpleNamespace(st_uid=1000, st_dev=device, st_ino=inode)

        def open_directory(path: Path, *, uid: int | None = None) -> int:
            self.assertEqual(uid, value.uid)
            if str(path) == "/synthetic" + value.slice_relative_path:
                trace.append("open_slice")
                return 75
            self.assertEqual(str(path), "/synthetic" + value.service_relative_path)
            trace.append("open_leaf")
            if leaf == "pruned":
                raise FileNotFoundError("synthetic pruned leaf")
            if leaf == "permission":
                raise PermissionError("synthetic inaccessible leaf")
            return 76

        def properties(unit: str, fields: tuple[str, ...], deadline: float) -> dict[str, str]:
            if unit == value.slice_unit:
                trace.append("slice_manager")
                return {"InvocationID": value.slice_invocation_id, "ControlGroup": value.slice_relative_path,
                        "MemoryMax": str(value.cap_bytes), "MemorySwapMax": "0", "ActiveState": "active"}
            self.assertEqual(unit, value.service_unit)
            trace.append("service_manager")
            return {"InvocationID": value.service_invocation_id,
                    "ControlGroup": value.service_relative_path if control_group is None else control_group,
                    "MainPID": "0", "ExecMainPID": str(exec_pid), "ActiveState": "active", "SubState": "exited"}

        def read_at(descriptor: int, name: str) -> str:
            if descriptor == 70:
                return {"memory.max": str(value.cap_bytes), "memory.swap.max": "0"}[name]
            self.assertEqual(descriptor, 71)
            self.assertIn(name, ("memory.max", "memory.swap.max"))
            return "max"

        def census(descriptor: int, relative: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
            trace.append("census")
            return ((relative, members),)

        def process_identity(pid: int) -> tuple[int, float]:
            trace.append("process_exited")
            raise FileNotFoundError("synthetic exited PID")

        def pidfd_ready(read: list[int], write: list[int], errors: list[int], timeout: int) -> tuple[list[int], list[int], list[int]]:
            trace.append("pidfd_exit")
            return ([72] if exited else [], [], [])

        with ExitStack() as stack:
            for target, name, replacement in ((MODULE.sys, "platform", "linux"), (MODULE.os, "getuid", lambda: 1000),
                  (MODULE, "_boot_id", lambda: value.boot_id), (MODULE, "_mount", lambda: (29, "/synthetic")),
                  (MODULE.os, "fstat", fstat), (MODULE.os, "close", lambda descriptor: None),
                  (MODULE, "_open_directory", open_directory), (MODULE, "_filesystem_magic", lambda descriptor: None),
                  (MODULE, "_properties", properties), (MODULE, "_read_at", read_at),
                  (MODULE, "_ancestor_limits", lambda *args: ()), (MODULE, "_recursive_members", census),
                  (MODULE, "_process_identity", process_identity), (MODULE.select, "select", pidfd_ready),
                  (MODULE, "_peer", lambda connection: (value.main_pid, value.uid))):
                stack.enter_context(patch.object(target, name, replacement))
            result = MODULE.LinuxAdapter().inspect(record, time.monotonic() + 10.0)
        return result, trace

    def test_actual_inspector_accepts_only_proven_terminal_pruning(self) -> None:
        for control_group in (None, ""):
            with self.subTest(control_group=control_group):
                result, trace = self.inspect_terminal(control_group=control_group)
                self.assertEqual(result.members, ())
                self.assertEqual(result.leaf_memory_max, "unavailable_pruned")
                self.assertEqual(result.leaf_swap_max, "unavailable_pruned")
                for proof in ("service_manager", "census", "pidfd_exit", "process_exited"):
                    self.assertLess(trace.index(proof), trace.index("open_leaf"))

    def test_actual_inspector_refuses_pruning_without_owned_terminal_proof(self) -> None:
        for changes in ({"control_group": "/foreign.service"}, {"exec_pid": 99},
                        {"members": (43, 99)}, {"exited": False}, {"leaf": "permission"},
                        {"leaf": "replacement"}, {"leaf": "present", "control_group": ""}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, RuntimeError, OSError)):
                self.inspect_terminal(**changes)
        present, _ = self.inspect_terminal(leaf="present")
        self.assertEqual((present.leaf_memory_max, present.leaf_swap_max), ("max", "max"))

    def capture_terminal(self, *, pruned: bool, missing_leaf_events: bool = False,
                         slice_decreased: bool = False) -> tuple[object, dict[str, object]]:
        value = identity()
        terminal = replace(evidence(value), members=())
        if pruned:
            terminal = replace(terminal, leaf_memory_max="unavailable_pruned", leaf_swap_max="unavailable_pruned")
        baseline = (("max", 1), ("oom", 0))
        payloads: list[dict[str, object]] = []

        class Store:
            directory = 75
            hashes = {"manifest-v1.json": "e" * 64}

            def observe(self, name: str, **kwargs: object) -> None:
                return None

            def append(self, phase: str, payload: dict[str, object]) -> tuple[int, str]:
                payloads.append(payload)
                return 5, "f" * 64

            def verify(self) -> None:
                return None

        scope = SimpleNamespace(slice_fd=70, service_fd=71, baseline=baseline, baseline_local=baseline,
                                leaf_baseline=baseline, leaf_baseline_local=baseline)
        record = MODULE.LinuxProvisionalLaunch(value, SimpleNamespace(store=Store()), 72, scope, object())
        adapter = MODULE.LinuxAdapter()
        manager = {"InvocationID": value.service_invocation_id, "ExecMainPID": str(value.main_pid),
                   "MainPID": "0", "ExecMainCode": "1", "ExecMainStatus": "0", "Result": "success", "SubState": "exited"}

        def read_at(descriptor: int, name: str) -> str:
            self.assertEqual(descriptor, 70)
            if name == "memory.peak":
                raise FileNotFoundError("synthetic unavailable peak")
            return {"cgroup.events": "populated 0\n", "memory.current": "123"}[name]

        def events(descriptor: int, name: str = "memory.events") -> tuple[tuple[str, int], ...]:
            if descriptor == 71:
                self.assertFalse(pruned, "A pruned leaf must not receive counter reads")
                if missing_leaf_events:
                    raise FileNotFoundError("synthetic missing leaf counter")
            self.assertIn(descriptor, (70, 71))
            return (("max", 0 if slice_decreased and descriptor == 70 else 2), ("oom", 1))

        with ExitStack() as stack:
            stack.enter_context(patch.object(adapter, "inspect", return_value=terminal))
            stack.enter_context(patch.object(MODULE, "_properties", return_value=manager))
            stack.enter_context(patch.object(MODULE, "_recursive_members", return_value=((value.slice_relative_path, ()),)))
            stack.enter_context(patch.object(MODULE, "_read_at", read_at))
            stack.enter_context(patch.object(MODULE, "_controller_events", events))
            result = adapter.capture(record, value, time.monotonic() + 10.0)
        self.assertEqual(len(payloads), 1)
        return result, payloads[0]

    def test_actual_capture_retains_pruned_baselines_and_null_after(self) -> None:
        result, payload = self.capture_terminal(pruned=True)
        self.assertEqual(result.events_before, (("max", 1), ("oom", 0)))
        self.assertEqual(result.events_after, (("max", 2), ("oom", 1)))
        self.assertEqual(payload["leaf_hierarchical_before"], result.events_before)
        self.assertEqual(payload["leaf_local_before"], result.events_before)
        for name in ("leaf_hierarchical_after", "leaf_local_after", "leaf_hierarchical_deltas", "leaf_local_deltas"):
            self.assertIsNone(payload[name])
        self.assertIs(payload["leaf_counters_available"], False)
        self.assertEqual(payload["leaf_counter_status"], "unavailable_pruned")
        self.assertIsNone(result.memory_peak)

    def test_actual_capture_requires_slice_counters_and_real_pruning(self) -> None:
        with self.assertRaises(RuntimeError):
            self.capture_terminal(pruned=True, slice_decreased=True)
        with self.assertRaises(RuntimeError):
            self.capture_terminal(pruned=False, missing_leaf_events=True)
        _, payload = self.capture_terminal(pruned=False)
        self.assertIs(payload["leaf_counters_available"], True)
        self.assertEqual(payload["leaf_counter_status"], "available")
        self.assertEqual(payload["leaf_hierarchical_deltas"], (("max", 1), ("oom", 1)))


class TestLinuxRetainedPreexecPackets(unittest.TestCase):
    def validate_reader(self, method: str, payload: object, value: object) -> object:
        """Run each actual retained-packet validation block without native I/O."""
        tree = ast.parse(MODULE_PATH.read_text())
        if method == "inspect_existing":
            adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LinuxAdapter")
            function = next(node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == method)
        else:
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == method)
        guarded = next(node for node in function.body if isinstance(node, ast.Try))
        start = next(index for index, node in enumerate(guarded.body)
                     if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "preexec_data" for target in node.targets))
        end = next(index for index in range(start + 1, len(guarded.body))
                   if isinstance(guarded.body[index], ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "barrier" for target in guarded.body[index].targets))
        wrapper = ast.parse("def validate(preexec_data, identity, identity_sha):\n    pass\n")
        wrapper.body[0].body = guarded.body[start + 1:end + 1] + [ast.Return(value=ast.Name(id="barrier", ctx=ast.Load()))]
        namespace = {"_literal_packet": MODULE._literal_packet, "_preexec_payload": MODULE._preexec_payload,
                     "EnforcementUnavailable": MODULE.EnforcementUnavailable}
        exec(compile(ast.fix_missing_locations(wrapper), "<retained-preexec-validation>", "exec"), namespace)
        return namespace["validate"](payload, value, "f" * 64)

    def test_retained_readers_accept_exact_packets(self) -> None:
        value = identity()
        barrier = {"descriptor": 5, "device": 7, "inode": 11}
        expected = MODULE._preexec_payload(value, "f" * 64, barrier)
        for method in ("inspect_existing", "_restore_child_context"):
            with self.subTest(method=method):
                self.assertEqual(self.validate_reader(method, dict(expected), value), barrier)

    def test_retained_readers_refuse_equal_scalar_type_aliases(self) -> None:
        value = identity()
        expected = MODULE._preexec_payload(value, "f" * 64, {"descriptor": 5, "device": 7, "inode": 11})
        invalid_fields = (("version", True), ("version", 1.0), ("pid", float(value.main_pid)),
                          ("start_ticks", float(value.start_ticks)), ("kernel_cap_bytes", float(value.cap_bytes)),
                          ("physical_hard_cap", 0), ("physical_hard_cap", 0.0),
                          ("dynamic_registry_cap", 0), ("dynamic_registry_cap", 0.0))
        for method in ("inspect_existing", "_restore_child_context"):
            for name, invalid in invalid_fields:
                with self.subTest(method=method, name=name, invalid=invalid), self.assertRaises((ValueError, RuntimeError)):
                    self.validate_reader(method, {**expected, name: invalid}, value)

    def test_retained_readers_refuse_non_dict_and_changed_schema(self) -> None:
        class Packet(dict[str, object]):
            pass

        class GetTrap:
            def get(self, key: str) -> object:
                raise AssertionError("Non-dict retained packet accessed get")

        value = identity()
        expected = MODULE._preexec_payload(value, "f" * 64, {"descriptor": 5, "device": 7, "inode": 11})
        missing = dict(expected)
        del missing["version"]
        for method in ("inspect_existing", "_restore_child_context"):
            for invalid in (None, [], False, 0, "{}", Packet(expected), GetTrap(), missing,
                            {**expected, "extra": 1}, {**expected, 1: "extra"}):
                with self.subTest(method=method, invalid_type=type(invalid).__name__), self.assertRaises(MODULE.EnforcementUnavailable):
                    self.validate_reader(method, invalid, value)

    def test_retained_readers_preserve_strict_nested_barrier_validation(self) -> None:
        value = identity()
        barrier = {"descriptor": 5, "device": 7, "inode": 11}
        expected = MODULE._preexec_payload(value, "f" * 64, barrier)
        invalid_barriers = (None, [], {"descriptor": 5, "device": 7}, {**barrier, "extra": 1},
                            {**barrier, "descriptor": True}, {**barrier, "descriptor": 1},
                            {**barrier, "device": 7.0}, {**barrier, "inode": 11.0})
        for method in ("inspect_existing", "_restore_child_context"):
            for invalid in invalid_barriers:
                with self.subTest(method=method, invalid=invalid), self.assertRaises((ValueError, RuntimeError)):
                    self.validate_reader(method, {**expected, "barrier": invalid}, value)


class TerminationStore:
    directory = 75

    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.path = Path("/synthetic/evidence")
        self.metadata = SimpleNamespace(st_dev=211, st_ino=223, st_uid=1000)
        self.callback = None
        self.sequence = 0

    def verify(self) -> None:
        self.trace.append("store_verify")

    def append(self, phase: str, payload: object) -> tuple[int, str]:
        self.trace.append(phase)
        self.sequence += 1
        if self.callback is not None:
            self.callback(phase)
        return self.sequence, "f" * 64


class TerminationAdapter(InertAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.trace: list[str] = []
        self.store = TerminationStore(self.trace)
        self.record = MODULE.LinuxProvisionalLaunch(self.identity, SimpleNamespace(store=self.store),
                                                   object(), object(), object())
        self.callback = None

    def terminate(self, record: object, value: object, deadline: float) -> None:
        self.calls.append("terminate")
        self.trace.append("terminate")
        self.deadlines.append(deadline)
        if self.callback is not None:
            self.callback()
        self.terminal = True


def native_termination_record() -> tuple[object, list[str]]:
    value = identity()
    trace: list[str] = []
    store = TerminationStore(trace)
    scope = MODULE._NativeScope(29, "/synthetic", 70, 71, (), pidfd_device=101, pidfd_inode=103)
    runner = MODULE._NativeRunner(object(), store, {"version": 1})
    endpoint = MODULE._NativeEndpoint(SimpleNamespace(fileno=lambda: 73), SimpleNamespace(fileno=lambda: 74),
                value.main_pid, value.uid, {}, 107, 109, 113, 127)
    return MODULE.LinuxProvisionalLaunch(value, runner, 72, scope, endpoint), trace


class TestLinuxOwnedTerminationContract(unittest.TestCase):
    def owned(self) -> tuple[object, object]:
        adapter = TerminationAdapter()
        handle = MODULE.OwnedLinuxLaunch(adapter.record, MODULE.LinuxDaemonLaunchPolicy(1, 67108864, 0), adapter)
        return handle, adapter

    def test_deadline_exact_float_before_callbacks(self) -> None:
        for invalid in (1005, True, None, float("nan"), float("inf"), 1000.0, 999.0):
            handle, adapter = self.owned()
            with self.subTest(invalid=invalid), patch.object(MODULE.time, "monotonic", return_value=1000.0), self.assertRaises((ValueError, TypeError, TimeoutError)):
                MODULE.terminate_owned_exact(handle, deadline=invalid)
            self.assertEqual(adapter.calls, [])
            self.assertEqual(adapter.trace, [])

    def test_public_owner_rejects_nonhandle_and_subtype(self) -> None:
        handle, adapter = self.owned()
        class Subtype(MODULE.OwnedLinuxLaunch):
            pass
        subtype = Subtype(handle.record, handle.policy, adapter)
        for invalid in (None, handle.record, evidence(adapter.identity), subtype):
            with self.subTest(invalid_type=type(invalid).__name__), patch.object(MODULE.time, "monotonic", return_value=1000.0), self.assertRaises((ValueError, TypeError)):
                MODULE.terminate_owned_exact(invalid, deadline=1005.0)
        self.assertEqual(adapter.trace, [])

    def test_intent_precedes_adapter_and_capture_precedes_stop(self) -> None:
        handle, adapter = self.owned()
        with patch.object(MODULE.time, "monotonic", return_value=1000.0):
            result = MODULE.terminate_owned_exact(handle, deadline=1100.0)
        self.assertIs(type(result), MODULE.LinuxExitReceipt)
        self.assertLess(adapter.trace.index("termination-intent"), adapter.trace.index("terminate"))
        self.assertLess(adapter.calls.index("capture"), adapter.calls.index("stop"))
        self.assertTrue(all(value == 1005.0 for value in adapter.deadlines))
        self.assertIsNone(getattr(MODULE._termination_context, "owner", None))

    def test_shorter_original_deadline_is_not_extended(self) -> None:
        handle, adapter = self.owned()
        with patch.object(MODULE.time, "monotonic", return_value=1000.0):
            MODULE.terminate_owned_exact(handle, deadline=1001.5)
        self.assertTrue(all(value == 1001.5 for value in adapter.deadlines))

    def test_durable_callback_mutation_refuses_before_adapter(self) -> None:
        handle, adapter = self.owned()
        adapter.store.callback = lambda phase: setattr(handle, "_adapter", object())
        with patch.object(MODULE.time, "monotonic", return_value=1000.0), self.assertRaises(RuntimeError):
            MODULE.terminate_owned_exact(handle, deadline=1005.0)
        self.assertEqual(adapter.calls, [])
        self.assertIn("termination-intent", adapter.trace)

    def test_adapter_reflective_policy_mutation_blocks_capture(self) -> None:
        handle, adapter = self.owned()
        adapter.callback = lambda: object.__setattr__(handle.policy, "cap_bytes", True)
        with patch.object(MODULE.time, "monotonic", return_value=1000.0), self.assertRaises((ValueError, RuntimeError)):
            MODULE.terminate_owned_exact(handle, deadline=1005.0)
        self.assertNotIn("capture", adapter.calls)
        self.assertNotIn("stop", adapter.calls)

    def test_store_swap_blocks_foreign_failure_write(self) -> None:
        handle, adapter = self.owned()
        foreign = TerminationStore([])
        adapter.callback = lambda: setattr(handle.record.runner, "store", foreign)
        with patch.object(MODULE.time, "monotonic", return_value=1000.0), self.assertRaises(RuntimeError):
            MODULE.terminate_owned_exact(handle, deadline=1005.0)
        self.assertEqual(foreign.trace, [])
        self.assertNotIn("stop", adapter.calls)

    def test_expiration_after_intent_blocks_new_effects(self) -> None:
        handle, adapter = self.owned()
        clock = [1000.0]
        adapter.store.callback = lambda phase: clock.__setitem__(0, 1006.0)
        with patch.object(MODULE.time, "monotonic", side_effect=lambda: clock[0]), self.assertRaises(TimeoutError):
            MODULE.terminate_owned_exact(handle, deadline=1005.0)
        self.assertEqual(adapter.calls, [])

    def native_sequence(self, *, observations: list[tuple[bool, bool]], signals: list[bool], waits: list[bool]) -> tuple[list[int], list[float], list[str]]:
        record, trace = native_termination_record()
        sent: list[int] = []
        wait_deadlines: list[float] = []
        def send(*args: object) -> bool:
            sent.append(int(args[3]))
            return signals.pop(0)
        def wait(*args: object) -> bool:
            wait_deadlines.append(args[3])
            return waits.pop(0)
        with ExitStack() as stack:
            stack.enter_context(patch.object(MODULE.time, "monotonic", return_value=1000.0))
            stack.enter_context(patch.object(MODULE, "_termination_native_snapshot", side_effect=observations))
            stack.enter_context(patch.object(MODULE, "_signal_owned_pidfd", side_effect=send))
            stack.enter_context(patch.object(MODULE, "_wait_pidfd_exit", side_effect=wait))
            stack.enter_context(patch.object(MODULE, "_wait_owned_terminal", side_effect=lambda *args: trace.append("terminal_verified")))
            MODULE.LinuxAdapter().terminate(record, record.identity, 1100.0)
        return sent, wait_deadlines, trace

    def test_native_term_then_conditional_kill(self) -> None:
        sent, waits, trace = self.native_sequence(observations=[(False, False)] * 3, signals=[False, True], waits=[False])
        self.assertEqual(sent, [int(MODULE.signal.SIGTERM), int(MODULE.signal.SIGKILL)])
        self.assertEqual(waits, [1001.0])
        self.assertLess(trace.index("termination-native-intent"), trace.index("termination-signal-intent"))
        self.assertLess(trace.index("terminal_verified"), trace.index("termination-manager-reaped"))

    def test_native_term_exit_skips_kill(self) -> None:
        sent, waits, _ = self.native_sequence(observations=[(False, False)] * 2, signals=[False], waits=[True])
        self.assertEqual(sent, [int(MODULE.signal.SIGTERM)])
        self.assertEqual(waits, [1001.0])

    def test_already_exited_skips_all_signals(self) -> None:
        sent, waits, trace = self.native_sequence(observations=[(True, True)], signals=[], waits=[])
        self.assertEqual(sent, [])
        self.assertEqual(waits, [])
        self.assertIn("termination-manager-reaped", trace)

    def test_exit_before_kill_skips_kill_signal(self) -> None:
        sent, waits, _ = self.native_sequence(observations=[(False, False), (False, False), (True, False)], signals=[False], waits=[False])
        self.assertEqual(sent, [int(MODULE.signal.SIGTERM)])
        self.assertEqual(waits, [1001.0])

    def test_manager_reap_wait_has_no_deadline_renewal(self) -> None:
        record, _ = native_termination_record()
        resources = MODULE._termination_resources(record)
        seen: list[float] = []
        def pause(read: list[int], write: list[int], errors: list[int], delay: float) -> tuple[list[int], list[int], list[int]]:
            self.assertEqual(read, [])
            seen.append(delay)
            return [], [], []
        with patch.object(MODULE.time, "monotonic", return_value=1000.0), patch.object(MODULE, "_termination_native_snapshot", side_effect=[(True, False), (True, True)]), patch.object(MODULE.select, "select", side_effect=pause):
            MODULE._wait_owned_terminal(MODULE.LinuxAdapter(), record, record.identity, resources, 1000.01)
        self.assertEqual(len(seen), 1)
        self.assertLessEqual(seen[0], 0.01 + 1e-9)

    def test_pidfd_target_requires_exact_native_identity(self) -> None:
        with patch.object(MODULE, "_read_path", return_value=b"Pid:\t43\nNSpid:\t43\n") as reader:
            self.assertIs(MODULE._pidfd_target(90, 43), True)
        self.assertEqual(reader.call_args.kwargs, {"bound": 4096})
        for raw in (b"Pid: 44\nNSpid: 44\n", b"Pid: 43\nNSpid: 44\n", b"Pid: 43\nPid: 43\nNSpid: 43\n", b"Pid: 43\n", b"Pid: -1\nNSpid: 43\n"):
            with self.subTest(raw=raw), patch.object(MODULE, "_read_path", return_value=raw), self.assertRaises((ValueError, RuntimeError)):
                MODULE._pidfd_target(90, 43)
        with patch.object(MODULE, "_read_path", return_value=b"Pid: -1\nNSpid: -1\n"):
            self.assertIs(MODULE._pidfd_target(90, 43), False)

    def pidfd_call(self, *, error: OSError | None = None, rebound: bool = False, duplicate: int = 90,
                   exit_on_esrch: bool = True) -> tuple[list[str], list[int]]:
        record, trace = native_termination_record()
        resources = MODULE._termination_resources(record)
        exited = [False]
        replaced = [False]
        closed: list[int] = []
        def fstat(fd: int) -> object:
            return SimpleNamespace(st_dev=101, st_ino=999 if fd == 90 and replaced[0] else 103)
        def ready(read: list[int], write: list[int], errors: list[int], timeout: float) -> tuple[list[int], list[int], list[int]]:
            return (list(read) if exited[0] else []), [], []
        def send(fd: int, requested_signal: int, info: object, flags: int) -> None:
            self.assertEqual((fd, info, flags), (90, None, 0))
            trace.append("pidfd_signal")
            if rebound:
                replaced[0] = True
            if isinstance(error, ProcessLookupError) and exit_on_esrch:
                exited[0] = True
            if error is not None:
                raise error
        def atomic(fd: int, command: int, minimum: int) -> int:
            self.assertEqual((fd, command, minimum), (72, 1030, 3))
            trace.append("atomic_duplicate")
            return duplicate
        with ExitStack() as stack:
            for target, name, replacement in ((MODULE.time, "monotonic", lambda: 1000.0),
                    (MODULE.os, "fstat", fstat), (MODULE.os, "close", lambda fd: closed.append(fd)),
                    (MODULE.select, "select", ready),
                    (MODULE, "_read_path", lambda *args, **kwargs: b"Pid: 43\nNSpid: 43\n"),
                    (MODULE.importlib, "import_module", lambda name: SimpleNamespace(F_DUPFD_CLOEXEC=1030, fcntl=atomic))):
                stack.enter_context(patch.object(target, name, replacement))
            stack.enter_context(patch.object(MODULE.signal, "pidfd_send_signal", side_effect=send, create=True))
            MODULE._signal_owned_pidfd(record, record.identity, resources, MODULE.signal.SIGTERM, 1005.0)
        return trace, closed

    def test_private_atomic_duplicate_only_is_signaled_and_closed(self) -> None:
        trace, closed = self.pidfd_call()
        self.assertLess(trace.index("atomic_duplicate"), trace.index("pidfd_signal"))
        self.assertEqual(closed, [90])
        self.assertNotIn(72, closed)

    def test_private_esrch_requires_actual_exit(self) -> None:
        trace, closed = self.pidfd_call(error=ProcessLookupError(3, "synthetic exited target"))
        self.assertIn("pidfd_signal", trace)
        self.assertEqual(closed, [90])

    def test_native_permission_error_never_falls_back(self) -> None:
        with self.assertRaises(PermissionError):
            self.pidfd_call(error=PermissionError(1, "synthetic denied"))

    def test_esrch_without_observed_exit_refuses(self) -> None:
        with self.assertRaises(RuntimeError):
            self.pidfd_call(error=ProcessLookupError(3, "synthetic unresolved target"), exit_on_esrch=False)

    def test_private_signal_selection_rejects_before_fd_observers(self) -> None:
        record, _ = native_termination_record()
        resources = MODULE._termination_resources(record)
        for invalid in (True, None, "TERM", 0, int(MODULE.signal.SIGUSR1)):
            with self.subTest(invalid=invalid), patch.object(MODULE.time, "monotonic", return_value=1000.0), patch.object(MODULE.os, "fstat") as observer, self.assertRaises(ValueError):
                MODULE._signal_owned_pidfd(record, record.identity, resources, invalid, 1005.0)
            observer.assert_not_called()

    def test_native_deadline_rejects_before_any_store_observer(self) -> None:
        record, trace = native_termination_record()
        for invalid in (1005, True, float("inf"), 1000.0):
            with self.subTest(invalid=invalid), patch.object(MODULE.time, "monotonic", return_value=1000.0), self.assertRaises((ValueError, TimeoutError)):
                MODULE.LinuxAdapter().terminate(record, record.identity, invalid)
        self.assertEqual(trace, [])

    def test_rebound_private_descriptor_is_not_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            self.pidfd_call(rebound=True)

    def test_original_descriptor_is_never_used_as_private_duplicate(self) -> None:
        with self.assertRaises(RuntimeError):
            self.pidfd_call(duplicate=72)

    def test_fresh_public_inspect_precedes_delegate(self) -> None:
        handle, adapter = self.owned()
        original = adapter.inspect
        def inspect(record: object, deadline: float) -> object:
            adapter.trace.append("fresh-inspect")
            return original(record, deadline)
        adapter.inspect = inspect
        with patch.object(MODULE.time, "monotonic", return_value=1000.0):
            MODULE.terminate_owned_exact(handle, deadline=1100.0)
        self.assertLess(adapter.trace.index("termination-intent"), adapter.trace.index("fresh-inspect"))
        self.assertLess(adapter.trace.index("fresh-inspect"), adapter.trace.index("terminate"))
        self.assertTrue(all(value == 1005.0 for value in adapter.deadlines))

    def test_public_empty_domain_skips_delegate(self) -> None:
        handle, adapter = self.owned()
        adapter.terminal = True
        with patch.object(MODULE.time, "monotonic", return_value=1000.0):
            result = MODULE.terminate_owned_exact(handle, deadline=1005.0)
        self.assertIs(type(result), MODULE.LinuxExitReceipt)
        self.assertNotIn("terminate", adapter.calls)
        self.assertIn("capture", adapter.calls)
        self.assertLess(adapter.calls.index("capture"), adapter.calls.index("stop"))

    def test_invalid_fresh_evidence_blocks_delegate(self) -> None:
        for field, invalid in (("mount_authenticated", False), ("namespace_authenticated", False),
                ("membership_authenticated", False), ("kernel_limit_preexec", False),
                ("cap_bytes", True), ("cap_bytes", 67108865), ("swap_bytes", 1),
                ("members", (43, 44)), ("members", [43]), ("peer_pid", 44),
                ("peer_start_ticks", 48), ("generation", "b" * 32),
                ("manifest_sha256", "a" * 64), ("leaf_memory_max", "0")):
            handle, adapter = self.owned()
            observed = evidence(adapter.identity)
            object.__setattr__(observed, field, invalid)
            adapter.inspect = lambda record, deadline: observed
            with self.subTest(field=field, invalid=invalid), patch.object(MODULE.time, "monotonic", return_value=1000.0), self.assertRaises((ValueError, RuntimeError)):
                MODULE.terminate_owned_exact(handle, deadline=1005.0)
            self.assertNotIn("terminate", adapter.calls)
            self.assertNotIn("capture", adapter.calls)
            self.assertNotIn("stop", adapter.calls)

    def test_fresh_evidence_exact_type_is_required(self) -> None:
        for invalid in (None, False, SimpleNamespace(**vars(evidence(identity())))):
            handle, adapter = self.owned()
            adapter.inspect = lambda record, deadline: invalid
            with self.subTest(invalid_type=type(invalid).__name__), patch.object(MODULE.time, "monotonic", return_value=1000.0), self.assertRaises((TypeError, ValueError, RuntimeError)):
                MODULE.terminate_owned_exact(handle, deadline=1005.0)
            self.assertNotIn("terminate", adapter.calls)
            self.assertNotIn("capture", adapter.calls)
            self.assertNotIn("stop", adapter.calls)

    def test_fresh_inspect_owner_mutation_blocks_delegate(self) -> None:
        mutations = (
            ("record", lambda handle, adapter: setattr(handle, "record", MODULE.LinuxProvisionalLaunch(adapter.identity, object(), object(), object(), object()))),
            ("policy", lambda handle, adapter: setattr(handle, "policy", MODULE.LinuxDaemonLaunchPolicy(1, 67108864, 0))),
            ("adapter", lambda handle, adapter: setattr(handle, "_adapter", object())),
            ("released", lambda handle, adapter: setattr(handle, "_released", True)),
            ("identity", lambda handle, adapter: object.__setattr__(handle.record, "identity", replace(adapter.identity, start_ticks=48))),
            ("runner", lambda handle, adapter: object.__setattr__(handle.record, "runner", SimpleNamespace(store=adapter.store))),
            ("pidfd", lambda handle, adapter: object.__setattr__(handle.record, "pidfd", object())),
            ("scope", lambda handle, adapter: object.__setattr__(handle.record, "scope_fd", object())),
            ("endpoint", lambda handle, adapter: object.__setattr__(handle.record, "endpoint", object())),
            ("store", lambda handle, adapter: setattr(handle.record.runner, "store", TerminationStore([]))),
            ("directory", lambda handle, adapter: setattr(adapter.store, "directory", 76)),
            ("metadata", lambda handle, adapter: setattr(adapter.store, "metadata", SimpleNamespace(st_dev=211, st_ino=223, st_uid=1000))),
            ("path", lambda handle, adapter: setattr(adapter.store, "path", Path("/synthetic/rebound"))),
        )
        for empty in (False, True):
            for name, mutate in mutations:
                handle, adapter = self.owned()
                observed = evidence(adapter.identity)
                if empty:
                    observed = replace(observed, members=())
                def inspect(record: object, deadline: float) -> object:
                    mutate(handle, adapter)
                    return observed
                adapter.inspect = inspect
                with self.subTest(empty=empty, mutation=name), patch.object(MODULE.time, "monotonic", return_value=1000.0), self.assertRaises((TypeError, ValueError, RuntimeError)):
                    MODULE.terminate_owned_exact(handle, deadline=1005.0)
                self.assertNotIn("terminate", adapter.calls)
                self.assertNotIn("capture", adapter.calls)
                self.assertNotIn("stop", adapter.calls)

    def test_public_inspect_expiration_blocks_delegate(self) -> None:
        handle, adapter = self.owned()
        clock = [1000.0]
        def inspect(record: object, deadline: float) -> object:
            clock[0] = 1006.0
            return evidence(adapter.identity)
        adapter.inspect = inspect
        with patch.object(MODULE.time, "monotonic", side_effect=lambda: clock[0]), self.assertRaises(TimeoutError):
            MODULE.terminate_owned_exact(handle, deadline=1005.0)
        self.assertNotIn("terminate", adapter.calls)
        self.assertNotIn("capture", adapter.calls)
        self.assertNotIn("stop", adapter.calls)

    def test_public_inspect_failure_blocks_delegate(self) -> None:
        handle, adapter = self.owned()
        def inspect(record: object, deadline: float) -> object:
            raise PermissionError(1, "synthetic observer denial")
        adapter.inspect = inspect
        with patch.object(MODULE.time, "monotonic", return_value=1000.0), self.assertRaises(PermissionError):
            MODULE.terminate_owned_exact(handle, deadline=1005.0)
        self.assertNotIn("terminate", adapter.calls)
        self.assertNotIn("capture", adapter.calls)
        self.assertNotIn("stop", adapter.calls)

    def test_live_fresh_evidence_requires_terminal_after_delegate(self) -> None:
        handle, adapter = self.owned()
        adapter.terminate = lambda record, value, deadline: adapter.calls.append("terminate")
        with patch.object(MODULE.time, "monotonic", return_value=1000.0), self.assertRaises(RuntimeError):
            MODULE.terminate_owned_exact(handle, deadline=1005.0)
        self.assertIn("terminate", adapter.calls)
        self.assertNotIn("capture", adapter.calls)
        self.assertNotIn("stop", adapter.calls)

    def test_empty_fresh_evidence_requires_terminal_receipt(self) -> None:
        handle, adapter = self.owned()
        adapter.terminal = True
        adapter.capture = lambda record, value, deadline: None
        with patch.object(MODULE.time, "monotonic", return_value=1000.0), self.assertRaises((TypeError, ValueError, RuntimeError)):
            MODULE.terminate_owned_exact(handle, deadline=1005.0)
        self.assertNotIn("terminate", adapter.calls)
        self.assertNotIn("stop", adapter.calls)


if __name__ == "__main__":
    unittest.main()
