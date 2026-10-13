"""Pure Stage A policy tests. Never launch a child, signal, sensor or fixture."""
from __future__ import annotations

import copy
from contextlib import ExitStack
import os
import stat
import sys
import types
import unittest
from pathlib import Path
from unittest import mock



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
    return module


_TARGET_OS = _target_os() if _host_sys.platform == "win32" else _host_os
if _host_sys.platform == "win32":
    Path = _TargetPath


def _inert_fcntl():
    module = _FixtureModule("fcntl")
    module.F_GETFL, module.LOCK_EX, module.LOCK_NB = 3, 2, 4
    module.fcntl = module.flock = _forbidden_target_effect
    return module


def _install_inert_fcntl(test):
    if _host_sys.platform == "win32":
        stack = ExitStack()
        test.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(sys.modules, {"fcntl": _inert_fcntl()}))


if _host_sys.platform == "win32":
    os = _TARGET_OS


def load_source(name: str, filename: str) -> types.ModuleType:
    path = Path(__file__).parent / "native" / filename
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return _prepare_target_module(module)


POLICY = load_source("native_acceptance_policy_test_v1", "mac-native-acceptance-v1.py")
BOOT = load_source("native_acceptance_bootstrap_test_v1", "mac-native-bootstrap-v1.py")


def case(kind: str = "capture-a1") -> dict:
    name = {"capture-revision": "capture-a2", "invalid-state": "negative-a3"}.get(kind, kind)
    database = "capture-a2" if kind == "capture-revision" else "negative-a3" if kind == "invalid-state" else kind
    row = {"id": kind, "source": str(POLICY.RUN_ROOT / "home/.claude/projects/native-acceptance-v1" / (name + ".jsonl")),
           "source_sha256": "a" * 64, "source_bytes": 24, "source_identity": [1, 2, 24, 3, 4],
           "session": "synthetic-native-" + name, "database": str(POLICY.RUN_ROOT / (database + "-v1.db")), "previous": None}
    if kind == "flock-a4":
        for field in ("source", "source_sha256", "source_bytes", "source_identity"):
            row[field] = None
    if kind == "capture-revision":
        row["previous"] = {"key": "b" * 64, "sha256": "c" * 64, "byte_count": 12}
    return row


def manifest() -> dict:
    return {"version": 1, "stage": "A", "synthetic": True, "root_execution_go": True,
            "run_root": str(POLICY.RUN_ROOT), "home": str(POLICY.RUN_ROOT / "home"),
            "source_root": str(POLICY.SOURCE_ROOT), "source_commit": "a" * 40,
            "sources": {name: "a" * 64 for name in POLICY.SOURCE_FILES},
            "supervisor_sha256": "b" * 64, "bootstrap_sha256": "c" * 64,
            "limits": dict(POLICY.LIMITS), "cases": [case()]}


def mutation(seq: int, event: str, path: Path, flags: int = 0, source: Path | None = None) -> dict:
    return {"type": "mutation", "version": 1, "seq": seq, "event": event,
            "path": str(path), "flags": flags, "source": str(source) if source is not None else None}


class FakeProtocol:
    def __init__(self) -> None:
        self.sent: list[object] = []
        self.answers: list[dict] = []

    def emit(self, value: object) -> None:
        self.sent.append(value)

    def receive(self, maximum: int, seconds: float) -> dict:
        if maximum != 2048 or seconds != 0.1:
            raise AssertionError("Changed frame or ACK bound")
        return self.answers.pop(0)


class FakeSelector:
    def close(self) -> None:
        return None


class FakePsutilError(Exception):
    pass


class PurePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        _install_inert_fcntl(self)

    def test_fixed_limits_and_proof_scope(self) -> None:
        self.assertEqual(POLICY.LIMITS, BOOT.LIMITS)
        self.assertEqual(POLICY.FREE_MIN - POLICY.LIMITS["rss_abort_bytes"], 201326592)
        self.assertEqual(POLICY.PROOF_SCOPE, BOOT.PROOF_SCOPE)
        self.assertEqual(set(POLICY.SOURCE_FILES), {path for _, path in BOOT.SOURCE_MODULES})
        self.assertNotIn("installed", POLICY.PROOF_SCOPE)

    def test_exact_integer_and_hash(self) -> None:
        for value in (True, False, 1.0, None, -1, 1 << 80):
            with self.subTest(value=value), self.assertRaises(POLICY.SafetyError):
                POLICY.integer(value)
        for value in (True, "A" * 64, "a" * 63, "a" * 65, None):
            with self.subTest(value=value), self.assertRaises(POLICY.SafetyError):
                POLICY.hash_text(value)
        self.assertEqual(POLICY.integer(0), 0)
        self.assertEqual(POLICY.hash_text("a" * 64), "a" * 64)

    def test_sensor_admission_exact_floor_and_abi(self) -> None:
        samples = [POLICY.Sensor(1_000_000_000 + n * 1_000_000_000, 1, POLICY.FREE_MIN, 100) for n in range(3)]
        POLICY.admission(samples, 3_500_000_000)
        changes = [("pressure", 0), ("pressure", 2), ("pressure", 3), ("pressure", 4),
                   ("pressure", True), ("free_bytes", POLICY.FREE_MIN - 1),
                   ("free_bytes", True), ("swap_used_bytes", 101), ("monotonic_ns", 1_000_000_000)]
        for field, value in changes:
            altered = [POLICY.asdict(row) for row in samples]
            altered[-1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(POLICY.SafetyError):
                POLICY.admission([POLICY.Sensor(**row) for row in altered], 3_500_000_000)
        for now in (2_999_999_999, 3_500_000_001):
            with self.subTest(now=now), self.assertRaises(POLICY.SafetyError):
                POLICY.admission(samples, now)
        with self.assertRaises(POLICY.SafetyError):
            POLICY.admission(samples[:2], 3_500_000_000)
        with self.assertRaises(POLICY.SafetyError):
            POLICY.admission([POLICY.Sensor(1 + n, 1, POLICY.FREE_MIN, 0) for n in range(3)], 3)

    def test_manifest_and_case_fences(self) -> None:
        self.assertEqual(POLICY.validate_manifest(manifest()), manifest())
        for kind in POLICY.CASES:
            POLICY.validate_case(case(kind))
        changes = [("version", True), ("stage", "B"), ("synthetic", False), ("root_execution_go", False),
                   ("run_root", "/tmp"), ("source_root", "/Users/j/dev/TrueMemory"),
                   ("sources", {}), ("cases", []), ("cases", [case(), case()])]
        for field, value in changes:
            altered = manifest()
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(POLICY.SafetyError):
                POLICY.validate_manifest(altered)
        for field, value in (("id", []), ("session", "unknown"), ("database", "/tmp/personal.db"),
                             ("source", "/Users/j/dev/TrueMemory/transcript.jsonl"), ("source_bytes", True),
                             ("source_identity", [1, 2, 23, 3, 4])):
            altered = case()
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(POLICY.SafetyError):
                POLICY.validate_case(altered)
        altered = manifest()
        altered["limits"]["wall_seconds"] = 6
        with self.assertRaises(POLICY.SafetyError):
            POLICY.validate_manifest(altered)
        altered = manifest()
        altered["limits"]["core_bytes"] = False
        with self.assertRaises(POLICY.SafetyError):
            POLICY.validate_manifest(altered)

    def test_paths_and_versioned_results(self) -> None:
        for name in ("v1", "v2"):
            self.assertEqual(POLICY.evidence_paths(POLICY.RUN_ROOT / ("manifest-" + name + ".json")),
                             (POLICY.RUN_ROOT / ("results-" + name + ".json"), name))
        for text in ("relative", str(POLICY.RUN_ROOT) + "/../outside", str(POLICY.RUN_ROOT) + "//escape", "/tmp/file"):
            with self.subTest(text=text), self.assertRaises(POLICY.SafetyError):
                POLICY.canonical_path(text, POLICY.RUN_ROOT)
        with self.assertRaises(POLICY.SafetyError):
            POLICY.evidence_paths(POLICY.RUN_ROOT / "manifest-v3.json")

    def test_identity_is_current_owned_incarnation(self) -> None:
        row = POLICY.Identity(72887, 42, 12345, 72887)
        self.assertTrue(POLICY.same_owned_identity(row, row, 42))
        for changed in (POLICY.Identity(72887, 42, 12346, 72887), POLICY.Identity(72887, 43, 12345, 72887),
                        POLICY.Identity(72887, 42, 12345, 1), POLICY.Identity(True, 42, 12345, 1)):
            with self.subTest(changed=changed):
                self.assertFalse(POLICY.same_owned_identity(row, changed, 42))

    def test_ready_and_result_exact_protocol(self) -> None:
        ready = {"type": "ready", "version": 1, "pid": 100, "audit_installed": True,
                 "limits": {"cpu": [2, 3], "core": [0, 0], "nofile": [64, 64], "fsize": [1048576, 1048576]}}
        POLICY.validate_ready(ready, 100)
        for field, value in (("pid", 101), ("version", True), ("audit_installed", 1), ("limits", {})):
            altered = copy.deepcopy(ready)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(POLICY.SafetyError):
                POLICY.validate_ready(altered, 100)
        altered = copy.deepcopy(ready)
        altered["limits"]["core"] = [False, False]
        with self.assertRaises(POLICY.SafetyError):
            POLICY.validate_ready(altered, 100)
        done = {"type": "result", "version": 1, "case": "capture-a1", "passed": True,
                "proof_scope": POLICY.PROOF_SCOPE, "import_mode": "isolated_source_module", "sources": manifest()["sources"], "details": {}}
        POLICY.validate_result(done, case(), manifest()["sources"])
        for field, value in (("passed", 1), ("version", True), ("proof_scope", "installed_host"),
                             ("case", "capture-a2"), ("sources", {}), ("details", [])):
            altered = dict(done)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(POLICY.SafetyError):
                POLICY.validate_result(altered, case(), manifest()["sources"])

    def test_mutation_positive_and_one_pending_slot(self) -> None:
        policy = POLICY.MutationPolicy(case())
        policy.decide(mutation(1, "os.mkdir", policy.account, 0o700))
        lock = policy.account / "state.lock"
        policy.decide(mutation(2, "open", lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW))
        pending = policy.account / ("state-" + "a" * 32 + ".pending")
        policy.decide(mutation(3, "open", pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW))
        grant = policy.decide(mutation(4, "os.rename", policy.account / "state.json", source=pending))
        self.assertEqual(grant["path"], str(policy.account / "state.json"))
        with self.assertRaises(POLICY.SafetyError):
            policy.decide(mutation(5, "open", policy.account / ("state-" + "b" * 32 + ".pending"), os.O_WRONLY | os.O_CREAT | os.O_EXCL))

    def test_mutation_negative_controls(self) -> None:
        for event, filename, flags, source in (("open", "state.json", os.O_WRONLY | os.O_CREAT, None),
                                              ("open", "state.lock", os.O_RDWR | os.O_CREAT | os.O_TRUNC, None),
                                              ("open", "unknown", os.O_WRONLY | os.O_CREAT | os.O_EXCL, None),
                                              ("os.mkdir", "foreign", 0o700, None),
                                              ("os.mkdir", "captures", 0o755, None),
                                              ("os.rename", "state.json", 0, POLICY.RUN_ROOT / "arbitrary"),
                                              ("unlink", "state.json", 0, None)):
            policy = POLICY.MutationPolicy(case())
            with self.subTest(event=event, filename=filename), self.assertRaises(POLICY.SafetyError):
                policy.decide(mutation(1, event, policy.account / filename, flags, source))
        for sequence in (0, 2, True, 33):
            policy = POLICY.MutationPolicy(case())
            with self.subTest(sequence=sequence), self.assertRaises(POLICY.SafetyError):
                policy.decide(mutation(sequence, "os.mkdir", policy.account, 0o700))
        policy = POLICY.MutationPolicy(case("negative-a3"))
        with self.assertRaises(POLICY.SafetyError):
            policy.decide(mutation(1, "os.mkdir", policy.account, 0o700))

    def test_fd_wrappers_bind_latest_consumed_intent(self) -> None:
        import fcntl
        row = case()
        guard = BOOT.AuditGuard(FakeProtocol(), {"case": row}, POLICY)
        path = Path(row["source"])
        info = types.SimpleNamespace(st_dev=1, st_ino=2, st_uid=42, st_mode=stat.S_IFREG | 0o600)
        with mock.patch.object(BOOT.os, "fstat", return_value=info), mock.patch.object(BOOT.os, "getuid", return_value=42), mock.patch.object(fcntl, "fcntl", return_value=os.O_RDONLY), mock.patch.object(Path, "lstat", return_value=info), mock.patch.object(Path, "is_symlink", return_value=False):
            guard.open_intents[:] = [(path, os.O_RDONLY)]
            guard.descriptor(7, "r")
            self.assertEqual(guard.known_fds, {7: (1, 2)})
            with self.assertRaises(BOOT.BootstrapError):
                guard.descriptor(7, "rb")
            info.st_ino = 3
            guard.open_intents[:] = [(path, os.O_RDONLY)]
            guard.descriptor(7, "rb")
            self.assertEqual(guard.known_fds, {7: (1, 3)})
            guard.open_intents[:] = [(path, os.O_WRONLY)]
            with self.assertRaises(BOOT.BootstrapError):
                guard.descriptor(7, "rb")
            self.assertFalse(guard.open_intents)
            for fd, mode in ((0, "r"), (7, "a"), (7, "r+"), (True, "r")):
                with self.subTest(fd=fd, mode=mode), self.assertRaises(BOOT.BootstrapError):
                    guard.descriptor(fd, mode)

    def test_audit_preopen_grant_and_denials(self) -> None:
        protocol = FakeProtocol()
        guard = BOOT.AuditGuard(protocol, {"case": case()}, POLICY)
        path = POLICY.account_directory(case()) / "state.lock"
        protocol.answers.append({"type": "grant", "version": 1, "seq": 1, "path": str(path), "purpose": "Owned fixed lock"})
        guard.path = lambda value: Path(value)
        guard("open", (str(path), None, os.O_RDWR | os.O_CREAT))
        self.assertEqual(len(protocol.sent), 1)
        self.assertEqual(guard.open_intents, [(path, os.O_RDWR)])
        guard("truememory.native.audit_probe", ("fixed-stage-a-v1",))
        self.assertTrue(guard.activated)
        for event, args in (("socket.connect", ()), ("subprocess.Popen", ()), ("os.remove", (str(path), -1)),
                            ("sqlite3.connect", ("memory",)), ("resource.setrlimit", (0, (1, 1))),
                            ("import", ("truememory.engine", None)), ("import", ("torch", None)),
                            ("os.mkdir", (str(path), 0o700, 1)), ("os.rename", (str(path), str(path), 1, -1))):
            with self.subTest(event=event), self.assertRaises(BOOT.BootstrapError):
                guard(event, args)

    def test_prelaunch_refusals_retain_nonexecuted_result(self) -> None:
        fake = types.ModuleType("fake_psutil")
        fake.Error = FakePsutilError
        with mock.patch.object(POLICY.selectors, "DefaultSelector", return_value=FakeSelector()), mock.patch.object(POLICY, "sample", side_effect=POLICY.SafetyError("pressure_or_headroom")), mock.patch.object(POLICY.subprocess, "Popen") as launch:
            result = POLICY.execute_case(manifest(), case(), fake, None)
            self.assertFalse(result["executed"])
            self.assertFalse(result["passed"])
            self.assertTrue(result["reaped"])
            self.assertIn("pressure_or_headroom", result["blocked"])
            launch.assert_not_called()
        samples = [POLICY.Sensor(1_000_000_000 + n * 1_000_000_000, 1, POLICY.FREE_MIN, 100) for n in range(3)]
        with mock.patch.object(POLICY.selectors, "DefaultSelector", return_value=FakeSelector()), mock.patch.object(POLICY, "sample", side_effect=samples), mock.patch.object(POLICY.time, "sleep"), mock.patch.object(POLICY.time, "monotonic_ns", return_value=3_000_000_001), mock.patch.object(POLICY, "owned_path", side_effect=POLICY.SafetyError("fixture_changed")), mock.patch.object(POLICY.subprocess, "Popen") as launch:
            result = POLICY.execute_case(manifest(), case(), fake, None)
            self.assertFalse(result["executed"])
            self.assertFalse(result["passed"])
            self.assertIn("fixture_changed", result["blocked"])
            self.assertEqual(len(result["sensors"]), 3)
            launch.assert_not_called()

    def test_pipe_deadline_and_partial_writes_are_pure(self) -> None:
        with mock.patch.object(POLICY.time, "monotonic", return_value=1), mock.patch.object(POLICY.os, "write", side_effect=[2, 3]) as writer:
            POLICY.write_pipe(7, b"12345", 1.1)
            self.assertEqual(writer.call_count, 2)
        with mock.patch.object(POLICY.time, "monotonic", return_value=1), mock.patch.object(POLICY.os, "write", side_effect=[BlockingIOError(), 3]), mock.patch.object(POLICY.select, "select", return_value=([], [7], [])) as poll:
            POLICY.write_pipe(7, b"123", 1.1)
            self.assertEqual(poll.call_count, 1)
        with mock.patch.object(POLICY.time, "monotonic", return_value=1), mock.patch.object(POLICY.os, "write") as writer:
            with self.assertRaises(POLICY.SafetyError):
                POLICY.write_pipe(7, b"123", 0.9)
            writer.assert_not_called()

    def test_owned_popen_fallback_and_group_probe_are_pure(self) -> None:
        fake = types.ModuleType("fake_psutil")
        fake.NoSuchProcess = FakePsutilError
        child = mock.Mock()
        child.poll.return_value = None
        child.wait.return_value = 0
        child.returncode = 0
        result = {"reaped": False, "descendants": [], "descendant_containment": "not_observed"}
        with mock.patch.object(POLICY.time, "monotonic", return_value=1):
            POLICY.terminate_owned(child, None, [], fake, result)
        child.send_signal.assert_called_once_with(POLICY.signal.SIGTERM)
        self.assertTrue(result["reaped"])
        self.assertEqual(result["termination_authority"], "unreaped_direct_popen_child")
        identity = POLICY.Identity(100, 42, 123, 100)
        changed = POLICY.Identity(100, 42, 124, 100)
        child = mock.Mock()
        child.poll.return_value = None
        result = {"reaped": False, "descendants": [], "descendant_containment": "not_observed"}
        with mock.patch.object(POLICY.time, "monotonic", return_value=1), mock.patch.object(POLICY, "observe_identity", return_value=changed), mock.patch.object(POLICY.os, "getuid", return_value=42), mock.patch.object(POLICY.os, "killpg") as signal_group:
            with self.assertRaises(POLICY.SafetyError):
                POLICY.terminate_owned(child, identity, [], fake, result)
            child.send_signal.assert_not_called()
            signal_group.assert_not_called()
        with mock.patch.object(POLICY.os, "killpg", side_effect=ProcessLookupError()):
            self.assertTrue(POLICY.group_gone(100))
        with mock.patch.object(POLICY.os, "killpg", return_value=None):
            self.assertFalse(POLICY.group_gone(100))
        with mock.patch.object(POLICY.os, "killpg", side_effect=PermissionError()):
            self.assertFalse(POLICY.group_gone(100))
        member = POLICY.Identity(101, 42, 123, 100)
        self.assertTrue(POLICY.member_identity_matches(member, member, 100, 42))
        self.assertFalse(POLICY.member_identity_matches(member, member, 101, 42))

    def test_protocol_bounds_use_no_real_stdio(self) -> None:
        protocol = BOOT.Protocol()
        with mock.patch.object(BOOT.os, "write", side_effect=lambda fd, data: len(data)) as writer:
            protocol.emit({"type": "synthetic"})
            self.assertEqual(writer.call_count, 1)
            with self.assertRaises(BOOT.BootstrapError):
                protocol.emit({"payload": "x" * 2048})
            protocol.written_bytes = BOOT.LIMITS["output_bytes"]
            with self.assertRaises(BOOT.BootstrapError):
                protocol.emit({"type": "synthetic"})
        with mock.patch.object(BOOT.select, "select", return_value=([], [], [])):
            with self.assertRaises(BOOT.BootstrapError):
                BOOT.Protocol().receive(2048, 0.1)
        protocol = BOOT.Protocol()
        with mock.patch.object(BOOT.select, "select", return_value=([0], [], [])), mock.patch.object(BOOT.os, "read", return_value=b"x" * 2049):
            with self.assertRaises(BOOT.BootstrapError):
                protocol.receive(2048, 0.1)


if __name__ == "__main__":
    unittest.main()
