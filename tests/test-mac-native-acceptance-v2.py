"""Pure Stage A policy tests. Never launch a child, signal, sensor or fixture."""
from __future__ import annotations

import ast
import copy
import json
import os
import stat
import sys
import types
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator
from unittest import mock


def load_source(name: str, filename: str) -> types.ModuleType:
    path = Path(__file__).parent / "native" / filename
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


POLICY = load_source("native_acceptance_policy_test_v2", "mac-native-acceptance-v2.py")
BOOT = load_source("native_acceptance_bootstrap_test_v2", "mac-native-bootstrap-v2.py")


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


@contextmanager
def main_scope(version: str = "v3", fixture: dict | None = None) -> Iterator[dict[str, mock.Mock]]:
    row = copy.deepcopy(fixture) if fixture is not None else manifest()
    parser = mock.Mock()
    parser.parse_args.return_value = types.SimpleNamespace(manifest=str(POLICY.RUN_ROOT / ("manifest-" + version + ".json")), execute_root_authorized=True)
    fake = types.ModuleType("fake_psutil")
    fake.Error = FakePsutilError
    with ExitStack() as patches:
        patches.enter_context(mock.patch.object(POLICY.argparse, "ArgumentParser", return_value=parser))
        patches.enter_context(mock.patch.object(POLICY.sys, "platform", "darwin"))
        patches.enter_context(mock.patch.object(POLICY.sys, "flags", types.SimpleNamespace(isolated=True, no_site=True, dont_write_bytecode=True)))
        patches.enter_context(mock.patch.object(POLICY.sys, "dont_write_bytecode", True))
        patches.enter_context(mock.patch.object(POLICY.os, "umask", return_value=0o077))
        calls = {
            "owned_path": patches.enter_context(mock.patch.object(POLICY, "owned_path")),
            "read_bounded": patches.enter_context(mock.patch.object(POLICY, "read_bounded", return_value=POLICY.encoded(row))),
            "digest": patches.enter_context(mock.patch.object(POLICY, "digest", return_value="b" * 64)),
            "new_log": patches.enter_context(mock.patch.object(POLICY, "new_log", return_value=7)),
            "pinned_runtime": patches.enter_context(mock.patch.object(POLICY, "pinned_runtime", return_value=fake)),
            "PressureReader": patches.enter_context(mock.patch.object(POLICY, "PressureReader")),
            "execute_case": patches.enter_context(mock.patch.object(POLICY, "execute_case", return_value={"passed": False, "reaped": True, "executed": False, "blocked": "pressure_or_headroom"})),
            "write_all": patches.enter_context(mock.patch.object(POLICY, "write_all")),
            "fsync": patches.enter_context(mock.patch.object(POLICY.os, "fsync")),
            "close": patches.enter_context(mock.patch.object(POLICY.os, "close")),
        }
        yield calls


class PurePolicyTests(unittest.TestCase):
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
        for name in ("v" + str(number) for number in range(1, 100)):
            self.assertEqual(POLICY.evidence_paths(POLICY.RUN_ROOT / ("manifest-" + name + ".json")),
                             (POLICY.RUN_ROOT / ("results-" + name + ".json"), name))
        for text in ("relative", str(POLICY.RUN_ROOT) + "/../outside", str(POLICY.RUN_ROOT) + "//escape", "/tmp/file"):
            with self.subTest(text=text), self.assertRaises(POLICY.SafetyError):
                POLICY.canonical_path(text, POLICY.RUN_ROOT)
        for name in ("v0", "v01", "v100", "v3x", "v3\n", "v３", "v٣"):
            with self.subTest(name=name), self.assertRaises(POLICY.SafetyError):
                POLICY.evidence_paths(POLICY.RUN_ROOT / ("manifest-" + name + ".json"))
        with self.assertRaises(POLICY.SafetyError):
            POLICY.evidence_paths(POLICY.RUN_ROOT.parent / "manifest-v3.json")

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


    def test_attempt_versions_are_canonical_before_io(self) -> None:
        for number in range(1, 100):
            self.assertEqual(POLICY.validate_attempt_version("v" + str(number)), "v" + str(number))
        fake = types.ModuleType("fake_psutil")
        fake.Error = FakePsutilError
        invalid = (True, False, None, 3, b"v3", [], "v0", "v00", "v01", "v100", "V3", "v3\n", "v３", "v٣", "../v3", "v3.log")
        for version in invalid:
            with self.subTest(version=version), mock.patch.object(POLICY.selectors, "DefaultSelector") as selector, mock.patch.object(POLICY, "sample") as sensor, mock.patch.object(POLICY, "new_log") as writer, mock.patch.object(POLICY.subprocess, "Popen") as launch:
                with self.assertRaises(POLICY.SafetyError):
                    POLICY.validate_attempt_version(version)
                with self.assertRaises(POLICY.SafetyError):
                    POLICY.execute_case(manifest(), case(), fake, None, attempt_version=version)
                selector.assert_not_called()
                sensor.assert_not_called()
                writer.assert_not_called()
                launch.assert_not_called()

    def test_actual_retry_logs_and_legacy_default(self) -> None:
        fake = types.ModuleType("fake_psutil")
        fake.Error = FakePsutilError
        info = types.SimpleNamespace(st_dev=1, st_ino=2, st_size=24, st_mtime_ns=3, st_ctime_ns=4)
        samples = [POLICY.Sensor(1_000_000_000 + n * 1_000_000_000, 1, POLICY.FREE_MIN, 100) for n in range(3)]
        fixture = case()
        fixture_bytes = b"x" * 24
        fixture["source_sha256"] = POLICY.digest(fixture_bytes)
        plan = manifest()
        source_bytes = {POLICY.SOURCE_ROOT / name: ("synthetic:" + name).encode("utf-8") for name in POLICY.SOURCE_FILES}
        plan["sources"] = {name: POLICY.digest(source_bytes[POLICY.SOURCE_ROOT / name]) for name in POLICY.SOURCE_FILES}
        bootstrap_path = POLICY.SOURCE_ROOT / "tests/native/mac-native-bootstrap-v2.py"
        bootstrap_bytes = b"fixed-bootstrap-v2"
        plan["bootstrap_sha256"] = POLICY.digest(bootstrap_bytes)

        def read_fixture(path: Path, maximum: int, *, private: bool = True) -> bytes:
            if path == Path(fixture["source"]):
                self.assertEqual((maximum, private), (4096, True))
                return fixture_bytes
            self.assertEqual((maximum, private), (262144, False))
            if path == bootstrap_path:
                return bootstrap_bytes
            if path in source_bytes:
                return source_bytes[path]
            raise AssertionError("Unplanned synthetic source read")

        for version in (None, "v3", "v99"):
            expected = version if version is not None else "v1"
            with self.subTest(version=version), ExitStack() as patches:
                patches.enter_context(mock.patch.object(POLICY.selectors, "DefaultSelector", return_value=FakeSelector()))
                patches.enter_context(mock.patch.object(POLICY, "sample", side_effect=samples))
                patches.enter_context(mock.patch.object(POLICY.time, "sleep"))
                patches.enter_context(mock.patch.object(POLICY.time, "monotonic_ns", return_value=3_000_000_001))
                patches.enter_context(mock.patch.object(POLICY, "owned_path", return_value=info))
                reader = patches.enter_context(mock.patch.object(POLICY, "read_bounded", side_effect=read_fixture))
                writer = patches.enter_context(mock.patch.object(POLICY, "new_log", side_effect=[7, 8, POLICY.SafetyError("stop_before_launch")]))
                patches.enter_context(mock.patch.object(POLICY.os, "fsync"))
                patches.enter_context(mock.patch.object(POLICY.os, "close"))
                launch = patches.enter_context(mock.patch.object(POLICY.subprocess, "Popen"))
                result = POLICY.execute_case(plan, fixture, fake, None) if version is None else POLICY.execute_case(plan, fixture, fake, None, attempt_version=version)
                self.assertEqual([call.args[0] for call in writer.call_args_list], [POLICY.RUN_ROOT / "logs" / ("capture-a1-" + channel + "-" + expected + ".log") for channel in ("stdout", "stderr", "events")])
                self.assertEqual(reader.call_args_list[-1].args[0], POLICY.SOURCE_ROOT / "tests/native/mac-native-bootstrap-v2.py")
                self.assertFalse(result["executed"])
                self.assertFalse(result["passed"])
                self.assertTrue(result["reaped"])
                self.assertIn("stop_before_launch", result["blocked"])
                launch.assert_not_called()

    def test_main_reserves_result_before_runtime_and_case(self) -> None:
        with main_scope("v3") as calls:
            def runtime() -> types.ModuleType:
                self.assertEqual(calls["new_log"].call_count, 1)
                calls["execute_case"].assert_not_called()
                fake = types.ModuleType("fake_psutil")
                fake.Error = FakePsutilError
                return fake
            calls["pinned_runtime"].side_effect = runtime
            self.assertEqual(POLICY.main(), 1)
            self.assertEqual(calls["new_log"].call_args.args[0], POLICY.RUN_ROOT / "results-v3.json")
            calls["new_log"].assert_called_once()
            self.assertEqual(calls["execute_case"].call_args.kwargs, {"attempt_version": "v3"})
            payload = json.loads(calls["write_all"].call_args.args[1])
            self.assertEqual(payload["manifest_version"], "v3")
            self.assertFalse(payload["complete"])
            self.assertFalse(payload["cases"][0]["executed"])
            calls["fsync"].assert_called_once_with(7)
            calls["close"].assert_called_once_with(7)

    def test_main_existing_result_refuses_without_runtime(self) -> None:
        with main_scope("v3") as calls:
            calls["new_log"].side_effect = FileExistsError("retained_attempt")
            with self.assertRaises(FileExistsError):
                POLICY.main()
            for name in ("pinned_runtime", "PressureReader", "execute_case", "write_all", "fsync", "close"):
                calls[name].assert_not_called()

    def test_main_runtime_refusal_retains_reserved_result(self) -> None:
        with main_scope("v99") as calls:
            calls["pinned_runtime"].side_effect = POLICY.SafetyError("runtime_refusal")
            self.assertEqual(POLICY.main(), 1)
            payload = json.loads(calls["write_all"].call_args.args[1])
            self.assertEqual(payload["manifest_version"], "v99")
            self.assertEqual(payload["cases"], [])
            self.assertIn("runtime_refusal", payload["setup_failure"])
            self.assertFalse(payload["complete"])
            calls["execute_case"].assert_not_called()
            calls["fsync"].assert_called_once_with(7)
            calls["close"].assert_called_once_with(7)

    def test_main_fatal_retains_incomplete_reserved_result(self) -> None:
        with main_scope("v3") as calls:
            calls["pinned_runtime"].side_effect = RuntimeError("fatal_inert_fixture")
            with self.assertRaisesRegex(RuntimeError, "fatal_inert_fixture"):
                POLICY.main()
            calls["new_log"].assert_called_once()
            calls["write_all"].assert_not_called()
            calls["execute_case"].assert_not_called()
            calls["fsync"].assert_called_once_with(7)
            calls["close"].assert_called_once_with(7)

    def test_main_revision_only_reserved_for_v2(self) -> None:
        revision = manifest()
        revised = case("capture-revision")
        revised["previous"]["key"] = "d" * 64
        revision["cases"] = [revised]
        for version in ("v1", "v3", "v99"):
            with self.subTest(version=version), main_scope(version, revision) as calls:
                with self.assertRaisesRegex(POLICY.SafetyError, "unexpected_manifest_case_stage"):
                    POLICY.main()
                calls["new_log"].assert_not_called()
                calls["pinned_runtime"].assert_not_called()
        with main_scope("v2") as calls:
            with self.assertRaisesRegex(POLICY.SafetyError, "unexpected_manifest_case_stage"):
                POLICY.main()
            calls["new_log"].assert_not_called()
        with main_scope("v2", revision) as calls:
            calls["pinned_runtime"].side_effect = POLICY.SafetyError("controlled_stage_stop")
            self.assertEqual(POLICY.main(), 1)
            self.assertEqual(calls["new_log"].call_args.args[0], POLICY.RUN_ROOT / "results-v2.json")

    def test_bootstrap_uses_actual_v2_policy(self) -> None:
        guard = BOOT.AuditGuard(FakeProtocol(), {"case": case()}, POLICY)
        self.assertIn(POLICY.SOURCE_ROOT / "tests/native/mac-native-acceptance-v2.py", guard.read_paths)
        self.assertNotIn(POLICY.SOURCE_ROOT / "tests/native/mac-native-acceptance-v1.py", guard.read_paths)
        tree = ast.parse(Path(BOOT.__file__).read_bytes())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "compile_module"]
        policy = [node for node in calls if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "mac_native_policy_v2"]
        self.assertEqual(len(policy), 1)
        self.assertIsInstance(policy[0].args[1], ast.BinOp)
        self.assertEqual(policy[0].args[1].right.value, "tests/native/mac-native-acceptance-v2.py")


if __name__ == "__main__":
    unittest.main()
