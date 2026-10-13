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
from typing import Callable, Iterator
from unittest import mock


def load_source(name: str, filename: str) -> types.ModuleType:
    path = Path(__file__).parent / "native" / filename
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True),
        module.__dict__,
    )
    return module


POLICY = load_source("native_acceptance_policy_test_v4", "mac-native-acceptance-v4.py")
BOOT = load_source("native_acceptance_bootstrap_test_v4", "mac-native-bootstrap-v4.py")


def case(kind: str = "capture-a1") -> dict:
    name = {"capture-revision": "capture-a2", "invalid-state": "negative-a3"}.get(
        kind, kind
    )
    database = (
        "capture-a2"
        if kind == "capture-revision"
        else "negative-a3"
        if kind == "invalid-state"
        else kind
    )
    row = {
        "id": kind,
        "source": str(
            POLICY.RUN_ROOT
            / "home/.claude/projects/native-acceptance-v1"
            / (name + ".jsonl")
        ),
        "source_sha256": "a" * 64,
        "source_bytes": 24,
        "source_identity": [1, 2, 24, 3, 4],
        "session": "synthetic-native-" + name,
        "database": str(POLICY.RUN_ROOT / (database + "-v1.db")),
        "previous": None,
    }
    if kind == "flock-a4":
        for field in ("source", "source_sha256", "source_bytes", "source_identity"):
            row[field] = None
    if kind == "capture-revision":
        row["previous"] = {"key": "b" * 64, "sha256": "c" * 64, "byte_count": 12}
    return row


def manifest() -> dict:
    return {
        "version": 1,
        "stage": "A",
        "synthetic": True,
        "root_execution_go": True,
        "run_root": str(POLICY.RUN_ROOT),
        "home": str(POLICY.RUN_ROOT / "home"),
        "source_root": str(POLICY.SOURCE_ROOT),
        "source_commit": "a" * 40,
        "sources": {name: "a" * 64 for name in POLICY.SOURCE_FILES},
        "supervisor_sha256": "b" * 64,
        "bootstrap_sha256": "c" * 64,
        "limits": dict(POLICY.LIMITS),
        "cases": [case()],
    }


def mutation(
    seq: int, event: str, path: Path, flags: int = 0, source: Path | None = None
) -> dict:
    return {
        "type": "mutation",
        "version": 1,
        "seq": seq,
        "event": event,
        "path": str(path),
        "flags": flags,
        "source": str(source) if source is not None else None,
    }


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


class DirectoryProtocol(FakeProtocol):
    def __init__(self, row: dict) -> None:
        super().__init__()
        self.parent = POLICY.MutationPolicy(row)

    def emit(self, value: object) -> None:
        self.sent.append(value)
        grant = self.parent.decide(value)
        POLICY.mutation_target(grant, value)
        self.answers.append(grant)


def actual_sync_module() -> types.ModuleType:
    path = Path(__file__).parent.parent / "truememory/ingest/completion.py"
    text = path.read_text()
    node = next(
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name == "_sync_directory"
    )
    source = ast.get_source_segment(text, node)
    if source is None:
        raise AssertionError("Actual directory helper missing")
    module = types.ModuleType("truememory.ingest.completion")
    module.__file__ = str(BOOT.SOURCE_ROOT / "truememory/ingest/completion.py")
    module.Path = Path
    module.os = types.SimpleNamespace(
        name="posix", O_RDONLY=os.O_RDONLY, O_DIRECTORY=os.O_DIRECTORY
    )
    exec(compile(source, module.__file__, "exec", dont_inherit=True), module.__dict__)
    return module


@contextmanager
def directory_scope(
    kind: str = "capture-a1", after_open: Callable[[], None] | None = None
) -> Iterator[dict[str, object]]:
    import fcntl

    row = case(kind)
    protocol = DirectoryProtocol(row)
    guard = BOOT.AuditGuard(protocol, {"case": row}, POLICY)
    module = actual_sync_module()
    adapter = BOOT.DirectorySyncAdapter(guard, module)
    guard.directory_sync = adapter
    target_info = types.SimpleNamespace(
        st_dev=1, st_ino=2, st_uid=42, st_mode=stat.S_IFDIR | 0o700
    )
    fd_info = types.SimpleNamespace(
        st_dev=1, st_ino=2, st_uid=42, st_mode=stat.S_IFDIR | 0o700
    )
    adapter.raw_fsync = mock.Mock()
    adapter.raw_close = mock.Mock()

    def raw_open(path: object, flags: int, mode: int = 0o777) -> int:
        guard("open", (os.fspath(path), None, flags | getattr(os, "O_CLOEXEC", 0)))
        if after_open is not None:
            after_open()
        return 87

    adapter.raw_open = mock.Mock(side_effect=raw_open)
    module.os.open, module.os.fsync, module.os.close = (
        adapter.open,
        adapter.fsync,
        adapter.close,
    )
    with ExitStack() as patches:
        patches.enter_context(mock.patch.object(Path, "resolve", lambda value: value))
        patches.enter_context(mock.patch.object(Path, "is_symlink", return_value=False))
        patches.enter_context(mock.patch.object(BOOT.os, "getuid", return_value=42))
        patches.enter_context(
            mock.patch.object(POLICY, "owned_path", return_value=target_info)
        )
        fstat = patches.enter_context(
            mock.patch.object(BOOT.os, "fstat", return_value=fd_info)
        )
        access = patches.enter_context(
            mock.patch.object(fcntl, "fcntl", return_value=os.O_RDONLY)
        )
        yield {
            "row": row,
            "protocol": protocol,
            "guard": guard,
            "module": module,
            "adapter": adapter,
            "target_info": target_info,
            "fd_info": fd_info,
            "fstat": fstat,
            "access": access,
        }


class FakePsutilError(Exception):
    pass


@contextmanager
def main_scope(
    version: str = "v3", fixture: dict | None = None
) -> Iterator[dict[str, mock.Mock]]:
    row = copy.deepcopy(fixture) if fixture is not None else manifest()
    parser = mock.Mock()
    parser.parse_args.return_value = types.SimpleNamespace(
        manifest=str(POLICY.RUN_ROOT / ("manifest-" + version + ".json")),
        execute_root_authorized=True,
    )
    fake = types.ModuleType("fake_psutil")
    fake.Error = FakePsutilError
    with ExitStack() as patches:
        patches.enter_context(
            mock.patch.object(POLICY.argparse, "ArgumentParser", return_value=parser)
        )
        patches.enter_context(mock.patch.object(POLICY.sys, "platform", "darwin"))
        patches.enter_context(
            mock.patch.object(
                POLICY.sys,
                "flags",
                types.SimpleNamespace(
                    isolated=True, no_site=True, dont_write_bytecode=True
                ),
            )
        )
        patches.enter_context(
            mock.patch.object(POLICY.sys, "dont_write_bytecode", True)
        )
        patches.enter_context(mock.patch.object(POLICY.os, "umask", return_value=0o077))
        calls = {
            "owned_path": patches.enter_context(
                mock.patch.object(POLICY, "owned_path")
            ),
            "read_bounded": patches.enter_context(
                mock.patch.object(
                    POLICY, "read_bounded", return_value=POLICY.encoded(row)
                )
            ),
            "digest": patches.enter_context(
                mock.patch.object(POLICY, "digest", return_value="b" * 64)
            ),
            "new_log": patches.enter_context(
                mock.patch.object(POLICY, "new_log", return_value=7)
            ),
            "pinned_runtime": patches.enter_context(
                mock.patch.object(POLICY, "pinned_runtime", return_value=fake)
            ),
            "PressureReader": patches.enter_context(
                mock.patch.object(POLICY, "PressureReader")
            ),
            "execute_case": patches.enter_context(
                mock.patch.object(
                    POLICY,
                    "execute_case",
                    return_value={
                        "passed": False,
                        "reaped": True,
                        "executed": False,
                        "blocked": "pressure_or_headroom",
                    },
                )
            ),
            "write_all": patches.enter_context(mock.patch.object(POLICY, "write_all")),
            "fsync": patches.enter_context(mock.patch.object(POLICY.os, "fsync")),
            "close": patches.enter_context(mock.patch.object(POLICY.os, "close")),
        }
        yield calls


class PurePolicyTests(unittest.TestCase):
    def test_directory_sync_live_bound_return_never_closes_or_replaces_prior_binding(
        self,
    ) -> None:
        for original_identity in ((1, 2), (1, 99)):
            with (
                self.subTest(original_identity=original_identity),
                directory_scope() as scope,
            ):
                adapter = scope["adapter"]
                account = POLICY.account_directory(scope["row"])
                previous = (account / "captures", original_identity, 1)
                adapter.bindings[87] = previous
                retained = {88: (1, 55, stat.S_IFDIR, 42)}
                adapter.failed_open_bindings.update(retained)
                with self.assertRaises(BOOT.BootstrapError):
                    scope["module"]._sync_directory(account)
                adapter.raw_fsync.assert_not_called()
                adapter.raw_close.assert_not_called()
                self.assertEqual({87: previous}, adapter.bindings)
                self.assertIs(adapter.bindings[87], previous)
                self.assertEqual(retained, adapter.failed_open_bindings)
                self.assertIsNone(adapter.pending)
                self.assertEqual(1, len(adapter.failures))
                self.assertEqual(87, adapter.failures[0]["fd"])
                self.assertFalse(adapter.failures[0]["closed"])
                self.assertIn("bound_conflict", adapter.failures[0]["reason"])

    def test_directory_sync_bound_conflict_retains_bounded_failure_even_without_observed_identity(
        self,
    ) -> None:
        with directory_scope() as scope:
            adapter = scope["adapter"]
            previous = (POLICY.account_directory(scope["row"]), (1, 2), 1)
            adapter.bindings[87] = previous
            for observed in (None, scope["fd_info"]):
                adapter.close_failed_open(
                    87, observed, BOOT.BootstrapError("Synthetic conflict")
                )
            adapter.raw_close.assert_not_called()
            self.assertEqual({87: previous}, adapter.bindings)
            self.assertFalse(adapter.failed_open_bindings)
            self.assertEqual(2, len(adapter.failures))
            self.assertTrue(
                all(
                    not row["closed"] and "bound_conflict" in row["reason"]
                    for row in adapter.failures
                )
            )
            adapter.failures[:] = [
                {
                    "fd": 87,
                    "reason": "Retained synthetic bound conflict",
                    "closed": False,
                }
                for _ in range(BOOT.LIMITS["events"])
            ]
            original = copy.deepcopy(adapter.failures)
            with self.assertRaises(BOOT.BootstrapError):
                adapter.close_failed_open(
                    87, None, BOOT.BootstrapError("Over-limit conflict")
                )
            self.assertEqual(original, adapter.failures)
            self.assertEqual({87: previous}, adapter.bindings)
            adapter.raw_close.assert_not_called()

    def test_directory_sync_exact_account_and_captures_parent_grants(self) -> None:
        for kind in ("capture-a1", "capture-a2", "capture-revision"):
            for child in (None, "captures"):
                with (
                    self.subTest(kind=kind, child=child),
                    directory_scope(kind) as scope,
                ):
                    guard = scope["guard"]
                    adapter = scope["adapter"]
                    account = POLICY.account_directory(scope["row"])
                    target = account if child is None else account / child
                    regular_intent = (Path(scope["row"]["source"]), os.O_RDONLY)
                    guard.open_intents[:] = [regular_intent]
                    scope["module"]._sync_directory(target)
                    adapter.raw_open.assert_called_once_with(
                        target, os.O_RDONLY | os.O_DIRECTORY, 0o777
                    )
                    adapter.raw_fsync.assert_called_once_with(87)
                    adapter.raw_close.assert_called_once_with(87)
                    self.assertEqual({}, adapter.bindings)
                    self.assertEqual({}, adapter.failed_open_bindings)
                    self.assertEqual([], adapter.failures)
                    self.assertEqual([regular_intent], guard.open_intents)
                    self.assertNotIn(target, guard.approved)
                    self.assertFalse(guard.readable(target))
                    self.assertEqual(1, scope["protocol"].parent.sequence)
                    request = scope["protocol"].sent[0]
                    self.assertEqual(str(target), request["path"])
                    self.assertEqual(
                        os.O_RDONLY | os.O_DIRECTORY,
                        request["flags"] & ~getattr(os, "O_CLOEXEC", 0),
                    )

    def test_directory_sync_refuses_other_cases_and_paths_before_open(self) -> None:
        for kind in ("negative-a3", "invalid-state", "parser-a5", "flock-a4"):
            with self.subTest(kind=kind), directory_scope(kind) as scope:
                with self.assertRaises(BOOT.BootstrapError):
                    scope["module"]._sync_directory(
                        POLICY.account_directory(scope["row"])
                    )
                scope["adapter"].raw_open.assert_not_called()
                scope["adapter"].raw_fsync.assert_not_called()
        with directory_scope() as scope:
            account = POLICY.account_directory(scope["row"])
            targets = (
                account / "receipts",
                account / "logs",
                account.parent,
                account.parent / ("b" * 64),
                POLICY.RUN_ROOT / "completion" / "foreign" / "captures",
                POLICY.RUN_ROOT / "logs",
                POLICY.RUN_ROOT.parent / "outside",
            )
            for target in targets:
                with (
                    self.subTest(target=target),
                    self.assertRaises(BOOT.BootstrapError),
                ):
                    scope["module"]._sync_directory(target)
            scope["adapter"].raw_open.assert_not_called()
            scope["adapter"].raw_fsync.assert_not_called()

    def test_directory_sync_strict_flags_owner_type_and_ungranted_reads(self) -> None:
        with directory_scope() as scope:
            target = POLICY.account_directory(scope["row"])
            for extra in (
                os.O_CREAT,
                os.O_WRONLY,
                os.O_RDWR,
                os.O_TRUNC,
                os.O_NOFOLLOW,
                os.O_NONBLOCK,
            ):
                with self.subTest(extra=extra), self.assertRaises(BOOT.BootstrapError):
                    scope["adapter"].target(target, os.O_DIRECTORY | extra)
            for flags in (True, os.O_RDONLY):
                with self.assertRaises(BOOT.BootstrapError):
                    scope["adapter"].target(target, flags)
            for field, value in (
                ("st_uid", 43),
                ("st_mode", stat.S_IFDIR | 0o755),
                ("st_mode", stat.S_IFREG | 0o600),
            ):
                previous = getattr(scope["target_info"], field)
                setattr(scope["target_info"], field, value)
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(BOOT.BootstrapError),
                ):
                    scope["module"]._sync_directory(target)
                setattr(scope["target_info"], field, previous)
            with self.assertRaises(BOOT.BootstrapError):
                scope["guard"]("open", (str(target), None, os.O_DIRECTORY))
            with self.assertRaises(BOOT.BootstrapError):
                scope["adapter"].open(target, os.O_DIRECTORY)
            scope["adapter"].raw_open.assert_not_called()

    def test_directory_sync_postopen_identity_failure_closes_matching_return(
        self,
    ) -> None:
        for mode in (
            "path_swap",
            "returned_identity",
            "returned_type",
            "returned_uid",
            "access",
        ):
            with self.subTest(mode=mode), directory_scope() as scope:
                if mode == "path_swap":
                    original = scope["adapter"].raw_open.side_effect

                    def swap_path(path: object, flags: int, mode: int = 0o777) -> int:
                        fd = original(path, flags, mode)
                        scope["target_info"].st_ino = 3
                        return fd

                    scope["adapter"].raw_open.side_effect = swap_path
                elif mode == "returned_identity":
                    scope["fd_info"].st_ino = 3
                elif mode == "returned_type":
                    scope["fd_info"].st_mode = stat.S_IFREG | 0o600
                elif mode == "returned_uid":
                    scope["fd_info"].st_uid = 43
                else:
                    scope["access"].return_value = os.O_RDWR
                with self.assertRaises(BOOT.BootstrapError):
                    scope["module"]._sync_directory(
                        POLICY.account_directory(scope["row"])
                    )
                scope["adapter"].raw_fsync.assert_not_called()
                scope["adapter"].raw_close.assert_called_once_with(87)
                self.assertFalse(scope["adapter"].bindings)
                self.assertIsNone(scope["adapter"].pending)
                self.assertTrue(scope["adapter"].failures[0]["closed"])

    def test_directory_sync_reused_fd_before_fsync_refuses_foreign_close(self) -> None:
        with directory_scope() as scope:
            original = scope["fd_info"]
            reused = types.SimpleNamespace(
                st_dev=1, st_ino=99, st_uid=42, st_mode=stat.S_IFDIR | 0o700
            )
            scope["fstat"].side_effect = [original, original, reused, reused]
            with self.assertRaises(BOOT.BootstrapError):
                scope["module"]._sync_directory(POLICY.account_directory(scope["row"]))
            scope["adapter"].raw_fsync.assert_not_called()
            scope["adapter"].raw_close.assert_not_called()
            self.assertIn(87, scope["adapter"].bindings)
            self.assertFalse(scope["adapter"].failures[0]["closed"])

    def test_directory_sync_failure_uses_original_finally_and_retains_close_failure(
        self,
    ) -> None:
        for mode in ("fsync", "close", "postopen_close"):
            with self.subTest(mode=mode), directory_scope() as scope:
                if mode == "fsync":
                    scope["adapter"].raw_fsync.side_effect = OSError(
                        "Synthetic fsync failure"
                    )
                else:
                    scope["adapter"].raw_close.side_effect = OSError(
                        "Synthetic close failure"
                    )
                    if mode == "postopen_close":
                        scope["fd_info"].st_ino = 3
                with self.assertRaises((OSError, BOOT.BootstrapError)):
                    scope["module"]._sync_directory(
                        POLICY.account_directory(scope["row"])
                    )
                scope["adapter"].raw_close.assert_called_once_with(87)
                if mode == "fsync":
                    self.assertFalse(scope["adapter"].bindings)
                elif mode == "close":
                    self.assertIn(87, scope["adapter"].bindings)
                else:
                    self.assertIn(87, scope["adapter"].failed_open_bindings)

    def test_directory_sync_windows_preservation_and_regular_fdopen_refusal(
        self,
    ) -> None:
        with directory_scope() as scope:
            scope["module"].os.name = "nt"
            scope["module"]._sync_directory(POLICY.account_directory(scope["row"]))
            scope["adapter"].raw_open.assert_not_called()
            self.assertEqual([], scope["protocol"].sent)
            scope["guard"].open_intents[:] = [
                (Path(scope["row"]["source"]), os.O_RDONLY)
            ]
            with self.assertRaises(BOOT.BootstrapError):
                scope["guard"].descriptor(87, "rb")
            self.assertFalse(scope["guard"].known_fds)

    def test_directory_sync_parent_exact_existing_directory_proof(self) -> None:
        row = case()
        account = POLICY.account_directory(row)
        for target in (account, account / "captures"):
            policy = POLICY.MutationPolicy(row)
            request = mutation(
                1,
                "open",
                target,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
            )
            grant = policy.decide(request)
            with mock.patch.object(POLICY, "owned_path") as owned:
                POLICY.mutation_target(grant, request)
                self.assertEqual(
                    [
                        mock.call(target.parent, directory=True),
                        mock.call(target, directory=True),
                    ],
                    owned.call_args_list,
                )
        for kind in ("invalid-state", "negative-a3", "parser-a5", "flock-a4"):
            policy = POLICY.MutationPolicy(case(kind))
            with self.assertRaises(POLICY.SafetyError):
                policy.decide(
                    mutation(1, "open", policy.account, os.O_RDONLY | os.O_DIRECTORY)
                )

    def test_fixed_limits_and_proof_scope(self) -> None:
        self.assertEqual(POLICY.LIMITS, BOOT.LIMITS)
        self.assertEqual(POLICY.FREE_MIN - POLICY.LIMITS["rss_abort_bytes"], 201326592)
        self.assertEqual(POLICY.PROOF_SCOPE, BOOT.PROOF_SCOPE)
        self.assertEqual(
            set(POLICY.SOURCE_FILES), {path for _, path in BOOT.SOURCE_MODULES}
        )
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
        samples = [
            POLICY.Sensor(1_000_000_000 + n * 1_000_000_000, 1, POLICY.FREE_MIN, 100)
            for n in range(3)
        ]
        POLICY.admission(samples, 3_500_000_000)
        changes = [
            ("pressure", 0),
            ("pressure", 2),
            ("pressure", 3),
            ("pressure", 4),
            ("pressure", True),
            ("free_bytes", POLICY.FREE_MIN - 1),
            ("free_bytes", True),
            ("swap_used_bytes", 101),
            ("monotonic_ns", 1_000_000_000),
        ]
        for field, value in changes:
            altered = [POLICY.asdict(row) for row in samples]
            altered[-1][field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(POLICY.SafetyError),
            ):
                POLICY.admission(
                    [POLICY.Sensor(**row) for row in altered], 3_500_000_000
                )
        for now in (2_999_999_999, 3_500_000_001):
            with self.subTest(now=now), self.assertRaises(POLICY.SafetyError):
                POLICY.admission(samples, now)
        with self.assertRaises(POLICY.SafetyError):
            POLICY.admission(samples[:2], 3_500_000_000)
        with self.assertRaises(POLICY.SafetyError):
            POLICY.admission(
                [POLICY.Sensor(1 + n, 1, POLICY.FREE_MIN, 0) for n in range(3)], 3
            )

    def test_manifest_and_case_fences(self) -> None:
        self.assertEqual(POLICY.validate_manifest(manifest()), manifest())
        for kind in POLICY.CASES:
            POLICY.validate_case(case(kind))
        changes = [
            ("version", True),
            ("stage", "B"),
            ("synthetic", False),
            ("root_execution_go", False),
            ("run_root", "/tmp"),
            ("source_root", "/Users/j/dev/TrueMemory"),
            ("sources", {}),
            ("cases", []),
            ("cases", [case(), case()]),
        ]
        for field, value in changes:
            altered = manifest()
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(POLICY.SafetyError):
                POLICY.validate_manifest(altered)
        for field, value in (
            ("id", []),
            ("session", "unknown"),
            ("database", "/tmp/personal.db"),
            ("source", "/Users/j/dev/TrueMemory/transcript.jsonl"),
            ("source_bytes", True),
            ("source_identity", [1, 2, 23, 3, 4]),
        ):
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
            self.assertEqual(
                POLICY.evidence_paths(POLICY.RUN_ROOT / ("manifest-" + name + ".json")),
                (POLICY.RUN_ROOT / ("results-" + name + ".json"), name),
            )
        for text in (
            "relative",
            str(POLICY.RUN_ROOT) + "/../outside",
            str(POLICY.RUN_ROOT) + "//escape",
            "/tmp/file",
        ):
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
        for changed in (
            POLICY.Identity(72887, 42, 12346, 72887),
            POLICY.Identity(72887, 43, 12345, 72887),
            POLICY.Identity(72887, 42, 12345, 1),
            POLICY.Identity(True, 42, 12345, 1),
        ):
            with self.subTest(changed=changed):
                self.assertFalse(POLICY.same_owned_identity(row, changed, 42))

    def test_ready_and_result_exact_protocol(self) -> None:
        ready = {
            "type": "ready",
            "version": 1,
            "pid": 100,
            "audit_installed": True,
            "limits": {
                "cpu": [2, 3],
                "core": [0, 0],
                "nofile": [64, 64],
                "fsize": [1048576, 1048576],
            },
        }
        POLICY.validate_ready(ready, 100)
        for field, value in (
            ("pid", 101),
            ("version", True),
            ("audit_installed", 1),
            ("limits", {}),
        ):
            altered = copy.deepcopy(ready)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(POLICY.SafetyError):
                POLICY.validate_ready(altered, 100)
        altered = copy.deepcopy(ready)
        altered["limits"]["core"] = [False, False]
        with self.assertRaises(POLICY.SafetyError):
            POLICY.validate_ready(altered, 100)
        done = {
            "type": "result",
            "version": 1,
            "case": "capture-a1",
            "passed": True,
            "proof_scope": POLICY.PROOF_SCOPE,
            "import_mode": "isolated_source_module",
            "sources": manifest()["sources"],
            "details": {},
        }
        POLICY.validate_result(done, case(), manifest()["sources"])
        for field, value in (
            ("passed", 1),
            ("version", True),
            ("proof_scope", "installed_host"),
            ("case", "capture-a2"),
            ("sources", {}),
            ("details", []),
        ):
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
        policy.decide(
            mutation(
                3, "open", pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            )
        )
        grant = policy.decide(
            mutation(4, "os.rename", policy.account / "state.json", source=pending)
        )
        self.assertEqual(grant["path"], str(policy.account / "state.json"))
        with self.assertRaises(POLICY.SafetyError):
            policy.decide(
                mutation(
                    5,
                    "open",
                    policy.account / ("state-" + "b" * 32 + ".pending"),
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                )
            )

    def test_mutation_negative_controls(self) -> None:
        for event, filename, flags, source in (
            ("open", "state.json", os.O_WRONLY | os.O_CREAT, None),
            ("open", "state.lock", os.O_RDWR | os.O_CREAT | os.O_TRUNC, None),
            ("open", "unknown", os.O_WRONLY | os.O_CREAT | os.O_EXCL, None),
            ("os.mkdir", "foreign", 0o700, None),
            ("os.mkdir", "captures", 0o755, None),
            ("os.rename", "state.json", 0, POLICY.RUN_ROOT / "arbitrary"),
            ("unlink", "state.json", 0, None),
        ):
            policy = POLICY.MutationPolicy(case())
            with (
                self.subTest(event=event, filename=filename),
                self.assertRaises(POLICY.SafetyError),
            ):
                policy.decide(
                    mutation(1, event, policy.account / filename, flags, source)
                )
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
        info = types.SimpleNamespace(
            st_dev=1, st_ino=2, st_uid=42, st_mode=stat.S_IFREG | 0o600
        )
        with (
            mock.patch.object(BOOT.os, "fstat", return_value=info),
            mock.patch.object(BOOT.os, "getuid", return_value=42),
            mock.patch.object(fcntl, "fcntl", return_value=os.O_RDONLY),
            mock.patch.object(Path, "lstat", return_value=info),
            mock.patch.object(Path, "is_symlink", return_value=False),
        ):
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
                with (
                    self.subTest(fd=fd, mode=mode),
                    self.assertRaises(BOOT.BootstrapError),
                ):
                    guard.descriptor(fd, mode)

    def test_audit_preopen_grant_and_denials(self) -> None:
        protocol = FakeProtocol()
        guard = BOOT.AuditGuard(protocol, {"case": case()}, POLICY)
        path = POLICY.account_directory(case()) / "state.lock"
        protocol.answers.append(
            {
                "type": "grant",
                "version": 1,
                "seq": 1,
                "path": str(path),
                "purpose": "Owned fixed lock",
            }
        )
        guard.path = lambda value: Path(value)
        guard("open", (str(path), None, os.O_RDWR | os.O_CREAT))
        self.assertEqual(len(protocol.sent), 1)
        self.assertEqual(guard.open_intents, [(path, os.O_RDWR)])
        guard("truememory.native.audit_probe", ("fixed-stage-a-v1",))
        self.assertTrue(guard.activated)
        for event, args in (
            ("socket.connect", ()),
            ("subprocess.Popen", ()),
            ("os.remove", (str(path), -1)),
            ("sqlite3.connect", ("memory",)),
            ("resource.setrlimit", (0, (1, 1))),
            ("import", ("truememory.engine", None)),
            ("import", ("torch", None)),
            ("os.mkdir", (str(path), 0o700, 1)),
            ("os.rename", (str(path), str(path), 1, -1)),
        ):
            with self.subTest(event=event), self.assertRaises(BOOT.BootstrapError):
                guard(event, args)

    def test_prelaunch_refusals_retain_nonexecuted_result(self) -> None:
        fake = types.ModuleType("fake_psutil")
        fake.Error = FakePsutilError
        with (
            mock.patch.object(
                POLICY.selectors, "DefaultSelector", return_value=FakeSelector()
            ),
            mock.patch.object(
                POLICY, "sample", side_effect=POLICY.SafetyError("pressure_or_headroom")
            ),
            mock.patch.object(POLICY.subprocess, "Popen") as launch,
        ):
            result = POLICY.execute_case(manifest(), case(), fake, None)
            self.assertFalse(result["executed"])
            self.assertFalse(result["passed"])
            self.assertTrue(result["reaped"])
            self.assertIn("pressure_or_headroom", result["blocked"])
            launch.assert_not_called()
        samples = [
            POLICY.Sensor(1_000_000_000 + n * 1_000_000_000, 1, POLICY.FREE_MIN, 100)
            for n in range(3)
        ]
        with (
            mock.patch.object(
                POLICY.selectors, "DefaultSelector", return_value=FakeSelector()
            ),
            mock.patch.object(POLICY, "sample", side_effect=samples),
            mock.patch.object(POLICY.time, "sleep"),
            mock.patch.object(POLICY.time, "monotonic_ns", return_value=3_000_000_001),
            mock.patch.object(
                POLICY, "owned_path", side_effect=POLICY.SafetyError("fixture_changed")
            ),
            mock.patch.object(POLICY.subprocess, "Popen") as launch,
        ):
            result = POLICY.execute_case(manifest(), case(), fake, None)
            self.assertFalse(result["executed"])
            self.assertFalse(result["passed"])
            self.assertIn("fixture_changed", result["blocked"])
            self.assertEqual(len(result["sensors"]), 3)
            launch.assert_not_called()

    def test_pipe_deadline_and_partial_writes_are_pure(self) -> None:
        with (
            mock.patch.object(POLICY.time, "monotonic", return_value=1),
            mock.patch.object(POLICY.os, "write", side_effect=[2, 3]) as writer,
        ):
            POLICY.write_pipe(7, b"12345", 1.1)
            self.assertEqual(writer.call_count, 2)
        with (
            mock.patch.object(POLICY.time, "monotonic", return_value=1),
            mock.patch.object(POLICY.os, "write", side_effect=[BlockingIOError(), 3]),
            mock.patch.object(
                POLICY.select, "select", return_value=([], [7], [])
            ) as poll,
        ):
            POLICY.write_pipe(7, b"123", 1.1)
            self.assertEqual(poll.call_count, 1)
        with (
            mock.patch.object(POLICY.time, "monotonic", return_value=1),
            mock.patch.object(POLICY.os, "write") as writer,
        ):
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
        result = {
            "reaped": False,
            "descendants": [],
            "descendant_containment": "not_observed",
        }
        with mock.patch.object(POLICY.time, "monotonic", return_value=1):
            POLICY.terminate_owned(child, None, [], fake, result)
        child.send_signal.assert_called_once_with(POLICY.signal.SIGTERM)
        self.assertTrue(result["reaped"])
        self.assertEqual(result["termination_authority"], "unreaped_direct_popen_child")
        identity = POLICY.Identity(100, 42, 123, 100)
        changed = POLICY.Identity(100, 42, 124, 100)
        child = mock.Mock()
        child.poll.return_value = None
        result = {
            "reaped": False,
            "descendants": [],
            "descendant_containment": "not_observed",
        }
        with (
            mock.patch.object(POLICY.time, "monotonic", return_value=1),
            mock.patch.object(POLICY, "observe_identity", return_value=changed),
            mock.patch.object(POLICY.os, "getuid", return_value=42),
            mock.patch.object(POLICY.os, "killpg") as signal_group,
        ):
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
        with mock.patch.object(
            BOOT.os, "write", side_effect=lambda fd, data: len(data)
        ) as writer:
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
        with (
            mock.patch.object(BOOT.select, "select", return_value=([0], [], [])),
            mock.patch.object(BOOT.os, "read", return_value=b"x" * 2049),
        ):
            with self.assertRaises(BOOT.BootstrapError):
                protocol.receive(2048, 0.1)

    def test_attempt_versions_are_canonical_before_io(self) -> None:
        for number in range(1, 100):
            self.assertEqual(
                POLICY.validate_attempt_version("v" + str(number)), "v" + str(number)
            )
        fake = types.ModuleType("fake_psutil")
        fake.Error = FakePsutilError
        invalid = (
            True,
            False,
            None,
            3,
            b"v3",
            [],
            "v0",
            "v00",
            "v01",
            "v100",
            "V3",
            "v3\n",
            "v３",
            "v٣",
            "../v3",
            "v3.log",
        )
        for version in invalid:
            with (
                self.subTest(version=version),
                mock.patch.object(POLICY.selectors, "DefaultSelector") as selector,
                mock.patch.object(POLICY, "sample") as sensor,
                mock.patch.object(POLICY, "new_log") as writer,
                mock.patch.object(POLICY.subprocess, "Popen") as launch,
            ):
                with self.assertRaises(POLICY.SafetyError):
                    POLICY.validate_attempt_version(version)
                with self.assertRaises(POLICY.SafetyError):
                    POLICY.execute_case(
                        manifest(), case(), fake, None, attempt_version=version
                    )
                selector.assert_not_called()
                sensor.assert_not_called()
                writer.assert_not_called()
                launch.assert_not_called()

    def test_actual_retry_logs_and_legacy_default(self) -> None:
        fake = types.ModuleType("fake_psutil")
        fake.Error = FakePsutilError
        info = types.SimpleNamespace(
            st_dev=1, st_ino=2, st_size=24, st_mtime_ns=3, st_ctime_ns=4
        )
        samples = [
            POLICY.Sensor(1_000_000_000 + n * 1_000_000_000, 1, POLICY.FREE_MIN, 100)
            for n in range(3)
        ]
        fixture = case()
        fixture_bytes = b"x" * 24
        fixture["source_sha256"] = POLICY.digest(fixture_bytes)
        plan = manifest()
        source_bytes = {
            POLICY.SOURCE_ROOT / name: ("synthetic:" + name).encode("utf-8")
            for name in POLICY.SOURCE_FILES
        }
        plan["sources"] = {
            name: POLICY.digest(source_bytes[POLICY.SOURCE_ROOT / name])
            for name in POLICY.SOURCE_FILES
        }
        bootstrap_path = POLICY.SOURCE_ROOT / "tests/native/mac-native-bootstrap-v4.py"
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
                patches.enter_context(
                    mock.patch.object(
                        POLICY.selectors, "DefaultSelector", return_value=FakeSelector()
                    )
                )
                patches.enter_context(
                    mock.patch.object(POLICY, "sample", side_effect=samples)
                )
                patches.enter_context(mock.patch.object(POLICY.time, "sleep"))
                patches.enter_context(
                    mock.patch.object(
                        POLICY.time, "monotonic_ns", return_value=3_000_000_001
                    )
                )
                patches.enter_context(
                    mock.patch.object(POLICY, "owned_path", return_value=info)
                )
                reader = patches.enter_context(
                    mock.patch.object(POLICY, "read_bounded", side_effect=read_fixture)
                )
                writer = patches.enter_context(
                    mock.patch.object(
                        POLICY,
                        "new_log",
                        side_effect=[7, 8, POLICY.SafetyError("stop_before_launch")],
                    )
                )
                patches.enter_context(mock.patch.object(POLICY.os, "fsync"))
                patches.enter_context(mock.patch.object(POLICY.os, "close"))
                launch = patches.enter_context(
                    mock.patch.object(POLICY.subprocess, "Popen")
                )
                result = (
                    POLICY.execute_case(plan, fixture, fake, None)
                    if version is None
                    else POLICY.execute_case(
                        plan, fixture, fake, None, attempt_version=version
                    )
                )
                self.assertEqual(
                    [call.args[0] for call in writer.call_args_list],
                    [
                        POLICY.RUN_ROOT
                        / "logs"
                        / ("capture-a1-" + channel + "-" + expected + ".log")
                        for channel in ("stdout", "stderr", "events")
                    ],
                )
                self.assertEqual(
                    reader.call_args_list[-1].args[0],
                    POLICY.SOURCE_ROOT / "tests/native/mac-native-bootstrap-v4.py",
                )
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
            self.assertEqual(
                calls["new_log"].call_args.args[0], POLICY.RUN_ROOT / "results-v3.json"
            )
            calls["new_log"].assert_called_once()
            self.assertEqual(
                calls["execute_case"].call_args.kwargs, {"attempt_version": "v3"}
            )
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
            for name in (
                "pinned_runtime",
                "PressureReader",
                "execute_case",
                "write_all",
                "fsync",
                "close",
            ):
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
                with self.assertRaisesRegex(
                    POLICY.SafetyError, "unexpected_manifest_case_stage"
                ):
                    POLICY.main()
                calls["new_log"].assert_not_called()
                calls["pinned_runtime"].assert_not_called()
        with main_scope("v2") as calls:
            with self.assertRaisesRegex(
                POLICY.SafetyError, "unexpected_manifest_case_stage"
            ):
                POLICY.main()
            calls["new_log"].assert_not_called()
        with main_scope("v2", revision) as calls:
            calls["pinned_runtime"].side_effect = POLICY.SafetyError(
                "controlled_stage_stop"
            )
            self.assertEqual(POLICY.main(), 1)
            self.assertEqual(
                calls["new_log"].call_args.args[0], POLICY.RUN_ROOT / "results-v2.json"
            )

    def test_bootstrap_uses_actual_v2_policy(self) -> None:
        guard = BOOT.AuditGuard(FakeProtocol(), {"case": case()}, POLICY)
        self.assertIn(
            POLICY.SOURCE_ROOT / "tests/native/mac-native-acceptance-v4.py",
            guard.read_paths,
        )
        self.assertNotIn(
            POLICY.SOURCE_ROOT / "tests/native/mac-native-acceptance-v1.py",
            guard.read_paths,
        )
        tree = ast.parse(Path(BOOT.__file__).read_bytes())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "compile_module"
        ]
        policy = [
            node
            for node in calls
            if node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "mac_native_policy_v4"
        ]
        self.assertEqual(len(policy), 1)
        self.assertIsInstance(policy[0].args[1], ast.BinOp)
        self.assertEqual(
            policy[0].args[1].right.value, "tests/native/mac-native-acceptance-v4.py"
        )


if __name__ == "__main__":
    unittest.main()
