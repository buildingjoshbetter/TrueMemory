"""Synthetic recovery controls for the opt-in capped launcher; no manager or kernel effects."""
from __future__ import annotations

import ast
import copy
from contextlib import ExitStack
import errno
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import time
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import truememory_memory_bootstrap as BOOT  # noqa: E402
from truememory import model_client as client  # noqa: E402

FLAG = "TRUEMEMORY_REQUIRE_PROCESS_MEMORY_CAP"
GENERATION = "a" * 32
CAP = 4294967296


def identity(cap: int = CAP) -> object:
    return BOOT.LinuxInvocationIdentity(
        1, GENERATION, "12345678-1234-1234-1234-123456789abc", 1000,
        "app-truememory" + GENERATION + ".slice", "b" * 32, 29, 31, 37,
        "/user.slice/app.slice/app-truememory" + GENERATION + ".slice",
        "truememory-memory" + GENERATION + ".service", "c" * 32, 31, 41,
        "/user.slice/app.slice/app-truememory" + GENERATION + ".slice/truememory-memory" + GENERATION + ".service",
        43, 47, 1000.5, cap, 0, "d" * 64, "e" * 64,
    )


def evidence(value: object) -> object:
    return BOOT.LinuxScopeEvidence(value, 0x63677270, "cgroup2", True, True, True,
               value.uid, value.main_pid, value.start_ticks, value.generation,
               value.manifest_sha256, (value.main_pid,), value.cap_bytes, 0, "max", "max", True)


def owned_handle(*, cap: int = CAP) -> object:
    value = identity(cap)
    runner = SimpleNamespace(stop_complete=False, store=SimpleNamespace(directory=75), manifest={"deadline": 1000.0})
    scope = SimpleNamespace(mount_point="/synthetic", pidfd_device=101, pidfd_inode=103)
    record = BOOT.LinuxProvisionalLaunch(value, runner, 72, scope, object())
    adapter = SimpleNamespace(_close_resolved=MagicMock())
    return BOOT.OwnedLinuxLaunch(record, BOOT.LinuxDaemonLaunchPolicy(1, cap, 0), adapter)


def private_directory() -> Path:
    # Resolve so every component is a real directory for no-follow traversal.
    path = Path(tempfile.mkdtemp()).resolve()
    os.chmod(path, 0o700)
    return path


def write_private(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, raw)
    finally:
        os.close(descriptor)


# Properties that exist only on org.freedesktop.systemd1.Service objects.
SERVICE_ONLY = frozenset(("MainPID", "ExecMainPID", "ExecMainCode", "ExecMainStatus", "Result"))


class FakeUserManager:
    """systemctl --user as observed on systemd 249.

    `show -p` prints only properties the unit object has, so a slice never
    prints MainPID, and an unknown unit prints a not-found stub with rc 0.
    Stopping a transient unit unloads it once inactive; stop or reset-failed
    on an unloaded unit fails with status 5. The real _properties parser runs
    on this output.
    """

    def __init__(self, units: dict[str, dict[str, str]], *, sticky: frozenset[str] = frozenset()) -> None:
        self.units = units
        self.sticky = sticky  # units whose stop job leaves them active
        self.commands: list[tuple[str, str]] = []

    def state(self, unit: str) -> dict[str, str]:
        if unit in self.units:
            state = dict(self.units[unit])
        else:
            state = {"LoadState": "not-found", "ActiveState": "inactive", "SubState": "dead", "ControlGroup": "",
                     "InvocationID": "", "MemoryMax": "infinity", "MemorySwapMax": "infinity"}
            if unit.endswith(".service"):
                state.update(MainPID="0", ExecMainPID="0", ExecMainCode="0", ExecMainStatus="0", Result="success")
        if unit.endswith(".slice"):
            state = {name: value for name, value in state.items() if name not in SERVICE_ONLY}
        return state

    def run(self, arguments: list[str], deadline: float, **kwargs: object) -> tuple[object, bytes]:
        if arguments[:2] != ["/usr/bin/systemctl", "--user"]:
            raise AssertionError(f"Unexpected manager command {arguments}")
        command, unit = arguments[2], arguments[3]
        if command == "show":
            names = arguments[5].split("=", 1)[1].split(",")
            state = self.state(unit)
            return SimpleNamespace(), "".join(f"{name}={state[name]}\n" for name in names if name in state).encode()
        self.commands.append((command, unit))
        if unit not in self.units:
            raise BOOT.EnforcementUnavailable("Owned manager command failed with status 5")
        if command not in ("stop", "reset-failed"):
            raise AssertionError(f"Unexpected manager command {arguments}")
        if unit not in self.sticky:
            del self.units[unit]
        return SimpleNamespace(), b""


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class StaleEndpointRecovery(unittest.TestCase):
    """F1: a dead claimed daemon leaves a stale endpoint that must not wedge capped mode."""

    def claim(self, child: dict | None = None) -> dict:
        return {"generation": GENERATION, "launcher": {"pid": 101, "born": 2000.5},
                "child": {"pid": 43, "born": 1000.5} if child is None else child}

    def verify(self, claim: object, birth: object, *, ready: bool = True, live: bool = False,
               inspect: MagicMock | None = None) -> tuple[bool, MagicMock, MagicMock]:
        inspect = inspect if inspect is not None else MagicMock(return_value="synthetic-evidence")
        adopt = MagicMock(return_value=True)
        with patch.object(client, "read_start_claim", return_value=claim), \
             patch.object(client, "process_birth", return_value=birth), \
             patch.object(BOOT, "generation_ready", return_value=ready), \
             patch.object(BOOT, "claimed_daemon_live", return_value=live), \
             patch.object(BOOT, "inspect_existing_owned_linux", inspect), \
             patch.object(BOOT, "adopt_exit_cleanup", adopt):
            result = client._capped_daemon_verified(object(), time.monotonic() + 10.0)
        return result, inspect, adopt

    def test_dead_claimed_daemon_is_absent_without_inspection(self) -> None:
        result, inspect, adopt = self.verify(self.claim(), None)
        self.assertIs(result, False)
        inspect.assert_not_called()
        adopt.assert_not_called()

    def test_reused_pid_is_absent(self) -> None:
        result, inspect, _ = self.verify(self.claim(), 2000.0)
        self.assertIs(result, False)
        inspect.assert_not_called()

    def test_missing_or_pending_claim_is_absent(self) -> None:
        for claim in (None, {"generation": GENERATION, "launcher": {"pid": 101, "born": 2000.5}, "child": None}):
            with self.subTest(claim=claim):
                result, inspect, _ = self.verify(claim, 1000.5)
                self.assertIs(result, False)
                inspect.assert_not_called()

    def test_live_unverifiable_daemon_is_refused(self) -> None:
        inspect = MagicMock(side_effect=BOOT.EnforcementUnavailable("synthetic foreign member"))
        with self.assertRaises(BOOT.EnforcementUnavailable):
            self.verify(self.claim(), 1000.5, inspect=inspect)

    def test_live_daemon_still_starting_is_pending_not_an_error(self) -> None:
        result, inspect, _ = self.verify(self.claim(), 1000.5, ready=False)
        self.assertIs(result, False)
        inspect.assert_not_called()

    def test_clock_step_on_a_live_daemon_is_refused_fast(self) -> None:
        with self.assertRaises(BOOT.EnforcementUnavailable):
            self.verify(self.claim(), 2000.0, live=True)

    def test_request_waits_for_a_starting_daemon_instead_of_raising(self) -> None:
        starter = MagicMock(return_value=False)
        with patch.dict(os.environ, {FLAG: "1"}), \
             patch.object(client, "_required_capped_policy", return_value=object()), \
             patch.object(client, "_server_ready", return_value=True), \
             patch.object(client, "read_start_claim", return_value=self.claim()), \
             patch.object(client, "process_birth", return_value=1000.5), \
             patch.object(BOOT, "generation_ready", return_value=False), \
             patch.object(BOOT, "inspect_existing_owned_linux", side_effect=FileNotFoundError("ready-v1.json")), \
             patch.object(client, "_start_server", starter), self.assertRaises(ConnectionError):
            client._request_with_autostart({"op": "ping"}, timeout=5.0)
        starter.assert_called_once()

    def test_live_verified_daemon_is_joined_and_adopted(self) -> None:
        result, inspect, adopt = self.verify(self.claim(), 1000.5)
        self.assertIs(result, True)
        self.assertEqual(inspect.call_args.kwargs["pid"], 43)
        adopt.assert_called_once_with("synthetic-evidence")

    def test_stale_endpoint_probe_never_contacts_it(self) -> None:
        sender = MagicMock(return_value={"ok": True})
        with patch.dict(os.environ, {FLAG: "1"}), \
             patch.object(client, "_required_capped_policy", return_value=object()), \
             patch.object(client, "_server_ready", return_value=True), \
             patch.object(client, "_capped_daemon_verified", return_value=False), \
             patch.object(client, "_send_request", sender):
            self.assertIs(client._probe_server(time.monotonic() + 10.0), False)
        sender.assert_not_called()

    def test_request_over_stale_endpoint_relaunches_instead_of_raising(self) -> None:
        sender = MagicMock(return_value={"ok": True})
        starter = MagicMock(return_value=False)
        with patch.dict(os.environ, {FLAG: "1"}), \
             patch.object(client, "_required_capped_policy", return_value=object()), \
             patch.object(client, "_server_ready", return_value=True), \
             patch.object(client, "read_start_claim", return_value=self.claim()), \
             patch.object(client, "process_birth", return_value=None), \
             patch.object(client, "_send_request", sender), \
             patch.object(client, "_start_server", starter), \
             self.assertRaises(ConnectionError):
            client._request_with_autostart({"op": "ping"}, timeout=5.0)
        starter.assert_called_once()
        sender.assert_not_called()

    def test_start_server_launches_over_stale_endpoint(self) -> None:
        directory = private_directory()
        state = {"claim": self.claim()}
        births = {43: None, os.getpid(): 3000.5, 77: 1234.5}
        handle = SimpleNamespace(pid=77, record=SimpleNamespace(identity=SimpleNamespace(birth_time=1234.5, cap_bytes=CAP)),
                                 release_after_claim=MagicMock())
        spawn = MagicMock(return_value=handle)
        retain = MagicMock()
        locks: list[int] = []

        def lock(path: object) -> int:
            descriptor = os.open(os.devnull, os.O_RDONLY)
            locks.append(descriptor)
            return descriptor

        environment = {key: value for key, value in os.environ.items() if key != "TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB"}
        environment[FLAG] = "1"
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environment, clear=True))
            for target, name, value in (
                    (client.sys, "platform", "linux"), (client, "use_model_server", lambda: True),
                    (client, "_TRUEMEMORY_DIR", directory), (client, "_server_ready", lambda: True),
                    (client, "read_start_claim", lambda path: copy.deepcopy(state["claim"])),
                    (client, "_write_start_claim", lambda claim: state.__setitem__("claim", copy.deepcopy(claim))),
                    (client, "process_birth", lambda pid: births[pid]), (client, "try_file_lock", lock),
                    (client, "_spawn_server", spawn), (client, "_retain_owned_capped_launch", retain),
                    (client, "_send_request", lambda request, timeout=None: {"ok": True}),
                    (BOOT, "_physical_total", lambda: 4 * CAP), (BOOT, "_required_linux_support", lambda *args: None),
                    (BOOT, "inspect_existing_owned_linux", MagicMock(return_value="synthetic-evidence")),
                    (BOOT, "generation_ready", lambda generation: True),
                    (BOOT, "adopt_exit_cleanup", MagicMock(return_value=True))):
                stack.enter_context(patch.object(target, name, value))
            self.assertIs(client._start_server(wait_timeout=5.0), True)
        spawn.assert_called_once()
        handle.release_after_claim.assert_called_once()
        retain.assert_called_once_with(handle)
        self.assertEqual(state["claim"]["child"], {"pid": 77, "born": 1234.5})
        for descriptor in locks:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_held_bind_lock_blocks_relaunch_over_a_stale_endpoint(self) -> None:
        directory = private_directory()
        spawn = MagicMock()
        locks: list[int] = []

        def lock(path: object) -> int | None:
            if path == client.LOCK_PATH:
                return None  # A live bind owner whose claim is not current.
            descriptor = os.open(os.devnull, os.O_RDONLY)
            locks.append(descriptor)
            return descriptor

        environment = {key: value for key, value in os.environ.items() if key != "TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB"}
        environment[FLAG] = "1"
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environment, clear=True))
            for target, name, value in (
                    (client.sys, "platform", "linux"), (client, "use_model_server", lambda: True),
                    (client, "_TRUEMEMORY_DIR", directory), (client, "_server_ready", lambda: True),
                    (client, "read_start_claim", lambda path: self.claim()), (client, "process_birth", lambda pid: None),
                    (client, "try_file_lock", lock), (client, "_spawn_server", spawn),
                    (BOOT, "_physical_total", lambda: 4 * CAP), (BOOT, "_required_linux_support", lambda *args: None)):
                stack.enter_context(patch.object(target, name, value))
            self.assertIs(client._start_server(wait_timeout=0.3), False)
        spawn.assert_not_called()
        self.assertTrue(locks)
        for descriptor in locks:
            with self.assertRaises(OSError):
                os.fstat(descriptor)


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class FailedLaunchClaimRetirement(unittest.TestCase):
    """F3: a failed launch retires its claim; F2: it never leaves its workload unowned."""

    def run_start(self, spawn: object, *, abandon: object = None,
                  handed_off: bool = False) -> tuple[Path, MagicMock, object]:
        directory = private_directory()
        claim_path = directory / "model_server.start.json"
        retain = MagicMock()
        abandon = abandon if abandon is not None else MagicMock(return_value=True)
        births = {os.getpid(): 3000.5, 77: 1234.5}
        environment = {key: value for key, value in os.environ.items() if key != "TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB"}
        environment[FLAG] = "1"
        outcome: object = None
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environment, clear=True))
            for target, name, value in (
                    (client.sys, "platform", "linux"), (client, "use_model_server", lambda: True),
                    (client, "_TRUEMEMORY_DIR", directory), (client, "START_STATE_PATH", claim_path),
                    (client, "_server_ready", lambda: False), (client, "process_birth", lambda pid: births.get(pid)),
                    (client, "try_file_lock", lambda path: os.open(os.devnull, os.O_RDONLY)),
                    (client, "_spawn_server", spawn), (client, "_retain_owned_capped_launch", retain),
                    (BOOT, "_physical_total", lambda: 4 * CAP), (BOOT, "_required_linux_support", lambda *args: None),
                    (BOOT, "abandon_owned_launch", abandon),
                    (BOOT, "launch_handed_off", lambda handle: handed_off)):
                stack.enter_context(patch.object(target, name, value))
            try:
                outcome = client._start_server(wait_timeout=0.5)
            except (RuntimeError, OSError) as error:
                outcome = error
        return claim_path, retain, outcome

    def test_failed_spawn_retires_claim_so_the_next_client_launches(self) -> None:
        spawn = MagicMock(side_effect=BOOT.EnforcementUnavailable("synthetic manager refusal"))
        claim_path, _, outcome = self.run_start(spawn)
        self.assertIsInstance(outcome, BOOT.EnforcementUnavailable)
        self.assertFalse(claim_path.exists())
        _, _, outcome = self.run_start(spawn)
        self.assertIsInstance(outcome, BOOT.EnforcementUnavailable)
        self.assertEqual(spawn.call_count, 2)

    def test_os_failure_is_logged_and_still_retires_claim(self) -> None:
        spawn = MagicMock(side_effect=OSError("synthetic spawn failure"))
        claim_path, _, outcome = self.run_start(spawn)
        self.assertIs(outcome, False)
        self.assertFalse(claim_path.exists())
        self.assertEqual(spawn.call_count, 1)

    def handle(self, error: BaseException | None = None) -> object:
        error = error if error is not None else BOOT.EnforcementUnavailable("synthetic census")
        return SimpleNamespace(pid=77, record=SimpleNamespace(identity=SimpleNamespace(birth_time=1234.5, cap_bytes=CAP)),
                               release_after_claim=MagicMock(side_effect=error))

    def test_failed_release_terminates_owned_workload_within_the_caller_budget(self) -> None:
        handle = self.handle()
        abandon = MagicMock(return_value=True)
        started = time.monotonic()
        claim_path, retain, outcome = self.run_start(MagicMock(return_value=handle), abandon=abandon)
        self.assertIsInstance(outcome, BOOT.EnforcementUnavailable)
        self.assertEqual(abandon.call_args.args, (handle,))
        # The 0.5 s caller budget plus the cleanup grace bounds work under the start lock.
        self.assertLessEqual(abandon.call_args.kwargs["deadline"], started + 0.5 + BOOT._CLEANUP_GRACE + 0.5)
        retain.assert_not_called()
        self.assertFalse(claim_path.exists())

    def test_unresolved_termination_continues_off_the_start_lock(self) -> None:
        handle = self.handle()
        claim_path, retain, outcome = self.run_start(MagicMock(return_value=handle), abandon=MagicMock(return_value=False))
        self.assertIsInstance(outcome, BOOT.EnforcementUnavailable)
        retain.assert_called_once_with(handle, terminate=True)
        self.assertFalse(claim_path.exists())

    def test_budget_expiry_after_handoff_keeps_the_starting_workload(self) -> None:
        handle = self.handle(TimeoutError("synthetic hook budget"))
        abandon = MagicMock(return_value=True)
        claim_path, retain, outcome = self.run_start(MagicMock(return_value=handle), abandon=abandon, handed_off=True)
        self.assertIs(outcome, False)
        abandon.assert_not_called()
        retain.assert_called_once_with(handle, handed_off=True)
        claim = BOOT._strict_json(claim_path.read_bytes())
        self.assertEqual(claim["child"], {"pid": 77, "born": 1234.5})

    def test_integrity_failure_after_handoff_is_still_abandoned(self) -> None:
        for error in (BOOT.EnforcementUnavailable("foreign member at ready"), ValueError("foreign ready receipt")):
            handle = self.handle(error)
            abandon = MagicMock(return_value=True)
            with self.subTest(error=type(error).__name__):
                claim_path, retain, _ = self.run_start(MagicMock(return_value=handle), abandon=abandon, handed_off=True)
                abandon.assert_called_once()
                retain.assert_not_called()
                self.assertFalse(claim_path.exists())

    def test_budget_expiry_before_handoff_is_abandoned(self) -> None:
        handle = self.handle(TimeoutError("synthetic hook budget"))
        abandon = MagicMock(return_value=True)
        claim_path, retain, outcome = self.run_start(MagicMock(return_value=handle), abandon=abandon, handed_off=False)
        self.assertIs(outcome, False)
        abandon.assert_called_once()
        retain.assert_not_called()
        self.assertFalse(claim_path.exists())


class McpStartupCapFailure(unittest.TestCase):
    """F4: a required-cap failure is logged and the MCP server keeps starting."""

    def startup_block(self):
        source = ROOT / "truememory" / "mcp_server.py"
        tree = ast.parse(source.read_text())
        guarded = next(node for node in ast.walk(tree) if isinstance(node, ast.Try)
                       and any(isinstance(item, ast.Assign) and isinstance(item.value, ast.Call)
                               and getattr(item.value.func, "id", None) == "ensure_server_running"
                               for item in node.body))
        wrapper = ast.parse("def run(ensure_server_running, log, os):\n    pass\n")
        wrapper.body[0].body = [guarded, ast.parse("return shared_ready").body[0]]
        namespace: dict = {}
        exec(compile(ast.fix_missing_locations(wrapper), str(source), "exec"), namespace)
        return namespace["run"]

    def start(self, error: BaseException, environment: dict[str, str]) -> tuple[object, MagicMock]:
        log = MagicMock()
        synthetic_os = SimpleNamespace(environ=environment)
        result = self.startup_block()(MagicMock(side_effect=error), log, synthetic_os)
        return result, log

    def test_invalid_flag_value_is_explicit_and_not_fatal(self) -> None:
        error = ValueError("Invalid " + FLAG)
        result, log = self.start(error, {FLAG: "0"})
        self.assertIs(result, False)
        self.assertIn(FLAG, str(log.error.call_args))

    def test_unavailable_enforcement_is_logged_and_not_fatal(self) -> None:
        result, log = self.start(BOOT.EnforcementUnavailable("A required Linux charged-domain cap is unsupported"), {FLAG: "1"})
        self.assertIs(result, False)
        log.error.assert_called_once()

    def test_identity_schema_drift_is_not_fatal(self) -> None:
        result, log = self.start(TypeError("LinuxInvocationIdentity got an unexpected keyword"), {FLAG: "1"})
        self.assertIs(result, False)
        log.error.assert_called_once()

    def test_uncapped_runtime_errors_keep_their_previous_behavior(self) -> None:
        with self.assertRaises(RuntimeError):
            self.start(RuntimeError("synthetic uncapped failure"), {})

    def test_connection_failures_still_warn(self) -> None:
        result, log = self.start(ConnectionError("synthetic"), {FLAG: "1"})
        self.assertIs(result, False)
        log.warning.assert_called_once()
        log.error.assert_not_called()


class ReadinessWaitWithoutCensus(unittest.TestCase):
    """F2: waiting for readiness must not census the workload's own startup helpers."""

    def wait(self, reads: list[object], *, exited: list[bool] | None = None) -> tuple[list[str], object]:
        value = identity()
        trace: list[str] = []
        store = SimpleNamespace(directory=75, observe=lambda name: trace.append("observe:" + name),
                                append=lambda phase, payload: trace.append("append:" + phase))
        record = BOOT.LinuxProvisionalLaunch(value, SimpleNamespace(store=store), 72,
                                            SimpleNamespace(pidfd_device=101, pidfd_inode=103), object())
        adapter = BOOT.LinuxAdapter()

        def read(directory: int, name: str) -> tuple[bytes, str]:
            self.assertEqual((directory, name), (75, "ready-v1.json"))
            trace.append("read")
            item = reads.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item, "f" * 64

        def inspect(target: object, deadline: float) -> object:
            trace.append("inspect")
            return evidence(value)

        states = list(exited or [])
        with patch.object(BOOT, "_private_artifact", read), \
             patch.object(BOOT, "_owned_exited", lambda target: states.pop(0) if states else False), \
             patch.object(BOOT.time, "sleep", lambda delay: trace.append("sleep")), \
             patch.object(adapter, "inspect", inspect):
            adapter._await_ready(record, value, time.monotonic() + 10.0)
        return trace, value

    def ready(self, value: object, **changes: object) -> bytes:
        payload = {"version": 1, "generation": value.generation, "manifest_sha256": value.manifest_sha256,
                   "pid": value.main_pid, "kernel_cap_bytes": value.cap_bytes, "metric": "linux_cgroup_charged_domain",
                   "physical_hard_cap": False, "dynamic_registry_cap": False}
        payload.update(changes)
        return BOOT._canonical(payload)

    def test_census_runs_once_after_receipt_including_partial_reads(self) -> None:
        value = identity()
        reads = [FileNotFoundError(), FileNotFoundError(), b"", self.ready(value)[:20],
                 BOOT.EnforcementUnavailable("Retained artifact changed during readback"), self.ready(value)]
        trace, _ = self.wait(reads)
        self.assertEqual(trace.count("inspect"), 1)
        self.assertEqual(trace[-1], "inspect")
        self.assertLess(trace.index("append:application-ready"), trace.index("inspect"))
        self.assertEqual(trace.count("read"), 6)

    def test_exit_before_readiness_fails_without_census(self) -> None:
        with self.assertRaises(BOOT.EnforcementUnavailable):
            self.wait([FileNotFoundError()], exited=[False, True])

    def test_complete_foreign_receipt_is_refused(self) -> None:
        value = identity()
        with self.assertRaises((ValueError, RuntimeError)):
            self.wait([self.ready(value, pid=99)])

    def test_deadline_bounds_the_wait(self) -> None:
        value = identity()
        record = BOOT.LinuxProvisionalLaunch(value, SimpleNamespace(store=SimpleNamespace(directory=75)), 72,
                                            SimpleNamespace(pidfd_device=101, pidfd_inode=103), object())
        clock = [1000.0]
        with patch.object(BOOT.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(BOOT.time, "sleep", side_effect=lambda delay: clock.__setitem__(0, clock[0] + 0.5)), \
             patch.object(BOOT, "_owned_exited", return_value=False), \
             patch.object(BOOT, "_private_artifact", side_effect=FileNotFoundError()), \
             self.assertRaises(TimeoutError):
            BOOT.LinuxAdapter()._await_ready(record, value, 1001.0)

    def test_barrier_has_no_polling_census(self) -> None:
        tree = ast.parse((ROOT / "truememory_memory_bootstrap.py").read_text())
        adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LinuxAdapter")
        barrier = next(node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == "barrier")
        self.assertFalse(any(isinstance(node, ast.While) for node in ast.walk(barrier)))
        wait = next(node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == "_await_ready")
        loop = next(node for node in ast.walk(wait) if isinstance(node, ast.While))
        census = [node for node in ast.walk(loop) if isinstance(node, ast.Attribute) and node.attr == "inspect"]
        self.assertEqual(len(census), 1)


class OwnedAbandonAndRelease(unittest.TestCase):
    """F2/F5: failed or exited launches are terminated only via the pidfd and then retired."""

    def setUp(self) -> None:
        self.stack = ExitStack()
        self.marked = MagicMock(return_value=True)
        self.stack.enter_context(patch.object(BOOT, "_mark_generation_retired", self.marked))
        self.stack.enter_context(patch.object(BOOT.time, "sleep", lambda delay: None))

    def tearDown(self) -> None:
        self.stack.close()

    def test_transient_termination_refusal_is_retried(self) -> None:
        handle = owned_handle()
        terminate = MagicMock(side_effect=[BOOT.EnforcementUnavailable("helper"), BOOT.EnforcementUnavailable("helper"), None])
        with patch.object(BOOT, "terminate_owned_exact", terminate), patch.object(BOOT, "_owned_exited", return_value=False):
            self.assertIs(BOOT.abandon_owned_launch(handle), True)
        self.assertEqual(terminate.call_count, 3)
        self.marked.assert_called_once_with(GENERATION)

    def test_live_unterminated_workload_stays_owned(self) -> None:
        handle = owned_handle()
        release = MagicMock(return_value=True)
        with patch.object(BOOT, "terminate_owned_exact", side_effect=BOOT.EnforcementUnavailable("refused")), \
             patch.object(BOOT, "_owned_exited", return_value=False), patch.object(BOOT, "release_exited_launch", release):
            self.assertIs(BOOT.abandon_owned_launch(handle), False)
        release.assert_not_called()

    def test_exited_workload_is_released_without_signals(self) -> None:
        handle = owned_handle()
        terminate = MagicMock()
        with patch.object(BOOT, "terminate_owned_exact", terminate), patch.object(BOOT, "_owned_exited", return_value=True), \
             patch.object(BOOT, "release_exited_launch", return_value=True) as release:
            self.assertIs(BOOT.abandon_owned_launch(handle), True)
        terminate.assert_not_called()
        self.assertEqual(release.call_args.args, (handle,))

    def test_abandon_stops_at_its_deadline(self) -> None:
        handle = owned_handle()
        clock = [1000.0]
        deadlines: list[float] = []

        def refuse(target: object, *, deadline: float) -> None:
            deadlines.append(deadline)
            clock[0] += 0.6
            raise BOOT.EnforcementUnavailable("refused")

        with patch.object(BOOT.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(BOOT.time, "sleep", side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay)), \
             patch.object(BOOT, "terminate_owned_exact", refuse), patch.object(BOOT, "_owned_exited", return_value=False):
            self.assertIs(BOOT.abandon_owned_launch(handle, deadline=1001.0), False)
        self.assertTrue(deadlines)
        self.assertTrue(all(value <= 1001.0 for value in deadlines))
        self.assertLessEqual(clock[0], 1001.0 + 0.6)

    def test_release_stops_at_its_deadline(self) -> None:
        handle = owned_handle()
        clock = [1000.0]
        retire = MagicMock(return_value=True)

        def slow_stop(target: object, *, deadline: float) -> None:
            self.assertLessEqual(deadline, 1000.5)
            clock[0] += 1.0
            raise TimeoutError("synthetic manager stall")

        with patch.object(BOOT.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(BOOT, "stop_owned_exact", slow_stop), patch.object(BOOT, "_owned_exited", return_value=True), \
             patch.object(BOOT, "_retire_generation_units", retire):
            self.assertIs(BOOT.release_exited_launch(handle, deadline=1000.5), False)
        retire.assert_not_called()

    def test_release_prefers_terminal_evidence(self) -> None:
        handle = owned_handle()
        retire = MagicMock(return_value=True)
        with patch.object(BOOT, "stop_owned_exact", return_value="receipt"), patch.object(BOOT, "_retire_generation_units", retire):
            self.assertIs(BOOT.release_exited_launch(handle), True)
        retire.assert_not_called()
        self.marked.assert_called_once_with(GENERATION)

    def test_release_never_retires_a_live_workload(self) -> None:
        handle = owned_handle()
        retire = MagicMock(return_value=True)
        with patch.object(BOOT, "stop_owned_exact", side_effect=BOOT.EnforcementUnavailable("live")), \
             patch.object(BOOT, "_owned_exited", return_value=False), patch.object(BOOT, "_retire_generation_units", retire):
            self.assertIs(BOOT.release_exited_launch(handle), False)
        retire.assert_not_called()

    def test_collected_failed_service_is_retired_without_evidence(self) -> None:
        handle = owned_handle()
        retire = MagicMock(return_value=True)
        with patch.object(BOOT, "stop_owned_exact", side_effect=BOOT.EnforcementUnavailable("Manager InvocationID changed")), \
             patch.object(BOOT, "_owned_exited", return_value=True), patch.object(BOOT, "_retire_generation_units", retire):
            self.assertIs(BOOT.release_exited_launch(handle), True)
        self.assertEqual(retire.call_args.args, (GENERATION,))
        self.assertEqual(retire.call_args.kwargs["mount_point"], "/synthetic")
        self.assertEqual(retire.call_args.kwargs["parent_relative"], "/user.slice/app.slice")
        handle._adapter._close_resolved.assert_called_once()
        self.assertIs(handle.record.runner.stop_complete, True)

    def test_unreturned_native_launch_is_abandoned(self) -> None:
        adapter = BOOT.LinuxAdapter()
        value = identity(67108864)
        record = BOOT.LinuxProvisionalLaunch(value, object(), object(), object(), object())
        bad = evidence(value)
        object.__setattr__(bad, "filesystem_magic", 0x01021994)
        abandon = MagicMock(return_value=True)
        with patch.object(BOOT, "get_linux_adapter", return_value=adapter), \
             patch.object(adapter, "start", return_value=record), patch.object(adapter, "inspect", return_value=bad), \
             patch.object(adapter, "abandon", abandon), self.assertRaises(ValueError):
            BOOT.launch_owned_linux(BOOT.LinuxDaemonLaunchPolicy(1, 67108864, 0), generation=GENERATION,
                                    python_executable="/synthetic/python", deadline=time.monotonic() + 10.0)
        self.assertIs(abandon.call_args.args[0], record)


class PartialStartCleanup(unittest.TestCase):
    """F5: a start that fails before returning a record retires its own units."""

    def test_barrier_closes_then_units_retire(self) -> None:
        trace: list[str] = []
        store = SimpleNamespace(directory=os.open(os.devnull, os.O_RDONLY),
                                append=lambda phase, payload: trace.append(phase))
        descriptors = [os.open(os.devnull, os.O_RDONLY) for _ in range(2)]
        listener = SimpleNamespace(close=lambda: trace.append("listener"))
        connection = SimpleNamespace(close=lambda: trace.append("connection"))
        retire = MagicMock(side_effect=lambda *args, **kwargs: trace.append("retire") or True)
        with patch.object(BOOT, "_retire_generation_units", retire), \
             patch.object(BOOT, "_mark_generation_retired", MagicMock(return_value=True)) as marked:
            retired = BOOT.LinuxAdapter()._abandon_partial(store, listener, connection, None,
                      BOOT.LinuxDaemonLaunchPolicy(1, CAP, 0), GENERATION, "/synthetic", "/user.slice/app.slice", descriptors,
                      time.monotonic() + 10.0)
        self.assertIs(retired, True)
        self.assertEqual(trace, ["connection", "listener", "retire", "startup-retired"])
        marked.assert_called_once_with(GENERATION)
        for descriptor in [*descriptors, store.directory]:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_unproven_domain_is_deferred_to_the_sweep(self) -> None:
        trace: list[str] = []
        store = SimpleNamespace(directory=os.open(os.devnull, os.O_RDONLY),
                                append=lambda phase, payload: trace.append(phase))
        with patch.object(BOOT, "_retire_generation_units", return_value=False), \
             patch.object(BOOT, "_mark_generation_retired") as marked:
            retired = BOOT.LinuxAdapter()._abandon_partial(store, SimpleNamespace(close=lambda: None), None, None,
                      BOOT.LinuxDaemonLaunchPolicy(1, CAP, 0), GENERATION, "/synthetic", "/user.slice/app.slice", [],
                      time.monotonic() + 10.0)
        self.assertIs(retired, False)
        self.assertEqual(trace, ["startup-cleanup-deferred"])
        marked.assert_not_called()

    def test_expired_cleanup_budget_defers_without_manager_calls(self) -> None:
        store = SimpleNamespace(directory=os.open(os.devnull, os.O_RDONLY), append=MagicMock())
        retire = MagicMock(return_value=True)
        with patch.object(BOOT, "_retire_generation_units", retire):
            retired = BOOT.LinuxAdapter()._abandon_partial(store, SimpleNamespace(close=lambda: None), None, None,
                      BOOT.LinuxDaemonLaunchPolicy(1, CAP, 0), GENERATION, "/synthetic", "/user.slice/app.slice", [],
                      time.monotonic() - 1.0)
        self.assertIs(retired, False)
        retire.assert_not_called()

    def test_terminated_record_is_not_closed_twice(self) -> None:
        handle = owned_handle()
        store = SimpleNamespace(directory=75, append=MagicMock())
        closer = MagicMock()
        with patch.object(BOOT, "abandon_owned_launch", return_value=True), patch.object(BOOT.os, "close", closer):
            retired = BOOT.LinuxAdapter()._abandon_partial(store, SimpleNamespace(close=MagicMock()), None, handle.record,
                      handle.policy, GENERATION, "/synthetic", "/user.slice/app.slice", [72], time.monotonic() + 10.0)
        self.assertIs(retired, True)
        closer.assert_not_called()
        store.append.assert_not_called()


class UnitRetirement(unittest.TestCase):
    """F5/R1: only provably empty generation units are stopped, through the real readback parser."""

    SLICE = "/user.slice/app.slice/app-truememory" + GENERATION + ".slice"

    def units(self, *, service: dict[str, str] | None = None,
              slice_state: dict[str, str] | None = None) -> dict[str, dict[str, str]]:
        slice_unit, service_unit = BOOT._generation_units(GENERATION)
        result = {}
        if service is not None:
            result[service_unit] = {"LoadState": "loaded", "ActiveState": "active", "SubState": "exited",
                                    "MainPID": "0", "ControlGroup": self.SLICE + "/" + service_unit,
                                    "InvocationID": "c" * 32, **service}
        if slice_state is not None:
            result[slice_unit] = {"LoadState": "loaded", "ActiveState": "active", "SubState": "active",
                                  "ControlGroup": self.SLICE, "InvocationID": "b" * 32, "MemoryMax": str(CAP),
                                  "MemorySwapMax": "0", **slice_state}
        return result

    def manager(self, units: dict[str, dict[str, str]], *, empty: list[bool] | None = None) -> tuple[FakeUserManager, ExitStack]:
        manager = FakeUserManager(units)
        census = list(empty if empty is not None else [True] * 8)
        stack = ExitStack()
        stack.enter_context(patch.object(BOOT, "_run_command", manager.run))
        stack.enter_context(patch.object(BOOT, "_generation_domain_empty", lambda *args: census.pop(0)))
        stack.enter_context(patch.object(BOOT.time, "sleep", lambda delay: None))
        return manager, stack

    def retire(self, **kwargs: object) -> bool:
        return BOOT._retire_generation_units(GENERATION, mount_point="/synthetic", parent_relative="/user.slice/app.slice",
                                             deadline=time.monotonic() + 10.0, **kwargs)

    def test_fake_matches_observed_slice_readback(self) -> None:
        manager = FakeUserManager(self.units(slice_state={}))
        slice_unit, service_unit = BOOT._generation_units(GENERATION)
        with patch.object(BOOT, "_run_command", manager.run):
            with self.assertRaises(BOOT.EnforcementUnavailable):
                BOOT._properties(slice_unit, BOOT._SERVICE_STATE, time.monotonic() + 10.0)
            self.assertEqual(BOOT._properties(slice_unit, BOOT._SLICE_STATE, time.monotonic() + 10.0)["LoadState"], "loaded")
            missing = BOOT._properties(service_unit, BOOT._SERVICE_STATE, time.monotonic() + 10.0)
        self.assertEqual((missing["LoadState"], missing["MainPID"]), ("not-found", "0"))

    def test_unloaded_generation_needs_no_commands(self) -> None:
        manager, stack = self.manager({})
        with stack:
            self.assertIs(self.retire(), True)
        self.assertEqual(manager.commands, [])

    def test_failed_service_is_reset_and_empty_slice_stopped(self) -> None:
        slice_unit, service_unit = BOOT._generation_units(GENERATION)
        manager, stack = self.manager(self.units(service={"ActiveState": "failed", "SubState": "failed"}, slice_state={}))
        with stack:
            self.assertIs(self.retire(), True)
        self.assertEqual(manager.commands, [("reset-failed", service_unit), ("stop", slice_unit)])
        self.assertEqual(manager.units, {})

    def test_collected_service_leaves_only_the_slice_to_stop(self) -> None:
        slice_unit, _ = BOOT._generation_units(GENERATION)
        manager, stack = self.manager(self.units(slice_state={}))
        with stack:
            self.assertIs(self.retire(), True)
        self.assertEqual(manager.commands, [("stop", slice_unit)])

    def test_exited_service_then_slice_are_stopped(self) -> None:
        slice_unit, service_unit = BOOT._generation_units(GENERATION)
        manager, stack = self.manager(self.units(service={}, slice_state={}))
        with stack:
            self.assertIs(self.retire(), True)
        self.assertEqual(manager.commands, [("stop", service_unit), ("stop", slice_unit)])

    def test_live_main_process_is_never_stopped(self) -> None:
        manager, stack = self.manager(self.units(service={"MainPID": "43", "SubState": "running"}, slice_state={}))
        with stack:
            self.assertIs(self.retire(settle=1.0), False)
        self.assertEqual(manager.commands, [])

    def test_populated_or_foreign_domain_is_never_stopped(self) -> None:
        for units, empty in ((self.units(service={}, slice_state={}), [False] * 8),
                             (self.units(service={"ControlGroup": "/foreign.service"}, slice_state={}), None),
                             (self.units(slice_state={"ControlGroup": "/user.slice/foreign.slice"}), None)):
            manager, stack = self.manager(units, empty=empty)
            with self.subTest(units=units, empty=empty), stack:
                self.assertIs(self.retire(), False)
            self.assertEqual(manager.commands, [])

    def test_slice_is_kept_when_a_member_appears_after_service_stop(self) -> None:
        _, service_unit = BOOT._generation_units(GENERATION)
        manager, stack = self.manager(self.units(service={}, slice_state={}), empty=[True, False])
        with stack:
            self.assertIs(self.retire(), False)
        self.assertEqual(manager.commands, [("stop", service_unit)])

    def test_unit_collected_between_readback_and_command(self) -> None:
        _, service_unit = BOOT._generation_units(GENERATION)
        units = self.units(service={})
        manager = FakeUserManager(units)

        def run(arguments: list[str], deadline: float, **kwargs: object) -> tuple[object, bytes]:
            if arguments[2] == "stop":
                units.pop(service_unit, None)  # collected by the manager first
            return manager.run(arguments, deadline, **kwargs)

        with patch.object(BOOT, "_run_command", run):
            self.assertIs(BOOT._stop_unit(service_unit, time.monotonic() + 10.0), True)
        self.assertEqual(manager.commands, [("stop", service_unit)])

    def owned_stop(self, units: dict[str, dict[str, str]], *,
                   sticky: frozenset[str] = frozenset()) -> tuple[FakeUserManager, object, MagicMock]:
        slice_dir = private_directory()
        descriptor = os.open(slice_dir, os.O_RDONLY)
        try:
            metadata = os.fstat(descriptor)
            value = identity()
            value = BOOT.LinuxInvocationIdentity(**{**asdict(value), "slice_device": metadata.st_dev,
                                                    "slice_inode": metadata.st_ino, "slice_invocation_id": "b" * 32})
            store = SimpleNamespace(verify=MagicMock(), append=MagicMock(return_value=(1, "f" * 64)))
            runner = SimpleNamespace(stop_complete=False, store=store)
            record = BOOT.LinuxProvisionalLaunch(value, runner, 72, SimpleNamespace(slice_fd=descriptor), object())
            adapter = BOOT.LinuxAdapter()
            manager = FakeUserManager(units, sticky=sticky)
            closer = MagicMock()
            with patch.object(BOOT, "_run_command", manager.run), \
                 patch.object(adapter, "inspect", return_value=replace(evidence(value), members=())), \
                 patch.object(BOOT, "_recursive_members", return_value=((value.slice_relative_path, ()),)), \
                 patch.object(adapter, "_close_resolved", closer):
                adapter.stop(record, value, time.monotonic() + 10.0)
        finally:
            os.close(descriptor)
        return manager, record, closer

    def test_owned_stop_stops_service_and_slice_on_a_clean_exit(self) -> None:
        slice_unit, service_unit = BOOT._generation_units(GENERATION)
        manager, record, closer = self.owned_stop(self.units(service={}, slice_state={}))
        self.assertEqual(manager.commands, [("stop", service_unit), ("stop", slice_unit)])
        self.assertEqual(manager.units, {})
        closer.assert_called_once()
        self.assertIs(record.runner.stop_complete, True)
        phases = [call.args[0] for call in record.runner.store.append.call_args_list]
        self.assertEqual(phases[-1], "owned-stop-complete")

    def test_owned_stop_does_not_claim_completion_for_a_unit_left_active(self) -> None:
        slice_unit, _ = BOOT._generation_units(GENERATION)
        with self.assertRaises(BOOT.EnforcementUnavailable):
            self.owned_stop(self.units(service={}, slice_state={}), sticky=frozenset((slice_unit,)))

    def test_release_fallback_retires_and_marks_under_real_readback(self) -> None:
        slice_unit, _ = BOOT._generation_units(GENERATION)
        handle = owned_handle()
        manager, stack = self.manager(self.units(slice_state={}))
        with stack, patch.object(BOOT, "stop_owned_exact", side_effect=BOOT.EnforcementUnavailable("collected")), \
             patch.object(BOOT, "_owned_exited", return_value=True), \
             patch.object(BOOT, "_mark_generation_retired", return_value=True) as marked:
            self.assertIs(BOOT.release_exited_launch(handle, deadline=time.monotonic() + 10.0), True)
        self.assertEqual(manager.commands, [("stop", slice_unit)])
        marked.assert_called_once_with(GENERATION)
        handle._adapter._close_resolved.assert_called_once()
        self.assertIs(handle.record.runner.stop_complete, True)

    def test_launch_collects_failed_services(self) -> None:
        tree = ast.parse((ROOT / "truememory_memory_bootstrap.py").read_text())
        adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LinuxAdapter")
        start = next(node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == "start")
        launch = next(node for node in ast.walk(start) if isinstance(node, ast.List)
                      and any(isinstance(item, ast.Constant) and item.value == "/usr/bin/systemd-run" for item in node.elts))
        values = [item.value for item in launch.elts if isinstance(item, ast.Constant)]
        self.assertIn("--collect", values)
        self.assertIn("--property=RemainAfterExit=yes", values)


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class OrphanSweep(unittest.TestCase):
    """F5: launches sweep generations whose launcher is gone and prune old retired evidence."""

    def setUp(self) -> None:
        self.home = private_directory()
        self.root = self.home / ".truememory" / "memory-cap-launches"
        self.root.mkdir(parents=True, mode=0o700)
        os.chmod(self.home / ".truememory", 0o700)
        os.chmod(self.root, 0o700)
        self.clock = 1_700_000_000

    def generation(self, index: int, *, launcher: int | None = 7, retired: bool = False, extra: str | None = None) -> str:
        name = f"{index:032x}"
        path = self.root / name
        path.mkdir(mode=0o700)
        os.chmod(path, 0o700)
        if launcher is not None:
            write_private(path / "manifest-v1.json", BOOT._canonical({"boot_id": "boot", "parent_pid": launcher,
                                                                     "parent_start_ticks": 11}))
        if retired:
            write_private(path / "retired-v1.json", b"{}")
        if extra == "directory":
            (path / "nested").mkdir()
        elif extra == "symlink":
            (path / "link").symlink_to(self.home)
        self.clock += 10
        os.utime(path, (self.clock, self.clock))
        return name

    def sweep(self) -> MagicMock:
        retire = MagicMock(return_value=True)
        with patch.object(BOOT.Path, "home", return_value=self.home), \
             patch.object(BOOT, "_boot_id", return_value="boot"), \
             patch.object(BOOT, "_incarnation_alive", side_effect=lambda pid, ticks: pid == 1), \
             patch.object(BOOT, "_retire_generation_units", retire):
            BOOT._sweep_retired_generations(mount_point="/synthetic", parent_relative="/user.slice/app.slice",
                                            deadline=time.monotonic() + 10.0)
        return retire

    def test_live_launchers_are_left_to_their_owner(self) -> None:
        live = self.generation(1, launcher=1)
        orphan = self.generation(2)
        retire = self.sweep()
        self.assertEqual([item.args[0] for item in retire.call_args_list], [orphan])
        self.assertFalse((self.root / live / "retired-v1.json").exists())
        self.assertTrue((self.root / orphan / "retired-v1.json").exists())

    def test_old_retired_evidence_is_pruned_beyond_the_retention_window(self) -> None:
        old = [self.generation(index, retired=True) for index in range(1, 13)]
        retire = self.sweep()
        retire.assert_not_called()
        remaining = sorted(path.name for path in self.root.iterdir())
        self.assertEqual(remaining, sorted(old[-8:]))

    def test_unsafe_directories_are_never_removed(self) -> None:
        unsafe = [self.generation(1, retired=True, extra="directory"), self.generation(2, retired=True, extra="symlink")]
        for index in range(3, 13):
            self.generation(index, retired=True)
        self.sweep()
        for name in unsafe:
            self.assertTrue((self.root / name).is_dir())

    def test_unit_checks_are_bounded_per_launch(self) -> None:
        for index in range(1, 11):
            self.generation(index)
        retire = self.sweep()
        self.assertEqual(retire.call_count, BOOT._SWEEP_LIMIT)

    def test_live_launcher_detected_from_process_ticks(self) -> None:
        live = self.generation(1, launcher=4242)
        orphan = self.generation(2, launcher=4343)
        retire = MagicMock(return_value=True)

        def process_identity(pid: int) -> tuple[int, float]:
            if pid == 4242:
                return 11, 3000.5
            raise FileNotFoundError(pid)

        with patch.object(BOOT.Path, "home", return_value=self.home), patch.object(BOOT, "_boot_id", return_value="boot"), \
             patch.object(BOOT, "_process_identity", process_identity), patch.object(BOOT, "_retire_generation_units", retire):
            BOOT._sweep_retired_generations(mount_point="/synthetic", parent_relative="/user.slice/app.slice",
                                            deadline=time.monotonic() + 10.0)
        self.assertEqual([item.args[0] for item in retire.call_args_list], [orphan])
        self.assertFalse((self.root / live / "retired-v1.json").exists())

    def test_unrelated_names_are_ignored(self) -> None:
        (self.root / "not-a-generation").mkdir(mode=0o700)
        retire = self.sweep()
        retire.assert_not_called()
        self.assertTrue((self.root / "not-a-generation").is_dir())


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class JoinedExitAdoption(unittest.TestCase):
    """F5: a joined client retires units of a daemon whose launcher is already gone."""

    def setUp(self) -> None:
        BOOT._adopted_generations.clear()

    def tearDown(self) -> None:
        BOOT._adopted_generations.clear()

    def test_adoption_pins_the_verified_incarnation(self) -> None:
        value = identity()
        reader, writer = os.pipe()
        thread = MagicMock()
        with patch.object(BOOT.os, "pidfd_open", return_value=reader, create=True), \
             patch.object(BOOT, "_process_identity", return_value=(value.start_ticks, value.birth_time)), \
             patch.object(BOOT, "_membership", return_value=value.service_relative_path), \
             patch.object(BOOT.threading, "Thread", thread):
            self.assertIs(BOOT.adopt_exit_cleanup(evidence(value)), True)
            self.assertIs(BOOT.adopt_exit_cleanup(evidence(value)), False)
        self.assertEqual(thread.call_args.kwargs["args"], (reader, value))
        thread.return_value.start.assert_called_once()
        os.close(reader)
        os.close(writer)

    def test_changed_incarnation_is_not_adopted(self) -> None:
        value = identity()
        reader, writer = os.pipe()
        with patch.object(BOOT.os, "pidfd_open", return_value=reader, create=True), \
             patch.object(BOOT, "_process_identity", return_value=(value.start_ticks + 1, value.birth_time)), \
             patch.object(BOOT, "_membership", return_value=value.service_relative_path):
            self.assertIs(BOOT.adopt_exit_cleanup(evidence(value)), False)
        with self.assertRaises(OSError):
            os.fstat(reader)
        self.assertNotIn(GENERATION, BOOT._adopted_generations)
        os.close(writer)

    def exit_watch(self, state: str) -> MagicMock:
        value = identity()
        reader, writer = os.pipe()
        os.write(writer, b"x")
        retire = MagicMock(return_value=True)
        with patch.object(BOOT, "_open_directory", side_effect=lambda *args, **kwargs: os.open(os.devnull, os.O_RDONLY)), \
             patch.object(BOOT, "_generation_state", return_value=state), \
             patch.object(BOOT, "_mount", return_value=(29, "/synthetic")), \
             patch.object(BOOT, "_retire_generation_units", retire), \
             patch.object(BOOT, "_mark_generation_retired", return_value=True):
            BOOT._retire_after_exit(reader, value)
        with self.assertRaises(OSError):
            os.fstat(reader)
        os.close(writer)
        return retire

    def test_exit_with_launcher_gone_retires_units(self) -> None:
        retire = self.exit_watch("orphaned")
        self.assertEqual(retire.call_args.kwargs["parent_relative"], "/user.slice/app.slice")

    def test_exit_with_live_launcher_leaves_evidence_to_the_owner(self) -> None:
        self.exit_watch("live").assert_not_called()


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class ConsoleLauncherFailure(unittest.TestCase):
    """F2/F3/R2: the console retires a failed claim, owns what it cannot terminate, keeps a starting workload."""

    def run_main(self, abandon: list[bool], *, error: BaseException | None = None,
                 handed_off: bool = False) -> tuple[Path, MagicMock, MagicMock, BaseException | None]:
        self.enforced = MagicMock()
        home = private_directory()
        error = error if error is not None else BOOT.EnforcementUnavailable("synthetic census")
        handle = SimpleNamespace(pid=77, record=SimpleNamespace(pidfd=72, identity=SimpleNamespace(birth_time=1234.5)),
                                 release_after_claim=MagicMock(side_effect=error))
        abandon_mock = MagicMock(side_effect=list(abandon))
        release = MagicMock(return_value=True)
        environment = {key: value for key, value in os.environ.items() if key != "TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB"}
        raised = None
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environment, clear=True))
            for target, name, value in (
                    (BOOT.sys, "argv", ["truememory-model-server-capped"]), (BOOT.sys, "platform", "linux"),
                    (BOOT, "_physical_total", lambda: 4 * CAP), (BOOT, "_required_linux_support", lambda *args: None),
                    (BOOT.Path, "home", lambda: home), (BOOT, "_process_identity", lambda pid: (11, 3000.5)),
                    (BOOT, "_verified_psutil_birth", lambda pid, candidate: candidate),
                    (BOOT, "launch_owned_linux", MagicMock(return_value=handle)),
                    (BOOT, "abandon_owned_launch", abandon_mock), (BOOT, "release_exited_launch", release),
                    (BOOT, "launch_handed_off", lambda target: handed_off),
                    (BOOT, "enforce_startup_deadline", self.enforced),
                    (BOOT.select, "select", lambda read, write, error, timeout: (read, [], []))):
                stack.enter_context(patch.object(target, name, value))
            try:
                BOOT.main()
            except (RuntimeError, OSError) as failure:
                raised = failure
        return home / ".truememory" / "model_server.start.json", abandon_mock, release, raised

    def test_terminated_failure_retires_claim(self) -> None:
        claim, abandon, release, raised = self.run_main([True])
        self.assertIsInstance(raised, BOOT.EnforcementUnavailable)
        self.assertFalse(claim.exists())
        self.assertIn("deadline", abandon.call_args.kwargs)
        release.assert_not_called()

    def test_unterminated_failure_is_retried_off_the_lock_then_watched(self) -> None:
        claim, abandon, release, raised = self.run_main([False, False])
        self.assertIsInstance(raised, BOOT.EnforcementUnavailable)
        self.assertFalse(claim.exists())
        self.assertEqual(abandon.call_count, 2)
        self.assertNotIn("deadline", abandon.call_args.kwargs)
        release.assert_called_once()

    def test_budget_expiry_after_handoff_keeps_owning_the_starting_workload(self) -> None:
        claim, abandon, release, raised = self.run_main([], error=TimeoutError("synthetic"), handed_off=True)
        self.assertIsNone(raised)
        abandon.assert_not_called()
        self.assertTrue(claim.exists())
        self.enforced.assert_called_once()
        release.assert_called_once()

    def test_failed_launch_is_not_held_to_a_startup_deadline(self) -> None:
        self.run_main([True])
        self.enforced.assert_not_called()


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class RetainedOwnerWatcher(unittest.TestCase):
    """R4: termination left unfinished under the start lock continues on the watcher thread."""

    def watch(self, *, terminate: bool, abandoned: bool) -> tuple[MagicMock, MagicMock, MagicMock]:
        handle = SimpleNamespace(record=SimpleNamespace(pidfd=72))
        abandon = MagicMock(return_value=abandoned)
        release = MagicMock(return_value=True)
        waits = MagicMock(return_value=([72], [], []))

        class Inline:
            def __init__(self, target: object, name: str, daemon: bool) -> None:
                self.target = target

            def start(self) -> None:
                self.target()

        import select as select_module
        import threading as threading_module
        with patch.object(threading_module, "Thread", Inline), patch.object(select_module, "select", waits), \
             patch.object(BOOT, "abandon_owned_launch", abandon), patch.object(BOOT, "release_exited_launch", release), \
             patch.object(client, "_owned_capped_launches", []):
            client._retain_owned_capped_launch(handle, terminate=terminate)
        return abandon, release, waits

    def test_failed_launch_is_terminated_by_the_watcher(self) -> None:
        abandon, release, waits = self.watch(terminate=True, abandoned=True)
        abandon.assert_called_once()
        self.assertNotIn("deadline", abandon.call_args.kwargs)
        release.assert_not_called()
        waits.assert_not_called()

    def test_unterminated_failed_launch_is_still_released_after_exit(self) -> None:
        abandon, release, waits = self.watch(terminate=True, abandoned=False)
        waits.assert_called()
        release.assert_called_once()

    def test_healthy_launch_is_only_watched(self) -> None:
        abandon, release, _ = self.watch(terminate=False, abandoned=True)
        abandon.assert_not_called()
        release.assert_called_once()


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class GenerationClassification(unittest.TestCase):
    """T2: the sweep acts only on generations whose launcher is provably gone."""

    def setUp(self) -> None:
        self.root = private_directory()
        self.descriptor = os.open(self.root, os.O_RDONLY)

    def tearDown(self) -> None:
        os.close(self.descriptor)

    def generation(self, manifest: dict | None, *, age: float = 0.0) -> str:
        name = "b" * 32
        path = self.root / name
        path.mkdir(mode=0o700)
        if manifest is not None:
            write_private(path / "manifest-v1.json", BOOT._canonical(manifest))
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
        return name

    def state(self, name: str, *, stat_result: object = (11, 3000.5)) -> str:
        def process_identity(pid: int) -> tuple[int, float]:
            if isinstance(stat_result, BaseException):
                raise stat_result
            return stat_result
        with patch.object(BOOT, "_boot_id", return_value="boot"), patch.object(BOOT, "_process_identity", process_identity):
            return BOOT._generation_state(self.descriptor, name)

    def manifest(self, **changes: object) -> dict:
        return {"boot_id": "boot", "parent_pid": 4242, "parent_start_ticks": 11, **changes}

    def test_live_launcher_incarnation_is_live(self) -> None:
        self.assertEqual(self.state(self.generation(self.manifest())), "live")

    def test_exited_launcher_is_orphaned(self) -> None:
        self.assertEqual(self.state(self.generation(self.manifest()), stat_result=FileNotFoundError()), "orphaned")

    def test_reused_launcher_pid_is_orphaned(self) -> None:
        self.assertEqual(self.state(self.generation(self.manifest()), stat_result=(12, 3000.5)), "orphaned")

    def test_unreadable_launcher_is_conservatively_live(self) -> None:
        self.assertEqual(self.state(self.generation(self.manifest()), stat_result=BOOT.EnforcementUnavailable("zombie")), "live")

    def test_previous_boot_is_orphaned_even_with_a_matching_process(self) -> None:
        self.assertEqual(self.state(self.generation(self.manifest(boot_id="other"))), "orphaned")

    def test_malformed_launcher_fields_are_live(self) -> None:
        for changes in ({"parent_pid": "4242"}, {"parent_start_ticks": 11.0}, {"parent_pid": 0}):
            with self.subTest(changes=changes):
                path = self.root / ("b" * 32)
                if path.exists():
                    for item in path.iterdir():
                        item.unlink()
                    path.rmdir()
                self.assertEqual(self.state(self.generation(self.manifest(**changes)), stat_result=FileNotFoundError()), "live")

    def test_missing_manifest_waits_out_the_creation_grace(self) -> None:
        self.assertEqual(self.state(self.generation(None)), "live")

    def test_missing_manifest_past_the_grace_is_orphaned(self) -> None:
        self.assertEqual(self.state(self.generation(None, age=7200.0)), "orphaned")

    def test_retired_marker_wins(self) -> None:
        name = self.generation(self.manifest())
        write_private(self.root / name / "retired-v1.json", b"{}")
        self.assertEqual(self.state(name), "retired")

    def test_foreign_owner_is_live(self) -> None:
        name = self.generation(self.manifest())
        with patch.object(BOOT.os, "getuid", return_value=os.getuid() + 1):
            self.assertEqual(self.state(name, stat_result=FileNotFoundError()), "live")


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class LauncherHandoff(unittest.TestCase):
    """R2/R3/R5: the workload outlives its launcher after handoff and liveness uses start ticks."""

    def test_child_tolerates_an_exited_launcher_only_when_allowed(self) -> None:
        manifest = {"parent_pid": 4242, "parent_start_ticks": 11, "parent_birth": 3000.5}
        for result, allowed, expected in ((FileNotFoundError(), True, True), (FileNotFoundError(), False, False),
                                          ((11, 3000.5), False, True), ((12, 3000.5), True, False)):
            def process_identity(pid: int) -> tuple[int, float]:
                if isinstance(result, BaseException):
                    raise result
                return result
            with self.subTest(result=result, allowed=allowed), patch.object(BOOT, "_process_identity", process_identity):
                self.assertIs(BOOT._launcher_unchanged(manifest, launcher_may_exit=allowed), expected)

    def test_unreadable_live_launcher_is_never_taken_as_exited(self) -> None:
        manifest = {"parent_pid": 4242, "parent_start_ticks": 11, "parent_birth": 3000.5}
        for allowed in (True, False):
            with self.subTest(allowed=allowed), \
                 patch.object(BOOT, "_process_identity", side_effect=BOOT.EnforcementUnavailable("Malformed native process stat framing")), \
                 patch.object(BOOT, "_read_path", return_value=b"4242 (python3) S 1 0 0"), \
                 self.assertRaises(BOOT.EnforcementUnavailable):
                BOOT._launcher_unchanged(manifest, launcher_may_exit=allowed)

    def native_handle(self, *, handoff: bool) -> object:
        value = identity()
        runner = BOOT._NativeRunner(object(), SimpleNamespace(), {}, handoff_complete=handoff)
        record = BOOT.LinuxProvisionalLaunch(value, runner, 72, SimpleNamespace(pidfd_device=101, pidfd_inode=103), object())
        return BOOT.OwnedLinuxLaunch(record, BOOT.LinuxDaemonLaunchPolicy(1, CAP, 0), object())

    def test_handoff_requires_a_live_workload_that_received_its_environment(self) -> None:
        for handoff, exited, expected in ((True, False, True), (True, True, False), (False, False, False)):
            with self.subTest(handoff=handoff, exited=exited), patch.object(BOOT, "_owned_exited", return_value=exited):
                self.assertIs(BOOT.launch_handed_off(self.native_handle(handoff=handoff)), expected)

    def evidence_home(self, *, ready: bool, identity_payload: dict | None) -> Path:
        home = private_directory()
        directory = home / ".truememory" / "memory-cap-launches" / GENERATION
        directory.mkdir(parents=True, mode=0o700)
        if ready:
            write_private(directory / "ready-v1.json", b"{}")
        if identity_payload is not None:
            write_private(directory / "identity-v1.json", BOOT._canonical(identity_payload))
        return home

    def test_generation_ready_reads_the_receipt(self) -> None:
        for ready in (True, False):
            home = self.evidence_home(ready=ready, identity_payload=None)
            with self.subTest(ready=ready), patch.object(BOOT.Path, "home", return_value=home):
                self.assertIs(BOOT.generation_ready(GENERATION), ready)
        with patch.object(BOOT.Path, "home", return_value=private_directory()):
            self.assertIs(BOOT.generation_ready(GENERATION), False)

    def test_claimed_daemon_liveness_uses_start_ticks(self) -> None:
        home = self.evidence_home(ready=True, identity_payload={"main_pid": 43, "start_ticks": 47, "boot_id": "boot"})
        for result, pid, boot, expected in (((47, 9999.0), 43, "boot", True), ((48, 1000.5), 43, "boot", False),
                                            (FileNotFoundError(), 43, "boot", False), ((47, 1000.5), 44, "boot", False),
                                            ((47, 9999.0), 43, "next-boot", False)):
            def process_identity(target: int) -> tuple[int, float]:
                if isinstance(result, BaseException):
                    raise result
                return result
            with self.subTest(result=result, pid=pid, boot=boot), patch.object(BOOT.Path, "home", return_value=home), \
                 patch.object(BOOT, "_boot_id", return_value=boot), patch.object(BOOT, "_process_identity", process_identity):
                self.assertIs(BOOT.claimed_daemon_live(GENERATION, pid), expected)


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class StartupDeadlineEnforcement(unittest.TestCase):
    """N1: a handed-off child that misses its own startup deadline no longer holds the claim."""

    def enforce(self, *, exited: bool = False, ready_at: float | None = None) -> tuple[MagicMock, float]:
        handle = owned_handle()  # manifest deadline 1000.0
        clock = [995.0]
        abandon = MagicMock(return_value=True)

        def wait(read: list[int], write: list[int], errors: list[int], timeout: float) -> tuple[list, list, list]:
            self.assertEqual(read, [72])
            clock[0] += timeout
            return [], [], []

        with patch.object(BOOT.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(BOOT.select, "select", wait), patch.object(BOOT, "_owned_exited", return_value=exited), \
             patch.object(BOOT, "_ready_published", side_effect=lambda record: ready_at is not None and clock[0] >= ready_at), \
             patch.object(BOOT, "abandon_owned_launch", abandon):
            BOOT.enforce_startup_deadline(handle)
        return abandon, clock[0]

    def test_stalled_child_is_terminated_at_its_deadline(self) -> None:
        abandon, now = self.enforce()
        abandon.assert_called_once()
        self.assertGreaterEqual(now, 1000.0 + BOOT._CLEANUP_GRACE)
        self.assertLess(now, 1000.0 + BOOT._CLEANUP_GRACE + 1.0)

    def test_ready_child_is_kept(self) -> None:
        abandon, now = self.enforce(ready_at=997.0)
        abandon.assert_not_called()
        self.assertLess(now, 1000.0)

    def test_exited_child_is_left_to_release(self) -> None:
        abandon, _ = self.enforce(exited=True)
        abandon.assert_not_called()

    def test_ready_receipt_is_read_from_the_owned_store(self) -> None:
        directory = private_directory()
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            record = SimpleNamespace(runner=SimpleNamespace(store=SimpleNamespace(directory=descriptor)))
            self.assertIs(BOOT._ready_published(record), False)
            write_private(directory / "ready-v1.json", b"{}")
            self.assertIs(BOOT._ready_published(record), True)
        finally:
            os.close(descriptor)

    def evidence(self, *, deadline_offset: float, ready: bool = False, boot: str = "boot") -> Path:
        home = private_directory()
        directory = home / ".truememory" / "memory-cap-launches" / GENERATION
        directory.mkdir(parents=True, mode=0o700)
        write_private(directory / "manifest-v1.json",
                      BOOT._canonical({"boot_id": boot, "deadline": time.monotonic() + deadline_offset}))
        if ready:
            write_private(directory / "ready-v1.json", b"{}")
        return home

    def test_expiry_is_read_from_the_generation_manifest(self) -> None:
        for offset, ready, boot, expected in ((-10.0, False, "boot", True), (-10.0, True, "boot", False),
                                              (30.0, False, "boot", False), (-10.0, False, "other", False),
                                              (-1.0, False, "boot", False)):
            home = self.evidence(deadline_offset=offset, ready=ready, boot=boot)
            with self.subTest(offset=offset, ready=ready, boot=boot), patch.object(BOOT.Path, "home", return_value=home), \
                 patch.object(BOOT, "_boot_id", return_value="boot"):
                self.assertIs(BOOT.generation_startup_expired(GENERATION), expected)
        with patch.object(BOOT.Path, "home", return_value=private_directory()):
            self.assertIs(BOOT.generation_startup_expired(GENERATION), False)

    def start(self, *, expired: bool) -> MagicMock:
        directory = private_directory()
        claim = {"generation": GENERATION, "launcher": {"pid": 101, "born": 2000.5}, "child": {"pid": 43, "born": 1000.5}}
        births = {43: 1000.5, 101: None, os.getpid(): 3000.5}
        spawn = MagicMock(side_effect=OSError("synthetic stop after the launch decision"))
        environment = {key: value for key, value in os.environ.items() if key != "TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB"}
        environment[FLAG] = "1"
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environment, clear=True))
            for target, name, value in (
                    (client.sys, "platform", "linux"), (client, "use_model_server", lambda: True),
                    (client, "_TRUEMEMORY_DIR", directory), (client, "START_STATE_PATH", directory / "claim.json"),
                    (client, "_server_ready", lambda: False), (client, "read_start_claim", lambda path: copy.deepcopy(claim)),
                    (client, "_write_start_claim", lambda value: None), (client, "process_birth", lambda pid: births[pid]),
                    (client, "try_file_lock", lambda path: os.open(os.devnull, os.O_RDONLY)),
                    (client, "_spawn_server", spawn),
                    (BOOT, "_physical_total", lambda: 4 * CAP), (BOOT, "_required_linux_support", lambda *args: None),
                    (BOOT, "generation_startup_expired", lambda generation: expired)):
                stack.enter_context(patch.object(target, name, value))
            client._start_server(wait_timeout=0.3)
        return spawn

    def test_expired_pending_claim_is_relaunched_by_any_client(self) -> None:
        self.start(expired=True).assert_called_once()

    def test_pending_claim_within_its_budget_is_waited_on(self) -> None:
        self.start(expired=False).assert_not_called()

    def test_kept_launch_is_held_to_its_deadline_by_the_watcher(self) -> None:
        handle = SimpleNamespace(record=SimpleNamespace(pidfd=72))
        order: list[str] = []

        class Inline:
            def __init__(self, target: object, name: str, daemon: bool) -> None:
                self.target = target

            def start(self) -> None:
                self.target()

        import select as select_module
        import threading as threading_module
        with patch.object(threading_module, "Thread", Inline), \
             patch.object(select_module, "select", lambda *args: order.append("wait") or ([72], [], [])), \
             patch.object(BOOT, "enforce_startup_deadline", lambda target: order.append("enforce")), \
             patch.object(BOOT, "release_exited_launch", lambda target: order.append("release") or True), \
             patch.object(client, "_owned_capped_launches", []):
            client._retain_owned_capped_launch(handle, handed_off=True)
        self.assertEqual(order, ["enforce", "wait", "release"])


BOOT_ID = "12345678-1234-1234-1234-123456789abc"
LAUNCHER = 4242


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class RestoreAfterHandoff(unittest.TestCase):
    """N2/N4: the re-executed workload restores its gate after its launcher exits, and only then."""

    def setUp(self) -> None:
        self.home = private_directory()
        state = self.home / ".truememory"
        self.directory = state / "memory-cap-launches" / GENERATION
        self.directory.mkdir(parents=True, mode=0o700)
        for path in (state, state / "memory-cap-launches", self.directory):
            os.chmod(path, 0o700)
        claim_path = state / "model_server.start.json"
        bootstrap_sha = BOOT._hash_path(Path(BOOT.__file__).absolute(), bound=1048576)
        manifest = {"version": 1, "generation": GENERATION, "uid": os.getuid(), "boot_id": BOOT_ID,
                    "slice_unit": "app-truememory" + GENERATION + ".slice",
                    "service_unit": "truememory-memory" + GENERATION + ".service", "cap_bytes": CAP, "swap_bytes": 0,
                    "bootstrap_sha256": bootstrap_sha, "deadline": time.monotonic() + 60.0,
                    "parent_pid": LAUNCHER, "parent_start_ticks": 11, "parent_birth": 3000.5,
                    "claim_path": str(claim_path), "ancestor_memory_max": [],
                    "forwarded_names": ["SYNTHETIC"], "interpreter": BOOT._executable_pin(os.path.abspath(sys.executable))}
        manifest_raw = BOOT._canonical(manifest)
        write_private(self.directory / "manifest-v1.json", manifest_raw)
        self.identity = BOOT.LinuxInvocationIdentity(
            1, GENERATION, BOOT_ID, os.getuid(), manifest["slice_unit"], "b" * 32, 29, 31, 37,
            "/user.slice/app.slice/" + manifest["slice_unit"], manifest["service_unit"], "c" * 32, 31, 41,
            "/user.slice/app.slice/" + manifest["slice_unit"] + "/" + manifest["service_unit"],
            os.getpid(), 99, 5000.5, CAP, 0, bootstrap_sha, hashlib.sha256(manifest_raw).hexdigest())
        identity_raw = BOOT._canonical(asdict(self.identity))
        write_private(self.directory / "identity-v1.json", identity_raw)
        left, self.peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.descriptor = os.dup(left.fileno())
        left.close()
        metadata = os.fstat(self.descriptor)
        self.barrier = {"descriptor": self.descriptor, "device": 7, "inode": metadata.st_ino}
        identity_sha = hashlib.sha256(identity_raw).hexdigest()
        write_private(self.directory / "pre-reexec-v1.json",
                      BOOT._canonical(BOOT._preexec_payload(self.identity, identity_sha, self.barrier)))
        claim_raw = BOOT._canonical({"generation": GENERATION, "launcher": {"pid": LAUNCHER, "born": 3000.5},
                                     "child": {"pid": os.getpid(), "born": 5000.5}})
        write_private(claim_path, claim_raw)
        write_private(self.directory / "receipt-000001.json", BOOT._canonical({
            "version": 1, "sequence": 1, "previous_sha256": "0" * 64, "phase": "claim-durable-before-release",
            "payload": {"claim_sha256": hashlib.sha256(claim_raw).hexdigest()}, "monotonic": 1.0}))
        self.environment = {FLAG: "1", "TRUEMEMORY_CAP_MANIFEST": str(self.directory / "manifest-v1.json"),
                            "TRUEMEMORY_CAP_IDENTITY_SHA256": identity_sha, "TRUEMEMORY_CAP_BARRIER_FD": str(self.descriptor)}

    def tearDown(self) -> None:
        context = BOOT._child_context
        BOOT._child_context = None
        if context is not None:
            context[2].close()
        try:
            os.close(self.descriptor)
        except OSError:
            pass
        self.peer.close()

    def restore(self, launcher: object, *, peer_uid: int | None = None, peer_open: bool = True) -> None:
        if not peer_open:
            self.peer.close()
        real_fstat, real_read = os.fstat, BOOT._read_path

        def fstat(descriptor: int) -> os.stat_result:
            result = real_fstat(descriptor)
            if descriptor != self.descriptor:
                return result
            # macOS reports st_dev -1 for sockets; Linux reports the sockfs device.
            return SimpleNamespace(st_dev=7, st_ino=result.st_ino, st_uid=result.st_uid, st_mode=result.st_mode)

        def process_identity(pid: int) -> tuple[int, float]:
            if pid == os.getpid():
                return 99, 5000.5
            if isinstance(launcher, BaseException):
                raise launcher
            return launcher

        def read_path(path: Path, *, bound: int = 65536) -> bytes:
            if str(path) == f"/proc/{LAUNCHER}/stat":
                return f"{LAUNCHER} (python3) Z 1 0 0".encode()
            return real_read(path, bound=bound)

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, self.environment))
            for target, name, value in (
                    (BOOT.sys, "platform", "linux"), (BOOT, "_boot_id", lambda: BOOT_ID),
                    (BOOT, "_process_identity", process_identity), (BOOT, "_read_path", read_path),
                    (BOOT.os, "fstat", fstat), (BOOT, "child_gate", lambda identity, deadline: None),
                    (BOOT, "_peer", lambda endpoint: (LAUNCHER, os.getuid() if peer_uid is None else peer_uid))):
                stack.enter_context(patch.object(target, name, value))
            BOOT._restore_child_context()

    def test_live_launcher_acknowledgement_is_delivered(self) -> None:
        self.restore((11, 3000.5))
        self.assertTrue((self.directory / "first-import-v1.json").exists())
        self.assertIn(b"gate_passed", self.peer.recv(4096))

    def test_exited_launcher_is_accepted_after_handoff(self) -> None:
        self.restore(FileNotFoundError(), peer_open=False)
        self.assertTrue((self.directory / "first-import-v1.json").exists())

    def test_zombie_launcher_counts_as_exited(self) -> None:
        self.restore(BOOT.EnforcementUnavailable("Workload process is absent or unreaped"), peer_open=False)
        self.assertTrue((self.directory / "first-import-v1.json").exists())

    def test_reused_launcher_pid_is_refused(self) -> None:
        with self.assertRaises(BOOT.EnforcementUnavailable):
            self.restore((12, 3000.5))
        self.assertIsNone(BOOT._child_context)

    def test_live_launcher_that_closed_the_barrier_is_refused(self) -> None:
        with self.assertRaises((BrokenPipeError, ConnectionResetError)):
            self.restore((11, 3000.5), peer_open=False)

    def test_foreign_barrier_peer_is_refused(self) -> None:
        with self.assertRaises(BOOT.EnforcementUnavailable):
            self.restore((11, 3000.5), peer_uid=os.getuid() + 1)

    def test_only_a_zombie_state_counts_as_unreaped(self) -> None:
        for line, expected in ((b"4242 (a) Z 1", True), (b"4242 (a (b)) Z 1", True), (b"4242 (a) S 1", False)):
            with self.subTest(line=line), patch.object(BOOT, "_read_path", return_value=line):
                self.assertIs(BOOT._process_unreaped(LAUNCHER), expected)
        with patch.object(BOOT, "_read_path", side_effect=FileNotFoundError()):
            self.assertIs(BOOT._process_unreaped(LAUNCHER), True)


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class HandoffBoundary(unittest.TestCase):
    """N4: handoff starts only once the environment is sent; the child budget outlives a hook budget."""

    def barrier(self, *, environment_error: BaseException | None = None,
                acknowledgement_error: BaseException | None = None) -> object:
        home = private_directory()
        claim_path = home / "model_server.start.json"
        value = BOOT.LinuxInvocationIdentity(**{**asdict(identity()), "uid": os.getuid()})
        write_private(claim_path, BOOT._canonical({"generation": GENERATION, "launcher": {"pid": LAUNCHER, "born": 3000.5},
                                                  "child": {"pid": value.main_pid, "born": value.birth_time}}))
        forwarded = {"SYNTHETIC": "1", FLAG: "1", "TRUEMEMORY_CAP_CHILD": "1", "TRUEMEMORY_MODEL_SERVER_GENERATION": GENERATION}
        manifest = {"claim_path": str(claim_path), "parent_pid": LAUNCHER, "parent_birth": 3000.5,
                    "parent_start_ticks": 11, "forwarded_names": sorted(forwarded)}
        store = SimpleNamespace(append=MagicMock(return_value=(1, "f" * 64)), observe=MagicMock())
        runner = BOOT._NativeRunner(object(), store, manifest, forwarded=MappingProxyType(forwarded))
        record = BOOT.LinuxProvisionalLaunch(value, runner, 72, object(), SimpleNamespace(connection=object()))
        adapter = BOOT.LinuxAdapter()
        replies = [{"generation": GENERATION, "manifest_sha256": value.manifest_sha256, "preexec_gate_passed": True},
                   {"generation": GENERATION, "manifest_sha256": value.manifest_sha256, "gate_passed": True}]

        def send(endpoint: object, packet: dict, deadline: float) -> None:
            if "forwarded_environment" in packet and environment_error is not None:
                raise environment_error

        def receive(endpoint: object, deadline: float) -> dict:
            if len(replies) == 1 and acknowledgement_error is not None:
                raise acknowledgement_error
            return replies.pop(0)

        with patch.object(adapter, "inspect", return_value=evidence(value)), \
             patch.object(BOOT, "_verified_psutil_birth", lambda pid, birth: birth), \
             patch.object(BOOT, "_process_identity", lambda pid: (11, 3000.5)), \
             patch.object(BOOT, "_packet_send", send), patch.object(BOOT, "_packet_receive", receive), \
             patch.object(adapter, "_await_ready", lambda *args: None):
            try:
                adapter.barrier(record, value, time.monotonic() + 10.0)
            except TimeoutError:
                pass
        return runner

    def test_environment_send_timeout_is_not_a_handoff(self) -> None:
        self.assertIs(self.barrier(environment_error=TimeoutError("synthetic")).handoff_complete, False)

    def test_sent_environment_is_a_handoff_even_without_acknowledgement(self) -> None:
        self.assertIs(self.barrier(acknowledgement_error=TimeoutError("synthetic")).handoff_complete, True)

    def test_completed_barrier_is_a_handoff(self) -> None:
        self.assertIs(self.barrier().handoff_complete, True)

    def test_child_startup_budget_outlives_a_short_caller_budget(self) -> None:
        home = private_directory()
        captured: dict = {}

        class Captured(Exception):
            pass

        def store(path: Path, manifest: dict) -> None:
            captured.update(manifest)
            raise Captured()

        caller_deadline = time.monotonic() + 5.0
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"PATH": "/usr/bin", FLAG: "1"}, clear=True))
            for target, name, value in (
                    (BOOT.sys, "platform", "linux"), (BOOT.Path, "home", lambda: home),
                    (BOOT, "_required_linux_support", lambda *args: (29, "/synthetic", "/user.slice/app.slice", ())),
                    (BOOT, "_sweep_retired_generations", lambda **kwargs: 0),
                    (BOOT, "_hash_path", lambda *args, **kwargs: "d" * 64),
                    (BOOT, "_executable_pin", lambda path: {"path": path}),
                    (BOOT, "_process_identity", lambda pid: (11, 3000.5)), (BOOT, "_boot_id", lambda: BOOT_ID),
                    (BOOT, "_EvidenceStore", store)):
                stack.enter_context(patch.object(target, name, value))
            stack.enter_context(patch.object(BOOT, "_pidfd_supported", lambda: True))
            stack.enter_context(patch.object(BOOT.socket, "SO_PEERCRED", 17, create=True))
            with self.assertRaises(Captured):
                BOOT.LinuxAdapter().start(BOOT.LinuxDaemonLaunchPolicy(1, CAP, 0), GENERATION,
                                          os.path.abspath(sys.executable), caller_deadline)
        self.assertGreaterEqual(captured["deadline"], caller_deadline + BOOT._CHILD_STARTUP_SECONDS - 5.0 - 1.0)
        self.assertGreater(captured["deadline"], caller_deadline + 50.0)


def short_private_directory() -> Path:
    # macOS limits sun_path to 104 bytes, so socket fixtures need a short root.
    path = Path(tempfile.mkdtemp(dir="/private/tmp" if sys.platform == "darwin" else None)).resolve()
    os.chmod(path, 0o700)
    return path


def parse_environment_file(raw: bytes) -> dict[str, str]:
    """systemd's EnvironmentFile= rules for the double-quoted form the launcher writes."""
    text, result, index = raw.decode("utf-8"), {}, 0
    while index < len(text):
        equals = text.index("=", index)
        name = text[index:equals]
        if text[equals + 1] != '"':
            raise AssertionError("value is not double quoted")
        position, value = equals + 2, []
        while True:
            character = text[position]
            if character == "\\":
                escaped = text[position + 1]
                if escaped not in '"\\`$':
                    raise AssertionError(f"unexpected escape {escaped!r}")
                value.append(escaped)
                position += 2
            elif character == '"':
                position += 1
                break
            else:
                value.append(character)
                position += 1
        if text[position] != "\n":
            raise AssertionError("assignment does not end the line")
        result[name] = "".join(value)
        index = position + 1
    return result


class PidfdFallback(unittest.TestCase):
    """D1: interpreters without os.pidfd_open/signal.pidfd_send_signal use the raw syscalls."""

    def linux(self, machine: str = "x86_64") -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(BOOT.sys, "platform", "linux"))
        stack.enter_context(patch.object(BOOT.os, "uname", lambda: SimpleNamespace(machine=machine)))
        return stack

    def without_native(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(BOOT.os, "pidfd_open", None, create=True))
        stack.enter_context(patch.object(BOOT.signal, "pidfd_send_signal", None, create=True))
        return stack

    def test_unified_syscall_numbers_for_supported_architectures(self) -> None:
        for machine in ("x86_64", "aarch64"):
            with self.subTest(machine=machine), self.linux(machine):
                self.assertEqual((BOOT._pidfd_syscall("pidfd_open"), BOOT._pidfd_syscall("pidfd_send_signal")), (434, 424))
        with self.linux("riscv64"):
            self.assertIsNone(BOOT._pidfd_syscall("pidfd_open"))
        with patch.object(BOOT.sys, "platform", "darwin"):
            self.assertIsNone(BOOT._pidfd_syscall("pidfd_open"))

    def test_native_functions_are_preferred(self) -> None:
        raw = MagicMock()
        native_open, native_signal = MagicMock(return_value=9), MagicMock()
        with self.linux(), patch.object(BOOT.os, "pidfd_open", native_open, create=True), \
             patch.object(BOOT.signal, "pidfd_send_signal", native_signal, create=True), patch.object(BOOT, "_raw_syscall", raw):
            self.assertEqual(BOOT._pidfd_open(43), 9)
            BOOT._pidfd_send_signal(9, BOOT.signal.SIGTERM)
        native_signal.assert_called_once_with(9, BOOT.signal.SIGTERM, None, 0)
        raw.assert_not_called()

    def test_fallback_opens_and_signals_through_syscalls(self) -> None:
        raw = MagicMock(return_value=11)
        with self.linux("aarch64"), self.without_native(), patch.object(BOOT, "_raw_syscall", raw):
            self.assertIs(BOOT._pidfd_supported(), True)
            self.assertEqual(BOOT._pidfd_open(43), 11)
            BOOT._pidfd_send_signal(11, BOOT.signal.SIGKILL)
        self.assertEqual([call.args for call in raw.call_args_list], [(434, 43, 0), (424, 11, int(BOOT.signal.SIGKILL), 0, 0)])

    def test_unsupported_architecture_fails_closed(self) -> None:
        with self.linux("riscv64"), self.without_native():
            self.assertIs(BOOT._pidfd_supported(), False)
            with self.assertRaises(BOOT.EnforcementUnavailable):
                BOOT._pidfd_open(43)
            with self.assertRaises(BOOT.EnforcementUnavailable):
                BOOT._pidfd_send_signal(11, BOOT.signal.SIGTERM)

    def raw(self, result: int, error: int = 0) -> tuple[object, MagicMock]:
        function = MagicMock(return_value=result)
        library = SimpleNamespace(syscall=function)
        with patch.object(BOOT.ctypes, "CDLL", lambda name, use_errno: library), \
             patch.object(BOOT.ctypes, "get_errno", lambda: error), patch.object(BOOT.ctypes, "set_errno", lambda value: 0):
            try:
                return BOOT._raw_syscall(434, 43, 0), function
            except OSError as failure:
                return failure, function

    def test_raw_syscall_passes_long_arguments_and_returns_the_result(self) -> None:
        result, function = self.raw(7)
        self.assertEqual(result, 7)
        self.assertEqual([argument.value for argument in function.call_args.args], [434, 43, 0])
        self.assertIs(function.restype, BOOT.ctypes.c_long)

    def test_raw_syscall_errors_map_to_errno_specific_exceptions(self) -> None:
        for error, expected in ((errno.ESRCH, ProcessLookupError), (errno.EPERM, PermissionError),
                                (errno.ENOSYS, OSError), (errno.EINVAL, OSError)):
            with self.subTest(error=error):
                failure, _ = self.raw(-1, error)
                self.assertIsInstance(failure, expected)
                self.assertEqual(failure.errno, error)

    def test_owned_signal_requires_only_the_signal_primitive(self) -> None:
        with self.linux("x86_64"), self.without_native():
            self.assertIs(BOOT._pidfd_available(BOOT.signal, "pidfd_send_signal"), True)
        with self.linux("riscv64"), self.without_native():
            self.assertIs(BOOT._pidfd_available(BOOT.signal, "pidfd_send_signal"), False)

    def test_owned_termination_signals_through_the_fallback(self) -> None:
        value = identity()
        store = SimpleNamespace(directory=75, path=Path("/synthetic/evidence"),
                                metadata=SimpleNamespace(st_dev=211, st_ino=223, st_uid=1000),
                                verify=lambda: None, append=lambda phase, payload: (1, "f" * 64))
        scope = BOOT._NativeScope(29, "/synthetic", 70, 71, (), pidfd_device=101, pidfd_inode=103)
        runner = BOOT._NativeRunner(object(), store, {"version": 1})
        endpoint = BOOT._NativeEndpoint(SimpleNamespace(fileno=lambda: 73), SimpleNamespace(fileno=lambda: 74),
                                        value.main_pid, value.uid, {}, 107, 109, 113, 127)
        record = BOOT.LinuxProvisionalLaunch(value, runner, 72, scope, endpoint)
        resources = BOOT._termination_resources(record)
        exited = [False]
        raw = MagicMock(side_effect=lambda *args: exited.__setitem__(0, True) or 0)
        fcntl = SimpleNamespace(F_DUPFD_CLOEXEC=1030, fcntl=lambda descriptor, command, minimum: 90)
        with self.linux("x86_64"), self.without_native(), ExitStack() as stack:
            for target, name, replacement in (
                    (BOOT.time, "monotonic", lambda: 1000.0),
                    (BOOT.os, "fstat", lambda descriptor: SimpleNamespace(st_dev=101, st_ino=103)),
                    (BOOT.os, "close", lambda descriptor: None),
                    (BOOT.select, "select", lambda read, write, errors, timeout: ((list(read) if exited[0] else []), [], [])),
                    (BOOT, "_read_path", lambda *args, **kwargs: b"Pid: 43\nNSpid: 43\n"),
                    (BOOT.importlib, "import_module", lambda name: fcntl), (BOOT, "_raw_syscall", raw)):
                stack.enter_context(patch.object(target, name, replacement))
            self.assertIs(BOOT._signal_owned_pidfd(record, value, resources, BOOT.signal.SIGTERM, 1005.0), True)
        raw.assert_called_once_with(424, 90, int(BOOT.signal.SIGTERM), 0, 0)

    def test_required_support_accepts_fallback_primitives(self) -> None:
        class PastPrimitives(Exception):
            pass

        def manager(environment: object) -> None:
            raise PastPrimitives()

        for supported, expected in ((True, PastPrimitives), (False, BOOT.EnforcementUnavailable)):
            with self.subTest(supported=supported), self.linux("x86_64"), \
                 patch.object(BOOT, "_pidfd_supported", lambda: supported), \
                 patch.object(BOOT.socket, "SO_PEERCRED", 17, create=True), \
                 patch.object(BOOT, "_manager_environment", manager), self.assertRaises(expected):
                BOOT._required_linux_support(BOOT.LinuxDaemonLaunchPolicy(1, CAP, 0), time.monotonic() + 10.0)


class EnvironmentFileRendering(unittest.TestCase):
    """D2: values reach the manager through a file systemd parses back exactly."""

    def test_values_round_trip_through_systemd_quoting(self) -> None:
        values = {"PLAIN": "value", "SPACES": "  padded value  ", "QUOTE": 'say "hi"', "BACKSLASH": "a\\b\\",
                  "DOLLAR": "$HOME and ${PATH}", "BACKTICK": "`id`", "NEWLINE": "one\ntwo\n", "TAB": "a\tb",
                  "EMPTY": "", "EQUALS": "a=b=c", "HASH": "#not a comment", "UNICODE": "caf\u00e9 \u2713"}
        self.assertEqual(parse_environment_file(BOOT._environment_file(values)), values)

    def test_non_utf8_value_is_refused_by_name_only(self) -> None:
        with self.assertRaises(BOOT.EnforcementUnavailable) as caught:
            BOOT._environment_file({"BROKEN": "secret\udcff"})
        self.assertIn("BROKEN", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_invalid_names_are_refused(self) -> None:
        with self.assertRaises(BOOT.EnforcementUnavailable):
            BOOT._environment_file({"BAD-NAME": "x"})


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class RuntimeBarrierDirectory(unittest.TestCase):
    """D3: the barrier lives in a short owner-only runtime directory, whatever HOME is."""

    def test_generation_directory_is_owner_only_and_short(self) -> None:
        root = short_private_directory() / "truememory-cap"
        with patch.object(BOOT, "_runtime_root", lambda: root):
            path = BOOT._prepare_runtime_generation(GENERATION)
        self.assertEqual(path, root / GENERATION)
        self.assertEqual(stat.S_IMODE(os.stat(root).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o700)
        self.assertLessEqual(len(os.fsencode(str(path / BOOT._BARRIER_NAME))), BOOT._SUN_PATH_MAX)

    def test_shared_or_linked_runtime_roots_are_refused(self) -> None:
        base = short_private_directory()
        shared = base / "shared"
        shared.mkdir(mode=0o700)
        os.chmod(shared, 0o755)
        linked = base / "linked"
        linked.symlink_to(base / "elsewhere")
        (base / "elsewhere").mkdir(mode=0o700)
        for root, expected in ((shared, BOOT.EnforcementUnavailable), (linked, OSError)):
            with self.subTest(root=root.name), patch.object(BOOT, "_runtime_root", lambda: root), self.assertRaises(expected):
                BOOT._prepare_runtime_generation(GENERATION)

    def test_overlong_barrier_path_is_refused_before_bind(self) -> None:
        root = short_private_directory() / "truememory-cap"
        with patch.object(BOOT, "_runtime_root", lambda: root), patch.object(BOOT, "_SUN_PATH_MAX", 20), \
             self.assertRaises(BOOT.EnforcementUnavailable):
            BOOT._prepare_runtime_generation(GENERATION)

    def test_discard_removes_barrier_and_environment(self) -> None:
        root = short_private_directory() / "truememory-cap"
        with patch.object(BOOT, "_runtime_root", lambda: root):
            path = BOOT._prepare_runtime_generation(GENERATION)
            write_private(path / BOOT._ENVIRONMENT_NAME, b"A=\"1\"\n")
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(path / BOOT._BARRIER_NAME))
            finally:
                listener.close()
            BOOT._discard_runtime_generation(GENERATION)
            self.assertFalse(path.exists())
            BOOT._discard_runtime_generation(GENERATION)  # idempotent

    def test_retirement_discards_crash_leftovers(self) -> None:
        root = short_private_directory() / "truememory-cap"
        home = private_directory()
        (home / ".truememory" / "memory-cap-launches" / GENERATION).mkdir(parents=True, mode=0o700)
        with patch.object(BOOT, "_runtime_root", lambda: root), patch.object(BOOT.Path, "home", return_value=home):
            path = BOOT._prepare_runtime_generation(GENERATION)
            write_private(path / BOOT._ENVIRONMENT_NAME, b"SECRET=\"value\"\n")
            self.assertIs(BOOT._mark_generation_retired(GENERATION), True)
        self.assertFalse(path.exists())

    def test_child_connects_to_the_runtime_barrier(self) -> None:
        root = short_private_directory() / "truememory-cap"
        evidence = private_directory() / GENERATION
        evidence.mkdir(mode=0o700)
        descriptor = os.open(evidence, os.O_RDONLY)
        connected: list[str] = []

        class Endpoint:
            def settimeout(self, value: float) -> None:
                pass

            def connect(self, address: str) -> None:
                connected.append(address)
                raise ConnectionRefusedError("synthetic absent launcher")

            def close(self) -> None:
                pass

        manifest = {"deadline": time.monotonic() + 30.0, "generation": GENERATION, "bootstrap_sha256": "d" * 64}
        with patch.object(BOOT.sys, "platform", "linux"), \
             patch.object(BOOT.sys, "flags", SimpleNamespace(no_site=1, isolated=1)), \
             patch.object(BOOT, "_runtime_root", lambda: root), \
             patch.object(BOOT, "_child_bundle", lambda path: (descriptor, evidence / "manifest-v1.json", manifest, "e" * 64)), \
             patch.object(BOOT, "_process_identity", lambda pid: (11, 3000.5)), \
             patch.object(BOOT.socket, "socket", lambda *args: Endpoint()), self.assertRaises(ConnectionRefusedError):
            BOOT._child_main(str(evidence / "manifest-v1.json"))
        self.assertEqual(connected, [str(root / GENERATION / BOOT._BARRIER_NAME)])
        self.assertTrue((evidence / "child-startup-failed-v1.json").exists())


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class LaunchArgumentsAndCleanup(unittest.TestCase):
    """D2/D3: start() passes no environment values in argv, binds safely for any HOME, and cleans up."""

    SECRET = 'tok-"quoted"-$VALUE-`x`'

    def start(self, *, home_length: int = 0, systemd_run_error: BaseException | None = None,
              bind_error: BaseException | None = None, systemd_run_succeeds: bool = False,
              child_connects: bool = False) -> tuple[list[list[str]], dict, Path, Path, MagicMock]:
        base = private_directory()
        home = base
        if home_length:
            home = base / ("h" * max(1, home_length - len(str(base)) - 1))
            home.mkdir(mode=0o700)
        runtime = short_private_directory() / "truememory-cap"
        slice_relative = "/user.slice/app.slice/app-truememory" + GENERATION + ".slice"
        commands: list[list[str]] = []
        observed: dict = {}
        error = systemd_run_error if systemd_run_error is not None else BOOT.EnforcementUnavailable("synthetic stop")

        def run(arguments: list[str], deadline: float, **kwargs: object) -> tuple[object, bytes]:
            commands.append(list(arguments))
            if arguments[0] == "/usr/bin/busctl":
                return SimpleNamespace(), b""
            if arguments[0] != "/usr/bin/systemd-run":
                raise AssertionError(arguments)
            observed["kwargs"] = kwargs
            environment_path = Path(next(item for item in arguments if item.startswith("--property=EnvironmentFile=")).split("=", 2)[2])
            observed["environment_path"] = environment_path
            observed["mode"] = stat.S_IMODE(os.stat(environment_path).st_mode)
            observed["directory_mode"] = stat.S_IMODE(os.stat(environment_path.parent).st_mode)
            observed["environment"] = parse_environment_file(environment_path.read_bytes())
            barrier = environment_path.parent / BOOT._BARRIER_NAME
            observed["barrier"] = barrier
            observed["barrier_is_socket"] = stat.S_ISSOCK(os.stat(barrier).st_mode)
            if child_connects:
                # Stand in for the started child reaching the barrier.
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.connect(str(barrier))
                observed["client"] = client
                return SimpleNamespace(), b""
            if systemd_run_succeeds:
                return SimpleNamespace(), b""
            raise error

        def peer(connection: object) -> tuple[int, int]:
            # The first step after accept: the barrier path must already be gone.
            observed["runtime_after_accept"] = (runtime / GENERATION).exists()
            raise BOOT.EnforcementUnavailable("synthetic stop after accept")

        real_left = BOOT._left

        def left(deadline: float) -> float:
            # The first wait after a successful systemd-run is the barrier accept.
            if systemd_run_succeeds and "environment_path" in observed and "after_run" not in observed:
                observed["after_run"] = observed["environment_path"].exists()
                raise TimeoutError("synthetic: stop before accept")
            return real_left(deadline)

        real_open = BOOT._open_directory

        def open_directory(path: Path, *, uid: int | None = None) -> int:
            if str(path).startswith("/synthetic"):
                return os.open(os.devnull, os.O_RDONLY)
            return real_open(path, uid=uid)

        slice_properties = {"LoadState": "loaded", "ActiveState": "active", "InvocationID": "b" * 32,
                            "ControlGroup": slice_relative, "MemoryMax": str(CAP), "MemorySwapMax": "0"}
        abandon = MagicMock(return_value=True)
        environment = {"PATH": "/usr/bin", "SECRET_TOKEN": self.SECRET, "MULTILINE": "a\nb"}

        class Listener:
            def bind(self, address: str) -> None:
                raise bind_error

            def close(self) -> None:
                pass

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environment, clear=True))
            for target, name, value in (
                    (BOOT.sys, "platform", "linux"), (BOOT.Path, "home", lambda: home),
                    (BOOT, "_pidfd_supported", lambda: True),
                    (BOOT, "_required_linux_support", lambda *args: (29, "/synthetic", "/user.slice/app.slice", ())),
                    (BOOT, "_sweep_retired_generations", lambda **kwargs: 0),
                    (BOOT, "_executable_pin", lambda path: {"path": path}),
                    (BOOT, "_process_identity", lambda pid: (11, 3000.5)), (BOOT, "_boot_id", lambda: BOOT_ID),
                    (BOOT, "_runtime_root", lambda: runtime), (BOOT, "_run_command", run), (BOOT, "_left", left),
                    (BOOT, "_peer", peer),
                    (BOOT, "_properties", lambda unit, names, deadline: {key: slice_properties[key] for key in names}),
                    (BOOT, "_open_directory", open_directory), (BOOT, "_filesystem_magic", lambda descriptor: None),
                    (BOOT, "_read_at", lambda descriptor, name: {"memory.max": str(CAP), "memory.swap.max": "0", "memory.current": "0"}[name]),
                    (BOOT, "_controller_events", lambda *args: ()),
                    (BOOT, "_recursive_members", lambda descriptor, relative: ((relative, ()),))):
                stack.enter_context(patch.object(target, name, value))
            stack.enter_context(patch.object(BOOT.socket, "SO_PEERCRED", 17, create=True))
            stack.enter_context(patch.object(BOOT.LinuxAdapter, "_abandon_partial", abandon))
            if bind_error is not None:
                stack.enter_context(patch.object(BOOT.socket, "socket", lambda *args: Listener()))
            if child_connects:
                expected = (BOOT.EnforcementUnavailable,)
            elif systemd_run_succeeds:
                expected = (TimeoutError,)
            else:
                expected = (type(error), type(bind_error) if bind_error is not None else type(error))
            try:
                with self.assertRaises(expected):
                    BOOT.LinuxAdapter().start(BOOT.LinuxDaemonLaunchPolicy(1, CAP, 0), GENERATION,
                                              os.path.abspath(sys.executable), time.monotonic() + 10.0)
            finally:
                if "client" in observed:
                    observed["client"].close()
        evidence = home / ".truememory" / "memory-cap-launches" / GENERATION
        return commands, observed, runtime, evidence, abandon

    def test_no_environment_value_or_name_only_setenv_reaches_argv(self) -> None:
        commands, observed, _, _, _ = self.start()
        launch = next(command for command in commands if command[0] == "/usr/bin/systemd-run")
        self.assertFalse(any(item.startswith("--setenv") or item.startswith("-E") for item in launch))
        self.assertFalse(any(self.SECRET in item or "MULTILINE" in item for item in launch))
        self.assertEqual(observed["kwargs"], {})

    def test_environment_file_is_private_complete_and_exact(self) -> None:
        _, observed, _, _, _ = self.start()
        self.assertEqual((observed["mode"], observed["directory_mode"]), (0o600, 0o700))
        self.assertEqual(observed["environment"], {"PATH": "/usr/bin", "SECRET_TOKEN": self.SECRET, "MULTILINE": "a\nb",
                                                   "TRUEMEMORY_MODEL_SERVER_GENERATION": GENERATION, FLAG: "1",
                                                   "TRUEMEMORY_CAP_CHILD": "1"})

    def test_environment_file_is_removed_once_the_manager_has_started_the_child(self) -> None:
        _, observed, _, _, _ = self.start(systemd_run_succeeds=True)
        self.assertIs(observed["after_run"], False)

    def test_barrier_path_is_removed_as_soon_as_the_child_connects(self) -> None:
        _, observed, runtime, _, _ = self.start(child_connects=True)
        self.assertIs(observed["runtime_after_accept"], False)
        self.assertEqual(list(runtime.iterdir()), [])

    def test_environment_file_and_runtime_directory_are_removed_after_a_failed_launch(self) -> None:
        _, observed, runtime, _, abandon = self.start()
        self.assertFalse(observed["environment_path"].exists())
        self.assertFalse((runtime / GENERATION).exists())
        abandon.assert_called_once()

    def test_barrier_binds_in_the_runtime_directory_for_a_long_home(self) -> None:
        commands, observed, runtime, evidence, _ = self.start(home_length=220)
        self.assertGreater(len(str(evidence)), 200)
        self.assertTrue(any(command[0] == "/usr/bin/systemd-run" for command in commands))
        self.assertIs(observed["barrier_is_socket"], True)
        self.assertEqual(observed["barrier"], runtime / GENERATION / BOOT._BARRIER_NAME)
        self.assertLessEqual(len(os.fsencode(str(observed["barrier"]))), BOOT._SUN_PATH_MAX)

    def test_bind_failure_runs_launch_cleanup(self) -> None:
        commands, _, runtime, evidence, abandon = self.start(bind_error=OSError("AF_UNIX path too long"))
        abandon.assert_called_once()
        self.assertFalse(any(command[0] == "/usr/bin/busctl" for command in commands))
        phases = [BOOT._strict_json(path.read_bytes())["phase"] for path in sorted(evidence.glob("receipt-*.json"))]
        self.assertEqual(phases, ["startup-unresolved"])
        self.assertFalse((runtime / GENERATION).exists())


if __name__ == "__main__":
    unittest.main()
