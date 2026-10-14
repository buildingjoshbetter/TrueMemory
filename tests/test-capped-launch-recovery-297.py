"""Synthetic recovery controls for the opt-in capped launcher; no manager or kernel effects."""
from __future__ import annotations

import ast
import copy
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
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
    runner = SimpleNamespace(stop_complete=False, store=SimpleNamespace(directory=75))
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


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class StaleEndpointRecovery(unittest.TestCase):
    """F1: a dead claimed daemon leaves a stale endpoint that must not wedge capped mode."""

    def claim(self, child: dict | None = None) -> dict:
        return {"generation": GENERATION, "launcher": {"pid": 101, "born": 2000.5},
                "child": {"pid": 43, "born": 1000.5} if child is None else child}

    def verify(self, claim: object, birth: object, **patches: object) -> tuple[bool, MagicMock, MagicMock]:
        inspect = patches.get("inspect", MagicMock(return_value="synthetic-evidence"))
        adopt = MagicMock(return_value=True)
        with patch.object(client, "read_start_claim", return_value=claim), \
             patch.object(client, "process_birth", return_value=birth), \
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


@unittest.skipIf(sys.platform == "win32", "Linux-only launcher; POSIX synthetic fixtures")
class FailedLaunchClaimRetirement(unittest.TestCase):
    """F3: a failed launch retires its claim; F2: it never leaves its workload unowned."""

    def run_start(self, spawn: object, *, abandon: object = None) -> tuple[Path, MagicMock, object]:
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
                    (BOOT, "abandon_owned_launch", abandon)):
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

    def handle(self) -> object:
        return SimpleNamespace(pid=77, record=SimpleNamespace(identity=SimpleNamespace(birth_time=1234.5, cap_bytes=CAP)),
                               release_after_claim=MagicMock(side_effect=BOOT.EnforcementUnavailable("synthetic census")))

    def test_failed_release_terminates_owned_workload(self) -> None:
        handle = self.handle()
        abandon = MagicMock(return_value=True)
        claim_path, retain, outcome = self.run_start(MagicMock(return_value=handle), abandon=abandon)
        self.assertIsInstance(outcome, BOOT.EnforcementUnavailable)
        abandon.assert_called_once_with(handle)
        retain.assert_not_called()
        self.assertFalse(claim_path.exists())

    def test_unresolved_termination_keeps_owning_the_workload(self) -> None:
        handle = self.handle()
        claim_path, retain, outcome = self.run_start(MagicMock(return_value=handle), abandon=MagicMock(return_value=False))
        self.assertIsInstance(outcome, BOOT.EnforcementUnavailable)
        retain.assert_called_once_with(handle)
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
        release.assert_called_once_with(handle)

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
                      BOOT.LinuxDaemonLaunchPolicy(1, CAP, 0), GENERATION, "/synthetic", "/user.slice/app.slice", descriptors)
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
                      BOOT.LinuxDaemonLaunchPolicy(1, CAP, 0), GENERATION, "/synthetic", "/user.slice/app.slice", [])
        self.assertIs(retired, False)
        self.assertEqual(trace, ["startup-cleanup-deferred"])
        marked.assert_not_called()

    def test_terminated_record_is_not_closed_twice(self) -> None:
        handle = owned_handle()
        store = SimpleNamespace(directory=75, append=MagicMock())
        closer = MagicMock()
        with patch.object(BOOT, "abandon_owned_launch", return_value=True), patch.object(BOOT.os, "close", closer):
            retired = BOOT.LinuxAdapter()._abandon_partial(store, SimpleNamespace(close=MagicMock()), None, handle.record,
                      handle.policy, GENERATION, "/synthetic", "/user.slice/app.slice", [72])
        self.assertIs(retired, True)
        closer.assert_not_called()
        store.append.assert_not_called()


class UnitRetirement(unittest.TestCase):
    """F5: only provably empty generation units are stopped and failed state is cleared."""

    SLICE = "/user.slice/app.slice/app-truememory" + GENERATION + ".slice"

    def manager(self, units: dict[str, dict[str, str]], *, empty: list[bool] | None = None,
                collect_on_stop: bool = False, stop_error: bool = False) -> tuple[list[tuple[str, str]], ExitStack]:
        commands: list[tuple[str, str]] = []
        census = list(empty if empty is not None else [True] * 8)

        def properties(unit: str, names: tuple[str, ...], deadline: float) -> dict[str, str]:
            state = units.get(unit, {"LoadState": "not-found", "ActiveState": "inactive", "SubState": "dead",
                                     "MainPID": "0", "ControlGroup": ""})
            return {name: state[name] for name in names}

        def run(arguments: list[str], deadline: float, **kwargs: object) -> tuple[object, bytes]:
            self.assertEqual(arguments[:2], ["/usr/bin/systemctl", "--user"])
            command, unit = arguments[2], arguments[3]
            commands.append((command, unit))
            if stop_error:
                units.pop(unit, None)
                raise BOOT.EnforcementUnavailable("Owned manager command failed with status 5")
            if command == "reset-failed" or collect_on_stop:
                units.pop(unit, None)
            else:
                units[unit].update(ActiveState="inactive", SubState="dead", MainPID="0", ControlGroup="")
            return SimpleNamespace(), b""

        stack = ExitStack()
        stack.enter_context(patch.object(BOOT, "_properties", properties))
        stack.enter_context(patch.object(BOOT, "_run_command", run))
        stack.enter_context(patch.object(BOOT, "_generation_domain_empty", lambda *args: census.pop(0)))
        stack.enter_context(patch.object(BOOT.time, "sleep", lambda delay: None))
        return commands, stack

    def units(self, *, service: dict[str, str] | None = None, slice_state: dict[str, str] | None = None) -> dict[str, dict[str, str]]:
        slice_unit, service_unit = BOOT._generation_units(GENERATION)
        result = {}
        if service is not None:
            result[service_unit] = {"LoadState": "loaded", "SubState": "exited", "MainPID": "0",
                                    "ControlGroup": self.SLICE + "/" + service_unit, "ActiveState": "active", **service}
        if slice_state is not None:
            result[slice_unit] = {"LoadState": "loaded", "SubState": "active", "MainPID": "0",
                                  "ControlGroup": self.SLICE, "ActiveState": "active", **slice_state}
        return result

    def retire(self, **kwargs: object) -> bool:
        return BOOT._retire_generation_units(GENERATION, mount_point="/synthetic", parent_relative="/user.slice/app.slice",
                                             deadline=time.monotonic() + 10.0, **kwargs)

    def test_unloaded_generation_needs_no_commands(self) -> None:
        commands, stack = self.manager({})
        with stack:
            self.assertIs(self.retire(), True)
        self.assertEqual(commands, [])

    def test_failed_service_is_reset_and_empty_slice_stopped(self) -> None:
        slice_unit, service_unit = BOOT._generation_units(GENERATION)
        commands, stack = self.manager(self.units(service={"ActiveState": "failed", "SubState": "failed"}, slice_state={}))
        with stack:
            self.assertIs(self.retire(), True)
        self.assertEqual(commands, [("reset-failed", service_unit), ("stop", slice_unit)])

    def test_exited_service_then_slice_are_stopped(self) -> None:
        slice_unit, service_unit = BOOT._generation_units(GENERATION)
        commands, stack = self.manager(self.units(service={}, slice_state={}))
        with stack:
            self.assertIs(self.retire(), True)
        self.assertEqual(commands, [("stop", service_unit), ("stop", slice_unit)])

    def test_live_main_process_is_never_stopped(self) -> None:
        commands, stack = self.manager(self.units(service={"MainPID": "43", "SubState": "running"}, slice_state={}))
        with stack:
            self.assertIs(self.retire(settle=1.0), False)
        self.assertEqual(commands, [])

    def test_populated_or_foreign_domain_is_never_stopped(self) -> None:
        for units, empty in ((self.units(service={}, slice_state={}), [False] * 8),
                             (self.units(service={"ControlGroup": "/foreign.service"}, slice_state={}), None),
                             (self.units(slice_state={"ControlGroup": "/user.slice/foreign.slice"}), None)):
            commands, stack = self.manager(units, empty=empty)
            with self.subTest(units=units, empty=empty), stack:
                self.assertIs(self.retire(), False)
            self.assertEqual(commands, [])

    def test_slice_is_kept_when_a_member_appears_after_service_stop(self) -> None:
        _, service_unit = BOOT._generation_units(GENERATION)
        commands, stack = self.manager(self.units(service={}, slice_state={}), empty=[True, False])
        with stack:
            self.assertIs(self.retire(), False)
        self.assertEqual(commands, [("stop", service_unit)])

    def test_unit_collected_between_readback_and_command(self) -> None:
        _, service_unit = BOOT._generation_units(GENERATION)
        commands, stack = self.manager(self.units(service={}), stop_error=True)
        with stack:
            self.assertIs(BOOT._stop_unit(service_unit, time.monotonic() + 10.0), True)
        self.assertEqual(commands, [("stop", service_unit)])

    def test_launch_collects_failed_services_and_stop_clears_failures(self) -> None:
        tree = ast.parse((ROOT / "truememory_memory_bootstrap.py").read_text())
        adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LinuxAdapter")
        methods = {node.name: node for node in adapter.body if isinstance(node, ast.FunctionDef)}
        launch = next(node for node in ast.walk(methods["start"]) if isinstance(node, ast.List)
                      and any(isinstance(item, ast.Constant) and item.value == "/usr/bin/systemd-run" for item in node.elts))
        values = [item.value for item in launch.elts if isinstance(item, ast.Constant)]
        self.assertIn("--collect", values)
        self.assertIn("--property=RemainAfterExit=yes", values)
        stops = [node for node in ast.walk(methods["stop"]) if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "_stop_unit"]
        self.assertEqual(len(stops), 2)
        self.assertFalse(any(isinstance(node, ast.Constant) and node.value == "stop" for node in ast.walk(methods["stop"])))


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
    """F2/F3: the console retires its claim and keeps owning an unterminated workload."""

    def run_main(self, abandon: bool) -> tuple[Path, MagicMock, MagicMock]:
        home = private_directory()
        handle = SimpleNamespace(pid=77, record=SimpleNamespace(pidfd=72, identity=SimpleNamespace(birth_time=1234.5)),
                                 release_after_claim=MagicMock(side_effect=BOOT.EnforcementUnavailable("synthetic census")))
        abandon_mock = MagicMock(return_value=abandon)
        release = MagicMock(return_value=True)
        environment = {key: value for key, value in os.environ.items() if key != "TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB"}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environment, clear=True))
            for target, name, value in (
                    (BOOT.sys, "argv", ["truememory-model-server-capped"]), (BOOT.sys, "platform", "linux"),
                    (BOOT, "_physical_total", lambda: 4 * CAP), (BOOT, "_required_linux_support", lambda *args: None),
                    (BOOT.Path, "home", lambda: home), (BOOT, "_process_identity", lambda pid: (11, 3000.5)),
                    (BOOT, "_verified_psutil_birth", lambda pid, candidate: candidate),
                    (BOOT, "launch_owned_linux", MagicMock(return_value=handle)),
                    (BOOT, "abandon_owned_launch", abandon_mock), (BOOT, "release_exited_launch", release),
                    (BOOT.select, "select", lambda read, write, error, timeout: (read, [], []))):
                stack.enter_context(patch.object(target, name, value))
            with self.assertRaises(BOOT.EnforcementUnavailable):
                BOOT.main()
        return home / ".truememory" / "model_server.start.json", abandon_mock, release

    def test_terminated_failure_retires_claim(self) -> None:
        claim, abandon, release = self.run_main(True)
        self.assertFalse(claim.exists())
        abandon.assert_called_once()
        release.assert_not_called()

    def test_unterminated_failure_is_watched_until_exit(self) -> None:
        claim, abandon, release = self.run_main(False)
        self.assertFalse(claim.exists())
        release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
