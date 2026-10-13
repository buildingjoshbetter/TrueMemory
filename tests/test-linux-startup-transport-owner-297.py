"""Synthetic inactive transport controls. No DLL, bus, socket or pointer reads."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace
from typing import Callable
import unittest
from unittest.mock import patch


def load_source() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "truememory_memory_bootstrap.py"
    name = "truememory_transport_author_source"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Function:
    def __init__(self, runtime: InertRuntime, name: str) -> None:
        self.runtime = runtime
        self.name = name
        self.argtypes = None
        self.restype = None
        self.errcheck = None

    def __call__(self, *arguments: object) -> object:
        return self.runtime.call(self.name, arguments)


class InertRuntime:
    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.clock = 100.0
        self.calls = []
        self.strings = {}
        self.next_string = 100000
        self.messages = {}
        self.next_message = 1000
        self.next_slot = 4000
        self.next_cookie = 70
        self.next_serial = 7000
        self.bad_signature = None
        self.bad_version = None
        self.bad_uid = None
        self.bad_reply_cookie = False
        self.bad_sender = False
        self.skip_reply = False
        self.reuse_after_close = False
        self.rebound = False
        self.close_calls = []
        self.closed = False
        self.raise_name = None
        self.expire_name = None
        self.library = SimpleNamespace()
        names = (
            "new", "set_fd", "set_bus_client", "set_anonymous", "negotiate_fds",
            "set_exit_on_disconnect", "start", "is_ready", "process", "get_fd",
            "get_events", "get_timeout", "get_bus_id", "get_unique_name", "call_async",
            "add_filter", "close", "unref", "send", "add_match_async",
        )
        message_names = (
            "new_method_call", "set_auto_start", "set_allow_interactive_authorization",
            "append_basic", "get_cookie", "get_reply_cookie", "get_sender", "get_destination",
            "get_type", "get_signature", "get_path", "get_interface", "get_member", "read_basic",
            "enter_container", "exit_container", "at_end", "unref", "ref", "open_container", "close_container",
        )
        for suffix in names:
            setattr(self.library, "sd_bus_" + suffix, Function(self, "sd_bus_" + suffix))
        for suffix in message_names:
            setattr(self.library, "sd_bus_message_" + suffix, Function(self, "sd_bus_message_" + suffix))
        self.library.sd_bus_slot_unref = Function(self, "sd_bus_slot_unref")

    def string(self, value: str) -> int:
        pointer = self.next_string
        data = value.encode("ascii") + b"\x00"
        self.next_string += len(data) + 16
        for index, byte in enumerate(data):
            self.strings[pointer + index] = byte
        return pointer

    def byte(self, address: int) -> SimpleNamespace:
        if address not in self.strings:
            raise AssertionError("Nonowned synthetic address")
        return SimpleNamespace(value=self.strings[address])

    def metadata(self, descriptor: int) -> SimpleNamespace:
        if descriptor != 33:
            raise AssertionError("Unexpected synthetic descriptor")
        inode = 99 if self.rebound else 33
        return SimpleNamespace(st_dev=1, st_ino=inode, st_mode=stat.S_IFSOCK | 0o600,
                               st_uid=1000, st_gid=77, st_size=0, st_mtime_ns=0,
                               st_ctime_ns=0, st_nlink=1)

    def load_backend(self, record: object, token: object) -> None:
        record.resources.library = self.library
        backend = self.module._NativeStartupBackend(self.library)
        record.resources.backend = record.seal.backend = backend

    def open_endpoint(self, record: object, token: object) -> None:
        module = self.module
        resources = record.resources
        resources.raw_fd = 33
        resources.raw_pin = (1, 33, stat.S_IFSOCK | 0o600, 1000, 77)
        resources.transfer = "bus_owned"
        resources.peer = (200, 1000, 77)
        out = module.ctypes.c_void_p(100)
        reference = module._NativeStartupReference("bus", out)
        reference.pointer, reference.state = 100, "owned"
        resources.bus = record.seal.bus = reference
        resources.references.append(reference)
        module._native_startup_commit(record, peer_credentials=resources.peer,
                                      connection_closed=False, native_objects_released=False)

    def call(self, name: str, arguments: tuple[object, ...]) -> object:
        self.calls.append(name)
        if name == self.raise_name:
            raise RuntimeError("Synthetic lost return")
        if name == self.expire_name:
            self.clock = 201.0
        if name == "sd_bus_is_ready":
            return 1
        if name == "sd_bus_get_bus_id":
            for index in range(16):
                arguments[1]._obj.bytes[index] = index + 1
            return 0
        if name == "sd_bus_get_unique_name":
            arguments[1]._obj.value = self.string(":1.2")
            return 0
        if name == "sd_bus_add_filter":
            self.next_slot += 1
            arguments[1]._obj.value = self.next_slot
            return 0
        if name == "sd_bus_message_new_method_call":
            self.next_message += 1
            self.next_cookie += 1
            pointer = self.next_message
            arguments[1]._obj.value = pointer
            self.messages[pointer] = {"destination": arguments[2].decode(), "member": arguments[5].decode(),
                                      "cookie": self.next_cookie, "kind": 1}
            return 0
        if name == "sd_bus_call_async":
            self.next_slot += 1
            arguments[1]._obj.value = self.next_slot
            if not self.skip_reply:
                original = self.messages[arguments[2].value]
                self.next_message += 1
                self.next_serial += 1
                pointer = self.next_message
                member = original["member"]
                signature = {"AddMatch": "", "GetNameOwner": "s", "GetConnectionUnixUser": "u", "Get": "v", "Subscribe": ""}[member]
                value = {"GetNameOwner": ":1.9", "GetConnectionUnixUser": 1000 if self.bad_uid is None else self.bad_uid,
                         "Get": "249.11" if self.bad_version is None else self.bad_version}.get(member)
                self.messages[pointer] = {"sender": ":1.999" if self.bad_sender else original["destination"],
                                          "destination": ":1.2", "signature": self.bad_signature if self.bad_signature is not None else signature,
                                          "reply_cookie": original["cookie"] + (1 if self.bad_reply_cookie else 0),
                                          "serial": self.next_serial, "kind": 2, "value": value}
                arguments[3](pointer, arguments[4].value, None)
            return 1
        if name == "sd_bus_message_get_cookie":
            item = self.messages[arguments[0].value]
            arguments[1]._obj.value = item.get("serial", item.get("cookie"))
            return 0
        if name == "sd_bus_message_get_reply_cookie":
            arguments[1]._obj.value = self.messages[arguments[0].value]["reply_cookie"]
            return 0
        if name == "sd_bus_message_get_type":
            arguments[1]._obj.value = self.messages[arguments[0].value]["kind"]
            return 0
        if name in ("sd_bus_message_get_sender", "sd_bus_message_get_destination", "sd_bus_message_get_signature"):
            field = name.removeprefix("sd_bus_message_get_")
            return self.string(self.messages[arguments[0].value][field])
        if name == "sd_bus_message_read_basic":
            value = self.messages[arguments[0].value]["value"]
            arguments[2]._obj.value = self.string(value) if arguments[1] == b"s" else value
            return 1
        if name in ("sd_bus_message_enter_container", "sd_bus_message_exit_container", "sd_bus_message_at_end"):
            return 1
        if name in ("sd_bus_message_unref", "sd_bus_slot_unref", "sd_bus_unref"):
            return None
        if name == "sd_bus_close":
            self.closed = True
            if self.reuse_after_close:
                self.rebound = True
            return None
        if name == "sd_bus_process":
            self.clock += 100.0
            return 0
        return 0


class TransportOwnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_source()
        self.runtime = InertRuntime(self.module)
        runtime = self.runtime
        byte_type = type("InertByte", (), {"from_address": staticmethod(runtime.byte)})
        patches = (
            patch.object(self.module.time, "monotonic", lambda: runtime.clock),
            patch.object(self.module.os, "getuid", lambda: 1000),
            patch.object(self.module.os, "geteuid", lambda: 1000),
            patch.object(self.module.os, "fstat", runtime.metadata),
            patch.object(self.module.os, "close", lambda fd: runtime.close_calls.append(fd)),
            patch.object(self.module.ctypes, "c_ubyte", byte_type),
            patch.object(self.module.ctypes, "CFUNCTYPE", lambda *args: (lambda callback: callback)),
            patch.object(self.module, "_native_startup_load_backend", runtime.load_backend),
            patch.object(self.module, "_native_startup_open_endpoint", runtime.open_endpoint),
            patch.object(self.module.ctypes, "CDLL", side_effect=AssertionError("DLL is forbidden")),
            patch.object(self.module.socket, "socket", side_effect=AssertionError("Socket is forbidden")),
            patch.object(self.module.select, "poll", side_effect=AssertionError("Poll is forbidden")),
            patch.object(self.module.os, "open", side_effect=AssertionError("Filesystem is forbidden")),
            patch.object(self.module.os, "read", side_effect=AssertionError("Filesystem is forbidden")),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def request(self) -> object:
        module = self.module
        pin = module.StartupNativeBackendPin(1, module._NATIVE_STARTUP_LIBRARY,
                                              module._NATIVE_STARTUP_LIBRARY_SHA,
                                              module._NATIVE_STARTUP_LIBRARY_STAT)
        return module.StartupNativeRequest(1, module.LinuxDaemonLaunchPolicy(1, 268435456, 0),
                                           "0" * 32, "/synthetic/python", pin)

    def owner(self) -> object:
        return self.module.retain_native_startup(self.request(), deadline=200.0)

    def refusal(self, callback: Callable[[], object], code: str) -> None:
        with self.assertRaises(self.module.EnforcementUnavailable) as caught:
            callback()
        self.assertEqual(str(caught.exception), "native_startup:" + code)

    def test_prepared_required_fields_and_identity(self) -> None:
        owner = self.owner()
        state = self.module.native_startup_snapshot(owner)
        self.assertIs(state.owner, owner)
        self.assertEqual((state.phase, state.operation_sequence, state.cleanup_sequence), ("prepared", 0, 0))
        self.assertIsNone(state.context)
        self.assertIsNone(state.peer_credentials)
        self.assertTrue(state.connection_closed)
        self.assertTrue(state.native_objects_released)
        self.assertFalse(state.slice_attempted)
        self.assertFalse(state.service_attempted)
        self.assertEqual(self.runtime.calls, [])

    def test_sequence_exact_domains_and_boundaries(self) -> None:
        module = self.module
        self.assertEqual(module._native_startup_sequence_pair(0), (1, 2))
        self.assertEqual(module._native_startup_sequence_pair(module._NATIVE_STARTUP_U64 - 2),
                         (module._NATIVE_STARTUP_U64 - 1, module._NATIVE_STARTUP_U64))
        for value in (True, False, -1, 1.0, None, module._NATIVE_STARTUP_U64 + 1):
            with self.subTest(value=value):
                self.refusal(lambda: module._native_startup_sequence_pair(value), "invalid_input")
        for value in (module._NATIVE_STARTUP_U64 - 1, module._NATIVE_STARTUP_U64):
            with self.subTest(value=value):
                self.refusal(lambda: module._native_startup_sequence_pair(value), "sequence_exhausted")
                self.refusal(lambda: module._native_startup_sequence_pair(value, cleanup=True), "cleanup_counter_exhausted")
        self.refusal(lambda: module._native_startup_sequence_pair(0, cleanup=1), "invalid_input")

    def test_deadline_exact_float_and_future(self) -> None:
        for deadline in (True, 200, None, float("inf"), float("nan")):
            with self.subTest(deadline=deadline):
                self.refusal(lambda: self.module.retain_native_startup(self.request(), deadline=deadline), "invalid_input")
        for deadline in (99.0, 100.0):
            with self.subTest(deadline=deadline):
                self.refusal(lambda: self.module.retain_native_startup(self.request(), deadline=deadline), "deadline_expired")

    def test_strict_constructor_subclasses(self) -> None:
        module = self.module
        class OwnerSubtype(module.NativeStartupOwner):
            pass
        self.refusal(lambda: OwnerSubtype(1, 1, 1), "invalid_input")
        class StringSubtype(str):
            pass
        pin = self.request().backend_pin
        self.refusal(lambda: module.StartupNativeBackendPin(1, StringSubtype(pin.canonical_path), pin.sha256, pin.stat_pin), "invalid_input")

    def test_pin_boolean_and_mode_rejected(self) -> None:
        module = self.module
        original = self.request().backend_pin
        for index in range(9):
            fields = list(original.stat_pin)
            fields[index] = True
            with self.subTest(index=index):
                self.refusal(lambda: module.StartupNativeBackendPin(1, original.canonical_path, original.sha256, tuple(fields)), "invalid_input")
        fields = list(original.stat_pin)
        fields[2] |= 0o002
        self.refusal(lambda: module.StartupNativeBackendPin(1, original.canonical_path, original.sha256, tuple(fields)), "invalid_input")

    def test_request_copy_independent(self) -> None:
        request = self.request()
        owner = self.module.retain_native_startup(request, deadline=200.0)
        object.__setattr__(request.policy, "cap_bytes", 1)
        state = self.module.native_startup_snapshot(owner)
        self.assertEqual(state.request.policy.cap_bytes, 268435456)
        object.__setattr__(state.request.policy, "cap_bytes", 2)
        self.assertEqual(self.module.native_startup_snapshot(owner).request.policy.cap_bytes, 268435456)

    def test_reflected_owner_is_not_authoritative(self) -> None:
        owner = self.owner()
        reflected = self.module.NativeStartupOwner(owner.version, owner.owner_id, owner.creator_pid)
        self.refusal(lambda: self.module.native_startup_snapshot(reflected), "invalid_owner")
        self.assertEqual(len(self.module._NATIVE_STARTUP_RECORDS), 1)

    def test_isolated_state_mutation_refuses(self) -> None:
        owner = self.owner()
        record = self.module._NATIVE_STARTUP_RECORDS[owner.owner_id]
        object.__setattr__(record.state, "original_deadline", 300.0)
        self.refusal(lambda: self.module.native_startup_snapshot(owner), "invalid_input")
        self.assertEqual(self.runtime.calls, [])

    def test_capacity_and_release_no_reused_ids(self) -> None:
        owners = [self.owner() for _ in range(16)]
        self.refusal(self.owner, "capacity_exhausted")
        self.module.close_precreate_startup(owners[0], expected_sequence=0, expected_cleanup_sequence=0, cleanup_deadline=102.0)
        released = self.module.release_closed_native_startup(owners[0], expected_sequence=0, expected_cleanup_sequence=2)
        self.assertEqual(released.phase, "released")
        self.assertGreater(self.owner().owner_id, owners[-1].owner_id)
        self.refusal(lambda: self.module.native_startup_snapshot(owners[0]), "invalid_owner")

    def test_inclusive_expiry_keeps_counters(self) -> None:
        owner = self.owner()
        self.runtime.clock = 200.0
        state = self.module.native_startup_snapshot(owner)
        self.assertEqual((state.phase, state.reason), ("terminal_unknown", "deadline_expired"))
        self.assertEqual((state.operation_sequence, state.cleanup_sequence), (0, 0))
        self.assertEqual(len(self.module._NATIVE_STARTUP_RECORDS), 1)
        self.assertEqual(self.runtime.calls, [])

    def test_clock_regression_refuses_without_authority(self) -> None:
        owner = self.owner()
        self.runtime.clock = 99.0
        self.refusal(lambda: self.module.native_startup_snapshot(owner), "clock_regressed")
        self.assertEqual(self.module._NATIVE_STARTUP_RECORDS[owner.owner_id].state.operation_sequence, 0)

    def test_fork_refuses_before_lock_clock(self) -> None:
        owner = self.owner()
        with patch.object(self.module.os, "getpid", lambda: owner.creator_pid + 1):
            with patch.object(self.module.time, "monotonic", side_effect=AssertionError("Clock must not run")):
                self.refusal(lambda: self.module.native_startup_snapshot(owner), "forked_owner")

    def test_complete_authentication_fixed_protocol(self) -> None:
        owner = self.owner()
        state = self.module.authenticate_startup_bus(owner, expected_sequence=0)
        self.assertEqual((state.phase, state.operation_sequence), ("subscribed", 2))
        self.assertEqual(state.manager_version, "249.11")
        self.assertEqual(state.peer_credentials, (200, 1000, 77))
        self.assertEqual(state.context.manager_unique_name, ":1.9")
        self.assertTrue(state.owner_match_confirmed)
        self.assertTrue(state.subscription_confirmed)
        self.assertEqual(self.runtime.calls.count("sd_bus_call_async"), 5)
        self.assertNotIn("sd_bus_send", self.runtime.calls)
        self.assertNotIn("sd_bus_add_match_async", self.runtime.calls)

    def test_stale_sequence_precedes_native_effect(self) -> None:
        owner = self.owner()
        self.refusal(lambda: self.module.authenticate_startup_bus(owner, expected_sequence=1), "stale_sequence")
        self.assertEqual(self.runtime.calls, [])

    def test_lost_native_return_retains_active_operation(self) -> None:
        owner = self.owner()
        self.runtime.raise_name = "sd_bus_is_ready"
        state = self.module.authenticate_startup_bus(owner, expected_sequence=0)
        record = self.module._NATIVE_STARTUP_RECORDS[owner.owner_id]
        self.assertEqual(state.phase, "terminal_unknown")
        self.assertEqual(record.resources.active_calls, 1)
        self.assertIs(self.module._NATIVE_STARTUP_OPERATION, record)
        frame = record.resources.frames[-1]
        self.assertTrue(frame.entered)
        self.assertFalse(frame.returned)
        before = list(self.runtime.calls)
        self.refusal(lambda: self.module.native_startup_snapshot(owner), "busy")
        self.refusal(lambda: self.module.close_precreate_startup(owner, expected_sequence=2, expected_cleanup_sequence=0, cleanup_deadline=102.0), "busy")
        self.refusal(lambda: self.module.release_closed_native_startup(owner, expected_sequence=2, expected_cleanup_sequence=0), "busy")
        self.assertEqual(self.runtime.calls, before)
        self.assertEqual(self.runtime.close_calls, [])

    def test_known_return_after_deadline_keeps_history_but_not_active_c(self) -> None:
        owner = self.owner()
        self.runtime.expire_name = "sd_bus_get_bus_id"
        state = self.module.authenticate_startup_bus(owner, expected_sequence=0)
        record = self.module._NATIVE_STARTUP_RECORDS[owner.owner_id]
        self.assertEqual((state.phase, state.reason), ("terminal_unknown", "deadline_expired"))
        self.assertEqual(record.resources.active_calls, 0)
        self.assertIsNone(self.module._NATIVE_STARTUP_OPERATION)
        self.assertTrue(record.resources.frames[-1].returned)
        self.assertEqual(record.operation.outcome[0], 2)
        self.runtime.expire_name = None
        closed = self.module.close_precreate_startup(owner, expected_sequence=2, expected_cleanup_sequence=0, cleanup_deadline=202.0)
        self.assertEqual(closed.phase, "closed_precreate")
        self.assertEqual(closed.original_deadline, 200.0)

    def test_invalid_authentication_replies_ungraded_unknown(self) -> None:
        for field, value in (("bad_signature", "ss"), ("bad_reply_cookie", True), ("bad_sender", True), ("bad_uid", 1001), ("bad_version", "2490")):
            with self.subTest(field=field):
                setattr(self.runtime, field, value)
                owner = self.owner()
                state = self.module.authenticate_startup_bus(owner, expected_sequence=0)
                self.assertEqual(state.phase, "terminal_unknown")
                self.assertIsNotNone(state.reason)
                self.assertEqual(state.operation_sequence, 2)
                self.assertFalse(state.subscription_confirmed)
                setattr(self.runtime, field, None if field in ("bad_signature", "bad_uid", "bad_version") else False)

    def test_strict_observed_version_grammar(self) -> None:
        for value in ("249", "249.11", "249.11-0ubuntu3.16pop0~synthetic"):
            with self.subTest(value=value):
                self.runtime.bad_version = value
                state = self.module.authenticate_startup_bus(self.owner(), expected_sequence=0)
                self.assertEqual(state.phase, "subscribed")
                self.assertEqual(state.manager_version, value)

    def test_cleanup_deadline_separate_exact_boundary(self) -> None:
        owner = self.owner()
        for value, code in ((102, "invalid_input"), (True, "invalid_input"), (100.0, "cleanup_deadline_invalid"), (102.000000001, "cleanup_deadline_invalid")):
            with self.subTest(value=value):
                self.refusal(lambda: self.module.close_precreate_startup(owner, expected_sequence=0, expected_cleanup_sequence=0, cleanup_deadline=value), code)
        self.assertEqual(self.module.close_precreate_startup(owner, expected_sequence=0, expected_cleanup_sequence=0, cleanup_deadline=102.0).phase, "closed_precreate")

    def test_expired_auth_cleanup_does_not_renew_original(self) -> None:
        owner = self.owner()
        self.runtime.clock = 201.0
        state = self.module.close_precreate_startup(owner, expected_sequence=0, expected_cleanup_sequence=0, cleanup_deadline=202.0)
        self.assertEqual(state.phase, "closed_precreate")
        self.assertEqual(state.original_deadline, 200.0)
        self.assertEqual((state.operation_sequence, state.cleanup_sequence), (0, 2))
        self.assertEqual(self.runtime.calls, [])

    def test_bus_close_before_reference_releases(self) -> None:
        owner = self.owner()
        self.module.authenticate_startup_bus(owner, expected_sequence=0)
        before = len(self.runtime.calls)
        state = self.module.close_precreate_startup(owner, expected_sequence=2, expected_cleanup_sequence=0, cleanup_deadline=102.0)
        self.assertEqual(state.phase, "closed_precreate")
        cleanup_calls = self.runtime.calls[before:]
        self.assertEqual(cleanup_calls[0], "sd_bus_close")
        self.assertEqual(cleanup_calls[-1], "sd_bus_unref")
        self.assertNotIn("sd_bus_process", cleanup_calls)
        self.assertNotIn("sd_bus_call_async", cleanup_calls)
        self.assertEqual(self.runtime.close_calls, [])

    def test_observed_close_then_reused_number_not_closed(self) -> None:
        owner = self.owner()
        self.module.authenticate_startup_bus(owner, expected_sequence=0)
        self.runtime.reuse_after_close = True
        state = self.module.close_precreate_startup(owner, expected_sequence=2, expected_cleanup_sequence=0, cleanup_deadline=102.0)
        self.assertEqual(state.phase, "closed_precreate")
        self.assertTrue(state.connection_closed)
        self.assertEqual(self.runtime.close_calls, [])

    def test_preclose_rebound_quarantines(self) -> None:
        owner = self.owner()
        self.module.authenticate_startup_bus(owner, expected_sequence=0)
        self.runtime.rebound = True
        before = len(self.runtime.calls)
        state = self.module.close_precreate_startup(owner, expected_sequence=2, expected_cleanup_sequence=0, cleanup_deadline=102.0)
        self.assertEqual(state.phase, "terminal_unknown")
        self.assertFalse(state.connection_closed)
        self.assertEqual(self.runtime.calls[before:], [])
        self.assertEqual(self.runtime.close_calls, [])

    def test_release_pure_preserves_final_counters(self) -> None:
        owner = self.owner()
        self.module.authenticate_startup_bus(owner, expected_sequence=0)
        self.module.close_precreate_startup(owner, expected_sequence=2, expected_cleanup_sequence=0, cleanup_deadline=102.0)
        before = len(self.runtime.calls)
        with patch.object(self.module.time, "monotonic", side_effect=AssertionError("Release is pure")):
            state = self.module.release_closed_native_startup(owner, expected_sequence=2, expected_cleanup_sequence=2)
        self.assertEqual((state.operation_sequence, state.cleanup_sequence), (2, 2))
        self.assertEqual(state.phase, "released")
        self.assertEqual(len(self.runtime.calls), before)

    def test_direct_evidence_boundary_checked_before_copy(self) -> None:
        owner = self.owner()
        record = self.module._NATIVE_STARTUP_RECORDS[owner.owner_id]
        event = {"sender": "org.freedesktop.DBus", "serial": 1, "value": "synthetic"}
        size = self.module._native_startup_evidence_size(event)
        record.resources.evidence_bytes = 65536 - size
        self.module._native_startup_evidence(record, event)
        self.assertEqual(record.resources.evidence_bytes, 65536)
        self.refusal(lambda: self.module._native_startup_evidence(record, {"sender": "org.freedesktop.DBus", "serial": 2}), "evidence_overflow")
        self.assertEqual(len(record.resources.events), 1)

    def test_direct_event_boundary_checked_before_copy(self) -> None:
        owner = self.owner()
        record = self.module._NATIVE_STARTUP_RECORDS[owner.owner_id]
        for serial in range(64):
            self.module._native_startup_evidence(record, {"sender": "org.freedesktop.DBus", "serial": serial + 1})
        self.refusal(lambda: self.module._native_startup_evidence(record, {"sender": "org.freedesktop.DBus", "serial": 65}), "evidence_overflow")
        self.assertEqual(len(record.resources.events), 64)


if __name__ == "__main__":
    unittest.main()
