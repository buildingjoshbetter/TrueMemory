"""Synthetic pure S0 controls. No native shim or application import."""
from __future__ import annotations

import ast
from dataclasses import fields, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys
import types
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "truememory_memory_bootstrap.py"
PUBLIC_CLASSES = ("StartupCreateStructure", "StartupBusContext", "StartupProperty", "StartupMessageEnvelope", "StartupCorrelationState", "StartupInvocationBinding")
PUBLIC_FUNCTIONS = ("redacted_create_metadata_sha256", "new_startup_correlation", "mark_startup_send_attempt", "bind_startup_request_cookie", "parse_startup_envelope", "startup_envelope_evidence_sha256", "advance_startup_correlation", "invalidate_startup_correlation", "expire_startup_correlation", "startup_correlation_binding")


def load_pure_source() -> types.ModuleType:
    tree = ast.parse(SOURCE.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) and target.id.startswith("_STARTUP_") for target in node.targets):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in PUBLIC_CLASSES + ("EnforcementUnavailable",):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and (node.name in PUBLIC_FUNCTIONS or node.name.startswith("_startup_")):
            selected.append(node)
    module = types.ModuleType("synthetic_startup_correlation_297")
    module.__dict__.update({"dataclass": __import__("dataclasses").dataclass, "hashlib": hashlib, "json": json, "math": math, "re": re})
    sys.modules[module.__name__] = module
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(SOURCE), "exec"), module.__dict__)
    return module


class StartupCorrelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.api = load_pure_source()

    def setUp(self) -> None:
        a = self.api
        self.context = a.StartupBusContext(1, "a" * 32, ":1.7", ":1.42", 1000)
        self.name = "app-synthetic.service"
        self.path = "/org/freedesktop/systemd1/unit/app_2dsynthetic_2eservice"
        self.job = "/org/freedesktop/systemd1/job/17"
        self.metadata = a.StartupCreateStructure(1, self.name, self.path, "fail", (("Environment", "as"), ("ExecStart", "a(sasb)")), ("LANG", "PYTHONPATH"), ())
        self.invocation = bytes(range(1, 17))

    def state(self, *, cookie: bool = True) -> object:
        a = self.api
        state = a.new_startup_correlation(self.context, self.metadata, deadline=10.0, now=1.0)
        state = a.mark_startup_send_attempt(state, now=1.1)
        return a.bind_startup_request_cookie(state, cookie=42, now=1.2) if cookie else state

    def envelope(self, kind: str, receive_ordinal: int, *, serial: int | None = None, **changes: object) -> object:
        packet: dict[str, object] = dict(version=1, ordinal=receive_ordinal, kind=kind, sender=":1.7", destination=None, serial=receive_ordinal + 100 if serial is None else serial, reply_cookie=None, path="/org/freedesktop/systemd1", interface="org.freedesktop.systemd1.Manager", member=None, signature="", payload=(), error_name=None, error_body_redacted=False)
        if kind == "unit_new" or kind == "unit_removed":
            packet.update(member="UnitNew" if kind == "unit_new" else "UnitRemoved", signature="so", payload=(self.name, self.path))
        elif kind == "job_new" or kind == "job_removed":
            packet.update(member="JobNew" if kind == "job_new" else "JobRemoved", signature="uos" if kind == "job_new" else "uoss", payload=(17, self.job, self.name) if kind == "job_new" else (17, self.job, self.name, "done"))
        elif kind == "unit_properties":
            packet.update(path=self.path, interface="org.freedesktop.DBus.Properties", member="PropertiesChanged", signature="sa{sv}as", payload=("org.freedesktop.systemd1.Unit", (self.api.StartupProperty("Job", "(uo)", (17, self.job)), self.api.StartupProperty("InvocationID", "ay", self.invocation)), ()))
        elif kind == "create_return" or kind == "create_error":
            packet.update(destination=":1.42", reply_cookie=42, path=None, interface=None, member=None, signature="o" if kind == "create_return" else "s", payload=(self.job,) if kind == "create_return" else (), error_name=None if kind == "create_return" else "org.freedesktop.systemd1.UnitExists", error_body_redacted=kind == "create_error")
        packet.update(changes)
        return self.api.parse_startup_envelope(packet)

    def advance(self, state: object, *envelopes: object) -> object:
        for envelope in envelopes:
            state = self.api.advance_startup_correlation(state, envelope, now=2.0)
        return state

    def correlated(self, *, done: bool = False) -> object:
        kinds = ["unit_new", "job_new", "unit_properties", "create_return"]
        if done:
            kinds.append("job_removed")
        return self.advance(self.state(), *(self.envelope(kind, index) for index, kind in enumerate(kinds, 1)))

    def assert_refuses(self, callback: object) -> None:
        with self.assertRaises(self.api.EnforcementUnavailable) as caught:
            callback()
        self.assertEqual(str(caught.exception), "Invalid bounded startup correlation evidence")

    def test_public_fields_required_frozen_slots(self) -> None:
        for name in PUBLIC_CLASSES:
            with self.subTest(name=name):
                cls = getattr(self.api, name)
                self.assertTrue(cls.__dataclass_params__.frozen)
                self.assertTrue(hasattr(cls, "__slots__"))
                self.assertTrue(all(parameter.default is inspect.Parameter.empty for parameter in inspect.signature(cls).parameters.values()))
        self.assertFalse(hasattr(self.context, "__dict__"))

    def test_subtype_construction_refuses(self) -> None:
        class Derived(self.api.StartupBusContext):
            pass
        self.assert_refuses(lambda: Derived(1, "a" * 32, ":1.7", ":1.42", 1000))

    def test_context_boundaries(self) -> None:
        for changes in ({"version": True}, {"bus_id": "A" * 32}, {"manager_unique_name": "org.systemd"}, {"client_unique_name": ":1.7"}, {"manager_uid": True}, {"manager_uid": 2**32 - 1}):
            with self.subTest(changes=tuple(changes)):
                self.assert_refuses(lambda: replace(self.context, **changes))

    def test_metadata_context_independent_hash(self) -> None:
        other = replace(self.context, manager_unique_name=":1.9", client_unique_name=":1.10")
        first = self.api.new_startup_correlation(self.context, self.metadata, deadline=10.0, now=1.0)
        second = self.api.new_startup_correlation(other, self.metadata, deadline=10.0, now=1.0)
        self.assertEqual(first.redacted_request_metadata_sha256, second.redacted_request_metadata_sha256)
        self.assertNotEqual(first.context, second.context)

    def test_metadata_sha_domain_projection(self) -> None:
        projection = {"domain": "truememory.startup.redacted_create_structure.v1", "version": 1, "path": "/org/freedesktop/systemd1", "interface": "org.freedesktop.systemd1.Manager", "member": "StartTransientUnit", "signature": "ssa(sv)a(sa(sv))", "unit_name": self.name, "unit_path": self.path, "job_mode": "fail", "property_signatures": self.metadata.property_signatures, "environment_names": self.metadata.environment_names, "auxiliary_units": ()}
        raw = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        self.assertEqual(self.api.redacted_create_metadata_sha256(self.metadata), hashlib.sha256(raw).hexdigest())
        self.assertNotIn("destination", projection)
        self.assertNotIn("environment_values", {field.name for field in fields(self.metadata)})

    def test_metadata_sort_uniqueness_and_values(self) -> None:
        for changes in ({"job_mode": "replace"}, {"auxiliary_units": (("x",),)}, {"property_signatures": (("X", "s"), ("X", "s"))}, {"property_signatures": (("Z", "s"), ("A", "s"))}, {"environment_names": ("PYTHONPATH", "LANG")}, {"environment_names": ("LANG", "LANG")}, {"environment_names": ("SECRET=value",)}, {"property_signatures": (("Environment", "s"),)}):
            with self.subTest(field=tuple(changes)):
                self.assert_refuses(lambda: replace(self.metadata, **changes))

    def test_unit_path_original_escaping(self) -> None:
        value = self.api.StartupCreateStructure(1, "1_a-b.service", "/org/freedesktop/systemd1/unit/_31_5fa_2db_2eservice", "fail", (), (), ())
        self.assertEqual(value.unit_name, "1_a-b.service")
        self.assert_refuses(lambda: replace(value, unit_path=self.path))

    def test_metadata_reflective_mutation_revalidated(self) -> None:
        object.__setattr__(self.metadata, "auxiliary_units", ["secret"])
        self.assert_refuses(lambda: self.api.redacted_create_metadata_sha256(self.metadata))

    def test_metadata_environment_names_not_shell_identifiers(self) -> None:
        metadata = replace(self.metadata, environment_names=("1-NAME", "unicode-é"))
        self.assertEqual(metadata.environment_names, ("1-NAME", "unicode-é"))

    def test_variants_exact_immutable(self) -> None:
        good = (("s", ""), ("b", False), ("u", 2**32 - 1), ("t", 2**64 - 1), ("ay", b"x"), ("as", ("a", "b")), ("(uo)", (0, "/")), ("(uo)", (17, "/other_object")))
        for signature, value in good:
            with self.subTest(signature=signature):
                self.assertEqual(self.api.StartupProperty("Extra", signature, value).value, value)
        bad = (("u", True), ("t", 2**64), ("ay", bytearray(b"x")), ("as", ["x"]), ("b", 1), ("s", object()), ("a{sv}", ()), ("(uo)", (17, "/bad-path")))
        for signature, value in bad:
            with self.subTest(signature=signature):
                self.assert_refuses(lambda: self.api.StartupProperty("Extra", signature, value))

    def test_all_native21_fields_and_invalidations(self) -> None:
        a = self.api
        changed = tuple(a.StartupProperty(name, signature, (17, self.job) if name == "Job" else self.invocation if signature == "ay" else "active" if signature == "s" else True if signature == "b" else 7) for name, signature in a._STARTUP_NATIVE_PROPERTIES)
        self.assertEqual(len(changed), 21)
        packet = self.envelope("unit_properties", 3, payload=("org.freedesktop.systemd1.Unit", changed, ("Conditions", "Asserts")))
        state = self.advance(self.state(), self.envelope("unit_new", 1), self.envelope("job_new", 2), packet, self.envelope("create_return", 4))
        self.assertEqual(state.phase, "invocation_observed")

    def test_invocation_empty_and_zero_valid_nonbinding(self) -> None:
        for value in (b"", bytes(16)):
            with self.subTest(length=len(value)):
                packet = self.envelope("unit_properties", 3, payload=("org.freedesktop.systemd1.Unit", (self.api.StartupProperty("Job", "(uo)", (17, self.job)), self.api.StartupProperty("InvocationID", "ay", value)), ()))
                state = self.advance(self.state(), self.envelope("unit_new", 1), self.envelope("job_new", 2), packet, self.envelope("create_return", 4))
                self.assertEqual(state.phase, "reply_bound_pending_invocation")
                self.assertIsNone(self.api.startup_correlation_binding(state))

    def test_invocation_invalid_lengths(self) -> None:
        for size in (1, 15, 17, 4097):
            with self.subTest(size=size):
                self.assert_refuses(lambda: self.api.StartupProperty("InvocationID", "ay", b"x" * size))

    def test_native_name_signature_constraints(self) -> None:
        for name, signature, value in (("Job", "s", "17"), ("InvocationID", "s", "secret"), ("ActiveState", "u", 0)):
            self.assert_refuses(lambda: self.api.StartupProperty(name, signature, value))

    def test_property_per_use_reflective_mutation(self) -> None:
        value = self.api.StartupProperty("Extra", "s", "valid")
        object.__setattr__(value, "value", ["secret"])
        self.assert_refuses(lambda: self.envelope("unit_properties", 1, payload=("org.freedesktop.systemd1.Unit", (value,), ())))

    def test_plain_dict_parser_only_precise_keys(self) -> None:
        packet = self.envelope("unit_new", 1)
        mapping = {field.name: getattr(packet, field.name) for field in fields(packet)}
        self.assertEqual(self.api.parse_startup_envelope(mapping), packet)
        for bad in ({**mapping, "environment_values": "secret"}, {key: value for key, value in mapping.items() if key != "version"}, list(mapping.items())):
            self.assert_refuses(lambda: self.api.parse_startup_envelope(bad))
        class DerivedDict(dict):
            pass
        self.assert_refuses(lambda: self.api.parse_startup_envelope(DerivedDict(mapping)))

    def test_all_seven_envelope_kinds(self) -> None:
        for kind in ("create_return", "create_error", "unit_new", "unit_removed", "job_new", "job_removed", "unit_properties"):
            with self.subTest(kind=kind):
                self.assertEqual(self.envelope(kind, 1).kind, kind)

    def test_envelope_strict_scalars_and_headers(self) -> None:
        for changes in ({"version": True}, {"serial": True}, {"ordinal": 0}, {"serial": 2**32}, {"ordinal": 2**64}, {"member": "JobNew"}, {"interface": "org.freedesktop.systemd1.Unit"}, {"signature": "s"}, {"reply_cookie": 1}, {"error_body_redacted": 0}, {"payload": [self.name, self.path]}):
            with self.subTest(field=tuple(changes)):
                self.assert_refuses(lambda: self.envelope("unit_new", 1, **changes))

    def test_create_reply_canonical_job_path(self) -> None:
        for path in ("/org/freedesktop/systemd1/job/017", "/org/freedesktop/systemd1/job/0", "/org/freedesktop/systemd1/job/4294967296", "/foreign/17"):
            self.assert_refuses(lambda: self.envelope("create_return", 1, payload=(path,)))

    def test_error_body_never_admitted(self) -> None:
        self.assert_refuses(lambda: self.envelope("create_error", 1, payload=("synthetic-secret",)))
        self.assert_refuses(lambda: self.envelope("create_error", 1, error_body_redacted=False))
        value = self.envelope("create_error", 1)
        self.assertEqual(value.payload, ())
        self.assertEqual(len(self.api.startup_envelope_evidence_sha256(value)), 64)

    def test_properties_duplicate_changed_and_invalidated(self) -> None:
        value = self.api.StartupProperty("Extra", "s", "x")
        for changed, invalidated in (((value, value), ()), ((), ("Job", "Job"))):
            self.assert_refuses(lambda: self.envelope("unit_properties", 1, payload=("org.freedesktop.systemd1.Unit", changed, invalidated)))

    def test_subtype_packet_not_selected_generic(self) -> None:
        self.assert_refuses(lambda: self.envelope("unit_properties", 1, payload=("org.freedesktop.systemd1.Service", (), ())))

    def test_evidence_digest_native_header_sensitive(self) -> None:
        first = self.envelope("unit_new", 1)
        second = replace(first, serial=200)
        self.assertNotEqual(self.api.startup_envelope_evidence_sha256(first), self.api.startup_envelope_evidence_sha256(second))
        self.assertEqual(self.api.startup_envelope_evidence_sha256(first), self.api.startup_envelope_evidence_sha256(first))

    def test_evidence_digest_tagged_bytes(self) -> None:
        packet = self.envelope("unit_properties", 1)
        projection = self.api._startup_envelope_projection(packet)
        self.assertEqual(projection["domain"], "truememory.startup.typed_evidence.v1")
        self.assertEqual(projection["payload"][1][1]["value"], {"type": "ay", "hex": self.invocation.hex()})

    def test_success_pending_then_done_binding_fields(self) -> None:
        state = self.correlated()
        binding = self.api.startup_correlation_binding(state)
        self.assertEqual(state.phase, "invocation_observed")
        self.assertEqual(binding.context, self.context)
        self.assertEqual((binding.job_id, binding.job_path, binding.invocation_id), (17, self.job, self.invocation))
        self.assertIsNone(binding.job_result)
        state = self.advance(state, self.envelope("job_removed", 5))
        binding = self.api.startup_correlation_binding(state)
        self.assertEqual(state.phase, "correlated_done")
        self.assertEqual(binding.job_result, "done")
        for field in ("reply_evidence_sha256", "job_new_evidence_sha256", "joint_evidence_sha256", "unit_new_evidence_sha256", "job_removed_evidence_sha256"):
            self.assertEqual(len(getattr(binding, field)), 64)
        self.assertFalse(hasattr(binding, "pid"))

    def test_all_early_packets_before_cookie(self) -> None:
        state = self.advance(self.state(cookie=False), *(self.envelope(kind, index) for index, kind in enumerate(("unit_new", "job_new", "unit_properties", "job_removed", "create_return"), 1)))
        self.assertEqual(state.phase, "attempted_unknown")
        self.assertIsNone(self.api.startup_correlation_binding(state))
        state = self.api.bind_startup_request_cookie(state, cookie=42, now=2.1)
        self.assertEqual(state.phase, "correlated_done")

    def test_send_attempt_exactly_once(self) -> None:
        state = self.state(cookie=False)
        self.assert_refuses(lambda: self.api.mark_startup_send_attempt(state, now=2.0))
        self.assertEqual(state.phase, "attempted_unknown")

    def test_cookie_repeat_and_conflict(self) -> None:
        state = self.api.bind_startup_request_cookie(self.state(), cookie=42, now=2.0)
        self.assertEqual(state.request_cookie, 42)
        state = self.api.bind_startup_request_cookie(state, cookie=43, now=2.1)
        self.assertEqual((state.phase, state.reason), ("unknown", "conflicting_cookie"))

    def test_cookie_and_manager_serial_independent(self) -> None:
        state = self.advance(self.state(), self.envelope("unit_new", 1, serial=42), self.envelope("job_new", 2, serial=1000), self.envelope("unit_properties", 3, serial=1500), self.envelope("create_return", 4, serial=2000))
        self.assertEqual(state.phase, "invocation_observed")

    def test_preexisting_unit_does_not_bind(self) -> None:
        state = self.api.new_startup_correlation(self.context, self.metadata, deadline=10.0, now=1.0)
        state = self.advance(state, self.envelope("unit_new", 1))
        self.assertEqual((state.phase, state.reason), ("unknown", "preexisting_unit"))
        self.assertIsNone(self.api.startup_correlation_binding(state))

    def test_missing_unit_new_is_not_fresh(self) -> None:
        state = self.advance(self.state(), self.envelope("job_new", 1))
        self.assertEqual(state.phase, "unknown")

    def test_replacement_unit_new_and_removal_unknown(self) -> None:
        for kind in ("unit_new", "unit_removed"):
            with self.subTest(kind=kind):
                state = self.advance(self.correlated(), self.envelope(kind, 5))
                self.assertEqual((state.phase, state.reason), ("unknown", "conflicting_unit"))
                self.assertIsNone(self.api.startup_correlation_binding(state))

    def test_foreign_sender_and_destination_unknown(self) -> None:
        for packet, reason in ((self.envelope("unit_new", 1, sender=":1.99"), "manager_owner_changed"), (self.envelope("create_return", 1, destination=":1.99"), "conflicting_reply")):
            state = self.advance(self.state(), packet)
            self.assertEqual((state.phase, state.reason), ("unknown", reason))

    def test_wrong_reply_cookie_unknown(self) -> None:
        state = self.advance(self.state(), self.envelope("create_return", 1, reply_cookie=43))
        self.assertEqual(state.reason, "conflicting_reply")

    def test_local_ordinal_gap_unknown(self) -> None:
        state = self.advance(self.state(), self.envelope("unit_new", 2))
        self.assertEqual(state.reason, "transport_gap")

    def test_native_serial_replay_even_identical_payload(self) -> None:
        state = self.advance(self.state(), self.envelope("unit_new", 1, serial=101))
        state = self.advance(state, self.envelope("unit_new", 2, serial=101))
        self.assertEqual(state.reason, "replayed_native_serial")
        self.assertEqual(len(state.envelopes), 1)

    def test_separate_job_and_invocation_messages_do_not_join(self) -> None:
        a = self.api
        first = self.envelope("unit_properties", 3, payload=("org.freedesktop.systemd1.Unit", (a.StartupProperty("Job", "(uo)", (17, self.job)),), ()))
        second = self.envelope("unit_properties", 4, payload=("org.freedesktop.systemd1.Unit", (a.StartupProperty("InvocationID", "ay", self.invocation),), ()))
        state = self.advance(self.state(), self.envelope("unit_new", 1), self.envelope("job_new", 2), first, second, self.envelope("create_return", 5))
        self.assertEqual(state.phase, "reply_bound_pending_invocation")
        self.assertIsNone(a.startup_correlation_binding(state))

    def test_invalidated_required_property_nonbinding(self) -> None:
        packet = self.envelope("unit_properties", 3, payload=("org.freedesktop.systemd1.Unit", (self.api.StartupProperty("Job", "(uo)", (17, self.job)),), ("InvocationID",)))
        state = self.advance(self.state(), self.envelope("unit_new", 1), self.envelope("job_new", 2), packet, self.envelope("create_return", 4))
        self.assertIsNone(self.api.startup_correlation_binding(state))

    def test_changed_joint_with_invalidation_is_nonbinding(self) -> None:
        packet = self.envelope("unit_properties", 3)
        packet = replace(packet, payload=(packet.payload[0], packet.payload[1], ("Job",)))
        state = self.advance(self.state(), self.envelope("unit_new", 1), self.envelope("job_new", 2), packet, self.envelope("create_return", 4))
        self.assertEqual(state.phase, "reply_bound_pending_invocation")
        self.assertIsNone(self.api.startup_correlation_binding(state))

    def test_done_without_joint_stays_pending(self) -> None:
        state = self.advance(self.state(), self.envelope("unit_new", 1), self.envelope("job_new", 2), self.envelope("job_removed", 3), self.envelope("create_return", 4))
        self.assertEqual(state.phase, "reply_bound_pending_invocation")
        self.assertIsNone(self.api.startup_correlation_binding(state))

    def test_positive_job_conflict_unknown(self) -> None:
        other = "/org/freedesktop/systemd1/job/18"
        packet = self.envelope("unit_properties", 5, payload=("org.freedesktop.systemd1.Unit", (self.api.StartupProperty("Job", "(uo)", (18, other)), self.api.StartupProperty("InvocationID", "ay", self.invocation)), ()))
        self.assertEqual(self.advance(self.correlated(), packet).reason, "conflicting_job")

    def test_invocation_replacement_unknown(self) -> None:
        packet = self.envelope("unit_properties", 5, payload=("org.freedesktop.systemd1.Unit", (self.api.StartupProperty("Job", "(uo)", (17, self.job)), self.api.StartupProperty("InvocationID", "ay", b"z" * 16)), ()))
        self.assertEqual(self.advance(self.correlated(), packet).reason, "ambiguous_invocation")

    def test_later_zero_job_preserves_historical_joint_only(self) -> None:
        packet = self.envelope("unit_properties", 5, payload=("org.freedesktop.systemd1.Unit", (self.api.StartupProperty("Job", "(uo)", (0, "/")),), ()))
        state = self.advance(self.correlated(), packet)
        self.assertEqual(state.phase, "invocation_observed")
        self.assertEqual(self.api.startup_correlation_binding(state).invocation_id, self.invocation)

    def test_manager_error_and_non_done_are_unknown(self) -> None:
        self.assertEqual(self.advance(self.state(), self.envelope("create_error", 1)).reason, "manager_error_unknown")
        packet = self.envelope("job_removed", 5, payload=(17, self.job, self.name, "failed"))
        self.assertEqual(self.advance(self.correlated(), packet).reason, "job_outcome_non_done")

    def test_time_exact_nonnegative_finite_floats(self) -> None:
        for value in (1, True, -1.0, float("inf"), float("nan")):
            self.assert_refuses(lambda: self.api.new_startup_correlation(self.context, self.metadata, deadline=10.0, now=value))
            self.assert_refuses(lambda: self.api.expire_startup_correlation(self.state(), now=value))
        self.assert_refuses(lambda: self.api.new_startup_correlation(self.context, self.metadata, deadline=1.0, now=1.0))

    def test_time_backward_refusal_preserves_state(self) -> None:
        state = self.state()
        self.assert_refuses(lambda: self.api.expire_startup_correlation(state, now=1.0))
        self.assertEqual(state.last_now, 1.2)

    def test_expire_at_exact_original_deadline(self) -> None:
        prepared = self.api.new_startup_correlation(self.context, self.metadata, deadline=10.0, now=1.0)
        for state, reason in ((prepared, "expired_unsent"), (self.correlated(), "expired_unknown")):
            expired = self.api.expire_startup_correlation(state, now=10.0)
            self.assertEqual(expired.reason, reason)
            self.assertEqual(expired.original_deadline, 10.0)
            self.assertIsNone(self.api.startup_correlation_binding(expired))

    def test_sticky_terminal_state_no_regain(self) -> None:
        state = self.api.invalidate_startup_correlation(self.correlated(), reason="transport_disconnected", now=3.0)
        prior = state.envelopes
        state = self.api.bind_startup_request_cookie(state, cookie=42, now=4.0)
        state = self.api.advance_startup_correlation(state, self.envelope("job_removed", 5), now=5.0)
        self.assertEqual(state.reason, "transport_disconnected")
        self.assertEqual(state.envelopes, prior)
        self.assertIsNone(self.api.startup_correlation_binding(state))

    def test_fixed_reasons_only_no_secret_echo(self) -> None:
        self.assert_refuses(lambda: self.api.invalidate_startup_correlation(self.state(), reason="synthetic-secret", now=2.0))
        self.assert_refuses(lambda: self.envelope("unit_new", 1, sender="synthetic-secret"))

    def test_count_overflow_retains_preceding_stream(self) -> None:
        state = self.state()
        for index in range(1, 65):
            state = self.advance(state, self.envelope("unit_properties", index, payload=("org.freedesktop.systemd1.Unit", (), ())))
        self.assertEqual(len(state.envelopes), 64)
        previous = state.envelopes
        state = self.advance(state, self.envelope("unit_properties", 65, payload=("org.freedesktop.systemd1.Unit", (), ())))
        self.assertEqual(state.reason, "evidence_overflow")
        self.assertIs(state.envelopes, previous)

    def test_single_canonical_overflow_does_not_retain_or_hash(self) -> None:
        properties = tuple(self.api.StartupProperty("Field" + str(index), "s", "x" * 1024) for index in range(32))
        packet = self.envelope("unit_properties", 1, payload=("org.freedesktop.systemd1.Unit", properties, ()))
        state = self.advance(self.state(), packet)
        self.assertEqual(state.reason, "evidence_overflow")
        self.assertEqual(state.envelopes, ())
        self.assertEqual(state.canonical_evidence_bytes, 0)
        self.assert_refuses(lambda: self.api.startup_envelope_evidence_sha256(packet))

    def test_aggregate_canonical_overflow_no_eviction(self) -> None:
        state = self.state()
        previous = state
        for index in range(1, 20):
            packet = self.envelope("unit_properties", index, payload=("org.freedesktop.systemd1.Unit", (self.api.StartupProperty("Large", "s", "x" * 1024),), ()))
            previous = state
            state = self.advance(state, packet)
            if state.phase == "unknown":
                break
        if state.phase != "unknown":
            for index in range(20, 65):
                previous = state
                state = self.advance(state, self.envelope("unit_properties", index, payload=("org.freedesktop.systemd1.Unit", (self.api.StartupProperty("Large", "s", "x" * 1024),), ())))
                if state.phase == "unknown":
                    break
        self.assertEqual(state.reason, "evidence_overflow")
        self.assertEqual(state.envelopes, previous.envelopes)
        self.assertLessEqual(state.canonical_evidence_bytes, 65536)

    def test_precopy_container_and_utf8_bounds(self) -> None:
        for value in ("x" * 1025, "é" * 513, "\ud800"):
            self.assert_refuses(lambda: self.api.StartupProperty("Extra", "s", value))
        self.assert_refuses(lambda: self.api.StartupProperty("Extra", "as", tuple("x" for _ in range(65))))
        self.assert_refuses(lambda: self.envelope("unit_properties", 1, payload=("org.freedesktop.systemd1.Unit", tuple(self.api.StartupProperty("F" + str(index), "s", "") for index in range(65)), ())))

    def test_state_phase_hash_size_mutations_refuse(self) -> None:
        for field, value in (("phase", "invocation_observed"), ("canonical_evidence_bytes", 1), ("redacted_request_metadata_sha256", "b" * 64), ("request_cookie", True), ("send_attempted", 1)):
            state = self.state()
            object.__setattr__(state, field, value)
            self.assert_refuses(lambda: self.api.startup_correlation_binding(state))

    def test_binding_mutation_revalidated(self) -> None:
        binding = self.api.startup_correlation_binding(self.correlated())
        object.__setattr__(binding, "invocation_id", bytes(16))
        self.assert_refuses(lambda: self.api._startup_validate_binding(binding))

    def test_pure_diagnostics_do_not_authenticate_context(self) -> None:
        supplied = replace(self.context, bus_id="b" * 32)
        state = self.api.new_startup_correlation(supplied, self.metadata, deadline=10.0, now=1.0)
        self.assertEqual(state.context.bus_id, "b" * 32)
        self.assertFalse(hasattr(state, "owner"))
        self.assertFalse(hasattr(state, "pidfd"))

    def test_selected_graph_no_native_or_clock_calls(self) -> None:
        tree = ast.parse(SOURCE.read_text())
        chosen = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (getattr(node, "name", "").startswith("_startup_") or getattr(node, "name", "") in PUBLIC_CLASSES + PUBLIC_FUNCTIONS)]
        forbidden = {"time", "os", "sys", "Path", "ctypes", "importlib", "socket", "signal", "select", "fcntl", "subprocess", "threading", "open", "exec", "eval", "__import__", "get_linux_adapter", "launch_owned_linux", "capture_owned_exit", "stop_owned_exact", "terminate_owned_exact"}
        referenced = {node.id for root in chosen for node in ast.walk(root) if isinstance(node, ast.Name)}
        self.assertEqual(referenced.intersection(forbidden), set())


if __name__ == "__main__":
    unittest.main()
