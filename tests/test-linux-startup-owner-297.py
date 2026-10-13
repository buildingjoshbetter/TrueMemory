"""Synthetic S1 retention controls; no application or native launcher import."""
from __future__ import annotations

import ast
from dataclasses import dataclass, fields, FrozenInstanceError, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys
import threading
import types
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "truememory_memory_bootstrap.py"
S0_CLASSES = ("StartupCreateStructure", "StartupBusContext", "StartupProperty", "StartupMessageEnvelope", "StartupCorrelationState", "StartupInvocationBinding")
S0_FUNCTIONS = ("redacted_create_metadata_sha256", "new_startup_correlation", "mark_startup_send_attempt", "bind_startup_request_cookie", "parse_startup_envelope", "startup_envelope_evidence_sha256", "advance_startup_correlation", "invalidate_startup_correlation", "expire_startup_correlation", "startup_correlation_binding")
S1_CLASSES = ("StartupRetentionRequest", "StartupRetentionObservation", "StartupRetentionOwner", "StartupRetentionSnapshot")
S1_FUNCTIONS = ("prepare_startup_retention", "startup_retention_snapshot", "publish_startup_retention", "invalidate_startup_retention", "cancel_unsent_startup_retention", "release_unsent_startup_retention")


def load_pure_source() -> types.ModuleType:
    selected = []
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) and target.id.startswith(("_STARTUP_", "_RETENTION_")) for target in node.targets):
            selected.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id.startswith("_RETENTION_"):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in S0_CLASSES + S1_CLASSES + ("EnforcementUnavailable", "_RetentionSeal", "_RetentionRecord"):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and (node.name in S0_FUNCTIONS + S1_FUNCTIONS or node.name.startswith(("_startup_", "_retention_"))):
            selected.append(node)
    module = types.ModuleType("synthetic_startup_retention_297")
    module.__dict__.update({"dataclass": dataclass, "hashlib": hashlib, "json": json, "math": math,
                            "re": re, "os": types.SimpleNamespace(getpid=lambda: 1000), "threading": threading})
    sys.modules[module.__name__] = module
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(SOURCE), "exec"), module.__dict__)
    return module


class StartupRetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.api = load_pure_source()
        self.context = self.api.StartupBusContext(1, "a" * 32, ":1.7", ":1.42", 1000)
        self.request = self.make_request(1)

    def make_request(self, number: int) -> object:
        a = self.api
        generation = f"{number:032x}"
        metadata = []
        for stage in ("slice", "service"):
            name = ("app-truememory" if stage == "slice" else "truememory-memory") + generation + "." + stage
            metadata.append(a.StartupCreateStructure(1, name, a._startup_unit_path(name), "fail",
                                                    (("Environment", "as"), ("MemoryMax", "t")), ("LANG", "PYTHONPATH"), ()))
        return a.StartupRetentionRequest(1, generation, 268435456, 0, *metadata)

    def prepare(self) -> object:
        return self.api.prepare_startup_retention(self.request, deadline=10.0, now=1.0)

    def observation(self, stage: str = "slice", state: object = None) -> object:
        return self.api.StartupRetentionObservation(1, stage, self.context, state, None)

    def state(self, *, stage: str = "slice", attempted: bool = False, cookie: bool = False) -> object:
        a = self.api
        state = a.new_startup_correlation(self.context, getattr(self.request, stage + "_metadata"), deadline=10.0, now=1.1)
        if attempted:
            state = a.mark_startup_send_attempt(state, now=1.2)
        return a.bind_startup_request_cookie(state, cookie=42, now=1.3) if cookie else state

    def envelope(self, state: object, ordinal: int = 1) -> object:
        return self.api.StartupMessageEnvelope(1, ordinal, "unit_new", ":1.7", None, ordinal + 100, None,
                                               "/org/freedesktop/systemd1", "org.freedesktop.systemd1.Manager", "UnitNew", "so",
                                               (state.metadata.unit_name, state.metadata.unit_path), None, False)

    def assert_refuses(self, action: object, message: str = "Invalid startup retention input") -> None:
        with self.assertRaises(self.api.EnforcementUnavailable) as caught:
            action()
        self.assertEqual(str(caught.exception), message)
        self.assertTrue(caught.exception.__suppress_context__)

    def test_public_records_frozen_slots_required_fields(self) -> None:
        expected = {
            "StartupRetentionRequest": ("version", "generation", "requested_cap_bytes", "requested_swap_bytes", "slice_metadata", "service_metadata"),
            "StartupRetentionObservation": ("version", "stage", "context", "correlation", "reason"),
            "StartupRetentionOwner": ("version", "owner_id", "creator_pid"),
            "StartupRetentionSnapshot": ("version", "owner", "request", "original_deadline", "last_now", "operation_sequence", "phase", "slice_context", "slice_correlation", "service_context", "service_correlation", "slice_attempted", "service_attempted", "reason"),
        }
        for name, order in expected.items():
            with self.subTest(name=name):
                cls = getattr(self.api, name)
                self.assertEqual(tuple(field.name for field in fields(cls)), order)
                self.assertTrue(cls.__dataclass_params__.frozen)
                self.assertTrue(hasattr(cls, "__slots__"))
                self.assertTrue(all(p.default is inspect.Parameter.empty for p in inspect.signature(cls).parameters.values()))
        self.assertFalse(hasattr(self.request, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            self.request.version = 2

    def test_constructor_subtypes_refuse(self) -> None:
        class Derived(self.api.StartupRetentionRequest):
            pass
        self.assert_refuses(lambda: Derived(*(getattr(self.request, f.name) for f in fields(self.request))))

    def test_omitted_extra_constructor_fields_are_typeerror(self) -> None:
        with self.assertRaises(TypeError):
            self.api.StartupRetentionOwner()
        with self.assertRaises(TypeError):
            self.api.StartupRetentionOwner(1, 1, 1000, extra=True)

    def test_request_scalar_boundaries(self) -> None:
        for changes in ({"version": True}, {"generation": "A" * 32}, {"generation": "a" * 31},
                        {"requested_cap_bytes": True}, {"requested_cap_bytes": 0}, {"requested_cap_bytes": 2**63},
                        {"requested_swap_bytes": True}, {"requested_swap_bytes": 1}):
            with self.subTest(changes=changes):
                self.assert_refuses(lambda: replace(self.request, **changes))
        self.assertEqual(replace(self.request, requested_cap_bytes=2**63 - 1).requested_cap_bytes, 2**63 - 1)

    def test_custom_scalar_equality_types_refuse(self) -> None:
        class Custom(str):
            pass
        self.assert_refuses(lambda: replace(self.request, generation=Custom(self.request.generation)))
        self.assert_refuses(lambda: self.api.StartupRetentionObservation(1, Custom("slice"), self.context, None, None))

    def test_request_stage_metadata_matches_generation(self) -> None:
        self.assert_refuses(lambda: replace(self.request, slice_metadata=self.request.service_metadata))
        self.assert_refuses(lambda: replace(self.request, service_metadata=self.make_request(2).service_metadata))

    def test_canonical_metadata_bound_is_enforced(self) -> None:
        large = replace(self.request.service_metadata, environment_names=tuple(f"N{n:03d}" + "x" * 240 for n in range(100)))
        self.assert_refuses(lambda: replace(self.request, service_metadata=large))

    def test_request_hash_names_only_domain(self) -> None:
        a = self.api
        projection = {"domain": "truememory.startup.retention_request.v1", "version": 1,
                      "generation": self.request.generation, "requested_cap_bytes": 268435456, "requested_swap_bytes": 0,
                      "slice_metadata_sha256": a.redacted_create_metadata_sha256(self.request.slice_metadata),
                      "service_metadata_sha256": a.redacted_create_metadata_sha256(self.request.service_metadata)}
        raw = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        self.assertEqual(a._retention_request_sha256(self.request), hashlib.sha256(raw).hexdigest())
        self.assertNotIn("environment_values", {f.name for f in fields(self.request)})

    def test_prepare_and_read_initial_snapshot(self) -> None:
        owner = self.prepare()
        row = self.api.startup_retention_snapshot(owner, now=1.5)
        self.assertIs(row.owner, owner)
        self.assertEqual((row.phase, row.operation_sequence, row.last_now, row.original_deadline), ("prepared", 0, 1.5, 10.0))
        self.assertIsNot(row.request, self.request)
        self.assertIsNone(row.reason)

    def test_prepare_finite_exact_float_times(self) -> None:
        for changes in ({"now": 1}, {"now": True}, {"now": -1.0}, {"now": math.nan},
                        {"deadline": 10}, {"deadline": math.inf}, {"deadline": 1.0}, {"deadline": 0.0}):
            kwargs = dict(deadline=10.0, now=1.0)
            kwargs.update(changes)
            with self.subTest(changes=tuple(changes)):
                self.assert_refuses(lambda: self.api.prepare_startup_retention(self.request, **kwargs))
                self.assertEqual(self.api._RETENTION_RECORDS, {})

    def test_registry_capacity_and_duplicate_generation(self) -> None:
        owner = self.prepare()
        self.assert_refuses(lambda: self.prepare(), "Startup retention generation already retained")
        for number in range(2, 17):
            self.api.prepare_startup_retention(self.make_request(number), deadline=10.0, now=1.0)
        self.assert_refuses(lambda: self.api.prepare_startup_retention(self.make_request(17), deadline=10.0, now=1.0), "Startup retention registry full")
        self.assertEqual(len(self.api._RETENTION_RECORDS), 16)
        self.assertIs(self.api._RETENTION_RECORDS[owner.owner_id].seal.owner, owner)

    def test_owner_ids_do_not_reuse_released_slot(self) -> None:
        owner = self.prepare()
        self.api.cancel_unsent_startup_retention(owner, expected_sequence=0, now=2.0)
        self.api.release_unsent_startup_retention(owner, expected_sequence=1, now=2.1)
        other = self.prepare()
        self.assertEqual(other.owner_id, owner.owner_id + 1)

    def test_identifier_exhaustion_refuses_unchanged(self) -> None:
        self.api._RETENTION_NEXT_OWNER_ID = 2**64 - 1
        self.assert_refuses(lambda: self.prepare(), "Startup retention identifier exhausted")
        self.assertEqual(self.api._RETENTION_RECORDS, {})

    def test_clone_and_unregistered_owner_are_stale(self) -> None:
        owner = self.prepare()
        clone = replace(owner)
        self.assert_refuses(lambda: self.api.startup_retention_snapshot(clone, now=2.0), "Startup retention owner is stale")
        self.assert_refuses(lambda: self.api.startup_retention_snapshot(self.api.StartupRetentionOwner(1, 99, 1000), now=2.0), "Startup retention owner is stale")

    def test_registered_handle_scalar_mutation_refuses(self) -> None:
        owner = self.prepare()
        object.__setattr__(owner, "owner_id", owner.owner_id + 1)
        self.assert_refuses(lambda: self.api.startup_retention_snapshot(owner, now=2.0), "Startup retention invariant changed")

    def test_request_and_snapshot_mutations_do_not_renew_private_seal(self) -> None:
        owner = self.prepare()
        row = self.api.startup_retention_snapshot(owner, now=2.0)
        object.__setattr__(self.request, "requested_cap_bytes", 1)
        object.__setattr__(row, "original_deadline", 100.0)
        object.__setattr__(row.request, "requested_cap_bytes", 2)
        actual = self.api.startup_retention_snapshot(owner, now=10.0)
        self.assertEqual((actual.original_deadline, actual.request.requested_cap_bytes, actual.phase), (10.0, 268435456, "expired_unsent"))

    def test_context_only_cross_stage_agreement(self) -> None:
        owner = self.prepare()
        first = self.api.publish_startup_retention(owner, self.observation(), expected_sequence=0, now=2.0)
        second = self.api.publish_startup_retention(owner, self.observation("service"), expected_sequence=1, now=2.1)
        self.assertEqual((first.phase, second.operation_sequence), ("local_observed", 2))
        self.assertEqual(second.slice_context, second.service_context)
        self.assertIsNot(second.slice_context, self.context)

    def test_conflicting_context_refuses_without_time_commit(self) -> None:
        owner = self.prepare()
        self.api.publish_startup_retention(owner, self.observation(), expected_sequence=0, now=2.0)
        before = self.api._RETENTION_RECORDS[owner.owner_id]
        other = replace(self.context, client_unique_name=":1.99")
        observation = self.api.StartupRetentionObservation(1, "service", other, None, None)
        self.assert_refuses(lambda: self.api.publish_startup_retention(owner, observation, expected_sequence=1, now=3.0))
        self.assertIs(self.api._RETENTION_RECORDS[owner.owner_id], before)

    def test_observation_only_allowed_shapes(self) -> None:
        for values in ((1, "slice", None, None, None), (True, "slice", self.context, None, None),
                       (1, "other", self.context, None, None), (1, "slice", self.context, None, "transport_gap"),
                       (1, "slice", None, None, "expired_unsent")):
            with self.subTest(values=values[:2]):
                self.assert_refuses(lambda: self.api.StartupRetentionObservation(*values))

    def test_matching_correlation_attempt_latches(self) -> None:
        owner = self.prepare()
        state = self.state(attempted=True, cookie=True)
        row = self.api.publish_startup_retention(owner, self.observation(state=state), expected_sequence=0, now=2.0)
        self.assertTrue(row.slice_attempted)
        self.assertFalse(row.service_attempted)
        self.assertIsNot(row.slice_correlation, state)
        self.assert_refuses(lambda: self.api.cancel_unsent_startup_retention(owner, expected_sequence=1, now=3.0), "Startup retention has an attempted send")

    def test_correlation_wrong_metadata_deadline_future_refuse(self) -> None:
        owner = self.prepare()
        states = (self.state(stage="service"), replace(self.state(), original_deadline=11.0), replace(self.state(), last_now=3.0))
        for state in states:
            with self.subTest(state=state.metadata.unit_name):
                self.assert_refuses(lambda: self.api.publish_startup_retention(owner, self.observation(state=state), expected_sequence=0, now=2.0))

    def test_prefix_cookie_time_and_attempt_rollback_refuse(self) -> None:
        owner = self.prepare()
        state = self.state(attempted=True, cookie=True)
        progressed = self.api.advance_startup_correlation(state, self.envelope(state), now=2.0)
        self.api.publish_startup_retention(owner, self.observation(state=progressed), expected_sequence=0, now=2.1)
        candidates = (state, replace(progressed, request_cookie=43), self.state())
        for candidate in candidates:
            with self.subTest(cookie=candidate.request_cookie):
                self.assert_refuses(lambda: self.api.publish_startup_retention(owner, self.observation(state=candidate), expected_sequence=1, now=3.0))

    def test_copied_evidence_nested_graph_is_independent(self) -> None:
        owner = self.prepare()
        state = self.state(attempted=True)
        state = self.api.advance_startup_correlation(state, self.envelope(state), now=2.0)
        row = self.api.publish_startup_retention(owner, self.observation(state=state), expected_sequence=0, now=2.1)
        self.assertIsNot(row.slice_correlation.envelopes[0], state.envelopes[0])
        object.__setattr__(row.slice_correlation.envelopes[0], "sender", ":1.99")
        actual = self.api.startup_retention_snapshot(owner, now=2.2)
        self.assertEqual(actual.slice_correlation.envelopes[0].sender, ":1.7")

    def test_local_unknown_reason_sticky_through_cancel_release(self) -> None:
        owner = self.prepare()
        row = self.api.publish_startup_retention(owner, self.api.StartupRetentionObservation(1, "slice", None, None, "transport_gap"), expected_sequence=0, now=2.0)
        self.assertEqual((row.phase, row.reason), ("terminal_unknown", "transport_gap"))
        row = self.api.cancel_unsent_startup_retention(owner, expected_sequence=1, now=11.0)
        row = self.api.release_unsent_startup_retention(owner, expected_sequence=2, now=12.0)
        self.assertEqual((row.phase, row.reason, row.operation_sequence), ("released_unsent", "transport_gap", 3))

    def test_s0_unknown_retained_attempt_cannot_release(self) -> None:
        owner = self.prepare()
        state = self.api.invalidate_startup_correlation(self.state(attempted=True), reason="transport_disconnected", now=2.0)
        row = self.api.publish_startup_retention(owner, self.observation(state=state), expected_sequence=0, now=2.1)
        self.assertEqual((row.phase, row.reason), ("terminal_unknown", "transport_disconnected"))
        self.assert_refuses(lambda: self.api.release_unsent_startup_retention(owner, expected_sequence=1, now=3.0), "Startup retention has an attempted send")
        self.assertEqual(len(self.api._RETENTION_RECORDS), 1)

    def test_invalidation_closed_reasons(self) -> None:
        owner = self.prepare()
        for reason in ("private-input-value", "expired_unknown", "cancelled_unsent", None):
            with self.subTest(reason=reason):
                self.assert_refuses(lambda: self.api.invalidate_startup_retention(owner, reason=reason, expected_sequence=0, now=2.0))
        row = self.api.invalidate_startup_retention(owner, reason="manager_owner_changed", expected_sequence=0, now=2.0)
        self.assertEqual((row.operation_sequence, row.reason), (1, "manager_owner_changed"))

    def test_publish_invalidate_late_preserve_entire_record(self) -> None:
        owner = self.prepare()
        before = self.api._RETENTION_RECORDS[owner.owner_id]
        self.assert_refuses(lambda: self.api.publish_startup_retention(owner, self.observation(), expected_sequence=0, now=10.0), "Startup retention is terminal")
        self.assert_refuses(lambda: self.api.invalidate_startup_retention(owner, reason="transport_gap", expected_sequence=0, now=11.0), "Startup retention is terminal")
        self.assertIs(self.api._RETENTION_RECORDS[owner.owner_id], before)

    def test_snapshot_inclusive_expiry_commits_once(self) -> None:
        owner = self.prepare()
        row = self.api.startup_retention_snapshot(owner, now=10.0)
        self.assertEqual((row.phase, row.reason, row.operation_sequence), ("expired_unsent", "expired_unsent", 1))
        again = self.api.startup_retention_snapshot(owner, now=11.0)
        self.assertEqual((again.phase, again.reason, again.operation_sequence, again.last_now), (row.phase, row.reason, 1, 11.0))

    def test_attempted_snapshot_expiry_retains_capacity(self) -> None:
        owner = self.prepare()
        self.api.publish_startup_retention(owner, self.observation(state=self.state(attempted=True)), expected_sequence=0, now=2.0)
        row = self.api.startup_retention_snapshot(owner, now=10.0)
        self.assertEqual((row.phase, row.reason, row.operation_sequence), ("expired_unknown", "expired_unknown", 2))
        self.assertEqual(len(self.api._RETENTION_RECORDS), 1)
        self.assert_refuses(lambda: self.api.cancel_unsent_startup_retention(owner, expected_sequence=2, now=11.0), "Startup retention has an attempted send")

    def test_cancel_after_deadline_does_not_first_expire(self) -> None:
        owner = self.prepare()
        row = self.api.cancel_unsent_startup_retention(owner, expected_sequence=0, now=11.0)
        self.assertEqual((row.phase, row.reason, row.operation_sequence), ("cancelled_unsent", "cancelled_unsent", 1))
        before = self.api._RETENTION_RECORDS[owner.owner_id]
        self.assert_refuses(lambda: self.api.cancel_unsent_startup_retention(owner, expected_sequence=1, now=12.0), "Startup retention is terminal")
        self.assertIs(self.api._RETENTION_RECORDS[owner.owner_id], before)

    def test_release_expired_unsent_once_without_tombstone(self) -> None:
        owner = self.prepare()
        self.api.startup_retention_snapshot(owner, now=10.0)
        row = self.api.release_unsent_startup_retention(owner, expected_sequence=1, now=11.0)
        self.assertEqual((row.phase, row.reason, row.operation_sequence), ("released_unsent", "expired_unsent", 2))
        self.assertEqual(self.api._RETENTION_RECORDS, {})
        self.assert_refuses(lambda: self.api.release_unsent_startup_retention(owner, expected_sequence=2, now=12.0), "Startup retention owner is stale")

    def test_release_open_refuses_without_mutation(self) -> None:
        owner = self.prepare()
        before = self.api._RETENTION_RECORDS[owner.owner_id]
        self.assert_refuses(lambda: self.api.release_unsent_startup_retention(owner, expected_sequence=0, now=11.0), "Startup retention is terminal")
        self.assertIs(self.api._RETENTION_RECORDS[owner.owner_id], before)

    def test_sequence_types_stale_and_backwards_preserve(self) -> None:
        owner = self.prepare()
        before = self.api._RETENTION_RECORDS[owner.owner_id]
        for sequence in (True, -1, 2**64):
            with self.subTest(sequence=sequence):
                self.assert_refuses(lambda: self.api.cancel_unsent_startup_retention(owner, expected_sequence=sequence, now=2.0))
        self.assert_refuses(lambda: self.api.cancel_unsent_startup_retention(owner, expected_sequence=1, now=2.0), "Startup retention sequence mismatch")
        self.assert_refuses(lambda: self.api.startup_retention_snapshot(owner, now=0.9), "Startup retention time moved backwards")
        self.assertIs(self.api._RETENTION_RECORDS[owner.owner_id], before)

    def test_sequence_exhaustion_has_no_wrap(self) -> None:
        owner = self.prepare()
        self.api.publish_startup_retention(owner, self.observation(), expected_sequence=0, now=2.0)
        record = self.api._RETENTION_RECORDS[owner.owner_id]
        object.__setattr__(record, "operation_sequence", 2**64 - 1)
        self.assert_refuses(lambda: self.api.cancel_unsent_startup_retention(owner, expected_sequence=2**64 - 1, now=3.0), "Startup retention sequence exhausted")
        self.assert_refuses(lambda: self.api.startup_retention_snapshot(owner, now=10.0), "Startup retention sequence exhausted")
        self.assertIs(self.api._RETENTION_RECORDS[owner.owner_id], record)

    def test_pid_mismatch_refuses_before_lock(self) -> None:
        class ForbiddenLock:
            def acquire(self, **kwargs: object) -> bool:
                raise AssertionError("inherited lock used")
        self.api.os.getpid = lambda: 1001
        self.api._RETENTION_LOCK = ForbiddenLock()
        self.assert_refuses(lambda: self.prepare(), "Startup retention belongs to a different process")

    def test_nonblocking_busy_preserves_record(self) -> None:
        owner = self.prepare()
        before = self.api._RETENTION_RECORDS[owner.owner_id]
        self.api._RETENTION_BUSY = True
        self.assert_refuses(lambda: self.api.startup_retention_snapshot(owner, now=2.0), "Startup retention busy")
        self.assertIs(self.api._RETENTION_RECORDS[owner.owner_id], before)

    def test_reentry_during_copy_refuses_without_nested_mutation(self) -> None:
        owner = self.prepare()
        copy = self.api._retention_copy_request
        observed = []
        def copy_with_reentry(value: object) -> object:
            try:
                self.api.cancel_unsent_startup_retention(owner, expected_sequence=0, now=2.0)
            except self.api.EnforcementUnavailable as error:
                observed.append(str(error))
            return copy(value)
        self.api._retention_copy_request = copy_with_reentry
        row = self.api.startup_retention_snapshot(owner, now=2.0)
        self.assertEqual(observed, ["Startup retention busy"])
        self.assertEqual((row.phase, row.operation_sequence), ("prepared", 0))

    def test_registry_binding_revalidated_after_output_allocation(self) -> None:
        owner = self.prepare()
        original = self.api._RETENTION_RECORDS[owner.owner_id]
        replacement = replace(original)
        copy = self.api._retention_copy_request
        def copy_with_rebinding(value: object) -> object:
            self.api._RETENTION_RECORDS[owner.owner_id] = replacement
            return copy(value)
        self.api._retention_copy_request = copy_with_rebinding
        self.assert_refuses(lambda: self.api.startup_retention_snapshot(owner, now=2.0), "Startup retention invariant changed")
        self.assertIs(self.api._RETENTION_RECORDS[owner.owner_id], replacement)
        self.assertEqual((replacement.last_now, replacement.operation_sequence), (1.0, 0))

    def test_fixed_failure_does_not_echo_private_value(self) -> None:
        self.assert_refuses(lambda: self.api._retention_fail("synthetic-private-environment-value"))

    def test_registry_object_revalidated_after_cloning(self) -> None:
        owner = self.prepare()
        original = self.api._RETENTION_RECORDS
        copy = self.api._retention_copy_request
        def copy_with_registry_swap(value: object) -> object:
            self.api._RETENTION_RECORDS = dict(original)
            return copy(value)
        self.api._retention_copy_request = copy_with_registry_swap
        self.assert_refuses(lambda: self.api.startup_retention_snapshot(owner, now=2.0), "Startup retention invariant changed")
        self.assertEqual(original[owner.owner_id].last_now, 1.0)

    def test_identifier_rollback_cannot_overwrite_registered_owner(self) -> None:
        owner = self.prepare()
        original = self.api._RETENTION_RECORDS[owner.owner_id]
        self.api._RETENTION_NEXT_OWNER_ID = 0
        self.assert_refuses(lambda: self.api.prepare_startup_retention(self.make_request(2), deadline=10.0, now=1.0), "Startup retention invariant changed")
        self.assertIs(self.api._RETENTION_RECORDS[owner.owner_id], original)

    def test_pure_graph_has_no_native_runtime_calls(self) -> None:
        forbidden = {"open", "sleep", "monotonic", "time", "Popen", "socket", "execve", "pidfd_open", "pidfd_send_signal", "launch_owned_linux", "capture_owned_exit", "stop_owned_exact", "terminate_owned_exact"}
        for node in ast.parse(SOURCE.read_text()).body:
            if isinstance(node, ast.FunctionDef) and (node.name in S1_FUNCTIONS or node.name.startswith("_retention_")):
                names = {call.func.id if isinstance(call.func, ast.Name) else call.func.attr for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, (ast.Name, ast.Attribute))}
                self.assertEqual(names & forbidden, set(), node.name)


if __name__ == "__main__":
    unittest.main()
