"""Synthetic controls for sampled RSS plus declared future allocation admission.

Run only in root's isolated test environment. No model or package initializer loads.
"""
from __future__ import annotations

import ast
import __future__
import gc
import importlib.util
import json
from pathlib import Path
import sys
import struct
import threading
from types import ModuleType, SimpleNamespace
import unittest
import weakref
from unittest.mock import patch


_PATH = Path(__file__).resolve().parents[1] / "truememory" / "memory_budget.py"
_SPEC = importlib.util.spec_from_file_location("_memory_budget_author_297", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
budget = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = budget
_SPEC.loader.exec_module(budget)


class Buffer:
    def __init__(self, nbytes: int, base: object = None) -> None:
        self.nbytes = nbytes
        self.base = base


class FalseSensor:
    def __bool__(self) -> bool:
        return False

    def __call__(self) -> int:
        return 100


def key(stage: str = "synthetic stage") -> object:
    return budget.ProfileKey(stage, "synthetic")


def plan(amount: int, stage: str = "synthetic stage") -> object:
    return budget.AllocationPlan(key(stage), amount)


def native_key() -> object:
    return budget.ProfileKey(
        "native synthetic", "synthetic", runtime_fingerprint="1" * 64,
        source_fingerprint="2" * 64, model_identity=("synthetic/model", 32, "synthetic"),
        resolved_revision="3" * 40, asset_manifest_sha256="4" * 64,
        device="cpu", backend="synthetic", dtype="float32", attention_impl="eager",
        token_shape=((1, 8), (1, 8)), native_tensor_shape=(1, 8, 32),
        batch_size=1, max_seq_length=8, padding_policy="longest", truncation_policy="max_length",
        thread_config_fingerprint="5" * 64,
    )


class RegistryControls(unittest.TestCase):
    def test_sum_boundary_and_draining_minimum(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        high = registry.register_owner(1000, kind="high")
        low = registry.register_owner(500, kind="low")
        disabled = registry.register_owner(0, kind="disabled")
        ticket = low.reserve(plan(300))
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 300)
        with self.assertRaises(budget.MemoryAdmissionRefused):
            high.reserve(plan(101))
        low.close()
        self.assertEqual(registry.snapshot()["effective_limit_bytes"], 500)
        disabled.close()
        self.assertEqual(registry.snapshot()["effective_limit_bytes"], 500)
        ticket.close()
        self.assertEqual(registry.snapshot()["effective_limit_bytes"], 1000)
        high.close()
        self.assertEqual(registry.snapshot()["owner_count"], 0)

    def test_current_rss_equal_limit_refuses_even_zero_demand(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(100, kind="equal")
        with self.assertRaises(budget.MemoryAdmissionRefused):
            owner.reserve(plan(0))
        self.assertEqual(registry.snapshot()["lease_count"], 0)
        owner.close()

    def test_sum_equal_limit_accepts(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(500, kind="equal")
        ticket = owner.reserve(plan(400))
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 400)
        ticket.close()
        owner.close()

    def test_disabled_owners_do_not_read_sensor(self) -> None:
        def unavailable() -> object:
            raise AssertionError("disabled sensor was queried")
        registry = budget.ProcessRegistry(usage_reader=unavailable)
        owner = registry.register_owner(0, kind="disabled")
        ticket = owner.reserve(plan(500))
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 500)
        ticket.close()
        owner.close()

    def test_sensor_boundaries(self) -> None:
        for value in (None, False, True, 0, -1, 1.0, 2**63):
            with self.subTest(value=value):
                registry = budget.ProcessRegistry(usage_reader=lambda: value)
                owner = registry.register_owner(1000, kind="sensor")
                with self.assertRaises(budget.MemoryAdmissionRefused):
                    owner.reserve(plan(1))
                owner.close()

    def test_false_callable_sensor_is_preserved(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=FalseSensor())
        owner = registry.register_owner(1000, kind="sensor", usage_reader=FalseSensor())
        ticket = owner.reserve(plan(900))
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 900)
        ticket.close()
        owner.close()
        for invalid in (0, False, "sensor"):
            with self.assertRaises(ValueError):
                budget.ProcessRegistry(usage_reader=invalid)

    def test_atomic_concurrent_admission(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="concurrent")
        barrier = threading.Barrier(3)
        successes = []
        failures = []
        def worker() -> None:
            barrier.wait(timeout=5)
            try:
                successes.append(owner.reserve(plan(600)))
            except budget.MemoryAdmissionRefused as error:
                failures.append(error)
        workers = [threading.Thread(target=worker) for _ in range(2)]
        for worker_thread in workers:
            worker_thread.start()
        barrier.wait(timeout=5)
        for worker_thread in workers:
            worker_thread.join(timeout=5)
            self.assertFalse(worker_thread.is_alive())
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 600)
        successes[0].close()
        failures.clear()
        owner.close()

    def test_overflow_and_bool_are_not_estimates(self) -> None:
        for value in (True, False, -1, 2**63):
            with self.assertRaises(ValueError):
                plan(value)
        registry = budget.ProcessRegistry(usage_reader=lambda: 1)
        owner = registry.register_owner(0, kind="overflow")
        ticket = owner.reserve(plan(2**63 - 1))
        with self.assertRaises(budget.MemoryAdmissionRefused):
            owner.reserve(plan(1))
        ticket.close()
        owner.close()

    def test_unknown_coverage_is_none_not_native_zero(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        unknown = registry.resolve_profile(key("unknown"))
        self.assertIsNone(unknown.future_bytes)
        self.assertIsNone(unknown.retained_bytes)
        self.assertEqual(unknown.basis, "unprofiled")
        owner = registry.register_owner(1000, kind="unknown")
        ticket = owner.reserve(unknown)
        self.assertIn("unknown", registry.snapshot()["unprofiled_stages"])
        self.assertIsNone(ticket.plan.future_bytes)
        ticket.close()
        owner.close()

    def test_retained_split_preserves_empirical_provenance(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        empirical = budget.AllocationPlan(
            native_key(), 300, 50, "empirical_native_profile",
            profile_id="6" * 64, evidence_sha256=("7" * 64,), reason="Synthetic measured fixture",
        )
        owner = registry.register_owner(1000, kind="retention")
        original = owner.reserve(empirical)
        retained = original.split_retained(50)
        self.assertEqual(retained.plan.basis, empirical.basis)
        self.assertEqual(retained.plan.profile_id, empirical.profile_id)
        self.assertEqual(retained.plan.evidence_sha256, empirical.evidence_sha256)
        self.assertEqual(retained.plan.reason, empirical.reason)
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 300)
        original.close()
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 50)
        owner.close()
        self.assertEqual(registry.snapshot()["effective_limit_bytes"], 1000)
        retained.close()
        self.assertEqual(registry.snapshot()["owner_count"], 0)

    def test_repeated_buffer_and_views_share_one_retained_ticket(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="result")
        result = Buffer(40)
        view = Buffer(20, result)
        with budget.allocation_scope(owner) as session:
            session.charge(key("result assembly"), known_bytes=40)
            session.retain_buffer(result)
            session.retain_buffer(result)
            session.retain_buffer(view)
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 40)
        owner.close()
        del result
        gc.collect()
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 40)
        del view
        gc.collect()
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 0)
        self.assertEqual(registry.snapshot()["owner_count"], 0)

    def test_exception_lifetime_keeps_declared_reservation(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="failure")
        _failure = None
        try:
            with budget.allocation_scope(owner) as session:
                session.charge(key("failed stage"), known_bytes=200)
                raise RuntimeError("synthetic retained failure")
        except RuntimeError as error:
            _failure = error
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 200)
        owner.close()
        self.assertEqual(registry.snapshot()["effective_limit_bytes"], 1000)
        session = None
        _failure = None
        gc.collect()
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 0)

    def test_exact_profiles_bind_only_before_first_owner(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        expected = plan(200)
        registry.bind_profiles((expected,))
        self.assertEqual(registry.resolve_profile(expected.key), expected)
        owner = registry.register_owner(1000, kind="bound")
        with self.assertRaises(budget.InvalidAllocationPlan):
            registry.bind_profiles((expected,))
        owner.close()

    def test_identity_binding_fills_only_static_exact_fields(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        runtime = budget.ProfileKey("runtime context", "context", runtime_fingerprint="1" * 64,
                                    source_fingerprint="2" * 64, thread_config_fingerprint="3" * 64)
        model = native_key()
        registry.bind_identity_context(runtime, (model,))
        actual = registry.enrich_key(budget.ProfileKey(
            "construction", "synthetic", model_identity=model.model_identity, device="cpu",
        ))
        self.assertEqual(actual.asset_manifest_sha256, model.asset_manifest_sha256)
        self.assertEqual(actual.attention_impl, "eager")
        self.assertIsNone(actual.token_shape)
        self.assertIsNone(actual.native_tensor_shape)
        self.assertEqual(actual.runtime_fingerprint, runtime.runtime_fingerprint)

    def test_no_physical_enforcement_claim(self) -> None:
        registry = budget.ProcessRegistry()
        self.assertFalse(registry.snapshot()["physical_hard_cap"])
        self.assertFalse(registry.capability.physical_hard_cap)
        with self.assertRaises(ValueError):
            budget.EnforcementCapability(physical_hard_cap=True)


class SchemaControls(unittest.TestCase):
    def test_custom_strings_rejected_before_hashing(self) -> None:
        class CustomString(str):
            pass
        for changes in ({"stage": CustomString("stage")}, {"owner_kind": CustomString("owner")},
                        {"device": CustomString("cpu")},
                        {"model_identity": (CustomString("model"), 32, "tier")},
                        {"live_residency_signature": (("m", "cpu", CustomString("float32"), "r"),)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                budget.ProfileKey(**({"stage": "stage", "owner_kind": "owner"} | changes))
        for reason in (1, False, CustomString("reason"), ""):
            with self.assertRaises(budget.InvalidAllocationPlan):
                budget.AllocationPlan(key(), None, None, "unprofiled",
                                      unknown_components=("workspace",), reason=reason)

    def test_subclasses_cannot_skip_boundary_validation(self) -> None:
        class InvalidKey(budget.ProfileKey):
            def __post_init__(self) -> None:
                pass
        class InvalidPlan(budget.AllocationPlan):
            def __post_init__(self) -> None:
                pass
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="schema")
        with self.assertRaises(budget.InvalidAllocationPlan):
            budget.AllocationPlan(InvalidKey("stage", "owner"), 1)
        invalid = InvalidPlan(key(), True)
        with self.assertRaises(budget.InvalidAllocationPlan):
            owner.reserve(invalid)
        with self.assertRaises(budget.InvalidAllocationPlan):
            registry.bind_profiles((invalid,))
        owner.close()

    def test_mutated_exact_instance_is_revalidated(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="schema")
        invalid = plan(1)
        object.__setattr__(invalid, "future_bytes", True)
        with self.assertRaises(ValueError):
            owner.reserve(invalid)
        owner.close()

    def test_complete_native_identity_and_canonical_evidence_required(self) -> None:
        complete = native_key()
        for field in ("attention_impl", "max_seq_length", "padding_policy", "truncation_policy"):
            with self.subTest(field=field):
                fields = complete.__dict__ | {field: None}
                with self.assertRaises(budget.InvalidAllocationPlan):
                    budget.AllocationPlan(budget.ProfileKey(**fields), 100, 0,
                                          "empirical_native_profile", "6" * 64, ("7" * 64,))
        for evidence in (("bad",), ("A" * 64,), (True,), ["7" * 64]):
            with self.assertRaises((ValueError, budget.InvalidAllocationPlan)):
                budget.AllocationPlan(complete, 100, 0, "empirical_native_profile",
                                      "6" * 64, evidence)
        with self.assertRaises(ValueError):
            budget.ProfileKey("stage", "owner", token_shape=(None,))
        self.assertEqual(budget.ProfileKey("stage", "owner", token_shape=((0,),)).token_shape, ((0,),))

    def test_resolver_mismatch_is_rejected_before_reservation(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="resolver")
        with patch.object(budget, "resolve_allocation_plan", return_value=plan(10, "other")):
            with budget.allocation_scope(owner) as session:
                with self.assertRaises(budget.InvalidAllocationPlan):
                    session.charge(key("requested"))
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 0)
        owner.close()


class ReentrantLifetimeControls(unittest.TestCase):
    def test_finalizers_reenter_each_lock_without_disabling_gc(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="finalizer")
        for lock in (budget._registry_lock, registry._lock, budget._model_owners_lock,
                     budget._retained_buffers_lock):
            called = []
            failures = []
            def action() -> None:
                with lock:
                    called.append(True)
            def worker() -> None:
                try:
                    witness = Buffer(0)
                    finalize = weakref.finalize(witness, action)
                    with lock:
                        finalize()
                except Exception as error:
                    failures.append(error)
            child = threading.Thread(target=worker, daemon=True)
            child.start()
            child.join(timeout=5)
            self.assertFalse(child.is_alive(), "Finalizer reentry deadlocked")
            self.assertEqual(failures, [])
            self.assertEqual(called, [True])
        self.assertTrue(gc.isenabled())
        owner.close()

    def test_reader_lowered_ceiling_is_recomputed_before_commit(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="reader")
        def sensor() -> int:
            owner.set_limit(200)
            return 100
        owner._usage_reader = sensor
        with self.assertRaises(budget.MemoryAdmissionRefused):
            owner.reserve(plan(101))
        self.assertEqual(registry.snapshot()["lease_count"], 0)
        owner.close()

    def test_reader_nested_reservation_is_added_to_actual_total(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="reader")
        held = []
        entered = False
        def sensor() -> int:
            nonlocal entered
            if not entered:
                entered = True
                held.append(owner.reserve(plan(600, "nested")))
            return 100
        owner._usage_reader = sensor
        with self.assertRaises(budget.MemoryAdmissionRefused):
            owner.reserve(plan(301, "outer"))
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 600)
        held[0].close()
        owner.close()

    def test_reader_closed_owner_cannot_reserve_new_ticket(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="reader")
        def sensor() -> int:
            owner.close()
            return 100
        owner._usage_reader = sensor
        with self.assertRaises(budget.StaleBudgetOwner):
            owner.reserve(plan(1))
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 0)

    def test_reader_retired_grow_ticket_is_revalidated(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="reader")
        ticket = owner.reserve(plan(100))
        def sensor() -> int:
            ticket.close()
            return 100
        owner._usage_reader = sensor
        with self.assertRaises(budget.StaleBudgetOwner):
            ticket.grow(plan(200))
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 0)
        owner.close()

    def test_nested_growth_recomputes_increment_from_live_ticket(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="reader")
        ticket = owner.reserve(plan(100))
        entered = False
        def sensor() -> int:
            nonlocal entered
            if not entered:
                entered = True
                ticket.grow(plan(500))
            return 100
        owner._usage_reader = sensor
        ticket.grow(plan(300))
        self.assertEqual(ticket.reserved_known_bytes, 500)
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 500)
        ticket.close()
        owner.close()

    def test_closing_owner_growth_drains_under_retained_minimum(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(500, kind="draining")
        ticket = owner.reserve(plan(100))
        owner.close()
        ticket.grow(plan(300))
        self.assertEqual(registry.snapshot()["effective_limit_bytes"], 500)
        with self.assertRaises(budget.MemoryAdmissionRefused):
            ticket.grow(plan(401))
        with self.assertRaises(budget.StaleBudgetOwner):
            owner.reserve(plan(0))
        ticket.close()
        self.assertEqual(registry.snapshot()["owner_count"], 0)

    def test_fallback_reader_is_followed_by_deadline_validation(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(0, kind="disabled")
        held = []
        calls = []
        def deadline() -> None:
            calls.append("deadline")
            if len(calls) == 2:
                held.append(registry.register_owner(1000, kind="late finite"))
        def sensor() -> int:
            calls.append("sensor")
            return 100
        owner._usage_reader = sensor
        ticket = owner.reserve(plan(1), deadline=deadline)
        self.assertEqual(calls, ["deadline", "deadline", "sensor", "deadline"])
        ticket.close()
        owner.close()
        held[0].close()

    def test_duplicate_model_owner_closes_outside_global_map_lock(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        original = registry.register_owner(1000, kind="model")
        duplicate = registry.register_owner(1000, kind="duplicate")
        model = Buffer(0)
        budget.bind_model_owner(model, original)
        observed = []
        close = duplicate.close
        def checked_close() -> None:
            acquired = []
            def worker() -> None:
                got_lock = budget._model_owners_lock.acquire(timeout=1)
                acquired.append(got_lock)
                if got_lock:
                    budget._model_owners_lock.release()
            child = threading.Thread(target=worker, daemon=True)
            child.start()
            child.join(timeout=2)
            observed.append(acquired)
            close()
        duplicate.close = checked_close
        with self.assertRaises(budget.InvalidAllocationPlan):
            budget.bind_model_owner(model, duplicate)
        self.assertEqual(observed, [[True]])
        self.assertIs(budget.model_owner(model), original)
        del model
        gc.collect()


class CooperativeBorrowControls(unittest.TestCase):
    def test_active_owner_is_sensor_free_existing_positive_minimum(self) -> None:
        def forbidden() -> int:
            self.fail("Borrow selection must not sample RSS")
        registry = budget.ProcessRegistry(usage_reader=forbidden)
        large = registry.register_owner(1000, kind="large")
        disabled = registry.register_owner(0, kind="disabled")
        small = registry.register_owner(500, kind="small")
        self.assertIs(registry.active_owner(), small)
        self.assertEqual(registry.snapshot()["owner_count"], 3)
        small.close()
        self.assertIs(registry.active_owner(), large)
        disabled.close()
        self.assertIs(registry.active_owner(), large)
        large.close()
        self.assertIsNone(registry.active_owner())

    def test_closing_owner_is_not_borrowed_but_keeps_draining_ceiling(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        large = registry.register_owner(1000, kind="large")
        small = registry.register_owner(500, kind="small")
        held = small.reserve(plan(1))
        small.close()
        self.assertIs(registry.active_owner(), large)
        self.assertEqual(registry.snapshot()["effective_limit_bytes"], 500)
        with self.assertRaises(budget.MemoryAdmissionRefused):
            large.reserve(plan(400))
        held.close()
        self.assertEqual(registry.snapshot()["effective_limit_bytes"], 1000)
        large.close()

    def test_cooperative_scope_reuses_exact_current_session(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="current")
        with budget.allocation_scope(owner) as current:
            with patch.object(budget, "get_process_registry", side_effect=AssertionError("No registry lookup")):
                with budget.cooperative_scope() as borrowed:
                    self.assertIs(borrowed, current)
                    self.assertIs(budget.current_session(), current)
        owner.close()

    def test_cooperative_scope_enrolls_no_owner_and_keeps_legacy_none(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        with patch.object(budget, "get_process_registry", return_value=registry):
            with budget.cooperative_scope() as legacy:
                self.assertIsNone(legacy)
            owner = registry.register_owner(1000, kind="existing")
            with budget.cooperative_scope() as borrowed:
                self.assertIs(borrowed.owner, owner)
                self.assertEqual(registry.snapshot()["owner_count"], 1)
                borrowed.charge(key("borrowed"), known_bytes=1)
            self.assertEqual(registry.snapshot()["reserved_known_bytes"], 0)
        owner.close()

    def test_closed_borrow_refuses_before_deadline_or_reader(self) -> None:
        def forbidden() -> int:
            self.fail("Retired owner must be checked before external callbacks")
        registry = budget.ProcessRegistry(usage_reader=forbidden)
        owner = registry.register_owner(1000, kind="closed")
        selected = registry.active_owner()
        owner.close()
        with self.assertRaises(budget.StaleBudgetOwner):
            selected.reserve(plan(1), deadline=forbidden)
        self.assertEqual(registry.snapshot()["lease_count"], 0)

    def test_actual_client_transport_borrows_through_socket_close(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="client")
        observed = []
        class SyntheticSocket:
            def settimeout(self, timeout: float) -> None:
                self.assert_scope()
            def sendall(self, data: bytes) -> None:
                self.assert_scope()
            def close(self) -> None:
                self.assert_scope()
                observed.append("closed")
            def assert_scope(self) -> None:
                observed.append(budget.current_session().owner)
        socket = SyntheticSocket()
        source = _PATH.with_name("model_client.py")
        tree = ast.parse(source.read_bytes())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_send_request")
        namespace = {"json": json, "struct": struct, "_HEADER_FMT": "!I", "_HEADER_SIZE": 4,
                     "_REQUEST_TIMEOUT": 120, "_remaining": lambda deadline: 120,
                     "_connect": lambda deadline: socket,
                     "_receive_response": lambda socket, deadline: {"ok": True}}
        code = compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec",
                       flags=__future__.annotations.compiler_flag)
        exec(code, namespace)
        package = ModuleType("truememory")
        package.memory_budget = budget
        with patch.dict(sys.modules, {"truememory": package, "truememory.memory_budget": budget}), \
             patch.object(budget, "get_process_registry", return_value=registry):
            result = namespace["_send_request"]({"synthetic": 1})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(observed, [owner, owner, owner, "closed"])
        self.assertEqual(registry.snapshot()["owner_count"], 1)
        self.assertEqual(registry.snapshot()["reserved_known_bytes"], 0)
        self.assertIsNone(budget.current_session())
        owner.close()


class ActualResultMaterializationControls(unittest.TestCase):
    def functions(self, events: list[str]) -> dict:
        class Array:
            ndim = 2
            shape = (2, 3)

        def asarray(value: object, dtype: object) -> object:
            events.append("asarray")
            return value if isinstance(value, Array) else Array()

        source = _PATH.with_name("model_server.py")
        tree = ast.parse(source.read_bytes())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "_checked_batch"]
        namespace = {"np": SimpleNamespace(ndarray=Array, float32="float32", asarray=asarray),
                     "_check_result_size": lambda shape, field: events.append("size check")}
        code = compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec",
                       flags=__future__.annotations.compiler_flag)
        exec(code, namespace)
        return namespace

    def test_actual_nonarray_materialization_is_preceded_by_profile_hook(self) -> None:
        events = []
        functions = self.functions(events)
        functions["_checked_batch"]([[0.0] * 3] * 2, 2, 2, "vectors",
                                      admit=lambda shape, cast: events.append("result admission"),
                                      before_materialize=lambda: events.append("materialization admission"))
        self.assertEqual(events, ["materialization admission", "asarray", "size check",
                                  "result admission", "asarray"])

    def test_actual_refusal_prevents_nonarray_allocation(self) -> None:
        events = []
        functions = self.functions(events)
        def refused() -> None:
            raise budget.MemoryAdmissionRefused("synthetic materialization demand")
        with self.assertRaises(budget.MemoryAdmissionRefused):
            functions["_checked_batch"]([[0.0] * 3] * 2, 2, 2, "vectors",
                                          before_materialize=refused)
        self.assertEqual(events, [])

    def test_actual_legacy_array_path_does_not_require_new_dtype_probe(self) -> None:
        events = []
        functions = self.functions(events)
        array = functions["np"].ndarray()
        result = functions["_checked_batch"](array, 2, 2, "vectors")
        self.assertIs(result, array)
        self.assertEqual(events, ["size check", "asarray"])


class LocalRoutingControls(unittest.TestCase):
    def test_inherited_policy_constructor_and_reflective_mutation_refuse(self) -> None:
        class InheritedPolicy(budget.LocalMemoryPolicy):
            pass
        with self.assertRaises(budget.InvalidAllocationPlan):
            InheritedPolicy()
        self.assertIs(budget.validate_local_policy(budget.DEFAULT_LOCAL_MEMORY_POLICY),
                      budget.DEFAULT_LOCAL_MEMORY_POLICY)
        for field, value in (("version", True), ("mode", "disabled")):
            policy = budget.LocalMemoryPolicy()
            object.__setattr__(policy, field, value)
            with self.subTest(field=field), self.assertRaises(budget.InvalidAllocationPlan):
                budget.validate_local_policy(policy)

    def test_policy_requires_exact_fields_type_and_version(self) -> None:
        policy = budget.LocalMemoryPolicy()
        self.assertIs(budget.validate_local_policy(policy), policy)
        for fields in ({"version": True}, {"version": 2}, {"mode": None}, {"mode": "disabled"}):
            with self.subTest(fields=fields), self.assertRaises(budget.InvalidAllocationPlan):
                budget.LocalMemoryPolicy(**fields)
        class SubPolicy(budget.LocalMemoryPolicy):
            def __post_init__(self) -> None:
                pass
        for invalid in (None, {}, SubPolicy(version=True)):
            with self.subTest(invalid=invalid), self.assertRaises(budget.InvalidAllocationPlan):
                budget.validate_local_policy(invalid)
        object.__setattr__(policy, "mode", "disabled")
        with self.assertRaises(budget.InvalidAllocationPlan):
            budget.validate_local_policy(policy)

    def test_configured_factory_validates_before_sensor_and_preserves_kind(self) -> None:
        with patch.object(budget, "new_local_owner") as factory:
            for policy, kind in ((None, "local"), (budget.LocalMemoryPolicy(), ""),
                                 (budget.LocalMemoryPolicy(), None)):
                with self.subTest(kind=kind), self.assertRaises(budget.InvalidAllocationPlan):
                    budget.configured_local_owner(policy, kind=kind)
            factory.assert_not_called()
            result = budget.configured_local_owner(budget.LocalMemoryPolicy(), kind="local_embedding")
            self.assertIs(result, factory.return_value)
            factory.assert_called_once_with("local_embedding")

    def test_unbound_operation_preserves_exact_current_session(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="current")
        with patch.object(budget, "get_process_registry", return_value=registry):
            with budget.allocation_scope(owner) as current:
                with budget.model_operation_scope(object(), key("unbound")) as operation:
                    self.assertIs(operation, current)
                    self.assertIs(budget.current_session(), current)
                    self.assertEqual(registry.snapshot()["owner_count"], 1)
        owner.close()

    def test_unbound_operation_borrows_existing_positive_without_enrollment(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(1000, kind="existing")
        with patch.object(budget, "get_process_registry", return_value=registry):
            with budget.model_operation_scope(object(), key("borrowed")) as operation:
                self.assertIs(operation.owner, owner)
                self.assertEqual(registry.snapshot()["owner_count"], 1)
            owner.close()
            with budget.model_operation_scope(object(), key("uncovered")) as operation:
                self.assertIsNone(operation)

    def test_bound_closed_owner_never_falls_back_to_eligible_owner(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        bound = registry.register_owner(500, kind="bound")
        eligible = registry.register_owner(1000, kind="eligible")
        model = Buffer(0)
        with patch.object(budget, "get_process_registry", return_value=registry):
            budget.bind_model_owner(model, bound)
            bound.close()
            with self.assertRaises(budget.StaleBudgetOwner):
                with budget.model_operation_scope(model, key("closed")):
                    self.fail("Closed bound owner must not enter native operation")
        eligible.close()

    def test_operation_validates_key_and_deadline_before_context_lookup(self) -> None:
        with patch.object(budget, "model_owner", side_effect=AssertionError("No context lookup")):
            for supplied_key, deadline in ((None, None), (key(), 1)):
                with self.subTest(deadline=deadline), self.assertRaises(budget.InvalidAllocationPlan):
                    with budget.model_operation_scope(object(), supplied_key, deadline):
                        self.fail("Invalid operation cannot enter")

    def test_both_wrappers_preserve_raw_arguments_and_retain_backing(self) -> None:
        for name in ("budgeted_encode_with_model_ownership", "budgeted_encode_with_mps_fallback"):
            with self.subTest(wrapper=name):
                registry = budget.ProcessRegistry(usage_reader=lambda: 100)
                owner = registry.register_owner(1000, kind="existing")
                model, texts = object(), ["synthetic text"]
                result = Buffer(24)
                events = []
                def encode(actual_model: object, actual_texts: object, **kwargs: object) -> object:
                    self.assertIs(actual_model, model)
                    self.assertIs(actual_texts, texts)
                    self.assertEqual(kwargs, {"show_progress_bar": False, "batch_size": 7})
                    self.assertIsNotNone(budget.current_session())
                    events.append("native")
                    return result
                package = ModuleType("truememory")
                package.__path__ = []
                mps = ModuleType("truememory.mps_utils")
                mps.encode_with_model_ownership = encode
                mps.encode_with_mps_fallback = encode
                with patch.dict(sys.modules, {"truememory": package, "truememory.mps_utils": mps}), \
                        patch.object(budget, "get_process_registry", return_value=registry):
                    returned = getattr(budget, name)(
                        model, texts, key("opaque operation"),
                        native_kwargs={"show_progress_bar": False, "batch_size": 7},
                    )
                    self.assertIs(returned, result)
                    self.assertEqual(events, ["native"])
                    self.assertEqual(registry.snapshot()["reserved_known_bytes"], 24)
                    owner.close()
                    self.assertEqual(registry.snapshot()["effective_limit_bytes"], 1000)
                    del returned
                    result = None
                    gc.collect()
                    self.assertEqual(registry.snapshot()["owner_count"], 0)

    def test_wrapper_refusal_happens_before_native_call(self) -> None:
        registry = budget.ProcessRegistry(usage_reader=lambda: 100)
        owner = registry.register_owner(100, kind="full")
        package = ModuleType("truememory")
        package.__path__ = []
        mps = ModuleType("truememory.mps_utils")
        mps.encode_with_model_ownership = lambda *args, **kwargs: self.fail("Native call after refusal")
        with patch.dict(sys.modules, {"truememory": package, "truememory.mps_utils": mps}), \
                patch.object(budget, "get_process_registry", return_value=registry):
            with self.assertRaises(budget.MemoryAdmissionRefused):
                budget.budgeted_encode_with_model_ownership(object(), ["synthetic"], key("refused"))
        owner.close()

    def test_prepared_result_payload_admission_precedes_both_actual_encodes(self) -> None:
        source = _PATH.with_name("vector_search.py")
        module = ast.parse(source.read_text())
        frozen = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_load_frozen_embedding_target")
        lease = next(node for node in module.body if isinstance(node, ast.ClassDef)
                     and node.name == "_PreparedEmbeddingLease")
        encode = next(node for node in lease.body if isinstance(node, ast.FunctionDef) and node.name == "encode")
        for method in (frozen, encode):
            with self.subTest(method=method.name):
                charges = [node for node in ast.walk(method) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute) and node.func.attr == "charge"
                           and any(keyword.arg == "group" and isinstance(keyword.value, ast.Constant)
                                   and keyword.value.value == "result assembly" for keyword in node.keywords)]
                encodes = [node for node in ast.walk(method) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute) and node.func.attr == "encode"
                           and isinstance(node.func.value, ast.Name) and node.func.value.id == "encoder"]
                self.assertEqual(len(charges), 1)
                self.assertTrue(encodes)
                self.assertLess(charges[0].lineno, max(node.lineno for node in encodes))
                self.assertTrue(any(keyword.arg == "retained_bytes" for keyword in charges[0].keywords))


if __name__ == "__main__":
    unittest.main()
