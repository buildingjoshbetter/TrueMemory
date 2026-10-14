"""Process-local future allocation admission; this is not an OS physical cap."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
import os
import sys
import threading
from typing import Literal
import weakref

_MAX_BYTES = 2**63 - 1
Basis = Literal["exact_owned_payload", "empirical_runtime_profile",
                "empirical_native_profile", "unprofiled"]


class MemoryAdmissionRefused(RuntimeError):
    """An enabled process ceiling cannot admit an allocation."""


class InvalidAllocationPlan(ValueError):
    """A resolver returned an invalid or mismatched allocation plan."""


class StaleBudgetOwner(RuntimeError):
    """An owner belongs to another PID or retired registry epoch."""


@dataclass(frozen=True)
class LocalMemoryPolicy:
    version: int = 1
    mode: str = "automatic"

    def __post_init__(self) -> None:
        if type(self) is not LocalMemoryPolicy:
            raise InvalidAllocationPlan("Local memory policy requires the exact policy type")
        if type(self.version) is not int or self.version != 1:
            raise InvalidAllocationPlan("Local memory policy requires version 1")
        if type(self.mode) is not str or self.mode != "automatic":
            raise InvalidAllocationPlan("Local memory policy requires automatic mode")


def validate_local_policy(policy: object) -> LocalMemoryPolicy:
    if type(policy) is not LocalMemoryPolicy:
        raise InvalidAllocationPlan("Local memory policy requires the exact policy type")
    LocalMemoryPolicy.__post_init__(policy)
    return policy


DEFAULT_LOCAL_MEMORY_POLICY = LocalMemoryPolicy()


def _bytes(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_BYTES:
        raise ValueError(f"{name} must be a nonnegative signed64 integer")
    return value


def _sum_bytes(left: int, right: int) -> int:
    if left > _MAX_BYTES - right:
        raise MemoryAdmissionRefused("Future allocation accounting overflow")
    return left + right


def _shape(value: object, name: str) -> None:
    if value is None:
        return
    if type(value) is not tuple:
        raise ValueError(f"{name} must be an immutable tuple")
    product = 1
    for dimension in value:
        dimension = _bytes(dimension, name)
        if dimension and product > _MAX_BYTES // dimension:
            raise ValueError(f"{name} product exceeds signed64")
        product *= dimension


def _pin(value: object, name: str) -> None:
    if value is not None and (type(value) is not str or len(value) != 64
                              or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{name} must be a canonical SHA256 or None")


@dataclass(frozen=True)
class ProfileKey:
    stage: str
    owner_kind: str
    runtime_fingerprint: str | None = None
    source_fingerprint: str | None = None
    model_identity: tuple[str, int, str] | None = None
    resolved_revision: str | None = None
    asset_manifest_sha256: str | None = None
    device: str | None = None
    backend: str | None = None
    dtype: str | None = None
    attention_impl: str | None = None
    token_shape: tuple[tuple[int, ...], ...] | None = None
    native_tensor_shape: tuple[int, ...] | None = None
    batch_size: int | None = None
    max_seq_length: int | None = None
    padding_policy: str | None = None
    truncation_policy: str | None = None
    live_residency_signature: tuple[tuple[str, str, str, str], ...] = ()
    thread_config_fingerprint: str | None = None
    input_item_count: int | None = None
    wire_signature: str | None = None

    def __post_init__(self) -> None:
        if type(self.stage) is not str or not self.stage:
            raise ValueError("stage must be a nonempty string")
        if type(self.owner_kind) is not str or not self.owner_kind:
            raise ValueError("owner_kind must be a nonempty string")
        for name in ("runtime_fingerprint", "source_fingerprint", "resolved_revision",
                     "asset_manifest_sha256", "device", "backend", "dtype", "attention_impl",
                     "padding_policy", "truncation_policy", "thread_config_fingerprint"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise ValueError(f"{name} must be a nonempty string or None")
        for name in ("runtime_fingerprint", "source_fingerprint", "asset_manifest_sha256",
                     "thread_config_fingerprint", "wire_signature"):
            _pin(getattr(self, name), name)
        if self.model_identity is not None:
            identity = self.model_identity
            if (type(identity) is not tuple or len(identity) != 3
                    or type(identity[0]) is not str or not identity[0]
                    or type(identity[2]) is not str or not identity[2]):
                raise ValueError("model_identity must be an immutable identity triple")
            _bytes(identity[1], "model dimension")
        _shape(self.native_tensor_shape, "native_tensor_shape")
        if self.token_shape is not None:
            if type(self.token_shape) is not tuple:
                raise ValueError("token_shape must be immutable")
            for shape in self.token_shape:
                if shape is None:
                    raise ValueError("Inner token shapes cannot be None")
                _shape(shape, "token_shape")
        for name in ("batch_size", "max_seq_length", "input_item_count"):
            value = getattr(self, name)
            if value is not None:
                _bytes(value, name)
        if type(self.live_residency_signature) is not tuple:
            raise ValueError("live_residency_signature must be immutable")
        for identity in self.live_residency_signature:
            if (type(identity) is not tuple or len(identity) != 4
                    or not all(type(part) is str for part in identity)):
                raise ValueError("residency entries must be immutable string quadruples")


@dataclass(frozen=True)
class AllocationPlan:
    key: ProfileKey
    future_bytes: int | None
    retained_bytes: int | None = 0
    basis: Basis = "exact_owned_payload"
    profile_id: str | None = None
    evidence_sha256: tuple[str, ...] = ()
    unknown_components: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.key) is not ProfileKey:
            raise InvalidAllocationPlan("key must be a ProfileKey")
        ProfileKey.__post_init__(self.key)
        if type(self.basis) is not str or self.basis not in ("exact_owned_payload", "empirical_runtime_profile",
                              "empirical_native_profile", "unprofiled"):
            raise InvalidAllocationPlan("Unsupported allocation basis")
        for name in ("evidence_sha256", "unknown_components"):
            value = getattr(self, name)
            if type(value) is not tuple or not all(type(item) is str and item for item in value):
                raise InvalidAllocationPlan(f"{name} must be immutable strings")
        _pin(self.profile_id, "profile_id")
        for evidence in self.evidence_sha256:
            _pin(evidence, "evidence SHA256")
        if self.reason is not None and (type(self.reason) is not str or not self.reason):
            raise InvalidAllocationPlan("reason must be a nonempty exact string or None")
        if self.future_bytes is None:
            if (self.basis != "unprofiled" or self.retained_bytes is not None
                    or not self.reason or not self.unknown_components):
                raise InvalidAllocationPlan("Unknown demand requires None bytes and explicit coverage")
        else:
            _bytes(self.future_bytes, "future_bytes")
            if self.retained_bytes is None:
                raise InvalidAllocationPlan("Known demand needs a retained portion")
            _bytes(self.retained_bytes, "retained_bytes")
            if self.retained_bytes > self.future_bytes or self.basis == "unprofiled":
                raise InvalidAllocationPlan("Invalid retained portion or known-demand basis")
        if self.basis.startswith("empirical"):
            if not self.profile_id or not self.evidence_sha256:
                raise InvalidAllocationPlan("Empirical demand needs immutable profile evidence")
            if self.key.runtime_fingerprint is None or self.key.source_fingerprint is None:
                raise InvalidAllocationPlan("Empirical demand needs exact runtime and source pins")
            if self.basis == "empirical_native_profile" and any(value is None for value in (
                self.key.model_identity, self.key.resolved_revision, self.key.asset_manifest_sha256,
                self.key.device, self.key.backend, self.key.dtype, self.key.token_shape,
                self.key.native_tensor_shape, self.key.batch_size, self.key.thread_config_fingerprint,
                self.key.attention_impl, self.key.max_seq_length,
                self.key.padding_policy, self.key.truncation_policy,
            )):
                raise InvalidAllocationPlan("Native profile coverage is incomplete")


def _validate_plan(plan: object) -> AllocationPlan:
    if type(plan) is not AllocationPlan:
        raise InvalidAllocationPlan("AllocationPlan subclasses are not accepted")
    AllocationPlan.__post_init__(plan)
    return plan


def _validate_key(key: object) -> ProfileKey:
    if type(key) is not ProfileKey:
        raise InvalidAllocationPlan("ProfileKey subclasses are not accepted")
    ProfileKey.__post_init__(key)
    return key


@dataclass(frozen=True)
class EnforcementCapability:
    status: Literal["unsupported", "unverified", "verified"] = "unsupported"
    metric: str = "none"
    limit_bytes: int | None = None
    scope: str | None = None
    evidence_sha256: tuple[str, ...] = ()
    physical_hard_cap: bool = False
    reason: str = "No native enforcement witness has been accepted"

    def __post_init__(self) -> None:
        if self.status not in ("unsupported", "unverified", "verified"):
            raise ValueError("Invalid enforcement status")
        if self.limit_bytes is not None:
            _bytes(self.limit_bytes, "limit_bytes")
        if self.physical_hard_cap:
            raise ValueError("Phase A cannot certify a unified physical hard cap")
        if self.status == "verified" and (not self.scope or not self.evidence_sha256):
            raise ValueError("Verified capability requires owned-scope evidence")


@dataclass
class _OwnerRecord:
    limit: int
    kind: str
    handle: weakref.ReferenceType[BudgetOwner]
    leases: set[int]
    closing: bool = False


@dataclass
class _LeaseRecord:
    owner: int
    reserved: int


def _default_usage() -> object:
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError, TypeError):
        return None


class ProcessRegistry:
    def __init__(self, *, usage_reader: Callable[[], object] | None = None) -> None:
        if usage_reader is not None and not callable(usage_reader):
            raise ValueError("usage_reader must be callable or None")
        self.pid = os.getpid()
        self.epoch = object()
        self._lock = threading.RLock()
        self._usage_reader = _default_usage if usage_reader is None else usage_reader
        self._owners: dict[int, _OwnerRecord] = {}
        self._leases: dict[int, _LeaseRecord] = {}
        self._next_owner = 0
        self._owner_generation = 0
        self._next_lease = 0
        self._future = 0
        self._high_water = 0
        self._profiles: dict[ProfileKey, AllocationPlan] = {}
        self._runtime_context: ProfileKey | None = None
        self._model_context: dict[tuple[tuple[str, int, str], str], ProfileKey] = {}
        self._unknown_stages: set[str] = set()
        self._coverage_truncated = False
        self._enrolled = False
        self.capability = EnforcementCapability()

    def bind_identity_context(
        self, runtime: ProfileKey, models: tuple[ProfileKey, ...] = (),
    ) -> None:
        self._identity()
        runtime = _validate_key(runtime)
        if any(value is None for value in (runtime.runtime_fingerprint,
                                           runtime.source_fingerprint,
                                           runtime.thread_config_fingerprint)):
            raise InvalidAllocationPlan("Runtime context needs verified exact pins")
        if type(models) is not tuple:
            raise InvalidAllocationPlan("Model contexts must be immutable")
        table: dict[tuple[tuple[str, int, str], str], ProfileKey] = {}
        for model in models:
            model = _validate_key(model)
            if model.model_identity is None or model.device is None:
                raise InvalidAllocationPlan("Model context needs captured identity and device")
            identity = (model.model_identity, model.device)
            if identity in table:
                raise InvalidAllocationPlan("Duplicate exact model context")
            table[identity] = replace(model)
        with self._lock:
            if self._enrolled:
                raise InvalidAllocationPlan("Identity context binds before enrollment only")
            self._runtime_context = replace(runtime)
            self._model_context = table

    def enrich_key(self, key: ProfileKey) -> ProfileKey:
        self._identity()
        key = _validate_key(key)
        with self._lock:
            runtime = self._runtime_context
            model = self._model_context.get((key.model_identity, key.device))
        updates: dict[str, object] = {}
        if runtime is not None:
            for name in ("runtime_fingerprint", "source_fingerprint", "thread_config_fingerprint"):
                supplied, actual = getattr(runtime, name), getattr(key, name)
                if actual is not None and supplied != actual:
                    raise InvalidAllocationPlan("Runtime identity changed before allocation")
                updates[name] = supplied
        if model is not None:
            for name in ("resolved_revision", "asset_manifest_sha256", "backend", "dtype",
                         "attention_impl", "max_seq_length", "padding_policy", "truncation_policy"):
                supplied, actual = getattr(model, name), getattr(key, name)
                if actual is not None and supplied != actual:
                    raise InvalidAllocationPlan("Model identity changed before allocation")
                updates[name] = supplied
        # Input tokens/padded native shapes are never supplied by a static identity manifest.
        return replace(key, **updates) if updates else key

    def _identity(self) -> None:
        if self.pid != os.getpid():
            raise StaleBudgetOwner("Inherited registry is not the child's registry")

    def bind_profiles(self, profiles: tuple[AllocationPlan, ...]) -> None:
        self._identity()
        if type(profiles) is not tuple:
            raise InvalidAllocationPlan("Profiles must be an immutable tuple")
        table: dict[ProfileKey, AllocationPlan] = {}
        for plan in profiles:
            plan = _validate_plan(plan)
            if plan.future_bytes is None:
                raise InvalidAllocationPlan("Only validated known plans can be bound")
            if plan.key in table:
                raise InvalidAllocationPlan("Duplicate exact profile key")
            table[plan.key] = replace(plan, key=replace(plan.key))
        with self._lock:
            if self._enrolled:
                raise InvalidAllocationPlan("Profiles bind only before first owner enrollment")
            self._profiles = table

    def resolve_profile(self, key: ProfileKey) -> AllocationPlan:
        self._identity()
        _validate_key(key)
        with self._lock:
            plan = self._profiles.get(key)
        if plan is not None:
            return _validate_plan(plan)
        return AllocationPlan(
            key, None, None, "unprofiled", unknown_components=(key.stage,),
            reason="No exact calibrated profile covers this runtime/model/input stage",
        )

    def register_owner(
        self, limit_bytes: int, *, kind: str,
        usage_reader: Callable[[], object] | None = None,
    ) -> BudgetOwner:
        self._identity()
        limit = _bytes(limit_bytes, "limit_bytes")
        if type(kind) is not str or not kind:
            raise ValueError("kind must be a nonempty string")
        if usage_reader is not None and not callable(usage_reader):
            raise ValueError("usage_reader must be callable")
        with self._lock:
            self._next_owner += 1
            token = self._next_owner
            owner = BudgetOwner(self, token, self._usage_reader if usage_reader is None else usage_reader)
            self._owners[token] = _OwnerRecord(limit, kind, weakref.ref(owner), set())
            self._owner_generation += 1
            self._enrolled = True
        return owner

    def active_owner(self) -> BudgetOwner | None:
        self._identity()
        with self._lock:
            while True:
                generation = self._owner_generation
                records = self._owners.copy()
                selected = None
                limit = 0
                for token, record in records.items():
                    if record.closing or record.limit <= 0:
                        continue
                    handle = record.handle()
                    if (handle is None or handle.registry is not self
                            or handle.pid != self.pid or handle.epoch is not self.epoch
                            or self._owners.get(token) is not record):
                        continue
                    if not limit or record.limit < limit:
                        selected, limit = handle, record.limit
                # Snapshot metadata or releasing temporary handles can run a
                # finalizer. Selection must reflect the current owner generation.
                if generation != self._owner_generation:
                    continue
                self._identity()
                return selected

    def _record(self, token: int) -> _OwnerRecord:
        record = self._owners.get(token)
        if record is None:
            raise StaleBudgetOwner("Owner has retired")
        return record

    def _ceiling(self) -> int:
        ceiling = 0
        for record in self._owners.values():
            if record.limit > 0 and (ceiling == 0 or record.limit < ceiling):
                ceiling = record.limit
        return ceiling

    def _increment(self, owner: BudgetOwner, requested: int | None, lease_token: int | None) -> int:
        owner._identity()
        if owner.registry is not self or owner.epoch is not self.epoch:
            raise StaleBudgetOwner("Allocation belongs to another registry epoch")
        record = self._record(owner.token)
        if record.handle() is not owner:
            raise StaleBudgetOwner("Owner generation has retired")
        if lease_token is None:
            if record.closing:
                raise StaleBudgetOwner("Owner is closing")
            return 0 if requested is None else requested
        lease = self._leases.get(lease_token)
        if lease is None or lease.owner != owner.token or lease_token not in record.leases:
            raise StaleBudgetOwner("Lease has retired")
        return 0 if requested is None else max(0, requested - lease.reserved)

    def _admit(
        self, owner: BudgetOwner, requested: int | None,
        deadline: Callable[[], None] | None, lease_token: int | None = None,
    ) -> int:
        self._increment(owner, requested, lease_token)
        if deadline is not None:
            deadline()
        ceiling = self._ceiling()
        usage = None
        if ceiling:
            try:
                usage = owner._usage_reader()
            except (AttributeError, OSError, RuntimeError, ValueError, TypeError, OverflowError):
                usage = None
            if deadline is not None:
                deadline()
        if deadline is not None:
            deadline()
        # Readers/deadlines may collect an owner, retire a ticket, or lower a ceiling.
        # No cached owner/lease/accounting state crosses those callbacks.
        if not ceiling and self._ceiling():
            try:
                usage = owner._usage_reader()
            except (AttributeError, OSError, RuntimeError, ValueError, TypeError, OverflowError):
                usage = None
            if deadline is not None:
                deadline()
        ceiling = self._ceiling()
        increment = self._increment(owner, requested, lease_token)
        total = _sum_bytes(self._future, increment)
        if ceiling:
            if type(usage) is not int or not 0 < usage <= _MAX_BYTES:
                raise MemoryAdmissionRefused("Current process RSS is unavailable")
            if usage >= ceiling or _sum_bytes(usage, total) > ceiling:
                raise MemoryAdmissionRefused("Process RSS plus future allocation budget exhausted")
        return increment

    def _note_unknown(self, plan: AllocationPlan) -> None:
        if plan.future_bytes is None or plan.unknown_components:
            if len(self._unknown_stages) < 128:
                self._unknown_stages.add(plan.key.stage)
            elif plan.key.stage not in self._unknown_stages:
                self._coverage_truncated = True

    def _reserve(
        self, owner: BudgetOwner, plan: AllocationPlan, deadline: Callable[[], None] | None,
    ) -> BudgetLease:
        self._identity()
        plan = _validate_plan(plan)
        with self._lock:
            # Unknown native demand remains explicit; only named known bytes enter the sum.
            reserved = 0 if plan.future_bytes is None else plan.future_bytes
            self._next_lease += 1
            token = self._next_lease
            ticket = BudgetLease(owner, token, plan)
            allocation = _LeaseRecord(owner.token, reserved)
            self._admit(owner, reserved, deadline)
            record = self._record(owner.token)
            self._leases[token] = allocation
            record.leases.add(token)
            self._future = _sum_bytes(self._future, reserved)
            self._high_water = max(self._high_water, self._future)
            self._note_unknown(plan)
        return ticket

    def _grow(
        self, lease: BudgetLease, plan: AllocationPlan, deadline: Callable[[], None] | None,
    ) -> None:
        self._identity()
        plan = _validate_plan(plan)
        with self._lock:
            # Hold the previous immutable plan through replacement, so retiring
            # its metadata cannot run a finalizer halfway through accounting.
            _previous = lease.plan
            increment = self._admit(lease.owner, plan.future_bytes, deadline, lease.token)
            record = self._leases[lease.token]
            self._future = _sum_bytes(self._future, increment)
            record.reserved += increment
            self._high_water = max(self._high_water, self._future)
            self._note_unknown(plan)
            lease.plan = plan

    def _split(self, lease: BudgetLease, byte_count: int) -> BudgetLease:
        self._identity()
        amount = _bytes(byte_count, "retained bytes")
        with self._lock:
            self._next_lease += 1
            token = self._next_lease
            original = _validate_plan(lease.plan)
            if original.future_bytes is None:
                plan = replace(original)
            else:
                plan = replace(original, future_bytes=amount, retained_bytes=amount)
            ticket = BudgetLease(lease.owner, token, plan)
            allocation = _LeaseRecord(lease.owner.token, amount)
            # Dataclass/finalizer creation may reenter; transfer only a still-live
            # ticket with unchanged provenance and enough currently owned bytes.
            self._increment(lease.owner, None, lease.token)
            record = self._leases[lease.token]
            if lease.plan is not original or amount > record.reserved:
                raise InvalidAllocationPlan("Retained portion or provenance changed during transfer")
            record.reserved -= amount
            self._leases[token] = allocation
            self._record(record.owner).leases.add(token)
        return ticket

    def _release(self, token: int, pid: int, epoch: object) -> None:
        if pid != os.getpid() or epoch is not self.epoch:
            return
        with self._lock:
            record = self._leases.pop(token, None)
            if record is None:
                return
            self._future -= record.reserved
            if self._future < 0:
                raise RuntimeError("Negative future allocation accounting")
            owner = self._record(record.owner)
            owner.leases.discard(token)
            if owner.closing and not owner.leases:
                self._owners.pop(record.owner)
                self._owner_generation += 1

    def _close_owner(self, token: int, pid: int, epoch: object) -> None:
        if pid != os.getpid() or epoch is not self.epoch:
            return
        with self._lock:
            record = self._owners.get(token)
            if record is None:
                return
            if not record.closing:
                record.closing = True
                self._owner_generation += 1
            if not record.leases:
                self._owners.pop(token)

    def _set_limit(self, token: int, limit_bytes: int) -> None:
        self._identity()
        limit = _bytes(limit_bytes, "limit_bytes")
        with self._lock:
            record = self._record(token)
            if record.closing:
                raise StaleBudgetOwner("Closing owner cannot change its ceiling")
            if record.limit != limit:
                record.limit = limit
                self._owner_generation += 1

    def snapshot(self) -> dict[str, object]:
        self._identity()
        with self._lock:
            return {
                "pid": self.pid, "metric": "process_rss_sampled",
                "effective_limit_bytes": self._ceiling(),
                "owner_count": len(self._owners),
                "closing_owner_count": sum(record.closing for record in self._owners.values()),
                "lease_count": len(self._leases), "reserved_known_bytes": self._future,
                "high_water_known_bytes": self._high_water,
                "unprofiled_stages": tuple(sorted(self._unknown_stages)),
                "coverage_truncated": self._coverage_truncated,
                "physical_hard_cap": False, "os_enforcement": self.capability,
            }


class BudgetOwner:
    def __init__(
        self, registry: ProcessRegistry, token: int, usage_reader: Callable[[], object],
    ) -> None:
        self.registry = registry
        self.token = token
        self.pid = registry.pid
        self.epoch = registry.epoch
        self._usage_reader = usage_reader
        self._finalizer = weakref.finalize(
            self, registry._close_owner, token, self.pid, self.epoch,
        )

    def _identity(self) -> None:
        if self.pid != os.getpid() or self.epoch is not self.registry.epoch:
            raise StaleBudgetOwner("Inherited owner cannot allocate in a new process")

    def reserve(
        self, plan: AllocationPlan, *, deadline: Callable[[], None] | None = None,
    ) -> BudgetLease:
        self._identity()
        return self.registry._reserve(self, plan, deadline)

    def set_limit(self, limit_bytes: int) -> None:
        self._identity()
        self.registry._set_limit(self.token, limit_bytes)

    def close(self) -> None:
        self._finalizer()


class BudgetLease:
    def __init__(self, owner: BudgetOwner, token: int, plan: AllocationPlan) -> None:
        self.owner = owner
        self.token = token
        self.plan = plan
        self._finalizer = weakref.finalize(
            self, owner.registry._release, token, owner.pid, owner.epoch,
        )

    @property
    def reserved_known_bytes(self) -> int:
        self.owner._identity()
        with self.owner.registry._lock:
            record = self.owner.registry._leases.get(self.token)
            return 0 if record is None else record.reserved

    def grow(
        self, plan: AllocationPlan, *, deadline: Callable[[], None] | None = None,
    ) -> None:
        self.owner._identity()
        self.owner.registry._grow(self, plan, deadline)

    def split_retained(self, byte_count: int) -> BudgetLease:
        self.owner._identity()
        return self.owner.registry._split(self, byte_count)

    def close(self) -> None:
        self._finalizer()


_registry_lock = threading.RLock()
_registry_pid = os.getpid()
_registry: ProcessRegistry | None = None
_registry_initializing = False
_sessions = threading.local()


def _after_fork_child() -> None:
    global _registry_lock, _registry_pid, _registry, _registry_initializing, _sessions
    global _model_owners_lock, _model_owners, _retained_buffers_lock, _retained_buffers
    _registry_lock = threading.RLock()
    _registry_pid = os.getpid()
    _registry = None
    _registry_initializing = False
    _sessions = threading.local()
    _model_owners_lock = threading.RLock()
    _model_owners = {}
    _retained_buffers_lock = threading.RLock()
    _retained_buffers = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def get_process_registry() -> ProcessRegistry:
    global _registry_lock, _registry_pid, _registry, _registry_initializing
    pid = os.getpid()
    # Never acquire a lock inherited from a vanished parent thread.
    if pid != _registry_pid:
        _after_fork_child()
    with _registry_lock:
        if _registry is None:
            if _registry_initializing:
                raise MemoryAdmissionRefused("Process registry initialization is reentered")
            _registry_initializing = True
            try:
                _registry = ProcessRegistry()
            finally:
                _registry_initializing = False
        return _registry


def resolve_allocation_plan(key: ProfileKey) -> AllocationPlan:
    registry = getattr(_sessions, "resolution_registry", None)
    return (registry or get_process_registry()).resolve_profile(key)


def automatic_budget_bytes() -> int:
    try:
        import psutil
        physical = psutil.virtual_memory().total
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError, TypeError) as exc:
        raise MemoryAdmissionRefused("Local automatic physical-memory sensor unavailable") from exc
    if type(physical) is not int or not 0 < physical <= _MAX_BYTES:
        raise MemoryAdmissionRefused("Local physical-memory sensor invalid")
    budget = physical // 4
    if budget:
        # Same small-host floor as the model server (float32 Qwen on macOS CPU).
        floor = (4 if sys.platform == "darwin" else 3) * 1024**3
        budget = min(8 * 1024**3, max(budget, min(physical // 2, floor)))
    if not budget:
        raise MemoryAdmissionRefused("Local automatic budget is not positive")
    return budget


def new_local_owner(kind: str) -> BudgetOwner:
    return get_process_registry().register_owner(automatic_budget_bytes(), kind=kind)


def configured_local_owner(policy: object, *, kind: str) -> BudgetOwner:
    validate_local_policy(policy)
    if type(kind) is not str or not kind:
        raise InvalidAllocationPlan("Local owner kind must be a nonempty exact string")
    return new_local_owner(kind)


def _resolved_plan(key: ProfileKey, known_bytes: int, retained_bytes: int) -> AllocationPlan:
    known = _bytes(known_bytes, "known bytes")
    retained = _bytes(retained_bytes, "retained bytes")
    _validate_key(key)
    plan = _validate_plan(resolve_allocation_plan(key))
    if plan.key != key:
        raise InvalidAllocationPlan("Resolver returned a mismatched allocation plan")
    if plan.future_bytes is None:
        if not known:
            return plan
        return AllocationPlan(
            key, known, retained, unknown_components=plan.unknown_components,
            reason=plan.reason,
        )
    if plan.future_bytes < known:
        raise InvalidAllocationPlan("Profile demand is below the enumerated named copies")
    return replace(plan, retained_bytes=max(retained, plan.retained_bytes or 0))


class AllocationSession:
    def __init__(self, owner: BudgetOwner) -> None:
        self.owner = owner
        self._leases: dict[str, BudgetLease] = {}
        self._closed = False

    def charge(
        self, key: ProfileKey, *, known_bytes: int = 0, retained_bytes: int = 0,
        deadline: Callable[[], None] | None = None, group: str | None = None,
    ) -> BudgetLease:
        if self._closed:
            raise StaleBudgetOwner("Allocation session is closed")
        key = self.owner.registry.enrich_key(key)
        previous = getattr(_sessions, "resolution_registry", None)
        _sessions.resolution_registry = self.owner.registry
        try:
            plan = _resolved_plan(key, known_bytes, retained_bytes)
        finally:
            _sessions.resolution_registry = previous
        label = group or key.stage
        lease = self._leases.get(label)
        if lease is None:
            lease = self.owner.reserve(plan, deadline=deadline)
            self._leases[label] = lease
        else:
            lease.grow(plan, deadline=deadline)
        return lease

    def retain_buffer(self, value: object, *, deadline: Callable[[], None] | None = None) -> None:
        candidate = value
        seen: set[int] = set()
        while id(candidate) not in seen:
            seen.add(id(candidate))
            base = getattr(candidate, "base", None)
            if base is None:
                break
            try:
                weakref.ref(base)
            except TypeError:
                break
            candidate = base
        amount = getattr(candidate, "nbytes", None)
        if type(amount) is not int or amount < 0:
            # A native buffer outside ndarray-owned bytes needs explicit profile coverage.
            self.charge(ProfileKey("retained external buffer", "retained_result"),
                        deadline=deadline)
            return
        token = id(candidate)
        get_process_registry()  # Reset every inherited process-local lock before use.
        with _retained_buffers_lock:
            entry = _retained_buffers.get(token)
            if entry is not None and entry[0]() is candidate:
                if any(lease.owner is self.owner for lease in entry[1]):
                    return
                existing = True
            else:
                existing = False
        key = ProfileKey("retained result", "retained_result", native_tensor_shape=(amount,))
        lease = self._leases.get("result assembly")
        if lease is None or lease.reserved_known_bytes < amount:
            lease = self.charge(key, known_bytes=amount, retained_bytes=amount,
                                deadline=deadline, group="retained result")
        portion = lease.split_retained(0 if existing else amount)

        def retired(reference: weakref.ReferenceType[object]) -> None:
            with _retained_buffers_lock:
                found = _retained_buffers.get(token)
                leases = _retained_buffers.pop(token)[1] if found is not None and found[0] is reference else ()
            for ticket in leases:
                ticket.close()

        try:
            reference = weakref.ref(candidate, retired)
        except TypeError:
            portion.close()
            raise InvalidAllocationPlan("Result owner cannot retain its byte lease")
        spare = portion.split_retained(0)
        prepared_entry = (reference, [portion])
        with _retained_buffers_lock:
            found = _retained_buffers.get(token)
            if found is not None and found[0]() is candidate:
                found[1].append(spare)
                unused = portion
            else:
                _retained_buffers[token] = prepared_entry
                unused = spare
        unused.close()

    def defer_to_exception(self, error: BaseException) -> None:
        # The exception already owns its traceback; the ticket must follow that lifetime.
        error.__dict__.setdefault("_truememory_allocation_sessions", []).append(self)

    def release(self, label: str) -> None:
        """Return one finished stage's reservation before the session closes."""
        lease = self._leases.pop(label, None)
        if lease is not None:
            lease.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        leases, self._leases = self._leases, {}
        for lease in leases.values():
            lease.close()


def current_session() -> AllocationSession | None:
    stack = getattr(_sessions, "stack", ())
    return stack[-1] if stack else None


@contextmanager
def cooperative_scope() -> Iterator[AllocationSession | None]:
    session = current_session()
    if session is not None:
        yield session
        return
    with allocation_scope(get_process_registry().active_owner()) as session:
        yield session


@contextmanager
def allocation_scope(owner: BudgetOwner | None) -> Iterator[AllocationSession | None]:
    if owner is None:
        yield None
        return
    stack = getattr(_sessions, "stack", None)
    if stack is None:
        stack = _sessions.stack = []
    if stack and stack[-1].owner is owner:
        yield stack[-1]
        return
    session = AllocationSession(owner)
    stack.append(session)
    failed = False
    try:
        yield session
    except BaseException as error:
        failed = True
        session.defer_to_exception(error)
        raise
    finally:
        stack.pop()
        if not failed:
            session.close()


_model_owners_lock = threading.RLock()
_model_owners: dict[int, tuple[weakref.ReferenceType[object], BudgetOwner]] = {}
_retained_buffers_lock = threading.RLock()
_retained_buffers: dict[int, tuple[weakref.ReferenceType[object], list[BudgetLease]]] = {}


def bind_model_owner(model: object, owner: BudgetOwner) -> None:
    get_process_registry()
    token = id(model)

    def retired(reference: weakref.ReferenceType[object]) -> None:
        with _model_owners_lock:
            found = _model_owners.get(token)
            if found is not None and found[0] is reference:
                _model_owners.pop(token)
        owner.close()

    try:
        reference = weakref.ref(model, retired)
    except TypeError:
        owner.close()
        raise InvalidAllocationPlan("Local model cannot retain an automatic owner")
    prepared_entry = (reference, owner)
    with _model_owners_lock:
        duplicate = token in _model_owners
        if not duplicate:
            _model_owners[token] = prepared_entry
    if duplicate:
        owner.close()
        raise InvalidAllocationPlan("Local model already has an owner")


def model_owner(model: object) -> BudgetOwner | None:
    get_process_registry()
    with _model_owners_lock:
        found = _model_owners.get(id(model))
        return found[1] if found is not None and found[0]() is model else None


@contextmanager
def model_operation_scope(
    model: object, key: ProfileKey, deadline: Callable[[], object] | None = None,
) -> Iterator[AllocationSession | None]:
    _validate_key(key)
    if deadline is not None and not callable(deadline):
        raise InvalidAllocationPlan("Operation deadline must be callable or None")
    owner = model_owner(model)
    scope = allocation_scope(owner) if owner is not None else cooperative_scope()
    with scope as memory:
        if memory is not None:
            memory.charge(key, deadline=deadline)
        yield memory


def _native_arguments(native_kwargs: dict[str, object] | None) -> dict[str, object]:
    if native_kwargs is None:
        return {}
    if type(native_kwargs) is not dict or any(type(key) is not str for key in native_kwargs):
        raise InvalidAllocationPlan("Native arguments must be an exact dictionary with exact string keys")
    return native_kwargs


def budgeted_encode_with_model_ownership(
    model: object, texts: object, key: ProfileKey, *,
    native_kwargs: dict[str, object] | None = None,
    deadline: Callable[[], object] | None = None,
) -> object:
    arguments = _native_arguments(native_kwargs)
    with model_operation_scope(model, key, deadline) as memory:
        from truememory.mps_utils import encode_with_model_ownership
        values = encode_with_model_ownership(model, texts, **arguments)
        if memory is not None:
            memory.retain_buffer(values, deadline=deadline)
        return values


def budgeted_encode_with_mps_fallback(
    model: object, texts: object, key: ProfileKey, *,
    native_kwargs: dict[str, object] | None = None,
    deadline: Callable[[], object] | None = None,
) -> object:
    arguments = _native_arguments(native_kwargs)
    with model_operation_scope(model, key, deadline) as memory:
        from truememory.mps_utils import encode_with_mps_fallback
        values = encode_with_mps_fallback(model, texts, **arguments)
        if memory is not None:
            memory.retain_buffer(values, deadline=deadline)
        return values
