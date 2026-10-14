"""Opt-in Linux charged-domain launch, separate from physical RSS admission.

The child enters its manager-owned cgroup before the interpreter starts. This
module uses only the standard library until that child has passed its barrier.
Kernel ceiling C is immutable; it does not track the cooperative RSS budget B.
"""
from __future__ import annotations

from collections.abc import Mapping
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import select
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from types import MappingProxyType


_MAX = 2**63 - 1
_READ_BOUND = 65536
_CGROUP2_MAGIC = 0x63677270
_METRIC = "linux_cgroup_charged_domain"
_FLAG = "TRUEMEMORY_REQUIRE_PROCESS_MEMORY_CAP"
_GENERATION_ENV = "TRUEMEMORY_MODEL_SERVER_GENERATION"
_MANIFEST_ENV = "TRUEMEMORY_CAP_MANIFEST"
_IDENTITY_ENV = "TRUEMEMORY_CAP_IDENTITY_SHA256"
_BARRIER_ENV = "TRUEMEMORY_CAP_BARRIER_FD"
_INTERNAL_ENV = (_GENERATION_ENV, "TRUEMEMORY_CAP_CHILD", _MANIFEST_ENV, _IDENTITY_ENV, _BARRIER_ENV)
_HEX = re.compile(r"^[0-9a-f]+$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_UNIT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@:-]*$")


class EnforcementUnavailable(RuntimeError):
    """Required enforcement or complete owned evidence could not be proved."""


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or not (1 if positive else 0) <= value <= _MAX:
        raise ValueError(f"{name} must be an exact {'positive' if positive else 'nonnegative'} signed64 integer")
    return value


def _literal(value: object, expected: object, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"Invalid {name}")


def _literal_packet(value: object, expected: dict[str, object], name: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value) or set(value) != set(expected):
        raise EnforcementUnavailable(f"Invalid {name} schema")
    for key, literal in expected.items():
        _literal(value[key], literal, name + " " + key)
    return value


def _hex(value: object, length: int, name: str) -> str:
    if type(value) is not str or len(value) != length or _HEX.fullmatch(value) is None:
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def _text(value: object, name: str, *, bound: int = 256) -> str:
    if type(value) is not str or not value or len(value) > bound or "\x00" in value:
        raise ValueError(f"Invalid {name}")
    return value


def _relative(value: object, name: str) -> str:
    _text(value, name, bound=4096)
    if not value.startswith("/") or value == "/" or any(part in ("", ".", "..") for part in value[1:].split("/")):
        raise ValueError(f"{name} must be a canonical cgroup relative path")
    return value


def _deadline(value: object) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("deadline must be a finite exact float")
    if value <= time.monotonic():
        raise TimeoutError("Owned Linux launch deadline expired")
    return value


def _complete_fields(value: object, cls: type) -> None:
    if type(value) is not cls:
        raise TypeError(f"Expected exact {cls.__name__}")
    if any(name not in vars(value) for name in cls.__dataclass_fields__):
        raise ValueError(f"Incomplete {cls.__name__}")


def _left(deadline: float) -> float:
    _deadline(deadline)
    return deadline - time.monotonic()


_STARTUP_MANAGER_PATH = "/org/freedesktop/systemd1"
_STARTUP_MANAGER_INTERFACE = "org.freedesktop.systemd1.Manager"
_STARTUP_UNIT_INTERFACE = "org.freedesktop.systemd1.Unit"
_STARTUP_PROPERTIES_INTERFACE = "org.freedesktop.DBus.Properties"
_STARTUP_METADATA_DOMAIN = "truememory.startup.redacted_create_structure.v1"
_STARTUP_EVIDENCE_DOMAIN = "truememory.startup.typed_evidence.v1"
_STARTUP_PHASES = ("prepared", "attempted_unknown", "reply_bound_pending_invocation", "invocation_observed", "correlated_done", "unknown")
_STARTUP_REASONS = ("expired_unsent", "expired_unknown", "transport_gap", "transport_disconnected", "manager_owner_changed", "evidence_overflow", "malformed_evidence", "unsupported_signature", "conflicting_cookie", "conflicting_reply", "conflicting_job", "conflicting_unit", "ambiguous_invocation", "manager_error_unknown", "job_outcome_non_done", "replayed_native_serial", "preexisting_unit")
_STARTUP_VARIANTS = ("s", "b", "u", "t", "ay", "as", "(uo)")
_STARTUP_METADATA_SIGNATURES = _STARTUP_VARIANTS + ("a(sasb)",)
_STARTUP_KINDS = ("create_return", "create_error", "job_new", "job_removed", "unit_new", "unit_removed", "unit_properties")
_STARTUP_ENVELOPE_FIELDS = ("version", "ordinal", "kind", "sender", "destination", "serial", "reply_cookie", "path", "interface", "member", "signature", "payload", "error_name", "error_body_redacted")
_STARTUP_NATIVE_PROPERTIES = (("ActiveState", "s"), ("FreezerState", "s"), ("SubState", "s"), ("StateChangeTimestamp", "t"), ("StateChangeTimestampMonotonic", "t"), ("InactiveExitTimestamp", "t"), ("InactiveExitTimestampMonotonic", "t"), ("ActiveEnterTimestamp", "t"), ("ActiveEnterTimestampMonotonic", "t"), ("ActiveExitTimestamp", "t"), ("ActiveExitTimestampMonotonic", "t"), ("InactiveEnterTimestamp", "t"), ("InactiveEnterTimestampMonotonic", "t"), ("ConditionTimestamp", "t"), ("ConditionTimestampMonotonic", "t"), ("AssertTimestamp", "t"), ("AssertTimestampMonotonic", "t"), ("Job", "(uo)"), ("ConditionResult", "b"), ("AssertResult", "b"), ("InvocationID", "ay"))


@dataclass(frozen=True, slots=True)
class StartupCreateStructure:
    version: int
    unit_name: str
    unit_path: str
    job_mode: str
    property_signatures: tuple[tuple[str, str], ...]
    environment_names: tuple[str, ...]
    auxiliary_units: tuple

    def __post_init__(self) -> None:
        _startup_validate_metadata(self)


@dataclass(frozen=True, slots=True)
class StartupBusContext:
    version: int
    bus_id: str
    manager_unique_name: str
    client_unique_name: str
    manager_uid: int

    def __post_init__(self) -> None:
        _startup_validate_context(self)


@dataclass(frozen=True, slots=True)
class StartupProperty:
    name: str
    signature: str
    value: object

    def __post_init__(self) -> None:
        _startup_validate_property(self)


@dataclass(frozen=True, slots=True)
class StartupMessageEnvelope:
    version: int
    ordinal: int
    kind: str
    sender: str
    destination: str | None
    serial: int
    reply_cookie: int | None
    path: str | None
    interface: str | None
    member: str | None
    signature: str
    payload: tuple
    error_name: str | None
    error_body_redacted: bool

    def __post_init__(self) -> None:
        _startup_validate_envelope(self)


@dataclass(frozen=True, slots=True)
class StartupCorrelationState:
    version: int
    context: StartupBusContext
    metadata: StartupCreateStructure
    redacted_request_metadata_sha256: str
    original_deadline: float
    last_now: float
    phase: str
    send_attempted: bool
    request_cookie: int | None
    envelopes: tuple[StartupMessageEnvelope, ...]
    canonical_evidence_bytes: int
    reason: str | None

    def __post_init__(self) -> None:
        _startup_validate_state(self)


@dataclass(frozen=True, slots=True)
class StartupInvocationBinding:
    version: int
    context: StartupBusContext
    unit_name: str
    unit_path: str
    request_cookie: int
    job_id: int
    job_path: str
    invocation_id: bytes
    redacted_request_metadata_sha256: str
    reply_evidence_sha256: str
    job_new_evidence_sha256: str
    joint_evidence_sha256: str
    job_removed_evidence_sha256: str | None
    job_result: str | None
    unit_new_evidence_sha256: str

    def __post_init__(self) -> None:
        _startup_validate_binding(self)


def _startup_fail() -> None:
    raise EnforcementUnavailable("Invalid bounded startup correlation evidence")


def _startup_fields(value: object, cls: type) -> None:
    if type(value) is not cls:
        _startup_fail()
    for name in cls.__dataclass_fields__:
        try:
            object.__getattribute__(value, name)
        except AttributeError:
            _startup_fail()


def _startup_integer(value: object, maximum: int, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _startup_fail()
    return value


def _startup_float(value: object) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        _startup_fail()
    return value


def _startup_text(value: object, bound: int, *, empty: bool = False, ascii_only: bool = False) -> str:
    if type(value) is not str or len(value) > bound or (not empty and not value) or "\x00" in value:
        _startup_fail()
    if ascii_only and not value.isascii():
        _startup_fail()
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        _startup_fail()
    if len(encoded) > bound:
        _startup_fail()
    return value


def _startup_hex(value: object, length: int) -> str:
    text = _startup_text(value, length, ascii_only=True)
    if len(text) != length or re.fullmatch(r"[0-9a-f]+", text) is None:
        _startup_fail()
    return text


def _startup_name(value: object) -> str:
    text = _startup_text(value, 128, ascii_only=True)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text) is None:
        _startup_fail()
    return text


def _startup_bus_name(value: object) -> str:
    text = _startup_text(value, 255, ascii_only=True)
    if re.fullmatch(r":[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+", text) is None:
        _startup_fail()
    return text


def _startup_unit_path(value: object) -> str:
    name = _startup_text(value, 255, ascii_only=True)
    if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*", name) is None or not name.endswith((".service", ".slice")):
        _startup_fail()
    label = "".join(char if char.isalnum() and not (index == 0 and char.isdigit()) else "_" + format(ord(char), "02x") for index, char in enumerate(name))
    return _STARTUP_MANAGER_PATH + "/unit/" + label


def _startup_job_path(value: object, *, zero: bool = False) -> int:
    text = _startup_text(value, 1024, ascii_only=True)
    if zero and text == "/":
        return 0
    prefix = _STARTUP_MANAGER_PATH + "/job/"
    if not text.startswith(prefix):
        _startup_fail()
    number = text[len(prefix):]
    if len(number) > 10 or re.fullmatch(r"[1-9][0-9]*", number) is None:
        _startup_fail()
    return _startup_integer(int(number), 2**32 - 1, minimum=1)


def _startup_object_path(value: object) -> str:
    text = _startup_text(value, 1024, ascii_only=True)
    if text != "/" and re.fullmatch(r"/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+", text) is None:
        _startup_fail()
    return text


def _startup_validate_metadata(value: object) -> StartupCreateStructure:
    _startup_fields(value, StartupCreateStructure)
    if type(value.version) is not int or value.version != 1 or type(value.job_mode) is not str or value.job_mode != "fail":
        _startup_fail()
    expected = _startup_unit_path(value.unit_name)
    if _startup_text(value.unit_path, 1024, ascii_only=True) != expected:
        _startup_fail()
    pairs = value.property_signatures
    if type(pairs) is not tuple or len(pairs) > 64:
        _startup_fail()
    names = []
    for pair in pairs:
        if type(pair) is not tuple or len(pair) != 2:
            _startup_fail()
        name = _startup_name(pair[0])
        signature = _startup_text(pair[1], 64, ascii_only=True)
        if signature not in _STARTUP_METADATA_SIGNATURES or (name == "Environment" and signature != "as"):
            _startup_fail()
        names.append(name)
    if names != sorted(names) or len(set(names)) != len(names):
        _startup_fail()
    if type(value.environment_names) is not tuple or len(value.environment_names) > 256:
        _startup_fail()
    for name in value.environment_names:
        text = _startup_text(name, 256)
        if "=" in text:
            _startup_fail()
    if tuple(sorted(value.environment_names)) != value.environment_names or len(set(value.environment_names)) != len(value.environment_names):
        _startup_fail()
    if value.environment_names and ("Environment", "as") not in pairs:
        _startup_fail()
    if type(value.auxiliary_units) is not tuple or value.auxiliary_units:
        _startup_fail()
    return value


def _startup_validate_context(value: object) -> StartupBusContext:
    _startup_fields(value, StartupBusContext)
    if type(value.version) is not int or value.version != 1:
        _startup_fail()
    _startup_hex(value.bus_id, 32)
    _startup_bus_name(value.manager_unique_name)
    _startup_bus_name(value.client_unique_name)
    if value.manager_unique_name == value.client_unique_name:
        _startup_fail()
    _startup_integer(value.manager_uid, 2**32 - 2)
    return value


def _startup_validate_property(value: object) -> StartupProperty:
    _startup_fields(value, StartupProperty)
    name = _startup_name(value.name)
    signature = _startup_text(value.signature, 64, ascii_only=True)
    if signature not in _STARTUP_VARIANTS:
        _startup_fail()
    for native_name, native_signature in _STARTUP_NATIVE_PROPERTIES:
        if name == native_name and signature != native_signature:
            _startup_fail()
    item = value.value
    if signature == "s":
        _startup_text(item, 1024, empty=True)
    elif signature == "b":
        if type(item) is not bool:
            _startup_fail()
    elif signature in ("u", "t"):
        _startup_integer(item, 2**(32 if signature == "u" else 64) - 1)
    elif signature == "ay":
        if type(item) is not bytes or len(item) > 4096 or (name == "InvocationID" and len(item) not in (0, 16)):
            _startup_fail()
    elif signature == "as":
        if type(item) is not tuple or len(item) > 64:
            _startup_fail()
        for text in item:
            _startup_text(text, 1024, empty=True)
    else:
        if type(item) is not tuple or len(item) != 2:
            _startup_fail()
        number = _startup_integer(item[0], 2**32 - 1)
        _startup_object_path(item[1])
        if name == "Job" and _startup_job_path(item[1], zero=True) != number:
            _startup_fail()
    return value


def _startup_validate_envelope(value: object) -> StartupMessageEnvelope:
    _startup_fields(value, StartupMessageEnvelope)
    if type(value.version) is not int or value.version != 1:
        _startup_fail()
    _startup_integer(value.ordinal, 2**64 - 1, minimum=1)
    _startup_integer(value.serial, 2**32 - 1, minimum=1)
    _startup_bus_name(value.sender)
    if value.destination is not None:
        _startup_bus_name(value.destination)
    kind = _startup_text(value.kind, 64, ascii_only=True)
    if kind not in _STARTUP_KINDS or type(value.payload) is not tuple or type(value.error_body_redacted) is not bool:
        _startup_fail()
    signature = _startup_text(value.signature, 64, empty=True, ascii_only=True)
    payload = value.payload
    if kind in ("create_return", "create_error"):
        _startup_integer(value.reply_cookie, 2**32 - 1, minimum=1)
        if value.destination is None or any(item is not None for item in (value.path, value.interface, value.member)):
            _startup_fail()
        if kind == "create_return":
            if signature != "o" or len(payload) != 1 or value.error_name is not None or value.error_body_redacted:
                _startup_fail()
            _startup_job_path(payload[0])
        else:
            if signature not in ("", "s") or payload or value.error_body_redacted is not True:
                _startup_fail()
            error = _startup_text(value.error_name, 255, ascii_only=True)
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", error) is None:
                _startup_fail()
        return value
    if value.reply_cookie is not None or value.error_name is not None or value.error_body_redacted:
        _startup_fail()
    for header in (value.path, value.interface, value.member):
        _startup_text(header, 1024, ascii_only=True)
    _startup_object_path(value.path)
    if kind == "unit_properties":
        if not value.path.startswith(_STARTUP_MANAGER_PATH + "/unit/") or value.interface != _STARTUP_PROPERTIES_INTERFACE or value.member != "PropertiesChanged" or signature != "sa{sv}as" or len(payload) != 3:
            _startup_fail()
        if type(payload[0]) is not str or payload[0] != _STARTUP_UNIT_INTERFACE:
            _startup_fail()
        changed, invalidated = payload[1:]
        if type(changed) is not tuple or len(changed) > 64 or type(invalidated) is not tuple or len(invalidated) > 64:
            _startup_fail()
        changed_names = []
        for item in changed:
            _startup_validate_property(item)
            changed_names.append(item.name)
        for name in invalidated:
            _startup_name(name)
        if len(set(changed_names)) != len(changed_names) or len(set(invalidated)) != len(invalidated):
            _startup_fail()
    else:
        member, expected_signature, arity = {"job_new": ("JobNew", "uos", 3), "job_removed": ("JobRemoved", "uoss", 4), "unit_new": ("UnitNew", "so", 2), "unit_removed": ("UnitRemoved", "so", 2)}[kind]
        if value.path != _STARTUP_MANAGER_PATH or value.interface != _STARTUP_MANAGER_INTERFACE or value.member != member or signature != expected_signature or len(payload) != arity:
            _startup_fail()
        if kind in ("job_new", "job_removed"):
            number = _startup_integer(payload[0], 2**32 - 1, minimum=1)
            if _startup_job_path(payload[1]) != number:
                _startup_fail()
            _startup_unit_path(payload[2])
            if kind == "job_removed":
                _startup_text(payload[3], 1024)
        elif _startup_text(payload[1], 1024, ascii_only=True) != _startup_unit_path(payload[0]):
            _startup_fail()
    return value


def _startup_metadata_projection(value: StartupCreateStructure) -> dict[str, object]:
    return {"domain": _STARTUP_METADATA_DOMAIN, "version": value.version, "path": _STARTUP_MANAGER_PATH, "interface": _STARTUP_MANAGER_INTERFACE, "member": "StartTransientUnit", "signature": "ssa(sv)a(sa(sv))", "unit_name": value.unit_name, "unit_path": value.unit_path, "job_mode": value.job_mode, "property_signatures": value.property_signatures, "environment_names": value.environment_names, "auxiliary_units": value.auxiliary_units}


def _startup_property_projection(value: StartupProperty) -> dict[str, object]:
    item = {"type": "ay", "hex": value.value.hex()} if value.signature == "ay" else value.value
    return {"name": value.name, "signature": value.signature, "value": item}


def _startup_envelope_projection(value: StartupMessageEnvelope) -> dict[str, object]:
    payload = value.payload
    if value.kind == "unit_properties":
        payload = (payload[0], tuple(_startup_property_projection(item) for item in payload[1]), payload[2])
    return {"domain": _STARTUP_EVIDENCE_DOMAIN, "version": value.version, "ordinal": value.ordinal, "kind": value.kind, "sender": value.sender, "destination": value.destination, "serial": value.serial, "reply_cookie": value.reply_cookie, "path": value.path, "interface": value.interface, "member": value.member, "signature": value.signature, "payload": payload, "error_name": value.error_name, "error_body_redacted": value.error_body_redacted}


def _startup_canonical(value: dict[str, object], bound: int) -> bytes | None:
    chunks = []
    size = 0
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    for piece in encoder.iterencode(value):
        chunk = piece.encode("utf-8")
        size += len(chunk)
        if size > bound:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def redacted_create_metadata_sha256(metadata: StartupCreateStructure) -> str:
    _startup_validate_metadata(metadata)
    canonical = _startup_canonical(_startup_metadata_projection(metadata), 16384)
    if canonical is None:
        _startup_fail()
    return hashlib.sha256(canonical).hexdigest()


def parse_startup_envelope(value: object) -> StartupMessageEnvelope:
    if type(value) is not dict or len(value) != len(_STARTUP_ENVELOPE_FIELDS) or any(type(name) is not str for name in value) or set(value) != set(_STARTUP_ENVELOPE_FIELDS):
        _startup_fail()
    return StartupMessageEnvelope(*(value[name] for name in _STARTUP_ENVELOPE_FIELDS))


def startup_envelope_evidence_sha256(envelope: StartupMessageEnvelope) -> str:
    _startup_validate_envelope(envelope)
    canonical = _startup_canonical(_startup_envelope_projection(envelope), 16384)
    if canonical is None:
        _startup_fail()
    return hashlib.sha256(canonical).hexdigest()


def _startup_scan(context: StartupBusContext, metadata: StartupCreateStructure, cookie: int | None, envelopes: tuple[StartupMessageEnvelope, ...]) -> tuple[str, str | None, dict[str, object]]:
    facts: dict[str, object] = {}
    seen_serials = set()
    observed_job = None
    observed_invocation = None
    for index, envelope in enumerate(envelopes, 1):
        if envelope.ordinal != index:
            return "unknown", "transport_gap", facts
        if envelope.sender != context.manager_unique_name:
            return "unknown", "manager_owner_changed", facts
        if envelope.serial in seen_serials:
            return "unknown", "replayed_native_serial", facts
        seen_serials.add(envelope.serial)
        if envelope.destination not in (None, context.client_unique_name):
            return "unknown", "conflicting_reply", facts
        kind, payload = envelope.kind, envelope.payload
        if kind in ("unit_new", "unit_removed"):
            if payload != (metadata.unit_name, metadata.unit_path):
                return "unknown", "conflicting_unit", facts
            if kind == "unit_removed" or "unit_new" in facts or "job_new" in facts:
                return "unknown", "conflicting_unit", facts
            facts["unit_new"] = envelope
        elif kind in ("create_return", "create_error"):
            if "reply" in facts:
                return "unknown", "conflicting_reply", facts
            facts["reply"] = envelope
            if cookie is not None and envelope.reply_cookie != cookie:
                return "unknown", "conflicting_reply", facts
            if kind == "create_error":
                if cookie is not None:
                    return "unknown", "manager_error_unknown", facts
                continue
            number = _startup_job_path(payload[0])
            job = (number, payload[0])
            if observed_job is not None and job != observed_job:
                return "unknown", "conflicting_job", facts
            observed_job = job
        elif kind in ("job_new", "job_removed"):
            if payload[2] != metadata.unit_name:
                return "unknown", "conflicting_unit", facts
            job = payload[:2]
            if observed_job is not None and job != observed_job:
                return "unknown", "conflicting_job", facts
            observed_job = job
            if kind == "job_new":
                if "unit_new" not in facts:
                    return "unknown", "preexisting_unit", facts
                if "job_new" in facts:
                    return "unknown", "conflicting_job", facts
                facts["job_new"] = envelope
            else:
                if "job_removed" in facts:
                    return "unknown", "conflicting_job", facts
                facts["job_removed"] = envelope
                if payload[3] != "done":
                    return "unknown", "job_outcome_non_done", facts
        else:
            if envelope.path != metadata.unit_path:
                return "unknown", "conflicting_unit", facts
            properties = {item.name: item for item in payload[1]}
            job_property = properties.get("Job")
            invocation_property = properties.get("InvocationID")
            job = None if job_property is None or job_property.value[0] == 0 else job_property.value
            invocation = None if invocation_property is None or not invocation_property.value or not any(invocation_property.value) else invocation_property.value
            if job is not None:
                if observed_job is not None and observed_job != job:
                    return "unknown", "conflicting_job", facts
                observed_job = job
            if invocation is not None:
                if observed_invocation is not None and observed_invocation != invocation:
                    return "unknown", "ambiguous_invocation", facts
                observed_invocation = invocation
            if job is not None and invocation is not None and "Job" not in payload[2] and "InvocationID" not in payload[2]:
                if "joint" not in facts:
                    facts["joint"] = envelope
                    facts["invocation_id"] = invocation
    reply = facts.get("reply")
    if cookie is None or reply is None or reply.kind != "create_return":
        return "attempted_unknown", None, facts
    if all(name in facts for name in ("unit_new", "job_new", "joint")):
        return ("correlated_done" if "job_removed" in facts else "invocation_observed"), None, facts
    return "reply_bound_pending_invocation", None, facts


def _startup_validate_state(value: object) -> StartupCorrelationState:
    _startup_fields(value, StartupCorrelationState)
    if type(value.version) is not int or value.version != 1:
        _startup_fail()
    _startup_validate_context(value.context)
    _startup_validate_metadata(value.metadata)
    if _startup_hex(value.redacted_request_metadata_sha256, 64) != redacted_create_metadata_sha256(value.metadata):
        _startup_fail()
    _startup_float(value.original_deadline)
    _startup_float(value.last_now)
    if value.original_deadline == 0.0 or type(value.phase) is not str or value.phase not in _STARTUP_PHASES or type(value.send_attempted) is not bool:
        _startup_fail()
    if value.request_cookie is not None:
        _startup_integer(value.request_cookie, 2**32 - 1, minimum=1)
        if not value.send_attempted:
            _startup_fail()
    if type(value.envelopes) is not tuple or len(value.envelopes) > 64:
        _startup_fail()
    if not value.send_attempted and value.envelopes:
        _startup_fail()
    total = 0
    for envelope in value.envelopes:
        _startup_validate_envelope(envelope)
        canonical = _startup_canonical(_startup_envelope_projection(envelope), 16384)
        if canonical is None:
            _startup_fail()
        total += len(canonical)
        if total > 65536:
            _startup_fail()
    _startup_integer(value.canonical_evidence_bytes, 65536)
    if value.canonical_evidence_bytes != total:
        _startup_fail()
    if value.phase == "unknown":
        if type(value.reason) is not str or value.reason not in _STARTUP_REASONS:
            _startup_fail()
        if value.reason in ("expired_unsent", "expired_unknown") and (value.last_now < value.original_deadline or (value.reason == "expired_unknown") != value.send_attempted):
            _startup_fail()
    else:
        if value.reason is not None or value.last_now >= value.original_deadline:
            _startup_fail()
        if not value.send_attempted:
            if value.phase != "prepared" or value.envelopes or value.request_cookie is not None:
                _startup_fail()
        else:
            phase, reason, _ = _startup_scan(value.context, value.metadata, value.request_cookie, value.envelopes)
            if phase != value.phase or reason is not None:
                _startup_fail()
    return value


def _startup_validate_binding(value: object) -> StartupInvocationBinding:
    _startup_fields(value, StartupInvocationBinding)
    if type(value.version) is not int or value.version != 1:
        _startup_fail()
    _startup_validate_context(value.context)
    if _startup_text(value.unit_path, 1024, ascii_only=True) != _startup_unit_path(value.unit_name):
        _startup_fail()
    _startup_integer(value.request_cookie, 2**32 - 1, minimum=1)
    _startup_integer(value.job_id, 2**32 - 1, minimum=1)
    if _startup_job_path(value.job_path) != value.job_id or type(value.invocation_id) is not bytes or len(value.invocation_id) != 16 or not any(value.invocation_id):
        _startup_fail()
    for name in ("redacted_request_metadata_sha256", "reply_evidence_sha256", "job_new_evidence_sha256", "joint_evidence_sha256", "unit_new_evidence_sha256"):
        _startup_hex(getattr(value, name), 64)
    if value.job_removed_evidence_sha256 is None:
        if value.job_result is not None:
            _startup_fail()
    else:
        _startup_hex(value.job_removed_evidence_sha256, 64)
        if type(value.job_result) is not str or value.job_result != "done":
            _startup_fail()
    return value


def _startup_now(state: StartupCorrelationState, now: object) -> float:
    _startup_validate_state(state)
    moment = _startup_float(now)
    if moment < state.last_now:
        _startup_fail()
    return moment


def _startup_state(state: StartupCorrelationState, now: float, *, phase: str, reason: str | None = None, attempted: bool | None = None, cookie: int | None = None, envelopes: tuple[StartupMessageEnvelope, ...] | None = None, size: int | None = None) -> StartupCorrelationState:
    return StartupCorrelationState(1, state.context, state.metadata, state.redacted_request_metadata_sha256, state.original_deadline, now, phase, state.send_attempted if attempted is None else attempted, state.request_cookie if cookie is None else cookie, state.envelopes if envelopes is None else envelopes, state.canonical_evidence_bytes if size is None else size, reason)


def _startup_unknown(state: StartupCorrelationState, now: float, reason: str) -> StartupCorrelationState:
    return _startup_state(state, now, phase="unknown", reason=state.reason if state.phase == "unknown" else reason)


def _startup_expired(state: StartupCorrelationState, now: float) -> StartupCorrelationState | None:
    if state.phase == "unknown":
        return _startup_unknown(state, now, state.reason)
    if now >= state.original_deadline:
        return _startup_unknown(state, now, "expired_unknown" if state.send_attempted else "expired_unsent")
    return None


def new_startup_correlation(context: StartupBusContext, metadata: StartupCreateStructure, *, deadline: float, now: float) -> StartupCorrelationState:
    _startup_validate_context(context)
    _startup_validate_metadata(metadata)
    moment, end = _startup_float(now), _startup_float(deadline)
    if end <= moment:
        _startup_fail()
    return StartupCorrelationState(1, context, metadata, redacted_create_metadata_sha256(metadata), end, moment, "prepared", False, None, (), 0, None)


def mark_startup_send_attempt(state: StartupCorrelationState, *, now: float) -> StartupCorrelationState:
    moment = _startup_now(state, now)
    terminal = _startup_expired(state, moment)
    if terminal is not None:
        return terminal
    if state.send_attempted:
        _startup_fail()
    return _startup_state(state, moment, phase="attempted_unknown", attempted=True)


def bind_startup_request_cookie(state: StartupCorrelationState, *, cookie: int, now: float) -> StartupCorrelationState:
    moment = _startup_now(state, now)
    _startup_integer(cookie, 2**32 - 1, minimum=1)
    terminal = _startup_expired(state, moment)
    if terminal is not None:
        return terminal
    if not state.send_attempted:
        _startup_fail()
    if state.request_cookie is not None and cookie != state.request_cookie:
        return _startup_unknown(state, moment, "conflicting_cookie")
    phase, reason, _ = _startup_scan(state.context, state.metadata, cookie, state.envelopes)
    return _startup_state(state, moment, phase=phase, reason=reason, cookie=cookie)


def advance_startup_correlation(state: StartupCorrelationState, envelope: StartupMessageEnvelope, *, now: float) -> StartupCorrelationState:
    moment = _startup_now(state, now)
    _startup_validate_envelope(envelope)
    terminal = _startup_expired(state, moment)
    if terminal is not None:
        return terminal
    if not state.send_attempted:
        return _startup_unknown(state, moment, "preexisting_unit" if envelope.kind in ("unit_new", "unit_removed") else "transport_gap")
    if len(state.envelopes) >= 64:
        return _startup_unknown(state, moment, "evidence_overflow")
    canonical = _startup_canonical(_startup_envelope_projection(envelope), 16384)
    if canonical is None or state.canonical_evidence_bytes + len(canonical) > 65536:
        return _startup_unknown(state, moment, "evidence_overflow")
    if envelope.serial in tuple(item.serial for item in state.envelopes):
        return _startup_unknown(state, moment, "replayed_native_serial")
    envelopes = state.envelopes + (envelope,)
    phase, reason, _ = _startup_scan(state.context, state.metadata, state.request_cookie, envelopes)
    return _startup_state(state, moment, phase=phase, reason=reason, envelopes=envelopes, size=state.canonical_evidence_bytes + len(canonical))


def invalidate_startup_correlation(state: StartupCorrelationState, *, reason: str, now: float) -> StartupCorrelationState:
    moment = _startup_now(state, now)
    if type(reason) is not str or reason not in _STARTUP_REASONS:
        _startup_fail()
    terminal = _startup_expired(state, moment)
    if terminal is not None:
        return terminal
    if reason in ("expired_unsent", "expired_unknown"):
        _startup_fail()
    return _startup_unknown(state, moment, reason)


def expire_startup_correlation(state: StartupCorrelationState, *, now: float) -> StartupCorrelationState:
    moment = _startup_now(state, now)
    terminal = _startup_expired(state, moment)
    if terminal is not None:
        return terminal
    return _startup_state(state, moment, phase=state.phase)


def startup_correlation_binding(state: StartupCorrelationState) -> StartupInvocationBinding | None:
    _startup_validate_state(state)
    if state.phase not in ("invocation_observed", "correlated_done"):
        return None
    _, _, facts = _startup_scan(state.context, state.metadata, state.request_cookie, state.envelopes)
    reply, job_new, joint, unit_new = (facts[name] for name in ("reply", "job_new", "joint", "unit_new"))
    removed = facts.get("job_removed")
    return StartupInvocationBinding(1, state.context, state.metadata.unit_name, state.metadata.unit_path, state.request_cookie, job_new.payload[0], job_new.payload[1], facts["invocation_id"], state.redacted_request_metadata_sha256, startup_envelope_evidence_sha256(reply), startup_envelope_evidence_sha256(job_new), startup_envelope_evidence_sha256(joint), None if removed is None else startup_envelope_evidence_sha256(removed), None if removed is None else removed.payload[3], startup_envelope_evidence_sha256(unit_new))


_RETENTION_VERSION = 1
_RETENTION_MAX_OWNERS = 16
_RETENTION_REQUEST_DOMAIN = "truememory.startup.retention_request.v1"
_RETENTION_PHASES = ("prepared", "local_observed", "terminal_unknown", "expired_unsent", "expired_unknown", "cancelled_unsent", "released_unsent")
_RETENTION_REASONS = _STARTUP_REASONS + ("cancelled_unsent",)


@dataclass(frozen=True, slots=True)
class StartupRetentionRequest:
    version: int
    generation: str
    requested_cap_bytes: int
    requested_swap_bytes: int
    slice_metadata: StartupCreateStructure
    service_metadata: StartupCreateStructure

    def __post_init__(self) -> None:
        _retention_validate_request(self)


@dataclass(frozen=True, slots=True)
class StartupRetentionObservation:
    version: int
    stage: str
    context: StartupBusContext | None
    correlation: StartupCorrelationState | None
    reason: str | None

    def __post_init__(self) -> None:
        _retention_validate_observation(self)


@dataclass(frozen=True, slots=True)
class StartupRetentionOwner:
    version: int
    owner_id: int
    creator_pid: int

    def __post_init__(self) -> None:
        _retention_validate_owner(self)


@dataclass(frozen=True, slots=True)
class StartupRetentionSnapshot:
    version: int
    owner: StartupRetentionOwner
    request: StartupRetentionRequest
    original_deadline: float
    last_now: float
    operation_sequence: int
    phase: str
    slice_context: StartupBusContext | None
    slice_correlation: StartupCorrelationState | None
    service_context: StartupBusContext | None
    service_correlation: StartupCorrelationState | None
    slice_attempted: bool
    service_attempted: bool
    reason: str | None

    def __post_init__(self) -> None:
        _retention_validate_snapshot(self)


@dataclass(frozen=True, slots=True)
class _RetentionSeal:
    owner: StartupRetentionOwner
    owner_id: int
    creator_pid: int
    original_deadline: float
    request_sha256: str
    request: StartupRetentionRequest


@dataclass(frozen=True, slots=True)
class _RetentionRecord:
    seal: _RetentionSeal
    last_now: float
    operation_sequence: int
    phase: str
    slice_context: StartupBusContext | None
    slice_correlation: StartupCorrelationState | None
    service_context: StartupBusContext | None
    service_correlation: StartupCorrelationState | None
    slice_attempted: bool
    service_attempted: bool
    reason: str | None


_RETENTION_CREATOR_PID = os.getpid()
_RETENTION_LOCK = threading.RLock()
_RETENTION_RECORDS: dict[int, _RetentionRecord] = {}
_RETENTION_NEXT_OWNER_ID = 0
_RETENTION_BUSY = False


def _retention_fail(message: str = "Invalid startup retention input") -> None:
    if type(message) is not str or message not in (
            "Invalid startup retention input", "Startup retention belongs to a different process",
            "Startup retention registry full", "Startup retention generation already retained",
            "Startup retention identifier exhausted", "Startup retention sequence exhausted",
            "Startup retention busy", "Startup retention owner is stale", "Startup retention sequence mismatch",
            "Startup retention time moved backwards", "Startup retention invariant changed",
            "Startup retention is terminal", "Startup retention has an attempted send"):
        message = "Invalid startup retention input"
    raise EnforcementUnavailable(message) from None


def _retention_exact_fields(value: object, cls: type) -> None:
    if type(value) is not cls:
        _retention_fail()
    for name in cls.__dataclass_fields__:
        try:
            object.__getattribute__(value, name)
        except AttributeError:
            _retention_fail()


def _retention_validate_request(value: object) -> StartupRetentionRequest:
    _retention_exact_fields(value, StartupRetentionRequest)
    if (type(value.version) is not int or value.version != _RETENTION_VERSION
            or type(value.generation) is not str or re.fullmatch(r"[0-9a-f]{32}", value.generation) is None
            or type(value.requested_cap_bytes) is not int or not 1 <= value.requested_cap_bytes <= 2**63 - 1
            or type(value.requested_swap_bytes) is not int or value.requested_swap_bytes != 0):
        _retention_fail()
    try:
        for stage, metadata in (("slice", value.slice_metadata), ("service", value.service_metadata)):
            _startup_validate_metadata(metadata)
            name = ("app-truememory" if stage == "slice" else "truememory-memory") + value.generation + "." + stage
            if metadata.unit_name != name or metadata.unit_path != _startup_unit_path(name):
                _retention_fail()
            redacted_create_metadata_sha256(metadata)
    except EnforcementUnavailable:
        _retention_fail()
    return value


def _retention_validate_observation(value: object) -> StartupRetentionObservation:
    _retention_exact_fields(value, StartupRetentionObservation)
    if type(value.version) is not int or value.version != _RETENTION_VERSION or type(value.stage) is not str or value.stage not in ("slice", "service"):
        _retention_fail()
    try:
        if value.context is None:
            if (value.correlation is not None or type(value.reason) is not str
                    or value.reason not in _STARTUP_REASONS or value.reason in ("expired_unsent", "expired_unknown")):
                _retention_fail()
        else:
            _startup_validate_context(value.context)
            if value.reason is not None:
                _retention_fail()
            if value.correlation is not None:
                _startup_validate_state(value.correlation)
                if value.correlation.context != value.context:
                    _retention_fail()
    except EnforcementUnavailable:
        _retention_fail()
    return value


def _retention_validate_owner(value: object) -> StartupRetentionOwner:
    _retention_exact_fields(value, StartupRetentionOwner)
    if (type(value.version) is not int or value.version != _RETENTION_VERSION
            or type(value.owner_id) is not int or not 1 <= value.owner_id <= 2**64 - 1
            or type(value.creator_pid) is not int or not 1 <= value.creator_pid <= 2**63 - 1):
        _retention_fail()
    return value


def _retention_validate_snapshot(value: object) -> StartupRetentionSnapshot:
    _retention_exact_fields(value, StartupRetentionSnapshot)
    _retention_validate_owner(value.owner)
    _retention_validate_request(value.request)
    if (type(value.version) is not int or value.version != _RETENTION_VERSION
            or type(value.original_deadline) is not float or not math.isfinite(value.original_deadline) or value.original_deadline <= 0.0
            or type(value.last_now) is not float or not math.isfinite(value.last_now) or value.last_now < 0.0
            or type(value.operation_sequence) is not int or not 0 <= value.operation_sequence <= 2**64 - 1
            or type(value.phase) is not str or value.phase not in _RETENTION_PHASES
            or type(value.slice_attempted) is not bool or type(value.service_attempted) is not bool
            or (value.reason is not None and (type(value.reason) is not str or value.reason not in _RETENTION_REASONS))):
        _retention_fail()
    try:
        for stage in ("slice", "service"):
            context = getattr(value, stage + "_context")
            correlation = getattr(value, stage + "_correlation")
            attempted = getattr(value, stage + "_attempted")
            if context is not None:
                _startup_validate_context(context)
            if correlation is None:
                if attempted:
                    _retention_fail()
            else:
                _startup_validate_state(correlation)
                if (context is None or context != correlation.context
                        or correlation.metadata != getattr(value.request, stage + "_metadata")
                        or correlation.original_deadline != value.original_deadline
                        or correlation.last_now > value.last_now or correlation.send_attempted is not attempted):
                    _retention_fail()
                if correlation.phase == "unknown" and value.reason != correlation.reason:
                    _retention_fail()
    except EnforcementUnavailable:
        _retention_fail()
    if value.slice_context is not None and value.service_context is not None and value.slice_context != value.service_context:
        _retention_fail()
    attempted = value.slice_attempted or value.service_attempted
    if value.phase == "prepared":
        if (value.operation_sequence != 0 or value.reason is not None or attempted
                or any(item is not None for item in (value.slice_context, value.slice_correlation, value.service_context, value.service_correlation))
                or value.last_now >= value.original_deadline):
            _retention_fail()
    elif value.phase == "local_observed":
        if (value.operation_sequence == 0 or value.reason is not None
                or (value.slice_context is None and value.service_context is None) or value.last_now >= value.original_deadline):
            _retention_fail()
    else:
        if value.operation_sequence == 0 or value.reason is None:
            _retention_fail()
        if value.phase == "terminal_unknown" and value.reason not in _STARTUP_REASONS[2:]:
            _retention_fail()
        if value.phase in ("expired_unsent", "expired_unknown"):
            if (value.last_now < value.original_deadline or value.reason != value.phase
                    or attempted != (value.phase == "expired_unknown")):
                _retention_fail()
        if value.phase in ("cancelled_unsent", "released_unsent") and (attempted or value.reason == "expired_unknown"):
            _retention_fail()
    return value


def _retention_validate_record(value: object) -> _RetentionRecord:
    _retention_exact_fields(value, _RetentionRecord)
    seal = value.seal
    _retention_exact_fields(seal, _RetentionSeal)
    _retention_validate_owner(seal.owner)
    _retention_validate_request(seal.request)
    if (type(seal.owner_id) is not int or seal.owner_id != seal.owner.owner_id
            or type(seal.creator_pid) is not int or seal.creator_pid != seal.owner.creator_pid
            or type(seal.request_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", seal.request_sha256) is None
            or seal.request_sha256 != _retention_request_sha256(seal.request)):
        _retention_fail("Startup retention invariant changed")
    StartupRetentionSnapshot(_RETENTION_VERSION, seal.owner, seal.request, seal.original_deadline,
                             value.last_now, value.operation_sequence, value.phase,
                             value.slice_context, value.slice_correlation, value.service_context,
                             value.service_correlation, value.slice_attempted, value.service_attempted, value.reason)
    return value


def _retention_pid() -> int:
    pid = os.getpid()
    if (type(pid) is not int or not 1 <= pid <= 2**63 - 1
            or type(_RETENTION_CREATOR_PID) is not int or pid != _RETENTION_CREATOR_PID):
        _retention_fail("Startup retention belongs to a different process")
    return pid


def _retention_enter() -> object:
    global _RETENTION_BUSY
    _retention_pid()
    lock = _RETENTION_LOCK
    if not lock.acquire(blocking=False):
        _retention_fail("Startup retention busy")
    try:
        _retention_pid()
        if type(_RETENTION_BUSY) is not bool or _RETENTION_BUSY or _RETENTION_LOCK is not lock:
            _retention_fail("Startup retention busy")
        _RETENTION_BUSY = True
        return lock
    except EnforcementUnavailable:
        lock.release()
        raise


def _retention_exit(lock: object) -> None:
    global _RETENTION_BUSY
    _RETENTION_BUSY = False
    lock.release()


def _retention_now(record: _RetentionRecord, now: object) -> float:
    if type(now) is not float or not math.isfinite(now) or now < 0.0:
        _retention_fail()
    if now < record.last_now:
        _retention_fail("Startup retention time moved backwards")
    return now


def _retention_expected_sequence(record: _RetentionRecord, value: object) -> int:
    if type(value) is not int or not 0 <= value <= 2**64 - 1:
        _retention_fail()
    if value != record.operation_sequence:
        _retention_fail("Startup retention sequence mismatch")
    if value == 2**64 - 1:
        _retention_fail("Startup retention sequence exhausted")
    return value + 1


def _retention_request_sha256(request: StartupRetentionRequest) -> str:
    _retention_validate_request(request)
    projection = {"domain": _RETENTION_REQUEST_DOMAIN, "version": request.version,
                  "generation": request.generation, "requested_cap_bytes": request.requested_cap_bytes,
                  "requested_swap_bytes": request.requested_swap_bytes,
                  "slice_metadata_sha256": redacted_create_metadata_sha256(request.slice_metadata),
                  "service_metadata_sha256": redacted_create_metadata_sha256(request.service_metadata)}
    raw = _startup_canonical(projection, 16384)
    if raw is None:
        _retention_fail()
    return hashlib.sha256(raw).hexdigest()


def _retention_copy_metadata(value: StartupCreateStructure) -> StartupCreateStructure:
    try:
        _startup_validate_metadata(value)
        redacted_create_metadata_sha256(value)
        return StartupCreateStructure(value.version, value.unit_name, value.unit_path, value.job_mode,
                                      tuple((name, signature) for name, signature in value.property_signatures),
                                      tuple(name for name in value.environment_names), ())
    except EnforcementUnavailable:
        _retention_fail()


def _retention_copy_context(value: StartupBusContext | None) -> StartupBusContext | None:
    if value is None:
        return None
    try:
        _startup_validate_context(value)
        return StartupBusContext(value.version, value.bus_id, value.manager_unique_name, value.client_unique_name, value.manager_uid)
    except EnforcementUnavailable:
        _retention_fail()


def _retention_copy_property(value: StartupProperty) -> StartupProperty:
    try:
        _startup_validate_property(value)
        item = value.value
        if value.signature in ("as", "(uo)"):
            item = tuple(part for part in item)
        elif value.signature == "ay":
            item = bytes(part for part in item)
        return StartupProperty(value.name, value.signature, item)
    except EnforcementUnavailable:
        _retention_fail()


def _retention_copy_envelope(value: StartupMessageEnvelope) -> StartupMessageEnvelope:
    try:
        _startup_validate_envelope(value)
        payload = value.payload
        if value.kind == "unit_properties":
            payload = (payload[0], tuple(_retention_copy_property(item) for item in payload[1]), tuple(name for name in payload[2]))
        else:
            payload = tuple(item for item in payload)
        return StartupMessageEnvelope(value.version, value.ordinal, value.kind, value.sender, value.destination,
                                      value.serial, value.reply_cookie, value.path, value.interface, value.member,
                                      value.signature, payload, value.error_name, value.error_body_redacted)
    except EnforcementUnavailable:
        _retention_fail()


def _retention_copy_correlation(value: StartupCorrelationState | None) -> StartupCorrelationState | None:
    if value is None:
        return None
    try:
        _startup_validate_state(value)
        return StartupCorrelationState(value.version, _retention_copy_context(value.context),
                                       _retention_copy_metadata(value.metadata), value.redacted_request_metadata_sha256,
                                       value.original_deadline, value.last_now, value.phase, value.send_attempted,
                                       value.request_cookie, tuple(_retention_copy_envelope(item) for item in value.envelopes),
                                       value.canonical_evidence_bytes, value.reason)
    except EnforcementUnavailable:
        _retention_fail()


def _retention_copy_request(value: StartupRetentionRequest) -> StartupRetentionRequest:
    _retention_validate_request(value)
    return StartupRetentionRequest(value.version, value.generation, value.requested_cap_bytes,
                                   value.requested_swap_bytes, _retention_copy_metadata(value.slice_metadata),
                                   _retention_copy_metadata(value.service_metadata))


def _retention_stage_consistency(record: _RetentionRecord, observation: StartupRetentionObservation, now: float) -> None:
    _retention_validate_observation(observation)
    stage = observation.stage
    context, correlation = observation.context, observation.correlation
    previous_context = getattr(record, stage + "_context")
    previous_correlation = getattr(record, stage + "_correlation")
    other = getattr(record, ("service" if stage == "slice" else "slice") + "_context")
    if context is not None and ((previous_context is not None and context != previous_context) or (other is not None and context != other)):
        _retention_fail()
    if correlation is not None:
        if (correlation.metadata != getattr(record.seal.request, stage + "_metadata")
                or correlation.original_deadline != record.seal.original_deadline or correlation.last_now > now):
            _retention_fail()
        if previous_correlation is not None:
            if (correlation.last_now < previous_correlation.last_now
                    or correlation.envelopes[:len(previous_correlation.envelopes)] != previous_correlation.envelopes
                    or (previous_correlation.request_cookie is not None and correlation.request_cookie != previous_correlation.request_cookie)
                    or (previous_correlation.send_attempted and not correlation.send_attempted)
                    or (previous_correlation.reason is not None and correlation.reason != previous_correlation.reason)):
                _retention_fail()
        if getattr(record, stage + "_attempted") and not correlation.send_attempted:
            _retention_fail()


def _retention_expire(record: _RetentionRecord, now: float) -> _RetentionRecord:
    if record.phase in ("prepared", "local_observed") and now >= record.seal.original_deadline:
        sequence = _retention_expected_sequence(record, record.operation_sequence)
        phase = "expired_unknown" if record.slice_attempted or record.service_attempted else "expired_unsent"
        return _retention_make_record(record, now, sequence, phase, phase)
    return _retention_make_record(record, now, record.operation_sequence, record.phase, record.reason)


def _retention_make_record(record: _RetentionRecord, now: float, sequence: int, phase: str, reason: str | None,
                           *, stage: str | None = None, context: StartupBusContext | None = None,
                           correlation: StartupCorrelationState | None = None) -> _RetentionRecord:
    values = [record.slice_context, record.slice_correlation, record.service_context,
              record.service_correlation, record.slice_attempted, record.service_attempted]
    if stage is not None:
        index, attempt_index = (0, 4) if stage == "slice" else (2, 5)
        if context is not None:
            values[index] = context
        if correlation is not None:
            values[index + 1] = correlation
            values[attempt_index] = values[attempt_index] or correlation.send_attempted
    result = _RetentionRecord(record.seal, now, sequence, phase, *values, reason)
    _retention_validate_record(result)
    return result


def _retention_commit(owner: StartupRetentionOwner, previous: _RetentionRecord,
                      candidate: _RetentionRecord, result: StartupRetentionSnapshot,
                      *, registry: dict[int, _RetentionRecord]) -> StartupRetentionSnapshot:
    _retention_validate_record(candidate)
    _retention_validate_snapshot(result)
    if (candidate.seal is not previous.seal or result.owner is not owner
            or result.request != candidate.seal.request or result.original_deadline != candidate.seal.original_deadline
            or tuple(getattr(result, name) for name in _RetentionRecord.__dataclass_fields__ if name != "seal")
            != tuple(getattr(candidate, name) for name in _RetentionRecord.__dataclass_fields__ if name != "seal")):
        _retention_fail("Startup retention invariant changed")
    _retention_guard_seal(owner, previous, registry=registry)
    _RETENTION_RECORDS[previous.seal.owner_id] = candidate
    return result


def _retention_snapshot(record: _RetentionRecord, *, phase: str | None = None, sequence: int | None = None) -> StartupRetentionSnapshot:
    _retention_validate_record(record)
    return StartupRetentionSnapshot(_RETENTION_VERSION, record.seal.owner, _retention_copy_request(record.seal.request),
                                    record.seal.original_deadline, record.last_now,
                                    record.operation_sequence if sequence is None else sequence,
                                    record.phase if phase is None else phase,
                                    _retention_copy_context(record.slice_context), _retention_copy_correlation(record.slice_correlation),
                                    _retention_copy_context(record.service_context), _retention_copy_correlation(record.service_correlation),
                                    record.slice_attempted, record.service_attempted, record.reason)


def _retention_guard_seal(owner: StartupRetentionOwner, expected: _RetentionRecord | None = None,
                         *, registry: dict[int, _RetentionRecord] | None = None) -> _RetentionRecord:
    _retention_pid()
    if type(owner) is not StartupRetentionOwner:
        _retention_fail()
    if (type(_RETENTION_RECORDS) is not dict or len(_RETENTION_RECORDS) > _RETENTION_MAX_OWNERS or _RETENTION_BUSY is not True
            or (registry is not None and _RETENTION_RECORDS is not registry)
            or type(_RETENTION_NEXT_OWNER_ID) is not int or not 0 <= _RETENTION_NEXT_OWNER_ID <= 2**64 - 1):
        _retention_fail("Startup retention invariant changed")
    found = None
    for key, record in _RETENTION_RECORDS.items():
        _retention_exact_fields(record, _RetentionRecord)
        _retention_exact_fields(record.seal, _RetentionSeal)
        if type(key) is not int or key != record.seal.owner_id or not 1 <= key <= _RETENTION_NEXT_OWNER_ID:
            _retention_fail("Startup retention invariant changed")
        if record.seal.owner is owner:
            if found is not None or type(key) is not int or key != record.seal.owner_id:
                _retention_fail("Startup retention invariant changed")
            found = record
    if found is None:
        _retention_validate_owner(owner)
        _retention_fail("Startup retention owner is stale")
    if expected is not None and found is not expected:
        _retention_fail("Startup retention invariant changed")
    try:
        _retention_validate_record(found)
    except EnforcementUnavailable:
        _retention_fail("Startup retention invariant changed")
    if found.seal.creator_pid != _RETENTION_CREATOR_PID:
        _retention_fail("Startup retention invariant changed")
    return found


def _retention_unsent(record: _RetentionRecord) -> None:
    if record.slice_attempted or record.service_attempted:
        _retention_fail("Startup retention has an attempted send")


def _retention_release(owner: StartupRetentionOwner, record: _RetentionRecord,
                       now: float, sequence: int, *, registry: dict[int, _RetentionRecord]) -> StartupRetentionSnapshot:
    _retention_unsent(record)
    if record.phase not in ("cancelled_unsent", "expired_unsent"):
        _retention_fail("Startup retention is terminal")
    candidate = _retention_make_record(record, now, sequence, "released_unsent", record.reason)
    result = _retention_snapshot(candidate)
    _retention_validate_record(candidate)
    _retention_validate_snapshot(result)
    if (result.owner is not owner or result.request != candidate.seal.request
            or result.original_deadline != candidate.seal.original_deadline
            or tuple(getattr(result, name) for name in _RetentionRecord.__dataclass_fields__ if name != "seal")
            != tuple(getattr(candidate, name) for name in _RetentionRecord.__dataclass_fields__ if name != "seal")):
        _retention_fail("Startup retention invariant changed")
    _retention_guard_seal(owner, record, registry=registry)
    del _RETENTION_RECORDS[record.seal.owner_id]
    return result


def prepare_startup_retention(request: StartupRetentionRequest, *, deadline: float, now: float) -> StartupRetentionOwner:
    global _RETENTION_NEXT_OWNER_ID
    lock = _retention_enter()
    try:
        _retention_validate_request(request)
        if (type(now) is not float or not math.isfinite(now) or now < 0.0
                or type(deadline) is not float or not math.isfinite(deadline) or deadline <= now):
            _retention_fail()
        records = _RETENTION_RECORDS
        previous_id = _RETENTION_NEXT_OWNER_ID
        if type(records) is not dict or len(records) > _RETENTION_MAX_OWNERS or type(previous_id) is not int or not 0 <= previous_id <= 2**64 - 1:
            _retention_fail("Startup retention invariant changed")
        if len(records) >= _RETENTION_MAX_OWNERS:
            _retention_fail("Startup retention registry full")
        for key, record in records.items():
            _retention_validate_record(record)
            if type(key) is not int or key != record.seal.owner_id or not 1 <= key <= previous_id or record.seal.creator_pid != _RETENTION_CREATOR_PID:
                _retention_fail("Startup retention invariant changed")
            if record.seal.request.generation == request.generation:
                _retention_fail("Startup retention generation already retained")
        if previous_id == 2**64 - 1:
            _retention_fail("Startup retention identifier exhausted")
        requested_sha = _retention_request_sha256(request)
        copied_request = _retention_copy_request(request)
        owner = StartupRetentionOwner(_RETENTION_VERSION, previous_id + 1, _retention_pid())
        seal = _RetentionSeal(owner, owner.owner_id, owner.creator_pid, deadline, requested_sha, copied_request)
        record = _RetentionRecord(seal, now, 0, "prepared", None, None, None, None, False, False, None)
        _retention_validate_record(record)
        if (records is not _RETENTION_RECORDS or _RETENTION_NEXT_OWNER_ID != previous_id
                or len(records) >= _RETENTION_MAX_OWNERS or _retention_request_sha256(request) != requested_sha
                or _RETENTION_BUSY is not True or owner.creator_pid != _retention_pid()):
            _retention_fail("Startup retention invariant changed")
        for key, existing in records.items():
            _retention_validate_record(existing)
            if type(key) is not int or key != existing.seal.owner_id or not 1 <= key <= previous_id or existing.seal.creator_pid != _RETENTION_CREATOR_PID:
                _retention_fail("Startup retention invariant changed")
            if existing.seal.request.generation == copied_request.generation:
                _retention_fail("Startup retention generation already retained")
        _retention_validate_record(record)
        if (_retention_request_sha256(request) != requested_sha or records is not _RETENTION_RECORDS
                or _RETENTION_NEXT_OWNER_ID != previous_id or _RETENTION_BUSY is not True
                or owner.creator_pid != _retention_pid()):
            _retention_fail("Startup retention invariant changed")
        records[owner.owner_id] = record
        _RETENTION_NEXT_OWNER_ID = owner.owner_id
        return owner
    finally:
        _retention_exit(lock)


def startup_retention_snapshot(owner: StartupRetentionOwner, *, now: float) -> StartupRetentionSnapshot:
    lock = _retention_enter()
    try:
        registry = _RETENTION_RECORDS
        record = _retention_guard_seal(owner)
        moment = _retention_now(record, now)
        candidate = _retention_expire(record, moment)
        result = _retention_snapshot(candidate)
        return _retention_commit(owner, record, candidate, result, registry=registry)
    finally:
        _retention_exit(lock)


def publish_startup_retention(owner: StartupRetentionOwner, observation: StartupRetentionObservation,
                              *, expected_sequence: int, now: float) -> StartupRetentionSnapshot:
    lock = _retention_enter()
    try:
        registry = _RETENTION_RECORDS
        record = _retention_guard_seal(owner)
        moment = _retention_now(record, now)
        sequence = _retention_expected_sequence(record, expected_sequence)
        _retention_validate_observation(observation)
        if record.phase not in ("prepared", "local_observed") or moment >= record.seal.original_deadline:
            _retention_fail("Startup retention is terminal")
        _retention_stage_consistency(record, observation, moment)
        stage, observed_reason = observation.stage, observation.reason
        context = _retention_copy_context(observation.context)
        correlation = _retention_copy_correlation(observation.correlation)
        reason = observed_reason if correlation is None else correlation.reason
        phase = "terminal_unknown" if reason is not None else "local_observed"
        candidate = _retention_make_record(record, moment, sequence, phase, reason,
                                           stage=stage, context=context, correlation=correlation)
        result = _retention_snapshot(candidate)
        _retention_stage_consistency(record, observation, moment)
        if (observation.stage != stage or observation.context != context
                or observation.correlation != correlation or observation.reason != observed_reason):
            _retention_fail("Startup retention invariant changed")
        return _retention_commit(owner, record, candidate, result, registry=registry)
    finally:
        _retention_exit(lock)


def invalidate_startup_retention(owner: StartupRetentionOwner, *, reason: str,
                                 expected_sequence: int, now: float) -> StartupRetentionSnapshot:
    lock = _retention_enter()
    try:
        registry = _RETENTION_RECORDS
        record = _retention_guard_seal(owner)
        moment = _retention_now(record, now)
        sequence = _retention_expected_sequence(record, expected_sequence)
        if type(reason) is not str or reason not in _STARTUP_REASONS[2:]:
            _retention_fail()
        if record.phase not in ("prepared", "local_observed") or moment >= record.seal.original_deadline:
            _retention_fail("Startup retention is terminal")
        candidate = _retention_make_record(record, moment, sequence, "terminal_unknown", reason)
        return _retention_commit(owner, record, candidate, _retention_snapshot(candidate), registry=registry)
    finally:
        _retention_exit(lock)


def cancel_unsent_startup_retention(owner: StartupRetentionOwner, *, expected_sequence: int,
                                    now: float) -> StartupRetentionSnapshot:
    lock = _retention_enter()
    try:
        registry = _RETENTION_RECORDS
        record = _retention_guard_seal(owner)
        moment = _retention_now(record, now)
        sequence = _retention_expected_sequence(record, expected_sequence)
        _retention_unsent(record)
        if record.phase not in ("prepared", "local_observed", "terminal_unknown", "expired_unsent"):
            _retention_fail("Startup retention is terminal")
        candidate = _retention_make_record(record, moment, sequence, "cancelled_unsent",
                                           record.reason if record.reason is not None else "cancelled_unsent")
        return _retention_commit(owner, record, candidate, _retention_snapshot(candidate), registry=registry)
    finally:
        _retention_exit(lock)


def release_unsent_startup_retention(owner: StartupRetentionOwner, *, expected_sequence: int,
                                     now: float) -> StartupRetentionSnapshot:
    lock = _retention_enter()
    try:
        registry = _RETENTION_RECORDS
        record = _retention_guard_seal(owner)
        moment = _retention_now(record, now)
        sequence = _retention_expected_sequence(record, expected_sequence)
        return _retention_release(owner, record, moment, sequence, registry=registry)
    finally:
        _retention_exit(lock)


@dataclass(frozen=True)
class LinuxDaemonLaunchPolicy:
    version: int = 1
    cap_bytes: int = 8589934592
    swap_bytes: int = 0
    metric: str = _METRIC
    physical_hard_cap: bool = False
    dynamic_registry_cap: bool = False

    def __post_init__(self) -> None:
        _validate_policy(self)


def _validate_policy(policy: object) -> LinuxDaemonLaunchPolicy:
    _complete_fields(policy, LinuxDaemonLaunchPolicy)
    _literal(policy.version, 1, "policy version")
    _integer(policy.cap_bytes, "cap_bytes", positive=True)
    _literal(policy.swap_bytes, 0, "swap_bytes")
    _literal(policy.metric, _METRIC, "metric")
    _literal(policy.physical_hard_cap, False, "physical_hard_cap")
    _literal(policy.dynamic_registry_cap, False, "dynamic_registry_cap")
    return policy


@dataclass(frozen=True)
class LinuxInvocationIdentity:
    version: int
    generation: str
    boot_id: str
    uid: int
    slice_unit: str
    slice_invocation_id: str
    slice_mount_id: int
    slice_device: int
    slice_inode: int
    slice_relative_path: str
    service_unit: str
    service_invocation_id: str
    service_device: int
    service_inode: int
    service_relative_path: str
    main_pid: int
    start_ticks: int
    birth_time: float
    cap_bytes: int
    swap_bytes: int
    bootstrap_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        _validate_identity(self)


def _validate_identity(identity: object) -> LinuxInvocationIdentity:
    _complete_fields(identity, LinuxInvocationIdentity)
    _literal(identity.version, 1, "identity version")
    _hex(identity.generation, 32, "generation")
    if type(identity.boot_id) is not str or _UUID.fullmatch(identity.boot_id) is None:
        raise ValueError("Invalid boot UUID")
    _integer(identity.uid, "uid")
    for name, suffix in (("slice_unit", ".slice"), ("service_unit", ".service")):
        value = _text(getattr(identity, name), name)
        if _UNIT.fullmatch(value) is None or not value.endswith(suffix) or identity.generation not in value:
            raise ValueError(f"Invalid generation-scoped {name}")
    if not identity.slice_unit.startswith("app-"):
        raise ValueError("Owned slice must be below the existing app slice")
    for name in ("slice_invocation_id", "service_invocation_id"):
        _hex(getattr(identity, name), 32, name)
    for name in ("slice_mount_id", "slice_device", "slice_inode", "service_device", "service_inode", "main_pid", "start_ticks", "cap_bytes"):
        _integer(getattr(identity, name), name, positive=True)
    _relative(identity.slice_relative_path, "slice_relative_path")
    _relative(identity.service_relative_path, "service_relative_path")
    if not identity.service_relative_path.startswith(identity.slice_relative_path + "/"):
        raise ValueError("Child service must be strictly inside the owned slice")
    if type(identity.birth_time) is not float or not math.isfinite(identity.birth_time) or identity.birth_time <= 0:
        raise ValueError("birth_time must be a finite positive exact float")
    _literal(identity.swap_bytes, 0, "swap_bytes")
    _hex(identity.bootstrap_sha256, 64, "bootstrap_sha256")
    _hex(identity.manifest_sha256, 64, "manifest_sha256")
    return identity


@dataclass(frozen=True)
class LinuxProvisionalLaunch:
    identity: LinuxInvocationIdentity
    runner: object
    pidfd: object
    scope_fd: object
    endpoint: object

    def __post_init__(self) -> None:
        _validate_record(self)


def _validate_record(record: object) -> LinuxProvisionalLaunch:
    _complete_fields(record, LinuxProvisionalLaunch)
    _validate_identity(record.identity)
    for name in ("runner", "pidfd", "scope_fd", "endpoint"):
        value = getattr(record, name)
        if value is None or type(value) in (bool, float, str, bytes, tuple, dict, list):
            raise ValueError(f"Missing strong owned {name} handle")
        if type(value) is int and (name != "pidfd" or value < 0):
            raise ValueError(f"Invalid owned {name} handle")
    return record


@dataclass(frozen=True)
class LinuxScopeEvidence:
    identity: LinuxInvocationIdentity
    filesystem_magic: int
    mount_type: str
    mount_authenticated: bool
    namespace_authenticated: bool
    membership_authenticated: bool
    peer_uid: int
    peer_pid: int
    peer_start_ticks: int
    generation: str
    manifest_sha256: str
    members: tuple[int, ...]
    cap_bytes: int
    swap_bytes: int
    leaf_memory_max: str
    leaf_swap_max: str
    kernel_limit_preexec: bool

    def __post_init__(self) -> None:
        _validate_evidence(self, self.identity, terminal=True)


def _validate_evidence(evidence: object, identity: LinuxInvocationIdentity, *, terminal: bool = False) -> LinuxScopeEvidence:
    _validate_identity(identity)
    _complete_fields(evidence, LinuxScopeEvidence)
    _validate_identity(evidence.identity)
    if evidence.identity != identity:
        raise EnforcementUnavailable("Invocation identity drift")
    _literal(evidence.filesystem_magic, _CGROUP2_MAGIC, "filesystem_magic")
    _literal(evidence.mount_type, "cgroup2", "mount_type")
    for name in ("mount_authenticated", "namespace_authenticated", "membership_authenticated", "kernel_limit_preexec"):
        _literal(getattr(evidence, name), True, name)
    for name, expected in (("peer_uid", identity.uid), ("peer_pid", identity.main_pid), ("peer_start_ticks", identity.start_ticks), ("cap_bytes", identity.cap_bytes), ("swap_bytes", 0)):
        _literal(getattr(evidence, name), expected, name)
    _literal(evidence.generation, identity.generation, "generation")
    _literal(evidence.manifest_sha256, identity.manifest_sha256, "manifest_sha256")
    if type(evidence.members) is not tuple:
        raise ValueError("members must be an exact tuple")
    for value in evidence.members:
        _integer(value, "member PID", positive=True)
    if evidence.members != (identity.main_pid,) and not (terminal and evidence.members == ()):
        raise EnforcementUnavailable("Missing or foreign startup member")
    leaf = "max"
    if terminal and evidence.members == () and type(evidence.leaf_memory_max) is str and evidence.leaf_memory_max == "unavailable_pruned":
        leaf = "unavailable_pruned"
    _literal(evidence.leaf_memory_max, leaf, "leaf_memory_max")
    _literal(evidence.leaf_swap_max, leaf, "leaf_swap_max")
    return evidence


@dataclass(frozen=True)
class LinuxExitReceipt:
    version: int
    identity: LinuxInvocationIdentity
    exit_code: int | None
    signal: int | None
    manager_result: str
    memory_current: int
    memory_peak: int | None
    events_before: tuple[tuple[str, int], ...]
    events_after: tuple[tuple[str, int], ...]
    populated: bool
    descendant_members: tuple[tuple[str, tuple[int, ...]], ...]
    durable_before_stop: bool
    receipt_sequence: int
    receipt_chain_sha256: str
    artifact_sha256: tuple[tuple[str, str], ...]
    metric: str = _METRIC
    physical_hard_cap: bool = False
    kernel_cap_bytes: int = 8589934592
    dynamic_registry_cap: bool = False

    def __post_init__(self) -> None:
        _validate_receipt(self, self.identity)


def _events(value: object, name: str) -> tuple[tuple[str, int], ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be an exact tuple")
    names = set()
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError(f"Invalid {name} entry")
        key = _text(item[0], name, bound=64)
        if not key.isascii() or not re.fullmatch(r"[A-Za-z0-9_]+", key) or key in names:
            raise ValueError(f"Invalid or duplicate {name} event")
        _integer(item[1], name)
        names.add(key)
    return value


def _validate_receipt(receipt: object, identity: LinuxInvocationIdentity) -> LinuxExitReceipt:
    _validate_identity(identity)
    _complete_fields(receipt, LinuxExitReceipt)
    _literal(receipt.version, 1, "receipt version")
    _validate_identity(receipt.identity)
    if receipt.identity != identity:
        raise EnforcementUnavailable("Terminal identity drift")
    if receipt.exit_code is not None:
        _integer(receipt.exit_code, "exit_code")
    if receipt.signal is not None:
        _integer(receipt.signal, "signal", positive=True)
    _text(receipt.manager_result, "manager_result")
    _integer(receipt.memory_current, "memory_current")
    if receipt.memory_peak is not None:
        _integer(receipt.memory_peak, "memory_peak")
    before = dict(_events(receipt.events_before, "events_before"))
    after = dict(_events(receipt.events_after, "events_after"))
    if any(name not in after or after[name] < count for name, count in before.items()):
        raise EnforcementUnavailable("Controller events decreased or disappeared")
    _literal(receipt.populated, False, "populated")
    if type(receipt.descendant_members) is not tuple or not receipt.descendant_members:
        raise ValueError("Missing recursive membership evidence")
    paths = set()
    for item in receipt.descendant_members:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Malformed descendant membership")
        path, members = item
        _relative(path, "descendant path")
        if path in paths or not (path == identity.slice_relative_path or path.startswith(identity.slice_relative_path + "/")):
            raise EnforcementUnavailable("Foreign or duplicate descendant path")
        if type(members) is not tuple or members != ():
            raise EnforcementUnavailable("Terminal domain still contains a process")
        paths.add(path)
    if identity.slice_relative_path not in paths:
        raise EnforcementUnavailable("Persistent slice empty proof is missing")
    _literal(receipt.durable_before_stop, True, "durable_before_stop")
    _integer(receipt.receipt_sequence, "receipt_sequence", positive=True)
    _hex(receipt.receipt_chain_sha256, 64, "receipt_chain_sha256")
    if type(receipt.artifact_sha256) is not tuple or not receipt.artifact_sha256:
        raise ValueError("Missing retained artifact hashes")
    names = set()
    for item in receipt.artifact_sha256:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Malformed artifact hash")
        name = _text(item[0], "artifact name")
        if name in names or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ValueError("Unsafe or duplicate artifact name")
        _hex(item[1], 64, "artifact SHA256")
        names.add(name)
    _literal(receipt.metric, _METRIC, "metric")
    _literal(receipt.physical_hard_cap, False, "physical_hard_cap")
    _literal(receipt.kernel_cap_bytes, identity.cap_bytes, "kernel_cap_bytes")
    _literal(receipt.dynamic_registry_cap, False, "dynamic_registry_cap")
    return receipt


def required_cap_selected(environment: Mapping[str, str]) -> bool:
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be a mapping")
    if _FLAG not in environment:
        return False
    _literal(environment[_FLAG], "1", _FLAG)
    return True


def plan_required_linux_launch(environment: Mapping[str, str], *, physical_total: int | None) -> LinuxDaemonLaunchPolicy:
    if not required_cap_selected(environment):
        raise ValueError("Required Linux mode was not explicitly selected")
    if "TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB" in environment:
        raw = environment["TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB"]
        if type(raw) is not str or not re.fullmatch(r"[1-9][0-9]*", raw) or len(raw) > 19:
            raise ValueError("Required MAX_RSS_MB must be canonical positive decimal")
        megabytes = _integer(int(raw), "MAX_RSS_MB", positive=True)
        cap = _integer(megabytes * 1048576, "cap_bytes", positive=True)
    else:
        total = _integer(physical_total, "physical_total", positive=True)
        cap = _integer(min(8589934592, total // 4), "cap_bytes", positive=True)
    return LinuxDaemonLaunchPolicy(1, cap, 0)


def _validate_claim(claim: object, *, pid: int | None = None, birth_time: float | None = None,
                    generation: str | None = None, pending: bool = False) -> dict[str, object]:
    if type(claim) is not dict or any(type(key) is not str for key in claim) or set(claim) != {"generation", "launcher", "child"}:
        raise ValueError("Malformed current startup claim")
    _hex(claim["generation"], 32, "claim generation")
    for name in ("launcher", "child"):
        item = claim[name]
        if name == "child" and item is None and pending:
            continue
        if type(item) is not dict or any(type(key) is not str for key in item) or set(item) != {"pid", "born"}:
            raise ValueError("Malformed current process claim")
        _integer(item["pid"], "claim pid", positive=True)
        if type(item["born"]) is not float or not math.isfinite(item["born"]) or item["born"] <= 0:
            raise ValueError("Claim birth must be a finite positive exact float")
    if generation is not None:
        _literal(claim["generation"], generation, "claim generation")
    if pid is not None:
        _literal(claim["child"]["pid"], pid, "claim child pid")
    if birth_time is not None:
        _literal(claim["child"]["born"], birth_time, "claim child birth")
    return claim


def inspect_existing_owned_linux(policy: LinuxDaemonLaunchPolicy, *, pid: int, birth_time: float,
                                 generation: str, claim: dict[str, object], deadline: float,
                                 adapter: object | None = None) -> LinuxScopeEvidence:
    """Observe current capped membership without acquiring launch authority."""
    _validate_policy(policy)
    expected_policy = _policy_snapshot(policy)
    _integer(pid, "existing pid", positive=True)
    if type(birth_time) is not float or not math.isfinite(birth_time) or birth_time <= 0:
        raise ValueError("Existing birth must be a finite positive exact float")
    _hex(generation, 32, "existing generation")
    _validate_claim(claim, pid=pid, birth_time=birth_time, generation=generation)
    expected_claim = _canonical(claim)
    _deadline(deadline)
    observer = get_linux_adapter() if adapter is None else adapter
    inspect_existing = getattr(observer, "inspect_existing", None)
    if not callable(inspect_existing):
        raise EnforcementUnavailable("Readonly Linux inspection is unavailable")
    _validate_policy(policy)
    _validate_claim(claim, pid=pid, birth_time=birth_time, generation=generation)
    if policy != expected_policy or _canonical(claim) != expected_claim:
        raise EnforcementUnavailable("Existing launch inputs changed during observer resolution")
    _deadline(deadline)
    result = inspect_existing(policy, pid=pid, birth_time=birth_time, generation=generation,
                              claim=claim, deadline=deadline)
    _complete_fields(result, LinuxScopeEvidence)
    identity = _validate_identity(result.identity)
    _validate_evidence(result, identity)
    _literal(identity.main_pid, pid, "existing MainPID")
    _literal(identity.birth_time, birth_time, "existing birth")
    _literal(identity.generation, generation, "existing generation")
    _validate_policy(policy)
    _validate_claim(claim, pid=pid, birth_time=birth_time, generation=generation)
    if policy != expected_policy or _canonical(claim) != expected_claim or identity.cap_bytes > expected_policy.cap_bytes:
        raise EnforcementUnavailable("Current capped daemon is incompatible with the requested policy")
    _deadline(deadline)
    return result


class OwnedLinuxLaunch:
    def __init__(self, record: LinuxProvisionalLaunch, policy: LinuxDaemonLaunchPolicy, adapter: object) -> None:
        self.record = _validate_record(record)
        self.policy = _validate_policy(policy)
        self._adapter = adapter
        self.pid = record.identity.main_pid
        self._released = False

    def release_after_claim(self, claim: Mapping[str, object], *, deadline: float) -> None:
        record, policy = _validate_handle(self)
        _deadline(deadline)
        if self._released:
            raise EnforcementUnavailable("Owned launch barrier was already released")
        identity = _identity_snapshot(record.identity)
        expected_policy = _policy_snapshot(policy)
        adapter = self._adapter
        resources = _resource_snapshot(record)
        _validate_claim(claim, pid=identity.main_pid, birth_time=identity.birth_time, generation=identity.generation)
        expected_claim = _canonical(claim)
        _deadline(deadline)
        _validate_evidence(adapter.inspect(record, deadline), identity)
        _validate_handle(self)
        _same_snapshot(record, policy, identity, expected_policy, resources)
        if self.record is not record or self.policy is not policy or self._released:
            raise EnforcementUnavailable("Owned handle changed during claim inspection")
        if self._adapter is not adapter:
            raise EnforcementUnavailable("Retained adapter changed during claim inspection")
        _validate_claim(claim, pid=identity.main_pid, birth_time=identity.birth_time, generation=identity.generation)
        if _canonical(claim) != expected_claim:
            raise EnforcementUnavailable("Startup claim changed during inspection")
        _deadline(deadline)
        adapter.barrier(record, identity, deadline)
        _validate_handle(self)
        _same_snapshot(record, policy, identity, expected_policy, resources)
        if self.record is not record or self.policy is not policy or self._adapter is not adapter:
            raise EnforcementUnavailable("Owned handle changed during barrier release")
        _validate_claim(claim, pid=identity.main_pid, birth_time=identity.birth_time, generation=identity.generation)
        if _canonical(claim) != expected_claim:
            raise EnforcementUnavailable("Startup claim changed during release")
        _deadline(deadline)
        self._released = True


def _validate_handle(handle: object) -> tuple[LinuxProvisionalLaunch, LinuxDaemonLaunchPolicy]:
    if type(handle) is not OwnedLinuxLaunch:
        raise TypeError("Expected exact OwnedLinuxLaunch")
    record, policy = _validate_record(handle.record), _validate_policy(handle.policy)
    _literal(handle.pid, record.identity.main_pid, "owned MainPID")
    _literal(record.identity.cap_bytes, policy.cap_bytes, "owned cap")
    _literal(record.identity.swap_bytes, policy.swap_bytes, "owned swap")
    if type(handle._released) is not bool:
        raise ValueError("Malformed release state")
    if handle._adapter is None:
        raise ValueError("Missing retained adapter")
    return record, policy


def _identity_snapshot(identity: LinuxInvocationIdentity) -> LinuxInvocationIdentity:
    _validate_identity(identity)
    return LinuxInvocationIdentity(**asdict(identity))


def _policy_snapshot(policy: LinuxDaemonLaunchPolicy) -> LinuxDaemonLaunchPolicy:
    _validate_policy(policy)
    return LinuxDaemonLaunchPolicy(**asdict(policy))


def _resource_snapshot(record: LinuxProvisionalLaunch) -> tuple[object, ...]:
    _validate_record(record)
    return (record.runner, record.pidfd, record.scope_fd, record.endpoint)


def _same_snapshot(record: LinuxProvisionalLaunch, policy: LinuxDaemonLaunchPolicy,
                   identity: LinuxInvocationIdentity, expected_policy: LinuxDaemonLaunchPolicy,
                   resources: tuple[object, ...]) -> None:
    _validate_record(record)
    _validate_policy(policy)
    if record.identity != identity or policy != expected_policy:
        raise EnforcementUnavailable("Owned policy or invocation changed during a callback")
    if any(actual is not expected for actual, expected in zip(_resource_snapshot(record), resources)):
        raise EnforcementUnavailable("Strong owned resources changed during a callback")


def launch_owned_linux(policy: LinuxDaemonLaunchPolicy, *, generation: str, python_executable: str, deadline: float) -> OwnedLinuxLaunch:
    _validate_policy(policy)
    expected_policy = _policy_snapshot(policy)
    _hex(generation, 32, "generation")
    _text(python_executable, "python_executable", bound=4096)
    if not Path(python_executable).is_absolute():
        raise ValueError("python_executable must be absolute")
    _deadline(deadline)
    adapter = get_linux_adapter()
    _deadline(deadline)
    record = _validate_record(adapter.start(policy, generation, python_executable, deadline))
    try:
        _literal(record.identity.generation, generation, "returned generation")
        _literal(record.identity.cap_bytes, policy.cap_bytes, "returned cap")
        _literal(record.identity.swap_bytes, policy.swap_bytes, "returned swap")
        identity = _identity_snapshot(record.identity)
        resources = _resource_snapshot(record)
        _same_snapshot(record, policy, identity, expected_policy, resources)
        _validate_policy(policy)
        _deadline(deadline)
        _validate_evidence(adapter.inspect(record, deadline), identity)
        _same_snapshot(record, policy, identity, expected_policy, resources)
        _deadline(deadline)
        return OwnedLinuxLaunch(record, policy, adapter)
    except (OSError, RuntimeError, ValueError, TypeError, TimeoutError):
        # The started child is ours even though it will not be returned.
        if type(adapter) is LinuxAdapter:
            try:
                adapter.abandon(record, expected_policy)
            except (OSError, RuntimeError, ValueError, TypeError, TimeoutError, AttributeError):
                pass
        raise


def child_gate(identity: LinuxInvocationIdentity, *, deadline: float) -> None:
    _validate_identity(identity)
    expected = _identity_snapshot(identity)
    _deadline(deadline)
    adapter = get_linux_adapter()
    _validate_evidence(adapter.inspect(identity, deadline), expected)
    _validate_identity(identity)
    if identity != expected:
        raise EnforcementUnavailable("Child identity changed during inspection")
    _deadline(deadline)


def capture_owned_exit(handle: OwnedLinuxLaunch, *, deadline: float) -> LinuxExitReceipt:
    record, policy = _validate_handle(handle)
    identity, expected_policy, adapter = _identity_snapshot(record.identity), _policy_snapshot(policy), handle._adapter
    resources = _resource_snapshot(record)
    _deadline(deadline)
    evidence = _validate_evidence(adapter.inspect(record, deadline), identity, terminal=True)
    if evidence.members:
        raise EnforcementUnavailable("Terminal capture requires an actually empty owned domain")
    _validate_handle(handle)
    _same_snapshot(record, policy, identity, expected_policy, resources)
    if handle.record is not record or handle.policy is not policy or handle._adapter is not adapter:
        raise EnforcementUnavailable("Owned handle changed during terminal inspection")
    _deadline(deadline)
    receipt = _validate_receipt(adapter.capture(record, identity, deadline), identity)
    _validate_handle(handle)
    _same_snapshot(record, policy, identity, expected_policy, resources)
    if handle.record is not record or handle.policy is not policy or handle._adapter is not adapter:
        raise EnforcementUnavailable("Retained adapter changed during capture")
    _deadline(deadline)
    return receipt


def stop_owned_exact(handle: OwnedLinuxLaunch, *, deadline: float) -> LinuxExitReceipt:
    record, policy = _validate_handle(handle)
    identity, expected_policy, adapter = _identity_snapshot(record.identity), _policy_snapshot(policy), handle._adapter
    resources = _resource_snapshot(record)
    _deadline(deadline)
    receipt = capture_owned_exit(handle, deadline=deadline)
    evidence = _validate_evidence(adapter.inspect(record, deadline), identity, terminal=True)
    if evidence.members:
        raise EnforcementUnavailable("Owned stop requires an actually empty owned domain")
    _validate_receipt(receipt, identity)
    _validate_handle(handle)
    _same_snapshot(record, policy, identity, expected_policy, resources)
    if handle.record is not record or handle.policy is not policy or handle._adapter is not adapter:
        raise EnforcementUnavailable("Retained adapter changed before stop")
    _deadline(deadline)
    adapter.stop(record, identity, deadline)
    return receipt


def _owned_exited(record: LinuxProvisionalLaunch) -> bool:
    scope = record.scope_fd
    return _pidfd_exited(record.pidfd, scope.pidfd_device, scope.pidfd_inode)


def release_exited_launch(handle: OwnedLinuxLaunch) -> bool:
    """Retire an owned launch whose workload has exited.

    Terminal evidence is captured when the manager still holds it. A failed
    service is collected by the manager at exit, so its units are then retired
    from the empty-domain proof alone. Nothing is signaled.
    """
    record, _ = _validate_handle(handle)
    identity = record.identity
    if record.runner.stop_complete:
        return True
    for attempt in range(2):
        try:
            stop_owned_exact(handle, deadline=time.monotonic() + 30.0)
        except (OSError, RuntimeError, ValueError, TypeError, TimeoutError):
            if record.runner.stop_complete:
                return True
            if attempt == 0:
                # The manager may not have reaped the workload yet.
                time.sleep(0.5)
            continue
        _mark_generation_retired(identity.generation)
        return True
    try:
        if not _owned_exited(record):
            return False
        retired = _retire_generation_units(identity.generation, mount_point=record.scope_fd.mount_point,
                                           parent_relative=identity.slice_relative_path.rsplit("/", 1)[0],
                                           deadline=time.monotonic() + 30.0, settle=10.0)
    except (OSError, RuntimeError, ValueError, TypeError, TimeoutError):
        return False
    if retired:
        _mark_generation_retired(identity.generation)
        try:
            handle._adapter._close_resolved(record, identity)
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
            pass
        record.runner.stop_complete = True
    return retired


def abandon_owned_launch(handle: OwnedLinuxLaunch) -> bool:
    """Terminate and retire an owned launch that will not be published.

    Only the retained pidfd is signaled. False means the workload is still live
    and remains owned by this handle; the caller must keep watching it.
    """
    record, _ = _validate_handle(handle)
    for attempt in range(3):
        if record.runner.stop_complete:
            return True
        try:
            if _owned_exited(record):
                return release_exited_launch(handle)
            terminate_owned_exact(handle, deadline=time.monotonic() + 5.0)
        except (OSError, RuntimeError, ValueError, TypeError, TimeoutError):
            # The workload's own short-lived manager helpers can make a single
            # census transiently non-exact; they exit within milliseconds.
            if attempt < 2:
                time.sleep(0.2)
            continue
        _mark_generation_retired(record.identity.generation)
        return True
    try:
        return _owned_exited(record) and release_exited_launch(handle)
    except (OSError, RuntimeError, ValueError, TypeError):
        return False


_termination_context = threading.local()


def _termination_deadline(deadline: float, entry: float) -> float:
    if type(deadline) is not float or not math.isfinite(deadline):
        raise ValueError("Termination deadline must be a finite exact float")
    if deadline <= entry:
        raise TimeoutError("Owned termination deadline expired")
    return min(deadline, entry + 5.0)


def _termination_owner_guard(record: LinuxProvisionalLaunch) -> None:
    context = getattr(_termination_context, "owner", None)
    if context is None:
        return
    handle, expected_record, policy, identity, expected_policy, resources, adapter, released, store, directory, metadata, path = context
    _validate_handle(handle)
    _same_snapshot(record, policy, identity, expected_policy, resources)
    if record is not expected_record or handle.record is not record or handle.policy is not policy or handle._adapter is not adapter or handle._released is not released or record.runner.store is not store or type(store.directory) is not type(directory) or store.directory != directory or store.metadata is not metadata or store.path is not path:
        raise EnforcementUnavailable("Public termination owner changed during a callback")


def _termination_emit(record: LinuxProvisionalLaunch, phase: str, payload: dict[str, object], deadline: float) -> None:
    _deadline(deadline)
    _validate_record(record)
    _termination_owner_guard(record)
    record.runner.store.verify()
    _termination_owner_guard(record)
    _deadline(deadline)
    result = record.runner.store.append(phase, payload)
    if type(result) is not tuple or len(result) != 2:
        raise EnforcementUnavailable("Termination intent has no durable receipt")
    _integer(result[0], "termination receipt sequence", positive=True)
    _hex(result[1], 64, "termination receipt hash")
    _termination_owner_guard(record)
    _deadline(deadline)


def terminate_owned_exact(handle: OwnedLinuxLaunch, *, deadline: float) -> LinuxExitReceipt:
    """Terminate only the retained incarnation, then capture and empty-stop it."""
    effective = _termination_deadline(deadline, time.monotonic())
    record, policy = _validate_handle(handle)
    identity, expected_policy = _identity_snapshot(record.identity), _policy_snapshot(policy)
    resources, adapter, released = _resource_snapshot(record), handle._adapter, handle._released
    method = getattr(adapter, "terminate", None)
    if not callable(method):
        raise EnforcementUnavailable("Retained adapter cannot perform owned pidfd termination")
    previous = getattr(_termination_context, "owner", None)
    context = (handle, record, policy, identity, expected_policy, resources, adapter, released,
               record.runner.store, record.runner.store.directory, record.runner.store.metadata,
               record.runner.store.path)
    _termination_context.owner = context
    try:
        _termination_emit(record, "termination-intent", {"version": 1, "generation": identity.generation,
                          "main_pid": identity.main_pid, "start_ticks": identity.start_ticks,
                          "birth_time": identity.birth_time, "original_deadline": deadline,
                          "effective_deadline": effective, "kernel_cap_bytes": identity.cap_bytes,
                          "swap_bytes": 0, "metric": _METRIC, "physical_hard_cap": False,
                          "dynamic_registry_cap": False}, effective)
        current = _validate_evidence(adapter.inspect(record, effective), identity, terminal=True)
        _termination_owner_guard(record)
        _deadline(effective)
        if current.members:
            if method(record, identity, effective) is not None:
                raise EnforcementUnavailable("Owned termination returned an unsupported result")
        _termination_owner_guard(record)
        _deadline(effective)
        terminal = _validate_evidence(adapter.inspect(record, effective), identity, terminal=True)
        _termination_owner_guard(record)
        if terminal.members:
            raise EnforcementUnavailable("Owned termination did not observe an empty terminal domain")
        _deadline(effective)
        result = stop_owned_exact(handle, deadline=effective)
        _termination_owner_guard(record)
        _deadline(effective)
        return _validate_receipt(result, identity)
    except (OSError, RuntimeError, ValueError, TypeError, TimeoutError) as error:
        try:
            _termination_emit(record, "termination-unresolved", {"generation": identity.generation,
                              "error_type": type(error).__name__}, effective)
        except (OSError, RuntimeError, ValueError, TypeError, TimeoutError):
            pass
        raise
    finally:
        _termination_context.owner = previous


def _termination_resources(record: LinuxProvisionalLaunch) -> tuple[tuple[object, ...], tuple[object, ...], bytes]:
    _validate_record(record)
    if type(record.runner) is not _NativeRunner or type(record.scope_fd) is not _NativeScope or type(record.endpoint) is not _NativeEndpoint:
        raise EnforcementUnavailable("Native termination resources changed")
    runner, scope, endpoint = record.runner, record.scope_fd, record.endpoint
    if type(runner.stop_complete) is not bool or runner.stop_complete:
        raise EnforcementUnavailable("Owned manager lifetime already stopped")
    for value in (record.pidfd, scope.slice_fd, scope.service_fd, scope.pidfd_device, scope.pidfd_inode):
        _integer(value, "owned termination descriptor or pin")
    objects = (runner, scope, endpoint, runner.store, runner.store.metadata, runner.store.path,
               runner.process, runner.manifest, runner.forwarded,
               endpoint.listener, endpoint.connection, endpoint.hello)
    values = (record.pidfd, scope.slice_fd, scope.service_fd, scope.mount_id, scope.mount_point,
              scope.pidfd_device, scope.pidfd_inode, endpoint.listener.fileno(), endpoint.connection.fileno(),
              endpoint.listener_device, endpoint.listener_inode, endpoint.connection_device,
              endpoint.connection_inode, endpoint.peer_pid, endpoint.peer_uid, runner.store.directory,
              runner.store.metadata.st_dev, runner.store.metadata.st_ino, runner.store.metadata.st_uid)
    return objects, values, _canonical(runner.manifest)


def _same_termination_resources(record: LinuxProvisionalLaunch, identity: LinuxInvocationIdentity,
                                expected: tuple[tuple[object, ...], tuple[object, ...], bytes]) -> None:
    _validate_identity(identity)
    _validate_record(record)
    if record.identity != identity:
        raise EnforcementUnavailable("Termination invocation changed")
    objects, values, manifest = _termination_resources(record)
    if any(actual is not saved for actual, saved in zip(objects, expected[0])) or len(objects) != len(expected[0]):
        raise EnforcementUnavailable("Termination resource reference changed")
    if len(values) != len(expected[1]) or any(type(actual) is not type(saved) or actual != saved for actual, saved in zip(values, expected[1])) or manifest != expected[2]:
        raise EnforcementUnavailable("Termination resource pin or manifest changed")
    _termination_owner_guard(record)


def _pidfd_pin(descriptor: int, device: int, inode: int) -> None:
    _integer(descriptor, "pidfd")
    _integer(device, "pidfd device")
    _integer(inode, "pidfd inode", positive=True)
    metadata = os.fstat(descriptor)
    _integer(metadata.st_dev, "observed pidfd device")
    _integer(metadata.st_ino, "observed pidfd inode", positive=True)
    if (metadata.st_dev, metadata.st_ino) != (device, inode):
        raise EnforcementUnavailable("Authenticated pidfd descriptor was replaced")


def _pidfd_exited(descriptor: int, device: int, inode: int, *, timeout: float = 0.0) -> bool:
    _pidfd_pin(descriptor, device, inode)
    ready = select.select([descriptor], [], [], timeout)[0]
    _pidfd_pin(descriptor, device, inode)
    if type(ready) is not list or any(type(value) is not int or value != descriptor for value in ready):
        raise EnforcementUnavailable("Pidfd wait returned an unrelated descriptor")
    return bool(ready)


def _pidfd_target(descriptor: int, target: int) -> bool:
    _integer(descriptor, "pidfd")
    _integer(target, "pidfd target", positive=True)
    raw = _read_path(Path(f"/proc/self/fdinfo/{descriptor}"), bound=4096).decode("ascii")
    values: dict[str, str] = {}
    for line in raw.splitlines():
        name, separator, value = line.partition(":")
        if not separator or name in values:
            raise EnforcementUnavailable("Malformed pidfd descriptor information")
        values[name] = value.strip()
    if "Pid" not in values or "NSpid" not in values:
        raise EnforcementUnavailable("Descriptor is not an authenticated pidfd")
    if values["Pid"] == "-1":
        if values["NSpid"] not in ("-1", ""):
            raise EnforcementUnavailable("Exited pidfd has inconsistent namespace identity")
        return False
    if _number(values["Pid"], "pidfd target", positive=True) != target:
        raise EnforcementUnavailable("Pidfd target differs from the admitted MainPID")
    namespace = values["NSpid"].split()
    if not namespace or len(namespace) > 16 or any(_number(value, "pidfd namespace PID", positive=True) <= 0 for value in namespace) or _number(namespace[0], "pidfd namespace target", positive=True) != target:
        raise EnforcementUnavailable("Pidfd namespace target differs")
    return True


def _signal_owned_pidfd(record: LinuxProvisionalLaunch, identity: LinuxInvocationIdentity,
                        resources: tuple[tuple[object, ...], tuple[object, ...], bytes],
                        requested_signal: int, deadline: float) -> bool:
    """Observers run before this helper; final verification has no callbacks."""
    _deadline(deadline)
    if type(requested_signal) not in (int, signal.Signals) or requested_signal not in (signal.SIGTERM, signal.SIGKILL):
        raise ValueError("Only owned TERM or KILL signals are supported")
    _same_termination_resources(record, identity, resources)
    scope = record.scope_fd
    owned_store = record.runner.store
    original, device, inode = record.pidfd, scope.pidfd_device, scope.pidfd_inode
    _pidfd_pin(original, device, inode)
    if _pidfd_exited(original, device, inode):
        return True
    if not _pidfd_target(original, identity.main_pid):
        if _pidfd_exited(original, device, inode):
            return True
        raise EnforcementUnavailable("Pidfd target disappeared without observed exit")
    fcntl = importlib.import_module("fcntl")
    if not hasattr(fcntl, "F_DUPFD_CLOEXEC") or not callable(getattr(signal, "pidfd_send_signal", None)):
        raise EnforcementUnavailable("Atomic pidfd signaling is unavailable")
    duplicate = fcntl.fcntl(original, fcntl.F_DUPFD_CLOEXEC, 3)
    _integer(duplicate, "private pidfd")
    if duplicate < 3 or duplicate == original:
        raise EnforcementUnavailable("Atomic private pidfd duplication failed")
    try:
        _same_termination_resources(record, identity, resources)
        _pidfd_pin(original, device, inode)
        _pidfd_pin(duplicate, device, inode)
        if not _pidfd_target(duplicate, identity.main_pid):
            if _pidfd_exited(duplicate, device, inode):
                return True
            raise EnforcementUnavailable("Private pidfd has no live target or observed exit")
        if _pidfd_exited(duplicate, device, inode):
            return True
        _same_termination_resources(record, identity, resources)
        _deadline(deadline)
        _pidfd_pin(original, device, inode)
        _pidfd_pin(duplicate, device, inode)
        try:
            signal.pidfd_send_signal(duplicate, requested_signal, None, 0)
        except ProcessLookupError:
            if not _pidfd_exited(duplicate, device, inode):
                raise EnforcementUnavailable("ESRCH without actual retained pidfd exit")
            return True
        return _pidfd_exited(duplicate, device, inode)
    finally:
        try:
            _pidfd_pin(duplicate, device, inode)
            os.close(duplicate)
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            _unresolved_launches.append(("termination-private-fd", duplicate, device, inode, owned_store))
            try:
                _same_termination_resources(record, identity, resources)
                _termination_emit(record, "termination-private-close-failed", {"generation": identity.generation,
                                  "descriptor": duplicate, "device": device, "inode": inode,
                                  "error_type": type(error).__name__}, deadline)
            except (OSError, RuntimeError, ValueError, TypeError, TimeoutError):
                pass
            raise


def _wait_pidfd_exit(record: LinuxProvisionalLaunch, identity: LinuxInvocationIdentity,
                     resources: tuple[tuple[object, ...], tuple[object, ...], bytes], deadline: float) -> bool:
    _same_termination_resources(record, identity, resources)
    scope = record.scope_fd
    exited = _pidfd_exited(record.pidfd, scope.pidfd_device, scope.pidfd_inode, timeout=_left(deadline))
    _same_termination_resources(record, identity, resources)
    return exited


def _termination_native_snapshot(adapter: LinuxAdapter, record: LinuxProvisionalLaunch,
                                 identity: LinuxInvocationIdentity,
                                 resources: tuple[tuple[object, ...], tuple[object, ...], bytes],
                                 deadline: float) -> tuple[bool, bool]:
    _deadline(deadline)
    _same_termination_resources(record, identity, resources)
    scope = record.scope_fd
    record.runner.store.verify()
    _same_termination_resources(record, identity, resources)
    if sys.platform != "linux" or os.getuid() != identity.uid or _boot_id() != identity.boot_id or _mount() != (scope.mount_id, scope.mount_point):
        raise EnforcementUnavailable("Termination native boot, UID or mount changed")
    for descriptor, device, inode in ((scope.slice_fd, identity.slice_device, identity.slice_inode),
                                      (scope.service_fd, identity.service_device, identity.service_inode)):
        metadata = os.fstat(descriptor)
        if (metadata.st_uid, metadata.st_dev, metadata.st_ino) != (identity.uid, device, inode):
            raise EnforcementUnavailable("Termination controller identity changed")
        _filesystem_magic(descriptor)
    slice_props = _properties(identity.slice_unit, ("InvocationID", "ControlGroup", "MemoryMax", "MemorySwapMax", "ActiveState"), deadline)
    service = _properties(identity.service_unit, ("InvocationID", "ControlGroup", "MainPID", "ExecMainPID", "ActiveState", "SubState"), deadline)
    if slice_props["InvocationID"] != identity.slice_invocation_id or service["InvocationID"] != identity.service_invocation_id or slice_props["ControlGroup"] != identity.slice_relative_path or slice_props["ActiveState"] != "active":
        raise EnforcementUnavailable("Termination manager invocation or persistent scope changed")
    if _number(slice_props["MemoryMax"], "termination manager cap", positive=True) != identity.cap_bytes or _number(slice_props["MemorySwapMax"], "termination manager swap") != 0 or _number(_read_at(scope.slice_fd, "memory.max"), "termination kernel cap", positive=True) != identity.cap_bytes or _number(_read_at(scope.slice_fd, "memory.swap.max"), "termination kernel swap") != 0:
        raise EnforcementUnavailable("Termination charged-domain ceiling changed")
    _ancestor_limits(scope.mount_point, identity.slice_relative_path.rsplit("/", 1)[0], identity.cap_bytes)
    census = _recursive_members(scope.slice_fd, identity.slice_relative_path)
    members = tuple(pid for _, values in census for pid in values)
    if members not in ((), (identity.main_pid,)) or _number(service["ExecMainPID"], "termination ExecMainPID", positive=True) != identity.main_pid:
        raise EnforcementUnavailable("Termination census or retained workload changed")
    main = _number(service["MainPID"], "termination MainPID")
    exited = _pidfd_exited(record.pidfd, scope.pidfd_device, scope.pidfd_inode)
    _same_termination_resources(record, identity, resources)
    _deadline(deadline)
    if not exited:
        if main != identity.main_pid or members != (identity.main_pid,):
            raise EnforcementUnavailable("Live termination workload is not the exact owned member")
        _validate_evidence(adapter.inspect(record, deadline), identity)
        _same_termination_resources(record, identity, resources)
        _deadline(deadline)
        return _pidfd_exited(record.pidfd, scope.pidfd_device, scope.pidfd_inode), False
    if members or main not in (0, identity.main_pid):
        raise EnforcementUnavailable("Exited pidfd has a foreign or unresolved member")
    if main == 0 and service["SubState"] in ("exited", "failed", "dead"):
        _validate_evidence(adapter.inspect(record, deadline), identity, terminal=True)
        _same_termination_resources(record, identity, resources)
        _deadline(deadline)
        return True, True
    if service["ControlGroup"] != identity.service_relative_path or service["SubState"] not in ("running", "stop-sigterm", "stop-sigkill", "stop-post", "final-sigterm", "final-sigkill"):
        raise EnforcementUnavailable("Manager state is not an authenticated pending reap")
    return True, False


def _wait_owned_terminal(adapter: LinuxAdapter, record: LinuxProvisionalLaunch,
                         identity: LinuxInvocationIdentity,
                         resources: tuple[tuple[object, ...], tuple[object, ...], bytes], deadline: float) -> None:
    while True:
        exited, reaped = _termination_native_snapshot(adapter, record, identity, resources, deadline)
        if exited and reaped:
            return
        if not exited:
            raise TimeoutError("Owned pidfd has not exited within bounded termination waits")
        select.select([], [], [], min(0.05, _left(deadline)))
        _same_termination_resources(record, identity, resources)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _read_fd(descriptor: int, *, bound: int = _READ_BOUND) -> bytes:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise EnforcementUnavailable("Expected a regular bounded evidence file")
    result = bytearray()
    while len(result) <= bound:
        piece = os.read(descriptor, min(4096, bound + 1 - len(result)))
        if not piece:
            return bytes(result)
        result.extend(piece)
    raise EnforcementUnavailable("Readback exceeded its finite bound")


def _read_path(path: Path, *, bound: int = _READ_BOUND) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        return _read_fd(descriptor, bound=bound)
    finally:
        os.close(descriptor)


def _read_at(directory: int, name: str) -> str:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
    try:
        return _read_fd(descriptor).decode("ascii").strip()
    finally:
        os.close(descriptor)


def _number(text: str, name: str, *, positive: bool = False) -> int:
    if type(text) is not str or not re.fullmatch(r"0|[1-9][0-9]*", text) or len(text) > 19:
        raise EnforcementUnavailable(f"Noncanonical native {name}")
    return _integer(int(text), name, positive=positive)


def _process_identity(pid: int) -> tuple[int, float]:
    _integer(pid, "pid", positive=True)
    raw = _read_path(Path(f"/proc/{pid}/stat"), bound=8192).decode("ascii")
    prefix, opening, _ = raw.partition(" (")
    closing = raw.rfind(")")
    if not opening or _number(prefix, "proc stat pid", positive=True) != pid or closing < len(prefix) + 2 or raw[closing + 1:closing + 2] != " ":
        raise EnforcementUnavailable("Malformed native process stat framing")
    tail = raw[raw.rfind(")") + 2:].split()
    if len(tail) < 20 or tail[0] not in tuple("RSDTtWXxKWPIN"):
        raise EnforcementUnavailable("Workload process is absent or unreaped")
    ticks = _number(tail[19], "start_ticks", positive=True)
    lines = _read_path(Path("/proc/stat"), bound=1048576).decode("ascii").splitlines()
    boot = [line.split()[1] for line in lines if line.startswith("btime ")]
    if len(boot) != 1:
        raise EnforcementUnavailable("Cannot verify process birth clock")
    clock = os.sysconf("SC_CLK_TCK")
    _integer(clock, "clock ticks", positive=True)
    birth = float(_number(boot[0], "boot time", positive=True) + ticks / clock)
    if not math.isfinite(birth) or birth <= 0:
        raise EnforcementUnavailable("Invalid process birth candidate")
    return ticks, birth


def _boot_id() -> str:
    value = _read_path(Path("/proc/sys/kernel/random/boot_id"), bound=128).decode("ascii").strip()
    if _UUID.fullmatch(value) is None:
        raise EnforcementUnavailable("Malformed current boot identity")
    return value


def _load_capsule() -> object:
    path = Path(__file__).absolute().parent / "truememory" / "memory_bootstrap.py"
    raw = _read_path(path, bound=1048576)
    if hashlib.sha256(raw).hexdigest() != "a33c0f4bf25bfa590390f50296623455491333ace98da97e4a3511e6aee50c5d":
        raise EnforcementUnavailable("Protected readback capsule changed")
    name = "_truememory_owned_readback_capsule"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise EnforcementUnavailable("Absolute readback capsule is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _StatFS(ctypes.Structure):
    _fields_ = [("f_type", ctypes.c_long), ("f_bsize", ctypes.c_long),
                ("f_blocks", ctypes.c_ulong), ("f_bfree", ctypes.c_ulong),
                ("f_bavail", ctypes.c_ulong), ("f_files", ctypes.c_ulong),
                ("f_ffree", ctypes.c_ulong), ("f_fsid", ctypes.c_int * 2),
                ("f_namelen", ctypes.c_long), ("f_frsize", ctypes.c_long),
                ("f_flags", ctypes.c_long), ("f_spare", ctypes.c_long * 4)]


def _filesystem_magic(descriptor: int) -> int:
    if sys.platform != "linux" or ctypes.sizeof(ctypes.c_long) != 8 or os.uname().machine not in ("x86_64", "aarch64"):
        raise EnforcementUnavailable("No validated Linux statfs ABI for this architecture")
    function = ctypes.CDLL(None, use_errno=True).fstatfs
    function.argtypes = (ctypes.c_int, ctypes.POINTER(_StatFS))
    function.restype = ctypes.c_int
    result = _StatFS()
    if function(descriptor, ctypes.byref(result)) != 0:
        raise OSError(ctypes.get_errno(), "Owned cgroup fstatfs unavailable")
    if result.f_type != _CGROUP2_MAGIC:
        raise EnforcementUnavailable("Controller directory is not authentic cgroup2fs")
    return result.f_type


def _unescape_mount(value: str) -> str:
    escapes = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}
    if any(match.group(1) not in escapes for match in re.finditer(r"\\([^ ]{3})", value)):
        raise EnforcementUnavailable("Unsupported mountinfo escape")
    remainder = re.sub(r"\\(?:040|011|012|134)", "", value)
    if "\\" in remainder:
        raise EnforcementUnavailable("Malformed mountinfo escape")
    return re.sub(r"\\(040|011|012|134)", lambda match: escapes[match.group(1)], value)


def _mount() -> tuple[int, str]:
    raw = _read_path(Path("/proc/self/mountinfo"), bound=1048576).decode("utf-8")
    candidates = []
    for line in raw.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            raise EnforcementUnavailable("Malformed mountinfo")
        fields, kind = before.split(), after.split()
        if len(fields) < 6 or len(kind) < 3:
            raise EnforcementUnavailable("Malformed mount record")
        if kind[0] == "cgroup2":
            if _unescape_mount(fields[3]) != "/":
                raise EnforcementUnavailable("Unsupported delegated cgroup namespace root")
            point = _unescape_mount(fields[4])
            if not Path(point).is_absolute() or "\x00" in point:
                raise EnforcementUnavailable("Invalid cgroup mount point")
            candidates.append((_number(fields[0], "mount id", positive=True), point))
    if len(candidates) != 1:
        raise EnforcementUnavailable("Exactly one authenticated cgroup2 mount is required")
    return candidates[0]


def _membership(pid: int) -> str:
    raw = _read_path(Path(f"/proc/{pid}/cgroup"), bound=_READ_BOUND).decode("ascii").splitlines()
    values = [line[3:] for line in raw if line.startswith("0::")]
    if len(values) != 1:
        raise EnforcementUnavailable("Missing unified cgroup membership")
    return _relative(values[0], "actual membership")


def _open_directory(path: Path, *, uid: int | None = None) -> int:
    if not path.is_absolute():
        raise EnforcementUnavailable("Controller path must be absolute")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            if part in ("", ".", ".."):
                raise EnforcementUnavailable("Noncanonical native directory")
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if uid is not None and metadata.st_uid != uid:
            raise EnforcementUnavailable("Owned directory UID changed")
        return descriptor
    except (OSError, ValueError, RuntimeError):
        os.close(descriptor)
        raise


def _controller_events(descriptor: int, name: str = "memory.events") -> tuple[tuple[str, int], ...]:
    values = []
    for line in _read_at(descriptor, name).splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise EnforcementUnavailable("Malformed controller events")
        values.append((fields[0], _number(fields[1], "controller event")))
    return _events(tuple(values), "native memory events")


def _members(descriptor: int) -> tuple[int, ...]:
    values = tuple(_number(line, "controller member", positive=True) for line in _read_at(descriptor, "cgroup.procs").splitlines())
    if len(values) > 4096 or len(set(values)) != len(values):
        raise EnforcementUnavailable("Controller membership exceeds the ownership bound")
    return values


def _recursive_members(descriptor: int, relative: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    result = []
    pending = [(os.dup(descriptor), relative, 0)]
    try:
        while pending:
            current, path, depth = pending.pop()
            try:
                if len(result) >= 128 or depth > 16:
                    raise EnforcementUnavailable("Owned recursive census exceeds its bound")
                _filesystem_magic(current)
                result.append((path, _members(current)))
                with os.scandir(current) as entries:
                    for entry in entries:
                        if entry.is_symlink():
                            raise EnforcementUnavailable("Symlink in owned cgroup census")
                        if entry.is_dir(follow_symlinks=False):
                            if len(pending) + len(result) >= 128:
                                raise EnforcementUnavailable("Owned recursive census exceeds its bound")
                            child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
                            pending.append((child, path + "/" + entry.name, depth + 1))
            finally:
                os.close(current)
        return tuple(sorted(result))
    finally:
        for descriptor, _, _ in pending:
            os.close(descriptor)


def _hash_path(path: Path, *, bound: int = 16777216) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    digest = hashlib.sha256()
    total = 0
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise EnforcementUnavailable("Artifact is not a regular file")
        while True:
            piece = os.read(descriptor, min(65536, bound + 1 - total))
            if not piece:
                return digest.hexdigest()
            total += len(piece)
            if total > bound:
                raise EnforcementUnavailable("Artifact exceeds retained hashing bound")
            digest.update(piece)
    finally:
        os.close(descriptor)


def _exclusive_file(directory: int, name: str, raw: bytes) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        raise ValueError("Unsafe exclusive artifact name")
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
    try:
        position = 0
        while position < len(raw):
            written = os.write(descriptor, raw[position:])
            if written <= 0:
                raise EnforcementUnavailable("Exclusive evidence write made no progress")
            position += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory)


def _executable_pin(executable: str) -> dict[str, object]:
    _text(executable, "selected interpreter", bound=4096)
    selected = Path(executable)
    if not selected.is_absolute():
        raise EnforcementUnavailable("Selected interpreter is not absolute")
    resolved = selected.resolve(strict=True)
    metadata = os.stat(selected)
    target = os.stat(resolved, follow_symlinks=False)
    if not stat.S_ISREG(target.st_mode) or (metadata.st_dev, metadata.st_ino) != (target.st_dev, target.st_ino) or not os.access(selected, os.X_OK):
        raise EnforcementUnavailable("Selected interpreter target is not a stable executable")
    digest = _hash_path(resolved, bound=67108864)
    after = os.stat(selected)
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns):
        raise EnforcementUnavailable("Selected interpreter changed during readback")
    return {"path": executable, "resolved_path": str(resolved), "device": target.st_dev,
            "inode": target.st_ino, "sha256": digest}


def _private_artifact(directory: int, name: str, *, durable: bool = False,
                      bound: int = _READ_BOUND) -> tuple[bytes, str]:
    if type(name) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        raise ValueError("Unsafe private artifact name")
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) != 0o600:
            raise EnforcementUnavailable("Retained artifact is not a private regular file")
        raw = _read_fd(descriptor, bound=bound)
        if durable:
            os.fsync(descriptor)
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_uid", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) or getattr(after, field) != getattr(current, field) for field in fields):
            raise EnforcementUnavailable("Retained artifact changed during readback")
        if durable:
            os.fsync(directory)
        return raw, hashlib.sha256(raw).hexdigest()
    finally:
        os.close(descriptor)


def _barrier_pin(value: object) -> dict[str, int]:
    if type(value) is not dict or any(type(key) is not str for key in value) or set(value) != {"descriptor", "device", "inode"}:
        raise EnforcementUnavailable("Malformed retained barrier pin")
    for name in value:
        _integer(value[name], "barrier " + name, positive=True)
    if value["descriptor"] < 3 or value["descriptor"] > 2147483647:
        raise EnforcementUnavailable("Retained barrier descriptor is out of range")
    return value


def _preexec_payload(identity: LinuxInvocationIdentity, identity_sha: str, barrier: object) -> dict[str, object]:
    _validate_identity(identity)
    _hex(identity_sha, 64, "retained identity hash")
    return {"version": 1, "generation": identity.generation, "manifest_sha256": identity.manifest_sha256,
            "pid": identity.main_pid, "start_ticks": identity.start_ticks, "identity_sha256": identity_sha,
            "bootstrap_sha256": identity.bootstrap_sha256, "kernel_cap_bytes": identity.cap_bytes,
            "metric": _METRIC, "physical_hard_cap": False, "dynamic_registry_cap": False,
            "barrier": _barrier_pin(barrier)}


def _receipt_chain(directory: int, *, expected: dict[str, str] | None = None) -> tuple[int, str, list[dict[str, object]]]:
    names = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if len(names) >= 4120:
                raise EnforcementUnavailable("Retained generation exceeds its file bound")
            names.append(entry.name)
    receipts = sorted(name for name in names if name.startswith("receipt-"))
    if len(receipts) > 4096 or any(re.fullmatch(r"receipt-[0-9]{6}\.json", name) is None for name in receipts):
        raise EnforcementUnavailable("Malformed retained receipt sequence")
    chain = "0" * 64
    payloads = []
    for sequence, name in enumerate(receipts, 1):
        if name != f"receipt-{sequence:06d}.json":
            raise EnforcementUnavailable("Retained receipt sequence has a gap")
        raw, digest = _private_artifact(directory, name)
        if expected is not None and expected.get(name) != digest:
            raise EnforcementUnavailable("Retained receipt hash changed")
        item = _strict_json(raw)
        if type(item) is not dict or set(item) != {"version", "sequence", "previous_sha256", "phase", "payload", "monotonic"}:
            raise EnforcementUnavailable("Malformed retained receipt")
        _literal(item["version"], 1, "receipt version")
        _literal(item["sequence"], sequence, "receipt sequence")
        _literal(item["previous_sha256"], chain, "receipt previous hash")
        _text(item["phase"], "receipt phase")
        if type(item["monotonic"]) is not float or not math.isfinite(item["monotonic"]) or item["monotonic"] <= 0:
            raise EnforcementUnavailable("Invalid retained receipt clock")
        chain = digest
        payloads.append(item)
    return len(receipts), chain, payloads


class _EvidenceStore:
    def __init__(self, path: Path, manifest: dict[str, object]) -> None:
        self.path = path
        self.directory = _open_directory(path, uid=os.getuid())
        self.metadata = os.fstat(self.directory)
        self.lock = threading.RLock()
        self.sequence = 0
        self.chain = "0" * 64
        self.hashes: dict[str, str] = {}
        self.manifest_raw = _canonical(manifest)
        self.manifest_sha256 = hashlib.sha256(self.manifest_raw).hexdigest()
        _exclusive_file(self.directory, "manifest-v1.json", self.manifest_raw)
        self.hashes["manifest-v1.json"] = self.manifest_sha256

    def append(self, phase: str, payload: object) -> tuple[int, str]:
        with self.lock:
            if self.sequence >= 4096:
                raise EnforcementUnavailable("Retained receipt sequence exceeds its bound")
            sequence = self.sequence + 1
            raw = _canonical({"version": 1, "sequence": sequence, "previous_sha256": self.chain,
                              "phase": phase, "payload": payload, "monotonic": time.monotonic()})
            name = f"receipt-{sequence:06d}.json"
            _exclusive_file(self.directory, name, raw)
            self.sequence = sequence
            self.chain = hashlib.sha256(raw).hexdigest()
            self.hashes[name] = self.chain
            return sequence, self.chain

    def retain(self, name: str, payload: object) -> None:
        raw = _canonical(payload)
        _exclusive_file(self.directory, name, raw)
        self.hashes[name] = hashlib.sha256(raw).hexdigest()

    def observe(self, name: str, *, durable: bool = False, bound: int = _READ_BOUND) -> None:
        with self.lock:
            _, digest = _private_artifact(self.directory, name, durable=durable, bound=bound)
            if name in self.hashes and self.hashes[name] != digest:
                raise EnforcementUnavailable("Retained immutable artifact changed")
            self.hashes[name] = digest

    def verify(self) -> None:
        with self.lock:
            current = os.stat(self.path, follow_symlinks=False)
            metadata = os.fstat(self.directory)
            if (current.st_dev, current.st_ino, metadata.st_dev, metadata.st_ino, metadata.st_uid,
                stat.S_IMODE(metadata.st_mode)) != (self.metadata.st_dev, self.metadata.st_ino,
                self.metadata.st_dev, self.metadata.st_ino, os.getuid(), 0o700):
                raise EnforcementUnavailable("Retained generation directory changed")
            sequence, chain, _ = _receipt_chain(self.directory, expected=self.hashes)
            if (sequence, chain) != (self.sequence, self.chain):
                raise EnforcementUnavailable("Retained receipt chain differs from owned lifetime")
            for name, expected in self.hashes.items():
                _, digest = _private_artifact(self.directory, name, bound=16777216 if name == "stderr-v1.log" else _READ_BOUND)
                if digest != expected:
                    raise EnforcementUnavailable("Retained owned artifact changed")
            os.fsync(self.directory)


def _run_command(arguments: list[str], deadline: float, *, environment: dict[str, str] | None = None) -> tuple[subprocess.Popen[bytes], bytes]:
    _deadline(deadline)
    environment = _manager_environment(environment)
    process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=environment, close_fds=True)
    output: dict[int, bytearray] = {}
    streams = []
    try:
        for stream in (process.stdout, process.stderr):
            if stream is None:
                raise EnforcementUnavailable("Manager command lacks bounded output handles")
            os.set_blocking(stream.fileno(), False)
            streams.append(stream)
            output[stream.fileno()] = bytearray()
        active = list(output)
        while active:
            ready, _, _ = select.select(active, [], [], _left(deadline))
            if not ready:
                raise TimeoutError("Manager command deadline expired")
            for descriptor in ready:
                piece = os.read(descriptor, 4096)
                if not piece:
                    active.remove(descriptor)
                else:
                    output[descriptor].extend(piece)
                    if sum(map(len, output.values())) > _READ_BOUND:
                        raise EnforcementUnavailable("Manager property output exceeds its bound")
        status = process.wait(timeout=_left(deadline))
        if status != 0:
            raise EnforcementUnavailable(f"Owned manager command failed with status {status}")
        return process, bytes(output[process.stdout.fileno()])
    except (OSError, RuntimeError, ValueError, TimeoutError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                _unresolved_launches.append(process)
        raise
    finally:
        for stream in streams:
            stream.close()


def _properties(unit: str, names: tuple[str, ...], deadline: float) -> dict[str, str]:
    _, raw = _run_command(["/usr/bin/systemctl", "--user", "show", unit, "--no-pager",
                           "--property=" + ",".join(names)], deadline)
    result = {}
    for line in raw.decode("utf-8").splitlines():
        name, separator, value = line.partition("=")
        if not separator or name not in names or name in result:
            raise EnforcementUnavailable("Malformed bounded manager properties")
        result[name] = value
    if set(result) != set(names):
        raise EnforcementUnavailable("Manager omitted required properties")
    return result


def _packet_send(endpoint: socket.socket, value: object, deadline: float) -> None:
    raw = _canonical(value)
    if len(raw) + 4 > _READ_BOUND:
        raise EnforcementUnavailable("Barrier packet exceeds its bound")
    endpoint.settimeout(_left(deadline))
    endpoint.sendall(struct.pack(">I", len(raw)) + raw)


def _packet_receive(endpoint: socket.socket, deadline: float) -> dict[str, object]:
    def receive(length: int) -> bytes:
        value = bytearray()
        while len(value) < length:
            endpoint.settimeout(_left(deadline))
            piece = endpoint.recv(min(4096, length - len(value)))
            if not piece:
                raise EnforcementUnavailable("Owned barrier peer closed before completion")
            value.extend(piece)
        return bytes(value)
    length = struct.unpack(">I", receive(4))[0]
    if length == 0 or length + 4 > _READ_BOUND:
        raise EnforcementUnavailable("Invalid barrier packet length")
    value = _strict_json(receive(length))
    if type(value) is not dict:
        raise EnforcementUnavailable("Barrier packet must be one object")
    return value


def _peer(endpoint: socket.socket) -> tuple[int, int]:
    pid, uid, _ = struct.unpack("3i", endpoint.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    _integer(pid, "peer PID", positive=True)
    _integer(uid, "peer UID")
    return pid, uid


@dataclass
class _NativeScope:
    mount_id: int
    mount_point: str
    slice_fd: int
    service_fd: int
    baseline: tuple[tuple[str, int], ...]
    baseline_local: tuple[tuple[str, int], ...] = ()
    leaf_baseline: tuple[tuple[str, int], ...] = ()
    leaf_baseline_local: tuple[tuple[str, int], ...] = ()
    pidfd_device: int = 0
    pidfd_inode: int = 0


@dataclass
class _NativeRunner:
    process: subprocess.Popen[bytes]
    store: _EvidenceStore
    manifest: dict[str, object]
    stop_complete: bool = False
    forwarded: Mapping[str, str] | None = None


@dataclass
class _NativeEndpoint:
    listener: socket.socket
    connection: socket.socket
    peer_pid: int
    peer_uid: int
    hello: dict[str, object]
    listener_device: int = 0
    listener_inode: int = 0
    connection_device: int = 0
    connection_inode: int = 0


_unresolved_launches: list[object] = []
_child_context: tuple[Path, LinuxInvocationIdentity, socket.socket, int, int] | None = None


def _original_environment(value: object, *, generation: str, names: list[str] | None = None) -> dict[str, str]:
    if type(value) not in (dict, MappingProxyType) or not value or len(value) > 4096:
        raise EnforcementUnavailable("Original environment snapshot is malformed or exceeds its name bound")
    result = {}
    for name, item in value.items():
        if type(name) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None or type(item) is not str or "\x00" in item:
            raise EnforcementUnavailable("Original environment snapshot contains an invalid string")
        result[name] = item
    if names is not None and sorted(result) != names:
        raise EnforcementUnavailable("Original environment snapshot names differ from the frozen manifest")
    for name, expected in ((_GENERATION_ENV, generation), (_FLAG, "1"), ("TRUEMEMORY_CAP_CHILD", "1")):
        _literal(result.get(name), expected, "original environment marker " + name)
    return result


def _original_environment_packet(value: object, *, generation: str, manifest_sha256: str,
                                 names: list[str] | None = None) -> dict[str, object]:
    _hex(generation, 32, "environment generation")
    _hex(manifest_sha256, 64, "environment manifest pin")
    packet = {"generation": generation, "manifest_sha256": manifest_sha256, "preexec_witness_retained": True,
              "forwarded_environment": _original_environment(value, generation=generation, names=names)}
    if len(_canonical(packet)) + 4 > _READ_BOUND:
        raise EnforcementUnavailable("Complete original environment frame exceeds its bound")
    return packet


def _manager_environment(environment: dict[str, str] | None) -> dict[str, str]:
    supplied = dict(os.environ if environment is None else environment)
    runtime = f"/run/user/{os.getuid()}"
    address = "unix:path=" + runtime + "/bus"
    for name, expected in (("XDG_RUNTIME_DIR", runtime), ("DBUS_SESSION_BUS_ADDRESS", address)):
        if name in supplied and supplied[name] != expected:
            raise EnforcementUnavailable("Required local manager transport has an unsupported address")
        supplied[name] = expected
    return supplied


def _ancestor_limits(mount_point: str, relative: str, cap_bytes: int) -> tuple[tuple[str, str], ...]:
    result = []
    parts = relative[1:].split("/")
    if len(parts) > 32:
        raise EnforcementUnavailable("Native ancestor chain exceeds its bound")
    for count in range(len(parts), 0, -1):
        path = "/" + "/".join(parts[:count])
        descriptor = _open_directory(Path(mount_point + path))
        try:
            _filesystem_magic(descriptor)
            maximum = _read_at(descriptor, "memory.max")
            if maximum != "max" and _number(maximum, "ancestor memory.max") < cap_bytes:
                raise EnforcementUnavailable("An existing stricter ancestor cap cannot be represented as selected C")
            result.append((path, maximum))
        finally:
            os.close(descriptor)
    return tuple(result)


def _required_linux_support(policy: LinuxDaemonLaunchPolicy, deadline: float) -> tuple[int, str, str, tuple[tuple[str, str], ...]]:
    _validate_policy(policy)
    _deadline(deadline)
    if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(socket, "SO_PEERCRED"):
        raise EnforcementUnavailable("Required Linux ownership primitives are unavailable")
    if ctypes.sizeof(ctypes.c_long) != 8 or os.uname().machine not in ("x86_64", "aarch64"):
        raise EnforcementUnavailable("Required Linux filesystem ABI is unsupported")
    _manager_environment(None)
    runtime = Path(f"/run/user/{os.getuid()}")
    runtime_fd = _open_directory(runtime, uid=os.getuid())
    try:
        if stat.S_IMODE(os.fstat(runtime_fd).st_mode) != 0o700:
            raise EnforcementUnavailable("User manager runtime directory is not private")
        bus = os.stat("bus", dir_fd=runtime_fd, follow_symlinks=False)
        if not stat.S_ISSOCK(bus.st_mode) or bus.st_uid != os.getuid():
            raise EnforcementUnavailable("User manager bus endpoint is not owned")
    finally:
        os.close(runtime_fd)
    _load_capsule()
    _, raw = _run_command(["/usr/bin/busctl", "--user", "get-property", "org.freedesktop.systemd1",
                          "/org/freedesktop/systemd1", "org.freedesktop.systemd1.Manager", "Version"], deadline)
    match = re.fullmatch(r's "([0-9]{3})([A-Za-z0-9._+~()-]*)"\n?', raw.decode("ascii"))
    if match is None or int(match.group(1)) < 249:
        raise EnforcementUnavailable("Required running user-manager version is unsupported")
    parent = _properties("app.slice", ("LoadState", "ActiveState", "ControlGroup"), deadline)
    if parent["LoadState"] != "loaded" or parent["ActiveState"] != "active":
        raise EnforcementUnavailable("An existing active delegated app.slice is required")
    parent_relative = _relative(parent["ControlGroup"], "existing app slice")
    if not parent_relative.endswith("/app.slice"):
        raise EnforcementUnavailable("Manager app.slice is not the admitted existing parent")
    _membership(os.getpid())
    mount_id, mount_point = _mount()
    parent_fd = _open_directory(Path(mount_point + parent_relative), uid=os.getuid())
    try:
        _filesystem_magic(parent_fd)
        if "memory" not in _read_at(parent_fd, "cgroup.controllers").split():
            raise EnforcementUnavailable("Existing app.slice has no memory controller")
    finally:
        os.close(parent_fd)
    ancestors = _ancestor_limits(mount_point, parent_relative, policy.cap_bytes)
    _deadline(deadline)
    return mount_id, mount_point, parent_relative, ancestors


_UNIT_STATE = ("LoadState", "ActiveState", "SubState", "MainPID", "ControlGroup")
_EVIDENCE_KEEP = 8
_SWEEP_LIMIT = 4


def _evidence_root() -> Path:
    return Path.home() / ".truememory" / "memory-cap-launches"


def _generation_units(generation: str) -> tuple[str, str]:
    _hex(generation, 32, "generation")
    return f"app-truememory{generation}.slice", f"truememory-memory{generation}.service"


def _stop_unit(unit: str, deadline: float) -> bool:
    """Stop one generation-scoped unit and clear a failed state so it unloads."""
    state = _properties(unit, _UNIT_STATE, deadline)
    for command in ("stop", "reset-failed"):
        if state["LoadState"] != "loaded":
            return True
        if command == "stop" and state["ActiveState"] in ("inactive", "failed"):
            continue
        if command == "reset-failed" and state["ActiveState"] != "failed":
            continue
        try:
            _run_command(["/usr/bin/systemctl", "--user", command, unit], deadline)
        except EnforcementUnavailable:
            # The manager may collect the unit between readback and command.
            if _properties(unit, _UNIT_STATE, deadline)["LoadState"] == "loaded":
                raise
        state = _properties(unit, _UNIT_STATE, deadline)
    return state["LoadState"] != "loaded" or state["ActiveState"] == "inactive"


def _generation_domain_empty(mount_point: str, slice_relative: str) -> bool:
    try:
        descriptor = _open_directory(Path(mount_point + slice_relative), uid=os.getuid())
    except FileNotFoundError:
        return True
    try:
        _filesystem_magic(descriptor)
        return not any(members for _, members in _recursive_members(descriptor, slice_relative))
    finally:
        os.close(descriptor)


def _retire_generation_units(generation: str, *, mount_point: str, parent_relative: str,
                             deadline: float, settle: float = 0.0) -> bool:
    """Stop this generation's service and slice once its domain is provably empty.

    Unit names are unique to the generation. A unit is stopped only when the
    manager reports no main process, its cgroup is this generation's path and
    the recursive census is empty, so no live or foreign process is affected.
    """
    slice_unit, service_unit = _generation_units(generation)
    slice_relative = _relative(parent_relative + "/" + slice_unit, "retired slice path")
    service_relative = slice_relative + "/" + service_unit
    settle_until = min(deadline, time.monotonic() + settle)
    while True:
        service = _properties(service_unit, _UNIT_STATE, deadline)
        slice_state = _properties(slice_unit, _UNIT_STATE, deadline)
        if service["LoadState"] != "loaded" and slice_state["LoadState"] != "loaded":
            return True
        if service["ControlGroup"] not in ("", service_relative) or slice_state["ControlGroup"] not in ("", slice_relative):
            return False
        if service["MainPID"] == "0" and _generation_domain_empty(mount_point, slice_relative):
            break
        if time.monotonic() >= settle_until:
            return False
        time.sleep(min(0.25, _left(deadline)))
    service_stopped = _stop_unit(service_unit, deadline)
    if not _generation_domain_empty(mount_point, slice_relative):
        return False
    return _stop_unit(slice_unit, deadline) and service_stopped


def _mark_generation_retired(generation: str) -> bool:
    try:
        _hex(generation, 32, "generation")
        directory = _open_directory(_evidence_root() / generation, uid=os.getuid())
    except (OSError, ValueError, RuntimeError):
        return False
    try:
        _exclusive_file(directory, "retired-v1.json", _canonical({"version": 1, "generation": generation}))
    except FileExistsError:
        pass
    except OSError:
        return False
    finally:
        os.close(directory)
    return True


def _incarnation_alive(pid: object, start_ticks: object) -> bool:
    """True unless the recorded incarnation provably no longer exists."""
    if type(pid) is not int or type(start_ticks) is not int or pid <= 0:
        return True
    try:
        ticks, _ = _process_identity(pid)
    except FileNotFoundError:
        return False
    except (OSError, RuntimeError, ValueError):
        return True
    return ticks == start_ticks


def _generation_state(root: int, generation: str) -> str:
    """Classify retained evidence as retired, live (an owner may still act) or orphaned."""
    directory = os.open(generation, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root)
    try:
        metadata = os.fstat(directory)
        if metadata.st_uid != os.getuid():
            return "live"
        try:
            os.stat("retired-v1.json", dir_fd=directory, follow_symlinks=False)
            return "retired"
        except FileNotFoundError:
            pass
        try:
            raw, _ = _private_artifact(directory, "manifest-v1.json")
        except FileNotFoundError:
            # A launcher writes its manifest immediately after creating the directory.
            return "orphaned" if time.time() - metadata.st_mtime > 3600 else "live"
        manifest = _strict_json(raw)
        if type(manifest) is not dict:
            return "live"
        if manifest.get("boot_id") != _boot_id():
            return "orphaned"
        return "live" if _incarnation_alive(manifest.get("parent_pid"), manifest.get("parent_start_ticks")) else "orphaned"
    finally:
        os.close(directory)


def _remove_generation_dir(root: int, generation: str) -> bool:
    _hex(generation, 32, "retired generation")
    directory = os.open(generation, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root)
    try:
        metadata = os.fstat(directory)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            return False
        names = []
        with os.scandir(directory) as entries:
            for entry in entries:
                if len(names) >= 4200 or re.fullmatch(r"[A-Za-z0-9_.-]+", entry.name) is None:
                    return False
                item = entry.stat(follow_symlinks=False)
                if item.st_uid != os.getuid() or not (stat.S_ISREG(item.st_mode) or stat.S_ISSOCK(item.st_mode)):
                    return False
                names.append(entry.name)
        for name in names:
            os.unlink(name, dir_fd=directory)
    finally:
        os.close(directory)
    os.rmdir(generation, dir_fd=root)
    return True


def _sweep_retired_generations(*, mount_point: str, parent_relative: str, deadline: float) -> int:
    """Retire units whose launcher is gone and prune old retired evidence.

    Best effort and bounded per call: every capped launch sweeps, so units left
    by a launcher that exited before its server cannot accumulate.
    """
    try:
        root = _open_directory(_evidence_root(), uid=os.getuid())
    except FileNotFoundError:
        return 0
    retired = checks = removals = kept = 0
    try:
        if stat.S_IMODE(os.fstat(root).st_mode) != 0o700:
            return 0
        candidates = []
        with os.scandir(root) as entries:
            for entry in entries:
                if len(candidates) >= 4096:
                    break
                if len(entry.name) == 32 and _HEX.fullmatch(entry.name) and entry.is_dir(follow_symlinks=False):
                    candidates.append((entry.stat(follow_symlinks=False).st_mtime_ns, entry.name))
        for _, generation in sorted(candidates, reverse=True):
            if time.monotonic() >= deadline:
                break
            try:
                state = _generation_state(root, generation)
                if state == "live":
                    continue
                if state == "orphaned":
                    if checks >= _SWEEP_LIMIT:
                        continue
                    checks += 1
                    if not _retire_generation_units(generation, mount_point=mount_point,
                                                    parent_relative=parent_relative, deadline=deadline):
                        continue
                    _mark_generation_retired(generation)
                    retired += 1
                kept += 1
                if kept > _EVIDENCE_KEEP and removals < 2 * _SWEEP_LIMIT:
                    removals += 1
                    _remove_generation_dir(root, generation)
            except TimeoutError:
                break
            except (OSError, RuntimeError, ValueError):
                continue
    finally:
        os.close(root)
    return retired


_adopted_generations: set[str] = set()
_adoption_lock = threading.Lock()


def _retire_after_exit(pidfd: int, identity: LinuxInvocationIdentity) -> None:
    try:
        while not select.select([pidfd], [], [], 1.0)[0]:
            pass
    finally:
        os.close(pidfd)
    try:
        root = _open_directory(_evidence_root(), uid=identity.uid)
        try:
            # A live launcher still owns terminal evidence for this generation.
            if _generation_state(root, identity.generation) != "orphaned":
                return
        finally:
            os.close(root)
        _, mount_point = _mount()
        if _retire_generation_units(identity.generation, mount_point=mount_point,
                                    parent_relative=identity.slice_relative_path.rsplit("/", 1)[0],
                                    deadline=time.monotonic() + 30.0, settle=10.0):
            _mark_generation_retired(identity.generation)
    except (OSError, RuntimeError, ValueError, TimeoutError):
        pass


def adopt_exit_cleanup(evidence: LinuxScopeEvidence) -> bool:
    """Watch a verified joined daemon and retire its units if its launcher is gone at exit."""
    identity = _validate_evidence(evidence, evidence.identity).identity
    with _adoption_lock:
        if identity.generation in _adopted_generations:
            return False
        _adopted_generations.add(identity.generation)
    try:
        pidfd = os.pidfd_open(identity.main_pid)
    except OSError:
        _adopted_generations.discard(identity.generation)
        return False
    try:
        # The pidfd pins one incarnation; prove it is the verified workload.
        if _process_identity(identity.main_pid)[0] != identity.start_ticks or _membership(identity.main_pid) != identity.service_relative_path:
            raise EnforcementUnavailable("Joined workload changed before exit adoption")
    except (OSError, RuntimeError, ValueError):
        os.close(pidfd)
        _adopted_generations.discard(identity.generation)
        return False
    threading.Thread(target=_retire_after_exit, args=(pidfd, identity),
                     name="truememory-capped-exit", daemon=True).start()
    return True


class LinuxAdapter:
    """Actual user-manager adapter. Native support is established by readback."""

    def inspect_existing(self, policy: LinuxDaemonLaunchPolicy, *, pid: int, birth_time: float,
                         generation: str, claim: dict[str, object], deadline: float) -> LinuxScopeEvidence:
        _validate_policy(policy)
        _integer(pid, "existing pid", positive=True)
        _hex(generation, 32, "existing generation")
        _validate_claim(claim, pid=pid, birth_time=birth_time, generation=generation)
        _deadline(deadline)
        if sys.platform != "linux":
            raise EnforcementUnavailable("Readonly charged-domain inspection requires Linux")
        path = Path.home() / ".truememory" / "memory-cap-launches" / generation
        directory = _open_directory(path, uid=os.getuid())
        owned = [directory]
        endpoint = None
        try:
            if stat.S_IMODE(os.fstat(directory).st_mode) != 0o700:
                raise EnforcementUnavailable("Current generation directory is not private")
            manifest_raw, manifest_sha = _private_artifact(directory, "manifest-v1.json")
            identity_raw, identity_sha = _private_artifact(directory, "identity-v1.json")
            manifest, data = _strict_json(manifest_raw), _strict_json(identity_raw)
            if type(manifest) is not dict or type(data) is not dict:
                raise EnforcementUnavailable("Current invocation metadata is malformed")
            identity = LinuxInvocationIdentity(**data)
            if (identity.main_pid, identity.birth_time, identity.generation, identity.uid, identity.boot_id) != (pid, birth_time, generation, os.getuid(), _boot_id()) or identity.cap_bytes > policy.cap_bytes:
                raise EnforcementUnavailable("Current invocation identity or ceiling is incompatible")
            _literal(identity.manifest_sha256, manifest_sha, "current manifest pin")
            _literal(identity.bootstrap_sha256, _hash_path(Path(__file__).absolute(), bound=1048576), "current bootstrap pin")
            for name, expected in (("version", 1), ("generation", generation), ("uid", identity.uid), ("boot_id", identity.boot_id),
                                   ("slice_unit", identity.slice_unit), ("service_unit", identity.service_unit),
                                   ("cap_bytes", identity.cap_bytes), ("swap_bytes", 0), ("bootstrap_sha256", identity.bootstrap_sha256)):
                _literal(manifest.get(name), expected, "current manifest " + name)
            expected_claim_path = Path.home() / ".truememory" / "model_server.start.json"
            _literal(manifest.get("claim_path"), str(expected_claim_path), "current claim path")
            claim_directory = _open_directory(expected_claim_path.parent, uid=identity.uid)
            owned.append(claim_directory)
            claim_raw, claim_sha = _private_artifact(claim_directory, expected_claim_path.name, bound=4096)
            current_claim = _validate_claim(_strict_json(claim_raw), pid=pid, birth_time=birth_time, generation=generation)
            if current_claim != claim or current_claim["launcher"] != {"pid": manifest.get("parent_pid"), "born": manifest.get("parent_birth")}:
                raise EnforcementUnavailable("Current durable claim differs from the requested generation")
            _, _, receipts = _receipt_chain(directory)
            durable = [item["payload"] for item in receipts if item["phase"] == "claim-durable-before-release"]
            if len(durable) != 1 or type(durable[0]) is not dict or durable[0].get("claim_sha256") != claim_sha or durable[0].get("generation") != generation or durable[0].get("pid") != pid or durable[0].get("birth_time") != birth_time:
                raise EnforcementUnavailable("Current claim has no matching durable admission witness")
            preexec_raw, _ = _private_artifact(directory, "pre-reexec-v1.json")
            preexec_data = _strict_json(preexec_raw)
            if type(preexec_data) is not dict:
                raise EnforcementUnavailable("Current pre-exec witness is incompatible")
            _literal_packet(preexec_data, _preexec_payload(identity, identity_sha, preexec_data.get("barrier")), "current pre-exec witness")
            barrier = preexec_data["barrier"]
            barrier_metadata = os.stat(f"/proc/{pid}/fd/{barrier['descriptor']}")
            if (barrier_metadata.st_dev, barrier_metadata.st_ino, barrier_metadata.st_uid) != (barrier["device"], barrier["inode"], identity.uid) or not stat.S_ISSOCK(barrier_metadata.st_mode):
                raise EnforcementUnavailable("Current inherited barrier descriptor changed")
            ready_raw, _ = _private_artifact(directory, "ready-v1.json")
            expected_ready = {"version": 1, "generation": generation, "manifest_sha256": manifest_sha, "pid": pid,
                              "kernel_cap_bytes": identity.cap_bytes, "metric": _METRIC,
                              "physical_hard_cap": False, "dynamic_registry_cap": False}
            _literal_packet(_strict_json(ready_raw), expected_ready, "current application readiness")
            actual_policy = LinuxDaemonLaunchPolicy(1, identity.cap_bytes, 0)
            mount_id, mount_point, parent, _ = _required_linux_support(actual_policy, deadline)
            if mount_id != identity.slice_mount_id or identity.slice_relative_path != parent + "/" + identity.slice_unit:
                raise EnforcementUnavailable("Current charged-domain mount or parent changed")
            descriptors = []
            for relative, device, inode in ((identity.slice_relative_path, identity.slice_device, identity.slice_inode),
                                             (identity.service_relative_path, identity.service_device, identity.service_inode)):
                descriptor = _open_directory(Path(mount_point + relative), uid=identity.uid)
                owned.append(descriptor)
                descriptors.append(descriptor)
                metadata = os.fstat(descriptor)
                _filesystem_magic(descriptor)
                if (metadata.st_dev, metadata.st_ino) != (device, inode):
                    raise EnforcementUnavailable("Current controller inode changed")
            slice_fd, service_fd = descriptors
            slice_props = _properties(identity.slice_unit, ("InvocationID", "ControlGroup", "MemoryMax", "MemorySwapMax", "ActiveState"), deadline)
            service_props = _properties(identity.service_unit, ("InvocationID", "ControlGroup", "MainPID", "ExecMainPID", "ActiveState"), deadline)
            if slice_props["InvocationID"] != identity.slice_invocation_id or service_props["InvocationID"] != identity.service_invocation_id or slice_props["ControlGroup"] != identity.slice_relative_path or service_props["ControlGroup"] != identity.service_relative_path or slice_props["ActiveState"] != "active" or service_props["ActiveState"] != "active":
                raise EnforcementUnavailable("Current manager invocation changed")
            if _number(slice_props["MemoryMax"], "current manager cap", positive=True) != identity.cap_bytes or _number(slice_props["MemorySwapMax"], "current manager swap") != 0 or _number(_read_at(slice_fd, "memory.max"), "current kernel cap", positive=True) != identity.cap_bytes or _number(_read_at(slice_fd, "memory.swap.max"), "current kernel swap") != 0:
                raise EnforcementUnavailable("Current charged ceiling changed")
            if _number(service_props["MainPID"], "current MainPID", positive=True) != pid or _number(service_props["ExecMainPID"], "current ExecMainPID", positive=True) != pid:
                raise EnforcementUnavailable("Current actual workload PID changed")
            if _process_identity(pid) != (identity.start_ticks, birth_time) or _membership(pid) != identity.service_relative_path:
                raise EnforcementUnavailable("Current workload incarnation or membership changed")
            _verified_psutil_birth(pid, birth_time)
            for namespace in ("mnt", "cgroup"):
                own_namespace, workload_namespace = os.stat(f"/proc/self/ns/{namespace}"), os.stat(f"/proc/{pid}/ns/{namespace}")
                if (own_namespace.st_dev, own_namespace.st_ino) != (workload_namespace.st_dev, workload_namespace.st_ino):
                    raise EnforcementUnavailable("Current workload namespace changed")
            census = _recursive_members(slice_fd, identity.slice_relative_path)
            members = tuple(member for _, values in census for member in values)
            if members != (pid,):
                raise EnforcementUnavailable("Current charged domain has missing or foreign members")
            socket_path = expected_claim_path.parent / "model.sock"
            socket_metadata = os.stat(socket_path, follow_symlinks=False)
            if not stat.S_ISSOCK(socket_metadata.st_mode) or socket_metadata.st_uid != identity.uid or stat.S_IMODE(socket_metadata.st_mode) != 0o600:
                raise EnforcementUnavailable("Current serving endpoint is not private")
            endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            endpoint.settimeout(_left(deadline))
            endpoint.connect(str(socket_path))
            if _peer(endpoint) != (pid, identity.uid):
                raise EnforcementUnavailable("Current serving peer is not the actual workload")
            socket_after = os.stat(socket_path, follow_symlinks=False)
            if (socket_after.st_dev, socket_after.st_ino, socket_after.st_uid, socket_after.st_mode) != (socket_metadata.st_dev, socket_metadata.st_ino, socket_metadata.st_uid, socket_metadata.st_mode):
                raise EnforcementUnavailable("Serving endpoint changed during observation")
            if _private_artifact(claim_directory, expected_claim_path.name, bound=4096)[0] != claim_raw or _process_identity(pid) != (identity.start_ticks, birth_time):
                raise EnforcementUnavailable("Current claim or incarnation changed during observation")
            if _properties(identity.slice_unit, ("InvocationID", "ControlGroup", "MemoryMax", "MemorySwapMax", "ActiveState"), deadline) != slice_props or _properties(identity.service_unit, ("InvocationID", "ControlGroup", "MainPID", "ExecMainPID", "ActiveState"), deadline) != service_props:
                raise EnforcementUnavailable("Current manager identity changed during observation")
            for relative, device, inode in ((identity.slice_relative_path, identity.slice_device, identity.slice_inode),
                                             (identity.service_relative_path, identity.service_device, identity.service_inode)):
                current_fd = _open_directory(Path(mount_point + relative), uid=identity.uid)
                try:
                    current_meta = os.fstat(current_fd)
                    _filesystem_magic(current_fd)
                    if (current_meta.st_dev, current_meta.st_ino) != (device, inode):
                        raise EnforcementUnavailable("Current controller path changed during observation")
                finally:
                    os.close(current_fd)
            if tuple(member for _, values in _recursive_members(slice_fd, identity.slice_relative_path) for member in values) != (pid,):
                raise EnforcementUnavailable("Current membership changed during observation")
            result = LinuxScopeEvidence(identity, _CGROUP2_MAGIC, "cgroup2", True, True, True, identity.uid, pid,
                         identity.start_ticks, generation, manifest_sha, members, identity.cap_bytes, 0,
                         _read_at(service_fd, "memory.max"), _read_at(service_fd, "memory.swap.max"), True)
            _deadline(deadline)
            return result
        finally:
            if endpoint is not None:
                endpoint.close()
            for descriptor in reversed(owned):
                os.close(descriptor)

    def start(self, policy: LinuxDaemonLaunchPolicy, generation: str, python_executable: str, deadline: float) -> LinuxProvisionalLaunch:
        _validate_policy(policy)
        _hex(generation, 32, "generation")
        _deadline(deadline)
        if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(socket, "SO_PEERCRED"):
            raise EnforcementUnavailable("Required Linux ownership primitives are unavailable")
        if not Path(python_executable).is_absolute() or not os.access(python_executable, os.X_OK):
            raise EnforcementUnavailable("Owned interpreter must be an absolute executable")
        mount_id, mount_point, parent_relative, ancestors = _required_linux_support(policy, deadline)
        forwarded = {}
        for name, value in os.environ.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or type(value) is not str or "\x00" in value:
                raise EnforcementUnavailable("Malformed application environment forwarding")
            if name not in _INTERNAL_ENV:
                forwarded[name] = value
        forwarded[_GENERATION_ENV] = generation
        forwarded[_FLAG] = "1"
        forwarded["TRUEMEMORY_CAP_CHILD"] = "1"
        _original_environment_packet(forwarded, generation=generation, manifest_sha256="0" * 64)
        slice_unit = f"app-truememory{generation}.slice"
        service_unit = f"truememory-memory{generation}.service"
        try:
            _sweep_retired_generations(mount_point=mount_point, parent_relative=parent_relative,
                                       deadline=min(deadline, time.monotonic() + 1.5))
        except (OSError, RuntimeError, ValueError, TimeoutError):
            pass
        evidence_root = _evidence_root()
        evidence_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_fd = _open_directory(evidence_root, uid=os.getuid())
        try:
            if stat.S_IMODE(os.fstat(root_fd).st_mode) != 0o700:
                raise EnforcementUnavailable("Evidence root must already be owner-only")
            os.mkdir(generation, mode=0o700, dir_fd=root_fd)
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
        path = evidence_root / generation
        bootstrap = Path(__file__).absolute()
        bootstrap_sha = _hash_path(bootstrap, bound=1048576)
        interpreter = _executable_pin(python_executable)
        parent_ticks, parent_birth = _process_identity(os.getpid())
        manifest = {"version": 1, "generation": generation, "uid": os.getuid(), "boot_id": _boot_id(),
                    "slice_unit": slice_unit, "service_unit": service_unit, "cap_bytes": policy.cap_bytes,
                    "swap_bytes": 0, "bootstrap_sha256": bootstrap_sha, "deadline": deadline,
                    "parent_pid": os.getpid(), "parent_start_ticks": parent_ticks, "parent_birth": parent_birth,
                    "claim_path": str(Path.home() / ".truememory" / "model_server.start.json"),
                    "forwarded_names": sorted(forwarded), "interpreter": interpreter,
                    "ancestor_memory_max": ancestors}
        store = _EvidenceStore(path, manifest)
        _unresolved_launches.append(store)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _unresolved_launches.append(listener)
        listener.bind(str(path / "barrier-v1.sock"))
        os.chmod(path / "barrier-v1.sock", 0o600)
        listener.listen(1)
        store.append("prepared", {"generation": generation, "kernel_cap_bytes": policy.cap_bytes, "swap_bytes": 0,
                                  "metric": _METRIC, "physical_hard_cap": False, "dynamic_registry_cap": False})
        connection = record = None
        descriptors: list[int] = []
        try:
            _run_command(["/usr/bin/busctl", "--user", "call", "org.freedesktop.systemd1",
                          "/org/freedesktop/systemd1", "org.freedesktop.systemd1.Manager", "StartTransientUnit",
                          "ssa(sv)a(sa(sv))", slice_unit, "fail", "3", "MemoryAccounting", "b", "true",
                          "MemoryMax", "t", str(policy.cap_bytes), "MemorySwapMax", "t", "0", "0"], deadline)
            slice_properties = _properties(slice_unit, ("LoadState", "ActiveState", "InvocationID", "ControlGroup", "MemoryMax", "MemorySwapMax"), deadline)
            if slice_properties["LoadState"] != "loaded" or slice_properties["ActiveState"] != "active":
                raise EnforcementUnavailable("Owned slice did not enter its persistent active state")
            slice_relative = _relative(slice_properties["ControlGroup"], "slice path")
            if slice_relative != parent_relative + "/" + slice_unit:
                raise EnforcementUnavailable("Owned slice is not a direct child of admitted app.slice")
            slice_fd = _open_directory(Path(mount_point + slice_relative), uid=os.getuid())
            _unresolved_launches.append(slice_fd)
            descriptors.append(slice_fd)
            _filesystem_magic(slice_fd)
            if _number(_read_at(slice_fd, "memory.max"), "memory.max", positive=True) != policy.cap_bytes or _number(_read_at(slice_fd, "memory.swap.max"), "memory.swap.max") != 0:
                raise EnforcementUnavailable("Slice kernel ceiling is not installed before exec")
            baseline = _controller_events(slice_fd)
            baseline_local = _controller_events(slice_fd, "memory.events.local")
            if any(members for _, members in _recursive_members(slice_fd, slice_relative)):
                raise EnforcementUnavailable("Fresh slice already contains a process")
            store.append("preexec-slice", {"invocation_id": slice_properties["InvocationID"], "relative_path": slice_relative,
                                          "events_hierarchical": baseline, "events_local": baseline_local,
                                          "memory_current": _number(_read_at(slice_fd, "memory.current"), "memory.current")})
            stderr_fd = os.open("stderr-v1.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=store.directory)
            os.fsync(stderr_fd)
            os.close(stderr_fd)
            os.fsync(store.directory)
            # --collect unloads a failed service at exit so no stale failure
            # degrades the user manager; a successful exit remains for capture.
            arguments = ["/usr/bin/systemd-run", "--user", "--collect", "--unit=" + service_unit, "--slice=" + slice_unit,
                         "--service-type=exec", "--property=MemoryMax=infinity", "--property=MemorySwapMax=infinity",
                         "--property=RemainAfterExit=yes", "--property=StandardOutput=null",
                         "--property=StandardError=append:" + str(path / "stderr-v1.log"),
                         "--working-directory=" + os.getcwd()]
            arguments.extend("--setenv=" + name for name in sorted(forwarded))
            arguments.extend([python_executable, "-I", "-S", str(bootstrap), "--child", str(path / "manifest-v1.json")])
            runner_process, _ = _run_command(arguments, deadline, environment=forwarded)
            runner = _NativeRunner(runner_process, store, manifest, forwarded=MappingProxyType(dict(forwarded)))
            _unresolved_launches.append(runner)
            listener.settimeout(_left(deadline))
            connection, _ = listener.accept()
            _unresolved_launches.append(connection)
            peer_pid, peer_uid = _peer(connection)
            if peer_uid != os.getuid():
                raise EnforcementUnavailable("Barrier workload UID differs")
            hello = _packet_receive(connection, deadline)
            service = _properties(service_unit, ("LoadState", "ActiveState", "InvocationID", "ControlGroup", "MainPID", "ExecMainPID"), deadline)
            main_pid = _number(service["MainPID"], "actual MainPID", positive=True)
            if peer_pid != main_pid or _number(service["ExecMainPID"], "ExecMainPID", positive=True) != main_pid:
                raise EnforcementUnavailable("Manager runner is not the authenticated workload MainPID")
            ticks, birth = _process_identity(main_pid)
            service_relative = _relative(service["ControlGroup"], "service path")
            service_fd = _open_directory(Path(mount_point + service_relative), uid=os.getuid())
            _unresolved_launches.append(service_fd)
            descriptors.append(service_fd)
            _filesystem_magic(service_fd)
            slice_meta, service_meta = os.fstat(slice_fd), os.fstat(service_fd)
            identity = LinuxInvocationIdentity(1, generation, manifest["boot_id"], os.getuid(), slice_unit,
                       slice_properties["InvocationID"], mount_id, slice_meta.st_dev, slice_meta.st_ino, slice_relative,
                       service_unit, service["InvocationID"], service_meta.st_dev, service_meta.st_ino, service_relative,
                       main_pid, ticks, birth, policy.cap_bytes, 0, bootstrap_sha, store.manifest_sha256)
            listener_meta, connection_meta = os.fstat(listener.fileno()), os.fstat(connection.fileno())
            endpoint = _NativeEndpoint(listener, connection, peer_pid, peer_uid, hello,
                                       listener_meta.st_dev, listener_meta.st_ino,
                                       connection_meta.st_dev, connection_meta.st_ino)
            pidfd = os.pidfd_open(main_pid)
            _unresolved_launches.append(pidfd)
            descriptors.append(pidfd)
            pidfd_meta = os.fstat(pidfd)
            scope = _NativeScope(mount_id, mount_point, slice_fd, service_fd, baseline, baseline_local,
                                 _controller_events(service_fd), _controller_events(service_fd, "memory.events.local"),
                                 pidfd_meta.st_dev, pidfd_meta.st_ino)
            record = LinuxProvisionalLaunch(identity, runner, pidfd, scope, endpoint)
            _unresolved_launches.append(record)
            store.retain("identity-v1.json", asdict(identity))
            store.append("started-at-barrier", {"identity": asdict(identity)})
            self.inspect(record, deadline)
            return record
        except (OSError, RuntimeError, ValueError, TypeError, TimeoutError) as error:
            try:
                store.append("startup-unresolved", {"error_type": type(error).__name__, "generation": generation})
            except (OSError, RuntimeError, ValueError, TypeError):
                pass
            self._abandon_partial(store, listener, connection, record, policy, generation, mount_point,
                                  parent_relative, descriptors)
            raise

    def abandon(self, record: LinuxProvisionalLaunch, policy: LinuxDaemonLaunchPolicy) -> bool:
        """Retire a started launch that will not be returned to an owner."""
        _validate_record(record)
        scope, endpoint = record.scope_fd, record.endpoint
        return self._abandon_partial(record.runner.store, endpoint.listener, endpoint.connection, record, policy,
                                     record.identity.generation, scope.mount_point,
                                     record.identity.slice_relative_path.rsplit("/", 1)[0],
                                     [record.pidfd, scope.slice_fd, scope.service_fd])

    def _abandon_partial(self, store: _EvidenceStore, listener: socket.socket, connection: socket.socket | None,
                         record: LinuxProvisionalLaunch | None, policy: LinuxDaemonLaunchPolicy, generation: str,
                         mount_point: str, parent_relative: str, descriptors: list[int]) -> bool:
        """Terminate or release a failed launch, then retire its units.

        A child still waiting at the barrier exits on its own once the barrier
        closes; the units are stopped only after the manager and census prove
        the domain empty. Anything unproven is left for the next launch sweep.
        """
        retired = False
        try:
            if record is not None and abandon_owned_launch(OwnedLinuxLaunch(record, policy, self)):
                return True
        except (OSError, RuntimeError, ValueError, TypeError, TimeoutError):
            pass
        try:
            for endpoint in (connection, listener):
                if endpoint is not None:
                    endpoint.close()
            retired = _retire_generation_units(generation, mount_point=mount_point, parent_relative=parent_relative,
                                               deadline=time.monotonic() + 10.0, settle=5.0)
        except (OSError, RuntimeError, ValueError, TypeError, TimeoutError):
            retired = False
        try:
            store.append("startup-retired" if retired else "startup-cleanup-deferred", {"generation": generation})
        except (OSError, RuntimeError, ValueError, TypeError):
            pass
        if retired:
            _mark_generation_retired(generation)
        if record is None or not record.runner.stop_complete:
            # The record is discarded, so nothing else will use or close these.
            for descriptor in [*descriptors, store.directory]:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        return retired

    def inspect(self, record: LinuxProvisionalLaunch | LinuxInvocationIdentity, deadline: float) -> LinuxScopeEvidence:
        _deadline(deadline)
        child = type(record) is LinuxInvocationIdentity
        identity = _validate_identity(record) if child else _validate_record(record).identity
        if sys.platform != "linux" or _boot_id() != identity.boot_id or os.getuid() != identity.uid:
            raise EnforcementUnavailable("Boot or UID ownership changed")
        mount_id, mount_point = _mount()
        if mount_id != identity.slice_mount_id:
            raise EnforcementUnavailable("Cgroup mount identity changed")
        owned = []
        try:
            if child:
                if _child_context is None or _child_context[1] != identity or _child_context[3] <= 0 or _child_context[4] != identity.uid:
                    raise EnforcementUnavailable("Child has no authenticated parent barrier")
                slice_fd = _open_directory(Path(mount_point + identity.slice_relative_path), uid=identity.uid)
                owned.append(slice_fd)
                service_fd = _open_directory(Path(mount_point + identity.service_relative_path), uid=identity.uid)
                owned.append(service_fd)
            else:
                if type(record.runner) is not _NativeRunner or type(record.scope_fd) is not _NativeScope or type(record.endpoint) is not _NativeEndpoint or type(record.pidfd) is not int:
                    raise EnforcementUnavailable("Native launch resources changed")
                slice_fd, service_fd = record.scope_fd.slice_fd, record.scope_fd.service_fd
                if record.runner.stop_complete:
                    raise EnforcementUnavailable("Owned manager lifetime already stopped")
                pidfd_meta = os.fstat(record.pidfd)
                if (pidfd_meta.st_dev, pidfd_meta.st_ino) != (record.scope_fd.pidfd_device, record.scope_fd.pidfd_inode):
                    raise EnforcementUnavailable("Strong MainPID pidfd was replaced")
            for descriptor, device, inode in ((slice_fd, identity.slice_device, identity.slice_inode), (service_fd, identity.service_device, identity.service_inode)):
                metadata = os.fstat(descriptor)
                if (metadata.st_uid, metadata.st_dev, metadata.st_ino) != (identity.uid, device, inode):
                    raise EnforcementUnavailable("Owned controller directory identity changed")
                _filesystem_magic(descriptor)
            current = _open_directory(Path(mount_point + identity.slice_relative_path), uid=identity.uid)
            try:
                metadata = os.fstat(current)
                _filesystem_magic(current)
                if (metadata.st_dev, metadata.st_ino) != (identity.slice_device, identity.slice_inode):
                    raise EnforcementUnavailable("Current owned slice path was replaced")
            finally:
                os.close(current)
            slice_props = _properties(identity.slice_unit, ("InvocationID", "ControlGroup", "MemoryMax", "MemorySwapMax", "ActiveState"), deadline)
            service_props = _properties(identity.service_unit, ("InvocationID", "ControlGroup", "MainPID", "ExecMainPID", "ActiveState", "SubState"), deadline)
            if slice_props["InvocationID"] != identity.slice_invocation_id or service_props["InvocationID"] != identity.service_invocation_id:
                raise EnforcementUnavailable("Manager InvocationID changed")
            if slice_props["ControlGroup"] != identity.slice_relative_path or service_props["ControlGroup"] not in (identity.service_relative_path, "") or slice_props["ActiveState"] != "active":
                raise EnforcementUnavailable("Persistent native scope changed")
            if _number(slice_props["MemoryMax"], "manager MemoryMax", positive=True) != identity.cap_bytes or _number(slice_props["MemorySwapMax"], "manager MemorySwapMax") != 0:
                raise EnforcementUnavailable("Manager ceiling drift")
            if _number(_read_at(slice_fd, "memory.max"), "memory.max", positive=True) != identity.cap_bytes or _number(_read_at(slice_fd, "memory.swap.max"), "memory.swap.max") != 0:
                raise EnforcementUnavailable("Kernel ceiling drift")
            _ancestor_limits(mount_point, identity.slice_relative_path.rsplit("/", 1)[0], identity.cap_bytes)
            census = _recursive_members(slice_fd, identity.slice_relative_path)
            members = tuple(pid for _, values in census for pid in values)
            if len(set(members)) != len(members) or members not in ((), (identity.main_pid,)):
                raise EnforcementUnavailable("Owned slice contains an unrelated process")
            main = _number(service_props["MainPID"], "MainPID")
            if _number(service_props["ExecMainPID"], "ExecMainPID", positive=True) != identity.main_pid:
                raise EnforcementUnavailable("Retained ExecMainPID differs from actual admitted workload")
            if members:
                if service_props["ControlGroup"] != identity.service_relative_path or main != identity.main_pid or _process_identity(main) != (identity.start_ticks, identity.birth_time) or _membership(main) != identity.service_relative_path:
                    raise EnforcementUnavailable("Workload process incarnation or membership changed")
                for namespace in ("mnt", "cgroup"):
                    own_namespace = os.stat(f"/proc/self/ns/{namespace}")
                    workload_namespace = os.stat(f"/proc/{main}/ns/{namespace}")
                    if (own_namespace.st_dev, own_namespace.st_ino) != (workload_namespace.st_dev, workload_namespace.st_ino):
                        raise EnforcementUnavailable("Workload native namespace differs")
            else:
                if child or main != 0 or service_props["SubState"] not in ("exited", "failed", "dead"):
                    raise EnforcementUnavailable("Manager has not reaped the admitted workload")
                if not select.select([record.pidfd], [], [], 0)[0]:
                    raise EnforcementUnavailable("Owned MainPID pidfd has not observed exit")
                try:
                    ticks, birth = _process_identity(identity.main_pid)
                except FileNotFoundError:
                    pass
                else:
                    if (ticks, birth) != (identity.start_ticks, identity.birth_time):
                        raise EnforcementUnavailable("MainPID number now names a different process")
            try:
                current = _open_directory(Path(mount_point + identity.service_relative_path), uid=identity.uid)
            except FileNotFoundError:
                if child or members:
                    raise EnforcementUnavailable("Live owned service leaf disappeared")
                current = _open_directory(Path(mount_point + identity.slice_relative_path), uid=identity.uid)
                try:
                    metadata = os.fstat(current)
                    _filesystem_magic(current)
                    if (metadata.st_dev, metadata.st_ino) != (identity.slice_device, identity.slice_inode):
                        raise EnforcementUnavailable("Persistent slice changed during leaf pruning readback")
                finally:
                    os.close(current)
                if _number(_read_at(slice_fd, "memory.max"), "terminal memory.max", positive=True) != identity.cap_bytes or _number(_read_at(slice_fd, "memory.swap.max"), "terminal memory.swap.max") != 0:
                    raise EnforcementUnavailable("Persistent cap changed during leaf pruning readback")
                if any(values for _, values in _recursive_members(slice_fd, identity.slice_relative_path)):
                    raise EnforcementUnavailable("Persistent slice acquired a member during leaf pruning readback")
                leaf_max = leaf_swap = "unavailable_pruned"
            else:
                try:
                    metadata = os.fstat(current)
                    _filesystem_magic(current)
                    if (metadata.st_dev, metadata.st_ino) != (identity.service_device, identity.service_inode) or service_props["ControlGroup"] != identity.service_relative_path:
                        raise EnforcementUnavailable("Current owned service path was replaced")
                finally:
                    os.close(current)
                leaf_max, leaf_swap = _read_at(service_fd, "memory.max"), _read_at(service_fd, "memory.swap.max")
            if child:
                if identity.main_pid != os.getpid():
                    raise EnforcementUnavailable("Child gate identity is not its own actual PID")
                peer_uid, peer_pid = identity.uid, identity.main_pid
            else:
                endpoint = record.endpoint
                for descriptor, device, inode in ((endpoint.listener.fileno(), endpoint.listener_device, endpoint.listener_inode),
                                                  (endpoint.connection.fileno(), endpoint.connection_device, endpoint.connection_inode)):
                    metadata = os.fstat(descriptor)
                    if (metadata.st_dev, metadata.st_ino) != (device, inode):
                        raise EnforcementUnavailable("Strong barrier descriptor was replaced")
                if _peer(endpoint.connection) != (identity.main_pid, identity.uid):
                    raise EnforcementUnavailable("Owned barrier peer changed")
                hello = endpoint.hello
                if type(hello) is not dict or set(hello) != {"generation", "manifest_sha256", "pid", "start_ticks", "bootstrap_sha256"}:
                    raise EnforcementUnavailable("Malformed authenticated hello schema")
                for name, expected in (("generation", identity.generation), ("manifest_sha256", identity.manifest_sha256), ("pid", identity.main_pid), ("start_ticks", identity.start_ticks), ("bootstrap_sha256", identity.bootstrap_sha256)):
                    _literal(hello.get(name), expected, "hello " + name)
                peer_uid, peer_pid = endpoint.peer_uid, endpoint.peer_pid
            _deadline(deadline)
            return LinuxScopeEvidence(identity, _CGROUP2_MAGIC, "cgroup2", True, True, True,
                                      peer_uid, peer_pid, identity.start_ticks, identity.generation,
                                      identity.manifest_sha256, members, identity.cap_bytes, 0, leaf_max, leaf_swap, True)
        finally:
            for descriptor in owned:
                os.close(descriptor)

    def barrier(self, record: LinuxProvisionalLaunch, identity: LinuxInvocationIdentity, deadline: float) -> None:
        _validate_record(record)
        _validate_identity(identity)
        _validate_evidence(self.inspect(record, deadline), identity)
        _verified_psutil_birth(identity.main_pid, identity.birth_time)
        runner = record.runner
        claim_path = Path(runner.manifest["claim_path"])
        descriptor = os.open(claim_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        directory = _open_directory(claim_path.parent, uid=identity.uid)
        try:
            raw = _read_fd(descriptor, bound=4096)
            metadata = os.fstat(descriptor)
            claim = _strict_json(raw)
            if metadata.st_uid != identity.uid or stat.S_IMODE(metadata.st_mode) != 0o600:
                raise EnforcementUnavailable("Current startup claim is not owner-only")
            if type(claim) is not dict or set(claim) != {"generation", "launcher", "child"} or claim.get("generation") != identity.generation or claim.get("child") != {"pid": identity.main_pid, "born": identity.birth_time}:
                raise EnforcementUnavailable("Durable current startup claim differs")
            for name in ("launcher", "child"):
                item = claim[name]
                if type(item) is not dict or set(item) != {"pid", "born"} or type(item["pid"]) is not int or type(item["born"]) is not float:
                    raise EnforcementUnavailable("Durable startup claim types differ")
            if claim["launcher"] != {"pid": runner.manifest["parent_pid"], "born": runner.manifest["parent_birth"]}:
                raise EnforcementUnavailable("Durable startup launcher differs from authenticated parent")
            if _process_identity(runner.manifest["parent_pid"]) != (runner.manifest["parent_start_ticks"], runner.manifest["parent_birth"]):
                raise EnforcementUnavailable("Parent incarnation changed before release")
            _verified_psutil_birth(runner.manifest["parent_pid"], runner.manifest["parent_birth"])
            _deadline(deadline)
            os.fsync(descriptor)
            os.fsync(directory)
            current = os.stat(claim_path, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino) or _read_path(claim_path, bound=4096) != raw:
                raise EnforcementUnavailable("Current startup claim changed during durability readback")
        finally:
            os.close(descriptor)
            os.close(directory)
        runner.store.append("claim-durable-before-release", {"claim_sha256": hashlib.sha256(raw).hexdigest(), "generation": identity.generation,
                                                              "pid": identity.main_pid, "birth_time": identity.birth_time})
        _validate_evidence(self.inspect(record, deadline), identity)
        _packet_send(record.endpoint.connection, {"identity": asdict(identity), "release": True}, deadline)
        preexec = _packet_receive(record.endpoint.connection, deadline)
        _literal_packet(preexec, {"generation": identity.generation, "manifest_sha256": identity.manifest_sha256,
                        "preexec_gate_passed": True}, "initial pre-exec gate")
        runner.store.observe("pre-reexec-v1.json")
        runner.store.append("accepted-pre-reexec", {"generation": identity.generation, "pid": identity.main_pid,
                            "start_ticks": identity.start_ticks, "birth_time": identity.birth_time})
        environment_packet = _original_environment_packet(runner.forwarded, generation=identity.generation,
                              manifest_sha256=identity.manifest_sha256, names=runner.manifest["forwarded_names"])
        _packet_send(record.endpoint.connection, environment_packet, deadline)
        acknowledgement = _packet_receive(record.endpoint.connection, deadline)
        _literal_packet(acknowledgement, {"generation": identity.generation, "manifest_sha256": identity.manifest_sha256,
                        "gate_passed": True}, "restored pre-import gate")
        runner.store.observe("first-import-v1.json")
        runner.store.append("first-import", {"generation": identity.generation, "source_sha256": identity.bootstrap_sha256})
        self._await_ready(record, identity, deadline)

    def _await_ready(self, record: LinuxProvisionalLaunch, identity: LinuxInvocationIdentity, deadline: float) -> None:
        """Wait for the workload's ready receipt, then prove its domain once.

        Until it is ready the workload gates itself with short-lived systemctl
        helpers inside its own cgroup, so an exact-membership census taken while
        waiting would count them as foreign. Liveness is watched via the pidfd
        instead, and the census runs after the receipt exists and the helpers
        have been reaped.
        """
        runner = record.runner
        expected = {"version": 1, "generation": identity.generation, "manifest_sha256": identity.manifest_sha256,
                    "pid": identity.main_pid, "kernel_cap_bytes": identity.cap_bytes, "metric": _METRIC,
                    "physical_hard_cap": False, "dynamic_registry_cap": False}
        while True:
            if _owned_exited(record):
                raise EnforcementUnavailable("Admitted workload exited before application readiness")
            try:
                raw, _ = _private_artifact(runner.store.directory, "ready-v1.json")
                observed = _strict_json(raw)
            except FileNotFoundError:
                observed = None
            except (ValueError, EnforcementUnavailable):
                # The receipt is created exclusively and then written, so a
                # read can briefly see it empty or still growing.
                observed = None
            if observed is None:
                time.sleep(min(0.01, _left(deadline)))
                continue
            _literal_packet(observed, expected, "application ready receipt")
            runner.store.observe("ready-v1.json")
            runner.store.append("application-ready", expected)
            _validate_evidence(self.inspect(record, deadline), identity)
            _deadline(deadline)
            return

    def terminate(self, record: LinuxProvisionalLaunch, identity: LinuxInvocationIdentity, deadline: float) -> None:
        effective = _termination_deadline(deadline, time.monotonic())
        _validate_record(record)
        expected = _identity_snapshot(identity)
        if record.identity != expected:
            raise EnforcementUnavailable("Termination identity differs from the strong record")
        resources = _termination_resources(record)
        _same_termination_resources(record, expected, resources)
        try:
            _termination_emit(record, "termination-native-intent", {"version": 1,
                              "generation": expected.generation, "main_pid": expected.main_pid,
                              "start_ticks": expected.start_ticks, "birth_time": expected.birth_time,
                              "original_pidfd": record.pidfd, "effective_deadline": effective,
                              "kernel_cap_bytes": expected.cap_bytes, "swap_bytes": 0}, effective)
            _same_termination_resources(record, expected, resources)
            exited, _ = _termination_native_snapshot(self, record, expected, resources, effective)
            if not exited:
                for requested_signal in (signal.SIGTERM, signal.SIGKILL):
                    _termination_emit(record, "termination-signal-intent", {"generation": expected.generation,
                                      "signal": int(requested_signal)}, effective)
                    _same_termination_resources(record, expected, resources)
                    exited, _ = _termination_native_snapshot(self, record, expected, resources, effective)
                    if exited:
                        break
                    _same_termination_resources(record, expected, resources)
                    exited = _signal_owned_pidfd(record, expected, resources, requested_signal, effective)
                    _termination_emit(record, "termination-signal-result", {"generation": expected.generation,
                                      "signal": int(requested_signal), "exit_observed": exited}, effective)
                    _same_termination_resources(record, expected, resources)
                    if exited or _wait_pidfd_exit(record, expected, resources, min(effective, time.monotonic() + 1.0)):
                        break
            _wait_owned_terminal(self, record, expected, resources, effective)
            _termination_emit(record, "termination-manager-reaped", {"generation": expected.generation,
                              "main_pid": expected.main_pid, "exit_observed": True}, effective)
            _same_termination_resources(record, expected, resources)
            _deadline(effective)
        except (OSError, RuntimeError, ValueError, TypeError, TimeoutError) as error:
            try:
                _same_termination_resources(record, expected, resources)
                _termination_emit(record, "termination-native-unresolved", {"generation": expected.generation,
                                  "error_type": type(error).__name__,
                                  "errno": error.errno if isinstance(error, OSError) else None}, effective)
            except (OSError, RuntimeError, ValueError, TypeError, TimeoutError):
                pass
            raise

    def capture(self, record: LinuxProvisionalLaunch, identity: LinuxInvocationIdentity, deadline: float) -> LinuxExitReceipt:
        _validate_record(record)
        _validate_identity(identity)
        evidence = _validate_evidence(self.inspect(record, deadline), identity, terminal=True)
        if evidence.members:
            raise EnforcementUnavailable("Cannot capture terminal evidence for a live workload")
        scope, store = record.scope_fd, record.runner.store
        manager = _properties(identity.service_unit, ("InvocationID", "ExecMainPID", "ExecMainCode", "ExecMainStatus", "Result", "MainPID", "SubState"), deadline)
        if manager["InvocationID"] != identity.service_invocation_id or _number(manager["ExecMainPID"], "ExecMainPID", positive=True) != identity.main_pid or manager["MainPID"] != "0":
            raise EnforcementUnavailable("Terminal manager identity is unresolved")
        code = _number(manager["ExecMainCode"], "ExecMainCode")
        status = _number(manager["ExecMainStatus"], "ExecMainStatus")
        if code not in (1, 2, 3):
            raise EnforcementUnavailable("Manager has no explicit reaped exit or signal")
        census = _recursive_members(scope.slice_fd, identity.slice_relative_path)
        populated_lines = dict(line.split() for line in _read_at(scope.slice_fd, "cgroup.events").splitlines())
        if populated_lines.get("populated") != "0" or any(members for _, members in census):
            raise EnforcementUnavailable("Persistent aggregate domain is not actually empty")
        current = _number(_read_at(scope.slice_fd, "memory.current"), "memory.current")
        after = _controller_events(scope.slice_fd)
        slice_local_after = _controller_events(scope.slice_fd, "memory.events.local")
        leaf_after: tuple[tuple[str, int], ...] | None = None
        leaf_local_after: tuple[tuple[str, int], ...] | None = None
        if evidence.leaf_memory_max != "unavailable_pruned":
            try:
                leaf_after = _controller_events(scope.service_fd)
                leaf_local_after = _controller_events(scope.service_fd, "memory.events.local")
            except FileNotFoundError:
                evidence = _validate_evidence(self.inspect(record, deadline), identity, terminal=True)
                if evidence.members or evidence.leaf_memory_max != "unavailable_pruned":
                    raise EnforcementUnavailable("Leaf counters disappeared without proven terminal pruning")
                leaf_after = leaf_local_after = None
        try:
            peak = _number(_read_at(scope.slice_fd, "memory.peak"), "memory.peak")
        except FileNotFoundError:
            peak = None
        counter_pairs = [(scope.baseline, after), (scope.baseline_local, slice_local_after)]
        if leaf_after is not None and leaf_local_after is not None:
            counter_pairs.extend(((scope.leaf_baseline, leaf_after), (scope.leaf_baseline_local, leaf_local_after)))
        for previous, current_events in counter_pairs:
            current_values = dict(current_events)
            if any(name not in current_values or current_values[name] < value for name, value in previous):
                raise EnforcementUnavailable("Owned controller counters decreased or disappeared")
        for name in ("stderr-v1.log", "pre-reexec-v1.json", "first-import-v1.json", "ready-v1.json", "child-startup-failed-v1.json"):
            try:
                store.observe(name, durable=True, bound=16777216 if name == "stderr-v1.log" else _READ_BOUND)
            except FileNotFoundError:
                if name == "stderr-v1.log":
                    raise EnforcementUnavailable("Retained stderr evidence disappeared")
        before = dict(scope.baseline)
        deltas = tuple((name, value - before.get(name, 0)) for name, value in after)
        leaf_deltas = None if leaf_after is None else tuple((name, dict(leaf_after)[name] - value) for name, value in scope.leaf_baseline)
        leaf_local_deltas = None if leaf_local_after is None else tuple((name, dict(leaf_local_after)[name] - value) for name, value in scope.leaf_baseline_local)
        payload = {"identity": asdict(identity), "exit_code": status if code == 1 else None,
                   "signal": status if code in (2, 3) else None, "manager_result": manager["Result"],
                   "memory_current": current, "memory_peak": peak, "events_before": scope.baseline,
                   "events_after": after, "event_deltas": deltas, "populated": False,
                   "counter_scope": "cap_bearing_slice_hierarchical",
                   "slice_local_before": scope.baseline_local, "slice_local_after": slice_local_after,
                   "leaf_hierarchical_before": scope.leaf_baseline, "leaf_hierarchical_after": leaf_after,
                   "leaf_local_before": scope.leaf_baseline_local, "leaf_local_after": leaf_local_after,
                   "leaf_hierarchical_deltas": leaf_deltas, "leaf_local_deltas": leaf_local_deltas,
                   "leaf_counters_available": leaf_after is not None,
                   "leaf_counter_status": "available" if leaf_after is not None else "unavailable_pruned",
                   "descendant_members": census, "kernel_cap_bytes": identity.cap_bytes, "physical_hard_cap": False,
                   "dynamic_registry_cap": False, "metric": _METRIC}
        sequence, chain = store.append("terminal-before-stop", payload)
        store.verify()
        artifacts = tuple(sorted(store.hashes.items()))
        _deadline(deadline)
        return LinuxExitReceipt(1, identity, payload["exit_code"], payload["signal"], manager["Result"], current, peak,
                                scope.baseline, after, False, census, True, sequence, chain, artifacts,
                                _METRIC, False, identity.cap_bytes, False)

    def stop(self, record: LinuxProvisionalLaunch, identity: LinuxInvocationIdentity, deadline: float) -> None:
        _validate_record(record)
        _validate_identity(identity)
        if self.inspect(record, deadline).members:
            raise EnforcementUnavailable("Owned stop requires a reaped empty workload")
        store = record.runner.store
        store.verify()
        store.append("owned-stop-request", {"generation": identity.generation, "service_invocation_id": identity.service_invocation_id,
                                           "slice_invocation_id": identity.slice_invocation_id})
        self.inspect(record, deadline)
        _stop_unit(identity.service_unit, deadline)
        slice_props = _properties(identity.slice_unit, ("InvocationID", "ControlGroup", "MemoryMax", "MemorySwapMax", "ActiveState"), deadline)
        metadata = os.fstat(record.scope_fd.slice_fd)
        if slice_props["InvocationID"] != identity.slice_invocation_id or slice_props["ControlGroup"] != identity.slice_relative_path or (metadata.st_dev, metadata.st_ino) != (identity.slice_device, identity.slice_inode):
            raise EnforcementUnavailable("Slice replaced before owned final stop")
        if _number(slice_props["MemoryMax"], "MemoryMax", positive=True) != identity.cap_bytes or _number(slice_props["MemorySwapMax"], "MemorySwapMax") != 0:
            raise EnforcementUnavailable("Slice cap changed before owned final stop")
        if any(members for _, members in _recursive_members(record.scope_fd.slice_fd, identity.slice_relative_path)):
            raise EnforcementUnavailable("Slice acquired a member before owned final stop")
        _stop_unit(identity.slice_unit, deadline)
        store.append("owned-stop-complete", {"generation": identity.generation})
        self._close_resolved(record, identity)
        record.runner.stop_complete = True

    def _close_resolved(self, record: LinuxProvisionalLaunch, identity: LinuxInvocationIdentity) -> None:
        scope, endpoint, store = record.scope_fd, record.endpoint, record.runner.store
        expected = ((scope.slice_fd, identity.slice_device, identity.slice_inode),
                    (scope.service_fd, identity.service_device, identity.service_inode),
                    (record.pidfd, scope.pidfd_device, scope.pidfd_inode),
                    (endpoint.listener.fileno(), endpoint.listener_device, endpoint.listener_inode),
                    (endpoint.connection.fileno(), endpoint.connection_device, endpoint.connection_inode),
                    (store.directory, store.metadata.st_dev, store.metadata.st_ino))
        for descriptor, device, inode in expected:
            metadata = os.fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino) != (device, inode):
                raise EnforcementUnavailable("Resolved resource identity changed; no borrowed descriptor was closed")
        endpoint.connection.close()
        endpoint.listener.close()
        os.close(record.pidfd)
        os.close(scope.service_fd)
        os.close(scope.slice_fd)
        os.close(store.directory)


def get_linux_adapter() -> LinuxAdapter:
    return LinuxAdapter()


def _physical_total() -> int:
    pages = _integer(os.sysconf("SC_PHYS_PAGES"), "physical pages", positive=True)
    page_size = _integer(os.sysconf("SC_PAGE_SIZE"), "physical page size", positive=True)
    return _integer(pages * page_size, "physical_total", positive=True)


def _child_bundle(manifest_path: str) -> tuple[int, Path, dict[str, object], str]:
    path = Path(manifest_path)
    if not path.is_absolute() or path.name != "manifest-v1.json":
        raise EnforcementUnavailable("Child requires an absolute generation manifest")
    directory = _open_directory(path.parent, uid=os.getuid())
    try:
        if stat.S_IMODE(os.fstat(directory).st_mode) != 0o700:
            raise EnforcementUnavailable("Child generation directory is not private")
        raw, manifest_sha = _private_artifact(directory, path.name)
        manifest = _strict_json(raw)
        fields = {"version", "generation", "uid", "boot_id", "slice_unit", "service_unit", "cap_bytes", "swap_bytes",
                  "bootstrap_sha256", "deadline", "parent_pid", "parent_start_ticks", "parent_birth", "claim_path",
                  "ancestor_memory_max", "forwarded_names", "interpreter"}
        if type(manifest) is not dict or set(manifest) != fields:
            raise EnforcementUnavailable("Manifest must have the complete exact schema")
        _literal(manifest.get("version"), 1, "manifest version")
        generation = _hex(manifest.get("generation"), 32, "manifest generation")
        if path.parent.name != generation:
            raise EnforcementUnavailable("Manifest generation directory differs")
        _literal(manifest.get("uid"), os.getuid(), "manifest uid")
        _literal(manifest.get("boot_id"), _boot_id(), "manifest boot")
        _literal(manifest.get("swap_bytes"), 0, "manifest swap")
        _integer(manifest.get("cap_bytes"), "manifest cap", positive=True)
        _deadline(manifest.get("deadline"))
        _integer(manifest.get("parent_pid"), "parent pid", positive=True)
        _integer(manifest.get("parent_start_ticks"), "parent start ticks", positive=True)
        if type(manifest["parent_birth"]) is not float or not math.isfinite(manifest["parent_birth"]) or manifest["parent_birth"] <= 0:
            raise EnforcementUnavailable("Manifest parent birth is invalid")
        if _process_identity(manifest["parent_pid"]) != (manifest["parent_start_ticks"], manifest["parent_birth"]):
            raise EnforcementUnavailable("Parent process incarnation changed")
        names = manifest["forwarded_names"]
        if type(names) is not list or not names or len(names) > 4096 or any(type(name) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None for name in names) or names != sorted(set(names)):
            raise EnforcementUnavailable("Manifest forwarding names are malformed")
        interpreter = manifest["interpreter"]
        if type(interpreter) is not dict or set(interpreter) != {"path", "resolved_path", "device", "inode", "sha256"}:
            raise EnforcementUnavailable("Manifest interpreter pin is malformed")
        if _executable_pin(interpreter["path"]) != interpreter or os.path.abspath(sys.executable) != interpreter["path"]:
            raise EnforcementUnavailable("Selected interpreter identity changed")
        bootstrap_sha = _hash_path(Path(__file__).absolute(), bound=1048576)
        _literal(manifest.get("bootstrap_sha256"), bootstrap_sha, "manifest bootstrap")
        _load_capsule()
        return directory, path, manifest, manifest_sha
    except (OSError, RuntimeError, ValueError, TypeError, TimeoutError):
        os.close(directory)
        raise


def _child_main(manifest_path: str) -> None:
    global _child_context
    if sys.platform != "linux" or not sys.flags.no_site or not sys.flags.isolated:
        raise EnforcementUnavailable("Initial child must start on Linux with isolated site disabled")
    directory, path, manifest, manifest_sha = _child_bundle(manifest_path)
    endpoint = None
    try:
        deadline = manifest["deadline"]
        generation, bootstrap_sha = manifest["generation"], manifest["bootstrap_sha256"]
        ticks, _ = _process_identity(os.getpid())
        endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        endpoint.settimeout(_left(deadline))
        endpoint.connect(str(path.parent / "barrier-v1.sock"))
        parent_pid, parent_uid = _peer(endpoint)
        if parent_pid != manifest["parent_pid"] or parent_uid != os.getuid() or _process_identity(parent_pid)[0] != manifest["parent_start_ticks"]:
            raise EnforcementUnavailable("Child barrier is not its authenticated parent incarnation")
        _packet_send(endpoint, {"generation": generation, "manifest_sha256": manifest_sha, "pid": os.getpid(),
                                "start_ticks": ticks, "bootstrap_sha256": bootstrap_sha}, deadline)
        release = _packet_receive(endpoint, deadline)
        if set(release) != {"identity", "release"} or release["release"] is not True or type(release["identity"]) is not dict:
            raise EnforcementUnavailable("Parent did not send a valid release identity")
        identity = LinuxInvocationIdentity(**release["identity"])
        if identity.generation != generation or identity.manifest_sha256 != manifest_sha or identity.bootstrap_sha256 != bootstrap_sha:
            raise EnforcementUnavailable("Child release identity differs from its manifest")
        identity_raw, identity_sha = _private_artifact(directory, "identity-v1.json")
        if _strict_json(identity_raw) != asdict(identity):
            raise EnforcementUnavailable("Retained release identity differs from child gate")
        _child_context = (path.parent, identity, endpoint, parent_pid, parent_uid)
        child_gate(identity, deadline=deadline)
        barrier_metadata = os.fstat(endpoint.fileno())
        barrier = {"descriptor": endpoint.fileno(), "device": barrier_metadata.st_dev, "inode": barrier_metadata.st_ino}
        _exclusive_file(directory, "pre-reexec-v1.json", _canonical(_preexec_payload(identity, identity_sha, barrier)))
        _packet_send(endpoint, {"generation": generation, "manifest_sha256": manifest_sha,
                                "preexec_gate_passed": True}, deadline)
        retained = _packet_receive(endpoint, deadline)
        if set(retained) != {"generation", "manifest_sha256", "preexec_witness_retained", "forwarded_environment"}:
            raise EnforcementUnavailable("Parent did not retain the accepted pre-exec witness")
        _literal(retained["generation"], generation, "retained environment generation")
        _literal(retained["manifest_sha256"], manifest_sha, "retained environment manifest")
        _literal(retained["preexec_witness_retained"], True, "retained pre-exec witness")
        clean_environment = _original_environment(retained["forwarded_environment"], generation=generation,
                                                  names=manifest["forwarded_names"])
        clean_environment[_MANIFEST_ENV] = str(path)
        clean_environment[_IDENTITY_ENV] = identity_sha
        clean_environment[_BARRIER_ENV] = str(endpoint.fileno())
        child_gate(identity, deadline=deadline)
        descriptors = []
        with os.scandir("/proc/self/fd") as entries:
            for entry in entries:
                if len(descriptors) >= 128:
                    raise EnforcementUnavailable("Child descriptor inventory exceeds its bound")
                descriptors.append(int(entry.name))
        for descriptor in descriptors:
            if descriptor > 2 and descriptor != endpoint.fileno():
                try:
                    os.set_inheritable(descriptor, False)
                except OSError as error:
                    if error.errno != 9:
                        raise
        os.set_inheritable(endpoint.fileno(), True)
        # exec preserves the admitted PID/cgroup while restoring ordinary flags,
        # site startup and all originally forwarded PYTHON* interpreter settings.
        os.execve(manifest["interpreter"]["path"],
                  [manifest["interpreter"]["path"], "-m", "truememory.model_server"], clean_environment)
    except (OSError, RuntimeError, ValueError, TypeError, TimeoutError) as error:
        _exclusive_file(directory, "child-startup-failed-v1.json", _canonical({"version": 1,
                        "generation": manifest["generation"], "pid": os.getpid(), "error_type": type(error).__name__}))
        raise
    finally:
        if endpoint is not None:
            endpoint.close()
        os.close(directory)


def _restore_child_context() -> None:
    global _child_context
    if sys.platform != "linux" or not required_cap_selected(os.environ):
        raise EnforcementUnavailable("Required daemon cannot restore an unsupported gate")
    manifest_path = _text(os.environ.get(_MANIFEST_ENV), "inherited manifest", bound=4096)
    expected_identity_sha = _hex(os.environ.get(_IDENTITY_ENV), 64, "inherited identity hash")
    raw_descriptor = os.environ.get(_BARRIER_ENV)
    if type(raw_descriptor) is not str or re.fullmatch(r"[0-9]+", raw_descriptor) is None or len(raw_descriptor) > 10:
        raise EnforcementUnavailable("Required daemon has no bounded inherited barrier descriptor")
    descriptor = _integer(int(raw_descriptor), "inherited barrier descriptor")
    if descriptor < 3:
        raise EnforcementUnavailable("Barrier cannot replace a standard stream")
    directory, path, manifest, manifest_sha = _child_bundle(manifest_path)
    endpoint = None
    try:
        identity_raw, identity_sha = _private_artifact(directory, "identity-v1.json")
        _literal(identity_sha, expected_identity_sha, "inherited identity hash")
        identity_data = _strict_json(identity_raw)
        if type(identity_data) is not dict:
            raise EnforcementUnavailable("Retained identity is malformed")
        identity = LinuxInvocationIdentity(**identity_data)
        if identity.manifest_sha256 != manifest_sha or identity.main_pid != os.getpid() or identity.generation != manifest["generation"] or _process_identity(os.getpid()) != (identity.start_ticks, identity.birth_time):
            raise EnforcementUnavailable("Reexecuted process is not the admitted incarnation")
        preexec_raw, _ = _private_artifact(directory, "pre-reexec-v1.json")
        preexec_data = _strict_json(preexec_raw)
        if type(preexec_data) is not dict:
            raise EnforcementUnavailable("Durable pre-exec witness differs from the admitted identity")
        _literal_packet(preexec_data, _preexec_payload(identity, identity_sha, preexec_data.get("barrier")), "durable pre-exec witness")
        barrier = preexec_data["barrier"]
        _literal(descriptor, barrier["descriptor"], "inherited barrier number")
        barrier_metadata = os.fstat(descriptor)
        if (barrier_metadata.st_dev, barrier_metadata.st_ino, barrier_metadata.st_uid) != (barrier["device"], barrier["inode"], identity.uid) or not stat.S_ISSOCK(barrier_metadata.st_mode):
            raise EnforcementUnavailable("Inherited barrier descriptor was reused; it was left untouched")
        os.set_inheritable(descriptor, False)
        endpoint = socket.socket(fileno=descriptor)
        if endpoint.family != socket.AF_UNIX or endpoint.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM:
            raise EnforcementUnavailable("Inherited barrier is not the original local stream")
        parent_pid, parent_uid = _peer(endpoint)
        if (parent_pid, parent_uid) != (manifest["parent_pid"], identity.uid) or _process_identity(parent_pid) != (manifest["parent_start_ticks"], manifest["parent_birth"]):
            raise EnforcementUnavailable("Inherited barrier no longer names the original parent")
        claim_path = Path(manifest["claim_path"])
        claim_directory = _open_directory(claim_path.parent, uid=identity.uid)
        try:
            claim_raw, claim_sha = _private_artifact(claim_directory, claim_path.name, bound=4096)
            claim = _validate_claim(_strict_json(claim_raw), pid=identity.main_pid, birth_time=identity.birth_time, generation=identity.generation)
            if claim["launcher"] != {"pid": parent_pid, "born": manifest["parent_birth"]}:
                raise EnforcementUnavailable("Inherited gate launcher differs from the current durable claim")
            _, _, receipts = _receipt_chain(directory)
            witnesses = [item["payload"] for item in receipts if item["phase"] == "claim-durable-before-release"]
            if len(witnesses) != 1 or type(witnesses[0]) is not dict or witnesses[0].get("claim_sha256") != claim_sha:
                raise EnforcementUnavailable("Inherited gate has no current durable claim witness")
        finally:
            os.close(claim_directory)
        _child_context = (path.parent, identity, endpoint, parent_pid, parent_uid)
        child_gate(identity, deadline=manifest["deadline"])
        _exclusive_file(directory, "first-import-v1.json", _canonical(preexec_data))
        _packet_send(endpoint, {"generation": identity.generation, "manifest_sha256": manifest_sha,
                                "gate_passed": True}, manifest["deadline"])
    except (OSError, RuntimeError, ValueError, TypeError, TimeoutError):
        if endpoint is not None:
            endpoint.close()
        _child_context = None
        raise
    finally:
        os.close(directory)


def require_child_scope() -> None:
    if not required_cap_selected(os.environ):
        raise EnforcementUnavailable("Required daemon was not admitted through the pre-exec child gate")
    if _child_context is None:
        _restore_child_context()
    path, identity, _, _, _ = _child_context
    directory = _open_directory(path, uid=identity.uid)
    try:
        raw, digest = _private_artifact(directory, "manifest-v1.json")
        _literal(digest, identity.manifest_sha256, "required current manifest")
        manifest = _strict_json(raw)
        if type(manifest) is not dict:
            raise EnforcementUnavailable("Required current manifest is malformed")
        child_gate(identity, deadline=manifest["deadline"])
    finally:
        os.close(directory)


def record_server_ready() -> None:
    require_child_scope()
    path, identity, _, _, _ = _child_context
    directory = _open_directory(path, uid=identity.uid)
    try:
        _exclusive_file(directory, "ready-v1.json", _canonical({"version": 1, "generation": identity.generation,
                        "manifest_sha256": identity.manifest_sha256, "pid": identity.main_pid,
                        "kernel_cap_bytes": identity.cap_bytes, "metric": _METRIC,
                        "physical_hard_cap": False, "dynamic_registry_cap": False}))
    finally:
        os.close(directory)


def _write_console_claim(path: Path, claim: dict[str, object]) -> None:
    raw = _canonical(claim)
    directory = _open_directory(path.parent, uid=os.getuid())
    name = path.name + "." + secrets.token_hex(8) + ".tmp"
    try:
        _exclusive_file(directory, name, raw)
        os.rename(name, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        os.close(directory)


def _retire_console_claim(directory: int, name: str, generation: str) -> None:
    """Remove this generation's claim so a failed launch blocks no later launcher."""
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
    except FileNotFoundError:
        return
    try:
        claim = _strict_json(_read_fd(descriptor, bound=4096))
    finally:
        os.close(descriptor)
    if type(claim) is dict and claim.get("generation") == generation:
        os.unlink(name, dir_fd=directory)
        os.fsync(directory)


def _verified_psutil_birth(pid: int, candidate: float) -> float:
    # This parent-only verification does not run in the pre-site child gate.
    psutil = importlib.import_module("psutil")
    process = psutil.Process(pid)
    observed = process.create_time()
    if process.status() == psutil.STATUS_ZOMBIE or type(observed) is not float or observed != candidate:
        raise EnforcementUnavailable("Actual psutil process birth differs from the native incarnation")
    return observed


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        _child_main(sys.argv[2])
        return
    if len(sys.argv) != 1:
        raise SystemExit("The capped launcher accepts no application arguments")
    environment = dict(os.environ)
    if _FLAG in environment:
        required_cap_selected(environment)
    else:
        environment[_FLAG] = "1"
    if sys.platform != "linux":
        raise EnforcementUnavailable("A required Linux charged-domain cap is unsupported on this platform")
    policy = plan_required_linux_launch(environment, physical_total=None if "TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB" in environment else _physical_total())
    deadline = time.monotonic() + 30.0
    _required_linux_support(policy, deadline)
    base = Path.home() / ".truememory"
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = _open_directory(base, uid=os.getuid())
    if stat.S_IMODE(os.fstat(directory).st_mode) != 0o700:
        os.close(directory)
        raise EnforcementUnavailable("Existing startup directory must be private")
    fcntl = importlib.import_module("fcntl")
    lock = os.open("model_server.start.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claim_path = base / "model_server.start.json"
        try:
            old = _strict_json(_read_path(claim_path, bound=4096))
        except FileNotFoundError:
            old = None
        if old is not None:
            _validate_claim(old, pending=True)
            previous = old.get("child") or old.get("launcher")
            try:
                _, birth = _process_identity(previous["pid"])
            except FileNotFoundError:
                pass
            else:
                if birth == previous.get("born"):
                    raise EnforcementUnavailable("Existing launch ownership is still live or unresolved")
        bind = os.open("model_server.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
        try:
            fcntl.flock(bind, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(bind)
        generation = secrets.token_hex(16)
        _, candidate = _process_identity(os.getpid())
        parent_born = _verified_psutil_birth(os.getpid(), candidate)
        claim = {"generation": generation, "launcher": {"pid": os.getpid(), "born": parent_born}, "child": None}
        _write_console_claim(claim_path, claim)
        handle = None
        unresolved = None
        try:
            handle = launch_owned_linux(policy, generation=generation, python_executable=os.path.abspath(sys.executable), deadline=deadline)
            child_born = _verified_psutil_birth(handle.pid, handle.record.identity.birth_time)
            claim["child"] = {"pid": handle.pid, "born": child_born}
            _write_console_claim(claim_path, claim)
            handle.release_after_claim(claim, deadline=deadline)
        except (OSError, RuntimeError, ValueError, TypeError, TimeoutError) as error:
            # A failed generation must neither block later launches nor run unowned.
            _retire_console_claim(directory, claim_path.name, generation)
            if handle is None or abandon_owned_launch(handle):
                raise
            unresolved = error
    finally:
        os.close(lock)
        os.close(directory)
    # The foreground console retains ownership through exit, including a
    # failed launch whose workload could not be terminated.
    while not select.select([handle.record.pidfd], [], [], 1.0)[0]:
        pass
    released = release_exited_launch(handle)
    if unresolved is not None:
        raise unresolved
    if not released:
        raise EnforcementUnavailable("Exited workload units were left for the next capped launch sweep")


_NATIVE_STARTUP_U64 = (1 << 64) - 1
_NATIVE_STARTUP_U32 = (1 << 32) - 1
_NATIVE_STARTUP_LIBRARY = "/usr/lib/x86_64-linux-gnu/libsystemd.so.0.32.0"
_NATIVE_STARTUP_LIBRARY_SHA = "9d70722be19814a7710be5f8df42fb1a36566a4038ae789d63e948c2b4b65ca1"
_NATIVE_STARTUP_LIBRARY_STAT = (64513, 23495170, 33188, 0, 0, 807936, 1749652895000000000, 1771515970026234838, 1)
_NATIVE_STARTUP_MATCH = "type='signal',sender='org.freedesktop.DBus',path='/org/freedesktop/DBus',interface='org.freedesktop.DBus',member='NameOwnerChanged',arg0='org.freedesktop.systemd1'"
_NATIVE_STARTUP_VERSION = r"249(?:\.[0-9]+)*(?:[-+~][A-Za-z0-9][A-Za-z0-9.+~_-]*)?"
_NATIVE_STARTUP_CODES = frozenset((
    "invalid_input", "invalid_owner", "capacity_exhausted", "sequence_exhausted", "stale_sequence",
    "busy", "forked_owner", "wrong_thread", "deadline_expired", "backend_unavailable",
    "backend_pin_mismatch", "backend_abi_unproven", "endpoint_unavailable", "endpoint_pin_mismatch",
    "peer_auth_failed", "native_call_failed", "transport_disconnected", "manager_owner_changed",
    "manager_auth_failed", "subscription_refused", "malformed_evidence", "evidence_overflow",
    "callback_failed", "native_resource_drift", "cleanup_unresolved", "already_closed",
    "already_released", "loader_override_present", "clock_regressed", "cleanup_deadline_invalid",
    "cleanup_counter_exhausted", "cleanup_attempts_exhausted", "native_string_unterminated",
    "native_pointer_unproven", "native_release_unobserved",
))
_NATIVE_STARTUP_PHASES = ("prepared", "authenticating", "subscribed", "terminal_unknown", "closing_precreate", "closed_precreate", "released")
_NATIVE_STARTUP_RECORDS: dict[int, object] = {}
_NATIVE_STARTUP_LOCK = threading.RLock()
_NATIVE_STARTUP_CREATOR_PID = os.getpid()
_NATIVE_STARTUP_NEXT_OWNER_ID = 1
_NATIVE_STARTUP_OPERATION: object | None = None


def _native_startup_fail(code: str) -> None:
    if type(code) is not str or code not in _NATIVE_STARTUP_CODES:
        code = "invalid_input"
    raise EnforcementUnavailable("native_startup:" + code)


def _native_startup_fields(value: object, cls: type) -> None:
    if type(value) is not cls:
        _native_startup_fail("invalid_input")
    try:
        for name in cls.__dataclass_fields__:
            object.__getattribute__(value, name)
    except (AttributeError, TypeError):
        _native_startup_fail("invalid_input")


def _native_startup_integer(value: object, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0) or value > _NATIVE_STARTUP_U64:
        _native_startup_fail("invalid_input")
    return value


def _native_startup_float(value: object) -> float:
    if type(value) is not float or not math.isfinite(value):
        _native_startup_fail("invalid_input")
    return value


def _native_startup_text(value: object, *, bound: int, empty: bool = False, ascii_only: bool = False) -> str:
    if type(value) is not str or "\x00" in value or (not value and not empty):
        _native_startup_fail("invalid_input")
    try:
        encoded = value.encode("ascii" if ascii_only else "utf-8", "strict")
    except UnicodeError:
        _native_startup_fail("invalid_input")
    if len(encoded) > bound:
        _native_startup_fail("invalid_input")
    return value


def _native_startup_sequence_pair(sequence: int, *, cleanup: bool = False) -> tuple[int, int]:
    _native_startup_integer(sequence)
    if type(cleanup) is not bool:
        _native_startup_fail("invalid_input")
    if sequence > _NATIVE_STARTUP_U64 - 2:
        _native_startup_fail("cleanup_counter_exhausted" if cleanup else "sequence_exhausted")
    return sequence + 1, sequence + 2


@dataclass(frozen=True, slots=True)
class StartupNativeBackendPin:
    version: int
    canonical_path: str
    sha256: str
    stat_pin: tuple[int, int, int, int, int, int, int, int, int]

    def __post_init__(self) -> None:
        _native_startup_validate_pin(self)


@dataclass(frozen=True, slots=True)
class StartupNativeRequest:
    version: int
    policy: LinuxDaemonLaunchPolicy
    generation: str
    python_executable: str
    backend_pin: StartupNativeBackendPin

    def __post_init__(self) -> None:
        _native_startup_validate_request(self)


@dataclass(frozen=True, slots=True)
class NativeStartupOwner:
    version: int
    owner_id: int
    creator_pid: int

    def __post_init__(self) -> None:
        _native_startup_validate_owner(self)


@dataclass(frozen=True, slots=True)
class NativeStartupSnapshot:
    version: int
    owner: NativeStartupOwner
    request: StartupNativeRequest
    original_deadline: float
    last_observed_now: float
    operation_sequence: int
    cleanup_sequence: int
    cleanup_deadline: float | None
    cleanup_attempted: bool
    cleanup_outcome: str | None
    phase: str
    context: StartupBusContext | None
    peer_credentials: tuple[int, int, int] | None
    manager_version: str | None
    authentication_attempted: bool
    owner_match_confirmed: bool
    subscription_confirmed: bool
    slice_attempted: bool
    service_attempted: bool
    connection_closed: bool
    native_objects_released: bool
    reason: str | None

    def __post_init__(self) -> None:
        _native_startup_validate_snapshot(self)


def _native_startup_validate_pin(pin: object) -> StartupNativeBackendPin:
    _native_startup_fields(pin, StartupNativeBackendPin)
    if type(pin.version) is not int or pin.version != 1:
        _native_startup_fail("invalid_input")
    path = _native_startup_text(pin.canonical_path, bound=4096)
    if not path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/")[1:]):
        _native_startup_fail("invalid_input")
    if type(pin.sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", pin.sha256) is None:
        _native_startup_fail("invalid_input")
    if type(pin.stat_pin) is not tuple or len(pin.stat_pin) != 9:
        _native_startup_fail("invalid_input")
    for item in pin.stat_pin:
        _native_startup_integer(item)
    dev, ino, mode, uid, _, size, _, _, links = pin.stat_pin
    if not dev or not ino or not size or size > 4194304 or not links or not stat.S_ISREG(mode) or uid != 0 or mode & 0o022:
        _native_startup_fail("invalid_input")
    return pin


def _native_startup_validate_request(request: object) -> StartupNativeRequest:
    _native_startup_fields(request, StartupNativeRequest)
    if type(request.version) is not int or request.version != 1:
        _native_startup_fail("invalid_input")
    try:
        _validate_policy(request.policy)
    except (TypeError, ValueError, AttributeError, EnforcementUnavailable):
        _native_startup_fail("invalid_input")
    if type(request.generation) is not str or re.fullmatch(r"[0-9a-f]{32}", request.generation) is None:
        _native_startup_fail("invalid_input")
    path = _native_startup_text(request.python_executable, bound=4096)
    if not path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/")[1:]):
        _native_startup_fail("invalid_input")
    _native_startup_validate_pin(request.backend_pin)
    return request


def _native_startup_validate_owner(owner: object) -> NativeStartupOwner:
    _native_startup_fields(owner, NativeStartupOwner)
    if type(owner.version) is not int or owner.version != 1:
        _native_startup_fail("invalid_input")
    _native_startup_integer(owner.owner_id, positive=True)
    _native_startup_integer(owner.creator_pid, positive=True)
    return owner


def _native_startup_validate_snapshot(state: object) -> NativeStartupSnapshot:
    _native_startup_fields(state, NativeStartupSnapshot)
    if type(state.version) is not int or state.version != 1:
        _native_startup_fail("invalid_input")
    _native_startup_validate_owner(state.owner)
    _native_startup_validate_request(state.request)
    _native_startup_float(state.original_deadline)
    _native_startup_float(state.last_observed_now)
    _native_startup_integer(state.operation_sequence)
    _native_startup_integer(state.cleanup_sequence)
    if state.cleanup_deadline is not None:
        _native_startup_float(state.cleanup_deadline)
    for name in ("cleanup_attempted", "authentication_attempted", "owner_match_confirmed", "subscription_confirmed", "connection_closed", "native_objects_released"):
        if type(getattr(state, name)) is not bool:
            _native_startup_fail("invalid_input")
    for name in ("slice_attempted", "service_attempted"):
        if type(getattr(state, name)) is not bool or getattr(state, name):
            _native_startup_fail("invalid_input")
    if type(state.phase) is not str or state.phase not in _NATIVE_STARTUP_PHASES:
        _native_startup_fail("invalid_input")
    if state.reason is not None and (type(state.reason) is not str or state.reason not in _NATIVE_STARTUP_CODES):
        _native_startup_fail("invalid_input")
    if state.cleanup_outcome is not None and (type(state.cleanup_outcome) is not str or state.cleanup_outcome not in ("in_progress", "closed", "partial", "unknown")):
        _native_startup_fail("invalid_input")
    if state.peer_credentials is not None:
        if type(state.peer_credentials) is not tuple or len(state.peer_credentials) != 3:
            _native_startup_fail("invalid_input")
        for i, value in enumerate(state.peer_credentials):
            _native_startup_integer(value, positive=i == 0)
    if state.manager_version is not None:
        _native_startup_text(state.manager_version, bound=256, ascii_only=True)
        if re.fullmatch(_NATIVE_STARTUP_VERSION, state.manager_version, flags=re.ASCII) is None:
            _native_startup_fail("invalid_input")
    if state.context is not None:
        try:
            _startup_validate_context(state.context)
        except (TypeError, ValueError, AttributeError, EnforcementUnavailable):
            _native_startup_fail("invalid_input")
        if state.manager_version is None or state.peer_credentials is None:
            _native_startup_fail("invalid_input")
    if state.subscription_confirmed and (not state.owner_match_confirmed or state.context is None):
        _native_startup_fail("invalid_input")
    if state.owner_match_confirmed and not state.authentication_attempted:
        _native_startup_fail("invalid_input")
    if state.phase == "prepared" and (state.operation_sequence != 0 or state.authentication_attempted or state.context is not None or state.peer_credentials is not None or not state.connection_closed or not state.native_objects_released):
        _native_startup_fail("invalid_input")
    if state.phase == "prepared" and (state.cleanup_sequence != 0 or state.cleanup_attempted or state.reason is not None):
        _native_startup_fail("invalid_input")
    if state.phase in ("prepared", "authenticating", "subscribed") and state.last_observed_now >= state.original_deadline:
        _native_startup_fail("invalid_input")
    if state.phase == "authenticating" and (not state.authentication_attempted or state.operation_sequence != 1 or state.cleanup_attempted):
        _native_startup_fail("invalid_input")
    if state.phase == "subscribed" and (not state.subscription_confirmed or state.connection_closed or state.native_objects_released):
        _native_startup_fail("invalid_input")
    if state.phase in ("closed_precreate", "released") and (not state.connection_closed or not state.native_objects_released):
        _native_startup_fail("invalid_input")
    if state.phase == "subscribed" and (state.operation_sequence != 2 or state.cleanup_attempted or state.reason is not None):
        _native_startup_fail("invalid_input")
    if state.phase == "terminal_unknown" and state.reason is None:
        _native_startup_fail("invalid_input")
    if not state.authentication_attempted and (state.operation_sequence != 0 or state.owner_match_confirmed or state.subscription_confirmed or state.context is not None or state.peer_credentials is not None or state.manager_version is not None):
        _native_startup_fail("invalid_input")
    if not state.cleanup_attempted and state.cleanup_sequence != 0:
        _native_startup_fail("invalid_input")
    if state.phase == "closing_precreate" and (not state.cleanup_attempted or state.cleanup_outcome != "in_progress"):
        _native_startup_fail("invalid_input")
    if state.phase in ("closed_precreate", "released") and (not state.cleanup_attempted or state.cleanup_outcome != "closed"):
        _native_startup_fail("invalid_input")
    if state.cleanup_attempted != (state.cleanup_deadline is not None) or state.cleanup_attempted != (state.cleanup_outcome is not None):
        _native_startup_fail("invalid_input")
    return state


def _native_startup_copy_request(request: StartupNativeRequest) -> StartupNativeRequest:
    _native_startup_validate_request(request)
    pin = request.backend_pin
    return StartupNativeRequest(1, _policy_snapshot(request.policy), request.generation, request.python_executable,
                                StartupNativeBackendPin(1, pin.canonical_path, pin.sha256, tuple(pin.stat_pin)))


def _native_startup_request_key(request: StartupNativeRequest) -> tuple:
    _native_startup_validate_request(request)
    policy = request.policy
    return (request.version, policy.version, policy.cap_bytes, policy.swap_bytes, policy.metric,
            policy.physical_hard_cap, policy.dynamic_registry_cap, request.generation,
            request.python_executable, request.backend_pin.canonical_path, request.backend_pin.sha256,
            request.backend_pin.stat_pin)


def _native_startup_state_key(state: NativeStartupSnapshot) -> tuple:
    _native_startup_validate_snapshot(state)
    context = state.context
    context_key = None if context is None else (context.version, context.bus_id, context.manager_unique_name, context.client_unique_name, context.manager_uid)
    return tuple(getattr(state, name) if name not in ("request", "context") else
                 (_native_startup_request_key(state.request) if name == "request" else context_key)
                 for name in NativeStartupSnapshot.__dataclass_fields__)


class _NativeStartupId128(ctypes.Union):
    _fields_ = (("bytes", ctypes.c_uint8 * 16), ("qwords", ctypes.c_uint64 * 2))


class _NativeStartupReference:
    def __init__(self, role: str, out: object) -> None:
        self.role = role
        self.original_role = role
        self.out = out
        self.original_out = out
        self.pointer: int | None = None
        self.state = "acquisition_pending"
        self.release_frame: object | None = None


class _NativeStartupResources:
    def __init__(self) -> None:
        self.backend: object | None = None
        self.library: object | None = None
        self.library_fd: int | None = None
        self.directory_fds: list[tuple[int, tuple, str]] = []
        self.socket: object | None = None
        self.raw_fd: int | None = None
        self.raw_pin: tuple | None = None
        self.transfer = "none"
        self.fd_release_observed = False
        self.bus: _NativeStartupReference | None = None
        self.references: list[_NativeStartupReference] = []
        self.filter_binding: object | None = None
        self.pending: object | None = None
        self.callbacks: list[object] = []
        self.frames: list[object] = []
        self.active_calls = 0
        self.active_callbacks = 0
        self.callback_error: str | None = None
        self.events: list[dict] = []
        self.evidence_bytes = 0
        self.callback_count = 0
        self.copy_bytes = 0
        self.iterations = 0
        self.client_name: str | None = None
        self.bus_id: str | None = None
        self.manager_name: str | None = None
        self.manager_uid: int | None = None
        self.manager_version: str | None = None
        self.peer: tuple[int, int, int] | None = None
        self.bus_closed = False
        self.last_frame: object | None = None
        self.borrowed_message: int | None = None
        self.seen_serials: set[tuple[str, int]] = set()


class _NativeStartupSeal:
    def __init__(self, owner: NativeStartupOwner, request: StartupNativeRequest, deadline: float, resources: _NativeStartupResources) -> None:
        self.owner = owner
        self.owner_values = (owner.version, owner.owner_id, owner.creator_pid)
        self.request = _native_startup_copy_request(request)
        self.request_key = _native_startup_request_key(self.request)
        self.original_deadline = deadline
        self.thread = threading.current_thread()
        self.thread_ident = threading.get_ident()
        self.native_id = threading.get_native_id()
        self.uid = os.getuid()
        self.euid = os.geteuid()
        self.resources = resources


class _NativeStartupRecord:
    def __init__(self, seal: _NativeStartupSeal, state: NativeStartupSnapshot, resources: _NativeStartupResources) -> None:
        self.seal = seal
        self.state = state
        self.state_key = _native_startup_state_key(state)
        self.resources = resources
        self.operation: object | None = None
        self.cleanup_operation: object | None = None
        self.cleanup_attempts: list[object] = []


class _NativeStartupAuthToken:
    def __init__(self, record: _NativeStartupRecord, intent: int, terminal: int, deadline: float) -> None:
        self.record = record
        self.resources = record.resources
        self.seal = record.seal
        self.intent = intent
        self.terminal = terminal
        self.deadline = deadline
        self.cleanup = False
        self.failed = False
        self.anchor = (intent, terminal, deadline, False)
        self.outcome: tuple | None = None


class _NativeStartupCleanupToken(_NativeStartupAuthToken):
    def __init__(self, record: _NativeStartupRecord, intent: int, terminal: int, deadline: float) -> None:
        super().__init__(record, intent, terminal, deadline)
        self.cleanup = True
        self.anchor = (intent, terminal, deadline, True)


class _NativeStartupCallFrame:
    def __init__(self, name: str, function: object, arguments: tuple, outputs: tuple) -> None:
        self.name = name
        self.function = function
        self.arguments = arguments
        self.outputs = outputs
        self.entered = False
        self.returned = False
        self.result: object | None = None


class _NativeStartupCallbackBinding:
    def __init__(self, record: _NativeStartupRecord, token: _NativeStartupAuthToken, kind: str) -> None:
        self.record = record
        self.token = token
        self.kind = kind
        self.cell = ctypes.c_uint64(record.seal.owner.owner_id)
        self.userdata = ctypes.c_void_p(ctypes.addressof(self.cell))
        self.callback: object | None = None
        self.slot: _NativeStartupReference | None = None
        self.anchor = (record, token, record.resources, record.seal, self.userdata.value, kind)


class _NativeStartupPendingReply:
    def __init__(self, destination: str, result_kind: str, binding: _NativeStartupCallbackBinding) -> None:
        self.destination = destination
        self.result_kind = result_kind
        self.binding = binding
        self.cookie: int | None = None
        self.reply: dict | None = None
        self.message: _NativeStartupReference | None = None
        self.slot: _NativeStartupReference | None = None
        self.inputs: list[object] = []


def _native_startup_creator() -> int:
    pid = os.getpid()
    if type(pid) is not int or pid != _NATIVE_STARTUP_CREATOR_PID:
        _native_startup_fail("forked_owner")
    return pid


def _native_startup_validate_record(record: _NativeStartupRecord, *, effect: bool) -> None:
    _native_startup_creator()
    if type(record) is not _NativeStartupRecord or type(record.seal) is not _NativeStartupSeal:
        _native_startup_fail("invalid_owner")
    seal = record.seal
    _native_startup_validate_owner(seal.owner)
    if (seal.owner.version, seal.owner.owner_id, seal.owner.creator_pid) != seal.owner_values or seal.owner.creator_pid != _NATIVE_STARTUP_CREATOR_PID:
        _native_startup_fail("invalid_owner")
    if _NATIVE_STARTUP_RECORDS.get(seal.owner.owner_id) is not record:
        _native_startup_fail("invalid_owner")
    if type(record.resources) is not _NativeStartupResources or record.resources is not seal.resources:
        _native_startup_fail("native_resource_drift")
    if _native_startup_request_key(seal.request) != seal.request_key:
        _native_startup_fail("invalid_input")
    if _native_startup_state_key(record.state) != record.state_key or record.state.owner is not seal.owner or record.state.original_deadline != seal.original_deadline:
        _native_startup_fail("invalid_input")
    if _native_startup_request_key(record.state.request) != seal.request_key:
        _native_startup_fail("invalid_input")
    if type(record.cleanup_attempts) is not list or len(record.cleanup_attempts) > 16:
        _native_startup_fail("native_resource_drift")
    if record.operation is not None:
        _native_startup_validate_token(record, record.operation, cleanup=False)
    if record.cleanup_operation is not None:
        _native_startup_validate_token(record, record.cleanup_operation, cleanup=True)
        if not record.cleanup_attempts or record.cleanup_attempts[-1] is not record.cleanup_operation:
            _native_startup_fail("native_resource_drift")
    for token in record.cleanup_attempts:
        _native_startup_validate_token(record, token, cleanup=True)
    resources = record.resources
    if type(resources.references) is not list or type(resources.frames) is not list or type(resources.directory_fds) is not list or type(resources.callbacks) is not list or type(resources.events) is not list or type(resources.seen_serials) is not set:
        _native_startup_fail("native_resource_drift")
    for value in (resources.active_calls, resources.active_callbacks, resources.evidence_bytes, resources.callback_count, resources.copy_bytes, resources.iterations):
        _native_startup_integer(value)
    if type(resources.fd_release_observed) is not bool or type(resources.bus_closed) is not bool or type(resources.transfer) is not str or resources.transfer not in ("none", "python_owned", "detach_entered", "caller_owned", "attempted_unknown", "bus_owned", "release_entered", "release_unobserved", "released"):
        _native_startup_fail("native_resource_drift")
    if resources.callback_error is not None and (type(resources.callback_error) is not str or resources.callback_error not in _NATIVE_STARTUP_CODES):
        _native_startup_fail("native_resource_drift")
    for reference in resources.references:
        if type(reference) is not _NativeStartupReference or type(reference.role) is not str or reference.role != reference.original_role or reference.role not in ("bus", "filter", "method_slot", "method_message", "process_message") or type(reference.out) is not ctypes.c_void_p or reference.out is not reference.original_out or type(reference.state) is not str or reference.state not in ("acquisition_pending", "owned", "acquisition_unobserved", "release_entered", "released", "release_unobserved") or (reference.pointer is not None and (type(reference.pointer) is not int or not 0 < reference.pointer <= _NATIVE_STARTUP_U64)):
            _native_startup_fail("native_resource_drift")
    if effect and (threading.current_thread() is not seal.thread or threading.get_ident() != seal.thread_ident or threading.get_native_id() != seal.native_id):
        _native_startup_fail("wrong_thread")


def _native_startup_validate_token(record: _NativeStartupRecord, token: object, *, cleanup: bool) -> None:
    expected = _NativeStartupCleanupToken if cleanup else _NativeStartupAuthToken
    if type(token) is not expected or token.record is not record or token.seal is not record.seal or token.resources is not record.resources:
        _native_startup_fail("native_resource_drift")
    _native_startup_integer(token.intent, positive=True)
    _native_startup_integer(token.terminal, positive=True)
    _native_startup_float(token.deadline)
    if type(token.cleanup) is not bool or token.cleanup is not cleanup or type(token.failed) is not bool or type(token.anchor) is not tuple or len(token.anchor) != 4 or tuple(type(value) for value in token.anchor) != (int, int, float, bool) or token.anchor != (token.intent, token.terminal, token.deadline, cleanup) or token.terminal != token.intent + 1:
        _native_startup_fail("native_resource_drift")
    if not cleanup and token.deadline != record.seal.original_deadline:
        _native_startup_fail("native_resource_drift")
    if token.outcome is not None:
        outcome = token.outcome
        if type(outcome) is not tuple or len(outcome) != 5 or type(outcome[0]) is not int or outcome[0] != token.terminal or (outcome[1] is not None and (type(outcome[1]) is not str or outcome[1] not in _NATIVE_STARTUP_CODES)) or type(outcome[2]) is not float or not math.isfinite(outcome[2]) or type(outcome[3]) is not bool or type(outcome[4]) is not bool:
            _native_startup_fail("native_resource_drift")


def _native_startup_enter(owner: NativeStartupOwner | None, *, effect: bool) -> _NativeStartupRecord | None:
    global _NATIVE_STARTUP_OPERATION
    _native_startup_creator()
    if not _NATIVE_STARTUP_LOCK.acquire(blocking=False):
        _native_startup_fail("busy")
    try:
        if _NATIVE_STARTUP_OPERATION is not None:
            _native_startup_fail("busy")
        record = None
        if owner is not None:
            _native_startup_validate_owner(owner)
            record = _NATIVE_STARTUP_RECORDS.get(owner.owner_id)
            if type(record) is not _NativeStartupRecord or record.seal.owner is not owner:
                _native_startup_fail("invalid_owner")
            try:
                _native_startup_validate_record(record, effect=effect)
            except (TypeError, ValueError, AttributeError, KeyError, OverflowError):
                _native_startup_fail("invalid_input")
            if record.resources.active_calls or record.resources.active_callbacks:
                _native_startup_fail("busy")
        _NATIVE_STARTUP_OPERATION = record if record is not None else object()
        return record
    finally:
        _NATIVE_STARTUP_LOCK.release()


def _native_startup_leave(record: _NativeStartupRecord | None) -> None:
    global _NATIVE_STARTUP_OPERATION
    _native_startup_creator()
    if not _NATIVE_STARTUP_LOCK.acquire(blocking=False):
        _native_startup_fail("busy")
    try:
        if record is not None and (record.resources.active_calls or record.resources.active_callbacks):
            return
        _NATIVE_STARTUP_OPERATION = None
    finally:
        _NATIVE_STARTUP_LOCK.release()


def _native_startup_clock(record: _NativeStartupRecord | None = None) -> float:
    now = time.monotonic()
    if type(now) is not float or not math.isfinite(now):
        _native_startup_fail("clock_regressed")
    if record is not None and now < record.state.last_observed_now:
        _native_startup_fail("clock_regressed")
    return now


def _native_startup_commit(record: _NativeStartupRecord, **changes: object) -> None:
    values = {name: getattr(record.state, name) for name in NativeStartupSnapshot.__dataclass_fields__}
    values.update(changes)
    state = NativeStartupSnapshot(**values)
    key = _native_startup_state_key(state)
    _native_startup_validate_record(record, effect=False)
    record.state = state
    record.state_key = key


def _native_startup_snapshot_copy(record: _NativeStartupRecord) -> NativeStartupSnapshot:
    _native_startup_validate_record(record, effect=False)
    values = {name: getattr(record.state, name) for name in NativeStartupSnapshot.__dataclass_fields__}
    values["owner"] = record.seal.owner
    values["request"] = _native_startup_copy_request(record.seal.request)
    if values["context"] is not None:
        context = values["context"]
        values["context"] = StartupBusContext(context.version, context.bus_id, context.manager_unique_name, context.client_unique_name, context.manager_uid)
    return NativeStartupSnapshot(**values)


def retain_native_startup(request: StartupNativeRequest, *, deadline: float) -> NativeStartupOwner:
    global _NATIVE_STARTUP_NEXT_OWNER_ID
    _native_startup_creator()
    _native_startup_validate_request(request)
    _native_startup_float(deadline)
    _native_startup_enter(None, effect=False)
    try:
        now = _native_startup_clock()
        if deadline <= now:
            _native_startup_fail("deadline_expired")
        if len(_NATIVE_STARTUP_RECORDS) >= 16:
            _native_startup_fail("capacity_exhausted")
        if _NATIVE_STARTUP_NEXT_OWNER_ID > _NATIVE_STARTUP_U64:
            _native_startup_fail("sequence_exhausted")
        owner = NativeStartupOwner(1, _NATIVE_STARTUP_NEXT_OWNER_ID, _NATIVE_STARTUP_CREATOR_PID)
        resources = _NativeStartupResources()
        seal = _NativeStartupSeal(owner, request, deadline, resources)
        state = NativeStartupSnapshot(1, owner, _native_startup_copy_request(seal.request), deadline, now, 0, 0,
                                      None, False, None, "prepared", None, None, None, False, False, False,
                                      False, False, True, True, None)
        record = _NativeStartupRecord(seal, state, resources)
        if _native_startup_request_key(request) != seal.request_key:
            _native_startup_fail("invalid_input")
        _NATIVE_STARTUP_RECORDS[owner.owner_id] = record
        _NATIVE_STARTUP_NEXT_OWNER_ID += 1
        return owner
    finally:
        _native_startup_leave(None)


def native_startup_snapshot(owner: NativeStartupOwner) -> NativeStartupSnapshot:
    _native_startup_validate_owner(owner)
    record = _native_startup_enter(owner, effect=False)
    try:
        now = _native_startup_clock(record)
        _native_startup_validate_record(record, effect=False)
        changes: dict[str, object] = {"last_observed_now": now}
        if now >= record.seal.original_deadline and record.state.phase in ("prepared", "subscribed"):
            changes.update(phase="terminal_unknown", reason="deadline_expired")
        _native_startup_commit(record, **changes)
        return _native_startup_snapshot_copy(record)
    finally:
        _native_startup_leave(record)


class _NativeStartupBackend:
    def __init__(self, library: object) -> None:
        pointer = ctypes.c_void_p
        out_pointer = ctypes.POINTER(pointer)
        out_u64 = ctypes.POINTER(ctypes.c_uint64)
        self.handler = ctypes.CFUNCTYPE(ctypes.c_int, pointer, pointer, pointer)
        rows = (
            ("sd_bus_new", (out_pointer,), ctypes.c_int),
            ("sd_bus_set_fd", (pointer, ctypes.c_int, ctypes.c_int), ctypes.c_int),
            ("sd_bus_set_bus_client", (pointer, ctypes.c_int), ctypes.c_int),
            ("sd_bus_set_anonymous", (pointer, ctypes.c_int), ctypes.c_int),
            ("sd_bus_negotiate_fds", (pointer, ctypes.c_int), ctypes.c_int),
            ("sd_bus_set_exit_on_disconnect", (pointer, ctypes.c_int), ctypes.c_int),
            ("sd_bus_start", (pointer,), ctypes.c_int),
            ("sd_bus_is_ready", (pointer,), ctypes.c_int),
            ("sd_bus_process", (pointer, out_pointer), ctypes.c_int),
            ("sd_bus_get_fd", (pointer,), ctypes.c_int),
            ("sd_bus_get_events", (pointer,), ctypes.c_int),
            ("sd_bus_get_timeout", (pointer, out_u64), ctypes.c_int),
            ("sd_bus_get_bus_id", (pointer, ctypes.POINTER(_NativeStartupId128)), ctypes.c_int),
            ("sd_bus_get_unique_name", (pointer, out_pointer), ctypes.c_int),
            ("sd_bus_call_async", (pointer, out_pointer, pointer, self.handler, pointer, ctypes.c_uint64), ctypes.c_int),
            ("sd_bus_add_filter", (pointer, out_pointer, self.handler, pointer), ctypes.c_int),
            ("sd_bus_message_new_method_call", (pointer, out_pointer, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p), ctypes.c_int),
            ("sd_bus_message_set_auto_start", (pointer, ctypes.c_int), ctypes.c_int),
            ("sd_bus_message_set_allow_interactive_authorization", (pointer, ctypes.c_int), ctypes.c_int),
            ("sd_bus_message_append_basic", (pointer, ctypes.c_char, pointer), ctypes.c_int),
            ("sd_bus_message_get_cookie", (pointer, out_u64), ctypes.c_int),
            ("sd_bus_message_get_reply_cookie", (pointer, out_u64), ctypes.c_int),
            ("sd_bus_message_get_sender", (pointer,), pointer),
            ("sd_bus_message_get_destination", (pointer,), pointer),
            ("sd_bus_message_get_type", (pointer, ctypes.POINTER(ctypes.c_uint8)), ctypes.c_int),
            ("sd_bus_message_get_signature", (pointer, ctypes.c_int), pointer),
            ("sd_bus_message_get_path", (pointer,), pointer),
            ("sd_bus_message_get_interface", (pointer,), pointer),
            ("sd_bus_message_get_member", (pointer,), pointer),
            ("sd_bus_message_read_basic", (pointer, ctypes.c_char, pointer), ctypes.c_int),
            ("sd_bus_message_enter_container", (pointer, ctypes.c_char, ctypes.c_char_p), ctypes.c_int),
            ("sd_bus_message_exit_container", (pointer,), ctypes.c_int),
            ("sd_bus_message_at_end", (pointer, ctypes.c_int), ctypes.c_int),
            ("sd_bus_message_unref", (pointer,), pointer),
            ("sd_bus_slot_unref", (pointer,), pointer),
            ("sd_bus_close", (pointer,), None),
            ("sd_bus_unref", (pointer,), pointer),
        )
        self.library = library
        self.functions: dict[str, tuple[object, tuple, object, object]] = {}
        for name, arguments, result in rows:
            function = getattr(library, name)
            if not callable(function):
                _native_startup_fail("backend_abi_unproven")
            function.argtypes = arguments
            function.restype = result
            self.functions[name] = (function, arguments, result, getattr(function, "errcheck", None))
        for name in ("sd_bus_send", "sd_bus_message_ref", "sd_bus_message_open_container", "sd_bus_message_close_container", "sd_bus_add_match_async"):
            if not callable(getattr(library, name)):
                _native_startup_fail("backend_abi_unproven")
        self.captured = tuple((name, *binding) for name, binding in self.functions.items())


def _native_startup_stat_pin(metadata: object) -> tuple:
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_uid, metadata.st_gid,
            metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns, metadata.st_nlink)


def _native_startup_loader_names() -> None:
    for name in os.environ:
        if type(name) is not str:
            _native_startup_fail("invalid_input")
        if name.startswith("LD_") or name == "GLIBC_TUNABLES":
            _native_startup_fail("loader_override_present")


def _native_startup_boundary(record: _NativeStartupRecord, token: _NativeStartupAuthToken, *, clock: bool = True) -> float:
    _native_startup_validate_record(record, effect=True)
    if _NATIVE_STARTUP_OPERATION is not record or token.record is not record or token.seal is not record.seal or token.resources is not record.resources:
        _native_startup_fail("native_resource_drift")
    current = record.cleanup_operation if token.cleanup else record.operation
    if current is not token or type(token) is not (_NativeStartupCleanupToken if token.cleanup else _NativeStartupAuthToken):
        _native_startup_fail("native_resource_drift")
    _native_startup_integer(token.intent)
    _native_startup_integer(token.terminal)
    _native_startup_float(token.deadline)
    if type(token.cleanup) is not bool or (token.intent, token.terminal, token.deadline, token.cleanup) != token.anchor:
        _native_startup_fail("native_resource_drift")
    expected = record.state.cleanup_sequence if token.cleanup else record.state.operation_sequence
    if expected != token.intent or token.terminal != token.intent + 1 or token.deadline != (record.state.cleanup_deadline if token.cleanup else record.seal.original_deadline):
        _native_startup_fail("native_resource_drift")
    resources = record.resources
    for value in (resources.active_calls, resources.active_callbacks, resources.evidence_bytes, resources.callback_count, resources.copy_bytes, resources.iterations):
        _native_startup_integer(value)
    if resources.socket is not None and resources.socket is not record.seal.socket:
        _native_startup_fail("native_resource_drift")
    if resources.bus is not None and resources.bus is not record.seal.bus:
        _native_startup_fail("native_resource_drift")
    if resources.backend is not None:
        _native_startup_validate_backend(record)
    if resources.raw_fd is not None and not resources.fd_release_observed:
        _native_startup_integer(resources.raw_fd)
        if type(resources.raw_pin) is not tuple or len(resources.raw_pin) != 5:
            _native_startup_fail("native_resource_drift")
        for value in resources.raw_pin:
            _native_startup_integer(value)
        try:
            metadata = os.fstat(resources.raw_fd)
        except OSError:
            _native_startup_fail("native_resource_drift")
        if _native_startup_stat_pin(metadata)[:5] != resources.raw_pin:
            _native_startup_fail("native_resource_drift")
    if resources.callback_error is not None and not token.cleanup:
        _native_startup_fail(resources.callback_error)
    if not clock:
        return record.state.last_observed_now
    now = _native_startup_clock(record)
    _native_startup_validate_record(record, effect=True)
    if _NATIVE_STARTUP_OPERATION is not record or (record.cleanup_operation if token.cleanup else record.operation) is not token:
        _native_startup_fail("native_resource_drift")
    if now >= token.deadline:
        _native_startup_fail("deadline_expired")
    _native_startup_commit(record, last_observed_now=now)
    return now


def _native_startup_validate_backend(record: _NativeStartupRecord) -> _NativeStartupBackend:
    backend = record.resources.backend
    if type(backend) is not _NativeStartupBackend or backend is not record.seal.backend or backend.library is not record.resources.library:
        _native_startup_fail("native_resource_drift")
    if type(backend.functions) is not dict or len(backend.functions) != 37 or type(backend.captured) is not tuple or len(backend.captured) != 37:
        _native_startup_fail("backend_abi_unproven")
    try:
        for name, function, arguments, result, checker in backend.captured:
            if type(name) is not str or backend.functions.get(name) != (function, arguments, result, checker) or getattr(backend.library, name) is not function or tuple(getattr(function, "argtypes", ())) != arguments or getattr(function, "restype", object()) is not result or getattr(function, "errcheck", None) is not checker:
                _native_startup_fail("backend_abi_unproven")
    except (TypeError, ValueError, AttributeError, KeyError):
        _native_startup_fail("backend_abi_unproven")
    return backend


def _native_startup_pointer(reference: _NativeStartupReference) -> ctypes.c_void_p:
    if type(reference) is not _NativeStartupReference or type(reference.role) is not str or reference.role != reference.original_role or reference.state != "owned" or type(reference.pointer) is not int or not 0 < reference.pointer <= _NATIVE_STARTUP_U64 or type(reference.out) is not ctypes.c_void_p or reference.out is not reference.original_out or type(reference.out.value) is not int or reference.out.value != reference.pointer:
        _native_startup_fail("native_pointer_unproven")
    return ctypes.c_void_p(reference.pointer)


def _native_startup_call(record: _NativeStartupRecord, token: _NativeStartupAuthToken, name: str, arguments: tuple,
                         *, acquire: _NativeStartupReference | None = None,
                         release: _NativeStartupReference | None = None) -> object:
    resources = record.resources
    backend = _native_startup_validate_backend(record)
    if type(name) is not str or name not in backend.functions:
        _native_startup_fail("invalid_input")
    if token.cleanup and name not in ("sd_bus_close", "sd_bus_message_unref", "sd_bus_slot_unref", "sd_bus_unref"):
        _native_startup_fail("invalid_input")
    function = backend.functions[name][0]
    outputs = () if acquire is None else (acquire,)
    frame = _NativeStartupCallFrame(name, function, arguments, outputs)
    resources.frames.append(frame)
    if acquire is not None:
        resources.references.append(acquire)
        _native_startup_commit(record, native_objects_released=False)
    if release is not None and release.state != "owned":
        _native_startup_fail("native_release_unobserved")
    _native_startup_boundary(record, token)
    if backend.functions[name][0] is not function:
        _native_startup_fail("backend_abi_unproven")
    if release is not None:
        release.state = "release_entered"
        release.release_frame = frame
    if name == "sd_bus_set_fd":
        resources.transfer = "attempted_unknown"
    frame.entered = True
    resources.active_calls += 1
    try:
        result = function(*arguments)
        frame.result = result
        frame.returned = True
        resources.last_frame = frame
        if acquire is not None:
            pointer = acquire.out.value
            valid = type(result) is int and (result == 0 if name in ("sd_bus_new", "sd_bus_message_new_method_call", "sd_bus_add_filter") else result >= 0)
            if name == "sd_bus_process":
                valid = type(result) is int and result > 0
            if valid and type(pointer) is int and 0 < pointer <= _NATIVE_STARTUP_U64:
                acquire.pointer, acquire.state = pointer, "owned"
            elif pointer is None and name == "sd_bus_process" and type(result) is int and result >= 0:
                acquire.state = "released"
            else:
                acquire.pointer, acquire.state = pointer, "acquisition_unobserved"
        if name == "sd_bus_set_fd":
            if type(result) is int and result == 0:
                resources.transfer = "bus_owned"
            elif type(result) is int and result < 0:
                resources.transfer = "caller_owned"
        if name == "sd_bus_close" and result is None:
            resources.bus_closed = True
            if resources.transfer == "bus_owned":
                resources.fd_release_observed = True
                resources.transfer = "released"
        if release is not None:
            release.state = "released" if result is None else "release_unobserved"
    except BaseException:
        if release is not None:
            release.state = "release_unobserved"
        raise
    finally:
        # A lost native return cannot authorize reuse or cleanup of its resources.
        if frame.returned:
            resources.active_calls -= 1
    _native_startup_boundary(record, token)
    return result


def _native_startup_result(result: object, *, value: int | None = None, nonnegative: bool = False) -> int:
    if type(result) is not int or (value is not None and result != value) or (nonnegative and result < 0):
        _native_startup_fail("native_call_failed")
    return result


def _native_startup_open_directory(record: _NativeStartupRecord, token: _NativeStartupAuthToken, path: str, uid: int) -> int:
    _native_startup_boundary(record, token)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    entry = [descriptor, None, "owned"]
    record.resources.directory_fds.append(entry)
    metadata = os.fstat(descriptor)
    entry[1] = _native_startup_stat_pin(metadata)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != uid or metadata.st_mode & 0o022:
        _native_startup_fail("endpoint_pin_mismatch")
    current = os.stat(path, follow_symlinks=False)
    if _native_startup_stat_pin(current) != entry[1]:
        _native_startup_fail("endpoint_pin_mismatch")
    _native_startup_boundary(record, token)
    return descriptor


def _native_startup_close_directory(record: _NativeStartupRecord, token: _NativeStartupAuthToken, entry: list) -> None:
    if entry[2] == "released":
        return
    if entry[2] != "owned" or entry[1] is None:
        _native_startup_fail("cleanup_unresolved")
    _native_startup_boundary(record, token)
    if _native_startup_stat_pin(os.fstat(entry[0])) != entry[1]:
        _native_startup_fail("native_resource_drift")
    entry[2] = "release_entered"
    os.close(entry[0])
    entry[2] = "released"
    _native_startup_boundary(record, token)


def _native_startup_load_backend(record: _NativeStartupRecord, token: _NativeStartupAuthToken) -> None:
    request = record.seal.request
    pin = request.backend_pin
    if (pin.canonical_path, pin.sha256, pin.stat_pin) != (_NATIVE_STARTUP_LIBRARY, _NATIVE_STARTUP_LIBRARY_SHA, _NATIVE_STARTUP_LIBRARY_STAT):
        _native_startup_fail("backend_pin_mismatch")
    _native_startup_loader_names()
    if sys.platform != "linux" or sys.byteorder != "little" or ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_uint64) != 8 or ctypes.sizeof(ctypes.c_uint32) != 4 or ctypes.sizeof(ctypes.c_uint8) != 1 or ctypes.sizeof(_NativeStartupId128) != 16 or ctypes.alignment(_NativeStartupId128) != 8:
        _native_startup_fail("backend_abi_unproven")
    paths = ("/", "/usr", "/usr/lib", "/usr/lib/x86_64-linux-gnu")
    for path in paths:
        _native_startup_open_directory(record, token, path, 0)
    _native_startup_boundary(record, token)
    descriptor = os.open(pin.canonical_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    record.resources.library_fd = descriptor
    entry = [descriptor, None, "owned"]
    record.resources.directory_fds.append(entry)
    entry[1] = _native_startup_stat_pin(os.fstat(descriptor))
    if entry[1] != pin.stat_pin:
        _native_startup_fail("backend_pin_mismatch")
    digest = hashlib.sha256()
    total = 0
    while True:
        _native_startup_boundary(record, token)
        block = os.read(descriptor, min(65536, 4194305 - total))
        if not block:
            break
        total += len(block)
        if total > 4194304:
            _native_startup_fail("backend_pin_mismatch")
        digest.update(block)
    if total != pin.stat_pin[5] or digest.hexdigest() != pin.sha256 or _native_startup_stat_pin(os.fstat(descriptor)) != pin.stat_pin or _native_startup_stat_pin(os.stat(pin.canonical_path, follow_symlinks=False)) != pin.stat_pin:
        _native_startup_fail("backend_pin_mismatch")
    _native_startup_loader_names()
    _native_startup_boundary(record, token)
    library = ctypes.CDLL(pin.canonical_path, mode=os.RTLD_LOCAL | os.RTLD_NOW)
    record.resources.library = library
    _native_startup_boundary(record, token)
    if _native_startup_stat_pin(os.stat(pin.canonical_path, follow_symlinks=False)) != pin.stat_pin:
        _native_startup_fail("backend_pin_mismatch")
    for directory, path in zip(record.resources.directory_fds[:-1], paths):
        if _native_startup_stat_pin(os.fstat(directory[0])) != directory[1] or _native_startup_stat_pin(os.stat(path, follow_symlinks=False)) != directory[1]:
            _native_startup_fail("backend_pin_mismatch")
    backend = _NativeStartupBackend(library)
    record.resources.backend = backend
    record.seal.backend = backend
    _native_startup_boundary(record, token)
    for directory in record.resources.directory_fds:
        _native_startup_close_directory(record, token, directory)


def _native_startup_poll(record: _NativeStartupRecord, token: _NativeStartupAuthToken, descriptor: int, events: int, timeout: float) -> None:
    resources = record.resources
    if resources.iterations >= 4096:
        _native_startup_fail("evidence_overflow")
    resources.iterations += 1
    now = _native_startup_boundary(record, token)
    remaining = min(token.deadline - now, timeout)
    if not math.isfinite(remaining) or remaining < 0:
        _native_startup_fail("deadline_expired")
    poller = select.poll()
    poller.register(descriptor, events)
    returned = poller.poll(min(2147483647, max(0, int(remaining * 1000))))
    _native_startup_boundary(record, token)
    for fd, flags in returned:
        if fd != descriptor or flags & (select.POLLERR | select.POLLHUP | select.POLLNVAL):
            _native_startup_fail("transport_disconnected")


def _native_startup_open_endpoint(record: _NativeStartupRecord, token: _NativeStartupAuthToken) -> None:
    seal, resources = record.seal, record.resources
    if type(seal.uid) is not int or seal.uid <= 0 or seal.uid != seal.euid or os.getuid() != seal.uid or os.geteuid() != seal.euid:
        _native_startup_fail("peer_auth_failed")
    root = "/run/user/" + str(seal.uid)
    runtime = _native_startup_open_directory(record, token, root, seal.uid)
    if stat.S_IMODE(os.fstat(runtime).st_mode) != 0o700:
        _native_startup_fail("endpoint_pin_mismatch")
    endpoint = os.stat("bus", dir_fd=runtime, follow_symlinks=False)
    endpoint_pin = _native_startup_stat_pin(endpoint)
    if not stat.S_ISSOCK(endpoint.st_mode) or endpoint.st_uid != seal.uid:
        _native_startup_fail("endpoint_pin_mismatch")
    _native_startup_boundary(record, token)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_NONBLOCK | socket.SOCK_CLOEXEC)
    resources.socket = connection
    record.seal.socket = connection
    descriptor = connection.fileno()
    resources.raw_fd = descriptor
    resources.raw_pin = _native_startup_stat_pin(os.fstat(descriptor))[:5]
    resources.transfer = "python_owned"
    _native_startup_commit(record, connection_closed=False)
    _native_startup_boundary(record, token)
    result = connection.connect_ex(root + "/bus")
    if type(result) is not int or result not in (0, 4, 11, 114, 115):
        _native_startup_fail("endpoint_unavailable")
    if result:
        _native_startup_poll(record, token, descriptor, select.POLLOUT, token.deadline - record.state.last_observed_now)
        if connection.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) != 0:
            _native_startup_fail("endpoint_unavailable")
    raw_peer = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    if type(raw_peer) is not bytes or len(raw_peer) != 12:
        _native_startup_fail("peer_auth_failed")
    peer = struct.unpack("3i", raw_peer)
    if peer[0] <= 0 or peer[1] != seal.uid or peer[2] < 0:
        _native_startup_fail("peer_auth_failed")
    resources.peer = peer
    _native_startup_commit(record, peer_credentials=peer)
    if _native_startup_stat_pin(os.stat("bus", dir_fd=runtime, follow_symlinks=False)) != endpoint_pin:
        _native_startup_fail("endpoint_pin_mismatch")
    _native_startup_boundary(record, token)
    resources.transfer = "detach_entered"
    detached = connection.detach()
    if type(detached) is not int or detached != descriptor:
        _native_startup_fail("native_resource_drift")
    resources.transfer = "caller_owned"
    _native_startup_boundary(record, token)
    out = ctypes.c_void_p()
    bus = _NativeStartupReference("bus", out)
    resources.bus = bus
    record.seal.bus = bus
    _native_startup_result(_native_startup_call(record, token, "sd_bus_new", (ctypes.byref(out),), acquire=bus), value=0)
    pointer = _native_startup_pointer(bus)
    _native_startup_result(_native_startup_call(record, token, "sd_bus_set_fd", (pointer, descriptor, descriptor)), value=0)
    for name, value in (("sd_bus_set_bus_client", 1), ("sd_bus_set_anonymous", 0), ("sd_bus_negotiate_fds", 0), ("sd_bus_set_exit_on_disconnect", 0)):
        _native_startup_result(_native_startup_call(record, token, name, (pointer, value)), value=0)
    _native_startup_result(_native_startup_call(record, token, "sd_bus_start", (pointer,)), nonnegative=True)


def _native_startup_unref(record: _NativeStartupRecord, token: _NativeStartupAuthToken, reference: _NativeStartupReference) -> None:
    if reference.state == "released":
        return
    name = "sd_bus_unref" if reference.role == "bus" else ("sd_bus_slot_unref" if reference.role in ("filter", "method_slot") else "sd_bus_message_unref")
    pointer = _native_startup_pointer(reference)
    result = _native_startup_call(record, token, name, (pointer,), release=reference)
    if result is not None:
        _native_startup_fail("native_release_unobserved")


def _native_startup_process_once(record: _NativeStartupRecord, token: _NativeStartupAuthToken) -> None:
    resources = record.resources
    if resources.iterations >= 4096:
        _native_startup_fail("evidence_overflow")
    resources.iterations += 1
    bus = _native_startup_pointer(resources.bus)
    out = ctypes.c_void_p()
    reference = _NativeStartupReference("process_message", out)
    result = _native_startup_call(record, token, "sd_bus_process", (bus, ctypes.byref(out)), acquire=reference)
    _native_startup_result(result, nonnegative=True)
    if reference.state == "owned":
        _native_startup_unref(record, token, reference)
    elif reference.state != "released":
        _native_startup_fail("native_pointer_unproven")
    if result > 0:
        return
    descriptor = _native_startup_result(_native_startup_call(record, token, "sd_bus_get_fd", (bus,)), nonnegative=True)
    if descriptor != resources.raw_fd:
        _native_startup_fail("native_resource_drift")
    events = _native_startup_result(_native_startup_call(record, token, "sd_bus_get_events", (bus,)), nonnegative=True)
    if events & ~(select.POLLIN | select.POLLOUT | select.POLLPRI | select.POLLERR | select.POLLHUP | select.POLLNVAL):
        _native_startup_fail("malformed_evidence")
    timeout = ctypes.c_uint64()
    timed = _native_startup_result(_native_startup_call(record, token, "sd_bus_get_timeout", (bus, ctypes.byref(timeout))))
    if timed not in (0, 1):
        _native_startup_fail("malformed_evidence")
    now = _native_startup_boundary(record, token)
    seconds = token.deadline - now
    if timed == 1 and timeout.value != _NATIVE_STARTUP_U64:
        seconds = min(seconds, max(0.0, timeout.value / 1000000.0 - now))
    _native_startup_poll(record, token, descriptor, events, seconds)


def _native_startup_timeout_us(record: _NativeStartupRecord, token: _NativeStartupAuthToken) -> int:
    now = _native_startup_boundary(record, token)
    residual = token.deadline - now
    scaled = residual * 1000000.0
    if not math.isfinite(scaled) or scaled > _NATIVE_STARTUP_U64:
        _native_startup_fail("invalid_input")
    value = math.floor(scaled)
    if value < 1:
        _native_startup_fail("deadline_expired")
    return value


def _native_startup_scan_string(record: _NativeStartupRecord, token: _NativeStartupAuthToken, pointer: object,
                                limit: int, *, nullable: bool = False, empty: bool = False) -> str | None:
    frame = record.resources.last_frame
    if type(frame) is not _NativeStartupCallFrame or not frame.returned:
        _native_startup_fail("native_pointer_unproven")
    getters = ("sd_bus_get_unique_name", "sd_bus_message_get_sender", "sd_bus_message_get_destination", "sd_bus_message_get_signature", "sd_bus_message_get_path", "sd_bus_message_get_interface", "sd_bus_message_get_member", "sd_bus_message_read_basic")
    if frame.name not in getters:
        _native_startup_fail("native_pointer_unproven")
    if frame.name in ("sd_bus_get_unique_name", "sd_bus_message_read_basic"):
        cell = getattr(frame.arguments[-1], "_obj", None)
        observed = cell.value if type(cell) is ctypes.c_void_p else object()
    else:
        observed = frame.result
    if pointer != observed or type(pointer) is not type(observed):
        _native_startup_fail("native_pointer_unproven")
    if pointer is None:
        if nullable:
            return None
        _native_startup_fail("native_pointer_unproven")
    if type(pointer) is not int or not 0 < pointer <= _NATIVE_STARTUP_U64 or type(limit) is not int or not 0 <= limit <= 4096:
        _native_startup_fail("native_pointer_unproven")
    data = bytearray()
    for index in range(limit + 1):
        _native_startup_boundary(record, token)
        if pointer > _NATIVE_STARTUP_U64 - index:
            _native_startup_fail("native_pointer_unproven")
        value = ctypes.c_ubyte.from_address(pointer + index).value
        if type(value) is not int or not 0 <= value <= 255:
            _native_startup_fail("native_pointer_unproven")
        if value == 0:
            if not data and not empty:
                _native_startup_fail("malformed_evidence")
            try:
                return bytes(data).decode("ascii", "strict")
            except UnicodeError:
                _native_startup_fail("malformed_evidence")
        if index == limit:
            _native_startup_fail("native_string_unterminated")
        if record.resources.evidence_bytes + record.resources.copy_bytes + 1 > 65536:
            _native_startup_fail("evidence_overflow")
        record.resources.copy_bytes += 1
        data.append(value)
    _native_startup_fail("native_string_unterminated")


def _native_startup_cookie(record: _NativeStartupRecord, token: _NativeStartupAuthToken, message: ctypes.c_void_p, *, reply: bool = False) -> int:
    out = ctypes.c_uint64()
    name = "sd_bus_message_get_reply_cookie" if reply else "sd_bus_message_get_cookie"
    _native_startup_result(_native_startup_call(record, token, name, (message, ctypes.byref(out))), value=0)
    if not 0 < out.value <= _NATIVE_STARTUP_U32:
        _native_startup_fail("malformed_evidence")
    return out.value


def _native_startup_header_string(record: _NativeStartupRecord, token: _NativeStartupAuthToken, message: ctypes.c_void_p,
                                  name: str, *, nullable: bool = False, empty: bool = False) -> str | None:
    arguments = (message, 1) if name == "sd_bus_message_get_signature" else (message,)
    pointer = _native_startup_call(record, token, name, arguments)
    return _native_startup_scan_string(record, token, pointer, 4096 if name == "sd_bus_message_get_path" else 255, nullable=nullable, empty=empty)


def _native_startup_message_type(record: _NativeStartupRecord, token: _NativeStartupAuthToken, message: ctypes.c_void_p) -> int:
    out = ctypes.c_uint8()
    _native_startup_result(_native_startup_call(record, token, "sd_bus_message_get_type", (message, ctypes.byref(out))), value=0)
    if out.value not in (1, 2, 3, 4):
        _native_startup_fail("malformed_evidence")
    return out.value


def _native_startup_read_string(record: _NativeStartupRecord, token: _NativeStartupAuthToken, message: ctypes.c_void_p,
                                *, limit: int = 255, empty: bool = False) -> str:
    out = ctypes.c_void_p()
    _native_startup_result(_native_startup_call(record, token, "sd_bus_message_read_basic", (message, b"s", ctypes.byref(out))), value=1)
    return _native_startup_scan_string(record, token, out.value, limit, empty=empty)


def _native_startup_evidence_size(value: object) -> int:
    if value is None:
        return 4
    if type(value) is bool:
        return 4 if value else 5
    if type(value) is int:
        return len(str(value))
    if type(value) is str:
        size = 2
        for character in value:
            number = ord(character)
            size += 2 if character in ('"', "\\", "\b", "\f", "\n", "\r", "\t") else (6 if number < 32 or 127 <= number <= 65535 else (12 if number > 65535 else 1))
        return size
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            _native_startup_fail("malformed_evidence")
        return 2 + max(0, len(value) - 1) + sum(_native_startup_evidence_size(key) + 1 + _native_startup_evidence_size(item) for key, item in value.items())
    if type(value) in (tuple, list):
        return 2 + max(0, len(value) - 1) + sum(_native_startup_evidence_size(item) for item in value)
    _native_startup_fail("malformed_evidence")


def _native_startup_evidence(record: _NativeStartupRecord, event: dict) -> None:
    size = _native_startup_evidence_size(event)
    if len(record.resources.events) >= 64 or record.resources.evidence_bytes + size > 65536:
        _native_startup_fail("evidence_overflow")
    serial_key = (event["sender"], event["serial"])
    if serial_key in record.resources.seen_serials:
        _native_startup_fail("malformed_evidence")
    copied = json.loads(json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    record.resources.events.append(copied)
    record.resources.seen_serials.add(serial_key)
    record.resources.evidence_bytes += size


def _native_startup_callback_enter(binding: _NativeStartupCallbackBinding, message: object, userdata: object) -> ctypes.c_void_p:
    if type(binding) is not _NativeStartupCallbackBinding or type(binding.anchor) is not tuple or len(binding.anchor) != 6:
        _native_startup_fail("callback_failed")
    record, token = binding.record, binding.token
    if (binding.anchor[0] is not record or binding.anchor[1] is not token or binding.anchor[2] is not record.resources or binding.anchor[3] is not record.seal or type(binding.cell) is not ctypes.c_uint64 or type(binding.userdata) is not ctypes.c_void_p or type(binding.anchor[4]) is not int or binding.anchor[4] != binding.userdata.value or binding.anchor[4] != ctypes.addressof(binding.cell) or type(binding.kind) is not str or binding.kind != binding.anchor[5] or binding.kind not in ("filter", "reply")):
        _native_startup_fail("callback_failed")
    _native_startup_validate_record(record, effect=True)
    if userdata != binding.userdata.value or type(userdata) is not int or type(message) is not int or not 0 < message <= _NATIVE_STARTUP_U64:
        _native_startup_fail("callback_failed")
    if binding not in record.resources.callbacks or record.resources.active_callbacks or not record.resources.active_calls:
        _native_startup_fail("callback_failed")
    if record.resources.callback_count >= 64 or record.resources.evidence_bytes >= 65536:
        _native_startup_fail("evidence_overflow")
    record.resources.callback_count += 1
    record.resources.copy_bytes = 0
    record.resources.active_callbacks += 1
    record.resources.borrowed_message = message
    try:
        _native_startup_boundary(record, token)
        return ctypes.c_void_p(message)
    except BaseException:
        record.resources.borrowed_message = None
        record.resources.active_callbacks -= 1
        raise


def _native_startup_callback_exit(resources: object, entered: bool) -> None:
    try:
        if entered and type(resources) is _NativeStartupResources:
            resources.borrowed_message = None
            if type(resources.active_callbacks) is not int or resources.active_callbacks <= 0:
                resources.callback_error = "callback_failed"
            else:
                resources.active_callbacks -= 1
    except BaseException:
        return


def _native_startup_latch_callback(resources: object, error: BaseException) -> None:
    try:
        if type(resources) is _NativeStartupResources:
            resources.callback_error = _native_startup_error_code(error, "callback_failed")
    except BaseException:
        return


def _native_startup_error_code(error: BaseException, fallback: str) -> str:
    if type(error) is not EnforcementUnavailable:
        return fallback
    arguments = error.args
    if len(arguments) == 1 and type(arguments[0]) is str:
        message = arguments[0]
        if message.startswith("native_startup:") and message[15:] in _NATIVE_STARTUP_CODES:
            return message[15:]
    return fallback


def _native_startup_filter(binding: _NativeStartupCallbackBinding, message: object, userdata: object, error: object,
                           *, resources: object) -> int:
    entered = False
    try:
        pointer = _native_startup_callback_enter(binding, message, userdata)
        entered = True
        record, token = binding.record, binding.token
        if _native_startup_message_type(record, token, pointer) != 4:
            return 0
        path = _native_startup_header_string(record, token, pointer, "sd_bus_message_get_path", nullable=True)
        interface = _native_startup_header_string(record, token, pointer, "sd_bus_message_get_interface", nullable=True)
        member = _native_startup_header_string(record, token, pointer, "sd_bus_message_get_member", nullable=True)
        if (path, interface, member) != ("/org/freedesktop/DBus", "org.freedesktop.DBus", "NameOwnerChanged"):
            return 0
        sender = _native_startup_header_string(record, token, pointer, "sd_bus_message_get_sender")
        if sender != "org.freedesktop.DBus":
            return 0
        destination = _native_startup_header_string(record, token, pointer, "sd_bus_message_get_destination", nullable=True)
        if destination not in (None, record.resources.client_name):
            _native_startup_fail("malformed_evidence")
        signature = _native_startup_header_string(record, token, pointer, "sd_bus_message_get_signature", empty=True)
        if signature != "sss":
            _native_startup_fail("malformed_evidence")
        serial = _native_startup_cookie(record, token, pointer)
        name = _native_startup_read_string(record, token, pointer)
        old = _native_startup_read_string(record, token, pointer, empty=True)
        new = _native_startup_read_string(record, token, pointer, empty=True)
        _native_startup_result(_native_startup_call(record, token, "sd_bus_message_at_end", (pointer, 1)), value=1)
        if name != "org.freedesktop.systemd1":
            return 0
        _native_startup_evidence(record, {"kind": "owner_change", "sender": sender, "serial": serial, "destination": destination, "old": old, "new": new})
        _native_startup_fail("manager_owner_changed")
    except BaseException as caught:
        _native_startup_latch_callback(resources, caught)
        return -74
    finally:
        _native_startup_callback_exit(resources, entered)


def _native_startup_reply(binding: _NativeStartupCallbackBinding, message: object, userdata: object, error: object,
                          *, resources: object) -> int:
    entered = False
    try:
        pointer = _native_startup_callback_enter(binding, message, userdata)
        entered = True
        record, token = binding.record, binding.token
        pending = record.resources.pending
        if type(pending) is not _NativeStartupPendingReply or pending.binding is not binding or pending.reply is not None:
            _native_startup_fail("malformed_evidence")
        kind = _native_startup_message_type(record, token, pointer)
        sender = _native_startup_header_string(record, token, pointer, "sd_bus_message_get_sender")
        destination = _native_startup_header_string(record, token, pointer, "sd_bus_message_get_destination")
        cookie = _native_startup_cookie(record, token, pointer, reply=True)
        serial = _native_startup_cookie(record, token, pointer)
        if sender != pending.destination or destination != record.resources.client_name or (pending.cookie is not None and cookie != pending.cookie):
            _native_startup_fail("malformed_evidence")
        if kind != 2:
            _native_startup_fail("native_call_failed")
        signature = _native_startup_header_string(record, token, pointer, "sd_bus_message_get_signature", empty=True)
        expected = {"empty": "", "owner": "s", "uid": "u", "version": "v"}[pending.result_kind]
        if signature != expected:
            _native_startup_fail("malformed_evidence")
        value: object = None
        if pending.result_kind == "owner":
            value = _native_startup_read_string(record, token, pointer)
            if re.fullmatch(r":[0-9]+\.[0-9]+", value, flags=re.ASCII) is None:
                _native_startup_fail("manager_auth_failed")
        elif pending.result_kind == "uid":
            out = ctypes.c_uint32()
            _native_startup_result(_native_startup_call(record, token, "sd_bus_message_read_basic", (pointer, b"u", ctypes.byref(out))), value=1)
            value = out.value
        elif pending.result_kind == "version":
            _native_startup_result(_native_startup_call(record, token, "sd_bus_message_enter_container", (pointer, b"v", b"s")), value=1)
            value = _native_startup_read_string(record, token, pointer, limit=256)
            if re.fullmatch(_NATIVE_STARTUP_VERSION, value, flags=re.ASCII) is None:
                _native_startup_fail("manager_auth_failed")
            _native_startup_result(_native_startup_call(record, token, "sd_bus_message_exit_container", (pointer,)), value=1)
        _native_startup_result(_native_startup_call(record, token, "sd_bus_message_at_end", (pointer, 1)), value=1)
        event = {"kind": "reply", "sender": sender, "destination": destination, "serial": serial, "cookie": cookie, "result_kind": pending.result_kind, "value": value}
        _native_startup_evidence(record, event)
        pending.reply = event
        return 1
    except BaseException as caught:
        _native_startup_latch_callback(resources, caught)
        return -74
    finally:
        _native_startup_callback_exit(resources, entered)


def _native_startup_install_filter(record: _NativeStartupRecord, token: _NativeStartupAuthToken) -> None:
    backend = _native_startup_validate_backend(record)
    binding = _NativeStartupCallbackBinding(record, token, "filter")
    resources = record.resources
    binding.callback = backend.handler(lambda message, userdata, error: _native_startup_filter(binding, message, userdata, error, resources=resources))
    record.resources.callbacks.append(binding)
    record.resources.filter_binding = binding
    out = ctypes.c_void_p()
    slot = _NativeStartupReference("filter", out)
    binding.slot = slot
    _native_startup_result(_native_startup_call(record, token, "sd_bus_add_filter", (_native_startup_pointer(record.resources.bus), ctypes.byref(out), binding.callback, binding.userdata), acquire=slot), value=0)
    _native_startup_pointer(slot)


def _native_startup_method(record: _NativeStartupRecord, token: _NativeStartupAuthToken, destination: str,
                            path: str, interface: str, member: str, inputs: tuple[str, ...], result_kind: str) -> object:
    if record.resources.pending is not None:
        _native_startup_fail("busy")
    backend = _native_startup_validate_backend(record)
    binding = _NativeStartupCallbackBinding(record, token, "reply")
    resources = record.resources
    binding.callback = backend.handler(lambda message, userdata, error: _native_startup_reply(binding, message, userdata, error, resources=resources))
    pending = _NativeStartupPendingReply(destination, result_kind, binding)
    record.resources.callbacks.append(binding)
    record.resources.pending = pending
    out = ctypes.c_void_p()
    message = _NativeStartupReference("method_message", out)
    pending.message = message
    headers = tuple(value.encode("ascii", "strict") for value in (destination, path, interface, member))
    pending.inputs.extend(headers)
    _native_startup_result(_native_startup_call(record, token, "sd_bus_message_new_method_call", (_native_startup_pointer(record.resources.bus), ctypes.byref(out), *headers), acquire=message), value=0)
    pointer = _native_startup_pointer(message)
    for name in ("sd_bus_message_set_auto_start", "sd_bus_message_set_allow_interactive_authorization"):
        _native_startup_result(_native_startup_call(record, token, name, (pointer, 0)), value=0)
    for value in inputs:
        _native_startup_text(value, bound=4096, ascii_only=True)
        buffer = ctypes.create_string_buffer(value.encode("ascii", "strict"))
        pending.inputs.append(buffer)
        _native_startup_result(_native_startup_call(record, token, "sd_bus_message_append_basic", (pointer, b"s", ctypes.cast(buffer, ctypes.c_void_p))), nonnegative=True)
    slot_out = ctypes.c_void_p()
    slot = _NativeStartupReference("method_slot", slot_out)
    pending.slot = binding.slot = slot
    timeout = _native_startup_timeout_us(record, token)
    _native_startup_result(_native_startup_call(record, token, "sd_bus_call_async", (_native_startup_pointer(record.resources.bus), ctypes.byref(slot_out), pointer, binding.callback, binding.userdata, timeout), acquire=slot), nonnegative=True)
    _native_startup_pointer(slot)
    pending.cookie = _native_startup_cookie(record, token, pointer)
    if pending.reply is not None and pending.reply["cookie"] != pending.cookie:
        _native_startup_fail("malformed_evidence")
    while pending.reply is None:
        _native_startup_process_once(record, token)
    _native_startup_boundary(record, token)
    value = pending.reply["value"]
    _native_startup_unref(record, token, slot)
    _native_startup_unref(record, token, message)
    record.resources.pending = None
    return value


def _native_startup_authenticate(record: _NativeStartupRecord, token: _NativeStartupAuthToken) -> None:
    _native_startup_load_backend(record, token)
    _native_startup_open_endpoint(record, token)
    bus = _native_startup_pointer(record.resources.bus)
    while True:
        ready = _native_startup_result(_native_startup_call(record, token, "sd_bus_is_ready", (bus,)))
        if ready not in (0, 1):
            _native_startup_fail("malformed_evidence")
        if ready:
            break
        _native_startup_process_once(record, token)
    identifier = _NativeStartupId128()
    _native_startup_result(_native_startup_call(record, token, "sd_bus_get_bus_id", (bus, ctypes.byref(identifier))), value=0)
    record.resources.bus_id = bytes(identifier.bytes).hex()
    out = ctypes.c_void_p()
    _native_startup_result(_native_startup_call(record, token, "sd_bus_get_unique_name", (bus, ctypes.byref(out))), value=0)
    name = _native_startup_scan_string(record, token, out.value, 255)
    if re.fullmatch(r":[0-9]+\.[0-9]+", name, flags=re.ASCII) is None:
        _native_startup_fail("malformed_evidence")
    record.resources.client_name = name
    _native_startup_install_filter(record, token)
    broker, broker_path = "org.freedesktop.DBus", "/org/freedesktop/DBus"
    _native_startup_method(record, token, broker, broker_path, broker, "AddMatch", (_NATIVE_STARTUP_MATCH,), "empty")
    _native_startup_commit(record, owner_match_confirmed=True)
    manager = _native_startup_method(record, token, broker, broker_path, broker, "GetNameOwner", ("org.freedesktop.systemd1",), "owner")
    record.resources.manager_name = manager
    uid = _native_startup_method(record, token, broker, broker_path, broker, "GetConnectionUnixUser", (manager,), "uid")
    if type(uid) is not int or uid != record.seal.uid or uid != record.seal.euid:
        _native_startup_fail("manager_auth_failed")
    record.resources.manager_uid = uid
    version = _native_startup_method(record, token, manager, "/org/freedesktop/systemd1", "org.freedesktop.DBus.Properties", "Get", ("org.freedesktop.systemd1.Manager", "Version"), "version")
    record.resources.manager_version = version
    context = StartupBusContext(1, record.resources.bus_id, manager, name, uid)
    _native_startup_commit(record, context=context, manager_version=version)
    _native_startup_method(record, token, manager, "/org/freedesktop/systemd1", "org.freedesktop.systemd1.Manager", "Subscribe", (), "empty")
    _native_startup_commit(record, subscription_confirmed=True)


def _native_startup_gone(record: _NativeStartupRecord) -> tuple[bool, bool]:
    resources = record.resources
    connection = resources.raw_fd is None or resources.fd_release_observed
    native = all(reference.state == "released" for reference in resources.references)
    native = native and all(entry[2] == "released" for entry in resources.directory_fds)
    if resources.bus is not None:
        connection = connection and resources.bus_closed
    return connection, native


def _native_startup_finish(record: _NativeStartupRecord, token: _NativeStartupAuthToken, code: str | None) -> NativeStartupSnapshot:
    token.failed = code is not None
    now = record.state.last_observed_now
    try:
        now = _native_startup_clock(record)
        _native_startup_validate_record(record, effect=True)
        if now >= token.deadline and code is None:
            code = "deadline_expired"
    except EnforcementUnavailable:
        code = "clock_regressed"
        now = record.state.last_observed_now
    connection, native = _native_startup_gone(record)
    changes: dict[str, object] = {"last_observed_now": now, "connection_closed": connection, "native_objects_released": native}
    if token.cleanup:
        closed = code is None and connection and native
        changes.update(cleanup_sequence=token.terminal, phase="closed_precreate" if closed else "terminal_unknown",
                       cleanup_outcome="closed" if closed else ("unknown" if code in ("clock_regressed", "native_release_unobserved") else "partial"),
                       reason=record.state.reason if closed else (code or "cleanup_unresolved"))
    else:
        changes.update(operation_sequence=token.terminal, phase="subscribed" if code is None else "terminal_unknown", reason=code)
    _native_startup_commit(record, **changes)
    token.outcome = (token.terminal, code, now, connection, native)
    return _native_startup_snapshot_copy(record)


def authenticate_startup_bus(owner: NativeStartupOwner, *, expected_sequence: int) -> NativeStartupSnapshot:
    _native_startup_validate_owner(owner)
    _native_startup_integer(expected_sequence)
    record = _native_startup_enter(owner, effect=True)
    try:
        if expected_sequence != record.state.operation_sequence:
            _native_startup_fail("stale_sequence")
        if record.state.phase != "prepared":
            _native_startup_fail("already_closed" if record.state.phase in ("closed_precreate", "released") else "invalid_input")
        now = _native_startup_clock(record)
        _native_startup_validate_record(record, effect=True)
        if now >= record.seal.original_deadline:
            _native_startup_fail("deadline_expired")
        intent, terminal = _native_startup_sequence_pair(expected_sequence)
        token = _NativeStartupAuthToken(record, intent, terminal, record.seal.original_deadline)
        record.operation = token
        _native_startup_commit(record, phase="authenticating", operation_sequence=intent, last_observed_now=now, authentication_attempted=True)
        code = None
        try:
            _native_startup_authenticate(record, token)
        except BaseException as caught:
            code = _native_startup_error_code(caught, "native_call_failed")
        return _native_startup_finish(record, token, code)
    finally:
        _native_startup_leave(record)


def _native_startup_close_connection(record: _NativeStartupRecord, token: _NativeStartupCleanupToken) -> None:
    resources = record.resources
    if resources.transfer in ("attempted_unknown", "detach_entered", "release_entered", "release_unobserved"):
        _native_startup_fail("cleanup_unresolved")
    if resources.bus is not None and not resources.bus_closed:
        pointer = _native_startup_pointer(resources.bus)
        _native_startup_call(record, token, "sd_bus_close", (pointer,))
        if not resources.bus_closed:
            _native_startup_fail("native_release_unobserved")
    if resources.fd_release_observed or resources.raw_fd is None:
        return
    _native_startup_boundary(record, token)
    if resources.transfer == "python_owned":
        if resources.socket.fileno() != resources.raw_fd:
            _native_startup_fail("native_resource_drift")
        resources.transfer = "release_entered"
        resources.socket.close()
    elif resources.transfer == "caller_owned" and (resources.bus is None or resources.bus_closed):
        resources.transfer = "release_entered"
        os.close(resources.raw_fd)
    else:
        _native_startup_fail("cleanup_unresolved")
    resources.fd_release_observed = True
    resources.transfer = "released"
    _native_startup_boundary(record, token)


def _native_startup_cleanup(record: _NativeStartupRecord, token: _NativeStartupCleanupToken) -> None:
    _native_startup_close_connection(record, token)
    resources = record.resources
    references = [reference for role in ("method_slot", "filter", "method_message", "process_message", "bus")
                  for reference in resources.references if reference.role == role]
    for reference in references:
        if reference.state == "released":
            continue
        if reference.state == "acquisition_pending" and reference.out.value is None:
            entered = any(reference in frame.outputs and frame.entered for frame in resources.frames)
            if not entered:
                reference.state = "released"
                continue
        if reference.state != "owned":
            _native_startup_fail("cleanup_unresolved")
        if reference.role != "bus" and resources.bus is not None and not resources.bus_closed:
            _native_startup_fail("cleanup_unresolved")
        if reference.role == "bus" and any(other.state != "released" for other in resources.references if other is not reference):
            _native_startup_fail("cleanup_unresolved")
        _native_startup_unref(record, token, reference)
    for entry in resources.directory_fds:
        _native_startup_close_directory(record, token, entry)
    if _native_startup_gone(record) != (True, True):
        _native_startup_fail("cleanup_unresolved")


def close_precreate_startup(owner: NativeStartupOwner, *, expected_sequence: int, expected_cleanup_sequence: int,
                             cleanup_deadline: float) -> NativeStartupSnapshot:
    _native_startup_validate_owner(owner)
    _native_startup_integer(expected_sequence)
    _native_startup_integer(expected_cleanup_sequence)
    _native_startup_float(cleanup_deadline)
    record = _native_startup_enter(owner, effect=True)
    try:
        if expected_sequence != record.state.operation_sequence or expected_cleanup_sequence != record.state.cleanup_sequence:
            _native_startup_fail("stale_sequence")
        if record.state.phase in ("closed_precreate", "released"):
            _native_startup_fail("already_closed")
        entry = _native_startup_clock(record)
        _native_startup_validate_record(record, effect=True)
        delta = cleanup_deadline - entry
        if not 0 < delta <= 2.0:
            _native_startup_fail("cleanup_deadline_invalid")
        if len(record.cleanup_attempts) >= 16:
            _native_startup_fail("cleanup_attempts_exhausted")
        intent, terminal = _native_startup_sequence_pair(expected_cleanup_sequence, cleanup=True)
        token = _NativeStartupCleanupToken(record, intent, terminal, cleanup_deadline)
        record.cleanup_operation = token
        record.cleanup_attempts.append(token)
        _native_startup_commit(record, cleanup_sequence=intent, cleanup_deadline=cleanup_deadline, cleanup_attempted=True,
                               cleanup_outcome="in_progress", phase="closing_precreate", last_observed_now=entry)
        code = None
        try:
            _native_startup_cleanup(record, token)
        except BaseException as caught:
            code = _native_startup_error_code(caught, "cleanup_unresolved")
        return _native_startup_finish(record, token, code)
    finally:
        _native_startup_leave(record)


def release_closed_native_startup(owner: NativeStartupOwner, *, expected_sequence: int, expected_cleanup_sequence: int) -> NativeStartupSnapshot:
    _native_startup_validate_owner(owner)
    _native_startup_integer(expected_sequence)
    _native_startup_integer(expected_cleanup_sequence)
    record = _native_startup_enter(owner, effect=True)
    try:
        if expected_sequence != record.state.operation_sequence or expected_cleanup_sequence != record.state.cleanup_sequence:
            _native_startup_fail("stale_sequence")
        if record.state.phase != "closed_precreate" or _native_startup_gone(record) != (True, True):
            _native_startup_fail("cleanup_unresolved")
        _native_startup_commit(record, phase="released")
        result = _native_startup_snapshot_copy(record)
        if _NATIVE_STARTUP_RECORDS.get(owner.owner_id) is not record:
            _native_startup_fail("invalid_owner")
        del _NATIVE_STARTUP_RECORDS[owner.owner_id]
        return result
    finally:
        _native_startup_leave(record)


if __name__ == "__main__":
    main()
