"""Process-local serving admission; no model, database or caller integration.

Readers and reserved children pin one caller-supplied selection key. Exclusive
activation drains those registrations. This gate does not validate a database
generation or apply a selection, and is not a cross-process publication lock.
"""

from __future__ import annotations

import math
import os
import threading
import time
import weakref
from collections import deque
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass

SelectionKey = tuple[str, ...]
_CANCEL_POLL_SECONDS = 0.05


class ServingLeaseError(RuntimeError):
    """Serving admission or an opaque token's lifecycle is invalid."""


class ServingLeaseTimeout(TimeoutError):
    """The serving admission deadline expired."""


class ServingLeaseCancelled(ServingLeaseError):
    """Serving admission was cancelled before work could start."""


@dataclass(frozen=True)
class _Control:
    deadline: float | None
    events: tuple[threading.Event, ...]


def _control(timeout: float | None, deadline: float | None, cancelled: threading.Event | None) -> _Control:
    for value in (timeout, deadline):
        try:
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ServingLeaseError("Serving admission requires a finite time bound")
        except OverflowError:
            raise ServingLeaseError("Serving admission requires a finite time bound") from None
    if timeout is not None and timeout < 0:
        raise ServingLeaseError("Serving admission timeout must be nonnegative")
    if cancelled is not None and type(cancelled) is not threading.Event:
        raise ServingLeaseError("Serving cancellation requires a threading Event")
    if timeout is not None:
        relative = time.monotonic() + timeout
        deadline = relative if deadline is None else min(relative, deadline)
    return _Control(deadline, () if cancelled is None else (cancelled,))


def _combine(parent: _Control, child: _Control) -> _Control:
    bounds = tuple(value for value in (parent.deadline, child.deadline) if value is not None)
    events = parent.events + tuple(event for event in child.events if all(event is not old for old in parent.events))
    return _Control(min(bounds) if bounds else None, events)


def _check(control: _Control) -> None:
    if any(threading.Event.is_set(event) for event in control.events):
        raise ServingLeaseCancelled("Serving admission was cancelled")
    if control.deadline is not None and time.monotonic() >= control.deadline:
        raise ServingLeaseTimeout("Serving admission deadline expired")


def _wait_seconds(control: _Control) -> float | None:
    seconds = None if control.deadline is None else max(0.0, control.deadline - time.monotonic())
    if control.events:
        seconds = _CANCEL_POLL_SECONDS if seconds is None else min(seconds, _CANCEL_POLL_SECONDS)
    return None if seconds is None else min(seconds, threading.TIMEOUT_MAX)


def _selection_key(value: SelectionKey) -> SelectionKey:
    # Exact built-in types exclude user equality/hash callbacks under the lock.
    if (type(value) is not tuple or not 1 <= len(value) <= 8
            or any(type(part) is not str or not 1 <= len(part) <= 512 for part in value)):
        raise ServingLeaseError("Serving selection requires a bounded immutable key")
    return value


class _OpaqueToken:
    __slots__ = ("_gate", "_closed", "__weakref__")

    def __new__(cls) -> _OpaqueToken:
        raise ServingLeaseError("Serving tokens are issued only by their gate")

    def __setattr__(self, name: str, value: object) -> None:
        raise ServingLeaseError("Serving tokens are opaque")

    def __repr__(self) -> str:
        return "<serving admission token>"


class OperationLease(_OpaqueToken):
    __slots__ = ()

    @property
    def selection_key(self) -> SelectionKey:
        return _issuer(self)._lease_key(self)

    def fork_child(self) -> ChildReservation:
        """Reserve one child now; close it if the task never enters join()."""
        return _issuer(self)._fork_child(self)

    def check_admission(self) -> None:
        """Recheck inherited controls before a caller's deferred body admission."""
        _issuer(self)._check_lease_admission(self)


class ChildReservation(_OpaqueToken):
    __slots__ = ()

    @contextmanager
    def join(
        self, *, timeout: float | None = None, deadline: float | None = None,
        cancelled: threading.Event | None = None,
    ) -> Iterator[OperationLease]:
        """Consume once on admission; an unadmitted reservation still needs close()."""
        gate = _issuer(self)
        with gate._join_child(self, _control(timeout, deadline, cancelled)) as lease:
            yield lease

    def close(self) -> None:
        """Release an unused reservation; never release a running joined child."""
        _issuer(self)._close_child(self)


def _issuer(token: _OpaqueToken) -> ServingGate:
    gate = getattr(token, "_gate", None)
    if type(token) not in (OperationLease, ChildReservation) or type(gate) is not ServingGate:
        raise ServingLeaseError("Invalid serving admission token")
    gate._check_process()
    return gate


@dataclass(frozen=True)
class _Reader:
    thread: threading.Thread
    key: SelectionKey
    control: _Control
    writer_owned: bool
    reservation: ChildReservation | None = None


@dataclass
class _Reserved:
    key: SelectionKey
    control: _Control
    joined: OperationLease | None = None


class ServingGate:
    """Concurrent readers, FIFO exclusive writers, and explicit pinned children.

    An explicit gate is invalid after fork. Module-level functions receive a new
    child-process gate through register_at_fork; inherited tokens remain invalid.
    """

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._condition = threading.Condition(threading.Lock())
        self._local = threading.local()
        self._readers: dict[OperationLease, _Reader] = {}
        self._children: dict[ChildReservation, _Reserved] = {}
        self._retired_children: weakref.WeakSet[ChildReservation] = weakref.WeakSet()
        self._writers: dict[object, threading.Thread] = {}
        self._waiting_writers: deque[object] = deque()

    def _check_process(self) -> None:
        if os.getpid() != self._pid:
            raise ServingLeaseError("Serving state cannot be inherited by another process")

    @contextmanager
    def _locked(self, control: _Control | None = None, *, blocking: bool = True) -> Iterator[None]:
        self._check_process()
        acquired = False
        try:
            while not acquired:
                if not blocking:
                    if control is not None:
                        _check(control)
                    acquired = self._condition.acquire(blocking=False)
                    if not acquired:
                        raise ServingLeaseError("Serving admission is busy")
                elif control is None:
                    acquired = self._condition.acquire()
                else:
                    _check(control)
                    seconds = _wait_seconds(control)
                    acquired = self._condition.acquire() if seconds is None else self._condition.acquire(timeout=seconds)
            self._check_process()
            if control is not None:
                _check(control)
            yield
        finally:
            if acquired:
                self._condition.release()

    def _token(self, kind: type[OperationLease] | type[ChildReservation]) -> OperationLease | ChildReservation:
        token = object.__new__(kind)
        object.__setattr__(token, "_gate", self)
        object.__setattr__(token, "_closed", False)
        return token

    def _thread_reader(self, thread: threading.Thread) -> _Reader | None:
        if thread is not threading.current_thread():
            raise ServingLeaseError("Serving thread state belongs to another thread")
        readers = getattr(self._local, "readers", ())
        return readers[-1] if readers else None

    def _writer_thread(self) -> threading.Thread | None:
        return next(iter(self._writers.values()), None)

    def current_thread_admitted(self) -> bool:
        """Inspect nesting without waiting behind another gate operation."""
        with self._locked(blocking=False):
            thread = threading.current_thread()
            return self._thread_reader(thread) is not None or self._writer_thread() is thread

    def _check_key(self, key: SelectionKey) -> None:
        current = next((reader.key for reader in self._readers.values()), None)
        if current is None:
            current = next((child.key for child in self._children.values()), None)
        if current is not None and current != key:
            raise ServingLeaseError("Serving selection differs from an admitted operation")

    def _owned_reader(self, lease: OperationLease) -> _Reader:
        if type(lease) is not OperationLease or getattr(lease, "_gate", None) is not self:
            raise ServingLeaseError("Invalid serving operation lease")
        reader = self._readers.get(lease)
        if reader is None or reader.thread is not threading.current_thread():
            raise ServingLeaseError("Serving operation lease is closed or belongs to another thread")
        return reader

    def _register_reader(self, lease: OperationLease, reader: _Reader) -> None:
        readers = (*getattr(self._local, "readers", ()), reader)
        self._readers[lease] = reader
        self._local.readers = readers

    def _lease_key(self, lease: OperationLease) -> SelectionKey:
        with self._locked():
            return self._owned_reader(lease).key

    def _check_lease_admission(self, lease: OperationLease) -> None:
        self._check_process()
        control = self._owned_reader(lease).control
        with self._locked(control):
            _check(self._owned_reader(lease).control)

    def _release_reader(self, lease: OperationLease) -> None:
        with self._locked():
            reader = self._readers.get(lease)
            if reader is not None:
                self._owned_reader(lease)
                del self._readers[lease]
                self._local.readers = tuple(item for item in self._local.readers if item is not reader)
                if reader.reservation is not None:
                    del self._children[reader.reservation]
                    self._retired_children.add(reader.reservation)
                    object.__setattr__(reader.reservation, "_closed", True)
            object.__setattr__(lease, "_closed", True)
            self._condition.notify_all()

    @contextmanager
    def operation_lease(
        self, selection_key: SelectionKey, *, timeout: float | None = None,
        deadline: float | None = None, cancelled: threading.Event | None = None,
        blocking: bool = True,
    ) -> Iterator[OperationLease]:
        if type(blocking) is not bool:
            raise ServingLeaseError("Serving admission blocking must be boolean")
        key, control = _selection_key(selection_key), _control(timeout, deadline, cancelled)
        lease = self._token(OperationLease)
        thread = threading.current_thread()
        self._check_process()
        parent = self._thread_reader(thread)
        if parent is not None:
            control = _combine(parent.control, control)
        entered = False
        try:
            with self._locked(control, blocking=blocking):
                entered = True
                while True:
                    _check(control)
                    self._check_key(key)
                    writer = self._writer_thread()
                    if writer is thread or (writer is None and (parent is not None or not self._waiting_writers)):
                        self._register_reader(lease, _Reader(thread, key, control, writer is thread or bool(parent and parent.writer_owned)))
                        break
                    if not blocking:
                        raise ServingLeaseError("Serving admission is busy")
                    self._condition.wait(_wait_seconds(control))
            yield lease
        finally:
            if entered:
                self._release_reader(lease)

    def _fork_child(self, lease: OperationLease) -> ChildReservation:
        self._check_process()
        # The owner's record is immutable and cannot be released by another
        # thread. Revalidate membership after acquiring the condition as well.
        control = self._owned_reader(lease).control
        with self._locked(control):
            reader = self._owned_reader(lease)
            _check(reader.control)
            if reader.writer_owned:
                raise ServingLeaseError("Writer-owned reads cannot grant cross-thread admission")
            child = self._token(ChildReservation)
            self._children[child] = _Reserved(reader.key, reader.control)
            return child

    def _close_child(self, child: ChildReservation) -> None:
        with self._locked():
            if type(child) is not ChildReservation or getattr(child, "_gate", None) is not self:
                raise ServingLeaseError("Invalid serving child reservation")
            reserved = self._children.get(child)
            if reserved is None:
                if child not in self._retired_children:
                    raise ServingLeaseError("Invalid serving child reservation")
                return
            if reserved.joined is not None:
                raise ServingLeaseError("A joined child must release its own operation lease")
            del self._children[child]
            self._retired_children.add(child)
            object.__setattr__(child, "_closed", True)
            self._condition.notify_all()

    @contextmanager
    def _join_child(self, child: ChildReservation, control: _Control) -> Iterator[OperationLease]:
        self._check_process()
        if type(child) is not ChildReservation or getattr(child, "_gate", None) is not self:
            raise ServingLeaseError("Invalid serving child reservation")
        # Snapshot immutable inherited bounds before mutex admission. The
        # reservation is revalidated under the condition before consumption.
        snapshot = self._children.get(child)
        if snapshot is None:
            raise ServingLeaseError("Serving child reservation is closed or already joined")
        control = _combine(snapshot.control, control)
        thread = threading.current_thread()
        parent = self._thread_reader(thread)
        if parent is not None:
            control = _combine(parent.control, control)
        lease = self._token(OperationLease)
        entered = False
        try:
            with self._locked(control):
                entered = True
                reserved = self._children.get(child)
                if reserved is None or reserved.joined is not None:
                    raise ServingLeaseError("Serving child reservation is closed or already joined")
                control = _combine(reserved.control, control)
                _check(control)
                self._check_key(reserved.key)
                if self._writers:
                    raise ServingLeaseError("A reserved child cannot inherit writer ownership")
                self._register_reader(lease, _Reader(thread, reserved.key, control, False, child))
                reserved.joined = lease
            yield lease
        finally:
            if entered:
                self._release_reader(lease)

    @contextmanager
    def exclusive_activation(
        self, *, timeout: float | None = None, deadline: float | None = None,
        cancelled: threading.Event | None = None,
    ) -> Iterator[None]:
        control = _control(timeout, deadline, cancelled)
        ticket = object()
        thread = threading.current_thread()
        entered = False
        try:
            with self._locked(control):
                entered = True
                if self._writer_thread() is thread:
                    self._writers[ticket] = thread
                else:
                    if self._thread_reader(thread) is not None:
                        raise ServingLeaseError("A serving read lease cannot be upgraded to activation")
                    self._waiting_writers.append(ticket)
                    while True:
                        _check(control)
                        if (not self._writers and not self._readers and not self._children
                                and self._waiting_writers[0] is ticket):
                            self._waiting_writers.popleft()
                            self._writers[ticket] = thread
                            break
                        self._condition.wait(_wait_seconds(control))
            yield
        finally:
            if entered:
                with self._locked():
                    owner = self._writers.get(ticket)
                    if owner is not None and owner is not threading.current_thread():
                        raise ServingLeaseError("Activation lease belongs to another thread")
                    self._writers.pop(ticket, None)
                    if ticket in self._waiting_writers:
                        self._waiting_writers.remove(ticket)
                    self._condition.notify_all()


_DEFAULT_GATE = ServingGate()


def _after_fork() -> None:
    global _DEFAULT_GATE
    _DEFAULT_GATE = ServingGate()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def operation_lease(
    selection_key: SelectionKey, *, timeout: float | None = None, deadline: float | None = None,
    cancelled: threading.Event | None = None, blocking: bool = True,
) -> AbstractContextManager[OperationLease]:
    """Acquire the process-wide serving boundary for one logical operation."""
    return _DEFAULT_GATE.operation_lease(selection_key, timeout=timeout, deadline=deadline, cancelled=cancelled, blocking=blocking)


def exclusive_activation(
    *, timeout: float | None = None, deadline: float | None = None,
    cancelled: threading.Event | None = None,
) -> AbstractContextManager[None]:
    """Drain process-wide operations before a caller's activation work."""
    return _DEFAULT_GATE.exclusive_activation(timeout=timeout, deadline=deadline, cancelled=cancelled)


def current_thread_admitted() -> bool:
    """Report a reader/exclusive scope on this thread, or refuse a busy gate."""
    return _DEFAULT_GATE.current_thread_admitted()
