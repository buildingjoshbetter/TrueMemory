"""Minimal stdlib-only owned Linux cgroup readback.

This file is intended to be addressed by absolute path. Phase B launch activation
is held; importing it neither enrolls a process nor changes an OS limit.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat


_MAX_BYTES = 2**63 - 1
_READ_LIMIT = 65536


class EnforcementUnavailable(RuntimeError):
    """A requested owned controller scope cannot be verified."""


@dataclass(frozen=True)
class LinuxScopeExpectation:
    scope: str
    pid: int
    uid: int
    device: int
    inode: int
    memory_max: int
    swap_max: int

    def __post_init__(self) -> None:
        if type(self.scope) is not str or not self.scope or not Path(self.scope).is_absolute():
            raise ValueError("scope must be an explicit absolute path")
        for name in ("pid", "uid", "device", "inode", "memory_max", "swap_max"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= _MAX_BYTES:
                raise ValueError(f"{name} must be a nonnegative signed64 integer")
        if self.pid <= 0 or self.memory_max <= 0:
            raise ValueError("pid and memory_max must be positive")


@dataclass(frozen=True)
class LinuxScopeReadback:
    expectation: LinuxScopeExpectation
    members: tuple[int, ...]
    memory_current: int
    memory_events: tuple[tuple[str, int], ...]
    metric: str = "controller_file_readback_unverified"
    physical_hard_cap: bool = False
    lifetime_enforcement_proven: bool = False


def _read_owned_file(directory: int, name: str) -> str:
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_NONBLOCK", 0))
    descriptor = os.open(name, flags, dir_fd=directory)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise EnforcementUnavailable(f"{name} is not a regular controller file")
        chunks = bytearray()
        while len(chunks) <= _READ_LIMIT:
            chunk = os.read(descriptor, min(4096, _READ_LIMIT + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) > _READ_LIMIT:
            raise EnforcementUnavailable(f"{name} exceeds the readback bound")
        return bytes(chunks).decode("ascii").strip()
    finally:
        os.close(descriptor)


def _integer(text: str, name: str) -> int:
    if not text.isascii() or not text.isdecimal():
        raise EnforcementUnavailable(f"{name} is not a finite integer")
    value = int(text)
    if value > _MAX_BYTES:
        raise EnforcementUnavailable(f"{name} exceeds signed64")
    return value


def inspect_owned_linux_scope(expectation: LinuxScopeExpectation) -> LinuxScopeReadback:
    """Read an already prepared owned scope, without creating or modifying it.

    This is a point-in-time ownership/readback check, not an installed launch or
    kernel allocation witness. Filesystem/controller authenticity must be proved
    by a separate root-owned witness. The separate swap domain is not a unified
    physical process footprint.
    """
    if type(expectation) is not LinuxScopeExpectation:
        raise TypeError("Expected an exact immutable LinuxScopeExpectation")
    LinuxScopeExpectation.__post_init__(expectation)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise EnforcementUnavailable("No no-follow directory ownership primitive")
    directory = os.open(expectation.scope, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                        | getattr(os, "O_CLOEXEC", 0))
    try:
        metadata = os.fstat(directory)
        if (metadata.st_uid != expectation.uid or metadata.st_dev != expectation.device
                or metadata.st_ino != expectation.inode):
            raise EnforcementUnavailable("Owned controller directory identity changed")
        maximum = _integer(_read_owned_file(directory, "memory.max"), "memory.max")
        swap = _integer(_read_owned_file(directory, "memory.swap.max"), "memory.swap.max")
        if maximum != expectation.memory_max or swap != expectation.swap_max:
            raise EnforcementUnavailable("Finite controller limits differ from the owned receipt")
        raw_members = _read_owned_file(directory, "cgroup.procs").splitlines()
        members = tuple(_integer(value, "cgroup.procs") for value in raw_members)
        if members != (expectation.pid,):
            raise EnforcementUnavailable("Owned scope contains a missing or unrelated process")
        current = _integer(_read_owned_file(directory, "memory.current"), "memory.current")
        events = []
        for line in _read_owned_file(directory, "memory.events").splitlines():
            parts = line.split()
            if len(parts) != 2 or not parts[0].isascii():
                raise EnforcementUnavailable("Malformed controller event readback")
            events.append((parts[0], _integer(parts[1], "memory.events")))
        return LinuxScopeReadback(expectation, members, current, tuple(events))
    except (OSError, UnicodeError, ValueError) as error:
        raise EnforcementUnavailable("Owned Linux scope readback is unavailable") from error
    finally:
        os.close(directory)


def main() -> None:
    raise SystemExit(
        "Phase B launch activation is held pending a root-approved owned kernel witness"
    )


if __name__ == "__main__":
    main()
