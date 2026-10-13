"""ROOT-run Stage A supervisor. Importing this module never starts native work.

This instrumented source-module harness proves tiny filesystem/lock behavior.
It does not certify installed packages, host dispatch, models or a physical cap.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
import re
import select
import selectors
import signal
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

RUN_ROOT = Path("/Users/j/dev/truememory-maintenance-v1/.work/native-mac-v1/run-0001")
SOURCE_ROOT = Path("/Users/j/dev/truememory-maintenance-v1/.work/worktrees/mac-native-acceptance-v1")
RUNTIME_PACKET = Path("/Users/j/dev/truememory-maintenance-v1/.work/reports/mac-native-runtime-pins-v1.json")
RUNTIME_PACKET_SHA = "8db57e3f700aebb3adeb383c719a4bc5e2525917f4c017c853ee4694be1ee1e4"
PROOF_SCOPE = "instrumented_unmodified_source_native_filesystem_lock_capture"
SOURCE_FILES = (
    "truememory/_platform.py", "truememory/ingest/hooks/_shared.py",
    "truememory/ingest/completion.py", "truememory/ingest/transcript.py",
    "truememory/ingest/provenance.py",
)
CASES = frozenset({"capture-a1", "capture-a2", "capture-revision", "negative-a3", "invalid-state", "flock-a4", "parser-a5"})
LIMITS = {"wall_seconds": 5, "termination_seconds": 1, "cpu_soft": 2, "cpu_hard": 3,
          "rss_abort_bytes": 67108864, "core_bytes": 0, "nofile": 64,
          "fsize_bytes": 1048576, "input_bytes": 65536, "output_bytes": 262144,
          "events": 32, "frame_bytes": 2048, "ack_ms": 100, "threads": 1}
FREE_MIN = 268435456
MAX_INT = (1 << 63) - 1
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_PENDING = re.compile(r"state-[0-9a-f]{32}\.pending\Z")
_ATTEMPT = re.compile(r"v(?:[1-9]|[1-9][0-9])\Z")
_announcement_bytes = 0


class SafetyError(ValueError):
    pass


def integer(value: object, *, minimum: int = 0, maximum: int = MAX_INT) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SafetyError("invalid_integer")
    return value


def exact_keys(value: object, keys: set[str]) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise SafetyError("invalid_fields")
    return value


def encoded(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_text(value: object) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise SafetyError("invalid_hash")
    return value


def canonical_path(value: object, root: Path) -> Path:
    if type(value) is not str or not value or "\0" in value or len(value.encode("utf-8")) > 1024:
        raise SafetyError("invalid_path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise SafetyError("noncanonical_path")
    if path != root and not path.is_relative_to(root):
        raise SafetyError("path_escape")
    return path


def owned_path(path: Path, *, directory: bool = False) -> os.stat_result:
    if path.resolve() != path or any(item.is_symlink() for item in (path, *path.parents)):
        raise SafetyError("path_link")
    info = path.lstat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise SafetyError("path_not_private_owned")
    if directory:
        if not stat.S_ISDIR(info.st_mode):
            raise SafetyError("path_not_directory")
    elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SafetyError("path_not_regular_single_link")
    return info


def read_bounded(path: Path, maximum: int, *, private: bool = True) -> bytes:
    if private:
        owned_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise SafetyError("read_limit_or_type")
        if private and (before.st_uid != os.getuid() or before.st_mode & 0o077 or before.st_nlink != 1):
            raise SafetyError("read_not_private")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        data = b"".join(chunks)
        if file_identity(before) != file_identity(after) or len(data) > maximum or len(data) != after.st_size:
            raise SafetyError("read_changed_or_limit")
        return data
    finally:
        os.close(fd)


def file_identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


@dataclass(frozen=True)
class Sensor:
    monotonic_ns: int
    pressure: int
    free_bytes: int
    swap_used_bytes: int


def valid_sensor(value: object) -> Sensor:
    row = exact_keys(value, {"monotonic_ns", "pressure", "free_bytes", "swap_used_bytes"})
    integer(row["monotonic_ns"], minimum=1)
    integer(row["pressure"], minimum=1, maximum=4)
    integer(row["free_bytes"])
    integer(row["swap_used_bytes"])
    if row["pressure"] not in (1, 2, 4):
        raise SafetyError("unknown_pressure_abi")
    return Sensor(**row)


def admission(samples: list[Sensor], now_ns: int) -> None:
    integer(now_ns, minimum=1)
    if len(samples) < 3:
        raise SafetyError("sensor_count")
    previous = None
    for sample in samples:
        valid_sensor(asdict(sample))
        if sample.pressure != 1 or sample.free_bytes < FREE_MIN:
            raise SafetyError("pressure_or_headroom")
        if previous is not None and (sample.monotonic_ns <= previous.monotonic_ns or sample.swap_used_bytes > previous.swap_used_bytes):
            raise SafetyError("unordered_or_growing_swap")
        previous = sample
    if samples[-1].monotonic_ns - samples[0].monotonic_ns < 2_000_000_000:
        raise SafetyError("sensor_window")
    if not 0 <= now_ns - samples[-1].monotonic_ns <= 500_000_000:
        raise SafetyError("sensor_stale")


@dataclass(frozen=True)
class Identity:
    pid: int
    uid: int
    birth_ns: int
    pgid: int


def same_owned_identity(expected: Identity, actual: Identity, worker_uid: int) -> bool:
    try:
        for row in (expected, actual):
            integer(row.pid, minimum=1)
            integer(row.uid)
            integer(row.birth_ns, minimum=1)
            integer(row.pgid, minimum=1)
        integer(worker_uid)
    except SafetyError:
        return False
    return expected == actual and actual.uid == worker_uid and actual.pgid == actual.pid


def account_directory(case: dict) -> Path:
    key = digest(encoded(["native-acceptance-v1", case["database"]]))
    return RUN_ROOT / "completion" / key


def capture_metadata(case: dict) -> dict:
    return {"version": 1, "user_id": "native-acceptance-v1", "database": case["database"],
            "host": "claude", "session_id": case["session"], "source": case["source"],
            "source_suffix": ".jsonl", "file_identity": case["source_identity"],
            "byte_count": case["source_bytes"], "sha256": case["source_sha256"]}


def validate_case(value: object) -> dict:
    row = exact_keys(value, {"id", "source", "source_sha256", "source_bytes", "source_identity", "session", "database", "previous"})
    if type(row["id"]) is not str or row["id"] not in CASES:
        raise SafetyError("unsupported_case")
    base = RUN_ROOT / "home/.claude/projects/native-acceptance-v1"
    filename = {"capture-revision": "capture-a2", "invalid-state": "negative-a3"}.get(row["id"], row["id"])
    if row["id"] != "flock-a4":
        if canonical_path(row["source"], RUN_ROOT) != base / (filename + ".jsonl"):
            raise SafetyError("unexpected_fixture")
        hash_text(row["source_sha256"])
        integer(row["source_bytes"], minimum=1, maximum=4096)
        if type(row["source_identity"]) is not list or len(row["source_identity"]) != 5:
            raise SafetyError("fixture_identity")
        for number in row["source_identity"]:
            integer(number)
        if row["source_identity"][2] != row["source_bytes"]:
            raise SafetyError("fixture_size_identity")
    elif any(row[name] is not None for name in ("source", "source_sha256", "source_bytes", "source_identity")):
        raise SafetyError("unexpected_lock_fixture")
    if row["session"] != "synthetic-native-" + filename:
        raise SafetyError("unexpected_session")
    database_case = "capture-a2" if row["id"] == "capture-revision" else "negative-a3" if row["id"] == "invalid-state" else row["id"]
    if row["database"] != str(RUN_ROOT / (database_case + "-v1.db")):
        raise SafetyError("unexpected_database_locator")
    if row["id"] == "capture-revision":
        old = exact_keys(row["previous"], {"key", "sha256", "byte_count"})
        hash_text(old["key"])
        hash_text(old["sha256"])
        integer(old["byte_count"], minimum=1, maximum=4096)
        if old["key"] == digest(encoded(capture_metadata(row))):
            raise SafetyError("revision_not_new_evidence")
    elif row["previous"] is not None:
        raise SafetyError("unexpected_previous")
    return row


def validate_manifest(value: object) -> dict:
    row = exact_keys(value, {"version", "stage", "synthetic", "root_execution_go", "run_root", "home", "source_root", "source_commit", "sources", "supervisor_sha256", "bootstrap_sha256", "limits", "cases"})
    if type(row["version"]) is not int or row["version"] != 1 or row["stage"] != "A" or row["synthetic"] is not True or row["root_execution_go"] is not True:
        raise SafetyError("manifest_authority_or_version")
    if row["run_root"] != str(RUN_ROOT) or row["home"] != str(RUN_ROOT / "home") or row["source_root"] != str(SOURCE_ROOT):
        raise SafetyError("unexpected_root")
    if type(row["source_commit"]) is not str or re.fullmatch(r"[0-9a-f]{40}", row["source_commit"]) is None:
        raise SafetyError("source_commit")
    sources = exact_keys(row["sources"], set(SOURCE_FILES))
    for value in sources.values():
        hash_text(value)
    hash_text(row["supervisor_sha256"])
    hash_text(row["bootstrap_sha256"])
    if type(row["limits"]) is not dict or row["limits"] != LIMITS or any(type(v) is not int for v in row["limits"].values()):
        raise SafetyError("limit_change")
    if type(row["cases"]) is not list or not 1 <= len(row["cases"]) <= len(CASES):
        raise SafetyError("case_count")
    cases = [validate_case(case) for case in row["cases"]]
    if len({case["id"] for case in cases}) != len(cases):
        raise SafetyError("duplicate_case")
    return row


def validate_ready(value: object, pid: int) -> dict:
    row = exact_keys(value, {"type", "version", "pid", "limits", "audit_installed"})
    integer(pid, minimum=1)
    if row["type"] != "ready" or type(row["version"]) is not int or row["version"] != 1 or type(row["pid"]) is not int or row["pid"] != pid or row["audit_installed"] is not True:
        raise SafetyError("limits_ready_mismatch")
    expected = {"cpu": [2, 3], "core": [0, 0], "nofile": [64, 64], "fsize": [1048576, 1048576]}
    limits = exact_keys(row["limits"], set(expected))
    for name, pair in limits.items():
        if type(pair) is not list or len(pair) != 2 or any(type(number) is not int for number in pair) or pair != expected[name]:
            raise SafetyError("limits_ready_mismatch")
    return row


def validate_result(value: object, case: dict, sources: dict) -> dict:
    row = exact_keys(value, {"type", "version", "case", "passed", "proof_scope", "import_mode", "sources", "details"})
    if row["type"] != "result" or type(row["version"]) is not int or row["version"] != 1 or row["case"] != case["id"] or row["passed"] is not True or row["proof_scope"] != PROOF_SCOPE or row["import_mode"] != "isolated_source_module" or row["sources"] != sources or type(row["details"]) is not dict:
        raise SafetyError("result_scope_mismatch")
    return row


def validate_attempt_version(value: object) -> str:
    if type(value) is not str or _ATTEMPT.fullmatch(value) is None:
        raise SafetyError("invalid_attempt_version")
    return value


def evidence_paths(manifest_path: Path) -> tuple[Path, str]:
    name = manifest_path.name
    if manifest_path.parent != RUN_ROOT or not name.startswith("manifest-") or not name.endswith(".json"):
        raise SafetyError("unexpected_manifest_path")
    version = validate_attempt_version(name[len("manifest-"):-len(".json")])
    return RUN_ROOT / ("results-" + version + ".json"), version


class MutationPolicy:
    def __init__(self, case: dict) -> None:
        self.case = case
        self.account = account_directory(case)
        self.directory_paths = {self.account, self.account.parent, *(self.account / name for name in ("captures", "receipts", "logs"))}
        self.fixed = {self.account / "state.lock": "Completion account lock", self.account / "state.json": "Atomic completion state"}
        if case["id"] not in {"flock-a4", "parser-a5"}:
            self.fixed[self.account / "captures" / (digest(encoded(capture_metadata(case))) + ".jsonl")] = "Immutable synthetic source capture"
        if case["id"] in {"negative-a3", "parser-a5"}:
            self.directory_paths = set()
            self.fixed = {}
        elif case["id"] == "invalid-state":
            self.fixed = {self.account / "state.lock": "Completion account lock"}
        elif case["id"] == "flock-a4":
            self.directory_paths = set()
            self.fixed = {RUN_ROOT / "locks/contention-v1.lock": "Native lock contention witness"}
        self.pending: set[Path] = set()
        self.sequence = 0

    def decide(self, value: object) -> dict:
        row = exact_keys(value, {"type", "version", "seq", "event", "path", "source", "flags"})
        if row["type"] != "mutation" or type(row["version"]) is not int or row["version"] != 1:
            raise SafetyError("mutation_shape")
        seq = integer(row["seq"], minimum=1, maximum=LIMITS["events"])
        if seq != self.sequence + 1:
            raise SafetyError("mutation_sequence")
        path = canonical_path(row["path"], RUN_ROOT)
        flags = integer(row["flags"], maximum=(1 << 31) - 1)
        purpose = self.fixed.get(path)
        if row["event"] == "os.mkdir":
            if row["source"] is not None or path not in self.directory_paths or flags != 0o700:
                raise SafetyError("unplanned_directory")
            purpose = "Private completion directory"
        elif row["event"] == "open":
            if row["source"] is not None or flags & os.O_TRUNC:
                raise SafetyError("unplanned_open")
            access = flags & os.O_ACCMODE
            extras = flags & ~(os.O_ACCMODE | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
            if extras or access not in (os.O_WRONLY, os.O_RDWR) or not flags & os.O_CREAT:
                raise SafetyError("unplanned_open_flags")
            if path.parent == self.account and _PENDING.fullmatch(path.name):
                if self.case["id"] not in {"capture-a1", "capture-a2", "capture-revision"} or access != os.O_WRONLY or not flags & os.O_EXCL or len(self.pending | {path}) > 1:
                    raise SafetyError("pending_slot_limit")
                self.pending.add(path)
                purpose = "Atomic completion-state temporary; failed evidence retained"
            elif path not in self.fixed or path.name == "state.json":
                raise SafetyError("unplanned_file")
            elif path.name == "state.lock" or path == RUN_ROOT / "locks/contention-v1.lock":
                if access != os.O_RDWR or flags & os.O_EXCL:
                    raise SafetyError("lock_flags")
            elif access != os.O_WRONLY or not flags & os.O_EXCL:
                raise SafetyError("capture_flags")
        elif row["event"] == "os.rename":
            source = canonical_path(row["source"], RUN_ROOT)
            if flags != 0 or path != self.account / "state.json" or source not in self.pending:
                raise SafetyError("unplanned_replace")
            purpose = "Publish current-run completion state from its approved temporary"
        else:
            raise SafetyError("unsupported_mutation")
        self.sequence = seq
        return {"type": "grant", "version": 1, "seq": seq, "path": str(path), "purpose": purpose}


def mutation_target(grant: dict, request: dict) -> None:
    path = Path(grant["path"])
    owned_path(path.parent, directory=True)
    if path.exists() or path.is_symlink():
        owned_path(path, directory=request["event"] == "os.mkdir")
    if request["event"] == "os.rename":
        owned_path(Path(request["source"]))


class PressureReader:
    def __init__(self) -> None:
        import ctypes
        self.ctypes = ctypes
        self.library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        self.read = self.library.sysctlbyname
        self.read.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
        self.read.restype = ctypes.c_int

    def pressure(self) -> int:
        c = self.ctypes
        output = c.c_uint32()
        size = c.c_size_t(c.sizeof(output))
        if self.read(b"kern.memorystatus_vm_pressure_level", c.byref(output), c.byref(size), None, 0) != 0 or size.value != 4:
            raise SafetyError("pressure_unavailable")
        if output.value not in (1, 2, 4):
            raise SafetyError("pressure_abi_unknown")
        return output.value


class PinnedPythonLoader(importlib.abc.Loader):
    def __init__(self, path: Path, checksum: str) -> None:
        self.path, self.checksum = path, checksum

    def create_module(self, spec: object) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        data = read_bounded(self.path, 262144, private=False)
        if digest(data) != self.checksum:
            raise SafetyError("dependency_hash_changed")
        exec(compile(data, str(self.path), "exec", dont_inherit=True), module.__dict__)


class PinnedPsutilFinder(importlib.abc.MetaPathFinder):
    def __init__(self, packet: dict) -> None:
        self.root = Path(packet["dependency_root"])
        self.files = packet["dependency_files"]

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> object:
        if fullname != "psutil" and not fullname.startswith("psutil."):
            return None
        relative = fullname.split(".")[1:]
        candidate = self.root.joinpath(*relative) if relative else self.root
        source = candidate / "__init__.py" if not relative else candidate.with_suffix(".py")
        entry = self.files.get(str(source))
        if entry is not None:
            return importlib.util.spec_from_file_location(fullname, source, loader=PinnedPythonLoader(source, entry["sha256"]), submodule_search_locations=[str(self.root)] if not relative else None)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            extension = Path(str(candidate) + suffix)
            entry = self.files.get(str(extension))
            if entry is not None:
                if digest(read_bounded(extension, 262144, private=False)) != entry["sha256"]:
                    raise SafetyError("dependency_extension_changed")
                return importlib.util.spec_from_file_location(fullname, extension)
        raise SafetyError("unpinned_dependency_module")


def pinned_runtime() -> ModuleType:
    raw = read_bounded(RUNTIME_PACKET, 65536, private=False)
    if digest(raw) != RUNTIME_PACKET_SHA:
        raise SafetyError("runtime_packet_changed")
    packet = json.loads(raw)
    if Path(sys.executable).resolve() != Path(packet["python_realpath"]) or sys.version_info[:3] != (3, 12, 12):
        raise SafetyError("interpreter_mismatch")
    if digest(read_bounded(Path(packet["python_realpath"]), 1048576, private=False)) != packet["python_executable_sha256"]:
        raise SafetyError("interpreter_hash_changed")
    if "psutil" in sys.modules:
        raise SafetyError("dependency_preloaded")
    for name, entry in packet["dependency_files"].items():
        if digest(read_bounded(Path(name), 262144, private=False)) != entry["sha256"]:
            raise SafetyError("dependency_hash_changed")
    sys.meta_path.insert(0, PinnedPsutilFinder(packet))
    import psutil
    if psutil.__version__ != packet["psutil_version"]:
        raise SafetyError("dependency_version_changed")
    return psutil


def sample(psutil: ModuleType, pressure: PressureReader) -> Sensor:
    started = time.monotonic_ns()
    row = Sensor(started, pressure.pressure(), psutil.virtual_memory().free, psutil.swap_memory().used)
    valid_sensor(asdict(row))
    if time.monotonic_ns() - started > 500_000_000:
        raise SafetyError("sensor_call_stale")
    return row


def observe_identity(psutil: ModuleType, pid: int) -> Identity:
    process = psutil.Process(pid)
    born = process.create_time()
    try:
        valid = type(born) in (int, float) and math.isfinite(born) and born > 0
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise SafetyError("birth_unavailable")
    return Identity(pid, process.uids().real, int(born * 1_000_000_000), os.getpgid(pid))


def announce(path: Path, purpose: str) -> None:
    global _announcement_bytes
    line = f"Create {path}: {purpose}\n"
    _announcement_bytes += len(line.encode("utf-8"))
    if _announcement_bytes > 65536:
        raise SafetyError("announcement_byte_limit")
    sys.stdout.write(line)
    sys.stdout.flush()


def new_log(path: Path, purpose: str) -> int:
    owned_path(path.parent, directory=True)
    announce(path, purpose)
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)


def write_all(fd: int, data: bytes) -> None:
    while data:
        count = os.write(fd, data)
        if count <= 0:
            raise SafetyError("write_no_progress")
        data = data[count:]


def write_pipe(fd: int, data: bytes, deadline: float) -> None:
    if len(data) > LIMITS["input_bytes"]:
        raise SafetyError("pipe_frame_input_limit")
    while data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SafetyError("pipe_write_deadline")
        try:
            count = os.write(fd, data)
        except BlockingIOError:
            select.select([], [fd], [], min(remaining, 0.01))
            continue
        if count <= 0:
            raise SafetyError("pipe_write_no_progress")
        data = data[count:]


def group_gone(pgid: int) -> bool:
    integer(pgid, minimum=1)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def member_identity_matches(expected: Identity, actual: Identity, group: int, worker_uid: int) -> bool:
    try:
        for row in (expected, actual):
            integer(row.pid, minimum=1)
            integer(row.uid)
            integer(row.birth_ns, minimum=1)
            integer(row.pgid, minimum=1)
        integer(group, minimum=1)
        integer(worker_uid)
    except SafetyError:
        return False
    return expected == actual and actual.uid == worker_uid and actual.pgid == group


def terminate_owned(child: subprocess.Popen, expected: Identity | None, members: list[Identity], psutil: ModuleType, result: dict) -> None:
    deadline = time.monotonic() + LIMITS["termination_seconds"]
    for sig, interval in ((signal.SIGTERM, 0.25), (signal.SIGKILL, 0.75)):
        if time.monotonic() >= deadline:
            raise SafetyError("termination_deadline")
        if child.poll() is None:
            if expected is None:
                # Popen still owns an unreaped direct child; its PID cannot be reused.
                child.send_signal(sig)
                result["termination_authority"] = "unreaped_direct_popen_child"
            elif same_owned_identity(expected, observe_identity(psutil, child.pid), os.getuid()):
                if result["descendant_containment"] == "unproven":
                    os.killpg(expected.pgid, sig)
                    result["termination_authority"] = "fresh_owned_leader_group"
                else:
                    child.send_signal(sig)
                    result["termination_authority"] = "fresh_owned_direct_child"
            else:
                raise SafetyError("termination_identity_changed")
        if sig == signal.SIGKILL and members and expected is not None:
            for member in members:
                if time.monotonic() >= deadline:
                    raise SafetyError("member_termination_deadline")
                try:
                    actual = observe_identity(psutil, member.pid)
                except psutil.NoSuchProcess:
                    continue
                if not member_identity_matches(member, actual, expected.pgid, os.getuid()):
                    raise SafetyError("member_termination_identity_changed")
                psutil.Process(member.pid).send_signal(sig)
        if not result["reaped"]:
            try:
                child.wait(timeout=max(0.001, min(interval, deadline - time.monotonic())))
                result["reaped"] = True
            except subprocess.TimeoutExpired:
                pass
        if result["reaped"] and not members:
            break
    result["exit_code"] = child.returncode
    if not result["reaped"]:
        raise SafetyError("direct_child_reap_unproven")
    if result["descendant_containment"] == "unproven":
        if expected is None or len(result["descendants"]) != len(members):
            raise SafetyError("member_inventory_unproven")
        terminal = True
        for member in members:
            if time.monotonic() >= deadline:
                raise SafetyError("containment_proof_deadline")
            try:
                actual = observe_identity(psutil, member.pid)
            except psutil.NoSuchProcess:
                continue
            if member_identity_matches(member, actual, expected.pgid, os.getuid()):
                terminal = False
        result["descendant_containment"] = "proven_terminal" if terminal and group_gone(expected.pgid) else "unproven"
        if result["descendant_containment"] != "proven_terminal":
            raise SafetyError("owned_group_containment_unproven")


def execute_case(manifest: dict, case: dict, psutil: ModuleType, pressure: PressureReader, attempt_version: str = "v1") -> dict:
    attempt_version = validate_attempt_version(attempt_version)
    bootstrap = SOURCE_ROOT / "tests/native/mac-native-bootstrap-v2.py"
    policy = MutationPolicy(case)
    records: list[Sensor] = []
    result = {"case": case["id"], "implemented": True, "executed": False, "passed": False,
              "blocked": None, "unproven": ["installed_package", "installed_host", "models", "physical_hard_cap"],
              "proof_scope": PROOF_SCOPE, "import_mode": "isolated_source_module", "synthetic": True,
              "source_commit": manifest["source_commit"], "sources": manifest["sources"],
              "sensors": [asdict(row) for row in records], "rss_sampled_peak": 0, "limits": LIMITS,
              "reaped": False, "exit_code": None, "identity": None, "output_bytes": 0,
              "descendants": [], "descendant_containment": "not_observed"}
    logs: dict[str, int] = {}
    child = None
    expected = None
    members: list[Identity] = []
    selector = selectors.DefaultSelector()
    held_lock = None
    frames = bytearray()
    protocol_result = None
    start = time.monotonic()
    try:
        records = []
        for index in range(3):
            records.append(sample(psutil, pressure))
            if index < 2:
                time.sleep(1)
        admission(records, time.monotonic_ns())
        if case["source"] is not None:
            source = Path(case["source"])
            info = owned_path(source)
            raw = read_bounded(source, 4096)
            if file_identity(info) != case["source_identity"] or digest(raw) != case["source_sha256"] or len(raw) != case["source_bytes"]:
                raise SafetyError("fixture_changed_before_launch")
        for name, checksum in manifest["sources"].items():
            if digest(read_bounded(SOURCE_ROOT / name, 262144, private=False)) != checksum:
                raise SafetyError("source_hash_changed")
        if digest(read_bounded(bootstrap, 262144, private=False)) != manifest["bootstrap_sha256"]:
            raise SafetyError("bootstrap_hash_changed")
        result["sensors"] = [asdict(row) for row in records]
        for channel in ("stdout", "stderr", "events"):
            logs[channel] = new_log(RUN_ROOT / "logs" / (case["id"] + "-" + channel + "-" + attempt_version + ".log"), "Retained native " + channel + " evidence")
        if case["id"] == "flock-a4":
            import fcntl
            lock = RUN_ROOT / "locks/contention-v1.lock"
            owned_path(lock)
            held_lock = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
            fcntl.flock(held_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = {"HOME": manifest["home"], "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        request = encoded({"version": 1, "source_root": str(SOURCE_ROOT), "run_root": str(RUN_ROOT),
                           "home": manifest["home"], "sources": manifest["sources"], "limits": LIMITS, "case": case,
                           "supervisor_sha256": manifest["supervisor_sha256"]}) + b"\n"
        if len(request) > LIMITS["input_bytes"] - LIMITS["events"] * 256:
            raise SafetyError("input_limit")
        fresh = sample(psutil, pressure)
        admission([*records[:-1], fresh], time.monotonic_ns())
        result["sensors"].append(asdict(fresh))
        start = time.monotonic()
        child = subprocess.Popen([sys.executable, "-I", "-S", "-B", str(bootstrap)], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                 cwd=str(RUN_ROOT), close_fds=True, start_new_session=True)
        result["executed"] = True
        expected = observe_identity(psutil, child.pid)
        if not same_owned_identity(expected, expected, os.getuid()):
            raise SafetyError("child_identity_unowned")
        result["identity"] = asdict(expected)
        assert child.stdin is not None and child.stdout is not None and child.stderr is not None
        for stream, channel in ((child.stdout, "stdout"), (child.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, channel)
        os.set_blocking(child.stdin.fileno(), False)
        write_pipe(child.stdin.fileno(), request, min(start + LIMITS["wall_seconds"], time.monotonic() + 0.1))
        input_count = len(request)
        previous = fresh
        ready = False
        next_sample = time.monotonic()
        while selector.get_map():
            now = time.monotonic()
            if now - start >= LIMITS["wall_seconds"]:
                raise SafetyError("child_wall_deadline")
            if child.poll() is None and now >= next_sample:
                current = sample(psutil, pressure)
                if current.pressure != 1 or current.free_bytes < FREE_MIN or current.swap_used_bytes > previous.swap_used_bytes or time.monotonic_ns() - current.monotonic_ns > 500_000_000:
                    raise SafetyError("during_child_sensor_refusal")
                actual = observe_identity(psutil, child.pid)
                if not same_owned_identity(expected, actual, os.getuid()):
                    raise SafetyError("child_identity_changed")
                process = psutil.Process(child.pid)
                descendants = process.children(recursive=True)
                if descendants:
                    result["descendants"] = [item.pid for item in descendants[:32]]
                    result["descendant_containment"] = "unproven"
                    if len(descendants) > 32:
                        raise SafetyError("descendant_inventory_limit")
                    members = [observe_identity(psutil, item.pid) for item in descendants]
                    if any(member.uid != os.getuid() or member.pgid != expected.pgid for member in members):
                        raise SafetyError("descendant_identity_unowned")
                    raise SafetyError("unexpected_descendant")
                rss = integer(process.memory_info().rss)
                result["rss_sampled_peak"] = max(result["rss_sampled_peak"], rss)
                if rss > LIMITS["rss_abort_bytes"]:
                    raise SafetyError("sampled_rss_limit")
                if len(result["sensors"]) >= 64:
                    raise SafetyError("sensor_inventory_limit")
                result["sensors"].append(asdict(current))
                previous = current
                next_sample = now + 0.1
            for key, _ in selector.select(0.01):
                try:
                    chunk = os.read(key.fileobj.fileno(), 4096)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                result["output_bytes"] += len(chunk)
                if result["output_bytes"] > LIMITS["output_bytes"]:
                    raise SafetyError("combined_output_limit")
                channel = key.data
                write_all(logs[channel], chunk)
                if channel != "stdout":
                    continue
                frames.extend(chunk)
                while b"\n" in frames:
                    line, _, rest = frames.partition(b"\n")
                    frames = bytearray(rest)
                    if len(line) > LIMITS["frame_bytes"]:
                        raise SafetyError("protocol_frame_limit")
                    frame_deadline = min(start + LIMITS["wall_seconds"], time.monotonic() + 0.1)
                    frame = json.loads(line)
                    if type(frame) is not dict:
                        raise SafetyError("protocol_object_required")
                    if frame.get("type") == "mutation":
                        if not ready or protocol_result is not None:
                            raise SafetyError("mutation_outside_ready_execution")
                        try:
                            grant = policy.decide(frame)
                            mutation_target(grant, frame)
                        except (SafetyError, OSError, ValueError) as error:
                            write_all(logs["events"], encoded({"request": frame, "decision": {"type": "deny", "reason": type(error).__name__ + ":" + str(error)[:256]}}) + b"\n")
                            os.fsync(logs["events"])
                            raise
                        write_all(logs["events"], encoded({"request": frame, "decision": grant}) + b"\n")
                        os.fsync(logs["events"])
                        announce(Path(grant["path"]), grant["purpose"])
                        reply = encoded(grant) + b"\n"
                        input_count += len(reply)
                        if input_count > LIMITS["input_bytes"]:
                            raise SafetyError("input_limit")
                        write_pipe(child.stdin.fileno(), reply, frame_deadline)
                    elif frame == {"type": "lock_busy"} and ready and protocol_result is None and case["id"] == "flock-a4" and held_lock is not None:
                        os.close(held_lock)
                        held_lock = None
                        reply = b'{"type":"lock_released"}\n'
                        input_count += len(reply)
                        if input_count > LIMITS["input_bytes"]:
                            raise SafetyError("input_limit")
                        write_pipe(child.stdin.fileno(), reply, frame_deadline)
                    elif frame.get("type") == "result" and protocol_result is None:
                        if not ready:
                            raise SafetyError("result_before_ready")
                        protocol_result = validate_result(frame, case, manifest["sources"])
                    elif frame.get("type") == "ready" and not ready:
                        validate_ready(frame, child.pid)
                        ready = True
                        result["limit_readbacks"] = frame["limits"]
                    else:
                        raise SafetyError("unexpected_protocol_frame")
                if len(frames) > LIMITS["frame_bytes"]:
                    raise SafetyError("protocol_frame_limit")
        if frames:
            raise SafetyError("truncated_protocol_frame")
        remaining = max(0.001, start + LIMITS["wall_seconds"] - time.monotonic())
        result["exit_code"] = child.wait(timeout=remaining)
        result["reaped"] = True
        expected_exit = 73 if case["id"] == "flock-a4" else 0
        if result["exit_code"] != expected_exit or type(protocol_result) is not dict or protocol_result.get("passed") is not True or protocol_result.get("proof_scope") != PROOF_SCOPE:
            raise SafetyError("case_nonterminal_or_failed")
        if case["id"] == "flock-a4":
            lock_fd = os.open(RUN_ROOT / "locks/contention-v1.lock", os.O_RDWR | os.O_NOFOLLOW)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(lock_fd)
        if not group_gone(expected.pgid):
            result["descendant_containment"] = "unproven"
            raise SafetyError("post_exit_group_containment_unproven")
        result["passed"] = True
        result["child_result"] = protocol_result
    except (SafetyError, OSError, ValueError, subprocess.SubprocessError, psutil.Error) as error:
        result["blocked"] = type(error).__name__ + ":" + str(error)[:256]
    finally:
        if not result["sensors"]:
            result["sensors"] = [asdict(row) for row in records]
        if child is None:
            result["reaped"] = True
        if held_lock is not None:
            try:
                os.close(held_lock)
            except OSError as error:
                result["blocked"] = "lock_close_failed:" + str(error)[:256]
                result["passed"] = False
        if child is not None and (not result["reaped"] or result["descendant_containment"] == "unproven"):
            try:
                terminate_owned(child, expected, members, psutil, result)
            except (SafetyError, OSError, ValueError, subprocess.SubprocessError, psutil.Error) as error:
                result["blocked"] = "identity_or_reap_unproven:" + str(error)[:256]
                result["passed"] = False
        result["wall_seconds"] = time.monotonic() - start
        try:
            selector.close()
        except OSError as error:
            result["blocked"] = "selector_close_failed:" + str(error)[:256]
            result["passed"] = False
        if child is not None:
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError as error:
                        result["blocked"] = "pipe_close_failed:" + str(error)[:256]
                        result["passed"] = False
        for fd in logs.values():
            try:
                os.fsync(fd)
            except OSError as error:
                result["blocked"] = "evidence_sync_failed:" + str(error)[:256]
                result["passed"] = False
            finally:
                try:
                    os.close(fd)
                except OSError as error:
                    result["blocked"] = "evidence_close_failed:" + str(error)[:256]
                    result["passed"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--execute-root-authorized", action="store_true")
    args = parser.parse_args()
    if not args.execute_root_authorized or sys.platform != "darwin" or not sys.flags.isolated or not sys.flags.no_site or not sys.flags.dont_write_bytecode:
        raise SafetyError("root_isolated_native_execution_required")
    sys.dont_write_bytecode = True
    os.umask(0o077)
    manifest_path = canonical_path(args.manifest, RUN_ROOT)
    target, version = evidence_paths(manifest_path)
    owned_path(RUN_ROOT, directory=True)
    owned_path(RUN_ROOT / "home", directory=True)
    manifest_bytes = read_bounded(manifest_path, 65536)
    manifest = validate_manifest(json.loads(manifest_bytes))
    if (version != "v2" and any(case["id"] == "capture-revision" for case in manifest["cases"])) or (version == "v2" and [case["id"] for case in manifest["cases"]] != ["capture-revision"]):
        raise SafetyError("unexpected_manifest_case_stage")
    if digest(read_bounded(Path(__file__).resolve(), 262144, private=False)) != manifest["supervisor_sha256"]:
        raise SafetyError("supervisor_hash_changed")
    results: list[dict] = []
    setup_failure = None
    # Reserve before runtime admission so a retained refusal cannot be replayed.
    fd = new_log(target, "Retained Stage A results; native host/model/hard-cap acceptance remains unproven")
    try:
        try:
            psutil = pinned_runtime()
            pressure = PressureReader()
            for case in manifest["cases"]:
                result = execute_case(manifest, case, psutil, pressure, attempt_version=version)
                results.append(result)
                if not result["passed"] or not result["reaped"]:
                    break
        except (SafetyError, OSError, ValueError, ImportError) as error:
            setup_failure = type(error).__name__ + ":" + str(error)[:256]
        data = encoded({"version": 1, "manifest_version": version, "manifest_sha256": digest(manifest_bytes),
                        "proof_scope": PROOF_SCOPE, "cases": results, "setup_failure": setup_failure,
                        "complete": setup_failure is None and len(results) == len(manifest["cases"]) and all(item["passed"] for item in results)}) + b"\n"
        if len(data) > LIMITS["output_bytes"]:
            raise SafetyError("results_byte_limit")
        write_all(fd, data)
    finally:
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return 0 if setup_failure is None and len(results) == len(manifest["cases"]) and all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
