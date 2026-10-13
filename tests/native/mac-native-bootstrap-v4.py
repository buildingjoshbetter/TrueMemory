"""Fixed Stage A child. Importing definitions never launches native work."""
from __future__ import annotations

import hashlib
import json
import os
import select
import signal
import stat
import sys
import time
import types
from pathlib import Path

LIMITS = {"wall_seconds": 5, "termination_seconds": 1, "cpu_soft": 2, "cpu_hard": 3,
          "rss_abort_bytes": 67108864, "core_bytes": 0, "nofile": 64,
          "fsize_bytes": 1048576, "input_bytes": 65536, "output_bytes": 262144,
          "events": 32, "frame_bytes": 2048, "ack_ms": 100, "threads": 1}
RUN_ROOT = Path("/Users/j/dev/truememory-maintenance-v1/.work/native-mac-v1/run-0001")
SOURCE_ROOT = Path("/Users/j/dev/truememory-maintenance-v1/.work/worktrees/mac-native-acceptance-v1")
PROOF_SCOPE = "instrumented_unmodified_source_native_filesystem_lock_capture"
SOURCE_MODULES = (
    ("truememory._platform", "truememory/_platform.py"),
    ("truememory.ingest.hooks._shared", "truememory/ingest/hooks/_shared.py"),
    ("truememory.ingest.completion", "truememory/ingest/completion.py"),
    ("truememory.ingest.transcript", "truememory/ingest/transcript.py"),
    ("truememory.ingest.provenance", "truememory/ingest/provenance.py"),
)


class BootstrapError(ValueError):
    pass


def encoded(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class Protocol:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.read_bytes = 0
        self.written_bytes = 0

    def emit(self, value: object) -> None:
        data = encoded(value) + b"\n"
        if len(data) > LIMITS["frame_bytes"]:
            raise BootstrapError("output_frame_limit")
        self.written_bytes += len(data)
        if self.written_bytes > LIMITS["output_bytes"]:
            raise BootstrapError("output_limit")
        while data:
            count = os.write(1, data)
            if count <= 0:
                raise BootstrapError("output_no_progress")
            data = data[count:]

    def receive(self, maximum: int, seconds: float) -> dict:
        deadline = time.monotonic() + seconds
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([0], [], [], remaining)[0]:
                raise BootstrapError("input_deadline")
            chunk = os.read(0, min(4096, maximum + 1))
            if not chunk:
                raise BootstrapError("input_eof")
            self.read_bytes += len(chunk)
            self.buffer.extend(chunk)
            if self.read_bytes > LIMITS["input_bytes"] or len(self.buffer) > maximum:
                raise BootstrapError("input_limit")
        line, _, remaining = self.buffer.partition(b"\n")
        self.buffer = bytearray(remaining)
        if len(line) > maximum:
            raise BootstrapError("input_frame_limit")
        value = json.loads(line)
        if type(value) is not dict:
            raise BootstrapError("input_shape")
        return value


def arm_limits() -> dict[str, list[int]]:
    if sys.platform != "darwin" or not sys.flags.isolated or not sys.flags.no_site or not sys.flags.dont_write_bytecode:
        raise BootstrapError("isolated_mac_child_required")
    import resource

    os.umask(0o077)
    sys.dont_write_bytecode = True
    expected = {"cpu": (2, 3), "core": (0, 0), "nofile": (64, 64), "fsize": (1048576, 1048576)}
    kinds = {"cpu": resource.RLIMIT_CPU, "core": resource.RLIMIT_CORE,
             "nofile": resource.RLIMIT_NOFILE, "fsize": resource.RLIMIT_FSIZE}
    for name, limits in expected.items():
        current = resource.getrlimit(kinds[name])
        if any(old != resource.RLIM_INFINITY and old < new for old, new in zip(current, limits)):
            raise BootstrapError("existing_limit_below_reviewed_limit")
        resource.setrlimit(kinds[name], limits)
        if resource.getrlimit(kinds[name]) != limits:
            raise BootstrapError("limit_readback_failed")
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, 5)
    return {name: list(limits) for name, limits in expected.items()}


def compile_module(name: str, path: Path, checksum: str, *, package: bool = False) -> types.ModuleType:
    if path.is_symlink() or path.resolve() != path:
        raise BootstrapError("source_path_link")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 262144:
            raise BootstrapError("source_read_type_or_limit")
        stream = os.fdopen(fd, "rb")
        fd = None
        with stream:
            data = stream.read(262145)
            after = os.fstat(stream.fileno())
    finally:
        if fd is not None:
            os.close(fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise BootstrapError("source_changed_during_read")
    if len(data) > 262144 or hashlib.sha256(data).hexdigest() != checksum:
        raise BootstrapError("source_hash_changed")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name if package else name.rpartition(".")[0]
    if package:
        module.__path__ = [str(path.parent)]
    sys.modules[name] = module
    exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


class AuditGuard:
    """Bounded telemetry/grants for fixed trusted source, not a security sandbox."""
    def __init__(self, protocol: Protocol, request: dict, policy_module: types.ModuleType) -> None:
        self.protocol, self.request, self.policy_module = protocol, request, policy_module
        self.case = request["case"]
        self.sequence = 0
        self.activated = False
        self.approved: dict[Path, int] = {}
        self.known_fds: dict[int, tuple[int, int]] = {}
        self.open_intents: list[tuple[Path, int]] = []
        self.directory_sync: DirectorySyncAdapter | None = None
        self.source = Path(self.case["source"]) if self.case["source"] is not None else None
        self.read_paths = {SOURCE_ROOT / relative for _, relative in SOURCE_MODULES}
        self.read_paths.add(SOURCE_ROOT / "tests/native/mac-native-acceptance-v4.py")
        if self.source is not None:
            self.read_paths.add(self.source)
        if self.case["id"] == "negative-a3":
            self.read_paths.add(RUN_ROOT / "home/.claude/projects/native-acceptance-v1/negative-directory-v1.jsonl")
        if self.case["previous"] is not None:
            self.read_paths.add(policy_module.account_directory(self.case) / "captures" / (self.case["previous"]["key"] + ".jsonl"))
        self.runtime_root = Path(sys.base_prefix) / "lib/python3.12"

    def grant(self, event: str, path: Path, *, flags: int, source: Path | None = None) -> None:
        self.sequence += 1
        if self.sequence > LIMITS["events"]:
            raise BootstrapError("mutation_event_limit")
        self.protocol.emit({"type": "mutation", "version": 1, "seq": self.sequence,
                            "event": event, "path": str(path), "source": str(source) if source is not None else None,
                            "flags": flags})
        answer = self.protocol.receive(LIMITS["frame_bytes"], LIMITS["ack_ms"] / 1000)
        if set(answer) != {"type", "version", "seq", "path", "purpose"} or answer["type"] != "grant" or type(answer["version"]) is not int or answer["version"] != 1 or type(answer["seq"]) is not int or answer["seq"] != self.sequence or answer["path"] != str(path) or type(answer["purpose"]) is not str or not answer["purpose"]:
            raise BootstrapError("grant_mismatch")
        if event == "open":
            self.approved[path] = flags & os.O_ACCMODE
        elif event == "os.rename" and source is not None:
            if source not in self.approved:
                raise BootstrapError("replace_source_not_approved")
            self.approved[path] = self.approved.pop(source)

    def path(self, value: object) -> Path:
        if not isinstance(value, (str, bytes)):
            raise BootstrapError("path_argument_type")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if "\0" in value or not Path(value).is_absolute() or ".." in Path(value).parts:
            raise BootstrapError("path_not_absolute")
        path = Path(value)
        if str(path) != value or path.resolve() != path or any(part.is_symlink() for part in (path, *path.parents)):
            raise BootstrapError("path_not_canonical")
        return path

    def readable(self, path: Path) -> bool:
        if path in self.read_paths or path in self.approved:
            return True
        account = self.policy_module.account_directory(self.case)
        if path in {account / "state.json", account / "state.lock"}:
            return True
        if path.is_relative_to(self.runtime_root) and "site-packages" not in path.parts:
            return True
        return False

    def descriptor(self, fd: int, mode: object) -> None:
        import fcntl
        # CPython FileIO reports r/w for an explicitly binary rb/wb fd wrapper.
        if type(fd) is not int or fd in (0, 1, 2) or type(mode) is not str or mode not in ("r", "rb", "w", "wb"):
            raise BootstrapError("unsupported_descriptor_wrapper")
        info = os.fstat(fd)
        identity = (info.st_dev, info.st_ino)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise BootstrapError("descriptor_not_regular_owned")
        access = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
        if not self.open_intents:
            raise BootstrapError("descriptor_not_preapproved")
        path, allowed = self.open_intents.pop()
        current = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(current.st_mode) or current.st_uid != os.getuid() or (current.st_dev, current.st_ino) != identity:
            raise BootstrapError("descriptor_intent_identity_mismatch")
        if allowed != access or (mode.startswith("r") and access != os.O_RDONLY) or (mode.startswith("w") and access != os.O_WRONLY):
            raise BootstrapError("descriptor_access_mismatch")
        if len(self.known_fds | {fd: identity}) > LIMITS["events"]:
            raise BootstrapError("descriptor_inventory_limit")
        self.known_fds[fd] = identity

    def __call__(self, event: str, args: tuple[object, ...]) -> None:
        if event == "truememory.native.audit_probe":
            if args != ("fixed-stage-a-v1",):
                raise BootstrapError("audit_probe_mismatch")
            self.activated = True
            return
        if event == "open":
            path, mode, flags = args
            if type(flags) is not int:
                raise BootstrapError("open_flags_type")
            if type(path) is int:
                self.descriptor(path, mode)
                return
            target = self.path(path)
            if flags & getattr(os, "O_DIRECTORY", 0):
                if self.directory_sync is None:
                    raise BootstrapError("directory_sync_adapter_missing")
                self.directory_sync.audit_open(target, mode, flags)
                return
            if flags & (os.O_CREAT | os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_APPEND):
                if not target.is_relative_to(RUN_ROOT):
                    raise BootstrapError("write_escape")
                self.grant("open", target, flags=flags)
            elif not self.readable(target):
                raise BootstrapError("unplanned_read")
            if mode is None:
                # Only the latest approved raw open may supply a wrapper intent.
                self.open_intents[:] = [(target, flags & os.O_ACCMODE)]
            else:
                self.open_intents.clear()
        elif event == "os.mkdir":
            path, mode, dir_fd = args
            if dir_fd not in (-1, None) or type(mode) is not int:
                raise BootstrapError("directory_descriptor_forbidden")
            self.grant(event, self.path(path), flags=mode)
        elif event == "os.rename":
            source, destination, source_fd, destination_fd = args
            if source_fd not in (-1, None) or destination_fd not in (-1, None):
                raise BootstrapError("rename_descriptor_forbidden")
            self.grant(event, self.path(destination), flags=0, source=self.path(source))
        elif event in ("os.listdir", "os.scandir"):
            target = self.path(args[0])
            if not target.is_relative_to(RUN_ROOT) and not (target.is_relative_to(self.runtime_root) and "site-packages" not in target.parts):
                raise BootstrapError("unplanned_directory_read")
        elif event.startswith("socket.") or event in {"subprocess.Popen", "os.fork", "os.forkpty", "os.posix_spawn", "os.spawn", "os.system", "os.exec", "_thread.start_new_thread", "ctypes.dlopen", "sqlite3.connect", "sqlite3.load_extension", "sqlite3.enable_load_extension", "os.remove", "os.rmdir", "os.link", "os.symlink", "os.truncate", "os.chmod", "os.chown", "os.chflags", "resource.setrlimit", "sys.addaudithook"}:
            raise BootstrapError("forbidden_native_operation:" + event)
        elif event in ("os.kill", "os.killpg"):
            if event != "os.kill" or args != (os.getpid(), 0):
                raise BootstrapError("external_signal_forbidden")
        elif event == "import":
            name = args[0]
            if type(name) is not str:
                raise BootstrapError("import_name_type")
            if name.split(".")[0] in {"torch", "transformers", "mlx", "numpy", "sentence_transformers", "requests", "httpx", "ctypes", "psutil"}:
                raise BootstrapError("model_network_dependency_forbidden")
            if name.startswith("truememory.") and name not in {item[0] for item in SOURCE_MODULES} and name not in {"truememory.ingest", "truememory.ingest.hooks"}:
                raise BootstrapError("unplanned_production_import")


class DirectorySyncAdapter:
    """Bind the actual source helper's directory handle, without rewriting it."""
    def __init__(self, guard: AuditGuard, completion: types.ModuleType) -> None:
        self.guard = guard
        self.completion = completion
        self.function = completion._sync_directory
        if (type(self.function) is not types.FunctionType
                or self.function.__globals__ is not completion.__dict__
                or self.function.__code__.co_filename != str(SOURCE_ROOT / "truememory/ingest/completion.py")):
            raise BootstrapError("directory_sync_source_identity")
        self.raw_open, self.raw_fsync, self.raw_close = os.open, os.fsync, os.close
        self.pending: tuple[Path, int, tuple[int, int]] | None = None
        self.granted = False
        self.bindings: dict[int, tuple[Path, tuple[int, int], int]] = {}
        self.failed_open_bindings: dict[int, tuple[int, int, int, int] | None] = {}
        self.failures: list[dict[str, object]] = []
        self.installed = False

    def source_caller(self, frame: types.FrameType) -> bool:
        return frame.f_code is self.function.__code__ and frame.f_globals is self.completion.__dict__

    def target(self, value: object, flags: object) -> tuple[Path, tuple[int, int]]:
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        expected = os.O_RDONLY | directory_flag
        if (os.name == "nt" or not directory_flag or type(flags) is not int
                or flags & ~getattr(os, "O_CLOEXEC", 0) != expected
                or self.guard.case["id"] not in {"capture-a1", "capture-a2", "capture-revision"}):
            raise BootstrapError("unsupported_directory_sync_flags_or_case")
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        path = self.guard.path(value)
        account = self.guard.policy_module.account_directory(self.guard.case)
        if path not in {account, account / "captures"}:
            raise BootstrapError("unplanned_directory_sync_target")
        info = self.guard.policy_module.owned_path(path, directory=True)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise BootstrapError("directory_sync_target_not_private_owned")
        return path, (info.st_dev, info.st_ino)

    def audit_open(self, path: Path, mode: object, flags: int) -> None:
        if self.pending is None or mode is not None or self.granted:
            raise BootstrapError("directory_sync_not_in_source_open")
        target, expected_flags, expected_identity = self.pending
        current, identity = self.target(path, flags)
        if current != target or flags & ~getattr(os, "O_CLOEXEC", 0) != expected_flags or identity != expected_identity:
            raise BootstrapError("directory_sync_preopen_changed")
        self.guard.grant("open", target, flags=flags)
        # Directory grants do not authorize ordinary reads or fdopen wrappers.
        self.guard.approved.pop(target, None)
        self.granted = True

    def verify(self, fd: int, path: Path, identity: tuple[int, int], *, check_path: bool) -> os.stat_result:
        import fcntl
        if type(fd) is not int or fd < 3:
            raise BootstrapError("directory_sync_descriptor_type")
        info = os.fstat(fd)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or (info.st_dev, info.st_ino) != identity
                or fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY):
            raise BootstrapError("directory_sync_descriptor_changed")
        if check_path:
            current = self.guard.policy_module.owned_path(path, directory=True)
            if (not stat.S_ISDIR(current.st_mode) or current.st_uid != os.getuid()
                    or current.st_mode & 0o077 or (current.st_dev, current.st_ino) != identity):
                raise BootstrapError("directory_sync_path_changed")
        return info

    def failure(self, fd: int, reason: str, *, closed: bool) -> None:
        if len(self.failures) >= LIMITS["events"]:
            raise BootstrapError("directory_sync_failure_inventory")
        self.failures.append({"fd": fd, "reason": reason, "closed": closed})

    def close_failed_open(self, fd: int, observed: os.stat_result | None, error: BaseException) -> None:
        if fd in self.bindings:
            self.failure(fd, "returned_descriptor_bound_conflict:" + type(error).__name__, closed=False)
            return
        if len(self.failed_open_bindings) >= LIMITS["events"]:
            raise BootstrapError("directory_sync_failed_binding_inventory")
        if observed is None:
            self.failed_open_bindings[fd] = None
            self.failure(fd, "returned_descriptor_identity_unavailable:" + type(error).__name__, closed=False)
            return
        try:
            current = os.fstat(fd)
            if ((current.st_dev, current.st_ino, stat.S_IFMT(current.st_mode), current.st_uid)
                    != (observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode), observed.st_uid)):
                raise BootstrapError("failed_open_descriptor_reused")
            self.raw_close(fd)
        except (OSError, BootstrapError) as close_error:
            self.failed_open_bindings[fd] = (observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode), observed.st_uid)
            self.failure(fd, "returned_descriptor_close_unproven:" + type(close_error).__name__, closed=False)
            raise BootstrapError("directory_sync_failed_open_close_unproven") from close_error
        self.failure(fd, type(error).__name__, closed=True)

    def open(self, path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if not self.source_caller(sys._getframe(1)):
            if type(flags) is int and flags & getattr(os, "O_DIRECTORY", 0):
                raise BootstrapError("directory_sync_foreign_caller")
            if dir_fd is None:
                return self.raw_open(path, flags, mode)
            return self.raw_open(path, flags, mode, dir_fd=dir_fd)
        if type(mode) is not int or mode != 0o777 or dir_fd is not None or self.pending is not None:
            raise BootstrapError("directory_sync_open_arguments")
        target, identity = self.target(path, flags)
        if len(self.bindings) >= LIMITS["events"]:
            raise BootstrapError("directory_sync_binding_inventory")
        self.pending = (target, flags & ~getattr(os, "O_CLOEXEC", 0), identity)
        self.granted = False
        fd = None
        observed = None
        try:
            fd = self.raw_open(path, flags, mode)
            if type(fd) is not int or fd < 3:
                raise BootstrapError("directory_sync_open_return")
            observed = os.fstat(fd)
            if not self.granted:
                raise BootstrapError("directory_sync_grant_missing")
            self.verify(fd, target, identity, check_path=True)
            if fd in self.bindings:
                raise BootstrapError("directory_sync_returned_fd_reused")
            self.bindings[fd] = (target, identity, self.guard.sequence)
            return fd
        except (OSError, BootstrapError) as error:
            if type(fd) is int and fd >= 3:
                self.close_failed_open(fd, observed, error)
            raise
        finally:
            self.pending = None
            self.granted = False

    def fsync(self, fd: int) -> None:
        if not self.source_caller(sys._getframe(1)):
            if fd in self.bindings:
                raise BootstrapError("directory_sync_foreign_fsync")
            self.raw_fsync(fd)
            return
        if type(fd) is not int or fd not in self.bindings:
            raise BootstrapError("directory_sync_fsync_unbound")
        path, identity, sequence = self.bindings[fd]
        if not 1 <= sequence <= self.guard.sequence <= LIMITS["events"]:
            raise BootstrapError("directory_sync_grant_binding")
        self.verify(fd, path, identity, check_path=True)
        self.raw_fsync(fd)

    def close(self, fd: int) -> None:
        if not self.source_caller(sys._getframe(1)):
            if fd in self.bindings:
                raise BootstrapError("directory_sync_foreign_close")
            self.raw_close(fd)
            return
        if type(fd) is not int or fd not in self.bindings:
            raise BootstrapError("directory_sync_close_unbound")
        path, identity, _ = self.bindings[fd]
        try:
            # A replaced directory path still permits closing our original handle.
            self.verify(fd, path, identity, check_path=False)
            self.raw_close(fd)
        except (OSError, BootstrapError) as error:
            self.failure(fd, "bound_descriptor_close_unproven:" + type(error).__name__, closed=False)
            raise
        del self.bindings[fd]

    def install(self) -> None:
        if self.installed or self.guard.directory_sync is not None:
            raise BootstrapError("directory_sync_adapter_already_installed")
        self.guard.directory_sync = self
        os.open, os.fsync, os.close = self.open, self.fsync, self.close
        self.installed = True


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise BootstrapError(reason)


def execute_case(case: dict, modules: dict[str, types.ModuleType], protocol: Protocol, policy: types.ModuleType) -> dict:
    completion = modules["truememory.ingest.completion"]
    platform = modules["truememory._platform"]
    transcript = modules["truememory.ingest.transcript"]
    provenance = modules["truememory.ingest.provenance"]
    kind = case["id"]
    details: dict[str, object] = {}
    if kind in {"capture-a1", "capture-a2", "capture-revision", "invalid-state"}:
        try:
            capture = completion.admit_capture(case["source"], case["session"], "native-acceptance-v1", case["database"], "claude", root=RUN_ROOT / "completion")
        except completion.CompletionError as error:
            if kind != "invalid-state" or str(error) != "invalid_retained_state":
                raise
            details["category"] = str(error)
        else:
            require(kind != "invalid-state", "invalid_state_was_admitted")
            require(capture.key == policy.digest(policy.encoded(policy.capture_metadata(case))), "capture_key_mismatch")
            completion.verify_capture(capture)
            raw = capture.path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == case["source_sha256"] and len(raw) == case["source_bytes"], "capture_bytes_mismatch")
            require(capture.metadata == policy.capture_metadata(case), "capture_scope_mismatch")
            require(capture.path.stat().st_mode & 0o077 == 0, "capture_not_private")
            second = completion.admit_capture(case["source"], case["session"], "native-acceptance-v1", case["database"], "claude", root=RUN_ROOT / "completion")
            require(second.key == capture.key and second.path == capture.path, "capture_not_idempotent")
            if kind == "capture-revision":
                old = case["previous"]
                old_path = capture.directory / "captures" / (old["key"] + ".jsonl")
                previous = old_path.read_bytes()
                require(capture.key != old["key"] and hashlib.sha256(previous).hexdigest() == old["sha256"] and len(previous) == old["byte_count"], "revision_lost_previous_evidence")
            details.update({"category": "immutable_capture_verified", "capture_key": capture.key, "source_sha256": case["source_sha256"], "byte_count": len(raw), "idempotent": True})
    elif kind == "negative-a3":
        base = RUN_ROOT / "home/.claude/projects/native-acceptance-v1"
        controls = [(case["source"], "unknown", "claude", "source_not_admitted"),
                    (case["source"], case["session"], "unknown", "source_not_admitted"),
                    (str(RUN_ROOT / "fixtures/negative-outside-v1.jsonl"), case["session"], "claude", "source_not_admitted"),
                    (str(base / "negative-source-link-v1.jsonl"), case["session"], "claude", "source_link"),
                    (str(base / "negative-directory-v1.jsonl"), case["session"], "claude", "source_not_regular")]
        categories = []
        for source, session, host, expected in controls:
            try:
                completion.admit_capture(source, session, "native-acceptance-v1", case["database"], host, root=RUN_ROOT / "completion")
            except completion.CompletionError as error:
                require(str(error) == expected, "negative_category_mismatch")
                categories.append(str(error))
            else:
                raise BootstrapError("negative_source_admitted")
        details.update({"category": "source_negatives_verified", "categories": categories})
    elif kind == "flock-a4":
        lock = RUN_ROOT / "locks/contention-v1.lock"
        require(platform.try_file_lock(lock) is None, "contention_not_observed")
        protocol.emit({"type": "lock_busy"})
        require(protocol.receive(2048, 0.1) == {"type": "lock_released"}, "lock_release_protocol")
        fd = platform.try_file_lock(lock)
        require(type(fd) is int and fd >= 0, "released_lock_not_acquired")
        details.update({"category": "native_contention_then_acquired", "post_exit_release_requires_parent": True})
        # Leave the actual acquired descriptor open: the parent proves exit release.
    elif kind == "parser-a5":
        parsed = transcript.parse_transcript_outcome(Path(case["source"]))
        require(parsed.status == "complete" and parsed.file_version is not None and parsed.file_version.stable, "parser_scope_unproven")
        require(parsed.file_version.sha256 == case["source_sha256"], "parser_version_mismatch")
        require(any(message.timestamp for message in parsed.messages) and any(not message.timestamp for message in parsed.messages), "fixture_lacks_known_unknown_times")
        formatted = transcript.format_for_extraction(parsed.messages)
        context = provenance.source_context(formatted, case["session"])
        require(context.length == len(formatted) and context.digest == hashlib.sha256(formatted.encode("utf-8")).hexdigest(), "provenance_context_mismatch")
        require(provenance._known_time("") is None and provenance._known_time("2020-01-01T00:00:00Z") is not None, "source_time_guard_mismatch")
        details.update({"category": "parser_and_source_context_verified", "messages": len(parsed.messages), "known_unknown_time": True,
                        "scorer_extractor_store_proven": False})
    else:
        raise BootstrapError("unsupported_case")
    return details


def main() -> int:
    readback = arm_limits()
    protocol = Protocol()
    request = protocol.receive(LIMITS["input_bytes"], 1)
    required = {"version", "source_root", "run_root", "home", "sources", "limits", "case", "supervisor_sha256"}
    require(set(request) == required and type(request["version"]) is int and request["version"] == 1, "request_shape")
    require(request["source_root"] == str(SOURCE_ROOT) and request["run_root"] == str(RUN_ROOT) and request["home"] == str(RUN_ROOT / "home") and os.environ.get("HOME") == request["home"], "request_roots")
    require(request["limits"] == LIMITS and all(type(number) is int for number in request["limits"].values()), "request_limits")
    require(type(request["sources"]) is dict and set(request["sources"]) == {relative for _, relative in SOURCE_MODULES}, "source_pin_set")
    policy = compile_module("mac_native_policy_v4", SOURCE_ROOT / "tests/native/mac-native-acceptance-v4.py", request["supervisor_sha256"])
    policy.validate_case(request["case"])
    for checksum in request["sources"].values():
        policy.hash_text(checksum)
    guard = AuditGuard(protocol, request, policy)
    sys.addaudithook(guard)
    sys.audit("truememory.native.audit_probe", "fixed-stage-a-v1")
    require(guard.activated, "audit_hook_not_installed")
    protocol.emit({"type": "ready", "version": 1, "pid": os.getpid(), "limits": readback, "audit_installed": True})
    for name, relative in (("truememory", "truememory"), ("truememory.ingest", "truememory/ingest"), ("truememory.ingest.hooks", "truememory/ingest/hooks")):
        module = types.ModuleType(name)
        module.__path__ = [str(SOURCE_ROOT / relative)]
        module.__package__ = name
        sys.modules[name] = module
    modules = {name: compile_module(name, SOURCE_ROOT / relative, request["sources"][relative]) for name, relative in SOURCE_MODULES}
    directory_sync = DirectorySyncAdapter(guard, modules["truememory.ingest.completion"])
    directory_sync.install()
    details = execute_case(request["case"], modules, protocol, policy)
    require(not directory_sync.bindings and not directory_sync.failed_open_bindings and not directory_sync.failures, "directory_sync_nonterminal")
    protocol.emit({"type": "result", "version": 1, "case": request["case"]["id"], "passed": True,
                   "proof_scope": PROOF_SCOPE, "import_mode": "isolated_source_module", "sources": request["sources"], "details": details})
    if request["case"]["id"] == "flock-a4":
        os._exit(73)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
