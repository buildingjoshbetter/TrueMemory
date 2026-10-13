"""ROOT-only Stage B worker; native entry point requires isolated Mac execution.

Verified source is unmodified. Audit telemetry is not a native security sandbox.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import select
import signal
import socket
import stat
import sys
import threading
import time
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path

HARNESS_ROOT = Path("/Users/j/dev/truememory-maintenance-v1/.work/worktrees/mac-native-acceptance-v1")
SOURCE_ROOT = Path("/Users/j/dev/truememory-maintenance-v1/.work/worktrees/whole-process-memory-297-v1")
RUN_ROOT = Path("/Users/j/dev/truememory-maintenance-v1/.work/native-mac-v1/run-0002")
LIMITS = {"wall_seconds":5,"termination_seconds":1,"cpu_soft":2,"cpu_hard":3,
          "rss_abort_bytes":67108864,"core_bytes":0,"nofile":64,"fsize_bytes":1048576,
          "input_bytes":65536,"output_bytes":262144,"events":32,"frame_bytes":2048,
          "ack_ms":100,"threads":1}
PROOF_SCOPE = "instrumented_unmodified_full_package_source_af_unix_ping"
NEURAL_PREFIXES = frozenset({"torch","transformers","sentence_transformers","model2vec","mlx","mlx_lm","onnxruntime","tensorflow","jax","flax","safetensors","huggingface_hub","llama_cpp","openai","anthropic"})

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


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise BootstrapError(reason)


class AuditGuard:
    def __init__(self,protocol: Protocol,manifest: dict[str,object],policy: types.ModuleType,runtime: dict[str,object]) -> None:
        self.protocol,self.manifest,self.policy,self.runtime = protocol,manifest,policy,runtime
        self.sequence = 0
        self.activated = False
        self.exchange_lock = threading.Lock()
        self.open_intents: list[tuple[Path,int]] = []
        self.known_fds: dict[int,tuple[int,int]] = {}
        self.approved: dict[Path,int] = {}
        self.stdlib = Path(runtime["interpreter"]["trusted_stdlib"]["canonical_root"])
        self.excluded = set(runtime["interpreter"]["trusted_stdlib"]["excluded_subtrees"])
        self.read_paths = {SOURCE_ROOT/name for name in policy.SOURCE_HASHES} | {SOURCE_ROOT/name for name in policy.DATA_HASHES}
        self.read_paths |= set(policy.runtime_files(runtime))
        self.read_paths |= {policy.RUNTIME_PACKET,policy.INTERPRETER_PACKET,policy.LINKAGE_PACKET,HARNESS_ROOT/"tests/native/mac-native-package-acceptance-v4.py"}
        self.endpoint = RUN_ROOT/"home/.truememory/model.sock"
        self.listener: socket.socket | None = None
        self.client: socket.socket | None = None
        self.accepted: socket.socket | None = None
        self.accepted_identity: tuple[int,int] | None = None
        self.accepted_checked = False
        self.neural_imports: list[str] = []
        self.socket_count = 0
        self.terminal = False
        self.ctypes_namespace_loads = 0

    def path(self,value: object) -> Path:
        if type(value) is bytes:
            value = value.decode("utf-8")
        if type(value) is not str or not value or "\0" in value or not Path(value).is_absolute() or ".." in Path(value).parts:
            raise BootstrapError("path_argument")
        path = Path(value)
        if str(path) != value or path.resolve() != path or any(part.is_symlink() for part in (path,*path.parents)):
            raise BootstrapError("path_link_or_noncanonical")
        return path

    def readable(self,path: Path) -> bool:
        if path in self.read_paths or path in self.approved:
            return True
        if path.is_relative_to(self.stdlib):
            relative = path.relative_to(self.stdlib)
            return path.suffix == ".py" and not any(part in self.excluded for part in relative.parts)
        return False

    def grant(self,event: str,path: Path,*,flags: int) -> None:
        if self.terminal or not self.exchange_lock.acquire(timeout=0.1):
            raise BootstrapError("B_exchange_lock_or_terminal")
        try:
            self.sequence += 1
            require(self.sequence<=LIMITS["events"],"mutation_event_limit")
            self.protocol.emit({"type":"mutation","version":1,"seq":self.sequence,"event":event,"path":str(path),"source":None,"flags":flags})
            answer = self.protocol.receive(LIMITS["frame_bytes"],0.1)
            require(set(answer)=={"type","version","seq","path","purpose"} and answer["type"]=="grant" and type(answer["version"]) is int and answer["version"]==1 and type(answer["seq"]) is int and answer["seq"]==self.sequence and answer["path"]==str(path) and type(answer["purpose"]) is str and bool(answer["purpose"]),"grant_mismatch")
            if event == "open":
                self.approved[path] = flags & os.O_ACCMODE
        finally:
            self.exchange_lock.release()

    def descriptor(self,fd: int,mode: object) -> None:
        import fcntl
        require(type(fd) is int and fd not in (0,1,2) and type(mode) is str and mode in ("r","rb","w","wb"),"unsupported_descriptor_wrapper")
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and bool(self.open_intents),"descriptor_not_preapproved_regular")
        path,allowed = self.open_intents.pop()
        current = path.lstat()
        require(not path.is_symlink() and stat.S_ISREG(current.st_mode) and (current.st_dev,current.st_ino)==(info.st_dev,info.st_ino),"descriptor_fresh_identity")
        access = fcntl.fcntl(fd,fcntl.F_GETFL)&os.O_ACCMODE
        require(allowed==access and ((mode.startswith("r") and access==os.O_RDONLY) or (mode.startswith("w") and access==os.O_WRONLY)),"descriptor_access")
        if path.is_relative_to(RUN_ROOT):
            require(info.st_uid==os.getuid() and stat.S_IMODE(info.st_mode)==0o600,"owned_descriptor_mode")
        require(len(self.known_fds|{fd:(info.st_dev,info.st_ino)})<=32,"descriptor_inventory_limit")
        self.known_fds[fd] = (info.st_dev,info.st_ino)

    def socket_new(self,args: tuple[object,...]) -> None:
        obj,family,kind,protocol = args
        require(type(family) is int and family==socket.AF_UNIX and type(kind) is int and kind==socket.SOCK_STREAM and type(protocol) is int and protocol==0 and self.socket_count<3,"unplanned_socket_type_or_count")
        frame = sys._getframe(2)
        initializer = None
        caller = None
        for _ in range(6):
            filename = Path(frame.f_code.co_filename)
            if filename==self.stdlib/"socket.py" and frame.f_code.co_name=="__init__":
                initializer = frame
            if initializer is not None and frame is not initializer:
                caller = frame
                break
            if frame.f_back is None:
                break
            frame = frame.f_back
        require(initializer is not None and caller is not None,"socket_source_frame")
        fileno = initializer.f_locals.get("fileno")
        filename = Path(caller.f_code.co_filename)
        if fileno is None and filename==SOURCE_ROOT/"truememory/model_server.py" and caller.f_code.co_name=="run" and self.listener is None:
            self.listener = obj
        elif fileno is None and filename==SOURCE_ROOT/"truememory/model_client.py" and caller.f_code.co_name=="_connect" and self.client is None:
            self.client = obj
        elif type(fileno) is int and filename==self.stdlib/"socket.py" and caller.f_code.co_name=="accept" and caller.f_locals.get("self") is self.listener and caller.f_locals.get("fd")==fileno and self.accepted is None:
            require(self.listener is not None and self.listener.getsockname()==str(self.endpoint),"accepted_listener_endpoint")
            info = os.fstat(fileno)
            require(stat.S_ISSOCK(info.st_mode) and info.st_uid==os.getuid(),"accepted_descriptor_owner_type")
            self.accepted,self.accepted_identity = obj,(info.st_dev,info.st_ino)
            self.accepted_checked = True
        else:
            raise BootstrapError("foreign_socket_or_fileno")
        self.socket_count += 1
        self.grant("socket.__new__",self.endpoint,flags=0)

    def ctypes_process_namespace(self,args: tuple[object,...]) -> None:
        require(type(args) is tuple and len(args)==1 and args[0] is None and type(self.ctypes_namespace_loads) is int and self.ctypes_namespace_loads==0,"ctypes_process_namespace_args_or_replay")
        module = sys.modules.get("ctypes")
        require(type(module) is types.ModuleType,"ctypes_actual_module_missing")
        namespace = module.__dict__
        frame = sys._getframe(2)
        initializer = frame.f_back
        filename = str(self.stdlib/"ctypes/__init__.py")
        require(frame.f_code.co_filename==filename and frame.f_code.co_name=="__init__" and frame.f_globals is namespace,"ctypes_constructor_frame")
        require(initializer is not None and initializer.f_code.co_filename==filename and initializer.f_code.co_name=="<module>" and initializer.f_globals is namespace,"ctypes_module_initializer_frame")
        constructor = namespace.get("CDLL")
        dll_type = namespace.get("PyDLL")
        require(type(constructor) is type and type(dll_type) is type and "pythonapi" not in namespace,"ctypes_initializer_namespace")
        require(type(constructor.__init__) is types.FunctionType and frame.f_code is constructor.__init__.__code__,"ctypes_actual_constructor_code")
        instance = frame.f_locals.get("self")
        require(type(instance) is dll_type and getattr(instance,"_name",False) is None,"ctypes_actual_PyDLL_None")
        self.ctypes_namespace_loads += 1

    def __call__(self,event: str,args: tuple[object,...]) -> None:
        if event=="truememory.native.audit_probe":
            require(args==("fixed-stage-B-v1",),"audit_probe_mismatch")
            self.activated = True
            return
        if event=="open":
            path,mode,flags = args
            require(type(flags) is int,"open_flags_type")
            if type(path) is int:
                self.descriptor(path,mode)
                return
            target = self.path(path)
            if flags&(os.O_CREAT|os.O_WRONLY|os.O_RDWR|os.O_TRUNC|os.O_APPEND):
                require(target.is_relative_to(RUN_ROOT),"write_escape")
                self.grant("open",target,flags=flags)
            else:
                require(self.readable(target),"unplanned_read")
            self.open_intents[:] = [(target,flags&os.O_ACCMODE)] if mode is None else []
        elif event=="os.mkdir":
            path,mode,dir_fd = args
            require(dir_fd in (-1,None) and type(mode) is int,"mkdir_descriptor")
            self.grant(event,self.path(path),flags=mode)
        elif event=="os.chmod":
            path,mode,dir_fd = args
            require(dir_fd in (-1,None) and type(mode) is int,"chmod_descriptor")
            self.grant(event,self.path(path),flags=mode)
        elif event=="socket.__new__":
            self.socket_new(args)
        elif event in ("socket.bind","socket.connect"):
            obj,address = args
            require(address==str(self.endpoint),"foreign_socket_address")
            require(obj is (self.listener if event=="socket.bind" else self.client),"foreign_socket_actor")
            self.grant(event,self.endpoint,flags=0)
        elif event.startswith("socket."):
            raise BootstrapError("foreign_socket_operation:"+event)
        elif event in ("os.listdir","os.scandir"):
            target = self.path(args[0])
            require(target==self.stdlib or (target.is_relative_to(self.stdlib) and not any(part in self.excluded for part in target.relative_to(self.stdlib).parts)) or target in {path.parent for path in self.read_paths},"unplanned_directory_read")
        elif event=="ctypes.dlopen":
            self.ctypes_process_namespace(args)
        elif event in {"subprocess.Popen","os.fork","os.forkpty","os.posix_spawn","os.spawn","os.system","os.exec","sqlite3.connect","sqlite3.load_extension","sqlite3.enable_load_extension","os.remove","os.rmdir","os.rename","os.link","os.symlink","os.truncate","os.chown","os.chflags","resource.setrlimit","sys.addaudithook","sys.setprofile","sys.settrace"}:
            raise BootstrapError("forbidden_native_operation:"+event)
        elif event in ("os.kill","os.killpg"):
            require(event=="os.kill" and args==(os.getpid(),0),"foreign_signal")
        elif event=="import":
            name = args[0]
            require(type(name) is str,"import_name_type")
            if name.split(".")[0] in NEURAL_PREFIXES:
                self.neural_imports.append(name)
                raise BootstrapError("neural_import_forbidden")

    def terminal_result(self,result: dict[str,object]) -> None:
        require(self.exchange_lock.acquire(timeout=0.1),"terminal_exchange_lock")
        try:
            self.terminal = True
            self.protocol.emit(result)
            answer = self.protocol.receive(LIMITS["frame_bytes"],0.1)
            self.policy.terminal_ACK_contract(answer)
            os._exit(73)
        finally:
            self.exchange_lock.release()


def cache_observations(server: object,vector: types.ModuleType,reranker: types.ModuleType) -> dict[str,object]:
    return {"vector_model":vector._model,"reranker_model":reranker._model,
            "server_embed_state":server._embed_state,"server_reranker":server._reranker,"server_fast_encoder":server._fast_encoder}


def listener_eligible(guard: AuditGuard,listener: socket.socket | None,pid_path: Path) -> bool:
    if listener is None or listener is not guard.listener or pid_path != guard.endpoint.with_name("model_server.pid"):
        return False
    try:
        descriptor = listener.fileno()
        if type(descriptor) is not int or descriptor < 3:
            return False
        endpoint = listener.getsockname()
        if type(endpoint) is not str or endpoint != str(guard.endpoint):
            return False
        info = guard.policy.owned_path(pid_path)
        return stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077 and info.st_nlink == 1
    except (OSError,ValueError):
        return False


def execute_case(manifest: dict[str,object],policy: types.ModuleType,runtime: dict[str,object],guard: AuditGuard) -> dict[str,object]:
    finder = policy.PinnedPackageFinder(runtime,full_package=True)
    sys.meta_path.insert(0,finder)
    sys.path[:] = [str(guard.stdlib),str(guard.stdlib/"lib-dynload")]
    for name in tuple(sys.modules):
        require(name.split(".")[0] not in NEURAL_PREFIXES,"neural_dependency_preloaded")
    package = importlib.import_module("truememory")
    require(package.__file__==str(SOURCE_ROOT/"truememory/__init__.py") and package.__version__=="0.7.6.2" and hasattr(package,"Memory") and hasattr(package,"TrueMemoryEngine"),"actual_initializer_missing")
    model_server = importlib.import_module("truememory.model_server")
    model_client = importlib.import_module("truememory.model_client")
    vector = importlib.import_module("truememory.vector_search")
    reranker = importlib.import_module("truememory.reranker")
    salience = importlib.import_module("truememory.salience")
    memory = importlib.import_module("truememory.memory_budget")
    for name,checksum in manifest["sources"].items():
        module_name = name[:-12].replace("/",".") if name.endswith("/__init__.py") else name[:-3].replace("/",".")
        require(module_name in sys.modules and sys.modules[module_name].__file__==str(SOURCE_ROOT/name),"source_closure_not_loaded")
    raw = policy.read_bounded(SOURCE_ROOT/"truememory/data/l3_weights.json",845,private=False)
    require(len(raw)==845 and policy.digest(raw)==manifest["data"]["truememory/data/l3_weights.json"],"source_data_changed")
    data = json.loads(raw)
    require(salience._L3_WEIGHTS==tuple(data["weights"]) and len(salience._L3_WEIGHTS)==13 and salience._L3_BIAS==float(data["bias"]),"actual_salience_data_missing")
    require(os.environ.get("TRUEMEMORY_NO_MODEL_SERVER")=="1" and model_server._USE_UNIX is True and model_client._USE_UNIX is True and model_server.SOCK_PATH==guard.endpoint and model_client.SOCK_PATH==guard.endpoint,"owned_AF_UNIX_paths")
    server = model_server.ModelServer(automatic_rss_budget=True)
    before = cache_observations(server,vector,reranker)
    require(all(value is None for value in before.values()),"pre_ping_neural_cache")
    runner = threading.Thread(target=server.run,name="native-package-server-v1",daemon=True)
    runner.start()
    ready_deadline = time.monotonic()+0.5
    while True:
        require(runner.is_alive(),"server_thread_stopped")
        listener = guard.listener
        if listener_eligible(guard,listener,model_server.PID_PATH):
            break
        require(time.monotonic()<ready_deadline,"owned_listener_deadline")
        time.sleep(0.001)
    ping = model_client._send_request({"op":"ping"},timeout=0.2)
    require(ping=={"ok":True,"protocol":1} and type(ping["protocol"]) is int,"actual_ping_mismatch")
    terminal_deadline = time.monotonic()+0.2
    while True:
        snapshot = memory.get_process_registry().snapshot()
        if snapshot["lease_count"]==0 and snapshot["reserved_known_bytes"]==0 and not server._clients and server._inflight==0:
            break
        require(time.monotonic()<terminal_deadline,"request_leases_nonterminal")
        time.sleep(0.001)
    after = cache_observations(server,vector,reranker)
    require(all(value is None for value in after.values()),"post_ping_neural_cache")
    require(guard.accepted_checked and guard.socket_count==3 and guard.listener is not None and guard.listener.getsockname()==str(guard.endpoint),"actual_accept_authority_missing")
    socket_info = policy.socket_owned(guard.endpoint)
    lock_info = policy.owned_path(model_server.LOCK_PATH)
    require(type(server._lock_fd) is int,"bind_lock_descriptor_missing")
    held = os.fstat(server._lock_fd)
    require((held.st_dev,held.st_ino)==(lock_info.st_dev,lock_info.st_ino),"held_lock_identity")
    pid_info = policy.owned_path(model_server.PID_PATH)
    require(policy.read_bounded(model_server.PID_PATH,32)==str(os.getpid()).encode("ascii"),"actual_PID_digits")
    require(is_dataclass(snapshot["os_enforcement"]),"actual_capability_not_dataclass")
    registry = dict(snapshot)
    registry["os_enforcement"] = asdict(snapshot["os_enforcement"])
    registry = json.loads(encoded(registry))
    details = {"package_version":package.__version__,"initializer_executed":True,"paths_exact":True,"ping":ping,"timeout_ms":200,"source_data_loaded":True,"neural_imports":list(guard.neural_imports),"caches_before":before,"caches_after":after,"socket_identity":{"device":socket_info.st_dev,"inode":socket_info.st_ino},"socket_uid":socket_info.st_uid,"socket_mode":stat.S_IMODE(socket_info.st_mode),"pid_file":{"pid":os.getpid(),"device":pid_info.st_dev,"inode":pid_info.st_ino,"mode":stat.S_IMODE(pid_info.st_mode)},"lock_identity":{"device":lock_info.st_dev,"inode":lock_info.st_ino},"bind_lock_held":True,"registry":registry}
    result = {"type":"result","version":1,"case":"package-ping","passed":True,"proof_scope":PROOF_SCOPE,"import_mode":"full_source_package","source_commit":manifest["source_commit"],"sources_sha256":manifest["sources_sha256"],"source_count":37,"data_sha256":manifest["data_sha256"],"runtime_packet_sha256":manifest["runtime_packet_sha256"],"interpreter_packet_sha256":manifest["interpreter_packet_sha256"],"details":details}
    policy.result_contract(result,manifest,os.getpid(),os.getuid())
    return result


def main() -> int:
    readback = arm_limits()
    protocol = Protocol()
    request = protocol.receive(LIMITS["input_bytes"],1)
    require(set(request)=={"version","manifest"} and type(request["version"]) is int and request["version"]==1,"request_shape")
    manifest = request["manifest"]
    require(type(manifest) is dict and manifest.get("home")==str(RUN_ROOT/"home") and os.environ.get("HOME")==manifest["home"],"request_HOME")
    policy = compile_module("mac_native_package_policy_v4",HARNESS_ROOT/"tests/native/mac-native-package-acceptance-v4.py",manifest["supervisor_sha256"])
    policy.validate_manifest(manifest)
    runtime = policy.load_runtime_contract()
    guard = AuditGuard(protocol,manifest,policy,runtime)
    sys.addaudithook(guard)
    sys.audit("truememory.native.audit_probe","fixed-stage-B-v1")
    require(guard.activated,"audit_hook_missing")
    protocol.emit({"type":"ready","version":1,"pid":os.getpid(),"limits":readback,"audit_installed":True})
    result = execute_case(manifest,policy,runtime,guard)
    guard.terminal_result(result)
    raise BootstrapError("terminal_exit_missing")


if __name__=="__main__":
    raise SystemExit(main())
