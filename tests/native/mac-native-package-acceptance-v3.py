"""ROOT-run Stage B supervisor. Importing definitions never starts native work.

This instrumented full-source harness proves one owned AF_UNIX package ping.
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

RUN_ROOT = Path("/Users/j/dev/truememory-maintenance-v1/.work/native-mac-v1/run-0002")
HARNESS_ROOT = Path("/Users/j/dev/truememory-maintenance-v1/.work/worktrees/mac-native-acceptance-v1")
SOURCE_ROOT = Path("/Users/j/dev/truememory-maintenance-v1/.work/worktrees/whole-process-memory-297-v1")
RUNTIME_PACKET = Path("/Users/j/dev/truememory-maintenance-v1/.work/reports/mac-native-package-runtime-pins-v2.json")
RUNTIME_PACKET_SHA = "9c7ed6eda037c200530fbb913352006886c312d4011dac5e82d1805d3ead69aa"
PROOF_SCOPE = "instrumented_unmodified_full_package_source_af_unix_ping"
SOURCE_HASHES = {'truememory/__init__.py': '00d3ab14540092b63b449d14533b17292018ae4a69cd3f3ef1345e1a02b54d98', 'truememory/_platform.py': '0688763a6fc26ef8db2b87866aabd5a2373a18f4ee735859cc6d595147e310c0', 'truememory/client.py': 'f020ac930617cd5d4848775ddd7390a61665e0fbdc62d51ccaf19bccdeeb3a33', 'truememory/clustering.py': 'ea9c2dbb7d083217983bb962f696e7bfcf08359ba5969f916fa63241baa535a5', 'truememory/consolidation.py': 'dcff2bf4c8f80315453b5078cdae575e4414adf879b69ef8f7983e72478fd474', 'truememory/embedding_target.py': 'da61626141f1ca5587eaa2fa3c27381f80834e79bc32351b5d7a3c20e86c7ddc', 'truememory/engine.py': '898f2b8dd79e8ca18154f8e4b3ef0cbfe785c27b543719f975f4a1f2cf0a010c', 'truememory/fts_search.py': '9c954bd2513ea8e3c82b33e0fbc311cd5b7346d12ee70da011df3f4244dc4389', 'truememory/hybrid.py': '17b281113b2927c6e63088c0d48dc34c2681a9943e1d4f4aa4d0a70d1c36dd17', 'truememory/hyde.py': '127e4d1c01767e13c0d963d6878f01a7329bb749b4510e831ef51c9c879276e4', 'truememory/maintenance.py': '413ffbc08f92ba3ed331cab5e0aebd67ba2cadf2e5400f8b5c09c47f80521819', 'truememory/memory_budget.py': '2d2578554d1c029c5205ac1fd27511f30e3d372a1178bf88cb7b81b0501409a4', 'truememory/model_client.py': 'a8a9332ab7cf532c98594f4f7455642f00d6235d9ca8a4552a75c68a49b7d0f3', 'truememory/model_server.py': '424986a108d0ecd9e984ed77e352fba36305bf9aa3d79517de8fd1cfa4f8701b', 'truememory/personality.py': '4a55a594360435bd56152738af061988b73e3374c3e8757e934c0b88cf4cd2af', 'truememory/personality_style_vec.py': 'f294e67b46ce4473b2cf15a49c80fc348de27f5236834dfbec832fc092c5362e', 'truememory/predictive.py': '88ae1a2f8436b305855a5884e3ac5ebe435106808533dc0bc563d2054d374c41', 'truememory/query_classifier.py': '52f1f3bfe5a75f45eb5aef389b490953422bf1d8fb03c5181bff816440bf2476', 'truememory/rebuild_source.py': 'e46639a50d35b711bc0c0b0de9e7a90652b1386b25356c14d095551673c7bd76', 'truememory/reranker.py': 'c02740b54246ca9f52fd6df90d424f18d3417c36734346ede128c91d3c66e90d', 'truememory/salience.py': 'a005256ecc20c44ad7d4825a11fafa2c2b5951a762450dfce9fd80ec7e12940e', 'truememory/storage.py': 'adc89850a897a2f9b478c8a23ad62fa438d7a214405ea21df886e029cdf4481a', 'truememory/temporal.py': '52e98b3924573db6a209ef27d0318e254e5c976a7a65d75762f901d712505a65', 'truememory/tier_config.py': 'a8e690000c74f7629690c67fb99e34d4bd87f02b78795fb68ec2009f18348489', 'truememory/tier_switch/__init__.py': '2333416f4985986b9fcf6558478f5c3cf9f75c277961c14902162f8f11ea1705', 'truememory/tier_switch/activation.py': '24b820b0ff129fb42b9c13544e42b7c4b2ec8397b427d23b61c60c3e8ff19d47', 'truememory/tier_switch/cache.py': '77bc24c176763629f83d199cc588eca9a9e7432e36e29584a2bd3d475ac314b4', 'truememory/tier_switch/job.py': 'c1def76a9510c3cd600ec11737ee068d71cedae2bc980048fba6c70bdee7a1f0', 'truememory/tier_switch/manager.py': '160a61ff2b2a4b2a7fdeb3c7fa2442ce161f5b1a426735ea3cac53829b6b8c4b', 'truememory/tier_switch/runtime.py': '861893308506c0189e50eea2d21a9da43b2c7be0eb90c56f8fd62e6bb40dc5be', 'truememory/tier_switch/sensors.py': '6471176f75ab5ddce3aa523340a6567ec24f9243c3551f4203baa8ea2f15b55b', 'truememory/tier_switch/serving.py': '25c1524a02affc127e170c16ceee0f61c2b3b221e28c5a19e9fb347aec57bef7', 'truememory/tier_switch/source.py': 'f8ffe9cd5890c5e6520ca52cfe20bad49cb789d2222e29fffc4300bdbc64f227', 'truememory/tier_switch/state_machine.py': '39ccf8d3e944728f7bf18e6efe10d0e8f9db625f9d812220554dd980a932d1b9', 'truememory/tier_switch/throttler.py': '3416ad7681a22b2478e9e4375041f7d2d364b474a81255103e11abe79d5b5f32', 'truememory/tier_switch/worker.py': '8bae8daf54ab06c0693df46578e340cbde874dda0dccfdb5bcfc2a30d3081ef7', 'truememory/vector_search.py': '2acd58a0dd858cc6a3a8c69117c1666f5e74d166ed9aeb57b81da4da45804739'}
SOURCE_FILES = tuple(SOURCE_HASHES)
SOURCE_BYTES = {'truememory/__init__.py': 6932, 'truememory/_platform.py': 5947, 'truememory/client.py': 13639, 'truememory/clustering.py': 22412, 'truememory/consolidation.py': 62951, 'truememory/embedding_target.py': 5225, 'truememory/engine.py': 174355, 'truememory/fts_search.py': 10285, 'truememory/hybrid.py': 12551, 'truememory/hyde.py': 8143, 'truememory/maintenance.py': 137455, 'truememory/memory_budget.py': 41094, 'truememory/model_client.py': 34164, 'truememory/model_server.py': 100943, 'truememory/personality.py': 68823, 'truememory/personality_style_vec.py': 21995, 'truememory/predictive.py': 18593, 'truememory/query_classifier.py': 5045, 'truememory/rebuild_source.py': 19160, 'truememory/reranker.py': 25401, 'truememory/salience.py': 20238, 'truememory/storage.py': 68710, 'truememory/temporal.py': 49851, 'truememory/tier_config.py': 6988, 'truememory/tier_switch/__init__.py': 319, 'truememory/tier_switch/activation.py': 33404, 'truememory/tier_switch/cache.py': 7779, 'truememory/tier_switch/job.py': 25028, 'truememory/tier_switch/manager.py': 47952, 'truememory/tier_switch/runtime.py': 38531, 'truememory/tier_switch/sensors.py': 4336, 'truememory/tier_switch/serving.py': 18642, 'truememory/tier_switch/source.py': 28269, 'truememory/tier_switch/state_machine.py': 5509, 'truememory/tier_switch/throttler.py': 14364, 'truememory/tier_switch/worker.py': 27782, 'truememory/vector_search.py': 96703}
DATA_HASHES = {'truememory/data/l3_weights.json': '5d91643f843b4639375a4324aa87cbd09815a4242b11f3fe4138bae641b2d5b8'}
SOURCE_COMMIT = 'aeecf25f78279629789d98a29bbd8bf817233391'
SOURCE_MAP_SHA = 'e1cbacd6cdfa558110647572b614d0d3983413af3dc988530a30b56d6a06cbd6'
DATA_MAP_SHA = '19d20af55c695fe0952cfc26e4fd339037ac3bbfe05fab4b14bb1ae814ca4ea2'
CASES = frozenset({"package-ping"})
B_OS_THREAD_MAX = 4
INTERPRETER_PACKET = RUNTIME_PACKET.with_name("mac-native-interpreter-runtime-v1.json")
INTERPRETER_PACKET_SHA = "73a77a325ba667ff84d395c8ecddd9d21d63be27914c4586c6d5e65e2ca55596"
LINKAGE_PACKET = RUNTIME_PACKET.with_name("mac-native-interpreter-linkage-v1.json")
LINKAGE_PACKET_SHA = "762d535cdb9f021729fe101c654966e52a9d8da285d5fe77bc0b1db815d43843"
ENDPOINT = RUN_ROOT / "home/.truememory/model.sock"
TERMINAL_ACK = {"type":"result_ack","version":1,"case":"package-ping"}
CACHE_KEYS = frozenset({"vector_model","reranker_model","server_embed_state","server_reranker","server_fast_encoder"})
UNKNOWN_STAGES = ('client frame receive', 'client request measure', 'client request serialize', 'client response materialize', 'frame receive', 'request materialize', 'response metadata', 'response serialize')
CAPABILITY = {'evidence_sha256': [], 'limit_bytes': None, 'metric': 'none', 'physical_hard_cap': False, 'reason': 'No native enforcement witness has been accepted', 'scope': None, 'status': 'unsupported'}
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








def case_contract(value: object) -> dict[str, object]:
    row = exact_keys(value, {"id"})
    if type(row["id"]) is not str or row["id"] != "package-ping":
        raise SafetyError("unsupported_B_case")
    return row



def validate_manifest(value: object) -> dict[str, object]:
    row = exact_keys(value, {"version", "stage", "synthetic", "root_execution_go", "run_root", "home", "source_root", "source_commit", "sources", "sources_sha256", "data", "data_sha256", "runtime_packet_sha256", "interpreter_packet_sha256", "supervisor_sha256", "bootstrap_sha256", "limits", "cases"})
    if type(row["version"]) is not int or row["version"] != 1 or row["stage"] != "B" or row["synthetic"] is not True or row["root_execution_go"] is not True:
        raise SafetyError("manifest_authority_or_version")
    if any(type(row[name]) is not str for name in ("stage","run_root","home","source_root","source_commit","sources_sha256","data_sha256","runtime_packet_sha256","interpreter_packet_sha256")):
        raise SafetyError("manifest_string_type")
    if row["run_root"] != str(RUN_ROOT) or row["home"] != str(RUN_ROOT / "home") or row["source_root"] != str(SOURCE_ROOT) or row["source_commit"] != SOURCE_COMMIT:
        raise SafetyError("unexpected_root_or_source")
    if type(row["sources"]) is not dict or row["sources"] != SOURCE_HASHES or digest(encoded(row["sources"])) != SOURCE_MAP_SHA or row["sources_sha256"] != SOURCE_MAP_SHA:
        raise SafetyError("source_map_mismatch")
    if type(row["data"]) is not dict or row["data"] != DATA_HASHES or digest(encoded(row["data"])) != DATA_MAP_SHA or row["data_sha256"] != DATA_MAP_SHA:
        raise SafetyError("data_map_mismatch")
    if row["runtime_packet_sha256"] != RUNTIME_PACKET_SHA or row["interpreter_packet_sha256"] != INTERPRETER_PACKET_SHA:
        raise SafetyError("runtime_packet_mismatch")
    hash_text(row["supervisor_sha256"])
    hash_text(row["bootstrap_sha256"])
    if type(row["limits"]) is not dict or row["limits"] != LIMITS or any(type(v) is not int for v in row["limits"].values()):
        raise SafetyError("limit_change")
    if type(row["cases"]) is not list or len(row["cases"]) != 1:
        raise SafetyError("case_count")
    case_contract(row["cases"][0])
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


def result_contract(value: object, manifest: dict[str, object], child_pid: int, worker_uid: int) -> dict[str, object]:
    integer(child_pid, minimum=1)
    integer(worker_uid)
    validate_manifest(manifest)
    row = exact_keys(value, {"type", "version", "case", "passed", "proof_scope", "import_mode", "source_commit", "sources_sha256", "source_count", "data_sha256", "runtime_packet_sha256", "interpreter_packet_sha256", "details"})
    expected = {"type": "result", "version": 1, "case": "package-ping", "passed": True, "proof_scope": PROOF_SCOPE, "import_mode": "full_source_package", "source_commit": SOURCE_COMMIT, "sources_sha256": SOURCE_MAP_SHA, "source_count": 37, "data_sha256": DATA_MAP_SHA, "runtime_packet_sha256": RUNTIME_PACKET_SHA, "interpreter_packet_sha256": INTERPRETER_PACKET_SHA}
    for name, expected_value in expected.items():
        if type(row[name]) is not type(expected_value) or row[name] != expected_value:
            raise SafetyError("result_scope_mismatch")
    detail = exact_keys(row["details"], {"package_version", "initializer_executed", "paths_exact", "ping", "timeout_ms", "source_data_loaded", "neural_imports", "caches_before", "caches_after", "socket_identity", "socket_uid", "socket_mode", "pid_file", "lock_identity", "bind_lock_held", "registry"})
    if detail["package_version"] != "0.7.6.2" or type(detail["package_version"]) is not str or detail["neural_imports"] != [] or type(detail["neural_imports"]) is not list:
        raise SafetyError("package_or_neural_state")
    for name in ("initializer_executed", "paths_exact", "source_data_loaded", "bind_lock_held"):
        if detail[name] is not True:
            raise SafetyError("source_observation_missing")
    ping = exact_keys(detail["ping"], {"ok", "protocol"})
    if ping["ok"] is not True or type(ping["protocol"]) is not int or ping["protocol"] != 1 or type(detail["timeout_ms"]) is not int or detail["timeout_ms"] != 200:
        raise SafetyError("ping_mismatch")
    for name in ("caches_before", "caches_after"):
        cache = exact_keys(detail[name], set(CACHE_KEYS))
        if any(value is not None for value in cache.values()):
            raise SafetyError("neural_cache_present")
    for name in ("socket_identity", "lock_identity"):
        identity = exact_keys(detail[name], {"device", "inode"})
        integer(identity["device"])
        integer(identity["inode"], minimum=1)
    if integer(detail["socket_uid"]) != worker_uid or integer(detail["socket_mode"]) != 0o600:
        raise SafetyError("socket_owner_mode")
    pid = exact_keys(detail["pid_file"], {"pid", "device", "inode", "mode"})
    if integer(pid["pid"], minimum=1) != child_pid or integer(pid["mode"]) != 0o600:
        raise SafetyError("pid_file_mismatch")
    integer(pid["device"])
    integer(pid["inode"], minimum=1)
    registry = exact_keys(detail["registry"], {"pid", "metric", "effective_limit_bytes", "owner_count", "closing_owner_count", "lease_count", "reserved_known_bytes", "high_water_known_bytes", "unprofiled_stages", "coverage_truncated", "physical_hard_cap", "os_enforcement"})
    if integer(registry["pid"], minimum=1) != child_pid or (type(registry["metric"]) is not str or registry["metric"] != "process_rss_sampled"):
        raise SafetyError("registry_identity")
    limit = integer(registry["effective_limit_bytes"], minimum=1, maximum=8 * 1024**3)
    if integer(registry["owner_count"]) != 1 or any(integer(registry[name]) != 0 for name in ("closing_owner_count", "lease_count", "reserved_known_bytes")):
        raise SafetyError("registry_nonterminal")
    integer(registry["high_water_known_bytes"], maximum=limit)
    unknown = registry["unprofiled_stages"]
    if type(unknown) is not list or tuple(unknown) != UNKNOWN_STAGES or any(type(item) is not str for item in unknown):
        raise SafetyError("unknown_coverage_mismatch")
    if registry["coverage_truncated"] is not False or registry["physical_hard_cap"] is not False:
        raise SafetyError("unsupported_coverage_claim")
    capability = exact_keys(registry["os_enforcement"], set(CAPABILITY))
    if capability != CAPABILITY or any(type(capability[name]) is not type(expected_value) for name, expected_value in CAPABILITY.items()):
        raise SafetyError("unsupported_enforcement_claim")
    if len(encoded(row)) + 1 > LIMITS["frame_bytes"]:
        raise SafetyError("result_frame_limit")
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
    def __init__(self, case: dict[str, object]) -> None:
        case_contract(case)
        self.sequence = 0
        self.socket_objects = 0
        self.model_dir = RUN_ROOT / "home/.truememory"
        self.lock = self.model_dir / "model_server.lock"
        self.pid = self.model_dir / "model_server.pid"
        self.bound = False
        self.connected = False

    def decide(self, value: object) -> dict[str, object]:
        row = exact_keys(value, {"type", "version", "seq", "event", "path", "source", "flags"})
        if row["type"] != "mutation" or type(row["version"]) is not int or row["version"] != 1 or row["source"] is not None:
            raise SafetyError("mutation_shape")
        seq = integer(row["seq"], minimum=1, maximum=LIMITS["events"])
        if seq != self.sequence + 1:
            raise SafetyError("mutation_sequence")
        path = canonical_path(row["path"], RUN_ROOT)
        flags = integer(row["flags"], maximum=(1 << 31) - 1)
        event = row["event"]
        if event == "os.mkdir" and path == self.model_dir and flags == 0o777:
            purpose = "Fresh private synthetic model-server directory under child umask077"
        elif event == "os.chmod" and ((path == self.model_dir and flags == 0o700) or (path == ENDPOINT and flags == 0o600 and self.bound)):
            purpose = "Owned synthetic endpoint permissions"
        elif event == "open" and path in (self.lock, self.pid):
            ordinary = flags & ~getattr(os, "O_CLOEXEC", 0)
            expected = os.O_CREAT | (os.O_RDWR if path == self.lock else os.O_WRONLY | os.O_TRUNC)
            if ordinary != expected:
                raise SafetyError("producer_open_flags")
            purpose = "Fresh retained bind lock" if path == self.lock else "Fresh owned PID digits without newline"
        elif event == "socket.__new__" and path == ENDPOINT and flags == 0 and self.socket_objects < 3:
            self.socket_objects += 1
            purpose = "Owned AF_UNIX listener/client/accepted socket wrapper"
        elif event == "socket.bind" and path == ENDPOINT and flags == 0 and not self.bound:
            self.bound = True
            purpose = "Bind fresh private AF_UNIX endpoint"
        elif event == "socket.connect" and path == ENDPOINT and flags == 0 and self.bound and not self.connected:
            self.connected = True
            purpose = "Single synthetic client connection to owned endpoint"
        else:
            raise SafetyError("unplanned_B_mutation")
        self.sequence = seq
        return {"type":"grant", "version":1, "seq":seq, "path":str(path), "purpose":purpose}



def mutation_target(grant: dict[str, object], request: dict[str, object]) -> None:
    path = Path(grant["path"])
    event = request["event"]
    owned_path(path.parent, directory=True)
    exists = path.exists() or path.is_symlink()
    if event in ("open", "os.mkdir", "socket.bind") and exists:
        raise SafetyError("retained_artifact_reuse_forbidden")
    if event == "os.chmod":
        if path == RUN_ROOT / "home/.truememory":
            owned_path(path, directory=True)
        else:
            socket_owned(path,pre_chmod=True)
    elif event == "socket.connect":
        socket_owned(path)


def socket_owned(path: Path,*,pre_chmod: bool = False) -> os.stat_result:
    if path.resolve() != path or any(part.is_symlink() for part in (path, *path.parents)):
        raise SafetyError("socket_path_link")
    info = path.lstat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) not in ((0o700,0o600) if pre_chmod else (0o600,)):
        raise SafetyError("socket_not_private_owned")
    return info


def verify_artifacts(value: dict[str, object], child_pid: int) -> None:
    detail = value["details"]
    socket_info = socket_owned(ENDPOINT)
    lock_info = owned_path(ENDPOINT.with_name("model_server.lock"))
    pid_path = ENDPOINT.with_name("model_server.pid")
    pid_info = owned_path(pid_path)
    if read_bounded(pid_path, 32) != str(child_pid).encode("ascii"):
        raise SafetyError("actual_PID_bytes")
    for name, info in (("socket_identity", socket_info), ("lock_identity", lock_info)):
        if detail[name] != {"device":info.st_dev, "inode":info.st_ino}:
            raise SafetyError("actual_artifact_identity")
    if detail["pid_file"] != {"pid":child_pid,"device":pid_info.st_dev,"inode":pid_info.st_ino,"mode":stat.S_IMODE(pid_info.st_mode)}:
        raise SafetyError("actual_PID_identity")



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


def runtime_contract(package_raw: bytes, interpreter_raw: bytes, linkage_raw: bytes) -> dict[str, object]:
    values: list[dict[str, object]] = []
    for raw, checksum in ((package_raw, RUNTIME_PACKET_SHA), (interpreter_raw, INTERPRETER_PACKET_SHA), (linkage_raw, LINKAGE_PACKET_SHA)):
        if type(raw) is not bytes or not 0 < len(raw) <= 65536 or digest(raw) != checksum:
            raise SafetyError("runtime_packet_changed_or_limit")
        value = json.loads(raw)
        if type(value) is not dict:
            raise SafetyError("runtime_packet_shape")
        values.append(value)
    package, interpreter, linkage = values
    if type(package.get("schema")) is not int or package["schema"] != 2 or package["sources"] != SOURCE_HASHES or package["source_bytes"] != SOURCE_BYTES or package["source_commit"] != SOURCE_COMMIT or package["source_root"] != str(SOURCE_ROOT) or package["data"] != DATA_HASHES:
        raise SafetyError("runtime_source_shape")
    if package["source_map_sha256"] != SOURCE_MAP_SHA or package["data_map_sha256"] != DATA_MAP_SHA or package["interpreter_runtime_packet_sha256"] != INTERPRETER_PACKET_SHA or package["interpreter_linkage_packet_sha256"] != LINKAGE_PACKET_SHA:
        raise SafetyError("runtime_linkage_mismatch")
    if package["numpy_version"] != "2.4.6" or len(package["numpy_files"]) != 193 or len(package["psutil_pins"]) != 11 or len(interpreter["exact_content_files"]) != 84 or len(linkage["trusted_system_loader_names"]) != 13:
        raise SafetyError("runtime_inventory_mismatch")
    if package["staged_wheel_sha256"] != "f723b880e455fea1545c31a9b7fe01def1902af3a36609cfa084fc9548296c51" or package["runtime_closure_complete"] is not False or interpreter["runtime_closure_complete"] is not False or package["stdlib_complete_dependency_bytes_hashed"] is not False or package["trusted_OS_sharedcache_bytes_hashed"] is not False or linkage["sharedcache_bytes_hashed"] is not False:
        raise SafetyError("unproved_runtime_claim")
    stdlib = interpreter["trusted_stdlib"]
    if stdlib["canonical_root"] != "/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12" or stdlib["python_sources_byte_hashed"] is not False or stdlib["excluded_subtrees"] != ["site-packages","test","tests","idlelib","turtledemo","ensurepip","venv"]:
        raise SafetyError("stdlib_trust_mismatch")
    return {"package":package,"interpreter":interpreter,"linkage":linkage}


def load_runtime_contract() -> dict[str, object]:
    return runtime_contract(read_bounded(RUNTIME_PACKET,65536,private=False),read_bounded(INTERPRETER_PACKET,65536,private=False),read_bounded(LINKAGE_PACKET,65536,private=False))


def native_file_sha256(path: Path, pin: dict[str, object]) -> str:
    size = integer(pin["bytes"], maximum=64 * 1024**2)
    checksum = hash_text(pin["sha256"])
    if path.resolve() != path or path.is_symlink():
        raise SafetyError("native_path_link")
    fd = os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != size or any(key in pin and pin[key] != getattr(before,"st_dev" if key=="device" else "st_ino") for key in ("device","inode")):
            raise SafetyError("native_file_type_or_size")
        hasher = hashlib.sha256()
        count = 0
        while count <= size:
            data = os.read(fd,min(65536,size-count+1))
            if not data:
                break
            hasher.update(data)
            count += len(data)
        after = os.fstat(fd)
        if count != size or file_identity(before) != file_identity(after) or hasher.hexdigest() != checksum:
            raise SafetyError("native_file_changed")
        return checksum
    finally:
        os.close(fd)


def runtime_files(runtime: dict[str, object]) -> dict[Path, dict[str, object]]:
    package, interpreter = runtime["package"], runtime["interpreter"]
    files: dict[Path, dict[str, object]] = {}
    groups = [package["numpy_files"],package["psutil_pins"],package["numpy_metadata"],interpreter["exact_content_files"]]
    for group in groups:
        for name, pin in group.items():
            path = Path(name)
            if not path.is_absolute() or str(path) != name or ".." in path.parts:
                raise SafetyError("runtime_file_path")
            if path in files and any(files[path][key] != pin[key] for key in ("bytes","sha256")):
                raise SafetyError("runtime_overlapping_pin")
            integer(pin["bytes"],maximum=64*1024**2)
            hash_text(pin["sha256"])
            files[path] = pin
    metadata = package["psutil_metadata"]
    files[Path(metadata["path"])] = metadata
    for name,pin in package["staged_truememory_distribution_metadata"].items():
        path = Path("/Users/j/dev/truememory-maintenance-v1") / name
        canonical_path(str(path),Path("/Users/j/dev/truememory-maintenance-v1/.work/artifacts/memory-installed-wheel-v2/metadata"))
        files[path] = pin
    return files


class PinnedPythonLoader(importlib.abc.Loader):
    def __init__(self,path: Path,pin: dict[str, object]) -> None:
        self.path,self.pin = path,pin

    def create_module(self,spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self,module: ModuleType) -> None:
        raw = read_bounded(self.path,262144,private=False)
        if len(raw) != self.pin["bytes"] or digest(raw) != self.pin["sha256"]:
            raise SafetyError("python_source_hash_changed")
        exec(compile(raw,str(self.path),"exec",dont_inherit=True),module.__dict__)


class TrustedStdlibLoader(importlib.abc.Loader):
    """Compile trusted-unhashed stdlib Python bytes; ignore bytecode caches."""
    def __init__(self,path: Path) -> None:
        self.path = path

    def create_module(self,spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self,module: ModuleType) -> None:
        raw = read_bounded(self.path,262144,private=False)
        exec(compile(raw,str(self.path),"exec",dont_inherit=True),module.__dict__)


class PinnedPackageFinder(importlib.abc.MetaPathFinder):
    def __init__(self,runtime: dict[str, object],*,full_package: bool) -> None:
        self.runtime,self.full_package = runtime,full_package
        self.files = runtime_files(runtime)
        self.modules: dict[str,tuple[Path,dict[str,object],bool]] = {}
        groups = [runtime["package"]["psutil_pins"]]
        if full_package:
            groups += [runtime["package"]["numpy_files"],{str(SOURCE_ROOT/name):{"bytes":SOURCE_BYTES[name],"sha256":checksum} for name,checksum in SOURCE_HASHES.items()}]
        for group in groups:
            for name,pin in group.items():
                path = Path(name)
                if path.is_relative_to(SOURCE_ROOT):
                    parts = path.relative_to(SOURCE_ROOT).parts
                else:
                    prefix = next(part for part in ("numpy","psutil") if part in path.parts)
                    parts = path.parts[path.parts.index(prefix):]
                leaf = parts[-1]
                package = leaf == "__init__.py"
                stem = leaf.split(".")[0]
                module = ".".join(parts[:-1] if package else (*parts[:-1],stem))
                self.modules[module] = (path,pin,package)
        self.stdlib = Path(runtime["interpreter"]["trusted_stdlib"]["canonical_root"])
        self.excluded = set(runtime["interpreter"]["trusted_stdlib"]["excluded_subtrees"])

    def find_spec(self,fullname: str,path: object = None,target: object = None) -> importlib.machinery.ModuleSpec | None:
        prefix = fullname.split(".")[0]
        if prefix in ("truememory","numpy","psutil"):
            if fullname not in self.modules:
                raise ImportError("Unannounced package module: " + fullname)
            source,pin,package = self.modules[fullname]
            if source.suffix == ".so":
                native_file_sha256(source,pin)
                loader = importlib.machinery.ExtensionFileLoader(fullname,str(source))
            else:
                loader = PinnedPythonLoader(source,pin)
            return importlib.util.spec_from_file_location(fullname,str(source),loader=loader,submodule_search_locations=[str(source.parent)] if package else None)
        if importlib.machinery.BuiltinImporter.find_spec(fullname) is not None or importlib.machinery.FrozenImporter.find_spec(fullname) is not None:
            return None
        search = path if path is not None else [str(self.stdlib),str(self.stdlib/"lib-dynload")]
        spec = importlib.machinery.PathFinder.find_spec(fullname,search)
        if spec is None or type(spec.origin) is not str:
            raise ImportError("Unannounced runtime module: " + fullname)
        source = Path(spec.origin)
        if source.resolve() != source or not source.is_relative_to(self.stdlib) or any(part in self.excluded for part in source.relative_to(self.stdlib).parts):
            raise ImportError("Untrusted runtime origin: " + fullname)
        if source.suffix == ".so":
            if source not in self.files:
                raise ImportError("Unpinned extension: " + fullname)
            native_file_sha256(source,self.files[source])
        elif source.suffix == ".py":
            spec.loader = TrustedStdlibLoader(source)
            spec.cached = None
        else:
            raise ImportError("Untrusted runtime type: " + fullname)
        return spec

    def find_distributions(self,context: object = None) -> list[object]:
        if not self.full_package or context is None or getattr(context,"name",None) != "truememory":
            return []
        from importlib.metadata import PathDistribution
        root = Path("/Users/j/dev/truememory-maintenance-v1/.work/artifacts/memory-installed-wheel-v2/metadata")
        return [PathDistribution(root)]


def pinned_runtime() -> ModuleType:
    runtime = load_runtime_contract()
    package,_interpreter = runtime["package"],runtime["interpreter"]
    if Path(sys.executable).resolve() != Path(package["python_realpath"]) or sys.version_info[:3] != (3,12,12):
        raise SafetyError("interpreter_mismatch")
    for path,pin in runtime_files(runtime).items():
        native_file_sha256(path,pin)
    if "psutil" in sys.modules:
        raise SafetyError("dependency_preloaded")
    sys.meta_path.insert(0,PinnedPackageFinder(runtime,full_package=False))
    import psutil
    if psutil.__version__ != "7.2.2":
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
    for sig, interval in ((signal.SIGKILL, 0.99),):
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


def thread_sample_contract(value: object) -> int:
    count = integer(value,minimum=1)
    if count>B_OS_THREAD_MAX:
        raise SafetyError("sampled_thread_limit")
    return count


def terminal_ACK_contract(value: object) -> dict[str, object]:
    row = exact_keys(value,set(TERMINAL_ACK))
    if row["type"]!="result_ack" or type(row["version"]) is not int or row["version"]!=1 or row["case"]!="package-ping":
        raise SafetyError("terminal_ACK_mismatch")
    return row


def execute_case(manifest: dict, case: dict, psutil: ModuleType, pressure: PressureReader, attempt_version: str = "v1") -> dict:
    attempt_version = validate_attempt_version(attempt_version)
    bootstrap = HARNESS_ROOT / "tests/native/mac-native-package-bootstrap-v3.py"
    policy = MutationPolicy(case)
    records: list[Sensor] = []
    result = {"case": case["id"], "implemented": True, "executed": False, "passed": False,
              "blocked": None, "unproven": ["installed_Mac_package", "installed_host", "models", "retrieval", "physical_hard_cap", "complete_runtime_bytes", "native_sandbox"],
              "proof_scope": PROOF_SCOPE, "import_mode": "full_source_package", "synthetic": True,
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
        case_contract(case)
        if ENDPOINT.parent.exists() or ENDPOINT.parent.is_symlink():
            raise SafetyError("retained_endpoint_artifacts_forbidden")
        load_runtime_contract()
        if digest(read_bounded(SOURCE_ROOT/"truememory/data/l3_weights.json",845,private=False)) != DATA_HASHES["truememory/data/l3_weights.json"]:
            raise SafetyError("source_data_changed")
        for name, checksum in manifest["sources"].items():
            if digest(read_bounded(SOURCE_ROOT / name, 262144, private=False)) != checksum:
                raise SafetyError("source_hash_changed")
        if digest(read_bounded(bootstrap, 262144, private=False)) != manifest["bootstrap_sha256"]:
            raise SafetyError("bootstrap_hash_changed")
        result["sensors"] = [asdict(row) for row in records]
        for channel in ("stdout", "stderr", "events"):
            logs[channel] = new_log(RUN_ROOT / "logs" / (case["id"] + "-" + channel + "-" + attempt_version + ".log"), "Retained native " + channel + " evidence")
        env = {"HOME": manifest["home"], "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "TRUEMEMORY_NO_MODEL_SERVER": "1"}
        request = encoded({"version":1,"manifest":manifest}) + b"\n"
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
                threads = thread_sample_contract(process.num_threads())
                result["threads_sampled_peak"] = max(result.get("threads_sampled_peak",0),threads)
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
                    elif frame.get("type") == "result" and protocol_result is None:
                        if not ready:
                            raise SafetyError("result_before_ready")
                        protocol_result = result_contract(frame,manifest,child.pid,os.getuid())
                        verify_artifacts(protocol_result,child.pid)
                        reply = encoded(TERMINAL_ACK) + b"\n"
                        input_count += len(reply)
                        if input_count > LIMITS["input_bytes"]:
                            raise SafetyError("input_limit")
                        write_pipe(child.stdin.fileno(),reply,frame_deadline)
                        result["terminal_ACK_sent"] = True
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
        expected_exit = 73
        if result.get("terminal_ACK_sent") is not True or result["exit_code"] != expected_exit or type(protocol_result) is not dict or protocol_result.get("passed") is not True or protocol_result.get("proof_scope") != PROOF_SCOPE:
            raise SafetyError("case_nonterminal_or_failed")
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
    if digest(read_bounded(Path(__file__).resolve(), 262144, private=False)) != manifest["supervisor_sha256"]:
        raise SafetyError("supervisor_hash_changed")
    results: list[dict] = []
    setup_failure = None
    # Reserve before runtime admission so a retained refusal cannot be replayed.
    fd = new_log(target, "Retained Stage B results; native host/model/hard-cap acceptance remains unproven")
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
                        "runtime_closure_complete":False,"stdlib_bytes_hashed":False,"OS_sharedcache_bytes_hashed":False,"native_security_sandbox":False,"complete": setup_failure is None and len(results) == len(manifest["cases"]) and all(item["passed"] for item in results)}) + b"\n"
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
