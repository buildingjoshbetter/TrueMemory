"""Run bounded control-flow probes against pinned source without loading models.

Extracted AST definitions are unmodified. Imports, model factories, sensor reads,
and clocks are stubbed explicitly. This is not a real-model memory benchmark.
"""
from __future__ import annotations

import ast
import base64
import json
import logging
import os
import subprocess
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
PR_HEAD = "693ac64f4bb9d806c555e2030d8444e133d1cff7"
RESULTS: list[dict] = []


def record(probe: str, observations: dict) -> None:
    RESULTS.append({"probe": probe, **observations})


def extract(source: str, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in selected} == names
    prefix = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[prefix, *selected], type_ignores=[]))
    exec(compile(module, "<pinned-ast>", "exec"), namespace)
    return namespace


def source_text(relative: str) -> str:
    return (SOURCE / relative).read_text()


def server_namespace(text: str | None = None) -> dict:
    module = types.ModuleType("runtime_probe_server")
    sys.modules[module.__name__] = module
    module.__dict__.update({
        "dataclass": dataclass, "threading": threading, "time": time,
        "np": np, "log": logging.getLogger("runtime-probe"), "sys": sys,
        "os": os, "MAX_MEMORY_MB": 1500.0, "EMBED_CACHE_SIZE": 1024,
        "RERANK_CACHE_SIZE": 2048,
    })
    return extract(text or source_text("truememory/model_server.py"), {"_EmbedState", "ModelServer"}, module.__dict__)


def basic_model() -> object:
    class FakeModel:
        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            return np.zeros((len(texts), 2), dtype=np.float32)
    return FakeModel()


def probe_batch_controls() -> None:
    captured: list[dict] = []
    def request(req: dict, timeout: float | None = None) -> dict:
        captured.append(req)
        return {"ok": True, "vectors": np.zeros((1, 2)), "scores": np.zeros(1)}
    ns = extract(source_text("truememory/model_client.py"), {"EmbeddingProxy", "RerankerProxy"}, {
        "np": np, "_request_with_autostart": request,
    })
    ns["EmbeddingProxy"]("qwen3_256").encode(["x"], batch_size=1)
    ns["RerankerProxy"]("fixed-reranker").predict([("q", "d")], batch_size=1)
    assert all("batch_size" not in payload for payload in captured)
    server = server_namespace()["ModelServer"]()
    sizes: list[int] = []
    class FakeModel:
        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            sizes.append(len(texts))
            return np.zeros((len(texts), 2), dtype=np.float32)
    class FakeThrottler:
        def before_batch(self) -> tuple[int, dict]:
            return 1, {}
        def after_batch(self, count: int, elapsed: float) -> None:
            pass
        def should_flush_cache(self) -> bool:
            return False
    server._get_embed_model = lambda tier: FakeModel()
    server._throttler = FakeThrottler()
    server._throttler_active = True
    result = server.handle_request({"op": "embed", "tier": "qwen3_256", "texts": ["a", "b", "c", "d"]})
    assert result["ok"] and sizes == [4]
    record("R01_batch_controls", {"requested_proxy_batch": 1, "wire_batch_size_present": False, "throttler_batch": 1, "actual_encode_sizes": sizes})


def probe_cpu_builder() -> None:
    calls: list[dict] = []
    fake_st = types.ModuleType("sentence_transformers")
    def factory(name: str, **kwargs: object) -> object:
        calls.append({"name": name, **kwargs})
        return basic_model()
    fake_st.SentenceTransformer = factory
    with patch.dict(sys.modules, {"sentence_transformers": fake_st}), patch.object(sys, "platform", "darwin"):
        server_namespace()["ModelServer"]._build_embed_model("qwen3_256", "cpu")
    assert calls[0]["device"] == "cpu"
    assert calls[0]["model_kwargs"] == {"attn_implementation": "eager"}
    record("R02_cpu_builder", {"factory_call": calls[0], "one_fp32_attention_tensor_gib_at_8192": 16 * 8192**2 * 4 / 1024**3, "real_inference_run": False})


def probe_queued_deadline() -> None:
    server = server_namespace()["ModelServer"]()
    started = threading.Event()
    encode_times: list[float] = []
    responses: list[dict] = []
    class FakeModel:
        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            encode_times.append(time.time())
            return np.zeros((len(texts), 2))
    server._get_embed_model = lambda tier: FakeModel()
    inner = server._handle_request_inner
    def entered(req: dict) -> dict:
        started.set()
        return inner(req)
    server._handle_request_inner = entered
    server._lock.acquire()
    deadline = time.time() + 0.1
    request = {"op": "embed", "tier": "qwen3_256", "texts": ["a", "b"], "deadline": deadline}
    thread = threading.Thread(target=lambda: responses.append(server.handle_request(request)))
    thread.start()
    assert started.wait(1)
    time.sleep(0.15)
    server._lock.release()
    thread.join(2)
    assert not thread.is_alive() and encode_times[0] > deadline and responses[0]["ok"]
    record("R03_queued_deadline", {"encoded_after_deadline": True, "response_ok": True, "controlled_lock_wait_seconds": 0.15})


def fake_mps_module() -> types.ModuleType:
    mod = types.ModuleType("truememory.mps_utils")
    mod.is_mps_oom = lambda error: "MPS out of memory" in str(error)
    mod.flush_mps_cache = lambda: None
    return mod


def probe_retry_concurrency() -> None:
    server = server_namespace()["ModelServer"]()
    retry_entered = threading.Event()
    release_retry = threading.Event()
    counts = {"active": 0, "peak": 0, "first": True}
    lock = threading.Lock()
    responses: list[dict] = []
    class FakeModel:
        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            if texts[0] == "oom" and counts["first"]:
                counts["first"] = False
                raise RuntimeError("MPS out of memory")
            with lock:
                counts["active"] += 1
                counts["peak"] = max(counts["active"], counts["peak"])
            if texts[0] == "oom":
                retry_entered.set()
                assert release_retry.wait(2)
            with lock:
                counts["active"] -= 1
            return np.zeros((len(texts), 2))
    model = FakeModel()
    server._get_embed_model = lambda tier: model
    server._recover_embed_oom_locked = lambda model: server._sticky_cpu.add("embed")
    with patch.dict(sys.modules, {"truememory.mps_utils": fake_mps_module()}):
        thread = threading.Thread(target=lambda: responses.append(server.handle_request({"op": "embed", "texts": ["oom", "x"]})))
        thread.start()
        assert retry_entered.wait(1)
        server.handle_request({"op": "embed", "texts": ["normal", "x"]})
        release_retry.set()
        thread.join(2)
    assert not thread.is_alive() and counts["peak"] == 2
    record("R05_retry_concurrency", {"same_model_peak_concurrent_encodes": counts["peak"], "real_models": False})


def probe_local_loading() -> None:
    calls: list[dict] = []
    model_client = types.ModuleType("truememory.model_client")
    model_client.use_model_server = lambda: False
    model_client.get_embedding_proxy = lambda tier: None
    st = types.ModuleType("sentence_transformers")
    def factory(name: str, **kwargs: object) -> object:
        calls.append({"name": name, **kwargs})
        return basic_model()
    st.SentenceTransformer = factory
    namespaces: list[dict] = []
    with patch.dict(sys.modules, {"truememory.model_client": model_client, "sentence_transformers": st}), patch.dict(os.environ, {"TRUEMEMORY_DEVICE": "cpu"}):
        for _ in range(2):
            ns = extract(source_text("truememory/vector_search.py"), {"get_model"}, {
                "_model": None, "_embedding_dim": 256, "_lock": threading.Lock(),
                "EMBEDDING_MODEL": "qwen3_256", "logger": logging.getLogger("probe"), "os": os,
            })
            ns["get_model"]()
            namespaces.append(ns)
    assert len(calls) == 2 and all("device" not in call for call in calls)
    assert namespaces[0]["_model"] is not namespaces[1]["_model"]
    record("R06_local_device_override", {"environment_device": "cpu", "factory_device_argument_present": False, "factory_calls": calls})
    record("R08_server_absent_local_copies", {"independent_loader_namespaces": 2, "distinct_models_loaded": 2, "real_processes_spawned": 0})


def probe_local_device_race() -> None:
    cpu_retry = threading.Event()
    normal_running = threading.Event()
    release_retry = threading.Event()
    release_normal = threading.Event()
    moves: list[dict] = []
    state = {"first": True}
    class FakeModel:
        def encode(self, texts: list[str], **kwargs: object) -> list[int]:
            if texts[0] == "retry":
                if state["first"]:
                    state["first"] = False
                    raise RuntimeError("MPS out of memory")
                cpu_retry.set()
                assert release_retry.wait(2)
            else:
                normal_running.set()
                assert release_normal.wait(2)
                normal_running.clear()
            return [1]
        def to(self, device: str) -> None:
            moves.append({"device": device, "another_encode_active": normal_running.is_set()})
    ns = extract(source_text("truememory/mps_utils.py"), {"is_mps_oom", "encode_with_mps_fallback"}, {
        "_device_lock": threading.Lock(), "logger": logging.getLogger("probe"), "flush_mps_cache": lambda: None,
    })
    torch_stub = types.ModuleType("torch")
    torch_stub.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True))
    model = FakeModel()
    with patch.dict(sys.modules, {"torch": torch_stub}):
        first = threading.Thread(target=lambda: ns["encode_with_mps_fallback"](model, ["retry"]))
        second = threading.Thread(target=lambda: ns["encode_with_mps_fallback"](model, ["normal"]))
        first.start()
        assert cpu_retry.wait(1)
        second.start()
        assert normal_running.wait(1)
        release_retry.set()
        first.join(2)
        release_normal.set()
        second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert moves[-1] == {"device": "mps", "another_encode_active": True}
    record("R07_local_device_race", {"device_moves": moves, "real_torch_imported": False})


def probe_throttler_sleep() -> None:
    class FakeClock:
        now = 1000.0
        sleeps: list[float] = []
        def time(self) -> float:
            return self.now
        def sleep(self, duration: float) -> None:
            self.sleeps.append(duration)
            self.now += duration
    clock = FakeClock()
    ns = extract(source_text("truememory/tier_switch/state_machine.py"), {"ThrottlerStateMachine"}, {"time": clock, "log": logging.getLogger("probe")})
    ns.update({
        "psutil": types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(total=32 * 1024**3)),
        "_MACHINE_PROFILES": {(0, 12): (.5, 1, 4, 1), (12, 20): (.5, 1, 8, 1), (20, 30): (.5, 1, 12, 2), (30, 1024): (.55, 1, 16, 2)},
        "GrowthRateTracker": lambda cap_gb: None,
    })
    extract(source_text("truememory/tier_switch/throttler.py"), {"_get_profile", "DynamicThrottler"}, ns)
    throttler = ns["DynamicThrottler"]("cpu")
    throttler._read_all_channels = lambda: {key: {"status": "ok"} for key in ("mps_level", "growth_rate", "thermal")}
    for _ in range(15):
        throttler.before_batch()
    assert clock.sleeps.count(10) == 2
    record("R09_throttler_sleep", {"before_batch_calls": 15, "ten_second_sleeps": 2, "simulated_total_sleep_seconds": round(sum(clock.sleeps), 2), "real_sleep_seconds": 0, "cpu_throttler_mps_cap_gib": throttler.mps_cap_gb})


def pr_source() -> str:
    endpoint = f"repos/buildingjoshbetter/TrueMemory/contents/truememory/model_server.py?ref={PR_HEAD}"
    response = json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))
    return base64.b64decode(response["content"]).decode()


def probe_pr_watchdog_and_cache() -> None:
    ns = server_namespace(pr_source())
    server = ns["ModelServer"]()
    pressure = {"rss_mb": 2000.0, "max_mb": 1500.0, "warning": True, "exceeded": True}
    mps = fake_mps_module()
    mps.check_memory_pressure = lambda limit: pressure
    server._flush_mps_cache = lambda: None
    server._embed_state = ns["_EmbedState"](basic_model(), "qwen3_256", "qwen3_256")
    with patch.dict(sys.modules, {"truememory.mps_utils": mps}):
        server._check_memory_watchdog()
    assert server._running and server._embed_state is None
    record("R11_pr729_recycle_noop", {"pressure_before_and_after_mb": 2000, "limit_mb": 1500, "running_after_recycle_log": server._running, "models_cleared": server._embed_state is None, "scope": "unmerged PR729"})
    backing = np.zeros((1024, 256), dtype=np.float32)
    server._set_cached_embed("qwen3_256", "repeated text", backing[-1])
    cached = server._get_cached_embed("qwen3_256", "repeated text")
    assert cached.base is backing
    record("R12_pr729_cache_view_retention", {"cache_entries": 1, "visible_vector_bytes": cached.nbytes, "retained_backing_bytes": backing.nbytes, "amplification": backing.nbytes // cached.nbytes, "scope": "unmerged PR729"})


def main() -> None:
    logging.basicConfig(level=logging.CRITICAL)
    assert "torch" not in sys.modules
    probe_batch_controls()
    probe_cpu_builder()
    probe_queued_deadline()
    probe_retry_concurrency()
    probe_local_loading()
    probe_local_device_race()
    probe_throttler_sleep()
    probe_pr_watchdog_and_cache()
    assert "torch" not in sys.modules
    print(json.dumps({
        "canonical_commit": "063e5b8844af735a52fde886217a5d26a0f13064",
        "pr729_head": PR_HEAD, "probe_count": len(RESULTS),
        "real_models_loaded": 0, "real_torch_imported": False,
        "source_mutated": False, "results": RESULTS,
    }, indent=2))


if __name__ == "__main__":
    main()
