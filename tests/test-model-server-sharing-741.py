"""Shared ownership tests using production definitions and stdlib-only fakes.

This file can run directly without importing TrueMemory, numpy or torch.
Temporary endpoints and synthetic child identities never reference live state.
"""
from __future__ import annotations

import ast
import atexit
import base64
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hmac
import json
import logging
import math
import os
from pathlib import Path
import platform
import plistlib
import secrets
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import weakref
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "truememory"


def load_platform() -> types.ModuleType:
    module = types.ModuleType("synthetic_platform_741")
    exec(compile((SOURCE / "_platform.py").read_text(), "synthetic-platform", "exec"), module.__dict__)
    return module


def load_definitions(filename: str, namespace: dict, names: set[str] | None = None) -> types.ModuleType:
    parsed = ast.parse((SOURCE / filename).read_text())
    selected = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and (names is None or node.name in names)]
    tree = ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected,
    ], type_ignores=[]))
    module = types.ModuleType("synthetic_" + filename.replace(".", "_"))
    module.__dict__.update(namespace)
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(tree, filename, "exec"), module.__dict__)
    return module


def shared_namespace(root: Path) -> dict:
    helpers = load_platform()
    result = {"os": os, "sys": sys, "time": time, "json": json, "socket": socket,
              "struct": struct, "secrets": secrets, "logging": logging, "Path": Path,
              "log": logging.getLogger("synthetic-sharing-741"), "_TRUEMEMORY_DIR": root,
              "_USE_UNIX": sys.platform != "win32", "_LOOPBACK_HOST": "127.0.0.1",
              "_MODEL_SERVER_GENERATION_ENV": helpers._MODEL_SERVER_GENERATION_ENV,
              "try_file_lock": helpers.try_file_lock, "read_start_claim": helpers.read_start_claim,
              "process_birth": lambda pid: float(pid), "spawn_kwargs": helpers.spawn_kwargs,
              "pid_is_alive": lambda pid: pid == os.getpid(),
              "_HEADER_FMT": ">I", "_HEADER_SIZE": 4, "_MAX_MESSAGE_SIZE": 10 * 1024 * 1024,
              "PROTOCOL_VERSION": 1, "_SERVER_START_TIMEOUT": .5, "_REQUEST_TIMEOUT": 120,
              "_default_request_timeout": None, "base64": base64,
              "np": types.SimpleNamespace(ndarray=list, integer=int)}
    for name, filename in (("SOCK_PATH", "model.sock"), ("PID_PATH", "model_server.pid"),
                           ("PORT_PATH", "model_server.port"), ("TOKEN_PATH", "model_server.token"),
                           ("LOCK_PATH", "model_server.lock"), ("START_LOCK_PATH", "model_server.start.lock"),
                           ("START_STATE_PATH", "model_server.start.json")):
        result[name] = root / filename
    return result


def client_module(root: Path) -> types.ModuleType:
    namespace = shared_namespace(root)
    namespace.update(subprocess=subprocess, platform=platform, plistlib=plistlib, shutil=shutil,
                     _APP_BUNDLE_PATH=root / "TrueMemory.app",
                     _APP_EXECUTABLE=root / "TrueMemory.app" / "Contents" / "MacOS" / "TrueMemory",
                     _LSREGISTER=str(root / "synthetic-lsregister"))
    module = load_definitions("model_client.py", namespace)
    module._ensure_app_bundle = lambda _deadline=None: None
    return module


def server_module(root: Path) -> types.ModuleType:
    namespace = shared_namespace(root)
    namespace.update(dataclass=dataclass, contextmanager=contextmanager, ThreadPoolExecutor=ThreadPoolExecutor,
                     Future=Future, threading=threading, math=math, gc=gc, hmac=hmac,
                     stat=stat, atexit=atexit, signal=signal, IDLE_TIMEOUT=300,
                     _HMAC_TOKEN_BYTES=32, _EMBED_BATCH_LIMIT=32, _RERANK_BATCH_LIMIT=64)
    return load_definitions("model_server.py", namespace)


def racing_client(root: Path) -> None:
    module = client_module(root)
    module._SERVER_START_TIMEOUT = 3
    module._probe_server = lambda _deadline: (root / "synthetic-ready").exists()
    def spawn(*_args: object, **_kwargs: object) -> types.SimpleNamespace:
        with (root / "synthetic-launches").open("a") as stream:
            stream.write("launch\n")
        (root / "synthetic-ready").touch()
        return types.SimpleNamespace(pid=7654321)
    module.subprocess = types.SimpleNamespace(Popen=spawn, DEVNULL=subprocess.DEVNULL)
    if not module._start_server(wait_timeout=2):
        raise RuntimeError("synthetic concurrent starter failed")


class SharedModelOwnership(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="tm-sharing-741-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.client = client_module(self.root)
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("TRUEMEMORY_NO_MODEL_SERVER", None)
        os.environ.pop("TRUEMEMORY_MODEL_SERVER_GENERATION", None)

    def make_server(self) -> tuple[types.ModuleType, object]:
        module = server_module(self.root)
        server = module.ModelServer()
        self.addCleanup(server._cleanup)
        return module, server

    @contextmanager
    def loaders(self) -> Iterator[tuple[types.ModuleType, types.ModuleType, Mock]]:
        factories = Mock(side_effect=AssertionError("unexpected local model construction"))
        transformers = types.ModuleType("sentence_transformers")
        transformers.SentenceTransformer = factories
        transformers.CrossEncoder = factories
        mps = types.ModuleType("truememory.mps_utils")
        mps.resolve_device = lambda _value=None: "cpu"
        mps.auto_detect_device = lambda: "cpu"
        mps.ensure_mps_memory_budget = lambda _device: None
        namespace = {"_model": None, "_lock": threading.Lock(), "_embedding_dim": 256,
                     "EMBEDDING_MODEL": "qwen3_256", "os": os, "sys": sys,
                     "_model_name": None, "get_current_reranker_name": lambda: "synthetic-reranker"}
        vectors = load_definitions("vector_search.py", dict(namespace), {"get_model"})
        reranker = load_definitions("reranker.py", dict(namespace), {"get_reranker"})
        package = types.ModuleType("truememory")
        package.__path__ = []
        with patch.dict(sys.modules, {"truememory": package, "truememory.model_client": self.client,
                                     "truememory.mps_utils": mps, "sentence_transformers": transformers}):
            yield vectors, reranker, factories

    def test_missing_endpoints_return_proxies_in_independent_clients(self) -> None:
        self.client._server_ready = Mock(side_effect=AssertionError("policy read endpoint"))
        self.client._server_is_alive = Mock(side_effect=AssertionError("policy read PID"))
        for _ in range(2):
            with self.loaders() as (vectors, reranker, factories):
                embedding = vectors.get_model()
                cross_encoder = reranker.get_reranker()
                self.assertIsInstance(embedding, self.client.EmbeddingProxy)
                self.assertIsInstance(cross_encoder, self.client.RerankerProxy)
                self.client._request_with_autostart = Mock(side_effect=ConnectionError("synthetic absent"))
                with self.assertRaises(ConnectionError):
                    embedding.encode(["synthetic"])
                with self.assertRaises(ConnectionError):
                    cross_encoder.predict([("synthetic", "synthetic")])
                factories.assert_not_called()

    def test_proxy_constructor_failure_never_falls_into_local_factories(self) -> None:
        for name in ("get_embedding_proxy", "get_reranker_proxy"):
            with patch.object(self.client, name, side_effect=RuntimeError("synthetic proxy failure")):
                with self.loaders() as (vectors, reranker, factories):
                    getter = vectors.get_model if name == "get_embedding_proxy" else reranker.get_reranker
                    with self.assertRaisesRegex(RuntimeError, "synthetic proxy failure"):
                        getter()
                    factories.assert_not_called()

    def test_explicit_local_mode_preserves_models_and_device(self) -> None:
        os.environ["TRUEMEMORY_NO_MODEL_SERVER"] = "1"
        self.client._server_ready = Mock(side_effect=AssertionError("local policy probed endpoint"))
        with self.loaders() as (vectors, reranker, factories):
            factories.side_effect = None
            factories.return_value = object()
            self.assertIs(vectors.get_model(), factories.return_value)
            self.assertEqual(factories.call_args.args, ("Qwen/Qwen3-Embedding-0.6B",))
            self.assertEqual(factories.call_args.kwargs["truncate_dim"], 256)
            self.assertEqual(factories.call_args.kwargs["device"], "cpu")
            reranker.get_reranker(device="synthetic-explicit-device")
            self.assertEqual(factories.call_args.args, ("synthetic-reranker",))
            self.assertEqual(factories.call_args.kwargs["device"], "synthetic-explicit-device")
        with patch.object(self.client, "_spawn_server", side_effect=AssertionError("local mode spawned")):
            self.assertFalse(self.client._start_server())

    def test_sixteen_cross_process_cold_starters_launch_once(self) -> None:
        clients = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--startup-racer", str(self.root)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                   for _ in range(16)]
        try:
            for child in clients:
                _stdout, stderr = child.communicate(timeout=15)
                self.assertEqual(child.returncode, 0, stderr.decode())
        finally:
            for child in clients:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=3)
        self.assertEqual((self.root / "synthetic-launches").read_text().splitlines(), ["launch"])

    def test_pending_child_survives_waiter_timeout_and_recovers_after_death(self) -> None:
        live = {os.getpid(): 1.0, 7654321: 2.0}
        ready = threading.Event()
        launches = []
        self.client.process_birth = lambda pid: live.get(pid)
        self.client._probe_server = lambda _deadline: ready.is_set()
        def spawn(*_args: object) -> types.SimpleNamespace:
            launches.append(True)
            live[7654321] = float(len(launches) + 1)
            return types.SimpleNamespace(pid=7654321)
        self.client._spawn_server = spawn
        self.assertFalse(self.client._start_server(.02))
        self.assertFalse(self.client._start_server(.02))
        self.assertEqual(len(launches), 1)
        ready.set()
        self.assertTrue(self.client._start_server(.02))
        ready.clear()  # Idle exit/crash leaves no usable endpoint.
        live.pop(7654321)
        self.assertFalse(self.client._start_server(.02))
        self.assertEqual(len(launches), 2)

    def test_crashed_launcher_or_reused_child_identity_allows_one_replacement(self) -> None:
        for child in (None, {"pid": 7654321, "born": 1.0}):
            with self.subTest(child=child):
                self.client._write_start_claim({"generation": "a" * 32,
                    "launcher": {"pid": 7654320, "born": 1.0}, "child": child})
                self.client.process_birth = lambda pid: {os.getpid(): 2.0, 7654321: 2.0}.get(pid)
                self.client._spawn_server = Mock(return_value=types.SimpleNamespace(pid=7654321))
                self.assertFalse(self.client._start_server(.02))
                self.client._spawn_server.assert_called_once()
                self.assertNotEqual(self.client.read_start_claim(self.client.START_STATE_PATH)["generation"], "a" * 32)

    def test_stale_pid_and_artifacts_do_not_imply_readiness_or_get_deleted(self) -> None:
        self.client.PID_PATH.write_text(str(os.getpid()))
        for name in ("SOCK_PATH", "PORT_PATH", "TOKEN_PATH"):
            getattr(self.client, name).write_text("synthetic sentinel")
        self.client._probe_server = lambda _deadline: False
        self.client._spawn_server = Mock(return_value=types.SimpleNamespace(pid=7654321))
        self.assertFalse(self.client._start_server(.02))
        self.client._spawn_server.assert_called_once()
        self.assertEqual(self.client.PID_PATH.read_text(), str(os.getpid()))
        for name in ("SOCK_PATH", "PORT_PATH", "TOKEN_PATH"):
            self.assertEqual(getattr(self.client, name).read_text(), "synthetic sentinel")

    def test_held_bind_lock_prevents_launch_even_without_endpoint_or_pid(self) -> None:
        owner = self.client.try_file_lock(self.client.LOCK_PATH)
        self.assertIsNotNone(owner)
        try:
            self.client._spawn_server = Mock(side_effect=AssertionError("existing owner duplicated"))
            self.assertFalse(self.client._start_server(.02))
            self.client._spawn_server.assert_not_called()
        finally:
            os.close(owner)

    def test_busy_and_foreign_protocol_never_spawn(self) -> None:
        self.client._server_ready = lambda: True
        self.client._spawn_server = Mock(side_effect=AssertionError("responding endpoint restarted"))
        self.client._send_request = Mock(side_effect=self.client.ModelServerBusyError("synthetic busy"))
        self.assertTrue(self.client._start_server(.1))
        with self.assertRaises(self.client.ModelServerBusyError):
            self.client.EmbeddingProxy().encode(["synthetic"])
        self.client._send_request.side_effect = self.client.ProtocolMismatchError("synthetic foreign")
        with self.assertRaises(self.client.ProtocolMismatchError):
            self.client._start_server(.1)
        with self.assertRaises(self.client.ProtocolMismatchError):
            self.client.EmbeddingProxy().encode(["synthetic"])
        self.client._spawn_server.assert_not_called()

    def test_deadline_bounds_gate_wait_and_prevents_late_spawn(self) -> None:
        class Clock:
            value = 0.0
            def monotonic(self) -> float:
                return self.value
            def sleep(self, amount: float) -> None:
                self.value += amount
        clock = Clock()
        self.client.time = clock
        self.client.try_file_lock = Mock(return_value=None)
        self.client._spawn_server = Mock(side_effect=AssertionError("expired caller spawned"))
        self.assertFalse(self.client._start_server(0))
        self.client.try_file_lock.assert_not_called()
        self.assertFalse(self.client._start_server(.012))
        self.assertAlmostEqual(clock.value, .012)
        self.client._spawn_server.assert_not_called()

    def test_app_setup_consumes_same_budget_and_expiry_stops_popen(self) -> None:
        clock = types.SimpleNamespace(value=0.0)
        self.client.time = types.SimpleNamespace(monotonic=lambda: clock.value)
        def app_setup(deadline: float) -> None:
            self.assertEqual(deadline, .1)
            clock.value = .11
            return None
        self.client._ensure_app_bundle = app_setup
        with patch.object(subprocess, "Popen", side_effect=AssertionError("expired caller spawned")):
            with self.assertRaises(TimeoutError):
                self.client._spawn_server("a" * 32, .1)

    def test_expired_start_returns_timeout_and_legacy_retry_contract_survives(self) -> None:
        clock = types.SimpleNamespace(value=0.0)
        self.client.time = types.SimpleNamespace(monotonic=lambda: clock.value)
        self.client._send_request = Mock(side_effect=FileNotFoundError("synthetic absent"))
        def expire(wait_timeout: float | None = None) -> bool:
            assert wait_timeout is not None
            clock.value += wait_timeout
            return False
        self.client._start_server = expire
        with self.assertRaises(TimeoutError):
            self.client.EmbeddingProxy().encode(["synthetic"], timeout=.1)
        self.client._send_request = Mock(side_effect=[socket.timeout("synthetic slow"), {"ok": True, "vectors": [[1]]}])
        self.client._start_server = Mock(return_value=True)
        self.assertEqual(self.client.EmbeddingProxy().encode(["synthetic"]), [[1]])
        self.assertEqual(self.client._REQUEST_TIMEOUT, 120)
        self.assertEqual(self.client._send_request.call_count, 2)

    def test_non_owner_cleanup_preserves_winner_between_bind_and_pid(self) -> None:
        module, winner = self.make_server()
        winner._lock_fd = winner._acquire_bind_lock()
        module.SOCK_PATH.write_text("synthetic winning socket")
        _module, loser = self.make_server()
        with self.assertRaises(RuntimeError):
            loser._lock_fd = loser._acquire_bind_lock()
        loser._cleanup()
        self.assertEqual(module.SOCK_PATH.read_text(), "synthetic winning socket")
        winner._cleanup()
        self.assertFalse(module.SOCK_PATH.exists())

    def test_superseded_generation_releases_lock_without_touching_artifacts(self) -> None:
        module, server = self.make_server()
        self.client._write_start_claim({"generation": "b" * 32,
            "launcher": {"pid": os.getpid(), "born": float(os.getpid())}, "child": None})
        os.environ[module._MODEL_SERVER_GENERATION_ENV] = "a" * 32
        server._lock_fd = server._acquire_bind_lock()
        module.SOCK_PATH.write_text("synthetic untouched socket")
        with self.assertRaisesRegex(RuntimeError, "superseded"):
            server._validate_launch_generation_locked()
        self.assertIsNone(server._lock_fd)
        server._cleanup()
        self.assertTrue(module.SOCK_PATH.exists())
        fd = self.client.try_file_lock(self.client.LOCK_PATH)
        self.assertIsNotNone(fd)
        os.close(fd)

    def test_shutdown_retains_bind_owner_while_admitted_native_work_drains(self) -> None:
        module, server = self.make_server()
        module._USE_UNIX = True  # socketpair is the synthetic authenticated transport.
        server._lock_fd = server._acquire_bind_lock()
        retained_fd = server._lock_fd
        module.PID_PATH.write_text(str(os.getpid()))
        module.SOCK_PATH.write_text("synthetic owned socket")
        model = object()
        server._reranker = model
        entered, release = threading.Event(), threading.Event()
        def native_boundary(_request: dict) -> dict:
            entered.set()
            if not release.wait(2):
                raise RuntimeError("synthetic native watchdog")
            return {"ok": True}
        server.handle_request = native_boundary
        peer, connection = socket.socketpair()
        data = json.dumps({"op": "synthetic-native"}).encode()
        peer.sendall(struct.pack(">I", len(data)) + data)
        future = server._dispatch_client(connection)
        try:
            self.assertTrue(entered.wait(1))
            started = time.monotonic()
            server._cleanup()
            self.assertLess(time.monotonic() - started, .5)
            self.assertIs(server._reranker, model)
            self.assertEqual(server._lock_fd, retained_fd)
            self.assertTrue(module.SOCK_PATH.exists())
            self.assertIsNone(self.client.try_file_lock(self.client.LOCK_PATH))
            server._cleanup()
            with self.assertRaisesRegex(RuntimeError, "cannot be restarted"):
                server.run()
        finally:
            release.set()
            future.result(timeout=2)
            peer.close()
            server._workers.shutdown(wait=True)
            # The test process continues; emulate OS teardown of this owned
            # synthetic fd without signaling any other process.
            os.close(retained_fd)
            server._lock_fd = None

    def test_shutdown_releases_model_references_before_bind_ownership(self) -> None:
        for cyclic in (False, True):
            with self.subTest(cyclic=cyclic):
                module, server = self.make_server()
                server._lock_fd = server._acquire_bind_lock()
                module.PID_PATH.write_text(str(os.getpid()))
                observed, references = [], []
                lock_path = self.client.LOCK_PATH
                acquire = self.client.try_file_lock

                class Model:
                    def __init__(self, name: str) -> None:
                        self.name = name
                        if cyclic:
                            self.cycle = self

                    def __del__(self) -> None:
                        fd = acquire(lock_path)
                        observed.append((self.name, fd is None))
                        if fd is not None:
                            os.close(fd)

                for name in ("embed", "rerank", "fast"):
                    model = Model(name)
                    references.append(weakref.ref(model))
                    if name == "embed":
                        server._embed_state = module._EmbedState(model, "base", "qwen3_256")
                    elif name == "rerank":
                        server._reranker = model
                    else:
                        server._fast_encoder = model
                del model
                server._cleanup()
                self.assertEqual(sorted(observed), [("embed", True), ("fast", True), ("rerank", True)])
                self.assertTrue(all(reference() is None for reference in references))
                self.assertIsNone(server._lock_fd)
                successor = acquire(lock_path)
                self.assertIsNotNone(successor)
                os.close(successor)

    def test_shutdown_collection_failure_retains_bind_ownership(self) -> None:
        module, server = self.make_server()
        server._lock_fd = server._acquire_bind_lock()
        retained_fd = server._lock_fd
        module.PID_PATH.write_text(str(os.getpid()))
        module.SOCK_PATH.write_text("synthetic retained socket")
        try:
            with patch.object(module.gc, "collect", side_effect=RuntimeError("synthetic collection failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic collection failure"):
                    server._cleanup()
            self.assertEqual(server._lock_fd, retained_fd)
            self.assertTrue(module.SOCK_PATH.exists())
            self.assertIsNone(self.client.try_file_lock(self.client.LOCK_PATH))
            server._cleanup()
            self.assertEqual(server._lock_fd, retained_fd)
        finally:
            # Emulate OS exit for the synthetic server in this continuing test process.
            if server._lock_fd is not None:
                os.close(server._lock_fd)
                server._lock_fd = None

    def test_current_generation_and_direct_launch_remain_valid(self) -> None:
        module, server = self.make_server()
        server._lock_fd = server._acquire_bind_lock()
        server._validate_launch_generation_locked()
        self.client._write_start_claim({"generation": "a" * 32,
            "launcher": {"pid": os.getpid(), "born": float(os.getpid())}, "child": None})
        os.environ[module._MODEL_SERVER_GENERATION_ENV] = "a" * 32
        server._validate_launch_generation_locked()

    def test_windows_readiness_uses_real_token_connection_and_protocol(self) -> None:
        self.client._USE_UNIX = False
        token = bytes(range(32))
        self.client.PORT_PATH.write_text("12345")
        self.client.TOKEN_PATH.write_text(token.hex())
        response = json.dumps({"ok": True, "protocol": 1}).encode()
        incoming = bytearray(struct.pack(">I", len(response)) + response)
        sent = []
        class FakeSocket:
            def settimeout(self, _timeout: float) -> None:
                pass
            def connect(self, address: tuple[str, int]) -> None:
                self.address = address
            def sendall(self, data: bytes) -> None:
                sent.append(data)
            def recv(self, size: int) -> bytes:
                value = bytes(incoming[:size])
                del incoming[:size]
                return value
            def close(self) -> None:
                pass
        with patch.object(socket, "socket", return_value=FakeSocket()):
            self.assertTrue(self.client._probe_server(time.monotonic() + 1))
        self.assertEqual(sent[0], token)
        self.client.TOKEN_PATH.write_text("invalid-token")
        with patch.object(socket, "socket", side_effect=AssertionError("invalid token connected")):
            self.assertFalse(self.client._probe_server(time.monotonic() + 1))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--startup-racer":
        racing_client(Path(sys.argv[2]))
    else:
        unittest.main()
