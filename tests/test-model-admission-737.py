"""Synthetic transport/admission regressions; no package or model imports."""
from __future__ import annotations

import ast
import json
import os
import socket
import struct
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "truememory"


def load_source(name: str) -> types.ModuleType:
    tree = ast.parse((SOURCE / (name + ".py")).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(alias.name in ("numpy", "psutil") for alias in node.names)
        or isinstance(node, ast.ImportFrom) and (node.module or "").startswith("truememory.")
        or isinstance(node, ast.Try) and any(
            isinstance(child, ast.Import) and any(alias.name == "psutil" for alias in child.names)
            for child in node.body
        )
        or isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "_set_mps_memory_cap"
    )]
    module = types.ModuleType("synthetic_" + name)
    module.__dict__.update({
        "np": types.SimpleNamespace(ndarray=list, float32="float32", asarray=lambda value, **kwargs: value),
        "psutil": None, "_USE_UNIX": True, "_LOOPBACK_HOST": "127.0.0.1",
        "_env_int": lambda name, default, **kwargs: default,
        "pid_is_alive": lambda pid: False, "spawn_kwargs": lambda: {},
    })
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE / (name + ".py")), "exec"), module.__dict__)
    return module


def frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


class TestModelAdmission(unittest.TestCase):
    def setUp(self) -> None:
        self.server_module = load_source("model_server")
        self.client_module = load_source("model_client")
        with patch.dict(os.environ, {"TRUEMEMORY_MODEL_SERVER_MAX_HANDLERS": "2"}):
            self.server = self.server_module.ModelServer()
        self.server._REJECT_TIMEOUT = 0.02
        self.clients: list[socket.socket] = []
        self.releases: list[threading.Event] = []
        self.futures = []
        self.server._get_embed_model = Mock(side_effect=AssertionError("Unexpected model load"))
        self.server._get_reranker = Mock(side_effect=AssertionError("Unexpected model load"))

    def tearDown(self) -> None:
        for event in self.releases:
            event.set()
        self.server._stop_clients()
        self.server._workers.shutdown(wait=True, cancel_futures=True)
        for client in self.clients:
            client.close()
        self.assertEqual(len(self.server._clients), 0)
        self.assertEqual(self.server._reserved_request_bytes, 0)
        for module in (self.server_module, self.client_module):
            sys.modules.pop(module.__name__, None)

    def pair(self) -> tuple[socket.socket, socket.socket]:
        client, server = socket.socketpair()
        client.settimeout(2)
        self.clients.append(client)
        return client, server

    def send(self, payload: bytes, advertised: int | None = None) -> tuple[socket.socket, object]:
        client, server = self.pair()
        client.sendall(frame(payload) if advertised is None else struct.pack(">I", advertised) + payload)
        future = self.server._dispatch_client(server)
        if future is not None:
            self.futures.append(future)
        return client, future

    def response(self, client: socket.socket) -> dict:
        header = self.server._recv_exact(client, 4, time.monotonic() + 2)
        self.assertIsNotNone(header)
        body = self.server._recv_exact(client, struct.unpack(">I", header)[0], time.monotonic() + 2)
        return json.loads(body)

    def blocked_handler(self) -> tuple[threading.Event, threading.Event]:
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)

        def handle(request: dict) -> dict:
            entered.set()
            if not release.wait(2):
                raise AssertionError("Synthetic request was not released")
            return {"ok": True}

        self.server.handle_request = handle
        return entered, release

    def assert_empty(self) -> None:
        self.assertEqual(len(self.server._clients), 0)
        self.assertEqual(self.server._reserved_request_bytes, 0)

    def test_strict_configuration_and_documented_default_arithmetic(self) -> None:
        parse = self.server_module._admission_setting
        for invalid in ("", "0", "-1", "1.5", "nan", "inf", " 2", "+2", "129", "٢"):
            with self.subTest(invalid=invalid), patch.dict(os.environ, {"SYNTHETIC_LIMIT": invalid}):
                with self.assertRaisesRegex(ValueError, "SYNTHETIC_LIMIT"):
                    parse("SYNTHETIC_LIMIT", 16, 1, 128)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(parse("SYNTHETIC_LIMIT", 16, 1, 128), 16)
        self.assertEqual(self.server._max_request_bytes, 33554432)
        self.assertLessEqual(3 * self.server_module._MAX_MESSAGE_SIZE, 33554432)
        self.assertGreater(4 * self.server_module._MAX_MESSAGE_SIZE, 33554432)

    def test_handler_capacity_rejects_before_body_decode_or_submission(self) -> None:
        entered, release = self.blocked_handler()
        payload = b'{"op":"ping"}'
        first, first_future = self.send(payload)
        self.assertTrue(entered.wait(1))
        second, second_future = self.send(payload)
        with patch.object(self.server._workers, "submit", side_effect=AssertionError("Unbounded submit")):
            third, rejected = self.send(b"", advertised=len(payload))
        self.assertIsNone(rejected)
        response = self.response(third)
        self.assertEqual(response["error_code"], "server_busy")
        self.assertEqual(response["retry_after_ms"], 250)
        self.assertEqual(len(self.server._clients), 2)
        self.assertEqual(self.server._reserved_request_bytes, 2 * len(payload))
        self.assertLessEqual(len(self.server._workers._threads), 2)
        release.set()
        for client, future in ((first, first_future), (second, second_future)):
            self.assertTrue(self.response(client)["ok"])
            future.result(2)
        self.assert_empty()

    def test_byte_credit_is_reserved_from_header_and_held_through_inference(self) -> None:
        self.server._max_request_bytes = 20
        client, future = self.send(b"", advertised=20)
        self.assertEqual(self.server._reserved_request_bytes, 20)
        with patch.object(self.server_module.json, "loads", side_effect=AssertionError("Rejected body decoded")):
            rejected, rejected_future = self.send(b"", advertised=1)
        self.assertIsNone(rejected_future)
        self.assertEqual(self.response(rejected)["error_code"], "server_busy")
        client.shutdown(socket.SHUT_WR)
        future.result(2)
        self.assert_empty()
        entered, release = self.blocked_handler()
        client, future = self.send(b'{"op":"ping"}')
        self.assertTrue(entered.wait(1))
        self.assertEqual(self.server._reserved_request_bytes, 13)
        release.set()
        self.response(client)
        future.result(2)
        self.assert_empty()

    def test_malformed_empty_oversized_and_truncated_frames_return_credits(self) -> None:
        for payload in (b"{", b"[]", b"null", b"\xff"):
            with self.subTest(payload=payload):
                client, future = self.send(payload)
                self.assertFalse(self.response(client)["ok"])
                future.result(2)
                self.assert_empty()
        for length in (0, self.server_module._MAX_MESSAGE_SIZE + 1):
            client, future = self.send(b"", advertised=length)
            self.assertIsNone(future)
            self.assertEqual(client.recv(1), b"")
            self.assert_empty()
        client, future = self.send(b"{", advertised=20)
        client.shutdown(socket.SHUT_WR)
        future.result(2)
        self.assert_empty()

    def test_authentication_stays_before_application_responses(self) -> None:
        self.server._token = b"x" * 32
        with patch.object(self.server_module, "_USE_UNIX", False):
            client, peer = self.pair()
            client.sendall(b"y" * 32)
            self.assertIsNone(self.server._dispatch_client(peer))
            self.assertEqual(client.recv(1), b"")
            client, peer = self.pair()
            client.sendall(b"x" * 32 + frame(b'{"op":"ping"}'))
            future = self.server._dispatch_client(peer)
            self.assertTrue(self.response(client)["ok"])
            future.result(2)
        self.assert_empty()

    def test_header_and_body_timeouts_are_absolute_even_for_slow_drips(self) -> None:
        class DrippingSocket:
            def __init__(self) -> None:
                self.timeouts: list[float] = []
                self.reads = 0

            def settimeout(self, seconds: float) -> None:
                self.timeouts.append(seconds)

            def recv(self, count: int) -> bytes:
                self.reads += 1
                return b"x"

        drip = DrippingSocket()
        with patch.object(self.server_module.time, "monotonic", side_effect=[0.0, 0.4, 0.8, 1.2]):
            with self.assertRaises(TimeoutError):
                self.server._recv_exact(drip, 4, 1.0)
        self.assertEqual(drip.reads, 3)
        self.assertAlmostEqual(drip.timeouts[-1], 0.2)
        self.server._frame_timeout = 0.02
        client, future = self.send(b"", advertised=20)
        self.assertIn("timed out", self.response(client)["error"])
        future.result(2)
        self.assert_empty()
        self.server._header_timeout = 0.02
        client, peer = self.pair()
        client.sendall(b"\x00")
        self.assertIsNone(self.server._dispatch_client(peer))
        self.assertEqual(client.recv(1), b"")

    def test_total_frame_budget_includes_header_time_and_caps_header_budget(self) -> None:
        for header_timeout, frame_timeout, expected_header in ((10, 30, 110), (30, 10, 110)):
            with self.subTest(header_timeout=header_timeout, frame_timeout=frame_timeout):
                self.server._header_timeout = header_timeout
                self.server._frame_timeout = frame_timeout
                client, peer = self.pair()
                with patch.object(self.server_module.time, "monotonic", side_effect=[100, 107]), \
                        patch.object(self.server, "_recv_exact", return_value=struct.pack(">I", 13)) as receive:
                    admitted = self.server._admit_client(peer)
                self.assertIsNotNone(admitted)
                self.assertEqual(receive.call_args.args, (peer, 4, expected_header))
                self.assertEqual(admitted.frame_expires_at, 100 + frame_timeout)
                self.server._release_client(peer)
                self.assert_empty()

    def test_valid_caller_deadline_replaces_only_the_legacy_queue_ceiling(self) -> None:
        admitted = self.server_module._TransportRequest(13, time.monotonic() - 1, self.server._stopped)
        deadline = self.server_module._RequestDeadline(time.monotonic() + 300, admitted)
        self.assertGreater(deadline.remaining(), 120)
        with self.assertRaises(TimeoutError):
            self.server_module._RequestDeadline(None, admitted).check()
        with self.assertRaises(TimeoutError):
            self.server_module._RequestDeadline(time.monotonic() - 1, admitted).check()

    def test_expired_caller_and_legacy_queue_wait_never_load_models(self) -> None:
        self.server._inference_lock.acquire()
        try:
            for request in (
                {"op": "rerank", "pairs": [["synthetic", "document"]], "deadline": time.time() - 1},
                {"op": "rerank", "pairs": [["synthetic", "document"]],
                 "_transport_context": {"started": True}, "queue_expires_at": None},
            ):
                self.server._queue_timeout = 0.03
                client, future = self.send(json.dumps(request).encode())
                self.assertIn("deadline", self.response(client)["error"])
                future.result(2)
                self.server._get_reranker.assert_not_called()
                self.assertTrue(self.server._inference_lock.locked())
                self.assert_empty()
        finally:
            self.server._inference_lock.release()

    def test_active_native_work_outlives_legacy_queue_ceiling_and_keeps_credit(self) -> None:
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)

        def predict(pairs: list, **kwargs: object) -> list:
            entered.set()
            self.assertTrue(release.wait(2))
            return [0.5] * len(pairs)

        self.server._get_reranker = lambda name: types.SimpleNamespace(predict=predict)
        self.server_module._store_batch_result = lambda result, values, *args: values
        self.server._queue_timeout = 0.05
        payload = b'{"op":"rerank","pairs":[["synthetic","document"]]}'
        client, future = self.send(payload)
        self.assertTrue(entered.wait(1))
        admitted = next(iter(self.server._clients.values()))
        self.assertTrue(admitted.started)
        admitted.queue_expires_at = time.monotonic() - 1
        self.assertEqual(self.server._reserved_request_bytes, len(payload))
        release.set()
        self.assertTrue(self.response(client)["ok"])
        future.result(2)
        self.assert_empty()

    def test_shutdown_wakes_queued_work_but_retains_active_native_credit(self) -> None:
        entered, release = self.blocked_handler()
        client, future = self.send(b'{"op":"ping"}')
        self.assertTrue(entered.wait(1))
        self.server._stop_clients()
        self.assertEqual(len(self.server._clients), 1)
        self.assertEqual(self.server._reserved_request_bytes, 13)
        self.assertFalse(future.done())
        release.set()
        future.result(2)
        self.assert_empty()

    def test_shutdown_cancels_lock_wait_and_incomplete_body(self) -> None:
        waiting = threading.Event()
        original_lock = self.server._inference_lock

        class ObservedLock:
            def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
                if timeout >= 0:
                    waiting.set()
                return original_lock.acquire(blocking, timeout)

            def release(self) -> None:
                original_lock.release()

        self.server._inference_lock = ObservedLock()
        self.server._inference_lock.acquire()
        try:
            queued, queued_future = self.send(b'{"op":"rerank","pairs":[]}')
            self.assertTrue(waiting.wait(1), "Request did not reach the owned inference lock")
            partial, partial_future = self.send(b"", advertised=20)
            self.server._stop_clients()
            for future in (queued_future, partial_future):
                if not future.cancelled():
                    future.result(2)
            self.assert_empty()
            self.server._get_reranker.assert_not_called()
        finally:
            self.server._inference_lock.release()

    def test_enqueued_work_is_withdrawn_when_executor_thread_start_fails(self) -> None:
        self.server.handle_request = Mock(side_effect=AssertionError("Orphan request executed"))
        with patch.object(self.server._workers, "_adjust_thread_count", side_effect=RuntimeError("synthetic start failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic start failure"):
                self.send(b'{"op":"ping"}')
        self.assertTrue(self.server._stopped.is_set())
        self.assert_empty()
        self.server.handle_request.assert_not_called()
        client, future = self.send(b'{"op":"ping"}')
        self.assertIsNone(future)
        self.assert_empty()

    def test_submission_failure_retains_credit_if_existing_worker_claimed_work(self) -> None:
        client, future = self.send(b'{"op":"ping"}')
        self.response(client)
        future.result(2)
        entered, release = self.blocked_handler()

        def fail_after_claim() -> None:
            self.assertTrue(entered.wait(1))
            raise RuntimeError("synthetic failure after worker claim")

        with patch.object(self.server._workers, "_adjust_thread_count", side_effect=fail_after_claim):
            with self.assertRaisesRegex(RuntimeError, "after worker claim"):
                self.send(b'{"op":"ping"}')
        self.assertTrue(self.server._stopped.is_set())
        self.assertEqual(len(self.server._clients), 1)
        self.assertEqual(self.server._reserved_request_bytes, 13)
        release.set()
        self.server._workers.shutdown(wait=True, cancel_futures=True)
        self.assert_empty()

    def test_send_and_inference_errors_return_all_credits(self) -> None:
        for failure in (ValueError("synthetic inference failure"), TypeError("synthetic serialization failure")):
            self.server.handle_request = Mock(side_effect=failure)
            client, future = self.send(b'{"op":"ping"}')
            self.assertFalse(self.response(client)["ok"])
            future.result(2)
            self.assert_empty()
        self.server.handle_request = lambda request: {"ok": True, "value": object()}
        client, future = self.send(b'{"op":"ping"}')
        self.assertFalse(self.response(client)["ok"])
        future.result(2)
        self.assert_empty()
        self.server.handle_request = lambda request: {"ok": True}
        with patch.object(self.server, "_send_response", side_effect=BrokenPipeError):
            client, future = self.send(b'{"op":"ping"}')
            future.result(2)
        self.assert_empty()

    def test_new_client_busy_is_explicit_without_autostart(self) -> None:
        self.server._max_request_bytes = 1
        for timeout in (None, 1):
            with self.subTest(timeout=timeout):
                client, peer = self.pair()
                worker = threading.Thread(target=self.server.handle_client, args=(peer,))
                worker.start()
                try:
                    with patch.object(self.client_module, "_connect", return_value=client), \
                            patch.object(self.client_module, "_start_server", side_effect=AssertionError("Busy restarted server")):
                        with self.assertRaises(self.client_module.ModelServerBusyError) as caught:
                            self.client_module._request_with_autostart({"op": "ping"}, timeout=timeout)
                        self.assertEqual(caught.exception.retry_after_ms, 250)
                finally:
                    worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assert_empty()

    def test_client_recovers_busy_frame_after_early_broken_pipe(self) -> None:
        payload = json.dumps({"ok": False, "error_code": "server_busy", "protocol": 1}).encode()

        class EarlyRejection:
            def __init__(self) -> None:
                self.data = frame(payload)

            def settimeout(self, timeout: float) -> None:
                self.timeout = timeout

            def sendall(self, data: bytes) -> None:
                raise BrokenPipeError("synthetic early rejection")

            def recv(self, count: int) -> bytes:
                result, self.data = self.data[:count], self.data[count:]
                return result

            def close(self) -> None:
                pass

        with patch.object(self.client_module, "_connect", return_value=EarlyRejection()), \
                patch.object(self.client_module, "_start_server", side_effect=AssertionError("Busy restarted server")):
            with self.assertRaises(self.client_module.ModelServerBusyError):
                self.client_module._request_with_autostart({"op": "ping"}, timeout=1)

    def test_admitted_fast_query_still_runs_while_main_ownership_is_busy(self) -> None:
        self.server._inference_lock.acquire()
        self.server._get_fast_encoder = lambda tier: types.SimpleNamespace(encode=lambda texts, **kwargs: [[0.25]])
        try:
            client, future = self.send(b'{"op":"embed","texts":["synthetic"]}')
            self.assertEqual(self.response(client)["vectors"], [[0.25]])
            future.result(2)
            self.server._get_embed_model.assert_not_called()
        finally:
            self.server._inference_lock.release()

    def test_half_closed_valid_client_receives_legacy_protocol_response(self) -> None:
        client, peer = self.pair()
        client.sendall(frame(b'{"op":"ping"}'))
        client.shutdown(socket.SHUT_WR)
        future = self.server._dispatch_client(peer)
        self.assertEqual(self.response(client), {"ok": True, "protocol": 1})
        future.result(2)
        self.assert_empty()


if __name__ == "__main__":
    unittest.main()
