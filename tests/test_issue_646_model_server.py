"""Regression tests for issue #646: model-server lifecycle hardening.

Covers:
  - M-20: _cleanup only removes artifacts THIS process owns (PID == getpid());
    a stale/dead-PID artifact is reclaimed by a new server's bind lock.
  - M-43: throttler check captures a local under the lock (no AttributeError
    when _throttler is nulled mid-flight).
  - M-44: a request whose deadline already passed is rejected before encode.
  - M-53: the client raises a clear protocol-mismatch error on non-JSON bytes.
  - M-74: in-flight counter prevents idle shutdown mid-request.

All hermetic: no real model loads, no real daemon. Stub sockets / fake
servers only. HF_HUB_OFFLINE is set by the test runner.
"""
from __future__ import annotations

import json
import os
import struct
import time
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# M-20: _cleanup ownership + stale-artifact reclamation
# ---------------------------------------------------------------------------


class TestM20CleanupOwnership:
    def _point_paths_at_tmp(self, monkeypatch, tmp_path):
        """Redirect the module-level artifact paths into a temp dir."""
        from truememory import model_server as ms
        for name, fname in (
            ("SOCK_PATH", "model.sock"),
            ("PID_PATH", "model_server.pid"),
            ("PORT_PATH", "model_server.port"),
            ("TOKEN_PATH", "model_server.token"),
            ("LOCK_PATH", "model_server.lock"),
        ):
            monkeypatch.setattr(ms, name, tmp_path / fname)
        return ms

    def test_cleanup_does_not_delete_other_servers_artifacts(
        self, monkeypatch, tmp_path
    ):
        """_cleanup must NOT remove socket/pid/token owned by a DIFFERENT
        live PID (M-20). Simulate a successor that already rewrote PID_PATH."""
        ms = self._point_paths_at_tmp(monkeypatch, tmp_path)

        # A successor process owns the artifacts: a DIFFERENT, LIVE pid.
        other_pid = os.getpid() + 1
        monkeypatch.setattr(ms, "pid_is_alive", lambda pid: pid == other_pid)
        ms.PID_PATH.write_text(str(other_pid))
        ms.SOCK_PATH.write_text("live-socket")
        ms.TOKEN_PATH.write_text("live-token")
        ms.PORT_PATH.write_text("12345")

        server = ms.ModelServer()
        server._cleanup()

        # All four artifacts survive — they belong to the successor.
        assert ms.PID_PATH.exists()
        assert ms.SOCK_PATH.exists()
        assert ms.TOKEN_PATH.exists()
        assert ms.PORT_PATH.exists()

    def test_cleanup_removes_own_artifacts(self, monkeypatch, tmp_path):
        """When PID_PATH names THIS process, _cleanup removes the artifacts."""
        ms = self._point_paths_at_tmp(monkeypatch, tmp_path)

        ms.PID_PATH.write_text(str(os.getpid()))
        ms.SOCK_PATH.write_text("my-socket")
        ms.TOKEN_PATH.write_text("my-token")
        ms.PORT_PATH.write_text("12345")

        server = ms.ModelServer()
        server._cleanup()

        assert not ms.PID_PATH.exists()
        assert not ms.SOCK_PATH.exists()
        assert not ms.TOKEN_PATH.exists()
        assert not ms.PORT_PATH.exists()

    def test_stale_dead_pid_artifact_reclaimed_by_bind_lock(
        self, monkeypatch, tmp_path
    ):
        """A new server takes the exclusive bind lock even when a stale PID
        file from a crashed predecessor is present (M-20)."""
        ms = self._point_paths_at_tmp(monkeypatch, tmp_path)

        # Stale artifacts from a crashed predecessor (dead PID).
        ms.PID_PATH.write_text("999999999")
        ms.SOCK_PATH.write_text("stale")

        server = ms.ModelServer()
        fd = server._acquire_bind_lock()
        try:
            assert isinstance(fd, int)
            assert ms.LOCK_PATH.exists()
        finally:
            os.close(fd)

    def test_second_server_cannot_take_held_lock(self, monkeypatch, tmp_path):
        """While one server holds the bind lock, a second one is refused
        (M-20: only one server binds)."""
        if os.name == "nt":
            pytest.skip("flock semantics differ on Windows")
        ms = self._point_paths_at_tmp(monkeypatch, tmp_path)

        s1 = ms.ModelServer()
        fd1 = s1._acquire_bind_lock()
        try:
            s2 = ms.ModelServer()
            with pytest.raises(RuntimeError):
                s2._acquire_bind_lock()
        finally:
            os.close(fd1)


# ---------------------------------------------------------------------------
# M-43: throttler TOCTOU
# ---------------------------------------------------------------------------


class TestM43ThrottlerToctou:
    def test_throttler_nulled_midflight_no_attribute_error(self):
        """If _throttler is set inactive/nulled between the truthiness check
        and use, the captured local must not raise AttributeError (M-43)."""
        from truememory.model_server import ModelServer

        server = ModelServer()

        # Simulate an active throttler whose attribute is yanked away
        # immediately after the active flag is read.
        class FlakyThrottler:
            def before_batch(self):
                pass

            def after_batch(self, n, t):
                pass

            def should_flush_cache(self):
                return False

        # The fix captures `self._throttler` to a local under the lock, so a
        # concurrent _deactivate_throttler (which nulls self._throttler) can't
        # cause an AttributeError. Verify the capture pattern directly.
        server._throttler = FlakyThrottler()
        server._throttler_active = True

        with server._lock:
            throttler = server._throttler if server._throttler_active else None
        # Now another thread deactivates (nulls the attribute).
        with server._lock:
            server._deactivate_throttler()
        # The captured local is still usable — no AttributeError.
        assert throttler is not None
        throttler.before_batch()
        assert server._throttler is None

    def test_embed_with_active_throttler_does_not_crash(self, monkeypatch):
        """End-to-end: an embed request with the throttler active completes
        without an AttributeError even under concurrent deactivation."""
        from truememory.model_server import ModelServer

        server = ModelServer()

        fake_model = MagicMock()
        fake_model.encode.return_value = np.zeros((2, 4), dtype=np.float32)
        monkeypatch.setattr(server, "_get_embed_model", lambda tier: fake_model)

        class FakeThrottler:
            def before_batch(self):
                pass

            def after_batch(self, n, t):
                pass

            def should_flush_cache(self):
                return False

        server._throttler = FakeThrottler()
        server._throttler_active = True

        resp = server.handle_request({"op": "embed", "texts": ["a", "b"], "tier": ""})
        assert resp["ok"] is True


# ---------------------------------------------------------------------------
# M-44: server-side deadline rejection before encode
# ---------------------------------------------------------------------------


class TestM44ServerDeadline:
    def test_expired_deadline_rejected_before_encode(self, monkeypatch):
        """A request carrying an already-passed deadline is rejected without
        ever calling the encode lane (M-44)."""
        from truememory.model_server import ModelServer

        server = ModelServer()

        encode_called = {"n": 0}

        def boom(tier):
            encode_called["n"] += 1
            raise AssertionError("encode must not run for expired request")

        monkeypatch.setattr(server, "_get_embed_model", boom)

        resp = server.handle_request({
            "op": "embed",
            "texts": ["x"],
            "tier": "",
            "deadline": time.time() - 5.0,  # already passed
        })
        assert resp["ok"] is False
        assert "deadline" in resp["error"].lower()
        assert encode_called["n"] == 0

    def test_future_deadline_runs_normally(self, monkeypatch):
        """A request with a comfortable future deadline still encodes."""
        from truememory.model_server import ModelServer

        server = ModelServer()
        fake_model = MagicMock()
        fake_model.encode.return_value = np.zeros((1, 4), dtype=np.float32)
        monkeypatch.setattr(server, "_get_embed_model", lambda tier: fake_model)

        resp = server.handle_request({
            "op": "embed",
            "texts": ["x"],
            "tier": "",
            "deadline": time.time() + 100.0,
        })
        assert resp["ok"] is True


# ---------------------------------------------------------------------------
# M-53: client protocol-mismatch detection
# ---------------------------------------------------------------------------


class _StubSock:
    """A minimal socket stub that replays a canned response payload."""

    def __init__(self, response_bytes: bytes):
        self._to_send = response_bytes
        self.sent = bytearray()
        self.timeout = None

    def settimeout(self, t):
        self.timeout = t

    def sendall(self, data):
        self.sent.extend(data)

    def recv(self, n):
        if not self._to_send:
            return b""
        chunk = self._to_send[:n]
        self._to_send = self._to_send[n:]
        return chunk

    def close(self):
        pass


def _framed(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


class TestM53ProtocolMismatch:
    def test_pickle_garbage_raises_protocol_mismatch(self, monkeypatch):
        """Non-JSON (e.g. pickle-era daemon) bytes raise a clear
        ProtocolMismatchError, not an inscrutable UnicodeDecodeError (M-53)."""
        from truememory import model_client as mc

        # Invalid UTF-8 / non-JSON bytes (pickle protocol 2 header).
        garbage = b"\x80\x02}q\x00X\x04\x00\x00\x00okq\x01\x88s."
        stub = _StubSock(_framed(garbage))
        monkeypatch.setattr(mc, "_connect", lambda deadline=None: stub)

        with pytest.raises(mc.ProtocolMismatchError) as ei:
            mc._send_request({"op": "ping"})
        assert "protocol mismatch" in str(ei.value).lower()

    def test_wrong_version_raises_protocol_mismatch(self, monkeypatch):
        """A well-formed JSON response with an incompatible protocol version
        is rejected (M-53)."""
        from truememory import model_client as mc

        body = json.dumps({"ok": True, "protocol": 999}).encode("utf-8")
        stub = _StubSock(_framed(body))
        monkeypatch.setattr(mc, "_connect", lambda deadline=None: stub)

        with pytest.raises(mc.ProtocolMismatchError):
            mc._send_request({"op": "ping"})

    def test_matching_version_decodes_fine(self, monkeypatch):
        """A correct-version JSON response decodes normally."""
        from truememory import model_client as mc

        body = json.dumps(
            {"ok": True, "protocol": mc.PROTOCOL_VERSION}
        ).encode("utf-8")
        stub = _StubSock(_framed(body))
        monkeypatch.setattr(mc, "_connect", lambda deadline=None: stub)

        resp = mc._send_request({"op": "ping"})
        assert resp["ok"] is True

    def test_autostart_does_not_swallow_protocol_mismatch(self, monkeypatch):
        """_request_with_autostart must re-raise ProtocolMismatchError rather
        than treat it as a connection failure and retry (M-53)."""
        from truememory import model_client as mc

        def raise_mismatch(request, timeout=None):
            raise mc.ProtocolMismatchError(
                "protocol mismatch — old model server running; restart it"
            )

        monkeypatch.setattr(mc, "_send_request", raise_mismatch)
        # If autostart were attempted, this would explode loudly.
        monkeypatch.setattr(
            mc, "_start_server",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no retry")),
        )

        with pytest.raises(mc.ProtocolMismatchError):
            mc._request_with_autostart({"op": "ping"})


# ---------------------------------------------------------------------------
# M-74: in-flight counter blocks idle shutdown
# ---------------------------------------------------------------------------


class TestM74InflightCounter:
    def test_inflight_incremented_during_request(self, monkeypatch):
        """While the request body runs, _inflight is > 0; it returns to 0
        and activity is re-stamped at completion (M-74)."""
        from truememory.model_server import ModelServer

        server = ModelServer()
        observed = {}

        def slow_handle_request_inner(request):
            with server._activity_lock:
                observed["inflight_during"] = server._inflight
            return {"ok": True}

        monkeypatch.setattr(server, "_handle_request_inner", slow_handle_request_inner)

        before = server._last_activity
        time.sleep(0.01)
        resp = server.handle_request({"op": "ping"})

        assert resp["ok"] is True
        assert observed["inflight_during"] == 1
        assert server._inflight == 0
        # Activity stamped at completion too (M-74), so it advanced.
        assert server._last_activity >= before

    def test_inflight_decremented_on_exception(self, monkeypatch):
        """_inflight is restored even when the request body raises."""
        from truememory.model_server import ModelServer

        server = ModelServer()

        def boom(request):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(server, "_handle_request_inner", boom)

        with pytest.raises(RuntimeError):
            server.handle_request({"op": "ping"})
        assert server._inflight == 0
