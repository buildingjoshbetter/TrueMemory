"""Tests for issue #592: surface silent degradation in truememory_status.

Verifies that truememory_status returns a ``degradation`` section that
surfaces per-subsystem health (model server, reranker, HyDE LLM,
encoding gate, vectors).
"""
from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_status(status_id: int = 0) -> dict:
    """Call truememory_status and parse the JSON result."""
    from truememory.mcp_server import truememory_status
    raw = truememory_status(status_id)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStatusIncludesDegradation:
    """truememory_status must return a 'degradation' key."""

    def test_degradation_section_present(self):
        """The response contains both 'rebuild' and 'degradation' keys."""
        result = _call_status()
        assert "degradation" in result, "missing 'degradation' key in status"
        assert "rebuild" in result, "missing 'rebuild' key in status"

    def test_degradation_subsystems(self):
        """The degradation section covers all expected subsystems."""
        result = _call_status()
        deg = result["degradation"]
        expected = {"model_server", "reranker", "hyde_llm", "vectors", "encoding_gate"}
        assert expected.issubset(set(deg.keys())), (
            f"missing subsystems: {expected - set(deg.keys())}"
        )

    def test_healthy_status_shows_ok(self):
        """When nothing is degraded, subsystems report 'ok'."""
        # Reset any lingering error state.
        import truememory.mcp_server as srv
        srv._reranker_last_error = None
        with srv._llm_error_lock:
            srv._llm_last_error.clear()
        with srv._encoding_gate_error_lock:
            srv._encoding_gate_last_error = None
            srv._encoding_gate_degradation_count = 0

        result = _call_status()
        deg = result["degradation"]

        assert deg["reranker"]["status"] == "ok"
        assert deg["hyde_llm"]["status"] == "ok"
        assert deg["vectors"]["status"] == "ok"
        assert deg["encoding_gate"]["status"] == "ok"
        assert deg["encoding_gate"]["degradation_count"] == 0


class TestRerankerDegradation:
    """Reranker errors surface in the degradation section."""

    def test_reranker_error_surfaces(self):
        import truememory.mcp_server as srv
        srv._record_reranker_error("RuntimeError: MPS OOM")
        try:
            result = _call_status()
            rr = result["degradation"]["reranker"]
            assert rr["status"] == "degraded"
            assert "MPS OOM" in (rr["last_error"] or "")
        finally:
            srv._clear_reranker_error()


class TestHydeDegradation:
    """HyDE LLM errors surface in the degradation section."""

    def test_hyde_error_surfaces(self):
        import truememory.mcp_server as srv
        srv._record_llm_error("anthropic", Exception("rate limit exceeded"))
        try:
            result = _call_status()
            hyde = result["degradation"]["hyde_llm"]
            assert hyde["status"] == "degraded"
            assert hyde["last_error_by_provider"] is not None
            assert "anthropic" in hyde["last_error_by_provider"]
        finally:
            with srv._llm_error_lock:
                srv._llm_last_error.clear()


class TestEncodingGateDegradation:
    """Encoding gate / PE degradation surfaces in the degradation section."""

    def test_encoding_gate_error_surfaces(self):
        import truememory.mcp_server as srv
        srv.record_encoding_gate_error("ImportError: torch not found")
        try:
            result = _call_status()
            eg = result["degradation"]["encoding_gate"]
            assert eg["status"] == "degraded"
            assert "torch not found" in (eg["last_error"] or "")
            assert eg["degradation_count"] >= 1
        finally:
            srv.clear_encoding_gate_error()
            with srv._encoding_gate_error_lock:
                srv._encoding_gate_degradation_count = 0

    def test_degradation_count_increments(self):
        import truememory.mcp_server as srv
        with srv._encoding_gate_error_lock:
            srv._encoding_gate_degradation_count = 0
            srv._encoding_gate_last_error = None
        srv.record_encoding_gate_error("err1")
        srv.record_encoding_gate_error("err2")
        srv.record_encoding_gate_error("err3")
        try:
            result = _call_status()
            eg = result["degradation"]["encoding_gate"]
            assert eg["degradation_count"] == 3
        finally:
            srv.clear_encoding_gate_error()
            with srv._encoding_gate_error_lock:
                srv._encoding_gate_degradation_count = 0


class TestModelServerDegradation:
    """Model server health surfaces in the degradation section."""

    def test_model_server_down_shows_degraded(self):
        """When the model server is not running, status is degraded."""
        with patch("truememory.mcp_server._build_model_server_health") as mock_health:
            mock_health.return_value = {
                "status": "degraded",
                "running": False,
                "device": "auto",
                "sticky_cpu": None,
                "socket_exists": False,
            }
            result = _call_status()
            ms = result["degradation"]["model_server"]
            assert ms["status"] == "degraded"
            assert ms["running"] is False

    def test_model_server_ok(self):
        """When the model server is alive with no OOM, status is ok."""
        with patch("truememory.mcp_server._build_model_server_health") as mock_health:
            mock_health.return_value = {
                "status": "ok",
                "running": True,
                "device": "mps",
                "sticky_cpu": None,
                "socket_exists": True,
            }
            result = _call_status()
            ms = result["degradation"]["model_server"]
            assert ms["status"] == "ok"
            assert ms["running"] is True

    def test_model_server_sticky_cpu_shows_degraded(self):
        """Sticky CPU degradation from OOM is reported."""
        with patch("truememory.mcp_server._build_model_server_health") as mock_health:
            mock_health.return_value = {
                "status": "degraded",
                "running": True,
                "device": "auto",
                "sticky_cpu": ["embed", "rerank"],
                "socket_exists": True,
            }
            result = _call_status()
            ms = result["degradation"]["model_server"]
            assert ms["status"] == "degraded"
            assert ms["sticky_cpu"] == ["embed", "rerank"]


class TestModelServerStatusFile:
    """Model server writes sticky-CPU state to a status file (issue #592)."""

    def test_write_status_file(self, tmp_path, monkeypatch):
        """_mark_sticky_cpu writes model_server.status."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        (tmp_path / ".truememory").mkdir()

        from truememory.model_server import ModelServer
        server = ModelServer()
        with server._lock:
            server._mark_sticky_cpu("embed")

        status_path = tmp_path / ".truememory" / "model_server.status"
        assert status_path.exists()
        data = json.loads(status_path.read_text())
        assert "embed" in data["sticky_cpu"]
