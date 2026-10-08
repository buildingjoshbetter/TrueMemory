"""Tests for tier_switch/sensors.py — the three monitoring channels."""

import subprocess
import time
from unittest.mock import MagicMock, patch

from truememory.tier_switch.sensors import (
    GrowthRateTracker,
    read_mps_memory,
    read_thermal_pressure,
)


# ── Channel 1: MPS Memory Level ──


def test_mps_memory_ok():
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.mps.driver_allocated_memory.return_value = int(2 * 1024**3)
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = read_mps_memory(cap_gb=12.0)
    assert result["status"] == "ok"
    assert abs(result["used_gb"] - 2.0) < 0.01
    assert abs(result["ratio"] - 2.0 / 12.0) < 0.01


def test_mps_memory_warning():
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.mps.driver_allocated_memory.return_value = int(10.5 * 1024**3)
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = read_mps_memory(cap_gb=12.0)
    assert result["status"] == "warning"
    assert abs(result["ratio"] - 10.5 / 12.0) < 0.01


def test_mps_memory_critical():
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.mps.driver_allocated_memory.return_value = int(11.5 * 1024**3)
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = read_mps_memory(cap_gb=12.0)
    assert result["status"] == "critical"
    assert result["ratio"] >= 0.95


def test_mps_memory_no_torch():
    with patch.dict("sys.modules", {"torch": None}):
        result = read_mps_memory(cap_gb=12.0)
    assert result["status"] == "unknown"
    assert result["used_gb"] is None


# ── Channel 2: Growth Rate Tracker ──


def test_growth_rate_first_reading_needs_another_observation():
    tracker = GrowthRateTracker(cap_gb=12.0)
    result = tracker.update(5.0)
    assert result["status"] == "unknown"
    assert result["slope_gb_per_20s"] is None


def test_growth_rate_stable():
    tracker = GrowthRateTracker(cap_gb=12.0)
    tracker._prev_gb = 5.0
    tracker._prev_time = time.monotonic() - 20
    result = tracker.update(5.1)
    assert result["status"] == "ok"
    assert result["slope_pct"] < 3.0


def test_growth_rate_warning():
    tracker = GrowthRateTracker(cap_gb=12.0)
    tracker._prev_gb = 5.0
    tracker._prev_time = time.monotonic() - 20
    result = tracker.update(5.7)
    assert result["status"] == "warning"
    assert result["slope_pct"] >= 5.0


def test_growth_rate_critical():
    tracker = GrowthRateTracker(cap_gb=12.0)
    tracker._prev_gb = 5.0
    tracker._prev_time = time.monotonic() - 20
    result = tracker.update(6.5)
    assert result["status"] == "critical"
    assert result["slope_pct"] >= 10.0


# ── Channel 3: Thermal Pressure ──


def test_thermal_ok():
    fake_output = "CPU_Scheduler_Limit = 100\n"
    with patch("sys.platform", "darwin"), patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=fake_output, stderr=""
        )
        result = read_thermal_pressure()
    assert result["status"] == "ok"
    assert result["scheduler_limit"] == 100


def test_thermal_warning():
    fake_output = "CPU_Scheduler_Limit = 85\n"
    with patch("sys.platform", "darwin"), patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=fake_output, stderr=""
        )
        result = read_thermal_pressure()
    assert result["status"] == "warning"
    assert result["scheduler_limit"] == 85


def test_thermal_critical():
    fake_output = "CPU_Scheduler_Limit = 60\n"
    with patch("sys.platform", "darwin"), patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=fake_output, stderr=""
        )
        result = read_thermal_pressure()
    assert result["status"] == "critical"
    assert result["scheduler_limit"] == 60


def test_thermal_command_fails():
    with patch("sys.platform", "darwin"), patch("subprocess.run", side_effect=subprocess.TimeoutExpired("pmset", 0.25)):
        result = read_thermal_pressure()
    assert result["status"] == "unknown"
    assert result["scheduler_limit"] is None
