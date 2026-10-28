"""Sensor stack for the adaptive MPS throttler.

Three independent monitoring channels: MPS memory level, memory growth
rate, and thermal pressure. Each returns a status dict usable by the
state machine in the throttler.
"""

import logging
import math
import subprocess
import sys
import time

log = logging.getLogger(__name__)

THERMAL_PROBE_TIMEOUT = 0.25


def read_mps_memory(cap_gb: float) -> dict:
    """Read MPS driver-allocated memory and classify against cap."""
    try:
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return {"used_gb": None, "ratio": None, "status": "unknown"}

        used_bytes = torch.mps.driver_allocated_memory()
        used_gb = used_bytes / (1024**3)
    except (ImportError, AttributeError, RuntimeError):
        return {"used_gb": None, "ratio": None, "status": "unknown"}

    if not math.isfinite(used_gb) or used_gb < 0 or not math.isfinite(cap_gb) or cap_gb <= 0:
        return {"used_gb": None, "ratio": None, "status": "unknown"}

    ratio = used_gb / cap_gb

    if ratio >= 0.95:
        status = "critical"
    elif ratio >= 0.85:
        status = "warning"
    else:
        status = "ok"

    return {"used_gb": used_gb, "ratio": ratio, "status": status}


class GrowthRateTracker:
    """Tracks MPS memory growth rate between readings."""

    def __init__(self, cap_gb: float):
        self._cap_gb = cap_gb
        self._prev_gb: float = 0.0
        self._prev_time: float | None = None

    def update(self, current_gb: float) -> dict:
        """Record a new reading and return growth rate status."""
        now = time.monotonic()

        if not math.isfinite(current_gb) or current_gb < 0:
            return {"slope_gb_per_20s": None, "slope_pct": None, "status": "unknown"}

        if self._prev_time is None:
            self._prev_gb = current_gb
            self._prev_time = now
            return {"slope_gb_per_20s": None, "slope_pct": None, "status": "unknown"}

        dt = now - self._prev_time
        if dt <= 0:
            return {"slope_gb_per_20s": None, "slope_pct": None, "status": "unknown"}

        raw_slope = (current_gb - self._prev_gb) / dt
        slope_20s = raw_slope * 20.0
        slope_pct = (slope_20s / self._cap_gb * 100.0) if self._cap_gb > 0 else 0.0

        self._prev_gb = current_gb
        self._prev_time = now

        if slope_pct >= 10.0:
            status = "critical"
        elif slope_pct >= 5.0:
            status = "warning"
        else:
            status = "ok"

        return {"slope_gb_per_20s": slope_20s, "slope_pct": slope_pct, "status": status}


def read_thermal_pressure() -> dict:
    """Read macOS thermal pressure without treating missing data as healthy."""
    if sys.platform != "darwin":
        return {"scheduler_limit": None, "status": "unsupported", "required": False}
    try:
        proc = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=THERMAL_PROBE_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {"scheduler_limit": None, "status": "unknown"}

    if proc.returncode != 0:
        return {"scheduler_limit": None, "status": "unknown"}
    limits: list[int] = []
    invalid_field = False
    for line in proc.stdout.splitlines():
        if "CPU_Scheduler_Limit" in line or "CPU_Speed_Limit" in line:
            parts = line.split("=")
            if len(parts) != 2:
                invalid_field = True
                continue
            try:
                limit = int(parts[1].strip())
            except ValueError:
                invalid_field = True
                continue
            if not 0 <= limit <= 100:
                invalid_field = True
                continue
            limits.append(limit)

    if not limits:
        return {"scheduler_limit": None, "status": "unknown"}
    limit = min(limits)

    # Incomplete telemetry cannot establish health or erase known throttling.
    if limit <= 70:
        status = "critical"
    elif limit < 100:
        status = "warning"
    elif invalid_field:
        return {"scheduler_limit": None, "status": "unknown"}
    else:
        status = "ok"

    return {"scheduler_limit": limit, "status": status}
