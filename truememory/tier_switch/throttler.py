"""Adaptive MPS throttler for tier-switch re-embedding.

Three-channel monitoring (MPS memory level, growth rate, thermal
pressure) with a PROBING/STABLE/BACKOFF state machine. Starts at
batch=1, ramps up slowly, backs off quickly.
"""

import gc
import logging
import time

import psutil

from truememory.tier_switch.sensors import (
    GrowthRateTracker,
    read_mps_memory,
    read_thermal_pressure,
)
from truememory.tier_switch.state_machine import ThrottlerStateMachine

log = logging.getLogger(__name__)

_MACHINE_PROFILES = {
    # (min_gb, max_gb): (ratio, start, max_batch, ramp_step)
    (0, 12): (0.50, 1, 4, 1),
    (12, 20): (0.50, 1, 8, 1),
    (20, 30): (0.50, 1, 12, 2),
    (30, 1024): (0.55, 1, 16, 2),
}


def _get_profile(total_gb: float) -> tuple[float, int, int, int]:
    for (lo, hi), profile in _MACHINE_PROFILES.items():
        if lo <= total_gb < hi:
            return profile
    return (0.50, 1, 12, 2)


class DynamicThrottler:
    """Adaptive 3-channel throttler with state machine."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.total_gb = psutil.virtual_memory().total / (1024**3)

        ratio, start, max_batch, ramp_step = _get_profile(self.total_gb)
        self.mps_cap_gb = self.total_gb * ratio

        self.state_machine = ThrottlerStateMachine(
            start_batch=start,
            max_batch=max_batch,
            ramp_step=ramp_step,
        )

        self.growth_tracker = GrowthRateTracker(cap_gb=self.mps_cap_gb)

        self.items_processed = 0
        self.start_time = time.time()
        self.batch_times: list[float] = []
        self.last_throttle_time = 0.0  # backward compat: worker sets this on OOM
        self._last_readings: dict = {}

        log.info(
            "Throttler init: device=%s total_ram=%.0fGB mps_cap=%.1fGB "
            "start=%d max=%d step=%d",
            device, self.total_gb, self.mps_cap_gb,
            start, max_batch, ramp_step,
        )

    @property
    def batch_size(self) -> int:
        return self.state_machine.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self.state_machine.batch_size = value

    def before_batch(self) -> tuple[int, dict]:
        """Check sensors, update state machine, return (batch_size, metrics)."""
        self.state_machine.on_batch_complete()

        if self.state_machine.should_safety_check():
            readings = self._read_all_channels()
            self._last_readings = readings
            self.state_machine.safety_check(readings)

        if self.state_machine.should_ramp_check():
            means = self._triple_sample()
            self.state_machine.ramp_up(means)

        sleep_time = 0.05 + self.state_machine.batch_size * 0.02
        time.sleep(sleep_time)

        metrics = self._build_metrics()
        return self.state_machine.batch_size, metrics

    def after_batch(self, batch_items: int, batch_time: float):
        """Record batch completion for throughput tracking."""
        self.items_processed += batch_items
        self.batch_times.append(batch_time)
        if len(self.batch_times) > 20:
            self.batch_times.pop(0)

    def get_throughput(self) -> float:
        """Items per second since start."""
        elapsed = time.time() - self.start_time
        return self.items_processed / elapsed if elapsed > 0 else 0.0

    def get_eta_seconds(self, remaining: int) -> float:
        """Estimated seconds to process remaining items."""
        throughput = self.get_throughput()
        return remaining / throughput if throughput > 0 else float("inf")

    def should_flush_cache(self) -> bool:
        """Return True only on WARNING/BACKOFF — not during normal PROBING."""
        return self.state_machine.state in (
            ThrottlerStateMachine.STABLE,
            ThrottlerStateMachine.BACKOFF,
        )

    def on_oom(self):
        """Handle OOM by triggering BACKOFF in the state machine."""
        self.state_machine._do_backoff("oom", {"status": "critical"})

    def _read_all_channels(self) -> dict:
        """Single quick reading of all 3 channels."""
        try:
            mps = read_mps_memory(self.mps_cap_gb)
        except Exception:
            mps = {"used_gb": 0.0, "ratio": 0.0, "status": "ok"}

        try:
            growth = self.growth_tracker.update(mps["used_gb"])
        except Exception:
            growth = {"slope_gb_per_20s": 0.0, "slope_pct": 0.0, "status": "ok"}

        try:
            thermal = read_thermal_pressure()
        except Exception:
            thermal = {"scheduler_limit": 100, "status": "ok"}

        return {
            "mps_level": mps,
            "growth_rate": growth,
            "thermal": thermal,
        }

    def _triple_sample(self) -> dict:
        """Take 3 readings 10s apart, return mean-based status."""
        samples = []
        for i in range(3):
            if i > 0:
                time.sleep(10)
            samples.append(self._read_all_channels())
        return self._compute_means(samples)

    def _compute_means(self, samples: list[dict]) -> dict:
        """Compute mean status across 3 samples.

        For ramp-up: ANY warning/critical in any sample → mean is that level.
        """
        result = {}
        for channel in ("mps_level", "growth_rate", "thermal"):
            statuses = [s[channel]["status"] for s in samples]
            if "critical" in statuses:
                result[channel] = {"status": "critical"}
            elif "warning" in statuses:
                result[channel] = {"status": "warning"}
            else:
                result[channel] = {"status": "ok"}
        return result

    def _build_metrics(self) -> dict:
        """Build metrics dict for status reporting."""
        readings = self._last_readings
        return {
            "batch_size": self.state_machine.batch_size,
            "state": self.state_machine.state,
            "mps_used_gb": readings.get("mps_level", {}).get("used_gb", 0.0),
            "mps_ratio": readings.get("mps_level", {}).get("ratio", 0.0),
            "growth_slope_pct": readings.get("growth_rate", {}).get("slope_pct", 0.0),
            "thermal_limit": readings.get("thermal", {}).get("scheduler_limit", 100),
            "good_streak": self.state_machine.good_streak,
        }

    @staticmethod
    def flush_gpu_cache():
        """Flush MPS/CUDA cache and run garbage collection."""
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
