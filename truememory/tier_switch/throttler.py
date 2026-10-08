"""Adaptive MPS throttler for tier-switch re-embedding.

Three-channel monitoring (MPS memory level, growth rate, thermal
pressure) with a PROBING/STABLE/BACKOFF state machine. Starts at
batch=1, ramps up slowly, backs off quickly.
"""

import gc
import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import psutil

from truememory.tier_switch.sensors import (
    GrowthRateTracker,
    read_mps_memory,
    read_thermal_pressure,
)
from truememory.tier_switch.state_machine import ThrottlerStateMachine

if TYPE_CHECKING:
    from truememory.mps_utils import MPSMemoryBudget

log = logging.getLogger(__name__)

_MACHINE_PROFILES = {
    # (min_gb, max_gb): (start, max_batch, ramp_step). RAM sizes batches only.
    (0, 12): (1, 4, 1),
    (12, 20): (1, 8, 1),
    (20, 30): (1, 12, 2),
    (30, 1024): (1, 16, 2),
}


def _get_profile(total_gb: float) -> tuple[int, int, int]:
    for (lo, hi), profile in _MACHINE_PROFILES.items():
        if lo <= total_gb < hi:
            return profile
    return (1, 12, 2)


class DynamicThrottler:
    """Adaptive 3-channel throttler with state machine."""

    SAMPLE_SPACING = 10.0
    SAMPLE_MAX_AGE = 30.0

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.total_gb = psutil.virtual_memory().total / (1024**3)

        start, max_batch, ramp_step = _get_profile(self.total_gb)

        self.state_machine = ThrottlerStateMachine(
            start_batch=start,
            max_batch=max_batch,
            ramp_step=ramp_step,
        )

        self.growth_tracker: GrowthRateTracker | None = None
        self._growth_cap_gb: float | None = None
        self._state_lock = threading.RLock()
        self._sample_lock = threading.Lock()
        self._samples: deque[tuple[float, dict]] = deque(maxlen=3)
        self._last_sample_time: float | None = None
        self._fault_generation = 0

        self.items_processed = 0
        self.start_time = time.monotonic()
        self.batch_times: list[float] = []
        self.last_throttle_time = 0.0  # backward compat: worker sets this on OOM
        self._last_readings: dict = {}

        log.info(
            "Throttler init: device=%s total_ram=%.0fGiB "
            "start=%d max=%d step=%d",
            device, self.total_gb,
            start, max_batch, ramp_step,
        )

    def _budget_snapshot(self) -> "MPSMemoryBudget | None":
        # Reading telemetry must never initialize a local allocator for a proxy
        # or race model setup. The loader publishes only after the setter works.
        # A shared daemon can have CPU embedding and MPS reranking (or the
        # reverse). A configured budget describes the process's MPS allocator,
        # regardless of which device this admission's model currently uses.
        from truememory.mps_utils import get_mps_memory_budget
        return get_mps_memory_budget()

    @property
    def mps_cap_gb(self) -> float | None:
        with self._state_lock:
            budget = self._budget_snapshot()
            return budget.effective_bytes / (1024**3) if budget and budget.effective_bytes else None

    @property
    def batch_size(self) -> int:
        with self._state_lock:
            return self.state_machine.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        with self._state_lock:
            self.state_machine.batch_size = value

    def before_batch(self) -> tuple[int, dict]:
        """Use bounded observations; never wait for a three-sample ramp window."""
        with self._state_lock:
            self.state_machine.on_batch_complete()
            self._expire_samples(time.monotonic())
        self._sample_if_due()

        # Retain duty-cycle pacing, outside both locks. Ramping no longer adds
        # two ten-second sleeps to an admission. Native sensor calls are separate.
        paced_size = self.batch_size
        sleep_time = 0.05 + paced_size * 0.02
        time.sleep(sleep_time)
        with self._state_lock:
            metrics = self._build_metrics()
            # A concurrent backoff applies immediately; a concurrent increase
            # cannot grant more work than this admission's pacing covered.
            size = min(paced_size, self.state_machine.batch_size)
            metrics["batch_size"] = size
            return size, metrics

    def _expire_samples(self, now: float) -> None:
        """Caller holds the state lock; old evidence cannot authorize a ramp."""
        while self._samples and now - self._samples[0][0] > self.SAMPLE_MAX_AGE:
            self._samples.popleft()
        if self._last_sample_time is not None and now - self._last_sample_time > self.SAMPLE_MAX_AGE:
            self.state_machine.good_streak = 0

    def _sample_due(self, now: float) -> bool:
        if self.state_machine.should_safety_check():
            return True
        if self._last_sample_time is None:
            return False
        age = now - self._last_sample_time
        # Collect fresh evidence before ramp eligibility: eligibility itself
        # requires three good observations and cannot trigger their collection.
        return age >= self.SAMPLE_SPACING

    def _sample_if_due(self) -> None:
        # Only one caller samples. Other admissions keep the current bound;
        # they never line up behind the sensor subprocess or create a monitor.
        if not self._sample_lock.acquire(blocking=False):
            return
        try:
            with self._state_lock:
                if not self._sample_due(time.monotonic()):
                    return
                generation = self._fault_generation
                budget = self._budget_snapshot()
            readings = self._read_all_channels()
            now = time.monotonic()
            with self._state_lock:
                # An OOM during the probe invalidates its pre-fault evidence.
                if generation != self._fault_generation:
                    return
                if budget is not self._budget_snapshot():
                    # The first MPS model may finish setup while a CPU-only
                    # thermal probe is pending. Its earlier unsupported MPS
                    # observations cannot authorize a ramp for the new state.
                    self._samples.clear()
                    self._last_readings = {}
                    self._last_sample_time = None
                    self.state_machine.good_streak = 0
                    return
                self._last_readings = readings
                self._last_sample_time = now
                self._expire_samples(now)
                self.state_machine.safety_check(readings)
                if not self.state_machine.all_required_channels_ok(readings):
                    self._samples.clear()
                    return
                if not self._samples or now - self._samples[-1][0] >= self.SAMPLE_SPACING:
                    self._samples.append((now, readings))
                if len(self._samples) == 3 and self.state_machine.should_ramp_check():
                    self.state_machine.ramp_up(self._compute_means([sample for _, sample in self._samples]))
                    self._samples.clear()
        finally:
            self._sample_lock.release()

    def after_batch(self, batch_items: int, batch_time: float):
        """Record batch completion for throughput tracking."""
        with self._state_lock:
            self.items_processed += batch_items
            self.batch_times.append(batch_time)
            if len(self.batch_times) > 20:
                self.batch_times.pop(0)

    def get_throughput(self) -> float:
        """Items per second since start."""
        with self._state_lock:
            elapsed = time.monotonic() - self.start_time
            return self.items_processed / elapsed if elapsed > 0 else 0.0

    def get_eta_seconds(self, remaining: int) -> float:
        """Estimated seconds to process remaining items."""
        throughput = self.get_throughput()
        return remaining / throughput if throughput > 0 else float("inf")

    def should_flush_cache(self) -> bool:
        """Return True only on WARNING/BACKOFF — not during normal PROBING."""
        with self._state_lock:
            return self.state_machine.state in (
                ThrottlerStateMachine.STABLE,
                ThrottlerStateMachine.BACKOFF,
            )

    def on_oom(self):
        """Handle OOM by triggering BACKOFF in the state machine."""
        with self._state_lock:
            self._fault_generation += 1
            self._samples.clear()
            self.state_machine._do_backoff("oom", {"status": "critical"})

    def _read_all_channels(self) -> dict:
        """Observe applicable channels; unavailable data must not authorize ramping."""
        mps = {"used_gb": None, "ratio": None, "status": "unsupported", "required": False}
        growth = {"slope_gb_per_20s": None, "slope_pct": None,
                  "status": "unsupported", "required": False}
        with self._state_lock:
            budget = self._budget_snapshot()
            is_mps = self.device == "mps" or budget is not None
            cap_gb = budget.effective_bytes / (1024**3) if budget and budget.effective_bytes else None
            if cap_gb != self._growth_cap_gb:
                self.growth_tracker = GrowthRateTracker(cap_gb=cap_gb) if cap_gb else None
                self._growth_cap_gb = cap_gb
            tracker = self.growth_tracker
        if is_mps:
            # Unlimited or unconfigured is not evidence of finite headroom.
            mps = {"used_gb": None, "ratio": None, "status": "unknown"}
            growth = {"slope_gb_per_20s": None, "slope_pct": None, "status": "unknown"}
            if cap_gb is not None:
                try:
                    mps = read_mps_memory(cap_gb)
                except Exception:
                    pass
            if mps.get("used_gb") is not None and tracker is not None:
                try:
                    growth = tracker.update(mps["used_gb"])
                except Exception:
                    pass
            elif tracker is not None:
                # Do not calculate a slope across a gap with unknown memory.
                with self._state_lock:
                    if self.growth_tracker is tracker:
                        self.growth_tracker = GrowthRateTracker(cap_gb=cap_gb)

        try:
            thermal = read_thermal_pressure()
        except Exception:
            thermal = {"scheduler_limit": None, "status": "unknown"}

        return {
            "mps_level": mps,
            "growth_rate": growth,
            "thermal": thermal,
        }

    def _compute_means(self, samples: list[dict]) -> dict:
        """Compute mean status across 3 samples.

        For ramp-up: ANY warning/critical in any sample → mean is that level.
        """
        result = {}
        for channel in ("mps_level", "growth_rate", "thermal"):
            readings = [sample.get(channel, {}) for sample in samples]
            statuses = [reading.get("status", "unknown") for reading in readings]
            if "critical" in statuses:
                result[channel] = {"status": "critical"}
            elif "warning" in statuses:
                result[channel] = {"status": "warning"}
            elif all(status == "ok" for status in statuses):
                result[channel] = {"status": "ok"}
            elif all(reading.get("required") is False and reading.get("status") == "unsupported"
                     for reading in readings):
                result[channel] = {"status": "unsupported", "required": False}
            else:
                result[channel] = {"status": "unknown"}
        return result

    def _build_metrics(self) -> dict:
        """Build metrics dict for status reporting."""
        readings = self._last_readings
        age = None if self._last_sample_time is None else time.monotonic() - self._last_sample_time
        budget = self._budget_snapshot()
        return {
            "batch_size": self.state_machine.batch_size,
            "state": self.state_machine.state,
            "mps_used_gb": readings.get("mps_level", {}).get("used_gb"),
            "mps_ratio": readings.get("mps_level", {}).get("ratio"),
            "growth_slope_pct": readings.get("growth_rate", {}).get("slope_pct"),
            "thermal_limit": readings.get("thermal", {}).get("scheduler_limit"),
            "good_streak": self.state_machine.good_streak,
            "sensor_status": {channel: readings.get(channel, {}).get("status", "unknown")
                              for channel in ("mps_level", "growth_rate", "thermal")},
            "sample_age_seconds": age,
            "sample_stale": age is None or age > self.SAMPLE_MAX_AGE,
            "sample_pending": self._sample_lock.locked(),
            "mps_budget": {
                "scope": "process_mps_allocator",
                "intended_bytes": budget.intended_bytes if budget else None,
                "recommended_bytes": budget.recommended_bytes if budget else None,
                "effective_bytes": budget.effective_bytes if budget else None,
                "source": budget.source if budget else None,
                "enforcement": budget.enforcement if budget else (
                    "unknown" if self.device == "mps" else "not_applicable"
                ),
            },
        }

    @staticmethod
    def flush_gpu_cache():
        """Flush MPS/CUDA cache and run garbage collection."""
        try:
            from truememory.mps_utils import get_mps_memory_budget
            import torch

            if get_mps_memory_budget() is not None and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
