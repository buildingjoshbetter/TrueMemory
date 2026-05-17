"""Throttler state machine — PROBING/STABLE/BACKOFF logic.

Takes sensor readings as input, outputs batch size decisions.
Implements asymmetric intervals: fast decrease, slow increase.

Does NOT call sensors directly — receives readings from the
DynamicThrottler class which plugs in real sensors.
"""

import logging
import time

log = logging.getLogger(__name__)


class ThrottlerStateMachine:
    """Three-state decision engine for adaptive batch sizing."""

    PROBING = "probing"
    STABLE = "stable"
    BACKOFF = "backoff"

    RAMP_COOLDOWN = 120
    BACKOFF_COOLDOWN = 120
    SAFETY_CHECK_INTERVAL = 5
    GOOD_WINDOWS_REQUIRED = 3

    def __init__(self, start_batch: int, max_batch: int, ramp_step: int):
        self.batch_size = start_batch
        self.max_batch = max_batch
        self.ramp_step = ramp_step
        self.state = self.PROBING
        self.good_streak = 0
        self.batch_count = 0
        self.last_ramp_time = 0.0
        self.last_backoff_time = 0.0

    def on_batch_complete(self):
        """Call after every batch. Increments batch counter."""
        self.batch_count += 1

    def should_safety_check(self) -> bool:
        """Returns True every SAFETY_CHECK_INTERVAL batches."""
        return self.batch_count >= self.SAFETY_CHECK_INTERVAL

    def safety_check(self, readings: dict) -> int:
        """Fast safety check — called every 5 batches.

        Args:
            readings: dict with keys "mps_level", "growth_rate", "thermal"
                      each containing a dict with "status" key

        Returns:
            Updated batch_size
        """
        self.batch_count = 0

        for channel in ("mps_level", "growth_rate", "thermal"):
            if readings.get(channel, {}).get("status") == "critical":
                return self._do_backoff(channel, readings[channel])

        for channel in ("mps_level", "growth_rate", "thermal"):
            if readings.get(channel, {}).get("status") == "warning":
                return self._do_step_down(channel, readings[channel])

        self.good_streak += 1
        return self.batch_size

    def should_ramp_check(self) -> bool:
        """Returns True if conditions are met to consider a ramp-up."""
        if self.state == self.BACKOFF:
            if time.time() - self.last_backoff_time >= self.BACKOFF_COOLDOWN:
                self.state = self.PROBING
                log.info("Backoff cooldown expired, re-entering PROBING")
            else:
                return False

        if self.state == self.STABLE:
            if self.good_streak >= self.GOOD_WINDOWS_REQUIRED:
                self.state = self.PROBING
                log.info("Stable with %d good checks, re-entering PROBING", self.good_streak)
            else:
                return False

        if self.state != self.PROBING:
            return False
        if self.batch_size >= self.max_batch:
            return False
        if self.good_streak < self.GOOD_WINDOWS_REQUIRED:
            return False
        if time.time() - self.last_ramp_time < self.RAMP_COOLDOWN:
            return False
        if time.time() - self.last_backoff_time < self.BACKOFF_COOLDOWN:
            return False

        return True

    def ramp_up(self, triple_sample_means: dict) -> int:
        """Slow ramp-up check — called every 120s with triple-sample means.

        Args:
            triple_sample_means: dict with same structure as safety_check
                                 readings, but values are MEANS of 3 samples

        Returns:
            Updated batch_size (may or may not have increased)
        """
        for channel in ("mps_level", "growth_rate", "thermal"):
            if triple_sample_means.get(channel, {}).get("status") != "ok":
                self.good_streak = 0
                return self.batch_size

        old = self.batch_size
        self.batch_size = min(self.max_batch, self.batch_size + self.ramp_step)
        self.last_ramp_time = time.time()
        self.good_streak = 0
        log.info("Ramp-up: %d → %d", old, self.batch_size)
        return self.batch_size

    def _do_step_down(self, channel: str, reading: dict) -> int:
        """WARNING response — step down by ramp_step."""
        old = self.batch_size
        self.batch_size = max(1, self.batch_size - self.ramp_step)
        self.state = self.STABLE
        self.good_streak = 0
        log.info(
            "Step-down: %d → %d (WARNING on %s)", old, self.batch_size, channel
        )
        return self.batch_size

    def _do_backoff(self, channel: str, reading: dict) -> int:
        """CRITICAL response — halve batch immediately."""
        old = self.batch_size
        self.batch_size = max(1, self.batch_size // 2)
        self.state = self.BACKOFF
        self.good_streak = 0
        self.last_backoff_time = time.time()
        log.warning(
            "BACKOFF: %d → %d (CRITICAL on %s)", old, self.batch_size, channel
        )
        return self.batch_size
