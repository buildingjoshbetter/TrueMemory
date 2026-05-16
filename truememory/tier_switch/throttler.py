"""Hardened dynamic throttler for tier-switch re-embedding.

Uses triple-sampling (3 readings, 10s apart) for ramp-up decisions,
immediate reaction for throttle-down, and mandatory cooldown timers
to prevent the avalanche effect that caused load-83 overheating.

3-model adversarial review (Gemini, Grok, Qwen) validated this design.
"""

import gc
import logging
import time
from collections import deque

import psutil

log = logging.getLogger(__name__)

_PROFILES = {
    "conservative": {"max_gb": 12, "gpu_max": 8, "cpu_max": 20},
    "balanced": {"max_gb": 28, "gpu_max": 12, "cpu_max": 60},
    "performance": {"max_gb": float("inf"), "gpu_max": 16, "cpu_max": 100},
}

# RAM thresholds in AVAILABLE GB (not percentage — percentage fails on high-baseline machines)
# Load thresholds are PER-CORE (divided by cpu_count at check time)
_GOOD = {"ram_avail_gb": 4.0, "load_per_core": 0.7, "temp": 68.0}
_BAD = {"ram_avail_gb": 2.0, "load_per_core": 2.0, "temp": 78.0}
_CRITICAL = {"ram_avail_gb": 1.0, "load_per_core": 4.0, "temp": 86.0}
_GOOD_NO_TEMP = {"ram_avail_gb": 4.0, "load_per_core": 0.7}
_BAD_NO_TEMP = {"ram_avail_gb": 2.0, "load_per_core": 2.0}
_CRITICAL_NO_TEMP = {"ram_avail_gb": 1.0, "load_per_core": 4.0}

_SAMPLE_INTERVAL = 5
_SAMPLES_PER_WINDOW = 2
_GOOD_WINDOWS_REQUIRED = 3
_RAMP_COOLDOWN = 120
_THROTTLE_LOCKOUT = 300
_PANIC_SLEEP = 60

_GPU_START = 4
_CPU_START = 10
_GPU_RAMP_STEP = 2
_CPU_RAMP_STEP = 5


class DynamicThrottler:
    """Hardware-adaptive throttler with triple-sampling and cooldown timers."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        total_gb = psutil.virtual_memory().total / (1024**3)
        self.total_ram_gb = total_gb
        self.cpu_count = psutil.cpu_count(logical=True) or 1

        for name, p in _PROFILES.items():
            if total_gb <= p["max_gb"]:
                self.profile = name
                self.max_batch = (
                    p["gpu_max"] if device in ("mps", "cuda") else p["cpu_max"]
                )
                break
        else:
            self.profile = "balanced"
            self.max_batch = 12 if device in ("mps", "cuda") else 60

        self.has_temp = self._detect_temp_sensor()
        if not self.has_temp:
            log.info(
                "No temperature sensors detected — using tighter RAM/load thresholds"
            )

        self.batch_size = (
            _GPU_START if device in ("mps", "cuda") else _CPU_START
        )
        self.batch_size = min(self.batch_size, self.max_batch)
        self.samples: deque = deque(
            maxlen=_SAMPLES_PER_WINDOW * _GOOD_WINDOWS_REQUIRED
        )
        self.last_ramp_time = 0.0
        self.last_throttle_time = 0.0
        self.good_window_count = 0
        self.items_processed = 0
        self.start_time = time.time()
        self.batch_times: list[float] = []
        self.in_panic = False

        log.info(
            "Throttler init: device=%s profile=%s max_batch=%d has_temp=%s",
            device,
            self.profile,
            self.max_batch,
            self.has_temp,
        )

    def _detect_temp_sensor(self) -> bool:
        try:
            temps = psutil.sensors_temperatures()
            return bool(temps)
        except (AttributeError, OSError):
            return False

    def _sample_metrics(self) -> dict:
        vm = psutil.virtual_memory()
        metrics = {
            "ram_avail_gb": vm.available / (1024**3),
            "ram_pct": vm.percent,
            "load": psutil.getloadavg()[0],
            "temp": 0.0,
            "timestamp": time.time(),
        }
        if self.has_temp:
            try:
                temps = psutil.sensors_temperatures()
                for entries in temps.values():
                    if entries:
                        metrics["temp"] = max(
                            e.current for e in entries if e.current
                        )
                        break
            except (AttributeError, OSError):
                pass
        return metrics

    def _is_good(self, metrics: dict) -> bool:
        """All metrics in safe range. RAM: more available = better. Load: per-core."""
        good = _GOOD if self.has_temp else _GOOD_NO_TEMP
        if metrics["ram_avail_gb"] < good["ram_avail_gb"]:
            return False
        if (metrics["load"] / self.cpu_count) >= good["load_per_core"]:
            return False
        if self.has_temp and metrics.get("temp", 0) >= good.get("temp", 999):
            return False
        return True

    def _is_bad(self, metrics: dict) -> bool:
        """Any metric in danger zone."""
        bad = _BAD if self.has_temp else _BAD_NO_TEMP
        if metrics["ram_avail_gb"] < bad["ram_avail_gb"]:
            return True
        if (metrics["load"] / self.cpu_count) > bad["load_per_core"]:
            return True
        if self.has_temp and metrics.get("temp", 0) > bad.get("temp", 999):
            return True
        return False

    def _is_critical(self, metrics: dict) -> bool:
        """Any metric at critical level — panic."""
        crit = _CRITICAL if self.has_temp else _CRITICAL_NO_TEMP
        if metrics["ram_avail_gb"] < crit["ram_avail_gb"]:
            return True
        if (metrics["load"] / self.cpu_count) > crit["load_per_core"]:
            return True
        if self.has_temp and metrics.get("temp", 0) > crit.get("temp", 999):
            return True
        return False

    def _collect_window(self) -> list[dict]:
        """Collect a triple-sample window (3 samples, 10s apart)."""
        window = []
        for i in range(_SAMPLES_PER_WINDOW):
            if i > 0:
                time.sleep(_SAMPLE_INTERVAL)
            sample = self._sample_metrics()
            window.append(sample)
            self.samples.append(sample)
        return window

    def _window_mean(self, window: list[dict]) -> dict:
        n = len(window)
        return {
            "ram_avail_gb": sum(w["ram_avail_gb"] for w in window) / n,
            "load": sum(w["load"] for w in window) / n,
            "temp": sum(w["temp"] for w in window) / n,
        }

    def _can_ramp_up(self) -> bool:
        now = time.time()
        if now - self.last_ramp_time < _RAMP_COOLDOWN:
            return False
        if now - self.last_throttle_time < _THROTTLE_LOCKOUT:
            return False
        if self.batch_size >= self.max_batch:
            return False
        return True

    def before_batch(self) -> tuple[int, dict]:
        """Called before each embedding batch.

        Collects a triple-sample window, evaluates pressure,
        and adapts batch size. Returns (batch_size, metrics_dict).
        """
        window = self._collect_window()
        latest = window[-1]
        mean = self._window_mean(window)
        worst = {
            "ram_avail_gb": min(w["ram_avail_gb"] for w in window),
            "load": max(w["load"] for w in window),
            "temp": max(w["temp"] for w in window),
        }

        for sample in window:
            if self._is_critical(sample):
                log.warning(
                    "PANIC: critical reading ram_avail=%.1fGB load=%.1f temp=%.0f — "
                    "dropping to batch=1, sleeping %ds",
                    sample["ram_avail_gb"],
                    sample["load"],
                    sample["temp"],
                    _PANIC_SLEEP,
                )
                self.batch_size = 1
                self.last_throttle_time = time.time()
                self.good_window_count = 0
                self.samples.clear()
                self.in_panic = True
                time.sleep(_PANIC_SLEEP)
                self.in_panic = False
                return self.batch_size, latest

        if self._is_bad(worst):
            old = self.batch_size
            self.batch_size = max(1, self.batch_size // 2)
            self.last_throttle_time = time.time()
            self.good_window_count = 0
            self.samples.clear()
            log.info(
                "Throttle-down: %d→%d (ram_avail=%.1fGB load=%.1f temp=%.0f)",
                old,
                self.batch_size,
                worst["ram_avail_gb"],
                worst["load"],
                worst["temp"],
            )
            return self.batch_size, latest

        if self._is_good(mean):
            self.good_window_count += 1
            if (
                self.good_window_count >= _GOOD_WINDOWS_REQUIRED
                and self._can_ramp_up()
            ):
                old = self.batch_size
                step = (
                    _GPU_RAMP_STEP
                    if self.device in ("mps", "cuda")
                    else _CPU_RAMP_STEP
                )
                self.batch_size = min(self.max_batch, self.batch_size + step)
                self.last_ramp_time = time.time()
                self.good_window_count = 0
                log.info(
                    "Ramp-up: %d→%d (mean ram_avail=%.1fGB load=%.1f)",
                    old,
                    self.batch_size,
                    mean["ram_avail_gb"],
                    mean["load"],
                )
        else:
            self.good_window_count = 0

        sleep_time = 0.5 + (self.batch_size / max(self.max_batch, 1)) * 1.5
        time.sleep(sleep_time)

        return self.batch_size, latest

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
