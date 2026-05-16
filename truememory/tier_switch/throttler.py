"""Simplified throttler for tier-switch re-embedding.

Checks available RAM before each batch and pauses only when memory is
low. The worker handles OOM by halving batch_size and retrying.
"""

import gc
import logging
import time

import psutil

log = logging.getLogger(__name__)

_LOW_RAM_GB = 2.0
_LOW_RAM_PAUSE = 5.0
_INTER_BATCH_PAUSE = 0.2


class DynamicThrottler:
    """RAM-aware throttler with minimal overhead."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        total_gb = psutil.virtual_memory().total / (1024**3)

        if device in ("mps", "cuda"):
            if total_gb >= 32:
                self.batch_size = 32
            elif total_gb >= 16:
                self.batch_size = 16
            else:
                self.batch_size = 8
        else:
            if total_gb >= 16:
                self.batch_size = 64
            else:
                self.batch_size = 32

        self.items_processed = 0
        self.start_time = time.time()
        self.batch_times: list[float] = []
        self.last_throttle_time = 0.0

        log.info(
            "Throttler init: device=%s batch_size=%d total_ram=%.0fGB",
            device, self.batch_size, total_gb,
        )

    def before_batch(self) -> tuple[int, dict]:
        """Check RAM and return (batch_size, metrics). Pauses if low."""
        vm = psutil.virtual_memory()
        avail_gb = vm.available / (1024**3)

        if avail_gb < _LOW_RAM_GB:
            log.warning(
                "Low RAM (%.1fGB available), pausing %.0fs",
                avail_gb, _LOW_RAM_PAUSE,
            )
            time.sleep(_LOW_RAM_PAUSE)
            self.flush_gpu_cache()
        else:
            time.sleep(_INTER_BATCH_PAUSE)

        metrics = {
            "ram_avail_gb": avail_gb,
            "ram_pct": vm.percent,
        }
        return self.batch_size, metrics

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
