"""Bounded admission tests with synthetic sensors, clocks, and no model imports."""
from __future__ import annotations

import ast
import copy
import subprocess
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "truememory" / "tier_switch"


def load_source(name: str, dependencies: dict | None = None) -> types.ModuleType:
    """Execute production code, replacing only package imports and host RAM reads."""
    source = ast.parse((SOURCE / f"{name}.py").read_text(encoding="utf-8"))
    source.body = [node for node in source.body if not (
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("truememory.")
        or isinstance(node, ast.Import) and any(alias.name == "psutil" for alias in node.names)
    )]
    module = types.ModuleType("synthetic_" + name)
    module.__dict__.update(dependencies or {})
    exec(compile(ast.fix_missing_locations(source), name + ".py", "exec"), module.__dict__)
    return module


class FakeClock:
    def __init__(self) -> None:
        self.now = -1.0
        self.wall = 5000.0
        self.wall_reads = 0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        self.wall_reads += 1
        return self.wall

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        self.wall += seconds


def readings(status: str = "ok") -> dict:
    return {
        "mps_level": {"used_gb": 1.0, "ratio": 0.1, "status": status},
        "growth_rate": {"slope_gb_per_20s": 0.0, "slope_pct": 0.0, "status": status},
        "thermal": {"scheduler_limit": 100, "status": status},
    }


class TestThrottleSampling(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.sensors = load_source("sensors")
        self.state = load_source("state_machine")
        self.throttler_module = load_source("throttler", {
            "psutil": types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(total=32 * 1024**3)),
            "GrowthRateTracker": self.sensors.GrowthRateTracker,
            "read_mps_memory": self.sensors.read_mps_memory,
            "read_thermal_pressure": self.sensors.read_thermal_pressure,
            "ThrottlerStateMachine": self.state.ThrottlerStateMachine,
        })
        for module in (self.sensors, self.state, self.throttler_module):
            module.time = self.clock
        self.throttler = self.throttler_module.DynamicThrottler("mps")
        self.observe = Mock(side_effect=lambda: readings())
        self.throttler._read_all_channels = self.observe

    def sample_at(self, seconds: float, snapshot: dict | None = None) -> tuple[int, dict]:
        self.clock.now = seconds
        self.observe.side_effect = lambda: copy.deepcopy(snapshot if snapshot is not None else readings())
        return self.throttler.before_batch()

    def warmup_before_first_sample(self) -> None:
        for _ in range(4):
            self.throttler.before_batch()
        self.observe.assert_not_called()

    def test_original_fifteen_admissions_have_only_existing_short_pacing(self) -> None:
        for _ in range(15):
            size, _metrics = self.throttler.before_batch()
            self.assertEqual(size, 1)
        self.assertEqual(self.observe.call_count, 3)
        self.assertEqual(len(self.clock.sleeps), 15)
        self.assertAlmostEqual(sum(self.clock.sleeps), 15 * (0.05 + 1 * 0.02))
        self.assertLess(max(self.clock.sleeps), 1.0)
        self.assertEqual(len(self.throttler._samples), 1)
        self.assertEqual(self.clock.wall_reads, 0)

    def test_ramp_requires_three_distinct_fresh_observations_spanning_twenty_seconds(self) -> None:
        self.warmup_before_first_sample()
        self.assertEqual(self.sample_at(0)[0], 1)
        self.assertEqual(self.sample_at(9)[0], 1)
        self.assertEqual(self.sample_at(10)[0], 1)
        self.assertEqual(self.sample_at(20)[0], 3)
        self.assertEqual(self.throttler.state_machine.good_streak, 0)
        self.assertEqual(len(self.throttler._samples), 0)
        self.assertEqual(self.throttler.state_machine.last_ramp_time, 20)

    def test_pending_ramp_collects_one_due_observation_without_waiting_for_more_batches(self) -> None:
        for _ in range(15):
            self.throttler.before_batch()
        initial_samples = self.observe.call_count
        self.clock.now += 10
        self.throttler.before_batch()
        self.assertEqual(self.observe.call_count, initial_samples + 1)
        self.assertEqual(self.throttler.batch_size, 1)
        self.clock.now += 10
        self.throttler.before_batch()
        self.assertEqual(self.throttler.batch_size, 3)

    def test_idle_gap_discards_old_good_streak_and_cannot_catch_up_to_a_ramp(self) -> None:
        self.warmup_before_first_sample()
        self.sample_at(0)
        self.sample_at(10)
        self.assertEqual(self.sample_at(100)[0], 1)
        self.assertEqual(self.observe.call_count, 3)
        self.assertEqual(len(self.throttler._samples), 1)
        self.assertEqual(self.throttler.state_machine.good_streak, 1)
        self.assertEqual(self.sample_at(110)[0], 1)
        self.assertEqual(self.sample_at(120)[0], 3)

    def test_unknown_warning_and_critical_clear_ramp_evidence(self) -> None:
        for status, expected_size in (("unknown", 8), ("warning", 6), ("critical", 4)):
            with self.subTest(status=status):
                self.setUp()
                self.throttler.batch_size = 8
                self.clock.now = -2
                self.warmup_before_first_sample()
                self.sample_at(0)
                self.sample_at(10)
                snapshot = readings()
                snapshot["thermal"] = {"status": status}
                size, metrics = self.sample_at(20, snapshot)
                self.assertEqual(size, expected_size)
                self.assertEqual(metrics["sensor_status"]["thermal"], status)
                self.assertEqual(len(self.throttler._samples), 0)
                self.assertEqual(self.throttler.state_machine.good_streak, 0)

    def test_ramp_and_backoff_cooldowns_keep_existing_120_second_threshold(self) -> None:
        state = self.throttler.state_machine
        self.warmup_before_first_sample()
        self.sample_at(0)
        self.sample_at(10)
        self.sample_at(20)
        for seconds in range(30, 131, 10):
            self.sample_at(seconds)
        self.clock.now = 139.99
        self.clock.wall += 10000
        self.assertFalse(state.should_ramp_check())
        self.clock.now = 140
        self.clock.wall -= 20000
        self.assertTrue(state.should_ramp_check())
        self.throttler.on_oom()
        for seconds in range(150, 251, 10):
            self.sample_at(seconds)
        self.clock.now = 259.99
        self.assertFalse(state.should_ramp_check())
        self.clock.now = 260
        self.assertTrue(state.should_ramp_check())
        self.assertEqual(self.clock.wall_reads, 0)

    def test_ordinary_ten_second_admissions_ramp_with_fresh_bounded_history(self) -> None:
        increases: list[tuple[float, int]] = []
        previous = 1
        for index in range(120):
            size, metrics = self.sample_at(index * 10)
            if size > previous:
                increases.append((index * 10, size))
            previous = size
            self.assertLessEqual(len(self.throttler._samples), 3)
            self.assertLessEqual(self.throttler.batch_size, 16)
            if index >= 4:
                self.assertFalse(metrics["sample_stale"])
                self.assertLess(metrics["sample_age_seconds"], 1)
        self.assertEqual(increases, [(60, 3), (180, 5), (300, 7), (420, 9),
                                     (540, 11), (660, 13), (780, 15), (900, 16)])
        self.assertEqual(self.observe.call_count, 116)
        self.assertLessEqual(max(self.clock.sleeps), 0.05 + 16 * 0.02)

    def test_unsupported_channels_are_visible_and_no_telemetry_cannot_enable_ramp(self) -> None:
        self.warmup_before_first_sample()
        unsupported = {channel: {"status": "unsupported", "required": False} for channel in readings()}
        for seconds in (0, 10, 20, 140):
            size, metrics = self.sample_at(seconds, unsupported)
            self.assertEqual(size, 1)
            self.assertEqual(metrics["good_streak"], 0)
        self.assertEqual(set(metrics["sensor_status"].values()), {"unsupported"})
        self.assertIsNone(metrics["thermal_limit"])

    def test_cpu_can_use_real_thermal_evidence_without_probing_mps(self) -> None:
        throttler = self.throttler_module.DynamicThrottler("cpu")
        self.throttler_module.read_mps_memory = Mock(side_effect=AssertionError("MPS must not be probed"))
        self.throttler_module.read_thermal_pressure = Mock(return_value={"status": "ok", "scheduler_limit": 100})
        for seconds in range(0, 61, 10):
            self.clock.now = seconds
            size, metrics = throttler.before_batch()
        self.assertEqual(size, 3)
        self.throttler_module.read_mps_memory.assert_not_called()
        self.assertEqual(metrics["sensor_status"], {
            "mps_level": "unsupported", "growth_rate": "unsupported", "thermal": "ok",
        })

    def test_sensor_errors_are_unknown_and_do_not_reuse_previous_healthy_value(self) -> None:
        throttler = self.throttler_module.DynamicThrottler("mps")
        self.throttler_module.read_mps_memory = Mock(side_effect=RuntimeError("synthetic sensor failure"))
        self.throttler_module.read_thermal_pressure = Mock(side_effect=OSError("synthetic unavailable sensor"))
        for _ in range(5):
            size, metrics = throttler.before_batch()
        self.assertEqual(size, 1)
        self.assertEqual(set(metrics["sensor_status"].values()), {"unknown"})
        self.assertIsNone(metrics["mps_used_gb"])
        self.assertIsNone(metrics["thermal_limit"])

    def test_single_flight_sampler_does_not_hold_state_lock_or_delay_other_admissions(self) -> None:
        started, release = threading.Event(), threading.Event()
        errors: list[Exception] = []

        def blocked_sensor() -> dict:
            started.set()
            if not release.wait(timeout=2):
                raise RuntimeError("synthetic sampler was not released")
            return readings()

        def admit() -> None:
            try:
                self.throttler.before_batch()
            except Exception as error:
                errors.append(error)

        self.warmup_before_first_sample()
        self.observe.side_effect = blocked_sensor
        observer = threading.Thread(target=admit)
        observer.start()
        try:
            self.assertTrue(started.wait(timeout=2))
            size, metrics = self.throttler.before_batch()
            self.assertEqual(size, 1)
            self.assertTrue(metrics["sample_pending"])
            self.assertTrue(metrics["sample_stale"])
            self.throttler.after_batch(2, 0.1)
            self.assertEqual(self.throttler.items_processed, 2)
            self.assertEqual(self.observe.call_count, 1)
        finally:
            release.set()
            observer.join(timeout=3)
        self.assertFalse(observer.is_alive())
        self.assertEqual(errors, [])
        self.assertFalse(self.throttler._sample_lock.locked())

    def test_oom_during_sensor_probe_invalidates_prefault_ramp_evidence(self) -> None:
        self.throttler.batch_size = 8
        self.clock.now = -2
        self.warmup_before_first_sample()
        self.sample_at(0)
        self.sample_at(10)
        self.clock.now = 20

        def fault_during_probe() -> dict:
            self.throttler.on_oom()
            return readings()

        self.observe.side_effect = fault_during_probe
        size, _metrics = self.throttler.before_batch()
        self.assertEqual(size, 4)
        self.assertEqual(self.throttler.state_machine.state, self.state.ThrottlerStateMachine.BACKOFF)
        self.assertEqual(self.throttler.state_machine.good_streak, 0)
        self.assertEqual(len(self.throttler._samples), 0)
        self.assertEqual(self.throttler._last_sample_time, 10)

    def test_sampler_exception_releases_single_flight_gate(self) -> None:
        self.warmup_before_first_sample()
        self.observe.side_effect = ValueError("synthetic unexpected observation failure")
        with self.assertRaises(ValueError):
            self.throttler.before_batch()
        self.assertFalse(self.throttler._sample_lock.locked())
        self.observe.side_effect = lambda: readings()
        self.assertEqual(self.throttler.before_batch()[0], 1)

    def test_concurrent_size_changes_cannot_bypass_pacing_or_backoff(self) -> None:
        for initial, changed, expected in ((1, 8, 1), (8, 4, 4)):
            with self.subTest(initial=initial, changed=changed):
                self.throttler.batch_size = initial

                def change_during_pacing(seconds: float) -> None:
                    self.assertAlmostEqual(seconds, 0.05 + initial * 0.02)
                    self.throttler.batch_size = changed

                with patch.object(self.clock, "sleep", side_effect=change_during_pacing):
                    size, metrics = self.throttler.before_batch()
                self.assertEqual(size, expected)
                self.assertEqual(metrics["batch_size"], expected)

    def test_growth_uses_monotonic_time_and_needs_two_valid_observations(self) -> None:
        tracker = self.sensors.GrowthRateTracker(12.0)
        self.clock.now = 0
        self.assertEqual(tracker.update(5.0)["status"], "unknown")
        self.clock.now = 20
        self.clock.wall -= 10000
        self.assertEqual(tracker.update(5.7)["status"], "warning")
        self.assertEqual(tracker.update(5.8)["status"], "unknown")
        self.assertEqual(self.clock.wall_reads, 0)

    def test_bounded_thermal_probe_returns_unknown_on_timeout_or_invalid_output(self) -> None:
        for result in (
            subprocess.TimeoutExpired("pmset", 0.25),
            subprocess.CompletedProcess([], 1, "CPU_Scheduler_Limit = 100\n"),
            subprocess.CompletedProcess([], 0, ""),
            subprocess.CompletedProcess([], 0, "CPU_Scheduler_Limit = invalid\n"),
            subprocess.CompletedProcess([], 0, "CPU_Scheduler_Limit = 101\n"),
            subprocess.CompletedProcess([], 0, "CPU_Scheduler_Limit = 100\nCPU_Speed_Limit = invalid\n"),
            subprocess.CompletedProcess([], 0, "CPU_Scheduler_Limit = 100\nCPU_Speed_Limit\n"),
        ):
            with self.subTest(result=type(result).__name__), patch.object(self.sensors.sys, "platform", "darwin"):
                with patch.object(self.sensors.subprocess, "run") as run:
                    if isinstance(result, Exception):
                        run.side_effect = result
                    else:
                        run.return_value = result
                    observed = self.sensors.read_thermal_pressure()
                self.assertEqual(observed["status"], "unknown")
                self.assertIsNone(observed["scheduler_limit"])
                self.assertEqual(run.call_args.kwargs["timeout"], 0.25)

    def test_thermal_uses_most_restrictive_limit_independent_of_field_order(self) -> None:
        for speed_limit, status in ((100, "ok"), (85, "warning"), (60, "critical")):
            fields = ["CPU_Scheduler_Limit = 100", f"CPU_Speed_Limit = {speed_limit}"]
            for lines in (fields, list(reversed(fields))):
                with self.subTest(lines=lines), patch.object(self.sensors.sys, "platform", "darwin"):
                    result = subprocess.CompletedProcess([], 0, "\n".join(lines))
                    with patch.object(self.sensors.subprocess, "run", return_value=result):
                        observed = self.sensors.read_thermal_pressure()
                    self.assertEqual(observed, {"scheduler_limit": speed_limit, "status": status})

    def test_malformed_thermal_field_cannot_hide_valid_backoff_evidence(self) -> None:
        fields = ("CPU_Scheduler_Limit", "CPU_Speed_Limit")
        malformed_suffixes = (" = invalid", " = 101", " = -1", " =", " = = 60", "")
        for valid_limit, status, expected_batch in ((60, "critical", 4), (85, "warning", 6), (100, "unknown", 8)):
            for valid_field, invalid_field in (fields, tuple(reversed(fields))):
                for suffix in malformed_suffixes:
                    lines = [f"{valid_field} = {valid_limit}", invalid_field + suffix]
                    for ordered in (lines, list(reversed(lines))):
                        with self.subTest(lines=ordered), patch.object(self.sensors.sys, "platform", "darwin"):
                            result = subprocess.CompletedProcess([], 0, "\n".join(ordered))
                            with patch.object(self.sensors.subprocess, "run", return_value=result):
                                observed = self.sensors.read_thermal_pressure()
                            expected_limit = valid_limit if status != "unknown" else None
                            self.assertEqual(observed, {"scheduler_limit": expected_limit, "status": status})
                            state = self.state.ThrottlerStateMachine(8, 16, 2)
                            state.good_streak = 2
                            snapshot = readings()
                            snapshot["thermal"] = observed
                            self.assertEqual(state.safety_check(snapshot), expected_batch)
                            self.assertEqual(state.good_streak, 0)

    def test_unsupported_thermal_platforms_never_spawn_pmset(self) -> None:
        for platform in ("linux", "win32"):
            with self.subTest(platform=platform), patch.object(self.sensors.sys, "platform", platform):
                with patch.object(self.sensors.subprocess, "run") as run:
                    observed = self.sensors.read_thermal_pressure()
                    run.assert_not_called()
                self.assertEqual(observed, {"scheduler_limit": None, "status": "unsupported", "required": False})


if __name__ == "__main__":
    unittest.main()
