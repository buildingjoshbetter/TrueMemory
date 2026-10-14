"""Inert daemon policy controls using actual source and synthetic sensors only."""

from __future__ import annotations

import ast
import hashlib
import inspect
import logging
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "truememory/model_server.py"
GIB = 1 << 30


def load_source(environment: dict[str, str], sensor: object) -> types.ModuleType:
    names = {"_admission_setting", "_automatic_rss_budget_bytes", "_physical_memory_bytes", "_ProcessMemoryRefused",
             "_RequestDeadlineExceeded", "ModelServer", "main"}
    selected = [node for node in ast.parse(SOURCE.read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    tree = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__",
                    names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[]))
    module = types.ModuleType("synthetic_daemon_admission_297")
    module.created_workers = []

    class Workers:
        def __init__(self, **kwargs: object) -> None:
            module.created_workers.append(kwargs)

    module.__dict__.update({"os": types.SimpleNamespace(environ=environment), "psutil": sensor,
                           "sys": types.SimpleNamespace(platform="linux"),
                           "threading": threading, "ThreadPoolExecutor": Workers,
                           "time": types.SimpleNamespace(time=lambda: 0.0),
                           "_MAX_MESSAGE_SIZE": 10 * 1024 * 1024, "logging": logging,
                           "log": logging.getLogger(module.__name__)})
    exec(compile(tree, str(SOURCE), "exec"), module.__dict__)
    return module


class DaemonAdmissionControls(unittest.TestCase):
    def setUp(self) -> None:
        self.environment: dict[str, str] = {}
        self.physical = Mock(return_value=types.SimpleNamespace(total=16 * GIB))
        self.current = Mock(return_value=types.SimpleNamespace(rss=1 * GIB))
        self.sensor = types.SimpleNamespace(virtual_memory=self.physical,
                      Process=lambda: types.SimpleNamespace(memory_info=self.current))
        self.module = load_source(self.environment, self.sensor)

    def automatic(self, value: object = True) -> object:
        self.assertIn("automatic_rss_budget", inspect.signature(self.module.ModelServer).parameters)
        return self.module.ModelServer(automatic_rss_budget=value)

    def test_direct_constructor_legacy_default_has_no_sensor_reads(self) -> None:
        server = self.module.ModelServer()
        self.assertEqual(server._max_rss_bytes, 0)
        server._check_process_memory("synthetic legacy")
        self.physical.assert_not_called()
        self.current.assert_not_called()

    def test_original_default_disabled_test_preserved_byte_exact(self) -> None:
        path = ROOT / "tests/test-model-process-admission-297.py"
        source = path.read_text()
        node = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)
                    and node.name == "test_default_budget_is_disabled")
        segment = ast.get_source_segment(source, node)
        self.assertEqual(hashlib.sha256(segment.encode()).hexdigest(),
                         "1f18c4eb6d3fc61a08b34cf75a79fdb5ec663570129dd587adc63acedee83082")

    def test_automatic_physical_arithmetic_and_eight_GiB_ceiling(self) -> None:
        for physical_gib, expected_gib in [(4, 2), (8, 3), (12, 3), (16, 4), (32, 8), (64, 8)]:
            with self.subTest(physical_gib=physical_gib):
                self.physical.return_value = types.SimpleNamespace(total=physical_gib * GIB)
                server = self.automatic()
                self.assertEqual(server._max_rss_bytes, expected_gib * GIB)
        self.assertEqual(self.physical.call_count, 6)
        self.current.assert_not_called()

    def test_macos_small_hosts_keep_room_for_float32_cpu_models(self) -> None:
        self.module.sys = types.SimpleNamespace(platform="darwin")
        for physical_gib, expected_gib in [(4, 2), (8, 4), (12, 4), (16, 4), (32, 8)]:
            with self.subTest(physical_gib=physical_gib):
                self.physical.return_value = types.SimpleNamespace(total=physical_gib * GIB)
                self.assertEqual(self.automatic()._max_rss_bytes, expected_gib * GIB)

    def test_physical_bytes_floor_division_and_signed64_boundary(self) -> None:
        for physical, expected in [(4 * GIB + 3, 2 * GIB + 1), (2**63 - 1, 8 * GIB)]:
            with self.subTest(physical=physical):
                self.physical.return_value = types.SimpleNamespace(total=physical)
                self.assertEqual(self.automatic()._max_rss_bytes, expected)

    def test_invalid_physical_values_fail_before_workers_or_models(self) -> None:
        for value in [None, True, False, 0, -1, 1, 2, 3, "17179869184", 1.0,
                      float("nan"), float("inf"), 2**63, 2**100]:
            with self.subTest(value=value):
                self.physical.return_value = types.SimpleNamespace(total=value)
                with self.assertRaises(ValueError):
                    self.automatic()
                self.assertEqual(self.module.created_workers, [])
        self.current.assert_not_called()

    def test_unavailable_or_malformed_sensor_fails_before_workers(self) -> None:
        for sensor in [None, types.SimpleNamespace(),
                       types.SimpleNamespace(virtual_memory=lambda: None)]:
            with self.subTest(sensor=sensor):
                self.module.psutil = sensor
                with self.assertRaises(ValueError):
                    self.automatic()
                self.assertEqual(self.module.created_workers, [])

    def test_physical_sensor_failure_is_controlled_without_workers(self) -> None:
        for error in [OSError("synthetic unavailable"), RuntimeError("synthetic sensor failure"),
                      AttributeError("synthetic missing total"), TypeError("synthetic invalid measurement")]:
            with self.subTest(error=type(error).__name__):
                self.physical.side_effect = error
                with self.assertRaises(ValueError):
                    self.automatic()
                self.assertEqual(self.module.created_workers, [])

    def test_explicit_zero_wins_and_reads_neither_sensor(self) -> None:
        self.environment["TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB"] = "0"
        self.module.psutil = None
        server = self.automatic()
        self.assertEqual(server._max_rss_bytes, 0)
        server._check_process_memory("synthetic disabled")
        self.physical.assert_not_called()
        self.current.assert_not_called()

    def test_explicit_finite_budget_wins_without_physical_read(self) -> None:
        for value in [1, 512, 2**31 - 1]:
            with self.subTest(value=value):
                self.environment["TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB"] = str(value)
                self.assertEqual(self.automatic()._max_rss_bytes, value * 1048576)
        self.physical.assert_not_called()
        self.current.assert_not_called()

    def test_malformed_explicit_values_keep_original_rejection(self) -> None:
        for value in ["-1", "1.0", "nan", "inf", "true", "", " 1", "١", str(2**31)]:
            with self.subTest(value=value):
                self.environment["TRUEMEMORY_MODEL_SERVER_MAX_RSS_MB"] = value
                with self.assertRaises(ValueError):
                    self.automatic()
                self.assertEqual(self.module.created_workers, [])
        self.physical.assert_not_called()

    def test_boolean_keyword_only_policy_shape(self) -> None:
        for value in [None, 0, 1, "true", [], {}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.automatic(value)
        with self.assertRaises(TypeError):
            self.module.ModelServer(True)
        self.assertEqual(self.module.created_workers, [])

    def test_automatic_budget_keeps_exact_sampled_RSS_boundary(self) -> None:
        server = self.automatic()
        for usage in [server._max_rss_bytes - 1, server._max_rss_bytes, server._max_rss_bytes + 1]:
            with self.subTest(usage=usage), server._inference_lock:
                self.current.return_value = types.SimpleNamespace(rss=usage)
                if usage < server._max_rss_bytes:
                    server._check_process_memory("synthetic main")
                else:
                    with self.assertRaises(self.module._ProcessMemoryRefused):
                        server._check_process_memory("synthetic main")
        self.assertEqual(self.current.call_count, 3)

    def test_missing_current_RSS_refuses_finite_work(self) -> None:
        server = self.automatic()
        self.module.psutil = None
        with server._fast_lock, self.assertRaises(self.module._ProcessMemoryRefused):
            server._check_process_memory("synthetic fast")

    def test_sampled_admission_deadline_priority_is_unchanged(self) -> None:
        server = self.automatic()
        deadline = Mock()
        deadline.check.side_effect = self.module._RequestDeadlineExceeded("synthetic expiry")
        with self.assertRaises(self.module._RequestDeadlineExceeded):
            server._check_process_memory("synthetic expired", deadline)
        self.current.assert_not_called()
        deadline.check.side_effect = [None, self.module._RequestDeadlineExceeded("synthetic sensor consumed deadline")]
        self.current.return_value = types.SimpleNamespace(rss=server._max_rss_bytes)
        with self.assertRaises(self.module._RequestDeadlineExceeded):
            server._check_process_memory("synthetic expiry after sensor", deadline)
        self.assertEqual(self.current.call_count, 1)

    def test_installed_main_requests_automatic_policy_without_real_start(self) -> None:
        policies: list[bool] = []
        ran: list[bool] = []
        class Server:
            def __init__(self, *, automatic_rss_budget: bool = False) -> None:
                policies.append(automatic_rss_budget)
            def run(self) -> None:
                ran.append(True)
            def _cleanup(self) -> None:
                pass
        self.module.ModelServer = Server
        self.module.signal = types.SimpleNamespace(SIGTERM=15, SIGINT=2, SIGHUP=1, signal=Mock())
        self.module._handle_signal = Mock()
        self.module.atexit = types.SimpleNamespace(register=Mock())
        with patch.dict("sys.modules", {"setproctitle": None}), patch.object(logging, "basicConfig"):
            self.module.main()
        self.assertEqual(policies, [True])
        self.assertEqual(ran, [True])

    def test_original_module_and_console_launch_contract_is_preserved(self) -> None:
        client = (ROOT / "truememory/model_client.py").read_text()
        self.assertIn('[executable, "-m", "truememory.model_server"]', client)
        self.assertIn('truememory-model-server = "truememory.model_server:main"', (ROOT / "pyproject.toml").read_text())
        self.assertEqual(hashlib.sha256((ROOT / "truememory/mps_utils.py").read_bytes()).hexdigest(),
                         "58b68d386350a0a17ca1579ec2e4847871dcbdbd1cc7d497bd3b16717a963a20")


if __name__ == "__main__":
    unittest.main()
