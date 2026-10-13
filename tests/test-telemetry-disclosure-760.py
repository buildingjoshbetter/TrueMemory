"""Verify the disclosed telemetry payload using synthetic identity and transport."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import AbstractContextManager
from pathlib import Path
import sys
import tempfile
import types
import unittest
from typing import TypeVar
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "truememory" / "telemetry.py"
SYNTHETIC_EMAIL = "synthetic@example.invalid"
SYNTHETIC_UUID = "00000000-0000-4000-8000-000000000760"
T = TypeVar("T")


def load_telemetry() -> types.ModuleType:
    # Loading this one file avoids package startup, models and live MCP config.
    module = types.ModuleType("synthetic_telemetry_760")
    exec(compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec"), module.__dict__)
    return module


class TelemetryDisclosure(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="synthetic-telemetry-760-")
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name)
        self.module = load_telemetry()
        self.module.Path = Mock(wraps=Path)
        self.module.Path.home.return_value = self.home
        self.module.get_device_id = Mock(return_value="synthetic-device")
        self.module._save_user_id = Mock()
        self.module._get_version = Mock(return_value="0.0.760")
        self.module.uuid = types.SimpleNamespace(uuid4=Mock(return_value=SYNTHETIC_UUID))
        self.module.sys = types.SimpleNamespace(platform="synthetic-os")
        self.module.platform = types.SimpleNamespace(
            machine=lambda: "synthetic-arch", python_version=lambda: "3.10.0",
        )
        self.module.time = types.SimpleNamespace(
            time=lambda: 1000.0, monotonic=lambda: 100.0,
        )
        # Record requested threads, but execute neither transport nor flush loop.
        self.thread_factory = Mock()
        self.module.threading = types.SimpleNamespace(Thread=self.thread_factory)
        response = types.SimpleNamespace(json=lambda: {})
        self.post = Mock(return_value=response)
        transport = types.ModuleType("httpx")
        transport.post = self.post
        self.enter(patch.dict(sys.modules, {"httpx": transport}))
        self.enter(patch.dict(os.environ, {"TRUEMEMORY_TELEMETRY": ""}))

    def enter(self, context: AbstractContextManager[T]) -> T:
        result = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return result

    def config(self, **extra: object) -> dict:
        return {"user_id": SYNTHETIC_UUID, "tier": "base", **extra}

    def assert_disabled(self) -> None:
        config = self.config(email=SYNTHETIC_EMAIL)
        self.assertIsNone(self.module.init(config))
        self.module.track("synthetic-event", {"synthetic": True})
        self.module.identify(SYNTHETIC_EMAIL, {"tier": "base"})

        @self.module.tracked("tool_synthetic")
        def synthetic_tool(value: str) -> str:
            return value

        self.assertEqual(synthetic_tool("synthetic-result"), "synthetic-result")
        self.assertIsNone(self.module._flush_sync())
        self.assertFalse(self.module.is_enabled())
        self.assertEqual(self.module._session_events, [])
        self.post.assert_not_called()
        self.thread_factory.assert_not_called()
        self.module.get_device_id.assert_not_called()
        self.module._save_user_id.assert_not_called()
        self.module.uuid.uuid4.assert_not_called()

    def test_session_start_exact_disclosed_fields_and_transport(self) -> None:
        self.module.init(self.config(
            email=SYNTHETIC_EMAIL,
            api_key="fake-secret-marker",
            memory="synthetic-memory-marker",
            path="synthetic-path-marker",
        ))
        expected = {
            "event": "session_start", "user_id": SYNTHETIC_UUID,
            "timestamp": 1000.0,
            "properties": {
                "tier": "base", "version": "0.0.760", "platform": "synthetic-os",
                "arch": "synthetic-arch", "python": "3.10.0",
                "device_id": "synthetic-device", "email": SYNTHETIC_EMAIL,
            },
        }
        self.assertEqual(self.module._session_events, [expected])
        self.assertEqual(self.thread_factory.call_count, 2)
        self.module._flush_sync()
        self.post.assert_called_once_with(
            "https://telemetry-api-production-c2a3.up.railway.app/v1/events",
            json={"events": [expected]}, timeout=3,
        )
        self.assertEqual(self.module._session_events, [])

    def test_email_is_included_on_each_configured_session_start(self) -> None:
        config = self.config(email=SYNTHETIC_EMAIL)
        self.module.init(config)
        self.module.init(config)
        self.assertEqual(len(self.module._session_events), 2)
        for event in self.module._session_events:
            self.assertEqual(event["properties"]["email"], SYNTHETIC_EMAIL)
            self.assertEqual(event["user_id"], SYNTHETIC_UUID)
        self.module.uuid.uuid4.assert_not_called()

    def test_unconfigured_email_is_absent(self) -> None:
        self.module.init(self.config())
        props = self.module._session_events[0]["properties"]
        self.assertEqual(set(props), {"tier", "version", "platform", "arch", "python", "device_id"})

    def test_generated_user_uuid_is_persisted_and_reused(self) -> None:
        config = {"tier": "edge"}
        self.module.init(config)
        self.assertEqual(config["user_id"], SYNTHETIC_UUID)
        self.module._save_user_id.assert_called_once_with(config)
        self.module.init(config)
        self.module.uuid.uuid4.assert_called_once()
        self.module._save_user_id.assert_called_once()
        self.assertEqual({event["user_id"] for event in self.module._session_events}, {SYNTHETIC_UUID})

    def test_environment_disable_prevents_enqueue_identity_and_transport(self) -> None:
        for value in ("off", "false", "0", "no", "OFF", "False", "NO"):
            with self.subTest(value=value), patch.dict(os.environ, {"TRUEMEMORY_TELEMETRY": value}):
                self.module._enabled = None
                self.assert_disabled()
                self.module.Path.home.assert_not_called()

    def test_config_disable_prevents_enqueue_identity_and_transport(self) -> None:
        config_dir = self.home / ".truememory"
        config_dir.mkdir()
        (config_dir / "config.json").write_text('{"telemetry": false}', encoding="utf-8")
        self.assert_disabled()

    def test_tool_events_exclude_inputs_outputs_and_exception_text(self) -> None:
        self.module.init(self.config())
        self.module._session_events.clear()

        @self.module.tracked("tool_search")
        def search(query: str, **options: str) -> dict:
            return {"content": "synthetic-result-marker", "query": query, **options}

        @self.module.tracked("tool_store")
        def store(content: str) -> None:
            raise ValueError("synthetic-exception-marker " + content)

        result = search("synthetic-query-marker", path="synthetic-path-marker", api_key="fake-key-marker")
        self.assertEqual(result["content"], "synthetic-result-marker")
        with self.assertRaisesRegex(ValueError, "synthetic-exception-marker"):
            store("synthetic-memory-marker")
        self.module._flush_sync()
        events = self.post.call_args.kwargs["json"]["events"]
        self.assertEqual(events, [
            {"event": "tool_search", "user_id": SYNTHETIC_UUID, "timestamp": 1000.0,
             "properties": {"latency_ms": 0.0, "success": True}},
            {"event": "tool_store", "user_id": SYNTHETIC_UUID, "timestamp": 1000.0,
             "properties": {"latency_ms": 0.0, "success": False}},
        ])
        self.assertNotIn("-marker", json.dumps(events))

    def test_identity_event_has_email_tier_and_common_envelope(self) -> None:
        self.module.init(self.config())
        self.module._session_events.clear()
        self.module.identify(SYNTHETIC_EMAIL, {"tier": "base"})
        self.assertEqual(self.module._session_events, [{
            "event": "identify", "user_id": SYNTHETIC_UUID, "timestamp": 1000.0,
            "properties": {"email": SYNTHETIC_EMAIL, "tier": "base"},
        }])

    def test_device_hash_is_stable_across_fresh_module_instances(self) -> None:
        raw = "synthetic-machine-identifier-760"
        expected = hashlib.sha256(raw.encode()).hexdigest()[:16]
        for _ in range(2):
            module = load_telemetry()
            module.sys = types.SimpleNamespace(platform="linux")
            machine_file = Mock()
            machine_file.exists.return_value = True
            machine_file.read_text.return_value = raw
            module.Path = Mock(return_value=machine_file)
            self.assertEqual(module.get_device_id(), expected)
            self.assertEqual(module.get_device_id(), expected)
            module.Path.assert_called_once_with("/etc/machine-id")
            machine_file.read_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
