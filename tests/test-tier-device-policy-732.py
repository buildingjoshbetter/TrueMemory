"""Synthetic actual-source witnesses for tier-switch device policy."""
from __future__ import annotations

import ast
import contextlib
import io
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def selected_functions(relative: str, names: set[str], namespace: dict[str, object]) -> None:
    path = ROOT / relative
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise AssertionError("Actual source functions were not found")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


class TierDevicePolicyTests(unittest.TestCase):
    def observed(self, override: str | None, *, cuda: bool, mps: bool) -> str:
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
        shared = types.ModuleType("truememory.mps_utils")
        shared.__dict__.update({
            "os": os, "logger": logging.getLogger("tier-policy-test"),
            "_VALID_DEVICE_VALUES": {"cpu", "mps", "cuda", "auto"},
        })
        selected_functions("truememory/mps_utils.py", {"auto_detect_device", "resolve_device"}, shared.__dict__)
        namespace: dict[str, object] = {}
        selected_functions("truememory/tier_switch/manager.py", {"_detect_device"}, namespace)
        package = types.ModuleType("truememory")
        package.__path__ = []
        with patch.dict(sys.modules, {"torch": torch, "truememory": package, "truememory.mps_utils": shared}):
            with patch.dict(os.environ):
                if override is None:
                    os.environ.pop("TRUEMEMORY_DEVICE", None)
                else:
                    os.environ["TRUEMEMORY_DEVICE"] = override
                result = namespace["_detect_device"]()
        self.assertIsInstance(result, str)
        return result

    def test_cpu_override_beats_each_accelerator(self) -> None:
        for cuda, mps in [(True, True), (True, False), (False, True), (False, False)]:
            with self.subTest(cuda=cuda, mps=mps):
                self.assertEqual(self.observed("cpu", cuda=cuda, mps=mps), "cpu")

    def test_auto_uses_shared_cuda_priority_and_identity(self) -> None:
        for override in [None, "", "auto", " AUTO "]:
            with self.subTest(override=override):
                self.assertEqual(self.observed(override, cuda=True, mps=True), "cuda:0")

    def test_auto_mps_and_cpu_fallbacks(self) -> None:
        self.assertEqual(self.observed(None, cuda=False, mps=True), "mps")
        self.assertEqual(self.observed("auto", cuda=False, mps=False), "cpu")

    def test_explicit_available_devices(self) -> None:
        self.assertEqual(self.observed("mps", cuda=True, mps=True), "mps")
        self.assertEqual(self.observed("cuda", cuda=True, mps=True), "cuda:0")
        self.assertEqual(self.observed(" CPU ", cuda=True, mps=True), "cpu")

    def test_unavailable_override_uses_shared_fallback(self) -> None:
        with self.assertLogs("tier-policy-test", level="WARNING"):
            self.assertEqual(self.observed("mps", cuda=True, mps=False), "cuda:0")
        with self.assertLogs("tier-policy-test", level="WARNING"):
            self.assertEqual(self.observed("cuda", cuda=False, mps=True), "mps")
        with self.assertLogs("tier-policy-test", level="WARNING"):
            self.assertEqual(self.observed("cuda", cuda=False, mps=False), "cpu")

    def test_invalid_override_retains_shared_warning_and_fallback(self) -> None:
        with self.assertLogs("tier-policy-test", level="WARNING"):
            self.assertEqual(self.observed("not-a-device", cuda=False, mps=True), "mps")

    def test_edge_tier_cpu_selection_is_preserved(self) -> None:
        source = (ROOT / "truememory/tier_switch/manager.py").read_text(encoding="utf-8")
        self.assertIn('device = "cpu" if job.target.tier_group == "edge" else _detect_device()', source)

    def test_actual_upgrade_cli_preserves_offline_environment(self) -> None:
        namespace: dict[str, object] = {
            "sys": sys, "_load_truememory_config": lambda: {},
        }
        selected_functions("truememory/ingest/cli.py", {"_run_upgrade_tier"}, namespace)
        manager = types.ModuleType("truememory.tier_switch.manager")

        class SyntheticManager:
            def run_rebuild_sync(self, **kwargs: object) -> bool:
                return True

        manager.RebuildManager = SyntheticManager
        with patch.dict(sys.modules, {"truememory.tier_switch.manager": manager}):
            for hf, transformers in [("1", "1"), ("0", "0"), ("custom", "custom")]:
                with self.subTest(hf=hf, transformers=transformers), patch.dict(os.environ, {
                    "HF_HUB_OFFLINE": hf, "TRANSFORMERS_OFFLINE": transformers,
                }), contextlib.redirect_stdout(io.StringIO()):
                    namespace["_run_upgrade_tier"](types.SimpleNamespace(tier="base", force=False))
                    self.assertEqual(os.environ["HF_HUB_OFFLINE"], hf)
                    self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], transformers)


if __name__ == "__main__":
    unittest.main()
