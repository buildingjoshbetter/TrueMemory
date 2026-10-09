"""Synthetic installer/setup outcome checks. No model or package downloads."""
from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import io
import itertools
import os
import runpy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
LABELS = ("Edge reranker", "Base/Pro embedder", "Base/Pro reranker")

FAKE_UV = """#!/bin/sh
case "$*" in
  --version) printf 'synthetic uv\\n' ;;
  'tool dir') printf '%s\\n' "$SYNTHETIC_TOOL_DIR" ;;
  'python install '*|'tool uninstall '*|'tool install '*|'tool update-shell') ;;
  *) exit 99 ;;
esac
"""

FAKE_TOOL_PYTHON = """#!/bin/sh
case "$*" in
  *ms-marco-MiniLM-L-6-v2*) printf 'synthetic-model-call edge-reranker\\n'; exit "$SYNTHETIC_EDGE_RERANKER" ;;
  *Qwen/Qwen3-Embedding-0.6B*) printf 'synthetic-model-call base-embedder\\n'; exit "$SYNTHETIC_BASE_EMBEDDER" ;;
  *Alibaba-NLP/gte-reranker-modernbert-base*) printf 'synthetic-model-call base-reranker\\n'; exit "$SYNTHETIC_BASE_RERANKER" ;;
  *importlib.metadata*) printf '0.0.0-synthetic\\n' ;;
  '-m truememory.mcp_server --setup') exit "$SYNTHETIC_SETUP_EXIT" ;;
  '-m truememory.ingest.cli install') exit "$SYNTHETIC_HOOK_EXIT" ;;
  *) exit 99 ;;
esac
"""

# Functions shadow external tools in this child PowerShell process only.
# The actual installer body still controls statuses, summaries, and branches.
POWERSHELL_WRAPPER = r"""
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
function Get-ExecutionPolicy { param($Scope) return 'Bypass' }
function uv {
    $global:LASTEXITCODE = 0
    if (($args -join ' ') -eq '--version') { 'synthetic uv' }
    elseif (($args -join ' ') -eq 'tool dir') { $env:SYNTHETIC_TOOL_DIR }
}
function Join-Path {
    param($Path, $ChildPath)
    if ($ChildPath -eq 'truememory\Scripts\python.exe') { return 'Invoke-SyntheticPython' }
    Microsoft.PowerShell.Management\Join-Path -Path $Path -ChildPath $ChildPath
}
function Test-Path {
    param($Path, $LiteralPath)
    $candidate = if ($LiteralPath) { $LiteralPath } else { $Path }
    if ($candidate -eq 'Invoke-SyntheticPython') { return $env:SYNTHETIC_MISSING_PYTHON -ne '1' }
    Microsoft.PowerShell.Management\Test-Path -LiteralPath $candidate
}
function Invoke-SyntheticPython {
    $command = $args -join ' '
    $global:LASTEXITCODE = 0
    if ($command -match 'ms-marco-MiniLM-L-6-v2') {
        'synthetic-model-call edge-reranker'
        if ($env:SYNTHETIC_MODEL_THROW -eq '1') { throw 'synthetic launch failure' }
        $global:LASTEXITCODE = [int]$env:SYNTHETIC_EDGE_RERANKER
    } elseif ($command -match 'Qwen/Qwen3-Embedding-0.6B') {
        'synthetic-model-call base-embedder'
        $global:LASTEXITCODE = [int]$env:SYNTHETIC_BASE_EMBEDDER
    } elseif ($command -match 'Alibaba-NLP/gte-reranker-modernbert-base') {
        'synthetic-model-call base-reranker'
        $global:LASTEXITCODE = [int]$env:SYNTHETIC_BASE_RERANKER
    } elseif ($command -match 'importlib.metadata') { '0.0.0-synthetic' }
    elseif ($command -eq '-m truememory.mcp_server --setup') { $global:LASTEXITCODE = [int]$env:SYNTHETIC_SETUP_EXIT }
    elseif ($command -eq '-m truememory.ingest.cli install') { $global:LASTEXITCODE = [int]$env:SYNTHETIC_HOOK_EXIT }
    else { throw 'unexpected synthetic command' }
}
& $env:SYNTHETIC_INSTALLER
"""


class InstallerFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="truememory-installer-767-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.home = self.root / "home"
        self.home.mkdir()
        self.tool_dir = self.root / "tools"
        self.tool_python = self.tool_dir / "truememory" / "bin" / "python"
        self.tool_python.parent.mkdir(parents=True)
        self._executable(self.bin_dir / "uv", FAKE_UV)
        self._executable(self.bin_dir / "curl", "#!/bin/sh\nexit 99\n")
        self._executable(self.tool_python, FAKE_TOOL_PYTHON)

    @staticmethod
    def _executable(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")
        path.chmod(0o700)

    def environment(self, outcomes: tuple[bool, bool, bool], **extra: str) -> dict[str, str]:
        result = {
            "PATH": str(self.bin_dir) + os.pathsep + os.defpath,
            "HOME": str(self.home),
            "USERPROFILE": str(self.home),
            "LOCALAPPDATA": str(self.home),
            "SYNTHETIC_TOOL_DIR": str(self.tool_dir),
            "SYNTHETIC_EDGE_RERANKER": "0" if outcomes[0] else "1",
            "SYNTHETIC_BASE_EMBEDDER": "0" if outcomes[1] else "1",
            "SYNTHETIC_BASE_RERANKER": "0" if outcomes[2] else "1",
            "SYNTHETIC_SETUP_EXIT": "0",
            "SYNTHETIC_HOOK_EXIT": "0",
        }
        # Windows needs its system directory for native process startup.
        for key in ("SYSTEMROOT", "WINDIR", "COMSPEC"):
            if key in os.environ:
                result[key] = os.environ[key]
        result.update(extra)
        return result

    def assert_summary(self, completed: subprocess.CompletedProcess[str], outcomes: tuple[bool, bool, bool]) -> str:
        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, output)
        self.assertIn("package installed.", output)
        self.assertIn(f"Model pre-download checks: {sum(outcomes)}/3 succeeded.", output)
        for label, ready in zip(LABELS, outcomes):
            self.assertIn(f"{label}: {'ready' if ready else 'failed'}", output)
        self.assertEqual("All three requested model pre-download checks passed." in output, all(outcomes))
        self.assertEqual("Model readiness incomplete" in output, not all(outcomes))
        self.assertEqual(output.count("Retry this model only:"), outcomes.count(False))
        self.assertEqual(output.count("synthetic-model-call"), 3)
        self.assertNotIn("tier switching is instant", output)
        self.assertNotIn("all models pre-downloaded", output)
        self.assertIn("re-embedding stored memories", output)
        self.assertIn("Edge embedder was not checked", output)
        return output


@unittest.skipIf(os.name == "nt", "POSIX installer runs on Mac/Linux")
class TestShellInstallerReadiness(InstallerFixture):
    def run_installer(self, outcomes: tuple[bool, bool, bool], **extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/sh", str(ROOT / "install.sh")],
            env=self.environment(outcomes, **extra), cwd=self.root,
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )

    def test_all_model_outcome_combinations(self) -> None:
        for outcomes in itertools.product((False, True), repeat=3):
            with self.subTest(outcomes=outcomes):
                self.assert_summary(self.run_installer(outcomes), outcomes)

    def test_skipped_setup_does_not_claim_registration_or_hooks(self) -> None:
        output = self.assert_summary(self.run_installer((True, True, True), TRUEMEMORY_SKIP_SETUP="1"), (True, True, True))
        self.assertIn("Claude registration: skipped", output)
        self.assertIn("Hooks: skipped", output)

    def test_setup_failures_remain_visible_when_models_succeed(self) -> None:
        output = self.assert_summary(self.run_installer((True, True, True), SYNTHETIC_SETUP_EXIT="1", SYNTHETIC_HOOK_EXIT="1"), (True, True, True))
        self.assertIn("Claude registration: failed", output)
        self.assertIn("Hooks: failed", output)

    def test_missing_tool_python_is_unverified(self) -> None:
        completed = self.run_installer((True, True, True), SYNTHETIC_TOOL_DIR=str(self.root / "missing-tools"))
        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, output)
        self.assertIn("Model pre-download checks: 0/3 succeeded.", output)
        self.assertIn("Model readiness incomplete", output)
        self.assertIn("uv tool dir", output)
        self.assertIn("Claude registration: skipped", output)
        self.assertIn("Hooks: skipped", output)
        self.assertNotIn("synthetic-model-call", output)
        for label in LABELS:
            self.assertIn(f"{label}: not checked", output)


@unittest.skipUnless(POWERSHELL, "Requires PowerShell; exercised by Windows CI")
class TestPowerShellInstallerReadiness(InstallerFixture):
    def run_installer(self, outcomes: tuple[bool, bool, bool], **extra: str) -> subprocess.CompletedProcess[str]:
        wrapper = self.root / "synthetic-installer-wrapper.ps1"
        wrapper.write_text(POWERSHELL_WRAPPER, encoding="utf-8")
        return subprocess.run(
            [str(POWERSHELL), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(wrapper)],
            env=self.environment(outcomes, SYNTHETIC_INSTALLER=str(ROOT / "install.ps1"), **extra),
            cwd=self.root, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )

    def test_all_model_outcome_combinations(self) -> None:
        for outcomes in itertools.product((False, True), repeat=3):
            with self.subTest(outcomes=outcomes):
                self.assert_summary(self.run_installer(outcomes), outcomes)

    def test_skipped_setup_does_not_claim_registration_or_hooks(self) -> None:
        output = self.assert_summary(self.run_installer((True, True, True), TRUEMEMORY_SKIP_SETUP="1"), (True, True, True))
        self.assertIn("Claude registration: skipped", output)
        self.assertIn("Hooks: skipped", output)

    def test_setup_failures_remain_visible_when_models_succeed(self) -> None:
        output = self.assert_summary(self.run_installer((True, True, True), SYNTHETIC_SETUP_EXIT="1", SYNTHETIC_HOOK_EXIT="1"), (True, True, True))
        self.assertIn("Claude registration: failed", output)
        self.assertIn("Hooks: failed", output)

    def test_missing_tool_python_is_unverified(self) -> None:
        completed = self.run_installer((True, True, True), SYNTHETIC_MISSING_PYTHON="1")
        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, output)
        self.assertIn("Model pre-download checks: 0/3 succeeded.", output)
        self.assertIn("Model readiness incomplete", output)
        self.assertIn("Claude registration: skipped", output)
        self.assertIn("Hooks: skipped", output)
        self.assertNotIn("synthetic-model-call", output)
        for label in LABELS:
            self.assertIn(f"{label}: not checked", output)

    def test_model_launch_exception_is_a_failed_check(self) -> None:
        self.assert_summary(self.run_installer((False, True, True), SYNTHETIC_MODEL_THROW="1"), (False, True, True))


class TestSelectedTierSetupReadiness(unittest.TestCase):
    def run_setup(
        self, tier: str, failures: set[str], fail_on_load: bool = False,
        *, cli: str = "", integration_outcomes: dict[str, bool] | None = None,
        manager_action: str = "noop",
    ) -> tuple[int, str, str, list[dict], list[str]]:
        # Execute both real setup functions while replacing filesystem, adapter
        # installation, and model boundaries. Package __init__ is never imported.
        source = ast.parse((ROOT / "truememory" / "ingest" / "cli.py").read_text(encoding="utf-8"))
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("_setup_cli_integrations", "_setup_model_readiness", "_run_setup")]
        program = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
        saved: list[dict] = []
        calls: list[str] = []
        config_path = Path("synthetic-setup/config.json")
        config_state = {"tier": tier, "user_id": "synthetic-user"}

        class SyntheticModel:
            def encode(self, texts: list[str], **_kwargs: object) -> list[list[float]]:
                calls.append("encode")
                self_check.assertEqual(texts, ["TrueMemory setup readiness check."])
                if "embed" in failures:
                    raise RuntimeError("synthetic embedding failure")
                return [[1.0, 0.0]]

            def predict(self, pairs: list[tuple[str, str]], **_kwargs: object) -> list[float]:
                calls.append("predict")
                self_check.assertEqual(pairs, [("TrueMemory setup readiness check.", "TrueMemory setup readiness check.")])
                if "rerank" in failures:
                    raise RuntimeError("synthetic reranker failure")
                return [0.5]

        self_check = self

        def load(kind: str) -> SyntheticModel:
            calls.append("load-" + kind)
            if fail_on_load and kind in failures:
                raise OSError("synthetic model download failure")
            return SyntheticModel()

        package = types.ModuleType("truememory")
        package.__path__ = []
        vectors = types.ModuleType("truememory.vector_search")
        vectors.set_embedding_model = lambda value: calls.append("embed-tier-" + value)
        vectors.get_model = lambda: load("embed")
        vectors._encode_with_mps_fallback = lambda model, texts, **kwargs: model.encode(texts, **kwargs)
        reranker = types.ModuleType("truememory.reranker")
        reranker.set_active_tier = lambda value: calls.append("rerank-tier-" + value)
        reranker.get_reranker = lambda: load("rerank")
        hooks = types.ModuleType("truememory.hooks")
        hooks.__path__ = []
        registry = types.ModuleType("truememory.hooks.registry")
        adapters = {
            cli_id: types.SimpleNamespace(cli_id=cli_id, name=name, is_configured=lambda: False)
            for cli_id, name in (("claude", "Claude Code"), ("codex", "Codex"))
        }
        registry.get_adapter = adapters.get
        registry.detect_installed = lambda: list(adapters.values())
        hooks_cli = types.ModuleType("truememory.hooks.cli")

        def install_cli(cli_id: str, user_id: str) -> bool:
            calls.append("install-cli-" + cli_id)
            self.assertEqual(user_id, "synthetic-user")
            return (integration_outcomes or {}).get(cli_id, True)

        hooks_cli.install_cli = install_cli
        namespace = {
            "argparse": argparse, "os": os, "sys": sys, "_SAURON_BANNER": "synthetic banner",
            "_load_truememory_config": lambda: config_state.copy(),
            "_save_truememory_config": lambda config: saved.append(dict(config)),
            "_TRUEMEMORY_CONFIG_PATH": config_path,
        }
        bridge = runpy.run_path(str(ROOT / "tests/test-tier-runtime-bridge-795.py"))
        fixture = bridge["TestRuntimeBridge"]()
        fixture.setUp()
        fixture.vector.resolve_tier = lambda: tier
        fixture.vector.get_model = vectors.get_model
        fixture.vector._encode_with_mps_fallback = vectors._encode_with_mps_fallback
        fixture.reranker.get_reranker = reranker.get_reranker
        primitives = runpy.run_path(str(ROOT / "tests/test-maintenance-source-revision-753.py"))
        maintenance = primitives["MAINTENANCE"]

        class SyntheticManager:
            _last_action = manager_action

            def run_rebuild_sync(self, requested_tier: str) -> bool:
                self_check.assertEqual(requested_tier, tier)
                vectors.set_embedding_model(requested_tier)
                reranker.set_active_tier(requested_tier)
                return True

        modules = {
            "truememory.vector_search": fixture.vector, "truememory.reranker": fixture.reranker,
            "truememory.maintenance": maintenance,
            "truememory.tier_switch.runtime": fixture.api,
            "truememory.tier_switch.manager": types.SimpleNamespace(
                RebuildManager=SyntheticManager, _get_db_path=lambda: None, _open_db=lambda **kwargs: fixture.conn),
            "truememory.hooks.registry": registry, "truememory.hooks.cli": hooks_cli,
        }

        def synthetic_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in modules:
                return modules[name]
            if name.startswith("truememory") or name in {"torch", "numpy", "sqlite_vec", "sentence_transformers"}:
                raise AssertionError("Unexpected setup dependency: " + name)
            return builtins.__import__(name, globals, locals, fromlist, level)

        namespace["__builtins__"] = dict(vars(builtins), __import__=synthetic_import)
        projection = types.ModuleType("synthetic_setup_projection")
        projection.__dict__["__builtins__"] = namespace["__builtins__"]
        projection_path = ROOT / "truememory/tier_switch/projection.py"
        exec(compile(projection_path.read_text(encoding="utf-8"), str(projection_path), "exec"), projection.__dict__)
        config_file_lock = threading.Lock()

        @contextlib.contextmanager
        def locked_config(path: Path, *, strict: bool = False):
            self_check.assertEqual(path, config_path.with_name("config.json.lock"))
            self_check.assertTrue(strict)
            with config_file_lock:
                yield

        def read_config(path: Path) -> dict:
            self_check.assertEqual(path, config_path)
            self_check.assertTrue(config_file_lock.locked())
            self_check.assertTrue(projection.CONFIG_WRITE_LOCK.locked())
            return config_state.copy()

        def write_config(path: Path, config: dict) -> None:
            self_check.assertEqual(path, config_path)
            self_check.assertTrue(config_file_lock.locked())
            self_check.assertTrue(projection.CONFIG_WRITE_LOCK.locked())
            config_state.clear()
            config_state.update(config)
            saved.append(config_state.copy())

        projection.ConfigFileLock = locked_config
        projection._read_config = read_config
        projection._write_config = write_config
        modules["truememory.tier_switch.projection"] = projection
        exec(compile(program, "synthetic-cli-setup", "exec"), namespace)
        stdout, stderr = io.StringIO(), io.StringIO()
        exit_code = 0
        try:
            with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    namespace["_run_setup"](argparse.Namespace(non_interactive=True, cli=cli))
                except SystemExit as exc:
                    exit_code = int(exc.code)
        finally:
            fixture.doCleanups()
        return exit_code, stdout.getvalue(), stderr.getvalue(), saved, calls

    def test_selected_models_fail_truthfully_and_retain_configuration(self) -> None:
        for fail_on_load in (False, True):
            for failures in ({"embed"}, {"rerank"}, {"embed", "rerank"}):
                with self.subTest(failures=failures, fail_on_load=fail_on_load):
                    code, output, error, saved, calls = self.run_setup("base", failures, fail_on_load)
                    self.assertEqual(code, 1)
                    self.assertIn("model readiness is incomplete", output)
                    self.assertIn("truememory-ingest setup --non-interactive", output)
                    self.assertNotIn("Setup complete!", output)
                    self.assertIn("readiness check failed", error)
                    self.assertEqual(saved, [{"tier": "base", "user_id": "synthetic-user"}])
                    self.assertIn("install-cli-claude", calls)
                    self.assertIn("install-cli-codex", calls)
                    if "embed" in failures:
                        self.assertNotIn("Matryoshka ready", output)
                    if "rerank" in failures:
                        self.assertNotIn("Cross-encoder reranker ready", output)

    def test_successful_setup_checks_and_preserves_each_selected_tier(self) -> None:
        for tier in ("edge", "base", "pro"):
            with self.subTest(tier=tier):
                code, output, error, saved, calls = self.run_setup(tier, set())
                self.assertEqual(code, 0)
                self.assertIn("Setup complete!", output)
                self.assertEqual(error, "")
                self.assertEqual(saved[0]["tier"], tier)
                self.assertIn("embed-tier-" + tier, calls)
                self.assertIn("rerank-tier-" + tier, calls)
                self.assertEqual(calls.count("encode"), 1)
                self.assertEqual(calls.count("predict"), 1)

    def test_requested_integration_outcomes_control_setup_completion(self) -> None:
        for cli in ("", "claude,codex"):
            for outcomes in itertools.product((False, True), repeat=2):
                with self.subTest(cli=cli or "detected", outcomes=outcomes):
                    results = dict(zip(("claude", "codex"), outcomes))
                    code, output, error, saved, calls = self.run_setup(
                        "base", set(), cli=cli, integration_outcomes=results,
                    )
                    self.assertEqual(code, 0 if all(outcomes) else 1)
                    self.assertEqual("Setup complete!" in output, all(outcomes))
                    self.assertEqual(saved, [{"tier": "base", "user_id": "synthetic-user"}])
                    self.assertEqual(error, "")
                    self.assertIn("Matryoshka ready", output)
                    self.assertIn("Cross-encoder reranker ready", output)
                    for cli_id in results:
                        self.assertEqual(calls.count("install-cli-" + cli_id), 1)
                    if not all(outcomes):
                        self.assertIn("Failed CLI integrations:", output)
                        self.assertIn("truememory-ingest setup --non-interactive --cli", output)
                        self.assertNotIn("TrueMemory is ready", output)

    def test_unknown_requested_integration_prevents_setup_completion(self) -> None:
        code, output, _error, saved, calls = self.run_setup("edge", set(), cli="claude,unknown")
        self.assertEqual(code, 1)
        self.assertIn("Unknown CLI: unknown", output)
        self.assertIn("Failed CLI integrations: unknown", output)
        self.assertNotIn("Setup complete!", output)
        self.assertEqual(saved[0]["tier"], "edge")
        self.assertIn("install-cli-claude", calls)
        self.assertNotIn("install-cli-unknown", calls)

    def test_config_only_setup_reports_no_unperformed_model_readiness(self) -> None:
        code, output, error, saved, calls = self.run_setup("pro", set(), manager_action="config_only")
        self.assertEqual(code, 0)
        self.assertEqual(error, "")
        self.assertEqual(saved, [{"tier": "pro", "user_id": "synthetic-user"}])
        self.assertNotIn("encode", calls)
        self.assertNotIn("predict", calls)
        self.assertNotIn("load-embed", calls)
        self.assertNotIn("load-rerank", calls)
        self.assertNotIn("Matryoshka ready", output)
        self.assertNotIn("Cross-encoder reranker ready", output)


if __name__ == "__main__":
    unittest.main()
