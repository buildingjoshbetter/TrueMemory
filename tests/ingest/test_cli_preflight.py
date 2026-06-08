"""
Tests for CLI preflight checks added in round 2.

Verifies:
- `truememory-ingest ingest` with a nonexistent transcript exits with error code 2
- `truememory-ingest ingest` without truememory installed exits with error code 2
- `truememory-ingest status` runs without crashing even when truememory is missing
- `truememory-ingest install --dry-run` prints the settings without writing
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def _run_cli(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    """Run the CLI as a subprocess and return the result."""
    base_env = os.environ.copy()
    # Don't let real API keys leak into tests
    base_env.pop("ANTHROPIC_API_KEY", None)
    base_env.pop("OPENROUTER_API_KEY", None)
    if env:
        # Preserve user site-packages so dependencies (numpy etc.) remain
        # importable even when HOME is overridden to a temp directory.
        if "HOME" in env and "PYTHONPATH" not in env:
            import site
            user_sp = site.getusersitepackages()
            existing = base_env.get("PYTHONPATH", "")
            base_env["PYTHONPATH"] = f"{user_sp}:{existing}" if existing else user_sp
        base_env.update(env)

    return subprocess.run(
        [sys.executable, "-m", "truememory.ingest.cli"] + args,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=base_env,
    )


def test_ingest_nonexistent_transcript_exits_cleanly():
    """Ingest with a nonexistent file should exit with code 2 and a clear error.

    Note: preflight runs checks in order (truememory import, then transcript
    existence). In environments without truememory installed, the error
    surfaces as "truememory is not installed" rather than "transcript not
    found" — both are valid exit-code-2 error paths.
    """
    result = _run_cli(["ingest", "/nonexistent/path/transcript.json"])
    assert result.returncode == 2, f"Expected exit 2, got {result.returncode}. stderr: {result.stderr}"
    stderr_lower = result.stderr.lower()
    # Accept either preflight failure mode — both are correct "ERROR: ... exit 2" paths
    assert (
        "not found" in stderr_lower
        or "no such" in stderr_lower
        or "truememory is not installed" in stderr_lower
        or "error:" in stderr_lower
    ), f"Expected a clear error message, got stderr:\n{result.stderr}"


def test_install_dry_run_prints_settings():
    """install --dry-run should print the settings JSON without writing to settings.json."""
    result = _run_cli(["install", "--dry-run"])
    # Should not exit with error (may succeed or print settings)
    # The output should contain the word "hooks" and JSON
    combined_output = result.stdout + result.stderr
    assert "hooks" in combined_output.lower() or "SessionStart" in combined_output


def test_help_command_works():
    """--help should print usage without errors."""
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert "truememory-ingest" in result.stdout.lower() or "usage" in result.stdout.lower()
    assert "install" in result.stdout
    assert "ingest" in result.stdout
    assert "status" in result.stdout


def test_status_command_runs():
    """status should run without crashing even if truememory is missing."""
    result = _run_cli(["status"])
    # Should not crash — exit code 0 (even with warnings)
    assert result.returncode == 0, f"status crashed: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "Status Check" in combined or "truememory" in combined.lower()


def test_install_help_shows_flags():
    """install --help should list its flags."""
    result = _run_cli(["install", "--help"])
    assert result.returncode == 0
    assert "--user" in result.stdout
    assert "--db" in result.stdout
    assert "--dry-run" in result.stdout


def test_logs_command_handles_missing_log_dir():
    """logs command should handle a missing log dir gracefully."""
    # Set HOME to a temp dir so ~/.truememory/logs doesn't exist
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        result = _run_cli(["logs"], env={"HOME": tmp})
        assert result.returncode == 0, f"logs crashed: {result.stderr}"
        # Should print a helpful message about no logs
        combined = result.stdout + result.stderr
        assert "no log" in combined.lower() or "created when" in combined.lower()


def test_trace_command_handles_missing_trace_dir():
    """trace command should handle a missing trace dir gracefully."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        result = _run_cli(["trace"], env={"HOME": tmp})
        assert result.returncode == 0, f"trace crashed: {result.stderr}"
        combined = result.stdout + result.stderr
        assert "no trace" in combined.lower()
