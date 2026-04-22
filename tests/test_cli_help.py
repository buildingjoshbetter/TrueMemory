"""CLI flag handling for `truememory-mcp`.

Regression tests for the v0.4.1 fix to the --help hang bug: without this
handling, any unknown argument (including --help) fell through to
mcp.run(transport="stdio") which blocks on stdin forever, making
`pip install truememory && truememory-mcp --help` hang.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from truememory import __version__


def _truememory_mcp_bin() -> str | None:
    """Locate the installed truememory-mcp console script, or None.

    Prefer the script installed by pip. Fall back to invoking via
    `python -m truememory.mcp_server` — slower because it re-runs all
    module-level imports, but works in any environment where truememory
    is importable.
    """
    return shutil.which("truememory-mcp")


def _run_cli(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    bin_path = _truememory_mcp_bin()
    if bin_path:
        cmd = [bin_path] + args
    else:
        cmd = [sys.executable, "-m", "truememory.mcp_server"] + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def test_help_long_flag_exits_cleanly():
    """`truememory-mcp --help` must exit 0 with usage text, not hang."""
    result = _run_cli(["--help"])
    assert result.returncode == 0, f"non-zero exit: {result.returncode}\nstderr: {result.stderr}"
    assert "Usage: truememory-mcp" in result.stdout, f"stdout missing usage line:\n{result.stdout}"
    assert "--setup" in result.stdout
    assert "--version" in result.stdout


def test_help_short_flag_exits_cleanly():
    """`truememory-mcp -h` must behave identically to --help."""
    result = _run_cli(["-h"])
    assert result.returncode == 0, f"non-zero exit: {result.returncode}\nstderr: {result.stderr}"
    assert "Usage: truememory-mcp" in result.stdout


def test_version_flag_prints_current_version():
    """`truememory-mcp --version` must print the exact package version and exit 0."""
    result = _run_cli(["--version"])
    assert result.returncode == 0, f"non-zero exit: {result.returncode}\nstderr: {result.stderr}"
    assert __version__ in result.stdout, f"stdout missing version {__version__}:\n{result.stdout}"


def test_version_short_flag_prints_current_version():
    """`truememory-mcp -V` must behave identically to --version."""
    result = _run_cli(["-V"])
    assert result.returncode == 0, f"non-zero exit: {result.returncode}\nstderr: {result.stderr}"
    assert __version__ in result.stdout


@pytest.mark.skipif(not shutil.which("truememory-mcp"), reason="console script not on PATH")
def test_help_via_console_script_does_not_hang():
    """Explicit end-to-end test via the installed console script.

    This is the exact invocation a user would type after `pip install truememory`.
    Uses a tight 15s timeout because the fix should return in milliseconds;
    a hang would mean the --help handling regressed back into the mcp.run() path.
    """
    result = subprocess.run(
        ["truememory-mcp", "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "Usage:" in result.stdout


# --- Regression lock: unknown flags must exit non-zero, not hang ---


def test_unknown_flag_exits_nonzero_not_hang():
    """Any flag we don't recognize must error out, not fall through to
    mcp.run(transport='stdio') and block on stdin.

    This is the same class of bug as the --help hang PR #3 fixed; the
    original fix only handled four specific flags, so an unknown flag
    like `--halp` (user typo) regressed into the original hang.
    The 10s timeout will fire if this test ever catches a hang.
    """
    result = _run_cli(["--halp"], timeout=10)
    # Must exit non-zero (spec: exit 2 for unknown flags, Unix convention)
    assert result.returncode != 0, (
        f"unknown flag did not error; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    # Stderr should mention the unknown flag or point to --help
    assert (
        "unknown" in result.stderr.lower()
        or "usage" in result.stderr.lower()
        or "--help" in result.stderr
    ), f"stderr lacked usage hint: {result.stderr!r}"


# --- truememory-ingest --version parity with truememory-mcp ---


def test_ingest_version_flag_exits_cleanly():
    """`truememory-ingest --version` must exit 0 with the version string,
    matching `truememory-mcp --version` behavior. Before this patch the
    ingest CLI returned 'error: unrecognized arguments: --version' (exit 2),
    which was inconsistent with the MCP CLI.
    """
    bin_path = shutil.which("truememory-ingest")
    if not bin_path:
        pytest.skip("truememory-ingest console script not installed")
    result = subprocess.run(
        [bin_path, "--version"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, (
        f"non-zero exit: {result.returncode}; stderr: {result.stderr}"
    )
    assert __version__ in result.stdout, (
        f"stdout lacks version {__version__}: {result.stdout!r}"
    )
    assert "truememory-ingest" in result.stdout, (
        f"stdout lacks binary name: {result.stdout!r}"
    )


# --- Regression lock: positional args must also exit non-zero, not hang ---


def test_positional_arg_exits_nonzero_not_hang():
    """`truememory-mcp help` (positional arg, no dashes) must NOT fall through
    to mcp.run(transport='stdio') and hang on stdin.

    The round-1 CLI patch only rejected flag-shaped unknowns (startswith('-')),
    leaving positional typos like `help`, `setup`, `halp` to reach the
    mcp.run() fallthrough and block indefinitely. This test locks the
    round-2 fix: any non-empty argv after known-flag processing must exit 2.

    The 10s timeout will fire if this test ever catches a hang regression.
    """
    result = _run_cli(["help"], timeout=10)
    # Must exit non-zero (spec: exit 2 for unexpected positional args)
    assert result.returncode != 0, (
        f"positional arg did not error; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    # Stderr should mention the unexpected arg or point to --help
    assert (
        "unexpected" in result.stderr.lower()
        or "usage" in result.stderr.lower()
        or "--help" in result.stderr
    ), f"stderr lacked usage hint: {result.stderr!r}"
