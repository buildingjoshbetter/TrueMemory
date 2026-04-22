"""Regression lock for Hunter F23 + F25 — subprocess.run timeouts.

F23: the four `claude` CLI calls in `mcp_server._setup_claude` now route
through a `_run_claude` helper that passes `timeout=30` and reports
`TimeoutExpired` to stderr instead of hanging forever.

F25: the `pip install truememory[gpu]` call in `ingest/cli.py` now has
`timeout=600` and a clear stderr message + fallback to Edge tier on
timeout.
"""
from __future__ import annotations

import subprocess


# ---------------------------------------------------------------------------
# F23 — `claude` CLI calls bounded by timeout
# ---------------------------------------------------------------------------


def test_setup_claude_claude_add_timeout_reports_and_falls_through(monkeypatch, capsys):
    """If `claude mcp add` times out, setup must print to stderr and not hang."""
    import truememory.mcp_server as ms
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/claude" if name == "claude" else None)

    def _boom(cmd, **kwargs):
        assert "timeout" in kwargs, "subprocess.run must be called with timeout="
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

    # Bypass Claude Desktop path so this test is scoped to the CLI path.
    from pathlib import Path
    monkeypatch.setattr(
        Path, "exists",
        lambda self: False if "Application Support" in str(self) else True,
    )
    monkeypatch.setattr(subprocess, "run", _boom)
    ms._setup_claude()
    captured = capsys.readouterr()
    assert "timed out" in captured.err.lower()
    assert "claude code" in captured.err.lower()


def test_setup_claude_uses_30s_timeout_on_add_call(monkeypatch):
    """The first `claude mcp add` call must include a timeout kwarg."""
    import truememory.mcp_server as ms
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/claude" if name == "claude" else None)

    from pathlib import Path
    monkeypatch.setattr(
        Path, "exists",
        lambda self: False if "Application Support" in str(self) else True,
    )
    seen_calls = []

    def _capture(cmd, **kwargs):
        seen_calls.append((cmd, kwargs))
        # Simulate a successful add so _setup_claude short-circuits
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _capture)
    ms._setup_claude()
    assert seen_calls, "subprocess.run was never called"
    first_cmd, first_kwargs = seen_calls[0]
    assert "timeout" in first_kwargs
    assert first_kwargs["timeout"] == 30  # the finding's recommendation


def test_setup_claude_timeout_on_list_call_is_handled(monkeypatch, capsys):
    """When add reports 'already exists' and the follow-up `claude mcp list`
    times out, setup must NOT crash — it should fall through to Claude
    Desktop (preserving the 'existing_cmd = ""' default path)."""
    import truememory.mcp_server as ms
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/claude" if name == "claude" else None)
    from pathlib import Path
    monkeypatch.setattr(
        Path, "exists",
        lambda self: False if "Application Support" in str(self) else True,
    )

    call_count = [0]

    def _flaky(cmd, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First: `claude mcp add` → returns "already exists"
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="MCP server 'truememory' already exists"
            )
        # Subsequent: `claude mcp list` times out
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 30))

    monkeypatch.setattr(subprocess, "run", _flaky)
    # Must not raise
    ms._setup_claude()
    captured = capsys.readouterr()
    assert "timed out" in captured.err.lower()


# ---------------------------------------------------------------------------
# F25 — `pip install truememory[gpu]` bounded by timeout
# ---------------------------------------------------------------------------


def test_pip_install_has_timeout_kwarg(monkeypatch):
    """Source-level check: the pip install subprocess.run call must include
    a timeout kwarg. This guards against accidental removal."""
    import pathlib
    cli_source = pathlib.Path(__file__).parent.parent / "truememory" / "ingest" / "cli.py"
    text = cli_source.read_text()
    # Locate the pip install call and verify timeout is declared nearby.
    idx = text.find('"truememory[gpu]"]')
    assert idx != -1, "pip install call for truememory[gpu] not found in cli.py"
    surrounding = text[idx : idx + 300]
    assert "timeout=600" in surrounding, (
        "F25 regression: pip install must declare timeout=600 (10 min) "
        "so PyPI/mirror stalls don't wedge `truememory-ingest setup`"
    )


def test_pip_install_timeout_handler_prints_stderr_and_falls_back_to_edge():
    """Simulate the TimeoutExpired path by reading the source and
    confirming the expected recovery behaviour is wired up."""
    import pathlib
    cli_source = pathlib.Path(__file__).parent.parent / "truememory" / "ingest" / "cli.py"
    text = cli_source.read_text()
    # Must have a TimeoutExpired handler for the pip call.
    assert "subprocess.TimeoutExpired" in text, (
        "F25 regression: TimeoutExpired must be caught, not propagated"
    )
    # Must warn the user to stderr with an actionable message.
    assert "file=sys.stderr" in text
    # Must fall back to edge tier on timeout.
    pip_idx = text.find('"truememory[gpu]"]')
    assert pip_idx != -1
    post = text[pip_idx : pip_idx + 800]
    assert 'tier = "edge"' in post, (
        "F25 regression: on pip timeout, setup must fall back to Edge tier "
        "rather than leave the user in an indeterminate state"
    )
