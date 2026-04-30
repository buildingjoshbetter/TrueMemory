"""Tests for first-run onboarding (issues #74, #127)."""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_first_run_detected_without_marker():
    """No marker file → first run → banner should appear."""
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / ".onboarded"
        with patch("truememory.ingest.hooks.session_start.ONBOARDED_MARKER", marker):
            import truememory.ingest.hooks.session_start as ss
            assert ss._is_first_run()


def test_not_first_run_with_marker():
    """Marker file exists → not first run → no banner."""
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / ".onboarded"
        marker.write_text("tier=base\n")
        with patch("truememory.ingest.hooks.session_start.ONBOARDED_MARKER", marker):
            import truememory.ingest.hooks.session_start as ss
            assert not ss._is_first_run()


def test_first_run_context_has_banner():
    """First-run context should include the ASCII banner."""
    import truememory.ingest.hooks.session_start as ss
    ctx = ss._first_run_context()
    assert "████████" in ctx


def test_first_run_context_has_setup_guide():
    """First-run context should include tier selection instructions."""
    import truememory.ingest.hooks.session_start as ss
    ctx = ss._first_run_context()
    assert "Edge" in ctx
    assert "Base" in ctx
    assert "Pro" in ctx
    assert "truememory_configure" in ctx


def test_first_run_hook_outputs_valid_json():
    """On first run, the hook should output valid JSON with additionalContext."""
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / ".onboarded"
        with patch("truememory.ingest.hooks.session_start.ONBOARDED_MARKER", marker):
            import truememory.ingest.hooks.session_start as ss
            import sys
            with patch("sys.stdin", io.StringIO("{}")):
                old_stdout = sys.stdout
                sys.stdout = captured = io.StringIO()
                ss.main()
                sys.stdout = old_stdout
                output = captured.getvalue().strip()
                assert output, "Hook should produce output on first run"
                data = json.loads(output)
                assert "additionalContext" in data
                assert "████████" in data["additionalContext"]


def test_banner_not_shown_on_subsequent_runs():
    """After onboarding, the hook should inject memories, not the banner."""
    with tempfile.TemporaryDirectory() as tmpdir:
        marker = Path(tmpdir) / ".onboarded"
        marker.write_text("tier=base\n")
        with patch("truememory.ingest.hooks.session_start.ONBOARDED_MARKER", marker):
            import truememory.ingest.hooks.session_start as ss
            import sys
            with patch("sys.stdin", io.StringIO("{}")):
                old_stdout = sys.stdout
                sys.stdout = captured = io.StringIO()
                ss.main()
                sys.stdout = old_stdout
                output = captured.getvalue().strip()
                if output:
                    data = json.loads(output)
                    assert "████████" not in data.get("additionalContext", "")
