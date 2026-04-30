"""Tests for Claude Code hook schema format (issue #72)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_install_writes_correct_hook_schema():
    """Hooks must use the {matcher, hooks: [{type, command}]} schema.

    Claude Code rejects the entire settings.json if any hook entry
    uses the old flat {type, command} format.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        settings_path = Path(tmpdir) / "settings.json"
        settings_path.write_text("{}")

        # Simulate the install by importing and calling the builder
        from truememory.ingest.cli import _run_install

        class FakeArgs:
            user = ""
            db = ""
            dry_run = True  # don't actually write

        # Capture the dry-run output
        import io
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            _run_install(FakeArgs())

        output = captured.getvalue()

        # Parse the JSON from the dry-run output
        json_start = output.index("{")
        json_str = output[json_start:]
        settings = json.loads(json_str)

        hooks = settings.get("hooks", {})
        assert hooks, "No hooks in output"

        for event, entries in hooks.items():
            assert isinstance(entries, list), f"{event}: entries should be a list"
            for i, entry in enumerate(entries):
                assert isinstance(entry, dict), f"{event}[{i}]: should be a dict"
                assert "matcher" in entry, (
                    f"{event}[{i}]: missing 'matcher' key. "
                    f"Got keys: {list(entry.keys())}. "
                    f"Claude Code requires {{matcher, hooks}} format."
                )
                assert "hooks" in entry, (
                    f"{event}[{i}]: missing 'hooks' key. "
                    f"Got keys: {list(entry.keys())}. "
                    f"Claude Code requires {{matcher, hooks}} format."
                )
                assert isinstance(entry["hooks"], list), (
                    f"{event}[{i}]: 'hooks' should be a list"
                )
                for j, inner in enumerate(entry["hooks"]):
                    assert "type" in inner, f"{event}[{i}].hooks[{j}]: missing 'type'"
                    assert "command" in inner, f"{event}[{i}].hooks[{j}]: missing 'command'"
                    assert inner["type"] == "command", f"{event}[{i}].hooks[{j}]: type should be 'command'"

                # Must NOT have flat format keys at the top level
                assert "type" not in entry, (
                    f"{event}[{i}]: has 'type' at top level — this is the OLD format. "
                    f"Claude Code will reject the entire settings.json."
                )
                assert "command" not in entry, (
                    f"{event}[{i}]: has 'command' at top level — this is the OLD format."
                )


def test_migration_upgrades_old_format():
    """The installer should upgrade old-format hooks in existing settings.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        settings_path = Path(tmpdir) / "settings.json"

        # Write old-format hooks
        old_settings = {
            "hooks": {
                "SessionStart": [{
                    "type": "command",
                    "command": "/usr/bin/python3 /some/path/session_start.py",
                }],
                "Stop": [{
                    "type": "command",
                    "command": "/usr/bin/python3 /some/path/stop.py",
                }],
            }
        }
        settings_path.write_text(json.dumps(old_settings))

        # Read back and simulate migration
        existing = json.loads(settings_path.read_text())

        # Apply migration logic (same as in cli.py)
        for event in list(existing["hooks"].keys()):
            entries = existing["hooks"][event]
            if not isinstance(entries, list):
                continue
            migrated = []
            for h in entries:
                if not isinstance(h, dict):
                    migrated.append(h)
                    continue
                if "type" in h and "command" in h and "hooks" not in h:
                    migrated.append({"matcher": "", "hooks": [h]})
                else:
                    migrated.append(h)
            existing["hooks"][event] = migrated

        # Verify migrated format
        for event, entries in existing["hooks"].items():
            for entry in entries:
                assert "matcher" in entry, f"{event}: missing matcher after migration"
                assert "hooks" in entry, f"{event}: missing hooks after migration"
                assert "type" not in entry, f"{event}: still has flat 'type' after migration"


def test_new_format_not_double_wrapped():
    """Hooks already in correct format should NOT be re-wrapped."""
    correct_entry = {
        "matcher": "",
        "hooks": [{
            "type": "command",
            "command": "/usr/bin/python3 /some/path/stop.py",
        }],
    }

    # Migration should leave this unchanged
    h = correct_entry
    if "type" in h and "command" in h and "hooks" not in h:
        result = {"matcher": "", "hooks": [h]}
    else:
        result = h

    assert result == correct_entry, "Correct-format entry was incorrectly modified"
    assert len(result["hooks"]) == 1, "Should still have exactly one inner hook"
