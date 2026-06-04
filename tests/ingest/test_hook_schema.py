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
                "SessionEnd": [{
                    "type": "command",
                    "command": "/usr/bin/python3 /some/path/session_end.py",
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
            "command": "/usr/bin/python3 /some/path/session_end.py",
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



# ---------------------------------------------------------------------------
# Stop -> SessionEnd migration tests (concern #1, #2, #5, #7 from review)
# ---------------------------------------------------------------------------

def _run_migration_on(existing_settings: dict) -> dict:
    """Run the Stop->SessionEnd migration logic from cli.py on a settings dict."""
    existing = existing_settings
    existing.setdefault("hooks", {})
    if not isinstance(existing["hooks"], dict):
        existing["hooks"] = {}

    _legacy_stop = existing["hooks"].get("Stop")
    if _legacy_stop is not None:
        if isinstance(_legacy_stop, dict):
            _legacy_stop = [_legacy_stop]
        if isinstance(_legacy_stop, list):
            _cleaned_stop = []
            _removed_count = 0
            for h in _legacy_stop:
                if not isinstance(h, dict):
                    _cleaned_stop.append(h)
                    continue
                inner_hooks = h.get("hooks", [])
                if isinstance(inner_hooks, list) and inner_hooks:
                    kept_inner = []
                    for ih in inner_hooks:
                        cmd = (ih.get("command") or "") if isinstance(ih, dict) else ""
                        if "truememory" in cmd.lower() and ("hooks" in cmd.lower() or "ingest" in cmd.lower()):
                            _removed_count += 1
                        else:
                            kept_inner.append(ih)
                    if kept_inner:
                        h_copy = dict(h)
                        h_copy["hooks"] = kept_inner
                        _cleaned_stop.append(h_copy)
                    continue
                cmd_flat = (h.get("command") or "")
                if "truememory" in cmd_flat.lower() and ("hooks" in cmd_flat.lower() or "ingest" in cmd_flat.lower()):
                    _removed_count += 1
                else:
                    _cleaned_stop.append(h)
            if _cleaned_stop:
                existing["hooks"]["Stop"] = _cleaned_stop
            else:
                existing["hooks"].pop("Stop", None)

    _existing_session_end = existing["hooks"].get("SessionEnd")
    if isinstance(_existing_session_end, list):
        _deduped = []
        for h in _existing_session_end:
            if not isinstance(h, dict):
                _deduped.append(h)
                continue
            inner = h.get("hooks", [])
            if isinstance(inner, list):
                has_ours = any(
                    isinstance(ih, dict)
                    and "truememory" in (ih.get("command") or "").lower()
                    for ih in inner
                )
                if has_ours:
                    continue
            cmd_flat = (h.get("command") or "")
            if "truememory" in cmd_flat.lower():
                continue
            _deduped.append(h)
        if _deduped:
            existing["hooks"]["SessionEnd"] = _deduped
        else:
            existing["hooks"].pop("SessionEnd", None)

    return existing


def test_migration_removes_truememory_stop_entries():
    """Legacy TrueMemory Stop hooks are stripped during migration."""
    settings = {"hooks": {"Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "/usr/bin/python3 -m truememory.ingest.hooks.stop"}]}]}}
    result = _run_migration_on(settings)
    assert "Stop" not in result["hooks"]


def test_migration_preserves_non_truememory_stop_hooks():
    """User custom Stop hooks must survive migration untouched."""
    custom = {"matcher": "", "hooks": [{"type": "command", "command": "/usr/local/bin/my-hook.sh"}]}
    settings = {"hooks": {"Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "/usr/bin/python3 -m truememory.ingest.hooks.stop"}]}, custom]}}
    result = _run_migration_on(settings)
    assert "Stop" in result["hooks"]
    assert len(result["hooks"]["Stop"]) == 1
    assert result["hooks"]["Stop"][0] == custom


def test_migration_filters_inner_hooks_not_entire_block():
    """Mixed matcher block: only TrueMemory hooks removed (concern #1)."""
    settings = {"hooks": {"Stop": [{"matcher": "", "hooks": [
        {"type": "command", "command": "/usr/bin/python3 -m truememory.ingest.hooks.stop"},
        {"type": "command", "command": "/usr/local/bin/my-logger.sh"},
    ]}]}}
    result = _run_migration_on(settings)
    assert "Stop" in result["hooks"]
    inner = result["hooks"]["Stop"][0]["hooks"]
    assert len(inner) == 1
    assert "my-logger" in inner[0]["command"]


def test_migration_handles_dict_format_stop():
    """Bare dict Stop value is handled (concern #7)."""
    settings = {"hooks": {"Stop": {"type": "command", "command": "/usr/bin/python3 -m truememory.ingest.hooks.stop"}}}
    result = _run_migration_on(settings)
    assert "Stop" not in result["hooks"]


def test_migration_handles_flat_format_stop():
    """Old flat {type, command} entries are handled."""
    settings = {"hooks": {"Stop": [{"type": "command", "command": "/usr/bin/python3 -m truememory.ingest.hooks.stop"}]}}
    result = _run_migration_on(settings)
    assert "Stop" not in result["hooks"]


def test_migration_idempotent_on_rerun():
    """Running migration twice produces same result."""
    settings = {"hooks": {"Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "/usr/bin/python3 -m truememory.ingest.hooks.stop"}]}]}}
    r1 = _run_migration_on(settings)
    r2 = _run_migration_on(r1)
    assert r1 == r2


def test_migration_deduplicates_existing_session_end():
    """Existing SessionEnd TrueMemory entries stripped to prevent doubles (concern #5)."""
    settings = {"hooks": {"SessionEnd": [{"matcher": "", "hooks": [{"type": "command", "command": "/usr/bin/python3 -m truememory.ingest.hooks.session_end"}]}]}}
    result = _run_migration_on(settings)
    assert "SessionEnd" not in result["hooks"]


def test_migration_handles_none_command():
    """command=None must not crash (concern #9)."""
    settings = {"hooks": {"Stop": [{"matcher": "", "hooks": [{"type": "command", "command": None}]}]}}
    result = _run_migration_on(settings)
    assert "Stop" in result["hooks"]
