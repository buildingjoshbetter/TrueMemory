"""Regression lock: extracted/ marker pruning must run even when there is
nothing to scan (SRE-03 / issue #694).

Pre-fix: _cleanup_extracted_markers ran only at the very END of
_scan_stale_sessions, after a claude_dir-missing early return (and others), so
when ~/.claude/projects was absent the markers were never pruned and grew
unbounded (54K+ observed in prod). Post-fix it runs right after the scan
interval gate, before the claude_dir check.

No model loads.
"""
import os


import time


def test_pruning_runs_when_claude_projects_absent(tmp_path, monkeypatch):
    import truememory.ingest.hooks.session_start as ss
    import truememory.ingest.hooks._shared as shared

    # Isolate all of ~/.truememory + ~/.claude into tmp, with NO ~/.claude/projects.
    fake_home = tmp_path / "home"
    extracted = fake_home / ".truememory" / "extracted"
    extracted.mkdir(parents=True)
    scan_marker = fake_home / ".truememory" / ".last_stale_scan"
    monkeypatch.setattr(shared, "EXTRACTED_DIR", extracted)
    monkeypatch.setattr(ss, "_SCAN_MARKER", scan_marker)
    monkeypatch.setattr(ss, "_EXTRACTED_MARKER_MAX_AGE", 30 * 86400)
    # Point Path.home() at fake_home so claude_dir = fake_home/.claude/projects (absent).
    monkeypatch.setattr(ss.Path, "home", staticmethod(lambda: fake_home))

    # Seed an OLD marker (older than the max age) and a FRESH one.
    old = extracted / "old-session"
    old.write_text("{}")
    old_time = time.time() - 40 * 86400
    os.utime(old, (old_time, old_time))
    fresh = extracted / "fresh-session"
    fresh.write_text("{}")

    assert not (fake_home / ".claude" / "projects").exists()  # nothing to scan

    ss._scan_stale_sessions()

    # The old marker is pruned even though there was no claude_dir to scan.
    assert not old.exists(), "old marker not pruned when ~/.claude/projects absent (SRE-03 regression)"
    assert fresh.exists(), "fresh marker should be kept"
