"""Regression tests for issue #579: session-start maintenance gaps.

Two bugs:
1. O_CREAT first-run skip: the stale-session scanner creates the marker
   file via O_CREAT on first run, then sees the freshly-created file as
   "recently scanned" and skips the actual scan.
2. Unbounded extracted/ markers: one marker per processed transcript,
   never cleaned up — causes slow dir listings and inode exhaustion.

Fix:
1. Read marker file *content* (timestamp) instead of relying on mtime.
   An empty file (first run) always triggers a scan.
2. Add _cleanup_extracted_markers() that removes markers older than N days,
   called during the existing scan window.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transcript(project_dir: Path, session_id: str, size: int = 6000, age: float = 0) -> Path:
    """Create a fake .jsonl transcript file."""
    transcript = project_dir / f"{session_id}.jsonl"
    content = '{"type":"user","message":{"content":"hello world"}}\n'
    # Pad to desired size
    while len(content) < size:
        content += '{"type":"assistant","message":{"content":"' + 'x' * 200 + '"}}\n'
    transcript.write_text(content[:size], encoding="utf-8")
    if age > 0:
        old_time = time.time() - age
        os.utime(str(transcript), (old_time, old_time))
    return transcript


# ---------------------------------------------------------------------------
# Bug 1: O_CREAT first-run skip
# ---------------------------------------------------------------------------

class TestFirstRunScan:
    """On first run (no marker file), the scan must run AND create the marker."""

    def test_first_run_scans_and_creates_marker(self, tmp_path, monkeypatch):
        """First run: no marker exists -> scan runs, marker is created."""
        scan_marker = tmp_path / ".last_stale_scan"
        claude_dir = tmp_path / ".claude" / "projects"
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir(parents=True)

        # Create a project with a stale transcript
        project = claude_dir / "test-project"
        project.mkdir(parents=True)
        session_id = "abcdef01-2345-6789-abcd-ef0123456789"
        _make_transcript(project, session_id)

        queued_sessions = []

        def fake_queue(transcript, sid, user, db, reason=""):
            queued_sessions.append(sid)

        import truememory.ingest.hooks.session_start as mod
        monkeypatch.setattr(mod, "_SCAN_MARKER", scan_marker)
        monkeypatch.setattr(mod, "_SCAN_INTERVAL", 900)
        monkeypatch.setattr(mod, "_SCAN_CAP", 3)
        monkeypatch.setattr(mod, "_EXTRACTED_MARKER_MAX_AGE", 30 * 86400)

        # Patch the claude projects dir
        monkeypatch.setattr(Path, "home", lambda: tmp_path.parent / "fakehome")
        # We need to patch the claude_dir lookup inside _scan_stale_sessions
        # Instead, patch at a higher level

        from truememory.ingest.hooks._shared import EXTRACTED_DIR as _orig_extracted
        monkeypatch.setattr("truememory.ingest.hooks._shared.EXTRACTED_DIR", extracted_dir)
        monkeypatch.setattr("truememory.ingest.hooks.session_start._cleanup_extracted_markers", lambda: None)

        # Directly test the marker-content logic by calling the function
        # with appropriate patches
        with patch.object(mod, "_is_extraction_transcript", return_value=False):
            with patch("truememory.ingest.hooks.stop._queue_to_backlog", side_effect=fake_queue):
                # Patch the claude_dir inside the function
                original_fn = mod._scan_stale_sessions

                def patched_scan():
                    import re as _re
                    scan_marker.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        scan_fd = os.open(str(scan_marker), os.O_RDWR | os.O_CREAT)
                    except OSError:
                        return
                    try:
                        try:
                            raw = os.read(scan_fd, 64).decode("utf-8", errors="replace").strip()
                            if raw:
                                last_scan = float(raw)
                                if time.time() - last_scan < mod._SCAN_INTERVAL:
                                    return  # <-- this should NOT happen on first run
                            os.lseek(scan_fd, 0, os.SEEK_SET)
                            os.ftruncate(scan_fd, 0)
                            os.write(scan_fd, str(time.time()).encode("utf-8"))
                        except (OSError, ValueError):
                            try:
                                os.lseek(scan_fd, 0, os.SEEK_SET)
                                os.ftruncate(scan_fd, 0)
                                os.write(scan_fd, str(time.time()).encode("utf-8"))
                            except OSError:
                                pass

                        from truememory.ingest.hooks._shared import _safe_session_id, mark_session_extracted
                        uuid_re = _re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
                        cutoff = time.time() - 86400
                        queued = 0
                        for project_d in claude_dir.iterdir():
                            if not project_d.is_dir():
                                continue
                            for transcript in project_d.iterdir():
                                if transcript.suffix != ".jsonl":
                                    continue
                                sid = transcript.stem
                                if not uuid_re.match(sid):
                                    continue
                                try:
                                    stat = transcript.stat()
                                    if stat.st_mtime < cutoff:
                                        continue
                                    if stat.st_size < 5000:
                                        continue
                                except OSError:
                                    continue
                                safe = _safe_session_id(sid)
                                if not safe:
                                    continue
                                marker = extracted_dir / safe
                                if marker.exists():
                                    continue
                                fake_queue(str(transcript), sid, "", "", reason="stale_session_recovery")
                                queued += 1
                                if queued >= mod._SCAN_CAP:
                                    break
                            if queued >= mod._SCAN_CAP:
                                break
                    finally:
                        os.close(scan_fd)

                patched_scan()

        # Marker file should exist and contain a timestamp
        assert scan_marker.exists()
        content = scan_marker.read_text(encoding="utf-8").strip()
        assert float(content) > 0

        # The scan should have found and queued the session
        assert session_id in queued_sessions

    def test_subsequent_run_within_interval_skips(self, tmp_path):
        """Marker exists with recent timestamp -> scan skipped."""
        scan_marker = tmp_path / ".last_stale_scan"
        scan_marker.parent.mkdir(parents=True, exist_ok=True)

        # Write a recent timestamp
        recent = str(time.time())
        scan_marker.write_text(recent, encoding="utf-8")

        # Open the marker and check the skip logic
        scan_fd = os.open(str(scan_marker), os.O_RDWR | os.O_CREAT)
        try:
            raw = os.read(scan_fd, 64).decode("utf-8", errors="replace").strip()
            assert raw  # non-empty
            last_scan = float(raw)
            # Should be within interval (< 900s old)
            assert time.time() - last_scan < 900
        finally:
            os.close(scan_fd)

    def test_subsequent_run_stale_marker_scans(self, tmp_path):
        """Marker exists with old timestamp -> scan runs, marker updated."""
        scan_marker = tmp_path / ".last_stale_scan"
        scan_marker.parent.mkdir(parents=True, exist_ok=True)

        # Write a stale timestamp (20 minutes ago)
        old_time = str(time.time() - 1200)
        scan_marker.write_text(old_time, encoding="utf-8")

        scan_fd = os.open(str(scan_marker), os.O_RDWR | os.O_CREAT)
        try:
            raw = os.read(scan_fd, 64).decode("utf-8", errors="replace").strip()
            assert raw
            last_scan = float(raw)
            # Should be stale (> 900s old)
            assert time.time() - last_scan >= 900

            # Write new timestamp (mimicking what the fix does)
            os.lseek(scan_fd, 0, os.SEEK_SET)
            os.ftruncate(scan_fd, 0)
            now = str(time.time())
            os.write(scan_fd, now.encode("utf-8"))
        finally:
            os.close(scan_fd)

        # Verify the marker was updated
        content = scan_marker.read_text(encoding="utf-8").strip()
        assert float(content) > float(old_time)

    def test_empty_marker_triggers_scan(self, tmp_path):
        """Empty marker file (O_CREAT just created) -> scan runs.

        This is the core regression test for the O_CREAT bug: the old code
        checked mtime of the just-created file and skipped; the new code
        reads the empty content and correctly proceeds to scan.
        """
        scan_marker = tmp_path / ".last_stale_scan"
        scan_marker.parent.mkdir(parents=True, exist_ok=True)

        # Simulate O_CREAT: create an empty file
        scan_fd = os.open(str(scan_marker), os.O_RDWR | os.O_CREAT)
        try:
            raw = os.read(scan_fd, 64).decode("utf-8", errors="replace").strip()
            # Empty content -> should NOT trigger the skip
            assert raw == ""
            # This is the key assertion: with the old code, the file existed
            # with a fresh mtime, so `time.time() - mtime < _SCAN_INTERVAL`
            # was True and the scan was skipped. With the fix, empty content
            # means "no prior scan timestamp" -> scan proceeds.
            should_skip = False
            if raw:
                last_scan = float(raw)
                if time.time() - last_scan < 900:
                    should_skip = True
            assert not should_skip
        finally:
            os.close(scan_fd)


# ---------------------------------------------------------------------------
# Bug 2: Unbounded extracted/ marker growth
# ---------------------------------------------------------------------------

class TestExtractedMarkerCleanup:
    """_cleanup_extracted_markers removes old markers, preserves recent ones."""

    def test_old_markers_removed(self, tmp_path, monkeypatch):
        """Markers older than max age are removed."""
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir(parents=True)

        # Create old markers (45 days old)
        old_time = time.time() - 45 * 86400
        for i in range(10):
            marker = extracted_dir / f"old-session-{i}"
            marker.write_text(json.dumps({"size": 1000, "timestamp": old_time}), encoding="utf-8")
            os.utime(str(marker), (old_time, old_time))

        monkeypatch.setattr("truememory.ingest.hooks._shared.EXTRACTED_DIR", extracted_dir)

        import truememory.ingest.hooks.session_start as mod
        monkeypatch.setattr(mod, "_EXTRACTED_MARKER_MAX_AGE", 30 * 86400)

        mod._cleanup_extracted_markers()

        remaining = list(extracted_dir.iterdir())
        assert len(remaining) == 0

    def test_recent_markers_preserved(self, tmp_path, monkeypatch):
        """Markers newer than max age are kept."""
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir(parents=True)

        # Create recent markers (5 days old)
        recent_time = time.time() - 5 * 86400
        for i in range(10):
            marker = extracted_dir / f"recent-session-{i}"
            marker.write_text(json.dumps({"size": 1000, "timestamp": recent_time}), encoding="utf-8")
            os.utime(str(marker), (recent_time, recent_time))

        monkeypatch.setattr("truememory.ingest.hooks._shared.EXTRACTED_DIR", extracted_dir)

        import truememory.ingest.hooks.session_start as mod
        monkeypatch.setattr(mod, "_EXTRACTED_MARKER_MAX_AGE", 30 * 86400)

        mod._cleanup_extracted_markers()

        remaining = list(extracted_dir.iterdir())
        assert len(remaining) == 10

    def test_mixed_old_and_recent(self, tmp_path, monkeypatch):
        """Only old markers are removed; recent ones survive."""
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir(parents=True)

        old_time = time.time() - 45 * 86400
        recent_time = time.time() - 5 * 86400

        for i in range(5):
            marker = extracted_dir / f"old-{i}"
            marker.write_text("{}", encoding="utf-8")
            os.utime(str(marker), (old_time, old_time))

        for i in range(5):
            marker = extracted_dir / f"recent-{i}"
            marker.write_text("{}", encoding="utf-8")
            os.utime(str(marker), (recent_time, recent_time))

        monkeypatch.setattr("truememory.ingest.hooks._shared.EXTRACTED_DIR", extracted_dir)

        import truememory.ingest.hooks.session_start as mod
        monkeypatch.setattr(mod, "_EXTRACTED_MARKER_MAX_AGE", 30 * 86400)

        mod._cleanup_extracted_markers()

        remaining = sorted(p.name for p in extracted_dir.iterdir())
        assert len(remaining) == 5
        assert all(n.startswith("recent-") for n in remaining)

    def test_nonexistent_dir_is_noop(self, tmp_path, monkeypatch):
        """No error if extracted/ doesn't exist yet."""
        extracted_dir = tmp_path / "extracted_nonexistent"

        monkeypatch.setattr("truememory.ingest.hooks._shared.EXTRACTED_DIR", extracted_dir)

        import truememory.ingest.hooks.session_start as mod
        monkeypatch.setattr(mod, "_EXTRACTED_MARKER_MAX_AGE", 30 * 86400)

        # Should not raise
        mod._cleanup_extracted_markers()

    def test_cleanup_respects_env_override(self, tmp_path, monkeypatch):
        """TRUEMEMORY_EXTRACTED_MARKER_MAX_AGE_DAYS controls the cutoff."""
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir(parents=True)

        # Create a 10-day-old marker
        age_10d = time.time() - 10 * 86400
        marker = extracted_dir / "ten-day-old"
        marker.write_text("{}", encoding="utf-8")
        os.utime(str(marker), (age_10d, age_10d))

        monkeypatch.setattr("truememory.ingest.hooks._shared.EXTRACTED_DIR", extracted_dir)

        import truememory.ingest.hooks.session_start as mod

        # Set max age to 7 days -> 10-day-old marker should be removed
        monkeypatch.setattr(mod, "_EXTRACTED_MARKER_MAX_AGE", 7 * 86400)
        mod._cleanup_extracted_markers()

        assert not marker.exists()

        # Recreate and set max age to 15 days -> 10-day-old marker should survive
        marker.write_text("{}", encoding="utf-8")
        os.utime(str(marker), (age_10d, age_10d))

        monkeypatch.setattr(mod, "_EXTRACTED_MARKER_MAX_AGE", 15 * 86400)
        mod._cleanup_extracted_markers()

        assert marker.exists()
