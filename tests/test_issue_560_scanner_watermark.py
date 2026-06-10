"""Tests for watermark-based stale session scanner (issue #560).

The scanner previously did an O(n) walk of every .jsonl file under
~/.claude/projects/. With the watermark optimization it only checks
files modified after the last scan timestamp, making it O(new).

Covers:
  - First scan (no watermark) uses 24-hour fallback
  - Old files (before watermark) are skipped
  - New files (after watermark) are scanned and queued
  - Watermark is updated after each scan
  - _read_scan_watermark handles empty / corrupt data
"""
from __future__ import annotations

import json
import os
import time
import uuid

import pytest

from truememory.ingest.hooks import session_start as ss


@pytest.fixture
def scanner_env(monkeypatch, tmp_path):
    """Set up isolated directories for scanner testing."""
    # Scan marker
    marker = tmp_path / "truememory" / ".last_stale_scan"
    marker.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ss, "_SCAN_MARKER", marker)

    # Claude projects dir (must match Path.home() / ".claude" / "projects")
    claude_dir = tmp_path / ".claude" / "projects"
    claude_dir.mkdir(parents=True)

    # Extracted dir
    extracted = tmp_path / "truememory" / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)

    # Backlog dir
    backlog = tmp_path / "truememory" / "backlog"
    backlog.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ss, "BACKLOG_DIR", backlog)

    # Patch Path.home() — setenv HOME so Path.home() returns tmp_path.
    # On Windows, Path.home() reads USERPROFILE (not HOME), so set both.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    # Patch _shared references used inside _scan_stale_sessions
    from truememory.ingest.hooks import _shared
    monkeypatch.setattr(_shared, "EXTRACTED_DIR", extracted)

    # Disable fcntl locking for test isolation
    monkeypatch.setattr(ss, "_HAS_FCNTL", False)

    return {
        "tmp_path": tmp_path,
        "marker": marker,
        "claude_dir": claude_dir,
        "extracted": extracted,
        "backlog": backlog,
    }


def _make_transcript(project_dir, session_id=None, mtime=None, size=6000):
    """Create a fake .jsonl transcript file."""
    if session_id is None:
        session_id = str(uuid.uuid4())
    path = project_dir / f"{session_id}.jsonl"
    # Write enough content to pass the 5000 byte minimum, with a non-extraction
    # first user message so it isn't flagged as noise.
    line = json.dumps({"type": "user", "message": {"content": "Hello world " + "x" * 200}})
    content = (line + "\n") * (size // len(line) + 1)
    path.write_text(content[:max(size, len(line) + 1)], encoding="utf-8")
    if mtime is not None:
        os.utime(str(path), (mtime, mtime))
    return session_id, path


class TestReadScanWatermark:
    """Unit tests for _read_scan_watermark."""

    def test_valid_timestamp(self, tmp_path):
        p = tmp_path / "marker"
        fd = os.open(str(p), os.O_RDWR | os.O_CREAT)
        try:
            ts = time.time() - 3600
            os.write(fd, str(ts).encode("utf-8"))
            result = ss._read_scan_watermark(fd)
            assert abs(result - ts) < 0.01
        finally:
            os.close(fd)

    def test_empty_file_returns_zero(self, tmp_path):
        p = tmp_path / "marker"
        fd = os.open(str(p), os.O_RDWR | os.O_CREAT)
        try:
            result = ss._read_scan_watermark(fd)
            assert result == 0.0
        finally:
            os.close(fd)

    def test_corrupt_content_returns_zero(self, tmp_path):
        p = tmp_path / "marker"
        fd = os.open(str(p), os.O_RDWR | os.O_CREAT)
        try:
            os.write(fd, b"not-a-number")
            result = ss._read_scan_watermark(fd)
            assert result == 0.0
        finally:
            os.close(fd)


class TestScannerWatermark:
    """Integration tests for watermark-based stale session scanning."""

    def test_first_scan_no_watermark_uses_24h_fallback(self, scanner_env, monkeypatch):
        """First scan with no prior watermark should scan files from the last 24h."""
        env = scanner_env
        proj = env["claude_dir"] / "test-project"
        proj.mkdir()

        now = time.time()
        # File from 12 hours ago — should be scanned (within 24h)
        sid_recent, _ = _make_transcript(proj, mtime=now - 43200)
        # File from 48 hours ago — should be skipped (older than 24h)
        sid_old, _ = _make_transcript(proj, mtime=now - 172800)

        # Patch _queue_to_backlog to track calls
        queued_sessions = []
        from truememory.ingest.hooks import stop as stop_mod
        monkeypatch.setattr(stop_mod, "_queue_to_backlog",
                            lambda tp, sid, u, d, reason="": queued_sessions.append(sid))

        # Ensure scan interval has passed (no marker exists yet)
        ss._scan_stale_sessions()

        assert sid_recent in queued_sessions
        assert sid_old not in queued_sessions

    def test_old_files_before_watermark_skipped(self, scanner_env, monkeypatch):
        """Files modified before the watermark timestamp should be skipped."""
        env = scanner_env
        proj = env["claude_dir"] / "test-project"
        proj.mkdir()

        now = time.time()
        watermark_time = now - 1800  # 30 minutes ago

        # Write a watermark from 30 minutes ago
        env["marker"].write_text(str(watermark_time), encoding="utf-8")
        # Set mtime old enough to pass the scan interval check
        os.utime(str(env["marker"]), (now - _scan_interval_plus(), now - _scan_interval_plus()))

        # File modified BEFORE watermark — should be skipped
        sid_old, _ = _make_transcript(proj, mtime=watermark_time - 600)
        # File modified AFTER watermark — should be scanned
        sid_new, _ = _make_transcript(proj, mtime=watermark_time + 300)

        queued_sessions = []
        from truememory.ingest.hooks import stop as stop_mod
        monkeypatch.setattr(stop_mod, "_queue_to_backlog",
                            lambda tp, sid, u, d, reason="": queued_sessions.append(sid))

        ss._scan_stale_sessions()

        assert sid_new in queued_sessions
        assert sid_old not in queued_sessions

    def test_watermark_updated_after_scan(self, scanner_env, monkeypatch):
        """After a scan, the marker file should contain the new watermark."""
        env = scanner_env

        # No marker file yet
        assert not env["marker"].exists()

        from truememory.ingest.hooks import stop as stop_mod
        monkeypatch.setattr(stop_mod, "_queue_to_backlog",
                            lambda tp, sid, u, d, reason="": None)

        before = time.time()
        ss._scan_stale_sessions()
        after = time.time()

        assert env["marker"].exists()
        watermark = float(env["marker"].read_text(encoding="utf-8").strip())
        assert before <= watermark <= after

    def test_new_files_after_watermark_scanned(self, scanner_env, monkeypatch):
        """Files modified after the watermark should be checked and queued."""
        env = scanner_env
        proj = env["claude_dir"] / "test-project"
        proj.mkdir()

        now = time.time()
        watermark_time = now - 1800

        env["marker"].write_text(str(watermark_time), encoding="utf-8")
        os.utime(str(env["marker"]), (now - _scan_interval_plus(), now - _scan_interval_plus()))

        # Create 3 new files after watermark
        sids = []
        for i in range(3):
            sid, _ = _make_transcript(proj, mtime=watermark_time + 60 * (i + 1))
            sids.append(sid)

        queued_sessions = []
        from truememory.ingest.hooks import stop as stop_mod
        monkeypatch.setattr(stop_mod, "_queue_to_backlog",
                            lambda tp, sid, u, d, reason="": queued_sessions.append(sid))

        ss._scan_stale_sessions()

        # All 3 should be queued (cap is 3)
        for sid in sids:
            assert sid in queued_sessions

    def test_already_extracted_sessions_skipped(self, scanner_env, monkeypatch):
        """Files with an extraction marker should still be skipped."""
        env = scanner_env
        proj = env["claude_dir"] / "test-project"
        proj.mkdir()

        now = time.time()
        watermark_time = now - 1800

        env["marker"].write_text(str(watermark_time), encoding="utf-8")
        os.utime(str(env["marker"]), (now - _scan_interval_plus(), now - _scan_interval_plus()))

        sid, _ = _make_transcript(proj, mtime=now - 60)
        # Create extraction marker
        (env["extracted"] / sid).write_text("done", encoding="utf-8")

        queued_sessions = []
        from truememory.ingest.hooks import stop as stop_mod
        monkeypatch.setattr(stop_mod, "_queue_to_backlog",
                            lambda tp, sid, u, d, reason="": queued_sessions.append(sid))

        ss._scan_stale_sessions()

        assert sid not in queued_sessions

    def test_scan_respects_interval(self, scanner_env, monkeypatch):
        """Scanner should not run if the marker mtime is within the interval."""
        env = scanner_env

        # Create marker with recent mtime (within interval)
        now = time.time()
        env["marker"].write_text(str(now), encoding="utf-8")
        # mtime is current — within _SCAN_INTERVAL

        proj = env["claude_dir"] / "test-project"
        proj.mkdir()
        sid, _ = _make_transcript(proj, mtime=now)

        queued_sessions = []
        from truememory.ingest.hooks import stop as stop_mod
        monkeypatch.setattr(stop_mod, "_queue_to_backlog",
                            lambda tp, sid, u, d, reason="": queued_sessions.append(sid))

        ss._scan_stale_sessions()

        # Should not have scanned (interval not elapsed)
        assert len(queued_sessions) == 0


def _scan_interval_plus():
    """Return a value larger than _SCAN_INTERVAL for mtime backdating."""
    return ss._SCAN_INTERVAL + 60
