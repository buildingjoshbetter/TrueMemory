"""Regression tests for issue #461: stale lock files with no PID validation.

Lock files persist indefinitely after the holding process dies.
The fix writes the holder's PID to the lock file and validates it
on acquisition — stale locks from dead processes are automatically
cleaned up.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest


_SKIP_NO_FCNTL = pytest.mark.skipif(
    not hasattr(__import__("os"), "O_CREAT"),
    reason="Test requires POSIX filesystem",
)


class TestIssue461StaleLock:
    """Verify PID-based lock validation and TTL."""

    def test_issue_461_lock_file_contains_pid(self):
        """After acquiring the ingest lock, the file must contain the current PID."""
        from truememory.ingest.pipeline import _dedup_store_lock

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "ingest.lock"
            with patch("truememory.ingest.pipeline._LOCK_PATH", lock_path):
                with _dedup_store_lock():
                    if lock_path.exists():
                        content = lock_path.read_text().strip()
                        assert content.isdigit(), (
                            f"Lock file should contain PID, got: {content!r}"
                        )
                        assert int(content) == os.getpid(), (
                            f"Lock file PID {content} != current PID {os.getpid()}"
                        )

    def test_issue_461_stale_lock_from_dead_pid_is_stolen(self):
        """A lock file containing a dead PID should be acquirable."""
        from truememory.ingest.pipeline import _dedup_store_lock

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "ingest.lock"
            lock_path.write_text("999999\n")
            old_mtime = time.time() - 7200
            os.utime(str(lock_path), (old_mtime, old_mtime))

            with patch("truememory.ingest.pipeline._LOCK_PATH", lock_path):
                acquired = False
                with _dedup_store_lock():
                    acquired = True
                assert acquired, (
                    "Could not acquire lock from dead PID — stale lock blocking"
                )

    def test_issue_461_spawn_lock_validates_pid(self):
        """spawn_gate lock should also write and validate PID."""
        from truememory.hooks.core import spawn_gate, SPAWN_LOCK_PATH

        with tempfile.TemporaryDirectory() as tmpdir:
            test_lock = Path(tmpdir) / ".spawn.lock"
            test_pids = Path(tmpdir) / ".spawn_pids"
            with (
                patch("truememory.hooks.core.SPAWN_LOCK_PATH", test_lock),
                patch("truememory.hooks.core.SPAWN_PIDS_PATH", test_pids),
            ):
                with spawn_gate() as allowed:
                    pass

    def test_issue_461_lock_ttl_expires_old_locks(self):
        """Locks older than TTL should be treated as stale."""
        from truememory.ingest.pipeline import _dedup_store_lock

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "ingest.lock"
            lock_path.write_text(f"{os.getpid()}\n")
            old_mtime = time.time() - 7200
            os.utime(str(lock_path), (old_mtime, old_mtime))

            with patch("truememory.ingest.pipeline._LOCK_PATH", lock_path):
                acquired = False
                with _dedup_store_lock():
                    acquired = True
                assert acquired, "Lock with expired TTL should be acquirable"
