"""Every try_file_lock holder releases through _platform.release_file_lock.

Windows releases a msvcrt region lock on close only at an unspecified later
time, so the helper unlocks first; POSIX unlocks the shared open file
description explicitly too.
"""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from truememory import _platform
import truememory.maintenance as maintenance
from truememory.tier_switch import job


class ReleaseFileLock(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "store.db"

    def recorder(self, real):
        released = []

        def record(fd: int) -> None:
            released.append(fd)
            real(fd)
        return released, record

    def test_windows_unlocks_the_region_before_closing(self) -> None:
        calls = []
        fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o600)
        msvcrt = types.SimpleNamespace(LK_UNLCK=0, locking=lambda descriptor, mode, size: calls.append(
            (descriptor, mode, size, os.lseek(descriptor, 0, os.SEEK_CUR))))
        with patch.object(_platform, "sys", types.SimpleNamespace(platform="win32")), \
                patch.dict(sys.modules, {"msvcrt": msvcrt}):
            _platform.release_file_lock(fd)
        self.assertEqual(calls, [(fd, 0, 1, 0)])  # Byte 0, the region try_file_lock locked.
        with self.assertRaises(OSError):
            os.fstat(fd)  # Closed only after the unlock.

    @unittest.skipIf(sys.platform == "win32", "flock semantics")
    def test_posix_release_frees_a_lock_another_copy_still_references(self) -> None:
        fd = _platform.try_file_lock(self.path)
        copy = os.dup(fd)  # For example a descriptor a forked child has not closed yet.
        self.addCleanup(os.close, copy)
        _platform.release_file_lock(fd)
        other = _platform.try_file_lock(self.path)
        self.assertIsNotNone(other)
        _platform.release_file_lock(other)

    def test_maintenance_owner_wanted_lock_and_probes_release_through_the_helper(self) -> None:
        released, record = self.recorder(maintenance.release_file_lock)
        with patch.object(maintenance, "release_file_lock", record):
            with maintenance.maintenance_owner(self.path):
                pass
            with maintenance.maintenance_wanted(self.path) as wanted:
                self.assertTrue(wanted)
            self.assertFalse(maintenance.maintenance_handoff_pending(self.path))
            with maintenance.maintenance_owner(self.path):
                pass
        # Owner, wanted lock, handoff probe of the wanted file, then probe and owner.
        self.assertEqual(len(released), 5)

    def test_job_and_activation_locks_release_through_the_helper(self) -> None:
        released, record = self.recorder(_platform.release_file_lock)
        with patch.object(_platform, "release_file_lock", record):
            lock = job.acquire_tier_job_lock(self.path)
            lock.release()
            activation = job.acquire_tier_activation_lock(self.path, timeout=0)
            activation.release()
            self.assertFalse(job.tier_activation_in_progress(self.path))
        self.assertEqual(len(released), 3)


if __name__ == "__main__":
    unittest.main()
