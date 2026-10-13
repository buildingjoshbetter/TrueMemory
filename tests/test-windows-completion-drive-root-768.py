"""Windows completion directories refuse a missing drive root without recursing."""
from __future__ import annotations

import importlib.util
import os
import string
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("windows_completion_drive_root_768", ROOT / "truememory/ingest/completion.py")
assert SPEC is not None and SPEC.loader is not None
COMPLETION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COMPLETION
SPEC.loader.exec_module(COMPLETION)


@unittest.skipUnless(os.name == "nt", "drive roots exist only on Windows")
class MissingDriveRoot(unittest.TestCase):
    def test_missing_drive_root_is_refused_without_recursion(self) -> None:
        free = [letter for letter in reversed(string.ascii_uppercase) if not Path(f"{letter}:\\").exists()]
        if not free:
            self.skipTest("every drive letter is mapped")
        target = Path(f"{free[0]}:\\truememory-missing\\ingest-completion-v1")
        with self.assertRaises(COMPLETION.CompletionError) as caught:
            COMPLETION._private_directory(target)
        self.assertEqual(caught.exception.args, ("unsafe_directory",))
        self.assertFalse(Path(f"{free[0]}:\\").exists())


if __name__ == "__main__":
    unittest.main()
