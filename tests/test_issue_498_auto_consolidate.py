"""Regression test for M1 #498: auto-consolidation trigger.

Consolidation should run automatically after a configurable number of
add() calls, not require manual truememory_consolidate invocation.
"""

import inspect
import unittest


class TestAutoConsolidation(unittest.TestCase):

    def test_auto_consolidate_threshold_exists(self):
        from truememory.engine import TrueMemoryEngine
        source = inspect.getsource(TrueMemoryEngine.__init__)
        self.assertIn("_auto_consolidate_threshold", source)
        self.assertIn("TRUEMEMORY_AUTO_CONSOLIDATE_EVERY", source)

    def test_add_calls_maybe_auto_consolidate(self):
        from truememory.engine import TrueMemoryEngine
        source = inspect.getsource(TrueMemoryEngine.add)
        self.assertIn("_maybe_auto_consolidate", source,
                       "add() should call _maybe_auto_consolidate()")

    def test_maybe_auto_consolidate_checks_threshold(self):
        from truememory.engine import TrueMemoryEngine
        source = inspect.getsource(TrueMemoryEngine._maybe_auto_consolidate)
        self.assertIn("_auto_consolidate_threshold", source)
        self.assertIn("_has_consolidation", source)

    def test_bg_consolidate_exists(self):
        from truememory.engine import TrueMemoryEngine
        self.assertTrue(hasattr(TrueMemoryEngine, "_bg_consolidate"))


if __name__ == "__main__":
    unittest.main()
