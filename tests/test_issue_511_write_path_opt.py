"""Regression tests for write-path optimizations (M14 #511, M15 #512, M16 #513).

M14: Triple conn.commit() per add() — personality and style_vec had their own commits
M15: compute_style_vector runs inside write lock — should be pre-computed outside
M16: Fallback embed_single inside write lock — removed in favor of skip-and-log
"""

import inspect
import unittest


class TestM14NoRedundantCommits(unittest.TestCase):
    """M14: personality and style_vec incremental functions must not commit."""

    def test_personality_incremental_no_commit(self):
        from truememory.personality import update_entity_profile_incremental
        source = inspect.getsource(update_entity_profile_incremental)
        self.assertNotIn("conn.commit()", source,
                         "update_entity_profile_incremental should not call conn.commit()")

    def test_style_vec_incremental_no_commit(self):
        from truememory.personality_style_vec import update_entity_style_vector_incremental
        source = inspect.getsource(update_entity_style_vector_incremental)
        self.assertNotIn("conn.commit()", source,
                         "update_entity_style_vector_incremental should not call conn.commit()")


class TestM15StyleVecOutsideLock(unittest.TestCase):
    """M15: compute_style_vector should run outside write lock."""

    def test_add_precomputes_style_vec(self):
        from truememory.engine import TrueMemoryEngine
        source = inspect.getsource(TrueMemoryEngine.add)
        lock_pos = source.find("self._write_lock")
        precompute_pos = source.find("compute_style_vector")
        self.assertGreater(precompute_pos, -1,
                           "add() should reference compute_style_vector for pre-computation")
        self.assertLess(precompute_pos, lock_pos,
                        "compute_style_vector call should appear before _write_lock")

    def test_style_vec_accepts_precomputed(self):
        from truememory.personality_style_vec import update_entity_style_vector_incremental
        source = inspect.getsource(update_entity_style_vector_incremental)
        self.assertIn("_pre_computed_vec", source,
                       "Should accept _pre_computed_vec parameter")


class TestM16NoFallbackEmbedInsideLock(unittest.TestCase):
    """M16: embed_single should not be called inside the write lock in add()."""

    def test_no_embed_single_in_add_lock_section(self):
        from truememory.engine import TrueMemoryEngine
        source = inspect.getsource(TrueMemoryEngine.add)
        lock_pos = source.find("self._write_lock")
        lock_section = source[lock_pos:]
        self.assertNotIn("embed_single", lock_section,
                         "embed_single should not appear inside the _write_lock section of add()")


if __name__ == "__main__":
    unittest.main()
