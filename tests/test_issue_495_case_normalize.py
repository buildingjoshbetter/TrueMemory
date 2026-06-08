"""Regression tests for H9 #495: PR #467 case normalization incomplete.

Gaps: recipients in personality profiles, build_dunbar_hierarchy(),
and summary entity normalization all missed .lower().
"""

import inspect
import unittest


class TestRecipientNormalization(unittest.TestCase):
    """Recipient keys in relationship tracking must be lowercased."""

    def test_incremental_profile_lowercases_recipient(self):
        from truememory.personality import update_entity_profile_incremental
        source = inspect.getsource(update_entity_profile_incremental)
        rel_section = source[source.find("Relationships"):]
        self.assertIn("recipient.lower()", rel_section,
                       "Recipient should be lowercased in relationship tracking")

    def test_recipient_key_is_lowercase(self):
        from truememory.personality import update_entity_profile_incremental
        source = inspect.getsource(update_entity_profile_incremental)
        self.assertIn("recipient_key", source,
                       "Should use a lowercased recipient_key variable")


class TestDunbarHierarchyNormalization(unittest.TestCase):
    """build_dunbar_hierarchy must normalize entity names."""

    def test_primary_entity_lowercased(self):
        from truememory.personality import build_dunbar_hierarchy
        source = inspect.getsource(build_dunbar_hierarchy)
        self.assertIn("primary_entity.lower()", source,
                       "primary_entity should be lowercased")

    def test_contact_names_lowercased(self):
        from truememory.personality import build_dunbar_hierarchy
        source = inspect.getsource(build_dunbar_hierarchy)
        self.assertIn("LOWER(name)", source,
                       "Contact names should be lowercased in SQL query")


class TestSummaryEntityNormalization(unittest.TestCase):
    """Summary entity column must use lowercase sender names."""

    def test_summary_entity_is_lowercase(self):
        from truememory.consolidation import build_summaries
        source = inspect.getsource(build_summaries)
        entity_grouping = source[source.find("Per-entity summaries"):]
        self.assertIn('.lower()', entity_grouping[:200],
                       "Entity grouping should use sender.lower()")


if __name__ == "__main__":
    unittest.main()
