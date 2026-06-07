"""Regression test for M4 #501: delete_all summaries cleanup not case-normalized.

Entity profiles, style vectors, and relationships all use .lower() for
delete cleanup but summaries did not, leaving orphaned summary rows.
"""

import sqlite3
import unittest


class TestDeleteAllSummaryCaseNormalization(unittest.TestCase):

    def _make_engine(self):
        from truememory.engine import TrueMemoryEngine
        import tempfile, os
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "test.db")
        e = TrueMemoryEngine(db_path=db_path)
        e._ensure_connection()
        return e, tmp

    def test_delete_all_user_cleans_summaries_case_insensitive(self):
        e, tmp = self._make_engine()
        e.add("test memory", sender="Alice")
        try:
            e.conn.execute(
                "INSERT INTO summaries (entity, summary, updated_at) VALUES (?, ?, ?)",
                ("alice", "Alice is a test user", "2026-01-01"),
            )
            e.conn.commit()
        except Exception:
            self.skipTest("summaries table not available")
        e.delete_all(user_id="Alice")
        rows = e.conn.execute(
            "SELECT * FROM summaries WHERE entity = ?", ("alice",)
        ).fetchall()
        self.assertEqual(len(rows), 0, "Summaries for 'alice' should be deleted when user_id='Alice'")

    def test_delete_all_user_source_uses_lower(self):
        """Verify the source code normalizes user_id for summaries delete."""
        import inspect
        from truememory.engine import TrueMemoryEngine
        source = inspect.getsource(TrueMemoryEngine.delete_all)
        summaries_idx = source.find("DELETE FROM summaries WHERE entity")
        self.assertGreater(summaries_idx, -1, "Should have summaries delete")
        nearby = source[summaries_idx:summaries_idx + 100]
        self.assertIn(".lower()", nearby,
                       "summaries delete should use user_id.lower()")


if __name__ == "__main__":
    unittest.main()
