"""Regression tests for M13 #510 and M22 #519: schema consolidation.

M13: surprise_scores table not in base schema (storage.py)
M22: Schema fragmented across 4+ files — tables should be in storage.py DDL
"""

import unittest


class TestSchemaConsolidation(unittest.TestCase):

    def test_surprise_scores_in_base_schema(self):
        from truememory import storage
        self.assertIn("surprise_scores", storage._SCHEMA_SQL,
                       "surprise_scores DDL should be in storage.py _SCHEMA")

    def test_message_clusters_in_base_schema(self):
        from truememory import storage
        self.assertIn("message_clusters", storage._SCHEMA_SQL,
                       "message_clusters DDL should be in storage.py _SCHEMA")

    def test_cluster_centroids_in_base_schema(self):
        from truememory import storage
        self.assertIn("cluster_centroids", storage._SCHEMA_SQL,
                       "cluster_centroids DDL should be in storage.py _SCHEMA")

    def test_tables_created_on_create_db(self):
        from truememory.storage import create_db
        import sqlite3
        conn = create_db(":memory:")
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for expected in ("surprise_scores", "message_clusters", "cluster_centroids"):
            self.assertIn(expected, tables,
                          f"{expected} should be created by create_db()")
        conn.close()


if __name__ == "__main__":
    unittest.main()
