"""Behavioral coverage for all maintenance layers and startup provenance."""

import runpy
from pathlib import Path
from unittest.mock import patch


_ROUTING = runpy.run_path(str(Path(__file__).with_name("test-maintenance-engine-routing-753.py")))


class TestConsolidationGaps(_ROUTING["RoutingFixture"]):
    def test_clustering_count_matches_real_published_centroids(self):
        self.add(5)
        self.native.fit = lambda: setattr(self.native, "labels", [0, 0, 1, 1, 2])
        result = self.engine.consolidate()
        self.assertIn("3 clusters", result["cluster_messages"])
        self.assertEqual(self.conn.execute("SELECT count(*) FROM cluster_centroids").fetchone()[0], 3)

    def test_preferences_runs_on_the_worker_connection(self):
        function = self.modules["truememory.personality"].extract_preferences
        with patch.object(self.modules["truememory.personality"], "extract_preferences", wraps=function) as preference:
            result = self.engine.consolidate()
        self.assertIn("extract_preferences", result)
        self.assertNotIn("ERROR", result["extract_preferences"])
        preference.assert_called_once()
        self.assertIsNot(preference.call_args.args[0], self.conn)

    def test_missing_vector_index_reports_unavailable_without_skipping_siblings(self):
        self.conn.execute("DROP TABLE vec_messages_edge")
        self.conn.commit()
        result = self.engine.consolidate()
        self.assertIn("UNAVAILABLE", result["cluster_messages"])
        self.assertIn("build_summaries", result)
        self.assertNotIn("ERROR", result["build_summaries"])

    def test_cluster_error_is_categorical_and_siblings_still_run(self):
        with patch.object(self.cluster, "cluster_messages", side_effect=RuntimeError("synthetic private sentinel")):
            result = self.engine.consolidate()
        self.assertEqual(result["cluster_messages"], "ERROR (RuntimeError)")
        self.assertNotIn("synthetic private sentinel", str(result))
        self.assertNotIn("ERROR", result["detect_episodes"])

    def test_startup_attempts_initial_small_corpus_once(self):
        self.add(5)
        self.engine._has_consolidation = True
        self.engine._maybe_startup_consolidate()
        self.completed()
        self.assertEqual(self.state().outcome, "success_empty")
        with patch.object(self.coordinator, "request_layers") as request:
            self.engine._maybe_startup_consolidate()
            request.assert_not_called()

    def test_successful_empty_clusters_do_not_retrigger_all_layers(self):
        self.add(30)
        self.engine.consolidate()
        self.assertEqual(self.conn.execute("SELECT count(*) FROM cluster_centroids").fetchone()[0], 0)
        self.engine._has_consolidation = True
        with patch.object(self.coordinator, "request_layers") as request:
            self.engine._maybe_startup_consolidate()
            request.assert_not_called()
