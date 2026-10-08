"""Durable automatic maintenance thresholds across engine instances."""

import runpy
from pathlib import Path
from unittest.mock import patch


_ROUTING = runpy.run_path(str(Path(__file__).with_name("test-maintenance-engine-routing-753.py")))


class TestAutoConsolidation(_ROUTING["RoutingFixture"]):
    def test_default_threshold_is_twenty_five(self):
        self.assertEqual(self.engine._auto_consolidate_threshold, 25)

    def test_configured_threshold_uses_committed_source_inserts(self):
        self.engine.consolidate()
        self.engine._auto_consolidate_threshold = 3
        self.engine._has_consolidation = True
        with patch.object(self.coordinator, "request_layers", wraps=self.coordinator.request_layers) as request:
            for _ in range(2):
                self.engine.add("synthetic committed append")
            request.assert_not_called()
            self.engine.add("synthetic threshold append")
            self.completed()
            self.assertTrue(request.called)
            self.assertTrue(all(call.kwargs["threshold"] == 3 for call in request.call_args_list))

    def test_close_and_reopen_does_not_reset_threshold_progress(self):
        self.engine.consolidate()
        self.add(24)
        self.engine.close()
        reopened = self.new_engine()
        reopened._has_consolidation = True
        with patch.object(self.coordinator, "request_layers", wraps=self.coordinator.request_layers) as request:
            reopened._maybe_auto_consolidate()
            request.assert_not_called()
            reopened.add("synthetic twenty fifth")
            self.completed()
            self.assertTrue(request.called)
