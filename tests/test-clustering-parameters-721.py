"""Parameter controls and synthetic HDBSCAN sensitivity, without memory data."""
from __future__ import annotations

import ast
import math
import runpy
import sqlite3
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def builder() -> object:
    path = ROOT / "truememory/clustering.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "cluster_messages")
    namespace: dict[str, object] = {"sqlite3": sqlite3}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["cluster_messages"]


class ClusterParameterControls(unittest.TestCase):
    def test_invalid_parameters_fail_before_import_lock_or_database_access(self) -> None:
        call = builder()
        invalid = [
            {"min_cluster_size": value} for value in [1, 0, -1, True, 2.5, "3", None, 2147483648]
        ] + [
            {"min_samples": value} for value in [0, -1, False, 2.5, "2", None, 2147483648]
        ] + [
            {"cluster_selection_method": value} for value in ["", "LEAF", "unknown", True, None, []]
        ]
        with patch.dict(sys.modules, {"hdbscan": None, "truememory": None}):
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    call(object(), **kwargs)

    def test_default_values_and_keyword_only_method(self) -> None:
        import inspect
        signature = inspect.signature(builder())
        self.assertEqual(signature.parameters["min_cluster_size"].default, 10)
        self.assertEqual(signature.parameters["min_samples"].default, 5)
        self.assertEqual(signature.parameters["cluster_selection_method"].default, "eom")
        self.assertEqual(signature.parameters["cluster_selection_method"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_valid_parameters_continue_to_original_dependency_boundary(self) -> None:
        call = builder()
        with patch.dict(sys.modules, {"hdbscan": None}):
            for method in ["eom", "leaf"]:
                with self.subTest(method=method), self.assertRaises(ModuleNotFoundError):
                    call(object(), 2, 1, cluster_selection_method=method)

    def test_actual_constructor_preserves_metric_and_forwards_explicit_leaf(self) -> None:
        harness = runpy.run_path(str(ROOT / "tests/test-cluster-publication-749.py"))
        module, vector, boundary = harness["load_clustering"]()
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(harness["SCHEMA"] + module._CLUSTER_SCHEMA)
            for mid, values in enumerate([(1, 0), (0, 1), (0, 0), (2, 2), (4, 0)], 1):
                conn.execute("INSERT INTO messages VALUES (?, ?, 'source', '', '2020-01-01', '', 'text')", (mid, "synthetic"))
                conn.execute("INSERT INTO vec_messages_edge(rowid, embedding) VALUES (?, ?)", (mid, struct.pack("2f", *values)))
            conn.execute("INSERT INTO metadata VALUES ('embed_model', 'synthetic-model-a', '')")
            conn.execute("INSERT INTO metadata VALUES ('embed_dim', '2', '')")
            conn.execute("INSERT INTO vector_cache_registry VALUES ('edge', 'vec_messages_edge', 'vec_messages_sep_edge', 5, 5, 'synthetic-model-a', 2, 1.0, 1.0)")
            conn.commit()
            self.assertEqual(module.cluster_messages(conn, 3, 2, cluster_selection_method="leaf"), 2)
            self.assertEqual(boundary.parameters, [{
                "min_cluster_size": 3, "min_samples": 2, "metric": "euclidean", "cluster_selection_method": "leaf",
            }])
            self.assertFalse(conn.in_transaction)
            self.assertFalse(vector._lock.locked())

    def test_scheduled_default_parameter_identity_is_preserved(self) -> None:
        parsed = ast.parse((ROOT / "truememory/maintenance.py").read_text(encoding="utf-8"))
        function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "_cluster_schedule_dependency")
        assignment = next(node for node in function.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "parameters" for target in node.targets))
        self.assertEqual(ast.literal_eval(assignment.value), {
            "min_cluster_size": 10, "min_samples": 5, "metric": "euclidean", "cluster_selection_method": "eom",
        })


class SyntheticHdbscanSensitivity(unittest.TestCase):
    def test_actual_backend_is_deterministic_across_explicit_policies(self) -> None:
        import hdbscan
        import numpy as np

        rng = np.random.default_rng(721)
        angles = []
        for parent in [0.0, 2 * math.pi / 3, 4 * math.pi / 3]:
            for offset in [-0.12, 0.0, 0.12]:
                angles.extend(parent + offset + rng.normal(0.0, 0.008, 15))
        values = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
        observations = []
        for size, samples, method in [(10, 5, "eom"), (5, 3, "leaf")]:
            kwargs = {
                "min_cluster_size": size, "min_samples": samples,
                "metric": "euclidean", "cluster_selection_method": method,
                "core_dist_n_jobs": 1,
            }
            first = hdbscan.HDBSCAN(**kwargs).fit_predict(values)
            second = hdbscan.HDBSCAN(**kwargs).fit_predict(values)
            self.assertEqual(first.tolist(), second.tolist())
            self.assertEqual(len(first), 135)
            self.assertTrue(np.isfinite(first).all())
            groups = len(set(first.tolist()) - {-1})
            self.assertGreaterEqual(groups, 3)
            self.assertLessEqual(groups, 27)
            observations.append(groups)
        self.assertGreaterEqual(observations[1], observations[0])


if __name__ == "__main__":
    if "--stdlib-controls" in sys.argv:
        sys.argv.remove("--stdlib-controls")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ClusterParameterControls)
        outcome = unittest.TextTestRunner().run(suite)
        raise SystemExit(0 if outcome.wasSuccessful() else 1)
    unittest.main()
