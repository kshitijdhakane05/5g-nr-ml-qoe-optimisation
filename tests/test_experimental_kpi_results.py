"""Focused checks for the isolated offline KPI dashboard view."""

import json
import ast
import tempfile
import unittest
from pathlib import Path

from dashboard import experimental_kpi_results as view


class ExperimentalKpiResultsTests(unittest.TestCase):
    def test_saved_results_validate_and_use_documented_units(self):
        metrics, rows = view.load_saved_results()
        self.assertTrue(rows)
        self.assertEqual(view.TARGETS["throughput"]["unit"], "Mbps")
        self.assertEqual(view.TARGETS["latency"]["unit"], "ms")
        self.assertEqual(view.TARGETS["jitter"]["unit"], "ms")
        self.assertEqual(view.TARGETS["packet_loss"]["unit"], "% (percentage points)")
        self.assertEqual(metrics["targets"]["packet_loss"]["unit"], "percentage points")

    def test_rate_only_is_primary_and_ridge_invalid_outputs_are_reported(self):
        metrics, _ = view.load_saved_results()
        for item in metrics["targets"].values():
            self.assertEqual(item["primary_benchmark"], "rate_only")
            self.assertTrue(item["rate_only_beats_ridge_mae"])
        self.assertEqual(metrics["targets"]["jitter"]["ridge_negative_prediction_count"], 35)
        self.assertEqual(metrics["targets"]["packet_loss"]["ridge_negative_prediction_count"], 10)

    def test_offline_filter_fields_are_not_target_or_model_inputs(self):
        self.assertEqual(view.OFFLINE_FILTER_FIELDS, ("target", "configured_rate_Mbps", "qIdx"))
        self.assertNotIn("observed", view.OFFLINE_FILTER_FIELDS)
        self.assertNotIn("rate_only_prediction", view.OFFLINE_FILTER_FIELDS)
        self.assertNotIn("trial", view.OFFLINE_FILTER_FIELDS)

    def test_missing_artifact_is_reported_without_model_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                view.load_saved_results(root / "missing.json", root / "missing.csv")

    def test_module_is_separate_from_production_qoe_and_simulation_path(self):
        source = Path(view.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        self.assertFalse(any(name.startswith(("ml", "optimisation", "simulation")) for name in imported))

    def test_artifact_schema_mismatch_is_rejected(self):
        metrics_path = view.METRICS_PATH
        predictions_path = view.PREDICTIONS_PATH
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["targets"]["throughput"]["unit"] = "packets/s"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaises(ValueError):
                view.load_saved_results(path, predictions_path)


if __name__ == "__main__":
    unittest.main()
