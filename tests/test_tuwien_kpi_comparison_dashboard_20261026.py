"""Tests for the offline TU Wien comparison dashboard and benchmark labels."""

from __future__ import annotations

import unittest

from experiments.tuwien_kpi_baseline_20261026.baseline import (
    DEFAULT_DATA_DIR,
    EXPECTED_GROUPS,
    load_trials,
)
from experiments.tuwien_kpi_comparison_dashboard_20261026.build_dashboard import (
    EXPERIMENT_DIR,
    build_comparison,
    rate_only_prediction,
)


class TestTuwienComparisonDashboard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.comparison = build_comparison()
        cls.rows = load_trials()

    def test_rate_only_is_primary_and_stronger_than_ridge(self):
        self.assertEqual(len(self.comparison["targets"]), 4)
        for target, metrics in self.comparison["targets"].items():
            with self.subTest(target=target):
                self.assertEqual(metrics["primary_benchmark"], "rate_only")
                self.assertTrue(metrics["rate_only_beats_ridge_mae"])
                self.assertLess(metrics["rate_only"]["mae"], metrics["ridge"]["mae"])

    def test_rate_only_prediction_ignores_held_out_group_targets(self):
        metadata = __import__("json").loads(
            (EXPERIMENT_DIR / "run_metadata.json").read_text(encoding="utf-8")
        )
        assignment = {k: int(v) for k, v in metadata["validation"]["group_assignment"].items()}
        target_group = next(group for group, fold in assignment.items() if fold == 0)
        q_text, rate_text = target_group.split("_rate")
        rate = int(rate_text)
        before = rate_only_prediction(
            self.rows, assignment, test_fold=0, target="throughput", configured_rate=rate
        )
        changed = [dict(row) for row in self.rows]
        for row in changed:
            if assignment[row["group_id"]] == 0 and row["target_Mbps"] == rate:
                row["throughput"] = 1e12
        after = rate_only_prediction(
            changed, assignment, test_fold=0, target="throughput", configured_rate=rate
        )
        self.assertEqual(before, after)
        self.assertEqual(len(assignment), EXPECTED_GROUPS)

    def test_negative_ridge_outputs_are_counted_invalid_and_retained(self):
        for target, expected_count in (("jitter", 35), ("packet_loss", 10)):
            with self.subTest(target=target):
                metrics = self.comparison["targets"][target]
                self.assertEqual(metrics["ridge_negative_prediction_count"], expected_count)
                self.assertTrue(metrics["ridge_negative_values_retained"])
                invalid_rows = [
                    row for row in self.comparison["predictions"]
                    if row["target"] == target and row["ridge_prediction_raw"] < 0
                ]
                self.assertEqual(len(invalid_rows), expected_count)
                self.assertTrue(all(row["ridge_physical_status"] == "INVALID_NEGATIVE" for row in invalid_rows))
                self.assertTrue(all(row["ridge_prediction_raw"] < 0 for row in invalid_rows))

    def test_dashboard_warns_and_does_not_offer_arbitrary_predictions(self):
        from experiments.tuwien_kpi_comparison_dashboard_20261026.build_dashboard import _html_document

        page = _html_document(self.comparison)
        self.assertIn("No arbitrary-input prediction is shown", page)
        self.assertIn("INVALID_NEGATIVE", page)
        self.assertIn("none are clipped", page)


if __name__ == "__main__":
    unittest.main()
