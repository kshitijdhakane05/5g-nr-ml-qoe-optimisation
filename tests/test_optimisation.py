"""Validation tests for Phase 2 offline ML-based network optimiser."""

import unittest
import numpy as np
import pandas as pd

from ml.predict import format_input_features, load_model, predict_qoe
from optimisation.search_space import (
    SUPPORTED_SEARCH_SPACE,
    generate_candidate_configurations,
    validate_network_context,
)
from optimisation.optimizer import NetworkOptimiser


class TestNetworkOptimiser(unittest.TestCase):

    def test_feature_transformation_compatibility(self):
        """Verify that features produced for candidates strictly match the Phase 1 model schema."""
        bundle = load_model("models/qoe_model.joblib")
        expected_cols = bundle["feature_columns"]

        # Generate sample input
        df_feat = format_input_features(
            ue_count=4,
            bandwidth_mhz=50.0,
            total_tx_power_dbm=35.0,
            scenario_size_m=30.0,
            flow_type="voice_be",
            lambda_val=10000.0,
        )

        self.assertEqual(
            list(df_feat.columns),
            expected_cols,
            f"Feature mismatch: {list(df_feat.columns)} vs {expected_cols}",
        )

        # Confirm model can predict without schema or type warnings
        pred = predict_qoe(bundle, df_feat)
        self.assertEqual(len(pred), 1)
        self.assertTrue(0.0 <= pred[0] <= 1.0)

    def test_candidate_generation_and_search_space_validity(self):
        """Verify candidates strictly adhere to observed training support."""
        candidates = generate_candidate_configurations(
            ue_count=4,
            scenario_size_m=30.0,
            flow_type="voice_be",
            lambda_val=10000.0,
        )

        self.assertEqual(len(candidates), 2)  # 2 bandwidths (20, 50) x 1 power (35)
        for c in candidates:
            self.assertIn(c["ue_count"], SUPPORTED_SEARCH_SPACE["ue_count"])
            self.assertIn(c["bandwidth_mhz"], SUPPORTED_SEARCH_SPACE["bandwidth_mhz"])
            self.assertIn(c["total_tx_power_dbm"], SUPPORTED_SEARCH_SPACE["total_tx_power_dbm"])
            self.assertIn(c["scenario_size_m"], SUPPORTED_SEARCH_SPACE["scenario_size_m"])

    def test_no_extrapolation_on_invalid_context(self):
        """Verify that invalid inputs outside training support raise ValueError rather than extrapolate."""
        with self.assertRaises(ValueError):
            generate_candidate_configurations(ue_count=99, scenario_size_m=30.0)

        with self.assertRaises(ValueError):
            generate_candidate_configurations(ue_count=4, scenario_size_m=500.0)

        with self.assertRaises(ValueError):
            generate_candidate_configurations(ue_count=4, flow_type="unknown_traffic")

    def test_constraint_handling(self):
        """Verify constraint checker flags violations correctly."""
        optimiser = NetworkOptimiser("models/qoe_model.joblib")

        cand = {
            "ue_count": 4,
            "bandwidth_mhz": 20.0,
            "total_tx_power_dbm": 35.0,
        }

        # Case 1: Max power constraint violated
        satisfied, violations = optimiser.check_constraints(
            cand, predicted_qoe=0.8, constraints={"max_power_dbm": 30.0}
        )
        self.assertFalse(satisfied)
        self.assertTrue(any("power" in v for v in violations))

        # Case 2: Min QoE constraint violated
        satisfied, violations = optimiser.check_constraints(
            cand, predicted_qoe=0.0, constraints={"min_predicted_qoe": 0.5}
        )
        self.assertFalse(satisfied)
        self.assertTrue(any("minimum" in v for v in violations))

        # Case 3: All satisfied
        satisfied, violations = optimiser.check_constraints(
            cand, predicted_qoe=0.7, constraints={"max_power_dbm": 40.0, "min_predicted_qoe": 0.5}
        )
        self.assertTrue(satisfied)
        self.assertEqual(len(violations), 0)

    def test_optimiser_selection_and_baseline_comparison(self):
        """Verify end-to-end optimisation logic and baseline comparison."""
        optimiser = NetworkOptimiser("models/qoe_model.joblib")

        # When baseline is 20 MHz (outage condition), optimiser should select 50 MHz
        res = optimiser.optimise(
            ue_count=4,
            scenario_size_m=30.0,
            flow_type="voice_be",
            lambda_val=10000.0,
            baseline_bandwidth_mhz=20.0,
            baseline_power_dbm=35.0,
        )

        base = res["baseline"]
        best = res["best_configuration"]

        # Baseline QoE at 20 MHz is 0.0 (outage)
        self.assertEqual(base["predicted_qoe"], 0.0)

        # Best configuration must be 50 MHz
        self.assertEqual(best["bandwidth_mhz"], 50.0)
        self.assertGreater(best["predicted_qoe"], 0.80)

        # Improvement must be positive
        self.assertGreater(res["predicted_qoe_improvement"], 0.80)
        self.assertEqual(
            res["predicted_improvement_label"],
            "Predicted improvement — pending ns-3 verification",
        )

    def test_infeasible_constraint_behavior(self):
        """Verify behavior when constraint cannot be satisfied by any candidate."""
        optimiser = NetworkOptimiser("models/qoe_model.joblib")

        # Demand impossible QoE > 0.999
        res = optimiser.optimise(
            ue_count=8,
            scenario_size_m=100.0,
            flow_type="voice_be",
            lambda_val=40000.0,
            constraints={"min_predicted_qoe": 0.999},
        )

        self.assertEqual(res["selection_status"], "NO_FEASIBLE_CANDIDATE_HIGHEST_QOE_RETURNED")
        self.assertFalse(res["best_configuration"]["feasible"])


if __name__ == "__main__":
    unittest.main()

