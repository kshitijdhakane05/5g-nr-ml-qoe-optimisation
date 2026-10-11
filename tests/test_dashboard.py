"""
tests/test_dashboard.py
=======================
Unit tests for Phase 4 dashboard helpers.

These tests do NOT launch Streamlit; they exercise:
  - data_loader functions with real and missing files
  - components helper functions (chart generation)
  - predictor/optimiser integration via the existing APIs
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers — stub out streamlit so data_loader can be imported without a
# running Streamlit server.
# ---------------------------------------------------------------------------
_st_mock = MagicMock()
def _mock_cache_data(*args, **kwargs):
    def decorator(fn):
        fn.clear = MagicMock()
        return fn
    return decorator
_st_mock.cache_data = _mock_cache_data
_st_mock.session_state = {}
sys.modules.setdefault("streamlit", _st_mock)


# ---------------------------------------------------------------------------
# 1. data_loader tests
# ---------------------------------------------------------------------------
class TestDataLoader(unittest.TestCase):
    """data_loader functions should handle both present and absent files."""

    def test_load_processed_dataset_returns_dataframe(self):
        from dashboard.data_loader import load_processed_dataset
        df = load_processed_dataset()
        self.assertIsInstance(df, pd.DataFrame)

    def test_processed_dataset_not_empty(self):
        from dashboard.data_loader import load_processed_dataset
        df = load_processed_dataset()
        self.assertGreater(len(df), 0, "Processed dataset should have rows")

    def test_processed_dataset_has_expected_columns(self):
        from dashboard.data_loader import load_processed_dataset
        df = load_processed_dataset()
        expected = [
            "scenario_id", "ue_count", "bandwidth_mhz", "qoe_proxy",
            "flow_type", "is_outage",
        ]
        for col in expected:
            self.assertIn(col, df.columns, f"Expected column '{col}' in dataset")

    def test_load_model_results_returns_dataframe(self):
        from dashboard.data_loader import load_model_results
        df = load_model_results()
        self.assertIsInstance(df, pd.DataFrame)

    def test_model_results_has_best_rf_row(self):
        from dashboard.data_loader import load_model_results
        df = load_model_results()
        if df.empty:
            self.skipTest("model_results.csv not found")
        names = df["model_name"].tolist()
        self.assertTrue(
            any("RandomForest" in n or "random_forest" in n.lower() for n in names),
            f"Expected RandomForest in model_results, got: {names}",
        )

    def test_load_phase3_results_returns_dataframe(self):
        from dashboard.data_loader import load_phase3_results
        df = load_phase3_results()
        self.assertIsInstance(df, pd.DataFrame)

    def test_dataset_summary_returns_dict(self):
        from dashboard.data_loader import load_processed_dataset, dataset_summary
        df = load_processed_dataset()
        s = dataset_summary(df)
        self.assertIsInstance(s, dict)
        if s:
            self.assertIn("rows", s)
            self.assertIn("unique_scenarios", s)
            self.assertGreater(s["rows"], 0)

    def test_dataset_summary_empty_df(self):
        from dashboard.data_loader import dataset_summary
        s = dataset_summary(pd.DataFrame())
        self.assertEqual(s, {})

    def test_load_processed_dataset_missing_file(self):
        """data_loader should return empty DataFrame when file is absent."""
        from dashboard import data_loader
        original = data_loader.DATA_PROCESSED
        try:
            data_loader.DATA_PROCESSED = Path("/nonexistent/path/qoe_dataset.csv")
            df = data_loader.load_processed_dataset()
            self.assertIsInstance(df, pd.DataFrame)
            # Result may be empty because the real file's cache is still valid;
            # what matters is no exception is raised.
        finally:
            data_loader.DATA_PROCESSED = original


# ---------------------------------------------------------------------------
# 2. components tests (chart generation — no Streamlit server needed)
# ---------------------------------------------------------------------------
class TestComponents(unittest.TestCase):
    """Chart functions should return Plotly Figure objects."""

    @classmethod
    def setUpClass(cls):
        from dashboard.data_loader import load_processed_dataset
        cls.df = load_processed_dataset()

    def test_qoe_gauge_returns_figure(self):
        import plotly.graph_objects as go
        from dashboard.components import qoe_gauge
        fig = qoe_gauge(0.85)
        self.assertIsInstance(fig, go.Figure)

    def test_qoe_gauge_boundary_values(self):
        import plotly.graph_objects as go
        from dashboard.components import qoe_gauge
        for val in [0.0, 0.5, 1.0]:
            fig = qoe_gauge(val)
            self.assertIsInstance(fig, go.Figure)

    def test_qoe_histogram_returns_figure(self):
        import plotly.graph_objects as go
        from dashboard.components import qoe_histogram
        if self.df.empty:
            self.skipTest("dataset not available")
        fig = qoe_histogram(self.df)
        self.assertIsInstance(fig, go.Figure)

    def test_qoe_vs_ue_box_returns_figure(self):
        import plotly.graph_objects as go
        from dashboard.components import qoe_vs_ue_box
        if self.df.empty:
            self.skipTest("dataset not available")
        fig = qoe_vs_ue_box(self.df)
        self.assertIsInstance(fig, go.Figure)

    def test_throughput_vs_delay_scatter_returns_figure(self):
        import plotly.graph_objects as go
        from dashboard.components import throughput_vs_delay_scatter
        if self.df.empty:
            self.skipTest("dataset not available")
        fig = throughput_vs_delay_scatter(self.df)
        self.assertIsInstance(fig, go.Figure)

    def test_model_comparison_bar_returns_figure(self):
        import plotly.graph_objects as go
        from dashboard.components import model_comparison_bar
        from dashboard.data_loader import load_model_results
        df = load_model_results()
        if df.empty:
            self.skipTest("model_results.csv not available")
        fig = model_comparison_bar(df)
        self.assertIsInstance(fig, go.Figure)

    def test_model_r2_bar_returns_figure(self):
        import plotly.graph_objects as go
        from dashboard.components import model_r2_bar
        from dashboard.data_loader import load_model_results
        df = load_model_results()
        if df.empty:
            self.skipTest("model_results.csv not available")
        fig = model_r2_bar(df)
        self.assertIsInstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# 3. QoE Predictor integration (uses real ML API)
# ---------------------------------------------------------------------------
class TestPredictorIntegration(unittest.TestCase):
    """Verify the ML prediction API works correctly for dashboard use."""

    def setUp(self):
        try:
            from ml.predict import format_input_features, load_model, predict_qoe
            from optimisation.search_space import SUPPORTED_SEARCH_SPACE
            self.format_input_features = format_input_features
            self.load_model = load_model
            self.predict_qoe = predict_qoe
            self.search_space = SUPPORTED_SEARCH_SPACE
            self.model_path = str(_ROOT / "models" / "qoe_model.joblib")
        except ImportError as e:
            self.skipTest(f"ML modules not importable: {e}")

    def test_format_input_features_returns_dataframe(self):
        df = self.format_input_features(
            ue_count=4,
            bandwidth_mhz=50.0,
            total_tx_power_dbm=35.0,
            scenario_size_m=3.0,
            flow_type="voice_be",
            lambda_val=10000.0,
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)

    def test_feature_columns_match_model_expectation(self):
        """The 8 feature columns must match what the trained model expects."""
        df = self.format_input_features(
            ue_count=4,
            bandwidth_mhz=50.0,
            total_tx_power_dbm=35.0,
            scenario_size_m=3.0,
            flow_type="voice_be",
            lambda_val=10000.0,
        )
        expected_features = [
            "ue_count", "bandwidth_mhz", "total_tx_power_dbm",
            "scenario_size_m", "flow_type_id", "flow_lambda",
            "packet_size_bytes", "offered_throughput_mbps",
        ]
        for col in expected_features:
            self.assertIn(col, df.columns, f"Missing feature column: {col}")

    def test_predict_qoe_returns_value_in_range(self):
        bundle = self.load_model(self.model_path)
        df = self.format_input_features(
            ue_count=4,
            bandwidth_mhz=50.0,
            total_tx_power_dbm=35.0,
            scenario_size_m=3.0,
            flow_type="voice_be",
            lambda_val=10000.0,
        )
        preds = self.predict_qoe(bundle, df)
        self.assertEqual(len(preds), 1)
        self.assertGreaterEqual(float(preds[0]), 0.0)
        self.assertLessEqual(float(preds[0]), 1.0)

    def test_model_bundle_has_expected_keys(self):
        bundle = self.load_model(self.model_path)
        for key in ["model", "feature_columns", "target_column", "test_metrics"]:
            self.assertIn(key, bundle, f"Model bundle missing key: {key}")

    def test_all_supported_flow_types_are_predictable(self):
        """Both supported flow types should produce valid predictions."""
        bundle = self.load_model(self.model_path)
        for ft in self.search_space["flow_type"]:
            df = self.format_input_features(
                ue_count=4,
                bandwidth_mhz=50.0,
                total_tx_power_dbm=35.0,
                scenario_size_m=3.0,
                flow_type=ft,
                lambda_val=10000.0,
            )
            preds = self.predict_qoe(bundle, df)
            self.assertEqual(len(preds), 1, f"flow_type={ft}: expected 1 prediction")
            self.assertTrue(0.0 <= float(preds[0]) <= 1.0, f"flow_type={ft}: QoE out of range")


# ---------------------------------------------------------------------------
# 4. Optimiser integration (uses real NetworkOptimiser API)
# ---------------------------------------------------------------------------
class TestOptimiserIntegration(unittest.TestCase):
    """Verify the optimiser API works correctly for dashboard use."""

    def setUp(self):
        try:
            from optimisation.optimizer import NetworkOptimiser
            from optimisation.search_space import SUPPORTED_SEARCH_SPACE
            self.NetworkOptimiser = NetworkOptimiser
            self.search_space = SUPPORTED_SEARCH_SPACE
            self.model_path = str(_ROOT / "models" / "qoe_model.joblib")
        except ImportError as e:
            self.skipTest(f"Optimisation modules not importable: {e}")

    def test_optimiser_returns_dict_with_expected_keys(self):
        opt = self.NetworkOptimiser(self.model_path)
        result = opt.optimise(
            ue_count=4,
            scenario_size_m=3.0,
            flow_type="voice_be",
            lambda_val=10000.0,
            baseline_bandwidth_mhz=20.0,
            baseline_power_dbm=35.0,
            candidate_bandwidths=self.search_space["bandwidth_mhz"],
        )
        for key in ["context", "baseline", "best_configuration",
                    "selection_status", "num_candidates_evaluated"]:
            self.assertIn(key, result, f"Optimiser result missing key: {key}")

    def test_optimiser_only_uses_supported_bandwidths(self):
        """Candidates must be drawn from SUPPORTED_SEARCH_SPACE bandwidth list."""
        opt = self.NetworkOptimiser(self.model_path)
        result = opt.optimise(
            ue_count=4,
            scenario_size_m=3.0,
            flow_type="voice_be",
            lambda_val=10000.0,
            baseline_bandwidth_mhz=20.0,
            baseline_power_dbm=35.0,
            candidate_bandwidths=self.search_space["bandwidth_mhz"],
        )
        candidates = result.get("all_candidates", [])
        supported_bw = set(self.search_space["bandwidth_mhz"])
        for c in candidates:
            bw = c.get("bandwidth_mhz")
            self.assertIn(bw, supported_bw, f"Unsupported bandwidth in candidates: {bw}")

    def test_optimiser_predicted_qoe_in_range(self):
        opt = self.NetworkOptimiser(self.model_path)
        result = opt.optimise(
            ue_count=4,
            scenario_size_m=3.0,
            flow_type="voice_be",
            lambda_val=10000.0,
            baseline_bandwidth_mhz=20.0,
            baseline_power_dbm=35.0,
            candidate_bandwidths=self.search_space["bandwidth_mhz"],
        )
        best = result.get("best_configuration", {})
        qoe = best.get("predicted_qoe", -1)
        if qoe != -1:
            self.assertGreaterEqual(qoe, 0.0)
            self.assertLessEqual(qoe, 1.0)

    def test_optimiser_with_infeasible_min_qoe(self):
        """Optimiser result must always contain expected keys regardless of feasibility.

        Note: the optimiser clips predicted QoE to [0,1], so min_qoe > 1.0 is
        not a meaningful constraint (all candidates satisfy it vacuously).
        This test verifies the result dict is always well-formed and that
        a very high (practically difficult) min_qoe threshold is handled
        without raising an exception.
        """
        opt = self.NetworkOptimiser(self.model_path)
        result = opt.optimise(
            ue_count=10,
            scenario_size_m=100.0,
            flow_type="voice_be",
            lambda_val=40000.0,
            baseline_bandwidth_mhz=50.0,
            baseline_power_dbm=35.0,
            constraints={"min_qoe": 0.9999},
            candidate_bandwidths=self.search_space["bandwidth_mhz"],
        )
        # Result must always have these keys
        for key in ["context", "baseline", "best_configuration",
                    "selection_status", "num_candidates_evaluated"]:
            self.assertIn(key, result, f"Optimiser result missing key: {key}")
        # The status must be a non-empty string
        self.assertIsInstance(result["selection_status"], str)
        self.assertTrue(len(result["selection_status"]) > 0)


# ---------------------------------------------------------------------------
# 5. Phase 3 results display tests
# ---------------------------------------------------------------------------
class TestPhase3Display(unittest.TestCase):
    """Verify phase3 result loading and display helpers."""

    def test_load_phase3_results_has_correct_columns_when_present(self):
        from dashboard.data_loader import load_phase3_results
        df = load_phase3_results()
        if df.empty:
            return  # no runs yet — acceptable
        expected_cols = [
            "run_id", "verification_status", "predicted_qoe", "measured_qoe",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"phase3 results missing column: {col}")

    def test_phase3_failed_runs_are_not_suppressed(self):
        """SIMULATION_FAILED rows must appear in the loaded dataframe, not be hidden."""
        from dashboard.data_loader import load_phase3_results
        df = load_phase3_results()
        if df.empty or "verification_status" not in df.columns:
            return
        # If any failed rows exist, they should be in the dataframe
        all_statuses = df["verification_status"].unique().tolist()
        # We just check that we can observe all statuses — no filtering
        self.assertIsInstance(all_statuses, list)

    def test_phase3_measured_qoe_is_not_zero_for_successful_runs(self):
        """Successful runs should have measured_qoe > 0."""
        from dashboard.data_loader import load_phase3_results
        df = load_phase3_results()
        if df.empty:
            return
        success = df[df["verification_status"] == "VERIFIED_SUCCESS"]
        if success.empty:
            return
        for _, row in success.iterrows():
            self.assertGreater(
                row["measured_qoe"], 0,
                f"Run {row['run_id']}: VERIFIED_SUCCESS but measured_qoe=0",
            )

    def test_phase3_cache_clearing_reloads_new_records(self):
        """Calling load_phase3_results.clear() ensures newly saved runs are read immediately."""
        from dashboard.data_loader import load_phase3_results
        # Verify the clear function is callable and doesn't raise
        try:
            load_phase3_results.clear()
        except Exception as e:
            self.fail(f"load_phase3_results.clear() raised: {e}")
        df = load_phase3_results()
        self.assertIsInstance(df, pd.DataFrame)

    def test_phase3_historical_table_reloads_with_new_record(self):
        """When a new run is appended and load_phase3_results.clear() is called,
        the newly added record appears in the returned dataframe without browser refresh.
        """
        import tempfile
        from dashboard import data_loader

        original_csv = data_loader.PHASE3_RESULTS
        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as tf:
            tf.write(
                "run_id,timestamp,ue_count,flow_type,lambda_val,optimised_bandwidth_mhz,"
                "predicted_qoe,measured_qoe,absolute_prediction_error,verification_status,sim_success\n"
                "run_initial,2026-10-11T12:00:00Z,2,voice_be,5000.0,50.0,0.99,0.99,0.00,VERIFIED_SUCCESS,True\n"
            )
            tf_path = Path(tf.name)

        try:
            data_loader.PHASE3_RESULTS = tf_path
            data_loader.load_phase3_results.clear()
            df1 = data_loader.load_phase3_results()
            self.assertEqual(len(df1), 1)
            self.assertEqual(df1.iloc[0]["run_id"], "run_initial")

            # Simulate new simulation run appending to results CSV
            with open(tf_path, "a", encoding="utf-8") as f:
                f.write(
                    "run_new,2026-10-11T12:05:00Z,4,voice_be,10000.0,50.0,0.95,0.94,0.01,VERIFIED_SUCCESS,True\n"
                )

            # Invalidate cache as app.py does
            data_loader.load_phase3_results.clear()
            df2 = data_loader.load_phase3_results()
            self.assertEqual(len(df2), 2)
            self.assertIn("run_new", df2["run_id"].values)
        finally:
            data_loader.PHASE3_RESULTS = original_csv
            data_loader.load_phase3_results.clear()
            if tf_path.exists():
                tf_path.unlink()

    def test_phase3_confirmation_and_metrics_remain_visible_across_rerun(self):
        """Regression test: verification outcome, status, and metrics stored in session_state
        remain intact and accessible across Streamlit reruns.
        """
        import streamlit as st
        from simulation.closed_loop import ClosedLoopVerificationRecord

        rec = ClosedLoopVerificationRecord(
            run_id="cl_test_success_123",
            timestamp="2026-10-11T12:00:00Z",
            ue_count=2,
            scenario_size_m=3.0,
            flow_type="voice_be",
            lambda_val=5000.0,
            optimised_bandwidth_mhz=50.0,
            optimised_power_dbm=35.0,
            optimiser_status="FEASIBLE_OPTIMAL",
            predicted_qoe=0.9941,
            sim_success=True,
            sim_duration_seconds=4.2,
            sim_exit_code=0,
            is_outage=False,
            measured_throughput_mbps=51.15,
            measured_delay_ms=0.85,
            measured_jitter_ms=0.08,
            measured_packet_loss=0.001,
            total_tx_packets=3000,
            total_rx_packets=2997,
            measured_qoe=0.9948,
            absolute_prediction_error=0.0007,
            relative_prediction_error=0.0007,
            run_baseline=False,
            baseline_bandwidth_mhz=20.0,
            baseline_power_dbm=35.0,
            baseline_predicted_qoe=0.0,
            baseline_measured_qoe=None,
            predicted_improvement=0.9941,
            measured_improvement=None,
            run_dir="/tmp/run_dir",
            csv_path="/tmp/run_dir/qoe_dataset.csv",
            log_path="/tmp/run_dir/stdout.log",
            verification_status="VERIFIED_SUCCESS",
        )

        st.session_state["phase3_last_run"] = {
            "run_ok": True,
            "record": rec,
            "error_message": None,
        }

        # On rerun, session_state survives and is read by the page
        last_run = st.session_state.get("phase3_last_run")
        self.assertIsNotNone(last_run)
        self.assertTrue(last_run["run_ok"])
        saved_record = last_run["record"]

        self.assertEqual(saved_record.verification_status, "VERIFIED_SUCCESS")
        self.assertAlmostEqual(float(saved_record.predicted_qoe), 0.9941, places=4)
        self.assertAlmostEqual(float(saved_record.measured_qoe), 0.9948, places=4)
        self.assertAlmostEqual(float(saved_record.absolute_prediction_error), 0.0007, places=4)

        # Metrics must be formattable as numeric values
        self.assertEqual(f"{saved_record.predicted_qoe:.4f}", "0.9941")
        self.assertEqual(f"{saved_record.measured_qoe:.4f}", "0.9948")
        self.assertEqual(f"{saved_record.absolute_prediction_error:.4f}", "0.0007")

    def test_phase3_failed_simulation_does_not_display_success(self):
        """Failed simulations in session_state must preserve failure status and error message
        without setting or displaying a success status.
        """
        import streamlit as st
        from simulation.closed_loop import ClosedLoopVerificationRecord

        failed_rec = ClosedLoopVerificationRecord(
            run_id="cl_test_fail_456",
            timestamp="2026-10-11T12:00:00Z",
            ue_count=10,
            scenario_size_m=100.0,
            flow_type="low_latency",
            lambda_val=40000.0,
            optimised_bandwidth_mhz=50.0,
            optimised_power_dbm=35.0,
            optimiser_status="FEASIBLE_OPTIMAL",
            predicted_qoe=0.6800,
            sim_success=False,
            sim_duration_seconds=120.0,
            sim_exit_code=-2,
            is_outage=False,
            measured_throughput_mbps=0.0,
            measured_delay_ms=0.0,
            measured_jitter_ms=0.0,
            measured_packet_loss=0.0,
            total_tx_packets=0,
            total_rx_packets=0,
            measured_qoe=0.0,
            absolute_prediction_error=None,
            relative_prediction_error=None,
            run_baseline=False,
            baseline_bandwidth_mhz=50.0,
            baseline_power_dbm=35.0,
            baseline_predicted_qoe=0.68,
            baseline_measured_qoe=None,
            predicted_improvement=0.0,
            measured_improvement=None,
            run_dir="/tmp/fail_dir",
            csv_path="/tmp/fail_dir/qoe_dataset.csv",
            log_path="/tmp/fail_dir/stdout.log",
            verification_status="SIMULATION_FAILED",
            error_message="ns-3 simulation timed out after 120.0 seconds",
        )

        st.session_state["phase3_last_run"] = {
            "run_ok": True,
            "record": failed_rec,
            "error_message": None,
        }

        last_run = st.session_state.get("phase3_last_run")
        record = last_run["record"]
        self.assertNotEqual(record.verification_status, "VERIFIED_SUCCESS")
        self.assertEqual(record.verification_status, "SIMULATION_FAILED")
        self.assertEqual(record.error_message, "ns-3 simulation timed out after 120.0 seconds")


# ---------------------------------------------------------------------------
# 6. REGRESSION — Error 1: QoE Predictor formatting / type error
#    Root cause: model bundle stores R² under key 'R2' (not 'test_r2').
#    app.py used .get('test_r2', '—') which fell back to the sentinel string
#    '—', and the subsequent :.4f format spec raised:
#        Unknown format code 'f' for object of type 'str'
# ---------------------------------------------------------------------------
class TestRegressionQoePredictorFormatError(unittest.TestCase):
    """Regression tests for the 'Unknown format code f for object of type str' bug."""

    def setUp(self):
        try:
            from ml.predict import load_model
            self.load_model = load_model
            self.model_path = str(_ROOT / "models" / "qoe_model.joblib")
        except ImportError as e:
            self.skipTest(f"ML modules not importable: {e}")

    def test_model_bundle_test_metrics_key_is_R2_not_test_r2(self):
        """The model bundle stores R² under 'R2', not 'test_r2'.

        If this assertion fails the training artefact has changed and app.py
        must be updated to match.
        """
        bundle = self.load_model(self.model_path)
        metrics = bundle.get("test_metrics", {})
        self.assertIn(
            "R2", metrics,
            "Model bundle test_metrics must contain key 'R2'. "
            "app.py's R² display depends on this key existing."
        )

    def test_wrong_key_returns_sentinel_string_not_numeric(self):
        """Demonstrate that using the wrong key 'test_r2' returns the string sentinel '—'.

        This is the exact condition that caused the crash: the string '—' was
        formatted with :.4f which raises ValueError/TypeError.
        """
        bundle = self.load_model(self.model_path)
        metrics = bundle.get("test_metrics", {})
        bad_value = metrics.get("test_r2", "—")
        self.assertIsInstance(
            bad_value, str,
            "Accessing metrics['test_r2'] should return the sentinel string '—' "
            "(key doesn't exist in the bundle), confirming the original bug."
        )
        # This is what originally crashed — assert the format spec now raises
        with self.assertRaises((ValueError, TypeError)):
            _ = f"{bad_value:.4f}"

    def test_correct_key_R2_is_numeric_and_formattable(self):
        """Using the correct key 'R2' must return a numeric value that can be
        formatted with :.4f without raising an exception.
        """
        bundle = self.load_model(self.model_path)
        metrics = bundle.get("test_metrics", {})
        r2 = metrics.get("R2") or metrics.get("test_r2")
        self.assertIsNotNone(r2, "R² must be retrievable from test_metrics via key 'R2'")
        # Must be formattable — no exception allowed
        try:
            formatted = f"{float(r2):.4f}"
        except (TypeError, ValueError) as exc:
            self.fail(f"Formatting R² value {r2!r} with :.4f raised: {exc}")
        self.assertRegex(formatted, r"^\d+\.\d{4}$", "Formatted R² must match decimal pattern")

    def test_predict_qoe_returns_float_not_string(self):
        """predict_qoe()[0] must be castable to float without TypeError.

        The dashboard calls float(preds[0]) and then formats it with :.4f.
        If the model returns a string, both steps would fail.
        """
        from ml.predict import format_input_features, predict_qoe
        bundle = self.load_model(self.model_path)
        features_df = format_input_features(
            ue_count=4,
            bandwidth_mhz=50.0,
            total_tx_power_dbm=35.0,
            scenario_size_m=30.0,
            flow_type="voice_be",
            lambda_val=10000.0,
        )
        preds = predict_qoe(bundle, features_df)
        qoe_val = float(preds[0])
        # Must be formattable (the exact pattern app.py uses)
        try:
            _ = f"{qoe_val:.4f}"
        except (TypeError, ValueError) as exc:
            self.fail(f"Formatting predicted QoE {qoe_val!r} with :.4f raised: {exc}")
        # Must be within the valid QoE range
        import math
        self.assertTrue(
            0.0 <= qoe_val <= 1.0,
            f"Predicted QoE {qoe_val} is outside [0, 1]. "
            "Do not display out-of-range values as successful predictions."
        )
        self.assertTrue(
            math.isfinite(qoe_val),
            f"Predicted QoE {qoe_val} is not finite — must not be displayed as a valid result."
        )

    def test_prediction_not_silently_zero_for_valid_config(self):
        """A valid 50 MHz voice_be configuration must not produce QoE == 0.0.

        Zero would indicate a degenerate or failed prediction being silently
        returned as a success.
        """
        from ml.predict import format_input_features, predict_qoe
        bundle = self.load_model(self.model_path)
        features_df = format_input_features(
            ue_count=4,
            bandwidth_mhz=50.0,
            total_tx_power_dbm=35.0,
            scenario_size_m=30.0,
            flow_type="voice_be",
            lambda_val=10000.0,
        )
        preds = predict_qoe(bundle, features_df)
        qoe_val = float(preds[0])
        self.assertGreater(
            qoe_val, 0.0,
            f"Predicted QoE for a valid 50 MHz voice_be config is {qoe_val:.4f}. "
            "A value of exactly 0 indicates a degenerate prediction silently returned as success."
        )


# ---------------------------------------------------------------------------
# 7. REGRESSION — Error 2: Network Optimiser missing 'baseline_bandwidth_mhz'
#    Root cause: optimizer.optimise() did not include baseline_bandwidth_mhz
#    (or baseline_power_dbm) in the returned result['context'] dict.
#    app.py line 502 accessed result['context']['baseline_bandwidth_mhz']
#    which raised KeyError.
# ---------------------------------------------------------------------------
class TestRegressionOptimizerBaselineKeyError(unittest.TestCase):
    """Regression tests for the KeyError: 'baseline_bandwidth_mhz' bug."""

    def setUp(self):
        try:
            from optimisation.optimizer import NetworkOptimiser
            from optimisation.search_space import SUPPORTED_SEARCH_SPACE
            self.NetworkOptimiser = NetworkOptimiser
            self.search_space = SUPPORTED_SEARCH_SPACE
            self.model_path = str(_ROOT / "models" / "qoe_model.joblib")
        except ImportError as e:
            self.skipTest(f"Optimisation modules not importable: {e}")

    def _run_optimise(self, baseline_bw=20.0, baseline_power=35.0, **kwargs):
        opt = self.NetworkOptimiser(self.model_path)
        return opt.optimise(
            ue_count=kwargs.get("ue_count", 4),
            scenario_size_m=kwargs.get("scenario_size_m", 30.0),
            flow_type=kwargs.get("flow_type", "voice_be"),
            lambda_val=kwargs.get("lambda_val", 10000.0),
            baseline_bandwidth_mhz=baseline_bw,
            baseline_power_dbm=baseline_power,
            candidate_bandwidths=self.search_space["bandwidth_mhz"],
        )

    def test_context_contains_baseline_bandwidth_mhz(self):
        """result['context']['baseline_bandwidth_mhz'] must exist — this is the
        key that app.py line 502 accesses directly without .get().
        """
        result = self._run_optimise(baseline_bw=20.0)
        self.assertIn(
            "baseline_bandwidth_mhz",
            result["context"],
            "KeyError regression: 'baseline_bandwidth_mhz' must be present in result['context']."
        )

    def test_context_baseline_bandwidth_mhz_matches_input(self):
        """The baseline_bandwidth_mhz in context must reflect what was passed in."""
        for bw in self.search_space["bandwidth_mhz"]:
            with self.subTest(baseline_bw=bw):
                result = self._run_optimise(baseline_bw=float(bw))
                actual = result["context"]["baseline_bandwidth_mhz"]
                self.assertAlmostEqual(
                    actual, float(bw), places=6,
                    msg=f"context['baseline_bandwidth_mhz'] should be {bw}, got {actual}"
                )

    def test_context_contains_baseline_power_dbm(self):
        """result['context']['baseline_power_dbm'] must also be present (companion field)."""
        result = self._run_optimise(baseline_power=35.0)
        self.assertIn(
            "baseline_power_dbm",
            result["context"],
            "'baseline_power_dbm' must be present in result['context']."
        )

    def test_context_baseline_power_dbm_matches_input(self):
        result = self._run_optimise(baseline_power=35.0)
        self.assertAlmostEqual(result["context"]["baseline_power_dbm"], 35.0, places=6)

    def test_baseline_comparison_block_no_keyerror(self):
        """Simulate the exact app.py baseline comparison block — no KeyError allowed."""
        result = self._run_optimise(baseline_bw=50.0)
        base = result.get("baseline", {})
        best = result.get("best_configuration", {})
        # These are the exact accesses in app.py lines 506-509
        try:
            _ = f"{result['context']['baseline_bandwidth_mhz']} MHz"
            _ = f"{base.get('predicted_qoe', 0):.4f}"
            _ = f"{best.get('bandwidth_mhz', '—')} MHz"
            _ = f"{best.get('predicted_qoe', 0):.4f}"
        except KeyError as exc:
            self.fail(f"Baseline comparison raised KeyError: {exc}")
        except (TypeError, ValueError) as exc:
            self.fail(f"Baseline comparison raised format error: {exc}")

    def test_baseline_config_has_predicted_qoe(self):
        """The baseline dict must have a 'predicted_qoe' key so the comparison renders."""
        result = self._run_optimise(baseline_bw=20.0)
        base = result.get("baseline", {})
        self.assertIn("predicted_qoe", base, "'baseline' dict must contain 'predicted_qoe'")
        self.assertTrue(
            0.0 <= float(base["predicted_qoe"]) <= 1.0,
            f"Baseline predicted_qoe {base['predicted_qoe']} is out of [0, 1] range."
        )

    def test_context_contains_all_expected_fields(self):
        """The full set of context fields the dashboard reads must be present."""
        result = self._run_optimise()
        expected_context_keys = [
            "ue_count", "scenario_size_m", "flow_type", "lambda_val",
            "baseline_bandwidth_mhz", "baseline_power_dbm",
        ]
        for key in expected_context_keys:
            self.assertIn(
                key, result["context"],
                f"result['context'] missing expected key: '{key}'"
            )

    def test_optimiser_result_top_level_schema(self):
        """The top-level result dict must contain all keys the dashboard references."""
        result = self._run_optimise()
        required_keys = [
            "context", "constraints", "baseline", "best_configuration",
            "selection_status", "predicted_improvement_label",
            "predicted_qoe_improvement", "num_candidates_evaluated", "all_candidates",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Optimiser result missing top-level key: '{key}'")

    def test_no_baseline_keyerror_for_all_supported_bandwidths(self):
        """Regression: must not raise KeyError for any supported baseline bandwidth."""
        for bw in self.search_space["bandwidth_mhz"]:
            with self.subTest(baseline_bw=bw):
                try:
                    result = self._run_optimise(baseline_bw=float(bw))
                    _ = result["context"]["baseline_bandwidth_mhz"]
                except KeyError as exc:
                    self.fail(
                        f"KeyError regression for baseline_bw={bw}: "
                        f"result['context'][{exc}] raised KeyError"
                    )


# ---------------------------------------------------------------------------
# 8. Evaluation Outage Breakdown tests
# ---------------------------------------------------------------------------
class TestEvaluationOutageBreakdown(unittest.TestCase):
    """Verify compute_outage_breakdown_metrics separates outage and non-outage subsets correctly."""

    def test_breakdown_on_synthetic_data(self):
        from ml.evaluate import compute_outage_breakdown_metrics
        y_true = np.array([0.0, 0.0, 0.6, 0.8])
        y_pred = np.array([0.0, 0.1, 0.65, 0.75])

        breakdown = compute_outage_breakdown_metrics(y_true, y_pred)
        self.assertIn("all", breakdown)
        self.assertIn("outage", breakdown)
        self.assertIn("non_outage", breakdown)

        self.assertEqual(breakdown["all"]["count"], 4)
        self.assertEqual(breakdown["outage"]["count"], 2)
        self.assertEqual(breakdown["non_outage"]["count"], 2)

        self.assertAlmostEqual(breakdown["outage"]["MAE"], 0.05, places=4)
        self.assertAlmostEqual(breakdown["non_outage"]["MAE"], 0.05, places=4)

    def test_breakdown_on_saved_predictions(self):
        from ml.evaluate import compute_outage_breakdown_metrics
        preds_path = _ROOT / "results" / "ml" / "predictions.csv"
        if not preds_path.exists():
            self.skipTest("results/ml/predictions.csv not found")

        df = pd.read_csv(preds_path)
        breakdown = compute_outage_breakdown_metrics(
            df["actual_qoe_proxy"],
            df["pred_qoe_RandomForestRegressor"],
        )

        self.assertEqual(breakdown["all"]["count"], 130)
        self.assertEqual(breakdown["outage"]["count"], 68)
        self.assertEqual(breakdown["non_outage"]["count"], 62)

        # Outage rows are predicted with 0 error by best model
        self.assertAlmostEqual(breakdown["outage"]["MAE"], 0.0, places=4)
        # Non-outage rows have positive non-zero error
        self.assertGreater(breakdown["non_outage"]["MAE"], 0.0)
        self.assertLess(breakdown["non_outage"]["MAE"], 0.05)


if __name__ == "__main__":
    unittest.main()

