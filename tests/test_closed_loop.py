"""Unit tests for simulation/closed_loop.py using mocked ns-3 runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from simulation.closed_loop import ClosedLoopVerifier
from simulation.ns3_runner import Ns3RunResult, Ns3Runner


class TestClosedLoop(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.results_csv = self.tmp_path / "verification_results.csv"

        # Mock runner
        self.mock_runner = MagicMock(spec=Ns3Runner)
        self.verifier = ClosedLoopVerifier(
            model_path="models/qoe_model.joblib",
            output_base_dir=self.tmp_path / "runs",
            results_csv_path=self.results_csv,
            runner=self.mock_runner,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_closed_loop_success_with_baseline(self):
        """Verify successful closed-loop verification and error calculation."""
        # Create mock run directory and CSV outputs for optimal and baseline
        opt_run_dir = self.tmp_path / "opt_run"
        opt_run_dir.mkdir(parents=True)
        opt_csv = opt_run_dir / "qoe_dataset.csv"
        opt_csv.write_text(
            "scenario_id,ue_count,lambda_be,bandwidth_mhz,total_tx_power_dbm,scenario_size_m,"
            "flow_id,tx_packets,rx_packets,lost_packets,throughput_mbps,delay_ms,jitter_ms,packet_loss,qoe_proxy\n"
            "1,2,5000,50,35,30,1,6000,5998,0,10.2366,0.2760,0.0300,0.0003,0.6801\n"
            "1,2,5000,50,35,30,2,3000,2997,0,51.1488,0.8460,0.0817,0.0010,0.9948\n"
        )

        base_run_dir = self.tmp_path / "base_run"
        base_run_dir.mkdir(parents=True)
        base_csv = base_run_dir / "qoe_dataset.csv"
        base_csv.write_text(
            "scenario_id,ue_count,lambda_be,bandwidth_mhz,total_tx_power_dbm,scenario_size_m,"
            "flow_id,tx_packets,rx_packets,lost_packets,throughput_mbps,delay_ms,jitter_ms,packet_loss,qoe_proxy\n"
            "1,2,5000,20,35,30,1,6000,0,0,0.0,0.0,0.0,1.0,0.0\n"
            "1,2,5000,20,35,30,2,3000,0,0,0.0,0.0,0.0,1.0,0.0\n"
        )

        # Configure runner mock responses
        self.mock_runner.run_simulation.side_effect = [
            Ns3RunResult(
                run_id="opt_run",
                success=True,
                exit_code=0,
                run_dir=opt_run_dir,
                csv_path=opt_csv,
                summary_path=opt_run_dir / "summary.txt",
                stdout="OK",
                stderr="",
                duration_seconds=4.2,
                command=["ns3"],
                config={},
            ),
            Ns3RunResult(
                run_id="base_run",
                success=True,
                exit_code=0,
                run_dir=base_run_dir,
                csv_path=base_csv,
                summary_path=base_run_dir / "summary.txt",
                stdout="OK",
                stderr="",
                duration_seconds=3.8,
                command=["ns3"],
                config={},
            ),
        ]

        rec = self.verifier.run_closed_loop_verification(
            ue_count=2,
            scenario_size_m=30.0,
            flow_type="voice_be",
            lambda_val=5000.0,
            baseline_bandwidth_mhz=20.0,
            run_baseline_sim=True,
        )

        self.assertTrue(rec.sim_success)
        self.assertEqual(rec.optimised_bandwidth_mhz, 50.0)
        self.assertAlmostEqual(rec.measured_qoe, 0.9948, places=3)
        self.assertIsNotNone(rec.absolute_prediction_error)
        self.assertIsNotNone(rec.relative_prediction_error)
        # Baseline was 0.0, so measured improvement should be positive
        self.assertAlmostEqual(rec.baseline_measured_qoe, 0.0, places=3)
        self.assertAlmostEqual(rec.measured_improvement, 0.9948, places=3)
        # Results CSV must be created
        self.assertTrue(self.results_csv.exists())

    def test_closed_loop_failed_simulation_handling(self):
        """Verify handling when ns-3 simulation fails."""
        self.mock_runner.run_simulation.return_value = Ns3RunResult(
            run_id="fail_run",
            success=False,
            exit_code=1,
            run_dir=self.tmp_path,
            csv_path=self.tmp_path / "missing.csv",
            summary_path=self.tmp_path / "summary.txt",
            stdout="",
            stderr="Error encountered",
            duration_seconds=0.5,
            command=["ns3"],
            config={},
            error_message="Simulation crashed",
        )

        rec = self.verifier.run_closed_loop_verification(
            ue_count=2,
            scenario_size_m=30.0,
            flow_type="voice_be",
            lambda_val=5000.0,
            run_baseline_sim=False,
        )

        self.assertFalse(rec.sim_success)
        self.assertEqual(rec.verification_status, "SIMULATION_FAILED")
        self.assertIsNone(rec.absolute_prediction_error)
        self.assertIsNone(rec.relative_prediction_error)


if __name__ == "__main__":
    unittest.main()

