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

    def test_recovery_not_needed_when_initial_qoe_meets_target(self):
        """When initial simulation already achieves target QoE, recovery status is NOT_NEEDED and no corrective run is made."""
        run_dir = self.tmp_path / "run_50mhz"
        run_dir.mkdir(parents=True)
        csv_file = run_dir / "qoe_dataset.csv"
        csv_file.write_text(
            "scenario_id,ue_count,lambda_be,bandwidth_mhz,total_tx_power_dbm,scenario_size_m,"
            "flow_id,tx_packets,rx_packets,lost_packets,throughput_mbps,delay_ms,jitter_ms,packet_loss,qoe_proxy\n"
            "1,2,5000,50,35,30,1,6000,5998,0,10.2,0.25,0.03,0.0003,0.85\n"
            "1,2,5000,50,35,30,2,3000,2997,0,51.1,0.80,0.08,0.0010,0.92\n"
        )

        self.mock_runner.run_simulation.return_value = Ns3RunResult(
            run_id="run_50mhz",
            success=True,
            exit_code=0,
            run_dir=run_dir,
            csv_path=csv_file,
            summary_path=run_dir / "summary.txt",
            stdout="OK",
            stderr="",
            duration_seconds=3.0,
            command=["ns3"],
            config={},
        )

        rec = self.verifier.run_verification_with_recovery(
            ue_count=2,
            initial_bandwidth_mhz=50.0,
            target_qoe=0.80,
            recovery_results_csv_path=self.tmp_path / "recovery_results.csv",
        )

        self.assertEqual(rec.status, "NOT_NEEDED")
        self.assertFalse(rec.recovery_attempted)
        self.assertTrue(rec.target_achieved)
        self.assertIsNone(rec.corrected_bandwidth_mhz)
        self.assertIsNone(rec.corrective_record)
        # Runner called exactly once for initial simulation
        self.assertEqual(self.mock_runner.run_simulation.call_count, 1)

    def test_recovery_success_from_20_to_50_mhz(self):
        """When initial simulation is at 20 MHz (outage/low QoE), corrective simulation at 50 MHz is executed and succeeds."""
        init_run_dir = self.tmp_path / "run_20mhz"
        init_run_dir.mkdir(parents=True)
        init_csv = init_run_dir / "qoe_dataset.csv"
        init_csv.write_text(
            "scenario_id,ue_count,lambda_be,bandwidth_mhz,total_tx_power_dbm,scenario_size_m,"
            "flow_id,tx_packets,rx_packets,lost_packets,throughput_mbps,delay_ms,jitter_ms,packet_loss,qoe_proxy\n"
            "1,2,5000,20,35,30,1,6000,0,0,0.0,0.0,0.0,1.0,0.0\n"
            "1,2,5000,20,35,30,2,3000,0,0,0.0,0.0,0.0,1.0,0.0\n"
        )

        corr_run_dir = self.tmp_path / "run_50mhz"
        corr_run_dir.mkdir(parents=True)
        corr_csv = corr_run_dir / "qoe_dataset.csv"
        corr_csv.write_text(
            "scenario_id,ue_count,lambda_be,bandwidth_mhz,total_tx_power_dbm,scenario_size_m,"
            "flow_id,tx_packets,rx_packets,lost_packets,throughput_mbps,delay_ms,jitter_ms,packet_loss,qoe_proxy\n"
            "1,2,5000,50,35,30,1,6000,5998,0,10.2,0.25,0.03,0.0003,0.68\n"
            "1,2,5000,50,35,30,2,3000,2997,0,51.1,0.80,0.08,0.0010,0.95\n"
        )

        self.mock_runner.run_simulation.side_effect = [
            Ns3RunResult(
                run_id="run_20mhz",
                success=True,
                exit_code=0,
                run_dir=init_run_dir,
                csv_path=init_csv,
                summary_path=init_run_dir / "summary.txt",
                stdout="OK",
                stderr="",
                duration_seconds=3.0,
                command=["ns3"],
                config={},
            ),
            Ns3RunResult(
                run_id="run_50mhz",
                success=True,
                exit_code=0,
                run_dir=corr_run_dir,
                csv_path=corr_csv,
                summary_path=corr_run_dir / "summary.txt",
                stdout="OK",
                stderr="",
                duration_seconds=3.5,
                command=["ns3"],
                config={},
            ),
        ]

        rec = self.verifier.run_verification_with_recovery(
            ue_count=2,
            initial_bandwidth_mhz=20.0,
            target_qoe=0.80,
            recovery_results_csv_path=self.tmp_path / "recovery_results.csv",
        )

        self.assertEqual(rec.status, "SUCCESS")
        self.assertTrue(rec.recovery_attempted)
        self.assertTrue(rec.target_achieved)
        self.assertEqual(rec.previous_bandwidth_mhz, 20.0)
        self.assertEqual(rec.corrected_bandwidth_mhz, 50.0)
        self.assertAlmostEqual(rec.initial_metrics["measured_qoe"], 0.0, places=3)
        self.assertGreaterEqual(rec.corrected_metrics["measured_qoe"], 0.80)
        self.assertAlmostEqual(rec.corrected_metrics["measured_qoe"], 0.995, places=2)
        # Runner called exactly twice: 1 initial + 1 corrective
        self.assertEqual(self.mock_runner.run_simulation.call_count, 2)

        # Verify persistence of recovery results CSV
        self.assertTrue((self.tmp_path / "recovery_results.csv").exists())

    def test_recovery_no_alternative_configuration(self):
        """When initial bandwidth is already 50 MHz and QoE < target, NO_ALTERNATIVE_CONFIGURATION is reported honestly."""
        init_run_dir = self.tmp_path / "run_50mhz_low"
        init_run_dir.mkdir(parents=True)
        init_csv = init_run_dir / "qoe_dataset.csv"
        init_csv.write_text(
            "scenario_id,ue_count,lambda_be,bandwidth_mhz,total_tx_power_dbm,scenario_size_m,"
            "flow_id,tx_packets,rx_packets,lost_packets,throughput_mbps,delay_ms,jitter_ms,packet_loss,qoe_proxy\n"
            "1,2,5000,50,35,30,1,6000,5000,1000,5.0,20.0,5.0,0.1667,0.50\n"
            "1,2,5000,50,35,30,2,3000,2500,500,15.0,25.0,8.0,0.1667,0.55\n"
        )

        self.mock_runner.run_simulation.return_value = Ns3RunResult(
            run_id="run_50mhz_low",
            success=True,
            exit_code=0,
            run_dir=init_run_dir,
            csv_path=init_csv,
            summary_path=init_run_dir / "summary.txt",
            stdout="OK",
            stderr="",
            duration_seconds=3.0,
            command=["ns3"],
            config={},
        )

        rec = self.verifier.run_verification_with_recovery(
            ue_count=2,
            initial_bandwidth_mhz=50.0,
            target_qoe=0.80,
            recovery_results_csv_path=self.tmp_path / "recovery_results.csv",
        )

        self.assertEqual(rec.status, "NO_ALTERNATIVE_CONFIGURATION")
        self.assertFalse(rec.recovery_attempted)
        self.assertFalse(rec.target_achieved)
        self.assertEqual(self.mock_runner.run_simulation.call_count, 1)

    def test_recovery_corrective_simulation_fails(self):
        """When corrective ns-3 simulation fails/crashes, recovery status is SIMULATION_FAILED with error details."""
        init_run_dir = self.tmp_path / "run_20mhz"
        init_run_dir.mkdir(parents=True)
        init_csv = init_run_dir / "qoe_dataset.csv"
        init_csv.write_text(
            "scenario_id,ue_count,lambda_be,bandwidth_mhz,total_tx_power_dbm,scenario_size_m,"
            "flow_id,tx_packets,rx_packets,lost_packets,throughput_mbps,delay_ms,jitter_ms,packet_loss,qoe_proxy\n"
            "1,2,5000,20,35,30,1,6000,0,0,0.0,0.0,0.0,1.0,0.0\n"
            "1,2,5000,20,35,30,2,3000,0,0,0.0,0.0,0.0,1.0,0.0\n"
        )

        self.mock_runner.run_simulation.side_effect = [
            Ns3RunResult(
                run_id="run_20mhz",
                success=True,
                exit_code=0,
                run_dir=init_run_dir,
                csv_path=init_csv,
                summary_path=init_run_dir / "summary.txt",
                stdout="OK",
                stderr="",
                duration_seconds=3.0,
                command=["ns3"],
                config={},
            ),
            Ns3RunResult(
                run_id="run_50mhz_fail",
                success=False,
                exit_code=1,
                run_dir=self.tmp_path / "run_50mhz_fail",
                csv_path=self.tmp_path / "missing.csv",
                summary_path=self.tmp_path / "summary.txt",
                stdout="",
                stderr="Segmentation fault",
                duration_seconds=0.2,
                command=["ns3"],
                config={},
                error_message="Simulation terminated unexpectedly",
            ),
        ]

        rec = self.verifier.run_verification_with_recovery(
            ue_count=2,
            initial_bandwidth_mhz=20.0,
            target_qoe=0.80,
            recovery_results_csv_path=self.tmp_path / "recovery_results.csv",
        )

        self.assertEqual(rec.status, "SIMULATION_FAILED")
        self.assertTrue(rec.recovery_attempted)
        self.assertFalse(rec.target_achieved)
        self.assertIn("Simulation terminated unexpectedly", rec.error_message)
        # Runner called at most twice (1 initial + 1 corrective, no infinite retry)
        self.assertEqual(self.mock_runner.run_simulation.call_count, 2)

    def test_recovery_target_not_met_after_correction(self):
        """When corrective simulation succeeds but measured QoE is still below target, status is TARGET_NOT_MET."""
        init_run_dir = self.tmp_path / "run_20mhz"
        init_run_dir.mkdir(parents=True)
        init_csv = init_run_dir / "qoe_dataset.csv"
        init_csv.write_text(
            "scenario_id,ue_count,lambda_be,bandwidth_mhz,total_tx_power_dbm,scenario_size_m,"
            "flow_id,tx_packets,rx_packets,lost_packets,throughput_mbps,delay_ms,jitter_ms,packet_loss,qoe_proxy\n"
            "1,2,5000,20,35,30,1,6000,0,0,0.0,0.0,0.0,1.0,0.0\n"
            "1,2,5000,20,35,30,2,3000,0,0,0.0,0.0,0.0,1.0,0.0\n"
        )

        corr_run_dir = self.tmp_path / "run_50mhz"
        corr_run_dir.mkdir(parents=True)
        corr_csv = corr_run_dir / "qoe_dataset.csv"
        corr_csv.write_text(
            "scenario_id,ue_count,lambda_be,bandwidth_mhz,total_tx_power_dbm,scenario_size_m,"
            "flow_id,tx_packets,rx_packets,lost_packets,throughput_mbps,delay_ms,jitter_ms,packet_loss,qoe_proxy\n"
            "1,2,5000,50,35,30,1,6000,4000,2000,5.0,20.0,5.0,0.3333,0.50\n"
            "1,2,5000,50,35,30,2,3000,2000,1000,15.0,25.0,8.0,0.3333,0.60\n"
        )

        self.mock_runner.run_simulation.side_effect = [
            Ns3RunResult(
                run_id="run_20mhz",
                success=True,
                exit_code=0,
                run_dir=init_run_dir,
                csv_path=init_csv,
                summary_path=init_run_dir / "summary.txt",
                stdout="OK",
                stderr="",
                duration_seconds=3.0,
                command=["ns3"],
                config={},
            ),
            Ns3RunResult(
                run_id="run_50mhz",
                success=True,
                exit_code=0,
                run_dir=corr_run_dir,
                csv_path=corr_csv,
                summary_path=corr_run_dir / "summary.txt",
                stdout="OK",
                stderr="",
                duration_seconds=3.5,
                command=["ns3"],
                config={},
            ),
        ]

        rec = self.verifier.run_verification_with_recovery(
            ue_count=2,
            initial_bandwidth_mhz=20.0,
            target_qoe=0.80,  # Target is 0.80, but 50 MHz measured QoE is 0.60
            recovery_results_csv_path=self.tmp_path / "recovery_results.csv",
        )

        self.assertEqual(rec.status, "TARGET_NOT_MET")
        self.assertTrue(rec.recovery_attempted)
        self.assertFalse(rec.target_achieved)
        self.assertEqual(rec.corrected_bandwidth_mhz, 50.0)
        self.assertEqual(self.mock_runner.run_simulation.call_count, 2)


if __name__ == "__main__":
    unittest.main()


