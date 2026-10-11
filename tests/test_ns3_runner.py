"""Unit tests for simulation/ns3_runner.py with mocked subprocess execution."""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from simulation.ns3_runner import Ns3RunResult, Ns3Runner


class TestNs3Runner(unittest.TestCase):

    def setUp(self):
        self.runner = Ns3Runner(
            executable_path="/mock/path/ns3-exec",
            default_output_base="/tmp/mock_phase3_runs",
            default_timeout_seconds=30.0,
        )

    def test_build_cli_arguments_mapping(self):
        """Verify explicit mapping of configuration dictionary to ns-3 CLI arguments."""
        config = {
            "ue_count": 4,
            "bandwidth_mhz": 50.0,
            "total_tx_power_dbm": 35.0,
            "scenario_size_m": 30.0,
            "lambda_be": 20000.0,
            "scenario_id": 42,
            "sim_time": "1s",
        }
        run_dir = Path("/tmp/mock_run_1")
        args = self.runner.build_cli_arguments(config, run_dir, sim_tag="sim_summary.txt")

        self.assertEqual(args[0], "/mock/path/ns3-exec")
        self.assertIn("--ueNumPergNb=4", args)
        self.assertIn("--bandwidthBand1=50000000", args)
        self.assertIn("--totalTxPower=35.0", args)
        self.assertIn("--scenarioSize=30.0", args)
        self.assertIn("--lambdaBe=20000", args)
        self.assertIn("--scenarioId=42", args)
        self.assertIn("--simTime=1s", args)
        self.assertIn("--outputDir=/tmp/mock_run_1", args)
        self.assertIn("--simTag=sim_summary.txt", args)

    @patch("subprocess.run")
    def test_successful_simulation_run(self, mock_subproc):
        """Verify successful subprocess execution and artifact tracking."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Flow 1 Rx Packets: 6000\n"
        mock_proc.stderr = ""
        mock_subproc.return_value = mock_proc

        config = {"ue_count": 2, "bandwidth_mhz": 50.0}
        with patch.object(Path, "mkdir"), patch("builtins.open", unittest.mock.mock_open()):
            with patch.object(Path, "exists", return_value=True):
                res = self.runner.run_simulation(config, run_id="test_run_success")

        self.assertTrue(res.success)
        self.assertEqual(res.exit_code, 0)
        self.assertIn("Flow 1", res.stdout)
        self.assertIsNone(res.error_message)

    @patch("subprocess.run")
    def test_failed_process_exit_code(self, mock_subproc):
        """Verify that non-zero exit code is captured and marked as failure."""
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "Error: invalid configuration\n"
        mock_subproc.return_value = mock_proc

        config = {"ue_count": 2}
        with patch.object(Path, "mkdir"), patch("builtins.open", unittest.mock.mock_open()):
            with patch.object(Path, "exists", return_value=False):
                res = self.runner.run_simulation(config, run_id="test_run_fail")

        self.assertFalse(res.success)
        self.assertEqual(res.exit_code, 1)
        self.assertIsNotNone(res.error_message)
        self.assertIn("non-zero code 1", res.error_message)

    @patch("subprocess.run")
    def test_timeout_handling(self, mock_subproc):
        """Verify subprocess.TimeoutExpired is safely captured without crashing."""
        mock_subproc.side_effect = subprocess.TimeoutExpired(
            cmd=["/mock/ns3"], timeout=30.0, output="partial stdout", stderr="timeout"
        )

        config = {"ue_count": 2}
        with patch.object(Path, "mkdir"), patch("builtins.open", unittest.mock.mock_open()):
            res = self.runner.run_simulation(config, run_id="test_run_timeout")

        self.assertFalse(res.success)
        self.assertEqual(res.exit_code, -2)
        self.assertIn("timed out", res.error_message)


if __name__ == "__main__":
    unittest.main()

