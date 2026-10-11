"""Unit tests for simulation/result_parser.py."""

import math
import tempfile
import unittest
from pathlib import Path

from simulation.result_parser import (
    calculate_packet_loss,
    calculate_qoe_proxy,
    parse_simulation_csv,
)


class TestResultParser(unittest.TestCase):

    def test_calculate_packet_loss_boundaries(self):
        """Verify packet loss calculation including zero denominator protection."""
        # Normal delivery
        self.assertAlmostEqual(calculate_packet_loss(1000, 950), 0.05)
        # Complete loss
        self.assertAlmostEqual(calculate_packet_loss(1000, 0), 1.0)
        # Zero tx packets (avoid division by zero)
        self.assertEqual(calculate_packet_loss(0, 0), 0.0)
        # Rx > Tx anomaly clamping
        self.assertEqual(calculate_packet_loss(100, 105), 0.0)

    def test_calculate_qoe_proxy_consistency(self):
        """Verify QoE proxy calculation strictly follows scratch/qoe-dataset.cc formula."""
        # 1. Complete outage: rx_packets = 0 -> QoE must be 0.0
        qoe_outage = calculate_qoe_proxy(
            throughput_mbps=0.0,
            delay_ms=0.0,
            jitter_ms=0.0,
            packet_loss=1.0,
            rx_packets=0,
        )
        self.assertEqual(qoe_outage, 0.0)

        # 2. Benchmark case: throughput=50, delay=0, jitter=0, loss=0 -> QoE = 1.0
        qoe_ideal = calculate_qoe_proxy(
            throughput_mbps=50.0,
            delay_ms=0.0,
            jitter_ms=0.0,
            packet_loss=0.0,
            rx_packets=1000,
        )
        self.assertAlmostEqual(qoe_ideal, 1.0, places=5)

        # 3. Known scenario test case
        # throughput=25 Mbps (25/50 * 0.4 = 0.20)
        # delay=50 ms (0.25 * exp(-1) = 0.09197)
        # jitter=20 ms (0.20 * exp(-1) = 0.07358)
        # loss=0.10 (0.15 * 0.90 = 0.135)
        # expected = 0.20 + 0.09197 + 0.07358 + 0.135 = 0.50055
        expected = (
            0.40 * (25.0 / 50.0)
            + 0.25 * math.exp(-1.0)
            + 0.20 * math.exp(-1.0)
            + 0.15 * (1.0 - 0.10)
        )
        qoe_calc = calculate_qoe_proxy(
            throughput_mbps=25.0,
            delay_ms=50.0,
            jitter_ms=20.0,
            packet_loss=0.10,
            rx_packets=500,
        )
        self.assertAlmostEqual(qoe_calc, expected, places=5)

    def test_parse_simulation_csv_valid_flows(self):
        """Verify parsing of valid simulation CSV with flow type separation."""
        csv_content = (
            "scenario_id,ue_count,lambda_be,bandwidth_mhz,total_tx_power_dbm,scenario_size_m,"
            "flow_id,tx_packets,rx_packets,lost_packets,throughput_mbps,delay_ms,jitter_ms,packet_loss,qoe_proxy\n"
            "1,2,5000,50,35,30,1,6000,5998,0,10.2366,0.2760,0.0300,0.0003,0.6801\n"
            "1,2,5000,50,35,30,2,3000,2997,0,51.1488,0.8460,0.0817,0.0010,0.9948\n"
        )
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
            tmp.write(csv_content)
            tmp_path = Path(tmp.name)

        try:
            # Test Voice flow parsing (flow_id=2)
            metrics_voice = parse_simulation_csv(tmp_path, target_flow_type="voice_be")
            self.assertTrue(metrics_voice.success)
            self.assertFalse(metrics_voice.is_outage)
            self.assertEqual(metrics_voice.target_flow_count, 1)
            self.assertAlmostEqual(metrics_voice.mean_throughput_mbps, 51.1488, places=3)
            self.assertAlmostEqual(metrics_voice.mean_qoe_proxy, 0.9948, places=3)

            # Test LowLat flow parsing (flow_id=1)
            metrics_lowlat = parse_simulation_csv(tmp_path, target_flow_type="low_latency")
            self.assertTrue(metrics_lowlat.success)
            self.assertEqual(metrics_lowlat.target_flow_count, 1)
            self.assertAlmostEqual(metrics_lowlat.mean_throughput_mbps, 10.2366, places=3)
        finally:
            tmp_path.unlink()

    def test_parse_simulation_csv_outage_condition(self):
        """Verify that zero packet delivery is parsed as a valid outage without crashing."""
        csv_content = (
            "scenario_id,ue_count,lambda_be,bandwidth_mhz,total_tx_power_dbm,scenario_size_m,"
            "flow_id,tx_packets,rx_packets,lost_packets,throughput_mbps,delay_ms,jitter_ms,packet_loss,qoe_proxy\n"
            "1,2,5000,20,35,30,1,6000,0,0,0.0,0.0,0.0,1.0,0.0\n"
            "1,2,5000,20,35,30,2,3000,0,0,0.0,0.0,0.0,1.0,0.0\n"
        )
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
            tmp.write(csv_content)
            tmp_path = Path(tmp.name)

        try:
            metrics = parse_simulation_csv(tmp_path, target_flow_type="voice_be")
            self.assertTrue(metrics.success)
            self.assertTrue(metrics.is_outage)
            self.assertEqual(metrics.mean_throughput_mbps, 0.0)
            self.assertEqual(metrics.mean_packet_loss, 1.0)
            self.assertEqual(metrics.mean_qoe_proxy, 0.0)
        finally:
            tmp_path.unlink()

    def test_missing_or_malformed_csv(self):
        """Verify that missing or malformed files return failure without raising exceptions."""
        # Non-existent path
        res_missing = parse_simulation_csv(Path("/non/existent/file.csv"))
        self.assertFalse(res_missing.success)
        self.assertIn("not found", res_missing.error_message)

        # Malformed path
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
            tmp.write("invalid,header\n1,2\n")
            tmp_path = Path(tmp.name)

        try:
            res_bad = parse_simulation_csv(tmp_path)
            self.assertFalse(res_bad.success)
            self.assertIn("Malformed row", res_bad.error_message)
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()

