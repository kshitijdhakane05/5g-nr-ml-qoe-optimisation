from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from simulation.result_parser import (
    UE_MEASUREMENT_COLUMNS, calculate_qoe_proxy, parse_ue_measurements,
)


class UeMeasurementParserTests(unittest.TestCase):
    def _write(self, rows: list[dict[str, object]]) -> Path:
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        path = folder / "ue_measurements.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=UE_MEASUREMENT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_valid_per_ue_records_validate_qoe_and_outage(self):
        good_qoe = calculate_qoe_proxy(3.2, 18.0, 2.0, 0.1, rx_packets=90)
        path = self._write([
            {"ue_id": 1, "traffic_class": "low_latency", "application_profile": "Low-latency UDP workload",
             "flow_id": 1, "tx_packets": 100, "rx_packets": 90, "throughput_mbps": 3.2,
             "delay_ms": 18, "jitter_ms": 2, "packet_loss_fraction": 0.1, "qoe_proxy": good_qoe,
             "serving_cell_id": 2},
            {"ue_id": 2, "traffic_class": "voice_qos", "application_profile": "Voice QoS UDP workload",
             "flow_id": 2, "tx_packets": 100, "rx_packets": 0, "throughput_mbps": 0,
             "delay_ms": 0, "jitter_ms": 0, "packet_loss_fraction": 1, "qoe_proxy": 0,
             "serving_cell_id": 1},
        ])
        rows = parse_ue_measurements(path, expected_ue_count=2, expected_gnb_count=2)
        self.assertEqual([row["ue_id"] for row in rows], [1, 2])
        self.assertFalse(rows[0]["is_outage"])
        self.assertTrue(rows[1]["is_outage"])
        self.assertEqual(rows[1]["qoe_proxy"], 0.0)
        self.assertEqual(rows[0]["serving_cell_id"], 2)

    def test_serving_cell_outside_configured_topology_is_rejected(self):
        good_qoe = calculate_qoe_proxy(1.0, 1.0, 1.0, 0.0, rx_packets=10)
        row = {"ue_id": 1, "traffic_class": "low_latency", "application_profile": "Low-latency UDP workload",
               "flow_id": 1, "tx_packets": 10, "rx_packets": 10, "throughput_mbps": 1,
               "delay_ms": 1, "jitter_ms": 1, "packet_loss_fraction": 0, "qoe_proxy": good_qoe,
               "serving_cell_id": 2}
        with self.assertRaisesRegex(ValueError, "exceeds configured gNB count"):
            parse_ue_measurements(self._write([row]), expected_gnb_count=1)

    def test_duplicate_ue_identity_is_rejected(self):
        row = {"ue_id": 1, "traffic_class": "low_latency", "application_profile": "Low-latency UDP workload",
               "flow_id": 1, "tx_packets": 0, "rx_packets": 0, "throughput_mbps": 0,
               "delay_ms": 0, "jitter_ms": 0, "packet_loss_fraction": 0, "qoe_proxy": 0,
               "serving_cell_id": 1}
        with self.assertRaisesRegex(ValueError, "Duplicate or invalid"):
            parse_ue_measurements(self._write([row, {**row, "flow_id": 2}]), expected_ue_count=2)

    def test_row_count_and_qoe_mismatch_are_rejected(self):
        row = {"ue_id": 1, "traffic_class": "low_latency", "application_profile": "Low-latency UDP workload",
               "flow_id": 1, "tx_packets": 100, "rx_packets": 100, "throughput_mbps": 1,
               "delay_ms": 1, "jitter_ms": 1, "packet_loss_fraction": 0, "qoe_proxy": 0.1,
               "serving_cell_id": 1}
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            parse_ue_measurements(self._write([row]), expected_ue_count=2)
        with self.assertRaisesRegex(ValueError, "QoE recomputation mismatch"):
            parse_ue_measurements(self._write([row]))


if __name__ == "__main__":
    unittest.main()
