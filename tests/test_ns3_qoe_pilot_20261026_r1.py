"""Unit tests for the isolated seven-run ns-3 QoE pilot helpers."""

import csv
import tempfile
import unittest
from pathlib import Path

from experiments.ns3_qoe_expansion_20261026_r1.pilot import (
    CSV_COLUMNS,
    _write_manifest,
    build_command,
    pilot_configurations,
    validate_run_output,
)
from experiments.ns3_qoe_expansion_20261026_r1.preprocessing import (
    derive_effective_band_powers_dbm,
    derive_flow_inputs,
    recompute_qoe_proxy,
)


class Ns3QoePilotTests(unittest.TestCase):
    def test_pilot_has_exact_seven_requested_runs_with_explicit_rng(self):
        configs = pilot_configurations()
        self.assertEqual(len(configs), 7)
        self.assertEqual(
            [(c["bandwidth_band1_mhz"], c["total_tx_power_dbm"], c["ue_count"], c["rng_run"]) for c in configs],
            [(50, 35, 2, 1), (50, 35, 2, 2), (20, 35, 2, 1),
             (100, 35, 2, 1), (50, 30, 2, 1), (50, 40, 2, 1), (50, 35, 10, 1)],
        )
        self.assertTrue(all(c["bandwidth_band2_mhz"] == 50 for c in configs))
        self.assertTrue(all(c["rng_seed"] == 20261026 for c in configs))
        self.assertEqual(len({c["scenario_id"] for c in configs}), 7)
        self.assertEqual(len({c["run_id"] for c in configs}), 7)

    def test_command_passes_bandwidth_power_ues_and_rng_globals(self):
        config = pilot_configurations()[3]
        command = build_command(Path("/ns3-binary"), config, Path("/tmp/run"))
        for option in (
            "--ueNumPergNb=2", "--bandwidthBand1=100000000", "--bandwidthBand2=50000000",
            "--totalTxPower=35", "--RngSeed=20261026", "--RngRun=1",
        ):
            self.assertIn(option, command)

    def test_configured_offered_load_does_not_use_tx_counter(self):
        lowlat = derive_flow_inputs(ue_count=2, flow_id=1, lambda_be_pps=5000)
        voice = derive_flow_inputs(ue_count=2, flow_id=2, lambda_be_pps=5000)
        self.assertEqual(lowlat["flow_type"], "low_latency")
        self.assertEqual(lowlat["offered_throughput_mbps_configured"], 8.0)
        self.assertEqual(voice["flow_type"], "voice_be")
        self.assertAlmostEqual(voice["offered_throughput_mbps_configured"], 50.08)

    def test_power_split_matches_simulator_formula_and_conserves_power(self):
        p1, p2 = derive_effective_band_powers_dbm(35, 50, 50)
        self.assertAlmostEqual(p1, 35 - 10 * __import__("math").log10(2))
        self.assertAlmostEqual(p2, p1)
        total_mw = 10 ** (p1 / 10) + 10 ** (p2 / 10)
        self.assertAlmostEqual(total_mw, 10 ** (35 / 10), places=8)

    def test_qoe_recomputation_handles_outage(self):
        outage = {"rx_packets": "0", "throughput_mbps": "0", "delay_ms": "0", "jitter_ms": "0", "packet_loss": "1"}
        self.assertEqual(recompute_qoe_proxy(outage), 0.0)
        ideal = {"rx_packets": "100", "throughput_mbps": "50", "delay_ms": "0", "jitter_ms": "0", "packet_loss": "0"}
        self.assertAlmostEqual(recompute_qoe_proxy(ideal), 1.0)

    def test_pilot_output_validation_checks_settings_and_outage_semantics(self):
        config = pilot_configurations()[0]
        rows = []
        for flow_id in (1, 2):
            rows.append({
                "scenario_id": str(config["scenario_id"]), "ue_count": "2",
                "lambda_be": "5000", "bandwidth_mhz": "50", "total_tx_power_dbm": "35",
                "scenario_size_m": "30", "flow_id": str(flow_id), "tx_packets": "100",
                "rx_packets": "0" if flow_id == 1 else "100", "lost_packets": "0",
                "throughput_mbps": "0" if flow_id == 1 else "50",
                "delay_ms": "0", "jitter_ms": "0", "packet_loss": "1" if flow_id == 1 else "0",
                "qoe_proxy": "0" if flow_id == 1 else "1",
            })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qoe_dataset.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            result = validate_run_output(path, config)
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["outage_rows"], 1)
        self.assertEqual(result["missing_values"], 0)

    def test_manifest_serializes_command_and_metrics_without_extra_columns(self):
        record = {
            "run_id": "pilot-test", "status": "PASS", "command": ["/ns3", "--RngRun=1"],
            "metrics": {"row_count": 2}, "duration_seconds": 1.25,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            _write_manifest(path, [record])
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "pilot-test")
        self.assertIn("--RngRun=1", rows[0]["command_json"])
        self.assertEqual(rows[0]["metrics_json"], '{"row_count":2}')


if __name__ == "__main__":
    unittest.main()
