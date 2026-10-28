from __future__ import annotations

import unittest

from experiments.ns3_qoe_expansion_20261026_r1.expansion import (
    EXPANSION_TABLE,
    expansion_configurations,
    _validate_design_records,
)
from experiments.ns3_qoe_expansion_20261026_r1.pilot import build_command
from experiments.ns3_qoe_expansion_20261026_r1.preprocessing import (
    derive_effective_band_powers_dbm,
    derive_flow_inputs,
)


class Ns3QoeExpansionTests(unittest.TestCase):
    def test_expansion_has_exact_reported_eleven_conditions(self):
        self.assertEqual(EXPANSION_TABLE, (
            ("L9-01", 20, 30, 2), ("L9-02", 20, 35, 6), ("L9-03", 20, 40, 10),
            ("L9-04", 50, 30, 6), ("L9-05", 50, 35, 10), ("L9-06", 50, 40, 2),
            ("L9-07", 100, 30, 10), ("L9-08", 100, 35, 2), ("L9-09", 100, 40, 6),
            ("ANCHOR-UE2", 50, 35, 2), ("ANCHOR-UE6", 50, 35, 6),
        ))
        runs = expansion_configurations()
        self.assertEqual(len(runs), 22)
        self.assertEqual({r["rng_run"] for r in runs}, {1, 2})
        self.assertEqual({r["bandwidth_band2_mhz"] for r in runs}, {50})
        self.assertEqual(len({r["scenario_id"] for r in runs}), 22)
        self.assertEqual(len({r["run_id"] for r in runs}), 22)

    def test_rng_pairings_group_by_config_and_record_unique_scenarios(self):
        configs = expansion_configurations()
        records = [{**c, "status": "PASS"} for c in configs]
        checked = _validate_design_records(records, configs)
        self.assertTrue(checked["all_runs_pass"])
        records[-1]["status"] = "FAIL"
        checked = _validate_design_records(records, configs)
        self.assertFalse(checked["paired_rng_runs_1_2_per_configuration"])
        self.assertFalse(checked["all_runs_pass"])

    def test_command_has_configured_traffic_and_explicit_rng(self):
        run = next(c for c in expansion_configurations() if c["config_id"] == "L9-01" and c["rng_run"] == 2)
        command = build_command("/tmp/ns3-binary", run, "/tmp/isolated-run")
        for expected in ("--bandwidthBand1=20000000", "--bandwidthBand2=50000000",
                         "--totalTxPower=30", "--ueNumPergNb=2", "--lambdaBe=20000",
                         "--lambdaUll=10000", "--packetSizeBe=1252", "--packetSizeUll=100",
                         "--RngSeed=20261026", "--RngRun=2"):
            self.assertIn(expected, command)

    def test_effective_power_matches_proportional_simulator_allocation(self):
        b1, b2 = derive_effective_band_powers_dbm(35, 20, 50)
        self.assertAlmostEqual(b1, 29.5593, places=3)
        self.assertAlmostEqual(b2, 33.5387, places=3)

    def test_offered_load_comes_from_configured_rate_and_payload(self):
        be = derive_flow_inputs(ue_count=2, flow_id=2, lambda_be_pps=20000)
        ull = derive_flow_inputs(ue_count=2, flow_id=1, lambda_be_pps=20000)
        self.assertAlmostEqual(be["offered_throughput_mbps_configured"], 200.32)
        self.assertAlmostEqual(ull["offered_throughput_mbps_configured"], 8.0)


if __name__ == "__main__":
    unittest.main()
